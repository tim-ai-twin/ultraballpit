//! GPU buffer management for SPH particle data.
//!
//! Creates and manages wgpu storage buffers for particle arrays, boundary
//! particles, and neighbor grid data. Handles CPU->GPU upload and GPU->CPU
//! readback.

use wgpu;
use wgpu::util::DeviceExt;

use crate::boundary::BoundaryParticles;
use crate::particle::ParticleArrays;

/// Simulation parameters uniform buffer layout.
/// Must match the SimParams struct in all WGSL shaders exactly.
#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuSimParams {
    pub dt: f32,
    pub h: f32,
    pub speed_of_sound: f32,
    pub gravity_x: f32,
    pub gravity_y: f32,
    pub gravity_z: f32,
    pub domain_min_x: f32,
    pub domain_min_y: f32,
    pub domain_min_z: f32,
    pub domain_max_x: f32,
    pub domain_max_y: f32,
    pub domain_max_z: f32,
    pub n_particles: u32,
    pub n_boundary: u32,
    pub grid_dim_x: u32,
    pub grid_dim_y: u32,
    pub grid_dim_z: u32,
    pub cell_size: f32,
    pub viscosity_alpha: f32,
    pub viscosity_beta: f32,
    pub pass_index: u32,
    /// Grid cells searched on each side of a particle's cell.
    pub search_cells: u32,
}

/// All GPU buffers needed for the SPH simulation.
pub struct GpuBuffers {
    // Uniform buffer
    pub params_buffer: wgpu::Buffer,

    // Fluid particle buffers
    pub pos_x: wgpu::Buffer,
    pub pos_y: wgpu::Buffer,
    pub pos_z: wgpu::Buffer,
    pub vel_x: wgpu::Buffer,
    pub vel_y: wgpu::Buffer,
    pub vel_z: wgpu::Buffer,
    pub acc_x: wgpu::Buffer,
    pub acc_y: wgpu::Buffer,
    pub acc_z: wgpu::Buffer,
    pub density: wgpu::Buffer,
    pub pressure: wgpu::Buffer,
    pub mass: wgpu::Buffer,
    pub fluid_type: wgpu::Buffer,

    // Boundary particle buffers
    pub bnd_x: wgpu::Buffer,
    pub bnd_y: wgpu::Buffer,
    pub bnd_z: wgpu::Buffer,
    pub bnd_mass: wgpu::Buffer,
    pub bnd_pressure: wgpu::Buffer,

    // Neighbor grid buffers (fluid particles, rebuilt every step)
    pub cell_indices: wgpu::Buffer,
    pub cell_counts: wgpu::Buffer,
    pub cell_offsets: wgpu::Buffer,
    pub sorted_indices: wgpu::Buffer,
    pub write_heads: wgpu::Buffer,

    // Boundary particle grid (built once at init, boundary particles are static)
    pub bnd_cell_counts: wgpu::Buffer,
    pub bnd_cell_offsets: wgpu::Buffer,
    pub bnd_sorted_indices: wgpu::Buffer,

    pub staging_mass: wgpu::Buffer,

    // Staging buffers for readback
    pub staging_density: wgpu::Buffer,
    pub staging_pos_x: wgpu::Buffer,
    pub staging_pos_y: wgpu::Buffer,
    pub staging_pos_z: wgpu::Buffer,
    pub staging_vel_x: wgpu::Buffer,
    pub staging_vel_y: wgpu::Buffer,
    pub staging_vel_z: wgpu::Buffer,
    pub staging_pressure: wgpu::Buffer,
    pub staging_fluid_type: wgpu::Buffer,
    pub staging_acc_x: wgpu::Buffer,
    pub staging_acc_y: wgpu::Buffer,
    pub staging_acc_z: wgpu::Buffer,

    /// Gather sources for sorting particle data into cell order.
    pub sort_tmp: Vec<wgpu::Buffer>,

    // PCISPH state buffers (always allocated; unused for WCSPH).
    // Packed per-particle step state, see pcisph_predict.wgsl / pcisph_solve.wgsl:
    /// Step-start (x, y, z, mass).
    pub pcisph_orig4: wgpu::Buffer,
    /// Step-start velocity (x, y, z, 0).
    pub pcisph_vel4: wgpu::Buffer,
    /// Non-pressure acceleration (x, y, z, 0).
    pub pcisph_np4: wgpu::Buffer,
    /// Pressure acceleration of the latest correction iteration (x, y, z, 0).
    pub pcisph_pacc4: wgpu::Buffer,
    /// Current predicted position + mass (x, y, z, m): one 16-byte gather per
    /// neighbor candidate in the correction loop.
    pub pcisph_pos4: wgpu::Buffer,
    /// pressure / density² per particle, written with the pressure correction.
    pub pcisph_p_rho2: wgpu::Buffer,
    /// Per-step neighbor-list counts: fluid particles, then boundary particles.
    pub pcisph_counts: wgpu::Buffer,
    /// Per-step neighbor lists (layout in pcisph_solve.wgsl).
    pub pcisph_lists: wgpu::Buffer,
    /// Per-particle PCISPH pressure scaling factor (dt-independent base).
    pub pcisph_delta: wgpu::Buffer,
    /// Convergence state: [0]=sum of density errors (fixed-point),
    /// [1]=count of over-compressed, [2]=correction iterations run this step.
    pub pcisph_convergence: wgpu::Buffer,
    /// Staging buffer for convergence readback (4 × u32)
    pub staging_convergence: wgpu::Buffer,
    /// Indirect dispatch args for the PCISPH correction loop, 3 × (x, y, z):
    /// [particles, boundary particles, pcisph_solve neighbor sums]. Reset to full
    /// counts every step and zeroed on-device once the solve converges, so
    /// the remaining iterations dispatch nothing.
    pub pcisph_args: wgpu::Buffer,

    /// Number of fluid particles
    pub n_particles: u32,
    /// Number of boundary particles
    pub n_boundary: u32,
    /// Total number of grid cells
    pub total_cells: u32,
}

/// Minimum buffer size (wgpu requires non-zero buffers).
const MIN_BUF_SIZE: u64 = 4;

/// PCISPH neighbor-list capacities (entries within 2.5h; ~145 fluid
/// neighbors at rest density): fluid and boundary neighbors of a fluid
/// particle, fluid neighbors of a boundary particle. Overflowing particles
/// fall back to the grid.
pub const PCISPH_LIST_CAP_FLUID: u32 = 192;
pub const PCISPH_LIST_CAP_BOUNDARY: u32 = 128;
pub const PCISPH_LIST_CAP_BND_FLUID: u32 = 128;

/// Size of `GpuBuffers::pcisph_args`: three (x, y, z) indirect dispatches.
pub const PCISPH_ARGS_BYTES: u64 = 3 * 3 * 4;

/// Number of per-particle arrays permuted into cell order after each grid
/// build (see `GpuBuffers::sorted_arrays`).
pub const SORTED_ARRAY_COUNT: usize = 10;

/// Create a storage buffer from f32 slice data. If the slice is empty, creates
/// a minimal buffer.
fn create_storage_buf(device: &wgpu::Device, label: &str, data: &[f32]) -> wgpu::Buffer {
    if data.is_empty() {
        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: MIN_BUF_SIZE,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    } else {
        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(label),
            contents: bytemuck::cast_slice(data),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
        })
    }
}

/// Create a storage buffer from u32 slice data.
fn create_storage_buf_u32(device: &wgpu::Device, label: &str, data: &[u32]) -> wgpu::Buffer {
    if data.is_empty() {
        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: MIN_BUF_SIZE,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    } else {
        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(label),
            contents: bytemuck::cast_slice(data),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
        })
    }
}

/// Create a staging (MAP_READ) buffer for readback.
fn create_staging_buf(device: &wgpu::Device, label: &str, size: u64) -> wgpu::Buffer {
    let size = size.max(MIN_BUF_SIZE);
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}

impl GpuBuffers {
    /// Create all GPU buffers from initial particle and boundary data.
    ///
    /// The PCISPH step-state and neighbor-list buffers (~1.3 KB per particle)
    /// are only sized for the particles when `pcisph` is set.
    pub fn new(
        device: &wgpu::Device,
        particles: &ParticleArrays,
        boundary: &BoundaryParticles,
        grid_dims: [u32; 3],
        params: &GpuSimParams,
        pcisph: bool,
    ) -> Self {
        let n = particles.len();
        let n_bnd = boundary.len();
        let total_cells = (grid_dims[0] as usize) * (grid_dims[1] as usize) * (grid_dims[2] as usize);

        // Params uniform buffer
        let params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("sim_params"),
            contents: bytemuck::bytes_of(params),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        // Fluid particle buffers
        let pos_x = create_storage_buf(device, "pos_x", &particles.x);
        let pos_y = create_storage_buf(device, "pos_y", &particles.y);
        let pos_z = create_storage_buf(device, "pos_z", &particles.z);
        let vel_x = create_storage_buf(device, "vel_x", &particles.vx);
        let vel_y = create_storage_buf(device, "vel_y", &particles.vy);
        let vel_z = create_storage_buf(device, "vel_z", &particles.vz);
        let acc_x = create_storage_buf(device, "acc_x", &particles.ax);
        let acc_y = create_storage_buf(device, "acc_y", &particles.ay);
        let acc_z = create_storage_buf(device, "acc_z", &particles.az);
        let density = create_storage_buf(device, "density", &particles.density);
        let pressure = create_storage_buf(device, "pressure", &particles.pressure);
        let mass = create_storage_buf(device, "mass", &particles.mass);

        // Convert FluidType to u32
        let ft_u32: Vec<u32> = particles.fluid_type.iter().map(|ft| *ft as u32).collect();
        let fluid_type = create_storage_buf_u32(device, "fluid_type", &ft_u32);

        // Neighbor grid buffers (fluid particles)
        let zeros_n = vec![0u32; n.max(1)];
        let zeros_cells = vec![0u32; total_cells.max(1)];

        let cell_indices = create_storage_buf_u32(device, "cell_indices", &zeros_n);
        let cell_counts = create_storage_buf_u32(device, "cell_counts", &zeros_cells);
        let cell_offsets = create_storage_buf_u32(device, "cell_offsets", &zeros_cells);
        let sorted_indices = create_storage_buf_u32(device, "sorted_indices", &zeros_n);
        let write_heads = create_storage_buf_u32(device, "write_heads", &zeros_cells);

        // Boundary particle grid (built once, boundary particles are static).
        // Boundary arrays are stored in cell order so each grid row is a
        // contiguous index range; bnd_sorted_indices is then the identity.
        let (bnd_cell_counts_data, bnd_cell_offsets_data, bnd_order) =
            build_boundary_grid(boundary, params, grid_dims, total_cells);
        let permute = |v: &[f32]| -> Vec<f32> { bnd_order.iter().map(|&i| v[i as usize]).collect() };
        let bnd_x = create_storage_buf(device, "bnd_x", &permute(&boundary.x));
        let bnd_y = create_storage_buf(device, "bnd_y", &permute(&boundary.y));
        let bnd_z = create_storage_buf(device, "bnd_z", &permute(&boundary.z));
        let bnd_mass = create_storage_buf(device, "bnd_mass", &permute(&boundary.mass));
        let bnd_pressure = create_storage_buf(device, "bnd_pressure", &permute(&boundary.pressure));
        let bnd_identity: Vec<u32> = (0..n_bnd.max(1) as u32).collect();
        let bnd_cell_counts = create_storage_buf_u32(device, "bnd_cell_counts", &bnd_cell_counts_data);
        let bnd_cell_offsets = create_storage_buf_u32(device, "bnd_cell_offsets", &bnd_cell_offsets_data);
        let bnd_sorted_indices = create_storage_buf_u32(device, "bnd_sorted_indices", &bnd_identity);

        // Staging buffers for readback
        let f32_size = std::mem::size_of::<f32>() as u64;
        let u32_size = std::mem::size_of::<u32>() as u64;
        let particle_bytes = (n as u64) * f32_size;
        let particle_u32_bytes = (n as u64) * u32_size;

        let staging_density = create_staging_buf(device, "staging_density", particle_bytes);
        let staging_pos_x = create_staging_buf(device, "staging_pos_x", particle_bytes);
        let staging_pos_y = create_staging_buf(device, "staging_pos_y", particle_bytes);
        let staging_pos_z = create_staging_buf(device, "staging_pos_z", particle_bytes);
        let staging_vel_x = create_staging_buf(device, "staging_vel_x", particle_bytes);
        let staging_vel_y = create_staging_buf(device, "staging_vel_y", particle_bytes);
        let staging_vel_z = create_staging_buf(device, "staging_vel_z", particle_bytes);
        let staging_pressure = create_staging_buf(device, "staging_pressure", particle_bytes);
        let staging_fluid_type = create_staging_buf(device, "staging_fluid_type", particle_u32_bytes);
        let staging_acc_x = create_staging_buf(device, "staging_acc_x", particle_bytes);
        let staging_acc_y = create_staging_buf(device, "staging_acc_y", particle_bytes);
        let staging_acc_z = create_staging_buf(device, "staging_acc_z", particle_bytes);

        // Temp buffer for particle reordering (one array at a time)
        let staging_mass = create_staging_buf(device, "staging_mass", particle_bytes);
        // Scratch copies used as gather sources when sorting particle data
        // into cell order (one per array in SORTED_ARRAYS order).
        let sort_tmp: Vec<wgpu::Buffer> = (0..SORTED_ARRAY_COUNT)
            .map(|_| device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("sort_tmp"),
                size: particle_bytes.max(MIN_BUF_SIZE),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }))
            .collect();

        // PCISPH state buffers (allocated for all solver types; small overhead)
        let zeros_f32_n = vec![0.0f32; n.max(1)];
        let pcisph_delta = create_storage_buf(device, "pcisph_delta", &zeros_f32_n);
        let pcisph_convergence = create_storage_buf_u32(device, "pcisph_convergence", &[0u32; 4]);
        let staging_convergence = create_staging_buf(device, "staging_convergence", 16); // 4 × u32
        let pcisph_args = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("pcisph_args"),
            size: PCISPH_ARGS_BYTES,
            usage: wgpu::BufferUsages::INDIRECT
                | wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        // Placeholders (one vec4) when the solver is not PCISPH.
        let pcisph_buf = |label: &str, words: usize| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: if pcisph { (words * 4) as u64 } else { 0 }.max(16),
                usage: wgpu::BufferUsages::STORAGE,
                mapped_at_creation: false,
            })
        };
        let n1 = n.max(1);
        let pcisph_orig4 = pcisph_buf("pcisph_orig4", 4 * n1);
        let pcisph_vel4 = pcisph_buf("pcisph_vel4", 4 * n1);
        let pcisph_np4 = pcisph_buf("pcisph_np4", 4 * n1);
        let pcisph_pacc4 = pcisph_buf("pcisph_pacc4", 4 * n1);
        let pcisph_pos4 = pcisph_buf("pcisph_pos4", 4 * n1);
        let pcisph_p_rho2 = pcisph_buf("pcisph_p_rho2", n1);
        let pcisph_counts = pcisph_buf("pcisph_counts", n + n_bnd + 1);
        let pcisph_lists = pcisph_buf(
            "pcisph_lists",
            (PCISPH_LIST_CAP_FLUID + PCISPH_LIST_CAP_BOUNDARY) as usize * n
                + PCISPH_LIST_CAP_BND_FLUID as usize * n_bnd
                + 1,
        );

        Self {
            params_buffer,
            pos_x,
            pos_y,
            pos_z,
            vel_x,
            vel_y,
            vel_z,
            acc_x,
            acc_y,
            acc_z,
            density,
            pressure,
            mass,
            fluid_type,
            bnd_x,
            bnd_y,
            bnd_z,
            bnd_mass,
            bnd_pressure,
            cell_indices,
            cell_counts,
            cell_offsets,
            sorted_indices,
            write_heads,
            bnd_cell_counts,
            bnd_cell_offsets,
            bnd_sorted_indices,
            sort_tmp,
            staging_mass,
            pcisph_orig4,
            pcisph_vel4,
            pcisph_np4,
            pcisph_pacc4,
            pcisph_pos4,
            pcisph_p_rho2,
            pcisph_counts,
            pcisph_lists,
            pcisph_delta,
            pcisph_convergence,
            staging_convergence,
            pcisph_args,
            staging_density,
            staging_pos_x,
            staging_pos_y,
            staging_pos_z,
            staging_vel_x,
            staging_vel_y,
            staging_vel_z,
            staging_pressure,
            staging_fluid_type,
            staging_acc_x,
            staging_acc_y,
            staging_acc_z,
            n_particles: n as u32,
            n_boundary: n_bnd as u32,
            total_cells: total_cells as u32,
        }
    }

    /// Update the uniform params buffer.
    pub fn update_params(&self, queue: &wgpu::Queue, params: &GpuSimParams) {
        queue.write_buffer(&self.params_buffer, 0, bytemuck::bytes_of(params));
    }

    /// Read back all particle data from GPU to CPU.
    ///
    /// Maps all staging buffers in parallel with a single `poll(Wait)`,
    /// then reads all mapped ranges. This eliminates 11 sequential
    /// map+poll+unmap cycles.
    pub fn readback_particles(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) -> ParticleArrays {
        let n = self.n_particles as usize;
        if n == 0 {
            return ParticleArrays::new();
        }

        let byte_len = (n * std::mem::size_of::<f32>()) as u64;
        let u32_byte_len = (n * std::mem::size_of::<u32>()) as u64;

        // Encode copy commands
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("readback"),
        });

        encoder.copy_buffer_to_buffer(&self.mass, 0, &self.staging_mass, 0, byte_len);
        encoder.copy_buffer_to_buffer(&self.pos_x, 0, &self.staging_pos_x, 0, byte_len);
        encoder.copy_buffer_to_buffer(&self.pos_y, 0, &self.staging_pos_y, 0, byte_len);
        encoder.copy_buffer_to_buffer(&self.pos_z, 0, &self.staging_pos_z, 0, byte_len);
        encoder.copy_buffer_to_buffer(&self.vel_x, 0, &self.staging_vel_x, 0, byte_len);
        encoder.copy_buffer_to_buffer(&self.vel_y, 0, &self.staging_vel_y, 0, byte_len);
        encoder.copy_buffer_to_buffer(&self.vel_z, 0, &self.staging_vel_z, 0, byte_len);
        encoder.copy_buffer_to_buffer(&self.density, 0, &self.staging_density, 0, byte_len);
        encoder.copy_buffer_to_buffer(&self.pressure, 0, &self.staging_pressure, 0, byte_len);
        encoder.copy_buffer_to_buffer(&self.fluid_type, 0, &self.staging_fluid_type, 0, u32_byte_len);
        encoder.copy_buffer_to_buffer(&self.acc_x, 0, &self.staging_acc_x, 0, byte_len);
        encoder.copy_buffer_to_buffer(&self.acc_y, 0, &self.staging_acc_y, 0, byte_len);
        encoder.copy_buffer_to_buffer(&self.acc_z, 0, &self.staging_acc_z, 0, byte_len);

        queue.submit(std::iter::once(encoder.finish()));

        // Issue all map_async calls before polling — this lets the driver
        // process all mappings in a single poll(Wait) round-trip.
        let staging_bufs: &[&wgpu::Buffer] = &[
            &self.staging_pos_x, &self.staging_pos_y, &self.staging_pos_z,
            &self.staging_vel_x, &self.staging_vel_y, &self.staging_vel_z,
            &self.staging_acc_x, &self.staging_acc_y, &self.staging_acc_z,
            &self.staging_density, &self.staging_pressure, &self.staging_fluid_type,
            &self.staging_mass,
        ];

        let senders: Vec<_> = staging_bufs.iter().map(|buf| {
            let slice = buf.slice(..);
            let (tx, rx) = std::sync::mpsc::channel();
            slice.map_async(wgpu::MapMode::Read, move |result| {
                let _ = tx.send(result);
            });
            rx
        }).collect();

        // Single poll to complete all mappings
        device.poll(wgpu::Maintain::Wait);

        // Verify all mappings succeeded
        for rx in &senders {
            rx.recv().unwrap().unwrap();
        }

        // Read all mapped ranges
        let read_f32 = |buf: &wgpu::Buffer| -> Vec<f32> {
            let data = buf.slice(..).get_mapped_range();
            let result: Vec<f32> = bytemuck::cast_slice(&data)[..n].to_vec();
            drop(data);
            buf.unmap();
            result
        };
        let read_u32 = |buf: &wgpu::Buffer| -> Vec<u32> {
            let data = buf.slice(..).get_mapped_range();
            let result: Vec<u32> = bytemuck::cast_slice(&data)[..n].to_vec();
            drop(data);
            buf.unmap();
            result
        };

        let x = read_f32(&self.staging_pos_x);
        let y = read_f32(&self.staging_pos_y);
        let z = read_f32(&self.staging_pos_z);
        let vx = read_f32(&self.staging_vel_x);
        let vy = read_f32(&self.staging_vel_y);
        let vz = read_f32(&self.staging_vel_z);
        let ax = read_f32(&self.staging_acc_x);
        let ay = read_f32(&self.staging_acc_y);
        let az = read_f32(&self.staging_acc_z);
        let density_vec = read_f32(&self.staging_density);
        let pressure_vec = read_f32(&self.staging_pressure);
        let ft_u32 = read_u32(&self.staging_fluid_type);
        let mass = read_f32(&self.staging_mass);

        let fluid_type: Vec<crate::particle::FluidType> = ft_u32
            .iter()
            .map(|&v| {
                if v == 0 {
                    crate::particle::FluidType::Water
                } else {
                    crate::particle::FluidType::Air
                }
            })
            .collect();

        ParticleArrays {
            x,
            y,
            z,
            vx,
            vy,
            vz,
            ax,
            ay,
            az,
            density: density_vec,
            pressure: pressure_vec,
            mass,
            temperature: vec![293.15; n],
            fluid_type,
        }
    }

    /// Per-particle arrays that persist across steps and must be permuted
    /// together when particles are sorted into cell order. Arrays recomputed
    /// every step before use (acc, pressure, PCISPH scratch) are omitted.
    pub fn sorted_arrays(&self) -> [&wgpu::Buffer; SORTED_ARRAY_COUNT] {
        [
            &self.pos_x, &self.pos_y, &self.pos_z,
            &self.vel_x, &self.vel_y, &self.vel_z,
            &self.density, &self.mass, &self.fluid_type, &self.pcisph_delta,
        ]
    }

    /// Read back the PCISPH convergence state (see `pcisph_convergence`):
    /// (sum_density_error_fixed_point, count_over_compressed, iterations).
    pub fn readback_convergence(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) -> [u32; 3] {
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("readback_convergence"),
        });
        encoder.copy_buffer_to_buffer(&self.pcisph_convergence, 0, &self.staging_convergence, 0, 16);
        queue.submit(std::iter::once(encoder.finish()));

        let slice = self.staging_convergence.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = tx.send(result);
        });
        device.poll(wgpu::Maintain::Wait);
        rx.recv().unwrap().unwrap();

        let data = slice.get_mapped_range();
        let vals: &[u32] = bytemuck::cast_slice(&data);
        let result = [vals[0], vals[1], vals[2]];
        drop(data);
        self.staging_convergence.unmap();
        result
    }

    /// Read back only the density buffer from GPU to CPU.
    pub fn readback_density(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) -> Vec<f32> {
        let n = self.n_particles as usize;
        if n == 0 {
            return Vec::new();
        }

        let byte_len = (n * std::mem::size_of::<f32>()) as u64;
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("readback_density"),
        });
        encoder.copy_buffer_to_buffer(&self.density, 0, &self.staging_density, 0, byte_len);
        queue.submit(std::iter::once(encoder.finish()));

        read_f32_buffer(device, &self.staging_density, n)
    }
}

/// Block on mapping a staging buffer and read f32 data.
fn read_f32_buffer(device: &wgpu::Device, buffer: &wgpu::Buffer, count: usize) -> Vec<f32> {
    let slice = buffer.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        tx.send(result).unwrap();
    });
    device.poll(wgpu::Maintain::Wait);
    rx.recv().unwrap().unwrap();

    let data = slice.get_mapped_range();
    let result: Vec<f32> = bytemuck::cast_slice(&data)[..count].to_vec();
    drop(data);
    buffer.unmap();
    result
}

/// Build a spatial hash grid for boundary particles on the CPU.
/// Returns (cell_counts, cell_offsets, sorted_indices).
fn build_boundary_grid(
    boundary: &BoundaryParticles,
    params: &GpuSimParams,
    grid_dims: [u32; 3],
    total_cells: usize,
) -> (Vec<u32>, Vec<u32>, Vec<u32>) {
    let n_bnd = boundary.len();
    if n_bnd == 0 {
        return (vec![0u32; total_cells.max(1)], vec![0u32; total_cells.max(1)], Vec::new());
    }

    let cell_size = params.cell_size;
    let dmin = [params.domain_min_x, params.domain_min_y, params.domain_min_z];
    let gdims = [grid_dims[0] as usize, grid_dims[1] as usize, grid_dims[2] as usize];

    // Hash each boundary particle to its cell
    let mut cell_for_particle = vec![0usize; n_bnd];
    let mut counts = vec![0u32; total_cells];

    for i in 0..n_bnd {
        let cx = ((boundary.x[i] - dmin[0]) / cell_size).floor().max(0.0).min((gdims[0] - 1) as f32) as usize;
        let cy = ((boundary.y[i] - dmin[1]) / cell_size).floor().max(0.0).min((gdims[1] - 1) as f32) as usize;
        let cz = ((boundary.z[i] - dmin[2]) / cell_size).floor().max(0.0).min((gdims[2] - 1) as f32) as usize;
        let cell = cx + cy * gdims[0] + cz * gdims[0] * gdims[1];
        cell_for_particle[i] = cell;
        counts[cell] += 1;
    }

    // Exclusive prefix sum
    let mut offsets = vec![0u32; total_cells];
    let mut running = 0u32;
    for c in 0..total_cells {
        offsets[c] = running;
        running += counts[c];
    }

    // Scatter boundary particle indices into sorted order
    let mut write_heads = offsets.clone();
    let mut sorted = vec![0u32; n_bnd];
    for i in 0..n_bnd {
        let cell = cell_for_particle[i];
        let pos = write_heads[cell] as usize;
        sorted[pos] = i as u32;
        write_heads[cell] += 1;
    }

    (counts, offsets, sorted)
}

