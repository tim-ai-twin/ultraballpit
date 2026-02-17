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
    pub _pad1: u32,
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

    // Original f32 mass data (mass never changes during simulation, so we
    // store it CPU-side and skip GPU readback).
    pub mass_f32: Vec<f32>,

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

    /// Number of fluid particles
    pub n_particles: u32,
    /// Number of boundary particles
    pub n_boundary: u32,
    /// Total number of grid cells
    pub total_cells: u32,
}

/// Minimum buffer size (wgpu requires non-zero buffers).
const MIN_BUF_SIZE: u64 = 4;

/// Convert f32 to IEEE 754 half-precision (f16) bits.
fn f32_to_f16_bits(x: f32) -> u16 {
    let bits = x.to_bits();
    let sign = (bits >> 31) & 1;
    let exp = ((bits >> 23) & 0xFF) as i32;
    let frac = bits & 0x7F_FFFF;

    if exp == 0xFF {
        // Inf/NaN
        return ((sign << 15) | 0x7C00 | if frac != 0 { 1 } else { 0 }) as u16;
    }

    let new_exp = exp - 127 + 15;
    if new_exp >= 31 {
        return ((sign << 15) | 0x7C00) as u16; // overflow → Inf
    }
    if new_exp <= 0 {
        if new_exp < -10 {
            return (sign << 15) as u16; // underflow → zero
        }
        let frac_with_implicit = frac | 0x80_0000;
        let shift = 1 - new_exp;
        return ((sign << 15) | (frac_with_implicit >> (13 + shift))) as u16;
    }

    ((sign << 15) | ((new_exp as u32) << 10) | (frac >> 13)) as u16
}

/// Pack a slice of f32 mass values into u32 pairs of f16.
/// Each u32 holds two f16 values: low bits = even index, high bits = odd index.
fn pack_mass_f16(masses: &[f32]) -> Vec<u32> {
    let n_packed = (masses.len() + 1) / 2;
    let mut packed = Vec::with_capacity(n_packed);
    for i in (0..masses.len()).step_by(2) {
        let lo = f32_to_f16_bits(masses[i]) as u32;
        let hi = if i + 1 < masses.len() {
            f32_to_f16_bits(masses[i + 1]) as u32
        } else {
            0u32
        };
        packed.push(lo | (hi << 16));
    }
    packed
}

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
    pub fn new(
        device: &wgpu::Device,
        particles: &ParticleArrays,
        boundary: &BoundaryParticles,
        grid_dims: [u32; 3],
        params: &GpuSimParams,
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
        // Pack mass as f16 pairs in u32s to halve memory bandwidth.
        // Mass is read-only during simulation, so f16 precision (~3 decimal
        // digits) is more than sufficient.
        let mass_packed = pack_mass_f16(&particles.mass);
        let mass = create_storage_buf_u32(device, "mass_packed", &mass_packed);

        // Convert FluidType to u32
        let ft_u32: Vec<u32> = particles.fluid_type.iter().map(|ft| *ft as u32).collect();
        let fluid_type = create_storage_buf_u32(device, "fluid_type", &ft_u32);

        // Boundary particle buffers
        let bnd_x = create_storage_buf(device, "bnd_x", &boundary.x);
        let bnd_y = create_storage_buf(device, "bnd_y", &boundary.y);
        let bnd_z = create_storage_buf(device, "bnd_z", &boundary.z);
        let bnd_mass = create_storage_buf(device, "bnd_mass", &boundary.mass);
        let bnd_pressure = create_storage_buf(device, "bnd_pressure", &boundary.pressure);

        // Neighbor grid buffers (fluid particles)
        let zeros_n = vec![0u32; n.max(1)];
        let zeros_cells = vec![0u32; total_cells.max(1)];

        let cell_indices = create_storage_buf_u32(device, "cell_indices", &zeros_n);
        let cell_counts = create_storage_buf_u32(device, "cell_counts", &zeros_cells);
        let cell_offsets = create_storage_buf_u32(device, "cell_offsets", &zeros_cells);
        let sorted_indices = create_storage_buf_u32(device, "sorted_indices", &zeros_n);
        let write_heads = create_storage_buf_u32(device, "write_heads", &zeros_cells);

        // Boundary particle grid (built once, boundary particles are static)
        let cell_size = params.cell_size;
        let (bnd_cell_counts_data, bnd_cell_offsets_data, bnd_sorted_indices_data) =
            build_boundary_grid(boundary, params, grid_dims, total_cells);
        let bnd_cell_counts = create_storage_buf_u32(device, "bnd_cell_counts", &bnd_cell_counts_data);
        let bnd_cell_offsets = create_storage_buf_u32(device, "bnd_cell_offsets", &bnd_cell_offsets_data);
        let bnd_sorted_indices = create_storage_buf_u32(device, "bnd_sorted_indices",
            if bnd_sorted_indices_data.is_empty() { &[0u32] } else { &bnd_sorted_indices_data });
        let _ = cell_size; // used above via params

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
            mass_f32: particles.mass.clone(),
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

        // Note: mass is not copied — it's stored as f16-packed on GPU and
        // never changes during simulation, so we use self.mass_f32 instead.
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

        // Map and read
        let x = read_f32_buffer(device, &self.staging_pos_x, n);
        let y = read_f32_buffer(device, &self.staging_pos_y, n);
        let z = read_f32_buffer(device, &self.staging_pos_z, n);
        let vx = read_f32_buffer(device, &self.staging_vel_x, n);
        let vy = read_f32_buffer(device, &self.staging_vel_y, n);
        let vz = read_f32_buffer(device, &self.staging_vel_z, n);
        let ax = read_f32_buffer(device, &self.staging_acc_x, n);
        let ay = read_f32_buffer(device, &self.staging_acc_y, n);
        let az = read_f32_buffer(device, &self.staging_acc_z, n);
        let density_vec = read_f32_buffer(device, &self.staging_density, n);
        let pressure_vec = read_f32_buffer(device, &self.staging_pressure, n);
        let ft_u32 = read_u32_buffer(device, &self.staging_fluid_type, n);

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
            mass: self.mass_f32.clone(),
            temperature: vec![293.15; n],
            fluid_type,
        }
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

/// Block on mapping a staging buffer and read u32 data.
fn read_u32_buffer(device: &wgpu::Device, buffer: &wgpu::Buffer, count: usize) -> Vec<u32> {
    let slice = buffer.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        tx.send(result).unwrap();
    });
    device.poll(wgpu::Maintain::Wait);
    rx.recv().unwrap().unwrap();

    let data = slice.get_mapped_range();
    let result: Vec<u32> = bytemuck::cast_slice(&data)[..count].to_vec();
    drop(data);
    buffer.unmap();
    result
}
