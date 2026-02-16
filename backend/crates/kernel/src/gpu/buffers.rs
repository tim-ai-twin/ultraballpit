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

    // Neighbor grid buffers
    pub cell_indices: wgpu::Buffer,
    pub cell_counts: wgpu::Buffer,
    pub cell_offsets: wgpu::Buffer,
    pub sorted_indices: wgpu::Buffer,
    pub write_heads: wgpu::Buffer,

    // Staging buffers for readback
    pub staging_density: wgpu::Buffer,
    pub staging_pos_x: wgpu::Buffer,
    pub staging_pos_y: wgpu::Buffer,
    pub staging_pos_z: wgpu::Buffer,
    pub staging_vel_x: wgpu::Buffer,
    pub staging_vel_y: wgpu::Buffer,
    pub staging_vel_z: wgpu::Buffer,
    pub staging_pressure: wgpu::Buffer,
    pub staging_mass: wgpu::Buffer,
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
        let mass = create_storage_buf(device, "mass", &particles.mass);

        // Convert FluidType to u32
        let ft_u32: Vec<u32> = particles.fluid_type.iter().map(|ft| *ft as u32).collect();
        let fluid_type = create_storage_buf_u32(device, "fluid_type", &ft_u32);

        // Boundary particle buffers
        let bnd_x = create_storage_buf(device, "bnd_x", &boundary.x);
        let bnd_y = create_storage_buf(device, "bnd_y", &boundary.y);
        let bnd_z = create_storage_buf(device, "bnd_z", &boundary.z);
        let bnd_mass = create_storage_buf(device, "bnd_mass", &boundary.mass);
        let bnd_pressure = create_storage_buf(device, "bnd_pressure", &boundary.pressure);

        // Neighbor grid buffers
        let zeros_n = vec![0u32; n.max(1)];
        let zeros_cells = vec![0u32; total_cells.max(1)];

        let cell_indices = create_storage_buf_u32(device, "cell_indices", &zeros_n);
        let cell_counts = create_storage_buf_u32(device, "cell_counts", &zeros_cells);
        let cell_offsets = create_storage_buf_u32(device, "cell_offsets", &zeros_cells);
        let sorted_indices = create_storage_buf_u32(device, "sorted_indices", &zeros_n);
        let write_heads = create_storage_buf_u32(device, "write_heads", &zeros_cells);

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
        let staging_mass = create_staging_buf(device, "staging_mass", particle_bytes);
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
            staging_density,
            staging_pos_x,
            staging_pos_y,
            staging_pos_z,
            staging_vel_x,
            staging_vel_y,
            staging_vel_z,
            staging_pressure,
            staging_mass,
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

        encoder.copy_buffer_to_buffer(&self.pos_x, 0, &self.staging_pos_x, 0, byte_len);
        encoder.copy_buffer_to_buffer(&self.pos_y, 0, &self.staging_pos_y, 0, byte_len);
        encoder.copy_buffer_to_buffer(&self.pos_z, 0, &self.staging_pos_z, 0, byte_len);
        encoder.copy_buffer_to_buffer(&self.vel_x, 0, &self.staging_vel_x, 0, byte_len);
        encoder.copy_buffer_to_buffer(&self.vel_y, 0, &self.staging_vel_y, 0, byte_len);
        encoder.copy_buffer_to_buffer(&self.vel_z, 0, &self.staging_vel_z, 0, byte_len);
        encoder.copy_buffer_to_buffer(&self.density, 0, &self.staging_density, 0, byte_len);
        encoder.copy_buffer_to_buffer(&self.pressure, 0, &self.staging_pressure, 0, byte_len);
        encoder.copy_buffer_to_buffer(&self.mass, 0, &self.staging_mass, 0, byte_len);
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
        let mass_vec = read_f32_buffer(device, &self.staging_mass, n);
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
            mass: mass_vec,
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
