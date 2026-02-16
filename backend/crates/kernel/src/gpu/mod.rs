//! GPU (Metal/Vulkan via wgpu) implementation of the SPH simulation kernel.
//!
//! `GpuKernel` implements `SimulationKernel` using wgpu compute shaders.
//! On macOS, wgpu compiles to Metal for GPU acceleration.
//!
//! # Architecture
//! - Each simulation step dispatches 4 compute shader passes:
//!   1. Neighbor grid construction (4 sub-passes: clear, count, prefix-sum, scatter)
//!   2. Density summation + EOS pressure
//!   3. Force computation (pressure + viscous + gravity + boundary)
//!   4. Time integration (Velocity Verlet kick-drift-kick)
//! - Particle data lives on the GPU between steps; readback only on demand.
//!
//! # Bind group layout
//! Bindings are split across 4 bind groups to stay within Metal's 8-storage-buffer
//! per shader stage limit:
//!
//! - Group 0: SimParams (uniform) + particle positions + mass
//! - Group 1: Velocity + acceleration
//! - Group 2: SPH state (density, pressure, fluid_type) + boundary particle data
//! - Group 3: Neighbor grid data

pub mod buffers;

use buffers::{GpuBuffers, GpuSimParams};
use crate::boundary::BoundaryParticles;
use crate::eos;
use crate::particle::{FluidType, ParticleArrays};
use crate::{ErrorMetrics, SimulationKernel};

/// GPU-accelerated SPH simulation kernel using wgpu compute shaders.
pub struct GpuKernel {
    // wgpu resources
    device: wgpu::Device,
    queue: wgpu::Queue,

    // Compute pipelines
    pipeline_grid_clear: wgpu::ComputePipeline,
    pipeline_grid_count: wgpu::ComputePipeline,
    pipeline_grid_prefix: wgpu::ComputePipeline,
    pipeline_grid_scatter: wgpu::ComputePipeline,
    pipeline_density: wgpu::ComputePipeline,
    pipeline_boundary_pressure: wgpu::ComputePipeline,
    pipeline_forces: wgpu::ComputePipeline,
    pipeline_half_kick: wgpu::ComputePipeline,
    pipeline_drift: wgpu::ComputePipeline,

    // Bind group layouts -- per-group, per-shader-family
    // Grid shader: groups 0, 3
    bgl_grid_g0: wgpu::BindGroupLayout,
    bgl_grid_g3: wgpu::BindGroupLayout,
    // Density shader: groups 0, 2, 3
    bgl_density_g0: wgpu::BindGroupLayout,
    bgl_density_g2: wgpu::BindGroupLayout,
    bgl_density_g3: wgpu::BindGroupLayout,
    // Forces shader: groups 0, 1, 2, 3
    bgl_forces_g0: wgpu::BindGroupLayout,
    bgl_forces_g1: wgpu::BindGroupLayout,
    bgl_forces_g2: wgpu::BindGroupLayout,
    bgl_forces_g3: wgpu::BindGroupLayout,
    // Integrate shader: groups 0, 1
    bgl_integrate_g0: wgpu::BindGroupLayout,
    bgl_integrate_g1: wgpu::BindGroupLayout,

    // Empty bind group layout for unused group slots in pipeline layouts
    bgl_empty: wgpu::BindGroupLayout,

    // GPU buffers
    bufs: GpuBuffers,

    // Simulation parameters
    h: f32,
    gravity: [f32; 3],
    speed_of_sound: f32,
    #[allow(dead_code)]
    cfl_number: f32,
    #[allow(dead_code)]
    viscosity: f32,
    domain_min: [f32; 3],
    domain_max: [f32; 3],
    grid_dims: [u32; 3],

    // Cached CPU-side particle data (refreshed after each step).
    cached_particles: ParticleArrays,

    // Conservation tracking
    initial_energy: f64,
    initial_mass: f64,

    // First-step bootstrap flag
    needs_init: bool,
}

/// Error returned when GPU initialization fails.
#[derive(Debug)]
pub struct GpuInitError(pub String);

impl std::fmt::Display for GpuInitError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "GPU initialization failed: {}", self.0)
    }
}

impl std::error::Error for GpuInitError {}

/// Check whether a GPU (Metal on macOS) is available.
pub fn gpu_available() -> bool {
    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
        backends: wgpu::Backends::all(),
        ..Default::default()
    });
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        compatible_surface: None,
        force_fallback_adapter: false,
    }));
    adapter.is_some()
}

impl GpuKernel {
    /// Create a new GPU simulation kernel.
    ///
    /// Returns `Err(GpuInitError)` if no suitable GPU adapter is found, allowing
    /// callers to fall back to `CpuKernel`.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        particles: ParticleArrays,
        boundary: BoundaryParticles,
        h: f32,
        gravity: [f32; 3],
        speed_of_sound: f32,
        cfl_number: f32,
        viscosity: f32,
        domain_min: [f32; 3],
        domain_max: [f32; 3],
    ) -> Result<Self, GpuInitError> {
        // --- Device initialization ---
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });

        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        }))
        .ok_or_else(|| GpuInitError("No suitable GPU adapter found".into()))?;

        tracing::info!("GPU adapter: {:?}", adapter.get_info().name);

        // Request higher storage buffer limit.  The forces shader uses up to 21
        // storage buffers (split across 4 bind groups).  Metal on Apple Silicon
        // supports 31, but wgpu defaults to 8.  Ask for the adapter's actual
        // limit so compute shaders are not artificially constrained.
        let adapter_limits = adapter.limits();
        let mut required_limits = wgpu::Limits::default();
        required_limits.max_storage_buffers_per_shader_stage =
            adapter_limits.max_storage_buffers_per_shader_stage;
        // Also raise bind-group count to 4 (default is already 4, but be explicit).
        required_limits.max_bind_groups = adapter_limits.max_bind_groups.max(4);

        tracing::info!(
            "Requesting max_storage_buffers_per_shader_stage = {} (adapter supports {})",
            required_limits.max_storage_buffers_per_shader_stage,
            adapter_limits.max_storage_buffers_per_shader_stage,
        );

        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("sph_gpu_device"),
                required_features: wgpu::Features::empty(),
                required_limits,
                memory_hints: wgpu::MemoryHints::Performance,
            },
            None,
        ))
        .map_err(|e| GpuInitError(format!("Failed to create device: {e}")))?;

        // --- Grid dimensions ---
        let cell_size = 2.0 * h;
        let grid_dims = [
            ((domain_max[0] - domain_min[0]) / cell_size).ceil().max(1.0) as u32,
            ((domain_max[1] - domain_min[1]) / cell_size).ceil().max(1.0) as u32,
            ((domain_max[2] - domain_min[2]) / cell_size).ceil().max(1.0) as u32,
        ];

        // --- Initial params ---
        let sim_params = GpuSimParams {
            dt: 0.0,
            h,
            speed_of_sound,
            gravity_x: gravity[0],
            gravity_y: gravity[1],
            gravity_z: gravity[2],
            domain_min_x: domain_min[0],
            domain_min_y: domain_min[1],
            domain_min_z: domain_min[2],
            domain_max_x: domain_max[0],
            domain_max_y: domain_max[1],
            domain_max_z: domain_max[2],
            n_particles: particles.len() as u32,
            n_boundary: boundary.len() as u32,
            grid_dim_x: grid_dims[0],
            grid_dim_y: grid_dims[1],
            grid_dim_z: grid_dims[2],
            cell_size,
            viscosity_alpha: 1.0,
            viscosity_beta: 2.0,
            pass_index: 0,
            _pad1: 0,
        };

        // --- Conservation initial values ---
        let initial_mass: f64 = particles.mass.iter().map(|&m| m as f64).sum();
        let initial_energy = compute_total_energy(&particles, gravity);

        // --- Create buffers ---
        let bufs = GpuBuffers::new(&device, &particles, &boundary, grid_dims, &sim_params);

        // Cache initial particles
        let cached_particles = particles.clone();

        // --- Load shaders ---
        let grid_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("neighbor_grid"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("shaders/neighbor_grid.wgsl").into(),
            ),
        });

        let density_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("density"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("shaders/density.wgsl").into(),
            ),
        });

        let forces_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("forces"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("shaders/forces.wgsl").into(),
            ),
        });

        let integrate_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("integrate"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("shaders/integrate.wgsl").into(),
            ),
        });

        // --- Bind group layouts ---
        // Empty layout for unused group slots
        let bgl_empty = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("empty_bgl"),
            entries: &[],
        });

        // -- Grid shader layouts (group 0, group 3) --
        // Group 0: params(uniform), pos_x/y/z(read) -- no mass
        let bgl_grid_g0 = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("grid_g0_bgl"),
            entries: &[
                bgl_uniform(0),    // params
                bgl_storage_ro(1), // pos_x
                bgl_storage_ro(2), // pos_y
                bgl_storage_ro(3), // pos_z
            ],
        });
        // Group 3: cell_indices(rw), cell_counts(rw), cell_offsets(rw), sorted_indices(rw), write_heads(rw)
        let bgl_grid_g3 = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("grid_g3_bgl"),
            entries: &[
                bgl_storage_rw(0), // cell_indices
                bgl_storage_rw(1), // cell_counts
                bgl_storage_rw(2), // cell_offsets
                bgl_storage_rw(3), // sorted_indices
                bgl_storage_rw(4), // write_heads
            ],
        });

        // -- Density shader layouts (group 0, group 2, group 3) --
        // Group 0: params(uniform), pos_x/y/z(read), mass(read)
        let bgl_density_g0 = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("density_g0_bgl"),
            entries: &[
                bgl_uniform(0),    // params
                bgl_storage_ro(1), // pos_x
                bgl_storage_ro(2), // pos_y
                bgl_storage_ro(3), // pos_z
                bgl_storage_ro(4), // mass
            ],
        });
        // Group 2: density(rw), pressure(rw), fluid_type(read), bnd_x/y/z(read), bnd_mass(read) -- no bnd_pressure
        let bgl_density_g2 = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("density_g2_bgl"),
            entries: &[
                bgl_storage_rw(0), // density
                bgl_storage_rw(1), // pressure
                bgl_storage_ro(2), // fluid_type
                bgl_storage_ro(3), // bnd_x
                bgl_storage_ro(4), // bnd_y
                bgl_storage_ro(5), // bnd_z
                bgl_storage_ro(6), // bnd_mass
            ],
        });
        // Group 3: cell_counts(read), cell_offsets(read), sorted_indices(read) -- bindings 1,2,3
        let bgl_density_g3 = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("density_g3_bgl"),
            entries: &[
                bgl_storage_ro(1), // cell_counts
                bgl_storage_ro(2), // cell_offsets
                bgl_storage_ro(3), // sorted_indices
            ],
        });

        // -- Forces shader layouts (group 0, group 1, group 2, group 3) --
        // Group 0: same as density (params + pos + mass, all read)
        let bgl_forces_g0 = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("forces_g0_bgl"),
            entries: &[
                bgl_uniform(0),    // params
                bgl_storage_ro(1), // pos_x
                bgl_storage_ro(2), // pos_y
                bgl_storage_ro(3), // pos_z
                bgl_storage_ro(4), // mass
            ],
        });
        // Group 1: vel(read), acc(rw)
        let bgl_forces_g1 = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("forces_g1_bgl"),
            entries: &[
                bgl_storage_ro(0), // vel_x
                bgl_storage_ro(1), // vel_y
                bgl_storage_ro(2), // vel_z
                bgl_storage_rw(3), // acc_x
                bgl_storage_rw(4), // acc_y
                bgl_storage_rw(5), // acc_z
            ],
        });
        // Group 2: density(read), pressure(read), fluid_type(read), bnd(read), bnd_mass(read), bnd_pressure(rw)
        let bgl_forces_g2 = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("forces_g2_bgl"),
            entries: &[
                bgl_storage_ro(0), // density
                bgl_storage_ro(1), // pressure
                bgl_storage_ro(2), // fluid_type
                bgl_storage_ro(3), // bnd_x
                bgl_storage_ro(4), // bnd_y
                bgl_storage_ro(5), // bnd_z
                bgl_storage_ro(6), // bnd_mass
                bgl_storage_rw(7), // bnd_pressure
            ],
        });
        // Group 3: same as density group 3 (cell_counts, cell_offsets, sorted_indices read)
        let bgl_forces_g3 = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("forces_g3_bgl"),
            entries: &[
                bgl_storage_ro(1), // cell_counts
                bgl_storage_ro(2), // cell_offsets
                bgl_storage_ro(3), // sorted_indices
            ],
        });

        // -- Integrate shader layouts (group 0, group 1) --
        // Group 0: params(uniform), pos_x/y/z(rw) -- no mass
        let bgl_integrate_g0 = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("integrate_g0_bgl"),
            entries: &[
                bgl_uniform(0),    // params
                bgl_storage_rw(1), // pos_x
                bgl_storage_rw(2), // pos_y
                bgl_storage_rw(3), // pos_z
            ],
        });
        // Group 1: vel(rw), acc(read)
        let bgl_integrate_g1 = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("integrate_g1_bgl"),
            entries: &[
                bgl_storage_rw(0), // vel_x
                bgl_storage_rw(1), // vel_y
                bgl_storage_rw(2), // vel_z
                bgl_storage_ro(3), // acc_x
                bgl_storage_ro(4), // acc_y
                bgl_storage_ro(5), // acc_z
            ],
        });

        // --- Pipeline layouts ---
        // Grid: uses groups 0 and 3; empty groups at 1 and 2
        let pl_layout_grid = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("grid_pl"),
            bind_group_layouts: &[&bgl_grid_g0, &bgl_empty, &bgl_empty, &bgl_grid_g3],
            push_constant_ranges: &[],
        });
        // Density: uses groups 0, 2, 3; empty group at 1
        let pl_layout_density = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("density_pl"),
            bind_group_layouts: &[&bgl_density_g0, &bgl_empty, &bgl_density_g2, &bgl_density_g3],
            push_constant_ranges: &[],
        });
        // Forces: uses all 4 groups
        let pl_layout_forces = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("forces_pl"),
            bind_group_layouts: &[&bgl_forces_g0, &bgl_forces_g1, &bgl_forces_g2, &bgl_forces_g3],
            push_constant_ranges: &[],
        });
        // Integrate: uses groups 0, 1; empty groups at 2, 3 not needed (just list 2)
        let pl_layout_integrate = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("integrate_pl"),
            bind_group_layouts: &[&bgl_integrate_g0, &bgl_integrate_g1],
            push_constant_ranges: &[],
        });

        // --- Compute pipelines ---
        let pipeline_grid_clear = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("grid_clear"),
            layout: Some(&pl_layout_grid),
            module: &grid_shader,
            entry_point: Some("clear_counts"),
            compilation_options: Default::default(),
            cache: None,
        });
        let pipeline_grid_count = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("grid_count"),
            layout: Some(&pl_layout_grid),
            module: &grid_shader,
            entry_point: Some("count_particles"),
            compilation_options: Default::default(),
            cache: None,
        });
        let pipeline_grid_prefix = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("grid_prefix"),
            layout: Some(&pl_layout_grid),
            module: &grid_shader,
            entry_point: Some("prefix_sum"),
            compilation_options: Default::default(),
            cache: None,
        });
        let pipeline_grid_scatter = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("grid_scatter"),
            layout: Some(&pl_layout_grid),
            module: &grid_shader,
            entry_point: Some("scatter_particles"),
            compilation_options: Default::default(),
            cache: None,
        });

        let pipeline_density = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("density"),
            layout: Some(&pl_layout_density),
            module: &density_shader,
            entry_point: Some("compute_density"),
            compilation_options: Default::default(),
            cache: None,
        });

        let pipeline_boundary_pressure = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("boundary_pressure"),
            layout: Some(&pl_layout_forces),
            module: &forces_shader,
            entry_point: Some("update_boundary_pressures"),
            compilation_options: Default::default(),
            cache: None,
        });

        let pipeline_forces = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("forces"),
            layout: Some(&pl_layout_forces),
            module: &forces_shader,
            entry_point: Some("compute_forces"),
            compilation_options: Default::default(),
            cache: None,
        });

        let pipeline_half_kick = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("half_kick"),
            layout: Some(&pl_layout_integrate),
            module: &integrate_shader,
            entry_point: Some("half_kick"),
            compilation_options: Default::default(),
            cache: None,
        });

        let pipeline_drift = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("drift"),
            layout: Some(&pl_layout_integrate),
            module: &integrate_shader,
            entry_point: Some("drift"),
            compilation_options: Default::default(),
            cache: None,
        });

        Ok(Self {
            device,
            queue,
            pipeline_grid_clear,
            pipeline_grid_count,
            pipeline_grid_prefix,
            pipeline_grid_scatter,
            pipeline_density,
            pipeline_boundary_pressure,
            pipeline_forces,
            pipeline_half_kick,
            pipeline_drift,
            bgl_grid_g0,
            bgl_grid_g3,
            bgl_density_g0,
            bgl_density_g2,
            bgl_density_g3,
            bgl_forces_g0,
            bgl_forces_g1,
            bgl_forces_g2,
            bgl_forces_g3,
            bgl_integrate_g0,
            bgl_integrate_g1,
            bgl_empty,
            bufs,
            h,
            gravity,
            speed_of_sound,
            cfl_number,
            viscosity,
            domain_min,
            domain_max,
            grid_dims,
            cached_particles,
            initial_energy,
            initial_mass,
            needs_init: true,
        })
    }

    /// Dispatch the full force computation pipeline on the GPU:
    /// neighbor grid -> density -> boundary pressure -> forces
    fn compute_forces_gpu(&self, params: &GpuSimParams) {
        let n_particles = self.bufs.n_particles;
        let total_cells = self.bufs.total_cells;
        let n_boundary = self.bufs.n_boundary;

        let wg_particles = dispatch_size(n_particles, 256);
        let wg_cells = dispatch_size(total_cells, 256);
        let wg_boundary = dispatch_size(n_boundary.max(1), 256);

        // Update params
        self.bufs.update_params(&self.queue, params);

        // Create bind groups for grid shader (groups 0 and 3)
        let grid_bg0 = self.create_grid_bg0();
        let grid_bg3 = self.create_grid_bg3();
        let empty_bg = self.create_empty_bind_group();

        // Create bind groups for density shader (groups 0, 2, 3)
        let density_bg0 = self.create_density_bg0();
        let density_bg2 = self.create_density_bg2();
        let density_bg3 = self.create_density_forces_bg3();

        // Create bind groups for forces shader (groups 0, 1, 2, 3)
        let forces_bg0 = self.create_forces_bg0();
        let forces_bg1 = self.create_forces_bg1();
        let forces_bg2 = self.create_forces_bg2();
        let forces_bg3 = self.create_forces_bg3();

        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("force_pipeline"),
        });

        // 1. Neighbor grid: clear
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("grid_clear"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline_grid_clear);
            pass.set_bind_group(0, &grid_bg0, &[]);
            pass.set_bind_group(1, &empty_bg, &[]);
            pass.set_bind_group(2, &empty_bg, &[]);
            pass.set_bind_group(3, &grid_bg3, &[]);
            pass.dispatch_workgroups(wg_cells, 1, 1);
        }

        // 2. Neighbor grid: count
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("grid_count"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline_grid_count);
            pass.set_bind_group(0, &grid_bg0, &[]);
            pass.set_bind_group(1, &empty_bg, &[]);
            pass.set_bind_group(2, &empty_bg, &[]);
            pass.set_bind_group(3, &grid_bg3, &[]);
            pass.dispatch_workgroups(wg_particles, 1, 1);
        }

        // 3. Neighbor grid: prefix sum (single thread)
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("grid_prefix"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline_grid_prefix);
            pass.set_bind_group(0, &grid_bg0, &[]);
            pass.set_bind_group(1, &empty_bg, &[]);
            pass.set_bind_group(2, &empty_bg, &[]);
            pass.set_bind_group(3, &grid_bg3, &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }

        // 4. Neighbor grid: scatter
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("grid_scatter"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline_grid_scatter);
            pass.set_bind_group(0, &grid_bg0, &[]);
            pass.set_bind_group(1, &empty_bg, &[]);
            pass.set_bind_group(2, &empty_bg, &[]);
            pass.set_bind_group(3, &grid_bg3, &[]);
            pass.dispatch_workgroups(wg_particles, 1, 1);
        }

        // 5. Density summation + EOS pressure
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("density"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline_density);
            pass.set_bind_group(0, &density_bg0, &[]);
            pass.set_bind_group(1, &empty_bg, &[]);
            pass.set_bind_group(2, &density_bg2, &[]);
            pass.set_bind_group(3, &density_bg3, &[]);
            pass.dispatch_workgroups(wg_particles, 1, 1);
        }

        // 6. Boundary pressure mirroring
        if n_boundary > 0 {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("boundary_pressure"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline_boundary_pressure);
            pass.set_bind_group(0, &forces_bg0, &[]);
            pass.set_bind_group(1, &forces_bg1, &[]);
            pass.set_bind_group(2, &forces_bg2, &[]);
            pass.set_bind_group(3, &forces_bg3, &[]);
            pass.dispatch_workgroups(wg_boundary, 1, 1);
        }

        // 7. All forces (pressure + viscous + gravity + boundary repulsive)
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("forces"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline_forces);
            pass.set_bind_group(0, &forces_bg0, &[]);
            pass.set_bind_group(1, &forces_bg1, &[]);
            pass.set_bind_group(2, &forces_bg2, &[]);
            pass.set_bind_group(3, &forces_bg3, &[]);
            pass.dispatch_workgroups(wg_particles, 1, 1);
        }

        self.queue.submit(std::iter::once(encoder.finish()));
        self.device.poll(wgpu::Maintain::Wait);
    }

    /// Refresh the cached CPU-side particle data from GPU.
    fn refresh_cache(&mut self) {
        self.cached_particles = self.bufs.readback_particles(&self.device, &self.queue);
    }

    // ---- Bind group creation helpers ----

    /// Create an empty bind group for unused group slots.
    fn create_empty_bind_group(&self) -> wgpu::BindGroup {
        self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("empty_bg"),
            layout: &self.bgl_empty,
            entries: &[],
        })
    }

    // -- Grid shader bind groups --

    /// Grid group 0: params + pos_x/y/z (read)
    fn create_grid_bg0(&self) -> wgpu::BindGroup {
        self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("grid_bg0"),
            layout: &self.bgl_grid_g0,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: self.bufs.params_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: self.bufs.pos_x.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: self.bufs.pos_y.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: self.bufs.pos_z.as_entire_binding() },
            ],
        })
    }

    /// Grid group 3: cell_indices, cell_counts, cell_offsets, sorted_indices, write_heads (all rw)
    fn create_grid_bg3(&self) -> wgpu::BindGroup {
        self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("grid_bg3"),
            layout: &self.bgl_grid_g3,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: self.bufs.cell_indices.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: self.bufs.cell_counts.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: self.bufs.cell_offsets.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: self.bufs.sorted_indices.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: self.bufs.write_heads.as_entire_binding() },
            ],
        })
    }

    // -- Density shader bind groups --

    /// Density group 0: params + pos_x/y/z (read) + mass (read)
    fn create_density_bg0(&self) -> wgpu::BindGroup {
        self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("density_bg0"),
            layout: &self.bgl_density_g0,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: self.bufs.params_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: self.bufs.pos_x.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: self.bufs.pos_y.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: self.bufs.pos_z.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: self.bufs.mass.as_entire_binding() },
            ],
        })
    }

    /// Density group 2: density(rw), pressure(rw), fluid_type(read), bnd_x/y/z(read), bnd_mass(read)
    fn create_density_bg2(&self) -> wgpu::BindGroup {
        self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("density_bg2"),
            layout: &self.bgl_density_g2,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: self.bufs.density.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: self.bufs.pressure.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: self.bufs.fluid_type.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: self.bufs.bnd_x.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: self.bufs.bnd_y.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: self.bufs.bnd_z.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 6, resource: self.bufs.bnd_mass.as_entire_binding() },
            ],
        })
    }

    /// Density/Forces group 3 (read-only): cell_counts, cell_offsets, sorted_indices
    /// Used by density shader. Bindings at 1, 2, 3 (matching the shader declarations).
    fn create_density_forces_bg3(&self) -> wgpu::BindGroup {
        self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("density_forces_bg3"),
            layout: &self.bgl_density_g3,
            entries: &[
                wgpu::BindGroupEntry { binding: 1, resource: self.bufs.cell_counts.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: self.bufs.cell_offsets.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: self.bufs.sorted_indices.as_entire_binding() },
            ],
        })
    }

    // -- Forces shader bind groups --

    /// Forces group 0: params + pos_x/y/z (read) + mass (read)
    fn create_forces_bg0(&self) -> wgpu::BindGroup {
        self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("forces_bg0"),
            layout: &self.bgl_forces_g0,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: self.bufs.params_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: self.bufs.pos_x.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: self.bufs.pos_y.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: self.bufs.pos_z.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: self.bufs.mass.as_entire_binding() },
            ],
        })
    }

    /// Forces group 1: vel_x/y/z (read), acc_x/y/z (rw)
    fn create_forces_bg1(&self) -> wgpu::BindGroup {
        self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("forces_bg1"),
            layout: &self.bgl_forces_g1,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: self.bufs.vel_x.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: self.bufs.vel_y.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: self.bufs.vel_z.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: self.bufs.acc_x.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: self.bufs.acc_y.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: self.bufs.acc_z.as_entire_binding() },
            ],
        })
    }

    /// Forces group 2: density(read), pressure(read), fluid_type(read), bnd(read), bnd_pressure(rw)
    fn create_forces_bg2(&self) -> wgpu::BindGroup {
        self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("forces_bg2"),
            layout: &self.bgl_forces_g2,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: self.bufs.density.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: self.bufs.pressure.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: self.bufs.fluid_type.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: self.bufs.bnd_x.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: self.bufs.bnd_y.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: self.bufs.bnd_z.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 6, resource: self.bufs.bnd_mass.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 7, resource: self.bufs.bnd_pressure.as_entire_binding() },
            ],
        })
    }

    /// Forces group 3: cell_counts(read), cell_offsets(read), sorted_indices(read) -- bindings 1,2,3
    fn create_forces_bg3(&self) -> wgpu::BindGroup {
        self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("forces_bg3"),
            layout: &self.bgl_forces_g3,
            entries: &[
                wgpu::BindGroupEntry { binding: 1, resource: self.bufs.cell_counts.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: self.bufs.cell_offsets.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: self.bufs.sorted_indices.as_entire_binding() },
            ],
        })
    }

    // -- Integrate shader bind groups --

    /// Integrate group 0: params + pos_x/y/z (rw)
    fn create_integrate_bg0(&self) -> wgpu::BindGroup {
        self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("integrate_bg0"),
            layout: &self.bgl_integrate_g0,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: self.bufs.params_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: self.bufs.pos_x.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: self.bufs.pos_y.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: self.bufs.pos_z.as_entire_binding() },
            ],
        })
    }

    /// Integrate group 1: vel_x/y/z (rw), acc_x/y/z (read)
    fn create_integrate_bg1(&self) -> wgpu::BindGroup {
        self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("integrate_bg1"),
            layout: &self.bgl_integrate_g1,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: self.bufs.vel_x.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: self.bufs.vel_y.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: self.bufs.vel_z.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: self.bufs.acc_x.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: self.bufs.acc_y.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: self.bufs.acc_z.as_entire_binding() },
            ],
        })
    }

    fn make_params(&self, dt: f32) -> GpuSimParams {
        let cell_size = 2.0 * self.h;
        GpuSimParams {
            dt,
            h: self.h,
            speed_of_sound: self.speed_of_sound,
            gravity_x: self.gravity[0],
            gravity_y: self.gravity[1],
            gravity_z: self.gravity[2],
            domain_min_x: self.domain_min[0],
            domain_min_y: self.domain_min[1],
            domain_min_z: self.domain_min[2],
            domain_max_x: self.domain_max[0],
            domain_max_y: self.domain_max[1],
            domain_max_z: self.domain_max[2],
            n_particles: self.bufs.n_particles,
            n_boundary: self.bufs.n_boundary,
            grid_dim_x: self.grid_dims[0],
            grid_dim_y: self.grid_dims[1],
            grid_dim_z: self.grid_dims[2],
            cell_size,
            viscosity_alpha: 1.0,
            viscosity_beta: 2.0,
            pass_index: 0,
            _pad1: 0,
        }
    }
}

impl SimulationKernel for GpuKernel {
    fn step(&mut self, dt: f32) {
        let n_particles = self.bufs.n_particles;
        if n_particles == 0 {
            return;
        }

        let params = self.make_params(dt);
        let wg_particles = dispatch_size(n_particles, 256);

        // --- 0. Bootstrap: compute initial forces on first step ---
        if self.needs_init {
            self.compute_forces_gpu(&params);
            self.needs_init = false;
        }

        // Update params buffer with current dt
        self.bufs.update_params(&self.queue, &params);

        // --- 1. Half-kick: v += a * dt/2 ---
        {
            let bg0 = self.create_integrate_bg0();
            let bg1 = self.create_integrate_bg1();
            let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("half_kick"),
            });
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("half_kick"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.pipeline_half_kick);
                pass.set_bind_group(0, &bg0, &[]);
                pass.set_bind_group(1, &bg1, &[]);
                pass.dispatch_workgroups(wg_particles, 1, 1);
            }
            self.queue.submit(std::iter::once(encoder.finish()));
            self.device.poll(wgpu::Maintain::Wait);
        }

        // --- 2. Drift: x += v * dt + domain clamping ---
        {
            let bg0 = self.create_integrate_bg0();
            let bg1 = self.create_integrate_bg1();
            let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("drift"),
            });
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("drift"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.pipeline_drift);
                pass.set_bind_group(0, &bg0, &[]);
                pass.set_bind_group(1, &bg1, &[]);
                pass.dispatch_workgroups(wg_particles, 1, 1);
            }
            self.queue.submit(std::iter::once(encoder.finish()));
            self.device.poll(wgpu::Maintain::Wait);
        }

        // --- 3-7. Full force computation pipeline ---
        self.compute_forces_gpu(&params);

        // --- 8. Second half-kick: v += a * dt/2 ---
        {
            let bg0 = self.create_integrate_bg0();
            let bg1 = self.create_integrate_bg1();
            let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("half_kick_2"),
            });
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("half_kick_2"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.pipeline_half_kick);
                pass.set_bind_group(0, &bg0, &[]);
                pass.set_bind_group(1, &bg1, &[]);
                pass.dispatch_workgroups(wg_particles, 1, 1);
            }
            self.queue.submit(std::iter::once(encoder.finish()));
            self.device.poll(wgpu::Maintain::Wait);
        }

        // Readback particle data to keep CPU cache in sync
        self.refresh_cache();
    }

    fn particles(&self) -> &ParticleArrays {
        &self.cached_particles
    }

    fn error_metrics(&self) -> ErrorMetrics {
        let particles = self.particles();
        let n = particles.len();

        // Maximum density variation
        let mut max_density_var = 0.0_f32;
        for i in 0..n {
            let rest_rho = match particles.fluid_type[i] {
                FluidType::Water => eos::WATER_REST_DENSITY,
                FluidType::Air => eos::AIR_REST_DENSITY,
            };
            let var = (particles.density[i] - rest_rho).abs() / rest_rho;
            if var > max_density_var {
                max_density_var = var;
            }
        }

        // Energy conservation
        let current_energy = compute_total_energy(particles, self.gravity);
        let energy_drift = if self.initial_energy.abs() > 1.0e-12 {
            ((current_energy - self.initial_energy) / self.initial_energy).abs() as f32
        } else {
            (current_energy - self.initial_energy).abs() as f32
        };

        // Mass conservation
        let current_mass: f64 = particles.mass.iter().map(|&m| m as f64).sum();
        let mass_drift = if self.initial_mass.abs() > 1.0e-12 {
            ((current_mass - self.initial_mass) / self.initial_mass).abs() as f32
        } else {
            (current_mass - self.initial_mass).abs() as f32
        };

        ErrorMetrics {
            max_density_variation: max_density_var,
            energy_conservation: energy_drift,
            mass_conservation: mass_drift,
        }
    }

    fn particle_count(&self) -> usize {
        self.bufs.n_particles as usize
    }
}

/// Compute total energy (kinetic + gravitational potential).
fn compute_total_energy(particles: &ParticleArrays, gravity: [f32; 3]) -> f64 {
    let mut energy = 0.0_f64;
    for i in 0..particles.len() {
        let m = particles.mass[i] as f64;
        let vx = particles.vx[i] as f64;
        let vy = particles.vy[i] as f64;
        let vz = particles.vz[i] as f64;
        energy += 0.5 * m * (vx * vx + vy * vy + vz * vz);
        let x = particles.x[i] as f64;
        let y = particles.y[i] as f64;
        let z = particles.z[i] as f64;
        energy -= m * (gravity[0] as f64 * x + gravity[1] as f64 * y + gravity[2] as f64 * z);
    }
    energy
}

/// Calculate dispatch workgroup count: ceil(total / workgroup_size).
fn dispatch_size(total: u32, workgroup_size: u32) -> u32 {
    (total + workgroup_size - 1) / workgroup_size
}

// ---- Bind group layout entry helpers ----

fn bgl_uniform(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn bgl_storage_ro(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only: true },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn bgl_storage_rw(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only: false },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}
