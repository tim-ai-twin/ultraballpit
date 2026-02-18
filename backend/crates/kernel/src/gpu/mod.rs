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

use std::cell::{Cell, UnsafeCell};
use std::time::Instant;

use wgpu::util::DeviceExt;
use buffers::{GpuBuffers, GpuSimParams};
use crate::boundary::BoundaryParticles;
use crate::eos;
use crate::particle::{FluidType, ParticleArrays};
use crate::{ErrorMetrics, SimulationKernel};

/// Per-pass wall-clock timing breakdown of a single GPU simulation step.
#[derive(Debug, Clone, Copy, Default)]
pub struct GpuStepProfile {
    /// Neighbor grid build: clear + count + prefix-sum + scatter (microseconds).
    pub grid_build_us: u64,
    /// Density summation + EOS pressure (microseconds).
    pub density_us: u64,
    /// Adami boundary pressure mirroring (microseconds).
    pub boundary_pressure_us: u64,
    /// Force computation: pressure + viscous + gravity + repulsive (microseconds).
    pub forces_us: u64,
    /// Integration: half_kick + drift + half_kick (microseconds).
    pub integrate_us: u64,
    /// GPU-to-CPU particle data readback (microseconds).
    pub readback_us: u64,
    /// Total wall-clock time for the entire step (microseconds).
    pub total_us: u64,
}

/// Uniform params for the reorder shader.
#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct ReorderParams {
    n_particles: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

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
    pipeline_xsph: wgpu::ComputePipeline,
    pipeline_reorder: wgpu::ComputePipeline,

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

    // Reorder shader: group 0 (params + perm + source + dest)
    bgl_reorder_g0: wgpu::BindGroupLayout,
    reorder_params_buffer: wgpu::Buffer,

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

    // Cached CPU-side particle data (refreshed lazily via interior mutability).
    cached_particles: UnsafeCell<ParticleArrays>,

    // Conservation tracking
    initial_energy: f64,
    initial_mass: f64,

    // First-step bootstrap flag
    needs_init: bool,

    // Lazy readback: true when GPU data is newer than cached_particles.
    // Uses Cell for interior mutability so particles() can trigger readback.
    cache_dirty: Cell<bool>,

    // Workgroup size used for compute dispatches (default 256).
    workgroup_size: u32,

    // Verlet neighbor list: skip grid rebuild when particles haven't moved far.
    // Accumulated estimated max displacement since last grid build.
    verlet_displacement: f32,
    // Skin distance: rebuild when displacement exceeds skin/2.
    // Grid uses support_radius + skin for neighbor search, so the list
    // remains valid as long as no particle moves more than skin/2.
    verlet_skin: f32,

    // Particle reorder: step counter for periodic Morton-order resorting
    steps_since_reorder: u32,
    reorder_interval: u32,

    // GPU timestamp query resources for precise profiling
    timestamp_query_set: wgpu::QuerySet,
    timestamp_resolve_buf: wgpu::Buffer,
    timestamp_staging_buf: wgpu::Buffer,
    timestamp_period: f32,
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
                required_features: wgpu::Features::TIMESTAMP_QUERY,
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

        // Cache initial particles (wrapped in UnsafeCell for lazy readback)
        let cached_particles = UnsafeCell::new(particles.clone());

        // --- Load shaders ---
        // Default workgroup size; can be overridden via set_workgroup_size() before use.
        let workgroup_size = 256u32;
        let wg_str = format!("@workgroup_size({})", workgroup_size);

        let grid_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("neighbor_grid"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("shaders/neighbor_grid.wgsl").into(),
            ),
        });

        let density_src: String = include_str!("shaders/density.wgsl")
            .replace("@workgroup_size(256)", &wg_str);
        let density_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("density"),
            source: wgpu::ShaderSource::Wgsl(density_src.into()),
        });

        let forces_src: String = include_str!("shaders/forces.wgsl")
            .replace("@workgroup_size(256)", &wg_str);
        let forces_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("forces"),
            source: wgpu::ShaderSource::Wgsl(forces_src.into()),
        });

        let integrate_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("integrate"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("shaders/integrate.wgsl").into(),
            ),
        });

        let xsph_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("xsph"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("shaders/xsph.wgsl").into(),
            ),
        });

        let reorder_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("reorder"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("shaders/reorder.wgsl").into(),
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
        // Group 2: density(rw), pressure(rw), fluid_type(read), bnd(read), bnd_grid(read)
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
                bgl_storage_ro(7), // bnd_cell_counts
                bgl_storage_ro(8), // bnd_cell_offsets
                bgl_storage_ro(9), // bnd_sorted_indices
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
        // Group 2: density(read), pressure(read), fluid_type(read), bnd(read), bnd_pressure(rw), bnd_grid(read)
        let bgl_forces_g2 = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("forces_g2_bgl"),
            entries: &[
                bgl_storage_ro(0),  // density
                bgl_storage_ro(1),  // pressure
                bgl_storage_ro(2),  // fluid_type
                bgl_storage_ro(3),  // bnd_x
                bgl_storage_ro(4),  // bnd_y
                bgl_storage_ro(5),  // bnd_z
                bgl_storage_ro(6),  // bnd_mass
                bgl_storage_rw(7),  // bnd_pressure
                bgl_storage_ro(8),  // bnd_cell_counts
                bgl_storage_ro(9),  // bnd_cell_offsets
                bgl_storage_ro(10), // bnd_sorted_indices
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

        // -- Reorder shader layout (group 0) --
        // Group 0: params(uniform), perm(read), source(read), dest(rw)
        let bgl_reorder_g0 = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("reorder_g0_bgl"),
            entries: &[
                bgl_uniform(0),    // reorder params
                bgl_storage_ro(1), // perm (sorted_indices)
                bgl_storage_ro(2), // source buffer
                bgl_storage_rw(3), // dest buffer (temp)
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

        // XSPH pipeline (uses forces layout — same bind groups)
        let pipeline_xsph = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("xsph"),
            layout: Some(&pl_layout_forces),
            module: &xsph_shader,
            entry_point: Some("compute_xsph"),
            compilation_options: Default::default(),
            cache: None,
        });

        // Reorder pipeline
        let pl_layout_reorder = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("reorder_pl"),
            bind_group_layouts: &[&bgl_reorder_g0],
            push_constant_ranges: &[],
        });
        let pipeline_reorder = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("reorder_scatter"),
            layout: Some(&pl_layout_reorder),
            module: &reorder_shader,
            entry_point: Some("scatter"),
            compilation_options: Default::default(),
            cache: None,
        });

        // Reorder params uniform buffer
        let reorder_params = ReorderParams {
            n_particles: particles.len() as u32,
            _pad0: 0, _pad1: 0, _pad2: 0,
        };
        let reorder_params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("reorder_params"),
            contents: bytemuck::bytes_of(&reorder_params),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        // --- Timestamp query resources ---
        // 16 timestamps: pairs of (begin, end) for up to 8 passes
        let timestamp_query_set = device.create_query_set(&wgpu::QuerySetDescriptor {
            label: Some("timestamps"),
            ty: wgpu::QueryType::Timestamp,
            count: 16,
        });
        let timestamp_resolve_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("timestamp_resolve"),
            size: 16 * 8, // 16 u64 timestamps
            usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let timestamp_staging_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("timestamp_staging"),
            size: 16 * 8,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let timestamp_period = queue.get_timestamp_period();

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
            pipeline_xsph,
            pipeline_reorder,
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
            bgl_reorder_g0,
            reorder_params_buffer,
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
            cache_dirty: Cell::new(false),
            workgroup_size,
            // Verlet skin: fraction of smoothing length. Conservative choice.
            verlet_skin: 0.5 * h,
            verlet_displacement: f32::MAX, // Force first rebuild
            steps_since_reorder: 0,
            reorder_interval: 50, // Reorder every 50 steps
            timestamp_query_set,
            timestamp_resolve_buf,
            timestamp_staging_buf,
            timestamp_period,
        })
    }

    /// Set workgroup size for density and forces shaders and rebuild pipelines.
    ///
    /// Only 32, 64, 128, 256 are valid. Grid and integrate shaders stay at 256.
    /// Must be called before any `step()` calls.
    pub fn set_workgroup_size(&mut self, wg: u32) {
        assert!(matches!(wg, 32 | 64 | 128 | 256), "workgroup_size must be 32, 64, 128, or 256");
        self.workgroup_size = wg;

        let wg_str = format!("@workgroup_size({})", wg);

        let density_src: String = include_str!("shaders/density.wgsl")
            .replace("@workgroup_size(256)", &wg_str);
        let density_shader = self.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("density"),
            source: wgpu::ShaderSource::Wgsl(density_src.into()),
        });

        let forces_src: String = include_str!("shaders/forces.wgsl")
            .replace("@workgroup_size(256)", &wg_str);
        let forces_shader = self.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("forces"),
            source: wgpu::ShaderSource::Wgsl(forces_src.into()),
        });

        // Rebuild affected pipeline layouts
        let pl_layout_density = self.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("density_pl"),
            bind_group_layouts: &[&self.bgl_density_g0, &self.bgl_empty, &self.bgl_density_g2, &self.bgl_density_g3],
            push_constant_ranges: &[],
        });
        let pl_layout_forces = self.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("forces_pl"),
            bind_group_layouts: &[&self.bgl_forces_g0, &self.bgl_forces_g1, &self.bgl_forces_g2, &self.bgl_forces_g3],
            push_constant_ranges: &[],
        });

        self.pipeline_density = self.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("density"),
            layout: Some(&pl_layout_density),
            module: &density_shader,
            entry_point: Some("compute_density"),
            compilation_options: Default::default(),
            cache: None,
        });
        self.pipeline_boundary_pressure = self.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("boundary_pressure"),
            layout: Some(&pl_layout_forces),
            module: &forces_shader,
            entry_point: Some("update_boundary_pressures"),
            compilation_options: Default::default(),
            cache: None,
        });
        self.pipeline_forces = self.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("forces"),
            layout: Some(&pl_layout_forces),
            module: &forces_shader,
            entry_point: Some("compute_forces"),
            compilation_options: Default::default(),
            cache: None,
        });
    }

    /// Encode the neighbor grid build passes into a command encoder.
    fn encode_grid(&self, encoder: &mut wgpu::CommandEncoder) {
        let n_particles = self.bufs.n_particles;
        let total_cells = self.bufs.total_cells;

        let wg_grid_particles = dispatch_size(n_particles, 256);
        let wg_cells = dispatch_size(total_cells, 256);

        let grid_bg0 = self.create_grid_bg0();
        let grid_bg3 = self.create_grid_bg3();
        let empty_bg = self.create_empty_bind_group();

        // 1. Clear
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("grid_clear"), timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline_grid_clear);
            pass.set_bind_group(0, &grid_bg0, &[]); pass.set_bind_group(1, &empty_bg, &[]);
            pass.set_bind_group(2, &empty_bg, &[]); pass.set_bind_group(3, &grid_bg3, &[]);
            pass.dispatch_workgroups(wg_cells, 1, 1);
        }
        // 2. Count
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("grid_count"), timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline_grid_count);
            pass.set_bind_group(0, &grid_bg0, &[]); pass.set_bind_group(1, &empty_bg, &[]);
            pass.set_bind_group(2, &empty_bg, &[]); pass.set_bind_group(3, &grid_bg3, &[]);
            pass.dispatch_workgroups(wg_grid_particles, 1, 1);
        }
        // 3. Prefix sum (parallel, 256 threads)
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("grid_prefix"), timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline_grid_prefix);
            pass.set_bind_group(0, &grid_bg0, &[]); pass.set_bind_group(1, &empty_bg, &[]);
            pass.set_bind_group(2, &empty_bg, &[]); pass.set_bind_group(3, &grid_bg3, &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }
        // 4. Scatter
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("grid_scatter"), timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline_grid_scatter);
            pass.set_bind_group(0, &grid_bg0, &[]); pass.set_bind_group(1, &empty_bg, &[]);
            pass.set_bind_group(2, &empty_bg, &[]); pass.set_bind_group(3, &grid_bg3, &[]);
            pass.dispatch_workgroups(wg_grid_particles, 1, 1);
        }
    }

    /// Encode density + boundary pressure + forces passes into a command encoder.
    fn encode_density_forces(&self, encoder: &mut wgpu::CommandEncoder, params: &GpuSimParams) {
        let n_particles = self.bufs.n_particles;
        let n_boundary = self.bufs.n_boundary;
        let wg = self.workgroup_size;

        let wg_particles = dispatch_size(n_particles, wg);
        let wg_boundary = dispatch_size(n_boundary.max(1), wg);

        self.bufs.update_params(&self.queue, params);

        let density_bg0 = self.create_density_bg0();
        let density_bg2 = self.create_density_bg2();
        let density_bg3 = self.create_density_forces_bg3();
        let forces_bg0 = self.create_forces_bg0();
        let forces_bg1 = self.create_forces_bg1();
        let forces_bg2 = self.create_forces_bg2();
        let forces_bg3 = self.create_forces_bg3();
        let empty_bg = self.create_empty_bind_group();

        // Density summation + EOS pressure
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("density"), timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline_density);
            pass.set_bind_group(0, &density_bg0, &[]); pass.set_bind_group(1, &empty_bg, &[]);
            pass.set_bind_group(2, &density_bg2, &[]); pass.set_bind_group(3, &density_bg3, &[]);
            pass.dispatch_workgroups(wg_particles, 1, 1);
        }

        // Boundary pressure mirroring
        if n_boundary > 0 {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("boundary_pressure"), timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline_boundary_pressure);
            pass.set_bind_group(0, &forces_bg0, &[]); pass.set_bind_group(1, &forces_bg1, &[]);
            pass.set_bind_group(2, &forces_bg2, &[]); pass.set_bind_group(3, &forces_bg3, &[]);
            pass.dispatch_workgroups(wg_boundary, 1, 1);
        }

        // All forces
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("forces"), timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline_forces);
            pass.set_bind_group(0, &forces_bg0, &[]); pass.set_bind_group(1, &forces_bg1, &[]);
            pass.set_bind_group(2, &forces_bg2, &[]); pass.set_bind_group(3, &forces_bg3, &[]);
            pass.dispatch_workgroups(wg_particles, 1, 1);
        }
    }

    /// Encode the full force computation pipeline into a command encoder:
    /// [neighbor grid if needed] -> density -> boundary pressure -> forces.
    ///
    /// Uses Verlet neighbor lists to skip grid rebuild when particles
    /// haven't moved more than skin/2 since the last rebuild.
    fn encode_forces(&mut self, encoder: &mut wgpu::CommandEncoder, params: &GpuSimParams) {
        let needs_grid = self.verlet_displacement >= self.verlet_skin * 0.5;
        if needs_grid {
            self.encode_grid(encoder);
            self.verlet_displacement = 0.0;
        }
        self.encode_density_forces(encoder, params);
    }

    /// Submit the force computation pipeline as a standalone operation.
    /// Used only for the initial bootstrap step (always rebuilds grid).
    fn compute_forces_gpu(&mut self, params: &GpuSimParams) {
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("force_pipeline"),
        });
        // Always build grid on init — force Verlet rebuild
        self.verlet_displacement = f32::MAX;
        self.encode_forces(&mut encoder, params);
        self.queue.submit(std::iter::once(encoder.finish()));
        self.device.poll(wgpu::Maintain::Wait);
    }

    /// Encode integration passes (half_kick, xsph, drift) into an encoder.
    fn encode_integrate(&self, encoder: &mut wgpu::CommandEncoder, wg_particles: u32) {
        let bg0 = self.create_integrate_bg0();
        let bg1 = self.create_integrate_bg1();

        // Half-kick: v += a * dt/2
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

        // XSPH: compute smoothed velocity correction, write to acc buffers.
        // Uses forces-style bind groups (acc as read_write, grid access).
        // The previous step's neighbor grid is still valid since positions
        // haven't changed yet in this step.
        {
            let xsph_bg0 = self.create_forces_bg0();
            let xsph_bg1 = self.create_forces_bg1();
            let xsph_bg2 = self.create_forces_bg2();
            let xsph_bg3 = self.create_forces_bg3();
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("xsph"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline_xsph);
            pass.set_bind_group(0, &xsph_bg0, &[]);
            pass.set_bind_group(1, &xsph_bg1, &[]);
            pass.set_bind_group(2, &xsph_bg2, &[]);
            pass.set_bind_group(3, &xsph_bg3, &[]);
            pass.dispatch_workgroups(wg_particles, 1, 1);
        }

        // Drift: x += (v + xsph_correction) * dt + domain clamping
        // acc buffers now hold XSPH corrections (from XSPH pass above)
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
    }

    /// Encode the final half-kick pass into an encoder.
    fn encode_half_kick(&self, encoder: &mut wgpu::CommandEncoder, wg_particles: u32) {
        let bg0 = self.create_integrate_bg0();
        let bg1 = self.create_integrate_bg1();
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("half_kick_2"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pipeline_half_kick);
        pass.set_bind_group(0, &bg0, &[]);
        pass.set_bind_group(1, &bg1, &[]);
        pass.dispatch_workgroups(wg_particles, 1, 1);
    }

    /// Ensure the CPU-side particle cache is up-to-date with GPU data.
    ///
    /// Uses interior mutability (UnsafeCell + Cell) so this can be called
    /// from `&self` methods like `particles()` and `error_metrics()`.
    ///
    /// # Safety
    /// Safe because GpuKernel is `!Sync` (wgpu::Device is !Sync), so no
    /// concurrent access is possible. The UnsafeCell is only mutated here,
    /// and only when cache_dirty is true (preventing re-entrant mutation).
    fn ensure_cache(&self) {
        if self.cache_dirty.get() {
            let data = self.bufs.readback_particles(&self.device, &self.queue);
            // SAFETY: No other reference to cached_particles can exist because
            // we only hand out references after this call completes, and
            // GpuKernel is !Sync so no concurrent access.
            unsafe { *self.cached_particles.get() = data; }
            self.cache_dirty.set(false);
        }
    }

    /// Execute one simulation step without waiting for GPU completion.
    ///
    /// The command buffer is submitted but `poll(Wait)` is not called.
    /// Call `sync()` after a batch of steps to ensure all GPU work is done.
    /// This allows the GPU command processor to queue multiple steps,
    /// eliminating CPU-GPU sync latency between steps.
    pub fn step_no_sync(&mut self, dt: f32) {
        let n_particles = self.bufs.n_particles;
        if n_particles == 0 {
            return;
        }

        let params = self.make_params(dt);
        let wg_particles = dispatch_size(n_particles, 256);

        if self.needs_init {
            self.compute_forces_gpu(&params);
            self.needs_init = false;
        }

        // Periodic particle reorder
        self.steps_since_reorder += 1;
        if self.steps_since_reorder >= self.reorder_interval {
            self.reorder_particles();
        }

        self.bufs.update_params(&self.queue, &params);

        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("step_nosync"),
        });
        self.encode_integrate(&mut encoder, wg_particles);
        self.encode_forces(&mut encoder, &params);
        self.encode_half_kick(&mut encoder, wg_particles);

        self.queue.submit(std::iter::once(encoder.finish()));

        // Track estimated max displacement for Verlet neighbor list reuse.
        self.verlet_displacement += self.speed_of_sound * dt;

        self.cache_dirty.set(true);
    }

    /// Wait for all submitted GPU work to complete.
    pub fn sync(&self) {
        self.device.poll(wgpu::Maintain::Wait);
    }

    /// Execute one simulation step with per-pass GPU timestamp profiling.
    ///
    /// Uses hardware timestamp queries for precise GPU-side timing.
    /// Each phase gets its own compute pass with timestamp writes.
    /// All passes are batched into a single submit for minimal overhead.
    pub fn step_profiled(&mut self, dt: f32) -> GpuStepProfile {
        let total_start = Instant::now();
        let n_particles = self.bufs.n_particles;
        if n_particles == 0 {
            return GpuStepProfile::default();
        }

        let params = self.make_params(dt);
        let wg_integrate = dispatch_size(n_particles, 256);
        let wg = self.workgroup_size;

        if self.needs_init {
            self.compute_forces_gpu(&params);
            self.needs_init = false;
        }

        self.bufs.update_params(&self.queue, &params);

        let n_total = self.bufs.n_particles;
        let total_cells = self.bufs.total_cells;
        let n_boundary = self.bufs.n_boundary;
        let wg_grid_p = dispatch_size(n_total, 256);
        let wg_cells = dispatch_size(total_cells, 256);
        let wg_particles = dispatch_size(n_total, wg);
        let wg_boundary = dispatch_size(n_boundary.max(1), wg);

        // Create all bind groups upfront
        let grid_bg0 = self.create_grid_bg0();
        let grid_bg3 = self.create_grid_bg3();
        let empty_bg = self.create_empty_bind_group();
        let density_bg0 = self.create_density_bg0();
        let density_bg2 = self.create_density_bg2();
        let density_bg3 = self.create_density_forces_bg3();
        let forces_bg0 = self.create_forces_bg0();
        let forces_bg1 = self.create_forces_bg1();
        let forces_bg2 = self.create_forces_bg2();
        let forces_bg3 = self.create_forces_bg3();
        let integrate_bg0 = self.create_integrate_bg0();
        let integrate_bg1 = self.create_integrate_bg1();

        let qs = &self.timestamp_query_set;
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("step_profiled"),
        });

        // TS 0-1: Integrate (half_kick + xsph + drift)
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("prof_half_kick"),
                timestamp_writes: Some(wgpu::ComputePassTimestampWrites {
                    query_set: qs, beginning_of_pass_write_index: Some(0), end_of_pass_write_index: None,
                }),
            });
            pass.set_pipeline(&self.pipeline_half_kick);
            pass.set_bind_group(0, &integrate_bg0, &[]);
            pass.set_bind_group(1, &integrate_bg1, &[]);
            pass.dispatch_workgroups(wg_integrate, 1, 1);
        }
        // XSPH: compute velocity correction → acc buffers
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("prof_xsph"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline_xsph);
            pass.set_bind_group(0, &forces_bg0, &[]);
            pass.set_bind_group(1, &forces_bg1, &[]);
            pass.set_bind_group(2, &forces_bg2, &[]);
            pass.set_bind_group(3, &forces_bg3, &[]);
            pass.dispatch_workgroups(wg_integrate, 1, 1);
        }
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("prof_drift"),
                timestamp_writes: Some(wgpu::ComputePassTimestampWrites {
                    query_set: qs, beginning_of_pass_write_index: None, end_of_pass_write_index: Some(1),
                }),
            });
            pass.set_pipeline(&self.pipeline_drift);
            pass.set_bind_group(0, &integrate_bg0, &[]);
            pass.set_bind_group(1, &integrate_bg1, &[]);
            pass.dispatch_workgroups(wg_integrate, 1, 1);
        }

        // TS 2-3: Grid build (clear, count, prefix, scatter)
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("prof_grid_clear"),
                timestamp_writes: Some(wgpu::ComputePassTimestampWrites {
                    query_set: qs, beginning_of_pass_write_index: Some(2), end_of_pass_write_index: None,
                }),
            });
            pass.set_pipeline(&self.pipeline_grid_clear);
            pass.set_bind_group(0, &grid_bg0, &[]); pass.set_bind_group(1, &empty_bg, &[]);
            pass.set_bind_group(2, &empty_bg, &[]); pass.set_bind_group(3, &grid_bg3, &[]);
            pass.dispatch_workgroups(wg_cells, 1, 1);
        }
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("prof_grid_count"), timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline_grid_count);
            pass.set_bind_group(0, &grid_bg0, &[]); pass.set_bind_group(1, &empty_bg, &[]);
            pass.set_bind_group(2, &empty_bg, &[]); pass.set_bind_group(3, &grid_bg3, &[]);
            pass.dispatch_workgroups(wg_grid_p, 1, 1);
        }
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("prof_grid_prefix"), timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline_grid_prefix);
            pass.set_bind_group(0, &grid_bg0, &[]); pass.set_bind_group(1, &empty_bg, &[]);
            pass.set_bind_group(2, &empty_bg, &[]); pass.set_bind_group(3, &grid_bg3, &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("prof_grid_scatter"),
                timestamp_writes: Some(wgpu::ComputePassTimestampWrites {
                    query_set: qs, beginning_of_pass_write_index: None, end_of_pass_write_index: Some(3),
                }),
            });
            pass.set_pipeline(&self.pipeline_grid_scatter);
            pass.set_bind_group(0, &grid_bg0, &[]); pass.set_bind_group(1, &empty_bg, &[]);
            pass.set_bind_group(2, &empty_bg, &[]); pass.set_bind_group(3, &grid_bg3, &[]);
            pass.dispatch_workgroups(wg_grid_p, 1, 1);
        }

        // TS 4-5: Density
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("prof_density"),
                timestamp_writes: Some(wgpu::ComputePassTimestampWrites {
                    query_set: qs, beginning_of_pass_write_index: Some(4), end_of_pass_write_index: Some(5),
                }),
            });
            pass.set_pipeline(&self.pipeline_density);
            pass.set_bind_group(0, &density_bg0, &[]); pass.set_bind_group(1, &empty_bg, &[]);
            pass.set_bind_group(2, &density_bg2, &[]); pass.set_bind_group(3, &density_bg3, &[]);
            pass.dispatch_workgroups(wg_particles, 1, 1);
        }

        // TS 6-7: Boundary pressure
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("prof_boundary_pressure"),
                timestamp_writes: Some(wgpu::ComputePassTimestampWrites {
                    query_set: qs, beginning_of_pass_write_index: Some(6), end_of_pass_write_index: Some(7),
                }),
            });
            if n_boundary > 0 {
                pass.set_pipeline(&self.pipeline_boundary_pressure);
                pass.set_bind_group(0, &forces_bg0, &[]); pass.set_bind_group(1, &forces_bg1, &[]);
                pass.set_bind_group(2, &forces_bg2, &[]); pass.set_bind_group(3, &forces_bg3, &[]);
                pass.dispatch_workgroups(wg_boundary, 1, 1);
            }
            // Empty dispatch if no boundary — timestamps still written
        }

        // TS 8-9: Forces
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("prof_forces"),
                timestamp_writes: Some(wgpu::ComputePassTimestampWrites {
                    query_set: qs, beginning_of_pass_write_index: Some(8), end_of_pass_write_index: Some(9),
                }),
            });
            pass.set_pipeline(&self.pipeline_forces);
            pass.set_bind_group(0, &forces_bg0, &[]); pass.set_bind_group(1, &forces_bg1, &[]);
            pass.set_bind_group(2, &forces_bg2, &[]); pass.set_bind_group(3, &forces_bg3, &[]);
            pass.dispatch_workgroups(wg_particles, 1, 1);
        }

        // TS 10-11: Second half-kick
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("prof_half_kick_2"),
                timestamp_writes: Some(wgpu::ComputePassTimestampWrites {
                    query_set: qs, beginning_of_pass_write_index: Some(10), end_of_pass_write_index: Some(11),
                }),
            });
            pass.set_pipeline(&self.pipeline_half_kick);
            pass.set_bind_group(0, &integrate_bg0, &[]);
            pass.set_bind_group(1, &integrate_bg1, &[]);
            pass.dispatch_workgroups(wg_integrate, 1, 1);
        }

        // Resolve timestamps and copy to staging
        encoder.resolve_query_set(qs, 0..12, &self.timestamp_resolve_buf, 0);
        encoder.copy_buffer_to_buffer(
            &self.timestamp_resolve_buf, 0,
            &self.timestamp_staging_buf, 0,
            12 * 8,
        );

        self.queue.submit(std::iter::once(encoder.finish()));
        self.device.poll(wgpu::Maintain::Wait);

        // Read timestamps
        let timestamps = self.read_timestamps(12);
        let ns_per_tick = self.timestamp_period as f64;

        let ts_to_us = |begin: u64, end: u64| -> u64 {
            ((end.saturating_sub(begin)) as f64 * ns_per_tick / 1000.0) as u64
        };

        let integrate_us = ts_to_us(timestamps[0], timestamps[1])
            + ts_to_us(timestamps[10], timestamps[11]);
        let grid_build_us = ts_to_us(timestamps[2], timestamps[3]);
        let density_us = ts_to_us(timestamps[4], timestamps[5]);
        let boundary_pressure_us = ts_to_us(timestamps[6], timestamps[7]);
        let forces_us = ts_to_us(timestamps[8], timestamps[9]);

        // Reset Verlet displacement since profiling always rebuilds grid
        self.verlet_displacement = 0.0;
        self.verlet_displacement += self.speed_of_sound * dt;

        // Readback
        let t0 = Instant::now();
        let data = self.bufs.readback_particles(&self.device, &self.queue);
        unsafe { *self.cached_particles.get() = data; }
        self.cache_dirty.set(false);
        let readback_us = t0.elapsed().as_micros() as u64;

        let total_us = total_start.elapsed().as_micros() as u64;

        GpuStepProfile {
            grid_build_us,
            density_us,
            boundary_pressure_us,
            forces_us,
            integrate_us,
            readback_us,
            total_us,
        }
    }

    /// Read back N u64 timestamps from the staging buffer.
    fn read_timestamps(&self, count: usize) -> Vec<u64> {
        let slice = self.timestamp_staging_buf.slice(..((count * 8) as u64));
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            tx.send(result).unwrap();
        });
        self.device.poll(wgpu::Maintain::Wait);
        rx.recv().unwrap().unwrap();

        let data = slice.get_mapped_range();
        let result: Vec<u64> = bytemuck::cast_slice(&data)[..count].to_vec();
        drop(data);
        self.timestamp_staging_buf.unmap();
        result
    }

    /// Reorder all particle buffers into spatially-sorted order.
    ///
    /// Uses the grid's sorted_indices as a permutation to rearrange particle
    /// data so that spatially adjacent particles are contiguous in memory.
    /// This dramatically improves GPU cache hit rates for neighbor traversal.
    ///
    /// After reorder, forces a grid rebuild on the next step since particle
    /// indices have changed.
    fn reorder_particles(&mut self) {
        let n = self.bufs.n_particles;
        if n == 0 {
            return;
        }

        let wg = dispatch_size(n, 256);
        let byte_len = (n as u64) * 4;

        // The buffers to reorder (all f32 or u32 arrays indexed by particle ID).
        // Mass is skipped (f16-packed, read-only, small cache footprint).
        let buffers_to_reorder: Vec<&wgpu::Buffer> = vec![
            &self.bufs.pos_x, &self.bufs.pos_y, &self.bufs.pos_z,
            &self.bufs.vel_x, &self.bufs.vel_y, &self.bufs.vel_z,
            &self.bufs.acc_x, &self.bufs.acc_y, &self.bufs.acc_z,
            &self.bufs.density, &self.bufs.pressure, &self.bufs.fluid_type,
        ];

        // For each buffer: scatter to temp, then copy temp back.
        for source_buf in &buffers_to_reorder {
            // Create bind group: params, perm=sorted_indices, source, dest=temp
            let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("reorder_bg"),
                layout: &self.bgl_reorder_g0,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: self.reorder_params_buffer.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: self.bufs.sorted_indices.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 2, resource: source_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 3, resource: self.bufs.reorder_temp.as_entire_binding() },
                ],
            });

            let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("reorder"),
            });

            // Scatter: temp[k] = source[sorted_indices[k]]
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("reorder_scatter"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.pipeline_reorder);
                pass.set_bind_group(0, &bg, &[]);
                pass.dispatch_workgroups(wg, 1, 1);
            }

            // Copy temp back to source
            encoder.copy_buffer_to_buffer(&self.bufs.reorder_temp, 0, source_buf, 0, byte_len);

            self.queue.submit(std::iter::once(encoder.finish()));
        }

        // Wait for all reorder operations to complete
        self.device.poll(wgpu::Maintain::Wait);

        // Also reorder the CPU-side mass_f32 to match
        if !self.bufs.mass_f32.is_empty() {
            // Read sorted_indices from GPU
            let perm = self.read_sorted_indices();
            let old_mass = self.bufs.mass_f32.clone();
            for k in 0..n as usize {
                if k < perm.len() && perm[k] < old_mass.len() {
                    self.bufs.mass_f32[k] = old_mass[perm[k]];
                }
            }
            // Re-upload packed mass
            let mass_packed = buffers::pack_mass_f16(&self.bufs.mass_f32);
            self.queue.write_buffer(
                &self.bufs.mass,
                0,
                bytemuck::cast_slice(&mass_packed),
            );
        }

        // Force grid rebuild since particle indices changed
        self.verlet_displacement = f32::MAX;
        self.steps_since_reorder = 0;
    }

    /// Read sorted_indices from GPU (for CPU-side mass reorder).
    fn read_sorted_indices(&self) -> Vec<usize> {
        let n = self.bufs.n_particles as usize;
        // Use the reorder_temp buffer as a staging buffer for this read
        let byte_len = (n as u64) * 4;

        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("perm_staging"),
            size: byte_len.max(4),
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("read_perm"),
        });
        encoder.copy_buffer_to_buffer(&self.bufs.sorted_indices, 0, &staging, 0, byte_len);
        self.queue.submit(std::iter::once(encoder.finish()));

        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = tx.send(result);
        });
        self.device.poll(wgpu::Maintain::Wait);
        rx.recv().unwrap().unwrap();

        let data = slice.get_mapped_range();
        let u32_data: &[u32] = bytemuck::cast_slice(&data);
        let result: Vec<usize> = u32_data[..n].iter().map(|&v| v as usize).collect();
        drop(data);
        staging.unmap();
        result
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

    /// Density group 2: density(rw), pressure(rw), fluid_type(read), bnd(read), bnd_grid(read)
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
                wgpu::BindGroupEntry { binding: 7, resource: self.bufs.bnd_cell_counts.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 8, resource: self.bufs.bnd_cell_offsets.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 9, resource: self.bufs.bnd_sorted_indices.as_entire_binding() },
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

    /// Forces group 2: density(read), pressure(read), fluid_type(read), bnd(read), bnd_pressure(rw), bnd_grid(read)
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
                wgpu::BindGroupEntry { binding: 8, resource: self.bufs.bnd_cell_counts.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 9, resource: self.bufs.bnd_cell_offsets.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 10, resource: self.bufs.bnd_sorted_indices.as_entire_binding() },
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

        // Periodic particle reorder for cache-friendly memory access.
        // Done before the step so the grid rebuild this triggers is free
        // (grid would be rebuilt anyway this step).
        self.steps_since_reorder += 1;
        if self.steps_since_reorder >= self.reorder_interval {
            self.reorder_particles();
        }

        // Update params buffer with current dt
        self.bufs.update_params(&self.queue, &params);

        // Batch all passes into a single command encoder + submit + poll.
        // wgpu guarantees pass ordering within an encoder, so this is safe
        // and eliminates 3 CPU-GPU sync round-trips vs the old approach.
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("step_batched"),
        });

        // 1-2. Half-kick + drift
        self.encode_integrate(&mut encoder, wg_particles);

        // 3-7. Force computation pipeline (grid + density + boundary + forces)
        self.encode_forces(&mut encoder, &params);

        // 8. Second half-kick
        self.encode_half_kick(&mut encoder, wg_particles);

        self.queue.submit(std::iter::once(encoder.finish()));
        self.device.poll(wgpu::Maintain::Wait);

        // Track estimated max displacement for Verlet neighbor list reuse.
        // Conservative upper bound: no particle exceeds speed of sound.
        self.verlet_displacement += self.speed_of_sound * dt;

        // Mark cache as stale; readback deferred until particles() is called
        self.cache_dirty.set(true);
    }

    fn particles(&self) -> &ParticleArrays {
        self.ensure_cache();
        // SAFETY: ensure_cache() guarantees no outstanding mutable reference.
        // The returned reference borrows `self`, preventing step() from being
        // called while it's alive (step takes &mut self).
        unsafe { &*self.cached_particles.get() }
    }

    fn error_metrics(&self) -> ErrorMetrics {
        let particles = self.particles(); // triggers lazy readback if needed
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
