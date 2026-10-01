//! GPU (Metal/Vulkan via wgpu) implementation of the SPH simulation kernel.
//!
//! `GpuKernel` implements `SimulationKernel` using wgpu compute shaders.
//! On macOS, wgpu compiles to Metal for GPU acceleration.
//!
//! # Architecture
//! - Each simulation step dispatches 4 compute shader passes:
//!   1. Neighbor grid construction (3 sub-passes: count, prefix-sum, scatter)
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
use std::collections::VecDeque;
use std::time::Instant;

use wgpu::util::DeviceExt;
use buffers::{GpuBuffers, GpuSimParams};
use crate::boundary::BoundaryParticles;
use crate::eos;
use crate::particle::{FluidType, ParticleArrays};
use crate::{ErrorMetrics, SimulationKernel, SolverType, StepStats};

/// Per-pass wall-clock timing breakdown of a single GPU simulation step.
#[derive(Debug, Clone, Copy, Default)]
pub struct GpuStepProfile {
    /// Neighbor grid build: count + prefix-sum + scatter + sort (microseconds).
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

/// `{n_particles, pad, pad, pad}` uniform shared by the sort and stats shaders.
#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct CountParams {
    n_particles: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

/// Bind groups built once at init (see `GpuKernel::bg_cache`).
struct BindGroupCache {
    empty_bind_group: wgpu::BindGroup,
    grid_bg0: wgpu::BindGroup,
    grid_bg3: wgpu::BindGroup,
    density_bg0: wgpu::BindGroup,
    density_bg2: wgpu::BindGroup,
    density_forces_bg3: wgpu::BindGroup,
    forces_bg0: wgpu::BindGroup,
    forces_bg1: wgpu::BindGroup,
    forces_bg2: wgpu::BindGroup,
    forces_bg3: wgpu::BindGroup,
    integrate_bg0: wgpu::BindGroup,
    integrate_bg1: wgpu::BindGroup,
    pcisph_predict_bg0: wgpu::BindGroup,
    pcisph_predict_bg1: wgpu::BindGroup,
    pcisph_predict_bg2: wgpu::BindGroup,
    pcisph_bg3: wgpu::BindGroup,
}

/// GPU-accelerated SPH simulation kernel using wgpu compute shaders.
pub struct GpuKernel {
    // wgpu resources
    device: wgpu::Device,
    queue: wgpu::Queue,

    // Compute pipelines
    pipeline_grid_count: wgpu::ComputePipeline,
    pipeline_grid_prefix: wgpu::ComputePipeline,
    pipeline_density: wgpu::ComputePipeline,
    pipeline_boundary_pressure: wgpu::ComputePipeline,
    pipeline_forces: wgpu::ComputePipeline,
    pipeline_half_kick: wgpu::ComputePipeline,
    pipeline_drift: wgpu::ComputePipeline,
    pipeline_xsph: wgpu::ComputePipeline,
    pipeline_sort_scatter: wgpu::ComputePipeline,
    // Sort scatter bind groups: [params + grid, sources (sort_tmp), destinations].
    sort_bgs: [wgpu::BindGroup; 3],

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

    // PCISPH predict shader: groups 0, 1, 2, 3
    bgl_pcisph_predict_g0: wgpu::BindGroupLayout,
    bgl_pcisph_predict_g1: wgpu::BindGroupLayout,
    bgl_pcisph_predict_g2: wgpu::BindGroupLayout,
    bgl_pcisph_g3: wgpu::BindGroupLayout,

    // PCISPH pipelines (7 total)
    pipeline_pcisph_save_init: wgpu::ComputePipeline,
    pipeline_pcisph_predict_pos: wgpu::ComputePipeline,
    pipeline_pcisph_correct_pressure: wgpu::ComputePipeline,
    pipeline_pcisph_update_vel: wgpu::ComputePipeline,
    pipeline_pcisph_final_integrate: wgpu::ComputePipeline,
    pipeline_pcisph_clear_convergence: wgpu::ComputePipeline,
    pipeline_pcisph_pressure_force: wgpu::ComputePipeline,

    // Empty bind group layout for unused group slots in pipeline layouts
    bgl_empty: wgpu::BindGroupLayout,

    // Solver type
    solver_type: SolverType,

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


    // GPU timestamp query resources for precise profiling
    timestamp_query_set: wgpu::QuerySet,
    timestamp_resolve_buf: wgpu::Buffer,
    timestamp_staging_buf: wgpu::Buffer,
    timestamp_period: f32,

    // Cached bind groups (None only during construction).
    bg_cache: Option<BindGroupCache>,

    // On-device max reduction for step_stats() (16-byte readback instead of
    // a full particle readback).
    pipeline_stats: wgpu::ComputePipeline,
    stats_bg: wgpu::BindGroup,
    stats_buf: wgpu::Buffer,
    stats_staging: wgpu::Buffer,
    stats_cache: Cell<Option<StepStats>>,

    // Submissions not yet known to be complete. `step()` keeps at most
    // MAX_IN_FLIGHT queued so the CPU can encode ahead without running away.
    in_flight: VecDeque<wgpu::SubmissionIndex>,
}

/// Neighbor grid cells per support radius (2h). The search covers
/// CELLS_PER_SUPPORT cells on each side, so smaller cells trade more rows
/// for fewer out-of-range candidates.
const CELLS_PER_SUPPORT: u32 = 2;

/// Maximum number of step submissions queued ahead of the GPU in `step()`.
const MAX_IN_FLIGHT: usize = 2;

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
        solver_type: SolverType,
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
        let cell_size = 2.0 * h / CELLS_PER_SUPPORT as f32;
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
            search_cells: CELLS_PER_SUPPORT,
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

        let grid_scan_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("grid_scan"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/grid_scan.wgsl").into()),
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

        let sort_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("sort_scatter"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/sort_scatter.wgsl").into()),
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
        // Group 3: cell_indices, cell_counts, cell_offsets, sorted_indices, cell_fill, scan_tile_sums (all rw)
        let bgl_grid_g3 = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("grid_g3_bgl"),
            entries: &[
                bgl_storage_rw(0), // cell_indices
                bgl_storage_rw(1), // cell_counts
                bgl_storage_rw(2), // cell_offsets
                bgl_storage_rw(3), // sorted_indices
                bgl_storage_rw(4), // cell_fill
                bgl_storage_rw(5), // scan_tile_sums
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

        // -- Sort scatter layouts --
        // Group 0: params, cell_indices(read), cell_offsets(read), cell_fill(rw)
        let bgl_sort_g0 = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("sort_g0_bgl"),
            entries: &[bgl_uniform(0), bgl_storage_ro(1), bgl_storage_ro(2), bgl_storage_rw(3)],
        });
        let sort_src_entries: Vec<_> = (0..buffers::SORTED_ARRAY_COUNT as u32).map(bgl_storage_ro).collect();
        let bgl_sort_src = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("sort_src_bgl"),
            entries: &sort_src_entries,
        });
        let sort_dst_entries: Vec<_> = (0..buffers::SORTED_ARRAY_COUNT as u32).map(bgl_storage_rw).collect();
        let bgl_sort_dst = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("sort_dst_bgl"),
            entries: &sort_dst_entries,
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
            module: &grid_scan_shader,
            entry_point: Some("prefix_sum"),
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

        // Sort scatter pipeline
        let pipeline_sort_scatter = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("sort_scatter"),
            layout: Some(&device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("sort_pl"),
                bind_group_layouts: &[&bgl_sort_g0, &bgl_sort_src, &bgl_sort_dst],
                push_constant_ranges: &[],
            })),
            module: &sort_shader,
            entry_point: Some("scatter"),
            compilation_options: Default::default(),
            cache: None,
        });

        // --- PCISPH shaders and pipelines ---
        let pcisph_predict_src: String = include_str!("shaders/pcisph_predict.wgsl")
            .replace("@workgroup_size(256)", &wg_str);
        let pcisph_predict_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("pcisph_predict"),
            source: wgpu::ShaderSource::Wgsl(pcisph_predict_src.into()),
        });

        let pcisph_pforce_src: String = include_str!("shaders/pcisph_pressure_force.wgsl")
            .replace("@workgroup_size(256)", &wg_str);
        let pcisph_pforce_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("pcisph_pressure_force"),
            source: wgpu::ShaderSource::Wgsl(pcisph_pforce_src.into()),
        });

        // PCISPH predict shader bind group layouts
        // Group 0: params(uniform), pos_x/y/z(rw), mass(read)
        let bgl_pcisph_predict_g0 = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("pcisph_predict_g0_bgl"),
            entries: &[
                bgl_uniform(0),    // params
                bgl_storage_rw(1), // pos_x (rw for predict_positions + final_integrate)
                bgl_storage_rw(2), // pos_y
                bgl_storage_rw(3), // pos_z
                bgl_storage_ro(4), // mass
            ],
        });
        // Group 1: vel_x/y/z(rw), acc_x/y/z(rw)
        let bgl_pcisph_predict_g1 = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("pcisph_predict_g1_bgl"),
            entries: &[
                bgl_storage_rw(0), // vel_x
                bgl_storage_rw(1), // vel_y
                bgl_storage_rw(2), // vel_z
                bgl_storage_rw(3), // acc_x
                bgl_storage_rw(4), // acc_y
                bgl_storage_rw(5), // acc_z
            ],
        });
        // Group 2: density(read), pressure(rw), fluid_type(read)
        let bgl_pcisph_predict_g2 = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("pcisph_predict_g2_bgl"),
            entries: &[
                bgl_storage_ro(0), // density
                bgl_storage_rw(1), // pressure
                bgl_storage_ro(2), // fluid_type
            ],
        });
        // Group 3: PCISPH state (11 bindings)
        let bgl_pcisph_g3 = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("pcisph_g3_bgl"),
            entries: &[
                bgl_storage_rw(0),  // orig_pos_x
                bgl_storage_rw(1),  // orig_pos_y
                bgl_storage_rw(2),  // orig_pos_z
                bgl_storage_rw(3),  // pred_vel_x
                bgl_storage_rw(4),  // pred_vel_y
                bgl_storage_rw(5),  // pred_vel_z
                bgl_storage_rw(6),  // np_acc_x
                bgl_storage_rw(7),  // np_acc_y
                bgl_storage_rw(8),  // np_acc_z
                bgl_storage_rw(9),  // pcisph_delta
                bgl_storage_rw(10), // convergence (atomic u32)
            ],
        });

        // PCISPH predict pipeline layout
        let pl_layout_pcisph_predict = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("pcisph_predict_pl"),
            bind_group_layouts: &[&bgl_pcisph_predict_g0, &bgl_pcisph_predict_g1, &bgl_pcisph_predict_g2, &bgl_pcisph_g3],
            push_constant_ranges: &[],
        });

        // PCISPH pressure force uses the same layout as forces.wgsl
        let pl_layout_pcisph_pforce = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("pcisph_pforce_pl"),
            bind_group_layouts: &[&bgl_forces_g0, &bgl_forces_g1, &bgl_forces_g2, &bgl_forces_g3],
            push_constant_ranges: &[],
        });

        // Create PCISPH pipelines
        let pipeline_pcisph_save_init = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("pcisph_save_init"),
            layout: Some(&pl_layout_pcisph_predict),
            module: &pcisph_predict_shader,
            entry_point: Some("save_and_init_pcisph"),
            compilation_options: Default::default(),
            cache: None,
        });
        let pipeline_pcisph_predict_pos = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("pcisph_predict_pos"),
            layout: Some(&pl_layout_pcisph_predict),
            module: &pcisph_predict_shader,
            entry_point: Some("predict_positions_pcisph"),
            compilation_options: Default::default(),
            cache: None,
        });
        let pipeline_pcisph_correct_pressure = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("pcisph_correct_pressure"),
            layout: Some(&pl_layout_pcisph_predict),
            module: &pcisph_predict_shader,
            entry_point: Some("correct_pressure_pcisph"),
            compilation_options: Default::default(),
            cache: None,
        });
        let pipeline_pcisph_update_vel = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("pcisph_update_vel"),
            layout: Some(&pl_layout_pcisph_predict),
            module: &pcisph_predict_shader,
            entry_point: Some("update_pred_vel_pcisph"),
            compilation_options: Default::default(),
            cache: None,
        });
        let pipeline_pcisph_final_integrate = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("pcisph_final_integrate"),
            layout: Some(&pl_layout_pcisph_predict),
            module: &pcisph_predict_shader,
            entry_point: Some("final_integrate_pcisph"),
            compilation_options: Default::default(),
            cache: None,
        });
        let pipeline_pcisph_clear_convergence = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("pcisph_clear_convergence"),
            layout: Some(&pl_layout_pcisph_predict),
            module: &pcisph_predict_shader,
            entry_point: Some("clear_convergence"),
            compilation_options: Default::default(),
            cache: None,
        });
        let pipeline_pcisph_pressure_force = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("pcisph_pressure_force"),
            layout: Some(&pl_layout_pcisph_pforce),
            module: &pcisph_pforce_shader,
            entry_point: Some("compute_pressure_forces"),
            compilation_options: Default::default(),
            cache: None,
        });

        // Reorder params uniform buffer
        let count_params = CountParams {
            n_particles: particles.len() as u32,
            _pad0: 0, _pad1: 0, _pad2: 0,
        };
        let count_params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("count_params"),
            contents: bytemuck::bytes_of(&count_params),
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

        // --- Stats reduction resources ---
        let stats_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("stats"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/stats.wgsl").into()),
        });
        let bgl_stats = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("stats_bgl"),
            entries: &[
                bgl_uniform(0),
                bgl_storage_ro(1), bgl_storage_ro(2), bgl_storage_ro(3),
                bgl_storage_ro(4), bgl_storage_ro(5), bgl_storage_ro(6),
                bgl_storage_ro(7), bgl_storage_ro(8), bgl_storage_ro(9),
                bgl_storage_ro(10), bgl_storage_ro(11),
                bgl_storage_rw(12),
            ],
        });
        let pipeline_stats = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("stats"),
            layout: Some(&device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("stats_pl"),
                bind_group_layouts: &[&bgl_stats],
                push_constant_ranges: &[],
            })),
            module: &stats_shader,
            entry_point: Some("reduce_stats"),
            compilation_options: Default::default(),
            cache: None,
        });
        let stats_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("stats"),
            size: 16,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let stats_staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("stats_staging"),
            size: 16,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let sort_bgs = {
            let tmp: Vec<&wgpu::Buffer> = bufs.sort_tmp.iter().collect();
            [
                bind_buffers(
                    &device, "sort_g0", &bgl_sort_g0,
                    &[&count_params_buffer, &bufs.cell_indices, &bufs.cell_offsets, &bufs.cell_fill],
                ),
                bind_buffers(&device, "sort_src", &bgl_sort_src, &tmp),
                bind_buffers(&device, "sort_dst", &bgl_sort_dst, &bufs.sorted_arrays()),
            ]
        };
        let stats_bg = {
            let b = &bufs;
            let res = [
                &count_params_buffer, // {n_particles, pad, pad, pad}
                &b.pos_x, &b.pos_y, &b.pos_z,
                &b.vel_x, &b.vel_y, &b.vel_z,
                &b.acc_x, &b.acc_y, &b.acc_z,
                &b.density, &b.fluid_type, &stats_buf,
            ];
            let entries: Vec<_> = res.iter().enumerate()
                .map(|(i, buf)| wgpu::BindGroupEntry { binding: i as u32, resource: buf.as_entire_binding() })
                .collect();
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("stats_bg"), layout: &bgl_stats, entries: &entries,
            })
        };

        let mut kernel = Self {
            device,
            queue,
            pipeline_grid_count,
            pipeline_grid_prefix,
            pipeline_density,
            pipeline_boundary_pressure,
            pipeline_forces,
            pipeline_half_kick,
            pipeline_drift,
            pipeline_xsph,
            pipeline_sort_scatter,
            sort_bgs,
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
            bgl_pcisph_predict_g0,
            bgl_pcisph_predict_g1,
            bgl_pcisph_predict_g2,
            bgl_pcisph_g3,
            pipeline_pcisph_save_init,
            pipeline_pcisph_predict_pos,
            pipeline_pcisph_correct_pressure,
            pipeline_pcisph_update_vel,
            pipeline_pcisph_final_integrate,
            pipeline_pcisph_clear_convergence,
            pipeline_pcisph_pressure_force,
            bgl_empty,
            solver_type,
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
            timestamp_query_set,
            timestamp_resolve_buf,
            timestamp_staging_buf,
            timestamp_period,
            bg_cache: None,
            pipeline_stats,
            stats_bg,
            stats_buf,
            stats_staging,
            stats_cache: Cell::new(None),
            in_flight: VecDeque::new(),
        };
        kernel.bg_cache = Some(kernel.build_bg_cache());
        Ok(kernel)
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
        self.encode_grid_timed(encoder, None);
    }

    /// Build the neighbor grid and move all persistent particle arrays into
    /// cell order (sorted_indices stays the identity), so neighbors are
    /// contiguous in memory. Optional timestamp indices bracket
    /// the whole sequence for `step_profiled`.
    fn encode_grid_timed(&self, encoder: &mut wgpu::CommandEncoder, ts: Option<(u32, u32)>) {
        let n_particles = self.bufs.n_particles;
        let total_cells = self.bufs.total_cells;

        let wg_grid_particles = dispatch_size(n_particles, 256);
        let wg_scan_tiles = dispatch_size(total_cells, buffers::GRID_SCAN_TILE);

        let grid_bg0 = self.create_grid_bg0();
        let grid_bg3 = self.create_grid_bg3();
        let empty_bg = self.create_empty_bind_group();

        let ts_writes = |begin: Option<u32>, end: Option<u32>| {
            ts.map(|_| wgpu::ComputePassTimestampWrites {
                query_set: &self.timestamp_query_set,
                beginning_of_pass_write_index: begin,
                end_of_pass_write_index: end,
            })
        };

        // Snapshot the arrays to be sorted (the grid passes don't modify them).
        let byte_len = n_particles as u64 * 4;
        for (src, tmp) in self.bufs.sorted_arrays().iter().zip(&self.bufs.sort_tmp) {
            encoder.copy_buffer_to_buffer(src, 0, tmp, 0, byte_len);
        }
        encoder.clear_buffer(&self.bufs.scan_tile_sums, 0, None);

        // count, prefix sum (one workgroup per scan tile), then scatter the
        // snapshot into cell order. No clear pass: the scatter counts
        // cell_fill back to zero. All dispatches share one compute pass
        // (WebGPU orders dispatches within a pass and makes each one's writes
        // visible to the next), saving per-pass encoder overhead.
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("grid_build"),
            timestamp_writes: ts_writes(ts.map(|t| t.0), ts.map(|t| t.1)),
        });
        pass.set_bind_group(0, &grid_bg0, &[]);
        pass.set_bind_group(1, &empty_bg, &[]);
        pass.set_bind_group(2, &empty_bg, &[]);
        pass.set_bind_group(3, &grid_bg3, &[]);
        for (pipeline, groups) in [
            (&self.pipeline_grid_count, wg_grid_particles),
            (&self.pipeline_grid_prefix, wg_scan_tiles),
        ] {
            pass.set_pipeline(pipeline);
            pass.dispatch_workgroups(groups, 1, 1);
        }
        pass.set_pipeline(&self.pipeline_sort_scatter);
        for (g, bg) in self.sort_bgs.iter().enumerate() {
            pass.set_bind_group(g as u32, bg, &[]);
        }
        pass.dispatch_workgroups(wg_grid_particles, 1, 1);
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
        self.stats_cache.set(None);
    }

    /// Wait for all submitted GPU work to complete.
    pub fn sync(&self) {
        self.device.poll(wgpu::Maintain::Wait);
    }

    /// Test hook: rebuild the neighbor grid (and sort particles into cell
    /// order) right now, then read back the grid buffers.
    #[doc(hidden)]
    pub fn debug_rebuild_grid(&mut self) -> GridDebug {
        let params = self.make_params(0.0);
        self.bufs.update_params(&self.queue, &params);
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("debug_grid"),
        });
        self.encode_grid(&mut encoder);
        self.queue.submit(std::iter::once(encoder.finish()));
        self.cache_dirty.set(true);
        let cells = self.bufs.total_cells as usize;
        let tiles = cells.div_ceil(buffers::GRID_SCAN_TILE as usize);
        let read = |b: &wgpu::Buffer, n: usize| debug_read_u32(&self.device, &self.queue, b, n);
        GridDebug {
            cell_counts: read(&self.bufs.cell_counts, cells),
            cell_offsets: read(&self.bufs.cell_offsets, cells),
            cell_fill: read(&self.bufs.cell_fill, cells),
            scan_tile_sums: read(&self.bufs.scan_tile_sums, tiles),
            grid_dims: self.grid_dims,
            cell_size: params.cell_size,
            domain_min: self.domain_min,
        }
    }

    /// Test hook: boundary particle positions and mirrored pressures, in the
    /// GPU's (cell-sorted) boundary order, as of the last step.
    #[doc(hidden)]
    pub fn debug_boundary_pressures(&self) -> [Vec<f32>; 4] {
        self.sync();
        let n = self.bufs.n_boundary as usize;
        let read = |b: &wgpu::Buffer| -> Vec<f32> {
            debug_read_u32(&self.device, &self.queue, b, n).into_iter().map(f32::from_bits).collect()
        };
        [read(&self.bufs.bnd_x), read(&self.bufs.bnd_y), read(&self.bufs.bnd_z), read(&self.bufs.bnd_pressure)]
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
        let n_boundary = self.bufs.n_boundary;
        let wg_particles = dispatch_size(n_total, wg);
        let wg_boundary = dispatch_size(n_boundary.max(1), wg);

        // Create all bind groups upfront
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

        // TS 2-3: Grid build + sort
        self.encode_grid_timed(&mut encoder, Some((2, 3)));

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
        self.stats_cache.set(None);
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

    // ---- Bind group creation helpers ----
    //
    // Every buffer lives for the kernel's lifetime, so bind groups are built
    // once and cloned (a refcount bump) instead of recreated on every step.

    fn bg_cache(&self) -> &BindGroupCache {
        self.bg_cache.as_ref().expect("bind group cache built in new()")
    }

    fn build_bg_cache(&self) -> BindGroupCache {
        BindGroupCache {
            empty_bind_group: self.build_empty_bind_group(),
            grid_bg0: self.build_grid_bg0(),
            grid_bg3: self.build_grid_bg3(),
            density_bg0: self.build_density_bg0(),
            density_bg2: self.build_density_bg2(),
            density_forces_bg3: self.build_density_forces_bg3(),
            forces_bg0: self.build_forces_bg0(),
            forces_bg1: self.build_forces_bg1(),
            forces_bg2: self.build_forces_bg2(),
            forces_bg3: self.build_forces_bg3(),
            integrate_bg0: self.build_integrate_bg0(),
            integrate_bg1: self.build_integrate_bg1(),
            pcisph_predict_bg0: self.build_pcisph_predict_bg0(),
            pcisph_predict_bg1: self.build_pcisph_predict_bg1(),
            pcisph_predict_bg2: self.build_pcisph_predict_bg2(),
            pcisph_bg3: self.build_pcisph_bg3(),
        }
    }

    fn create_empty_bind_group(&self) -> wgpu::BindGroup { self.bg_cache().empty_bind_group.clone() }
    fn create_grid_bg0(&self) -> wgpu::BindGroup { self.bg_cache().grid_bg0.clone() }
    fn create_grid_bg3(&self) -> wgpu::BindGroup { self.bg_cache().grid_bg3.clone() }
    fn create_density_bg0(&self) -> wgpu::BindGroup { self.bg_cache().density_bg0.clone() }
    fn create_density_bg2(&self) -> wgpu::BindGroup { self.bg_cache().density_bg2.clone() }
    fn create_density_forces_bg3(&self) -> wgpu::BindGroup { self.bg_cache().density_forces_bg3.clone() }
    fn create_forces_bg0(&self) -> wgpu::BindGroup { self.bg_cache().forces_bg0.clone() }
    fn create_forces_bg1(&self) -> wgpu::BindGroup { self.bg_cache().forces_bg1.clone() }
    fn create_forces_bg2(&self) -> wgpu::BindGroup { self.bg_cache().forces_bg2.clone() }
    fn create_forces_bg3(&self) -> wgpu::BindGroup { self.bg_cache().forces_bg3.clone() }
    fn create_integrate_bg0(&self) -> wgpu::BindGroup { self.bg_cache().integrate_bg0.clone() }
    fn create_integrate_bg1(&self) -> wgpu::BindGroup { self.bg_cache().integrate_bg1.clone() }
    fn create_pcisph_predict_bg0(&self) -> wgpu::BindGroup { self.bg_cache().pcisph_predict_bg0.clone() }
    fn create_pcisph_predict_bg1(&self) -> wgpu::BindGroup { self.bg_cache().pcisph_predict_bg1.clone() }
    fn create_pcisph_predict_bg2(&self) -> wgpu::BindGroup { self.bg_cache().pcisph_predict_bg2.clone() }
    fn create_pcisph_bg3(&self) -> wgpu::BindGroup { self.bg_cache().pcisph_bg3.clone() }

    /// Create an empty bind group for unused group slots.
    fn build_empty_bind_group(&self) -> wgpu::BindGroup {
        self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("empty_bg"),
            layout: &self.bgl_empty,
            entries: &[],
        })
    }

    // -- Grid shader bind groups --

    /// Grid group 0: params + pos_x/y/z (read)
    fn build_grid_bg0(&self) -> wgpu::BindGroup {
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

    /// Grid group 3: cell_indices, cell_counts, cell_offsets, sorted_indices, cell_fill, scan_tile_sums (all rw)
    fn build_grid_bg3(&self) -> wgpu::BindGroup {
        self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("grid_bg3"),
            layout: &self.bgl_grid_g3,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: self.bufs.cell_indices.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: self.bufs.cell_counts.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: self.bufs.cell_offsets.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: self.bufs.sorted_indices.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: self.bufs.cell_fill.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: self.bufs.scan_tile_sums.as_entire_binding() },
            ],
        })
    }

    // -- Density shader bind groups --

    /// Density group 0: params + pos_x/y/z (read) + mass (read)
    fn build_density_bg0(&self) -> wgpu::BindGroup {
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
    fn build_density_bg2(&self) -> wgpu::BindGroup {
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
    fn build_density_forces_bg3(&self) -> wgpu::BindGroup {
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
    fn build_forces_bg0(&self) -> wgpu::BindGroup {
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
    fn build_forces_bg1(&self) -> wgpu::BindGroup {
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
    fn build_forces_bg2(&self) -> wgpu::BindGroup {
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
    fn build_forces_bg3(&self) -> wgpu::BindGroup {
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
    fn build_integrate_bg0(&self) -> wgpu::BindGroup {
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
    fn build_integrate_bg1(&self) -> wgpu::BindGroup {
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

    // -- PCISPH predict shader bind groups --

    /// PCISPH predict group 0: params + pos_x/y/z (rw) + mass (read)
    fn build_pcisph_predict_bg0(&self) -> wgpu::BindGroup {
        self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("pcisph_predict_bg0"),
            layout: &self.bgl_pcisph_predict_g0,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: self.bufs.params_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: self.bufs.pos_x.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: self.bufs.pos_y.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: self.bufs.pos_z.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: self.bufs.mass.as_entire_binding() },
            ],
        })
    }

    /// PCISPH predict group 1: vel_x/y/z (rw), acc_x/y/z (rw)
    fn build_pcisph_predict_bg1(&self) -> wgpu::BindGroup {
        self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("pcisph_predict_bg1"),
            layout: &self.bgl_pcisph_predict_g1,
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

    /// PCISPH predict group 2: density(read), pressure(rw), fluid_type(read)
    fn build_pcisph_predict_bg2(&self) -> wgpu::BindGroup {
        self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("pcisph_predict_bg2"),
            layout: &self.bgl_pcisph_predict_g2,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: self.bufs.density.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: self.bufs.pressure.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: self.bufs.fluid_type.as_entire_binding() },
            ],
        })
    }

    /// PCISPH group 3: PCISPH state buffers (11 bindings)
    fn build_pcisph_bg3(&self) -> wgpu::BindGroup {
        self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("pcisph_bg3"),
            layout: &self.bgl_pcisph_g3,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: self.bufs.pcisph_orig_pos_x.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: self.bufs.pcisph_orig_pos_y.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: self.bufs.pcisph_orig_pos_z.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: self.bufs.pcisph_pred_vel_x.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: self.bufs.pcisph_pred_vel_y.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: self.bufs.pcisph_pred_vel_z.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 6, resource: self.bufs.pcisph_np_acc_x.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 7, resource: self.bufs.pcisph_np_acc_y.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 8, resource: self.bufs.pcisph_np_acc_z.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 9, resource: self.bufs.pcisph_delta.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 10, resource: self.bufs.pcisph_convergence.as_entire_binding() },
            ],
        })
    }

    /// Upload per-particle delta scaling factors from CPU.
    fn upload_pcisph_delta(&self, delta: &[f32]) {
        self.queue.write_buffer(
            &self.bufs.pcisph_delta,
            0,
            bytemuck::cast_slice(delta),
        );
    }

    /// Compute PCISPH per-particle delta on CPU and upload to GPU.
    /// Called once at initialization.
    fn upload_pcisph_delta_from_cpu(&self) {
        let particles = self.bufs.readback_particles(&self.device, &self.queue);
        let n = particles.len();
        if n == 0 {
            return;
        }

        // Build neighbor grid on CPU
        let mut grid = crate::neighbor::NeighborGrid::new(
            self.h * 2.0, // cell_size = support_radius
            self.domain_min,
            self.domain_max,
        );
        grid.update(&particles.x, &particles.y, &particles.z);

        // Compute dt-independent delta (using dt=1.0). The shader applies the
        // actual dt scaling: effective_delta = delta_base / (dt * dt).
        // This avoids recomputing deltas when dt changes (adaptive timestep).
        let deltas = crate::sph::compute_pcisph_per_particle_delta(
            &particles, &grid, self.h, 1.0,
        );

        self.upload_pcisph_delta(&deltas);
    }

    /// Execute a full PCISPH step, managing its own command encoder lifecycle.
    ///
    /// PCISPH step structure:
    /// 1. Non-pressure forces (reuse WCSPH density + forces pipeline)
    /// 2. Save state + init prediction
    /// 3. Correction loop: predict positions → density → correct pressure → pressure forces → update vel
    /// 4. Final integration
    ///
    /// Each iteration batch gets its own encoder since convergence checks require
    /// GPU→CPU readback (submit + poll + map) between iterations.
    fn step_pcisph(&mut self, params: &GpuSimParams) {
        let n_particles = self.bufs.n_particles;
        let n_boundary = self.bufs.n_boundary;
        let wg = self.workgroup_size;
        let wg_particles = dispatch_size(n_particles, wg);
        let wg_boundary = dispatch_size(n_boundary.max(1), wg);

        let min_iterations = 3u32;
        let max_iterations = 10u32;

        // --- Phase A: Non-pressure forces + save/init ---
        // For PCISPH, we compute density WITHOUT Tait EOS (pass_index=1) so pressure
        // stays at 0. The forces shader then produces only non-pressure forces
        // (viscous + gravity + boundary repulsion) since pressure gradient = 0.
        //
        // IMPORTANT: Each encoder submission must use a single params state because
        // queue.write_buffer calls are all applied BEFORE the command buffer executes.
        // Multiple writes to the same buffer in one submit → only the last value is seen.

        // Sub-phase A1: Clear pressure + density-only (pass_index=1)
        {
            let mut density_params = *params;
            density_params.pass_index = 1;
            self.bufs.update_params(&self.queue, &density_params);

            let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("pcisph_density_init"),
            });
            encoder.clear_buffer(&self.bufs.pressure, 0, None);

            let density_bg0 = self.create_density_bg0();
            let density_bg2 = self.create_density_bg2();
            let density_bg3 = self.create_density_forces_bg3();
            let empty_bg = self.create_empty_bind_group();

            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("pcisph_density"), timestamp_writes: None,
                });
                pass.set_pipeline(&self.pipeline_density);
                pass.set_bind_group(0, &density_bg0, &[]); pass.set_bind_group(1, &empty_bg, &[]);
                pass.set_bind_group(2, &density_bg2, &[]); pass.set_bind_group(3, &density_bg3, &[]);
                pass.dispatch_workgroups(wg_particles, 1, 1);
            }

            self.queue.submit(std::iter::once(encoder.finish()));
            self.device.poll(wgpu::Maintain::Wait);
        }

        // Sub-phase A2: Forces (pass_index=0, pressure=0) + save/init
        {
            self.bufs.update_params(&self.queue, params);

            let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("pcisph_forces_init"),
            });

            let forces_bg0 = self.create_forces_bg0();
            let forces_bg1 = self.create_forces_bg1();
            let forces_bg2 = self.create_forces_bg2();
            let forces_bg3 = self.create_forces_bg3();

            // Boundary pressure mirroring (pressure=0 → boundary pressure=0)
            if n_boundary > 0 {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("pcisph_bnd_init"), timestamp_writes: None,
                });
                pass.set_pipeline(&self.pipeline_boundary_pressure);
                pass.set_bind_group(0, &forces_bg0, &[]); pass.set_bind_group(1, &forces_bg1, &[]);
                pass.set_bind_group(2, &forces_bg2, &[]); pass.set_bind_group(3, &forces_bg3, &[]);
                pass.dispatch_workgroups(wg_boundary, 1, 1);
            }

            // Forces: only non-pressure forces (viscous + gravity + boundary repulsion)
            // since pressure=0 → pressure gradient term = 0
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("pcisph_forces"), timestamp_writes: None,
                });
                pass.set_pipeline(&self.pipeline_forces);
                pass.set_bind_group(0, &forces_bg0, &[]); pass.set_bind_group(1, &forces_bg1, &[]);
                pass.set_bind_group(2, &forces_bg2, &[]); pass.set_bind_group(3, &forces_bg3, &[]);
                pass.dispatch_workgroups(wg_particles, 1, 1);
            }

            // Save state and initialize prediction
            let pcisph_bg0 = self.create_pcisph_predict_bg0();
            let pcisph_bg1 = self.create_pcisph_predict_bg1();
            let pcisph_bg2 = self.create_pcisph_predict_bg2();
            let pcisph_bg3 = self.create_pcisph_bg3();

            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("pcisph_save_init"), timestamp_writes: None,
                });
                pass.set_pipeline(&self.pipeline_pcisph_save_init);
                pass.set_bind_group(0, &pcisph_bg0, &[]);
                pass.set_bind_group(1, &pcisph_bg1, &[]);
                pass.set_bind_group(2, &pcisph_bg2, &[]);
                pass.set_bind_group(3, &pcisph_bg3, &[]);
                pass.dispatch_workgroups(wg_particles, 1, 1);
            }

            self.queue.submit(std::iter::once(encoder.finish()));
            self.device.poll(wgpu::Maintain::Wait);
        }

        // --- Phase B: Correction iterations ---
        // Use pass_index=1 for all iteration passes (only density shader checks it;
        // other shaders ignore it). This avoids the queue.write_buffer race condition
        // where multiple writes to the same buffer before submit cause the last one to win.
        let mut iter_params = *params;
        iter_params.pass_index = 1;

        for iter in 0..max_iterations {
            self.bufs.update_params(&self.queue, &iter_params);

            let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("pcisph_iter"),
            });

            let pcisph_bg0 = self.create_pcisph_predict_bg0();
            let pcisph_bg1 = self.create_pcisph_predict_bg1();
            let pcisph_bg2 = self.create_pcisph_predict_bg2();
            let pcisph_bg3 = self.create_pcisph_bg3();
            let forces_bg0 = self.create_forces_bg0();
            let forces_bg1 = self.create_forces_bg1();
            let forces_bg2 = self.create_forces_bg2();
            let forces_bg3 = self.create_forces_bg3();
            let density_bg0 = self.create_density_bg0();
            let density_bg2 = self.create_density_bg2();
            let density_bg3 = self.create_density_forces_bg3();
            let empty_bg = self.create_empty_bind_group();

            // Clear convergence
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("pcisph_clear_conv"), timestamp_writes: None,
                });
                pass.set_pipeline(&self.pipeline_pcisph_clear_convergence);
                pass.set_bind_group(0, &pcisph_bg0, &[]);
                pass.set_bind_group(1, &pcisph_bg1, &[]);
                pass.set_bind_group(2, &pcisph_bg2, &[]);
                pass.set_bind_group(3, &pcisph_bg3, &[]);
                pass.dispatch_workgroups(1, 1, 1);
            }

            // Predict positions
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("pcisph_predict_pos"), timestamp_writes: None,
                });
                pass.set_pipeline(&self.pipeline_pcisph_predict_pos);
                pass.set_bind_group(0, &pcisph_bg0, &[]);
                pass.set_bind_group(1, &pcisph_bg1, &[]);
                pass.set_bind_group(2, &pcisph_bg2, &[]);
                pass.set_bind_group(3, &pcisph_bg3, &[]);
                pass.dispatch_workgroups(wg_particles, 1, 1);
            }

            // Density summation (density-only mode: pass_index=1, set above)
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("pcisph_density"), timestamp_writes: None,
                });
                pass.set_pipeline(&self.pipeline_density);
                pass.set_bind_group(0, &density_bg0, &[]);
                pass.set_bind_group(1, &empty_bg, &[]);
                pass.set_bind_group(2, &density_bg2, &[]);
                pass.set_bind_group(3, &density_bg3, &[]);
                pass.dispatch_workgroups(wg_particles, 1, 1);
            }

            // Correct pressure from density error
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("pcisph_correct_pressure"), timestamp_writes: None,
                });
                pass.set_pipeline(&self.pipeline_pcisph_correct_pressure);
                pass.set_bind_group(0, &pcisph_bg0, &[]);
                pass.set_bind_group(1, &pcisph_bg1, &[]);
                pass.set_bind_group(2, &pcisph_bg2, &[]);
                pass.set_bind_group(3, &pcisph_bg3, &[]);
                pass.dispatch_workgroups(wg_particles, 1, 1);
            }

            // Boundary pressure mirroring
            if n_boundary > 0 {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("pcisph_bnd_pressure"), timestamp_writes: None,
                });
                pass.set_pipeline(&self.pipeline_boundary_pressure);
                pass.set_bind_group(0, &forces_bg0, &[]); pass.set_bind_group(1, &forces_bg1, &[]);
                pass.set_bind_group(2, &forces_bg2, &[]); pass.set_bind_group(3, &forces_bg3, &[]);
                pass.dispatch_workgroups(wg_boundary, 1, 1);
            }

            // Pressure-only forces
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("pcisph_pressure_force"), timestamp_writes: None,
                });
                pass.set_pipeline(&self.pipeline_pcisph_pressure_force);
                pass.set_bind_group(0, &forces_bg0, &[]); pass.set_bind_group(1, &forces_bg1, &[]);
                pass.set_bind_group(2, &forces_bg2, &[]); pass.set_bind_group(3, &forces_bg3, &[]);
                pass.dispatch_workgroups(wg_particles, 1, 1);
            }

            // Update predicted velocity
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("pcisph_update_vel"), timestamp_writes: None,
                });
                pass.set_pipeline(&self.pipeline_pcisph_update_vel);
                pass.set_bind_group(0, &pcisph_bg0, &[]);
                pass.set_bind_group(1, &pcisph_bg1, &[]);
                pass.set_bind_group(2, &pcisph_bg2, &[]);
                pass.set_bind_group(3, &pcisph_bg3, &[]);
                pass.dispatch_workgroups(wg_particles, 1, 1);
            }

            self.queue.submit(std::iter::once(encoder.finish()));
            self.device.poll(wgpu::Maintain::Wait);

            // Check convergence after minimum iterations
            if iter >= min_iterations - 1 {
                let conv = self.bufs.readback_convergence(&self.device, &self.queue);
                let sum_error = conv[0] as f64 / 1_000_000.0;
                let count = conv[1];
                let mean_error = if count > 0 { sum_error / count as f64 } else { 0.0 };
                if mean_error < 0.01 {
                    break;
                }
            }
        }

        // --- Phase C: Final integration ---
        {
            let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("pcisph_final"),
            });

            let pcisph_bg0 = self.create_pcisph_predict_bg0();
            let pcisph_bg1 = self.create_pcisph_predict_bg1();
            let pcisph_bg2 = self.create_pcisph_predict_bg2();
            let pcisph_bg3 = self.create_pcisph_bg3();

            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("pcisph_final_integrate"), timestamp_writes: None,
                });
                pass.set_pipeline(&self.pipeline_pcisph_final_integrate);
                pass.set_bind_group(0, &pcisph_bg0, &[]);
                pass.set_bind_group(1, &pcisph_bg1, &[]);
                pass.set_bind_group(2, &pcisph_bg2, &[]);
                pass.set_bind_group(3, &pcisph_bg3, &[]);
                pass.dispatch_workgroups(wg_particles, 1, 1);
            }

            self.queue.submit(std::iter::once(encoder.finish()));
            self.device.poll(wgpu::Maintain::Wait);
        }
    }

    /// Record a submission and block until at most MAX_IN_FLIGHT remain queued.
    fn track_submission(&mut self, idx: wgpu::SubmissionIndex) {
        self.in_flight.push_back(idx);
        while self.in_flight.len() > MAX_IN_FLIGHT {
            let oldest = self.in_flight.pop_front().unwrap();
            self.device.poll(wgpu::Maintain::WaitForSubmissionIndex(oldest));
        }
    }

    /// Reduce max |v|, |a|, density deviation and a non-finite flag on the GPU.
    fn compute_stats_gpu(&self) -> StepStats {
        let n = self.bufs.n_particles;
        if n == 0 {
            return StepStats::default();
        }
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("stats"),
        });
        encoder.clear_buffer(&self.stats_buf, 0, None);
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("stats"), timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline_stats);
            pass.set_bind_group(0, &self.stats_bg, &[]);
            pass.dispatch_workgroups(dispatch_size(n, 256), 1, 1);
        }
        encoder.copy_buffer_to_buffer(&self.stats_buf, 0, &self.stats_staging, 0, 16);
        self.queue.submit(std::iter::once(encoder.finish()));

        let slice = self.stats_staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| { let _ = tx.send(r); });
        self.device.poll(wgpu::Maintain::Wait);
        rx.recv().unwrap().unwrap();
        let bits: [u32; 4] = {
            let data = slice.get_mapped_range();
            let b: &[u32] = bytemuck::cast_slice(&data);
            [b[0], b[1], b[2], b[3]]
        };
        self.stats_staging.unmap();
        StepStats {
            max_speed: f32::from_bits(bits[0]).sqrt(),
            max_accel: f32::from_bits(bits[1]).sqrt(),
            max_density_variation: f32::from_bits(bits[2]),
            non_finite: bits[3] != 0,
        }
    }

    fn make_params(&self, dt: f32) -> GpuSimParams {
        let cell_size = 2.0 * self.h / CELLS_PER_SUPPORT as f32;
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
            search_cells: CELLS_PER_SUPPORT,
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
            if self.solver_type == SolverType::Pcisph {
                // PCISPH: upload delta scaling factors and build grid only.
                // step_pcisph handles its own force computation.
                self.upload_pcisph_delta_from_cpu();
                let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("pcisph_bootstrap_grid"),
                });
                self.encode_grid(&mut encoder);
                self.queue.submit(std::iter::once(encoder.finish()));
                self.device.poll(wgpu::Maintain::Wait);
                self.verlet_displacement = 0.0;
            } else {
                self.compute_forces_gpu(&params);
            }
            self.needs_init = false;
        }

        // Update params buffer with current dt
        self.bufs.update_params(&self.queue, &params);

        match self.solver_type {
            SolverType::Wcsph => {
                let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("step_wcsph"),
                });
                // WCSPH: Velocity Verlet (KDK) integration
                self.encode_integrate(&mut encoder, wg_particles);
                self.encode_forces(&mut encoder, &params);
                self.encode_half_kick(&mut encoder, wg_particles);
                // No wait: the CPU encodes the next step while the GPU runs
                // this one. Readbacks (particles(), step_stats()) wait for
                // completion themselves.
                let idx = self.queue.submit(std::iter::once(encoder.finish()));
                self.track_submission(idx);
            }
            SolverType::Pcisph => {
                // Grid rebuild if needed (PCISPH manages its own encoders)
                let needs_grid = self.verlet_displacement >= self.verlet_skin * 0.5;
                if needs_grid {
                    let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("pcisph_grid"),
                    });
                    self.encode_grid(&mut encoder);
                    self.queue.submit(std::iter::once(encoder.finish()));
                    self.device.poll(wgpu::Maintain::Wait);
                    self.verlet_displacement = 0.0;
                }
                // PCISPH: non-pressure forces + iterative correction + final integrate
                self.step_pcisph(&params);
            }
        }

        // Track estimated max displacement for Verlet neighbor list reuse.
        self.verlet_displacement += self.speed_of_sound * dt;

        // Mark cache as stale; readback deferred until particles() is called
        self.cache_dirty.set(true);
        self.stats_cache.set(None);
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

    fn solver_type(&self) -> SolverType {
        self.solver_type
    }

    fn step_stats(&self) -> StepStats {
        if let Some(s) = self.stats_cache.get() {
            return s;
        }
        let s = self.compute_stats_gpu();
        self.stats_cache.set(Some(s));
        s
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

/// Grid buffers read back by [`GpuKernel::debug_rebuild_grid`].
#[doc(hidden)]
#[derive(Debug, Clone)]
pub struct GridDebug {
    pub cell_counts: Vec<u32>,
    pub cell_offsets: Vec<u32>,
    pub cell_fill: Vec<u32>,
    pub scan_tile_sums: Vec<u32>,
    pub grid_dims: [u32; 3],
    pub cell_size: f32,
    pub domain_min: [f32; 3],
}

/// Blocking readback of the first `count` u32 words of a storage buffer (tests only).
fn debug_read_u32(device: &wgpu::Device, queue: &wgpu::Queue, buf: &wgpu::Buffer, count: usize) -> Vec<u32> {
    if count == 0 {
        return Vec::new();
    }
    let bytes = count as u64 * 4;
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("debug_staging"),
        size: bytes,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("debug_read") });
    encoder.copy_buffer_to_buffer(buf, 0, &staging, 0, bytes);
    queue.submit(std::iter::once(encoder.finish()));
    let slice = staging.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| {
        let _ = tx.send(r);
    });
    device.poll(wgpu::Maintain::Wait);
    rx.recv().unwrap().unwrap();
    let out = bytemuck::cast_slice(&slice.get_mapped_range()).to_vec();
    staging.unmap();
    out
}

/// Bind group whose binding `i` is the whole of `bufs[i]`.
fn bind_buffers(
    device: &wgpu::Device,
    label: &str,
    layout: &wgpu::BindGroupLayout,
    bufs: &[&wgpu::Buffer],
) -> wgpu::BindGroup {
    let entries: Vec<_> = bufs
        .iter()
        .enumerate()
        .map(|(i, b)| wgpu::BindGroupEntry { binding: i as u32, resource: b.as_entire_binding() })
        .collect();
    device.create_bind_group(&wgpu::BindGroupDescriptor { label: Some(label), layout, entries: &entries })
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
