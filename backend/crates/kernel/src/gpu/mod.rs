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

    // Bind group layouts
    bg_layout_grid: wgpu::BindGroupLayout,
    bg_layout_density: wgpu::BindGroupLayout,
    bg_layout_forces: wgpu::BindGroupLayout,
    bg_layout_integrate: wgpu::BindGroupLayout,

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

        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("sph_gpu_device"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
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
            pass: 0,
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
        let bg_layout_grid = create_grid_bind_group_layout(&device);
        let bg_layout_density = create_density_bind_group_layout(&device);
        let bg_layout_forces = create_forces_bind_group_layout(&device);
        let bg_layout_integrate = create_integrate_bind_group_layout(&device);

        // --- Pipeline layouts ---
        let pl_layout_grid = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("grid_pl"),
            bind_group_layouts: &[&bg_layout_grid],
            push_constant_ranges: &[],
        });
        let pl_layout_density = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("density_pl"),
            bind_group_layouts: &[&bg_layout_density],
            push_constant_ranges: &[],
        });
        let pl_layout_forces = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("forces_pl"),
            bind_group_layouts: &[&bg_layout_forces],
            push_constant_ranges: &[],
        });
        let pl_layout_integrate = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("integrate_pl"),
            bind_group_layouts: &[&bg_layout_integrate],
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
            bg_layout_grid,
            bg_layout_density,
            bg_layout_forces,
            bg_layout_integrate,
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

        // Create bind groups
        let bg_grid = self.create_grid_bind_group();
        let bg_density = self.create_density_bind_group();
        let bg_forces = self.create_forces_bind_group();

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
            pass.set_bind_group(0, &bg_grid, &[]);
            pass.dispatch_workgroups(wg_cells, 1, 1);
        }

        // 2. Neighbor grid: count
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("grid_count"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline_grid_count);
            pass.set_bind_group(0, &bg_grid, &[]);
            pass.dispatch_workgroups(wg_particles, 1, 1);
        }

        // 3. Neighbor grid: prefix sum (single thread)
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("grid_prefix"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline_grid_prefix);
            pass.set_bind_group(0, &bg_grid, &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }

        // 4. Neighbor grid: scatter
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("grid_scatter"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline_grid_scatter);
            pass.set_bind_group(0, &bg_grid, &[]);
            pass.dispatch_workgroups(wg_particles, 1, 1);
        }

        // 5. Density summation + EOS pressure
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("density"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline_density);
            pass.set_bind_group(0, &bg_density, &[]);
            pass.dispatch_workgroups(wg_particles, 1, 1);
        }

        // 6. Boundary pressure mirroring
        if n_boundary > 0 {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("boundary_pressure"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline_boundary_pressure);
            pass.set_bind_group(0, &bg_forces, &[]);
            pass.dispatch_workgroups(wg_boundary, 1, 1);
        }

        // 7. All forces (pressure + viscous + gravity + boundary repulsive)
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("forces"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline_forces);
            pass.set_bind_group(0, &bg_forces, &[]);
            pass.dispatch_workgroups(wg_particles, 1, 1);
        }

        self.queue.submit(std::iter::once(encoder.finish()));
        self.device.poll(wgpu::Maintain::Wait);
    }

    /// Refresh the cached CPU-side particle data from GPU.
    fn refresh_cache(&mut self) {
        self.cached_particles = self.bufs.readback_particles(&self.device, &self.queue);
    }

    /// Create bind group for the neighbor grid shader.
    fn create_grid_bind_group(&self) -> wgpu::BindGroup {
        self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("grid_bg"),
            layout: &self.bg_layout_grid,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: self.bufs.params_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: self.bufs.pos_x.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: self.bufs.pos_y.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: self.bufs.pos_z.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: self.bufs.cell_indices.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: self.bufs.cell_counts.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 6, resource: self.bufs.cell_offsets.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 7, resource: self.bufs.sorted_indices.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 8, resource: self.bufs.write_heads.as_entire_binding() },
            ],
        })
    }

    /// Create bind group for the density shader.
    fn create_density_bind_group(&self) -> wgpu::BindGroup {
        self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("density_bg"),
            layout: &self.bg_layout_density,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: self.bufs.params_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: self.bufs.pos_x.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: self.bufs.pos_y.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: self.bufs.pos_z.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: self.bufs.mass.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: self.bufs.density.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 6, resource: self.bufs.pressure.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 7, resource: self.bufs.fluid_type.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 8, resource: self.bufs.bnd_x.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 9, resource: self.bufs.bnd_y.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 10, resource: self.bufs.bnd_z.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 11, resource: self.bufs.bnd_mass.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 12, resource: self.bufs.cell_offsets.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 13, resource: self.bufs.cell_counts.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 14, resource: self.bufs.sorted_indices.as_entire_binding() },
            ],
        })
    }

    /// Create bind group for the forces shader.
    fn create_forces_bind_group(&self) -> wgpu::BindGroup {
        self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("forces_bg"),
            layout: &self.bg_layout_forces,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: self.bufs.params_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: self.bufs.pos_x.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: self.bufs.pos_y.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: self.bufs.pos_z.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: self.bufs.vel_x.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: self.bufs.vel_y.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 6, resource: self.bufs.vel_z.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 7, resource: self.bufs.mass.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 8, resource: self.bufs.density.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 9, resource: self.bufs.pressure.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 10, resource: self.bufs.fluid_type.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 11, resource: self.bufs.acc_x.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 12, resource: self.bufs.acc_y.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 13, resource: self.bufs.acc_z.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 14, resource: self.bufs.bnd_x.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 15, resource: self.bufs.bnd_y.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 16, resource: self.bufs.bnd_z.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 17, resource: self.bufs.bnd_mass.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 18, resource: self.bufs.bnd_pressure.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 19, resource: self.bufs.cell_offsets.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 20, resource: self.bufs.cell_counts.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 21, resource: self.bufs.sorted_indices.as_entire_binding() },
            ],
        })
    }

    /// Create bind group for the integrate shader.
    fn create_integrate_bind_group(&self) -> wgpu::BindGroup {
        self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("integrate_bg"),
            layout: &self.bg_layout_integrate,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: self.bufs.params_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: self.bufs.pos_x.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: self.bufs.pos_y.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: self.bufs.pos_z.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: self.bufs.vel_x.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: self.bufs.vel_y.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 6, resource: self.bufs.vel_z.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 7, resource: self.bufs.acc_x.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 8, resource: self.bufs.acc_y.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 9, resource: self.bufs.acc_z.as_entire_binding() },
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
            pass: 0,
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
            let bg = self.create_integrate_bind_group();
            let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("half_kick"),
            });
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("half_kick"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.pipeline_half_kick);
                pass.set_bind_group(0, &bg, &[]);
                pass.dispatch_workgroups(wg_particles, 1, 1);
            }
            self.queue.submit(std::iter::once(encoder.finish()));
            self.device.poll(wgpu::Maintain::Wait);
        }

        // --- 2. Drift: x += v * dt + domain clamping ---
        {
            let bg = self.create_integrate_bind_group();
            let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("drift"),
            });
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("drift"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.pipeline_drift);
                pass.set_bind_group(0, &bg, &[]);
                pass.dispatch_workgroups(wg_particles, 1, 1);
            }
            self.queue.submit(std::iter::once(encoder.finish()));
            self.device.poll(wgpu::Maintain::Wait);
        }

        // --- 3-7. Full force computation pipeline ---
        self.compute_forces_gpu(&params);

        // --- 8. Second half-kick: v += a * dt/2 ---
        {
            let bg = self.create_integrate_bind_group();
            let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("half_kick_2"),
            });
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("half_kick_2"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.pipeline_half_kick);
                pass.set_bind_group(0, &bg, &[]);
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

// ---- Bind group layout creation helpers ----

fn create_grid_bind_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("grid_bgl"),
        entries: &[
            // 0: params (uniform)
            bgl_uniform(0),
            // 1-3: pos_x, pos_y, pos_z (read-only storage)
            bgl_storage_ro(1),
            bgl_storage_ro(2),
            bgl_storage_ro(3),
            // 4: cell_indices (read-write, atomic)
            bgl_storage_rw(4),
            // 5: cell_counts (read-write, atomic)
            bgl_storage_rw(5),
            // 6: cell_offsets (read-write)
            bgl_storage_rw(6),
            // 7: sorted_indices (read-write)
            bgl_storage_rw(7),
            // 8: write_heads (read-write, atomic)
            bgl_storage_rw(8),
        ],
    })
}

fn create_density_bind_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("density_bgl"),
        entries: &[
            bgl_uniform(0),       // params
            bgl_storage_ro(1),    // pos_x
            bgl_storage_ro(2),    // pos_y
            bgl_storage_ro(3),    // pos_z
            bgl_storage_ro(4),    // mass
            bgl_storage_rw(5),    // density
            bgl_storage_rw(6),    // pressure
            bgl_storage_ro(7),    // fluid_type
            bgl_storage_ro(8),    // bnd_x
            bgl_storage_ro(9),    // bnd_y
            bgl_storage_ro(10),   // bnd_z
            bgl_storage_ro(11),   // bnd_mass
            bgl_storage_ro(12),   // cell_offsets
            bgl_storage_ro(13),   // cell_counts
            bgl_storage_ro(14),   // sorted_indices
        ],
    })
}

fn create_forces_bind_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("forces_bgl"),
        entries: &[
            bgl_uniform(0),       // params
            bgl_storage_ro(1),    // pos_x
            bgl_storage_ro(2),    // pos_y
            bgl_storage_ro(3),    // pos_z
            bgl_storage_ro(4),    // vel_x
            bgl_storage_ro(5),    // vel_y
            bgl_storage_ro(6),    // vel_z
            bgl_storage_ro(7),    // mass
            bgl_storage_ro(8),    // density
            bgl_storage_ro(9),    // pressure
            bgl_storage_ro(10),   // fluid_type
            bgl_storage_rw(11),   // acc_x
            bgl_storage_rw(12),   // acc_y
            bgl_storage_rw(13),   // acc_z
            bgl_storage_ro(14),   // bnd_x
            bgl_storage_ro(15),   // bnd_y
            bgl_storage_ro(16),   // bnd_z
            bgl_storage_ro(17),   // bnd_mass
            bgl_storage_rw(18),   // bnd_pressure
            bgl_storage_ro(19),   // cell_offsets
            bgl_storage_ro(20),   // cell_counts
            bgl_storage_ro(21),   // sorted_indices
        ],
    })
}

fn create_integrate_bind_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("integrate_bgl"),
        entries: &[
            bgl_uniform(0),       // params
            bgl_storage_rw(1),    // pos_x
            bgl_storage_rw(2),    // pos_y
            bgl_storage_rw(3),    // pos_z
            bgl_storage_rw(4),    // vel_x
            bgl_storage_rw(5),    // vel_y
            bgl_storage_rw(6),    // vel_z
            bgl_storage_ro(7),    // acc_x
            bgl_storage_ro(8),    // acc_y
            bgl_storage_ro(9),    // acc_z
        ],
    })
}

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
