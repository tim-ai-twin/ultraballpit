//! Simulation runner that wraps the kernel and provides thread-safe access

use kernel::{CpuKernel, ErrorMetrics, ParticleArrays, SimulationKernel};
use orchestrator::config::SimulationConfig;
use orchestrator::domain;
use orchestrator::geometry;
use std::sync::{Arc, Mutex};

use crate::state::SimStatus;

/// Thread-safe simulation runner
pub struct SimulationRunner {
    /// Simulation kernel (CPU for now)
    kernel: Arc<Mutex<CpuKernel>>,
    /// Simulation status
    status: Arc<Mutex<SimStatus>>,
    /// Timestep counter
    timestep: Arc<Mutex<u64>>,
    /// Simulation time
    sim_time: Arc<Mutex<f64>>,
    /// Fixed timestep (seconds)
    dt: f32,
    /// Subsample count (~5% of particles)
    subsample_count: usize,
    /// Domain minimum bounds
    domain_min: [f32; 3],
    /// Domain maximum bounds
    domain_max: [f32; 3],
    /// Fluid type (0=Water, 1=Air, 2=Mixed)
    fluid_type: u8,
}

impl SimulationRunner {
    /// Create a new simulation runner from configuration
    ///
    /// `config_dir` is the directory containing the config file, used to resolve
    /// relative geometry file paths.
    pub fn new(config: SimulationConfig, config_dir: &std::path::Path) -> Result<Self, String> {
        // Resolve geometry path relative to config directory
        let geometry_path = config_dir.join(&config.geometry_file);
        let mesh = geometry::load_stl(geometry_path.to_str().ok_or("Invalid geometry path")?)?;
        let sdf = geometry::mesh_to_sdf(&mesh, &config.domain, 0.5 * config.particle_spacing);

        // Initialize domain (fluid and boundary particles)
        let (fluid_particles, boundary_data) = domain::setup_domain(&config, &sdf);

        // Convert boundary data to BoundaryParticles structure
        let mut boundary_particles = kernel::BoundaryParticles::new();
        for b in boundary_data {
            boundary_particles.push(b.x, b.y, b.z, b.mass, b.nx, b.ny, b.nz);
        }

        // Calculate smoothing length
        let h = config.smoothing_length();

        // Create kernel
        let kernel = CpuKernel::new(
            fluid_particles,
            boundary_particles,
            h,
            config.gravity,
            config.speed_of_sound,
            config.cfl_number,
            config.viscosity,
            config.domain.min,
            config.domain.max,
        );

        let particle_count = kernel.particle_count();
        let subsample_count = (particle_count as f32 * 0.05).max(1.0) as usize;

        // Calculate timestep (CFL condition)
        let dt = config.cfl_number * h / config.speed_of_sound;

        let fluid_type = match config.fluid_type {
            orchestrator::config::ConfigFluidType::Water => 0,
            orchestrator::config::ConfigFluidType::Air => 1,
            orchestrator::config::ConfigFluidType::Mixed => 2,
        };

        Ok(Self {
            kernel: Arc::new(Mutex::new(kernel)),
            status: Arc::new(Mutex::new(SimStatus::Created)),
            timestep: Arc::new(Mutex::new(0)),
            sim_time: Arc::new(Mutex::new(0.0)),
            dt,
            subsample_count,
            domain_min: config.domain.min,
            domain_max: config.domain.max,
            fluid_type,
        })
    }

    /// Start the simulation
    pub fn start(&self) {
        *self.status.lock().unwrap() = SimStatus::Running;
    }

    /// Pause the simulation
    pub fn pause(&self) {
        *self.status.lock().unwrap() = SimStatus::Paused;
    }

    /// Resume the simulation
    pub fn resume(&self) {
        *self.status.lock().unwrap() = SimStatus::Running;
    }

    /// Stop the simulation
    pub fn stop(&self) {
        *self.status.lock().unwrap() = SimStatus::Stopped;
    }

    /// Get current status
    pub fn status(&self) -> SimStatus {
        *self.status.lock().unwrap()
    }

    /// Execute one timestep (if running)
    pub fn step(&self) {
        let status = *self.status.lock().unwrap();
        if status != SimStatus::Running {
            return;
        }

        // Execute kernel step
        self.kernel.lock().unwrap().step(self.dt);

        // Increment counters
        *self.timestep.lock().unwrap() += 1;
        *self.sim_time.lock().unwrap() += self.dt as f64;
    }

    /// Get current particle snapshot (cloned)
    pub fn particles(&self) -> ParticleArrays {
        self.kernel.lock().unwrap().particles().clone()
    }

    /// Get error metrics
    pub fn error_metrics(&self) -> ErrorMetrics {
        self.kernel.lock().unwrap().error_metrics()
    }

    /// Get simulation time
    pub fn sim_time(&self) -> f64 {
        *self.sim_time.lock().unwrap()
    }

    /// Get timestep count
    pub fn timestep_count(&self) -> u64 {
        *self.timestep.lock().unwrap()
    }

    /// Get particle count
    pub fn particle_count(&self) -> usize {
        self.kernel.lock().unwrap().particle_count()
    }

    /// Get subsample count (~5% of particles)
    pub fn subsample_count(&self) -> usize {
        self.subsample_count
    }

    /// Get timestep duration
    pub fn dt(&self) -> f32 {
        self.dt
    }

    /// Get domain minimum bounds
    pub fn domain_min(&self) -> [f32; 3] {
        self.domain_min
    }

    /// Get domain maximum bounds
    pub fn domain_max(&self) -> [f32; 3] {
        self.domain_max
    }

    /// Get fluid type (0=Water, 1=Air, 2=Mixed)
    pub fn fluid_type(&self) -> u8 {
        self.fluid_type
    }
}
