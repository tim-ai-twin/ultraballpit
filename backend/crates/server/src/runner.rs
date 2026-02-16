//! Simulation runner that wraps the kernel and provides thread-safe access

use kernel::{CpuKernel, ErrorMetrics, ParticleArrays, SimulationKernel};
use orchestrator::config::SimulationConfig;
use orchestrator::domain;
use orchestrator::force;
use orchestrator::geometry;
use orchestrator::geometry::GridSDF;
use std::sync::{Arc, Mutex};

use crate::state::SimStatus;

/// Force measurement at a single timestep
#[derive(Debug, Clone, serde::Serialize)]
pub struct ForceRecord {
    /// Timestep number
    pub timestep: u64,
    /// Simulation time (seconds)
    pub sim_time: f64,
    /// Net force vector [Fx, Fy, Fz] (Newtons)
    pub net_force: [f32; 3],
    /// Net moment vector [Tx, Ty, Tz] (N·m)
    pub net_moment: [f32; 3],
}

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
    /// Signed distance field for force computation
    sdf: Arc<GridSDF>,
    /// Smoothing length
    h: f32,
    /// Force history (timestep -> force)
    force_history: Arc<Mutex<Vec<ForceRecord>>>,
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
            sdf: Arc::new(sdf),
            h,
            force_history: Arc::new(Mutex::new(Vec::new())),
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

        // Check for instabilities (clone to avoid holding lock)
        let (metrics, particles) = {
            let kernel = self.kernel.lock().unwrap();
            (kernel.error_metrics(), kernel.particles().clone())
        };

        // Detect runaway density variations (> 100x rest density)
        if metrics.max_density_variation > 100.0 {
            tracing::error!(
                "Simulation instability detected: density variation = {:.2}x (> 100x threshold). Auto-pausing simulation.",
                metrics.max_density_variation
            );
            *self.status.lock().unwrap() = SimStatus::Paused;
            return;
        }

        // Check for NaN/Inf in particle arrays
        let has_nan_inf = particles.x.iter()
            .chain(particles.y.iter())
            .chain(particles.z.iter())
            .chain(particles.vx.iter())
            .chain(particles.vy.iter())
            .chain(particles.vz.iter())
            .chain(particles.density.iter())
            .chain(particles.pressure.iter())
            .any(|&v| !v.is_finite());

        if has_nan_inf {
            tracing::error!(
                "Simulation instability detected: NaN or Inf values in particle data. Auto-pausing simulation."
            );
            *self.status.lock().unwrap() = SimStatus::Paused;
            return;
        }

        // Increment counters
        let new_timestep = {
            let mut ts = self.timestep.lock().unwrap();
            *ts += 1;
            *ts
        };
        let new_sim_time = {
            let mut st = self.sim_time.lock().unwrap();
            *st += self.dt as f64;
            *st
        };

        // Compute surface forces every timestep
        let particles = self.kernel.lock().unwrap().particles().clone();
        let surface_force = force::compute_surface_forces(&particles, &self.sdf, self.h);

        // Store force record
        let force_record = ForceRecord {
            timestep: new_timestep,
            sim_time: new_sim_time,
            net_force: surface_force.net_force,
            net_moment: surface_force.net_moment,
        };
        self.force_history.lock().unwrap().push(force_record);
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

    /// Get force history (cloned)
    pub fn force_history(&self) -> Vec<ForceRecord> {
        self.force_history.lock().unwrap().clone()
    }

    /// Get force records in a time range with optional aggregation
    pub fn get_forces(
        &self,
        from_timestep: Option<u64>,
        to_timestep: Option<u64>,
    ) -> Vec<ForceRecord> {
        let history = self.force_history.lock().unwrap();

        let from = from_timestep.unwrap_or(0);
        let to = to_timestep.unwrap_or(u64::MAX);

        history
            .iter()
            .filter(|r| r.timestep >= from && r.timestep <= to)
            .cloned()
            .collect()
    }

    /// Get peak force magnitude in history
    pub fn peak_force(&self) -> Option<f32> {
        let history = self.force_history.lock().unwrap();
        history
            .iter()
            .map(|r| {
                let fx = r.net_force[0];
                let fy = r.net_force[1];
                let fz = r.net_force[2];
                (fx * fx + fy * fy + fz * fz).sqrt()
            })
            .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
    }

    /// Get mean force vector in history
    pub fn mean_force(&self) -> Option<[f32; 3]> {
        let history = self.force_history.lock().unwrap();
        if history.is_empty() {
            return None;
        }

        let mut sum = [0.0, 0.0, 0.0];
        for record in history.iter() {
            sum[0] += record.net_force[0];
            sum[1] += record.net_force[1];
            sum[2] += record.net_force[2];
        }

        let count = history.len() as f32;
        Some([sum[0] / count, sum[1] / count, sum[2] / count])
    }
}
