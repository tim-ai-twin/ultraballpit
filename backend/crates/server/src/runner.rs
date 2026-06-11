//! Simulation runner that wraps the kernel and provides thread-safe access

use kernel::{ErrorMetrics, ParticleArrays, SimulationKernel};
use orchestrator::config::SimulationConfig;
use orchestrator::domain;
use orchestrator::force;
use orchestrator::geometry;
use orchestrator::geometry::GridSDF;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

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

/// Thread-safe simulation runner.
///
/// All mutable state is behind `Arc`s, so the runner is cheaply cloneable;
/// a clone can be moved onto a blocking thread for stepping while the
/// original stays in the `AppState` map.
#[derive(Clone)]
pub struct SimulationRunner {
    /// Simulation kernel (CPU or GPU, WCSPH or PCISPH)
    kernel: Arc<Mutex<Box<dyn SimulationKernel + Send>>>,
    /// Simulation status
    status: Arc<Mutex<SimStatus>>,
    /// Timestep counter
    timestep: Arc<Mutex<u64>>,
    /// Simulation time
    sim_time: Arc<Mutex<f64>>,
    /// Current adaptive timestep (seconds), updated each step
    dt: Arc<Mutex<f32>>,
    /// Measured simulation throughput (steps per wall-clock second)
    steps_per_sec: Arc<Mutex<f32>>,
    /// Smoothing length for adaptive timestep computation
    h: f32,
    /// Speed of sound for adaptive timestep computation
    speed_of_sound: f32,
    /// CFL number for adaptive timestep computation
    cfl_number: f32,
    /// Stop automatically after this much simulated time
    max_time: Option<f64>,
    /// Stop automatically after this many timesteps
    max_timesteps: Option<u64>,
    /// Initial particle spacing (meters) — the particles' physical size
    particle_spacing: f32,
    /// Domain minimum bounds
    domain_min: [f32; 3],
    /// Domain maximum bounds
    domain_max: [f32; 3],
    /// Fluid type (0=Water, 1=Air, 2=Mixed)
    fluid_type: u8,
    /// Solver type (0=WCSPH, 1=PCISPH)
    solver: u8,
    /// Obstacle geometry triangles (for client-side rendering)
    triangles: Arc<Vec<geometry::Triangle>>,
    /// Signed distance field for force computation
    sdf: Arc<GridSDF>,
    /// Force history (timestep -> force)
    force_history: Arc<Mutex<Vec<ForceRecord>>>,
}

impl SimulationRunner {
    /// Create a new simulation runner from configuration
    ///
    /// `config_dir` is the directory containing the config file, used to resolve
    /// relative geometry file paths.
    pub fn new(config: SimulationConfig, config_dir: &std::path::Path) -> Result<Self, String> {
        let triangles = geometry::resolve_geometry(&config, config_dir)?;
        let sdf = geometry::generate_sdf(
            &triangles,
            config.domain.min,
            config.domain.max,
            0.5 * config.particle_spacing,
        );

        // Initialize domain (fluid and boundary particles)
        let (fluid_particles, boundary_data) = domain::setup_domain(&config, &sdf);

        // Convert boundary data to BoundaryParticles structure
        let mut boundary_particles = kernel::BoundaryParticles::new();
        for b in boundary_data {
            boundary_particles.push(b.x, b.y, b.z, b.mass, b.nx, b.ny, b.nz);
        }

        // Calculate smoothing length
        let h = config.smoothing_length();
        let solver_type = config.solver.to_kernel_solver_type();

        // Create kernel honoring the config's backend (cpu/gpu/auto) and solver
        let kernel = orchestrator::create_kernel(
            &config.backend,
            fluid_particles,
            boundary_particles,
            h,
            config.gravity,
            config.speed_of_sound,
            config.cfl_number,
            config.viscosity,
            config.domain.min,
            config.domain.max,
            solver_type,
        );

        // Initial timestep estimate via CFL condition (will be adaptively updated)
        let initial_dt = config.cfl_number * h / config.speed_of_sound;

        let fluid_type = match config.fluid_type {
            orchestrator::config::ConfigFluidType::Water => 0,
            orchestrator::config::ConfigFluidType::Air => 1,
            orchestrator::config::ConfigFluidType::Mixed => 2,
        };
        let solver = match config.solver {
            orchestrator::config::ConfigSolverType::Wcsph => 0,
            orchestrator::config::ConfigSolverType::Pcisph => 1,
        };

        Ok(Self {
            kernel: Arc::new(Mutex::new(kernel)),
            status: Arc::new(Mutex::new(SimStatus::Created)),
            timestep: Arc::new(Mutex::new(0)),
            sim_time: Arc::new(Mutex::new(0.0)),
            dt: Arc::new(Mutex::new(initial_dt)),
            steps_per_sec: Arc::new(Mutex::new(0.0)),
            h,
            speed_of_sound: config.speed_of_sound,
            cfl_number: config.cfl_number,
            max_time: config.max_time,
            max_timesteps: config.max_timesteps,
            particle_spacing: config.particle_spacing,
            domain_min: config.domain.min,
            domain_max: config.domain.max,
            fluid_type,
            solver,
            triangles: Arc::new(triangles),
            sdf: Arc::new(sdf),
            force_history: Arc::new(Mutex::new(Vec::new())),
        })
    }

    /// Start the simulation
    pub fn start(&self) {
        let mut status = self.status.lock().unwrap();
        // A finished simulation stays finished
        if *status != SimStatus::Stopped {
            *status = SimStatus::Running;
        }
    }

    /// Pause the simulation
    pub fn pause(&self) {
        *self.status.lock().unwrap() = SimStatus::Paused;
    }

    /// Resume the simulation
    pub fn resume(&self) {
        let mut status = self.status.lock().unwrap();
        // A finished simulation cannot be resumed
        if *status != SimStatus::Stopped {
            *status = SimStatus::Running;
        }
    }

    /// Stop the simulation
    pub fn stop(&self) {
        *self.status.lock().unwrap() = SimStatus::Stopped;
    }

    /// Get current status
    pub fn status(&self) -> SimStatus {
        *self.status.lock().unwrap()
    }

    /// Run simulation steps until `budget` wall-clock time is spent (or the
    /// simulation pauses/finishes). Returns the number of steps executed.
    ///
    /// Designed to be called from a blocking thread in a loop. Instability
    /// checks and force recording happen once per batch (not per step) to
    /// avoid GPU readbacks and full-array clones in the hot loop.
    pub fn step_batch(&self, budget: Duration) -> u32 {
        if *self.status.lock().unwrap() != SimStatus::Running {
            return 0;
        }

        let start = Instant::now();
        let mut steps: u32 = 0;
        let mut kernel = self.kernel.lock().unwrap();
        let mut dt = *self.dt.lock().unwrap();
        let mut batch_sim_time = 0.0_f64;

        loop {
            // Recompute the adaptive CFL timestep every few steps. Velocities
            // change little across a handful of steps, and on the GPU backend
            // each `particles()` call costs a device readback.
            if steps % 4 == 0 {
                dt = kernel::sph::compute_timestep(
                    kernel.particles(),
                    self.h,
                    self.speed_of_sound,
                    self.cfl_number,
                );
            }

            kernel.step(dt);
            steps += 1;
            batch_sim_time += dt as f64;

            if start.elapsed() >= budget {
                break;
            }
            // Respect external pause requests mid-batch
            if steps % 16 == 0 && *self.status.lock().unwrap() != SimStatus::Running {
                break;
            }
        }

        *self.dt.lock().unwrap() = dt;

        // Update counters
        let new_timestep = {
            let mut ts = self.timestep.lock().unwrap();
            *ts += steps as u64;
            *ts
        };
        let new_sim_time = {
            let mut st = self.sim_time.lock().unwrap();
            *st += batch_sim_time;
            *st
        };

        // Throughput measurement
        let elapsed = start.elapsed().as_secs_f32();
        if elapsed > 0.0 {
            *self.steps_per_sec.lock().unwrap() = steps as f32 / elapsed;
        }

        // Per-batch health checks + force recording on a single borrowed snapshot
        let metrics = kernel.error_metrics();
        let particles = kernel.particles();

        if metrics.max_density_variation > 100.0 {
            tracing::error!(
                "Simulation instability detected: density variation = {:.2}x (> 100x threshold). Auto-pausing simulation.",
                metrics.max_density_variation
            );
            *self.status.lock().unwrap() = SimStatus::Paused;
            return steps;
        }

        let has_nan_inf = particles
            .x
            .iter()
            .chain(particles.y.iter())
            .chain(particles.z.iter())
            .chain(particles.vx.iter())
            .chain(particles.vy.iter())
            .chain(particles.vz.iter())
            .any(|&v| !v.is_finite());

        if has_nan_inf {
            tracing::error!(
                "Simulation instability detected: NaN or Inf values in particle data. Auto-pausing simulation."
            );
            *self.status.lock().unwrap() = SimStatus::Paused;
            return steps;
        }

        // Record surface forces once per batch
        let surface_force = force::compute_surface_forces(particles, &self.sdf, self.h);
        self.force_history.lock().unwrap().push(ForceRecord {
            timestep: new_timestep,
            sim_time: new_sim_time,
            net_force: surface_force.net_force,
            net_moment: surface_force.net_moment,
        });

        // Auto-finish at configured limits
        let time_limit_reached = self.max_time.is_some_and(|t| new_sim_time >= t);
        let step_limit_reached = self.max_timesteps.is_some_and(|n| new_timestep >= n);
        if time_limit_reached || step_limit_reached {
            tracing::info!(
                "Simulation finished at t={:.4}s after {} steps",
                new_sim_time,
                new_timestep
            );
            *self.status.lock().unwrap() = SimStatus::Stopped;
        }

        steps
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

    /// Get subsample count (all particles are streamed)
    pub fn subsample_count(&self) -> usize {
        self.particle_count()
    }

    /// Get current timestep duration
    pub fn dt(&self) -> f32 {
        *self.dt.lock().unwrap()
    }

    /// Get measured simulation throughput (steps/second)
    pub fn steps_per_sec(&self) -> f32 {
        *self.steps_per_sec.lock().unwrap()
    }

    /// Get initial particle spacing (meters)
    pub fn particle_spacing(&self) -> f32 {
        self.particle_spacing
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

    /// Get solver type (0=WCSPH, 1=PCISPH)
    pub fn solver(&self) -> u8 {
        self.solver
    }

    /// Obstacle geometry triangles
    pub fn triangles(&self) -> &[geometry::Triangle] {
        &self.triangles
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
