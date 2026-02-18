//! Simulation runner with lifecycle management
//!
//! This module provides the `SimulationRunner` which manages the simulation
//! lifecycle in a background thread, including start, pause, resume, and
//! status tracking.

use kernel::{ErrorMetrics, ParticleArrays, SimulationKernel};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Instant;

/// Runner state enum
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RunnerState {
    /// Simulation created but not yet started
    Created,
    /// Simulation actively running
    Running,
    /// Simulation paused
    Paused,
    /// Simulation finished (reached stopping condition)
    Finished,
    /// Simulation encountered an error
    Error,
}

/// Shared state between the runner thread and control interface
struct SharedState {
    /// Current runner state
    state: RunnerState,
    /// Current simulation time (seconds)
    sim_time: f64,
    /// Number of timesteps executed
    timestep_count: u64,
    /// Most recent error message (if state is Error)
    error_message: Option<String>,
}

/// Handle for controlling and querying a running simulation
pub struct SimulationRunner {
    /// Shared state (protected by mutex)
    shared: Arc<Mutex<SharedState>>,
    /// Handle to the background thread
    thread_handle: Option<thread::JoinHandle<()>>,
}

impl SimulationRunner {
    /// Create a new simulation runner with the given kernel
    ///
    /// # Arguments
    /// * `kernel` - The simulation kernel to run
    /// * `h` - Smoothing length (for CFL timestep calculation)
    /// * `speed_of_sound` - Speed of sound (for CFL timestep calculation)
    /// * `cfl_number` - CFL number (for CFL timestep calculation)
    /// * `max_timesteps` - Optional maximum number of timesteps
    /// * `max_time` - Optional maximum simulation time (seconds)
    pub fn new(
        mut kernel: Box<dyn SimulationKernel + Send>,
        h: f32,
        speed_of_sound: f32,
        cfl_number: f32,
        max_timesteps: Option<u64>,
        max_time: Option<f64>,
    ) -> Self {
        let shared = Arc::new(Mutex::new(SharedState {
            state: RunnerState::Created,
            sim_time: 0.0,
            timestep_count: 0,
            error_message: None,
        }));

        let shared_clone = Arc::clone(&shared);

        // Spawn background thread
        let thread_handle = thread::spawn(move || {
            run_simulation_loop(
                kernel.as_mut(),
                shared_clone,
                h,
                speed_of_sound,
                cfl_number,
                max_timesteps,
                max_time,
            );
        });

        Self {
            shared,
            thread_handle: Some(thread_handle),
        }
    }

    /// Get current runner state
    pub fn state(&self) -> RunnerState {
        self.shared.lock().unwrap().state.clone()
    }

    /// Get current simulation time (seconds)
    pub fn sim_time(&self) -> f64 {
        self.shared.lock().unwrap().sim_time
    }

    /// Get current timestep count
    pub fn timestep_count(&self) -> u64 {
        self.shared.lock().unwrap().timestep_count
    }

    /// Get error message if state is Error
    pub fn error_message(&self) -> Option<String> {
        self.shared.lock().unwrap().error_message.clone()
    }

    /// Pause the simulation
    pub fn pause(&self) {
        let mut state = self.shared.lock().unwrap();
        if state.state == RunnerState::Running {
            state.state = RunnerState::Paused;
        }
    }

    /// Resume the simulation
    pub fn resume(&self) {
        let mut state = self.shared.lock().unwrap();
        if state.state == RunnerState::Paused {
            state.state = RunnerState::Running;
        }
    }

    /// Start the simulation (transition from Created to Running)
    pub fn start(&self) {
        let mut state = self.shared.lock().unwrap();
        if state.state == RunnerState::Created {
            state.state = RunnerState::Running;
        }
    }

    /// Wait for the simulation thread to complete
    pub fn join(mut self) -> Result<(), String> {
        if let Some(handle) = self.thread_handle.take() {
            handle.join().map_err(|_| "Thread panicked".to_string())?;
        }
        Ok(())
    }
}

impl Drop for SimulationRunner {
    fn drop(&mut self) {
        // Set state to Finished to signal thread to exit
        if let Ok(mut state) = self.shared.lock() {
            if state.state == RunnerState::Running || state.state == RunnerState::Paused {
                state.state = RunnerState::Finished;
            }
        }
    }
}

/// Main simulation loop executed in background thread
fn run_simulation_loop(
    kernel: &mut dyn SimulationKernel,
    shared: Arc<Mutex<SharedState>>,
    h: f32,
    speed_of_sound: f32,
    cfl_number: f32,
    max_timesteps: Option<u64>,
    max_time: Option<f64>,
) {
    // Wait for start signal
    loop {
        let state = {
            let guard = shared.lock().unwrap();
            guard.state.clone()
        };

        match state {
            RunnerState::Created => {
                // Wait a bit and check again
                thread::sleep(std::time::Duration::from_millis(10));
            }
            RunnerState::Running => break,
            _ => return, // Exit if finished or error
        }
    }

    let start_wall_time = Instant::now();
    let mut sim_time = 0.0_f64;
    let mut timestep_count = 0_u64;

    // Optimistic timestepping: try larger dt, rollback if quality degrades
    const OPTIMISTIC_ALPHA: f32 = 1.5;
    const DENSITY_VAR_LIMIT: f32 = 0.03; // 3% max density variation (spec SC-003)
    const COOLDOWN_STEPS: u32 = 10; // steps to skip optimistic after a rollback
    let mut cooldown = 0u32;
    let mut optimistic_accepted = 0u64;
    let mut optimistic_rejected = 0u64;

    loop {
        // Check state
        let current_state = {
            let guard = shared.lock().unwrap();
            guard.state.clone()
        };

        match current_state {
            RunnerState::Running => {
                // Execute one timestep

                // Compute adaptive timestep using CFL condition
                let dt_safe = kernel::sph::compute_timestep(
                    kernel.particles(),
                    h,
                    speed_of_sound,
                    cfl_number,
                );

                // Optimistic timestepping: try a larger dt when not in cooldown
                let dt_used;
                if cooldown > 0 {
                    cooldown -= 1;
                    kernel.step(dt_safe);
                    dt_used = dt_safe;
                } else if kernel.save_checkpoint() {
                    let dt_try = dt_safe * OPTIMISTIC_ALPHA;
                    kernel.step(dt_try);
                    let metrics = kernel.error_metrics();
                    if metrics.max_density_variation > DENSITY_VAR_LIMIT {
                        // Optimistic step violated density bound, rollback
                        kernel.restore_checkpoint();
                        kernel.step(dt_safe);
                        dt_used = dt_safe;
                        optimistic_rejected += 1;
                        cooldown = COOLDOWN_STEPS;
                    } else {
                        dt_used = dt_try;
                        optimistic_accepted += 1;
                    }
                } else {
                    // Kernel doesn't support checkpointing, use safe dt
                    kernel.step(dt_safe);
                    dt_used = dt_safe;
                };

                // Update counters
                sim_time += dt_used as f64;
                timestep_count += 1;

                // Update shared state
                {
                    let mut guard = shared.lock().unwrap();
                    guard.sim_time = sim_time;
                    guard.timestep_count = timestep_count;
                }

                // Check stopping conditions
                if let Some(max_steps) = max_timesteps {
                    if timestep_count >= max_steps {
                        tracing::info!(
                            "Simulation finished: reached max_timesteps = {}",
                            max_steps
                        );
                        let mut guard = shared.lock().unwrap();
                        guard.state = RunnerState::Finished;
                        break;
                    }
                }

                if let Some(max_t) = max_time {
                    if sim_time >= max_t {
                        tracing::info!(
                            "Simulation finished: reached max_time = {:.3}s",
                            max_t
                        );
                        let mut guard = shared.lock().unwrap();
                        guard.state = RunnerState::Finished;
                        break;
                    }
                }

                // Log progress periodically
                if timestep_count % 100 == 0 {
                    let wall_time = start_wall_time.elapsed().as_secs_f64();
                    tracing::debug!(
                        "Step {}: sim_time={:.4}s, dt={:.6}s, wall_time={:.2}s, optimistic={}/{}",
                        timestep_count,
                        sim_time,
                        dt_used,
                        wall_time,
                        optimistic_accepted,
                        optimistic_accepted + optimistic_rejected,
                    );
                }
            }
            RunnerState::Paused => {
                // Wait while paused
                thread::sleep(std::time::Duration::from_millis(50));
            }
            RunnerState::Finished | RunnerState::Error => {
                // Exit loop
                break;
            }
            RunnerState::Created => {
                // Shouldn't happen, but treat as finished
                break;
            }
        }
    }

    let total_optimistic = optimistic_accepted + optimistic_rejected;
    if total_optimistic > 0 {
        tracing::info!(
            "Simulation thread exiting: {} timesteps, {:.4}s simulated, \
             optimistic {}/{} accepted ({:.0}%)",
            timestep_count,
            sim_time,
            optimistic_accepted,
            total_optimistic,
            100.0 * optimistic_accepted as f64 / total_optimistic as f64,
        );
    } else {
        tracing::info!(
            "Simulation thread exiting: {} timesteps, {:.4}s simulated",
            timestep_count,
            sim_time
        );
    }
}

/// Extension trait to get particle snapshots and metrics from the runner
///
/// Note: This requires accessing the kernel, which is owned by the background
/// thread. For now, we don't support live snapshots. A production implementation
/// would use channels or shared memory for this.
pub trait RunnerQuery {
    /// Get a snapshot of current particles (not yet implemented)
    fn particles_snapshot(&self) -> Option<ParticleArrays>;

    /// Get current error metrics (not yet implemented)
    fn error_metrics(&self) -> Option<ErrorMetrics>;
}

impl RunnerQuery for SimulationRunner {
    fn particles_snapshot(&self) -> Option<ParticleArrays> {
        // TODO: Implement via channel or shared memory
        // For now, return None
        None
    }

    fn error_metrics(&self) -> Option<ErrorMetrics> {
        // TODO: Implement via channel or shared memory
        // For now, return None
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use kernel::{BoundaryParticles, CpuKernel, ParticleArrays};

    #[test]
    fn test_runner_lifecycle() {
        // Create a minimal simulation
        let particles = ParticleArrays::new();
        let boundary = BoundaryParticles::new();
        let h = 0.013;
        let gravity = [0.0, -9.81, 0.0];
        let speed_of_sound = 50.0;
        let cfl_number = 0.4;
        let viscosity = 0.001;
        let domain_min = [0.0, 0.0, 0.0];
        let domain_max = [1.0, 1.0, 1.0];

        let kernel = CpuKernel::new(
            particles,
            boundary,
            h,
            gravity,
            speed_of_sound,
            cfl_number,
            viscosity,
            domain_min,
            domain_max,
        );

        let runner = SimulationRunner::new(
            Box::new(kernel),
            h,
            speed_of_sound,
            cfl_number,
            Some(10),
            None,
        );

        // Initially Created
        assert_eq!(runner.state(), RunnerState::Created);

        // Start
        runner.start();
        assert_eq!(runner.state(), RunnerState::Running);

        // Wait a bit for it to run
        thread::sleep(std::time::Duration::from_millis(100));

        // Should have made progress or finished
        let steps = runner.timestep_count();
        assert!(steps <= 10);

        // Wait for completion
        runner.join().unwrap();
    }

    #[test]
    fn test_runner_pause_resume() {
        let particles = ParticleArrays::new();
        let boundary = BoundaryParticles::new();
        let h = 0.013;
        let gravity = [0.0, -9.81, 0.0];
        let speed_of_sound = 50.0;
        let cfl_number = 0.4;
        let viscosity = 0.001;
        let domain_min = [0.0, 0.0, 0.0];
        let domain_max = [1.0, 1.0, 1.0];

        let kernel = CpuKernel::new(
            particles,
            boundary,
            h,
            gravity,
            speed_of_sound,
            cfl_number,
            viscosity,
            domain_min,
            domain_max,
        );

        let runner = SimulationRunner::new(
            Box::new(kernel),
            h,
            speed_of_sound,
            cfl_number,
            Some(100),
            None,
        );

        runner.start();
        thread::sleep(std::time::Duration::from_millis(50));

        // Pause
        runner.pause();

        // Wait for pause to take effect
        thread::sleep(std::time::Duration::from_millis(100));
        assert_eq!(runner.state(), RunnerState::Paused);

        let steps_paused = runner.timestep_count();
        thread::sleep(std::time::Duration::from_millis(100));

        // Should not advance significantly while paused (allow for 1 step race condition)
        let steps_after_pause = runner.timestep_count();
        assert!(
            steps_after_pause <= steps_paused + 1,
            "Steps should not advance while paused: before={}, after={}",
            steps_paused,
            steps_after_pause
        );

        // Resume
        runner.resume();
        assert_eq!(runner.state(), RunnerState::Running);

        runner.join().unwrap();
    }
}
