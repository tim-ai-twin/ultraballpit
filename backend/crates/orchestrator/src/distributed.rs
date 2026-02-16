//! Distributed parallel execution coordinator (T068)
//!
//! Provides multi-threaded parallel simulation using domain decomposition.
//! Each subdomain runs its own CpuKernel instance in a separate thread,
//! with ghost particle exchange between timesteps for correctness near
//! subdomain boundaries.
//!
//! This module uses a local thread-based approach (multiple CpuKernel instances
//! in separate threads) rather than network-based distribution. The API is
//! designed so that network distribution can be added later as a drop-in
//! replacement for the thread-based coordinator.

use std::sync::{Arc, Mutex};
use std::thread;

use kernel::{BoundaryParticles, CpuKernel, ParticleArrays, SimulationKernel};

use crate::domain::{
    build_subdomains, decompose_domain, exchange_ghost_particles,
    merge_subdomain_particles, merge_with_ghosts, strip_ghosts,
    BoundaryParticleData, AABB,
};

/// Configuration for the distributed coordinator
#[derive(Debug, Clone)]
pub struct DistributedConfig {
    /// Number of parallel instances (subdomains)
    pub num_instances: usize,
    /// SPH smoothing length
    pub smoothing_length: f32,
    /// Gravitational acceleration
    pub gravity: [f32; 3],
    /// Speed of sound
    pub speed_of_sound: f32,
    /// CFL number
    pub cfl_number: f32,
    /// Artificial viscosity coefficient
    pub viscosity: f32,
    /// Global domain minimum bounds
    pub domain_min: [f32; 3],
    /// Global domain maximum bounds
    pub domain_max: [f32; 3],
}

/// Result of a distributed simulation run
#[derive(Debug, Clone)]
pub struct DistributedResult {
    /// Final merged particle state
    pub particles: ParticleArrays,
    /// Number of timesteps executed
    pub timesteps: u64,
    /// Total simulation time (seconds)
    pub sim_time: f64,
}

/// Run a distributed parallel simulation for a specified number of timesteps.
///
/// This function:
/// 1. Decomposes the domain into N subdomains
/// 2. Distributes particles to subdomains
/// 3. Creates a CpuKernel for each subdomain
/// 4. Runs timesteps in parallel across threads
/// 5. Exchanges ghost particles between steps
/// 6. Merges results back into a single ParticleArrays
///
/// # Arguments
/// * `config` - Distributed simulation configuration
/// * `particles` - Initial fluid particles
/// * `boundary_data` - Static boundary particles
/// * `num_timesteps` - Number of timesteps to execute
/// * `dt` - Fixed timestep duration (seconds). If None, uses CFL-based adaptive timestep.
///
/// # Returns
/// A `DistributedResult` with the merged final state.
pub fn run_distributed(
    config: &DistributedConfig,
    particles: &ParticleArrays,
    boundary_data: &[BoundaryParticleData],
    num_timesteps: u64,
    dt: Option<f32>,
) -> DistributedResult {
    let n = config.num_instances;

    // 1. Decompose domain
    let global_bounds = AABB::new(config.domain_min, config.domain_max);
    let aabbs = decompose_domain(&global_bounds, n);

    // 2. Build subdomains (distribute particles)
    let subdomains = build_subdomains(
        &aabbs,
        particles,
        boundary_data,
        config.smoothing_length,
    );

    tracing::info!(
        "Distributed simulation: {} subdomains, {} total particles",
        n,
        particles.len()
    );
    for (i, sub) in subdomains.iter().enumerate() {
        tracing::debug!(
            "  Subdomain {}: {} particles, {} boundary particles, {} neighbors",
            i,
            sub.particles.len(),
            sub.boundary_particles.len(),
            sub.neighbor_ids.len()
        );
    }

    // 3. Run the simulation loop with ghost particle exchange
    let mut current_subdomains = subdomains;
    let mut sim_time = 0.0_f64;

    for step in 0..num_timesteps {
        // a. Exchange ghost particles
        let ghost_data = exchange_ghost_particles(&current_subdomains, config.smoothing_length);

        // b. Determine timestep (use fixed dt or compute adaptive from all subdomains)
        let step_dt = if let Some(fixed_dt) = dt {
            fixed_dt
        } else {
            // Use the minimum dt across all subdomains for synchronization
            let mut min_dt = f32::MAX;
            for sub in &current_subdomains {
                let sub_dt = kernel::sph::compute_timestep(
                    &sub.particles,
                    config.smoothing_length,
                    config.speed_of_sound,
                    config.cfl_number,
                );
                if sub_dt < min_dt {
                    min_dt = sub_dt;
                }
            }
            min_dt
        };

        // c. Run each subdomain's timestep (using threads for parallelism)
        let mut handles = Vec::with_capacity(n);
        let step_results: Arc<Mutex<Vec<Option<ParticleArrays>>>> =
            Arc::new(Mutex::new((0..n).map(|_| None).collect()));

        for (i, sub) in current_subdomains.iter().enumerate() {
            // Prepare data for this subdomain's thread
            let sub_particles = sub.particles.clone();
            let sub_boundary = sub.boundary_particles.clone();
            let sub_ghosts = ghost_data[i].clone();
            let h = config.smoothing_length;
            let gravity = config.gravity;
            let speed_of_sound = config.speed_of_sound;
            let cfl_number = config.cfl_number;
            let viscosity = config.viscosity;
            let domain_min = config.domain_min;
            let domain_max = config.domain_max;
            let results = Arc::clone(&step_results);

            let handle = thread::spawn(move || {
                // Merge owned particles with ghost particles
                let (merged_particles, owned_count) =
                    merge_with_ghosts(&sub_particles, &sub_ghosts);

                // Convert boundary data to BoundaryParticles
                let mut bp = BoundaryParticles::new();
                for b in &sub_boundary {
                    bp.push(b.x, b.y, b.z, b.mass, b.nx, b.ny, b.nz);
                }

                // Create a temporary CpuKernel with the merged particles
                // Use the global domain bounds for the neighbor grid so that
                // particles near subdomain boundaries are handled correctly.
                let mut kernel = CpuKernel::new(
                    merged_particles,
                    bp,
                    h,
                    gravity,
                    speed_of_sound,
                    cfl_number,
                    viscosity,
                    domain_min,
                    domain_max,
                );

                // Step the kernel
                kernel.step(step_dt);

                // Extract only owned particles (strip ghosts)
                let updated = strip_ghosts(kernel.particles(), owned_count);

                // Store result
                let mut results_lock = results.lock().unwrap();
                results_lock[i] = Some(updated);
            });

            handles.push(handle);
        }

        // Wait for all threads to complete
        for handle in handles {
            handle.join().expect("Subdomain thread panicked");
        }

        // d. Collect results and update subdomains
        let results = match Arc::try_unwrap(step_results) {
            Ok(mutex) => mutex.into_inner().unwrap(),
            Err(_) => panic!("Arc should have no other references after thread join"),
        };

        for (i, result) in results.into_iter().enumerate() {
            current_subdomains[i].particles = result.expect("All subdomains should have results");
        }

        sim_time += step_dt as f64;

        if (step + 1) % 100 == 0 {
            tracing::debug!(
                "Distributed step {}/{}: sim_time={:.6}s, dt={:.8}s",
                step + 1,
                num_timesteps,
                sim_time,
                step_dt,
            );
        }
    }

    // 4. Merge results from all subdomains
    let merged = merge_subdomain_particles(&current_subdomains);

    tracing::info!(
        "Distributed simulation complete: {} timesteps, {:.6}s simulated, {} particles",
        num_timesteps,
        sim_time,
        merged.len()
    );

    DistributedResult {
        particles: merged,
        timesteps: num_timesteps,
        sim_time,
    }
}

/// Run a single-instance (non-distributed) simulation for comparison.
///
/// This provides a reference implementation that can be compared against the
/// distributed version for validation.
///
/// # Arguments
/// * `config` - Distributed config (uses smoothing_length, gravity, etc.)
/// * `particles` - Initial fluid particles
/// * `boundary_data` - Static boundary particles
/// * `num_timesteps` - Number of timesteps to execute
/// * `dt` - Fixed timestep duration. If None, uses adaptive CFL timestep.
///
/// # Returns
/// A `DistributedResult` with the final state.
pub fn run_single_instance(
    config: &DistributedConfig,
    particles: &ParticleArrays,
    boundary_data: &[BoundaryParticleData],
    num_timesteps: u64,
    dt: Option<f32>,
) -> DistributedResult {
    // Convert boundary data to BoundaryParticles
    let mut bp = BoundaryParticles::new();
    for b in boundary_data {
        bp.push(b.x, b.y, b.z, b.mass, b.nx, b.ny, b.nz);
    }

    let mut kernel = CpuKernel::new(
        particles.clone(),
        bp,
        config.smoothing_length,
        config.gravity,
        config.speed_of_sound,
        config.cfl_number,
        config.viscosity,
        config.domain_min,
        config.domain_max,
    );

    let mut sim_time = 0.0_f64;

    for step in 0..num_timesteps {
        let step_dt = if let Some(fixed_dt) = dt {
            fixed_dt
        } else {
            kernel::sph::compute_timestep(
                kernel.particles(),
                config.smoothing_length,
                config.speed_of_sound,
                config.cfl_number,
            )
        };

        kernel.step(step_dt);
        sim_time += step_dt as f64;

        if (step + 1) % 100 == 0 {
            tracing::debug!(
                "Single-instance step {}/{}: sim_time={:.6}s",
                step + 1,
                num_timesteps,
                sim_time,
            );
        }
    }

    DistributedResult {
        particles: kernel.particles().clone(),
        timesteps: num_timesteps,
        sim_time,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use kernel::FluidType;

    /// Helper to create a small test configuration
    fn test_config(num_instances: usize) -> DistributedConfig {
        DistributedConfig {
            num_instances,
            smoothing_length: 0.013,
            gravity: [0.0, -9.81, 0.0],
            speed_of_sound: 50.0,
            cfl_number: 0.4,
            viscosity: 0.001,
            domain_min: [0.0, 0.0, 0.0],
            domain_max: [0.1, 0.1, 0.1],
        }
    }

    /// Helper to create a small regular grid of particles
    fn create_test_particles(spacing: f32, domain_min: [f32; 3], domain_max: [f32; 3]) -> ParticleArrays {
        let mut particles = ParticleArrays::new();
        let volume = spacing.powi(3);
        let mass = 1000.0 * volume;

        let nx = ((domain_max[0] - domain_min[0]) / spacing).ceil() as usize;
        let ny = ((domain_max[1] - domain_min[1]) / spacing).ceil() as usize;
        let nz = ((domain_max[2] - domain_min[2]) / spacing).ceil() as usize;

        for i in 0..nx {
            for j in 0..ny {
                for k in 0..nz {
                    let x = domain_min[0] + (i as f32 + 0.5) * spacing;
                    let y = domain_min[1] + (j as f32 + 0.5) * spacing;
                    let z = domain_min[2] + (k as f32 + 0.5) * spacing;

                    if x <= domain_max[0] && y <= domain_max[1] && z <= domain_max[2] {
                        particles.push_particle(x, y, z, mass, 1000.0, 293.15, FluidType::Water);
                    }
                }
            }
        }

        particles
    }

    #[test]
    fn test_distributed_preserves_particle_count() {
        let config = test_config(2);
        let particles = create_test_particles(0.01, config.domain_min, config.domain_max);
        let initial_count = particles.len();

        let result = run_distributed(&config, &particles, &[], 5, Some(1e-5));

        assert_eq!(
            result.particles.len(),
            initial_count,
            "Distributed simulation should preserve total particle count"
        );
    }

    #[test]
    fn test_distributed_runs_correct_timesteps() {
        let config = test_config(2);
        let particles = create_test_particles(0.01, config.domain_min, config.domain_max);

        let result = run_distributed(&config, &particles, &[], 10, Some(1e-5));

        assert_eq!(result.timesteps, 10);
        assert!((result.sim_time - 10.0 * 1e-5).abs() < 1e-10);
    }

    #[test]
    fn test_distributed_four_subdomains() {
        let config = test_config(4);
        let particles = create_test_particles(0.01, config.domain_min, config.domain_max);
        let initial_count = particles.len();

        let result = run_distributed(&config, &particles, &[], 3, Some(1e-5));

        assert_eq!(
            result.particles.len(),
            initial_count,
            "4-subdomain run should preserve particle count"
        );
    }

    #[test]
    fn test_single_instance_reference() {
        let config = test_config(1);
        let particles = create_test_particles(0.01, config.domain_min, config.domain_max);
        let initial_count = particles.len();

        let result = run_single_instance(&config, &particles, &[], 5, Some(1e-5));

        assert_eq!(result.particles.len(), initial_count);
        assert_eq!(result.timesteps, 5);
    }
}
