//! Orchestration Layer
//!
//! This crate provides orchestration for the SPH simulation, including:
//! - STL file parsing and mesh processing
//! - SDF (Signed Distance Field) generation
//! - Simulation state management
//! - Simulation runner with lifecycle management
//! - Domain decomposition and distributed parallel execution (T066-T069)

#![warn(missing_docs)]

pub mod config;
pub mod geometry;
pub mod domain;
pub mod runner;
pub mod force;
pub mod distributed;

pub use config::SimulationConfig;
pub use runner::SimulationRunner;

use kernel::{BoundaryParticles, CpuKernel};
use std::path::Path;

/// Create a complete simulation from a configuration file
///
/// This function performs the full simulation setup pipeline:
/// 1. Load and validate the configuration
/// 2. Load the STL geometry file
/// 3. Generate the signed distance field (SDF)
/// 4. Set up the domain with fluid and boundary particles
/// 5. Create the CPU simulation kernel
/// 6. Wrap in a SimulationRunner for lifecycle management
///
/// # Arguments
/// * `config_path` - Path to the JSON configuration file
///
/// # Returns
/// A `SimulationRunner` ready to be started, or an error if setup fails
///
/// # Example
/// ```no_run
/// use orchestrator::create_simulation;
///
/// let runner = create_simulation("config/dam_break.json")?;
/// runner.start();
/// // ... query status, pause, resume, etc.
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn create_simulation(config_path: &str) -> Result<SimulationRunner, Box<dyn std::error::Error>> {
    tracing::info!("Creating simulation from config: {}", config_path);

    // 1. Load and validate configuration
    let config = SimulationConfig::load(config_path)?;
    tracing::info!("Configuration loaded: {}", config.name);

    // 2. Find and load STL geometry file
    // Resolve geometry file path relative to config file directory
    let config_dir = Path::new(config_path)
        .parent()
        .unwrap_or_else(|| Path::new("."));
    let geometry_path = config_dir.join(&config.geometry_file);
    let geometry_path_str = geometry_path
        .to_str()
        .ok_or("Invalid geometry file path")?;

    tracing::info!("Loading STL geometry: {}", geometry_path_str);
    let triangles = geometry::load_stl(geometry_path_str)?;
    tracing::info!("Loaded {} triangles", triangles.len());

    // 3. Generate SDF
    tracing::info!("Generating signed distance field...");
    let cell_size = config.particle_spacing; // Use particle spacing as grid resolution
    let sdf = geometry::generate_sdf(
        &triangles,
        config.domain.min,
        config.domain.max,
        cell_size,
    );
    tracing::info!(
        "SDF generated: {}x{}x{} grid",
        sdf.dimensions[0],
        sdf.dimensions[1],
        sdf.dimensions[2]
    );

    // 4. Set up domain with fluid and boundary particles
    tracing::info!("Setting up simulation domain...");
    let (fluid_particles, boundary_data) = domain::setup_domain(&config, &sdf);
    tracing::info!(
        "Domain setup: {} fluid particles, {} boundary particles",
        fluid_particles.len(),
        boundary_data.len()
    );

    // 5. Convert BoundaryParticleData to BoundaryParticles
    let mut boundary_particles = BoundaryParticles::new();
    for bp in boundary_data {
        boundary_particles.push(bp.x, bp.y, bp.z, bp.mass, bp.nx, bp.ny, bp.nz);
    }

    // 6. Create CPU kernel
    tracing::info!("Creating CPU simulation kernel...");
    let h = config.smoothing_length();
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

    // 7. Wrap in SimulationRunner
    tracing::info!("Creating simulation runner...");
    let runner = SimulationRunner::new(
        Box::new(kernel),
        h,
        config.speed_of_sound,
        config.cfl_number,
        config.max_timesteps,
        config.max_time,
    );

    tracing::info!("Simulation ready to start");
    Ok(runner)
}

// ===========================================================================
// T069: Result Aggregation Utilities
// ===========================================================================

/// Aggregate force records from multiple subdomains by summing the net force
/// and net moment vectors at each timestep.
///
/// This is used when combining force results from distributed subdomains into
/// a single total force on the geometry.
///
/// # Arguments
/// * `force_records` - A slice of per-subdomain force record vectors. Each
///   inner vector contains force records from one subdomain.
///
/// # Returns
/// A combined vector of force records where forces from all subdomains at the
/// same timestep are summed. If subdomains have different numbers of records,
/// the result length equals the minimum across all subdomains.
pub fn aggregate_force_records(
    force_records: &[Vec<force::SurfaceForce>],
) -> Vec<force::SurfaceForce> {
    if force_records.is_empty() {
        return Vec::new();
    }

    // Find minimum record count
    let min_len = force_records.iter().map(|r| r.len()).min().unwrap_or(0);
    let mut combined = Vec::with_capacity(min_len);

    for t in 0..min_len {
        let mut total = force::SurfaceForce::default();
        for sub_records in force_records {
            total.net_force[0] += sub_records[t].net_force[0];
            total.net_force[1] += sub_records[t].net_force[1];
            total.net_force[2] += sub_records[t].net_force[2];
            total.net_moment[0] += sub_records[t].net_moment[0];
            total.net_moment[1] += sub_records[t].net_moment[1];
            total.net_moment[2] += sub_records[t].net_moment[2];
        }
        combined.push(total);
    }

    combined
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_aggregate_force_records_empty() {
        let result = aggregate_force_records(&[]);
        assert!(result.is_empty());
    }

    #[test]
    fn test_aggregate_force_records_single() {
        let records = vec![vec![
            force::SurfaceForce {
                net_force: [1.0, 2.0, 3.0],
                net_moment: [0.1, 0.2, 0.3],
            },
        ]];
        let result = aggregate_force_records(&records);
        assert_eq!(result.len(), 1);
        assert!((result[0].net_force[0] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_aggregate_force_records_two_subdomains() {
        let records = vec![
            vec![
                force::SurfaceForce {
                    net_force: [1.0, 0.0, 0.0],
                    net_moment: [0.0, 0.0, 0.1],
                },
                force::SurfaceForce {
                    net_force: [2.0, 0.0, 0.0],
                    net_moment: [0.0, 0.0, 0.2],
                },
            ],
            vec![
                force::SurfaceForce {
                    net_force: [0.5, 1.0, 0.0],
                    net_moment: [0.0, 0.0, 0.05],
                },
                force::SurfaceForce {
                    net_force: [1.0, 2.0, 0.0],
                    net_moment: [0.0, 0.0, 0.1],
                },
            ],
        ];

        let result = aggregate_force_records(&records);
        assert_eq!(result.len(), 2);
        // First timestep: [1.0+0.5, 0.0+1.0, 0.0] = [1.5, 1.0, 0.0]
        assert!((result[0].net_force[0] - 1.5).abs() < 1e-6);
        assert!((result[0].net_force[1] - 1.0).abs() < 1e-6);
        // Second timestep: [2.0+1.0, 0.0+2.0, 0.0] = [3.0, 2.0, 0.0]
        assert!((result[1].net_force[0] - 3.0).abs() < 1e-6);
        assert!((result[1].net_force[1] - 2.0).abs() < 1e-6);
    }
}
