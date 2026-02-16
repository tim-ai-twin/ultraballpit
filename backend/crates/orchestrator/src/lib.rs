//! Orchestration Layer
//!
//! This crate provides orchestration for the SPH simulation, including:
//! - STL file parsing and mesh processing
//! - SDF (Signed Distance Field) generation
//! - Simulation state management
//! - Simulation runner with lifecycle management

#![warn(missing_docs)]

pub mod config;
pub mod geometry;
pub mod domain;
pub mod runner;
pub mod force;

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
