//! GPU backend reference tests (T105)
//!
//! Re-runs the Phase 4 reference tests using GpuKernel instead of CpuKernel,
//! verifying that the GPU backend produces physically correct results.
//!
//! Gated behind `#[cfg(feature = "gpu")]`.

#![cfg(feature = "gpu")]

use kernel::{
    BoundaryParticles, FluidType, ParticleArrays, SimulationKernel,
};
use orchestrator::config::SimulationConfig;
use orchestrator::{domain, geometry};
use std::path::Path;

/// Resolve a path relative to the project root (two levels up from this crate)
fn project_path(relative: &str) -> String {
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let project_root = std::path::Path::new(manifest_dir)
        .parent()
        .and_then(|p| p.parent())
        .expect("Could not find project root");
    project_root.join(relative).to_string_lossy().to_string()
}

/// Helper: create a GpuKernel from a config path.
/// Returns None if the GPU is not available (test is skipped).
fn create_gpu_kernel_from_config(
    config_path: &str,
) -> Option<(SimulationConfig, Box<dyn SimulationKernel>)> {
    let config = SimulationConfig::load(config_path).expect("Failed to load config");

    let config_file_path = Path::new(config_path);
    let config_dir = config_file_path.parent().expect("Invalid config path");
    let geometry_path = config_dir.join(&config.geometry_file);
    let mesh = geometry::load_stl(geometry_path.to_str().expect("Invalid geometry path"))
        .expect("Failed to load STL");
    let sdf = geometry::mesh_to_sdf(&mesh, &config.domain, 0.5 * config.particle_spacing);

    let (fluid_particles, boundary_data) = domain::setup_domain(&config, &sdf);
    let mut boundary_particles = BoundaryParticles::new();
    for b in boundary_data {
        boundary_particles.push(b.x, b.y, b.z, b.mass, b.nx, b.ny, b.nz);
    }

    let h = config.smoothing_length();

    match kernel::GpuKernel::new(
        fluid_particles,
        boundary_particles,
        h,
        config.gravity,
        config.speed_of_sound,
        config.cfl_number,
        config.viscosity,
        config.domain.min,
        config.domain.max,
    ) {
        Ok(gpu) => Some((config, Box::new(gpu))),
        Err(e) => {
            eprintln!("Skipping GPU reference test: {e}");
            None
        }
    }
}

/// T105: Gravity settling with GPU backend
#[test]
fn gpu_gravity_settling() {
    let config_path = project_path("configs/water-box-1cm.json");
    let (config, mut kernel) = match create_gpu_kernel_from_config(&config_path) {
        Some(pair) => pair,
        None => return,
    };

    let h = config.smoothing_length();
    let dt_fixed = 1e-5_f32;
    let n_steps = 5000;

    println!("GPU gravity settling: {} particles, {} steps", kernel.particle_count(), n_steps);

    for step in 0..n_steps {
        kernel.step(dt_fixed);
        if (step + 1) % 1000 == 0 {
            println!("  Step {}/{}", step + 1, n_steps);
        }
    }

    let particles = kernel.particles();
    let metrics = kernel.error_metrics();

    // Check: all particles within bounds (with small tolerance)
    let tol = 0.001;
    for i in 0..particles.len() {
        assert!(
            particles.x[i] >= config.domain.min[0] - tol
                && particles.x[i] <= config.domain.max[0] + tol
                && particles.y[i] >= config.domain.min[1] - tol
                && particles.y[i] <= config.domain.max[1] + tol
                && particles.z[i] >= config.domain.min[2] - tol
                && particles.z[i] <= config.domain.max[2] + tol,
            "Particle {} out of bounds: ({}, {}, {})",
            i, particles.x[i], particles.y[i], particles.z[i]
        );
    }

    // Check: mass conservation
    assert!(
        metrics.mass_conservation < 0.001,
        "Mass conservation error: {}",
        metrics.mass_conservation
    );

    println!("GPU gravity settling PASSED");
    println!(
        "  Max density variation: {:.2}%, Mass conservation: {:.6}%, Energy: {:.1}%",
        metrics.max_density_variation * 100.0,
        metrics.mass_conservation * 100.0,
        metrics.energy_conservation * 100.0
    );
}

/// T105: GPU results match CPU results for water box
#[test]
fn gpu_matches_cpu_water_box() {
    let config_path = project_path("configs/water-box-1cm.json");
    let config = SimulationConfig::load(&config_path).expect("Failed to load config");

    let config_file_path = Path::new(&config_path);
    let config_dir = config_file_path.parent().expect("Invalid config path");
    let geometry_path = config_dir.join(&config.geometry_file);
    let mesh = geometry::load_stl(geometry_path.to_str().expect("Invalid geometry path"))
        .expect("Failed to load STL");
    let sdf = geometry::mesh_to_sdf(&mesh, &config.domain, 0.5 * config.particle_spacing);

    let (fluid_particles, boundary_data) = domain::setup_domain(&config, &sdf);
    let mut boundary_particles = BoundaryParticles::new();
    for b in &boundary_data {
        boundary_particles.push(b.x, b.y, b.z, b.mass, b.nx, b.ny, b.nz);
    }

    let h = config.smoothing_length();

    // Create CPU kernel
    let mut cpu_kernel = kernel::CpuKernel::new(
        fluid_particles.clone(),
        boundary_particles.clone(),
        h,
        config.gravity,
        config.speed_of_sound,
        config.cfl_number,
        config.viscosity,
        config.domain.min,
        config.domain.max,
    );

    // Create GPU kernel
    let mut gpu_kernel = match kernel::GpuKernel::new(
        fluid_particles,
        boundary_particles,
        h,
        config.gravity,
        config.speed_of_sound,
        config.cfl_number,
        config.viscosity,
        config.domain.min,
        config.domain.max,
    ) {
        Ok(k) => k,
        Err(e) => {
            eprintln!("Skipping GPU/CPU comparison: {e}");
            return;
        }
    };

    // Run both for 500 steps
    let dt = 1e-5_f32;
    let n_steps = 500;
    println!("GPU/CPU comparison: {} particles, {} steps", cpu_kernel.particle_count(), n_steps);

    for _ in 0..n_steps {
        cpu_kernel.step(dt);
        gpu_kernel.step(dt);
    }

    let cpu_p = cpu_kernel.particles();
    let gpu_p = gpu_kernel.particles();

    // Compare density metrics
    let cpu_metrics = cpu_kernel.error_metrics();
    let gpu_metrics = gpu_kernel.error_metrics();

    println!(
        "  CPU: density_var={:.4}%, mass={:.6}%, energy={:.1}%",
        cpu_metrics.max_density_variation * 100.0,
        cpu_metrics.mass_conservation * 100.0,
        cpu_metrics.energy_conservation * 100.0
    );
    println!(
        "  GPU: density_var={:.4}%, mass={:.6}%, energy={:.1}%",
        gpu_metrics.max_density_variation * 100.0,
        gpu_metrics.mass_conservation * 100.0,
        gpu_metrics.energy_conservation * 100.0
    );

    // Compute max position error
    let n = cpu_p.len();
    let mut max_pos_err = 0.0_f32;
    for i in 0..n {
        let dx = cpu_p.x[i] - gpu_p.x[i];
        let dy = cpu_p.y[i] - gpu_p.y[i];
        let dz = cpu_p.z[i] - gpu_p.z[i];
        let err = (dx * dx + dy * dy + dz * dz).sqrt();
        max_pos_err = max_pos_err.max(err);
    }
    println!("  Max position error: {:.6e} m", max_pos_err);

    // Position tolerance: 1e-4 m
    assert!(
        max_pos_err < 1e-4,
        "Position error too large: {:.6e}",
        max_pos_err
    );

    // Density metric should be in similar ballpark
    assert!(
        (gpu_metrics.max_density_variation - cpu_metrics.max_density_variation).abs() < 0.1,
        "Density variation differs too much: GPU={:.4}, CPU={:.4}",
        gpu_metrics.max_density_variation,
        cpu_metrics.max_density_variation
    );

    println!("GPU/CPU comparison PASSED");
}
