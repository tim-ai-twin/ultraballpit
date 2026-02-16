//! Reference test framework for SPH fluid simulation validation
//!
//! This crate provides a comprehensive testing framework for validating
//! the physical accuracy of SPH fluid simulations through reference tests.

#[cfg(test)]
mod tests;

use kernel::{CpuKernel, ErrorMetrics, FluidType, ParticleArrays, SimulationKernel};
use orchestrator::config::SimulationConfig;
use orchestrator::{domain, geometry};
use std::path::Path;

/// Expected result criteria for a reference test
#[derive(Debug, Clone)]
pub struct ExpectedResult {
    /// Particle position bounds validation
    pub position_bounds: Option<PositionBoundsCheck>,
    /// Pressure validation at specific locations
    pub pressure_check: Option<PressureCheck>,
    /// Pressure uniformity check
    pub pressure_uniformity: Option<PressureUniformityCheck>,
    /// Conservation metrics validation
    pub conservation: Option<ConservationCheck>,
    /// Settling check (particles near floor)
    pub settling: Option<SettlingCheck>,
}

/// Check that particles remain within specified bounds
#[derive(Debug, Clone)]
pub struct PositionBoundsCheck {
    /// Minimum allowed position [x, y, z]
    pub min: [f32; 3],
    /// Maximum allowed position [x, y, z]
    pub max: [f32; 3],
}

/// Check pressure at specific regions
#[derive(Debug, Clone)]
pub struct PressureCheck {
    /// Expected bottom pressure (Pa)
    pub expected_bottom: f32,
    /// Relative tolerance (0.0 to 1.0)
    pub tolerance: f32,
}

/// Check pressure variation across the domain
#[derive(Debug, Clone)]
pub struct PressureUniformityCheck {
    /// Maximum allowed relative pressure variation (0.0 to 1.0)
    pub max_variation: f32,
}

/// Check conservation metrics
#[derive(Debug, Clone)]
pub struct ConservationCheck {
    /// Maximum allowed mass conservation error (0.0 to 1.0)
    pub max_mass_error: f32,
    /// Maximum allowed energy conservation error (0.0 to 1.0)
    pub max_energy_error: f32,
}

/// Check that particles have settled near the floor
#[derive(Debug, Clone)]
pub struct SettlingCheck {
    /// Floor Y position
    pub floor_y: f32,
    /// Maximum distance from floor (in particle diameters)
    pub max_distance: f32,
    /// Particle spacing for diameter calculation
    pub particle_spacing: f32,
}

/// Result of running a reference test
#[derive(Debug)]
pub struct TestResult {
    /// Test name
    pub name: String,
    /// Whether test passed
    pub passed: bool,
    /// Individual check results
    pub checks: Vec<CheckResult>,
    /// Final error metrics
    pub error_metrics: ErrorMetrics,
    /// Number of timesteps executed
    pub timesteps: usize,
    /// Simulated time (seconds)
    pub sim_time: f64,
}

/// Result of an individual validation check
#[derive(Debug)]
pub struct CheckResult {
    /// Check name
    pub name: String,
    /// Whether check passed
    pub passed: bool,
    /// Error message if failed
    pub message: Option<String>,
}

/// A reference test case
pub struct ReferenceTest {
    /// Test name
    pub name: String,
    /// Path to configuration file
    pub config_path: String,
    /// Number of timesteps to run
    pub timesteps: usize,
    /// Expected results to validate
    pub expected: ExpectedResult,
}

impl ReferenceTest {
    /// Run the reference test and return results
    pub fn run(&self) -> Result<TestResult, String> {
        tracing::info!("Running reference test: {}", self.name);

        // Initialize kernel (idempotent)
        kernel::simulation::init();

        // Load configuration
        let config = SimulationConfig::load(&self.config_path)?;

        // Resolve geometry path relative to config directory
        let config_path = Path::new(&self.config_path);
        let config_dir = config_path.parent().ok_or("Invalid config path")?;
        let geometry_path = config_dir.join(&config.geometry_file);
        let mesh = geometry::load_stl(
            geometry_path
                .to_str()
                .ok_or("Invalid geometry path")?,
        )?;
        let sdf = geometry::mesh_to_sdf(&mesh, &config.domain, 0.5 * config.particle_spacing);

        // Setup domain
        let (fluid_particles, boundary_data) = domain::setup_domain(&config, &sdf);
        let mut boundary_particles = kernel::BoundaryParticles::new();
        for b in boundary_data {
            boundary_particles.push(b.x, b.y, b.z, b.mass, b.nx, b.ny, b.nz);
        }

        // Calculate smoothing length
        let h = config.smoothing_length();

        tracing::info!(
            "Initialized: {} particles, h={}",
            fluid_particles.len(),
            h,
        );

        // Create kernel
        let mut kernel = CpuKernel::new(
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

        // Run simulation with adaptive timestep
        tracing::info!("Running {} timesteps...", self.timesteps);
        let mut sim_time = 0.0_f64;
        for step in 0..self.timesteps {
            // Compute adaptive timestep from current state
            let dt = kernel::sph::compute_timestep(
                kernel.particles(),
                h,
                config.speed_of_sound,
                config.cfl_number,
            );
            kernel.step(dt);
            sim_time += dt as f64;

            // Log progress every 10% of steps
            if (step + 1) % (self.timesteps / 10).max(1) == 0 {
                let progress = ((step + 1) as f32 / self.timesteps as f32) * 100.0;
                tracing::info!("Progress: {:.0}% ({}/{})", progress, step + 1, self.timesteps);
            }
        }
        tracing::info!("Simulation complete: {} steps, {:.6}s simulated", self.timesteps, sim_time);

        // Get final state
        let particles = kernel.particles();
        let error_metrics = kernel.error_metrics();

        // Validate results
        let mut checks = Vec::new();
        let mut all_passed = true;

        // Check position bounds
        if let Some(ref bounds) = self.expected.position_bounds {
            let check = validate_position_bounds(particles, bounds);
            all_passed &= check.passed;
            checks.push(check);
        }

        // Check pressure
        if let Some(ref pressure_check) = self.expected.pressure_check {
            let check = validate_pressure(particles, pressure_check, &config);
            all_passed &= check.passed;
            checks.push(check);
        }

        // Check pressure uniformity
        if let Some(ref uniformity) = self.expected.pressure_uniformity {
            let check = validate_pressure_uniformity(particles, uniformity);
            all_passed &= check.passed;
            checks.push(check);
        }

        // Check conservation
        if let Some(ref conservation) = self.expected.conservation {
            let check = validate_conservation(&error_metrics, conservation);
            all_passed &= check.passed;
            checks.push(check);
        }

        // Check settling
        if let Some(ref settling) = self.expected.settling {
            let check = validate_settling(particles, settling);
            all_passed &= check.passed;
            checks.push(check);
        }

        Ok(TestResult {
            name: self.name.clone(),
            passed: all_passed,
            checks,
            error_metrics,
            timesteps: self.timesteps,
            sim_time,
        })
    }
}

/// Validate that particles remain within specified bounds
fn validate_position_bounds(
    particles: &ParticleArrays,
    bounds: &PositionBoundsCheck,
) -> CheckResult {
    let mut violations = 0;
    let mut max_violation = 0.0_f32;

    for i in 0..particles.len() {
        let pos = [particles.x[i], particles.y[i], particles.z[i]];

        for axis in 0..3 {
            if pos[axis] < bounds.min[axis] {
                violations += 1;
                let violation = bounds.min[axis] - pos[axis];
                max_violation = max_violation.max(violation);
            }
            if pos[axis] > bounds.max[axis] {
                violations += 1;
                let violation = pos[axis] - bounds.max[axis];
                max_violation = max_violation.max(violation);
            }
        }
    }

    if violations == 0 {
        CheckResult {
            name: "Position Bounds".to_string(),
            passed: true,
            message: None,
        }
    } else {
        CheckResult {
            name: "Position Bounds".to_string(),
            passed: false,
            message: Some(format!(
                "{} particles out of bounds (max violation: {:.6} m)",
                violations, max_violation
            )),
        }
    }
}

/// Validate pressure at bottom of domain
fn validate_pressure(
    particles: &ParticleArrays,
    check: &PressureCheck,
    config: &SimulationConfig,
) -> CheckResult {
    // Find water particles in bottom 10% of domain
    let domain_height = config.domain.max[1] - config.domain.min[1];
    let bottom_threshold = config.domain.min[1] + 0.1 * domain_height;

    let mut bottom_pressures = Vec::new();
    for i in 0..particles.len() {
        if particles.fluid_type[i] == FluidType::Water && particles.y[i] < bottom_threshold {
            bottom_pressures.push(particles.pressure[i]);
        }
    }

    if bottom_pressures.is_empty() {
        return CheckResult {
            name: "Bottom Pressure".to_string(),
            passed: false,
            message: Some("No water particles found in bottom region".to_string()),
        };
    }

    let avg_bottom_pressure: f32 = bottom_pressures.iter().sum::<f32>() / bottom_pressures.len() as f32;
    let error = (avg_bottom_pressure - check.expected_bottom).abs() / check.expected_bottom.abs().max(1.0);

    if error <= check.tolerance {
        CheckResult {
            name: "Bottom Pressure".to_string(),
            passed: true,
            message: Some(format!(
                "Expected: {:.1} Pa, Got: {:.1} Pa (error: {:.1}%)",
                check.expected_bottom,
                avg_bottom_pressure,
                error * 100.0
            )),
        }
    } else {
        CheckResult {
            name: "Bottom Pressure".to_string(),
            passed: false,
            message: Some(format!(
                "Expected: {:.1} Pa, Got: {:.1} Pa (error: {:.1}%, tolerance: {:.1}%)",
                check.expected_bottom,
                avg_bottom_pressure,
                error * 100.0,
                check.tolerance * 100.0
            )),
        }
    }
}

/// Validate pressure uniformity across domain
fn validate_pressure_uniformity(
    particles: &ParticleArrays,
    check: &PressureUniformityCheck,
) -> CheckResult {
    if particles.len() == 0 {
        return CheckResult {
            name: "Pressure Uniformity".to_string(),
            passed: false,
            message: Some("No particles".to_string()),
        };
    }

    // Calculate mean pressure
    let mean_pressure: f32 = particles.pressure.iter().sum::<f32>() / particles.len() as f32;

    // Calculate max deviation from mean
    let mut max_deviation = 0.0_f32;
    for &p in &particles.pressure {
        let deviation = (p - mean_pressure).abs();
        max_deviation = max_deviation.max(deviation);
    }

    let variation = if mean_pressure.abs() > 1.0 {
        max_deviation / mean_pressure.abs()
    } else {
        max_deviation
    };

    if variation <= check.max_variation {
        CheckResult {
            name: "Pressure Uniformity".to_string(),
            passed: true,
            message: Some(format!(
                "Mean: {:.1} Pa, Max variation: {:.1}%",
                mean_pressure,
                variation * 100.0
            )),
        }
    } else {
        CheckResult {
            name: "Pressure Uniformity".to_string(),
            passed: false,
            message: Some(format!(
                "Mean: {:.1} Pa, Max variation: {:.1}% (limit: {:.1}%)",
                mean_pressure,
                variation * 100.0,
                check.max_variation * 100.0
            )),
        }
    }
}

/// Validate conservation metrics
fn validate_conservation(
    metrics: &ErrorMetrics,
    check: &ConservationCheck,
) -> CheckResult {
    let mass_ok = metrics.mass_conservation <= check.max_mass_error;
    let energy_ok = metrics.energy_conservation <= check.max_energy_error;

    if mass_ok && energy_ok {
        CheckResult {
            name: "Conservation".to_string(),
            passed: true,
            message: Some(format!(
                "Mass: {:.3}%, Energy: {:.1}%",
                metrics.mass_conservation * 100.0,
                metrics.energy_conservation * 100.0
            )),
        }
    } else {
        let mut issues = Vec::new();
        if !mass_ok {
            issues.push(format!(
                "Mass: {:.3}% (limit: {:.3}%)",
                metrics.mass_conservation * 100.0,
                check.max_mass_error * 100.0
            ));
        }
        if !energy_ok {
            issues.push(format!(
                "Energy: {:.1}% (limit: {:.1}%)",
                metrics.energy_conservation * 100.0,
                check.max_energy_error * 100.0
            ));
        }
        CheckResult {
            name: "Conservation".to_string(),
            passed: false,
            message: Some(issues.join(", ")),
        }
    }
}

/// Validate that particles have settled near the floor
fn validate_settling(particles: &ParticleArrays, check: &SettlingCheck) -> CheckResult {
    let max_allowed_height = check.floor_y + check.max_distance * check.particle_spacing;

    let mut settled_count = 0;
    let mut unsettled_count = 0;
    let mut max_height = check.floor_y;

    for i in 0..particles.len() {
        if particles.y[i] <= max_allowed_height {
            settled_count += 1;
        } else {
            unsettled_count += 1;
        }
        max_height = max_height.max(particles.y[i]);
    }

    // All particles should be settled
    if unsettled_count == 0 {
        CheckResult {
            name: "Settling".to_string(),
            passed: true,
            message: Some(format!(
                "All {} particles settled (max height: {:.6} m, limit: {:.6} m)",
                settled_count, max_height, max_allowed_height
            )),
        }
    } else {
        CheckResult {
            name: "Settling".to_string(),
            passed: false,
            message: Some(format!(
                "{} / {} particles not settled (max height: {:.6} m, limit: {:.6} m)",
                unsettled_count,
                particles.len(),
                max_height,
                max_allowed_height
            )),
        }
    }
}

impl TestResult {
    /// Print a summary of the test result
    pub fn print_summary(&self) {
        println!("\n{}", "=".repeat(80));
        println!("Test: {}", self.name);
        println!("{}", "=".repeat(80));
        println!("Status: {}", if self.passed { "PASSED" } else { "FAILED" });
        println!("Timesteps: {}", self.timesteps);
        println!("Simulated time: {:.6} s", self.sim_time);
        println!("\nError Metrics:");
        println!("  Max density variation: {:.2}%", self.error_metrics.max_density_variation * 100.0);
        println!("  Mass conservation: {:.3}%", self.error_metrics.mass_conservation * 100.0);
        println!("  Energy conservation: {:.1}%", self.error_metrics.energy_conservation * 100.0);
        println!("\nValidation Checks:");
        for check in &self.checks {
            let status = if check.passed { "PASS" } else { "FAIL" };
            print!("  [{}] {}", status, check.name);
            if let Some(ref msg) = check.message {
                print!(" - {}", msg);
            }
            println!();
        }
        println!("{}", "=".repeat(80));
    }
}
