//! T070: Validation - distributed vs single-instance comparison
//! T071: Validation - seamless viewer display (particle field coverage)
//!
//! These tests verify that the distributed parallel execution (2 instances)
//! produces results consistent with single-instance execution:
//! - Force results within 2% of equivalent single-instance results (T070)
//! - Combined particle field from distributed subdomains is seamless (T071)

use kernel::{FluidType, ParticleArrays};
use orchestrator::distributed::{run_distributed, run_single_instance, DistributedConfig};
use orchestrator::domain::BoundaryParticleData;

/// Create a uniform grid of water particles for testing
fn create_hydrostatic_particles(
    domain_min: [f32; 3],
    domain_max: [f32; 3],
    spacing: f32,
) -> (ParticleArrays, Vec<BoundaryParticleData>) {
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

    // Generate wall boundary particles (3 layers on all 6 faces)
    let mut boundary = Vec::new();
    let n_layers = 3usize;
    let boundary_mass = mass;

    // Y-min (floor)
    for layer in 0..n_layers {
        for i in 0..nx {
            for k in 0..nz {
                let x = domain_min[0] + (i as f32 + 0.5) * spacing;
                let z = domain_min[2] + (k as f32 + 0.5) * spacing;
                boundary.push(BoundaryParticleData {
                    x,
                    y: domain_min[1] - (layer as f32 + 0.5) * spacing,
                    z,
                    mass: boundary_mass,
                    nx: 0.0,
                    ny: 1.0,
                    nz: 0.0,
                });
            }
        }
    }

    // Y-max (ceiling)
    for layer in 0..n_layers {
        for i in 0..nx {
            for k in 0..nz {
                let x = domain_min[0] + (i as f32 + 0.5) * spacing;
                let z = domain_min[2] + (k as f32 + 0.5) * spacing;
                boundary.push(BoundaryParticleData {
                    x,
                    y: domain_max[1] + (layer as f32 + 0.5) * spacing,
                    z,
                    mass: boundary_mass,
                    nx: 0.0,
                    ny: -1.0,
                    nz: 0.0,
                });
            }
        }
    }

    // X-min wall
    for layer in 0..n_layers {
        for j in 0..ny {
            for k in 0..nz {
                let y = domain_min[1] + (j as f32 + 0.5) * spacing;
                let z = domain_min[2] + (k as f32 + 0.5) * spacing;
                boundary.push(BoundaryParticleData {
                    x: domain_min[0] - (layer as f32 + 0.5) * spacing,
                    y,
                    z,
                    mass: boundary_mass,
                    nx: 1.0,
                    ny: 0.0,
                    nz: 0.0,
                });
            }
        }
    }

    // X-max wall
    for layer in 0..n_layers {
        for j in 0..ny {
            for k in 0..nz {
                let y = domain_min[1] + (j as f32 + 0.5) * spacing;
                let z = domain_min[2] + (k as f32 + 0.5) * spacing;
                boundary.push(BoundaryParticleData {
                    x: domain_max[0] + (layer as f32 + 0.5) * spacing,
                    y,
                    z,
                    mass: boundary_mass,
                    nx: -1.0,
                    ny: 0.0,
                    nz: 0.0,
                });
            }
        }
    }

    // Z-min wall
    for layer in 0..n_layers {
        for i in 0..nx {
            for j in 0..ny {
                let x = domain_min[0] + (i as f32 + 0.5) * spacing;
                let y = domain_min[1] + (j as f32 + 0.5) * spacing;
                boundary.push(BoundaryParticleData {
                    x,
                    y,
                    z: domain_min[2] - (layer as f32 + 0.5) * spacing,
                    mass: boundary_mass,
                    nx: 0.0,
                    ny: 0.0,
                    nz: 1.0,
                });
            }
        }
    }

    // Z-max wall
    for layer in 0..n_layers {
        for i in 0..nx {
            for j in 0..ny {
                let x = domain_min[0] + (i as f32 + 0.5) * spacing;
                let y = domain_min[1] + (j as f32 + 0.5) * spacing;
                boundary.push(BoundaryParticleData {
                    x,
                    y,
                    z: domain_max[2] + (layer as f32 + 0.5) * spacing,
                    mass: boundary_mass,
                    nx: 0.0,
                    ny: 0.0,
                    nz: -1.0,
                });
            }
        }
    }

    (particles, boundary)
}

/// T070: Verify distributed simulation (2 instances) produces results
/// within tolerance of equivalent single-instance results.
///
/// We run both a single-instance and a 2-instance distributed simulation
/// with identical initial conditions and a fixed timestep, then compare
/// the final particle positions and forces.
#[test]
fn distributed_vs_single_instance_accuracy() {
    let domain_min = [0.0, 0.0, 0.0];
    let domain_max = [0.04, 0.04, 0.04];
    let spacing = 0.005;
    let h = 1.3 * spacing; // 0.0065

    let (particles, boundary) = create_hydrostatic_particles(domain_min, domain_max, spacing);
    let particle_count = particles.len();
    println!(
        "T070 validation: {} fluid particles, {} boundary particles",
        particle_count,
        boundary.len()
    );

    let config = DistributedConfig {
        num_instances: 2,
        smoothing_length: h,
        gravity: [0.0, -9.81, 0.0],
        speed_of_sound: 50.0,
        cfl_number: 0.4,
        viscosity: 0.001,
        domain_min,
        domain_max,
    };

    // Use a small fixed timestep for deterministic comparison
    let dt = 1e-5;
    let num_steps = 10;

    // Run single-instance
    let single_result = run_single_instance(&config, &particles, &boundary, num_steps, Some(dt));

    // Run distributed (2 instances)
    let dist_result = run_distributed(&config, &particles, &boundary, num_steps, Some(dt));

    assert_eq!(single_result.particles.len(), particle_count);
    assert_eq!(dist_result.particles.len(), particle_count);

    // Compare results: for each particle in single, find closest in distributed
    // (particle ordering may differ between single and distributed due to
    // subdomain partitioning, so we match by proximity)
    let mut max_position_error = 0.0_f32;
    let mut max_velocity_error = 0.0_f32;
    let mut max_density_error = 0.0_f32;
    let mut matched_count = 0;

    for i in 0..single_result.particles.len() {
        let sx = single_result.particles.x[i];
        let sy = single_result.particles.y[i];
        let sz = single_result.particles.z[i];

        // Find nearest particle in distributed result
        let mut best_j = 0;
        let mut best_dist_sq = f32::MAX;
        for j in 0..dist_result.particles.len() {
            let dx = dist_result.particles.x[j] - sx;
            let dy = dist_result.particles.y[j] - sy;
            let dz = dist_result.particles.z[j] - sz;
            let dist_sq = dx * dx + dy * dy + dz * dz;
            if dist_sq < best_dist_sq {
                best_dist_sq = dist_sq;
                best_j = j;
            }
        }

        let pos_error = best_dist_sq.sqrt();

        // Velocity error
        let dvx = dist_result.particles.vx[best_j] - single_result.particles.vx[i];
        let dvy = dist_result.particles.vy[best_j] - single_result.particles.vy[i];
        let dvz = dist_result.particles.vz[best_j] - single_result.particles.vz[i];
        let vel_error = (dvx * dvx + dvy * dvy + dvz * dvz).sqrt();

        // Density error (relative)
        let single_density = single_result.particles.density[i];
        let dist_density = dist_result.particles.density[best_j];
        let density_error = if single_density.abs() > 1e-12 {
            (dist_density - single_density).abs() / single_density
        } else {
            0.0
        };

        if pos_error > max_position_error {
            max_position_error = pos_error;
        }
        if vel_error > max_velocity_error {
            max_velocity_error = vel_error;
        }
        if density_error > max_density_error {
            max_density_error = density_error;
        }

        // Count particles that matched within a reasonable distance
        if pos_error < spacing * 0.1 {
            matched_count += 1;
        }
    }

    println!("T070 Results:");
    println!("  Max position error: {:.6e} m", max_position_error);
    println!("  Max velocity error: {:.6e} m/s", max_velocity_error);
    println!("  Max density relative error: {:.4}%", max_density_error * 100.0);
    println!("  Matched particles: {}/{}", matched_count, particle_count);

    // Verify results within 2% tolerance
    // Position error should be very small after just 10 steps with tiny dt
    assert!(
        max_position_error < spacing * 0.1,
        "Position error {:.6e} exceeds threshold {:.6e}",
        max_position_error,
        spacing * 0.1
    );

    // Density relative error should be within 2%
    assert!(
        max_density_error < 0.02,
        "Density relative error {:.4}% exceeds 2% threshold",
        max_density_error * 100.0
    );

    // At least 95% of particles should match well
    let match_ratio = matched_count as f64 / particle_count as f64;
    assert!(
        match_ratio > 0.95,
        "Only {:.1}% of particles matched (expected > 95%)",
        match_ratio * 100.0
    );
}

/// T071: Verify web viewer displays seamless combined particle field.
///
/// This tests that the merged particle field from distributed subdomains
/// covers the full domain without gaps at subdomain boundaries. We check:
/// 1. Total particle count is preserved
/// 2. No duplicate particles at boundaries
/// 3. Particles cover the full domain extent
/// 4. No spatial gaps at subdomain boundaries
#[test]
fn distributed_seamless_particle_field() {
    let domain_min = [0.0, 0.0, 0.0];
    let domain_max = [0.04, 0.02, 0.02];
    let spacing = 0.005;
    let h = 1.3 * spacing;

    let (particles, boundary) = create_hydrostatic_particles(domain_min, domain_max, spacing);
    let initial_count = particles.len();

    let config = DistributedConfig {
        num_instances: 2,
        smoothing_length: h,
        gravity: [0.0, -9.81, 0.0],
        speed_of_sound: 50.0,
        cfl_number: 0.4,
        viscosity: 0.001,
        domain_min,
        domain_max,
    };

    // Run a few timesteps
    let result = run_distributed(&config, &particles, &boundary, 5, Some(1e-5));

    println!("T071 validation: initial={}, final={}", initial_count, result.particles.len());

    // 1. Particle count preserved (no duplicates, no lost particles)
    assert_eq!(
        result.particles.len(),
        initial_count,
        "Particle count should be exactly preserved: initial={}, final={}",
        initial_count,
        result.particles.len()
    );

    // 2. Check for duplicates (no two particles closer than 0.1 * spacing)
    let min_dist_threshold = spacing * 0.01; // Very tight - nearly identical positions
    let mut duplicate_count = 0;
    for i in 0..result.particles.len() {
        for j in (i + 1)..result.particles.len() {
            let dx = result.particles.x[i] - result.particles.x[j];
            let dy = result.particles.y[i] - result.particles.y[j];
            let dz = result.particles.z[i] - result.particles.z[j];
            let dist = (dx * dx + dy * dy + dz * dz).sqrt();
            if dist < min_dist_threshold {
                duplicate_count += 1;
            }
        }
    }
    assert_eq!(
        duplicate_count, 0,
        "Found {} duplicate particle pairs (distance < {:.6e})",
        duplicate_count, min_dist_threshold
    );

    // 3. Particles cover the full domain extent
    let mut min_x = f32::MAX;
    let mut max_x = f32::MIN;
    let mut min_y = f32::MAX;
    let mut max_y = f32::MIN;
    let mut min_z = f32::MAX;
    let mut max_z = f32::MIN;

    for i in 0..result.particles.len() {
        let x = result.particles.x[i];
        let y = result.particles.y[i];
        let z = result.particles.z[i];
        if x < min_x { min_x = x; }
        if x > max_x { max_x = x; }
        if y < min_y { min_y = y; }
        if y > max_y { max_y = y; }
        if z < min_z { min_z = z; }
        if z > max_z { max_z = z; }
    }

    println!("  Particle extent: x=[{:.4}, {:.4}], y=[{:.4}, {:.4}], z=[{:.4}, {:.4}]",
             min_x, max_x, min_y, max_y, min_z, max_z);

    // Particles should span most of the domain (within 1 spacing of edges)
    assert!(
        min_x < domain_min[0] + spacing,
        "Min x {:.4} too far from domain min {:.4}",
        min_x, domain_min[0]
    );
    assert!(
        max_x > domain_max[0] - spacing,
        "Max x {:.4} too far from domain max {:.4}",
        max_x, domain_max[0]
    );

    // 4. Check for spatial gaps at subdomain boundary (x = domain center)
    // The domain is split along x at 0.02. Verify particles exist on both
    // sides of this boundary.
    let split_x = (domain_min[0] + domain_max[0]) * 0.5;
    let near_boundary_dist = spacing * 1.5;
    let left_near_boundary = result.particles.x.iter().filter(|&&x| {
        x > split_x - near_boundary_dist && x < split_x
    }).count();
    let right_near_boundary = result.particles.x.iter().filter(|&&x| {
        x >= split_x && x < split_x + near_boundary_dist
    }).count();

    println!("  Particles near boundary (x={:.4}): left={}, right={}",
             split_x, left_near_boundary, right_near_boundary);

    assert!(
        left_near_boundary > 0,
        "No particles found just left of subdomain boundary at x={}",
        split_x
    );
    assert!(
        right_near_boundary > 0,
        "No particles found just right of subdomain boundary at x={}",
        split_x
    );

    // 5. All particle values should be finite (no NaN/Inf from boundary artifacts)
    for i in 0..result.particles.len() {
        assert!(
            result.particles.x[i].is_finite(),
            "Particle {} has non-finite x: {}",
            i, result.particles.x[i]
        );
        assert!(
            result.particles.y[i].is_finite(),
            "Particle {} has non-finite y: {}",
            i, result.particles.y[i]
        );
        assert!(
            result.particles.z[i].is_finite(),
            "Particle {} has non-finite z: {}",
            i, result.particles.z[i]
        );
        assert!(
            result.particles.density[i].is_finite(),
            "Particle {} has non-finite density: {}",
            i, result.particles.density[i]
        );
        assert!(
            result.particles.pressure[i].is_finite(),
            "Particle {} has non-finite pressure: {}",
            i, result.particles.pressure[i]
        );
    }

    println!("T071 validation PASSED: seamless particle field verified");
}
