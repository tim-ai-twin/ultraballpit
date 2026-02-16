//! GPU vs CPU parity test (T103)
//!
//! Runs the same water-box-1cm scenario on both CpuKernel and GpuKernel,
//! then compares final particle positions and densities to ensure they match
//! within acceptable tolerances.
//!
//! Gated behind `#[cfg(feature = "gpu")]` so it only runs when the GPU feature
//! is enabled.

#![cfg(feature = "gpu")]

use kernel::{
    BoundaryParticles, CpuKernel, FluidType, GpuKernel, ParticleArrays, SimulationKernel,
};

/// Create a small water-box test case with known particle layout.
fn create_water_box() -> (ParticleArrays, BoundaryParticles, f32, [f32; 3], f32, f32, f32, [f32; 3], [f32; 3]) {
    let spacing = 0.002; // 2mm spacing
    let domain_min = [0.0_f32, 0.0, 0.0];
    let domain_max = [0.01_f32, 0.01, 0.01];
    let h = 1.3 * spacing;
    let gravity = [0.0_f32, -9.81, 0.0];
    let speed_of_sound = 10.0_f32;
    let cfl_number = 0.4_f32;
    let viscosity = 0.001_f32;

    let mut particles = ParticleArrays::new();

    // Fill domain with water particles
    let rest_density = 1000.0_f32;
    let particle_volume = spacing * spacing * spacing;
    let particle_mass = rest_density * particle_volume;

    let mut x = domain_min[0] + 0.5 * spacing;
    while x < domain_max[0] {
        let mut y = domain_min[1] + 0.5 * spacing;
        while y < domain_max[1] {
            let mut z = domain_min[2] + 0.5 * spacing;
            while z < domain_max[2] {
                particles.push_particle(
                    x,
                    y,
                    z,
                    particle_mass,
                    rest_density,
                    293.15,
                    FluidType::Water,
                );
                z += spacing;
            }
            y += spacing;
        }
        x += spacing;
    }

    // Create simple boundary particles (just a few on the bottom face)
    let mut boundary = BoundaryParticles::new();
    let mut bx = domain_min[0];
    while bx <= domain_max[0] {
        let mut bz = domain_min[2];
        while bz <= domain_max[2] {
            boundary.push(bx, domain_min[1], bz, particle_mass, 0.0, 1.0, 0.0);
            bz += spacing;
        }
        bx += spacing;
    }

    (particles, boundary, h, gravity, speed_of_sound, cfl_number, viscosity, domain_min, domain_max)
}

#[test]
fn gpu_cpu_parity_100_steps() {
    let (particles, boundary, h, gravity, speed_of_sound, cfl_number, viscosity, domain_min, domain_max) =
        create_water_box();

    let n = particles.len();
    println!("Parity test: {} particles", n);

    // Create CPU kernel
    let mut cpu_kernel = CpuKernel::new(
        particles.clone(),
        boundary.clone(),
        h,
        gravity,
        speed_of_sound,
        cfl_number,
        viscosity,
        domain_min,
        domain_max,
    );

    // Create GPU kernel
    let gpu_result = GpuKernel::new(
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

    let mut gpu_kernel = match gpu_result {
        Ok(k) => k,
        Err(e) => {
            eprintln!("Skipping GPU parity test: {e}");
            return;
        }
    };

    // Run both for 100 steps with fixed dt
    let dt = 1.0e-5_f32;
    let n_steps = 100;

    for _ in 0..n_steps {
        cpu_kernel.step(dt);
        gpu_kernel.step(dt);
    }

    let cpu_p = cpu_kernel.particles();
    let gpu_p = gpu_kernel.particles();

    assert_eq!(cpu_p.len(), gpu_p.len(), "Particle counts differ");

    // Compare positions
    let mut max_pos_error = 0.0_f32;
    for i in 0..n {
        let dx = (cpu_p.x[i] - gpu_p.x[i]).abs();
        let dy = (cpu_p.y[i] - gpu_p.y[i]).abs();
        let dz = (cpu_p.z[i] - gpu_p.z[i]).abs();
        let err = (dx * dx + dy * dy + dz * dz).sqrt();
        if err > max_pos_error {
            max_pos_error = err;
        }
    }

    // Compare densities
    let mut max_density_error = 0.0_f32;
    for i in 0..n {
        let rest = 1000.0_f32; // water
        let rel_err = (cpu_p.density[i] - gpu_p.density[i]).abs() / rest;
        if rel_err > max_density_error {
            max_density_error = rel_err;
        }
    }

    println!("Max position error: {:.6e}", max_pos_error);
    println!("Max density relative error: {:.6e}", max_density_error);

    // Tolerances: GPU floating point may differ due to ordering, fused operations, etc.
    // Position tolerance: 1e-4 meters (relative to domain size 0.01m = 1% which is generous)
    assert!(
        max_pos_error < 1e-4,
        "Position error too large: {:.6e} > 1e-4",
        max_pos_error
    );

    // Density tolerance: 0.1% relative
    assert!(
        max_density_error < 0.001,
        "Density error too large: {:.6e} > 0.001",
        max_density_error
    );
}
