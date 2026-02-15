//! T024: Two-particle symmetry test.
//!
//! Verifies Newton's 3rd law (forces equal and opposite) and momentum
//! conservation for a simple two-particle system.

use kernel::{
    BoundaryParticles, CpuKernel, FluidType, ParticleArrays, SimulationKernel,
};

/// Create a two-particle system with particles separated by distance `h` along the x-axis.
fn setup_two_particles(h: f32) -> (ParticleArrays, BoundaryParticles) {
    let mut particles = ParticleArrays::new();
    let mass = 0.001; // 1 gram
    let rest_density = 1000.0;
    let temp = 293.15;

    // Particle 0 at origin
    particles.push_particle(0.0, 0.0, 0.0, mass, rest_density, temp, FluidType::Water);
    // Particle 1 at distance h along x-axis
    particles.push_particle(h, 0.0, 0.0, mass, rest_density, temp, FluidType::Water);

    let boundary = BoundaryParticles::new();
    (particles, boundary)
}

#[test]
fn forces_equal_and_opposite() {
    let h = 0.05;
    let speed_of_sound = 20.0;
    let cfl = 0.3;
    let gravity = [0.0, 0.0, 0.0]; // No gravity for symmetry test
    let viscosity = 1.0;

    let (particles, boundary) = setup_two_particles(h);

    let domain_min = [-1.0, -1.0, -1.0];
    let domain_max = [1.0, 1.0, 1.0];

    let mut kernel = CpuKernel::new(
        particles,
        boundary,
        h,
        gravity,
        speed_of_sound,
        cfl,
        viscosity,
        domain_min,
        domain_max,
    );

    // Run one step with a small dt
    let dt = 0.0001;
    kernel.step(dt);

    let p = kernel.particles();

    // After one step, forces on particle 0 and 1 should be equal and opposite.
    // Since this is after the step (Velocity Verlet), we check accelerations.
    let ax0 = p.ax[0];
    let ay0 = p.ay[0];
    let az0 = p.az[0];
    let ax1 = p.ax[1];
    let ay1 = p.ay[1];
    let az1 = p.az[1];

    // Forces should be equal and opposite (Newton's 3rd law)
    let tol = 1.0e-6;
    assert!(
        (ax0 + ax1).abs() < tol,
        "ax not equal and opposite: ax0={ax0}, ax1={ax1}, sum={}",
        ax0 + ax1
    );
    assert!(
        (ay0 + ay1).abs() < tol,
        "ay not equal and opposite: ay0={ay0}, ay1={ay1}"
    );
    assert!(
        (az0 + az1).abs() < tol,
        "az not equal and opposite: az0={az0}, az1={az1}"
    );

    // The force should be along the x-axis only (by symmetry)
    assert!(
        ay0.abs() < tol,
        "ay0 should be ~0 for x-axis alignment, got {ay0}"
    );
    assert!(
        az0.abs() < tol,
        "az0 should be ~0 for x-axis alignment, got {az0}"
    );
}

#[test]
fn momentum_conserved() {
    let h = 0.05;
    let speed_of_sound = 20.0;
    let cfl = 0.3;
    let gravity = [0.0, 0.0, 0.0]; // No gravity so momentum is conserved
    let viscosity = 1.0;

    let (particles, boundary) = setup_two_particles(h);
    let mass = particles.mass[0]; // both have same mass

    let domain_min = [-1.0, -1.0, -1.0];
    let domain_max = [1.0, 1.0, 1.0];

    let mut kernel = CpuKernel::new(
        particles,
        boundary,
        h,
        gravity,
        speed_of_sound,
        cfl,
        viscosity,
        domain_min,
        domain_max,
    );

    // Initial momentum should be zero (both particles at rest)
    let p = kernel.particles();
    let initial_px: f32 = (0..p.len()).map(|i| p.mass[i] * p.vx[i]).sum();
    let initial_py: f32 = (0..p.len()).map(|i| p.mass[i] * p.vy[i]).sum();
    let initial_pz: f32 = (0..p.len()).map(|i| p.mass[i] * p.vz[i]).sum();

    assert!(initial_px.abs() < 1.0e-10, "initial px should be 0");

    // Run several steps
    let dt = 0.0001;
    for _ in 0..10 {
        kernel.step(dt);
    }

    let p = kernel.particles();
    let final_px: f32 = (0..p.len()).map(|i| p.mass[i] * p.vx[i]).sum();
    let final_py: f32 = (0..p.len()).map(|i| p.mass[i] * p.vy[i]).sum();
    let final_pz: f32 = (0..p.len()).map(|i| p.mass[i] * p.vz[i]).sum();

    // Momentum should be conserved (no external forces)
    let tol = mass * 1.0e-4; // relative to particle momentum scale
    assert!(
        (final_px - initial_px).abs() < tol,
        "px not conserved: initial={initial_px}, final={final_px}, diff={}",
        (final_px - initial_px).abs()
    );
    assert!(
        (final_py - initial_py).abs() < tol,
        "py not conserved: initial={initial_py}, final={final_py}"
    );
    assert!(
        (final_pz - initial_pz).abs() < tol,
        "pz not conserved: initial={initial_pz}, final={final_pz}"
    );
}
