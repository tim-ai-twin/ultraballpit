//! T025: Kernel normalization test via SPH density summation.
//!
//! Places 27 particles on a 3x3x3 lattice at rest density spacing (spacing = h/1.3).
//! Computes density for the center particle and verifies it matches rho_0 = 1000 kg/m^3
//! within 2%.

use kernel::{NeighborGrid, ParticleArrays, FluidType};
use kernel::sph::compute_density;

#[test]
fn density_at_rest_lattice_matches_rho0() {
    let h = 0.05_f32;
    let spacing = h / 1.3;
    let rest_density = 1000.0_f32;

    // Particle mass for rest density: m = rho_0 * spacing^3
    let mass = rest_density * spacing * spacing * spacing;

    let mut particles = ParticleArrays::new();

    // Create a 3x3x3 lattice centered at origin
    let offsets = [-1i32, 0, 1];
    for &iz in &offsets {
        for &iy in &offsets {
            for &ix in &offsets {
                let px = ix as f32 * spacing;
                let py = iy as f32 * spacing;
                let pz = iz as f32 * spacing;
                particles.push_particle(px, py, pz, mass, rest_density, 293.15, FluidType::Water);
            }
        }
    }

    assert_eq!(particles.len(), 27);

    // The center particle is at index 13 (offsets (0,0,0))
    // Verify it's at origin
    let center = 13;
    assert!(particles.x[center].abs() < 1.0e-6);
    assert!(particles.y[center].abs() < 1.0e-6);
    assert!(particles.z[center].abs() < 1.0e-6);

    // Build neighbor grid
    let support_radius = 2.0 * h;
    let domain_extent = 2.0 * spacing + support_radius;
    let domain_min = [-domain_extent; 3];
    let domain_max = [domain_extent; 3];
    let mut grid = NeighborGrid::new(support_radius, domain_min, domain_max);
    grid.update(&particles.x, &particles.y, &particles.z);

    // Compute density
    let no_boundary: Vec<f32> = Vec::new();
    compute_density(
        &mut particles,
        &no_boundary,
        &no_boundary,
        &no_boundary,
        &no_boundary,
        &grid,
        h,
    );

    let computed_density = particles.density[center];
    let relative_error = (computed_density - rest_density).abs() / rest_density;

    eprintln!(
        "Center particle density: {computed_density:.2} kg/m^3 (expected {rest_density:.2}), \
         relative error: {:.4}%",
        relative_error * 100.0
    );

    assert!(
        relative_error < 0.02,
        "Density should match rho_0 within 2%, got {computed_density:.2} (error {:.2}%)",
        relative_error * 100.0
    );
}
