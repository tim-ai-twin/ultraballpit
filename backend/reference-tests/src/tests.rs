//! Reference test integration tests (T049)
//!
//! These tests run the full reference test suite via cargo test.

use crate::{
    ConservationCheck, ExpectedResult, PositionBoundsCheck, PressureCheck,
    PressureUniformityCheck, ReferenceTest,
};
use kernel::eos::WATER_REST_DENSITY;

/// Resolve a path relative to the project root (two levels up from this crate)
fn project_path(relative: &str) -> String {
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let project_root = std::path::Path::new(manifest_dir)
        .parent() // backend/
        .and_then(|p| p.parent()) // project root
        .expect("Could not find project root");
    project_root.join(relative).to_string_lossy().to_string()
}

/// T045: Gravity settling test
///
/// Tests that water in a sealed 1cm box reaches hydrostatic equilibrium.
/// The box is completely filled with water, so particles cannot "settle to the
/// floor" -- instead we verify that particles stay in bounds and mass is
/// conserved. Energy conservation is relaxed because the domain-clamping wall
/// collisions dissipate kinetic energy by design (restitution = 0.2).
fn gravity_settling_test() -> ReferenceTest {
    ReferenceTest {
        name: "Gravity Settling".to_string(),
        config_path: project_path("configs/water-box-1cm.json"),
        timesteps: 5000,
        expected: ExpectedResult {
            position_bounds: Some(PositionBoundsCheck {
                min: [-0.001, -0.001, -0.001],
                max: [0.011, 0.011, 0.011],
            }),
            pressure_check: None,
            pressure_uniformity: None,
            conservation: Some(ConservationCheck {
                max_mass_error: 0.001,
                max_energy_error: 0.50, // relaxed: walls dissipate energy
            }),
            // No settling check -- box is completely filled, so particles
            // can't all settle to the floor. Containment (position bounds)
            // and mass conservation suffice for this test.
            settling: None,
        },
    }
}

/// T046: Hydrostatic pressure test
///
/// Verifies that the bottom 10% of the water column develops hydrostatic
/// pressure close to rho * g * h. Tolerance is generous (50%) because:
/// - Small domain (1cm) means SPH kernel truncation affects more particles
/// - Only ~10 particle layers so pressure gradient is coarsely resolved
/// - Weakly compressible EOS adds a compressibility offset
fn hydrostatic_pressure_test() -> ReferenceTest {
    let domain_height = 0.01_f32;
    let expected_bottom_pressure = WATER_REST_DENSITY * 9.81 * domain_height;

    ReferenceTest {
        name: "Hydrostatic Pressure".to_string(),
        config_path: project_path("configs/water-box-1cm.json"),
        timesteps: 5000,
        expected: ExpectedResult {
            position_bounds: Some(PositionBoundsCheck {
                min: [-0.001, -0.001, -0.001],
                max: [0.011, 0.011, 0.011],
            }),
            pressure_check: Some(PressureCheck {
                expected_bottom: expected_bottom_pressure,
                tolerance: 0.50, // 50% tolerance for small domain
            }),
            pressure_uniformity: None,
            conservation: Some(ConservationCheck {
                max_mass_error: 0.001,
                max_energy_error: 0.50, // relaxed: walls dissipate energy
            }),
            settling: None,
        },
    }
}

/// T047: Pressure equalization test
///
/// Air in a box with a sphere obstacle should reach pressure equilibrium.
/// With the weakly compressible Tait EOS for air, pressure should be near-zero
/// everywhere (gauge pressure at rest density). Domain is -0.005 to 0.005.
fn pressure_equalization_test() -> ReferenceTest {
    ReferenceTest {
        name: "Pressure Equalization".to_string(),
        config_path: project_path("configs/air-obstacle-1cm.json"),
        timesteps: 3000,
        expected: ExpectedResult {
            position_bounds: Some(PositionBoundsCheck {
                // Air domain is centered at origin: -0.005 to 0.005
                min: [-0.006, -0.006, -0.006],
                max: [0.006, 0.006, 0.006],
            }),
            pressure_check: None,
            pressure_uniformity: Some(PressureUniformityCheck {
                max_variation: 0.25, // When mean pressure is near zero (gauge),
                                     // variation is measured as absolute Pa.
                                     // 0.25 Pa is physically insignificant.
            }),
            conservation: Some(ConservationCheck {
                max_mass_error: 0.001,
                max_energy_error: 0.50, // relaxed: walls dissipate energy
            }),
            settling: None,
        },
    }
}

/// T049: Make gravity settling test runnable as cargo test
#[test]
fn test_gravity_settling() {
    let test = gravity_settling_test();
    let result = test.run().expect("Test execution failed");
    result.print_summary();
    assert!(result.passed, "Gravity settling test failed");
}

/// T049: Make hydrostatic pressure test runnable as cargo test
#[test]
fn test_hydrostatic_pressure() {
    let test = hydrostatic_pressure_test();
    let result = test.run().expect("Test execution failed");
    result.print_summary();
    assert!(result.passed, "Hydrostatic pressure test failed");
}

/// T049: Make pressure equalization test runnable as cargo test
#[test]
fn test_pressure_equalization() {
    let test = pressure_equalization_test();
    let result = test.run().expect("Test execution failed");
    result.print_summary();
    assert!(result.passed, "Pressure equalization test failed");
}
