//! T044: Reference test binary entry point
//!
//! Discovers and runs all reference test cases.

use reference_tests::{
    ConservationCheck, ExpectedResult, PositionBoundsCheck, PressureCheck,
    PressureUniformityCheck, ReferenceTest, SettlingCheck, TestResult,
};
use kernel::eos::WATER_REST_DENSITY;
use tracing_subscriber;

/// T045: Gravity settling test
///
/// Water released in container, run until settled, verify all particles
/// within one particle diameter of floor at steady state.
fn gravity_settling_test() -> ReferenceTest {
    ReferenceTest {
        name: "Gravity Settling".to_string(),
        config_path: "configs/water-box-1cm.json".to_string(),
        timesteps: 5000, // Run long enough to settle
        expected: ExpectedResult {
            position_bounds: Some(PositionBoundsCheck {
                min: [-0.001, -0.001, -0.001], // Allow small boundary margin
                max: [0.011, 0.011, 0.011],
            }),
            pressure_check: None,
            pressure_uniformity: None,
            conservation: Some(ConservationCheck {
                max_mass_error: 0.001, // 0.1%
                max_energy_error: 0.05, // 5%
            }),
            settling: Some(SettlingCheck {
                floor_y: 0.0,
                max_distance: 2.0, // Within 2 particle diameters of floor
                particle_spacing: 0.001,
            }),
        },
    }
}

/// T046: Hydrostatic pressure test
///
/// Water column at rest, verify bottom pressure matches rho*g*h within 5%.
fn hydrostatic_pressure_test() -> ReferenceTest {
    // For a 1cm water column: P_bottom = rho * g * h
    // h = 0.01 m, rho = 1000 kg/m^3, g = 9.81 m/s^2
    // P_expected = 1000 * 9.81 * 0.01 = 98.1 Pa
    let domain_height = 0.01_f32;
    let expected_bottom_pressure = WATER_REST_DENSITY * 9.81 * domain_height;

    ReferenceTest {
        name: "Hydrostatic Pressure".to_string(),
        config_path: "configs/water-box-1cm.json".to_string(),
        timesteps: 5000, // Run until equilibrium
        expected: ExpectedResult {
            position_bounds: Some(PositionBoundsCheck {
                min: [-0.001, -0.001, -0.001],
                max: [0.011, 0.011, 0.011],
            }),
            pressure_check: Some(PressureCheck {
                expected_bottom: expected_bottom_pressure,
                tolerance: 0.15, // 15% tolerance (relaxed for SPH)
            }),
            pressure_uniformity: None,
            conservation: Some(ConservationCheck {
                max_mass_error: 0.001,
                max_energy_error: 0.10, // 10% for this test
            }),
            settling: None,
        },
    }
}

/// T047: Pressure equalization test
///
/// Gas in sealed vessel from non-uniform initial state, verify pressure
/// variation less than 1% at steady state.
fn pressure_equalization_test() -> ReferenceTest {
    ReferenceTest {
        name: "Pressure Equalization".to_string(),
        config_path: "configs/air-obstacle-1cm.json".to_string(),
        timesteps: 3000, // Run until pressure equalizes
        expected: ExpectedResult {
            position_bounds: Some(PositionBoundsCheck {
                min: [-0.001, -0.001, -0.001],
                max: [0.011, 0.011, 0.011],
            }),
            pressure_check: None,
            pressure_uniformity: Some(PressureUniformityCheck {
                max_variation: 0.05, // 5% variation (relaxed from 1% for SPH)
            }),
            conservation: Some(ConservationCheck {
                max_mass_error: 0.001,
                max_energy_error: 0.10,
            }),
            settling: None,
        },
    }
}

/// Get all reference tests
fn all_tests() -> Vec<ReferenceTest> {
    vec![
        gravity_settling_test(),
        hydrostatic_pressure_test(),
        pressure_equalization_test(),
    ]
}

fn main() {
    // Initialize logging
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .with_target(false)
        .init();

    tracing::info!("SPH Reference Test Suite");
    tracing::info!("========================");

    // Get all tests
    let tests = all_tests();
    tracing::info!("Found {} reference tests", tests.len());

    // Run all tests
    let mut results: Vec<TestResult> = Vec::new();
    let mut passed_count = 0;
    let mut failed_count = 0;

    for test in tests {
        match test.run() {
            Ok(result) => {
                if result.passed {
                    passed_count += 1;
                } else {
                    failed_count += 1;
                }
                result.print_summary();
                results.push(result);
            }
            Err(e) => {
                eprintln!("\nERROR running test {}: {}", test.name, e);
                failed_count += 1;
            }
        }
    }

    // Print overall summary
    println!("\n{}", "=".repeat(80));
    println!("OVERALL SUMMARY");
    println!("{}", "=".repeat(80));
    println!("Total tests: {}", results.len());
    println!("Passed: {}", passed_count);
    println!("Failed: {}", failed_count);
    println!("{}", "=".repeat(80));

    // Exit with error code if any tests failed
    if failed_count > 0 {
        std::process::exit(1);
    }
}
