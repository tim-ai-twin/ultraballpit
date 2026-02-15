//! Equations of state for SPH fluid simulation.
//!
//! Provides pressure-density relations for water (Tait) and air (ideal gas).
//! All units are SI: meters, kg, seconds, Kelvin, Pascals.

/// Rest density for liquid water (kg/m^3).
pub const WATER_REST_DENSITY: f32 = 1000.0;

/// Rest density for dry air at 20 C, 1 atm (kg/m^3).
pub const AIR_REST_DENSITY: f32 = 1.204;

/// Tait equation exponent (gamma) for water -- standard WCSPH value.
pub const WATER_GAMMA: f32 = 7.0;

/// Specific gas constant for dry air, R_specific (J/(kg K)).
pub const AIR_R_SPECIFIC: f32 = 287.058;

/// Tait equation of state for weakly-compressible liquid (WCSPH).
///
/// ```text
/// P = B * ((rho / rho0)^gamma - 1)
/// ```
/// where `B = rho0 * c_s^2 / gamma`.
///
/// # Arguments
/// * `density` - Current density rho (kg/m^3).
/// * `rest_density` - Reference rest density rho0 (kg/m^3).
/// * `speed_of_sound` - Numerical speed of sound c_s (m/s).
/// * `gamma` - Tait exponent (7 for water).
///
/// # Returns
/// Pressure in Pascals.  Can be negative (tension) if `density < rest_density`.
pub fn tait_eos(density: f32, rest_density: f32, speed_of_sound: f32, gamma: f32) -> f32 {
    let b = rest_density * speed_of_sound * speed_of_sound / gamma;
    let ratio = density / rest_density;
    b * (ratio.powf(gamma) - 1.0)
}

/// Ideal gas equation of state for air.
///
/// ```text
/// P = rho * R_specific * T
/// ```
/// where `R_specific = 287.058 J/(kg K)` for dry air.
///
/// # Arguments
/// * `density` - Current density (kg/m^3).
/// * `temperature` - Temperature (K).
///
/// # Returns
/// Pressure in Pascals.
pub fn ideal_gas_eos(density: f32, temperature: f32) -> f32 {
    density * AIR_R_SPECIFIC * temperature
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tait_at_rest_density_is_zero() {
        let p = tait_eos(WATER_REST_DENSITY, WATER_REST_DENSITY, 20.0, WATER_GAMMA);
        assert!(
            p.abs() < 1.0e-3,
            "pressure at rest density should be ~0, got {p}"
        );
    }

    #[test]
    fn tait_positive_when_compressed() {
        let rho = 1010.0; // slightly compressed
        let p = tait_eos(rho, WATER_REST_DENSITY, 20.0, WATER_GAMMA);
        assert!(p > 0.0, "compressed water should have positive pressure, got {p}");
    }

    #[test]
    fn tait_negative_when_expanded() {
        let rho = 990.0; // slightly expanded
        let p = tait_eos(rho, WATER_REST_DENSITY, 20.0, WATER_GAMMA);
        assert!(p < 0.0, "expanded water should have negative pressure, got {p}");
    }

    #[test]
    fn ideal_gas_standard_conditions() {
        // At 20 C = 293.15 K, rho = 1.204 kg/m^3
        // P should be close to 1 atm ~ 101325 Pa
        let p = ideal_gas_eos(AIR_REST_DENSITY, 293.15);
        let atm = 101325.0_f32;
        let error = (p - atm).abs() / atm;
        assert!(
            error < 0.01,
            "ideal gas at standard conditions should be ~101325 Pa, got {p} (error {error:.4})"
        );
    }

    #[test]
    fn ideal_gas_proportional_to_density() {
        let t = 300.0;
        let p1 = ideal_gas_eos(1.0, t);
        let p2 = ideal_gas_eos(2.0, t);
        assert!((p2 - 2.0 * p1).abs() < 1.0e-3);
    }

    #[test]
    fn ideal_gas_proportional_to_temperature() {
        let rho = 1.2;
        let p1 = ideal_gas_eos(rho, 300.0);
        let p2 = ideal_gas_eos(rho, 600.0);
        assert!((p2 - 2.0 * p1).abs() < 1.0e-3);
    }
}
