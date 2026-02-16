//! Analytical reference solutions for SPH benchmark validation (T085).
//!
//! Provides closed-form solutions against which SPH simulation results can be
//! compared for quantitative accuracy assessment.

/// Analytical solution for Poiseuille (channel) flow between parallel plates.
///
/// In SPH, the driving "pressure gradient" is typically specified as a body-force
/// acceleration G (m/s^2) in the gravity vector. The Navier-Stokes equation for
/// the steady-state velocity profile simplifies to:
///
/// ```text
/// u(y) = G / (2 * nu) * y * (H - y)
/// ```
///
/// where:
/// - `y` is the distance from the bottom wall (0 <= y <= H)
/// - `G` is the driving body-force acceleration (m/s^2)
/// - `nu` is the kinematic viscosity (m^2/s)
/// - `H` is the channel height (m)
///
/// The maximum velocity occurs at the centerline (y = H/2):
/// ```text
/// u_max = G * H^2 / (8 * nu)
/// ```
pub struct PoiseuilleFlow {
    /// Channel height H (m)
    pub channel_height: f64,
    /// Driving body-force acceleration G (m/s^2)
    pub body_force_accel: f64,
    /// Kinematic viscosity nu (m^2/s)
    pub kinematic_viscosity: f64,
}

impl PoiseuilleFlow {
    /// Create a new Poiseuille flow analytical solution.
    ///
    /// # Arguments
    /// * `channel_height` - Distance between the two parallel plates (m)
    /// * `body_force_accel` - Driving body-force acceleration G (m/s^2)
    /// * `kinematic_viscosity` - Kinematic viscosity nu (m^2/s)
    pub fn new(
        channel_height: f64,
        body_force_accel: f64,
        kinematic_viscosity: f64,
    ) -> Self {
        Self {
            channel_height,
            body_force_accel,
            kinematic_viscosity,
        }
    }

    /// Compute the analytical velocity at height y from the bottom wall.
    ///
    /// Returns u(y) = G / (2 * nu) * y * (H - y)
    pub fn velocity_at(&self, y: f64) -> f64 {
        poiseuille_velocity(y, self.channel_height, self.body_force_accel, self.kinematic_viscosity)
    }

    /// Compute the maximum (centerline) velocity.
    pub fn max_velocity(&self) -> f64 {
        poiseuille_max_velocity(self.channel_height, self.body_force_accel, self.kinematic_viscosity)
    }

    /// Compute the analytical velocity profile at N evenly spaced points.
    ///
    /// Returns a vector of (y, u) pairs from y=0 to y=H.
    pub fn velocity_profile(&self, n_points: usize) -> Vec<(f64, f64)> {
        let mut profile = Vec::with_capacity(n_points);
        for i in 0..n_points {
            let y = self.channel_height * (i as f64) / ((n_points - 1) as f64);
            profile.push((y, self.velocity_at(y)));
        }
        profile
    }

    /// Compare a simulated velocity profile against the analytical solution.
    ///
    /// # Arguments
    /// * `y_positions` - Y coordinates of sampled particles
    /// * `velocities` - Corresponding x-velocities
    ///
    /// # Returns
    /// RMS relative error (0.0 = perfect match, 1.0 = 100% error)
    pub fn rms_error(&self, y_positions: &[f64], velocities: &[f64]) -> f64 {
        poiseuille_rms_error(
            y_positions,
            velocities,
            self.channel_height,
            self.body_force_accel,
            self.kinematic_viscosity,
        )
    }
}

/// Analytical Poiseuille flow with proper unit handling for SPH simulations.
///
/// In SPH, the body force G is specified as an acceleration (m/s^2) in the
/// gravity vector. The Poiseuille solution for velocity driven by acceleration
/// G_x through a fluid of kinematic viscosity nu in a channel of height H is:
///
/// ```text
/// u(y) = G_x / (2 * nu) * y * (H - y)
/// ```
pub fn poiseuille_velocity(y: f64, channel_height: f64, body_force_accel: f64, kinematic_viscosity: f64) -> f64 {
    if y < 0.0 || y > channel_height {
        return 0.0;
    }
    body_force_accel / (2.0 * kinematic_viscosity) * y * (channel_height - y)
}

/// Maximum velocity for Poiseuille flow (at centerline y = H/2).
pub fn poiseuille_max_velocity(channel_height: f64, body_force_accel: f64, kinematic_viscosity: f64) -> f64 {
    body_force_accel * channel_height * channel_height / (8.0 * kinematic_viscosity)
}

/// Compute the RMS error between simulated and analytical Poiseuille profiles.
///
/// # Arguments
/// * `y_positions` - Y coordinates of sampled particles (m)
/// * `velocities` - Corresponding x-velocities (m/s)
/// * `channel_height` - Channel height H (m)
/// * `body_force_accel` - Driving acceleration G_x (m/s^2)
/// * `kinematic_viscosity` - Kinematic viscosity nu (m^2/s)
///
/// # Returns
/// RMS error normalized by the maximum analytical velocity.
pub fn poiseuille_rms_error(
    y_positions: &[f64],
    velocities: &[f64],
    channel_height: f64,
    body_force_accel: f64,
    kinematic_viscosity: f64,
) -> f64 {
    assert_eq!(y_positions.len(), velocities.len());
    if y_positions.is_empty() {
        return f64::MAX;
    }

    let u_max = poiseuille_max_velocity(channel_height, body_force_accel, kinematic_viscosity);
    if u_max.abs() < 1e-15 {
        return f64::MAX;
    }

    let mut sum_sq_error = 0.0;
    let mut count = 0;
    for (&y, &u_sim) in y_positions.iter().zip(velocities.iter()) {
        let u_analytical = poiseuille_velocity(y, channel_height, body_force_accel, kinematic_viscosity);
        let error = (u_sim - u_analytical) / u_max;
        sum_sq_error += error * error;
        count += 1;
    }

    (sum_sq_error / count as f64).sqrt()
}

/// Linear wave theory: angular frequency for a surface gravity wave.
///
/// For a standing wave in a tank of depth h with wavelength lambda:
/// ```text
/// omega = sqrt(g * k * tanh(k * h))
/// ```
/// where k = 2*pi/lambda is the wavenumber.
///
/// # Arguments
/// * `wavelength` - Wavelength lambda (m)
/// * `depth` - Water depth h (m)
/// * `gravity` - Gravitational acceleration g (m/s^2)
///
/// # Returns
/// Angular frequency omega (rad/s)
pub fn standing_wave_omega(wavelength: f64, depth: f64, gravity: f64) -> f64 {
    let k = 2.0 * std::f64::consts::PI / wavelength;
    (gravity * k * (k * depth).tanh()).sqrt()
}

/// Linear wave theory: period for a surface gravity wave.
///
/// T = 2 * pi / omega
pub fn standing_wave_period(wavelength: f64, depth: f64, gravity: f64) -> f64 {
    2.0 * std::f64::consts::PI / standing_wave_omega(wavelength, depth, gravity)
}

/// Martin & Moyce (1952) dam break reference data.
///
/// Provides dimensionless experimental data for the water front position
/// during a dam break collapse.
#[derive(Debug, Clone)]
pub struct MartinMoyceData {
    /// Dimensionless time T* = t * sqrt(2*g/a)
    pub dimensionless_time: Vec<f64>,
    /// Dimensionless front position Z* = z/a
    pub dimensionless_front_position: Vec<f64>,
}

impl MartinMoyceData {
    /// Load the Martin & Moyce 1952 reference data from the embedded JSON.
    pub fn load() -> Self {
        // Hardcoded from published graph (Martin & Moyce 1952, Figure 4)
        Self {
            dimensionless_time: vec![0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5],
            dimensionless_front_position: vec![1.0, 1.0, 1.15, 1.5, 2.0, 2.5, 3.1, 3.7],
        }
    }

    /// Interpolate the reference front position at a given dimensionless time.
    ///
    /// Uses linear interpolation between data points. Extrapolates linearly
    /// beyond the data range.
    pub fn interpolate_front_position(&self, t_star: f64) -> f64 {
        let times = &self.dimensionless_time;
        let positions = &self.dimensionless_front_position;

        if t_star <= times[0] {
            return positions[0];
        }
        if t_star >= *times.last().unwrap() {
            // Linear extrapolation from last two points
            let n = times.len();
            let slope = (positions[n - 1] - positions[n - 2]) / (times[n - 1] - times[n - 2]);
            return positions[n - 1] + slope * (t_star - times[n - 1]);
        }

        // Find bracketing interval
        for i in 0..times.len() - 1 {
            if t_star >= times[i] && t_star <= times[i + 1] {
                let frac = (t_star - times[i]) / (times[i + 1] - times[i]);
                return positions[i] + frac * (positions[i + 1] - positions[i]);
            }
        }

        positions[0]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_poiseuille_velocity_at_walls() {
        let h = 0.01; // 1cm channel
        let g = 0.001; // m/s^2 body force
        let nu = 0.001; // m^2/s kinematic viscosity
        assert!((poiseuille_velocity(0.0, h, g, nu)).abs() < 1e-15);
        assert!((poiseuille_velocity(h, h, g, nu)).abs() < 1e-15);
    }

    #[test]
    fn test_poiseuille_velocity_parabolic() {
        let h = 0.01;
        let g = 0.001;
        let nu = 0.001;
        let u_max = poiseuille_max_velocity(h, g, nu);
        let u_center = poiseuille_velocity(h / 2.0, h, g, nu);
        assert!(
            (u_center - u_max).abs() < 1e-12,
            "Center velocity {u_center} should equal max {u_max}"
        );
        // Check symmetry
        let u_quarter = poiseuille_velocity(h / 4.0, h, g, nu);
        let u_three_quarter = poiseuille_velocity(3.0 * h / 4.0, h, g, nu);
        assert!(
            (u_quarter - u_three_quarter).abs() < 1e-12,
            "Profile should be symmetric"
        );
    }

    #[test]
    fn test_poiseuille_rms_perfect_match() {
        let h = 0.01;
        let g = 0.001;
        let nu = 0.001;
        let n = 20;
        let y_pos: Vec<f64> = (0..n).map(|i| h * (i as f64) / ((n - 1) as f64)).collect();
        let vels: Vec<f64> = y_pos.iter().map(|&y| poiseuille_velocity(y, h, g, nu)).collect();
        let rms = poiseuille_rms_error(&y_pos, &vels, h, g, nu);
        assert!(rms < 1e-10, "Perfect match should give zero RMS error, got {rms}");
    }

    #[test]
    fn test_standing_wave_shallow_water() {
        // In shallow water (kh << 1), omega ~ sqrt(g*h) * k
        let depth = 0.001; // very shallow
        let wavelength = 1.0; // long wave
        let g = 9.81;
        let omega = standing_wave_omega(wavelength, depth, g);
        let k = 2.0 * std::f64::consts::PI / wavelength;
        let omega_shallow = (g * depth).sqrt() * k;
        assert!(
            (omega - omega_shallow).abs() / omega_shallow < 0.01,
            "Shallow water limit: omega={omega}, expected~{omega_shallow}"
        );
    }

    #[test]
    fn test_standing_wave_period() {
        let wavelength = 0.10;
        let depth = 0.025;
        let g = 9.81;
        let t = standing_wave_period(wavelength, depth, g);
        // Should be a reasonable period (order of magnitude ~0.1s for 10cm wave)
        assert!(t > 0.01 && t < 1.0, "Period {t}s seems unreasonable");
    }

    #[test]
    fn test_martin_moyce_interpolation() {
        let data = MartinMoyceData::load();
        // At t*=0, front should be at z*=1.0
        assert!((data.interpolate_front_position(0.0) - 1.0).abs() < 1e-10);
        // At t*=2.0, front should be at z*=2.0
        assert!((data.interpolate_front_position(2.0) - 2.0).abs() < 1e-10);
        // Interpolated value at t*=0.75 should be between 1.0 and 1.15
        let z_075 = data.interpolate_front_position(0.75);
        assert!(z_075 >= 1.0 && z_075 <= 1.15, "z*(0.75) = {z_075}");
    }
}
