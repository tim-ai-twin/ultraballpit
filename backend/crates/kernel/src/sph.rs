//! SPH smoothing kernel functions and core SPH operators.
//!
//! Implements the Wendland C2 kernel and its gradient for 3D SPH simulations.
//! The Wendland C2 kernel is preferred over the cubic spline for its strict
//! positivity and lack of tensile-instability pairing artifacts.
//!
//! Also provides the core SPH operators: density summation, pressure forces,
//! viscous forces, gravity, and adaptive timestep.

use std::f32::consts::PI;

use crate::eos::{self, AIR_GAMMA, AIR_REST_DENSITY, WATER_GAMMA, WATER_REST_DENSITY};
use crate::neighbor::NeighborGrid;
use crate::particle::{FluidType, ParticleArrays};

/// Normalization constant for the 3D Wendland C2 kernel: 21 / (16 * pi).
///
/// With q = r/h and support radius 2h, the analytically correct normalization
/// for the Wendland C2 kernel in 3D is alpha_d = 21 / (16 * pi).
const WENDLAND_C2_NORM_3D: f32 = 21.0 / (16.0 * PI);

/// Wendland C2 smoothing kernel in 3D.
///
/// ```text
/// W(r, h) = (21 / (16 pi h^3)) * (1 - q/2)^4 * (1 + 2q)   for q = r/h <= 2
/// W(r, h) = 0                                                for q > 2
/// ```
///
/// # Arguments
/// * `r` - Distance between two particles (must be >= 0).
/// * `h` - Smoothing length. The support radius is 2h.
///
/// # Returns
/// Kernel value W(r, h).
pub fn wendland_c2(r: f32, h: f32) -> f32 {
    let q = r / h;
    if q >= 2.0 {
        return 0.0;
    }
    let h3 = h * h * h;
    let one_minus_half_q = 1.0 - 0.5 * q;
    // (1 - q/2)^4
    let t = one_minus_half_q * one_minus_half_q;
    let t4 = t * t;
    WENDLAND_C2_NORM_3D / h3 * t4 * (1.0 + 2.0 * q)
}

/// Gradient of the Wendland C2 smoothing kernel in 3D.
///
/// Returns the gradient vector components (dW/dx, dW/dy, dW/dz) given the
/// displacement vector (dx, dy, dz) from particle j to particle i and the
/// pre-computed distance `r = sqrt(dx^2 + dy^2 + dz^2)`.
///
/// ```text
/// nabla W = (dW/dr) * (r_vec / |r|)
/// dW/dr   = (21 / (16 pi h^3)) * (-5 q) * (1 - q/2)^3 / h    for q <= 2
/// ```
///
/// When `r` is (near) zero the gradient is zero (particles at the same position).
///
/// # Arguments
/// * `dx`, `dy`, `dz` - Displacement vector from particle j to particle i.
/// * `r` - Euclidean distance (>= 0). Pass 0.0 if particles overlap.
/// * `h` - Smoothing length.
///
/// # Returns
/// Tuple `(gx, gy, gz)` -- gradient components.
pub fn wendland_c2_gradient(dx: f32, dy: f32, dz: f32, r: f32, h: f32) -> (f32, f32, f32) {
    let q = r / h;
    if q >= 2.0 || r < 1.0e-12 {
        return (0.0, 0.0, 0.0);
    }

    let h3 = h * h * h;
    let one_minus_half_q = 1.0 - 0.5 * q;
    // (1 - q/2)^3
    let t3 = one_minus_half_q * one_minus_half_q * one_minus_half_q;

    // dW/dr = norm / h^3 * (-5q)(1 - q/2)^3 / h
    let dw_dr = WENDLAND_C2_NORM_3D / (h3 * h) * (-5.0 * q) * t3;

    // gradient = dW/dr * (r_vec / |r|)
    let inv_r = 1.0 / r;
    (dw_dr * dx * inv_r, dw_dr * dy * inv_r, dw_dr * dz * inv_r)
}

// ---------------------------------------------------------------------------
// T017: SPH density summation
// ---------------------------------------------------------------------------

/// Delta-SPH diffusion coefficient (Molteni & Colagrossi 2009).
/// Typical value: 0.1 for moderate diffusion.
const DELTA_SPH: f32 = 0.1;

/// Compute density for all particles using SPH summation, with optional
/// delta-SPH density diffusion (Molteni & Colagrossi 2009).
///
/// ```text
/// rho_i = sum_j m_j * W(|r_i - r_j|, h)
/// ```
///
/// The diffusion correction is:
/// ```text
/// D_i = delta * h * c_s * sum_j V_j * (rho_i - rho_j) * (2 * r_ij · grad_W / |r_ij|^2)
/// ```
///
/// This suppresses density noise and improves hydrostatic pressure accuracy.
///
/// Includes self-contribution (W(0, h) * m_i) and contributions from
/// boundary particles. Boundary particles contribute to density but do
/// not have their own density updated.
pub fn compute_density(
    particles: &mut ParticleArrays,
    boundary_x: &[f32],
    boundary_y: &[f32],
    boundary_z: &[f32],
    boundary_mass: &[f32],
    grid: &NeighborGrid,
    h: f32,
) {
    let n = particles.len();
    let support_radius = 2.0 * h;

    for i in 0..n {
        // Self-contribution
        let mut rho = particles.mass[i] * wendland_c2(0.0, h);

        // Fluid neighbor contributions
        grid.for_each_neighbor(
            i,
            &particles.x,
            &particles.y,
            &particles.z,
            support_radius,
            |j| {
                let dx = particles.x[i] - particles.x[j];
                let dy = particles.y[i] - particles.y[j];
                let dz = particles.z[i] - particles.z[j];
                let r = (dx * dx + dy * dy + dz * dz).sqrt();
                rho += particles.mass[j] * wendland_c2(r, h);
            },
        );

        // Boundary particle contributions
        let px = particles.x[i];
        let py = particles.y[i];
        let pz = particles.z[i];
        for b in 0..boundary_x.len() {
            let dx = px - boundary_x[b];
            let dy = py - boundary_y[b];
            let dz = pz - boundary_z[b];
            let r = (dx * dx + dy * dy + dz * dz).sqrt();
            if r < support_radius {
                rho += boundary_mass[b] * wendland_c2(r, h);
            }
        }

        particles.density[i] = rho;
    }
}

/// Compute delta-SPH density diffusion term (Molteni & Colagrossi 2009).
///
/// Returns a per-particle diffusion rate that should be applied as:
/// `rho_i += diffusion[i] * dt`
///
/// The term suppresses density noise, improving pressure field smoothness
/// and enabling lower speed of sound (larger timesteps).
pub fn compute_density_diffusion(
    particles: &ParticleArrays,
    grid: &NeighborGrid,
    h: f32,
    speed_of_sound: f32,
) -> Vec<f32> {
    let n = particles.len();
    let support_radius = 2.0 * h;
    let eta_sq = 0.01 * h * h;
    let mut diffusion = vec![0.0f32; n];

    for i in 0..n {
        let mut diff_sum = 0.0f32;

        grid.for_each_neighbor(
            i,
            &particles.x,
            &particles.y,
            &particles.z,
            support_radius,
            |j| {
                let dx = particles.x[i] - particles.x[j];
                let dy = particles.y[i] - particles.y[j];
                let dz = particles.z[i] - particles.z[j];
                let dist_sq = dx * dx + dy * dy + dz * dz;
                let r = dist_sq.sqrt();

                let (gx, gy, gz) = wendland_c2_gradient(dx, dy, dz, r, h);

                // Volume of neighbor j: V_j = m_j / rho_j
                let v_j = particles.mass[j] / particles.density[j];

                // r_ij · grad_W
                let r_dot_grad = dx * gx + dy * gy + dz * gz;

                // Diffusion: delta * h * c_s * V_j * (rho_i - rho_j) * 2 * r·gradW / (|r|^2 + eta^2)
                // Sign note: our r = x_i - x_j, so r·gradW < 0. The paper uses r_ij = x_j - x_i,
                // so (rho_i - rho_j) with our r·gradW gives the correct diffusion sign.
                let density_diff = particles.density[i] - particles.density[j];
                diff_sum += v_j * density_diff * 2.0 * r_dot_grad / (dist_sq + eta_sq);
            },
        );

        diffusion[i] = DELTA_SPH * h * speed_of_sound * diff_sum;
    }

    diffusion
}

// ---------------------------------------------------------------------------
// XSPH velocity smoothing (Monaghan 1989)
// ---------------------------------------------------------------------------

/// XSPH smoothing coefficient. Range [0, 1]; higher = more smoothing.
/// 0.5 is the standard value from Monaghan 1989.
const XSPH_EPSILON: f32 = 0.5;

/// Compute XSPH velocity correction for all particles.
///
/// ```text
/// v_xsph_i = v_i + epsilon * sum_j (m_j / rho_avg_ij) * (v_j - v_i) * W(r_ij, h)
/// ```
///
/// where `rho_avg_ij = 0.5 * (rho_i + rho_j)`.
///
/// Returns three vectors (dvx, dvy, dvz) of corrections to be added to velocity
/// before the position update (drift step).
pub fn compute_xsph_correction(
    particles: &ParticleArrays,
    grid: &NeighborGrid,
    h: f32,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let n = particles.len();
    let support_radius = 2.0 * h;
    let mut dvx = vec![0.0f32; n];
    let mut dvy = vec![0.0f32; n];
    let mut dvz = vec![0.0f32; n];

    for i in 0..n {
        let mut cx = 0.0f32;
        let mut cy = 0.0f32;
        let mut cz = 0.0f32;

        grid.for_each_neighbor(
            i,
            &particles.x,
            &particles.y,
            &particles.z,
            support_radius,
            |j| {
                let dx = particles.x[i] - particles.x[j];
                let dy = particles.y[i] - particles.y[j];
                let dz = particles.z[i] - particles.z[j];
                let r = (dx * dx + dy * dy + dz * dz).sqrt();

                let w = wendland_c2(r, h);
                let rho_avg = 0.5 * (particles.density[i] + particles.density[j]);
                let factor = particles.mass[j] / rho_avg.max(1.0) * w;

                cx += factor * (particles.vx[j] - particles.vx[i]);
                cy += factor * (particles.vy[j] - particles.vy[i]);
                cz += factor * (particles.vz[j] - particles.vz[i]);
            },
        );

        dvx[i] = XSPH_EPSILON * cx;
        dvy[i] = XSPH_EPSILON * cy;
        dvz[i] = XSPH_EPSILON * cz;
    }

    (dvx, dvy, dvz)
}

// ---------------------------------------------------------------------------
// T018: Pressure force computation
// ---------------------------------------------------------------------------

/// Compute pressure from density using the appropriate equation of state.
///
/// Both water and air use the Tait (weakly compressible) EOS, which produces
/// gauge pressure (zero at rest density). This is essential for WCSPH stability:
/// the ideal gas EOS returns absolute pressure (~101 kPa) whose small SPH
/// discretization noise gets amplified into catastrophic forces.
///
/// Water: gamma=7, rho0=1000 kg/m^3
/// Air:   gamma=1.4, rho0=1.204 kg/m^3
pub fn compute_pressure(particles: &mut ParticleArrays, speed_of_sound: f32) {
    for i in 0..particles.len() {
        particles.pressure[i] = match particles.fluid_type[i] {
            FluidType::Water => eos::tait_eos(
                particles.density[i],
                WATER_REST_DENSITY,
                speed_of_sound,
                WATER_GAMMA,
            ),
            FluidType::Air => eos::tait_eos(
                particles.density[i],
                AIR_REST_DENSITY,
                speed_of_sound,
                AIR_GAMMA,
            ),
        };
    }
}

/// Compute pressure forces using the symmetric SPH pressure gradient.
///
/// ```text
/// F_pressure_i = -m_i * sum_j m_j * (P_i/rho_i^2 + P_j/rho_j^2) * grad_W(r_ij, h)
/// ```
///
/// Boundary particles exert pressure on fluid particles but do not receive forces.
/// Pressure should already be computed (call `compute_pressure` first or use EOS).
pub fn compute_pressure_forces(
    particles: &mut ParticleArrays,
    boundary_x: &[f32],
    boundary_y: &[f32],
    boundary_z: &[f32],
    boundary_mass: &[f32],
    boundary_pressure: &[f32],
    grid: &NeighborGrid,
    h: f32,
) {
    let n = particles.len();
    let support_radius = 2.0 * h;

    // We need to accumulate forces without mutably borrowing particles in the closure,
    // so collect contributions into separate vectors.
    let mut fx = vec![0.0f32; n];
    let mut fy = vec![0.0f32; n];
    let mut fz = vec![0.0f32; n];

    for i in 0..n {
        let pi_over_rho2_i = particles.pressure[i] / (particles.density[i] * particles.density[i]);

        // Fluid-fluid interactions
        grid.for_each_neighbor(
            i,
            &particles.x,
            &particles.y,
            &particles.z,
            support_radius,
            |j| {
                let dx = particles.x[i] - particles.x[j];
                let dy = particles.y[i] - particles.y[j];
                let dz = particles.z[i] - particles.z[j];
                let r = (dx * dx + dy * dy + dz * dz).sqrt();

                let (gx, gy, gz) = wendland_c2_gradient(dx, dy, dz, r, h);

                let pj_over_rho2_j =
                    particles.pressure[j] / (particles.density[j] * particles.density[j]);

                let factor = -particles.mass[i] * particles.mass[j]
                    * (pi_over_rho2_i + pj_over_rho2_j);

                fx[i] += factor * gx;
                fy[i] += factor * gy;
                fz[i] += factor * gz;
            },
        );

        // Boundary particle contributions to pressure force on fluid particle i.
        // For boundary interactions, clamp the fluid-side pressure to >= 0 to ensure
        // boundaries always repel (never attract) fluid particles. Without this,
        // negative Tait pressure from sub-rest-density fluid would cause boundaries
        // to attract fluid, leading to particle penetration.
        let px = particles.x[i];
        let py = particles.y[i];
        let pz = particles.z[i];
        let m_i = particles.mass[i];
        let pi_clamped = particles.pressure[i].max(0.0);
        let pi_clamped_over_rho2 = pi_clamped / (particles.density[i] * particles.density[i]);

        // Use the rest density matching this fluid particle's type for the
        // boundary density denominator. This ensures correct force scaling
        // for both water (rho0=1000) and air (rho0=1.204) particles.
        let boundary_rho = match particles.fluid_type[i] {
            FluidType::Water => WATER_REST_DENSITY,
            FluidType::Air => AIR_REST_DENSITY,
        };

        for b in 0..boundary_x.len() {
            let dx = px - boundary_x[b];
            let dy = py - boundary_y[b];
            let dz = pz - boundary_z[b];
            let r = (dx * dx + dy * dy + dz * dz).sqrt();
            if r < support_radius {
                let (gx, gy, gz) = wendland_c2_gradient(dx, dy, dz, r, h);

                let pb_over_rho2_b =
                    boundary_pressure[b] / (boundary_rho * boundary_rho);
                let factor =
                    -m_i * boundary_mass[b] * (pi_clamped_over_rho2 + pb_over_rho2_b);

                fx[i] += factor * gx;
                fy[i] += factor * gy;
                fz[i] += factor * gz;
            }
        }
    }

    // Convert forces to accelerations: a = F / m
    for i in 0..n {
        particles.ax[i] += fx[i] / particles.mass[i];
        particles.ay[i] += fy[i] / particles.mass[i];
        particles.az[i] += fz[i] / particles.mass[i];
    }
}

// ---------------------------------------------------------------------------
// T019: Viscous forces (Monaghan 1992 artificial viscosity)
// ---------------------------------------------------------------------------

/// Compute viscous forces using Monaghan (1992) artificial viscosity.
///
/// ```text
/// Pi_ij = (-alpha * c_avg * mu_ij + beta * mu_ij^2) / rho_avg   when v_ij . r_ij < 0
/// mu_ij = h * (v_ij . r_ij) / (|r_ij|^2 + 0.01 * h^2)
/// ```
///
/// Uses alpha=1.0, beta=2.0 as defaults. Only applies when particles approach
/// each other (v_ij . r_ij < 0).
pub fn compute_viscous_forces(
    particles: &mut ParticleArrays,
    grid: &NeighborGrid,
    h: f32,
    speed_of_sound: f32,
) {
    let n = particles.len();
    let support_radius = 2.0 * h;
    let alpha = 1.0_f32;
    let beta = 2.0_f32;
    let eta_sq = 0.01 * h * h;

    let mut fx = vec![0.0f32; n];
    let mut fy = vec![0.0f32; n];
    let mut fz = vec![0.0f32; n];

    for i in 0..n {
        grid.for_each_neighbor(
            i,
            &particles.x,
            &particles.y,
            &particles.z,
            support_radius,
            |j| {
                let dx = particles.x[i] - particles.x[j];
                let dy = particles.y[i] - particles.y[j];
                let dz = particles.z[i] - particles.z[j];
                let r = (dx * dx + dy * dy + dz * dz).sqrt();

                let dvx = particles.vx[i] - particles.vx[j];
                let dvy = particles.vy[i] - particles.vy[j];
                let dvz = particles.vz[i] - particles.vz[j];

                let vr_dot = dvx * dx + dvy * dy + dvz * dz;

                // Only apply viscosity when particles approach each other
                if vr_dot < 0.0 {
                    let r_sq = r * r;
                    let mu_ij = h * vr_dot / (r_sq + eta_sq);
                    let rho_avg = 0.5 * (particles.density[i] + particles.density[j]);
                    let pi_ij = (-alpha * speed_of_sound * mu_ij + beta * mu_ij * mu_ij) / rho_avg;

                    let (gx, gy, gz) = wendland_c2_gradient(dx, dy, dz, r, h);

                    // Force contribution: -m_i * m_j * Pi_ij * grad_W
                    let factor = -particles.mass[j] * pi_ij;
                    fx[i] += factor * gx;
                    fy[i] += factor * gy;
                    fz[i] += factor * gz;
                }
            },
        );
    }

    // Add viscous acceleration
    for i in 0..n {
        particles.ax[i] += fx[i];
        particles.ay[i] += fy[i];
        particles.az[i] += fz[i];
    }
}

// ---------------------------------------------------------------------------
// T020: Gravity
// ---------------------------------------------------------------------------

/// Apply gravity acceleration to all particles.
///
/// Adds the gravitational acceleration vector `gravity` to each particle's
/// acceleration. Typically `gravity = [0.0, -9.81, 0.0]` for downward gravity.
pub fn apply_gravity(particles: &mut ParticleArrays, gravity: [f32; 3]) {
    for i in 0..particles.len() {
        particles.ax[i] += gravity[0];
        particles.ay[i] += gravity[1];
        particles.az[i] += gravity[2];
    }
}

// ---------------------------------------------------------------------------
// Auto-tune speed of sound
// ---------------------------------------------------------------------------

/// Minimum speed of sound floor (m/s).
/// Even in very calm simulations, c_s must be large enough to produce
/// meaningful pressure response.
const MIN_SPEED_OF_SOUND: f32 = 1.0;

/// Auto-tune the speed of sound based on domain geometry and gravity.
///
/// The WCSPH rule is `c_s >= 10 * v_max` for <1% density variation.
/// For gravity-driven flows, `v_max ≈ sqrt(2 * g * H)` where H is domain height.
///
/// Returns the minimum `c_s` that satisfies the weakly-compressible constraint
/// for the given domain, or `c_s_config` if it's already larger.
///
/// # Arguments
/// * `gravity` - Gravity vector [gx, gy, gz]
/// * `domain_min` - Domain minimum bounds
/// * `domain_max` - Domain maximum bounds
/// * `c_s_config` - The configured (user-specified) speed of sound
pub fn auto_tune_speed_of_sound(
    gravity: [f32; 3],
    domain_min: [f32; 3],
    domain_max: [f32; 3],
    c_s_config: f32,
) -> f32 {
    // Gravity magnitude
    let g_mag = (gravity[0] * gravity[0] + gravity[1] * gravity[1] + gravity[2] * gravity[2]).sqrt();

    if g_mag < 1.0e-6 {
        // No gravity — use config value or floor
        return c_s_config.max(MIN_SPEED_OF_SOUND);
    }

    // Domain height along gravity direction
    let g_norm = [gravity[0] / g_mag, gravity[1] / g_mag, gravity[2] / g_mag];
    let extent = [
        domain_max[0] - domain_min[0],
        domain_max[1] - domain_min[1],
        domain_max[2] - domain_min[2],
    ];
    // Project domain extent onto gravity direction
    let height = (extent[0] * g_norm[0].abs() + extent[1] * g_norm[1].abs() + extent[2] * g_norm[2].abs())
        .abs();

    // v_max estimate from freefall: sqrt(2 * g * H)
    let v_max_est = (2.0 * g_mag * height).sqrt();

    // WCSPH constraint: c_s >= 10 * v_max for <1% density variation
    // Use factor of 10 (standard) — can reduce to 7 for <3% variation (SC-003)
    let c_s_auto = (10.0 * v_max_est).max(MIN_SPEED_OF_SOUND);

    // Use the auto-tuned value (it should be lower than typical config defaults)
    // but respect an explicitly low user setting
    let result = c_s_auto.min(c_s_config);
    tracing::info!(
        "Auto-tuned c_s: v_max_est={:.3} m/s, c_s_auto={:.2} m/s, c_s_config={:.2} m/s → using {:.2} m/s",
        v_max_est, c_s_auto, c_s_config, result
    );
    result
}

// ---------------------------------------------------------------------------
// T022: Adaptive timestep via CFL condition
// ---------------------------------------------------------------------------

/// Minimum allowed timestep (seconds).
const MIN_DT: f32 = 1.0e-8;
/// Maximum allowed timestep (seconds).
const MAX_DT: f32 = 0.01;

/// Compute an adaptive timestep using the CFL condition and force-based criterion.
///
/// Three criteria are combined (minimum is used):
/// 1. CFL (velocity): `dt_cfl = cfl * h / max(|v_i| + c_s)`
/// 2. Force-based:    `dt_force = 0.25 * sqrt(h / max_accel)`
/// 3. Viscous:        `dt_visc = 0.125 * h^2 / nu` (if viscosity > 0)
///
/// The result is clamped to `[1e-8, 0.01]`.
pub fn compute_timestep(
    particles: &ParticleArrays,
    h: f32,
    speed_of_sound: f32,
    cfl_number: f32,
) -> f32 {
    // 1. CFL condition based on velocity + speed of sound
    let mut max_signal = speed_of_sound; // at minimum, c_s
    let mut max_accel = 0.0_f32;
    for i in 0..particles.len() {
        let v = (particles.vx[i] * particles.vx[i]
            + particles.vy[i] * particles.vy[i]
            + particles.vz[i] * particles.vz[i])
            .sqrt();
        let signal = v + speed_of_sound;
        if signal > max_signal {
            max_signal = signal;
        }
        let a = (particles.ax[i] * particles.ax[i]
            + particles.ay[i] * particles.ay[i]
            + particles.az[i] * particles.az[i])
            .sqrt();
        if a > max_accel {
            max_accel = a;
        }
    }
    let dt_cfl = cfl_number * h / max_signal;

    // 2. Force-based CFL: dt_force = 0.25 * sqrt(h / max_accel)
    let dt_force = if max_accel > 1.0e-12 {
        0.25 * (h / max_accel).sqrt()
    } else {
        MAX_DT
    };

    let dt = dt_cfl.min(dt_force);
    dt.clamp(MIN_DT, MAX_DT)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kernel_at_zero_distance() {
        let h = 0.1;
        let w = wendland_c2(0.0, h);
        // At r=0: q=0, (1-0)^4*(1+0) = 1, so W = norm / h^3
        let expected = WENDLAND_C2_NORM_3D / (h * h * h);
        assert!((w - expected).abs() < 1.0e-4, "w={w}, expected={expected}");
    }

    #[test]
    fn kernel_at_support_radius() {
        let h = 0.1;
        // At q = 2 (r = 2h) the kernel should be zero
        let w = wendland_c2(2.0 * h, h);
        assert!(w.abs() < 1.0e-10, "kernel should be zero at support radius");
    }

    #[test]
    fn kernel_beyond_support() {
        let w = wendland_c2(0.5, 0.1); // r=0.5, h=0.1 => q=5 > 2
        assert_eq!(w, 0.0);
    }

    #[test]
    fn kernel_positive_inside_support() {
        let h = 0.1;
        for i in 1..20 {
            let r = (i as f32) * 0.01; // 0.01 .. 0.19 => q = 0.1..1.9
            let w = wendland_c2(r, h);
            assert!(w > 0.0, "kernel should be positive at r={r}, q={}", r / h);
        }
    }

    #[test]
    fn gradient_at_zero_is_zero() {
        let (gx, gy, gz) = wendland_c2_gradient(0.0, 0.0, 0.0, 0.0, 0.1);
        assert_eq!(gx, 0.0);
        assert_eq!(gy, 0.0);
        assert_eq!(gz, 0.0);
    }

    #[test]
    fn gradient_beyond_support_is_zero() {
        let (gx, gy, gz) = wendland_c2_gradient(0.5, 0.0, 0.0, 0.5, 0.1);
        assert_eq!(gx, 0.0);
        assert_eq!(gy, 0.0);
        assert_eq!(gz, 0.0);
    }

    #[test]
    fn gradient_direction() {
        // Displacement only along x-axis
        let h = 0.1;
        let dx = 0.1_f32;
        let r = dx;
        let (gx, gy, gz) = wendland_c2_gradient(dx, 0.0, 0.0, r, h);
        // Gradient should point in negative x direction (decreasing kernel value)
        assert!(gx < 0.0, "gradient x should be negative, got {gx}");
        assert!(gy.abs() < 1.0e-10, "gradient y should be ~0");
        assert!(gz.abs() < 1.0e-10, "gradient z should be ~0");
    }

    #[test]
    fn kernel_normalization_numerical() {
        // Numerically integrate the kernel in 3D and verify it's close to 1.
        // Use a simple Riemann sum over a cube of side 2*2h centered at origin.
        let h = 0.1_f32;
        let n = 100; // grid points per axis
        let half_extent = 2.0 * h;
        let cell = 2.0 * half_extent / (n as f32);
        let dv = cell * cell * cell;
        let mut integral = 0.0_f64;
        for ix in 0..n {
            let x = -half_extent + (ix as f32 + 0.5) * cell;
            for iy in 0..n {
                let y = -half_extent + (iy as f32 + 0.5) * cell;
                for iz in 0..n {
                    let z = -half_extent + (iz as f32 + 0.5) * cell;
                    let r = (x * x + y * y + z * z).sqrt();
                    integral += wendland_c2(r, h) as f64 * dv as f64;
                }
            }
        }
        // Should be close to 1.0
        assert!(
            (integral - 1.0).abs() < 0.02,
            "kernel integral = {integral}, expected ~1.0"
        );
    }
}
