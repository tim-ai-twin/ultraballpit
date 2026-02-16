//! Boundary particle data and pressure mirroring.
//!
//! Boundary particles are static particles placed along walls and obstacles.
//! They participate in density summation and exert pressure forces on fluid
//! particles, but never move themselves.
//!
//! Pressure mirroring follows Adami et al. (2012): boundary particle pressure
//! is extrapolated from the nearest fluid particle using the hydrostatic
//! correction.

use crate::neighbor::NeighborGrid;
use crate::particle::ParticleArrays;
use crate::sph::wendland_c2;

/// Boundary particle data stored in SoA (struct-of-arrays) format.
///
/// Boundary particles have fixed positions and outward normals. Their pressure
/// is updated each timestep by mirroring from the nearest fluid particles.
#[derive(Clone)]
pub struct BoundaryParticles {
    /// X positions (meters)
    pub x: Vec<f32>,
    /// Y positions (meters)
    pub y: Vec<f32>,
    /// Z positions (meters)
    pub z: Vec<f32>,
    /// Particle mass (kg)
    pub mass: Vec<f32>,
    /// Outward normal X component
    pub nx: Vec<f32>,
    /// Outward normal Y component
    pub ny: Vec<f32>,
    /// Outward normal Z component
    pub nz: Vec<f32>,
    /// Pressure (Pa), mirrored from nearest fluid particle
    pub pressure: Vec<f32>,
}

impl BoundaryParticles {
    /// Create an empty boundary particle collection.
    pub fn new() -> Self {
        Self {
            x: Vec::new(),
            y: Vec::new(),
            z: Vec::new(),
            mass: Vec::new(),
            nx: Vec::new(),
            ny: Vec::new(),
            nz: Vec::new(),
            pressure: Vec::new(),
        }
    }

    /// Return the number of boundary particles.
    pub fn len(&self) -> usize {
        self.x.len()
    }

    /// Return `true` if there are no boundary particles.
    pub fn is_empty(&self) -> bool {
        self.x.is_empty()
    }

    /// Append a single boundary particle.
    ///
    /// Pressure is initialized to zero and will be updated via `update_pressures`.
    pub fn push(&mut self, x: f32, y: f32, z: f32, mass: f32, nx: f32, ny: f32, nz: f32) {
        self.x.push(x);
        self.y.push(y);
        self.z.push(z);
        self.mass.push(mass);
        self.nx.push(nx);
        self.ny.push(ny);
        self.nz.push(nz);
        self.pressure.push(0.0);
    }

    /// Update boundary particle pressures from nearest fluid particles.
    ///
    /// Implements the Adami et al. (2012) pressure mirroring scheme:
    /// ```text
    /// P_boundary = sum_f W(r_bf, h) * (P_f + rho_f * g . (r_b - r_f)) / sum_f W(r_bf, h)
    /// ```
    ///
    /// This is a kernel-weighted average of fluid pressures with hydrostatic
    /// correction for the height difference.
    pub fn update_pressures(
        &mut self,
        particles: &ParticleArrays,
        _grid: &NeighborGrid,
        gravity: [f32; 3],
        h: f32,
    ) {
        let support_radius = 2.0 * h;
        let n_fluid = particles.len();

        for b in 0..self.len() {
            let bx = self.x[b];
            let by = self.y[b];
            let bz = self.z[b];

            let mut weighted_pressure = 0.0_f32;
            let mut weight_sum = 0.0_f32;

            // Search fluid particles near this boundary particle
            // We iterate over all fluid particles and check distance.
            // This is O(N_boundary * N_fluid) but correct; for large simulations
            // one would build a separate grid for boundary lookups.
            for f in 0..n_fluid {
                let dx = bx - particles.x[f];
                let dy = by - particles.y[f];
                let dz = bz - particles.z[f];
                let r = (dx * dx + dy * dy + dz * dz).sqrt();

                if r < support_radius {
                    let w = wendland_c2(r, h);
                    // Hydrostatic correction: g . (r_b - r_f)
                    let g_dot_dr = gravity[0] * dx + gravity[1] * dy + gravity[2] * dz;
                    let p_extrapolated = particles.pressure[f] + particles.density[f] * g_dot_dr;
                    weighted_pressure += w * p_extrapolated;
                    weight_sum += w;
                }
            }

            if weight_sum > 1.0e-12 {
                self.pressure[b] = (weighted_pressure / weight_sum).max(0.0);
            } else {
                self.pressure[b] = 0.0;
            }
        }
    }

    /// Compute repulsive boundary forces on fluid particles.
    ///
    /// Uses a smooth polynomial repulsive force as a last-resort safety net
    /// to prevent fluid particles from penetrating the boundary.
    ///
    /// The force activates only within 0.5*h of a boundary particle and is
    /// scaled to hydrostatic pressure level (~10 * g * H) rather than c_s^2,
    /// avoiding the extreme accelerations (~200,000g) that caused blow-up.
    ///
    /// The force acts along the direction from boundary to fluid particle.
    pub fn compute_repulsive_forces(
        &self,
        particles: &mut ParticleArrays,
        h: f32,
        _speed_of_sound: f32,
    ) {
        // Tighten cutoff to 0.5*h -- only activate as last-resort safety net
        let r0 = 0.5 * h;
        // Scale force to hydrostatic pressure level instead of c_s^2.
        // For a 1cm domain with g=9.81: d ~ 10 * 9.81 * 0.01 = 0.981
        // This is ~2500x smaller than the old c_s^2 = 2500 value.
        let d = 10.0 * 9.81 * 0.01;

        for i in 0..particles.len() {
            let px = particles.x[i];
            let py = particles.y[i];
            let pz = particles.z[i];

            for b in 0..self.len() {
                let dx = px - self.x[b];
                let dy = py - self.y[b];
                let dz = pz - self.z[b];
                let r = (dx * dx + dy * dy + dz * dz).sqrt();

                if r < r0 && r > 1.0e-12 {
                    // Smooth polynomial repulsion: F ~ (1 - r/r0)^2
                    // This avoids the singularity of LJ-type forces
                    let s = 1.0 - r / r0; // s in (0, 1]
                    let force_mag = d * s * s / r0;

                    // Apply force along the direction from boundary to fluid particle
                    let inv_r = 1.0 / r;
                    particles.ax[i] += force_mag * dx * inv_r;
                    particles.ay[i] += force_mag * dy * inv_r;
                    particles.az[i] += force_mag * dz * inv_r;
                }
            }
        }
    }

    /// Enforce no-penetration by projecting fluid particles back if they
    /// cross any boundary plane.
    ///
    /// Each boundary particle defines a local plane via its position and
    /// outward normal. If a fluid particle is on the "wrong side" of this
    /// plane (inside the boundary), it is projected back to the plane surface
    /// and its normal velocity component is reflected with damping.
    pub fn enforce_no_penetration(&self, particles: &mut ParticleArrays) {
        if self.is_empty() {
            return;
        }

        // Check each fluid particle against each boundary particle
        for i in 0..particles.len() {
            for b in 0..self.len() {
                let bnx = self.nx[b];
                let bny = self.ny[b];
                let bnz = self.nz[b];

                // Vector from boundary to fluid particle
                let dx = particles.x[i] - self.x[b];
                let dy = particles.y[i] - self.y[b];
                let dz = particles.z[i] - self.z[b];

                // Distance along normal
                let d_normal = dx * bnx + dy * bny + dz * bnz;

                // If the particle is behind the boundary (on the wrong side of normal)
                if d_normal < 0.0 {
                    // Project particle back to the boundary plane
                    particles.x[i] -= d_normal * bnx;
                    particles.y[i] -= d_normal * bny;
                    particles.z[i] -= d_normal * bnz;

                    // Reflect and damp the normal velocity component
                    let v_normal = particles.vx[i] * bnx
                        + particles.vy[i] * bny
                        + particles.vz[i] * bnz;

                    if v_normal < 0.0 {
                        // Remove normal velocity component and add a damped reflection
                        let restitution = 0.2; // heavy damping for quick settling
                        particles.vx[i] -= (1.0 + restitution) * v_normal * bnx;
                        particles.vy[i] -= (1.0 + restitution) * v_normal * bny;
                        particles.vz[i] -= (1.0 + restitution) * v_normal * bnz;
                    }
                }
            }
        }
    }

    /// Enforce no-penetration using simple axis-aligned domain bound clamping.
    ///
    /// This replaces the O(N*M) per-boundary-particle loop with 6 simple
    /// axis-aligned checks against domain_min/domain_max. Much faster and
    /// avoids artifacts from checking against individual boundary particles.
    pub fn enforce_no_penetration_domain(
        &self,
        particles: &mut ParticleArrays,
        domain_min: [f32; 3],
        domain_max: [f32; 3],
    ) {
        let restitution = 0.2_f32; // heavy damping for quick settling

        for i in 0..particles.len() {
            // X-min wall
            if particles.x[i] < domain_min[0] {
                particles.x[i] = domain_min[0];
                if particles.vx[i] < 0.0 {
                    particles.vx[i] = -restitution * particles.vx[i];
                }
            }
            // X-max wall
            if particles.x[i] > domain_max[0] {
                particles.x[i] = domain_max[0];
                if particles.vx[i] > 0.0 {
                    particles.vx[i] = -restitution * particles.vx[i];
                }
            }
            // Y-min wall
            if particles.y[i] < domain_min[1] {
                particles.y[i] = domain_min[1];
                if particles.vy[i] < 0.0 {
                    particles.vy[i] = -restitution * particles.vy[i];
                }
            }
            // Y-max wall
            if particles.y[i] > domain_max[1] {
                particles.y[i] = domain_max[1];
                if particles.vy[i] > 0.0 {
                    particles.vy[i] = -restitution * particles.vy[i];
                }
            }
            // Z-min wall
            if particles.z[i] < domain_min[2] {
                particles.z[i] = domain_min[2];
                if particles.vz[i] < 0.0 {
                    particles.vz[i] = -restitution * particles.vz[i];
                }
            }
            // Z-max wall
            if particles.z[i] > domain_max[2] {
                particles.z[i] = domain_max[2];
                if particles.vz[i] > 0.0 {
                    particles.vz[i] = -restitution * particles.vz[i];
                }
            }
        }
    }
}

impl Default for BoundaryParticles {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_boundary() {
        let bp = BoundaryParticles::new();
        assert_eq!(bp.len(), 0);
        assert!(bp.is_empty());
    }

    #[test]
    fn push_and_len() {
        let mut bp = BoundaryParticles::new();
        bp.push(0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0);
        assert_eq!(bp.len(), 1);
        assert!(!bp.is_empty());
        assert_eq!(bp.ny[0], 1.0);
        assert_eq!(bp.pressure[0], 0.0);
    }
}
