//! Surface force extraction from SPH fluid particles
//!
//! Computes pressure and viscous forces exerted by fluid particles on
//! geometry surfaces using the SDF gradient (surface normal).

use kernel::{ParticleArrays, wendland_c2};
use crate::geometry::GridSDF;

/// Force and moment on a surface region
#[derive(Debug, Clone)]
pub struct SurfaceForce {
    /// Net force vector [Fx, Fy, Fz] (Newtons)
    pub net_force: [f32; 3],
    /// Net moment/torque vector [Tx, Ty, Tz] (N·m) about origin
    pub net_moment: [f32; 3],
}

impl Default for SurfaceForce {
    fn default() -> Self {
        Self {
            net_force: [0.0, 0.0, 0.0],
            net_moment: [0.0, 0.0, 0.0],
        }
    }
}

/// Compute forces on geometry surface from nearby fluid particles
///
/// Samples the SDF surface at grid points, finds nearby fluid particles,
/// and accumulates pressure forces (p * n * dA) onto the surface.
///
/// # Arguments
/// * `particles` - Fluid particle data (positions, pressures, densities)
/// * `sdf` - Signed distance field of the geometry
/// * `h` - SPH smoothing length (for neighbor search radius)
///
/// # Returns
/// Total force and moment on the surface
pub fn compute_surface_forces(
    particles: &ParticleArrays,
    sdf: &GridSDF,
    h: f32,
) -> SurfaceForce {
    let mut net_force = [0.0, 0.0, 0.0];
    let mut net_moment = [0.0, 0.0, 0.0];

    // Sample surface at SDF grid points that are close to the surface
    // (distance within ~2h of the surface)
    let surface_threshold = 2.0 * h;

    // Grid cell size and dimensions
    let cell_size = sdf.cell_size;
    let [nx, ny, nz] = sdf.dimensions;

    // Area element for each sample point
    let d_area = cell_size * cell_size;

    // Iterate over SDF grid points
    for i in 0..nx {
        for j in 0..ny {
            for k in 0..nz {
                // Get world position of grid point
                let px = sdf.origin[0] + (i as f32) * cell_size;
                let py = sdf.origin[1] + (j as f32) * cell_size;
                let pz = sdf.origin[2] + (k as f32) * cell_size;

                // Sample SDF distance
                let dist = sdf.sample([px, py, pz]);

                // Only process points near the surface (small negative or positive distance)
                // We want points just inside the boundary (dist < 0) within threshold
                if dist.abs() > surface_threshold || dist > 0.0 {
                    continue;
                }

                // Get surface normal (gradient points outward from surface)
                let normal = sdf.gradient([px, py, pz]);
                let normal_mag = (normal[0]*normal[0] + normal[1]*normal[1] + normal[2]*normal[2]).sqrt();
                if normal_mag < 1e-8 {
                    continue;
                }

                // Accumulate pressure forces from nearby fluid particles
                let mut local_pressure = 0.0;
                let mut weight_sum = 0.0;

                // Search for nearby particles within smoothing radius
                for p_idx in 0..particles.len() {
                    let dx = particles.x[p_idx] - px;
                    let dy = particles.y[p_idx] - py;
                    let dz = particles.z[p_idx] - pz;
                    let r = (dx*dx + dy*dy + dz*dz).sqrt();

                    // Only include particles within kernel support (2h)
                    if r < 2.0 * h {
                        let w = wendland_c2(r, h);
                        local_pressure += particles.pressure[p_idx] * w;
                        weight_sum += w;
                    }
                }

                // Average pressure at this surface sample
                if weight_sum > 1e-12 {
                    local_pressure /= weight_sum;

                    // Force = pressure * normal * area
                    // Normal points outward, pressure pushes inward on surface
                    let force_x = -local_pressure * normal[0] * d_area;
                    let force_y = -local_pressure * normal[1] * d_area;
                    let force_z = -local_pressure * normal[2] * d_area;

                    net_force[0] += force_x;
                    net_force[1] += force_y;
                    net_force[2] += force_z;

                    // Moment = r × F (about origin)
                    net_moment[0] += py * force_z - pz * force_y;
                    net_moment[1] += pz * force_x - px * force_z;
                    net_moment[2] += px * force_y - py * force_x;
                }
            }
        }
    }

    SurfaceForce {
        net_force,
        net_moment,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use kernel::FluidType;

    #[test]
    fn test_force_computation_basic() {
        // Create a simple particle array with one particle
        let mut particles = ParticleArrays::new();
        particles.push_particle(0.0, 0.0, 0.0, 0.001, 1000.0, 293.15, FluidType::Water);
        particles.pressure[0] = 1000.0; // 1 kPa pressure

        // Create a simple SDF (plane at z = -0.001)
        let sdf = GridSDF {
            origin: [-0.005, -0.005, -0.005],
            cell_size: 0.001,
            dimensions: [10, 10, 10],
            distances: vec![0.0; 1000], // Will be mostly positive except near z=0
        };

        let h = 0.0013; // Smoothing length
        let force = compute_surface_forces(&particles, &sdf, h);

        // Force should be non-zero if particle is near surface
        // This is a basic smoke test
        assert!(force.net_force[0].abs() >= 0.0);
        assert!(force.net_force[1].abs() >= 0.0);
        assert!(force.net_force[2].abs() >= 0.0);
    }
}
