//! Domain setup: particle placement and boundary particle generation

use kernel::{FluidType, ParticleArrays};
use crate::config::{ConfigFluidType, SimulationConfig};
use crate::geometry::GridSDF;

/// Boundary particle data for walls and geometry surfaces
#[derive(Debug, Clone)]
pub struct BoundaryParticleData {
    /// X position
    pub x: f32,
    /// Y position
    pub y: f32,
    /// Z position
    pub z: f32,
    /// Particle mass
    pub mass: f32,
    /// Outward normal X component
    pub nx: f32,
    /// Outward normal Y component
    pub ny: f32,
    /// Outward normal Z component
    pub nz: f32,
}

/// Set up the simulation domain with fluid and boundary particles
///
/// Returns (fluid_particles, boundary_particles)
pub fn setup_domain(
    config: &SimulationConfig,
    sdf: &GridSDF,
) -> (ParticleArrays, Vec<BoundaryParticleData>) {
    let mut fluid_particles = ParticleArrays::new();
    let mut boundary_particles = Vec::new();

    // Calculate particle mass based on spacing and fluid type
    let volume_per_particle = config.particle_spacing.powi(3);
    let water_mass = 1000.0 * volume_per_particle; // Water density: 1000 kg/m^3
    let air_mass = 1.2 * volume_per_particle;      // Air density: 1.2 kg/m^3

    // Place fluid particles on a regular grid
    place_fluid_particles(
        &mut fluid_particles,
        config,
        sdf,
        water_mass,
        air_mass,
    );

    // Generate boundary particles on domain walls
    generate_wall_boundary_particles(
        &mut boundary_particles,
        config,
        water_mass,
    );

    // Generate boundary particles on geometry surfaces
    generate_geometry_boundary_particles(
        &mut boundary_particles,
        config,
        sdf,
        water_mass,
    );

    tracing::info!(
        "Domain setup complete: {} fluid particles, {} boundary particles",
        fluid_particles.len(),
        boundary_particles.len()
    );

    (fluid_particles, boundary_particles)
}

/// Place fluid particles in the domain
fn place_fluid_particles(
    particles: &mut ParticleArrays,
    config: &SimulationConfig,
    sdf: &GridSDF,
    water_mass: f32,
    air_mass: f32,
) {
    let spacing = config.particle_spacing;
    let domain_min = config.domain.min;
    let domain_max = config.domain.max;

    // Calculate domain center Y for Mixed fluid type
    let domain_center_y = (domain_min[1] + domain_max[1]) * 0.5;

    // Grid dimensions
    let nx = ((domain_max[0] - domain_min[0]) / spacing).ceil() as usize;
    let ny = ((domain_max[1] - domain_min[1]) / spacing).ceil() as usize;
    let nz = ((domain_max[2] - domain_min[2]) / spacing).ceil() as usize;

    // Reserve capacity
    particles.x.reserve(nx * ny * nz);

    // Place particles on grid
    for i in 0..nx {
        for j in 0..ny {
            for k in 0..nz {
                let x = domain_min[0] + (i as f32 + 0.5) * spacing;
                let y = domain_min[1] + (j as f32 + 0.5) * spacing;
                let z = domain_min[2] + (k as f32 + 0.5) * spacing;

                // Check if position is inside domain
                if x < domain_min[0] || x > domain_max[0]
                    || y < domain_min[1] || y > domain_max[1]
                    || z < domain_min[2] || z > domain_max[2]
                {
                    continue;
                }

                // Check SDF: skip if inside geometry (negative SDF)
                let dist = sdf.sample([x, y, z]);
                if dist < 0.0 {
                    continue; // Inside solid geometry
                }

                // Determine fluid type based on configuration
                let (fluid_type, mass, density) = match config.fluid_type {
                    ConfigFluidType::Water => (FluidType::Water, water_mass, 1000.0),
                    ConfigFluidType::Air => (FluidType::Air, air_mass, 1.2),
                    ConfigFluidType::Mixed => {
                        // Bottom half is water, top half is air
                        if y < domain_center_y {
                            (FluidType::Water, water_mass, 1000.0)
                        } else {
                            (FluidType::Air, air_mass, 1.2)
                        }
                    }
                };

                // Add particle
                particles.push_particle(
                    x,
                    y,
                    z,
                    mass,
                    density,
                    config.initial_temperature,
                    fluid_type,
                );
            }
        }
    }
}

/// Generate boundary particles on domain walls
fn generate_wall_boundary_particles(
    boundary_particles: &mut Vec<BoundaryParticleData>,
    config: &SimulationConfig,
    mass: f32,
) {
    use crate::config::{BoundaryType, SimpleBoundary};

    let spacing = config.particle_spacing;
    let domain_min = config.domain.min;
    let domain_max = config.domain.max;

    // X-min face (yz plane)
    if matches!(config.boundary_conditions.x_min,
        BoundaryType::Simple(SimpleBoundary::Wall))
    {
        let ny = ((domain_max[1] - domain_min[1]) / spacing).ceil() as usize;
        let nz = ((domain_max[2] - domain_min[2]) / spacing).ceil() as usize;

        for j in 0..ny {
            for k in 0..nz {
                let y = domain_min[1] + (j as f32 + 0.5) * spacing;
                let z = domain_min[2] + (k as f32 + 0.5) * spacing;

                boundary_particles.push(BoundaryParticleData {
                    x: domain_min[0],
                    y,
                    z,
                    mass,
                    nx: 1.0, // Outward normal (pointing into domain)
                    ny: 0.0,
                    nz: 0.0,
                });
            }
        }
    }

    // X-max face
    if matches!(config.boundary_conditions.x_max,
        BoundaryType::Simple(SimpleBoundary::Wall))
    {
        let ny = ((domain_max[1] - domain_min[1]) / spacing).ceil() as usize;
        let nz = ((domain_max[2] - domain_min[2]) / spacing).ceil() as usize;

        for j in 0..ny {
            for k in 0..nz {
                let y = domain_min[1] + (j as f32 + 0.5) * spacing;
                let z = domain_min[2] + (k as f32 + 0.5) * spacing;

                boundary_particles.push(BoundaryParticleData {
                    x: domain_max[0],
                    y,
                    z,
                    mass,
                    nx: -1.0, // Outward normal
                    ny: 0.0,
                    nz: 0.0,
                });
            }
        }
    }

    // Y-min face (xz plane)
    if matches!(config.boundary_conditions.y_min,
        BoundaryType::Simple(SimpleBoundary::Wall))
    {
        let nx = ((domain_max[0] - domain_min[0]) / spacing).ceil() as usize;
        let nz = ((domain_max[2] - domain_min[2]) / spacing).ceil() as usize;

        for i in 0..nx {
            for k in 0..nz {
                let x = domain_min[0] + (i as f32 + 0.5) * spacing;
                let z = domain_min[2] + (k as f32 + 0.5) * spacing;

                boundary_particles.push(BoundaryParticleData {
                    x,
                    y: domain_min[1],
                    z,
                    mass,
                    nx: 0.0,
                    ny: 1.0, // Outward normal
                    nz: 0.0,
                });
            }
        }
    }

    // Y-max face
    if matches!(config.boundary_conditions.y_max,
        BoundaryType::Simple(SimpleBoundary::Wall))
    {
        let nx = ((domain_max[0] - domain_min[0]) / spacing).ceil() as usize;
        let nz = ((domain_max[2] - domain_min[2]) / spacing).ceil() as usize;

        for i in 0..nx {
            for k in 0..nz {
                let x = domain_min[0] + (i as f32 + 0.5) * spacing;
                let z = domain_min[2] + (k as f32 + 0.5) * spacing;

                boundary_particles.push(BoundaryParticleData {
                    x,
                    y: domain_max[1],
                    z,
                    mass,
                    nx: 0.0,
                    ny: -1.0, // Outward normal
                    nz: 0.0,
                });
            }
        }
    }

    // Z-min face (xy plane)
    if matches!(config.boundary_conditions.z_min,
        BoundaryType::Simple(SimpleBoundary::Wall))
    {
        let nx = ((domain_max[0] - domain_min[0]) / spacing).ceil() as usize;
        let ny = ((domain_max[1] - domain_min[1]) / spacing).ceil() as usize;

        for i in 0..nx {
            for j in 0..ny {
                let x = domain_min[0] + (i as f32 + 0.5) * spacing;
                let y = domain_min[1] + (j as f32 + 0.5) * spacing;

                boundary_particles.push(BoundaryParticleData {
                    x,
                    y,
                    z: domain_min[2],
                    mass,
                    nx: 0.0,
                    ny: 0.0,
                    nz: 1.0, // Outward normal
                });
            }
        }
    }

    // Z-max face
    if matches!(config.boundary_conditions.z_max,
        BoundaryType::Simple(SimpleBoundary::Wall))
    {
        let nx = ((domain_max[0] - domain_min[0]) / spacing).ceil() as usize;
        let ny = ((domain_max[1] - domain_min[1]) / spacing).ceil() as usize;

        for i in 0..nx {
            for j in 0..ny {
                let x = domain_min[0] + (i as f32 + 0.5) * spacing;
                let y = domain_min[1] + (j as f32 + 0.5) * spacing;

                boundary_particles.push(BoundaryParticleData {
                    x,
                    y,
                    z: domain_max[2],
                    mass,
                    nx: 0.0,
                    ny: 0.0,
                    nz: -1.0, // Outward normal
                });
            }
        }
    }
}

/// Generate boundary particles on geometry surfaces
fn generate_geometry_boundary_particles(
    boundary_particles: &mut Vec<BoundaryParticleData>,
    config: &SimulationConfig,
    sdf: &GridSDF,
    mass: f32,
) {
    let spacing = config.particle_spacing;
    let domain_min = config.domain.min;
    let domain_max = config.domain.max;

    // Sample geometry surface at particle spacing resolution
    // Look for zero-crossings in the SDF
    let nx = ((domain_max[0] - domain_min[0]) / spacing).ceil() as usize;
    let ny = ((domain_max[1] - domain_min[1]) / spacing).ceil() as usize;
    let nz = ((domain_max[2] - domain_min[2]) / spacing).ceil() as usize;

    let surface_threshold = spacing * 0.5; // Within half a particle spacing of surface

    for i in 0..nx {
        for j in 0..ny {
            for k in 0..nz {
                let x = domain_min[0] + (i as f32 + 0.5) * spacing;
                let y = domain_min[1] + (j as f32 + 0.5) * spacing;
                let z = domain_min[2] + (k as f32 + 0.5) * spacing;

                let dist = sdf.sample([x, y, z]);

                // If near the surface (small positive distance), place a boundary particle
                if dist > 0.0 && dist < surface_threshold {
                    let normal = sdf.gradient([x, y, z]);

                    boundary_particles.push(BoundaryParticleData {
                        x,
                        y,
                        z,
                        mass,
                        nx: normal[0],
                        ny: normal[1],
                        nz: normal[2],
                    });
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{DomainBounds, BoundaryConditions};

    #[test]
    fn test_setup_domain_basic() {
        let config = SimulationConfig {
            name: "test".to_string(),
            fluid_type: ConfigFluidType::Water,
            geometry_file: "test.stl".to_string(),
            domain: DomainBounds {
                min: [0.0, 0.0, 0.0],
                max: [0.01, 0.01, 0.01],
            },
            boundary_conditions: BoundaryConditions::default(),
            particle_spacing: 0.005,
            gravity: [0.0, -9.81, 0.0],
            speed_of_sound: 50.0,
            viscosity: 0.001,
            initial_temperature: 293.15,
            max_timesteps: None,
            max_time: None,
            cfl_number: 0.4,
        };

        // Create empty SDF (no geometry)
        let sdf = GridSDF {
            origin: [0.0, 0.0, 0.0],
            cell_size: 0.005,
            dimensions: [3, 3, 3],
            distances: vec![1.0; 27], // All positive (outside)
        };

        let (fluid_particles, boundary_particles) = setup_domain(&config, &sdf);

        // Should have some fluid particles
        assert!(fluid_particles.len() > 0);
        // Should have boundary particles (6 walls)
        assert!(boundary_particles.len() > 0);

        // All fluid particles should be water
        for ft in &fluid_particles.fluid_type {
            assert_eq!(*ft, FluidType::Water);
        }
    }

    #[test]
    fn test_mixed_fluid_type() {
        let config = SimulationConfig {
            name: "test".to_string(),
            fluid_type: ConfigFluidType::Mixed,
            geometry_file: "test.stl".to_string(),
            domain: DomainBounds {
                min: [0.0, 0.0, 0.0],
                max: [0.01, 0.02, 0.01], // Taller domain
            },
            boundary_conditions: BoundaryConditions::default(),
            particle_spacing: 0.005,
            gravity: [0.0, -9.81, 0.0],
            speed_of_sound: 50.0,
            viscosity: 0.001,
            initial_temperature: 293.15,
            max_timesteps: None,
            max_time: None,
            cfl_number: 0.4,
        };

        let sdf = GridSDF {
            origin: [0.0, 0.0, 0.0],
            cell_size: 0.005,
            dimensions: [3, 5, 3],
            distances: vec![1.0; 45], // All positive
        };

        let (fluid_particles, _) = setup_domain(&config, &sdf);

        // Should have both water and air particles
        let has_water = fluid_particles.fluid_type.iter().any(|&ft| ft == FluidType::Water);
        let has_air = fluid_particles.fluid_type.iter().any(|&ft| ft == FluidType::Air);

        assert!(has_water, "Mixed domain should have water particles");
        assert!(has_air, "Mixed domain should have air particles");
    }
}
