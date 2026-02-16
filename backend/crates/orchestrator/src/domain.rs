//! Domain setup: particle placement, boundary particle generation, and domain decomposition
//!
//! This module handles:
//! - Initial particle placement on a regular grid
//! - Wall and geometry boundary particle generation
//! - Domain decomposition for distributed parallel execution (T066)
//! - Ghost particle exchange between adjacent subdomains (T067)

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

    // Generate boundary particles on domain walls.
    // Boundary mass must match the fluid rest density for correct SPH density estimation.
    let boundary_mass = match config.fluid_type {
        ConfigFluidType::Water => water_mass,
        ConfigFluidType::Air => air_mass,
        // For mixed: use water mass for boundary (dominates density near walls)
        ConfigFluidType::Mixed => water_mass,
    };
    generate_wall_boundary_particles(
        &mut boundary_particles,
        config,
        boundary_mass,
    );

    // Generate boundary particles on geometry surfaces
    generate_geometry_boundary_particles(
        &mut boundary_particles,
        config,
        sdf,
        boundary_mass,
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

    // Number of boundary particle layers per wall (Adami et al. 2012 needs >= 3)
    let n_layers: usize = 3;

    // Boundary layers are offset by (layer + 0.5) * spacing outward from the
    // domain face.  This mirrors the fluid particle grid across the wall:
    //   fluid at  domain_min + 0.5 * spacing
    //   layer 0   domain_min - 0.5 * spacing
    //   layer 1   domain_min - 1.5 * spacing
    //   layer 2   domain_min - 2.5 * spacing
    //
    // Without the 0.5-offset, layer 0 sits AT the domain edge, only 0.5 spacing
    // from the first fluid layer (vs 1.0 between fluid layers), inflating SPH
    // density by ~25% near walls and producing wildly wrong pressures.

    // X-min face (yz plane), inward normal = (+1, 0, 0)
    if matches!(config.boundary_conditions.x_min,
        BoundaryType::Simple(SimpleBoundary::Wall))
    {
        let ny = ((domain_max[1] - domain_min[1]) / spacing).ceil() as usize;
        let nz = ((domain_max[2] - domain_min[2]) / spacing).ceil() as usize;

        for layer in 0..n_layers {
            for j in 0..ny {
                for k in 0..nz {
                    let y = domain_min[1] + (j as f32 + 0.5) * spacing;
                    let z = domain_min[2] + (k as f32 + 0.5) * spacing;

                    boundary_particles.push(BoundaryParticleData {
                        x: domain_min[0] - (layer as f32 + 0.5) * spacing,
                        y,
                        z,
                        mass,
                        nx: 1.0,
                        ny: 0.0,
                        nz: 0.0,
                    });
                }
            }
        }
    }

    // X-max face, inward normal = (-1, 0, 0)
    if matches!(config.boundary_conditions.x_max,
        BoundaryType::Simple(SimpleBoundary::Wall))
    {
        let ny = ((domain_max[1] - domain_min[1]) / spacing).ceil() as usize;
        let nz = ((domain_max[2] - domain_min[2]) / spacing).ceil() as usize;

        for layer in 0..n_layers {
            for j in 0..ny {
                for k in 0..nz {
                    let y = domain_min[1] + (j as f32 + 0.5) * spacing;
                    let z = domain_min[2] + (k as f32 + 0.5) * spacing;

                    boundary_particles.push(BoundaryParticleData {
                        x: domain_max[0] + (layer as f32 + 0.5) * spacing,
                        y,
                        z,
                        mass,
                        nx: -1.0,
                        ny: 0.0,
                        nz: 0.0,
                    });
                }
            }
        }
    }

    // Y-min face (xz plane), inward normal = (0, +1, 0)
    if matches!(config.boundary_conditions.y_min,
        BoundaryType::Simple(SimpleBoundary::Wall))
    {
        let nx = ((domain_max[0] - domain_min[0]) / spacing).ceil() as usize;
        let nz = ((domain_max[2] - domain_min[2]) / spacing).ceil() as usize;

        for layer in 0..n_layers {
            for i in 0..nx {
                for k in 0..nz {
                    let x = domain_min[0] + (i as f32 + 0.5) * spacing;
                    let z = domain_min[2] + (k as f32 + 0.5) * spacing;

                    boundary_particles.push(BoundaryParticleData {
                        x,
                        y: domain_min[1] - (layer as f32 + 0.5) * spacing,
                        z,
                        mass,
                        nx: 0.0,
                        ny: 1.0,
                        nz: 0.0,
                    });
                }
            }
        }
    }

    // Y-max face, inward normal = (0, -1, 0)
    if matches!(config.boundary_conditions.y_max,
        BoundaryType::Simple(SimpleBoundary::Wall))
    {
        let nx = ((domain_max[0] - domain_min[0]) / spacing).ceil() as usize;
        let nz = ((domain_max[2] - domain_min[2]) / spacing).ceil() as usize;

        for layer in 0..n_layers {
            for i in 0..nx {
                for k in 0..nz {
                    let x = domain_min[0] + (i as f32 + 0.5) * spacing;
                    let z = domain_min[2] + (k as f32 + 0.5) * spacing;

                    boundary_particles.push(BoundaryParticleData {
                        x,
                        y: domain_max[1] + (layer as f32 + 0.5) * spacing,
                        z,
                        mass,
                        nx: 0.0,
                        ny: -1.0,
                        nz: 0.0,
                    });
                }
            }
        }
    }

    // Z-min face (xy plane), inward normal = (0, 0, +1)
    if matches!(config.boundary_conditions.z_min,
        BoundaryType::Simple(SimpleBoundary::Wall))
    {
        let nx = ((domain_max[0] - domain_min[0]) / spacing).ceil() as usize;
        let ny = ((domain_max[1] - domain_min[1]) / spacing).ceil() as usize;

        for layer in 0..n_layers {
            for i in 0..nx {
                for j in 0..ny {
                    let x = domain_min[0] + (i as f32 + 0.5) * spacing;
                    let y = domain_min[1] + (j as f32 + 0.5) * spacing;

                    boundary_particles.push(BoundaryParticleData {
                        x,
                        y,
                        z: domain_min[2] - (layer as f32 + 0.5) * spacing,
                        mass,
                        nx: 0.0,
                        ny: 0.0,
                        nz: 1.0,
                    });
                }
            }
        }
    }

    // Z-max face, inward normal = (0, 0, -1)
    if matches!(config.boundary_conditions.z_max,
        BoundaryType::Simple(SimpleBoundary::Wall))
    {
        let nx = ((domain_max[0] - domain_min[0]) / spacing).ceil() as usize;
        let ny = ((domain_max[1] - domain_min[1]) / spacing).ceil() as usize;

        for layer in 0..n_layers {
            for i in 0..nx {
                for j in 0..ny {
                    let x = domain_min[0] + (i as f32 + 0.5) * spacing;
                    let y = domain_min[1] + (j as f32 + 0.5) * spacing;

                    boundary_particles.push(BoundaryParticleData {
                        x,
                        y,
                        z: domain_max[2] + (layer as f32 + 0.5) * spacing,
                        mass,
                        nx: 0.0,
                        ny: 0.0,
                        nz: -1.0,
                    });
                }
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
    let n_layers: usize = 3;

    for i in 0..nx {
        for j in 0..ny {
            for k in 0..nz {
                let x = domain_min[0] + (i as f32 + 0.5) * spacing;
                let y = domain_min[1] + (j as f32 + 0.5) * spacing;
                let z = domain_min[2] + (k as f32 + 0.5) * spacing;

                let dist = sdf.sample([x, y, z]);

                // If near the surface (small positive distance), place boundary particles
                // in 3 layers along the inward normal (into the solid)
                if dist > 0.0 && dist < surface_threshold {
                    let normal = sdf.gradient([x, y, z]);

                    for layer in 0..n_layers {
                        // Offset each layer by (layer + 0.5) * spacing along the
                        // inward normal, mirroring the fluid grid across the surface.
                        let offset = (layer as f32 + 0.5) * spacing;
                        boundary_particles.push(BoundaryParticleData {
                            x: x - normal[0] * offset,
                            y: y - normal[1] * offset,
                            z: z - normal[2] * offset,
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
}

// ===========================================================================
// T066: Domain Decomposition Algorithm
// ===========================================================================

/// Axis-aligned bounding box for a subdomain
#[derive(Debug, Clone)]
pub struct AABB {
    /// Minimum corner [x, y, z]
    pub min: [f32; 3],
    /// Maximum corner [x, y, z]
    pub max: [f32; 3],
}

impl AABB {
    /// Create a new AABB from min/max corners
    pub fn new(min: [f32; 3], max: [f32; 3]) -> Self {
        Self { min, max }
    }

    /// Return the extent (size) along each axis
    pub fn extent(&self) -> [f32; 3] {
        [
            self.max[0] - self.min[0],
            self.max[1] - self.min[1],
            self.max[2] - self.min[2],
        ]
    }

    /// Return the index of the longest axis (0=x, 1=y, 2=z)
    pub fn longest_axis(&self) -> usize {
        let ext = self.extent();
        if ext[0] >= ext[1] && ext[0] >= ext[2] {
            0
        } else if ext[1] >= ext[2] {
            1
        } else {
            2
        }
    }

    /// Check if a point is inside this AABB (inclusive)
    pub fn contains(&self, x: f32, y: f32, z: f32) -> bool {
        x >= self.min[0] && x <= self.max[0]
            && y >= self.min[1] && y <= self.max[1]
            && z >= self.min[2] && z <= self.max[2]
    }
}

/// A subdomain produced by domain decomposition, holding its AABB bounds,
/// assigned particles, boundary particles, and neighbor subdomain indices.
#[derive(Debug, Clone)]
pub struct Subdomain {
    /// Unique subdomain index
    pub id: usize,
    /// Axis-aligned bounding box for this subdomain
    pub bounds: AABB,
    /// Fluid particles assigned to this subdomain (owned particles only)
    pub particles: ParticleArrays,
    /// Boundary particles (walls/geometry) that overlap with this subdomain
    pub boundary_particles: Vec<BoundaryParticleData>,
    /// Indices of neighboring subdomains (share a split plane)
    pub neighbor_ids: Vec<usize>,
}

/// Decompose an AABB into `n` subdomains by recursively splitting along the
/// longest axis.
///
/// The splitting is binary: at each level the longest axis of the current AABB
/// is halved. For non-power-of-two counts the left child gets `ceil(n/2)`
/// subdomains and the right child gets `floor(n/2)`.
///
/// # Arguments
/// * `bounds` - The overall AABB to decompose
/// * `n` - Number of desired subdomains (must be >= 1)
///
/// # Returns
/// A vector of `n` AABBs covering the original domain without gaps or overlaps.
pub fn decompose_domain(bounds: &AABB, n: usize) -> Vec<AABB> {
    assert!(n >= 1, "Must have at least 1 subdomain");

    if n == 1 {
        return vec![bounds.clone()];
    }

    let axis = bounds.longest_axis();
    let mid = (bounds.min[axis] + bounds.max[axis]) * 0.5;

    // Split into two halves
    let mut left_max = bounds.max;
    left_max[axis] = mid;
    let left = AABB::new(bounds.min, left_max);

    let mut right_min = bounds.min;
    right_min[axis] = mid;
    let right = AABB::new(right_min, bounds.max);

    // Distribute subdomain count: left gets ceil(n/2), right gets floor(n/2)
    let n_left = (n + 1) / 2;
    let n_right = n - n_left;

    let mut result = decompose_domain(&left, n_left);
    result.extend(decompose_domain(&right, n_right));

    result
}

/// Build subdomains from a decomposed domain, distributing particles and
/// boundary particles to each subdomain and establishing neighbor relationships.
///
/// # Arguments
/// * `aabbs` - Subdomain bounding boxes from `decompose_domain`
/// * `particles` - All fluid particles to distribute
/// * `boundary_data` - All boundary particles to distribute
/// * `smoothing_length` - SPH smoothing length (for ghost region overlap computation)
///
/// # Returns
/// A vector of `Subdomain`s, each with its assigned particles and neighbors.
pub fn build_subdomains(
    aabbs: &[AABB],
    particles: &ParticleArrays,
    boundary_data: &[BoundaryParticleData],
    smoothing_length: f32,
) -> Vec<Subdomain> {
    let n_sub = aabbs.len();
    let support_radius = 2.0 * smoothing_length;

    // Distribute fluid particles to subdomains (each particle goes to exactly one)
    let mut sub_particles: Vec<ParticleArrays> = (0..n_sub).map(|_| ParticleArrays::new()).collect();

    for i in 0..particles.len() {
        let px = particles.x[i];
        let py = particles.y[i];
        let pz = particles.z[i];

        // Find the subdomain that owns this particle
        for (s, aabb) in aabbs.iter().enumerate() {
            if aabb.contains(px, py, pz) {
                sub_particles[s].push_particle(
                    px, py, pz,
                    particles.mass[i],
                    particles.density[i],
                    particles.temperature[i],
                    particles.fluid_type[i],
                );
                // Copy velocity and acceleration
                let idx = sub_particles[s].len() - 1;
                sub_particles[s].vx[idx] = particles.vx[i];
                sub_particles[s].vy[idx] = particles.vy[i];
                sub_particles[s].vz[idx] = particles.vz[i];
                sub_particles[s].ax[idx] = particles.ax[i];
                sub_particles[s].ay[idx] = particles.ay[i];
                sub_particles[s].az[idx] = particles.az[i];
                sub_particles[s].pressure[idx] = particles.pressure[i];
                break;
            }
        }
    }

    // Distribute boundary particles to subdomains (a boundary particle may go
    // to multiple subdomains if near the boundary region)
    let mut sub_boundary: Vec<Vec<BoundaryParticleData>> = (0..n_sub).map(|_| Vec::new()).collect();

    for bp in boundary_data {
        for (s, aabb) in aabbs.iter().enumerate() {
            // Expand AABB by support_radius for boundary particle inclusion
            let expanded_min = [
                aabb.min[0] - support_radius,
                aabb.min[1] - support_radius,
                aabb.min[2] - support_radius,
            ];
            let expanded_max = [
                aabb.max[0] + support_radius,
                aabb.max[1] + support_radius,
                aabb.max[2] + support_radius,
            ];
            if bp.x >= expanded_min[0] && bp.x <= expanded_max[0]
                && bp.y >= expanded_min[1] && bp.y <= expanded_max[1]
                && bp.z >= expanded_min[2] && bp.z <= expanded_max[2]
            {
                sub_boundary[s].push(bp.clone());
            }
        }
    }

    // Determine neighbor relationships: two subdomains are neighbors if their
    // AABBs (expanded by support_radius) overlap.
    let mut neighbors: Vec<Vec<usize>> = (0..n_sub).map(|_| Vec::new()).collect();
    for i in 0..n_sub {
        for j in (i + 1)..n_sub {
            if aabbs_overlap_with_margin(&aabbs[i], &aabbs[j], support_radius) {
                neighbors[i].push(j);
                neighbors[j].push(i);
            }
        }
    }

    // Build subdomains
    let mut subdomains = Vec::with_capacity(n_sub);
    for s in 0..n_sub {
        subdomains.push(Subdomain {
            id: s,
            bounds: aabbs[s].clone(),
            particles: std::mem::take(&mut sub_particles[s]),
            boundary_particles: std::mem::take(&mut sub_boundary[s]),
            neighbor_ids: std::mem::take(&mut neighbors[s]),
        });
    }

    subdomains
}

/// Check if two AABBs overlap when each is expanded by `margin`.
fn aabbs_overlap_with_margin(a: &AABB, b: &AABB, margin: f32) -> bool {
    for axis in 0..3 {
        if a.max[axis] + margin < b.min[axis] - margin {
            return false;
        }
        if b.max[axis] + margin < a.min[axis] - margin {
            return false;
        }
    }
    true
}

// ===========================================================================
// T067: Subdomain Boundary Particle Exchange Protocol (Ghost Particles)
// ===========================================================================

/// Ghost particles received from a neighboring subdomain.
///
/// These are copies of fluid particles near the subdomain boundary that are
/// needed for correct SPH computations near edges. They are treated as
/// read-only during force computation and discarded after each timestep.
#[derive(Debug, Clone)]
pub struct GhostParticles {
    /// Source subdomain ID
    pub source_id: usize,
    /// Ghost particle data (positions, velocities, densities, etc.)
    pub particles: ParticleArrays,
}

/// Extract particles from a subdomain that lie within the ghost region of
/// a neighbor subdomain. The ghost region is defined as the overlap zone:
/// particles within `overlap` distance of the neighbor's boundary.
///
/// # Arguments
/// * `source` - The subdomain whose particles we are extracting from
/// * `neighbor_bounds` - The AABB of the neighbor subdomain
/// * `overlap` - Ghost region width (typically 2 * smoothing_length)
///
/// # Returns
/// A `ParticleArrays` containing copies of particles in the ghost region.
pub fn extract_ghost_particles(
    source: &Subdomain,
    neighbor_bounds: &AABB,
    overlap: f32,
) -> ParticleArrays {
    let mut ghosts = ParticleArrays::new();

    // The ghost region is the zone of `source` that lies within `overlap`
    // distance of `neighbor_bounds`. A particle from `source` is a ghost
    // candidate if it is within `overlap` of the neighbor's boundary.
    for i in 0..source.particles.len() {
        let px = source.particles.x[i];
        let py = source.particles.y[i];
        let pz = source.particles.z[i];

        // Check if this particle is near the neighbor boundary (within overlap)
        if point_within_expanded_aabb(px, py, pz, neighbor_bounds, overlap) {
            ghosts.push_particle(
                px, py, pz,
                source.particles.mass[i],
                source.particles.density[i],
                source.particles.temperature[i],
                source.particles.fluid_type[i],
            );
            let idx = ghosts.len() - 1;
            ghosts.vx[idx] = source.particles.vx[i];
            ghosts.vy[idx] = source.particles.vy[i];
            ghosts.vz[idx] = source.particles.vz[i];
            ghosts.ax[idx] = source.particles.ax[i];
            ghosts.ay[idx] = source.particles.ay[i];
            ghosts.az[idx] = source.particles.az[i];
            ghosts.pressure[idx] = source.particles.pressure[i];
        }
    }

    ghosts
}

/// Check if a point is within an AABB expanded by `margin` on each side.
fn point_within_expanded_aabb(x: f32, y: f32, z: f32, aabb: &AABB, margin: f32) -> bool {
    x >= aabb.min[0] - margin && x <= aabb.max[0] + margin
        && y >= aabb.min[1] - margin && y <= aabb.max[1] + margin
        && z >= aabb.min[2] - margin && z <= aabb.max[2] + margin
}

/// Perform a full ghost particle exchange between all neighboring subdomains.
///
/// For each pair of neighbors (i, j), particles near the shared boundary are
/// extracted and sent as ghost particles. Each subdomain receives ghost
/// particles from all its neighbors.
///
/// # Arguments
/// * `subdomains` - All subdomains (mutable access not needed; we read only)
/// * `smoothing_length` - SPH smoothing length
///
/// # Returns
/// A vector of vectors: `result[i]` contains all `GhostParticles` for subdomain `i`.
pub fn exchange_ghost_particles(
    subdomains: &[Subdomain],
    smoothing_length: f32,
) -> Vec<Vec<GhostParticles>> {
    let overlap = 2.0 * smoothing_length;
    let n = subdomains.len();
    let mut all_ghosts: Vec<Vec<GhostParticles>> = (0..n).map(|_| Vec::new()).collect();

    for i in 0..n {
        for &j in &subdomains[i].neighbor_ids {
            // Extract particles from subdomain j that are ghosts for subdomain i
            let ghost_particles = extract_ghost_particles(
                &subdomains[j],
                &subdomains[i].bounds,
                overlap,
            );

            if !ghost_particles.is_empty() {
                all_ghosts[i].push(GhostParticles {
                    source_id: j,
                    particles: ghost_particles,
                });
            }
        }
    }

    all_ghosts
}

/// Merge ghost particles into a subdomain's particle arrays for the purpose
/// of SPH computation. Returns the combined particle array and the count of
/// original (owned) particles so they can be separated after computation.
///
/// # Arguments
/// * `owned` - The subdomain's own particles
/// * `ghosts` - Ghost particles from all neighbors
///
/// # Returns
/// `(merged_particles, owned_count)` - The merged array and the index boundary
/// between owned and ghost particles.
pub fn merge_with_ghosts(
    owned: &ParticleArrays,
    ghosts: &[GhostParticles],
) -> (ParticleArrays, usize) {
    let owned_count = owned.len();
    let mut merged = owned.clone();

    for gp in ghosts {
        for i in 0..gp.particles.len() {
            merged.push_particle(
                gp.particles.x[i],
                gp.particles.y[i],
                gp.particles.z[i],
                gp.particles.mass[i],
                gp.particles.density[i],
                gp.particles.temperature[i],
                gp.particles.fluid_type[i],
            );
            let idx = merged.len() - 1;
            merged.vx[idx] = gp.particles.vx[i];
            merged.vy[idx] = gp.particles.vy[i];
            merged.vz[idx] = gp.particles.vz[i];
            merged.ax[idx] = gp.particles.ax[i];
            merged.ay[idx] = gp.particles.ay[i];
            merged.az[idx] = gp.particles.az[i];
            merged.pressure[idx] = gp.particles.pressure[i];
        }
    }

    (merged, owned_count)
}

/// Extract only the owned particles from a merged array (strip ghost particles).
///
/// # Arguments
/// * `merged` - The merged particle array (owned + ghosts)
/// * `owned_count` - Number of owned particles (from `merge_with_ghosts`)
///
/// # Returns
/// A `ParticleArrays` containing only the owned particles with updated state.
pub fn strip_ghosts(merged: &ParticleArrays, owned_count: usize) -> ParticleArrays {
    let mut owned = ParticleArrays::new();
    for i in 0..owned_count {
        owned.push_particle(
            merged.x[i],
            merged.y[i],
            merged.z[i],
            merged.mass[i],
            merged.density[i],
            merged.temperature[i],
            merged.fluid_type[i],
        );
        let idx = owned.len() - 1;
        owned.vx[idx] = merged.vx[i];
        owned.vy[idx] = merged.vy[i];
        owned.vz[idx] = merged.vz[i];
        owned.ax[idx] = merged.ax[i];
        owned.ay[idx] = merged.ay[i];
        owned.az[idx] = merged.az[i];
        owned.pressure[idx] = merged.pressure[i];
    }
    owned
}

// ===========================================================================
// T069: Result Aggregation
// ===========================================================================

/// Merge particle arrays from multiple subdomains into a single combined array.
///
/// This collects all owned particles from each subdomain into one unified
/// `ParticleArrays`. Ghost particles are excluded (each particle appears once).
///
/// # Arguments
/// * `subdomains` - Slice of subdomains to merge
///
/// # Returns
/// A single `ParticleArrays` containing all particles from all subdomains.
pub fn merge_subdomain_particles(subdomains: &[Subdomain]) -> ParticleArrays {
    let total_count: usize = subdomains.iter().map(|s| s.particles.len()).sum();
    let mut merged = ParticleArrays::new();
    merged.x.reserve(total_count);
    merged.y.reserve(total_count);
    merged.z.reserve(total_count);
    merged.vx.reserve(total_count);
    merged.vy.reserve(total_count);
    merged.vz.reserve(total_count);
    merged.ax.reserve(total_count);
    merged.ay.reserve(total_count);
    merged.az.reserve(total_count);
    merged.mass.reserve(total_count);
    merged.density.reserve(total_count);
    merged.pressure.reserve(total_count);
    merged.temperature.reserve(total_count);
    merged.fluid_type.reserve(total_count);

    for sub in subdomains {
        for i in 0..sub.particles.len() {
            merged.push_particle(
                sub.particles.x[i],
                sub.particles.y[i],
                sub.particles.z[i],
                sub.particles.mass[i],
                sub.particles.density[i],
                sub.particles.temperature[i],
                sub.particles.fluid_type[i],
            );
            let idx = merged.len() - 1;
            merged.vx[idx] = sub.particles.vx[i];
            merged.vy[idx] = sub.particles.vy[i];
            merged.vz[idx] = sub.particles.vz[i];
            merged.ax[idx] = sub.particles.ax[i];
            merged.ay[idx] = sub.particles.ay[i];
            merged.az[idx] = sub.particles.az[i];
            merged.pressure[idx] = sub.particles.pressure[i];
        }
    }

    merged
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
            backend: crate::config::BackendType::default(),
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
            backend: crate::config::BackendType::default(),
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

    // -----------------------------------------------------------------------
    // T066: Domain decomposition tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_aabb_longest_axis() {
        // X is longest
        let aabb = AABB::new([0.0, 0.0, 0.0], [10.0, 5.0, 3.0]);
        assert_eq!(aabb.longest_axis(), 0);

        // Y is longest
        let aabb = AABB::new([0.0, 0.0, 0.0], [3.0, 10.0, 5.0]);
        assert_eq!(aabb.longest_axis(), 1);

        // Z is longest
        let aabb = AABB::new([0.0, 0.0, 0.0], [3.0, 5.0, 10.0]);
        assert_eq!(aabb.longest_axis(), 2);
    }

    #[test]
    fn test_aabb_contains() {
        let aabb = AABB::new([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]);
        assert!(aabb.contains(0.5, 0.5, 0.5));
        assert!(aabb.contains(0.0, 0.0, 0.0)); // edge
        assert!(aabb.contains(1.0, 1.0, 1.0)); // edge
        assert!(!aabb.contains(1.1, 0.5, 0.5));
        assert!(!aabb.contains(-0.1, 0.5, 0.5));
    }

    #[test]
    fn test_decompose_domain_single() {
        let bounds = AABB::new([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]);
        let subs = decompose_domain(&bounds, 1);
        assert_eq!(subs.len(), 1);
        assert!((subs[0].min[0] - 0.0).abs() < 1e-6);
        assert!((subs[0].max[0] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_decompose_domain_two() {
        // Longest axis is x (2.0 > 1.0)
        let bounds = AABB::new([0.0, 0.0, 0.0], [2.0, 1.0, 1.0]);
        let subs = decompose_domain(&bounds, 2);
        assert_eq!(subs.len(), 2);

        // Should split along x at 1.0
        assert!((subs[0].min[0] - 0.0).abs() < 1e-6);
        assert!((subs[0].max[0] - 1.0).abs() < 1e-6);
        assert!((subs[1].min[0] - 1.0).abs() < 1e-6);
        assert!((subs[1].max[0] - 2.0).abs() < 1e-6);

        // Y and Z should be unchanged
        assert!((subs[0].min[1] - 0.0).abs() < 1e-6);
        assert!((subs[0].max[1] - 1.0).abs() < 1e-6);
        assert!((subs[1].min[1] - 0.0).abs() < 1e-6);
        assert!((subs[1].max[1] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_decompose_domain_four() {
        let bounds = AABB::new([0.0, 0.0, 0.0], [4.0, 2.0, 1.0]);
        let subs = decompose_domain(&bounds, 4);
        assert_eq!(subs.len(), 4);

        // Should have 4 non-overlapping regions covering the full domain
        // Total volume should equal original
        let orig_vol: f32 = 4.0 * 2.0 * 1.0;
        let total_vol: f32 = subs.iter().map(|s| {
            let ext = s.extent();
            ext[0] * ext[1] * ext[2]
        }).sum();
        assert!((total_vol - orig_vol).abs() < 1e-4,
            "Total volume {} should equal original volume {}", total_vol, orig_vol);
    }

    #[test]
    fn test_decompose_domain_three() {
        // Non-power-of-two
        let bounds = AABB::new([0.0, 0.0, 0.0], [3.0, 1.0, 1.0]);
        let subs = decompose_domain(&bounds, 3);
        assert_eq!(subs.len(), 3);

        // Total volume should equal original
        let orig_vol: f32 = 3.0 * 1.0 * 1.0;
        let total_vol: f32 = subs.iter().map(|s| {
            let ext = s.extent();
            ext[0] * ext[1] * ext[2]
        }).sum();
        assert!((total_vol - orig_vol).abs() < 1e-4);
    }

    #[test]
    fn test_build_subdomains_particle_distribution() {
        // Create particles spread across a domain
        let mut particles = ParticleArrays::new();
        // Left half particles
        particles.push_particle(0.25, 0.5, 0.5, 0.001, 1000.0, 293.15, FluidType::Water);
        particles.push_particle(0.4, 0.5, 0.5, 0.001, 1000.0, 293.15, FluidType::Water);
        // Right half particles
        particles.push_particle(0.75, 0.5, 0.5, 0.001, 1000.0, 293.15, FluidType::Water);
        particles.push_particle(0.9, 0.5, 0.5, 0.001, 1000.0, 293.15, FluidType::Water);

        let bounds = AABB::new([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]);
        let aabbs = decompose_domain(&bounds, 2);
        let h = 0.05;
        let subdomains = build_subdomains(&aabbs, &particles, &[], h);

        assert_eq!(subdomains.len(), 2);
        assert_eq!(subdomains[0].particles.len(), 2, "Left subdomain should have 2 particles");
        assert_eq!(subdomains[1].particles.len(), 2, "Right subdomain should have 2 particles");

        // They should be neighbors
        assert!(subdomains[0].neighbor_ids.contains(&1));
        assert!(subdomains[1].neighbor_ids.contains(&0));
    }

    // -----------------------------------------------------------------------
    // T067: Ghost particle exchange tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_ghost_particle_exchange() {
        let mut particles = ParticleArrays::new();
        // Particle near the boundary between subdomains (at x=0.48, close to split at 0.5)
        particles.push_particle(0.48, 0.5, 0.5, 0.001, 1000.0, 293.15, FluidType::Water);
        // Particle far from boundary
        particles.push_particle(0.1, 0.5, 0.5, 0.001, 1000.0, 293.15, FluidType::Water);
        // Particle near boundary in right subdomain
        particles.push_particle(0.52, 0.5, 0.5, 0.001, 1000.0, 293.15, FluidType::Water);
        // Particle far from boundary in right subdomain
        particles.push_particle(0.9, 0.5, 0.5, 0.001, 1000.0, 293.15, FluidType::Water);

        let bounds = AABB::new([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]);
        let aabbs = decompose_domain(&bounds, 2);
        let h = 0.05; // support radius = 0.1
        let subdomains = build_subdomains(&aabbs, &particles, &[], h);

        let ghosts = exchange_ghost_particles(&subdomains, h);

        // Subdomain 0 (x: 0..0.5) should get ghost from subdomain 1 (the particle at x=0.52)
        assert!(!ghosts[0].is_empty(), "Subdomain 0 should receive ghost particles");
        let ghost_count_0: usize = ghosts[0].iter().map(|g| g.particles.len()).sum();
        assert!(ghost_count_0 >= 1, "Subdomain 0 should have at least 1 ghost particle from sub 1");

        // Subdomain 1 (x: 0.5..1.0) should get ghost from subdomain 0 (the particle at x=0.48)
        assert!(!ghosts[1].is_empty(), "Subdomain 1 should receive ghost particles");
        let ghost_count_1: usize = ghosts[1].iter().map(|g| g.particles.len()).sum();
        assert!(ghost_count_1 >= 1, "Subdomain 1 should have at least 1 ghost particle from sub 0");
    }

    #[test]
    fn test_merge_and_strip_ghosts() {
        let mut owned = ParticleArrays::new();
        owned.push_particle(0.1, 0.5, 0.5, 0.001, 1000.0, 293.15, FluidType::Water);
        owned.push_particle(0.2, 0.5, 0.5, 0.001, 1000.0, 293.15, FluidType::Water);

        let mut ghost_p = ParticleArrays::new();
        ghost_p.push_particle(0.6, 0.5, 0.5, 0.001, 1000.0, 293.15, FluidType::Water);

        let ghosts = vec![GhostParticles {
            source_id: 1,
            particles: ghost_p,
        }];

        let (merged, owned_count) = merge_with_ghosts(&owned, &ghosts);
        assert_eq!(merged.len(), 3);
        assert_eq!(owned_count, 2);

        let stripped = strip_ghosts(&merged, owned_count);
        assert_eq!(stripped.len(), 2);
        assert!((stripped.x[0] - 0.1).abs() < 1e-6);
        assert!((stripped.x[1] - 0.2).abs() < 1e-6);
    }

    // -----------------------------------------------------------------------
    // T069: Result aggregation tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_merge_subdomain_particles() {
        let mut p0 = ParticleArrays::new();
        p0.push_particle(0.1, 0.5, 0.5, 0.001, 1000.0, 293.15, FluidType::Water);
        p0.push_particle(0.2, 0.5, 0.5, 0.001, 1000.0, 293.15, FluidType::Water);

        let mut p1 = ParticleArrays::new();
        p1.push_particle(0.7, 0.5, 0.5, 0.001, 1000.0, 293.15, FluidType::Water);

        let subdomains = vec![
            Subdomain {
                id: 0,
                bounds: AABB::new([0.0, 0.0, 0.0], [0.5, 1.0, 1.0]),
                particles: p0,
                boundary_particles: vec![],
                neighbor_ids: vec![1],
            },
            Subdomain {
                id: 1,
                bounds: AABB::new([0.5, 0.0, 0.0], [1.0, 1.0, 1.0]),
                particles: p1,
                boundary_particles: vec![],
                neighbor_ids: vec![0],
            },
        ];

        let merged = merge_subdomain_particles(&subdomains);
        assert_eq!(merged.len(), 3);
        // Particles should be ordered: sub0 first, then sub1
        assert!((merged.x[0] - 0.1).abs() < 1e-6);
        assert!((merged.x[1] - 0.2).abs() < 1e-6);
        assert!((merged.x[2] - 0.7).abs() < 1e-6);
    }
}
