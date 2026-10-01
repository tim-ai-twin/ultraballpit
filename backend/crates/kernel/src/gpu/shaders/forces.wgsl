// Pressure + viscous force computation shader
//
// Computes:
// 1. Pressure gradient forces (symmetric SPH formulation)
// 2. Monaghan artificial viscosity
// 3. Gravity
// 4. Boundary repulsive forces
// 5. Boundary pressure mirroring (Adami et al. 2012)
//
// Two entry points:
// - update_boundary_pressures: Adami pressure mirroring for boundary particles
// - compute_forces: all forces on fluid particles

const PI: f32 = 3.14159265358979323846;
const WENDLAND_C2_NORM_3D: f32 = 0.41780189; // 21 / (16 * PI)

const WATER_REST_DENSITY: f32 = 1000.0;
const AIR_REST_DENSITY: f32 = 1.204;

struct SimParams {
    dt: f32,
    h: f32,
    speed_of_sound: f32,
    gravity_x: f32,
    gravity_y: f32,
    gravity_z: f32,
    domain_min_x: f32,
    domain_min_y: f32,
    domain_min_z: f32,
    domain_max_x: f32,
    domain_max_y: f32,
    domain_max_z: f32,
    n_particles: u32,
    n_boundary: u32,
    grid_dim_x: u32,
    grid_dim_y: u32,
    grid_dim_z: u32,
    cell_size: f32,
    viscosity_alpha: f32,
    viscosity_beta: f32,
    pass_index: u32,
    // Cells searched on each side of a particle's cell (ceil(2h / cell_size)).
    search_cells: u32,
};

// Group 0: SimParams + positions + mass
@group(0) @binding(0) var<uniform> params: SimParams;
@group(0) @binding(1) var<storage, read> pos_x: array<f32>;
@group(0) @binding(2) var<storage, read> pos_y: array<f32>;
@group(0) @binding(3) var<storage, read> pos_z: array<f32>;
@group(0) @binding(4) var<storage, read> mass: array<f32>;

// Group 1: Velocity + acceleration
@group(1) @binding(0) var<storage, read> vel_x: array<f32>;
@group(1) @binding(1) var<storage, read> vel_y: array<f32>;
@group(1) @binding(2) var<storage, read> vel_z: array<f32>;
@group(1) @binding(3) var<storage, read_write> acc_x: array<f32>;
@group(1) @binding(4) var<storage, read_write> acc_y: array<f32>;
@group(1) @binding(5) var<storage, read_write> acc_z: array<f32>;
// Neighbor lists written by the density pass (see density.wgsl).
@group(1) @binding(6) var<storage, read> nbr_list: array<u32>;
@group(1) @binding(7) var<storage, read> nbr_count: array<u32>;
@group(1) @binding(8) var<storage, read> bnd_nbr_list: array<u32>;
@group(1) @binding(9) var<storage, read> bnd_nbr_count: array<u32>;
// Packed caches written by the density pass: (x, y, z, mass), (v, density).
@group(1) @binding(10) var<storage, read> posm: array<vec4<f32>>;
@group(1) @binding(11) var<storage, read> velr: array<vec4<f32>>;

const MAX_NBR: u32 = 128u;

// Decode neighbor k of particle i from the packed 16-bit offset list.
fn nbr_at(i: u32, k: u32) -> u32 {
    let word = nbr_list[(k >> 1u) * params.n_particles + i];
    let off = (word >> ((k & 1u) * 16u)) & 0xffffu;
    return u32(i32(i) + i32(off) - 32768);
}
const MAX_BND_NBR: u32 = 64u;

// Group 2: SPH state + boundary
@group(2) @binding(0) var<storage, read> density: array<f32>;
@group(2) @binding(1) var<storage, read> pressure: array<f32>;
@group(2) @binding(2) var<storage, read> fluid_type: array<u32>;
@group(2) @binding(3) var<storage, read> bnd_x: array<f32>;
@group(2) @binding(4) var<storage, read> bnd_y: array<f32>;
@group(2) @binding(5) var<storage, read> bnd_z: array<f32>;
@group(2) @binding(6) var<storage, read> bnd_mass: array<f32>;
@group(2) @binding(7) var<storage, read_write> bnd_pressure: array<f32>;
@group(2) @binding(8) var<storage, read> bnd_cell_counts: array<u32>;
@group(2) @binding(9) var<storage, read> bnd_cell_offsets: array<u32>;

// Group 3: Grid data (read-only for forces)
@group(3) @binding(2) var<storage, read> cell_offsets: array<u32>;
@group(3) @binding(1) var<storage, read> cell_counts: array<u32>;

fn wendland_c2(r: f32, h: f32) -> f32 {
    let q = r / h;
    if q >= 2.0 {
        return 0.0;
    }
    let h3 = h * h * h;
    let one_minus_half_q = 1.0 - 0.5 * q;
    let t = one_minus_half_q * one_minus_half_q;
    let t4 = t * t;
    return WENDLAND_C2_NORM_3D / h3 * t4 * (1.0 + 2.0 * q);
}

fn read_mass(idx: u32) -> f32 {
    return mass[idx];
}

fn wendland_c2_gradient_from_dist_sq(dx: f32, dy: f32, dz: f32, dist_sq: f32, h: f32) -> vec3<f32> {
    let inv_r = inverseSqrt(dist_sq);
    let r = dist_sq * inv_r;
    let q = r / h;
    if q >= 2.0 || dist_sq < 1.0e-24 {
        return vec3<f32>(0.0, 0.0, 0.0);
    }

    let h3 = h * h * h;
    let one_minus_half_q = 1.0 - 0.5 * q;
    let t3 = one_minus_half_q * one_minus_half_q * one_minus_half_q;

    let dw_dr = WENDLAND_C2_NORM_3D / (h3 * h) * (-5.0 * q) * t3;

    return vec3<f32>(dw_dr * dx * inv_r, dw_dr * dy * inv_r, dw_dr * dz * inv_r);
}

fn pos_to_cell_i32(px: f32, py: f32, pz: f32) -> vec3<i32> {
    let cx = i32(floor((px - params.domain_min_x) / params.cell_size));
    let cy = i32(floor((py - params.domain_min_y) / params.cell_size));
    let cz = i32(floor((pz - params.domain_min_z) / params.cell_size));
    return vec3<i32>(
        clamp(cx, 0, i32(params.grid_dim_x) - 1),
        clamp(cy, 0, i32(params.grid_dim_y) - 1),
        clamp(cz, 0, i32(params.grid_dim_z) - 1)
    );
}

fn cell_hash(cx: u32, cy: u32, cz: u32) -> u32 {
    return cx + cy * params.grid_dim_x + cz * params.grid_dim_x * params.grid_dim_y;
}

// Contiguous particle index range covering cells [cell.x - search, cell.x + search]
// of grid row (ny, nz). Valid because particles are stored in cell order.
fn row_range(cell: vec3<i32>, ny: i32, nz: i32) -> vec2<u32> {
    let search = i32(params.search_cells);
    let x0 = max(cell.x - search, 0);
    let x1 = min(cell.x + search, i32(params.grid_dim_x) - 1);
    let c0 = cell_hash(u32(x0), u32(ny), u32(nz));
    let c1 = cell_hash(u32(x1), u32(ny), u32(nz));
    return vec2<u32>(cell_offsets[c0], cell_offsets[c1] + cell_counts[c1]);
}

// Boundary-particle analogue of row_range (boundary arrays are stored in cell order).
fn bnd_row_range(cell: vec3<i32>, ny: i32, nz: i32) -> vec2<u32> {
    let search = i32(params.search_cells);
    let x0 = max(cell.x - search, 0);
    let x1 = min(cell.x + search, i32(params.grid_dim_x) - 1);
    let c0 = cell_hash(u32(x0), u32(ny), u32(nz));
    let c1 = cell_hash(u32(x1), u32(ny), u32(nz));
    return vec2<u32>(bnd_cell_offsets[c0], bnd_cell_offsets[c1] + bnd_cell_counts[c1]);
}

// Entry point: Update boundary pressures using Adami et al. (2012) mirroring
@compute @workgroup_size(256)
fn update_boundary_pressures(@builtin(global_invocation_id) gid: vec3<u32>) {
    let b = gid.x;
    if b >= params.n_boundary {
        return;
    }

    let h = params.h;
    let search = i32(params.search_cells);
    let support_radius = 2.0 * h;
    let support_radius_sq = support_radius * support_radius;

    let bx = bnd_x[b];
    let by = bnd_y[b];
    let bz = bnd_z[b];

    var weighted_pressure = 0.0;
    var weight_sum = 0.0;

    // Search fluid particles near this boundary particle via neighbor grid
    let bcell = pos_to_cell_i32(bx, by, bz);
    for (var dz_off = -search; dz_off <= search; dz_off = dz_off + 1) {
        let nz = bcell.z + dz_off;
        if nz < 0 || nz >= i32(params.grid_dim_z) { continue; }
        for (var dy_off = -search; dy_off <= search; dy_off = dy_off + 1) {
            let ny = bcell.y + dy_off;
            if ny < 0 || ny >= i32(params.grid_dim_y) { continue; }
            {
                // The row's cells are adjacent in hash order and particles are
                // stored in cell order, so the whole row is one index range.
                let f_range = row_range(bcell, ny, nz);
                for (var f = f_range.x; f < f_range.y; f = f + 1u) {
                    let dx = bx - pos_x[f];
                    let dy = by - pos_y[f];
                    let dz = bz - pos_z[f];
                    let dist_sq = dx * dx + dy * dy + dz * dz;

                    if dist_sq < support_radius_sq {
                        let inv_r = inverseSqrt(max(dist_sq, 1.0e-24));
                        let r = dist_sq * inv_r;
                        let w = wendland_c2(r, h);
                        let g_dot_dr = params.gravity_x * dx + params.gravity_y * dy + params.gravity_z * dz;
                        let p_extrapolated = pressure[f] + density[f] * g_dot_dr;
                        weighted_pressure = weighted_pressure + w * p_extrapolated;
                        weight_sum = weight_sum + w;
                    }
                }
            }
        }
    }

    if weight_sum > 1.0e-12 {
        bnd_pressure[b] = max(weighted_pressure / weight_sum, 0.0);
    } else {
        bnd_pressure[b] = 0.0;
    }
}

// Per-particle state of the particle whose forces are being summed.
struct ParticleI {
    pos: vec3<f32>,
    vel: vec3<f32>,
    m: f32,
    rho: f32,
    p_over_rho2: f32,
    // Fluid-side pressure clamped to >= 0 for boundary interactions.
    p_clamped_over_rho2: f32,
    // Rest density used as the boundary particle density.
    boundary_rho: f32,
};

// Pressure + Monaghan viscosity force (times m_i) from fluid neighbor j.
fn fluid_pair_force(pi: ParticleI, j: u32) -> vec3<f32> {
    let h = params.h;
    let support_radius_sq = 4.0 * h * h;
    let eta_sq = 0.01 * h * h;
    let pm = posm[j];
    let d = pi.pos - pm.xyz;
    let dist_sq = dot(d, d);
    if dist_sq > support_radius_sq {
        return vec3<f32>(0.0);
    }
    let vr = velr[j];
    let grad = wendland_c2_gradient_from_dist_sq(d.x, d.y, d.z, dist_sq, h);
    let m_j = pm.w;
    let rho_j = vr.w;

    // Pressure forces
    let pj_over_rho2_j = pressure[j] / (rho_j * rho_j);
    var factor = -pi.m * m_j * (pi.p_over_rho2 + pj_over_rho2_j);

    // Viscous forces (Monaghan artificial viscosity)
    let dv = pi.vel - vr.xyz;
    let vr_dot = dot(dv, d);
    if vr_dot < 0.0 {
        let mu_ij = h * vr_dot / (dist_sq + eta_sq);
        let rho_avg = 0.5 * (pi.rho + rho_j);
        let pi_ij = (-params.viscosity_alpha * params.speed_of_sound * mu_ij
            + params.viscosity_beta * mu_ij * mu_ij) / rho_avg;
        // Multiply by m_i so that the final `f / m_i` conversion
        // to acceleration correctly cancels to -m_j * pi_ij * grad_W.
        factor = factor - pi.m * m_j * pi_ij;
    }
    return factor * grad;
}

// Pressure + short-range repulsion force (times m_i) from boundary particle b.
fn boundary_pair_force(pi: ParticleI, b: u32) -> vec3<f32> {
    let h = params.h;
    let support_radius_sq = 4.0 * h * h;
    let r0 = 0.5 * h;
    let d_repulsive = 10.0 * 9.81 * 0.01;
    let d = pi.pos - vec3<f32>(bnd_x[b], bnd_y[b], bnd_z[b]);
    let dist_sq = dot(d, d);
    if dist_sq >= support_radius_sq {
        return vec3<f32>(0.0);
    }
    let grad = wendland_c2_gradient_from_dist_sq(d.x, d.y, d.z, dist_sq, h);
    let pb_over_rho2_b = bnd_pressure[b] / (pi.boundary_rho * pi.boundary_rho);
    var f = (-pi.m * bnd_mass[b] * (pi.p_clamped_over_rho2 + pb_over_rho2_b)) * grad;

    let inv_r_bnd = inverseSqrt(max(dist_sq, 1.0e-24));
    let r_bnd = dist_sq * inv_r_bnd;
    if r_bnd < r0 && dist_sq > 1.0e-24 {
        let s = 1.0 - r_bnd / r0;
        let force_mag = d_repulsive * s * s / r0;
        f = f + (force_mag * inv_r_bnd * pi.m) * d;
    }
    return f;
}

// Entry point: Compute all forces on fluid particles
@compute @workgroup_size(256)
fn compute_forces(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i >= params.n_particles {
        return;
    }

    let n = params.n_particles;
    let search = i32(params.search_cells);
    let rho_i = density[i];
    let p_i = pressure[i];

    var pi: ParticleI;
    pi.pos = vec3<f32>(pos_x[i], pos_y[i], pos_z[i]);
    pi.vel = vec3<f32>(vel_x[i], vel_y[i], vel_z[i]);
    pi.m = read_mass(i);
    pi.rho = rho_i;
    pi.p_over_rho2 = p_i / (rho_i * rho_i);
    pi.p_clamped_over_rho2 = max(p_i, 0.0) / (rho_i * rho_i);
    pi.boundary_rho = select(AIR_REST_DENSITY, WATER_REST_DENSITY, fluid_type[i] == 0u);

    var f = vec3<f32>(0.0);
    let cell = pos_to_cell_i32(pi.pos.x, pi.pos.y, pi.pos.z);

    // Fluid-fluid interactions: walk the density pass's neighbor list (same
    // positions, so it is exact); grid scan only if the list overflowed.
    let n_nbr = nbr_count[i];
    if n_nbr <= MAX_NBR {
        for (var k = 0u; k < n_nbr; k = k + 1u) {
            f = f + fluid_pair_force(pi, nbr_at(i, k));
        }
    } else {
        for (var dz = -search; dz <= search; dz = dz + 1) {
            let nz = cell.z + dz;
            if nz < 0 || nz >= i32(params.grid_dim_z) { continue; }
            for (var dy = -search; dy <= search; dy = dy + 1) {
                let ny = cell.y + dy;
                if ny < 0 || ny >= i32(params.grid_dim_y) { continue; }
                let j_range = row_range(cell, ny, nz);
                for (var j = j_range.x; j < j_range.y; j = j + 1u) {
                    if j == i { continue; }
                    f = f + fluid_pair_force(pi, j);
                }
            }
        }
    }

    // Boundary particle contributions: pressure forces + repulsive forces
    if params.n_boundary > 0u {
        let n_bnd = bnd_nbr_count[i];
        if n_bnd <= MAX_BND_NBR {
            for (var k = 0u; k < n_bnd; k = k + 1u) {
                f = f + boundary_pair_force(pi, bnd_nbr_list[k * n + i]);
            }
        } else {
            for (var dz = -search; dz <= search; dz = dz + 1) {
                let nz = cell.z + dz;
                if nz < 0 || nz >= i32(params.grid_dim_z) { continue; }
                for (var dy = -search; dy <= search; dy = dy + 1) {
                    let ny = cell.y + dy;
                    if ny < 0 || ny >= i32(params.grid_dim_y) { continue; }
                    let b_range = bnd_row_range(cell, ny, nz);
                    for (var b = b_range.x; b < b_range.y; b = b + 1u) {
                        f = f + boundary_pair_force(pi, b);
                    }
                }
            }
        }
    }

    // Convert forces to accelerations: a = F / m, plus gravity
    acc_x[i] = f.x / pi.m + params.gravity_x;
    acc_y[i] = f.y / pi.m + params.gravity_y;
    acc_z[i] = f.z / pi.m + params.gravity_z;
}
