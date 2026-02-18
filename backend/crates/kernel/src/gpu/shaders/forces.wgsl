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
    _pad1: u32,
};

// Group 0: SimParams + positions + mass
@group(0) @binding(0) var<uniform> params: SimParams;
@group(0) @binding(1) var<storage, read> pos_x: array<f32>;
@group(0) @binding(2) var<storage, read> pos_y: array<f32>;
@group(0) @binding(3) var<storage, read> pos_z: array<f32>;
// Mass stored as f16 pairs packed in u32.
@group(0) @binding(4) var<storage, read> mass_packed: array<u32>;

// Group 1: Velocity + acceleration
@group(1) @binding(0) var<storage, read> vel_x: array<f32>;
@group(1) @binding(1) var<storage, read> vel_y: array<f32>;
@group(1) @binding(2) var<storage, read> vel_z: array<f32>;
@group(1) @binding(3) var<storage, read_write> acc_x: array<f32>;
@group(1) @binding(4) var<storage, read_write> acc_y: array<f32>;
@group(1) @binding(5) var<storage, read_write> acc_z: array<f32>;

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
@group(2) @binding(10) var<storage, read> bnd_sorted_indices: array<u32>;

// Group 3: Grid data (read-only for forces)
@group(3) @binding(2) var<storage, read> cell_offsets: array<u32>;
@group(3) @binding(1) var<storage, read> cell_counts: array<u32>;
@group(3) @binding(3) var<storage, read> sorted_indices: array<u32>;

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
    let pair = unpack2x16float(mass_packed[idx >> 1u]);
    return pair[idx & 1u];
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

// Entry point: Update boundary pressures using Adami et al. (2012) mirroring
@compute @workgroup_size(256)
fn update_boundary_pressures(@builtin(global_invocation_id) gid: vec3<u32>) {
    let b = gid.x;
    if b >= params.n_boundary {
        return;
    }

    let h = params.h;
    let support_radius = 2.0 * h;
    let support_radius_sq = support_radius * support_radius;

    let bx = bnd_x[b];
    let by = bnd_y[b];
    let bz = bnd_z[b];

    var weighted_pressure = 0.0;
    var weight_sum = 0.0;

    // Search fluid particles near this boundary particle via neighbor grid
    let bcell = pos_to_cell_i32(bx, by, bz);
    for (var dz_off = -1; dz_off <= 1; dz_off = dz_off + 1) {
        let nz = bcell.z + dz_off;
        if nz < 0 || nz >= i32(params.grid_dim_z) { continue; }
        for (var dy_off = -1; dy_off <= 1; dy_off = dy_off + 1) {
            let ny = bcell.y + dy_off;
            if ny < 0 || ny >= i32(params.grid_dim_y) { continue; }
            for (var dx_off = -1; dx_off <= 1; dx_off = dx_off + 1) {
                let nx = bcell.x + dx_off;
                if nx < 0 || nx >= i32(params.grid_dim_x) { continue; }

                let c = cell_hash(u32(nx), u32(ny), u32(nz));
                let start = cell_offsets[c];
                let count = cell_counts[c];

                for (var s = start; s < start + count; s = s + 1u) {
                    let f = sorted_indices[s];
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

// Entry point: Compute all forces on fluid particles
@compute @workgroup_size(256)
fn compute_forces(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i >= params.n_particles {
        return;
    }

    let h = params.h;
    let support_radius = 2.0 * h;
    let support_radius_sq = support_radius * support_radius;
    let alpha = params.viscosity_alpha;
    let beta = params.viscosity_beta;
    let eta_sq = 0.01 * h * h;

    let px = pos_x[i];
    let py = pos_y[i];
    let pz = pos_z[i];
    let m_i = read_mass(i);
    let rho_i = density[i];
    let p_i = pressure[i];
    let ft_i = fluid_type[i];

    let pi_over_rho2_i = p_i / (rho_i * rho_i);

    // For boundary interactions, clamp fluid-side pressure to >= 0
    let pi_clamped = max(p_i, 0.0);
    let pi_clamped_over_rho2 = pi_clamped / (rho_i * rho_i);

    // Rest density for boundary denominator
    var boundary_rho = WATER_REST_DENSITY;
    if ft_i != 0u {
        boundary_rho = AIR_REST_DENSITY;
    }

    var fx = 0.0;
    var fy = 0.0;
    var fz = 0.0;

    // Fluid-fluid interactions via neighbor grid
    let cell = pos_to_cell_i32(px, py, pz);

    for (var dz_off = -1; dz_off <= 1; dz_off = dz_off + 1) {
        let nz = cell.z + dz_off;
        if nz < 0 || nz >= i32(params.grid_dim_z) { continue; }
        for (var dy_off = -1; dy_off <= 1; dy_off = dy_off + 1) {
            let ny = cell.y + dy_off;
            if ny < 0 || ny >= i32(params.grid_dim_y) { continue; }
            for (var dx_off = -1; dx_off <= 1; dx_off = dx_off + 1) {
                let nx = cell.x + dx_off;
                if nx < 0 || nx >= i32(params.grid_dim_x) { continue; }

                let c = cell_hash(u32(nx), u32(ny), u32(nz));
                let start = cell_offsets[c];
                let count = cell_counts[c];

                for (var s = start; s < start + count; s = s + 1u) {
                    let j = sorted_indices[s];
                    if j == i { continue; }

                    let ddx = px - pos_x[j];
                    let ddy = py - pos_y[j];
                    let ddz = pz - pos_z[j];
                    let dist_sq = ddx * ddx + ddy * ddy + ddz * ddz;

                    if dist_sq > support_radius_sq { continue; }

                    let grad = wendland_c2_gradient_from_dist_sq(ddx, ddy, ddz, dist_sq, h);

                    // Pressure forces
                    let pj_over_rho2_j = pressure[j] / (density[j] * density[j]);
                    let p_factor = -m_i * read_mass(j) * (pi_over_rho2_i + pj_over_rho2_j);
                    fx = fx + p_factor * grad.x;
                    fy = fy + p_factor * grad.y;
                    fz = fz + p_factor * grad.z;

                    // Viscous forces (Monaghan artificial viscosity)
                    let dvx = vel_x[i] - vel_x[j];
                    let dvy = vel_y[i] - vel_y[j];
                    let dvz = vel_z[i] - vel_z[j];
                    let vr_dot = dvx * ddx + dvy * ddy + dvz * ddz;

                    if vr_dot < 0.0 {
                        let r_sq = dist_sq;
                        let mu_ij = h * vr_dot / (r_sq + eta_sq);
                        let rho_avg = 0.5 * (rho_i + density[j]);
                        let pi_ij = (-alpha * params.speed_of_sound * mu_ij + beta * mu_ij * mu_ij) / rho_avg;
                        // Multiply by m_i so that the final `fx / m_i` conversion
                        // to acceleration correctly cancels to -m_j * pi_ij * grad_W.
                        let v_factor = -m_i * read_mass(j) * pi_ij;
                        fx = fx + v_factor * grad.x;
                        fy = fy + v_factor * grad.y;
                        fz = fz + v_factor * grad.z;
                    }
                }
            }
        }
    }

    // Boundary particle contributions: pressure forces + repulsive forces
    // Grid-accelerated lookup replaces O(N_b) linear scan.
    let r0 = 0.5 * h;
    let d_repulsive = 10.0 * 9.81 * 0.01;
    if params.n_boundary > 0u {
        for (var bz_off = -1; bz_off <= 1; bz_off = bz_off + 1) {
            let bnz = cell.z + bz_off;
            if bnz < 0 || bnz >= i32(params.grid_dim_z) { continue; }
            for (var by_off = -1; by_off <= 1; by_off = by_off + 1) {
                let bny = cell.y + by_off;
                if bny < 0 || bny >= i32(params.grid_dim_y) { continue; }
                for (var bx_off = -1; bx_off <= 1; bx_off = bx_off + 1) {
                    let bnx = cell.x + bx_off;
                    if bnx < 0 || bnx >= i32(params.grid_dim_x) { continue; }

                    let bc = cell_hash(u32(bnx), u32(bny), u32(bnz));
                    let b_start = bnd_cell_offsets[bc];
                    let b_count = bnd_cell_counts[bc];

                    for (var bs = b_start; bs < b_start + b_count; bs = bs + 1u) {
                        let b = bnd_sorted_indices[bs];
                        let ddx = px - bnd_x[b];
                        let ddy = py - bnd_y[b];
                        let ddz = pz - bnd_z[b];
                        let dist_sq = ddx * ddx + ddy * ddy + ddz * ddz;

                        if dist_sq < support_radius_sq {
                            let grad = wendland_c2_gradient_from_dist_sq(ddx, ddy, ddz, dist_sq, h);
                            let pb_over_rho2_b = bnd_pressure[b] / (boundary_rho * boundary_rho);
                            let factor = -m_i * bnd_mass[b] * (pi_clamped_over_rho2 + pb_over_rho2_b);
                            fx = fx + factor * grad.x;
                            fy = fy + factor * grad.y;
                            fz = fz + factor * grad.z;

                            let inv_r_bnd = inverseSqrt(max(dist_sq, 1.0e-24));
                            let r_bnd = dist_sq * inv_r_bnd;
                            if r_bnd < r0 && dist_sq > 1.0e-24 {
                                let s = 1.0 - r_bnd / r0;
                                let force_mag = d_repulsive * s * s / r0;
                                fx = fx + force_mag * ddx * inv_r_bnd * m_i;
                                fy = fy + force_mag * ddy * inv_r_bnd * m_i;
                                fz = fz + force_mag * ddz * inv_r_bnd * m_i;
                            }
                        }
                    }
                }
            }
        }
    }

    // Convert forces to accelerations: a = F / m, plus gravity
    acc_x[i] = fx / m_i + params.gravity_x;
    acc_y[i] = fy / m_i + params.gravity_y;
    acc_z[i] = fz / m_i + params.gravity_z;
}
