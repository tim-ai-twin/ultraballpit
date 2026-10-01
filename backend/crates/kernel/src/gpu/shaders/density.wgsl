// SPH density summation compute shader
//
// Computes density for each particle using the Wendland C2 kernel:
//   rho_i = sum_j m_j * W(|r_i - r_j|, h)
//
// Includes self-contribution, fluid neighbor contributions (via neighbor grid),
// and boundary particle contributions.
//
// Also computes delta-SPH density diffusion (Molteni & Colagrossi 2009):
//   D_i = delta * h * c_s * sum_j V_j * (rho_i - rho_j) * 2 * (r · gradW) / |r|^2
// Using the previous step's density for the diffusion source term.

const PI: f32 = 3.14159265358979323846;
const WENDLAND_C2_NORM_3D: f32 = 0.41780189; // 21 / (16 * PI)
const DELTA_SPH: f32 = 0.1;

// EOS constants
const WATER_REST_DENSITY: f32 = 1000.0;
const AIR_REST_DENSITY: f32 = 1.204;
const WATER_GAMMA: f32 = 7.0;
const AIR_GAMMA: f32 = 1.4;

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

// Group 1: Neighbor lists, written here and reused by the forces pass (same
// step) and the XSPH pass (next step), which see the same positions. Stored
// interleaved, list[k * n_particles + i], so reads coalesce across threads.
// Counts record every neighbor; a count above the cap means the list is
// incomplete and consumers fall back to a grid scan.
@group(1) @binding(0) var<storage, read_write> nbr_list: array<u32>;
@group(1) @binding(1) var<storage, read_write> nbr_count: array<u32>;
@group(1) @binding(2) var<storage, read_write> bnd_nbr_list: array<u32>;
@group(1) @binding(3) var<storage, read_write> bnd_nbr_count: array<u32>;

const MAX_NBR: u32 = 128u;
const MAX_BND_NBR: u32 = 64u;

// Group 2: SPH state + boundary
@group(2) @binding(0) var<storage, read_write> density: array<f32>;
@group(2) @binding(1) var<storage, read_write> pressure: array<f32>;
@group(2) @binding(2) var<storage, read> fluid_type: array<u32>;
@group(2) @binding(3) var<storage, read> bnd_x: array<f32>;
@group(2) @binding(4) var<storage, read> bnd_y: array<f32>;
@group(2) @binding(5) var<storage, read> bnd_z: array<f32>;
@group(2) @binding(6) var<storage, read> bnd_mass: array<f32>;
@group(2) @binding(7) var<storage, read> bnd_cell_counts: array<u32>;
@group(2) @binding(8) var<storage, read> bnd_cell_offsets: array<u32>;
@group(2) @binding(9) var<storage, read> bnd_sorted_indices: array<u32>;

// Group 3: Grid data (read-only for density)
@group(3) @binding(2) var<storage, read> cell_offsets: array<u32>;
@group(3) @binding(1) var<storage, read> cell_counts: array<u32>;
@group(3) @binding(3) var<storage, read> sorted_indices: array<u32>;

fn read_mass(idx: u32) -> f32 {
    return mass[idx];
}

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

fn tait_eos(rho: f32, rho0: f32, cs: f32, gamma: f32) -> f32 {
    let b = rho0 * cs * cs / gamma;
    let ratio = rho / rho0;
    let p = b * (pow(ratio, gamma) - 1.0);
    return max(p, 0.0);
}

@compute @workgroup_size(256)
fn compute_density(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i >= params.n_particles {
        return;
    }

    let h = params.h;
    let search = i32(params.search_cells);
    let support_radius = 2.0 * h;
    let support_radius_sq = support_radius * support_radius;
    let eta_sq = 0.01 * h * h;

    let px = pos_x[i];
    let py = pos_y[i];
    let pz = pos_z[i];

    // Read old density for delta-SPH diffusion (before overwrite)
    let old_rho_i = density[i];

    // Self-contribution
    var rho = read_mass(i) * wendland_c2(0.0, h);

    // Delta-SPH diffusion accumulator
    var diff_sum = 0.0;

    var n_nbr = 0u;
    var n_bnd_nbr = 0u;

    // Fluid neighbor contributions via neighbor grid
    let cell = pos_to_cell_i32(px, py, pz);

    for (var dz = -search; dz <= search; dz = dz + 1) {
        let nz = cell.z + dz;
        if nz < 0 || nz >= i32(params.grid_dim_z) { continue; }
        for (var dy = -search; dy <= search; dy = dy + 1) {
            let ny = cell.y + dy;
            if ny < 0 || ny >= i32(params.grid_dim_y) { continue; }
            {
                // The row's cells are adjacent in hash order and particles are
                // stored in cell order, so the whole row is one index range.
                let j_range = row_range(cell, ny, nz);
                for (var j = j_range.x; j < j_range.y; j = j + 1u) {
                    if j == i { continue; }

                    let ddx = px - pos_x[j];
                    let ddy = py - pos_y[j];
                    let ddz = pz - pos_z[j];
                    let dist_sq = ddx * ddx + ddy * ddy + ddz * ddz;

                    if dist_sq <= support_radius_sq {
                        if n_nbr < MAX_NBR {
                            nbr_list[n_nbr * params.n_particles + i] = j;
                        }
                        n_nbr = n_nbr + 1u;
                        let r = dist_sq * inverseSqrt(max(dist_sq, 1.0e-24));
                        rho = rho + read_mass(j) * wendland_c2(r, h);

                        // Delta-SPH diffusion using previous-step densities
                        let grad = wendland_c2_gradient_from_dist_sq(ddx, ddy, ddz, dist_sq, h);
                        let old_rho_j = density[j];
                        let v_j = read_mass(j) / max(old_rho_j, 1.0);
                        let r_dot_grad = ddx * grad.x + ddy * grad.y + ddz * grad.z;
                        // Sign: our r = x_i - x_j, so r·gradW < 0. Use (rho_i - rho_j)
                        // to get correct diffusion sign (paper uses r_ij = x_j - x_i).
                        diff_sum = diff_sum + v_j * (old_rho_i - old_rho_j) * 2.0 * r_dot_grad / (dist_sq + eta_sq);
                    }
                }
            }
        }
    }

    // Boundary particle contributions (grid-accelerated)
    if params.n_boundary > 0u {
        for (var bz_off = -search; bz_off <= search; bz_off = bz_off + 1) {
            let bnz = cell.z + bz_off;
            if bnz < 0 || bnz >= i32(params.grid_dim_z) { continue; }
            for (var by_off = -search; by_off <= search; by_off = by_off + 1) {
                let bny = cell.y + by_off;
                if bny < 0 || bny >= i32(params.grid_dim_y) { continue; }
                {
                    // The row's cells are adjacent in hash order and particles are
                    // stored in cell order, so the whole row is one index range.
                    let b_range = bnd_row_range(cell, bny, bnz);
                    for (var b = b_range.x; b < b_range.y; b = b + 1u) {
                        let ddx = px - bnd_x[b];
                        let ddy = py - bnd_y[b];
                        let ddz = pz - bnd_z[b];
                        let dist_sq = ddx * ddx + ddy * ddy + ddz * ddz;

                        if dist_sq < support_radius_sq {
                            if n_bnd_nbr < MAX_BND_NBR {
                                bnd_nbr_list[n_bnd_nbr * params.n_particles + i] = b;
                            }
                            n_bnd_nbr = n_bnd_nbr + 1u;
                            let r = dist_sq * inverseSqrt(max(dist_sq, 1.0e-24));
                            rho = rho + bnd_mass[b] * wendland_c2(r, h);
                        }
                    }
                }
            }
        }
    }

    // When pass_index == 0: full density with delta-SPH + EOS (WCSPH mode)
    // When pass_index != 0: density-only summation (PCISPH prediction mode)
    if params.pass_index == 0u {
        let diffusion = DELTA_SPH * h * params.speed_of_sound * diff_sum;
        rho = rho + diffusion * params.dt;
    }

    density[i] = rho;
    nbr_count[i] = n_nbr;
    bnd_nbr_count[i] = n_bnd_nbr;

    if params.pass_index == 0u {
        let ft = fluid_type[i];
        if ft == 0u {
            // Water
            pressure[i] = tait_eos(rho, WATER_REST_DENSITY, params.speed_of_sound, WATER_GAMMA);
        } else {
            // Air
            pressure[i] = tait_eos(rho, AIR_REST_DENSITY, params.speed_of_sound, AIR_GAMMA);
        }
    }
}
