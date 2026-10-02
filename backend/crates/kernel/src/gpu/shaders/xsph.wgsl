// XSPH velocity smoothing (Monaghan 1989)
//
// Computes smoothed velocity correction for each particle:
//   dv_i = epsilon * sum_j (m_j / rho_avg_ij) * (v_j - v_i) * W(r_ij, h)
//
// Writes corrections to acc_x/y/z buffers (which hold stale forces after
// half-kick consumes them). The drift shader then uses acc as XSPH correction.

const PI: f32 = 3.14159265358979323846;
const WENDLAND_C2_NORM_3D: f32 = 0.41780189; // 21 / (16 * PI)
const XSPH_EPSILON: f32 = 0.5;

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

// Group 0: SimParams + positions + mass (same as forces)
@group(0) @binding(0) var<uniform> params: SimParams;
@group(0) @binding(1) var<storage, read> pos_x: array<f32>;
@group(0) @binding(2) var<storage, read> pos_y: array<f32>;
@group(0) @binding(3) var<storage, read> pos_z: array<f32>;
@group(0) @binding(4) var<storage, read> mass: array<f32>;

// Group 1: Velocity (read) + acceleration (write — used for XSPH output)
@group(1) @binding(0) var<storage, read> vel_x: array<f32>;
@group(1) @binding(1) var<storage, read> vel_y: array<f32>;
@group(1) @binding(2) var<storage, read> vel_z: array<f32>;
@group(1) @binding(3) var<storage, read_write> acc_x: array<f32>;
@group(1) @binding(4) var<storage, read_write> acc_y: array<f32>;
@group(1) @binding(5) var<storage, read_write> acc_z: array<f32>;
// Neighbor lists written by the density pass (see density.wgsl).
@group(1) @binding(6) var<storage, read> nbr_list: array<u32>;
@group(1) @binding(7) var<storage, read> nbr_count: array<u32>;
// Packed caches: posm from the density pass, velr refreshed by the half-kick.
@group(1) @binding(10) var<storage, read> posm: array<vec4<f32>>;
@group(1) @binding(11) var<storage, read> velr: array<vec4<f32>>;

const MAX_NBR: u32 = 128u;

// Decode neighbor k of particle i from the packed 16-bit offset list.
fn nbr_at(i: u32, k: u32) -> u32 {
    let word = nbr_list[(k >> 1u) * params.n_particles + i];
    let off = (word >> ((k & 1u) * 16u)) & 0xffffu;
    return u32(i32(i) + i32(off) - 32768);
}

// Group 2: density (only density is used, rest are bound but unused)
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

// Group 3: Grid data
@group(3) @binding(2) var<storage, read> cell_offsets: array<u32>;
@group(3) @binding(1) var<storage, read> cell_counts: array<u32>;

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

// Smoothed-velocity contribution of neighbor j (zero outside the support).
fn xsph_pair(px: f32, py: f32, pz: f32, vi: vec3<f32>, rho_i: f32, j: u32) -> vec3<f32> {
    let h = params.h;
    let support_radius_sq = 4.0 * h * h;
    let pm = posm[j];
    let d = vec3<f32>(px, py, pz) - pm.xyz;
    let dist_sq = dot(d, d);
    if dist_sq > support_radius_sq {
        return vec3<f32>(0.0);
    }
    let vr = velr[j];
    let r = dist_sq * inverseSqrt(max(dist_sq, 1.0e-24));
    let w = wendland_c2(r, h);
    let rho_avg = 0.5 * (rho_i + vr.w);
    let factor = pm.w / max(rho_avg, 1.0) * w;
    return factor * (vr.xyz - vi);
}

@compute @workgroup_size(256)
fn compute_xsph(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i >= params.n_particles {
        return;
    }

    let search = i32(params.search_cells);
    let px = pos_x[i];
    let py = pos_y[i];
    let pz = pos_z[i];
    let vi = vec3<f32>(vel_x[i], vel_y[i], vel_z[i]);
    let rho_i = density[i];

    var c = vec3<f32>(0.0);

    // Positions are unchanged since the last density pass, so its neighbor
    // list is exact here. Fall back to the grid scan if it overflowed.
    let n_nbr = nbr_count[i];
    if n_nbr <= MAX_NBR {
        for (var k = 0u; k < n_nbr; k = k + 1u) {
            c = c + xsph_pair(px, py, pz, vi, rho_i, nbr_at(i, k));
        }
    } else {
        let cell = pos_to_cell_i32(px, py, pz);
        for (var dz = -search; dz <= search; dz = dz + 1) {
            let nz = cell.z + dz;
            if nz < 0 || nz >= i32(params.grid_dim_z) { continue; }
            for (var dy = -search; dy <= search; dy = dy + 1) {
                let ny = cell.y + dy;
                if ny < 0 || ny >= i32(params.grid_dim_y) { continue; }
                let j_range = row_range(cell, ny, nz);
                for (var j = j_range.x; j < j_range.y; j = j + 1u) {
                    if j == i { continue; }
                    c = c + xsph_pair(px, py, pz, vi, rho_i, j);
                }
            }
        }
    }

    // Write XSPH correction to acc buffers (reused as temp storage)
    acc_x[i] = XSPH_EPSILON * c.x;
    acc_y[i] = XSPH_EPSILON * c.y;
    acc_z[i] = XSPH_EPSILON * c.z;
}
