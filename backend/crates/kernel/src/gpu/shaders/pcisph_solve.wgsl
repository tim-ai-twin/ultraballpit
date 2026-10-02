// PCISPH neighbor passes and correction loop.
//
// Every correction iteration sums over each particle's neighbors twice
// (predicted density, pressure forces) and over each boundary particle's fluid
// neighbors once (pressure mirroring), up to 10 times per step. This module
// makes that cheap:
//
// 1. Verlet neighbor lists. `build_neighbors` / `build_boundary_neighbors` run
//    once per step on the step-start positions (with a freshly sorted grid)
//    and record every neighbor within LIST_RADIUS_H * h: the 2h support plus a
//    skin covering the displacement during the prediction (|v| dt ≲ 0.34 h
//    per particle under the advective CFL). The iterations walk only those
//    candidates and apply the exact support test on predicted positions. A
//    particle whose list overflowed falls back to the grid block.
//
// 2. Several threads per particle. One thread per particle leaves a small
//    scene latency-bound, so density_* / pressure_force give each particle
//    SLICES threads that split its candidates (list entries k ≡ slice mod
//    SLICES, or one z-slab of the grid block on the fallback path); partial
//    sums are combined through workgroup memory in a fixed order. Workgroup
//    layout: lid = slice * PPW + p.
//
// 3. Packed state. Predicted positions + mass are gathered as one vec4 per
//    candidate (pos4), pressure / density² is precomputed per particle
//    (p_rho2), and all loop passes share one bind group.
//
// Per-pair arithmetic is identical to the one-thread shaders this replaces
// (density.wgsl with pass_index != 0, forces.wgsl update_boundary_pressures,
// and the former correct_pressure_pcisph / pcisph_pressure_force.wgsl); only
// the summation order differs.
//
// SLICES, PPW, WG, CAP_F, CAP_B, CAP_BF and LIST_RADIUS_H are substituted by
// the host (gpu/mod.rs).
//
// List layout (one buffer, interleaved so SIMD lanes read adjacent words):
//   fluid neighbors of particle i, entry k:     k * n + i
//   boundary neighbors of particle i, entry k:  (CAP_F + k) * n + i
//   fluid neighbors of boundary b, entry k:     (CAP_F + CAP_B) * n + k * nb + b
// counts[i] = fluid | boundary << 16 (particle i); counts[n + b] (boundary b).
// A count above its capacity marks an overflowed list.

const SLICES: u32 = {SLICES}u;
const PPW: u32 = {PPW}u;
const WG: u32 = {WG}u;
const CAP_F: u32 = {CAP_F}u;
const CAP_B: u32 = {CAP_B}u;
const CAP_BF: u32 = {CAP_BF}u;
const LIST_RADIUS_H: f32 = {LIST_RADIUS_H};

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
    search_cells: u32,
};

@group(0) @binding(0) var<uniform> params: SimParams;
@group(0) @binding(1) var<storage, read> pos_x: array<f32>;
@group(0) @binding(2) var<storage, read> pos_y: array<f32>;
@group(0) @binding(3) var<storage, read> pos_z: array<f32>;
@group(0) @binding(4) var<storage, read> mass: array<f32>;
@group(0) @binding(5) var<storage, read_write> density: array<f32>;
@group(0) @binding(6) var<storage, read_write> pressure: array<f32>;
@group(0) @binding(7) var<storage, read> fluid_type: array<u32>;
@group(0) @binding(8) var<storage, read> bnd_x: array<f32>;
@group(0) @binding(9) var<storage, read> bnd_y: array<f32>;
@group(0) @binding(10) var<storage, read> bnd_z: array<f32>;
@group(0) @binding(11) var<storage, read> bnd_mass: array<f32>;
@group(0) @binding(12) var<storage, read_write> bnd_pressure: array<f32>;
@group(0) @binding(13) var<storage, read> bnd_cell_counts: array<u32>;
@group(0) @binding(14) var<storage, read> bnd_cell_offsets: array<u32>;
@group(0) @binding(15) var<storage, read> cell_counts: array<u32>;
@group(0) @binding(16) var<storage, read> cell_offsets: array<u32>;
// PCISPH scaling factor (dt-independent prototype value, uploaded at init;
// see sph::compute_pcisph_prototype_delta).
@group(0) @binding(17) var<storage, read> pcisph_delta: array<f32>;
@group(0) @binding(18) var<storage, read_write> convergence: array<atomic<u32>>;
@group(0) @binding(19) var<storage, read_write> counts: array<u32>;
@group(0) @binding(20) var<storage, read_write> lists: array<u32>;
// Current (predicted) position + mass; pressure / density².
@group(0) @binding(21) var<storage, read_write> pos4: array<vec4<f32>>;
@group(0) @binding(22) var<storage, read_write> p_rho2: array<f32>;
// Packed step state (see pcisph_predict.wgsl).
@group(0) @binding(23) var<storage, read_write> orig4: array<vec4<f32>>;
@group(0) @binding(24) var<storage, read> vel4: array<vec4<f32>>;
@group(0) @binding(25) var<storage, read> np4: array<vec4<f32>>;
@group(0) @binding(26) var<storage, read_write> pacc4: array<vec4<f32>>;

var<workgroup> part_rho: array<f32, WG>;
var<workgroup> part_acc: array<vec3<f32>, WG>;

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
fn row_range(cell: vec3<i32>, search: i32, ny: i32, nz: i32) -> vec2<u32> {
    let x0 = max(cell.x - search, 0);
    let x1 = min(cell.x + search, i32(params.grid_dim_x) - 1);
    let c0 = cell_hash(u32(x0), u32(ny), u32(nz));
    let c1 = cell_hash(u32(x1), u32(ny), u32(nz));
    return vec2<u32>(cell_offsets[c0], cell_offsets[c1] + cell_counts[c1]);
}

// Boundary-particle analogue of row_range (boundary arrays are stored in cell order).
fn bnd_row_range(cell: vec3<i32>, search: i32, ny: i32, nz: i32) -> vec2<u32> {
    let x0 = max(cell.x - search, 0);
    let x1 = min(cell.x + search, i32(params.grid_dim_x) - 1);
    let c0 = cell_hash(u32(x0), u32(ny), u32(nz));
    let c1 = cell_hash(u32(x1), u32(ny), u32(nz));
    return vec2<u32>(bnd_cell_offsets[c0], bnd_cell_offsets[c1] + bnd_cell_counts[c1]);
}

// Distance along one axis from local coordinate `x` to grid slab `c` (0 if
// inside), less a rounding margin. The first/last slabs extend to infinity:
// they also hold particles clamped in from outside the domain.
fn axis_gap(x: f32, c: i32, dim: i32, cs: f32) -> f32 {
    var lo = f32(c) * cs;
    var hi = lo + cs;
    if c == 0 { lo = -1.0e30; }
    if c == dim - 1 { hi = 1.0e30; }
    return max(max(lo - x, x - hi) - 1.0e-4 * cs, 0.0);
}

// Cells [x0, x1] of grid row (ny, nz) that can hold particles within
// sqrt(radius_sq) of p (x0 > x1: none). The query point may lie anywhere;
// the grid's cell membership is exact because it was built from the
// positions being searched (edge slabs are treated as unbounded).
fn sphere_row(p: vec3<f32>, cell: vec3<i32>, search: i32, ny: i32, nz: i32, radius_sq: f32) -> vec2<i32> {
    let cs = params.cell_size;
    let gy = axis_gap(p.y - params.domain_min_y, ny, i32(params.grid_dim_y), cs);
    let gz = axis_gap(p.z - params.domain_min_z, nz, i32(params.grid_dim_z), cs);
    let rem = radius_sq - gy * gy - gz * gz;
    if rem < 0.0 { return vec2<i32>(1, 0); }
    let rx = sqrt(rem) + 1.0e-4 * cs;
    let lx = p.x - params.domain_min_x;
    let x0 = clamp(max(cell.x - search, i32(floor((lx - rx) / cs))), 0, i32(params.grid_dim_x) - 1);
    let x1 = clamp(min(cell.x + search, i32(floor((lx + rx) / cs))), 0, i32(params.grid_dim_x) - 1);
    return vec2<i32>(x0, x1);
}

fn list_ok(count: u32) -> bool {
    return (count & 0xffffu) <= CAP_F && (count >> 16u) <= CAP_B;
}

// ---- Per-pair terms (identical arithmetic to the one-thread shaders) ----

fn rho_fluid(i: u32, p: vec3<f32>, j: u32, h: f32, support_radius_sq: f32) -> f32 {
    if j == i { return 0.0; }
    let q = pos4[j];
    let ddx = p.x - q.x;
    let ddy = p.y - q.y;
    let ddz = p.z - q.z;
    let dist_sq = ddx * ddx + ddy * ddy + ddz * ddz;
    if dist_sq <= support_radius_sq {
        let r = dist_sq * inverseSqrt(max(dist_sq, 1.0e-24));
        return q.w * wendland_c2(r, h);
    }
    return 0.0;
}

fn rho_bnd(p: vec3<f32>, b: u32, h: f32, support_radius_sq: f32) -> f32 {
    let ddx = p.x - bnd_x[b];
    let ddy = p.y - bnd_y[b];
    let ddz = p.z - bnd_z[b];
    let dist_sq = ddx * ddx + ddy * ddy + ddz * ddz;
    if dist_sq < support_radius_sq {
        let r = dist_sq * inverseSqrt(max(dist_sq, 1.0e-24));
        return bnd_mass[b] * wendland_c2(r, h);
    }
    return 0.0;
}

// Symmetric SPH pressure gradient (acceleration: the factor carries no m_i).
fn acc_fluid(i: u32, p: vec3<f32>, p_over_rho2_i: f32, j: u32, h: f32, support_radius_sq: f32) -> vec3<f32> {
    if j == i { return vec3<f32>(0.0); }
    let q = pos4[j];
    let ddx = p.x - q.x;
    let ddy = p.y - q.y;
    let ddz = p.z - q.z;
    let dist_sq = ddx * ddx + ddy * ddy + ddz * ddz;
    if dist_sq > support_radius_sq { return vec3<f32>(0.0); }
    let grad = wendland_c2_gradient_from_dist_sq(ddx, ddy, ddz, dist_sq, h);
    let factor = -q.w * (p_over_rho2_i + p_rho2[j]);
    return factor * grad;
}

// Boundary pressure mirroring (Adami et al. 2012), pressure forces only.
fn acc_bnd(p: vec3<f32>, p_over_rho2_i: f32, boundary_rho: f32, b: u32, h: f32, support_radius_sq: f32) -> vec3<f32> {
    let ddx = p.x - bnd_x[b];
    let ddy = p.y - bnd_y[b];
    let ddz = p.z - bnd_z[b];
    let dist_sq = ddx * ddx + ddy * ddy + ddz * ddz;
    if dist_sq < support_radius_sq {
        let grad = wendland_c2_gradient_from_dist_sq(ddx, ddy, ddz, dist_sq, h);
        let p_over_rho2_b = bnd_pressure[b] / (boundary_rho * boundary_rho);
        let factor = -bnd_mass[b] * (p_over_rho2_i + p_over_rho2_b);
        return factor * grad;
    }
    return vec3<f32>(0.0);
}

// Hydrostatically extrapolated fluid pressure seen by a boundary particle at
// bp (as forces.wgsl update_boundary_pressures): (w, w * p_extrapolated).
fn bnd_weight(bp: vec3<f32>, f: u32, h: f32, support_radius_sq: f32) -> vec2<f32> {
    let q = pos4[f];
    let dx = bp.x - q.x;
    let dy = bp.y - q.y;
    let dz = bp.z - q.z;
    let dist_sq = dx * dx + dy * dy + dz * dz;
    if dist_sq < support_radius_sq {
        let inv_r = inverseSqrt(max(dist_sq, 1.0e-24));
        let r = dist_sq * inv_r;
        let w = wendland_c2(r, h);
        let g_dot_dr = params.gravity_x * dx + params.gravity_y * dy + params.gravity_z * dz;
        let p_extrapolated = pressure[f] + density[f] * g_dot_dr;
        return vec2<f32>(w, w * p_extrapolated);
    }
    return vec2<f32>(0.0);
}

// ---------------------------------------------------------------------------
// Entry point: per-step neighbor lists of fluid particles; also snapshots the
// step-start position + mass into pos4 / orig4. One thread per particle.
// ---------------------------------------------------------------------------
@compute @workgroup_size(64)
fn build_neighbors(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let n = params.n_particles;
    if i >= n {
        return;
    }

    let h = params.h;
    let list_radius = LIST_RADIUS_H * h;
    let list_radius_sq = list_radius * list_radius;
    // Enough cells to cover the list radius from anywhere in the cell.
    let search = i32(ceil(list_radius / params.cell_size));

    let p = vec3<f32>(pos_x[i], pos_y[i], pos_z[i]);
    let snapshot = vec4<f32>(p, mass[i]);
    pos4[i] = snapshot;
    orig4[i] = snapshot;
    let cell = pos_to_cell_i32(p.x, p.y, p.z);

    // Only cells that intersect the list sphere are scanned (sphere_row).
    var nf = 0u;
    var nb = 0u;
    for (var dz = -search; dz <= search; dz = dz + 1) {
        let nz = cell.z + dz;
        if nz < 0 || nz >= i32(params.grid_dim_z) { continue; }
        for (var dy = -search; dy <= search; dy = dy + 1) {
            let ny = cell.y + dy;
            if ny < 0 || ny >= i32(params.grid_dim_y) { continue; }
            let xr = sphere_row(p, cell, search, ny, nz, list_radius_sq);
            if xr.x > xr.y { continue; }
            let c0 = cell_hash(u32(xr.x), u32(ny), u32(nz));
            let c1 = cell_hash(u32(xr.y), u32(ny), u32(nz));

            for (var j = cell_offsets[c0]; j < cell_offsets[c1] + cell_counts[c1]; j = j + 1u) {
                if j == i { continue; }
                let ddx = p.x - pos_x[j];
                let ddy = p.y - pos_y[j];
                let ddz = p.z - pos_z[j];
                if ddx * ddx + ddy * ddy + ddz * ddz <= list_radius_sq {
                    if nf < CAP_F {
                        lists[nf * n + i] = j;
                    }
                    nf = nf + 1u;
                }
            }
            if params.n_boundary > 0u {
                for (var b = bnd_cell_offsets[c0]; b < bnd_cell_offsets[c1] + bnd_cell_counts[c1]; b = b + 1u) {
                    let ddx = p.x - bnd_x[b];
                    let ddy = p.y - bnd_y[b];
                    let ddz = p.z - bnd_z[b];
                    if ddx * ddx + ddy * ddy + ddz * ddz <= list_radius_sq {
                        if nb < CAP_B {
                            lists[(CAP_F + nb) * n + i] = b;
                        }
                        nb = nb + 1u;
                    }
                }
            }
        }
    }
    counts[i] = min(nf, CAP_F + 1u) | (min(nb, CAP_B + 1u) << 16u);
}

// ---------------------------------------------------------------------------
// Entry point: per-step fluid-neighbor lists of boundary particles
// ---------------------------------------------------------------------------
@compute @workgroup_size(64)
fn build_boundary_neighbors(@builtin(global_invocation_id) gid: vec3<u32>) {
    let b = gid.x;
    let n = params.n_particles;
    let nb_total = params.n_boundary;
    if b >= nb_total {
        return;
    }

    let h = params.h;
    let list_radius = LIST_RADIUS_H * h;
    let list_radius_sq = list_radius * list_radius;
    let search = i32(ceil(list_radius / params.cell_size));

    let bp = vec3<f32>(bnd_x[b], bnd_y[b], bnd_z[b]);
    let cell = pos_to_cell_i32(bp.x, bp.y, bp.z);
    let base = (CAP_F + CAP_B) * n;

    var nf = 0u;
    for (var dz = -search; dz <= search; dz = dz + 1) {
        let nz = cell.z + dz;
        if nz < 0 || nz >= i32(params.grid_dim_z) { continue; }
        for (var dy = -search; dy <= search; dy = dy + 1) {
            let ny = cell.y + dy;
            if ny < 0 || ny >= i32(params.grid_dim_y) { continue; }
            let xr = sphere_row(bp, cell, search, ny, nz, list_radius_sq);
            if xr.x > xr.y { continue; }
            let c0 = cell_hash(u32(xr.x), u32(ny), u32(nz));
            let c1 = cell_hash(u32(xr.y), u32(ny), u32(nz));
            for (var f = cell_offsets[c0]; f < cell_offsets[c1] + cell_counts[c1]; f = f + 1u) {
                let dx = bp.x - pos_x[f];
                let dy2 = bp.y - pos_y[f];
                let dz2 = bp.z - pos_z[f];
                if dx * dx + dy2 * dy2 + dz2 * dz2 <= list_radius_sq {
                    if nf < CAP_BF {
                        lists[base + nf * nb_total + b] = f;
                    }
                    nf = nf + 1u;
                }
            }
        }
    }
    counts[n + b] = min(nf, CAP_BF + 1u);
}

// ---------------------------------------------------------------------------
// Entry point: predict positions from the latest pressure acceleration:
//   v* = v + (a_np + a_p) * dt,  x* = x + v* * dt
// Thread 0 also starts the iteration's convergence counters.
// ---------------------------------------------------------------------------
@compute @workgroup_size(64)
fn predict_positions(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i == 0u {
        atomicStore(&convergence[0], 0u);
        atomicStore(&convergence[1], 0u);
        atomicAdd(&convergence[2], 1u);
    }
    if i >= params.n_particles {
        return;
    }
    let dt = params.dt;
    let v0 = vel4[i];
    let a_np = np4[i];
    let a_p = pacc4[i];
    let o = orig4[i];
    let vx = v0.x + (a_np.x + a_p.x) * dt;
    let vy = v0.y + (a_np.y + a_p.y) * dt;
    let vz = v0.z + (a_np.z + a_p.z) * dt;
    pos4[i] = vec4<f32>(o.x + vx * dt, o.y + vy * dt, o.z + vz * dt, o.w);
}

// This slice's share of particle i's density sum at pos4.
fn density_partial(i: u32, slice: u32) -> f32 {
    let n = params.n_particles;
    let h = params.h;
    let support_radius = 2.0 * h;
    let support_radius_sq = support_radius * support_radius;
    let q = pos4[i];
    let p = q.xyz;

    var rho = 0.0;
    // Self-contribution (counted once, by slice 0)
    if slice == 0u {
        rho = q.w * wendland_c2(0.0, h);
    }

    let count = counts[i];
    if list_ok(count) {
        let nf = count & 0xffffu;
        for (var k = slice; k < nf; k = k + SLICES) {
            rho = rho + rho_fluid(i, p, lists[k * n + i], h, support_radius_sq);
        }
        let nb = count >> 16u;
        for (var k = slice; k < nb; k = k + SLICES) {
            rho = rho + rho_bnd(p, lists[(CAP_F + k) * n + i], h, support_radius_sq);
        }
    } else {
        // Grid block, one z-slab per slice
        let search = i32(params.search_cells);
        let cell = pos_to_cell_i32(p.x, p.y, p.z);
        let nz = cell.z + i32(slice) - search;
        if i32(slice) <= 2 * search && nz >= 0 && nz < i32(params.grid_dim_z) {
            for (var dy = -search; dy <= search; dy = dy + 1) {
                let ny = cell.y + dy;
                if ny < 0 || ny >= i32(params.grid_dim_y) { continue; }
                let j_range = row_range(cell, search, ny, nz);
                for (var j = j_range.x; j < j_range.y; j = j + 1u) {
                    rho = rho + rho_fluid(i, p, j, h, support_radius_sq);
                }
                if params.n_boundary > 0u {
                    let b_range = bnd_row_range(cell, search, ny, nz);
                    for (var b = b_range.x; b < b_range.y; b = b + 1u) {
                        rho = rho + rho_bnd(p, b, h, support_radius_sq);
                    }
                }
            }
        }
    }
    return rho;
}

// Sum of the SLICES partials of this invocation's particle (every invocation
// of the workgroup must call this: it contains a barrier).
fn reduce_rho(rho: f32, lid: u32) -> f32 {
    part_rho[lid] = rho;
    workgroupBarrier();
    let p = lid % PPW;
    var total = part_rho[p];
    for (var s = 1u; s < SLICES; s = s + 1u) {
        total = total + part_rho[s * PPW + p];
    }
    return total;
}

// ---------------------------------------------------------------------------
// Entry point: density only (step start; pressure stays 0)
// ---------------------------------------------------------------------------
@compute @workgroup_size(WG)
fn density_only(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_index) lid: u32,
) {
    let slice = lid / PPW;
    let i = wid.x * PPW + lid % PPW;
    var rho = 0.0;
    // No early return: every invocation must reach the barrier.
    if i < params.n_particles {
        rho = density_partial(i, slice);
    }
    let total = reduce_rho(rho, lid);
    if slice == 0u && i < params.n_particles {
        density[i] = total;
    }
}

// ---------------------------------------------------------------------------
// Entry point: predicted density + pressure correction + convergence counters
// ---------------------------------------------------------------------------
@compute @workgroup_size(WG)
fn density_correct(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_index) lid: u32,
) {
    let slice = lid / PPW;
    let i = wid.x * PPW + lid % PPW;
    var rho = 0.0;
    if i < params.n_particles {
        rho = density_partial(i, slice);
    }
    let total = reduce_rho(rho, lid);

    if slice == 0u && i < params.n_particles {
        density[i] = total;

        // Pressure correction from the predicted density error
        // (as the former correct_pressure_pcisph).
        var rho0 = WATER_REST_DENSITY;
        if fluid_type[i] != 0u {
            rho0 = AIR_REST_DENSITY;
        }
        // Same update as the CPU solver: p += delta * (rho* - rho0), with the
        // absolute error (delta is in Pa per kg/m^3) and both signs, so an
        // under-dense particle relaxes its pressure; clamped at 0 (no tension).
        let density_error = (total - rho0) / rho0;
        let dt = params.dt;
        let effective_delta = pcisph_delta[i] / (dt * dt);
        let correction = effective_delta * (total - rho0);
        let p_new = max(pressure[i] + correction, 0.0);
        pressure[i] = p_new;
        p_rho2[i] = p_new / (total * total);

        if density_error > 0.0 {
            atomicAdd(&convergence[0], u32(density_error * 1000000.0));
            atomicAdd(&convergence[1], 1u);
        }
    }
}

// ---------------------------------------------------------------------------
// Entry point: boundary pressure mirroring (one thread per boundary particle)
// ---------------------------------------------------------------------------
@compute @workgroup_size(64)
fn boundary_pressure(@builtin(global_invocation_id) gid: vec3<u32>) {
    let b = gid.x;
    let n = params.n_particles;
    let nb_total = params.n_boundary;
    if b >= nb_total {
        return;
    }

    let h = params.h;
    let support_radius = 2.0 * h;
    let support_radius_sq = support_radius * support_radius;
    let bp = vec3<f32>(bnd_x[b], bnd_y[b], bnd_z[b]);

    var sums = vec2<f32>(0.0); // (weight_sum, weighted_pressure)
    let count = counts[n + b];
    if count <= CAP_BF {
        let base = (CAP_F + CAP_B) * n;
        for (var k = 0u; k < count; k = k + 1u) {
            sums = sums + bnd_weight(bp, lists[base + k * nb_total + b], h, support_radius_sq);
        }
    } else {
        let search = i32(params.search_cells);
        let bcell = pos_to_cell_i32(bp.x, bp.y, bp.z);
        for (var dz = -search; dz <= search; dz = dz + 1) {
            let nz = bcell.z + dz;
            if nz < 0 || nz >= i32(params.grid_dim_z) { continue; }
            for (var dy = -search; dy <= search; dy = dy + 1) {
                let ny = bcell.y + dy;
                if ny < 0 || ny >= i32(params.grid_dim_y) { continue; }
                let f_range = row_range(bcell, search, ny, nz);
                for (var f = f_range.x; f < f_range.y; f = f + 1u) {
                    sums = sums + bnd_weight(bp, f, h, support_radius_sq);
                }
            }
        }
    }

    if sums.x > 1.0e-12 {
        bnd_pressure[b] = max(sums.y / sums.x, 0.0);
    } else {
        bnd_pressure[b] = 0.0;
    }
}

// ---------------------------------------------------------------------------
// Entry point: pressure-gradient acceleration (fluid + mirrored boundary)
// ---------------------------------------------------------------------------
@compute @workgroup_size(WG)
fn pressure_force(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_index) lid: u32,
) {
    let slice = lid / PPW;
    let i = wid.x * PPW + lid % PPW;
    let n = params.n_particles;

    var f = vec3<f32>(0.0, 0.0, 0.0);
    if i < n {
        let h = params.h;
        let support_radius = 2.0 * h;
        let support_radius_sq = support_radius * support_radius;
        let p = pos4[i].xyz;
        let p_over_rho2_i = p_rho2[i];

        // Rest density for boundary pressure denominator
        var boundary_rho = WATER_REST_DENSITY;
        if fluid_type[i] != 0u {
            boundary_rho = AIR_REST_DENSITY;
        }

        let count = counts[i];
        if list_ok(count) {
            let nf = count & 0xffffu;
            for (var k = slice; k < nf; k = k + SLICES) {
                f = f + acc_fluid(i, p, p_over_rho2_i, lists[k * n + i], h, support_radius_sq);
            }
            let nb = count >> 16u;
            for (var k = slice; k < nb; k = k + SLICES) {
                f = f + acc_bnd(p, p_over_rho2_i, boundary_rho, lists[(CAP_F + k) * n + i], h, support_radius_sq);
            }
        } else {
            // Grid block, one z-slab per slice
            let search = i32(params.search_cells);
            let cell = pos_to_cell_i32(p.x, p.y, p.z);
            let nz = cell.z + i32(slice) - search;
            if i32(slice) <= 2 * search && nz >= 0 && nz < i32(params.grid_dim_z) {
                for (var dy = -search; dy <= search; dy = dy + 1) {
                    let ny = cell.y + dy;
                    if ny < 0 || ny >= i32(params.grid_dim_y) { continue; }
                    let j_range = row_range(cell, search, ny, nz);
                    for (var j = j_range.x; j < j_range.y; j = j + 1u) {
                        f = f + acc_fluid(i, p, p_over_rho2_i, j, h, support_radius_sq);
                    }
                    if params.n_boundary > 0u {
                        let b_range = bnd_row_range(cell, search, ny, nz);
                        for (var b = b_range.x; b < b_range.y; b = b + 1u) {
                            f = f + acc_bnd(p, p_over_rho2_i, boundary_rho, b, h, support_radius_sq);
                        }
                    }
                }
            }
        }
    }

    part_acc[lid] = f;
    workgroupBarrier();

    if slice == 0u && i < n {
        var total = part_acc[lid];
        for (var s = 1u; s < SLICES; s = s + 1u) {
            total = total + part_acc[s * PPW + lid];
        }
        pacc4[i] = vec4<f32>(total, 0.0);
    }
}
