// Boundary pressure mirroring (Adami et al. 2012).
//
// For each boundary particle b, the Wendland-weighted average of the
// hydrostatically extrapolated pressure of the fluid particles within 2h:
//
//   p_b = max(sum_f W_bf (p_f + rho_f g . (x_b - x_f)) / sum_f W_bf, 0)
//
// and 0 when no fluid particle is in range.

const WENDLAND_C2_NORM_3D: f32 = 0.41780189; // 21 / (16 * PI)

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

@group(0) @binding(0) var<uniform> params: SimParams;
// Fluid particles (stored in cell order)
@group(0) @binding(1) var<storage, read> pos_x: array<f32>;
@group(0) @binding(2) var<storage, read> pos_y: array<f32>;
@group(0) @binding(3) var<storage, read> pos_z: array<f32>;
@group(0) @binding(4) var<storage, read> density: array<f32>;
@group(0) @binding(5) var<storage, read> pressure: array<f32>;
// Fluid neighbor grid
@group(0) @binding(6) var<storage, read> cell_offsets: array<u32>;
@group(0) @binding(7) var<storage, read> cell_counts: array<u32>;
// Boundary particles
@group(0) @binding(8) var<storage, read> bnd_x: array<f32>;
@group(0) @binding(9) var<storage, read> bnd_y: array<f32>;
@group(0) @binding(10) var<storage, read> bnd_z: array<f32>;
@group(0) @binding(11) var<storage, read_write> bnd_pressure: array<f32>;

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

// Distance from coordinate `p` to the slab of grid cells with index `c` along
// one axis. Edge cells extend to infinity, since positions outside the
// domain are clamped into them.
fn slab_dist(p: f32, c: i32, dmin: f32, dim: u32) -> f32 {
    let lo = dmin + f32(c) * params.cell_size;
    var d = 0.0;
    if c > 0 && p < lo {
        d = lo - p;
    }
    if c < i32(dim) - 1 && p > lo + params.cell_size {
        d = p - (lo + params.cell_size);
    }
    return d;
}

// Each grid row (dy, dz) of the search block is culled against the support
// sphere: rows whose cells are all farther than 2h are skipped, and the rest
// are clipped to the x cells the sphere can reach. Only fluid particles that
// fail the `dist_sq < support_radius_sq` test are dropped, so the result is
// the same as scanning the full block.
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
    // Slightly enlarged sphere for culling, so rounding never drops a pair
    // that passes the exact test below.
    let cull_r2 = support_radius_sq * 1.0001;
    let inv_cs = 1.0 / params.cell_size;

    let bx = bnd_x[b];
    let by = bnd_y[b];
    let bz = bnd_z[b];

    var weighted_pressure = 0.0;
    var weight_sum = 0.0;

    let bcell = pos_to_cell_i32(bx, by, bz);
    let max_x = i32(params.grid_dim_x) - 1;
    for (var dz_off = -search; dz_off <= search; dz_off = dz_off + 1) {
        let nz = bcell.z + dz_off;
        if nz < 0 || nz >= i32(params.grid_dim_z) { continue; }
        let dz_slab = slab_dist(bz, nz, params.domain_min_z, params.grid_dim_z);
        let rem_z = cull_r2 - dz_slab * dz_slab;
        if rem_z < 0.0 { continue; }
        for (var dy_off = -search; dy_off <= search; dy_off = dy_off + 1) {
            let ny = bcell.y + dy_off;
            if ny < 0 || ny >= i32(params.grid_dim_y) { continue; }
            let dy_slab = slab_dist(by, ny, params.domain_min_y, params.grid_dim_y);
            let rem = rem_z - dy_slab * dy_slab;
            if rem < 0.0 { continue; }

            // x cells the sphere reaches in this row, within the search block.
            let rx = sqrt(rem);
            let x0 = max(max(i32(floor((bx - rx - params.domain_min_x) * inv_cs)), bcell.x - search), 0);
            let x1 = min(min(i32(floor((bx + rx - params.domain_min_x) * inv_cs)), bcell.x + search), max_x);
            if x0 > x1 { continue; }
            let c0 = cell_hash(u32(x0), u32(ny), u32(nz));
            let c1 = cell_hash(u32(x1), u32(ny), u32(nz));
            let f_end = cell_offsets[c1] + cell_counts[c1];
            for (var f = cell_offsets[c0]; f < f_end; f = f + 1u) {
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

    if weight_sum > 1.0e-12 {
        bnd_pressure[b] = max(weighted_pressure / weight_sum, 0.0);
    } else {
        bnd_pressure[b] = 0.0;
    }
}
