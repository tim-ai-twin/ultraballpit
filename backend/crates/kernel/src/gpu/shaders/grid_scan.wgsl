// Grid build pass 2 (see neighbor_grid.wgsl): exclusive prefix sum of the
// per-cell particle counts, one workgroup per tile of SCAN_TILE cells. Each
// workgroup sums the totals of the tiles before it (counted by
// count_particles) for its base offset, then scans its own tile.
//
// A separate module from neighbor_grid.wgsl so the counters, atomic there,
// can be read here with plain loads.

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
    // Pass selector for multi-pass shaders
    pass_index: u32,
    search_cells: u32,
};

@group(0) @binding(0) var<uniform> params: SimParams;

@group(3) @binding(1) var<storage, read_write> cell_counts: array<u32>;
@group(3) @binding(2) var<storage, read_write> cell_offsets: array<u32>;
@group(3) @binding(4) var<storage, read_write> cell_fill: array<u32>;
@group(3) @binding(5) var<storage, read_write> tile_sums: array<u32>;

// Cells per prefix_sum workgroup (256 threads x 4 cells). Must match
// GRID_SCAN_TILE in gpu/buffers.rs.
const SCAN_TILE: u32 = 1024u;

fn total_cells() -> u32 {
    return params.grid_dim_x * params.grid_dim_y * params.grid_dim_z;
}

// Ping-pong buffer for the workgroup scan (two halves of 256).
var<workgroup> scan_buf: array<vec2<u32>, 512>;

// Inclusive Hillis-Steele scan of one vec2 per thread across the workgroup
// (both components scanned independently). Returns (this thread's inclusive
// sum, workgroup total). Must be called from uniform control flow.
fn workgroup_scan2(lid: u32, v: vec2<u32>) -> array<vec2<u32>, 2> {
    scan_buf[lid] = v;
    workgroupBarrier();
    var src = 0u;
    for (var off = 1u; off < 256u; off = off << 1u) {
        var x = scan_buf[src + lid];
        if lid >= off {
            x = x + scan_buf[src + lid - off];
        }
        src = 256u - src;
        scan_buf[src + lid] = x;
        workgroupBarrier();
    }
    return array<vec2<u32>, 2>(scan_buf[src + lid], scan_buf[src + 255u]);
}

@compute @workgroup_size(256)
fn prefix_sum(
    @builtin(local_invocation_id) lid3: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
) {
    let lid = lid3.x;
    let tile = wid.x;
    let n_cells = total_cells();

    // x: this thread's share of the particles in all earlier tiles (their
    // sum is the tile's base offset); y: particles in this thread's 4 cells.
    var before = 0u;
    for (var t = lid; t < tile; t = t + 256u) {
        before = before + tile_sums[t];
    }
    let c0 = tile * SCAN_TILE + lid * 4u;
    var counts = vec4<u32>(0u);
    for (var k = 0u; k < 4u; k = k + 1u) {
        if c0 + k < n_cells {
            counts[k] = cell_fill[c0 + k];
        }
    }
    let thread_total = counts.x + counts.y + counts.z + counts.w;
    let scan = workgroup_scan2(lid, vec2<u32>(before, thread_total));
    let base = scan[1].x;
    var running = base + scan[0].y - thread_total;
    for (var k = 0u; k < 4u; k = k + 1u) {
        if c0 + k < n_cells {
            cell_counts[c0 + k] = counts[k];
            cell_offsets[c0 + k] = running;
        }
        running = running + counts[k];
    }
}
