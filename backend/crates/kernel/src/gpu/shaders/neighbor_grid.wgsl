// Neighbor grid construction compute shader
// Implements uniform-grid spatial hashing for particle neighbor search.
//
// A grid build is three dispatches:
// 1. count_particles: hash particles to cells; count particles per cell
//    (cell_fill) and per scan tile of SCAN_TILE consecutive cells (tile_sums).
// 2. prefix_sum: one workgroup per scan tile. Each workgroup sums the tile
//    totals before it to get its base offset, then scans its tile's counts
//    -> cell_counts / cell_offsets.
// 3. sort_scatter.wgsl `scatter`: move every particle to its slot in cell
//    order, counting cell_fill back down to zero.
//
// No clear pass is needed: cell_fill returns to zero during the scatter and
// tile_sums is cleared with a buffer fill alongside the pre-build snapshot.

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

// Group 0: SimParams + positions + mass
@group(0) @binding(0) var<uniform> params: SimParams;
@group(0) @binding(1) var<storage, read> pos_x: array<f32>;
@group(0) @binding(2) var<storage, read> pos_y: array<f32>;
@group(0) @binding(3) var<storage, read> pos_z: array<f32>;

// Group 3: Grid data
@group(3) @binding(0) var<storage, read_write> cell_indices: array<u32>;
@group(3) @binding(1) var<storage, read_write> cell_counts: array<u32>;
@group(3) @binding(2) var<storage, read_write> cell_offsets: array<u32>;
@group(3) @binding(3) var<storage, read_write> sorted_indices: array<u32>;
// Per-cell particle counter: counted up by count_particles, back down to zero
// by the sort scatter.
@group(3) @binding(4) var<storage, read_write> cell_fill: array<atomic<u32>>;
// Particle count per scan tile (cleared before each build).
@group(3) @binding(5) var<storage, read_write> tile_sums: array<atomic<u32>>;

// Cells per prefix_sum workgroup (256 threads x 4 cells). Must match
// GRID_SCAN_TILE in gpu/buffers.rs.
const SCAN_TILE: u32 = 1024u;

fn pos_to_cell(px: f32, py: f32, pz: f32) -> u32 {
    let cx = clamp(
        u32(floor((px - params.domain_min_x) / params.cell_size)),
        0u, params.grid_dim_x - 1u
    );
    let cy = clamp(
        u32(floor((py - params.domain_min_y) / params.cell_size)),
        0u, params.grid_dim_y - 1u
    );
    let cz = clamp(
        u32(floor((pz - params.domain_min_z) / params.cell_size)),
        0u, params.grid_dim_z - 1u
    );
    return cx + cy * params.grid_dim_x + cz * params.grid_dim_x * params.grid_dim_y;
}

fn total_cells() -> u32 {
    return params.grid_dim_x * params.grid_dim_y * params.grid_dim_z;
}

// Pass 1: Count particles per cell and per scan tile.
//
// Particles are (nearly) in cell order from the previous build, so a
// workgroup's particles fall in a few consecutive tiles: tally those in
// workgroup memory and flush one global atomic per tile, instead of 256
// contended global atomics on the same counter.
const WG_TILES: u32 = 4u;
var<workgroup> wg_tile_counts: array<atomic<u32>, WG_TILES>;
var<workgroup> wg_first_tile: u32;

@compute @workgroup_size(256)
fn count_particles(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid3: vec3<u32>,
) {
    let i = gid.x;
    let lid = lid3.x;
    let valid = i < params.n_particles;
    var tile = 0u;
    if valid {
        let cell = pos_to_cell(pos_x[i], pos_y[i], pos_z[i]);
        cell_indices[i] = cell;
        atomicAdd(&cell_fill[cell], 1u);
        tile = cell / SCAN_TILE;
    }
    if lid < WG_TILES {
        atomicStore(&wg_tile_counts[lid], 0u);
    }
    if lid == 0u {
        wg_first_tile = tile;
    }
    workgroupBarrier();
    let first = wg_first_tile;
    if valid {
        let d = tile - first; // wraps to a huge value if tile < first
        if d < WG_TILES {
            atomicAdd(&wg_tile_counts[d], 1u);
        } else {
            atomicAdd(&tile_sums[tile], 1u);
        }
    }
    workgroupBarrier();
    if lid < WG_TILES {
        let c = atomicLoad(&wg_tile_counts[lid]);
        if c > 0u {
            atomicAdd(&tile_sums[first + lid], c);
        }
    }
}
