// Neighbor grid construction compute shader
// Implements uniform-grid spatial hashing for particle neighbor search.
//
// Four dispatches are needed:
// Pass 0 (clear): Zero out cell_counts
// Pass 1 (count): Hash particles to cells, count particles per cell via atomics
// Pass 2 (prefix_sum): Compute exclusive prefix sum of cell_counts -> cell_offsets
// Pass 3 (scatter): Scatter particle indices into sorted order

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
    _pad1: u32,
};

// Group 0: SimParams + positions + mass
@group(0) @binding(0) var<uniform> params: SimParams;
@group(0) @binding(1) var<storage, read> pos_x: array<f32>;
@group(0) @binding(2) var<storage, read> pos_y: array<f32>;
@group(0) @binding(3) var<storage, read> pos_z: array<f32>;

// Group 3: Grid data
@group(3) @binding(0) var<storage, read_write> cell_indices: array<atomic<u32>>;
@group(3) @binding(1) var<storage, read_write> cell_counts: array<atomic<u32>>;
@group(3) @binding(2) var<storage, read_write> cell_offsets: array<u32>;
@group(3) @binding(3) var<storage, read_write> sorted_indices: array<u32>;
@group(3) @binding(4) var<storage, read_write> write_heads: array<atomic<u32>>;

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

// Pass 0: Clear cell counts
@compute @workgroup_size(256)
fn clear_counts(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if idx >= total_cells() {
        return;
    }
    atomicStore(&cell_counts[idx], 0u);
    cell_offsets[idx] = 0u;
}

// Pass 1: Count particles per cell
@compute @workgroup_size(256)
fn count_particles(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i >= params.n_particles {
        return;
    }
    let cell = pos_to_cell(pos_x[i], pos_y[i], pos_z[i]);
    atomicStore(&cell_indices[i], cell);
    atomicAdd(&cell_counts[cell], 1u);
}

// Pass 2: Parallel prefix sum using 256 threads with shared memory.
// Each thread handles a contiguous block of cells. A shared-memory scan
// of per-block totals propagates offsets across blocks.
// Handles up to 256 * 256 = 65536 cells in a single workgroup dispatch.
var<workgroup> block_totals: array<u32, 256>;

@compute @workgroup_size(256)
fn prefix_sum(@builtin(local_invocation_id) lid: vec3<u32>) {
    let tid = lid.x;
    let n_cells = total_cells();
    let block_size = (n_cells + 255u) / 256u;
    let start = tid * block_size;
    let end = min(start + block_size, n_cells);

    // Phase 1: each thread computes local prefix sum for its block
    var local_sum = 0u;
    for (var c = start; c < end; c = c + 1u) {
        let count = atomicLoad(&cell_counts[c]);
        cell_offsets[c] = local_sum;
        local_sum = local_sum + count;
    }
    block_totals[tid] = local_sum;

    workgroupBarrier();

    // Phase 2: thread 0 scans the 256 block totals (exclusive prefix sum)
    if tid == 0u {
        var running = 0u;
        for (var i = 0u; i < 256u; i = i + 1u) {
            let old = block_totals[i];
            block_totals[i] = running;
            running = running + old;
        }
    }

    workgroupBarrier();

    // Phase 3: add block offset to local results and initialize write_heads
    let offset = block_totals[tid];
    for (var c = start; c < end; c = c + 1u) {
        cell_offsets[c] = cell_offsets[c] + offset;
        atomicStore(&write_heads[c], cell_offsets[c]);
    }
}

// Pass 3: Scatter particle indices into sorted order
@compute @workgroup_size(256)
fn scatter_particles(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i >= params.n_particles {
        return;
    }
    let cell = atomicLoad(&cell_indices[i]);
    let pos = atomicAdd(&write_heads[cell], 1u);
    sorted_indices[pos] = i;
}
