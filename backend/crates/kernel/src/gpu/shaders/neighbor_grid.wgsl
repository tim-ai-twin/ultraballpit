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
    pass: u32,
    _pad1: u32,
};

@group(0) @binding(0) var<uniform> params: SimParams;

// Particle positions (read-only for grid building)
@group(0) @binding(1) var<storage, read> pos_x: array<f32>;
@group(0) @binding(2) var<storage, read> pos_y: array<f32>;
@group(0) @binding(3) var<storage, read> pos_z: array<f32>;

// Grid data (read-write)
@group(0) @binding(4) var<storage, read_write> cell_indices: array<atomic<u32>>;
@group(0) @binding(5) var<storage, read_write> cell_counts: array<atomic<u32>>;
@group(0) @binding(6) var<storage, read_write> cell_offsets: array<u32>;
@group(0) @binding(7) var<storage, read_write> sorted_indices: array<u32>;
// write_heads: temporary per-cell write pointer for scatter pass
@group(0) @binding(8) var<storage, read_write> write_heads: array<atomic<u32>>;

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

// Pass 2: Prefix sum (sequential, run with 1 workgroup of 1 thread)
// For small grids this is fine. For large grids, a parallel prefix sum would be needed.
@compute @workgroup_size(1)
fn prefix_sum(@builtin(global_invocation_id) gid: vec3<u32>) {
    let n_cells = total_cells();
    var running = 0u;
    for (var c = 0u; c < n_cells; c = c + 1u) {
        let count = atomicLoad(&cell_counts[c]);
        cell_offsets[c] = running;
        atomicStore(&write_heads[c], running);
        running = running + count;
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
