// Particle reorder shader
//
// Scatters particle data from original order into spatially-sorted order
// using the grid's sorted_indices as a permutation.
// After reorder, particles in nearby cells are contiguous in memory,
// improving cache hit rates for neighbor traversal.
//
// Uses a single-buffer-at-a-time approach with one temp buffer.

struct ReorderParams {
    n_particles: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
};

@group(0) @binding(0) var<uniform> rparams: ReorderParams;
@group(0) @binding(1) var<storage, read> perm: array<u32>;
@group(0) @binding(2) var<storage, read> source: array<u32>;
@group(0) @binding(3) var<storage, read_write> dest: array<u32>;

// Scatter: dest[k] = source[perm[k]]
@compute @workgroup_size(256)
fn scatter(@builtin(global_invocation_id) gid: vec3<u32>) {
    let k = gid.x;
    if k >= rparams.n_particles {
        return;
    }
    dest[k] = source[perm[k]];
}
