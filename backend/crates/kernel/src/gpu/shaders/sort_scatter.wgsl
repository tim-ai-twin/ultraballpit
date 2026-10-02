// Scatter particles into cell order: the last pass of a grid build.
//
//   d_n[cell_offsets[c] + slot] = s_n[i]   for each persistent array n
//
// where c = cell_indices[i] and slot is claimed by counting cell_fill[c] back
// down (which also leaves cell_fill zeroed for the next build). The s_n are
// snapshots of the arrays taken before the build; arrays are copied as raw
// 32-bit words (f32 and u32 alike). Order within a cell is arbitrary, as with
// any atomic scatter.

struct SortParams {
    n_particles: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
};

@group(0) @binding(0) var<uniform> sp: SortParams;
@group(0) @binding(1) var<storage, read> cell_indices: array<u32>;
@group(0) @binding(2) var<storage, read> cell_offsets: array<u32>;
@group(0) @binding(3) var<storage, read_write> cell_fill: array<atomic<u32>>;

@group(1) @binding(0) var<storage, read> s0: array<u32>;
@group(1) @binding(1) var<storage, read> s1: array<u32>;
@group(1) @binding(2) var<storage, read> s2: array<u32>;
@group(1) @binding(3) var<storage, read> s3: array<u32>;
@group(1) @binding(4) var<storage, read> s4: array<u32>;
@group(1) @binding(5) var<storage, read> s5: array<u32>;
@group(1) @binding(6) var<storage, read> s6: array<u32>;
@group(1) @binding(7) var<storage, read> s7: array<u32>;
@group(1) @binding(8) var<storage, read> s8: array<u32>;
@group(1) @binding(9) var<storage, read> s9: array<u32>;

@group(2) @binding(0) var<storage, read_write> d0: array<u32>;
@group(2) @binding(1) var<storage, read_write> d1: array<u32>;
@group(2) @binding(2) var<storage, read_write> d2: array<u32>;
@group(2) @binding(3) var<storage, read_write> d3: array<u32>;
@group(2) @binding(4) var<storage, read_write> d4: array<u32>;
@group(2) @binding(5) var<storage, read_write> d5: array<u32>;
@group(2) @binding(6) var<storage, read_write> d6: array<u32>;
@group(2) @binding(7) var<storage, read_write> d7: array<u32>;
@group(2) @binding(8) var<storage, read_write> d8: array<u32>;
@group(2) @binding(9) var<storage, read_write> d9: array<u32>;

@compute @workgroup_size(256)
fn scatter(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i >= sp.n_particles {
        return;
    }
    let cell = cell_indices[i];
    let k = cell_offsets[cell] + atomicSub(&cell_fill[cell], 1u) - 1u;
    d0[k] = s0[i];
    d1[k] = s1[i];
    d2[k] = s2[i];
    d3[k] = s3[i];
    d4[k] = s4[i];
    d5[k] = s5[i];
    d6[k] = s6[i];
    d7[k] = s7[i];
    d8[k] = s8[i];
    d9[k] = s9[i];
}
