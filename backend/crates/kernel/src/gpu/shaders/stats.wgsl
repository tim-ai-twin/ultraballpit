// Whole-system maxima for adaptive timestepping and health checks.
//
// Produces 4 u32s via atomicMax:
//   out[0] = bits(max |v|^2), out[1] = bits(max |a|^2),
//   out[2] = bits(max |rho - rho0| / rho0), out[3] = 1 if any pos/vel is non-finite.
// For non-negative floats, IEEE-754 bit patterns order the same as the values,
// so a u32 max is a float max. Non-finite terms are excluded from the maxima.

const WATER_REST_DENSITY: f32 = 1000.0;
const AIR_REST_DENSITY: f32 = 1.204;
const WG: u32 = 256u;

struct StatsParams {
    n_particles: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
};

@group(0) @binding(0) var<uniform> sp: StatsParams;
@group(0) @binding(1) var<storage, read> pos_x: array<f32>;
@group(0) @binding(2) var<storage, read> pos_y: array<f32>;
@group(0) @binding(3) var<storage, read> pos_z: array<f32>;
@group(0) @binding(4) var<storage, read> vel_x: array<f32>;
@group(0) @binding(5) var<storage, read> vel_y: array<f32>;
@group(0) @binding(6) var<storage, read> vel_z: array<f32>;
@group(0) @binding(7) var<storage, read> acc_x: array<f32>;
@group(0) @binding(8) var<storage, read> acc_y: array<f32>;
@group(0) @binding(9) var<storage, read> acc_z: array<f32>;
@group(0) @binding(10) var<storage, read> density: array<f32>;
@group(0) @binding(11) var<storage, read> fluid_type: array<u32>;
@group(0) @binding(12) var<storage, read_write> out: array<atomic<u32>, 4>;

var<workgroup> s_v: array<u32, WG>;
var<workgroup> s_a: array<u32, WG>;
var<workgroup> s_r: array<u32, WG>;
var<workgroup> s_n: array<u32, WG>;

fn finite_bits(x: f32) -> u32 {
    let b = bitcast<u32>(x) & 0x7fffffffu;
    return select(b, 0u, b >= 0x7f800000u);
}

fn is_nonfinite(x: f32) -> bool {
    return (bitcast<u32>(x) & 0x7fffffffu) >= 0x7f800000u;
}

@compute @workgroup_size(256)
fn reduce_stats(@builtin(global_invocation_id) gid: vec3<u32>,
                @builtin(local_invocation_id) lid: vec3<u32>) {
    let i = gid.x;
    let t = lid.x;
    var v = 0u;
    var a = 0u;
    var r = 0u;
    var nf = 0u;
    if i < sp.n_particles {
        let vx = vel_x[i];
        let vy = vel_y[i];
        let vz = vel_z[i];
        let ax = acc_x[i];
        let ay = acc_y[i];
        let az = acc_z[i];
        v = finite_bits(vx * vx + vy * vy + vz * vz);
        a = finite_bits(ax * ax + ay * ay + az * az);
        let rest = select(AIR_REST_DENSITY, WATER_REST_DENSITY, fluid_type[i] == 0u);
        r = finite_bits(abs(density[i] - rest) / rest);
        if is_nonfinite(pos_x[i]) || is_nonfinite(pos_y[i]) || is_nonfinite(pos_z[i])
            || is_nonfinite(vx) || is_nonfinite(vy) || is_nonfinite(vz) {
            nf = 1u;
        }
    }
    s_v[t] = v;
    s_a[t] = a;
    s_r[t] = r;
    s_n[t] = nf;
    workgroupBarrier();
    for (var stride = WG / 2u; stride > 0u; stride = stride / 2u) {
        if t < stride {
            s_v[t] = max(s_v[t], s_v[t + stride]);
            s_a[t] = max(s_a[t], s_a[t + stride]);
            s_r[t] = max(s_r[t], s_r[t + stride]);
            s_n[t] = max(s_n[t], s_n[t + stride]);
        }
        workgroupBarrier();
    }
    if t == 0u {
        atomicMax(&out[0], s_v[0]);
        atomicMax(&out[1], s_a[0]);
        atomicMax(&out[2], s_r[0]);
        atomicMax(&out[3], s_n[0]);
    }
}
