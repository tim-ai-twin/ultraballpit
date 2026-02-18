// Velocity Verlet time integration compute shader
//
// Implements kick-drift-kick scheme:
// - half_kick: v += a * dt/2
// - drift:     x += v * dt, then clamp to domain bounds
// - full_kick (second half): v += a * dt/2 (reuses half_kick after force recompute)
//
// Domain clamping with velocity reflection for no-penetration boundaries.

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
    _pad1: u32,
};

// Group 0: SimParams + positions (read-write for integrate)
@group(0) @binding(0) var<uniform> params: SimParams;
@group(0) @binding(1) var<storage, read_write> pos_x: array<f32>;
@group(0) @binding(2) var<storage, read_write> pos_y: array<f32>;
@group(0) @binding(3) var<storage, read_write> pos_z: array<f32>;

// Group 1: Velocity + acceleration
@group(1) @binding(0) var<storage, read_write> vel_x: array<f32>;
@group(1) @binding(1) var<storage, read_write> vel_y: array<f32>;
@group(1) @binding(2) var<storage, read_write> vel_z: array<f32>;
@group(1) @binding(3) var<storage, read> acc_x: array<f32>;
@group(1) @binding(4) var<storage, read> acc_y: array<f32>;
@group(1) @binding(5) var<storage, read> acc_z: array<f32>;

const RESTITUTION: f32 = 0.2;

// Entry point: Half-kick (v += a * dt/2)
@compute @workgroup_size(256)
fn half_kick(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i >= params.n_particles {
        return;
    }

    let half_dt = 0.5 * params.dt;
    vel_x[i] = vel_x[i] + acc_x[i] * half_dt;
    vel_y[i] = vel_y[i] + acc_y[i] * half_dt;
    vel_z[i] = vel_z[i] + acc_z[i] * half_dt;
}

// Entry point: Drift with XSPH correction (x += (v + dv_xsph) * dt) + domain clamping.
// Before this pass, the XSPH pass writes corrections to acc_x/y/z.
@compute @workgroup_size(256)
fn drift(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i >= params.n_particles {
        return;
    }

    let dt = params.dt;

    // Drift with XSPH velocity smoothing: acc holds XSPH correction
    pos_x[i] = pos_x[i] + (vel_x[i] + acc_x[i]) * dt;
    pos_y[i] = pos_y[i] + (vel_y[i] + acc_y[i]) * dt;
    pos_z[i] = pos_z[i] + (vel_z[i] + acc_z[i]) * dt;

    // Domain clamping with velocity reflection
    // X-min
    if pos_x[i] < params.domain_min_x {
        pos_x[i] = params.domain_min_x;
        if vel_x[i] < 0.0 {
            vel_x[i] = -RESTITUTION * vel_x[i];
        }
    }
    // X-max
    if pos_x[i] > params.domain_max_x {
        pos_x[i] = params.domain_max_x;
        if vel_x[i] > 0.0 {
            vel_x[i] = -RESTITUTION * vel_x[i];
        }
    }
    // Y-min
    if pos_y[i] < params.domain_min_y {
        pos_y[i] = params.domain_min_y;
        if vel_y[i] < 0.0 {
            vel_y[i] = -RESTITUTION * vel_y[i];
        }
    }
    // Y-max
    if pos_y[i] > params.domain_max_y {
        pos_y[i] = params.domain_max_y;
        if vel_y[i] > 0.0 {
            vel_y[i] = -RESTITUTION * vel_y[i];
        }
    }
    // Z-min
    if pos_z[i] < params.domain_min_z {
        pos_z[i] = params.domain_min_z;
        if vel_z[i] < 0.0 {
            vel_z[i] = -RESTITUTION * vel_z[i];
        }
    }
    // Z-max
    if pos_z[i] > params.domain_max_z {
        pos_z[i] = params.domain_max_z;
        if vel_z[i] > 0.0 {
            vel_z[i] = -RESTITUTION * vel_z[i];
        }
    }
}
