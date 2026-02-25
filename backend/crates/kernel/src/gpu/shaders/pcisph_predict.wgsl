// PCISPH prediction and correction compute shader
//
// Predictive-Corrective Incompressible SPH (Solenthaler & Pajarola 2009).
// Iteratively corrects pressure to enforce density constraints.
//
// Entry points:
// - save_and_init_pcisph:      save original state, initialize prediction
// - predict_positions_pcisph:  predict positions from predicted velocities
// - correct_pressure_pcisph:   correct pressure from density error
// - update_pred_vel_pcisph:    update predicted velocity with pressure acceleration
// - final_integrate_pcisph:    commit final velocity/position with domain clamping
// - clear_convergence:         zero out convergence counters

const PI: f32 = 3.14159265358979323846;
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
    _pad1: u32,
};

// Group 0: SimParams + positions (read_write) + mass
@group(0) @binding(0) var<uniform> params: SimParams;
@group(0) @binding(1) var<storage, read_write> pos_x: array<f32>;
@group(0) @binding(2) var<storage, read_write> pos_y: array<f32>;
@group(0) @binding(3) var<storage, read_write> pos_z: array<f32>;
@group(0) @binding(4) var<storage, read> mass_packed: array<u32>;

// Group 1: Velocity (read_write for final integration) + acceleration (read_write)
@group(1) @binding(0) var<storage, read_write> vel_x: array<f32>;
@group(1) @binding(1) var<storage, read_write> vel_y: array<f32>;
@group(1) @binding(2) var<storage, read_write> vel_z: array<f32>;
@group(1) @binding(3) var<storage, read_write> acc_x: array<f32>;
@group(1) @binding(4) var<storage, read_write> acc_y: array<f32>;
@group(1) @binding(5) var<storage, read_write> acc_z: array<f32>;

// Group 2: SPH state (PCISPH-specific: fewer bindings than WCSPH)
@group(2) @binding(0) var<storage, read> density: array<f32>;
@group(2) @binding(1) var<storage, read_write> pressure: array<f32>;
@group(2) @binding(2) var<storage, read> fluid_type: array<u32>;

// Group 3: PCISPH state buffers
@group(3) @binding(0) var<storage, read_write> orig_pos_x: array<f32>;
@group(3) @binding(1) var<storage, read_write> orig_pos_y: array<f32>;
@group(3) @binding(2) var<storage, read_write> orig_pos_z: array<f32>;
@group(3) @binding(3) var<storage, read_write> pred_vel_x: array<f32>;
@group(3) @binding(4) var<storage, read_write> pred_vel_y: array<f32>;
@group(3) @binding(5) var<storage, read_write> pred_vel_z: array<f32>;
@group(3) @binding(6) var<storage, read_write> np_acc_x: array<f32>;
@group(3) @binding(7) var<storage, read_write> np_acc_y: array<f32>;
@group(3) @binding(8) var<storage, read_write> np_acc_z: array<f32>;
@group(3) @binding(9) var<storage, read_write> pcisph_delta: array<f32>;
@group(3) @binding(10) var<storage, read_write> convergence: array<atomic<u32>>;

fn read_mass(idx: u32) -> f32 {
    let pair = unpack2x16float(mass_packed[idx >> 1u]);
    return pair[idx & 1u];
}

const RESTITUTION: f32 = 0.2;

// ---------------------------------------------------------------------------
// Entry point: Save original state and initialize PCISPH prediction
// ---------------------------------------------------------------------------
@compute @workgroup_size(256)
fn save_and_init_pcisph(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i >= params.n_particles {
        return;
    }

    let dt = params.dt;

    // Save original positions
    orig_pos_x[i] = pos_x[i];
    orig_pos_y[i] = pos_y[i];
    orig_pos_z[i] = pos_z[i];

    // Store non-pressure accelerations (viscosity + gravity + boundary repulsion)
    np_acc_x[i] = acc_x[i];
    np_acc_y[i] = acc_y[i];
    np_acc_z[i] = acc_z[i];

    // Initialize predicted velocity: v* = v + a_np * dt
    pred_vel_x[i] = vel_x[i] + acc_x[i] * dt;
    pred_vel_y[i] = vel_y[i] + acc_y[i] * dt;
    pred_vel_z[i] = vel_z[i] + acc_z[i] * dt;

    // Warm-start pressure: retain 50% from previous step
    pressure[i] = pressure[i] * 0.5;
}

// ---------------------------------------------------------------------------
// Entry point: Predict positions from predicted velocities
// ---------------------------------------------------------------------------
@compute @workgroup_size(256)
fn predict_positions_pcisph(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i >= params.n_particles {
        return;
    }

    let dt = params.dt;

    pos_x[i] = orig_pos_x[i] + pred_vel_x[i] * dt;
    pos_y[i] = orig_pos_y[i] + pred_vel_y[i] * dt;
    pos_z[i] = orig_pos_z[i] + pred_vel_z[i] * dt;
}

// ---------------------------------------------------------------------------
// Entry point: Correct pressure from predicted density error
// ---------------------------------------------------------------------------
@compute @workgroup_size(256)
fn correct_pressure_pcisph(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i >= params.n_particles {
        return;
    }

    // Determine rest density from fluid type
    let ft = fluid_type[i];
    var rho0 = WATER_REST_DENSITY;
    if ft != 0u {
        rho0 = AIR_REST_DENSITY;
    }

    // Compute relative density error
    let density_error = (density[i] - rho0) / rho0;

    // Pressure correction: only correct over-compression (clamp error >= 0).
    // pcisph_delta stores the dt-independent base; divide by dt² for actual scaling.
    let dt = params.dt;
    let effective_delta = pcisph_delta[i] / (dt * dt);
    let correction = effective_delta * max(density_error, 0.0);
    pressure[i] = max(pressure[i] + correction, 0.0);

    // Accumulate convergence info for over-compressed particles
    if density_error > 0.0 {
        // Fixed-point: multiply by 1e6 and cast to u32 for atomic add
        atomicAdd(&convergence[0], u32(density_error * 1000000.0));
        // Count of over-compressed particles
        atomicAdd(&convergence[1], 1u);
    }
}

// ---------------------------------------------------------------------------
// Entry point: Update predicted velocity with pressure acceleration
// ---------------------------------------------------------------------------
@compute @workgroup_size(256)
fn update_pred_vel_pcisph(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i >= params.n_particles {
        return;
    }

    let dt = params.dt;

    // acc_x/y/z hold pressure acceleration at this point
    // Predicted velocity = initial velocity + (non-pressure + pressure) acceleration * dt
    pred_vel_x[i] = vel_x[i] + (np_acc_x[i] + acc_x[i]) * dt;
    pred_vel_y[i] = vel_y[i] + (np_acc_y[i] + acc_y[i]) * dt;
    pred_vel_z[i] = vel_z[i] + (np_acc_z[i] + acc_z[i]) * dt;
}

// ---------------------------------------------------------------------------
// Entry point: Final integration — commit velocity, position, total acceleration
// ---------------------------------------------------------------------------
@compute @workgroup_size(256)
fn final_integrate_pcisph(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i >= params.n_particles {
        return;
    }

    let dt = params.dt;

    // Total acceleration = non-pressure + pressure (acc currently holds pressure acc)
    let total_ax = np_acc_x[i] + acc_x[i];
    let total_ay = np_acc_y[i] + acc_y[i];
    let total_az = np_acc_z[i] + acc_z[i];

    // Commit final velocity from prediction
    vel_x[i] = pred_vel_x[i];
    vel_y[i] = pred_vel_y[i];
    vel_z[i] = pred_vel_z[i];

    // Final position from original + final velocity * dt
    pos_x[i] = orig_pos_x[i] + vel_x[i] * dt;
    pos_y[i] = orig_pos_y[i] + vel_y[i] * dt;
    pos_z[i] = orig_pos_z[i] + vel_z[i] * dt;

    // Store total acceleration for adaptive timestep computation
    acc_x[i] = total_ax;
    acc_y[i] = total_ay;
    acc_z[i] = total_az;

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

// ---------------------------------------------------------------------------
// Entry point: Clear convergence counters (dispatch with 1 thread)
// ---------------------------------------------------------------------------
@compute @workgroup_size(1)
fn clear_convergence(@builtin(global_invocation_id) gid: vec3<u32>) {
    atomicStore(&convergence[0], 0u);
    atomicStore(&convergence[1], 0u);
}
