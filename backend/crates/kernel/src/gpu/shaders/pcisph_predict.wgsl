// PCISPH step setup and final integration
//
// Predictive-Corrective Incompressible SPH (Solenthaler & Pajarola 2009).
// The correction loop itself (prediction, density, pressure correction,
// pressure forces) lives in pcisph_solve.wgsl; this module brackets it.
//
// PCISPH state is packed per particle (vec4) so the correction loop's single
// bind group stays within the storage-buffer limit:
//   orig4  = step-start (x, y, z, mass)     (written by build_neighbors)
//   vel4   = step-start velocity
//   np4    = non-pressure acceleration (viscosity + gravity + boundary repulsion)
//   pacc4  = pressure acceleration from the latest correction iteration
// The predicted velocity is always vel4 + (np4 + pacc4) * dt.
//
// Entry points:
// - save_and_init_pcisph:   save velocity / non-pressure acceleration, zero
//                           the pressure acceleration, warm-start pressure
// - final_integrate_pcisph: commit final velocity/position with domain clamping

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
    search_cells: u32,
};

// Group 0: SimParams + positions (read_write)
@group(0) @binding(0) var<uniform> params: SimParams;
@group(0) @binding(1) var<storage, read_write> pos_x: array<f32>;
@group(0) @binding(2) var<storage, read_write> pos_y: array<f32>;
@group(0) @binding(3) var<storage, read_write> pos_z: array<f32>;

// Group 1: Velocity + acceleration (read_write)
@group(1) @binding(0) var<storage, read_write> vel_x: array<f32>;
@group(1) @binding(1) var<storage, read_write> vel_y: array<f32>;
@group(1) @binding(2) var<storage, read_write> vel_z: array<f32>;
@group(1) @binding(3) var<storage, read_write> acc_x: array<f32>;
@group(1) @binding(4) var<storage, read_write> acc_y: array<f32>;
@group(1) @binding(5) var<storage, read_write> acc_z: array<f32>;

// Group 2: pressure + the previous step's final pressure (warm start; the
// pressure buffer itself is zeroed for the non-pressure force pass)
@group(2) @binding(0) var<storage, read_write> pressure: array<f32>;
@group(2) @binding(1) var<storage, read> pressure_prev: array<f32>;

// Group 3: packed PCISPH state + convergence counters
@group(3) @binding(0) var<storage, read_write> orig4: array<vec4<f32>>;
@group(3) @binding(1) var<storage, read_write> vel4: array<vec4<f32>>;
@group(3) @binding(2) var<storage, read_write> np4: array<vec4<f32>>;
@group(3) @binding(3) var<storage, read_write> pacc4: array<vec4<f32>>;
@group(3) @binding(4) var<storage, read_write> convergence: array<atomic<u32>>;

const RESTITUTION: f32 = 0.2;

// ---------------------------------------------------------------------------
// Entry point: Save step-start state and initialize the prediction
// ---------------------------------------------------------------------------
@compute @workgroup_size(256)
fn save_and_init_pcisph(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i >= params.n_particles {
        return;
    }

    // (orig4 was written by build_neighbors from the same positions.)
    vel4[i] = vec4<f32>(vel_x[i], vel_y[i], vel_z[i], 0.0);

    // Non-pressure accelerations (viscosity + gravity + boundary repulsion)
    np4[i] = vec4<f32>(acc_x[i], acc_y[i], acc_z[i], 0.0);

    // No pressure acceleration yet: the first prediction uses v + a_np * dt.
    pacc4[i] = vec4<f32>(0.0);

    // Warm-start pressure: retain 50% of the previous step's final pressure
    // (as the CPU solver does).
    pressure[i] = pressure_prev[i] * 0.5;

    // Reset the per-step correction-iteration counter.
    if i == 0u {
        atomicStore(&convergence[2], 0u);
    }
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
    let v0 = vel4[i];
    let a_np = np4[i];
    let a_p = pacc4[i];
    let o = orig4[i];

    // Total acceleration = non-pressure + pressure
    let total_ax = a_np.x + a_p.x;
    let total_ay = a_np.y + a_p.y;
    let total_az = a_np.z + a_p.z;

    // Commit the final predicted velocity
    var vx = v0.x + total_ax * dt;
    var vy = v0.y + total_ay * dt;
    var vz = v0.z + total_az * dt;

    // Final position from original + final velocity * dt
    var px = o.x + vx * dt;
    var py = o.y + vy * dt;
    var pz = o.z + vz * dt;

    // Domain clamping with velocity reflection
    if px < params.domain_min_x {
        px = params.domain_min_x;
        if vx < 0.0 { vx = -RESTITUTION * vx; }
    }
    if px > params.domain_max_x {
        px = params.domain_max_x;
        if vx > 0.0 { vx = -RESTITUTION * vx; }
    }
    if py < params.domain_min_y {
        py = params.domain_min_y;
        if vy < 0.0 { vy = -RESTITUTION * vy; }
    }
    if py > params.domain_max_y {
        py = params.domain_max_y;
        if vy > 0.0 { vy = -RESTITUTION * vy; }
    }
    if pz < params.domain_min_z {
        pz = params.domain_min_z;
        if vz < 0.0 { vz = -RESTITUTION * vz; }
    }
    if pz > params.domain_max_z {
        pz = params.domain_max_z;
        if vz > 0.0 { vz = -RESTITUTION * vz; }
    }

    pos_x[i] = px;
    pos_y[i] = py;
    pos_z[i] = pz;
    vel_x[i] = vx;
    vel_y[i] = vy;
    vel_z[i] = vz;

    // Store total acceleration for adaptive timestep computation
    acc_x[i] = total_ax;
    acc_y[i] = total_ay;
    acc_z[i] = total_az;
}
