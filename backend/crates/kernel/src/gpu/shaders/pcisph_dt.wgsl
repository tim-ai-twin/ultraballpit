// On-device adaptive timestep for PCISPH (SimulationKernel::step_adaptive).
//
// Single thread, run at the start of a step right after stats.wgsl reduced the
// current state. Mirrors sph::AdvectiveDtPolicy::next_dt and
// sph::timestep_advective_from_stats exactly (same f32 operations), then
// records the chosen dt; the host copies it into SimParams.dt before the
// step's passes run.

struct Policy {
    h: f32,
    cfl_number: f32,
    safety: f32,
    max_growth: f32,
    hold_steps: f32,
    v_floor: f32,
    min_dt: f32,
    max_dt: f32,
};

// stats.wgsl output: [max |v|^2 bits, max |a|^2 bits, max density deviation bits, non-finite]
@group(0) @binding(0) var<storage, read> stats: array<u32>;
// [dt bits, accumulated sim time bits, step count, _]; dt holds the previous
// step's dt on entry.
@group(0) @binding(1) var<storage, read_write> state: array<u32>;
@group(0) @binding(2) var<uniform> policy: Policy;

@compute @workgroup_size(1)
fn choose_dt() {
    let max_speed = sqrt(bitcast<f32>(stats[0]));
    let max_accel = sqrt(bitcast<f32>(stats[1]));
    let prev_dt = bitcast<f32>(state[0]);

    // AdvectiveDtPolicy::next_dt: CFL velocity reachable within the hold.
    let reach = max_speed + max_accel * policy.hold_steps * prev_dt;

    // timestep_advective_from_stats
    let dt_cfl = policy.cfl_number * policy.h / max(reach, policy.v_floor);
    var dt_force = policy.max_dt;
    if max_accel > 1.0e-12 {
        dt_force = 0.25 * sqrt(policy.h / max_accel);
    }
    let dt_adv = clamp(min(dt_cfl, dt_force), policy.min_dt, policy.max_dt);

    let dt = min(policy.safety * dt_adv, policy.max_growth * prev_dt);

    state[0] = bitcast<u32>(dt);
    state[1] = bitcast<u32>(bitcast<f32>(state[1]) + dt);
    state[2] = state[2] + 1u;
}
