// PCISPH on-device convergence check.
//
// Runs as a single thread after a correction iteration's correct_pressure
// pass. If the iteration converged, it zeroes the indirect dispatch args used
// by the remaining iterations, so they dispatch no work. This keeps the whole
// correction loop in one submission (no per-iteration CPU readback) while
// preserving the CPU loop's semantics: break after the first iteration (at or
// past the minimum) whose mean over-compression is below the tolerance.
//
// The pass that does not converge leaves the args untouched. Once zeroed they
// stay zero: the following clear_convergence/correct_pressure passes are
// skipped, so the counters keep the converged iteration's values.

const TOLERANCE_FIXED: f32 = 10000.0; // 0.01 mean relative error × 1e6 fixed point

@group(0) @binding(0) var<storage, read_write> convergence: array<u32>; // [sum, count, iterations, _]
@group(0) @binding(1) var<storage, read_write> args: array<u32>;        // 3 × (x, y, z)

@compute @workgroup_size(1)
fn check_convergence() {
    let sum = convergence[0];
    let count = convergence[1];
    // mean_error = (sum / 1e6) / count, and 0 when nothing is over-compressed.
    if count == 0u || f32(sum) < TOLERANCE_FIXED * f32(count) {
        for (var k = 0u; k < 9u; k = k + 1u) {
            args[k] = 0u;
        }
    }
}
