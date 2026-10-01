//! GPU PCISPH: on-device convergence control and on-device adaptive dt.

#![cfg(feature = "gpu")]

use kernel::sph::AdvectiveDtPolicy;
use kernel::{
    BoundaryParticles, FluidType, GpuKernel, ParticleArrays, SimulationKernel, SolverType,
};

/// A cubic block of water particles (`n`³, given lattice spacing, mass of a
/// rest-density particle at `rest_spacing`) above a boundary floor.
fn block_kernel(n: usize, spacing: f32, rest_spacing: f32) -> Option<GpuKernel> {
    let h = 1.3 * rest_spacing;
    let mass = 1000.0 * rest_spacing.powi(3);
    let domain_min = [0.0_f32; 3];
    let domain_max = [0.04_f32; 3];
    let mut particles = ParticleArrays::new();
    for ix in 0..n {
        for iy in 0..n {
            for iz in 0..n {
                particles.push_particle(
                    0.01 + ix as f32 * spacing,
                    0.004 + iy as f32 * spacing,
                    0.01 + iz as f32 * spacing,
                    mass,
                    1000.0,
                    293.15,
                    FluidType::Water,
                );
            }
        }
    }
    let mut boundary = BoundaryParticles::new();
    let nb = (0.04 / rest_spacing) as usize;
    for ix in 0..=nb {
        for iz in 0..=nb {
            let (x, z) = (ix as f32 * rest_spacing, iz as f32 * rest_spacing);
            boundary.push(x, 0.0, z, mass, 0.0, 1.0, 0.0);
        }
    }
    match GpuKernel::new(
        particles,
        boundary,
        h,
        [0.0, -9.81, 0.0],
        20.0,
        0.4,
        0.001,
        domain_min,
        domain_max,
        SolverType::Pcisph,
    ) {
        Ok(k) => Some(k),
        Err(e) => {
            eprintln!("Skipping GPU PCISPH test: {e}");
            None
        }
    }
}

#[test]
fn gpu_pcisph_iteration_count_follows_convergence() {
    // Dilute block: nothing is over-compressed, so the solve converges at the
    // first check and runs exactly the minimum iterations.
    let Some(mut dilute) = block_kernel(8, 0.0026, 0.002) else {
        return;
    };
    for _ in 0..3 {
        dilute.step(2.0e-4);
        assert_eq!(
            dilute.pcisph_last_iterations(),
            3,
            "dilute block should stop at the minimum"
        );
    }

    // Compressed block: within [min, max] (the solve may need more than the
    // minimum).
    let Some(mut dense) = block_kernel(8, 0.0017, 0.002) else {
        return;
    };
    for _ in 0..3 {
        dense.step(2.0e-4);
        let iters = dense.pcisph_last_iterations();
        assert!(
            (3..=10).contains(&iters),
            "compressed block ran {iters} iterations"
        );
    }
    let p = dense.particles();
    assert!(p.x.iter().chain(&p.vx).all(|v| v.is_finite()));
}

#[test]
fn gpu_pcisph_adaptive_dt_matches_cpu_policy() {
    let Some(mut k) = block_kernel(10, 0.002, 0.002) else {
        return;
    };
    let policy = AdvectiveDtPolicy {
        h: 1.3 * 0.002,
        cfl_number: 0.4,
        safety: 0.85,
        max_growth: 2.0,
        hold_steps: 1.0,
    };

    // Nothing has run yet: no progress to report.
    assert_eq!(k.take_adaptive_progress().steps, 0);

    // Get the block moving (and the bootstrap out of the way) first.
    for _ in 0..5 {
        k.step(1.0e-4);
    }

    // One on-device step chains from the seed exactly like the CPU policy.
    let mut prev = 1.0e-4_f32;
    for _ in 0..4 {
        let expected = policy.next_dt(&k.step_stats(), prev);
        assert!(k.step_adaptive(&policy, prev));
        let progress = k.take_adaptive_progress();
        assert_eq!(progress.steps, 1);
        assert!(
            (progress.last_dt - expected).abs() <= 1.0e-5 * expected,
            "device dt {} vs CPU policy {}",
            progress.last_dt,
            expected
        );
        assert!((progress.sim_time - expected as f64).abs() <= 1.0e-5 * expected as f64);
        prev = progress.last_dt;
    }

    // Several steps per take: counts and sim time accumulate.
    for _ in 0..6 {
        assert!(k.step_adaptive(&policy, prev));
    }
    let progress = k.take_adaptive_progress();
    assert_eq!(progress.steps, 6);
    assert!(progress.last_dt > 0.0 && progress.last_dt <= prev * 2f32.powi(6));
    assert!(progress.sim_time >= progress.last_dt as f64);
    let p = k.particles();
    assert!(p.x.iter().chain(&p.vx).all(|v| v.is_finite()));
}
