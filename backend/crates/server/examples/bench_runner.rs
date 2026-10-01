//! End-to-end throughput benchmark for the server's simulation loop.
//!
//! Drives `SimulationRunner::step_batch` exactly as the WebSocket stepper does
//! (24 ms batches), with an optional 30 fps snapshot thread standing in for the
//! frame builder. Reports steps/s and simulated seconds per wall second.
//!
//!   cargo run --release -p server --example bench_runner -- [throughput|fingerprint] [scenario...]
//!
//! Scenarios: dam25 dam15 dam10 pillar25 pcisph25 (default: dam25 dam15 dam10 pillar25)
//! Env: BENCH_SECS (default 6), BENCH_FRAMES=0 to disable the snapshot thread,
//! FP_STEPS (fingerprint step count, default 3000), FP_TRACE=N (print health
//! every N fingerprint steps), BENCH_BACKEND=cpu (default gpu).

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use orchestrator::config::SimulationConfig;
use server::runner::SimulationRunner;

fn scenario(name: &str) -> SimulationConfig {
    let (spacing, pillar, solver) = match name {
        "dam25" => (0.0025, false, "wcsph"),
        "dam15" => (0.0015, false, "wcsph"),
        "dam10" => (0.0010, false, "wcsph"),
        "pillar25" => (0.0025, true, "wcsph"),
        "pillar15" => (0.0015, true, "wcsph"),
        "pcisph25" => (0.0025, false, "pcisph"),
        _ => panic!("unknown scenario {name}"),
    };
    let mut cfg = serde_json::json!({
        "name": name, "fluid_type": "Water",
        "domain": {"min": [0.0,0.0,0.0], "max": [0.12,0.08,0.06]},
        "fluid_region": {"min": [0.0,0.0,0.0], "max": [0.036,0.06,0.06]},
        "boundary_conditions": {"x_min":"Wall","x_max":"Wall","y_min":"Wall","y_max":"Outflow","z_min":"Wall","z_max":"Wall"},
        "particle_spacing": spacing, "gravity": [0.0,-9.81,0.0],
        "speed_of_sound": 20.0, "viscosity": 0.001, "cfl_number": 0.4,
        "backend": std::env::var("BENCH_BACKEND").unwrap_or_else(|_| "gpu".into()),
        "solver": solver
    });
    if pillar {
        cfg["geometry"] = serde_json::json!({
            "type": "cylinder", "center": [0.075,0.04,0.03],
            "radius": 0.008, "height": 0.08, "axis": "y"
        });
    }
    serde_json::from_value(cfg).unwrap()
}

fn throughput(name: &str, secs: f64, frames: bool) {
    let runner = SimulationRunner::new(scenario(name), std::path::Path::new(".")).unwrap();
    let n = runner.particle_count();
    runner.start();

    let stop = Arc::new(AtomicBool::new(false));
    let snap = frames.then(|| {
        let r = runner.clone();
        let stop = stop.clone();
        std::thread::spawn(move || {
            let mut count = 0u64;
            while !stop.load(Ordering::Relaxed) {
                let p = r.particles();
                std::hint::black_box(&p);
                count += 1;
                std::thread::sleep(Duration::from_millis(33));
            }
            count
        })
    });

    // Warm up (pipeline compile, first reorder, initial transient).
    let warm = Instant::now();
    while warm.elapsed().as_secs_f64() < 1.0 {
        runner.step_batch(Duration::from_millis(24));
        std::thread::sleep(Duration::from_millis(1));
    }

    let t0 = Instant::now();
    let (s0, st0) = (runner.timestep_count(), runner.sim_time());
    while t0.elapsed().as_secs_f64() < secs {
        runner.step_batch(Duration::from_millis(24));
        // Mirrors ws.rs: 1 ms yield so frame builds can grab the kernel lock.
        std::thread::sleep(Duration::from_millis(1));
    }
    let wall = t0.elapsed().as_secs_f64();
    let steps = runner.timestep_count() - s0;
    let sim = runner.sim_time() - st0;
    stop.store(true, Ordering::Relaxed);
    let nframes = snap.map(|h| h.join().unwrap()).unwrap_or(0);

    // Health of the end state, so a throughput number from an exploded run is
    // recognisable as such.
    let p = runner.particles();
    let bad = (0..p.len())
        .filter(|&i| !(p.x[i].is_finite() && p.vx[i].is_finite()))
        .count();
    let st = kernel::StepStats::from_particles(&p);
    println!(
        "{name:<10} n={n:>7}  steps/s={:>8.1}  sim_s/wall_s={:.5}  dt={:.3e}  frames={nframes} status={:?} sim_t={:.3} nonfinite={bad} vmax={:.3} max_rho_var={:.4}",
        steps as f64 / wall,
        sim / wall,
        runner.dt(),
        runner.status(),
        runner.sim_time(),
        st.max_speed,
        st.max_density_variation,
    );
}

/// Deterministic-ish physics fingerprint: fixed step count, dt recomputed with
/// the runner's policy. Compare before/after a change for sanity.
fn fingerprint(name: &str, steps: usize) {
    let config = scenario(name);
    let triangles =
        orchestrator::geometry::resolve_geometry(&config, std::path::Path::new(".")).unwrap();
    let sdf = orchestrator::geometry::generate_sdf(
        &triangles,
        config.domain.min,
        config.domain.max,
        0.5 * config.particle_spacing,
    );
    let (fluid, bdata) = orchestrator::domain::setup_domain(&config, &sdf);
    let mut boundary = kernel::BoundaryParticles::new();
    for b in bdata {
        boundary.push(b.x, b.y, b.z, b.mass, b.nx, b.ny, b.nz);
    }
    let h = config.smoothing_length();
    let mut k = orchestrator::create_kernel(
        &config.backend,
        fluid,
        boundary,
        h,
        config.gravity,
        config.speed_of_sound,
        config.cfl_number,
        config.viscosity,
        config.domain.min,
        config.domain.max,
        config.solver.to_kernel_solver_type(),
    );
    let trace: usize = std::env::var("FP_TRACE").ok().and_then(|s| s.parse().ok()).unwrap_or(0);
    let mut dt = config.cfl_number * h / config.speed_of_sound; // runner's initial dt
    let mut t = 0.0f64;
    let start = Instant::now();
    for s in 0..steps {
        if s % server::runner::dt_recompute_interval(k.solver_type()) as usize == 0 {
            // Same dt policy as the server runner.
            let stats = k.step_stats();
            dt = server::runner::adaptive_dt(
                k.solver_type(),
                &stats,
                dt,
                h,
                config.speed_of_sound,
                config.cfl_number,
            );
        }
        k.step(dt);
        t += dt as f64;
        if trace > 0 && (s + 1) % trace == 0 {
            let p = k.particles();
            let bad = (0..p.len())
                .filter(|&i| !(p.x[i].is_finite() && p.vx[i].is_finite() && p.density[i].is_finite()))
                .count();
            let st = kernel::StepStats::from_particles(p);
            println!(
                "  step {:>5} t={t:.4} dt={dt:.3e} nonfinite={bad} vmax={:.3} amax={:.1} max_rho_var={:.4}",
                s + 1, st.max_speed, st.max_accel, st.max_density_variation
            );
        }
    }
    let wall = start.elapsed().as_secs_f64();
    let p = k.particles();
    let n = p.len() as f64;
    let mean = |v: &[f32]| v.iter().map(|&x| x as f64).sum::<f64>() / n;
    let ke: f64 = (0..p.len())
        .map(|i| {
            0.5 * p.mass[i] as f64
                * ((p.vx[i] * p.vx[i] + p.vy[i] * p.vy[i] + p.vz[i] * p.vz[i]) as f64)
        })
        .sum();
    let vmax = (0..p.len())
        .map(|i| (p.vx[i] * p.vx[i] + p.vy[i] * p.vy[i] + p.vz[i] * p.vz[i]).sqrt())
        .fold(0.0f32, f32::max);
    let m = k.error_metrics();
    // Over-compression (what PCISPH's pressure solve corrects), water rest density.
    let comp: Vec<f64> = p
        .density
        .iter()
        .map(|&r| (r as f64 - 1000.0) / 1000.0)
        .filter(|&e| e > 0.0)
        .collect();
    let comp_mean = comp.iter().sum::<f64>() / comp.len().max(1) as f64;
    let comp_max = comp.iter().cloned().fold(0.0f64, f64::max);
    println!(
        "{name:<10} steps={steps} t={t:.5}s wall={wall:.2}s  mean_pos=({:.6},{:.6},{:.6}) KE={ke:.6e} vmax={vmax:.4} rho_mean={:.3} max_rho_var={:.4} compressed={} comp_mean={comp_mean:.5} comp_max={comp_max:.4}",
        mean(&p.x), mean(&p.y), mean(&p.z), mean(&p.density), m.max_density_variation, comp.len(),
    );
}

/// Kernel-only: fixed dt, compare per-step sync vs. pipelined submission.
fn kernel_only(name: &str) {
    use kernel::SimulationKernel;
    let config = scenario(name);
    let triangles =
        orchestrator::geometry::resolve_geometry(&config, std::path::Path::new(".")).unwrap();
    let sdf = orchestrator::geometry::generate_sdf(
        &triangles,
        config.domain.min,
        config.domain.max,
        0.5 * config.particle_spacing,
    );
    let (fluid, bdata) = orchestrator::domain::setup_domain(&config, &sdf);
    let mut boundary = kernel::BoundaryParticles::new();
    for b in bdata {
        boundary.push(b.x, b.y, b.z, b.mass, b.nx, b.ny, b.nz);
    }
    let h = config.smoothing_length();
    let cs = kernel::sph::auto_tune_speed_of_sound(
        config.gravity,
        config.domain.min,
        config.domain.max,
        config.speed_of_sound,
    );
    let mut k = kernel::GpuKernel::new(
        fluid,
        boundary,
        h,
        config.gravity,
        cs,
        config.cfl_number,
        config.viscosity,
        config.domain.min,
        config.domain.max,
        config.solver.to_kernel_solver_type(),
    )
    .unwrap();
    let pcisph = config.solver.to_kernel_solver_type() == kernel::SolverType::Pcisph;
    // PCISPH: a typical dam-break dt (iteration counts depend on dt).
    let dt = if pcisph { 4.0e-4 } else { 0.85 * config.cfl_number * h / config.speed_of_sound };
    for _ in 0..100 {
        k.step(dt);
    }
    let n = 400;
    let t = Instant::now();
    for _ in 0..n {
        k.step(dt);
    }
    let sync_sps = n as f64 / t.elapsed().as_secs_f64();
    let t = Instant::now();
    for _ in 0..n {
        k.step_no_sync(dt);
    }
    k.sync();
    let async_sps = n as f64 / t.elapsed().as_secs_f64();
    let p = k.particles();
    let bad = (0..p.len()).filter(|&i| !p.x[i].is_finite()).count();
    println!("{name:<10} n={:>7} kernel step(): {sync_sps:>8.1} steps/s   step_no_sync(): {async_sps:>8.1} steps/s  nonfinite={bad}", k.particle_count());
}

fn main() {
    kernel::simulation::init();
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mode = args.first().cloned().unwrap_or_else(|| "throughput".into());
    let mut names: Vec<String> = args.iter().skip(1).cloned().collect();
    if names.is_empty() {
        names = ["dam25", "dam15", "dam10", "pillar25"]
            .iter()
            .map(|s| s.to_string())
            .collect();
    }
    let secs: f64 = std::env::var("BENCH_SECS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(6.0);
    let frames = std::env::var("BENCH_FRAMES")
        .map(|v| v != "0")
        .unwrap_or(true);
    let fp_steps: usize = std::env::var("FP_STEPS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(3000);
    for name in &names {
        match mode.as_str() {
            "throughput" => throughput(name, secs, frames),
            "fingerprint" => fingerprint(name, fp_steps),
            "kernel" => kernel_only(name),
            _ => panic!("mode must be throughput or fingerprint"),
        }
    }
}
