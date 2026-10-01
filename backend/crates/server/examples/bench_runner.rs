//! End-to-end throughput benchmark for the server's simulation loop.
//!
//! Drives `SimulationRunner::step_batch` exactly as the WebSocket stepper does
//! (24 ms batches), with an optional 30 fps snapshot thread standing in for the
//! frame builder. Reports steps/s and simulated seconds per wall second.
//!
//!   cargo run --release -p server --example bench_runner -- [mode] [scenario...]
//!
//! Modes:
//!   throughput  steps/s and sim_s/wall_s over BENCH_SECS of wall time
//!   quality     run to BENCH_SIMT simulated seconds through the runner's own
//!               dt logic; print a density-deviation / dam-front time series
//!   fingerprint fixed step count physics fingerprint
//!   kernel      kernel-only step() vs step_no_sync() rate
//!
//! Scenarios: dam25 dam15 dam10 pillar25 pcisph25 (default: dam25 dam15 dam10 pillar25)
//! Env: BENCH_SECS (default 6), BENCH_FRAMES=0 to disable the snapshot thread,
//!      BENCH_SIMT (quality mode, default 0.4), BENCH_CFL to override cfl_number.

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
        "backend": "gpu", "solver": solver
    });
    if let Some(cfl) = std::env::var("BENCH_CFL").ok().and_then(|s| s.parse::<f64>().ok()) {
        cfg["cfl_number"] = serde_json::json!(cfl);
    }
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

    println!(
        "{name:<10} n={n:>7}  steps/s={:>8.1}  sim_s/wall_s={:.5}  dt={:.3e}  frames={nframes} status={:?}",
        steps as f64 / wall,
        sim / wall,
        runner.dt(),
        runner.status(),
    );
}

/// Water density deviation statistics from a particle snapshot:
/// (max |rho/rho0 - 1|, max compression rho/rho0 - 1, 99th-percentile compression).
///
/// Free-surface particles are always under-dense with summation density, so
/// the compression side is what reflects EOS stiffness and time integration.
fn density_stats(p: &kernel::ParticleArrays) -> (f32, f32, f32) {
    let rho0 = kernel::eos::WATER_REST_DENSITY;
    let mut max_abs = 0.0f32;
    let mut comp: Vec<f32> = Vec::with_capacity(p.len());
    for &rho in &p.density {
        let d = rho / rho0 - 1.0;
        max_abs = max_abs.max(d.abs());
        comp.push(d.max(0.0));
    }
    comp.sort_unstable_by(|a, b| a.total_cmp(b));
    let max_c = comp.last().copied().unwrap_or(0.0);
    let p99 = comp.get(comp.len() * 99 / 100).copied().unwrap_or(0.0);
    (max_abs, max_c, p99)
}

/// Physics quality through the runner's own adaptive-dt loop: run to
/// `sim_end` simulated seconds, sampling density deviation and the dam-break
/// front position after every batch.
fn quality(name: &str, sim_end: f64) {
    let runner = SimulationRunner::new(scenario(name), std::path::Path::new(".")).unwrap();
    runner.start();
    let x0 = runner.particles().x.iter().copied().fold(0.0f32, f32::max);

    let start = Instant::now();
    let mut stepping = Duration::ZERO;
    let mut next_print = 0.0f64;
    let (mut peak_abs, mut peak_c, mut peak_p99) = (0.0f32, 0.0f32, 0.0f32);
    let (mut sum_c, mut sum_p99, mut sum_dt, mut samples) = (0.0f64, 0.0f64, 0.0f64, 0u32);
    let mut t_front: [Option<f64>; 2] = [None, None];
    println!("{name}: t[s]    dt         max|dev|  max_comp  p99_comp  front_x[m]");
    while runner.sim_time() < sim_end {
        let t = Instant::now();
        runner.step_batch(Duration::from_millis(24));
        stepping += t.elapsed();
        if runner.status() != server::state::SimStatus::Running {
            break;
        }
        let p = runner.particles();
        let (abs, c, p99) = density_stats(&p);
        let front = p.x.iter().copied().fold(0.0f32, f32::max);
        let sim_t = runner.sim_time();
        // Surge-front arrival times (resolution: one 24 ms batch).
        if front >= 0.06 && t_front[0].is_none() {
            t_front[0] = Some(sim_t);
        }
        if front >= 0.09 && t_front[1].is_none() {
            t_front[1] = Some(sim_t);
        }
        // Skip the first 20 ms (initial settling transient) in the aggregates.
        if sim_t > 0.02 {
            peak_abs = peak_abs.max(abs);
            peak_c = peak_c.max(c);
            peak_p99 = peak_p99.max(p99);
            sum_c += c as f64;
            sum_p99 += p99 as f64;
            sum_dt += runner.dt() as f64;
            samples += 1;
        }
        if sim_t >= next_print {
            println!(
                "{name}: {sim_t:.4}  {:.3e}  {abs:.4}    {c:.4}    {p99:.4}    {front:.4}",
                runner.dt()
            );
            next_print += 0.05;
        }
    }
    let n = samples.max(1) as f64;
    println!(
        "{name} SUMMARY t={:.4} steps={} status={:?} sim_s/stepping_wall_s={:.5} wall={:.1}s \
         mean_dt={:.3e} peak|dev|={peak_abs:.4} peak_comp={peak_c:.4} mean_comp={:.4} \
         peak_p99={peak_p99:.4} mean_p99={:.4} front_x0={x0:.4} t(front>=0.06)={:.4} \
         t(front>=0.09)={:.4}",
        runner.sim_time(),
        runner.timestep_count(),
        runner.status(),
        runner.sim_time() / stepping.as_secs_f64(),
        start.elapsed().as_secs_f64(),
        sum_dt / n,
        sum_c / n,
        sum_p99 / n,
        t_front[0].unwrap_or(f64::NAN),
        t_front[1].unwrap_or(f64::NAN),
    );
}

/// Deterministic-ish physics fingerprint: fixed step count, dt recomputed every
/// 16 steps like the runner. Compare before/after a change for sanity.
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
    let cs = config.effective_speed_of_sound();
    let mut k = orchestrator::create_kernel(
        &config.backend,
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
    );
    let mut dt = 0.0;
    let mut t = 0.0f64;
    let start = Instant::now();
    for s in 0..steps {
        if s % 16 == 0 {
            dt = 0.85
                * kernel::sph::compute_timestep(
                    k.particles(),
                    h,
                    cs,
                    config.cfl_number,
                );
        }
        k.step(dt);
        t += dt as f64;
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
    println!(
        "{name:<10} steps={steps} t={t:.5}s wall={wall:.2}s  mean_pos=({:.6},{:.6},{:.6}) KE={ke:.6e} vmax={vmax:.4} rho_mean={:.3} max_rho_var={:.4}",
        mean(&p.x), mean(&p.y), mean(&p.z), mean(&p.density), m.max_density_variation,
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
    let cs = config.effective_speed_of_sound();
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
    let dt = 0.85 * config.cfl_number * h / config.speed_of_sound;
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
    let _ = k.particles();
    println!("{name:<10} n={:>7} kernel step(): {sync_sps:>8.1} steps/s   step_no_sync(): {async_sps:>8.1} steps/s", k.particle_count());
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
    for name in &names {
        match mode.as_str() {
            "throughput" => throughput(name, secs, frames),
            "fingerprint" => fingerprint(name, 3000),
            "kernel" => kernel_only(name),
            "quality" => quality(
                name,
                std::env::var("BENCH_SIMT")
                    .ok()
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(0.4),
            ),
            _ => panic!("mode must be throughput, quality, fingerprint or kernel"),
        }
    }
}
