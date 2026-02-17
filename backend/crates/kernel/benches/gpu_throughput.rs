//! GPU throughput benchmark (T104)
//!
//! Measures timesteps/second for various particle counts on GPU vs CPU.
//! Includes per-pass GPU profiling breakdown and async-batching comparison.
//!
//! Run with: cargo bench --features gpu -p kernel --bench gpu_throughput

#![cfg(feature = "gpu")]

use std::time::Instant;

use kernel::{
    BoundaryParticles, CpuKernel, FluidType, GpuKernel, GpuStepProfile, ParticleArrays,
    SimulationKernel,
};

/// Create a particle cube with the given number of particles (approximate).
fn create_particle_cube(target_count: usize) -> (ParticleArrays, BoundaryParticles, f32) {
    let domain_size = 0.1_f32;
    let n_per_axis = (target_count as f32).cbrt().ceil() as usize;
    let spacing = domain_size / n_per_axis as f32;
    let h = 1.3 * spacing;

    let rest_density = 1000.0_f32;
    let particle_volume = spacing * spacing * spacing;
    let particle_mass = rest_density * particle_volume;

    let mut particles = ParticleArrays::new();
    for ix in 0..n_per_axis {
        for iy in 0..n_per_axis {
            for iz in 0..n_per_axis {
                let x = (ix as f32 + 0.5) * spacing;
                let y = (iy as f32 + 0.5) * spacing;
                let z = (iz as f32 + 0.5) * spacing;
                particles.push_particle(x, y, z, particle_mass, rest_density, 293.15, FluidType::Water);
            }
        }
    }

    let boundary = BoundaryParticles::new();
    (particles, boundary, h)
}

fn benchmark_cpu(n_particles: usize, n_steps: usize) -> f64 {
    let (particles, boundary, h) = create_particle_cube(n_particles);
    let actual_n = particles.len();
    let domain_min = [0.0_f32; 3];
    let domain_max = [0.1_f32; 3];

    let mut kernel = CpuKernel::new(
        particles, boundary, h,
        [0.0, -9.81, 0.0], 10.0, 0.4, 0.001,
        domain_min, domain_max,
    );

    let dt = 1e-5_f32;
    let start = Instant::now();
    for _ in 0..n_steps {
        kernel.step(dt);
    }
    let elapsed = start.elapsed().as_secs_f64();
    let steps_per_sec = n_steps as f64 / elapsed;
    println!("  CPU      {:>6} particles  {:>6} steps  {:.3}s  {:>8.1} steps/s",
        actual_n, n_steps, elapsed, steps_per_sec);
    steps_per_sec
}

struct GpuResult {
    batched_sps: f64,
    async_sps: f64,
    profile: GpuStepProfile,
    profile_steps: usize,
}

fn benchmark_gpu(n_particles: usize, n_steps: usize) -> Option<GpuResult> {
    let (particles, boundary, h) = create_particle_cube(n_particles);
    let actual_n = particles.len();
    let domain_min = [0.0_f32; 3];
    let domain_max = [0.1_f32; 3];

    let mut kernel = match GpuKernel::new(
        particles, boundary, h,
        [0.0, -9.81, 0.0], 10.0, 0.4, 0.001,
        domain_min, domain_max,
    ) {
        Ok(k) => k,
        Err(e) => {
            eprintln!("GPU unavailable: {e}");
            return None;
        }
    };

    let dt = 1e-5_f32;

    // Warmup
    for _ in 0..5 {
        kernel.step(dt);
    }

    // Batched (sync per step) -- this is the production path
    let start = Instant::now();
    for _ in 0..n_steps {
        kernel.step(dt);
    }
    let batched_sps = n_steps as f64 / start.elapsed().as_secs_f64();
    println!("  GPU bat  {:>6} particles  {:>6} steps  {:.3}s  {:>8.1} steps/s",
        actual_n, n_steps, start.elapsed().as_secs_f64(), batched_sps);

    // Async (submit all, sync once)
    let start = Instant::now();
    for _ in 0..n_steps {
        kernel.step_no_sync(dt);
    }
    kernel.sync();
    let async_sps = n_steps as f64 / start.elapsed().as_secs_f64();
    println!("  GPU async{:>6} particles  {:>6} steps  {:.3}s  {:>8.1} steps/s",
        actual_n, n_steps, start.elapsed().as_secs_f64(), async_sps);

    // Profiled (fewer steps -- this is slow)
    let profile_steps = (n_steps / 2).max(3);
    let mut accum = GpuStepProfile::default();
    for _ in 0..profile_steps {
        let p = kernel.step_profiled(dt);
        accum.grid_build_us += p.grid_build_us;
        accum.density_us += p.density_us;
        accum.boundary_pressure_us += p.boundary_pressure_us;
        accum.forces_us += p.forces_us;
        accum.integrate_us += p.integrate_us;
        accum.readback_us += p.readback_us;
        accum.total_us += p.total_us;
    }

    Some(GpuResult { batched_sps, async_sps, profile: accum, profile_steps })
}

fn print_profile_breakdown(profile: &GpuStepProfile, n_steps: usize) {
    let total = profile.total_us as f64;
    if total == 0.0 { return; }

    let pct = |v: u64| -> f64 { 100.0 * v as f64 / total };
    let avg = |v: u64| -> f64 { v as f64 / n_steps as f64 };

    println!("  ┌─────────────────────────┬──────────┬────────┬───────────┐");
    println!("  │ Pass                    │ Total ms │    %   │  Avg µs   │");
    println!("  ├─────────────────────────┼──────────┼────────┼───────────┤");
    for (label, val) in [
        ("Grid build             ", profile.grid_build_us),
        ("Density + EOS          ", profile.density_us),
        ("Boundary pressure      ", profile.boundary_pressure_us),
        ("Forces                 ", profile.forces_us),
        ("Integrate (kick+drift) ", profile.integrate_us),
        ("Readback               ", profile.readback_us),
    ] {
        println!("  │ {} │ {:>8.1} │ {:>5.1}% │ {:>9.0} │",
            label, val as f64 / 1000.0, pct(val), avg(val));
    }
    println!("  ├─────────────────────────┼──────────┼────────┼───────────┤");
    println!("  │ Total                   │ {:>8.1} │ 100.0% │ {:>9.0} │",
        total / 1000.0, avg(profile.total_us));
    println!("  └─────────────────────────┴──────────┴────────┴───────────┘");
}

fn main() {
    println!("=== GPU Throughput Benchmark (T104) ===\n");

    let configs = [
        (1_000, 100),
        (8_000, 40),
        (27_000, 20),
        (64_000, 10),
    ];

    let mut results = Vec::new();

    for &(n, steps) in &configs {
        println!("--- {} particles, {} steps ---", n, steps);
        let cpu_rate = benchmark_cpu(n, steps);
        if let Some(gpu) = benchmark_gpu(n, steps) {
            println!();
            print_profile_breakdown(&gpu.profile, gpu.profile_steps);
            println!();
            results.push((n, cpu_rate, gpu.batched_sps, gpu.async_sps));
        } else {
            println!("  GPU not available\n");
        }
    }

    println!("\n=== Summary ===");
    println!("{:>10} {:>10} {:>10} {:>10} {:>10} {:>10}",
        "Particles", "CPU", "GPU bat", "GPU async", "bat/CPU", "async/bat");
    for &(n, cpu, batched, async_r) in &results {
        println!("{:>10} {:>10.1} {:>10.1} {:>10.1} {:>10.2}x {:>10.2}x",
            n, cpu, batched, async_r, batched / cpu, async_r / batched);
    }
}
