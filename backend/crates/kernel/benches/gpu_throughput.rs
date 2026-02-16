//! GPU throughput benchmark (T104)
//!
//! Measures timesteps/second for various particle counts on GPU vs CPU.
//! Verifies GPU achieves meaningful speedup at larger particle counts.
//!
//! Run with: cargo bench --features gpu -p kernel --bench gpu_throughput

#![cfg(feature = "gpu")]

use std::time::Instant;

use kernel::{
    BoundaryParticles, CpuKernel, FluidType, GpuKernel, ParticleArrays, SimulationKernel,
};

/// Create a particle cube with the given number of particles (approximate).
fn create_particle_cube(target_count: usize) -> (ParticleArrays, BoundaryParticles, f32) {
    // Determine spacing from target count: n = (domain_size / spacing)^3
    // domain_size = 0.1m (10cm box)
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

    let boundary = BoundaryParticles::new(); // No boundary for throughput test
    (particles, boundary, h)
}

fn benchmark_cpu(n_particles: usize, n_steps: usize) -> f64 {
    let (particles, boundary, h) = create_particle_cube(n_particles);
    let actual_n = particles.len();
    let domain_min = [0.0_f32; 3];
    let domain_max = [0.1_f32; 3];

    let mut kernel = CpuKernel::new(
        particles,
        boundary,
        h,
        [0.0, -9.81, 0.0],
        10.0,
        0.4,
        0.001,
        domain_min,
        domain_max,
    );

    let dt = 1e-5_f32;
    let start = Instant::now();
    for _ in 0..n_steps {
        kernel.step(dt);
    }
    let elapsed = start.elapsed().as_secs_f64();

    let steps_per_sec = n_steps as f64 / elapsed;
    println!(
        "CPU  {:>6} particles, {} steps in {:.3}s = {:.1} steps/s",
        actual_n, n_steps, elapsed, steps_per_sec
    );
    steps_per_sec
}

fn benchmark_gpu(n_particles: usize, n_steps: usize) -> Option<f64> {
    let (particles, boundary, h) = create_particle_cube(n_particles);
    let actual_n = particles.len();
    let domain_min = [0.0_f32; 3];
    let domain_max = [0.1_f32; 3];

    let mut kernel = match GpuKernel::new(
        particles,
        boundary,
        h,
        [0.0, -9.81, 0.0],
        10.0,
        0.4,
        0.001,
        domain_min,
        domain_max,
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

    let start = Instant::now();
    for _ in 0..n_steps {
        kernel.step(dt);
    }
    let elapsed = start.elapsed().as_secs_f64();

    let steps_per_sec = n_steps as f64 / elapsed;
    println!(
        "GPU  {:>6} particles, {} steps in {:.3}s = {:.1} steps/s",
        actual_n, n_steps, elapsed, steps_per_sec
    );
    Some(steps_per_sec)
}

fn main() {
    println!("=== GPU Throughput Benchmark (T104) ===\n");

    let configs = [
        (1_000, 50),   // Small: 10^3
        (8_000, 20),   // Medium: ~20^3
        (27_000, 10),  // Large: ~30^3
    ];

    let mut results = Vec::new();

    for &(n, steps) in &configs {
        println!("--- {} particles, {} steps ---", n, steps);
        let cpu_rate = benchmark_cpu(n, steps);
        let gpu_rate = benchmark_gpu(n, steps);
        if let Some(gpu) = gpu_rate {
            let speedup = gpu / cpu_rate;
            println!("  Speedup: {:.2}x\n", speedup);
            results.push((n, cpu_rate, gpu, speedup));
        } else {
            println!("  GPU not available\n");
        }
    }

    println!("\n=== Summary ===");
    println!("{:>10} {:>12} {:>12} {:>10}", "Particles", "CPU (s/s)", "GPU (s/s)", "Speedup");
    for (n, cpu, gpu, speedup) in &results {
        println!("{:>10} {:>12.1} {:>12.1} {:>10.2}x", n, cpu, gpu, speedup);
    }
}
