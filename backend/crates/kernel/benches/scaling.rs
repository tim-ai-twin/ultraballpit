//! GPU scaling test -- push to larger particle counts.
//!
//! Run with: cargo bench --features gpu -p kernel --bench scaling

#![cfg(feature = "gpu")]

use std::time::Instant;
use kernel::{BoundaryParticles, FluidType, GpuKernel, ParticleArrays, SimulationKernel};

fn create_particle_cube(target_count: usize) -> (ParticleArrays, BoundaryParticles, f32) {
    let domain_size = 0.1_f32;
    let n_per_axis = (target_count as f32).cbrt().ceil() as usize;
    let spacing = domain_size / n_per_axis as f32;
    let h = 1.3 * spacing;
    let rest_density = 1000.0_f32;
    let particle_mass = rest_density * spacing * spacing * spacing;

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
    (particles, BoundaryParticles::new(), h)
}

fn main() {
    println!("=== GPU Scaling Test ===\n");

    // (target particles, steps) -- fewer steps at larger counts
    let configs = [
        (8_000, 20),
        (27_000, 10),
        (64_000, 5),
        (125_000, 3),
        (216_000, 2),
    ];

    println!("{:>10} {:>10} {:>10} {:>12} {:>12}",
        "Particles", "Steps", "Time (s)", "steps/s", "ms/step");

    for &(n, steps) in &configs {
        let (particles, boundary, h) = create_particle_cube(n);
        let actual_n = particles.len();

        let mut kernel = GpuKernel::new(
            particles, boundary, h,
            [0.0, -9.81, 0.0], 10.0, 0.4, 0.001,
            [0.0; 3], [0.1; 3],
        ).expect("GPU required");

        let dt = 1e-5_f32;
        // Warmup
        for _ in 0..2 {
            kernel.step(dt);
        }

        // Async path (best throughput)
        let start = Instant::now();
        for _ in 0..steps {
            kernel.step_no_sync(dt);
        }
        kernel.sync();
        let elapsed = start.elapsed().as_secs_f64();
        let sps = steps as f64 / elapsed;
        let ms_per_step = elapsed * 1000.0 / steps as f64;

        println!("{:>10} {:>10} {:>10.3} {:>12.1} {:>12.2}",
            actual_n, steps, elapsed, sps, ms_per_step);
    }
}
