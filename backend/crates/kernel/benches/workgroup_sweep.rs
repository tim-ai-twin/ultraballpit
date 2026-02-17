//! Workgroup size sweep for density + forces shaders.
//!
//! Tests workgroup sizes 32, 64, 128, 256 at 27K and 64K particles.
//! Run with: cargo bench --features gpu -p kernel --bench workgroup_sweep

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

fn bench_workgroup(n_particles: usize, wg_size: u32, n_steps: usize) -> f64 {
    let (particles, boundary, h) = create_particle_cube(n_particles);
    let actual_n = particles.len();

    let mut kernel = GpuKernel::new(
        particles, boundary, h,
        [0.0, -9.81, 0.0], 10.0, 0.4, 0.001,
        [0.0; 3], [0.1; 3],
    ).expect("GPU required");

    kernel.set_workgroup_size(wg_size);

    let dt = 1e-5_f32;
    // Warmup
    for _ in 0..3 {
        kernel.step(dt);
    }

    let start = Instant::now();
    for _ in 0..n_steps {
        kernel.step(dt);
    }
    let elapsed = start.elapsed().as_secs_f64();
    let sps = n_steps as f64 / elapsed;

    println!("  wg={:>3}  {:>6} particles  {:>3} steps  {:.3}s  {:>8.1} steps/s",
        wg_size, actual_n, n_steps, elapsed, sps);
    sps
}

fn main() {
    println!("=== Workgroup Size Sweep ===\n");

    let workgroup_sizes = [32, 64, 128, 256];

    for &(n, steps) in &[(27_000, 20), (64_000, 10)] {
        println!("--- {} particles ---", n);
        let mut best_wg = 0u32;
        let mut best_sps = 0.0f64;

        for &wg in &workgroup_sizes {
            let sps = bench_workgroup(n, wg, steps);
            if sps > best_sps {
                best_sps = sps;
                best_wg = wg;
            }
        }
        println!("  Best: wg={} ({:.1} steps/s)\n", best_wg, best_sps);
    }
}
