//! GPU neighbor-grid build and boundary-pressure checks against brute-force
//! CPU references.

#![cfg(feature = "gpu")]

use kernel::{BoundaryParticles, FluidType, GpuKernel, ParticleArrays, SimulationKernel, SolverType};

/// Small deterministic PRNG (xorshift) so the test needs no extra crates.
struct Rng(u64);
impl Rng {
    fn next_f32(&mut self) -> f32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 40) as f32 / (1u64 << 24) as f32
    }
}

/// Large grid (> 1M cells, > 1000 scan tiles) with a mix of uniformly
/// scattered particles and dense clusters, so many cells are empty and some
/// hold dozens of particles.
#[test]
fn gpu_grid_prefix_sum_large() {
    let h = 0.002_f32;
    let dmin = [0.0_f32; 3];
    let dmax = [0.3_f32, 0.2, 0.15];
    let mut rng = Rng(0x9e3779b97f4a7c15);
    let mut particles = ParticleArrays::new();
    let mass = 1e-6;
    for _ in 0..60_000 {
        let (x, y, z) = (rng.next_f32() * dmax[0], rng.next_f32() * dmax[1], rng.next_f32() * dmax[2]);
        particles.push_particle(x, y, z, mass, 1000.0, 293.15, FluidType::Water);
    }
    for c in 0..40 {
        let centre = [rng.next_f32() * dmax[0], rng.next_f32() * dmax[1], rng.next_f32() * dmax[2]];
        for _ in 0..1000 {
            let p: Vec<f32> = (0..3)
                .map(|a| (centre[a] + (rng.next_f32() - 0.5) * 0.004 * (1 + c % 3) as f32).clamp(0.0, dmax[a] * 0.9999))
                .collect();
            particles.push_particle(p[0], p[1], p[2], mass, 1000.0, 293.15, FluidType::Water);
        }
    }
    let n = particles.len();

    let mut gpu = match GpuKernel::new(
        particles, BoundaryParticles::new(), h, [0.0, -9.81, 0.0], 10.0, 0.4, 0.001, dmin, dmax, SolverType::Wcsph,
    ) {
        Ok(k) => k,
        Err(e) => {
            eprintln!("skip: {e}");
            return;
        }
    };

    // Build twice: the second build checks that the first left its
    // counters zeroed.
    for build in 0..2 {
        let g = gpu.debug_rebuild_grid();
        let cells = g.cell_counts.len();
        assert_eq!(cells, (g.grid_dims[0] * g.grid_dims[1] * g.grid_dims[2]) as usize);
        assert!(cells > 1_000_000 && g.scan_tile_sums.len() > 1000, "grid too small to exercise the scan");

        assert!(g.cell_fill.iter().all(|&c| c == 0), "build {build}: cell_fill not reset");
        // Scan tiles are GRID_SCAN_TILE = 1024 cells.
        assert_eq!(g.scan_tile_sums.len(), cells.div_ceil(1024));
        for (t, chunk) in g.cell_counts.chunks(1024).enumerate() {
            assert_eq!(g.scan_tile_sums[t], chunk.iter().sum::<u32>(), "build {build}: tile {t} total");
        }

        let mut running = 0u32;
        for c in 0..cells {
            assert_eq!(g.cell_offsets[c], running, "build {build}: offset mismatch at cell {c}");
            running += g.cell_counts[c];
        }
        assert_eq!(running as usize, n, "build {build}: counts don't sum to particle count");
        assert!(g.cell_counts.iter().any(|&c| c >= 20), "expected some dense cells");

        // Particles are now stored in cell order: particle k must lie in the
        // cell whose [offset, offset + count) range contains k.
        let p = gpu.particles().clone();
        let mut k = 0usize;
        for c in 0..cells {
            for _ in 0..g.cell_counts[c] {
                let ix = c as u32 % g.grid_dims[0];
                let iy = (c as u32 / g.grid_dims[0]) % g.grid_dims[1];
                let iz = c as u32 / (g.grid_dims[0] * g.grid_dims[1]);
                let tol = 1e-4 * g.cell_size;
                for (pos, (i, m)) in [p.x[k], p.y[k], p.z[k]].into_iter().zip([ix, iy, iz].into_iter().zip(g.domain_min)) {
                    let lo = m + i as f32 * g.cell_size;
                    assert!(
                        pos >= lo - tol && pos <= lo + g.cell_size + tol,
                        "build {build}: particle {k} at {pos} outside cell {c} [{lo}, {}]",
                        lo + g.cell_size
                    );
                }
                k += 1;
            }
        }
    }
}

/// GPU boundary pressures (Adami mirroring) vs a brute-force CPU evaluation
/// over the GPU's own fluid state, for every boundary particle.
#[test]
fn gpu_boundary_pressure_matches_brute_force() {
    let spacing = 0.002_f32;
    let h = 1.3 * spacing;
    let dmin = [0.0_f32; 3];
    let dmax = [0.08_f32, 0.06, 0.05];
    let mass = 1000.0 * spacing.powi(3);
    let mut particles = ParticleArrays::new();
    // Fluid block in one corner of the box.
    for ix in 0..12 {
        for iy in 0..14 {
            for iz in 0..24 {
                particles.push_particle(
                    (ix as f32 + 0.5) * spacing,
                    (iy as f32 + 0.5) * spacing,
                    (iz as f32 + 0.5) * spacing,
                    mass, 1000.0, 293.15, FluidType::Water,
                );
            }
        }
    }
    // Floor and two walls, 3 layers thick, covering the whole box.
    let mut boundary = BoundaryParticles::new();
    let (nx, ny, nz) = ((dmax[0] / spacing) as i32, (dmax[1] / spacing) as i32, (dmax[2] / spacing) as i32);
    for l in 0..3 {
        let d = -(l as f32 + 0.5) * spacing;
        for i in 0..nx {
            for k in 0..nz {
                boundary.push((i as f32 + 0.5) * spacing, d, (k as f32 + 0.5) * spacing, mass, 0.0, 1.0, 0.0);
            }
        }
        for j in 0..ny {
            for k in 0..nz {
                boundary.push(d, (j as f32 + 0.5) * spacing, (k as f32 + 0.5) * spacing, mass, 1.0, 0.0, 0.0);
            }
        }
        for i in 0..nx {
            for j in 0..ny {
                boundary.push((i as f32 + 0.5) * spacing, (j as f32 + 0.5) * spacing, d, mass, 0.0, 0.0, 1.0);
            }
        }
    }
    let g = [0.0, -9.81, 0.0];
    let mut gpu = match GpuKernel::new(particles, boundary, h, g, 20.0, 0.4, 0.001, dmin, dmax, SolverType::Wcsph) {
        Ok(k) => k,
        Err(e) => {
            eprintln!("skip: {e}");
            return;
        }
    };
    for _ in 0..30 {
        gpu.step(5e-5);
    }
    let [bx, by, bz, bp] = gpu.debug_boundary_pressures();
    let f = gpu.particles().clone();
    let r2_max = 4.0 * h * h;
    let (mut active, mut max_err, mut max_p) = (0usize, 0.0f32, 0.0f32);
    for b in 0..bx.len() {
        let (mut wp, mut ws) = (0.0f64, 0.0f64);
        for j in 0..f.len() {
            let (dx, dy, dz) = (bx[b] - f.x[j], by[b] - f.y[j], bz[b] - f.z[j]);
            let d2 = dx * dx + dy * dy + dz * dz;
            if d2 < r2_max {
                let w = kernel::sph::wendland_c2(d2.sqrt(), h) as f64;
                let pe = f.pressure[j] as f64 + f.density[j] as f64 * (g[0] * dx + g[1] * dy + g[2] * dz) as f64;
                wp += w * pe;
                ws += w;
            }
        }
        let expect = if ws > 1e-12 { (wp / ws).max(0.0) as f32 } else { 0.0 };
        if ws > 1e-12 {
            active += 1;
        } else {
            assert_eq!(bp[b], 0.0, "boundary {b} has no fluid neighbours but pressure {}", bp[b]);
        }
        max_err = max_err.max((bp[b] - expect).abs());
        max_p = max_p.max(expect);
    }
    println!("boundary: {active}/{} active, max |dp| = {max_err:.4e} Pa (max p = {max_p:.1} Pa)", bx.len());
    assert!(active > 100 && active < bx.len() / 2, "unexpected active count {active}");
    assert!(max_p > 10.0, "pressures too small to be meaningful");
    assert!(max_err <= 1e-4 * max_p, "boundary pressure mismatch {max_err}");
}
