//! PCISPH validation tests.
//!
//! Verifies that the PCISPH solver:
//! 1. Converges (pressure iterations terminate without explosion)
//! 2. Maintains density better than freefall for dynamic scenarios
//! 3. Agrees qualitatively with WCSPH for identical initial conditions
//! 4. Allows larger timesteps (advective CFL vs acoustic CFL)

use kernel::{
    BoundaryParticles, CpuKernel, FluidType, ParticleArrays, SimulationKernel, SolverType,
};
use kernel::eos::WATER_REST_DENSITY;

/// Helper: create a water block test setup with boundary particles at the floor.
fn create_water_block(
    nx: usize,
    ny: usize,
    nz: usize,
    h: f32,
    speed_of_sound: f32,
    solver_type: SolverType,
) -> CpuKernel {
    let spacing = h / 1.3;
    let rest_density = WATER_REST_DENSITY;
    let mass = rest_density * spacing * spacing * spacing;
    let gravity = [0.0_f32, -9.81, 0.0];

    let mut particles = ParticleArrays::new();
    for iy in 0..ny {
        for iz in 0..nz {
            for ix in 0..nx {
                let px = (ix as f32 + 0.5) * spacing;
                let py = (iy as f32 + 1.0) * spacing;
                let pz = (iz as f32 + 0.5) * spacing;
                particles.push_particle(px, py, pz, mass, rest_density, 293.15, FluidType::Water);
            }
        }
    }

    // 3 boundary layers below y=0
    let mut boundary = BoundaryParticles::new();
    for layer in 0..3 {
        let by = -(layer as f32) * spacing;
        for iz in -1..(nz as i32 + 2) {
            for ix in -1..(nx as i32 + 2) {
                let bx = (ix as f32 + 0.5) * spacing;
                let bz = (iz as f32 + 0.5) * spacing;
                boundary.push(bx, by, bz, mass, 0.0, 1.0, 0.0);
            }
        }
    }

    let margin = 6.0 * h;
    let domain_min = [-margin, -margin, -margin];
    let domain_max = [
        (nx + 3) as f32 * spacing + margin,
        (ny + 10) as f32 * spacing + margin,
        (nz + 3) as f32 * spacing + margin,
    ];

    CpuKernel::with_solver(
        particles,
        boundary,
        h,
        gravity,
        speed_of_sound,
        0.4,
        1.0,
        domain_min,
        domain_max,
        solver_type,
    )
}

#[test]
fn pcisph_solver_type() {
    let h = 0.02_f32;
    let sim = create_water_block(4, 4, 4, h, 20.0, SolverType::Pcisph);
    assert_eq!(sim.solver_type(), SolverType::Pcisph);
}

#[test]
fn pcisph_runs_without_explosion() {
    // Basic sanity: PCISPH should run for many steps without NaN or infinite values.
    let h = 0.02_f32;
    let mut sim = create_water_block(4, 4, 4, h, 20.0, SolverType::Pcisph);

    let dt = 0.002;
    for _ in 0..100 {
        sim.step(dt);
    }

    let p = sim.particles();
    for i in 0..p.len() {
        assert!(p.x[i].is_finite(), "NaN/Inf in positions");
        assert!(p.vx[i].is_finite(), "NaN/Inf in velocities");
        assert!(p.density[i].is_finite(), "NaN/Inf in density");
        assert!(p.density[i] > 0.0, "Non-positive density");
    }
}

#[test]
fn pcisph_density_bounded() {
    // Run PCISPH and verify the over-compression stays bounded.
    // PCISPH's iterative correction should prevent extreme density spikes.
    let h = 0.02_f32;
    let mut sim = create_water_block(6, 6, 6, h, 20.0, SolverType::Pcisph);

    let dt = 0.002;
    let n_steps = 50;
    let mut max_over_compression = 0.0f32;

    for _ in 0..n_steps {
        sim.step(dt);
        let p = sim.particles();
        for i in 0..p.len() {
            let rho_err = (p.density[i] - WATER_REST_DENSITY) / WATER_REST_DENSITY;
            if rho_err > max_over_compression {
                max_over_compression = rho_err;
            }
        }
    }

    eprintln!("PCISPH max over-compression: {:.1}%", max_over_compression * 100.0);

    // PCISPH targets <1% mean over-compression per iteration.
    // Max over-compression across all steps and particles should stay bounded.
    assert!(
        max_over_compression < 0.10,
        "Max over-compression should be < 10%, got {:.1}%",
        max_over_compression * 100.0,
    );
}

#[test]
fn pcisph_vs_wcsph_qualitative_agreement() {
    // Run both PCISPH and WCSPH on identical initial conditions.
    // Both should produce finite results with particles in the same general region.
    let h = 0.02_f32;
    let speed_of_sound = 20.0;

    let mut wcsph = create_water_block(4, 4, 4, h, speed_of_sound, SolverType::Wcsph);
    let mut pcisph = create_water_block(4, 4, 4, h, speed_of_sound, SolverType::Pcisph);

    // Use the same small timestep so WCSPH is well-resolved
    let dt = 0.0001;
    let n_steps = 200;

    for _ in 0..n_steps {
        wcsph.step(dt);
        pcisph.step(dt);
    }

    let pw = wcsph.particles();
    let pp = pcisph.particles();

    assert_eq!(pw.len(), pp.len());

    // Compare center of mass — should be in similar region
    let com_w_y: f32 = pw.y.iter().sum::<f32>() / pw.len() as f32;
    let com_p_y: f32 = pp.y.iter().sum::<f32>() / pp.len() as f32;

    let spacing = h / 1.3;
    eprintln!("WCSPH CoM y = {com_w_y:.4}, PCISPH CoM y = {com_p_y:.4}");

    // Centers of mass should be within a few particle spacings
    assert!(
        (com_w_y - com_p_y).abs() < 5.0 * spacing,
        "PCISPH and WCSPH center of mass should be similar. \
         WCSPH={com_w_y:.4}, PCISPH={com_p_y:.4}"
    );
}

#[test]
fn pcisph_advective_timestep_larger_than_acoustic() {
    // Verify that the advective CFL timestep (PCISPH) is significantly larger
    // than the acoustic CFL timestep (WCSPH).
    let h = 0.02_f32;
    let speed_of_sound = 20.0;

    let mut sim = create_water_block(4, 4, 4, h, speed_of_sound, SolverType::Wcsph);
    // Run a few steps to establish forces
    for _ in 0..5 {
        sim.step(0.0001);
    }

    let dt_acoustic = kernel::sph::compute_timestep(
        sim.particles(), h, speed_of_sound, 0.4,
    );
    let dt_advective = kernel::sph::compute_timestep_advective(
        sim.particles(), h, 0.4,
    );

    eprintln!("dt_acoustic (WCSPH) = {dt_acoustic:.6}");
    eprintln!("dt_advective (PCISPH) = {dt_advective:.6}");

    // Advective CFL should be larger since it doesn't include c_s
    assert!(
        dt_advective > dt_acoustic,
        "Advective CFL ({dt_advective:.6}) should be larger than acoustic CFL ({dt_acoustic:.6})"
    );
}

#[test]
fn pcisph_larger_timestep_still_stable() {
    // PCISPH should remain stable at timesteps larger than WCSPH's acoustic CFL.
    let h = 0.02_f32;

    let mut sim = create_water_block(4, 4, 4, h, 20.0, SolverType::Pcisph);

    // Use a timestep that would be unstable for WCSPH (> acoustic CFL)
    // Acoustic CFL for c_s=20, h=0.02: dt ≈ 0.4 * 0.02 / 20 = 0.0004
    // Use 5x larger: dt = 0.002
    let dt = 0.002;
    let n_steps = 50;

    for _ in 0..n_steps {
        sim.step(dt);
    }

    // Should not explode
    let p = sim.particles();
    let max_v: f32 = (0..p.len())
        .map(|i| (p.vx[i].powi(2) + p.vy[i].powi(2) + p.vz[i].powi(2)).sqrt())
        .fold(0.0, f32::max);

    eprintln!("PCISPH at dt=0.002 (5x acoustic CFL): max_v = {max_v:.2} m/s");

    // Velocities should be physical (not exploding)
    assert!(
        max_v < 10.0,
        "Max velocity should be < 10 m/s at this setup, got {max_v:.2}"
    );
}
