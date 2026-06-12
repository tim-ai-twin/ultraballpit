//! Per-pass GPU profiling: no-obstacle vs pillar dam break.
//!
//! Run with: cargo run --release -p orchestrator --example profile_obstacle --features gpu

#[cfg(not(feature = "gpu"))]
fn main() {
    eprintln!("build with --features gpu");
}

#[cfg(feature = "gpu")]
fn main() {
    use orchestrator::config::SimulationConfig;
    use orchestrator::{domain, geometry};

    kernel::simulation::init();

    let base = serde_json::json!({
        "name": "prof", "fluid_type": "Water",
        "domain": {"min": [0.0,0.0,0.0], "max": [0.12,0.08,0.06]},
        "fluid_region": {"min": [0.0,0.0,0.0], "max": [0.036,0.06,0.06]},
        "boundary_conditions": {"x_min":"Wall","x_max":"Wall","y_min":"Wall","y_max":"Outflow","z_min":"Wall","z_max":"Wall"},
        "particle_spacing": 0.0025, "gravity": [0.0,-9.81,0.0],
        "speed_of_sound": 20.0, "viscosity": 0.001, "cfl_number": 0.4,
        "backend": "gpu", "solver": "wcsph"
    });

    for (label, with_obstacle) in [("no-obstacle", false), ("pillar", true)] {
        let mut cfg_json = base.clone();
        if with_obstacle {
            cfg_json["geometry"] = serde_json::json!({
                "type": "cylinder", "center": [0.075,0.04,0.03],
                "radius": 0.02, "height": 0.08, "axis": "y"
            });
        }
        let config: SimulationConfig = serde_json::from_value(cfg_json).unwrap();
        let config_dir = std::path::Path::new(".");
        let triangles = geometry::resolve_geometry(&config, config_dir).unwrap();
        let sdf = geometry::generate_sdf(&triangles, config.domain.min, config.domain.max, 0.5 * config.particle_spacing);
        let (fluid, boundary_data) = domain::setup_domain(&config, &sdf);
        let mut boundary = kernel::BoundaryParticles::new();
        for b in &boundary_data {
            boundary.push(b.x, b.y, b.z, b.mass, b.nx, b.ny, b.nz);
        }
        let n_fluid = fluid.len();
        let n_bnd = boundary.len();
        let h = config.smoothing_length();

        let mut k = kernel::GpuKernel::new(
            fluid, boundary, h, config.gravity, config.speed_of_sound,
            config.cfl_number, config.viscosity, config.domain.min, config.domain.max,
            kernel::SolverType::Wcsph,
        ).unwrap();

        let dt = config.cfl_number * h / config.speed_of_sound;
        for _ in 0..20 { k.step_profiled(dt); } // warm up

        let n = 60;
        let mut acc = kernel::gpu::GpuStepProfile::default();
        for _ in 0..n {
            let p = k.step_profiled(dt);
            acc.grid_build_us += p.grid_build_us;
            acc.density_us += p.density_us;
            acc.boundary_pressure_us += p.boundary_pressure_us;
            acc.forces_us += p.forces_us;
            acc.integrate_us += p.integrate_us;
            acc.readback_us += p.readback_us;
            acc.total_us += p.total_us;
        }
        let d = n as u64;
        println!("\n=== {label}: {n_fluid} fluid, {n_bnd} boundary ===");
        println!("  grid_build       {:>7} us", acc.grid_build_us / d);
        println!("  density          {:>7} us", acc.density_us / d);
        println!("  boundary_pressure{:>7} us", acc.boundary_pressure_us / d);
        println!("  forces           {:>7} us", acc.forces_us / d);
        println!("  integrate        {:>7} us", acc.integrate_us / d);
        println!("  readback         {:>7} us", acc.readback_us / d);
        println!("  TOTAL            {:>7} us/step", acc.total_us / d);
    }
}
