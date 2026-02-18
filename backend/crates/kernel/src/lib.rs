//! SPH Fluid Simulation Kernel
//!
//! This crate provides the core simulation kernel for Smoothed Particle Hydrodynamics (SPH)
//! fluid simulation. It is designed to be separable and compute-focused.
//!
//! # Modules
//! - [`particle`] -- Struct-of-arrays particle storage and `FluidType` enum.
//! - [`sph`] -- Wendland C2 smoothing kernel, gradient, and core SPH operators.
//! - [`neighbor`] -- GPU-friendly uniform-grid spatial hash for neighbor search.
//! - [`eos`] -- Equations of state (Tait for water, ideal gas for air).
//! - [`boundary`] -- Boundary particle data and pressure mirroring (Adami et al. 2012).

#![warn(missing_docs)]

pub mod boundary;
pub mod eos;
pub mod neighbor;
pub mod particle;
pub mod sph;

#[cfg(feature = "gpu")]
#[allow(missing_docs)]
pub mod gpu;

pub use boundary::BoundaryParticles;
pub use eos::{ideal_gas_eos, tait_eos};
pub use neighbor::NeighborGrid;
pub use particle::{FluidType, ParticleArrays};
pub use sph::{wendland_c2, wendland_c2_gradient};

#[cfg(feature = "gpu")]
pub use gpu::{GpuKernel, GpuStepProfile};

// ---------------------------------------------------------------------------
// SimulationKernel trait
// ---------------------------------------------------------------------------

/// Aggregate error / conservation metrics for a simulation snapshot.
#[derive(Debug, Clone, Copy)]
pub struct ErrorMetrics {
    /// Maximum relative density deviation from rest density across all particles.
    pub max_density_variation: f32,
    /// Relative total energy drift from initial energy (|E - E0| / |E0|).
    pub energy_conservation: f32,
    /// Relative total mass drift from initial mass (|M - M0| / |M0|).
    pub mass_conservation: f32,
}

/// Trait that all simulation back-ends (CPU, GPU, hybrid) must implement.
///
/// A `SimulationKernel` owns particle data and advances the simulation through
/// distinct phases each time-step:
///
/// 1. Neighbor search
/// 2. Density summation
/// 3. Force computation (pressure + viscous + gravity + boundary)
/// 4. Time integration (Velocity Verlet)
pub trait SimulationKernel {
    /// Execute one simulation step of duration `dt` seconds.
    fn step(&mut self, dt: f32);

    /// Read back current particle state (immutable reference).
    fn particles(&self) -> &ParticleArrays;

    /// Get current error / conservation metrics.
    fn error_metrics(&self) -> ErrorMetrics;

    /// Number of particles in the simulation.
    fn particle_count(&self) -> usize;

    /// Save a checkpoint of the current particle state for potential rollback.
    /// Returns true if checkpointing is supported and succeeded.
    fn save_checkpoint(&mut self) -> bool { false }

    /// Restore the last saved checkpoint, undoing any steps since save.
    /// Returns true if restoration succeeded.
    fn restore_checkpoint(&mut self) -> bool { false }
}

// ---------------------------------------------------------------------------
// T023: CpuKernel -- reference CPU implementation of SimulationKernel
// ---------------------------------------------------------------------------

/// Reference CPU implementation of the SPH simulation kernel.
///
/// Uses Velocity Verlet (kick-drift-kick) time integration with:
/// - Wendland C2 smoothing kernel
/// - Tait EOS (water) / ideal gas EOS (air)
/// - Monaghan artificial viscosity
/// - Adami et al. (2012) boundary pressure mirroring
pub struct CpuKernel {
    /// Fluid particle data.
    particles: ParticleArrays,
    /// Boundary particle data.
    boundary: BoundaryParticles,
    /// Neighbor grid for spatial hashing.
    grid: NeighborGrid,
    /// Smoothing length (meters).
    h: f32,
    /// Gravitational acceleration vector (m/s^2).
    gravity: [f32; 3],
    /// Numerical speed of sound (m/s).
    speed_of_sound: f32,
    /// CFL number for adaptive timestep.
    #[allow(dead_code)]
    cfl_number: f32,
    /// Artificial viscosity coefficient (alpha in Monaghan model).
    #[allow(dead_code)]
    viscosity: f32,
    /// Initial total energy for conservation tracking.
    initial_energy: f64,
    /// Initial total mass for conservation tracking.
    initial_mass: f64,
    /// Domain minimum bounds for no-penetration clamping.
    domain_min: [f32; 3],
    /// Domain maximum bounds for no-penetration clamping.
    domain_max: [f32; 3],
    /// Whether the initial force computation has been performed.
    needs_init: bool,
    /// Saved particle state for optimistic timestepping rollback.
    checkpoint: Option<ParticleArrays>,
}

impl CpuKernel {
    /// Create a new CPU simulation kernel.
    ///
    /// # Arguments
    /// * `particles` - Initial fluid particle data.
    /// * `boundary` - Boundary particle data (static).
    /// * `h` - Smoothing length.
    /// * `gravity` - Gravitational acceleration vector.
    /// * `speed_of_sound` - Numerical speed of sound for Tait EOS and CFL.
    /// * `cfl_number` - CFL number (typically 0.2-0.4).
    /// * `viscosity` - Artificial viscosity coefficient.
    /// * `domain_min` - Minimum corner of the simulation domain.
    /// * `domain_max` - Maximum corner of the simulation domain.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        particles: ParticleArrays,
        boundary: BoundaryParticles,
        h: f32,
        gravity: [f32; 3],
        speed_of_sound: f32,
        cfl_number: f32,
        viscosity: f32,
        domain_min: [f32; 3],
        domain_max: [f32; 3],
    ) -> Self {
        let cell_size = 2.0 * h;
        let grid = NeighborGrid::new(cell_size, domain_min, domain_max);

        // Compute initial mass
        let initial_mass: f64 = particles.mass.iter().map(|&m| m as f64).sum();

        // Compute initial energy (kinetic + potential gravitational)
        let initial_energy = Self::compute_total_energy_static(&particles, gravity);

        Self {
            particles,
            boundary,
            grid,
            h,
            gravity,
            speed_of_sound,
            cfl_number,
            viscosity,
            initial_energy,
            initial_mass,
            domain_min,
            domain_max,
            needs_init: true,
            checkpoint: None,
        }
    }

    /// Compute total energy (kinetic + gravitational potential) for given particles.
    fn compute_total_energy_static(particles: &ParticleArrays, gravity: [f32; 3]) -> f64 {
        let mut energy = 0.0_f64;
        for i in 0..particles.len() {
            let m = particles.mass[i] as f64;
            // Kinetic energy: 0.5 * m * |v|^2
            let vx = particles.vx[i] as f64;
            let vy = particles.vy[i] as f64;
            let vz = particles.vz[i] as f64;
            energy += 0.5 * m * (vx * vx + vy * vy + vz * vz);
            // Gravitational potential: -m * g . r
            // (using the convention that potential = -m * g . r, with g pointing down)
            let x = particles.x[i] as f64;
            let y = particles.y[i] as f64;
            let z = particles.z[i] as f64;
            energy -= m * (gravity[0] as f64 * x + gravity[1] as f64 * y + gravity[2] as f64 * z);
        }
        energy
    }
}

impl CpuKernel {
    /// Run the full force computation pipeline (density, pressure, boundary
    /// pressures, forces) without integration. Used for bootstrapping
    /// initial accelerations for Velocity Verlet.
    ///
    /// `dt` is the simulation timestep used for delta-SPH density diffusion.
    fn compute_forces(&mut self, dt: f32) {
        let n = self.particles.len();

        // Update neighbor grid
        self.grid.update(
            &self.particles.x,
            &self.particles.y,
            &self.particles.z,
        );

        // Delta-SPH density diffusion (Molteni & Colagrossi 2009)
        // Computed BEFORE density summation so it uses previous-step density,
        // matching the GPU shader which reads density[i] before overwriting.
        let diffusion = sph::compute_density_diffusion(
            &self.particles, &self.grid, self.h, self.speed_of_sound,
        );

        // Compute density (T017) -- overwrites density with summation values
        sph::compute_density(
            &mut self.particles,
            &self.boundary.x,
            &self.boundary.y,
            &self.boundary.z,
            &self.boundary.mass,
            &self.grid,
            self.h,
        );

        // Apply diffusion correction to the new summation density
        for i in 0..n {
            self.particles.density[i] += diffusion[i] * dt;
        }

        // Compute pressure from EOS
        sph::compute_pressure(&mut self.particles, self.speed_of_sound);

        // Update boundary pressures (T021)
        self.boundary.update_pressures(
            &self.particles,
            &self.grid,
            self.gravity,
            self.h,
        );

        // Zero accelerations and compute forces
        for i in 0..n {
            self.particles.ax[i] = 0.0;
            self.particles.ay[i] = 0.0;
            self.particles.az[i] = 0.0;
        }

        // Pressure forces (T018)
        sph::compute_pressure_forces(
            &mut self.particles,
            &self.boundary.x,
            &self.boundary.y,
            &self.boundary.z,
            &self.boundary.mass,
            &self.boundary.pressure,
            &self.grid,
            self.h,
        );

        // Viscous forces (T019)
        sph::compute_viscous_forces(
            &mut self.particles,
            &self.grid,
            self.h,
            self.speed_of_sound,
        );

        // Gravity (T020)
        sph::apply_gravity(&mut self.particles, self.gravity);

        // Boundary repulsive forces (Monaghan & Kos 1999)
        self.boundary.compute_repulsive_forces(
            &mut self.particles,
            self.h,
            self.speed_of_sound,
        );
    }
}

impl SimulationKernel for CpuKernel {
    fn step(&mut self, dt: f32) {
        let n = self.particles.len();
        let half_dt = 0.5 * dt;

        // --- 0. Bootstrap: compute initial forces on first step ---
        // On the very first timestep, all accelerations are zero because no
        // forces have been computed yet. The first half-kick would do nothing,
        // producing incorrect results. So we run the force pipeline once first.
        if self.needs_init {
            self.compute_forces(dt);
            self.needs_init = false;
        }

        // --- 1. Half-kick: v(t + dt/2) = v(t) + a(t) * dt/2 ---
        for i in 0..n {
            self.particles.vx[i] += self.particles.ax[i] * half_dt;
            self.particles.vy[i] += self.particles.ay[i] * half_dt;
            self.particles.vz[i] += self.particles.az[i] * half_dt;
        }

        // --- 2. Drift with XSPH correction: x(t+dt) = x(t) + (v + dv_xsph) * dt ---
        // XSPH smooths the velocity field used for position updates, reducing
        // particle disorder and improving stability (Monaghan 1989).
        let (dvx, dvy, dvz) = sph::compute_xsph_correction(
            &self.particles, &self.grid, self.h,
        );
        for i in 0..n {
            self.particles.x[i] += (self.particles.vx[i] + dvx[i]) * dt;
            self.particles.y[i] += (self.particles.vy[i] + dvy[i]) * dt;
            self.particles.z[i] += (self.particles.vz[i] + dvz[i]) * dt;
        }

        // --- 2b. Boundary collision: clamp positions to domain bounds ---
        self.boundary.enforce_no_penetration_domain(
            &mut self.particles,
            self.domain_min,
            self.domain_max,
        );

        // --- 3-7. Compute all forces ---
        self.compute_forces(dt);

        // --- 8. Second half-kick: v(t + dt) = v(t + dt/2) + a(t + dt) * dt/2 ---
        for i in 0..n {
            self.particles.vx[i] += self.particles.ax[i] * half_dt;
            self.particles.vy[i] += self.particles.ay[i] * half_dt;
            self.particles.vz[i] += self.particles.az[i] * half_dt;
        }
    }

    fn particles(&self) -> &ParticleArrays {
        &self.particles
    }

    fn error_metrics(&self) -> ErrorMetrics {
        // Maximum density variation
        let mut max_density_var = 0.0_f32;
        for i in 0..self.particles.len() {
            let rest_rho = match self.particles.fluid_type[i] {
                FluidType::Water => eos::WATER_REST_DENSITY,
                FluidType::Air => eos::AIR_REST_DENSITY,
            };
            let var = (self.particles.density[i] - rest_rho).abs() / rest_rho;
            if var > max_density_var {
                max_density_var = var;
            }
        }

        // Energy conservation
        let current_energy =
            Self::compute_total_energy_static(&self.particles, self.gravity);
        let energy_drift = if self.initial_energy.abs() > 1.0e-12 {
            ((current_energy - self.initial_energy) / self.initial_energy).abs() as f32
        } else {
            (current_energy - self.initial_energy).abs() as f32
        };

        // Mass conservation
        let current_mass: f64 = self.particles.mass.iter().map(|&m| m as f64).sum();
        let mass_drift = if self.initial_mass.abs() > 1.0e-12 {
            ((current_mass - self.initial_mass) / self.initial_mass).abs() as f32
        } else {
            (current_mass - self.initial_mass).abs() as f32
        };

        ErrorMetrics {
            max_density_variation: max_density_var,
            energy_conservation: energy_drift,
            mass_conservation: mass_drift,
        }
    }

    fn particle_count(&self) -> usize {
        self.particles.len()
    }

    fn save_checkpoint(&mut self) -> bool {
        self.checkpoint = Some(self.particles.clone());
        true
    }

    fn restore_checkpoint(&mut self) -> bool {
        if let Some(cp) = self.checkpoint.take() {
            self.particles = cp;
            true
        } else {
            false
        }
    }
}

// ---------------------------------------------------------------------------
// Legacy shim -- keeps `kernel::simulation::init()` working for the
// orchestrator crate during early development.
// ---------------------------------------------------------------------------

/// Placeholder module for backward compatibility.
pub mod simulation {
    /// Placeholder initialization function.
    pub fn init() {
        tracing::info!("SPH kernel initialized");
    }
}
