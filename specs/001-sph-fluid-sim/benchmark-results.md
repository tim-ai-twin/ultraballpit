# SPH Validation Benchmark Results

Phase 8 (US6) -- Standard SPH Validation Benchmarks

## Overview

This document records the results of four standard SPH validation benchmarks
implemented in Phase 8. These benchmarks compare the WCSPH solver output against
analytical solutions and published experimental data.

## Benchmark Summary

| Benchmark         | Particles | Status    | Accuracy vs Reference        | Notes                        |
|-------------------|-----------|-----------|------------------------------|------------------------------|
| Dam Break (T086)  | ~20,000   | Ready     | <= 10% vs Martin & Moyce     | Requires release-mode run    |
| Hydrostatic (T087)| ~500,000  | Ready     | <= 1% vs rho*g*h at 10 levels| High-res, 50+ layers         |
| Poiseuille (T088) | ~250,000  | Blocked   | <= 5% RMS vs parabolic       | Needs periodic BCs           |
| Standing Wave (T089)| ~20,000 | Blocked   | <= 5% vs linear wave theory  | Needs periodic BCs           |

## Running Benchmarks

```sh
# Run all benchmarks in release mode:
just test-benchmarks

# Or directly:
cd backend && cargo test --release -p reference-tests -- --ignored --nocapture
```

## T086: Dam Break (Martin & Moyce 1952)

**Configuration:** `configs/dam-break-2d.json`

- Domain: 5cm x 10cm x 0.5mm (2D slab, 1 particle thick in z)
- Particle spacing: 0.5mm
- Approximate particle count: ~20,000
- Water column: left quarter of domain (width a = 1.25cm)
- Simulation time: 0.5s

**Reference:** Martin, J. C. & Moyce, W. J. (1952). "An experimental study of
the collapse of liquid columns on a rigid horizontal plane." Phil. Trans. Roy.
Soc. A, 244(882), 312-324.

**Metric:** Dimensionless water front position Z* = z/a vs dimensionless time
T* = t * sqrt(2g/a). Pass criterion: all comparison points within 10% of
experimental data.

**Reference data:** `backend/reference-tests/data/martin-moyce-1952.json`

## T087: High-Resolution Hydrostatic Pressure

**Configuration:** `configs/hydrostatic-hires.json`

- Domain: 2cm x 10cm x 2cm (3D box)
- Particle spacing: 0.2mm
- Approximate particle count: ~500,000
- All-wall boundaries (fully enclosed)
- Simulation time: 1.0s (to reach steady state)

**Analytical Reference:** Hydrostatic pressure at depth h:
P(h) = rho * g * h, where rho = 1000 kg/m^3, g = 9.81 m/s^2.

**Metric:** Average pressure at 10 equally-spaced depth levels compared against
analytical hydrostatic pressure. Pass criterion: all levels within 1%.

## T088: Poiseuille Channel Flow

**Configuration:** `configs/poiseuille-2d.json`

- Domain: 10cm x 1cm x 0.2mm (2D slab)
- Particle spacing: 0.2mm
- Approximate particle count: ~250,000
- Periodic x-boundaries, wall y-boundaries
- Body force: Gx = 0.001 m/s^2 (drives flow in x-direction)
- Kinematic viscosity: 0.001 m^2/s

**Status:** BLOCKED -- requires periodic boundary conditions in the kernel.
The test infrastructure and analytical comparator are implemented, but the
test early-returns with a skip message until periodic BCs are available.

**Analytical Reference:** Poiseuille parabolic velocity profile:
u(y) = G / (2 * nu) * y * (H - y)

**Metric:** RMS error of velocity profile normalized by maximum analytical
velocity. Pass criterion: RMS error < 5%.

**Implementation:** `backend/reference-tests/src/analytical.rs` provides
`poiseuille_velocity()`, `poiseuille_max_velocity()`, and `poiseuille_rms_error()`.

## T089: Standing Wave

**Configuration:** `configs/standing-wave.json`

- Domain: 10cm x 5cm x 0.5mm (2D slab)
- Particle spacing: 0.5mm
- Approximate particle count: ~20,000
- Periodic x-boundaries, wall bottom, open top
- Water depth: ~2.5cm (half domain height)
- Wavelength: 10cm (= domain width, fundamental mode)

**Status:** BLOCKED -- requires periodic boundary conditions in the kernel.

**Analytical Reference:** Linear wave theory dispersion relation:
omega = sqrt(g * k * tanh(k * h))
T = 2 * pi / omega

**Metric:** Oscillation period extracted from surface displacement time series
compared against linear theory. Pass criterion: period within 5%.

**Implementation:** `backend/reference-tests/src/analytical.rs` provides
`standing_wave_omega()` and `standing_wave_period()`.

## Implementation Notes

### File Structure

```
configs/
  dam-break-2d.json           -- T080: Dam break configuration
  hydrostatic-hires.json       -- T081: High-res hydrostatic configuration
  poiseuille-2d.json           -- T082: Poiseuille flow configuration
  standing-wave.json           -- T083: Standing wave configuration

geometries/
  dam-break-slab.stl           -- T080: Dam break slab geometry
  hydrostatic-hires-box.stl    -- T081: Hydrostatic box geometry
  poiseuille-slab.stl          -- T082: Poiseuille slab geometry
  standing-wave-slab.stl       -- T083: Standing wave slab geometry

backend/reference-tests/
  data/
    martin-moyce-1952.json     -- T084: Experimental reference data
  src/
    analytical.rs              -- T085: Analytical solutions (Poiseuille, wave theory)
    benchmarks.rs              -- T086-T089: Benchmark test cases
    lib.rs                     -- Framework (run_simulation_to_time helper added)

justfile                       -- T090: test-benchmarks recipe added
specs/001-sph-fluid-sim/
  benchmark-results.md         -- T091: This document
```

### Periodic Boundary Conditions

Two of the four benchmarks (Poiseuille and standing wave) require periodic
boundary conditions, which are not yet implemented in the kernel. The config
format already supports `"Periodic"` boundary types and validation ensures
they are properly paired, but the kernel's `NeighborGrid` and
`enforce_no_penetration_domain` do not yet wrap particles or search across
periodic boundaries.

To unblock these benchmarks:
1. Modify `kernel::neighbor::NeighborGrid::for_each_neighbor` to search
   wrapped cells when periodic BCs are active.
2. Modify `kernel::boundary::BoundaryParticles::enforce_no_penetration_domain`
   to wrap particle positions instead of clamping at periodic boundaries.
3. Pass periodic BC configuration from `SimulationConfig` through to the kernel.

### Test Organization

- **Smoke tests** (`tests.rs`): Quick (seconds) tests that verify basic physical
  correctness. Run with `cargo test --workspace`.
- **Benchmark tests** (`benchmarks.rs`): Long-running (minutes) validation tests
  against reference data. Marked `#[ignore]`, run with
  `cargo test --release -p reference-tests -- --ignored`.
