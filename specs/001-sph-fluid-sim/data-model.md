# Data Model: SPH Fluid Simulation

**Feature**: `001-sph-fluid-sim`
**Date**: 2026-02-15

## Entities

### Particle

The fundamental simulation element. Stored in a struct-of-arrays
(SoA) layout for cache efficiency and SIMD/GPU friendliness.

| Field | Type | Description |
|-------|------|-------------|
| position | Vec3 (f32×3) | World-space position |
| velocity | Vec3 (f32×3) | Velocity vector |
| acceleration | Vec3 (f32×3) | Accumulated forces / mass |
| density | f32 | SPH-computed density at particle location |
| pressure | f32 | Computed from equation of state |
| mass | f32 | Particle mass (constant per particle) |
| temperature | f32 | Thermal state in Kelvin |
| fluid_type | enum { Water, Air } | Particle's fluid type (immutable after creation) |

Smoothing length `h` is a global simulation parameter (= 1.3 x
particle_spacing), not stored per-particle.

**Layout**: Struct-of-arrays — each field is a contiguous `Vec<f32>`
or `Vec<FluidType>`. This enables vectorized operations and easy
serialization for WebSocket streaming (send positions as a flat f32
slice).

**Identity**: Particles are identified by index into the SoA arrays.
No persistent ID needed — particles are not individually tracked
across frames by the user.

**Lifecycle**: Created at simulation start. Removed if they escape
the domain boundary. Recycled at inflow boundaries.

### SimulationConfig

Deserialized from a JSON file. Defines all parameters for a
simulation run.

| Field | Type | Description |
|-------|------|-------------|
| name | string | Human-readable simulation name |
| fluid_type | enum { Water, Air, Mixed } | Which fluids to simulate |
| geometry_file | string (path) | Path to STL file |
| domain_bounds | { min: Vec3, max: Vec3 } | Axis-aligned bounding box |
| boundary_conditions | { x_min, x_max, y_min, y_max, z_min, z_max: BoundaryType } | Per-face: Wall, Inflow, Outflow, Periodic |
| inflow_velocity | Vec3 (optional) | Velocity for inflow faces |
| inflow_temperature | f32 (optional) | Temperature for inflow particles |
| particle_spacing | f32 | Initial inter-particle distance (meters) |
| gravity | Vec3 | Gravity vector (default: [0, -9.81, 0]) |
| speed_of_sound | f32 | WCSPH parameter controlling compressibility |
| viscosity | f32 | Kinematic viscosity |
| initial_temperature | f32 | Initial fluid temperature (K) |
| max_timesteps | u64 (optional) | Stop after N timesteps |
| max_time | f64 (optional) | Stop after simulated time (seconds) |
| cfl_number | f32 | CFL condition for adaptive timestep (default: 0.4) |

**BoundaryType**: `Wall | Inflow { velocity: Vec3, temperature: f32 } | Outflow | Periodic`

**Validation rules**:
- `domain_bounds.min < domain_bounds.max` on all axes
- `geometry_file` must exist and be a valid STL
- `particle_spacing > 0`
- `speed_of_sound > 0`
- If `fluid_type` is Water, `initial_temperature` must be > 0 K
- Periodic boundaries must be paired (x_min ↔ x_max, etc.)

### SimulationState

The complete state of a running or paused simulation. Used for
checkpointing (save/resume).

| Field | Type | Description |
|-------|------|-------------|
| config | SimulationConfig | Original config (immutable) |
| particles | ParticleArrays | All particle SoA data |
| sdf | GridSDF | Cached signed distance field |
| timestep | u64 | Current timestep number |
| sim_time | f64 | Current simulated time (seconds) |
| dt | f64 | Current adaptive timestep size |
| error_metrics | ErrorMetrics | Accumulated error tracking |
| force_records | Vec<ForceRecord> | Time-series of force data |

**Checkpoint format**: Binary serialization of SimulationState to
disk. Format TBD — likely `bincode` for simplicity.

### GridSDF

Voxelized signed distance field generated from STL geometry.

The SDF is used for computing surface normals (force extraction) and
for generating frozen boundary particles at setup time. It is NOT
used for penalty-force wall boundaries at runtime.

| Field | Type | Description |
|-------|------|-------------|
| origin | Vec3 | Grid origin (min corner) |
| cell_size | f32 | Voxel edge length |
| dimensions | [u32; 3] | Grid dimensions (nx, ny, nz) |
| distances | Vec<f32> | Flat array of signed distances |

**Query**: For a world-space point p, compute grid indices, trilinear
interpolate the distance value. Gradient (normal) via central
differences on the distance field.

### ForceRecord

Time-stamped force data on the geometry.

| Field | Type | Description |
|-------|------|-------------|
| timestep | u64 | Timestep when recorded |
| sim_time | f64 | Simulated time |
| pressure_force | Vec3 | Net pressure force on geometry |
| viscous_force | Vec3 | Net viscous force on geometry |
| thermal_flux | f32 | Net heat transfer rate (W) |
| per_surface | Vec<SurfaceForce> | Per-triangle-group breakdown |

### SurfaceForce

| Field | Type | Description |
|-------|------|-------------|
| surface_id | u32 | Triangle group identifier |
| pressure_force | Vec3 | Pressure force on this surface |
| viscous_force | Vec3 | Viscous force on this surface |
| thermal_flux | f32 | Heat transfer on this surface |

### BoundaryParticle

Frozen particles placed on domain walls and geometry surfaces at setup
time. They participate in SPH density summation and pressure calculation
but do not move.

| Field | Type | Description |
|-------|------|-------------|
| position | Vec3 (f32×3) | Fixed world-space position |
| mass | f32 | Particle mass (matches fluid particle mass) |
| normal | Vec3 (f32×3) | Outward-facing surface normal |

**Lifecycle**: Created at domain setup. Never moves, never deleted.
Stored in the same SoA structure as fluid particles but flagged as
boundary (e.g., via a `is_boundary: bool` field or a separate array).

### ErrorMetrics

Per-timestep quality tracking.

| Field | Type | Description |
|-------|------|-------------|
| max_density_variation | f32 | Max (rho - rho0) / rho0 |
| energy_conservation | f32 | (E_total - E_initial) / E_initial |
| mass_conservation | f32 | (M_total - M_initial) / M_initial |

### ReferenceTest

Definition of a validation test case.

| Field | Type | Description |
|-------|------|-------------|
| name | string | Test case name |
| description | string | What is being validated |
| config | SimulationConfig | Simulation setup |
| expected | ExpectedResult | Analytical/published result |
| tolerance | f32 | Acceptable relative error |

### ExpectedResult

| Field | Type | Description |
|-------|------|-------------|
| metric | enum { Pressure, Force, Temperature, Conservation } | What to measure (Temperature is for air temperature validation only) |
| location | Vec3 or "global" | Where to measure |
| value | f64 | Expected value |
| unit | string | Physical unit |

## State Transitions

### Particle Fluid Type

Particles have a fixed `fluid_type` (Water or Air) assigned at
creation. Fluid type does not change during simulation. Water uses
Tait EOS; Air uses ideal gas EOS.

### Simulation Lifecycle

```
Configured ──[start]──> Running ──[pause]──> Paused
                           │                    │
                           │              [resume]──> Running
                           │                    │
                           │              [checkpoint]──> Paused (saved)
                           │
                      [complete/error]──> Finished
                           │
                      [instability]──> Error (paused with diagnostics)
```
