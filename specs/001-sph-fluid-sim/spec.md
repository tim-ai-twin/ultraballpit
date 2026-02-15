# Feature Specification: SPH Fluid Simulation

**Feature Branch**: `001-sph-fluid-sim`
**Created**: 2026-02-15
**Status**: Draft
**Input**: User description: "Particle-based fluid simulation (water and air) around 3D STL geometry with interactive web visualization and force reporting. WCSPH method targeting engineer/artist users with ballpark accuracy and error bounds. Subsonic, normal environmental conditions only."

## Work Streams

This feature comprises four distinct work streams that can progress
in parallel after shared foundations are in place:

1. **Visualization (Frontend)** -- Interactive web UI served on
   localhost for viewing simulation state, camera controls, debug
   overlays, and force heatmaps.
2. **Simulation Kernel (Backend -- Compute)** -- The SPH computation
   core. Executes timesteps for a given subdomain. Designed as a
   separable, swappable component with multiple implementation
   variants (CPU, Metal/GPU, future accelerators).
3. **Orchestration (Backend -- Control)** -- Sets up simulations,
   converts STL geometry to SDF, defines domain bounds, coordinates
   execution, and aggregates results.
4. **Reference Tests & Validation** -- Identifies canonical test
   cases with known analytical solutions, builds them as automated
   validation suites, and uses them to verify simulation correctness
   at each development stage.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Run a Basic Fluid Simulation on Simple Geometry (Priority: P1)

An engineer or artist selects a fluid type (water or air), loads a
simple geometry (starting with small ~1cm boxes and basic shapes),
configures basic simulation parameters (particle count, domain size,
initial conditions, gravity), and starts a WCSPH fluid simulation.
They watch particles flow around the geometry in real time through an
interactive web visualization served on localhost. They can orbit, pan,
and zoom the camera to inspect the flow from any angle.

The system supports two primary fluid modes:
- **Water mode**: Liquid water under gravity, flowing and pooling
  around geometry.
- **Air mode**: Gas-phase air flowing through and around geometry at
  normal atmospheric conditions with temperature tracking.

**Why this priority**: This is the foundational capability. Without
fluid flowing around geometry and a way to see it, nothing else
matters. This story delivers the core value proposition end-to-end
and exercises all four work streams at their simplest level.

**Independent Test**: Can be fully tested by running a simulation of
water settling under gravity in a small box (~1cm), or air flowing
past a small obstacle. Delivers value as a standalone fluid
visualization tool.

**Acceptance Scenarios**:

1. **Given** a simple box geometry and water selected as the fluid, **When** the user starts the simulation, **Then** water particles fall under gravity, pool at the bottom, and interact with the box walls, visible in the web viewer within 5 seconds.
2. **Given** a simple obstacle geometry and air selected as the fluid, **When** the user starts the simulation, **Then** air particles flow around the obstacle at subsonic speeds with visible flow patterns.
3. **Given** a running simulation displayed in the web viewer, **When** the user clicks and drags or scrolls, **Then** the camera orbits, pans, or zooms smoothly without interrupting the simulation.
4. **Given** a running simulation, **When** particles contact geometry surfaces, **Then** they deflect realistically (no particles passing through walls, no explosive behavior).
5. **Given** no geometry loaded, **When** the user starts a simulation, **Then** the system displays a clear error message requesting geometry input.

---

### User Story 2 - Reference Tests and Validation Cases (Priority: P2)

The development team runs a suite of reference tests with known
analytical solutions to validate that the simulation kernel produces
correct results. These tests use simple geometries (~1cm scale) and
well-understood boundary conditions. Each test has a pass/fail
criterion based on comparison with the analytical or published
experimental result.

Example reference tests (to be expanded during development):
- **Pressure equalization**: Gas in a sealed vessel reaches uniform
  pressure from a non-uniform initial state.
- **Gravity settling**: Water settles to the lowest point in a
  container under gravity.
- **Hydrostatic pressure**: Pressure at the bottom of a water column
  matches rho * g * h within error bounds.
- **Stokes drag**: Drag force on a sphere in slow uniform flow matches
  analytical Stokes drag formula.
- **Conservation**: Total mass and total energy are conserved within
  documented error bounds across all tests.

**Why this priority**: Reference tests are the foundation of trust.
Without them, you cannot tell if the simulation is producing
meaningful results. They should be built alongside (or slightly
behind) the core simulation kernel.

**Independent Test**: Each reference test is independently runnable
and has its own pass/fail criterion. The full suite can run as part
of CI.

**Acceptance Scenarios**:

1. **Given** the pressure equalization test case, **When** the simulation runs to steady state, **Then** pressure variation across the vessel is below 1% of mean pressure.
2. **Given** the gravity settling test case, **When** water is released in a container, **Then** all water particles are within one particle diameter of the container floor at steady state.
3. **Given** the hydrostatic pressure test case, **When** a water column is at rest, **Then** bottom pressure matches rho * g * h within 5%.
4. **Given** any reference test, **When** it completes, **Then** mass conservation error is below 0.1% and energy conservation error is below 5%.

---

### User Story 3 - Force Extraction and Reporting (Priority: P3)

The user runs a simulation to completion (or pauses at a steady state)
and extracts surface forces acting on the STL geometry. They can view
force data as a visual overlay on the geometry (pressure heatmap),
retrieve time-series force data through an API, or generate a summary
report with peak forces, time-averaged values, and error bounds.

**Why this priority**: This is the engineering payoff -- turning a
visualization into a useful analysis tool. It depends on stable
simulation (US1) and ideally validated results (US2).

**Independent Test**: Can be tested by running the Stokes drag
reference test and comparing computed drag force against the
analytical value. The force API returns data, and the report includes
error bounds.

**Acceptance Scenarios**:

1. **Given** a completed or paused simulation, **When** the user requests force data, **Then** the system returns pressure and viscous force vectors per surface region of the geometry.
2. **Given** force data available, **When** the user enables force overlay in the viewer, **Then** a pressure/force heatmap is displayed on the geometry surface.
3. **Given** a simulation run, **When** the user queries the force API, **Then** the response includes time-series data with timestamps, force vectors, and aggregate statistics (peak, mean, RMS).
4. **Given** force results, **When** the user generates a report, **Then** the report includes error bounds (compressibility error, spatial resolution estimate, energy conservation deviation).

---

### User Story 4 - Simulation Diagnostics and Error Bounds (Priority: P4)

The user monitors simulation quality in real time through a debug
overlay showing frame timing, particle count, density variation
(compressibility error), and energy conservation. Error bounds are
tracked per-timestep and summarized at the end of a run.

**Why this priority**: Error bounds and diagnostics are what make this
tool trustworthy for engineering use. They build on all prior stories.

**Independent Test**: Can be tested by running any simulation and
toggling the debug overlay. Diagnostics should display non-zero,
plausible values that update each frame.

**Acceptance Scenarios**:

1. **Given** a running simulation, **When** the user enables the debug overlay, **Then** frame time, particle count, max density variation, and energy conservation error are displayed and update in real time.
2. **Given** a completed simulation, **When** the user requests a quality summary, **Then** the system reports max/mean error bounds for compressibility and energy conservation across the full run.

---

### User Story 5 - Distributed Parallel Execution (Priority: P5)

The user configures a simulation that exceeds single-machine capacity
(high particle count or large domain) and launches it across multiple
cloud instances. The orchestration layer automatically decomposes the
domain, distributes subdomains to compute instances, coordinates
boundary data exchange, and aggregates results. The user visualizes
the combined result in the same web interface.

**Why this priority**: Cloud parallelism is an optimization for scale,
not a core feature. The local single-machine experience must work
first.

**Independent Test**: Can be tested by running the same simulation
locally and on 2+ instances, comparing results for consistency.

**Acceptance Scenarios**:

1. **Given** a simulation configuration and compute instance endpoints, **When** the user starts a distributed simulation, **Then** the orchestration layer partitions the domain and each instance begins processing its subdomain.
2. **Given** a running distributed simulation, **When** the user views it in the web interface, **Then** particles from all subdomains appear seamlessly as a single continuous simulation.
3. **Given** a distributed and a local simulation of the same scene, **When** force results are compared, **Then** they agree within documented numerical tolerance bounds.

---

### Edge Cases

- What happens when the STL file has non-manifold geometry or holes? The system should warn the user and attempt to generate an SDF anyway, or reject with a clear error if the geometry is too broken.
- What happens when particles escape the simulation domain? Particles that leave the bounding box should be removed or recycled, with a warning if the loss rate is high.
- What happens if the simulation becomes unstable (exploding particles)? The system should detect runaway velocity/density values, pause the simulation, and report the instability with suggested parameter adjustments.
- What happens when the user loads a very large STL (millions of triangles)? The SDF generation should handle it but may take longer; a progress indicator should be shown.
- What happens when the density ratio between water and air (~1000x) causes numerical instability? The system should detect this and warn the user.

## Clarifications

### Session 2026-02-15

- Q: How should domain boundaries work -- closed walls only, or support open inflow/outflow? -> A: Closed walls by default on all faces; user can mark individual faces as inflow, outflow, or periodic.
- Q: How does the user configure and launch a simulation? -> A: JSON config files define simulations. Web UI selects from available configs and visualizes in-progress work and results. Web UI does not edit configs.
- Q: Should simulations support save/resume (checkpointing)? -> A: Manual checkpoint -- user can save simulation state to disk and resume from that checkpoint later.
- Q: How are heat sources specified? -> A: Boundary temperatures -- user assigns fixed temperatures to geometry surfaces in the JSON config (e.g., pot wall = 120C, cold glass = 5C). Volumetric heat sources are out of scope.
- Q: What's the target upper particle count for single-machine (MacBook) execution? -> A: ~10K particles on CPU for fast iteration, ceiling ~50-100K on CPU. Metal GPU acceleration deferred.

### Session 2026-02-15 (Review)

- Q: Should phase transitions (boiling/condensation) be in scope? -> A: No. Water and air coexist but do not transition between phases. Phase change removed to reduce complexity.
- Q: How many particles should be streamed to the web viewer? -> A: ~5% subsample for visualization. Full particle set used for simulation accuracy and metrics.
- Q: What is the target particle count for M1 MacBook Air? -> A: ~10K for fast iteration. CPU ceiling ~50-100K. GPU acceleration deferred.
- Q: Should wall boundaries use SDF penalty forces or boundary particles? -> A: Boundary particles (frozen particles on walls/surfaces). More physically accurate density near walls.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: System MUST accept STL files (binary and ASCII format) as geometry input and convert them to a signed distance field (SDF) for boundary representation.
- **FR-002**: System MUST simulate fluid particles using the Weakly Compressible SPH (WCSPH) method with configurable speed-of-sound parameter controlling compressibility error. Simulation is limited to subsonic flow at normal environmental conditions.
- **FR-003**: System MUST support two fluid types: water (liquid phase only) and air (gas phase with temperature). Users MUST be able to run simulations with water only, air only, or mixed (both water and air in the same domain). Phase transitions between water and air are out of scope.
- **FR-004**: System MUST track per-particle state including mass, position, velocity, temperature, pressure, density, and fluid_type (water or air).
- **FR-006**: System MUST include gravity as a configurable body force.
- **FR-006a**: Domain boundaries MUST default to closed walls on all faces. Users MAY mark individual domain faces as inflow, outflow, or periodic to support open-flow scenarios (e.g., air flowing past an obstacle).
- **FR-007**: Simulations MUST be defined via JSON configuration files specifying fluid type, geometry, domain bounds, boundary conditions, initial conditions, gravity vector, solver parameters, and heat sources.
- **FR-007a**: System MUST render an interactive 3D visualization in a web browser served on localhost, with camera controls (orbit, pan, zoom) and visually distinct representations for water and air. The web UI MUST allow selecting from available JSON configs and visualizing in-progress and completed simulation results. The web UI is read-only with respect to configuration -- it does not edit config files. The WebSocket viewer streams a configurable subsample (~5%) of particles for visualization. The full particle set is used for simulation accuracy and force/metric calculations.
- **FR-008**: System MUST compute and report surface forces (pressure and viscous) on the geometry, with time-series data accessible through an API.
- **FR-009**: System MUST track and report error bounds: compressibility (max density variation), energy conservation, and mass conservation.
- **FR-010**: System MUST expose simulation diagnostics (frame timing, particle count, error metrics) through a debug overlay in the web viewer.
- **FR-011**: The simulation kernel MUST be a separable component with a defined interface, enabling multiple implementation variants (CPU, Metal/GPU, future accelerators) without changing the orchestration or visualization layers.
- **FR-012**: The orchestration layer MUST handle geometry conversion (STL to SDF), domain decomposition, simulation setup, and coordination of parallel compute instances.
- **FR-013**: System MUST run locally on a MacBook (Apple Silicon M1). Initial target: ~10K particles on CPU for fast iteration. CPU kernel should handle up to ~50-100K particles. Metal GPU acceleration is a future optimization.
- **FR-015**: System MUST include a suite of automated reference tests with known analytical solutions that validate simulation correctness.
- **FR-016**: System MUST generate summary reports including peak/mean forces, error bounds, and validation metrics.
- **FR-017**: System MUST detect simulation instabilities (runaway values) and pause with actionable feedback.
- **FR-018**: System MUST start with very small geometries (~1cm scale simple boxes and shapes) for initial development and testing, scaling to larger/complex STL geometries as the system matures.
- **FR-019**: System MUST support manual checkpointing -- the user can save full simulation state to disk at any point and resume from that checkpoint later. Auto-checkpointing is out of scope for initial release.
- **FR-020**: System MUST use boundary particles (frozen particles on domain walls and geometry surfaces) for wall boundary enforcement, rather than SDF penalty forces. The SDF is retained for computing surface normals and for force extraction.
- **FR-021**: The simulation kernel step function MUST be structured as distinct sequential phases (neighbor search, density, forces, integration) to enable future GPU porting.

### Key Entities

- **Particle**: The fundamental simulation element. Carries mass, position, velocity, temperature, density, pressure, and fluid_type (water or air). Participates in neighbor searches and force calculations.
- **STL Geometry**: User-provided 3D boundary mesh. Converted to SDF for runtime collision handling and surface normal computation. Receives force accumulations from nearby particles. May have fixed-temperature thermal boundary conditions assigned per surface.
- **Signed Distance Field (SDF)**: Voxelized representation of the geometry boundary. Used for computing surface normals and for force extraction. Resolution matched to particle spacing.
- **Simulation Configuration**: A JSON file defining fluid type, particle count, domain bounds, boundary conditions per face (wall/inflow/outflow/periodic, default: wall), initial conditions, gravity vector, heat sources, speed-of-sound, and timestep constraints. This is the primary interface for setting up simulations.
- **Simulation Kernel**: The compute component that executes SPH timesteps. Has a defined interface so implementations can be swapped (CPU, GPU, distributed).
- **Orchestration Controller**: Manages simulation lifecycle -- geometry preprocessing, domain setup, kernel dispatch, result aggregation, and distributed coordination.
- **Force Record**: Time-stamped force data (pressure, viscous) accumulated per surface region of the geometry.
- **Error Metrics**: Per-timestep quality measurements (density variation, energy conservation, mass conservation).
- **Reference Test**: A canonical simulation scenario with known analytical solution, boundary conditions, geometry, and pass/fail criteria.
- **Boundary Particle**: A frozen (non-moving) particle placed on domain walls and geometry surfaces to enforce wall boundaries. Participates in SPH density and force calculations but does not move.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Users can go from selecting a fluid and geometry to seeing flow in the web viewer in under 30 seconds for small-scale (~1cm) simulations with ~10K particles.
- **SC-002**: The web visualization maintains at least 30 FPS display rate during interactive camera manipulation.
- **SC-003**: Compressibility error (max density variation) stays below 3% for default simulation parameters.
- **SC-004**: Computed drag force on a sphere in uniform flow agrees with published Stokes/empirical drag correlations within 15% for the intended Reynolds number range.
- **SC-006**: Total energy (kinetic + potential + thermal) conservation error remains below 5% over a full simulation run.
- **SC-007**: Mass conservation error remains below 0.1% across all simulation runs.
- **SC-008**: Hydrostatic pressure at the bottom of a water column matches rho * g * h within 5%.
- **SC-009**: All reference tests in the automated suite pass on every CI run.
- **SC-010**: Force API response time is under 500ms for queries spanning up to 10,000 timesteps.
- **SC-011**: Distributed simulation across 2+ instances produces force results within 2% of equivalent single-instance results. (P5 priority -- deferred until US5.)
- **SC-012**: 90% of engineer/artist users can configure and run a basic simulation without consulting documentation beyond the in-app interface.

### Assumptions

- Users have access to macOS with Apple Silicon (M1 or later) for local execution. Cloud instances with appropriate GPU support for distributed runs.
- STL files provided by users are reasonably well-formed (closed, manifold). Mildly defective meshes should be handled gracefully; severely broken meshes may be rejected.
- The target audience has basic familiarity with fluid simulation concepts (particle count, timestep) but does not need CFD expertise.
- Initial release targets water (liquid only) and air only. Phase transitions (boiling, condensation) are explicitly out of scope for this release. Multi-species fluids, combustion, and solid-phase interactions are also out of scope.
- All simulations are subsonic, low-speed, everyday physics (e.g., water flowing around objects, air flowing around an obstacle). Shock waves, sound propagation, detonations, and compressible high-Mach-number flows are out of scope.
- The force API serves localhost clients (same machine or local network). Public internet exposure is out of scope.
- Initial development uses simple small-scale geometries (~1cm boxes and shapes). Complex large-scale STL geometries are a scaling concern addressed after core correctness is established.
- The simulation kernel starts as a CPU implementation targeting ~10K particles for fast iteration, with a ceiling of ~50-100K on CPU. Metal/GPU acceleration is a planned optimization, not a launch requirement.
- The web viewer displays a subsample (~5%) of particles. Full resolution is used for computation only.
