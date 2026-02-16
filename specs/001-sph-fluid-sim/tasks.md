# Tasks: SPH Fluid Simulation

**Input**: Design documents from `/specs/001-sph-fluid-sim/`
**Prerequisites**: plan.md, spec.md, data-model.md, research.md, contracts/
**Branch**: `001-sph-fluid-sim`
**Generated**: 2026-02-15

**Tests**: Test tasks are included for the reference test suite (US2) and early kernel unit tests (Phase 3). Unit tests for individual modules are expected to be written alongside implementation per constitution principle III (Test Discipline).

**Organization**: Tasks are grouped by phase and user story to enable independent implementation and testing. Start with very small simulations (~1cm, <1K particles) on CPU for fast iteration. Target ~10K particles on M1 MacBook Air CPU at steady state.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies on incomplete tasks)
- **[Story]**: Which user story this task belongs to (e.g., US1, US2, US3)
- Include exact file paths in descriptions

---

## Phase 1: Setup (Project Scaffolding)

**Purpose**: Create the Rust workspace, frontend project, build orchestration, and sample geometry files. No business logic.

- [ ] T001 Create Rust workspace with three crates (kernel, orchestrator, server) in backend/Cargo.toml and backend/crates/kernel/Cargo.toml, backend/crates/orchestrator/Cargo.toml, backend/crates/server/Cargo.toml
- [ ] T002 [P] Initialize frontend project with Vite + TypeScript strict mode in frontend/package.json, frontend/tsconfig.json, frontend/src/index.html
- [ ] T003 [P] Create justfile with recipes: build, build-backend, build-frontend, serve, test, test-backend, test-frontend, test-e2e, test-reference, bench, watch
- [ ] T004 [P] Create simple STL geometry files for development: 1cm box (geometries/box-1cm.stl), 1cm sphere (geometries/sphere-1cm.stl) -- binary STL format, programmatically generated or hand-crafted
- [ ] T005 [P] Add Rust dependencies to workspace Cargo.toml and crate-level Cargo.toml files: nalgebra, serde, serde_json, tracing, tracing-subscriber, bytemuck (kernel); nom_stl, mesh_to_sdf (orchestrator); axum, tokio, tower-http (server). Do NOT include if97 or wgpu.

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Core data structures, algorithms, and infrastructure that MUST be complete before any user story. These are the building blocks all user stories share.

**CRITICAL**: No user story work can begin until this phase is complete.

- [ ] T006 [P] Implement ParticleArrays struct-of-arrays data structures with SEPARATE x/y/z position arrays (not Vec3) for GPU-readiness and SIMD, plus velocity (vx/vy/vz), acceleration (ax/ay/az), density, pressure, mass, temperature, fluid_type arrays in backend/crates/kernel/src/particle.rs. No phase, no phase_progress, no per-particle smoothing_length.
- [ ] T007 [P] Define FluidType enum (Water, Air) in backend/crates/kernel/src/particle.rs. No Phase enum.
- [ ] T008 [P] Implement Wendland C2 smoothing kernel W(r,h) and its gradient in backend/crates/kernel/src/sph.rs. Wendland C2 chosen over cubic spline for better stability.
- [ ] T009 Implement uniform-grid spatial hash for fixed-radius neighbor search using SORTED-INDEX + CELL-OFFSET arrays (not HashMap) for GPU-friendly memory layout (cell size = smoothing length, 27-cell neighborhood query) in backend/crates/kernel/src/neighbor.rs
- [ ] T010 [P] Implement Tait equation of state for liquid water and ideal gas EOS for air in backend/crates/kernel/src/eos.rs. Water and air coexist but do not transition between phases.
- [ ] T011 [P] Define SimulationKernel trait with step() method structured as distinct phases (neighbor search, density, forces, integrate) and readback method for particle state in backend/crates/kernel/src/lib.rs
- [ ] T012 Implement SimulationConfig struct with serde deserialization and validation (domain bounds, particle spacing, boundary conditions, fluid type, gravity, smoothing length, etc.) matching contracts/config-schema.json in backend/crates/orchestrator/src/config.rs. No surface_temperatures field.
- [ ] T013 [P] Implement STL parsing via nom_stl and SDF generation via mesh_to_sdf (GridSDF struct with origin, cell_size, dimensions, distances, surface_ids) in backend/crates/orchestrator/src/geometry.rs. SDF used for normals and force extraction, NOT for penalty boundary forces.
- [ ] T014 Implement domain setup: compute particle spacing grid, fill domain with particles of the correct fluid type, apply initial conditions from config, and GENERATE BOUNDARY PARTICLES on walls and geometry surfaces in backend/crates/orchestrator/src/domain.rs
- [ ] T015 [P] Set up tracing subscriber with structured logging in backend/crates/server/src/main.rs
- [ ] T016 [P] Define transport protocol message types (TypeScript) matching contracts/transport-protocol.md in frontend/src/types/protocol.ts. Use fluid_type (not phase), no phase_progress field.

**Checkpoint**: Foundation ready -- all core data structures, neighbor search, EOS, config parsing, geometry pipeline, boundary particle generation, and domain initialization work. User story implementation can now begin.

---

## Phase 3: US1 -- Run a Basic Fluid Simulation (Priority: P1) MVP

**Goal**: End-to-end: select a config, start a simulation, see particles flowing around geometry in the web viewer with camera controls. Supports water (gravity settling) and air (flow past obstacle) modes. Fixed global smoothing length. Boundary particles enforce walls. Velocity Verlet integrator.

**Independent Test**: Run water-box-1cm config -- water particles fall under gravity and pool at the bottom of a 1cm box, visible in the web viewer. Run air-obstacle-1cm -- air in a closed box with an obstacle.

### Backend Kernel (US1)

- [ ] T017 [US1] Implement SPH density summation (sum over neighbors using Wendland C2 smoothing kernel with fixed global smoothing length) in backend/crates/kernel/src/sph.rs
- [ ] T018 [US1] Implement pressure force computation from density + EOS in backend/crates/kernel/src/sph.rs
- [ ] T019 [US1] Implement viscous force computation (artificial viscosity) in backend/crates/kernel/src/sph.rs
- [ ] T020 [US1] Implement gravity body force application in backend/crates/kernel/src/sph.rs
- [ ] T021 [US1] Implement boundary particle wall enforcement in backend/crates/kernel/src/boundary.rs -- fluid particles interact with frozen boundary particles via repulsive SPH forces, NOT SDF penalty forces
- [ ] T022 [US1] Implement adaptive timestep via CFL condition (dt = cfl_number * h / max_velocity) in backend/crates/kernel/src/sph.rs
- [ ] T023 [US1] Implement CpuKernel struct implementing SimulationKernel trait -- full step() method with VELOCITY VERLET integrator, structured as DISTINCT PHASES: neighbor search, density summation, force computation, integration in backend/crates/kernel/src/lib.rs

### Early Kernel Unit Tests (US1)

- [ ] T024 [P] [US1] Two-particle symmetry test: place 2 particles, verify Newton's 3rd law (equal and opposite forces) and momentum conservation in backend/crates/kernel/tests/two_particle.rs
- [ ] T025 [P] [US1] Kernel normalization test: 27 particles on 3x3x3 lattice at rest density spacing, verify computed density = rho_0 +/- 2% in backend/crates/kernel/tests/kernel_normalization.rs
- [ ] T026 [US1] Hydrostatic column test: 48 particles in a column, verify bottom pressure = rho*g*h +/- 10% in backend/crates/kernel/tests/hydrostatic.rs

### Backend Orchestrator (US1)

- [ ] T027 [US1] Implement SimulationRunner with lifecycle (start, pause, resume, step loop in background thread, status tracking) in backend/crates/orchestrator/src/runner.rs
- [ ] T028 [US1] Wire orchestrator: load config, parse STL, generate SDF, generate boundary particles, init domain, create CpuKernel, hand to SimulationRunner in backend/crates/orchestrator/src/lib.rs

### Backend Server (US1)

- [ ] T029 [US1] Implement axum HTTP server with static file serving (tower-http ServeDir for frontend/dist/) and API router in backend/crates/server/src/main.rs
- [ ] T030 [US1] Implement REST endpoints: GET /api/configs (list configs dir), GET /api/configs/{name}, POST /api/simulations, GET /api/simulations/{id}, POST /api/simulations/{id}/pause, POST /api/simulations/{id}/resume in backend/crates/server/src/api.rs
- [ ] T031 [US1] Implement WebSocket endpoint ws://localhost:{PORT}/ws/simulation with binary frame streaming per transport-protocol.md (SimInfo on connect, Frame each step, SimStatus on state change) with PARTICLE SUBSAMPLING (~5% of particles per frame) in backend/crates/server/src/ws.rs
- [ ] T032 [US1] Implement WebSocket flow control: drop intermediate frames if client send buffer exceeds 5 frames, send only latest in backend/crates/server/src/ws.rs

### Frontend (US1)

- [ ] T033 [P] [US1] Create Three.js scene with PerspectiveCamera, ambient + directional lighting, and OrbitControls in frontend/src/viewer/scene.ts
- [ ] T034 [P] [US1] Implement WebSocket client: connect, parse binary SimInfo/Frame/SimStatus messages per transport-protocol.md, handle subsampled particle frames, expose particle data to renderer in frontend/src/transport/client.ts
- [ ] T035 [US1] Implement particle renderer using THREE.Points with custom ShaderMaterial (position from Float32Array, color by fluid_type and temperature) in frontend/src/viewer/particles.ts
- [ ] T036 [US1] Implement STL geometry mesh rendering (load geometry from server, render as semi-transparent wireframe/solid) in frontend/src/viewer/geometry.ts
- [ ] T037 [US1] Implement config selector UI: fetch GET /api/configs, display list, allow selection in frontend/src/ui/config-list.ts
- [ ] T038 [US1] Implement simulation controls: Start button (POST /api/simulations), Pause/Resume buttons in frontend/src/ui/controls.ts
- [ ] T039 [US1] Wire up main.ts: initialize scene, create WebSocket client, connect config selector, start sim, stream particles, render loop in frontend/src/main.ts

### Configs & Integration (US1)

- [ ] T040 [P] [US1] Create water-box-1cm.json config (water in 1cm closed box, ~1K particles, gravity, all-wall boundaries) in configs/water-box-1cm.json
- [ ] T041 [P] [US1] Create air-obstacle-1cm.json config (air in 1cm closed box with obstacle, ~1K particles, all-wall boundaries, no inflow/outflow -- deferred to polish) in configs/air-obstacle-1cm.json
- [ ] T042 [US1] End-to-end integration test: `just serve`, open browser, select water-box-1cm, verify particles fall and pool, verify camera controls work

**Checkpoint**: MVP complete. User can select a config, start a water or air simulation, and see particles in the browser with interactive camera controls. Boundary particles enforce walls. Velocity Verlet integrator. ~10K particles feasible on M1 MacBook Air CPU.

---

## Phase 4: US2 -- Reference Tests and Validation (Priority: P2)

**Goal**: Automated test suite with known analytical solutions to validate simulation correctness. Pass/fail criteria based on comparison with expected results. Moved earlier in the pipeline (before force extraction) to catch regressions sooner.

**Independent Test**: Each reference test is independently runnable with its own pass/fail criterion. `just test-reference` runs the full suite.

**Dependencies**: Requires US1 complete. Stokes drag test depends on US3 (force extraction) -- noted below.

### Framework

- [ ] T043 [P] [US2] Create reference test framework: ReferenceTest and ExpectedResult structs, test runner that loads case JSON, runs simulation to completion, compares results against expected values with tolerances in backend/reference-tests/src/lib.rs
- [ ] T044 [P] [US2] Create reference test binary entry point that discovers and runs all cases from backend/reference-tests/cases/ in backend/reference-tests/src/main.rs

### Test Cases

- [ ] T045 [P] [US2] Create gravity settling test case: water released in container, verify all particles within one particle diameter of floor at steady state in backend/reference-tests/cases/gravity-settling.json
- [ ] T046 [P] [US2] Create hydrostatic pressure test case (larger scale than kernel unit test): water column at rest, verify bottom pressure matches rho*g*h within 5% in backend/reference-tests/cases/hydrostatic-pressure.json
- [ ] T047 [P] [US2] Create pressure equalization test case: gas in sealed vessel from non-uniform initial state, verify pressure variation <1% at steady state in backend/reference-tests/cases/pressure-equalization.json
- [ ] T048 [US2] Create conservation validation: verify mass conservation error <0.1% and energy conservation error <5% across all test runs in backend/reference-tests/src/lib.rs

### Integration

- [ ] T049 [US2] Add `just test-reference` recipe to justfile and verify all implemented test cases pass
- [ ] T050 [US2] Verify all reference tests pass with documented tolerances; document any failing tests with analysis. Note: Stokes drag test deferred to Phase 5 (depends on US3 force extraction).

**Checkpoint**: Reference test suite runs via `just test-reference`. Gravity settling, hydrostatic pressure, pressure equalization, and conservation tests pass. No boiling onset test (phase change removed).

---

## Phase 5: US3 -- Force Extraction and Reporting (Priority: P3)

**Goal**: Compute surface forces on geometry, expose via API, display as heatmap overlay, generate summary reports.

**Independent Test**: Run Stokes drag reference test, compare computed drag force against analytical value. Force API returns data.

**Dependencies**: Requires US1 complete. Stokes drag test uses US2 framework.

### Backend (US3)

- [ ] T051 [US3] Implement surface force accumulation: sum pressure and viscous contributions from nearby fluid particles onto geometry surface regions (using SDF surface_ids and normals) in backend/crates/orchestrator/src/force.rs
- [ ] T052 [US3] Implement ForceRecord and SurfaceForce storage: accumulate per-timestep force time-series in SimulationRunner in backend/crates/orchestrator/src/runner.rs
- [ ] T053 [US3] Implement GET /api/simulations/{id}/forces endpoint with query params (from_timestep, to_timestep, aggregation: raw|mean|peak) returning time-series + aggregates in backend/crates/server/src/api.rs
- [ ] T054 [US3] Implement GET /api/simulations/{id}/report endpoint returning summary report (peak/mean forces, diagnostics) in backend/crates/server/src/api.rs

### Frontend (US3)

- [ ] T055 [US3] Implement force/pressure heatmap overlay on geometry surface (fetch force data from API, map pressure magnitude to color gradient on STL mesh) in frontend/src/viewer/overlays.ts
- [ ] T056 [US3] Add force overlay toggle control in frontend/src/ui/controls.ts

### Validation (US3)

- [ ] T057 [US3] Create Stokes drag reference test case: drag force on sphere in uniform flow matches analytical Stokes drag within 15% in backend/reference-tests/cases/stokes-drag.json
- [ ] T058 [US3] Verify force API returns correct data: run simulation, query forces endpoint, confirm response matches transport contract

**Checkpoint**: Force data computable, queryable via API, and visible as heatmap overlay on geometry. Stokes drag reference test passes.

---

## Phase 6: US4 -- Simulation Diagnostics and Error Bounds (Priority: P4)

**Goal**: Real-time debug overlay showing frame timing, particle count, density variation, energy conservation, and mass conservation. Error bounds tracked per-timestep and summarized. Instability detection auto-pauses simulation.

**Independent Test**: Run any simulation, toggle debug overlay, verify non-zero plausible values that update each frame.

**Dependencies**: Requires US1 complete. No thermodynamic deviation tracking (phase change removed).

### Backend (US4)

- [ ] T059 [P] [US4] Implement ErrorMetrics computation per timestep (max density variation, energy conservation error, mass conservation error) integrated into CpuKernel step() in backend/crates/kernel/src/sph.rs. No thermodynamic deviation metric.
- [ ] T060 [US4] Implement instability detection: monitor for runaway velocity/density values, auto-pause simulation with actionable error message in backend/crates/orchestrator/src/runner.rs
- [ ] T061 [US4] Implement Diagnostics WebSocket message (tag 0x03) per transport-protocol.md: frame_time_ms, max_density_variation, energy/mass conservation errors, dt, particle_count in backend/crates/server/src/ws.rs
- [ ] T062 [US4] Implement WebSocket Command handling for enable/disable diagnostics (tag 0x80, commands 0x04/0x05) in backend/crates/server/src/ws.rs

### Frontend (US4)

- [ ] T063 [US4] Implement debug overlay UI: parse Diagnostics binary messages, display frame time, particle count, max density variation, energy conservation error as HUD overlay in frontend/src/viewer/overlays.ts
- [ ] T064 [US4] Add diagnostics toggle button (sends enable/disable command via WebSocket) in frontend/src/ui/controls.ts

### Validation (US4)

- [ ] T065 [US4] Verify debug overlay shows live updating metrics during a running simulation; values are non-zero and plausible

**Checkpoint**: Debug overlay shows real-time diagnostics. Instability detection pauses simulation with feedback. Error bounds tracked and reported.

---

## Phase 7: US5 -- Distributed Parallel Execution (Priority: P5)

**Goal**: Simulations exceeding single-machine capacity run across multiple cloud instances with automatic domain decomposition and result aggregation.

**Independent Test**: Run the same simulation locally and on 2+ instances; compare results for consistency within documented tolerance.

**Dependencies**: Requires US1 complete. Benefits from US2 (reference tests) and US3 (force comparison) for validation.

### Backend (US5)

- [ ] T066 [US5] Design and implement domain decomposition algorithm (split AABB into subdomains along longest axis, assign to instances) in backend/crates/orchestrator/src/domain.rs
- [ ] T067 [US5] Implement subdomain boundary particle exchange protocol (nearest-neighbor ghost particle communication between adjacent subdomains) in backend/crates/orchestrator/src/domain.rs
- [ ] T068 [US5] Implement multi-instance coordination: orchestrator dispatches subdomains to remote kernel instances, synchronizes timesteps, collects results in backend/crates/orchestrator/src/runner.rs
- [ ] T069 [US5] Implement result aggregation from distributed subdomains (merge particle arrays, combine force records) in backend/crates/orchestrator/src/lib.rs

### Validation (US5)

- [ ] T070 [US5] Verify distributed simulation (2 instances) produces force results within 2% of equivalent single-instance results
- [ ] T071 [US5] Verify web viewer displays seamless combined particle field from distributed subdomains

**Checkpoint**: Distributed execution works across multiple instances. Results match single-instance within tolerance. Viewer shows seamless combined output.

---

## Phase 8: US6 -- Standard SPH Validation Benchmarks (Priority: P6)

**Goal**: Run industry-standard SPH validation benchmarks with published experimental or analytical reference data at higher resolution than the fast smoke tests. These benchmarks prove the simulation is physically correct, not just stable.

**Independent Test**: Each benchmark has quantitative pass/fail criteria based on published data. `just test-benchmarks` runs the full suite.

**Dependencies**: Requires US1 complete. Benefits from US2 framework (reference test infrastructure), US3 (force extraction for drag), and accuracy fixes. These benchmarks use 50k-500k particles and take minutes to run, so they are scheduled after all other development work.

**References**:
- SPHERIC SPH benchmark suite (spheric-sph.org)
- Martin & Moyce 1952 (dam break experimental data)
- Morris et al. 1997 (Poiseuille flow analytical solution)
- Adami et al. 2012 (boundary handling validation)

### Benchmark Configs & Geometry

- [ ] T080 [P] [US6] Create dam break geometry: 2D slab domain (5cm × 10cm × 1-particle-thick, spacing 0.5mm) with water column occupying left quarter, open right side in configs/dam-break-2d.json and geometries/dam-break-slab.stl
- [ ] T081 [P] [US6] Create high-resolution hydrostatic config: water column (2cm × 10cm × 2cm, spacing 0.2mm, 50+ layers, all-wall boundaries) in configs/hydrostatic-hires.json
- [ ] T082 [P] [US6] Create Poiseuille flow config: channel flow between parallel plates (10cm × 1cm × 1-particle-thick, spacing 0.2mm, periodic in flow direction, pressure gradient body force, wall top/bottom) in configs/poiseuille-2d.json
- [ ] T083 [P] [US6] Create standing wave config: periodic domain (10cm × 5cm × 1-particle-thick, spacing 0.5mm, periodic x boundaries, wall bottom, open top) with small-amplitude initial surface displacement in configs/standing-wave.json

### Benchmark Reference Data

- [ ] T084 [P] [US6] Encode Martin & Moyce 1952 dam break experimental data (water front position vs dimensionless time) as reference data file in backend/reference-tests/data/martin-moyce-1952.json
- [ ] T085 [P] [US6] Implement analytical Poiseuille flow solution (parabolic velocity profile u(y) = G/(2*mu) * y * (H-y)) as reference comparator in backend/reference-tests/src/lib.rs

### Benchmark Test Cases

- [ ] T086 [US6] Create dam break benchmark test: run simulation to t=0.5s, extract water front position at each frame, compare against Martin & Moyce data, pass if within 10% in backend/reference-tests/src/benchmarks.rs
- [ ] T087 [US6] Create high-resolution hydrostatic benchmark: run to steady state, extract pressure at 10 depth levels, compare each against rho*g*h, pass if all within 1% in backend/reference-tests/src/benchmarks.rs
- [ ] T088 [US6] Create Poiseuille flow benchmark: run to steady state, extract velocity profile across channel, compare against analytical parabolic solution, pass if RMS error <5% in backend/reference-tests/src/benchmarks.rs
- [ ] T089 [US6] Create standing wave benchmark: run for 10+ wave periods, extract surface displacement over time, compare period against linear theory (T=2*pi/omega), pass if within 5% in backend/reference-tests/src/benchmarks.rs

### Integration

- [ ] T090 [US6] Add `just test-benchmarks` recipe to justfile; verify all benchmarks pass in release mode
- [ ] T091 [US6] Document benchmark results (particle count, runtime, accuracy vs reference) in specs/001-sph-fluid-sim/benchmark-results.md

**Checkpoint**: All standard SPH benchmarks pass against published data. The simulation is validated for water flows, viscous forces, pressure-velocity coupling, and transient dynamics. Results documented with accuracy metrics.

---

## Phase 9: US7 -- Metal GPU Acceleration (Priority: P7)

**Goal**: Implement a Metal GPU kernel backend (via wgpu) that runs the full SPH simulation on Apple Silicon GPU, achieving 5x+ throughput over the CPU kernel for 50K+ particles. The GPU kernel implements the same `SimulationKernel` trait as `CpuKernel`, so orchestration, visualization, and force extraction work unchanged.

**Independent Test**: Run the same simulation on CPU and GPU backends. Compare final particle positions (within 1e-4 relative error) and force results (within 2%). Benchmark throughput at 50K particles.

**Dependencies**: Requires US1 complete (working CPU kernel). Benefits from US2+US6 (benchmarks to validate GPU matches CPU). Should be implemented after CPU kernel is fully validated.

**References**:
- wgpu documentation (wgpu.rs)
- Metal Best Practices Guide (Apple)
- "Parallel SPH on GPU" (various papers on GPU-based neighbor search + force computation)

### Setup & Infrastructure

- [ ] T092 [US7] Add `wgpu` and `bytemuck` dependencies to backend/crates/kernel/Cargo.toml. Gate GPU code behind a `gpu` cargo feature flag so CPU-only builds remain lightweight.
- [ ] T093 [US7] Create backend/crates/kernel/src/gpu/ module directory with mod.rs. Define `GpuKernel` struct implementing `SimulationKernel` trait. Include Metal device initialization, pipeline creation, and auto-detection (fall back to CPU if Metal unavailable).

### Compute Shaders (WGSL)

- [ ] T094 [P] [US7] Write WGSL compute shader for neighbor grid construction: hash particles into uniform grid cells, build cell start/end index arrays in backend/crates/kernel/src/gpu/shaders/neighbor_grid.wgsl
- [ ] T095 [P] [US7] Write WGSL compute shader for SPH density summation: iterate neighbors, compute Wendland C2 kernel contributions, accumulate density per particle in backend/crates/kernel/src/gpu/shaders/density.wgsl
- [ ] T096 [P] [US7] Write WGSL compute shader for pressure + viscous force computation: compute Tait EOS pressure, pressure gradient forces, viscous forces per particle in backend/crates/kernel/src/gpu/shaders/forces.wgsl
- [ ] T097 [P] [US7] Write WGSL compute shader for Velocity Verlet time integration: kick-drift-kick update of positions and velocities, domain clamping in backend/crates/kernel/src/gpu/shaders/integrate.wgsl

### GPU Kernel Implementation

- [ ] T098 [US7] Implement GPU buffer management in backend/crates/kernel/src/gpu/buffers.rs: create GPU storage buffers for particle arrays (position, velocity, density, pressure, force), boundary particles, and neighbor grid. Handle CPU↔GPU data transfer for initialization and result readback.
- [ ] T099 [US7] Implement `GpuKernel::step()` in backend/crates/kernel/src/gpu/mod.rs: dispatch compute shaders in sequence (neighbor grid → density → forces → integrate), synchronize between passes, handle boundary particle pressure mirroring on GPU.
- [ ] T100 [US7] Implement `GpuKernel::compute_error_metrics()`: readback density array from GPU, compute max density variation, energy conservation, mass conservation on CPU (metrics are infrequent, no need for GPU compute).

### Backend Selection & Config

- [ ] T101 [US7] Add `backend: "cpu" | "gpu" | "auto"` field to SimulationConfig in backend/crates/orchestrator/src/config.rs. Default to "auto" (use GPU if available, fall back to CPU). Update configs/water-box-1cm.json and configs/air-obstacle-1cm.json with default "auto".
- [ ] T102 [US7] Update runner.rs to instantiate `GpuKernel` or `CpuKernel` based on config backend selection and Metal availability in backend/crates/orchestrator/src/runner.rs

### Validation & Benchmarks

- [ ] T103 [US7] Create GPU vs CPU comparison test: run water-box-1cm on both backends for 1000 timesteps, compare final particle positions (max relative error < 1e-4) and density (< 0.1% difference) in backend/crates/kernel/tests/gpu_cpu_parity.rs
- [ ] T104 [US7] Create GPU throughput benchmark: measure timesteps/second for 10K, 50K, 100K particles on GPU vs CPU, verify GPU achieves 5x+ speedup at 50K+ in backend/crates/kernel/benches/gpu_throughput.rs
- [ ] T105 [US7] Run existing reference tests (Phase 4) and standard benchmarks (Phase 8) with GPU backend to verify physics match CPU results

**Checkpoint**: Metal GPU kernel runs the full SPH simulation on Apple Silicon. GPU matches CPU physics within floating-point tolerance. GPU achieves 5x+ throughput for 50K+ particles. Auto-detection falls back to CPU gracefully.

---

## Phase 10: Polish & Cross-Cutting Concerns

**Purpose**: Features and improvements that span multiple user stories or are quality-of-life enhancements.

- [ ] T106 [P] Implement manual checkpointing: save SimulationState to disk (binary serialization) and resume from checkpoint in backend/crates/orchestrator/src/runner.rs
- [ ] T107 [P] Implement checkpoint REST endpoints: POST /api/simulations/{id}/checkpoint and POST /api/simulations/resume in backend/crates/server/src/api.rs
- [ ] T108 Implement inflow/outflow boundary conditions (spawn particles at inflow face with configured velocity, remove particles exiting at outflow face) in backend/crates/kernel/src/boundary.rs -- deferred from MVP
- [ ] T109 [P] Set up Playwright e2e test infrastructure in frontend/e2e/ per constitution principle III
- [ ] T110 [P] Write Playwright e2e test: load page, select config, start simulation, verify particles render, verify camera controls in frontend/e2e/basic-sim.spec.ts
- [ ] T111 Performance profiling of CPU kernel with cargo bench: identify hot paths, optimize neighbor search and force computation in backend/crates/kernel/benches/
- [ ] T112 Run quickstart.md validation end-to-end: follow every step in quickstart.md and verify it works
- [ ] T113 Code cleanup: ensure all public APIs have doc comments, remove dead code, verify cargo clippy clean

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies -- can start immediately
- **Foundational (Phase 2)**: Depends on Setup completion -- BLOCKS all user stories
- **US1 (Phase 3)**: Depends on Foundational (Phase 2) -- this is the MVP
- **US2 (Phase 4)**: Depends on US1 -- reference tests validate the MVP
- **US3 (Phase 5)**: Depends on US1 -- force extraction on running simulation; Stokes drag test uses US2 framework
- **US4 (Phase 6)**: Depends on US1 -- diagnostics on running simulation
- **US5 (Phase 7)**: Depends on US1 -- distributed execution; validation benefits from US2 + US3
- **US6 (Phase 8)**: Depends on US1+US2+US3 -- standard benchmarks require working sim, test framework, and force extraction. Runs AFTER all other development work.
- **US7 (Phase 9)**: Depends on US1 -- GPU kernel implements same trait as CPU. Benefits from US6 benchmarks for validation. Should run after CPU kernel is fully validated.
- **Polish (Phase 10)**: Can start after US1; some tasks independent

### Dependency Graph

```
Phase 1 (Setup)
    |
    v
Phase 2 (Foundational) ---- BLOCKS ALL ----+
    |                                       |
    v                                       |
Phase 3 (US1 - MVP) <----------------------+
    |
    +------+----------+-----------+---------+
    |      |          |           |         |
    v      v          v           v         v
  Ph 4   Ph 5      Ph 6        Ph 7      Ph 10
 (US2)  (US3)     (US4)       (US5)    (Polish)
  Ref   Force     Diag       Distrib
 Tests  Extract
    |      |
    v      v
  Ph 5: Stokes drag test uses US2 framework + US3 forces
    |      |
    +------+
    |
    v
  Ph 8 (US6 - Standard SPH Benchmarks)
  Dam break, Poiseuille, hydrostatic hires, standing wave
    |
    v
  Ph 9 (US7 - Metal GPU Acceleration)
  wgpu/Metal compute shaders, GPU kernel, auto-detection
  Validated against CPU results + existing benchmarks
```

### Within Each User Story

1. Backend kernel changes first (data structures, algorithms)
2. Backend orchestrator wiring (lifecycle, coordination)
3. Backend server endpoints (API, WebSocket)
4. Frontend rendering and UI
5. Configs and end-to-end integration last

### Parallel Opportunities

**Within Phase 2 (Foundational)**:
- T006+T007 (particle structs) | T008 (Wendland C2 kernel) | T010 (EOS) | T011 (trait def) | T013 (geometry) | T015 (logging) | T016 (TS types) -- all independent files
- T009 (neighbor search) depends on T006 (particle data)
- T012 (config parsing) is independent
- T014 (domain setup + boundary particle generation) depends on T012 + T013

**Within Phase 3 (US1)**:
- T033 (Three.js scene) | T034 (WS client) -- independent frontend files
- T040 (water config) | T041 (air config) -- independent config files
- T024 + T025 (two-particle + kernel normalization tests) can run in parallel once kernel code exists
- Backend kernel (T017-T023) is mostly sequential (density, pressure, forces, integrate)
- Backend server (T029-T032) can parallelize with frontend (T033-T039) once kernel is done

**Within Phase 4 (US2)**:
- T043 (framework) | T044 (runner entry point) -- independent files
- T045 + T046 + T047 (test cases) -- all independent once framework exists

**Across Phases** (after US1 complete):
- US2 (Phase 4) | US3 (Phase 5) | US4 (Phase 6) can start in parallel
- US5 (Phase 7) can start in parallel but is lowest priority
- Polish (Phase 8) tasks with [P] can start after US1

---

## Agent Team Parallel Example

```bash
# After Phase 2 foundational is complete, three agents can work in parallel:

# AGENT 1: Backend kernel work (sequential within):
Task: "T017 - Implement SPH density summation"
Task: "T018 - Implement pressure force computation" (after T017)
Task: "T019 - Implement viscous force" (after T018)
Task: "T020 - Gravity body force" (after T019)
Task: "T021 - Boundary particle enforcement" (after T020)
Task: "T022 - Adaptive timestep" (after T021)
Task: "T023 - CpuKernel with Velocity Verlet" (after T022)

# AGENT 2: Frontend work (independent of kernel until integration):
Task: "T033 - Create Three.js scene with camera and OrbitControls"
Task: "T034 - Implement WebSocket client for binary frame parsing"
Task: "T035 - Particle renderer" (after T033)
Task: "T036 - STL geometry rendering" (after T033)
Task: "T037 - Config selector UI"
Task: "T038 - Simulation controls"

# AGENT 3: Configs + early kernel tests (once kernel compiles):
Task: "T040 - Create water-box-1cm.json config"
Task: "T041 - Create air-obstacle-1cm.json config"
Task: "T024 - Two-particle symmetry test" (after T023)
Task: "T025 - Kernel normalization test" (after T023)
Task: "T026 - Hydrostatic column test" (after T023)

# Once kernel + orchestrator done, wire up server (T029-T032)
# Once server + frontend done, integration (T039, T042)

# After US1 complete, three agents can again work in parallel:
# AGENT 1: US2 reference tests (Phase 4)
# AGENT 2: US3 force extraction (Phase 5)
# AGENT 3: US4 diagnostics (Phase 6)
```

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Phase 1: Setup (~5 tasks)
2. Complete Phase 2: Foundational (~11 tasks)
3. Complete Phase 3: User Story 1 (~26 tasks, including 3 early kernel tests)
4. **STOP and VALIDATE**: Water settles in box, air contained with obstacle, camera works
5. This is the minimum viable product -- a working fluid sim with web viewer

### Incremental Delivery

1. Setup + Foundational -> Foundation ready
2. **US1** -> Basic fluid sim in browser (MVP!) with boundary particles, Velocity Verlet, ~10K particles on CPU
3. **US2** -> Reference tests and validation (fast smoke tests) -- builds trust in correctness
4. **US3** -> Force extraction + reporting -- engineering payoff
5. **US4** -> Diagnostics overlay -- quality assurance
6. **US5** -> Distributed execution -- scale
7. **US6** -> Standard SPH benchmarks (dam break, Poiseuille, etc.) -- definitive accuracy proof
8. **US7** -> Metal GPU acceleration -- 5x+ throughput for 50K+ particles
9. Polish -> Checkpointing, inflow/outflow, e2e tests, performance tuning

### Key Simplifications from Original Plan

- Phase transitions REMOVED: no boiling/condensation, no steam tables, no if97 crate
- Water and air coexist but do not transition between phases
- Fixed global smoothing length (not per-particle)
- Boundary particles enforce walls (not SDF penalty forces)
- Velocity Verlet integrator (not "Euler or leapfrog")
- WebSocket streams ~5% of particles per frame for visualization
- Inflow/outflow deferred to polish phase
- Target: ~10K particles on M1 MacBook Air CPU

### Dev Iteration Guidance

Per user guidance: start with very small simulations (~1cm geometry, <1K particles) on CPU to iterate fast. Every task should be testable at this tiny scale. Metal GPU acceleration is a future optimization after the CPU kernel is validated.

---

## Summary

| Phase | User Story | Tasks | Parallel Opportunities |
|-------|-----------|-------|----------------------|
| 1 | Setup | 5 | 4 of 5 parallelizable |
| 2 | Foundational | 11 | 7 of 11 parallelizable |
| 3 | US1 - Basic Fluid Sim (MVP) | 26 | Frontend + configs + tests parallel with backend |
| 4 | US2 - Reference Tests (smoke) | 8 | Test cases parallelizable |
| 5 | US3 - Force Extraction | 8 | Limited (sequential dependency) |
| 6 | US4 - Diagnostics | 7 | Some parallel (backend/frontend split) |
| 7 | US5 - Distributed | 6 | Limited (sequential dependency) |
| 8 | US6 - Standard SPH Benchmarks | 12 | Configs + reference data parallelizable |
| 9 | US7 - Metal GPU Acceleration | 14 | Compute shaders parallelizable |
| 10 | Polish | 8 | 4 of 8 parallelizable |
| **Total** | | **105** | |

## Notes

- [P] tasks = different files, no dependencies on incomplete tasks
- [Story] label maps task to specific user story for traceability
- Each user story should be independently completable and testable
- Commit after each task or logical group
- Stop at any checkpoint to validate story independently
- All development uses ~1cm geometry and <1K particles on CPU until Metal GPU acceleration (Phase 9) is ready
- Metal GPU kernel (Phase 9) uses wgpu which compiles to Metal on macOS and Vulkan on Linux
- Constitution principle III requires Playwright e2e tests (covered in Phase 10)
- Constitution principle V requires structured logging (covered in Phase 2, T015)
- phase.rs has been REMOVED from the kernel crate -- no phase transition logic anywhere
- Water and air coexist as separate FluidType values but never convert between each other
- Boundary particles (frozen SPH particles on walls/geometry) replace SDF penalty forces
- Smoothing length is a global simulation parameter, not per-particle
