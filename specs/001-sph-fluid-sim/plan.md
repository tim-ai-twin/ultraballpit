# Implementation Plan: SPH Fluid Simulation

**Branch**: `001-sph-fluid-sim` | **Date**: 2026-02-15 | **Spec**: [spec.md](spec.md)
**Input**: Feature specification from `/specs/001-sph-fluid-sim/spec.md`

## Summary

Build a particle-based fluid simulation (WCSPH) for mixed water-air
simulation without phase transitions, around 3D STL geometry, with
interactive web visualization, force extraction, and error bound
reporting. The system has four work streams: a TypeScript web viewer
(frontend), a Rust simulation kernel (separable compute backend), a
Rust orchestration layer (geometry prep, domain setup, coordination),
and an automated reference test suite. Target: ~10K particles on M1
MacBook Air for fast iteration. Per user guidance: start with very
small simulations (tiny particle counts, ~1cm geometry) on CPU to
iterate fast, then add Metal GPU acceleration.

## Technical Context

**Language/Version**: Rust (latest stable) for backend (kernel +
orchestration); TypeScript (strict mode) for frontend
**Primary Dependencies**: Rust: `nom_stl` (STL parsing),
`mesh_to_sdf` (SDF generation), `axum` + `tokio` + `tower-http`
(HTTP/WS server), `nalgebra` (math), `bytemuck` (zero-copy
serialization), `tracing` (logging), `serde`/`serde_json` (config).
TypeScript: `three` (3D rendering), `vite` (build), `vitest` (test),
`playwright` (e2e). SPH neighbor search: custom uniform grid hash.
See [research.md](research.md) for full justifications.
**Storage**: File-based — JSON config files, checkpoint files on
disk, STL files as input, force time-series as output
**Testing**: `cargo test` for Rust, Vitest or Jest for TypeScript,
Playwright for e2e, `cargo bench` for performance
**Target Platform**: macOS Apple Silicon (primary), Linux (secondary).
~10K particles CPU for dev iteration; ~50-100K CPU ceiling; Metal GPU
future. Web viewer in any modern browser on localhost.
**Project Type**: Web application (Rust backend + TypeScript frontend)
**Performance Goals**: 30 FPS web viewer; CPU kernel handles ~10K
particles for dev, ~50-100K at scale; start dev with ~1K particles.
**Boundary handling**: Frozen boundary particles (not SDF penalty
forces).
**Constraints**: Subsonic only; ~1cm starting geometry; CPU-first;
localhost-only; no public internet exposure
**Scale/Scope**: Single user, local machine. Cloud distributed
execution is P6 (last priority).

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1
design. Constitution v1.2.0.*

| # | Principle | Status | Notes |
|---|-----------|--------|-------|
| I | Simulation-UI Separation | PASS | Rust backend and TypeScript frontend are independent subsystems. Transport contract will be documented before implementation (per Technical Constraints). |
| II | Performance-First Simulation | PASS | Kernel in Rust. CPU-first with Metal fallback planned. Arena/pool allocation for hot paths. Benchmarks via `cargo bench`. |
| III | Test Discipline | PASS | Deterministic sim tests with fixed-seed RNG. Playwright e2e for UI stories. Integration tests for transport contract. Reference test suite (US3) covers validation. |
| IV | Simplicity & YAGNI | PASS | Phase transitions removed. Fixed smoothing length. Boundary particles. 10K particle target. |
| V | Observability | PASS | Debug overlay (US5) for sim diagnostics. Structured logging via `tracing` (Rust) and structured console (TS). Diagnostics over same transport as sim data. |
| TC | Transport contract documented | PASS | Produced in Phase 1: contracts/transport-protocol.md, contracts/rest-api.md, contracts/config-schema.json |
| TC | Independent builds | PASS | `cargo build` and frontend build are separate. Top-level Makefile/justfile planned. |
| TC | macOS + Linux | PASS | Rust + wgpu targets both. TypeScript runs in any browser. |
| DW | Living documentation | PASS | spec.md, plan.md, research.md maintained per constitution. |

**Gate result: PASS** — all principles satisfied or have clear
path to satisfaction. Transport contract is pending Phase 1 output.

## Project Structure

### Documentation (this feature)

```text
specs/001-sph-fluid-sim/
├── plan.md              # This file
├── research.md          # Phase 0 output
├── data-model.md        # Phase 1 output
├── quickstart.md        # Phase 1 output
├── contracts/           # Phase 1 output (transport protocol, force API, config schema)
├── checklists/          # Quality checklists
└── tasks.md             # Phase 2 output (/speckit.tasks command)
```

### Source Code (repository root)

```text
backend/
├── Cargo.toml
├── crates/
│   ├── kernel/              # Simulation kernel (separable compute)
│   │   ├── src/
│   │   │   ├── lib.rs       # Kernel trait + CPU implementation
│   │   │   ├── particle.rs  # Particle data structures
│   │   │   ├── sph.rs       # WCSPH solver (density, pressure, forces)
│   │   │   ├── neighbor.rs  # Spatial hashing / neighbor search
│   │   │   ├── eos.rs       # Equations of state (Tait, ideal gas)
│   │   │   └── boundary.rs  # Boundary particle generation + enforcement (frozen boundary particles on walls/geometry)
│   │   ├── benches/         # cargo bench benchmarks
│   │   └── tests/           # Unit tests (deterministic, fixed-seed)
│   ├── orchestrator/        # Orchestration layer
│   │   ├── src/
│   │   │   ├── lib.rs
│   │   │   ├── config.rs    # JSON config parsing + validation
│   │   │   ├── geometry.rs  # STL parsing + SDF generation
│   │   │   ├── domain.rs    # Domain setup, decomposition
│   │   │   ├── runner.rs    # Simulation lifecycle (run, pause, checkpoint)
│   │   │   └── force.rs     # Force accumulation + extraction
│   │   └── tests/
│   └── server/              # HTTP + WebSocket server
│       ├── src/
│       │   ├── main.rs      # Entry point, serves frontend + WS
│       │   ├── api.rs       # REST endpoints (config list, force data, reports)
│       │   └── ws.rs        # WebSocket particle streaming
│       └── tests/
├── reference-tests/         # Validation test suite (US3)
│   ├── cases/               # Test case definitions (JSON configs + expected results)
│   └── src/                 # Test runner + comparison logic
└── tests/
    ├── integration/         # Cross-crate integration tests
    └── contract/            # Transport contract tests

frontend/
├── package.json
├── tsconfig.json
├── src/
│   ├── index.html
│   ├── main.ts             # App entry point
│   ├── viewer/             # 3D viewer (Three.js or WebGPU)
│   │   ├── scene.ts        # Scene setup, camera, lighting
│   │   ├── particles.ts    # Particle rendering (instanced/points)
│   │   ├── geometry.ts     # STL mesh rendering
│   │   └── overlays.ts     # Force heatmap, debug overlay
│   ├── transport/          # WebSocket client for sim data
│   │   └── client.ts
│   ├── ui/                 # Config selector, controls
│   │   ├── config-list.ts
│   │   └── controls.ts
│   └── types/              # Shared type definitions
│       └── protocol.ts     # Transport message types
├── tests/                  # Vitest component tests
└── e2e/                    # Playwright e2e tests

configs/                    # Sample JSON simulation configs
├── water-box-1cm.json      # Tiny water-in-box (dev default)
└── air-obstacle-1cm.json   # Tiny air flow past obstacle

justfile                    # Top-level build orchestration (just command runner)
```

**Structure Decision**: Web application layout with `backend/` (Rust
workspace with three crates: kernel, orchestrator, server) and
`frontend/` (TypeScript). The kernel crate is deliberately isolated
so it can be compiled and tested independently, and swapped with a
Metal/GPU variant later. The three-crate workspace mirrors the spec's
work streams (kernel = compute, orchestrator = control, server =
transport bridge).

## Complexity Tracking

| Violation | Why Needed | Simpler Alternative Rejected Because |
|-----------|------------|-------------------------------------|
| 3 Rust crates instead of 1 | Kernel must be separable per FR-011 (swap CPU/GPU/distributed). Orchestrator handles lifecycle. Server handles transport. | Single crate would couple kernel interface to HTTP/WS concerns, preventing clean kernel replacement. |
| Removed phase transitions | Simplifies kernel (no steam tables, no EOS switching, no latent heat) | Phase change was identified as 5-8x complexity risk by review team |
| Boundary particles instead of SDF penalty | Better density accuracy near walls; standard SPH approach | SDF penalty requires stiffness tuning and causes density deficiency |
| Standard validation benchmarks (Phase 9) | Proves correctness against published experimental/analytical data | Fast smoke tests (~1cm) catch regressions but can't validate accuracy at scale |
| wgpu/Metal GPU kernel (Phase 10) | 5-50x throughput for 50K+ particles; enables real interactive exploration | CPU kernel is correct but too slow for large domains. Metal chosen because Apple Silicon is primary platform; wgpu abstracts to Vulkan on Linux. |
