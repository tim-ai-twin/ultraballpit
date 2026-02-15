# Research: SPH Fluid Simulation

**Feature**: `001-sph-fluid-sim`
**Date**: 2026-02-15

## Decision: SPH Implementation Approach

**Decision**: Build custom WCSPH kernel from scratch in Rust. Do not
use `salva3d`.

**Rationale**: `salva3d` (v0.9.0, Feb 2024, ~94 downloads/month) is
the only Rust SPH crate but is game/animation-focused, not designed
for separable kernel architecture (FR-011), and lacks phase change
support. The kernel must be a swappable component with a clean trait
interface. Building from scratch gives full control over the solver
loop, memory layout, and future Metal/GPU porting.

**Alternatives considered**:
- `salva3d`: Too tightly coupled to its own solver loop. Would need
  extensive modification. Low download count suggests limited community
  support.
- Port an existing C++/CUDA SPH codebase: Would lose Rust safety
  guarantees and complicate the build. Not worth it for the scope.

## Decision: Neighbor Search

**Decision**: Custom uniform-grid spatial hash. Fall back to `kiddo`
KD-tree only if needed for non-uniform particle distributions.

**Rationale**: SPH neighbor search has a fixed radius (smoothing
length h), and particles are roughly uniformly distributed in space.
A uniform grid with cell size = h provides O(1) amortized lookup per
particle by checking only 27 adjacent cells (3x3x3). This is
significantly faster than KD-tree approaches for SPH specifically.
`kiddo` v5.2.2 is a solid KD-tree but adds overhead for the
construction/rebuild that a grid hash avoids.

**Alternatives considered**:
- `kiddo` v5.2.2: Excellent for general nearest-neighbor, but
  overkill for fixed-radius SPH queries. KD-tree rebuild per frame
  is O(N log N) vs O(N) for grid hash.
- `rstar` R-tree: Similar drawbacks — designed for spatial indexing,
  not fixed-radius bulk queries.

## Decision: STL Parsing

**Decision**: Use `nom_stl` crate for STL file parsing.

**Rationale**: `nom_stl` v0.2.2 parses both binary and ASCII STL
formats, handles a 30MB binary file in <20ms, and has minimal
dependencies (just `nom`). Small codebase (662 LOC), MIT licensed.
STL parsing is a solved problem — no reason to build from scratch.

**Alternatives considered**:
- `stl_io`: Also handles binary + ASCII. Similar maturity. Either
  would work; `nom_stl` is slightly faster in benchmarks.
- Custom parser: Unnecessary for a well-defined format.

## Decision: SDF Generation

**Decision**: Use `mesh_to_sdf` crate for generating signed distance
fields from STL triangle meshes.

**Rationale**: `mesh_to_sdf` v0.4.0 (Sep 2024) does exactly what we
need — generates grid-based SDFs from triangle meshes. Supports
multiple acceleration structures (BVH, R-tree), handles both
watertight and non-watertight meshes, and integrates with common math
libraries (nalgebra, glam). Well-maintained (updated 2024).

**Alternatives considered**:
- `parry3d` for closest-point queries + manual SDF construction:
  More work, same result. `mesh_to_sdf` wraps this pattern.
- Custom voxelization: Reinventing the wheel.

## Decision: GPU Compute (Future)

**Decision**: Use `wgpu` for Metal/GPU compute shaders when ready.
CPU-first for initial development.

**Note**: Target hardware is M1 MacBook Air. CPU-first with ~10K
particles for iteration. GPU deferred until CPU kernel is validated.

**Rationale**: `wgpu` is mature, compiles to Metal on macOS and
Vulkan on Linux/cloud. Compute shader support is stable. The SPH
kernel can be ported to WGSL compute shaders that operate on storage
buffers of particle data. The CPU kernel serves as the reference
implementation and pure-Rust fallback (per constitution principle II).

**Alternatives considered**:
- Raw Metal API via `metal-rs`: macOS-only, no Linux portability.
- `vulkano`: Vulkan-only, no Metal.
- `wgpu` is the clear winner for cross-platform GPU compute.

## Decision: 3D Visualization Library

**Decision**: Three.js with custom ShaderMaterial.

**Rationale**: Three.js provides the best performance-to-development-
speed ratio for particle rendering. `THREE.Points` with custom
shaders handles 1M+ particles at 60 FPS. `InstancedMesh` works for
<500K with actual sphere geometry. The ecosystem is massive with
extensive documentation and examples. Camera controls (OrbitControls),
post-processing, and custom GLSL shaders are all well-supported.

**Alternatives considered**:
- Raw WebGPU: Maximum performance but 5-10x development effort.
  No scene graph, camera controls, or post-processing pipeline.
  Browser support still incomplete (Safari, Firefox). Since the
  simulation runs in Rust, not in-browser, we don't benefit from
  WebGPU compute shaders.
- Babylon.js: Solid but heavier framework, fewer examples for
  large-scale particle visualization, less active community for
  this use case.

## Decision: Screen-Space Fluid Rendering

**Decision**: Implement in Three.js using multi-pass custom
ShaderMaterial with WebGLRenderTarget.

**Rationale**: The standard technique (Simon Green / NVIDIA Flex):
depth pass → bilateral blur → normal reconstruction → shading.
Implementable in Three.js with ~200 lines of GLSL across 3 passes.
This is a later enhancement — start with colored points/spheres.

**Implementation plan**:
1. Pass 1: Render particles as sphere impostors to a float depth
   texture.
2. Pass 2: Bilateral blur (separable H+V) to smooth the depth field.
3. Pass 3: Reconstruct normals from smoothed depth, apply Fresnel +
   environment mapping + absorption.

## Decision: Transport Protocol

**Decision**: WebSocket with binary frames.

**Rationale**: Zero-copy serialization on both ends. Rust side:
`bytemuck::cast_slice::<f32, u8>()` to convert particle arrays to
bytes. JS side: `new Float32Array(event.data)` to wrap received
ArrayBuffer directly. Localhost throughput >100 MB/s easily handles
500K particles × 12 bytes × 30 FPS = 180 MB/s. Persistent TCP
connection means <1ms latency per frame.

**Protocol format**: Each binary WebSocket message is one frame:
- Header (16 bytes): frame number (u64), particle count (u32),
  flags (u32)
- Body: flat f32 array `[x0, y0, z0, temp0, phase0, x1, y1, z1,
  temp1, phase1, ...]`

**Alternatives considered**:
- Server-Sent Events: Text-only, would need Base64 encoding. 33%
  overhead. Not suitable for bulk binary streaming.
- SharedArrayBuffer: Only works within the same process (Web
  Workers). Cannot bridge Rust backend → browser.
- WebTransport (QUIC/UDP): Lower latency for remote, but more
  complex setup and on localhost TCP is already sub-millisecond.

## Decision: Rust HTTP/WebSocket Server

**Decision**: Use `axum` + `tokio` + `tower-http`.

**Rationale**: `axum` is the most actively maintained Rust web
framework, built on `tokio`/`hyper`/`tower`. Built-in WebSocket
support via extractors. Static file serving via `tower-http::ServeDir`.
Binary WebSocket frames via `ws::Message::Binary(Vec<u8>)`.
Ergonomic router structure. Best documentation and community support.

**Alternatives considered**:
- `actix-web`: Historically fastest in benchmarks, but actor model
  adds unnecessary complexity for "receive data, broadcast to WS."
  Community has shifted toward axum.
- `warp`: Filter-based API is confusing for complex routing.
  Development has slowed. Effectively superseded by axum.

## Decision: Build Orchestration

**Decision**: `just` (justfile) as the top-level task runner.

**Rationale**: Simpler than Makefiles with better cross-platform
support. Can orchestrate `cargo build`, `npm run build`, `cargo test`,
`npm test`, and combined workflows. Already popular in the Rust
ecosystem.

**Alternatives considered**:
- Makefile: Works but has arcane syntax for non-trivial recipes.
- npm scripts calling cargo: Awkward coupling.
- `cargo-make`: Heavier, less standard.

## Decision: Boundary Handling (Wall Forces)

**Decision**: Use frozen boundary particles on domain walls and geometry surfaces. Retain SDF for normal computation and force extraction.

**Rationale**: Penalty-based SDF boundary forces cause density deficiency near walls (particles near walls compute lower density because they have fewer neighbors). This produces artificial pressure gradients and fails hydrostatic pressure validation. Frozen boundary particles participate in the density summation and pressure calculation, producing correct wall pressure profiles. This is the standard approach in the SPH literature (Adami et al. 2012).

**Alternatives considered**:
- SDF penalty forces: Simpler to implement but requires stiffness tuning and Shepard density correction. Identified as a 3x risk by review team.
- Ghost/mirror particles: More complex, mainly used for free-surface flows.

## Decision: Smoothing Length

**Decision**: Use a single global smoothing length `h = 1.3 * particle_spacing` for all particles. Do not use adaptive per-particle smoothing length.

**Rationale**: Adaptive smoothing length introduces variable kernel support radius (neighbor search grid cell must accommodate the largest h), asymmetric kernels (momentum conservation requires grad-h corrections), and per-particle CFL conditions. A fixed global h keeps the neighbor search trivially correct with a single cell size and avoids grad-h corrections. For the initial ~10K particle target, fixed h is sufficient.

**Alternatives considered**:
- Per-particle adaptive h: Needed for large density ratios (gas expansion). Deferred until gas-phase resolution becomes a problem at higher particle counts.

## Decision: Time Integration

**Decision**: Use the Velocity Verlet (leapfrog) integrator.

**Rationale**: Symplectic integrator with O(dt^2) accuracy and excellent long-term energy conservation. ~5 lines more code than forward Euler but dramatically better conservation properties. Essential for passing the energy conservation reference test (SC-006: <5% error). Forward Euler causes monotonic energy growth; symplectic Euler has drift with adaptive timesteps.

**Alternatives considered**:
- Forward Euler: Energy grows monotonically, fails conservation tests within hundreds of steps.
- Symplectic Euler: Better than forward Euler but still drifts with adaptive dt.
- RK4: Non-symplectic, more expensive, no better conservation than Verlet for SPH.

## Decision: WebSocket Particle Subsampling

**Decision**: Stream ~5% of particles to the web viewer. Full particle set used for simulation and metrics.

**Rationale**: At 10K particles x 20 bytes = 200KB/frame, full streaming is feasible. But at future scale (100K+), bandwidth becomes a concern: 100K x 20 bytes x 30 FPS = 60 MB/s. Designing with subsampling from the start avoids a future architectural refactor. The server selects a uniform random or strided subsample and sends only those particles per frame. The viewer never needs to know the full particle count.

**Alternatives considered**:
- Always stream all particles: Works for 10K but creates a hard ceiling at ~200-500K.
- LOD (level-of-detail) based on camera distance: More complex, marginal benefit on localhost.

## Dependency Summary

### Rust (backend/Cargo.toml)

| Crate | Version | Purpose | Justification |
|-------|---------|---------|---------------|
| `nom_stl` | 0.2.x | STL parsing | Fast binary+ASCII parser, minimal deps |
| `mesh_to_sdf` | 0.4.x | SDF generation | Grid SDF from triangle mesh, BVH acceleration |
| `axum` | 0.7.x | HTTP + WS server | Best ergonomics, built-in WebSocket |
| `tokio` | 1.x | Async runtime | Required by axum |
| `tower-http` | 0.5.x | Static file serving | ServeDir for frontend assets |
| `bytemuck` | 1.x | Zero-copy serialization | f32 slices to byte slices for WS |
| `serde` / `serde_json` | 1.x | JSON config parsing | Deserialize simulation configs |
| `tracing` | 0.1.x | Structured logging | Constitution principle V |
| `nalgebra` | 0.33.x | Linear algebra | Vector math for SPH kernel |
| `wgpu` | 24.x | GPU compute (future -- not in initial implementation) | Metal/Vulkan compute shaders |

### TypeScript (frontend/package.json)

| Package | Purpose | Justification |
|---------|---------|---------------|
| `three` | 3D rendering | Points, InstancedMesh, custom shaders |
| `@types/three` | Type definitions | TypeScript strict mode |
| `vite` | Build tool / dev server | Fast HMR, ESM-native |
| `vitest` | Unit tests | Fast, Vite-native |
| `playwright` | E2e tests | Constitution principle III |

### Dev Tools

| Tool | Purpose |
|------|---------|
| `just` | Top-level task runner |
| `cargo-watch` | Auto-rebuild on file change |
