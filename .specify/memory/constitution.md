<!--
Sync Impact Report
==================
Version change: 1.1.0 → 1.2.0 (MINOR — added living documentation mandate)
Modified sections:
  - Development Workflow: added Living Documentation subsection
Added sections: None
Removed sections: None
Templates requiring updates:
  - .specify/templates/plan-template.md — no changes needed (generic)
  - .specify/templates/spec-template.md — no changes needed (generic)
  - .specify/templates/tasks-template.md — no changes needed (generic)
Follow-up TODOs: None
-->

# Ultraballpit Constitution

## Core Principles

### I. Simulation-UI Separation

The physics simulation and the web UI are independent subsystems
with a clearly defined boundary.

- The simulation MUST run as a standalone Rust process (or library)
  with no dependency on the frontend.
- The frontend MUST be a TypeScript web application served on
  localhost that communicates with the simulation over a local
  transport (WebSocket, HTTP, or shared memory).
- Either side MUST be replaceable without modifying the other,
  provided the transport contract is honored.
- Rationale: Decoupling enables independent testing, independent
  optimization, and the option to swap renderers or simulation
  backends without cross-cutting rewrites.

### II. Performance-First Simulation

The simulation engine is the computational core; its performance
budget takes priority over convenience.

- Simulation code MUST be written in Rust. Additional languages
  (CUDA, OpenCL, ISPC, or similar) MAY be introduced strictly
  for parallel-processing acceleration.
- Hot paths MUST avoid heap allocation where feasible; prefer
  arena or pool allocation patterns.
- Every acceleration extension MUST expose a pure-Rust fallback
  so the project builds and runs on any platform with a Rust
  toolchain.
- Frame-budget targets MUST be defined per feature and tracked
  in benchmarks (e.g., `cargo bench`).
- Rationale: A ball-pit simulation is only useful if it runs at
  interactive rates. Performance is a feature, not an afterthought.

### III. Test Discipline

Correctness of simulation logic and UI behavior MUST be
demonstrable through automated tests.

- Simulation unit tests MUST use deterministic inputs and
  fixed-seed RNG so results are reproducible.
- Frontend tests MUST cover component behavior; visual snapshot
  tests are encouraged but not mandatory.
- End-to-end tests MUST use Playwright (or Playwright MCP) to
  verify critical user journeys through the web UI against a
  running simulation backend. E2e tests MUST be included for
  every user story that involves UI interaction.
- Integration tests MUST verify the transport contract between
  simulation and UI (message format, sequencing, error cases).
- New code MUST NOT decrease existing test coverage without
  explicit justification in the PR description.
- Rationale: Physics bugs are subtle and hard to catch visually.
  Deterministic tests are the primary defense. E2e tests catch
  integration failures that unit and contract tests miss.

### IV. Simplicity & YAGNI

Start with the minimum viable design; extend only when a
concrete need is demonstrated.

- No abstraction layer, config option, or indirection MUST be
  added unless it solves a problem that exists today.
- Prefer three similar lines of code over a premature helper
  function.
- Third-party dependencies MUST be justified: each dependency
  added to `Cargo.toml` or `package.json` requires a one-line
  rationale in the PR.
- Rationale: Complexity is the primary threat to a two-language
  project. Every unnecessary layer multiplies the maintenance
  surface.

### V. Observability

Runtime behavior of both simulation and UI MUST be inspectable
without attaching a debugger.

- The simulation MUST expose structured diagnostic output
  (frame timing, particle count, collision stats) over the
  same transport used by the UI, gated behind a debug flag.
- The frontend MUST include a debug overlay or console panel
  that displays simulation diagnostics in real time.
- Logging MUST use structured formats (JSON for Rust via
  `tracing` or equivalent; structured console for TypeScript).
- Rationale: Two-process architectures are harder to debug;
  built-in observability compensates for this.

## Technical Constraints

- **Rust version**: Latest stable toolchain. Nightly features
  MAY be used only when gated behind a feature flag with a
  stable fallback.
- **TypeScript**: Strict mode enabled (`"strict": true`).
  No `any` types outside of third-party type shims.
- **Transport**: The simulation-UI protocol MUST be documented
  in a contract file (`specs/transport-contract.md` or
  equivalent) before implementation begins.
- **Build**: `cargo build` and the frontend build MUST succeed
  independently. A top-level task runner (Makefile, justfile,
  or equivalent) MUST orchestrate both.
- **Platform**: MUST build and run on macOS (primary dev) and
  Linux. Windows support is optional until explicitly scoped.

## Development Workflow

- **Branching**: One branch per feature or fix. Merge to `main`
  via pull request.
- **Commits**: Atomic commits with conventional-commit prefixes
  (`feat:`, `fix:`, `perf:`, `docs:`, `test:`, `chore:`).
- **Code review**: Every PR MUST pass CI (build + test for both
  Rust and TypeScript) before merge.
- **Benchmarks**: Performance-sensitive PRs MUST include before/
  after benchmark numbers in the PR description.
- **Documentation**: Public APIs and transport messages MUST have
  doc comments. Internal code is documented only where intent is
  non-obvious.
- **Living documentation**: Design artifacts (`spec.md`,
  `plan.md`, `research.md`, and related files under `specs/`)
  MUST be kept in sync with the implementation as it evolves.
  When implementation reveals a deviation from the spec or plan
  (new requirement, changed approach, dropped scope), the
  corresponding artifact MUST be updated in the same PR or
  immediately following PR. These files serve as the persistent
  record of PRDs, architectural decisions, and business
  rationale — stale artifacts are treated as bugs.

## Governance

This constitution is the highest-authority document for project
decisions. All PRs and code reviews MUST verify compliance with
these principles.

- **Amendments**: Any principle change MUST be documented in a
  PR with rationale, approved by the project owner, and reflected
  in a version bump of this document.
- **Versioning**: This document follows semantic versioning.
  MAJOR for principle removals or incompatible redefinitions,
  MINOR for new principles or material expansions, PATCH for
  wording clarifications.
- **Compliance review**: At each spec or plan review, the
  Constitution Check section MUST reference the current version
  of this document and confirm alignment.
- **Conflict resolution**: Where this constitution conflicts with
  a feature spec or plan, the constitution takes precedence.
  Update the constitution first if the conflict reveals a needed
  change.

**Version**: 1.2.0 | **Ratified**: 2026-02-15 | **Last Amended**: 2026-02-15
