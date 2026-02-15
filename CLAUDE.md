# ultraballpit Development Guidelines

Auto-generated from feature plans. Last updated: 2026-02-15

## Active Technologies

- **Backend**: Rust (latest stable) — kernel, orchestrator, server crates
- **Frontend**: TypeScript (strict mode) — Three.js, Vite
- **Key Rust crates**: nom_stl, mesh_to_sdf, if97, axum, tokio, tower-http, nalgebra, bytemuck, tracing, serde
- **Key JS packages**: three, vite, vitest, playwright

## Project Structure

```text
backend/          # Rust workspace (kernel, orchestrator, server)
frontend/         # TypeScript web viewer (Three.js)
configs/          # JSON simulation configs
specs/            # Feature specifications and plans
```

## Commands

```bash
just build          # Build backend + frontend
just serve          # Start server on localhost:3000
just test           # Run all tests
just test-backend   # cargo test --workspace
just test-frontend  # cd frontend && npm test
just test-e2e       # Playwright e2e tests
just test-reference # Run reference validation tests
just bench          # cargo bench
just watch          # Auto-rebuild on changes
```

## Code Style

- Rust: Follow standard conventions, `cargo clippy`, `cargo fmt`
- TypeScript: Strict mode, no `any` types outside type shims
- Commits: Conventional commit prefixes (feat:, fix:, perf:, etc.)

## Constitution

See `.specify/memory/constitution.md` (v1.2.0) for project principles.

## Recent Changes

- 001-sph-fluid-sim: SPH fluid simulation feature in progress

<!-- MANUAL ADDITIONS START -->
<!-- MANUAL ADDITIONS END -->
