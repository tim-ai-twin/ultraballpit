# Quickstart: SPH Fluid Simulation

## Prerequisites

- Rust (latest stable): `rustup update stable`
- Node.js 20+ and npm
- `just` command runner: `cargo install just`
- macOS (Apple Silicon) or Linux

## Build

```bash
# Clone and enter the repo
git clone <repo-url> && cd ultraballpit

# Build everything
just build

# This runs:
#   cargo build --workspace       (backend)
#   cd frontend && npm install && npm run build  (frontend)
```

## Run Your First Simulation

### 1. Start the server

```bash
just serve
# Starts backend on http://localhost:3000
# Serves web viewer at http://localhost:3000
# WebSocket at ws://localhost:3000/ws/simulation/{id}
```

### 2. Open the web viewer

Navigate to `http://localhost:3000` in your browser.

### 3. Start a simulation

The web UI lists available configs from the `configs/` directory.
Select `water-box-1cm` (tiny water settling in a 1cm box) and
click Start.

Or via the API:

```bash
curl -X POST http://localhost:3000/api/simulations \
  -H "Content-Type: application/json" \
  -d '{"config": "water-box-1cm"}'
```

### 4. Observe

- Water particles fall under gravity and pool at the bottom of the
  box.
- Use mouse to orbit (left-click drag), pan (right-click drag),
  and zoom (scroll).
- Toggle the debug overlay to see frame time, particle count, and
  error metrics.

## Sample Configs

| Config | Description | Particles | Fluid |
|--------|-------------|-----------|-------|
| `water-box-1cm.json` | Water settling in a 1cm box | ~1K | Water |
| `air-obstacle-1cm.json` | Air in a 1cm box with obstacle | ~1K | Air |

## Create Your Own Config

Copy an existing config from `configs/` and modify it. See
[config-schema.json](contracts/config-schema.json) for the full
schema.

Minimal example:

```json
{
  "name": "my-simulation",
  "fluid_type": "Water",
  "geometry_file": "geometries/my-shape.stl",
  "domain": {
    "min": [-0.005, -0.005, -0.005],
    "max": [0.005, 0.005, 0.005]
  },
  "particle_spacing": 0.0005,
  "gravity": [0.0, -9.81, 0.0],
  "initial_temperature": 293.15
}
```

## Run Reference Tests

```bash
just test-reference
# Runs all reference test cases and reports pass/fail
```

## Checkpointing

```bash
# Save current state (via API)
curl -X POST http://localhost:3000/api/simulations/sim-001/checkpoint

# Resume from checkpoint
curl -X POST http://localhost:3000/api/simulations/resume \
  -H "Content-Type: application/json" \
  -d '{"checkpoint": "checkpoints/sim-001-ts5000.bin"}'
```

## Development

```bash
# Run all tests
just test

# Run only backend tests
just test-backend

# Run only frontend tests
just test-frontend

# Run e2e tests (Playwright)
just test-e2e

# Run benchmarks
just bench

# Watch mode (auto-rebuild on changes)
just watch
```

## Validation

After running a simulation, verify correctness:

1. Check the debug overlay — density variation should be < 3%.
2. Run `just test-reference` — all reference tests should pass.
3. Query the force API and compare against expected values.
