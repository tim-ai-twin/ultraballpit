# REST API Contract

**Base URL**: `http://localhost:{PORT}/api`
**Content-Type**: `application/json`

## Endpoints

### GET /api/configs

List available simulation configuration files.

**Response 200**:
```json
{
  "configs": [
    {
      "name": "water-box-1cm",
      "path": "configs/water-box-1cm.json",
      "fluid_type": "Water",
      "particle_count_estimate": 1000
    }
  ]
}
```

### GET /api/configs/{name}

Get a specific configuration file contents.

**Response 200**: Raw JSON config file contents.
**Response 404**: Config not found.

### POST /api/simulations

Start a new simulation from a config.

**Request**:
```json
{
  "config": "water-box-1cm"
}
```

**Response 201**:
```json
{
  "simulation_id": "sim-001",
  "status": "running",
  "ws_url": "ws://localhost:{PORT}/ws/simulation/sim-001",
  "particle_count": 10000,
  "subsample_count": 500
}
```

### GET /api/simulations/{id}

Get simulation status.

**Response 200**:
```json
{
  "simulation_id": "sim-001",
  "status": "running",
  "timestep": 5000,
  "sim_time": 0.25,
  "particle_count": 1000,
  "error_metrics": {
    "max_density_variation": 0.018,
    "energy_conservation": 0.003,
    "mass_conservation": 0.0001
  }
}
```

### POST /api/simulations/{id}/pause

Pause a running simulation.

**Response 200**: `{ "status": "paused" }`

### POST /api/simulations/{id}/resume

Resume a paused simulation.

**Response 200**: `{ "status": "running" }`

### POST /api/simulations/{id}/checkpoint

Save simulation state to disk.

**Response 200**:
```json
{
  "checkpoint_path": "checkpoints/sim-001-ts5000.bin"
}
```

### POST /api/simulations/resume

Resume from a checkpoint file.

**Request**:
```json
{
  "checkpoint": "checkpoints/sim-001-ts5000.bin"
}
```

**Response 201**: Same as POST /api/simulations.

### GET /api/simulations/{id}/forces

Get force time-series data.

**Query parameters**:
- `from_timestep` (u64, optional): Start of range
- `to_timestep` (u64, optional): End of range
- `aggregation` (string, optional): "raw" | "mean" | "peak"

**Response 200**:
```json
{
  "simulation_id": "sim-001",
  "records": [
    {
      "timestep": 100,
      "sim_time": 0.005,
      "pressure_force": [0.0, -0.15, 0.0],
      "viscous_force": [0.001, 0.0, 0.0],
      "thermal_flux": 5.2
    }
  ],
  "aggregate": {
    "peak_pressure_force": [0.0, -0.32, 0.0],
    "mean_pressure_force": [0.0, -0.15, 0.0],
    "rms_pressure_force": [0.0, 0.18, 0.0]
  }
}
```

### GET /api/simulations/{id}/report

Generate a summary report.

**Response 200**:
```json
{
  "simulation_id": "sim-001",
  "config_name": "water-box-1cm",
  "total_timesteps": 10000,
  "total_sim_time": 0.5,
  "forces": {
    "peak_pressure": [0.0, -0.32, 0.0],
    "mean_pressure": [0.0, -0.15, 0.0],
    "peak_viscous": [0.002, 0.001, 0.0]
  },
  "error_bounds": {
    "max_density_variation": 0.023,
    "energy_conservation": 0.041,
    "mass_conservation": 0.00008
  },
  "diagnostics": {
    "mean_frame_time_ms": 12.5,
    "particle_count": 1000
  }
}
```

## Error Responses

All errors follow:
```json
{
  "error": "description of the error"
}
```

| Status | Meaning |
|--------|---------|
| 400 | Invalid request (bad config name, invalid params) |
| 404 | Simulation or config not found |
| 409 | Conflict (e.g., start simulation while one is running) |
| 500 | Internal server error |
