//! REST API endpoints for simulation management

use axum::{
    extract::{Path, State},
    http::StatusCode,
    Json,
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

use crate::state::{AppState, SimStatus};

// ---------------------------------------------------------------------------
// Request/Response Types
// ---------------------------------------------------------------------------

/// Request body for creating a simulation
#[derive(Debug, Deserialize)]
pub struct CreateSimulationRequest {
    /// Configuration name (e.g., "water-box-1cm")
    pub config: String,
}

/// Response for simulation creation
#[derive(Debug, Serialize)]
pub struct CreateSimulationResponse {
    /// Unique simulation ID
    pub simulation_id: String,
    /// Current status
    pub status: String,
    /// WebSocket URL for connecting to this simulation
    pub ws_url: String,
    /// Total particle count
    pub particle_count: usize,
    /// Number of particles in each subsampled frame (~5%)
    pub subsample_count: usize,
}

/// Configuration file metadata
#[derive(Debug, Serialize)]
pub struct ConfigInfo {
    /// Configuration name
    pub name: String,
    /// File path
    pub path: String,
    /// Fluid type
    pub fluid_type: String,
    /// Estimated particle count
    pub particle_count_estimate: usize,
}

/// List of available configurations
#[derive(Debug, Serialize)]
pub struct ConfigListResponse {
    /// Available configurations
    pub configs: Vec<ConfigInfo>,
}

/// Simulation status response
#[derive(Debug, Serialize)]
pub struct SimulationStatusResponse {
    /// Simulation ID
    pub simulation_id: String,
    /// Current status
    pub status: String,
    /// Current timestep number
    pub timestep: u64,
    /// Current simulation time (seconds)
    pub sim_time: f64,
    /// Particle count
    pub particle_count: usize,
    /// Error metrics
    pub error_metrics: ErrorMetricsResponse,
}

/// Error metrics
#[derive(Debug, Serialize)]
pub struct ErrorMetricsResponse {
    /// Maximum density variation
    pub max_density_variation: f32,
    /// Energy conservation error
    pub energy_conservation: f32,
    /// Mass conservation error
    pub mass_conservation: f32,
}

/// Generic status response
#[derive(Debug, Serialize)]
pub struct StatusResponse {
    /// Status string
    pub status: String,
}

// ---------------------------------------------------------------------------
// API Handlers
// ---------------------------------------------------------------------------

/// GET /api/configs - List available configuration files
pub async fn list_configs(
    State(state): State<Arc<AppState>>,
) -> Result<Json<ConfigListResponse>, (StatusCode, String)> {
    let configs_dir = &state.configs_dir;

    // Read directory
    let entries = std::fs::read_dir(configs_dir)
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("Failed to read configs directory: {}", e)))?;

    let mut configs = Vec::new();

    for entry in entries {
        let entry = entry.map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
        let path = entry.path();

        // Only process .json files
        if path.extension().and_then(|s| s.to_str()) != Some("json") {
            continue;
        }

        // Try to parse the config
        match orchestrator::config::SimulationConfig::load(path.to_str().unwrap()) {
            Ok(config) => {
                // Estimate particle count from domain size and spacing
                let dx = config.domain.max[0] - config.domain.min[0];
                let dy = config.domain.max[1] - config.domain.min[1];
                let dz = config.domain.max[2] - config.domain.min[2];
                let volume = dx * dy * dz;
                let particle_volume = config.particle_spacing.powi(3);
                let particle_count_estimate = (volume / particle_volume) as usize;

                let fluid_type = match config.fluid_type {
                    orchestrator::config::ConfigFluidType::Water => "Water",
                    orchestrator::config::ConfigFluidType::Air => "Air",
                    orchestrator::config::ConfigFluidType::Mixed => "Mixed",
                };

                configs.push(ConfigInfo {
                    name: config.name.clone(),
                    path: path.to_string_lossy().to_string(),
                    fluid_type: fluid_type.to_string(),
                    particle_count_estimate,
                });
            }
            Err(e) => {
                tracing::warn!("Failed to parse config {:?}: {}", path, e);
                continue;
            }
        }
    }

    Ok(Json(ConfigListResponse { configs }))
}

/// GET /api/configs/{name} - Get raw configuration JSON
pub async fn get_config(
    State(state): State<Arc<AppState>>,
    Path(name): Path<String>,
) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    // Construct path (sanitize name to prevent directory traversal)
    let safe_name = name.replace("..", "").replace("/", "");
    let config_path = state.configs_dir.join(format!("{}.json", safe_name));

    // Check if file exists
    if !config_path.exists() {
        return Err((StatusCode::NOT_FOUND, format!("Configuration '{}' not found", name)));
    }

    // Read and parse JSON
    let content = std::fs::read_to_string(&config_path)
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("Failed to read config: {}", e)))?;

    let json: serde_json::Value = serde_json::from_str(&content)
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("Failed to parse config: {}", e)))?;

    Ok(Json(json))
}

/// POST /api/simulations - Create a new simulation
pub async fn create_simulation(
    State(state): State<Arc<AppState>>,
    Json(req): Json<CreateSimulationRequest>,
) -> Result<Json<CreateSimulationResponse>, (StatusCode, String)> {
    // Construct config path
    let safe_name = req.config.replace("..", "").replace("/", "");
    let config_path = state.configs_dir.join(format!("{}.json", safe_name));

    if !config_path.exists() {
        return Err((StatusCode::NOT_FOUND, format!("Configuration '{}' not found", req.config)));
    }

    // Load configuration
    let config = orchestrator::config::SimulationConfig::load(config_path.to_str().unwrap())
        .map_err(|e| (StatusCode::BAD_REQUEST, format!("Invalid configuration: {}", e)))?;

    // Create simulation ID
    let sim_id = uuid::Uuid::new_v4().to_string();

    // Create simulation runner
    let runner = crate::runner::SimulationRunner::new(config)
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("Failed to create simulation: {}", e)))?;

    let particle_count = runner.particle_count();
    let subsample_count = runner.subsample_count();

    // Store runner
    state.simulations.lock().unwrap().insert(sim_id.clone(), runner);

    // Build WebSocket URL
    let ws_url = format!("ws://localhost:{}/ws/simulation/{}", state.port, sim_id);

    Ok(Json(CreateSimulationResponse {
        simulation_id: sim_id,
        status: "created".to_string(),
        ws_url,
        particle_count,
        subsample_count,
    }))
}

/// GET /api/simulations/{id} - Get simulation status
pub async fn get_simulation(
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
) -> Result<Json<SimulationStatusResponse>, (StatusCode, String)> {
    let simulations = state.simulations.lock().unwrap();
    let runner = simulations.get(&id)
        .ok_or_else(|| (StatusCode::NOT_FOUND, format!("Simulation '{}' not found", id)))?;

    let status = runner.status();
    let status_str = match status {
        SimStatus::Created => "created",
        SimStatus::Running => "running",
        SimStatus::Paused => "paused",
        SimStatus::Stopped => "stopped",
    };

    let metrics = runner.error_metrics();

    Ok(Json(SimulationStatusResponse {
        simulation_id: id,
        status: status_str.to_string(),
        timestep: runner.timestep_count(),
        sim_time: runner.sim_time(),
        particle_count: runner.particle_count(),
        error_metrics: ErrorMetricsResponse {
            max_density_variation: metrics.max_density_variation,
            energy_conservation: metrics.energy_conservation,
            mass_conservation: metrics.mass_conservation,
        },
    }))
}

/// POST /api/simulations/{id}/pause - Pause simulation
pub async fn pause_simulation(
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
) -> Result<Json<StatusResponse>, (StatusCode, String)> {
    let simulations = state.simulations.lock().unwrap();
    let runner = simulations.get(&id)
        .ok_or_else(|| (StatusCode::NOT_FOUND, format!("Simulation '{}' not found", id)))?;

    runner.pause();

    Ok(Json(StatusResponse {
        status: "paused".to_string(),
    }))
}

/// POST /api/simulations/{id}/resume - Resume simulation
pub async fn resume_simulation(
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
) -> Result<Json<StatusResponse>, (StatusCode, String)> {
    let simulations = state.simulations.lock().unwrap();
    let runner = simulations.get(&id)
        .ok_or_else(|| (StatusCode::NOT_FOUND, format!("Simulation '{}' not found", id)))?;

    runner.resume();

    Ok(Json(StatusResponse {
        status: "running".to_string(),
    }))
}
