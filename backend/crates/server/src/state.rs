//! Shared application state

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Mutex;

/// Shared application state
pub struct AppState {
    /// Active simulations (ID -> Runner)
    pub simulations: Mutex<HashMap<String, crate::runner::SimulationRunner>>,
    /// Path to configs directory
    pub configs_dir: PathBuf,
    /// Path to geometries directory
    pub geometries_dir: PathBuf,
    /// Server port
    pub port: u16,
}

/// Simulation status
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SimStatus {
    /// Simulation created but not started
    Created,
    /// Simulation running
    Running,
    /// Simulation paused
    Paused,
    /// Simulation stopped
    Stopped,
}

impl AppState {
    /// Create new application state
    pub fn new(configs_dir: PathBuf, geometries_dir: PathBuf, port: u16) -> Self {
        Self {
            simulations: Mutex::new(HashMap::new()),
            configs_dir,
            geometries_dir,
            port,
        }
    }
}
