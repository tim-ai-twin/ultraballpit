//! Configuration parsing and validation for SPH simulations

use kernel::SolverType;
use serde::{Deserialize, Serialize};
use std::fs;

/// Compute backend selection for the simulation kernel.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum BackendType {
    /// CPU-only reference implementation
    Cpu,
    /// GPU-accelerated via wgpu/Metal
    Gpu,
    /// Automatically select GPU if available, fall back to CPU
    Auto,
}

impl Default for BackendType {
    fn default() -> Self {
        Self::Auto
    }
}

/// SPH solver type selection for configuration files.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum ConfigSolverType {
    /// Weakly Compressible SPH (explicit EOS, acoustic CFL). Default.
    Wcsph,
    /// Predictive-Corrective Incompressible SPH (iterative pressure solve, advective CFL).
    Pcisph,
}

impl Default for ConfigSolverType {
    fn default() -> Self {
        Self::Wcsph
    }
}

impl ConfigSolverType {
    /// Convert to kernel `SolverType`.
    pub fn to_kernel_solver_type(&self) -> SolverType {
        match self {
            Self::Wcsph => SolverType::Wcsph,
            Self::Pcisph => SolverType::Pcisph,
        }
    }
}

/// Axis selection for primitive geometry
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum Axis {
    /// X axis
    X,
    /// Y axis
    Y,
    /// Z axis
    Z,
}

impl Default for Axis {
    fn default() -> Self {
        Self::Y
    }
}

/// Procedural obstacle geometry (alternative to an STL `geometry_file`)
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum GeometryPrimitive {
    /// No obstacle
    None,
    /// Sphere obstacle
    Sphere {
        /// Sphere center [x, y, z]
        center: [f32; 3],
        /// Sphere radius (meters)
        radius: f32,
    },
    /// Axis-aligned box obstacle
    Box {
        /// Minimum corner [x, y, z]
        min: [f32; 3],
        /// Maximum corner [x, y, z]
        max: [f32; 3],
    },
    /// Cylinder obstacle along an axis
    Cylinder {
        /// Center of the cylinder [x, y, z]
        center: [f32; 3],
        /// Cylinder radius (meters)
        radius: f32,
        /// Cylinder height (meters)
        height: f32,
        /// Cylinder axis (default Y)
        #[serde(default)]
        axis: Axis,
    },
}

/// Main simulation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimulationConfig {
    /// Human-readable simulation name
    pub name: String,
    /// Which fluid(s) to simulate
    pub fluid_type: ConfigFluidType,
    /// Path to STL geometry file (optional; alternative to `geometry`)
    #[serde(default)]
    pub geometry_file: Option<String>,
    /// Procedural obstacle geometry (optional; takes precedence over `geometry_file`)
    #[serde(default)]
    pub geometry: Option<GeometryPrimitive>,
    /// Simulation domain bounds
    pub domain: DomainBounds,
    /// Region initially filled with fluid (defaults to the whole domain)
    #[serde(default)]
    pub fluid_region: Option<DomainBounds>,
    /// Boundary conditions per face
    #[serde(default)]
    pub boundary_conditions: BoundaryConditions,
    /// Initial inter-particle distance (meters)
    pub particle_spacing: f32,
    /// Gravity vector (m/s^2)
    #[serde(default = "default_gravity")]
    pub gravity: [f32; 3],
    /// WCSPH speed of sound parameter
    #[serde(default = "default_speed_of_sound")]
    pub speed_of_sound: f32,
    /// Kinematic viscosity (m^2/s)
    #[serde(default = "default_viscosity")]
    pub viscosity: f32,
    /// Initial fluid temperature (Kelvin)
    #[serde(default = "default_temperature")]
    pub initial_temperature: f32,
    /// Stop after this many timesteps
    pub max_timesteps: Option<u64>,
    /// Stop after this much simulated time (seconds)
    pub max_time: Option<f64>,
    /// CFL condition for adaptive timestep
    #[serde(default = "default_cfl")]
    pub cfl_number: f32,
    /// Compute backend: "cpu", "gpu", or "auto" (default: "auto")
    #[serde(default)]
    pub backend: BackendType,
    /// SPH solver: "wcsph" or "pcisph" (default: "wcsph")
    #[serde(default)]
    pub solver: ConfigSolverType,
}

/// Fluid type configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConfigFluidType {
    /// Water only
    Water,
    /// Air only
    Air,
    /// Both water and air (bottom half water, top half air)
    Mixed,
}

/// Domain bounding box
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DomainBounds {
    /// Minimum corner [x, y, z]
    pub min: [f32; 3],
    /// Maximum corner [x, y, z]
    pub max: [f32; 3],
}

/// Boundary condition type
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum BoundaryType {
    /// Simple boundary types (Wall, Outflow, Periodic)
    Simple(SimpleBoundary),
    /// Inflow boundary with velocity and temperature
    Inflow {
        /// Inflow parameters
        #[serde(rename = "Inflow")]
        params: InflowParams
    },
}

/// Simple boundary condition variants
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SimpleBoundary {
    /// Solid wall (frozen boundary particles)
    Wall,
    /// Open boundary (particles can leave)
    Outflow,
    /// Periodic boundary
    Periodic,
}

/// Inflow boundary parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InflowParams {
    /// Inflow velocity [vx, vy, vz] (m/s)
    pub velocity: [f32; 3],
    /// Inflow temperature (Kelvin)
    pub temperature: f32,
}

/// Boundary conditions for all six domain faces
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BoundaryConditions {
    /// X minimum face boundary
    pub x_min: BoundaryType,
    /// X maximum face boundary
    pub x_max: BoundaryType,
    /// Y minimum face boundary
    pub y_min: BoundaryType,
    /// Y maximum face boundary
    pub y_max: BoundaryType,
    /// Z minimum face boundary
    pub z_min: BoundaryType,
    /// Z maximum face boundary
    pub z_max: BoundaryType,
}

// Default values
fn default_gravity() -> [f32; 3] {
    [0.0, -9.81, 0.0]
}

fn default_speed_of_sound() -> f32 {
    50.0
}

fn default_viscosity() -> f32 {
    0.001
}

fn default_temperature() -> f32 {
    293.15
}

fn default_cfl() -> f32 {
    0.5
}

impl Default for BoundaryConditions {
    fn default() -> Self {
        Self {
            x_min: BoundaryType::Simple(SimpleBoundary::Wall),
            x_max: BoundaryType::Simple(SimpleBoundary::Wall),
            y_min: BoundaryType::Simple(SimpleBoundary::Wall),
            y_max: BoundaryType::Simple(SimpleBoundary::Wall),
            z_min: BoundaryType::Simple(SimpleBoundary::Wall),
            z_max: BoundaryType::Simple(SimpleBoundary::Wall),
        }
    }
}

impl SimulationConfig {
    /// Load configuration from a JSON file
    pub fn load(path: &str) -> Result<Self, String> {
        let contents = fs::read_to_string(path)
            .map_err(|e| format!("Failed to read config file {}: {}", path, e))?;

        let config: SimulationConfig = serde_json::from_str(&contents)
            .map_err(|e| format!("Failed to parse config JSON: {}", e))?;

        config.validate()?;
        Ok(config)
    }

    /// Validate the configuration
    pub fn validate(&self) -> Result<(), String> {
        // Check domain bounds
        if self.domain.min[0] >= self.domain.max[0] {
            return Err("Domain min.x must be less than max.x".to_string());
        }
        if self.domain.min[1] >= self.domain.max[1] {
            return Err("Domain min.y must be less than max.y".to_string());
        }
        if self.domain.min[2] >= self.domain.max[2] {
            return Err("Domain min.z must be less than max.z".to_string());
        }

        // Check particle spacing
        if self.particle_spacing <= 0.0 {
            return Err("Particle spacing must be positive".to_string());
        }

        // Check speed of sound
        if self.speed_of_sound <= 0.0 {
            return Err("Speed of sound must be positive".to_string());
        }

        // Check viscosity
        if self.viscosity < 0.0 {
            return Err("Viscosity must be non-negative".to_string());
        }

        // Check temperature
        if self.initial_temperature <= 0.0 {
            return Err("Initial temperature must be positive (Kelvin)".to_string());
        }

        // Check CFL number
        if self.cfl_number <= 0.0 || self.cfl_number > 1.0 {
            return Err("CFL number must be in range (0, 1]".to_string());
        }

        // Check max_timesteps
        if let Some(max_timesteps) = self.max_timesteps {
            if max_timesteps == 0 {
                return Err("max_timesteps must be at least 1".to_string());
            }
        }

        // Check max_time
        if let Some(max_time) = self.max_time {
            if max_time <= 0.0 {
                return Err("max_time must be positive".to_string());
            }
        }

        // Validate periodic boundaries are paired
        self.validate_periodic_boundaries()?;

        // Validate fluid region (must be a valid box; clamped to domain at placement)
        if let Some(region) = &self.fluid_region {
            for axis in 0..3 {
                if region.min[axis] >= region.max[axis] {
                    return Err("fluid_region min must be less than max on every axis".to_string());
                }
            }
        }

        // Validate primitive geometry
        if let Some(geometry) = &self.geometry {
            match geometry {
                GeometryPrimitive::None => {}
                GeometryPrimitive::Sphere { radius, .. } => {
                    if *radius <= 0.0 {
                        return Err("Sphere radius must be positive".to_string());
                    }
                }
                GeometryPrimitive::Box { min, max } => {
                    for axis in 0..3 {
                        if min[axis] >= max[axis] {
                            return Err("Box geometry min must be less than max on every axis".to_string());
                        }
                    }
                }
                GeometryPrimitive::Cylinder { radius, height, .. } => {
                    if *radius <= 0.0 || *height <= 0.0 {
                        return Err("Cylinder radius and height must be positive".to_string());
                    }
                }
            }
        }

        Ok(())
    }

    /// Validate that periodic boundaries are properly paired
    fn validate_periodic_boundaries(&self) -> Result<(), String> {
        let x_min_periodic = matches!(self.boundary_conditions.x_min,
            BoundaryType::Simple(SimpleBoundary::Periodic));
        let x_max_periodic = matches!(self.boundary_conditions.x_max,
            BoundaryType::Simple(SimpleBoundary::Periodic));
        let y_min_periodic = matches!(self.boundary_conditions.y_min,
            BoundaryType::Simple(SimpleBoundary::Periodic));
        let y_max_periodic = matches!(self.boundary_conditions.y_max,
            BoundaryType::Simple(SimpleBoundary::Periodic));
        let z_min_periodic = matches!(self.boundary_conditions.z_min,
            BoundaryType::Simple(SimpleBoundary::Periodic));
        let z_max_periodic = matches!(self.boundary_conditions.z_max,
            BoundaryType::Simple(SimpleBoundary::Periodic));

        if x_min_periodic != x_max_periodic {
            return Err("Periodic boundaries must be paired: x_min and x_max".to_string());
        }
        if y_min_periodic != y_max_periodic {
            return Err("Periodic boundaries must be paired: y_min and y_max".to_string());
        }
        if z_min_periodic != z_max_periodic {
            return Err("Periodic boundaries must be paired: z_min and z_max".to_string());
        }

        Ok(())
    }

    /// Calculate smoothing length from particle spacing
    pub fn smoothing_length(&self) -> f32 {
        1.3 * self.particle_spacing
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_smoothing_length() {
        let config = SimulationConfig {
            name: "test".to_string(),
            fluid_type: ConfigFluidType::Water,
            geometry_file: Some("test.stl".to_string()),
            geometry: None,
            fluid_region: None,
            domain: DomainBounds {
                min: [0.0, 0.0, 0.0],
                max: [1.0, 1.0, 1.0],
            },
            boundary_conditions: BoundaryConditions::default(),
            particle_spacing: 0.01,
            gravity: default_gravity(),
            speed_of_sound: default_speed_of_sound(),
            viscosity: default_viscosity(),
            initial_temperature: default_temperature(),
            max_timesteps: None,
            max_time: None,
            cfl_number: default_cfl(),
            backend: BackendType::default(),
            solver: ConfigSolverType::default(),
        };

        assert!((config.smoothing_length() - 0.013).abs() < 1e-6);
    }

    #[test]
    fn test_validation_domain_bounds() {
        let mut config = SimulationConfig {
            name: "test".to_string(),
            fluid_type: ConfigFluidType::Water,
            geometry_file: Some("test.stl".to_string()),
            geometry: None,
            fluid_region: None,
            domain: DomainBounds {
                min: [1.0, 0.0, 0.0],
                max: [0.0, 1.0, 1.0],
            },
            boundary_conditions: BoundaryConditions::default(),
            particle_spacing: 0.01,
            gravity: default_gravity(),
            speed_of_sound: default_speed_of_sound(),
            viscosity: default_viscosity(),
            initial_temperature: default_temperature(),
            max_timesteps: None,
            max_time: None,
            cfl_number: default_cfl(),
            backend: BackendType::default(),
            solver: ConfigSolverType::default(),
        };

        assert!(config.validate().is_err());

        config.domain.min[0] = 0.0;
        config.domain.max[0] = 1.0;
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_validation_particle_spacing() {
        let mut config = SimulationConfig {
            name: "test".to_string(),
            fluid_type: ConfigFluidType::Water,
            geometry_file: Some("test.stl".to_string()),
            geometry: None,
            fluid_region: None,
            domain: DomainBounds {
                min: [0.0, 0.0, 0.0],
                max: [1.0, 1.0, 1.0],
            },
            boundary_conditions: BoundaryConditions::default(),
            particle_spacing: -0.01,
            gravity: default_gravity(),
            speed_of_sound: default_speed_of_sound(),
            viscosity: default_viscosity(),
            initial_temperature: default_temperature(),
            max_timesteps: None,
            max_time: None,
            cfl_number: default_cfl(),
            backend: BackendType::default(),
            solver: ConfigSolverType::default(),
        };

        assert!(config.validate().is_err());

        config.particle_spacing = 0.01;
        assert!(config.validate().is_ok());
    }
}
