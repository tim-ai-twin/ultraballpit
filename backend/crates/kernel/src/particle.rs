//! Particle data structures using struct-of-arrays layout for GPU-readiness and SIMD.

/// Fluid type discriminator.
///
/// Used to distinguish different fluids for equation-of-state selection
/// and rendering.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
#[repr(u8)]
pub enum FluidType {
    /// Liquid water (incompressible via Tait EOS)
    Water = 0,
    /// Air (ideal gas EOS)
    Air = 1,
}

/// Struct-of-arrays particle storage.
///
/// All arrays are parallel: index `i` across every array refers to the same particle.
/// Separate x/y/z arrays (rather than Vec3) are used deliberately for SIMD lane
/// utilization and straightforward GPU buffer mapping.
#[derive(Debug, Clone)]
pub struct ParticleArrays {
    // ---- Positions ----
    /// X positions (meters)
    pub x: Vec<f32>,
    /// Y positions (meters)
    pub y: Vec<f32>,
    /// Z positions (meters)
    pub z: Vec<f32>,

    // ---- Velocities ----
    /// X velocities (m/s)
    pub vx: Vec<f32>,
    /// Y velocities (m/s)
    pub vy: Vec<f32>,
    /// Z velocities (m/s)
    pub vz: Vec<f32>,

    // ---- Accelerations ----
    /// X accelerations (m/s^2)
    pub ax: Vec<f32>,
    /// Y accelerations (m/s^2)
    pub ay: Vec<f32>,
    /// Z accelerations (m/s^2)
    pub az: Vec<f32>,

    // ---- Scalar fields ----
    /// Density (kg/m^3)
    pub density: Vec<f32>,
    /// Pressure (Pa)
    pub pressure: Vec<f32>,
    /// Particle mass (kg)
    pub mass: Vec<f32>,
    /// Temperature (K)
    pub temperature: Vec<f32>,
    /// Fluid type tag
    pub fluid_type: Vec<FluidType>,
}

impl ParticleArrays {
    /// Create an empty particle collection with no particles allocated.
    pub fn new() -> Self {
        Self {
            x: Vec::new(),
            y: Vec::new(),
            z: Vec::new(),
            vx: Vec::new(),
            vy: Vec::new(),
            vz: Vec::new(),
            ax: Vec::new(),
            ay: Vec::new(),
            az: Vec::new(),
            density: Vec::new(),
            pressure: Vec::new(),
            mass: Vec::new(),
            temperature: Vec::new(),
            fluid_type: Vec::new(),
        }
    }

    /// Return the number of particles currently stored.
    pub fn len(&self) -> usize {
        self.x.len()
    }

    /// Return `true` if there are no particles.
    pub fn is_empty(&self) -> bool {
        self.x.is_empty()
    }

    /// Append a single particle with the given initial state.
    ///
    /// Velocity, acceleration, density, and pressure are initialized to zero.
    /// Temperature defaults to 293.15 K (20 C).
    #[allow(clippy::too_many_arguments)]
    pub fn push_particle(
        &mut self,
        px: f32,
        py: f32,
        pz: f32,
        mass: f32,
        density: f32,
        temperature: f32,
        fluid_type: FluidType,
    ) {
        self.x.push(px);
        self.y.push(py);
        self.z.push(pz);
        self.vx.push(0.0);
        self.vy.push(0.0);
        self.vz.push(0.0);
        self.ax.push(0.0);
        self.ay.push(0.0);
        self.az.push(0.0);
        self.density.push(density);
        self.pressure.push(0.0);
        self.mass.push(mass);
        self.temperature.push(temperature);
        self.fluid_type.push(fluid_type);
    }
}

impl Default for ParticleArrays {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_particle_arrays() {
        let pa = ParticleArrays::new();
        assert_eq!(pa.len(), 0);
        assert!(pa.is_empty());
    }

    #[test]
    fn push_and_len() {
        let mut pa = ParticleArrays::new();
        pa.push_particle(1.0, 2.0, 3.0, 0.001, 1000.0, 293.15, FluidType::Water);
        assert_eq!(pa.len(), 1);
        assert!(!pa.is_empty());
        assert_eq!(pa.x[0], 1.0);
        assert_eq!(pa.y[0], 2.0);
        assert_eq!(pa.z[0], 3.0);
        assert_eq!(pa.mass[0], 0.001);
        assert_eq!(pa.density[0], 1000.0);
        assert_eq!(pa.temperature[0], 293.15);
        assert_eq!(pa.fluid_type[0], FluidType::Water);
        // Velocity and acceleration should be zero
        assert_eq!(pa.vx[0], 0.0);
        assert_eq!(pa.ax[0], 0.0);
        assert_eq!(pa.pressure[0], 0.0);
    }

    #[test]
    fn fluid_type_repr() {
        assert_eq!(FluidType::Water as u8, 0);
        assert_eq!(FluidType::Air as u8, 1);
    }
}
