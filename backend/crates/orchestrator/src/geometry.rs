//! STL geometry parsing and SDF generation

use std::fs::File;
use std::io::BufReader;

/// Triangle structure from nom_stl
pub use nom_stl::Triangle;

/// Grid-based signed distance field
#[derive(Debug, Clone)]
pub struct GridSDF {
    /// Grid origin (minimum corner)
    pub origin: [f32; 3],
    /// Voxel edge length
    pub cell_size: f32,
    /// Grid dimensions [nx, ny, nz]
    pub dimensions: [u32; 3],
    /// Flat array of signed distances (row-major: z varies fastest)
    pub distances: Vec<f32>,
}

/// Load an STL file and return its triangles
pub fn load_stl(path: &str) -> Result<Vec<Triangle>, String> {
    let file = File::open(path)
        .map_err(|e| format!("Failed to open STL file {}: {}", path, e))?;

    let mut reader = BufReader::new(file);

    let mesh = nom_stl::parse_stl(&mut reader)
        .map_err(|e| format!("Failed to parse STL file {}: {:?}", path, e))?;

    Ok(mesh.triangles().to_vec())
}

/// Generate a signed distance field from a mesh
pub fn mesh_to_sdf(
    triangles: &[Triangle],
    domain: &crate::config::DomainBounds,
    cell_size: f32,
) -> GridSDF {
    generate_sdf(triangles, domain.min, domain.max, cell_size)
}

/// Generate a signed distance field from triangles
pub fn generate_sdf(
    triangles: &[Triangle],
    domain_min: [f32; 3],
    domain_max: [f32; 3],
    cell_size: f32,
) -> GridSDF {
    // Calculate grid dimensions
    let nx = ((domain_max[0] - domain_min[0]) / cell_size).ceil() as u32 + 1;
    let ny = ((domain_max[1] - domain_min[1]) / cell_size).ceil() as u32 + 1;
    let nz = ((domain_max[2] - domain_min[2]) / cell_size).ceil() as u32 + 1;

    // Extract vertices and indices from triangles
    let mut vertices = Vec::new();
    let mut indices = Vec::new();

    for (tri_idx, triangle) in triangles.iter().enumerate() {
        let base_idx = tri_idx * 3;

        // Add vertices for this triangle
        vertices.push([triangle.vertices()[0][0], triangle.vertices()[0][1], triangle.vertices()[0][2]]);
        vertices.push([triangle.vertices()[1][0], triangle.vertices()[1][1], triangle.vertices()[1][2]]);
        vertices.push([triangle.vertices()[2][0], triangle.vertices()[2][1], triangle.vertices()[2][2]]);

        // Add indices
        indices.push(base_idx as u32);
        indices.push((base_idx + 1) as u32);
        indices.push((base_idx + 2) as u32);
    }

    // Create grid for mesh_to_sdf
    let grid = mesh_to_sdf::Grid::from_bounding_box(
        &domain_min,
        &domain_max,
        [nx as usize, ny as usize, nz as usize],
    );

    // Generate SDF using mesh_to_sdf
    let distances = mesh_to_sdf::generate_grid_sdf(
        &vertices,
        mesh_to_sdf::Topology::TriangleList(Some(&indices)),
        &grid,
        mesh_to_sdf::SignMethod::Raycast,
    );

    GridSDF {
        origin: domain_min,
        cell_size,
        dimensions: [nx, ny, nz],
        distances,
    }
}

impl GridSDF {
    /// Sample the SDF at a world-space point using trilinear interpolation
    pub fn sample(&self, point: [f32; 3]) -> f32 {
        // Convert world space to grid space
        let gx = (point[0] - self.origin[0]) / self.cell_size;
        let gy = (point[1] - self.origin[1]) / self.cell_size;
        let gz = (point[2] - self.origin[2]) / self.cell_size;

        // Get integer grid cell
        let i0 = gx.floor() as i32;
        let j0 = gy.floor() as i32;
        let k0 = gz.floor() as i32;

        // Check bounds
        if i0 < 0 || j0 < 0 || k0 < 0 {
            return f32::MAX; // Outside grid (far positive = outside)
        }

        let i1 = i0 + 1;
        let j1 = j0 + 1;
        let k1 = k0 + 1;

        if i1 >= self.dimensions[0] as i32
            || j1 >= self.dimensions[1] as i32
            || k1 >= self.dimensions[2] as i32
        {
            return f32::MAX; // Outside grid
        }

        // Interpolation weights
        let fx = gx - i0 as f32;
        let fy = gy - j0 as f32;
        let fz = gz - k0 as f32;

        // Trilinear interpolation
        let c000 = self.get_distance(i0 as u32, j0 as u32, k0 as u32);
        let c001 = self.get_distance(i0 as u32, j0 as u32, k1 as u32);
        let c010 = self.get_distance(i0 as u32, j1 as u32, k0 as u32);
        let c011 = self.get_distance(i0 as u32, j1 as u32, k1 as u32);
        let c100 = self.get_distance(i1 as u32, j0 as u32, k0 as u32);
        let c101 = self.get_distance(i1 as u32, j0 as u32, k1 as u32);
        let c110 = self.get_distance(i1 as u32, j1 as u32, k0 as u32);
        let c111 = self.get_distance(i1 as u32, j1 as u32, k1 as u32);

        let c00 = c000 * (1.0 - fz) + c001 * fz;
        let c01 = c010 * (1.0 - fz) + c011 * fz;
        let c10 = c100 * (1.0 - fz) + c101 * fz;
        let c11 = c110 * (1.0 - fz) + c111 * fz;

        let c0 = c00 * (1.0 - fy) + c01 * fy;
        let c1 = c10 * (1.0 - fy) + c11 * fy;

        c0 * (1.0 - fx) + c1 * fx
    }

    /// Compute the gradient (surface normal) at a point using central differences
    pub fn gradient(&self, point: [f32; 3]) -> [f32; 3] {
        let h = self.cell_size * 0.5;

        let dx = (self.sample([point[0] + h, point[1], point[2]])
            - self.sample([point[0] - h, point[1], point[2]]))
            / (2.0 * h);

        let dy = (self.sample([point[0], point[1] + h, point[2]])
            - self.sample([point[0], point[1] - h, point[2]]))
            / (2.0 * h);

        let dz = (self.sample([point[0], point[1], point[2] + h])
            - self.sample([point[0], point[1], point[2] - h]))
            / (2.0 * h);

        // Normalize to get unit normal
        let mag = (dx * dx + dy * dy + dz * dz).sqrt();
        if mag > 1e-8 {
            [dx / mag, dy / mag, dz / mag]
        } else {
            [0.0, 0.0, 0.0]
        }
    }

    /// Get distance value at grid indices
    fn get_distance(&self, i: u32, j: u32, k: u32) -> f32 {
        let idx = (i * self.dimensions[1] * self.dimensions[2] + j * self.dimensions[2] + k) as usize;
        self.distances.get(idx).copied().unwrap_or(f32::MAX)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_grid_sdf_sample() {
        // Create a simple 3x2x2 grid with known values
        // Grid indexing: distances[(i * ny * nz) + (j * nz) + k]
        let sdf = GridSDF {
            origin: [0.0, 0.0, 0.0],
            cell_size: 1.0,
            dimensions: [3, 2, 2],
            distances: vec![
                // i=0
                -1.0, -1.0, // j=0, k=0..1
                -1.0, -1.0, // j=1, k=0..1
                // i=1
                0.0, 0.0,   // j=0, k=0..1
                0.0, 0.0,   // j=1, k=0..1
                // i=2
                1.0, 1.0,   // j=0, k=0..1
                1.0, 1.0,   // j=1, k=0..1
            ],
        };

        // Sample at grid point should return exact value (with interpolation tolerance)
        let val_0 = sdf.sample([0.0, 0.0, 0.0]);
        assert!((val_0 - (-1.0)).abs() < 0.1, "Expected -1.0, got {}", val_0);

        // Sample in the middle of the grid (between i=0 and i=1)
        let mid = sdf.sample([0.5, 0.0, 0.0]);
        assert!((mid - (-0.5)).abs() < 0.1, "Expected ~-0.5, got {}", mid);

        // Sample between i=1 and i=2
        let val_15 = sdf.sample([1.5, 0.0, 0.0]);
        assert!((val_15 - 0.5).abs() < 0.1, "Expected ~0.5, got {}", val_15);
    }

    #[test]
    fn test_gradient() {
        // Create a simple gradient field
        let sdf = GridSDF {
            origin: [0.0, 0.0, 0.0],
            cell_size: 1.0,
            dimensions: [3, 3, 3],
            distances: vec![
                // Linear gradient in x direction: d = x - 1
                -1.0, -1.0, -1.0,  // x=0
                -1.0, -1.0, -1.0,
                -1.0, -1.0, -1.0,

                0.0, 0.0, 0.0,     // x=1
                0.0, 0.0, 0.0,
                0.0, 0.0, 0.0,

                1.0, 1.0, 1.0,     // x=2
                1.0, 1.0, 1.0,
                1.0, 1.0, 1.0,
            ],
        };

        let grad = sdf.gradient([1.0, 1.0, 1.0]);
        // Gradient should point in +x direction and be normalized
        assert!((grad[0] - 1.0).abs() < 0.1);
        assert!(grad[1].abs() < 0.1);
        assert!(grad[2].abs() < 0.1);
    }
}
