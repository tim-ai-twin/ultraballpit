//! GPU-friendly uniform-grid spatial hash for neighbor search.
//!
//! Uses sorted-index + cell-offset arrays rather than `HashMap` so the data
//! layout maps directly to GPU buffers (no pointer chasing).

/// Uniform-grid spatial hash for O(1) neighbor cell lookup.
///
/// The grid covers a fixed axis-aligned domain.  Cell size should equal the
/// kernel support radius (2h) so that for any particle the 27 (3x3x3)
/// adjacent cells contain all potential neighbors within distance 2h.
pub struct NeighborGrid {
    cell_size: f32,
    grid_min: [f32; 3],
    grid_dims: [u32; 3],
    /// Cell index for each particle (parallel to particle arrays).
    cell_indices: Vec<u32>,
    /// Particle indices sorted by cell index.
    sorted_indices: Vec<u32>,
    /// Start offset in `sorted_indices` for each cell.
    cell_offsets: Vec<u32>,
    /// Number of particles in each cell.
    cell_counts: Vec<u32>,
}

impl NeighborGrid {
    /// Create a new neighbor grid covering `[domain_min, domain_max]`.
    ///
    /// `cell_size` should be set to the kernel support radius (typically 2h).
    pub fn new(cell_size: f32, domain_min: [f32; 3], domain_max: [f32; 3]) -> Self {
        assert!(cell_size > 0.0, "cell_size must be positive");
        let dims = [
            ((domain_max[0] - domain_min[0]) / cell_size).ceil().max(1.0) as u32,
            ((domain_max[1] - domain_min[1]) / cell_size).ceil().max(1.0) as u32,
            ((domain_max[2] - domain_min[2]) / cell_size).ceil().max(1.0) as u32,
        ];
        let total_cells = (dims[0] as usize) * (dims[1] as usize) * (dims[2] as usize);
        Self {
            cell_size,
            grid_min: domain_min,
            grid_dims: dims,
            cell_indices: Vec::new(),
            sorted_indices: Vec::new(),
            cell_offsets: vec![0; total_cells],
            cell_counts: vec![0; total_cells],
        }
    }

    /// Total number of cells in the grid.
    fn total_cells(&self) -> usize {
        (self.grid_dims[0] as usize)
            * (self.grid_dims[1] as usize)
            * (self.grid_dims[2] as usize)
    }

    /// Map a world-space position to a cell (cx, cy, cz), clamped to grid bounds.
    #[inline]
    fn pos_to_cell(&self, px: f32, py: f32, pz: f32) -> (u32, u32, u32) {
        let cx = ((px - self.grid_min[0]) / self.cell_size)
            .floor()
            .max(0.0)
            .min((self.grid_dims[0] - 1) as f32) as u32;
        let cy = ((py - self.grid_min[1]) / self.cell_size)
            .floor()
            .max(0.0)
            .min((self.grid_dims[1] - 1) as f32) as u32;
        let cz = ((pz - self.grid_min[2]) / self.cell_size)
            .floor()
            .max(0.0)
            .min((self.grid_dims[2] - 1) as f32) as u32;
        (cx, cy, cz)
    }

    /// Flat cell index from (cx, cy, cz).
    #[inline]
    fn cell_hash(&self, cx: u32, cy: u32, cz: u32) -> u32 {
        cx + cy * self.grid_dims[0] + cz * self.grid_dims[0] * self.grid_dims[1]
    }

    /// Rebuild the grid from current particle positions.
    ///
    /// The three slices must all have the same length (one entry per particle).
    pub fn update(&mut self, x: &[f32], y: &[f32], z: &[f32]) {
        let n = x.len();
        debug_assert_eq!(n, y.len());
        debug_assert_eq!(n, z.len());

        let total_cells = self.total_cells();

        // --- 1. Compute cell index for each particle ---
        self.cell_indices.resize(n, 0);
        for i in 0..n {
            let (cx, cy, cz) = self.pos_to_cell(x[i], y[i], z[i]);
            self.cell_indices[i] = self.cell_hash(cx, cy, cz);
        }

        // --- 2. Count particles per cell ---
        self.cell_counts.clear();
        self.cell_counts.resize(total_cells, 0);
        for &ci in &self.cell_indices {
            self.cell_counts[ci as usize] += 1;
        }

        // --- 3. Prefix-sum to get cell offsets ---
        self.cell_offsets.clear();
        self.cell_offsets.resize(total_cells, 0);
        let mut running = 0u32;
        for c in 0..total_cells {
            self.cell_offsets[c] = running;
            running += self.cell_counts[c];
        }

        // --- 4. Scatter particle indices into sorted order ---
        self.sorted_indices.resize(n, 0);
        // We need a temporary write-head per cell; reuse a scratch copy of offsets.
        let mut write_heads: Vec<u32> = self.cell_offsets.clone();
        for i in 0..n {
            let ci = self.cell_indices[i] as usize;
            let pos = write_heads[ci] as usize;
            self.sorted_indices[pos] = i as u32;
            write_heads[ci] += 1;
        }
    }

    /// Iterate over all neighbors of `particle_idx` within `radius`.
    ///
    /// Checks the 27 (3x3x3) adjacent cells around the particle's cell.
    /// For each candidate, the caller's closure `f` is invoked with the neighbor
    /// particle index.  Distance filtering to `radius` is performed here.
    pub fn for_each_neighbor<F>(
        &self,
        particle_idx: usize,
        x: &[f32],
        y: &[f32],
        z: &[f32],
        radius: f32,
        mut f: F,
    ) where
        F: FnMut(usize),
    {
        let px = x[particle_idx];
        let py = y[particle_idx];
        let pz = z[particle_idx];
        let (cx, cy, cz) = self.pos_to_cell(px, py, pz);
        let radius_sq = radius * radius;

        // Iterate over 3x3x3 neighborhood
        for dz in -1i32..=1 {
            let nz = cz as i32 + dz;
            if nz < 0 || nz >= self.grid_dims[2] as i32 {
                continue;
            }
            for dy in -1i32..=1 {
                let ny = cy as i32 + dy;
                if ny < 0 || ny >= self.grid_dims[1] as i32 {
                    continue;
                }
                for dx in -1i32..=1 {
                    let nx = cx as i32 + dx;
                    if nx < 0 || nx >= self.grid_dims[0] as i32 {
                        continue;
                    }
                    let cell = self.cell_hash(nx as u32, ny as u32, nz as u32) as usize;
                    let start = self.cell_offsets[cell] as usize;
                    let count = self.cell_counts[cell] as usize;

                    for s in start..start + count {
                        let j = self.sorted_indices[s] as usize;
                        if j == particle_idx {
                            continue;
                        }
                        let ddx = px - x[j];
                        let ddy = py - y[j];
                        let ddz = pz - z[j];
                        let dist_sq = ddx * ddx + ddy * ddy + ddz * ddz;
                        if dist_sq <= radius_sq {
                            f(j);
                        }
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_grid() {
        let grid = NeighborGrid::new(0.1, [0.0; 3], [1.0; 3]);
        assert_eq!(grid.total_cells(), 10 * 10 * 10);
    }

    #[test]
    fn single_particle_no_neighbors() {
        let mut grid = NeighborGrid::new(0.2, [0.0; 3], [1.0; 3]);
        let x = [0.5];
        let y = [0.5];
        let z = [0.5];
        grid.update(&x, &y, &z);
        let mut neighbors = Vec::new();
        grid.for_each_neighbor(0, &x, &y, &z, 0.2, |j| neighbors.push(j));
        assert!(neighbors.is_empty());
    }

    #[test]
    fn two_close_particles() {
        let mut grid = NeighborGrid::new(0.2, [0.0; 3], [1.0; 3]);
        let x = [0.5, 0.51];
        let y = [0.5, 0.5];
        let z = [0.5, 0.5];
        grid.update(&x, &y, &z);

        let mut neighbors = Vec::new();
        grid.for_each_neighbor(0, &x, &y, &z, 0.2, |j| neighbors.push(j));
        assert_eq!(neighbors, vec![1]);

        let mut neighbors2 = Vec::new();
        grid.for_each_neighbor(1, &x, &y, &z, 0.2, |j| neighbors2.push(j));
        assert_eq!(neighbors2, vec![0]);
    }

    #[test]
    fn two_far_particles() {
        let mut grid = NeighborGrid::new(0.2, [0.0; 3], [1.0; 3]);
        let x = [0.1, 0.9];
        let y = [0.1, 0.9];
        let z = [0.1, 0.9];
        grid.update(&x, &y, &z);

        let mut neighbors = Vec::new();
        grid.for_each_neighbor(0, &x, &y, &z, 0.2, |j| neighbors.push(j));
        assert!(neighbors.is_empty());
    }

    #[test]
    fn particles_across_cell_boundary() {
        // Two particles in adjacent cells but within search radius
        let cell_size = 0.2;
        let mut grid = NeighborGrid::new(cell_size, [0.0; 3], [1.0; 3]);
        // Particle 0 near right edge of a cell, particle 1 near left edge of next cell
        let x = [0.19, 0.21];
        let y = [0.5, 0.5];
        let z = [0.5, 0.5];
        grid.update(&x, &y, &z);

        let mut neighbors = Vec::new();
        grid.for_each_neighbor(0, &x, &y, &z, cell_size, |j| neighbors.push(j));
        assert_eq!(neighbors, vec![1]);
    }

    #[test]
    fn many_particles_in_cluster() {
        let cell_size = 0.2;
        let mut grid = NeighborGrid::new(cell_size, [0.0; 3], [1.0; 3]);
        let n = 10;
        // Place particles in a small cluster
        let x: Vec<f32> = (0..n).map(|i| 0.5 + (i as f32) * 0.01).collect();
        let y: Vec<f32> = vec![0.5; n];
        let z: Vec<f32> = vec![0.5; n];
        grid.update(&x, &y, &z);

        // Particle 0 should see all others (they're all within 0.09 of each other)
        let mut neighbors = Vec::new();
        grid.for_each_neighbor(0, &x, &y, &z, 0.2, |j| neighbors.push(j));
        assert_eq!(neighbors.len(), n - 1);
    }
}
