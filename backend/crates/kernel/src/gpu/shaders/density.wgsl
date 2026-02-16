// SPH density summation compute shader
//
// Computes density for each particle using the Wendland C2 kernel:
//   rho_i = sum_j m_j * W(|r_i - r_j|, h)
//
// Includes self-contribution, fluid neighbor contributions (via neighbor grid),
// and boundary particle contributions.

const PI: f32 = 3.14159265358979323846;
const WENDLAND_C2_NORM_3D: f32 = 0.41780189; // 21 / (16 * PI)

// EOS constants
const WATER_REST_DENSITY: f32 = 1000.0;
const AIR_REST_DENSITY: f32 = 1.204;
const WATER_GAMMA: f32 = 7.0;
const AIR_GAMMA: f32 = 1.4;

struct SimParams {
    dt: f32,
    h: f32,
    speed_of_sound: f32,
    gravity_x: f32,
    gravity_y: f32,
    gravity_z: f32,
    domain_min_x: f32,
    domain_min_y: f32,
    domain_min_z: f32,
    domain_max_x: f32,
    domain_max_y: f32,
    domain_max_z: f32,
    n_particles: u32,
    n_boundary: u32,
    grid_dim_x: u32,
    grid_dim_y: u32,
    grid_dim_z: u32,
    cell_size: f32,
    viscosity_alpha: f32,
    viscosity_beta: f32,
    pass: u32,
    _pad1: u32,
};

@group(0) @binding(0) var<uniform> params: SimParams;

// Fluid particle data
@group(0) @binding(1) var<storage, read> pos_x: array<f32>;
@group(0) @binding(2) var<storage, read> pos_y: array<f32>;
@group(0) @binding(3) var<storage, read> pos_z: array<f32>;
@group(0) @binding(4) var<storage, read> mass: array<f32>;
@group(0) @binding(5) var<storage, read_write> density: array<f32>;
@group(0) @binding(6) var<storage, read_write> pressure: array<f32>;
@group(0) @binding(7) var<storage, read> fluid_type: array<u32>;

// Boundary particle data
@group(0) @binding(8) var<storage, read> bnd_x: array<f32>;
@group(0) @binding(9) var<storage, read> bnd_y: array<f32>;
@group(0) @binding(10) var<storage, read> bnd_z: array<f32>;
@group(0) @binding(11) var<storage, read> bnd_mass: array<f32>;

// Neighbor grid data
@group(0) @binding(12) var<storage, read> cell_offsets: array<u32>;
@group(0) @binding(13) var<storage, read> cell_counts: array<u32>;
@group(0) @binding(14) var<storage, read> sorted_indices: array<u32>;

fn wendland_c2(r: f32, h: f32) -> f32 {
    let q = r / h;
    if q >= 2.0 {
        return 0.0;
    }
    let h3 = h * h * h;
    let one_minus_half_q = 1.0 - 0.5 * q;
    let t = one_minus_half_q * one_minus_half_q;
    let t4 = t * t;
    return WENDLAND_C2_NORM_3D / h3 * t4 * (1.0 + 2.0 * q);
}

fn pos_to_cell_i32(px: f32, py: f32, pz: f32) -> vec3<i32> {
    let cx = i32(floor((px - params.domain_min_x) / params.cell_size));
    let cy = i32(floor((py - params.domain_min_y) / params.cell_size));
    let cz = i32(floor((pz - params.domain_min_z) / params.cell_size));
    return vec3<i32>(
        clamp(cx, 0, i32(params.grid_dim_x) - 1),
        clamp(cy, 0, i32(params.grid_dim_y) - 1),
        clamp(cz, 0, i32(params.grid_dim_z) - 1)
    );
}

fn cell_hash(cx: u32, cy: u32, cz: u32) -> u32 {
    return cx + cy * params.grid_dim_x + cz * params.grid_dim_x * params.grid_dim_y;
}

fn tait_eos(rho: f32, rho0: f32, cs: f32, gamma: f32) -> f32 {
    let b = rho0 * cs * cs / gamma;
    let ratio = rho / rho0;
    let p = b * (pow(ratio, gamma) - 1.0);
    return max(p, 0.0);
}

@compute @workgroup_size(256)
fn compute_density(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i >= params.n_particles {
        return;
    }

    let h = params.h;
    let support_radius = 2.0 * h;
    let support_radius_sq = support_radius * support_radius;

    let px = pos_x[i];
    let py = pos_y[i];
    let pz = pos_z[i];

    // Self-contribution
    var rho = mass[i] * wendland_c2(0.0, h);

    // Fluid neighbor contributions via neighbor grid
    let cell = pos_to_cell_i32(px, py, pz);

    for (var dz = -1; dz <= 1; dz = dz + 1) {
        let nz = cell.z + dz;
        if nz < 0 || nz >= i32(params.grid_dim_z) { continue; }
        for (var dy = -1; dy <= 1; dy = dy + 1) {
            let ny = cell.y + dy;
            if ny < 0 || ny >= i32(params.grid_dim_y) { continue; }
            for (var dx = -1; dx <= 1; dx = dx + 1) {
                let nx = cell.x + dx;
                if nx < 0 || nx >= i32(params.grid_dim_x) { continue; }

                let c = cell_hash(u32(nx), u32(ny), u32(nz));
                let start = cell_offsets[c];
                let count = cell_counts[c];

                for (var s = start; s < start + count; s = s + 1u) {
                    let j = sorted_indices[s];
                    if j == i { continue; }

                    let ddx = px - pos_x[j];
                    let ddy = py - pos_y[j];
                    let ddz = pz - pos_z[j];
                    let dist_sq = ddx * ddx + ddy * ddy + ddz * ddz;

                    if dist_sq <= support_radius_sq {
                        let r = sqrt(dist_sq);
                        rho = rho + mass[j] * wendland_c2(r, h);
                    }
                }
            }
        }
    }

    // Boundary particle contributions
    for (var b = 0u; b < params.n_boundary; b = b + 1u) {
        let ddx = px - bnd_x[b];
        let ddy = py - bnd_y[b];
        let ddz = pz - bnd_z[b];
        let dist_sq = ddx * ddx + ddy * ddy + ddz * ddz;

        if dist_sq < support_radius_sq {
            let r = sqrt(dist_sq);
            rho = rho + bnd_mass[b] * wendland_c2(r, h);
        }
    }

    density[i] = rho;

    // Also compute pressure from EOS here for efficiency
    let ft = fluid_type[i];
    if ft == 0u {
        // Water
        pressure[i] = tait_eos(rho, WATER_REST_DENSITY, params.speed_of_sound, WATER_GAMMA);
    } else {
        // Air
        pressure[i] = tait_eos(rho, AIR_REST_DENSITY, params.speed_of_sound, AIR_GAMMA);
    }
}
