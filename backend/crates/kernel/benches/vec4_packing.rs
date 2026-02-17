//! Microbenchmark: SoA (4 separate f32 arrays) vs vec4 packed access patterns.
//!
//! Tests two access patterns:
//! 1. "Local" - neighbors are nearby in index space (simulates spatial sorting)
//! 2. "Scattered" - neighbors are random (worst case)
//!
//! Run with: cargo bench --features gpu -p kernel --bench vec4_packing

#![cfg(feature = "gpu")]

use std::time::Instant;

fn main() {
    println!("=== Vec4 Packing Benchmark ===\n");

    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
        backends: wgpu::Backends::all(),
        ..Default::default()
    });
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        ..Default::default()
    }))
    .expect("No GPU adapter found");

    println!("Adapter: {}", adapter.get_info().name);

    let mut limits = wgpu::Limits::default();
    limits.max_storage_buffers_per_shader_stage =
        adapter.limits().max_storage_buffers_per_shader_stage;

    let (device, queue) = pollster::block_on(adapter.request_device(
        &wgpu::DeviceDescriptor {
            label: Some("vec4_bench"),
            required_features: wgpu::Features::empty(),
            required_limits: limits,
            memory_hints: wgpu::MemoryHints::Performance,
        },
        None,
    ))
    .expect("Failed to create device");

    let particle_counts = [8_000, 27_000, 64_000, 125_000];
    let n_neighbors = 30u32;

    // Test 1: Spatially local neighbors (nearby indices)
    println!("--- Spatially local neighbors (simulates grid-sorted access) ---");
    println!(
        "{:>10} {:>8} {:>12} {:>12} {:>10}",
        "Particles", "Iters", "SoA (ms)", "Vec4 (ms)", "Speedup"
    );

    for &n in &particle_counts {
        let iters = if n <= 27_000 { 100 } else if n <= 64_000 { 50 } else { 20 };
        let neighbors = generate_local_neighbors(n, n_neighbors);
        let (soa_ms, vec4_ms) =
            bench_layouts(&device, &queue, n, n_neighbors, iters, &neighbors);
        let speedup = soa_ms / vec4_ms;
        println!(
            "{:>10} {:>8} {:>12.3} {:>12.3} {:>9.2}x",
            n, iters, soa_ms, vec4_ms, speedup
        );
    }

    // Test 2: Scattered neighbors (random indices)
    println!("\n--- Scattered neighbors (worst case, random access) ---");
    println!(
        "{:>10} {:>8} {:>12} {:>12} {:>10}",
        "Particles", "Iters", "SoA (ms)", "Vec4 (ms)", "Speedup"
    );

    for &n in &particle_counts {
        let iters = if n <= 27_000 { 100 } else if n <= 64_000 { 50 } else { 20 };
        let neighbors = generate_scattered_neighbors(n, n_neighbors);
        let (soa_ms, vec4_ms) =
            bench_layouts(&device, &queue, n, n_neighbors, iters, &neighbors);
        let speedup = soa_ms / vec4_ms;
        println!(
            "{:>10} {:>8} {:>12.3} {:>12.3} {:>9.2}x",
            n, iters, soa_ms, vec4_ms, speedup
        );
    }

    // Test 3: Also test with larger neighbor reads (pos + vel + density + pressure = 10 fields)
    println!("\n--- Full field access: SoA (10 buffers) vs packed (3 vec4 buffers) ---");
    println!(
        "{:>10} {:>8} {:>12} {:>12} {:>10}",
        "Particles", "Iters", "SoA (ms)", "Vec4 (ms)", "Speedup"
    );

    for &n in &particle_counts {
        let iters = if n <= 27_000 { 100 } else if n <= 64_000 { 50 } else { 20 };
        let neighbors = generate_local_neighbors(n, n_neighbors);
        let (soa_ms, vec4_ms) =
            bench_full_field(&device, &queue, n, n_neighbors, iters, &neighbors);
        let speedup = soa_ms / vec4_ms;
        println!(
            "{:>10} {:>8} {:>12.3} {:>12.3} {:>9.2}x",
            n, iters, soa_ms, vec4_ms, speedup
        );
    }
}

/// Generate spatially-local neighbor indices (nearby in memory, like grid-sorted).
fn generate_local_neighbors(n: usize, n_neighbors: u32) -> Vec<u32> {
    let total = n * n_neighbors as usize;
    let mut ids = vec![0u32; total];
    for i in 0..n {
        for k in 0..n_neighbors as usize {
            // Neighbors are within ±50 of current index (simulates spatial sorting)
            let offset = (k as i64 - n_neighbors as i64 / 2) * 3;
            let j = ((i as i64 + offset).rem_euclid(n as i64)) as u32;
            ids[i * n_neighbors as usize + k] = j;
        }
    }
    ids
}

/// Generate scattered neighbor indices (pseudo-random, poor locality).
fn generate_scattered_neighbors(n: usize, n_neighbors: u32) -> Vec<u32> {
    let total = n * n_neighbors as usize;
    let mut ids = vec![0u32; total];
    for i in 0..n {
        for k in 0..n_neighbors as usize {
            // LCG-style scatter: large strides through the buffer
            let j = ((i * 7919 + k * 104729 + 31) % n) as u32;
            ids[i * n_neighbors as usize + k] = j;
        }
    }
    ids
}

const SHADER_SOA: &str = r#"
struct Params {
    n: u32,
    n_neighbors: u32,
    _pad0: u32,
    _pad1: u32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> pos_x: array<f32>;
@group(0) @binding(2) var<storage, read> pos_y: array<f32>;
@group(0) @binding(3) var<storage, read> pos_z: array<f32>;
@group(0) @binding(4) var<storage, read> mass: array<f32>;
@group(0) @binding(5) var<storage, read> neighbor_ids: array<u32>;
@group(0) @binding(6) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(256)
fn main_soa(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i >= params.n { return; }

    let px = pos_x[i];
    let py = pos_y[i];
    let pz = pos_z[i];

    var acc = 0.0;
    let base = i * params.n_neighbors;

    for (var k = 0u; k < params.n_neighbors; k = k + 1u) {
        let j = neighbor_ids[base + k];
        let dx = px - pos_x[j];
        let dy = py - pos_y[j];
        let dz = pz - pos_z[j];
        let dist_sq = dx * dx + dy * dy + dz * dz;
        let r = sqrt(dist_sq + 1e-10);
        acc = acc + mass[j] / (r * r * r + 1e-6);
    }

    output[i] = acc;
}
"#;

const SHADER_VEC4: &str = r#"
struct Params {
    n: u32,
    n_neighbors: u32,
    _pad0: u32,
    _pad1: u32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> pos_mass: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> neighbor_ids: array<u32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(256)
fn main_vec4(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i >= params.n { return; }

    let pi = pos_mass[i];
    let px = pi.x;
    let py = pi.y;
    let pz = pi.z;

    var acc = 0.0;
    let base = i * params.n_neighbors;

    for (var k = 0u; k < params.n_neighbors; k = k + 1u) {
        let j = neighbor_ids[base + k];
        let pj = pos_mass[j];
        let dx = px - pj.x;
        let dy = py - pj.y;
        let dz = pz - pj.z;
        let dist_sq = dx * dx + dy * dy + dz * dz;
        let r = sqrt(dist_sq + 1e-10);
        acc = acc + pj.w / (r * r * r + 1e-6);
    }

    output[i] = acc;
}
"#;

/// Full-field SoA: 10 separate buffers (pos_xyz, vel_xyz, mass, density, pressure, fluid_type)
const SHADER_FULL_SOA: &str = r#"
struct Params {
    n: u32,
    n_neighbors: u32,
    _pad0: u32,
    _pad1: u32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> pos_x: array<f32>;
@group(0) @binding(2) var<storage, read> pos_y: array<f32>;
@group(0) @binding(3) var<storage, read> pos_z: array<f32>;
@group(0) @binding(4) var<storage, read> mass: array<f32>;
@group(0) @binding(5) var<storage, read> vel_x: array<f32>;
@group(0) @binding(6) var<storage, read> vel_y: array<f32>;
@group(0) @binding(7) var<storage, read> vel_z: array<f32>;
@group(0) @binding(8) var<storage, read> density: array<f32>;

@group(1) @binding(0) var<storage, read> pressure: array<f32>;
@group(1) @binding(1) var<storage, read> neighbor_ids: array<u32>;
@group(1) @binding(2) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(256)
fn main_full_soa(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i >= params.n { return; }

    let px = pos_x[i]; let py = pos_y[i]; let pz = pos_z[i];
    let rho_i = density[i]; let p_i = pressure[i];

    var fx = 0.0; var fy = 0.0; var fz = 0.0;
    let base = i * params.n_neighbors;

    for (var k = 0u; k < params.n_neighbors; k = k + 1u) {
        let j = neighbor_ids[base + k];
        let dx = px - pos_x[j]; let dy = py - pos_y[j]; let dz = pz - pos_z[j];
        let dist_sq = dx * dx + dy * dy + dz * dz;
        let r = sqrt(dist_sq + 1e-10);
        let inv_r3 = 1.0 / (r * r * r + 1e-6);

        // Pressure force (reads mass, density, pressure of neighbor)
        let m_j = mass[j];
        let rho_j = density[j];
        let p_j = pressure[j];
        let p_factor = -m_j * (p_i / (rho_i * rho_i) + p_j / (rho_j * rho_j));
        fx = fx + p_factor * dx * inv_r3;
        fy = fy + p_factor * dy * inv_r3;
        fz = fz + p_factor * dz * inv_r3;

        // Viscous force (reads velocity)
        let dvx = vel_x[i] - vel_x[j];
        let dvy = vel_y[i] - vel_y[j];
        let dvz = vel_z[i] - vel_z[j];
        let vr_dot = dvx * dx + dvy * dy + dvz * dz;
        if vr_dot < 0.0 {
            let mu = 0.01 * vr_dot / (dist_sq + 0.01);
            fx = fx - m_j * mu * dx * inv_r3;
            fy = fy - m_j * mu * dy * inv_r3;
            fz = fz - m_j * mu * dz * inv_r3;
        }
    }

    output[i] = fx + fy + fz;
}
"#;

/// Full-field packed: 3 vec4 buffers (pos_mass, vel_pad, density_pressure_pad)
const SHADER_FULL_VEC4: &str = r#"
struct Params {
    n: u32,
    n_neighbors: u32,
    _pad0: u32,
    _pad1: u32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> pos_mass: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> vel_pad: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> density_pressure: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read> neighbor_ids: array<u32>;
@group(0) @binding(5) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(256)
fn main_full_vec4(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i >= params.n { return; }

    let pi = pos_mass[i];
    let px = pi.x; let py = pi.y; let pz = pi.z;
    let dp_i = density_pressure[i];
    let rho_i = dp_i.x; let p_i = dp_i.y;

    var fx = 0.0; var fy = 0.0; var fz = 0.0;
    let base = i * params.n_neighbors;

    for (var k = 0u; k < params.n_neighbors; k = k + 1u) {
        let j = neighbor_ids[base + k];
        let pj = pos_mass[j];
        let dx = px - pj.x; let dy = py - pj.y; let dz = pz - pj.z;
        let dist_sq = dx * dx + dy * dy + dz * dz;
        let r = sqrt(dist_sq + 1e-10);
        let inv_r3 = 1.0 / (r * r * r + 1e-6);

        // Pressure force (mass from pj.w, density/pressure from packed buffer)
        let m_j = pj.w;
        let dp_j = density_pressure[j];
        let rho_j = dp_j.x;
        let p_j = dp_j.y;
        let p_factor = -m_j * (p_i / (rho_i * rho_i) + p_j / (rho_j * rho_j));
        fx = fx + p_factor * dx * inv_r3;
        fy = fy + p_factor * dy * inv_r3;
        fz = fz + p_factor * dz * inv_r3;

        // Viscous force (velocity from packed buffer)
        let vi = vel_pad[i];
        let vj = vel_pad[j];
        let dvx = vi.x - vj.x;
        let dvy = vi.y - vj.y;
        let dvz = vi.z - vj.z;
        let vr_dot = dvx * dx + dvy * dy + dvz * dz;
        if vr_dot < 0.0 {
            let mu = 0.01 * vr_dot / (dist_sq + 0.01);
            fx = fx - m_j * mu * dx * inv_r3;
            fy = fy - m_j * mu * dy * inv_r3;
            fz = fz - m_j * mu * dz * inv_r3;
        }
    }

    output[i] = fx + fy + fz;
}
"#;

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct BenchParams {
    n: u32,
    n_neighbors: u32,
    _pad0: u32,
    _pad1: u32,
}

fn bench_layouts(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    n: usize,
    n_neighbors: u32,
    iters: usize,
    neighbor_ids: &[u32],
) -> (f64, f64) {
    use wgpu::util::DeviceExt;

    let (pos_x, pos_y, pos_z, mass_data, pos_mass) = generate_particle_data(n);

    let params = BenchParams { n: n as u32, n_neighbors, _pad0: 0, _pad1: 0 };

    // --- SoA buffers ---
    let buf_params = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("params"), contents: bytemuck::bytes_of(&params),
        usage: wgpu::BufferUsages::UNIFORM,
    });
    let mk = |label, data: &[f32]| device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some(label), contents: bytemuck::cast_slice(data),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let buf_pos_x = mk("pos_x", &pos_x);
    let buf_pos_y = mk("pos_y", &pos_y);
    let buf_pos_z = mk("pos_z", &pos_z);
    let buf_mass = mk("mass", &mass_data);
    let buf_neighbors = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("neighbors"), contents: bytemuck::cast_slice(neighbor_ids),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let buf_output_soa = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("output_soa"), size: (n * 4) as u64,
        usage: wgpu::BufferUsages::STORAGE, mapped_at_creation: false,
    });

    // --- Vec4 buffers ---
    let buf_params2 = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("params2"), contents: bytemuck::bytes_of(&params),
        usage: wgpu::BufferUsages::UNIFORM,
    });
    let buf_pos_mass = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("pos_mass"), contents: bytemuck::cast_slice(&pos_mass),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let buf_neighbors2 = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("neighbors2"), contents: bytemuck::cast_slice(neighbor_ids),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let buf_output_vec4 = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("output_vec4"), size: (n * 4) as u64,
        usage: wgpu::BufferUsages::STORAGE, mapped_at_creation: false,
    });

    // --- SoA pipeline ---
    let shader_soa = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("soa"), source: wgpu::ShaderSource::Wgsl(SHADER_SOA.into()),
    });
    let bgl_soa = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("soa_bgl"),
        entries: &[
            uniform_entry(0), storage_entry(1, true), storage_entry(2, true),
            storage_entry(3, true), storage_entry(4, true), storage_entry(5, true),
            storage_entry(6, false),
        ],
    });
    let bg_soa = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("soa_bg"), layout: &bgl_soa,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: buf_params.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: buf_pos_x.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: buf_pos_y.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: buf_pos_z.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 4, resource: buf_mass.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 5, resource: buf_neighbors.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 6, resource: buf_output_soa.as_entire_binding() },
        ],
    });
    let pipeline_soa = make_pipeline(device, &bgl_soa, &shader_soa, "main_soa", "soa_pl");

    // --- Vec4 pipeline ---
    let shader_vec4 = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("vec4"), source: wgpu::ShaderSource::Wgsl(SHADER_VEC4.into()),
    });
    let bgl_vec4 = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("vec4_bgl"),
        entries: &[
            uniform_entry(0), storage_entry(1, true),
            storage_entry(2, true), storage_entry(3, false),
        ],
    });
    let bg_vec4 = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("vec4_bg"), layout: &bgl_vec4,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: buf_params2.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: buf_pos_mass.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: buf_neighbors2.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: buf_output_vec4.as_entire_binding() },
        ],
    });
    let pipeline_vec4 = make_pipeline(device, &bgl_vec4, &shader_vec4, "main_vec4", "vec4_pl");

    let wg_count = (n as u32 + 255) / 256;

    // Warmup
    for _ in 0..5 {
        let mut enc = device.create_command_encoder(&Default::default());
        dispatch(&mut enc, &pipeline_soa, &bg_soa, wg_count);
        dispatch(&mut enc, &pipeline_vec4, &bg_vec4, wg_count);
        queue.submit(std::iter::once(enc.finish()));
        device.poll(wgpu::Maintain::Wait);
    }

    // Benchmark SoA (batch all dispatches into one submit)
    let start = Instant::now();
    let mut enc = device.create_command_encoder(&Default::default());
    for _ in 0..iters {
        dispatch(&mut enc, &pipeline_soa, &bg_soa, wg_count);
    }
    queue.submit(std::iter::once(enc.finish()));
    device.poll(wgpu::Maintain::Wait);
    let soa_ms = start.elapsed().as_secs_f64() * 1000.0 / iters as f64;

    // Benchmark vec4
    let start = Instant::now();
    let mut enc = device.create_command_encoder(&Default::default());
    for _ in 0..iters {
        dispatch(&mut enc, &pipeline_vec4, &bg_vec4, wg_count);
    }
    queue.submit(std::iter::once(enc.finish()));
    device.poll(wgpu::Maintain::Wait);
    let vec4_ms = start.elapsed().as_secs_f64() * 1000.0 / iters as f64;

    (soa_ms, vec4_ms)
}

fn bench_full_field(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    n: usize,
    n_neighbors: u32,
    iters: usize,
    neighbor_ids: &[u32],
) -> (f64, f64) {
    use wgpu::util::DeviceExt;

    let (pos_x, pos_y, pos_z, mass_data, pos_mass) = generate_particle_data(n);
    let vel_x: Vec<f32> = (0..n).map(|i| (i as f32 * 0.01).sin()).collect();
    let vel_y: Vec<f32> = (0..n).map(|i| (i as f32 * 0.02).cos()).collect();
    let vel_z: Vec<f32> = (0..n).map(|i| (i as f32 * 0.03).sin()).collect();
    let density_data: Vec<f32> = (0..n).map(|i| 1000.0 + (i as f32 * 0.1).sin() * 10.0).collect();
    let pressure_data: Vec<f32> = (0..n).map(|i| 100.0 + (i as f32 * 0.05).cos() * 50.0).collect();

    // Packed versions
    let vel_pad: Vec<[f32; 4]> = (0..n).map(|i| [vel_x[i], vel_y[i], vel_z[i], 0.0]).collect();
    let density_pressure: Vec<[f32; 4]> = (0..n).map(|i| [density_data[i], pressure_data[i], 0.0, 0.0]).collect();

    let params = BenchParams { n: n as u32, n_neighbors, _pad0: 0, _pad1: 0 };

    let mk = |label: &str, data: &[f32]| device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some(label), contents: bytemuck::cast_slice(data),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let mk4 = |label: &str, data: &[[f32; 4]]| device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some(label), contents: bytemuck::cast_slice(data),
        usage: wgpu::BufferUsages::STORAGE,
    });

    // --- Full SoA (10 buffers across 2 groups) ---
    let buf_params_soa = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("params_soa"), contents: bytemuck::bytes_of(&params),
        usage: wgpu::BufferUsages::UNIFORM,
    });
    let bufs_soa_g0: Vec<wgpu::Buffer> = vec![
        mk("pos_x", &pos_x), mk("pos_y", &pos_y), mk("pos_z", &pos_z),
        mk("mass", &mass_data), mk("vel_x", &vel_x), mk("vel_y", &vel_y),
        mk("vel_z", &vel_z), mk("density", &density_data),
    ];
    let buf_pressure = mk("pressure", &pressure_data);
    let buf_neighbors_soa = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("neighbors"), contents: bytemuck::cast_slice(neighbor_ids),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let buf_output_soa = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("output_soa"), size: (n * 4) as u64,
        usage: wgpu::BufferUsages::STORAGE, mapped_at_creation: false,
    });

    let shader_soa = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("full_soa"), source: wgpu::ShaderSource::Wgsl(SHADER_FULL_SOA.into()),
    });
    let bgl_soa_g0 = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("full_soa_g0"),
        entries: &[
            uniform_entry(0),
            storage_entry(1, true), storage_entry(2, true), storage_entry(3, true),
            storage_entry(4, true), storage_entry(5, true), storage_entry(6, true),
            storage_entry(7, true), storage_entry(8, true),
        ],
    });
    let bgl_soa_g1 = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("full_soa_g1"),
        entries: &[storage_entry(0, true), storage_entry(1, true), storage_entry(2, false)],
    });
    let bg_soa_g0 = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("full_soa_bg0"), layout: &bgl_soa_g0,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: buf_params_soa.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: bufs_soa_g0[0].as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: bufs_soa_g0[1].as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: bufs_soa_g0[2].as_entire_binding() },
            wgpu::BindGroupEntry { binding: 4, resource: bufs_soa_g0[3].as_entire_binding() },
            wgpu::BindGroupEntry { binding: 5, resource: bufs_soa_g0[4].as_entire_binding() },
            wgpu::BindGroupEntry { binding: 6, resource: bufs_soa_g0[5].as_entire_binding() },
            wgpu::BindGroupEntry { binding: 7, resource: bufs_soa_g0[6].as_entire_binding() },
            wgpu::BindGroupEntry { binding: 8, resource: bufs_soa_g0[7].as_entire_binding() },
        ],
    });
    let bg_soa_g1 = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("full_soa_bg1"), layout: &bgl_soa_g1,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: buf_pressure.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: buf_neighbors_soa.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: buf_output_soa.as_entire_binding() },
        ],
    });
    let pl_soa = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("full_soa_pl"),
        bind_group_layouts: &[&bgl_soa_g0, &bgl_soa_g1],
        push_constant_ranges: &[],
    });
    let pipeline_soa = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("full_soa_pipe"), layout: Some(&pl_soa),
        module: &shader_soa, entry_point: Some("main_full_soa"),
        compilation_options: Default::default(), cache: None,
    });

    // --- Full vec4 packed (3 vec4 buffers in 1 group) ---
    let buf_params_vec4 = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("params_vec4"), contents: bytemuck::bytes_of(&params),
        usage: wgpu::BufferUsages::UNIFORM,
    });
    let buf_pos_mass = mk4("pos_mass_f", &pos_mass);
    let buf_vel_pad = mk4("vel_pad", &vel_pad);
    let buf_dp = mk4("density_pressure", &density_pressure);
    let buf_neighbors_vec4 = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("neighbors_v4"), contents: bytemuck::cast_slice(neighbor_ids),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let buf_output_vec4 = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("output_vec4"), size: (n * 4) as u64,
        usage: wgpu::BufferUsages::STORAGE, mapped_at_creation: false,
    });

    let shader_vec4 = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("full_vec4"), source: wgpu::ShaderSource::Wgsl(SHADER_FULL_VEC4.into()),
    });
    let bgl_vec4 = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("full_vec4_bgl"),
        entries: &[
            uniform_entry(0), storage_entry(1, true), storage_entry(2, true),
            storage_entry(3, true), storage_entry(4, true), storage_entry(5, false),
        ],
    });
    let bg_vec4 = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("full_vec4_bg"), layout: &bgl_vec4,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: buf_params_vec4.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: buf_pos_mass.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: buf_vel_pad.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: buf_dp.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 4, resource: buf_neighbors_vec4.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 5, resource: buf_output_vec4.as_entire_binding() },
        ],
    });
    let pipeline_vec4 = make_pipeline(device, &bgl_vec4, &shader_vec4, "main_full_vec4", "full_vec4_pl");

    let wg_count = (n as u32 + 255) / 256;

    // Warmup
    for _ in 0..5 {
        let mut enc = device.create_command_encoder(&Default::default());
        {
            let mut pass = enc.begin_compute_pass(&Default::default());
            pass.set_pipeline(&pipeline_soa);
            pass.set_bind_group(0, &bg_soa_g0, &[]);
            pass.set_bind_group(1, &bg_soa_g1, &[]);
            pass.dispatch_workgroups(wg_count, 1, 1);
        }
        dispatch(&mut enc, &pipeline_vec4, &bg_vec4, wg_count);
        queue.submit(std::iter::once(enc.finish()));
        device.poll(wgpu::Maintain::Wait);
    }

    // Benchmark SoA
    let start = Instant::now();
    let mut enc = device.create_command_encoder(&Default::default());
    for _ in 0..iters {
        let mut pass = enc.begin_compute_pass(&Default::default());
        pass.set_pipeline(&pipeline_soa);
        pass.set_bind_group(0, &bg_soa_g0, &[]);
        pass.set_bind_group(1, &bg_soa_g1, &[]);
        pass.dispatch_workgroups(wg_count, 1, 1);
    }
    queue.submit(std::iter::once(enc.finish()));
    device.poll(wgpu::Maintain::Wait);
    let soa_ms = start.elapsed().as_secs_f64() * 1000.0 / iters as f64;

    // Benchmark vec4
    let start = Instant::now();
    let mut enc = device.create_command_encoder(&Default::default());
    for _ in 0..iters {
        dispatch(&mut enc, &pipeline_vec4, &bg_vec4, wg_count);
    }
    queue.submit(std::iter::once(enc.finish()));
    device.poll(wgpu::Maintain::Wait);
    let vec4_ms = start.elapsed().as_secs_f64() * 1000.0 / iters as f64;

    (soa_ms, vec4_ms)
}

fn generate_particle_data(n: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<[f32; 4]>) {
    let mut pos_x = vec![0.0f32; n];
    let mut pos_y = vec![0.0f32; n];
    let mut pos_z = vec![0.0f32; n];
    let mut mass = vec![0.0f32; n];
    let mut pos_mass = vec![[0.0f32; 4]; n];

    for i in 0..n {
        let fi = i as f32;
        pos_x[i] = (fi * 0.001).sin();
        pos_y[i] = (fi * 0.002).cos();
        pos_z[i] = (fi * 0.003).sin();
        mass[i] = 0.001 + (fi * 0.0001).sin().abs() * 0.0001;
        pos_mass[i] = [pos_x[i], pos_y[i], pos_z[i], mass[i]];
    }
    (pos_x, pos_y, pos_z, mass, pos_mass)
}

fn uniform_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn storage_entry(binding: u32, read_only: bool) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn make_pipeline(
    device: &wgpu::Device,
    bgl: &wgpu::BindGroupLayout,
    shader: &wgpu::ShaderModule,
    entry: &str,
    label: &str,
) -> wgpu::ComputePipeline {
    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some(label),
        bind_group_layouts: &[bgl],
        push_constant_ranges: &[],
    });
    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some(label),
        layout: Some(&layout),
        module: shader,
        entry_point: Some(entry),
        compilation_options: Default::default(),
        cache: None,
    })
}

fn dispatch(
    encoder: &mut wgpu::CommandEncoder,
    pipeline: &wgpu::ComputePipeline,
    bind_group: &wgpu::BindGroup,
    wg_count: u32,
) {
    let mut pass = encoder.begin_compute_pass(&Default::default());
    pass.set_pipeline(pipeline);
    pass.set_bind_group(0, bind_group, &[]);
    pass.dispatch_workgroups(wg_count, 1, 1);
}
