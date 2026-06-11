#[cfg(feature = "gpu")]
fn main() {
    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
        backends: wgpu::Backends::all(),
        ..Default::default()
    });
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        ..Default::default()
    })).expect("No adapter");

    println!("Adapter: {}", adapter.get_info().name);
    println!("Backend: {:?}", adapter.get_info().backend);

    let features = adapter.features();
    println!("\nRelevant features:");
    println!("  TIMESTAMP_QUERY:       {}", features.contains(wgpu::Features::TIMESTAMP_QUERY));
    println!("  SHADER_F16:            {}", features.contains(wgpu::Features::SHADER_F16));
    println!("  FLOAT32_FILTERABLE:    {}", features.contains(wgpu::Features::FLOAT32_FILTERABLE));

    let limits = adapter.limits();
    println!("\nRelevant limits:");
    println!("  max_compute_workgroup_size_x:       {}", limits.max_compute_workgroup_size_x);
    println!("  max_compute_workgroup_size_y:       {}", limits.max_compute_workgroup_size_y);
    println!("  max_compute_workgroup_size_z:       {}", limits.max_compute_workgroup_size_z);
    println!("  max_compute_invocations_per_workgroup: {}", limits.max_compute_invocations_per_workgroup);
    println!("  max_compute_workgroups_per_dimension: {}", limits.max_compute_workgroups_per_dimension);
    println!("  max_storage_buffers_per_shader_stage: {}", limits.max_storage_buffers_per_shader_stage);
    println!("  max_buffer_size:                    {}", limits.max_buffer_size);
}

#[cfg(not(feature = "gpu"))]
fn main() {
    println!("GPU feature not enabled");
}
