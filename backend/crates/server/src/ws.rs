//! WebSocket endpoint for real-time simulation streaming

use axum::{
    extract::{
        ws::{Message, WebSocket},
        Path, State, WebSocketUpgrade,
    },
    response::IntoResponse,
};
use futures_util::{SinkExt, StreamExt};
use kernel::FluidType;
use std::sync::Arc;
use tokio::time::{interval, Duration};

use crate::state::{AppState, SimStatus};

// ---------------------------------------------------------------------------
// Binary Protocol Tags
// ---------------------------------------------------------------------------

const TAG_SIM_INFO: u8 = 0x01;
const TAG_FRAME: u8 = 0x02;
const TAG_DIAGNOSTICS: u8 = 0x03;
const TAG_SIM_STATUS: u8 = 0x04;

const CMD_PAUSE: u8 = 0x01;
const CMD_RESUME: u8 = 0x02;
const CMD_ENABLE_DIAGNOSTICS: u8 = 0x04;
const CMD_DISABLE_DIAGNOSTICS: u8 = 0x05;

// ---------------------------------------------------------------------------
// WebSocket Handler
// ---------------------------------------------------------------------------

/// WebSocket upgrade handler for /ws/simulation/{id}
pub async fn ws_simulation_handler(
    ws: WebSocketUpgrade,
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
) -> impl IntoResponse {
    // Verify simulation exists
    let exists = state.simulations.lock().unwrap().contains_key(&id);
    if !exists {
        return (axum::http::StatusCode::NOT_FOUND, "Simulation not found").into_response();
    }

    ws.on_upgrade(move |socket| handle_websocket(socket, state, id))
}

/// Handle WebSocket connection
async fn handle_websocket(socket: WebSocket, state: Arc<AppState>, sim_id: String) {
    use futures_util::stream::SplitSink;
    use futures_util::stream::SplitStream;

    let (mut sender, mut receiver): (SplitSink<WebSocket, Message>, SplitStream<WebSocket>) = socket.split();

    // Per-connection diagnostics state
    let mut diagnostics_enabled = false;

    // Verify simulation exists
    {
        let sims = state.simulations.lock().unwrap();
        if !sims.contains_key(&sim_id) {
            tracing::error!("Simulation {} not found", sim_id);
            return;
        }
    }

    // Send initial SimInfo
    let sim_info = {
        let sims = state.simulations.lock().unwrap();
        let runner = match sims.get(&sim_id) {
            Some(r) => r,
            None => return,
        };

        build_sim_info(runner)
    };

    if let Err(e) = sender.send(Message::Binary(sim_info)).await {
        tracing::error!("Failed to send SimInfo: {}", e);
        return;
    }

    // Start the simulation
    {
        let sims = state.simulations.lock().unwrap();
        if let Some(runner) = sims.get(&sim_id) {
            runner.start();
        }
    }

    // Frame buffer for flow control (max 5 pending frames)
    let mut pending_frames = Vec::new();
    const MAX_PENDING: usize = 5;

    // Frame generation interval (~60 FPS)
    let mut frame_timer = interval(Duration::from_millis(16));

    // Simulation step interval (run simulation faster than frame rate)
    let mut sim_timer = interval(Duration::from_millis(1));

    loop {
        tokio::select! {
            // Simulation step
            _ = sim_timer.tick() => {
                let sims = state.simulations.lock().unwrap();
                if let Some(runner) = sims.get(&sim_id) {
                    runner.step();
                }
            }

            // Frame generation and sending
            _ = frame_timer.tick() => {
                // Build frame and diagnostics (scoped lock)
                let (frame_data, diagnostics_data) = {
                    let sims = state.simulations.lock().unwrap();
                    let runner = match sims.get(&sim_id) {
                        Some(r) => r,
                        None => break,
                    };
                    let frame = build_frame(runner);
                    let diag = if diagnostics_enabled {
                        Some(build_diagnostics(runner))
                    } else {
                        None
                    };
                    (frame, diag)
                    // Lock is dropped here
                };

                // Flow control: if buffer is full, drop intermediate frames
                if pending_frames.len() >= MAX_PENDING {
                    // Drop all but the last frame
                    pending_frames.clear();
                    tracing::debug!("Flow control: dropped intermediate frames for simulation {}", sim_id);
                }

                // Try to send immediately if buffer is getting full
                if pending_frames.is_empty() {
                    if let Err(e) = sender.send(Message::Binary(frame_data)).await {
                        tracing::error!("Failed to send frame: {}", e);
                        break;
                    }

                    // Send diagnostics if enabled
                    if let Some(diag) = diagnostics_data {
                        if let Err(e) = sender.send(Message::Binary(diag)).await {
                            tracing::error!("Failed to send diagnostics: {}", e);
                            break;
                        }
                    }
                } else {
                    pending_frames.push(frame_data);
                }

                // Try to flush pending frames
                while !pending_frames.is_empty() {
                    let frame = pending_frames.remove(0);
                    if let Err(e) = sender.send(Message::Binary(frame)).await {
                        tracing::error!("Failed to send pending frame: {}", e);
                        break;
                    }
                }
            }

            // Receive commands from client
            msg = receiver.next() => {
                match msg {
                    Some(Ok(Message::Binary(data))) => {
                        if let Err(e) = handle_client_command(&state, &sim_id, &data, &mut sender, &mut diagnostics_enabled).await {
                            tracing::error!("Error handling command: {}", e);
                        }
                    }
                    Some(Ok(Message::Close(_))) | None => {
                        tracing::info!("WebSocket closed for simulation {}", sim_id);
                        break;
                    }
                    Some(Err(e)) => {
                        tracing::error!("WebSocket error: {}", e);
                        break;
                    }
                    _ => {}
                }
            }
        }
    }

    // Cleanup: pause simulation when client disconnects
    let sims = state.simulations.lock().unwrap();
    if let Some(runner) = sims.get(&sim_id) {
        runner.pause();
    }
}

// ---------------------------------------------------------------------------
// Binary Protocol Builders
// ---------------------------------------------------------------------------

/// Build SimInfo message (tag 0x01)
/// Format: tag(u8) + particle_count(u32) + surface_count(u32) + domain_min(f32x3) + domain_max(f32x3) + fluid_type(u8) + subsample_rate(u8)
fn build_sim_info(runner: &crate::runner::SimulationRunner) -> Vec<u8> {
    let mut buf = Vec::with_capacity(35);

    // Tag
    buf.push(TAG_SIM_INFO);

    // Particle counts
    let particle_count = runner.particle_count() as u32;
    let subsample_count = runner.subsample_count() as u32;
    buf.extend_from_slice(&particle_count.to_le_bytes());
    buf.extend_from_slice(&subsample_count.to_le_bytes());

    // Domain bounds
    for &v in &runner.domain_min() {
        buf.extend_from_slice(&v.to_le_bytes());
    }
    for &v in &runner.domain_max() {
        buf.extend_from_slice(&v.to_le_bytes());
    }

    // Fluid type (0 = Water, 1 = Air, 2 = Mixed)
    buf.push(runner.fluid_type());

    // Subsample rate (~5% = 20:1)
    let subsample_rate = (particle_count / subsample_count).max(1) as u8;
    buf.push(subsample_rate);

    buf
}

/// Build Frame message (tag 0x02) with particle subsampling
/// Format: tag(u8) + frame_number(u64) + particle_count(u32) + sim_time(f64) + [particles...]
/// Each particle: x(f32) + y(f32) + z(f32) + temperature(f32) + fluid_type(u8) + density_ratio(u16) + reserved(u8)
fn build_frame(runner: &crate::runner::SimulationRunner) -> Vec<u8> {
    let particles = runner.particles();
    let n = particles.len();
    let subsample_count = runner.subsample_count();

    // Calculate stride for deterministic subsampling
    let stride = if subsample_count > 0 {
        (n / subsample_count).max(1)
    } else {
        1
    };

    let actual_count = (n / stride).min(subsample_count);

    // Allocate buffer
    let mut buf = Vec::with_capacity(1 + 8 + 4 + 8 + actual_count * 20);

    // Tag
    buf.push(TAG_FRAME);

    // Frame number
    let frame_number = runner.timestep_count();
    buf.extend_from_slice(&frame_number.to_le_bytes());

    // Particle count (subsampled)
    buf.extend_from_slice(&(actual_count as u32).to_le_bytes());

    // Simulation time
    let sim_time = runner.sim_time();
    buf.extend_from_slice(&sim_time.to_le_bytes());

    // Serialize subsampled particles
    for i in (0..n).step_by(stride).take(actual_count) {
        // Position
        buf.extend_from_slice(&particles.x[i].to_le_bytes());
        buf.extend_from_slice(&particles.y[i].to_le_bytes());
        buf.extend_from_slice(&particles.z[i].to_le_bytes());

        // Temperature
        buf.extend_from_slice(&particles.temperature[i].to_le_bytes());

        // Fluid type
        let fluid_type_byte = match particles.fluid_type[i] {
            FluidType::Water => 0u8,
            FluidType::Air => 1u8,
        };
        buf.push(fluid_type_byte);

        // Density ratio (fixed-point: (density / rho0) * 1000 as u16)
        let rest_density = match particles.fluid_type[i] {
            FluidType::Water => kernel::eos::WATER_REST_DENSITY,
            FluidType::Air => kernel::eos::AIR_REST_DENSITY,
        };
        let density_ratio = ((particles.density[i] / rest_density) * 1000.0).clamp(0.0, 65535.0) as u16;
        buf.extend_from_slice(&density_ratio.to_le_bytes());

        // Reserved byte
        buf.push(0);
    }

    buf
}

/// Build SimStatus message (tag 0x04)
/// Format: tag(u8) + status(u8) + message_length(u16) + message(utf8)
fn build_sim_status(status: SimStatus, message: &str) -> Vec<u8> {
    let mut buf = Vec::new();

    // Tag
    buf.push(TAG_SIM_STATUS);

    // Status byte (matches frontend: 0=Running, 1=Paused, 2=Finished, 3=Error)
    let status_byte = match status {
        SimStatus::Running | SimStatus::Created => 0u8,
        SimStatus::Paused => 1u8,
        SimStatus::Stopped => 2u8,
    };
    buf.push(status_byte);

    // Message length (u16) and content
    let msg_bytes = message.as_bytes();
    buf.extend_from_slice(&(msg_bytes.len() as u16).to_le_bytes());
    buf.extend_from_slice(msg_bytes);

    buf
}

/// Build Diagnostics message (tag 0x03)
/// Format: tag(u8) + frame_number(u64) + frame_time_ms(f32) + max_density_var(f32) +
///         energy_conservation(f32) + mass_conservation(f32) + dt(f32) + particle_count(u32)
fn build_diagnostics(runner: &crate::runner::SimulationRunner) -> Vec<u8> {
    let mut buf = Vec::with_capacity(33);

    // Tag
    buf.push(TAG_DIAGNOSTICS);

    // Frame number
    let frame_number = runner.timestep_count();
    buf.extend_from_slice(&frame_number.to_le_bytes());

    // Frame time (measured as 0 for now; could be enhanced with actual timing)
    let frame_time_ms = 0.0_f32;
    buf.extend_from_slice(&frame_time_ms.to_le_bytes());

    // Get error metrics
    let metrics = runner.error_metrics();

    // Max density variation
    buf.extend_from_slice(&metrics.max_density_variation.to_le_bytes());

    // Energy conservation error
    buf.extend_from_slice(&metrics.energy_conservation.to_le_bytes());

    // Mass conservation error
    buf.extend_from_slice(&metrics.mass_conservation.to_le_bytes());

    // Timestep dt
    let dt = runner.dt();
    buf.extend_from_slice(&dt.to_le_bytes());

    // Particle count
    let particle_count = runner.particle_count() as u32;
    buf.extend_from_slice(&particle_count.to_le_bytes());

    buf
}

// ---------------------------------------------------------------------------
// Client Command Handling
// ---------------------------------------------------------------------------

/// Handle incoming command from client
async fn handle_client_command(
    state: &Arc<AppState>,
    sim_id: &str,
    data: &[u8],
    sender: &mut futures_util::stream::SplitSink<WebSocket, Message>,
    diagnostics_enabled: &mut bool,
) -> Result<(), String> {

    if data.len() < 2 {
        return Err("Command too short".to_string());
    }

    let tag = data[0];
    if tag != 0x80 {
        return Err(format!("Unknown command tag: 0x{:02x}", tag));
    }

    let command = data[1];

    // Handle diagnostics commands locally (no status message needed)
    match command {
        CMD_ENABLE_DIAGNOSTICS => {
            *diagnostics_enabled = true;
            tracing::info!("Diagnostics enabled for simulation {}", sim_id);
            return Ok(());
        }
        CMD_DISABLE_DIAGNOSTICS => {
            *diagnostics_enabled = false;
            tracing::info!("Diagnostics disabled for simulation {}", sim_id);
            return Ok(());
        }
        _ => {}
    }

    // Handle simulation control commands and build response (scoped lock)
    let status_msg = {
        let sims = state.simulations.lock().unwrap();
        let runner = sims.get(sim_id)
            .ok_or_else(|| "Simulation not found".to_string())?;

        match command {
            CMD_PAUSE => {
                runner.pause();
                build_sim_status(SimStatus::Paused, "Simulation paused")
            }
            CMD_RESUME => {
                runner.resume();
                build_sim_status(SimStatus::Running, "Simulation resumed")
            }
            _ => {
                return Err(format!("Unknown command: 0x{:02x}", command));
            }
        }
        // Lock is dropped here
    };

    // Send response
    sender.send(Message::Binary(status_msg)).await
        .map_err(|e| format!("Failed to send status: {}", e))?;

    Ok(())
}
