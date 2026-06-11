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
use std::time::Duration;
use tokio::time::interval;

use crate::runner::SimulationRunner;
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

/// Wall-clock budget for each stepping batch (per blocking-thread call).
/// Larger batches amortize per-batch overhead (readbacks, health checks,
/// force recording); the frame builder grabs the kernel lock in the ~1ms
/// gaps between batches.
const STEP_BATCH_BUDGET: Duration = Duration::from_millis(24);

/// Frame streaming interval (~30 FPS; all particles are sent each frame)
const FRAME_INTERVAL: Duration = Duration::from_millis(33);

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
    let (mut sender, mut receiver) = socket.split();

    // Clone the runner out of the state map: all runner state is shared via
    // Arcs, so this clone observes (and controls) the same simulation without
    // holding the AppState lock.
    let runner = {
        let sims = state.simulations.lock().unwrap();
        match sims.get(&sim_id) {
            Some(r) => r.clone(),
            None => {
                tracing::error!("Simulation {} not found", sim_id);
                return;
            }
        }
    };

    // Per-connection diagnostics state (on by default; the HUD decides display)
    let mut diagnostics_enabled = true;

    // Send initial SimInfo
    if let Err(e) = sender.send(Message::Binary(build_sim_info(&runner))).await {
        tracing::error!("Failed to send SimInfo: {}", e);
        return;
    }

    // Start the simulation and its dedicated stepping task. Stepping runs in
    // spawn_blocking batches so the async runtime is never starved, and the
    // kernel lock is released between batches for frame snapshots.
    runner.start();
    let stepper = tokio::spawn({
        let runner = runner.clone();
        async move {
            loop {
                let r = runner.clone();
                let status = tokio::task::spawn_blocking(move || {
                    r.step_batch(STEP_BATCH_BUDGET);
                    r.status()
                })
                .await
                .unwrap_or(SimStatus::Stopped);

                match status {
                    SimStatus::Running | SimStatus::Created => {
                        // Brief yield so frame builds can grab the kernel lock
                        tokio::time::sleep(Duration::from_millis(1)).await;
                    }
                    SimStatus::Paused => {
                        tokio::time::sleep(Duration::from_millis(50)).await;
                    }
                    SimStatus::Stopped => break,
                }
            }
        }
    });

    let mut frame_timer = interval(FRAME_INTERVAL);
    frame_timer.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
    let mut last_status = runner.status();

    loop {
        tokio::select! {
            // Frame generation and sending
            _ = frame_timer.tick() => {
                let frame_data = build_frame(&runner);
                if sender.send(Message::Binary(frame_data)).await.is_err() {
                    break;
                }

                if diagnostics_enabled {
                    let diag = build_diagnostics(&runner);
                    if sender.send(Message::Binary(diag)).await.is_err() {
                        break;
                    }
                }

                // Push status transitions (e.g. auto-pause on instability,
                // auto-finish at max_time) to the client.
                let status = runner.status();
                if status != last_status {
                    last_status = status;
                    let msg = match status {
                        SimStatus::Running | SimStatus::Created => build_sim_status(status, "Simulation running"),
                        SimStatus::Paused => build_sim_status(status, "Simulation paused"),
                        SimStatus::Stopped => build_sim_status(status, "Simulation finished"),
                    };
                    if sender.send(Message::Binary(msg)).await.is_err() {
                        break;
                    }
                }
            }

            // Receive commands from client
            msg = receiver.next() => {
                match msg {
                    Some(Ok(Message::Binary(data))) => {
                        if let Err(e) = handle_client_command(&runner, &sim_id, &data, &mut sender, &mut diagnostics_enabled).await {
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

    // Cleanup: stop stepping and pause simulation when client disconnects
    stepper.abort();
    if runner.status() != SimStatus::Stopped {
        runner.pause();
    }
}

// ---------------------------------------------------------------------------
// Binary Protocol Builders
// ---------------------------------------------------------------------------

/// Build SimInfo message (tag 0x01)
/// Format: tag(u8) + particle_count(u32) + subsample_count(u32) +
///         domain_min(f32x3) + domain_max(f32x3) + fluid_type(u8) +
///         subsample_rate(u8) + particle_spacing(f32) + solver(u8)
fn build_sim_info(runner: &SimulationRunner) -> Vec<u8> {
    let mut buf = Vec::with_capacity(40);

    buf.push(TAG_SIM_INFO);

    let particle_count = runner.particle_count() as u32;
    buf.extend_from_slice(&particle_count.to_le_bytes());
    buf.extend_from_slice(&particle_count.to_le_bytes()); // all particles streamed

    for &v in &runner.domain_min() {
        buf.extend_from_slice(&v.to_le_bytes());
    }
    for &v in &runner.domain_max() {
        buf.extend_from_slice(&v.to_le_bytes());
    }

    buf.push(runner.fluid_type());
    buf.push(1); // subsample rate 1:1

    buf.extend_from_slice(&runner.particle_spacing().to_le_bytes());
    buf.push(runner.solver());

    buf
}

/// Build Frame message (tag 0x02) with all particles
/// Format: tag(u8) + frame_number(u64) + particle_count(u32) + sim_time(f64) +
///         dt(f32) + steps_per_sec(f32) + [particles...]
/// Each particle (32 bytes): x,y,z(f32) + vx,vy,vz(f32) + temperature(f32) +
///         fluid_type(u8) + density_ratio(u16) + reserved(u8)
fn build_frame(runner: &SimulationRunner) -> Vec<u8> {
    let particles = runner.particles();
    let n = particles.len();

    let mut buf = Vec::with_capacity(1 + 8 + 4 + 8 + 4 + 4 + n * 32);

    buf.push(TAG_FRAME);
    buf.extend_from_slice(&runner.timestep_count().to_le_bytes());
    buf.extend_from_slice(&(n as u32).to_le_bytes());
    buf.extend_from_slice(&runner.sim_time().to_le_bytes());
    buf.extend_from_slice(&runner.dt().to_le_bytes());
    buf.extend_from_slice(&runner.steps_per_sec().to_le_bytes());

    for i in 0..n {
        buf.extend_from_slice(&particles.x[i].to_le_bytes());
        buf.extend_from_slice(&particles.y[i].to_le_bytes());
        buf.extend_from_slice(&particles.z[i].to_le_bytes());
        buf.extend_from_slice(&particles.vx[i].to_le_bytes());
        buf.extend_from_slice(&particles.vy[i].to_le_bytes());
        buf.extend_from_slice(&particles.vz[i].to_le_bytes());
        buf.extend_from_slice(&particles.temperature[i].to_le_bytes());

        let fluid_type_byte = match particles.fluid_type[i] {
            FluidType::Water => 0u8,
            FluidType::Air => 1u8,
        };
        buf.push(fluid_type_byte);

        let rest_density = match particles.fluid_type[i] {
            FluidType::Water => kernel::eos::WATER_REST_DENSITY,
            FluidType::Air => kernel::eos::AIR_REST_DENSITY,
        };
        let density_ratio =
            ((particles.density[i] / rest_density) * 1000.0).clamp(0.0, 65535.0) as u16;
        buf.extend_from_slice(&density_ratio.to_le_bytes());

        buf.push(0); // reserved
    }

    buf
}

/// Build SimStatus message (tag 0x04)
/// Format: tag(u8) + status(u8) + message_length(u16) + message(utf8)
fn build_sim_status(status: SimStatus, message: &str) -> Vec<u8> {
    let mut buf = Vec::new();

    buf.push(TAG_SIM_STATUS);

    // Status byte (matches frontend: 0=Running, 1=Paused, 2=Finished, 3=Error)
    let status_byte = match status {
        SimStatus::Running | SimStatus::Created => 0u8,
        SimStatus::Paused => 1u8,
        SimStatus::Stopped => 2u8,
    };
    buf.push(status_byte);

    let msg_bytes = message.as_bytes();
    buf.extend_from_slice(&(msg_bytes.len() as u16).to_le_bytes());
    buf.extend_from_slice(msg_bytes);

    buf
}

/// Build Diagnostics message (tag 0x03)
/// Format: tag(u8) + frame_number(u64) + frame_time_ms(f32) + max_density_var(f32) +
///         energy_conservation(f32) + mass_conservation(f32) + dt(f32) + particle_count(u32)
fn build_diagnostics(runner: &SimulationRunner) -> Vec<u8> {
    let mut buf = Vec::with_capacity(33);

    buf.push(TAG_DIAGNOSTICS);
    buf.extend_from_slice(&runner.timestep_count().to_le_bytes());

    // Frame time: wall-clock ms per simulation step (derived from throughput)
    let sps = runner.steps_per_sec();
    let frame_time_ms = if sps > 0.0 { 1000.0 / sps } else { 0.0 };
    buf.extend_from_slice(&frame_time_ms.to_le_bytes());

    let metrics = runner.error_metrics();
    buf.extend_from_slice(&metrics.max_density_variation.to_le_bytes());
    buf.extend_from_slice(&metrics.energy_conservation.to_le_bytes());
    buf.extend_from_slice(&metrics.mass_conservation.to_le_bytes());
    buf.extend_from_slice(&runner.dt().to_le_bytes());
    buf.extend_from_slice(&(runner.particle_count() as u32).to_le_bytes());

    buf
}

// ---------------------------------------------------------------------------
// Client Command Handling
// ---------------------------------------------------------------------------

/// Handle incoming command from client
async fn handle_client_command(
    runner: &SimulationRunner,
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

    let status_msg = match command {
        CMD_ENABLE_DIAGNOSTICS => {
            *diagnostics_enabled = true;
            return Ok(());
        }
        CMD_DISABLE_DIAGNOSTICS => {
            *diagnostics_enabled = false;
            return Ok(());
        }
        CMD_PAUSE => {
            runner.pause();
            tracing::info!("Simulation {} paused by client", sim_id);
            build_sim_status(SimStatus::Paused, "Simulation paused")
        }
        CMD_RESUME => {
            runner.resume();
            tracing::info!("Simulation {} resumed by client", sim_id);
            build_sim_status(runner.status(), "Simulation resumed")
        }
        _ => {
            return Err(format!("Unknown command: 0x{:02x}", command));
        }
    };

    sender
        .send(Message::Binary(status_msg))
        .await
        .map_err(|e| format!("Failed to send status: {}", e))?;

    Ok(())
}
