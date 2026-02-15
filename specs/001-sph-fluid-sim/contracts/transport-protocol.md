# Transport Protocol: Simulation ↔ Web Viewer

**Version**: 1.0.0
**Transport**: WebSocket (binary frames) over localhost
**Endpoint**: `ws://localhost:{PORT}/ws/simulation`

**Particle Subsampling**: The server streams a configurable subsample
of the total particle set (default: ~5%) in each Frame message. The
`particle_count` in Frame headers reflects the subsampled count, not
the total simulation particle count. The `SimInfo` message contains the
total particle count for the full simulation.

## Connection Lifecycle

1. Client connects to WebSocket endpoint.
2. Server sends `SimInfo` message with simulation metadata.
3. Server streams `Frame` messages at simulation rate.
4. Client may send `Command` messages to control simulation.
5. Connection closed on simulation end or client disconnect.

## Message Format

All messages are binary WebSocket frames. Each message starts with
a 1-byte message type tag, followed by type-specific payload.

### Server → Client Messages

#### SimInfo (tag: 0x01)

Sent once on connection, before any frames.

| Offset | Size | Type | Field |
|--------|------|------|-------|
| 0 | 1 | u8 | tag = 0x01 |
| 1 | 4 | u32 | particle_count |
| 5 | 4 | u32 | surface_count (STL triangle groups) |
| 9 | 12 | f32×3 | domain_min (x, y, z) |
| 21 | 12 | f32×3 | domain_max (x, y, z) |
| 33 | 1 | u8 | fluid_type (0=Water, 1=Air, 2=Mixed) |
| 34 | 1 | u8 | subsample_rate (percentage of particles included in frames, e.g. 5 = 5%) |

Total: 35 bytes

#### Frame (tag: 0x02)

Sent each simulation frame. Contains particle state for rendering.

| Offset | Size | Type | Field |
|--------|------|------|-------|
| 0 | 1 | u8 | tag = 0x02 |
| 1 | 8 | u64 | frame_number |
| 9 | 4 | u32 | particle_count |
| 13 | 8 | f64 | sim_time (seconds) |
| 21 | N×20 | per-particle | particle data (see below) |

**Note**: `particle_count` in the Frame header is the SUBSAMPLED count
(e.g., ~5% of total simulation particles).

Per-particle data (20 bytes each):

| Offset | Size | Type | Field |
|--------|------|------|-------|
| 0 | 4 | f32 | x |
| 4 | 4 | f32 | y |
| 8 | 4 | f32 | z |
| 12 | 4 | f32 | temperature |
| 16 | 1 | u8 | fluid_type (0=Water, 1=Air) |
| 17 | 2 | u16 | density_ratio (fixed-point, density/rho0 x 1000) |
| 19 | 1 | u8 | _reserved (padding for alignment) |

Total per frame: 21 + (particle_count × 20) bytes
At 50K particles subsampled to 5%: ~50 KB per frame
At 500K particles subsampled to 5%: ~500 KB per frame

#### Diagnostics (tag: 0x03)

Sent alongside Frame messages when debug mode is enabled.

| Offset | Size | Type | Field |
|--------|------|------|-------|
| 0 | 1 | u8 | tag = 0x03 |
| 1 | 8 | u64 | frame_number |
| 9 | 4 | f32 | frame_time_ms |
| 13 | 4 | f32 | max_density_variation |
| 17 | 4 | f32 | energy_conservation_error |
| 21 | 4 | f32 | mass_conservation_error |
| 25 | 4 | f32 | dt (current timestep) |
| 29 | 4 | u32 | particle_count |

Total: 33 bytes

#### SimStatus (tag: 0x04)

Sent on state changes (started, paused, finished, error).

| Offset | Size | Type | Field |
|--------|------|------|-------|
| 0 | 1 | u8 | tag = 0x04 |
| 1 | 1 | u8 | status (0=Running, 1=Paused, 2=Finished, 3=Error) |
| 2 | 4 | u32 | message_length |
| 6 | N | utf8 | status message |

### Client → Server Messages

#### Command (tag: 0x80)

| Offset | Size | Type | Field |
|--------|------|------|-------|
| 0 | 1 | u8 | tag = 0x80 |
| 1 | 1 | u8 | command (see below) |

Commands:
- 0x01: Pause
- 0x02: Resume
- 0x03: Checkpoint (save state to disk)
- 0x04: Enable diagnostics
- 0x05: Disable diagnostics

## Flow Control

- Server sends frames as they are produced by the simulation.
- If the client's WebSocket send buffer exceeds a configurable
  threshold (default: 5 frames), the server drops intermediate
  frames and sends only the latest. This prevents back-pressure
  from slowing the simulation.
- Frame numbers are monotonically increasing, allowing the client
  to detect dropped frames.

## Error Handling

- If the WebSocket connection drops, the simulation continues
  running. A new client can connect and receive the current state.
- Malformed client messages are logged and ignored.
- Server sends SimStatus with Error status if simulation becomes
  unstable.
