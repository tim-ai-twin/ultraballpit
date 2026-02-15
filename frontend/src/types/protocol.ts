// Binary transport protocol types for SPH simulation
//
// This module defines the message format and provides parsers/builders
// for the binary WebSocket protocol used to communicate with the simulation.

// Message tags
export const MSG_SIM_INFO = 0x01;
export const MSG_FRAME = 0x02;
export const MSG_DIAGNOSTICS = 0x03;
export const MSG_SIM_STATUS = 0x04;
export const MSG_COMMAND = 0x80;

// Command codes
export const CMD_PAUSE = 0x01;
export const CMD_RESUME = 0x02;
export const CMD_CHECKPOINT = 0x03;
export const CMD_ENABLE_DIAGNOSTICS = 0x04;
export const CMD_DISABLE_DIAGNOSTICS = 0x05;

// Fluid types
export const FLUID_WATER = 0;
export const FLUID_AIR = 1;
export const FLUID_MIXED = 2;

// Simulation status
export const STATUS_RUNNING = 0;
export const STATUS_PAUSED = 1;
export const STATUS_FINISHED = 2;
export const STATUS_ERROR = 3;

// Parsed message types
export interface SimInfo {
  particleCount: number;    // u32
  surfaceCount: number;     // u32
  domainMin: [number, number, number];  // f32x3
  domainMax: [number, number, number];  // f32x3
  fluidType: number;        // u8 (0=Water, 1=Air, 2=Mixed)
  subsampleRate: number;    // u8 (percentage, e.g. 5 = 5%)
}

export interface ParticleData {
  x: Float32Array;          // positions
  y: Float32Array;
  z: Float32Array;
  temperature: Float32Array;
  fluidType: Uint8Array;    // 0=Water, 1=Air
  densityRatio: Uint16Array; // fixed-point, density/rho0 * 1000
}

export interface Frame {
  frameNumber: bigint;      // u64
  particleCount: number;    // u32
  simTime: number;          // f64
  particles: ParticleData;
}

export interface Diagnostics {
  frameNumber: bigint;
  frameTimeMs: number;
  maxDensityVariation: number;
  energyConservationError: number;
  massConservationError: number;
  dt: number;
  particleCount: number;
}

export interface SimStatus {
  status: number;           // 0-3
  message: string;
}

/**
 * Parse SimInfo message (tag 0x01)
 *
 * Format:
 * - u32 particle_count
 * - u32 surface_count
 * - f32x3 domain_min
 * - f32x3 domain_max
 * - u8 fluid_type
 * - u8 subsample_rate
 */
export function parseSimInfo(buffer: ArrayBuffer): SimInfo {
  const view = new DataView(buffer);
  let offset = 0;

  const particleCount = view.getUint32(offset, true);
  offset += 4;

  const surfaceCount = view.getUint32(offset, true);
  offset += 4;

  const domainMin: [number, number, number] = [
    view.getFloat32(offset, true),
    view.getFloat32(offset + 4, true),
    view.getFloat32(offset + 8, true),
  ];
  offset += 12;

  const domainMax: [number, number, number] = [
    view.getFloat32(offset, true),
    view.getFloat32(offset + 4, true),
    view.getFloat32(offset + 8, true),
  ];
  offset += 12;

  const fluidType = view.getUint8(offset);
  offset += 1;

  const subsampleRate = view.getUint8(offset);
  offset += 1;

  return {
    particleCount,
    surfaceCount,
    domainMin,
    domainMax,
    fluidType,
    subsampleRate,
  };
}

/**
 * Parse Frame message (tag 0x02)
 *
 * Format:
 * - u64 frame_number
 * - u32 particle_count
 * - f64 sim_time
 * - [particle_data] (20 bytes each)
 *
 * Per-particle data layout (20 bytes):
 * - offset 0: f32 x
 * - offset 4: f32 y
 * - offset 8: f32 z
 * - offset 12: f32 temperature
 * - offset 16: u8 fluid_type
 * - offset 17: u16 density_ratio (little-endian)
 * - offset 19: u8 reserved
 */
export function parseFrame(buffer: ArrayBuffer): Frame {
  const view = new DataView(buffer);
  let offset = 0;

  const frameNumber = view.getBigUint64(offset, true);
  offset += 8;

  const particleCount = view.getUint32(offset, true);
  offset += 4;

  const simTime = view.getFloat64(offset, true);
  offset += 8;

  // Allocate arrays for particle data
  const x = new Float32Array(particleCount);
  const y = new Float32Array(particleCount);
  const z = new Float32Array(particleCount);
  const temperature = new Float32Array(particleCount);
  const fluidType = new Uint8Array(particleCount);
  const densityRatio = new Uint16Array(particleCount);

  // Parse particle data (20 bytes per particle)
  for (let i = 0; i < particleCount; i++) {
    x[i] = view.getFloat32(offset, true);
    offset += 4;

    y[i] = view.getFloat32(offset, true);
    offset += 4;

    z[i] = view.getFloat32(offset, true);
    offset += 4;

    temperature[i] = view.getFloat32(offset, true);
    offset += 4;

    fluidType[i] = view.getUint8(offset);
    offset += 1;

    densityRatio[i] = view.getUint16(offset, true);
    offset += 2;

    // Skip reserved byte
    offset += 1;
  }

  return {
    frameNumber,
    particleCount,
    simTime,
    particles: {
      x,
      y,
      z,
      temperature,
      fluidType,
      densityRatio,
    },
  };
}

/**
 * Parse Diagnostics message (tag 0x03)
 *
 * Format:
 * - u64 frame_number
 * - f32 frame_time_ms
 * - f32 max_density_variation
 * - f32 energy_conservation_error
 * - f32 mass_conservation_error
 * - f32 dt
 * - u32 particle_count
 */
export function parseDiagnostics(buffer: ArrayBuffer): Diagnostics {
  const view = new DataView(buffer);
  let offset = 0;

  const frameNumber = view.getBigUint64(offset, true);
  offset += 8;

  const frameTimeMs = view.getFloat32(offset, true);
  offset += 4;

  const maxDensityVariation = view.getFloat32(offset, true);
  offset += 4;

  const energyConservationError = view.getFloat32(offset, true);
  offset += 4;

  const massConservationError = view.getFloat32(offset, true);
  offset += 4;

  const dt = view.getFloat32(offset, true);
  offset += 4;

  const particleCount = view.getUint32(offset, true);
  offset += 4;

  return {
    frameNumber,
    frameTimeMs,
    maxDensityVariation,
    energyConservationError,
    massConservationError,
    dt,
    particleCount,
  };
}

/**
 * Parse SimStatus message (tag 0x04)
 *
 * Format:
 * - u8 status (0=Running, 1=Paused, 2=Finished, 3=Error)
 * - u16 message_len
 * - [u8] message (UTF-8)
 */
export function parseSimStatus(buffer: ArrayBuffer): SimStatus {
  const view = new DataView(buffer);
  let offset = 0;

  const status = view.getUint8(offset);
  offset += 1;

  const messageLen = view.getUint16(offset, true);
  offset += 2;

  const messageBytes = new Uint8Array(buffer, offset, messageLen);
  const message = new TextDecoder().decode(messageBytes);

  return {
    status,
    message,
  };
}

/**
 * Build a Command message (tag 0x80)
 *
 * Format:
 * - u8 tag (0x80)
 * - u8 command_code
 *
 * @param command Command code (CMD_PAUSE, CMD_RESUME, etc.)
 * @returns ArrayBuffer containing the binary message
 */
export function buildCommand(command: number): ArrayBuffer {
  const buffer = new ArrayBuffer(2);
  const view = new DataView(buffer);

  view.setUint8(0, MSG_COMMAND);
  view.setUint8(1, command);

  return buffer;
}
