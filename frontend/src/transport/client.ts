// T034: WebSocket client for binary simulation data transport

import {
  MSG_SIM_INFO,
  MSG_FRAME,
  MSG_DIAGNOSTICS,
  MSG_SIM_STATUS,
  parseSimInfo,
  parseFrame,
  parseDiagnostics,
  parseSimStatus,
  buildCommand,
  type SimInfo,
  type Frame,
  type Diagnostics,
  type SimStatus,
} from '../types/protocol.js';

type SimInfoCallback = (info: SimInfo) => void;
type FrameCallback = (frame: Frame) => void;
type DiagnosticsCallback = (diagnostics: Diagnostics) => void;
type StatusCallback = (status: SimStatus) => void;

/**
 * WebSocket client for SPH simulation data streaming
 *
 * Handles:
 * - Binary message parsing (SimInfo, Frame, Diagnostics, SimStatus)
 * - Command sending (pause, resume, etc.)
 * - Automatic reconnection on disconnect
 * - Subsampled particle data reception
 */
export class SimulationClient {
  private ws: WebSocket | null = null;
  private wsUrl: string = '';
  private reconnectAttempts = 0;
  private maxReconnectAttempts = 5;
  private reconnectDelay = 1000;

  private simInfoCallbacks: SimInfoCallback[] = [];
  private frameCallbacks: FrameCallback[] = [];
  private diagnosticsCallbacks: DiagnosticsCallback[] = [];
  private statusCallbacks: StatusCallback[] = [];

  /**
   * Connect to simulation WebSocket
   */
  connect(wsUrl: string): void {
    this.wsUrl = wsUrl;
    this.reconnectAttempts = 0;
    this._connect();
  }

  private _connect(): void {
    if (this.ws?.readyState === WebSocket.OPEN) {
      console.warn('WebSocket already connected');
      return;
    }

    console.log('Connecting to simulation:', this.wsUrl);
    this.ws = new WebSocket(this.wsUrl);
    this.ws.binaryType = 'arraybuffer';

    this.ws.onopen = () => {
      console.log('WebSocket connected');
      this.reconnectAttempts = 0;
    };

    this.ws.onmessage = (event) => {
      this.handleMessage(event.data);
    };

    this.ws.onerror = (error) => {
      console.error('WebSocket error:', error);
    };

    this.ws.onclose = () => {
      console.log('WebSocket closed');
      this.attemptReconnect();
    };
  }

  /**
   * Disconnect from simulation
   */
  disconnect(): void {
    if (this.ws) {
      this.ws.close();
      this.ws = null;
    }
    this.reconnectAttempts = this.maxReconnectAttempts; // Prevent reconnection
  }

  /**
   * Attempt to reconnect with exponential backoff
   */
  private attemptReconnect(): void {
    if (this.reconnectAttempts >= this.maxReconnectAttempts) {
      console.error('Max reconnection attempts reached');
      return;
    }

    this.reconnectAttempts++;
    const delay = this.reconnectDelay * Math.pow(2, this.reconnectAttempts - 1);
    console.log(`Reconnecting in ${delay}ms (attempt ${this.reconnectAttempts})`);

    setTimeout(() => {
      this._connect();
    }, delay);
  }

  /**
   * Handle incoming binary message
   */
  private handleMessage(data: ArrayBuffer): void {
    if (data.byteLength < 1) {
      console.warn('Received empty message');
      return;
    }

    // Read message tag (first byte)
    const view = new DataView(data);
    const tag = view.getUint8(0);

    // Skip the tag byte - parsers expect data AFTER the tag
    const payload = data.slice(1);

    try {
      switch (tag) {
        case MSG_SIM_INFO: {
          const simInfo = parseSimInfo(payload);
          this.simInfoCallbacks.forEach((cb) => cb(simInfo));
          break;
        }

        case MSG_FRAME: {
          const frame = parseFrame(payload);
          this.frameCallbacks.forEach((cb) => cb(frame));
          break;
        }

        case MSG_DIAGNOSTICS: {
          const diagnostics = parseDiagnostics(payload);
          this.diagnosticsCallbacks.forEach((cb) => cb(diagnostics));
          break;
        }

        case MSG_SIM_STATUS: {
          const status = parseSimStatus(payload);
          this.statusCallbacks.forEach((cb) => cb(status));
          break;
        }

        default:
          console.warn('Unknown message tag:', tag);
      }
    } catch (error) {
      console.error('Error parsing message:', error);
    }
  }

  /**
   * Register callback for SimInfo messages
   */
  onSimInfo(cb: SimInfoCallback): void {
    this.simInfoCallbacks.push(cb);
  }

  /**
   * Register callback for Frame messages
   */
  onFrame(cb: FrameCallback): void {
    this.frameCallbacks.push(cb);
  }

  /**
   * Register callback for Diagnostics messages
   */
  onDiagnostics(cb: DiagnosticsCallback): void {
    this.diagnosticsCallbacks.push(cb);
  }

  /**
   * Register callback for SimStatus messages
   */
  onStatus(cb: StatusCallback): void {
    this.statusCallbacks.push(cb);
  }

  /**
   * Send command to simulation (pause, resume, etc.)
   */
  sendCommand(cmd: number): void {
    if (!this.ws || this.ws.readyState !== WebSocket.OPEN) {
      console.error('Cannot send command: WebSocket not connected');
      return;
    }

    const message = buildCommand(cmd);
    this.ws.send(message);
  }
}
