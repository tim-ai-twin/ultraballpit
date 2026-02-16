// T038: Simulation control buttons (Start, Pause, Resume)

type StartCallback = (configName: string) => void;
type PauseCallback = () => void;
type ResumeCallback = () => void;
type DiagnosticsToggleCallback = (enabled: boolean) => void;
type ForceOverlayToggleCallback = (enabled: boolean) => void;

/**
 * Simulation control UI component
 *
 * Provides buttons for starting, pausing, and resuming simulations
 * with appropriate state management.
 */
export class SimControls {
  private container: HTMLElement;
  private startCallbacks: StartCallback[] = [];
  private pauseCallbacks: PauseCallback[] = [];
  private resumeCallbacks: ResumeCallback[] = [];
  private diagnosticsToggleCallbacks: DiagnosticsToggleCallback[] = [];
  private forceOverlayToggleCallbacks: ForceOverlayToggleCallback[] = [];

  private configName: string = '';
  private status: string = 'idle';
  private diagnosticsEnabled: boolean = false;
  private forceOverlayEnabled: boolean = false;

  constructor(container: HTMLElement) {
    this.container = container;
    this.render();
  }

  /**
   * Register callback for start button
   */
  onStart(cb: StartCallback): void {
    this.startCallbacks.push(cb);
  }

  /**
   * Register callback for pause button
   */
  onPause(cb: PauseCallback): void {
    this.pauseCallbacks.push(cb);
  }

  /**
   * Register callback for resume button
   */
  onResume(cb: ResumeCallback): void {
    this.resumeCallbacks.push(cb);
  }

  /**
   * Register callback for diagnostics toggle
   */
  onDiagnosticsToggle(cb: DiagnosticsToggleCallback): void {
    this.diagnosticsToggleCallbacks.push(cb);
  }

  /**
   * Register callback for force overlay toggle
   */
  onForceOverlayToggle(cb: ForceOverlayToggleCallback): void {
    this.forceOverlayToggleCallbacks.push(cb);
  }

  /**
   * Set current simulation status
   */
  setStatus(status: string): void {
    this.status = status;
    this.render();
  }

  /**
   * Set selected config name
   */
  setConfigName(name: string): void {
    this.configName = name;
    this.render();
  }

  /**
   * Render control buttons
   */
  private render(): void {
    const canStart = this.configName !== '' && this.status === 'idle';
    const canPause = this.status === 'running';
    const canResume = this.status === 'paused';
    const canToggleDiagnostics = this.status === 'running' || this.status === 'paused';

    this.container.innerHTML = `
      <div style="padding: 12px; background: #2a2a2a; border-radius: 4px; margin-top: 12px;">
        <h3 style="margin: 0 0 12px 0; font-size: 14px; color: #fff;">
          Controls
        </h3>

        <div style="margin-bottom: 8px;">
          <div style="font-size: 11px; color: #888; margin-bottom: 4px;">
            Status: <span style="color: ${this.getStatusColor()}">${this.status}</span>
          </div>
          ${this.configName ? `
            <div style="font-size: 11px; color: #888;">
              Config: <span style="color: #fff">${this.configName}</span>
            </div>
          ` : ''}
        </div>

        <button
          id="btn-start"
          ${canStart ? '' : 'disabled'}
          style="
            display: block;
            width: 100%;
            padding: 10px;
            margin-bottom: 6px;
            background: ${canStart ? '#4CAF50' : '#333'};
            color: ${canStart ? '#fff' : '#666'};
            border: none;
            border-radius: 3px;
            cursor: ${canStart ? 'pointer' : 'not-allowed'};
            font-size: 13px;
            font-weight: 500;
            transition: background 0.2s;
          "
          ${canStart ? `
            onmouseover="this.style.background='#45a049'"
            onmouseout="this.style.background='#4CAF50'"
          ` : ''}
        >
          Start Simulation
        </button>

        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 6px;">
          <button
            id="btn-pause"
            ${canPause ? '' : 'disabled'}
            style="
              padding: 8px;
              background: ${canPause ? '#FF9800' : '#333'};
              color: ${canPause ? '#fff' : '#666'};
              border: none;
              border-radius: 3px;
              cursor: ${canPause ? 'pointer' : 'not-allowed'};
              font-size: 12px;
              transition: background 0.2s;
            "
            ${canPause ? `
              onmouseover="this.style.background='#e68900'"
              onmouseout="this.style.background='#FF9800'"
            ` : ''}
          >
            Pause
          </button>

          <button
            id="btn-resume"
            ${canResume ? '' : 'disabled'}
            style="
              padding: 8px;
              background: ${canResume ? '#2196F3' : '#333'};
              color: ${canResume ? '#fff' : '#666'};
              border: none;
              border-radius: 3px;
              cursor: ${canResume ? 'pointer' : 'not-allowed'};
              font-size: 12px;
              transition: background 0.2s;
            "
            ${canResume ? `
              onmouseover="this.style.background='#1976D2'"
              onmouseout="this.style.background='#2196F3'"
            ` : ''}
          >
            Resume
          </button>
        </div>

        <button
          id="btn-diagnostics"
          ${canToggleDiagnostics ? '' : 'disabled'}
          style="
            display: block;
            width: 100%;
            padding: 8px;
            margin-top: 6px;
            background: ${canToggleDiagnostics ? (this.diagnosticsEnabled ? '#9C27B0' : '#555') : '#333'};
            color: ${canToggleDiagnostics ? '#fff' : '#666'};
            border: none;
            border-radius: 3px;
            cursor: ${canToggleDiagnostics ? 'pointer' : 'not-allowed'};
            font-size: 12px;
            transition: background 0.2s;
          "
          ${canToggleDiagnostics ? `
            onmouseover="this.style.background='${this.diagnosticsEnabled ? '#7B1FA2' : '#666'}'"
            onmouseout="this.style.background='${this.diagnosticsEnabled ? '#9C27B0' : '#555'}'"
          ` : ''}
        >
          ${this.diagnosticsEnabled ? '✓ ' : ''}Diagnostics
        </button>

        <button
          id="btn-force-overlay"
          ${canToggleDiagnostics ? '' : 'disabled'}
          style="
            display: block;
            width: 100%;
            padding: 8px;
            margin-top: 6px;
            background: ${canToggleDiagnostics ? (this.forceOverlayEnabled ? '#00BCD4' : '#555') : '#333'};
            color: ${canToggleDiagnostics ? '#fff' : '#666'};
            border: none;
            border-radius: 3px;
            cursor: ${canToggleDiagnostics ? 'pointer' : 'not-allowed'};
            font-size: 12px;
            transition: background 0.2s;
          "
          ${canToggleDiagnostics ? `
            onmouseover="this.style.background='${this.forceOverlayEnabled ? '#0097A7' : '#666'}'"
            onmouseout="this.style.background='${this.forceOverlayEnabled ? '#00BCD4' : '#555'}'"
          ` : ''}
        >
          ${this.forceOverlayEnabled ? '✓ ' : ''}Force Overlay
        </button>
      </div>
    `;

    // Attach event listeners
    const startBtn = document.getElementById('btn-start');
    const pauseBtn = document.getElementById('btn-pause');
    const resumeBtn = document.getElementById('btn-resume');
    const diagnosticsBtn = document.getElementById('btn-diagnostics');
    const forceOverlayBtn = document.getElementById('btn-force-overlay');

    if (startBtn && canStart) {
      startBtn.addEventListener('click', () => {
        this.startCallbacks.forEach((cb) => cb(this.configName));
      });
    }

    if (pauseBtn && canPause) {
      pauseBtn.addEventListener('click', () => {
        this.pauseCallbacks.forEach((cb) => cb());
      });
    }

    if (resumeBtn && canResume) {
      resumeBtn.addEventListener('click', () => {
        this.resumeCallbacks.forEach((cb) => cb());
      });
    }

    if (diagnosticsBtn && canToggleDiagnostics) {
      diagnosticsBtn.addEventListener('click', () => {
        this.diagnosticsEnabled = !this.diagnosticsEnabled;
        this.diagnosticsToggleCallbacks.forEach((cb) => cb(this.diagnosticsEnabled));
        this.render();
      });
    }

    if (forceOverlayBtn && canToggleDiagnostics) {
      forceOverlayBtn.addEventListener('click', () => {
        this.forceOverlayEnabled = !this.forceOverlayEnabled;
        this.forceOverlayToggleCallbacks.forEach((cb) => cb(this.forceOverlayEnabled));
        this.render();
      });
    }
  }

  /**
   * Get status color for display
   */
  private getStatusColor(): string {
    switch (this.status) {
      case 'running':
        return '#4CAF50';
      case 'paused':
        return '#FF9800';
      case 'finished':
        return '#2196F3';
      case 'error':
        return '#f44336';
      default:
        return '#888';
    }
  }
}
