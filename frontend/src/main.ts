// Main entry point: wires scene, transport, parameter panel and HUD.

import { createScene } from './viewer/scene.js';
import { SimulationClient } from './transport/client.js';
import { ParticleRenderer, type ColorMode } from './viewer/particles.js';
import { ContainerRenderer, type WallSpec } from './viewer/container.js';
import { ParamPanel } from './ui/params.js';
import {
  CMD_PAUSE,
  CMD_RESUME,
  STATUS_RUNNING,
  STATUS_PAUSED,
  STATUS_FINISHED,
} from './types/protocol.js';

const API_BASE = '';

// --- DOM handles -----------------------------------------------------------

const $ = <T extends HTMLElement>(id: string): T => {
  const el = document.getElementById(id);
  if (!el) throw new Error(`Missing element #${id}`);
  return el as T;
};

const canvas = $<HTMLCanvasElement>('viewer');
const configListEl = $('config-list');
const runBtn = $<HTMLButtonElement>('btn-run');
const runStatusEl = $('run-status');
const hudEl = $('hud');
const legendEl = $('legend');
const playbackEl = $('playback');
const emptyHintEl = $('empty-hint');
const pauseBtn = $<HTMLButtonElement>('btn-pause');
const restartBtn = $<HTMLButtonElement>('btn-restart');
const resetViewBtn = $<HTMLButtonElement>('btn-reset-view');
const colorSel = $<HTMLSelectElement>('sel-color');
const sizeRng = $<HTMLInputElement>('rng-size');
const wallsRng = $<HTMLInputElement>('rng-walls');

// --- Scene & renderers ------------------------------------------------------

const sceneComponents = createScene(canvas);
const { scene, camera, animate, fitToDomain, resetView } = sceneComponents;
const particleRenderer = new ParticleRenderer(scene);
const containerRenderer = new ContainerRenderer(scene);
const paramPanel = new ParamPanel($('params'));

// --- App state ---------------------------------------------------------------

let currentSimulationId: string | null = null;
let client: SimulationClient | null = null;
let paused = false;
let finished = false;

// Stats
let renderFrames = 0;
let dataFrames = 0;
let lastStatsTime = performance.now();
let lastDensityVar: number | null = null;

// --- Preset list -------------------------------------------------------------

interface ConfigInfo {
  id: string;
  name: string;
  fluid_type: string;
  particle_count_estimate: number;
  uses_periodic?: boolean;
}

async function loadPresets(): Promise<void> {
  try {
    const response = await fetch(`${API_BASE}/api/configs`);
    const data = (await response.json()) as { configs: ConfigInfo[] };
    configListEl.innerHTML = '';
    for (const cfg of data.configs) {
      const btn = document.createElement('button');
      btn.className = 'preset';
      const count =
        cfg.particle_count_estimate >= 1000
          ? `${(cfg.particle_count_estimate / 1000).toFixed(1)}K`
          : `${cfg.particle_count_estimate}`;
      let meta = `${cfg.fluid_type.toLowerCase()} · ~${count} particles`;
      if (cfg.uses_periodic) {
        btn.classList.add('unsupported');
        meta += ` · <span class="badge-warn">needs periodic BCs — not implemented</span>`;
      }
      btn.innerHTML = `${cfg.name}<span class="meta">${meta}</span>`;
      btn.addEventListener('click', () => selectPreset(cfg.id, btn));
      configListEl.appendChild(btn);
    }
  } catch (error) {
    configListEl.innerHTML = '<div class="hint">failed to load presets — is the server running?</div>';
    console.error('Failed to load configs:', error);
  }
}

async function selectPreset(id: string, btn: HTMLElement): Promise<void> {
  configListEl.querySelectorAll('.preset').forEach((el) => el.classList.remove('selected'));
  btn.classList.add('selected');

  try {
    const response = await fetch(`${API_BASE}/api/configs/${id}`);
    const json = await response.json();
    paramPanel.loadConfig(json);
    runBtn.disabled = false;
    setRunStatus(`loaded "${id}" — tune & run`);
  } catch (error) {
    setRunStatus('failed to load preset', true);
    console.error(error);
  }
}

function setRunStatus(text: string, isError = false): void {
  runStatusEl.textContent = text;
  runStatusEl.classList.toggle('error', isError);
}

// --- Simulation lifecycle ------------------------------------------------------

async function teardownSimulation(): Promise<void> {
  if (client) {
    client.disconnect();
    client = null;
  }
  if (currentSimulationId) {
    fetch(`${API_BASE}/api/simulations/${currentSimulationId}`, { method: 'DELETE' }).catch(
      () => {},
    );
    currentSimulationId = null;
  }
}

async function startSimulation(): Promise<void> {
  runBtn.disabled = true;
  setRunStatus('building simulation…');
  await teardownSimulation();

  const config = paramPanel.buildConfig();

  try {
    const response = await fetch(`${API_BASE}/api/simulations`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ config_json: config }),
    });

    if (!response.ok) {
      const text = await response.text();
      throw new Error(text || `HTTP ${response.status}`);
    }

    const data = await response.json();
    currentSimulationId = data.simulation_id;
    paused = false;
    finished = false;
    lastDensityVar = null;
    updatePauseButton();

    const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${wsProtocol}//${window.location.host}/ws/simulation/${data.simulation_id}`;

    client = new SimulationClient();

    client.onSimInfo((info) => {
      fitToDomain(info.domainMin, info.domainMax);
      containerRenderer.build(info.domainMin, info.domainMax, paramPanel.wallSpec() as WallSpec);
      containerRenderer.setWallOpacity(parseFloat(wallsRng.value));
      if (currentSimulationId) {
        containerRenderer.loadObstacle(currentSimulationId);
      }

      particleRenderer.setParticleRadius(info.particleSpacing * 0.5);
      particleRenderer.setSizeScale(parseFloat(sizeRng.value));
      particleRenderer.setColorMode(colorSel.value as ColorMode);
      particleRenderer.updateProjection(camera, canvas.clientHeight);

      hudEl.style.display = '';
      legendEl.style.display = '';
      playbackEl.style.display = 'flex';
      emptyHintEl.classList.add('hidden');

      $('st-particles').textContent = formatCount(info.particleCount);
      setHudStatus('running');
      setRunStatus(
        `running · ${formatCount(info.particleCount)} particles · ${info.solver === 1 ? 'PCISPH' : 'WCSPH'}`,
      );
      runBtn.disabled = false;
    });

    client.onFrame((frame) => {
      particleRenderer.update(frame.particles, frame.particleCount);
      dataFrames++;

      $('st-time').textContent = formatSimTime(frame.simTime);
      $('st-dt').textContent = `${(frame.dt * 1e6).toFixed(1)}µs`;
      $('st-sps').textContent = `${Math.round(frame.stepsPerSec)}/s`;
      const rtf = frame.stepsPerSec * frame.dt;
      $('st-rtf').textContent = rtf >= 0.095 ? `${rtf.toFixed(2)}` : `${rtf.toFixed(3)}`;
      $('st-particles').textContent = formatCount(frame.particleCount);
      updateLegend();
    });

    client.onDiagnostics((diag) => {
      lastDensityVar = diag.maxDensityVariation;
      $('st-density').textContent = `${(diag.maxDensityVariation * 100).toFixed(1)}%`;
    });

    client.onStatus((status) => {
      if (status.status === STATUS_RUNNING) {
        paused = false;
        finished = false;
        setHudStatus('running');
      } else if (status.status === STATUS_PAUSED) {
        paused = true;
        setHudStatus('paused');
      } else if (status.status === STATUS_FINISHED) {
        finished = true;
        setHudStatus('finished');
        setRunStatus('simulation reached max time — restart or tweak & rerun');
      } else {
        setHudStatus('error');
        setRunStatus(status.message, true);
      }
      updatePauseButton();
    });

    client.connect(wsUrl);
  } catch (error) {
    console.error('Failed to start simulation:', error);
    setRunStatus(`failed: ${(error as Error).message}`.slice(0, 120), true);
    runBtn.disabled = false;
  }
}

// --- HUD helpers ---------------------------------------------------------------

function setHudStatus(label: 'running' | 'paused' | 'finished' | 'error' | 'idle'): void {
  $('hud-label').textContent = label;
  const dot = $('hud-dot');
  dot.className = '';
  if (label !== 'idle') dot.classList.add(label);
}

function formatCount(n: number): string {
  return n >= 1000 ? `${(n / 1000).toFixed(1)}K` : `${n}`;
}

function formatSimTime(t: number): string {
  if (t < 1.0) return `${(t * 1000).toFixed(1)}ms`;
  return `${t.toFixed(3)}s`;
}

function updatePauseButton(): void {
  pauseBtn.textContent = paused ? 'Resume' : 'Pause';
  pauseBtn.classList.toggle('is-paused', paused);
  pauseBtn.disabled = finished;
}

const LEGEND_GRADIENTS: Record<ColorMode, string> = {
  speed: 'linear-gradient(90deg, #0d3a8c, #1aa6f2, #ffffff)',
  density: 'linear-gradient(90deg, #2e73f2, #ebf0f5, #f25940)',
  temperature: 'linear-gradient(90deg, #408cf2, #ffb854)',
  type: 'linear-gradient(90deg, #298cf2 49%, #c7ccd6 51%)',
};

function updateLegend(): void {
  const mode = colorSel.value as ColorMode;
  const label = $('legend-label');
  const min = $('legend-min');
  const max = $('legend-max');
  ($('legend-bar') as HTMLElement).style.background = LEGEND_GRADIENTS[mode];

  switch (mode) {
    case 'speed':
      label.textContent = 'particle speed';
      min.textContent = '0';
      max.textContent = `${particleRenderer ? speedMax() : '—'}`;
      break;
    case 'density':
      label.textContent = 'density / rest';
      min.textContent = '0.95';
      max.textContent = '1.05';
      break;
    case 'temperature':
      label.textContent = 'temperature';
      min.textContent = '283K';
      max.textContent = '323K';
      break;
    case 'type':
      label.textContent = 'fluid type';
      min.textContent = 'water';
      max.textContent = 'air';
      break;
  }
}

function speedMax(): string {
  const v = particleRenderer.speedScale;
  return v >= 1 ? `${v.toFixed(1)}m/s` : `${(v * 100).toFixed(0)}cm/s`;
}

// --- Wire up controls -------------------------------------------------------------

runBtn.addEventListener('click', () => { void startSimulation(); });

pauseBtn.addEventListener('click', () => {
  if (!client) return;
  if (paused) {
    client.sendCommand(CMD_RESUME);
    paused = false;
    setHudStatus('running');
  } else {
    client.sendCommand(CMD_PAUSE);
    paused = true;
    setHudStatus('paused');
  }
  updatePauseButton();
});

restartBtn.addEventListener('click', () => { void startSimulation(); });
resetViewBtn.addEventListener('click', () => resetView());

colorSel.addEventListener('change', () => {
  particleRenderer.setColorMode(colorSel.value as ColorMode);
  updateLegend();
});

sizeRng.addEventListener('input', () => {
  particleRenderer.setSizeScale(parseFloat(sizeRng.value));
});

wallsRng.addEventListener('input', () => {
  containerRenderer.setWallOpacity(parseFloat(wallsRng.value));
});

// Keyboard: space toggles pause
window.addEventListener('keydown', (e) => {
  if (e.code === 'Space' && client && !finished && (e.target as HTMLElement).tagName !== 'INPUT' && (e.target as HTMLElement).tagName !== 'SELECT') {
    e.preventDefault();
    pauseBtn.click();
  }
});

// --- Stats ticker ------------------------------------------------------------------

setInterval(() => {
  const now = performance.now();
  const dtSec = (now - lastStatsTime) / 1000;
  lastStatsTime = now;
  if (hudEl.style.display !== 'none') {
    $('st-fps').textContent = `${Math.round(renderFrames / dtSec)}`;
    $('st-data-fps').textContent = `${Math.round(dataFrames / dtSec)}`;
    if (lastDensityVar === null) $('st-density').textContent = '—';
  }
  renderFrames = 0;
  dataFrames = 0;
}, 1000);

// --- Render loop --------------------------------------------------------------------

function animationLoop(): void {
  requestAnimationFrame(animationLoop);
  particleRenderer.updateProjection(camera, canvas.clientHeight);
  animate();
  renderFrames++;
}

void loadPresets();
animationLoop();
console.log('ultraballpit viewer ready');
