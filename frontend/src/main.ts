// T039: Main entry point - wire up scene, client, UI components

import { createScene } from './viewer/scene.js';
import { SimulationClient } from './transport/client.js';
import { ParticleRenderer } from './viewer/particles.js';
import { GeometryRenderer } from './viewer/geometry.js';
import { ConfigList } from './ui/config-list.js';
import { SimControls } from './ui/controls.js';
import { CMD_PAUSE, CMD_RESUME, STATUS_RUNNING, STATUS_PAUSED } from './types/protocol.js';

console.log('SPH Fluid Simulation Viewer initializing...');

// API base URL (empty for relative URLs - works with Vite proxy and production)
const API_BASE = '';

// Application state
let currentSimulationId: string | null = null;
let simulationClient: SimulationClient | null = null;
let particleRenderer: ParticleRenderer | null = null;

// Stats tracking
let frameCount = 0;
let lastFrameTime = performance.now();
let fps = 0;

// Initialize scene
const canvas = document.getElementById('viewer') as HTMLCanvasElement;
if (!canvas) {
  throw new Error('Canvas element not found');
}

const { scene, animate } = createScene(canvas);
console.log('Scene initialized');

// Initialize UI components
const configListContainer = document.getElementById('config-list');
const controlsContainer = document.getElementById('controls');

if (!configListContainer || !controlsContainer) {
  throw new Error('UI containers not found');
}

const configList = new ConfigList(configListContainer);
const simControls = new SimControls(controlsContainer);

// Load available configs
configList.refresh().catch((error) => {
  console.error('Failed to load configs:', error);
});

// Handle config selection
configList.onSelect((configName) => {
  console.log('Config selected:', configName);
  simControls.setConfigName(configName);
});

// Handle start simulation
simControls.onStart(async (configName) => {
  console.log('Starting simulation with config:', configName);
  simControls.setStatus('starting');

  try {
    // Create simulation via REST API
    const response = await fetch(`${API_BASE}/api/simulations`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        config: configName,
      }),
    });

    if (!response.ok) {
      throw new Error(`HTTP ${response.status}: ${response.statusText}`);
    }

    const data = await response.json();
    currentSimulationId = data.simulation_id;

    // Build WebSocket URL from current location (works with Vite proxy and production)
    const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${wsProtocol}//${window.location.host}/ws/simulation/${data.simulation_id}`;

    console.log('Simulation created:', data);

    // Connect to WebSocket
    simulationClient = new SimulationClient();

    // Handle SimInfo (sent once on connect)
    simulationClient.onSimInfo((simInfo) => {
      console.log('SimInfo received:', simInfo);

      // Initialize particle renderer
      particleRenderer = new ParticleRenderer(scene);

      // Initialize geometry renderer (available for future use)
      const geometryRenderer = new GeometryRenderer(scene);

      // Optionally load STL geometry (if available)
      // geometryRenderer.loadSTL(`${API_BASE}/api/geometry/${currentSimulationId}.stl`);
      // For now, suppress unused variable warning
      void geometryRenderer;

      updateStats({ particles: simInfo.particleCount });
    });

    // Handle Frame updates
    simulationClient.onFrame((frame) => {
      // Update particle renderer
      if (particleRenderer) {
        particleRenderer.update(frame.particles, frame.particleCount);
      }

      // Update stats
      updateStats({
        frame: Number(frame.frameNumber),
        time: frame.simTime.toFixed(4),
        particles: frame.particleCount,
      });

      // Calculate FPS
      frameCount++;
      const now = performance.now();
      if (now - lastFrameTime >= 1000) {
        fps = Math.round((frameCount * 1000) / (now - lastFrameTime));
        frameCount = 0;
        lastFrameTime = now;
        updateStats({ fps });
      }
    });

    // Handle status updates
    simulationClient.onStatus((status) => {
      console.log('SimStatus received:', status);

      if (status.status === STATUS_RUNNING) {
        simControls.setStatus('running');
      } else if (status.status === STATUS_PAUSED) {
        simControls.setStatus('paused');
      } else {
        simControls.setStatus(status.message);
      }
    });

    // Connect to WebSocket
    simulationClient.connect(wsUrl);
    simControls.setStatus('connecting');

    // After connection, status will be updated via onStatus callback
    setTimeout(() => {
      if (simControls) {
        simControls.setStatus('running');
      }
    }, 500);
  } catch (error) {
    console.error('Failed to start simulation:', error);
    simControls.setStatus('error');
    alert('Failed to start simulation: ' + (error as Error).message);
  }
});

// Handle pause
simControls.onPause(() => {
  console.log('Pausing simulation');
  if (simulationClient) {
    simulationClient.sendCommand(CMD_PAUSE);
    simControls.setStatus('paused');
  }

  // Also call REST API
  if (currentSimulationId) {
    fetch(`${API_BASE}/api/simulations/${currentSimulationId}/pause`, {
      method: 'POST',
    }).catch((error) => {
      console.error('Failed to pause via REST API:', error);
    });
  }
});

// Handle resume
simControls.onResume(() => {
  console.log('Resuming simulation');
  if (simulationClient) {
    simulationClient.sendCommand(CMD_RESUME);
    simControls.setStatus('running');
  }

  // Also call REST API
  if (currentSimulationId) {
    fetch(`${API_BASE}/api/simulations/${currentSimulationId}/resume`, {
      method: 'POST',
    }).catch((error) => {
      console.error('Failed to resume via REST API:', error);
    });
  }
});

// Update stats display
function updateStats(data: {
  fps?: number;
  particles?: number;
  frame?: number;
  time?: string;
}): void {
  if (data.fps !== undefined) {
    const fpsEl = document.getElementById('stats-fps');
    if (fpsEl) fpsEl.textContent = `FPS: ${data.fps}`;
  }

  if (data.particles !== undefined) {
    const particlesEl = document.getElementById('stats-particles');
    if (particlesEl) {
      const formatted = data.particles >= 1000
        ? `${(data.particles / 1000).toFixed(1)}K`
        : data.particles.toString();
      particlesEl.textContent = `Particles: ${formatted}`;
    }
  }

  if (data.frame !== undefined) {
    const frameEl = document.getElementById('stats-frame');
    if (frameEl) frameEl.textContent = `Frame: ${data.frame}`;
  }

  if (data.time !== undefined) {
    const timeEl = document.getElementById('stats-time');
    if (timeEl) timeEl.textContent = `Time: ${data.time}s`;
  }
}

// Animation loop
function animationLoop(): void {
  requestAnimationFrame(animationLoop);
  animate();
}

// Start animation
animationLoop();
console.log('Animation loop started');
