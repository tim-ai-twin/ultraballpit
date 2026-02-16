// Debug overlays for simulation diagnostics and force visualization

import * as THREE from 'three';
import type { Diagnostics } from '../types/protocol.js';

/**
 * Diagnostics HUD overlay
 *
 * Displays real-time diagnostics metrics in the top-right corner:
 * - Frame time (ms)
 * - Particle count
 * - Max density variation
 * - Energy conservation error
 * - Mass conservation error
 * - Timestep dt
 */
export class DiagnosticsOverlay {
  private container: HTMLElement;
  private visible: boolean = false;

  constructor() {
    // Create overlay container
    this.container = document.createElement('div');
    this.container.id = 'diagnostics-overlay';
    this.container.style.cssText = `
      position: absolute;
      top: 20px;
      right: 20px;
      background: rgba(0, 0, 0, 0.75);
      color: #fff;
      padding: 12px 16px;
      border-radius: 4px;
      font-family: 'Courier New', monospace;
      font-size: 12px;
      line-height: 1.6;
      pointer-events: none;
      z-index: 1000;
      min-width: 250px;
      display: none;
    `;

    // Add to document
    document.body.appendChild(this.container);
  }

  /**
   * Update diagnostics display
   */
  update(diagnostics: Diagnostics): void {
    const formatNumber = (value: number, decimals: number = 3): string => {
      if (!isFinite(value)) return 'N/A';
      return value.toFixed(decimals);
    };

    const formatScientific = (value: number): string => {
      if (!isFinite(value)) return 'N/A';
      if (value === 0) return '0.000';
      return value.toExponential(3);
    };

    // Color coding for errors
    const getErrorColor = (value: number, thresholdWarning: number, thresholdError: number): string => {
      if (value >= thresholdError) return '#f44336'; // Red
      if (value >= thresholdWarning) return '#FF9800'; // Orange
      return '#4CAF50'; // Green
    };

    const densityColor = getErrorColor(diagnostics.maxDensityVariation, 5.0, 20.0);
    const energyColor = getErrorColor(diagnostics.energyConservationError, 0.01, 0.1);
    const massColor = getErrorColor(diagnostics.massConservationError, 0.001, 0.01);

    this.container.innerHTML = `
      <div style="font-weight: bold; margin-bottom: 8px; border-bottom: 1px solid #555; padding-bottom: 4px;">
        DIAGNOSTICS
      </div>
      <div>Frame: ${diagnostics.frameNumber}</div>
      <div>Particles: ${diagnostics.particleCount.toLocaleString()}</div>
      <div>dt: ${formatNumber(diagnostics.dt, 6)} s</div>
      <div>Frame Time: ${formatNumber(diagnostics.frameTimeMs, 2)} ms</div>
      <div style="margin-top: 8px; border-top: 1px solid #555; padding-top: 4px;">
        <div style="color: ${densityColor}">
          Max Density Var: ${formatNumber(diagnostics.maxDensityVariation, 2)}
        </div>
        <div style="color: ${energyColor}">
          Energy Error: ${formatScientific(diagnostics.energyConservationError)}
        </div>
        <div style="color: ${massColor}">
          Mass Error: ${formatScientific(diagnostics.massConservationError)}
        </div>
      </div>
    `;
  }

  /**
   * Show the overlay
   */
  show(): void {
    this.visible = true;
    this.container.style.display = 'block';
  }

  /**
   * Hide the overlay
   */
  hide(): void {
    this.visible = false;
    this.container.style.display = 'none';
  }

  /**
   * Toggle visibility
   */
  toggle(): void {
    if (this.visible) {
      this.hide();
    } else {
      this.show();
    }
  }

  /**
   * Check if visible
   */
  isVisible(): boolean {
    return this.visible;
  }

  /**
   * Cleanup and remove from DOM
   */
  destroy(): void {
    if (this.container.parentNode) {
      this.container.parentNode.removeChild(this.container);
    }
  }
}

/**
 * Force data from API
 */
export interface ForceRecord {
  timestep: number;
  sim_time: number;
  net_force: [number, number, number];
  net_moment: [number, number, number];
}

/**
 * Force overlay manager
 *
 * Displays pressure/force magnitude as a colored overlay on the geometry mesh.
 * Uses a color gradient from blue (low) to red (high).
 */
export class ForceOverlay {
  private scene: THREE.Scene;
  private overlayMesh: THREE.Mesh | null = null;
  private simulationId: string | null = null;
  private enabled: boolean = false;
  private apiBaseUrl: string;

  constructor(scene: THREE.Scene, apiBaseUrl: string = 'http://localhost:3000/api') {
    this.scene = scene;
    this.apiBaseUrl = apiBaseUrl;
  }

  /**
   * Enable the force overlay for a simulation
   */
  async enable(simulationId: string): Promise<void> {
    this.simulationId = simulationId;
    this.enabled = true;
    await this.update();
  }

  /**
   * Disable the force overlay
   */
  disable(): void {
    this.enabled = false;
    if (this.overlayMesh) {
      this.scene.remove(this.overlayMesh);
      this.overlayMesh = null;
    }
  }

  /**
   * Toggle the overlay on/off
   */
  async toggle(simulationId: string): Promise<void> {
    if (this.enabled) {
      this.disable();
    } else {
      await this.enable(simulationId);
    }
  }

  /**
   * Update the overlay with latest force data
   */
  async update(): Promise<void> {
    if (!this.enabled || !this.simulationId) {
      return;
    }

    try {
      // Fetch latest force data (mean aggregation for simplicity)
      const response = await fetch(
        `${this.apiBaseUrl}/simulations/${this.simulationId}/forces?aggregation=mean`
      );

      if (!response.ok) {
        console.warn('Failed to fetch force data:', response.statusText);
        return;
      }

      const data = await response.json();
      const meanForce = data.mean_force as [number, number, number] | null;

      if (!meanForce) {
        console.log('No force data available yet');
        return;
      }

      // Compute force magnitude
      const [fx, fy, fz] = meanForce;
      const forceMagnitude = Math.sqrt(fx * fx + fy * fy + fz * fz);

      // Map magnitude to color (blue -> cyan -> green -> yellow -> red)
      const color = this.forceToColor(forceMagnitude);

      // Update or create overlay visualization
      this.updateOverlayVisualization(forceMagnitude, color);
    } catch (error) {
      console.error('Error updating force overlay:', error);
    }
  }

  /**
   * Map force magnitude to color
   * Uses a gradient: blue (0) -> cyan -> green -> yellow -> red (max)
   */
  private forceToColor(magnitude: number): THREE.Color {
    // Normalize magnitude to [0, 1]
    // For typical SPH simulations, forces in the range [0, 0.01] N
    const maxForce = 0.01; // 10 mN
    const normalized = Math.min(magnitude / maxForce, 1.0);

    // Create color gradient
    if (normalized < 0.25) {
      // Blue to cyan
      const t = normalized / 0.25;
      return new THREE.Color().setRGB(0, t, 1);
    } else if (normalized < 0.5) {
      // Cyan to green
      const t = (normalized - 0.25) / 0.25;
      return new THREE.Color().setRGB(0, 1, 1 - t);
    } else if (normalized < 0.75) {
      // Green to yellow
      const t = (normalized - 0.5) / 0.25;
      return new THREE.Color().setRGB(t, 1, 0);
    } else {
      // Yellow to red
      const t = (normalized - 0.75) / 0.25;
      return new THREE.Color().setRGB(1, 1 - t, 0);
    }
  }

  /**
   * Update or create the overlay visualization
   * For now, we'll just show a colored indicator rather than a full mesh overlay
   */
  private updateOverlayVisualization(magnitude: number, color: THREE.Color): void {
    // Remove old overlay if it exists
    if (this.overlayMesh) {
      this.scene.remove(this.overlayMesh);
    }

    // Create a simple visualization: a small sphere indicator
    // In a full implementation, this would color the geometry mesh
    const geometry = new THREE.SphereGeometry(0.001, 16, 16);
    const material = new THREE.MeshPhongMaterial({
      color: color,
      emissive: color,
      emissiveIntensity: 0.5,
      transparent: true,
      opacity: 0.8,
    });

    this.overlayMesh = new THREE.Mesh(geometry, material);
    this.overlayMesh.position.set(0.005, 0.008, 0); // Position in scene
    this.scene.add(this.overlayMesh);

    console.log(`Force overlay: magnitude=${magnitude.toExponential(3)} N, color=${color.getHexString()}`);
  }

  /**
   * Check if overlay is currently enabled
   */
  isEnabled(): boolean {
    return this.enabled;
  }
}
