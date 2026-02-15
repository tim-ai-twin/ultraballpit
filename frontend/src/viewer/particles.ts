// T035: Particle renderer using THREE.Points with custom shader

import * as THREE from 'three';
import { FLUID_WATER, FLUID_AIR, type ParticleData } from '../types/protocol.js';

// Vertex shader: position from attribute, point size based on distance
const vertexShader = `
  attribute vec3 color;
  varying vec3 vColor;

  void main() {
    vColor = color;
    vec4 mvPosition = modelViewMatrix * vec4(position, 1.0);
    gl_PointSize = 4.0 * (300.0 / -mvPosition.z);
    gl_Position = projectionMatrix * mvPosition;
  }
`;

// Fragment shader: circular points with color based on fluid type
const fragmentShader = `
  varying vec3 vColor;

  void main() {
    // Make points circular
    vec2 center = gl_PointCoord - vec2(0.5);
    float dist = length(center);
    if (dist > 0.5) discard;

    // Smooth edge
    float alpha = 1.0 - smoothstep(0.4, 0.5, dist);
    gl_FragColor = vec4(vColor, alpha);
  }
`;

/**
 * Particle renderer using GPU-accelerated point rendering
 *
 * Features:
 * - Efficient rendering of thousands of particles
 * - Color coding by fluid type (water=blue, air=gray)
 * - Temperature-based brightness modulation
 * - Circular point sprites
 */
export class ParticleRenderer {
  private scene: THREE.Scene;
  private points: THREE.Points | null = null;
  private geometry: THREE.BufferGeometry | null = null;
  private material: THREE.ShaderMaterial | null = null;
  private maxParticles: number = 0;

  constructor(scene: THREE.Scene) {
    this.scene = scene;
  }

  /**
   * Update particle positions and colors from simulation data
   */
  update(particles: ParticleData, count: number): void {
    if (!this.geometry || !this.material) {
      this.initializeGeometry(count);
    }

    if (!this.geometry || !this.material) {
      console.error('Failed to initialize geometry');
      return;
    }

    // Reallocate if particle count increased
    if (count > this.maxParticles) {
      this.dispose();
      this.initializeGeometry(count);
      if (!this.geometry || !this.material) {
        return;
      }
    }

    // Update position buffer
    const positions = new Float32Array(count * 3);
    for (let i = 0; i < count; i++) {
      positions[i * 3] = particles.x[i];
      positions[i * 3 + 1] = particles.y[i];
      positions[i * 3 + 2] = particles.z[i];
    }

    // Update color buffer based on fluid type and temperature
    const colors = new Float32Array(count * 3);
    for (let i = 0; i < count; i++) {
      const fluidType = particles.fluidType[i];
      const temp = particles.temperature[i];

      // Base color by fluid type
      let r: number, g: number, b: number;
      if (fluidType === FLUID_WATER) {
        // Water: blue (0x2196F3)
        r = 0x21 / 255;
        g = 0x96 / 255;
        b = 0xF3 / 255;
      } else if (fluidType === FLUID_AIR) {
        // Air: light gray (0xBDBDBD)
        r = 0xBD / 255;
        g = 0xBD / 255;
        b = 0xBD / 255;
      } else {
        // Mixed or unknown: purple
        r = 0.7;
        g = 0.3;
        b = 0.9;
      }

      // Modulate brightness by temperature
      // Assume temp range [290, 310] K for typical water simulations
      const tempNorm = Math.max(0, Math.min(1, (temp - 290) / 20));
      const brightness = 0.7 + tempNorm * 0.3;

      colors[i * 3] = r * brightness;
      colors[i * 3 + 1] = g * brightness;
      colors[i * 3 + 2] = b * brightness;
    }

    // Update geometry attributes
    this.geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    this.geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
    this.geometry.setDrawRange(0, count);
    this.geometry.attributes.position.needsUpdate = true;
    this.geometry.attributes.color.needsUpdate = true;
    this.geometry.computeBoundingSphere();
  }

  /**
   * Initialize geometry and material for particle rendering
   */
  private initializeGeometry(count: number): void {
    // Allocate geometry with buffer for particle count
    this.maxParticles = Math.max(count, 1000);
    this.geometry = new THREE.BufferGeometry();

    // Pre-allocate buffers
    const positions = new Float32Array(this.maxParticles * 3);
    const colors = new Float32Array(this.maxParticles * 3);

    this.geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    this.geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));

    // Create shader material
    this.material = new THREE.ShaderMaterial({
      vertexShader,
      fragmentShader,
      transparent: true,
      depthWrite: false,
      blending: THREE.AdditiveBlending,
    });

    // Create points mesh
    this.points = new THREE.Points(this.geometry, this.material);
    this.scene.add(this.points);
  }

  /**
   * Clean up GPU resources
   */
  dispose(): void {
    if (this.points) {
      this.scene.remove(this.points);
      this.points = null;
    }
    if (this.geometry) {
      this.geometry.dispose();
      this.geometry = null;
    }
    if (this.material) {
      this.material.dispose();
      this.material = null;
    }
  }
}
