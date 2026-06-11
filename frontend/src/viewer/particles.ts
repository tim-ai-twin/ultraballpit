// Particle renderer: physically-sized, sphere-shaded point sprites
//
// Particles render at their true physical size (the SPH particle spacing)
// with lambertian sphere shading in the fragment shader — no additive
// blending, no fixed pixel sizes. Color encodes a selectable quantity.

import * as THREE from 'three';
import { FLUID_WATER, type ParticleData } from '../types/protocol.js';

export type ColorMode = 'speed' | 'density' | 'temperature' | 'type';

// Vertex shader: world-size points with perspective-correct scaling
const vertexShader = `
  attribute vec3 color;
  varying vec3 vColor;

  uniform float uRadius;      // particle radius in world units
  uniform float uPointScale;  // viewportHeight / (2 * tan(fov/2))
  uniform float uSizeScale;   // user size multiplier

  void main() {
    vColor = color;
    vec4 mvPosition = modelViewMatrix * vec4(position, 1.0);
    gl_PointSize = uRadius * 2.0 * uSizeScale * uPointScale / -mvPosition.z;
    gl_Position = projectionMatrix * mvPosition;
  }
`;

// Fragment shader: shade each sprite as a lit sphere
const fragmentShader = `
  varying vec3 vColor;

  void main() {
    vec2 coord = gl_PointCoord * 2.0 - 1.0;
    float r2 = dot(coord, coord);
    if (r2 > 1.0) discard;

    // Reconstruct sphere normal (gl_PointCoord y is down)
    vec3 normal = vec3(coord.x, -coord.y, sqrt(1.0 - r2));

    vec3 lightDir = normalize(vec3(0.45, 0.7, 0.55));
    float diffuse = max(dot(normal, lightDir), 0.0);
    float spec = pow(max(dot(reflect(-lightDir, normal), vec3(0.0, 0.0, 1.0)), 0.0), 24.0);

    vec3 shaded = vColor * (0.38 + 0.62 * diffuse) + vec3(0.9, 0.95, 1.0) * spec * 0.25;
    gl_FragColor = vec4(shaded, 1.0);
  }
`;

// --- Color ramps -----------------------------------------------------------

/** Speed ramp: deep ocean blue -> cyan -> white foam */
function speedRamp(t: number, out: [number, number, number]): void {
  if (t < 0.5) {
    const u = t * 2.0;
    out[0] = 0.05 + (0.10 - 0.05) * u;
    out[1] = 0.22 + (0.65 - 0.22) * u;
    out[2] = 0.55 + (0.95 - 0.55) * u;
  } else {
    const u = (t - 0.5) * 2.0;
    out[0] = 0.10 + (1.0 - 0.10) * u;
    out[1] = 0.65 + (1.0 - 0.65) * u;
    out[2] = 0.95 + (1.0 - 0.95) * u;
  }
}

/** Speed ramp for air particles: smoke gray -> warm white (distinct from water) */
function airSpeedRamp(t: number, out: [number, number, number]): void {
  const u = Math.max(0, Math.min(1, t));
  out[0] = 0.42 + (1.0 - 0.42) * u;
  out[1] = 0.43 + (0.95 - 0.43) * u;
  out[2] = 0.46 + (0.82 - 0.46) * u;
}

/** Diverging density ramp: blue (light) <- white -> red (dense) */
function densityRamp(t: number, out: [number, number, number]): void {
  const u = Math.max(0, Math.min(1, t));
  if (u < 0.5) {
    const v = u * 2.0;
    out[0] = 0.18 + (0.92 - 0.18) * v;
    out[1] = 0.45 + (0.94 - 0.45) * v;
    out[2] = 0.95 + (0.96 - 0.95) * v;
  } else {
    const v = (u - 0.5) * 2.0;
    out[0] = 0.92 + (0.95 - 0.92) * v;
    out[1] = 0.94 - (0.94 - 0.35) * v;
    out[2] = 0.96 - (0.96 - 0.25) * v;
  }
}

/** Temperature ramp: ice blue -> amber */
function temperatureRamp(t: number, out: [number, number, number]): void {
  const u = Math.max(0, Math.min(1, t));
  out[0] = 0.25 + (1.0 - 0.25) * u;
  out[1] = 0.55 + (0.72 - 0.55) * u;
  out[2] = 0.95 - (0.95 - 0.25) * u;
}

/**
 * Particle renderer using GPU point sprites with sphere shading.
 */
export class ParticleRenderer {
  private scene: THREE.Scene;
  private points: THREE.Points | null = null;
  private geometry: THREE.BufferGeometry | null = null;
  private material: THREE.ShaderMaterial | null = null;
  private maxParticles = 0;

  private colorMode: ColorMode = 'speed';
  private sizeScale = 1.0;
  private particleRadius = 0.001;

  /** Smoothed max speed for stable speed-color normalization */
  private vmaxSmooth = 0.1;

  /** Current speed-color normalization ceiling (m/s) */
  get speedScale(): number {
    return this.vmaxSmooth;
  }

  private lastParticles: ParticleData | null = null;
  private lastCount = 0;

  constructor(scene: THREE.Scene) {
    this.scene = scene;
  }

  /** Set the physical particle radius (typically half the particle spacing) */
  setParticleRadius(radius: number): void {
    this.particleRadius = radius;
    if (this.material) {
      this.material.uniforms.uRadius.value = radius;
    }
  }

  /** User size multiplier (1 = physical size) */
  setSizeScale(scale: number): void {
    this.sizeScale = scale;
    if (this.material) {
      this.material.uniforms.uSizeScale.value = scale;
    }
  }

  setColorMode(mode: ColorMode): void {
    this.colorMode = mode;
    // Recolor the last frame immediately so the change is visible while paused
    if (this.lastParticles && this.geometry) {
      this.writeColors(this.lastParticles, this.lastCount);
      this.geometry.attributes.color.needsUpdate = true;
    }
  }

  /** Must be called when the camera FOV or viewport height changes */
  updateProjection(camera: THREE.PerspectiveCamera, viewportHeightPx: number): void {
    if (this.material) {
      const fovRad = (camera.fov * Math.PI) / 180;
      this.material.uniforms.uPointScale.value =
        viewportHeightPx / (2.0 * Math.tan(fovRad / 2.0));
    }
  }

  /**
   * Update particle positions and colors from simulation data
   */
  update(particles: ParticleData, count: number): void {
    if (!this.geometry || !this.material || count > this.maxParticles) {
      this.rebuild(count);
    }
    if (!this.geometry || !this.material) return;

    const positions = this.geometry.attributes.position.array as Float32Array;
    for (let i = 0; i < count; i++) {
      positions[i * 3] = particles.x[i];
      positions[i * 3 + 1] = particles.y[i];
      positions[i * 3 + 2] = particles.z[i];
    }

    this.lastParticles = particles;
    this.lastCount = count;
    this.writeColors(particles, count);

    this.geometry.setDrawRange(0, count);
    this.geometry.attributes.position.needsUpdate = true;
    this.geometry.attributes.color.needsUpdate = true;
    this.geometry.computeBoundingSphere();
  }

  private writeColors(particles: ParticleData, count: number): void {
    if (!this.geometry) return;
    const colors = this.geometry.attributes.color.array as Float32Array;
    const rgb: [number, number, number] = [0, 0, 0];

    switch (this.colorMode) {
      case 'speed': {
        // Track a smoothed max speed so normalization is stable across frames
        let vmax = 0;
        for (let i = 0; i < count; i++) {
          const s2 =
            particles.vx[i] * particles.vx[i] +
            particles.vy[i] * particles.vy[i] +
            particles.vz[i] * particles.vz[i];
          if (s2 > vmax) vmax = s2;
        }
        vmax = Math.sqrt(vmax);
        this.vmaxSmooth = Math.max(vmax, this.vmaxSmooth * 0.995, 0.05);

        const inv = 1.0 / this.vmaxSmooth;
        for (let i = 0; i < count; i++) {
          const speed = Math.sqrt(
            particles.vx[i] * particles.vx[i] +
              particles.vy[i] * particles.vy[i] +
              particles.vz[i] * particles.vz[i],
          );
          const t = Math.min(1, speed * inv);
          if (particles.fluidType[i] === FLUID_WATER) {
            speedRamp(t, rgb);
          } else {
            airSpeedRamp(t, rgb);
          }
          colors[i * 3] = rgb[0];
          colors[i * 3 + 1] = rgb[1];
          colors[i * 3 + 2] = rgb[2];
        }
        break;
      }

      case 'density': {
        // density_ratio is fixed-point (rho/rho0)*1000; map ±5% around rest
        for (let i = 0; i < count; i++) {
          const ratio = particles.densityRatio[i] / 1000.0;
          densityRamp((ratio - 0.95) / 0.10, rgb);
          colors[i * 3] = rgb[0];
          colors[i * 3 + 1] = rgb[1];
          colors[i * 3 + 2] = rgb[2];
        }
        break;
      }

      case 'temperature': {
        for (let i = 0; i < count; i++) {
          temperatureRamp((particles.temperature[i] - 283.15) / 40.0, rgb);
          colors[i * 3] = rgb[0];
          colors[i * 3 + 1] = rgb[1];
          colors[i * 3 + 2] = rgb[2];
        }
        break;
      }

      case 'type': {
        for (let i = 0; i < count; i++) {
          if (particles.fluidType[i] === FLUID_WATER) {
            colors[i * 3] = 0.16;
            colors[i * 3 + 1] = 0.55;
            colors[i * 3 + 2] = 0.95;
          } else {
            colors[i * 3] = 0.78;
            colors[i * 3 + 1] = 0.8;
            colors[i * 3 + 2] = 0.84;
          }
        }
        break;
      }
    }
  }

  private rebuild(count: number): void {
    const uniforms = this.material?.uniforms;
    this.dispose();

    this.maxParticles = Math.max(count, 1024);
    this.geometry = new THREE.BufferGeometry();
    this.geometry.setAttribute(
      'position',
      new THREE.BufferAttribute(new Float32Array(this.maxParticles * 3), 3),
    );
    this.geometry.setAttribute(
      'color',
      new THREE.BufferAttribute(new Float32Array(this.maxParticles * 3), 3),
    );

    this.material = new THREE.ShaderMaterial({
      vertexShader,
      fragmentShader,
      uniforms: {
        uRadius: { value: this.particleRadius },
        uPointScale: { value: uniforms?.uPointScale.value ?? 800.0 },
        uSizeScale: { value: this.sizeScale },
      },
    });

    this.points = new THREE.Points(this.geometry, this.material);
    this.points.frustumCulled = false;
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
    this.lastParticles = null;
    this.lastCount = 0;
  }
}
