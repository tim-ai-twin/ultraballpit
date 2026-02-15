// T036: STL geometry mesh rendering for domain boundaries

import * as THREE from 'three';
import { STLLoader } from 'three/examples/jsm/loaders/STLLoader.js';

/**
 * Renderer for STL geometry meshes (domain boundaries, obstacles)
 *
 * Features:
 * - Load STL files from server
 * - Semi-transparent rendering
 * - Wireframe overlay option
 * - Adjustable opacity
 */
export class GeometryRenderer {
  private scene: THREE.Scene;
  private mesh: THREE.Mesh | null = null;
  private wireframe: THREE.LineSegments | null = null;
  private loader: STLLoader;
  private opacity: number = 0.3;

  constructor(scene: THREE.Scene) {
    this.scene = scene;
    this.loader = new STLLoader();
  }

  /**
   * Load and render STL geometry from URL
   */
  async loadSTL(url: string): Promise<void> {
    return new Promise((resolve, reject) => {
      this.loader.load(
        url,
        (geometry) => {
          this.createMesh(geometry);
          resolve();
        },
        undefined,
        (error) => {
          console.error('Error loading STL:', error);
          reject(error);
        }
      );
    });
  }

  /**
   * Create mesh and wireframe from loaded geometry
   */
  private createMesh(geometry: THREE.BufferGeometry): void {
    // Clean up existing mesh
    this.dispose();

    // Compute normals for proper lighting
    geometry.computeVertexNormals();

    // Create semi-transparent mesh
    const material = new THREE.MeshStandardMaterial({
      color: 0x808080,
      transparent: true,
      opacity: this.opacity,
      side: THREE.DoubleSide,
      metalness: 0.3,
      roughness: 0.7,
    });

    this.mesh = new THREE.Mesh(geometry, material);
    this.scene.add(this.mesh);

    // Create wireframe overlay
    const wireframeGeometry = new THREE.WireframeGeometry(geometry);
    const wireframeMaterial = new THREE.LineBasicMaterial({
      color: 0xffffff,
      transparent: true,
      opacity: 0.2,
    });

    this.wireframe = new THREE.LineSegments(wireframeGeometry, wireframeMaterial);
    this.scene.add(this.wireframe);

    console.log('STL geometry loaded');
  }

  /**
   * Set opacity of the geometry mesh
   */
  setOpacity(opacity: number): void {
    this.opacity = Math.max(0, Math.min(1, opacity));

    if (this.mesh && this.mesh.material instanceof THREE.MeshStandardMaterial) {
      this.mesh.material.opacity = this.opacity;
      this.mesh.material.needsUpdate = true;
    }
  }

  /**
   * Clean up GPU resources
   */
  dispose(): void {
    if (this.mesh) {
      this.scene.remove(this.mesh);
      if (this.mesh.geometry) {
        this.mesh.geometry.dispose();
      }
      if (this.mesh.material instanceof THREE.Material) {
        this.mesh.material.dispose();
      }
      this.mesh = null;
    }

    if (this.wireframe) {
      this.scene.remove(this.wireframe);
      if (this.wireframe.geometry) {
        this.wireframe.geometry.dispose();
      }
      if (this.wireframe.material instanceof THREE.Material) {
        this.wireframe.material.dispose();
      }
      this.wireframe = null;
    }
  }
}
