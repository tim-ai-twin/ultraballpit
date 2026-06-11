// Container renderer: domain bounding box with per-wall boundary-condition
// visualization, plus the obstacle mesh fetched from the server.

import * as THREE from 'three';

/** Per-face boundary condition kind (matches config JSON values) */
export type WallKind = 'Wall' | 'Outflow' | 'Periodic' | 'Inflow';

export interface WallSpec {
  x_min: WallKind;
  x_max: WallKind;
  y_min: WallKind;
  y_max: WallKind;
  z_min: WallKind;
  z_max: WallKind;
}

const WALL_COLORS: Record<WallKind, number> = {
  Wall: 0x3d5166,
  Outflow: 0x2fbf71,
  Periodic: 0xc77dff,
  Inflow: 0xffb454,
};

/**
 * Renders the simulation container:
 * - hairline domain edges
 * - translucent face planes tinted by boundary condition
 *   (solid walls visible, open faces nearly invisible)
 * - obstacle geometry as a glassy solid
 */
export class ContainerRenderer {
  private scene: THREE.Scene;
  private group: THREE.Group | null = null;
  private obstacle: THREE.Mesh | null = null;
  private wallOpacity = 0.16;
  private wallMaterials: { mat: THREE.MeshBasicMaterial; kind: WallKind }[] = [];

  constructor(scene: THREE.Scene) {
    this.scene = scene;
  }

  /** Build (or rebuild) the container for the given domain and wall specs */
  build(
    min: [number, number, number],
    max: [number, number, number],
    walls: WallSpec,
  ): void {
    this.disposeGroup();
    this.group = new THREE.Group();
    this.wallMaterials = [];

    const size = new THREE.Vector3(max[0] - min[0], max[1] - min[1], max[2] - min[2]);
    const center = new THREE.Vector3(
      (min[0] + max[0]) / 2,
      (min[1] + max[1]) / 2,
      (min[2] + max[2]) / 2,
    );

    // Hairline edges
    const boxGeom = new THREE.BoxGeometry(size.x, size.y, size.z);
    const edges = new THREE.EdgesGeometry(boxGeom);
    const edgeLines = new THREE.LineSegments(
      edges,
      new THREE.LineBasicMaterial({ color: 0x7da7c4, transparent: true, opacity: 0.85 }),
    );
    edgeLines.position.copy(center);
    this.group.add(edgeLines);
    boxGeom.dispose();

    // Face planes
    const faces: {
      kind: WallKind;
      w: number;
      h: number;
      pos: THREE.Vector3;
      rot: THREE.Euler;
    }[] = [
      {
        kind: walls.x_min,
        w: size.z,
        h: size.y,
        pos: new THREE.Vector3(min[0], center.y, center.z),
        rot: new THREE.Euler(0, Math.PI / 2, 0),
      },
      {
        kind: walls.x_max,
        w: size.z,
        h: size.y,
        pos: new THREE.Vector3(max[0], center.y, center.z),
        rot: new THREE.Euler(0, -Math.PI / 2, 0),
      },
      {
        kind: walls.y_min,
        w: size.x,
        h: size.z,
        pos: new THREE.Vector3(center.x, min[1], center.z),
        rot: new THREE.Euler(-Math.PI / 2, 0, 0),
      },
      {
        kind: walls.y_max,
        w: size.x,
        h: size.z,
        pos: new THREE.Vector3(center.x, max[1], center.z),
        rot: new THREE.Euler(Math.PI / 2, 0, 0),
      },
      {
        kind: walls.z_min,
        w: size.x,
        h: size.y,
        pos: new THREE.Vector3(center.x, center.y, min[2]),
        rot: new THREE.Euler(0, 0, 0),
      },
      {
        kind: walls.z_max,
        w: size.x,
        h: size.y,
        pos: new THREE.Vector3(center.x, center.y, max[2]),
        rot: new THREE.Euler(0, Math.PI, 0),
      },
    ];

    for (const face of faces) {
      const mat = new THREE.MeshBasicMaterial({
        color: WALL_COLORS[face.kind],
        transparent: true,
        opacity: this.faceOpacity(face.kind),
        side: THREE.DoubleSide,
        depthWrite: false,
      });
      const plane = new THREE.Mesh(new THREE.PlaneGeometry(face.w, face.h), mat);
      plane.position.copy(face.pos);
      plane.rotation.copy(face.rot);
      this.group.add(plane);
      this.wallMaterials.push({ mat, kind: face.kind });
    }

    this.scene.add(this.group);
  }

  private faceOpacity(kind: WallKind): number {
    // Solid walls read as glass panes; open boundaries nearly invisible
    return kind === 'Wall' ? this.wallOpacity : this.wallOpacity * 0.35;
  }

  /** Adjust wall opacity (0 hides the container faces entirely) */
  setWallOpacity(opacity: number): void {
    this.wallOpacity = opacity;
    for (const { mat, kind } of this.wallMaterials) {
      mat.opacity = this.faceOpacity(kind);
      mat.visible = opacity > 0.005;
    }
  }

  /** Load and render the obstacle mesh for a simulation */
  async loadObstacle(simulationId: string): Promise<void> {
    this.disposeObstacle();
    try {
      const response = await fetch(`/api/simulations/${simulationId}/mesh`);
      if (!response.ok) return;
      const data = (await response.json()) as { triangle_count: number; vertices: number[] };
      if (!data.triangle_count) return;

      const geometry = new THREE.BufferGeometry();
      geometry.setAttribute(
        'position',
        new THREE.BufferAttribute(new Float32Array(data.vertices), 3),
      );
      geometry.computeVertexNormals();

      const material = new THREE.MeshStandardMaterial({
        color: 0xc8d4e0,
        metalness: 0.15,
        roughness: 0.35,
        transparent: true,
        opacity: 0.85,
      });
      this.obstacle = new THREE.Mesh(geometry, material);
      this.scene.add(this.obstacle);
    } catch (error) {
      console.warn('Failed to load obstacle mesh:', error);
    }
  }

  private disposeGroup(): void {
    if (this.group) {
      this.group.traverse((obj) => {
        if (obj instanceof THREE.Mesh || obj instanceof THREE.LineSegments) {
          obj.geometry.dispose();
          (obj.material as THREE.Material).dispose();
        }
      });
      this.scene.remove(this.group);
      this.group = null;
    }
    this.wallMaterials = [];
  }

  private disposeObstacle(): void {
    if (this.obstacle) {
      this.obstacle.geometry.dispose();
      (this.obstacle.material as THREE.Material).dispose();
      this.scene.remove(this.obstacle);
      this.obstacle = null;
    }
  }

  dispose(): void {
    this.disposeGroup();
    this.disposeObstacle();
  }
}
