// Three.js scene setup with camera, lighting, and controls

import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';

export interface SceneComponents {
  scene: THREE.Scene;
  camera: THREE.PerspectiveCamera;
  renderer: THREE.WebGLRenderer;
  controls: OrbitControls;
  animate: () => void;
  /** Frame the camera, grid and clip planes around a domain bounding box */
  fitToDomain: (min: [number, number, number], max: [number, number, number]) => void;
  /** Return the camera to the last fitted home position */
  resetView: () => void;
}

/**
 * Create and configure the Three.js scene for SPH visualization.
 *
 * The scene is scale-agnostic: call `fitToDomain` whenever a simulation's
 * bounds become known and the camera, lights, grid and clipping planes are
 * reframed around it.
 */
export function createScene(canvas: HTMLCanvasElement): SceneComponents {
  const scene = new THREE.Scene();
  // Transparent clear color: the page CSS provides a radial-gradient backdrop
  const renderer = new THREE.WebGLRenderer({
    canvas,
    antialias: true,
    alpha: true,
  });
  renderer.setSize(canvas.clientWidth, canvas.clientHeight, false);
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));

  const aspect = canvas.clientWidth / Math.max(1, canvas.clientHeight);
  const camera = new THREE.PerspectiveCamera(50, aspect, 0.001, 100);
  camera.position.set(0.15, 0.12, 0.18);
  camera.lookAt(0, 0, 0);

  const controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.dampingFactor = 0.08;
  controls.target.set(0, 0, 0);
  controls.update();

  // Lighting: cool ambient fill + warm key + subtle rim
  const ambientLight = new THREE.AmbientLight(0xbfd4e6, 0.55);
  scene.add(ambientLight);

  const keyLight = new THREE.DirectionalLight(0xfff4e0, 1.4);
  keyLight.position.set(1, 1.6, 0.8);
  scene.add(keyLight);

  const rimLight = new THREE.DirectionalLight(0x6fb7ff, 0.5);
  rimLight.position.set(-1, 0.4, -0.8);
  scene.add(rimLight);

  // Floor grid + axes, rebuilt on every fit so they match the domain scale
  let gridHelper: THREE.GridHelper | null = null;
  let axesHelper: THREE.AxesHelper | null = null;

  let homePosition = camera.position.clone();
  let homeTarget = controls.target.clone();

  const fitToDomain = (min: [number, number, number], max: [number, number, number]) => {
    const size = new THREE.Vector3(max[0] - min[0], max[1] - min[1], max[2] - min[2]);
    const center = new THREE.Vector3(
      (min[0] + max[0]) / 2,
      (min[1] + max[1]) / 2,
      (min[2] + max[2]) / 2,
    );
    const radius = Math.max(size.x, size.y, size.z, 1e-6);

    // Camera: three-quarter view from front-above-right, framed on the domain
    const dist = radius * 1.9;
    camera.position.set(
      center.x + dist * 0.85,
      center.y + dist * 0.6,
      center.z + dist * 0.95,
    );
    camera.near = radius * 0.01;
    camera.far = radius * 60;
    camera.updateProjectionMatrix();

    controls.target.copy(center);
    controls.minDistance = radius * 0.15;
    controls.maxDistance = radius * 12;
    controls.update();

    homePosition = camera.position.clone();
    homeTarget = controls.target.clone();

    // Rebuild grid under the domain
    if (gridHelper) {
      scene.remove(gridHelper);
      gridHelper.geometry.dispose();
      (gridHelper.material as THREE.Material).dispose();
    }
    if (axesHelper) {
      scene.remove(axesHelper);
      axesHelper.geometry.dispose();
      (axesHelper.material as THREE.Material).dispose();
    }

    const gridSize = Math.max(size.x, size.z) * 2.4;
    gridHelper = new THREE.GridHelper(gridSize, 24, 0x2a3b4d, 0x1a2733);
    gridHelper.position.set(center.x, min[1] - radius * 0.002, center.z);
    scene.add(gridHelper);

    axesHelper = new THREE.AxesHelper(radius * 0.16);
    axesHelper.position.set(min[0], min[1], min[2]);
    scene.add(axesHelper);
  };

  const resetView = () => {
    camera.position.copy(homePosition);
    controls.target.copy(homeTarget);
    controls.update();
  };

  // Handle resize (canvas is sized by CSS; track its client box)
  const handleResize = () => {
    const width = canvas.clientWidth;
    const height = canvas.clientHeight;
    if (width === 0 || height === 0) return;
    camera.aspect = width / height;
    camera.updateProjectionMatrix();
    renderer.setSize(width, height, false);
  };
  window.addEventListener('resize', handleResize);

  const animate = () => {
    controls.update();
    renderer.render(scene, camera);
  };

  return {
    scene,
    camera,
    renderer,
    controls,
    animate,
    fitToDomain,
    resetView,
  };
}
