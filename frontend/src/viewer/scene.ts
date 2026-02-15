// T033: Three.js scene setup with camera, lighting, and controls

import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';

export interface SceneComponents {
  scene: THREE.Scene;
  camera: THREE.PerspectiveCamera;
  renderer: THREE.WebGLRenderer;
  controls: OrbitControls;
  animate: () => void;
}

/**
 * Create and configure the Three.js scene for SPH visualization
 *
 * Sets up:
 * - PerspectiveCamera positioned for 1cm scale domains
 * - OrbitControls for interactive viewing
 * - Ambient + directional lighting
 * - Grid helper for spatial reference
 */
export function createScene(canvas: HTMLCanvasElement): SceneComponents {
  // Create scene
  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0x1a1a1a);

  // Create camera (FOV 60, positioned for 1cm scale)
  const aspect = canvas.clientWidth / canvas.clientHeight;
  const camera = new THREE.PerspectiveCamera(60, aspect, 0.001, 100);
  camera.position.set(0.02, 0.02, 0.02);
  camera.lookAt(0, 0, 0);

  // Create renderer
  const renderer = new THREE.WebGLRenderer({
    canvas,
    antialias: true,
  });
  renderer.setSize(canvas.clientWidth, canvas.clientHeight);
  renderer.setPixelRatio(window.devicePixelRatio);

  // Create orbit controls
  const controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.dampingFactor = 0.05;
  controls.target.set(0, 0, 0);
  controls.update();

  // Add lighting
  // Ambient light for base illumination
  const ambientLight = new THREE.AmbientLight(0xffffff, 0.4);
  scene.add(ambientLight);

  // Directional light from above-right
  const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
  directionalLight.position.set(1, 1, 0.5);
  scene.add(directionalLight);

  // Add grid helper (1cm scale, 10 divisions)
  const gridHelper = new THREE.GridHelper(0.01, 10, 0x444444, 0x222222);
  scene.add(gridHelper);

  // Add axes helper for orientation
  const axesHelper = new THREE.AxesHelper(0.005);
  scene.add(axesHelper);

  // Handle window resize
  const handleResize = () => {
    const width = canvas.clientWidth;
    const height = canvas.clientHeight;
    camera.aspect = width / height;
    camera.updateProjectionMatrix();
    renderer.setSize(width, height);
  };
  window.addEventListener('resize', handleResize);

  // Animation loop
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
  };
}
