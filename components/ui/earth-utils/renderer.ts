import * as THREE from "three";

interface RendererOptions {
  canvas: HTMLCanvasElement;
}

interface RendererInstance {
  ready: Promise<void>;
  dispose: () => void;
}

export function createRenderer({ canvas }: RendererOptions): RendererInstance {
  const scene = new THREE.Scene();
  
  const camera = new THREE.PerspectiveCamera(
    45,
    canvas.clientWidth / canvas.clientHeight,
    0.1,
    1000
  );
  camera.position.z = 3;

  const renderer = new THREE.WebGLRenderer({
    canvas,
    alpha: true,
    antialias: true,
  });
  renderer.setSize(canvas.clientWidth, canvas.clientHeight);
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));

  const geometry = new THREE.SphereGeometry(1, 64, 64);
  
  // Create a TextureLoader
  const textureLoader = new THREE.TextureLoader();
  
  // Publicly available Earth textures (cors-friendly)
  const EARTH_TEXTURE_URL = "https://unpkg.com/three-globe/example/img/earth-blue-marble.jpg";
  const EARTH_BUMP_URL = "https://unpkg.com/three-globe/example/img/earth-topology.png";
  const EARTH_WATER_URL = "https://unpkg.com/three-globe/example/img/earth-water.png";

  const material = new THREE.MeshPhongMaterial({
    color: 0xffffff,
    specular: 0x333333,
    shininess: 15,
  });
  
  const earthMesh = new THREE.Mesh(geometry, material);
  scene.add(earthMesh);

  // Lighting
  const ambientLight = new THREE.AmbientLight(0xffffff, 0.2);
  scene.add(ambientLight);

  const directionalLight = new THREE.DirectionalLight(0xffffff, 1.5);
  directionalLight.position.set(5, 3, 5);
  scene.add(directionalLight);

  // Resize handler
  const handleResize = () => {
    if (!canvas.parentElement) return;
    const width = canvas.parentElement.clientWidth;
    const height = canvas.parentElement.clientHeight;
    
    camera.aspect = width / height;
    camera.updateProjectionMatrix();
    
    renderer.setSize(width, height);
  };

  window.addEventListener("resize", handleResize);
  handleResize();

  let animationFrameId: number;
  
  const animate = () => {
    animationFrameId = requestAnimationFrame(animate);
    
    // Smooth rotation
    earthMesh.rotation.y += 0.001;
    earthMesh.rotation.x += 0.0002;
    
    renderer.render(scene, camera);
  };
  
  animate();

  // Load textures and resolve the ready promise once the main texture loads
  const readyPromise = new Promise<void>((resolve, reject) => {
    textureLoader.load(
      EARTH_TEXTURE_URL,
      (texture) => {
        texture.colorSpace = THREE.SRGBColorSpace;
        material.map = texture;
        material.needsUpdate = true;
        resolve(); // Component is ready when the diffuse map loads
      },
      undefined,
      (err) => reject(err)
    );

    textureLoader.load(EARTH_BUMP_URL, (bumpMap) => {
      material.bumpMap = bumpMap;
      material.bumpScale = 0.015;
      material.needsUpdate = true;
    });

    textureLoader.load(EARTH_WATER_URL, (specularMap) => {
      material.specularMap = specularMap;
      material.needsUpdate = true;
    });
  });

  return {
    ready: readyPromise,
    dispose: () => {
      window.removeEventListener("resize", handleResize);
      cancelAnimationFrame(animationFrameId);
      
      geometry.dispose();
      material.dispose();
      
      if (material.map) material.map.dispose();
      if (material.bumpMap) material.bumpMap.dispose();
      if (material.specularMap) material.specularMap.dispose();
      
      renderer.dispose();
    },
  };
}
