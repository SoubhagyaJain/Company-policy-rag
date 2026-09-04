import * as THREE from 'three';

export interface RendererOptions {
  canvas: HTMLCanvasElement;
  autoRotate?: boolean;
  autoRotateSpeed?: number;
  atmosphereColor?: string;
  glowColor?: string;
  interactive?: boolean;
}

export interface EarthRenderer {
  ready: Promise<void>;
  dispose: () => void;
  setTheme?: (isDark: boolean) => void;
}

/**
 * Procedural Earth texture generator that creates realistic continents,
 * oceans, latitude/longitude grid, and glowing tech/city points.
 * Ensures zero-dependency offline rendering with crisp high-DPI visuals.
 */
function createProceduralEarthTexture(isDark: boolean = true): THREE.CanvasTexture {
  const width = 2048;
  const height = 1024;
  const canvas = document.createElement('canvas');
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext('2d');

  if (!ctx) {
    const fallback = new THREE.CanvasTexture(canvas);
    return fallback;
  }

  // 1. Ocean Base
  const oceanGrad = ctx.createLinearGradient(0, 0, 0, height);
  if (isDark) {
    oceanGrad.addColorStop(0, '#040b14');
    oceanGrad.addColorStop(0.5, '#081426');
    oceanGrad.addColorStop(1, '#030810');
  } else {
    oceanGrad.addColorStop(0, '#102a45');
    oceanGrad.addColorStop(0.5, '#16385c');
    oceanGrad.addColorStop(1, '#0e233a');
  }
  ctx.fillStyle = oceanGrad;
  ctx.fillRect(0, 0, width, height);

  // 2. Latitude and Longitude Grid
  ctx.strokeStyle = isDark ? 'rgba(56, 189, 248, 0.08)' : 'rgba(255, 255, 255, 0.08)';
  ctx.lineWidth = 1;

  for (let lat = -80; lat <= 80; lat += 20) {
    const y = ((90 - lat) / 180) * height;
    ctx.beginPath();
    ctx.moveTo(0, y);
    ctx.lineTo(width, y);
    ctx.stroke();
  }

  for (let lon = -180; lon <= 180; lon += 30) {
    const x = ((lon + 180) / 360) * width;
    ctx.beginPath();
    ctx.moveTo(x, 0);
    ctx.lineTo(x, height);
    ctx.stroke();
  }

  // 3. Procedural Landmass Contours & Continents (Simplified Geo-Shapes)
  const landColor = isDark ? '#142840' : '#1e3a5f';
  ctx.fillStyle = landColor;
  ctx.strokeStyle = isDark ? '#38bdf8' : '#60a5fa';
  ctx.lineWidth = 1.5;

  // Helper to convert lat/lon to canvas coordinates
  const toX = (lon: number) => ((lon + 180) / 360) * width;
  const toY = (lat: number) => ((90 - lat) / 180) * height;

  // Major continent polygons (approximations for procedural rendering)
  const continents: Array<Array<[number, number]>> = [
    // North America
    [[-165, 68], [-140, 70], [-100, 72], [-65, 60], [-55, 48], [-70, 42], [-80, 25], [-98, 18], [-105, 22], [-122, 36], [-125, 48], [-140, 58], [-165, 60]],
    // South America
    [[-78, 10], [-50, -5], [-35, -7], [-38, -22], [-55, -38], [-70, -54], [-75, -45], [-80, -10], [-80, 5]],
    // Europe
    [[-10, 36], [0, 44], [12, 44], [25, 36], [32, 40], [28, 58], [15, 62], [8, 54], [-5, 48]],
    // Africa
    [[-15, 32], [10, 37], [32, 32], [50, 12], [42, -5], [30, -32], [18, -34], [10, -15], [-17, 12], [-18, 25]],
    // Asia & Siberia
    [[35, 40], [60, 42], [75, 28], [80, 12], [105, 12], [120, 24], [122, 38], [140, 50], [170, 65], [100, 75], [50, 70], [35, 55]],
    // Australia
    [[115, -22], [130, -12], [152, -22], [150, -36], [135, -38], [115, -32]],
  ];

  continents.forEach((poly) => {
    ctx.beginPath();
    poly.forEach(([lon, lat], i) => {
      const x = toX(lon);
      const y = toY(lat);
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.closePath();
    ctx.fill();
    ctx.stroke();
  });

  // 4. Dot matrix / Digital grid overlay on continents for high-tech look
  ctx.fillStyle = isDark ? 'rgba(56, 189, 248, 0.45)' : 'rgba(96, 165, 250, 0.4)';
  for (let lat = -60; lat <= 70; lat += 3) {
    for (let lon = -180; lon <= 180; lon += 3) {
      const px = toX(lon);
      const py = toY(lat);
      let inside = false;
      for (const poly of continents) {
        let isInside = false;
        for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {
          const xi = toX(poly[i][0]), yi = toY(poly[i][1]);
          const xj = toX(poly[j][0]), yj = toY(poly[j][1]);
          const intersect = ((yi > py) !== (yj > py)) && (px < (xj - xi) * (py - yi) / (yj - yi) + xi);
          if (intersect) isInside = !isInside;
        }
        if (isInside) {
          inside = true;
          break;
        }
      }

      if (inside) {
        ctx.beginPath();
        ctx.arc(px, py, 1.2, 0, Math.PI * 2);
        ctx.fill();
      }
    }
  }

  // 5. Glowing City Hubs & Knowledge Nodes
  const cityLightColor = '#f59e0b';
  const cityHubs: Array<[number, number, number]> = [
    [-122.4, 37.7, 4], // San Francisco
    [-74.0, 40.7, 5],   // New York
    [-0.1, 51.5, 4.5],  // London
    [2.3, 48.8, 3.5],   // Paris
    [77.2, 28.6, 4.5],  // New Delhi
    [139.7, 35.6, 5],   // Tokyo
    [103.8, 1.3, 3.8],  // Singapore
    [151.2, -33.8, 3.5],// Sydney
    [-46.6, -23.5, 3.2],// Sao Paulo
  ];

  cityHubs.forEach(([lon, lat, size]) => {
    const cx = toX(lon);
    const cy = toY(lat);

    const radGrad = ctx.createRadialGradient(cx, cy, 0, cx, cy, size * 5);
    radGrad.addColorStop(0, '#fef08a');
    radGrad.addColorStop(0.3, cityLightColor);
    radGrad.addColorStop(0.7, 'rgba(245, 158, 11, 0.25)');
    radGrad.addColorStop(1, 'rgba(245, 158, 11, 0)');

    ctx.fillStyle = radGrad;
    ctx.beginPath();
    ctx.arc(cx, cy, size * 5, 0, Math.PI * 2);
    ctx.fill();

    ctx.fillStyle = '#ffffff';
    ctx.beginPath();
    ctx.arc(cx, cy, size * 0.8, 0, Math.PI * 2);
    ctx.fill();
  });

  const texture = new THREE.CanvasTexture(canvas);
  texture.wrapS = THREE.RepeatWrapping;
  texture.wrapT = THREE.ClampToEdgeWrapping;
  texture.colorSpace = THREE.SRGBColorSpace;
  texture.needsUpdate = true;
  return texture;
}

/**
 * Creates 3D curved data arcs between major global nodes.
 */
function createNetworkArcs(radius: number): THREE.Group {
  const group = new THREE.Group();
  const hubs: Array<[number, number]> = [
    [-122.4, 37.7],
    [-74.0, 40.7],
    [-0.1, 51.5],
    [77.2, 28.6],
    [139.7, 35.6],
    [103.8, 1.3],
    [151.2, -33.8],
  ];

  const latLonToVector3 = (lat: number, lon: number, r: number): THREE.Vector3 => {
    const phi = (90 - lat) * (Math.PI / 180);
    const theta = (lon + 180) * (Math.PI / 180);
    return new THREE.Vector3(
      -r * Math.sin(phi) * Math.cos(theta),
      r * Math.cos(phi),
      r * Math.sin(phi) * Math.sin(theta)
    );
  };

  const connections: Array<[number, number]> = [
    [0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [0, 4], [1, 5]
  ];

  connections.forEach(([i1, i2]) => {
    const p1 = latLonToVector3(hubs[i1][1], hubs[i1][0], radius);
    const p2 = latLonToVector3(hubs[i2][1], hubs[i2][0], radius);

    // Compute midpoint arched away from the earth center
    const mid = p1.clone().add(p2).multiplyScalar(0.5);
    const dist = p1.distanceTo(p2);
    mid.normalize().multiplyScalar(radius + Math.min(dist * 0.35, 1.2));

    const curve = new THREE.QuadraticBezierCurve3(p1, mid, p2);
    const points = curve.getPoints(40);
    const geometry = new THREE.BufferGeometry().setFromPoints(points);

    const material = new THREE.LineBasicMaterial({
      color: 0x38bdf8,
      transparent: true,
      opacity: 0.55,
      blending: THREE.AdditiveBlending,
    });

    const line = new THREE.Line(geometry, material);
    group.add(line);
  });

  return group;
}

/**
 * Creates atmospheric glow surrounding the sphere with custom Fresnel shader.
 */
function createAtmosphereMesh(radius: number): THREE.Mesh {
  const geometry = new THREE.SphereGeometry(radius * 1.18, 48, 48);

  const vertexShader = `
    varying vec3 vNormal;
    varying vec3 vEyeVector;
    void main() {
      vNormal = normalize(normalMatrix * normal);
      vec4 worldPos = modelViewMatrix * vec4(position, 1.0);
      vEyeVector = normalize(-worldPos.xyz);
      gl_Position = projectionMatrix * worldPos;
    }
  `;

  const fragmentShader = `
    varying vec3 vNormal;
    varying vec3 vEyeVector;
    uniform vec3 uColor;
    void main() {
      float dotNV = dot(vNormal, vEyeVector);
      float intensity = pow(1.0 - max(dotNV, 0.0), 3.2) * 1.25;
      gl_FragColor = vec4(uColor, intensity);
    }
  `;

  const material = new THREE.ShaderMaterial({
    vertexShader,
    fragmentShader,
    uniforms: {
      uColor: { value: new THREE.Color(0x38bdf8) },
    },
    blending: THREE.AdditiveBlending,
    side: THREE.BackSide,
    transparent: true,
    depthWrite: false,
  });

  return new THREE.Mesh(geometry, material);
}

/**
 * Creates deep cosmic starfield.
 */
function createStarfield(count = 1200): THREE.Points {
  const positions = new Float32Array(count * 3);
  const colors = new Float32Array(count * 3);

  const colorOptions = [
    new THREE.Color(0xffffff),
    new THREE.Color(0xbae6fd),
    new THREE.Color(0xfde047),
    new THREE.Color(0xfbcfe8),
  ];

  for (let i = 0; i < count; i++) {
    const r = 25 + Math.random() * 50;
    const theta = Math.random() * Math.PI * 2;
    const phi = Math.acos(Math.random() * 2 - 1);

    positions[i * 3] = r * Math.sin(phi) * Math.cos(theta);
    positions[i * 3 + 1] = r * Math.sin(phi) * Math.sin(theta);
    positions[i * 3 + 2] = r * Math.cos(phi);

    const c = colorOptions[Math.floor(Math.random() * colorOptions.length)];
    colors[i * 3] = c.r;
    colors[i * 3 + 1] = c.g;
    colors[i * 3 + 2] = c.b;
  }

  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));

  const material = new THREE.PointsMaterial({
    size: 0.6,
    vertexColors: true,
    transparent: true,
    opacity: 0.75,
    sizeAttenuation: true,
  });

  return new THREE.Points(geometry, material);
}

/**
 * Main factory function to initialize the Three.js Earth renderer on the canvas.
 */
export function createRenderer(options: RendererOptions): EarthRenderer {
  const {
    canvas,
    autoRotate = true,
    autoRotateSpeed = 0.4,
    interactive = true,
  } = options;

  let isDisposed = false;
  let animFrameId: number | null = null;
  const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)');

  // 1. Scene, Camera, Renderer
  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(42, 1, 0.1, 1000);
  camera.position.set(0, 0, 8.5);

  const renderer = new THREE.WebGLRenderer({
    canvas,
    alpha: true,
    antialias: true,
    powerPreference: 'low-power',
  });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 1.5));
  renderer.outputColorSpace = THREE.SRGBColorSpace;

  // 2. Earth Globe Group
  const earthRadius = 2.4;
  const earthGroup = new THREE.Group();
  earthGroup.rotation.x = 0.22; // Earth axial tilt (~23 deg)
  scene.add(earthGroup);

  // Earth Sphere
  const sphereGeometry = new THREE.SphereGeometry(earthRadius, 64, 64);
  const earthTexture = createProceduralEarthTexture(true);

  const earthMaterial = new THREE.MeshStandardMaterial({
    map: earthTexture,
    roughness: 0.6,
    metalness: 0.1,
    emissive: new THREE.Color(0x0c2545),
    emissiveIntensity: 0.5,
  });

  const earthMesh = new THREE.Mesh(sphereGeometry, earthMaterial);
  earthGroup.add(earthMesh);

  // Atmospheric Fresnel Glow
  const atmosphereMesh = createAtmosphereMesh(earthRadius);
  earthGroup.add(atmosphereMesh);

  // Network Knowledge Arcs
  const networkArcs = createNetworkArcs(earthRadius);
  earthGroup.add(networkArcs);

  // Starfield
  const starfield = createStarfield(900);
  scene.add(starfield);

  // 3. Lighting
  const ambientLight = new THREE.AmbientLight(0xffffff, 1.2);
  scene.add(ambientLight);

  const sunLight = new THREE.DirectionalLight(0xfff7ed, 2.8);
  sunLight.position.set(6, 4, 7);
  scene.add(sunLight);

  const blueBackLight = new THREE.DirectionalLight(0x38bdf8, 1.6);
  blueBackLight.position.set(-6, -3, -5);
  scene.add(blueBackLight);

  // 4. Interactive Drag & Momentum
  let isDragging = false;
  let prevMousePos = { x: 0, y: 0 };
  let targetRotationY = 0;
  let targetRotationX = 0.22;
  let rotationVelocity = { x: 0, y: 0 };

  const onPointerDown = (e: PointerEvent) => {
    if (!interactive) return;
    isDragging = true;
    prevMousePos = { x: e.clientX, y: e.clientY };
  };

  const onPointerMove = (e: PointerEvent) => {
    if (!interactive || !isDragging) return;
    const deltaX = e.clientX - prevMousePos.x;
    const deltaY = e.clientY - prevMousePos.y;

    rotationVelocity.y = deltaX * 0.004;
    rotationVelocity.x = deltaY * 0.004;

    targetRotationY += rotationVelocity.y;
    targetRotationX = Math.max(-1.1, Math.min(1.1, targetRotationX + rotationVelocity.x));

    if (reducedMotion.matches) {
      earthGroup.rotation.set(targetRotationX, targetRotationY, earthGroup.rotation.z);
      renderer.render(scene, camera);
    }

    prevMousePos = { x: e.clientX, y: e.clientY };
  };

  const onPointerUp = () => {
    isDragging = false;
  };

  if (interactive) {
    canvas.addEventListener('pointerdown', onPointerDown);
    window.addEventListener('pointermove', onPointerMove);
    window.addEventListener('pointerup', onPointerUp);
  }

  // 5. Handle Resize
  const handleResize = () => {
    if (!canvas || isDisposed) return;
    const width = canvas.clientWidth || window.innerWidth;
    const height = canvas.clientHeight || window.innerHeight;

    if (width > 0 && height > 0) {
      camera.aspect = width / height;
      camera.updateProjectionMatrix();
      renderer.setSize(width, height, false);
      renderer.render(scene, camera);
    }
  };

  const resizeObserver = new ResizeObserver(handleResize);
  resizeObserver.observe(canvas);
  handleResize();

  // 6. Animation Loop
  let previousFrameTime: number | null = null;

  const requestNextFrame = () => {
    if (
      !isDisposed
      && animFrameId === null
      && !document.hidden
      && !reducedMotion.matches
    ) {
      animFrameId = requestAnimationFrame(animate);
    }
  };

  const animate = (frameTime: number) => {
    if (isDisposed) return;
    animFrameId = null;
    const delta = previousFrameTime === null
      ? 0
      : Math.min((frameTime - previousFrameTime) / 1000, 0.1);
    previousFrameTime = frameTime;

    if (autoRotate && !isDragging) {
      targetRotationY += delta * autoRotateSpeed * 0.35;
    }

    // Inertial damping
    earthGroup.rotation.y += (targetRotationY - earthGroup.rotation.y) * 0.08;
    earthGroup.rotation.x += (targetRotationX - earthGroup.rotation.x) * 0.08;

    // Slow rotation of starfield for subtle parallax
    starfield.rotation.y += delta * 0.02;

    renderer.render(scene, camera);
    requestNextFrame();
  };

  const syncAnimation = () => {
    if (document.hidden || reducedMotion.matches) {
      if (animFrameId !== null) {
        cancelAnimationFrame(animFrameId);
        animFrameId = null;
      }
      previousFrameTime = null;
      renderer.render(scene, camera);
      return;
    }
    requestNextFrame();
  };

  document.addEventListener('visibilitychange', syncAnimation);
  reducedMotion.addEventListener('change', syncAnimation);
  syncAnimation();

  const readyPromise = Promise.resolve();

  // 7. Cleanup & Disposal
  const dispose = () => {
    isDisposed = true;
    if (animFrameId !== null) {
      cancelAnimationFrame(animFrameId);
    }

    resizeObserver.disconnect();
    document.removeEventListener('visibilitychange', syncAnimation);
    reducedMotion.removeEventListener('change', syncAnimation);

    if (interactive) {
      canvas.removeEventListener('pointerdown', onPointerDown);
      window.removeEventListener('pointermove', onPointerMove);
      window.removeEventListener('pointerup', onPointerUp);
    }

    // Dispose Geometries and Materials
    sphereGeometry.dispose();
    earthTexture.dispose();
    earthMaterial.dispose();
    (atmosphereMesh.material as THREE.Material).dispose();
    atmosphereMesh.geometry.dispose();
    starfield.geometry.dispose();
    (starfield.material as THREE.Material).dispose();

    networkArcs.traverse((child) => {
      if (child instanceof THREE.Line) {
        child.geometry.dispose();
        (child.material as THREE.Material).dispose();
      }
    });

    renderer.dispose();
  };

  return {
    ready: readyPromise,
    dispose,
  };
}
