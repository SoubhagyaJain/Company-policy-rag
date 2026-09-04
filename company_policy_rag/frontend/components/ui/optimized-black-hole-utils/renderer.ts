import * as THREE from "three";

export interface RendererOptions {
  canvas: HTMLCanvasElement;
}

export interface RendererInstance {
  ready: Promise<void>;
  dispose: () => void;
}

export function createRenderer(options: RendererOptions): RendererInstance {
  const { canvas } = options;
  let disposed = false;
  let animationFrameId: number | null = null;
  let renderer: THREE.WebGLRenderer | null = null;
  
  let readyResolve: () => void;
  const readyPromise = new Promise<void>((resolve) => {
    readyResolve = resolve;
  });

  const dispose = () => {
    if (disposed) return;
    disposed = true;
    if (animationFrameId !== null) {
      cancelAnimationFrame(animationFrameId);
    }
    // Cleanups are handled in the try-catch block scope to ensure closures capture them
  };

  try {
    renderer = new THREE.WebGLRenderer({
      canvas,
      alpha: true, // Use transparent background to mix with bg-black
      antialias: true,
      powerPreference: "high-performance",
    });
  } catch (e) {
    console.warn("WebGL initialization failed. Black hole will fallback to static background.", e);
    // Resolve immediately so the page can still show the fallback
    setTimeout(() => { if (!disposed) readyResolve(); }, 0);
    return { ready: readyPromise, dispose };
  }

  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  renderer.setPixelRatio(dpr);
  renderer.setSize(canvas.clientWidth, canvas.clientHeight, false);

  const scene = new THREE.Scene();

  const camera = new THREE.PerspectiveCamera(
    45,
    canvas.clientWidth / canvas.clientHeight,
    0.1,
    100
  );
  camera.position.z = 8;
  camera.position.y = 2.5;
  camera.lookAt(0, 0, 0);

  const prefersReducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  const speedScale = prefersReducedMotion ? 0.05 : 1.0;

  // --- 1. Accretion Disk (Shader) ---
  const diskGeometry = new THREE.PlaneGeometry(12, 12, 64, 64);
  const diskMaterial = new THREE.ShaderMaterial({
    uniforms: {
      uTime: { value: 0 },
    },
    vertexShader: `
      varying vec2 vUv;
      varying vec3 vPos;
      void main() {
        vUv = uv;
        vPos = position;
        gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
      }
    `,
    fragmentShader: `
      uniform float uTime;
      varying vec2 vUv;
      varying vec3 vPos;

      // Hash function for noise
      float hash(vec2 p) { return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453123); }
      float noise(vec2 p) {
        vec2 i = floor(p), f = fract(p);
        vec2 u = f*f*(3.0-2.0*f);
        return mix(mix(hash(i + vec2(0.0,0.0)), hash(i + vec2(1.0,0.0)), u.x),
                   mix(hash(i + vec2(0.0,1.0)), hash(i + vec2(1.0,1.0)), u.x), u.y);
      }

      // FBM
      float fbm(vec2 p) {
        float f = 0.0, amp = 0.5;
        for(int i=0; i<5; i++){
          f += amp * noise(p);
          p *= 2.0;
          amp *= 0.5;
        }
        return f;
      }

      void main() {
        vec2 p = vUv * 2.0 - 1.0;
        float r = length(p);
        float angle = atan(p.y, p.x);

        // Event Horizon (completely dark center)
        float horizon = smoothstep(0.18, 0.22, r);

        // Accretion disk falloff
        float disk = smoothstep(0.20, 0.35, r) * smoothstep(0.95, 0.4, r);

        // Swirling noise effect
        vec2 swirlUv = vec2(r * 4.0, angle * 2.0 + uTime * 0.8);
        float n = fbm(swirlUv - uTime * 0.2);
        
        // Doppler beaming (brighter on one side)
        float doppler = 1.0 + 0.6 * sin(angle);

        float glow = disk * n * doppler * 2.5;

        // Color Palette (Interstellar inspired)
        vec3 colCenter = vec3(1.0, 0.8, 0.5); // Hot core
        vec3 colMid = vec3(1.0, 0.4, 0.05);   // Orange accretion
        vec3 colEdge = vec3(0.05, 0.2, 0.8);  // Blue outer edge

        vec3 finalColor = mix(colMid, colCenter, smoothstep(0.6, 0.2, r));
        finalColor = mix(colEdge, finalColor, smoothstep(0.85, 0.4, r));
        
        finalColor *= glow;
        finalColor *= horizon;

        gl_FragColor = vec4(finalColor, glow * horizon);
      }
    `,
    transparent: true,
    depthWrite: false,
    blending: THREE.AdditiveBlending,
    side: THREE.DoubleSide
  });

  const diskMesh = new THREE.Mesh(diskGeometry, diskMaterial);
  diskMesh.rotation.x = -Math.PI / 2;
  scene.add(diskMesh);

  // --- 2. Gravitational Lensing (Background Distortion simulated by Halo) ---
  const haloGeometry = new THREE.PlaneGeometry(10, 10);
  const haloMaterial = new THREE.ShaderMaterial({
    uniforms: { uTime: { value: 0 } },
    vertexShader: `
      varying vec2 vUv;
      void main() {
        vUv = uv;
        // Keep halo always facing camera
        gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
      }
    `,
    fragmentShader: `
      uniform float uTime;
      varying vec2 vUv;
      
      void main() {
        vec2 p = vUv * 2.0 - 1.0;
        float r = length(p);
        
        // Einstein ring effect
        float ring = smoothstep(0.23, 0.25, r) * smoothstep(0.35, 0.25, r);
        float glow = smoothstep(0.2, 0.8, r) * smoothstep(1.0, 0.4, r);
        
        vec3 color = vec3(1.0, 0.8, 0.6) * ring * 1.5;
        color += vec3(0.2, 0.4, 0.9) * glow * 0.3;
        
        gl_FragColor = vec4(color, max(ring, glow * 0.3));
      }
    `,
    transparent: true,
    depthWrite: false,
    blending: THREE.AdditiveBlending
  });
  const haloMesh = new THREE.Mesh(haloGeometry, haloMaterial);
  scene.add(haloMesh);

  // --- 3. Starfield / Dust ---
  const particleCount = prefersReducedMotion ? 300 : 1500;
  const particleGeo = new THREE.BufferGeometry();
  const pPositions = new Float32Array(particleCount * 3);
  const pColors = new Float32Array(particleCount * 3);
  const pSizes = new Float32Array(particleCount);

  for (let i = 0; i < particleCount; i++) {
    // Random spherical distribution, but avoid center
    let r = 2.0 + Math.random() * 20.0;
    let theta = Math.random() * Math.PI * 2;
    let phi = Math.acos(Math.random() * 2 - 1);
    
    pPositions[i*3] = r * Math.sin(phi) * Math.cos(theta);
    pPositions[i*3+1] = r * Math.sin(phi) * Math.sin(theta);
    pPositions[i*3+2] = r * Math.cos(phi);

    // Warm colors to match disk, some blue
    const isBlue = Math.random() > 0.85;
    pColors[i*3] = isBlue ? 0.3 : 1.0;
    pColors[i*3+1] = isBlue ? 0.6 : (0.5 + Math.random() * 0.4);
    pColors[i*3+2] = isBlue ? 1.0 : (0.2 + Math.random() * 0.2);
    
    pSizes[i] = Math.random();
  }

  particleGeo.setAttribute('position', new THREE.BufferAttribute(pPositions, 3));
  particleGeo.setAttribute('color', new THREE.BufferAttribute(pColors, 3));
  particleGeo.setAttribute('aSize', new THREE.BufferAttribute(pSizes, 1));

  const particleMat = new THREE.ShaderMaterial({
    uniforms: { uTime: { value: 0 } },
    vertexShader: `
      uniform float uTime;
      attribute float aSize;
      varying vec3 vColor;
      void main() {
        vColor = color;
        vec4 mvPosition = modelViewMatrix * vec4(position, 1.0);
        gl_PointSize = aSize * (30.0 / -mvPosition.z);
        gl_Position = projectionMatrix * mvPosition;
      }
    `,
    fragmentShader: `
      varying vec3 vColor;
      void main() {
        vec2 p = gl_PointCoord * 2.0 - 1.0;
        float r = length(p);
        if (r > 1.0) discard;
        float alpha = smoothstep(1.0, 0.0, r);
        gl_FragColor = vec4(vColor, alpha * 0.6);
      }
    `,
    transparent: true,
    depthWrite: false,
    blending: THREE.AdditiveBlending,
    vertexColors: true
  });

  const particleMesh = new THREE.Points(particleGeo, particleMat);
  scene.add(particleMesh);

  // Resize Handling
  let resizeObserver: ResizeObserver | null = null;
  if (typeof ResizeObserver !== 'undefined') {
    resizeObserver = new ResizeObserver(() => {
      if (disposed || !renderer) return;
      const w = canvas.clientWidth;
      const h = canvas.clientHeight;
      if (w === 0 || h === 0) return;
      
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
      renderer.setSize(w, h, false);
    });
    resizeObserver.observe(canvas);
  }

  // Animation Loop
  const clock = new THREE.Clock();
  
  const animate = () => {
    if (disposed || !renderer) return;
    animationFrameId = requestAnimationFrame(animate);
    
    const delta = clock.getDelta();
    const elapsed = clock.getElapsedTime() * speedScale;
    
    diskMaterial.uniforms.uTime.value = elapsed;
    haloMaterial.uniforms.uTime.value = elapsed;
    
    // Slow rotation
    particleMesh.rotation.y = elapsed * 0.05;
    particleMesh.rotation.z = elapsed * 0.02;
    
    // Slight camera drift for cinematic feel
    if (!prefersReducedMotion) {
      camera.position.x = Math.sin(elapsed * 0.1) * 0.5;
      camera.position.y = 2.5 + Math.cos(elapsed * 0.15) * 0.3;
      camera.lookAt(0, 0, 0);
    }

    renderer.render(scene, camera);
  };
  
  animate();
  
  // Resolve ready on next frame
  requestAnimationFrame(() => {
    if (!disposed) readyResolve();
  });

  // Re-define dispose to properly clean up WebGL
  const fullDispose = () => {
    if (disposed) return;
    disposed = true;
    
    if (animationFrameId !== null) cancelAnimationFrame(animationFrameId);
    if (resizeObserver) resizeObserver.disconnect();
    
    diskGeometry.dispose();
    diskMaterial.dispose();
    haloGeometry.dispose();
    haloMaterial.dispose();
    particleGeo.dispose();
    particleMat.dispose();
    
    if (renderer) renderer.dispose();
  };

  return {
    ready: readyPromise,
    dispose: fullDispose
  };
}
