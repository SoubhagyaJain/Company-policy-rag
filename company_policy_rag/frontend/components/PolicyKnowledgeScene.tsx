'use client';

import { useEffect, useRef } from 'react';
import { FileText, Link2, ShieldCheck } from 'lucide-react';

const GRAPH_NODES: Array<[number, number, number]> = [
  [-3.5, 1.35, 0.1],
  [-2.45, 0.35, 0.45],
  [-1.5, 1.72, -0.2],
  [-0.55, 0.68, 0.3],
  [0.58, 1.52, -0.35],
  [1.55, 0.35, 0.18],
  [2.85, 1.15, -0.18],
  [3.55, -0.2, 0.28],
  [2.25, -1.15, -0.1],
  [0.85, -0.65, 0.35],
  [-0.45, -1.25, -0.22],
  [-1.85, -0.65, 0.2],
  [-3.25, -1.35, -0.12],
];

const GRAPH_EDGES: Array<[number, number]> = [
  [0, 1], [0, 2], [1, 2], [1, 3], [1, 11], [2, 3], [2, 4], [3, 4],
  [3, 5], [3, 9], [3, 10], [4, 5], [4, 6], [5, 6], [5, 7], [5, 8],
  [5, 9], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [10, 12], [11, 12],
];

/**
 * A small, non-blocking Three.js knowledge graph. It is deliberately abstract:
 * documents become paper planes, retrieval becomes a linked graph, and the
 * central grounded answer is the warm node where those paths converge.
 */
export function PolicyKnowledgeScene() {
  const mountRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    let cancelled = false;
    let destroy = () => {};

    void import('three')
      .then((THREE) => {
        const mount = mountRef.current;
        if (!mount || cancelled) return;

        const prefersReducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)');
        const scene = new THREE.Scene();
        const camera = new THREE.PerspectiveCamera(38, 1, 0.1, 30);
        camera.position.set(0, 0.25, 9.5);

        const renderer = new THREE.WebGLRenderer({
          alpha: true,
          antialias: true,
          powerPreference: 'low-power',
        });
        renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 1.5));
        renderer.setClearColor(0x000000, 0);
        renderer.outputColorSpace = THREE.SRGBColorSpace;
        renderer.domElement.className = 'policy-graph__canvas';
        mount.appendChild(renderer.domElement);

        const graph = new THREE.Group();
        graph.rotation.x = -0.08;
        scene.add(graph);

        const edgePositions: number[] = [];
        for (const [from, to] of GRAPH_EDGES) {
          edgePositions.push(...GRAPH_NODES[from], ...GRAPH_NODES[to]);
        }
        const edgeGeometry = new THREE.BufferGeometry();
        edgeGeometry.setAttribute(
          'position',
          new THREE.Float32BufferAttribute(edgePositions, 3),
        );
        const edgeMaterial = new THREE.LineBasicMaterial({
          color: 0x6d887f,
          transparent: true,
          opacity: 0.28,
        });
        graph.add(new THREE.LineSegments(edgeGeometry, edgeMaterial));

        const nodeGeometry = new THREE.BufferGeometry();
        nodeGeometry.setAttribute(
          'position',
          new THREE.Float32BufferAttribute(GRAPH_NODES.flat(), 3),
        );
        const nodeMaterial = new THREE.PointsMaterial({
          color: 0xc75f38,
          size: 0.16,
          sizeAttenuation: true,
          transparent: true,
          opacity: 0.9,
        });
        const haloMaterial = new THREE.PointsMaterial({
          color: 0xdf916c,
          size: 0.42,
          sizeAttenuation: true,
          transparent: true,
          opacity: 0.14,
          depthWrite: false,
        });
        graph.add(new THREE.Points(nodeGeometry, haloMaterial));
        graph.add(new THREE.Points(nodeGeometry, nodeMaterial));

        const hubGeometry = new THREE.IcosahedronGeometry(0.27, 2);
        const hubMaterial = new THREE.MeshBasicMaterial({
          color: 0xc75f38,
          transparent: true,
          opacity: 0.92,
        });
        const hub = new THREE.Mesh(hubGeometry, hubMaterial);
        hub.position.set(...GRAPH_NODES[3]);
        graph.add(hub);

        const ringGeometry = new THREE.RingGeometry(0.42, 0.435, 48);
        const ringMaterial = new THREE.MeshBasicMaterial({
          color: 0xd98760,
          transparent: true,
          opacity: 0.42,
          side: THREE.DoubleSide,
          depthWrite: false,
        });
        const ring = new THREE.Mesh(ringGeometry, ringMaterial);
        ring.position.copy(hub.position);
        graph.add(ring);

        const paperGeometry = new THREE.PlaneGeometry(0.64, 0.42);
        const paperMaterial = new THREE.MeshBasicMaterial({
          color: 0xf0dfc4,
          transparent: true,
          opacity: 0.6,
          side: THREE.DoubleSide,
          depthWrite: false,
        });
        const paperNodes = [0, 6, 8, 12];
        const papers = paperNodes.map((nodeIndex, index) => {
          const paper = new THREE.Mesh(paperGeometry, paperMaterial);
          paper.position.set(...GRAPH_NODES[nodeIndex]);
          paper.position.z -= 0.08;
          paper.rotation.z = (index % 2 === 0 ? -1 : 1) * 0.08;
          graph.add(paper);
          return paper;
        });

        const pointer = { x: 0, y: 0 };
        const onPointerMove = (event: PointerEvent) => {
          pointer.x = event.clientX / window.innerWidth - 0.5;
          pointer.y = event.clientY / window.innerHeight - 0.5;
        };
        window.addEventListener('pointermove', onPointerMove, { passive: true });

        const resize = () => {
          const width = Math.max(mount.clientWidth, 1);
          const height = Math.max(mount.clientHeight, 1);
          camera.aspect = width / height;
          camera.updateProjectionMatrix();
          renderer.setSize(width, height, false);
          renderer.render(scene, camera);
        };
        const resizeObserver = new ResizeObserver(resize);
        resizeObserver.observe(mount);
        resize();

        const updateTheme = () => {
          const dark = document.documentElement.classList.contains('dark');
          edgeMaterial.color.setHex(dark ? 0x79a89a : 0x6d887f);
          edgeMaterial.opacity = dark ? 0.34 : 0.28;
          paperMaterial.color.setHex(dark ? 0x8f765c : 0xf0dfc4);
          paperMaterial.opacity = dark ? 0.34 : 0.6;
          nodeMaterial.color.setHex(dark ? 0xe68760 : 0xc75f38);
          hubMaterial.color.setHex(dark ? 0xec936b : 0xc75f38);
          renderer.render(scene, camera);
        };
        const themeObserver = new MutationObserver(updateTheme);
        themeObserver.observe(document.documentElement, {
          attributes: true,
          attributeFilter: ['class'],
        });
        updateTheme();

        let visible = !document.hidden;
        const onVisibilityChange = () => {
          visible = !document.hidden;
        };
        document.addEventListener('visibilitychange', onVisibilityChange);

        const clock = new THREE.Clock();
        const animate = () => {
          if (!visible) return;
          const elapsed = clock.getElapsedTime();

          graph.rotation.y += (pointer.x * 0.12 - graph.rotation.y) * 0.025;
          graph.rotation.x += (-0.08 - pointer.y * 0.08 - graph.rotation.x) * 0.025;
          hub.rotation.y = elapsed * 0.42;
          hub.rotation.x = elapsed * 0.23;
          const pulse = 1 + Math.sin(elapsed * 1.7) * 0.08;
          ring.scale.setScalar(pulse);
          ringMaterial.opacity = 0.34 + Math.sin(elapsed * 1.7) * 0.08;
          papers.forEach((paper, index) => {
            paper.position.y = GRAPH_NODES[paperNodes[index]][1]
              + Math.sin(elapsed * 0.7 + index * 1.2) * 0.055;
          });

          renderer.render(scene, camera);
        };

        if (prefersReducedMotion.matches) {
          renderer.render(scene, camera);
        } else {
          renderer.setAnimationLoop(animate);
        }

        destroy = () => {
          renderer.setAnimationLoop(null);
          window.removeEventListener('pointermove', onPointerMove);
          document.removeEventListener('visibilitychange', onVisibilityChange);
          resizeObserver.disconnect();
          themeObserver.disconnect();
          edgeGeometry.dispose();
          edgeMaterial.dispose();
          nodeGeometry.dispose();
          nodeMaterial.dispose();
          haloMaterial.dispose();
          hubGeometry.dispose();
          hubMaterial.dispose();
          ringGeometry.dispose();
          ringMaterial.dispose();
          paperGeometry.dispose();
          paperMaterial.dispose();
          renderer.dispose();
          renderer.domElement.remove();
        };
      })
      .catch(() => {
        // The CSS/SVG atlas remains visible when WebGL is unavailable.
      });

    return () => {
      cancelled = true;
      destroy();
    };
  }, []);

  return (
    <div className="policy-graph" aria-hidden="true">
      <div ref={mountRef} className="policy-graph__mount" />
      <div className="policy-graph__grid" />

      <div className="policy-graph__label policy-graph__label--documents">
        <FileText className="h-3 w-3" /> policy sources
      </div>
      <div className="policy-graph__label policy-graph__label--retrieval">
        <Link2 className="h-3 w-3" /> hybrid retrieval
      </div>
      <div className="policy-graph__label policy-graph__label--answer">
        <ShieldCheck className="h-3 w-3" /> grounded answer
      </div>

      <div className="policy-graph__caption">
        <span className="policy-graph__live-dot" />
        Knowledge map · live
      </div>
    </div>
  );
}
