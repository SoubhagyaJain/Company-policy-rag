'use client';

/**
 * LibraryHero — an animated canvas background for the Library tab.
 * Draws a floating document-lattice with gentle parallax, glowing nodes
 * at intersections (representing indexed documents), and drifting particles.
 * Uses the same smoothDamp approach as SpaceHero for buttery motion.
 */

import { useEffect, useRef } from 'react';

interface Node {
  x: number;
  y: number;
  vx: number;
  vy: number;
  r: number;
  phase: number;
  kind: number; // 0 = small, 1 = medium, 2 = large
}

export interface LibraryHeroProps {
  light?: boolean;
  className?: string;
}

export function LibraryHero({ light = false, className }: LibraryHeroProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const lightRef = useRef(light);

  useEffect(() => { lightRef.current = light; }, [light]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const reduced = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    const ctx = canvas.getContext('2d', { alpha: false });
    if (!ctx) return;

    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    let w = 0, h = 0;
    let t = 0;
    let lightMix = light ? 1 : 0;
    let raf = 0;
    let last = performance.now();

    const mouse = { x: 0.5, y: 0.5, tx: 0.5, ty: 0.5 };

    const smoothDamp = (cur: number, tgt: number, hl: number, dt: number) =>
      cur + (tgt - cur) * (1 - Math.pow(2, -dt / hl));

    // Generate nodes
    const NODE_COUNT = reduced ? 30 : 65;
    const nodes: Node[] = [];
    for (let i = 0; i < NODE_COUNT; i++) {
      nodes.push({
        x: Math.random(),
        y: Math.random(),
        vx: (Math.random() - 0.5) * 0.012,
        vy: (Math.random() - 0.5) * 0.012,
        r: 1.5 + Math.random() * 3,
        phase: Math.random() * Math.PI * 2,
        kind: Math.random() > 0.92 ? 2 : Math.random() > 0.7 ? 1 : 0,
      });
    }

    // Connection distance threshold
    const CONN_DIST = 0.14;

    const resize = () => {
      w = canvas.clientWidth;
      h = canvas.clientHeight;
      canvas.width = Math.round(w * dpr);
      canvas.height = Math.round(h * dpr);
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    };

    const onMove = (e: PointerEvent) => {
      if (reduced) return;
      mouse.tx = e.clientX / window.innerWidth;
      mouse.ty = e.clientY / window.innerHeight;
    };

    const lerp = (a: number, b: number, f: number) => a + (b - a) * f;

    const frame = (now: number) => {
      raf = requestAnimationFrame(frame);
      const dt = Math.min((now - last) / 1000, 0.05);
      last = now;
      if (!reduced) t += dt;

      mouse.x = smoothDamp(mouse.x, mouse.tx, 0.18, dt);
      mouse.y = smoothDamp(mouse.y, mouse.ty, 0.18, dt);
      lightMix = smoothDamp(lightMix, lightRef.current ? 1 : 0, 0.25, dt);

      // Background gradient
      const bgDark = '#04060c';
      const bgLight = '#f0f2f6';
      const r1 = lerp(4, 240, lightMix);
      const g1 = lerp(6, 242, lightMix);
      const b1 = lerp(12, 246, lightMix);
      const r2 = lerp(8, 230, lightMix);
      const g2 = lerp(14, 234, lightMix);
      const b2 = lerp(28, 242, lightMix);

      const grad = ctx.createRadialGradient(
        w * (0.48 + (mouse.x - 0.5) * 0.06),
        h * (0.42 + (mouse.y - 0.5) * 0.06),
        0,
        w * 0.5, h * 0.5, w * 0.8
      );
      grad.addColorStop(0, `rgb(${r1 + 12},${g1 + 16},${b1 + 30})`);
      grad.addColorStop(0.5, `rgb(${r1},${g1},${b1})`);
      grad.addColorStop(1, `rgb(${r2},${g2},${b2})`);
      ctx.fillStyle = grad;
      ctx.fillRect(0, 0, w, h);

      // Subtle grid overlay
      const gridAlpha = lerp(0.04, 0.06, lightMix);
      ctx.strokeStyle = lightMix > 0.5
        ? `rgba(22, 34, 58, ${gridAlpha})`
        : `rgba(120, 160, 220, ${gridAlpha})`;
      ctx.lineWidth = 0.5;
      const gridSize = 44;
      const ox = (t * 3) % gridSize;
      const oy = (t * 2) % gridSize;
      ctx.beginPath();
      for (let x = -gridSize + ox; x < w + gridSize; x += gridSize) {
        ctx.moveTo(x, 0);
        ctx.lineTo(x, h);
      }
      for (let y = -gridSize + oy; y < h + gridSize; y += gridSize) {
        ctx.moveTo(0, y);
        ctx.lineTo(w, y);
      }
      ctx.stroke();

      // Update nodes
      const parallaxX = (mouse.x - 0.5) * 0.02;
      const parallaxY = (mouse.y - 0.5) * 0.02;

      for (const n of nodes) {
        if (!reduced) {
          n.x += n.vx * dt;
          n.y += n.vy * dt;
        }
        // Wrap around
        if (n.x < -0.05) n.x = 1.05;
        if (n.x > 1.05) n.x = -0.05;
        if (n.y < -0.05) n.y = 1.05;
        if (n.y > 1.05) n.y = -0.05;
      }

      // Draw connections
      const connAlpha = lerp(0.12, 0.08, lightMix);
      for (let i = 0; i < nodes.length; i++) {
        const a = nodes[i];
        const ax = a.x * w + parallaxX * w * (a.kind + 1) * 0.5;
        const ay = a.y * h + parallaxY * h * (a.kind + 1) * 0.5;
        for (let j = i + 1; j < nodes.length; j++) {
          const b = nodes[j];
          const bx = b.x * w + parallaxX * w * (b.kind + 1) * 0.5;
          const by = b.y * h + parallaxY * h * (b.kind + 1) * 0.5;
          const dx = (a.x - b.x);
          const dy = (a.y - b.y);
          const dist = Math.sqrt(dx * dx + dy * dy);
          if (dist < CONN_DIST) {
            const fade = 1 - dist / CONN_DIST;
            ctx.strokeStyle = lightMix > 0.5
              ? `rgba(80, 120, 200, ${fade * connAlpha})`
              : `rgba(100, 160, 240, ${fade * connAlpha})`;
            ctx.lineWidth = fade * 1.2;
            ctx.beginPath();
            ctx.moveTo(ax, ay);
            ctx.lineTo(bx, by);
            ctx.stroke();
          }
        }
      }

      // Draw nodes
      for (const n of nodes) {
        const px = n.x * w + parallaxX * w * (n.kind + 1) * 0.5;
        const py = n.y * h + parallaxY * h * (n.kind + 1) * 0.5;
        const pulse = 0.7 + 0.3 * Math.sin(t * 1.2 + n.phase);
        const radius = n.r * (n.kind === 2 ? 2 : n.kind === 1 ? 1.4 : 1) * pulse;

        // Glow
        if (n.kind > 0) {
          const glowR = radius * (n.kind === 2 ? 6 : 4);
          const glow = ctx.createRadialGradient(px, py, 0, px, py, glowR);
          if (lightMix > 0.5) {
            const ga = 0.06 * pulse;
            glow.addColorStop(0, n.kind === 2 ? `rgba(40, 100, 220, ${ga})` : `rgba(60, 160, 130, ${ga})`);
            glow.addColorStop(1, 'rgba(40, 100, 220, 0)');
          } else {
            const ga = 0.12 * pulse;
            glow.addColorStop(0, n.kind === 2 ? `rgba(100, 180, 255, ${ga})` : `rgba(127, 227, 176, ${ga})`);
            glow.addColorStop(1, 'rgba(100, 180, 255, 0)');
          }
          ctx.fillStyle = glow;
          ctx.fillRect(px - glowR, py - glowR, glowR * 2, glowR * 2);
        }

        // Core
        ctx.beginPath();
        ctx.arc(px, py, radius, 0, Math.PI * 2);
        if (lightMix > 0.5) {
          ctx.fillStyle = n.kind === 2
            ? `rgba(40, 100, 220, ${0.5 * pulse})`
            : n.kind === 1
            ? `rgba(60, 160, 130, ${0.4 * pulse})`
            : `rgba(80, 120, 180, ${0.25 * pulse})`;
        } else {
          ctx.fillStyle = n.kind === 2
            ? `rgba(120, 190, 255, ${0.7 * pulse})`
            : n.kind === 1
            ? `rgba(127, 227, 176, ${0.5 * pulse})`
            : `rgba(140, 180, 240, ${0.3 * pulse})`;
        }
        ctx.fill();
      }

      // Vignette
      const vigGrad = ctx.createRadialGradient(w * 0.5, h * 0.5, w * 0.15, w * 0.5, h * 0.5, w * 0.75);
      vigGrad.addColorStop(0, 'rgba(0,0,0,0)');
      vigGrad.addColorStop(1, lightMix > 0.5 ? 'rgba(230,234,242,0.35)' : 'rgba(2,4,10,0.4)');
      ctx.fillStyle = vigGrad;
      ctx.fillRect(0, 0, w, h);
    };

    resize();
    window.addEventListener('resize', resize);
    window.addEventListener('pointermove', onMove, { passive: true });
    canvas.style.opacity = '1';
    raf = requestAnimationFrame(frame);

    return () => {
      cancelAnimationFrame(raf);
      window.removeEventListener('resize', resize);
      window.removeEventListener('pointermove', onMove);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  return (
    <div className={className} aria-hidden="true">
      <canvas
        ref={canvasRef}
        style={{
          position: 'absolute', inset: 0, width: '100%', height: '100%',
          display: 'block', opacity: 0, transition: 'opacity 1s cubic-bezier(0.22, 1, 0.36, 1)',
        }}
      />
    </div>
  );
}

export default LibraryHero;
