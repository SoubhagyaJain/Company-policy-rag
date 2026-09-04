'use client';

/**
 * BlackHoleAbsorption — a purely visual layer that animates a DOM element being
 * gravitationally pulled into the hero's black hole. It clones the source node,
 * spirals it toward the live black-hole screen point (read via `getTarget`),
 * applies tidal stretch / blur / rotation, then shatters it into glowing
 * particles + streaks with a lensing ripple at the event horizon.
 *
 * It touches no app state: callers keep their own delete/backend/undo logic and
 * simply run it in the `onComplete` callback. Respects prefers-reduced-motion
 * with a fade + shrink fallback.
 */

import { createContext, useContext, useEffect, useMemo, useRef, useState, ReactNode } from 'react';
import { createPortal } from 'react-dom';

export interface AbsorbOptions {
  /** Delay before the spiral starts (used to stagger Clear-all). The clone is
   *  captured immediately so staggered items keep their original positions. */
  delay?: number;
  onComplete?: () => void;
}

export interface AbsorbController {
  absorb: (sourceEl: HTMLElement, opts?: AbsorbOptions) => void;
}

const BlackHoleContext = createContext<AbsorbController | null>(null);
export const useBlackHole = () => useContext(BlackHoleContext);

type Point = { x: number; y: number };

const DURATION = 880; // ms — within the 700–1000ms cinematic window
const BREAK_AT = 0.72; // progress at which the card shatters into particles

function prefersReducedMotion(): boolean {
  return typeof window !== 'undefined' && window.matchMedia('(prefers-reduced-motion: reduce)').matches;
}

/** Fade + shrink fallback for reduced motion. */
function runReduced(layer: HTMLElement, source: HTMLElement, onComplete?: () => void) {
  const rect = source.getBoundingClientRect();
  const clone = makeClone(source, rect);
  layer.appendChild(clone);
  clone.style.transition = 'transform 260ms ease, opacity 260ms ease, filter 260ms ease';
  requestAnimationFrame(() => {
    clone.style.transform = 'translate3d(0,0,0) scale(0.6)';
    clone.style.opacity = '0';
    clone.style.filter = 'blur(2px)';
  });
  window.setTimeout(() => {
    clone.remove();
    onComplete?.();
  }, 300);
}

function makeClone(source: HTMLElement, rect: DOMRect): HTMLElement {
  const clone = source.cloneNode(true) as HTMLElement;
  clone.removeAttribute('data-session-row');
  clone.classList.add('bh-clone');
  clone.style.cssText +=
    `;position:fixed;left:${rect.left}px;top:${rect.top}px;width:${rect.width}px;` +
    `height:${rect.height}px;margin:0;opacity:1;pointer-events:none;` +
    `transform-origin:center center;will-change:transform,opacity,filter;z-index:1;`;
  return clone;
}

function spawnRipple(layer: HTMLElement, target: Point) {
  for (let i = 0; i < 2; i++) {
    const ring = document.createElement('div');
    ring.className = 'bh-ripple';
    const size = 26;
    ring.style.left = `${target.x - size / 2}px`;
    ring.style.top = `${target.y - size / 2}px`;
    ring.style.width = `${size}px`;
    ring.style.height = `${size}px`;
    layer.appendChild(ring);
    const delay = i * 90;
    const life = 520;
    const t0 = performance.now() + delay;
    const tick = (now: number) => {
      const q = (now - t0) / life;
      if (q < 0) {
        requestAnimationFrame(tick);
        return;
      }
      if (q >= 1) {
        ring.remove();
        return;
      }
      const scale = 0.3 + q * (14 + i * 6);
      ring.style.transform = `translate3d(0,0,0) scale(${scale})`;
      ring.style.opacity = String((1 - q) * 0.5);
      requestAnimationFrame(tick);
    };
    requestAnimationFrame(tick);
  }
}

function spawnParticles(layer: HTMLElement, from: Point, target: Point, dir: number) {
  const count = 18;
  const particles: Array<{ el: HTMLElement; startR: number; startA: number; streak: boolean; life: number; born: number }> = [];
  const now0 = performance.now();

  for (let i = 0; i < count; i++) {
    const el = document.createElement('div');
    const streak = i % 3 === 0;
    el.className = streak ? 'bh-particle bh-streak' : 'bh-particle';
    const s = streak ? 9 : 3 + Math.random() * 3;
    el.style.width = `${streak ? 12 : s}px`;
    el.style.height = `${streak ? 2 : s}px`;
    layer.appendChild(el);
    // scatter around the break point
    const a = Math.random() * Math.PI * 2;
    const r = 6 + Math.random() * 26;
    const px = from.x + Math.cos(a) * r;
    const py = from.y + Math.sin(a) * r;
    particles.push({
      el,
      startR: Math.hypot(px - target.x, py - target.y),
      startA: Math.atan2(py - target.y, px - target.x),
      streak,
      life: 320 + Math.random() * 220,
      born: now0 + Math.random() * 60,
    });
  }

  const tick = (now: number) => {
    let alive = 0;
    for (const p of particles) {
      const q = (now - p.born) / p.life;
      if (q < 0) {
        alive++;
        continue;
      }
      if (q >= 1) {
        if (p.el.isConnected) p.el.remove();
        continue;
      }
      alive++;
      const e = Math.pow(q, 1.5);
      const radius = p.startR * Math.pow(1 - q, 1.4);
      const angle = p.startA + dir * 2.4 * Math.PI * e;
      const x = target.x + Math.cos(angle) * radius;
      const y = target.y + Math.sin(angle) * radius;
      const rot = p.streak ? (Math.atan2(target.y - y, target.x - x) * 180) / Math.PI : 0;
      const scale = 1 - 0.7 * e;
      p.el.style.transform = `translate3d(${x}px, ${y}px, 0) rotate(${rot}deg) scale(${scale})`;
      p.el.style.opacity = String(Math.min(1, 2 * (1 - q)));
    }
    if (alive > 0) requestAnimationFrame(tick);
  };
  requestAnimationFrame(tick);
}

function runAbsorb(layer: HTMLElement, source: HTMLElement, getTarget: () => Point, opts?: AbsorbOptions) {
  const rect = source.getBoundingClientRect();
  const clone = makeClone(source, rect);
  layer.appendChild(clone);

  const begin = () => {
    const target = getTarget();
    const cx = rect.left + rect.width / 2;
    const cy = rect.top + rect.height / 2;
    const dx = cx - target.x;
    const dy = cy - target.y;
    const startR = Math.hypot(dx, dy) || 1;
    const startA = Math.atan2(dy, dx);
    const dir = dx >= 0 ? 1 : -1; // spiral toward the hole from whichever side
    const turns = 1.15;
    let broke = false;
    const t0 = performance.now();

    const frame = (now: number) => {
      const p = Math.min((now - t0) / DURATION, 1);
      const e = Math.pow(p, 1.75); // gravitational acceleration
      // brief forward lift at the very start
      const lift = p < 0.14 ? Math.sin((p / 0.14) * Math.PI) * -9 : 0;

      const radius = startR * Math.pow(1 - p, 1.55);
      const angle = startA + dir * turns * Math.PI * 2 * e;
      const x = target.x + Math.cos(angle) * radius;
      const y = target.y + Math.sin(angle) * radius + lift;
      const tx = x - cx;
      const ty = y - cy;

      const shrink = 1 - 0.9 * e; // → ~0.1
      const stretch = 1 + 0.6 * Math.pow(p, 3); // tidal elongation
      const radialDeg = (Math.atan2(target.y - y, target.x - x) * 180) / Math.PI;
      const spin = (angle - startA) * (180 / Math.PI) * 0.6 + p * 40;
      const blur = 3.5 * Math.pow(p, 2);

      clone.style.transform =
        `translate3d(${tx}px, ${ty}px, 0) rotate(${radialDeg}deg) ` +
        `scale(${shrink * stretch}, ${shrink / Math.sqrt(stretch)}) rotate(${spin}deg)`;
      clone.style.filter = `blur(${blur}px)`;
      clone.style.opacity = String(p < BREAK_AT ? 1 : Math.max(0, 1 - (p - BREAK_AT) / 0.14));

      if (!broke && p >= BREAK_AT) {
        broke = true;
        spawnParticles(layer, { x, y }, target, dir);
        spawnRipple(layer, target);
      }

      if (p < 1) {
        requestAnimationFrame(frame);
      } else {
        clone.remove();
        opts?.onComplete?.();
      }
    };
    requestAnimationFrame(frame);
  };

  if (opts?.delay) window.setTimeout(begin, opts.delay);
  else requestAnimationFrame(begin);
}

export function BlackHoleProvider({ getTarget, children }: { getTarget: () => Point; children: ReactNode }) {
  const layerRef = useRef<HTMLDivElement>(null);

  // The portal targets document.body, which doesn't exist during SSR. Only
  // mount it after hydration so the server and first client render agree
  // (both render nothing) — otherwise React reports a hydration mismatch.
  const [mounted, setMounted] = useState(false);
  useEffect(() => setMounted(true), []);

  const controller = useMemo<AbsorbController>(
    () => ({
      absorb: (sourceEl, opts) => {
        const layer = layerRef.current;
        if (!layer || !sourceEl) {
          // No layer/element — run the callback immediately so delete still happens.
          opts?.onComplete?.();
          return;
        }
        if (prefersReducedMotion()) {
          if (opts?.delay) window.setTimeout(() => runReduced(layer, sourceEl, opts?.onComplete), opts.delay);
          else runReduced(layer, sourceEl, opts?.onComplete);
          return;
        }
        runAbsorb(layer, sourceEl, getTarget, opts);
      },
    }),
    [getTarget],
  );

  return (
    <BlackHoleContext.Provider value={controller}>
      {children}
      {mounted &&
        createPortal(
          <div ref={layerRef} className="bh-layer pointer-events-none fixed inset-0 z-[200]" aria-hidden="true" />,
          document.body,
        )}
    </BlackHoleContext.Provider>
  );
}

export default BlackHoleProvider;
