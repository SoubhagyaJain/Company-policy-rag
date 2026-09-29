'use client';

/**
 * AlpineBackdrop — the app's cinematic, persistent background.
 *
 * Two photographs of the same alpine valley are stacked on one "stage": the
 * moonlit night scene (dark theme) and the golden-hour scene (light theme). The
 * theme swap is a slow opacity crossfade driven purely by the `html.dark` class,
 * so the mountains stay put while the sky changes and nothing flashes on load.
 *
 * The stage emulates `object-fit: cover`, which lets overlays (moon glow,
 * twinkling stars) be positioned in image-relative percentages and stay pinned
 * to the photo at every viewport size. Layered scrims keep text legible: a heavy
 * floor gradient over the busy flower field, a left-edge shade behind the
 * sidebar, and a `focus` veil that deepens whenever the UI is text-dense
 * (an active conversation, Library, Telemetry). Focus mode also pauses the slow
 * drift so the blurred glass panels are not re-composited every frame.
 * The scene fades in once both photos have decoded, and the theme crossfade is
 * only armed after first paint, so a reload never flashes the other photo.
 * Motion is transform/opacity only and is disabled under prefers-reduced-motion.
 */

import { memo, useEffect, useRef, useState } from 'react';

/** Night photo dimensions (the stage aspect ratio). */
const IMG_W = 1446;
const IMG_H = 1088;
/** Resting scale of the stage (headroom so the slow drift never exposes an edge). */
const STAGE_SCALE = 1.05;

/** Image-relative focal points. */
export const MOON = { x: 0.813, y: 0.207 };
export const SUNSET_GLOW = { x: 0.47, y: 0.59 };

/** Horizontal anchor of the crop — must match `.alp-stage` in space.css. */
const FOCUS_X = 0.62;

/** Screen position of an image-relative point under the cover crop. */
export function backdropPoint(rel: { x: number; y: number }): { x: number; y: number } {
  const vw = window.innerWidth;
  const vh = window.innerHeight;
  const ratio = IMG_W / IMG_H;
  const stageW = Math.max(vw, vh * ratio);
  const stageH = stageW / ratio;
  const left = (vw - stageW) * FOCUS_X;
  const top = (vh - stageH) * 0.5;
  const cx = left + stageW / 2;
  const cy = top + stageH / 2;
  const px = left + rel.x * stageW;
  const py = top + rel.y * stageH;
  return { x: cx + (px - cx) * STAGE_SCALE, y: cy + (py - cy) * STAGE_SCALE };
}

/* Deterministic star field (same on server and client → no hydration drift). */
function mulberry32(seed: number) {
  return () => {
    seed |= 0;
    seed = (seed + 0x6d2b79f5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

const STARS = (() => {
  const rand = mulberry32(20260929);
  const out: Array<{ x: number; y: number; s: number; d: number; delay: number; o: number }> = [];
  while (out.length < 34) {
    const x = rand() * 100;
    const y = 2 + rand() * 34;
    // keep the moon's halo clear
    if (Math.hypot(x - MOON.x * 100, (y - MOON.y * 100) * 1.33) < 9) continue;
    out.push({
      x,
      y,
      s: rand() < 0.16 ? 2.4 : rand() < 0.5 ? 1.6 : 1.1,
      d: 3.2 + rand() * 4.6,
      delay: -rand() * 8,
      o: 0.45 + rand() * 0.5,
    });
  }
  return out;
})();

interface AlpineBackdropProps {
  /** Deepen the veil and hold the scene still for text-dense views. */
  focus?: boolean;
  className?: string;
}

export const AlpineBackdrop = memo(function AlpineBackdrop({ focus = false, className = '' }: AlpineBackdropProps) {
  const [ready, setReady] = useState(false);
  const [live, setLive] = useState(false);
  const nightRef = useRef<HTMLImageElement>(null);
  const duskRef = useRef<HTMLImageElement>(null);

  useEffect(() => {
    let cancelled = false;
    const imgs = [nightRef.current, duskRef.current].filter(Boolean) as HTMLImageElement[];
    Promise.allSettled(imgs.map((img) => img.decode())).then(() => {
      if (!cancelled) setReady(true);
    });
    const raf = requestAnimationFrame(() => requestAnimationFrame(() => setLive(true)));
    return () => {
      cancelled = true;
      cancelAnimationFrame(raf);
    };
  }, []);

  return (
    <div
      aria-hidden
      className={`alp ${ready ? 'is-ready' : ''} ${live ? 'is-live' : ''} ${focus ? 'is-focus' : ''} ${className}`}
    >
      <div className="alp-stage">
        <div className="alp-drift">
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img
            ref={duskRef}
            className="alp-img alp-img--dusk"
            src="/alpine-dusk.webp"
            alt=""
            decoding="async"
            draggable={false}
          />
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img
            ref={nightRef}
            className="alp-img alp-img--night"
            src="/alpine-night.webp"
            alt=""
            decoding="async"
            fetchPriority="high"
            draggable={false}
          />

          <div className="alp-night-only">
            <span className="alp-moonglow" style={{ left: `${MOON.x * 100}%`, top: `${MOON.y * 100}%` }} />
            {STARS.map((st, i) => (
              <span
                key={i}
                className="alp-star"
                style={{
                  left: `${st.x}%`,
                  top: `${st.y}%`,
                  width: st.s,
                  height: st.s,
                  opacity: st.o,
                  animationDuration: `${st.d}s`,
                  animationDelay: `${st.delay}s`,
                }}
              />
            ))}
          </div>

          <span className="alp-mist" />
        </div>
      </div>

      {/* Readability scrims */}
      <div className="alp-scrim" />
      <div className="alp-veil" />
      <div className="alp-grain" />
    </div>
  );
});

export default AlpineBackdrop;
