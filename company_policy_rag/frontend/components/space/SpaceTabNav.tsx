'use client';

/** SpaceTabNav — the shared Ask / Library / Telemetry switcher.
 *
 *  Motion system: Magnetic Pill + Glow Trail.
 *  A single indicator pill glides between tabs with spring physics, stretching
 *  briefly toward the tab you clicked (scaleX pulse, origin on the trailing
 *  edge) before settling. A softer, blurred copy trails a beat behind and fades,
 *  reading as faint bloom rather than neon. Everything is transform/opacity only
 *  (GPU-composited), measured off layout so there is zero layout shift, and it
 *  collapses to an instant swap under prefers-reduced-motion. The directional
 *  content crossfade lives in app/page.tsx and shares this ordering. */

import { useCallback, useEffect, useLayoutEffect, useRef, useState } from 'react';
import { motion, useReducedMotion } from 'framer-motion';

export type ViewTab = 'chat' | 'documents' | 'observability';

export const NAV: Array<{ id: ViewTab; label: string }> = [
  { id: 'chat', label: 'Ask' },
  { id: 'documents', label: 'Library' },
  { id: 'observability', label: 'Telemetry' },
];

const ORDER = NAV.map((n) => n.id);
export const tabDirection = (from: ViewTab, to: ViewTab) =>
  Math.sign(ORDER.indexOf(to) - ORDER.indexOf(from));

interface Geo { x: number; w: number }

interface SpaceTabNavProps {
  activeTab: ViewTab;
  onChange: (t: ViewTab) => void;
}

export function SpaceTabNav({ activeTab, onChange }: SpaceTabNavProps) {
  const reduce = useReducedMotion();

  const navRef = useRef<HTMLElement>(null);
  const btnRefs = useRef<Record<string, HTMLButtonElement | null>>({});

  const [geo, setGeo] = useState<Geo | null>(null);
  const [instant, setInstant] = useState(true); // snap on mount / resize, animate on click
  const [dir, setDir] = useState(0);
  const prevTab = useRef<ViewTab>(activeTab);

  const measure = useCallback((): Geo | null => {
    const nav = navRef.current;
    const btn = btnRefs.current[activeTab];
    if (!nav || !btn) return null;
    const nb = nav.getBoundingClientRect();
    const bb = btn.getBoundingClientRect();
    return { x: bb.left - nb.left, w: bb.width };
  }, [activeTab]);

  // Re-place the pill whenever the active tab changes — this one animates.
  useLayoutEffect(() => {
    const next = measure();
    if (!next) return;
    setDir(tabDirection(prevTab.current, activeTab));
    setInstant(prevTab.current === activeTab); // first paint snaps, real switches glide
    prevTab.current = activeTab;
    setGeo(next);
  }, [activeTab, measure]);

  // Keep the pill glued to its tab through resize / font load — snap, never glide.
  useEffect(() => {
    const nav = navRef.current;
    if (!nav) return;
    const resync = () => {
      const next = measure();
      if (next) {
        setInstant(true);
        setGeo(next);
      }
    };
    const ro = new ResizeObserver(resync);
    ro.observe(nav);
    window.addEventListener('resize', resync);
    return () => {
      ro.disconnect();
      window.removeEventListener('resize', resync);
    };
  }, [measure]);

  const pillSpring = reduce
    ? { duration: 0 }
    : { type: 'spring' as const, stiffness: 520, damping: 42, mass: 0.9 };
  const glowSpring = reduce
    ? { duration: 0 }
    : { type: 'spring' as const, stiffness: 210, damping: 26, mass: 1 };

  const stretchOrigin = dir >= 0 ? 'left center' : 'right center';

  return (
    <nav ref={navRef} className="sp-nav relative mx-auto flex items-center gap-1 rounded-full p-1">
      {/* Glow trail — trails a beat behind the pill and fades. */}
      {geo && !reduce && (
        <motion.span
          aria-hidden
          className="sp-tabpill-glow pointer-events-none absolute z-0"
          style={{ top: 2, bottom: 2, left: 0, willChange: 'transform, opacity' }}
          initial={false}
          animate={{
            x: geo.x,
            width: geo.w,
            opacity: instant ? 0 : [0, 0.5, 0],
          }}
          transition={{
            x: glowSpring,
            width: glowSpring,
            opacity: { duration: 0.5, times: [0, 0.28, 1], ease: 'easeOut' },
          }}
        />
      )}

      {/* Magnetic pill — the active indicator. */}
      {geo && (
        <motion.span
          aria-hidden
          className="sp-tabpill pointer-events-none absolute z-[1]"
          style={{ top: 4, bottom: 4, left: 0, transformOrigin: stretchOrigin, willChange: 'transform' }}
          initial={false}
          animate={{
            x: geo.x,
            width: geo.w,
            scaleX: instant || reduce ? 1 : [1, 1.05, 1],
          }}
          transition={{
            x: pillSpring,
            width: pillSpring,
            scaleX: { duration: 0.34, times: [0, 0.35, 1], ease: [0.22, 1, 0.36, 1] },
          }}
        />
      )}

      {NAV.map((n) => {
        const active = activeTab === n.id;
        return (
          <button
            key={n.id}
            ref={(el) => { btnRefs.current[n.id] = el; }}
            type="button"
            onClick={() => { if (!active) onChange(n.id); }}
            aria-current={active ? 'page' : undefined}
            className={`sp-tab relative z-[2] rounded-full px-4 py-2 text-[13px] font-medium sm:px-5 ${active ? 'is-active' : ''}`}
          >
            {n.label}
          </button>
        );
      })}
    </nav>
  );
}

export default SpaceTabNav;
