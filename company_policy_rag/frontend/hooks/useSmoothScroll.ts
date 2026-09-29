'use client';

import { useEffect } from 'react';
import type { RefObject } from 'react';

/**
 * useSmoothScroll — buttery wheel-driven inertia scrolling for an overflow
 * container. Intercepts wheel input and lerps `scrollTop` toward a target with
 * frame-rate-independent exponential damping, so scrolling stays glassy over
 * the animated backdrop. If anything else moves the pane mid-glide (scroll
 * anchoring, the chat's follow-the-answer jump), the glide absorbs that offset
 * instead of stalling or dragging the pane back toward a stale target.
 *
 * It only hijacks vertical mouse-wheel / trackpad input. Touch scrolling keeps
 * its native momentum, horizontal gestures pass through, and it bails entirely
 * when the user prefers reduced motion or the element isn't overflowing.
 */
interface SmoothScrollOptions {
  /** Damping half-life in seconds — lower = snappier, higher = floatier. */
  halfLife?: number;
  /** Wheel-delta multiplier. 1 keeps the OS scroll distance. */
  speed?: number;
  /** Disable without unmounting (e.g. tab not visible). */
  enabled?: boolean;
}

export function useSmoothScroll(
  ref: RefObject<HTMLElement | null>,
  { halfLife = 0.11, speed = 1, enabled = true }: SmoothScrollOptions = {},
) {
  useEffect(() => {
    const el = ref.current;
    if (!el || !enabled) return;

    if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) return;

    let target = el.scrollTop;
    let animating = false;
    let raf = 0;
    let lastTime = 0;
    let lastSet = -1; // scrollTop we wrote last frame (-1 = none yet)

    const maxScroll = () => el.scrollHeight - el.clientHeight;

    const step = (time: number) => {
      const dt = Math.min((time - lastTime) / 1000, 0.05); // clamp tab-switch gaps
      lastTime = time;

      const factor = 1 - Math.pow(2, -dt / halfLife);
      const current = el.scrollTop;

      // Moved by someone else since our last write. Small moves (the browser's
      // scroll anchoring as streamed text re-lays out, the chat's follow-the-
      // answer jump) are carried into the target so the glide neither stalls
      // nor drags the pane back; a large move means someone took over.
      if (lastSet >= 0) {
        const drift = current - lastSet;
        if (Math.abs(drift) > 240) {
          animating = false;
          lastSet = -1;
          return;
        }
        if (Math.abs(drift) > 0.5) target = Math.max(0, Math.min(maxScroll(), target + drift));
      }

      const next = current + (target - current) * factor;

      if (Math.abs(target - next) < 0.4) {
        el.scrollTop = target;
        animating = false;
        lastSet = -1;
        return;
      }

      el.scrollTop = next;
      lastSet = next;
      raf = requestAnimationFrame(step);
    };

    const onWheel = (e: WheelEvent) => {
      // Let the browser own zoom, horizontal intent, and non-scrollable panes.
      if (e.ctrlKey || Math.abs(e.deltaX) > Math.abs(e.deltaY)) return;
      const limit = maxScroll();
      if (limit <= 0) return;

      let delta = e.deltaY;
      if (e.deltaMode === 1) delta *= 16; // lines → px
      else if (e.deltaMode === 2) delta *= el.clientHeight; // pages → px

      // Resync the target to the live position whenever we're idle so scrollbar
      // drags and programmatic jumps (auto-scroll to newest) aren't fought.
      if (!animating) target = el.scrollTop;

      const nextTarget = Math.max(0, Math.min(limit, target + delta * speed));

      // Only claim the gesture if we can actually move; otherwise let it bubble
      // (e.g. nested scroller already at its edge should scroll the parent).
      if (nextTarget === target) return;

      e.preventDefault();
      target = nextTarget;

      if (!animating) {
        animating = true;
        lastSet = -1;
        lastTime = performance.now();
        raf = requestAnimationFrame(step);
      }
    };

    el.addEventListener('wheel', onWheel, { passive: false });
    return () => {
      el.removeEventListener('wheel', onWheel);
      cancelAnimationFrame(raf);
    };
  }, [ref, halfLife, speed, enabled]);
}
