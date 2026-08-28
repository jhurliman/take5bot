import { useEffect, useLayoutEffect, useRef, useSyncExternalStore } from "react";

/**
 * Viewport predicates: the single source of truth for the responsive
 * layout. `useLayoutMode` publishes the result as data-layout on <html>,
 * and index.css keys its `compact:` variant off that same attribute, so
 * DOM structure and leaf styles can never disagree.
 *
 * These are HEIGHT and POINTER predicates, never width alone. A
 * landscape iPhone is 844-932px wide, so it passes Tailwind's sm (640)
 * and md (768) breakpoints while being only ~330px tall. Width is the
 * wrong axis for this problem.
 */

export type LayoutMode = "desktop" | "compact";

/** max-height 500 separates phones (SE 375, 15 Pro 393, Pro Max 430)
 *  from iPad landscape (768). pointer:coarse keeps a merely short
 *  desktop window out. */
const MQ_COMPACT = "(orientation: landscape) and (max-height: 500px) and (pointer: coarse)";

/** max-width 540 keeps iPad portrait (768+) out of the gate;
 *  pointer:coarse keeps a tall, narrow desktop window out. Both clauses
 *  are load-bearing, because CSS `orientation` is only height > width. */
const MQ_ROTATE = "(orientation: portrait) and (max-width: 540px) and (pointer: coarse)";

function forcedLayout(): LayoutMode | null {
  const forced = new URLSearchParams(window.location.search).get("layout");
  return forced === "compact" || forced === "desktop" ? forced : null;
}

/** iOS resizes the visual viewport (URL bar) and can report a stale
 *  orientation on the matchMedia event alone, so listen broadly. */
function subscribeViewport(onChange: () => void) {
  const mqs = [window.matchMedia(MQ_COMPACT), window.matchMedia(MQ_ROTATE)];
  for (const mq of mqs) mq.addEventListener("change", onChange);
  window.addEventListener("resize", onChange);
  window.addEventListener("orientationchange", onChange);
  return () => {
    for (const mq of mqs) mq.removeEventListener("change", onChange);
    window.removeEventListener("resize", onChange);
    window.removeEventListener("orientationchange", onChange);
  };
}

function readMode(): LayoutMode {
  return forcedLayout() ?? (window.matchMedia(MQ_COMPACT).matches ? "compact" : "desktop");
}

function readNeedsRotate(): boolean {
  // An explicit ?layout= override also suppresses the gate, so the
  // compact layout can be iterated on in a normal desktop window.
  if (forcedLayout()) return false;
  if (import.meta.env.DEV && new URLSearchParams(window.location.search).has("rotate")) {
    return true;
  }
  return window.matchMedia(MQ_ROTATE).matches;
}

/**
 * useSyncExternalStore rather than useState + effect: no flash of the
 * wrong layout on first paint, and no tearing. Both snapshots return
 * primitives, so there is no getSnapshot identity churn.
 */
export function useLayoutMode(): LayoutMode {
  const mode = useSyncExternalStore(subscribeViewport, readMode, () => "desktop" as const);
  // useLayoutEffect, not useEffect: the attribute drives --cw-hand and
  // --cw-board, so it must land before paint or the first frame renders
  // cards at their var() fallback width.
  useLayoutEffect(() => {
    document.documentElement.dataset.layout = mode;
  }, [mode]);
  return mode;
}

export function useNeedsRotate(): boolean {
  return useSyncExternalStore(subscribeViewport, readNeedsRotate, () => false);
}

/** Escape-to-close for dismissible overlays. The handler is held in a
 *  ref so callers need not memoise their callback. */
export function useEscape(onClose: () => void): void {
  const ref = useRef(onClose);
  ref.current = onClose;
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") ref.current();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);
}
