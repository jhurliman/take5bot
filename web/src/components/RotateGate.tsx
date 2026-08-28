import { RotateCw } from "lucide-react";

/**
 * "Turn your phone sideways" overlay.
 *
 * This is not a nicety: it IS the orientation lock on iOS, which
 * ignores the web app manifest's `orientation` field and throws on
 * screen.orientation.lock(). On Android, where the manifest lock does
 * work in an installed PWA, this simply never renders.
 *
 * Mounted as an overlay (last child of the app root), never as an early
 * return: an early return would unmount the game, replaying every
 * framer-motion enter animation on rotate-back and re-firing the coach
 * effects, which spawns a fresh worker analyze on every flip.
 */
export function RotateGate() {
  return (
    <div
      role="alertdialog"
      aria-modal="true"
      aria-label="Rotate your device"
      className="fixed inset-0 z-50 bg-slate-950 text-slate-100 flex flex-col
                 items-center justify-center gap-5 px-gutter pt-gutter pb-gutter
                 text-center select-none"
    >
      <RotateCw className="w-14 h-14 text-amber-400 animate-tilt" aria-hidden />
      <div className="text-lg font-semibold tracking-wide">Rotate your device</div>
      <p className="max-w-xs text-sm text-slate-400">
        Take 5 needs a wide screen. Turn your phone sideways to see the table and your
        hand at once.
      </p>
    </div>
  );
}
