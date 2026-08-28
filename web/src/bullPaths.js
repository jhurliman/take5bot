/**
 * Bull-head silhouette geometry (front view, official-card style: broad
 * lyre horns, side ears, tapering face), on a 100x100 canvas.
 *
 * Plain .js with a sibling .d.ts so both the React component
 * (src/App.tsx BullIcon) and the Node icon generator
 * (scripts/gen-icons.mjs) read the same source of truth. Node cannot
 * import .ts without a loader, hence .js rather than .ts.
 */

export const BULL_VIEWBOX = "0 0 100 100";

export const BULL_HORN_L =
  "M38 36 C20 38 8 28 8 12 C8 7 13 5 15.5 9 C18 21 27 28 39 28 C42 30 41 35 38 36 Z";

export const BULL_HORN_R =
  "M62 36 C80 38 92 28 92 12 C92 7 87 5 84.5 9 C82 21 73 28 61 28 C58 30 59 35 62 36 Z";

export const BULL_HEAD =
  "M50 30 C40 30 33.5 38 33.5 48 C33.5 60 42 68 45.5 78 C47.5 83.5 52.5 83.5 54.5 78 C58 68 66.5 60 66.5 48 C66.5 38 60 30 50 30 Z";

export const BULL_EARS = [
  { cx: 28, cy: 44, rx: 10, ry: 5.5, rotate: -18 },
  { cx: 72, cy: 44, rx: 10, ry: 5.5, rotate: 18 },
];

/**
 * The bull's shapes as raw SVG markup, without an <svg> wrapper.
 * Used by the icon generator; the React component builds the same
 * shapes as elements from the constants above.
 *
 * @param {string} fill
 * @returns {string}
 */
export function bullShapesMarkup(fill) {
  const ears = BULL_EARS.map(
    (e) =>
      `<ellipse cx="${e.cx}" cy="${e.cy}" rx="${e.rx}" ry="${e.ry}" transform="rotate(${e.rotate} ${e.cx} ${e.cy})"/>`,
  ).join("");
  return (
    `<g fill="${fill}">` +
    `<path d="${BULL_HORN_L}"/>` +
    `<path d="${BULL_HORN_R}"/>` +
    ears +
    `<path d="${BULL_HEAD}"/>` +
    `</g>`
  );
}
