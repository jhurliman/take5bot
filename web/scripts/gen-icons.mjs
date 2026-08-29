#!/usr/bin/env node
/**
 * Generates the PWA/favicon icon set from the same bull silhouette the
 * cards use (src/bullPaths.js), so the icon can never drift from the
 * art.
 *
 * `sharp` is deliberately NOT a dependency: CI runs `npm ci` on every PR
 * and every Pages deploy, and it should not pull ~30 MB of prebuilt
 * binaries for something that regenerates roughly never. Install it
 * transiently and commit the output:
 *
 *   npm i -D --no-save sharp && node scripts/gen-icons.mjs
 *
 * Outputs (all committed):
 *   public/icon.svg
 *   public/icons/icon-192.png       public/icons/icon-512.png
 *   public/icons/maskable-192.png   public/icons/maskable-512.png
 *   public/icons/apple-touch-icon-180.png
 */

import { mkdir, writeFile } from "node:fs/promises";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import sharp from "sharp";
import { BULL_VIEWBOX, bullShapesMarkup } from "../src/bullPaths.js";

const HERE = dirname(fileURLToPath(import.meta.url));
const PUBLIC = join(HERE, "..", "public");
const ICONS = join(PUBLIC, "icons");

/** Tailwind v4 slate-950 in sRGB. Must match theme_color and index.html. */
const BG = "#020618";
/** amber-400: warm, high contrast on near-black, and it echoes the
 *  amber bull tier on the printed cards. */
const FG = "#fbbf24";

/* The bull's drawn extent inside the 100x100 viewBox: the horn tips
 * reach x=8..92 and y=5, the chin bottoms out at y=83.5. Centring on the
 * canvas centre instead of the viewBox centre keeps it optically level. */
const BULL = { w: 84, cx: 50, cy: 44.25 };

/**
 * @param {number} size    output pixel size
 * @param {number} frac    fraction of the canvas the bull's width spans
 * @param {boolean} rounded  round the background (SVG favicon only)
 */
function svg(size, frac, rounded = false) {
  const s = (frac * 100) / BULL.w;
  const tx = 50 - s * BULL.cx;
  const ty = 50 - s * BULL.cy;
  const bg = rounded
    ? `<rect width="100" height="100" rx="22" fill="${BG}"/>`
    : `<rect width="100" height="100" fill="${BG}"/>`;
  return `<svg xmlns="http://www.w3.org/2000/svg" width="${size}" height="${size}" viewBox="${BULL_VIEWBOX}">${bg}<g transform="translate(${tx.toFixed(3)} ${ty.toFixed(3)}) scale(${s.toFixed(5)})">${bullShapesMarkup(FG)}</g></svg>`;
}

async function png(out, size, frac) {
  await sharp(Buffer.from(svg(size, frac)))
    .png({ compressionLevel: 9 })
    .toFile(join(ICONS, out));
  console.log("  wrote icons/%s (%dx%d)", out, size, size);
}

await mkdir(ICONS, { recursive: true });

// Favicon / manifest "any": rounded corners look right in a browser tab.
await writeFile(join(PUBLIC, "icon.svg"), svg(512, 0.72, true) + "\n");
console.log("  wrote icon.svg");

await png("icon-192.png", 192, 0.72);
await png("icon-512.png", 512, 0.72);

// Maskable needs its own art, not a rescale: Android crops to a circle
// whose 80% "safe zone" is a DIAMETER, so the mark must sit well inside.
await png("maskable-192.png", 192, 0.6);
await png("maskable-512.png", 512, 0.6);

// iOS: PNG only (SVG is ignored), opaque (transparency renders black),
// and no pre-rounded corners, because iOS applies its own squircle mask.
await png("apple-touch-icon-180.png", 180, 0.7);

console.log("done.");
