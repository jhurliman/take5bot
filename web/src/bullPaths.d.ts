/** Type declarations for bullPaths.js (plain JS so Node's icon
 * generator can import it without a TypeScript loader). */

export declare const BULL_VIEWBOX: string;
export declare const BULL_HORN_L: string;
export declare const BULL_HORN_R: string;
export declare const BULL_HEAD: string;
export declare const BULL_EARS: ReadonlyArray<{
  cx: number;
  cy: number;
  rx: number;
  ry: number;
  rotate: number;
}>;
export declare function bullShapesMarkup(fill: string): string;
