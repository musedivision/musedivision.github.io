// Copy the Neural Earth viewer lib from its source repo into the blog post.
// The viewer is developed in ../../dev/geo/web-demo (build it there first with
// `npm run build:lib:int4`); this repo only carries the minified bundle.
import { copyFileSync, statSync } from 'node:fs';
import { resolve, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

const here = dirname(fileURLToPath(import.meta.url));
const src = process.env.EARTH_LIB || resolve(here, '../../../dev/geo/web-demo/dist-lib/neural-earth.int4.js');
const dst = resolve(here, '../src/posts/2026-08-24-neural-earth/neural-earth.js');

try { statSync(src); } catch { console.error(`[sync-earth] not found: ${src}\nset EARTH_LIB=<path> to override`); process.exit(1); }
copyFileSync(src, dst);
console.log(`[sync-earth] ${src}\n          -> ${dst} (${(statSync(dst).size / 1024).toFixed(0)} kB)`);
