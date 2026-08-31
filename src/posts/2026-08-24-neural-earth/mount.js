// Gate the Neural Earth viewer behind a click (it downloads ~441 MB of weights), then mount it
// into #earth. Shared by the blog post (inline, in a column) and the /neural-earth/
// demo page (full viewport) — the host page only has to size #earth.
// A return visit whose weights are already in Cache Storage skips the gate and starts at once.
import { mountNeuralEarth } from './neural-earth.js';

const WEIGHTS = 'https://huggingface.co/musedivision/neural-earth/resolve/main';
const CACHE_NAME = 'neural-earth-weights-v1'; // the viewer's Cache Storage bucket (WeightLoader)
const CACHE_PROBE = 'coarse.grid.i4.bin';     // first leg's grid: cached ⇒ the download already happened
// The viewer's own first frame, captured at 2048² (same framing on every screen: see GlobeCamera.fitView).
// Behind the gate it is the dimmed backdrop, behind the loading screen too, and on a cached start it simply
// stays until the live frame is sharp.
const POSTER = new URL('./poster.webp', import.meta.url).href;

const GATE_CSS = `
#earth .earth-poster { position: absolute; inset: 0; width: 100%; height: 100%; object-fit: contain; background: #05080f; margin: 0; border: 0; opacity: .55; transition: opacity .6s ease; }
.earth-gate { position: absolute; inset: 0; display: flex; flex-direction: column; align-items: center; justify-content: center; gap: .9rem; text-align: center; padding: 1.5rem; font-family: 'Share Tech Mono', 'JetBrains Mono', Menlo, Consolas, monospace; transition: opacity .4s ease; }
.earth-gate.leaving, #earth.leaving .earth-poster { opacity: 0; pointer-events: none; }
.earth-start { font: inherit; font-size: 1.05rem; letter-spacing: .18em; text-transform: lowercase; color: #fff; background: rgba(0,0,0,.5); border: 1px solid rgba(255,255,255,.7); padding: .8rem 2.2rem; cursor: pointer; transition: background .2s ease, border-color .2s ease; }
.earth-start:hover { background: rgba(255,255,255,.14); border-color: #fff; }
.earth-note { font-size: .78rem; letter-spacing: .06em; color: rgba(255,255,255,.55); max-width: 28rem; line-height: 1.5; }
.earth-note code { font: inherit; color: rgba(255,255,255,.85); }
`;

const params = new URLSearchParams(location.search);
const weights = params.get('weights') || WEIGHTS;

// Why WebGPU is unavailable here, or null if it is — each cause needs a different fix,
// and a bare "needs WebGPU" sends people to update a browser that already has it.
async function probeWebGPU() {
  if (!window.isSecureContext) {
    return `WebGPU only runs on a secure page — open this over <code>https://</code> or <code>localhost</code>, not <code>http://${location.hostname}</code>`;
  }
  if (!navigator.gpu) {
    return 'needs WebGPU — Chrome / Edge 113+, Safari 26+ or Firefox 141+';
  }
  try {
    const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
    if (!adapter) {
      return 'this browser has WebGPU but offers no GPU adapter — check <code>chrome://gpu</code> (hardware acceleration must be on) or try Chrome / Edge';
    }
  } catch (e) {
    return `WebGPU adapter error: ${e.message}`;
  }
  return null;
}

// True if this origin already holds the weights (Cache Storage is per origin and
// keyed by URL, so a different `?weights=` base counts as a fresh download).
async function weightsCached() {
  try {
    const cache = await caches.open(CACHE_NAME);
    return !!(await cache.match(`${weights.replace(/\/+$/, '')}/${CACHE_PROBE}`));
  } catch {
    return false; // no Cache Storage (insecure context, private mode)
  }
}

function mount(root, { cached = false } = {}) {
  root.querySelector('.earth-poster')?.remove();
  // neural-earth.js is the int4 lib build: both legs + int4 grids are its defaults.
  mountNeuralEarth(root, {
    weights,
    poster: POSTER, quietLoad: cached,
    brand: false, help: true,
    onError: (e) => console.error('[earth]', e),
  });
}

function showGate(root, ready) {
  const gate = document.createElement('div');
  gate.className = 'earth-gate';
  gate.innerHTML = `<button type="button" class="earth-start">enter</button>`;
  root.appendChild(gate);
  const button = gate.querySelector('.earth-start');

  button.addEventListener('click', async () => {
    button.disabled = true;
    const problem = await ready;
    if (problem) {
      gate.innerHTML = `<span class="earth-note">${problem}</span>`;
      return;
    }
    gate.classList.add('leaving');
    root.classList.add('leaving');
    await new Promise(r => setTimeout(r, 400));
    root.classList.remove('leaving');
    gate.remove();
    mount(root);
  });

  ready.then(problem => {
    if (!problem) return;
    console.warn('[earth] WebGPU unavailable:', problem.replace(/<[^>]+>/g, ''));
    gate.innerHTML = `<span class="earth-note">${problem}</span>`;
  });
}

const root = document.getElementById('earth');
if (root) {
  const style = document.createElement('style');
  style.textContent = GATE_CSS;
  document.head.appendChild(style);

  const ready = probeWebGPU();
  weightsCached().then(async (cached) => {
    // Weights already here and a GPU to run them: no door, straight in.
    if (cached && !(await ready)) { mount(root, { cached: true }); return; }
    showGate(root, ready);
  });
}
