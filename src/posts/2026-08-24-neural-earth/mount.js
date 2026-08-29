// Gate the Neural Earth viewer behind a click (358 MB of weights), then mount it
// into #earth. Shared by the blog post (inline, in a column) and the /neural-earth/
// demo page (full viewport) — the host page only has to size #earth.
import { mountNeuralEarth } from './neural-earth.js';

const WEIGHTS = 'https://huggingface.co/musedivision/neural-earth/resolve/main';
const SIZE = '358 MB';

const GATE_CSS = `
#earth .earth-poster { position: absolute; inset: 0; width: 100%; height: 100%; object-fit: cover; margin: 0; border: 0; opacity: .55; }
.earth-gate { position: absolute; inset: 0; display: flex; flex-direction: column; align-items: center; justify-content: center; gap: .8rem; text-align: center; padding: 1.5rem; font-family: 'Share Tech Mono', 'JetBrains Mono', Menlo, Consolas, monospace; }
.earth-start { font: inherit; font-size: 1rem; letter-spacing: .08em; text-transform: lowercase; color: #fff; background: rgba(0,0,0,.55); border: 1px solid rgba(255,255,255,.6); padding: .7rem 1.4rem; cursor: pointer; }
.earth-start:hover { background: rgba(255,255,255,.12); }
.earth-note { font-size: .8rem; color: rgba(255,255,255,.6); max-width: 28rem; line-height: 1.5; }
.earth-note code { font: inherit; color: rgba(255,255,255,.85); }
`;

const params = new URLSearchParams(location.search);
const weights = params.get('weights') || WEIGHTS;

// Why WebGPU is (un)available here — the answer is different for each cause, and
// "needs WebGPU" alone sends people to update a browser that already has it.
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

const root = document.getElementById('earth');
if (root) {
  const style = document.createElement('style');
  style.textContent = GATE_CSS;
  document.head.appendChild(style);

  const gate = document.createElement('div');
  gate.className = 'earth-gate';
  gate.innerHTML = '<span class="earth-note">checking for WebGPU…</span>';
  root.appendChild(gate);

  probeWebGPU().then(problem => {
    if (problem) {
      console.warn('[earth] WebGPU unavailable:', problem.replace(/<[^>]+>/g, ''));
      gate.innerHTML = `<span class="earth-note">${problem}</span>`;
      return;
    }
    gate.innerHTML = `<button type="button" class="earth-start">load the earth</button>
       <span class="earth-note">${SIZE} of neural-field weights, once — then cached in this browser</span>`;
    gate.querySelector('.earth-start').addEventListener('click', () => {
      root.querySelector('.earth-poster')?.remove();
      gate.remove();
      mountNeuralEarth(root, {
        weights, legs: 'coarse', quant: 'int8',
        brand: false, help: true,
        onError: (e) => console.error('[earth]', e),
      });
    });
  });
}
