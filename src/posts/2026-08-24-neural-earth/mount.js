
import { mountNeuralEarth } from './neural-earth.js';


const WEIGHTS = 'https://huggingface.co/musedivision/neural-earth/resolve/main';
const SIZE = '358 MB';

const params = new URLSearchParams(location.search);
const weights = params.get('weights') || WEIGHTS;

const root = document.getElementById('earth');
if (root) {
  const gate = document.createElement('div');
  gate.className = 'earth-gate';
  const supported = !!navigator.gpu;
  gate.innerHTML = supported
    ? `<button type="button" class="earth-start">load the earth</button>
       <span class="earth-note">${SIZE} of neural-field weights, once — then cached in this browser</span>`
    : `<span class="earth-note">needs WebGPU — Chrome / Edge 113+, Safari 26+ or Firefox 141+ on a real GPU</span>`;
  root.appendChild(gate);

  gate.querySelector('.earth-start')?.addEventListener('click', () => {
    root.querySelector('.earth-poster')?.remove();
    gate.remove();
    mountNeuralEarth(root, {
      weights, legs: 'coarse', quant: 'int8',
      brand: false, help: true,
      onError: (e) => console.error('[earth]', e),
    });
  });
}
