import { initWebGL } from './webgl.js';
import vertexShader from '../shaders/vertex.glsl';
import fragmentShader from '../shaders/fragment.glsl';

function init() {
  const canvas = document.getElementById('webgl-canvas');
  
  if (!canvas) {
    console.error('Canvas not found');
    return;
  }

  const renderer = initWebGL(canvas, vertexShader, fragmentShader);
  
  if (!renderer) {
    console.error('WebGL initialization failed');
    return;
  }

  renderer.start();

  window.addEventListener('palette', (e) => {
    if (e.detail?.palette) renderer.setPalette(e.detail.palette);
  });

  document.addEventListener('visibilitychange', () => {
    if (document.hidden) {
      renderer.stop();
    } else {
      renderer.start();
    }
  });

  // Kill the render loop the instant a navigation starts, so the shader stops
  // competing with the browser for the main thread during the page transition.
  document.addEventListener('click', (e) => {
    const a = e.target.closest?.('a[href]');
    if (a && a.origin === location.origin && !a.target && !a.hasAttribute('download')) {
      renderer.stop();
    }
  }, true);
  window.addEventListener('pagehide', () => renderer.stop());

  // Restart if the page is restored from the back/forward cache.
  window.addEventListener('pageshow', (e) => {
    if (e.persisted) renderer.start();
  });
}

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', init);
} else {
  init();
}
