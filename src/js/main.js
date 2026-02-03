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
  
  if (renderer) {
    renderer.start();

    document.addEventListener('visibilitychange', () => {
      if (document.hidden) {
        renderer.stop();
      } else {
        renderer.start();
      }
    });
  } else {
    console.error('WebGL initialization failed');
  }
}

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', init);
} else {
  init();
}
