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

  // Set up gyroscope
  function handleOrientation(e) {
    renderer.setGyro(e.alpha, e.beta, e.gamma);
  }

  function enableGyro() {
    window.addEventListener('deviceorientation', handleOrientation);
  }

  // Check if we need to request permission (iOS 13+)
  if (typeof DeviceOrientationEvent !== 'undefined' && 
      typeof DeviceOrientationEvent.requestPermission === 'function') {
    // iOS - need user gesture to request permission
    const requestBtn = document.createElement('button');
    requestBtn.textContent = 'enter';
    requestBtn.style.cssText = `
      position: fixed;
      bottom: 3rem;
      left: 50%;
      transform: translateX(-50%);
      z-index: 10;
      padding: 1rem 3rem;
      font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
      font-size: 1.2rem;
      font-weight: 300;
      letter-spacing: 0.3em;
      text-transform: lowercase;
      background: transparent;
      color: rgba(255,255,255,0.7);
      border: 1px solid rgba(255,255,255,0.2);
      border-radius: 0;
      cursor: pointer;
      transition: all 0.3s ease;
    `;
    document.body.appendChild(requestBtn);

    // Hover effect
    requestBtn.addEventListener('mouseenter', () => {
      requestBtn.style.color = 'rgba(255,255,255,1)';
      requestBtn.style.borderColor = 'rgba(255,255,255,0.5)';
      requestBtn.style.background = 'rgba(255,255,255,0.05)';
    });
    requestBtn.addEventListener('mouseleave', () => {
      requestBtn.style.color = 'rgba(255,255,255,0.7)';
      requestBtn.style.borderColor = 'rgba(255,255,255,0.2)';
      requestBtn.style.background = 'transparent';
    });

    requestBtn.addEventListener('click', () => {
      DeviceOrientationEvent.requestPermission()
        .then(permission => {
          if (permission === 'granted') {
            enableGyro();
          }
          // Fade out and remove button regardless of permission
          requestBtn.style.opacity = '0';
          requestBtn.style.transition = 'opacity 0.5s ease';
          setTimeout(() => requestBtn.remove(), 500);
        })
        .catch(err => {
          console.error(err);
          requestBtn.style.opacity = '0';
          setTimeout(() => requestBtn.remove(), 500);
        });
    });
  } else if ('DeviceOrientationEvent' in window) {
    // Android - just enable it, no button needed
    enableGyro();
  }

  // Desktop: use mouse position to simulate gyro
  canvas.addEventListener('mousemove', (e) => {
    const rect = canvas.getBoundingClientRect();
    const x = (e.clientX - rect.left) / rect.width;
    const y = (e.clientY - rect.top) / rect.height;
    
    const alpha = x * 360;
    const beta = (y - 0.5) * 180;
    const gamma = 0;
    
    renderer.setGyro(alpha, beta, gamma);
  });

  document.addEventListener('visibilitychange', () => {
    if (document.hidden) {
      renderer.stop();
    } else {
      renderer.start();
    }
  });
}

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', init);
} else {
  init();
}
