function clamp01(v) {
  return Math.max(0, Math.min(1, v));
}

export class WebGLRenderer {
  constructor(canvas) {
    this.canvas = canvas;
    this.gl = canvas.getContext('webgl2') || canvas.getContext('webgl');
    
    if (!this.gl) {
      console.error('WebGL not supported');
      return;
    }

    this.program = null;
    this.uniforms = {};
    this.startTime = performance.now();
    this.animationId = null;

    // Palette colors driven by the showcased image; current lerps toward target each frame.
    // Default is neutral dark — when an image palette arrives it lights up into those colors.
    const DEFAULT_PALETTE = [
      [0.02, 0.02, 0.03],
      [0.05, 0.05, 0.06],
      [0.09, 0.09, 0.10],
      [0.14, 0.14, 0.16],
    ];
    this.currentPalette = DEFAULT_PALETTE.map(c => [...c]);
    this.targetPalette = DEFAULT_PALETTE.map(c => [...c]);
    this.paletteBuffer = new Float32Array(12);
  }

  compileShader(source, type) {
    const gl = this.gl;
    const shader = gl.createShader(type);
    
    gl.shaderSource(shader, source);
    gl.compileShader(shader);

    if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
      console.error('Shader compile error:', gl.getShaderInfoLog(shader));
      gl.deleteShader(shader);
      return null;
    }

    return shader;
  }

  createProgram(vertexSource, fragmentSource) {
    const gl = this.gl;
    
    const vertexShader = this.compileShader(vertexSource, gl.VERTEX_SHADER);
    const fragmentShader = this.compileShader(fragmentSource, gl.FRAGMENT_SHADER);

    if (!vertexShader || !fragmentShader) {
      return null;
    }

    const program = gl.createProgram();
    gl.attachShader(program, vertexShader);
    gl.attachShader(program, fragmentShader);
    gl.linkProgram(program);

    if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
      console.error('Program link error:', gl.getProgramInfoLog(program));
      return null;
    }

    this.program = program;
    return program;
  }

  setupFullscreenQuad() {
    const gl = this.gl;
    
    const vertices = new Float32Array([
      -1, -1,
       1, -1,
      -1,  1,
       1,  1,
    ]);

    const buffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
    gl.bufferData(gl.ARRAY_BUFFER, vertices, gl.STATIC_DRAW);

    const positionLoc = gl.getAttribLocation(this.program, 'a_position');
    gl.enableVertexAttribArray(positionLoc);
    gl.vertexAttribPointer(positionLoc, 2, gl.FLOAT, false, 0, 0);
  }

  getUniformLocation(name) {
    if (!(name in this.uniforms)) {
      this.uniforms[name] = this.gl.getUniformLocation(this.program, name);
    }
    return this.uniforms[name];
  }

  resize() {
    const canvas = this.canvas;
    const dpr = window.devicePixelRatio || 1;
    const displayWidth = Math.floor(canvas.clientWidth * dpr);
    const displayHeight = Math.floor(canvas.clientHeight * dpr);

    if (canvas.width !== displayWidth || canvas.height !== displayHeight) {
      canvas.width = displayWidth;
      canvas.height = displayHeight;
      this.gl.viewport(0, 0, canvas.width, canvas.height);
    }
  }

  setPalette(palette) {
    if (!Array.isArray(palette) || palette.length < 4) return;
    for (let i = 0; i < 4; i++) {
      this.targetPalette[i] = [
        clamp01(palette[i][0]),
        clamp01(palette[i][1]),
        clamp01(palette[i][2]),
      ];
    }
  }

  render() {
    const gl = this.gl;
    
    this.resize();
    
    gl.useProgram(this.program);

    const timeLoc = this.getUniformLocation('u_time');
    if (timeLoc) {
      gl.uniform1f(timeLoc, (performance.now() - this.startTime) / 1000);
    }

    const resolutionLoc = this.getUniformLocation('u_resolution');
    if (resolutionLoc) {
      gl.uniform2f(resolutionLoc, this.canvas.width, this.canvas.height);
    }

    // Lerp current palette toward target (~1.5s asymptotic convergence at 60fps)
    const k = 0.03;
    for (let i = 0; i < 4; i++) {
      for (let c = 0; c < 3; c++) {
        this.currentPalette[i][c] += (this.targetPalette[i][c] - this.currentPalette[i][c]) * k;
        this.paletteBuffer[i * 3 + c] = this.currentPalette[i][c];
      }
    }
    const paletteLoc = this.getUniformLocation('u_palette[0]');
    if (paletteLoc) {
      gl.uniform3fv(paletteLoc, this.paletteBuffer);
    }

    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
  }

  start() {
    if (this.animationId !== null) return; // already running — don't stack loops
    const loop = () => {
      this.render();
      this.animationId = requestAnimationFrame(loop);
    };
    loop();
  }

  stop() {
    if (this.animationId) {
      cancelAnimationFrame(this.animationId);
      this.animationId = null;
    }
  }
}

export function initWebGL(canvas, vertexSource, fragmentSource) {
  const renderer = new WebGLRenderer(canvas);
  
  if (!renderer.gl) {
    return null;
  }

  const program = renderer.createProgram(vertexSource, fragmentSource);
  if (!program) {
    return null;
  }
  
  renderer.gl.useProgram(renderer.program);
  renderer.setupFullscreenQuad();
  
  return renderer;
}
