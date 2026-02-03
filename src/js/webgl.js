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
    if (!this.uniforms[name]) {
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

    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
  }

  start() {
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
