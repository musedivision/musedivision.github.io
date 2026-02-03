precision highp float;

uniform float u_time;
uniform vec2 u_resolution;
uniform vec3 u_gyro; // alpha (compass), beta (front/back), gamma (left/right)

// Simplex noise functions
vec3 mod289(vec3 x) { return x - floor(x * (1.0 / 289.0)) * 289.0; }
vec2 mod289(vec2 x) { return x - floor(x * (1.0 / 289.0)) * 289.0; }
vec3 permute(vec3 x) { return mod289(((x*34.0)+1.0)*x); }

float snoise(vec2 v) {
  const vec4 C = vec4(0.211324865405187, 0.366025403784439,
                      -0.577350269189626, 0.024390243902439);
  vec2 i  = floor(v + dot(v, C.yy));
  vec2 x0 = v - i + dot(i, C.xx);
  vec2 i1;
  i1 = (x0.x > x0.y) ? vec2(1.0, 0.0) : vec2(0.0, 1.0);
  vec4 x12 = x0.xyxy + C.xxzz;
  x12.xy -= i1;
  i = mod289(i);
  vec3 p = permute(permute(i.y + vec3(0.0, i1.y, 1.0))
                   + i.x + vec3(0.0, i1.x, 1.0));
  vec3 m = max(0.5 - vec3(dot(x0, x0), dot(x12.xy, x12.xy),
                          dot(x12.zw, x12.zw)), 0.0);
  m = m*m;
  m = m*m;
  vec3 x = 2.0 * fract(p * C.www) - 1.0;
  vec3 h = abs(x) - 0.5;
  vec3 ox = floor(x + 0.5);
  vec3 a0 = x - ox;
  m *= 1.79284291400159 - 0.85373472095314 * (a0*a0 + h*h);
  vec3 g;
  g.x = a0.x * x0.x + h.x * x0.y;
  g.yz = a0.yz * x12.xz + h.yz * x12.yw;
  return 130.0 * dot(m, g);
}

// Fractal Brownian Motion
float fbm(vec2 p) {
  float value = 0.0;
  float amplitude = 0.5;
  float frequency = 1.0;
  
  for (int i = 0; i < 5; i++) {
    value += amplitude * snoise(p * frequency);
    amplitude *= 0.5;
    frequency *= 2.0;
  }
  
  return value;
}

// RGB to HSV
vec3 rgb2hsv(vec3 c) {
  vec4 K = vec4(0.0, -1.0 / 3.0, 2.0 / 3.0, -1.0);
  vec4 p = mix(vec4(c.bg, K.wz), vec4(c.gb, K.xy), step(c.b, c.g));
  vec4 q = mix(vec4(p.xyw, c.r), vec4(c.r, p.yzx), step(p.x, c.r));
  float d = q.x - min(q.w, q.y);
  float e = 1.0e-10;
  return vec3(abs(q.z + (q.w - q.y) / (6.0 * d + e)), d / (q.x + e), q.x);
}

// HSV to RGB
vec3 hsv2rgb(vec3 c) {
  vec4 K = vec4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
  vec3 p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
  return c.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);
}

void main() {
  vec2 uv = gl_FragCoord.xy / u_resolution.xy;
  vec2 p = uv * 3.0;
  
  // Animate the noise
  float t = u_time * 0.3;
  
  // Create flowing noise pattern
  float n1 = fbm(p + vec2(t, 0.0));
  float n2 = fbm(p + vec2(0.0, t * 0.7));
  float n3 = fbm(p + vec2(n1, n2) * 0.5);
  
  // Color palette - deep blues and purples
  vec3 color1 = vec3(0.04, 0.04, 0.1);   // Dark blue
  vec3 color2 = vec3(0.1, 0.05, 0.2);    // Purple
  vec3 color3 = vec3(0.2, 0.1, 0.4);     // Light purple
  vec3 color4 = vec3(0.4, 0.2, 0.6);     // Bright purple
  
  // Mix colors based on noise
  float blend = n3 * 0.5 + 0.5;
  vec3 color = mix(color1, color2, smoothstep(0.0, 0.4, blend));
  color = mix(color, color3, smoothstep(0.4, 0.7, blend));
  color = mix(color, color4, smoothstep(0.7, 1.0, blend));
  
  // Add subtle glow
  float glow = pow(blend, 3.0) * 0.3;
  color += vec3(0.3, 0.1, 0.5) * glow;
  
  // Apply gyro-based hue shift
  // Use alpha (compass direction 0-360) to rotate hue
  float hueShift = u_gyro.x / 360.0;
  vec3 hsv = rgb2hsv(color);
  hsv.x = fract(hsv.x + hueShift);
  color = hsv2rgb(hsv);
  
  // Vignette
  float vignette = 1.0 - length(uv - 0.5) * 0.8;
  color *= vignette;
  
  gl_FragColor = vec4(color, 1.0);
}
