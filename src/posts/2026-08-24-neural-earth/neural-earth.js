class Ee{static async create(e,t={}){if(!navigator.gpu)throw new Error("WebGPU is not available in this browser. Use Chrome / Edge 113+ (or Safari 26+, Firefox 141+) on a real GPU.");const s=t.forceFallbackAdapter?{forceFallbackAdapter:!0}:{powerPreference:t.powerPreference||"high-performance"},r=await navigator.gpu.requestAdapter(s);if(!r)throw new Error(`No WebGPU adapter for ${JSON.stringify(s)}.`);const A=r.limits,a=[];r.features.has("timestamp-query")&&a.push("timestamp-query");const g=await r.requestDevice({requiredFeatures:a,requiredLimits:{maxStorageBufferBindingSize:A.maxStorageBufferBindingSize,maxBufferSize:A.maxBufferSize,maxStorageBuffersPerShaderStage:A.maxStorageBuffersPerShaderStage,maxUniformBufferBindingSize:A.maxUniformBufferBindingSize}}),o=e.getContext("webgpu"),u=navigator.gpu.getPreferredCanvasFormat();o.configure({device:g,format:u,alphaMode:"opaque"});let E={};try{E=r.info||(r.requestAdapterInfo?await r.requestAdapterInfo():{})}catch{}const c=E.description||[E.vendor,E.architecture].filter(Boolean).join(" ")||"unknown GPU";return console.log("[WebGPU] adapter:",c,E,"limits:",{bindingSize:A.maxStorageBufferBindingSize,bufferSize:A.maxBufferSize,storageBuffers:A.maxStorageBuffersPerShaderStage,timestamps:a.includes("timestamp-query")}),new Ee(g,o,u,r,E,c,!!t.forceFallbackAdapter)}constructor(e,t,s,r,A,a,g){this.device=e,this.ctx=t,this.format=s,this.adapter=r,this.info=A,this.gpuName=a,this.isFallback=g,this.hasTimestamps=e.features.has("timestamp-query"),this.onLost=null,e.lost.then(o=>{o.reason!=="destroyed"&&(console.error("WebGPU device lost:",o.message),this.onLost?.(o))})}get limits(){return this.device.limits}get maxGridBuffers(){return this.device.limits.maxStorageBuffersPerShaderStage-1}}function Ge(i,e){if(i.bits)return i.bits;const t=e?.grid?.dtype??"float16";return t==="int8"?8:t==="int4"?4:16}function Fe(i,e){return(i.stride_vertices??i.size_vertices)*2*Ge(i,e)/32}function Ve(i,e,t=null){const s=[],r=[];let A=null,a=0;for(const g of i){const o=Fe(g,t);if(o>e)throw new Error(`level of ${o} words exceeds buffer cap ${e}`);(!A||A.wordCount+o>e)&&(A={startWord:a,wordCount:0},s.push(A)),r.push({bufIdx:s.length-1,localBase:a-A.startWord}),A.wordCount+=o,a+=o}return{buffers:s,placement:r}}const Ue="neural-earth-weights-v1",ne=8*1024*1024;class q{static async fetchManifest(e,t,s){const r=await fetch(`${e}/${t}.manifest.json`,{signal:s});if(!r.ok)throw new Error(`fetch ${t}.manifest.json: ${r.status}`);return r.json()}static plan(e,t){const s=e.grid?.dtype??"float16";if(!["float16","int8","int4"].includes(s))throw new Error(`unknown grid dtype ${s}`);const r=e.grid?.block_vertices??64;let A=Math.floor(Math.min(t.limits.maxStorageBufferBindingSize,t.limits.maxBufferSize)/4);A-=A%r;const{buffers:a,placement:g}=Ve(e.levels,A,e),o=e.levels.reduce((u,E)=>u+Fe(E,e),0)*4;return{buffers:a,placement:g,totalBytes:o,quantized:s!=="float16"}}static async loadLeg(e,t,s,r={}){const{onProgress:A=()=>{},signal:a,useCache:g=!0}=r,o=r.manifest||await q.fetchManifest(t,s,a),{buffers:u,placement:E,totalBytes:c,quantized:f}=q.plan(o,e),w=o.mlp?.file||`${s}.mlp.f32.bin`,l=o.grid?.file||`${s}.grid.f16.bin`;A(0,"mlp");const b=await fetch(`${t}/${w}`,{signal:a});if(!b.ok)throw new Error(`fetch ${w}: ${b.status}`);const F=new Float32Array(await b.arrayBuffer());let S=null;if(f){const Q=await fetch(`${t}/${o.grid.scales_file}`,{signal:a});if(!Q.ok)throw new Error(`fetch ${o.grid.scales_file}: ${Q.status}`);S=new Uint32Array(await Q.arrayBuffer())}let M=0;const d=u.map((Q,x)=>{const U=Q.wordCount*4,n={gpu:e.createBuffer({label:`${s} grid ${x}`,size:U,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST}),startByte:M,endByte:M+U};return M+=U,n});if(M!==c)throw new Error("grid plan does not tile the blob");const C=new Uint8Array(ne);let m=0,G=0,h=0;const v=()=>{let Q=0;for(;Q<m;){const x=G+Q;for(;d[h].endByte<=x;)h++;const U=d[h],n=Math.min(m-Q,U.endByte-x);e.queue.writeBuffer(U.gpu,x-U.startByte,C,Q,n),Q+=n}G+=m,m=0};try{if(await Xe(`${t}/${l}`,{signal:a,useCache:g,expectBytes:c,onChunk:Q=>{let x=0;for(;x<Q.length;){const U=Math.min(Q.length-x,ne-m);C.set(Q.subarray(x,x+U),m),m+=U,x+=U,m===ne&&v()}},onProgress:Q=>A(Q,"grid")}),m%4!==0)throw new Error("grid blob length is not a multiple of 4");if(v(),G!==c)throw new Error(`grid blob truncated: got ${G} of ${c} bytes`)}catch(Q){throw d.forEach(x=>x.gpu.destroy()),Q}return A(1,"done"),{stem:s,manifest:o,mlp:F,quantized:f,scales:S,gridBuffers:d.map(Q=>Q.gpu),placement:E,nGridBuffers:u.length,bytes:c+F.byteLength+(S?S.byteLength:0),dispose(){this.gridBuffers.forEach(Q=>Q.destroy()),this.gridBuffers=[]}}}}async function Xe(i,{signal:e,useCache:t,expectBytes:s,onChunk:r,onProgress:A}){let a=null,g=null;if(t&&typeof caches<"u")try{g=await caches.open(Ue),a=await g.match(i),a&&console.log("[weights] cache hit",i)}catch{g=null}if(!a){if(a=await fetch(i,{signal:e}),!a.ok)throw new Error(`fetch ${i}: ${a.status}`);g&&await et(s)&&g.put(i,a.clone()).catch(c=>console.warn("[weights] cache put failed",c))}const o=Number(a.headers.get("content-length"))||s||0;if(!a.body){const c=new Uint8Array(await a.arrayBuffer());r(c),A(1);return}const u=a.body.getReader();let E=0;for(;;){const{done:c,value:f}=await u.read();if(c)break;r(f),E+=f.byteLength,o&&A(Math.min(1,E/o))}}async function et(i){try{const{quota:e=0,usage:t=0}=await navigator.storage.estimate();return e-t>i*1.2}catch{return!1}}async function tt(){try{return await caches.delete(Ue)}catch{return!1}}const ke=720,Se=1440,He="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BwThAQ8GDwsBBtABAgWyBAYJ5AIHAgEB4wEOBBMIBQEBAdEBAgaxBAcI4gIJAgIB4wEKAxgEAQThAa8EBgjgAgoD6AEUAQ8FAQHJAQMarAQHBy0BsgIJAoYCCQTNAQcUrAQHBjABrwIEAQEBjQIIBM4BBgQBAgELrAQGBzACsgICAQMBAgGHAggEzgEMAQgDrAQGBy8BsgKVAgcEzgEWBqgEBgU1Aa0CvQEDVwYE0gEQAQEHAgKiBAUGNARXAdEBAQG9AQQBAli7AQUSBgEODKQEAwY1Bl8BAwHDAb4BCFu2AQoLGgulBAIGNQdVARcCAQGpAQELAQK4AQxbtQECAQkLGQulBAEGNQhtAbYBAgG5AQxbuAENChcLowQCBjQJPAE+AZoBAgkBAbwBDFy4AQIBCwgYCwMCngQCBjQJlQICCr0BAgMIXLkBDgUBARIBBA0BAZ8EAwY0CTkCSAGQAQMLuwEBBAQBA126AQ0HEhIBAp8EAgU0DYEBBYsBAgcBAb4BDUsBELkBDgcBBA4WnQQDAgEDMxCDAQOGAQIBAQcFAboBDUsDCAEEugEODQsBARwDAoMEAwEBCAgCNBCEAQEBAYQBBgTCAQ1PAQe9AQEBDQ0LIwECEgEBAugDAgICAQEGPw+JAQIBAYABAgq/AQ1QAQIBAr0BEQUBBQEBCyMJAwgG5gMBAQYFCgIzEZYCwAEMVsUBDQgMJAgCAQgCCgoE1ANME4sBAwEBBAF/wQEKWcMBDQgKJwUcBgUBAc4DAgJMGRcBbgQEAXUBAgECAQHDAQsBAVACAsYBCwIQJgUBAQEBAgIBASDNAwgDRhoRAXoDCQJqAQECAgEBwgERSAEDzAEEAQQEDikDAwIEAgoCGMoDBwFHGREBfQNwAQEEAQIBAgG1AR5G0wEBBgEDDSkDAQQCBQICH8wDTRmQAQcDAWgGAQYBsQEiQtwBAgIMJwcFBwEFG88DSxmSAQMDBAEBDAFWAQIEAQMBsQEbAQs/AQHcAQEBDSkIAQEDAQECAQYb0QNKGZYBCGEBAQMBAgGxASAECD0BAd4BCiwJCQkCAQEBFNMDSBoCApIBB14BAgEEAQKxASEBAQEKPN0BAgELKwwHDRTUA0celQEGAQFZAwEEAQIBrwEwOgEBAQHbAQMBCywHCw0BARHVA0cdmAEHDAFJAwIFAa4BMzvgAQ4qCAkQ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";function it(){const i=typeof atob=="function"?Uint8Array.from(atob(He),a=>a.charCodeAt(0)):new Uint8Array(Buffer.from(He,"base64")),e=ke*Se,t=new Uint32Array(Math.ceil(e/32));let s=0,r=0,A=0;for(;s<i.length;){let a=0,g=0,o;do o=i[s++],a|=(o&127)<<g,g+=7;while(o&128);if(A)for(let u=r;u<r+a;u++)t[u>>>5]|=1<<(u&31);r+=a,A^=1}if(r!==e)throw new Error(`land mask: decoded ${r} bits, expected ${e}`);return t}const ge={NLEV:16,MAX_LEGS:2},te=64*32+64*64+7*64,Be=8,Le=4+16*3,ce=(Be+ge.MAX_LEGS*Le)*16,le=3,he=64,Pe=`
struct Level {
  sg : vec4<f32>,   
  m  : vec4<u32>,   
  b  : vec4<u32>,   
};
struct Leg {
  dem  : vec4<f32>,  
  dc0  : vec4<f32>,  
  dc1  : vec4<f32>,  
  info : vec4<u32>,  
  levels : array<Level, 16>,
};
struct Params {
  rt0 : vec4<f32>, rt1 : vec4<f32>, rt2 : vec4<f32>,
  cam   : vec4<f32>,  
  opt   : vec4<f32>,  
  route : vec4<f32>,  
  refine : vec4<f32>, 
  pick  : vec4<f32>,  
  legs  : array<Leg, 2>,
};

const TWO_PI : f32 = 6.283185307179586;
const TOL_HI : f32 = 1.2;
const TOL_LO : f32 = 3;
const SPACE : vec3<f32> = vec3<f32>(0.02, 0.03, 0.06);


fn rayDir(pxf : vec2<f32>) -> vec3<f32> {
  let aspect = P.cam.z / P.cam.w;
  let halfH = tan(P.cam.y * 0.5);
  let halfW = halfH * aspect;
  let gx = -halfW + (pxf.x / max(P.cam.z - 1.0, 1.0)) * 2.0 * halfW;
  let gy = halfH - (pxf.y / max(P.cam.w - 1.0, 1.0)) * 2.0 * halfH;
  return normalize(vec3<f32>(gx, gy, -1.0));
}

fn rt() -> mat3x3<f32> {
  
  return mat3x3<f32>(
    vec3<f32>(P.rt0.x, P.rt1.x, P.rt2.x),
    vec3<f32>(P.rt0.y, P.rt1.y, P.rt2.y),
    vec3<f32>(P.rt0.z, P.rt1.z, P.rt2.z),
  );
}
`;function st(i){const e=Array.from({length:i},(g,o)=>`@group(0) @binding(${le+o}) var<storage, read> g${o} : array<u32>;`).join(`
`),t=Array.from({length:i},(g,o)=>`    case ${o}u: { return g${o}[idx]; }`).join(`
`),s=le+i,r=s+1,A=s+2,a=s+3;return`
${Pe}

@group(0) @binding(0) var<uniform> P : Params;
@group(0) @binding(1) var<storage, read> mlp : array<f32>;
@group(0) @binding(2) var<storage, read> scales : array<u32>; 
${e}
@group(0) @binding(${s}) var outTex : texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(${r}) var<storage, read> landMask : array<u32>; 
@group(0) @binding(${A}) var<storage, read> fineBoxes : array<vec4<f32>>; 
@group(0) @binding(${a}) var<storage, read_write> sigmaOut : array<f32>;

const MASK_N_LAT : u32 = ${ke}u;
const MASK_N_LON : u32 = ${Se}u;

var<workgroup> wgFlag  : array<u32, 64>;
var<workgroup> wgLand  : array<u32, 64>;
var<workgroup> wgOther : array<u32, 64>;

const PRIME1 : u32 = 2654435761u;
const PRIME2 : u32 = 805459861u;

fn fetchVtx(bufIdx : u32, idx : u32) -> u32 {
  switch (bufIdx) {
${t}
    default: { return 0u; }
  }
}


fn erf_pos(x : f32) -> f32 {
  let t = 1.0 / (1.0 + 0.3275911 * x);
  let y = 1.0 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t
            - 0.284496736) * t + 0.254829592) * t * exp(-x * x);
  return y;
}


fn hashCell(p : vec2<u32>) -> u32 {
  var h = (p.x * 0x9E3779B1u) ^ (p.y * 0x85EBCA77u);
  h = h ^ (h >> 16u); h = h * 0x7FEB352Du; h = h ^ (h >> 15u); h = h * 0x846CA68Bu; h = h ^ (h >> 16u);
  return h;
}

fn gridIndex(pg : vec3<u32>, lvl : Level) -> u32 {
  let size = lvl.m.x;
  if (lvl.m.z == 1u) { 
    let res = lvl.m.y;
    let idx = pg.x + pg.y * res + pg.z * res * res;
    return idx % size;
  }
  let h = (pg.x * 1u) ^ (pg.y * PRIME1) ^ (pg.z * PRIME2);
  return h % size;
}


fn encodeGate(L : u32, coord : vec3<f32>, sigma : f32, feat : ptr<function, array<f32, 32>>) {
  let s = max(abs(sigma) * P.legs[L].dem.z, 1e-12);
  for (var l = 0u; l < 16u; l = l + 1u) {
    let lvl = P.legs[L].levels[l];
    
    
    
    let g = erf_pos(1.0 / (2.8284271247461903 * s * lvl.sg.y));
    if (g < 1e-3) { (*feat)[2u * l] = 0.0; (*feat)[2u * l + 1u] = 0.0; continue; }
    let scale = lvl.sg.x;
    let p = coord * scale + vec3<f32>(0.5);
    let fl = floor(p);
    let frac = p - fl;
    let base = vec3<u32>(vec3<i32>(fl));
    var acc = vec2<f32>(0.0);
    for (var c = 0u; c < 8u; c = c + 1u) {
      let cx = c & 1u; let cy = (c >> 1u) & 1u; let cz = (c >> 2u) & 1u;
      let wx = select(1.0 - frac.x, frac.x, cx == 1u);
      let wy = select(1.0 - frac.y, frac.y, cy == 1u);
      let wz = select(1.0 - frac.z, frac.z, cz == 1u);
      let w = wx * wy * wz;
      let pg = base + vec3<u32>(cx, cy, cz);
      let local = gridIndex(pg, lvl);       
      var two : vec2<f32>;
      let quant = lvl.b.z;
      if (quant == 1u) {
        
        let q = unpack4x8snorm(fetchVtx(lvl.m.w, lvl.b.x + (local >> 1u)));
        let pair = select(q.xy, q.zw, (local & 1u) == 1u);
        two = pair * unpack2x16float(scales[lvl.b.y + (local >> 6u)]);
      } else if (quant == 2u) {
        
        let byte = (fetchVtx(lvl.m.w, lvl.b.x + (local >> 2u)) >> ((local & 3u) * 8u)) & 0xFFu;
        let q0 = (i32(byte & 0xFu) << 28u) >> 28u;      
        let q1 = (i32(byte >> 4u) << 28u) >> 28u;
        two = vec2<f32>(f32(q0), f32(q1)) * (unpack2x16float(scales[lvl.b.y + (local >> 6u)]) / 7.0);
      } else {
        two = unpack2x16float(fetchVtx(lvl.m.w, lvl.b.x + local));
      }
      acc = acc + w * two;
    }
    (*feat)[2u * l] = acc.x * g;
    (*feat)[2u * l + 1u] = acc.y * g;
  }
}


fn runMLP(mb : u32, feat : ptr<function, array<f32, 32>>, out : ptr<function, array<f32, 7>>) {
  var h0 : array<f32, 64>;
  for (var j = 0u; j < 64u; j = j + 1u) {
    var a = 0.0;
    for (var k = 0u; k < 32u; k = k + 1u) { a = a + mlp[mb + j * 32u + k] * (*feat)[k]; }
    h0[j] = max(a, 0.0);
  }
  var h1 : array<f32, 64>;
  let off1 = mb + 2048u; 
  for (var j = 0u; j < 64u; j = j + 1u) {
    var a = 0.0;
    for (var k = 0u; k < 64u; k = k + 1u) { a = a + mlp[off1 + j * 64u + k] * h0[k]; }
    h1[j] = max(a, 0.0);
  }
  let off2 = mb + 6144u; 
  for (var o = 0u; o < 7u; o = o + 1u) {
    var a = 0.0;
    for (var k = 0u; k < 64u; k = k + 1u) { a = a + mlp[off2 + o * 64u + k] * h1[k]; }
    (*out)[o] = a;
  }
}

fn terrain(t0 : f32) -> vec3<f32> {
  let t = clamp(t0, 0.0, 1.0);
  var k = array<vec3<f32>, 9>(
    vec3<f32>(0.200, 0.200, 0.600), vec3<f32>(0.033, 0.535, 0.935),
    vec3<f32>(0.004, 0.801, 0.401), vec3<f32>(0.506, 0.901, 0.501),
    vec3<f32>(0.996, 0.995, 0.598), vec3<f32>(0.745, 0.674, 0.462),
    vec3<f32>(0.506, 0.368, 0.338), vec3<f32>(0.757, 0.689, 0.674),
    vec3<f32>(1.000, 1.000, 1.000),
  );
  let seg = clamp(floor(t / 0.125), 0.0, 7.0);
  let i = u32(seg);
  let local = (t - seg * 0.125) / 0.125;
  return mix(k[i], k[i + 1u], local);
}





fn isLand(pxf : vec2<f32>) -> u32 {
  let dir = rayDir(pxf);
  let camDist = P.cam.x;
  let b = 2.0 * dir.z * camDist;
  let disc = b * b - 4.0 * (camDist * camDist - 1.0);
  if (disc < 0.0) { return 0u; }
  let t = (-b - sqrt(disc)) * 0.5;
  let m = rt() * (vec3<f32>(0.0, 0.0, camDist) + t * dir);
  let lat = degrees(asin(clamp(m.z, -1.0, 1.0)));
  let lon = degrees(atan2(m.y, m.x));
  let r = min(u32((lat + 90.0) * f32(MASK_N_LAT) / 180.0), MASK_N_LAT - 1u);
  let c = min(u32((lon + 180.0) * f32(MASK_N_LON) / 360.0), MASK_N_LON - 1u);
  let i = r * MASK_N_LON + c;
  return (landMask[i >> 5u] >> (i & 31u)) & 1u;
}


fn inFineCoverage(m : vec3<f32>) -> bool {
  let nBox = u32(P.route.w);
  if (nBox == 0u) { return true; }
  let lat = degrees(asin(clamp(m.z, -1.0, 1.0)));
  let lon = degrees(atan2(m.y, m.x));
  for (var i = 0u; i < nBox; i = i + 1u) {
    let b = fineBoxes[i];
    if (lat >= b.x && lat <= b.y && lon >= b.z && lon <= b.w) { return true; }
  }
  return false;
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid : vec3<u32>,
        @builtin(local_invocation_index) lid : u32,
        @builtin(workgroup_id) wid : vec3<u32>) {
  let W = u32(P.cam.z);
  let H = u32(P.cam.w);
  
  
  
  
  
  
  
  
  
  let S = max(u32(P.refine.x), 1u);
  var px = gid.xy;
  if (S > 1u) {
    let n = S * S;
    var land = 0u;
    if (gid.x * S < W && gid.y * S < H) { land = isLand((vec2<f32>(gid.xy) + 0.5) * f32(S) - 0.5); }
    wgFlag[lid] = land;
    workgroupBarrier();
    
    var nLand = 0u; var rank = 0u;
    for (var i = 0u; i < 64u; i = i + 1u) {
      let f = wgFlag[i];
      nLand = nLand + f;
      if (i < lid) { rank = rank + select(1u - f, f, land == 1u); }
    }
    if (land == 1u) { wgLand[rank] = lid; } else { wgOther[rank] = lid; }
    workgroupBarrier();
    
    let item = u32(P.refine.y) * 64u + lid;
    let landItems = nLand * n;
    var cellLid = 0u; var j = 0u;
    if (item < landItems) { cellLid = wgLand[item % nLand]; j = item / nLand; }
    else { let o = item - landItems; let nOther = 64u - nLand; cellLid = wgOther[o % nOther]; j = o / nOther; }
    let cell = wid.xy * 8u + vec2<u32>(cellLid % 8u, cellLid / 8u);
    let h = hashCell(cell);
    let jp = (j * (2u * h + 1u) + h) & (n - 1u); 
    px = cell * S + vec2<u32>(jp % S, jp / S);
  }
  if (px.x >= W || px.y >= H) { return; }
  
  
  let surface = P.refine.w > 0.5;
  let pixIdx = px.y * W + px.x;

  let camDist = P.cam.x;
  let fov = P.cam.y;
  let dir = rayDir(vec2<f32>(px));

  let b = 2.0 * dir.z * camDist;
  let cc = camDist * camDist - 1.0;
  let disc = b * b - 4.0 * cc;

  var col = SPACE;
  if (disc < 0.0 && surface && S > 1u) { return; } 
  if (disc >= 0.0) {
    let t = (-b - sqrt(disc)) * 0.5;
    let hp = vec3<f32>(0.0, 0.0, camDist) + t * dir;
    let facing = clamp(dot(hp, -dir), 0.0, 1.0);
    
    
    
    
    
    
    let sigmaRaw = (fov / max(P.cam.w - 1.0, 1.0)) * t / max(facing, 1e-2) / TWO_PI;
    let sigma = sigmaRaw * P.opt.x;
    if (surface && S > 1u) {
      let s0 = sigmaOut[pixIdx];
      if (s0 > 0.0 && s0 <= sigmaRaw * TOL_HI && s0 >= sigmaRaw / TOL_LO) { return; } 
    }
    let model = rt() * hp;
    let coord = (model + vec3<f32>(1.0)) * 0.5;

    
    var L = 0u;
    let cov = inFineCoverage(model);
    if (P.route.y > 1.5) {
      if (P.route.z >= 0.0) { L = u32(P.route.z); }
      else if (sigmaRaw < P.route.x && cov) { L = 1u; }
    }
    let leg = P.legs[L];

    var feat : array<f32, 32>;
    encodeGate(L, coord, sigma, &feat);
    var o : array<f32, 7>;
    runMLP(leg.info.x, &feat, &o);
    if (surface) { sigmaOut[pixIdx] = sigmaRaw; }

    let rgb = clamp(vec3<f32>(o[0], o[1], o[2]), vec3<f32>(0.0), vec3<f32>(1.0));
    var demM : f32; var dx : f32; var dy : f32;
    if (leg.dc0.x > 0.5) {
      let hh = leg.dc0.y * tanh(o[3]);
      let offM = o[4] * leg.dc0.w + leg.dc0.z;
      let scM = exp(o[5] * leg.dc1.y + leg.dc1.x);
      demM = offM + scM * hh; dx = 0.0; dy = 0.0;
    } else {
      demM = o[3] * leg.dem.x; dx = o[4] * leg.dem.y; dy = o[5] * leg.dem.y;
    }
    let tree = clamp(o[6], 0.0, 1.0);

    let sun = normalize(vec2<f32>(0.5, 0.5));
    let slope = -(dx * sun.x + dy * sun.y);
    let fade = exp(-abs(sigmaRaw) / 1e-4);
    let factor = 1.0 + 0.4 * tanh(slope / 20.0) * fade;

    let mode = u32(P.opt.y);
    if (mode == 0u) { col = rgb; }
    else if (mode == 1u) {
      let nrm = clamp((demM - P.opt.z) / (P.opt.w - P.opt.z), 0.0, 1.0);
      col = terrain(nrm) * factor;
    } else { col = vec3<f32>(0.0, tree, 0.0); }

    
    if (P.refine.z > 0.5) {
      if (P.route.w > 0.5 && cov) { col = mix(col, vec3<f32>(1.0, 0.55, 0.1), 0.35); }
      if (P.pick.w > 0.5) {
        let d = acos(clamp(dot(model, P.pick.xyz), -1.0, 1.0));
        let pxAng = sigmaRaw * TWO_PI;
        if (d < 7.0 * pxAng) { col = select(vec3<f32>(1.0), vec3<f32>(0.0), d > 5.0 * pxAng); }
      }
    }
  }

  textureStore(outTex, vec2<i32>(i32(px.x), i32(px.y)), vec4<f32>(col, 1.0));
}
`}const rt=`
${Pe}
struct Reproj {
  prt0 : vec4<f32>, prt1 : vec4<f32>, prt2 : vec4<f32>, 
  misc : vec4<f32>,  
};


const RESAMPLE_BLUR : f32 = 0.3;
@group(0) @binding(0) var<uniform> P : Params;
@group(0) @binding(1) var<uniform> R : Reproj;
@group(0) @binding(2) var histTex : texture_2d<f32>;
@group(0) @binding(3) var smp : sampler;
@group(0) @binding(4) var<storage, read> sigmaPrev : array<f32>;
@group(0) @binding(5) var loTex : texture_2d<f32>;
@group(0) @binding(6) var outTex : texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(7) var<storage, read_write> sigmaCur : array<f32>;
@group(0) @binding(8) var<storage, read_write> holeCount : array<atomic<u32>>;
var<workgroup> wgHoles : atomic<u32>;

fn rtPrev() -> mat3x3<f32> {
  return mat3x3<f32>(
    vec3<f32>(R.prt0.x, R.prt1.x, R.prt2.x),
    vec3<f32>(R.prt0.y, R.prt1.y, R.prt2.y),
    vec3<f32>(R.prt0.z, R.prt1.z, R.prt2.z),
  );
}



fn fetchHistory(p : vec2<f32>, size : vec2<f32>) -> vec3<f32> {
  let sp = p + 0.5;                       
  let tc1 = floor(sp - 0.5) + 0.5;
  let f = sp - tc1;
  let w0 = f * (-0.5 + f * (1.0 - 0.5 * f));
  let w1 = 1.0 + f * f * (-2.5 + 1.5 * f);
  let w2 = f * (0.5 + f * (2.0 - 1.5 * f));
  let w3 = f * f * (-0.5 + 0.5 * f);
  let w12 = w1 + w2;
  let tc0 = (tc1 - 1.0) / size;
  let tc3 = (tc1 + 2.0) / size;
  let tc12 = (tc1 + w2 / w12) / size;
  var c = vec3<f32>(0.0);
  c += textureSampleLevel(histTex, smp, vec2<f32>(tc0.x, tc0.y), 0.0).rgb * w0.x * w0.y;
  c += textureSampleLevel(histTex, smp, vec2<f32>(tc12.x, tc0.y), 0.0).rgb * w12.x * w0.y;
  c += textureSampleLevel(histTex, smp, vec2<f32>(tc3.x, tc0.y), 0.0).rgb * w3.x * w0.y;
  c += textureSampleLevel(histTex, smp, vec2<f32>(tc0.x, tc12.y), 0.0).rgb * w0.x * w12.y;
  c += textureSampleLevel(histTex, smp, vec2<f32>(tc12.x, tc12.y), 0.0).rgb * w12.x * w12.y;
  c += textureSampleLevel(histTex, smp, vec2<f32>(tc3.x, tc12.y), 0.0).rgb * w3.x * w12.y;
  c += textureSampleLevel(histTex, smp, vec2<f32>(tc0.x, tc3.y), 0.0).rgb * w0.x * w3.y;
  c += textureSampleLevel(histTex, smp, vec2<f32>(tc12.x, tc3.y), 0.0).rgb * w12.x * w3.y;
  c += textureSampleLevel(histTex, smp, vec2<f32>(tc3.x, tc3.y), 0.0).rgb * w3.x * w3.y;
  
  let i0 = vec2<i32>(clamp(vec2<i32>(floor(p)), vec2<i32>(0), vec2<i32>(size) - 1));
  let i1 = min(i0 + 1, vec2<i32>(size) - 1);
  let a = textureLoad(histTex, i0, 0).rgb;
  let b = textureLoad(histTex, vec2<i32>(i1.x, i0.y), 0).rgb;
  let cc = textureLoad(histTex, vec2<i32>(i0.x, i1.y), 0).rgb;
  let d = textureLoad(histTex, i1, 0).rgb;
  return clamp(c, min(min(a, b), min(cc, d)), max(max(a, b), max(cc, d)));
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid : vec3<u32>,
        @builtin(local_invocation_index) lid : u32) {
  let W = u32(P.cam.z);
  let H = u32(P.cam.w);
  let px = gid.xy;
  let inside = px.x < W && px.y < H;
  var hole = 0u;
  if (inside) { hole = reproject(px, W); }
  
  if (lid == 0u) { atomicStore(&wgHoles, 0u); }
  workgroupBarrier();
  atomicAdd(&wgHoles, hole);
  workgroupBarrier();
  if (lid == 0u) { atomicAdd(&holeCount[0], atomicLoad(&wgHoles)); }
}


fn reproject(px : vec2<u32>, W : u32) -> u32 {
  let idx = px.y * W + px.x;
  let camDist = P.cam.x;
  let fov = P.cam.y;
  let dir = rayDir(vec2<f32>(px));
  let b = 2.0 * dir.z * camDist;
  let disc = b * b - 4.0 * (camDist * camDist - 1.0);
  if (disc < 0.0) {
    textureStore(outTex, vec2<i32>(px), vec4<f32>(SPACE, 1.0));
    sigmaCur[idx] = 0.0;
    return 0u;
  }
  let t = (-b - sqrt(disc)) * 0.5;
  let hp = vec3<f32>(0.0, 0.0, camDist) + t * dir;
  let facing = clamp(dot(hp, -dir), 0.0, 1.0);
  let sigmaRaw = (fov / max(P.cam.w - 1.0, 1.0)) * t / max(facing, 1e-2) / TWO_PI;
  let model = rt() * hp;
  let lo = R.misc.zw;
  let sLo = sigmaRaw * P.cam.w / lo.y; 

  
  let pd = R.misc.x;
  let hpp = transpose(rtPrev()) * model;
  var col = vec3<f32>(0.0);
  var sig = 0.0;
  var reused = false;
  if (R.misc.y < 0.5 && hpp.z * pd > 1.0) { 
    let d = hpp - vec3<f32>(0.0, 0.0, pd); 
    let gx = d.x / -d.z;
    let gy = d.y / -d.z;
    let halfH = tan(fov * 0.5);
    let halfW = halfH * P.cam.z / P.cam.w;
    let fx = (gx + halfW) / (2.0 * halfW) * (P.cam.z - 1.0);
    let fy = (halfH - gy) / (2.0 * halfH) * (P.cam.w - 1.0);
    if (fx >= 0.0 && fy >= 0.0 && fx <= P.cam.z - 1.0 && fy <= P.cam.w - 1.0) {
      let ip = vec2<u32>(u32(fx + 0.5), u32(fy + 0.5));
      let s0 = sigmaPrev[ip.y * W + ip.x];
      
      let fr = fract(vec2<f32>(fx, fy));
      let v = RESAMPLE_BLUR * (fr.x * (1.0 - fr.x) + fr.y * (1.0 - fr.y));
      let s1 = sqrt(s0 * s0 + v * sigmaRaw * sigmaRaw);
      if (s0 > 0.0 && s1 < sLo && s0 >= sigmaRaw / TOL_LO) {
        col = fetchHistory(vec2<f32>(fx, fy), vec2<f32>(P.cam.z, P.cam.w));
        sig = s1;
        reused = true;
      }
    }
  }
  if (!reused) {
    
    let uv = (vec2<f32>(px) * (lo - 1.0) / (vec2<f32>(P.cam.z, P.cam.w) - 1.0) + 0.5) / lo;
    col = textureSampleLevel(loTex, smp, uv, 0.0).rgb;
    sig = sLo;
  }
  textureStore(outTex, vec2<i32>(px), vec4<f32>(col, 1.0));
  sigmaCur[idx] = sig;
  return select(0u, 1u, !reused);
}
`,At=`
@group(0) @binding(0) var src : texture_2d<f32>;
@group(0) @binding(1) var smp : sampler;
struct VsOut { @builtin(position) pos : vec4<f32>, @location(0) uv : vec2<f32> };

@vertex
fn vs(@builtin(vertex_index) i : u32) -> VsOut {
  var p = array<vec2<f32>, 3>(
    vec2<f32>(-1.0, -3.0), vec2<f32>(-1.0, 1.0), vec2<f32>(3.0, 1.0));
  var o : VsOut;
  o.pos = vec4<f32>(p[i], 0.0, 1.0);
  o.uv = (p[i] * vec2<f32>(0.5, -0.5)) + vec2<f32>(0.5);
  return o;
}

@fragment
fn fs(in : VsOut) -> @location(0) vec4<f32> {
  return textureSample(src, smp, in.uv);
}
`,at={rgb:0,dem:1,tree:2},ot={16:0,8:1,4:2},nt=-500,gt=8848,Re=3e-5;function Bt(i){return(Be+i*Le)*4}function ct(i,e,t,s,r){const A=s.manifest,a=Bt(t),g=A.model_kind==="decomp_unigrid"?1:0,o=A.decomp||{};i[a+0]=A.dem.dem_scale,i[a+1]=A.dem.dem_grad_scale,i[a+2]=A.gating.std_scale,i[a+3]=0,i[a+4]=g,i[a+5]=o.h_bound||0,i[a+6]=o.offset_mean||0,i[a+7]=o.offset_std||1,i[a+8]=o.log_scale_mean||0,i[a+9]=o.log_scale_std||1,i[a+10]=0,i[a+11]=0;const u=(A.grid?.dtype??"float16")!=="float16",E=A.grid?.block_vertices??64;if(u&&E!==64)throw new Error(`${A.grid.dtype} block must be 64 (kernel shifts by 6), got ${E}`);e[a+12]=t*te,e[a+13]=A.levels.length,e[a+14]=0,e[a+15]=0;const c=A.gating.gating_resolutions;if(A.levels.length!==ge.NLEV)throw new Error(`expected ${ge.NLEV} levels, got ${A.levels.length}`);A.levels.forEach((f,w)=>{const l=a+16+w*12;i[l+0]=f.scale,i[l+1]=c[w],i[l+2]=0,i[l+3]=0;const b=Ge(f,A),F=ot[b];if(F===void 0)throw new Error(`level ${w}: unsupported bits ${b}`);e[l+4]=f.size_vertices,e[l+5]=f.resolution,e[l+6]=f.dense?1:0,e[l+7]=r+s.placement[w].bufIdx,e[l+8]=s.placement[w].localBase,e[l+9]=F?(s.scaleOffset??0)+f.start_vertex/E:0,e[l+10]=F,e[l+11]=0})}function lt(i,e){const t=e.rt;i[0]=t[0],i[1]=t[1],i[2]=t[2],i[3]=0,i[4]=t[3],i[5]=t[4],i[6]=t[5],i[7]=0,i[8]=t[6],i[9]=t[7],i[10]=t[8],i[11]=0,i[12]=e.dist,i[13]=e.fovRad,i[14]=e.W,i[15]=e.H,i[16]=e.sigmaScale,i[17]=at[e.mode]??0,i[18]=nt,i[19]=gt,i[20]=e.crossover,i[21]=e.nLegs,i[22]=e.forceLeg,i[23]=e.nFineBoxes||0,i[24]=e.stride||1,i[25]=e.phase||0,i[26]=e.showCoverage?1:0,i[27]=e.surface?1:0;const s=e.pick;i[28]=s?s[0]:0,i[29]=s?s[1]:0,i[30]=s?s[2]:0,i[31]=s?1:0}function ht(i,{prevRt:e,prevDist:t,invalidate:s,loW:r,loH:A}){i[0]=e[0],i[1]=e[1],i[2]=e[2],i[3]=0,i[4]=e[3],i[5]=e[4],i[6]=e[5],i[7]=0,i[8]=e[6],i[9]=e[7],i[10]=e[8],i[11]=0,i[12]=t,i[13]=s?1:0,i[14]=r,i[15]=A}const _e=[{name:"alps-north-italy",lat:[40.499,45.046],lon:[5.545,11.651]},{name:"us-southwest",lat:[31.552,41.514],lon:[-118.461,-106.845]}];function ut(i){const e=new Float32Array(Math.max(1,i.length)*4);return i.forEach((t,s)=>{const[r,A]=t.lat,[a,g]=t.lon;if(!(r<=A&&a<=g))throw new Error(`fine coverage box ${t.name||s}: min must be <= max (split antimeridian boxes)`);e.set([r,A,a,g],s*4)}),e}const Qt=3,ue=4,ie=256,se=12,N=Be*16;class Et{constructor(e,t={}){this.gpu=e,this.device=e.device,this.legs=[],this.crossover=Re,this.version=0,this.inFlight=0,this.onError=null;const s=this.device;this.uniform=s.createBuffer({label:"params",size:ce,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST}),this._ubuf=new ArrayBuffer(ce),this._f32=new Float32Array(this._ubuf),this._u32=new Uint32Array(this._ubuf),this._staging=Array.from({length:se},(g,o)=>s.createBuffer({label:`header ${o}`,size:N,usage:GPUBufferUsage.COPY_SRC|GPUBufferUsage.COPY_DST})),this.reprojUniform=s.createBuffer({label:"reproj",size:he,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST}),this._rf32=new Float32Array(he/4),this.holeBuffer=s.createBuffer({label:"holes",size:4,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC|GPUBufferUsage.COPY_DST}),this._holeReadbacks=Array.from({length:ue},()=>({buf:s.createBuffer({size:4,usage:GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST}),busy:!1})),this.reuse=0,this.sigmaDummy=s.createBuffer({label:"sigma dummy",size:4,usage:GPUBufferUsage.STORAGE}),this.sampler=s.createSampler({magFilter:"linear",minFilter:"linear"});const r=s.createShaderModule({code:At,label:"present"});this.sampleLayout=s.createBindGroupLayout({entries:[{binding:0,visibility:GPUShaderStage.FRAGMENT,texture:{}},{binding:1,visibility:GPUShaderStage.FRAGMENT,sampler:{}}]}),this.presentPipeline=s.createRenderPipeline({label:"present",layout:s.createPipelineLayout({bindGroupLayouts:[this.sampleLayout]}),vertex:{module:r,entryPoint:"vs"},fragment:{module:r,entryPoint:"fs",targets:[{format:e.format}]},primitive:{topology:"triangle-list"}});const A=s.createShaderModule({code:rt,label:"reproject"});this._checkCompile(A,"reproject WGSL"),this.reprojPipeline=s.createComputePipeline({layout:"auto",compute:{module:A,entryPoint:"main"}});const a=it();this.maskBuffer=s.createBuffer({label:"land mask",size:a.byteLength,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST}),s.queue.writeBuffer(this.maskBuffer,0,a),this.coverBuffer=s.createBuffer({label:"fine coverage",size:ie*16,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST}),this.fineCoverage=[],this.setFineCoverage(t.fineCoverage??_e),this._targets=new Map,this.hist=null,this.computePipeline=null,this.mlpBuffer=null,this.bytes={grids:0,mlp:0,targets:0,history:0,uniform:ce+he+se*N,timing:0,mask:a.byteLength+ie*16},this._initTiming()}_checkCompile(e,t){e.getCompilationInfo().then(s=>{const r=s.messages.filter(a=>a.type==="error");if(!r.length)return;const A=r.map(a=>`L${a.lineNum}:${a.linePos} ${a.message}`).join(`
`);console.error(`[${t}]
`+A),this.onError?.(new Error(`shader compile failed (${t}):
${A}`))})}_initTiming(){if(this.timing=null,!this.gpu.hasTimestamps)return;const e=this.device;this.timing={querySet:e.createQuerySet({type:"timestamp",count:2}),resolve:e.createBuffer({size:16,usage:GPUBufferUsage.QUERY_RESOLVE|GPUBufferUsage.COPY_SRC}),readbacks:Array.from({length:ue},()=>({buf:e.createBuffer({size:16,usage:GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST}),busy:!1}))},this.bytes.timing=16*(ue+1)}setFineCoverage(e){if(e.length>ie)throw new Error(`at most ${ie} coverage boxes`);this.fineCoverage=e,this.device.queue.writeBuffer(this.coverBuffer,0,ut(e))}setLegs(e){if(e.length<1||e.length>2)throw new Error("1 or 2 legs supported");this.legs=e,this.crossover=e[e.length-1].manifest.routing?.sigma_crossover??Re,this.mlpBuffer?.destroy();const t=new Float32Array(te*e.length);e.forEach((o,u)=>{if(o.mlp.length!==te)throw new Error(`leg ${u}: mlp has ${o.mlp.length} floats`);t.set(o.mlp,u*te)}),this.mlpBuffer=this.device.createBuffer({label:"mlp",size:t.byteLength,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST}),this.device.queue.writeBuffer(this.mlpBuffer,0,t),this.scalesBuffer?.destroy();let s=0;for(const o of e)o.scaleOffset=s,s+=o.scales?o.scales.length:0;const r=new Uint32Array(Math.max(1,s));for(const o of e)o.scales&&r.set(o.scales,o.scaleOffset);this.scalesBuffer=this.device.createBuffer({label:"int8 scales",size:r.byteLength,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST}),this.device.queue.writeBuffer(this.scalesBuffer,0,r);let A=0;e.forEach((o,u)=>{ct(this._f32,this._u32,u,o,A),A+=o.nGridBuffers}),this.nGrid=A,this.device.queue.writeBuffer(this.uniform,N,this._ubuf,N);const a=st(this.nGrid),g=this.device.createShaderModule({code:a,label:`forward x${e.length}`});this._checkCompile(g,"forward WGSL"),this.computePipeline=this.device.createComputePipeline({layout:"auto",compute:{module:g,entryPoint:"main"}}),this._destroyTargets(),this.bytes.grids=e.reduce((o,u)=>o+u.gridBuffers.reduce((E,c)=>E+c.size,0),0),this.bytes.mlp=t.byteLength+r.byteLength,this.version++}get totalBytes(){const e=this.bytes;return e.grids+e.mlp+e.targets+e.history+e.uniform+e.timing+e.mask}_destroyTargets(){for(const e of this._targets.values())e.tex.destroy();this._targets.clear(),this.bytes.targets=0,this._destroyHistory()}_destroyHistory(){if(this.hist){for(const e of this.hist.slots)e.tex.destroy(),e.sigma.destroy();this.hist=null,this.bytes.history=0}}_computeBind(e,t){const s=[{binding:0,resource:{buffer:this.uniform}},{binding:1,resource:{buffer:this.mlpBuffer}},{binding:2,resource:{buffer:this.scalesBuffer}}];let r=le;for(const A of this.legs)for(const a of A.gridBuffers)s.push({binding:r++,resource:{buffer:a}});return s.push({binding:r,resource:e}),s.push({binding:r+1,resource:{buffer:this.maskBuffer}}),s.push({binding:r+2,resource:{buffer:this.coverBuffer}}),s.push({binding:r+3,resource:{buffer:t}}),this.device.createBindGroup({layout:this.computePipeline.getBindGroupLayout(0),entries:s})}_presentBind(e){return this.device.createBindGroup({layout:this.sampleLayout,entries:[{binding:0,resource:e},{binding:1,resource:this.sampler}]})}_texture(e,t,s){return this.device.createTexture({label:s,size:[e,t],format:"rgba8unorm",usage:GPUTextureUsage.STORAGE_BINDING|GPUTextureUsage.TEXTURE_BINDING|GPUTextureUsage.COPY_SRC})}_target(e,t){const s=`${e}x${t}`;let r=this._targets.get(s);if(r)return this._targets.delete(s),this._targets.set(s,r),r;if(this._targets.size>=Qt){const[g,o]=this._targets.entries().next().value;o.tex.destroy(),this._targets.delete(g),this.bytes.targets-=o.bytes}const A=this._texture(e,t,`target ${s}`),a=A.createView();return r={tex:A,view:a,W:e,H:t,bytes:e*t*4,computeBind:this._computeBind(a,this.sigmaDummy),presentBind:this._presentBind(a)},this._targets.set(s,r),this.bytes.targets+=r.bytes,r}_history(e,t){if(this.hist&&this.hist.W===e&&this.hist.H===t)return this.hist;this._destroyHistory();const s=[0,1].map(r=>{const A=this._texture(e,t,`history ${r}`),a=A.createView(),g=this.device.createBuffer({label:`history sigma ${r}`,size:e*t*4,usage:GPUBufferUsage.STORAGE});return{tex:A,view:a,sigma:g,computeBind:this._computeBind(a,g),presentBind:this._presentBind(a)}});return this.hist={W:e,H:t,slots:s,cur:0,cam:null},this.bytes.history=2*e*t*8,this.hist}frame({view:e,lo:t=null,reproject:s=null,refine:r=[],onGpuTime:A=null}){if(!this.computePipeline)return;if(s&&!t)throw new Error("reproject needs a low-res frame for the holes");const a=(t?1:0)+(s?1:0)+r.length;if(a>se)throw new Error(`at most ${se} passes per frame`);const g=this.timing?.readbacks.find(d=>!d.busy)||null,o=this.device.createCommandEncoder();let u=0;const E=()=>{if(!g)return{};const d={querySet:this.timing.querySet};return u===0&&(d.beginningOfPassWriteIndex=0),u===a-1&&(d.endOfPassWriteIndex=1),d.beginningOfPassWriteIndex!==void 0||d.endOfPassWriteIndex!==void 0?{timestampWrites:d}:{}},c=(d,C,m)=>{lt(this._f32,{...e,W:d,H:C,crossover:this.crossover,nLegs:this.legs.length,nFineBoxes:this.fineCoverage.length,...m});const G=this._staging[u];this.device.queue.writeBuffer(G,0,this._ubuf,0,N),o.copyBufferToBuffer(G,0,this.uniform,0,N)},f=(d,C,m,G)=>{const h=o.beginComputePass(E());h.setPipeline(d),h.setBindGroup(0,C),h.dispatchWorkgroups(m,G),h.end(),u++};let w=null;t&&(w=this._target(t.W,t.H),c(t.W,t.H,{stride:1,phase:0,surface:0}),f(this.computePipeline,w.computeBind,Math.ceil(t.W/8),Math.ceil(t.H/8)));let l=this.hist;if(s){l=this._history(e.W,e.H);const d=l.slots[l.cur],C=l.slots[1-l.cur],m=l.cam&&!s.invalidate;ht(this._rf32,{prevRt:m?l.cam.rt:e.rt,prevDist:m?l.cam.dist:e.dist,invalidate:!m,loW:t.W,loH:t.H}),this.device.queue.writeBuffer(this.reprojUniform,0,this._rf32),c(e.W,e.H,{stride:1,phase:0,surface:1});const G=this.device.createBindGroup({layout:this.reprojPipeline.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:this.uniform}},{binding:1,resource:{buffer:this.reprojUniform}},{binding:2,resource:d.view},{binding:3,resource:this.sampler},{binding:4,resource:{buffer:d.sigma}},{binding:5,resource:w.view},{binding:6,resource:C.view},{binding:7,resource:{buffer:C.sigma}},{binding:8,resource:{buffer:this.holeBuffer}}]});o.clearBuffer(this.holeBuffer),f(this.reprojPipeline,G,Math.ceil(e.W/8),Math.ceil(e.H/8));const h=this._holeReadbacks.find(v=>!v.busy);if(h){h.busy=!0,o.copyBufferToBuffer(this.holeBuffer,0,h.buf,0,4);const v=e.W*e.H;this.device.queue.onSubmittedWorkDone().then(()=>h.buf.mapAsync(GPUMapMode.READ)).then(()=>{const Q=new Uint32Array(h.buf.getMappedRange())[0];h.buf.unmap(),h.busy=!1,this.reuse=1-Q/v}).catch(()=>{h.busy=!1})}l.cur=1-l.cur,l.cam={rt:Array.from(e.rt),dist:e.dist}}if(l&&l.W===e.W&&l.H===e.H){const d=l.slots[l.cur];for(const C of r)c(e.W,e.H,{stride:C.stride,phase:C.phase,surface:1}),f(this.computePipeline,d.computeBind,Math.ceil(e.W/(8*C.stride)),Math.ceil(e.H/(8*C.stride)))}const b=l&&l.W===e.W&&l.H===e.H?l.slots[l.cur]:w;if(!b)return;const F=o.beginRenderPass({colorAttachments:[{view:this.gpu.ctx.getCurrentTexture().createView(),loadOp:"clear",storeOp:"store",clearValue:{r:0,g:0,b:0,a:1}}]});F.setPipeline(this.presentPipeline),F.setBindGroup(0,b.presentBind),F.draw(3),F.end(),g&&a>0&&(g.busy=!0,o.resolveQuerySet(this.timing.querySet,0,2,this.timing.resolve,0),o.copyBufferToBuffer(this.timing.resolve,0,g.buf,0,16));const S=performance.now();this.inFlight++,this.device.queue.submit([o.finish()]);const M=this.device.queue.onSubmittedWorkDone().then(()=>{this.inFlight--});g&&a>0?g.buf.mapAsync(GPUMapMode.READ).then(()=>{const d=new BigUint64Array(g.buf.getMappedRange()),C=Number(d[1]-d[0]);g.buf.unmap(),g.busy=!1,C>0&&A?.(C/1e6,!1)}).catch(()=>{g.busy=!1}):a>0&&M.then(()=>A?.(performance.now()-S,!0))}dispose(){this._destroyTargets(),this.uniform.destroy(),this.reprojUniform.destroy(),this.sigmaDummy.destroy(),this.holeBuffer.destroy();for(const e of this._holeReadbacks)e.buf.destroy();for(const e of this._staging)e.destroy();this.maskBuffer.destroy(),this.coverBuffer.destroy(),this.mlpBuffer?.destroy(),this.scalesBuffer?.destroy();for(const e of this.legs)e.dispose?.();this.legs=[],this.timing&&(this.timing.resolve.destroy(),this.timing.readbacks.forEach(e=>e.buf.destroy()),this.timing.querySet.destroy())}}const Y={sub:(i,e)=>[i[0]-e[0],i[1]-e[1],i[2]-e[2]],dot:(i,e)=>i[0]*e[0]+i[1]*e[1]+i[2]*e[2],cross:(i,e)=>[i[1]*e[2]-i[2]*e[1],i[2]*e[0]-i[0]*e[2],i[0]*e[1]-i[1]*e[0]],norm:i=>Math.hypot(i[0],i[1],i[2]),normalize:i=>{const e=Math.hypot(i[0],i[1],i[2])||1e-12;return[i[0]/e,i[1]/e,i[2]/e]}},_={identity:()=>[1,0,0,0,1,0,0,0,1],globeDefault:()=>[0,1,0,0,0,1,1,0,0],transpose:i=>[i[0],i[3],i[6],i[1],i[4],i[7],i[2],i[5],i[8]],mul:(i,e)=>{const t=new Array(9);for(let s=0;s<3;s++)for(let r=0;r<3;r++)t[s*3+r]=i[s*3]*e[r]+i[s*3+1]*e[3+r]+i[s*3+2]*e[6+r];return t},mulVec:(i,e)=>[i[0]*e[0]+i[1]*e[1]+i[2]*e[2],i[3]*e[0]+i[4]*e[1]+i[5]*e[2],i[6]*e[0]+i[7]*e[1]+i[8]*e[2]],axisAngle:(i,e)=>{const t=Y.normalize(i),s=Math.sin(e),r=Math.cos(e),[A,a,g]=t,o=[0,-g,a,g,0,-A,-a,A,0],u=_.mul(o,o),E=new Array(9);for(let c=0;c<9;c++)E[c]=(c%4===0?1:0)+s*o[c]+(1-r)*u[c];return E},between:(i,e)=>{const t=Y.normalize(i),s=Y.normalize(e),r=Y.cross(t,s),A=Y.norm(r),a=Math.max(-1,Math.min(1,Y.dot(t,s)));if(A<1e-8)return _.identity();const g=[r[0]/A,r[1]/A,r[2]/A],[o,u,E]=g,c=[0,-E,u,E,0,-o,-u,o,0],f=_.mul(c,c),w=new Array(9);for(let l=0;l<9;l++)w[l]=(l%4===0?1:0)+A*c[l]+(1-a)*f[l];return w}},ft=Math.PI*2;function Qe(i,e,t,s,r,A){const a=t/s,g=Math.tan(r/2),o=g*a,u=-o+i/Math.max(t-1,1)*2*o,E=g-e/Math.max(s-1,1)*2*g;let c=u,f=E,w=-1;const l=Math.hypot(c,f,w);c/=l,f/=l,w/=l;const b=2*w*A,F=A*A-1,S=b*b-4*F;if(S<0)return{hit:!1,dir:[c,f,w],hitPoint:[0,0,0],facing:0,sigma:0};const M=(-b-Math.sqrt(S))/2,d=[c*M,f*M,A+w*M],C=Math.max(0,Math.min(1,-(d[0]*c+d[1]*f+d[2]*w))),m=r/Math.max(s-1,1)/Math.max(C,.01)/ft,G=m*M;return{hit:!0,dir:[c,f,w],hitPoint:d,facing:C,sigma:m,sigmaSurface:G}}class dt{constructor(){this.rotation=_.globeDefault(),this.homeDist=2.4,this.dist=this.homeDist,this.minAlt=.004,this.maxDist=10,this.fovDeg=45,this.homeFov=45,this._dragStart=null,this._spinAxis=null,this._spinSpeed=0,this._friction=.93,this.viewH=1e3,this.autoSpin=0,this._lastTick=0}get fovRad(){return this.fovDeg*Math.PI/180}fitView(e,t,s=.8){const r=45*Math.PI/180,A=e>=t?r:2*Math.atan(Math.tan(r/2)*t/Math.max(e,1));this.homeFov=this.fovDeg=A*180/Math.PI,this.homeDist=this.dist=1/Math.sin(Math.atan(s*Math.tan(r/2)))}get rtArray(){return _.transpose(this.rotation)}reset(){this.rotation=_.globeDefault(),this.dist=this.homeDist,this.fovDeg=this.homeFov,this._spinAxis=null,this._spinSpeed=0}zoom(e){const t=Math.pow(.92,e),s=Math.max(this.minAlt,Math.min(this.maxDist-1,(this.dist-1)*t));this.dist=1+s}beginDrag(e,t,s,r){const A=Qe(e,t,s,r,this.fovRad,this.dist);this._dragStart=A.hit?A.hitPoint:null,this._dragging=!0,this._spinSpeed=0,this._spinAxis=null}endDrag(){this._dragStart=null,this._dragging=!1}drag(e,t,s,r){if(!this._dragStart)return;const A=Qe(e,t,s,r,this.fovRad,this.dist);if(!A.hit)return;const a=this._dragStart,g=_.between(a,A.hitPoint);this.rotation=_.mul(g,this.rotation),this._dragStart=A.hitPoint;const o=Y.cross(a,A.hitPoint),u=Math.acos(Math.max(-1,Math.min(1,Y.dot(A.hitPoint,a)))),E=Y.norm(o);if(E>1e-8&&u>1e-6){const c=[o[0]/E,o[1]/E,o[2]/E],f=.6;this._spinAxis?(this._spinAxis=Y.normalize([f*c[0]+(1-f)*this._spinAxis[0],f*c[1]+(1-f)*this._spinAxis[1],f*c[2]+(1-f)*this._spinAxis[2]]),this._spinSpeed=f*u+(1-f)*this._spinSpeed):(this._spinAxis=c,this._spinSpeed=u)}}get minStep(){return .25*(this.fovRad/Math.max(this.viewH,1))*Math.max(this.dist-1,this.minAlt)}tick(e=performance.now()){const t=this._lastTick?Math.min(.1,(e-this._lastTick)/1e3):0;this._lastTick=e,!this._dragging&&(this._spinAxis&&this._spinSpeed>0?(this.rotation=_.mul(_.axisAngle(this._spinAxis,this._spinSpeed),this.rotation),this._spinSpeed*=this._friction,this._spinSpeed<this.minStep&&(this._spinSpeed=0,this._spinAxis=null)):this.autoSpin&&t>0&&(this.rotation=_.mul(_.axisAngle([0,1,0],this.autoSpin*t),this.rotation)))}}class wt{constructor(e={}){this.auto=e.auto??!0,this.scale=e.scale??.5,this.minScale=e.minScale??.12,this.targetMs=e.targetMs??22,this.maxScale=e.maxScale??1,this.refineMs=e.refineMs??12,this.strides=e.strides??[2,4,8],this._ema=0,this._fullMs=0,this.refineMsLast=0,this._samples=0}onGpuTime(e,t){if(!t.lo&&t.passes>0){const r=e/t.passes;this.refineMsLast=r,this._fullMs=Math.max(this._fullMs*.97,r*t.stride*t.stride);return}if(this._ema=this._ema?this._ema*.7+e*.3:e,!this.auto||++this._samples<4)return;this._samples=0;const s=this._ema/this.targetMs;s>1.25?this.scale=Math.max(this.minScale,this.scale/Math.sqrt(Math.min(s,4))):s<.6&&this.scale<this.maxScale&&(this.scale=Math.min(this.maxScale,this.scale*1.15)),this.scale=Math.min(this.scale,this.maxScale),this.scale=Math.round(this.scale*100)/100}setScale(e){this.auto=!1,this.scale=Math.round(Math.max(this.minScale,Math.min(this.maxScale,e))*100)/100}setAuto(e){this.auto=e,this._ema=0,this._samples=0}fullCostMs(e,t,s,r){return this._fullMs||(this._ema||this.targetMs)*(s*r)/Math.max(1,e*t)}refineStride(e,t,s,r){const A=this.fullCostMs(e,t,s,r);for(const a of this.strides)if(A/(a*a)<=this.refineMs)return a;return this.strides[this.strides.length-1]}passCostMs(e,t,s,r,A){return Math.max(this.refineMsLast,.5*this.fullCostMs(e,t,s,r)/(A*A))||this.refineMs}movingPasses(e,t,s,r,A){const a=this.fullCostMs(e,t,s,r)*(e*t)/(s*r),g=this.targetMs-a-1.5;return Math.max(1,Math.min(4,Math.floor(g/this.passCostMs(e,t,s,r,A))))}setQuiet(e){this.targetMs=e?10:22,this.refineMs=e?6:12,this.maxScale=e?.5:1,this.scale=Math.min(this.scale,this.maxScale),this._ema=0,this._samples=0}interactiveSize(e,t){return pt(e,t,this.scale)}}function pt(i,e,t){return[Math.max(8,Math.round(i*t)),Math.max(8,Math.round(e*t))]}const Ye={Digit1:"rgb",Digit2:"dem",Digit3:"tree"};class Ct{constructor(e,t,s,r){this.canvas=e,this.camera=t,this.state=s,this.actions=r,this._pointers=new Map,this._pinchDist=0,this._bind()}_bind(){const e=this.canvas,t=()=>{this.camera.autoSpin=0};e.addEventListener("pointerdown",t),e.addEventListener("wheel",t,{passive:!0}),e.addEventListener("keydown",t),e.addEventListener("pointerdown",r=>{e.setPointerCapture(r.pointerId),this._pointers.set(r.pointerId,r),this._down=this._pointers.size===1?[r.clientX,r.clientY]:null,this._pointers.size===1?(e.classList.add("dragging"),this.camera.beginDrag(...this._px(r))):this._pointers.size===2&&(this.camera.endDrag(),this._pinchDist=this._pinch())}),e.addEventListener("pointermove",r=>{if(this._pointers.has(r.pointerId)){if(this._pointers.set(r.pointerId,r),this._pointers.size===1)this.camera.drag(...this._px(r));else if(this._pointers.size===2){const A=this._pinch();this._pinchDist>0&&A>0&&this.camera.zoom(Math.log(this._pinchDist/A)/Math.log(.92)),this._pinchDist=A}}});const s=r=>{if(this._pointers.delete(r.pointerId),this._pointers.size===0)e.classList.remove("dragging"),this.camera.endDrag();else if(this._pointers.size===1){const[A]=this._pointers.values();this.camera.beginDrag(...this._px(A))}};e.addEventListener("pointerup",r=>{const A=this._down;A&&this.state.pick&&this._pointers.size===1&&Math.hypot(r.clientX-A[0],r.clientY-A[1])<5&&this.actions.pickPoint(...this._px(r)),this._down=null,s(r)}),e.addEventListener("pointercancel",r=>{this._down=null,s(r)}),e.addEventListener("wheel",r=>{r.preventDefault(),this.camera.zoom(Math.sign(r.deltaY)*(r.shiftKey?3:1))},{passive:!1}),e.addEventListener("keydown",r=>{if(Ye[r.code]){this.actions.setMode(Ye[r.code]);return}const A=this.actions;switch(r.code){case"BracketLeft":A.scaleSigma(1/1.5);break;case"BracketRight":A.scaleSigma(1.5);break;case"KeyR":A.reset();break;case"KeyF":A.fullscreen();break;case"KeyQ":A.toggleQuiet();break;case"KeyS":A.toggleStats();break;case"KeyC":A.togglePick();break;case"KeyL":A.cycleLeg();break;case"KeyG":A.cycleGpu();break;default:return}r.preventDefault()})}_pinch(){const[e,t]=this._pointers.values();return Math.hypot(e.clientX-t.clientX,e.clientY-t.clientY)}_px(e){const t=this.canvas.getBoundingClientRect(),s=this.state.canvasW/t.width,r=this.state.canvasH/t.height;return[(e.clientX-t.left)*s,(e.clientY-t.top)*r,this.state.canvasW,this.state.canvasH]}}const mt=[["rgb","RGB"],["dem","DEM"],["tree","Tree"]];class It{constructor(e,t,s){this.el=e,this.state=t,this.actions=s,this._buttons=[],this._build()}_group(e,t,s,r){const A=document.createElement("div");if(A.className="tb-group",e){const a=document.createElement("span");a.className="tb-title",a.textContent=e,A.appendChild(a)}for(const[a,g,o]of t){const u=document.createElement("button");u.type="button",u.textContent=g,o&&(u.title=o),u.addEventListener("click",()=>r(a)),this._buttons.push({b:u,isActive:()=>s(a)}),A.appendChild(u)}return this.el.appendChild(A),A}_build(){const e=this.state,t=this.actions;this._group("layer",mt.map(([s,r],A)=>[s,r,`key ${A+1}`]),s=>e.mode===s,s=>t.setMode(s)),this._group("",[["quiet","Quiet","low GPU power while dragging: 24 renders/s, \u2264\xBD res, 10 ms budget (Q)"],["stats","Stats","toggle the stats overlay (S)"]],s=>({quiet:e.quiet,stats:e.showStats})[s],s=>({quiet:t.toggleQuiet,stats:t.toggleStats})[s]()),this._group("",[["reset","Reset","R"],["fs","Fullscreen","F"],["cache","Clear cache","forget downloaded weights"]],()=>!1,s=>({reset:t.reset,fs:t.fullscreen,cache:t.clearCache})[s]())}sync(){for(const{b:e,isActive:t}of this._buttons)e.classList.toggle("active",!!t())}}const vt={rgb:"RGB",dem:"DEM hillshade",tree:"Tree cover"};function xt(i,e){const t=i.fovRad/Math.max(e-1,1)*(i.dist-1)/(2*Math.PI);return{sigma:t,z:Math.log2(1/(256*t))}}const Mt={"-1":"auto routing",0:"coarse only",1:"fine only"};class Dt{constructor(){this._renders=[],this._gpu=[],this.lastGpuMs=0,this.gpuApprox=!1,this.fps=0,this.busy=0,this.evalsPerSec=0}submitted(e,t,s=1){this._renders.push({t:e,px:t*s})}gpuTime(e,t,s){this._gpu.push({t:e,ms:t}),this.lastGpuMs=t,this.gpuApprox=s}update(e){const t=e-1e3;for(;this._renders.length&&this._renders[0].t<t;)this._renders.shift();for(;this._gpu.length&&this._gpu[0].t<t;)this._gpu.shift();this.fps=this._renders.length,this.evalsPerSec=this._renders.reduce((s,r)=>s+r.px,0),this.busy=Math.min(1,this._gpu.reduce((s,r)=>s+r.ms,0)/1e3)}}function z(i){return i>=1e9?`${(i/1e9).toFixed(2)} GB`:i>=1e6?`${(i/1e6).toFixed(0)} MB`:i>=1e3?`${(i/1e3).toFixed(0)} kB`:`${i} B`}class bt{constructor(e){this.el=e}update(e,t,s){if(!e.showStats){this.el.textContent="";return}const{camera:r,mode:A,sigmaScale:a,forceLeg:g,quality:o,legs:u,fine:E,gpuName:c,gpuPref:f,canvasW:w,canvasH:l,refine:b}=e,[F,S]=o.interactiveSize(w,l),M=b?`surface refine ${Math.min(b.since,b.n)}/${b.n} (stride ${b.stride}, ${o.refineMsLast.toFixed(1)} ms/pass) \xB7 reuse ${(s.reuse*100).toFixed(0)}%`:"refine idle",d=s.fineCoverage.length,C=(u.length===2?`coarse + fine \xB7 ${Mt[g]}${d?` in ${d} area${d>1?"s":""}`:" everywhere"}`:`${u[0]||"no"} leg`)+` \xB7 ${e.quant}`,m=performance.memory?` \xB7 JS heap ${z(performance.memory.usedJSHeapSize)}`:"",G=s.bytes,{sigma:h,z:v}=xt(r,l),Q=[`${C}  \xB7  ${vt[A]}  \xB7  \u03C3\xD7${a.toFixed(2)}  \xB7  alt ${(r.dist-1).toFixed(3)}  \xB7  centre \u03C3 ${h.toExponential(1)} \u2248 z${v.toFixed(1)}`,`canvas ${w}\xD7${l}  \xB7  interactive ${F}\xD7${S} (${o.auto?"auto ":""}${o.scale.toFixed(2)})  \xB7  ${M}${e.quiet?"  \xB7  QUIET":""}`,`${t.fps} renders/s  \xB7  GPU ${t.gpuApprox?"\u2248":""}${t.lastGpuMs.toFixed(1)} ms/frame  \xB7  busy ${(t.busy*100).toFixed(0)}%  \xB7  ${(t.evalsPerSec/1e6).toFixed(1)} M evals/s`,`GPU memory ${z(s.totalBytes)}  (grids ${z(G.grids)} \xB7 targets ${z(G.targets)} \xB7 mlp ${z(G.mlp)})${m}`];E.status==="loading"?Q.push(`fine leg: downloading ${(E.progress*100).toFixed(0)}%`):E.status==="error"?Q.push(`fine leg: failed \u2014 ${E.error}`):E.status==="skipped"&&Q.push(`fine leg: skipped \u2014 ${E.error}`),Q.push(`${c}  [${f}]`),this.el.textContent=Q.join(`
`)}}const yt="attribute vec2 p; void main() { gl_Position = vec4(p, 0.0, 1.0); }",Gt=`
precision mediump float;
uniform vec2 uRes;
uniform float uP;    /* eased progress 0..1 */
uniform float uT;    /* seconds */

float hash3(vec3 p) {
  p = fract(p * 0.1031 + vec3(0.11, 0.37, 0.71));
  p += dot(p, p.yzx + 33.33);
  return fract((p.x + p.y) * p.z);
}

void main() {
  float R = 0.5 * min(uRes.x, uRes.y);
  vec2 uv = (gl_FragCoord.xy - 0.5 * uRes) / (R * 0.92);
  float r2 = dot(uv, uv);
  float px = 1.0 / (R * 0.92);
  float edge = 1.0 - smoothstep(1.0 - 2.0 * px, 1.0 + px, sqrt(r2));
  if (edge <= 0.0) { gl_FragColor = vec4(0.0); return; }

  float z = sqrt(max(1.0 - r2, 0.0));
  vec3 n = vec3(uv, z);

  /* slow spin about a tilted axis so the pattern drifts while downloading */
  float a = uT * 0.35;
  float c = cos(a), s = sin(a);
  vec3 q = vec3(c * n.x + s * n.z, n.y, -s * n.x + c * n.z);
  float tilt = 0.4;
  q = vec3(q.x, cos(tilt) * q.y - sin(tilt) * q.z, sin(tilt) * q.y + cos(tilt) * q.z);

  /* cell = cube-lattice voxel on the sphere surface; two lattices blended so
     the reveal has both coarse continents and fine speckle */
  float hA = hash3(floor(q * 5.0 + 100.0));
  float hB = hash3(floor(q * 11.0 + 300.0));
  float h = mix(hA, hB, 0.35);
  float on = 1.0 - smoothstep(uP, uP + 0.03, h);

  vec3 light = normalize(vec3(-0.45, 0.65, 0.62));
  float lam = max(dot(n, light), 0.0);
  float rim = pow(1.0 - z, 3.0);

  vec3 ghost = vec3(0.075, 0.095, 0.145) * (0.55 + 0.45 * lam) + rim * vec3(0.05, 0.07, 0.10);
  /* finished cells: a gentle two-tone so the whole sphere reads as one body */
  vec3 sea = vec3(0.17, 0.47, 0.56);
  vec3 land = vec3(0.30, 0.70, 0.72);
  vec3 lit = mix(sea, land, smoothstep(0.30, 0.80, hB)) * (0.35 + 0.75 * lam) + rim * vec3(0.30, 0.55, 0.60);

  vec3 col = mix(ghost, lit, on);
  gl_FragColor = vec4(col * edge, edge);
}`;class Ft{constructor(e){this.canvas=e,this.target=0,this.shown=0,this._raf=0,this._t0=performance.now(),this._running=!1;const t=e.getContext("webgl",{alpha:!0,premultipliedAlpha:!0,antialias:!1,depth:!1,stencil:!1,powerPreference:"low-power",preserveDrawingBuffer:!1});if(this.gl=t,!t){e.classList.add("geo-hidden");return}const s=(r,A)=>{const a=t.createShader(r);if(t.shaderSource(a,A),t.compileShader(a),!t.getShaderParameter(a,t.COMPILE_STATUS))throw new Error(t.getShaderInfoLog(a));return a};try{const r=t.createProgram();if(t.attachShader(r,s(t.VERTEX_SHADER,yt)),t.attachShader(r,s(t.FRAGMENT_SHADER,Gt)),t.linkProgram(r),!t.getProgramParameter(r,t.LINK_STATUS))throw new Error(t.getProgramInfoLog(r));t.useProgram(r);const A=t.createBuffer();t.bindBuffer(t.ARRAY_BUFFER,A),t.bufferData(t.ARRAY_BUFFER,new Float32Array([-1,-3,-1,1,3,1]),t.STATIC_DRAW);const a=t.getAttribLocation(r,"p");t.enableVertexAttribArray(a),t.vertexAttribPointer(a,2,t.FLOAT,!1,0,0),this._u={res:t.getUniformLocation(r,"uRes"),p:t.getUniformLocation(r,"uP"),t:t.getUniformLocation(r,"uT")},this._prog=r}catch(r){console.warn("[loader] shader unavailable",r),this.gl=null,e.classList.add("geo-hidden")}}setProgress(e){this.target=Math.max(0,Math.min(1,e))}start(){if(this._running||!this.gl)return;this._running=!0;const e=t=>{this._running&&(this._draw(t),this._raf=requestAnimationFrame(e))};this._raf=requestAnimationFrame(e)}stop(){this._running=!1,cancelAnimationFrame(this._raf)}_draw(e){const t=this.gl,s=this.canvas,r=Math.min(window.devicePixelRatio||1,2),A=Math.max(1,Math.round(s.clientWidth*r)),a=Math.max(1,Math.round(s.clientHeight*r));(s.width!==A||s.height!==a)&&(s.width=A,s.height=a),this.shown+=(this.target-this.shown)*.12,Math.abs(this.target-this.shown)<.002&&(this.shown=this.target),t.viewport(0,0,A,a),t.uniform2f(this._u.res,A,a),t.uniform1f(this._u.p,this.shown),t.uniform1f(this._u.t,(e-this._t0)/1e3),t.drawArrays(t.TRIANGLES,0,3)}dispose(){this.stop(),this.gl?.getExtension("WEBGL_lose_context")?.loseContext(),this.gl=null}}const Te="neural-earth.coverage",Ke=i=>Math.round(i*1e3)/1e3;class Ut{constructor(e,t){this.el=e,this.cb=t,this.boxes=this._load(),this.pending=null,this._note="",e.innerHTML=`
      <b>Fine-leg coverage picker</b>
      <div class="pk-status"></div>
      <div class="pk-list"></div>
      <div class="pk-actions">
        <button type="button" data-a="copy" title="copy the list as JS for src/data/fineCoverage.js">Copy JS</button>
        <button type="button" data-a="undo" title="cancel the pending corner, or remove the last box">Undo</button>
        <button type="button" data-a="clear">Clear</button>
        <button type="button" data-a="done" title="leave pick mode (C)">Done</button>
      </div>`,this._status=e.querySelector(".pk-status"),this._list=e.querySelector(".pk-list"),e.addEventListener("click",s=>{const r=s.target.closest("button");if(!r)return;const A=r.dataset.a;A==="copy"?this.copy():A==="undo"?this.undo():A==="clear"?this.clear():A==="done"?this.cb.onDone():A==="rm"&&this.remove(parseInt(r.dataset.i,10))}),e.addEventListener("keydown",s=>s.stopPropagation()),this.render()}_load(){try{const e=JSON.parse(localStorage.getItem(Te)||"[]");return Array.isArray(e)?e:[]}catch{return[]}}_save(){try{localStorage.setItem(Te,JSON.stringify(this.boxes))}catch{}}_changed(e=""){this._note=e,this._save(),this.render(),this.cb.onChange(this.boxes,this.pending)}addPoint(e,t){if(e=Ke(e),t=Ke(t),!this.pending){this.pending={lat:e,lon:t},this._changed();return}const s=this.pending;if(this.pending=null,Math.abs(t-s.lon)>180){this._changed("box would cross the antimeridian \u2014 pick it as two boxes");return}this.boxes.push({name:`area ${this.boxes.length+1}`,lat:[Math.min(s.lat,e),Math.max(s.lat,e)],lon:[Math.min(s.lon,t),Math.max(s.lon,t)]}),this._changed()}missed(){this._note="that click missed the globe",this.render()}undo(){if(this.pending){this.pending=null,this._changed("corner cancelled");return}this.boxes.length&&(this.boxes.pop(),this._changed())}remove(e){this.boxes.splice(e,1),this._changed()}clear(){this.boxes=[],this.pending=null,this._changed()}toSource(){return`export const FINE_COVERAGE = [
${this.boxes.map(e=>`  { name: ${JSON.stringify(e.name)}, lat: [${e.lat[0]}, ${e.lat[1]}], lon: [${e.lon[0]}, ${e.lon[1]}] },`).join(`
`)}
];`}async copy(){const e=this.toSource();console.log(`[coverage]
`+e);try{await navigator.clipboard.writeText(e),this._note="copied \u2014 paste into src/data/fineCoverage.js"}catch{this._note="clipboard blocked \u2014 the list is in the console"}this.render()}render(){const e=this.boxes.length;this._status.textContent=this._note||(this.pending?`corner 1: ${this.pending.lat}\xB0, ${this.pending.lon}\xB0 \u2014 click the opposite corner`:`click a corner on the globe${e?` \xB7 ${e} box${e>1?"es":""} \u2014 fine leg routes only inside them`:""}`),this._list.innerHTML=this.boxes.map((t,s)=>`<div class="pk-box"><span>${t.name}</span><span>lat ${t.lat[0]}\u2026${t.lat[1]} \xB7 lon ${t.lon[0]}\u2026${t.lon[1]}</span><button type="button" data-a="rm" data-i="${s}" title="remove">\u2715</button></div>`).join("")}}const kt=[{key:"coarse",stem:"coarse"},{key:"fine",stem:"fine"}],ze=2048,St=13e5,Ht=.08,Lt=1,Pt={normal:40,quiet:24},Rt={high:"low",low:"fallback",fallback:"high"},re=180/Math.PI,_t=(i,e)=>[i[0]*e[0]+i[1]*e[1]+i[2]*e[2],i[3]*e[0]+i[4]*e[1]+i[5]*e[2],i[6]*e[0]+i[7]*e[1]+i[8]*e[2]],Yt=i=>[Math.asin(Math.max(-1,Math.min(1,i[2])))*re,Math.atan2(i[1],i[0])*re],Oe=(i,e)=>{const t=i/re,s=e/re;return[Math.cos(t)*Math.cos(s),Math.cos(t)*Math.sin(s),Math.sin(t)]},Tt=(i,e)=>{try{const t=localStorage.getItem(`neural-earth.${i}`);return t===null?e:JSON.parse(t)}catch{return e}},Kt=(i,e)=>{try{localStorage.setItem(`neural-earth.${i}`,JSON.stringify(e))}catch{}},zt=`
.geo-earth { position: relative; overflow: hidden; background: #05070d; color: #d9e2f2;
  font-family: ui-monospace, "SF Mono", Menlo, Consolas, monospace; font-size: 12px; line-height: 1.4;
  --geo-dim: #8a98b8; --geo-panel: rgba(10,14,24,.72); --geo-acc: #56c2c2; }
.geo-earth, .geo-earth * { box-sizing: border-box; }
.geo-earth canvas { display: block; width: 100%; height: 100%; cursor: grab; touch-action: none; outline: none; }
.geo-earth canvas.dragging { cursor: grabbing; }
.geo-earth .geo-hud { position: absolute; top: 10px; left: 12px; font-size: 11.5px; line-height: 1.55;
  white-space: pre; pointer-events: none; text-shadow: 0 1px 2px #000, 0 0 6px #000a;
  margin: 0; padding: 0; background: none; border: 0; color: inherit; font-family: inherit; } /* host pages often style <pre> */
.geo-earth .geo-brand { position: absolute; top: 10px; right: 14px; text-align: right; pointer-events: none;
  text-shadow: 0 1px 2px #000; }
.geo-earth .geo-brand b { font-size: 13px; letter-spacing: .04em; }
.geo-earth .geo-brand span { display: block; font-size: 10.5px; color: var(--geo-dim); margin-top: 2px; }
.geo-earth .geo-toolbar { position: absolute; left: 50%; bottom: max(12px, env(safe-area-inset-bottom));
  transform: translateX(-50%); display: flex; gap: 10px; flex-wrap: wrap; justify-content: center;
  max-width: calc(100% - 24px); padding: 6px 8px; background: var(--geo-panel);
  backdrop-filter: blur(8px); border: 1px solid #1f2a40; border-radius: 10px; }
.geo-earth .tb-group { display: flex; align-items: center; gap: 3px; }
.geo-earth .tb-group.disabled { opacity: .35; pointer-events: none; }
.geo-earth .tb-title { font-size: 10px; color: var(--geo-dim); margin: 0 4px 0 2px; text-transform: uppercase; letter-spacing: .08em; }
.geo-earth .geo-toolbar button { font: inherit; font-size: 11.5px; color: #d9e2f2; background: #141b2b;
  border: 1px solid #253150; border-radius: 6px; padding: 5px 9px; cursor: pointer; line-height: 1.2; }
.geo-earth .geo-toolbar button:hover { border-color: var(--geo-acc); }
.geo-earth .geo-toolbar button.active { background: #1d3a44; border-color: var(--geo-acc); color: #dffbfb; }
.geo-earth .geo-help { position: absolute; bottom: 4px; right: 10px; font-size: 10px; color: var(--geo-dim);
  text-shadow: 0 1px 2px #000; pointer-events: none; margin: 0; }
.geo-earth .geo-overlay { position: absolute; inset: 0; display: flex; align-items: center; justify-content: center;
  flex-direction: column; gap: 18px; background: #05080f; z-index: 10; text-align: center; padding: 24px;
  transition: opacity .4s ease; }
.geo-earth .geo-overlay.hiding { opacity: 0; pointer-events: none; }
/* poster: the framed planet sits behind the loading screen (dimmed under the title and loader), so the
   reveal lands on the picture that was already there; quiet (weights local): no chrome, just a status line */
.geo-earth .geo-overlay.poster { background-position: center; background-size: contain; background-repeat: no-repeat; }
.geo-earth .geo-overlay.poster h1, .geo-earth .geo-overlay.poster p { text-shadow: 0 1px 2px #000, 0 0 18px #000c; }
.geo-earth .geo-overlay.quiet h1, .geo-earth .geo-overlay.quiet p, .geo-earth .geo-overlay.quiet .geo-loader { display: none; }
.geo-earth .geo-overlay.quiet .geo-status { position: absolute; left: 0; right: 0; bottom: 34px; font-size: 12px; color: var(--geo-dim);
  letter-spacing: .06em; margin: 0; }
.geo-earth .geo-overlay.quiet .geo-substatus { position: absolute; left: 0; right: 0; bottom: 16px; font-size: 12px; }
.geo-earth .geo-overlay h1 { font-size: clamp(32px, 6cqw, 56px); font-weight: 600; letter-spacing: .06em; margin: 0;
  font-family: ui-sans-serif, system-ui, -apple-system, "Segoe UI", Roboto, sans-serif; }
.geo-earth .geo-overlay p { color: var(--geo-dim); font-size: clamp(14px, 2cqw, 17px); max-width: 560px; line-height: 1.55; margin: 0;
  font-family: ui-sans-serif, system-ui, -apple-system, "Segoe UI", Roboto, sans-serif; }
.geo-earth .geo-status { font-size: clamp(20px, 3.2cqw, 28px); margin-top: 10px;
  font-family: ui-sans-serif, system-ui, -apple-system, "Segoe UI", Roboto, sans-serif; }
.geo-earth .geo-loader { width: clamp(170px, 30cqw, 260px); height: auto; aspect-ratio: 1; display: block; cursor: default; }
.geo-earth .geo-substatus { font-size: clamp(14px, 2cqw, 17px); color: var(--geo-dim); min-height: 1.2em; letter-spacing: .04em; }
.geo-earth .geo-overlay.err .geo-status { color: #ff8a8a; }
.geo-earth .geo-overlay.err .geo-loader { display: none; }
.geo-earth .geo-pick { position: absolute; left: 12px; bottom: max(64px, calc(env(safe-area-inset-bottom) + 52px));
  width: min(340px, calc(100% - 24px)); padding: 8px 10px; background: var(--geo-panel); backdrop-filter: blur(8px);
  border: 1px solid #1f2a40; border-radius: 10px; font-size: 11px; line-height: 1.5; }
.geo-earth .geo-pick b { display: block; font-size: 11.5px; letter-spacing: .04em; margin-bottom: 2px; }
.geo-earth .pk-status { color: var(--geo-acc); min-height: 1.5em; }
.geo-earth .pk-list { max-height: 30vh; overflow-y: auto; margin: 4px 0; }
.geo-earth .pk-box { display: grid; grid-template-columns: 4.5em 1fr auto; gap: 6px; align-items: center; color: var(--geo-dim); }
.geo-earth .pk-box span:first-child { color: #d9e2f2; }
.geo-earth .pk-box button, .geo-earth .pk-actions button { font: inherit; font-size: 10.5px; color: #d9e2f2; background: #141b2b;
  border: 1px solid #253150; border-radius: 5px; padding: 2px 7px; cursor: pointer; }
.geo-earth .pk-box button:hover, .geo-earth .pk-actions button:hover { border-color: var(--geo-acc); }
.geo-earth .pk-actions { display: flex; gap: 4px; margin-top: 4px; }
.geo-earth .geo-hidden { display: none !important; }
@container (max-width: 720px) {
  .geo-earth .geo-hud { font-size: 10px; white-space: pre-wrap; max-width: calc(100% - 24px); } .geo-earth .geo-brand span { display: none; } .geo-earth .geo-help { display: none; }
}
`;function Ot(){if(document.querySelector("style[data-geo-earth]"))return;const i=document.createElement("style");i.dataset.geoEarth="",i.textContent=zt,document.head.appendChild(i)}const Jt=`
<canvas class="geo-canvas" tabindex="0"></canvas>
<pre class="geo-hud"></pre>
<div class="geo-brand"><b>Neural Earth</b><span>coarse + fine hash-grid legs \xB7 one WGSL kernel per pixel</span></div>
<div class="geo-toolbar"></div>
<div class="geo-pick geo-hidden"></div>
<p class="geo-help">drag \xB7 scroll/pinch zoom \xB7 1-3 layer \xB7 [ ] \u03C3 \xB7 Q quiet \xB7 S stats \xB7 R reset \xB7 F fullscreen</p>
<div class="geo-overlay">
  <h1>Neural Earth</h1>
  <p>The whole planet is a neural field: each pixel ray-casts the globe, looks up a multi-resolution hash grid and runs a small MLP \u2014 live, in your GPU, with WebGPU.</p>
  <div class="geo-status">Waking up the GPU\u2026</div>
  <canvas class="geo-loader" aria-hidden="true"></canvas>
  <div class="geo-substatus"></div>
</div>`;function Je(i,e={}){const t={weights:"weights",legs:"both",quant:"int4",gpu:"high",cache:!0,maxPixels:matchMedia("(pointer: coarse)").matches?St:1/0,brand:!0,help:!0,autofocus:!1,pick:!1,spin:!0,poster:null,quietLoad:!1,onError:null,...e},s=new URL(t.weights,document.baseURI).toString().replace(/\/+$/,""),r={fp16:"",int8:".i8",int4:".i4"}[t.quant];if(r===void 0)throw new Error(`quant must be fp16 | int8 | int4, got ${t.quant}`);const A=kt.map(h=>({...h,stem:`${h.stem}${r}`}));Ot(),i.classList.add("geo-earth"),i.style.containerType||="inline-size",i.innerHTML=Jt;const a=h=>i.querySelector(`.${h}`),g=a("geo-canvas"),o=a("geo-overlay"),u=a("geo-status"),E=a("geo-substatus");t.brand||a("geo-brand").remove(),t.help||a("geo-help").remove(),t.poster&&(o.classList.add("poster"),o.style.backgroundImage=t.quietLoad?`url("${t.poster}")`:`linear-gradient(rgba(5,8,15,.72), rgba(5,8,15,.72)), url("${t.poster}")`),t.quietLoad&&o.classList.add("quiet");const c=new Ft(a("geo-loader"));t.quietLoad||c.start();const f=h=>u.textContent=h,w=h=>E.textContent=h,l=h=>c.setProgress(h),b=()=>{o.classList.add("hiding"),setTimeout(()=>{o.classList.add("geo-hidden"),c.stop()},450)},F=new AbortController,S=[];let M=!1,d=0;const C={root:i,opts:t,ready:null,destroy(){if(!M){M=!0,F.abort(),cancelAnimationFrame(d),c.dispose();for(const h of S.splice(0))try{h()}catch{}i.innerHTML="",i.classList.remove("geo-earth"),i.__geo===C&&delete i.__geo}}};i.__geo=C;const m=h=>{M||(o.classList.add("err"),o.classList.remove("geo-hidden"),c.stop(),f(`Error: ${h.message||h}`),w(""),console.error(h),t.onError?.(h))};async function G(){let h;try{h=await Ee.create(g,t.gpu==="fallback"?{forceFallbackAdapter:!0}:{powerPreference:t.gpu==="low"?"low-power":"high-performance"})}catch(B){return m(B)}if(M){h.device.destroy();return}h.onLost=B=>m(new Error(`GPU device lost (${B.message}). Reload the page.`));const v=new dt,Q=new Et(h,{fineCoverage:t.fineCoverage});Q.onError=m;const x=new wt({scale:h.isFallback?.25:.5}),U=new Dt,n={mode:"rgb",sigmaScale:1,forceLeg:-1,showStats:!1,quiet:Tt("quiet",!1),quant:t.quant,refine:null,pick:!1,pickPending:null,quality:x,camera:v,legs:[],fine:{status:"pending",progress:0,error:""},canvasW:1,canvasH:1,gpuName:h.gpuName,gpuPref:t.gpu};n.reveal="loading";const qe=new bt(a("geo-hud")),Ne=t.fineCoverage??_e,fe=a("geo-pick"),O=new Ut(fe,{onChange:(B,p)=>{Q.setFineCoverage(B),n.pickPending=p?Oe(p.lat,p.lon):null,T()},onDone:()=>W.togglePick()});Object.assign(C,{state:n,renderer:Q,quality:x,stats:U,camera:v,gpu:h,loader:c,overlay:o}),C.snapshot=async()=>{const B=Q.hist;if(!B)throw new Error("no frame rendered yet");const p=B.slots[B.cur],I=B.W,L=B.H,k=Math.ceil(I*4/256)*256,H=h.device.createBuffer({size:k*L,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ}),R=h.device.createCommandEncoder();R.copyTextureToBuffer({texture:p.tex},{buffer:H,bytesPerRow:k},[I,L]),h.device.queue.submit([R.finish()]),await H.mapAsync(GPUMapMode.READ);const oe=new Uint8Array(H.getMappedRange()),y=new ImageData(I,L);for(let D=0;D<L;D++)y.data.set(oe.subarray(D*k,D*k+I*4),D*I*4);H.unmap(),H.destroy();const P=document.createElement("canvas");return P.width=I,P.height=L,P.getContext("2d").putImageData(y,0,0),new Promise(D=>P.toBlob(D,"image/png"))},S.push(()=>{Q.dispose(),h.device.destroy()});let de=0;const T=()=>de++,W={setMode:B=>{n.mode=B,T()},setLeg:B=>{n.legs.length===2&&(n.forceLeg=B,T())},cycleLeg:()=>W.setLeg(n.forceLeg===-1?0:n.forceLeg===0?1:-1),togglePick:()=>{n.pick=!n.pick,fe.classList.toggle("geo-hidden",!n.pick),Q.setFineCoverage(n.pick?O.boxes:Ne),n.pickPending=n.pick&&O.pending?Oe(O.pending.lat,O.pending.lon):null,n.pick&&O.render(),T()},pickPoint:(B,p,I,L)=>{const k=Qe(B,p,I,L,v.fovRad,v.dist);if(!k.hit){O.missed();return}const[H,R]=Yt(_t(v.rtArray,k.hitPoint));O.addPoint(H,R)},setScale:B=>x.setScale(B),setAuto:B=>x.setAuto(B),toggleQuiet:()=>{n.quiet=!n.quiet,x.setQuiet(n.quiet),Kt("quiet",n.quiet)},toggleStats:()=>{n.showStats=!n.showStats},scaleSigma:B=>{n.sigmaScale=Math.max(.05,Math.min(20,n.sigmaScale*B)),T()},reset:()=>{v.reset(),n.sigmaScale=1,T()},fullscreen:()=>document.fullscreenElement?document.exitFullscreen():i.requestFullscreen(),cycleGpu:()=>{C.destroy(),Je(i,{...t,gpu:Rt[t.gpu]||"high"})},clearCache:async()=>{await tt(),w("weight cache cleared")}};x.setQuiet(n.quiet),t.pick&&W.togglePick(),new Ct(g,v,n,W);const we=new It(a("geo-toolbar"),n,W);t.autofocus&&g.focus({preventScroll:!0});function pe(){const B=Math.min(window.devicePixelRatio||1,2);let p=Math.max(1,Math.round((g.clientWidth||i.clientWidth)*B)),I=Math.max(1,Math.round((g.clientHeight||i.clientHeight)*B));const L=Math.max(p,I);if(L>ze){const k=ze/L;p=Math.round(p*k),I=Math.round(I*k)}if(p*I>t.maxPixels){const k=Math.sqrt(t.maxPixels/(p*I));p=Math.round(p*k),I=Math.round(I*k)}(p!==g.width||I!==g.height)&&(g.width=p,g.height=I,n.canvasW=p,n.canvasH=I,v.viewH=I,T())}const Ce=new ResizeObserver(()=>{pe(),(v.autoSpin||n.reveal!=="done")&&v.fitView(n.canvasW,n.canvasH)});Ce.observe(i),S.push(()=>Ce.disconnect()),pe(),t.spin&&v.fitView(n.canvasW,n.canvasH);const Ae=F.signal;let j;try{j=await Promise.all(A.map(B=>q.fetchManifest(s,B.stem,Ae)))}catch(B){return M?void 0:m(new Error(`cannot load weight manifests from ${s} (${B.message})`))}const $=j.map(B=>q.plan(B,h.device)),Z=h.maxGridBuffers,V=B=>B.reduce((p,I)=>p+$[I].buffers.length,0)<=Z;let K;if(t.legs==="coarse"&&V([0]))K=[0],n.fine={status:"skipped",progress:0,error:"lite mode (coarse only)"};else if(V([0,1]))K=[0,1];else if(V([0]))K=[0],n.fine={status:"skipped",progress:0,error:`adapter allows ${Z} grid buffers`};else if(V([1]))K=[1],n.fine={status:"skipped",progress:0,error:"coarse leg does not fit"};else return m(new Error(`neither model fits: adapter allows ${Z} storage buffers of ${z(h.limits.maxStorageBufferBindingSize)}`));const We=K.reduce((B,p)=>B+$[p].totalBytes,0);console.log(`[plan] legs ${K.map(B=>A[B].key).join("+")} \xB7 ${z(We)} \xB7 buffers ${K.map(B=>$[B].buffers.length).join("+")} / ${Z}`);const X=K[0];f(`Downloading the planet \xB7 ${z($[X].totalBytes)}`),w(t.cache?"first visit downloads the weights; they are cached for next time":"");let ee;try{ee=await q.loadLeg(h.device,s,A[X].stem,{manifest:j[X],useCache:t.cache,signal:Ae,onProgress:B=>{l(B),w(`${(B*100).toFixed(0)}%`)}})}catch(B){return M?void 0:m(B)}if(M){ee.dispose?.();return}Q.setLegs([ee]),n.legs=[A[X].key],l(1),f(t.quietLoad?"almost there":"Sharpening\u2026"),w(""),n.reveal="pending",n.revealBy=performance.now()+4e3,T(),K.length===2&&(n.fine={status:"loading",progress:0,error:""},q.loadLeg(h.device,s,A[1].stem,{manifest:j[1],useCache:t.cache,signal:Ae,onProgress:B=>{n.fine.progress=B}}).then(B=>{if(M){B.dispose?.();return}Q.setLegs([ee,B]),n.legs=["coarse","fine"],n.fine={status:"ready",progress:1,error:""},we.sync(),T()}).catch(B=>{M||(console.error(B),n.fine={status:"error",progress:0,error:B.message})}));let me=null,Ie=null,ae=-1e9;const je=()=>[...v.rotation,v.dist,v.fovRad],$e=()=>[n.canvasW,n.canvasH,n.mode,n.sigmaScale,n.forceLeg,n.pick?1:0,Q.version,de],ve=(B,p)=>B&&B.length===p.length&&B.every((I,L)=>I===p[L]),xe=()=>({rt:v.rtArray,dist:v.dist,fovRad:v.fovRad,W:n.canvasW,H:n.canvasH,mode:n.mode,sigmaScale:n.sigmaScale,forceLeg:n.forceLeg,showCoverage:n.pick,pick:n.pick?n.pickPending:null}),Me=B=>(p,I)=>{U.gpuTime(performance.now(),p,I),x.onGpuTime(p,B)},Ze=B=>((!n.refine||n.refine.stride!==B)&&(n.refine={stride:B,k:0,n:B*B,since:0}),n.refine);function De(B){if(M)return;v.tick(B);const p=je(),I=$e(),L=!ve(me,p),k=!ve(Ie,I),H=n.canvasW,R=n.canvasH,oe=Pt[n.quiet?"quiet":"normal"];if(L||k){if(Q.inFlight<Lt&&B-ae>=1e3/oe){me=p,Ie=I,ae=B;const[y,P]=x.interactiveSize(H,R),D=Ze(x.refineStride(y,P,H,R)),J=x.movingPasses(y,P,H,R,D.stride),be=[];for(let ye=0;ye<J;ye++)be.push({stride:D.stride,phase:D.k%D.n}),D.k++;D.since=J,U.submitted(B,y*P+J*Math.ceil(H/D.stride)*Math.ceil(R/D.stride)),Q.frame({view:xe(),lo:{W:y,H:P},reproject:{invalidate:k},refine:be,onGpuTime:Me({lo:[y,P],passes:J,stride:D.stride})})}}else if(n.refine&&n.refine.since<n.refine.n&&Q.inFlight===0){const y=n.refine,P=x.passCostMs(...x.interactiveSize(H,R),H,R,y.stride),D=[];let J=x.targetMs;do D.push({stride:y.stride,phase:y.k%y.n}),y.k++,y.since++,J-=P;while(y.since<y.n&&J>=P&&D.length<8);ae=B,U.submitted(B,D.length*Math.ceil(H/y.stride)*Math.ceil(R/y.stride)),Q.frame({view:xe(),refine:D,onGpuTime:Me({lo:null,passes:D.length,stride:y.stride})})}n.reveal==="pending"&&Q.inFlight===0&&(n.refine&&n.refine.since>=n.refine.n||B>n.revealBy)&&(n.reveal="done",b(),t.spin&&(v.autoSpin=t.spin===!0?Ht:t.spin)),U.update(B),qe.update(n,U,Q),we.sync(),d=requestAnimationFrame(De)}d=requestAnimationFrame(De)}return C.ready=G(),C}export{Je as mountNeuralEarth};
