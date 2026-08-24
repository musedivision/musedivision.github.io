class te{static async create(e,s={}){if(!navigator.gpu)throw new Error("WebGPU is not available in this browser. Use Chrome / Edge 113+ (or Safari 26+, Firefox 141+) on a real GPU.");const i=s.forceFallbackAdapter?{forceFallbackAdapter:!0}:{powerPreference:s.powerPreference||"high-performance"},r=await navigator.gpu.requestAdapter(i);if(!r)throw new Error(`No WebGPU adapter for ${JSON.stringify(i)}.`);const a=r.limits,o=[];r.features.has("timestamp-query")&&o.push("timestamp-query");const g=await r.requestDevice({requiredFeatures:o,requiredLimits:{maxStorageBufferBindingSize:a.maxStorageBufferBindingSize,maxBufferSize:a.maxBufferSize,maxStorageBuffersPerShaderStage:a.maxStorageBuffersPerShaderStage,maxUniformBufferBindingSize:a.maxUniformBufferBindingSize}}),l=e.getContext("webgpu"),f=navigator.gpu.getPreferredCanvasFormat();l.configure({device:g,format:f,alphaMode:"opaque"});let c={};try{c=r.info||(r.requestAdapterInfo?await r.requestAdapterInfo():{})}catch{}const u=c.description||[c.vendor,c.architecture].filter(Boolean).join(" ")||"unknown GPU";return console.log("[WebGPU] adapter:",u,c,"limits:",{bindingSize:a.maxStorageBufferBindingSize,bufferSize:a.maxBufferSize,storageBuffers:a.maxStorageBuffersPerShaderStage,timestamps:o.includes("timestamp-query")}),new te(g,l,f,r,c,u,!!s.forceFallbackAdapter)}constructor(e,s,i,r,a,o,g){this.device=e,this.ctx=s,this.format=i,this.adapter=r,this.info=a,this.gpuName=o,this.isFallback=g,this.hasTimestamps=e.features.has("timestamp-query"),this.onLost=null,e.lost.then(l=>{l.reason!=="destroyed"&&(console.error("WebGPU device lost:",l.message),this.onLost?.(l))})}get limits(){return this.device.limits}get maxGridBuffers(){return this.device.limits.maxStorageBuffersPerShaderStage-1}}function Se(t,e){const s=[],i=[];let r=null;for(const a of t){const o=a.stride_vertices??a.size_vertices;if(o>e)throw new Error(`level of ${o} vertices exceeds buffer cap ${e}`);(!r||r.vertexCount+o>e)&&(r={startVertex:a.start_vertex,vertexCount:0},s.push(r)),i.push({bufIdx:s.length-1,localBase:a.start_vertex-r.startVertex}),r.vertexCount+=o}return{buffers:s,placement:i}}const he="neural-earth-weights-v1",K=8*1024*1024;class F{static async fetchManifest(e,s,i){const r=await fetch(`${e}/${s}.manifest.json`,{signal:i});if(!r.ok)throw new Error(`fetch ${s}.manifest.json: ${r.status}`);return r.json()}static bytesPerVertex(e){return e.grid?.dtype==="int8"?2:4}static plan(e,s){const i=F.bytesPerVertex(e),r=e.grid?.block_vertices??64;let a=Math.floor(Math.min(s.limits.maxStorageBufferBindingSize,s.limits.maxBufferSize)/i);a-=a%r;const{buffers:o,placement:g}=Se(e.levels,a),l=e.levels.reduce((f,c)=>f+(c.stride_vertices??c.size_vertices),0)*i;return{buffers:o,placement:g,totalBytes:l,bpv:i}}static async loadLeg(e,s,i,r={}){const{onProgress:a=()=>{},signal:o,useCache:g=!0}=r,l=r.manifest||await F.fetchManifest(s,i,o),{buffers:f,placement:c,totalBytes:u,bpv:p}=F.plan(l,e),b=p===2,m=l.mlp?.file||`${i}.mlp.f32.bin`,k=l.grid?.file||`${i}.grid.f16.bin`;a(0,"mlp");const w=await fetch(`${s}/${m}`,{signal:o});if(!w.ok)throw new Error(`fetch ${m}: ${w.status}`);const L=new Float32Array(await w.arrayBuffer());let _=null;if(b){const n=await fetch(`${s}/${l.grid.scales_file}`,{signal:o});if(!n.ok)throw new Error(`fetch ${l.grid.scales_file}: ${n.status}`);_=new Uint32Array(await n.arrayBuffer())}let S=0;const E=f.map((n,M)=>{const $=n.vertexCount*p,z={gpu:e.createBuffer({label:`${i} grid ${M}`,size:$,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST}),startByte:S,endByte:S+$};return S+=$,z});if(S!==u)throw new Error("grid plan does not tile the blob");const d=new Uint8Array(K);let v=0,P=0,x=0;const R=()=>{let n=0;for(;n<v;){const M=P+n;for(;E[x].endByte<=M;)x++;const $=E[x],z=Math.min(v-n,$.endByte-M);e.queue.writeBuffer($.gpu,M-$.startByte,d,n,z),n+=z}P+=v,v=0};try{if(await Pe(`${s}/${k}`,{signal:o,useCache:g,expectBytes:u,onChunk:n=>{let M=0;for(;M<n.length;){const $=Math.min(n.length-M,K-v);d.set(n.subarray(M,M+$),v),v+=$,M+=$,v===K&&R()}},onProgress:n=>a(n,"grid")}),v%4!==0)throw new Error("grid blob length is not a multiple of 4");if(R(),P!==u)throw new Error(`grid blob truncated: got ${P} of ${u} bytes`)}catch(n){throw E.forEach(M=>M.gpu.destroy()),n}return a(1,"done"),{stem:i,manifest:l,mlp:L,isInt8:b,scales:_,bpv:p,gridBuffers:E.map(n=>n.gpu),placement:c,nGridBuffers:f.length,bytes:u+L.byteLength+(_?_.byteLength:0),dispose(){this.gridBuffers.forEach(n=>n.destroy()),this.gridBuffers=[]}}}}async function Pe(t,{signal:e,useCache:s,expectBytes:i,onChunk:r,onProgress:a}){let o=null,g=null;if(s&&typeof caches<"u")try{g=await caches.open(he),o=await g.match(t),o&&console.log("[weights] cache hit",t)}catch{g=null}if(!o){if(o=await fetch(t,{signal:e}),!o.ok)throw new Error(`fetch ${t}: ${o.status}`);g&&await Me(i)&&g.put(t,o.clone()).catch(u=>console.warn("[weights] cache put failed",u))}const l=Number(o.headers.get("content-length"))||i||0;if(!o.body){const u=new Uint8Array(await o.arrayBuffer());r(u),a(1);return}const f=o.body.getReader();let c=0;for(;;){const{done:u,value:p}=await f.read();if(u)break;r(p),c+=p.byteLength,l&&a(Math.min(1,c/l))}}async function Me(t){try{const{quota:e=0,usage:s=0}=await navigator.storage.estimate();return e-s>t*1.2}catch{return!1}}async function Be(){try{return await caches.delete(he)}catch{return!1}}const J={NLEV:16,MAX_LEGS:2},N=64*32+64*64+7*64,ue=6,ge=4+16*3,Z=(ue+J.MAX_LEGS*ge)*16,ee=3;function $e(t){const e=Array.from({length:t},(r,a)=>`@group(0) @binding(${ee+a}) var<storage, read> g${a} : array<u32>;`).join(`
`),s=Array.from({length:t},(r,a)=>`    case ${a}u: { return g${a}[idx]; }`).join(`
`),i=ee+t;return`
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
  legs  : array<Leg, 2>,
};

@group(0) @binding(0) var<uniform> P : Params;
@group(0) @binding(1) var<storage, read> mlp : array<f32>;
@group(0) @binding(2) var<storage, read> scales : array<u32>; 
${e}
@group(0) @binding(${i}) var outTex : texture_storage_2d<rgba8unorm, write>;

const PRIME1 : u32 = 2654435761u;
const PRIME2 : u32 = 805459861u;
const TWO_PI : f32 = 6.283185307179586;

fn fetchVtx(bufIdx : u32, idx : u32) -> u32 {
  switch (bufIdx) {
${s}
    default: { return 0u; }
  }
}


fn erf_pos(x : f32) -> f32 {
  let t = 1.0 / (1.0 + 0.3275911 * x);
  let y = 1.0 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t
            - 0.284496736) * t + 0.254829592) * t * exp(-x * x);
  return y;
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
  let isInt8 = P.legs[L].info.z == 1u;
  for (var l = 0u; l < 16u; l = l + 1u) {
    let lvl = P.legs[L].levels[l];
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
      let idx = lvl.b.x + local;            
      var two : vec2<f32>;
      if (isInt8) {
        
        let q = unpack4x8snorm(fetchVtx(lvl.m.w, idx >> 1u));
        let pair = select(q.xy, q.zw, (idx & 1u) == 1u);
        two = pair * unpack2x16float(scales[lvl.b.y + (local >> 6u)]);
      } else {
        two = unpack2x16float(fetchVtx(lvl.m.w, idx));
      }
      acc = acc + w * two;
    }
    
    let denom = 2.8284271247461903 * s * lvl.sg.y;
    let g = erf_pos(1.0 / denom);
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

fn rt() -> mat3x3<f32> {
  
  return mat3x3<f32>(
    vec3<f32>(P.rt0.x, P.rt1.x, P.rt2.x),
    vec3<f32>(P.rt0.y, P.rt1.y, P.rt2.y),
    vec3<f32>(P.rt0.z, P.rt1.z, P.rt2.z),
  );
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let W = u32(P.cam.z);
  let H = u32(P.cam.w);
  if (gid.x >= W || gid.y >= H) { return; }

  let camDist = P.cam.x;
  let fov = P.cam.y;
  let aspect = P.cam.z / P.cam.w;
  let halfH = tan(fov * 0.5);
  let halfW = halfH * aspect;
  let gx = -halfW + (f32(gid.x) / max(P.cam.z - 1.0, 1.0)) * 2.0 * halfW;
  let gy = halfH - (f32(gid.y) / max(P.cam.w - 1.0, 1.0)) * 2.0 * halfH;
  let dir = normalize(vec3<f32>(gx, gy, -1.0));

  let b = 2.0 * dir.z * camDist;
  let cc = camDist * camDist - 1.0;
  let disc = b * b - 4.0 * cc;

  var col = vec3<f32>(0.02, 0.03, 0.06); 
  if (disc >= 0.0) {
    let t = (-b - sqrt(disc)) * 0.5;
    let hp = vec3<f32>(0.0, 0.0, camDist) + t * dir;
    let facing = clamp(dot(hp, -dir), 0.0, 1.0);
    
    
    
    
    
    
    let sigmaRaw = (fov / max(P.cam.w - 1.0, 1.0)) * t / max(facing, 1e-2) / TWO_PI;
    let sigma = sigmaRaw * P.opt.x;
    let model = rt() * hp;
    let coord = (model + vec3<f32>(1.0)) * 0.5;

    
    var L = 0u;
    if (P.route.y > 1.5) {
      if (P.route.z >= 0.0) { L = u32(P.route.z); }
      else if (sigmaRaw < P.route.x) { L = 1u; }
    }
    let leg = P.legs[L];

    var feat : array<f32, 32>;
    encodeGate(L, coord, sigma, &feat);
    var o : array<f32, 7>;
    runMLP(leg.info.x, &feat, &o);

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
    } else if (mode == 2u) { col = vec3<f32>(0.0, tree, 0.0); }
    else { col = rgb * factor; } 

    
    if (P.route.w > 0.5 && L == 1u) { col = mix(col, vec3<f32>(1.0, 0.55, 0.1), 0.35); }
  }

  textureStore(outTex, vec2<i32>(i32(gid.x), i32(gid.y)), vec4<f32>(col, 1.0));
}
`}const ze=`
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
`,ke={rgb:0,dem:1,tree:2,lit:3},Le=-500,Ee=8848,fe=3e-5;function Ae(t){return(ue+t*ge)*4}function Ge(t,e,s,i,r){const a=i.manifest,o=Ae(s),g=a.model_kind==="decomp_unigrid"?1:0,l=a.decomp||{};t[o+0]=a.dem.dem_scale,t[o+1]=a.dem.dem_grad_scale,t[o+2]=a.gating.std_scale,t[o+3]=0,t[o+4]=g,t[o+5]=l.h_bound||0,t[o+6]=l.offset_mean||0,t[o+7]=l.offset_std||1,t[o+8]=l.log_scale_mean||0,t[o+9]=l.log_scale_std||1,t[o+10]=0,t[o+11]=0;const f=a.grid?.dtype==="int8",c=a.grid?.block_vertices??64;if(f&&c!==64)throw new Error(`int8 block must be 64 (kernel shifts by 6), got ${c}`);e[o+12]=s*N,e[o+13]=a.levels.length,e[o+14]=f?1:0,e[o+15]=0;const u=a.gating.gating_resolutions;if(a.levels.length!==J.NLEV)throw new Error(`expected ${J.NLEV} levels, got ${a.levels.length}`);a.levels.forEach((p,b)=>{const m=o+16+b*12;t[m+0]=p.scale,t[m+1]=u[b],t[m+2]=0,t[m+3]=0,e[m+4]=p.size_vertices,e[m+5]=p.resolution,e[m+6]=p.dense?1:0,e[m+7]=r+i.placement[b].bufIdx,e[m+8]=i.placement[b].localBase,e[m+9]=f?(i.scaleOffset??0)+p.start_vertex/c:0,e[m+10]=0,e[m+11]=0})}function qe(t,e){const s=e.rt;t[0]=s[0],t[1]=s[1],t[2]=s[2],t[3]=0,t[4]=s[3],t[5]=s[4],t[6]=s[5],t[7]=0,t[8]=s[6],t[9]=s[7],t[10]=s[8],t[11]=0,t[12]=e.dist,t[13]=e.fovRad,t[14]=e.W,t[15]=e.H,t[16]=e.sigmaScale,t[17]=ke[e.mode]??0,t[18]=Le,t[19]=Ee,t[20]=e.crossover,t[21]=e.nLegs,t[22]=e.forceLeg,t[23]=e.showRouting?1:0}const Ue=3,de=4;class Re{constructor(e){this.gpu=e,this.device=e.device,this.legs=[],this.crossover=fe,this.version=0,this.inFlight=0,this.onError=null,this.uniform=this.device.createBuffer({label:"params",size:Z,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST}),this._ubuf=new ArrayBuffer(Z),this._f32=new Float32Array(this._ubuf),this._u32=new Uint32Array(this._ubuf),this.sampler=this.device.createSampler({magFilter:"linear",minFilter:"linear"});const s=this.device.createShaderModule({code:ze,label:"present"});this.presentPipeline=this.device.createRenderPipeline({layout:"auto",vertex:{module:s,entryPoint:"vs"},fragment:{module:s,entryPoint:"fs",targets:[{format:e.format}]},primitive:{topology:"triangle-list"}}),this._targets=new Map,this.computePipeline=null,this.mlpBuffer=null,this.bytes={grids:0,mlp:0,targets:0,uniform:Z,timing:0},this._initTiming()}_initTiming(){if(this.timing=null,!this.gpu.hasTimestamps)return;const e=this.device;this.timing={querySet:e.createQuerySet({type:"timestamp",count:2}),resolve:e.createBuffer({size:16,usage:GPUBufferUsage.QUERY_RESOLVE|GPUBufferUsage.COPY_SRC}),readbacks:Array.from({length:de},()=>({buf:e.createBuffer({size:16,usage:GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST}),busy:!1}))},this.bytes.timing=16*(de+1)}setLegs(e){if(e.length<1||e.length>2)throw new Error("1 or 2 legs supported");this.legs=e,this.crossover=e[e.length-1].manifest.routing?.sigma_crossover??fe,this.mlpBuffer?.destroy();const s=new Float32Array(N*e.length);e.forEach((l,f)=>{if(l.mlp.length!==N)throw new Error(`leg ${f}: mlp has ${l.mlp.length} floats`);s.set(l.mlp,f*N)}),this.mlpBuffer=this.device.createBuffer({label:"mlp",size:s.byteLength,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST}),this.device.queue.writeBuffer(this.mlpBuffer,0,s),this.scalesBuffer?.destroy();let i=0;for(const l of e)l.scaleOffset=i,i+=l.scales?l.scales.length:0;const r=new Uint32Array(Math.max(1,i));for(const l of e)l.scales&&r.set(l.scales,l.scaleOffset);this.scalesBuffer=this.device.createBuffer({label:"int8 scales",size:r.byteLength,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST}),this.device.queue.writeBuffer(this.scalesBuffer,0,r);let a=0;e.forEach((l,f)=>{Ge(this._f32,this._u32,f,l,a),a+=l.nGridBuffers}),this.nGrid=a;const o=$e(this.nGrid),g=this.device.createShaderModule({code:o,label:`forward x${e.length}`});g.getCompilationInfo().then(l=>{const f=l.messages.filter(u=>u.type==="error");if(!f.length)return;const c=f.map(u=>`L${u.lineNum}:${u.linePos} ${u.message}`).join(`
`);console.error(`[forward WGSL]
`+c),this.onError?.(new Error(`shader compile failed:
${c}`))}),this.computePipeline=this.device.createComputePipeline({layout:"auto",compute:{module:g,entryPoint:"main"}}),this._destroyTargets(),this.bytes.grids=e.reduce((l,f)=>l+f.gridBuffers.reduce((c,u)=>c+u.size,0),0),this.bytes.mlp=s.byteLength+r.byteLength,this.version++}get totalBytes(){const e=this.bytes;return e.grids+e.mlp+e.targets+e.uniform+e.timing}_destroyTargets(){for(const e of this._targets.values())e.tex.destroy();this._targets.clear(),this.bytes.targets=0}_target(e,s){const i=`${e}x${s}`;let r=this._targets.get(i);if(r)return this._targets.delete(i),this._targets.set(i,r),r;if(this._targets.size>=Ue){const[f,c]=this._targets.entries().next().value;c.tex.destroy(),this._targets.delete(f),this.bytes.targets-=c.bytes}const a=this.device.createTexture({label:`target ${i}`,size:[e,s],format:"rgba8unorm",usage:GPUTextureUsage.STORAGE_BINDING|GPUTextureUsage.TEXTURE_BINDING|GPUTextureUsage.COPY_SRC}),o=a.createView(),g=[{binding:0,resource:{buffer:this.uniform}},{binding:1,resource:{buffer:this.mlpBuffer}},{binding:2,resource:{buffer:this.scalesBuffer}}];let l=ee;for(const f of this.legs)for(const c of f.gridBuffers)g.push({binding:l++,resource:{buffer:c}});return g.push({binding:l,resource:o}),r={tex:a,view:o,W:e,H:s,bytes:e*s*4,computeBind:this.device.createBindGroup({layout:this.computePipeline.getBindGroupLayout(0),entries:g}),presentBind:this.device.createBindGroup({layout:this.presentPipeline.getBindGroupLayout(0),entries:[{binding:0,resource:o},{binding:1,resource:this.sampler}]})},this._targets.set(i,r),this.bytes.targets+=r.bytes,r}render(e,s){if(!this.computePipeline)return;const i=this._target(e.W,e.H);qe(this._f32,{...e,crossover:this.crossover,nLegs:this.legs.length}),this.device.queue.writeBuffer(this.uniform,0,this._ubuf);const r=this.timing?.readbacks.find(c=>!c.busy)||null,a=this.device.createCommandEncoder(),o=a.beginComputePass(r?{timestampWrites:{querySet:this.timing.querySet,beginningOfPassWriteIndex:0,endOfPassWriteIndex:1}}:{});o.setPipeline(this.computePipeline),o.setBindGroup(0,i.computeBind),o.dispatchWorkgroups(Math.ceil(e.W/8),Math.ceil(e.H/8)),o.end();const g=a.beginRenderPass({colorAttachments:[{view:this.gpu.ctx.getCurrentTexture().createView(),loadOp:"clear",storeOp:"store",clearValue:{r:0,g:0,b:0,a:1}}]});g.setPipeline(this.presentPipeline),g.setBindGroup(0,i.presentBind),g.draw(3),g.end(),r&&(r.busy=!0,a.resolveQuerySet(this.timing.querySet,0,2,this.timing.resolve,0),a.copyBufferToBuffer(this.timing.resolve,0,r.buf,0,16));const l=performance.now();this.inFlight++,this.device.queue.submit([a.finish()]);const f=this.device.queue.onSubmittedWorkDone().then(()=>{this.inFlight--});r?r.buf.mapAsync(GPUMapMode.READ).then(()=>{const c=new BigUint64Array(r.buf.getMappedRange()),u=Number(c[1]-c[0]);r.buf.unmap(),r.busy=!1,u>0&&s?.(u/1e6,!1)}).catch(()=>{r.busy=!1}):f.then(()=>s?.(performance.now()-l,!0))}dispose(){this._destroyTargets(),this.uniform.destroy(),this.mlpBuffer?.destroy(),this.scalesBuffer?.destroy();for(const e of this.legs)e.dispose?.();this.legs=[],this.timing&&(this.timing.resolve.destroy(),this.timing.readbacks.forEach(e=>e.buf.destroy()),this.timing.querySet.destroy())}}const A={sub:(t,e)=>[t[0]-e[0],t[1]-e[1],t[2]-e[2]],dot:(t,e)=>t[0]*e[0]+t[1]*e[1]+t[2]*e[2],cross:(t,e)=>[t[1]*e[2]-t[2]*e[1],t[2]*e[0]-t[0]*e[2],t[0]*e[1]-t[1]*e[0]],norm:t=>Math.hypot(t[0],t[1],t[2]),normalize:t=>{const e=Math.hypot(t[0],t[1],t[2])||1e-12;return[t[0]/e,t[1]/e,t[2]/e]}},G={identity:()=>[1,0,0,0,1,0,0,0,1],globeDefault:()=>[0,1,0,0,0,1,1,0,0],transpose:t=>[t[0],t[3],t[6],t[1],t[4],t[7],t[2],t[5],t[8]],mul:(t,e)=>{const s=new Array(9);for(let i=0;i<3;i++)for(let r=0;r<3;r++)s[i*3+r]=t[i*3]*e[r]+t[i*3+1]*e[3+r]+t[i*3+2]*e[6+r];return s},mulVec:(t,e)=>[t[0]*e[0]+t[1]*e[1]+t[2]*e[2],t[3]*e[0]+t[4]*e[1]+t[5]*e[2],t[6]*e[0]+t[7]*e[1]+t[8]*e[2]],axisAngle:(t,e)=>{const s=A.normalize(t),i=Math.sin(e),r=Math.cos(e),[a,o,g]=s,l=[0,-g,o,g,0,-a,-o,a,0],f=G.mul(l,l),c=new Array(9);for(let u=0;u<9;u++)c[u]=(u%4===0?1:0)+i*l[u]+(1-r)*f[u];return c},between:(t,e)=>{const s=A.normalize(t),i=A.normalize(e),r=A.cross(s,i),a=A.norm(r),o=Math.max(-1,Math.min(1,A.dot(s,i)));if(a<1e-8)return G.identity();const g=[r[0]/a,r[1]/a,r[2]/a],[l,f,c]=g,u=[0,-c,f,c,0,-l,-f,l,0],p=G.mul(u,u),b=new Array(9);for(let m=0;m<9;m++)b[m]=(m%4===0?1:0)+a*u[m]+(1-o)*p[m];return b}},Ce=Math.PI*2;function pe(t,e,s,i,r,a){const o=s/i,g=Math.tan(r/2),l=g*o,f=-l+t/Math.max(s-1,1)*2*l,c=g-e/Math.max(i-1,1)*2*g;let u=f,p=c,b=-1;const m=Math.hypot(u,p,b);u/=m,p/=m,b/=m;const k=2*b*a,w=a*a-1,L=k*k-4*w;if(L<0)return{hit:!1,dir:[u,p,b],hitPoint:[0,0,0],facing:0,sigma:0};const _=(-k-Math.sqrt(L))/2,S=[u*_,p*_,a+b*_],E=Math.max(0,Math.min(1,-(S[0]*u+S[1]*p+S[2]*b))),d=r/Math.max(i-1,1)/Math.max(E,.01)/Ce,v=d*_;return{hit:!0,dir:[u,p,b],hitPoint:S,facing:E,sigma:d,sigmaSurface:v}}class Fe{constructor(){this.rotation=G.globeDefault(),this.dist=2.4,this.minAlt=.004,this.maxDist=10,this.fovDeg=45,this._dragStart=null,this._spinAxis=null,this._spinSpeed=0,this._friction=.93}get fovRad(){return this.fovDeg*Math.PI/180}get rtArray(){return G.transpose(this.rotation)}reset(){this.rotation=G.globeDefault(),this.dist=2.4,this._spinAxis=null,this._spinSpeed=0}zoom(e){const s=Math.pow(.92,e),i=Math.max(this.minAlt,Math.min(this.maxDist-1,(this.dist-1)*s));this.dist=1+i}beginDrag(e,s,i,r){const a=pe(e,s,i,r,this.fovRad,this.dist);this._dragStart=a.hit?a.hitPoint:null,this._dragging=!0,this._spinSpeed=0,this._spinAxis=null}endDrag(){this._dragStart=null,this._dragging=!1}drag(e,s,i,r){if(!this._dragStart)return;const a=pe(e,s,i,r,this.fovRad,this.dist);if(!a.hit)return;const o=this._dragStart,g=G.between(o,a.hitPoint);this.rotation=G.mul(g,this.rotation),this._dragStart=a.hitPoint;const l=A.cross(o,a.hitPoint),f=Math.acos(Math.max(-1,Math.min(1,A.dot(a.hitPoint,o)))),c=A.norm(l);if(c>1e-8&&f>1e-6){const u=[l[0]/c,l[1]/c,l[2]/c],p=.6;this._spinAxis?(this._spinAxis=A.normalize([p*u[0]+(1-p)*this._spinAxis[0],p*u[1]+(1-p)*this._spinAxis[1],p*u[2]+(1-p)*this._spinAxis[2]]),this._spinSpeed=p*f+(1-p)*this._spinSpeed):(this._spinAxis=u,this._spinSpeed=f)}}tick(){this._dragging||this._spinAxis&&this._spinSpeed>1e-5&&(this.rotation=G.mul(G.axisAngle(this._spinAxis,this._spinSpeed),this.rotation),this._spinSpeed*=this._friction,this._spinSpeed<1e-5&&(this._spinSpeed=0,this._spinAxis=null))}}class De{constructor(e={}){this.auto=e.auto??!0,this.scale=e.scale??.5,this.stillScale=e.stillScale??1,this.minScale=e.minScale??.2,this.targetMs=e.targetMs??22,this.maxScale=e.maxScale??1,this.maxStillMs=e.maxStillMs??700,this._ema=0,this._samples=0}onGpuTime(e,s){if(e==="still"){s>this.maxStillMs&&this.stillScale>this.scale&&(this.stillScale=Math.max(this.scale,this.stillScale*.8));return}if(!this.auto||(this._ema=this._ema?this._ema*.7+s*.3:s,++this._samples<4))return;this._samples=0;const i=this._ema/this.targetMs;i>1.25?this.scale=Math.max(this.minScale,this.scale/Math.sqrt(Math.min(i,4))):i<.6&&this.scale<this.maxScale&&(this.scale=Math.min(this.maxScale,this.scale*1.15)),this.scale=Math.min(this.scale,this.maxScale),this.scale=Math.round(this.scale*100)/100}nudge(e){this.auto=!1,this.scale=Math.round(Math.max(this.minScale,Math.min(1,this.scale+e))*100)/100}toggleAuto(){this.auto=!this.auto,this._ema=0,this._samples=0}setQuiet(e){this.targetMs=e?10:22,this.maxScale=e?.5:1,this.scale=Math.min(this.scale,this.maxScale),this._ema=0,this._samples=0}interactiveSize(e,s){return me(e,s,this.scale)}stillSize(e,s){return me(e,s,this.stillScale)}}function me(t,e,s){return[Math.max(8,Math.round(t*s)),Math.max(8,Math.round(e*s))]}const ve={Digit1:"rgb",Digit2:"dem",Digit3:"tree",Digit4:"lit"};class Te{constructor(e,s,i,r){this.canvas=e,this.camera=s,this.state=i,this.actions=r,this._pointers=new Map,this._pinchDist=0,this._bind()}_bind(){const e=this.canvas;e.addEventListener("pointerdown",i=>{e.setPointerCapture(i.pointerId),this._pointers.set(i.pointerId,i),this._pointers.size===1?(e.classList.add("dragging"),this.camera.beginDrag(...this._px(i))):this._pointers.size===2&&(this.camera.endDrag(),this._pinchDist=this._pinch())}),e.addEventListener("pointermove",i=>{if(this._pointers.has(i.pointerId)){if(this._pointers.set(i.pointerId,i),this._pointers.size===1)this.camera.drag(...this._px(i));else if(this._pointers.size===2){const r=this._pinch();this._pinchDist>0&&r>0&&this.camera.zoom(Math.log(this._pinchDist/r)/Math.log(.92)),this._pinchDist=r}}});const s=i=>{if(this._pointers.delete(i.pointerId),this._pointers.size===0)e.classList.remove("dragging"),this.camera.endDrag();else if(this._pointers.size===1){const[r]=this._pointers.values();this.camera.beginDrag(...this._px(r))}};e.addEventListener("pointerup",s),e.addEventListener("pointercancel",s),e.addEventListener("wheel",i=>{i.preventDefault(),this.camera.zoom(Math.sign(i.deltaY)*(i.shiftKey?3:1))},{passive:!1}),e.addEventListener("keydown",i=>{if(ve[i.code]){this.actions.setMode(ve[i.code]);return}const r=this.actions;switch(i.code){case"BracketLeft":r.scaleSigma(1/1.5);break;case"BracketRight":r.scaleSigma(1.5);break;case"KeyR":r.reset();break;case"KeyF":r.fullscreen();break;case"KeyL":r.cycleLeg();break;case"KeyX":r.toggleRouting();break;case"KeyA":r.toggleAuto();break;case"KeyQ":r.toggleQuiet();break;case"KeyS":r.toggleStats();break;case"Comma":r.nudgeScale(-.1);break;case"Period":r.nudgeScale(.1);break;case"KeyG":r.cycleGpu();break;default:return}i.preventDefault()})}_pinch(){const[e,s]=this._pointers.values();return Math.hypot(e.clientX-s.clientX,e.clientY-s.clientY)}_px(e){const s=this.canvas.getBoundingClientRect(),i=this.state.canvasW/s.width,r=this.state.canvasH/s.height;return[(e.clientX-s.left)*i,(e.clientY-s.top)*r,this.state.canvasW,this.state.canvasH]}}const Ie=[["rgb","RGB"],["dem","DEM"],["tree","Tree"],["lit","Lit"]],We=[[-1,"Auto"],[0,"Coarse"],[1,"Fine"]];class Oe{constructor(e,s,i){this.el=e,this.state=s,this.actions=i,this._buttons=[],this._build()}_group(e,s,i,r,a={}){const o=document.createElement("div");if(o.className="tb-group",a.id&&(o.id=a.id),e){const g=document.createElement("span");g.className="tb-title",g.textContent=e,o.appendChild(g)}for(const[g,l,f]of s){const c=document.createElement("button");c.type="button",c.textContent=l,f&&(c.title=f),c.addEventListener("click",()=>r(g)),this._buttons.push({b:c,isActive:()=>i(g)}),o.appendChild(c)}return this.el.appendChild(o),o}_build(){const e=this.state,s=this.actions;this._group("layer",Ie.map(([i,r],a)=>[i,r,`key ${a+1}`]),i=>e.mode===i,i=>s.setMode(i)),this._legGroup=this._group("model",We.map(([i,r])=>[i,r,"key L cycles"]),i=>e.forceLeg===i,i=>s.setLeg(i),{id:"tb-legs"}),this._group("",[["routing","Show routing","tint pixels drawn by the fine leg (X)"],["auto","Auto quality","tune interactive resolution to GPU speed (A)"],["quiet","Quiet","low GPU power while dragging: 24 renders/s, \u2264\xBD res, 10 ms budget (Q)"],["stats","Stats","toggle the stats overlay (S)"]],i=>({routing:e.showRouting,auto:e.quality.auto,quiet:e.quiet,stats:e.showStats})[i],i=>({routing:s.toggleRouting,auto:s.toggleAuto,quiet:s.toggleQuiet,stats:s.toggleStats})[i]()),this._group("",[["reset","Reset","R"],["fs","Fullscreen","F"],["cache","Clear cache","forget downloaded weights"]],()=>!1,i=>({reset:s.reset,fs:s.fullscreen,cache:s.clearCache})[i]())}sync(){for(const{b:e,isActive:s}of this._buttons)e.classList.toggle("active",!!s());this._legGroup.classList.toggle("disabled",this.state.legs.length<2)}}const He={rgb:"RGB",dem:"DEM hillshade",tree:"Tree cover",lit:"RGB + lighting"};function je(t,e){const s=t.fovRad/Math.max(e-1,1)*(t.dist-1)/(2*Math.PI);return{sigma:s,z:Math.log2(1/(256*s))}}const Ne={"-1":"auto routing",0:"coarse only",1:"fine only"};class Ve{constructor(){this._renders=[],this._gpu=[],this.lastGpuMs=0,this.gpuApprox=!1,this.fps=0,this.busy=0,this.evalsPerSec=0}submitted(e,s,i){this._renders.push({t:e,px:s*i})}gpuTime(e,s,i){this._gpu.push({t:e,ms:s}),this.lastGpuMs=s,this.gpuApprox=i}update(e){const s=e-1e3;for(;this._renders.length&&this._renders[0].t<s;)this._renders.shift();for(;this._gpu.length&&this._gpu[0].t<s;)this._gpu.shift();this.fps=this._renders.length,this.evalsPerSec=this._renders.reduce((i,r)=>i+r.px,0),this.busy=Math.min(1,this._gpu.reduce((i,r)=>i+r.ms,0)/1e3)}}function C(t){return t>=1e9?`${(t/1e9).toFixed(2)} GB`:t>=1e6?`${(t/1e6).toFixed(0)} MB`:t>=1e3?`${(t/1e3).toFixed(0)} kB`:`${t} B`}class Qe{constructor(e){this.el=e}update(e,s,i){if(!e.showStats){this.el.textContent="";return}const{camera:r,mode:a,sigmaScale:o,forceLeg:g,quality:l,legs:f,fine:c,gpuName:u,gpuPref:p,canvasW:b,canvasH:m}=e,[k,w]=l.interactiveSize(b,m),[L,_]=l.stillSize(b,m),S=(f.length===2?`coarse + fine \xB7 ${Ne[g]}`:`${f[0]||"no"} leg`)+` \xB7 ${e.quant}`,E=performance.memory?` \xB7 JS heap ${C(performance.memory.usedJSHeapSize)}`:"",d=i.bytes,{sigma:v,z:P}=je(r,m),x=[`${S}  \xB7  ${He[a]}  \xB7  \u03C3\xD7${o.toFixed(2)}  \xB7  alt ${(r.dist-1).toFixed(3)}  \xB7  centre \u03C3 ${v.toExponential(1)} \u2248 z${P.toFixed(1)}`,`canvas ${b}\xD7${m}  \xB7  interactive ${k}\xD7${w} (${l.auto?"auto ":""}${l.scale.toFixed(2)})  \xB7  still ${L}\xD7${_}${e.quiet?"  \xB7  QUIET":""}`,`${s.fps} renders/s  \xB7  GPU ${s.gpuApprox?"\u2248":""}${s.lastGpuMs.toFixed(1)} ms/frame  \xB7  busy ${(s.busy*100).toFixed(0)}%  \xB7  ${(s.evalsPerSec/1e6).toFixed(1)} M evals/s`,`GPU memory ${C(i.totalBytes)}  (grids ${C(d.grids)} \xB7 targets ${C(d.targets)} \xB7 mlp ${C(d.mlp)})${E}`];c.status==="loading"?x.push(`fine leg: downloading ${(c.progress*100).toFixed(0)}%`):c.status==="error"?x.push(`fine leg: failed \u2014 ${c.error}`):c.status==="skipped"&&x.push(`fine leg: skipped \u2014 ${c.error}`),x.push(`${u}  [${p}]`),this.el.textContent=x.join(`
`)}}const Ye=[{key:"coarse",stem:"coarse"},{key:"fine",stem:"fine"}],be=2048,Xe=140,Ke=1,Je={normal:40,quiet:24},Ze={high:"low",low:"fallback",fallback:"high"},et=(t,e)=>{try{const s=localStorage.getItem(`neural-earth.${t}`);return s===null?e:JSON.parse(s)}catch{return e}},tt=(t,e)=>{try{localStorage.setItem(`neural-earth.${t}`,JSON.stringify(e))}catch{}},st=`
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
  flex-direction: column; gap: 14px; background: #05070d; z-index: 10; text-align: center; padding: 24px; }
.geo-earth .geo-overlay h1 { font-size: 18px; font-weight: 600; letter-spacing: .04em; margin: 0; font-family: inherit; }
.geo-earth .geo-overlay p { color: var(--geo-dim); font-size: 12px; max-width: 520px; line-height: 1.5; margin: 0; }
.geo-earth .geo-status { font-size: 14px; }
.geo-earth .geo-substatus { font-size: 11.5px; color: var(--geo-dim); min-height: 1.2em; }
.geo-earth .geo-bar { width: min(360px, 80%); height: 6px; background: #1a2235; border-radius: 3px; overflow: hidden; }
.geo-earth .geo-barfill { height: 100%; width: 0; background: linear-gradient(90deg, #3a7bd5, var(--geo-acc)); transition: width .15s; }
.geo-earth .geo-overlay.err .geo-status { color: #ff8a8a; }
.geo-earth .geo-overlay.err .geo-bar { display: none; }
.geo-earth .geo-hidden { display: none !important; }
@container (max-width: 720px) {
  .geo-earth .geo-hud { font-size: 10px; } .geo-earth .geo-brand span { display: none; } .geo-earth .geo-help { display: none; }
}
`;function it(){if(document.querySelector("style[data-geo-earth]"))return;const t=document.createElement("style");t.dataset.geoEarth="",t.textContent=st,document.head.appendChild(t)}const rt=`
<canvas class="geo-canvas" tabindex="0"></canvas>
<pre class="geo-hud"></pre>
<div class="geo-brand"><b>Neural Earth</b><span>coarse + fine hash-grid legs \xB7 one WGSL kernel per pixel</span></div>
<div class="geo-toolbar"></div>
<p class="geo-help">drag \xB7 scroll/pinch zoom \xB7 1-4 layer \xB7 L leg \xB7 [ ] \u03C3 \xB7 , . res \xB7 A auto \xB7 Q quiet \xB7 X routing \xB7 S stats \xB7 R reset \xB7 F fullscreen</p>
<div class="geo-overlay">
  <h1>Neural Earth</h1>
  <p>The whole planet is a neural field: each pixel ray-casts the globe, looks up a multi-resolution hash grid and runs a small MLP \u2014 live, in your GPU, with WebGPU.</p>
  <div class="geo-status">Initializing WebGPU\u2026</div>
  <div class="geo-bar"><div class="geo-barfill"></div></div>
  <div class="geo-substatus"></div>
</div>`;function xe(t,e={}){const s={weights:"weights",legs:"both",quant:"fp16",gpu:"high",cache:!0,brand:!0,help:!0,autofocus:!1,onError:null,...e},i=new URL(s.weights,document.baseURI).toString().replace(/\/+$/,""),r=Ye.map(d=>({...d,stem:s.quant==="int8"?`${d.stem}.i8`:d.stem}));it(),t.classList.add("geo-earth"),t.style.containerType||="inline-size",t.innerHTML=rt;const a=d=>t.querySelector(`.${d}`),o=a("geo-canvas"),g=a("geo-overlay"),l=a("geo-status"),f=a("geo-substatus"),c=a("geo-barfill");s.brand||a("geo-brand").remove(),s.help||a("geo-help").remove();const u=d=>l.textContent=d,p=d=>f.textContent=d,b=d=>c.style.width=`${Math.round(d*100)}%`,m=new AbortController,k=[];let w=!1,L=0;const _={root:t,opts:s,ready:null,destroy(){if(!w){w=!0,m.abort(),cancelAnimationFrame(L);for(const d of k.splice(0))try{d()}catch{}t.innerHTML="",t.classList.remove("geo-earth"),t.__geo===_&&delete t.__geo}}};t.__geo=_;const S=d=>{w||(g.classList.add("err"),g.classList.remove("geo-hidden"),u(`Error: ${d.message||d}`),p(""),console.error(d),s.onError?.(d))};async function E(){let d;try{d=await te.create(o,s.gpu==="fallback"?{forceFallbackAdapter:!0}:{powerPreference:s.gpu==="low"?"low-power":"high-performance"})}catch(h){return S(h)}if(w){d.device.destroy();return}d.onLost=h=>S(new Error(`GPU device lost (${h.message}). Reload the page.`));const v=new Fe,P=new Re(d);P.onError=S;const x=new De({scale:d.isFallback?.25:.5}),R=new Ve,n={mode:"rgb",sigmaScale:1,forceLeg:-1,showRouting:!1,showStats:!0,quiet:et("quiet",!1),quant:s.quant,quality:x,camera:v,legs:[],fine:{status:"pending",progress:0,error:""},canvasW:1,canvasH:1,gpuName:d.gpuName,gpuPref:s.gpu},M=new Qe(a("geo-hud"));Object.assign(_,{state:n,renderer:P,quality:x,stats:R,camera:v,gpu:d}),k.push(()=>{P.dispose(),d.device.destroy()});let $=0;const z=()=>$++,V={setMode:h=>{n.mode=h,z()},setLeg:h=>{n.legs.length===2&&(n.forceLeg=h,z())},cycleLeg:()=>V.setLeg(n.forceLeg===-1?0:n.forceLeg===0?1:-1),toggleRouting:()=>{n.showRouting=!n.showRouting,z()},toggleAuto:()=>x.toggleAuto(),toggleQuiet:()=>{n.quiet=!n.quiet,x.setQuiet(n.quiet),tt("quiet",n.quiet)},toggleStats:()=>{n.showStats=!n.showStats},nudgeScale:h=>x.nudge(h),scaleSigma:h=>{n.sigmaScale=Math.max(.05,Math.min(20,n.sigmaScale*h)),z()},reset:()=>{v.reset(),n.sigmaScale=1,z()},fullscreen:()=>document.fullscreenElement?document.exitFullscreen():t.requestFullscreen(),cycleGpu:()=>{_.destroy(),xe(t,{...s,gpu:Ze[s.gpu]||"high"})},clearCache:async()=>{await Be(),p("weight cache cleared")}};x.setQuiet(n.quiet),new Te(o,v,n,V);const se=new Oe(a("geo-toolbar"),n,V);s.autofocus&&o.focus({preventScroll:!0});function ie(){const h=Math.min(window.devicePixelRatio||1,2);let y=Math.max(1,Math.round((o.clientWidth||t.clientWidth)*h)),B=Math.max(1,Math.round((o.clientHeight||t.clientHeight)*h));const U=Math.max(y,B);if(U>be){const D=be/U;y=Math.round(y*D),B=Math.round(B*D)}(y!==o.width||B!==o.height)&&(o.width=y,o.height=B,n.canvasW=y,n.canvasH=B,z())}const re=new ResizeObserver(ie);re.observe(t),k.push(()=>re.disconnect()),ie();const Q=m.signal;let I;try{I=await Promise.all(r.map(h=>F.fetchManifest(i,h.stem,Q)))}catch(h){return w?void 0:S(new Error(`cannot load weight manifests from ${i} (${h.message})`))}const W=I.map(h=>F.plan(h,d.device)),O=d.maxGridBuffers,H=h=>h.reduce((y,B)=>y+W[B].buffers.length,0)<=O;let q;if(s.legs==="coarse"&&H([0]))q=[0],n.fine={status:"skipped",progress:0,error:"lite mode (coarse only)"};else if(H([0,1]))q=[0,1];else if(H([0]))q=[0],n.fine={status:"skipped",progress:0,error:`adapter allows ${O} grid buffers`};else if(H([1]))q=[1],n.fine={status:"skipped",progress:0,error:"coarse leg does not fit"};else return S(new Error(`neither model fits: adapter allows ${O} storage buffers of ${C(d.limits.maxStorageBufferBindingSize)}`));const ye=q.reduce((h,y)=>h+W[y].totalBytes,0);console.log(`[plan] legs ${q.map(h=>r[h].key).join("+")} \xB7 ${C(ye)} \xB7 buffers ${q.map(h=>W[h].buffers.length).join("+")} / ${O}`);const T=q[0];u(`Loading ${r[T].key} model (${C(W[T].totalBytes)})`),p(s.cache?"first visit downloads the weights; they are cached for next time":"");let j;try{j=await F.loadLeg(d.device,i,r[T].stem,{manifest:I[T],useCache:s.cache,signal:Q,onProgress:(h,y)=>{b(h),p(`${y} ${(h*100).toFixed(0)}%`)}})}catch(h){return w?void 0:S(h)}if(w){j.dispose?.();return}P.setLegs([j]),n.legs=[r[T].key],g.classList.add("geo-hidden"),z(),q.length===2&&(n.fine={status:"loading",progress:0,error:""},F.loadLeg(d.device,i,r[1].stem,{manifest:I[1],useCache:s.cache,signal:Q,onProgress:h=>{n.fine.progress=h}}).then(h=>{if(w){h.dispose?.();return}P.setLegs([j,h]),n.legs=["coarse","fine"],n.fine={status:"ready",progress:1,error:""},se.sync(),z()}).catch(h=>{w||(console.error(h),n.fine={status:"error",progress:0,error:h.message})}));let ae=null,oe=0,ne=-1e9,Y=!1;const we=()=>[...v.rotation,v.dist,v.fovRad,n.canvasW,n.canvasH,n.mode,n.sigmaScale,n.forceLeg,n.showRouting?1:0,P.version,$,x.scale,x.stillScale],_e=(h,y)=>h&&h.length===y.length&&h.every((B,U)=>B===y[U]),le=(h,y)=>{const[B,U]=h==="still"?x.stillSize(n.canvasW,n.canvasH):x.interactiveSize(n.canvasW,n.canvasH);R.submitted(y,B,U),P.render({rt:v.rtArray,dist:v.dist,fovRad:v.fovRad,W:B,H:U,mode:n.mode,sigmaScale:n.sigmaScale,forceLeg:n.forceLeg,showRouting:n.showRouting},(D,X)=>{R.gpuTime(performance.now(),D,X),x.onGpuTime(h,D)})};function ce(h){if(w)return;v.tick();const y=we();if(!_e(ae,y))P.inFlight<Ke&&h-ne>=1e3/Je[n.quiet?"quiet":"normal"]&&(ae=y,oe=h,ne=h,Y=!1,le("interactive",h));else if(!Y&&h-oe>=Xe&&P.inFlight===0){Y=!0;const[B,U]=x.interactiveSize(n.canvasW,n.canvasH),[D,X]=x.stillSize(n.canvasW,n.canvasH);(D!==B||X!==U)&&le("still",h)}R.update(h),M.update(n,R,P),se.sync(),L=requestAnimationFrame(ce)}L=requestAnimationFrame(ce)}return _.ready=E(),_}export{xe as mountNeuralEarth};
