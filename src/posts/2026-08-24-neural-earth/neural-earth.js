class ae{static async create(e,t={}){if(!navigator.gpu)throw new Error("WebGPU is not available in this browser. Use Chrome / Edge 113+ (or Safari 26+, Firefox 141+) on a real GPU.");const s=t.forceFallbackAdapter?{forceFallbackAdapter:!0}:{powerPreference:t.powerPreference||"high-performance"},i=await navigator.gpu.requestAdapter(s);if(!i)throw new Error(`No WebGPU adapter for ${JSON.stringify(s)}.`);const a=i.limits,r=[];i.features.has("timestamp-query")&&r.push("timestamp-query");const n=await i.requestDevice({requiredFeatures:r,requiredLimits:{maxStorageBufferBindingSize:a.maxStorageBufferBindingSize,maxBufferSize:a.maxBufferSize,maxStorageBuffersPerShaderStage:a.maxStorageBuffersPerShaderStage,maxUniformBufferBindingSize:a.maxUniformBufferBindingSize}}),o=e.getContext("webgpu"),c=navigator.gpu.getPreferredCanvasFormat();o.configure({device:n,format:c,alphaMode:"opaque"});let Q={};try{Q=i.info||(i.requestAdapterInfo?await i.requestAdapterInfo():{})}catch{}const l=Q.description||[Q.vendor,Q.architecture].filter(Boolean).join(" ")||"unknown GPU";return console.log("[WebGPU] adapter:",l,Q,"limits:",{bindingSize:a.maxStorageBufferBindingSize,bufferSize:a.maxBufferSize,storageBuffers:a.maxStorageBuffersPerShaderStage,timestamps:r.includes("timestamp-query")}),new ae(n,o,c,i,Q,l,!!t.forceFallbackAdapter)}constructor(e,t,s,i,a,r,n){this.device=e,this.ctx=t,this.format=s,this.adapter=i,this.info=a,this.gpuName=r,this.isFallback=n,this.hasTimestamps=e.features.has("timestamp-query"),this.onLost=null,e.lost.then(o=>{o.reason!=="destroyed"&&(console.error("WebGPU device lost:",o.message),this.onLost?.(o))})}get limits(){return this.device.limits}get maxGridBuffers(){return this.device.limits.maxStorageBuffersPerShaderStage-1}}function we(A,e){if(A.bits)return A.bits;const t=e?.grid?.dtype??"float16";return t==="int8"?8:t==="int4"?4:16}function de(A,e){return(A.stride_vertices??A.size_vertices)*2*we(A,e)/32}function Je(A,e,t=null){const s=[],i=[];let a=null,r=0;for(const n of A){const o=de(n,t);if(o>e)throw new Error(`level of ${o} words exceeds buffer cap ${e}`);(!a||a.wordCount+o>e)&&(a={startWord:r,wordCount:0},s.push(a)),i.push({bufIdx:s.length-1,localBase:r-a.startWord}),a.wordCount+=o,r+=o}return{buffers:s,placement:i}}const fe="neural-earth-weights-v1",te=8*1024*1024;class _{static async fetchManifest(e,t,s){const i=await fetch(`${e}/${t}.manifest.json`,{signal:s});if(!i.ok)throw new Error(`fetch ${t}.manifest.json: ${i.status}`);return i.json()}static plan(e,t){const s=e.grid?.dtype??"float16";if(!["float16","int8","int4"].includes(s))throw new Error(`unknown grid dtype ${s}`);const i=e.grid?.block_vertices??64;let a=Math.floor(Math.min(t.limits.maxStorageBufferBindingSize,t.limits.maxBufferSize)/4);a-=a%i;const{buffers:r,placement:n}=Je(e.levels,a,e),o=e.levels.reduce((c,Q)=>c+de(Q,e),0)*4;return{buffers:r,placement:n,totalBytes:o,quantized:s!=="float16"}}static async loadLeg(e,t,s,i={}){const{onProgress:a=()=>{},signal:r,useCache:n=!0}=i,o=i.manifest||await _.fetchManifest(t,s,r),{buffers:c,placement:Q,totalBytes:l,quantized:E}=_.plan(o,e),f=o.mlp?.file||`${s}.mlp.f32.bin`,d=o.grid?.file||`${s}.grid.f16.bin`;a(0,"mlp");const M=await fetch(`${t}/${f}`,{signal:r});if(!M.ok)throw new Error(`fetch ${f}: ${M.status}`);const F=new Float32Array(await M.arrayBuffer());let G=null;if(E){const h=await fetch(`${t}/${o.grid.scales_file}`,{signal:r});if(!h.ok)throw new Error(`fetch ${o.grid.scales_file}: ${h.status}`);G=new Uint32Array(await h.arrayBuffer())}let m=0;const x=c.map((h,p)=>{const v=h.wordCount*4,g={gpu:e.createBuffer({label:`${s} grid ${p}`,size:v,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST}),startByte:m,endByte:m+v};return m+=v,g});if(m!==l)throw new Error("grid plan does not tile the blob");const b=new Uint8Array(te);let I=0,k=0,w=0;const D=()=>{let h=0;for(;h<I;){const p=k+h;for(;x[w].endByte<=p;)w++;const v=x[w],g=Math.min(I-h,v.endByte-p);e.queue.writeBuffer(v.gpu,p-v.startByte,b,h,g),h+=g}k+=I,I=0};try{if(await Te(`${t}/${d}`,{signal:r,useCache:n,expectBytes:l,onChunk:h=>{let p=0;for(;p<h.length;){const v=Math.min(h.length-p,te-I);b.set(h.subarray(p,p+v),I),I+=v,p+=v,I===te&&D()}},onProgress:h=>a(h,"grid")}),I%4!==0)throw new Error("grid blob length is not a multiple of 4");if(D(),k!==l)throw new Error(`grid blob truncated: got ${k} of ${l} bytes`)}catch(h){throw x.forEach(p=>p.gpu.destroy()),h}return a(1,"done"),{stem:s,manifest:o,mlp:F,quantized:E,scales:G,gridBuffers:x.map(h=>h.gpu),placement:Q,nGridBuffers:c.length,bytes:l+F.byteLength+(G?G.byteLength:0),dispose(){this.gridBuffers.forEach(h=>h.destroy()),this.gridBuffers=[]}}}}async function Te(A,{signal:e,useCache:t,expectBytes:s,onChunk:i,onProgress:a}){let r=null,n=null;if(t&&typeof caches<"u")try{n=await caches.open(fe),r=await n.match(A),r&&console.log("[weights] cache hit",A)}catch{n=null}if(!r){if(r=await fetch(A,{signal:e}),!r.ok)throw new Error(`fetch ${A}: ${r.status}`);n&&await Ne(s)&&n.put(A,r.clone()).catch(l=>console.warn("[weights] cache put failed",l))}const o=Number(r.headers.get("content-length"))||s||0;if(!r.body){const l=new Uint8Array(await r.arrayBuffer());i(l),a(1);return}const c=r.body.getReader();let Q=0;for(;;){const{done:l,value:E}=await c.read();if(l)break;i(E),Q+=E.byteLength,o&&a(Math.min(1,Q/o))}}async function Ne(A){try{const{quota:e=0,usage:t=0}=await navigator.storage.estimate();return e-t>A*1.2}catch{return!1}}async function ze(){try{return await caches.delete(fe)}catch{return!1}}const Ce=720,pe=1440,me="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BwThAQ8GDwsBBtABAgWyBAYJ5AIHAgEB4wEOBBMIBQEBAdEBAgaxBAcI4gIJAgIB4wEKAxgEAQThAa8EBgjgAgoD6AEUAQ8FAQHJAQMarAQHBy0BsgIJAoYCCQTNAQcUrAQHBjABrwIEAQEBjQIIBM4BBgQBAgELrAQGBzACsgICAQMBAgGHAggEzgEMAQgDrAQGBy8BsgKVAgcEzgEWBqgEBgU1Aa0CvQEDVwYE0gEQAQEHAgKiBAUGNARXAdEBAQG9AQQBAli7AQUSBgEODKQEAwY1Bl8BAwHDAb4BCFu2AQoLGgulBAIGNQdVARcCAQGpAQELAQK4AQxbtQECAQkLGQulBAEGNQhtAbYBAgG5AQxbuAENChcLowQCBjQJPAE+AZoBAgkBAbwBDFy4AQIBCwgYCwMCngQCBjQJlQICCr0BAgMIXLkBDgUBARIBBA0BAZ8EAwY0CTkCSAGQAQMLuwEBBAQBA126AQ0HEhIBAp8EAgU0DYEBBYsBAgcBAb4BDUsBELkBDgcBBA4WnQQDAgEDMxCDAQOGAQIBAQcFAboBDUsDCAEEugEODQsBARwDAoMEAwEBCAgCNBCEAQEBAYQBBgTCAQ1PAQe9AQEBDQ0LIwECEgEBAugDAgICAQEGPw+JAQIBAYABAgq/AQ1QAQIBAr0BEQUBBQEBCyMJAwgG5gMBAQYFCgIzEZYCwAEMVsUBDQgMJAgCAQgCCgoE1ANME4sBAwEBBAF/wQEKWcMBDQgKJwUcBgUBAc4DAgJMGRcBbgQEAXUBAgECAQHDAQsBAVACAsYBCwIQJgUBAQEBAgIBASDNAwgDRhoRAXoDCQJqAQECAgEBwgERSAEDzAEEAQQEDikDAwIEAgoCGMoDBwFHGREBfQNwAQEEAQIBAgG1AR5G0wEBBgEDDSkDAQQCBQICH8wDTRmQAQcDAWgGAQYBsQEiQtwBAgIMJwcFBwEFG88DSxmSAQMDBAEBDAFWAQIEAQMBsQEbAQs/AQHcAQEBDSkIAQEDAQECAQYb0QNKGZYBCGEBAQMBAgGxASAECD0BAd4BCiwJCQkCAQEBFNMDSBoCApIBB14BAgEEAQKxASEBAQEKPN0BAgELKwwHDRTUA0celQEGAQFZAwEEAQIBrwEwOgEBAQHbAQMBCywHCw0BARHVA0cdmAEHDAFJAwIFAa4BMzvgAQ4qCAkQE9UDRxqbAQgKAgEBRQUFowEDBjc82AEBBQEBDiwGBxEHAQwHBMsDRxidAQYHCUQDAQMBpAE/PNkBAQICAg8vAwcSBgILBgbMA0YXnwEHBgIBBEECAQIBAwGlAT872gECAgECBQEBAQcxAQYTBgINAgfPA0QZnAELBQIBAkAGAQMBpAE/PNwBAQQHPxMJAhTRA0QSAgKeAQ0EBD8FAgIBpQE9PN0BAQEBBQcoAQEFDhQf0QNHDqIBDgUBAQE8BAEDAacBOykEEuABAQQJJAsMFBQFAwIB0gNIDZABAQIBBgICAQMOBQE7BQEDAacBOykHDu4BAiIOCRQCARrYA0cNjwEDAwIBAwEVPQUBAgGlAUAkAQMKC+8BASMQBxkBAhECAtoDJQMfDAQEhQEBAR0/rwFCIwEBDQnzAQEgAQERBBwV2wMkCAICFg0FAoUBIAIBBgYrEAGiAUQgEQeWAhMBHhjeAwIDBAECBQICCw4UDIwBIQkFKrMBRCEQBpUCFQEhJOQDBRAbCncEBgUEJQQJBwEYuwFFIBEETwPEAQEBEwEhEAIWBAHwAxwJAgECAQQBagYFCAEmBQ4CAQzGAUUgEgNNBQEBqAEBGAECAQEyBAEGAQMBAQcV8gMdDgQCDQJDARkBBDMEDAIBAQEBAgMBAckBRCAFAg0BTQIBAQEDqAEBGAUBLwYCBgEBDwr6AxwQAQQKBGAGAS4DDAMBBs0BQyAUAUkBAQEBCqYBAhcyEBwB+wMdI2AzBAoBBAHSAUAgAgFVAgMBAQ0BAb8BAQEGASgQmwQRAwshWjoECAHXAUEbWgECEAEDvgEBAQEDAgMnEZsEDwYJI1ZAAeABQBccAT8BARi+AQkBJhKdBA0ICCNUogI3AQgZGAM+AgEZvgEvAQEQngQNCgEBAidToQIBATMDBhUBAhsBPgEBG40BATAFASoRpgQCAwEKBCtOogI0BAQOAQYWB0MbiwECMgYBKQ+4BAQtTaICAgEgBQsCBggIAhILBgE5HgEBwAEBAiwPuAQFMAkCP6QCAQEfBx8BDQMDCgYBOx7JASoNuwQDOz+mAh8BAQQrAQINBgIBATkjwwEEASkM9wQrAhQDA6ECFQEdARoNBQg3IwEDxgEqC/gEIwICBRyfAhQFGAQWDgUJAQE1KMQBAgEDAScM8wQlAQMCJZMCAQICAwwCBQYIBgwCFQEBCgIOAQEzKgEBUAtuAwElCvEEUowCAgEBCwwNBgYhHjUCASlIGWksCDoCtARRnAIBBggMAQYiHwIBMwMBKEsYaCwLMgeyBB8BAQIupgIGEg8HCiI0AgEDAShKAQEbZSwKLQgBAa8EIQcZEAalAgURDxABAQEhNQIBAQIoSCBmAQEnDSoHCgICAZ8EAQEBAhwLGRUBpwIEDRQvOAIBKkQpYioOKQYGCqMEAQkBARAOFQEBvgIDCxcsCgEBAisGASgBAUgBASBnJg8mAwEEAguyBA4PFbgCBAMEBQECHAMGAQEeCQUsMkgDAR1oJQ8kFrQEAxcCBQEBCroCAwgDBAEBLBkIAQEFKTYBAwMBAwE0CQUYaAEBKQkpE7IEBCEOtQIBDAMBAgEDAiYXCg4kAQEBATkBAgEBAQExAQEJBAIBBQENblgXrwQBAQMjDQICBwivAgYBAgEkHAcOIz4CBgEBMAECBAsBAgEDAQEGcFYNBwnABAEBAQ0UBwcBAQMCpgIGBgEBJhgBAgERAQMfBAE5AgoBAS8DFwMDAW9SCBMHBQRnCsIDBAsmAgLUAQFdAQYlLQECH02+AU8GGQ9rB8EDAwkkAgLXAQFaAQQBByQmAQYiAQJKvwFLAiENBQZhCL4DLd8BA2clHgEFJQsBRr4BRwEBASMMBQlhCL0DLN4BAi8DHwIBARUQCAwhKQYBAgFIvwFqCwULYwYOAgUBpQMs3QEBAgIpAQQDHgYUEQkKHAEEKgoCSMABZwwFDGMHCwOnAyzGARMEAwIBFRIDAgQDHQUUEwkBBQMZAQEBAy4BAQYBS7UBAQgBAQMBYA8KCmAGCgGmAyzLARUBBQEEAQEJBAEUCQYZBQIDDxQIAgEFGQEBAgExBQFPAQetAQEIAgFcFAkPDwFLBwoBowMs0wEMAQgHAQECAxoBAQULEAEJCA0eAQQZAgE2BwFLAQkBBK0BBQJYFwkUBAEGBBECIgEUAwwDngMxzwEEDQgJAQEOARAICQwCCAkSCgEMAwECARUEATkBAQIEYKcBAgFZFgIjBQ8HAh8GEQUKA5sDNQEEdAFMAQEBAwMBAhUIAwQBAQcIAQQBAQwBAQMCAgwCBw0OCAIKBAMcQ2WiAQIBAgRQLQYOAhIEBhgJDwoGBO8CBCZCZgdHExoHEAUCAQ4IDQEBAwMPDwkCDBoBBERnogECBAECSTEIKhQKEQUCAgIG8QIII0VgDgYBCwEUAgMEDg8mAg0BAw4IAQMHDAEBEw0BAgUBBgEEAgMkMwYHaaYBAgEKATUCATkFKxEJAQMSBt0CARsKIElZBgESAQUJARAEBQQKEDUBAxEDAQEHAgEKAgIWDQgDBAIDIgEDNAwCb6gBAQEFASsBAgEDbQ0NAQEVBCoCswIECA0CAgMEG1BTKwEBBAYHBwIdQgsBBhAnAwIBBQEHKjEKBgECbKkBAwEHASAIAWsBAQQZGAQnAbACIAYFCgECYEIIATQBAgEGBAcBAQIdQBIRAQEWHgQZAQ8zBwJgAgECDQcBpAECAQEBAwIFAhUDBHIEHhUFJAKzAp0BMVEDPicXEhAEAhwDKDcBBAoDYQMBvwEBAQgCFm8CIBYGJAKxAqABHosBAQEDAgUGJwsCEQ8NJwEfAQICBjgJBVsBCgYFAQOvAQQDAQEBAQIDAgMEAQdjAQEBBgEnFgYiAQIBsAIKAUgBVgcCBAIEAQGJATkeDg5HAQQCAUYDAV0BCQUKMwGGAQQDAQIFAQFjDScYBSIBAQK8AQEBAlMHFAYBSAQBClIGgQECEzYgDxQHATkBAUQCBQEBAQFdAQgFCzYBhgEBbAIDCSoXBQ8DEAS6AQhQbA4IAdABBhcBAS4UBAgCAQoMPQEEAQhBAwUBAmsBCwECrAILLhIECQgXArgBCggCDAIzcwLgAQ4IAQI2EwUCAQcIDxEEJwIDAQEBAUQGAwIFYAMPAQEBAacCETIQBQgDHAK1AQ0EGirTAhIGKwMLEw4FAxIPAycBBUoBAQcBeqkCEDQQBgUDBgMLB7gBDwEcJgEC0QIWBikDCxIQBgQRDQQEAQ4BBgEMTwECBwJ2BAGoAgECDDUQCQIICwm8ASon0wIZAxsBAQEJBAoOFQUHDg8EEwIBAwEBDk97BQKtAgw1EAkCBgQMwgEnHuECFwQPAQwFBAQLCxUBAgUGEAwEFQkHV3EBBgEHrgICAQk1DwkCFcMBLw3qAhwCDgIGBgMHCgwEBA0GEAoKBAQBEAIGBAZVdAIBAwEBAQEFqwICAQs0AQUCDQEQAwVkAV4tAfkCHwQIFwgCBBEJBQEEEQkMAwIBEQIOVHABAQIDAQoGAaoCDTYFJWYBAgEFASoTFaoDIyUIDRAHEgkLBg0BEFQBA28BBgEFAwavAgIBCTQFEAEUaQoQAhMUDAEFLwIDCvICIiMJDhAGEwsMBggDC1QCAn4CAwEKqgIBAgtjYAMECgwGCBoBAQUCATsK8gIgQAECBQECCRSGAQICeAEVqgIJAQR0AQFWBAY0AkMD9QIWAQdbAQEEjgEBA4gBAwOvAgECC2ABEFwKA1sBEQP6Ag+CAo8BAQSZAQGQAQ1uBAJaCAaRBA02AQYBEQEBBgMUAgtcAQKSAQEEqQIPAQFoBAFhngQPNgoGMlqRAQEDsAIRZAECAgJkZwYCAwYDFAv0AgIBBQEDAQEGFQQFDQwEDAYwWJQBAQEGBAECpAIDARBVARMCA2RgHgcPAgHuAg4CHwUCDQwGCgU0U5YBAQYFAa8CEVABGgEBYlcCBR4GAgIF8AIFBAYCDwsNCwgECQIBAQEFAQsLAgENFFiRAQEJtgIWbVgBAQEBWAIEH/ICAQUBAQEBAQQEBQIFCwgIAQMBBQcCAQIBAgEQCwEJBAEFKANNoAG7AhpCAS4LAUVeCQUG+gIIAQUBAQoQDQQPBwQFAQsKAQgIWQYFAxClAQQBugIBBBhuAQEDCQQCNAECZAIKASsC0gIQAgQFBw8IEAQFAwQKCRUIAQEBAgECBhAGJsEBCgHOAg96NgMB9QMMAQgaAxECCgMBAgECDAwFBQICARUBAgEDAREotgEBAQICBwKMAQMhAaEBDGgCAwEUFgUQAQH8AxEIAkQMESoBBBvJAQQGhwEGxwEIbQETGZUEDQkBJQI/JR/UAYUBCp0CAR0BFAyrBAUMBBgDNgYICwICAggCCxcFDQMDtwEBAYkBDhUCvAIQswQMEQIkCwoDCAkEBgEFARAfBQEFA7YBAQEDAwMBgwEODAuPAgEtAQEEsQQFAgohBAQGCAEBAiIDBgkBEhoBAcYBAQQBAoYBDAEFDQilBwEIBwkIHgMCAwIjAQEMzgEBAQcBhwEQCAmyAgUWBawECAEHEhQHDQgMAQQECgEmCs4BCAJ5AgMGCwgHBrgCE78EBhcRBgcMFgQSAR4NAQLGAQsBdAIDCgEBAgQBDcICE9gEAQIMHBcBBAYDDBUBBBrEAQcBcQEDJhwBAwKQAgkFBgEH1wQMJR8FAQEfAgUdBAG6AQcBcwECDgEBAQ6sAhQGBgICgwUEAgIBAQIeAh0BBAMBJ7cBegsBBgILBAEDBgILlAIVCgKLBR0CAgUNAQEEHCG2AXoHBgQCCAkBARWRAhiCBQMJJgIRAh4BDReyAQUHhgEDAQIDI4ECCgMNAgeBBQEBAQkCAwIBGAgOAyoBAQMBDgsCsAEBA5MBBAIKAgIDDQMBDgY7AwUCFwQCAgQBhAEJCAEBA5kFAQEaAQUVLwEDCc4BjwECBwNjBAEDDgEGAQICBAcHAnwUnQUCAhAMAQIMAwcFLQgBArkBAwUGA/MBCAUBAggRBgoKfBKeBQ8ICwIaAgIIJAqnAQEjgAIBAQMTBQEEEwk3BjMPqAUFBg8DCQMOAQECLwsBAQUDmwEEIv4BAhgDAgEXBGMHDQW1BQEDMAIlAwkOEgGMAQMFAQcBFooCAQYCAwIBA8MGBQNhEgsCEQENAmAFDQUCBgIDEpsCARAEtwYUAVgKBgMUAgcBBAERBDEGGQUJBwKQCRQDTA8SAgYFDgcCATkwAY8JCAZWKQgICgMPAUawCTYBFT4HBAwBAwRKtAkIAi4FAkECBAEBHgENAScDA8wJFWkCAQYBBAEBAQIBAwMBAhMCCAMBAQMMBOUKAgIDAQQBKfmqAg==";function Oe(){const A=typeof atob=="function"?Uint8Array.from(atob(me),r=>r.charCodeAt(0)):new Uint8Array(Buffer.from(me,"base64")),e=Ce*pe,t=new Uint32Array(Math.ceil(e/32));let s=0,i=0,a=0;for(;s<A.length;){let r=0,n=0,o;do o=A[s++],r|=(o&127)<<n,n+=7;while(o&128);if(a)for(let c=i;c<i+r;c++)t[c>>>5]|=1<<(c&31);i+=r,a^=1}if(i!==e)throw new Error(`land mask: decoded ${i} bits, expected ${e}`);return t}const Ae={NLEV:16,MAX_LEGS:2},Z=64*32+64*64+7*64,Ie=8,De=4+16*3,se=(Ie+Ae.MAX_LEGS*De)*16,ie=3;function qe(A){const e=Array.from({length:A},(r,n)=>`@group(0) @binding(${ie+n}) var<storage, read> g${n} : array<u32>;`).join(`
`),t=Array.from({length:A},(r,n)=>`    case ${n}u: { return g${n}[idx]; }`).join(`
`),s=ie+A,i=s+1,a=s+2;return`
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

@group(0) @binding(0) var<uniform> P : Params;
@group(0) @binding(1) var<storage, read> mlp : array<f32>;
@group(0) @binding(2) var<storage, read> scales : array<u32>; 
${e}
@group(0) @binding(${s}) var outTex : texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(${i}) var<storage, read> landMask : array<u32>; 
@group(0) @binding(${a}) var<storage, read> fineBoxes : array<vec4<f32>>; 

const MASK_N_LAT : u32 = ${Ce}u;
const MASK_N_LON : u32 = ${pe}u;

var<workgroup> wgFlag  : array<u32, 64>;
var<workgroup> wgLand  : array<u32, 64>;
var<workgroup> wgOther : array<u32, 64>;

const PRIME1 : u32 = 2654435761u;
const PRIME2 : u32 = 805459861u;
const TWO_PI : f32 = 6.283185307179586;

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


fn rayDir(pxf : vec2<f32>) -> vec3<f32> {
  let aspect = P.cam.z / P.cam.w;
  let halfH = tan(P.cam.y * 0.5);
  let halfW = halfH * aspect;
  let gx = -halfW + (pxf.x / max(P.cam.z - 1.0, 1.0)) * 2.0 * halfW;
  let gy = halfH - (pxf.y / max(P.cam.w - 1.0, 1.0)) * 2.0 * halfH;
  return normalize(vec3<f32>(gx, gy, -1.0));
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

fn rt() -> mat3x3<f32> {
  
  return mat3x3<f32>(
    vec3<f32>(P.rt0.x, P.rt1.x, P.rt2.x),
    vec3<f32>(P.rt0.y, P.rt1.y, P.rt2.y),
    vec3<f32>(P.rt0.z, P.rt1.z, P.rt2.z),
  );
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

  let camDist = P.cam.x;
  let fov = P.cam.y;
  let dir = rayDir(vec2<f32>(px));

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
`}const We=`
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
`,$e={rgb:0,dem:1,tree:2},je={16:0,8:1,4:2},Ze=-500,Ve=8848,Me=3e-5;function Xe(A){return(Ie+A*De)*4}function et(A,e,t,s,i){const a=s.manifest,r=Xe(t),n=a.model_kind==="decomp_unigrid"?1:0,o=a.decomp||{};A[r+0]=a.dem.dem_scale,A[r+1]=a.dem.dem_grad_scale,A[r+2]=a.gating.std_scale,A[r+3]=0,A[r+4]=n,A[r+5]=o.h_bound||0,A[r+6]=o.offset_mean||0,A[r+7]=o.offset_std||1,A[r+8]=o.log_scale_mean||0,A[r+9]=o.log_scale_std||1,A[r+10]=0,A[r+11]=0;const c=(a.grid?.dtype??"float16")!=="float16",Q=a.grid?.block_vertices??64;if(c&&Q!==64)throw new Error(`${a.grid.dtype} block must be 64 (kernel shifts by 6), got ${Q}`);e[r+12]=t*Z,e[r+13]=a.levels.length,e[r+14]=0,e[r+15]=0;const l=a.gating.gating_resolutions;if(a.levels.length!==Ae.NLEV)throw new Error(`expected ${Ae.NLEV} levels, got ${a.levels.length}`);a.levels.forEach((E,f)=>{const d=r+16+f*12;A[d+0]=E.scale,A[d+1]=l[f],A[d+2]=0,A[d+3]=0;const M=we(E,a),F=je[M];if(F===void 0)throw new Error(`level ${f}: unsupported bits ${M}`);e[d+4]=E.size_vertices,e[d+5]=E.resolution,e[d+6]=E.dense?1:0,e[d+7]=i+s.placement[f].bufIdx,e[d+8]=s.placement[f].localBase,e[d+9]=F?(s.scaleOffset??0)+E.start_vertex/Q:0,e[d+10]=F,e[d+11]=0})}function tt(A,e){const t=e.rt;A[0]=t[0],A[1]=t[1],A[2]=t[2],A[3]=0,A[4]=t[3],A[5]=t[4],A[6]=t[5],A[7]=0,A[8]=t[6],A[9]=t[7],A[10]=t[8],A[11]=0,A[12]=e.dist,A[13]=e.fovRad,A[14]=e.W,A[15]=e.H,A[16]=e.sigmaScale,A[17]=$e[e.mode]??0,A[18]=Ze,A[19]=Ve,A[20]=e.crossover,A[21]=e.nLegs,A[22]=e.forceLeg,A[23]=e.nFineBoxes||0,A[24]=e.stride||1,A[25]=e.phase||0,A[26]=e.showCoverage?1:0,A[27]=0;const s=e.pick;A[28]=s?s[0]:0,A[29]=s?s[1]:0,A[30]=s?s[2]:0,A[31]=s?1:0}const ve=[{name:"alps-north-italy",lat:[40.499,45.046],lon:[5.545,11.651]},{name:"us-southwest",lat:[31.552,41.514],lon:[-118.461,-106.845]}];function At(A){const e=new Float32Array(Math.max(1,A.length)*4);return A.forEach((t,s)=>{const[i,a]=t.lat,[r,n]=t.lon;if(!(i<=a&&r<=n))throw new Error(`fine coverage box ${t.name||s}: min must be <= max (split antimeridian boxes)`);e.set([i,a,r,n],s*4)}),e}const st=3,xe=4,V=256;class it{constructor(e,t={}){this.gpu=e,this.device=e.device,this.legs=[],this.crossover=Me,this.version=0,this.inFlight=0,this.onError=null,this.uniform=this.device.createBuffer({label:"params",size:se,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST}),this._ubuf=new ArrayBuffer(se),this._f32=new Float32Array(this._ubuf),this._u32=new Uint32Array(this._ubuf),this.sampler=this.device.createSampler({magFilter:"linear",minFilter:"linear"});const s=this.device.createShaderModule({code:We,label:"present"});this.sampleLayout=this.device.createBindGroupLayout({entries:[{binding:0,visibility:GPUShaderStage.FRAGMENT,texture:{}},{binding:1,visibility:GPUShaderStage.FRAGMENT,sampler:{}}]});const i=this.device.createPipelineLayout({bindGroupLayouts:[this.sampleLayout]}),a=(n,o)=>this.device.createRenderPipeline({label:o,layout:i,vertex:{module:s,entryPoint:"vs"},fragment:{module:s,entryPoint:"fs",targets:[{format:n}]},primitive:{topology:"triangle-list"}});this.presentPipeline=a(e.format,"present"),this.blitPipeline=a("rgba8unorm","blit"),this._lastTarget=null;const r=Oe();this.maskBuffer=this.device.createBuffer({label:"land mask",size:r.byteLength,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST}),this.device.queue.writeBuffer(this.maskBuffer,0,r),this.coverBuffer=this.device.createBuffer({label:"fine coverage",size:V*16,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST}),this.fineCoverage=[],this.setFineCoverage(t.fineCoverage??ve),this._targets=new Map,this.computePipeline=null,this.mlpBuffer=null,this.bytes={grids:0,mlp:0,targets:0,uniform:se,timing:0,mask:r.byteLength+V*16},this._initTiming()}_initTiming(){if(this.timing=null,!this.gpu.hasTimestamps)return;const e=this.device;this.timing={querySet:e.createQuerySet({type:"timestamp",count:2}),resolve:e.createBuffer({size:16,usage:GPUBufferUsage.QUERY_RESOLVE|GPUBufferUsage.COPY_SRC}),readbacks:Array.from({length:xe},()=>({buf:e.createBuffer({size:16,usage:GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST}),busy:!1}))},this.bytes.timing=16*(xe+1)}setFineCoverage(e){if(e.length>V)throw new Error(`at most ${V} coverage boxes`);this.fineCoverage=e,this.device.queue.writeBuffer(this.coverBuffer,0,At(e))}setLegs(e){if(e.length<1||e.length>2)throw new Error("1 or 2 legs supported");this.legs=e,this.crossover=e[e.length-1].manifest.routing?.sigma_crossover??Me,this.mlpBuffer?.destroy();const t=new Float32Array(Z*e.length);e.forEach((o,c)=>{if(o.mlp.length!==Z)throw new Error(`leg ${c}: mlp has ${o.mlp.length} floats`);t.set(o.mlp,c*Z)}),this.mlpBuffer=this.device.createBuffer({label:"mlp",size:t.byteLength,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST}),this.device.queue.writeBuffer(this.mlpBuffer,0,t),this.scalesBuffer?.destroy();let s=0;for(const o of e)o.scaleOffset=s,s+=o.scales?o.scales.length:0;const i=new Uint32Array(Math.max(1,s));for(const o of e)o.scales&&i.set(o.scales,o.scaleOffset);this.scalesBuffer=this.device.createBuffer({label:"int8 scales",size:i.byteLength,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST}),this.device.queue.writeBuffer(this.scalesBuffer,0,i);let a=0;e.forEach((o,c)=>{et(this._f32,this._u32,c,o,a),a+=o.nGridBuffers}),this.nGrid=a;const r=qe(this.nGrid),n=this.device.createShaderModule({code:r,label:`forward x${e.length}`});n.getCompilationInfo().then(o=>{const c=o.messages.filter(l=>l.type==="error");if(!c.length)return;const Q=c.map(l=>`L${l.lineNum}:${l.linePos} ${l.message}`).join(`
`);console.error(`[forward WGSL]
`+Q),this.onError?.(new Error(`shader compile failed:
${Q}`))}),this.computePipeline=this.device.createComputePipeline({layout:"auto",compute:{module:n,entryPoint:"main"}}),this._destroyTargets(),this.bytes.grids=e.reduce((o,c)=>o+c.gridBuffers.reduce((Q,l)=>Q+l.size,0),0),this.bytes.mlp=t.byteLength+i.byteLength,this.version++}get totalBytes(){const e=this.bytes;return e.grids+e.mlp+e.targets+e.uniform+e.timing+e.mask}_destroyTargets(){for(const e of this._targets.values())e.tex.destroy();this._targets.clear(),this._lastTarget=null,this.bytes.targets=0}_target(e,t){const s=`${e}x${t}`;let i=this._targets.get(s);if(i)return this._targets.delete(s),this._targets.set(s,i),i;if(this._targets.size>=st){const[c,Q]=this._targets.entries().next().value;Q.tex.destroy(),this._targets.delete(c),this._lastTarget===Q&&(this._lastTarget=null),this.bytes.targets-=Q.bytes}const a=this.device.createTexture({label:`target ${s}`,size:[e,t],format:"rgba8unorm",usage:GPUTextureUsage.STORAGE_BINDING|GPUTextureUsage.TEXTURE_BINDING|GPUTextureUsage.RENDER_ATTACHMENT|GPUTextureUsage.COPY_SRC}),r=a.createView(),n=[{binding:0,resource:{buffer:this.uniform}},{binding:1,resource:{buffer:this.mlpBuffer}},{binding:2,resource:{buffer:this.scalesBuffer}}];let o=ie;for(const c of this.legs)for(const Q of c.gridBuffers)n.push({binding:o++,resource:{buffer:Q}});return n.push({binding:o,resource:r}),n.push({binding:o+1,resource:{buffer:this.maskBuffer}}),n.push({binding:o+2,resource:{buffer:this.coverBuffer}}),i={tex:a,view:r,W:e,H:t,bytes:e*t*4,computeBind:this.device.createBindGroup({layout:this.computePipeline.getBindGroupLayout(0),entries:n}),presentBind:this.device.createBindGroup({layout:this.sampleLayout,entries:[{binding:0,resource:r},{binding:1,resource:this.sampler}]})},this._targets.set(s,i),this.bytes.targets+=i.bytes,i}render(e,t){if(!this.computePipeline)return;const s=e.stride>1?e.stride:1,i=this._target(e.W,e.H),a=s>1&&e.phase===0&&this._lastTarget&&this._lastTarget!==i?this._lastTarget:null;s===1&&(this._lastTarget=i),tt(this._f32,{...e,stride:s,crossover:this.crossover,nLegs:this.legs.length,nFineBoxes:this.fineCoverage.length}),this.device.queue.writeBuffer(this.uniform,0,this._ubuf);const r=this.timing?.readbacks.find(E=>!E.busy)||null,n=this.device.createCommandEncoder();if(a){const E=n.beginRenderPass({colorAttachments:[{view:i.view,loadOp:"clear",storeOp:"store",clearValue:{r:0,g:0,b:0,a:1}}]});E.setPipeline(this.blitPipeline),E.setBindGroup(0,a.presentBind),E.draw(3),E.end()}const o=n.beginComputePass(r?{timestampWrites:{querySet:this.timing.querySet,beginningOfPassWriteIndex:0,endOfPassWriteIndex:1}}:{});o.setPipeline(this.computePipeline),o.setBindGroup(0,i.computeBind),o.dispatchWorkgroups(Math.ceil(e.W/(8*s)),Math.ceil(e.H/(8*s))),o.end();const c=n.beginRenderPass({colorAttachments:[{view:this.gpu.ctx.getCurrentTexture().createView(),loadOp:"clear",storeOp:"store",clearValue:{r:0,g:0,b:0,a:1}}]});c.setPipeline(this.presentPipeline),c.setBindGroup(0,i.presentBind),c.draw(3),c.end(),r&&(r.busy=!0,n.resolveQuerySet(this.timing.querySet,0,2,this.timing.resolve,0),n.copyBufferToBuffer(this.timing.resolve,0,r.buf,0,16));const Q=performance.now();this.inFlight++,this.device.queue.submit([n.finish()]);const l=this.device.queue.onSubmittedWorkDone().then(()=>{this.inFlight--});r?r.buf.mapAsync(GPUMapMode.READ).then(()=>{const E=new BigUint64Array(r.buf.getMappedRange()),f=Number(E[1]-E[0]);r.buf.unmap(),r.busy=!1,f>0&&t?.(f/1e6,!1)}).catch(()=>{r.busy=!1}):l.then(()=>t?.(performance.now()-Q,!0))}dispose(){this._destroyTargets(),this.uniform.destroy(),this.maskBuffer.destroy(),this.coverBuffer.destroy(),this.mlpBuffer?.destroy(),this.scalesBuffer?.destroy();for(const e of this.legs)e.dispose?.();this.legs=[],this.timing&&(this.timing.resolve.destroy(),this.timing.readbacks.forEach(e=>e.buf.destroy()),this.timing.querySet.destroy())}}const U={sub:(A,e)=>[A[0]-e[0],A[1]-e[1],A[2]-e[2]],dot:(A,e)=>A[0]*e[0]+A[1]*e[1]+A[2]*e[2],cross:(A,e)=>[A[1]*e[2]-A[2]*e[1],A[2]*e[0]-A[0]*e[2],A[0]*e[1]-A[1]*e[0]],norm:A=>Math.hypot(A[0],A[1],A[2]),normalize:A=>{const e=Math.hypot(A[0],A[1],A[2])||1e-12;return[A[0]/e,A[1]/e,A[2]/e]}},S={identity:()=>[1,0,0,0,1,0,0,0,1],globeDefault:()=>[0,1,0,0,0,1,1,0,0],transpose:A=>[A[0],A[3],A[6],A[1],A[4],A[7],A[2],A[5],A[8]],mul:(A,e)=>{const t=new Array(9);for(let s=0;s<3;s++)for(let i=0;i<3;i++)t[s*3+i]=A[s*3]*e[i]+A[s*3+1]*e[3+i]+A[s*3+2]*e[6+i];return t},mulVec:(A,e)=>[A[0]*e[0]+A[1]*e[1]+A[2]*e[2],A[3]*e[0]+A[4]*e[1]+A[5]*e[2],A[6]*e[0]+A[7]*e[1]+A[8]*e[2]],axisAngle:(A,e)=>{const t=U.normalize(A),s=Math.sin(e),i=Math.cos(e),[a,r,n]=t,o=[0,-n,r,n,0,-a,-r,a,0],c=S.mul(o,o),Q=new Array(9);for(let l=0;l<9;l++)Q[l]=(l%4===0?1:0)+s*o[l]+(1-i)*c[l];return Q},between:(A,e)=>{const t=U.normalize(A),s=U.normalize(e),i=U.cross(t,s),a=U.norm(i),r=Math.max(-1,Math.min(1,U.dot(t,s)));if(a<1e-8)return S.identity();const n=[i[0]/a,i[1]/a,i[2]/a],[o,c,Q]=n,l=[0,-Q,c,Q,0,-o,-c,o,0],E=S.mul(l,l),f=new Array(9);for(let d=0;d<9;d++)f[d]=(d%4===0?1:0)+a*l[d]+(1-r)*E[d];return f}},rt=Math.PI*2;function re(A,e,t,s,i,a){const r=t/s,n=Math.tan(i/2),o=n*r,c=-o+A/Math.max(t-1,1)*2*o,Q=n-e/Math.max(s-1,1)*2*n;let l=c,E=Q,f=-1;const d=Math.hypot(l,E,f);l/=d,E/=d,f/=d;const M=2*f*a,F=a*a-1,G=M*M-4*F;if(G<0)return{hit:!1,dir:[l,E,f],hitPoint:[0,0,0],facing:0,sigma:0};const m=(-M-Math.sqrt(G))/2,x=[l*m,E*m,a+f*m],b=Math.max(0,Math.min(1,-(x[0]*l+x[1]*E+x[2]*f))),I=i/Math.max(s-1,1)/Math.max(b,.01)/rt,k=I*m;return{hit:!0,dir:[l,E,f],hitPoint:x,facing:b,sigma:I,sigmaSurface:k}}class at{constructor(){this.rotation=S.globeDefault(),this.dist=2.4,this.minAlt=.004,this.maxDist=10,this.fovDeg=45,this._dragStart=null,this._spinAxis=null,this._spinSpeed=0,this._friction=.93}get fovRad(){return this.fovDeg*Math.PI/180}get rtArray(){return S.transpose(this.rotation)}reset(){this.rotation=S.globeDefault(),this.dist=2.4,this._spinAxis=null,this._spinSpeed=0}zoom(e){const t=Math.pow(.92,e),s=Math.max(this.minAlt,Math.min(this.maxDist-1,(this.dist-1)*t));this.dist=1+s}beginDrag(e,t,s,i){const a=re(e,t,s,i,this.fovRad,this.dist);this._dragStart=a.hit?a.hitPoint:null,this._dragging=!0,this._spinSpeed=0,this._spinAxis=null}endDrag(){this._dragStart=null,this._dragging=!1}drag(e,t,s,i){if(!this._dragStart)return;const a=re(e,t,s,i,this.fovRad,this.dist);if(!a.hit)return;const r=this._dragStart,n=S.between(r,a.hitPoint);this.rotation=S.mul(n,this.rotation),this._dragStart=a.hitPoint;const o=U.cross(r,a.hitPoint),c=Math.acos(Math.max(-1,Math.min(1,U.dot(a.hitPoint,r)))),Q=U.norm(o);if(Q>1e-8&&c>1e-6){const l=[o[0]/Q,o[1]/Q,o[2]/Q],E=.6;this._spinAxis?(this._spinAxis=U.normalize([E*l[0]+(1-E)*this._spinAxis[0],E*l[1]+(1-E)*this._spinAxis[1],E*l[2]+(1-E)*this._spinAxis[2]]),this._spinSpeed=E*c+(1-E)*this._spinSpeed):(this._spinAxis=l,this._spinSpeed=c)}}tick(){this._dragging||this._spinAxis&&this._spinSpeed>1e-5&&(this.rotation=S.mul(S.axisAngle(this._spinAxis,this._spinSpeed),this.rotation),this._spinSpeed*=this._friction,this._spinSpeed<1e-5&&(this._spinSpeed=0,this._spinAxis=null))}}class ot{constructor(e={}){this.auto=e.auto??!0,this.scale=e.scale??.5,this.minScale=e.minScale??.2,this.targetMs=e.targetMs??22,this.maxScale=e.maxScale??1,this.refineMs=e.refineMs??12,this.strides=e.strides??[2,4,8],this._ema=0,this._fullMs=0,this.refineMsLast=0,this._samples=0}onGpuTime(e,t,s){if(e==="refine"){if(this.refineMsLast=t,s&&s.phase>0){const a=t*s.stride*s.stride;this._fullMs=this._fullMs?this._fullMs*.8+a*.2:a}return}if(this._ema=this._ema?this._ema*.7+t*.3:t,!this.auto||++this._samples<4)return;this._samples=0;const i=this._ema/this.targetMs;i>1.25?this.scale=Math.max(this.minScale,this.scale/Math.sqrt(Math.min(i,4))):i<.6&&this.scale<this.maxScale&&(this.scale=Math.min(this.maxScale,this.scale*1.15)),this.scale=Math.min(this.scale,this.maxScale),this.scale=Math.round(this.scale*100)/100}setScale(e){this.auto=!1,this.scale=Math.round(Math.max(this.minScale,Math.min(this.maxScale,e))*100)/100}setAuto(e){this.auto=e,this._ema=0,this._samples=0}refineStride(e,t,s,i){const a=this._fullMs||(this._ema||this.targetMs)*(s*i)/Math.max(1,e*t);for(const r of this.strides)if(a/(r*r)<=this.refineMs)return r;return this.strides[this.strides.length-1]}setQuiet(e){this.targetMs=e?10:22,this.refineMs=e?6:12,this.maxScale=e?.5:1,this.scale=Math.min(this.scale,this.maxScale),this._ema=0,this._samples=0}interactiveSize(e,t){return nt(e,t,this.scale)}}function nt(A,e,t){return[Math.max(8,Math.round(A*t)),Math.max(8,Math.round(e*t))]}const be={Digit1:"rgb",Digit2:"dem",Digit3:"tree"};class gt{constructor(e,t,s,i){this.canvas=e,this.camera=t,this.state=s,this.actions=i,this._pointers=new Map,this._pinchDist=0,this._bind()}_bind(){const e=this.canvas;e.addEventListener("pointerdown",s=>{e.setPointerCapture(s.pointerId),this._pointers.set(s.pointerId,s),this._down=this._pointers.size===1?[s.clientX,s.clientY]:null,this._pointers.size===1?(e.classList.add("dragging"),this.camera.beginDrag(...this._px(s))):this._pointers.size===2&&(this.camera.endDrag(),this._pinchDist=this._pinch())}),e.addEventListener("pointermove",s=>{if(this._pointers.has(s.pointerId)){if(this._pointers.set(s.pointerId,s),this._pointers.size===1)this.camera.drag(...this._px(s));else if(this._pointers.size===2){const i=this._pinch();this._pinchDist>0&&i>0&&this.camera.zoom(Math.log(this._pinchDist/i)/Math.log(.92)),this._pinchDist=i}}});const t=s=>{if(this._pointers.delete(s.pointerId),this._pointers.size===0)e.classList.remove("dragging"),this.camera.endDrag();else if(this._pointers.size===1){const[i]=this._pointers.values();this.camera.beginDrag(...this._px(i))}};e.addEventListener("pointerup",s=>{const i=this._down;i&&this.state.pick&&this._pointers.size===1&&Math.hypot(s.clientX-i[0],s.clientY-i[1])<5&&this.actions.pickPoint(...this._px(s)),this._down=null,t(s)}),e.addEventListener("pointercancel",s=>{this._down=null,t(s)}),e.addEventListener("wheel",s=>{s.preventDefault(),this.camera.zoom(Math.sign(s.deltaY)*(s.shiftKey?3:1))},{passive:!1}),e.addEventListener("keydown",s=>{if(be[s.code]){this.actions.setMode(be[s.code]);return}const i=this.actions;switch(s.code){case"BracketLeft":i.scaleSigma(1/1.5);break;case"BracketRight":i.scaleSigma(1.5);break;case"KeyR":i.reset();break;case"KeyF":i.fullscreen();break;case"KeyQ":i.toggleQuiet();break;case"KeyS":i.toggleStats();break;case"KeyC":i.togglePick();break;case"KeyL":i.cycleLeg();break;case"KeyG":i.cycleGpu();break;default:return}s.preventDefault()})}_pinch(){const[e,t]=this._pointers.values();return Math.hypot(e.clientX-t.clientX,e.clientY-t.clientY)}_px(e){const t=this.canvas.getBoundingClientRect(),s=this.state.canvasW/t.width,i=this.state.canvasH/t.height;return[(e.clientX-t.left)*s,(e.clientY-t.top)*i,this.state.canvasW,this.state.canvasH]}}const Bt=[["rgb","RGB"],["dem","DEM"],["tree","Tree"]];class ct{constructor(e,t,s){this.el=e,this.state=t,this.actions=s,this._buttons=[],this._build()}_group(e,t,s,i){const a=document.createElement("div");if(a.className="tb-group",e){const r=document.createElement("span");r.className="tb-title",r.textContent=e,a.appendChild(r)}for(const[r,n,o]of t){const c=document.createElement("button");c.type="button",c.textContent=n,o&&(c.title=o),c.addEventListener("click",()=>i(r)),this._buttons.push({b:c,isActive:()=>s(r)}),a.appendChild(c)}return this.el.appendChild(a),a}_build(){const e=this.state,t=this.actions;this._group("layer",Bt.map(([s,i],a)=>[s,i,`key ${a+1}`]),s=>e.mode===s,s=>t.setMode(s)),this._group("",[["quiet","Quiet","low GPU power while dragging: 24 renders/s, \u2264\xBD res, 10 ms budget (Q)"],["stats","Stats","toggle the stats overlay (S)"]],s=>({quiet:e.quiet,stats:e.showStats})[s],s=>({quiet:t.toggleQuiet,stats:t.toggleStats})[s]()),this._group("",[["reset","Reset","R"],["fs","Fullscreen","F"],["cache","Clear cache","forget downloaded weights"]],()=>!1,s=>({reset:t.reset,fs:t.fullscreen,cache:t.clearCache})[s]())}sync(){for(const{b:e,isActive:t}of this._buttons)e.classList.toggle("active",!!t())}}const lt={rgb:"RGB",dem:"DEM hillshade",tree:"Tree cover"};function Qt(A,e){const t=A.fovRad/Math.max(e-1,1)*(A.dist-1)/(2*Math.PI);return{sigma:t,z:Math.log2(1/(256*t))}}const Et={"-1":"auto routing",0:"coarse only",1:"fine only"};class ht{constructor(){this._renders=[],this._gpu=[],this.lastGpuMs=0,this.gpuApprox=!1,this.fps=0,this.busy=0,this.evalsPerSec=0}submitted(e,t,s=1){this._renders.push({t:e,px:t*s})}gpuTime(e,t,s){this._gpu.push({t:e,ms:t}),this.lastGpuMs=t,this.gpuApprox=s}update(e){const t=e-1e3;for(;this._renders.length&&this._renders[0].t<t;)this._renders.shift();for(;this._gpu.length&&this._gpu[0].t<t;)this._gpu.shift();this.fps=this._renders.length,this.evalsPerSec=this._renders.reduce((s,i)=>s+i.px,0),this.busy=Math.min(1,this._gpu.reduce((s,i)=>s+i.ms,0)/1e3)}}function R(A){return A>=1e9?`${(A/1e9).toFixed(2)} GB`:A>=1e6?`${(A/1e6).toFixed(0)} MB`:A>=1e3?`${(A/1e3).toFixed(0)} kB`:`${A} B`}class ut{constructor(e){this.el=e}update(e,t,s){if(!e.showStats){this.el.textContent="";return}const{camera:i,mode:a,sigmaScale:r,forceLeg:n,quality:o,legs:c,fine:Q,gpuName:l,gpuPref:E,canvasW:f,canvasH:d,refine:M}=e,[F,G]=o.interactiveSize(f,d),m=M?`refine ${Math.min(M.k,M.n)}/${M.n} (stride ${M.stride}, ${o.refineMsLast.toFixed(1)} ms/pass)`:"refine idle",x=s.fineCoverage.length,b=(c.length===2?`coarse + fine \xB7 ${Et[n]}${x?` in ${x} area${x>1?"s":""}`:" everywhere"}`:`${c[0]||"no"} leg`)+` \xB7 ${e.quant}`,I=performance.memory?` \xB7 JS heap ${R(performance.memory.usedJSHeapSize)}`:"",k=s.bytes,{sigma:w,z:D}=Qt(i,d),h=[`${b}  \xB7  ${lt[a]}  \xB7  \u03C3\xD7${r.toFixed(2)}  \xB7  alt ${(i.dist-1).toFixed(3)}  \xB7  centre \u03C3 ${w.toExponential(1)} \u2248 z${D.toFixed(1)}`,`canvas ${f}\xD7${d}  \xB7  interactive ${F}\xD7${G} (${o.auto?"auto ":""}${o.scale.toFixed(2)})  \xB7  ${m}${e.quiet?"  \xB7  QUIET":""}`,`${t.fps} renders/s  \xB7  GPU ${t.gpuApprox?"\u2248":""}${t.lastGpuMs.toFixed(1)} ms/frame  \xB7  busy ${(t.busy*100).toFixed(0)}%  \xB7  ${(t.evalsPerSec/1e6).toFixed(1)} M evals/s`,`GPU memory ${R(s.totalBytes)}  (grids ${R(k.grids)} \xB7 targets ${R(k.targets)} \xB7 mlp ${R(k.mlp)})${I}`];Q.status==="loading"?h.push(`fine leg: downloading ${(Q.progress*100).toFixed(0)}%`):Q.status==="error"?h.push(`fine leg: failed \u2014 ${Q.error}`):Q.status==="skipped"&&h.push(`fine leg: skipped \u2014 ${Q.error}`),h.push(`${l}  [${E}]`),this.el.textContent=h.join(`
`)}}const wt="attribute vec2 p; void main() { gl_Position = vec4(p, 0.0, 1.0); }",dt=`
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
}`;class ft{constructor(e){this.canvas=e,this.target=0,this.shown=0,this._raf=0,this._t0=performance.now(),this._running=!1;const t=e.getContext("webgl",{alpha:!0,premultipliedAlpha:!0,antialias:!1,depth:!1,stencil:!1,powerPreference:"low-power",preserveDrawingBuffer:!1});if(this.gl=t,!t){e.classList.add("geo-hidden");return}const s=(i,a)=>{const r=t.createShader(i);if(t.shaderSource(r,a),t.compileShader(r),!t.getShaderParameter(r,t.COMPILE_STATUS))throw new Error(t.getShaderInfoLog(r));return r};try{const i=t.createProgram();if(t.attachShader(i,s(t.VERTEX_SHADER,wt)),t.attachShader(i,s(t.FRAGMENT_SHADER,dt)),t.linkProgram(i),!t.getProgramParameter(i,t.LINK_STATUS))throw new Error(t.getProgramInfoLog(i));t.useProgram(i);const a=t.createBuffer();t.bindBuffer(t.ARRAY_BUFFER,a),t.bufferData(t.ARRAY_BUFFER,new Float32Array([-1,-3,-1,1,3,1]),t.STATIC_DRAW);const r=t.getAttribLocation(i,"p");t.enableVertexAttribArray(r),t.vertexAttribPointer(r,2,t.FLOAT,!1,0,0),this._u={res:t.getUniformLocation(i,"uRes"),p:t.getUniformLocation(i,"uP"),t:t.getUniformLocation(i,"uT")},this._prog=i}catch(i){console.warn("[loader] shader unavailable",i),this.gl=null,e.classList.add("geo-hidden")}}setProgress(e){this.target=Math.max(0,Math.min(1,e))}start(){if(this._running||!this.gl)return;this._running=!0;const e=t=>{this._running&&(this._draw(t),this._raf=requestAnimationFrame(e))};this._raf=requestAnimationFrame(e)}stop(){this._running=!1,cancelAnimationFrame(this._raf)}_draw(e){const t=this.gl,s=this.canvas,i=Math.min(window.devicePixelRatio||1,2),a=Math.max(1,Math.round(s.clientWidth*i)),r=Math.max(1,Math.round(s.clientHeight*i));(s.width!==a||s.height!==r)&&(s.width=a,s.height=r),this.shown+=(this.target-this.shown)*.12,Math.abs(this.target-this.shown)<.002&&(this.shown=this.target),t.viewport(0,0,a,r),t.uniform2f(this._u.res,a,r),t.uniform1f(this._u.p,this.shown),t.uniform1f(this._u.t,(e-this._t0)/1e3),t.drawArrays(t.TRIANGLES,0,3)}dispose(){this.stop(),this.gl?.getExtension("WEBGL_lose_context")?.loseContext(),this.gl=null}}const ye="neural-earth.coverage",Fe=A=>Math.round(A*1e3)/1e3;class Ct{constructor(e,t){this.el=e,this.cb=t,this.boxes=this._load(),this.pending=null,this._note="",e.innerHTML=`
      <b>Fine-leg coverage picker</b>
      <div class="pk-status"></div>
      <div class="pk-list"></div>
      <div class="pk-actions">
        <button type="button" data-a="copy" title="copy the list as JS for src/data/fineCoverage.js">Copy JS</button>
        <button type="button" data-a="undo" title="cancel the pending corner, or remove the last box">Undo</button>
        <button type="button" data-a="clear">Clear</button>
        <button type="button" data-a="done" title="leave pick mode (C)">Done</button>
      </div>`,this._status=e.querySelector(".pk-status"),this._list=e.querySelector(".pk-list"),e.addEventListener("click",s=>{const i=s.target.closest("button");if(!i)return;const a=i.dataset.a;a==="copy"?this.copy():a==="undo"?this.undo():a==="clear"?this.clear():a==="done"?this.cb.onDone():a==="rm"&&this.remove(parseInt(i.dataset.i,10))}),e.addEventListener("keydown",s=>s.stopPropagation()),this.render()}_load(){try{const e=JSON.parse(localStorage.getItem(ye)||"[]");return Array.isArray(e)?e:[]}catch{return[]}}_save(){try{localStorage.setItem(ye,JSON.stringify(this.boxes))}catch{}}_changed(e=""){this._note=e,this._save(),this.render(),this.cb.onChange(this.boxes,this.pending)}addPoint(e,t){if(e=Fe(e),t=Fe(t),!this.pending){this.pending={lat:e,lon:t},this._changed();return}const s=this.pending;if(this.pending=null,Math.abs(t-s.lon)>180){this._changed("box would cross the antimeridian \u2014 pick it as two boxes");return}this.boxes.push({name:`area ${this.boxes.length+1}`,lat:[Math.min(s.lat,e),Math.max(s.lat,e)],lon:[Math.min(s.lon,t),Math.max(s.lon,t)]}),this._changed()}missed(){this._note="that click missed the globe",this.render()}undo(){if(this.pending){this.pending=null,this._changed("corner cancelled");return}this.boxes.length&&(this.boxes.pop(),this._changed())}remove(e){this.boxes.splice(e,1),this._changed()}clear(){this.boxes=[],this.pending=null,this._changed()}toSource(){return`export const FINE_COVERAGE = [
${this.boxes.map(e=>`  { name: ${JSON.stringify(e.name)}, lat: [${e.lat[0]}, ${e.lat[1]}], lon: [${e.lon[0]}, ${e.lon[1]}] },`).join(`
`)}
];`}async copy(){const e=this.toSource();console.log(`[coverage]
`+e);try{await navigator.clipboard.writeText(e),this._note="copied \u2014 paste into src/data/fineCoverage.js"}catch{this._note="clipboard blocked \u2014 the list is in the console"}this.render()}render(){const e=this.boxes.length;this._status.textContent=this._note||(this.pending?`corner 1: ${this.pending.lat}\xB0, ${this.pending.lon}\xB0 \u2014 click the opposite corner`:`click a corner on the globe${e?` \xB7 ${e} box${e>1?"es":""} \u2014 fine leg routes only inside them`:""}`),this._list.innerHTML=this.boxes.map((t,s)=>`<div class="pk-box"><span>${t.name}</span><span>lat ${t.lat[0]}\u2026${t.lat[1]} \xB7 lon ${t.lon[0]}\u2026${t.lon[1]}</span><button type="button" data-a="rm" data-i="${s}" title="remove">\u2715</button></div>`).join("")}}const pt=[{key:"coarse",stem:"coarse"},{key:"fine",stem:"fine"}],Ge=2048,mt=140,It=1,Dt={normal:40,quiet:24},Mt={high:"low",low:"fallback",fallback:"high"},X=180/Math.PI,vt=(A,e)=>[A[0]*e[0]+A[1]*e[1]+A[2]*e[2],A[3]*e[0]+A[4]*e[1]+A[5]*e[2],A[6]*e[0]+A[7]*e[1]+A[8]*e[2]],xt=A=>[Math.asin(Math.max(-1,Math.min(1,A[2])))*X,Math.atan2(A[1],A[0])*X],ke=(A,e)=>{const t=A/X,s=e/X;return[Math.cos(t)*Math.cos(s),Math.cos(t)*Math.sin(s),Math.sin(t)]},bt=(A,e)=>{try{const t=localStorage.getItem(`neural-earth.${A}`);return t===null?e:JSON.parse(t)}catch{return e}},yt=(A,e)=>{try{localStorage.setItem(`neural-earth.${A}`,JSON.stringify(e))}catch{}},Ft=`
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
  flex-direction: column; gap: 18px; background: #05070d; z-index: 10; text-align: center; padding: 24px; }
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
  .geo-earth .geo-hud { font-size: 10px; } .geo-earth .geo-brand span { display: none; } .geo-earth .geo-help { display: none; }
}
`;function Gt(){if(document.querySelector("style[data-geo-earth]"))return;const A=document.createElement("style");A.dataset.geoEarth="",A.textContent=Ft,document.head.appendChild(A)}const kt=`
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
</div>`;function Ue(A,e={}){const t={weights:"weights",legs:"both",quant:"int4",gpu:"high",cache:!0,brand:!0,help:!0,autofocus:!1,pick:!1,onError:null,...e},s=new URL(t.weights,document.baseURI).toString().replace(/\/+$/,""),i={fp16:"",int8:".i8",int4:".i4"}[t.quant];if(i===void 0)throw new Error(`quant must be fp16 | int8 | int4, got ${t.quant}`);const a=pt.map(w=>({...w,stem:`${w.stem}${i}`}));Gt(),A.classList.add("geo-earth"),A.style.containerType||="inline-size",A.innerHTML=kt;const r=w=>A.querySelector(`.${w}`),n=r("geo-canvas"),o=r("geo-overlay"),c=r("geo-status"),Q=r("geo-substatus");t.brand||r("geo-brand").remove(),t.help||r("geo-help").remove();const l=new ft(r("geo-loader"));l.start();const E=w=>c.textContent=w,f=w=>Q.textContent=w,d=w=>l.setProgress(w),M=()=>{o.classList.add("geo-hidden"),l.stop()},F=new AbortController,G=[];let m=!1,x=0;const b={root:A,opts:t,ready:null,destroy(){if(!m){m=!0,F.abort(),cancelAnimationFrame(x),l.dispose();for(const w of G.splice(0))try{w()}catch{}A.innerHTML="",A.classList.remove("geo-earth"),A.__geo===b&&delete A.__geo}}};A.__geo=b;const I=w=>{m||(o.classList.add("err"),o.classList.remove("geo-hidden"),l.stop(),E(`Error: ${w.message||w}`),f(""),console.error(w),t.onError?.(w))};async function k(){let w;try{w=await ae.create(n,t.gpu==="fallback"?{forceFallbackAdapter:!0}:{powerPreference:t.gpu==="low"?"low-power":"high-performance"})}catch(B){return I(B)}if(m){w.device.destroy();return}w.onLost=B=>I(new Error(`GPU device lost (${B.message}). Reload the page.`));const D=new at,h=new it(w,{fineCoverage:t.fineCoverage});h.onError=I;const p=new ot({scale:w.isFallback?.25:.5}),v=new ht,g={mode:"rgb",sigmaScale:1,forceLeg:-1,showStats:!1,quiet:bt("quiet",!1),quant:t.quant,refine:null,pick:!1,pickPending:null,quality:p,camera:D,legs:[],fine:{status:"pending",progress:0,error:""},canvasW:1,canvasH:1,gpuName:w.gpuName,gpuPref:t.gpu},Se=new ut(r("geo-hud")),He=t.fineCoverage??ve,oe=r("geo-pick"),P=new Ct(oe,{onChange:(B,u)=>{h.setFineCoverage(B),g.pickPending=u?ke(u.lat,u.lon):null,H()},onDone:()=>K.togglePick()});Object.assign(b,{state:g,renderer:h,quality:p,stats:v,camera:D,gpu:w,loader:l,overlay:o}),G.push(()=>{h.dispose(),w.device.destroy()});let ne=0;const H=()=>ne++,K={setMode:B=>{g.mode=B,H()},setLeg:B=>{g.legs.length===2&&(g.forceLeg=B,H())},cycleLeg:()=>K.setLeg(g.forceLeg===-1?0:g.forceLeg===0?1:-1),togglePick:()=>{g.pick=!g.pick,oe.classList.toggle("geo-hidden",!g.pick),h.setFineCoverage(g.pick?P.boxes:He),g.pickPending=g.pick&&P.pending?ke(P.pending.lat,P.pending.lon):null,g.pick&&P.render(),H()},pickPoint:(B,u,C,y)=>{const Y=re(B,u,C,y,D.fovRad,D.dist);if(!Y.hit){P.missed();return}const[J,Ke]=xt(vt(D.rtArray,Y.hitPoint));P.addPoint(J,Ke)},setScale:B=>p.setScale(B),setAuto:B=>p.setAuto(B),toggleQuiet:()=>{g.quiet=!g.quiet,p.setQuiet(g.quiet),yt("quiet",g.quiet)},toggleStats:()=>{g.showStats=!g.showStats},scaleSigma:B=>{g.sigmaScale=Math.max(.05,Math.min(20,g.sigmaScale*B)),H()},reset:()=>{D.reset(),g.sigmaScale=1,H()},fullscreen:()=>document.fullscreenElement?document.exitFullscreen():A.requestFullscreen(),cycleGpu:()=>{b.destroy(),Ue(A,{...t,gpu:Mt[t.gpu]||"high"})},clearCache:async()=>{await ze(),f("weight cache cleared")}};p.setQuiet(g.quiet),t.pick&&K.togglePick(),new gt(n,D,g,K);const ge=new ct(r("geo-toolbar"),g,K);t.autofocus&&n.focus({preventScroll:!0});function Be(){const B=Math.min(window.devicePixelRatio||1,2);let u=Math.max(1,Math.round((n.clientWidth||A.clientWidth)*B)),C=Math.max(1,Math.round((n.clientHeight||A.clientHeight)*B));const y=Math.max(u,C);if(y>Ge){const Y=Ge/y;u=Math.round(u*Y),C=Math.round(C*Y)}(u!==n.width||C!==n.height)&&(n.width=u,n.height=C,g.canvasW=u,g.canvasH=C,H())}const ce=new ResizeObserver(Be);ce.observe(A),G.push(()=>ce.disconnect()),Be();const ee=F.signal;let T;try{T=await Promise.all(a.map(B=>_.fetchManifest(s,B.stem,ee)))}catch(B){return m?void 0:I(new Error(`cannot load weight manifests from ${s} (${B.message})`))}const N=T.map(B=>_.plan(B,w.device)),z=w.maxGridBuffers,O=B=>B.reduce((u,C)=>u+N[C].buffers.length,0)<=z;let L;if(t.legs==="coarse"&&O([0]))L=[0],g.fine={status:"skipped",progress:0,error:"lite mode (coarse only)"};else if(O([0,1]))L=[0,1];else if(O([0]))L=[0],g.fine={status:"skipped",progress:0,error:`adapter allows ${z} grid buffers`};else if(O([1]))L=[1],g.fine={status:"skipped",progress:0,error:"coarse leg does not fit"};else return I(new Error(`neither model fits: adapter allows ${z} storage buffers of ${R(w.limits.maxStorageBufferBindingSize)}`));const Le=L.reduce((B,u)=>B+N[u].totalBytes,0);console.log(`[plan] legs ${L.map(B=>a[B].key).join("+")} \xB7 ${R(Le)} \xB7 buffers ${L.map(B=>N[B].buffers.length).join("+")} / ${z}`);const q=L[0];E(`Downloading the planet \xB7 ${R(N[q].totalBytes)}`),f(t.cache?"first visit downloads the weights; they are cached for next time":"");let W;try{W=await _.loadLeg(w.device,s,a[q].stem,{manifest:T[q],useCache:t.cache,signal:ee,onProgress:B=>{d(B),f(`${(B*100).toFixed(0)}%`)}})}catch(B){return m?void 0:I(B)}if(m){W.dispose?.();return}h.setLegs([W]),g.legs=[a[q].key],d(1),M(),H(),L.length===2&&(g.fine={status:"loading",progress:0,error:""},_.loadLeg(w.device,s,a[1].stem,{manifest:T[1],useCache:t.cache,signal:ee,onProgress:B=>{g.fine.progress=B}}).then(B=>{if(m){B.dispose?.();return}h.setLegs([W,B]),g.legs=["coarse","fine"],g.fine={status:"ready",progress:1,error:""},ge.sync(),H()}).catch(B=>{m||(console.error(B),g.fine={status:"error",progress:0,error:B.message})}));let le=null,Qe=0,$=-1e9,j=null;const Ye=()=>[...D.rotation,D.dist,D.fovRad,g.canvasW,g.canvasH,g.mode,g.sigmaScale,g.forceLeg,g.pick?1:0,h.version,ne,p.scale],Re=(B,u)=>B&&B.length===u.length&&B.every((C,y)=>C===u[y]),Ee=(B,u,C)=>({rt:D.rtArray,dist:D.dist,fovRad:D.fovRad,W:B,H:u,mode:g.mode,sigmaScale:g.sigmaScale,forceLeg:g.forceLeg,showCoverage:g.pick,pick:g.pick?g.pickPending:null,...C}),he=(B,u)=>(C,y)=>{v.gpuTime(performance.now(),C,y),p.onGpuTime(B,C,u)},Pe=B=>{const[u,C]=p.interactiveSize(g.canvasW,g.canvasH);j=[u,C],v.submitted(B,u,C),h.render(Ee(u,C),he("interactive"))},_e=B=>{const u=g.refine,C=g.canvasW,y=g.canvasH;v.submitted(B,Math.ceil(C/u.stride)*Math.ceil(y/u.stride)),h.render(Ee(C,y,{stride:u.stride,phase:u.k}),he("refine",{stride:u.stride,phase:u.k})),u.k++};function ue(B){if(m)return;D.tick();const u=Ye(),C=Dt[g.quiet?"quiet":"normal"];if(!Re(le,u))h.inFlight<It&&B-$>=1e3/C&&(le=u,Qe=B,$=B,g.refine=null,Pe(B));else if(B-Qe>=mt&&h.inFlight===0&&B-$>=1e3/C){if(!g.refine&&j){const[y,Y]=j;if(y!==g.canvasW||Y!==g.canvasH){const J=p.refineStride(y,Y,g.canvasW,g.canvasH);g.refine={stride:J,k:0,n:J*J}}j=null}g.refine&&g.refine.k<g.refine.n&&($=B,_e(B))}v.update(B),Se.update(g,v,h),ge.sync(),x=requestAnimationFrame(ue)}x=requestAnimationFrame(ue)}return b.ready=k(),b}export{Ue as mountNeuralEarth};
