---
title: "Neural Earth, entangled computation"
date: 2026-08-24
cover: cover.png
tags: [ml, webgpu, terrain, neural-field]
featured: true
excerpt: "The whole planet as one hash-grid MLP, evaluated per pixel in a WGSL compute kernel. No tiles, no textures — drag it around below."
---

<style>
  /* the frontmatter cover is only for the homepage thumbnail — the viewer's poster shows it here */
  .post > .cover { display: none; }
  /* break out of the 42rem column */
  .earth-wrap { width: min(100vw - 3rem, 64rem); margin: 2rem 0 2rem 50%; transform: translateX(-50%); }
  #earth { position: relative; width: 100%; aspect-ratio: 16 / 10; background: #05080f; border: 1px solid rgba(255,255,255,0.08); overflow: hidden; }
  @media (max-width: 720px) { #earth { aspect-ratio: 4 / 5; } }
</style>

<div class="earth-wrap"><div id="earth"><img class="earth-poster" src="poster.webp" alt="" decoding="async"></div></div>
<script type="module" src="./mount.js"></script>

<p class="caption">Drag to rotate, scroll to zoom. Keys: 1–3 pick a layer, R resets, F goes fullscreen, S shows the stats.</p>

The globe above is using a neural texture. There is no tile pyramid, no mesh — each pixel casts a ray at a sphere, takes the hit point, looks it up in a multi-resolution hash grid and pushes the result through a small MLP. That happens for every pixel, every frame, in a single WebGPU compute kernel on your GPU.

The model is a multi-scale spherical neural field. It is an [instant-NGP](https://nvlabs.github.io/instant-ngp/) style hash grid: 16 resolution levels, two features per level, hashed into a 2<sup>24</sup>-entry table, followed by a 64-wide MLP with two hidden layers. 3 outputs — RGB, elevation, tree cover. There is no mip map pyramid involved here. The model takes a per-pixel footprint (sigma) as input and gates the hash grid levels with it, [Zip-NeRF](https://jonbarron.info/zipnerf/) style: each level's features are scaled by erf(1 / (√8 · σ · res)), so fine levels fade out under a wide pixel footprint. When a level is gated off the kernel skips its lookups entirely, which is why zoomed-out frames are cheap.

![Australia rendered by the model, with only sparkle where New Zealand should be](missingnz.png)

<p class="caption">Australia is there. New Zealand is not.</p>

There are lots of sparkly artifacts over the ocean because these data tiles were removed from the model training dataset. Except it seems the algorithm for removing ocean tiles was a bit aggressive. New Zealand is frequently forgotten on world maps, and I have committed the same crime accidentally here.

It's a fun demo, but also a silly one. The hash grid training works, and it works because every location shares the same 2<sup>24</sup>-entry table at every hashed level. Nothing is stored per place. Anywhere on the planet can land in any table slot, so every point is entangled with every other, and querying a single pixel needs the whole grid resident on the GPU. There is no such thing as loading the area around the camera.

![Zoomed deep into one of the two high-resolution regions](place1.png)

![Zoomed deep into the other high-resolution region](place2.png)

<p class="caption">Two places where you can keep zooming. No hints.</p>

I've also left two easter egg locations where you can zoom extra deep. Another model was trained on higher res data for smaller 200km2 areas. Can you find them?
