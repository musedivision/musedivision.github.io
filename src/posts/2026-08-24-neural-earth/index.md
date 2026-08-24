---
title: "Neural Earth, entangled computation"
date: 2026-08-24
cover: cover.png
tags: [ml, webgpu, terrain, neural-field]
excerpt: "The whole planet as one hash-grid MLP, evaluated per pixel in a WGSL compute kernel. No tiles, no textures — drag it around below."
---

<style>
  /* the frontmatter cover is only for the homepage thumbnail — the viewer's poster shows it here */
  .post > .cover { display: none; }
  /* break out of the 42rem column */
  .earth-wrap { width: min(100vw - 3rem, 64rem); margin: 2rem 0 2rem 50%; transform: translateX(-50%); }
  #earth { position: relative; width: 100%; aspect-ratio: 16 / 10; background: #05070d; border: 1px solid rgba(255,255,255,0.08); overflow: hidden; }
  #earth .earth-poster { position: absolute; inset: 0; width: 100%; height: 100%; object-fit: cover; margin: 0; border: 0; opacity: .55; }
  .earth-gate { position: absolute; inset: 0; display: flex; flex-direction: column; align-items: center; justify-content: center; gap: .8rem; text-align: center; padding: 1.5rem; }
  .earth-start { font: inherit; font-size: 1rem; letter-spacing: .08em; text-transform: lowercase; color: #fff; background: rgba(0,0,0,.55); border: 1px solid rgba(255,255,255,.6); padding: .7rem 1.4rem; cursor: pointer; }
  .earth-start:hover { background: rgba(255,255,255,.12); }
  .earth-note { font-size: .8rem; color: rgba(255,255,255,.6); max-width: 28rem; }
  @media (max-width: 720px) { #earth { aspect-ratio: 4 / 5; } }
</style>

<div class="earth-wrap"><div id="earth"><img class="earth-poster" src="cover.webp" alt="" decoding="async"></div></div>
<script type="module" src="./mount.js"></script>

<p class="caption">Drag to rotate, scroll to zoom. Keys: 1–4 pick a layer, R resets, F goes fullscreen, S hides the stats.</p>

Everything you see above is a neural network. There is no globe texture, no tile pyramid, no mesh — each pixel casts a ray at a sphere, takes the hit point, looks it up in a multi-resolution hash grid and pushes the result through a small MLP. That happens for every pixel, every frame, in a single WebGPU compute kernel on your GPU.

The model is a multi-scale spherical neural field. It is an [instant-NGP](https://nvlabs.github.io/instant-ngp/) style hash grid: 16 resolution levels, two features per level, hashed into a 2<sup>24</sup>-entry table, followed by a 64-wide MLP with two hidden layers. Seven output channels — RGB, elevation, tree cover and a couple of extras — come out of one forward pass, so the layer buttons just pick which channels to show.

- WebGPU only: Chrome and Edge 113+, Safari 26+, Firefox 141+, on a real GPU. On an integrated GPU the interactive resolution scales itself down and refines when you stop dragging, capped at 40 frames/s
