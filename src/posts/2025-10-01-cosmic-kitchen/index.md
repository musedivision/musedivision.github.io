---
title: "Cosmic Kitchen: Cooking Planets for Melba"
date: 2025-10-01
cover: cover.jpg
tags: [ml, terrain, procedural, preface]
excerpt: "How we cook low-resolution planet basemaps offline — spherical noise, a simple climate recipe, cellular-automata vegetation, and a satellite-texture diffusion model — for Preface 0.3."
featured: true
---

<p class="caption">Originally published on the <a href="https://playerunknownproductions.net/news/cosmic-kitchen-cooking-planets-for-melba">PLAYERUNKNOWN Productions blog</a>, 1 October 2025. Including work by Noel Lopez Gonzaga and Saul Chung.</p>

Our vision for [Artemis](https://playerunknownproductions.net/projects/project-artemis) is to create vast, Earth-scale worlds where thousands of players can explore together. Building natural environments of this scale, that are believable, consistent, and playable, is a massive challenge. But with the right tools, we believe it's possible.

Back in Preface 0.1, we showcased our Earth-scale terrain system powered by machine learning (ML) based on [multilayer upsampling](https://playerunknownproductions.net/news/a-multistep-approach-to-upsampling-for-planetary-scale). With Preface 0.3, we take it one step further: our planet-generation pipeline learns from real patterns while letting us [procedurally guide](https://playerunknownproductions.net/news/guided-generation-in-prologue-ml-generated-landscapes-based-on-game-designer-inpu) the outcome, allowing us to create a selection of new planets. We cook low-resolution basemaps offline, on which the Melba engine runs upsampling in real time on your GPU.

This blog will discuss this offline basemap generation tool and its mixture of procedural and machine learning techniques.

![There are new planets to explore in v0.3 like Anesidora](anesidora.png)

<p class="caption">There are new planets to explore in v0.3 like Anesidora</p>

## Growing Pains

Planets are a mixture of different ingredients: temperature and humidity, among many others, shaping land and sea. Distance from the star sets the baseline: too close like Venus and you bake; too far like Pluto and you freeze. Earth sits in a Goldilocks range with temperatures roughly between -50°C and 50°C where liquid water and an oxygen-rich atmosphere support organic life.

In Preface 0.1, our planet was a cube, with each face generated independently and containing a continent in the center, but lacking the global phenomena mentioned above. There were no ice caps or Sahara-like deserts. The smooth terrain noise that guided the heightmap generation, as well as procedural stages such as tree density, rainfall and temperature, had no relationship to their location on the globe. Furthermore, a whole planet required over 4,000 machine learning generated heightmaps and textures to be stitched together which took over 1.5 hours on consumer hardware.

![A continent on each face of the planet](cube-faces.jpg)

<p class="caption">A continent on each face of the planet</p>

## Cooking Up a Planet

The recipe for a Melba planet always starts with the land. Procedurally generated noise guides an ML model which adds realistic details. Now we have a planet like [Bob](https://preface.pp.studio/bookmarks?longitude=-1.8078319110397696&latitude=0.6951971428370112&altitude=600000.586887&rotation=1.3337607425500235,-0.8755991839578855,0&name=preface_teleport&planet_name=blondewalnut), no ocean, no life, only mountains. We must layer on rainfall and temperature which allow vegetation to grow (which we call tree density). Finally, we generate a far distance color texture guided by the heightmap, tree density and biome.

![Planet generation pipeline](scheme.png)

To fix our blending and consistency issues we had to move from a 2D → 3D space. Previously we generated noise or climate data in 2D, sampling noise at each point in a [cube face](https://mathproofs.blogspot.com/2005/07/mapping-cube-to-sphere.html) (x, y). In 3D, we project a unit sphere into our cube and sample noise from this spherical space (x, y, z). Because every point is generated from the same continuous 3D space, the noise smoothly transitions across cube face boundaries and is globally consistent.

Even with powerful tools, iteration times can become long, and when the process is still experimental, optimizing too early can slow progress rather than speed it up. Instead of implementing this for an Earth-sized planet, we scaled down our generation pipeline, and by reducing the planet size to half the Moon, we were able to generate a planet in less than 10 seconds. Once we solved the smaller version, we applied the same methods to reach Earth-sized worlds.

![Cube faces](faces.png)

![Sampling noise on a sphere projected into a cube](sphere-noise.png)

## A Simple Climate Recipe

We are not trying to prove a complex climate thesis, but rather build tooling for artists and players to create the worlds they can imagine. We want simple climate levers that let worlds swing from tropical rainforests to icy tundras to barren wastelands. Nudge the base temperature upward and watch snowy regions retreat and deserts grow.

By using a spherical coordinate space, rainfall and temperature can be modeled with broad, readable rules that act globally. For example, due to the "spherical" shape of Earth, the equator receives the most sunlight and poles the least. Using this latitude-based temperature baseline and mixing in noise for local variations we get familiar climate zones out of the box. We can also introduce other variations of temperature such as dependencies with elevation. Temperature can drop at higher altitudes. Deserts form when there is little rainfall and of course, rainforests when there is a lot. On Earth we see three peaks, most rain at the equator with two peaks at **50°** above and below.

![Temperature by latitude](temperature.png)

![Rainfall by latitude](rainfall.png)

## Life Finds a Way

Now that our planet has a climate it is time to mix in some vegetation. We simulate tree growth as a simple cellular-automata system driven by temperature, rainfall, and elevation. Like the rule set in [Conway's Game of Life](https://en.wikipedia.org/wiki/Conway%27s_Game_of_Life), trees spread to neighboring cells when local conditions fall within favorable ranges. This results in life-like growth patterns similar to nature, and responsive to climate conditions. We then use this to drive tree asset placement in the engine as well as guide the Satellite texture model.

Finally we generate the satellite texture, a far distance color texture which is guided by the globally changing heightmap, tree density and climate conditions. We feed in heightmap, tree density, and a biome class into a controlnet style [latent diffusion model](/posts/2024-09-10-satellite-like-landscape-visuals/) trained on real-world satellite images. When the input data is consistent globally and roughly realistic it makes the ML model's job easier.

![Vegetation spreading across the planet](vegetation.gif)

## A Planet We Have Already Experimented With

While working on vegetation and climate we had a chance to experiment by applying these to the real planet surface of Mars. You can explore [Mars](https://preface.pp.studio/bookmarks?longitude=-1.8078319110397696&latitude=0.6951971428370112&altitude=600000.586887&rotation=1.3337607425500235,-0.8755991839578855,0&name=preface_teleport&planet_name=Mars) as it was captured by NASA's [Mars Orbiter Laser Altimeter](https://science.gsfc.nasa.gov/attic/mola/) in Preface 0.3. While an ocean now floods the lower basins of the red planet, the dry cold conditions mean no life survives to grow. But what if Mars did support life?

If Mars gained an atmosphere and the temperatures began to rise we might see vegetation spread over its surface. In the equatorial regions forests would grow and grasses would stretch out over vast plains. Water would form lakes inside ancient impact craters and snowy caps would appear on its high mountain peaks.

![Did ancient oceans carve the scar-like Valles Marineris on Mars?](mars.png)

<p class="caption">Did ancient oceans carve the scar-like Valles Marineris on Mars?</p>

## The Frontier Is Everywhere

Our goal is to give artists and players the tools to cook up their own worlds. Think of our planet generator as a recipe book: you choose the ingredients, and it assembles land, climate, and vegetation. Machine learning learns patterns from the real world, while clear procedural rules keep the results steerable. By shrinking the problem during development, we even got small planets for free.

This foundation has let us ship **five new planets** in Preface **0.3**. Explore them [here](https://playerunknown-productions.github.io/preface-globe-navigator/) and drop into the forests of [Anesidora](https://preface.pp.studio/bookmarks?longitude=-1.8078319110397696&latitude=0.6951971428370112&altitude=600000.586887&rotation=1.3337607425500235,-0.8755991839578855,0&name=preface_teleport&planet_name=anesidora), skim the lochs of [Dractun](https://preface.pp.studio/bookmarks?longitude=-1.8078319110397696&latitude=0.6951971428370112&altitude=600000.586887&rotation=1.3337607425500235,-0.8755991839578855,0&name=preface_teleport&planet_name=dractun), wander the archipelagos of [Zihan](https://preface.pp.studio/bookmarks?longitude=-1.8078319110397696&latitude=0.6951971428370112&altitude=600000.586887&rotation=1.3337607425500235,-0.8755991839578855,0&name=preface_teleport&planet_name=zihan) or even visit barren [Bob](https://preface.pp.studio/bookmarks?longitude=-1.8078319110397696&latitude=0.6951971428370112&altitude=600000.586887&rotation=1.3337607425500235,-0.8755991839578855,0&name=preface_teleport&planet_name=blondewalnut) and [Mars](https://preface.pp.studio/bookmarks?longitude=-1.8078319110397696&latitude=0.6951971428370112&altitude=600000.586887&rotation=1.3337607425500235,-0.8755991839578855,0&name=preface_teleport&planet_name=Mars).

What kind of planet would you make?

*Note: To load a planet in v0.3, make sure [Preface](https://store.steampowered.com/app/2820060/Preface_Undiscovered_World) is not already running and then use any of the links above.*
