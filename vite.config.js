import { defineConfig } from 'vite'
import glsl from 'vite-plugin-glsl'
import { readdirSync, statSync, existsSync } from 'node:fs'
import { resolve, join, dirname } from 'node:path'
import { fileURLToPath } from 'node:url'

const __dirname = dirname(fileURLToPath(import.meta.url))
const SRC_DIR = resolve(__dirname, 'src')
const POSTS_DIR = resolve(SRC_DIR, 'posts')

function postInputs() {
  if (!existsSync(POSTS_DIR)) return {}
  const entries = {}
  for (const slug of readdirSync(POSTS_DIR)) {
    const html = join(POSTS_DIR, slug, 'index.html')
    if (existsSync(html) && statSync(html).isFile()) {
      entries[`posts/${slug}`] = html
    }
  }
  return entries
}

// src/blog/index.html plus every generated demo page (src/<slug>/index.html).
function pageInputs() {
  const entries = {}
  for (const name of readdirSync(SRC_DIR)) {
    if (name === 'posts') continue
    const html = join(SRC_DIR, name, 'index.html')
    if (existsSync(html) && statSync(html).isFile()) {
      entries[name] = html
    }
  }
  return entries
}

export default defineConfig({
  plugins: [glsl()],
  root: 'src',
  publicDir: '../public',
  build: {
    outDir: '../dist',
    emptyOutDir: true,
    rollupOptions: {
      input: {
        main: resolve(__dirname, 'src/index.html'),
        ...pageInputs(),
        ...postInputs(),
      },
    },
  },
})
