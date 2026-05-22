import { defineConfig } from 'vite'
import glsl from 'vite-plugin-glsl'
import { readdirSync, statSync, existsSync } from 'node:fs'
import { resolve, join, dirname } from 'node:path'
import { fileURLToPath } from 'node:url'

const __dirname = dirname(fileURLToPath(import.meta.url))
const POSTS_DIR = resolve(__dirname, 'src/posts')

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
        ...postInputs(),
      },
    },
  },
})
