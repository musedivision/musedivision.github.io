import { readdirSync, readFileSync, writeFileSync, mkdirSync, statSync, existsSync } from 'node:fs';
import { join, dirname, basename } from 'node:path';
import { fileURLToPath } from 'node:url';
import matter from 'gray-matter';
import MarkdownIt from 'markdown-it';
import sharp from 'sharp';

const __dirname = dirname(fileURLToPath(import.meta.url));
const ROOT = join(__dirname, '..');
const SRC_DIR = join(ROOT, 'src');
const POSTS_DIR = join(SRC_DIR, 'posts');
const DATA_DIR = join(SRC_DIR, 'data');

// Two content types share this pipeline:
//   src/posts/<slug>/index.md  -> blog post at /posts/<slug>/   (column of text, lofi)
//   src/<slug>/index.md        -> demo page at /<slug>/          (full-viewport stage + tiny nav)
const IMAGE_EXT_RE = /\.(jpe?g|png|gif)$/i;
const MAX_WIDTH = 1600;        // width of the shipped .webp
const WEBP_QUALITY = 80;
const THUMB_WIDTH = 240;       // width of the home-page list thumbnail
const THUMB_QUALITY = 72;

const md = new MarkdownIt({ html: true, linkify: true, typographer: true });

// Dimensions of the current post's images, keyed by .webp filename. Set per-post
// in build() so the image renderer can stamp width/height (prevents layout shift).
let currentDims = new Map();

md.renderer.rules.image = (tokens, idx, options, env, slf) => {
  const token = tokens[idx];
  const src = token.attrGet('src');
  if (src) {
    const webp = toWebpPath(src);
    token.attrSet('src', webp);
    token.attrSet('loading', 'lazy');
    token.attrSet('decoding', 'async');
    const d = currentDims.get(webp);
    if (d) {
      token.attrSet('width', String(d.w));
      token.attrSet('height', String(d.h));
    }
  }
  return slf.renderToken(tokens, idx, options);
};

function toWebpPath(src) {
  return IMAGE_EXT_RE.test(src) ? src.replace(IMAGE_EXT_RE, '.webp') : src;
}

function toThumbPath(webp) {
  return webp.replace(/\.webp$/i, '.thumb.webp');
}

function readPostDirs() {
  if (!existsSync(POSTS_DIR)) return [];
  return readdirSync(POSTS_DIR).filter(name => {
    const p = join(POSTS_DIR, name);
    return statSync(p).isDirectory() && existsSync(join(p, 'index.md'));
  });
}

// Demo pages are top-level folders in src/ that carry an index.md (posts/ is excluded).
function readDemoDirs() {
  return readdirSync(SRC_DIR).filter(name => {
    if (name === 'posts') return false;
    const p = join(SRC_DIR, name);
    return statSync(p).isDirectory() && existsSync(join(p, 'index.md'));
  });
}

function kb(bytes) {
  return `${(bytes / 1024).toFixed(0)} kB`;
}

// Compress png/jpg sources to .webp, then return a Map of .webp filename -> {w,h}.
async function compressImagesIn(dir) {
  const sources = readdirSync(dir).filter(f => IMAGE_EXT_RE.test(f));
  for (const f of sources) {
    const src = join(dir, f);
    const dest = join(dir, f.replace(IMAGE_EXT_RE, '.webp'));
    if (existsSync(dest) && statSync(dest).mtimeMs >= statSync(src).mtimeMs) continue;
    // animated: true keeps every frame of a .gif so it becomes an animated .webp
    await sharp(src, { animated: /\.gif$/i.test(f) })
      .rotate()
      .resize({ width: MAX_WIDTH, withoutEnlargement: true })
      .webp({ quality: WEBP_QUALITY })
      .toFile(dest);
    console.log(`[posts]   ${f} -> .webp (${kb(statSync(dest).size)})`);
  }

  const dims = new Map();
  for (const f of readdirSync(dir)) {
    if (/\.webp$/i.test(f) && !/\.thumb\.webp$/i.test(f)) {
      const meta = await sharp(join(dir, f)).metadata();
      dims.set(f, { w: meta.width, h: meta.height });
    }
  }
  return dims;
}

async function makeThumb(webpPath, thumbPath) {
  if (existsSync(thumbPath) && statSync(thumbPath).mtimeMs >= statSync(webpPath).mtimeMs) return;
  await sharp(webpPath)
    .resize({ width: THUMB_WIDTH, withoutEnlargement: true })
    .webp({ quality: THUMB_QUALITY })
    .toFile(thumbPath);
  console.log(`[posts]   ${basename(thumbPath)} (${kb(statSync(thumbPath).size)})`);
}

// Resolve the frontmatter cover to its .webp and make the list thumbnail.
async function resolveCover(dir, slug, data) {
  const cover = data.cover ? toWebpPath(data.cover) : null;
  if (!cover) return { cover: null, thumb: null };
  if (!existsSync(join(dir, cover))) {
    console.warn(`[posts]   ! ${slug}: cover "${data.cover}" resolves to "${cover}" — file not found in folder`);
    return { cover, thumb: null };
  }
  const thumb = toThumbPath(cover);
  await makeThumb(join(dir, cover), join(dir, thumb));
  return { cover, thumb };
}

function renderPostHtml({ title, date, body, cover, coverDim }) {
  const isoDate = new Date(date).toISOString().slice(0, 10);
  const dimAttr = coverDim ? ` width="${coverDim.w}" height="${coverDim.h}"` : '';
  const coverImg = cover
    ? `      <img class="cover" src="${escapeHtml(cover)}"${dimAttr} alt="" fetchpriority="high" decoding="async">\n`
    : '';
  return `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>${escapeHtml(title)} — musedivision</title>
  <link rel="preload" href="/fonts/share-tech-mono.woff2" as="font" type="font/woff2" crossorigin>
  <link rel="stylesheet" href="/css/post.css">
</head>
<body>
  <article class="post">
    <a href="/blog/" class="back">&larr; back</a>
    <header class="post-header">
      <time datetime="${isoDate}">${isoDate}</time>
      <h1>${escapeHtml(title)}</h1>
    </header>
${coverImg}    <div class="post-body">
${body}
    </div>
  </article>
</body>
</html>
`;
}

// Full-viewport page: the markdown body IS the stage (a demo mounts itself into it);
// the only chrome is a small top-right nav. The viewer's own HUD sits top-left,
// its toolbar bottom-centre, so top-right is the free corner.
function renderDemoHtml({ title, body, excerpt, postUrl }) {
  const writeup = postUrl
    ? `\n    <a href="${escapeHtml(postUrl)}" class="writeup">writeup &rarr;</a>`
    : '';
  const description = excerpt ? `\n  <meta name="description" content="${escapeHtml(excerpt)}">` : '';
  return `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0, viewport-fit=cover">
  <title>${escapeHtml(title)} — musedivision</title>${description}
  <link rel="preload" href="/fonts/share-tech-mono.woff2" as="font" type="font/woff2" crossorigin>
  <link rel="stylesheet" href="/css/demo.css">
</head>
<body class="demo">
  <main class="demo-stage">
${body}
  </main>
  <nav class="demo-nav">
    <a href="/" class="back">&larr; musedivision</a>
    <span class="demo-title">${escapeHtml(title)}</span>${writeup}
  </nav>
</body>
</html>
`;
}

function escapeHtml(s = '') {
  return String(s)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

function excerpt(body, n = 180) {
  const text = body.replace(/<[^>]+>/g, '').replace(/\s+/g, ' ').trim();
  return text.length > n ? text.slice(0, n).trimEnd() + '…' : text;
}

async function buildPosts() {
  const posts = [];
  const heatmap = {};

  for (const slug of readPostDirs()) {
    const dir = join(POSTS_DIR, slug);
    const { data, content } = matter(readFileSync(join(dir, 'index.md'), 'utf8'));

    if (data.draft) continue;

    const dims = await compressImagesIn(dir);
    currentDims = dims;

    const title = data.title ?? slug;
    const date = data.date ?? slug.slice(0, 10);
    const parsedDate = new Date(date);
    if (Number.isNaN(parsedDate.getTime())) {
      console.warn(`[posts]   ! ${slug}: invalid date "${date}" — add a valid \`date:\` to the frontmatter. Skipping.`);
      continue;
    }
    const isoDate = parsedDate.toISOString().slice(0, 10);
    const body = md.render(content);
    const { cover, thumb } = await resolveCover(dir, slug, data);

    writeFileSync(
      join(dir, 'index.html'),
      renderPostHtml({ title, date, body, cover, coverDim: cover ? dims.get(cover) : null })
    );

    posts.push({
      slug,
      dir: `posts/${slug}`,
      title,
      date: isoDate,
      cover,
      thumb,
      tags: data.tags ?? [],
      featured: Boolean(data.featured),
      excerpt: data.excerpt ?? excerpt(body),
      url: `/posts/${slug}/`,
    });

    heatmap[isoDate] = (heatmap[isoDate] ?? 0) + 1;
  }

  posts.sort((a, b) => b.date.localeCompare(a.date));
  return { posts, heatmap };
}

async function buildDemos(postSlugs) {
  const demos = [];

  for (const slug of readDemoDirs()) {
    const dir = join(SRC_DIR, slug);
    const { data, content } = matter(readFileSync(join(dir, 'index.md'), 'utf8'));

    if (data.draft) continue;

    const dims = await compressImagesIn(dir);
    currentDims = dims;

    const title = data.title ?? slug;
    const body = md.render(content);
    const { cover, thumb } = await resolveCover(dir, slug, data);

    // `post:` names the blog post that explains the demo (by folder slug).
    let postUrl = null;
    if (data.post) {
      if (postSlugs.has(data.post)) postUrl = `/posts/${data.post}/`;
      else console.warn(`[posts]   ! ${slug}: post "${data.post}" not found in src/posts — writeup link skipped`);
    }

    writeFileSync(join(dir, 'index.html'), renderDemoHtml({ title, body, excerpt: data.excerpt, postUrl }));

    demos.push({
      slug,
      dir: slug,
      title,
      cover,
      thumb,
      excerpt: data.excerpt ?? '',
      order: Number(data.order ?? 0),
      post: postUrl,
      url: `/${slug}/`,
    });
  }

  // Frontmatter `order` (lower first) decides which demo the landing page leads with.
  demos.sort((a, b) => a.order - b.order);
  return demos;
}

async function build() {
  mkdirSync(DATA_DIR, { recursive: true });

  const { posts, heatmap } = await buildPosts();
  const demos = await buildDemos(new Set(posts.map(p => p.slug)));

  writeFileSync(join(DATA_DIR, 'posts.json'), JSON.stringify(posts, null, 2));
  writeFileSync(join(DATA_DIR, 'heatmap.json'), JSON.stringify(heatmap, null, 2));
  writeFileSync(join(DATA_DIR, 'demos.json'), JSON.stringify(demos, null, 2));

  const n = (arr, word) => `${arr.length} ${word}${arr.length === 1 ? '' : 's'}`;
  console.log(`[posts] built ${n(posts, 'post')}, ${n(demos, 'demo')}`);
}

build().catch(err => {
  console.error(err);
  process.exit(1);
});
