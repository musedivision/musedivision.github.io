import { readdirSync, readFileSync, writeFileSync, mkdirSync, statSync, existsSync } from 'node:fs';
import { join, dirname, basename } from 'node:path';
import { fileURLToPath } from 'node:url';
import matter from 'gray-matter';
import MarkdownIt from 'markdown-it';
import sharp from 'sharp';

const __dirname = dirname(fileURLToPath(import.meta.url));
const ROOT = join(__dirname, '..');
const POSTS_DIR = join(ROOT, 'src', 'posts');
const DATA_DIR = join(ROOT, 'src', 'data');

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
    <a href="/" class="back">&larr; back</a>
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

async function build() {
  mkdirSync(DATA_DIR, { recursive: true });
  const slugs = readPostDirs();
  const posts = [];
  const heatmap = {};

  for (const slug of slugs) {
    const dir = join(POSTS_DIR, slug);
    const raw = readFileSync(join(dir, 'index.md'), 'utf8');
    const { data, content } = matter(raw);

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
    const cover = data.cover ? toWebpPath(data.cover) : null;

    let thumb = null;
    if (cover) {
      if (existsSync(join(dir, cover))) {
        const thumbName = toThumbPath(cover);
        await makeThumb(join(dir, cover), join(dir, thumbName));
        thumb = thumbName;
      } else {
        console.warn(`[posts]   ! ${slug}: cover "${data.cover}" resolves to "${cover}" — file not found in folder`);
      }
    }

    writeFileSync(
      join(dir, 'index.html'),
      renderPostHtml({ title, date, body, cover, coverDim: cover ? dims.get(cover) : null })
    );

    posts.push({
      slug,
      title,
      date: isoDate,
      cover,
      thumb,
      tags: data.tags ?? [],
      excerpt: data.excerpt ?? excerpt(body),
      url: `/posts/${slug}/`,
    });

    heatmap[isoDate] = (heatmap[isoDate] ?? 0) + 1;
  }

  posts.sort((a, b) => b.date.localeCompare(a.date));

  writeFileSync(join(DATA_DIR, 'posts.json'), JSON.stringify(posts, null, 2));
  writeFileSync(join(DATA_DIR, 'heatmap.json'), JSON.stringify(heatmap, null, 2));

  console.log(`[posts] built ${posts.length} post${posts.length === 1 ? '' : 's'}`);
}

build().catch(err => {
  console.error(err);
  process.exit(1);
});
