// Shared rendering for the index pages: cover URL resolution, the post list,
// and the big showcase hero. Both landing.js and blog.js build on this.
import { extractPalette, dispatchPalette } from './palette.js';

// Cover images for posts (src/posts/<slug>/) and demos (src/<slug>/), keyed by the
// folder relative to src/ — matching the `dir` field the build script writes.
const coverModules = {
  ...import.meta.glob('../posts/*/*.webp', { eager: true, query: '?url', import: 'default' }),
  ...import.meta.glob('../*/*.webp', { eager: true, query: '?url', import: 'default' }),
};
const coverUrlByDir = {};
for (const path in coverModules) {
  const m = path.match(/^\.\.\/(.+)\/([^/]+\.webp)$/);
  if (m) {
    coverUrlByDir[m[1]] ??= {};
    coverUrlByDir[m[1]][m[2]] = coverModules[path];
  }
}
export function coverUrlFor(item) {
  if (!item?.cover) return null;
  return coverUrlByDir[item.dir]?.[item.cover] ?? null;
}
export function thumbUrlFor(item) {
  if (!item?.thumb) return null;
  return coverUrlByDir[item.dir]?.[item.thumb] ?? null;
}

export function prefetch(href) {
  if (!href) return;
  const existing = [...document.querySelectorAll('link[rel="prefetch"]')];
  if (existing.some(l => l.href === href || l.getAttribute('href') === href)) return;
  const link = document.createElement('link');
  link.rel = 'prefetch';
  link.href = href;
  document.head.appendChild(link);
}

export function esc(s) {
  return String(s).replace(/[&<>"]/g, c => (
    { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]
  ));
}

export function renderPostList(container, list, { emptyText = 'no posts yet — run `npm run new "title"`' } = {}) {
  if (!container) return;

  if (list.length === 0) {
    container.innerHTML = `<div class="empty">${esc(emptyText)}</div>`;
    return;
  }

  container.innerHTML = list
    .map(p => {
      const t = thumbUrlFor(p);
      const thumb = t
        ? `<img class="thumb" src="${t}" width="48" height="48" loading="lazy" decoding="async" alt="">`
        : '<span class="thumb thumb-empty"></span>';
      return `<a href="${esc(p.url)}">${thumb}<time>${esc(p.date)}</time><span class="post-title">${esc(p.title)}</span></a>`;
    })
    .join('');

  // Warm the cache for a post (HTML + cover) the first time its link is hovered.
  container.querySelectorAll('a').forEach((a, i) => {
    a.addEventListener('mouseenter', () => {
      prefetch(a.href);
      prefetch(coverUrlFor(list[i]));
    }, { once: true });
  });
}

// Render one post's cover as the hero link and push its palette into the shader.
export function renderShowcase(el, post, { caption } = {}) {
  if (!el || !post) return;
  const url = coverUrlFor(post);
  if (!url) return;
  el.href = post.url;
  el.innerHTML = `<img src="${url}" alt=""><span class="showcase-caption">${esc(caption ?? post.title)}</span>`;
  el.hidden = false;
  el.addEventListener('mouseenter', () => prefetch(el.href), { once: true });

  extractPalette(url).then(dispatchPalette).catch(() => {});
}
