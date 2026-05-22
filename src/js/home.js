import posts from '../data/posts.json';
import heatmap from '../data/heatmap.json';
import { extractPalette, dispatchPalette } from './palette.js';

const coverModules = import.meta.glob('../posts/*/*.webp', { eager: true, query: '?url', import: 'default' });
const coverUrlBySlug = {};
for (const path in coverModules) {
  const m = path.match(/\/posts\/([^/]+)\/([^/]+\.webp)$/);
  if (m) {
    coverUrlBySlug[m[1]] ??= {};
    coverUrlBySlug[m[1]][m[2]] = coverModules[path];
  }
}
function coverUrlFor(post) {
  if (!post.cover) return null;
  return coverUrlBySlug[post.slug]?.[post.cover] ?? null;
}
function thumbUrlFor(post) {
  if (!post.thumb) return null;
  return coverUrlBySlug[post.slug]?.[post.thumb] ?? null;
}

function prefetch(href) {
  if (!href) return;
  const existing = [...document.querySelectorAll('link[rel="prefetch"]')];
  if (existing.some(l => l.href === href || l.getAttribute('href') === href)) return;
  const link = document.createElement('link');
  link.rel = 'prefetch';
  link.href = href;
  document.head.appendChild(link);
}

function esc(s) {
  return String(s).replace(/[&<>"]/g, c => (
    { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]
  ));
}

const WEEKLY_GOAL = 3;
const GOAL_END = '2026-12-31'; // goal: 3 posts/week until the end of 2026
const TZ = 'Europe/Amsterdam';

// All week math runs on UTC-midnight "civil date" anchors so it is timezone-stable
// wherever the page is viewed. The only TZ-aware step is reading today's Amsterdam date.
function civilToday() {
  const [y, m, d] = new Intl.DateTimeFormat('en-CA', {
    timeZone: TZ, year: 'numeric', month: '2-digit', day: '2-digit',
  }).format(new Date()).split('-').map(Number);
  return new Date(Date.UTC(y, m - 1, d));
}

function isoDay(d) {
  return d.toISOString().slice(0, 10);
}

function addDays(d, n) {
  const x = new Date(d);
  x.setUTCDate(x.getUTCDate() + n);
  return x;
}

function startOfWeek(d) {
  // Monday-start week: getUTCDay() is 0=Sun..6=Sat; map so Monday -> 0.
  return addDays(d, -((d.getUTCDay() + 6) % 7));
}

function countPostsInWeek(weekStart, today) {
  let count = 0;
  for (let d = 0; d < 7; d++) {
    const date = addDays(weekStart, d);
    if (date > today) break;
    count += heatmap[isoDay(date)] ?? 0;
  }
  return count;
}

function renderStats() {
  const today = civilToday();
  const thisWeekStart = startOfWeek(today);
  const weekCount = countPostsInWeek(thisWeekStart, today);

  let streak = 0;
  let cursorWeek = new Date(thisWeekStart);
  const currentWeekMet = weekCount >= WEEKLY_GOAL;
  if (!currentWeekMet) cursorWeek = addDays(cursorWeek, -7);
  while (countPostsInWeek(cursorWeek, today) >= WEEKLY_GOAL) {
    streak++;
    cursorWeek = addDays(cursorWeek, -7);
  }

  const weekEl = document.getElementById('week-count');
  const streakEl = document.getElementById('streak');
  if (weekEl) weekEl.textContent = weekCount;
  if (streakEl) streakEl.textContent = streak;
  if (weekEl && currentWeekMet) weekEl.classList.add('met');
}

function colorForWeek(count) {
  if (count === 0) return 'rgba(255,255,255,0.05)';
  if (count >= WEEKLY_GOAL) return 'rgba(110,225,140,0.95)';
  if (count === 2) return 'rgba(85,180,110,0.7)';
  return 'rgba(70,140,90,0.5)';
}

function civilFromIso(s) {
  const [y, m, d] = s.split('-').map(Number);
  return new Date(Date.UTC(y, m - 1, d));
}

function renderHeatmap() {
  const svg = document.getElementById('heatmap');
  if (!svg) return;

  const cellSize = 16;
  const gap = 5;
  const pad = 2;
  const today = civilToday();
  const thisWeek = startOfWeek(today);

  // Bounded campaign: from the first post's week through the last week of 2026.
  const endWeek = startOfWeek(civilFromIso(GOAL_END));
  const firstPost = posts.length ? civilFromIso(posts[posts.length - 1].date) : today;
  const startWeek = startOfWeek(firstPost);
  const WEEKS = Math.max(1, Math.round((endWeek - startWeek) / (7 * 86400000)) + 1);

  const width = WEEKS * (cellSize + gap) - gap + pad * 2;
  const height = cellSize + pad * 2;
  svg.setAttribute('viewBox', `0 0 ${width} ${height}`);
  svg.setAttribute('width', width);
  svg.setAttribute('height', height);

  const parts = [];
  for (let w = 0; w < WEEKS; w++) {
    // Chronological: w = 0 is the goal's first week (leftmost), end of 2026 at the right.
    const weekStart = addDays(startWeek, w * 7);
    const count = countPostsInWeek(weekStart, today);
    const isCurrent = weekStart.getTime() === thisWeek.getTime();
    const x = pad + w * (cellSize + gap);
    const outline = isCurrent ? ' stroke="rgba(255,255,255,0.7)" stroke-width="1.5"' : '';
    const label = `week of ${isoDay(weekStart)} — ${count}/${WEEKLY_GOAL}${count >= WEEKLY_GOAL ? ' ✓' : ''}`;
    parts.push(
      `<rect x="${x}" y="${pad}" width="${cellSize}" height="${cellSize}" rx="3" fill="${colorForWeek(count)}"${outline}><title>${label}</title></rect>`
    );
  }
  svg.innerHTML = parts.join('');
}

function renderPosts() {
  const container = document.getElementById('posts');
  if (!container) return;

  if (posts.length === 0) {
    container.innerHTML = '<div class="empty">no posts yet — run `npm run new "title"`</div>';
    return;
  }

  container.innerHTML = posts
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
      prefetch(coverUrlFor(posts[i]));
    }, { once: true });
  });
}

function renderShowcase() {
  const el = document.getElementById('showcase');
  if (!el) return;
  const withCovers = posts
    .map(p => ({ post: p, url: coverUrlFor(p) }))
    .filter(x => x.url);
  if (withCovers.length === 0) return;
  const pick = withCovers[Math.floor(Math.random() * withCovers.length)];
  el.href = pick.post.url;
  el.innerHTML = `<img src="${pick.url}" alt=""><span class="showcase-caption">${esc(pick.post.title)}</span>`;
  el.hidden = false;

  extractPalette(pick.url).then(dispatchPalette).catch(() => {});
}

renderStats();
renderHeatmap();
renderShowcase();
renderPosts();
