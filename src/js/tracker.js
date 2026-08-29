// Posting-habit tracker: weekly stats + heatmap strip. Blog page only.
import { esc } from './posts-ui.js';

const WEEKLY_GOAL = 3;
const GOAL_START = '2026-08-24'; // the campaign starts here; older posts stack up as a backlog
const GOAL_END = '2026-12-31';   // goal: 3 posts/week until the end of 2026
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

function civilFromIso(s) {
  const [y, m, d] = s.split('-').map(Number);
  return new Date(Date.UTC(y, m - 1, d));
}

function countPostsInWeek(heatmap, weekStart, today) {
  let count = 0;
  for (let d = 0; d < 7; d++) {
    const date = addDays(weekStart, d);
    if (date > today) break;
    count += heatmap[isoDay(date)] ?? 0;
  }
  return count;
}

function renderStats(heatmap) {
  const today = civilToday();
  const thisWeekStart = startOfWeek(today);
  const weekCount = countPostsInWeek(heatmap, thisWeekStart, today);

  let streak = 0;
  let cursorWeek = new Date(thisWeekStart);
  const currentWeekMet = weekCount >= WEEKLY_GOAL;
  if (!currentWeekMet) cursorWeek = addDays(cursorWeek, -7);
  while (countPostsInWeek(heatmap, cursorWeek, today) >= WEEKLY_GOAL) {
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

function renderHeatmap(posts, heatmap) {
  const svg = document.getElementById('heatmap');
  if (!svg) return;

  const cellSize = 16;
  const gap = 5;
  const pad = 2;
  const today = civilToday();
  const thisWeek = startOfWeek(today);

  // Bounded campaign: from GOAL_START's week through the last week of 2026.
  // Posts older than that don't get their own weeks — they pile up as a
  // backlog of small blips to the left of a divider, one per post.
  const endWeek = startOfWeek(civilFromIso(GOAL_END));
  const startWeek = startOfWeek(civilFromIso(GOAL_START));
  const WEEKS = Math.max(1, Math.round((endWeek - startWeek) / (7 * 86400000)) + 1);

  const backlog = posts.filter(p => civilFromIso(p.date) < startWeek).reverse(); // oldest first
  const blipSize = 8;
  const blipGap = 3;
  const backlogWidth = backlog.length
    ? backlog.length * (blipSize + blipGap) - blipGap + gap * 2 + 1 + gap // blips, divider line, gap
    : 0;

  const width = backlogWidth + WEEKS * (cellSize + gap) - gap + pad * 2;
  const height = cellSize + pad * 2;
  svg.setAttribute('viewBox', `0 0 ${width} ${height}`);
  svg.setAttribute('width', width);
  svg.setAttribute('height', height);

  const parts = [];
  backlog.forEach((p, i) => {
    const x = pad + i * (blipSize + blipGap);
    const y = pad + (cellSize - blipSize) / 2;
    parts.push(
      `<rect x="${x}" y="${y}" width="${blipSize}" height="${blipSize}" rx="2" fill="rgba(110,225,140,0.45)"><title>${esc(p.date)} — ${esc(p.title)}</title></rect>`
    );
  });
  if (backlog.length) {
    const lx = pad + backlog.length * (blipSize + blipGap) - blipGap + gap * 2;
    parts.push(
      `<line x1="${lx}" y1="${pad}" x2="${lx}" y2="${pad + cellSize}" stroke="rgba(255,255,255,0.25)" stroke-width="1"><title>goal starts ${GOAL_START}</title></line>`
    );
  }

  for (let w = 0; w < WEEKS; w++) {
    // Chronological: w = 0 is the goal's first week (leftmost), end of 2026 at the right.
    const weekStart = addDays(startWeek, w * 7);
    const count = countPostsInWeek(heatmap, weekStart, today);
    const isCurrent = weekStart.getTime() === thisWeek.getTime();
    const x = pad + backlogWidth + w * (cellSize + gap);
    const outline = isCurrent ? ' stroke="rgba(255,255,255,0.7)" stroke-width="1.5"' : '';
    const label = `week of ${isoDay(weekStart)} — ${count}/${WEEKLY_GOAL}${count >= WEEKLY_GOAL ? ' ✓' : ''}`;
    parts.push(
      `<rect x="${x}" y="${pad}" width="${cellSize}" height="${cellSize}" rx="3" fill="${colorForWeek(count)}"${outline}><title>${label}</title></rect>`
    );
  }
  svg.innerHTML = parts.join('');
}

export function renderTracker(posts, heatmap) {
  renderStats(heatmap);
  renderHeatmap(posts, heatmap);
}
