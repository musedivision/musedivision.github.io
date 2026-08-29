// /blog/ — the habit tracker, a random showcase, and every post.
import posts from '../data/posts.json';
import heatmap from '../data/heatmap.json';
import { renderTracker } from './tracker.js';
import { renderPostList, renderShowcase, coverUrlFor } from './posts-ui.js';

renderTracker(posts, heatmap);

const withCovers = posts.filter(p => coverUrlFor(p));
if (withCovers.length) {
  const pick = withCovers[Math.floor(Math.random() * withCovers.length)];
  renderShowcase(document.getElementById('showcase'), pick);
}

renderPostList(document.getElementById('posts'), posts);
