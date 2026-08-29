// / — curated landing page: the demo as a full-bleed hero, then the featured posts.
import posts from '../data/posts.json';
import demos from '../data/demos.json';
import { renderPostList, renderShowcase, coverUrlFor } from './posts-ui.js';

const featured = posts.filter(p => p.featured);

// Lead with the first demo that has a cover; fall back to a featured post if there are none.
const demo = demos.find(d => coverUrlFor(d));
const hero = demo ?? featured.find(p => coverUrlFor(p));

if (hero) {
  renderShowcase(document.getElementById('showcase'), hero, {
    caption: demo ? `${hero.title} · live demo` : hero.title,
  });
}

renderPostList(document.getElementById('posts'), featured.filter(p => p !== hero), {
  emptyText: 'no featured posts — add `featured: true` to a post',
});
