import { mkdirSync, writeFileSync, existsSync } from 'node:fs';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = dirname(fileURLToPath(import.meta.url));
const POSTS_DIR = join(__dirname, '..', 'src', 'posts');

const title = process.argv.slice(2).join(' ').trim();
if (!title) {
  console.error('Usage: npm run new "Post Title"');
  process.exit(1);
}

const today = new Intl.DateTimeFormat('en-CA', {
  timeZone: 'Europe/Amsterdam',
  year: 'numeric', month: '2-digit', day: '2-digit',
}).format(new Date());
const slugBase = title
  .toLowerCase()
  .replace(/[^\w\s-]/g, '')
  .replace(/\s+/g, '-')
  .replace(/-+/g, '-')
  .replace(/^-|-$/g, '');

const slug = slugBase ? `${today}-${slugBase}` : `${today}-post`;
const dir = join(POSTS_DIR, slug);

if (existsSync(dir)) {
  console.error(`Post already exists: ${slug}`);
  process.exit(1);
}

mkdirSync(dir, { recursive: true });

// JSON.stringify yields a valid YAML double-quoted scalar — safe for titles
// containing colons, quotes, '#', leading '[' / '*' etc.
const frontmatter = `---
title: ${JSON.stringify(title)}
date: ${today}
cover:
tags: []
---

`;

writeFileSync(join(dir, 'index.md'), frontmatter);
console.log(`Created src/posts/${slug}/index.md`);
