function luma([r, g, b]) {
  return 0.299 * r + 0.587 * g + 0.114 * b;
}

export function extractPalette(imgUrl, k = 4) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.crossOrigin = 'anonymous';
    img.onload = () => {
      const size = 48;
      const canvas = document.createElement('canvas');
      canvas.width = size;
      canvas.height = size;
      const ctx = canvas.getContext('2d');
      ctx.drawImage(img, 0, 0, size, size);
      const { data } = ctx.getImageData(0, 0, size, size);

      const buckets = new Map();
      for (let i = 0; i < data.length; i += 4) {
        if (data[i + 3] < 128) continue;
        const r = data[i] & 0xf0;
        const g = data[i + 1] & 0xf0;
        const b = data[i + 2] & 0xf0;
        const key = (r << 16) | (g << 8) | b;
        buckets.set(key, (buckets.get(key) ?? 0) + 1);
      }

      const sorted = [...buckets.entries()].sort((a, b) => b[1] - a[1]);
      const picked = [];
      const minDist = 55;
      for (const [key] of sorted) {
        const r = (key >> 16) & 0xff;
        const g = (key >> 8) & 0xff;
        const b = key & 0xff;
        const tooClose = picked.some(p => {
          const dr = p[0] * 255 - r;
          const dg = p[1] * 255 - g;
          const db = p[2] * 255 - b;
          return Math.sqrt(dr * dr + dg * dg + db * db) < minDist;
        });
        if (!tooClose) {
          picked.push([r / 255, g / 255, b / 255]);
          if (picked.length >= k) break;
        }
      }
      let i = 0;
      while (picked.length < k && i < sorted.length) {
        const [key] = sorted[i++];
        const r = (key >> 16) & 0xff;
        const g = (key >> 8) & 0xff;
        const b = key & 0xff;
        picked.push([r / 255, g / 255, b / 255]);
      }
      picked.sort((a, b) => luma(a) - luma(b));
      resolve(picked);
    };
    img.onerror = reject;
    img.src = imgUrl;
  });
}

export function dispatchPalette(palette) {
  window.dispatchEvent(new CustomEvent('palette', { detail: { palette } }));
}
