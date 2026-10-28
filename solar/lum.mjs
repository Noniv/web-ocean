// Mean / max luminance of a PNG, for checking exposure without opening an image viewer.
//   node lum.mjs shots/check.png [...]
import fs from 'node:fs';
import zlib from 'node:zlib';

function decode(file) {
  const d = fs.readFileSync(file);
  let i = 8, w = 0, h = 0, ct = 0, idat = [];
  while (i < d.length) {
    const len = d.readUInt32BE(i);
    const typ = d.toString('ascii', i + 4, i + 8);
    if (typ === 'IHDR') { w = d.readUInt32BE(i + 8); h = d.readUInt32BE(i + 12); ct = d[i + 17]; }
    if (typ === 'IDAT') idat.push(d.subarray(i + 8, i + 8 + len));
    i += 12 + len;
  }
  const raw = zlib.inflateSync(Buffer.concat(idat));
  const bpp = ct === 6 ? 4 : 3;
  const stride = w * bpp;
  const out = Buffer.alloc(h * stride);
  let pos = 0;
  for (let y = 0; y < h; y++) {
    const f = raw[pos++];
    const line = raw.subarray(pos, pos + stride); pos += stride;
    const cur = out.subarray(y * stride, (y + 1) * stride);
    const prev = y ? out.subarray((y - 1) * stride, y * stride) : Buffer.alloc(stride);
    for (let x = 0; x < stride; x++) {
      const a = x >= bpp ? cur[x - bpp] : 0;
      const b = prev[x];
      const c = x >= bpp ? prev[x - bpp] : 0;
      let v = line[x];
      if (f === 1) v += a;
      else if (f === 2) v += b;
      else if (f === 3) v += (a + b) >> 1;
      else if (f === 4) {
        const p = a + b - c, pa = Math.abs(p - a), pb = Math.abs(p - b), pc = Math.abs(p - c);
        v += (pa <= pb && pa <= pc) ? a : (pb <= pc ? b : c);
      }
      cur[x] = v & 255;
    }
  }
  return { w, h, bpp, px: out, stride };
}

for (const f of process.argv.slice(2)) {
  const { w, h, bpp, px, stride } = decode(f);
  let sum = 0, mx = 0, sat = 0, n = 0;
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const o = y * stride + x * bpp;
      const l = 0.2126 * px[o] + 0.7152 * px[o + 1] + 0.0722 * px[o + 2];
      sum += l; if (l > mx) mx = l; if (l > 250) sat++; n++;
    }
  }
  // High-frequency energy: mean |difference| between horizontally adjacent pixels, over
  // pixels bright enough to be scene rather than background. Aliasing shows up here, not
  // in the mean -- a prefiltered image and an aliased one can have identical averages.
  let hf = 0, hn = 0;
  for (let y = 0; y < h; y++) {
    for (let x = 1; x < w; x++) {
      const o = y * stride + x * bpp, q = o - bpp;
      const a = 0.2126 * px[o] + 0.7152 * px[o + 1] + 0.0722 * px[o + 2];
      const b = 0.2126 * px[q] + 0.7152 * px[q + 1] + 0.0722 * px[q + 2];
      if (a > 8 || b > 8) { hf += Math.abs(a - b); hn++; }
    }
  }
  console.log(`${f}  ${w}x${h}  mean ${(sum / n).toFixed(1)}  max ${mx.toFixed(0)}  ` +
    `sat ${(100 * sat / n).toFixed(1)}%  highfreq ${(hf / Math.max(hn, 1)).toFixed(3)} over ${hn}px`);
}
