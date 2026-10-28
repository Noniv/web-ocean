// Minimal static dev server (no dependencies).
//   node server.mjs [port]
// Serves this folder. POST /__shot?name=foo.png stores a PNG (sent as raw body)
// into ./shots — used for automated visual checks.
import http from 'node:http';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const root = path.dirname(fileURLToPath(import.meta.url));
const port = Number(process.argv[2] || process.env.PORT || 8080);
const types = {
  '.html': 'text/html; charset=utf-8',
  '.js': 'text/javascript; charset=utf-8',
  '.mjs': 'text/javascript; charset=utf-8',
  '.wgsl': 'text/plain; charset=utf-8',
  '.css': 'text/css; charset=utf-8',
  '.png': 'image/png',
  '.json': 'application/json',
};

http.createServer((req, res) => {
  const url = new URL(req.url, 'http://localhost');
  if (req.method === 'POST' && url.pathname === '/__shot') {
    const name = path.basename(url.searchParams.get('name') || 'shot.png');
    const chunks = [];
    req.on('data', (c) => chunks.push(c));
    req.on('end', () => {
      const dir = path.join(root, 'shots');
      fs.mkdirSync(dir, { recursive: true });
      fs.writeFileSync(path.join(dir, name), Buffer.concat(chunks));
      res.writeHead(200, { 'content-type': 'text/plain' });
      res.end('ok');
      console.log('saved shot', name);
    });
    return;
  }
  if (req.method === 'POST' && url.pathname === '/__log') {
    const chunks = [];
    req.on('data', (c) => chunks.push(c));
    req.on('end', () => {
      console.log('[client]', Buffer.concat(chunks).toString());
      res.end('ok');
    });
    return;
  }
  let p = decodeURIComponent(url.pathname);
  if (p.endsWith('/')) p += 'index.html';
  const file = path.join(root, p);
  if (!file.startsWith(root)) { res.writeHead(403); res.end(); return; }
  fs.readFile(file, (err, data) => {
    if (err) { res.writeHead(404); res.end('not found'); return; }
    res.writeHead(200, {
      'content-type': types[path.extname(file)] || 'application/octet-stream',
      'cache-control': 'no-store',
    });
    res.end(data);
  });
}).listen(port, () => console.log(`solar dev server: http://localhost:${port}/`));
