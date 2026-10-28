// Headless render check.
//   node check.mjs [--sw] [query...]
//
// Starts the dev server, points headless Chrome at the page, and waits for it to render a
// frame and POST the PNG back. The page forwards every WGSL compilation message and any
// fatal error to /__log, so a shader that does not compile shows up here as a file and line
// rather than as a blank canvas. --sw forces the software adapter, which still runs the real
// shader compiler and so still validates every shader.
//
// This is the loop the ocean renderer used to check its own work, reused.
import { spawn } from 'node:child_process';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const root = path.dirname(fileURLToPath(import.meta.url));
const args = process.argv.slice(2);
const sw = args.includes('--sw');
const extra = args.filter((a) => !a.startsWith('--'));
const pageArg = args.find((a) => a.startsWith('--page='));
const page = pageArg ? pageArg.slice(7) : '';
const PORT = 8123 + (sw ? 1 : 0);
const shotArg = args.find((a) => a.startsWith('--out='));
const shot = shotArg ? shotArg.slice(6) : 'check.png';
const query = extra.length ? extra.join('&') : 'w=480&h=270&frames=8&ui=0';
const url = `http://localhost:${PORT}/${page}?${query}&shot=${shot}`;

const CHROME = [
  'C:/Program Files/Google/Chrome/Application/chrome.exe',
  'C:/Program Files (x86)/Google/Chrome/Application/chrome.exe',
  process.env.CHROME_PATH,
].filter(Boolean).find((p) => fs.existsSync(p));
if (!CHROME) {
  console.error('Chrome not found; set CHROME_PATH');
  process.exit(2);
}

const profile = fs.mkdtempSync(path.join(os.tmpdir(), 'solar-chk-'));
const server = spawn(process.execPath, [path.join(root, 'server.mjs'), String(PORT)], {
  cwd: root, stdio: ['ignore', 'pipe', 'pipe'],
});

const logs = [];
let done = null;
const finish = (code, why) => { if (!done) { done = { code, why }; } };

function onLine(line) {
  if (!line.trim()) return;
  if (line.startsWith('[client]')) {
    const body = line.slice(8).trim();
    logs.push(body);
    if (/^FATAL|init failed/.test(body)) finish(1, 'page reported a fatal error');
  } else if (line.startsWith('saved shot')) {
    finish(0, 'frame rendered and saved');
  }
}
let buf = '';
for (const s of [server.stdout, server.stderr]) {
  s.setEncoding('utf8');
  s.on('data', (d) => {
    buf += d;
    const parts = buf.split('\n');
    buf = parts.pop();
    parts.forEach(onLine);
  });
}

await new Promise((r) => setTimeout(r, 400));

const flags = [
  '--headless=new', '--no-sandbox', '--disable-dev-shm-usage', '--mute-audio',
  '--disable-extensions', '--no-first-run', '--no-default-browser-check',
  `--user-data-dir=${profile}`, '--window-size=600,400',
  '--enable-unsafe-webgpu',
];
if (sw) flags.push('--enable-unsafe-swiftshader', '--use-angle=swiftshader');
const chrome = spawn(CHROME, [...flags, url], { stdio: ['ignore', 'pipe', 'pipe'] });
chrome.stdout.setEncoding('utf8');
chrome.stderr.setEncoding('utf8');
let chromeErr = '';
chrome.stderr.on('data', (d) => { chromeErr += d; });

const TIMEOUT = sw ? 420000 : 120000;
const t0 = Date.now();
while (!done && Date.now() - t0 < TIMEOUT) {
  await new Promise((r) => setTimeout(r, 250));
}
if (!done) finish(1, `timed out after ${(TIMEOUT / 1000) | 0}s with no frame`);

try { chrome.kill('SIGKILL'); } catch { /* already gone */ }
try { server.kill('SIGKILL'); } catch { /* already gone */ }

console.log(`--- client log (${logs.length} line${logs.length === 1 ? '' : 's'}) ---`);
for (const l of logs) console.log(l);
if (!logs.length) {
  const interesting = chromeErr.split('\n')
    .filter((l) => /gpu|webgpu|dawn|vulkan|error|fail/i.test(l)).slice(0, 12);
  if (interesting.length) {
    console.log('--- chrome stderr (filtered) ---');
    for (const l of interesting) console.log(l.trim());
  }
}
const png = path.join(root, 'shots', shot);
if (done.code === 0 && fs.existsSync(png)) {
  const { size } = fs.statSync(png);
  console.log(`\nOK  ${done.why}  (${png}, ${(size / 1024).toFixed(0)} KB, ${((Date.now() - t0) / 1000).toFixed(1)}s)`);
} else {
  console.log(`\nFAIL  ${done.why}`);
}
try { fs.rmSync(profile, { recursive: true, force: true }); } catch { /* locked */ }
process.exit(done.code);
