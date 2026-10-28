// WebGPU solar system -- custom renderer, no engine and no rendering library.
//
// Analytic ray/sphere bodies + LEAN-prefiltered surfaces + a statistically rendered ring
// system + physically based planetary atmospheres + energy-conserving star splats.
//
// Everything on the GPU is in megametres and camera-relative. Absolute positions and the
// camera live in f64 here on the CPU and are differenced before upload, which is the only
// reason a single f32 shader can hold metre-scale surface relief and 30 AU of space at once.
import {
  MM, AU, SURF_W, SURF_H, SURF_MIPS, RING_RES, RING_MIPS, MAX_BODIES, BODY_STRIDE,
  TRANS_W, TRANS_H, MS_SIZE, MAX_STARS, K_ROCKY, K_GAS, K_ICE, K_SHROUD,
} from './wgsl/common.js';
import { SHADERS } from './wgsl/all.mjs';

const qs = new URLSearchParams(location.search);
const DBG = Number(qs.get('dbg')) || 0;   // 1 = raw albedo, 2 = LEAN slope sigma
const D2R = Math.PI / 180;
const GU = GPUBufferUsage;
const TU = GPUTextureUsage;
const J2000 = 2451545.0;

// ----------------------------------------------------------------------------------------
// Tunable parameters (the UI panel edits these)
// ----------------------------------------------------------------------------------------
export const P = {
  timeScale: 600,      // simulated seconds per real second
  exposure: 1.0,       // multiplies the automatic distance-based exposure
  autoExposure: 1,
  bloom: 0.055,
  starGain: 1.0,
  milkyWay: 0.045,
  zodiacal: 0.5,
  ringOpacity: 1.0,
  ringBright: 1.0,
  ringAlbedo: 0.88,   // particle single-scattering albedo
  ringBack: 0.42,      // backscatter asymmetry
  ringFwd: 0.60,
  opposition: 0.55,
  atmosphere: 1,
  fov: 48,
};

// ----------------------------------------------------------------------------------------
// Ephemeris data
// ----------------------------------------------------------------------------------------
// Keplerian elements: a(AU) e I(deg) L(deg) longPeri(deg) longNode(deg), then the same six
// as rates per Julian century (JPL's approximate-positions table).
const KEP = {
  Mercury: [0.38709927, 0.20563593, 7.00497902, 252.25032350, 77.45779628, 48.33076593,
    0.00000037, 0.00001906, -0.00594749, 149472.67411175, 0.16047689, -0.12534081],
  Venus: [0.72333566, 0.00677672, 3.39467605, 181.97909950, 131.60246718, 76.67984255,
    0.00000390, -0.00004107, -0.00078890, 58517.81538729, 0.00268329, -0.27769418],
  Earth: [1.00000261, 0.01671123, -0.00001531, 100.46457166, 102.93768193, 0.0,
    0.00000562, -0.00004392, -0.01294668, 35999.37244981, 0.32327364, 0.0],
  Mars: [1.52371034, 0.09339410, 1.84969142, -4.55343205, -23.94362959, 49.55953891,
    0.00001847, 0.00007882, -0.00813131, 19140.30268499, 0.44441088, -0.29257343],
  Jupiter: [5.20288700, 0.04838624, 1.30439695, 34.39644051, 14.72847983, 100.47390909,
    -0.00011607, -0.00013253, -0.00183714, 3034.74612775, 0.21252668, 0.20469106],
  Saturn: [9.53667594, 0.05386179, 2.48599187, 49.95424423, 92.59887831, 113.66242448,
    -0.00125060, -0.00050991, 0.00193609, 1222.49362201, -0.41897216, -0.28867794],
  Uranus: [19.18916464, 0.04725744, 0.77263783, 313.23810451, 170.95427630, 74.01692503,
    -0.00196176, -0.00004397, -0.00242939, 428.48202785, 0.40805281, 0.04240589],
  Neptune: [30.06992276, 0.00859048, 1.77004347, -55.12002969, 44.96476227, 131.78422574,
    0.00026291, 0.00005105, 0.00035372, 218.45945325, -0.32241464, -0.00508664],
  Pluto: [39.48211675, 0.24882730, 17.14001206, 238.92903833, 224.06891629, 110.30393684,
    -0.00031596, 0.00005170, 0.00004818, 145.20780515, -0.04062942, -0.01183482],
};

// Atmosphere recipes. Lengths in Mm, coefficients per Mm.
// thick Hr Hm | rayleigh rgb | mie rgb | mieAbs mieG | absorb rgb | ozone groundAlb
const ATMOS = {
  Earth: { thick: 0.100, Hr: 0.0080, Hm: 0.0012, ray: [5.802, 13.558, 33.1], mie: [3.996, 3.996, 3.996],
    mieAbs: 0.404, g: 0.80, abs: [0.650, 1.881, 0.085], ozone: 1, ground: 0.10,
    shell: [0.175, 0.410, 1.000], dens: 1.5, haze: 0.25 },
  Mars: { thick: 0.090, Hr: 0.0113, Hm: 0.0110, ray: [0.347, 0.810, 1.978], mie: [15.0, 16.5, 18.0],
    mieAbs: 0.8, g: 0.68, abs: [1.0, 6.0, 19.0], ozone: 0, ground: 0.25,
    shell: [1.000, 0.720, 0.450], dens: 0.9, haze: 0.85 },
  Titan: { thick: 0.600, Hr: 0.0400, Hm: 0.0550, ray: [1.2, 2.8, 6.8], mie: [14.0, 11.0, 7.0],
    mieAbs: 1.2, g: 0.40, abs: [1.0, 4.0, 14.0], ozone: 0, ground: 0.20,
    shell: [1.000, 0.620, 0.280], dens: 4.0, haze: 1.0 },
  Venus: { thick: 0.090, Hr: 0.0157, Hm: 0.0050, ray: [2.6, 6.0, 14.0], mie: [60.0, 58.0, 52.0],
    mieAbs: 1.0, g: 0.75, abs: [0.3, 0.8, 3.5], ozone: 0, ground: 0.75,
    shell: [1.000, 0.940, 0.780], dens: 6.0, haze: 1.0 },
  Jupiter: { thick: 0.400, Hr: 0.0270, Hm: 0.0200, ray: [1.1, 2.5, 6.2], mie: [6.0, 5.5, 5.0],
    mieAbs: 0.5, g: 0.60, abs: [0.2, 0.5, 1.5], ozone: 0, ground: 0.50,
    shell: [0.350, 0.500, 1.000], dens: 0.5, haze: 0.5 },
  Saturn: { thick: 0.500, Hr: 0.0590, Hm: 0.0350, ray: [0.9, 2.1, 5.2], mie: [5.0, 4.6, 4.0],
    mieAbs: 0.4, g: 0.60, abs: [0.3, 0.7, 1.8], ozone: 0, ground: 0.50,
    shell: [0.600, 0.620, 1.000], dens: 0.4, haze: 0.5 },
};

// req/rpo in km, pole as [RA, Dec] in degrees (IAU), rot = sidereal rotation period in hours
// (negative = retrograde), relief = peak-to-peak surface relief in km for the height map.
const SPEC = [
  { n: 'Sun', sun: 1, req: 696000, rpo: 696000 },
  { n: 'Mercury', kind: K_ROCKY, map: 0, relief: 16, req: 2439.7, rpo: 2439.7,
    pole: [281.01, 61.45], rot: 1407.6, tint: [1, 1, 1] },
  { n: 'Venus', kind: K_SHROUD, req: 6051.8, rpo: 6051.8,
    pole: [272.76, 67.16], rot: -5832.5, tint: [0.92, 0.855, 0.68], atmo: 'Venus' },
  { n: 'Earth', kind: K_ROCKY, map: 1, relief: 12, req: 6378.137, rpo: 6356.752,
    pole: [0, 90], rot: 23.9345, tint: [1, 1, 1], atmo: 'Earth', sea: 0.5, cloud: 0.45 },
  { n: 'Moon', parent: 'Earth', kind: K_ROCKY, map: 2, relief: 24, req: 1737.4, rpo: 1737.4,
    pole: [270, 66.54], rot: 655.72, tint: [1, 1, 1],
    orb: { a: 384400, e: 0.0549, inc: 5.145, per: 27.32166, frame: 'ecliptic' } },
  { n: 'Mars', kind: K_ROCKY, map: 3, relief: 25, req: 3396.2, rpo: 3376.2,
    pole: [317.68, 52.89], rot: 24.6229, tint: [1, 1, 1], atmo: 'Mars' },
  { n: 'Jupiter', kind: K_GAS, req: 71492, rpo: 66854,
    pole: [268.057, 64.496], rot: 9.925, tint: [1.00, 0.95, 0.88], atmo: 'Jupiter' },
  { n: 'Io', parent: 'Jupiter', kind: K_ROCKY, map: 4, relief: 10, req: 1821.6, rpo: 1821.6,
    pole: [268.057, 64.496], rot: 42.456, tint: [1, 1, 1],
    orb: { a: 421700, e: 0.0041, inc: 0.05, per: 1.769138, frame: 'equator' } },
  { n: 'Europa', parent: 'Jupiter', kind: K_ROCKY, map: 5, relief: 1.2, req: 1560.8, rpo: 1560.8,
    pole: [268.057, 64.496], rot: 85.228, tint: [1, 1, 1],
    orb: { a: 671034, e: 0.0094, inc: 0.47, per: 3.551181, frame: 'equator' } },
  { n: 'Ganymede', parent: 'Jupiter', kind: K_ROCKY, map: 6, relief: 10, req: 2634.1, rpo: 2634.1,
    pole: [268.057, 64.496], rot: 171.71, tint: [1, 1, 1],
    orb: { a: 1070412, e: 0.0013, inc: 0.20, per: 7.154553, frame: 'equator' } },
  { n: 'Callisto', parent: 'Jupiter', kind: K_ROCKY, map: 7, relief: 20, req: 2410.3, rpo: 2410.3,
    pole: [268.057, 64.496], rot: 400.54, tint: [1, 1, 1],
    orb: { a: 1882709, e: 0.0074, inc: 0.19, per: 16.689017, frame: 'equator' } },
  { n: 'Saturn', kind: K_GAS, req: 60268, rpo: 54364, rings: 1,
    pole: [40.589, 83.537], rot: 10.656, tint: [1.02, 0.945, 0.78], atmo: 'Saturn' },
  { n: 'Titan', parent: 'Saturn', kind: K_SHROUD, req: 2574.7, rpo: 2574.7,
    pole: [40.589, 83.537], rot: 382.68, tint: [0.80, 0.55, 0.26], atmo: 'Titan',
    orb: { a: 1221870, e: 0.0288, inc: 0.35, per: 15.945, frame: 'equator' } },
  { n: 'Uranus', kind: K_ICE, req: 25559, rpo: 24973,
    pole: [257.311, -15.175], rot: -17.24, tint: [0.62, 0.84, 0.86] },
  { n: 'Neptune', kind: K_ICE, req: 24764, rpo: 24341,
    pole: [299.36, 43.46], rot: 16.11, tint: [0.33, 0.48, 0.80] },
  { n: 'Pluto', kind: K_ROCKY, map: 8, relief: 12, req: 1188.3, rpo: 1188.3,
    pole: [132.993, -6.163], rot: -153.29, tint: [1, 1, 1] },
];

// Bright stars: name, RA(hours), Dec(deg), V magnitude, B-V. Enough of them that the
// familiar asterisms -- Orion, the Plough, Crux, Cassiopeia -- are actually recognisable.
const BRIGHT = `Sirius 6.752 -16.716 -1.46 0.00|Canopus 6.399 -52.696 -0.74 0.15
Rigil 14.660 -60.834 -0.27 0.71|Arcturus 14.261 19.182 -0.05 1.23|Vega 18.616 38.784 0.03 0.00
Capella 5.278 45.998 0.08 0.80|Rigel 5.242 -8.202 0.13 -0.03|Procyon 7.655 5.225 0.34 0.42
Achernar 1.629 -57.237 0.46 -0.16|Betelgeuse 5.919 7.407 0.50 1.85|Hadar 14.064 -60.373 0.61 -0.23
Altair 19.846 8.868 0.77 0.22|Acrux 12.443 -63.099 0.77 -0.24|Aldebaran 4.599 16.509 0.85 1.54
Antares 16.490 -26.432 0.96 1.83|Spica 13.420 -11.161 0.97 -0.23|Pollux 7.755 28.026 1.14 1.00
Fomalhaut 22.961 -29.622 1.16 0.09|Deneb 20.690 45.280 1.25 0.09|Mimosa 12.795 -59.689 1.25 -0.24
Regulus 10.139 11.967 1.35 -0.11|Adhara 6.977 -28.972 1.50 -0.21|Castor 7.577 31.888 1.58 0.03
Shaula 17.560 -37.104 1.62 -0.22|Gacrux 12.519 -57.113 1.63 1.59|Bellatrix 5.418 6.350 1.64 -0.22
Elnath 5.438 28.608 1.65 -0.13|Miaplacidus 9.220 -69.717 1.67 0.00|Alnilam 5.604 -1.202 1.69 -0.18
Alnair 22.137 -46.961 1.74 -0.13|Alnitak 5.679 -1.943 1.74 -0.21|Alioth 12.900 55.960 1.77 -0.02
Dubhe 11.062 61.751 1.79 1.07|Mirfak 3.405 49.861 1.79 0.48|Wezen 7.140 -26.393 1.83 0.67
KausAus 18.403 -34.385 1.85 -0.03|Sargas 17.622 -42.998 1.86 0.40|Avior 8.375 -59.510 1.86 1.20
Alkaid 13.792 49.313 1.86 -0.19|Menkalinan 5.995 44.947 1.90 0.08|Atria 16.811 -69.028 1.92 1.44
Alhena 6.629 16.399 1.93 0.00|Peacock 20.427 -56.735 1.94 -0.12|Polaris 2.530 89.264 1.98 0.60
Mirzam 6.378 -17.956 1.98 -0.24|Alphard 9.460 -8.659 1.99 1.44|Hamal 2.120 23.463 2.00 1.15
DenebKaitos 0.726 -17.987 2.04 1.02|Nunki 18.921 -26.297 2.05 -0.22|Saiph 5.796 -9.670 2.06 -0.17
Alpheratz 0.140 29.090 2.06 -0.04|Tiaki 22.711 -46.885 2.07 1.61|Algieba 10.333 19.841 2.08 1.13
Rasalhague 17.582 12.560 2.08 0.15|Kochab 14.845 74.156 2.08 1.47|Algol 3.136 40.956 2.09 -0.05
Almach 2.065 42.330 2.10 1.37|Denebola 11.818 14.572 2.14 0.09|Muhlifain 12.692 -48.960 2.20 -0.01
Naos 8.060 -40.003 2.21 -0.27|Aspidiske 9.285 -59.275 2.21 0.19|Alphecca 15.578 26.715 2.22 -0.02
Suhail 9.133 -43.433 2.23 1.67|Mizar 13.399 54.925 2.23 0.06|Sadr 20.370 40.257 2.23 0.67
Eltanin 17.944 51.489 2.23 1.52|Schedar 0.675 56.537 2.24 1.17|Mintaka 5.533 -0.299 2.25 -0.17
Caph 0.153 59.150 2.28 0.34|Dschubba 16.005 -22.622 2.29 -0.12|Larawag 16.836 -34.293 2.29 1.15
Izar 14.750 27.074 2.37 0.97|Merak 11.031 56.382 2.37 0.03|Enif 21.736 9.875 2.38 1.52
Girtab 17.708 -39.030 2.39 -0.19|Ankaa 0.438 -42.306 2.40 1.08|Scheat 23.063 28.083 2.42 1.65
Sabik 17.173 -15.725 2.43 0.06|Phecda 11.897 53.695 2.44 0.04|Aludra 7.401 -29.303 2.45 -0.08
Markeb 9.368 -55.011 2.47 -0.18|Navi 0.945 60.717 2.47 -0.15|Markab 23.079 15.205 2.49 -0.04
Aljanah 20.770 33.970 2.48 1.02|Acrab 16.091 -19.805 2.56 -0.07|Megrez 12.257 57.033 3.31 0.08
Alnasl 18.097 -30.424 2.98 1.00|Tarazed 19.771 10.613 2.72 1.52|Rastaban 17.507 52.301 2.79 0.95`;

const OBLIQ = 23.4392911 * D2R;

function fail(msg) {
  const el = document.getElementById('err');
  el.style.display = 'block';
  el.textContent = msg;
  log('FATAL ' + msg);
  throw new Error(msg);
}
function log(msg) {
  console.log(msg);
  try { fetch('/__log', { method: 'POST', body: String(msg) }).catch(() => {}); } catch (e) { /* file:// */ }
}

// ----------------------------------------------------------------------------------------
// Frames and small maths helpers
// ----------------------------------------------------------------------------------------
// Equatorial (RA/Dec) -> the renderer's frame: ecliptic with +Y = ecliptic north.
function raDecToVec(raDeg, decDeg) {
  const a = raDeg * D2R, d = decDeg * D2R;
  const x = Math.cos(d) * Math.cos(a), y = Math.cos(d) * Math.sin(a), z = Math.sin(d);
  const ye = y * Math.cos(OBLIQ) + z * Math.sin(OBLIQ);
  const ze = -y * Math.sin(OBLIQ) + z * Math.cos(OBLIQ);
  return [x, ze, -ye];
}
function norm3(v) {
  const l = Math.hypot(v[0], v[1], v[2]) || 1;
  return [v[0] / l, v[1] / l, v[2] / l];
}
function cross3(a, b) {
  return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
}
// Any unit vector perpendicular to n.
function perp(n) {
  const t = Math.abs(n[1]) < 0.9 ? [0, 1, 0] : [1, 0, 0];
  return norm3(cross3(n, t));
}
function wrapDeg(d) { return d - 360 * Math.floor((d + 180) / 360); }

function kepler(M, e) {
  let E = M + e * Math.sin(M);
  for (let i = 0; i < 6; i++) {
    const d = (E - e * Math.sin(E) - M) / (1 - e * Math.cos(E));
    E -= d;
    if (Math.abs(d) < 1e-13) break;
  }
  return E;
}

// Heliocentric position from the approximate elements, returned in Mm in the render frame.
function planetPos(k, T, out) {
  const e = k[1] + k[7] * T;
  const L = k[3] + k[9] * T, peri = k[4] + k[10] * T;
  orbitPoint(k, T, kepler(wrapDeg(L - peri) * D2R, e), out);
}
// The point on the orbit ellipse at eccentric anomaly E; planetPos solves Kepler for E,
// the orbit overlay just sweeps it.
function orbitPoint(k, T, E, out) {
  const a = k[0] + k[6] * T, e = k[1] + k[7] * T;
  const I = (k[2] + k[8] * T) * D2R;
  const peri = k[4] + k[10] * T, node = k[5] + k[11] * T;
  const w = (peri - node) * D2R, O = node * D2R;
  const xp = a * (Math.cos(E) - e);
  const yp = a * Math.sqrt(Math.max(1 - e * e, 0)) * Math.sin(E);
  const cw = Math.cos(w), sw = Math.sin(w), cO = Math.cos(O), sO = Math.sin(O);
  const ci = Math.cos(I), si = Math.sin(I);
  const xe = (cw * cO - sw * sO * ci) * xp + (-sw * cO - cw * sO * ci) * yp;
  const ye = (cw * sO + sw * cO * ci) * xp + (-sw * sO + cw * cO * ci) * yp;
  const ze = (sw * si) * xp + (cw * si) * yp;
  // AU -> Mm, and ecliptic -> render frame
  out[0] = xe * AU; out[1] = ze * AU; out[2] = -ye * AU;
}

// B-V colour index -> unit-luminance RGB, via Ballesteros' temperature fit and Planck.
function bvToRGB(bv) {
  const T = 4600 * (1 / (0.92 * bv + 1.7) + 1 / (0.92 * bv + 0.62));
  const planck = (nm) => {
    const l = nm * 1e-9;
    return 1 / (Math.pow(l, 5) * (Math.exp(1.4388e-2 / (l * T)) - 1));
  };
  const r = planck(610), g = planck(550), b = planck(460);
  const y = 0.2126 * r + 0.7152 * g + 0.0722 * b;
  return [r / y, g / y, b / y];
}

async function main() {
  if (!navigator.gpu) fail('WebGPU is not available in this browser.');
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
  if (!adapter) fail('No WebGPU adapter found.');
  const hasTS = adapter.features.has('timestamp-query');
  const device = await adapter.requestDevice({ requiredFeatures: hasTS ? ['timestamp-query'] : [] });
  const ai = adapter.info || {};
  log(`adapter: ${ai.vendor || '?'} ${ai.architecture || ''} ${ai.device || ''} ${ai.description || ''}`.trim());
  device.lost.then((i) => log('device lost: ' + i.message));
  device.onuncapturederror = (e) => log('GPU error: ' + e.error.message);

  const canvas = document.getElementById('c');
  const ctx = canvas.getContext('webgpu');
  const swapFormat = navigator.gpu.getPreferredCanvasFormat();
  ctx.configure({ device, format: swapFormat, alphaMode: 'opaque', usage: TU.RENDER_ATTACHMENT | TU.COPY_SRC });

  async function shader(code, label) {
    const m = device.createShaderModule({ code, label });
    const info = await m.getCompilationInfo();
    let bad = false;
    for (const msg of info.messages) {
      if (msg.type === 'error') bad = true;
      if (msg.type !== 'info') {
        const line = code.split('\n')[msg.lineNum - 1];
        log(`[${label}] ${msg.type} ${msg.lineNum}:${msg.linePos} ${msg.message}\n    ${line}`);
      }
    }
    if (bad) fail('Shader compile error in ' + label + ' (see console)');
    return m;
  }
  const mods = {};
  await Promise.all(Object.entries(SHADERS).map(async ([k, v]) => { mods[k] = await shader(v, k); }));
  const computePipe = (key) => device.createComputePipelineAsync({
    label: key, layout: 'auto', compute: { module: mods[key], entryPoint: 'main' },
  });

  const mkBuf = (size, usage, label) => device.createBuffer({ size, usage, label });
  const bg = (pipe, entries, label) => device.createBindGroup({
    label, layout: pipe.getBindGroupLayout(0),
    entries: entries.map((r, i) => ({ binding: i, resource: r })),
  });

  // --------------------------------------------------------------------------------------
  // Resources
  // --------------------------------------------------------------------------------------
  const frameUBO = mkBuf(320, GU.UNIFORM | GU.COPY_DST, 'frame');
  const bodyBuf = mkBuf(MAX_BODIES * BODY_STRIDE * 4, GU.STORAGE | GU.COPY_DST, 'bodies');
  const recUBO = mkBuf(256, GU.UNIFORM | GU.COPY_DST, 'surface recipes');

  const MAPPED = SPEC.filter((s) => s.map !== undefined).sort((a, b) => a.map - b.map);
  const LAYERS = Math.max(1, MAPPED.length);
  const mkArr = (fmt, mips) => device.createTexture({
    size: [SURF_W, SURF_H, LAYERS], format: fmt, mipLevelCount: mips,
    usage: TU.TEXTURE_BINDING | TU.STORAGE_BINDING,
  });
  const surfTex = mkArr('rgba16float', SURF_MIPS);
  const leanTex = mkArr('rgba16float', SURF_MIPS);
  const crossTex = mkArr('rgba16float', SURF_MIPS);
  const auxTex = mkArr('rg32uint', 1);        // raw f32 bits: height, gloss
  const lvlA = (t, m) => t.createView({ dimension: '2d-array', baseMipLevel: m, mipLevelCount: 1 });
  const fullA = (t) => t.createView({ dimension: '2d-array' });

  const ringTau = device.createTexture({
    label: 'ring tau', size: [RING_RES, 1], format: 'rgba16float', mipLevelCount: RING_MIPS,
    usage: TU.TEXTURE_BINDING | TU.STORAGE_BINDING,
  });
  const lvl2 = (t, m) => t.createView({ baseMipLevel: m, mipLevelCount: 1 });

  const transLUT = device.createTexture({
    label: 'transmittance', size: [TRANS_W, TRANS_H], format: 'rgba16float',
    usage: TU.TEXTURE_BINDING | TU.STORAGE_BINDING,
  });
  const msLUT = device.createTexture({
    label: 'multiscatter', size: [MS_SIZE, MS_SIZE], format: 'rgba16float',
    usage: TU.TEXTURE_BINDING | TU.STORAGE_BINDING,
  });

  const sampLin = device.createSampler({
    magFilter: 'linear', minFilter: 'linear', mipmapFilter: 'linear',
    addressModeU: 'repeat', addressModeV: 'clamp-to-edge',
  });
  const sampClamp = device.createSampler({
    magFilter: 'linear', minFilter: 'linear', mipmapFilter: 'linear',
    addressModeU: 'clamp-to-edge', addressModeV: 'clamp-to-edge',
  });
  const linSamp = device.createSampler({
    magFilter: 'linear', minFilter: 'linear',
    addressModeU: 'clamp-to-edge', addressModeV: 'clamp-to-edge',
  });

  const [surfGenPipe, surfLeanPipe, mipRGBAPipe, ringGenPipe, ringMipPipe,
    transPipe, msPipe] = await Promise.all([
    computePipe('surf gen'), computePipe('surf lean'), computePipe('mip rgba16'),
    computePipe('ring gen'), computePipe('ring mip'),
    computePipe('trans lut'), computePipe('ms lut'),
  ]);

  // --------------------------------------------------------------------------------------
  // Render pipelines
  // --------------------------------------------------------------------------------------
  const HDR = 'rgba16float';
  const ADD = { color: { srcFactor: 'one', dstFactor: 'one', operation: 'add' },
    alpha: { srcFactor: 'one', dstFactor: 'one', operation: 'add' } };
  // premultiplied over: the body pass outputs the radiance it adds and the opacity it removes
  const OVER = { color: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha', operation: 'add' },
    alpha: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha', operation: 'add' } };

  const fsPipe = (key, blend) => device.createRenderPipelineAsync({
    label: key, layout: 'auto',
    vertex: { module: mods[key], entryPoint: 'vs' },
    fragment: { module: mods[key], entryPoint: 'fs', targets: [{ format: HDR, blend }] },
    primitive: { topology: 'triangle-list' },
  });

  const [bgPipe, bodiesPipe, sunPipe, downPipe, compPipe] = await Promise.all([
    fsPipe('background', undefined),
    fsPipe('bodies', OVER),
    fsPipe('sun', ADD),
    device.createRenderPipelineAsync({
      label: 'bloom down', layout: 'auto',
      vertex: { module: mods['bloom down'], entryPoint: 'vs' },
      fragment: { module: mods['bloom down'], entryPoint: 'fs', targets: [{ format: HDR }] },
    }),
    device.createRenderPipelineAsync({
      label: 'composite', layout: 'auto',
      vertex: { module: mods.composite, entryPoint: 'vs' },
      fragment: { module: mods.composite, entryPoint: 'fs', targets: [{ format: swapFormat }] },
    }),
  ]);
  const upPipe = await device.createRenderPipelineAsync({
    label: 'bloom up', layout: 'auto',
    vertex: { module: mods['bloom up'], entryPoint: 'vs' },
    fragment: { module: mods['bloom up'], entryPoint: 'fs', targets: [{ format: HDR, blend: ADD }] },
  });
  const starPipe = await device.createRenderPipelineAsync({
    label: 'stars', layout: 'auto',
    vertex: {
      module: mods.stars, entryPoint: 'vs',
      buffers: [{
        arrayStride: 32, stepMode: 'instance',
        attributes: [{ shaderLocation: 0, offset: 0, format: 'float32x4' },
          { shaderLocation: 1, offset: 16, format: 'float32x4' }],
      }],
    },
    fragment: { module: mods.stars, entryPoint: 'fs', targets: [{ format: HDR, blend: ADD }] },
    primitive: { topology: 'triangle-list' },
  });

  const bgBG = bg(bgPipe, [{ buffer: frameUBO }], 'background');
  const starBG = bg(starPipe, [{ buffer: frameUBO }], 'stars');
  const sunBG = bg(sunPipe, [{ buffer: frameUBO }, { buffer: bodyBuf }], 'sun');
  const bodiesBG = bg(bodiesPipe, [{ buffer: frameUBO }, { buffer: bodyBuf },
    fullA(surfTex), fullA(leanTex), fullA(crossTex), sampLin,
    ringTau.createView(), transLUT.createView(), msLUT.createView(), sampClamp], 'bodies');
  const transBG = bg(transPipe, [{ buffer: frameUBO }, transLUT.createView()], 'trans');
  const msBG = bg(msPipe, [{ buffer: frameUBO }, sampClamp, transLUT.createView(),
    msLUT.createView()], 'ms');

  // --------------------------------------------------------------------------------------
  // One-off generation: surface maps + LEAN pyramid, and the ring profile
  // --------------------------------------------------------------------------------------
  {
    const rec = new Float32Array(64);
    MAPPED.forEach((s, i) => {
      rec[i * 4] = s.map;
      rec[i * 4 + 1] = 13.0 + i * 7.0;        // per-body noise seed
      rec[i * 4 + 2] = s.relief;              // km
      rec[i * 4 + 3] = s.req;                 // km, for the slope metric
    });
    device.queue.writeBuffer(recUBO, 0, rec);

    const genBG = bg(surfGenPipe, [{ buffer: recUBO }, lvlA(surfTex, 0), lvlA(auxTex, 0)], 'surf gen');
    const leanBG = bg(surfLeanPipe, [{ buffer: recUBO }, fullA(auxTex), lvlA(leanTex, 0),
      lvlA(crossTex, 0)], 'surf lean');

    const enc = device.createCommandEncoder();
    const cp = enc.beginComputePass();
    cp.setPipeline(surfGenPipe);
    cp.setBindGroup(0, genBG);
    cp.dispatchWorkgroups(SURF_W / 8, SURF_H / 8, LAYERS);
    cp.setPipeline(surfLeanPipe);
    cp.setBindGroup(0, leanBG);
    cp.dispatchWorkgroups(SURF_W / 8, SURF_H / 8, LAYERS);
    // mip the albedo/height map and both moment maps; box-averaging the second moments is
    // what converts sub-pixel relief into a slope distribution
    for (let m = 1; m < SURF_MIPS; m++) {
      const w = Math.max(1, SURF_W >> m), h = Math.max(1, SURF_H >> m);
      const gx = Math.ceil(w / 8), gy = Math.ceil(h / 8);
      cp.setPipeline(mipRGBAPipe);
      for (const t of [surfTex, leanTex, crossTex]) {
        cp.setBindGroup(0, bg(mipRGBAPipe, [lvlA(t, m - 1), lvlA(t, m)]));
        cp.dispatchWorkgroups(gx, gy, LAYERS);
      }
    }
    cp.setPipeline(ringGenPipe);
    cp.setBindGroup(0, bg(ringGenPipe, [lvl2(ringTau, 0)]));
    cp.dispatchWorkgroups(Math.ceil(RING_RES / 64));
    cp.setPipeline(ringMipPipe);
    for (let m = 1; m < RING_MIPS; m++) {
      const w = Math.max(1, RING_RES >> m);
      cp.setBindGroup(0, bg(ringMipPipe, [lvl2(ringTau, m - 1), lvl2(ringTau, m)]));
      cp.dispatchWorkgroups(Math.ceil(w / 64));
    }
    cp.end();
    device.queue.submit([enc.finish()]);
  }

  // --------------------------------------------------------------------------------------
  // Stars
  // --------------------------------------------------------------------------------------
  const starData = new Float32Array(MAX_STARS * 8);
  let nStars = 0;
  function addStar(dir, mag, rgb, sigma) {
    if (nStars >= MAX_STARS - MAX_BODIES) return;
    const o = nStars * 8;
    starData[o] = dir[0]; starData[o + 1] = dir[1]; starData[o + 2] = dir[2];
    starData[o + 3] = Math.pow(10, -0.4 * mag);
    starData[o + 4] = rgb[0]; starData[o + 5] = rgb[1]; starData[o + 6] = rgb[2];
    starData[o + 7] = sigma;
    nStars++;
  }
  for (const row of BRIGHT.split(/[|\n]/)) {
    const f = row.trim().split(/\s+/);
    if (f.length < 5) continue;
    const mag = Number(f[3]);
    addStar(raDecToVec(Number(f[1]) * 15, Number(f[2])), mag, bvToRGB(Number(f[4])),
      1.15 + Math.max(0, 1.2 - mag) * 0.35);
  }
  // Procedural fill. N(m) grows as ~10^(0.6m), and the fainter population concentrates
  // toward the galactic plane, which is what gives the Milky Way its granular edge.
  {
    let seed = 0x2545f491;
    const rnd = () => {
      seed ^= seed << 13; seed |= 0; seed ^= seed >>> 17; seed ^= seed << 5; seed |= 0;
      return ((seed >>> 0) % 1000003) / 1000003;
    };
    const GP = norm3([-0.8676, 0.4971, 0.0003]);
    const gx = perp(GP), gy = cross3(GP, gx);
    const want = MAX_STARS - MAX_BODIES - nStars;
    for (let i = 0; i < want; i++) {
      // magnitude from the cumulative distribution, 2.6 .. 8.2
      const u = rnd();
      const mag = 2.6 + Math.log10(1 + u * (Math.pow(10, 0.6 * 5.6) - 1)) / 0.6;
      // galactic latitude: a two-component model, thin disc plus halo
      const disc = rnd() < 0.62;
      const b = disc ? (rnd() + rnd() + rnd() - 1.5) * 0.30 : Math.asin(rnd() * 2 - 1);
      const l = rnd() * 2 * Math.PI;
      const cb = Math.cos(b), sb = Math.sin(b);
      const d = norm3([
        gx[0] * cb * Math.cos(l) + gy[0] * cb * Math.sin(l) + GP[0] * sb,
        gx[1] * cb * Math.cos(l) + gy[1] * cb * Math.sin(l) + GP[1] * sb,
        gx[2] * cb * Math.cos(l) + gy[2] * cb * Math.sin(l) + GP[2] * sb,
      ]);
      addStar(d, mag, bvToRGB(rnd() * 2.0 - 0.3), 1.05);
    }
  }
  const starBuf = mkBuf(MAX_STARS * 32, GU.VERTEX | GU.COPY_DST, 'stars');
  device.queue.writeBuffer(starBuf, 0, starData, 0, nStars * 8);
  const splatBase = nStars;

  // --------------------------------------------------------------------------------------
  // Screen-size dependent targets
  // --------------------------------------------------------------------------------------
  const BLOOM_LEVELS = 8;
  const compUBO = mkBuf(32, GU.UNIFORM | GU.COPY_DST, 'composite');
  const compData = new Float32Array(8);
  const bloomUBOs = [];
  for (let i = 0; i < BLOOM_LEVELS * 2; i++) bloomUBOs.push(mkBuf(16, GU.UNIFORM | GU.COPY_DST, 'bloom'));

  let W = 0, H = 0;
  let hdrTex, bloomTex;
  const mainPass = {
    colorAttachments: [{ view: null, clearValue: { r: 0, g: 0, b: 0, a: 1 }, loadOp: 'clear', storeOp: 'store' }],
  };
  const compPass = { colorAttachments: [{ view: null, loadOp: 'clear', storeOp: 'store', clearValue: { r: 0, g: 0, b: 0, a: 1 } }], timestampWrites: undefined };
  let downPasses = [], upPasses = [], downBGs = [], upBGs = [], bgComp = null;

  function resize() {
    const fw = Number(qs.get('w')) || 0, fh = Number(qs.get('h')) || 0;
    const dpr = window.devicePixelRatio || 1;
    const w = fw || Math.max(1, Math.round(canvas.clientWidth * dpr));
    const h = fh || Math.max(1, Math.round(canvas.clientHeight * dpr));
    if (w === W && h === H) return;
    W = w; H = h;
    canvas.width = W; canvas.height = H;
    for (const t of [hdrTex, bloomTex]) t && t.destroy();
    hdrTex = device.createTexture({ size: [W, H], format: HDR, usage: TU.RENDER_ATTACHMENT | TU.TEXTURE_BINDING });
    const bw = Math.max(1, W >> 1), bh = Math.max(1, H >> 1);
    bloomTex = device.createTexture({ size: [bw, bh], format: HDR, mipLevelCount: BLOOM_LEVELS, usage: TU.RENDER_ATTACHMENT | TU.TEXTURE_BINDING });
    const hdrView = hdrTex.createView();
    mainPass.colorAttachments[0].view = hdrView;
    const bv = [];
    for (let m = 0; m < BLOOM_LEVELS; m++) bv.push(bloomTex.createView({ baseMipLevel: m, mipLevelCount: 1 }));
    downPasses = []; upPasses = []; downBGs = []; upBGs = [];
    const tmp = new Float32Array(4);
    for (let m = 0; m < BLOOM_LEVELS; m++) {
      const srcW = m === 0 ? W : Math.max(1, bw >> (m - 1));
      const srcH = m === 0 ? H : Math.max(1, bh >> (m - 1));
      tmp[0] = 1 / srcW; tmp[1] = 1 / srcH; tmp[2] = m === 0 ? 1 : 0; tmp[3] = 0;
      device.queue.writeBuffer(bloomUBOs[m], 0, tmp);
      downBGs.push(bg(downPipe, [m === 0 ? hdrView : bv[m - 1], linSamp, { buffer: bloomUBOs[m] }]));
      downPasses.push({ colorAttachments: [{ view: bv[m], clearValue: { r: 0, g: 0, b: 0, a: 1 }, loadOp: 'clear', storeOp: 'store' }] });
    }
    for (let m = BLOOM_LEVELS - 1; m > 0; m--) {
      const sw = Math.max(1, bw >> m), sh = Math.max(1, bh >> m);
      tmp[0] = 1 / sw; tmp[1] = 1 / sh; tmp[2] = 1.0; tmp[3] = 1.0;
      const ubo = bloomUBOs[BLOOM_LEVELS + m];
      device.queue.writeBuffer(ubo, 0, tmp);
      upBGs.push(bg(upPipe, [bv[m], linSamp, { buffer: ubo }]));
      upPasses.push({ colorAttachments: [{ view: bv[m - 1], loadOp: 'load', storeOp: 'store' }] });
    }
    bgComp = bg(compPipe, [hdrView, bv[0], linSamp, { buffer: compUBO }]);
  }

  // --------------------------------------------------------------------------------------
  // GPU timing
  // --------------------------------------------------------------------------------------
  let tsQuery = null, tsResolve = null, tsRead = null, tsPending = false, gpuMs = 0;
  if (hasTS) {
    tsQuery = device.createQuerySet({ type: 'timestamp', count: 2 });
    tsResolve = mkBuf(16, GU.QUERY_RESOLVE | GU.COPY_SRC, 'ts resolve');
    tsRead = mkBuf(16, GU.MAP_READ | GU.COPY_DST, 'ts read');
  }
  const onTs = () => {
    const t = new BigUint64Array(tsRead.getMappedRange());
    const ms = Number(t[1] - t[0]) / 1e6;
    if (ms > 0 && ms < 1000) gpuMs = gpuMs ? gpuMs * 0.8 + ms * 0.2 : ms;
    tsRead.unmap();
    tsPending = false;
  };
  const onTsFail = () => { tsPending = false; };

  // --------------------------------------------------------------------------------------
  // Scene state
  // --------------------------------------------------------------------------------------
  const bodies = SPEC.map((s) => ({
    spec: s,
    pos: [0, 0, 0],         // absolute, Mm
    axis: [0, 1, 0],
    east: [1, 0, 0],
    req: s.req / 1000,      // Mm
    rpo: s.rpo / 1000,
    dist: 0, angR: 0, cover: 0, irr: 1,
    poleVec: s.pole ? norm3(raDecToVec(s.pole[0], s.pole[1])) : [0, 1, 0],
  }));
  const byName = Object.fromEntries(bodies.map((b) => [b.spec.n, b]));
  const SUN_B = byName.Sun;

  let jd = J2000 + (Number(qs.get('jd')) || 0);

  function updateEphemeris() {
    const T = (jd - J2000) / 36525;
    const tmp = [0, 0, 0];
    for (const b of bodies) {
      const s = b.spec;
      if (s.sun) { b.pos[0] = 0; b.pos[1] = 0; b.pos[2] = 0; continue; }
      if (s.parent) continue;
      planetPos(KEP[s.n], T, tmp);
      b.pos[0] = tmp[0]; b.pos[1] = tmp[1]; b.pos[2] = tmp[2];
    }
    // moons, after their parents
    for (const b of bodies) {
      const s = b.spec;
      if (!s.parent) continue;
      const pb = byName[s.parent];
      const o = s.orb;
      const days = jd - J2000;
      const M = (days / o.per) * 2 * Math.PI;
      const E = kepler(M - Math.floor(M / (2 * Math.PI)) * 2 * Math.PI, o.e);
      const a = o.a / 1000;   // km -> Mm
      const xp = a * (Math.cos(E) - o.e);
      const yp = a * Math.sqrt(Math.max(1 - o.e * o.e, 0)) * Math.sin(E);
      // orbit plane: the parent's equator for the giants' moons, the ecliptic for the Moon
      let n;
      if (o.frame === 'equator') n = pb.poleVec;
      else {
        const inc = o.inc * D2R;
        n = norm3([Math.sin(inc) * 0.6, Math.cos(inc), Math.sin(inc) * 0.8]);
      }
      const u = perp(n), v = cross3(n, u);
      b.pos[0] = pb.pos[0] + u[0] * xp + v[0] * yp;
      b.pos[1] = pb.pos[1] + u[1] * xp + v[1] * yp;
      b.pos[2] = pb.pos[2] + u[2] * xp + v[2] * yp;
    }
    // spin
    const hours = (jd - J2000) * 24;
    for (const b of bodies) {
      const s = b.spec;
      if (s.sun || !s.rot) { b.axis = [0, 1, 0]; b.east = [1, 0, 0]; continue; }
      b.axis = b.poleVec;
      const u = perp(b.axis), v = cross3(b.axis, u);
      const th = (hours / s.rot) * 2 * Math.PI;
      const c = Math.cos(th), si = Math.sin(th);
      b.east = norm3([u[0] * c + v[0] * si, u[1] * c + v[1] * si, u[2] * c + v[2] * si]);
    }
  }

  // --------------------------------------------------------------------------------------
  // Camera
  // --------------------------------------------------------------------------------------
  const cam = { x: 0, y: 0, z: 0, yaw: 0, pitch: 0, speed: 0, fovY: P.fov * D2R };
  let target = byName.Saturn;
  let nearest = SUN_B, nearestGap = 1e30;

  // The camera rides in the target body's frame. Earth moves at 30 km/s and the time scale
  // multiplies that, so an inertial camera is left behind within a couple of frames; what
  // anyone actually wants when they fly to a planet is to hold station on it.
  const follow = [0, 0, 0];
  let following = false;
  function trackTarget() {
    if (following) {
      cam.x += target.pos[0] - follow[0];
      cam.y += target.pos[1] - follow[1];
      cam.z += target.pos[2] - follow[2];
    }
    follow[0] = target.pos[0]; follow[1] = target.pos[1]; follow[2] = target.pos[2];
    following = true;
  }
  // switching target must not teleport the camera, only change what it holds station on
  function setTarget(b) {
    target = b;
    follow[0] = b.pos[0]; follow[1] = b.pos[1]; follow[2] = b.pos[2];
    following = true;
    hudTarget();
  }

  // Park the camera off one shoulder of a body, with the sun roughly to the side so the
  // terminator is visible, and far enough out that the whole disc fits the field of view.
  // `phase` is the angle between the sun and the camera as seen from the body: small puts
  // the sun behind you (a full, flat disc), large puts it behind the body (a thin crescent,
  // and for Saturn the spectacular backlit rings).
  function parkView(b, mult, phase) {
    const d = (mult || 4.2) * Math.max(b.req, 0.002);
    const toSun = norm3([-b.pos[0], -b.pos[1], -b.pos[2]]);
    const up = [0, 1, 0];
    const side = norm3(cross3(toSun, up));
    const ang = (phase === undefined ? 55 : phase) * D2R;
    const k = Math.cos(ang), m = Math.sin(ang);
    let off = norm3([toSun[0] * k + side[0] * m, toSun[1] * k + side[1] * m + 0.35,
      toSun[2] * k + side[2] * m]);
    if (b.spec.rings) {
      // for Saturn, come in well above the ring plane so the system opens up
      off = norm3([off[0] + b.axis[0] * 0.75, off[1] + b.axis[1] * 0.75, off[2] + b.axis[2] * 0.75]);
    }
    // the camera sits at b + off * d and looks back along -off
    return { dist: d, pitch: clampPitch(Math.asin(Math.max(-1, Math.min(1, -off[1])))),
      yaw: Math.atan2(-off[0], off[2]) };
  }
  function gotoBody(b, mult, phase) {
    updateEphemeris();
    const v = parkView(b, mult, phase);
    cam.yaw = goal.yaw = v.yaw;
    cam.pitch = goal.pitch = v.pitch;
    orbit.dist = goal.dist = v.dist;
    orbit.pan = [0, 0, 0];
    flight = null;
    cam.speed = 0;
    setTarget(b);
    placeOrbit();
  }

  // Two ways to move. 'orbit', the default, keeps the camera on a sphere around the target:
  // drag to rotate, wheel to zoom, right-drag to pan, double-click a body to fly to it.
  // 'fly' is the free camera, with mouse look and WASD. F switches between them.
  let mode = qs.get('nav') === 'fly' || qs.get('cam') ? 'fly' : 'orbit';
  // the orbit camera sits at target + pan - forward * dist, looking along forward
  const orbit = { dist: 1, pan: [0, 0, 0] };
  // drags and wheel steps move the goal and the camera eases toward it, so nothing snaps
  const goal = { yaw: 0, pitch: 0, dist: 1 };
  let flight = null;
  const clampPitch = (p) => Math.max(-1.5, Math.min(1.5, p));
  // zooming scales the height above the surface, not the distance to the centre, so the
  // same wheel step is as useful 20 km above the Moon as it is 20 AU out
  const minGap = (b) => b.req * 0.003 + 1e-4;
  function zoomBy(k) {
    const gap = Math.max(goal.dist - target.req, minGap(target));
    goal.dist = target.req + Math.max(minGap(target), Math.min(2e7, gap * k));
  }

  function camBasis() {
    const cy = Math.cos(cam.yaw), sy = Math.sin(cam.yaw);
    const cp = Math.cos(cam.pitch), sp = Math.sin(cam.pitch);
    const f = [cp * sy, sp, -cp * cy], r = [cy, 0, sy];
    return { f, r, u: cross3(r, f) };
  }
  function placeOrbit(focus) {
    const c = focus || [target.pos[0] + orbit.pan[0], target.pos[1] + orbit.pan[1],
      target.pos[2] + orbit.pan[2]];
    const { f } = camBasis();
    cam.x = c[0] - f[0] * orbit.dist;
    cam.y = c[1] - f[1] * orbit.dist;
    cam.z = c[2] - f[2] * orbit.dist;
  }

  // Animated trip to a body. The focus slides from where the camera is looking now to the
  // new body (both keep moving along their orbits), the distance is interpolated in log
  // space, and the camera pulls back mid-flight so a trip between planets reads as a trip.
  function flyTo(b, view) {
    updateEphemeris();
    let from;
    if (mode === 'fly') {
      // express the free camera as an orbit about a point straight ahead of it
      const D = Math.max(Math.hypot(b.pos[0] - cam.x, b.pos[1] - cam.y, b.pos[2] - cam.z), 1e-3);
      const { f } = camBasis();
      from = { body: target, logD: Math.log(D), yaw: cam.yaw, pitch: cam.pitch,
        pan: [cam.x + f[0] * D - target.pos[0], cam.y + f[1] * D - target.pos[1],
          cam.z + f[2] * D - target.pos[2]] };
      mode = 'orbit';
      hudMode();
      if (document.pointerLockElement === canvas) document.exitPointerLock();
    } else {
      from = { body: target, logD: Math.log(orbit.dist), yaw: cam.yaw, pitch: cam.pitch,
        pan: orbit.pan.slice() };
    }
    const to = view || parkView(b, b.spec.sun ? 40 : 4.2);
    const a = [from.body.pos[0] + from.pan[0], from.body.pos[1] + from.pan[1], from.body.pos[2] + from.pan[2]];
    const travel = Math.hypot(a[0] - b.pos[0], a[1] - b.pos[1], a[2] - b.pos[2]);
    const dur = Math.max(0.8, Math.min(3, 0.8 + 0.35 * Math.log10(1 + travel / to.dist)));
    // shortest way round, so a trip never spins the long way
    let dyaw = (to.yaw - from.yaw) % (2 * Math.PI);
    if (dyaw > Math.PI) dyaw -= 2 * Math.PI;
    if (dyaw < -Math.PI) dyaw += 2 * Math.PI;
    flight = { from, to, b, t: 0, dur, travel, dyaw };
    setTarget(b);
  }
  function stepFlight(dt) {
    const fl = flight;
    fl.t = Math.min(1, fl.t + dt / fl.dur);
    const s = fl.t * fl.t * (3 - 2 * fl.t);
    const A = fl.from.body.pos, p = fl.from.pan, B = fl.b.pos;
    const focus = [0, 1, 2].map((i) => (A[i] + p[i]) * (1 - s) + B[i] * s);
    const lA = fl.from.logD, lB = Math.log(fl.to.dist);
    const hump = Math.max(0, Math.log(fl.travel * 0.6 + 1e-9) - (lA + lB) / 2) * Math.sin(Math.PI * s);
    orbit.dist = Math.exp(lA + (lB - lA) * s + hump);
    cam.yaw = fl.from.yaw + fl.dyaw * s;
    cam.pitch = fl.from.pitch + (fl.to.pitch - fl.from.pitch) * s;
    placeOrbit(focus);
    if (fl.t >= 1) {
      goal.yaw = cam.yaw; goal.pitch = cam.pitch; goal.dist = orbit.dist;
      orbit.pan = [0, 0, 0];
      flight = null;
    }
  }

  // Look down on the system from above the ecliptic, far enough out that a sphere of radius
  // rAU around the Sun fits the field of view.
  let overview = 0;
  function systemView(rAU, instant) {
    // W and H are still 0 when this runs at startup, before the first resize
    const aspect = W && H ? W / H
      : (Number(qs.get('w')) || canvas.clientWidth) / (Number(qs.get('h')) || canvas.clientHeight || 1);
    const fit = Math.min(cam.fovY, 2 * Math.atan(Math.tan(cam.fovY * 0.5) * aspect)) * 0.5;
    const view = { dist: rAU * AU * 1.08 / Math.sin(fit), yaw: cam.yaw, pitch: -62 * D2R };
    if (!instant) { flyTo(SUN_B, view); return; }
    if (mode === 'fly') { mode = 'orbit'; hudMode(); }
    cam.yaw = goal.yaw = view.yaw;
    cam.pitch = goal.pitch = view.pitch;
    orbit.dist = goal.dist = view.dist;
    orbit.pan = [0, 0, 0];
    flight = null;
    setTarget(SUN_B);
    placeOrbit();
  }

  // nearest body under a screen point; small or distant bodies get a few pixels of slack
  function pickBody(px, py) {
    const rect = canvas.getBoundingClientRect();
    const nx = ((px - rect.left) / rect.width) * 2 - 1;
    const ny = 1 - ((py - rect.top) / rect.height) * 2;
    const tanY = Math.tan(cam.fovY * 0.5), tanX = tanY * rect.width / rect.height;
    const { f, r, u } = camBasis();
    const d = norm3([0, 1, 2].map((i) => f[i] + r[i] * nx * tanX + u[i] * ny * tanY));
    const slack = 12 * 2 * tanY / rect.height;
    let hit = null, hitD = Infinity, near = null, nearMiss = slack;
    for (const b of bodies) {
      const rx = b.pos[0] - cam.x, ry = b.pos[1] - cam.y, rz = b.pos[2] - cam.z;
      const dist = Math.hypot(rx, ry, rz);
      const along = (rx * d[0] + ry * d[1] + rz * d[2]) / dist;
      if (along <= 0) continue;
      const miss = Math.acos(Math.min(1, along)) - Math.asin(Math.min(1, b.req / dist));
      if (miss <= 0 && dist < hitD) { hit = b; hitD = dist; }
      if (miss > 0 && miss < nearMiss) { near = b; nearMiss = miss; }
    }
    return hit || near;
  }

  const keys = Object.create(null);
  const JUMP = ['Sun', 'Mercury', 'Venus', 'Earth', 'Mars', 'Jupiter', 'Saturn', 'Uranus', 'Neptune', 'Pluto'];
  addEventListener('keydown', (e) => {
    keys[e.code] = true;
    if (e.code === 'KeyH') togglePanel();
    if (e.code === 'KeyO') showOrbits = !showOrbits;
    // V alternates between the whole system and the inner planets
    if (e.code === 'KeyV') { overview = overview === 1 ? 2 : 1; systemView(overview === 1 ? 33 : 1.75); }
    if (e.code === 'KeyF') {
      if (mode === 'orbit') {
        // stay exactly where we are and fly on from here
        mode = 'fly';
        flight = null;
        setTarget(target);
        hudMode();
      } else {
        // orbit the target from where the free camera is, turning to face it
        const dx = target.pos[0] - cam.x, dy = target.pos[1] - cam.y, dz = target.pos[2] - cam.z;
        const D = Math.max(Math.hypot(dx, dy, dz), target.req + minGap(target));
        flyTo(target, { dist: D, pitch: clampPitch(Math.asin(dy / D)), yaw: Math.atan2(dx, -dz) });
      }
    }
    if (e.code === 'BracketLeft' || e.code === 'BracketRight') {
      const list = bodies.filter((b) => !b.spec.sun);
      let i = list.indexOf(target);
      if (i < 0) i = 0;
      i = (i + (e.code === 'BracketRight' ? 1 : list.length - 1)) % list.length;
      if (mode === 'orbit') flyTo(list[i]); else setTarget(list[i]);
    }
    if (e.code === 'KeyG') { if (mode === 'orbit') flyTo(target); else gotoBody(target); }
    const d = /^Digit([0-9])$/.exec(e.code);
    if (d) {
      const b = byName[JUMP[Number(d[1])]];
      if (b && mode === 'orbit') flyTo(b);
      else if (b) gotoBody(b, b.spec.sun ? 40 : 4.2);
    }
  });
  addEventListener('keyup', (e) => { keys[e.code] = false; });
  addEventListener('blur', () => { for (const k in keys) keys[k] = false; });
  canvas.addEventListener('click', () => {
    if (mode === 'fly' && document.pointerLockElement !== canvas) canvas.requestPointerLock();
  });
  canvas.addEventListener('dblclick', (e) => {
    if (mode !== 'orbit') return;
    const b = pickBody(e.clientX, e.clientY);
    if (b) flyTo(b);
  });
  let dragging = false;
  // orbit-mode drag: left rotates, right / middle / shift+left pans
  let drag = null;
  canvas.addEventListener('pointerdown', (e) => {
    if (mode === 'fly') { if (e.button === 2) dragging = true; return; }
    drag = { pan: e.button === 1 || e.button === 2 || e.shiftKey };
    canvas.setPointerCapture(e.pointerId);
    canvas.style.cursor = 'grabbing';
  });
  const endDrag = () => {
    dragging = false;
    drag = null;
    canvas.style.cursor = mode === 'orbit' ? 'grab' : '';
  };
  addEventListener('pointerup', endDrag);
  addEventListener('pointercancel', endDrag);
  canvas.addEventListener('contextmenu', (e) => e.preventDefault());
  canvas.addEventListener('pointermove', (e) => {
    if (!drag || mode !== 'orbit') return;
    if (flight) {
      // grabbing the view mid-trip ends the trip where it is
      goal.yaw = cam.yaw; goal.pitch = cam.pitch; goal.dist = orbit.dist;
      const B = target.pos, { f } = camBasis();
      orbit.pan = [cam.x + f[0] * orbit.dist - B[0], cam.y + f[1] * orbit.dist - B[1],
        cam.z + f[2] * orbit.dist - B[2]];
      flight = null;
    }
    if (drag.pan) {
      // move the focus so the point under the cursor stays under the cursor
      const { r, u } = camBasis();
      const s = orbit.dist * 2 * Math.tan(cam.fovY * 0.5) / canvas.getBoundingClientRect().height;
      for (let i = 0; i < 3; i++) orbit.pan[i] += (-r[i] * e.movementX + u[i] * e.movementY) * s;
    } else {
      goal.yaw += e.movementX * 0.005;
      goal.pitch = clampPitch(goal.pitch - e.movementY * 0.005);
    }
  });
  addEventListener('mousemove', (e) => {
    if (mode === 'fly' && (document.pointerLockElement === canvas || dragging)) {
      cam.yaw += e.movementX * 0.0022;
      cam.pitch = clampPitch(cam.pitch - e.movementY * 0.0022);
    }
  });
  let speedScale = 1;
  canvas.addEventListener('wheel', (e) => {
    e.preventDefault();
    if (mode === 'fly') {
      speedScale = Math.max(1e-4, Math.min(1e4, speedScale * (e.deltaY < 0 ? 1.25 : 1 / 1.25)));
      return;
    }
    if (flight) return;
    // deltaMode 1 is lines (Firefox), otherwise pixels; trackpads send many small steps
    zoomBy(Math.exp(Math.max(-3, Math.min(3, e.deltaY * (e.deltaMode === 1 ? 0.05 : 0.0015)))));
  }, { passive: false });

  function flyCamera(dt) {
    // Flight speed is tied to how far the nearest surface is, which is the only way a single
    // control feels right both 2 km above Europa and 20 AU out in the dark.
    const base = Math.min(Math.max(nearestGap, 1e-5), 3.0e5) * 0.55;
    const v = base * speedScale * dt * (keys.ShiftLeft || keys.ShiftRight ? 6 : 1);
    const cy = Math.cos(cam.yaw), sy = Math.sin(cam.yaw);
    const cp = Math.cos(cam.pitch), sp = Math.sin(cam.pitch);
    const fwd = (keys.KeyW ? 1 : 0) - (keys.KeyS ? 1 : 0);
    const str = (keys.KeyD ? 1 : 0) - (keys.KeyA ? 1 : 0);
    const up = (keys.KeyE || keys.Space ? 1 : 0) - (keys.KeyQ || keys.KeyC ? 1 : 0);
    cam.x += (fwd * cp * sy + str * cy) * v;
    cam.y += (fwd * sp + up) * v;
    cam.z += (-fwd * cp * cy + str * sy) * v;
  }

  function updateCamera(dt) {
    if (mode === 'orbit') {
      if (flight) stepFlight(dt);
      else {
        // keys: A/D and arrows orbit, W/S zoom, Shift is faster
        const fast = keys.ShiftLeft || keys.ShiftRight ? 3 : 1;
        const kx = (keys.KeyD || keys.ArrowRight ? 1 : 0) - (keys.KeyA || keys.ArrowLeft ? 1 : 0);
        const ky = (keys.ArrowUp ? 1 : 0) - (keys.ArrowDown ? 1 : 0);
        const kz = (keys.KeyS ? 1 : 0) - (keys.KeyW ? 1 : 0);
        goal.yaw -= kx * dt * 1.2 * fast;
        goal.pitch = clampPitch(goal.pitch - ky * dt * 1.2 * fast);
        if (kz) zoomBy(Math.exp(kz * dt * 1.5 * fast));
        const k = 1 - Math.exp(-dt * 12);
        cam.yaw += (goal.yaw - cam.yaw) * k;
        cam.pitch += (goal.pitch - cam.pitch) * k;
        orbit.dist = Math.exp(Math.log(orbit.dist) + (Math.log(goal.dist) - Math.log(orbit.dist)) * k);
        placeOrbit();
      }
    } else {
      flyCamera(dt);
    }
    // never fall inside a body
    for (const b of bodies) {
      const dx = cam.x - b.pos[0], dy = cam.y - b.pos[1], dz = cam.z - b.pos[2];
      const d = Math.hypot(dx, dy, dz);
      const rmin = b.req * 1.004 + (b.spec.atmo ? 0 : 0);
      if (d < rmin && d > 1e-12) {
        const k = rmin / d;
        cam.x = b.pos[0] + dx * k; cam.y = b.pos[1] + dy * k; cam.z = b.pos[2] + dz * k;
      }
    }
  }

  // --------------------------------------------------------------------------------------
  // Per-frame uniforms
  // --------------------------------------------------------------------------------------
  const frameF = new Float32Array(80);
  const bodyF = new Float32Array(MAX_BODIES * BODY_STRIDE);
  const splatF = new Float32Array(MAX_BODIES * 8);
  const proj = new Float32Array(16), view = new Float32Array(16), viewProj = new Float32Array(16);
  let activeIdx = -1, activeAtmo = null, needLUT = true;
  let nSplat = 0, autoExp = 1;

  function mat4Mul(out, a, b) {
    for (let c = 0; c < 4; c++) {
      const b0 = b[c * 4], b1 = b[c * 4 + 1], b2 = b[c * 4 + 2], b3 = b[c * 4 + 3];
      out[c * 4] = a[0] * b0 + a[4] * b1 + a[8] * b2 + a[12] * b3;
      out[c * 4 + 1] = a[1] * b0 + a[5] * b1 + a[9] * b2 + a[13] * b3;
      out[c * 4 + 2] = a[2] * b0 + a[6] * b1 + a[10] * b2 + a[14] * b3;
      out[c * 4 + 3] = a[3] * b0 + a[7] * b1 + a[11] * b2 + a[15] * b3;
    }
  }

  function writeScene(time) {
    const aspect = W / H;
    const tanY = Math.tan(cam.fovY * 0.5), tanX = tanY * aspect;
    const pxAng = 2 * tanY / H;             // angular size of one pixel, radians

    // ---- geometry per body, camera-relative ----
    nearest = SUN_B; nearestGap = 1e30;
    nSplat = 0;
    bodyF.fill(0);
    for (let i = 0; i < bodies.length; i++) {
      const b = bodies[i];
      const s = b.spec;
      const rx = b.pos[0] - cam.x, ry = b.pos[1] - cam.y, rz = b.pos[2] - cam.z;
      b.dist = Math.hypot(rx, ry, rz);
      b.angR = Math.asin(Math.min(1, b.req / Math.max(b.dist, b.req)));
      const gap = b.dist - b.req;
      if (!s.sun && gap < nearestGap) { nearestGap = gap; nearest = b; }
      // distance from the sun, for the inverse-square irradiance
      const dsun = Math.max(Math.hypot(b.pos[0], b.pos[1], b.pos[2]), 1e-6) / AU;
      b.irr = s.sun ? 0 : 1 / (dsun * dsun);

      // Hand-over between a ray-traced disc and a splatted point. Below ~0.8 px the sphere
      // test is pure aliasing, so the body becomes a Gaussian of the same total flux; the
      // cross-fade over one octave of angular size makes the switch invisible.
      const px = b.angR / pxAng;
      b.cover = s.sun ? 0 : Math.max(0, Math.min(1, (px - 0.8) / 0.8));
      if (!s.sun && b.cover < 1) {
        // phase angle as seen from the camera
        const ls = [-b.pos[0], -b.pos[1], -b.pos[2]];
        const ll = Math.hypot(ls[0], ls[1], ls[2]) || 1;
        const vv = Math.hypot(rx, ry, rz) || 1;
        const cosA = Math.max(-1, Math.min(1, -(ls[0] * rx + ls[1] * ry + ls[2] * rz) / (ll * vv)));
        const alpha = Math.acos(cosA);
        const ph = (Math.sin(alpha) + (Math.PI - alpha) * Math.cos(alpha)) / Math.PI;
        const albedo = s.kind === K_GAS ? 0.5 : s.kind === K_ICE ? 0.5 : s.kind === K_SHROUD ? 0.65 : 0.17;
        // Irradiance from a Lambertian sphere: albedo * E * (R/d)^2 * phase. The splat lays
        // down a unit-area Gaussian of radiance, so its total is flux * (pixel solid angle);
        // dividing by pxAng^2 is what makes the handover from a ray-traced disc to a point
        // conserve energy, and keeps it doing so at any resolution or field of view.
        const flux = albedo * b.irr * Math.PI * ph *
          (b.req / b.dist) * (b.req / b.dist) / (pxAng * pxAng);
        const o = nSplat * 8;
        splatF[o] = rx / vv; splatF[o + 1] = ry / vv; splatF[o + 2] = rz / vv;
        splatF[o + 3] = Math.max(flux * (1 - b.cover), 0);
        const t = s.tint || [1, 1, 1];
        const tl = 0.2126 * t[0] + 0.7152 * t[1] + 0.0722 * t[2] || 1;
        splatF[o + 4] = t[0] / tl; splatF[o + 5] = t[1] / tl; splatF[o + 6] = t[2] / tl;
        splatF[o + 7] = -0.95;   // negative: a body, not a star
        nSplat++;
      }

      if (s.sun) continue;
      const o = i * BODY_STRIDE;
      bodyF[o] = rx; bodyF[o + 1] = ry; bodyF[o + 2] = rz; bodyF[o + 3] = b.req;
      bodyF[o + 4] = b.axis[0]; bodyF[o + 5] = b.axis[1]; bodyF[o + 6] = b.axis[2]; bodyF[o + 7] = b.rpo;
      bodyF[o + 8] = b.east[0]; bodyF[o + 9] = b.east[1]; bodyF[o + 10] = b.east[2]; bodyF[o + 11] = s.kind;
      const t = s.tint || [1, 1, 1];
      bodyF[o + 12] = t[0]; bodyF[o + 13] = t[1]; bodyF[o + 14] = t[2];
      bodyF[o + 15] = s.map === undefined ? -1 : s.map;
      const A = s.atmo && P.atmosphere ? ATMOS[s.atmo] : null;
      if (A) {
        bodyF[o + 16] = A.thick; bodyF[o + 17] = A.dens; bodyF[o + 18] = A.haze;
        bodyF[o + 19] = i === activeIdx ? 1 : 0;
        bodyF[o + 32] = A.shell[0]; bodyF[o + 33] = A.shell[1]; bodyF[o + 34] = A.shell[2];
      }
      // base slope variance: regolith has an RMS slope around 25 deg, cloud decks are smooth
      bodyF[o + 20] = s.kind === K_ROCKY ? 0.20 : 0.004;
      bodyF[o + 21] = s.sea === undefined ? -1 : s.sea;
      bodyF[o + 22] = 0.86;
      bodyF[o + 23] = s.cloud || 0;
      bodyF[o + 24] = s.rings ? 1 : 0;
      bodyF[o + 25] = s.rings ? 1 : 0;
      bodyF[o + 28] = b.cover;
      bodyF[o + 29] = b.irr;
    }
    device.queue.writeBuffer(bodyBuf, 0, bodyF);
    if (nSplat) device.queue.writeBuffer(starBuf, splatBase * 32, splatF, 0, nSplat * 8);

    // ---- which atmosphere gets the full treatment ----
    let best = -1, bestK = 1e30;
    for (let i = 0; i < bodies.length; i++) {
      const s = bodies[i].spec;
      if (!s.atmo || !P.atmosphere) continue;
      const k = (bodies[i].dist - bodies[i].req) / Math.max(bodies[i].req, 1e-6);
      if (k < bestK && k < 40) { bestK = k; best = i; }
    }
    if (best !== activeIdx) {
      activeIdx = best;
      activeAtmo = best >= 0 ? ATMOS[bodies[best].spec.atmo] : null;
      needLUT = true;
    }

    // ---- matrices ----
    const near = 1e-6;
    proj.fill(0);
    proj[0] = 1 / tanX; proj[5] = 1 / tanY; proj[11] = -1; proj[14] = near;
    const cy = Math.cos(cam.yaw), sy = Math.sin(cam.yaw);
    const cp = Math.cos(cam.pitch), sp = Math.sin(cam.pitch);
    const fx = cp * sy, fy = sp, fz = -cp * cy;
    const rx2 = cy, ry2 = 0, rz2 = sy;
    const ux = ry2 * fz - rz2 * fy, uy = rz2 * fx - rx2 * fz, uz = rx2 * fy - ry2 * fx;
    view.fill(0);
    view[0] = rx2; view[4] = ry2; view[8] = rz2;
    view[1] = ux; view[5] = uy; view[9] = uz;
    view[2] = -fx; view[6] = -fy; view[10] = -fz;
    view[15] = 1;
    mat4Mul(viewProj, proj, view);

    // ---- exposure: the sun is 900x brighter at Mercury than at Neptune ----
    const camSunD = Math.max(Math.hypot(cam.x, cam.y, cam.z), 1e-6) / AU;
    const lit = Math.max(camSunD, 0.25);
    autoExp = P.autoExposure ? Math.min(1200, lit * lit) : 1;

    const f = frameF;
    f.set(viewProj, 0);
    f[16] = rx2 * tanX; f[17] = ry2 * tanX; f[18] = rz2 * tanX; f[19] = 0;
    f[20] = ux * tanY; f[21] = uy * tanY; f[22] = uz * tanY; f[23] = 0;
    f[24] = fx; f[25] = fy; f[26] = fz; f[27] = time;
    f[28] = W; f[29] = H; f[30] = 1 / W; f[31] = 1 / H;
    f[32] = P.exposure * autoExp; f[33] = P.bloom; f[34] = frameIndex; f[35] = DBG;
    f[36] = -cam.x; f[37] = -cam.y; f[38] = -cam.z; f[39] = SUN_B.req;
    // solar irradiance at 1 AU normalised to pi, so that a perfectly white Lambertian
    // surface at normal incidence has radiance 1.0 and real albedos land where they should
    f[40] = 3.1416; f[41] = 3.0942; f[42] = 2.9999; f[43] = pxAng;
    f[44] = activeIdx; f[45] = activeIdx >= 0 ? 1 : 0; f[46] = 0;
    f[47] = activeAtmo ? activeAtmo.ozone : 0;
    if (activeAtmo && activeIdx >= 0) {
      const b = bodies[activeIdx], A = activeAtmo;
      f[48] = b.req; f[49] = b.req + A.thick; f[50] = A.Hr; f[51] = A.Hm;
      f[52] = A.ray[0]; f[53] = A.ray[1]; f[54] = A.ray[2]; f[55] = A.g;
      f[56] = A.mie[0]; f[57] = A.mie[1]; f[58] = A.mie[2]; f[59] = A.mieAbs;
      f[60] = A.abs[0]; f[61] = A.abs[1]; f[62] = A.abs[2]; f[63] = A.ground;
    } else {
      f[48] = 1; f[49] = 2; f[50] = 1; f[51] = 1;
      f[52] = 0; f[53] = 0; f[54] = 0; f[55] = 0.7;
      f[56] = 0; f[57] = 0; f[58] = 0; f[59] = 0;
      f[60] = 0; f[61] = 0; f[62] = 0; f[63] = 0.1;
    }
    f[64] = 1.0; f[65] = 3.2; f[66] = P.ringOpacity; f[67] = P.ringBright;
    f[68] = P.ringAlbedo; f[69] = P.ringBack; f[70] = P.ringFwd; f[71] = P.opposition;
    // The automatic exposure compensates the inverse-square falloff so every planet reads
    // the same, which means it also multiplies the sky by d^2 -- at Neptune that is 900x and
    // the Milky Way whites out the frame. A real long exposure does exactly that; no real
    // camera can hold both. Stars and the galaxy are therefore held at a fixed *display*
    // brightness, a deliberate break from physical. The zodiacal light is left physical: its
    // own d^-2.6 falloff against the d^2 exposure leaves a gentle d^-0.6 fade outward, which
    // is correct and worth seeing.
    f[72] = P.starGain / autoExp; f[73] = P.milkyWay / autoExp; f[74] = P.zodiacal; f[75] = 0;
    f[76] = cam.x; f[77] = cam.y; f[78] = cam.z; f[79] = nearestGap;
    device.queue.writeBuffer(frameUBO, 0, f);
  }

  // --------------------------------------------------------------------------------------
  // UI
  // --------------------------------------------------------------------------------------
  const panel = document.getElementById('panel');
  function togglePanel() { panel.classList.toggle('hidden'); }
  const SLIDERS = [
    ['timeScale', 'Time scale (s/s)', 0, 200000, 100, ''],
    ['fov', 'Field of view (deg)', 15, 100, 1, 'fov'],
    ['exposure', 'Exposure', 0.05, 8, 0.01, ''],
    ['autoExposure', 'Auto exposure', 0, 1, 1, ''],
    ['bloom', 'Bloom', 0, 0.3, 0.001, ''],
    ['ringOpacity', 'Ring opacity', 0, 1, 0.01, ''],
    ['ringBright', 'Ring brightness', 0, 3, 0.01, ''],
    ['ringAlbedo', 'Ring particle albedo', 0.05, 0.995, 0.005, ''],
    ['ringBack', 'Ring backscatter', 0, 0.9, 0.01, ''],
    ['opposition', 'Opposition surge', 0, 2, 0.01, ''],
    ['atmosphere', 'Atmospheres', 0, 1, 1, ''],
    ['starGain', 'Star brightness', 0, 4, 0.01, ''],
    ['milkyWay', 'Milky Way', 0, 2, 0.01, ''],
    ['zodiacal', 'Zodiacal light', 0, 3, 0.01, ''],
  ];
  for (const [key, label, min, max, step, kind] of SLIDERS) {
    const row = document.createElement('label');
    const name = document.createElement('span'); name.textContent = label;
    const val = document.createElement('output'); val.textContent = String(P[key]);
    const inp = document.createElement('input');
    Object.assign(inp, { type: 'range', min, max, step, value: P[key] });
    inp.addEventListener('input', () => {
      P[key] = Number(inp.value); val.textContent = inp.value;
      if (kind === 'fov') cam.fovY = P.fov * D2R;
    });
    row.append(name, inp, val);
    panel.appendChild(row);
  }
  for (const k of Object.keys(P)) if (qs.has(k)) P[k] = Number(qs.get(k));
  cam.fovY = P.fov * D2R;
  if (qs.get('ui') === '0') { document.getElementById('hud').style.display = 'none'; panel.classList.add('hidden'); }

  const hudFps = document.getElementById('fps');
  const hudInfo = document.getElementById('info');
  function hudTarget() { if (hudInfo) hudInfo.textContent = 'target: ' + target.spec.n; }
  const hudNav = document.getElementById('nav');
  function hudMode() {
    canvas.style.cursor = mode === 'orbit' ? 'grab' : '';
    if (!hudNav) return;
    hudNav.textContent = mode === 'orbit'
      ? 'orbit · drag rotate · wheel or W/S zoom · right-drag pan · double-click a body to go · F free-fly'
      : 'free-fly · click: mouse look · WASD move · Space/E up · Q/C down · Shift fast · wheel speed · F orbit';
  }
  hudMode();

  // --------------------------------------------------------------------------------------
  // Orbit overlay
  // --------------------------------------------------------------------------------------
  // Seen as a whole system every planet is far below a pixel, so a 2D layer over the render
  // draws the orbits as lines and marks each planet with a ring and its name. It fades out as
  // the camera closes on a body, where an orbit line would cut straight across the disc.
  const ov = document.getElementById('ov');
  const ovx = ov.getContext('2d');
  let showOrbits = qs.get('orbits') !== '0';
  const ORBIT_N = 360;
  const ORBIT_COL = {
    Mercury: '#b9b0a6', Venus: '#e9d29e', Earth: '#7fb4ff', Mars: '#e57d55', Jupiter: '#dcb68c',
    Saturn: '#e8d293', Uranus: '#a3e3e8', Neptune: '#7392ff', Pluto: '#c8a990',
  };
  const planets = bodies.filter((b) => !b.spec.sun && !b.spec.parent);
  // the ellipses drift over centuries, so sampling them once is plenty
  const orbitPts = planets.map((b) => {
    const T = (jd - J2000) / 36525, tmp = [0, 0, 0];
    const pts = new Float64Array(ORBIT_N * 3);
    for (let i = 0; i < ORBIT_N; i++) {
      orbitPoint(KEP[b.spec.n], T, (i / ORBIT_N) * 2 * Math.PI, tmp);
      pts[i * 3] = tmp[0]; pts[i * 3 + 1] = tmp[1]; pts[i * 3 + 2] = tmp[2];
    }
    return pts;
  });

  function drawOverlay() {
    if (ov.width !== W || ov.height !== H) { ov.width = W; ov.height = H; }
    ovx.clearRect(0, 0, W, H);
    // gone within ~1 Gm of a surface, full strength from ~10 Gm out
    const fade = Math.max(0, Math.min(1, Math.log10(Math.max(nearestGap, 1) / 1e3)));
    if (!showOrbits || fade <= 0) return;
    const tanY = Math.tan(cam.fovY * 0.5), tanX = tanY * W / H;
    const sx = W / (2 * tanX), sy = H / (2 * tanY);
    const { f, r, u } = camBasis();
    const px = W / Math.max(1, ov.clientWidth);   // render pixels per CSS pixel
    const out = [0, 0, 0];
    // screen position and depth, or false if behind the camera
    const project = (x, y, z) => {
      const dx = x - cam.x, dy = y - cam.y, dz = z - cam.z;
      const zc = dx * f[0] + dy * f[1] + dz * f[2];
      if (zc <= 1e-3) return false;
      out[0] = W * 0.5 + ((dx * r[0] + dy * r[1] + dz * r[2]) / zc) * sx;
      out[1] = H * 0.5 - ((dx * u[0] + dy * u[1] + dz * u[2]) / zc) * sy;
      out[2] = zc;
      return true;
    };

    ovx.lineWidth = 1.2 * px;
    for (let i = 0; i < planets.length; i++) {
      const pts = orbitPts[i];
      ovx.strokeStyle = ORBIT_COL[planets[i].spec.n] || '#ccc';
      ovx.globalAlpha = (planets[i] === target ? 0.8 : 0.4) * fade;
      ovx.beginPath();
      let pen = false;
      for (let j = 0; j <= ORBIT_N; j++) {
        const k = (j % ORBIT_N) * 3;
        if (project(pts[k], pts[k + 1], pts[k + 2])) {
          if (pen) ovx.lineTo(out[0], out[1]); else ovx.moveTo(out[0], out[1]);
          pen = true;
        } else pen = false;
      }
      ovx.stroke();
    }

    ovx.font = `${Math.round(12 * px)}px system-ui, sans-serif`;
    ovx.textBaseline = 'middle';
    // From far out the Sun is below a pixel too, and the renderer only ray-traces its disc.
    // Labels go to the target planet first, then outward in, the Sun last; one that would
    // overlap a label already placed is dropped, so the crowded inner system stays readable.
    const order = [...planets].sort((a, b) =>
      (b === target) - (a === target) || planets.indexOf(b) - planets.indexOf(a)).concat(SUN_B);
    const placed = [];
    for (const b of order) {
      if (!project(b.pos[0], b.pos[1], b.pos[2])) continue;
      const x = out[0], y = out[1];
      if (x < -50 || y < -50 || x > W + 50 || y > H + 50) continue;
      const col = b === SUN_B ? '#ffe9b0' : ORBIT_COL[b.spec.n] || '#ccc';
      const disc = (b.req / out[2]) * sy;
      const ring = Math.max(disc + 3 * px, 4 * px);
      if (b !== SUN_B || disc < 1.5 * px) {
        ovx.globalAlpha = fade;
        ovx.strokeStyle = col;
        ovx.beginPath();
        ovx.arc(x, y, ring, 0, 2 * Math.PI);
        ovx.stroke();
      }
      if (disc < 1.5 * px) {
        ovx.fillStyle = col;
        ovx.beginPath();
        ovx.arc(x, y, (b === SUN_B ? 2.5 : 1.6) * px, 0, 2 * Math.PI);
        ovx.fill();
      }
      const lx = x + ring + 4 * px, lw = ovx.measureText(b.spec.n).width, lh = 12 * px;
      if (placed.some((p) => lx < p[0] + p[2] && p[0] < lx + lw && y - lh / 2 < p[1] + lh / 2 && p[1] - lh / 2 < y + lh / 2)) continue;
      placed.push([lx, y, lw]);
      ovx.fillStyle = b === target ? '#ffd79a' : '#e8e2d6';
      ovx.globalAlpha = (b === target ? 1 : 0.85) * fade;
      ovx.fillText(b.spec.n, lx, y);
    }
    ovx.globalAlpha = 1;
  }

  // --------------------------------------------------------------------------------------
  // Frame
  // --------------------------------------------------------------------------------------
  let frameIndex = 0;
  const submitList = [null];
  const shotName = qs.get('shot');
  const shotFrames = Number(qs.get('frames')) || 90;
  let shooting = false;
  let last = performance.now();
  let fpsAcc = 0, fpsN = 0, fpsT = 0;

  // start wherever the URL says, otherwise at Saturn -- it is the showpiece
  updateEphemeris();
  if (qs.get('cam')) {
    const v = qs.get('cam').split(',').map(Number);
    cam.x = v[0]; cam.y = v[1]; cam.z = v[2];
    cam.yaw = (v[3] || 0) * D2R; cam.pitch = (v[4] || 0) * D2R;
    if (qs.get('to')) setTarget(byName[qs.get('to')] || target);
  } else if (qs.get('view') === 'system' || qs.get('view') === 'inner') {
    overview = qs.get('view') === 'system' ? 1 : 2;
    systemView(overview === 1 ? 33 : 1.75, true);
  } else {
    gotoBody(byName[qs.get('to') || 'Saturn'] || byName.Saturn, Number(qs.get('d')) || 4.2,
      qs.has('ph') ? Number(qs.get('ph')) : undefined);
  }
  hudTarget();

  function frame(now) {
    if (shooting) return;
    requestAnimationFrame(frame);
    resize();
    let dt = Math.min(0.1, (now - last) / 1000);
    last = now;
    if (shotName) dt = 1 / 60;
    fpsAcc += dt; fpsN++; fpsT += dt;
    frameIndex++;

    jd += (dt * P.timeScale) / 86400;
    updateEphemeris();
    trackTarget();
    updateCamera(dt);
    writeScene((jd - J2000) * 86400);
    drawOverlay();

    if (fpsT > 0.4) {
      const fps = fpsN / fpsAcc;
      const km = nearestGap * 1000;
      const d = km > 1e7 ? (nearestGap / AU).toFixed(3) + ' AU'
        : km > 1e4 ? (km / 1000).toFixed(0) + ' Mm' : km.toFixed(1) + ' km';
      hudFps.textContent = `${fps.toFixed(0)} fps  ${(1000 / fps).toFixed(2)} ms` +
        (hasTS ? `  |  GPU ${gpuMs.toFixed(2)} ms` : '') +
        `  |  ${nearest.spec.n} +${d}  |  ` +
        (mode === 'orbit' ? 'orbit' : `x${speedScale < 1 ? speedScale.toFixed(3) : speedScale.toFixed(1)}`) +
        `  ${W}x${H}`;
      fpsAcc = 0; fpsN = 0; fpsT = 0;
    }

    const enc = device.createCommandEncoder();
    if (needLUT && activeIdx >= 0) {
      const cp = enc.beginComputePass();
      cp.setPipeline(transPipe);
      cp.setBindGroup(0, transBG);
      cp.dispatchWorkgroups(TRANS_W / 8, TRANS_H / 8);
      cp.setPipeline(msPipe);
      cp.setBindGroup(0, msBG);
      cp.dispatchWorkgroups(MS_SIZE / 8, MS_SIZE / 8);
      cp.end();
      needLUT = false;
    }

    if (hasTS) {
      mainPass.timestampWrites = { querySet: tsQuery, beginningOfPassWriteIndex: 0 };
    }
    const rp = enc.beginRenderPass(mainPass);
    rp.setPipeline(bgPipe);
    rp.setBindGroup(0, bgBG);
    rp.draw(3);
    const nDraw = splatBase + nSplat;
    if (nDraw > 0) {
      rp.setPipeline(starPipe);
      rp.setBindGroup(0, starBG);
      rp.setVertexBuffer(0, starBuf);
      rp.draw(3, nDraw);
    }
    rp.setPipeline(bodiesPipe);
    rp.setBindGroup(0, bodiesBG);
    rp.draw(3);
    rp.setPipeline(sunPipe);
    rp.setBindGroup(0, sunBG);
    rp.draw(3);
    rp.end();

    for (let m = 0; m < BLOOM_LEVELS; m++) {
      const p = enc.beginRenderPass(downPasses[m]);
      p.setPipeline(downPipe);
      p.setBindGroup(0, downBGs[m]);
      p.draw(3);
      p.end();
    }
    for (let i = 0; i < upPasses.length; i++) {
      const p = enc.beginRenderPass(upPasses[i]);
      p.setPipeline(upPipe);
      p.setBindGroup(0, upBGs[i]);
      p.draw(3);
      p.end();
    }

    compData[0] = P.exposure * autoExp; compData[1] = P.bloom;
    compData[2] = frameIndex % 64; compData[3] = 0.06;
    // AgX look: saturation and contrast. Planetary albedo contrast is genuinely low,
    // and 16 stops of scene range compresses what is left, so the look restores some.
    compData[4] = 1.28; compData[5] = 1.06; compData[6] = -0.02; compData[7] = 0;
    device.queue.writeBuffer(compUBO, 0, compData);
    const swapTex = ctx.getCurrentTexture();
    compPass.colorAttachments[0].view = swapTex.createView();
    if (hasTS) compPass.timestampWrites = { querySet: tsQuery, endOfPassWriteIndex: 1 };
    const pc = enc.beginRenderPass(compPass);
    pc.setPipeline(compPipe);
    pc.setBindGroup(0, bgComp);
    pc.draw(3);
    pc.end();

    let doTs = false;
    if (hasTS && !tsPending && (frameIndex & 7) === 0) {
      enc.resolveQuerySet(tsQuery, 0, 2, tsResolve, 0);
      enc.copyBufferToBuffer(tsResolve, 0, tsRead, 0, 16);
      doTs = true;
    }
    let readBuf = null;
    if (shotName && frameIndex >= shotFrames) {
      shooting = true;
      const bpr = Math.ceil(W * 4 / 256) * 256;
      readBuf = device.createBuffer({ size: bpr * H, usage: GU.MAP_READ | GU.COPY_DST });
      enc.copyTextureToBuffer({ texture: swapTex }, { buffer: readBuf, bytesPerRow: bpr }, [W, H]);
    }
    submitList[0] = enc.finish();
    device.queue.submit(submitList);
    if (doTs) { tsPending = true; tsRead.mapAsync(GPUMapMode.READ).then(onTs, onTsFail); }
    if (readBuf) saveShot(readBuf, Math.ceil(W * 4 / 256) * 256);
  }

  async function saveShot(buf, bpr) {
    await buf.mapAsync(GPUMapMode.READ);
    const src = new Uint8Array(buf.getMappedRange());
    const img = new ImageData(W, H);
    const bgra = swapFormat === 'bgra8unorm';
    for (let y = 0; y < H; y++) {
      for (let x = 0; x < W; x++) {
        const s = y * bpr + x * 4, d = (y * W + x) * 4;
        img.data[d] = src[s + (bgra ? 2 : 0)];
        img.data[d + 1] = src[s + 1];
        img.data[d + 2] = src[s + (bgra ? 0 : 2)];
        img.data[d + 3] = 255;
      }
    }
    buf.unmap();
    const oc = new OffscreenCanvas(W, H);
    const oc2 = oc.getContext('2d');
    oc2.putImageData(img, 0, 0);
    oc2.drawImage(ov, 0, 0, W, H);
    const jpg = /\.jpe?g$/i.test(shotName);
    const blob = await oc.convertToBlob(jpg ? { type: 'image/jpeg', quality: 0.92 }
                                           : { type: 'image/png' });
    await fetch('/__shot?name=' + encodeURIComponent(shotName), { method: 'POST', body: blob });
    log('shot saved ' + shotName + ' gpu ' + gpuMs.toFixed(2) + 'ms');
    document.title = 'SHOT DONE';
    if (qs.get('keep') === '1') { shooting = false; requestAnimationFrame(frame); }
  }

  resize();
  addEventListener('resize', resize);
  requestAnimationFrame(frame);
}

main().catch((e) => { log('init failed: ' + (e && e.stack || e)); });
