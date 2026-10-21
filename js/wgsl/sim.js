// FFT ocean simulation (Tessendorf) — 4 cascades of 256^2.
import { N, LOGN, CASCADES, NOISE_WGSL, PERIODIC_WGSL } from './common.js';

const SIM_STRUCT = /* wgsl */ `
const N: u32 = ${N}u;
const LOGN: u32 = ${LOGN}u;
const PI: f32 = 3.14159265358979;
const GRAV: f32 = 9.81;
const SURF: f32 = 7.4e-5;   // surface tension / density

struct Sim {
  L:          array<vec4f, 2>,  // patch size per cascade ([c>>2][c&3])
  kMin:       array<vec4f, 2>,
  kMax:       array<vec4f, 2>,
  foamDecay:  array<vec4f, 2>,  // 1/s per cascade
  foamThresh: array<vec4f, 2>,  // Jacobian threshold per cascade
  wind:       vec4f,  // U10 m/s, direction rad, fetch m, gamma
  swell:      vec4f,  // Hs m, direction rad, peak omega, spreading exponent
  misc:       vec4f,  // choppiness, short wave cut (m), loop period s, seed
  timeP:      vec4f,  // time (looped), dt, foam gain, spectrum gain
};
@group(0) @binding(0) var<uniform> S: Sim;

fn waveVec(x: u32, y: u32, L: f32) -> vec2f {
  let nx = select(i32(x), i32(x) - i32(N), x >= N / 2u);
  let ny = select(i32(y), i32(y) - i32(N), y >= N / 2u);
  return vec2f(f32(nx), f32(ny)) * (2.0 * PI / L);
}
fn omegaOf(k: f32) -> f32 { return sqrt(GRAV * k + SURF * k * k * k); }
fn dOmegaOf(k: f32) -> f32 { return (GRAV + 3.0 * SURF * k * k) / (2.0 * omegaOf(k)); }
fn cmul(a: vec2f, b: vec2f) -> vec2f { return vec2f(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x); }
fn mulI(a: vec2f) -> vec2f { return vec2f(-a.y, a.x); }
`;

export const SIM_INIT = SIM_STRUCT + NOISE_WGSL + /* wgsl */ `
@group(0) @binding(1) var<storage, read_write> h0: array<vec4f>;

fn rand01(s: u32) -> f32 { return (f32(pcgHash(s) >> 8u) + 0.5) / 16777216.0; }
fn gauss2(seed: u32) -> vec2f {
  let u1 = rand01(seed * 2u + 17u);
  let u2 = rand01(seed * 2u + 18u + pcgHash(seed));
  let r = sqrt(-2.0 * log(u1));
  let a = 2.0 * PI * u2;
  return r * vec2f(cos(a), sin(a));
}
fn wrapAngle(a: f32) -> f32 {
  var x = a;
  x = x - 2.0 * PI * floor((x + PI) / (2.0 * PI));
  return x;
}
fn jonswap(w: f32, wp: f32, alpha: f32, gamma: f32) -> f32 {
  let sigma = select(0.09, 0.07, w <= wp);
  let d = (w - wp) / (sigma * wp);
  let r = exp(-0.5 * d * d);
  let q = wp / w;
  let q2 = q * q;
  return alpha * GRAV * GRAV / (w * w * w * w * w) * exp(-1.25 * q2 * q2) * pow(gamma, r);
}
// Donelan-Banner directional spreading (normalised over [-pi, pi])
fn donelanBanner(theta: f32, w: f32, wp: f32) -> f32 {
  let ratio = w / wp;
  var beta: f32;
  if (ratio < 0.95) {
    beta = 2.61 * pow(ratio, 1.3);
  } else if (ratio < 1.6) {
    beta = 2.28 * pow(ratio, -1.3);
  } else {
    let eps = -0.4 + 0.8393 * exp(-0.567 * log(ratio * ratio));
    beta = pow(10.0, eps);
  }
  let sh = 1.0 / cosh(clamp(beta * theta, -20.0, 20.0));
  return beta / (2.0 * tanh(beta * PI)) * sh * sh;
}
// cos^2s(theta/2) spreading, normalised
fn cosSpread(theta: f32, s: f32) -> f32 {
  let q = sqrt(s + 0.25) / (2.0 * sqrt(PI));
  return q * pow(abs(cos(theta * 0.5)), 2.0 * s);
}

fn spectrum(k: vec2f) -> f32 {
  let kl = length(k);
  let w = omegaOf(kl);
  let theta = atan2(k.y, k.x);
  // fetch-limited wind sea (JONSWAP)
  let U = S.wind.x;
  let F = S.wind.z;
  var Sw = 0.0;
  if (U > 0.1) {
    let alpha = 0.076 * pow(U * U / (F * GRAV), 0.22);
    let wp = 22.0 * pow(GRAV * GRAV / (U * F), 1.0 / 3.0);
    Sw = jonswap(w, wp, alpha, S.wind.w) * donelanBanner(wrapAngle(theta - S.wind.y), w, wp);
  }
  // swell: narrow, long-crested JONSWAP
  if (S.swell.x > 0.001) {
    let ws = S.swell.z;
    let gs = 6.0;
    let hs4 = S.swell.x * 0.25;
    let alphaS = hs4 * hs4 * 5.0 * ws * ws * ws * ws / (GRAV * GRAV) * (1.0 - 0.287 * log(gs));
    Sw += jonswap(w, ws, alphaS, gs) * cosSpread(wrapAngle(theta - S.swell.y), S.swell.w);
  }
  // damp the very shortest waves
  let sc = S.misc.y;
  Sw *= exp(-kl * kl * sc * sc);
  return Sw * dOmegaOf(kl) / kl * S.timeP.w;
}

fn amplitude(k: vec2f, c: u32) -> f32 {
  let kl = length(k);
  if (kl < 1e-5 || kl < S.kMin[c >> 2u][c & 3u] || kl >= S.kMax[c >> 2u][c & 3u]) { return 0.0; }
  let dk = 2.0 * PI / S.L[c >> 2u][c & 3u];
  // the finest cascades are toned down (they otherwise read as crumpled foil)
  let g = select(1.0, select(0.85, 0.6, c == 5u), c >= 4u);
  return sqrt(max(spectrum(k), 0.0)) * dk * 0.5 * g;
}

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) id: vec3u) {
  let c = id.z;
  let L = S.L[c >> 2u][c & 3u];
  let k = waveVec(id.x, id.y, L);
  let mx = (N - id.x) % N;
  let my = (N - id.y) % N;
  let seed = u32(S.misc.w) * 7919u + c * 1000003u;
  let a = gauss2(seed + (id.y * N + id.x)) * amplitude(k, c);
  let b = gauss2(seed + (my * N + mx)) * amplitude(-k, c);
  h0[(c * N + id.y) * N + id.x] = vec4f(a, b.x, -b.y);
}
`;

export const SIM_EVOLVE = SIM_STRUCT + /* wgsl */ `
@group(0) @binding(1) var<storage, read> h0: array<vec4f>;
@group(0) @binding(2) var<storage, read_write> specA: array<vec4f>;
@group(0) @binding(3) var<storage, read_write> specB: array<vec4f>;

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) id: vec3u) {
  let c = id.z;
  let idx = (c * N + id.y) * N + id.x;
  let k = waveVec(id.x, id.y, S.L[c >> 2u][c & 3u]);
  let kl = length(k);
  if (kl < 1e-5) {
    specA[idx] = vec4f(0.0);
    specB[idx] = vec4f(0.0);
    return;
  }
  let w0 = 2.0 * PI / S.misc.z;
  let w = floor(omegaOf(kl) / w0) * w0;
  let ph = w * S.timeP.x;
  let e = vec2f(cos(ph), sin(ph));
  let H0 = h0[idx];
  // h0(k) travels along +k: h(k,t) = h0(k) e^{-iwt} + conj(h0(-k)) e^{iwt}
  let h = cmul(H0.xy, vec2f(e.x, -e.y)) + cmul(H0.zw, e);
  let ih = mulI(h);
  let lam = S.misc.x;
  let ik = 1.0 / kl;
  let Dx = ih * (lam * k.x * ik);
  let Dz = ih * (lam * k.y * ik);
  let jxz = h * (-lam * k.x * k.y * ik);
  let sx = ih * k.x;
  let sz = ih * k.y;
  let jxx = h * (-lam * k.x * k.x * ik);
  let jzz = h * (-lam * k.y * k.y * ik);
  specA[idx] = vec4f(Dx + mulI(Dz), h + mulI(jxz));
  specB[idx] = vec4f(sx + mulI(sz), jxx + mulI(jzz));
}
`;

const FFT_CORE = /* wgsl */ `
const N: u32 = ${N}u;
const LOGN: u32 = ${LOGN}u;
const PI: f32 = 3.14159265358979;
var<workgroup> wa: array<vec4f, ${N}>;
var<workgroup> wb: array<vec4f, ${N}>;
fn cmul(a: vec2f, b: vec2f) -> vec2f { return vec2f(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x); }
fn bitrev(i: u32) -> u32 { return reverseBits(i) >> (32u - LOGN); }
// In-place radix-2 inverse FFT over shared memory (input already bit-reversed).
fn butterflies(t: u32) {
  for (var s = 0u; s < LOGN; s++) {
    let half = 1u << s;
    let pos = t & (half - 1u);
    let i0 = ((t >> s) << (s + 1u)) + pos;
    let i1 = i0 + half;
    let ang = PI * f32(pos) / f32(half);
    let w = vec2f(cos(ang), sin(ang));
    let a0 = wa[i0]; let a1 = wa[i1];
    let b0 = wb[i0]; let b1 = wb[i1];
    let ta = vec4f(cmul(a1.xy, w), cmul(a1.zw, w));
    let tb = vec4f(cmul(b1.xy, w), cmul(b1.zw, w));
    wa[i0] = a0 + ta; wa[i1] = a0 - ta;
    wb[i0] = b0 + tb; wb[i1] = b0 - tb;
    workgroupBarrier();
  }
}
`;

export const FFT_ROWS = FFT_CORE + /* wgsl */ `
@group(0) @binding(0) var<storage, read> inA: array<vec4f>;
@group(0) @binding(1) var<storage, read> inB: array<vec4f>;
@group(0) @binding(2) var<storage, read_write> outA: array<vec4f>;
@group(0) @binding(3) var<storage, read_write> outB: array<vec4f>;

@compute @workgroup_size(${N / 2})
fn main(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_index) t: u32) {
  let base = (wg.z * N + wg.x) * N;
  let e0 = t;
  let e1 = t + N / 2u;
  wa[bitrev(e0)] = inA[base + e0];
  wa[bitrev(e1)] = inA[base + e1];
  wb[bitrev(e0)] = inB[base + e0];
  wb[bitrev(e1)] = inB[base + e1];
  workgroupBarrier();
  butterflies(t);
  outA[base + e0] = wa[e0];
  outA[base + e1] = wa[e1];
  outB[base + e0] = wb[e0];
  outB[base + e1] = wb[e1];
}
`;

// Column pass + final assembly (displacement / derivative / slope-moment textures).
export const FFT_COLS = FFT_CORE + /* wgsl */ `
@group(0) @binding(0) var<storage, read> inA: array<vec4f>;
@group(0) @binding(1) var<storage, read> inB: array<vec4f>;
@group(0) @binding(2) var dispOut: texture_storage_2d_array<rgba16float, write>;
@group(0) @binding(3) var derivOut: texture_storage_2d_array<rgba16float, write>;
@group(0) @binding(4) var momOut: texture_storage_2d_array<rgba16float, write>;

fn emit(x: u32, y: u32, c: u32, a: vec4f, b: vec4f) {
  // a = (Dx, Dz, Dy, jxz), b = (sx, sz, jxx, jzz)
  let p = vec2u(x, y);
  textureStore(dispOut, p, c, vec4f(a.x, a.z, a.y, a.w));
  textureStore(derivOut, p, c, b);
  // slope moments plus a copy of jxz, so the ocean shader needs no dispTex fetch
  textureStore(momOut, p, c, vec4f(b.x * b.x, b.y * b.y, b.x * b.y, a.w));
}

@compute @workgroup_size(${N / 2})
fn main(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_index) t: u32) {
  let x = wg.x;
  let c = wg.z;
  let e0 = t;
  let e1 = t + N / 2u;
  let i0 = (c * N + e0) * N + x;
  let i1 = (c * N + e1) * N + x;
  wa[bitrev(e0)] = inA[i0];
  wa[bitrev(e1)] = inA[i1];
  wb[bitrev(e0)] = inB[i0];
  wb[bitrev(e1)] = inB[i1];
  workgroupBarrier();
  butterflies(t);
  emit(x, e0, c, wa[e0], wb[e0]);
  emit(x, e1, c, wa[e1], wb[e1]);
}
`;

// Box-filter mip generation for the three simulation textures (array of cascades).
export const SIM_MIPS = /* wgsl */ `
@group(0) @binding(0) var srcA: texture_2d_array<f32>;
@group(0) @binding(1) var srcB: texture_2d_array<f32>;
@group(0) @binding(2) var srcC: texture_2d_array<f32>;
@group(0) @binding(3) var dstA: texture_storage_2d_array<rgba16float, write>;
@group(0) @binding(4) var dstB: texture_storage_2d_array<rgba16float, write>;
@group(0) @binding(5) var dstC: texture_storage_2d_array<rgba16float, write>;

fn box(t: texture_2d_array<f32>, p: vec2i, l: u32) -> vec4f {
  return 0.25 * (textureLoad(t, p, l, 0) + textureLoad(t, p + vec2i(1, 0), l, 0) +
                 textureLoad(t, p + vec2i(0, 1), l, 0) + textureLoad(t, p + vec2i(1, 1), l, 0));
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) id: vec3u) {
  let size = textureDimensions(dstA);
  if (id.x >= size.x || id.y >= size.y) { return; }
  let p = vec2i(id.xy) * 2;
  textureStore(dstA, id.xy, id.z, box(srcA, p, id.z));
  textureStore(dstB, id.xy, id.z, box(srcB, p, id.z));
  textureStore(dstC, id.xy, id.z, box(srcC, p, id.z));
}
`;

// Tileable foam micro-structure: R = bubble lace (multi-scale Worley edges),
// G = finer lace, B = soft fbm density, A = streak noise.
export const FOAM_TEX = NOISE_WGSL + PERIODIC_WGSL + /* wgsl */ `
@group(0) @binding(0) var dst: texture_storage_2d<rgba8unorm, write>;
const SIZE: f32 = 512.0;

fn h2(i: vec2i, salt: u32) -> vec2f {
  let h = pcgHash((u32(i.x) * 73856093u) ^ pcgHash(u32(i.y) * 19349663u + salt));
  let h2v = pcgHash(h + 0x9e3779b9u);
  return vec2f(f32(h >> 8u), f32(h2v >> 8u)) / 16777216.0;
}
// "swiss cheese" foam: 1 in foam material, 0 inside bubbles; per-bubble radius
fn bubbles(uv: vec2f, period: i32, salt: u32, rmin: f32, rmax: f32) -> f32 {
  let w = vec2f(pfbm(uv * f32(period) * 0.5, period / 2, 3), pfbm(uv * f32(period) * 0.5 + 17.0, period / 2, 3));
  let p = uv * f32(period) + w * 0.9;
  let ip = vec2i(floor(p));
  let fp = p - floor(p);
  var v = 9.0;
  for (var j = -1; j <= 1; j++) {
    for (var i = -1; i <= 1; i++) {
      let o = vec2i(i, j);
      let cell = ((ip + o) % period + period) % period;
      let hh = h2(cell, salt);
      let r = vec2f(o) + hh * 0.9 + 0.05 - fp;
      let rad = mix(rmin, rmax, h2(cell, salt + 7u).x);
      v = min(v, length(r) / rad);
    }
  }
  return smoothstep(0.8, 1.25, v);
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) id: vec3u) {
  let uv = (vec2f(id.xy) + 0.5) / SIZE;
  // irregular holes from warped fbm (big sheets with ragged openings) ...
  let w1 = vec2f(pfbm(uv * 5.0, 5, 3), pfbm(uv * 5.0 + vec2f(3.7, 9.1), 5, 3));
  let m1 = smoothstep(-0.14, 0.06, pfbm(uv * 10.0 + w1 * 1.4, 10, 5));
  let w2 = vec2f(pfbm(uv * 12.0 + 1.3, 12, 2), pfbm(uv * 12.0 + 7.9, 12, 2));
  let m2 = smoothstep(-0.16, 0.04, pfbm(uv * 27.0 + w2 * 0.9, 27, 4));
  // ... plus small round bubbles
  let b3 = bubbles(uv, 29, 37u, 0.06, 0.45);
  let b4 = bubbles(uv, 61, 53u, 0.05, 0.42);
  let dens = pfbm(uv * 4.0, 4, 5) * 0.5 + 0.5;
  let fineD = pfbm(uv * 16.0, 16, 3) * 0.5 + 0.5;
  let big = m1 * (0.55 + 0.45 * m2) * (0.75 + 0.25 * b3);
  let fine = 0.45 + 0.55 * b3 * (0.7 + 0.3 * b4);
  let streak = pfbm(uv * vec2f(4.0, 16.0), 4, 5) * 0.5 + 0.5;
  textureStore(dst, id.xy, vec4f(clamp(big * 0.6 + dens * 0.4, 0.0, 1.0), clamp(fine * 0.8 + fineD * 0.2, 0.0, 1.0), dens, streak));
}
`;

export const RGBA8_MIPS = /* wgsl */ `
@group(0) @binding(0) var src: texture_2d<f32>;
@group(0) @binding(1) var dst: texture_storage_2d<rgba8unorm, write>;
@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) id: vec3u) {
  let size = textureDimensions(dst);
  if (id.x >= size.x || id.y >= size.y) { return; }
  let p = vec2i(id.xy) * 2;
  let v = 0.25 * (textureLoad(src, p, 0) + textureLoad(src, p + vec2i(1, 0), 0) +
                  textureLoad(src, p + vec2i(0, 1), 0) + textureLoad(src, p + vec2i(1, 1), 0));
  textureStore(dst, id.xy, v);
}
`;

// World-space foam clipmaps (toroidally addressed, centred on the camera).
// Foam is injected where the TOTAL surface Jacobian (all cascades) folds, then decays.
export const FOAM_RES = 1024;
export const FOAM_UPDATE = /* wgsl */ `
const RES: i32 = ${FOAM_RES};
struct FoamU {
  snap: array<vec4i, 2>,     // per level: current camera texel (x, z), previous (x, z)
  p0: vec4f,                 // texel size level 0, level 1, dt, decay rate
  p1: vec4f,                 // J threshold, gain, growth, -
  cascL: array<vec4f, 2>,
  cascInvL: array<vec4f, 2>,
  drift: vec4f,              // foam drift velocity (m/s) xz
};
@group(0) @binding(0) var<uniform> U: FoamU;
@group(0) @binding(1) var prevTex: texture_2d_array<f32>;
@group(0) @binding(2) var dst: texture_storage_2d_array<rgba16float, write>;
@group(0) @binding(3) var dispTex: texture_2d_array<f32>;
@group(0) @binding(4) var derivTex: texture_2d_array<f32>;
@group(0) @binding(5) var samp: sampler;
@group(0) @binding(6) var sampPrev: sampler;

fn wrapIdx(i: i32, lo: i32) -> i32 { return lo + (((i - lo) % RES) + RES) % RES; }

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) id: vec3u) {
  let lvl = id.z;
  let sn = U.snap[lvl];
  let ts = select(U.p0.x, U.p0.y, lvl == 1u);
  let lo = sn.xy - vec2i(RES / 2);
  let g = vec2i(wrapIdx(i32(id.x), lo.x), wrapIdx(i32(id.y), lo.y));
  let plo = sn.zw - vec2i(RES / 2);
  let inPrev = all(g >= plo) && all(g < plo + vec2i(RES));
  let p = (vec2f(g) + 0.5) * ts;
  var jxx = 0.0; var jzz = 0.0; var jxz = 0.0;
  for (var c = 0u; c < ${CASCADES}u; c++) {
    let texel = U.cascL[c >> 2u][c & 3u] / ${N}.0;
    let lod = log2(max(ts / texel, 1.0));
    if (lod < 7.5) {
      let uv = p * U.cascInvL[c >> 2u][c & 3u];
      let d = textureSampleLevel(derivTex, samp, uv, c, lod);
      let a = textureSampleLevel(dispTex, samp, uv, c, lod);
      jxx += d.z; jzz += d.w; jxz += a.w;
    }
  }
  let J = (1.0 + jxx) * (1.0 + jzz) - jxz * jxz;
  var prev = vec4f(0.0);
  if (inPrev) {
    // semi-Lagrangian advection: foam slowly drifts downwind (and spreads a little)
    let back = (vec2f(id.xy) + 0.5 - U.drift.xy * U.p0.z / ts) / f32(RES);
    prev = textureSampleLevel(prevTex, sampPrev, back, lvl, 0.0);
  }
  let dt = U.p0.z;
  let inject = clamp((U.p1.x - J) * U.p1.y, 0.0, 1.0);
  var f = prev.r * exp(-dt * U.p0.w);
  f = max(f, inject);
  f = min(f + inject * dt * U.p1.z, 1.6);
  // g: freshness (recently injected) used for brighter, denser crest foam
  let fresh = max(prev.g * exp(-dt * 1.6), inject);
  textureStore(dst, id.xy, lvl, vec4f(f, fresh, J, 1.0));
}
`;

export const ARRAY16_MIPS = /* wgsl */ `
@group(0) @binding(0) var src: texture_2d_array<f32>;
@group(0) @binding(1) var dst: texture_storage_2d_array<rgba16float, write>;
@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) id: vec3u) {
  let size = textureDimensions(dst);
  if (id.x >= size.x || id.y >= size.y) { return; }
  let p = vec2i(id.xy) * 2;
  let v = 0.25 * (textureLoad(src, p, id.z, 0) + textureLoad(src, p + vec2i(1, 0), id.z, 0) +
                  textureLoad(src, p + vec2i(0, 1), id.z, 0) + textureLoad(src, p + vec2i(1, 1), id.z, 0));
  textureStore(dst, id.xy, id.z, v);
}
`;
