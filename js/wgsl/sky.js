// Physically based atmosphere (Hillaire 2020 style LUTs) + procedural cloud deck.
import { FRAME_WGSL, ATMO_CORE_WGSL, LIGHT_WGSL, NOISE_WGSL, SKY_W, SKY_H, ENV_SIZE, ENV_MIPS } from './common.js';

const ATMO_UNIFORM = /* wgsl */ `
struct AtmoP { haze: vec4f };   // x = mie multiplier
@group(0) @binding(0) var<uniform> A: AtmoP;
`;

export const TRANS_LUT = ATMO_CORE_WGSL + ATMO_UNIFORM + /* wgsl */ `
@group(0) @binding(1) var dst: texture_storage_2d<rgba16float, write>;
@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) id: vec3u) {
  if (f32(id.x) >= TRANS_W || f32(id.y) >= TRANS_H) { return; }
  let uv = (vec2f(id.xy) + 0.5) / vec2f(TRANS_W, TRANS_H);
  let rm = transUVToRMu(uv);
  let ro = vec3f(0.0, rm.x, 0.0);
  let rd = vec3f(sqrt(max(0.0, 1.0 - rm.y * rm.y)), rm.y, 0.0);
  let tMax = raySphere(ro, rd, RT).y;
  var od = vec3f(0.0);
  let STEPS = 64u;
  let dt = tMax / f32(STEPS);
  for (var i = 0u; i < STEPS; i++) {
    let p = ro + rd * ((f32(i) + 0.5) * dt);
    od += medium(length(p) - RG, A.haze.x).ext * dt;
  }
  textureStore(dst, id.xy, vec4f(exp(-od), 1.0));
}
`;

export const MS_LUT = ATMO_CORE_WGSL + ATMO_UNIFORM + /* wgsl */ `
@group(0) @binding(1) var transLUT: texture_2d<f32>;
@group(0) @binding(2) var sampC: sampler;
@group(0) @binding(3) var dst: texture_storage_2d<rgba16float, write>;

fn sampleTrans(r: f32, mu: f32) -> vec3f {
  return textureSampleLevel(transLUT, sampC, transRMuToUV(r, mu), 0.0).rgb;
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) id: vec3u) {
  if (f32(id.x) >= MS_SIZE || f32(id.y) >= MS_SIZE) { return; }
  var uv = (vec2f(id.xy) + 0.5) / MS_SIZE;
  uv = vec2f(fromSubUvsToUnit(uv.x, MS_SIZE), fromSubUvsToUnit(uv.y, MS_SIZE));
  let cosSun = uv.x * 2.0 - 1.0;
  let h = max(uv.y * (RT - RG), 0.01);
  let r = RG + h;
  let ro = vec3f(0.0, r, 0.0);
  let sun = vec3f(sqrt(max(0.0, 1.0 - cosSun * cosSun)), cosSun, 0.0);
  var Lsum = vec3f(0.0);
  var Fsum = vec3f(0.0);
  let SQ = 8u;
  let STEPS = 20u;
  for (var a = 0u; a < SQ; a++) {
    for (var b = 0u; b < SQ; b++) {
      let th = acos(1.0 - 2.0 * (f32(b) + 0.5) / f32(SQ));
      let ph = 2.0 * PI * (f32(a) + 0.5) / f32(SQ);
      let rd = vec3f(sin(th) * cos(ph), cos(th), sin(th) * sin(ph));
      let tTop = raySphere(ro, rd, RT);
      let tGr = raySphere(ro, rd, RG);
      var tMax = tTop.y;
      let hitG = tGr.x > 0.0;
      if (hitG) { tMax = tGr.x; }
      let dt = tMax / f32(STEPS);
      var T = vec3f(1.0);
      var L = vec3f(0.0);
      var Fms = vec3f(0.0);
      for (var s = 0u; s < STEPS; s++) {
        let p = ro + rd * ((f32(s) + 0.5) * dt);
        let pr = length(p);
        let m = medium(pr - RG, A.haze.x);
        let scat = m.rayS + vec3f(m.mieS);
        let Ts = exp(-m.ext * dt);
        let smu = dot(p, sun) / pr;
        var sunT = sampleTrans(pr, smu);
        if (sunBelowHorizon(pr, smu)) { sunT = vec3f(0.0); }
        let ext = max(m.ext, vec3f(1e-9));
        let S = scat * sunT / (4.0 * PI);
        L += T * (S - S * Ts) / ext;
        Fms += T * (scat - scat * Ts) / ext;
        T *= Ts;
      }
      if (hitG) {
        let pg = ro + rd * tMax;
        let ng = normalize(pg);
        let smu = dot(ng, sun);
        L += T * sampleTrans(RG, smu) * max(smu, 0.0) * 0.06 / PI;
      }
      Lsum += L;
      Fsum += Fms;
    }
  }
  let invN = 1.0 / f32(SQ * SQ);
  let L2 = Lsum * invN;
  let fms = Fsum * invN;
  textureStore(dst, id.xy, vec4f(L2 / (1.0 - fms), 1.0));
}
`;

// Sky-view LUT, recomputed every frame (sun / camera altitude may change).
export const SKYVIEW_LUT = FRAME_WGSL + ATMO_CORE_WGSL + /* wgsl */ `
@group(0) @binding(0) var<uniform> F: Frame;
@group(0) @binding(1) var transLUT: texture_2d<f32>;
@group(0) @binding(2) var msLUT: texture_2d<f32>;
@group(0) @binding(3) var sampC: sampler;
@group(0) @binding(4) var dst: texture_storage_2d<rgba16float, write>;
const SKY_W: f32 = ${SKY_W}.0;
const SKY_H: f32 = ${SKY_H}.0;

fn sampleTrans(r: f32, mu: f32) -> vec3f {
  return textureSampleLevel(transLUT, sampC, transRMuToUV(r, mu), 0.0).rgb;
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) id: vec3u) {
  if (f32(id.x) >= SKY_W || f32(id.y) >= SKY_H) { return; }
  var uv = (vec2f(id.xy) + 0.5) / vec2f(SKY_W, SKY_H);
  uv = vec2f(fromSubUvsToUnit(uv.x, SKY_W), fromSubUvsToUnit(uv.y, SKY_H));
  let haze = F.atmo.x;
  let r = RG + F.atmo.y;
  let vHorizon = rhoOf(r);
  let cosBeta = vHorizon / r;
  let beta = acos(clamp(cosBeta, -1.0, 1.0));
  let zha = PI - beta;
  var vza: f32;
  if (uv.y < 0.5) {
    var coord = 2.0 * uv.y;
    coord = 1.0 - coord;
    coord *= coord;
    coord = 1.0 - coord;
    vza = zha * coord;
  } else {
    var coord = uv.y * 2.0 - 1.0;
    coord *= coord;
    vza = zha + beta * coord;
  }
  let vzc = cos(vza);
  let vzs = sin(vza);
  let cx = uv.x * uv.x;
  let lvc = -(cx * 2.0 - 1.0);
  let lvs = sqrt(max(0.0, 1.0 - lvc * lvc));
  let rd = vec3f(vzs * lvc, vzc, vzs * lvs);
  let smu = F.sunDir.y;
  let sun = vec3f(sqrt(max(0.0, 1.0 - smu * smu)), smu, 0.0);
  let ro = vec3f(0.0, r, 0.0);

  let tTop = raySphere(ro, rd, RT);
  let tGr = raySphere(ro, rd, RG);
  var tMax = tTop.y;
  if (tGr.x > 0.0) { tMax = tGr.x; }
  let cosT = dot(rd, sun);
  let pR = phaseRayleigh(cosT);
  let pM = phaseMie(cosT, 0.84);
  var L = vec3f(0.0);
  var T = vec3f(1.0);
  let STEPS = 48u;
  var tPrev = 0.0;
  for (var i = 0u; i < STEPS; i++) {
    let fa = (f32(i) + 1.0) / f32(STEPS);
    let tn = tMax * fa * fa;
    let dt = tn - tPrev;
    let t = tPrev + dt * 0.3;
    tPrev = tn;
    let p = ro + rd * t;
    let pr = length(p);
    let h = pr - RG;
    let m = medium(h, haze);
    let pmu = dot(p, sun) / pr;
    var sunT = sampleTrans(pr, pmu);
    if (sunBelowHorizon(pr, pmu)) { sunT = vec3f(0.0); }
    let msuv = vec2f(fromUnitToSubUvs(pmu * 0.5 + 0.5, MS_SIZE), fromUnitToSubUvs(clamp(h / (RT - RG), 0.0, 1.0), MS_SIZE));
    let ms = textureSampleLevel(msLUT, sampC, msuv, 0.0).rgb;
    let S = sunT * (m.rayS * pR + vec3f(m.mieS * pM)) + ms * (m.rayS + vec3f(m.mieS));
    let Ts = exp(-m.ext * dt);
    let ext = max(m.ext, vec3f(1e-9));
    L += T * (S - S * Ts) / ext;
    T *= Ts;
  }
  textureStore(dst, id.xy, vec4f(L, 1.0));
}
`;

// Functions for rendering the sky (background, env cube, ocean fog).
// Requires: F (Frame), transLUT, skyLUT, sampC declared by the including module.
export const SKY_RENDER_FUNCS = NOISE_WGSL + /* wgsl */ `
const SKY_W: f32 = ${SKY_W}.0;
const SKY_H: f32 = ${SKY_H}.0;

fn camR() -> f32 { return RG + F.atmo.y; }

fn sampleTrans(r: f32, mu: f32) -> vec3f {
  return textureSampleLevel(transLUT, sampC, transRMuToUV(r, mu), 0.0).rgb;
}

fn sunIlluminance() -> vec3f {
  return sampleTrans(camR(), F.sunDir.y) * F.sunDir.w;
}

fn skyViewUV(dir: vec3f) -> vec2f {
  let r = camR();
  let vHorizon = rhoOf(r);
  let cosBeta = vHorizon / r;
  let beta = acos(clamp(cosBeta, -1.0, 1.0));
  let zha = PI - beta;
  let vzc = clamp(dir.y, -1.0, 1.0);
  var uv: vec2f;
  if (vzc >= -cosBeta) {
    var coord = acos(vzc) / zha;
    coord = 1.0 - coord;
    coord = sqrt(max(coord, 0.0));
    coord = 1.0 - coord;
    uv.y = coord * 0.5;
  } else {
    var coord = (acos(vzc) - zha) / beta;
    coord = sqrt(max(coord, 0.0));
    uv.y = coord * 0.5 + 0.5;
  }
  let dl = length(dir.xz);
  let sl = length(F.sunDir.xz);
  var lvc = 1.0;
  if (dl > 1e-6 && sl > 1e-6) { lvc = clamp(dot(dir.xz, F.sunDir.xz) / (dl * sl), -1.0, 1.0); }
  uv.x = sqrt(max(-lvc * 0.5 + 0.5, 0.0));
  return vec2f(fromUnitToSubUvs(uv.x, SKY_W), fromUnitToSubUvs(uv.y, SKY_H));
}

fn skyAtmo(dir: vec3f) -> vec3f {
  return textureSampleLevel(skyLUT, sampC, skyViewUV(dir), 0.0).rgb * F.sunDir.w;
}
`;

// Clouds + full sky radiance (needs the same bindings as SKY_RENDER_FUNCS).
export const SKY_CLOUD_FUNCS = /* wgsl */ `

// ---- clouds -------------------------------------------------------------
// Two thin, light-aware layers: broken low cumulus with cauliflower edges (2 km) and
// high cirrus wisps (7 km). Each is shaded with self-shadowing towards the sun, bright
// sun-facing rims, silver linings on thin back-lit parts and warm multiple scattering.
const CLOUD_ROT: mat2x2f = mat2x2f(0.8, 0.6, -0.6, 0.8);
// fbm whose octaves fade once smaller than the pixel footprint fp (noise units);
// 'billow' turns the small octaves into bulgy |noise| for cauliflower-like edges
fn cloudFbm(p: vec2f, fp: f32, oct: i32, billow: f32) -> f32 {
  var s = 0.0;
  var a = 0.5;
  var q = p;
  var f = 1.0;
  for (var i = 0; i < oct; i++) {
    // fade octaves well before they reach pixel size (half-res buffer) -> no shimmer in motion
    let w = 1.0 - smoothstep(0.08, 0.28, fp * f);
    // the weight only falls as f grows: once an octave is fully faded, so are all the rest
    if (w <= 0.0) { break; }
    var n = gnoise(q);
    if (i >= 2) { n = mix(n, abs(n) * 2.0 - 0.38, billow); }
    s += a * n * w;
    q = CLOUD_ROT * q * 2.03 + vec2f(13.1, 7.7);
    f *= 2.03;
    a *= 0.5;
  }
  return s;
}
// ---- low cumulus: a ray-marched slab (km) whose density comes from a tiling 3D noise
@group(0) @binding(4) var noise3: texture_3d<f32>;
@group(0) @binding(5) var samp3: sampler;
const CB: f32 = 1.2;    // cloud base
const CT: f32 = 3.6;    // highest tops
const SIG: f32 = 30.0;  // extinction per km at density 1
fn lowDens(p: vec3f, detail: bool) -> f32 {
  let hf = (p.y - CB) / (CT - CB);
  let q = p + vec3f(F.cloudP.w * 0.004, 0.0, F.cloudP.w * 0.0012);
  // large-scale weather groups the clouds, leaving clear sky between the groups
  let wx = textureSampleLevel(noise3, samp3, vec3f(q.xz * 0.02, 0.5).xzy, 0.0).r;
  let cov = F.cloudP.y * mix(0.6, 1.2, smoothstep(0.3, 0.7, wx));
  // flat bases, rounded tops: the noise has to beat a threshold that rises with height
  var d = textureSampleLevel(noise3, samp3, q * vec3f(0.12, 0.35, 0.12), 0.0).r
        - (1.0 - cov) - 0.2 * hf - 0.3 * (1.0 - smoothstep(0.0, 0.06, hf));
  // billowy erosion only inside cloud: cauliflower edges
  if (detail && d > 0.0) { d -= 0.25 * (1.0 - textureSampleLevel(noise3, samp3, q * 0.7, 0.0).g); }
  return clamp((d - 0.015) * 12.0, 0.0, 1.0) * step(0.0, hf) * step(hf, 1.0);
}
// premultiplied radiance + opacity of the low layer along dir; jit in [0,1) offsets the steps
fn lowLayer(dir: vec3f, steps: i32, jit: f32) -> vec4f {
  let ro = vec3f(0.0, camR(), 0.0);
  let t0 = raySphere(ro, dir, RG + CB).y;
  // clear band near the horizon, where the low sun shines under the clouds
  let fade = 1.0 - smoothstep(20.0, 38.0, t0);
  if (fade <= 0.0 || F.cloudP.y < 0.001) { return vec4f(0.0); }
  let dt = (min(raySphere(ro, dir, RG + CT).y, t0 + 6.0) - t0) / f32(steps);
  let off = vec3f(F.camPos.x, 0.0, F.camPos.z) * 0.001;
  let cosT = dot(dir, F.sunDir.xyz);
  // dual-lobe phase per scattering octave (Wrenninge): later octaves are softer and reach deeper
  let ph0 = mix(phaseHG(cosT, 0.8), phaseHG(cosT, -0.25), 0.35);
  let ph1 = mix(phaseHG(cosT, 0.4), phaseHG(cosT, -0.12), 0.35);
  let ph2 = mix(phaseHG(cosT, 0.2), phaseHG(cosT, -0.06), 0.35);
  let sunC = sampleTrans(RG + CB, F.sunDir.y) * F.sunDir.w;
  let skyTop = skyAtmo(vec3f(0.0, 1.0, 0.0)) * 0.5;
  let skyBase = skyAtmo(normalize(vec3f(dir.x, 0.03, dir.z))) * 0.2;
  var T = 1.0;
  var L = vec3f(0.0);
  for (var i = 0; i < steps; i++) {
    let w = ro + dir * (t0 + (f32(i) + jit) * dt);
    let p = vec3f(w.x, length(w) - RG, w.z) + off;
    let d = lowDens(p, true);
    if (d <= 0.0) { continue; }
    // optical depth towards the sun from taps at 0.1, 0.4, 0.9 and 1.6 km
    var ts = 0.0;
    for (var j = 1; j <= 4; j++) {
      let fj = f32(j);
      ts += lowDens(p + F.sunDir.xyz * 0.1 * fj * fj, false) * 0.1 * (2.0 * fj - 1.0);
    }
    ts *= SIG;
    let hf = (p.y - CB) / (CT - CB);
    let S = sunC * (ph0 * exp(-ts) + 0.35 * ph1 * exp(-ts * 0.3) + 0.15 * ph2 * exp(-ts * 0.1))
          + mix(skyBase, skyTop, sqrt(hf));
    let a = exp(-d * SIG * dt);
    L += T * S * (1.0 - a);
    T *= a;
    if (T < 0.02) { break; }
  }
  return vec4f(L, 1.0 - T) * fade * exp(-t0 * 0.02 * (0.6 + 0.4 * F.atmo.x));
}
fn highEdge(q: vec2f, fp: f32, oct: i32) -> f32 {
  let t = F.cloudP.w;
  // altostratus deck: smooth, slightly fibrous
  let p = CLOUD_ROT * (q * vec2f(0.09, 0.20)) + vec2f(t * 0.0015, 0.0);
  let n = cloudFbm(p, fp * 0.2, oct, 0.0) * 1.25 + 0.5;
  let big = gnoise(q * 0.012 + vec2f(7.0, 3.0)) * 0.6;
  return n + big - (1.0 - F.atmo.z);
}
fn highSoft(fp: f32) -> f32 { return 0.85 + min(fp * 0.2, 1.0) * 0.3; }
fn highDensity(q: vec2f, fp: f32, oct: i32) -> f32 {
  return clamp(highEdge(q, fp, oct) / highSoft(fp), 0.0, 1.0);
}

struct CloudHit { t: f32, q: vec2f, mu: f32, up: vec3f };
fn hitLayer(dir: vec3f, hc: f32) -> CloudHit {
  var h: CloudHit;
  let ro = vec3f(0.0, camR(), 0.0);
  h.t = raySphere(ro, dir, RG + hc).y;
  let p = ro + dir * h.t;
  h.up = p / length(p);
  h.q = p.xz + F.camPos.xz * 0.001;
  h.mu = max(dot(h.up, dir), 0.03);
  return h;
}

// d: density at the hit; s1..s3: densities 80 m / 250 m / 600 m towards the sun
fn shadeLayer(dir: vec3f, h: CloudHit, hc: f32, d: f32, s1: f32, s2: f32, s3: f32, sigma: f32, g: f32) -> vec4f {
  let tau = d * sigma / h.mu;
  let alpha = 1.0 - exp(-tau);
  let tauS = (d * 0.5 + s1 * 0.6 + s2 + s3 * 0.7) * sigma * 0.55;
  let Ts = exp(-tauS);
  let sunC = sampleTrans(RG + hc, dot(h.up, F.sunDir.xyz)) * F.sunDir.w;
  let cosT = dot(dir, F.sunDir.xyz);
  let ph = mix(phaseHG(cosT, g), phaseHG(cosT, -0.2), 0.25);
  let fwd = smoothstep(-0.2, 0.7, cosT);
  // single scattering: reflected when front-lit, transmitted through thin parts when back-lit
  let single = sunC * Ts * ph * mix(alpha * 0.55, tau * exp(-tau) * 2.2, fwd) * 3.0;
  // edges where density falls off towards the sun catch direct light
  let rim = clamp((d - s1) * 2.5, 0.0, 1.0);
  // multiple scattering: a low sun lights cloud tops and sun-facing flanks, not the bases we
  // see from below, so diffused sunlight shows mostly when looking towards the sun
  let toward = smoothstep(-0.3, 0.9, cosT);
  let ms = sunC * (0.025 + 0.22 * rim) * mix(1.0, Ts, 0.5) * alpha * (0.25 + 0.75 * toward);
  // bases are lit by the sky from the sides/below (horizon band + sea), cooler than the tops
  let amb = (skyAtmo(normalize(vec3f(dir.x, 0.03, dir.z))) * 0.26 + skyAtmo(vec3f(0.0, 1.0, 0.0)) * 0.42)
            * alpha * (1.0 - 0.35 * d);
  return vec4f(single + ms + amb, alpha);
}

// returns (premultiplied radiance, fraction of the background sky hidden)
// aaHi: high-deck edge change per pixel (fwidth); negative = estimate it from the footprint
fn cloudLayers(dir: vec3f, pixAng: f32, aaHi: f32, steps: i32, jit: f32, sunT: ptr<function, f32>) -> vec4f {
  *sunT = 1.0;
  if (dir.y <= 0.0) { return vec4f(0.0); }
  let sxz = F.sunDir.xz / max(length(F.sunDir.xz), 1e-4);
  let lo = lowLayer(dir, steps, jit);
  var hi = vec4f(0.0);
  var hiVis = 1.0;
  if (F.atmo.z > 0.001) {
    let hc = 4.5;
    let h = hitLayer(dir, hc);
    let far = 1.0 - smoothstep(30.0, 70.0, h.t);
    if (far > 0.0) {
      let fp = h.t * pixAng / h.mu;
      let e = highEdge(h.q, fp, 6);
      let d = clamp(e / highSoft(fp), 0.0, 1.0) * far;
      if (d > 0.002) {
        let s1 = highDensity(h.q + sxz * 0.15, fp, 4) * far;
        let s2 = highDensity(h.q + sxz * 0.5, fp, 3) * far;
        hi = shadeLayer(dir, h, hc, d, s1, s2, s2, F.atmo.w, 0.80);
        let aw = select(fp * 0.2 * 2.5, aaHi, aaHi >= 0.0);
        hi *= smoothstep(0.0, aw * 1.5 + 1e-6, e);
        hiVis = exp(-h.t * 0.010 * (0.6 + 0.4 * F.atmo.x));
      }
    }
  }
  *sunT = (1.0 - lo.a) * (1.0 - hi.a);
  // low layer in front; aerial perspective fades each layer into the haze without making
  // it transparent to the layer behind
  let col = lo.rgb + hi.rgb * hiVis * (1.0 - lo.a);
  let hidden = lo.a + hi.a * hiVis * (1.0 - lo.a);
  return vec4f(col, hidden);
}

fn sunDisk(dir: vec3f) -> vec3f {
  let c = dot(dir, F.sunDir.xyz);
  let ang = sqrt(max(2.0 * (1.0 - c), 0.0));
  let pix = F.screen.w * 2.0 * length(F.camUp.xyz) * 0.75;
  let edge = 1.0 - smoothstep(SUN_ANG_R - pix, SUN_ANG_R + pix, ang);
  if (edge <= 0.0) { return vec3f(0.0); }
  let x = min(ang / SUN_ANG_R, 1.0);
  let mu = sqrt(max(1.0 - x * x, 0.0));
  let limb = vec3f(1.0) - vec3f(0.397, 0.503, 0.652) * (1.0 - mu);
  let rad = F.sunDir.w / (PI * SUN_ANG_R * SUN_ANG_R);
  return rad * limb * edge * sampleTrans(camR(), max(dir.y, -0.01));
}

fn skyRadiance(dirIn: vec3f, withSun: bool, pixAng: f32) -> vec3f {
  var dir = dirIn;
  dir.y = max(dir.y, 0.0005);
  dir = normalize(dir);
  var L = skyAtmo(dir);
  var sunT = 1.0;
  let cl = cloudLayers(dir, pixAng, -1.0, 12, 0.5, &sunT);
  L = L * (1.0 - cl.a) + cl.rgb;
  if (withSun) { L += sunDisk(dir) * sunT; }
  return L;
}
`;

export const ENV_CUBE = FRAME_WGSL + ATMO_CORE_WGSL + /* wgsl */ `
@group(0) @binding(0) var<uniform> F: Frame;
@group(0) @binding(1) var transLUT: texture_2d<f32>;
@group(0) @binding(2) var skyLUT: texture_2d<f32>;
@group(0) @binding(3) var sampC: sampler;
@group(0) @binding(6) var dst: texture_storage_2d_array<rgba16float, write>;
` + SKY_RENDER_FUNCS + SKY_CLOUD_FUNCS + /* wgsl */ `
const ENV: f32 = ${ENV_SIZE}.0;
fn faceDir(face: u32, uv: vec2f) -> vec3f {
  let u = uv.x * 2.0 - 1.0;
  let v = uv.y * 2.0 - 1.0;
  switch (face) {
    case 0u: { return vec3f(1.0, -v, -u); }
    case 1u: { return vec3f(-1.0, -v, u); }
    case 2u: { return vec3f(u, 1.0, v); }
    case 3u: { return vec3f(u, -1.0, -v); }
    case 4u: { return vec3f(u, -v, 1.0); }
    default: { return vec3f(-u, -v, -1.0); }
  }
}
@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) id: vec3u) {
  if (f32(id.x) >= ENV || f32(id.y) >= ENV) { return; }
  // faces are refreshed round-robin: face = (frame + z) % 6
  let face = (u32(F.post.z) + id.z) % 6u;
  let dir = normalize(faceDir(face, (vec2f(id.xy) + 0.5) / ENV));
  var L: vec3f;
  if (dir.y >= 0.0) {
    L = skyRadiance(dir, false, 1.5708 / ENV * 1.5);
  } else {
    // below the horizon: what a reflected ray would see -> far, hazy ocean
    let hz = skyRadiance(normalize(vec3f(dir.x, 0.004, dir.z)), false, 1.5708 / ENV * 1.5);
    let body = F.waterCol.rgb * (sunIlluminance() * max(F.sunDir.y, 0.0) + skyAtmo(vec3f(0.0, 1.0, 0.0)) * 2.0);
    L = mix(hz * 0.75, body, clamp(-dir.y * 6.0, 0.0, 1.0));
  }
  textureStore(dst, id.xy, face, vec4f(L, 1.0));
}
`;

// Mips are rebuilt only for the faces the env pass refreshed this frame (dispatch z = their
// count): face = (frame + z) % 6, the same rotation as ENV_CUBE. Each face's mips depend only
// on that face, so the others are already up to date.
export const CUBE_MIPS = FRAME_WGSL + /* wgsl */ `
@group(0) @binding(0) var src: texture_2d_array<f32>;
@group(0) @binding(1) var dst: texture_storage_2d_array<rgba16float, write>;
@group(0) @binding(2) var<uniform> F: Frame;
@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) id: vec3u) {
  let size = textureDimensions(dst);
  if (id.x >= size.x || id.y >= size.y) { return; }
  let face = (u32(F.post.z) + id.z) % 6u;
  let srcSize = vec2i(textureDimensions(src));
  let p = vec2i(id.xy) * 2;
  // 4x4 tent-ish filter for smoother roughness blur
  var acc = vec4f(0.0);
  var wsum = 0.0;
  for (var j = -1; j <= 2; j++) {
    for (var i = -1; i <= 2; i++) {
      let q = clamp(p + vec2i(i, j), vec2i(0), srcSize - 1);
      let w = select(1.0, 3.0, i == 0 || i == 1) * select(1.0, 3.0, j == 0 || j == 1);
      acc += textureLoad(src, q, face, 0) * w;
      wsum += w;
    }
  }
  textureStore(dst, id.xy, face, acc / wsum);
}
`;

// Cloud transmittance towards the sun (averaged over the solar disc), used to dim
// direct sunlight on the water when the sun is behind clouds. Runs after the env cube update
// and also stores the frame's lighting constants for the ocean shader (Light).
export const SUN_VIS = FRAME_WGSL + ATMO_CORE_WGSL + LIGHT_WGSL + /* wgsl */ `
@group(0) @binding(0) var<uniform> F: Frame;
@group(0) @binding(1) var transLUT: texture_2d<f32>;
@group(0) @binding(2) var skyLUT: texture_2d<f32>;
@group(0) @binding(3) var sampC: sampler;
@group(0) @binding(6) var<storage, read_write> light: Light;
@group(0) @binding(7) var envCube: texture_cube<f32>;
` + SKY_RENDER_FUNCS + SKY_CLOUD_FUNCS + /* wgsl */ `
const ENV_MAXMIP: f32 = ${ENV_MIPS - 1}.0;
var<workgroup> part: array<f32, 8>;
// one ray per thread (7 rays across the disc), then a tiny reduction
@compute @workgroup_size(8)
fn main(@builtin(local_invocation_index) k: u32) {
  let sd = F.sunDir.xyz;
  let tx = normalize(cross(sd, vec3f(0.0, 1.0, 0.0)));
  let ty = cross(tx, sd);
  var t = 1.0;
  if (k < 7u) {
    let a = f32(k) * 0.8975979;
    let o = select(vec2f(cos(a), sin(a)) * 0.65, vec2f(0.0), k == 0u);
    let dir = normalize(sd + (tx * o.x + ty * o.y) * SUN_ANG_R);
    let cl = cloudLayers(dir, 0.0, -1.0, 12, 0.5, &t);
  }
  part[k] = t;
  workgroupBarrier();
  if (k == 0u) {
    var acc = 0.0;
    for (var i = 0u; i < 7u; i++) { acc += part[i]; }
    let vis = acc / 7.0;
    light.sunVis = vec4f(vis, 0.0, 0.0, 0.0);
    light.sunE = vec4f(sunIlluminance() * vis, 0.0);
    let skyUp = textureSampleLevel(envCube, sampC, vec3f(0.0, 1.0, 0.0), ENV_MAXMIP).rgb;
    light.skyUp = vec4f(skyUp, 0.0);
    // average radiance of the whole upper hemisphere (incl. the bright horizon band)
    light.skyHemi = vec4f(skyUp * 0.4 + 0.15 * (
        textureSampleLevel(envCube, sampC, vec3f(0.94, 0.34, 0.0), ENV_MAXMIP - 1.0).rgb +
        textureSampleLevel(envCube, sampC, vec3f(-0.94, 0.34, 0.0), ENV_MAXMIP - 1.0).rgb +
        textureSampleLevel(envCube, sampC, vec3f(0.0, 0.34, 0.94), ENV_MAXMIP - 1.0).rgb +
        textureSampleLevel(envCube, sampC, vec3f(0.0, 0.34, -0.94), ENV_MAXMIP - 1.0).rgb), 0.0);
  }
}
`;

export const NOISE3D = NOISE_WGSL + /* wgsl */ `
@group(0) @binding(0) var dst: texture_storage_3d<rgba8unorm, write>;
fn n3(p: vec3f, per: i32) -> f32 {
  let i = vec3i(floor(p));
  let f = p - floor(p);
  let u = f * f * f * (f * (f * 6.0 - 15.0) + 10.0);
  var s = 0.0;
  for (var k = 0; k < 8; k++) {
    let o = vec3i(k & 1, (k >> 1) & 1, k >> 2);
    let c = ((i + o) % per + per) % per;
    let h = pcgHash(u32(c.x) + pcgHash(u32(c.y) + pcgHash(u32(c.z))));
    let g = vec3f(f32(h & 1023u), f32((h >> 10u) & 1023u), f32((h >> 20u) & 1023u)) / 511.5 - 1.0;
    let w = mix(1.0 - u, u, vec3f(o));
    s += dot(g, f - vec3f(o)) * w.x * w.y * w.z;
  }
  return s;
}
// tileable cellular noise: distance to the nearest feature point (0 at the points)
fn worley(p: vec3f, per: i32) -> f32 {
  let i = vec3i(floor(p));
  var m = 1.0;
  for (var k = 0; k < 27; k++) {
    let o = vec3i(k % 3, (k / 3) % 3, k / 9) - 1;
    let c = ((i + o) % per + per) % per;
    let h = pcgHash(u32(c.x) + pcgHash(u32(c.y) + pcgHash(u32(c.z) + 77u)));
    let f = vec3f(f32(h & 1023u), f32((h >> 10u) & 1023u), f32((h >> 20u) & 1023u)) / 1023.0;
    m = min(m, length(vec3f(i + o) + f - p));
  }
  return m;
}
fn wfbm(p: vec3f, per: i32) -> f32 {
  return (1.0 - worley(p, per)) * 0.625 + (1.0 - worley(p * 2.0, per * 2)) * 0.25 + (1.0 - worley(p * 4.0, per * 4)) * 0.125;
}
fn fbm3(p: vec3f, per: i32, oct: i32, billow: f32) -> f32 {
  var s = 0.0; var a = 0.5; var q = p; var pr = per;
  for (var o = 0; o < oct; o++) {
    let n = n3(q, pr);
    s += a * mix(n, abs(n) * 2.0 - 0.3, billow);
    q *= 2.0; pr *= 2; a *= 0.5;
  }
  return s;
}
@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) id: vec3u) {
  let p = vec3f(id) / 128.0;
  textureStore(dst, id, vec4f(mix(fbm3(p * 4.0, 4, 5, 0.0) + 0.5, wfbm(p * 4.0, 4), 0.55), wfbm(p * 8.0, 8), 0.0, 1.0));
}
`;
