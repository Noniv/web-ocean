// Shared constants and WGSL snippets.
export const N = 256;          // FFT resolution per cascade
export const LOGN = 8;
export const CASCADES = 6;
export const SKY_W = 256;      // sky-view LUT size
export const SKY_H = 144;
export const ENV_SIZE = 256;   // environment cube face size
export const ENV_MIPS = 9;

// Per-frame uniform block shared by every render / sky shader.
// Layout must match the Float32Array offsets in main.js (FR_* constants).
export const FRAME_WGSL = /* wgsl */ `
struct Frame {
  viewProj: mat4x4f,   // camera-relative view-projection (reverse-Z, infinite)
  camPos:   vec4f,     // xyz metres, w = time (s)
  camRight: vec4f,     // xyz * tan(fovX/2)
  camUp:    vec4f,     // xyz * tan(fovY/2)
  camFwd:   vec4f,     // xyz
  sunDir:   vec4f,     // xyz towards sun, w = solar illuminance scale
  screen:   vec4f,     // w, h, 1/w, 1/h
  cascL:    array<vec4f, 2>,  // cascade patch sizes (m), index [c>>2][c&3]
  cascInvL: array<vec4f, 2>,
  windP:    array<vec4f, 2>,  // [0].xy = wind direction (unit), [0].z = wind speed
  lodP:     vec4f,     // CDLOD base range, grid quads per node, -, -
  post:     vec4f,     // exposure, bloom strength, frame index, -
  atmo:     vec4f,     // haze (mie multiplier), camera altitude km, -, -
  waterCol: vec4f,     // body scatter colour rgb, unresolved slope variance
  sssCol:   vec4f,     // sss colour rgb, sss strength
  foamP:    vec4f,     // J threshold, J gain, foam intensity, lace sharpness
  fogExt:   vec4f,     // rgb extinction per metre near sea level
  cloudP:   vec4f,     // altitude km, coverage, optical density, wind time
};
`;

// Per-frame lighting constants, written once per frame by the sun-visibility pass (after the
// env cube update) so the ocean shader reads them instead of re-deriving them per pixel.
export const LIGHT_WGSL = /* wgsl */ `
struct Light {
  sunVis:  vec4f,   // x = cloud transmittance towards the sun
  sunE:    vec4f,   // rgb = sun illuminance at the camera * sunVis
  skyUp:   vec4f,   // rgb = env cube zenith radiance (coarsest mip)
  skyHemi: vec4f,   // rgb = average upper-hemisphere radiance
};
`;

export const ATMO_CORE_WGSL = /* wgsl */ `
const PI: f32 = 3.14159265358979;
const RG: f32 = 6360.0;
const RT: f32 = 6460.0;
const RAY_S: vec3f = vec3f(5.802e-3, 13.558e-3, 33.1e-3);
const MIE_S: f32 = 3.996e-3;
const MIE_E: f32 = 4.40e-3;
const OZO_A: vec3f = vec3f(0.650e-3, 1.881e-3, 0.085e-3);
const SUN_ANG_R: f32 = 0.004675;
const TRANS_W: f32 = 256.0;
const TRANS_H: f32 = 64.0;
const MS_SIZE: f32 = 32.0;

struct Medium { ext: vec3f, rayS: vec3f, mieS: f32 };

fn medium(h: f32, haze: f32) -> Medium {
  let hh = max(h, 0.0);
  let dR = exp(-hh / 8.0);
  // aerosols: regular 1.2 km layer plus a dense marine boundary layer
  let dM = exp(-hh / 1.2);
  let dO = max(0.0, 1.0 - abs(hh - 25.0) / 15.0);
  var m: Medium;
  m.rayS = RAY_S * dR;
  m.mieS = MIE_S * dM * haze;
  m.ext = m.rayS + vec3f(MIE_E * dM * haze) + OZO_A * dO;
  return m;
}

fn raySphere(ro: vec3f, rd: vec3f, R: f32) -> vec2f {
  let l = length(ro);
  let b = dot(ro, rd);
  let c = (l - R) * (l + R);
  let d = b * b - c;
  if (d < 0.0) { return vec2f(-1.0, -1.0); }
  let s = sqrt(d);
  return vec2f(-b - s, -b + s);
}

fn rhoOf(r: f32) -> f32 { return sqrt(max(0.0, (r - RG) * (r + RG))); }

fn transRMuToUV(r: f32, mu: f32) -> vec2f {
  let H = sqrt(RT * RT - RG * RG);
  let rho = rhoOf(r);
  let disc = r * r * (mu * mu - 1.0) + RT * RT;
  let d = max(0.0, -r * mu + sqrt(max(disc, 0.0)));
  let dMin = RT - r;
  let dMax = rho + H;
  let x = (d - dMin) / (dMax - dMin);
  let y = rho / H;
  return vec2f(x, y);
}

fn transUVToRMu(uv: vec2f) -> vec2f {
  let H = sqrt(RT * RT - RG * RG);
  let rho = H * uv.y;
  let r = sqrt(rho * rho + RG * RG);
  let dMin = RT - r;
  let dMax = rho + H;
  let d = dMin + uv.x * (dMax - dMin);
  var mu = 1.0;
  if (d > 0.0) { mu = (H * H - rho * rho - d * d) / (2.0 * r * d); }
  return vec2f(r, clamp(mu, -1.0, 1.0));
}

fn fromUnitToSubUvs(u: f32, res: f32) -> f32 { return (u + 0.5 / res) * (res / (res + 1.0)); }
fn fromSubUvsToUnit(u: f32, res: f32) -> f32 { return (u - 0.5 / res) * (res / (res - 1.0)); }

fn phaseRayleigh(c: f32) -> f32 { return 3.0 / (16.0 * PI) * (1.0 + c * c); }
fn phaseMie(c: f32, g: f32) -> f32 {
  let g2 = g * g;
  return 3.0 / (8.0 * PI) * ((1.0 - g2) * (1.0 + c * c)) / ((2.0 + g2) * pow(max(1.0 + g2 - 2.0 * g * c, 1e-4), 1.5));
}
fn phaseHG(c: f32, g: f32) -> f32 {
  let g2 = g * g;
  return (1.0 - g2) / (4.0 * PI * pow(max(1.0 + g2 - 2.0 * g * c, 1e-4), 1.5));
}

// true when the sun is below the geometric horizon of a point at radius r
fn sunBelowHorizon(r: f32, mu: f32) -> bool {
  return mu < -rhoOf(r) / r;
}
`;

// Hashing helpers used by the procedural texture generators.
export const NOISE_WGSL = /* wgsl */ `
fn pcgHash(v: u32) -> u32 {
  let s = v * 747796405u + 2891336453u;
  let w = ((s >> ((s >> 28u) + 4u)) ^ s) * 277803737u;
  return (w >> 22u) ^ w;
}
fn hashGrad(i: vec2i) -> vec2f {
  let h = pcgHash(u32(i.x) * 1664525u + pcgHash(u32(i.y) + 1013904223u));
  let a = f32(h) * (6.28318530718 / 4294967296.0);
  return vec2f(cos(a), sin(a));
}
fn gnoise(p: vec2f) -> f32 {
  let i = vec2i(floor(p));
  let f = p - floor(p);
  let u = f * f * f * (f * (f * 6.0 - 15.0) + 10.0);
  let a = dot(hashGrad(i), f);
  let b = dot(hashGrad(i + vec2i(1, 0)), f - vec2f(1.0, 0.0));
  let c = dot(hashGrad(i + vec2i(0, 1)), f - vec2f(0.0, 1.0));
  let d = dot(hashGrad(i + vec2i(1, 1)), f - vec2f(1.0, 1.0));
  return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
}
`;

// Periodic (tileable) noise for procedural textures. Requires NOISE_WGSL.
export const PERIODIC_WGSL = /* wgsl */ `
fn pnoise(p: vec2f, period: i32) -> f32 {
  let i = vec2i(floor(p));
  let f = p - floor(p);
  let u = f * f * f * (f * (f * 6.0 - 15.0) + 10.0);
  let i00 = (i % period + period) % period;
  let i10 = ((i + vec2i(1, 0)) % period + period) % period;
  let i01 = ((i + vec2i(0, 1)) % period + period) % period;
  let i11 = ((i + vec2i(1, 1)) % period + period) % period;
  let a = dot(hashGrad(i00), f);
  let b = dot(hashGrad(i10), f - vec2f(1.0, 0.0));
  let c = dot(hashGrad(i01), f - vec2f(0.0, 1.0));
  let d = dot(hashGrad(i11), f - vec2f(1.0, 1.0));
  return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
}
fn pfbm(p: vec2f, period: i32, oct: i32) -> f32 {
  var s = 0.0; var a = 0.5; var q = p; var per = period;
  for (var o = 0; o < oct; o++) {
    s += a * pnoise(q, per);
    q *= 2.0; per *= 2; a *= 0.5;
  }
  return s;
}
`;
