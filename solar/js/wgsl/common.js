// Shared constants and WGSL snippets.
//
// Units: every distance on the GPU is in megametres (Mm = 1e6 m) and every position is
// CAMERA-RELATIVE. The CPU keeps absolute positions in f64 and subtracts the camera before
// upload. That is what lets a single f32 renderer span metre-scale surface relief and 30 AU
// of empty space without depth or precision artefacts.
export const MM = 1e6;                      // metres per Mm
export const AU = 149597870.7 / 1e3;        // Mm per AU (the literal is in km)

export const SURF_W = 1024;                 // equirectangular surface map size
export const SURF_H = 512;
export const SURF_MIPS = 11;                // 1024 -> 1
export const RING_RES = 4096;               // radial samples of the ring optical depth
export const RING_MIPS = 13;                // 4096 -> 1
export const MAX_BODIES = 24;
export const BODY_STRIDE = 36;              // floats per Body (9 * vec4f)
export const TRANS_W = 256;                 // atmosphere transmittance LUT
export const TRANS_H = 64;
export const MS_SIZE = 32;                  // multiple-scattering LUT
export const SKY_W = 192;                   // sky-view LUT
export const SKY_H = 108;
export const MAX_STARS = 12000;

// Body kinds (must match the k* constants in the WGSL below)
export const K_ROCKY = 0;   // mapped surface + LEAN slope moments
export const K_GAS = 1;     // analytic banded zonal flow (Jupiter, Saturn)
export const K_ICE = 2;     // analytic smooth bands (Uranus, Neptune)
export const K_SHROUD = 3;  // opaque haze deck, no surface (Venus, Titan)

// ---------------------------------------------------------------------------------------
// Per-frame uniform block. Float offsets must match writeFrame() in main.js.
// ---------------------------------------------------------------------------------------
export const FRAME_WGSL = /* wgsl */ `
struct Frame {
  viewProj: mat4x4f,  //  0  camera-relative view-projection (reverse-Z, infinite far)
  camRight: vec4f,    // 16  xyz * tan(fovX/2)
  camUp:    vec4f,    // 20  xyz * tan(fovY/2)
  camFwd:   vec4f,    // 24  xyz, w = time (s)
  screen:   vec4f,    // 28  w, h, 1/w, 1/h
  post:     vec4f,    // 32  exposure, bloom, frame index, debug view
  sun:      vec4f,    // 36  camera-relative sun centre (Mm), w = sun radius (Mm)
  sunRad:   vec4f,    // 40  solar radiance scale rgb, w = pixel angular size (rad)
  flags:    vec4f,    // 44  active-atmosphere body index, atmo on, -, ozone-profile flag
  atmoR:    vec4f,    // 48  ground radius, top radius, H_rayleigh, H_mie (active body, Mm)
  atmoS:    vec4f,    // 52  rayleigh scattering rgb / Mm, w = mie phase g
  atmoM:    vec4f,    // 56  mie scattering rgb / Mm, w = mie absorption / Mm
  atmoO:    vec4f,    // 60  extra absorber rgb / Mm (ozone or dust), w = ground albedo
  ringP:    vec4f,    // 64  inner radius, outer radius (body radii), opacity, brightness
  ringQ:    vec4f,    // 68  particle albedo, backscatter g, forward g, opposition amplitude
  starP:    vec4f,    // 72  star gain, milky-way gain, zodiacal gain, -
  camPos:   vec4f,    // 76  absolute camera position in Mm (f32, display only), w = speed
};
`;

// ---------------------------------------------------------------------------------------
// One entry per rendered body. Camera-relative, uploaded every frame.
// ---------------------------------------------------------------------------------------
export const BODY_WGSL = /* wgsl */ `
const kRocky:  u32 = 0u;
const kGas:    u32 = 1u;
const kIce:    u32 = 2u;
const kShroud: u32 = 3u;

struct Body {
  rel:   vec4f,  // camera-relative centre (Mm), w = equatorial radius (Mm)
  axis:  vec4f,  // spin axis (unit), w = polar radius (Mm)
  east:  vec4f,  // prime-meridian east direction (unit), w = f32(kind)
  tint:  vec4f,  // albedo tint rgb, w = surface map layer (<0 = analytic only)
  atmo:  vec4f,  // shell thickness (Mm), density gain, haze gain, 1 = has full LUT
  mat:   vec4f,  // base roughness, sea level (map height units), ice latitude, cloud amount
  ring:  vec4f,  // 1 = has rings, shadow strength, -, -
  misc:  vec4f,  // sphere coverage weight (0 = too small, splatted instead), irradiance, -, -
  shell: vec4f,  // analytic-atmosphere scattering colour rgb per Mm, -
};
`;

// ---------------------------------------------------------------------------------------
// Geometry: numerically stable ray/sphere and ray/ellipsoid tests.
// ---------------------------------------------------------------------------------------
export const GEOM_WGSL = /* wgsl */ `
const PI: f32 = 3.14159265358979;

// ro is the ray origin relative to the sphere centre. Returns both roots, or (-1,-1).
// (l-R)*(l+R) instead of l*l-R*R keeps the precision when the camera sits just above a
// very large sphere, where the two terms would otherwise cancel.
fn raySphere(ro: vec3f, rd: vec3f, R: f32) -> vec2f {
  let l = length(ro);
  let b = dot(ro, rd);
  let c = (l - R) * (l + R);
  let d = b * b - c;
  if (d < 0.0) { return vec2f(-1.0, -1.0); }
  let s = sqrt(d);
  return vec2f(-b - s, -b + s);
}

// Oblate spheroid with polar axis n: squash along n and reuse the sphere test.
fn rayOblate(ro: vec3f, rd: vec3f, n: vec3f, Req: f32, Rpo: f32) -> vec2f {
  let k = Req / max(Rpo, 1e-6);
  let o = ro + n * (dot(ro, n) * (k - 1.0));
  let d = rd + n * (dot(rd, n) * (k - 1.0));
  let dl = length(d);
  let t = raySphere(o, d / dl, Req);
  return select(t / dl, t, t.y < 0.0);
}

// Surface normal of that spheroid at a point p relative to the centre.
fn oblateNormal(p: vec3f, n: vec3f, Req: f32, Rpo: f32) -> vec3f {
  let k = Req / max(Rpo, 1e-6);
  let along = dot(p, n);
  return normalize(p + n * (along * (k * k - 1.0)));
}

// Smooth coverage of a disc of angular radius ar seen at angular offset ao, for a pixel of
// angular size px. Analytic silhouette anti-aliasing: no MSAA needed for the limb.
fn discCoverage(ao: f32, ar: f32, px: f32) -> f32 {
  return clamp(0.5 + (ar - ao) / max(px, 1e-12), 0.0, 1.0);
}

fn phaseHG(c: f32, g: f32) -> f32 {
  let g2 = g * g;
  return (1.0 - g2) / (4.0 * PI * pow(max(1.0 + g2 - 2.0 * g * c, 1e-4), 1.5));
}
fn phaseRayleigh(c: f32) -> f32 { return 3.0 / (16.0 * PI) * (1.0 + c * c); }

// Reflected irradiance of a Lambertian sphere at phase angle a (radians).
fn lambertSpherePhase(a: f32) -> f32 {
  return (sin(a) + (PI - a) * cos(a)) / PI;
}

fn equirectUV(dir: vec3f, axis: vec3f, east: vec3f) -> vec2f {
  let north = cross(axis, east);
  let lat = asin(clamp(dot(dir, axis), -1.0, 1.0));
  let lon = atan2(dot(dir, north), dot(dir, east));
  return vec2f(lon * (0.5 / PI) + 0.5, 0.5 - lat / PI);
}
`;

// ---------------------------------------------------------------------------------------
// Hashing and 3-D gradient noise. Surface maps are generated from noise evaluated on the
// unit sphere in 3-D, so there is no equirectangular seam and no pole pinching.
// ---------------------------------------------------------------------------------------
export const NOISE_WGSL = /* wgsl */ `
fn pcgHash(v: u32) -> u32 {
  let s = v * 747796405u + 2891336453u;
  let w = ((s >> ((s >> 28u) + 4u)) ^ s) * 277803737u;
  return (w >> 22u) ^ w;
}
fn hash3(i: vec3i) -> u32 {
  return pcgHash(u32(i.x) * 1664525u + pcgHash(u32(i.y) * 22695477u + pcgHash(u32(i.z) + 1013904223u)));
}
fn hash1f(h: u32) -> f32 { return f32(h) * (1.0 / 4294967296.0); }

fn grad3(i: vec3i) -> vec3f {
  let h = hash3(i);
  // two hashed angles -> uniform direction on the sphere
  let z = hash1f(h) * 2.0 - 1.0;
  let a = hash1f(pcgHash(h)) * 6.28318530718;
  let r = sqrt(max(0.0, 1.0 - z * z));
  return vec3f(cos(a) * r, sin(a) * r, z);
}

fn gnoise3(p: vec3f) -> f32 {
  let i = vec3i(floor(p));
  let f = p - floor(p);
  let u = f * f * f * (f * (f * 6.0 - 15.0) + 10.0);
  var s = 0.0;
  for (var k = 0; k < 2; k++) {
    for (var j = 0; j < 2; j++) {
      for (var o = 0; o < 2; o++) {
        let c = vec3i(o, j, k);
        let w = vec3f(f32(o), f32(j), f32(k));
        let wx = mix(1.0 - u.x, u.x, w.x);
        let wy = mix(1.0 - u.y, u.y, w.y);
        let wz = mix(1.0 - u.z, u.z, w.z);
        s += wx * wy * wz * dot(grad3(i + c), f - w);
      }
    }
  }
  return s * 1.6;
}

fn fbm3(p: vec3f, oct: i32, lac: f32, gain: f32) -> f32 {
  var s = 0.0; var a = 0.5; var q = p;
  for (var o = 0; o < oct; o++) {
    s += a * gnoise3(q);
    q *= lac; a *= gain;
  }
  return s;
}

// Ridged multifractal: sharp crests, used for mountain belts and icy grooves.
fn ridged3(p: vec3f, oct: i32) -> f32 {
  var s = 0.0; var a = 0.5; var q = p; var prev = 1.0;
  for (var o = 0; o < oct; o++) {
    let n = 1.0 - abs(gnoise3(q));
    let v = n * n * prev;
    s += a * v;
    prev = v;
    q *= 2.04; a *= 0.5;
  }
  return s;
}

// Band-limited fBm: octaves whose period falls below the pixel footprint are faded out
// instead of sampled, and the caller gets back the variance they would have contributed.
// This is the same bargain as the LEAN moments -- unresolved detail becomes a statistic.
struct BandFbm { v: f32, var2: f32 };
fn fbm3Limited(p: vec3f, oct: i32, foot: f32) -> BandFbm {
  var s = 0.0; var a = 0.5; var q = p; var vr = 0.0; var scale = 1.0;
  for (var o = 0; o < oct; o++) {
    // one period of this octave spans 1/scale of the input; fade when the pixel covers it
    let w = 1.0 - smoothstep(0.35, 1.1, foot * scale);
    if (w > 0.002) { s += a * w * gnoise3(q); }
    vr += a * a * (1.0 - w * w) * 0.5;
    q *= 2.03; a *= 0.5; scale *= 2.03;
  }
  var o: BandFbm;
  o.v = s; o.var2 = vr;
  return o;
}
`;

// ---------------------------------------------------------------------------------------
// Crater fields. A jittered 3-D grid of craters with rim / floor / ejecta profiles, summed
// over several size octaves. Only used by the one-off surface generation pass.
// ---------------------------------------------------------------------------------------
export const CRATER_WGSL = /* wgsl */ `
// Radial profile of a single crater of unit radius: bowl, raised rim, ejecta skirt.
fn craterProfile(d: f32) -> f32 {
  if (d > 2.2) { return 0.0; }
  let bowl = -1.0 + d * d;                       // parabolic floor, rises to the rim
  let rim = exp(-pow((d - 1.0) * 3.4, 2.0)) * 0.55;
  let ejecta = exp(-pow((d - 1.25) * 1.5, 2.0)) * 0.12;
  return select(rim + ejecta, bowl * 0.55 + rim, d < 1.0);
}

// One octave: cells of unit size, one candidate crater per cell, 3x3x3 neighbourhood.
fn craterOctave(p: vec3f, dens: f32, seed: u32) -> f32 {
  let b = floor(p);
  var h = 0.0;
  for (var k = -1; k <= 1; k++) {
    for (var j = -1; j <= 1; j++) {
      for (var i = -1; i <= 1; i++) {
        let c = vec3i(b) + vec3i(i, j, k);
        let hh = hash3(c + vec3i(i32(seed)));
        if (hash1f(hh) > dens) { continue; }
        let jx = hash1f(pcgHash(hh));
        let jy = hash1f(pcgHash(hh ^ 0x9e3779b9u));
        let jz = hash1f(pcgHash(hh ^ 0x85ebca6bu));
        let rr = 0.25 + 0.55 * hash1f(pcgHash(hh ^ 0xc2b2ae35u));
        let ctr = vec3f(c) + vec3f(jx, jy, jz);
        h += craterProfile(length(p - ctr) / rr) * rr;
      }
    }
  }
  return h;
}

// Each octave's craters are 2.37x smaller, so for depth to stay a sensible fraction of
// diameter the amplitude must fall more slowly than the size does. At 0.46 the small
// craters came out a hundred times too shallow to shade at all; 0.70 keeps depth/diameter
// roughly in the real 0.02 (basins) to 0.2 (simple craters) range across the whole field.
fn craterField(p: vec3f, oct: i32, dens: f32) -> f32 {
  var s = 0.0; var a = 1.0; var q = p;
  for (var o = 0; o < oct; o++) {
    s += a * craterOctave(q, dens, u32(o) * 7919u);
    q *= 2.37; a *= 0.70;
  }
  return s;
}
`;

// Fullscreen-triangle vertex stage shared by every screen-space pass.
export const FS_TRI = /* wgsl */ `
struct VO { @builtin(position) pos: vec4f, @location(0) uv: vec2f };
@vertex fn vs(@builtin(vertex_index) i: u32) -> VO {
  let p = vec2f(f32((i << 1u) & 2u), f32(i & 2u));
  var o: VO;
  o.pos = vec4f(p * 2.0 - 1.0, 0.0, 1.0);
  o.uv = vec2f(p.x, 1.0 - p.y);
  return o;
}
`;

// Camera ray for a pixel, plus the two neighbouring rays. Every texture footprint in the
// renderer is derived from these analytically, because dpdx/dpdy may not be called inside
// the data-dependent branches that the body loop is made of.
export const RAYGEN_WGSL = /* wgsl */ `
struct Rays { d: vec3f, dx: vec3f, dy: vec3f, px: f32 };
fn makeRays(uv: vec2f) -> Rays {
  let n = vec2f(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);
  let s = vec2f(2.0 * F.screen.z, 2.0 * F.screen.w);
  var o: Rays;
  o.d = normalize(F.camFwd.xyz + F.camRight.xyz * n.x + F.camUp.xyz * n.y);
  o.dx = normalize(F.camFwd.xyz + F.camRight.xyz * (n.x + s.x) + F.camUp.xyz * n.y);
  o.dy = normalize(F.camFwd.xyz + F.camRight.xyz * n.x + F.camUp.xyz * (n.y + s.y));
  o.px = F.sunRad.w;
  return o;
}
`;
