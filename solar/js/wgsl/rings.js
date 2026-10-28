// Saturn's rings.
//
// This is the purest case of the idea the whole renderer is built on. The rings are of order
// 1e14 particles, so there was never a question of drawing geometry: what has to be rendered
// is the *distribution*. Two things follow.
//
// 1. The radial optical depth profile has real structure at every scale, down to the ~35 km
//    Keeler gap -- about one texel of the 4096-sample profile, and far below one pixel for
//    almost any camera. Sampling it point-wise produces the shimmering mess you see in most
//    real-time Saturns. So the profile is stored as a mip pyramid of the first *and second*
//    moments of tau, and the shader picks the level from the pixel's radial footprint. The
//    transmittance is then reconstructed from both moments, because exp() of an average is
//    not the average of exp(): E[e^-x] ~= e^-E[x] * (1 + Var[x]/2). That second-moment
//    correction is the direct analogue of LEAN mapping -- same trick, different integrand.
//
// 2. At grazing angles the footprint explodes, which automatically drives the lookup to the
//    coarse levels. The rings fade into a correct smooth sheen instead of aliasing, and at
//    exactly edge-on the path length through the layer diverges, so they collapse to the
//    thin bright line they actually are.
//
// The scattering itself is a Chandrasekhar single-scattering slab with a two-term
// Henyey-Greenstein particle phase function (rings are strongly backscattering), a
// saturating multiple-scattering term for the optically thick B ring, the opposition surge,
// and both analytic shadows: the planet on the rings and the rings on the planet.
import { RING_RES, NOISE_WGSL } from './common.js';

const R0 = 1.0;    // inner edge of the stored profile, in equatorial body radii
const R1 = 3.2;    // outer edge (just past the F ring)

export const RING_R0 = R0;
export const RING_R1 = R1;

// --------------------------------------------------------------------------------------
// Optical depth profile. Radii are in Saturn equatorial radii (R = 60268 km).
// --------------------------------------------------------------------------------------
export const RING_GEN = NOISE_WGSL + /* wgsl */ `
@group(0) @binding(0) var tauOut: texture_storage_2d<rgba16float, write>;

const RES: f32 = ${RING_RES}.0;
const RA: f32 = ${R0};
const RB: f32 = ${R1};

fn n1(x: f32, s: f32) -> f32 { return gnoise3(vec3f(x, s, 0.0)); }
fn fbm1(x: f32, oct: i32, s: f32) -> f32 {
  var v = 0.0; var a = 0.5; var q = x;
  for (var o = 0; o < oct; o++) { v += a * n1(q, s); q *= 2.07; a *= 0.5; }
  return v;
}

// A soft-edged band. Edges are a few hundred km wide for most ring boundaries, which is
// what the smoothstep widths encode.
fn band(r: f32, a: f32, b: f32, edge: f32) -> f32 {
  return smoothstep(a - edge, a + edge, r) * (1.0 - smoothstep(b - edge, b + edge, r));
}
fn gap(r: f32, c: f32, halfW: f32, edge: f32) -> f32 {
  return 1.0 - band(r, c - halfW, c + halfW, edge);
}

fn tauAt(r: f32) -> f32 {
  var t = 0.0;
  // D ring: very tenuous, inside the C ring
  t += band(r, 1.110, 1.236, 0.004) * 0.012;
  // C ring: optically thin, with bright plateaus
  let c = band(r, 1.239, 1.526, 0.003);
  var tc = 0.075 + 0.055 * smoothstep(-0.2, 0.5, fbm1(r * 160.0, 4, 1.0));
  tc += 0.11 * band(r, 1.35, 1.40, 0.006) + 0.09 * band(r, 1.46, 1.50, 0.006);
  t += c * tc;
  // B ring: optically thick, strongly structured; the densest part of the system
  let b = band(r, 1.526, 1.950, 0.0025);
  var tb = 1.45 + 0.95 * fbm1(r * 90.0, 5, 7.0);
  tb *= 1.0 - 0.35 * smoothstep(0.80, 1.00, (r - 1.526) / 0.424);   // thins at the outer edge
  tb *= 0.55 + 0.45 * smoothstep(0.0, 0.18, (r - 1.526) / 0.424);   // ramps up from the inner edge
  t += b * max(tb, 0.25);
  // Cassini Division: not empty, a few ringlets in a low-tau trough
  let cd = band(r, 1.950, 2.025, 0.0022);
  t += cd * (0.055 + 0.075 * smoothstep(0.55, 0.95, fbm1(r * 420.0, 3, 13.0) + 0.35));
  // A ring, with the Encke and Keeler gaps
  let a = band(r, 2.025, 2.269, 0.0022);
  var ta = 0.52 + 0.14 * fbm1(r * 150.0, 4, 23.0);
  ta *= 1.0 - 0.28 * smoothstep(0.75, 1.0, (r - 2.025) / 0.244);
  // spiral density waves inboard of the strong outer resonances: a chirped wave train
  let dw = smoothstep(2.21, 2.025, r) * smoothstep(2.025, 2.07, r);
  ta *= 1.0 + 0.30 * dw * sin((r - 2.025) * 900.0 - pow(max(r - 2.025, 0.0) * 6.0, 1.6) * 60.0);
  // Encke gap, 325 km wide; Keeler gap, 35 km wide -- about one texel of this profile
  ta *= gap(r, 2.2165, 0.00270, 0.00060);
  ta *= gap(r, 2.2648, 0.00029, 0.00022);
  t += a * ta;
  // F ring: a narrow, clumpy strand held by shepherd moons
  let fw = exp(-pow((r - 2.324) / 0.0010, 2.0));
  t += fw * (0.10 + 0.09 * fbm1(r * 2600.0, 3, 31.0));
  // G and inner E ring: barely there, but they catch forward-scattered light
  t += band(r, 2.75, 2.88, 0.02) * 0.0035;
  t += band(r, 2.88, 3.20, 0.06) * 0.0012;
  return max(t, 0.0);
}

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  if (gid.x >= u32(RES)) { return; }
  // Supersample within the texel so that mip 0 already carries honest moments of the
  // sub-texel structure rather than a point sample of it.
  let du = (RB - RA) / RES;
  let r0 = RA + (f32(gid.x) + 0.5) * du;
  var s1 = 0.0;
  var s2 = 0.0;
  const SS: i32 = 8;
  for (var i = 0; i < SS; i++) {
    let t = tauAt(r0 + ((f32(i) + 0.5) / f32(SS) - 0.5) * du);
    s1 += t;
    s2 += t * t;
  }
  let m1 = s1 / f32(SS);
  let m2 = s2 / f32(SS);
  textureStore(tauOut, vec2i(i32(gid.x), 0), vec4f(m1, m2, 0.0, 0.0));
}
`;

// Mip reduction for the 1-D profile (RES x 1, so only x halves).
export const RING_MIP = /* wgsl */ `
@group(0) @binding(0) var src: texture_2d<f32>;
@group(0) @binding(1) var dst: texture_storage_2d<rgba16float, write>;
@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let dim = textureDimensions(dst);
  if (gid.x >= dim.x) { return; }
  let sw = i32(textureDimensions(src).x);
  let a = textureLoad(src, vec2i(min(i32(gid.x) * 2, sw - 1), 0), 0);
  let b = textureLoad(src, vec2i(min(i32(gid.x) * 2 + 1, sw - 1), 0), 0);
  textureStore(dst, vec2i(i32(gid.x), 0), vec4f((a.rg + b.rg) * 0.5, 0.0, 0.0));
}
`;

// --------------------------------------------------------------------------------------
// Shading. Included by the body pass; expects ringTau / sampLin / F in scope.
// --------------------------------------------------------------------------------------
export const RING_SHADE = /* wgsl */ `
const RING_RA: f32 = ${R0};
const RING_RB: f32 = ${R1};
const RING_RES: f32 = ${RING_RES}.0;

struct Tau { m1: f32, var2: f32 };

// Prefiltered optical depth. foot is the pixel's radial footprint in body radii, which is
// computed analytically from the neighbouring camera rays -- at grazing angles it becomes
// enormous, which is exactly when the coarse levels are the right answer.
fn tauLookup(r: f32, foot: f32) -> Tau {
  let u = (r - RING_RA) / (RING_RB - RING_RA);
  let texel = (RING_RB - RING_RA) / RING_RES;
  var lod = clamp(log2(max(foot / texel, 1.0)), 0.0, ${(Math.log2(RING_RES)).toFixed(1)});
  // debug view 3: point-sample the profile instead, which is what this whole file exists
  // to avoid. Everything the pixel does not resolve turns straight into noise.
  if (F.post.w >= 2.5 && F.post.w < 3.5) { lod = 0.0; }
  let s = textureSampleLevel(ringTau, sampLin, vec2f(u, 0.5), lod).rg;
  var o: Tau;
  o.m1 = max(s.r, 0.0);
  // debug view 4: keep the mip average but drop the second moment, so transmittance is
  // reconstructed from the mean alone
  if (F.post.w >= 3.5) { o.var2 = 0.0; } else { o.var2 = max(s.g - s.r * s.r, 0.0); }
  return o;
}

// E[exp(-k*tau)] to second order in the sub-pixel spread of tau. Without this the gaps and
// the dense B ring average to the wrong brightness instead of just the wrong sharpness:
// a pixel that is half empty and half tau=2 is much brighter than a pixel of uniform tau=1.
fn meanTransmit(t: Tau, k: f32) -> f32 {
  let x = t.m1 * k;
  let v = t.var2 * k * k;
  return exp(-min(x, 40.0)) * (1.0 + min(v * 0.5, 3.0));
}

// Two-term HG: a strong backward lobe plus a weaker forward one. cs is the cosine of the
// scattering angle, so cs = -1 at opposition (zero phase angle).
fn ringPhase(cs: f32) -> f32 {
  let back = phaseHG(cs, -F.ringQ.y);
  let fwd = phaseHG(cs, F.ringQ.z);
  return 4.0 * PI * (0.72 * back + 0.28 * fwd);
}

// Single-scattering albedo of the particles, which varies across the system: the C ring and
// the Cassini Division are darker and greyer, the B and A rings cleaner water ice.
fn ringAlbedo(r: f32) -> vec3f {
  let clean = smoothstep(1.50, 1.60, r) * (1.0 - smoothstep(2.27, 2.33, r) * 0.35);
  let w = clamp(mix(0.42, 1.0, clean) * F.ringQ.x, 0.02, 0.995);
  // the rings are distinctly tan, not grey -- tholin staining on the ice
  return vec3f(w, w * 0.895, w * 0.745);
}

// Chandrasekhar's H function, in Hapke's rational approximation. This is what carries the
// multiple scattering between particles: a slab of bright ice grains reflects far more than
// the single-scattering term alone predicts, and without it the B ring comes out about three
// times too dark -- charcoal where it should be nearly as bright as the planet.
fn hFunc(x: f32, w0: f32) -> f32 {
  let g = sqrt(max(1.0 - w0, 0.0));
  return (1.0 + 2.0 * x) / (1.0 + 2.0 * x * g);
}
fn hFunc3(x: f32, w0: vec3f) -> vec3f {
  return vec3f(hFunc(x, w0.r), hFunc(x, w0.g), hFunc(x, w0.b));
}

struct RingHit { refl: vec3f, trans: f32 };

// r        radius of the intersection, in body radii
// foot     radial footprint of the pixel, in body radii
// muV      |cos| between the ring normal and the view ray
// muS      signed cos between the ring normal and the sun direction (<0 = backlit)
// cs       cosine of the scattering angle
// sunLit   fraction of the sun not blocked by the planet at this point
// sunRad   incident solar radiance
fn shadeRing(r: f32, foot: f32, muV: f32, muS: f32, cs: f32, sunLit: f32, sunRad: vec3f) -> RingHit {
  var o: RingHit;
  o.refl = vec3f(0.0);
  o.trans = 1.0;
  if (r < RING_RA || r > RING_RB) { return o; }
  let t = tauLookup(r, foot);
  let mv = max(muV, 1.0e-4);
  o.trans = clamp(meanTransmit(t, 1.0 / mv), 0.0, 1.0);
  if (t.m1 < 1.0e-5) { return o; }

  let alb = ringAlbedo(r);
  let P = ringPhase(cs);
  let ms = abs(muS);
  let lit = sunRad * sunLit;
  var I = vec3f(0.0);

  if (muS > 0.0) {
    // ---- sunlit face: Hapke's isotropic-multiple-scattering slab ----
    // R = (w/4) * mu0/(mu0+mu) * [ P(g) + H(mu0)H(mu) - 1 ], times the finite-depth
    // saturation. The bracket splits the two regimes cleanly: P carries the single
    // scattering (and with it the phase dependence and the opposition spike), while
    // H(mu0)H(mu)-1 carries everything that bounced more than once.
    let k = 1.0 / max(ms, 1.0e-4) + 1.0 / mv;
    let sat = 1.0 - meanTransmit(t, k);
    // opposition surge: shadow hiding plus coherent backscatter between particles. The
    // narrow spike is what makes the rings flare when Saturn is near opposition, and it
    // belongs to the singly scattered light only.
    let alpha = acos(clamp(-cs, -1.0, 1.0));
    let surge = 1.0 + F.ringQ.w * exp(-alpha / 0.0090) + 0.30 * exp(-alpha / 0.12);
    let multi = hFunc3(ms, alb) * hFunc3(mv, alb) - vec3f(1.0);
    I += alb * 0.25 * (ms / (ms + mv)) * (vec3f(P * surge) + multi) * sat;
  } else {
    // ---- backlit: light that scattered once on the way through ----
    // The classic Cassini view: the thin C ring and Cassini Division glow brightly while
    // the dense B ring goes dark, because transmission falls off with tau and the thin
    // regions forward-scatter efficiently.
    let a = max(ms, 1.0e-4);
    var slab: f32;
    if (abs(a - mv) < 1.0e-3) {
      slab = (t.m1 / mv) * meanTransmit(t, 1.0 / mv);
    } else {
      slab = (a / (a - mv)) * (meanTransmit(t, 1.0 / a) - meanTransmit(t, 1.0 / mv));
    }
    let multi = hFunc3(a, alb) * hFunc3(mv, alb) - vec3f(1.0);
    I += alb * 0.25 * (vec3f(P) + multi * 0.55) * abs(slab);
  }
  o.refl = I * lit * F.ringP.w;
  return o;
}

// Optical depth along an arbitrary direction from a point, used for the ring shadow cast on
// the planet and for the shadow the rings throw on each other's illumination.
fn ringOpticalDepth(pLocal: vec3f, dir: vec3f, axis: vec3f, Req: f32, foot: f32) -> f32 {
  let dn = dot(dir, axis);
  if (abs(dn) < 1.0e-5) { return 0.0; }
  let t = -dot(pLocal, axis) / dn;
  if (t <= 0.0) { return 0.0; }
  let hit = pLocal + dir * t;
  let r = length(hit) / Req;
  if (r < RING_RA || r > RING_RB) { return 0.0; }
  return tauLookup(r, foot).m1 / max(abs(dn), 1.0e-3);
}
`;
