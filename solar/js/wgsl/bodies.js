// The main pass: one fullscreen triangle that analytically intersects every body.
//
// Spheres are not rasterised. At these scale ratios (1 m of surface relief inside 30 AU of
// space) a depth buffer is the wrong tool, whereas a ray/sphere test is exact at any
// distance and gives the silhouette as a closed-form screen-space distance -- so the limb is
// analytically anti-aliased instead of needing MSAA, the same bargain the ocean made for its
// cloud edges.
//
// Prefiltering appears three more times here:
//   * mapped surfaces read the LEAN slope-moment pyramid and feed its covariance straight
//     into a rough-surface BRDF, so a planet 10,000 km away gets statistically correct
//     roughness instead of crawling normal-map moire;
//   * close-range procedural relief adds only the octaves the pixel can resolve and dumps
//     the rest into that same covariance;
//   * gas-giant bands are band-limited the same way, by octave rather than by mip.
import { FRAME_WGSL, BODY_WGSL, GEOM_WGSL, NOISE_WGSL, RAYGEN_WGSL, FS_TRI, SURF_W, SURF_H } from './common.js';
import { RING_SHADE } from './rings.js';
import { ATMO_CORE, ATMO_SHADE } from './atmo.js';

const BINDINGS = /* wgsl */ `
@group(0) @binding(0) var<uniform> F: Frame;
@group(0) @binding(1) var<storage, read> bodies: array<Body>;
@group(0) @binding(2) var surfTex: texture_2d_array<f32>;
@group(0) @binding(3) var leanTex: texture_2d_array<f32>;
@group(0) @binding(4) var crossTex: texture_2d_array<f32>;
@group(0) @binding(5) var sampSurf: sampler;   // repeat in longitude, clamp in latitude
@group(0) @binding(6) var ringTau: texture_2d<f32>;
@group(0) @binding(7) var transLUT: texture_2d<f32>;
@group(0) @binding(8) var msLUT: texture_2d<f32>;
@group(0) @binding(9) var sampLin: sampler;    // clamp both ways, for the 1-D LUTs
`;

const SHADING = /* wgsl */ `
const SW: f32 = ${SURF_W}.0;
const SH: f32 = ${SURF_H}.0;
const SURF_MAXLOD: f32 = ${Math.log2(SURF_W)}.0;

// Shortest wrapped difference in longitude, so the date line does not read as a huge
// footprint and blur a seam across the map.
fn wrapDU(a: f32) -> f32 { return a - round(a); }

struct Slopes { s: vec2f, cov: vec3f, gloss: f32 };

// Sample the LEAN pyramid and turn the raw moments into a slope mean and covariance.
fn leanSample(layer: i32, uv: vec2f, lod: f32) -> Slopes {
  let m1 = textureSampleLevel(leanTex, sampSurf, uv, layer, lod);
  let cr = textureSampleLevel(crossTex, sampSurf, uv, layer, lod);
  var o: Slopes;
  o.s = m1.xy;
  // Var[sx], Var[sy], Cov[sx,sy] -- the covariance of the slopes the pixel actually covers
  o.cov = vec3f(max(m1.z - m1.x * m1.x, 0.0), max(m1.w - m1.y * m1.y, 0.0), cr.r - m1.x * m1.y);
  o.gloss = cr.g;
  return o;
}

fn smithLambda(cosT: f32, sigma: f32) -> f32 {
  let st = sqrt(max(1.0 - cosT * cosT, 0.0));
  let a = max(cosT, 1e-3) / max(1.41421356 * sigma * max(st, 1e-4), 1e-6);
  if (a >= 1.6) { return 0.0; }
  return (1.0 - 1.259 * a + 0.396 * a * a) / (3.535 * a + 2.181 * a * a);
}

// Anisotropic Gaussian slope NDF, evaluated in the surface tangent frame. This is the
// ocean's sun-glitter lobe: it widens correctly with distance because cov came out of the
// mip chain, which is why a sea or a lake reads as a sharp spark up close and a broad sheen
// far away without ever aliasing.
fn slopeSpecular(Vt: vec3f, Lt: vec3f, s: vec2f, cov: vec3f) -> f32 {
  if (Lt.z <= 0.0 || Vt.z <= 0.0) { return 0.0; }
  let Hv = normalize(Vt + Lt);
  if (Hv.z < 1e-4) { return 0.0; }
  let hs = -Hv.xy / Hv.z;
  let d = hs - s;
  let det = max(cov.x * cov.y - cov.z * cov.z, 1e-12);
  let e = (d.x * d.x * cov.y + d.y * d.y * cov.x - 2.0 * d.x * d.y * cov.z) / det;
  let pdf = exp(-0.5 * min(e, 60.0)) / (2.0 * PI * sqrt(det));
  let h2 = Hv.z * Hv.z;
  let D = pdf / max(h2 * h2, 1e-8);
  // Smith masking from the same statistics
  let sg = sqrt(max(0.5 * (cov.x + cov.y), 1e-9));
  let lv = smithLambda(Vt.z, sg);
  let ll = smithLambda(Lt.z, sg);
  let G = 1.0 / (1.0 + lv + ll);
  return D * G / (4.0 * max(Vt.z, 1e-3));
}


fn fresnelSchlick(c: f32, f0: f32) -> f32 {
  let m = clamp(1.0 - c, 0.0, 1.0);
  return f0 + (1.0 - f0) * m * m * m * m * m;
}
fn fresnelWater(c: f32) -> f32 {
  // Fresnel for n = 1.33, accurate enough near grazing where it matters most
  let g = sqrt(max(1.33 * 1.33 - 1.0 + c * c, 0.0));
  let a = (g - c) / (g + c);
  let b = (c * (g + c) - 1.0) / (c * (g - c) + 1.0);
  return clamp(0.5 * a * a * (1.0 + b * b), 0.0, 1.0);
}

// Rough-surface diffuse driven by the LEAN slope variance. Oren-Nayar's roughness is exactly
// the slope standard deviation the moments already carry, so a dusty, deeply cratered body
// gets its characteristically flat disc and hard terminator for free, at every distance.
fn orenNayar(NdL: f32, NdV: f32, LdV: f32, sigma2: f32) -> f32 {
  let A = 1.0 - 0.5 * sigma2 / (sigma2 + 0.33);
  let B = 0.45 * sigma2 / (sigma2 + 0.09);
  let si = sqrt(max(1.0 - NdL * NdL, 0.0));
  let sr = sqrt(max(1.0 - NdV * NdV, 0.0));
  // cos of the azimuth difference, from the full 3-D dot products
  let cosAz = clamp((LdV - NdL * NdV) / max(si * sr, 1e-4), -1.0, 1.0);
  let s = max(si, sr);
  let tanMin = min(si, sr) / max(min(NdL, NdV), 1e-3);
  return A + B * max(cosAz, 0.0) * s * min(tanMin, 4.0);
}

// Shadow hiding / coherent backscatter surge. Regoliths brighten sharply at zero phase --
// this is why the full moon is far brighter than twice the half moon, and the amplitude
// scales with the roughness the moments report.
fn oppositionSurge(alpha: f32, sigma: f32) -> f32 {
  let amp = clamp(sigma * 1.4, 0.0, 0.9);
  return 1.0 + amp * exp(-alpha / 0.18) + 0.35 * amp * exp(-alpha / 0.9);
}

// --------------------------------------------------------------------------------------
// Gas and ice giants: analytic zonal flow, band-limited by octave instead of by mip.
// --------------------------------------------------------------------------------------
struct Bands { alb: vec3f, relief: f32, var2: f32 };

fn zonalSpeed(lat: f32) -> f32 {
  return 0.55 * sin(lat * 9.0) + 0.30 * sin(lat * 17.0 + 1.1) + 0.18 * sin(lat * 27.0 + 2.3);
}

fn gasBands(dir: vec3f, axis: vec3f, east: vec3f, kind: u32, tint: vec3f, t: f32, foot: f32) -> Bands {
  let north = cross(axis, east);
  let sinLat = clamp(dot(dir, axis), -1.0, 1.0);
  let lat = asin(sinLat);
  let lon = atan2(dot(dir, north), dot(dir, east));
  // Advect the turbulence field along the jets. Shear between neighbouring bands is what
  // produces the ragged festoons at their boundaries.
  let u = zonalSpeed(lat);
  let adv = lon + u * t * 0.012;
  // Build a position that is periodic in longitude by going back to 3-D
  let cl = cos(lat);
  let q = vec3f(cos(adv) * cl, sinLat, sin(adv) * cl);

  var bandSig = (sin(lat * 9.0) * 0.5 + sin(lat * 17.0 + 1.1) * 0.3 + sin(lat * 26.0 + 0.4) * 0.2) * 1.5;
  let turb = fbm3Limited(q * 9.0 + vec3f(0.0, t * 0.004, 0.0), 7, foot * 9.0);
  let fine = fbm3Limited(q * vec3f(26.0, 7.0, 26.0) + 17.0, 6, foot * 26.0);
  bandSig += turb.v * 0.55 + fine.v * 0.30;
  let belt = smoothstep(-0.30, 0.26, bandSig);

  var o: Bands;
  o.var2 = (turb.var2 + fine.var2) * 0.02;
  if (kind == kGas) {
    // Jupiter / Saturn: creamy zones, ochre belts, bright ammonia upwellings
    let zone = vec3f(0.95, 0.90, 0.78);
    let bl = vec3f(0.42, 0.28, 0.185);
    var a = mix(bl, zone, belt);
    a = mix(a, vec3f(0.70, 0.55, 0.37), smoothstep(0.30, 0.8, fine.v + 0.4) * 0.45);
    // equatorial zone is brighter and smoother
    a = mix(a, zone * 1.03, exp(-pow(lat * 7.0, 2.0)) * 0.25);
    // polar hoods are darker and bluer-grey
    a = mix(a, vec3f(0.33, 0.31, 0.32), smoothstep(0.72, 1.05, abs(lat)) * 0.80);
    // the Great Red Spot: an oval vortex south of the equator, with its own swirl
    let gLat = -0.38;
    let gLon = 1.9 - t * 0.0021;
    var dlon = lon - gLon;
    dlon = dlon - 6.28318530718 * round(dlon / 6.28318530718);
    let e2 = pow(dlon / 0.46, 2.0) + pow((lat - gLat) / 0.17, 2.0);
    let spot = exp(-pow(e2, 1.4));
    let swirl = fbm3Limited(q * 30.0 + 71.0, 5, foot * 30.0);
    a = mix(a, vec3f(0.66, 0.27, 0.16) * (0.85 + 0.3 * swirl.v), spot * 0.95);
    o.alb = a * tint;
    o.relief = (belt - 0.5) * 0.35 + turb.v * 0.3;
  } else {
    // Uranus / Neptune: nearly featureless, faint bands, one dark vortex
    var a = mix(tint * 0.86, tint * 1.08, belt * 0.5 + 0.25);
    a = mix(a, tint * 1.14, exp(-pow((abs(lat) - 0.55) * 4.0, 2.0)) * 0.25);
    let sLat = -0.55;
    var dlon2 = lon - (2.6 - t * 0.0015);
    dlon2 = dlon2 - 6.28318530718 * round(dlon2 / 6.28318530718);
    let dark = exp(-(pow(dlon2 / 0.40, 2.0) + pow((lat - sLat) / 0.14, 2.0)));
    a = mix(a, tint * 0.52, dark * 0.8);
    // bright methane cirrus streaks
    a = mix(a, vec3f(0.92, 0.96, 0.97), smoothstep(0.72, 0.95, fine.v + 0.45) * 0.30);
    o.alb = a;
    o.relief = turb.v * 0.1;
  }
  return o;
}

// Venus and Titan: an opaque haze deck. No surface is visible, and the deck itself is
// extremely smooth, so almost all of the character comes from the limb and the terminator.
fn shroudDeck(dir: vec3f, axis: vec3f, east: vec3f, tint: vec3f, t: f32, foot: f32) -> vec3f {
  let north = cross(axis, east);
  let sinLat = clamp(dot(dir, axis), -1.0, 1.0);
  let lat = asin(sinLat);
  // super-rotation: the deck circulates far faster than the body spins
  let lon = atan2(dot(dir, north), dot(dir, east)) + t * 0.05;
  let cl = cos(lat);
  let q = vec3f(cos(lon) * cl, sinLat, sin(lon) * cl);
  let n = fbm3Limited(q * vec3f(5.0, 13.0, 5.0), 6, foot * 13.0);
  // faint Y-shaped mid-latitude cloud features, plus brighter polar collars
  var a = tint * (0.94 + 0.10 * n.v);
  a = mix(a, tint * 1.08, smoothstep(0.55, 0.95, abs(lat)) * 0.5);
  a *= 1.0 - 0.07 * smoothstep(0.2, 0.8, abs(sin(lat * 5.0) + n.v * 0.6));
  return a;
}

// --------------------------------------------------------------------------------------
// Occlusion
// --------------------------------------------------------------------------------------
// Fraction of the solar disc still visible from a point, accounting for every other body.
// This is what produces moon shadows crossing Jupiter and lunar eclipses.
fn sunVisibility(pAbsRel: vec3f, sunDir: vec3f, sunAngR: f32, skipA: u32, skipB: u32) -> f32 {
  var vis = 1.0;
  let n = arrayLength(&bodies);
  for (var i = 0u; i < n; i++) {
    if (i == skipA || i == skipB) { continue; }
    let b = bodies[i];
    if (b.rel.w <= 0.0) { continue; }
    let oc = b.rel.xyz - pAbsRel;             // occluder centre, relative to the point
    let d = length(oc);
    if (d < 1e-6) { continue; }
    let cosA = dot(oc / d, sunDir);
    if (cosA <= 0.0) { continue; }            // occluder is behind us
    let ar = asin(clamp(b.rel.w / max(d, b.rel.w), 0.0, 1.0));
    let ao = acos(clamp(cosA, -1.0, 1.0));
    // smooth overlap of two discs, good enough for penumbrae
    vis *= clamp((ao - (ar - sunAngR)) / max(2.0 * sunAngR, 1e-7), 0.0, 1.0);
    if (vis <= 0.0) { return 0.0; }
  }
  return vis;
}
`;

const MAIN = /* wgsl */ `
struct Hit { t: f32, idx: u32, cov: f32, p: vec3f, limb: f32 };

fn traceBodies(rd: vec3f, pxAng: f32) -> Hit {
  var h: Hit;
  h.t = 1e30;
  h.idx = 0xffffffffu;
  h.cov = 0.0;
  h.p = vec3f(0.0);
  h.limb = 0.0;
  let n = arrayLength(&bodies);
  for (var i = 0u; i < n; i++) {
    let b = bodies[i];
    if (b.rel.w <= 0.0 || b.misc.x <= 0.0) { continue; }
    let ro = -b.rel.xyz;
    let Req = b.rel.w;
    let Rpo = b.axis.w;
    // work in the frame where the spheroid is a sphere
    let k = Req / max(Rpo, 1e-9);
    let ax = b.axis.xyz;
    let o = ro + ax * (dot(ro, ax) * (k - 1.0));
    var d = rd + ax * (dot(rd, ax) * (k - 1.0));
    let dl = length(d);
    d = d / dl;
    let tc = -dot(o, d);
    if (tc <= 0.0) { continue; }
    let closest = o + d * tc;
    let dmin = length(closest);
    // screen-space distance to the silhouette, in the squashed frame
    let pxAt = pxAng * tc;
    let cover = clamp(0.5 + (Req - dmin) / max(pxAt, 1e-9), 0.0, 1.0);
    if (cover <= 0.0) { continue; }
    var tHit: f32;
    var pLocal: vec3f;
    if (dmin <= Req) {
      tHit = tc - sqrt(max(Req * Req - dmin * dmin, 0.0));
      pLocal = o + d * tHit;
    } else {
      // inside the anti-aliased band but geometrically a miss: shade the tangent point
      tHit = tc;
      pLocal = closest * (Req / dmin);
    }
    let tWorld = tHit / dl;
    if (tWorld >= h.t) { continue; }
    h.t = tWorld;
    h.idx = i;
    h.cov = cover * b.misc.x;
    // undo the squash to get the point in the body's real frame
    h.p = pLocal + ax * (dot(pLocal, ax) * (1.0 / k - 1.0));
    h.limb = clamp((dmin - Req) / max(pxAt, 1e-9), 0.0, 1.0);
  }
  return h;
}

fn shadeSurface(hi: Hit, rd: vec3f, rdx: vec3f, rdy: vec3f, pxAng: f32) -> vec3f {
  let b = bodies[hi.idx];
  let Req = b.rel.w;
  let Rpo = b.axis.w;
  let ax = b.axis.xyz;
  let ea = b.east.xyz;
  let kind = u32(b.east.w + 0.5);
  let p = hi.p;
  let dirU = normalize(p);
  var N = oblateNormal(p, ax, Req, Rpo);

  // sun geometry at this point
  let sunRel = F.sun.xyz - (b.rel.xyz + p);
  let sunD = length(sunRel);
  let L = sunRel / max(sunD, 1e-9);
  let sunAngR = asin(clamp(F.sun.w / max(sunD, F.sun.w), 0.0, 1.0));
  let V = -rd;
  // irradiance scale: inverse square from the sun, normalised to 1 AU
  let irr = F.sunRad.xyz * b.misc.y;

  // ---- texture / noise footprint, from the neighbouring camera rays ----
  // Both neighbours are intersected against the tangent plane at p, which is exact enough
  // for a footprint and stays uniform (dpdx is not available inside this branchy code).
  let denom = max(abs(dot(rd, N)), 1e-4);
  let tp = length(p + b.rel.xyz);
  let stepX = length(rdx - rd) * tp / denom;
  let stepY = length(rdy - rd) * tp / denom;
  let foot = max(stepX, stepY);                       // Mm on the surface
  let footU = foot / max(Req, 1e-9);                  // in body radii (~radians)

  var alb = b.tint.xyz;
  var slope = vec2f(0.0);
  // (Var[sx], Var[sy], Cov[sx,sy]) -- the cross term starts at zero, not at the
  // base variance, or the covariance is singular and the NDF blows up
  var cov = vec3f(b.mat.x, b.mat.x, 0.0);
  var gloss = 0.0;
  var water = 0.0;
  var relief = 0.0;
  let layer = i32(b.tint.w);

  if (kind == kRocky && layer >= 0) {
    let uv = equirectUV(dirU, ax, ea);
    let uvx = equirectUV(normalize(dirU + (rdx - rd) * (tp / denom) / Req), ax, ea);
    let uvy = equirectUV(normalize(dirU + (rdy - rd) * (tp / denom) / Req), ax, ea);
    let dx = vec2f(wrapDU(uvx.x - uv.x) * SW, (uvx.y - uv.y) * SH);
    let dy = vec2f(wrapDU(uvy.x - uv.x) * SW, (uvy.y - uv.y) * SH);
    let texels = max(length(dx), length(dy));
    let lod = clamp(log2(max(texels, 1.0)), 0.0, SURF_MAXLOD);
    let sv = textureSampleLevel(surfTex, sampSurf, uv, layer, lod);
    let sl = leanSample(layer, uv, lod);
    alb = sv.rgb * b.tint.xyz;
    slope = sl.s;
    cov += sl.cov;
    gloss = clamp(sl.gloss, 0.0, 1.0);
    relief = sv.a;

    // Close-range relief: add only the octaves this pixel resolves, and fold the rest into
    // the slope covariance. Same bargain as the mip chain, continued past its finest level.
    if (footU < 0.004) {
      let det = fbm3Limited(dirU * 900.0, 7, footU * 900.0);
      let amp = 0.55;
      // finite-difference the resolved part to get its slope contribution
      let e1 = normalize(cross(ax, dirU) + vec3f(1e-6));
      let e2 = cross(dirU, e1);
      let eps = max(footU, 1e-6);
      let gx = fbm3Limited(normalize(dirU + e1 * eps) * 900.0, 7, footU * 900.0).v - det.v;
      let gy = fbm3Limited(normalize(dirU + e2 * eps) * 900.0, 7, footU * 900.0).v - det.v;
      slope += vec2f(gx, gy) * (amp / max(eps * Req * 1000.0, 1e-6));
      cov += vec3f(det.var2 * amp * 2.0, det.var2 * amp * 2.0, 0.0);
    }

    // Gloss is a blend weight, not a category, because it was box-averaged all the way up
    // the mip chain: a distant pixel covering half ocean and half coast gets half the
    // smoothness and half the glint, which is the right answer.
    // Open water is not a mirror. Cox and Munk measured the slope variance of a wind-roughened
    // sea at about 0.003 + 0.0051 U, so a few m/s of wind already gives ~0.03 -- which is what
    // turns the glint from a clipped blob into the broad soft lobe you actually see from orbit.
    let wrough = 0.030 + 0.020 * b.mat.w;
    cov = mix(cov, vec3f(wrough, wrough, 0.0), gloss * gloss);
    slope *= 1.0 - gloss * 0.85;
    // open water: deep-water body colour under its own Fresnel
    if (b.mat.y >= 0.0) {
      water = gloss * (1.0 - smoothstep(b.mat.y - 0.012, b.mat.y + 0.004, relief));
      alb = mix(alb, vec3f(0.016, 0.031, 0.052) * b.tint.xyz, water);
    }
  } else if (kind == kGas || kind == kIce) {
    let bd = gasBands(dirU, ax, ea, kind, b.tint.xyz, F.camFwd.w, footU);
    alb = bd.alb;
    cov += vec3f(bd.var2 + 0.004, bd.var2 + 0.004, 0.0);
    relief = bd.relief;
  } else {
    alb = shroudDeck(dirU, ax, ea, b.tint.xyz, F.camFwd.w, footU);
    cov = vec3f(0.0025, 0.0025, 0.0);
  }

  // Keep the covariance positive definite. Averaging second moments up a mip chain in f16
  // can drift the correlation past +-1, and a non-positive determinant turns the Gaussian
  // NDF into a divide-by-nothing: a hairline of blown-out pixels across the surface.
  let cmax = 0.98 * sqrt(max(cov.x * cov.y, 0.0));
  cov.z = clamp(cov.z, -cmax, cmax);
  cov.x = max(cov.x, 1.0e-7);
  cov.y = max(cov.y, 1.0e-7);

  // debug view 1: raw albedo, unlit, so the surface recipes can be judged on their own
  if (F.post.w > 0.5 && F.post.w < 1.5) { return alb; }
  // debug view 2: slope standard deviation from the moment pyramid, the LEAN roughness
  if (F.post.w >= 1.5 && F.post.w < 2.5) {
    let sd = sqrt(max(0.5 * (cov.x + cov.y), 0.0));
    return vec3f(clamp(sd * 2.0, 0.0, 1.0), clamp(sd * 0.6, 0.0, 1.0), clamp(sd * 0.2, 0.0, 1.0));
  }

  // ---- tangent frame, with the stored slopes applied ----
  let east0 = normalize(cross(ax, dirU) + vec3f(1e-7, 0.0, 0.0));
  let north0 = cross(dirU, east0);
  let Nb = normalize(N - east0 * slope.x - north0 * slope.y);
  N = Nb;

  // ---- shadowing ----
  var lit = 1.0;
  if (dot(N, L) > -0.2) {
    lit = sunVisibility(b.rel.xyz + p, L, sunAngR, hi.idx, 0xffffffffu);
    // the planet's own rings, cast on its cloud deck
    if (b.ring.x > 0.5) {
      let od = ringOpticalDepth(p, L, ax, Req, max(footU, 0.0004));
      lit *= mix(1.0, exp(-min(od, 20.0)), b.ring.y);
    }
  }

  let NdL = dot(N, L);
  let NdV = max(dot(N, V), 1e-4);
  let LdV = dot(L, V);
  let alpha = acos(clamp(LdV, -1.0, 1.0));   // phase angle
  // terminator softened by the finite angular size of the sun
  let shade = clamp((NdL + sunAngR) / max(2.0 * sunAngR, 1e-6), 0.0, 1.0) * clamp(NdL * 4.0 + 0.5, 0.0, 1.0);
  let E = irr * lit * max(NdL, 0.0) * shade;

  let sig2 = 0.5 * (cov.x + cov.y);
  let sig = sqrt(max(sig2, 1e-9));
  var col = vec3f(0.0);

  // diffuse: rough-surface BRDF whose roughness came out of the moment pyramid
  var dif = orenNayar(max(NdL, 0.0), NdV, LdV, sig2);
  if (kind == kRocky) { dif *= oppositionSurge(alpha, sig); }
  col += alb * (1.0 / PI) * E * dif;

  // specular from the same statistics
  let Vt = vec3f(dot(V, east0), dot(V, north0), dot(V, dirU));
  let Lt = vec3f(dot(L, east0), dot(L, north0), dot(L, dirU));
  let spec = slopeSpecular(Vt, Lt, slope, max(cov, vec3f(1e-7, 1e-7, -1e9)));
  let Hv = normalize(V + L);
  let VdH = max(dot(V, Hv), 0.0);
  // Dry regolith has almost no coherent specular lobe -- it is dust, not a polished
  // dielectric. Without this Mars grows a mirror highlight it has no business having,
  // while Earth's seas and Titan's lakes still need the full Fresnel.
  let specK = mix(0.05, 1.0, gloss);
  let Fr = mix(fresnelSchlick(VdH, 0.04), fresnelWater(VdH), water);
  col += irr * lit * spec * Fr * specK * step(0.0, NdL);
  return col;
}

// --------------------------------------------------------------------------------------
// Atmospheres are composited independently of whether the surface was hit, so the limb
// glow outside the solid disc -- the thing that makes Earth and Titan read as planets with
// air rather than painted spheres -- is there in the same code path as the haze over the
// day side. (col, opa) are premultiplied.
// --------------------------------------------------------------------------------------
struct Comp { col: vec3f, opa: f32 };

fn applyAtmo(bi: u32, rd: vec3f, hitT: f32, hasHit: bool, inCol: vec3f, inOpa: f32) -> Comp {
  var o: Comp;
  o.col = inCol;
  o.opa = inOpa;
  let b = bodies[bi];
  if (b.atmo.x <= 0.0 || b.rel.w <= 0.0) { return o; }
  let ro = -b.rel.xyz;
  let Rg = b.rel.w;
  let Rt = Rg + b.atmo.x;
  let ts = raySphere(ro, rd, Rt);
  if (ts.y <= 0.0) { return o; }
  let t0 = max(ts.x, 0.0);
  var t1 = ts.y;
  if (hasHit) { t1 = min(t1, hitT); }
  if (t1 <= t0) { return o; }

  let L = normalize(F.sun.xyz - b.rel.xyz);
  let irr = F.sunRad.xyz * b.misc.y;

  if (b.atmo.w > 0.5) {
    // the body the camera is nearest: full ray march against the LUTs
    let a = marchAtmosphere(ro, rd, t0, t1, L, 24, irr);
    o.col = o.col * a.trans + a.inscat;
    let tm = dot(a.trans, vec3f(0.33333));
    o.opa = o.opa * tm + (1.0 - tm);
    return o;
  }

  // everything else: a short exponential-profile march. No LUTs and no multiple
  // scattering, which is all a disc a few pixels across can show.
  let H = max(b.atmo.x * 0.25, 1e-5);
  let cosT = dot(rd, L);
  let pR = phaseRayleigh(cosT);
  let pM = phaseHG(cosT, 0.70);
  let sc = b.shell.xyz * b.atmo.y;
  let ha = b.atmo.z;
  var trans = vec3f(1.0);
  var ins = vec3f(0.0);
  const STEPS: i32 = 6;
  let dt = (t1 - t0) / f32(STEPS);
  for (var i = 0; i < STEPS; i++) {
    let p = ro + rd * (t0 + dt * (f32(i) + 0.5));
    let pr = max(length(p), 1e-9);
    let dens = exp(-max(pr - Rg, 0.0) / H);
    let ext = sc * (1.0 + ha) * dens;
    let stepT = exp(-ext * dt);
    let mu = dot(p / pr, L);
    // soft terminator, and the planet's own shadow for points behind it
    let horiz = -sqrt(max(1.0 - (Rg * Rg) / (pr * pr), 0.0));
    let lit = clamp((mu - horiz) * 6.0, 0.0, 1.0) * clamp(mu * 3.0 + 0.4, 0.0, 1.0);
    let s = sc * dens * (pR + pM * ha) * irr * lit;
    let safe = max(ext, vec3f(1e-9));
    ins += trans * (s - s * stepT) / safe;
    trans *= stepT;
  }
  o.col = o.col * trans + ins;
  let tm = dot(trans, vec3f(0.33333));
  o.opa = o.opa * tm + (1.0 - tm);
  return o;
}

// Rings, composited against whatever is behind them. The ring attenuates the background by
// its (prefiltered) transmittance and adds its own scattered radiance -- so the same code
// handles rings crossing in front of the planet, rings seen against empty space, and the
// backlit case where the dense B ring goes dark while the thin C ring glows.
fn applyRings(bi: u32, r: Rays, hitT: f32, hasHit: bool, inCol: vec3f, inOpa: f32) -> Comp {
  var o: Comp;
  o.col = inCol;
  o.opa = inOpa;
  let b = bodies[bi];
  if (b.ring.x <= 0.5 || b.rel.w <= 0.0) { return o; }
  let ax = b.axis.xyz;
  let ro = -b.rel.xyz;
  let dn = dot(r.d, ax);
  if (abs(dn) < 1e-9) { return o; }
  let t = -dot(ro, ax) / dn;
  if (t <= 0.0) { return o; }
  if (hasHit && t > hitT) { return o; }          // behind the planet
  let p = ro + r.d * t;
  let rr = length(p) / b.rel.w;
  if (rr < RING_RA || rr > RING_RB) { return o; }

  // Radial footprint of the pixel, from the neighbouring rays. This is the quantity that
  // picks the mip level; at grazing angles it blows up, which is precisely when the coarse
  // levels are the correct answer rather than a compromise.
  let dnx = dot(r.dx, ax);
  let dny = dot(r.dy, ax);
  var foot = 0.0;
  if (abs(dnx) > 1e-9) {
    foot = max(foot, abs(length(ro + r.dx * (-dot(ro, ax) / dnx)) / b.rel.w - rr));
  }
  if (abs(dny) > 1e-9) {
    foot = max(foot, abs(length(ro + r.dy * (-dot(ro, ax) / dny)) / b.rel.w - rr));
  }

  let sunRel = F.sun.xyz - (b.rel.xyz + p);
  let sunD = length(sunRel);
  let L = sunRel / max(sunD, 1e-9);
  let sunAngR = asin(clamp(F.sun.w / max(sunD, F.sun.w), 0.0, 1.0));
  let muV = abs(dn);
  let muS = dot(ax, L) * sign(-dn);              // signed relative to the viewer's side
  let cs = dot(r.d, L);
  // the planet's shadow thrown across its own rings
  let lit = sunVisibility(b.rel.xyz + p, L, sunAngR, 0xffffffffu, 0xffffffffu);
  let irr = F.sunRad.xyz * b.misc.y;
  let rh = shadeRing(rr, foot, muV, muS, cs, lit, irr);
  let tr = mix(1.0, rh.trans, clamp(F.ringP.z, 0.0, 1.0));
  o.col = o.col * tr + rh.refl;
  o.opa = o.opa * tr + (1.0 - tr);
  return o;
}

@fragment fn fs(in: VO) -> @location(0) vec4f {
  let r = makeRays(in.uv);
  let hi = traceBodies(r.d, r.px);
  let hasHit = hi.idx != 0xffffffffu;
  // premultiplied: col is what this pixel adds, opa is how much background it removes
  var col = vec3f(0.0);
  var opa = 0.0;

  if (hasHit) {
    col = shadeSurface(hi, r.d, r.dx, r.dy, r.px) * hi.cov;
    opa = hi.cov;
  }

  let n = arrayLength(&bodies);
  for (var i = 0u; i < n; i++) {
    if (bodies[i].atmo.x > 0.0 && F.post.w < 0.5) {
      let c = applyAtmo(i, r.d, hi.t, hasHit, col, opa);
      col = c.col; opa = c.opa;
    }
  }
  for (var i = 0u; i < n; i++) {
    if (bodies[i].ring.x > 0.5) {
      let c = applyRings(i, r, hi.t, hasHit, col, opa);
      col = c.col; opa = c.opa;
    }
  }

  return vec4f(max(col, vec3f(0.0)), clamp(opa, 0.0, 1.0));
}
`;

export const BODIES = FRAME_WGSL + BODY_WGSL + GEOM_WGSL + NOISE_WGSL + BINDINGS +
  RAYGEN_WGSL + FS_TRI + RING_SHADE + ATMO_CORE + ATMO_SHADE + SHADING + MAIN;

// The two atmosphere LUT passes share the medium code but bind their own storage targets.
// layout: 'auto' derives the bind group layout from what the shader actually references,
// so these preludes declare exactly the bindings their pass uses -- no more.
export const LUT_PRELUDE = /* wgsl */ `
@group(0) @binding(0) var<uniform> F: Frame;
@group(0) @binding(1) var transOut: texture_storage_2d<rgba16float, write>;
`;
export const MS_PRELUDE = /* wgsl */ `
@group(0) @binding(0) var<uniform> F: Frame;
@group(0) @binding(1) var sampLin: sampler;
@group(0) @binding(2) var transLUT: texture_2d<f32>;
@group(0) @binding(3) var msOut: texture_storage_2d<rgba16float, write>;
`;
