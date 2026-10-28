// Background, stars and the sun.
//
// Stars are the cheapest illustration of the whole thesis. A star is a point: sampled
// point-wise it either lands in a pixel or it does not, so panning the camera makes the sky
// crawl and twinkle. Instead each star's *flux* is splatted into a small Gaussian, so the
// energy in the pixel is a continuous function of where the star falls. Nothing flickers,
// and brightness is conserved under any rotation or resolution change.
//
// Distant planets go through the same path. Once a body's disc drops below about a pixel the
// body pass stops intersecting it and the CPU appends it to the splat buffer instead, with
// the flux of a Lambertian sphere at the right phase angle. The handover is cross-faded, so
// Jupiter shrinking into the distance turns into a point of light of exactly the brightness
// it had as a disc.
import { FRAME_WGSL, GEOM_WGSL, NOISE_WGSL, RAYGEN_WGSL, FS_TRI, BODY_WGSL } from './common.js';

// Galactic north pole, in the renderer's ecliptic frame (Y = ecliptic north).
const GAL_POLE = 'vec3f(-0.8676, 0.4971, 0.0003)';

export const BACKGROUND = FRAME_WGSL + GEOM_WGSL + NOISE_WGSL + /* wgsl */ `
@group(0) @binding(0) var<uniform> F: Frame;
` + RAYGEN_WGSL + FS_TRI + /* wgsl */ `
const GP: vec3f = ${GAL_POLE};

// Unresolved stars, dust lanes and the central bulge, all band-limited: the octave fade in
// fbm3Limited keeps the dust structure from aliasing when the camera spins.
fn milkyWay(d: vec3f, foot: f32) -> vec3f {
  let b = asin(clamp(dot(d, GP), -1.0, 1.0));           // galactic latitude
  let gx = normalize(cross(GP, vec3f(0.0, 1.0, 0.0)) + vec3f(1e-6));
  let gy = cross(GP, gx);
  let l = atan2(dot(d, gy), dot(d, gx));                // galactic longitude
  // the band itself: a thin disc of unresolved stars, thicker and brighter toward the centre
  let toCentre = exp(-pow(l / 1.25, 2.0));
  let thick = 0.085 + 0.075 * toCentre;
  var band = exp(-pow(b / thick, 2.0));
  // the bulge
  band += 0.55 * exp(-(pow(l / 0.34, 2.0) + pow(b / 0.20, 2.0)));
  // mottling and the dark dust lanes that cut the band in two
  let n = fbm3Limited(d * 26.0, 6, foot * 26.0);
  let dust = fbm3Limited(d * 14.0 + 31.0, 5, foot * 14.0);
  let lane = exp(-pow(b / (thick * 0.30), 2.0)) * smoothstep(-0.1, 0.5, dust.v);
  var i = band * (0.72 + 0.55 * n.v) * (1.0 - 0.72 * lane);
  i = max(i, 0.0);
  // integrated starlight is slightly warm; the dust reddens what shines through it
  let warm = vec3f(1.00, 0.95, 0.88);
  let red = vec3f(1.00, 0.72, 0.48);
  return mix(warm, red, lane * 0.6) * i * F.starP.y;
}

// Zodiacal light: sunlight scattered by interplanetary dust, concentrated along the ecliptic
// and strongest close to the sun.
fn zodiacal(d: vec3f) -> vec3f {
  let sunD = length(F.sun.xyz);
  if (sunD < 1.0) { return vec3f(0.0); }
  let sd = F.sun.xyz / sunD;
  // interplanetary dust thins with distance too, so this falls off faster than 1/r^2
  let fall = pow(149597.8707 / sunD, 2.6);
  let elong = acos(clamp(dot(d, sd), -1.0, 1.0));
  let lat = abs(asin(clamp(d.y, -1.0, 1.0)));
  let band = exp(-pow(lat / 0.28, 2.0));
  let near = pow(max(PI - elong, 0.0) / PI, 2.4);
  let gegen = 0.22 * exp(-pow(elong / 0.5, 2.0));       // gegenschein, opposite the sun
  return vec3f(1.00, 0.96, 0.90) * (near * band * 0.55 + gegen * band) * F.starP.z * fall;
}

@fragment fn fs(in: VO) -> @location(0) vec4f {
  let r = makeRays(in.uv);
  var c = milkyWay(r.d, r.px);
  c += zodiacal(r.d);
  return vec4f(c, 1.0);
}
`;

// --------------------------------------------------------------------------------------
// Star / splat point sprites. One instance per light source; the quad is built in the
// vertex stage and the fragment stage lays down a normalised Gaussian.
// --------------------------------------------------------------------------------------
export const STARS = FRAME_WGSL + /* wgsl */ `
@group(0) @binding(0) var<uniform> F: Frame;

struct SO {
  @builtin(position) pos: vec4f,
  @location(0) off: vec2f,      // offset from the centre, in pixels
  @location(1) col: vec3f,      // peak radiance
  @location(2) sig: f32,        // Gaussian sigma, pixels
};

// a: xyz = direction (unit), w = flux
// b: rgb = colour (unit-ish), a = sigma in pixels; a negative sigma marks a body splat
// rather than a star, so the star-brightness control does not also dim the planets
@vertex fn vs(@builtin(vertex_index) vi: u32,
              @location(0) a: vec4f,
              @location(1) b: vec4f) -> SO {
  var o: SO;
  let sig = abs(b.a);
  let gain = select(1.0, F.starP.x, b.a > 0.0);
  let clip = F.viewProj * vec4f(a.xyz * 1.0e7, 1.0);
  // quad corners: 3 vertices of a triangle that covers the [-1,1] square
  let q = vec2f(f32((vi << 1u) & 2u), f32(vi & 2u)) * 2.0 - 1.0;
  let half = max(sig * 3.2, 2.0);
  if (clip.w <= 0.0) {
    o.pos = vec4f(0.0, 0.0, -1.0, 1.0);   // behind the camera: collapse the quad
    o.off = vec2f(1e3);
    o.col = vec3f(0.0);
    o.sig = 1.0;
    return o;
  }
  let ndc = clip.xy / clip.w;
  let px = vec2f(F.screen.z, F.screen.w) * 2.0;    // ndc per pixel
  o.pos = vec4f(ndc + q * half * px, 0.0, 1.0);
  o.off = q * half;
  // peak of a normalised 2-D Gaussian carrying flux as its total energy, so the pixel
  // sum is invariant to where the point lands and to the resolution
  o.col = b.rgb * (a.w * gain / (6.28318530718 * sig * sig));
  o.sig = sig;
  return o;
}

@fragment fn fs(in: SO) -> @location(0) vec4f {
  let d2 = dot(in.off, in.off) / max(in.sig * in.sig, 1e-6);
  let g = exp(-0.5 * d2);
  return vec4f(in.col * g, g);
}
`;

// --------------------------------------------------------------------------------------
// The sun: limb-darkened disc, photographic glare, and analytic occlusion by every body
// (which is what gives eclipses, and the bead of light at the edge of one).
// --------------------------------------------------------------------------------------
export const SUN = FRAME_WGSL + BODY_WGSL + GEOM_WGSL + /* wgsl */ `
@group(0) @binding(0) var<uniform> F: Frame;
@group(0) @binding(1) var<storage, read> bodies: array<Body>;
` + RAYGEN_WGSL + FS_TRI + /* wgsl */ `
const AU_MM: f32 = 149597.8707;

// Fraction of the solar disc visible along a ray, sampled over the disc so that a body
// crossing it produces a real partial eclipse rather than a binary switch.
fn sunUncovered(rd: vec3f, sd: vec3f, ar: f32) -> f32 {
  let n = arrayLength(&bodies);
  var vis = 1.0;
  for (var i = 0u; i < n; i++) {
    let b = bodies[i];
    if (b.rel.w <= 0.0) { continue; }
    let d = length(b.rel.xyz);
    if (d < 1e-9) { continue; }
    let bd = b.rel.xyz / d;
    // only bodies between us and the sun can occult it
    if (dot(bd, sd) <= 0.0) { continue; }
    if (d > length(F.sun.xyz)) { continue; }
    let br = asin(clamp(b.rel.w / max(d, b.rel.w), 0.0, 1.0));
    let sep = acos(clamp(dot(rd, bd), -1.0, 1.0));
    vis *= clamp((sep - br) / max(F.sunRad.w, 1e-7) + 0.5, 0.0, 1.0);
    if (vis <= 0.0) { return 0.0; }
  }
  return vis;
}

@fragment fn fs(in: VO) -> @location(0) vec4f {
  let r = makeRays(in.uv);
  let sunD = length(F.sun.xyz);
  if (sunD < 1e-9) { return vec4f(0.0); }
  let sd = F.sun.xyz / sunD;
  let ar = asin(clamp(F.sun.w / max(sunD, F.sun.w), 0.0, 1.0));
  let ao = acos(clamp(dot(r.d, sd), -1.0, 1.0));
  if (ao > 1.25) { return vec4f(0.0); }

  // radiance of the photosphere, independent of distance; only the solid angle changes
  let L0 = F.sunRad.xyz * 5.1e3;
  var c = vec3f(0.0);

  // ---- the disc, with limb darkening ----
  let cov = discCoverage(ao, ar, F.sunRad.w);
  if (cov > 0.0) {
    let x = clamp(ao / max(ar, 1e-9), 0.0, 1.0);
    let mu = sqrt(max(1.0 - x * x, 0.0));
    // Eddington-like law, tuned for the visible band; the limb is ~35% as bright and redder
    let ld = 1.0 - 0.84 * (1.0 - mu) + 0.20 * (1.0 - mu * mu);
    let warm = mix(vec3f(1.0), vec3f(1.00, 0.72, 0.42), pow(1.0 - mu, 1.6) * 0.8);
    c += L0 * max(ld, 0.0) * warm * cov;
  }

  // ---- corona and photographic glare ----
  // The disc keeps L0: surface radiance does not fall off with distance. The glare does,
  // because it is scattered *flux*, so it scales with the irradiance reaching the camera.
  // Getting this wrong puts a full-screen grey wash over the outer solar system, where the
  // automatic exposure is boosting everything by d^2 to compensate for the dim sunlight.
  let irr = AU_MM / max(sunD, 1.0);
  let gl = L0 * irr * irr;
  let t = max(ao, ar * 0.9);
  let k = ar / t;
  // The veil term falls off slowly by design, which means it is still well above black at
  // the cutoff angle -- and a hard cutoff then draws a visible arc straight across the
  // frame. Window it to zero before the edge.
  let edge = smoothstep(1.25, 0.30, ao);
  c += gl * 0.015 * pow(k, 2.2);                           // inner corona
  c += gl * 0.0026 * pow(k, 1.6) * edge;                   // outer streamers
  c += gl * vec3f(0.9, 0.95, 1.1) * 0.0011 * pow(k, 1.1) * edge;  // lens veil

  let vis = sunUncovered(r.d, sd, ar);
  return vec4f(c * vis, 0.0);
}
`;
