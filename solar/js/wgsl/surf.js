// One-off generation of the mapped surfaces, and the LEAN slope-moment pyramid built from
// them.
//
// This is the ocean's trick moved from water to rock. The ocean stored mip-mapped first and
// second moments of the wave slopes so that sub-pixel wave detail turned into the correct
// anisotropic roughness instead of into aliasing. Terrain has exactly the same problem: a
// planet seen from 10,000 km has thousands of craters per pixel. Storing E[s] and E[s (x) s]
// and letting the mip chain average them means a distant planet gets a statistically correct
// dusty sheen and a correct soft terminator, with no crawling normal-map moire.
//
// Pass order: SURF_GEN (albedo + height) -> SURF_LEAN (slopes from the height map) -> mip
// reduction of the moments.
import { SURF_W, SURF_H, NOISE_WGSL, CRATER_WGSL } from './common.js';

const GEN_HEAD = /* wgsl */ `
struct Rec { p: vec4f };   // x = recipe, y = seed, z = max relief (km), w = radius (km)
@group(0) @binding(0) var<uniform> R: array<Rec, 16>;
@group(0) @binding(1) var surfOut: texture_storage_2d_array<rgba16float, write>;
// height and gloss go out as raw bits in a uint texture: the slope pass needs full f32
// height precision (a 10 m difference across a 70 km texel is the whole signal), and a
// uint target also keeps this pass from having to alias the map the slope pass writes.
@group(0) @binding(2) var auxOut: texture_storage_2d_array<rg32uint, write>;

const W: f32 = ${SURF_W}.0;
const H: f32 = ${SURF_H}.0;
const PI: f32 = 3.14159265358979;

// gloss is a specular-smoothness fraction in [0,1], not a category code: it has to
// survive being box-averaged all the way up the mip chain, and averaging category codes
// would turn rock next to ice into something else entirely.
struct Surf { alb: vec3f, h: f32, gloss: f32 };   // h in km

fn dirOf(uv: vec2f, axisUp: bool) -> vec3f {
  let lon = (uv.x - 0.5) * 2.0 * PI;
  let lat = (0.5 - uv.y) * PI;
  let cl = cos(lat);
  // local frame: x = east at lon 0, y = north pole, z = north at lon 0
  return vec3f(cos(lon) * cl, sin(lat), sin(lon) * cl);
}
`;

// --------------------------------------------------------------------------------------
// Per-body surface recipes. Noise is evaluated in 3-D on the unit sphere, so there is no
// seam at the date line and no pinching at the poles.
// --------------------------------------------------------------------------------------
const RECIPES = /* wgsl */ `
fn tempLat(d: vec3f) -> f32 { return abs(d.y); }

// ---- 0: Mercury -- heavily cratered, grey, bright ray systems ----
fn recMercury(d: vec3f, s: f32) -> Surf {
  let p = d * 7.0 + vec3f(s);
  var h = craterField(p, 4, 0.55) * 9.0;
  h += fbm3(p * 0.6, 4, 2.1, 0.5) * 1.2;
  // smooth intercrater plains in the low ground
  let plains = smoothstep(-0.3, 0.6, fbm3(p * 0.35 + 11.0, 3, 2.0, 0.5));
  h = mix(h, h * 0.45, plains);
  var o: Surf;
  let g = 0.085 + 0.030 * fbm3(p * 2.3, 3, 2.0, 0.5) + 0.018 * plains;
  // fresh crater rays: thin bright streaks from the youngest impacts
  let ray = smoothstep(0.62, 0.95, ridged3(p * 3.1 + 31.0, 3));
  o.alb = vec3f(g * 1.06, g * 1.00, g * 0.94) + vec3f(0.055) * ray;
  o.h = h; o.gloss = 0.0;
  return o;
}

// ---- 1: Earth -- continents, ocean, deserts, forests, ice caps ----
fn recEarth(d: vec3f, s: f32) -> Surf {
  let p = d * 2.2 + vec3f(s);
  // continent mask: warped low-frequency fBm, thresholded
  let warp = vec3f(fbm3(p * 0.9 + 5.0, 3, 2.0, 0.5), fbm3(p * 0.9 + 9.0, 3, 2.0, 0.5),
                   fbm3(p * 0.9 + 13.0, 3, 2.0, 0.5)) * 0.55;
  let cont = fbm3(p + warp, 6, 2.1, 0.52) + 0.06;
  let land = smoothstep(0.00, 0.16, cont);
  // mountain belts along continental margins
  let belt = ridged3(p * 3.2 + 3.0, 5) * smoothstep(0.02, 0.35, cont);
  let shelf = smoothstep(-0.18, 0.02, cont);
  var h = -5.0 + shelf * 4.6 + land * (0.9 + belt * 7.5) + fbm3(p * 6.0, 4, 2.0, 0.5) * 0.7 * land;
  var o: Surf;
  o.gloss = 0.0;
  if (h < 0.0) {
    // ocean floor: abyssal plains and ridges, albedo is the water body colour
    h += fbm3(p * 5.0 + 21.0, 3, 2.0, 0.5) * 0.8;
    let deep = smoothstep(-0.2, -3.5, h);
    o.alb = mix(vec3f(0.055, 0.105, 0.125), vec3f(0.012, 0.028, 0.055), deep);
    o.gloss = 1.0;
  } else {
    let lat = tempLat(d);
    let arid = fbm3(p * 1.7 + 41.0, 4, 2.1, 0.5);
    // desert belts around +-25 deg latitude
    let desert = smoothstep(0.18, 0.46, exp(-pow((lat - 0.40) * 5.0, 2.0)) + arid * 0.55);
    let green = vec3f(0.055, 0.085, 0.038);
    let sand = vec3f(0.215, 0.165, 0.105);
    let rock = vec3f(0.115, 0.100, 0.088);
    var a = mix(green, sand, desert);
    a = mix(a, rock, smoothstep(2.5, 6.0, h));
    // snow line: high ground and high latitude
    let snow = smoothstep(0.76, 0.90, lat + max(0.0, h - 3.0) * 0.08);
    o.alb = mix(a, vec3f(0.62, 0.64, 0.68), snow);
    o.gloss = snow * 0.45;
  }
  // polar sea ice
  let pol = smoothstep(0.86, 0.95, tempLat(d));
  if (pol > 0.0 && h < 0.0) {
    o.alb = mix(o.alb, vec3f(0.58, 0.62, 0.66), pol);
    o.gloss = mix(o.gloss, 0.60, pol);
    h = mix(h, 0.02, pol);
  }
  o.h = h;
  return o;
}

// ---- 2: Moon -- dark maria, bright highlands, dense craters, ray systems ----
fn recMoon(d: vec3f, s: f32) -> Surf {
  let p = d * 5.5 + vec3f(s);
  // maria: large low-albedo basalt floods in big basins, concentrated on one hemisphere
  let basin = fbm3(p * 0.42 + 7.0, 4, 2.0, 0.55) + d.z * 0.22;
  let mare = smoothstep(0.12, 0.30, basin);
  var h = craterField(p, 5, 0.62) * 10.0 + fbm3(p * 0.8, 4, 2.0, 0.5) * 1.6;
  h = mix(h, -1.1 + fbm3(p * 2.2, 3, 2.0, 0.5) * 0.35, mare);
  var o: Surf;
  let hl = 0.125 + 0.035 * fbm3(p * 3.0, 3, 2.0, 0.5);
  let mr = 0.058 + 0.012 * fbm3(p * 2.0 + 17.0, 3, 2.0, 0.5);
  var g = mix(hl, mr, mare);
  let ray = smoothstep(0.58, 0.96, ridged3(p * 2.6 + 23.0, 4)) * (1.0 - mare * 0.7);
  g += 0.045 * ray;
  o.alb = vec3f(g, g * 0.985, g * 0.945);
  o.h = h; o.gloss = 0.0;
  return o;
}

// ---- 3: Mars -- butterscotch dust, volcanoes, a rift canyon, polar caps ----
fn recMars(d: vec3f, s: f32) -> Surf {
  let p = d * 4.0 + vec3f(s);
  // hemispheric dichotomy: cratered southern highlands, smooth northern plains
  let dich = smoothstep(-0.25, 0.30, -d.y * 1.4 + fbm3(p * 0.5, 3, 2.0, 0.5) * 0.7);
  var h = 1.0 + dich * 3.2;
  h += craterField(p, 4, 0.50) * 7.0 * dich;
  h += fbm3(p * 1.3, 5, 2.1, 0.5) * 1.8;
  // Tharsis-like shield volcanoes: a few tall cones near the equator
  let vc = vec3f(0.78, 0.18, 0.60);
  let dv = length(d - normalize(vc));
  h += exp(-pow(dv * 7.0, 2.0)) * 17.0;
  h += exp(-pow(length(d - normalize(vec3f(0.62, 0.34, 0.71))) * 11.0, 2.0)) * 11.0;
  // Valles Marineris: a long rift along a great circle, east of the bulge
  let rift = normalize(vec3f(0.20, 0.06, 0.98));
  let along = dot(d, normalize(cross(rift, vec3f(0.0, 1.0, 0.0))));
  let across = dot(d, rift);
  let canyon = exp(-pow(across * 26.0, 2.0)) * smoothstep(0.52, 0.64, abs(along)) *
               smoothstep(0.0, 0.3, 0.86 - abs(along));
  h -= canyon * 7.5 * (0.7 + 0.3 * fbm3(p * 9.0, 3, 2.0, 0.5));
  var o: Surf;
  let dust = fbm3(p * 2.1 + 29.0, 4, 2.0, 0.5);
  let bright = vec3f(0.255, 0.148, 0.088);
  let dark = vec3f(0.112, 0.070, 0.048);
  var a = mix(dark, bright, smoothstep(-0.35, 0.40, dust));
  // dark wind streaks downwind of topography
  a *= 1.0 - 0.28 * smoothstep(0.55, 0.9, ridged3(p * vec3f(6.0, 2.0, 6.0) + 51.0, 3));
  let cap = smoothstep(0.88, 0.965, tempLat(d) + fbm3(p * 5.0, 3, 2.0, 0.5) * 0.035);
  o.alb = mix(a, vec3f(0.70, 0.70, 0.73), cap);
  o.gloss = cap * 0.50;
  o.h = h;
  return o;
}

// ---- 4: Io -- sulphur yellows, volcanic calderas, no impact craters ----
fn recIo(d: vec3f, s: f32) -> Surf {
  let p = d * 5.0 + vec3f(s);
  var h = fbm3(p * 1.1, 5, 2.1, 0.5) * 1.6 + ridged3(p * 2.4, 4) * 1.2;
  // paterae: dark irregular volcanic depressions
  let pat = smoothstep(0.52, 0.80, fbm3(p * 3.4 + 13.0, 4, 2.2, 0.5) + 0.25);
  h -= pat * 1.4;
  var o: Surf;
  let sul = fbm3(p * 2.0 + 5.0, 4, 2.0, 0.5);
  var a = mix(vec3f(0.44, 0.33, 0.12), vec3f(0.62, 0.56, 0.26), smoothstep(-0.3, 0.4, sul));
  a = mix(a, vec3f(0.52, 0.22, 0.10), smoothstep(0.25, 0.75, fbm3(p * 1.5 + 37.0, 3, 2.0, 0.5) + 0.2));
  a = mix(a, vec3f(0.085, 0.065, 0.055), pat);
  // bright SO2 frost deposits
  a = mix(a, vec3f(0.72, 0.70, 0.62), smoothstep(0.62, 0.88, fbm3(p * 4.0 + 61.0, 3, 2.0, 0.5) + 0.2) * 0.8);
  o.alb = a; o.h = h; o.gloss = mix(0.05, 0.25, step(0.6, pat));
  return o;
}

// ---- 5: Europa -- bright ice crust cut by reddish lineae ----
fn recEuropa(d: vec3f, s: f32) -> Surf {
  let p = d * 4.0 + vec3f(s);
  var h = fbm3(p * 2.0, 4, 2.0, 0.5) * 0.35;
  // lineae: long double ridges, several overlapping sets of great-circle-ish cracks
  var crack = 0.0;
  for (var i = 0; i < 4; i++) {
    let q = p * (1.3 + f32(i) * 0.7) + vec3f(f32(i) * 17.0);
    crack = max(crack, smoothstep(0.80, 0.995, ridged3(q, 3)));
  }
  h += crack * 0.25;
  var o: Surf;
  let dirt = smoothstep(0.3, 0.9, fbm3(p * 1.6 + 23.0, 4, 2.0, 0.5) + 0.3);
  var a = mix(vec3f(0.66, 0.64, 0.60), vec3f(0.50, 0.42, 0.33), dirt * 0.6);
  a = mix(a, vec3f(0.40, 0.24, 0.16), crack * 0.85);
  o.alb = a; o.h = h; o.gloss = 0.70;
  return o;
}

// ---- 6: Ganymede -- dark cratered terrain cut by bright grooved bands ----
fn recGanymede(d: vec3f, s: f32) -> Surf {
  let p = d * 4.5 + vec3f(s);
  let old = smoothstep(0.08, 0.34, fbm3(p * 0.55 + 3.0, 4, 2.0, 0.55));
  var h = craterField(p, 4, 0.52) * 5.0 * old;
  // sulcus bands: parallel grooves, direction varies per region
  let gr = ridged3(p * vec3f(8.0, 1.6, 8.0) + 9.0, 3);
  h += (1.0 - old) * (0.35 + gr * 0.55);
  h += fbm3(p * 1.4, 4, 2.0, 0.5) * 0.7;
  var o: Surf;
  let dark = vec3f(0.135, 0.120, 0.108);
  let light = vec3f(0.295, 0.285, 0.272);
  var a = mix(light, dark, old);
  a = mix(a, light * 1.08, (1.0 - old) * smoothstep(0.3, 0.8, gr) * 0.5);
  a += vec3f(0.07) * smoothstep(0.7, 0.97, ridged3(p * 2.8 + 43.0, 3));
  o.alb = a; o.h = h; o.gloss = 0.35;
  return o;
}

// ---- 7: Callisto -- the most heavily cratered surface in the system, dark and ancient ----
fn recCallisto(d: vec3f, s: f32) -> Surf {
  let p = d * 6.0 + vec3f(s);
  var h = craterField(p, 5, 0.74) * 9.0 + fbm3(p * 0.9, 4, 2.0, 0.5) * 1.1;
  var o: Surf;
  let g = 0.085 + 0.030 * fbm3(p * 2.2, 4, 2.0, 0.5);
  // bright ice exposed on crater rims (correlates with high ground)
  let fresh = smoothstep(0.8, 2.6, h);
  o.alb = vec3f(g * 1.04, g, g * 0.95) + vec3f(0.10, 0.10, 0.105) * fresh;
  o.h = h; o.gloss = 0.05;
  return o;
}

// ---- 8: Pluto -- nitrogen-ice plain, dark tholin equator, bright caps ----
fn recPluto(d: vec3f, s: f32) -> Surf {
  let p = d * 4.0 + vec3f(s);
  var h = craterField(p, 3, 0.40) * 4.0 + fbm3(p * 1.2, 5, 2.0, 0.5) * 1.4;
  // Sputnik Planitia: a large, very smooth, very bright basin
  let sp = normalize(vec3f(0.55, 0.30, -0.78));
  let dsp = length(d - sp);
  let plain = smoothstep(0.52, 0.34, dsp + fbm3(p * 2.0, 3, 2.0, 0.5) * 0.09);
  h = mix(h, -1.6, plain);
  var o: Surf;
  let thol = smoothstep(0.30, 0.70, (1.0 - tempLat(d)) * 0.9 + fbm3(p * 1.5 + 7.0, 4, 2.0, 0.5) * 0.55);
  var a = mix(vec3f(0.50, 0.44, 0.37), vec3f(0.26, 0.145, 0.095), thol);
  a = mix(a, vec3f(0.74, 0.72, 0.66), plain);
  a = mix(a, vec3f(0.66, 0.63, 0.58), smoothstep(0.80, 0.95, tempLat(d)));
  o.alb = a; o.h = h; o.gloss = 0.50;
  return o;
}

fn surfaceOf(rec: i32, d: vec3f, seed: f32) -> Surf {
  switch (rec) {
    case 0: { return recMercury(d, seed); }
    case 1: { return recEarth(d, seed); }
    case 2: { return recMoon(d, seed); }
    case 3: { return recMars(d, seed); }
    case 4: { return recIo(d, seed); }
    case 5: { return recEuropa(d, seed); }
    case 6: { return recGanymede(d, seed); }
    case 7: { return recCallisto(d, seed); }
    default: { return recPluto(d, seed); }
  }
}
`;

export const SURF_GEN = NOISE_WGSL + CRATER_WGSL + GEN_HEAD + RECIPES + /* wgsl */ `
@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  if (gid.x >= u32(W) || gid.y >= u32(H)) { return; }
  let layer = gid.z;
  let rec = R[layer].p;
  let uv = (vec2f(f32(gid.x), f32(gid.y)) + 0.5) / vec2f(W, H);
  let d = dirOf(uv, true);
  let s = surfaceOf(i32(rec.x), d, rec.y);
  let hk = s.h;
  // normalised height for shading; the exact value goes to the r32float map that the
  // slope pass reads, because f16 differences over a 70 km texel would be pure noise
  let hn = clamp(hk / max(rec.z, 0.001) * 0.5 + 0.5, 0.0, 1.0);
  textureStore(surfOut, vec2i(gid.xy), i32(layer), vec4f(s.alb, hn));
  textureStore(auxOut, vec2i(gid.xy), i32(layer),
               vec4u(bitcast<u32>(hk), bitcast<u32>(s.gloss), 0u, 0u));
}
`;

// --------------------------------------------------------------------------------------
// Slopes -> LEAN moments (mip 0). Central differences on the generated height map, in the
// local east/north tangent frame, converted to true dimensionless slopes using the
// equirectangular metric (which is why the cos(latitude) term is here).
// --------------------------------------------------------------------------------------
export const SURF_LEAN = /* wgsl */ `
struct Rec { p: vec4f };
@group(0) @binding(0) var<uniform> R: array<Rec, 16>;
@group(0) @binding(1) var auxIn: texture_2d_array<u32>;
@group(0) @binding(2) var leanOut: texture_storage_2d_array<rgba16float, write>;
@group(0) @binding(3) var crossOut: texture_storage_2d_array<rgba16float, write>;

const W: i32 = ${SURF_W};
const H: i32 = ${SURF_H};
const PI: f32 = 3.14159265358979;

fn hAt(x: i32, y: i32, layer: i32) -> f32 {
  // wrap in longitude, clamp in latitude
  let xi = (x % W + W) % W;
  let yi = clamp(y, 0, H - 1);
  return bitcast<f32>(textureLoad(auxIn, vec2i(xi, yi), layer, 0).r);
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  if (gid.x >= u32(W) || gid.y >= u32(H)) { return; }
  let layer = i32(gid.z);
  let x = i32(gid.x);
  let y = i32(gid.y);
  let radius = R[gid.z].p.w;                       // km
  let lat = (0.5 - (f32(y) + 0.5) / f32(H)) * PI;
  // arc length of one texel, km. Near the poles the east texel shrinks to nothing, so it is
  // clamped: without this the slope estimate explodes and the moments blow up.
  let dEast = max(radius * cos(lat) * (2.0 * PI / f32(W)), radius * 0.0015);
  let dNorth = radius * (PI / f32(H));
  let sx = (hAt(x + 1, y, layer) - hAt(x - 1, y, layer)) / (2.0 * dEast);
  let sy = (hAt(x, y - 1, layer) - hAt(x, y + 1, layer)) / (2.0 * dNorth);
  textureStore(leanOut, vec2i(gid.xy), layer, vec4f(sx, sy, sx * sx, sy * sy));
  let gl = bitcast<f32>(textureLoad(auxIn, vec2i(gid.xy), layer, 0).g);
  textureStore(crossOut, vec2i(gid.xy), layer, vec4f(sx * sy, gl, 0.0, 0.0));
}
`;

// --------------------------------------------------------------------------------------
// Mip reduction. Plain 2x2 box averages -- which is the whole point: averaging the second
// moments of a wrapped 2x2 neighbourhood is what turns geometry into a slope distribution.
// --------------------------------------------------------------------------------------
const MIP_BODY = (fmt, chans) => /* wgsl */ `
@group(0) @binding(0) var src: texture_2d_array<f32>;
@group(0) @binding(1) var dst: texture_storage_2d_array<${fmt}, write>;
@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let dim = textureDimensions(dst);
  if (gid.x >= dim.x || gid.y >= dim.y) { return; }
  let sd = vec2i(textureDimensions(src));
  let layer = i32(gid.z);
  let b = vec2i(gid.xy) * 2;
  // wrap in x (longitude is periodic), clamp in y
  let x0 = (b.x % sd.x + sd.x) % sd.x;
  let x1 = ((b.x + 1) % sd.x + sd.x) % sd.x;
  let y0 = clamp(b.y, 0, sd.y - 1);
  let y1 = clamp(b.y + 1, 0, sd.y - 1);
  let v = (textureLoad(src, vec2i(x0, y0), layer, 0) + textureLoad(src, vec2i(x1, y0), layer, 0) +
           textureLoad(src, vec2i(x0, y1), layer, 0) + textureLoad(src, vec2i(x1, y1), layer, 0)) * 0.25;
  textureStore(dst, vec2i(gid.xy), layer, ${chans});
}
`;

export const MIP_RGBA16 = MIP_BODY('rgba16float', 'v');
