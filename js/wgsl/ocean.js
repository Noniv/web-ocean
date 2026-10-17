// Ocean surface + sky background rendering.
import { FRAME_WGSL, ATMO_CORE_WGSL, LIGHT_WGSL, ENV_MIPS, ENV_SIZE, N, CASCADES } from './common.js';
import { SKY_RENDER_FUNCS, SKY_CLOUD_FUNCS } from './sky.js';

const RENDER_BINDINGS = /* wgsl */ `
@group(0) @binding(0) var<uniform> F: Frame;
@group(0) @binding(1) var transLUT: texture_2d<f32>;
@group(0) @binding(2) var skyLUT: texture_2d<f32>;
@group(0) @binding(3) var sampC: sampler;
`;

// Full-resolution sky + clouds. Cloud silhouettes are anti-aliased analytically: the
// screen-space derivative of each layer's edge function sets the width of its opacity edge.
export const SKY_BG = FRAME_WGSL + ATMO_CORE_WGSL + RENDER_BINDINGS + SKY_RENDER_FUNCS + SKY_CLOUD_FUNCS + /* wgsl */ `
struct VO { @builtin(position) pos: vec4f, @location(0) ndc: vec2f };
@vertex fn vs(@builtin(vertex_index) i: u32) -> VO {
  let p = vec2f(f32((i << 1u) & 2u), f32(i & 2u)) * 2.0 - 1.0;
  var o: VO;
  o.pos = vec4f(p, 0.0, 1.0);
  o.ndc = p;
  return o;
}
@fragment fn fs(in: VO) -> @location(0) vec4f {
  let dir = normalize(F.camFwd.xyz + in.ndc.x * F.camRight.xyz + in.ndc.y * F.camUp.xyz);
  let pixAng = 2.0 * length(F.camUp.xyz) * F.screen.w;
  var d2 = dir;
  d2.y = max(d2.y, 0.0005);
  d2 = normalize(d2);
  // edge functions at this pixel (uniform control flow, so fwidth is valid)
  let hl = hitLayer(d2, F.cloudP.x);
  let el = lowEdge(hl.q, hl.t * pixAng / hl.mu, 7);
  let hh = hitLayer(d2, 4.5);
  let eh = highEdge(hh.q, hh.t * pixAng / hh.mu, 6);
  let aa = vec2f(fwidth(el), fwidth(eh));
  var sunT = 1.0;
  // reuse the edge values: cloudLayers would evaluate the same two noise stacks again
  let cl = cloudLayersE(d2, pixAng, aa, vec2f(el, eh), true, &sunT);
  let L = skyAtmo(d2) * (1.0 - cl.a) + cl.rgb + sunDisk(dir) * sunT;
  return vec4f(L, 1.0);
}
`;

// The surface is drawn twice: a depth-only pass (fsDepth) and then the shading pass with
// depthCompare 'equal', so the expensive fragment shader runs only for the front-most surface
// instead of for every wave face hidden behind a nearer crest. @invariant keeps the depth of
// both passes identical.
export const OCEAN = FRAME_WGSL + ATMO_CORE_WGSL + LIGHT_WGSL + RENDER_BINDINGS + /* wgsl */ `
@group(0) @binding(4) var dispTex: texture_2d_array<f32>;
@group(0) @binding(5) var derivTex: texture_2d_array<f32>;
@group(0) @binding(6) var momTex: texture_2d_array<f32>;
@group(0) @binding(7) var foamTex: texture_2d<f32>;
@group(0) @binding(8) var envCube: texture_cube<f32>;
@group(0) @binding(9) var sampR: sampler;
@group(0) @binding(10) var foamWorld: texture_2d_array<f32>;
@group(0) @binding(11) var<storage, read> light: Light;
` + SKY_RENDER_FUNCS + /* wgsl */ `
const ENV_MAXMIP: f32 = ${ENV_MIPS - 1}.0;
const ENV_TEXELS_PER_RAD: f32 = ${ENV_SIZE}.0 / 1.5708;
const TEX_N: f32 = ${N}.0;

struct VSOut {
  @builtin(position) @invariant pos: vec4f,
  @location(0) rel: vec3f,     // displaced position relative to the camera
  @location(1) xz: vec2f,      // undisplaced (lagrangian) coordinates
  @location(2) height: f32,
};

@vertex fn vs(@builtin(vertex_index) vi: u32, @location(0) inst: vec4f) -> VSOut {
  let n = u32(F.lodP.y);
  let g = vec2f(f32(vi % (n + 1u)), f32(vi / (n + 1u)));
  let spacing = inst.z / f32(n);
  var xz = inst.xy + g * spacing;
  // CDLOD morph towards the next coarser level
  let R = F.lodP.x * exp2(inst.w);
  let d = distance(vec3f(xz.x, 0.0, xz.y), F.camPos.xyz);
  let mk = clamp((d - 0.8 * R) / (0.2 * R), 0.0, 1.0);
  xz -= fract(g * 0.5) * 2.0 * spacing * mk;
  let eff = spacing * (1.0 + mk);
  var disp = vec3f(0.0);
  for (var c = 0u; c < ${CASCADES}u; c++) {
    let texel = F.cascL[c >> 2u][c & 3u] / TEX_N;
    let lod = log2(max(eff / texel, 1.0)) + 0.5;
    if (lod < 8.0) {
      disp += textureSampleLevel(dispTex, sampR, xz * F.cascInvL[c >> 2u][c & 3u], c, lod).xyz;
    }
  }
  var o: VSOut;
  let world = vec3f(xz.x + disp.x, disp.y, xz.y + disp.z);
  o.rel = world - F.camPos.xyz;
  o.pos = F.viewProj * vec4f(o.rel, 1.0);
  o.xz = xz;
  o.height = disp.y;
  return o;
}

fn fresnelWater(c: f32) -> f32 {
  // Schlick with F0 for n = 1.333
  let x = 1.0 - clamp(c, 0.0, 1.0);
  let x2 = x * x;
  return 0.02 + 0.98 * x2 * x2 * x;
}

fn lambdaSmith(w: vec3f, cov: vec3f) -> f32 {
  let xz2 = max(dot(w.xz, w.xz), 1e-8);
  let il = inverseSqrt(xz2);
  let az = w.xz * il;
  let sig2 = az.x * az.x * cov.x + az.y * az.y * cov.y + 2.0 * az.x * az.y * cov.z;
  let cotT = w.y * il;
  let a = max(cotT, 1e-3) / sqrt(2.0 * max(sig2, 1e-7));
  if (a >= 1.6) { return 0.0; }
  return (1.0 - 1.259 * a + 0.396 * a * a) / (3.535 * a + 2.181 * a * a);
}

// true when the condition holds anywhere in this pixel's 2x2 quad (helper lanes included);
// built from fine derivatives, so it must be called in uniform control flow
fn quadAny(c: bool) -> bool {
  let v = select(0.0, 1.0, c);
  let h = v + abs(dpdxFine(v));
  return h + abs(dpdyFine(h)) > 0.0;
}

// only called from a quad-uniform branch (see quadAny), where implicit derivatives are valid
@diagnostic(off, derivative_uniformity)
fn foamPattern(xz: vec2f, t: f32) -> f32 {
  let a = textureSample(foamTex, sampR, xz * 0.23 + vec2f(0.013, 0.0) * t).r;
  let b = textureSample(foamTex, sampR, xz * 0.71 + vec2f(0.37, 0.71)).g;
  return a * 0.78 + b * 0.22;
}

@fragment fn fs(in: VSOut) -> @location(0) vec4f {
  let xz = in.xz;
  var s = vec2f(0.0);
  var jxx = 0.0;
  var jzz = 0.0;
  var jxz = 0.0;
  var cov = vec3f(0.0);
  // wind gusts ("cat's paws"): patches where the short ripples are stronger or calmer
  let gustN = textureSample(foamTex, sampR, xz * 0.011 + F.windP[0].xy * (F.camPos.w * 0.0004)).b;
  let gust = mix(0.35, 1.2, smoothstep(0.3, 0.72, gustN));
  for (var c = 0u; c < ${CASCADES}u; c++) {
    let uv = xz * F.cascInvL[c >> 2u][c & 3u];
    let gk = mix(1.0, gust, select(0.0, select(0.5, 1.0, c >= 4u), c >= 3u));
    let dv = textureSample(derivTex, sampR, uv, c) * gk;
    // momTex.w carries the jxz Jacobian term (also in dispTex.w), saving a fetch per cascade
    let m = textureSample(momTex, sampR, uv, c);
    let mv = m * (gk * gk);
    s += dv.xy;
    jxx += dv.z;
    jzz += dv.w;
    jxz += m.w * gk;
    cov += vec3f(max(mv.x - dv.x * dv.x, 0.0), max(mv.y - dv.y * dv.y, 0.0), mv.z - dv.x * dv.y);
  }
  let dist = length(in.rel);
  let V = -in.rel / dist;
  let Ld = F.sunDir.xyz;

  // slope statistics (LEAN): add unresolved capillary roughness + sun disk size
  // unresolved roughness; grows with distance to reach Cox-Munk-like totals in the far sheen
  let unres = F.waterCol.w + 0.0035 * smoothstep(30.0, 500.0, dist);
  cov.x += unres;
  cov.y += unres;
  let jx = max(1.0 + jxx, 0.15);
  let jz = max(1.0 + jzz, 0.15);
  let sl = vec2f(s.x / jx, s.y / jz);
  let nrm = normalize(vec3f(-sl.x, 1.0, -sl.y));
  // specular anti-aliasing: slope variation across the pixel widens the lobe
  let dsx = dpdx(sl);
  let dsy = dpdy(sl);
  let aa = vec3f(dsx.x * dsx.x + dsy.x * dsy.x, dsx.y * dsx.y + dsy.y * dsy.y, dsx.x * dsx.y + dsy.x * dsy.y) * 0.5;
  cov += aa * min(1.0, 0.03 / max(max(aa.x, aa.y), 1e-8));
  let J = (1.0 + jxx) * (1.0 + jzz) - jxz * jxz;

  // frame constants (see SUN_VIS): sun illuminance, zenith sky, upper-hemisphere average
  let sunE = light.sunE.rgb;
  let skyUp = light.skyUp.rgb;
  let skyHemi = light.skyHemi.rgb;
  let skyN = textureSampleLevel(envCube, sampC, nrm, ENV_MAXMIP - 1.0).rgb;

  // ---- sun glitter: anisotropic Gaussian slope distribution ----
  let Hv = normalize(V + Ld);
  let hs = -Hv.xz / max(Hv.y, 1e-3);
  let dsl = hs - sl;
  let det = max(cov.x * cov.y - cov.z * cov.z, 1e-10);
  let e = (dsl.x * dsl.x * cov.y + dsl.y * dsl.y * cov.x - 2.0 * dsl.x * dsl.y * cov.z) / det;
  let pdf = exp(-0.5 * e) / (2.0 * PI * sqrt(det));
  let hy2 = Hv.y * Hv.y;
  let Dm = pdf / max(hy2 * hy2, 1e-6);
  let G = 1.0 / (1.0 + lambdaSmith(V, cov) + lambdaSmith(Ld, cov));
  let Fh = fresnelWater(dot(V, Hv));
  let spec = sunE * (Fh * Dm * G / (4.0 * max(V.y, 0.02))) * step(0.0, Ld.y);

  // ---- sky reflection ----
  // At grazing angles the *visible* facets are biased towards the viewer (projected-area
  // weighting + masking), so the effective reflector tilts up and sees higher, greyer sky.
  let sig = sqrt(max(cov.x + cov.y, 1e-6) * 0.5);
  let vh = normalize(vec3f(V.x, 0.0, V.z) + vec3f(1e-5, 0.0, 0.0));
  let sigV = sqrt(max(vh.x * vh.x * cov.x + vh.z * vh.z * cov.y + 2.0 * vh.x * vh.z * cov.z, 1e-6));
  let aV = max(V.y, 1e-3) / max(sqrt(1.0 - V.y * V.y), 1e-3) / (1.4142 * sigV);
  let tilt = 0.32 * sigV * (1.0 - smoothstep(0.0, 2.5, aV));
  let nV = normalize(nrm + vh * tilt);
  var R = reflect(-V, nV);
  R.y = abs(R.y) * 0.4 + max(R.y, 0.0) * 0.6;
  R = normalize(R);
  let envLod = clamp(log2(max(2.0 * sig * ENV_TEXELS_PER_RAD, 1.0)), 0.0, ENV_MAXMIP);
  let refl = textureSampleLevel(envCube, sampC, R, envLod).rgb;
  let NdV = max(dot(nV, V), 0.0);
  // Fresnel from a partially smoothed normal: the long-wave facet plus the grazing-angle
  // statistics of everything finer (visible facets at grazing incidence stay highly reflective)
  let nF = normalize(mix(nrm, vec3f(0.0, 1.0, 0.0), 0.45));
  let Fr = fresnelWater(max(dot(nF, V), 0.0));

  // ---- water body: upwelling scatter + backlit subsurface glow ----
  let ambient = skyUp * 1.3 + skyN * 0.5;
  let NdL = max(dot(nrm, Ld), 0.0);
  // upwelling light is diffuse: light it with smooth terms only (no per-pixel normal noise)
  var body = F.waterCol.rgb * (skyHemi * 1.6 + sunE * max(Ld.y, 0.0) * 0.8);
  // light transmitted through the thin upper part of backlit crests
  let crest = clamp((in.height - 0.05) / 0.9, 0.0, 1.0);
  let toSun = max(dot(Ld, -V), 0.0);
  let thin = pow(clamp(0.5 - 0.5 * dot(Ld, nrm), 0.0, 1.0), 3.0);
  let k1 = F.sssCol.w * crest * crest * pow(toSun, 4.0) * thin * exp(-dist / 600.0);
  body += F.sssCol.rgb * sunE * k1;

  // ---- foam ----
  // persistent foam from the world-space clipmaps (near: fine, far: coarse), plus
  // instantaneous crest foam from the per-pixel Jacobian for anything beyond them
  let dc = abs(xz - F.camPos.xz);
  let cheb = max(dc.x, dc.y);
  let w0 = 1.0 - smoothstep(0.30 * F.lodP.z, 0.45 * F.lodP.z, cheb);
  let w1 = 1.0 - smoothstep(0.30 * F.lodP.w, 0.45 * F.lodP.w, cheb);
  // Texture work that cannot change this pixel is skipped, but only for whole 2x2 quads
  // (quadAny), so the implicit derivatives of every fetch that does run are unchanged. A
  // clipmap level with weight 0 contributes exactly 0, as does the foam pattern where there
  // is no cover and no fresh crest foam, so the result is the same as always fetching.
  let dbg = F.post.w > 0.5;
  var fw0 = vec4f(0.0);
  var fw1 = vec4f(0.0);
  @diagnostic(off, derivative_uniformity) if (quadAny(w0 > 0.0 || dbg)) {
    fw0 = textureSample(foamWorld, sampR, xz / F.lodP.z, 0u);
  }
  @diagnostic(off, derivative_uniformity) if (quadAny(w1 > 0.0 || dbg)) {
    fw1 = textureSample(foamWorld, sampR, xz / F.lodP.w, 1u);
  }
  let fwl = mix(fw1, fw0, w0) * w1;
  let fresh = clamp((F.foamP.x - J) * F.foamP.y, 0.0, 1.0);
  var cover = clamp((max(fwl.r, fresh * (1.0 - w1 * 0.7)) + fwl.g * 0.6) * F.foamP.z, 0.0, 1.5);
  var foamMask = 0.0;
  var foamLit = vec3f(0.0);
  @diagnostic(off, derivative_uniformity) if (quadAny(cover > 0.0 || fwl.g > 0.35 || dbg)) {
    // aged foam breaks into wind-aligned streaks; fresh crest foam stays dense
    let wd = F.windP[0].xy;
    let ws = vec2f(dot(xz, wd), dot(xz, vec2f(-wd.y, wd.x)));
    let streak = textureSample(foamTex, sampR, ws * vec2f(0.035, 0.11)).a;
    let freshK = clamp(fwl.g * 2.0 + fresh, 0.0, 1.0);
    cover *= mix(smoothstep(0.25, 0.75, streak) * 1.5, 1.0, freshK);
    let pat = foamPattern(xz, F.camPos.w);
    let fw = fwidth(pat) + 0.03;
    // soft threshold: thin foam is a translucent film, dense foam is opaque white
    // dense foam has crisp edges, thin decaying foam a soft translucent margin
    let soft = 0.04 + 0.14 * (1.0 - clamp(cover, 0.0, 1.0));
    let fm = smoothstep(1.0 - cover - fw - soft, 1.0 - cover + fw + soft, pat);
    foamMask = clamp(fm * smoothstep(0.0, 0.5, cover), 0.0, 1.0) * (0.68 + 0.32 * smoothstep(0.3, 0.9, cover));
    // freshly broken crests: dense, almost opaque core
    foamMask = max(foamMask, smoothstep(0.35, 0.9, fwl.g) * smoothstep(0.25, 0.6, pat + 0.25) * 0.95);
    // bubbly brightness variation inside the foam
    let bub = textureSample(foamTex, sampR, xz * 1.9 + vec2f(0.21, 0.53)).g;
    let foamAlb = vec3f(0.82, 0.87, 0.92) * (0.78 + 0.3 * bub);
    // diffuse: albedo * (sky irradiance / pi + sun * cos / pi); sky dominates at sunset
    // foam scatters light from the whole hemisphere (incl. the cool zenith): use a less saturated ambient
    let foamAmb = skyHemi * 1.0 + skyN * 0.3;
    let foamAmbN = mix(foamAmb, vec3f(dot(foamAmb, vec3f(0.2126, 0.7152, 0.0722))), 0.85);
    // grazing sun is largely self-shadowed by the foam's lumpy micro-structure
    foamLit = foamAlb * (sunE * max(dot(nrm, Ld), 0.0) * 0.318 * 0.5 + foamAmbN * 1.55);
  }
  // submerged bubbles brighten and tint the water just beneath foam
  body += F.sssCol.rgb * (sunE * 0.05 + ambient * 0.35) * clamp(cover, 0.0, 1.0);
  body += vec3f(0.10, 0.32, 0.30) * skyUp * fwl.g * 0.35;

  if (F.post.w > 0.5) {
    // debug: r = foam cover, g = clamp(J), b = world foam level-0 weight
    if (F.post.w < 1.5) { return vec4f(cover, clamp(J * 0.5, 0.0, 1.0), w0 * 0.3, 1.0) * 0.2; }
    if (F.post.w > 2.5 && F.post.w < 3.5) { return vec4f(refl, 1.0); }
    if (F.post.w > 3.5 && F.post.w < 4.5) { return vec4f(vec3f(Fr), 1.0) * 0.5; }
    if (F.post.w > 4.5 && F.post.w < 5.5) { return vec4f(body * (1.0 - Fr), 1.0); }
    if (F.post.w > 5.5 && F.post.w < 6.5) { return vec4f(foamLit, 1.0); }
    if (F.post.w > 6.5 && F.post.w < 7.5) { return vec4f(skyUp, 1.0); }
    return vec4f(fwl.r, fwl.g, fresh, 1.0) * 0.2;
  }
  var col = body * (1.0 - Fr) + refl * Fr + spec;
  col = mix(col, foamLit, foamMask * 0.95);

  // ---- aerial perspective ----
  // aerial perspective: atmosphere extinction plus a marine haze floor
  let T = exp(-max(F.fogExt.rgb, vec3f(F.fogExt.w)) * dist);
  let hz = normalize(vec3f(-V.x, 0.002, -V.z));
  let fogCol = skyAtmo(hz);
  col = col * T + fogCol * (1.0 - T);
  return vec4f(col, 1.0);
}

// depth pre-pass: no colour output (the pipeline's colour writeMask is 0)
@fragment fn fsDepth() {}
`;

// Samples the displaced surface height under the camera (inverting the horizontal
// displacement with a few fixed-point iterations).
export const HEIGHT_PROBE = FRAME_WGSL + /* wgsl */ `
@group(0) @binding(0) var<uniform> F: Frame;
@group(0) @binding(1) var dispTex: texture_2d_array<f32>;
@group(0) @binding(2) var samp: sampler;
@group(0) @binding(3) var<storage, read_write> outH: array<f32, 4>;
fn dispAt(xz: vec2f) -> vec3f {
  var d = vec3f(0.0);
  for (var c = 0u; c < ${CASCADES}u; c++) {
    d += textureSampleLevel(dispTex, samp, xz * F.cascInvL[c >> 2u][c & 3u], c, 0.0).xyz;
  }
  return d;
}
@compute @workgroup_size(1)
fn main() {
  let goal = F.camPos.xz;
  var x0 = goal;
  for (var i = 0; i < 4; i++) {
    let d = dispAt(x0);
    x0 = goal - d.xz;
  }
  outH[0] = dispAt(x0).y;
}
`;
