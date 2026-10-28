// Bloom (dual-filter chain) and final composite with AgX tonemapping.
// Carried over from the ocean renderer unchanged apart from the comment: an 8-level bloom
// and AgX are exactly what an HDR scene containing both the solar photosphere and a 14th
// magnitude star needs.
import { FS_TRI } from './common.js';

export const BLOOM_DOWN = FS_TRI + /* wgsl */ `
struct P { texel: vec4f };   // xy = 1/src size, z = first pass (karis + threshold)
@group(0) @binding(0) var src: texture_2d<f32>;
@group(0) @binding(1) var samp: sampler;
@group(0) @binding(2) var<uniform> U: P;
fn luma(c: vec3f) -> f32 { return dot(c, vec3f(0.2126, 0.7152, 0.0722)); }
fn tap(uv: vec2f) -> vec3f { return textureSampleLevel(src, samp, uv, 0.0).rgb; }
fn karis(c: vec3f) -> f32 { return 1.0 / (1.0 + luma(c) * 0.25); }
@fragment fn fs(in: VO) -> @location(0) vec4f {
  let t = U.texel.xy;
  let uv = in.uv;
  let a = tap(uv + t * vec2f(-2.0, -2.0));
  let b = tap(uv + t * vec2f(0.0, -2.0));
  let c = tap(uv + t * vec2f(2.0, -2.0));
  let d = tap(uv + t * vec2f(-2.0, 0.0));
  let e = tap(uv);
  let f = tap(uv + t * vec2f(2.0, 0.0));
  let g = tap(uv + t * vec2f(-2.0, 2.0));
  let h = tap(uv + t * vec2f(0.0, 2.0));
  let i = tap(uv + t * vec2f(2.0, 2.0));
  let j = tap(uv + t * vec2f(-1.0, -1.0));
  let k = tap(uv + t * vec2f(1.0, -1.0));
  let l = tap(uv + t * vec2f(-1.0, 1.0));
  let m = tap(uv + t * vec2f(1.0, 1.0));
  var o: vec3f;
  if (U.texel.z > 0.5) {
    // Karis average on the first downsample to tame fireflies from sun glints
    let g0 = (a + b + d + e) * 0.25;
    let g1 = (b + c + e + f) * 0.25;
    let g2 = (d + e + g + h) * 0.25;
    let g3 = (e + f + h + i) * 0.25;
    let g4 = (j + k + l + m) * 0.25;
    let w0 = karis(g0) * 0.125; let w1 = karis(g1) * 0.125; let w2 = karis(g2) * 0.125;
    let w3 = karis(g3) * 0.125; let w4 = karis(g4) * 0.5;
    o = (g0 * w0 + g1 * w1 + g2 * w2 + g3 * w3 + g4 * w4) / (w0 + w1 + w2 + w3 + w4);
  } else {
    o = e * 0.125 + (a + c + g + i) * 0.03125 + (b + d + f + h) * 0.0625 + (j + k + l + m) * 0.125;
  }
  return vec4f(min(o, vec3f(6.0e4)), 1.0);
}
`;

export const BLOOM_UP = FS_TRI + /* wgsl */ `
struct P { texel: vec4f };   // xy = 1/src size, z = radius, w = weight
@group(0) @binding(0) var src: texture_2d<f32>;
@group(0) @binding(1) var samp: sampler;
@group(0) @binding(2) var<uniform> U: P;
fn tap(uv: vec2f) -> vec3f { return textureSampleLevel(src, samp, uv, 0.0).rgb; }
@fragment fn fs(in: VO) -> @location(0) vec4f {
  let t = U.texel.xy * U.texel.z;
  let uv = in.uv;
  var o = tap(uv) * 4.0;
  o += (tap(uv + vec2f(-t.x, 0.0)) + tap(uv + vec2f(t.x, 0.0)) + tap(uv + vec2f(0.0, -t.y)) + tap(uv + vec2f(0.0, t.y))) * 2.0;
  o += tap(uv + vec2f(-t.x, -t.y)) + tap(uv + vec2f(t.x, -t.y)) + tap(uv + vec2f(-t.x, t.y)) + tap(uv + vec2f(t.x, t.y));
  return vec4f(o * (U.texel.w / 16.0), 1.0);
}
`;

export const COMPOSITE = FS_TRI + /* wgsl */ `
struct P { p: vec4f, q: vec4f };  // p: exposure, bloom, frame, vignette; q: saturation, contrast, warmth, grain
@group(0) @binding(0) var hdr: texture_2d<f32>;
@group(0) @binding(1) var bloom: texture_2d<f32>;
@group(0) @binding(2) var samp: sampler;
@group(0) @binding(3) var<uniform> U: P;

fn agxContrast(x: vec3f) -> vec3f {
  let x2 = x * x;
  let x4 = x2 * x2;
  return 15.5 * x4 * x2 - 40.14 * x4 * x + 31.96 * x4 - 6.868 * x2 * x + 0.4298 * x2 + 0.1191 * x - 0.00232;
}
fn agx(v: vec3f) -> vec3f {
  let m = mat3x3f(0.842479062253094, 0.0423282422610123, 0.0423756549057051,
                  0.0784335999999992, 0.878468636469772, 0.0784336,
                  0.0792237451477643, 0.0791661274605434, 0.879142973793104);
  let minEv = -12.47393;
  let maxEv = 4.026069;
  var x = m * max(v, vec3f(1e-10));
  x = clamp(log2(x), vec3f(minEv), vec3f(maxEv));
  x = (x - minEv) / (maxEv - minEv);
  return agxContrast(x);
}
fn agxEotf(v: vec3f) -> vec3f {
  let mi = mat3x3f(1.19687900512017, -0.0528968517574562, -0.0529716355144438,
                   -0.0980208811401368, 1.15190312990417, -0.0980434501171241,
                   -0.0990297440797205, -0.0989611768448433, 1.15107367264116);
  return mi * v;
}
fn agxLook(v: vec3f, sat: f32, power: f32) -> vec3f {
  let l = dot(v, vec3f(0.2126, 0.7152, 0.0722));
  let p = pow(max(v, vec3f(0.0)), vec3f(power));
  return l + sat * (p - l);
}
fn ign(p: vec2f) -> f32 { return fract(52.9829189 * fract(dot(p, vec2f(0.06711056, 0.00583715)))); }

@fragment fn fs(in: VO) -> @location(0) vec4f {
  let dims = vec2f(textureDimensions(hdr));
  let pc = vec2i(in.pos.xy);
  var c = textureLoad(hdr, pc, 0).rgb;
  // the up-chain accumulates 8 blur levels
  let b = textureSampleLevel(bloom, samp, in.uv, 0.0).rgb * (1.0 / 8.0);
  c = mix(c, b, U.p.y);
  c *= U.p.x;
  // subtle vignette
  let d = (in.uv - 0.5) * vec2f(dims.x / dims.y, 1.0);
  c *= 1.0 - U.p.w * dot(d, d);
  c *= vec3f(1.0 + U.q.z, 1.0, 1.0 - U.q.z);
  // film/sensor-like highlight desaturation: very bright glints and the sun go towards white
  let lum = dot(c, vec3f(0.2126, 0.7152, 0.0722));
  c = mix(c, vec3f(lum), smoothstep(0.7, 6.0, lum) * 0.55);
  var o = agx(c);
  o = agxLook(o, U.q.x, U.q.y);
  o = agxEotf(o);
  // triangular dither against banding in the sky gradients
  let n = ign(in.pos.xy + U.p.z * 5.588238) + ign(in.pos.xy * 1.37 + 17.0 + U.p.z * 3.1) - 1.0;
  o += n / 255.0;
  return vec4f(clamp(o, vec3f(0.0), vec3f(1.0)), 1.0);
}
`;
