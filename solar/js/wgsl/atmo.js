// Planetary atmospheres, following Hillaire's parameterisation but with the medium supplied
// from uniforms, so Earth, Mars, Titan and Saturn all run the same code with different
// coefficients. Only the body the camera is nearest gets the full treatment; everything
// further out uses the cheap analytic shell in bodies.js, which is all a few-pixel disc can
// resolve anyway.
//
// Mars is the interesting one to get right: its sky is butterscotch because suspended dust
// absorbs blue in the multiply-scattered light, yet the aureole around the sun is *blue*
// because the same dust has a strong forward diffraction lobe that favours short
// wavelengths. Giving the dust a colour-dependent scattering coefficient and a separate
// colour-dependent absorption reproduces both from one medium.
import { TRANS_W, TRANS_H, MS_SIZE } from './common.js';

export const ATMO_CORE = /* wgsl */ `
// F.atmoR = (Rg, Rt, Hrayleigh, Hmie)            Mm
// F.atmoS = (rayleigh scattering rgb / Mm, mie g)
// F.atmoM = (mie scattering rgb / Mm, mie absorption / Mm)
// F.atmoO = (extra absorber rgb / Mm, ground albedo)
// F.flags.w = 1 when the extra absorber sits in a tent layer (ozone) rather than
//              following the aerosol profile (dust)
struct Medium { scatR: vec3f, scatM: vec3f, ext: vec3f };

fn mediumAt(h: f32) -> Medium {
  let hh = max(h, 0.0);
  let dR = exp(-hh / max(F.atmoR.z, 1e-6));
  let dM = exp(-hh / max(F.atmoR.w, 1e-6));
  // ozone: a tent centred at 25 km, 30 km thick (expressed in Mm)
  let dO = select(dM, max(0.0, 1.0 - abs(hh - 0.025) / 0.015), F.flags.w > 0.5);
  var m: Medium;
  m.scatR = F.atmoS.xyz * dR;
  m.scatM = F.atmoM.xyz * dM;
  m.ext = m.scatR + m.scatM + vec3f(F.atmoM.w) * dM + F.atmoO.xyz * dO;
  return m;
}

fn rhoOf(r: f32) -> f32 { return sqrt(max(0.0, (r - F.atmoR.x) * (r + F.atmoR.x))); }

fn transRMuToUV(r: f32, mu: f32) -> vec2f {
  let Rg = F.atmoR.x;
  let Rt = F.atmoR.y;
  let Hh = sqrt(max(Rt * Rt - Rg * Rg, 1e-12));
  let rho = rhoOf(r);
  let disc = r * r * (mu * mu - 1.0) + Rt * Rt;
  let d = max(0.0, -r * mu + sqrt(max(disc, 0.0)));
  let dMin = Rt - r;
  let dMax = rho + Hh;
  return vec2f((d - dMin) / max(dMax - dMin, 1e-9), rho / Hh);
}

fn transUVToRMu(uv: vec2f) -> vec2f {
  let Rg = F.atmoR.x;
  let Rt = F.atmoR.y;
  let Hh = sqrt(max(Rt * Rt - Rg * Rg, 1e-12));
  let rho = Hh * uv.y;
  let r = sqrt(rho * rho + Rg * Rg);
  let dMin = Rt - r;
  let dMax = rho + Hh;
  let d = dMin + uv.x * (dMax - dMin);
  var mu = 1.0;
  if (d > 0.0) { mu = (Hh * Hh - rho * rho - d * d) / (2.0 * r * d); }
  return vec2f(r, clamp(mu, -1.0, 1.0));
}

fn fromUnitToSubUvs(u: f32, res: f32) -> f32 { return (u + 0.5 / res) * (res / (res + 1.0)); }
fn fromSubUvsToUnit(u: f32, res: f32) -> f32 { return (u - 0.5 / res) * (res / (res - 1.0)); }

fn sunBelowHorizon(r: f32, mu: f32) -> bool { return mu < -rhoOf(r) / max(r, 1e-6); }
`;

// Sun-to-point transmittance, 256 x 64, parameterised by (radius, cos sun angle).
export const TRANS_LUT = /* wgsl */ `
@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  if (gid.x >= ${TRANS_W}u || gid.y >= ${TRANS_H}u) { return; }
  let uv = vec2f(
    fromSubUvsToUnit((f32(gid.x) + 0.5) / ${TRANS_W}.0, ${TRANS_W}.0),
    fromSubUvsToUnit((f32(gid.y) + 0.5) / ${TRANS_H}.0, ${TRANS_H}.0));
  let rm = transUVToRMu(uv);
  let ro = vec3f(0.0, rm.x, 0.0);
  let rd = vec3f(sqrt(max(1.0 - rm.y * rm.y, 0.0)), rm.y, 0.0);
  // distance to the top of the atmosphere (the LUT only covers unobstructed paths)
  let tTop = raySphere(ro, rd, F.atmoR.y).y;
  let tGnd = raySphere(ro, rd, F.atmoR.x).x;
  let tMax = select(tTop, min(tTop, tGnd), tGnd > 0.0);
  var depth = vec3f(0.0);
  const STEPS: i32 = 48;
  for (var i = 0; i < STEPS; i++) {
    let t = tMax * (f32(i) + 0.5) / f32(STEPS);
    let p = ro + rd * t;
    depth += mediumAt(length(p) - F.atmoR.x).ext;
  }
  depth *= tMax / f32(STEPS);
  textureStore(transOut, vec2i(gid.xy), vec4f(exp(-depth), 1.0));
}
`;

// Isotropic multiple scattering, 32 x 32. Hillaire's second-order estimate summed into an
// infinite series, which is what keeps hazy atmospheres (Titan, dusty Mars) from going flat.
export const MS_LUT = /* wgsl */ `
const SQRT_SAMPLES: i32 = 8;
@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  if (gid.x >= ${MS_SIZE}u || gid.y >= ${MS_SIZE}u) { return; }
  let uv = vec2f(
    fromSubUvsToUnit((f32(gid.x) + 0.5) / ${MS_SIZE}.0, ${MS_SIZE}.0),
    fromSubUvsToUnit((f32(gid.y) + 0.5) / ${MS_SIZE}.0, ${MS_SIZE}.0));
  let cosSun = uv.x * 2.0 - 1.0;
  let r = mix(F.atmoR.x, F.atmoR.y, uv.y);
  let ro = vec3f(0.0, r, 0.0);
  let sunDir = normalize(vec3f(sqrt(max(1.0 - cosSun * cosSun, 0.0)), cosSun, 0.0));

  var lumTotal = vec3f(0.0);
  var fmsTotal = vec3f(0.0);
  const N: i32 = SQRT_SAMPLES * SQRT_SAMPLES;
  for (var i = 0; i < SQRT_SAMPLES; i++) {
    for (var j = 0; j < SQRT_SAMPLES; j++) {
      // uniform directions over the sphere
      let theta = PI * (f32(i) + 0.5) / f32(SQRT_SAMPLES);
      let phi = acos(1.0 - 2.0 * ((f32(j) + 0.5) / f32(SQRT_SAMPLES)));
      let rd = vec3f(cos(theta) * sin(phi), cos(phi), sin(theta) * sin(phi));

      let tTop = raySphere(ro, rd, F.atmoR.y).y;
      let tGnd = raySphere(ro, rd, F.atmoR.x).x;
      let hitGround = tGnd > 0.0;
      let tMax = select(tTop, min(tTop, tGnd), hitGround);
      if (tMax <= 0.0) { continue; }

      var lum = vec3f(0.0);
      var fms = vec3f(0.0);
      var trans = vec3f(1.0);
      const STEPS: i32 = 20;
      let dt = tMax / f32(STEPS);
      for (var s = 0; s < STEPS; s++) {
        let t = dt * (f32(s) + 0.5);
        let p = ro + rd * t;
        let pr = length(p);
        let m = mediumAt(pr - F.atmoR.x);
        let stepT = exp(-m.ext * dt);
        let up = p / pr;
        let mu = dot(up, sunDir);
        let sunT = textureSampleLevel(transLUT, sampLin, transRMuToUV(pr, mu), 0.0).rgb;
        let shadow = select(1.0, 0.0, sunBelowHorizon(pr, mu));
        // isotropic phase (1/4pi) for the multiple-scattering estimate
        let scat = (m.scatR + m.scatM) * (1.0 / (4.0 * PI));
        let safeExt = max(m.ext, vec3f(1e-9));
        let intS = (scat - scat * stepT) / safeExt;
        // lum is a radiance and is averaged over directions below, so the isotropic phase
        // stays in. fms is the *fraction* re-scattered, which is that same integral taken
        // over the whole sphere, so it gets the 4pi back. Conflating the two is a factor of
        // 4pi, and it buries a planet under about twelve times too much haze.
        lum += trans * intS * sunT * shadow;
        fms += trans * intS * 4.0 * PI;
        trans *= stepT;
      }
      if (hitGround) {
        let pg = ro + rd * tMax;
        let upg = normalize(pg);
        let mug = dot(upg, sunDir);
        let sunT = textureSampleLevel(transLUT, sampLin, transRMuToUV(F.atmoR.x, mug), 0.0).rgb;
        // ground bounce: albedo * E * cos / pi is already a radiance
        lum += trans * sunT * max(mug, 0.0) * F.atmoO.w / PI;
      }
      lumTotal += lum / f32(N);
      fmsTotal += fms / f32(N);
    }
  }
  // sum the infinite series of further scattering orders
  let ms = lumTotal / max(vec3f(1.0) - fmsTotal, vec3f(1e-4));
  textureStore(msOut, vec2i(gid.xy), vec4f(ms, 1.0));
}
`;

// --------------------------------------------------------------------------------------
// Ray-marched atmosphere, used for the shell around the active body. Returns the light
// scattered into the ray and the transmittance through it, so the caller can composite a
// surface (or empty space) behind it.
// --------------------------------------------------------------------------------------
export const ATMO_SHADE = /* wgsl */ `
struct AtmoOut { inscat: vec3f, trans: vec3f };

fn msLookup(r: f32, muSun: f32) -> vec3f {
  let u = clamp(muSun * 0.5 + 0.5, 0.0, 1.0);
  let v = clamp((r - F.atmoR.x) / max(F.atmoR.y - F.atmoR.x, 1e-9), 0.0, 1.0);
  return textureSampleLevel(msLUT, sampLin, vec2f(fromUnitToSubUvs(u, ${MS_SIZE}.0),
                                                  fromUnitToSubUvs(v, ${MS_SIZE}.0)), 0.0).rgb;
}

fn sunTransmit(r: f32, muSun: f32) -> vec3f {
  if (sunBelowHorizon(r, muSun)) { return vec3f(0.0); }
  return textureSampleLevel(transLUT, sampLin, transRMuToUV(clamp(r, F.atmoR.x, F.atmoR.y), muSun), 0.0).rgb;
}

// ro is relative to the planet centre, in Mm. [t0,t1] is the segment inside the shell.
fn marchAtmosphere(ro: vec3f, rd: vec3f, t0: f32, t1: f32, sunDir: vec3f,
                   steps: i32, sunRad: vec3f) -> AtmoOut {
  var o: AtmoOut;
  o.inscat = vec3f(0.0);
  o.trans = vec3f(1.0);
  if (t1 <= t0) { return o; }
  let cosT = dot(rd, sunDir);
  let pR = phaseRayleigh(cosT);
  let pM = phaseHG(cosT, F.atmoS.w);
  let dt = (t1 - t0) / f32(steps);
  for (var i = 0; i < steps; i++) {
    let t = t0 + dt * (f32(i) + 0.5);
    let p = ro + rd * t;
    let pr = length(p);
    let m = mediumAt(pr - F.atmoR.x);
    let stepT = exp(-m.ext * dt);
    let up = p / pr;
    let mu = dot(up, sunDir);
    let sT = sunTransmit(pr, mu);
    let ms = msLookup(pr, mu);
    let scat = m.scatR * pR + m.scatM * pM;
    let scatIso = m.scatR + m.scatM;
    let safeExt = max(m.ext, vec3f(1e-9));
    // analytic integration of the single-scattering term across the step
    let inS = (scat * sT + scatIso * ms) * sunRad;
    o.inscat += o.trans * (inS - inS * stepT) / safeExt;
    o.trans *= stepT;
  }
  return o;
}
`;
