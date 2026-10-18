// Procedural ocean soundscape (Web Audio, no sample files, so both bundles stay
// self-contained). Every layer follows the simulation parameters and the GPU probe
// around the listener (the camera):
//   wind     pink noise bands; speed from the log wind profile at the listener's height,
//            relative to the moving camera, panned towards upwind
//   wash     low swish of the passing waves: wave height and whitecap cover, loud while
//            a crest passes under the camera and nearly silent in the troughs
//   crests   breaking bursts wherever the surface folds around the listener (the same
//            Jacobian test that injects whitecap foam), panned and attenuated by distance
//   gulls    synthetic seagull calls at a random rate and direction (panel sliders)
// Browsers only let audio start after a user gesture, so input handlers call start().
import { PROBE_PTS } from './wgsl/common.js';

export function createOceanAudio(P) {
  const G = 9.81;
  const RING_R = [2.5, 7, 16, 36];   // probe ring radii (m) at low camera heights
  const PER_RING = PROBE_PTS / RING_R.length;
  const MAX_VOICES = 48;

  // listener-relative xz offsets of the probe points (uploaded by main.js every frame)
  const probePts = new Float32Array(PROBE_PTS * 2);
  // per point: 0 calm, 1 folding (tracking the deepest Jacobian), 2 broken (sounded)
  const breaking = new Uint8Array(PROBE_PTS);
  const jMin = new Float32Array(PROBE_PTS);
  // crest onsets per probe readback, merged by direction (energy, direction, distance,
  // strength); secNext: no new crest from that sector before this time
  const SECTORS = 8;
  const secE = new Float32Array(SECTORS), secX = new Float32Array(SECTORS);
  const secZ = new Float32Array(SECTORS), secR = new Float32Array(SECTORS);
  const secS = new Float32Array(SECTORS), secNext = new Float64Array(SECTORS);
  let ringScale = 0;
  function layoutRings(s) {
    ringScale = s;
    for (let i = 0; i < PROBE_PTS; i++) {
      const ring = Math.floor(i / PER_RING);
      const a = ((i % PER_RING) + 0.5 * (ring & 1)) / PER_RING * 2 * Math.PI;
      probePts[i * 2] = Math.cos(a) * RING_R[ring] * s;
      probePts[i * 2 + 1] = Math.sin(a) * RING_R[ring] * s;
    }
    // moved points only count as new crests once they have been seen unbroken
    breaking.fill(2);
  }
  layoutRings(1);

  let ctx = null, master = null, bus = null, L = null, pinkBuf = null, fizzBuf = null;
  let enabled = true, voices = 0, gulls = 0, gullWave = null;
  // listener state: height above the water, camera right / forward (xz), smoothed velocity
  let hc = 2.4, rx = 1, rz = 0, fx = 0, fz = -1;
  let vx = 0, vy = 0, vz = 0, px = 0, py = 0, pz = 0, havePos = false, pending = 0;
  // measured: water height under the camera, fraction of probe points currently breaking
  let lastH = 0, lastProbeT = 0, cover = 0;
  let gust = 0, gustTarget = 0, gustT = 0;

  // wave statistics of the simulated spectrum (same JONSWAP fit as SIM_INIT)
  const sea = { hsW: 0, hs: 0 };
  function seaState() {
    const U = P.windSpeed, F = P.fetchKm * 1000;
    const wp = 22 * Math.cbrt(G * G / (U * F));
    const alpha = 0.076 * Math.pow(U * U / (F * G), 0.22);
    sea.hsW = 4 * Math.sqrt(alpha * G * G * (1 - 0.287 * Math.log(3.3)) / (5 * wp ** 4));
    sea.hs = Math.hypot(sea.hsW, P.swellHs);
  }

  function start() {
    if (ctx) {
      if (enabled && ctx.state === 'suspended' && !document.hidden) ctx.resume();
      return;
    }
    const AC = window.AudioContext || window.webkitAudioContext;
    if (!AC) return;
    ctx = new AC();
    build();
  }

  function toggle() {
    if (!ctx) { enabled = true; start(); return; }
    enabled = !enabled;
    master.gain.setTargetAtTime(enabled ? P.sound : 0, ctx.currentTime, 0.05);
    if (enabled) ctx.resume();
    else setTimeout(() => { if (!enabled) ctx.suspend(); }, 300);
  }

  document.addEventListener('visibilitychange', () => {
    if (!ctx) return;
    if (document.hidden) ctx.suspend();
    else if (enabled) ctx.resume();
  });

  // --------------------------------------------------------------------------------------
  // Sound sources: looped noise and bubble buffers synthesised once
  // --------------------------------------------------------------------------------------
  function build() {
    const sr = ctx.sampleRate;
    const rnd = Math.random;

    function loopBuffer(seconds, fill) {
      const n = Math.round(seconds * sr), fade = Math.round(0.3 * sr);
      const tmp = new Float32Array(n + fade);
      fill(tmp);
      // equal-power crossfade of the tail into the head: the loop has no seam
      for (let i = 0; i < fade; i++) {
        const a = 0.5 * Math.PI * i / fade;
        tmp[i] = tmp[i] * Math.sin(a) + tmp[n + i] * Math.cos(a);
      }
      let e = 0;
      for (let i = 0; i < n; i++) e += tmp[i] * tmp[i];
      const k = 0.25 / Math.sqrt(e / n + 1e-12);   // all buffers at the same RMS
      const buf = ctx.createBuffer(1, n, sr);
      const ch = buf.getChannelData(0);
      for (let i = 0; i < n; i++) ch[i] = tmp[i] * k;
      return buf;
    }
    function pinkNoise(o) {   // Paul Kellet's filter
      let b0 = 0, b1 = 0, b2 = 0, b3 = 0, b4 = 0, b5 = 0, b6 = 0;
      for (let i = 0; i < o.length; i++) {
        const w = rnd() * 2 - 1;
        b0 = 0.99886 * b0 + w * 0.0555179; b1 = 0.99332 * b1 + w * 0.0750759;
        b2 = 0.96900 * b2 + w * 0.1538520; b3 = 0.86650 * b3 + w * 0.3104856;
        b4 = 0.55000 * b4 + w * 0.5329522; b5 = -0.7616 * b5 - w * 0.0168980;
        o[i] = b0 + b1 + b2 + b3 + b4 + b5 + b6 + w * 0.5362;
        b6 = w * 0.115926;
      }
    }
    function brownNoise(o) {
      let b = 0;
      for (let i = 0; i < o.length; i++) { b = (b + 0.02 * (rnd() * 2 - 1)) / 1.02; o[i] = b; }
    }
    // van den Doel's bubble: a damped sine whose pitch rises while it rings
    function bubble(o, i0, f0, amp, xi) {
      const d = 0.043 * f0 + 0.0014 * f0 * Math.sqrt(f0);
      const n = Math.min(o.length - i0, Math.ceil(7 / d * sr));
      const decay = Math.exp(-d / sr), w0 = 2 * Math.PI * f0 / sr, chirp = xi * d / sr;
      let env = amp, ph = 0;
      for (let k = 0; k < n; k++) {
        ph += w0 * (1 + chirp * k);
        o[i0 + k] += env * Math.sin(ph);
        env *= decay;
      }
    }
    // whitecap fizz: dense small bubbles (1.3-9 kHz) over a little spray hiss
    function fizz(o) {
      const n = o.length;
      for (let c = Math.round(n / sr * 1100); c > 0; c--) {
        const f0 = 1300 * Math.pow(9000 / 1300, rnd());
        bubble(o, Math.floor(rnd() * n), f0, (0.3 + rnd()) * Math.sqrt(1300 / f0), 0.1);
      }
      let lp = 0;
      for (let i = 0; i < n; i++) { const w = rnd() * 2 - 1; lp += 0.35 * (w - lp); o[i] += 0.05 * (w - lp); }
    }

    const brownBuf = loopBuffer(9.1, brownNoise);
    pinkBuf = loopBuffer(7.3, pinkNoise);
    fizzBuf = loopBuffer(4.7, fizz);
    // gull voice: strong low harmonics, a nasal formant is added per call
    const harm = [0, 1, 0.9, 0.75, 0.5, 0.4, 0.25, 0.18, 0.1, 0.06, 0.04];
    gullWave = ctx.createPeriodicWave(new Float32Array(harm.length), Float32Array.from(harm));

    const comp = ctx.createDynamicsCompressor();
    comp.threshold.value = -14; comp.knee.value = 10; comp.ratio.value = 4;
    comp.attack.value = 0.01; comp.release.value = 0.25;
    master = ctx.createGain();
    master.gain.value = 0;
    bus = ctx.createGain();
    bus.connect(comp).connect(master).connect(ctx.destination);
    master.gain.setTargetAtTime(enabled ? P.sound : 0, ctx.currentTime, 0.4);

    function layer(buf, offset, type, freq, q, pan) {
      const src = ctx.createBufferSource();
      src.buffer = buf; src.loop = true;
      const f = ctx.createBiquadFilter();
      f.type = type; f.frequency.value = freq; f.Q.value = q;
      const g = ctx.createGain();
      g.gain.value = 0;
      const p = ctx.createStereoPanner();
      p.pan.value = pan;
      src.connect(f).connect(g).connect(p).connect(bus);
      src.start(0, offset * buf.duration);
      return { f, g, p };
    }
    L = {
      windA: layer(pinkBuf, 0.0, 'bandpass', 400, 0.8, 0),
      windB: layer(pinkBuf, 0.5, 'bandpass', 520, 0.8, 0),
      buffet: layer(brownBuf, 0.7, 'lowpass', 110, 0, 0),
      washL: layer(brownBuf, 0.1, 'lowpass', 400, 0, -0.6),
      washR: layer(brownBuf, 0.55, 'lowpass', 400, 0, 0.6),
    };
  }

  // --------------------------------------------------------------------------------------
  // Breaking crests: one-shot fizz + whoosh voices
  // --------------------------------------------------------------------------------------
  function voice(buf, freq, amp, att, tau, dest) {
    const t = ctx.currentTime;
    const s = ctx.createBufferSource();
    s.buffer = buf; s.loop = true;
    const f = ctx.createBiquadFilter();
    f.type = 'bandpass'; f.frequency.value = freq; f.Q.value = 0.7;
    const g = ctx.createGain();
    g.gain.setValueAtTime(0, t);
    g.gain.linearRampToValueAtTime(amp, t + att);
    g.gain.setTargetAtTime(0, t + att, tau);
    s.connect(f).connect(g).connect(dest);
    s.start(t, Math.random() * buf.duration);
    s.stop(t + att + tau * 3);
    return s;
  }
  // amp: loudness at the listener, (dx, dz): unit direction, r: horizontal distance (m),
  // strength: fold depth 0..1 (deeper folds ring longer)
  function crest(amp, dx, dz, r, strength) {
    if (voices >= MAX_VOICES || amp < 0.02) return false;
    voices++;
    const size = Math.max(0.2, Math.min(2, sea.hsW / 1.2));   // breaker scale of the wind sea
    const dist = Math.hypot(r, hc);
    const p = ctx.createStereoPanner();
    // from high above everything is below the listener: less stereo spread
    p.pan.value = Math.max(-1, Math.min(1, (dx * rx + dz * rz) * r / (r + hc)));
    p.connect(bus);
    // air absorption, head shadow for crests behind the listener
    const air = (dx * fx + dz * fz < -0.3 ? 0.6 : 1) / (1 + dist / 70);
    const att = 0.04 + 0.12 * size * Math.random();
    const tau = (0.18 + 0.3 * size * (0.5 + Math.random())) * (0.6 + 0.6 * strength);
    voice(fizzBuf, (1800 + 1800 * Math.random()) * air, amp * 0.7, att, tau * 0.6, p);
    // the spilling water mass: lower, slower and louder for bigger breakers
    const whoosh = voice(pinkBuf, (300 + 500 * Math.random()) * Math.sqrt(air) / (0.7 + 0.3 * size),
      amp * 0.8 * Math.min(1, size), att * 1.6, tau * 1.3, p);
    whoosh.onended = () => { voices--; p.disconnect(); };
    return true;
  }
  const crestAmp = (strength, dist) => 1.4 * strength * (0.8 + 0.2 * Math.random()) / (1 + dist / 4);

  // --------------------------------------------------------------------------------------
  // Seagulls: each note jumps up and glides down ("kyow"), roughened by a fast amplitude
  // flutter; a call is a laughing series of notes, sometimes after a long wail
  // --------------------------------------------------------------------------------------
  function gullCall(delay) {
    if (gulls >= 4 || P.gullVol <= 0) return;
    gulls++;
    const rnd = Math.random;
    const t0 = ctx.currentTime + delay;
    const f0 = 700 + 400 * rnd();                   // this bird's voice
    const dist = 15 + 110 * rnd() ** 1.5;           // m, mostly close by
    const amp = 0.45 * P.gullVol / (1 + dist / 15);
    const osc = ctx.createOscillator();
    osc.setPeriodicWave(gullWave);
    const flutter = ctx.createOscillator();
    flutter.frequency.value = 35 + 30 * rnd();
    const depth = ctx.createGain();
    depth.gain.value = 0.35;
    const rough = ctx.createGain();
    rough.gain.value = 0.65;
    flutter.connect(depth).connect(rough.gain);
    const env = ctx.createGain();
    env.gain.value = 0;
    const formant = ctx.createBiquadFilter();
    formant.type = 'peaking'; formant.frequency.value = 2200 + 600 * rnd(); formant.Q.value = 1.2; formant.gain.value = 8;
    const air = ctx.createBiquadFilter();
    air.type = 'lowpass'; air.frequency.value = 12000 / (1 + dist / 60);
    const pan = ctx.createStereoPanner();
    osc.connect(rough).connect(env).connect(formant).connect(air).connect(pan).connect(bus);

    let t = t0;
    const note = (dur, a, up, down) => {
      osc.frequency.setValueAtTime(f0, t);
      osc.frequency.linearRampToValueAtTime(f0 * up, t + 0.04);
      osc.frequency.exponentialRampToValueAtTime(f0 * down, t + dur);
      env.gain.setValueAtTime(0, t);
      env.gain.linearRampToValueAtTime(a, t + 0.02);
      env.gain.setValueAtTime(a, t + dur * 0.6);
      env.gain.linearRampToValueAtTime(0, t + dur);
    };
    if (rnd() < 0.35) {
      const d = 0.45 + 0.35 * rnd();
      note(d, amp, 1.3, 0.75);
      t += d + 0.15 + 0.1 * rnd();
    }
    for (let k = 0, n = 2 + Math.floor(rnd() * 5); k < n; k++) {
      const d = 0.13 + 0.12 * rnd();
      note(d, amp * (1 - 0.08 * k), 1.25, 0.8);
      t += d + 0.07 + 0.08 * rnd();
    }
    // the bird flies on while it calls
    const p0 = 1.8 * rnd() - 0.9;
    pan.pan.setValueAtTime(p0, t0);
    pan.pan.linearRampToValueAtTime(Math.max(-1, Math.min(1, p0 + 0.6 * (rnd() - 0.5))), t);
    osc.start(t0); flutter.start(t0);
    osc.stop(t + 0.05); flutter.stop(t + 0.05);
    osc.onended = () => { gulls--; pan.disconnect(); };
    // another gull sometimes answers from elsewhere
    if (rnd() < 0.35) gullCall(t - ctx.currentTime + 0.3 + rnd());
  }

  // --------------------------------------------------------------------------------------
  // Probe results (water height under the camera, surface Jacobian around it)
  // --------------------------------------------------------------------------------------
  function onProbe(vals, foamThresh) {
    const now = performance.now() / 1000;
    const dtp = now - lastProbeT;
    const fresh = lastProbeT > 0 && dtp > 0.004 && dtp < 0.5;
    lastH = vals[0]; lastProbeT = now;
    if (!ctx || ctx.state !== 'running') return;
    // one crest folding across several neighbouring points is one sound: onsets are
    // energy-summed per 45 degree sector, and a sector rests while its crest sweeps on
    secE.fill(0); secX.fill(0); secZ.fill(0); secR.fill(0); secS.fill(0);
    let n = 0;
    for (let i = 0; i < PROBE_PTS; i++) {
      const J = vals[1 + i];
      const st = breaking[i];
      if (st === 0) {
        if (J < foamThresh) { breaking[i] = 1; jMin[i] = J; }
      } else if (st === 1 && J < jMin[i]) {
        jMin[i] = J;
      } else {
        // the fold has peaked (or recovered): sound it by its depth below the foam
        // threshold (median ~0.06, deep whitecaps 0.2-0.35 at default settings)
        if (st === 1) {
          breaking[i] = 2;
          const ox = probePts[i * 2], oz = probePts[i * 2 + 1], r = Math.hypot(ox, oz);
          const str = Math.min(1, (foamThresh - jMin[i]) / 0.25);
          // the shallowest folds stay silent: together they only add up to a steady hiss
          if (str >= 0.15) {
            const a = crestAmp(str, Math.hypot(r, hc)), e = a * a;
            const k = Math.floor((Math.atan2(oz, ox) / (2 * Math.PI) + 1) * SECTORS) % SECTORS;
            secE[k] += e; secX[k] += e * ox / r; secZ[k] += e * oz / r; secR[k] += e * r;
            secS[k] = Math.max(secS[k], str);
          }
        }
        if (J > foamThresh + 0.1) breaking[i] = 0;
      }
      if (breaking[i]) n++;
    }
    for (let k = 0; k < SECTORS; k++) {
      if (secE[k] === 0 || now < secNext[k]) continue;
      const l = Math.hypot(secX[k], secZ[k]);
      if (crest(Math.min(1.2, Math.sqrt(secE[k])), secX[k] / l, secZ[k] / l, secR[k] / secE[k], secS[k])) {
        secNext[k] = now + 0.5;
      }
    }
    if (fresh) cover += (n / PROBE_PTS - cover) * Math.min(1, dtp / 5);
  }

  // --------------------------------------------------------------------------------------
  // Per frame: listener geometry, continuous layers (updated at ~25 Hz)
  // --------------------------------------------------------------------------------------
  function set(param, v, tc = 0.12) {
    if (Number.isFinite(v)) param.setTargetAtTime(v, ctx.currentTime, tc);
  }

  function update(dt, cam, waterH) {
    hc = Math.max(0.3, cam.y - waterH);
    // rings grow with the height above mean sea level (not above the passing waves)
    const s = Math.max(1, cam.y / 2.5);
    if (Math.abs(s / ringScale - 1) > 0.15) layoutRings(s);
    if (!ctx || ctx.state !== 'running') return;
    pending += dt;
    if (pending < 0.04) return;
    const T = pending;
    pending = 0;

    const cy = Math.cos(cam.yaw), sy = Math.sin(cam.yaw);
    rx = cy; rz = sy; fx = sy; fz = -cy;
    if (havePos) {
      const k = Math.min(1, T * 3);
      vx += ((cam.x - px) / T - vx) * k; vy += ((cam.y - py) / T - vy) * k; vz += ((cam.z - pz) / T - vz) * k;
    }
    px = cam.x; py = cam.y; pz = cam.z; havePos = true;
    seaState();

    // wind at the listener's height (log profile, z0 = 0.2 mm), as the moving camera feels it
    const wa = P.windDir * Math.PI / 180;
    const uz = P.windSpeed * Math.log(Math.max(hc, 0.5) / 2e-4) / Math.log(10 / 2e-4);
    const ax = Math.cos(wa) * uz - vx, ay = -vy, az = Math.sin(wa) * uz - vz;
    const air = Math.min(45, Math.hypot(ax, ay, az));
    gustT -= T;
    if (gustT <= 0) { gustTarget = Math.random() * 2 - 1; gustT = 0.7 + Math.random() * 3.5; }
    gust += (gustTarget - gust) * Math.min(1, T / 1.4);
    const w = air * (1 + 0.22 * gust);
    const wl = Math.pow(w / 20, 1.5);
    const wpan = -(ax * rx + az * rz) / (Math.hypot(ax, az) + 1e-6) * 0.6;   // from upwind
    // gusts change the loudness only: a band swept over noise sounds like a phaser
    set(L.windA.g.gain, 1.1 * wl); set(L.windA.f.frequency, 160 + 38 * air, 1); set(L.windA.p.pan, wpan - 0.3);
    set(L.windB.g.gain, 0.8 * wl); set(L.windB.f.frequency, 240 + 52 * air, 1); set(L.windB.p.pan, wpan + 0.3);
    set(L.buffet.g.gain, 0.25 * wl * (0.7 + 0.3 * gust)); set(L.buffet.f.frequency, 70 + 4 * air, 1);

    // the sea itself: wave height and whitecap cover. Near the water it swishes as each
    // crest passes under the camera (height in units of its std, Hs / 4) and drops away
    // in the troughs; from high up the whole sea blends into an even hush.
    const sl = Math.min(1.6, sea.hs / 2.5);
    const cov = Math.min(0.25, cover);
    const x = lastH / (0.25 * sea.hs + 0.05);
    const kn = 1 / (1 + Math.max(0, hc - 0.9) / 5);
    const env = 0.5 + kn * (Math.max(0.08, Math.min(1.2, 0.5 + 0.35 * x)) - 0.5);
    const wash = 0.3 * env * Math.sqrt(sl) * (1 + 4 * cov) / (1 + hc / 40);
    const washF = 180 + 300 * sl / (1 + hc / 150);
    set(L.washL.g.gain, wash, 0.25); set(L.washL.f.frequency, washF, 1);
    set(L.washR.g.gain, wash, 0.25); set(L.washR.f.frequency, washF * 1.1, 1);

    // distant breakers beyond the probe rings, at the measured cover rate
    if (Math.random() < Math.min(2, 25 * cov) * T) {
      const a = Math.random() * 2 * Math.PI, r = (40 + 160 * Math.random()) * ringScale;
      crest(crestAmp(0.5, Math.hypot(r, hc)), Math.cos(a), Math.sin(a), r, 0.5);
    }
    // answers add ~50% more calls: P.gulls ends up as the total calls per minute
    if (Math.random() < P.gulls / 90 * T) gullCall(0);
    set(master.gain, enabled ? P.sound : 0, 0.08);
  }

  return {
    probePts, start, toggle, onProbe, update,
    get on() { return !!ctx && enabled; },
  };
}
