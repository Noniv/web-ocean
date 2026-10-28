// Single place where the shader sources are assembled, so main.js and lint.mjs always see
// the same strings in the same order. WGSL has no forward declarations, so the order of
// these concatenations is load-bearing.
import { FRAME_WGSL, GEOM_WGSL } from './common.js';
import { SURF_GEN, SURF_LEAN, MIP_RGBA16 } from './surf.js';
import { RING_GEN, RING_MIP } from './rings.js';
import { ATMO_CORE, TRANS_LUT, MS_LUT } from './atmo.js';
import { BODIES, LUT_PRELUDE, MS_PRELUDE } from './bodies.js';
import { BACKGROUND, STARS, SUN } from './sky.js';
import { BLOOM_DOWN, BLOOM_UP, COMPOSITE } from './post.js';

export const SHADERS = {
  'surf gen': SURF_GEN,
  'surf lean': SURF_LEAN,
  'mip rgba16': MIP_RGBA16,
  'ring gen': RING_GEN,
  'ring mip': RING_MIP,
  'trans lut': FRAME_WGSL + GEOM_WGSL + LUT_PRELUDE + ATMO_CORE + TRANS_LUT,
  'ms lut': FRAME_WGSL + GEOM_WGSL + MS_PRELUDE + ATMO_CORE + MS_LUT,
  'bodies': BODIES,
  'background': BACKGROUND,
  'stars': STARS,
  'sun': SUN,
  'bloom down': BLOOM_DOWN,
  'bloom up': BLOOM_UP,
  'composite': COMPOSITE,
};
