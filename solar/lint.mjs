// Static checks on the WGSL before a GPU ever sees it.
//   node lint.mjs
// Catches the two mistakes that are easy to make and annoying to debug from a driver
// message: reserved words used as identifiers, and functions called before they are
// declared (WGSL has no forward declarations).
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const root = path.dirname(fileURLToPath(import.meta.url));

// WGSL reserved words (spec "Reserved Words"), plus the keywords that cannot be identifiers.
const RESERVED = new Set(`NULL Self abstract active alignas alignof as asm asm_fragment async
attribute auto await become binding_array cast catch class co_await co_return co_yield
coherent column_major common compile compile_fragment concept const_cast consteval constexpr
constinit crate debugger decltype delete demote demote_to_helper do dynamic_cast enum explicit
export extends extern external fallthrough filter final finally friend from fxgroup get goto
groupshared highp impl implements import inline instanceof interface layout lowp macro
macro_rules match mediump meta mod module move mut mutable namespace new nil noexcept noinline
nointerpolation non_coherent noncoherent noperspective null nullptr of operator package
packoffset partition pass patch pixelfragment precise precision premerge priv protected pub
public readonly ref regardless register reinterpret_cast require resource restrict self set
shared sizeof smooth snorm static static_assert static_cast std subroutine super target
template this thread_local throw trait try type typedef typeid typename typeof union unless
unorm unsafe unsized use using varying virtual volatile wgsl where while writeonly yield`
  .split(/\s+/).filter(Boolean));

// Built-ins and type names that are legitimately called but never declared in our source.
const BUILTIN = new Set(`abs acos acosh all any asin asinh atan atan2 atanh atomicAdd atomicAnd
atomicCompareExchangeWeak atomicExchange atomicLoad atomicMax atomicMin atomicOr atomicStore
atomicSub atomicXor bitcast ceil clamp cos cosh countLeadingZeros countOneBits
countTrailingZeros cross degrees determinant distance dot dot4I8Packed dot4U8Packed dpdx
dpdxCoarse dpdxFine dpdy dpdyCoarse dpdyFine exp exp2 extractBits faceForward firstLeadingBit
firstTrailingBit floor fma fract frexp fwidth fwidthCoarse fwidthFine insertBits inverseSqrt
ldexp length log log2 max min mix modf normalize pack2x16float pack2x16snorm pack2x16unorm
pack4x8snorm pack4x8unorm pow quantizeToF16 radians reflect refract reverseBits round
saturate select sign sin sinh smoothstep sqrt step storageBarrier tan tanh textureBarrier
textureDimensions textureGather textureGatherCompare textureLoad textureNumLayers
textureNumLevels textureNumSamples textureSample textureSampleBias textureSampleCompare
textureSampleCompareLevel textureSampleGrad textureSampleLevel textureSampleBaseClampToEdge
textureStore transpose trunc unpack2x16float unpack2x16snorm unpack2x16unorm unpack4x8snorm
unpack4x8unorm workgroupBarrier workgroupUniformLoad arrayLength
f32 i32 u32 bool f16 vec2 vec3 vec4 vec2f vec3f vec4f vec2i vec3i vec4i vec2u vec3u vec4u
vec2h vec3h vec4h mat2x2 mat2x3 mat2x4 mat3x2 mat3x3 mat3x4 mat4x2 mat4x3 mat4x4
mat2x2f mat2x3f mat2x4f mat3x2f mat3x3f mat3x4f mat4x2f mat4x3f mat4x4f
array atomic ptr sampler sampler_comparison texture_1d texture_2d texture_2d_array texture_3d
texture_cube texture_cube_array texture_multisampled_2d texture_storage_1d
texture_storage_2d texture_storage_2d_array texture_storage_3d texture_depth_2d
texture_depth_2d_array texture_depth_cube texture_depth_cube_array
texture_depth_multisampled_2d if else for return switch case default break continue
discard loop while fn let var const struct alias override enable requires diagnostic
true false workgroup_size compute vertex fragment builtin location group binding
interpolate invariant size align must_use id`.split(/\s+/).filter(Boolean));

// Pull every WGSL template literal out of a module, with the line it started on.
function extractWGSL(src, file) {
  const out = [];
  const re = /\/\* wgsl \*\/\s*`/g;
  let m;
  while ((m = re.exec(src))) {
    const start = m.index + m[0].length;
    let i = start;
    let depth = 0;
    while (i < src.length) {
      const c = src[i];
      if (c === '\\') { i += 2; continue; }
      if (c === '$' && src[i + 1] === '{') { depth++; i += 2; continue; }
      if (c === '}' && depth > 0) { depth--; i++; continue; }
      if (c === '`' && depth === 0) break;
      i++;
    }
    const body = src.slice(start, i);
    out.push({ file, line: src.slice(0, start).split('\n').length, body });
    re.lastIndex = i;
  }
  return out;
}

let errors = 0;
// A backtick anywhere in a comment inside a wgsl template literal silently terminates the
// literal and turns the rest of the shader into JavaScript. Check the raw text first.
for (const f of fs.readdirSync(path.join(root, 'js/wgsl'))) {
  if (!/\.m?js$/.test(f)) continue;
  const lines = fs.readFileSync(path.join(root, 'js/wgsl', f), 'utf8').split(/\r?\n/);
  lines.forEach((ln, k) => {
    const i = ln.indexOf('//');
    if (i >= 0 && ln.slice(i).includes('`')) {
      console.log(`BACKTICK  ${f}:${k + 1}  backtick in a comment ends the template literal`);
      console.log(`          ${ln.trim()}`);
      errors++;
    }
  });
}

const files = fs.readdirSync(path.join(root, 'js/wgsl')).filter((f) => f.endsWith('.js'));
const allChunks = [];
for (const f of files) {
  const src = fs.readFileSync(path.join(root, 'js/wgsl', f), 'utf8');
  const chunks = extractWGSL(src, f);
  if (!chunks.length) console.log(`  (${f}: no wgsl literals found)`);
  allChunks.push(...chunks);
}

// ---- reserved words used as identifiers ----
for (const ch of allChunks) {
  // strip comments and ${...} interpolations before tokenising
  const clean = ch.body
    .replace(/\$\{[^}]*\}/g, ' 0 ')
    .replace(/\/\*[\s\S]*?\*\//g, ' ')
    .replace(/\/\/[^\n]*/g, ' ');
  const lines = clean.split('\n');
  lines.forEach((ln, k) => {
    // declarations and member accesses are where a reserved word would bite
    for (const m of ln.matchAll(/\b([A-Za-z_][A-Za-z0-9_]*)\b\s*:/g)) {
      if (RESERVED.has(m[1])) {
        console.log(`RESERVED  ${ch.file}:${ch.line + k}  "${m[1]}" used as a declared name`);
        console.log(`          ${ln.trim()}`);
        errors++;
      }
    }
    for (const m of ln.matchAll(/\.([A-Za-z_][A-Za-z0-9_]*)\b/g)) {
      if (RESERVED.has(m[1])) {
        console.log(`RESERVED  ${ch.file}:${ch.line + k}  ".${m[1]}" member name is reserved`);
        console.log(`          ${ln.trim()}`);
        errors++;
      }
    }
    for (const m of ln.matchAll(/\b(?:fn|let|var|const)\s+([A-Za-z_][A-Za-z0-9_]*)/g)) {
      if (RESERVED.has(m[1])) {
        console.log(`RESERVED  ${ch.file}:${ch.line + k}  "${m[1]}" cannot be an identifier`);
        errors++;
      }
    }
  });
}

// ---- declaration order, per assembled shader ----
// Rebuild the real shader strings so the order check sees what the GPU sees.
const mod = await import('./js/wgsl/all.mjs');
for (const [name, code] of Object.entries(mod.SHADERS)) {
  const clean = code.replace(/\/\*[\s\S]*?\*\//g, ' ').replace(/\/\/[^\n]*/g, ' ');
  const declared = new Map();   // fn name -> index of its declaration
  for (const m of clean.matchAll(/\bfn\s+([A-Za-z_][A-Za-z0-9_]*)/g)) {
    if (declared.has(m[1])) {
      console.log(`DUPLICATE ${name}: fn ${m[1]} declared twice`);
      errors++;
    } else declared.set(m[1], m.index);
  }
  const structs = new Map();
  for (const m of clean.matchAll(/\bstruct\s+([A-Za-z_][A-Za-z0-9_]*)/g)) {
    if (structs.has(m[1])) {
      console.log(`DUPLICATE ${name}: struct ${m[1]} declared twice`);
      errors++;
    } else structs.set(m[1], m.index);
  }
  // every call site must come after the declaration
  for (const m of clean.matchAll(/\b([A-Za-z_][A-Za-z0-9_]*)\s*\(/g)) {
    const id = m[1];
    if (BUILTIN.has(id) || structs.has(id)) continue;
    if (!declared.has(id)) {
      if (/^(if|for|while|switch|return|select|case)$/.test(id)) continue;
      console.log(`UNKNOWN   ${name}: call to "${id}" with no declaration`);
      errors++;
      continue;
    }
    if (m.index < declared.get(id) && !/\bfn\s+$/.test(clean.slice(Math.max(0, m.index - 4), m.index))) {
      const ln = clean.slice(0, m.index).split('\n').length;
      console.log(`ORDER     ${name}: "${id}" called at line ${ln} but declared later`);
      errors++;
    }
  }
  // struct member types referenced before declaration
  for (const [sname, si] of structs) {
    for (const m of clean.matchAll(new RegExp(`:\\s*${sname}\\b`, 'g'))) {
      if (m.index < si) {
        console.log(`ORDER     ${name}: struct ${sname} used before declaration`);
        errors++;
      }
    }
  }
}

console.log(errors ? `\n${errors} problem(s)` : '\nwgsl lint clean');
process.exit(errors ? 1 : 0);
