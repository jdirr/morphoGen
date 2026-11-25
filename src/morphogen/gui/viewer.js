/* morphoGen: the 3D view of the items of the workspace, drawn with WebGL 2 (no library,
 * no network: the page works offline).
 *
 * MorphoViewer.slot(key, options) is the HTML of a place for the view of an item; the
 * view opens there (as soon as it shows with options.auto, else by its button) and draws
 * the surface of /api/surface (morphogen/viz/surfaces.py): triangles in groups between
 * two regions (phases, or cells with their phases), the inside one behind them (their
 * normals point out of it), the outside one in front (-1: beyond the box). A fragment
 * shows the region that a ray entering through it meets: the inside from the front, the
 * outside from the back, the inside again behind the faces of the box (its shell seen
 * from within, through a section); hidden regions let the rays through. The volumes of
 * the surface (level set, regions, field of the voxels) colour the faces of the box and
 * the section plane, as the sections of the page.
 *
 * Mouse: drag to turn, shift-drag or right-drag to move, wheel to zoom; touch: one finger
 * turns, two zoom and move. Lighting: two-sided Lambert with some ambient light, from the
 * eye. The view is drawn on demand, in one call for all its triangles.
 *
 * The pure functions (matrices, section planes, colour maps, decoding) are exported for
 * Node, which tests them (tests/test_surfaces.py).
 */
(function () {
  "use strict";

  // -- vectors and matrices (column-major Float32Array(16), as WebGL takes them) ----------
  const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
  const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
  const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
  function normalize(a) {
    const n = Math.hypot(a[0], a[1], a[2]) || 1;
    return [a[0] / n, a[1] / n, a[2] / n];
  }
  function identity() {
    const m = new Float32Array(16);
    m[0] = m[5] = m[10] = m[15] = 1;
    return m;
  }
  function multiply(a, b) {  // the product a b
    const out = new Float32Array(16);
    for (let c = 0; c < 4; c++) {
      for (let r = 0; r < 4; r++) {
        let s = 0;
        for (let k = 0; k < 4; k++) s += a[k * 4 + r] * b[c * 4 + k];
        out[c * 4 + r] = s;
      }
    }
    return out;
  }
  function perspective(fovy, aspect, near, far) {  // as gluPerspective
    const f = 1 / Math.tan(fovy / 2), nf = 1 / (near - far);
    const m = new Float32Array(16);
    m[0] = f / aspect;
    m[5] = f;
    m[10] = (far + near) * nf;
    m[11] = -1;
    m[14] = 2 * far * near * nf;
    return m;
  }
  function lookAt(eye, target, up) {  // the view from eye to target, up upwards
    const z = normalize(sub(eye, target)), x = normalize(cross(up, z)), y = cross(z, x);
    const m = identity();
    m[0] = x[0]; m[4] = x[1]; m[8] = x[2];
    m[1] = y[0]; m[5] = y[1]; m[9] = y[2];
    m[2] = z[0]; m[6] = z[1]; m[10] = z[2];
    m[12] = -dot(x, eye); m[13] = -dot(y, eye); m[14] = -dot(z, eye);
    return m;
  }
  function transform(m, p) {  // m [p, 1]: homogeneous coordinates
    const out = [0, 0, 0, 0];
    for (let r = 0; r < 4; r++) out[r] = m[r] * p[0] + m[4 + r] * p[1] + m[8 + r] * p[2] + m[12 + r];
    return out;
  }
  function project(m, p, width, height) {  // pixels (from the top left) and depth; null behind
    const q = transform(m, p);
    if (q[3] <= 0) return null;
    return [(q[0] / q[3] + 1) * width / 2, (1 - q[1] / q[3]) * height / 2, q[2] / q[3]];
  }
  function orbitEye(target, distance, yaw, pitch) {  // z up; yaw from +x towards +y
    const c = Math.cos(pitch);
    return [target[0] + distance * c * Math.cos(yaw), target[1] + distance * c * Math.sin(yaw),
      target[2] + distance * Math.sin(pitch)];
  }
  function fit(lower, upper, fovy, aspect) {  // the target and distance that frame a box
    const target = [0, 1, 2].map((a) => (lower[a] + upper[a]) / 2);
    const radius = Math.hypot(upper[0] - lower[0], upper[1] - lower[1], upper[2] - lower[2]) / 2 || 1;
    // the narrower of the two angles of view: that across a view taller than wide
    const half = Math.min(fovy / 2, Math.atan((aspect || 1) * Math.tan(fovy / 2)));
    return {target, radius, distance: 1.08 * radius / Math.sin(half)};
  }

  // -- the section plane --------------------------------------------------------------------
  const AXES = {x: 0, y: 1, z: 2};
  function axisOf(axis) {
    if (typeof axis === "number") return axis >= 0 && axis <= 2 ? axis : -1;
    return axis in AXES ? AXES[axis] : -1;
  }
  // [a, b, c, d]: the points kept by a section normal to axis (x, y, z or 0-2) at fraction
  // of the box, a x + b y + c z + d >= 0: beyond the plane, or before it when flipped;
  // [0, 0, 0, 1] keeps everything (no section)
  function clipPlane(axis, fraction, lower, upper, flip) {
    const a = axisOf(axis);
    if (a < 0) return [0, 0, 0, 1];
    const at = lower[a] + Math.min(Math.max(fraction, 0), 1) * (upper[a] - lower[a]);
    const s = flip ? -1 : 1, plane = [0, 0, 0, 0];
    plane[a] = s;
    plane[3] = -s * at;
    return plane;
  }
  function sectionQuad(axis, fraction, lower, upper) {  // the corners of the cut of the box
    const a = axisOf(axis);
    if (a < 0) return null;
    const u = (a + 1) % 3, v = (a + 2) % 3;
    const at = lower[a] + Math.min(Math.max(fraction, 0), 1) * (upper[a] - lower[a]);
    return [[lower[u], lower[v]], [upper[u], lower[v]], [upper[u], upper[v]], [lower[u], upper[v]]].map(([pu, pv]) => {
      const p = [0, 0, 0];
      p[a] = at; p[u] = pu; p[v] = pv;
      return p;
    });
  }
  function boxFaces(lower, upper) {  // the six faces of a box: corners counter-clockwise from outside
    const out = [];
    for (let a = 0; a < 3; a++) {
      for (const side of [0, 1]) {
        const u = (a + 1) % 3, v = (a + 2) % 3, at = side ? upper[a] : lower[a];
        let corners = [[lower[u], lower[v]], [upper[u], lower[v]], [upper[u], upper[v]], [lower[u], upper[v]]];
        if (!side) corners = corners.reverse();
        const normal = [0, 0, 0];
        normal[a] = side ? 1 : -1;
        out.push({normal, corners: corners.map(([pu, pv]) => {
          const p = [0, 0, 0];
          p[a] = at; p[u] = pu; p[v] = pv;
          return p;
        })});
      }
    }
    return out;
  }

  // -- colour maps (those of the server, from matplotlib, at 33 stops) -----------------------
  const COLORMAPS = {
    viridis: "440154470d6048186a482374472d7b4537814240863e49893b528b375b8d33638d2f6b8e2c728e297a8e26828e23898e21918c1f988b1fa08822a78528ae8032b67a3fbc734ec36b5ec96270cf5784d44b98d83eaddc30c2df23d8e219ece51bfde725",
    plasma: "0d08872206903105973f049c4c02a15901a56600a77201a87e03a88a09a59511a1a01a9caa2395b32c8ebc3587c43e7fcc4778d35171da5a6ae06363e66c5ceb7655f0804ef58b47f89540fba139fdac33feb82cfdc527fcd225f8df25f4ed27f0f921",
    inferno: "0000040403120b0724150b37210c4a2f0a5b3d09654a0c6b57106e64156e71196e7d1e6d8a226a972766a32c61b0315bbc3754c73e4cd24644db503be45a31eb6628f1731df68013f98e09fb9d07fcac11fbbc21f9cb35f5db4cf2ea69f3f68afcffa4",
    magma: "0000040303120a0822130d341d114729115a36106b440f7651127c5d177f6a1c81762181832681902a819c2e7faa337db73779c43c75d0416fdc4869e75263ef5d5ef56b5cf9795dfc8961fd9869fea772feb67cfec488fed395fde2a3fcf0b2fcfdbf",
    cividis: "00224e00285b002e6a0533711a386f273e6e32436d3b496c434e6c4b546c535a6d5a5f6e61656f686a716f70737676767d7c788482798c8878938e789b9476a39a74aba072b4a76fbcae6cc4b468cdbb63d5c25edec958e7d150f0d846f9e03afee838",
    turbo: "30123b392a734040a24456c7466be34680f64294ff37a8fa28bceb1ccdd818ddc21fe9af32f2984ef97d6dfe628bff4ba4fc3cb9f635cdec34dfdf37eecf3af8be39fdac34fe962bfb7e21f46617eb500edf3f08d02f05be2102a91601920b017a0403",
    coolwarm: "3b4cc0445acc4e68d85875e16282ea6c8ff1779af782a6fb8db0fe98b9ffa3c2feaec9fcb9d0f9c3d5f4ccd9edd5dbe5dddcdce5d8d1ecd3c5f1ccb8f5c4acf7ba9ff7b093f6a586f4987af08b6eeb7d62e46e56dd5f4bd44e41ca3b37be242eb40426",
    RdBu_r: "0530610e41791752901f63a82a71b23480b93f8ec0529dc86bacd184bcd99bc9e0aed3e6c2ddecd4e6f1e0ecf3ecf2f5f7f6f6f9eee7fbe5d8fddcc9fbccb4f8bb9ef5aa89ee9677e48066db6b55d05548c53e3dba2832ab162a930e267c072267001f",
    jet: "0000800000a40000c80000ed0000ff0020ff0040ff0060ff0080ff00a0ff00c0ff00e0fb16ffe130ffc749ffad63ff947dff7a97ff60b1ff46caff2ce4ff13feed00ffd000ffb200ff9400ff7700ff5900ff3b00ff1e00e80000c400009f0000800000",
    gray: "000000080808101010181818202020282828303030383838404040484848505050585858606060686868707070787878808080888888909090989898a0a0a0a8a8a8b0b0b0b8b8b8c0c0c0c8c8c8d0d0d0d8d8d8e0e0e0e8e8e8f0f0f0f8f8f8ffffff",
  };
  const DIVERGING = ["coolwarm", "RdBu_r"];  // centred on zero for values of both signs
  const STOPS = {};
  function stopsOf(name) {
    const key = name in COLORMAPS ? name : "viridis";
    if (!STOPS[key]) {
      const hex = COLORMAPS[key], out = [];
      for (let i = 0; i < hex.length; i += 6) {
        out.push([0, 2, 4].map((k) => parseInt(hex.slice(i + k, i + k + 2), 16)));
      }
      STOPS[key] = out;
    }
    return STOPS[key];
  }
  function colormap(name, t) {  // [r, g, b] (0-255) of t in [0, 1], interpolated linearly
    const stops = stopsOf(name), n = stops.length - 1;
    const x = Math.min(Math.max(Number.isFinite(t) ? t : 0, 0), 1) * n;
    const i = Math.min(Math.floor(x), n - 1), f = x - i;
    return [0, 1, 2].map((c) => Math.round(stops[i][c] + f * (stops[i + 1][c] - stops[i][c])));
  }
  function colormapTable(name, size) {  // RGBA bytes of size colours, for a texture
    const n = size || 256, out = new Uint8Array(4 * n);
    for (let k = 0; k < n; k++) {
      const [r, g, b] = colormap(name, k / (n - 1));
      out.set([r, g, b, 255], 4 * k);
    }
    return out;
  }
  function gradient(name) {  // a CSS gradient of a colour map, from left to right
    const stops = [];
    for (let k = 0; k <= 16; k++) {
      const [r, g, b] = colormap(name, k / 16);
      stops.push(`rgb(${r},${g},${b}) ${(100 * k / 16).toFixed(2)}%`);
    }
    return `linear-gradient(to right, ${stops.join(", ")})`;
  }
  // the scale of a field: its range, from zero when it nearly starts there, centred on zero
  // for the diverging maps when the values change sign (as morphogen.viz.video), widened
  // when it collapses
  function defaultRange(lo, hi, cmap) {
    if (!Number.isFinite(lo) || !Number.isFinite(hi)) return [0, 1];
    if (lo > hi) [lo, hi] = [hi, lo];
    if (Math.abs(lo) < 1e-9 * Math.abs(hi)) lo = 0;
    if (DIVERGING.includes(cmap) && lo < 0 && hi > 0) {
      const m = Math.max(-lo, hi);
      lo = -m;
      hi = m;
    }
    if (hi <= lo) {
      const pad = Math.abs(lo) * 0.05 || 0.5;
      lo -= pad;
      hi += pad;
    }
    return [lo, hi];
  }

  // -- the data of /api/surface ---------------------------------------------------------------
  function decode(text, Type) {  // a typed array of base64 bytes (little-endian, as every browser)
    if (!text) return null;
    const bin = atob(text), n = bin.length, bytes = new Uint8Array(n);
    for (let i = 0; i < n; i++) bytes[i] = bin.charCodeAt(i);
    return new Type(bytes.buffer, 0, n / Type.BYTES_PER_ELEMENT);
  }
  // the arrays of the vertices: positions and normals (of voxel faces, from their nodes and
  // directions), regions (inside, outside), field values (front, back), displacement
  function surfaceArrays(data) {
    const n = data.vertices;
    let positions, normals;
    if (data.lattice) {
      const node = decode(data.lattice.node, Uint32Array), dir = decode(data.lattice.direction, Uint8Array);
      const [, ny, nz] = data.lattice.nodes;
      const h = [0, 1, 2].map((a) => (data.upper[a] - data.lower[a]) / (data.lattice.nodes[a] - 1));
      positions = new Float32Array(3 * n);
      normals = new Float32Array(3 * n);
      for (let v = 0; v < n; v++) {
        const id = node[v], k = id % nz, j = Math.floor(id / nz) % ny, i = Math.floor(id / (ny * nz));
        positions[3 * v] = data.lower[0] + i * h[0];
        positions[3 * v + 1] = data.lower[1] + j * h[1];
        positions[3 * v + 2] = data.lower[2] + k * h[2];
        normals[3 * v + (dir[v] >> 1)] = dir[v] & 1 ? -1 : 1;
      }
    } else {
      positions = decode(data.positions, Float32Array) || new Float32Array(0);
      normals = decode(data.normals, Float32Array) || new Float32Array(0);
    }
    const regions = new Int32Array(2 * n);
    for (const [inside, outside, first, count] of data.groups) {
      for (let v = first; v < first + count; v++) {
        regions[2 * v] = inside;
        regions[2 * v + 1] = outside;
      }
    }
    const field = data.field || {};
    return {
      positions, normals, regions,
      indices: decode(data.indices, Uint32Array) || new Uint32Array(0),
      values: field.values ? decode(field.values, Float32Array) : null,
      displacement: data.displacement ? decode(data.displacement, Float32Array) : null,
    };
  }
  function volumeArray(volume) {
    if (!volume) return null;
    const Type = {uint8: Uint8Array, uint16: Uint16Array, uint32: Uint32Array}[volume.dtype];
    return decode(volume.data, Type);
  }
  function hexColor(hex) {
    const h = String(hex || "#888888").replace("#", "");
    return [0, 2, 4].map((k) => parseInt(h.slice(k, k + 2), 16) || 0);
  }
  function fmt(v) {
    return Number.isFinite(v) ? String(+Number(v).toPrecision(4)) : "";
  }

  // -- WebGL ----------------------------------------------------------------------------------
  const HEAD = `#version 300 es
precision highp float;
precision highp int;
precision highp sampler3D;
precision highp usampler3D;
`;
  const SURFACE_VS = HEAD + `
in vec3 aPos;
in vec3 aNormal;
in ivec2 aRegion;
in vec2 aValue;
in vec3 aDisp;
uniform mat4 uProj;
uniform mat4 uView;
uniform float uMag;
uniform vec4 uClip;
out vec3 vNormal;
out vec3 vModelNormal;
out vec3 vPos;
out float vClip;
out vec2 vValue;
flat out ivec2 vRegion;
void main() {
  vec3 p = aPos + uMag * aDisp;
  vPos = aPos;
  vClip = dot(uClip.xyz, p) + uClip.w;
  vNormal = mat3(uView) * aNormal;
  vModelNormal = aNormal;
  vValue = aValue;
  vRegion = aRegion;
  gl_Position = uProj * uView * vec4(p, 1.0);
}`;
  const COMMON_FS = HEAD + `
uniform sampler2D uRegions;
uniform sampler2D uCmap;
uniform usampler3D uField;
uniform vec2 uRange;
uniform vec2 uFieldRange;
uniform vec3 uLower;
uniform vec3 uUpper;
uniform ivec3 uCells;
uniform vec3 uLight;
uniform float uAmbient;
uniform float uEps;
out vec4 color;
vec4 region(int id) {
  ivec2 size = textureSize(uRegions, 0);
  return texelFetch(uRegions, ivec2(id % size.x, id / size.x), 0);
}
vec3 cmap(float v) {
  float span = uRange.y - uRange.x;
  float t = span > 0.0 ? clamp((v - uRange.x) / span, 0.0, 1.0) : 0.5;
  return texture(uCmap, vec2(t, 0.5)).rgb;
}
ivec3 cellOf(vec3 p) {
  ivec3 c = ivec3(floor((p - uLower) / (uUpper - uLower) * vec3(uCells)));
  return clamp(c, ivec3(0), uCells - 1);
}
float voxel(vec3 p) {
  float q = float(texelFetch(uField, cellOf(p), 0).r) / 65535.0;
  return uFieldRange.x + q * (uFieldRange.y - uFieldRange.x);
}
vec4 lit(vec3 base, vec3 normal) {
  vec3 n = normalize(normal);
  if (!gl_FrontFacing) n = -n;
  return vec4(base * (uAmbient + (1.0 - uAmbient) * max(dot(n, uLight), 0.0)), 1.0);
}
`;
  const SURFACE_FS = COMMON_FS + `
uniform int uMode;  // 0 colours of the regions, 1 field at the vertices, 2 field of the voxels
in vec3 vNormal;
in vec3 vModelNormal;
in vec3 vPos;
in float vClip;
in vec2 vValue;
flat in ivec2 vRegion;
void main() {
  if (vClip < uEps) discard;
  bool near = gl_FrontFacing || vRegion.y < 0;  // the inside; behind the faces of the box too
  vec4 r = region(near ? vRegion.x : vRegion.y);
  if (r.a < 0.5) discard;
  vec3 base = r.rgb;
  if (uMode == 1) {
    base = cmap(near ? vValue.x : vValue.y);
  } else if (uMode == 2) {  // the voxel behind the face, or in front of it
    vec3 h = (uUpper - uLower) / vec3(uCells);
    base = cmap(voxel(vPos + (near ? -0.5 : 0.5) * h * vModelNormal));
  }
  color = lit(base, vNormal);
}`;
  const CAP_VS = HEAD + `
in vec3 aPos;
uniform mat4 uProj;
uniform mat4 uView;
uniform vec4 uClip;
out vec3 vPos;
out float vClip;
void main() {
  vPos = aPos;
  vClip = dot(uClip.xyz, aPos) + uClip.w;
  gl_Position = uProj * uView * vec4(aPos, 1.0);
}`;
  const CAP_FS = COMMON_FS + `
uniform int uMode;  // 0 colours of the regions, 2 field of the voxels
uniform int uVolume;  // 1 level set (two phases), 2 regions of the cells
uniform bool uClipped;  // the faces of the box, cut by the section (not the section)
uniform sampler3D uLevel;
uniform usampler3D uRegionVolume;
uniform vec3 uNormal;
in vec3 vPos;
in float vClip;
void main() {
  if (uClipped && vClip < -uEps) discard;
  int id;
  if (uVolume == 1) {
    id = texture(uLevel, (vPos - uLower) / (uUpper - uLower)).r < 127.5 / 255.0 ? 1 : 0;
  } else {
    id = int(texelFetch(uRegionVolume, cellOf(vPos), 0).r);
  }
  vec4 r = region(id);
  if (r.a < 0.5) discard;
  color = lit(uMode == 2 ? cmap(voxel(vPos)) : r.rgb, uNormal);
}`;
  const LINE_VS = HEAD + `
in vec3 aPos;
uniform mat4 uProj;
uniform mat4 uView;
void main() { gl_Position = uProj * uView * vec4(aPos, 1.0); }`;
  const LINE_FS = HEAD + `
uniform vec3 uColor;
out vec4 color;
void main() { color = vec4(uColor, 1.0); }`;

  function compile(gl, vs, fs) {
    const program = gl.createProgram();
    for (const [type, source] of [[gl.VERTEX_SHADER, vs], [gl.FRAGMENT_SHADER, fs]]) {
      const shader = gl.createShader(type);
      gl.shaderSource(shader, source);
      gl.compileShader(shader);
      if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
        throw new Error("shader: " + gl.getShaderInfoLog(shader));
      }
      gl.attachShader(program, shader);
    }
    gl.linkProgram(program);
    if (!gl.getProgramParameter(program, gl.LINK_STATUS)) throw new Error("program: " + gl.getProgramInfoLog(program));
    const uniforms = {};
    const n = gl.getProgramParameter(program, gl.ACTIVE_UNIFORMS);
    for (let i = 0; i < n; i++) {
      const name = gl.getActiveUniform(program, i).name;
      uniforms[name] = gl.getUniformLocation(program, name);
    }
    return {program, uniforms, attribute: (name) => gl.getAttribLocation(program, name)};
  }
  const UNITS = {uRegions: 0, uCmap: 1, uLevel: 2, uRegionVolume: 3, uField: 4};

  // -- the view ---------------------------------------------------------------------------------
  const FOV = 30 * Math.PI / 180;
  const VIEW = {yaw: -1.0, pitch: 0.45};
  const VIEWERS = new Set();
  let STYLED = false;
  const CSS = `
.v3d-slot { margin: 10px 0; }
.v3d { border: 1px solid var(--line); border-radius: 8px; background: var(--panel); padding: 8px;
       max-width: 980px; }
.v3d-stage { position: relative; height: min(520px, 70vh); border-radius: 6px; overflow: hidden;
             background: var(--surface); touch-action: none; }
.v3d-stage canvas { width: 100%; height: 100%; display: block; cursor: grab; outline: none; }
.v3d-stage canvas:active { cursor: grabbing; }
.v3d-msg { position: absolute; inset: 0; display: flex; align-items: center; justify-content: center;
           padding: 20px; text-align: center; color: var(--muted); font-size: 13px; pointer-events: none; }
.v3d-msg.error { color: var(--warn); }
.v3d-label { position: absolute; font: 600 11px system-ui, sans-serif; pointer-events: none;
             transform: translate(-50%, -50%); }
.v3d-bar { display: flex; flex-wrap: wrap; align-items: center; gap: 6px 14px; margin: 8px 2px 0;
           font-size: 12px; }
.v3d label { display: inline-flex; align-items: center; gap: 5px; margin: 0; font-weight: 400;
             font-size: 12px; }
.v3d input, .v3d select { width: auto; padding: 3px 6px; font-size: 12px; }
.v3d input[type=checkbox] { margin: 0; }
.v3d input[type=range] { width: 130px; padding: 0; }
.v3d input.num { width: 76px; }
.v3d button.plain { margin: 0; padding: 4px 9px; font-size: 12px; }
.v3d-swatch { display: inline-block; width: 12px; height: 12px; border-radius: 3px;
              border: 1px solid var(--line); }
.v3d-ramp { display: inline-block; width: 150px; height: 12px; border-radius: 3px;
            border: 1px solid var(--line); }
.v3d-info { color: var(--muted); font-size: 12px; }
`;
  function style() {
    if (STYLED || typeof document === "undefined") return;
    const el = document.createElement("style");
    el.id = "v3d-style";
    el.textContent = CSS;
    document.head.appendChild(el);
    STYLED = true;
  }
  function cssColor(el, name, fallback) {  // a colour variable of the page, as [r, g, b] in 0-1
    const value = getComputedStyle(el).getPropertyValue(name).trim();
    const m = /^#([0-9a-f]{6})$/i.exec(value);
    return (m ? hexColor(m[1]) : fallback).map((c) => c / 255);
  }
  function esc(s) {
    return String(s).replace(/[&<>"]/g, (c) => ({"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;"}[c]));
  }

  class Viewer {
    constructor(slot, options) {
      style();
      for (const v of [...VIEWERS]) if (!v.canvas.isConnected) v.dispose();
      VIEWERS.add(this);
      this.slot = slot;
      this.options = options;
      this.key = options.key;
      this.camera = null;
      this.data = null;
      this.mode = 0;
      slot.innerHTML = `<div class="v3d">
        <div class="v3d-stage"><canvas tabindex="0" aria-label="3D view"></canvas><div class="v3d-msg">Drawing the surface…</div></div>
        <div class="v3d-bar">
          <button class="plain" data-v="reset" title="the first view">Reset view</button>
          <label title="the edges and axes of the cell"><input type="checkbox" data-v="box" checked>box</label>
          <label title="a plane cutting the structure, to see inside">Section <select data-v="axis">
            <option value="none">none</option><option value="x">normal to x</option><option value="y">normal to y</option><option value="z">normal to z</option></select></label>
          <input type="range" data-v="cut" min="0.005" max="0.995" step="0.005" value="0.5" title="position of the section">
          <label title="keep the other side of the section"><input type="checkbox" data-v="flip">other side</label>
          <label>Colours <select data-v="colour"></select></label>
        </div>
        <div class="v3d-bar" data-v="phases"></div>
        <div class="v3d-bar" data-v="fieldbar" hidden>
          <label>Field <select data-v="field"></select></label>
          <label>Colormap <select data-v="cmap"></select></label>
          <input class="num" data-v="lo" title="smallest value of the scale">
          <span class="v3d-ramp" data-v="ramp"></span>
          <input class="num" data-v="hi" title="largest value of the scale">
          <span data-v="deform" hidden><label title="displacement drawn: magnification times the displacement">Magnification
            <input class="num" data-v="mag"></label> <input type="range" data-v="magr" min="0" max="3" step="0.02" value="1" title="0 to 3 times the automatic magnification"></span>
        </div>
        <div class="v3d-bar" data-v="stepbar" hidden>
          <button class="plain" data-v="play" title="go through the steps of the loading path">▶ Play</button>
          <label title="the step of the loading path drawn">Step <input type="range" data-v="step" min="0" max="0" step="1" value="0"></label>
          <span class="v3d-info" data-v="steplabel"></span>
          <label title="keep the colour scale while the step changes (else that of each step)"><input type="checkbox" data-v="hold" checked>fixed scale</label>
        </div>
        <div class="v3d-bar">
          <span data-v="resrow"><label title="cells along the longest edge of the grid of the surface">Resolution
            <input type="number" class="num" data-v="res" min="8" max="256" value="${options.resolution || 96}"></label>
            <button class="plain" data-v="redraw">Redraw</button></span>
          <span class="v3d-info" data-v="info"></span>
          <span class="v3d-info" data-v="hint" hidden>(the section is filled on the undeformed mesh: magnification 0)</span>
        </div>
        <div class="v3d-bar v3d-info">drag: turn · shift-drag or right-drag: move · wheel: zoom</div>
      </div>`;
      this.el = (name) => slot.querySelector(`[data-v="${name}"]`);
      this.canvas = slot.querySelector("canvas");
      this.message = slot.querySelector(".v3d-msg");
      this.stage = slot.querySelector(".v3d-stage");
      this.el("cmap").innerHTML = Object.keys(COLORMAPS).map((c) => `<option>${c}</option>`).join("");
      this.el("cmap").value = options.cmap && options.cmap in COLORMAPS ? options.cmap : "viridis";
      this.gl = this.canvas.getContext("webgl2", {antialias: true, alpha: false});
      if (!this.gl) {
        this.fail("The 3D view needs WebGL 2, which this browser does not offer (or has turned off). The sections and views above show the item.");
        return;
      }
      try {
        this.setup();
      } catch (e) {
        this.gl = null;
        this.fail("The 3D view cannot start here: " + e.message);
        return;
      }
      this.bind();
      this.load({field: options.field});
    }

    fail(text) {
      this.message.textContent = text;
      this.message.classList.add("error");
      this.message.hidden = false;
    }

    setup() {
      const gl = this.gl;
      this.programs = {
        surface: compile(gl, SURFACE_VS, SURFACE_FS),
        cap: compile(gl, CAP_VS, CAP_FS),
        line: compile(gl, LINE_VS, LINE_FS),
      };
      gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
      this.textures = {};
      const make = (target, unit) => {
        const t = gl.createTexture();
        gl.activeTexture(gl.TEXTURE0 + unit);
        gl.bindTexture(target, t);
        return t;
      };
      // a texture of each kind bound to its unit, so that every sampler finds its type
      this.textures.regions = make(gl.TEXTURE_2D, UNITS.uRegions);
      this.texture2D(this.textures.regions, UNITS.uRegions, 1, 1, new Uint8Array([128, 128, 128, 255]), false);
      this.textures.cmap = make(gl.TEXTURE_2D, UNITS.uCmap);
      this.textures.level = make(gl.TEXTURE_3D, UNITS.uLevel);
      this.texture3D(this.textures.level, UNITS.uLevel, [1, 1, 1], gl.R8, gl.RED, gl.UNSIGNED_BYTE, new Uint8Array([255]), gl.LINEAR, false);
      this.textures.regionVolume = make(gl.TEXTURE_3D, UNITS.uRegionVolume);
      this.texture3D(this.textures.regionVolume, UNITS.uRegionVolume, [1, 1, 1], gl.R8UI, gl.RED_INTEGER, gl.UNSIGNED_BYTE, new Uint8Array([0]), gl.NEAREST, false);
      this.textures.field = make(gl.TEXTURE_3D, UNITS.uField);
      this.texture3D(this.textures.field, UNITS.uField, [1, 1, 1], gl.R16UI, gl.RED_INTEGER, gl.UNSIGNED_SHORT, new Uint16Array([0]), gl.NEAREST, false);
      this.setColormap();
      this.surfaceVao = null;
      this.buffers = [];
      this.lines = gl.createBuffer();
      this.quads = gl.createBuffer();
      this.canvas.addEventListener("webglcontextlost", (e) => {
        e.preventDefault();
        this.lost = true;
        this.fail("The 3D view lost its graphics context (too many views open?): open it again.");
      });
    }

    texture2D(texture, unit, width, height, data, linear) {
      const gl = this.gl;
      gl.activeTexture(gl.TEXTURE0 + unit);
      gl.bindTexture(gl.TEXTURE_2D, texture);
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, width, height, 0, gl.RGBA, gl.UNSIGNED_BYTE, data);
      const filter = linear ? gl.LINEAR : gl.NEAREST;
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, filter);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, filter);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    }

    texture3D(texture, unit, shape, internal, format, type, data, filter, wrap) {
      const gl = this.gl;
      gl.activeTexture(gl.TEXTURE0 + unit);
      gl.bindTexture(gl.TEXTURE_3D, texture);
      gl.texImage3D(gl.TEXTURE_3D, 0, internal, shape[0], shape[1], shape[2], 0, format, type, data);
      gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_MIN_FILTER, filter);
      gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_MAG_FILTER, filter);
      const w = wrap ? gl.REPEAT : gl.CLAMP_TO_EDGE;
      for (const p of [gl.TEXTURE_WRAP_S, gl.TEXTURE_WRAP_T, gl.TEXTURE_WRAP_R]) gl.texParameteri(gl.TEXTURE_3D, p, w);
    }

    setColormap() {
      this.texture2D(this.textures.cmap, UNITS.uCmap, 256, 1, colormapTable(this.el("cmap").value, 256), true);
      this.el("ramp").style.background = gradient(this.el("cmap").value);
    }

    bind() {  // the controls and the mouse
      const on = (name, event, fn) => this.el(name).addEventListener(event, fn);
      on("reset", "click", () => { this.camera = this.home(); this.draw(); });
      for (const name of ["box", "axis", "cut", "flip"]) on(name, "input", () => this.draw());
      on("axis", "change", () => this.draw());
      on("colour", "change", () => { this.updateMode(); this.updateRegions(); this.draw(); });
      on("cmap", "change", () => { this.setColormap(); this.resetRange(); this.draw(); });
      for (const name of ["lo", "hi"]) on(name, "change", () => this.draw());
      on("field", "change", () => this.load({field: this.el("field").value, step: this.step()}));
      on("redraw", "click", () => this.load({field: this.el("field").value, step: this.step()}));
      on("res", "keydown", (e) => { if (e.key === "Enter") this.load({field: this.el("field").value}); });
      // the steps of a loading path: the slider, its label as it moves, the play button
      on("step", "input", () => this.stepLabel(Number(this.el("step").value)));
      on("step", "change", () => this.load({field: this.el("field").value, step: Number(this.el("step").value), keepScale: true}));
      on("play", "click", () => this.play());
      on("mag", "change", () => {
        const auto = this.autoMag || 1, v = Number(this.el("mag").value);
        if (Number.isFinite(v)) this.el("magr").value = String(Math.min(3, Math.max(0, v / auto)));
        this.draw();
      });
      on("magr", "input", () => {
        this.el("mag").value = fmt(Number(this.el("magr").value) * (this.autoMag || 0));
        this.draw();
      });
      const canvas = this.canvas, pointers = new Map();
      let drag = null;
      canvas.addEventListener("contextmenu", (e) => e.preventDefault());
      canvas.addEventListener("pointerdown", (e) => {
        try {
          canvas.setPointerCapture(e.pointerId);  // the drag goes on beyond the canvas
        } catch {
          // a pointer that cannot be captured: dragged within the canvas only
        }
        pointers.set(e.pointerId, [e.clientX, e.clientY]);
        drag = {pan: e.button === 2 || e.button === 1 || e.shiftKey, x: e.clientX, y: e.clientY};
      });
      canvas.addEventListener("pointermove", (e) => {
        if (!pointers.has(e.pointerId) || !this.camera) return;
        const before = [...pointers.values()];
        pointers.set(e.pointerId, [e.clientX, e.clientY]);
        if (pointers.size === 2) {  // pinch: zoom and move
          const after = [...pointers.values()];
          const span = (p) => Math.hypot(p[0][0] - p[1][0], p[0][1] - p[1][1]) || 1;
          const mid = (p) => [(p[0][0] + p[1][0]) / 2, (p[0][1] + p[1][1]) / 2];
          this.zoom(span(before) / span(after));
          const [x0, y0] = mid(before), [x1, y1] = mid(after);
          this.pan(x1 - x0, y1 - y0);
        } else if (drag) {
          const dx = e.clientX - drag.x, dy = e.clientY - drag.y;
          drag.x = e.clientX;
          drag.y = e.clientY;
          if (drag.pan || e.shiftKey) this.pan(dx, dy);
          else {
            this.camera.yaw -= dx * 0.008;
            this.camera.pitch = Math.min(1.55, Math.max(-1.55, this.camera.pitch + dy * 0.008));
          }
        }
        this.draw();
      });
      const up = (e) => {
        pointers.delete(e.pointerId);
        if (!pointers.size) drag = null;
        else {
          const [x, y] = [...pointers.values()][0];
          drag = {pan: false, x, y};
        }
      };
      canvas.addEventListener("pointerup", up);
      canvas.addEventListener("pointercancel", up);
      canvas.addEventListener("wheel", (e) => {
        if (!this.camera) return;
        e.preventDefault();
        const step = e.deltaMode === 1 ? 33 : (e.deltaMode === 2 ? 400 : 1);
        this.zoom(Math.exp(e.deltaY * step * 0.0015));
        this.draw();
      }, {passive: false});
      canvas.addEventListener("dblclick", () => { this.camera = this.home(); this.draw(); });
      if (typeof ResizeObserver !== "undefined") {
        this.resizer = new ResizeObserver(() => this.draw());
        this.resizer.observe(this.stage);
      }
      if (typeof matchMedia !== "undefined") {
        this.scheme = matchMedia("(prefers-color-scheme: dark)");
        this.onScheme = () => this.draw();
        this.scheme.addEventListener("change", this.onScheme);
      }
    }

    home() {
      const aspect = this.stage.clientWidth / Math.max(this.stage.clientHeight, 1) || 1;
      const box = fit(this.data.lower, this.data.upper, FOV, aspect);
      return {target: box.target, distance: box.distance, radius: box.radius, yaw: VIEW.yaw, pitch: VIEW.pitch};
    }
    zoom(factor) {
      const c = this.camera;
      c.distance = Math.min(40 * c.radius, Math.max(0.05 * c.radius, c.distance * factor));
    }
    pan(dx, dy) {
      const c = this.camera, view = this.viewMatrix();
      const scale = 2 * c.distance * Math.tan(FOV / 2) / Math.max(this.stage.clientHeight, 1);
      const right = [view[0], view[4], view[8]], up = [view[1], view[5], view[9]];
      for (let a = 0; a < 3; a++) c.target[a] += (-dx * right[a] + dy * up[a]) * scale;
    }
    viewMatrix() {
      const c = this.camera;
      return lookAt(orbitEye(c.target, c.distance, c.yaw, c.pitch), c.target, [0, 0, 1]);
    }

    async load(params) {
      const q = new URLSearchParams({key: this.key});
      const res = Number(this.el("res").value);
      if (res) q.set("resolution", String(res));
      if (params.field) q.set("field", params.field);
      if (Number.isInteger(params.step)) q.set("step", String(params.step));
      const query = q.toString();
      let data = this.cached(query);
      if (!data) {
        this.message.textContent = "Drawing the surface…";
        this.message.classList.remove("error");
        this.message.hidden = false;
        try {
          const reply = await fetch("/api/surface?" + query);
          data = await reply.json();
          if (!reply.ok || data.error) {
            // a field of the form that this simulation has not: its first field instead
            if (params.field && /^field/.test(data.error || "")) return this.load({step: params.step});
            throw new Error(data.error || reply.statusText);
          }
        } catch (e) {
          this.fail(e.message);
          return false;
        }
        this.cached(query, data);
      }
      if (!this.gl || this.lost) return false;
      // the scale of a field kept from a step to the next, when asked
      const same = this.data && this.data.field && data.field && this.data.field.name === data.field.name;
      this.keepScale = !!(params.keepScale && same && this.el("hold").checked);
      this.setData(data);
      return true;
    }

    cached(query, data) {  // the last surfaces drawn (the steps of a path played again)
      this.cache = this.cache || new Map();
      if (data === undefined) {
        const hit = this.cache.get(query);
        if (hit) { this.cache.delete(query); this.cache.set(query, hit); }
        return hit;
      }
      this.cache.set(query, data);
      while (this.cache.size > 8) this.cache.delete(this.cache.keys().next().value);
      return data;
    }

    step() {  // the step drawn, if the item has steps
      return this.data && Number.isInteger(this.data.step) ? this.data.step : undefined;
    }

    stepLabel(k) {
      const steps = (this.data && this.data.steps) || [];
      const s = steps[k];
      this.el("steplabel").textContent = s ? `${k + 1} / ${steps.length}${s.label ? " · " + s.label : ""}` : "";
    }

    async play() {  // the steps one after another, from the next (or the first after the last)
      if (this.playing) { this.playing = false; return; }
      const steps = (this.data && this.data.steps) || [];
      if (steps.length < 2) return;
      this.playing = true;
      this.el("play").textContent = "❚❚ Pause";
      let k = this.step() === steps.length - 1 ? 0 : (this.step() || 0) + 1;
      try {
        for (; k < steps.length && this.playing; k++) {
          const ok = await this.load({field: this.el("field").value, step: k, keepScale: k > 0});
          if (!ok) break;
          await new Promise((resolve) => setTimeout(resolve, 120));
        }
      } finally {
        this.playing = false;
        this.el("play").textContent = "▶ Play";
      }
    }

    setData(data) {
      const gl = this.gl, arrays = surfaceArrays(data);
      const first = !this.data;
      this.data = data;
      this.arrays = arrays;
      for (const b of this.buffers) gl.deleteBuffer(b);
      this.buffers = [];
      if (this.surfaceVao) gl.deleteVertexArray(this.surfaceVao);
      const program = this.programs.surface;
      this.surfaceVao = gl.createVertexArray();
      gl.bindVertexArray(this.surfaceVao);
      const attribute = (name, array, size, integer) => {
        const loc = program.attribute(name);
        if (loc < 0) return;
        if (!array) {
          gl.disableVertexAttribArray(loc);
          return;
        }
        const buffer = gl.createBuffer();
        this.buffers.push(buffer);
        gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
        gl.bufferData(gl.ARRAY_BUFFER, array, gl.STATIC_DRAW);
        gl.enableVertexAttribArray(loc);
        if (integer) gl.vertexAttribIPointer(loc, size, gl.INT, 0, 0);
        else gl.vertexAttribPointer(loc, size, gl.FLOAT, false, 0, 0);
      };
      attribute("aPos", arrays.positions, 3);
      attribute("aNormal", arrays.normals, 3);
      attribute("aRegion", arrays.regions, 2, true);
      attribute("aValue", arrays.values, 2);
      attribute("aDisp", arrays.displacement, 3);
      const index = gl.createBuffer();
      this.buffers.push(index);
      gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, index);
      gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, arrays.indices, gl.STATIC_DRAW);
      gl.bindVertexArray(null);
      // the volumes: level set or regions of the cells, field of the voxels
      const volume = data.volume, values = volumeArray(volume);
      this.volume = volume ? volume.kind : null;
      if (volume && volume.kind === "level_set") {
        this.texture3D(this.textures.level, UNITS.uLevel, volume.shape, gl.R8, gl.RED, gl.UNSIGNED_BYTE, values, gl.LINEAR, !!volume.periodic);
      } else if (volume) {
        const [internal, type] = {uint8: [gl.R8UI, gl.UNSIGNED_BYTE], uint16: [gl.R16UI, gl.UNSIGNED_SHORT], uint32: [gl.R32UI, gl.UNSIGNED_INT]}[volume.dtype];
        this.texture3D(this.textures.regionVolume, UNITS.uRegionVolume, volume.shape, internal, gl.RED_INTEGER, type, values, gl.NEAREST, false);
      }
      this.cells = volume ? volume.shape : [1, 1, 1];
      const fieldVolume = data.field && data.field.volume;
      if (fieldVolume) {
        this.texture3D(this.textures.field, UNITS.uField, fieldVolume.shape, gl.R16UI, gl.RED_INTEGER, gl.UNSIGNED_SHORT, volumeArray(fieldVolume), gl.NEAREST, false);
        this.cells = fieldVolume.shape;
      }
      this.fieldRange = fieldVolume ? fieldVolume.range : [0, 1];
      this.controls(first);
      if (first || !this.camera) this.camera = this.home();
      this.message.hidden = true;
      this.draw();
    }

    controls(first) {  // the controls of the data: colours, phases, field, magnification, info
      const data = this.data, regions = data.regions;
      const colour = this.el("colour"), keep = colour.value;
      const modes = [];
      if (data.field) modes.push(["field", data.field.name]);
      modes.push(["phase", "phases"]);
      if (regions.label) modes.push(["label", data.kind === "simulation" ? "grains" : "cells"]);
      colour.innerHTML = modes.map(([v, t]) => `<option value="${v}">${esc(t)}</option>`).join("");
      const labelled = regions.label && regions.label.some((k) => k >= 0);
      colour.value = modes.some(([v]) => v === keep) ? keep : (data.field ? "field" : (labelled && data.method === "cells" ? "label" : "phase"));
      colour.closest("label").hidden = modes.length < 2;
      // the phases, each shown or hidden
      const phases = [...new Set(regions.phase)].sort((a, b) => a - b);
      this.visible = this.visible || {};
      const colorOf = {};
      regions.phase.forEach((p, r) => { if (!(p in colorOf)) colorOf[p] = regions.color[r]; });
      this.el("phases").innerHTML = "Phases " + phases.map((p) => {
        const name = (data.phases || {})[String(p)];
        const on = this.visible[p] !== false;
        return `<label><input type="checkbox" data-phase="${p}" ${on ? "checked" : ""}><span class="v3d-swatch" style="background:${colorOf[p]}"></span>${esc(p)}${name ? ": " + esc(name) : ""}</label>`;
      }).join("");
      for (const box of this.el("phases").querySelectorAll("input[data-phase]")) {
        box.addEventListener("change", () => {
          this.visible[box.dataset.phase] = box.checked;
          this.updateRegions();
          this.draw();
        });
      }
      // the field: its name, scale and the magnification of the displacement
      const field = data.field;
      this.el("fieldbar").hidden = !field;
      if (field) {
        this.el("field").innerHTML = field.names.map((n) => `<option>${esc(n)}</option>`).join("");
        this.el("field").value = field.name;
        if (!this.keepScale) this.resetRange();
      }
      // the steps of a loading path, if it has several
      const steps = data.steps || [];
      this.el("stepbar").hidden = steps.length < 2;
      if (steps.length > 1) {
        this.el("step").max = String(steps.length - 1);
        this.el("step").value = String(data.step);
        this.stepLabel(data.step);
      }
      this.autoMag = data.magnification;
      this.el("deform").hidden = !data.displacement;
      if (data.displacement && (first || this.el("mag").value === "")) {
        this.el("mag").value = fmt(this.autoMag);
        this.el("magr").value = "1";
      }
      this.el("resrow").hidden = data.kind !== "morphology";
      if (data.kind === "morphology" && data.resolution) this.el("res").value = String(data.requested || data.resolution);
      const how = {"marching cubes": "marching cubes of the level set", cells: "exact cells", voxels: "faces of the voxels", mesh: "faces of the mesh"}[data.method] || data.method;
      const grid = data.resolution ? `, ${data.resolution} cells per edge` : "";
      const fewer = data.reduced ? ` (coarser to stay within ${data.budget.toLocaleString("en")} triangles)` : "";
      this.el("info").textContent = `${data.triangles.toLocaleString("en")} triangles: ${how}${grid}${fewer}, ${data.time_s} s`;
      this.updateMode();
      this.updateRegions();
    }

    resetRange() {
      const field = this.data && this.data.field;
      if (!field) return;
      const shown = field.shown || field.range;  // the percentiles 1 to 99: a singular peak stays bright
      const [lo, hi] = defaultRange(shown[0], shown[1], this.el("cmap").value);
      this.el("lo").value = fmt(lo);
      this.el("hi").value = fmt(hi);
    }

    updateMode() {  // the colours of the surface, and of the faces of the box and sections
      const data = this.data, field = this.el("colour").value === "field" && data.field;
      this.mode = field ? (field.values ? 1 : (field.volume ? 2 : 0)) : 0;
      this.capField = !!(field && field.volume);
    }

    updateRegions() {  // the colour and visibility of each region, in a texture
      // (one more region, transparent: that of the cells of no region, e.g. pores of a mesh)
      const regions = this.data.regions, n = regions.phase.length + 1;
      const width = Math.min(n, 1024), height = Math.ceil(n / width);
      const bytes = new Uint8Array(4 * width * height);
      const labels = this.el("colour").value === "label" && regions.label_color;
      regions.phase.forEach((p, r) => {
        const rgb = hexColor(labels ? regions.label_color[r] : regions.color[r]);
        bytes.set([...rgb, this.visible[p] === false ? 0 : 255], 4 * r);
      });
      this.texture2D(this.textures.regions, UNITS.uRegions, width, height, bytes, false);
    }

    draw() {
      if (this.pending || !this.gl || !this.data || this.lost) return;
      this.pending = true;
      requestAnimationFrame(() => {
        this.pending = false;
        if (this.canvas.isConnected) this.render();
      });
    }

    render() {
      const gl = this.gl, data = this.data, canvas = this.canvas;
      const ratio = Math.min(window.devicePixelRatio || 1, 2);
      const width = Math.max(1, Math.round(canvas.clientWidth * ratio));
      const height = Math.max(1, Math.round(canvas.clientHeight * ratio));
      if (canvas.width !== width || canvas.height !== height) {
        canvas.width = width;
        canvas.height = height;
      }
      gl.viewport(0, 0, width, height);
      const background = cssColor(this.stage, "--surface", [252, 252, 251]);
      gl.clearColor(background[0], background[1], background[2], 1);
      gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
      gl.enable(gl.DEPTH_TEST);
      gl.disable(gl.CULL_FACE);
      const c = this.camera;
      const near = Math.max(c.distance - 2.5 * c.radius, c.distance * 0.01);
      const proj = perspective(FOV, width / height, near, c.distance + 2.5 * c.radius);
      const view = this.viewMatrix();
      const light = normalize([0.3, 0.45, 0.85]);  // from the eye, a little above and right
      const clip = clipPlane(this.el("axis").value, Number(this.el("cut").value), data.lower, data.upper, this.el("flip").checked);
      const size = Math.max(...[0, 1, 2].map((a) => data.upper[a] - data.lower[a])) || 1;
      const [lo, hi] = [Number(this.el("lo").value), Number(this.el("hi").value)];
      const shared = (u) => {
        gl.uniformMatrix4fv(u.uProj, false, proj);
        gl.uniformMatrix4fv(u.uView, false, view);
        if (u.uClip) gl.uniform4fv(u.uClip, clip);
        if (u.uRegions) {
          for (const [name, unit] of Object.entries(UNITS)) if (u[name]) gl.uniform1i(u[name], unit);
          gl.uniform2f(u.uRange, Number.isFinite(lo) ? lo : 0, Number.isFinite(hi) ? hi : 1);
          gl.uniform2f(u.uFieldRange, this.fieldRange[0], this.fieldRange[1]);
          gl.uniform3fv(u.uLower, data.lower);
          gl.uniform3fv(u.uUpper, data.upper);
          gl.uniform3i(u.uCells, this.cells[0], this.cells[1], this.cells[2]);
          gl.uniform3fv(u.uLight, light);
          gl.uniform1f(u.uAmbient, 0.32);
          gl.uniform1f(u.uEps, 1e-5 * size);
        }
      };
      gl.enable(gl.POLYGON_OFFSET_FILL);  // the edges of the box drawn over the faces
      gl.polygonOffset(1, 1);
      if (this.arrays.indices.length) {
        const p = this.programs.surface;
        gl.useProgram(p.program);
        shared(p.uniforms);
        gl.uniform1i(p.uniforms.uMode, this.mode);
        const mag = this.arrays.displacement ? Number(this.el("mag").value) || 0 : 0;
        gl.uniform1f(p.uniforms.uMag, mag);
        gl.bindVertexArray(this.surfaceVao);
        const loc = p.attribute("aValue");
        if (!this.arrays.values && loc >= 0) gl.vertexAttrib2f(loc, 0, 0);
        const disp = p.attribute("aDisp");
        if (!this.arrays.displacement && disp >= 0) gl.vertexAttrib3f(disp, 0, 0, 0);
        gl.drawElements(gl.TRIANGLES, this.arrays.indices.length, gl.UNSIGNED_INT, 0);
        gl.bindVertexArray(null);
      }
      if (this.volume) this.caps(shared, clip, view);
      else this.el("hint").hidden = true;
      gl.disable(gl.POLYGON_OFFSET_FILL);
      this.outline(proj, view, width, height);
    }

    caps(shared, clip, view) {  // the faces of the box and the section, from the volumes
      const gl = this.gl, data = this.data, p = this.programs.cap;
      const quads = [];
      if (data.volume.box) for (const f of boxFaces(data.lower, data.upper)) quads.push({corners: f.corners, normal: f.normal, clipped: true});
      const axis = axisOf(this.el("axis").value);
      // the volume of a mesh is that of its undeformed elements: no filled section on it
      const deformed = !!this.arrays.displacement && (Number(this.el("mag").value) || 0) !== 0;
      this.el("hint").hidden = !(axis >= 0 && deformed);
      if (axis >= 0 && !deformed) {
        const normal = [0, 0, 0];
        normal[axis] = 1;  // that of its corners (counter-clockwise around +axis)
        quads.push({corners: sectionQuad(axis, Number(this.el("cut").value), data.lower, data.upper), normal, clipped: false});
      }
      if (!quads.length) return;
      gl.useProgram(p.program);
      shared(p.uniforms);
      gl.uniform1i(p.uniforms.uMode, this.capField ? 2 : 0);
      gl.uniform1i(p.uniforms.uVolume, this.volume === "level_set" ? 1 : 2);
      const loc = p.attribute("aPos");
      gl.bindBuffer(gl.ARRAY_BUFFER, this.quads);
      gl.enableVertexAttribArray(loc);
      gl.vertexAttribPointer(loc, 3, gl.FLOAT, false, 0, 0);
      for (const q of quads) {
        const [a, b, c2, d] = q.corners;
        gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([...a, ...b, ...c2, ...a, ...c2, ...d]), gl.STREAM_DRAW);
        const n = transform(view, [q.normal[0], q.normal[1], q.normal[2]]);
        const t = [n[0] - view[12], n[1] - view[13], n[2] - view[14]];  // the rotation alone
        gl.uniform3fv(p.uniforms.uNormal, normalize(t));
        gl.uniform1i(p.uniforms.uClipped, q.clipped ? 1 : 0);
        gl.drawArrays(gl.TRIANGLES, 0, 6);
      }
      gl.disableVertexAttribArray(loc);
    }

    outline(proj, view, width, height) {  // the edges of the box, its axes and their names
      const gl = this.gl, data = this.data, p = this.programs.line;
      for (const el of this.stage.querySelectorAll(".v3d-label")) el.remove();
      if (!this.el("box").checked) return;
      const [x0, y0, z0] = data.lower, [x1, y1, z1] = data.upper;
      const corners = [[x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0], [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1]];
      const edges = [[0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6], [6, 7], [7, 4], [0, 4], [1, 5], [2, 6], [3, 7]];
      const segments = edges.flatMap(([a, b]) => [...corners[a], ...corners[b]]);
      gl.useProgram(p.program);
      gl.uniformMatrix4fv(p.uniforms.uProj, false, proj);
      gl.uniformMatrix4fv(p.uniforms.uView, false, view);
      const loc = p.attribute("aPos");
      gl.bindBuffer(gl.ARRAY_BUFFER, this.lines);
      gl.enableVertexAttribArray(loc);
      gl.vertexAttribPointer(loc, 3, gl.FLOAT, false, 0, 0);
      gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(segments), gl.STREAM_DRAW);
      gl.uniform3fv(p.uniforms.uColor, cssColor(this.stage, "--c-muted", [137, 135, 129]));
      gl.drawArrays(gl.LINES, 0, 24);
      // the axes from the lower corner, coloured as the series of the charts
      const length = 0.22 * Math.max(x1 - x0, y1 - y0, z1 - z0);
      const colours = [["--series-2", [235, 104, 52]], ["--series-3", [27, 175, 122]], ["--series-1", [42, 120, 214]]];
      const both = multiply(proj, view);
      gl.disable(gl.DEPTH_TEST);
      for (let a = 0; a < 3; a++) {
        const end = [x0, y0, z0];
        end[a] += length;
        gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([x0, y0, z0, ...end]), gl.STREAM_DRAW);
        const rgb = cssColor(this.stage, colours[a][0], colours[a][1]);
        gl.uniform3fv(p.uniforms.uColor, rgb);
        gl.drawArrays(gl.LINES, 0, 2);
        const at = project(both, end, width, height);
        if (at) {
          const label = document.createElement("span");
          label.className = "v3d-label";
          label.textContent = "xyz"[a];
          label.style.color = `rgb(${rgb.map((v) => Math.round(255 * v)).join(",")})`;
          const ratio = width / Math.max(this.canvas.clientWidth, 1);
          label.style.left = `${at[0] / ratio + 8}px`;
          label.style.top = `${at[1] / ratio - 8}px`;
          this.stage.appendChild(label);
        }
      }
      gl.enable(gl.DEPTH_TEST);
      gl.disableVertexAttribArray(loc);
    }

    dispose() {
      VIEWERS.delete(this);
      if (this.resizer) this.resizer.disconnect();
      if (this.scheme) this.scheme.removeEventListener("change", this.onScheme);
      const gl = this.gl;
      this.gl = null;
      if (gl) {
        const ext = gl.getExtension("WEBGL_lose_context");
        if (ext) ext.loseContext();
      }
    }
  }

  // -- places of the view in the page ---------------------------------------------------------
  function slot(key, options) {  // the HTML of a place for the view of an item of the workspace
    const o = options || {};
    const data = [`data-key="${esc(key)}"`];
    for (const name of ["field", "cmap", "resolution"]) if (o[name]) data.push(`data-${name}="${esc(o[name])}"`);
    if (o.auto) data.push('data-auto="1"');
    const title = o.title === undefined ? "3D view" : o.title;
    const head = title ? `<h3>${esc(title)}</h3>` : "";
    const button = o.auto ? "" : `<button class="plain v3d-open">${esc(o.label || "View in 3D")}</button>`;
    return `<div class="v3d-slot" ${data.join(" ")}>${o.auto ? head : ""}${button}<div class="v3d-place"></div></div>`;
  }
  function open(element) {  // the view of the item of a place (an element made by slot)
    if (element.dataset.ready) return element.viewer;
    element.dataset.ready = "1";
    const button = element.querySelector(".v3d-open");
    if (button) button.remove();
    const d = element.dataset;
    element.viewer = new Viewer(element.querySelector(".v3d-place"), {
      key: d.key, field: d.field, cmap: d.cmap, resolution: Number(d.resolution) || 0,
    });
    return element.viewer;
  }
  function install() {  // buttons of the places, and the places that open as they show
    if (typeof document === "undefined") return;
    document.addEventListener("click", (e) => {
      const button = e.target.closest && e.target.closest(".v3d-open");
      if (button) open(button.closest(".v3d-slot"));
    });
    const seen = typeof IntersectionObserver === "undefined" ? null : new IntersectionObserver((entries) => {
      for (const entry of entries) {
        if (entry.isIntersecting) {
          seen.unobserve(entry.target);
          open(entry.target);
        }
      }
    });
    const scan = () => {
      for (const el of document.querySelectorAll(".v3d-slot[data-auto]:not([data-ready]):not([data-watched])")) {
        el.dataset.watched = "1";
        if (seen) seen.observe(el);
        else open(el);
      }
    };
    const start = () => {
      scan();
      new MutationObserver(scan).observe(document.body, {childList: true, subtree: true});
    };
    if (document.body) start();
    else document.addEventListener("DOMContentLoaded", start);
  }

  const api = {
    // pure functions, tested by Node
    identity, multiply, perspective, lookAt, transform, project, orbitEye, fit,
    clipPlane, sectionQuad, boxFaces, colormap, colormapTable, defaultRange, gradient,
    decode, surfaceArrays, COLORMAPS, DIVERGING,
    // the page
    slot, open, Viewer,
  };
  if (typeof module === "object" && module.exports) module.exports = api;
  if (typeof window !== "undefined") {
    window.MorphoViewer = api;
    install();
  }
})();
