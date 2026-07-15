/* Interactive finger cross-section: GPU field render + draggable tissue ellipses.
 * WebGPU when available, WebGL2 fallback. Vanilla Streamlit component protocol. */
"use strict";

import { OUTSIDE, SKIN, FAT, MUSCLE, BONE, LIGAMENT, ARTERY, rot, ellipseList, classify } from "./geometry.js";

// --------------------------------------------------------------------------- //
// Streamlit component protocol (bare, no npm dependency)
// --------------------------------------------------------------------------- //
function post(type, data) {
  window.parent.postMessage(Object.assign({ isStreamlitMessage: true, type }, data), "*");
}
const Streamlit = {
  ready: () => post("streamlit:componentReady", { apiVersion: 1 }),
  height: (h) => post("streamlit:setFrameHeight", { height: h }),
  value: (v) => post("streamlit:setComponentValue", { value: v, dataType: "json" }),
};

// --------------------------------------------------------------------------- //
// Shared shader source (GLSL). Classification packed into uniforms.
// --------------------------------------------------------------------------- //
const MAX_ELL = 5;
const FRAG_GLSL = `#version 300 es
precision highp float;
out vec4 outColor;
uniform vec2 uRes;
uniform float uScale;      // device px per mm
uniform float uFingerRot;  // radians
uniform vec2 uFingerHalf;
uniform float uSkinT, uFatT;
uniform int uColorMode;    // 0 tissue, 1 conductivity
uniform float uCondMin, uCondMax;
uniform int uEllCount;
uniform vec2 uEllCenter[${MAX_ELL}];
uniform vec2 uEllRadius[${MAX_ELL}];
uniform float uEllRot[${MAX_ELL}];
uniform int uEllKind[${MAX_ELL}];
uniform float uEllCond[${MAX_ELL}];
uniform float uSkinC, uFatC, uMuscleC;
uniform vec3 uColor[7];

vec2 rot(vec2 p, float a){ float c=cos(a), s=sin(a); return vec2(p.x*c+p.y*s, -p.x*s+p.y*c); }

vec3 cividis(float t){
  t = clamp(t, 0.0, 1.0);
  vec3 c0 = vec3(0.0,0.135,0.304);
  vec3 c1 = vec3(0.404,0.443,0.475);
  vec3 c2 = vec3(1.0,0.910,0.216);
  return t<0.5 ? mix(c0,c1,t*2.0) : mix(c1,c2,(t-0.5)*2.0);
}

void main(){
  vec2 world = (gl_FragCoord.xy - 0.5*uRes)/uScale;
  vec2 L = rot(world, uFingerRot);
  int label = 0; float sigma = -1.0;
  if((L.x/uFingerHalf.x)*(L.x/uFingerHalf.x)+(L.y/uFingerHalf.y)*(L.y/uFingerHalf.y) <= 1.0){ label=1; sigma=uSkinC; }
  vec2 sR = uFingerHalf - vec2(uSkinT);
  if(sR.x>0.0 && sR.y>0.0 && (L.x/sR.x)*(L.x/sR.x)+(L.y/sR.y)*(L.y/sR.y) <= 1.0){ label=2; sigma=uFatC; }
  vec2 fR = sR - vec2(uFatT);
  if(fR.x>0.0 && fR.y>0.0 && (L.x/fR.x)*(L.x/fR.x)+(L.y/fR.y)*(L.y/fR.y) <= 1.0){ label=3; sigma=uMuscleC; }
  for(int i=0;i<${MAX_ELL};i++){
    if(i>=uEllCount) break;
    vec2 c = rot(L - uEllCenter[i], uEllRot[i]);
    float rr = (c.x/uEllRadius[i].x)*(c.x/uEllRadius[i].x)+(c.y/uEllRadius[i].y)*(c.y/uEllRadius[i].y);
    if(rr<=1.0){ label=uEllKind[i]; sigma=uEllCond[i]; }
  }
  if(label==0){ outColor = vec4(0.0); return; }
  vec3 rgb;
  if(uColorMode==0){ rgb = uColor[label]; }
  else { float t=(sigma-uCondMin)/max(uCondMax-uCondMin,1e-6); rgb = cividis(t); }
  outColor = vec4(rgb, 1.0);
}`;

const VERT_GLSL = `#version 300 es
in vec2 aPos; void main(){ gl_Position = vec4(aPos,0.0,1.0); }`;

// --------------------------------------------------------------------------- //
// WebGL2 renderer
// --------------------------------------------------------------------------- //
class GLRenderer {
  constructor(canvas) {
    const gl = canvas.getContext("webgl2", { premultipliedAlpha: false, alpha: true });
    if (!gl) throw new Error("no webgl2");
    this.gl = gl;
    const prog = this._program(VERT_GLSL, FRAG_GLSL);
    this.prog = prog;
    const buf = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, buf);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 3, -1, -1, 3]), gl.STATIC_DRAW);
    const loc = gl.getAttribLocation(prog, "aPos");
    gl.enableVertexAttribArray(loc);
    gl.vertexAttribPointer(loc, 2, gl.FLOAT, false, 0, 0);
    this.u = {};
    const names = ["uRes", "uScale", "uFingerRot", "uFingerHalf", "uSkinT", "uFatT",
      "uColorMode", "uCondMin", "uCondMax", "uEllCount", "uSkinC", "uFatC", "uMuscleC"];
    for (const n of names) this.u[n] = gl.getUniformLocation(prog, n);
    for (const n of ["uEllCenter", "uEllRadius", "uEllRot", "uEllKind", "uEllCond", "uColor"])
      this.u[n] = gl.getUniformLocation(prog, n + "[0]");
  }
  _program(vs, fs) {
    const gl = this.gl;
    const compile = (type, src) => {
      const sh = gl.createShader(type); gl.shaderSource(sh, src); gl.compileShader(sh);
      if (!gl.getShaderParameter(sh, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(sh));
      return sh;
    };
    const p = gl.createProgram();
    gl.attachShader(p, compile(gl.VERTEX_SHADER, vs));
    gl.attachShader(p, compile(gl.FRAGMENT_SHADER, fs));
    gl.linkProgram(p);
    if (!gl.getProgramParameter(p, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(p));
    return p;
  }
  render(u, ells, colors) {
    const gl = this.gl;
    gl.viewport(0, 0, gl.canvas.width, gl.canvas.height);
    gl.clearColor(0, 0, 0, 0);
    gl.clear(gl.COLOR_BUFFER_BIT);
    gl.useProgram(this.prog);
    gl.uniform2f(this.u.uRes, gl.canvas.width, gl.canvas.height);
    gl.uniform1f(this.u.uScale, u.scale);
    gl.uniform1f(this.u.uFingerRot, u.fingerRot);
    gl.uniform2f(this.u.uFingerHalf, u.halfW, u.halfH);
    gl.uniform1f(this.u.uSkinT, u.skinT);
    gl.uniform1f(this.u.uFatT, u.fatT);
    gl.uniform1i(this.u.uColorMode, u.colorMode);
    gl.uniform1f(this.u.uCondMin, u.condMin);
    gl.uniform1f(this.u.uCondMax, u.condMax);
    gl.uniform1f(this.u.uSkinC, u.skinC);
    gl.uniform1f(this.u.uFatC, u.fatC);
    gl.uniform1f(this.u.uMuscleC, u.muscleC);
    gl.uniform1i(this.u.uEllCount, ells.length);
    const cen = new Float32Array(MAX_ELL * 2), rad = new Float32Array(MAX_ELL * 2);
    const rt = new Float32Array(MAX_ELL), kd = new Int32Array(MAX_ELL), cd = new Float32Array(MAX_ELL);
    ells.forEach((it, i) => {
      cen[i * 2] = it.e.center_x_mm; cen[i * 2 + 1] = it.e.center_y_mm;
      rad[i * 2] = it.e.radius_x_mm; rad[i * 2 + 1] = it.e.radius_y_mm;
      rt[i] = (it.e.rotation_deg * Math.PI) / 180; kd[i] = it.kind; cd[i] = it.cond;
    });
    gl.uniform2fv(this.u.uEllCenter, cen);
    gl.uniform2fv(this.u.uEllRadius, rad);
    gl.uniform1fv(this.u.uEllRot, rt);
    gl.uniform1iv(this.u.uEllKind, kd);
    gl.uniform1fv(this.u.uEllCond, cd);
    const col = new Float32Array(7 * 3);
    for (let i = 0; i < 7; i++) { const c = colors[i]; col[i * 3] = c[0]; col[i * 3 + 1] = c[1]; col[i * 3 + 2] = c[2]; }
    gl.uniform3fv(this.u.uColor, col);
    gl.drawArrays(gl.TRIANGLES, 0, 3);
  }
  resize(w, h) { this.gl.canvas.width = w; this.gl.canvas.height = h; }
}

// --------------------------------------------------------------------------- //
// State + main
// --------------------------------------------------------------------------- //
const field = document.getElementById("field");
const overlay = document.getElementById("overlay");
const octx = overlay.getContext("2d");
const tip = document.getElementById("tip");
const backendTag = document.getElementById("backend");
const hint = document.getElementById("hint");

let renderer = null, backend = "";
let model = null, palette = null, colorMode = 0, meshOverlay = null;
let view = { scale: 1, dpr: 1, side: 400, extent: 10 };
let drag = null, lastEmit = 0, lastEmitted = "";

function hexToRgb(h) {
  const n = parseInt(h.replace("#", ""), 16);
  return [((n >> 16) & 255) / 255, ((n >> 8) & 255) / 255, (n & 255) / 255];
}
function colorArray() {
  const out = [];
  for (let i = 0; i < 7; i++) out.push(hexToRgb((palette.colors[String(i)]) || "#000000"));
  return out;
}

async function initRenderer() {
  // Try WebGPU first, fall back to WebGL2. WebGL2 is the guaranteed path.
  if (navigator.gpu) {
    try {
      const mod = await import("./webgpu_renderer.js");
      renderer = await mod.createWebGPURenderer(field);
      backend = "WebGPU";
      return;
    } catch (err) { /* fall through */ }
  }
  renderer = new GLRenderer(field);
  backend = "WebGL2";
}

function worldToScreen(wx, wy) {
  return [view.side / 2 + wx * view.scaleCss, view.side / 2 - wy * view.scaleCss];
}
function screenToWorld(sx, sy) {
  return [(sx - view.side / 2) / view.scaleCss, -(sy - view.side / 2) / view.scaleCss];
}

function computeView() {
  const side = Math.max(240, Math.min(field.parentElement.clientWidth || 480, view.height));
  const dpr = window.devicePixelRatio || 1;
  const extent = 0.6 * Math.hypot(model.width_mm, model.height_mm);
  view.side = side; view.dpr = dpr; view.extent = extent;
  view.scaleCss = side / (2 * extent);
  view.scale = view.scaleCss * dpr;
  for (const c of [field, overlay]) { c.style.width = side + "px"; c.style.height = side + "px"; }
  renderer.resize(Math.round(side * dpr), Math.round(side * dpr));
  overlay.width = Math.round(side * dpr); overlay.height = Math.round(side * dpr);
  octx.setTransform(dpr, 0, 0, dpr, 0, 0);
  document.getElementById("wrap").style.height = side + "px";
}

function renderField() {
  const c = model.conductivities;
  const ells = ellipseList(model);
  let condMin = 1e9, condMax = -1e9;
  for (const v of [c.skin_s_m, c.fat_s_m, c.muscle_s_m, c.bone_s_m, c.ligament_s_m,
    ...model.arteries.map((a) => a.baseline_conductivity_s_m)]) { condMin = Math.min(condMin, v); condMax = Math.max(condMax, v); }
  renderer.render({
    scale: view.scale, fingerRot: (model.rotation_deg * Math.PI) / 180,
    halfW: model.width_mm / 2, halfH: model.height_mm / 2,
    skinT: model.skin_thickness_mm, fatT: model.fat_thickness_mm,
    colorMode, condMin, condMax,
    skinC: c.skin_s_m, fatC: c.fat_s_m, muscleC: c.muscle_s_m,
  }, ells, colorArray());
  drawOverlay();
  backendTag.textContent = backend;
}

// Handles for each ellipse + finger outline, in screen space.
function handles() {
  const hs = [];
  ellipseList(model).forEach((item) => {
    const e = item.e;
    const invRot = (p) => rot(p[0], p[1], -e.rotation_deg); // local ellipse -> finger-local
    const toWorld = (lx, ly) => rot(lx, ly, -model.rotation_deg); // finger-local -> world
    const place = (lx, ly) => {
      const fl = invRot([lx, ly]); // finger-local offset
      const w = toWorld(e.center_x_mm + fl[0], e.center_y_mm + fl[1]);
      return worldToScreen(w[0], w[1]);
    };
    const cw = toWorld(e.center_x_mm, e.center_y_mm);
    hs.push({ item, part: "center", xy: worldToScreen(cw[0], cw[1]), color: palette.arterial });
    hs.push({ item, part: "rx", xy: place(e.radius_x_mm, 0), color: palette.ink });
    hs.push({ item, part: "ry", xy: place(0, e.radius_y_mm), color: palette.ink });
    hs.push({ item, part: "rot", xy: place(0, e.radius_y_mm + 2.2), color: palette.arterial, ring: true });
  });
  // finger width/height handles (finger-local axes -> world)
  const wpt = rot(model.width_mm / 2, 0, -model.rotation_deg);
  const hpt = rot(0, model.height_mm / 2, -model.rotation_deg);
  hs.push({ item: { ref: "finger" }, part: "w", xy: worldToScreen(wpt[0], wpt[1]), color: "#5B6B7F" });
  hs.push({ item: { ref: "finger" }, part: "h", xy: worldToScreen(hpt[0], hpt[1]), color: "#5B6B7F" });
  return hs;
}

function drawOverlay() {
  octx.clearRect(0, 0, view.side, view.side);
  // mesh triangle preview (light)
  if (meshOverlay && meshOverlay.count) {
    const n = meshOverlay.nodes, el = meshOverlay.elements;
    octx.strokeStyle = "rgba(19,35,59,0.18)";
    octx.lineWidth = 0.6;
    octx.beginPath();
    for (let t = 0; t < el.length; t += 3) {
      const a = el[t] * 2, b = el[t + 1] * 2, c = el[t + 2] * 2;
      const pa = worldToScreen(n[a], n[a + 1]), pb = worldToScreen(n[b], n[b + 1]), pc = worldToScreen(n[c], n[c + 1]);
      octx.moveTo(pa[0], pa[1]); octx.lineTo(pb[0], pb[1]); octx.lineTo(pc[0], pc[1]); octx.closePath();
    }
    octx.stroke();
  }
  // handles
  for (const h of handles()) {
    octx.beginPath();
    octx.arc(h.xy[0], h.xy[1], h.part === "center" ? 5 : 4, 0, Math.PI * 2);
    octx.fillStyle = h.ring ? "#fff" : h.color;
    octx.strokeStyle = h.color; octx.lineWidth = 2;
    octx.fill(); octx.stroke();
  }
}

function hitHandle(sx, sy) {
  for (const h of handles()) {
    if (Math.hypot(h.xy[0] - sx, h.xy[1] - sy) <= 9) return h;
  }
  return null;
}

function clampModel() {
  const minLayer = 2 * (model.skin_thickness_mm + model.fat_thickness_mm) + 1.0;
  model.width_mm = Math.max(minLayer, Math.min(60, model.width_mm));
  model.height_mm = Math.max(minLayer, Math.min(60, model.height_mm));
  for (const it of ellipseList(model)) {
    it.e.radius_x_mm = Math.max(0.2, Math.min(12, it.e.radius_x_mm));
    it.e.radius_y_mm = Math.max(0.2, Math.min(12, it.e.radius_y_mm));
  }
}

function applyDrag(sx, sy) {
  const [wx, wy] = screenToWorld(sx, sy);
  const [lx, ly] = rot(wx, wy, model.rotation_deg); // finger-local pointer
  const h = drag;
  if (h.item.ref === "finger") {
    if (h.part === "w") model.width_mm = 2 * Math.abs(lx);
    else model.height_mm = 2 * Math.abs(ly);
  } else {
    const e = h.item.e;
    if (h.part === "center") { e.center_x_mm = lx; e.center_y_mm = ly; }
    else {
      const d = rot(lx - e.center_x_mm, ly - e.center_y_mm, e.rotation_deg); // ellipse-local
      if (h.part === "rx") e.radius_x_mm = Math.abs(d[0]);
      else if (h.part === "ry") e.radius_y_mm = Math.abs(d[1]);
      else if (h.part === "rot") {
        const vx = lx - e.center_x_mm, vy = ly - e.center_y_mm;
        e.rotation_deg = (Math.atan2(-vx, vy) * 180) / Math.PI;
      }
    }
  }
  clampModel();
  renderField();
  maybeEmit(false);
}

function maybeEmit(force) {
  const now = performance.now();
  if (!force && now - lastEmit < 110) return;
  lastEmit = now;
  const json = JSON.stringify(model);
  if (json === lastEmitted) return;
  lastEmitted = json;
  Streamlit.value(JSON.parse(json));
}

// ---- pointer events -------------------------------------------------------- //
overlay.addEventListener("pointerdown", (ev) => {
  const r = overlay.getBoundingClientRect();
  const sx = ev.clientX - r.left, sy = ev.clientY - r.top;
  const h = hitHandle(sx, sy);
  if (h) { drag = h; overlay.setPointerCapture(ev.pointerId); tip.style.display = "none"; hint.style.display = "none"; }
});
overlay.addEventListener("pointermove", (ev) => {
  const r = overlay.getBoundingClientRect();
  const sx = ev.clientX - r.left, sy = ev.clientY - r.top;
  if (drag) { applyDrag(sx, sy); return; }
  // hover identify
  const [wx, wy] = screenToWorld(sx, sy);
  const res = classify(wx, wy, model);
  if (res.label === OUTSIDE) { tip.style.display = "none"; overlay.style.cursor = hitHandle(sx, sy) ? "grab" : "crosshair"; return; }
  const name = palette.names[String(res.label)] || "tissue";
  tip.innerHTML = `${name} · <b>${isNaN(res.sigma) ? "–" : res.sigma.toFixed(3)} S/m</b>`;
  tip.style.display = "block";
  tip.style.left = (sx + 14) + "px";
  tip.style.top = (sy + 12) + "px";
  overlay.style.cursor = hitHandle(sx, sy) ? "grab" : "crosshair";
});
function endDrag(ev) {
  if (drag) { drag = null; maybeEmit(true); }
}
overlay.addEventListener("pointerup", endDrag);
overlay.addEventListener("pointercancel", endDrag);
overlay.addEventListener("pointerleave", () => { if (!drag) tip.style.display = "none"; });

// ---- Streamlit render ------------------------------------------------------ //
let started = false;
async function onRender(args) {
  const incoming = args.model;
  palette = args.palette;
  colorMode = args.color_by === "conductivity" ? 1 : 0;
  meshOverlay = args.mesh_overlay || null;
  view.height = args.height || 560;

  if (!renderer) { await initRenderer(); }

  const incomingJson = JSON.stringify(incoming);
  const adoptExternal = !drag && incomingJson !== JSON.stringify(model) && incomingJson !== lastEmitted;
  if (!model || adoptExternal) { model = JSON.parse(incomingJson); }

  computeView();
  renderField();
  Streamlit.height(view.side + 4);
  started = true;
}

window.addEventListener("message", (event) => {
  if (event.data && event.data.type === "streamlit:render") {
    onRender(event.data.args).catch((e) => { backendTag.textContent = "error: " + e.message; });
  }
});
window.addEventListener("resize", () => { if (started && model) { computeView(); renderField(); Streamlit.height(view.side + 4); } });

Streamlit.ready();
Streamlit.height(560);
