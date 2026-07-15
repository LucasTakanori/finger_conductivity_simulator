/* WebGPU field renderer. Same per-pixel tissue classification as the WebGL2
 * path, expressed in WGSL. Exposes the same {render, resize} interface so the
 * host code is backend-agnostic. Any failure here makes the host fall back. */
"use strict";

const WGSL = `
struct U {
  a: vec4<f32>,               // res.x, res.y, scale, fingerRot
  b: vec4<f32>,               // halfW, halfH, skinT, fatT
  c: vec4<f32>,               // colorMode, condMin, condMax, ellCount
  d: vec4<f32>,               // skinC, fatC, muscleC, _
  ell:  array<vec4<f32>, 5>,  // cx, cy, rx, ry
  meta: array<vec4<f32>, 5>,  // rot, kind, cond, _
  col:  array<vec4<f32>, 7>,  // r, g, b, _
};
@group(0) @binding(0) var<uniform> u: U;

fn rot2(p: vec2<f32>, a: f32) -> vec2<f32> {
  let c = cos(a); let s = sin(a);
  return vec2<f32>(p.x * c + p.y * s, -p.x * s + p.y * c);
}
fn cividis(t0: f32) -> vec3<f32> {
  let t = clamp(t0, 0.0, 1.0);
  let c0 = vec3<f32>(0.0, 0.135, 0.304);
  let c1 = vec3<f32>(0.404, 0.443, 0.475);
  let c2 = vec3<f32>(1.0, 0.910, 0.216);
  if (t < 0.5) { return mix(c0, c1, t * 2.0); }
  return mix(c1, c2, (t - 0.5) * 2.0);
}

@vertex fn vs(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4<f32> {
  var p = array<vec2<f32>, 3>(vec2<f32>(-1.0, -1.0), vec2<f32>(3.0, -1.0), vec2<f32>(-1.0, 3.0));
  return vec4<f32>(p[vi], 0.0, 1.0);
}

@fragment fn fs(@builtin(position) fragCoord: vec4<f32>) -> @location(0) vec4<f32> {
  let res = u.a.xy; let scale = u.a.z; let fingerRot = u.a.w;
  let px = vec2<f32>(fragCoord.x, res.y - fragCoord.y);   // flip to world-up
  let world = (px - 0.5 * res) / scale;
  let L = rot2(world, fingerRot);
  let half = u.b.xy; let skinT = u.b.z; let fatT = u.b.w;
  var label: i32 = 0; var sigma: f32 = -1.0;
  if ((L.x / half.x) * (L.x / half.x) + (L.y / half.y) * (L.y / half.y) <= 1.0) { label = 1; sigma = u.d.x; }
  let sR = half - vec2<f32>(skinT, skinT);
  if (sR.x > 0.0 && sR.y > 0.0 && (L.x / sR.x) * (L.x / sR.x) + (L.y / sR.y) * (L.y / sR.y) <= 1.0) { label = 2; sigma = u.d.y; }
  let fR = sR - vec2<f32>(fatT, fatT);
  if (fR.x > 0.0 && fR.y > 0.0 && (L.x / fR.x) * (L.x / fR.x) + (L.y / fR.y) * (L.y / fR.y) <= 1.0) { label = 3; sigma = u.d.z; }
  let count = i32(u.c.w);
  for (var i: i32 = 0; i < 5; i = i + 1) {
    if (i >= count) { break; }
    let e = u.ell[i]; let m = u.meta[i];
    let cc = rot2(L - e.xy, m.x);
    let rr = (cc.x / e.z) * (cc.x / e.z) + (cc.y / e.w) * (cc.y / e.w);
    if (rr <= 1.0) { label = i32(m.y); sigma = m.z; }
  }
  if (label == 0) { return vec4<f32>(0.0, 0.0, 0.0, 0.0); }
  var rgb: vec3<f32>;
  if (i32(u.c.x) == 0) { rgb = u.col[label].xyz; }
  else { let t = (sigma - u.c.y) / max(u.c.z - u.c.y, 1e-6); rgb = cividis(t); }
  return vec4<f32>(rgb, 1.0);
}
`;

export async function createWebGPURenderer(canvas) {
  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) throw new Error("no adapter");
  const device = await adapter.requestDevice();
  const ctx = canvas.getContext("webgpu");
  const format = navigator.gpu.getPreferredCanvasFormat();
  ctx.configure({ device, format, alphaMode: "premultiplied" });

  const module = device.createShaderModule({ code: WGSL });
  const pipeline = device.createRenderPipeline({
    layout: "auto",
    vertex: { module, entryPoint: "vs" },
    fragment: { module, entryPoint: "fs", targets: [{ format }] },
    primitive: { topology: "triangle-list" },
  });

  const UBO_FLOATS = 84;
  const ubo = device.createBuffer({
    size: UBO_FLOATS * 4,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  const bind = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [{ binding: 0, resource: { buffer: ubo } }],
  });
  const data = new Float32Array(UBO_FLOATS);

  return {
    resize(w, h) {
      canvas.width = w; canvas.height = h;
      ctx.configure({ device, format, alphaMode: "premultiplied" });
    },
    render(u, ells, colors) {
      data.fill(0);
      data[0] = canvas.width; data[1] = canvas.height; data[2] = u.scale; data[3] = u.fingerRot;
      data[4] = u.halfW; data[5] = u.halfH; data[6] = u.skinT; data[7] = u.fatT;
      data[8] = u.colorMode; data[9] = u.condMin; data[10] = u.condMax; data[11] = ells.length;
      data[12] = u.skinC; data[13] = u.fatC; data[14] = u.muscleC;
      ells.forEach((it, i) => {
        const o = 16 + i * 4;
        data[o] = it.e.center_x_mm; data[o + 1] = it.e.center_y_mm;
        data[o + 2] = it.e.radius_x_mm; data[o + 3] = it.e.radius_y_mm;
        const p = 36 + i * 4;
        data[p] = (it.e.rotation_deg * Math.PI) / 180; data[p + 1] = it.kind; data[p + 2] = it.cond;
      });
      for (let i = 0; i < 7; i++) {
        const o = 56 + i * 4; const c = colors[i];
        data[o] = c[0]; data[o + 1] = c[1]; data[o + 2] = c[2];
      }
      device.queue.writeBuffer(ubo, 0, data);
      const enc = device.createCommandEncoder();
      const pass = enc.beginRenderPass({
        colorAttachments: [{
          view: ctx.getCurrentTexture().createView(),
          clearValue: { r: 0, g: 0, b: 0, a: 0 },
          loadOp: "clear", storeOp: "store",
        }],
      });
      pass.setPipeline(pipeline);
      pass.setBindGroup(0, bind);
      pass.draw(3);
      pass.end();
      device.queue.submit([enc.finish()]);
    },
  };
}
