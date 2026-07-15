/* Pure tissue geometry — mirrors finger_sim.geometry exactly.
 * Kept in its own module so it can be unit-tested in node against Python. */
"use strict";

export const OUTSIDE = 0, SKIN = 1, FAT = 2, MUSCLE = 3, BONE = 4, LIGAMENT = 5, ARTERY = 6;

// Matches finger_sim.geometry._rotate: points @ [[c,s],[-s,c]].T
export function rot(x, y, deg) {
  const a = (deg * Math.PI) / 180, c = Math.cos(a), s = Math.sin(a);
  return [x * c + y * s, -x * s + y * c];
}

// Ordered interactive ellipses (later entries override earlier ones).
export function ellipseList(model) {
  const list = [];
  list.push({ ref: "bone", kind: BONE, e: model.bone, cond: model.conductivities.bone_s_m });
  model.ligaments.forEach((e, i) =>
    list.push({ ref: "lig" + i, kind: LIGAMENT, e, cond: model.conductivities.ligament_s_m }));
  model.arteries.forEach((e, i) =>
    list.push({ ref: "art" + i, kind: ARTERY, e, cond: e.baseline_conductivity_s_m }));
  return list;
}

// Classify a single world point. Returns {label, sigma}.
export function classify(wx, wy, model) {
  const [lx, ly] = rot(wx, wy, model.rotation_deg);
  const hw = model.width_mm / 2, hh = model.height_mm / 2;
  let label = OUTSIDE, sigma = NaN;
  if ((lx / hw) ** 2 + (ly / hh) ** 2 <= 1) { label = SKIN; sigma = model.conductivities.skin_s_m; }
  const sx = hw - model.skin_thickness_mm, sy = hh - model.skin_thickness_mm;
  if (sx > 0 && sy > 0 && (lx / sx) ** 2 + (ly / sy) ** 2 <= 1) { label = FAT; sigma = model.conductivities.fat_s_m; }
  const fx = sx - model.fat_thickness_mm, fy = sy - model.fat_thickness_mm;
  if (fx > 0 && fy > 0 && (lx / fx) ** 2 + (ly / fy) ** 2 <= 1) { label = MUSCLE; sigma = model.conductivities.muscle_s_m; }
  for (const item of ellipseList(model)) {
    const e = item.e;
    const [dx, dy] = rot(lx - e.center_x_mm, ly - e.center_y_mm, e.rotation_deg);
    if ((dx / e.radius_x_mm) ** 2 + (dy / e.radius_y_mm) ** 2 <= 1) { label = item.kind; sigma = item.cond; }
  }
  return { label, sigma };
}
