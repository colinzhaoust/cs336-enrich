// Widgets for thread rope (Su et al., RoFormer, arXiv:2104.09864v5). Register as "fixture:<id>" -> (root, notice) => void.
// Each widget has a pure model in MODELS (no DOM) that tools/check_widgets.mjs tests against the KPs' stored answers.
// Sources (papers/rope/sources.json): the paper as page images 4blue2brown/tmp/pdfs/rope/page-0X.png / results-0X.png,
// cited as rope:eqNN / rope:pN; equations are also in papers/rope/discovered.json. The only constant the paper fixes is
// the base 10000 of θ_i (Eq. 15, p5; §3.3, p5; §3.4.3, p8). Example vectors and angles are the KP prompts' own
// (cited as "KP <id>:<prompt>"); anything else is marked "constructed" or "inference" where it is used.
import { el, fmt, slider } from "../widgets_lib.js";

export const MODELS = {};
export const WIDGETS = {};

const C = { ink: "var(--ink)", muted: "var(--muted)", rule: "var(--rule)", a: "var(--accent)", b: "var(--accent2)", ok: "var(--ok)", hi: "var(--hilite)" };
const text = (x, y, s, o = {}) => `<text x="${(+x).toFixed(1)}" y="${(+y).toFixed(1)}" style="fill:${o.fill || C.ink};font:${o.size || 12}px var(--sans)" text-anchor="${o.anchor || "start"}">${s}</text>`;
const line = (x1, y1, x2, y2, col, w = 1, extra = "") => `<line x1="${x1.toFixed(1)}" y1="${y1.toFixed(1)}" x2="${x2.toFixed(1)}" y2="${y2.toFixed(1)}" style="stroke:${col};stroke-width:${w}" ${extra}/>`;
const circle = (cx, cy, r, stroke, fill = "none", extra = "") => `<circle cx="${cx.toFixed(1)}" cy="${cy.toFixed(1)}" r="${r.toFixed(1)}" style="stroke:${stroke};fill:${fill}" ${extra}/>`;
const path = (d, col, w = 1.5, extra = "") => `<path d="${d}" style="stroke:${col};stroke-width:${w};fill:none" ${extra}/>`;
const svg = (w, h, body) => `<svg viewBox="0 0 ${w} ${h}" width="100%" style="max-width:${w}px;border:1px solid var(--rule);border-radius:6px;background:#fff" role="img">${body}</svg>`;
const check = (label, value, onChange) => { const i = el("input", { type: "checkbox" }); i.checked = value; i.addEventListener("change", () => onChange(i.checked)); return el("label", {}, i, ` ${label}`); };
const select = (label, options, value, onChange) => {
  const s = el("select", {}); options.forEach(([v, t]) => { const o = el("option", { value: v }, t); if (v === value) o.selected = true; s.append(o); });
  s.addEventListener("change", () => onChange(s.value)); return el("label", {}, `${label} `, s);
};
// arrow from (x0, y0) along screen vector (dx, dy) with a small head
const arrow = (x0, y0, dx, dy, col, w = 3) => {
  const L = Math.hypot(dx, dy); if (L < 1e-6) return circle(x0, y0, 2, col, col);
  const ux = dx / L, uy = dy / L, h = Math.min(9, L * 0.4), x1 = x0 + dx, y1 = y0 + dy;
  return line(x0, y0, x1, y1, col, w) + `<path d="M${x1.toFixed(1)},${y1.toFixed(1)} L${(x1 - h * ux - h * 0.5 * uy).toFixed(1)},${(y1 - h * uy + h * 0.5 * ux).toFixed(1)} L${(x1 - h * ux + h * 0.5 * uy).toFixed(1)},${(y1 - h * uy - h * 0.5 * ux).toFixed(1)} Z" style="fill:${col}"/>`;
};

// ---------------------------------------------------------------- shared math ---
const DEG = Math.PI / 180;
const rot = (v, a) => [Math.cos(a) * v[0] - Math.sin(a) * v[1], Math.sin(a) * v[0] + Math.cos(a) * v[1]];
const dot = (a, b) => a[0] * b[0] + a[1] * b[1];
const wrapDeg = x => { const y = ((x % 360) + 540) % 360 - 180; return y === -180 ? 180 : y; };   // to (−180, 180]
const BASE = 10000;                                   // Eq. 15 (p5): θ_i = 10000^(−2(i−1)/d), i = 1..d/2
// Eq. 15's frequencies, 1-indexed as printed (§3.3 / §3.4.3 write the same set 0-indexed as 10000^(−2i/d)).
const thetas = (d, base = BASE) => Array.from({ length: d / 2 }, (_, j) => base ** (-2 * j / d));

// ============================================================ 1. rotation pair ==
// Two 2-vectors rotated by their own angles: R(α)q · R(β)k = q^T R(β−α) k (rotation is orthogonal, R^T R = I;
// the paper uses it at Eq. 16, p5: "R^d_{Θ,n−m} = (R^d_{Θ,m})^T R^d_{Θ,n}"). Complex view (Eq. 12, p4):
// with z = q1 + i q2, w = k1 + i k2, Re[z e^{iα} (w e^{iβ})^*] = Re[z w^* e^{i(α−β)}] is the same dot product;
// without the conjugate the phase is α + β. Vectors and angles are the prompts' (KP supp-rotation-preserves-dot,
// KP supp-complex-rotation); no other constant.
function rotationPair({ q, k, alpha, beta }) {
  const a = alpha * DEG, b = beta * DEG, rq = rot(q, a), rk = rot(k, b);
  const angQ = Math.atan2(rq[1], rq[0]) / DEG, angK = Math.atan2(rk[1], rk[0]) / DEG;
  const zw = [q[0] * k[0] + q[1] * k[1], q[1] * k[0] - q[0] * k[1]];          // z · w^*  (real, imaginary)
  const zwNo = [q[0] * k[0] - q[1] * k[1], q[1] * k[0] + q[0] * k[1]];        // z · w    (no conjugate)
  const ph = a - b, phNo = a + b;
  return {
    rq, rk, normQ: Math.hypot(...q), normK: Math.hypot(...k), normRq: Math.hypot(...rq), normRk: Math.hypot(...rk),
    angQ, angK, gapDeg: Math.abs(wrapDeg(angK - angQ)),                        // angle between the rotated arrows
    gap0Deg: Math.abs(wrapDeg(Math.atan2(k[1], k[0]) / DEG - Math.atan2(q[1], q[0]) / DEG)),
    diffDeg: beta - alpha,
    dot: dot(rq, rk), relDot: dot(q, rot(k, b - a)), dot0: dot(q, k),
    complex: zw[0] * Math.cos(ph) - zw[1] * Math.sin(ph),                       // Re[z w^* e^{i(α−β)}]
    noConj: zwNo[0] * Math.cos(phNo) - zwNo[1] * Math.sin(phNo),                // Re[z w e^{i(α+β)}]
    noConjPhaseDeg: alpha + beta,
  };
}
MODELS["fixture:rope--rotation-pair"] = {
  fn: rotationPair,
  cases: [
    // supp-rotation-preserves-dot predict (short): q = (1,0), k = (0.7,0.7), α = 30°, β = 90° -> norm 1, angle 105°
    { args: { q: [1, 0], k: [0.7, 0.7], alpha: 30, beta: 90 }, pick: "normRq", expect: 1 },
    { args: { q: [1, 0], k: [0.7, 0.7], alpha: 30, beta: 90 }, pick: "gapDeg", expect: 105 },
    // supp-rotation-preserves-dot transfer (short): rotating only q by α − β gives the same dot product
    { args: { q: [1, 0], k: [0.7, 0.7], alpha: -60, beta: 0 }, pick: "dot", expect: rotationPair({ q: [1, 0], k: [0.7, 0.7], alpha: 30, beta: 90 }).dot },
    { args: { q: [2, 0], k: [0, 3], alpha: 100, beta: 40 }, pick: "dot", expect: 5.196, tol: 0.003, from: "rope:supp-rotation-preserves-dot:check" },
    { args: { q: [2, 0], k: [0, 3], alpha: 100, beta: 40 }, pick: "relDot", expect: 5.196, tol: 0.003 },   // q^T R(β−α) k agrees
    { args: { q: [2, 0], k: [0, 3], alpha: 100, beta: 40 }, pick: "normRk", expect: 3 },
    { args: { q: [1, 0], k: [0.7, 0.7], alpha: 30, beta: 90 }, pick: "complex", expect: -0.256, tol: 0.05, from: "rope:supp-complex-rotation:predict" },
    // supp-complex-rotation check: z = 2+i, w = 1−i, m = 3, n = 1, θ = π/3 -> α = 180°, β = 60°
    { args: { q: [2, 1], k: [1, -1], alpha: 180, beta: 60 }, pick: "complex", expect: -3.098, tol: 0.005, from: "rope:supp-complex-rotation:check" },
    { args: { q: [2, 1], k: [1, -1], alpha: 180, beta: 60 }, pick: "dot", expect: -3.098, tol: 0.005 },     // the 2D dot product agrees
    { args: { q: [2, 1], k: [1, -1], alpha: 180, beta: 60 }, pick: "noConj", expect: -2.366, tol: 0.005 },  // check's "forgot the conjugate"
    { args: { q: [1, 0], k: [0.7, 0.7], alpha: 30, beta: 90 }, pick: "noConjPhaseDeg", expect: 120 },        // transfer: phase 2π/3 (m + n)
    { args: { q: [1, 0], k: [0.7, 0.7], alpha: 70, beta: 70 }, pick: "dot", expect: 0.7 },                   // edge: equal angles = unrotated
  ],
};
WIDGETS["fixture:rope--rotation-pair"] = (root) => {
  const Q = { "1,0": [1, 0], "2,0": [2, 0], "2,1": [2, 1] }, K = { "0.7,0.7": [0.7, 0.7], "0,3": [0, 3], "1,-1": [1, -1] };
  const s = { q: "1,0", k: "0.7,0.7", alpha: 60, beta: 20, conj: true };
  const pic = el("div"), read = el("div", { class: "readout" });
  const draw = () => {
    const q = Q[s.q], k = K[s.k], m = rotationPair({ q, k, alpha: s.alpha, beta: s.beta });
    const W = 300, c = 150, u = 40;                                         // 1 unit = 40 px; norm 3 reaches 120 px
    const P = v => [v[0] * u, -v[1] * u];
    let b = circle(c, c, m.normQ * u, C.a, "none", 'stroke-dasharray="3 3" opacity="0.6"') + circle(c, c, m.normK * u, C.b, "none", 'stroke-dasharray="3 3" opacity="0.6"');
    b += line(c - 140, c, c + 140, c, C.rule) + line(c, c - 140, c, c + 140, C.rule);
    b += `<g opacity="0.3">${arrow(c, c, ...P(q), C.a, 2)}${arrow(c, c, ...P(k), C.b, 2)}</g>`;
    b += arrow(c, c, ...P(m.rq), C.a, 3.5) + arrow(c, c, ...P(m.rk), C.b, 3.5);
    // arc for the gap between the rotated arrows (radius 26 px), drawn the short way round
    const a0 = m.angQ * DEG, d = wrapDeg(m.angK - m.angQ) * DEG, a1 = a0 + d, r = 26;
    b += path(`M${(c + r * Math.cos(a0)).toFixed(1)},${(c - r * Math.sin(a0)).toFixed(1)} A${r},${r} 0 0 ${d > 0 ? 0 : 1} ${(c + r * Math.cos(a1)).toFixed(1)},${(c - r * Math.sin(a1)).toFixed(1)}`, C.hi, 2.5);
    b += text(8, 16, "faint: q, k unrotated · bold: R(α)q, R(β)k · dashed: |q|, |k|", { fill: C.muted, size: 11 });
    b += text(8, W - 8, `gap ${fmt(m.gapDeg, 1)}° (was ${fmt(m.gap0Deg, 1)}°)`, { size: 12 });
    pic.innerHTML = svg(W, W, b);
    const same = Math.abs(m.dot - m.relDot) < 1e-9;
    read.innerHTML = `|q| = ${fmt(m.normQ, 3)} → |R(α)q| = <b>${fmt(m.normRq, 3)}</b> · |k| = ${fmt(m.normK, 3)} → |R(β)k| = <b>${fmt(m.normRk, 3)}</b> (lengths never change)<br>
      angle between the arrows: ${fmt(m.gap0Deg, 1)}° before, <b>${fmt(m.gapDeg, 1)}°</b> after (k's direction minus q's direction changed by β − α = ${s.beta - s.alpha}°, mod 360)<br>
      R(α)q · R(β)k = <span class="big">${m.dot.toFixed(4)}</span> &nbsp; q · R(β−α)k = <b>${m.relDot.toFixed(4)}</b> ${same ? `<span style="color:${C.ok}">equal: only β − α matters</span>` : ""}<br>
      complex form, z = q₁ + i q₂, w = k₁ + i k₂: ${s.conj
        ? `Re[z e<sup>iα</sup> (w e<sup>iβ</sup>)*] = Re[z w* e<sup>i(α−β)</sup>] = <b>${m.complex.toFixed(4)}</b> = the dot product (phase α − β = ${s.alpha - s.beta}°)`
        : `<span style="color:${C.b}">no conjugate</span>: Re[z e<sup>iα</sup> w e<sup>iβ</sup>] = Re[z w e<sup>i(α+β)</sup>] = <b>${m.noConj.toFixed(4)}</b> ≠ the dot product: phase α + β = ${m.noConjPhaseDeg}° (with α = mθ, β = nθ that is (m+n)θ, not relative)`}<br>
      <span class="muted small">provenance: fixture:rope--rotation-pair · R(α)^T R(β) = R(β−α) as used at rope:eq16 (p5); complex form rope:eq12 (p4). Vectors are the KP prompts' (supp-rotation-preserves-dot, supp-complex-rotation). In RoPE's 2D case α = mθ and β = nθ.</span>`;
  };
  root.append(el("div", { class: "widget" }, el("div", { class: "controls" },
    select("q", [["1,0", "(1, 0)"], ["2,0", "(2, 0)"], ["2,1", "(2, 1)  z = 2 + i"]], s.q, v => { s.q = v; draw(); }),
    select("k", [["0.7,0.7", "(0.7, 0.7)"], ["0,3", "(0, 3)"], ["1,-1", "(1, −1)  w = 1 − i"]], s.k, v => { s.k = v; draw(); }),
    slider("rotate q by α (deg)", -180, 360, s.alpha, 5, v => { s.alpha = v; draw(); }),
    slider("rotate k by β (deg)", -180, 360, s.beta, 5, v => { s.beta = v; draw(); }),
    check("conjugate the key (w*)", s.conj, v => { s.conj = v; draw(); }),
    read), pic)); draw();
};

// ========================================================= 2. frequency dials ==
// Block-diagonal RoPE (Eq. 14-15, p5): pair i of a d-vector rotates by m·θ_i, θ_i = 10000^(−2(i−1)/d), i = 1..d/2.
// The rotated vector here is x = (1, 0, 1, 0, ...), every pair a unit arrow at angle 0, so pair i's arrow points
// at m·θ_i (the KP rope-block-diagonal transfer uses x = (1, 0, 1, 0) at m = 2). Wavelength 2π/θ_i (positions
// per full turn) is derived, not printed in the paper. Only constant: the base 10000.
function frequencyDials({ d, m, base = BASE }) {
  const th = thetas(d, base), o = { pairs: d / 2, ratio: th.length > 1 ? th[1] / th[0] : 1 };
  o.rows = th.map((t, j) => {
    const ang = m * t, v = rot([1, 0], ang);
    return { i: j + 1, theta: t, angle: ang, wrapped: wrapDeg(ang / DEG) * DEG, turns: ang / (2 * Math.PI), wavelength: 2 * Math.PI / t, x: v[0], y: v[1] };
  });
  o.rows.forEach(r => { o[`theta${r.i}`] = r.theta; o[`angle${r.i}`] = r.angle; o[`x${r.i}`] = r.x; o[`y${r.i}`] = r.y; o[`wavelength${r.i}`] = r.wavelength; });
  o.thetaLast = th[th.length - 1];
  return o;
}
MODELS["fixture:rope--frequency-dials"] = {
  fn: frequencyDials,
  cases: [
    // rope-block-diagonal predict (short): d = 8 -> 1, 0.1, 0.01, 0.001
    { args: { d: 8, m: 1 }, pick: "theta1", expect: 1 },
    { args: { d: 8, m: 1 }, pick: "theta2", expect: 0.1 },
    { args: { d: 8, m: 1 }, pick: "theta3", expect: 0.01 },
    { args: { d: 8, m: 1 }, pick: "theta4", expect: 0.001 },
    // rope-block-diagonal transfer (short): d = 4, x = (1,0,1,0), m = 2 -> second pair (1.000, 0.020), first (−0.416, 0.909)
    { args: { d: 4, m: 2 }, pick: "x2", expect: 0.9998 },
    { args: { d: 4, m: 2 }, pick: "y2", expect: 0.0200 },
    { args: { d: 4, m: 2 }, pick: "x1", expect: -0.416 },
    { args: { d: 4, m: 2 }, pick: "y1", expect: 0.909 },
    { args: { d: 6, m: 500 }, pick: "angle3", expect: 1.0772, tol: 0.03, from: "rope:rope-block-diagonal:check" },
    { args: { d: 6, m: 500 }, pick: "x3", expect: 0.4738 },                   // the check's rotatePairs output (0.4738, 0.8806)
    { args: { d: 6, m: 500 }, pick: "y3", expect: 0.8806 },
    { args: { d: 128, m: 1 }, pick: "thetaLast", expect: 10000 ** (-126 / 128) },   // edge: last pair just above 10000^−1 (KP statement)
    { args: { d: 128, m: 1 }, pick: "wavelength1", expect: 2 * Math.PI },     // pair 1 turns once every 6.28 positions
    { args: { d: 128, m: 1 }, pick: "ratio", expect: 10000 ** (-2 / 128) },   // geometric: constant ratio between pairs
  ],
};
WIDGETS["fixture:rope--frequency-dials"] = (root) => {
  const s = { d: 16, m: 10 };
  const pic = el("div"), read = el("div", { class: "readout" });
  const draw = () => {
    const mo = frequencyDials({ d: s.d, m: s.m }), n = mo.pairs;
    // which pairs get a dial: all if ≤ 16, else 16 spread evenly including the first and the last
    const show = n <= 16 ? mo.rows : Array.from({ length: 16 }, (_, j) => mo.rows[Math.round(j * (n - 1) / 15)]);
    const W = 660, cols = 8, cw = W / cols, R = 26, rowH = 92;
    let b = text(8, 16, `x = (1, 0, 1, 0, …) at position m = ${s.m}: pair i turns by m·θ_i · faint: angle 0`, { fill: C.muted, size: 11 });
    show.forEach((r, j) => {
      const cx = cw * (j % cols) + cw / 2, cy = 30 + R + Math.floor(j / cols) * rowH;
      b += circle(cx, cy, R, C.rule) + line(cx, cy, cx + R, cy, C.rule, 1);
      b += arrow(cx, cy, R * r.x, -R * r.y, r.i === 1 ? C.b : C.a, 2.5);
      b += text(cx, cy + R + 14, `i = ${r.i}`, { size: 11, anchor: "middle" });
      b += text(cx, cy + R + 27, `θ ${fmt(r.theta, 4)}`, { size: 10, anchor: "middle", fill: C.muted });
    });
    // log-scale strip of θ_i against i: a straight line means a geometric sequence
    const y0 = 30 + Math.ceil(show.length / cols) * rowH + 10, H = 120, x0 = 60, span = W - x0 - 20;
    const ly = t => y0 + 10 + (H - 20) * (-Math.log10(t)) / 4;                // 10^0 at the top, 10^−4 at the bottom
    for (let e = 0; e <= 4; e++) { b += line(x0, ly(10 ** -e), x0 + span, ly(10 ** -e), C.rule, 1, 'stroke-dasharray="2 3"'); b += text(x0 - 6, ly(10 ** -e) + 4, `1e-${e}`, { size: 10, anchor: "end", fill: C.muted }); }
    mo.rows.forEach(r => { const x = x0 + span * (n === 1 ? 0.5 : (r.i - 1) / (n - 1)); b += circle(x, ly(r.theta), n > 32 ? 2 : 3, C.a, C.a); });
    b += text(x0, y0 + H + 6, "θ_i on a log axis, i = 1 … d/2 (a straight line: each pair is 10000^(−2/d) slower than the one before)", { size: 10, fill: C.muted });
    pic.innerHTML = svg(W, y0 + H + 14, b);
    const rows = (n <= 8 ? mo.rows : [...mo.rows.slice(0, 4), null, ...mo.rows.slice(-3)]).map(r => r === null ? "  …" :
      `${String(r.i).padStart(3)}  ${fmt(r.theta, 5).padStart(10)}  ${r.angle.toFixed(4).padStart(10)}  ${(wrapDeg(r.angle / DEG)).toFixed(1).padStart(7)}°  ${r.turns.toFixed(3).padStart(8)}  ${fmt(r.wavelength, 1).padStart(9)}  (${r.x.toFixed(4)}, ${r.y.toFixed(4)})`).join("\n");
    read.innerHTML = `d = ${s.d}: ${n} pairs, θ from 1 down to ${fmt(mo.thetaLast, 6)} (ratio ${fmt(mo.ratio, 4)} per pair)<br>
      <pre>  i         θ_i      m·θ_i   wrapped     turns  2π/θ_i   rotated pair\n${rows}</pre>
      <span class="muted small">provenance: fixture:rope--frequency-dials · rope:eq15 (p5) θ_i = 10000^(−2(i−1)/d), i = 1..d/2 (§3.3 writes it 0-indexed as 10000^(−2i/d)); rope:eq14 R^d_{Θ,m} acts pair by pair; "2π/θ_i" and "turns" are derived. Same as rope.mjs rotatePairs.</span>`;
  };
  root.append(el("div", { class: "widget" }, el("div", { class: "controls" },
    slider("head dimension d", 2, 128, s.d, 2, v => { s.d = v; draw(); }),
    slider("position m", 0, 1000, s.m, 1, v => { s.m = v; draw(); }),
    read), pic)); draw();
};

// ============================================================ 3. decay bound ==
// §3.4.3 (p8): score = Re Σ_{i=0}^{d/2−1} h_i e^{i(m−n)θ_i} (Eq. 35), h_i = q_[2i:2i+1] k*_[2i:2i+1],
// S_j = Σ_{k=0}^{j−1} e^{i(m−n)θ_k}, h_{d/2} = 0, S_0 = 0. Abel summation (Eq. 36) gives Eq. 37:
// |Σ h_i e^{i(m−n)θ_i}| ≤ (max_i |h_{i+1} − h_i|) · Σ_{i=0}^{d/2−1} |S_{i+1}|. Fig. 2 (p8) plots the "relative upper
// bound" (1/(d/2)) Σ_{i=1}^{d/2} |S_i| against relative distance with θ_i = 10000^(−2i/d).
// Fig. 2 does not print d. d = 128 reproduces its scale (≈ 20 near the start, 7-12 between 100 and 250): inference.
// "one θ" (all θ_i = θ_0 = 1) is the KP transfer's case. The aligned pair (h_i = e^{−i r0 θ_i}, unit magnitude) is
// constructed here to show one score against the bound; it is not from the paper.
function decayBound({ d, r, equal = false, r0 = 0, base = BASE }) {
  const th = equal ? Array(d / 2).fill(1) : thetas(d, base), half = d / 2;
  let Sre = 0, Sim = 0, sumAbs = 0;
  for (const t of th) { Sre += Math.cos(r * t); Sim += Math.sin(r * t); sumAbs += Math.hypot(Sre, Sim); }   // |S_1| … |S_{d/2}|
  let score = 0, content = 0;
  for (let i = 0; i < half; i++) {
    score += Math.cos((r - r0) * th[i]);                                      // Re[h_i e^{i r θ_i}], h_i = e^{−i r0 θ_i}
    const hi = [Math.cos(r0 * th[i]), -Math.sin(r0 * th[i])], hn = i + 1 < half ? [Math.cos(r0 * th[i + 1]), -Math.sin(r0 * th[i + 1])] : [0, 0];
    content = Math.max(content, Math.hypot(hn[0] - hi[0], hn[1] - hi[1]));  // max_i |h_{i+1} − h_i|, with h_{d/2} = 0
  }
  const fullBound = content * sumAbs;
  return { bound: sumAbs / half, bound0: (half + 1) / 2, rel: (sumAbs / half) / ((half + 1) / 2), sumAbs, score, content, fullBound,
    boundHolds: Math.abs(score) <= fullBound + 1e-9 ? 1 : 0, scoreMax: half };
}
MODELS["fixture:rope--decay-bound"] = {
  fn: decayBound,
  cases: [
    { args: { d: 128, r: 0 }, pick: "bound", expect: 32.5 },                   // distance 0: |S_j| = j, mean (d/2 + 1)/2
    { args: { d: 128, r: 10 }, pick: "bound", expect: 17.95 },                 // Fig. 2 near its start: about 18-20
    { args: { d: 128, r: 100 }, pick: "bound", expect: 10.23 },                // Fig. 2 at 100: about 9-11
    { args: { d: 128, r: 250 }, pick: "bound", expect: 6.55 },                 // Fig. 2 at 250: about 7
    // rope-long-term-decay transfer (short): all θ_i equal -> |S_j| = j at every distance, no decay
    { args: { d: 128, r: 100, equal: true }, pick: "bound", expect: 32.5 },
    { args: { d: 128, r: 250, equal: true }, pick: "bound", expect: 32.5 },
    // rope-long-term-decay check (choice): a head can score high at distance 300; the bound still holds
    { args: { d: 128, r: 300, r0: 300 }, pick: "score", expect: 64 },          // = d/2, the largest score unit h_i allow
    { args: { d: 128, r: 300, r0: 300 }, pick: "boundHolds", expect: 1 },
    { args: { d: 128, r: 0, r0: 300 }, pick: "boundHolds", expect: 1 },
  ],
};
WIDGETS["fixture:rope--decay-bound"] = (root) => {
  const s = { d: 128, r: 50, Rmax: 256, equal: false, aligned: false, r0: 150 };
  const pic = el("div"), read = el("div", { class: "readout" });
  const r0Slider = slider("aligned distance r0", 0, 2048, s.r0, 1, v => { s.r0 = v; draw(); });
  const draw = () => {
    const at = decayBound({ d: s.d, r: s.r, equal: s.equal, r0: s.r0 }), step = s.Rmax / 512;
    const xs = Array.from({ length: 513 }, (_, j) => j * step);
    const cur = xs.map(r => decayBound({ d: s.d, r, equal: s.equal }).bound);
    const geo = s.equal ? xs.map(r => decayBound({ d: s.d, r }).bound) : null;
    const W = 660, x0 = 50, span = W - x0 - 20, top = 26, H = 200, ymax = at.bound0 * 1.05;
    const X = r => x0 + span * r / s.Rmax, Y = v => top + H - H * v / ymax;
    const poly = (vals, Yf) => vals.map((v, j) => `${j ? "L" : "M"}${X(xs[j]).toFixed(1)},${Yf(v).toFixed(1)}`).join("");
    let b = text(8, 16, `relative upper bound (1/(d/2)) Σ|S_i| against relative distance m − n · d = ${s.d}${s.equal ? " · all θ_i = 1 (faint: Eq. 15's θ_i)" : " · θ_i = 10000^(−2i/d)"}`, { fill: C.muted, size: 11 });
    b += line(x0, top + H, x0 + span, top + H, C.rule) + line(x0, top, x0, top + H, C.rule);
    for (let k = 0; k <= 4; k++) { const v = ymax * k / 4; b += text(x0 - 6, Y(v) + 4, fmt(v, 1), { size: 10, anchor: "end", fill: C.muted }); }
    for (let k = 0; k <= 4; k++) { const r = s.Rmax * k / 4; b += text(X(r), top + H + 14, fmt(r, 0), { size: 10, anchor: "middle", fill: C.muted }); }
    if (geo) b += path(poly(geo, Y), C.a, 1.2, 'opacity="0.3"');
    b += path(poly(cur, Y), C.a, 1.6);
    const off = s.r > s.Rmax, xr = X(Math.min(s.r, s.Rmax));
    b += line(xr, top, xr, top + H, C.b, 1, 'stroke-dasharray="3 3"') + (off ? "" : circle(xr, Y(at.bound), 4, C.b, C.b));
    b += text(Math.min(xr + 6, W - 150), top + 12, off ? `m − n = ${s.r} is off this axis` : `m − n = ${s.r}: ${fmt(at.bound, 2)}`, { fill: C.b, size: 11 });
    let h = top + H + 24;
    if (s.aligned) {
      const H2 = 130, t2 = h + 18, sc = xs.map(r => decayBound({ d: s.d, r, equal: s.equal, r0: s.r0 }).score), half = s.d / 2;
      const Y2 = v => t2 + H2 / 2 - (H2 / 2) * v / half;
      b += text(8, h + 10, `one constructed pair aligned to r0 = ${s.r0}: score Re Σ h_i e^{i(m−n)θ_i} with |h_i| = 1 (range ±d/2 = ±${half})`, { fill: C.muted, size: 11 });
      b += line(x0, Y2(0), x0 + span, Y2(0), C.rule) + text(x0 - 6, Y2(half) + 4, `${half}`, { size: 10, anchor: "end", fill: C.muted }) + text(x0 - 6, Y2(-half) + 4, `−${half}`, { size: 10, anchor: "end", fill: C.muted });
      b += path(poly(sc, Y2), C.ok, 1.4);
      if (s.r0 <= s.Rmax) b += line(X(s.r0), t2, X(s.r0), t2 + H2, C.ok, 1, 'stroke-dasharray="3 3"');
      if (!off) b += circle(xr, Y2(at.score), 4, C.b, C.b);
      h = t2 + H2 + 8;
    }
    pic.innerHTML = svg(W, h, b);
    const trend = s.equal ? `<b>flat</b>: with one θ every S_j = j·e^{i(m−n)θ}, so |S_j| = j at every distance` : `<b>${fmt(at.rel * 100, 1)}%</b> of its distance-0 value (${fmt(at.bound0, 1)} = (d/2 + 1)/2)`;
    read.innerHTML = `relative bound at m − n = ${s.r}: <span class="big">${fmt(at.bound, 3)}</span> · ${trend}<br>
      ${s.aligned ? `aligned pair: score at m − n = ${s.r} is <b>${fmt(at.score, 2)}</b> (peak ${at.scoreMax} at r0 = ${s.r0}); its full Eq. 37 bound is (max|h_{i+1} − h_i| = ${fmt(at.content, 3)}) × Σ|S_{i+1}| (= ${fmt(at.sumAbs, 1)}) = ${fmt(at.fullBound, 1)} ${at.boundHolds ? `<span style="color:${C.ok}">≥ |score|</span>` : `<span style="color:${C.b}">violated</span>`}. Only the second factor depends on m − n; a decaying bound does not make this score decay.<br>` : ""}
      <span class="muted small">provenance: fixture:rope--decay-bound · rope:eq35-37 and rope:fig2 (p8). Fig. 2 does not state d; d = 128 matches its scale (inference). θ_i = 10000^(−2i/d), i = 0..d/2−1 (§3.3, p5). The aligned pair is constructed, not from the paper.</span>`;
  };
  root.append(el("div", { class: "widget" }, el("div", { class: "controls" },
    slider("head dimension d", 2, 256, s.d, 2, v => { s.d = v; draw(); }),
    slider("relative distance m − n", 0, 2048, s.r, 1, v => { s.r = v; draw(); }),
    select("axis", [["256", "0-256 (as Fig. 2)"], ["2048", "0-2048"]], String(s.Rmax), v => { s.Rmax = +v; draw(); }),
    check("all θ_i equal (one frequency)", s.equal, v => { s.equal = v; draw(); }),
    check("show one aligned q, k pair", s.aligned, v => { s.aligned = v; draw(); }), r0Slider,
    read), pic)); draw();
};

// ======================================================== 4. linear attention ==
// Eq. 18 (p5): Attention_m = Σ_n φ(q_m)^T φ(k_n) v_n / Σ_n φ(q_m)^T φ(k_n), with non-negative feature maps.
// Eq. 19 (p5-6): RoPE rotates the feature maps in the numerator only, (R_m φ(q_m))^T (R_n φ(k_n)), and keeps the
// unrotated denominator "to avoid the risk of dividing zero"; the numerator terms can then be negative (§3.3, p6).
// Effective weight of key n = rotated numerator term / shared unrotated denominator. 2D, so R_p rotates by p·θ.
// Feature vectors and θ = π/3 are the KP rope-linear-attention check's; nothing else.
function linearAttention({ theta, m, n1 = 0, n2 = 1, q = [1, 0], keys = [[1, 0], [0.6, 0.8]] }) {
  const th = theta * DEG, ns = [n1, n2];
  const unrot = keys.map(k => dot(q, k));                                   // φ(q)^T φ(k_n) ≥ 0
  const num = keys.map((k, j) => dot(rot(q, m * th), rot(k, ns[j] * th)));   // (R_m φ(q))^T (R_n φ(k_n)) = φ(q)^T R_{(n−m)θ} φ(k_n)
  const den = unrot.reduce((a, x) => a + x, 0), numSum = num.reduce((a, x) => a + x, 0);
  const w = num.map(x => x / den);
  return { unrot, num, den, w, w1: w[0], w2: w[1], sum: numSum / den, minW: Math.min(...w), negative: w.some(x => x < -1e-12) ? 1 : 0,
    rotDen: numSum };                                                        // a rotated denominator: can be ≤ 0
}
MODELS["fixture:rope--linear-attention"] = {
  fn: linearAttention,
  cases: [
    { args: { theta: 60, m: 1 }, pick: "sum", expect: 0.6875, tol: 0.03, from: "rope:rope-linear-attention:check" },
    { args: { theta: 60, m: 1 }, pick: "w1", expect: 0.3125 },                // 0.5 / 1.6
    { args: { theta: 60, m: 1 }, pick: "w2", expect: 0.375 },                 // 0.6 / 1.6
    { args: { theta: 60, m: 1 }, pick: "den", expect: 1.6 },                  // unrotated denominator 1 + 0.6
    { args: { theta: 0, m: 1 }, pick: "sum", expect: 1 },                     // edge: no rotation -> ordinary normalized weights
    { args: { theta: 60, m: 3 }, pick: "w1", expect: -0.625 },                // predict: rotated term at (n − m)θ = −180° -> negative weight
    { args: { theta: 60, m: 3 }, pick: "negative", expect: 1 },
    { args: { theta: 60, m: 0, n1: 0, n2: 0 }, pick: "sum", expect: 1 },      // all tokens at one position: rotations cancel
    { args: { theta: 60, m: 3 }, pick: "rotDen", expect: -0.6072 },           // a rotated denominator would change sign (pass through 0)
  ],
};
WIDGETS["fixture:rope--linear-attention"] = (root) => {
  const s = { theta: 30, m: 2, n1: 0, n2: 1 };
  const pic = el("div"), read = el("div", { class: "readout" });
  const draw = () => {
    const mo = linearAttention(s), W = 660, x0 = 200, mid = 420, half = 200, top = 34, row = 46;
    const X = v => mid + half * Math.max(-1.2, Math.min(1.2, v)) / 1.2;       // weights in [−1.2, 1.2]
    let b = text(8, 16, "per-key weight · faint: unrotated φ(q)ᵀφ(k_n) / denominator · bold: Eq. 19's rotated term / the same denominator", { fill: C.muted, size: 11 });
    b += line(X(0), top - 6, X(0), top + 3 * row - 6, C.ink, 1) + line(X(1), top - 6, X(1), top + 3 * row - 6, C.rule, 1, 'stroke-dasharray="3 3"');
    b += text(X(1), top + 3 * row + 6, "1", { size: 10, anchor: "middle", fill: C.muted }) + text(X(0), top + 3 * row + 6, "0", { size: 10, anchor: "middle", fill: C.muted }) + text(X(-1), top + 3 * row + 6, "−1", { size: 10, anchor: "middle", fill: C.muted });
    const bar = (y, v, col, op = 1) => { const a = X(Math.min(0, v)), c2 = X(Math.max(0, v)); return `<rect x="${a.toFixed(1)}" y="${y}" width="${Math.max(1, c2 - a).toFixed(1)}" height="14" style="fill:${col}" opacity="${op}"/>`; };
    const labels = [`key 1 φ(k) = (1, 0) at n = ${s.n1}`, `key 2 φ(k) = (0.6, 0.8) at n = ${s.n2}`, "sum over keys"];
    const unW = mo.unrot.map(x => x / mo.den), vals = [[unW[0], mo.w1], [unW[1], mo.w2], [1, mo.sum]];
    vals.forEach(([u, v], j) => {
      const y = top + j * row;
      b += text(8, y + 12, labels[j], { size: 12 }) + text(8, y + 26, j < 2 ? `rotation (n − m)θ = ${((j ? s.n2 : s.n1) - s.m) * s.theta}°` : "Eq. 18 would give 1", { size: 10, fill: C.muted });
      b += bar(y, u, C.a, 0.3) + bar(y + 16, v, v < 0 ? C.b : C.a);
      b += text(Math.min(X(Math.max(0, v)) + 6, W - 50), y + 28, fmt(v, 3), { size: 11, fill: v < 0 ? C.b : C.ink });
    });
    // strip: sum of effective weights and a hypothetical rotated denominator, as the query position m moves 0..8
    const t2 = top + 3 * row + 24, H2 = 90, Ym = v => t2 + H2 / 2 - (H2 / 2) * Math.max(-2, Math.min(2, v)) / 2, Xm = p => x0 + (W - x0 - 30) * p / 8;
    b += text(8, t2 + 10, "as the query position m moves:", { size: 11, fill: C.muted }) + text(8, t2 + 26, "● sum of weights", { size: 11, fill: C.a }) + text(8, t2 + 42, "○ a rotated denominator", { size: 11, fill: C.b });
    b += line(x0, Ym(0), W - 30, Ym(0), C.ink, 1) + line(x0, Ym(1), W - 30, Ym(1), C.rule, 1, 'stroke-dasharray="3 3"') + text(x0 - 6, Ym(1) + 4, "1", { size: 10, anchor: "end", fill: C.muted }) + text(x0 - 6, Ym(0) + 4, "0", { size: 10, anchor: "end", fill: C.muted });
    for (let p = 0; p <= 8; p++) {
      const o = linearAttention({ ...s, m: p });
      b += circle(Xm(p), Ym(o.sum), p === s.m ? 5 : 3.5, C.a, C.a) + circle(Xm(p), Ym(o.rotDen), p === s.m ? 5 : 3.5, C.b, "#fff");
      b += text(Xm(p), t2 + H2 + 12, `m=${p}`, { size: 10, anchor: "middle", fill: p === s.m ? C.ink : C.muted });
    }
    pic.innerHTML = svg(W, t2 + H2 + 20, b);
    const regime = mo.negative ? `<span style="color:${C.b}">a negative weight</span>: the rotated numerator term of key 1 or 2 is below 0, which Eq. 18's non-negative φ never allows`
      : Math.abs(mo.sum - 1) < 1e-9 ? `<span style="color:${C.ok}">weights sum to 1</span> (no net rotation between query and keys)` : `all weights ≥ 0, but they sum to <b>${fmt(mo.sum, 4)}</b>, not 1`;
    read.innerHTML = `unrotated denominator Σ φ(q)ᵀφ(k_n) = <b>${fmt(mo.den, 4)}</b> (always > 0) · rotated numerator terms ${mo.num.map(x => fmt(x, 4)).join(", ")}<br>
      effective weights ${mo.w.map(x => fmt(x, 4)).join(", ")} · sum <span class="big">${fmt(mo.sum, 4)}</span> · ${regime}<br>
      if the denominator were rotated too it would be ${fmt(mo.rotDen, 4)}${mo.rotDen <= 0 ? ` <span style="color:${C.b}">(≤ 0: division by zero or a sign flip)</span>` : ""}<br>
      <span class="muted small">provenance: fixture:rope--linear-attention · rope:eq18-19 (p5-6): rotate φ(q_m), φ(k_n) in the numerator only, keep the unrotated denominator. φ(q) = (1, 0) and the two key features are the KP rope-linear-attention check's; 2D, R_p rotates by p·θ.</span>`;
  };
  root.append(el("div", { class: "widget" }, el("div", { class: "controls" },
    slider("θ (deg)", 0, 180, s.theta, 5, v => { s.theta = v; draw(); }),
    slider("query position m", 0, 8, s.m, 1, v => { s.m = v; draw(); }),
    slider("key 1 position n", 0, 8, s.n1, 1, v => { s.n1 = v; draw(); }),
    slider("key 2 position n", 0, 8, s.n2, 1, v => { s.n2 = v; draw(); }),
    read), pic)); draw();
};
