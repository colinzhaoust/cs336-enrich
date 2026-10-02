// Widgets for thread chinchilla (Hoffmann et al. 2022, "Training Compute-Optimal Large Language Models").
// Register as "fixture:<id>" -> (root, notice) => void. Each widget has a pure model in MODELS (no DOM) that
// tools/check_widgets.mjs tests against the KPs' stored answers.
// Source text: papers/chinchilla/paper.txt (our extraction of arXiv 2203.15556v1 HTML), cited as paper.txt:L<line>.
// Every constant below is printed in the paper; the only author choices are marked "author" (how printed rows are
// combined: row-anchored power laws, log-log interpolation between rows, and anchoring Kaplan's exponent at D.4's point).
import { el, fmt, slider } from "../widgets_lib.js";

export const MODELS = {};
export const WIDGETS = {};

const C = { ink: "var(--ink)", muted: "var(--muted)", rule: "var(--rule)", a: "var(--accent)", b: "var(--accent2)", ok: "var(--ok)", hi: "var(--hilite)" };
const text = (x, y, s, o = {}) => `<text x="${(+x).toFixed(1)}" y="${(+y).toFixed(1)}" style="fill:${o.fill || C.ink};font:${o.size || 12}px var(--sans)" text-anchor="${o.anchor || "start"}">${s}</text>`;
const rect = (x, y, w, h, fill, extra = "") => `<rect x="${(+x).toFixed(1)}" y="${(+y).toFixed(1)}" width="${Math.max(0, w).toFixed(1)}" height="${h}" style="fill:${fill}" ${extra}/>`;
const line = (x1, y1, x2, y2, stroke, extra = "") => `<line x1="${(+x1).toFixed(1)}" y1="${(+y1).toFixed(1)}" x2="${(+x2).toFixed(1)}" y2="${(+y2).toFixed(1)}" style="stroke:${stroke};${extra}"/>`;
const dot = (x, y, r, fill, extra = "") => `<circle cx="${(+x).toFixed(1)}" cy="${(+y).toFixed(1)}" r="${r}" style="fill:${fill};${extra}"/>`;
const path = (pts, stroke, extra = "") => pts.length < 2 ? "" : `<path d="${pts.map((p, i) => `${i ? "L" : "M"}${p[0].toFixed(1)},${p[1].toFixed(1)}`).join("")}" style="fill:none;stroke:${stroke};${extra}"/>`;
const svg = (w, h, body) => `<svg viewBox="0 0 ${w} ${h}" width="100%" style="max-width:${w}px;border:1px solid var(--rule);border-radius:6px;background:#fff" role="img">${body}</svg>`;
const clip = (id, x, y, w, h) => `<clipPath id="${id}"><rect x="${x}" y="${y}" width="${w}" height="${h}"/></clipPath>`;
const e10 = (v, d = 2) => Number.isFinite(v) ? v.toExponential(d).replace("e+", "e") : "–";
const big = (v) => !Number.isFinite(v) ? "–" : v >= 1e12 ? `${fmt(v / 1e12, 2)}T` : v >= 1e9 ? `${fmt(v / 1e9, v >= 1e11 ? 0 : 2)}B` : `${fmt(v / 1e6, 0)}M`;
let uid = 0;

// --------------------------------------------------------------- paper constants --
// Eq. 10 fit (paper.txt:L1172-L1178): L(N,D) = E + A/N^0.34 + B/D^0.28, E = 1.69, A = 406.4, B = 410.7.
const FIT = { E: 1.69, A: 406.4, B: 410.7, alpha: 0.34, beta: 0.28 };
// Table 2 exponents a, b of N_opt ∝ C^a, D_opt ∝ C^b (paper.txt:L387-L411).
const EXP = { 1: [0.50, 0.50], 2: [0.49, 0.51], 3: [0.46, 0.54], K: [0.73, 0.27] };
// Printed rows [N, FLOPs, tokens]. Approach 1 = Table 3 (paper.txt:L421-L474); Approaches 2 and 3 = Table A3
// (paper.txt:L1206-L1274). Table A3's Approach-3 row for 175 Billion prints 1.26e+24 FLOPs, which is below the 67B
// row's 1.71e+24 and inconsistent with its own 12.0 Trillion tokens (6 × 175e9 × 12.0e12 = 1.26e25); we use 1.26e25.
const ROWS = {
  1: [[400e6, 1.92e19, 8.0e9], [1e9, 1.21e20, 20.2e9], [10e9, 1.23e22, 205.1e9], [67e9, 5.76e23, 1.5e12], [175e9, 3.85e24, 3.7e12],
      [280e9, 9.90e24, 5.9e12], [520e9, 3.43e25, 11.0e12], [1e12, 1.27e26, 21.2e12], [10e12, 1.30e28, 216.2e12]],
  2: [[400e6, 1.84e19, 7.7e9], [1e9, 1.20e20, 20.0e9], [10e9, 1.32e22, 219.5e9], [67e9, 6.88e23, 1.7e12], [175e9, 4.54e24, 4.3e12],
      [280e9, 1.18e25, 7.1e12], [520e9, 4.19e25, 13.4e12], [1e12, 1.59e26, 26.5e12], [10e12, 1.75e28, 292.0e12]],
  3: [[400e6, 2.21e19, 9.2e9], [1e9, 1.62e20, 27.1e9], [10e9, 2.46e22, 410.1e9], [67e9, 1.71e24, 4.1e12], [175e9, 1.26e25 /* printed 1.26e+24 */, 12.0e12],
      [280e9, 3.52e25, 20.1e12], [520e9, 1.36e26, 43.5e12], [1e12, 5.65e26, 94.1e12], [10e12, 8.55e28, 1425.5e12]],
};
// Kaplan et al.'s optimum at 1e21 FLOPs as the paper quotes it in D.4 (paper.txt:L1289), scaled with Table 2's 0.73 (author: anchoring).
const KAPLAN = { C: 1e21, N: 4.68e9 };
// Table 1 (paper.txt:L129-L160): Gopher 280B on 300B tokens, Chinchilla 70B on 1.4T. Budgets below are 6ND.
const MODELS_T1 = { Gopher: [280e9, 300e9], Chinchilla: [70e9, 1.4e12] };
const GOPHER_C = 5.76e23;   // "Gopher unit", Fig. 2 caption (paper.txt:L270) and Table 3
const NAMES = { 1: "Approach 1 (Table 3)", 2: "Approach 2 (Table A3)", 3: "Approach 3 (Table A3)", K: "Kaplan et al. (Table 2's 0.73, anchored at D.4)" };

// ============================================================ 1. allocation table ==
// The three approaches' printed projections and Kaplan's exponent, at one budget C.
// "rule": scale the printed row at or below C by k = C / C_row with Table 2's exponents, N = N_row k^a, D = D_row k^b
//         (this is what the prompts do by hand). "table": log-log interpolation between the two printed rows that bracket C.
function interpRows(rows, C) {
  const lc = Math.log10(C);
  let i = rows.findIndex(r => r[1] > C); if (i === -1) i = rows.length - 1; if (i === 0) i = 1;
  const [n0, c0, d0] = rows[i - 1], [n1, c1, d1] = rows[i], t = (lc - Math.log10(c0)) / (Math.log10(c1) - Math.log10(c0));
  return { N: 10 ** (Math.log10(n0) + t * (Math.log10(n1) - Math.log10(n0))), D: 10 ** (Math.log10(d0) + t * (Math.log10(d1) - Math.log10(d0))) };
}
function allocation({ approach = 1, row, k, C }) {
  const [a, b] = EXP[approach];
  if (approach === "K") {
    const kk = C / KAPLAN.C, N = KAPLAN.N * kk ** a, D = (KAPLAN.C / (6 * KAPLAN.N)) * kk ** b;
    return { C, a, b, N, D, NB: N / 1e9, DB: D / 1e9, ratio: D / N, nFactor10: 10 ** a, dFactor10: 10 ** b, check6ND: 6 * N * D / C };
  }
  const rows = ROWS[approach];
  if (row === undefined) { row = 0; rows.forEach((r, i) => { if (r[1] <= C * (1 + 1e-9)) row = i; }); }
  const [Nr, Cr, Dr] = rows[row];
  if (C === undefined) C = Cr * k; else k = C / Cr;
  const N = Nr * k ** a, D = Dr * k ** b, t = interpRows(rows, C);
  return { C, k, row, a, b, N, D, NB: N / 1e9, DB: D / 1e9, ratio: D / N, Ntable: t.N, Dtable: t.D, ratioTable: t.D / t.N,
    nFactor10: 10 ** a, dFactor10: 10 ** b, ratioPerDecade: 10 ** (b - a), check6ND: 6 * N * D / C, gopherUnits: C / GOPHER_C };
}
MODELS["fixture:chinchilla--allocation-table"] = {
  fn: allocation,
  cases: [
    { args: { approach: 1, row: 2, k: 9 }, pick: "D", expect: 615.3e9, tol: 0.05, from: "chinchilla:equal-scaling-rule:check" },
    { args: { approach: 1, row: 2, k: 10 }, pick: "nFactor10", expect: 3.162, tol: 0.05, from: "chinchilla:supp-power-law-fitting:predict" },
    { args: { approach: 1, row: 3, k: 1 }, pick: "DB", expect: 1500 },                   // equal-scaling predict: 67B -> 1.5T at 1 Gopher unit
    { args: { approach: 1, row: 3, k: 1 }, pick: "gopherUnits", expect: 1 },             // ... exactly 1 Gopher unit
    { args: { approach: 1, row: 3, k: 4 }, pick: "NB", expect: 134 },                     // equal-scaling transfer: 4x Gopher -> 134B
    { args: { approach: 1, row: 3, k: 4 }, pick: "D", expect: 3.0e12 },                   // ... on 3T tokens
    { args: { approach: 3, row: 3, k: 4 }, pick: "NB", expect: 67 * 4 ** 0.46 },          // ... Approach 3: N x1.89
    { args: { approach: 1, C: 1e21 }, pick: "NB", expect: 2.86, tol: 0.02 },              // why-kaplan predict: Approach 1 at 1e21 (D.4 prints 2.86B)
    { args: { approach: 1, C: 1e21 }, pick: "DB", expect: 58, tol: 0.02 },                // ... about 58B tokens
    { args: { approach: "K", C: 1e21 }, pick: "NB", expect: 4.68 },                       // why-kaplan predict: Kaplan at 1e21
    { args: { approach: "K", C: 1e21 }, pick: "DB", expect: 35.6, tol: 0.02 },            // ... about 36B tokens
    { args: { approach: 1, C: 1e21 }, pick: "Ntable", expect: 2.86e9, tol: 0.02 },        // the interpolated Table 3 agrees with D.4
    { args: { approach: 1, row: 1, k: 1 }, pick: "ratio", expect: 20.2 },                 // Table 3: 1B -> 20.2 tokens per parameter
    { args: { approach: 1, row: 8, k: 1 }, pick: "ratio", expect: 21.62 },                // ... and still 21.6 at 10T: flat
    { args: { approach: 3, row: 0, k: 1 }, pick: "ratio", expect: 23 },                   // Approach 3: 23 at 400M ...
    { args: { approach: 3, row: 8, k: 1 }, pick: "ratio", expect: 142.55 },               // ... 142.6 at 10T: rising
    { args: { approach: 3, C: GOPHER_C }, pick: "Ntable", expect: 40e9, tol: 0.05 },      // Fig. 4 caption: Approach 3 projects 40B at Gopher's budget
    { args: { approach: 3, row: 2, k: 1 }, pick: "ratioPerDecade", expect: 10 ** 0.08 },  // D/N grows x1.20 per decade of C when b - a = 0.08
    { args: { approach: "K", C: 1e21 }, pick: "nFactor10", expect: 5.37 },                // Kaplan: 10x compute -> 5.4x model (paper rounds to 5.5x)
  ],
};
WIDGETS["fixture:chinchilla--allocation-table"] = (root) => {
  const KS = [0.1, 0.25, 0.5, 1, 2, 3, 4, 9, 10, 16, 100];
  const s = { ap: 1, row: 1, ki: 4 };                                         // default: Table 3's 1B row x 2 (no prompt's case)
  const pic = el("div"), read = el("div", { class: "readout" }), id = `chc${++uid}`;
  const rowSlider = slider("printed row (parameters)", 0, 8, s.row, 1, v => { s.row = v; draw(); }, v => big(ROWS[1][v][0]));
  const draw = () => {
    const m = allocation({ approach: s.ap, row: s.row, k: KS[s.ki] }), kap = allocation({ approach: "K", C: m.C });
    const W = 640, px = 64, pw = 548, H1 = 128, top1 = 26, top2 = top1 + H1 + 30, H2 = 128;
    const X = lc => px + pw * (lc - 18.5) / 10.5;                                   // log10 C in [18.5, 29]
    const Y1 = ln => top1 + H1 * (14 - ln) / 6;                                     // log10 N in [8, 14]
    const Y2 = lr => top2 + H2 * (2.6 - lr) / 2.6;                                  // log10 (D/N) in [0, 2.6]
    let b = clip(id + "a", px, top1, pw, H1) + clip(id + "b", px, top2, pw, H2);
    b += text(px, 16, "printed projections · blue A1 (Table 3) · green A2 · orange A3 (Table A3) · grey dashed Kaplan 0.73", { fill: C.muted, size: 11 });
    for (const [t0, h] of [[top1, H1], [top2, H2]]) b += rect(px, t0, pw, h, "none", `stroke="${"#d9d3c7"}"`);
    for (let lc = 19; lc <= 29; lc++) b += text(X(lc), top2 + H2 + 14, `1e${lc}`, { fill: C.muted, size: 10, anchor: "middle" });
    for (let ln = 8; ln <= 14; ln += 2) b += text(px - 6, Y1(ln) + 4, big(10 ** ln), { fill: C.muted, size: 10, anchor: "end" });
    for (const r of [1, 10, 100]) b += text(px - 6, Y2(Math.log10(r)) + 4, `${r}`, { fill: C.muted, size: 10, anchor: "end" });
    b += text(px + 4, top1 + 12, "optimal parameters N (slope = a)", { fill: C.muted, size: 11 });
    b += text(px + 4, top2 + 12, "tokens per parameter D/N (slope = b − a)", { fill: C.muted, size: 11 });
    b += line(px, Y2(Math.log10(20)), px + pw, Y2(Math.log10(20)), C.muted, "stroke-dasharray:2 3;stroke-width:1");
    b += text(px + pw - 4, Y2(Math.log10(20)) + 13, "20 per parameter (Lecture 9 p54)", { fill: C.muted, size: 10, anchor: "end" });
    const col = { 1: C.a, 2: C.ok, 3: C.b };
    let g1 = "", g2 = "";
    for (const ap of [1, 2, 3]) {
      const w = ap === s.ap ? "stroke-width:2.5" : "stroke-width:1.2;opacity:0.7";
      g1 += path(ROWS[ap].map(r => [X(Math.log10(r[1])), Y1(Math.log10(r[0]))]), col[ap], w);
      g2 += path(ROWS[ap].map(r => [X(Math.log10(r[1])), Y2(Math.log10(r[2] / r[0]))]), col[ap], w);
      for (const r of ROWS[ap]) { g1 += dot(X(Math.log10(r[1])), Y1(Math.log10(r[0])), 2.5, col[ap]); g2 += dot(X(Math.log10(r[1])), Y2(Math.log10(r[2] / r[0])), 2.5, col[ap]); }
    }
    const kp = []; const kr = [];
    for (let lc = 18.5; lc <= 29.001; lc += 0.25) { const q = allocation({ approach: "K", C: 10 ** lc }); kp.push([X(lc), Y1(Math.log10(q.N))]); kr.push([X(lc), Y2(Math.log10(q.ratio))]); }
    g1 += path(kp, C.muted, "stroke-dasharray:5 3;stroke-width:1.5"); g2 += path(kr, C.muted, "stroke-dasharray:5 3;stroke-width:1.5");
    for (const [nm, [N, D]] of Object.entries(MODELS_T1)) {
      const lc = Math.log10(6 * N * D);
      g1 += dot(X(lc), Y1(Math.log10(N)), 4, "#fff", `stroke:${C.ink}`); g2 += dot(X(lc), Y2(Math.log10(D / N)), 4, "#fff", `stroke:${C.ink}`);
      g2 += text(X(lc) + 7, Y2(Math.log10(D / N)) + (nm === "Gopher" ? -4 : 14), nm, { size: 10 });
    }
    const lcNow = Math.log10(m.C);
    g1 += line(X(lcNow), top1, X(lcNow), top1 + H1, C.hi, "stroke-width:1.5") + dot(X(lcNow), Y1(Math.log10(m.N)), 5, C.hi, `stroke:${C.ink}`);
    g2 += line(X(lcNow), top2, X(lcNow), top2 + H2, C.hi, "stroke-width:1.5") + dot(X(lcNow), Y2(Math.log10(m.ratio)), 5, C.hi, `stroke:${C.ink}`);
    b += `<g clip-path="url(#${id}a)">${g1}</g><g clip-path="url(#${id}b)">${g2}</g>`;
    b += text(px + pw / 2, top2 + H2 + 28, "training FLOPs C (log scale)", { fill: C.muted, size: 11, anchor: "middle" });
    pic.innerHTML = svg(W, top2 + H2 + 34, b);
    const [Nr, Cr, Dr] = ROWS[s.ap][s.row], dpd = m.ratioPerDecade;
    const trend = Math.abs(dpd - 1) < 0.01 ? "flat: N and D grow by the same factor" : dpd > 1 ? `rising ×${fmt(dpd, 3)} per decade: D grows faster than N` : `falling ×${fmt(dpd, 3)} per decade: N grows faster than D`;
    read.innerHTML = `${NAMES[s.ap]}, printed row ${big(Nr)}: C_row = ${e10(Cr)}, D_row = ${big(Dr)}<br>
      budget C = ${KS[s.ki]} × C_row = <b>${e10(m.C, 3)}</b> FLOPs = ${fmt(m.gopherUnits, 3)} Gopher units<br>
      rule with Table 2's a = ${m.a.toFixed(2)}, b = ${m.b.toFixed(2)}: N = ${big(Nr)} × ${KS[s.ki]}^${m.a.toFixed(2)} = <span class="big">${big(m.N)}</span>
      D = ${big(Dr)} × ${KS[s.ki]}^${m.b.toFixed(2)} = <span class="big">${big(m.D)}</span> tokens · D/N = <b>${fmt(m.ratio, 1)}</b> · 6ND / C = ${fmt(m.check6ND, 3)} (the printed row itself gives ${fmt(6 * Nr * Dr / Cr, 3)}: its tokens are rounded)<br>
      printed rows interpolated at this C: N ≈ ${big(m.Ntable)}, D ≈ ${big(m.Dtable)}, D/N ≈ ${fmt(m.ratioTable, 1)}<br>
      ×10 budget → N ×${fmt(m.nFactor10, 2)}, D ×${fmt(m.dFactor10, 2)} · tokens per parameter: <b>${trend}</b><br>
      Kaplan at the same C: N = ${big(kap.N)}, D = ${big(kap.D)}, D/N = ${fmt(kap.ratio, 2)} (×10 budget → N ×${fmt(kap.nFactor10, 2)})<br>
      <span class="muted small">provenance: fixture:chinchilla--allocation-table · paper.txt Table 3 (L421-L474, Approach 1), Table A3 (L1206-L1274, Approaches 2 and 3; its 175B Approach-3 FLOPs print 1.26e+24, inconsistent with its 12.0T tokens, used here as 6ND = 1.26e25), Table 2 exponents (L387-L411), D.4 Kaplan optimum 4.68B at 1e21 (L1289), Table 1 Gopher and Chinchilla (L129-L160), Gopher unit 5.76e23 (L270). Author: the rule scales the printed row at or below C; Kaplan's line is Table 2's 0.73 through D.4's point; between rows the plot joins printed points.</span>`;
  };
  const apSlider = slider("approach", 1, 3, s.ap, 1, v => { s.ap = v; draw(); }, v => ["", "1 · min over training curves", "2 · IsoFLOP profiles", "3 · parametric fit"][v]);
  root.append(el("div", { class: "widget" }, el("div", { class: "controls" },
    apSlider, rowSlider,
    slider("budget = printed row's FLOPs ×", 0, KS.length - 1, s.ki, 1, v => { s.ki = v; draw(); }, v => `${KS[v]}`),
    read), pic)); draw();
};

// =========================================================== 2. IsoFLOP parabola ==
// Approach 2 (§3.2, paper.txt:L288-L306): at a fixed budget train several sizes with D = C/6N, fit a parabola to final
// loss against log N, read off its vertex. The paper does not print Fig. 3's measured points, so the samples here are the
// Eq. 10 surface L̂(N, C/6N) (stand-in, labelled). The paper's 9 budgets span 6e18 to 3e21 (L292-L294).
const Lhat = (N, D, f = FIT) => f.E + f.A / N ** f.alpha + f.B / D ** f.beta;
function fitParabola(pts) {
  const n = pts.length, mx = pts.reduce((q, p) => q + p[0], 0) / n;
  let s0 = n, s1 = 0, s2 = 0, s3 = 0, s4 = 0, t0 = 0, t1 = 0, t2 = 0;
  for (const [x0, y] of pts) { const x = x0 - mx; s1 += x; s2 += x * x; s3 += x ** 3; s4 += x ** 4; t0 += y; t1 += x * y; t2 += x * x * y; }
  const det3 = (m) => m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0]) + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
  const M = [[s4, s3, s2], [s3, s2, s1], [s2, s1, s0]], R = [t2, t1, t0], d = det3(M);
  const sub = (j) => det3(M.map((row, i) => row.map((v, c) => c === j ? R[i] : v)));
  const A2 = sub(0) / d, B1 = sub(1) / d, C0 = sub(2) / d;                    // y = A2 x² + B1 x + C0 in centred x
  return { A2, B1, C0, mx, at: (x0) => { const x = x0 - mx; return A2 * x * x + B1 * x + C0; }, vertexX: A2 > 0 ? mx - B1 / (2 * A2) : NaN };
}
function isoflop({ C, N, center, spread = 0.5, count = 5, points }) {
  const out = {};
  if (N !== undefined) { out.D = C / (6 * N); out.DB = out.D / 1e9; }
  if (!points && center !== undefined) points = Array.from({ length: count }, (_, i) => { const x = center + (i - (count - 1) / 2) * spread; return [x, Lhat(10 ** x, C / (6 * 10 ** x))]; });
  if (points) {
    const f = fitParabola(points), ys = points.map(p => p[1]), iMin = ys.indexOf(Math.min(...ys));
    Object.assign(out, { points, fit: f, vertexX: f.vertexX, vertexN: 10 ** f.vertexX, vertexB: 10 ** f.vertexX / 1e9, bracketed: iMin > 0 && iMin < points.length - 1 ? 1 : 0, valley: f.A2 > 0 ? 1 : 0 });
  }
  if (C !== undefined) {                                                        // the slice's true minimum: Eq. 4 with Eq. 10's constants
    const f = frontier({ C }); out.trueN = f.Nopt; out.trueX = Math.log10(f.Nopt);
    if (out.vertexX !== undefined) out.vertexOverTrue = out.vertexN / out.trueN;
  }
  return out;
}
MODELS["fixture:chinchilla--isoflop-parabola"] = {
  fn: isoflop,
  cases: [
    { args: { C: 1e21, N: 1e9 }, pick: "DB", expect: 166.67, tol: 0.03, from: "chinchilla:approach-2:predict" },
    { args: { points: [[9.0, 2.62], [9.5, 2.50], [10.0, 2.54]] }, pick: "vertexN", expect: 4.217e9, tol: 0.05, from: "chinchilla:approach-2:check" },
    { args: { C: 1e21, N: 2e9 }, pick: "DB", expect: 83.33 },                                         // doubling N halves D
    { args: { C: 1e20, center: Math.log10(frontier({ C: 1e20 }).Nopt), spread: 0.25, count: 5 }, pick: "vertexOverTrue", expect: 1, tol: 0.02 }, // bracketed, narrow: vertex on the valley
    { args: { C: 1e20, center: Math.log10(frontier({ C: 1e20 }).Nopt), spread: 0.25, count: 5 }, pick: "bracketed", expect: 1 },
    { args: { C: 1e20, center: 7.6, spread: 0.2, count: 5 }, pick: "bracketed", expect: 0 },         // approach-2 transfer: all sizes left of the valley, loss still falling
    { args: { C: 1e20, center: 7.6, spread: 0.2, count: 5 }, pick: "vertexX", expect: 8.5216, tol: 0.005 }, // ... vertex lands past the largest size tried (x = 8.0) yet short of the true valley (x = 8.81)
    { args: { C: 6e18, N: 1e8 }, pick: "DB", expect: 10 },                                             // smallest IsoFLOP budget
  ],
};
WIDGETS["fixture:chinchilla--isoflop-parabola"] = (root) => {
  const s = { logC: 19.5, center: 8.3, spread: 0.3, count: 5 };            // default: 3e19, sizes left of centre (not a prompt's case)
  const pic = el("div"), read = el("div", { class: "readout" }), id = `chi${++uid}`;
  const draw = () => {
    const Cb = 10 ** s.logC, m = isoflop({ C: Cb, center: s.center, spread: s.spread, count: s.count });
    const W = 640, H = 230, px = 54, py = 22, pw = 560, ph = 170, x0 = 7, x1 = 12;
    let lmin = Infinity; for (let x = x0; x <= x1; x += 0.02) lmin = Math.min(lmin, Lhat(10 ** x, Cb / (6 * 10 ** x)));
    const y0 = lmin - 0.03, y1 = lmin + 0.5;
    const X = x => px + pw * (x - x0) / (x1 - x0), Y = L => py + ph * (y1 - L) / (y1 - y0);
    let b = clip(id, px, py, pw, ph) + rect(px, py, pw, ph, "none", `stroke="#d9d3c7"`);
    b += text(px, 14, `C = ${e10(Cb)} · grey: L̂(N, C/6N) from Eq. 10 · dots: sampled sizes · dashed: fitted parabola`, { fill: C.muted, size: 11 });
    const curve = []; for (let x = x0; x <= x1 + 1e-9; x += 0.02) curve.push([X(x), Y(Lhat(10 ** x, Cb / (6 * 10 ** x)))]);
    let g = path(curve, C.muted, "stroke-width:1.5;opacity:0.6");
    const fp = []; for (let x = x0; x <= x1 + 1e-9; x += 0.02) fp.push([X(x), Y(m.fit.at(x))]);
    g += path(fp, C.a, "stroke-width:1.8;stroke-dasharray:6 4");
    for (const [x, L] of m.points) g += dot(X(x), Y(L), 4.5, C.a, `stroke:${C.ink}`);
    g += line(X(m.trueX), py, X(m.trueX), py + ph, C.ok, "stroke-width:1.5");
    if (m.valley && m.vertexX > x0 - 2 && m.vertexX < x1 + 2) g += line(X(m.vertexX), py, X(m.vertexX), py + ph, C.b, "stroke-width:1.5;stroke-dasharray:4 3");
    b += `<g clip-path="url(#${id})">${g}</g>`;
    b += text(X(m.trueX) + 4, py + 12, "true valley (Eq. 4)", { fill: C.ok, size: 10 });
    if (m.valley && m.vertexX > x0 && m.vertexX < x1) b += text(X(m.vertexX) - 4, py + 26, "parabola vertex", { fill: C.b, size: 10, anchor: "end" });
    for (let x = 7; x <= 12; x++) b += text(X(x), py + ph + 14, big(10 ** x), { fill: C.muted, size: 10, anchor: "middle" });
    for (const L of [y0 + 0.03, y0 + 0.28, y0 + 0.53]) b += text(px - 6, Y(L) + 4, L.toFixed(2), { fill: C.muted, size: 10, anchor: "end" });
    b += text(px + pw / 2, H - 6, "parameters N (log scale) · D = C / 6N falls to the right", { fill: C.muted, size: 11, anchor: "middle" });
    pic.innerHTML = svg(W, H, b);
    const regime = !m.valley ? "no valley: the fitted parabola opens downward, there is no minimum to read" : m.bracketed
      ? `bracketed: the lowest sampled loss has a higher neighbour on both sides; vertex = ${fmt(m.vertexOverTrue, 3)} × the true N_opt`
      : (() => { const lo = m.points.map(p => p[1]).indexOf(Math.min(...m.points.map(p => p[1]))) === 0, edge = 10 ** m.points[lo ? 0 : m.points.length - 1][0];
          return `not bracketed: the lowest loss is at the ${lo ? "smallest" : "largest"} size tried (${big(edge)}), so the vertex is an extrapolation ${lo ? "below" : "beyond"} it; here it is ${fmt(m.vertexOverTrue, 3)} × the true N_opt`; })();
    const inRange = Cb >= 6e18 * 0.999 && Cb <= 3e21 * 1.001;
    read.innerHTML = `<pre>${"N".padStart(8)} ${"D = C/6N".padStart(10)} ${"L̂".padStart(7)}\n${m.points.map(([x, L]) => `${big(10 ** x).padStart(8)} ${big(Cb / (6 * 10 ** x)).padStart(10)} ${L.toFixed(4).padStart(7)}`).join("\n")}</pre>
      parabola vertex N = <span class="big">${m.valley ? big(m.vertexN) : "–"}</span> · true valley of L̂ (Eq. 4) N_opt = ${big(m.trueN)}, D_opt = ${big(Cb / (6 * m.trueN))}<br>
      <b style="color:${m.bracketed && m.valley ? "var(--ok)" : "var(--bad)"}">${regime}</b><br>
      ${inRange ? "inside" : "outside"} the paper's IsoFLOP range (9 budgets, 6e18 to 3e21)<br>
      <span class="muted small">provenance: fixture:chinchilla--isoflop-parabola · §3.2 (paper.txt:L288-L306: 9 budgets 6e18-3e21, D set by size and budget, "a diverse enough set of model sizes to see a clear minimum", "We fit a parabola to each IsoFLOPs curve"); samples are Eq. 10's fitted L̂ (L1172-L1178), not the measured Fig. 3 losses, which the paper does not print; parabola in log10 N by least squares (author).</span>`;
  };
  root.append(el("div", { class: "widget" }, el("div", { class: "controls" },
    slider("budget C", 18, 22, s.logC, 0.05, v => { s.logC = v; draw(); }, v => e10(10 ** v)),
    slider("centre of sampled sizes", 7.5, 11.5, s.center, 0.05, v => { s.center = v; draw(); }, v => big(10 ** v)),
    slider("spacing (decades)", 0.1, 1.0, s.spread, 0.05, v => { s.spread = v; draw(); }, v => v.toFixed(2)),
    slider("number of sizes", 3, 9, s.count, 2, v => { s.count = v; draw(); }),
    read), pic)); draw();
};

// ========================================================= 3. parametric frontier ==
// Eq. 2 / Eq. 10 loss and its closed-form frontier Eq. 4 (paper.txt:L364): N_opt = G (C/6)^a, D_opt = G^-1 (C/6)^b,
// G = (αA/(βB))^(1/(α+β)), a = β/(α+β), b = α/(α+β). The constant 6 is the FLOPs ≈ 6ND constraint (L356-L357); k
// generalises it (the approach-3-frontier transfer asks for k = 12). With the rounded Eq. 10 constants a = 0.452
// (the paper prints 0.46, Table 2) and N_opt at Gopher's budget is 32B (Fig. 4 caption prints 40B, L377).
function frontier({ C, N, D, alpha = FIT.alpha, beta = FIT.beta, A = FIT.A, B = FIT.B, E = FIT.E, k = 6 }) {
  const a = beta / (alpha + beta), b = alpha / (alpha + beta), G = (alpha * A / (beta * B)) ** (1 / (alpha + beta));
  const o = { a, b, G };
  if (C !== undefined) {
    const Nopt = G * (C / k) ** a, Dopt = (C / k) ** b / G;
    Object.assign(o, { Nopt, Dopt, NoptB: Nopt / 1e9, ratioOpt: Dopt / Nopt, termNopt: A / Nopt ** alpha, termDopt: B / Dopt ** beta,
      Lopt: Lhat(Nopt, Dopt, { E, A, B, alpha, beta }), nOptVs6: (6 / k) ** a, ratioPerDecade: 10 ** (b - a) });
    o.termRatioOpt = o.termNopt / o.termDopt;
  }
  if (N !== undefined) {
    if (D === undefined) D = C / (k * N);
    Object.assign(o, { N, D, termN: A / N ** alpha, termD: B / D ** beta, L: Lhat(N, D, { E, A, B, alpha, beta }) });
    o.termRatio = o.termN / o.termD;
  }
  return o;
}
MODELS["fixture:chinchilla--parametric-frontier"] = {
  fn: frontier,
  cases: [
    { args: { C: 1e20 }, pick: "termRatioOpt", expect: 0.8235, tol: 0.04, from: "chinchilla:approach-3-frontier:check" },
    { args: { C: 1e24 }, pick: "termRatioOpt", expect: 0.28 / 0.34 },                   // the same ratio β/α at every budget
    { args: {}, pick: "a", expect: 0.4516 },                                              // approach-3-frontier predict: a ≈ 0.45 (paper prints 0.46)
    { args: {}, pick: "b", expect: 0.5484 },                                              // ... b ≈ 0.55 (paper prints 0.54)
    { args: { N: 7e10, D: 1.4e12 }, pick: "termN", expect: 0.0835, tol: 0.02 },           // approach-3-parametric-fit predict: A/N^α ≈ 0.08
    { args: { N: 7e10, D: 1.4e12 }, pick: "termD", expect: 0.1632, tol: 0.02 },           // ... B/D^β ≈ 0.16, the larger one
    { args: { N: 7e10, D: 1.4e12 }, pick: "L", expect: 1.937, tol: 0.01 },                // ... L̂ ≈ 1.94
    { args: { alpha: 0.28, beta: 0.34 }, pick: "a", expect: 0.5484 },                     // approach-3-parametric-fit transfer: swapped -> a > 0.5
    { args: { C: GOPHER_C, k: 12 }, pick: "nOptVs6", expect: 0.731, tol: 0.02 },           // approach-3-frontier transfer: C = 12ND -> N_opt x 2^-a
    { args: { C: 1e18 }, pick: "ratioOpt", expect: 25.67, tol: 0.02 },                     // D/N at 1e18 (isoflop-sweep fixture: ~26)
    { args: { C: 1e21 }, pick: "ratioOpt", expect: 50.08, tol: 0.02 },                     // ... at 1e21 (fixture's grid search: ~49)
    { args: { C: 1e21 }, pick: "ratioPerDecade", expect: 10 ** (0.06 / 0.62) },            // D/N ∝ C^(b−a) = C^((α−β)/(α+β))
    { args: { C: 1e18, alpha: 0.34, beta: 0.34 }, pick: "ratioPerDecade", expect: 1 },     // α = β: D/N flat
    { args: { C: GOPHER_C }, pick: "NoptB", expect: 32.19, tol: 0.02 },                    // rounded constants: 32B at Gopher's budget (Fig. 4 prints 40B)
    { args: { C: GOPHER_C }, pick: "Lopt", expect: 1.9307, tol: 0.01 },
  ],
};
WIDGETS["fixture:chinchilla--parametric-frontier"] = (root) => {
  const s = { logC: 20, logN: 8.3, alpha: FIT.alpha, beta: FIT.beta, k: 6 };   // default: 1e20, N left of the optimum
  const pic = el("div"), read = el("div", { class: "readout" }), id = `chp${++uid}`;
  const ctl = {};
  const draw = () => {
    const Cb = 10 ** s.logC, N = 10 ** s.logN, P = { alpha: s.alpha, beta: s.beta, k: s.k };
    const m = frontier({ C: Cb, N, ...P }), f = { ...FIT, alpha: s.alpha, beta: s.beta };
    const W = 640, px = 54, pw = 560, py = 22, ph = 140, x0 = 7, x1 = 13;
    let lmin = m.Lopt; const y0 = lmin - 0.03, y1 = lmin + 0.6;
    const X = x => px + pw * (x - x0) / (x1 - x0), Y = L => py + ph * (y1 - L) / (y1 - y0);
    let b = clip(id + "a", px, py, pw, ph) + rect(px, py, pw, ph, "none", `stroke="#d9d3c7"`);
    b += text(px, 14, `IsoFLOP slice of L̂ at C = ${e10(Cb)} (D = C / ${s.k}N) · green: Eq. 4 optimum · yellow: your N`, { fill: C.muted, size: 11 });
    const curve = []; for (let x = x0; x <= x1 + 1e-9; x += 0.02) curve.push([X(x), Y(Lhat(10 ** x, Cb / (s.k * 10 ** x), f))]);
    let g = path(curve, C.a, "stroke-width:2");
    g += line(X(Math.log10(m.Nopt)), py, X(Math.log10(m.Nopt)), py + ph, C.ok, "stroke-width:1.5") + dot(X(Math.log10(m.Nopt)), Y(m.Lopt), 4.5, C.ok);
    g += dot(X(s.logN), Y(m.L), 5.5, C.hi, `stroke:${C.ink}`);
    b += `<g clip-path="url(#${id}a)">${g}</g>`;
    for (let x = 7; x <= 13; x++) b += text(X(x), py + ph + 14, big(10 ** x), { fill: C.muted, size: 10, anchor: "middle" });
    b += text(px - 6, Y(y0 + 0.03) + 4, (y0 + 0.03).toFixed(2), { fill: C.muted, size: 10, anchor: "end" }) + text(px - 6, Y(y1 - 0.05) + 4, (y1 - 0.05).toFixed(2), { fill: C.muted, size: 10, anchor: "end" });
    // the two correction terms (E is the same everywhere and left out)
    const ty = py + ph + 42, bx = 130, bw = 300, sc = Math.max(m.termN + m.termD, m.termNopt + m.termDopt);
    b += text(10, ty - 6, `correction terms above E = ${FIT.E} · blue A/N^α (finite parameters) · orange B/D^β (finite tokens)`, { fill: C.muted, size: 11 });
    [[`your N = ${big(N)}`, m.termN, m.termD], [`optimum ${big(m.Nopt)}`, m.termNopt, m.termDopt]].forEach(([lab, tn, td], i) => {
      const y = ty + i * 26;
      b += text(10, y + 14, lab, { size: 11 }) + rect(bx, y, bw * tn / sc, 18, C.a) + rect(bx + bw * tn / sc, y, bw * td / sc, 18, C.b, 'opacity="0.85"');
      b += text(bx + bw * (tn + td) / sc + 6, y + 14, `${tn.toFixed(3)} + ${td.toFixed(3)} · ratio ${(tn / td).toFixed(3)}`, { size: 11 });
    });
    // D_opt/N_opt vs C
    const qy = ty + 70, qh = 90, X2 = lc => px + pw * (lc - 18) / 8, Y2 = lr => qy + qh * (2.6 - lr) / 2.6;
    b += clip(id + "c", px, qy, pw, qh) + rect(px, qy, pw, qh, "none", `stroke="#d9d3c7"`) + text(px + 4, qy + 12, `D_opt / N_opt along the frontier ∝ C^(b − a) = C^${(m.b - m.a).toFixed(3)}`, { fill: C.muted, size: 11 });
    const rp = []; for (let lc = 18; lc <= 26.001; lc += 0.1) rp.push([X2(lc), Y2(Math.log10(frontier({ C: 10 ** lc, ...P }).ratioOpt))]);
    let g2 = path(rp, C.b, "stroke-width:2") + line(px, Y2(Math.log10(20)), px + pw, Y2(Math.log10(20)), C.muted, "stroke-dasharray:2 3");
    for (const lc of [18, 21]) { const r = frontier({ C: 10 ** lc, ...P }).ratioOpt; g2 += dot(X2(lc), Y2(Math.log10(r)), 3.5, C.b) + text(X2(lc) + 5, Y2(Math.log10(r)) - 5, fmt(r, 1), { fill: C.b, size: 10 }); }
    g2 += dot(X2(s.logC), Y2(Math.log10(m.ratioOpt)), 5, C.hi, `stroke:${C.ink}`);
    b += `<g clip-path="url(#${id}c)">${g2}</g>` + text(px + pw - 4, Y2(Math.log10(20)) - 4, "20", { fill: C.muted, size: 10, anchor: "end" });
    for (const r of [1, 10, 100]) b += text(px - 6, Y2(Math.log10(r)) + 4, `${r}`, { fill: C.muted, size: 10, anchor: "end" });
    for (let lc = 18; lc <= 26; lc += 2) b += text(X2(lc), qy + qh + 14, `1e${lc}`, { fill: C.muted, size: 10, anchor: "middle" });
    b += text(px + pw / 2, qy + qh + 28, "training FLOPs C (log scale)", { fill: C.muted, size: 11, anchor: "middle" });
    pic.innerHTML = svg(W, qy + qh + 34, b);
    const side = Math.abs(s.logN - Math.log10(m.Nopt)) < 0.02 ? "at the optimum" : s.logN < Math.log10(m.Nopt) ? "left of the optimum: too few parameters, the A/N^α term dominates the excess" : "right of the optimum: too few tokens, the B/D^β term dominates the excess";
    const drift = Math.abs(s.alpha - s.beta) < 1e-9 ? "flat (α = β)" : s.alpha > s.beta ? "rising (α > β: the data term decays more slowly, so more of each new FLOP goes to D)" : "falling (α < β: more of each new FLOP goes to N)";
    read.innerHTML = `α = ${s.alpha.toFixed(2)}, β = ${s.beta.toFixed(2)} → a = β/(α+β) = <b>${m.a.toFixed(3)}</b>, b = α/(α+β) = <b>${m.b.toFixed(3)}</b>, G = ${m.G.toFixed(3)} · FLOPs ≈ ${s.k}ND<br>
      N_opt = G (C/${s.k})^a = <span class="big">${big(m.Nopt)}</span> D_opt = <span class="big">${big(m.Dopt)}</span> D/N = <b>${fmt(m.ratioOpt, 1)}</b>, L̂ = ${m.Lopt.toFixed(4)}<br>
      at the optimum (A/N^α) / (B/D^β) = ${m.termRatioOpt.toFixed(4)} = β/α at every budget${s.k !== 6 ? ` · N_opt is ${fmt(m.nOptVs6, 3)} × its value under 6ND` : ""}<br>
      your point: N = ${big(N)}, D = ${big(m.D)}, L̂ = ${m.L.toFixed(4)} (${m.L - m.Lopt < 5e-5 ? "at the optimum" : "+" + (m.L - m.Lopt).toFixed(4)}) · ${side}<br>
      tokens per parameter along the frontier: <b>${drift}</b>, ×${fmt(m.ratioPerDecade, 3)} per decade of C<br>
      <span class="muted small">provenance: fixture:chinchilla--parametric-frontier · Eq. 2 (paper.txt:L325), Eq. 10 constants E = 1.69, A = 406.4, B = 410.7, α = 0.34, β = 0.28 (L1174-L1178), Eq. 4 frontier under FLOPs ≈ 6ND (L356-L366), Table 2's printed 0.46/0.54 (L403-L406), Fig. 4 caption's 40B at the Gopher budget (L377), Table 1 for Chinchilla 70B / 1.4T and Gopher 280B / 300B (L129-L160). With the rounded constants a = 0.452 and N_opt(5.76e23) = 32B, not 0.46 and 40B. The k = 12 control is the transfer's hypothetical accounting, not the paper's.</span>`;
    for (const [key, sl] of Object.entries(ctl)) { const inp = sl.querySelector("input"), out = sl.querySelector(".readout"); if (+inp.value !== s[key]) { inp.value = s[key]; out.textContent = sl._fmt(s[key]); } }
  };
  const mk = (key, label, min, max, step, f) => { const sl = slider(label, min, max, s[key], step, v => { s[key] = v; draw(); }, f); sl._fmt = f; ctl[key] = sl; return sl; };
  const preset = (label, p) => el("button", { onclick: () => { Object.assign(s, typeof p === "function" ? p() : p); draw(); } }, label);
  root.append(el("div", { class: "widget" }, el("div", { class: "controls" },
    mk("logC", "budget C", 18, 26, 0.01, v => e10(10 ** v)),
    mk("logN", "your N on this IsoFLOP curve", 7, 13, 0.01, v => big(10 ** v)),
    mk("alpha", "α (parameter exponent)", 0.2, 0.6, 0.01, v => v.toFixed(2)),
    mk("beta", "β (data exponent)", 0.2, 0.6, 0.01, v => v.toFixed(2)),
    mk("k", "FLOPs ≈ k·N·D", 6, 12, 6, v => `k = ${v}${v === 6 ? " (paper)" : " (hypothetical)"}`),
    el("div", { style: "margin-top:8px" },
      preset("N = N_opt", () => ({ logN: Math.log10(frontier({ C: 10 ** s.logC, alpha: s.alpha, beta: s.beta, k: s.k }).Nopt) })), " ",
      preset("Chinchilla 70B / 1.4T", { logC: Math.log10(6 * 70e9 * 1.4e12), logN: Math.log10(70e9), k: 6 }), " ",
      preset("Gopher 280B / 300B", { logC: Math.log10(6 * 280e9 * 300e9), logN: Math.log10(280e9), k: 6 }), " ",
      el("button", { class: "ghost", onclick: () => { Object.assign(s, { alpha: FIT.alpha, beta: FIT.beta, k: 6 }); draw(); } }, "Eq. 10 exponents")),
    read), pic)); draw();
};
