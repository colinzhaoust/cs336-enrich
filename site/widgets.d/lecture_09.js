// Widgets for thread lecture_09 (CS336 L9, scaling laws basics). Register as "fixture:<id>" -> (root, notice) => void.
// Each widget has a pure model in MODELS (no DOM) that tools/check_widgets.mjs tests against the KPs' stored answers.
// Sources: lectures/lecture_09/lecture_09.txt (slide text layer, cited as pdf page + txt line),
//          lectures/lecture_09/transcript.json (cited as video M:SS), the papers listed in lectures/lecture_09/sources.json
//          (marked "paper"), and the KPs' own prompts and filtering notes (marked "author"). No constant here is new.
import { el, fmt, slider } from "../widgets_lib.js";

export const MODELS = {};
export const WIDGETS = {};

const C = { ink: "var(--ink)", muted: "var(--muted)", rule: "var(--rule)", a: "var(--accent)", b: "var(--accent2)", ok: "var(--ok)", hi: "var(--hilite)" };
const text = (x, y, s, opts = {}) => `<text x="${x.toFixed(1)}" y="${y.toFixed(1)}" style="fill:${opts.fill || C.ink};font:${opts.size || 12}px var(--sans)" text-anchor="${opts.anchor || "start"}">${s}</text>`;
const svg = (w, h, body) => `<svg viewBox="0 0 ${w} ${h}" width="100%" style="max-width:${w}px;border:1px solid var(--rule);border-radius:6px;background:#fff" role="img">${body}</svg>`;
const line = (x1, y1, x2, y2, stroke, extra = "") => `<line x1="${x1.toFixed(1)}" y1="${y1.toFixed(1)}" x2="${x2.toFixed(1)}" y2="${y2.toFixed(1)}" style="stroke:${stroke};${extra}"/>`;
const dot = (x, y, fill, r = 5) => `<circle cx="${x.toFixed(1)}" cy="${y.toFixed(1)}" r="${r}" style="fill:${fill};stroke:${C.ink}"/>`;
const path = (pts, stroke, extra = "") => pts.length < 2 ? "" : `<path d="${pts.map(([x, y], i) => `${i ? "L" : "M"}${x.toFixed(1)},${y.toFixed(1)}`).join("")}" style="fill:none;stroke:${stroke};${extra}"/>`;
const check = (label, value, onChange) => { const i = el("input", { type: "checkbox" }); i.checked = value; i.addEventListener("change", () => onChange(i.checked)); return el("label", {}, i, ` ${label}`); };
const sci = (x, d = 2) => Number.isFinite(x) ? x.toExponential(d).replace("e+", "e") : "–";
// A log-log frame: px,py,pw,ph in pixels; x and y given as log10 ranges. Returns mappers and the axis markup.
function logFrame({ px, py, pw, ph, x0, x1, y0, y1, xlab, ylab, xfmt = v => `1e${v}`, yfmt = v => `${fmt(10 ** v, 3)}` }) {
  const X = lx => px + pw * (lx - x0) / (x1 - x0), Y = ly => py + ph * (y1 - ly) / (y1 - y0);
  let b = `<rect x="${px}" y="${py}" width="${pw}" height="${ph}" style="fill:none;stroke:${C.rule}"/>`;
  for (let v = Math.ceil(x0); v <= Math.floor(x1); v++) b += line(X(v), py, X(v), py + ph, C.rule, "stroke-dasharray:2 3") + text(X(v), py + ph + 13, xfmt(v), { fill: C.muted, size: 10, anchor: "middle" });
  const ystep = y1 - y0 > 6 ? 2 : 1;
  for (let v = Math.ceil(y0); v <= Math.floor(y1); v += ystep) b += line(px, Y(v), px + pw, Y(v), C.rule, "stroke-dasharray:2 3") + text(px - 5, Y(v) + 4, yfmt(v), { fill: C.muted, size: 10, anchor: "end" });
  if (Math.floor(y1) - Math.ceil(y0) < 1) for (const v of [y0, y1]) b += text(px - 5, Y(v) + 4, yfmt(+v.toFixed(2)), { fill: C.muted, size: 10, anchor: "end" });
  b += text(px + pw / 2, py + ph + 27, xlab, { fill: C.muted, size: 11, anchor: "middle" }) + `<text transform="translate(${px - 40},${(py + ph / 2).toFixed(1)}) rotate(-90)" style="fill:${C.muted};font:11px var(--sans)" text-anchor="middle">${ylab}</text>`;
  const clipY = ly => Math.max(py, Math.min(py + ph, Y(ly)));
  return { X, Y, clipY, axes: b };
}

// ===================================================== 1. power law with a floor ==
// L(n) = E + A·(n/n0)^(−α), n measured in units of a reference size n0 where the reducible part equals A.
//   lecture_09.pdf:p15 (txt L137-L152): loss vs dataset size is linear on a log-log plot ("power law").
//   p17 (L166-L183): mean estimation E(μ̂−μ)² = σ²/n, log Error = −log n + 2 log σ, "any polynomial rate 1/n^α is a scaling law"
//     (so A plays σ², α = 1).
//   p19 (L200-L217): box-binning nonparametric estimator, Error = n^(−1/d), slope −1/d.
//   video 15:33-15:45 (a line "usually means that I'm very far away from my asymptote ... once I approach my asymptote, I'm going
//     to taper off"), 16:40-16:53 ("1 over n to the alpha plus some constant ... subtracting out the constant term"), 21:10-21:33
//     ("explicitly fit the asymptote and then you correct for it").
//   The local raw slope α·R/(E+R) and every number in the cases are the KPs' (author, irreducible-floor-bends-loglog filtering note).
function powerFloor({ E = 0, A, alpha, n = 1, k = 2 }) {
  const L = x => E + A * Math.pow(x, -alpha);
  const R = A * Math.pow(n, -alpha), l = E + R, l10 = L(10 * n);
  return {
    L: l, reducible: R,
    drop: L(1) / l,                                   // loss at n0 over loss at n
    decadeFactor: l10 / l,                            // loss multiplier for the next 10× of data
    rawSlope: alpha * R / (E + R),                    // −d log L / d log n at n (the raw, bent curve)
    rawSlope10: Math.log10(l / l10),                  // two-point raw slope between n and 10n
    fitAlpha10: Math.log10((l - E) / (l10 - E)),      // two-point slope of L − E between n and 10n (recovers α)
    naive: L(1) * Math.pow(n, -alpha),                // pure power law from n0, ignoring the floor
    dataFactor: Math.pow(k, 1 / alpha),               // data multiplier that cuts the reducible error k×
  };
}
MODELS["fixture:lecture_09--power-floor"] = {
  fn: powerFloor,
  cases: [
    { args: { E: 0, A: 4, alpha: Math.log10(2), n: 100 }, pick: "L", expect: 1.0, tol: 0.05, from: "lecture_09:scaling-law-definition:predict" },
    { args: { E: 0, A: 4, alpha: Math.log10(2), n: 10 }, pick: "L", expect: 2.0 },          // the predict's 10M-token point
    { args: { E: 0, A: 4, alpha: Math.log10(2), n: 10 }, pick: "decadeFactor", expect: 0.5 }, // same factor every decade
    { args: { E: 0, A: 1, alpha: 1, n: 100 }, pick: "L", expect: 0.01 },                     // σ = 1 mean estimation at n = 100
    { args: { E: 0, A: 9, alpha: 1, n: 100 }, pick: "decadeFactor", expect: 0.1 },           // σ = 3: same slope, offset ×9
    { args: { E: 0, A: 9, alpha: 1, n: 10000 }, pick: "L", expect: 0.0009 },                 // ... so 100× n cuts error 100× for both σ
    { args: { E: 0, A: 16, alpha: 0.5, k: 5 }, pick: "dataFactor", expect: 25, tol: 0.05, from: "lecture_09:power-law-from-estimation-rate:check" },
    { args: { E: 0, A: 1, alpha: 0.5, k: 10 }, pick: "dataFactor", expect: 100 },            // power-law transfer: 1/√n needs 100× for 10×
    { args: { E: 0, A: 1, alpha: 0.25, n: 16 }, pick: "drop", expect: 2, tol: 0.05, from: "lecture_09:nonparametric-dimension-exponent:predict" },
    { args: { E: 0, A: 1, alpha: 1 / 6, k: 2 }, pick: "dataFactor", expect: 64, tol: 0.05, from: "lecture_09:nonparametric-dimension-exponent:check" },
    { args: { E: 2, A: 2, alpha: 0.5, n: 100 }, pick: "L", expect: 2.2, tol: 0.05, from: "lecture_09:irreducible-floor-bends-loglog:predict" },
    { args: { E: 2, A: 2, alpha: 0.5, n: 100 }, pick: "naive", expect: 0.4 },                // the floor-ignoring misreading, below E
    { args: { E: 2, A: 1, alpha: Math.log10(5), n: 1 }, pick: "fitAlpha10", expect: 0.70, tol: 0.05, from: "lecture_09:irreducible-floor-bends-loglog:check" },
    { args: { E: 2, A: 1, alpha: Math.log10(5), n: 1 }, pick: "rawSlope10", expect: Math.log10(3 / 2.2) }, // raw fit 0.13
    { args: { E: 2, A: 2, alpha: 0.5, n: 1 }, pick: "rawSlope", expect: 0.25 },              // edge: reducible = floor -> slope α/2
    { args: { E: 0, A: 2, alpha: 0.5, n: 1000 }, pick: "rawSlope", expect: 0.5 },            // no floor: raw slope is α everywhere
  ],
};
WIDGETS["fixture:lecture_09--power-floor"] = (root) => {
  const s = { E: 1, A: 2, alpha: 0.4, logn: 1, k: 2, sub: false };
  const pic = el("div"), read = el("div", { class: "readout" });
  const ctl = {};
  const draw = () => {
    const n = 10 ** s.logn, m = powerFloor({ E: s.E, A: s.A, alpha: s.alpha, n, k: s.k });
    const L = x => s.E + s.A * x ** -s.alpha;
    const xs = []; for (let lx = -1; lx <= 4.0001; lx += 0.05) xs.push(lx);
    const ys = xs.map(lx => Math.log10(L(10 ** lx)));
    const naiveYs = xs.map(lx => Math.log10((s.E + s.A) * 10 ** (-s.alpha * lx)));
    const redYs = xs.map(lx => Math.log10(s.A) - s.alpha * lx);
    const all = [...ys, ...(s.sub ? redYs : naiveYs)];
    let y0 = Math.min(...all), y1 = Math.max(...all);
    y0 = Math.max(y0, y1 - 6); if (y1 - y0 < 0.6) { const c = (y0 + y1) / 2; y0 = c - 0.3; y1 = c + 0.3; }
    y0 -= 0.05; y1 += 0.1;
    const f = logFrame({ px: 56, py: 22, pw: 560, ph: 200, x0: -1, x1: 4, y0, y1, xlab: "data n / n₀ (log scale)", ylab: s.sub ? "L − E (log scale)" : "loss L (log scale)", xfmt: v => `${fmt(10 ** v, 1)}` });
    let b = f.axes;
    const P = arr => xs.map((lx, i) => [f.X(lx), f.clipY(arr[i])]).filter((_, i) => arr[i] >= y0 - 0.01 && arr[i] <= y1 + 0.01);
    if (s.E > 0 && !s.sub) { const ly = Math.log10(s.E); if (ly >= y0 && ly <= y1) b += line(56, f.Y(ly), 616, f.Y(ly), C.b, "stroke-dasharray:5 3") + text(612, f.Y(ly) - 4, `floor E = ${fmt(s.E, 2)}`, { fill: C.b, size: 11, anchor: "end" }); }
    if (!s.sub) b += path(P(naiveYs), C.muted, "stroke-dasharray:4 3;stroke-width:1.5");
    b += path(P(s.sub ? redYs : ys), C.a, "stroke-width:2.5");
    const yAt = lx => s.sub ? Math.log10(s.A) - s.alpha * lx : Math.log10(L(10 ** lx));
    const p1 = [f.X(s.logn), f.clipY(yAt(s.logn))];
    if (s.logn + 1 <= 4) { const p2 = [f.X(s.logn + 1), f.clipY(yAt(s.logn + 1))]; b += line(p1[0], p1[1], p2[0], p2[1], C.ink, "stroke-width:1.5") + dot(p2[0], p2[1], "#fff", 3.5); }
    b += dot(p1[0], p1[1], C.hi);
    b += text(60, 14, s.sub ? "blue: L − E, a straight line of slope −α" : "blue: L = E + A·(n/n₀)^−α · grey dashed: the same α with no floor · black: n to 10n", { fill: C.muted, size: 11 });
    pic.innerHTML = svg(640, 262, b);
    const ratio = s.E > 0 ? m.reducible / s.E : Infinity;
    const regime = s.E === 0 ? "no floor: a pure power law, the raw slope is α at every n"
      : ratio >= 10 ? `far from the floor (reducible part ${fmt(ratio, 1)}× E): the raw line is nearly straight`
      : `near the floor (reducible part ${fmt(ratio, 2)}× E): raw slope ${fmt(m.rawSlope, 3)} understates α = ${fmt(s.alpha, 3)}`;
    const d = 1 / s.alpha;
    read.innerHTML = `L(n) = ${fmt(s.E, 2)} + ${fmt(s.A, 2)}·(n/n₀)^−${fmt(s.alpha, 3)}${Math.abs(d - Math.round(d)) < 0.02 ? ` <span class="muted">(α = 1/d, d = ${Math.round(d)})</span>` : ""}<br>
      at n = ${fmt(n, 3)}·n₀: <span class="big">L = ${fmt(m.L, 4)}</span>
      reducible part A·(n/n₀)^−α = ${fmt(m.reducible, 4)} · L(n₀)/L(n) = ${fmt(m.drop, 3)}<br>
      next 10× of data multiplies L by ${fmt(m.decadeFactor, 3)}<br>
      slope between n and 10n: raw log L ${fmt(m.rawSlope10, 3)} · log(L − E) <b>${fmt(m.fitAlpha10, 3)}</b> (= α)<br>
      no-floor extrapolation from n₀: ${fmt(m.naive, 4)}${s.E > 0 && m.naive < s.E ? ` <b style="color:var(--bad)">below the floor</b>` : ""}<br>
      to cut the reducible error ${fmt(s.k, 1)}×: n must grow k^(1/α) = <b>${fmt(m.dataFactor, 4)}×</b><br>
      <b style="color:${s.E > 0 && ratio < 10 ? "var(--bad)" : "var(--ok)"}">${regime}</b><br>
      <span class="muted small">provenance: fixture:lecture_09--power-floor · lecture_09.pdf:p15 (L137-L152, linear on a log-log plot), p17 (L166-L183, σ²/n and "any polynomial rate 1/n^α"), p19 (L200-L217, n^(−1/d)); video 15:33-15:45 (a line means far from the asymptote), 16:40-16:53 (subtract the constant), 21:10-21:33 (fit the asymptote and correct for it). Local-slope formula α·R/(E+R) is the KP author's.</span>`;
  };
  const presets = [["mean estimation, α = 1 (p17)", 1], ["1/√n, α = 1/2", 0.5], ["boxes, d = 4", 0.25], ["d = 6", 1 / 6], ["d = 10 (slope −0.1)", 0.1]];
  const alphaSlider = slider("exponent α", 0.05, 1, s.alpha, 0.005, v => { s.alpha = v; draw(); }, v => fmt(v, 3));
  ctl.alpha = alphaSlider.querySelector("input"); ctl.alphaOut = alphaSlider.querySelector(".readout");
  root.append(el("div", { class: "widget" }, el("div", { class: "controls" },
    slider("floor E (irreducible loss)", 0, 3, s.E, 0.05, v => { s.E = v; draw(); }),
    slider("A: reducible part at n₀ (σ² in p17)", 0.1, 16, s.A, 0.1, v => { s.A = v; draw(); }),
    alphaSlider,
    el("div", { style: "margin-top:6px" }, ...presets.map(([name, a]) => el("button", { style: "margin:0 4px 4px 0;padding:3px 8px;font-size:12px", onclick: () => { s.alpha = a; ctl.alpha.value = a; ctl.alphaOut.textContent = fmt(a, 3); draw(); } }, name))),
    slider("data n / n₀", -1, 4, s.logn, 0.01, v => { s.logn = v; draw(); }, v => fmt(10 ** v, 3)),
    slider("cut the reducible error by k×", 1, 20, s.k, 0.5, v => { s.k = v; draw(); }),
    check("plot L − E (subtract the floor)", s.sub, v => { s.sub = v; draw(); }),
    read), pic)); draw();
};

// ======================================================= 2. two laws, one plot ==
// Two families on log-log axes: log10 L = a − b·log10(scale). Parallel lines (equal b) keep their gap; different slopes cross.
//   lecture_09.pdf:p22 (L241-L247): "Data composition affects the offset, not the slope." p23 (L253-L258): "just take the best small dataset".
//   p30 (L313-L322): Transformers vs LSTMs, "Scaling law way" (compare the curves at small scale). p32 (L329-L337): Adam vs SGD.
//   video 24:28-25:20 (best small-scale mix works because slopes are equal), 32:19-32:58 (LSTM: different intercept, maybe slope),
//   35:05-35:19 (SGD vs Adam: "the intercepts are different, but the slopes are very similar").
//   The lines in the cases are the KPs' own prompt numbers (author).
function twoLaws({ aA, bA, aB, bB, x }) {
  const yA = aA - bA * x, yB = aB - bB * x, crossLog = bA === bB ? NaN : (aB - aA) / (bB - bA);
  return { yA, yB, LA: 10 ** yA, LB: 10 ** yB, gap: yB - yA, ratio: 10 ** (yB - yA), gapPerDecade: bA - bB, crossLog, crossN: 10 ** crossLog, crossBillions: 10 ** crossLog / 1e9 };
}
MODELS["fixture:lecture_09--two-laws"] = {
  fn: twoLaws,
  cases: [
    { args: { aA: 0.8, bA: 0.05, aB: 1.25, bB: 0.10, x: 8 }, pick: "crossN", expect: 1e9, tol: 0.05, from: "lecture_09:architecture-optimizer-via-scaling:check" },
    { args: { aA: 0.8, bA: 0.05, aB: 1.25, bB: 0.10, x: 8 }, pick: "gap", expect: 0.05 },   // the check's "at 100M, B is higher"
    { args: { aA: 1.5, bA: 0.15, aB: 0.9, bB: 0.05, x: 7 }, pick: "gap", expect: 0.1 },     // architecture predict: 0.1 at 10M
    { args: { aA: 1.5, bA: 0.15, aB: 0.9, bB: 0.05, x: 8 }, pick: "gap", expect: 0.2 },     // ... 0.2 at 100M
    { args: { aA: 1.5, bA: 0.15, aB: 0.9, bB: 0.05, x: 9 }, pick: "gap", expect: 0.3 },     // ... widens to 0.3 at 1B
    { args: { aA: 1.0, bA: 0.1, aB: 1.1, bB: 0.1, x: 8 }, pick: "gap", expect: 0.1 },       // composition: offset 0.1 at 100M
    { args: { aA: 1.0, bA: 0.1, aB: 1.1, bB: 0.1, x: 10 }, pick: "gap", expect: 0.1 },      // ... and the same 0.1 at 10B
    { args: { aA: 2.0, bA: 0.3, aB: 0.2 - Math.log10(2) + 0.6, bB: 0.1, x: 9 }, pick: "ratio", expect: 1.98, tol: 0.02 }, // scaling-law-definition transfer: B ends 2× A
    { args: { aA: 2.0, bA: 0.3, aB: 0.2 - Math.log10(2) + 0.6, bB: 0.1, x: 6 }, pick: "ratio", expect: 0.5 },              // ... from half of A's loss at 1M
  ],
};
WIDGETS["fixture:lecture_09--two-laws"] = (root) => {
  const s = { yA: 0.6, bA: 0.08, yB: 0.5, bB: 0.04, x: 7 };                 // y* = log10 loss at scale 10^6
  const pic = el("div"), read = el("div", { class: "readout" });
  const inputs = {};
  const sl = (key, label, min, max, step, f) => { const w = slider(label, min, max, s[key], step, v => { s[key] = v; draw(); }, f); inputs[key] = [w.querySelector("input"), w.querySelector(".readout"), f]; return w; };
  const setAll = o => { Object.assign(s, o); for (const [k, [i, out, f]] of Object.entries(inputs)) { i.value = s[k]; out.textContent = f ? f(s[k]) : s[k]; } draw(); };
  const draw = () => {
    const aA = s.yA + 6 * s.bA, aB = s.yB + 6 * s.bB, m = twoLaws({ aA, bA: s.bA, aB, bB: s.bB, x: s.x });
    const yv = [6, 12].flatMap(x => [aA - s.bA * x, aB - s.bB * x]);
    let y0 = Math.min(...yv) - 0.1, y1 = Math.max(...yv) + 0.15; if (y1 - y0 < 0.5) { const c = (y0 + y1) / 2; y0 = c - 0.25; y1 = c + 0.25; }
    const f = logFrame({ px: 56, py: 22, pw: 560, ph: 200, x0: 6, x1: 12, y0, y1, xlab: "scale: parameters N or tokens n (log scale)", ylab: "loss (log scale)" });
    let b = f.axes + text(60, 14, "blue: option A · orange: option B · dashed: target scale", { fill: C.muted, size: 11 });
    b += path([[f.X(6), f.Y(aA - 6 * s.bA)], [f.X(12), f.Y(aA - 12 * s.bA)]], C.a, "stroke-width:2.5");
    b += path([[f.X(6), f.Y(aB - 6 * s.bB)], [f.X(12), f.Y(aB - 12 * s.bB)]], C.b, "stroke-width:2.5");
    b += line(f.X(s.x), 22, f.X(s.x), 222, C.muted, "stroke-dasharray:4 3");
    b += line(f.X(s.x), f.Y(m.yA), f.X(s.x), f.Y(m.yB), C.ink, "stroke-width:2");
    b += dot(f.X(s.x), f.Y(m.yA), C.a, 4) + dot(f.X(s.x), f.Y(m.yB), C.b, 4);
    const crossIn = Number.isFinite(m.crossLog) && m.crossLog >= 6 && m.crossLog <= 12;
    if (crossIn) b += dot(f.X(m.crossLog), f.Y(aA - s.bA * m.crossLog), C.hi, 6) + text(f.X(m.crossLog) + 8, f.Y(aA - s.bA * m.crossLog) - 8, `cross at 1e${fmt(m.crossLog, 2)}`, { size: 11 });
    pic.innerHTML = svg(640, 262, b);
    const win = m.gap > 0 ? "A" : m.gap < 0 ? "B" : "tie";
    const regime = !Number.isFinite(m.crossLog) ? "parallel: same slope, so the gap is the same at every scale and the small-scale ranking holds everywhere"
      : `slopes differ: the gap changes by ${fmt(Math.abs(m.gapPerDecade), 3)} per decade; the lines cross at 10^${fmt(m.crossLog, 2)} = ${sci(10 ** m.crossLog)}${crossIn ? "" : " (off the plot)"}, so a ranking at one scale does not hold at every scale`;
    read.innerHTML = `A: log10 L = ${fmt(aA, 3)} − ${fmt(s.bA, 3)}·log10 N · B: log10 L = ${fmt(aB, 3)} − ${fmt(s.bB, 3)}·log10 N<br>
      at scale 1e${fmt(s.x, 2)}: L_A = ${fmt(m.LA, 4)}, L_B = ${fmt(m.LB, 4)}<br>
      gap log10 L_B − log10 L_A = <span class="big">${fmt(m.gap, 3)}</span>
      (L_B / L_A = ${fmt(m.ratio, 3)}) · lower loss here: <b>${win}</b><br>
      <b style="color:${Number.isFinite(m.crossLog) ? "var(--bad)" : "var(--ok)"}">${regime}</b><br>
      <span class="muted small">provenance: fixture:lecture_09--two-laws · lecture_09.pdf:p22 (L241-L247, composition moves the offset, not the slope), p23 (L253-L258, take the best small dataset), p30 (L313-L322, the scaling-law way to compare architectures), p32 (L329-L337, Adam vs SGD); video 24:28-25:20, 32:19-32:58, 35:05-35:19 (SGD vs Adam: different intercepts, similar slopes). Lines are illustrative; the slides print no fitted values.</span>`;
  };
  const decade = v => `1e${fmt(v, 2)}`;
  root.append(el("div", { class: "widget" }, el("div", { class: "controls" },
    sl("yA", "A: log10 loss at 1e6", -0.5, 1.5, 0.01, v => fmt(v, 2)),
    sl("bA", "A: slope (loss falls 10^−b per decade)", 0, 0.4, 0.005, v => fmt(v, 3)),
    sl("yB", "B: log10 loss at 1e6", -0.5, 1.5, 0.01, v => fmt(v, 2)),
    sl("bB", "B: slope", 0, 0.4, 0.005, v => fmt(v, 3)),
    sl("x", "target scale", 6, 12, 0.05, decade),
    el("div", { style: "margin-top:6px" },
      el("button", { style: "margin:0 4px 4px 0;padding:3px 8px;font-size:12px", onclick: () => setAll({ bB: s.bA, yB: s.yA + 0.1 }) }, "B: same slope, +0.1 offset"),
      el("button", { style: "margin:0 4px 4px 0;padding:3px 8px;font-size:12px", onclick: () => setAll({ bB: s.bA + 0.06, yB: s.yA + 0.15 }) }, "B: steeper, starts worse")),
    read), pic)); draw();
};

// ============================================================== 3. joint law ==
// Rosenfeld et al. 2020 form, Error = n^(−α) + m^(−β) + C (lecture_09.pdf:p43, txt L430-L433; p44, L439-L445).
//   p43 (L427): "Clearly, lots of data is wasted on small models". video 57:32-58:01 (the smallest model's curve is flat,
//   "a complete waste of compute"), 58:52-59:21 (infinite data leaves a pure model-size law, and vice versa).
//   α = 0.5, β = 0.3, C = 0 are the KP's own prompt numbers (author); the slides print no fitted exponents.
//   The Kaplan form on p43 (L435-L436, [m^(−α) + n^(−1)]^β) is not modelled.
function jointLaw({ n, m, alpha, beta, C = 0 }) {
  const dn = n ** -alpha, dm = m ** -beta, err = dn + dm + C;
  return { dataTerm: dn, modelTerm: dm, err, floorAtM: dm + C, moreData: (100 * n) ** -alpha + dm + C, moreModel: dn + (100 * m) ** -beta + C, termRatio: dm / dn };
}
MODELS["fixture:lecture_09--joint-law"] = {
  fn: jointLaw,
  cases: [
    { args: { n: 1e4, m: 1e4, alpha: 0.5, beta: 0.3 }, pick: "err", expect: 0.0731 },            // the prompts' starting point
    { args: { n: 1e6, m: 1e4, alpha: 0.5, beta: 0.3 }, pick: "err", expect: 0.064, tol: 0.05, from: "lecture_09:joint-data-model-law:predict" },
    { args: { n: 1e4, m: 1e6, alpha: 0.5, beta: 0.3 }, pick: "err", expect: 0.026, tol: 0.05, from: "lecture_09:joint-data-model-law:transfer" },
    { args: { n: 1e4, m: 1e6, alpha: 0.5, beta: 0.3 }, pick: "termRatio", expect: 1.585, tol: 0.02 }, // ... terms now comparable
    { args: { n: 1e12, m: 1e4, alpha: 0.5, beta: 0.3 }, pick: "err", expect: 0.0631, tol: 0.01 },    // edge: n → ∞ leaves the model law
    { args: { n: 1e4, m: 1e4, alpha: 0.5, beta: 0.3 }, pick: "moreModel", expect: 0.0258, tol: 0.02 }, // the readout's "100× model"
  ],
};
WIDGETS["fixture:lecture_09--joint-law"] = (root) => {
  const s = { logn: 5, logm: 3, alpha: 0.5, beta: 0.3, C: 0 };
  const pic = el("div"), read = el("div", { class: "readout" });
  const draw = () => {
    const a = { n: 10 ** s.logn, m: 10 ** s.logm, alpha: s.alpha, beta: s.beta, C: s.C }, r = jointLaw(a);
    const xs = []; for (let lx = 2; lx <= 8.0001; lx += 0.05) xs.push(lx);
    const E = (lx, lm) => Math.log10(jointLaw({ ...a, n: 10 ** lx, m: 10 ** lm }).err);
    const y1 = Math.max(E(2, 2), E(2, s.logm)) + 0.1, y0 = Math.min(E(8, 8), E(8, s.logm)) - 0.15;
    const f = logFrame({ px: 56, py: 22, pw: 560, ph: 200, x0: 2, x1: 8, y0, y1, xlab: "data n (log scale)", ylab: "error (log scale)" });
    let b = f.axes + text(60, 14, `grey: m = 1e2 … 1e8 · blue: your m = 1e${fmt(s.logm, 1)} · dashed: its floor m^−β + C`, { fill: C.muted, size: 11 });
    for (let lm = 2; lm <= 8; lm++) { b += path(xs.map(lx => [f.X(lx), f.clipY(E(lx, lm))]), C.rule, "stroke-width:1.2"); b += text(f.X(8) - 2, f.clipY(E(8, lm)) - 3, `1e${lm}`, { fill: C.muted, size: 9, anchor: "end" }); }
    const fl = Math.log10(r.floorAtM);
    if (fl >= y0 && fl <= y1) b += line(56, f.Y(fl), 616, f.Y(fl), C.b, "stroke-dasharray:5 3");
    b += path(xs.map(lx => [f.X(lx), f.clipY(E(lx, s.logm))]), C.a, "stroke-width:2.5");
    b += dot(f.X(s.logn), f.clipY(Math.log10(r.err)), C.hi);
    pic.innerHTML = svg(640, 262, b);
    const lim = r.termRatio > 2 ? `model-limited: m^−β is ${fmt(r.termRatio, 2)}× n^−α, so more data is mostly wasted` : r.termRatio < 0.5 ? `data-limited: n^−α is ${fmt(1 / r.termRatio, 2)}× m^−β, so a bigger model is mostly wasted` : "balanced: the two terms are within 2×, so gains need both";
    read.innerHTML = `n = ${sci(a.n)}, m = ${sci(a.m)}<br>
      n^−α = ${fmt(r.dataTerm, 4)} · m^−β = ${fmt(r.modelTerm, 4)} · C = ${fmt(s.C, 3)}<br>
      Error = <span class="big">${fmt(r.err, 4)}</span>
      100× more data: ${fmt(r.moreData, 4)} (${fmt(100 * (1 - r.moreData / r.err), 1)}% lower) · 100× bigger model: ${fmt(r.moreModel, 4)} (${fmt(100 * (1 - r.moreModel / r.err), 1)}% lower)<br>
      <b style="color:${r.termRatio > 2 || r.termRatio < 0.5 ? "var(--bad)" : "var(--ok)"}">${lim}</b><br>
      <span class="muted small">provenance: fixture:lecture_09--joint-law · lecture_09.pdf:p43 (L424-L438, Rosenfeld+ 2020 Error = n^−α + m^−β + C; "lots of data is wasted on small models"), p44 (L439-L445, fit on small data and small models, predict the rest); video 57:32-58:01, 58:52-59:21. α = 0.5, β = 0.3 are the KP's example numbers; the deck prints no fitted exponents. The Kaplan bracket form on p43 is not modelled.</span>`;
  };
  root.append(el("div", { class: "widget" }, el("div", { class: "controls" },
    slider("data n", 2, 8, s.logn, 0.1, v => { s.logn = v; draw(); }, v => sci(10 ** v, 1)),
    slider("model size m", 2, 8, s.logm, 0.1, v => { s.logm = v; draw(); }, v => sci(10 ** v, 1)),
    slider("data exponent α", 0.05, 1, s.alpha, 0.05, v => { s.alpha = v; draw(); }, v => fmt(v, 2)),
    slider("model exponent β", 0.05, 1, s.beta, 0.05, v => { s.beta = v; draw(); }, v => fmt(v, 2)),
    slider("irreducible C", 0, 0.05, s.C, 0.001, v => { s.C = v; draw(); }, v => fmt(v, 3)),
    read), pic)); draw();
};

// ====================================================== 4. critical batch size ==
// For a target loss, steps S and examples E traded by the batch B along (S/S_min − 1)(E/E_min − 1) = 1 (lecture_09.pdf:p38,
// txt L369-L385, the curve is in the figure; McCandlish et al. 2018, paper). With B_crit = E_min/S_min (spoken 45:58):
// S = S_min(1 + B_crit/B), E = E_min(1 + B/B_crit) = S·B, and B = B_crit gives "roughly 2x the steps / passes optimal" (L383).
//   p37 (L361-L368): "strong diminishing returns past a certain point". video 42:34-43:59 (noise-limited, then bias-limited).
//   p39 (L386-L392): "The smaller the loss target, the bigger the batch". The fit B_crit(L) ≈ B*/L^(1/α_B), B* ≈ 2e8 tokens,
//   α_B ≈ 0.21, L in nats, is Kaplan et al. 2020 eq. 1.4 (paper; the slide shows only the figure). video 47:33-48:22.
const KB = { Bstar: 2e8, alphaB: 0.21 };
function criticalBatch({ Smin, Emin, B, L = 3, L2 = 3 }) {
  const Bc = Emin / Smin, S = Smin * (1 + Bc / B), E = Emin * (1 + B / Bc), S2 = Smin * (1 + Bc / (2 * B));
  return { Bc, S, E, Sx: S / Smin, Ex: E / Emin, doubleSaves: 1 - S2 / S, curve: (S / Smin - 1) * (E / Emin - 1),
    bcritTokens: KB.Bstar * L ** (-1 / KB.alphaB), bcritRatio: (L / L2) ** (1 / KB.alphaB) };
}
MODELS["fixture:lecture_09--critical-batch"] = {
  fn: criticalBatch,
  cases: [
    { args: { Smin: 1e4, Emin: 1e7, B: 1000 }, pick: "S", expect: 20000, tol: 0.05, from: "lecture_09:critical-batch-size:predict" },
    { args: { Smin: 1e4, Emin: 1e7, B: 1000 }, pick: "E", expect: 2e7 },          // ... and 20M examples
    { args: { Smin: 1e4, Emin: 1e7, B: 37 }, pick: "curve", expect: 1 },           // any batch stays on the curve
    { args: { Smin: 1e4, Emin: 1e7, B: 1000 }, pick: "doubleSaves", expect: 0.25 }, // at B_crit, doubling B saves 25% of steps
    { args: { Smin: 1e4, Emin: 1e7, B: 10 }, pick: "doubleSaves", expect: 0.495, tol: 0.01 },  // noise-limited: nearly halves
    { args: { Smin: 1e4, Emin: 1e7, B: 1e5 }, pick: "doubleSaves", expect: 0.00495, tol: 0.02 }, // bias-limited: almost nothing
    { args: { Smin: 1e4, Emin: 1e7, B: 1, L: 2.5, L2: 3.0 }, pick: "bcritRatio", expect: 0.419, tol: 0.02 }, // transfer: B_crit(3.0)/B_crit(2.5)
    { args: { Smin: 1e4, Emin: 1e7, B: 1, L: 3.0, L2: 2.5 }, pick: "bcritRatio", expect: 2.383, tol: 0.02 }, // run B (2.5) wants 2.4× the batch
  ],
};
WIDGETS["fixture:lecture_09--critical-batch"] = (root) => {
  const s = { logS: 3.5, logE: 7.5, logB: 2, L: 3.5 };
  const pic = el("div"), read = el("div", { class: "readout" });
  let bInput, bOut;
  const draw = () => {
    const Smin = 10 ** s.logS, Emin = 10 ** s.logE, B = 10 ** s.logB, m = criticalBatch({ Smin, Emin, B, L: s.L });
    // panel 1: steps vs examples, log-log, the curve swept over B in [Bc/1e3, Bc·1e3]
    const lBc = Math.log10(m.Bc);
    const lE0 = s.logE - 0.3, lE1 = s.logE + 3.2, lS0 = s.logS - 0.3, lS1 = s.logS + 3.2;
    const f = logFrame({ px: 50, py: 22, pw: 370, ph: 200, x0: lE0, x1: lE1, y0: lS0, y1: lS1, xlab: "examples E (log)", ylab: "steps S (log)", yfmt: v => `1e${v}` });
    let b = f.axes;
    b += line(f.X(lE0), f.Y(s.logS), f.X(lE1), f.Y(s.logS), C.b, "stroke-dasharray:5 3") + text(f.X(lE1) - 4, f.Y(s.logS) - 4, "S_min", { fill: C.b, size: 11, anchor: "end" });
    b += line(f.X(s.logE), f.Y(lS0), f.X(s.logE), f.Y(lS1), C.b, "stroke-dasharray:5 3") + text(f.X(s.logE) + 4, f.Y(lS1) + 12, "E_min", { fill: C.b, size: 11 });
    const pts = []; for (let lb = lBc - 3.5; lb <= lBc + 3.5; lb += 0.05) { const q = criticalBatch({ Smin, Emin, B: 10 ** lb }); const x = Math.log10(q.E), y = Math.log10(q.S); if (x <= lE1 && y <= lS1) pts.push([f.X(x), f.Y(y)]); }
    b += path(pts, C.a, "stroke-width:2.5");
    const crit = criticalBatch({ Smin, Emin, B: m.Bc });
    b += dot(f.X(Math.log10(crit.E)), f.Y(Math.log10(crit.S)), "#fff", 4) + text(f.X(Math.log10(crit.E)) + 7, f.Y(Math.log10(crit.S)) - 6, "B_crit (2×, 2×)", { size: 11 });
    const cx = Math.log10(m.E), cy = Math.log10(m.S);
    if (cx <= lE1 && cy <= lS1) b += dot(f.X(cx), f.Y(cy), C.hi);
    b += text(54, 14, "blue: S and E needed to reach the target, one point per batch size", { fill: C.muted, size: 11 });
    // panel 2: B_crit vs target loss (Kaplan fit), log-log
    const g = logFrame({ px: 470, py: 22, pw: 150, ph: 200, x0: Math.log10(2), x1: Math.log10(5), y0: 4, y1: 7.5, xlab: "target loss (nats)", ylab: "B_crit (tokens)", xfmt: () => "", yfmt: v => `1e${v}` });
    b += g.axes;
    for (const Lv of [2, 3, 4, 5]) b += text(g.X(Math.log10(Lv)), 22 + 213, `${Lv}`, { fill: C.muted, size: 10, anchor: "middle" });
    const kp = []; for (let Lv = 2; Lv <= 5.0001; Lv += 0.1) kp.push([g.X(Math.log10(Lv)), g.clipY(Math.log10(criticalBatch({ Smin, Emin, B, L: Lv }).bcritTokens))]);
    b += path(kp, C.ok, "stroke-width:2.5") + dot(g.X(Math.log10(s.L)), g.clipY(Math.log10(m.bcritTokens)), C.hi, 4);
    pic.innerHTML = svg(640, 262, b);
    const noise = B < m.Bc;
    read.innerHTML = `S_min = ${fmt(Smin, 0)} steps · E_min = ${sci(Emin)} examples · B_crit = E_min / S_min = <b>${fmt(m.Bc, 1)}</b><br>
      at batch B = ${fmt(B, 1)}: S = S_min(1 + B_crit/B) = ${fmt(m.S, 0)} (${fmt(m.Sx, 3)}× S_min) · E = E_min(1 + B/B_crit) = ${sci(m.E)} (${fmt(m.Ex, 3)}× E_min)<br>
      doubling B from here saves <span class="big">${fmt(100 * m.doubleSaves, 1)}% of steps</span>
      <b style="color:${noise ? "var(--ok)" : "var(--bad)"}">${noise ? "noise-limited (B < B_crit): extra examples mostly buy fewer steps (perfect scaling would save 50%)" : "bias-limited (B > B_crit): steps barely fall while examples keep growing"}</b><br>
      target loss ${fmt(s.L, 2)} nats → B_crit ≈ ${sci(m.bcritTokens)} tokens (Kaplan fit); lowering the target to ${fmt(s.L - 0.5, 2)} multiplies it by ${fmt((s.L / (s.L - 0.5)) ** (1 / KB.alphaB), 3)}<br>
      <span class="muted small">provenance: fixture:lecture_09--critical-batch · lecture_09.pdf:p37 (L361-L368, diminishing returns), p38 (L369-L385, steps vs examples curve, "roughly 2x", gradient-noise claim; curve (S/S_min−1)(E/E_min−1)=1 from McCandlish et al. 2018, figure only), p39 (L386-L392, smaller target → bigger batch); video 42:34-43:59 (noise- vs bias-limited), 45:48-46:26 (B_crit = E_min/S_min), 47:33-48:22. Right panel: B_crit(L) ≈ 2e8 / L^(1/0.21) tokens is Kaplan et al. 2020 eq. 1.4 (paper), not on the slides.</span>`;
  };
  const bw = slider("batch size B (examples per step)", 0, 6, s.logB, 0.05, v => { s.logB = v; draw(); }, v => fmt(10 ** v, 1));
  bInput = bw.querySelector("input"); bOut = bw.querySelector(".readout");
  root.append(el("div", { class: "widget" }, el("div", { class: "controls" },
    slider("S_min (steps, huge-batch limit)", 2, 6, s.logS, 0.1, v => { s.logS = v; draw(); }, v => fmt(10 ** v, 0)),
    slider("E_min (examples, tiny-batch limit)", 5, 10, s.logE, 0.1, v => { s.logE = v; draw(); }, v => sci(10 ** v, 1)),
    bw,
    el("button", { style: "margin-top:6px;padding:3px 8px;font-size:12px", onclick: () => { s.logB = s.logE - s.logS; bInput.value = s.logB; bOut.textContent = fmt(10 ** s.logB, 1); draw(); } }, "set B = B_crit"),
    slider("target loss L (right panel)", 2, 5, s.L, 0.05, v => { s.L = v; draw(); }, v => fmt(v, 2)),
    read), pic)); draw();
};

// ============================================== 5. Kaplan vs Chinchilla allocation ==
// Base point: the Chinchilla allocation at budget C, D = 20N and C = 6ND, so N = √(C/120) (lecture_09.pdf:p54, txt L507;
// C = 6ND is lecture_02:six-nd). Grow the budget k×:
//   Chinchilla: N and D each × k^0.5 (equal exponents, video 1:12:42-1:12:53 "the Chinchilla factor of 20").
//   Kaplan: N × k^0.73, D × k^0.27 from the same start (p45, txt L448 "N_opt = C^0.73, D_opt = C^0.27 (tokens per param
//   decreases w/ C)"; video 1:00:39-1:01:24).
//   Over-training: at budget kC pick r tokens per parameter, N = √(kC/(6r)), D = rN; per-token serving cost ∝ N
//   (p54, L501-L514: GPT-3 2, Chinchilla 20, LLaMA-65B 22, Llama 2 70B 29, Mistral 7B 110, Llama 3 70B 215 tokens/param;
//   video 1:14:25-1:15:14). Serving cost ∝ N (2N FLOPs per token) is the author's, after lecture_02's forward-pass count.
function allocation({ C, k = 1, r = 20 }) {
  const N0 = Math.sqrt(C / 120), D0 = 20 * N0, C1 = k * C;
  const chN = Math.sqrt(C1 / 120), chD = 20 * chN, kapN = N0 * k ** 0.73, kapD = D0 * k ** 0.27;
  const yourN = Math.sqrt(C1 / (6 * r)), yourD = r * yourN;
  return { N0, D0, chN, chD, chNfactor: chN / N0, kapN, kapD, kapNfactor: k ** 0.73, kapDfactor: k ** 0.27, kapRatio: kapD / kapN,
    kapFlops: 6 * kapN * kapD / C1, yourN, yourD, yourDT: yourD / 1e12, serveRatio: yourN / chN };
}
MODELS["fixture:lecture_09--allocation"] = {
  fn: allocation,
  cases: [
    { args: { C: 1.2e22 }, pick: "chN", expect: 1e10, tol: 0.02, from: "lecture_09:chinchilla-twenty-tokens-rule:predict" },
    { args: { C: 1.2e22 }, pick: "chD", expect: 2e11 },                                     // ... D = 200B
    { args: { C: 1.2e22, k: 25 }, pick: "chNfactor", expect: 5, tol: 0.05, from: "lecture_09:chinchilla-twenty-tokens-rule:check" },
    { args: { C: 1.2e22, k: 100 }, pick: "chN", expect: 1e11 },                             // chinchilla transfer: 100B
    { args: { C: 1.2e22, k: 100 }, pick: "kapN", expect: 2.88e11, tol: 0.02 },              // ... Kaplan ≈ 288B
    { args: { C: 1.2e22, k: 100 }, pick: "kapD", expect: 6.94e11, tol: 0.02 },              // ... on ≈ 694B tokens
    { args: { C: 1.2e22, k: 100 }, pick: "kapNfactor", expect: 29, tol: 0.05, from: "lecture_09:kaplan-compute-optimal-allocation:predict" },
    { args: { C: 1.2e22, k: 100 }, pick: "kapDfactor", expect: 3.47, tol: 0.02 },           // ... D ≈ 3.5×
    { args: { C: 1.2e20, k: 1000 }, pick: "kapRatio", expect: 0.83, tol: 0.1, from: "lecture_09:kaplan-compute-optimal-allocation:transfer" },
    { args: { C: 1.2e20, k: 1000 }, pick: "kapFlops", expect: 1 },                          // 0.73 + 0.27 = 1: the budget is spent exactly
    { args: { C: 6 * 70e9 * 215 * 70e9, r: 215 }, pick: "yourDT", expect: 15, tol: 0.05, from: "lecture_09:overtraining-for-inference:transfer" },
    { args: { C: 6 * 70e9 * 215 * 70e9, r: 215 }, pick: "yourN", expect: 70e9 },            // ... the 70B model
    { args: { C: 1.2e22, r: 80 }, pick: "serveRatio", expect: 0.5 },                         // overtraining predict: N*/2 is r = 80, half the serving cost
  ],
};
WIDGETS["fixture:lecture_09--allocation"] = (root) => {
  const ks = [1, 2, 4, 10, 25, 100, 1000];
  const s = { logC: 21, ki: 2, r: 20 };                                     // default: 1e21, budget ×4 (no prompt uses either)
  const pic = el("div"), read = el("div", { class: "readout" });
  let rIn, rOut;
  const draw = () => {
    const k = ks[s.ki], m = allocation({ C: 10 ** s.logC, k, r: s.r });
    const P = [[m.N0, m.D0], [m.chN, m.chD], [m.kapN, m.kapD], [m.yourN, m.yourD]].map(([n, d]) => [Math.log10(n), Math.log10(d)]);
    const xs = P.map(p => p[0]), ys = P.map(p => p[1]);
    const W = 560, H = 210, cxm = (Math.min(...xs) + Math.max(...xs)) / 2, cym = (Math.min(...ys) + Math.max(...ys)) / 2;
    const spanX = Math.max(...xs) - Math.min(...xs) + 1.4, spanY = Math.max(...ys) - Math.min(...ys) + 1.4;
    const ppd = Math.min(W / spanX, H / spanY);                                  // same pixels per decade on both axes
    const x0 = cxm - W / ppd / 2, x1 = cxm + W / ppd / 2, y0 = cym - H / ppd / 2, y1 = cym + H / ppd / 2;
    const f = logFrame({ px: 60, py: 22, pw: W, ph: H, x0, x1, y0, y1, xlab: "parameters N (log)", ylab: "tokens D (log)", yfmt: v => `1e${v}` });
    let b = f.axes;
    const clipLine = (fn, col, extra) => { const pts = []; for (let lx = x0; lx <= x1 + 1e-9; lx += (x1 - x0) / 60) { const ly = fn(lx); if (ly >= y0 && ly <= y1) pts.push([f.X(lx), f.Y(ly)]); } return path(pts, col, extra); };
    const iso = lc => lx => lc - Math.log10(6) - lx;
    b += clipLine(iso(s.logC), C.muted, "stroke-dasharray:4 3") + clipLine(iso(s.logC + Math.log10(k)), C.ink, "stroke-dasharray:4 3;stroke-width:1.5");
    b += clipLine(lx => lx + Math.log10(20), C.a, "stroke-width:1.5;opacity:0.6");
    b += clipLine(lx => P[0][1] + (0.27 / 0.73) * (lx - P[0][0]), C.b, "stroke-width:1.5;opacity:0.6");
    const lab = (p, col, t, dx = 8, dy = -7, anchor = "start") => dot(f.X(p[0]), f.Y(p[1]), col) + text(f.X(p[0]) + dx, f.Y(p[1]) + dy, t, { size: 11, fill: col === "#fff" ? C.ink : col, anchor });
    b += lab(P[0], "#fff", "start (C)", 8, 14);
    if (k > 1) b += lab(P[1], C.a, "Chinchilla", -8, -8, "end") + lab(P[2], C.b, "Kaplan");
    if (s.r !== 20) b += lab(P[3], C.ok, `${s.r} tok/param`, 8, 14);
    b += text(64, 14, "dashed: isoFLOP lines 6ND = C (grey) and k·C (black) · blue: D = 20N · orange: Kaplan path", { fill: C.muted, size: 11 });
    pic.innerHTML = svg(640, 270, b);
    read.innerHTML = `start: C = ${sci(10 ** s.logC)} FLOPs → N = √(C/120) = ${sci(m.N0)}, D = 20N = ${sci(m.D0)}<br>
      budget × ${k} = ${sci(k * 10 ** s.logC)} FLOPs<br>
      Chinchilla: N ×${fmt(m.chNfactor, 3)} = <b>${sci(m.chN)}</b>, D = ${sci(m.chD)} · 20 tokens/param<br>
      Kaplan: N ×${fmt(m.kapNfactor, 3)} = <b>${sci(m.kapN)}</b>, D ×${fmt(m.kapDfactor, 3)} = ${sci(m.kapD)} · <span class="big">${fmt(m.kapRatio, 3)} tokens/param</span>
      ${k > 1 ? `<b style="color:var(--bad)">Kaplan puts ${fmt(m.kapN / m.chN, 2)}× more parameters on ${fmt(m.chD / m.kapD, 2)}× fewer tokens than Chinchilla for the same budget</b>` : "set the budget multiplier above 1 to see the two rules split"}<br>
      over-train at ${s.r} tokens/param: N = √(kC/(6r)) = ${sci(m.yourN)}, D = ${sci(m.yourD)} (${fmt(m.yourDT, 3)} T tokens) · serving cost per token ${fmt(m.serveRatio, 3)}× the Chinchilla model's<br>
      <span class="muted small">provenance: fixture:lecture_09--allocation · lecture_09.pdf:p45 (L446-L455, Kaplan N_opt = C^0.73, D_opt = C^0.27), p54 (L501-L514, tokens per parameter table; "the more usage we expect, the more it becomes worth it"); C = 6ND from lecture_02:six-nd; video 1:00:39-1:01:24, 1:12:42-1:12:53, 1:14:25-1:15:14. Kaplan's path is anchored at the same start point (the KP prompts' convention); serving cost ∝ N is the author's.</span>`;
  };
  const rw = slider("tokens per parameter r (over-training)", 1, 300, s.r, 1, v => { s.r = v; draw(); });
  rIn = rw.querySelector("input"); rOut = rw.querySelector(".readout");
  const presets = [["GPT-3", 2], ["Chinchilla", 20], ["LLaMA-65B", 22], ["Llama 2 70B", 29], ["Mistral 7B", 110], ["Llama 3 70B", 215]];
  root.append(el("div", { class: "widget" }, el("div", { class: "controls" },
    slider("start budget C (FLOPs)", 18, 26, s.logC, 0.01, v => { s.logC = v; draw(); }, v => sci(10 ** v)),
    slider("budget multiplier k", 0, ks.length - 1, s.ki, 1, v => { s.ki = v; draw(); }, v => `${ks[v]}×`),
    rw,
    el("div", { style: "margin-top:6px" }, el("span", { class: "small muted" }, "p54 ratios: "), ...presets.map(([name, r]) => el("button", { style: "margin:0 4px 4px 0;padding:3px 8px;font-size:12px", onclick: () => { s.r = r; rIn.value = r; rOut.textContent = r; draw(); } }, `${name} ${r}`))),
    read), pic)); draw();
};

// ================================================== 6. repeated data is worth less ==
// D' = U_D + U_D·R_D*·(1 − e^(−R_D/R_D*)), R_D = epochs − 1 repetitions beyond the first pass (lecture_09.pdf:p24, txt L260-L268:
//   only the legend "D' = Effective data, Ud = Unique tokens, Rd* = Constant, Rd = Repetition"; the formula is in the figure and
//   is written out in the KP's filtering note, after Muennighoff et al. 2023, sources.json muennighoff_2023).
//   R_D* ≈ 15.4 is that paper's fitted value (paper; the deck prints no value), so it is a slider here.
//   video 25:42-26:02 ("up to four epochs ... you just don't get hurt at all", past that "much worse than ... fresh data").
//   p26 (L273-L276): "Given that repeated data is less valuable.. Data selection should then be adaptive to scale !"; video 27:19-28:28.
//   Filter panel: a filter keeps a fraction f of a pool P, so U = f·P, and the budget needs D tokens. Only the repetition discount
//   is modelled; the per-token quality gain of a stricter filter has no number in the thread's sources and is not drawn.
const RSTAR = 15.4;
function effectiveData({ U = 1, D, epochs, Rstar = RSTAR }) {
  if (D === undefined) D = epochs * U;
  const ep = D / U, Dp = t => t <= U ? t : U + U * Rstar * (1 - Math.exp(-(t / U - 1) / Rstar));
  const eff = Dp(D);
  return { epochs: ep, Dp: eff, value: eff / D, ceiling: U * (1 + Rstar), lastEpoch: ep >= 1 ? (eff - Dp(D - U)) / U : NaN, lost: D - eff };
}
MODELS["fixture:lecture_09--effective-data"] = {
  fn: effectiveData,
  cases: [
    { args: { epochs: 2 }, pick: "lastEpoch", expect: 0.968 },         // predict (trend "down"): the 2nd epoch adds 0.97 U ...
    { args: { epochs: 3 }, pick: "lastEpoch", expect: 0.907 },         // ... the 3rd 0.91 U ...
    { args: { epochs: 10 }, pick: "lastEpoch", expect: 0.576 },        // ... the 10th 0.58 U ...
    { args: { epochs: 40 }, pick: "lastEpoch", expect: 0.0821 },       // ... the 40th 0.08 U: each epoch adds less than the one before
    { args: { epochs: 1e6 }, pick: "Dp", expect: 16.4 },               // ceiling U(1 + R_D*), however many epochs
    { args: { epochs: 4 }, pick: "value", expect: 0.931 },             // video 25:42: 4 epochs still worth 93% of fresh data
    { args: { U: 0.1, D: 0.1 }, pick: "value", expect: 1 },            // transfer: a 10% filter at the small budget, one pass, no discount
    { args: { U: 0.1, D: 10 }, pick: "value", expect: 0.164, tol: 0.02 }, // ... 100x the budget: 100 epochs, worth 16% of tokens seen
    { args: { U: 1, D: 10 }, pick: "value", expect: 0.782 },           // ... the loose filter (all of P) at that budget: 10 epochs, 78%
    { args: { U: 0.1, D: 1 }, pick: "value", expect: 0.782 },          // check (trend "up"): a 10x smaller budget cuts the strict filter's epochs 100 -> 10
    { args: { U: 1, D: 0.5 }, pick: "value", expect: 1 },              // edge: under one epoch nothing repeats
  ],
};
WIDGETS["fixture:lecture_09--effective-data"] = (root) => {
  const s = { ep: 6, Rstar: RSTAR, logD: 0, logf: -0.5 };              // D in units of the pool P; f = filter keep fraction
  const pic = el("div"), read = el("div", { class: "readout" });
  const draw = () => {
    const m = effectiveData({ epochs: s.ep, Rstar: s.Rstar }), D = 10 ** s.logD, f = 10 ** s.logf, q = effectiveData({ U: f, D, Rstar: s.Rstar });
    // left: D'/U against epochs (linear), with the fresh-data line and the ceiling
    const px = 46, py = 24, pw = 300, ph = 190, E1 = 60, Ymax = Math.max(20, s.Rstar + 3);
    const X = e => px + pw * e / E1, Y = v => py + ph * (1 - Math.min(v, Ymax) / Ymax);
    let b = `<rect x="${px}" y="${py}" width="${pw}" height="${ph}" style="fill:none;stroke:${C.rule}"/>`;
    for (let e = 0; e <= E1; e += 10) b += text(X(e), py + ph + 13, `${e}`, { fill: C.muted, size: 10, anchor: "middle" });
    for (let v = 0; v <= Ymax; v += 5) b += text(px - 5, Y(v) + 4, `${v}`, { fill: C.muted, size: 10, anchor: "end" });
    b += text(px + pw / 2, py + ph + 27, "epochs over the same U_D", { fill: C.muted, size: 11, anchor: "middle" });
    b += path([[X(0), Y(0)], [X(Ymax), Y(Ymax)]], C.muted, "stroke-dasharray:4 3") + text(X(Ymax * 0.55) - 6, Y(Ymax * 0.55) - 4, "fresh data", { fill: C.muted, size: 10, anchor: "end" });
    b += line(px, Y(1 + s.Rstar), px + pw, Y(1 + s.Rstar), C.b, "stroke-dasharray:5 3") + text(px + pw - 4, Y(1 + s.Rstar) - 4, `ceiling 1 + R* = ${fmt(1 + s.Rstar, 1)}`, { fill: C.b, size: 10, anchor: "end" });
    const cur = []; for (let e = 0; e <= E1 + 1e-9; e += 0.25) cur.push([X(e), Y(effectiveData({ epochs: e, Rstar: s.Rstar }).Dp)]);
    b += path(cur, C.a, "stroke-width:2.5");
    if (s.ep >= 1) { const lo = effectiveData({ epochs: s.ep - 1, Rstar: s.Rstar }).Dp; b += `<rect x="${(X(s.ep) - 6).toFixed(1)}" y="${Y(m.Dp).toFixed(1)}" width="12" height="${(Y(lo) - Y(m.Dp)).toFixed(1)}" style="fill:${C.hi};stroke:${C.ink}"/>`; }
    b += text(px + 2, 14, "blue: D' / U_D · yellow: the last epoch's gain", { fill: C.muted, size: 11 });
    // right: value D'/D against filter keep fraction f (log), for the chosen budget D
    const qx = 400, qw = 220;
    const g = logFrame({ px: qx, py, pw: qw, ph, x0: -2, x1: 0, y0: 0, y1: 1, xlab: "filter keeps f of the pool (log)", ylab: "D' / D", xfmt: v => `${fmt(10 ** v, 2)}`, yfmt: v => fmt(v, 2) });
    const Yl = v => py + ph * (1 - v);
    b += g.axes.replace(/<text transform[^>]*>[^<]*<\/text>/, "");
    for (const v of [0.25, 0.5, 0.75]) b += line(qx, Yl(v), qx + qw, Yl(v), C.rule, "stroke-dasharray:2 3") + text(qx - 5, Yl(v) + 4, fmt(v, 2), { fill: C.muted, size: 10, anchor: "end" });
    const vc = []; for (let lf = -2; lf <= 1e-9; lf += 0.02) vc.push([g.X(lf), Yl(effectiveData({ U: 10 ** lf, D, Rstar: s.Rstar }).value)]);
    b += path(vc, C.ok, "stroke-width:2.5") + dot(g.X(s.logf), Yl(q.value), C.hi);
    b += text(qx - 30, 14, `green: D' / D at budget D = ${fmt(D, 2)}·P`, { fill: C.muted, size: 11 });
    pic.innerHTML = svg(640, 250, b);
    const zone = s.ep <= 4 ? "≤ 4 epochs: repeats are nearly as good as fresh tokens (video 25:42)" : m.lastEpoch > 0.5 ? "past 4 epochs: each repeat is visibly worth less" : "saturating: another epoch adds little; D' is near its ceiling";
    read.innerHTML = `U_D = 1 unit · ${fmt(s.ep, 2)} epochs → R_D = ${fmt(Math.max(0, s.ep - 1), 2)} repeats<br>
      tokens seen ${fmt(s.ep, 2)} · <span class="big">D' = ${fmt(m.Dp, 3)}</span> (${fmt(100 * m.value, 1)}% of seen) · the last epoch added <b>${fmt(m.lastEpoch, 3)}</b> of a fresh epoch<br>
      <b style="color:${s.ep <= 4 ? "var(--ok)" : "var(--bad)"}">${zone}</b><br>
      filter keeps f = ${fmt(f, 3)} of the pool, budget needs D = ${fmt(D, 3)}·P → ${fmt(q.epochs, 2)} epochs over the kept set, D' = ${fmt(q.Dp, 3)}·P, <b>${fmt(100 * q.value, 1)}%</b> of the tokens seen count<br>
      <span class="muted small">Only the repetition discount is drawn. A stricter filter's better tokens are worth more per token, but the thread gives no number for that, so the best f is not computed here: watch how much a strict filter loses to repetition as D grows.</span><br>
      <span class="muted small">provenance: fixture:lecture_09--effective-data · lecture_09.pdf:p24 (L260-L268, legend D', U_D, R_D, R_D*; formula figure-only, after Muennighoff et al. 2023), p26 (L273-L276, data selection adaptive to scale); video 25:42-26:02 (four epochs barely hurt), 27:19-28:28 (filter hard at small compute, loosen at large). R_D* ≈ 15.4 is the paper's fit, not on the slides.</span>`;
  };
  root.append(el("div", { class: "widget" }, el("div", { class: "controls" },
    slider("epochs over the same unique tokens", 0, 60, s.ep, 0.5, v => { s.ep = v; draw(); }),
    slider("R_D* (saturation constant)", 2, 40, s.Rstar, 0.1, v => { s.Rstar = v; draw(); }, v => fmt(v, 1)),
    slider("budget D (tokens needed, × pool P)", -2, 1, s.logD, 0.05, v => { s.logD = v; draw(); }, v => fmt(10 ** v, 3)),
    slider("filter keeps f of the pool", -2, 0, s.logf, 0.05, v => { s.logf = v; draw(); }, v => fmt(10 ** v, 3)),
    read), pic)); draw();
};

// ============================================ 7. scaling a model up: shape and LR ==
// What carries over from a small sweep when the model grows.
//   Shape: N ≈ 12·n_layer·d_model² (the scale-invariant-hyperparameters transfer prompt, after lecture_02's per-layer count; author).
//   At a fixed aspect ratio r = d_model/n_layer, N ≈ 12·n_layer³·r², so n_layer and d_model each grow as k^(1/3) for k× parameters;
//   at a fixed layer count, d_model grows as k^(1/2) and the aspect ratio drifts. video 36:21-36:45 ("The number of layers is not a
//   scale-invariant quantity ... maybe the aspect ratio, the optimal one should stay the same"), 36:45-37:07 ("around 100d model
//   for every layer, or maybe a little bit less"), 37:15-37:34 (fix the aspect ratio and scale up). lecture_09.pdf:p34 (L348).
//   LR: standard parametrization, the rule of thumb LR ∝ 1/width (video 49:15-49:32; reason 48:57-49:13: "I'm changing more
//   things at once"). muP: the tuned LR is meant to stay optimal across widths (p40, txt L393-L401: "If we naively scale up – optimal learning rate depends on scale"; video 49:32-49:57).
//   muP's per-layer rules are not modelled (the lecture defers them to the advanced scaling lecture, 48:36).
function scaleUp({ L, d, eta = 1e-3, k, dNew, fixLayers = false }) {
  if (dNew === undefined) dNew = d * k ** (fixLayers ? 1 / 2 : 1 / 3);
  const r = d / L, LNew = fixLayers ? L : dNew / r, N = 12 * L * d * d, NNew = 12 * LNew * dNew * dNew;
  const etaSP = eta * d / dNew;
  return { r, dNew, LNew, N, NNew, k: NNew / N, widthFactor: dNew / d, layerFactor: LNew / L, aspectNew: dNew / LNew,
    aspectFactor: (dNew / LNew) / r, etaSP, etaMuP: eta, overshoot: eta / etaSP };
}
MODELS["fixture:lecture_09--scale-up"] = {
  fn: scaleUp,
  cases: [
    { args: { L: 6, d: 512, eta: 6e-3, dNew: 3072 }, pick: "etaSP", expect: 1e-3, from: "lecture_09:lr-shrinks-with-width:predict" },
    { args: { L: 6, d: 512, eta: 4e-3, dNew: 4096 }, pick: "widthFactor", expect: 8, from: "lecture_09:lr-shrinks-with-width:check" },
    { args: { L: 6, d: 512, eta: 4e-3, dNew: 4096 }, pick: "etaSP", expect: 5e-4 },              // ... the check's 4e-3 -> 5e-4
    { args: { L: 12, d: 1024, eta: 3e-3, dNew: 8192 }, pick: "etaSP", expect: 3.75e-4 },          // lr transfer: optimum ≈ 4e-4 at 8192 ...
    { args: { L: 12, d: 1024, eta: 3e-3, dNew: 8192 }, pick: "overshoot", expect: 8 },            // ... so 3e-3 is 8× too high
    { args: { L: 6, d: 512, k: 8 }, pick: "layerFactor", expect: 2, tol: 0.05, from: "lecture_09:scale-invariant-hyperparameters:transfer" },
    { args: { L: 6, d: 512, k: 8 }, pick: "widthFactor", expect: 2 },                             // ... and d_model 2× as well
    { args: { L: 6, d: 512, k: 8, fixLayers: true }, pick: "aspectFactor", expect: Math.sqrt(8) }, // fixed layer count: aspect ×2.8 (transfer's why)
    { args: { L: 6, d: 512, dNew: 4096 }, pick: "LNew", expect: 48 },                             // predict: carry r ≈ 85 to d_model 4096 -> 48 layers
    { args: { L: 6, d: 512, dNew: 4096 }, pick: "aspectNew", expect: 85.33 },                     // ... the aspect ratio is what stays
    { args: { L: 6, d: 512, k: 64 }, pick: "layerFactor", expect: 4 },                            // check (trend "up"): 64× params -> 4× layers
    { args: { L: 6, d: 512, dNew: 4096, fixLayers: true }, pick: "k", expect: 64 },               // fixed layers: 8× width is 64× params
  ],
};
WIDGETS["fixture:lecture_09--scale-up"] = (root) => {
  const s = { L: 8, d: 768, logEta: Math.log10(2e-3), logk: Math.log10(4), fixLayers: false };   // not a prompt's case
  const pic = el("div"), read = el("div", { class: "readout" });
  let kIn, kOut, wIn, wOut;
  const cur = () => scaleUp({ L: s.L, d: s.d, eta: 10 ** s.logEta, k: 10 ** s.logk, fixLayers: s.fixLayers });
  const sync = () => { const m = cur(); kIn.value = s.logk; kOut.textContent = `${fmt(10 ** s.logk, 3)}×`; wIn.value = Math.round(m.dNew); wOut.textContent = fmt(m.dNew, 0); };
  const draw = () => {
    const m = cur();
    // shapes: width ∝ d_model, height ∝ n_layer·r_base, so the base model is a square and a fixed-aspect model stays one
    const W = 640, H = 230, top = 30, maxW = Math.max(s.d, m.dNew), maxH = Math.max(s.L, m.LNew) * m.r;
    const sc = Math.min(270 / maxW, 170 / maxH);
    const box = (x0, Lr, dd, fill, lab) => { const w = dd * sc, h = Lr * m.r * sc, y = top + 170 - h;
      let g = `<rect x="${x0.toFixed(1)}" y="${y.toFixed(1)}" width="${w.toFixed(1)}" height="${h.toFixed(1)}" style="fill:${fill};fill-opacity:0.25;stroke:${fill};stroke-width:2"/>`;
      if (Lr <= 96 && h / Lr >= 2.5) for (let i = 1; i < Math.round(Lr); i++) g += line(x0, y + h * i / Lr, x0 + w, y + h * i / Lr, fill, "stroke-opacity:0.35");
      return g + text(x0, top + 186, lab, { size: 11 }); };
    let b = text(10, 16, "width = d_model · height = layers (scaled so the base is a square) · horizontal lines: one per layer", { fill: C.muted, size: 11 });
    b += box(20, s.L, s.d, C.muted, `base: ${s.L} layers × ${s.d}`);
    b += box(330, m.LNew, m.dNew, s.fixLayers ? C.b : C.a, `scaled: ${fmt(m.LNew, 1)} layers × ${fmt(m.dNew, 0)}`);
    pic.innerHTML = svg(W, H, b);
    const aspectOk = Math.abs(m.aspectFactor - 1) < 0.02;
    read.innerHTML = `base: N ≈ 12·${s.L}·${s.d}² = ${sci(m.N)} · aspect d_model/n_layer = <b>${fmt(m.r, 1)}</b> · tuned LR ${sci(10 ** s.logEta)}<br>
      scaled ×${fmt(m.k, 3)} parameters (${sci(m.NNew)}): d_model ×${fmt(m.widthFactor, 3)} = ${fmt(m.dNew, 0)}, layers ×${fmt(m.layerFactor, 3)} = <span class="big">${fmt(m.LNew, 1)}</span> · aspect ${fmt(m.aspectNew, 1)}<br>
      <b style="color:${aspectOk ? "var(--ok)" : "var(--bad)"}">${aspectOk ? "fixed aspect ratio: the swept optimum carries over; the layer count does not (it grew)" : `fixed layer count: the aspect ratio moved ×${fmt(m.aspectFactor, 3)}, away from the value the small sweep found`}</b><br>
      learning rate at width ${fmt(m.dNew, 0)}: standard parametrization, 1/width rule → <span class="big">${sci(m.etaSP)}</span> (÷${fmt(m.widthFactor, 3)}); keeping ${sci(10 ** s.logEta)} would be ${fmt(m.overshoot, 3)}× too high · muP: ${sci(m.etaMuP)} (meant to stay put)<br>
      <span class="muted small">provenance: fixture:lecture_09--scale-up · video 36:21-37:34 (layer count is not scale-invariant, aspect ratio is; minima "around 100d model for every layer"), 48:57-49:32 (1/width rule of thumb under standard parametrization), 49:32-49:57 (muP keeps the LR minimum); lecture_09.pdf:p34 (L348), p40 (L393-L401). N ≈ 12·n_layer·d_model² ignores embeddings (the KP prompt's count). The muP column only restates the slide's claim; its per-layer rules are not modelled.</span>`;
  };
  const kw = slider("parameter multiplier k", -1, 3, s.logk, 0.01, v => { s.logk = v; sync(); draw(); }, v => `${fmt(10 ** v, 3)}×`);
  kIn = kw.querySelector("input"); kOut = kw.querySelector(".readout");
  const ww = slider("or set the target d_model", 128, 32768, Math.round(cur().dNew), 64, v => { s.logk = Math.log10(scaleUp({ L: s.L, d: s.d, dNew: v, fixLayers: s.fixLayers }).k); sync(); draw(); }, v => fmt(v, 0));
  wIn = ww.querySelector("input"); wOut = ww.querySelector(".readout");
  root.append(el("div", { class: "widget" }, el("div", { class: "controls" },
    slider("base layers", 1, 48, s.L, 1, v => { s.L = v; sync(); draw(); }),
    slider("base d_model", 128, 4096, s.d, 64, v => { s.d = v; sync(); draw(); }),
    slider("LR tuned at the base width", -4, -1.5, s.logEta, 0.01, v => { s.logEta = v; draw(); }, v => sci(10 ** v)),
    check("hold the layer count fixed (grow only the width)", s.fixLayers, v => { s.fixLayers = v; sync(); draw(); }),
    kw, ww, read), pic)); draw();
};
