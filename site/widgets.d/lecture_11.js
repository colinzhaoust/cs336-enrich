// Widgets for thread lecture_11 (CS336 L11, scaling case studies). Register as "fixture:<id>" -> (root, notice) => void.
// Each widget has a pure model in MODELS (no DOM) that tools/check_widgets.mjs tests against the KPs' stored answers.
// Sources: lectures/lecture_11/lecture_11.txt (slide text layer, cited as pdf page + txt line),
//          lectures/lecture_11/transcript.json (cited as video M:SS), and the KP anchors / filtering notes that already
//          record which numbers are the author's (marked "author" below). No constant here is new.
import { el, fmt, slider } from "../widgets_lib.js";

export const MODELS = {};
export const WIDGETS = {};

const C = { ink: "var(--ink)", muted: "var(--muted)", rule: "var(--rule)", a: "var(--accent)", b: "var(--accent2)", ok: "var(--ok)", hi: "var(--hilite)" };
const rect = (x, y, w, h, fill, extra = "") => `<rect x="${x.toFixed(1)}" y="${y.toFixed(1)}" width="${Math.max(0, w).toFixed(1)}" height="${h}" style="fill:${fill}" ${extra}/>`;
const text = (x, y, s, opts = {}) => `<text x="${x.toFixed(1)}" y="${y.toFixed(1)}" style="fill:${opts.fill || C.ink};font:${opts.size || 12}px var(--sans)" text-anchor="${opts.anchor || "start"}">${s}</text>`;
const line = (x1, y1, x2, y2, col, extra = "") => `<line x1="${x1.toFixed(1)}" y1="${y1.toFixed(1)}" x2="${x2.toFixed(1)}" y2="${y2.toFixed(1)}" style="stroke:${col};${extra}"/>`;
const path = (pts, col, extra = "") => `<path d="${pts.map((p, i) => `${i ? "L" : "M"}${p[0].toFixed(1)},${p[1].toFixed(1)}`).join("")}" style="fill:none;stroke:${col};${extra}"/>`;
const svg = (w, h, body) => `<svg viewBox="0 0 ${w} ${h}" width="100%" style="max-width:${w}px;border:1px solid var(--rule);border-radius:6px;background:#fff" role="img">${body}</svg>`;
const buttons = (opts, get, set) => el("div", { style: "margin:4px 0" }, ...opts.map(([k, n]) => el("button", { onclick: () => set(k) }, n)));

// ============================================================ 1. sweep cost: cosine vs WSD ==
// What it costs to get L(N, D) at k data sizes for one model.
//   Cosine: each point is its own from-scratch run (lecture_11.pdf:p13, txt L88-L94, "need to train from scratch, not just early
//   stop ... from n to n^2"; video 10:24-11:04: the cosine schedule needs the total budget up front, "a quadratic cost").
//   WSD: one stable-phase run to the largest D, then one decay branch per point (p14, txt L95-L101, "can restart the run at the end of
//   the stable phase"; video 12:06-12:44, 14:09-14:22: "rewind the checkpoints and then repeatedly decay").
//   The decay fraction d: "Decay ~ 10%" (p15, txt L103); "10% to 20%" (video 11:42); each redecay "maybe 10% of the total cost" (12:32).
//   Accounting (the KP's, wsd-schedule-branching check): a branch leaves the stable run at D_i and decays for d·D_i extra tokens,
//   so WSD total = D_max + d·ΣD_i, cosine total = ΣD_i. Warmup is not drawn: a constant number of steps (video 11:23-11:32), shared
//   by every branch. The decay is drawn linear to 0 (the deck does not say; spoken "down to 0" 11:37 vs "about 10% of max" 11:48);
//   the shape does not change the cost.
function sweepCost({ sizes, d }) {
  const sum = sizes.reduce((a, x) => a + x, 0), Dmax = Math.max(...sizes);
  const wsd = Dmax + d * sum;
  return { k: sizes.length, cosine: sum, single: Dmax, wsd, decays: d * sum, cosOverSingle: sum / Dmax, saving: sum / wsd,
    stableRuns: 1, branches: sizes.length, dStar: 1 - Dmax / sum, wsdCheaper: wsd < sum ? 1 : 0 };
}
const sizesOf = ({ k, spacing, Dmax }) => Array.from({ length: k }, (_, i) => spacing === "even" ? Dmax * (i + 1) / k : Dmax / 2 ** (k - 1 - i));
MODELS["fixture:lecture_11--sweep-cost"] = {
  fn: sweepCost,
  cases: [
    { args: { sizes: [1, 2, 3, 4, 5], d: 0.1 }, pick: "cosine", expect: 15, tol: 0.05, from: "lecture_11:chinchilla-fit-cost-n-squared:check" },
    { args: { sizes: [10e9, 20e9, 40e9, 80e9], d: 0.1 }, pick: "wsd", expect: 95e9, tol: 0.03, from: "lecture_11:wsd-schedule-branching:check" },
    { args: { sizes: [10e9, 20e9, 40e9, 80e9], d: 0.1 }, pick: "cosine", expect: 150e9 },          // the check's why: four cosine runs
    { args: { sizes: [10e9, 20e9, 40e9, 80e9], d: 0.1 }, pick: "dStar", expect: 1 - 80 / 150 },     // crossover: WSD loses only if d > 47%
    { args: { sizes: [1, 2, 3, 4, 5, 6, 7, 8], d: 0.1 }, pick: "k", expect: 8 },                    // chinchilla-fit-cost predict: 8 runs, not 1
    { args: { sizes: [1, 2, 3, 4, 5, 6, 7, 8], d: 0.1 }, pick: "cosOverSingle", expect: 4.5 },      // ... (n+1)/2 longest-run units vs 1
    { args: { sizes: [1, 2, 3, 4, 5, 6], d: 0.1 }, pick: "branches", expect: 6 },                   // wsd transfer: 1 stable run, 6 branches
    { args: { sizes: [1, 2, 3, 4, 5, 6], d: 0.1 }, pick: "wsd", expect: 8.1 },                      // ... 6 + 0.1·21
    { args: { sizes: [5], d: 0.1 }, pick: "wsdCheaper", expect: 0 },                                // edge: one point, nothing to share
    { args: { sizes: [5], d: 0.1 }, pick: "saving", expect: 1 / 1.1 },                              // ... WSD costs 1.1× cosine
    { args: { sizes: [1, 2, 3, 4, 5], d: 0 }, pick: "wsd", expect: 5 },                             // edge: free decay = one long run
  ],
};
WIDGETS["fixture:lecture_11--sweep-cost"] = (root) => {
  const s = { k: 3, spacing: "double", Dmax: 40, d: 0.1 };
  const pic = el("div"), read = el("div", { class: "readout" });
  const draw = () => {
    const sizes = sizesOf(s), m = sweepCost({ sizes, d: s.d });
    const W = 640, x0 = 92, span = 520, rowH = 15, X = t => x0 + span * t / (s.Dmax * (1 + s.d));
    const cosTop = 34, wsdTop = cosTop + m.k * rowH + 40;
    const rowLR = (y, pts, col, extra) => path(pts.map(([t, v]) => [X(t), y + rowH - 3 - v * (rowH - 5)]), col, extra);
    let b = text(10, 16, "each row is one data size D_i · line = LR schedule over tokens · filled = tokens you pay for", { fill: C.muted, size: 11 });
    b += text(10, cosTop - 6, `cosine: ${m.k} from-scratch runs`, { fill: C.b, size: 12 });
    sizes.forEach((D, i) => {
      const y = cosTop + i * rowH;
      b += text(x0 - 6, y + 11, `${fmt(D, 1)}B`, { fill: C.muted, size: 10, anchor: "end" });
      b += rect(X(0), y + 1, X(D) - X(0), rowH - 3, C.b, 'opacity="0.18"');
      b += rowLR(y, Array.from({ length: 41 }, (_, j) => [D * j / 40, 0.5 * (1 + Math.cos(Math.PI * j / 40))]), C.b, "stroke-width:1.5");
    });
    b += text(10, wsdTop - 6, "WSD: 1 stable run + 1 decay branch per point", { fill: C.a, size: 12 });
    const ys = wsdTop;
    b += text(x0 - 6, ys + 11, "stable", { fill: C.muted, size: 10, anchor: "end" });
    b += rect(X(0), ys + 1, X(s.Dmax) - X(0), rowH - 3, C.a, 'opacity="0.25"') + rowLR(ys, [[0, 1], [s.Dmax, 1]], C.a, "stroke-width:2");
    sizes.forEach((D, i) => {
      const y = wsdTop + (i + 1) * rowH;
      b += text(x0 - 6, y + 11, `${fmt(D, 1)}B`, { fill: C.muted, size: 10, anchor: "end" });
      b += rowLR(y, [[0, 1], [D, 1]], C.a, "stroke-width:1;stroke-dasharray:3 3;opacity:0.6");
      b += rect(X(D), y + 1, X(D * (1 + s.d)) - X(D), rowH - 3, C.a, 'opacity="0.45"');
      b += rowLR(y, [[D, 1], [D * (1 + s.d), 0]], C.a, "stroke-width:2");
    });
    const H = wsdTop + (m.k + 1) * rowH + 26;
    b += line(X(0), H - 20, X(s.Dmax * (1 + s.d)), H - 20, C.rule);
    for (let j = 0; j <= 4; j++) b += text(X(s.Dmax * j / 4), H - 6, `${fmt(s.Dmax * j / 4, 1)}B`, { fill: C.muted, size: 10, anchor: "middle" });
    b += text(W - 8, H - 6, "tokens", { fill: C.muted, size: 10, anchor: "end" });
    // totals: the whole sweep's bill, on its own scale
    const tTop = H + 10, big = Math.max(m.cosine, m.wsd), T = v => span * v / big;
    b += text(10, tTop + 2, "total tokens for the whole sweep", { fill: C.muted, size: 11 });
    b += text(x0 - 6, tTop + 22, "cosine", { fill: C.b, size: 11, anchor: "end" }) + rect(x0, tTop + 10, T(m.cosine), 16, C.b, 'opacity="0.6"') + text(x0 + T(m.cosine) - 4, tTop + 22, `${fmt(m.cosine, 1)}B`, { fill: "#fff", size: 11, anchor: "end" });
    b += text(x0 - 6, tTop + 44, "WSD", { fill: C.a, size: 11, anchor: "end" }) + rect(x0, tTop + 32, T(m.single), 16, C.a, 'opacity="0.3"') + rect(x0 + T(m.single), tTop + 32, T(m.decays), 16, C.a, 'opacity="0.7"');
    const wl = `${fmt(m.wsd, 1)}B (stable ${fmt(m.single, 1)} + decays ${fmt(m.decays, 1)})`, inside = T(m.wsd) > span - 240;
    b += text(inside ? x0 + 6 : x0 + T(m.wsd) + 4, tTop + 44, wl, { fill: inside ? C.ink : C.a, size: 11 });
    pic.innerHTML = svg(W, tTop + 56, b);
    const cheaper = m.wsdCheaper ? `WSD is cheaper: ${fmt(m.saving, 2)}× less compute than cosine` : m.k === 1 ? "one data size: nothing to share, the decay is pure overhead" : `cosine is cheaper here: the decay fraction exceeds the break-even d* = ${fmt(100 * m.dStar, 1)}%`;
    read.innerHTML = `data sizes (B tokens): ${sizes.map(D => fmt(D, 2)).join(", ")}<br>
      cosine = ΣD_i = <b>${fmt(m.cosine, 2)}B</b> tokens in ${m.k} runs (${fmt(m.cosOverSingle, 2)}× one run to ${fmt(m.single, 1)}B)<br>
      WSD = D_max + d·ΣD_i = ${fmt(m.single, 1)} + ${fmt(s.d, 2)}·${fmt(m.cosine, 2)} = <span class="big">${fmt(m.wsd, 2)}B</span> tokens in 1 stable run and ${m.branches} decay branches<br>
      <b style="color:${m.wsdCheaper ? "var(--ok)" : "var(--bad)"}">${cheaper}</b><br>
      <span class="muted small">provenance: fixture:lecture_11--sweep-cost · lecture_11.pdf:p13 (L88-L94: train from scratch, "from n to n^2"), p14 (L95-L101: restart at the end of the stable phase), p15 (L103: "Decay ~ 10%"); video 10:24-11:04 (cosine needs the total budget up front), 11:37-11:57 (decay 10-20%), 12:06-12:44 (roll back, redecay, "maybe 10% of the total cost"), 14:09-14:22. Accounting D_max + d·ΣD_i is the KP's (wsd-schedule-branching check). Loss is not modelled (p15's curves are figure-only); warmup (shared, a fixed number of steps) is not drawn; the linear decay to 0 is a drawing choice.</span>`;
  };
  root.append(el("div", { class: "widget" }, el("div", { class: "controls" },
    slider("number of data sizes k", 1, 10, s.k, 1, v => { s.k = v; draw(); }),
    buttons([["even", "even spacing D_max·i/k"], ["double", "doubling …, D_max/2, D_max"]], null, v => { s.spacing = v; draw(); }),
    slider("largest D (B tokens)", 10, 200, s.Dmax, 10, v => { s.Dmax = v; draw(); }),
    slider("decay fraction d", 0, 0.5, s.d, 0.01, v => { s.d = v; draw(); }, v => `${Math.round(v * 100)}%`),
    read), pic)); draw();
};

// ====================================================== 2. StepFun laws in N and D vs a compute law ==
// Fitted Step Law (arXiv:2503.04715, recorded in the stepfun-laws-along-chinchilla-path anchors and filtering note, author's check):
//   η_opt = 1.79 · N^-0.713 · D^0.307,   B_opt = 0.58 · D^0.571 (tokens).
// The deck prints no exponent (p36, txt L228-L230: batch "primarily dependent on dataset size", "higher optimal LR with D (for fixed M)").
// Spoken: batch "roughly square root of the number of data", LR up with data, down with model size (video 38:08-38:27); along a
// Chinchilla path N and D are both driven by compute and the result looks like DeepSeek's law, LR down and batch up with compute
// (38:27-39:03). A compute-only law fitted on runs along N, D ∝ C^0.5 is η ∝ C^((aN+bD)/2), B ∝ C^(bB/2) (the KP's reduction).
// Base point N = 1B, D = 100B is the professor's reading of the p35 slice (video 34:25-34:46).
function hpLaws({ nx, dx, aN = -0.713, bD = 0.307, bB = 0.571, N0 = 1e9, D0 = 1e11 }) {
  const lr = nx ** aN * dx ** bD, batch = dx ** bB, cx = nx * dx, cExpLR = (aN + bD) / 2, cExpB = bB / 2;
  const eta0 = 1.79 * N0 ** -0.713 * D0 ** 0.307, B0 = 0.58 * D0 ** 0.571;
  return { lr, batch, cx, cExpLR, cExpB, lrComputeLaw: cx ** cExpLR, batchComputeLaw: cx ** cExpB, eta0, B0, eta: eta0 * lr, B: B0 * batch,
    onPath: Math.abs(Math.log10(nx) - Math.log10(dx)) < 1e-6 ? 1 : 0 };
}
MODELS["fixture:lecture_11--hp-laws"] = {
  fn: hpLaws,
  cases: [
    { args: { nx: 10, dx: 10 }, pick: "lr", expect: 0.39, tol: 0.05, from: "lecture_11:stepfun-laws-along-chinchilla-path:predict" },
    { args: { nx: 10, dx: 10 }, pick: "lrComputeLaw", expect: 10 ** -0.406 },           // on the path the compute law agrees exactly
    { args: { nx: 1, dx: 10 }, pick: "lr", expect: 2.028, tol: 0.01 },                     // transfer: fixed N, 10× data: LR rises ~2×
    { args: { nx: 1, dx: 10 }, pick: "batch", expect: 3.724, tol: 0.01 },                  // ... batch 10^0.571 ≈ 3.7×
    { args: { nx: 1, dx: 10 }, pick: "lrComputeLaw", expect: 0.6266, tol: 0.01 },          // ... the compute law predicts a fall: wrong direction
    { args: { nx: 10, dx: 10, aN: -0.3, bD: 0.5 }, pick: "cExpLR", expect: 0.1 },          // chinchilla-path check: C^+0.1, so up
    { args: { nx: 10, dx: 10 }, pick: "cExpLR", expect: -0.203, tol: 0.01 },               // filtering note: LR exponent ≈ -0.20 along the path
    { args: { nx: 10, dx: 10 }, pick: "cExpB", expect: 0.2855, tol: 0.01 },                // ... batch exponent ≈ 0.29
    { args: { nx: 10, dx: 10, aN: -0.5, bD: 0.5 }, pick: "lr", expect: 1 },                // edge: equal exponents cancel ("would cancel", 38:27)
    { args: { nx: 2, dx: 1 }, pick: "batch", expect: 1 },                                  // stepfun-convexity predict: 2× N, same data: batch unchanged
    { args: { nx: 1, dx: 4 }, pick: "batch", expect: 2.207, tol: 0.01 },                   // stepfun-convexity check: 4× data: batch grows
    { args: { nx: 1, dx: 1 }, pick: "eta0", expect: 1.632e-3, tol: 0.01 },                 // base η at 1B params, 100B tokens
  ],
};
WIDGETS["fixture:lecture_11--hp-laws"] = (root) => {
  const s = { mode: "path", lc: 1, ln: 0.5, ld: 0, aN: -0.713, bD: 0.307 };
  const pic = el("div"), read = el("div", { class: "readout" }), ctl = el("div");
  const draw = () => {
    const ln = s.mode === "path" ? s.lc / 2 : s.ln, ld = s.mode === "path" ? s.lc / 2 : s.ld;
    const a = { nx: 10 ** ln, dx: 10 ** ld, aN: s.aN, bD: s.bD }, m = hpLaws(a);
    // plane: x = log10 N×, y = log10 D×, both in [-1, 3]
    const W = 640, H = 330, px = 60, py = 16, pw = 300, ph = 280, lo = -1, hi = 3;
    const X = v => px + pw * (v - lo) / (hi - lo), Y = v => py + ph * (hi - v) / (hi - lo);
    const clip = `<clipPath id="l11hp"><rect x="${px}" y="${py}" width="${pw}" height="${ph}"/></clipPath>`;
    // region where LR is above base: aN·x + bD·y > 0
    const slope = -s.aN / s.bD;                                                       // iso-LR line through base: y = slope·x
    let b = clip + `<g clip-path="url(#l11hp)">`;
    b += `<polygon points="${[[lo, lo * slope], [hi, hi * slope], [lo - 10, hi + 10]].map(([x, y]) => `${X(x).toFixed(1)},${Y(y).toFixed(1)}`).join(" ")}" style="fill:${C.ok};opacity:0.10"/>`;
    for (const f of [-1, -0.5, 0.5, 1]) {                                             // iso-LR lines at LR × 10^f
      const y = x => (f - s.aN * x) / s.bD;
      b += line(X(lo), Y(y(lo)), X(hi), Y(y(hi)), C.ok, "stroke-width:1;stroke-dasharray:2 3;opacity:0.7");
    }
    b += line(X(lo), Y(lo * slope), X(hi), Y(hi * slope), C.ok, "stroke-width:1.5");
    for (const f of [0.5, 1, 2]) b += line(X(lo), Y(f), X(hi), Y(f), C.b, "stroke-width:1;stroke-dasharray:6 4;opacity:0.5"); // iso-batch: horizontal
    b += line(X(lo), Y(lo), X(hi), Y(hi), C.a, "stroke-width:2");                     // Chinchilla path
    b += `</g>` + `<rect x="${px}" y="${py}" width="${pw}" height="${ph}" style="fill:none;stroke:${C.rule}"/>`;
    for (let v = lo; v <= hi; v++) { b += text(X(v), py + ph + 14, `${10 ** v}×`, { fill: C.muted, size: 10, anchor: "middle" }) + text(px - 6, Y(v) + 4, `${10 ** v}×`, { fill: C.muted, size: 10, anchor: "end" }); }
    b += text(px + pw / 2, H - 4, "model size N (× base, log)", { fill: C.muted, size: 11, anchor: "middle" });
    b += `<text transform="translate(14 ${py + ph / 2}) rotate(-90)" style="fill:${C.muted};font:11px var(--sans)" text-anchor="middle">data D (× base, log)</text>`;
    b += line(X(0), Y(0), X(ln), Y(ld), C.ink, "stroke-width:1.5") + `<circle cx="${X(0)}" cy="${Y(0)}" r="4" style="fill:${C.ink}"/>`;
    b += `<circle cx="${X(Math.max(lo, Math.min(hi, ln))).toFixed(1)}" cy="${Y(Math.max(lo, Math.min(hi, ld))).toFixed(1)}" r="6" style="fill:${C.hi};stroke:${C.ink}"/>`;
    // legend
    const lx = px + pw + 22;
    b += line(lx, 30, lx + 26, 30, C.a, "stroke-width:2") + text(lx + 32, 34, "Chinchilla path N, D ∝ C^0.5", { size: 11 });
    b += line(lx, 50, lx + 26, 50, C.ok, "stroke-width:1.5") + text(lx + 32, 54, `same LR as base (slope ${fmt(slope, 2)})`, { size: 11 });
    b += line(lx, 70, lx + 26, 70, C.ok, "stroke-dasharray:2 3") + text(lx + 32, 74, "LR × 10^±0.5, 10^±1", { size: 11 });
    b += rect(lx, 84, 26, 12, C.ok, 'opacity="0.18"') + text(lx + 32, 94, "LR higher than base", { size: 11 });
    b += line(lx, 110, lx + 26, 110, C.b, "stroke-dasharray:6 4;opacity:0.6") + text(lx + 32, 114, "same batch (B depends on D only)", { size: 11 });
    b += `<circle cx="${lx + 13}" cy="130" r="4" style="fill:${C.ink}"/>` + text(lx + 32, 134, "base: 1B params, 100B tokens", { size: 11 });
    b += `<circle cx="${lx + 13}" cy="150" r="6" style="fill:${C.hi};stroke:${C.ink}"/>` + text(lx + 32, 154, "your run", { size: 11 });
    const dir = v => v > 1 + 1e-9 ? "up" : v < 1 - 1e-9 ? "down" : "unchanged";
    const agree = dir(m.lr) === dir(m.lrComputeLaw);
    const verdict = m.onPath ? "on the Chinchilla path: the compute-only law and the N–D law agree exactly" : agree ? `off the path: the compute-only law gets the LR direction right (${dir(m.lr)}) but the size wrong` : `off the path: the compute-only law says LR goes ${dir(m.lrComputeLaw)}, the N–D law says ${dir(m.lr)}: wrong direction`;
    pic.innerHTML = svg(W, H, b);
    read.innerHTML = `N × ${fmt(m.cx / a.dx, 3)} · D × ${fmt(a.dx, 3)} · compute C = 6ND × ${fmt(m.cx, 3)}<br>
      N–D law: optimal LR × N×^${fmt(s.aN, 3)} · D×^${fmt(s.bD, 3)} = <span class="big">${fmt(m.lr, 3)}</span> (${dir(m.lr)}) · optimal batch × D×^0.571 = <b>${fmt(m.batch, 3)}</b> (N does not enter)<br>
      compute-only law fitted along the path: LR ∝ C^${fmt(m.cExpLR, 3)} → × ${fmt(m.lrComputeLaw, 3)} · batch ∝ C^${fmt(m.cExpB, 3)} → × ${fmt(m.batchComputeLaw, 3)}<br>
      <b style="color:${m.onPath || agree ? "var(--ok)" : "var(--bad)"}">${verdict}</b><br>
      absolute (Step Law fit, exponents fixed at the fitted values): η ≈ ${fmt(m.eta0 * a.nx ** -0.713 * a.dx ** 0.307, 3)} (base ${fmt(m.eta0, 3)}), B ≈ ${fmt(m.B, 3)} tokens (base ${fmt(m.B0, 3)})<br>
      <span class="muted small">provenance: fixture:lecture_11--hp-laws · fitted η = 1.79 N^-0.713 D^0.307 and B = 0.58 D^0.571 are the Step Law paper's (arXiv:2503.04715), recorded in the stepfun-laws-along-chinchilla-path KP (author's check); the deck prints no exponent: lecture_11.pdf:p36 (L228-L230: batch primarily depends on D; LR rises with D at fixed M, "likely more fragile if swapping to WSD"). Video 36:15-36:37 (LR down with model size, up with data, "counterintuitive"), 38:08-39:03 (batch ~ sqrt(D); along a Chinchilla path LR falls and batch grows with compute, like DeepSeek's law with different exponents); base point 1B / 100B from 34:25-34:46. DeepSeek's own exponents are figure-only (p21) and not used. The numbers are "likely contingent" on the data (37:43).</span>`;
  };
  const sl = () => {
    ctl.replaceChildren(...(s.mode === "path"
      ? [slider("compute C (× base, log10)", -2, 6, s.lc, 0.1, v => { s.lc = v; draw(); }, v => `${fmt(10 ** v, 3)}× (N, D each ${fmt(10 ** (v / 2), 3)}×)`)]
      : [slider("model size N (× base, log10)", -1, 3, s.ln, 0.1, v => { s.ln = v; draw(); }, v => `${fmt(10 ** v, 3)}×`),
         slider("data D (× base, log10)", -1, 3, s.ld, 0.1, v => { s.ld = v; draw(); }, v => `${fmt(10 ** v, 3)}×`)]));
  };
  root.append(el("div", { class: "widget" }, el("div", { class: "controls" },
    buttons([["path", "along the Chinchilla path"], ["free", "set N and D separately"]], null, v => { if (s.mode === v) return; if (v === "free") { s.ln = s.lc / 2; s.ld = s.lc / 2; } else s.lc = s.ln + s.ld; s.mode = v; sl(); draw(); }),
    ctl,
    slider("LR exponent of N (fit: −0.713)", -1, 0, s.aN, 0.001, v => { s.aN = v; draw(); }),
    slider("LR exponent of D (fit: 0.307)", 0.01, 1, s.bD, 0.001, v => { s.bD = v; draw(); }),
    read), pic)); sl(); draw();
};

// ================================================ 3. muP init and the worst-case activation bound ==
// lecture_11.pdf:p47 (txt L319-L337, layout-mangled; the KP statement records the reconstruction):
//   W_l ~ N(0, σ² I), ||W_l||_op ≈ σ(√n_{l−1} + √n_l) ("basic matrix concentration"; "the ≈ is an upper bound", "worst case").
//   σ = √(n_l/n_{l−1}) / (√n_l + √n_{l−1}) = Θ((1/√n_{l−1})·min(1, √(n_l/n_{l−1}))), so with ||h_{l−1}|| = √n_{l−1} the bound gives ||h_l|| ≤ √n_l.
//   Standard parametrization: σ = 1/√n_{l−1} (p50 recap).
// Per-coordinate worst-case size of h_l: r = σ(√n_{l−1} + √n_l)·√n_{l−1} / √n_l (target Θ(1), A1 on p46).
// Typical gain (author's check, mup-two-conditions-and-init filtering note): ||W h|| ≈ σ√n_l ||h||, so the per-coordinate size is σ√n_{l−1}.
function mupInit({ nIn, nOut }) {
  const ri = Math.sqrt(nIn), ro = Math.sqrt(nOut);
  const sp = 1 / ri, theta = Math.min(1, ro / ri) / ri, exact = (ro / ri) / (ro + ri);
  const bound = sig => sig * (ri + ro) * ri / ro, typical = sig => sig * ri;
  return { sp, theta, exact, spOverTheta: sp / theta, spOverExact: sp / exact, boundSP: bound(sp), boundTheta: bound(theta), boundExact: bound(exact),
    typSP: typical(sp), typTheta: typical(theta), typExact: typical(exact), ratio: nOut / nIn };
}
MODELS["fixture:lecture_11--mup-init"] = {
  fn: mupInit,
  cases: [
    { args: { nIn: 4096, nOut: 1024 }, pick: "spOverTheta", expect: 2, tol: 0.05, from: "lecture_11:mup-two-conditions-and-init:predict" },
    { args: { nIn: 4096, nOut: 1024 }, pick: "spOverExact", expect: 3 },                   // predict's why: 1/192 vs 1/64 with exact constants
    { args: { nIn: 1024, nOut: 4096 }, pick: "exact", expect: 1 / 48, tol: 0.05, from: "lecture_11:mup-two-conditions-and-init:check" },
    { args: { nIn: 1024, nOut: 4096 }, pick: "sp", expect: 0.03125 },                      // check's why: SP 1/sqrt(1024)
    { args: { nIn: 1024, nOut: 4096 }, pick: "boundExact", expect: 1 },                    // the exact σ returns exactly sqrt(n_l)
    { args: { nIn: 4096, nOut: 1024 }, pick: "boundSP", expect: 3 },                       // SP's bound at fan-out &lt; fan-in
    { args: { nIn: 1024, nOut: 1024 }, pick: "boundSP", expect: 2 },                       // square: SP's bound is a constant 2
    { args: { nIn: 65536, nOut: 256 }, pick: "boundSP", expect: 17 },                      // edge: 256:1 fan-in, SP's bound 1 + 16
    { args: { nIn: 65536, nOut: 256 }, pick: "boundTheta", expect: 17 / 16 },              // ... the Θ form stays within [1, 2]
    { args: { nIn: 256, nOut: 65536 }, pick: "spOverTheta", expect: 1 },                   // fan-out ≥ fan-in: muP Θ init = SP init
    { args: { nIn: 1024, nOut: 1024 }, pick: "typExact", expect: 0.5 },                    // typical gain of the exact σ, square layer
  ],
};
WIDGETS["fixture:lecture_11--mup-init"] = (root) => {
  const s = { li: 11, lo: 11 };
  const pic = el("div"), read = el("div", { class: "readout" });
  const draw = () => {
    const a = { nIn: 2 ** s.li, nOut: 2 ** s.lo }, m = mupInit(a);
    // plot worst-case per-coordinate size r against log2(n_l / n_{l−1}) for fixed n_{l−1}
    const W = 640, H = 240, px = 50, py = 14, pw = 400, ph = 190, xl = -8, xh = 8, yl = -1, yh = 5;
    const X = v => px + pw * (v - xl) / (xh - xl), Y = r => py + ph * (yh - Math.max(yl, Math.min(yh, Math.log2(r)))) / (yh - yl);
    const curve = (key, col, extra) => path(Array.from({ length: 161 }, (_, i) => { const lr = xl + (xh - xl) * i / 160; return [X(lr), Y(mupInit({ nIn: a.nIn, nOut: a.nIn * 2 ** lr })[key])]; }), col, extra);
    let b = rect(px, py, X(0) - px, ph, C.b, 'opacity="0.06"') + text(px + 4, py + 12, "fan-out &lt; fan-in", { fill: C.b, size: 10 }) + text(px + pw - 4, py + 12, "fan-out > fan-in", { fill: C.muted, size: 10, anchor: "end" });
    b += line(px, Y(1), px + pw, Y(1), C.muted, "stroke-dasharray:3 3") + text(px + pw + 4, Y(1) + 4, "target 1", { fill: C.muted, size: 10 });
    b += curve("boundSP", C.b, "stroke-width:2") + curve("boundTheta", C.a, "stroke-width:2;stroke-dasharray:6 3") + curve("boundExact", C.ok, "stroke-width:2");
    b += `<rect x="${px}" y="${py}" width="${pw}" height="${ph}" style="fill:none;stroke:${C.rule}"/>`;
    for (const v of [-8, -4, 0, 4, 8]) b += text(X(v), py + ph + 14, v === 0 ? "1" : v < 0 ? `1/${2 ** -v}` : `${2 ** v}`, { fill: C.muted, size: 10, anchor: "middle" });
    for (const v of [-1, 0, 1, 2, 3, 4, 5]) b += text(px - 6, Y(2 ** v) + 4, v < 0 ? "1/2" : `${2 ** v}`, { fill: C.muted, size: 10, anchor: "end" });
    b += text(px + pw / 2, H - 4, "n_l / n_{l−1} (fan-out / fan-in, log)", { fill: C.muted, size: 11, anchor: "middle" });
    const cx = Math.max(xl, Math.min(xh, s.lo - s.li));
    for (const [k, col] of [["boundSP", C.b], ["boundTheta", C.a], ["boundExact", C.ok]]) b += `<circle cx="${X(cx).toFixed(1)}" cy="${Y(m[k]).toFixed(1)}" r="4.5" style="fill:${col};stroke:${C.ink}"/>`;
    const lx = px + pw + 52;
    b += line(lx, 40, lx + 22, 40, C.b, "stroke-width:2") + text(lx + 26, 44, "SP 1/√n_{l−1}", { size: 11 });
    b += line(lx, 60, lx + 22, 60, C.a, "stroke-width:2;stroke-dasharray:6 3") + text(lx + 26, 64, "muP, Θ form", { size: 11 });
    b += line(lx, 80, lx + 22, 80, C.ok, "stroke-width:2") + text(lx + 26, 84, "muP, exact p47", { size: 11 });
    b += text(lx, 112, "y: worst-case size of", { fill: C.muted, size: 10 }) + text(lx, 125, "one output coordinate,", { fill: C.muted, size: 10 }) + text(lx, 138, "σ(√n_{l−1}+√n_l)·√n_{l−1}", { fill: C.muted, size: 10 }) + text(lx, 151, "  ÷ √n_l  (target 1)", { fill: C.muted, size: 10 });
    pic.innerHTML = svg(W, H, b);
    const regime = a.nOut < a.nIn ? `fan-out &lt; fan-in: muP shrinks the init by min(1, √(n_l/n_{l−1})) = ${fmt(1 / m.spOverTheta, 4)}; SP's bound grows like √(n_{l−1}/n_l)` : "fan-out ≥ fan-in: the Θ forms of muP and SP coincide (the min is 1); they differ only by the constant";
    read.innerHTML = `layer ${a.nIn} → ${a.nOut} (n_{l−1} → n_l)<br>
      init std σ: SP 1/√n_{l−1} = ${fmt(m.sp, 5)} · muP Θ form = ${fmt(m.theta, 5)} (SP / Θ = <b>${fmt(m.spOverTheta, 3)}</b>) · muP exact √(n_l/n_{l−1})/(√n_l+√n_{l−1}) = <span class="big">${fmt(m.exact, 5)}</span> (SP / exact = ${fmt(m.spOverExact, 3)})<br>
      worst-case output size per coordinate (operator-norm bound, input norm √n_{l−1}): SP ${fmt(m.boundSP, 3)} · Θ form ${fmt(m.boundTheta, 3)} · exact <b>${fmt(m.boundExact, 3)}</b><br>
      <b style="color:${a.nOut < a.nIn ? "var(--bad)" : "var(--ok)"}">${regime}</b><br>
      <span class="muted">typical Gaussian gain instead of the bound (σ√n_{l−1} per coordinate, author's check): SP ${fmt(m.typSP, 3)} · Θ ${fmt(m.typTheta, 3)} · exact ${fmt(m.typExact, 3)}. The p47 derivation controls the worst case, not the typical size.</span><br>
      <span class="muted small">provenance: fixture:lecture_11--mup-init · lecture_11.pdf:p46 (A1: activations Θ(1), norm √n), p47 (L319-L337: W_l ~ N(0, σ²I); operator norm ≈ σ(√n_{l−1}+√n_l); the chosen σ and its Θ form; "a kind of 'worst case' derivation – the ≈ is an upper bound"), p50 recap (L377-L399: SP init 1/√n_{l−1}; "init diffs when fanout n_l &lt; fanin"). The text layer drops the square roots; formulas follow the KP's reconstruction. Video 1:03:39-1:05:07 (the induction). The typical-gain line is the KP filtering note's author's check.</span>`;
  };
  root.append(el("div", { class: "widget" }, el("div", { class: "controls" },
    slider("fan-in n_{l−1}", 6, 16, s.li, 1, v => { s.li = v; draw(); }, v => `${2 ** v}`),
    slider("fan-out n_l", 6, 16, s.lo, 1, v => { s.lo = v; draw(); }, v => `${2 ** v}`),
    read), pic)); draw();
};
