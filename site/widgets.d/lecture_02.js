// Widgets for thread lecture_02 (CS336 L2, resource accounting).
// Register as "fixture:<id>" -> (root, notice) => void. Each widget's numeric model is a pure function in
// MODELS, tested by tools/check_widgets.mjs against the knowledge points' own answers.
// Every constant is from official/lectures/lecture_02.py (cited file:line) or derived from those lines.
import { el, fmt, slider } from "../widgets_lib.js";

export const MODELS = {};
export const WIDGETS = {};

// ------------------------------------------------------------------ constants --
const H100_BF16 = 1979e12 / 2;   // lecture_02.py:L74, L815 (dense = half the sparse datasheet figure)
const H100_FP32 = 67.5e12;       // lecture_02.py:L813
const H100_BW = 3.35e12;         // lecture_02.py:L351 (facts.py h100_bytes_per_sec)
const DTYPES = { bf16: { bytes: 2, peak: H100_BF16 }, fp32: { bytes: 4, peak: H100_FP32 } }; // L141-L151 sizes
const GELU_FLOPS = 20;           // lecture_02.py:L405 ("flops = 20 * n")

const NS = "http://www.w3.org/2000/svg";
const sv = (tag, attrs = {}, text) => {
  const e = document.createElementNS(NS, tag);
  for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v);
  if (text !== undefined) e.textContent = text;
  return e;
};
const C = { ink: "#25313d", muted: "#6b7680", rule: "#d9d3c7", accent: "#24668d", accent2: "#b8582a", hilite: "#e3b23c", ok: "#3c8d5a" };
const svgBox = (w, h) => sv("svg", { viewBox: `0 0 ${w} ${h}`, width: "100%", style: "border:1px solid var(--rule);border-radius:6px;background:#fff;max-width:100%" });
const radios = (name, opts, value, onChange) => el("div", { style: "margin-top:8px;font-size:13px" }, ...opts.map(([v, lab]) => {
  const inp = el("input", { type: "radio", name, value: v, onchange: () => onChange(v) });
  if (v === value) inp.checked = true;
  return el("label", { style: "display:inline-block;margin-right:12px;color:var(--ink)" }, inp, " ", lab);
}));
let uid = 0;

// ============================================================ backward-flops ==
// Forward and backward FLOPs of an L-layer linear network, in units of B·D² (lecture_02.py:L502-L556).
// Per layer h_out = h_in · w (D×D): forward 2BD² (L529). Backward: ∂L/∂h_in = G·wᵀ (L537) and
// ∂L/∂w = h_inᵀ·G (L540), each 2BD² (L543). Layer 1's input is data, so ∂L/∂x can be skipped.
function backwardFlops({ L = 2, skipInput = false, layer }) {
  L = Math.max(1, Math.round(L));
  const rows = [];
  for (let i = 1; i <= L; i++) rows.push({ fwd: 2, dIn: skipInput && i === 1 ? 0 : 2, dW: 2 });
  const fwd = rows.reduce((s, r) => s + r.fwd, 0), bwd = rows.reduce((s, r) => s + r.dIn + r.dW, 0);
  const li = Math.min(Math.max(1, layer ?? L), L);
  return { rows, fwd, bwd, total: fwd + bwd, six_nd: 6 * L, layer_bwd: rows[li - 1].dIn + rows[li - 1].dW, ratio: bwd / fwd };
}
MODELS["fixture:lecture_02--backward-flops"] = {
  fn: backwardFlops,
  cases: [
    { args: { L: 1 }, pick: "layer_bwd", expect: 4, from: "lecture_02:backward-2x-forward:predict" },
    { args: { L: 2, skipInput: true, layer: 1 }, pick: "layer_bwd", expect: 2, from: "lecture_02:backward-2x-forward:transfer" },
    { args: { L: 2 }, pick: "six_nd", expect: 12, from: "lecture_02:six-nd:predict" },
    { args: { L: 2 }, pick: "total", expect: 12 },                       // 6ND is exact when every input gradient is computed
    { args: { L: 2, skipInput: true }, pick: "total", expect: 10 },      // six-nd predict "why": exact count 10BD²
    { args: { L: 5 }, pick: "ratio", expect: 2 },                        // backward = 2× forward at any depth
  ],
};
WIDGETS["fixture:lecture_02--backward-flops"] = (root) => {
  const s = { L: 3, skipInput: false };
  const box = el("div", {}); const read = el("div", { class: "readout" });
  const draw = () => {
    const m = backwardFlops(s), u = 26, W = 420, rowH = 30, H = 46 + rowH * s.L + 30;
    const g = svgBox(W, H);
    g.append(sv("text", { x: 10, y: 18, "font-size": 12, fill: C.muted }, "each block = 2·B·D² FLOPs (one D×D matmul over the batch)"));
    const heads = [["forward", 92, C.accent], ["∂L/∂h_in", 92 + 3 * u, C.accent2], ["∂L/∂w", 92 + 6 * u, C.accent2]];
    for (const [t, x] of heads) g.append(sv("text", { x, y: 38, "font-size": 11, fill: C.muted }, t));
    m.rows.forEach((r, i) => {
      const y = 46 + i * rowH;
      g.append(sv("text", { x: 10, y: y + 18, "font-size": 12, fill: C.ink }, `layer ${i + 1}`));
      const blk = (x, on, col, lab) => {
        g.append(sv("rect", { x, y: y + 2, width: 2 * u - 4, height: rowH - 8, rx: 3, fill: on ? col : "#fff", stroke: on ? col : C.rule, "stroke-dasharray": on ? "" : "4 3", opacity: on ? 0.85 : 1 }));
        g.append(sv("text", { x: x + 4, y: y + 18, "font-size": 10, fill: on ? "#fff" : C.muted }, on ? lab : "skipped"));
      };
      blk(92, true, C.accent, "x·w");
      blk(92 + 3 * u, r.dIn > 0, C.accent2, "G·wᵀ (out)");
      blk(92 + 6 * u, true, C.accent2, "hᵀ·G (batch)");
    });
    g.append(sv("text", { x: 10, y: H - 10, "font-size": 11, fill: C.muted }, "(out) / (batch): the axis each backward matmul sums over"));
    box.replaceChildren(g);
    const exact = m.total === m.six_nd;
    read.innerHTML = `<span class="big">${m.total} B·D² per step</span>forward ${m.fwd} + backward ${m.bwd} (backward / forward = ${fmt(m.ratio, 2)})<br>
      6·B·N with N = ${s.L}·D² parameters: <b>${m.six_nd} B·D²</b> → ${exact ? "exact" : `6ND overcounts by ${m.six_nd - m.total} B·D² (no ∂L/∂x for the data)`}<br>
      <span class="muted small">lecture_02.py:L529 (forward), L537-L543 (two backward matmuls), L549-L553 (6ND)</span>`;
  };
  const cb = el("input", { type: "checkbox", onchange: e => { s.skipInput = e.target.checked; draw(); } });
  root.append(el("div", { class: "widget" }, el("div", { class: "controls" },
    slider("layers L", 1, 6, s.L, 1, v => { s.L = v; draw(); }),
    el("label", {}, cb, " layer 1's input x is data: skip ∂L/∂x"), read), box));
  draw();
};

// ====================================================== checkpoint-tradeoff ==
// Activation checkpointing (lecture_02.py:L751-L755, L770-L773). Counting convention of the KP:
// resident at the backward peak = stored checkpoints (one per segment of k layers) + the k layers
// recomputed inside the current segment. Each segment is re-run once, so recomputation = L layer-forwards
// (one extra forward pass). Store nothing: backprop through layer i rebuilds its input from the network
// input, about i layer-forwards, so L(L+1)/2 in total (L772, "recompute from the start for each layer").
function checkpointCost({ L = 16, k = 4, mode = "segments" }) {
  const recompNone = n => n * (n + 1) / 2;
  if (mode === "all" || (mode === "segments" && k <= 1)) return { peak: L, recompute: 0, extra_forwards: 0, growth_x2: 1, regime: "store all" };
  if (mode === "none") return { peak: 2, recompute: recompNone(L), extra_forwards: recompNone(L) / L, growth_x2: recompNone(2 * L) / recompNone(L), regime: "store nothing" };
  const kk = Math.min(k, L), stored = Math.ceil(L / kk);
  return { peak: stored + kk, stored, recompute: L, extra_forwards: 1, growth_x2: 2, regime: kk * kk < L ? "checkpoints dominate" : kk * kk > L ? "segment dominates" : "balanced (k = √L)" };
}
MODELS["fixture:lecture_02--checkpoint-tradeoff"] = {
  fn: checkpointCost,
  cases: [
    { args: { L: 64, k: 8 }, pick: "peak", expect: 16, from: "lecture_02:activation-checkpointing:predict" },
    { args: { L: 32, mode: "none" }, pick: "growth_x2", expect: 4, tol: 0.1, from: "lecture_02:activation-checkpointing:check" },
    { args: { L: 32, mode: "none" }, pick: "recompute", expect: 528 },   // check "why": 528 at L = 32
    { args: { L: 64, mode: "none" }, pick: "recompute", expect: 2080 },  // and 2080 at L = 64
    { args: { L: 64, k: 1 }, pick: "peak", expect: 64 },                 // store all: O(L), no recompute
    { args: { L: 64, k: 64 }, pick: "peak", expect: 65 },                // one segment = no saving
    { args: { L: 64, k: 8 }, pick: "recompute", expect: 64 },            // O(L) recomputation: one extra forward
  ],
};
WIDGETS["fixture:lecture_02--checkpoint-tradeoff"] = (root) => {
  const s = { L: 16, k: 2, mode: "segments" };
  const box = el("div", {}); const read = el("div", { class: "readout" });
  let kSlider;
  const draw = () => {
    const m = checkpointCost(s), W = 420, H = 250, x0 = 46, x1 = 404, y0 = 24, y1 = 170;
    const g = svgBox(W, H);
    // U-curve: peak resident layers vs checkpoint interval k (segments mode)
    const ks = Array.from({ length: s.L }, (_, i) => i + 1), peaks = ks.map(k => checkpointCost({ L: s.L, k }).peak);
    const pmax = Math.max(...peaks, s.L + 1);
    const X = k => x0 + (k - 1) / Math.max(1, s.L - 1) * (x1 - x0), Y = p => y1 - p / pmax * (y1 - y0);
    g.append(sv("line", { x1: x0, y1, x2: x1, y2: y1, stroke: C.rule }), sv("line", { x1: x0, y1: y0, x2: x0, y2: y1, stroke: C.rule }));
    g.append(sv("text", { x: x0, y: 14, "font-size": 11, fill: C.muted }, "peak resident layers vs checkpoint interval k"));
    g.append(sv("text", { x: x1 - 60, y: y1 + 14, "font-size": 11, fill: C.muted }, `k (1…${s.L})`));
    g.append(sv("line", { x1: x0, y1: Y(s.L), x2: x1, y2: Y(s.L), stroke: C.muted, "stroke-dasharray": "4 3" }));
    g.append(sv("text", { x: x0 + 4, y: Y(s.L) - 4, "font-size": 10, fill: C.muted }, `store all = ${s.L}`));
    const sq = Math.sqrt(s.L);
    g.append(sv("line", { x1: X(sq), y1: y0, x2: X(sq), y2: y1, stroke: C.ok, "stroke-dasharray": "2 3" }));
    g.append(sv("text", { x: X(sq) + 3, y: y0 + 10, "font-size": 10, fill: C.ok }, `√L = ${fmt(sq, 1)}`));
    g.append(sv("polyline", { points: ks.map(k => `${X(k)},${Y(peaks[k - 1])}`).join(" "), fill: "none", stroke: C.accent, "stroke-width": 2 }));
    if (s.mode === "segments") g.append(sv("circle", { cx: X(Math.min(s.k, s.L)), cy: Y(m.peak), r: 5, fill: C.accent2 }));
    // layer strip: which layers are kept after the forward pass
    const cw = (x1 - x0) / s.L;
    g.append(sv("text", { x: x0, y: 196, "font-size": 11, fill: C.muted }, "kept after forward (blue) · layers →"));
    for (let l = 0; l < s.L; l++) {
      const kept = s.mode === "all" || (s.mode === "segments" && (s.k <= 1 || l % s.k === 0));
      g.append(sv("rect", { x: x0 + l * cw, y: 204, width: Math.max(1, cw - 1.5), height: 22, fill: kept ? C.accent : "#f5f2ea", stroke: C.rule }));
    }
    box.replaceChildren(g);
    const head = s.mode === "segments" && s.k > 1 && s.k < s.L ? `${m.stored} checkpoints + ${Math.min(s.k, s.L)} recomputed = ` : "";
    read.innerHTML = `<span class="big">peak ${head}${m.peak} layers</span>
      recomputation: <b>${fmt(m.recompute, 0)}</b> layer-forwards (${fmt(m.extra_forwards, 2)} extra forward passes; one forward = 2ND, a third of 6ND)<br>
      doubling L multiplies recomputation by <b>×${fmt(m.growth_x2, 2)}</b><br>regime: <b>${m.regime}</b><br>
      <span class="muted small">lecture_02.py:L751-L755, L770-L773 (store all O(L); store none O(1) memory, O(L²) compute; every √L: O(√L) memory, O(L) recompute)</span>`;
    kSlider.querySelector("input").disabled = s.mode !== "segments";
  };
  const name = `ckpt-mode-${uid++}`;
  kSlider = slider("checkpoint every k layers", 1, 128, s.k, 1, v => { s.k = Math.min(v, s.L); draw(); });
  root.append(el("div", { class: "widget" }, el("div", { class: "controls" },
    slider("layers L", 4, 128, s.L, 4, v => {
      s.L = v; const ki = kSlider.querySelector("input"); ki.max = v;
      if (s.k > v) { s.k = v; ki.value = v; kSlider.querySelector("span").textContent = v; }
      draw();
    }),
    kSlider,
    radios(name, [["all", "store all"], ["segments", "every k"], ["none", "store nothing"]], s.mode, v => { s.mode = v; draw(); }), read), box));
  kSlider.querySelector("input").max = s.L;
  draw();
};

// ================================================================== roofline ==
// The lecture's five operations (lecture_02.py:L363-L467) on an H100 roofline (L384-L392).
// FLOPs and bytes follow the lecture lines; elementwise and dot intensities do not depend on n.
const OPS = {
  relu: n => ({ flops: n, elems: 2 * n }),                      // L368-L369: read x, write y; n comparisons
  gelu: n => ({ flops: GELU_FLOPS * n, elems: 2 * n }),         // L404-L405
  dot: n => ({ flops: 2 * n - 1, elems: 2 * n + 1 }),           // L424-L425
  matvec: n => ({ flops: n * (2 * n - 1), elems: n * n + 2 * n }), // L440-L441
  matmul: n => ({ flops: n * n * (2 * n - 1), elems: 3 * n * n }), // L455-L456
};
function roofline({ op = "matmul", n = 1024, dtype = "bf16", bw_mult = 1, ai }) {
  const d = DTYPES[dtype], kink = d.peak / (H100_BW * bw_mult);
  let a;
  if (op === "free") a = ai; else { const o = OPS[op](n); a = o.flops / (o.elems * d.bytes); }
  return { ai: a, kink, compute_bound: a > kink ? 1 : 0, mfu_bound: Math.min(1, a / kink), attainable: Math.min(d.peak, a * H100_BW * bw_mult) };
}
MODELS["fixture:lecture_02--roofline"] = {
  fn: roofline,
  cases: [
    { args: { op: "matmul", n: 1024 }, pick: "ai", expect: 341, from: "lecture_02:arithmetic-intensity:predict" },
    { args: { op: "free", ai: 30 }, pick: "mfu_bound", expect: 0.10, tol: 0.03, from: "lecture_02:roofline:predict" },
    { args: { op: "matvec", n: 8192 }, pick: "ai", expect: 1 },           // arithmetic-intensity transfer: ≈1 at any n
    { args: { op: "matvec", n: 8192 }, pick: "compute_bound", expect: 0 },// → memory-bound
    { args: {}, pick: "kink", expect: 295.4 },                           // L385: 9.9e14 / 3.35e12
    { args: { bw_mult: 2 }, pick: "kink", expect: 147.7 },               // roofline transfer: 2× bandwidth moves the kink left
    { args: { op: "matmul", n: 887 }, pick: "compute_bound", expect: 1 },// crossover: (2n−1)/6 passes 295 at n ≈ 887
    { args: { op: "matmul", n: 886 }, pick: "compute_bound", expect: 0 },
    { args: { op: "matmul", n: 1024, dtype: "fp32" }, pick: "kink", expect: 20.15 }, // L813: fp32 peak moves the kink
  ],
};
WIDGETS["fixture:lecture_02--roofline"] = (root) => {
  const s = { n: 256, dtype: "bf16", bw: 1, ai: 100 };
  const box = el("div", {}); const read = el("div", { class: "readout" });
  const draw = () => {
    const W = 440, H = 300, x0 = 52, x1 = 424, y0 = 18, y1 = 262, d = DTYPES[s.dtype];
    const lx = a => x0 + (Math.log10(Math.max(a, 0.01)) + 2) / 6 * (x1 - x0);   // 1e-2 … 1e4 FLOPs/byte
    const ly = f => y1 - (Math.log10(Math.max(f, 1e10)) - 10) / 6 * (y1 - y0);  // 1e10 … 1e16 FLOP/s
    const g = svgBox(W, H);
    g.append(sv("line", { x1: x0, y1, x2: x1, y2: y1, stroke: C.rule }), sv("line", { x1: x0, y1: y0, x2: x0, y2: y1, stroke: C.rule }));
    for (let e = -2; e <= 4; e++) g.append(sv("text", { x: lx(10 ** e) - 10, y: y1 + 14, "font-size": 10, fill: C.muted }, `1e${e}`));
    g.append(sv("text", { x: x1 - 170, y: H - 6, "font-size": 11, fill: C.muted }, "arithmetic intensity (FLOPs/byte)"));
    g.append(sv("text", { x: 4, y: y0 + 4, "font-size": 10, fill: C.muted }, "FLOP/s"));
    const roof = (bw, col, dash) => {
      const k = d.peak / bw;
      g.append(sv("polyline", { points: `${lx(0.01)},${ly(0.01 * bw)} ${lx(k)},${ly(d.peak)} ${lx(1e4)},${ly(d.peak)}`, fill: "none", stroke: col, "stroke-width": dash ? 1.5 : 2.5, "stroke-dasharray": dash ? "5 4" : "" }));
      return k;
    };
    if (s.bw !== 1) roof(H100_BW, C.muted, true);
    const kink = roof(H100_BW * s.bw, C.ink, false);
    g.append(sv("text", { x: lx(kink) + 4, y: ly(d.peak) + 16, "font-size": 11, fill: C.ink }, `kink ${fmt(kink, 0)}`));
    const rows = [];
    const pts = [...Object.keys(OPS).map(op => [op, roofline({ op, n: s.n, dtype: s.dtype, bw_mult: s.bw })]), ["your op", roofline({ op: "free", ai: s.ai, dtype: s.dtype, bw_mult: s.bw })]];
    pts.forEach(([name, r], i) => {
      const col = name === "your op" ? C.hilite : r.compute_bound ? C.ok : C.accent2;
      g.append(sv("circle", { cx: lx(r.ai), cy: ly(r.attainable), r: name === "your op" ? 6 : 4.5, fill: col, stroke: "#fff" }));
      g.append(sv("text", { x: lx(r.ai) + 7, y: ly(r.attainable) + 14 + (i % 2) * 10, "font-size": 10, fill: C.ink }, name));
      rows.push(`${name.padEnd(8)} AI ${fmt(r.ai, 1).padStart(7)}  ${r.compute_bound ? "compute" : "memory "}-bound  MFU ≤ ${r.mfu_bound.toFixed(2)}`);
    });
    box.replaceChildren(g);
    read.innerHTML = `<span class="big">kink = ${fmt(kink, 1)} FLOPs/byte</span>= peak ${fmt(d.peak, 2)} FLOP/s ÷ ${fmt(H100_BW * s.bw, 2)} B/s<pre style="font-size:12px;white-space:pre">${rows.join("\n")}</pre>
      <span class="muted small">relu/gelu/dot/matvec/matmul: lecture_02.py:L363-L467 · H100: L74, L351, L813 · MFU ≤ min(1, AI ÷ kink)</span>`;
  };
  const name = `roof-dtype-${uid++}`;
  root.append(el("div", { class: "widget" }, el("div", { class: "controls" },
    slider("matrix size n (matvec, matmul)", 16, 8192, s.n, 16, v => { s.n = v; draw(); }),
    slider("your op's intensity (log)", -1, 3, Math.log10(s.ai), 0.05, v => { s.ai = 10 ** v; draw(); }, v => fmt(10 ** v, 1)),
    slider("memory bandwidth × H100", 0.5, 4, s.bw, 0.5, v => { s.bw = v; draw(); }, v => `×${v}`),
    radios(name, [["bf16", "bf16 (2 B, 9.9e14)"], ["fp32", "fp32 (4 B, 6.75e13)"]], s.dtype, v => { s.dtype = v; draw(); }), read), box));
  draw();
};

// ======================================================================= mfu ==
// MFU = actual FLOP/s ÷ promised FLOP/s for the dtype (lecture_02.py:L319-L330, promised: L813-L815).
// Bands from the professor (video 37:52-38:21): ~0.5 good, ~0.8 a bare matmul, ~0.1 something is wrong.
function mfu({ flops, time_us, dtype = "bf16" }) {
  const actual = flops / (time_us * 1e-6), promised = DTYPES[dtype].peak;
  return { actual, promised, mfu: actual / promised };
}
MODELS["fixture:lecture_02--mfu"] = {
  fn: mfu,
  cases: [
    { args: { flops: 3.4e10, time_us: 50, dtype: "bf16" }, pick: "mfu", expect: 0.69, from: "lecture_02:mfu:predict" },
    { args: { flops: 1.1e12, time_us: 20, dtype: "bf16" }, pick: "mfu", expect: 56, from: "lecture_02:gpu-timing-synchronize:transfer" },
    { args: { flops: 3.4e10, time_us: 600, dtype: "fp32" }, pick: "mfu", expect: 0.84 }, // mfu transfer: 12× slower in fp32, MFU goes up
    { args: { flops: 3.4e10, time_us: 50, dtype: "bf16" }, pick: "actual", expect: 6.8e14 },
  ],
};
WIDGETS["fixture:lecture_02--mfu"] = (root) => {
  const s = { flops: 3.4e10, t: 200, dtype: "bf16" };
  const box = el("div", {}); const read = el("div", { class: "readout" });
  const draw = () => {
    const m = mfu({ flops: s.flops, time_us: s.t, dtype: s.dtype }), W = 420, H = 120, x0 = 20, x1 = 400;
    const g = svgBox(W, H), X = v => x0 + Math.min(v, 1.2) / 1.2 * (x1 - x0);
    g.append(sv("rect", { x: x0, y: 40, width: x1 - x0, height: 26, fill: "#f5f2ea", stroke: C.rule }));
    g.append(sv("rect", { x: X(1), y: 40, width: x1 - X(1), height: 26, fill: "#f3d9cc" }));
    g.append(sv("rect", { x: x0, y: 40, width: Math.max(1, X(m.mfu) - x0), height: 26, fill: m.mfu > 1 ? C.accent2 : C.accent, opacity: 0.85 }));
    for (const [v, t] of [[0.1, "0.1 wrong"], [0.5, "0.5 good"], [0.8, "0.8 bare matmul"], [1, "1 = promised"]]) {
      g.append(sv("line", { x1: X(v), y1: 34, x2: X(v), y2: 72, stroke: C.ink, "stroke-width": v === 1 ? 2 : 1 }));
      g.append(sv("text", { x: X(v) - 14, y: v === 0.8 ? 90 : 28, "font-size": 10, fill: C.ink }, t));
    }
    g.append(sv("text", { x: x0, y: 110, "font-size": 11, fill: C.muted }, m.mfu > 1.2 ? `MFU ${fmt(m.mfu, 1)}: off the scale` : "MFU"));
    box.replaceChildren(g);
    const verdict = m.mfu > 1 ? `<b style="color:${C.accent2}">impossible (&gt; 1): the timer is wrong, e.g. no torch.cuda.synchronize()</b>` : m.mfu >= 0.5 ? "good (≥ 0.5)" : m.mfu <= 0.15 ? "something is wrong (≈ 0.1)" : "below the 0.5 the professor calls good";
    read.innerHTML = `<span class="big">MFU = ${fmt(m.mfu, 3)}</span>actual ${fmt(s.flops, 2)} FLOPs ÷ ${fmt(s.t, 1)} µs = ${fmt(m.actual, 2)} FLOP/s<br>promised (${s.dtype}): ${fmt(m.promised, 2)} FLOP/s<br>${verdict}<br>
      <span class="muted small">lecture_02.py:L319-L330, L813-L815 · bands: video 37:52-38:21</span>`;
  };
  const name = `mfu-dtype-${uid++}`;
  const fl = el("input", { type: "number", value: s.flops, step: "any", style: "max-width:180px", oninput: e => { const v = +e.target.value; if (v > 0) { s.flops = v; draw(); } } });
  root.append(el("div", { class: "widget" }, el("div", { class: "controls" },
    el("label", {}, "FLOPs done by the kernel ", fl),
    slider("measured time (µs, log)", 0, 5, Math.log10(s.t), 0.01, v => { s.t = 10 ** v; draw(); }, v => fmt(10 ** v, 1)),
    radios(name, [["bf16", "bf16 promised 9.9e14"], ["fp32", "fp32 promised 6.75e13"]], s.dtype, v => { s.dtype = v; draw(); }), read), box));
  draw();
};

// ========================================================= elementwise-share ==
// h = x·W (B×D · D×D) then GELU. FLOPs: matmul 2·B·D² (L314), GELU 20 per element (L405).
// Time: each op as max(FLOPs ÷ peak, bytes ÷ bandwidth) (L371-L376), unfused, bf16 bytes (L368, L455).
function elementwiseShare({ D, B = 4096 }) {
  const fG = GELU_FLOPS * B * D, fM = 2 * B * D * D;
  const tM = Math.max(fM / H100_BF16, 2 * (B * D + D * D + B * D) / H100_BW);
  const tG = Math.max(fG / H100_BF16, 2 * (2 * B * D) / H100_BW);
  return { flop_share: fG / (fG + fM), time_share: tG / (tG + tM), matmul_compute_bound: fM / H100_BF16 >= 2 * (2 * B * D + D * D) / H100_BW ? 1 : 0 };
}
MODELS["fixture:lecture_02--elementwise-share"] = {
  fn: elementwiseShare,
  cases: [
    { args: { D: 4096 }, pick: "flop_share", expect: 0.0024, tol: 0.03, from: "lecture_02:elementwise-flops-negligible:predict" },
    { args: { D: 64 }, pick: "flop_share", expect: 0.135, from: "lecture_02:elementwise-flops-negligible:transfer" },
    { args: { D: 8192 }, pick: "flop_share", expect: 0.00122, tol: 0.02 },  // check: doubling D roughly halves the share
    { args: { D: 4096 }, pick: "time_share", expect: 0.126, tol: 0.03 },    // the caveat: memory-bound GELU is ~13% of time
    { args: { D: 4096 }, pick: "matmul_compute_bound", expect: 1 },
    { args: { D: 64 }, pick: "matmul_compute_bound", expect: 0 },
  ],
};
WIDGETS["fixture:lecture_02--elementwise-share"] = (root) => {
  const s = { lg: 10 };
  const box = el("div", {}); const read = el("div", { class: "readout" });
  const draw = () => {
    const D = 2 ** s.lg, m = elementwiseShare({ D }), W = 420, H = 130, x0 = 120, x1 = 400;
    const g = svgBox(W, H);
    const bar = (y, v, lab, col) => {
      g.append(sv("text", { x: 10, y: y + 16, "font-size": 12, fill: C.ink }, lab));
      g.append(sv("rect", { x: x0, y, width: x1 - x0, height: 22, fill: "#f5f2ea", stroke: C.rule }));
      g.append(sv("rect", { x: x0, y, width: Math.max(1, v * (x1 - x0)), height: 22, fill: col }));
      g.append(sv("text", { x: x0 + Math.min(v * (x1 - x0) + 4, x1 - 50), y: y + 16, "font-size": 12, fill: C.ink }, `${fmt(100 * v, 2)}%`));
    };
    g.append(sv("text", { x: 10, y: 16, "font-size": 11, fill: C.muted }, "GELU's share of the layer (rest: the matmul)"));
    bar(30, m.flop_share, "of FLOPs", C.accent);
    bar(70, m.time_share, "of time (est.)", C.accent2);
    g.append(sv("text", { x: 10, y: 120, "font-size": 10, fill: C.muted }, "time = max(FLOPs/peak, bytes/bandwidth) per op, unfused"));
    box.replaceChildren(g);
    read.innerHTML = `<span class="big">D = ${D}: ${fmt(100 * m.flop_share, 2)}% of FLOPs</span>20·B·D ÷ (20·B·D + 2·B·D²) = 20 ÷ (20 + 2·${D})<br>
      but ${fmt(100 * m.time_share, 1)}% of the time: GELU is memory-bound; the matmul is <b>${m.matmul_compute_bound ? "compute" : "memory"}-bound</b> at this D<br>
      <span class="muted small">batch B = 4096 · lecture_02.py:L314, L405, L371-L376 · H100 L74, L351 · video 32:31-32:58</span>`;
  };
  root.append(el("div", { class: "widget" }, el("div", { class: "controls" },
    slider("hidden size D (log₂)", 4, 14, s.lg, 1, v => { s.lg = v; draw(); }, v => `2^${v} = ${2 ** v}`), read), box));
  draw();
};
