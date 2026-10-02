// Widgets for thread lecture_03 (CS336 L3, architectures and hyperparameters). Register as "fixture:<id>" -> (root, notice) => void.
// Each widget has a pure model in MODELS (no DOM) that tools/check_widgets.mjs tests against the KPs' stored answers.
// Sources: lectures/lecture_03/lecture_03.txt (slide text layer, cited as pdf page + txt line), lectures/lecture_03/transcript.json
//          (cited as video M:SS), and the paper anchors the KPs already carry (PaLM, Gemma 2). Numbers the KP author derived are
//          marked "author". No constant here is new.
import { el, fmt, slider } from "../widgets_lib.js";

export const MODELS = {};
export const WIDGETS = {};

const C = { ink: "var(--ink)", muted: "var(--muted)", rule: "var(--rule)", a: "var(--accent)", b: "var(--accent2)", ok: "var(--ok)", hi: "var(--hilite)", bad: "var(--bad)" };
const rect = (x, y, w, h, fill, extra = "") => `<rect x="${x.toFixed(1)}" y="${y.toFixed(1)}" width="${Math.max(0, w).toFixed(1)}" height="${Math.max(0, h).toFixed(1)}" style="fill:${fill}" ${extra}/>`;
const text = (x, y, s, opts = {}) => `<text x="${x.toFixed(1)}" y="${y.toFixed(1)}" style="fill:${opts.fill || C.ink};font:${opts.weight || ""} ${opts.size || 12}px var(--sans)" text-anchor="${opts.anchor || "start"}">${s}</text>`;
const line = (x1, y1, x2, y2, col, extra = "") => `<line x1="${x1.toFixed(1)}" y1="${y1.toFixed(1)}" x2="${x2.toFixed(1)}" y2="${y2.toFixed(1)}" style="stroke:${col};stroke-width:1.6;${extra}"/>`;
const svg = (w, h, body) => `<svg viewBox="0 0 ${w} ${h}" width="100%" style="max-width:${w}px;border:1px solid var(--rule);border-radius:6px;background:#fff" role="img">${body}</svg>`;
const check = (label, value, onChange) => { const i = el("input", { type: "checkbox" }); i.checked = value; i.addEventListener("change", () => onChange(i.checked)); return el("label", {}, i, ` ${label}`); };
const choose = (label, opts, value, onChange) => {
  const sel = el("select", {}, ...opts.map(([v, t]) => { const o = el("option", { value: v }, t); if (v === value) o.selected = true; return o; }));
  sel.addEventListener("change", () => onChange(sel.value));
  return el("label", {}, `${label} `, sel);
};

// ============================================================ 1. residual path ==
// Where the norm sits in one layer, and how many norms the residual signal passes through across L layers.
//   lecture_03.pdf:p10 (txt L105-L110): "Set up LayerNorm so that it doesn't affect the main residual signal path"; pre-norm vs post-norm (BERT).
//   p13 (L126-L132): 'double' norm (Grok, Gemma 2) and non-residual post-norm (OLMo 2 only).
//   p27-p28 (L300-L315): serial blocks compute attention, then the MLP; parallel blocks (GPT-J, PaLM, GPT-NeoX): "LayerNorm can be shared".
//   p12, video 11:06-11:47: the clean residual path is why gradients propagate well; the widget shows wiring only, no gradient numbers.
// Counting: post-norm LN(x + F(x)) puts one norm on the residual path per sub-block (2 per layer); every other wiring puts 0 there.
const WIRINGS = {
  post:     { name: "post-norm (original transformer, BERT)", eq: "x ← LN(x + Attn(x));  x ← LN(x + MLP(x))", branch: ["F"], onLine: true, perLayer: 2 },
  pre:      { name: "pre-norm (LLaMA and most modern LMs)", eq: "x ← x + Attn(LN(x));  x ← x + MLP(LN(x))", branch: ["LN", "F"], perLayer: 2, finalNorm: true },
  double:   { name: "double norm (Grok, Gemma 2)", eq: "x ← x + LN(Attn(LN(x)));  x ← x + LN(MLP(LN(x)))", branch: ["LN", "F", "LN"], perLayer: 4, finalNorm: true },
  olmo2:    { name: "non-residual post-norm (OLMo 2)", eq: "x ← x + LN(Attn(x));  x ← x + LN(MLP(x))", branch: ["F", "LN"], perLayer: 2, finalNorm: true },
  parallel: { name: "parallel pre-norm (GPT-J, PaLM)", eq: "x ← x + Attn(LN(x)) + MLP(LN(x))   (one shared LN)", parallel: true, perLayer: 1, finalNorm: true },
};
function residualPath({ wiring, L }) {
  const w = WIRINGS[wiring];
  const onPathPerLayer = w.onLine ? 2 : 0;
  return { onPath: onPathPerLayer * L, perLayer: w.perLayer, applications: w.perLayer * L, clean: w.onLine ? 0 : 1, mlpSeesAttn: w.parallel ? 0 : 1 };
}
MODELS["fixture:lecture_03--residual-path"] = {
  fn: residualPath,
  cases: [
    { args: { wiring: "post", L: 4 }, pick: "onPath", expect: 8, from: "lecture_03:pre-norm-vs-post-norm:check" },
    { args: { wiring: "pre", L: 4 }, pick: "onPath", expect: 0 },          // pre-norm: 0 until the final norm
    { args: { wiring: "post", L: 48 }, pick: "onPath", expect: 96 },       // the 48-layer predict: the post-norm path is renormalised 96 times
    { args: { wiring: "double", L: 4 }, pick: "onPath", expect: 0 },       // non-residual-double-norm predict: "neither" norm is on the path
    { args: { wiring: "double", L: 1 }, pick: "perLayer", expect: 4 },     // ... although it has twice the norms per layer
    { args: { wiring: "olmo2", L: 4 }, pick: "onPath", expect: 0 },        // OLMo 2 x + LN(Attn(x)) keeps the path clean (check's lure)
    { args: { wiring: "parallel", L: 1 }, pick: "perLayer", expect: 1 },   // serial-vs-parallel predict: 1 shared LN ...
    { args: { wiring: "pre", L: 1 }, pick: "perLayer", expect: 2 },        // ... vs 2 in a serial pre-norm block
    { args: { wiring: "parallel", L: 1 }, pick: "mlpSeesAttn", expect: 0 }, // serial-vs-parallel check: the MLP loses this layer's attention output
    { args: { wiring: "pre", L: 1 }, pick: "mlpSeesAttn", expect: 1 },
  ],
};
function drawLayer(wk) {
  const w = WIRINGS[wk], X0 = 70, XB = 170, BW = 64, BH = 20;
  let b = text(X0, 326, "x (layer input)", { fill: C.muted, size: 11, anchor: "middle" }) + text(X0 + 10, 16, "to next layer", { fill: C.muted, size: 11 });
  b += line(X0, 314, X0, 14, C.ink, "stroke-width:3");                          // the residual stream
  const box = (cx, cy, lab, onLine) => rect(cx - BW / 2, cy - BH / 2, BW, BH, `${onLine ? C.bad : lab === "LN" ? C.a : "#f5f2ea"};stroke:${lab === "LN" || onLine ? "none" : C.rule}`, 'rx="4"') +
    text(cx, cy + 4, lab, { fill: lab === "LN" || onLine ? "#fff" : C.ink, size: 11, anchor: "middle", weight: "600" });
  const add = (y) => `<circle cx="${X0}" cy="${y}" r="8" style="fill:#fff;stroke:${C.ink};stroke-width:1.6"/>` + text(X0, y + 4, "+", { size: 13, anchor: "middle", weight: "700" });
  if (w.parallel) {
    const split = 290, join = 120, fy = 210;
    b += line(X0, split, XB, split, C.ink) + line(XB, split, XB, 240, C.ink) + box(XB, 255, "LN") + line(XB, 245, XB, 230, C.ink) + line(XB - 42, 230, XB + 42, 230, C.ink);
    for (const [dx, lab] of [[-42, "Attn"], [42, "MLP"]]) b += line(XB + dx, 230, XB + dx, fy + BH / 2, C.ink) + line(XB + dx, fy - BH / 2, XB + dx, join, C.ink) + box(XB + dx, fy, lab);
    b += line(XB - 42, join, X0, join, C.ink) + line(XB + 42, join, XB - 42, join, C.ink) + add(join);
    b += text(XB + 80, 258, "one LN feeds both branches", { fill: C.muted, size: 11 });
    return b;
  }
  for (const [split, addY, F] of [[285, 195, "Attn"], [150, 60, "MLP"]]) {
    const items = w.branch.map(x => (x === "F" ? F : x)), n = items.length, lo = split - 22, hi = addY + 22;
    b += line(X0, split, XB, split, C.ink) + line(XB, split, XB, addY, C.ink) + line(XB, addY, X0 + 8, addY, C.ink);
    items.forEach((lab, k) => { b += box(XB, n === 1 ? (split + addY) / 2 : lo - k * (lo - hi) / (n - 1), lab); });
    b += add(addY);
    if (w.onLine) b += box(X0, addY - 23, "LN", true) + text(X0 + BW / 2 + 6, addY - 19, "on the residual path", { fill: C.bad, size: 11 });
  }
  return b;
}
WIDGETS["fixture:lecture_03--residual-path"] = (root) => {
  const s = { wiring: "pre", L: 12 };
  const pic = el("div"), read = el("div", { class: "readout" });
  const draw = () => {
    const w = WIRINGS[s.wiring], m = residualPath(s);
    let b = drawLayer(s.wiring);
    // right: the residual stream through all L layers, one tick per norm the signal itself passes through
    const sx = 450, top = 46, bot = 300, h = (bot - top) / s.L;
    b += text(sx, 20, `residual stream through ${s.L} layers`, { fill: C.muted, size: 11, anchor: "middle" });
    b += rect(sx - 6, top, 12, bot - top, m.clean ? C.ok : C.rule, 'opacity="0.35"');
    for (let i = 0; i <= s.L; i++) b += line(sx - 14, bot - i * h, sx - 8, bot - i * h, C.muted, "stroke-width:1");
    if (!m.clean) for (let i = 0; i < s.L; i++) for (const f of [0.45, 0.95]) b += line(sx - 12, bot - (i + f) * h, sx + 12, bot - (i + f) * h, C.bad, "stroke-width:2");
    if (w.finalNorm) b += rect(sx - 20, top - 8, 40, 8, C.a) + text(sx + 26, top, "final norm (after the stack)", { fill: C.a, size: 10 });
    b += text(sx, bot + 16, "embedding", { fill: C.muted, size: 11, anchor: "middle" });
    b += text(sx + 24, (top + bot) / 2, m.clean ? "clean: 0 norms" : `${m.onPath} norms`, { fill: m.clean ? C.ok : C.bad, size: 13, weight: "600" });
    b += text(sx + 24, (top + bot) / 2 + 16, m.clean ? "skip path carries x unchanged" : "on the skip path", { fill: m.clean ? C.ok : C.bad, size: 11 });
    pic.innerHTML = svg(640, 334, b);
    read.innerHTML = `${w.name}: <code>${w.eq}</code><br>
      norms on the residual path, embedding → last block: <span class="big">${m.onPath}</span> ${m.clean ? "(the skip connection is never normalised)" : "(the sum itself is renormalised after every sub-block)"}<br>
      norm applications per layer: <b>${m.perLayer}</b> (${m.applications} in ${s.L} layers) · MLP reads this layer's attention output: <b>${m.mlpSeesAttn ? "yes (serial)" : "no (both branches read the same LN(x))"}</b><br>
      <span class="muted small">provenance: fixture:lecture_03--residual-path · lecture_03.pdf:p10 (L105-L110, keep LayerNorm off "the main residual signal path"; BERT post-norm), p13 (L126-L132, double norm in Grok and Gemma 2; OLMo 2 only non-residual post-norm), p27-p28 (L300-L315, serial vs parallel; "LayerNorm can be shared"); video 11:06-11:47 (a clean residual stream lets gradients propagate). Wiring and counts only: the gradient-size effect on p12 is a figure and is not simulated.</span>`;
  };
  root.append(el("div", { class: "widget" }, el("div", { class: "controls" },
    choose("wiring", Object.entries(WIRINGS).map(([k, w]) => [k, w.name]), s.wiring, v => { s.wiring = v; draw(); }),
    slider("layers L", 1, 48, s.L, 1, v => { s.L = v; draw(); }),
    read), pic)); draw();
};

// ============================================================ 2. layer budget ==
// One layer's parameters in units of d_model², split by matrix, with GQA, head width and the FFN ratio as controls.
//   Attention: W_Q and W_O are d × (h·d_head), W_K and W_V are d × (g·d_head) (lecture_03.pdf:p42-p43, L505-L521: the projection
//   width is num_heads × head_dim, which need not equal d_model; p61-p62: MQA/GQA keep h query heads and share g K/V heads).
//   So attention = (2h + 2g)·d_head / d in d² (= 2 + 2g/h when h·d_head = d; gqa-frees-ffn-budget statement, author).
//   FFN: 2 matrices (ReLU) or 3 (GLU, p22-p23) of d × d_ff; "GLU variants scale down by 2/3rd", d_ff = 8/3 d (p38, L451-L453).
//   Reference layer: full multi-head attention with the same heads + an 8 d² FFN (ReLU 4× or GLU 8/3×). "Spend the freed budget"
//   solves for the FFN ratio that keeps that total, the professor's reading of LLaMA-2's 3.5 (video 46:01-46:28; author arithmetic).
//   Decode step (p60-p61, video 1:19:05-1:19:28): K/V cache read per token ∝ 2·g·d_head per layer; q·k score FLOPs ∝ 2·h·d_head·n, no g.
function layerBudget({ d, h, g, hd, gated, dff, match = false }) {
  const mats = gated ? 3 : 2, width = h * hd;
  const q = width / d, o = width / d, k = g * hd / d, v = g * hd / d, attn = q + o + k + v;
  const ref = 4 * width / d + 8, matchRatio = (ref - attn) / mats;
  const ratio = match ? matchRatio : dff / d, dffEff = match ? matchRatio * d : dff;
  const ffn = mats * ratio, total = attn + ffn;
  return {
    q, k, attn, ffn, total, ref, freed: 4 * width / d - attn, matchRatio, matchDff: matchRatio * d, ratio,
    widthRatio: width / d, wq: d * width, ffnParams: mats * d * dffEff, gateGrowth: mats / 2, ffnVsRelu4: ffn / 8,
    kvShrink: h / g, kvPerTokLayer: 2 * g * hd, scorePerTokLayer: 2 * h * hd, layerParams: total * d * d,
  };
}
MODELS["fixture:lecture_03--layer-budget"] = {
  fn: layerBudget,
  cases: [
    { args: { d: 8192, h: 64, g: 8, hd: 128, gated: true, dff: 21888 }, pick: "attn", expect: 2.25, tol: 0.02, from: "lecture_03:gqa-frees-ffn-budget:predict" },
    { args: { d: 4096, h: 32, g: 8, hd: 128, gated: true, dff: 14336 }, pick: "matchRatio", expect: 3.1667, tol: 0.02, from: "lecture_03:gqa-frees-ffn-budget:check" },
    { args: { d: 8192, h: 64, g: 1, hd: 128, gated: true, dff: 21888 }, pick: "matchRatio", expect: 3.3229, tol: 0.01 },  // gqa transfer: MQA, about 3.32
    { args: { d: 8192, h: 64, g: 64, hd: 128, gated: true, dff: 21888 }, pick: "matchRatio", expect: 8 / 3 },             // edge: MHA frees nothing, 8/3
    { args: { d: 8192, h: 64, g: 64, hd: 128, gated: true, dff: 21888 }, pick: "freed", expect: 0 },
    { args: { d: 4096, h: 32, g: 32, hd: 128, gated: true, dff: 11008 }, pick: "gateGrowth", expect: 1.5, tol: 0.02, from: "lecture_03:gated-activations:predict" },
    { args: { d: 4096, h: 32, g: 32, hd: 128, gated: true, dff: 11008 }, pick: "ffnParams", expect: 135266304, tol: 0.02, from: "lecture_03:gated-activations:check" },
    { args: { d: 4096, h: 32, g: 32, hd: 128, gated: false, dff: 11008 }, pick: "ffnParams", expect: 90177536 },          // the 2-matrix lure, 9.0e7
    { args: { d: 1024, h: 128, g: 128, hd: 128, gated: false, dff: 65536 }, pick: "widthRatio", expect: 16, from: "lecture_03:head-dim-times-heads:predict" },
    { args: { d: 3072, h: 24, g: 24, hd: 256, gated: true, dff: 8192 }, pick: "wq", expect: 18874368, tol: 0.01, from: "lecture_03:head-dim-times-heads:check" },
    { args: { d: 2048, h: 32, g: 32, hd: 128, gated: true, dff: 5504 }, pick: "widthRatio", expect: 2 },                   // head-dim transfer: ratio 2
    { args: { d: 4096, h: 32, g: 8, hd: 128, gated: true, dff: 14336 }, pick: "kvShrink", expect: 4, from: "lecture_03:mqa-gqa-kv-cache:predict" },
    { args: { d: 4096, h: 32, g: 1, hd: 128, gated: true, dff: 14336 }, pick: "scorePerTokLayer", expect: 8192 },          // mqa check: score FLOPs ...
    { args: { d: 4096, h: 32, g: 32, hd: 128, gated: true, dff: 14336 }, pick: "scorePerTokLayer", expect: 8192 },         // ... are the same with g = h
    { args: { d: 4096, h: 32, g: 32, hd: 128, gated: true, dff: 16384, match: true }, pick: "matchDff", expect: 10923, tol: 0.02 }, // dff-ratio predict
    { args: { d: 4096, h: 32, g: 32, hd: 128, gated: true, dff: 10240 }, pick: "ffnVsRelu4", expect: 0.9375 },             // dff-ratio check: T5 v1.1 2.5
    { args: { d: 4096, h: 32, g: 8, hd: 128, gated: true, dff: 14336 }, pick: "ffnVsRelu4", expect: 1.3125 },              // dff-ratio transfer: Mistral 3.5, 1.31×
  ],
};
WIDGETS["fixture:lecture_03--layer-budget"] = (root) => {
  const s = { d: 4096, h: 32, g: 32, hd: 128, gated: true, r48: 128, match: false };   // default: MHA + SwiGLU 8/3 (12 d²)
  const pic = el("div"), read = el("div", { class: "readout" });
  const draw = () => {
    const g = Math.min(s.g, s.h), a = { d: s.d, h: s.h, g, hd: s.hd, gated: s.gated, dff: Math.round(s.r48 / 48 * s.d), match: s.match }, m = layerBudget(a);
    const W = 640, x0 = 150, span = 440, scale = span / Math.max(m.total, m.ref, 1) / 1.04;
    const seg = (x, y, v, col, lab, op = 1) => rect(x, y, v * scale, 24, `${col};stroke:#fff;stroke-width:1`, `opacity="${op}"`) + (v * scale > 26 ? text(x + v * scale / 2, y + 16, lab, { fill: "#fff", size: 11, anchor: "middle" }) : "");
    const bar = (y, title, attnParts, ffnV, mats) => {
      let x = x0, out = text(10, y + 16, title, { size: 12 });
      for (const [v, col, lab] of attnParts) { out += seg(x, y, v, col, lab); x += v * scale; }
      for (let i = 0; i < mats; i++) { out += seg(x, y, ffnV / mats, C.ok, ["W1", mats === 3 ? "V" : "W2", "W2"][i], 0.85); x += ffnV / mats * scale; }
      return out + text(Math.max(x, x0 + m.ref * scale) + 6, y + 16, `${fmt(attnParts.reduce((p, z) => p + z[0], 0) + ffnV, 3)} d²`, { size: 12 });
    };
    let b = text(10, 16, "one layer's parameters in units of d_model² · blue: Q, O · orange: K, V · green: FFN matrices", { fill: C.muted, size: 11 });
    b += bar(28, "reference (MHA)", [[m.q, C.a, "Q"], [m.q, C.a, "O"], [m.q, C.b, "K"], [m.q, C.b, "V"]], 8, s.gated ? 3 : 2);
    b += bar(66, `this layer (g = ${g})`, [[m.q, C.a, "Q"], [m.q, C.a, "O"], [m.k, C.b, "K"], [m.k, C.b, "V"]], m.ffn, s.gated ? 3 : 2);
    const xr = x0 + m.ref * scale;
    b += line(xr, 22, xr, 98, C.muted, "stroke-dasharray:4 3");
    if (m.freed > 1e-9) b += rect(x0 + 2 * m.q * scale + 2 * m.k * scale, 92, m.freed * scale, 5, C.b, 'opacity="0.5"') + text(x0 + (2 * m.q + 2 * m.k) * scale, 110, `freed by sharing K/V: ${fmt(m.freed, 3)} d²`, { fill: C.b, size: 11 });
    // decode-step bars: K/V bytes read vs q·k FLOPs, per token per layer, relative to MHA
    const y2 = 124, sp2 = 250;
    b += text(10, y2 + 12, "per decoded token, per layer, per cached position:", { fill: C.muted, size: 11 });
    b += text(10, y2 + 32, "K/V read (2·g·d_head)", { size: 12 }) + rect(x0 + 40, y2 + 20, sp2, 14, C.rule) + rect(x0 + 40, y2 + 20, sp2 * g / s.h, 14, C.b) + text(x0 + 46 + sp2, y2 + 32, `${fmt(m.kvPerTokLayer, 0)} (${m.kvShrink > 1 ? `${fmt(m.kvShrink, 2)}× smaller than MHA` : "same as MHA"})`, { size: 11 });
    b += text(10, y2 + 52, "q·k FLOPs (2·h·d_head)", { size: 12 }) + rect(x0 + 40, y2 + 40, sp2, 14, C.a) + text(x0 + 46 + sp2, y2 + 52, `${fmt(m.scorePerTokLayer, 0)} (unchanged by g)`, { size: 11 });
    pic.innerHTML = svg(W, 184, b);
    const regime = g === s.h ? "full multi-head attention: nothing freed" : g === 1 ? "multi-query attention (one shared K/V head)" : `grouped-query attention: ${s.h / g} query heads per K/V head`;
    read.innerHTML = `${regime}${s.h % g ? ` <span class="muted">(${g} does not divide ${s.h}: groups are uneven)</span>` : ""}<br>
      attention = (2h + 2g)·d_head / d = (2·${s.h} + 2·${g})·${s.hd} / ${s.d} = <b>${fmt(m.attn, 4)} d²</b> · h·d_head / d = <b>${fmt(m.widthRatio, 3)}</b> · W_Q = ${s.d} × ${s.h * s.hd} = ${fmt(m.wq, 0)} params<br>
      FFN = ${s.gated ? 3 : 2} × ${fmt(m.ratio, 4)} = <b>${fmt(m.ffn, 4)} d²</b> (d_ff = ${fmt(m.ratio * s.d, 0)}; ${fmt(m.ffnParams, 0)} params; ${s.gated ? "the gate's third matrix makes it 1.5× the 2-matrix FFN at this d_ff" : "no gate"}) · ${fmt(m.ffnVsRelu4, 4)}× a ReLU 4× FFN<br>
      layer total <span class="big">${fmt(m.total, 4)} d²</span> vs reference ${fmt(m.ref, 4)} d² · FFN ratio that spends the freed budget: <b>${fmt(m.matchRatio, 4)}</b>${s.gated ? ` (8/3 = 2.6667 at g = h)` : ""}<br>
      <span class="muted small">provenance: fixture:lecture_03--layer-budget · lecture_03.pdf:p22-p23 (GLU adds V), p38 (L451-L459, 8/3 d_model; PaLM 4, Mistral 7B and LLaMA-2 70B 3.5, LLaMA 2.68), p42-p43 (L505-L521, head_dim × heads need not equal d_model; T5 128 × 128 vs 1024), p60-p62 (MQA/GQA: fewer K/V heads); video 46:01-46:28 (LLaMA-2 multiplies 8/3 by "an arbitrary 1.33" because attention is cheaper), 1:19:05-1:19:28 (the n/d cache term). The (2h+2g)·d_head/d accounting and the budget-matching ratio are the KP author's. No biases, embeddings or norms counted.</span>`;
  };
  const ratioSl = slider("d_ff / d_model", 48, 288, s.r48, 1, v => { s.r48 = v; draw(); }, v => `${(v / 48).toFixed(4)}`);
  root.append(el("div", { class: "widget" }, el("div", { class: "controls" },
    slider("d_model", 512, 16384, s.d, 512, v => { s.d = v; draw(); }),
    slider("query heads h", 1, 128, s.h, 1, v => { s.h = v; draw(); }),
    slider("K/V heads g", 1, 128, s.g, 1, v => { s.g = v; draw(); }),
    slider("head_dim", 32, 512, s.hd, 32, v => { s.hd = v; draw(); }),
    check("gated FFN (SwiGLU: W1, V, W2)", s.gated, v => { s.gated = v; draw(); }),
    ratioSl,
    check("spend the freed budget on the FFN (hold the reference total; overrides the ratio)", s.match, v => { s.match = v; draw(); }),
    read), pic)); draw();
};

// ============================================================ 3. norm runtime share ==
// "FLOPS are not runtime" (lecture_03.pdf:p16, L164-L168). Spoken numbers (video 15:37-16:13): normalisation is "something like
// 0.17% of the total floating point operations" but "can be up to 25% of the runtime"; 16:54-17:12: for statistical normalisation
// "the majority of the workload is memory movement". The professor calls 25% "quite extreme ... tiny models" (17:18-17:31), so the
// baseline runtime share is a control. Time model (author, after lecture_02's max(FLOPs/peak, bytes/bandwidth) accounting):
// matmuls are compute-bound, time ∝ 1/FLOP-rate; norms take max(FLOP time, byte time), calibrated so the byte term gives the
// baseline share and the FLOP term is the 0.17% FLOP share run at the same FLOP rate.
function normRuntime({ timeShare = 0.25, flopShare = 0.0017, fx = 1, bx = 1, normFlops = 1, normBytes = 1 }) {
  const tm = (1 - timeShare) / fx;
  const tnFlop = flopShare / (1 - flopShare) * (1 - timeShare) * normFlops / fx;
  const tnByte = timeShare * normBytes / bx;
  const tn = Math.max(tnFlop, tnByte), total = tm + tn;
  return { share: tn / total, flopShareNow: flopShare * normFlops / (flopShare * normFlops + 1 - flopShare), total, tm, tn, tnFlop, tnByte, memoryBound: tnByte >= tnFlop ? 1 : 0, normTimeVsFlopTime: tnByte / tnFlop };
}
MODELS["fixture:lecture_03--norm-runtime"] = {
  fn: normRuntime,
  cases: [
    { args: {}, pick: "share", expect: 0.25 },                               // spoken baseline: 0.17% of FLOPs, 25% of runtime
    { args: {}, pick: "flopShareNow", expect: 0.0017 },
    { args: { fx: 2 }, pick: "share", expect: 0.4 },                         // rmsnorm check: 2× matmul FLOP/s, same bandwidth -> up
    { args: { fx: 2, bx: 2 }, pick: "share", expect: 0.25 },                 // both faster: share unchanged
    { args: { normFlops: 0.5 }, pick: "share", expect: 0.25 },               // halving the norm's FLOPs does not move its runtime
    { args: { normBytes: 0.5 }, pick: "share", expect: 1 / 7 },              // halving its bytes does
    { args: { timeShare: 0.10, flopShare: 0.01 }, pick: "share", expect: 0.10 }, // rmsnorm transfer: 1% of FLOPs, 10% of time
    { args: {}, pick: "memoryBound", expect: 1 },
  ],
};
WIDGETS["fixture:lecture_03--norm-runtime"] = (root) => {
  const s = { ts: 25, lfx: 0, lbx: 0, nf: 100, nb: 100 };
  const pic = el("div"), read = el("div", { class: "readout" });
  const draw = () => {
    const a = { timeShare: s.ts / 100, fx: 2 ** s.lfx, bx: 2 ** s.lbx, normFlops: s.nf / 100, normBytes: s.nb / 100 }, m = normRuntime(a), base = normRuntime({ timeShare: a.timeShare });
    const W = 640, x0 = 150, span = 440;
    const bar = (y, title, frac, col, lab) => text(10, y + 15, title, { size: 12 }) + rect(x0, y, span, 22, C.rule, 'opacity="0.5"') + rect(x0, y, Math.max(1, span * frac), 22, col) + text(x0 + Math.min(span * frac + 6, span - 60), y + 15, lab, { size: 12, weight: "600" });
    let b = text(10, 16, "normalisation's share of the step (the rest is matmuls)", { fill: C.muted, size: 11 });
    b += bar(26, "share of FLOPs", m.flopShareNow, C.a, `${fmt(100 * m.flopShareNow, 3)}%`);
    b += bar(58, "share of runtime", m.share, C.b, `${fmt(100 * m.share, 2)}%`);
    // step time, stacked: matmul time + norm time, against the baseline step = 1
    const sc = span / Math.max(1, m.total) / 1.02;
    b += text(10, 107, "step time", { size: 12 }) + rect(x0, 92, m.tm * sc, 22, C.a, 'opacity="0.8"') + rect(x0 + m.tm * sc, 92, m.tn * sc, 22, C.b);
    b += line(x0 + sc, 86, x0 + sc, 120, C.muted, "stroke-dasharray:4 3") + text(x0 + sc, 132, "baseline step", { fill: C.muted, size: 10, anchor: "middle" });
    b += text(x0, 132, `matmuls ${fmt(m.tm, 3)} + norms ${fmt(m.tn, 3)} = ${fmt(m.total, 3)}`, { size: 11 });
    pic.innerHTML = svg(W, 140, b);
    const dir = m.share > base.share + 1e-9 ? "up" : m.share < base.share - 1e-9 ? "down" : "same";
    read.innerHTML = `norm time = max(FLOP time ${fmt(m.tnFlop, 5)}, byte time ${fmt(m.tnByte, 4)}) → <b>${m.memoryBound ? `memory-bound: bytes set its time (${fmt(m.normTimeVsFlopTime, 0)}× the FLOP time)` : "compute-bound"}</b><br>
      runtime share <span class="big">${fmt(100 * m.share, 2)}%</span> vs ${fmt(100 * base.share, 2)}% at baseline: <b>${dir}</b> · FLOP share ${fmt(100 * m.flopShareNow, 3)}%<br>
      <span class="muted small">provenance: fixture:lecture_03--norm-runtime · lecture_03.pdf:p15 (L150-L161, "fewer operations (no mean calculation)"; "matrix multiplies are the vast majority of FLOPs"), p16 (L164-L173, "FLOPS are not runtime!"; "RMSNorm can still matter due to the importance of data movement"; Ivanov et al. 2023); video 15:37-16:13 (0.17% of FLOPs, up to 25% of runtime), 16:54-17:12 (norms are memory movement), 17:18-17:31 (25% is the extreme tiny-model case). The max(FLOP time, byte time) model is the author's; the slide's byte saving for RMSNorm is not quantified, so "norm bytes" is a free control.</span>`;
  };
  root.append(el("div", { class: "widget" }, el("div", { class: "controls" },
    slider("baseline norm runtime share (%)", 1, 25, s.ts, 1, v => { s.ts = v; draw(); }),
    slider("matmul FLOP/s", -1, 3, s.lfx, 1, v => { s.lfx = v; draw(); }, v => `${2 ** v}×`),
    slider("memory bandwidth", -1, 3, s.lbx, 1, v => { s.lbx = v; draw(); }, v => `${2 ** v}×`),
    slider("norm FLOPs (e.g. drop the mean pass)", 10, 100, s.nf, 10, v => { s.nf = v; draw(); }, v => `${v}%`),
    slider("norm bytes moved", 10, 100, s.nb, 10, v => { s.nb = v; draw(); }, v => `${v}%`),
    read), pic)); draw();
};

// ============================================================ 4. logit guards ==
// Two output-side stability tricks on the same logits.
//   z-loss: lecture_03.pdf:p53-p54 (L636-L655): softmaxes are ill-behaved through exponentials; the z-loss (Devlin 2014, PaLM)
//   penalises log Z, Z = Σ exp(z_i); PaLM's form α·log²Z with α = 1e-4 is from the PaLM anchor (arXiv:2204.02311 §2), not the slide.
//   Shift invariance: video 1:08:14-1:08:55 ("if I add a constant ... I can manipulate the z's without" changing the probabilities).
//   Soft-capping: p56 (L667-L670): "soft-capping the logits to some maximum value via Tanh ... might have perf issues?";
//   z -> c·tanh(z/c) and the caps 50 (attention) / 30 (final logits) are from the Gemma 2 anchor (arXiv:2408.00118 §2).
//   The base logits 5,5,5,5 are the z-loss check's own case; the spread pattern around them is a free control.
const lse = xs => { const m = Math.max(...xs); return m + Math.log(xs.reduce((p, x) => p + Math.exp(x - m), 0)); };
function logitGuards({ logits = [5, 5, 5, 5], shift = 0, c = 50, z1 = 0, z2 = 0 }) {
  const zs = logits.map(z => z + shift), logZ = lse(zs), probs = zs.map(z => Math.exp(z - logZ));
  const zeroShift = -lse(logits), logZAtZero = +lse(logits.map(z => z + zeroShift)).toFixed(9);
  const cap = z => c * Math.tanh(z / c), clip = z => Math.max(-c, Math.min(c, z));
  const y1 = cap(z1), y2 = cap(z2);
  return {
    logZ, penalty: logZ * logZ, palmTerm: 1e-4 * logZ * logZ, p0: probs[0], probs, zeroShift, logZAtZero,
    y1, y2, capGap: y2 - y1, rawGap: z2 - z1, clipGap: clip(z2) - clip(z1),
    pTopRaw: 1 / (1 + Math.exp(-(z2 - z1))), pTopCap: 1 / (1 + Math.exp(-(y2 - y1))), slope2: 1 - Math.tanh(z2 / c) ** 2,
  };
}
MODELS["fixture:lecture_03--z-loss"] = {
  fn: logitGuards,
  cases: [
    { args: {}, pick: "logZAtZero", expect: 0, from: "lecture_03:z-loss:predict" },   // the penalty's minimum sits at log Z = 0
    { args: { shift: 0 }, pick: "penalty", expect: 40.78, tol: 0.005 },               // z-loss check: (5 + ln 4)² ≈ 40.8 ...
    { args: { shift: -5 }, pick: "penalty", expect: 1.922, tol: 0.005 },              // ... falls to (ln 4)² ≈ 1.92 ("down")
    { args: { shift: -5 }, pick: "p0", expect: 0.25 },                                // ... with the probabilities still 1/4
  ],
};
MODELS["fixture:lecture_03--soft-cap"] = {
  fn: logitGuards,
  cases: [
    { args: { c: 50, z2: 200 }, pick: "y2", expect: 49.97, tol: 0.01, from: "lecture_03:logit-soft-capping:predict" },
    { args: { c: 50, z1: 100, z2: 400 }, pick: "capGap", expect: 1.7986, tol: 0.03, from: "lecture_03:logit-soft-capping:check" },
    { args: { c: 50, z1: 100, z2: 400 }, pick: "clipGap", expect: 0 },                 // a hard clip would leave no gap at all
    { args: { c: 30, z2: 3 }, pick: "y2", expect: 2.990, tol: 0.002 },                 // transfer: small logits nearly unchanged
    { args: { c: 30, z1: 300, z2: 600 }, pick: "y1", expect: 30, tol: 1e-6 },         // transfer: 300 maps to ≈ 30 ...
    { args: { c: 30, z1: 300, z2: 600 }, pick: "y2", expect: 30, tol: 1e-6 },         // ... and so does 600: their order is lost
  ],
};
const logitGuardsWidget = (mode) => (root) => {
  const s = { mode, shift: 0, spread: 1, c: 30, z1: 20, z2: 120 };
  const pic = el("div"), read = el("div", { class: "readout" }), zc = el("div"), sc = el("div");
  const W = 640, px = 56, pw = 540;
  const drawZ = () => {
    const logits = [1.5, 0.5, -0.5, -1.5].map(k => 5 + s.spread * k), m = logitGuards({ logits, shift: s.shift });
    const zs = logits.map(z => z + s.shift);
    // left: the shifted logits on a number line; right: the penalty log²Z with the current point
    const H = 210, ly = 30, lh = 150, Y = v => ly + lh * (12 - Math.max(-12, Math.min(12, v))) / 24;
    let b = text(10, 16, "logits after the shift (softmax probability under each bar)", { fill: C.muted, size: 11 });
    b += line(40, Y(0), 250, Y(0), C.muted, "stroke-dasharray:3 3") + text(36, Y(0) + 4, "0", { fill: C.muted, size: 10, anchor: "end" });
    for (const v of [-10, -5, 5, 10]) b += text(36, Y(v) + 4, `${v}`, { fill: C.muted, size: 10, anchor: "end" });
    zs.forEach((z, i) => { const x = 60 + i * 46, y0 = Y(0), y1 = Y(z); b += rect(x, Math.min(y0, y1), 30, Math.abs(y1 - y0), C.a, 'opacity="0.8"') + text(x + 15, ly + lh + 14, `p ${fmt(m.probs[i], 3)}`, { size: 10, anchor: "middle" }); });
    b += line(40, Y(m.logZ), 250, Y(m.logZ), C.b, "stroke-width:2") + text(252, Y(m.logZ) + 4, `log Z = ${fmt(m.logZ, 3)}`, { fill: C.b, size: 11 });
    const qx = 360, qw = 260, X = v => qx + qw * (v + 4) / 16, P = v => ly + lh - lh * Math.min(v, 150) / 150;
    let d = ""; for (let v = -4; v <= 12.001; v += 0.1) d += `${d ? "L" : "M"}${X(v).toFixed(1)},${P(v * v).toFixed(1)}`;
    b += text(qx, 16, "z-loss penalty (log Z)²", { fill: C.muted, size: 11 }) + `<path d="${d}" style="fill:none;stroke:${C.b};stroke-width:2"/>`;
    b += line(qx, ly + lh, qx + qw, ly + lh, C.rule) + line(X(0), ly, X(0), ly + lh, C.muted, "stroke-dasharray:3 3") + text(X(0), ly + lh + 14, "log Z = 0", { fill: C.muted, size: 10, anchor: "middle" });
    for (const v of [-4, 4, 8, 12]) b += text(X(v), ly + lh + 14, `${v}`, { fill: C.muted, size: 10, anchor: "middle" });
    if (m.logZ >= -4 && m.logZ <= 12) b += `<circle cx="${X(m.logZ).toFixed(1)}" cy="${P(m.penalty).toFixed(1)}" r="5" style="fill:${C.hi};stroke:${C.ink}"/>`;
    pic.innerHTML = svg(W, H, b);
    read.innerHTML = `logits ${zs.map(z => fmt(z, 2)).join(", ")} · log Z = log Σ exp(z) = <b>${fmt(m.logZ, 4)}</b> · penalty (log Z)² = <span class="big">${fmt(m.penalty, 4)}</span> (PaLM adds 1e-4 × this = ${fmt(m.palmTerm, 6)})<br>
      probabilities ${m.probs.map(p => fmt(p, 4)).join(", ")} <span class="muted">(the shift never changes them)</span> · the shift that makes log Z = 0: ${fmt(m.zeroShift, 4)}<br>
      <span class="muted small">provenance: fixture:lecture_03--z-loss · lecture_03.pdf:p53 (L637-L638, softmaxes "ill-behaved due to exponentials"), p54 (L640-L655, the output-softmax z-loss, Devlin 2014, PaLM); video 1:07:32-1:08:55 (log Z "might not be so OK"; adding a constant to every logit moves Z without changing the probabilities); α = 1e-4 from the PaLM paper anchor (§2), not the slide.</span>`;
  };
  const drawCap = () => {
    const m = logitGuards({ c: s.c, z1: s.z1, z2: s.z2 }), H = 230, py = 22, ph = 170, zmax = 600, ymax = 110;
    const X = z => px + pw * z / zmax, Y = y => py + ph - ph * Math.min(y, ymax) / ymax;
    let b = text(px, 14, "logit after the cap vs raw logit · blue: c·tanh(z/c) · orange: hard clip at c · dashed: unchanged", { fill: C.muted, size: 11 });
    b += line(px, py + ph, px + pw, py + ph, C.rule) + line(px, py, px, py + ph, C.rule);
    b += line(X(0), Y(0), X(ymax), Y(ymax), C.muted, "stroke-dasharray:4 3");
    b += line(X(0), Y(0), X(s.c), Y(s.c), C.b, "stroke-width:1.4;opacity:0.8") + line(X(s.c), Y(s.c), X(zmax), Y(s.c), C.b, "stroke-width:1.4;opacity:0.8");
    let d = ""; for (let z = 0; z <= zmax; z += 2) d += `${d ? "L" : "M"}${X(z).toFixed(1)},${Y(s.c * Math.tanh(z / s.c)).toFixed(1)}`;
    b += `<path d="${d}" style="fill:none;stroke:${C.a};stroke-width:2.4"/>`;
    for (const z of [0, 100, 200, 300, 400, 500, 600]) b += text(X(z), py + ph + 14, `${z}`, { fill: C.muted, size: 10, anchor: "middle" });
    for (const y of [0, 50, 100]) b += text(px - 6, Y(y) + 4, `${y}`, { fill: C.muted, size: 10, anchor: "end" });
    for (const [z, y, lab] of [[s.z1, m.y1, "z₁"], [s.z2, m.y2, "z₂"]]) b += line(X(z), Y(y), X(z), py + ph, C.hi, "stroke-dasharray:2 2") + `<circle cx="${X(z).toFixed(1)}" cy="${Y(y).toFixed(1)}" r="5" style="fill:${C.hi};stroke:${C.ink}"/>` + text(X(z) + 7, Y(y) + 16, lab, { size: 11 });
    b += text(px + pw / 2, H - 4, "raw logit z", { fill: C.muted, size: 11, anchor: "middle" });
    pic.innerHTML = svg(W, H, b);
    const hi = Math.max(s.z1, s.z2), lo = Math.min(s.z1, s.z2), regime = hi / s.c < 0.5 ? "both logits in the near-linear part: capping changes almost nothing" : lo / s.c > 2 ? "both logits saturated: the cap erases most of their gap" : "the larger logit is being squashed";
    read.innerHTML = `c = ${s.c}: z₁ ${s.z1} → <b>${fmt(m.y1, 4)}</b>, z₂ ${s.z2} → <b>${fmt(m.y2, 4)}</b> · gap ${fmt(m.rawGap, 2)} → <span class="big">${fmt(m.capGap, 4)}</span> (hard clip: ${fmt(m.clipGap, 4)}) · slope at z₂ ${fmt(m.slope2, 5)}<br>
      ${regime} · two-token softmax P(z₂): raw ${fmt(m.pTopRaw, 6)}, capped ${fmt(m.pTopCap, 6)}<br>
      <span class="muted small">provenance: fixture:lecture_03--soft-cap · lecture_03.pdf:p56 (L667-L670, soft-cap "via Tanh"; "prevents logits from blowing up, but also might have perf issues?"); video 1:13:28-1:13:41 (soft-capping alone loses quality: confident signals cannot be expressed); the form c·tanh(z/c) and caps 50 / 30 from the Gemma 2 anchor (arXiv:2408.00118 §2).</span>`;
  };
  const draw = () => (s.mode === "zloss" ? drawZ : drawCap)();
  zc.append(slider("shift every logit by", -12, 6, s.shift, 0.5, v => { s.shift = v; draw(); }), slider("spread (0 = all equal to 5)", 0, 3, s.spread, 0.5, v => { s.spread = v; draw(); }));
  sc.append(slider("cap c", 10, 100, s.c, 5, v => { s.c = v; draw(); }), slider("raw logit z₁", 0, 600, s.z1, 10, v => { s.z1 = v; draw(); }), slider("raw logit z₂", 0, 600, s.z2, 10, v => { s.z2 = v; draw(); }));
  root.append(el("div", { class: "widget" }, el("div", { class: "controls" }, s.mode === "zloss" ? zc : sc, read), pic)); draw();
};
WIDGETS["fixture:lecture_03--z-loss"] = logitGuardsWidget("zloss");
WIDGETS["fixture:lecture_03--soft-cap"] = logitGuardsWidget("softcap");
