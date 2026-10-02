// Widgets for thread lecture_10 (CS336 L10, inference). Register as "fixture:<id>" -> (root, notice) => void.
// Each widget has a pure model in MODELS (no DOM) that tools/check_widgets.mjs tests against the KPs' stored answers.
// Sources: official/lectures/lecture_10.py (cited as Lnnn; the lecture's own sympy formulas, re-evaluated here),
//          lectures/lecture_10/transcript.json (cited as video M:SS). Numbers marked "author" are compositions of
//          lecture formulas that the lecture itself does not write down; their KP filtering notes record that.
import { el, fmt, slider } from "../widgets_lib.js";

export const MODELS = {};
export const WIDGETS = {};

const C = { ink: "var(--ink)", muted: "var(--muted)", rule: "var(--rule)", a: "var(--accent)", b: "var(--accent2)", ok: "var(--ok)", hi: "var(--hilite)" };
const rect = (x, y, w, h, fill, extra = "") => `<rect x="${x.toFixed(1)}" y="${(+y).toFixed(1)}" width="${Math.max(0, w).toFixed(1)}" height="${Math.max(0, h).toFixed(1)}" style="fill:${fill}" ${extra}/>`;
const text = (x, y, s, o = {}) => `<text x="${x.toFixed(1)}" y="${(+y).toFixed(1)}" style="fill:${o.fill || C.ink};font:${o.weight || ""} ${o.size || 12}px var(--sans)" text-anchor="${o.anchor || "start"}">${s}</text>`;
const line = (x1, y1, x2, y2, stroke, extra = "") => `<line x1="${x1.toFixed(1)}" y1="${y1.toFixed(1)}" x2="${x2.toFixed(1)}" y2="${y2.toFixed(1)}" style="stroke:${stroke};${extra}"/>`;
const svg = (w, h, body) => `<svg viewBox="0 0 ${w} ${h}" width="100%" style="max-width:${w}px;border:1px solid var(--rule);border-radius:6px;background:#fff" role="img">${body}</svg>`;
const setRange = (lab, v, fmtv = x => x) => { const i = lab.querySelector("input"), o = lab.querySelector(".readout"); i.value = v; o.textContent = fmtv(+i.value); };
function buttons(label, items, value, on) {   // a radio-like button row; returns { node, set }
  const bs = items.map(([v, t]) => el("button", { class: "ghost small", onclick: () => { set(v); on(v); } }, t));
  const set = v => bs.forEach((b, i) => { b.style.fontWeight = items[i][0] === v ? "700" : "400"; b.style.borderColor = items[i][0] === v ? "var(--accent)" : ""; });
  set(value);
  return { node: el("div", { style: "margin:6px 0" }, el("span", { class: "small muted" }, `${label} `), ...bs.flatMap(b => [b, " "])), set };
}
const tag = (txt, good) => `<b style="color:${good ? "var(--ok)" : "var(--bad)"}">${txt}</b>`;

// Shared lecture constants.
const H100 = { flops: 989e12, bw: 3.35e12, mem: 80e9 };                  // L147-L148 (also L327), L351
const ACC = H100.flops / H100.bw;                                          // L149-L150: accelerator intensity ≈ 295
const LL13 = { S: 1024, D: 5120, F: 13824, N: 40, K: 40, H: 128, L: 40, V: 32000 }; // llama2_13b_config, L317-L327
// num_params = 2VD + 3DFL + (2DNH + 2DKH)L (L291) = 13,015,449,600 for Llama 2 13B; parameter_size = 2 · num_params (L294).
const paramsOf = ({ V, D, F, N, K, H, L }) => 2 * V * D + D * F * 3 * L + (2 * D * N * H + 2 * D * K * H) * L;
const LL13_PARAMS = paramsOf(LL13);
const kvPerToken = ({ K, H, L }, bytes = 2) => K * H * L * 2 * bytes;      // L297 without the S: 2 for key + value, 2 for bf16

// ======================================================== 1. arithmetic intensity: prefill vs decode ==
// MLP (L187-L211): FLOPs 6BTDF, bytes 4BTD + 4BTF + 6DF; limit B·T when B·T ≪ D, F.
// Attention (L226-L240): FLOPs 4BSTD, bytes 4BSD + 4BTD; intensity ST/(S+T) exactly, with no B.
// Prefill T = S gives S/2 (L245); generation T = 1 gives S/(S+1) < 1 (L248). Compute-bound iff intensity > 295 (L152-L154).
function intensity({ B, T, S, D = LL13.D, F = LL13.F }) {
  const mlp = 6 * B * T * D * F / (4 * B * T * D + 4 * B * T * F + 6 * D * F);
  const attn = 4 * B * S * T * D / (4 * B * S * D + 4 * B * T * D);
  return { mlp, mlpLimit: B * T, attn, acc: ACC, mlpCompute: mlp > ACC ? 1 : 0, attnCompute: attn > ACC ? 1 : 0 };
}
MODELS["fixture:lecture_10--intensity"] = {
  fn: intensity,
  cases: [
    { args: { B: 64, T: 1, S: 1024 }, pick: "mlpLimit", expect: 64, tol: 0.05, from: "lecture_10:mlp-intensity-batch:transfer" },
    { args: { B: 32, T: 1, S: 1024 }, pick: "mlpLimit", expect: 32, from: "lecture_10:mlp-intensity-batch:predict" },
    { args: { B: 32, T: 1, S: 1024, F: 55296 }, pick: "mlpLimit", expect: 32 },              // predict, after: F × 4 leaves the limit at 32
    { args: { B: 32, T: 1, S: 1024 }, pick: "mlp", expect: 31.82, tol: 0.002 },               // exact, F = 13824
    { args: { B: 32, T: 1, S: 1024, F: 55296 }, pick: "mlp", expect: 31.85, tol: 0.002 },     // exact, F × 4: up 0.1%, the "same" of the limit
    { args: { B: 1, T: 1, S: 1024 }, pick: "mlp", expect: 1, tol: 0.001 },                    // L156-L157: matrix-vector product, intensity 1
    { args: { B: 1, T: 4096, S: 4096 }, pick: "attn", expect: 2048, tol: 0.01, from: "lecture_10:attention-decode-intensity:transfer" },
    { args: { B: 128, T: 1, S: 4096 }, pick: "attn", expect: 4096 / 4097, tol: 0.01, from: "lecture_10:attention-decode-intensity:predict" },
    { args: { B: 8, T: 1, S: 4096 }, pick: "attn", expect: 4096 / 4097 },                     // ...the same at B = 8
    { args: { B: 1, T: 1, S: 1024 }, pick: "acc", expect: 295.22, tol: 0.001 },               // L150
    { args: { B: 512, T: 1, S: 1024 }, pick: "mlpCompute", expect: 1 },                       // decode MLP crosses 295 only at B ≈ 312 exact (296 in the limit)
    { args: { B: 300, T: 1, S: 1024 }, pick: "mlpCompute", expect: 0 },                       // limit 300 > 295, exact 285 < 295: still memory-bound
    { args: { B: 1, T: 1024, S: 1024 }, pick: "mlpCompute", expect: 1 },                      // prefill of one 1024-token prompt: compute-bound
  ],
};
WIDGETS["fixture:lecture_10--intensity"] = (root) => {
  const s = { logB: 2, logT: 0, logS: 10, Fx: 1 };
  const pic = el("div"), read = el("div", { class: "readout" });
  const fB = v => `${2 ** v}`;
  let sT;
  const draw = () => {
    const S = 2 ** s.logS, T = Math.min(2 ** s.logT, S), B = 2 ** s.logB, F = LL13.F * s.Fx, m = intensity({ B, T, S, F });
    const W = 640, x0 = 120, span = 490, lo = -1, hi = 5;                   // log10 intensity axis 0.1 .. 1e5
    const X = v => x0 + span * (Math.log10(Math.max(v, 10 ** lo)) - lo) / (hi - lo);
    let b = rect(x0, 26, X(ACC) - x0, 104, C.b, 'opacity="0.07"') + rect(X(ACC), 26, x0 + span - X(ACC), 104, C.ok, 'opacity="0.07"');
    b += text(x0 + 4, 40, "memory-bound", { fill: C.b, size: 11 }) + text(x0 + span - 4, 40, "compute-bound", { fill: C.ok, size: 11, anchor: "end" });
    b += line(X(ACC), 22, X(ACC), 132, C.ink, "stroke-dasharray:4 3") + text(X(ACC), 16, `H100: 989e12 / 3.35e12 = ${fmt(ACC, 0)} FLOPs/byte`, { size: 11, anchor: "middle" });
    const row = (y, name, sub, v, lim) => {
      let o = text(10, y + 4, name, { size: 12, weight: "600" }) + text(10, y + 18, sub, { fill: C.muted, size: 10 });
      o += rect(x0, y - 8, X(v) - x0, 18, v > ACC ? C.ok : C.b, 'opacity="0.85"') + text(Math.max(X(v), lim === undefined ? 0 : X(lim) + 5) + 6, y + 6, fmt(v, 3), { size: 12 });
      if (lim !== undefined) o += `<circle cx="${X(lim).toFixed(1)}" cy="${y + 1}" r="5" style="fill:none;stroke:${C.ink};stroke-width:1.5"/>`;
      return o;
    };
    b += row(64, "MLP", "shared weights", m.mlp, m.mlpLimit) + row(108, "attention", "own KV cache", m.attn);
    for (let e = lo; e <= hi; e++) b += text(X(10 ** e), 148, `${10 ** e}`, { fill: C.muted, size: 10, anchor: "middle" });
    b += text(x0 + span / 2, 162, "arithmetic intensity, FLOPs per byte (log scale) · ○ = the professor's limit B·T", { fill: C.muted, size: 10, anchor: "middle" });
    pic.innerHTML = svg(W, 170, b);
    const stage = T === 1 ? "generation (T = 1)" : T === S ? "prefill (T = S)" : `T = ${T} new tokens per sequence`;
    read.innerHTML = `${stage} · B = ${B} concurrent sequences · S = ${S} context · D = ${LL13.D}, F = ${F}${s.Fx > 1 ? ` (${s.Fx}× Llama 2 13B)` : ""}<br>
      MLP: 6BTDF / (4BTD + 4BTF + 6DF) = <span class="big">${fmt(m.mlp, 3)}</span> (limit B·T = ${m.mlpLimit}) ${tag(m.mlp > ACC ? "compute-bound" : "memory-bound", m.mlp > ACC)}<br>
      attention: ST / (S + T) = <span class="big">${fmt(m.attn, 4)}</span> ${tag(m.attn > ACC ? "compute-bound" : "memory-bound", m.attn > ACC)} · B does not appear: each sequence reads its own K and V<br>
      <span class="muted small">provenance: fixture:lecture_10--intensity · lecture_10.py:L187-L211 (MLP FLOPs and bytes, limit B·T), L226-L248 (attention ST/(S+T), prefill S/2, generation &lt; 1), L250-L253 (no B in attention: own KV cache), L147-L154 (H100 989e12 FLOP/s, 3.35e12 B/s, 295), L317-L327 (D, F). Attention uses D for all heads, as L226 does; the regime tag uses the exact intensity, not the limit.</span>`;
  };
  const ctlT = slider("new tokens per sequence T", 0, 17, s.logT, 1, v => { s.logT = Math.min(v, s.logS); setRange(ctlT, s.logT, fB); draw(); }, fB);
  const ctlS = slider("context S (cached tokens)", 7, 17, s.logS, 1, v => { s.logS = v; if (s.logT > v) { s.logT = v; setRange(ctlT, v, fB); } draw(); }, fB);
  sT = buttons("stage", [["decode", "generation: T = 1"], ["prefill", "prefill: T = S"]], "decode", v => { s.logT = v === "decode" ? 0 : s.logS; setRange(ctlT, s.logT, fB); draw(); });
  root.append(el("div", { class: "widget" }, el("div", { class: "controls" },
    slider("concurrent sequences B", 0, 10, s.logB, 1, v => { s.logB = v; draw(); }, fB),
    ctlS, ctlT, sT.node,
    slider("MLP width F", 1, 4, s.Fx, 1, v => { s.Fx = v; draw(); }, v => `${v} × 13824 = ${v * 13824}`),
    read), pic)); draw();
};

// ============================================================== 2. KV-cache ledger ==
// Per token per layer: 2·K·H·bytes (key and value, L297); MHA K = N, MQA K = 1, GQA in between (L378-L381), so GQA divides
// the cache by N/K (L387). MLA stores one latent of C = 512 plus 64 RoPE dims per layer instead of K and V (L410-L412);
// "N·H = 16384" is DeepSeek-V2's (L411). Sliding window: a local layer holds at most W tokens (L431), global layers hold
// all S (hybrid, L433). Total memory = B · kv_per_seq + parameter_size (L300); fits if ≤ 80e9 (L351).
function kvLedger({ L, N = 40, H = 128, scheme = "gqa", K = N, Cl = 512, R = 64, bytes = 2, S, B = 1, W = Infinity, Lg = L, paramBytes = 2 * LL13_PARAMS, mem = H100.mem }) {
  const Keff = scheme === "mha" ? N : scheme === "mqa" ? 1 : K;
  const perTokLayer = scheme === "mla" ? (Cl + R) * bytes : 2 * Keff * H * bytes;
  const mhaTokLayer = 2 * N * H * bytes;
  const g = Math.min(Lg, L), local = L - g, tokLocal = Math.min(S, W);
  const perSeq = perTokLayer * (g * S + local * tokLocal), allGlobal = perTokLayer * L * S;
  const free = mem - paramBytes;
  return {
    perTokLayer, perTok: perTokLayer * L, ratioVsMHA: mhaTokLayer / perTokLayer, perSeq, perSeqGB: perSeq / 1e9,
    tokLocal, windowRatio: perSeq / allGlobal, total: B * perSeq + paramBytes, kvTotal: B * perSeq,
    fits: B * perSeq + paramBytes <= mem ? 1 : 0, maxB: free > 0 ? Math.floor(free / perSeq) : 0, maxCached: free > 0 ? free / (perTokLayer * L) : 0,
  };
}
MODELS["fixture:lecture_10--kv-ledger"] = {
  fn: kvLedger,
  cases: [
    { args: { L: 40, N: 40, H: 128, scheme: "mha", S: 1024 }, pick: "maxB", expect: 64, from: "lecture_10:kv-cache-size:predict" },
    { args: { L: 32, N: 32, H: 128, scheme: "gqa", K: 8, S: 1 }, pick: "perTok", expect: 131072, tol: 0.02, from: "lecture_10:kv-cache-size:transfer" },
    { args: { L: 40, N: 40, H: 128, scheme: "mha", S: 1 }, pick: "perTok", expect: 819200 },                 // statement: 819,200 B per token
    { args: { L: 40, N: 40, H: 128, scheme: "mha", S: 1024 }, pick: "perSeqGB", expect: 0.8389, tol: 0.001 }, // 0.84 GB per 1024 tokens
    { args: { L: 40, N: 40, H: 128, scheme: "mha", S: 1024, B: 64 }, pick: "total", expect: 79.72e9, tol: 0.001 }, // L345, fits
    { args: { L: 40, N: 40, H: 128, scheme: "mha", S: 1024, B: 256 }, pick: "fits", expect: 0 },             // L349-L352 doesn't fit
    { args: { L: 40, N: 40, H: 128, scheme: "gqa", K: 8, S: 1024, B: 256 }, pick: "fits", expect: 1 },       // L400-L402 still fits
    { args: { L: 40, N: 40, H: 128, scheme: "gqa", K: 8, S: 1 }, pick: "ratioVsMHA", expect: 5, from: "lecture_10:gqa-kv-reduction:predict" },
    { args: { L: 80, N: 64, H: 128, scheme: "gqa", K: 8, S: 1 }, pick: "perTok", expect: 327680, tol: 0.02, from: "lecture_10:gqa-kv-reduction:transfer" },
    { args: { L: 40, N: 40, H: 128, scheme: "mha", S: 1 }, pick: "maxCached", expect: 65888, tol: 0.001 },   // ~66k cached tokens in 80 GB
    { args: { L: 40, N: 40, H: 128, scheme: "gqa", K: 8, S: 1 }, pick: "maxCached", expect: 329438, tol: 0.001 }, // ...K = 8: 5× more
    { args: { L: 60, N: 128, H: 128, scheme: "mla", S: 1 }, pick: "ratioVsMHA", expect: 56.89, tol: 0.01, from: "lecture_10:mla-latent-kv:predict" },
    { args: { L: 60, N: 128, H: 128, scheme: "mla", S: 1 }, pick: "perTok", expect: 69120, tol: 0.02, from: "lecture_10:mla-latent-kv:transfer" },
    { args: { L: 60, N: 128, H: 128, scheme: "mla", S: 131072 }, pick: "perSeqGB", expect: 9.06, tol: 0.002 }, // mla transfer: 128k context
    { args: { L: 32, N: 32, H: 128, scheme: "gqa", K: 8, S: 100000, W: 4096, Lg: 0 }, pick: "tokLocal", expect: 4096, from: "lecture_10:sliding-window-kv:predict" },
    { args: { L: 32, N: 32, H: 128, scheme: "gqa", K: 8, S: 32768, W: 4096, Lg: 8 }, pick: "windowRatio", expect: 0.34375, tol: 0.03, from: "lecture_10:sliding-window-kv:transfer" },
    { args: { L: 32, N: 32, H: 128, scheme: "gqa", K: 8, S: 2048, W: 4096, Lg: 0 }, pick: "windowRatio", expect: 1 },  // edge: S below the window, nothing saved
  ],
};
WIDGETS["fixture:lecture_10--kv-ledger"] = (root) => {
  const s = { scheme: "mha", K: 8, L: 40, N: 40, H: 128, logS: 11, B: 8, logW: 0, Lg: 40, wGB: 2 * LL13_PARAMS / 1e9, mem: 80 };
  const pic = el("div"), read = el("div", { class: "readout" });
  const draw = () => {
    const S = 2 ** s.logS, W = s.logW ? 2 ** s.logW : Infinity, K = Math.min(s.K, s.N);
    const a = { L: s.L, N: s.N, H: s.H, scheme: s.scheme, K, S, B: s.B, W, Lg: s.logW ? s.Lg : s.L, paramBytes: s.wGB * 1e9, mem: s.mem * 1e9 };
    const m = kvLedger(a), mha = kvLedger({ ...a, scheme: "mha" });
    const Wd = 640, x0 = 150, span = 380;
    // (a) bytes per token per layer: MHA reference vs the chosen scheme
    let b = text(10, 16, "KV bytes per token per layer", { fill: C.muted, size: 11 });
    const sc = span / mha.perTokLayer;
    b += text(10, 36, `MHA (K = N = ${s.N})`, { size: 12 }) + rect(x0, 25, mha.perTokLayer * sc, 15, C.rule) + text(x0 + mha.perTokLayer * sc - 4, 37, `${fmt(mha.perTokLayer, 0)} B`, { size: 11, anchor: "end" });
    const nm = { mha: `MHA (K = ${s.N})`, gqa: `GQA (K = ${K})`, mqa: "MQA (K = 1)", mla: "MLA (512 + 64 latent)" }[s.scheme];
    b += text(10, 58, nm, { size: 12, weight: "600" }) + rect(x0, 47, m.perTokLayer * sc, 15, C.a) + text(x0 + m.perTokLayer * sc + 6, 59, `${fmt(m.perTokLayer, 0)} B`, { size: 11 }) + text(Wd - 8, 59, m.ratioVsMHA > 1 ? `${fmt(m.ratioVsMHA, 2)}× smaller` : "", { size: 11, anchor: "end", weight: "600" });
    // (b) layer strip: tokens cached per layer
    const y1 = 84, cw = Math.min(14, span / s.L);
    b += text(10, y1 + 10, `tokens cached per layer`, { fill: C.muted, size: 11 });
    for (let i = 0; i < s.L; i++) {
      const glob = i < a.Lg ? 1 : 0, frac = (glob ? S : m.tokLocal) / S;
      b += rect(x0 + i * cw, y1, cw - (cw > 3 ? 1 : 0), 16, C.rule) + rect(x0 + i * cw, y1 + 16 * (1 - frac), cw - (cw > 3 ? 1 : 0), 16 * frac, glob ? C.a : C.ok);
    }
    b += text(x0, y1 + 30, a.Lg === s.L ? `all ${s.L} layers global: ${S} tokens each (blue)` : `${a.Lg} global × ${S} (blue) + ${s.L - a.Lg} local × ${m.tokLocal} (green, capped by the window)`, { size: 11 });
    // (c) GPU memory: weights + B × KV per sequence
    const y2 = 140, top = Math.max(m.total, a.mem) * 1.04, gs = span / top;
    b += text(10, y2 + 13, `GPU memory, B = ${s.B}`, { size: 12 });
    b += rect(x0, y2, a.paramBytes * gs, 18, C.ink, 'opacity="0.55"') + rect(x0 + a.paramBytes * gs, y2, m.kvTotal * gs, 18, C.b, 'opacity="0.85"');
    b += line(x0 + a.mem * gs, y2 - 6, x0 + a.mem * gs, y2 + 24, C.b, "stroke-dasharray:4 3;stroke-width:1.5");
    b += text(x0, y2 + 34, `weights ${fmt(a.paramBytes / 1e9, 1)} GB (grey) + KV ${fmt(m.kvTotal / 1e9, 1)} GB (orange) = ${fmt(m.total / 1e9, 1)} GB`, { size: 11 });
    b += text(10, y2 + 34, m.fits ? `fits in ${s.mem} GB` : `does not fit in ${s.mem} GB`, { fill: m.fits ? C.ok : C.b, size: 11, weight: "600" });
    pic.innerHTML = svg(Wd, 184, b);
    read.innerHTML = `KV per token (all ${s.L} layers, full context): ${s.scheme === "mla" ? `(512 + 64) × ${s.L} × 2 B` : `${s.scheme === "mha" ? s.N : s.scheme === "mqa" ? 1 : K} × ${s.H} × ${s.L} × 2 × 2 B`} = <span class="big">${fmt(m.perTok, 0)} bytes</span><br>
      per sequence at S = ${S}: ${fmt(m.perSeq / 1e9, 4)} GB${a.Lg < s.L ? ` (${fmt(100 * m.windowRatio, 1)}% of all-global)` : ""} · sequences that fit next to the weights: <b>${fmt(m.maxB, 0)}</b> · cached tokens B × S that fit (all-global): ${fmt(m.maxCached, 0)}<br>
      <span class="muted small">provenance: fixture:lecture_10--kv-ledger · lecture_10.py:L297 (S·K·H·L·2·2), L300 (B·kv + 2·params), L291/L294 and L317-L327 (Llama 2 13B: 13.02e9 params, 26.03 GB), L351 (80e9), L378-L381 and L387 (MHA/MQA/GQA, N/K), L409-L412 (MLA: N·H = 16384 → 512 + 64), L430-L433 (local window, hybrid layers). The weight bar stays fixed when K or the scheme changes, as the gqa predict says to; the lecture's num_params (L291) does drop by 2·D·(N−K)·H·L, about 1.7e9 parameters at K = 8.</span>`;
  };
  const fS = v => `${2 ** v}`, fW = v => v > 8 ? `${2 ** v} tokens` : "off (all global)";
  const ctlLg = slider("global layers (rest local)", 0, 128, s.Lg, 1, v => { s.Lg = Math.min(v, s.L); setRange(ctlLg, s.Lg); draw(); });
  const sch = buttons("attention", [["mha", "MHA"], ["gqa", "GQA"], ["mqa", "MQA"], ["mla", "MLA"]], s.scheme, v => { s.scheme = v; draw(); });
  root.append(el("div", { class: "widget" }, el("div", { class: "controls" },
    sch.node,
    slider("KV heads K (GQA)", 1, 128, s.K, 1, v => { s.K = v; if (s.scheme !== "gqa") { s.scheme = "gqa"; sch.set("gqa"); } draw(); }),
    slider("query heads N", 1, 128, s.N, 1, v => { s.N = v; draw(); }),
    slider("head dim H", 32, 256, s.H, 32, v => { s.H = v; draw(); }),
    slider("layers L", 1, 128, s.L, 1, v => { s.L = v; if (s.Lg > v) { s.Lg = v; setRange(ctlLg, v); } draw(); }),
    slider("context S", 9, 17, s.logS, 1, v => { s.logS = v; draw(); }, fS),
    slider("batch B (sequences)", 1, 512, s.B, 1, v => { s.B = v; draw(); }),
    slider("sliding window W", 8, 15, 8, 1, v => { s.logW = v === 8 ? 0 : v; draw(); }, fW),
    ctlLg,
    slider("weights (GB)", 1, 300, s.wGB, 0.01, v => { s.wGB = v; draw(); }, v => fmt(v, 2)),
    slider("GPU memory (GB)", 16, 192, s.mem, 1, v => { s.mem = v; draw(); }),
    read), pic)); draw();
};

// ============================================================ 3. batch sweep: latency vs throughput ==
// Napkin model (L297-L306, assumption L335): bytes per decode step = parameter_size + B · kv_per_seq; latency = bytes / bandwidth;
// throughput = B / latency. Weights bytes per parameter: bf16 2, fp8/int8 1, int4 0.5 (L463-L466). M GPUs that shard the model
// and cache each read 1/M of the bytes (L364, "harder parallelism"; the even split and perfect overlap are the author's).
// Knee B* = parameter_size / kv_per_seq (where the two reads are equal); ceiling = bandwidth / kv_per_seq (author, from L300-L306).
function batchSweep({ params = LL13_PARAMS, wbytes = 2, kvTok = kvPerToken(LL13), S = LL13.S, B = 1, bw = H100.bw, gpus = 1, mem = H100.mem }) {
  const pBytes = params * wbytes, kvSeq = S * kvTok, bytes = pBytes + B * kvSeq, lat = bytes / gpus / bw;
  return {
    pBytes, kvSeq, bytes, latency: lat, latencyMs: lat * 1e3, throughput: B / lat, perStream: 1 / lat,
    knee: kvSeq > 0 ? pBytes / kvSeq : Infinity, ceiling: kvSeq > 0 ? gpus * bw / kvSeq : Infinity,
    fits: bytes <= gpus * mem ? 1 : 0, maxB: kvSeq > 0 ? Math.floor((gpus * mem - pBytes) / kvSeq) : Infinity, kvShare: B * kvSeq / bytes,
  };
}
MODELS["fixture:lecture_10--batch-sweep"] = {
  fn: batchSweep,
  cases: [
    { args: { B: 256 }, pick: "throughput", expect: 3562, tol: 0.02, from: "lecture_10:batch-latency-throughput-tradeoff:predict" },
    { args: { B: 64 }, pick: "throughput", expect: 2689.5, tol: 0.002 },                       // L345 b64; the predict's "about 2690"
    { args: { S: 4096, B: 1024 }, pick: "ceiling", expect: 998.4, tol: 0.05, from: "lecture_10:batch-latency-throughput-tradeoff:transfer" },
    { args: { B: 1 }, pick: "ceiling", expect: 3993.5, tol: 0.002 },                            // ~3990 tok/s at S = 1024
    { args: { B: 1 }, pick: "knee", expect: 31.03, tol: 0.002 },                               // the bend: B × kv = parameter bytes
    { args: { B: 1 }, pick: "latencyMs", expect: 8.02, tol: 0.002 },                            // L342 b1: 8.0 ms
    { args: { B: 1 }, pick: "throughput", expect: 124.7, tol: 0.002 },                          // ...125 tokens/s
    { args: { B: 64 }, pick: "fits", expect: 1 }, { args: { B: 256 }, pick: "fits", expect: 0 }, // L352
    { args: { B: 1 }, pick: "maxB", expect: 64 },
    { args: { params: 13.0e9, wbytes: 0.5, S: 0, B: 1 }, pick: "latencyMs", expect: 1.94, tol: 0.05, from: "lecture_10:quantization-inference:transfer" },
    { args: { params: 13.0e9, wbytes: 2, S: 0, B: 1 }, pick: "latencyMs", expect: 7.76, tol: 0.002 }, // ...bf16: 4× more
    { args: { wbytes: 0.5, B: 256 }, pick: "throughput", expect: 3876, tol: 0.003 },            // int4 weights at B = 256: KV dominates, +9% only
    { args: { params: 70e9, S: 0, B: 1, gpus: 8 }, pick: "throughput", expect: 191.4, tol: 0.01, from: "lecture_10:decode-bandwidth-bound:transfer" }, // 70B over 8 GPUs (stored 190)
    { args: { params: 8e9, S: 0, B: 1 }, pick: "throughput", expect: 209.4, tol: 0.003 },          // decode-bandwidth-bound predict: 8B bf16, "about 200"
    { args: { params: paramsOf({ ...LL13, K: 8 }), kvTok: kvPerToken({ ...LL13, K: 8 }), B: 256 }, pick: "throughput", expect: 13068, tol: 0.003 }, // L400-L401 k8_b256 (num_params also drops with K, L291)
  ],
};
WIDGETS["fixture:lecture_10--batch-sweep"] = (root) => {
  const s = { logB: 3, logS: 10, K: 40, pB: LL13_PARAMS / 1e9, wbytes: 2, logG: 0 };
  const pic = el("div"), read = el("div", { class: "readout" });
  const draw = () => {
    const base = { params: s.pB * 1e9 - 2 * LL13.D * (LL13.K - s.K) * LL13.H * LL13.L, wbytes: s.wbytes, kvTok: kvPerToken({ ...LL13, K: s.K }), S: 2 ** s.logS, gpus: 2 ** s.logG };
    const B = 2 ** s.logB, m = batchSweep({ ...base, B });
    const Wd = 640, px = 60, pw = 520, X = lb => px + pw * lb / 10;
    const pts = []; for (let lb = 0; lb <= 10.001; lb += 0.125) pts.push([lb, batchSweep({ ...base, B: 2 ** lb })]);
    const tMax = Math.max(...pts.map(p => p[1].throughput)) * 1.08, lMax = Math.max(...pts.map(p => p[1].latencyMs)) * 1.08;
    const panel = (y0, h, key, vmax, col, name, unit) => {
      const Y = v => y0 + h - h * v / vmax;
      let o = rect(px, y0, pw, h, "#fafaf7") + text(px, y0 - 6, name, { fill: col, size: 11, weight: "600" });
      if (m.maxB < 1024 && m.maxB >= 0) { const xm = X(Math.log2(Math.max(m.maxB, 1))); o += rect(xm, y0, px + pw - xm, h, C.b, 'opacity="0.08"'); }
      o += `<path d="${pts.map((p, i) => `${i ? "L" : "M"}${X(p[0]).toFixed(1)},${Y(p[1][key]).toFixed(1)}`).join("")}" style="fill:none;stroke:${col};stroke-width:2"/>`;
      o += `<circle cx="${X(s.logB).toFixed(1)}" cy="${Y(m[key]).toFixed(1)}" r="5" style="fill:${C.hi};stroke:${C.ink}"/>`;
      o += text(px - 6, y0 + 10, fmt(vmax, 0), { fill: C.muted, size: 10, anchor: "end" }) + text(px - 6, y0 + h, "0", { fill: C.muted, size: 10, anchor: "end" }) + text(px - 6, y0 + 24, unit, { fill: C.muted, size: 10, anchor: "end" });
      return [o, Y];
    };
    let [b1, Yt] = panel(22, 110, "throughput", tMax, C.a, "throughput B / latency (all users)", "tok/s");
    if (m.ceiling < tMax) b1 += line(px, Yt(m.ceiling), px + pw, Yt(m.ceiling), C.a, "stroke-dasharray:4 3") + text(px + pw - 4, Yt(m.ceiling) - 4, `ceiling bandwidth / kv_per_seq = ${fmt(m.ceiling, 0)}`, { fill: C.a, size: 10, anchor: "end" });
    const [b2] = panel(158, 80, "latencyMs", lMax, C.b, "latency per token (every user)", "ms");
    let b = b1 + b2;
    if (m.knee >= 1 && m.knee <= 1024) { const xk = X(Math.log2(m.knee)), right = xk < px + pw - 200; b += line(xk, 22, xk, 238, C.ink, "stroke-dasharray:2 3") + text(right ? xk + 4 : xk - 4, 46, `B* = ${fmt(m.knee, 1)}: KV read = weight read`, { size: 10, anchor: right ? "start" : "end" }); }
    if (m.maxB < 1024 && m.maxB >= 1) { const xf = X(Math.log2(m.maxB)), rt = xf > px + pw - 150; b += text(rt ? xf - 4 : xf + 4, 172, `does not fit beyond B = ${m.maxB}`, { fill: C.b, size: 10, anchor: rt ? "end" : "start" }); }
    for (let lb = 0; lb <= 10; lb += 2) b += text(X(lb), 252, `${2 ** lb}`, { fill: C.muted, size: 10, anchor: "middle" });
    b += text(px + pw / 2, 266, "decode batch B (log scale)", { fill: C.muted, size: 10, anchor: "middle" });
    pic.innerHTML = svg(Wd, 272, b);
    const regime = m.kvSeq === 0 ? "KV cache ignored: only the weight read" : B < m.knee ? `weight read dominates (B < B* = ${fmt(m.knee, 1)}): a bigger batch is almost free throughput` : `KV read dominates (B > B* = ${fmt(m.knee, 1)}): throughput is near its ceiling, latency grows linearly in B`;
    read.innerHTML = `bytes per step = ${fmt(m.pBytes / 1e9, 2)} GB weights + ${B} × ${fmt(m.kvSeq / 1e9, 3)} GB KV = ${fmt(m.bytes / 1e9, 2)} GB${base.gpus > 1 ? `, ${fmt(m.bytes / base.gpus / 1e9, 2)} GB per GPU` : ""}<br>
      latency = bytes / 3.35e12 B/s = <span class="big">${fmt(m.latencyMs, 3)} ms</span> per token · throughput = B / latency = <span class="big">${fmt(m.throughput, 0)} tok/s</span><br>
      ${tag(regime, B < m.knee)} · ${tag(m.fits ? `fits in ${base.gpus * 80} GB` : `does not fit in ${base.gpus * 80} GB`, m.fits)}<br>
      <span class="muted small">time-to-first-token is a prefill quantity (L366) and is not on this plot. provenance: fixture:lecture_10--batch-sweep · lecture_10.py:L297-L306 (memory, latency = memory / bandwidth, throughput = B / latency), L335 (perfect overlap, no overhead), L291/L317-L327 (Llama 2 13B, 13.02e9 params), L342-L352 (B = 1, 64, 256; 256 does not fit 80e9), L355-L357 (why), L395-L401 (K = 8), L463-L466 (bytes per number), L364 (shard over GPUs). Compute is ignored, so this is an upper bound (author: B*, the ceiling and the even split over GPUs).</span>`;
  };
  root.append(el("div", { class: "widget" }, el("div", { class: "controls" },
    slider("decode batch B", 0, 10, s.logB, 1, v => { s.logB = v; draw(); }, v => `${2 ** v}`),
    slider("context S", 0, 15, s.logS, 1, v => { s.logS = v; draw(); }, v => `${2 ** v}`),
    slider("KV heads K (Llama 2 13B: 40; fewer K also drops 2·D·(40−K)·H·L params, L291)", 1, 40, s.K, 1, v => { s.K = v; draw(); }),
    slider("parameters (billions)", 1, 200, s.pB, 0.01, v => { s.pB = v; draw(); }, v => fmt(v, 2)),
    buttons("weights", [[2, "bf16 (2 B)"], [1, "fp8 / int8 (1 B)"], [0.5, "int4 (0.5 B)"]], s.wbytes, v => { s.wbytes = v; draw(); }).node,
    slider("GPUs sharing the model and cache", 0, 3, s.logG, 1, v => { s.logG = v; draw(); }, v => `${2 ** v}`),
    read), pic)); draw();
};

// ============================================================ 4. speculative decoding: draft length ==
// Mechanism (L514-L521): the draft p proposes k tokens; the target q scores them in one pass, keeps the prefix up to the first
// rejection and always adds one token of its own, so a pass yields (accepted + 1) tokens.
// Expected tokens per pass with an i.i.d. acceptance rate α: 1 + α + … + α^k = (1 − α^(k+1)) / (1 − α) (Leviathan et al. 2022,
// the paper L513 links; KP spec-draft-length-tradeoff states it). Cost per pass, in plain target decode steps: k draft steps of
// relative cost c plus one verify pass. Memory-bound napkin (L303): a step costs its weight read, so c = draft / target params
// (8B / 70B or 1B / 8B, L538-L539). The verify pass processes B·(k+1) tokens against shared weights; it costs one step while that
// MLP intensity stays below 295 and grows with it beyond (L211, L154) — author's composition, for the far transfer.
// Sweet spot "around three or four" for the paper's setting (video 1:15:45-1:15:58).
function specDecode({ alpha, k, c = 8 / 70, B = 1, accepted, kMax = 16 }) {
  const E = kk => alpha >= 1 ? kk + 1 : (1 - alpha ** (kk + 1)) / (1 - alpha);
  const step = x => Math.max(1, x / ACC);
  const verify = kk => step(B * (kk + 1)) / step(B);
  const speedup = kk => E(kk) / (kk * c + verify(kk));
  let bestK = 1; for (let kk = 2; kk <= kMax; kk++) if (speedup(kk) > speedup(bestK) + 1e-12) bestK = kk;
  return {
    E: E(k), ceiling: alpha < 1 ? 1 / (1 - alpha) : k + 1, verify: verify(k), cost: k * c + verify(k), speedup: speedup(k),
    bestK, bestSpeedup: speedup(bestK), verifyCompute: B * (k + 1) > ACC ? 1 : 0,
    tokensThisPass: accepted === undefined ? NaN : Math.min(accepted, k) + 1, curve: Array.from({ length: kMax }, (_, i) => [i + 1, E(i + 1), speedup(i + 1)]),
  };
}
MODELS["fixture:lecture_10--spec-sweet-spot"] = {
  fn: specDecode,
  cases: [
    { args: { alpha: 0.6, k: 4 }, pick: "E", expect: 2.3056, tol: 0.01, from: "lecture_10:spec-draft-length-tradeoff:predict" },
    { args: { alpha: 0.6, k: 16 }, pick: "E", expect: 2.4999, tol: 0.001 },                  // predict, after: barely rises
    { args: { alpha: 0.6, k: 16 }, pick: "ceiling", expect: 2.5 },                             // 1 / (1 − α)
    { args: { alpha: 0.6, k: 4 }, pick: "bestK", expect: 3 },                                  // check: α 0.6 → best k = 3 ...
    { args: { alpha: 0.8, k: 4 }, pick: "bestK", expect: 5 },                                  // ... α 0.8 → 5 (up)
    { args: { alpha: 0.7, k: 4 }, pick: "bestK", expect: 4 },                                  // the professor's "three or four"
    { args: { alpha: 0.7, k: 4, B: 256 }, pick: "bestK", expect: 1 },                          // transfer: compute-bound verify, best k → 1
    { args: { alpha: 0.7, k: 1, B: 256 }, pick: "bestSpeedup", expect: 0.9196, tol: 0.003 },   // ...and drafting no longer pays (< 1)
    { args: { alpha: 0.7, k: 4, B: 1 }, pick: "verify", expect: 1 },                           // memory-bound verify costs one step
    { args: { alpha: 0.6, k: 4, accepted: 2 }, pick: "tokensThisPass", expect: 3, tol: 0, from: "lecture_10:speculative-sampling-mechanism:predict" },
    { args: { alpha: 0.75, k: 4, accepted: 3 }, pick: "tokensThisPass", expect: 4, tol: 0.05, from: "lecture_10:speculative-sampling-mechanism:transfer" },
    { args: { alpha: 0.6, k: 4, accepted: 4 }, pick: "tokensThisPass", expect: 5 },           // all accepted: k + 1
  ],
};
WIDGETS["fixture:lecture_10--spec-sweet-spot"] = (root) => {
  const s = { alpha: 0.7, k: 2, c: 8 / 70, logB: 0, acc: 1 };
  const pic = el("div"), read = el("div", { class: "readout" });
  let ctlAcc;
  const draw = () => {
    const B = 2 ** s.logB, m = specDecode({ alpha: s.alpha, k: s.k, c: s.c, B, accepted: s.acc });
    const Wd = 640, px = 50, pw = 560, bw = pw / 16;
    const eMax = Math.max(m.ceiling, 2) * 1.1, sMax = Math.max(...m.curve.map(r => r[2]), 1.2) * 1.1;
    // (a) one pass: k draft boxes, first rejection at position acc + 1
    let b = text(10, 16, `one target pass (k = ${s.k}): green accepted, orange rejected and resampled, grey discarded, gold = the target's own token`, { fill: C.muted, size: 10 });
    const cell = Math.min(30, (pw - 270) / (s.k + 1) - 3), a = Math.min(s.acc, s.k), cf = cell < 22 ? 9 : 11;
    for (let j = 0; j < s.k; j++) b += rect(px + j * (cell + 3), 26, cell, 22, j < a ? C.ok : j === a ? C.b : C.rule) + text(px + j * (cell + 3) + cell / 2, 41, `d${j + 1}`, { fill: j <= a ? "#fff" : C.muted, size: cf, anchor: "middle" });
    b += rect(px + s.k * (cell + 3) + 8, 26, cell, 22, C.hi) + text(px + s.k * (cell + 3) + 8 + cell / 2, 41, a < s.k ? "new" : "+1", { size: cf, anchor: "middle" });
    b += text(px + (s.k + 1) * (cell + 3) + 16, 41, `${a} accepted + 1 = ${m.tokensThisPass} tokens from this pass`, { size: 12, weight: "600" });
    // (b) sweep over k: expected tokens per pass (bars); (c) speed-up over plain decoding (dots)
    b += text(10, 76, "expected tokens per target pass, k = 1 … 16 (blue bar = your k)", { fill: C.muted, size: 11 });
    const yB = 84, hB = 80, yS = 210, hS = 70;
    const Ye = v => yB + hB - hB * v / eMax, Ys = v => yS + hS - hS * v / sMax;
    b += line(px, Ye(m.ceiling), px + pw, Ye(m.ceiling), C.a, "stroke-dasharray:4 3") + text(px - 4, Ye(m.ceiling) + 4, fmt(m.ceiling, 2), { fill: C.a, size: 10, anchor: "end" });
    b += text(px + pw, yB - 4, `dashed: ceiling 1/(1−α) = ${fmt(m.ceiling, 3)}`, { fill: C.a, size: 10, anchor: "end" });
    b += text(10, yS - 10, "speed-up over plain decoding = tokens per pass / cost per pass", { fill: C.muted, size: 11 });
    b += rect(px, yS, pw, hS, "#fafaf7") + line(px, Ys(1), px + pw, Ys(1), C.b, "stroke-dasharray:3 3") + text(px - 4, Ys(1) + 4, "1×", { fill: C.b, size: 10, anchor: "end" });
    b += `<path d="${m.curve.map(([kk, , sp], i) => `${i ? "L" : "M"}${(px + (kk - 0.5) * bw).toFixed(1)},${Ys(sp).toFixed(1)}`).join("")}" style="fill:none;stroke:${C.b};stroke-width:1"/>`;
    m.curve.forEach(([kk, e, sp]) => {
      const x = px + (kk - 1) * bw;
      b += rect(x + 4, Ye(e), bw - 8, yB + hB - Ye(e), kk === s.k ? C.a : C.rule) + text(x + bw / 2, yB + hB + 13, kk, { fill: kk === s.k ? C.ink : C.muted, size: 10, anchor: "middle", weight: kk === s.k ? "700" : "" });
      b += `<circle cx="${(x + bw / 2).toFixed(1)}" cy="${Ys(sp).toFixed(1)}" r="${kk === m.bestK ? 6 : 3.5}" style="fill:${kk === m.bestK ? C.hi : C.b};stroke:${C.ink};stroke-width:${kk === m.bestK ? 1.5 : 0.5}"/>`;
    });
    const xb = px + (m.bestK - 0.5) * bw;
    b += line(xb, yS, xb, yS + hS, C.hi, "stroke-width:1.5") + text(px + pw, yS - 10, `best k = ${m.bestK}: ${fmt(m.bestSpeedup, 2)}×${m.bestSpeedup < 1 ? " (drafting does not pay)" : ""}`, { size: 11, weight: "600", anchor: "end" });
    pic.innerHTML = svg(Wd, yS + hS + 8, b);
    const vReg = m.verifyCompute ? `compute-bound: B·(k+1) = ${B * (s.k + 1)} > 295 tokens, so verifying costs ${fmt(m.verify, 2)} decode steps` : `memory-bound: B·(k+1) = ${B * (s.k + 1)} ≤ 295 tokens, so verifying k + 1 tokens costs one decode step`;
    read.innerHTML = `α = ${s.alpha.toFixed(2)}, k = ${s.k}: expected tokens per pass (1 − α^(k+1)) / (1 − α) = <span class="big">${fmt(m.E, 4)}</span> (ceiling ${fmt(m.ceiling, 3)})<br>
      cost per pass = k·c + verify = ${s.k} × ${fmt(s.c, 3)} + ${fmt(m.verify, 2)} = ${fmt(m.cost, 3)} plain steps → speed-up <span class="big">${fmt(m.speedup, 3)}×</span> · best k = <b>${m.bestK}</b> (${fmt(m.bestSpeedup, 3)}×)<br>
      target verify pass at B = ${B}: ${tag(vReg, !m.verifyCompute)}<br>
      <span class="muted small">provenance: fixture:lecture_10--spec-sweet-spot · lecture_10.py:L509-L511 (checking is parallel, generation sequential), L514-L521 (draft k tokens, verify in one pass, always at least one token), L538-L539 (70B/8B and 8B/1B pairs; c = their parameter ratio under L303's bytes/bandwidth), L154 and L211 (295, MLP intensity B·T); video 1:15:45-1:16:08 (too few / too many, sweet spot three or four, draft close to target). The geometric closed form assumes every draft token is accepted independently with the same α (Leviathan et al. 2022); real acceptance varies by position. The compute-bound verify cost is the author's composition.</span>`;
  };
  ctlAcc = slider("accepted before the first rejection (this pass)", 0, 16, s.acc, 1, v => { s.acc = Math.min(v, s.k); setRange(ctlAcc, s.acc, fAcc); draw(); }, v => fAcc(v));
  function fAcc(v) { return v >= s.k ? `${v} (all k accepted)` : `${v}`; }
  root.append(el("div", { class: "widget" }, el("div", { class: "controls" },
    slider("acceptance rate α", 0.05, 0.95, s.alpha, 0.05, v => { s.alpha = v; draw(); }, v => v.toFixed(2)),
    slider("draft length k", 1, 16, s.k, 1, v => { s.k = v; if (s.acc > v) { s.acc = v; setRange(ctlAcc, v, fAcc); } draw(); }),
    buttons("draft / target", [[8 / 70, "8B / 70B"], [1 / 8, "1B / 8B"]], s.c, v => { s.c = v; draw(); }).node,
    slider("concurrent sequences B at the target", 0, 10, s.logB, 1, v => { s.logB = v; draw(); }, v => `${2 ** v}`),
    ctlAcc,
    read), pic)); draw();
};

// ============================================================ 5. what a KV cache can reuse ==
// Spoken (video 22:24-22:41): the cached tokens "shouldn't change ... because it's a causal transformer. If it was bidirectional,
// then if you attach a token, then everything changes." Layer 1's K and V are projections of each token's own embedding, so they
// never change; from layer 2 up a token's K and V change whenever anything it attends to changed (KP kv-cache-needs-causality).
// Prefix-LM (bidirectional prompt block, causal outside it) is the KP transfer's case; position shifts from inserting a token are ignored.
function kvRecompute({ L, n, mask = "causal", where = "end", P = 0 }) {
  const all = mask === "bidir" || (mask === "prefix" && where === "prompt");
  const old = all ? n * (L - 1) : 0;
  return { pairs: L + old, newPairs: L, oldRecomputed: old, reused: n * L - old, total: (n + 1) * L, reusedShare: n ? (n * L - old) / (n * L) : 1 };
}
MODELS["fixture:lecture_10--kv-recompute"] = {
  fn: kvRecompute,
  cases: [
    { args: { L: 4, n: 100, mask: "bidir" }, pick: "pairs", expect: 304, from: "lecture_10:kv-cache-needs-causality:check" },
    { args: { L: 4, n: 100, mask: "causal" }, pick: "pairs", expect: 4 },                     // the check's why: a causal cache computes only the new 4
    { args: { L: 1, n: 100, mask: "bidir" }, pick: "pairs", expect: 1 },                      // edge: one layer, nothing above layer 1 to change
    { args: { L: 4, n: 100, mask: "bidir" }, pick: "reusedShare", expect: 0.25 },             // only layer 1 survives
    { args: { L: 4, n: 100, mask: "prefix", where: "end", P: 60 }, pick: "pairs", expect: 4 },  // transfer: generated tokens never touch the prompt
    { args: { L: 4, n: 100, mask: "prefix", where: "prompt", P: 60 }, pick: "pairs", expect: 304 }, // ...text added inside the prompt block
  ],
};
WIDGETS["fixture:lecture_10--kv-recompute"] = (root) => {
  const s = { L: 3, n: 7, mask: "causal", where: "end", P: 4 };
  const pic = el("div"), read = el("div", { class: "readout" });
  const draw = () => {
    const P = Math.min(s.P, s.n), m = kvRecompute({ ...s, P });
    const all = s.mask === "bidir" || (s.mask === "prefix" && s.where === "prompt");
    // columns: old tokens 0..n-1 plus the new token (inserted at the end of the prompt block for "prompt")
    const cols = []; for (let i = 0; i < s.n; i++) cols.push({ old: true, idx: i, prompt: s.mask === "prefix" && i < P });
    const at = s.mask === "prefix" && s.where === "prompt" ? P : s.n; cols.splice(at, 0, { old: false, prompt: s.mask === "prefix" && s.where === "prompt" });
    const maxCols = 33, show = cols.length <= maxCols ? cols.map((c, i) => [c, i]) : [...cols.slice(0, 16).map((c, i) => [c, i]), null, ...cols.slice(-16).map((c, i) => [c, cols.length - 16 + i])];
    const Wd = 640, x0 = 70, cw = Math.min(30, 550 / show.length), ch = Math.min(26, 150 / s.L), y0 = 30;
    let b = text(10, 16, "one cell = one (token, layer) K, V · blue: new token · orange: recomputed · grey: reused from the cache", { fill: C.muted, size: 11 });
    show.forEach((e, j) => {
      const x = x0 + j * cw;
      if (!e) { b += text(x + cw / 2, y0 + s.L * ch / 2, "…", { fill: C.muted, size: 14, anchor: "middle" }); return; }
      const [c] = e;
      for (let l = 0; l < s.L; l++) {
        const y = y0 + (s.L - 1 - l) * ch, fill = !c.old ? C.a : all && l >= 1 ? C.b : C.rule;
        b += rect(x + 1, y + 1, cw - 2, ch - 2, fill, c.prompt ? `stroke="${C.ink}" stroke-width="0.6"` : "");
      }
    });
    for (let l = 0; l < s.L; l++) b += text(x0 - 6, y0 + (s.L - 1 - l) * ch + ch / 2 + 4, `layer ${l + 1}`, { fill: C.muted, size: 10, anchor: "end" });
    const yb = y0 + s.L * ch + 16;
    b += text(x0, yb, s.mask === "causal" ? "causal: no earlier token attends to the new one, so nothing cached can change" :
      s.mask === "bidir" ? "bidirectional: every earlier token attends to the new one; from layer 2 up its K, V change" :
      s.where === "end" ? `prefix-LM: the ${P} outlined prompt tokens never see generated ones; nothing cached changes` :
      `prefix-LM: text added inside the outlined prompt block changes every K, V above layer 1`, { size: 11 });
    pic.innerHTML = svg(Wd, yb + 10, b);
    read.innerHTML = `(token, layer) pairs computed this step: new token ${m.newPairs} + earlier tokens ${m.oldRecomputed} = <span class="big">${m.pairs}</span>
      · reused from the cache: ${m.reused} of ${s.n * s.L} (${fmt(100 * m.reusedShare, 1)}%)<br>
      ${tag(all ? "cache invalid above layer 1: this is a recomputation, not a reuse" : "cache exact: every stored K and V is still correct", !all)}<br>
      <span class="muted small">provenance: fixture:lecture_10--kv-recompute · video 22:24-22:41 ("because it's a causal transformer. If it was bidirectional ... everything changes"); lecture_10.py:L169-L172 (work shared across prefixes, store K and V per token and layer). Layer 1's K, V depend on the token's own embedding only (KP kv-cache-needs-causality). Positions are ignored: inserting a token inside the prompt would also shift later positions.</span>`;
  };
  const maskB = buttons("attention mask", [["causal", "causal (decoder)"], ["bidir", "bidirectional (encoder)"], ["prefix", "prefix-LM"]], s.mask, v => { s.mask = v; draw(); });
  const whereB = buttons("append the new token", [["end", "after the last token"], ["prompt", "inside the prompt block (prefix-LM)"]], s.where, v => { s.where = v; if (v === "prompt" && s.mask !== "prefix") { s.mask = "prefix"; maskB.set("prefix"); } draw(); });
  root.append(el("div", { class: "widget" }, el("div", { class: "controls" },
    maskB.node, whereB.node,
    slider("layers L", 1, 8, s.L, 1, v => { s.L = v; draw(); }),
    slider("tokens already processed n", 1, 200, s.n, 1, v => { s.n = v; draw(); }),
    slider("prompt block length (prefix-LM)", 0, 200, s.P, 1, v => { s.P = v; draw(); }),
    read), pic)); draw();
};

// ============================================================ 6. PagedAttention: slab vs blocks ==
// Status quo (L582-L587): reserve a KV section for prompt + response up to a max length; the unused tail is internal
// fragmentation. PagedAttention (L589-L590): a sequence's cache lives in fixed-size non-contiguous blocks, allocated as tokens
// arrive, so at most one partly filled block per sequence. Sharing (L595-L599): samples of one prompt map to the same physical
// prompt blocks; copy-on-write at the block level. 16-token blocks are the KP prompts' (vLLM's default); 819,200 B per token is
// Llama 2 13B's (L297, KP kv-cache-size). Copy-on-write detail (author, vLLM paper §4.4): the partly filled last prompt block is
// copied for every writer but the last, so n samples hold n versions of it once they generate.
function pagedBlocks({ P, G, b = 16, M = 2048, n = 1, share = 1, bytesTok = 819200 }) {
  const used = P + G, per = Math.ceil(used / b), pBlocks = Math.ceil(P / b);
  const sh = share && n > 1;
  const fp = sh ? (G > 0 ? Math.floor(P / b) : pBlocks) : 0, r = P - fp * b;
  const ownPer = sh ? (G > 0 ? Math.ceil((r + G) / b) : 0) : per;
  const phys = fp + n * ownPer, stored = sh ? (G > 0 ? fp * b + n * (r + G) : P) : n * used;
  const promptBlocks = sh ? pBlocks : n * pBlocks;
  return {
    used, perSeqBlocks: per, slabFits: used <= M ? 1 : 0, slabWaste: Math.max(0, M - used), slabWasteTotal: n * Math.max(0, M - used),
    pagedWaste: per * b - used, phys, sharedBlocks: fp, ownPer, wasteTotal: phys * b - stored, promptBlocks,
    promptBytes: promptBlocks * b * bytesTok, promptMB: promptBlocks * b * bytesTok / 1e6, cow: sh && G > 0 && r > 0 ? n - 1 : 0,
  };
}
MODELS["fixture:lecture_10--paged-blocks"] = {
  fn: pagedBlocks,
  cases: [
    { args: { P: 37, G: 60 }, pick: "pagedWaste", expect: 15, tol: 0, from: "lecture_10:paged-attention:predict" },
    { args: { P: 37, G: 60 }, pick: "slabWaste", expect: 1951 },                                   // the predict's 2048 − 97
    { args: { P: 300, G: 0, n: 4 }, pick: "promptBlocks", expect: 19, tol: 0, from: "lecture_10:paged-attention:check" },
    { args: { P: 300, G: 0, n: 4, share: 0 }, pick: "promptBlocks", expect: 76 },                  // the check's why: 76 without sharing
    { args: { P: 500, G: 0, n: 8 }, pick: "promptMB", expect: 419.43, tol: 0.002, from: "lecture_10:paged-attention:transfer" },
    { args: { P: 500, G: 0, n: 8, share: 0 }, pick: "promptMB", expect: 3355.4, tol: 0.002 },      // ...3.36 GB without sharing
    { args: { P: 300, G: 1, n: 4 }, pick: "cow", expect: 3 },                                      // first token: 3 copies of block 19
    { args: { P: 300, G: 1, n: 4 }, pick: "phys", expect: 22 },                                    // 18 shared + 4 private
    { args: { P: 32, G: 0 }, pick: "pagedWaste", expect: 0 },                                      // edge: exact multiple of the block
    { args: { P: 37, G: 60, b: 1 }, pick: "pagedWaste", expect: 0 },                               // edge: 1-token blocks, no waste
    { args: { P: 37, G: 60, n: 3 }, pick: "wasteTotal", expect: 45 },                              // one partial block per sample
  ],
};
WIDGETS["fixture:lecture_10--paged-blocks"] = (root) => {
  const s = { P: 20, G: 30, b: 16, M: 2048, n: 2, share: 1 };
  const pic = el("div"), read = el("div", { class: "readout" });
  const draw = () => {
    const m = pagedBlocks(s), Wd = 640, x0 = 120, span = 500;
    // (a) the slab: one reserved section of M slots per request
    let b = text(10, 16, `pre-allocated slab: ${s.M} slots reserved per request (L586)`, { fill: C.muted, size: 11 });
    const u = Math.min(m.used, s.M) / s.M;
    b += text(10, 36, "slab", { size: 12, weight: "600" }) + rect(x0, 24, span, 16, C.rule) + rect(x0, 24, span * u, 16, C.a);
    b += text(x0 + span, 54, m.slabFits ? `${m.used} used (blue) · ${m.slabWaste} reserved but unused (grey)` : `${m.used} tokens do not fit the ${s.M}-slot slab`, { size: 11, anchor: "end", fill: m.slabFits ? C.ink : C.b });
    // (b) block tables: one row per sample, one cell per logical block, the number is the physical block
    const show = Math.min(s.n, 6), nb = m.perSeqBlocks, cw = Math.max(3, Math.min(22, span / Math.max(nb, 1))), y0 = 78, rh = 24;
    b += text(10, y0 - 6, `PagedAttention, ${s.b}-token blocks: one row per block table (blue = shared physical block, green = own)`, { fill: C.muted, size: 11 });
    let nextOwn = m.sharedBlocks;
    for (let i = 0; i < show; i++) {
      const y = y0 + i * rh;
      b += text(10, y + 14, s.n > 1 ? `sample ${i + 1}` : "request", { size: 11 });
      for (let j = 0; j < Math.min(nb, Math.floor(span / cw)); j++) {
        const shared = j < m.sharedBlocks, fill = Math.min(s.b, m.used - j * s.b) / s.b, id = shared ? j : nextOwn++;
        const x = x0 + j * cw;
        b += rect(x, y, cw - (cw > 4 ? 2 : 0), 18, C.rule) + rect(x, y + 18 * (1 - fill), cw - (cw > 4 ? 2 : 0), 18 * fill, shared ? C.a : C.ok);
        if (cw >= 18) b += text(x + (cw - 2) / 2, y + 13, id, { size: 9, anchor: "middle", fill: "#fff" });
      }
      if (nb * cw > span) b += text(x0 + span + 4, y + 13, "…", { fill: C.muted, size: 12 });
    }
    if (s.n > show) b += text(x0, y0 + show * rh + 12, `+ ${s.n - show} more samples with the same layout`, { fill: C.muted, size: 11 });
    const yEnd = y0 + show * rh + (s.n > show ? 20 : 6);
    pic.innerHTML = svg(Wd, yEnd + 4, b);
    const shareTxt = s.n > 1 ? (s.share ? `shared: ${m.promptBlocks} prompt blocks serve all ${s.n} samples` : `not shared: ${s.n} × ${Math.ceil(s.P / s.b)} = ${m.promptBlocks} prompt blocks`) : "one request: nothing to share";
    read.innerHTML = `${s.P} prompt + ${s.G} generated = ${m.used} tokens per sample · slab waste ${m.slabFits ? `${s.M} − ${m.used} = <b>${m.slabWaste}</b> slots` : "(does not fit)"} per request (internal fragmentation)<br>
      paged: ⌈${m.used} / ${s.b}⌉ = ${m.perSeqBlocks} blocks = ${m.perSeqBlocks * s.b} slots → <span class="big">${m.pagedWaste}</span> slots wasted per sample (always &lt; ${s.b})<br>
      prompt KV: ${shareTxt} = ${fmt(m.promptMB, 1)} MB at 819,200 B per token · physical blocks in use: <b>${m.phys}</b>${m.cow ? ` · copy-on-write: once the samples write into the partly filled block ${Math.ceil(s.P / s.b) - 1}, ${m.cow} copies are made` : ""}<br>
      ${tag(s.share && s.n > 1 ? "prefix shared: one physical copy of the prompt" : "no sharing: every sample stores its own prompt", s.share && s.n > 1)}<br>
      <span class="muted small">provenance: fixture:lecture_10--paged-blocks · lecture_10.py:L582-L587 (reserve up to a max length; internal and external fragmentation), L589-L590 (non-contiguous blocks), L592-L599 (shared system prompt, several samples per prompt, copy-on-write at the block level), L297 with llama2_13b_config (819,200 B per token). Block ids are numbered in allocation order here; vLLM's allocator picks any free block. External fragmentation (gaps between slabs) is not drawn.</span>`;
  };
  const shB = buttons("prefix sharing", [[1, "on"], [0, "off"]], s.share, v => { s.share = v; draw(); });
  root.append(el("div", { class: "widget" }, el("div", { class: "controls" },
    slider("prompt tokens", 0, 1024, s.P, 1, v => { s.P = v; draw(); }),
    slider("generated tokens", 0, 2048, s.G, 1, v => { s.G = v; draw(); }),
    buttons("block size", [[1, "1"], [8, "8"], [16, "16"], [32, "32"]], s.b, v => { s.b = v; draw(); }).node,
    slider("slab length (max tokens reserved)", 256, 4096, s.M, 256, v => { s.M = v; draw(); }),
    slider("samples per prompt", 1, 16, s.n, 1, v => { s.n = v; draw(); }),
    shB.node, read), pic)); draw();
};

// ============================================================ 7. continuous batching: schedule + selective batching ==
// Ragged array (L557-L559): requests finish at different times. Static batching holds a batch until its longest member ends;
// iteration-level scheduling (L561-L563) decodes step by step and admits a waiting request into a freed slot at the next step.
// Selective batching (L565-L573): non-attention ops run on all sequences concatenated, [3 + 9 + 5, H]; attention runs per sequence.
// Author's simplifications: every request is already waiting at step 0 (in list order), one token per request per step, prefill
// not drawn. The default queue reuses the lecture's lengths 3, 9, 5 (L572) as output lengths.
const parseLens = (txt, dflt) => { const v = String(txt).split(/[^0-9]+/).map(Number).filter(x => Number.isFinite(x) && x > 0).slice(0, 26); return v.length ? v : dflt; };
function batchSchedule({ lens, slots, policy = "static", toks = [] }) {
  const runs = [];                                   // { req, slot, start, end }
  if (policy === "static") {
    let t = 0;
    for (let i = 0; i < lens.length; i += slots) {
      const grp = lens.slice(i, i + slots);
      grp.forEach((l, j) => runs.push({ req: i + j, slot: j, start: t, end: t + l }));
      t += Math.max(...grp);
    }
  } else {
    const free = Array(slots).fill(0);
    lens.forEach((l, i) => { const j = free.indexOf(Math.min(...free)); runs.push({ req: i, slot: j, start: free[j], end: free[j] + l }); free[j] += l; });
  }
  const makespan = Math.max(0, ...runs.map(r => r.end)), busy = lens.reduce((a, x) => a + x, 0);
  const tok = toks.length ? toks : [0];
  return {
    runs, makespan, busy, occupancy: makespan ? busy / (slots * makespan) : 0, meanFinish: runs.reduce((a, r) => a + r.end, 0) / Math.max(runs.length, 1),
    mlpRows: toks.reduce((a, x) => a + x, 0), attnCalls: toks.length, paddedRows: toks.length * Math.max(...tok), padding: toks.length * Math.max(...tok) - toks.reduce((a, x) => a + x, 0),
  };
}
const inFlight = (m, t) => m.runs.filter(r => r.start <= t && t < r.end);
MODELS["fixture:lecture_10--batch-schedule"] = {
  fn: batchSchedule,
  cases: [
    { args: { lens: [50, 100, 150, 400], slots: 4 }, pick: "occupancy", expect: 0.4375, tol: 0.001, from: "lecture_10:continuous-batching:check" },
    { args: { lens: [50, 100, 150, 400], slots: 4, policy: "iteration" }, pick: "occupancy", expect: 0.4375, tol: 0.001 }, // edge: nobody waiting, nothing to refill
    { args: { lens: [], slots: 3, toks: [4, 7, 12] }, pick: "mlpRows", expect: 23, tol: 0, from: "lecture_10:continuous-batching:predict" },
    { args: { lens: [], slots: 3, toks: [4, 7, 12] }, pick: "attnCalls", expect: 3 },                    // ...three attention computations
    { args: { lens: [], slots: 3, toks: [3, 9, 5] }, pick: "mlpRows", expect: 17 },                      // L573: [3 + 9 + 5, H]
    { args: { lens: [], slots: 3, toks: [3, 9, 5] }, pick: "padding", expect: 10 },                      // padded to [3, 9, H]: 10 rows of padding
    { args: { lens: [9, 3, 5, 9, 3, 5], slots: 3 }, pick: "makespan", expect: 18 },                      // default queue, static
    { args: { lens: [9, 3, 5, 9, 3, 5], slots: 3, policy: "iteration" }, pick: "makespan", expect: 13 }, // ...iteration-level
    { args: { lens: [9, 3, 5, 9, 3, 5], slots: 3 }, pick: "occupancy", expect: 34 / 54, tol: 0.001 },
    { args: { lens: [9, 3, 5, 9, 3, 5], slots: 3, policy: "iteration" }, pick: "occupancy", expect: 34 / 39, tol: 0.001 },
    { args: { lens: [20, 400, 20, 20, 20, 20, 20, 20], slots: 8 }, pick: "occupancy", expect: 540 / 3200, tol: 0.001 }, // transfer shape: one long request holds a batch of 8
  ],
};
const REQ_COL = ["#24668d", "#b8582a", "#3c8d5a", "#8a5a9e", "#c08a1e", "#4f7f86", "#9e4f5a", "#6b7a2e"];
WIDGETS["fixture:lecture_10--batch-schedule"] = (root) => {
  const dfl = [9, 3, 5, 9, 3, 5];
  const s = { lens: dfl, slots: 3, policy: "static", t: 4, toks: [3, 9, 5] };
  const pic = el("div"), read = el("div", { class: "readout" }), pic2 = el("div"), read2 = el("div", { class: "readout" });
  let ctlT;
  const draw = () => {
    const m = batchSchedule(s), alt = batchSchedule({ ...s, policy: s.policy === "static" ? "iteration" : "static" });
    const horizon = Math.max(m.makespan, alt.makespan, 1), t = Math.min(s.t, horizon - 1);
    if (ctlT) { const inp = ctlT.querySelector("input"); if (inp) inp.max = horizon; }
    const Wd = 640, x0 = 70, span = 550, X = v => x0 + span * v / horizon, rh = Math.min(26, 170 / s.slots), y0 = 24;
    let b = text(10, 14, `${s.policy === "static" ? "static batching" : "iteration-level scheduling"}: rows = batch slots, columns = decode steps, grey = idle`, { fill: C.muted, size: 11 });
    for (let j = 0; j < s.slots; j++) { const y = y0 + j * rh; b += text(10, y + rh / 2 + 4, `slot ${j + 1}`, { fill: C.muted, size: 10 }) + rect(x0, y + 1, span, rh - 3, "#ece8de"); }
    m.runs.forEach(r => {
      const y = y0 + r.slot * rh, w = X(r.end) - X(r.start), nm = String.fromCharCode(65 + r.req);
      b += rect(X(r.start) + 0.5, y + 1, w - 1, rh - 3, REQ_COL[r.req % REQ_COL.length], 'opacity="0.9"');
      if (w > 12) b += text(X(r.start) + w / 2, y + rh / 2 + 4, w > 30 ? `${nm} (${r.end - r.start})` : nm, { fill: "#fff", size: 10, anchor: "middle" });
    });
    const yb = y0 + s.slots * rh;
    b += line(X(t + 0.5), y0 - 4, X(t + 0.5), yb + 2, C.ink, "stroke-width:1.5") + text(X(t + 0.5), yb + 14, `step ${t + 1}`, { size: 10, anchor: "middle" });
    b += line(X(m.makespan), y0 - 4, X(m.makespan), yb + 2, C.b, "stroke-dasharray:4 3") + text(X(m.makespan) > 420 ? X(m.makespan) - 4 : X(m.makespan) + 4, yb + 28, `all done: ${m.makespan} steps`, { fill: C.b, size: 10, anchor: X(m.makespan) > 420 ? "end" : "start" });
    pic.innerHTML = svg(Wd, yb + 36, b);
    const fl = inFlight(m, t), waiting = m.runs.filter(r => r.start > t).map(r => String.fromCharCode(65 + r.req));
    read.innerHTML = `busy slot-steps ${m.busy} of ${s.slots} × ${m.makespan} = ${s.slots * m.makespan} → occupancy <span class="big">${fmt(m.occupancy, 4)}</span> · all requests done after <b>${m.makespan}</b> steps (the other policy: ${alt.makespan} steps, occupancy ${fmt(alt.occupancy, 4)}) · mean finish step ${fmt(m.meanFinish, 2)}<br>
      at step ${t + 1}: ${fl.length} of ${s.slots} slots decoding (${fl.map(r => String.fromCharCode(65 + r.req)).join(", ") || "none"}) · waiting: ${waiting.join(", ") || "none"} · this step's MLP input is [${fl.length}, H], attention runs ${fl.length} times, each over its own cache<br>
      ${tag(s.policy === "static" ? "static: a freed slot stays idle until the batch's longest request ends" : "iteration-level: a freed slot takes the next waiting request at the next step", s.policy !== "static")}<br>
      <span class="muted small">provenance: fixture:lecture_10--batch-schedule · lecture_10.py:L557-L559 (ragged array), L561-L563 (iteration-level scheduling: decode step by step, add new requests as they arrive), L565-L573 (selective batching); Orca (Yu et al., OSDI 2022, L554). Simplified: all requests wait at step 0 in list order, one token per request per step, prefill not shown. Default lengths 3, 9, 5 are the lecture's L572 example reused as output lengths.</span>`;
    // selective batching panel
    const tk = s.toks, mx = Math.max(...tk), tot = tk.reduce((a, x) => a + x, 0), sc = Math.min(14, 260 / Math.max(tot, mx * tk.length));
    let c = text(10, 14, "selective batching: tokens of each sequence this iteration", { fill: C.muted, size: 11 });
    c += text(10, 38, "concatenated", { size: 11 }) + text(10, 70, `padded to ${mx}`, { size: 11 });
    let xx = 110; tk.forEach((n, i) => { c += rect(xx, 26, n * sc - 1, 16, REQ_COL[i % REQ_COL.length]); xx += n * sc; });
    c += text(xx + 6, 39, `[${tk.join(" + ")}, H] = [${tot}, H] → MLP, norms, projections`, { size: 11 });
    xx = 110; tk.forEach((n, i) => { c += rect(xx, 58, n * sc - 1, 16, REQ_COL[i % REQ_COL.length]) + rect(xx + n * sc, 58, (mx - n) * sc - 1, 16, C.rule); xx += mx * sc; });
    c += text(xx + 6, 71, `[${tk.length} × ${mx}, H] = ${tk.length * mx} rows (${tk.length * mx - tot} padding)`, { size: 11, fill: C.muted });
    c += text(10, 98, `attention: ${tk.length} separate computations, one per sequence (each needs its own length and KV cache)`, { size: 11 });
    pic2.innerHTML = svg(Wd, 108, c);
    read2.innerHTML = `non-attention input <span class="big">[${tot}, H]</span> · attention computations: <b>${tk.length}</b> · padding avoided: ${tk.length * mx - tot} rows <span class="muted small">(L570-L573)</span>`;
  };
  ctlT = slider("look at decode step", 1, 18, s.t + 1, 1, v => { s.t = Math.round(v) - 1; draw(); });
  const pol = buttons("scheduler", [["static", "static batching"], ["iteration", "iteration-level (Orca)"]], s.policy, v => { s.policy = v; draw(); });
  const lensIn = el("input", { type: "text", value: dfl.join(", "), size: 34, oninput: () => { s.lens = parseLens(lensIn.value, dfl); draw(); } });
  const toksIn = el("input", { type: "text", value: "3, 9, 5", size: 20, oninput: () => { s.toks = parseLens(toksIn.value, [3, 9, 5]); draw(); } });
  root.append(el("div", { class: "widget" }, el("div", { class: "controls" },
    pol.node, el("label", {}, "output lengths of the waiting requests, in order ", lensIn),
    slider("batch slots", 1, 8, s.slots, 1, v => { s.slots = Math.max(1, Math.round(v)); draw(); }), ctlT, read), pic),
    el("div", { class: "widget" }, el("div", { class: "controls" }, el("label", {}, "tokens per sequence ", toksIn), read2), pic2)); draw();
};
