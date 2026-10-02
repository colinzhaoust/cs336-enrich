// Widgets for thread lecture_04 (CS336 L4, attention alternatives and mixtures of experts). Register as "fixture:<id>" -> (root, notice) => void.
// Each widget has a pure model in MODELS (no DOM) that tools/check_widgets.mjs tests against the KPs' stored answers.
// Sources: lectures/lecture_04/lecture_04.txt (slide text layer, cited as pdf page + txt line), lectures/lecture_04/transcript.json
// (cited as video M:SS), official/lectures/lecture_10.py (KV-cache formula), and numbers that the KPs' own prompts already state
// (cited as "prompt <kp>:<name>"; those are the KP author's, e.g. DeepSeek-V2's MLA widths, and are not on the slides).
import { el, fmt, slider } from "../widgets_lib.js";

export const MODELS = {};
export const WIDGETS = {};

const C = { ink: "var(--ink)", muted: "var(--muted)", rule: "var(--rule)", a: "var(--accent)", b: "var(--accent2)", ok: "var(--ok)", hi: "var(--hilite)" };
const rect = (x, y, w, h, fill, extra = "") => `<rect x="${x.toFixed(1)}" y="${y.toFixed(1)}" width="${Math.max(0, w).toFixed(1)}" height="${Math.max(0, h).toFixed(1)}" style="fill:${fill}" ${extra}/>`;
const text = (x, y, s, opts = {}) => `<text x="${x.toFixed(1)}" y="${y.toFixed(1)}" style="fill:${opts.fill || C.ink};font:${opts.weight || ""} ${opts.size || 12}px var(--sans)" text-anchor="${opts.anchor || "start"}">${s}</text>`;
const line = (x1, y1, x2, y2, stroke, extra = "") => `<line x1="${x1.toFixed(1)}" y1="${y1.toFixed(1)}" x2="${x2.toFixed(1)}" y2="${y2.toFixed(1)}" style="stroke:${stroke};stroke-width:1.5" ${extra}/>`;
const svg = (w, h, body) => `<svg viewBox="0 0 ${w} ${h}" width="100%" style="max-width:${w}px;border:1px solid var(--rule);border-radius:6px;background:#fff" role="img">${body}</svg>`;
const check = (label, value, onChange) => { const i = el("input", { type: "checkbox" }); i.checked = value; i.addEventListener("change", () => onChange(i.checked)); return el("label", {}, i, ` ${label}`); };
const ints = x => Math.round(x).toLocaleString("en-US");
const pow2 = v => ints(2 ** v);
const kTok = n => n >= 1024 ? `${fmt(n / 1024, 1)}K` : `${n}`;

// Log-log line chart. series: [{ f: x => y, color, dash?, label }]; mark: x of a vertical marker.
function logPlot({ x0, y0, w, h, xmin, xmax, series, mark, title, xlabel }) {
  const N = 60, xs = Array.from({ length: N + 1 }, (_, i) => xmin * (xmax / xmin) ** (i / N));
  const ys = series.flatMap(s => xs.map(s.f)).filter(y => y > 0 && Number.isFinite(y));
  const lo = Math.log10(Math.min(...ys)), hi = Math.log10(Math.max(...ys));
  const ylo = Math.floor(lo), yhi = Math.max(Math.ceil(hi), ylo + 1);
  const X = x => x0 + w * Math.log(x / xmin) / Math.log(xmax / xmin), Y = y => y0 + h - h * (Math.log10(Math.max(y, 10 ** ylo)) - ylo) / (yhi - ylo);
  let b = text(x0, y0 - 8, title, { fill: C.muted, size: 11 });
  b += `<rect x="${x0}" y="${y0}" width="${w}" height="${h}" style="fill:none;stroke:${C.rule}"/>`;
  for (let e = ylo; e <= yhi; e += Math.max(1, Math.round((yhi - ylo) / 4))) b += text(x0 - 4, Y(10 ** e) + 4, `1e${e}`, { fill: C.muted, size: 9, anchor: "end" });
  for (let e = Math.ceil(Math.log2(xmin)); e <= Math.log2(xmax); e += 4) b += text(X(2 ** e), y0 + h + 12, kTok(2 ** e), { fill: C.muted, size: 9, anchor: "middle" });
  b += text(x0 + w, y0 + h + 24, xlabel, { fill: C.muted, size: 10, anchor: "end" });
  series.forEach((s, i) => {
    const pts = xs.map(x => `${X(x).toFixed(1)},${Y(s.f(x)).toFixed(1)}`).join(" ");
    b += `<polyline points="${pts}" style="fill:none;stroke:${s.color};stroke-width:2" ${s.dash ? `stroke-dasharray="${s.dash}"` : ""}/>`;
    const ly = y0 + h + 36 + i * 14;
    b += line(x0, ly - 4, x0 + 16, ly - 4, s.color, s.dash ? `stroke-dasharray="${s.dash}"` : "") + text(x0 + 22, ly, s.label, { fill: s.color, size: 10 });
  });
  if (mark) { b += line(X(mark), y0, X(mark), y0 + h, C.ink, 'stroke-dasharray="3 3"'); series.forEach(s => { b += `<circle cx="${X(mark).toFixed(1)}" cy="${Y(s.f(mark)).toFixed(1)}" r="3.5" style="fill:${s.color}"/>`; }); }
  return b;
}

// ======================================================= 1. attention cost vs context ==
// Per head, whole sequence of n tokens. Three answers to the cost of attention, on one pair of axes.
//   softmax: n²d_k + n²d_v FLOPs (lecture_04.pdf:p4, txt L31, L39); keeps K and V for every token: n·(d_k + d_v) numbers.
//   linear (ρ = identity, Q(KᵀV)): 2·n·d_v·d_k FLOPs (p4, txt L39); recurrent state S_t is one d_k × d_v matrix (p5, txt L50).
//   DSA: an indexer scores every (query, earlier token) pair, so it is n² pairs ("not going to be linear in cost", video 26:52),
//     made cheap by low dimension / small w (video 27:00, 28:01): here each indexer pair costs 1/R of an attention pair;
//     full attention then runs on k selected tokens per query: n·k pairs (video 24:36). KV cache is unchanged (it must be
//     stored for the indexer to choose from). R and k are controls; the prompts use R = 64 and k = 2,048 (prompt
//     dsa-indexer-still-quadratic:check, sparse-attention-dsa:check), which are the KP author's, not the slide's.
function attentionCost({ n, dk, dv, k, R }) {
  const pair = dk + dv;
  const at = m => ({ idx: m * m * pair / R, att: m * Math.min(k, m) * pair });
  const softFlops = n * n * pair, linFlops = 2 * n * dv * dk, kvNums = n * pair, stateNums = dk * dv;
  const d1 = at(n), d2 = at(2 * n);
  return {
    softFlops, linFlops, flopRatio: softFlops / linFlops,
    kvNums, stateNums, memRatio: kvNums / stateNums, memCross: stateNums / pair, capPerToken: stateNums / n,
    idx: d1.idx, att: d1.att, dsa: d1.idx + d1.att, idxShare: d1.idx / (d1.idx + d1.att), dsaCross: R * k, dsaVsDense: (d1.idx + d1.att) / softFlops,
    idxGrowth2x: d2.idx / d1.idx, attGrowth2x: d2.att / d1.att, dsaKvRatio: kvNums / kvNums,
  };
}
MODELS["fixture:lecture_04--attention-cost"] = {
  fn: attentionCost,
  cases: [
    { args: { n: 8192, dk: 128, dv: 128, k: 2048, R: 64 }, pick: "flopRatio", expect: 64, tol: 0.05, from: "lecture_04:linear-attention-reassociation:transfer" },
    { args: { n: 128, dk: 128, dv: 128, k: 2048, R: 64 }, pick: "flopRatio", expect: 1 },              // crossover: n = d, the two forms cost the same
    { args: { n: 64, dk: 128, dv: 128, k: 2048, R: 64 }, pick: "flopRatio", expect: 0.5 },             // short context: the quadratic form is cheaper
    { args: { n: 4096, dk: 128, dv: 128, k: 2048, R: 64 }, pick: "memCross", expect: 64, from: "lecture_04:finite-state-recall-tradeoff:transfer" },
    { args: { n: 64, dk: 128, dv: 128, k: 2048, R: 64 }, pick: "memRatio", expect: 1 },                // at n = 64 the KV cache equals the state
    { args: { n: 1048576, dk: 128, dv: 128, k: 2048, R: 64 }, pick: "memRatio", expect: 16384 },      // 1M tokens: 2·128·2^20 / 128² (the transfer's "about 15,600" is 1e6)
    { args: { n: 1024, dk: 128, dv: 128, k: 2048, R: 64 }, pick: "capPerToken", expect: 16 },          // finite-state check: capacity per token ...
    { args: { n: 2048, dk: 128, dv: 128, k: 2048, R: 64 }, pick: "capPerToken", expect: 8 },           // ... halves when the context doubles
    { args: { n: 65536, dk: 128, dv: 128, k: 2048, R: 64 }, pick: "idxGrowth2x", expect: 4 },          // dsa predict: the indexer quadruples
    { args: { n: 65536, dk: 128, dv: 128, k: 2048, R: 64 }, pick: "attGrowth2x", expect: 2, from: "lecture_04:dsa-indexer-still-quadratic:transfer" },
    { args: { n: 1024, dk: 128, dv: 128, k: 2048, R: 64 }, pick: "attGrowth2x", expect: 4 },           // edge: k ≥ n selects everything, so it is dense again
    { args: { n: 4096, dk: 128, dv: 128, k: 2048, R: 64 }, pick: "dsaCross", expect: 131072, tol: 0.03, from: "lecture_04:dsa-indexer-still-quadratic:check" },
    { args: { n: 131072, dk: 128, dv: 128, k: 2048, R: 64 }, pick: "idxShare", expect: 0.5 },          // at n = R·k the two parts cost the same
    { args: { n: 131072, dk: 128, dv: 128, k: 2048, R: 64 }, pick: "dsaKvRatio", expect: 1 },          // sparse-attention-dsa check: KV cache unchanged
  ],
};
WIDGETS["fixture:lecture_04--attention-cost"] = (root) => {
  const s = { ln: 12, dk: 128, dv: 128, lk: 11, lR: 4 };
  const pic = el("div"), read = el("div", { class: "readout" });
  const draw = () => {
    const a = { n: 2 ** s.ln, dk: s.dk, dv: s.dv, k: 2 ** s.lk, R: 2 ** s.lR }, m = attentionCost(a), f = n => attentionCost({ ...a, n });
    let b = logPlot({ x0: 52, y0: 24, w: 250, h: 170, xmin: 64, xmax: 2 ** 20, mark: a.n, title: "compute per head, whole sequence (FLOPs)", xlabel: "context n",
      series: [{ f: n => f(n).softFlops, color: C.b, label: "softmax n²(d_k+d_v)" }, { f: n => f(n).linFlops, color: C.ok, label: "linear 2n·d_k·d_v" },
        { f: n => f(n).dsa, color: C.a, label: "DSA indexer + top-k" }, { f: n => f(n).idx, color: C.a, dash: "4 3", label: "DSA indexer alone" }] });
    b += logPlot({ x0: 372, y0: 24, w: 250, h: 170, xmin: 64, xmax: 2 ** 20, mark: a.n, title: "memory per head (numbers kept)", xlabel: "context n",
      series: [{ f: n => f(n).kvNums, color: C.b, label: "KV cache: softmax and DSA" }, { f: n => f(n).stateNums, color: C.ok, label: "linear state d_k×d_v" }] });
    pic.innerHTML = svg(640, 286, b);
    const rc = m.flopRatio > 1 ? `linear form cheaper (n > d: ${fmt(m.flopRatio, 2)}× fewer FLOPs)` : m.flopRatio < 1 ? `quadratic form cheaper (n &lt; d)` : "the two forms cost the same (n = d)";
    const rm = m.memRatio > 1 ? `the fixed state is smaller than the cache (${fmt(m.memRatio, 1)}×): it must compress` : "the cache is still no bigger than the state";
    const rd = a.k >= a.n ? `k ≥ n: DSA selects every token, so it is dense attention plus an indexer` : m.idxShare > 0.5 ? `indexer-dominated (n > R·k = ${ints(m.dsaCross)}): the n² part is now the bill` : `top-k attention dominates (n &lt; R·k = ${ints(m.dsaCross)})`;
    read.innerHTML = `n = ${ints(a.n)} tokens · d_k = ${a.dk}, d_v = ${a.dv} · DSA: k = ${ints(a.k)} selected, indexer pair = 1/${a.R} of an attention pair<br>
      <pre>softmax   ${fmt(m.softFlops, 3).padStart(10)} FLOPs   KV cache ${ints(m.kvNums).padStart(10)} numbers
linear    ${fmt(m.linFlops, 3).padStart(10)} FLOPs   state    ${ints(m.stateNums).padStart(10)} numbers   (${fmt(m.capPerToken, 3)} per context token)
DSA       ${fmt(m.dsa, 3).padStart(10)} FLOPs   KV cache ${ints(m.kvNums).padStart(10)} numbers   (indexer ${fmt(100 * m.idxShare, 1)}% of DSA cost)</pre>
      compute: <b>${rc}</b><br>memory: <b>${rm}</b> · the cache equals the state at n = d_k·d_v/(d_k+d_v) = <b>${fmt(m.memCross, 1)}</b> tokens<br>
      DSA: <b>${rd}</b> · if n doubles: indexer ×${fmt(m.idxGrowth2x, 2)}, top-k attention ×${fmt(m.attGrowth2x, 2)}, KV cache ×2 (same as softmax)<br>
      <span class="muted small">provenance: fixture:lecture_04--attention-cost · lecture_04.pdf:p4 (txt L31, L39: n²d_k + n²d_v vs 2n·d_v·d_k), p5 (L50: S_t is one d_k×d_v matrix), p12 (L113: lightweight indexer); video 24:36, 26:52-27:08 (indexer is all-to-all, made cheap). Indexer cost ratio R and k are controls; R = 64, k = 2,048 come from the KP prompts, not the slides. FLOPs count one multiply-add per term, as the slide does.</span>`;
  };
  root.append(el("div", { class: "widget" }, el("div", { class: "controls" },
    slider("context n", 6, 20, s.ln, 1, v => { s.ln = v; draw(); }, pow2),
    slider("d_k", 32, 256, s.dk, 32, v => { s.dk = v; draw(); }), slider("d_v", 32, 256, s.dv, 32, v => { s.dv = v; draw(); }),
    slider("DSA selected k", 8, 13, s.lk, 1, v => { s.lk = v; draw(); }, pow2),
    slider("indexer cheapness R", 0, 8, s.lR, 1, v => { s.lR = v; draw(); }, v => `1/${2 ** v} per pair`),
    read), pic)); draw();
};

// ============================================================ 2. KV cache by design ==
// Bytes the decoder keeps per token, and in total at a context length, for each answer the lecture gives to the growing cache.
//   Base formula: official/lectures/lecture_10.py:L297 kv_cache_size_per_seq = S·(K·H)·L·2·2 (2 for K and V, 2 bytes bf16);
//   defaults L = 40, K = 40, H = 128, S = 1024 are llama2_13b_config (L317-L325). Bytes per value is a control.
//   Hybrids: only 1 layer in (m+1) is softmax and keeps a cache; the m linear/SSM layers keep a fixed state that does not grow
//     (lecture_04.pdf:p6 txt L63 MiniMax 7-to-1; p8 L81 Nemotron 3 "3-1 ish"; p10 L99 Qwen Next 3-1). The fixed states are ignored here.
//   MLA: cache only the latent c_t^KV (p58, txt L466 "only need to store c_t^KV") plus a few non-latent rotated key dims (L475).
//     The widths 512 + 64 are DeepSeek-V2's, stated in prompt mla-latent-kv-cache:transfer; the slide says only "much smaller".
//   GiB = 2^30 bytes, as the prompts use.
function kvCacheModel({ L, heads, headDim, bytes, ctx, m, mla, latent, rope }) {
  const softLayers = Math.max(1, Math.floor(L / (m + 1)));
  const full = 2 * heads * headDim, perLayer = mla ? latent + rope : full;
  const perToken = softLayers * perLayer * bytes, base = L * full * bytes;
  return { softLayers, perLayer, full, perToken, total: perToken * ctx, gib: perToken * ctx / 2 ** 30, baseGib: base * ctx / 2 ** 30,
    fraction: perToken / base, layerFrac: softLayers / L, mlaShrink: full / perLayer };
}
MODELS["fixture:lecture_04--kv-cache-model"] = {
  fn: kvCacheModel,
  cases: [
    { args: { L: 32, heads: 32, headDim: 128, bytes: 2, ctx: 32768, m: 0, mla: false, latent: 512, rope: 64 }, pick: "perToken", expect: 524288, from: "lecture_04:supp-kv-cache-bytes:predict" },
    { args: { L: 32, heads: 32, headDim: 128, bytes: 2, ctx: 32768, m: 0, mla: false, latent: 512, rope: 64 }, pick: "gib", expect: 16 },          // predict's why: 16 GiB at 32k
    { args: { L: 32, heads: 32, headDim: 128, bytes: 2, ctx: 32768, m: 3, mla: false, latent: 512, rope: 64 }, pick: "gib", expect: 4, tol: 0.05, from: "lecture_04:supp-kv-cache-bytes:transfer" },
    { args: { L: 56, heads: 32, headDim: 128, bytes: 2, ctx: 32768, m: 7, mla: false, latent: 512, rope: 64 }, pick: "softLayers", expect: 7, from: "lecture_04:hybrid-attention-ratios:predict" },
    { args: { L: 32, heads: 32, headDim: 128, bytes: 2, ctx: 131072, m: 3, mla: false, latent: 512, rope: 64 }, pick: "fraction", expect: 0.25, tol: 0.05, from: "lecture_04:hybrid-attention-ratios:transfer" },
    { args: { L: 56, heads: 32, headDim: 128, bytes: 2, ctx: 131072, m: 7, mla: false, latent: 512, rope: 64 }, pick: "fraction", expect: 0.125 },   // 7-to-1 leaves an eighth
    { args: { L: 60, heads: 128, headDim: 128, bytes: 2, ctx: 4096, m: 0, mla: true, latent: 512, rope: 64 }, pick: "mlaShrink", expect: 56.9, tol: 0.05, from: "lecture_04:mla-latent-kv-cache:transfer" },
    { args: { L: 60, heads: 128, headDim: 128, bytes: 1, ctx: 1048576, m: 0, mla: true, latent: 512, rope: 64 }, pick: "fraction", expect: 1 / 56.889 }, // independent of dtype and length
    { args: { L: 60, heads: 128, headDim: 128, bytes: 2, ctx: 4096, m: 0, mla: true, latent: 512, rope: 0 }, pick: "mlaShrink", expect: 64 },     // no RoPE dims: 32768/512
    { args: { L: 40, heads: 40, headDim: 128, bytes: 2, ctx: 1024, m: 0, mla: false, latent: 512, rope: 64 }, pick: "total", expect: 838860800 },   // lecture_10.py Llama 2 13B, S = 1024
  ],
};
WIDGETS["fixture:lecture_04--kv-cache-model"] = (root) => {
  const s = { L: 40, heads: 40, headDim: 128, bytes: 2, lctx: 12, mi: 0, mla: false, latent: 512, rope: 64 };
  const M = [0, 1, 3, 7];
  const pic = el("div"), read = el("div", { class: "readout" });
  const draw = () => {
    const a = { L: s.L, heads: s.heads, headDim: s.headDim, bytes: s.bytes, ctx: 2 ** s.lctx, m: M[s.mi], mla: s.mla, latent: s.latent, rope: s.rope }, r = kvCacheModel(a);
    const W = 640, x0 = 150, span = 460;
    let b = text(10, 25, "layers", { size: 12 }) + text(10, 39, `${r.softLayers} of ${a.L} keep a cache`, { size: 10, fill: C.muted });
    const cw = span / a.L;
    for (let i = 0; i < a.L; i++) { const soft = (i % (a.m + 1)) === a.m && Math.floor(i / (a.m + 1)) < r.softLayers; b += rect(x0 + i * cw, 14, cw - (cw > 4 ? 1.5 : 0.3), 16, soft ? C.a : C.rule); }
    b += text(x0, 42, "blue: softmax layer (growing KV cache) · grey: linear / SSM layer (fixed state)", { size: 9, fill: C.muted });
    const bw = Math.max(2, span * r.fraction), inside = bw > span - 150;
    b += text(10, 66, "bytes per token", { size: 12 }) + text(10, 80, "all layers, full K and V", { size: 10, fill: C.muted });
    b += rect(x0, 58, span, 18, C.b, 'opacity="0.35"') + text(x0 + 6, 71, `${ints(r.perToken / r.fraction)} B`, { size: 11 });
    b += text(10, 102, "bytes per token", { size: 12 }) + text(10, 116, "this design", { size: 10, fill: C.muted });
    b += rect(x0, 94, bw, 18, C.a) + text(inside ? x0 + 6 : x0 + bw + 6, 107, `${ints(r.perToken)} B = ${fmt(100 * r.fraction, 2)}%`, { size: 11, fill: inside ? "#fff" : C.ink });
    const cell = mla => mla ? `latent ${a.latent} + rotated key ${a.rope} = ${r.perLayer} values` : `K ${a.heads}·${a.headDim} + V ${a.heads}·${a.headDim} = ${r.full} values`;
    b += text(10, 140, `per softmax layer per token: ${cell(a.mla)}`, { size: 11, fill: C.muted });
    pic.innerHTML = svg(W, 150, b);
    const design = (a.m ? `${a.m}-to-1 hybrid (${r.softLayers} softmax layers)` : "all-softmax") + (a.mla ? " with MLA latent cache" : " with full K,V");
    read.innerHTML = `design: <b>${design}</b> · context ${ints(a.ctx)} tokens · ${a.bytes} B per value<br>
      per token = softmax layers · values per layer · bytes = ${r.softLayers} · ${ints(r.perLayer)} · ${a.bytes} = <b>${ints(r.perToken)} B</b><br>
      at ${ints(a.ctx)} tokens: <span class="big">${fmt(r.gib, 3)} GiB</span> (all-softmax full K,V: ${fmt(r.baseGib, 3)} GiB) · the cache grows linearly with context in every design here<br>
      layers keeping a cache: ${fmt(100 * r.layerFrac, 1)}% · MLA shrink per layer: ${a.mla ? `<b>${fmt(r.mlaShrink, 2)}×</b>` : "off"} (same at any dtype and length)<br>
      <span class="muted small">provenance: fixture:lecture_04--kv-cache-model · official/lectures/lecture_10.py:L297 (S·(K·H)·L·2·2), defaults llama2_13b_config L317-L325; lecture_04.pdf:p6 (txt L63, 7-to-1), p8 (L81, 3-1), p10 (L99, 3-1), p58 (L466, cache only c_t^KV; L475, a few rotated key dims). Latent 512 + rope 64 are DeepSeek-V2 widths from the KP prompt, not the slide. Linear layers' fixed states are left out.</span>`;
  };
  root.append(el("div", { class: "widget" }, el("div", { class: "controls" },
    slider("layers L", 4, 128, s.L, 4, v => { s.L = v; draw(); }), slider("KV heads", 1, 128, s.heads, 1, v => { s.heads = v; draw(); }),
    slider("head dim", 32, 256, s.headDim, 32, v => { s.headDim = v; draw(); }),
    slider("bytes / value", 0, 2, 1, 1, v => { s.bytes = [1, 2, 4][v]; draw(); }, v => ["1 (fp8)", "2 (bf16)", "4 (fp32)"][v]),
    slider("context", 10, 20, s.lctx, 1, v => { s.lctx = v; draw(); }, pow2),
    slider("linear : softmax layers", 0, 3, s.mi, 1, v => { s.mi = v; draw(); }, v => M[v] ? `${M[v]}-to-1` : "all softmax"),
    check("MLA: cache a latent instead of K, V", s.mla, v => { s.mla = v; draw(); }),
    slider("MLA latent width", 64, 1024, s.latent, 64, v => { s.latent = v; draw(); }), slider("MLA rotated key dims", 0, 128, s.rope, 32, v => { s.rope = v; draw(); }),
    read), pic)); draw();
};

// ============================================================ 3. what the state remembers ==
// One stored pair, read back later with q = k. Keys are unit-norm and the filler tokens in between have keys orthogonal to k,
// so the slot under k is one number x and the three recurrences on the slides reduce to scalar updates:
//   linear attention   S_t = S_{t-1} + k_t v_tᵀ                         (lecture_04.pdf:p5, txt L50)       x ← x (+ v on a same-key write)
//   decay / gating     S_t = γ_t S_{t-1} + k_t v_tᵀ (RetNet: fixed γ)   (p5 L57 RetNet; p7 L70 Mamba-2)   x ← γx (+ v)
//   Gated DeltaNet     S_t = γ_t (I − β_t k_t k_tᵀ) S_{t-1} + β_t k_t v_tᵀ (p9, txt L85-L91)              x ← γx; on a same-key write γ(1−β)x + βv
// y_t = q_tᵀS_t reads x. Mamba-2's skip term v_tᵀD (L70) does not read the state and is left out. The toy values (old 4, new 10,
// γ = 0.95 over 20 steps, β = 0.5) are the prompts' (prompt gated-linear-attention-variants:check, linear-attention-recurrent-duality:check).
function stateMemory({ gamma, beta, old, vnew, age, write }) {
  const traj = { linear: [old], decay: [old], gdn: [old] };
  for (let t = 1; t <= age; t++) {
    const w = write && t === age, [l, d, g] = [traj.linear[t - 1], traj.decay[t - 1], traj.gdn[t - 1]];
    traj.linear.push(w ? l + vnew : l);
    traj.decay.push(w ? gamma * d + vnew : gamma * d);
    traj.gdn.push(w ? gamma * (1 - beta) * g + beta * vnew : gamma * g);
  }
  const wOld = { linear: 1, decay: gamma ** age, gdn: gamma ** age * (write ? 1 - beta : 1) };
  return { traj, linear: traj.linear[age], decay: traj.decay[age], gdn: traj.gdn[age], oldLinear: wOld.linear, oldDecay: wOld.decay, oldGdn: wOld.gdn };
}
MODELS["fixture:lecture_04--state-memory"] = {
  fn: stateMemory,
  cases: [
    { args: { gamma: 0.95, beta: 0.5, old: 4, vnew: 10, age: 20, write: false }, pick: "oldDecay", expect: 0.358, tol: 0.02, from: "lecture_04:linear-attention-recurrent-duality:check" },
    { args: { gamma: 1, beta: 0.5, old: 4, vnew: 10, age: 20, write: false }, pick: "oldDecay", expect: 1 },              // γ = 1 is plain linear attention: no forgetting
    { args: { gamma: 1, beta: 0.5, old: 4, vnew: 10, age: 1, write: true }, pick: "gdn", expect: 7, from: "lecture_04:gated-linear-attention-variants:check" },
    { args: { gamma: 1, beta: 1, old: 4, vnew: 10, age: 1, write: true }, pick: "gdn", expect: 10 },                      // gated predict: β = 1 replaces the old value
    { args: { gamma: 1, beta: 0, old: 4, vnew: 10, age: 1, write: true }, pick: "gdn", expect: 4 },                       // gated transfer: β = 0 is the no-input step
    { args: { gamma: 1, beta: 1, old: 4, vnew: 10, age: 1, write: true }, pick: "linear", expect: 14 },                   // linear attention adds: a blend, not a replace
    { args: { gamma: 0.9, beta: 1, old: 4, vnew: 10, age: 1, write: true }, pick: "decay", expect: 13.6 },                // γ alone scales the old value but still adds
  ],
};
WIDGETS["fixture:lecture_04--state-memory"] = (root) => {
  const s = { gamma: 0.9, beta: 0.8, old: 4, vnew: 10, age: 8, write: true };
  const pic = el("div"), read = el("div", { class: "readout" });
  const draw = () => {
    const r = stateMemory(s), all = [...r.traj.linear, ...r.traj.decay, ...r.traj.gdn, 0];
    const lo = Math.min(...all), hi = Math.max(...all, lo + 1), x0 = 50, y0 = 22, w = 560, h = 150;
    const X = t => x0 + w * t / s.age, Y = v => y0 + h - h * (v - lo) / (hi - lo);
    let b = text(x0, 14, `value stored under key k, read with q = k · step 0 writes ${s.old}; ${s.write ? `step ${s.age} writes ${s.vnew} under the same k` : "no later write"}`, { fill: C.muted, size: 11 });
    b += `<rect x="${x0}" y="${y0}" width="${w}" height="${h}" style="fill:none;stroke:${C.rule}"/>` + line(x0, Y(0), x0 + w, Y(0), C.rule);
    [...new Set([hi, lo, 0])].forEach(v => { b += text(x0 - 4, Y(v) + 4, fmt(v, 2), { fill: C.muted, size: 9, anchor: "end" }); });
    for (let t = 0; t <= s.age; t += Math.max(1, Math.round(s.age / 10))) b += text(X(t), y0 + h + 13, `${t}`, { fill: C.muted, size: 9, anchor: "middle" });
    b += text(x0 + w, y0 + h + 26, "steps after the first write (filler tokens use other keys)", { fill: C.muted, size: 10, anchor: "end" });
    const ser = [["linear", C.muted, "linear attention (adds)"], ["decay", C.ok, `decay γ = ${s.gamma} (RetNet / Mamba-2)`], ["gdn", C.a, `Gated DeltaNet γ = ${s.gamma}, β = ${s.beta}`]];
    ser.forEach(([key, col, lab], i) => {
      b += `<polyline points="${r.traj[key].map((v, t) => `${X(t).toFixed(1)},${Y(v).toFixed(1)}`).join(" ")}" style="fill:none;stroke:${col};stroke-width:2"/>`;
      b += `<circle cx="${X(s.age).toFixed(1)}" cy="${Y(r[key]).toFixed(1)}" r="3.5" style="fill:${col}"/>`;
      b += rect(x0 + 8, y0 + 8 + i * 14, 10, 3, col) + text(x0 + 22, y0 + 13 + i * 14, `${lab}: ${fmt(r[key], 3)}`, { fill: col, size: 10 });
    });
    pic.innerHTML = svg(640, 205, b);
    const gdnSays = !s.write ? "no same-key write: only γ acts" : s.beta === 1 ? "β = 1: the old value is erased and replaced" : s.beta === 0 ? "β = 0: the 'no input' step, nothing written or erased" : `β = ${s.beta}: keeps ${fmt(1 - s.beta, 2)} of the old value, writes ${s.beta} of the new`;
    read.innerHTML = `read at step ${s.age}: linear <b>${fmt(r.linear, 3)}</b> · decay <b>${fmt(r.decay, 3)}</b> · Gated DeltaNet <b>${fmt(r.gdn, 3)}</b><br>
      weight left on the step-0 pair: linear ${fmt(r.oldLinear, 3)} · decay γ^${s.age} = ${fmt(r.oldDecay, 4)} · Gated DeltaNet ${fmt(r.oldGdn, 4)}<br>
      Gated DeltaNet: <b>${gdnSays}</b>. γ scales the whole state every step; β acts only along the current key.<br>
      <span class="muted small">provenance: fixture:lecture_04--state-memory · lecture_04.pdf:p5 (txt L50, L57 RetNet), p7 (L70 Mamba-2, γ_t = f(x_t)), p9 (L85-L91 Gated DeltaNet, β = 0 'no input', erase along k_t). Unit, orthogonal keys reduce S to one number per key; toy values are the prompts'. Mamba-2's v_tᵀD skip term is omitted.</span>`;
  };
  root.append(el("div", { class: "widget" }, el("div", { class: "controls" },
    slider("decay γ", 0.5, 1, s.gamma, 0.01, v => { s.gamma = v; draw(); }, v => v.toFixed(2)),
    slider("input gate β (Gated DeltaNet)", 0, 1, s.beta, 0.05, v => { s.beta = v; draw(); }, v => v.toFixed(2)),
    slider("steps since the first write", 1, 40, s.age, 1, v => { s.age = v; draw(); }),
    slider("first value", -10, 10, s.old, 1, v => { s.old = v; draw(); }), slider("new value (same key)", -10, 10, s.vnew, 1, v => { s.vnew = v; draw(); }),
    check("last step writes under the same key", s.write, v => { s.write = v; draw(); }),
    read), pic)); draw();
};

// ============================================================ 4. MoE parameters vs active FLOPs ==
// Counted in units of one standard dense FFN of the same width (params ∝ size, FLOPs per token ∝ size of what runs).
//   lecture_04.pdf:p15 (txt L131) "increase the # experts without affecting FLOPs"; p16 (L133) same FLOP, more params.
//   p32: fine-grained experts + a few shared experts that are always on. p35 table (txt L270-L282): routed / active / shared /
//   fine-grained ratio for each model; the presets below are exactly those rows. Shared experts are given the same size as the
//   routed ones (DeepSeekMoE-style; the table gives one ratio per model). Router cost is ignored, as p15 does.
//   Upcycling (p51-p53): every expert starts as a full copy of the dense FFN, i.e. size 1.
const SIZES = [1, 1 / 2, 1 / 4, 1 / 8, 1 / 14], SIZE_LAB = ["1", "1/2", "1/4", "1/8", "1/14"];
const PRESETS = [["Mixtral", 8, 2, 0, 0], ["DBRX", 16, 4, 0, 0], ["DeepSeek v1", 64, 6, 2, 2], ["Qwen 1.5", 60, 4, 4, 3], ["DeepSeek v3", 256, 8, 1, 4], ["OlMoE", 64, 8, 0, 3], ["Llama 4 Maverick", 128, 1, 1, 1]];
const log10C = (n, r) => { let s = 0; for (let i = 0; i < r; i++) s += Math.log10((n - i) / (i + 1)); return s; };
function moeFfnBudget({ E, k, g, S, base = { E: 1, k: 1, g: 1, S: 0 } }) {
  const p = (E + S) * g, a = (k + S) * g, pb = (base.E + base.S) * base.g, ab = (base.k + base.S) * base.g;
  return { params: p, active: a, activeFrac: k / E, ratio: p / a, log10Combos: log10C(E, k), paramsVsBase: p / pb, flopsVsBase: a / ab };
}
MODELS["fixture:lecture_04--moe-ffn-budget"] = {
  fn: moeFfnBudget,
  cases: [
    { args: { E: 64, k: 2, g: 1, S: 0, base: { E: 8, k: 2, g: 1, S: 0 } }, pick: "flopsVsBase", expect: 1, from: "lecture_04:moe-params-without-flops:predict" },
    { args: { E: 64, k: 2, g: 1, S: 0, base: { E: 8, k: 2, g: 1, S: 0 } }, pick: "paramsVsBase", expect: 8 },   // predict's why: parameters went up 8x
    { args: { E: 64, k: 2, g: 1, S: 0 }, pick: "params", expect: 64 },                                            // moe-params transfer (a)
    { args: { E: 64, k: 2, g: 1, S: 0 }, pick: "active", expect: 2 },                                             // moe-params transfer (b)
    { args: { E: 128, k: 8, g: 1 / 4, S: 0 }, pick: "active", expect: 2, tol: 0.02, from: "lecture_04:moe-params-without-flops:check" },
    { args: { E: 128, k: 8, g: 1 / 4, S: 0 }, pick: "params", expect: 32 },                                       // check's why: 32 dense FFNs of parameters
    { args: { E: 64, k: 4, g: 1 / 4, S: 0, base: { E: 16, k: 1, g: 1, S: 0 } }, pick: "flopsVsBase", expect: 1 }, // fine-grained predict: unchanged
    { args: { E: 64, k: 4, g: 1 / 4, S: 0 }, pick: "log10Combos", expect: Math.log10(635376) },                   // C(64,4) vs 16 choices
    { args: { E: 256, k: 8, g: 1 / 14, S: 0, base: { E: 8, k: 2, g: 1, S: 0 } }, pick: "flopsVsBase", expect: 0.286, tol: 0.03, from: "lecture_04:fine-grained-shared-experts:check" },
    { args: { E: 8, k: 2, g: 1, S: 0 }, pick: "activeFrac", expect: 0.25 },                                       // fine-grained transfer: Mixtral 25%
    { args: { E: 256, k: 8, g: 1 / 14, S: 1 }, pick: "activeFrac", expect: 0.03125 },                             // fine-grained transfer: v3 3.1%
    { args: { E: 16, k: 2, g: 1, S: 1 }, pick: "active", expect: 3, from: "lecture_04:upcycling:check" },
    { args: { E: 16, k: 2, g: 1, S: 1 }, pick: "params", expect: 17 },                                            // upcycling check's why: 17x parameters
    { args: { E: 60, k: 4, g: 1, S: 4 }, pick: "params", expect: 64 },                                            // upcycling transfer: 64x parameters
    { args: { E: 60, k: 4, g: 1, S: 4 }, pick: "active", expect: 8 },                                             // upcycling transfer: 8x active
    { args: { E: 1, k: 1, g: 1, S: 0 }, pick: "ratio", expect: 1 },                                               // edge: one expert is the dense FFN
  ],
};
WIDGETS["fixture:lecture_04--moe-ffn-budget"] = (root) => {
  const s = { E: 32, k: 2, S: 0, gi: 0 };
  let base = { E: 1, k: 1, g: 1, S: 0 }, baseName = "one dense FFN";
  const pic = el("div"), read = el("div", { class: "readout" }), ctl = el("div", { class: "controls" });
  const draw = () => {
    s.k = Math.min(s.k, s.E);
    const g = SIZES[s.gi], a = { E: s.E, k: s.k, g, S: s.S, base }, m = moeFfnBudget(a);
    const cols = 32, cs = 17, rows = Math.ceil(s.E / cols), x0 = 20, y0 = 24, side = Math.max(3, Math.round((cs - 3) * Math.sqrt(g)));
    const chosen = new Set(Array.from({ length: s.k }, (_, i) => Math.floor(i * s.E / s.k + (s.E / s.k) / 2)));
    let b = text(x0, 15, `one token: ${s.S} shared (always on) + top-${s.k} of ${s.E} routed experts · square area ∝ expert size (${SIZE_LAB[s.gi]} of a dense FFN)`, { fill: C.muted, size: 11 });
    for (let i = 0; i < s.S; i++) b += rect(x0 + i * cs + (cs - side) / 2, y0 + (cs - side) / 2, side, side, C.ok);
    if (s.S) b += text(x0 + s.S * cs + 6, y0 + 12, "shared", { fill: C.ok, size: 10 });
    const ry = y0 + (s.S ? cs + 8 : 0);
    for (let i = 0; i < s.E; i++) { const cx = x0 + (i % cols) * cs, cy = ry + Math.floor(i / cols) * cs; b += rect(cx + (cs - side) / 2, cy + (cs - side) / 2, side, side, chosen.has(i) ? C.a : C.rule); }
    const by = ry + rows * cs + 14, span = 360, mx = Math.max(m.params, 1), lx = 150 + span + 10;
    b += text(x0, by + 13, "parameters", { size: 12 }) + rect(150, by, span * m.params / mx, 18, C.b, 'opacity="0.5"') + text(lx, by + 13, `${fmt(m.params, 3)} dense FFNs`, { size: 11 });
    b += text(x0, by + 39, "FLOPs / token", { size: 12 }) + rect(150, by + 26, Math.max(2, span * m.active / mx), 18, C.a) + text(lx, by + 39, `${fmt(m.active, 3)} dense FFNs`, { size: 11 });
    b += line(150 + span / mx, by + 46, 150 + span / mx, by + 52, C.ink) + text(150 + span / mx, by + 62, "↑ 1 dense FFN", { fill: C.muted, size: 9, anchor: mx > 3 ? "start" : "end" });
    pic.innerHTML = svg(640, by + 62, b);
    read.innerHTML = `parameters = (E + shared) · size = (${s.E} + ${s.S}) · ${SIZE_LAB[s.gi]} = <b>${fmt(m.params, 3)}</b> dense FFNs<br>
      FLOPs / token = (k + shared) · size = (${s.k} + ${s.S}) · ${SIZE_LAB[s.gi]} = <span class="big">${fmt(m.active, 3)}</span> dense FFNs · parameters / FLOPs = ${fmt(m.ratio, 2)}×<br>
      routed experts a token touches: ${s.k}/${s.E} = ${fmt(100 * m.activeFrac, 2)}% · distinct top-${s.k} sets C(${s.E}, ${s.k}) ≈ 10^${fmt(m.log10Combos, 2)}<br>
      vs pinned baseline (${baseName}): parameters <b>×${fmt(m.paramsVsBase, 3)}</b> · FLOPs per token <b>×${fmt(m.flopsVsBase, 3)}</b><br>
      <span class="muted small">provenance: fixture:lecture_04--moe-ffn-budget · lecture_04.pdf:p15 (txt L131), p16 (L133), p32 (shared, always on), p35 (L270-L282, presets are its rows), p51-p53 (upcycled experts are full copies, size 1). Shared experts take the routed size; router cost ignored (p15).</span>`;
  };
  const sl = { E: slider("routed experts E", 1, 256, s.E, 1, v => { s.E = v; draw(); }), k: slider("top-k", 1, 16, s.k, 1, v => { s.k = v; draw(); }),
    S: slider("shared experts", 0, 4, s.S, 1, v => { s.S = v; draw(); }), g: slider("expert size (of a dense FFN)", 0, 4, s.gi, 1, v => { s.gi = v; draw(); }, v => SIZE_LAB[v]) };
  const setSl = (lab, v) => { const i = lab.querySelector("input"); i.value = v; i.dispatchEvent(new Event("input")); };
  const presets = el("div", { class: "small", style: "margin-top:6px" }, "p35 rows: ", ...PRESETS.map(([n, E, k, S, gi]) => el("button", { class: "ghost", onclick: () => { Object.assign(s, { E, k, S, gi }); setSl(sl.E, E); setSl(sl.k, k); setSl(sl.S, S); setSl(sl.g, gi); } }, n)));
  const pin = el("div", { style: "margin-top:6px" }, el("button", { onclick: () => { base = { E: s.E, k: s.k, g: SIZES[s.gi], S: s.S }; baseName = `E ${s.E}, k ${s.k}, shared ${s.S}, size ${SIZE_LAB[s.gi]}`; draw(); } }, "pin current as baseline"), " ",
    el("button", { class: "ghost", onclick: () => { base = { E: 1, k: 1, g: 1, S: 0 }; baseName = "one dense FFN"; draw(); } }, "baseline = dense FFN"));
  ctl.append(sl.E, sl.k, sl.S, sl.g, presets, pin, read);
  root.append(el("div", { class: "widget" }, ctl, pic)); draw();
};

// ============================================================ 5. capacity and dropped tokens ==
// Top-1 token-choice routing of one batch of T tokens over E experts, each with a fixed capacity of C token slots per batch.
// One "hot" expert receives your request's tokens plus the tokens other requests in the batch route to it; the rest of the
// batch is spread evenly over the other experts. Tokens beyond an expert's capacity are dropped (they skip that FFN).
//   lecture_04.pdf:p47 (txt L379-L380) "Token dropping from routing happens at a batch level – ... other people's queries can
//   drop your token"; video 1:15:42 "my queue is so long, I have to start dropping tokens".
//   Balancing push: p40 (txt L332-L333) d loss / d p_i(x) = (αN/T²)·Σ_x 1[argmax p(x) = i], proportional to the tokens routed
//   to i, "so more frequent use = stronger downweighting". The batch shapes are the prompts' (token-dropping-stochasticity,
//   load-balancing-loss); which of the hot expert's tokens are dropped depends on processing order, which the slides leave open.
function capacityDrop({ T, E, C, mine, other }) {
  const hot = Math.min(T, mine + other), rest = T - hot, nb = Math.floor(rest / (E - 1)), rem = rest - nb * (E - 1);
  const loads = [hot, ...Array.from({ length: E - 1 }, (_, i) => nb + (i < rem ? 1 : 0))];
  const drops = loads.map(l => Math.max(0, l - C)), dropped = drops.reduce((x, y) => x + y, 0);
  return { loads, drops, dropped, droppedHot: drops[0], aloneHot: Math.max(0, Math.min(T, mine) - C), mean: T / E, slots: E * C,
    capFactor: C * E / T, pushRatio: hot / (rest / (E - 1)), fHot: hot / T };
}
MODELS["fixture:lecture_04--capacity-drop"] = {
  fn: capacityDrop,
  cases: [
    { args: { T: 512, E: 8, C: 96, mine: 150, other: 0 }, pick: "dropped", expect: 54, from: "lecture_04:token-dropping-stochasticity:predict" },
    { args: { T: 512, E: 8, C: 96, mine: 150, other: 0 }, pick: "slots", expect: 768 },        // predict: 768 slots, far above 512
    { args: { T: 256, E: 8, C: 64, mine: 40, other: 50 }, pick: "droppedHot", expect: 26, from: "lecture_04:token-dropping-stochasticity:check" },
    { args: { T: 256, E: 8, C: 64, mine: 40, other: 50 }, pick: "dropped", expect: 26 },       // no other expert overflows
    { args: { T: 256, E: 8, C: 64, mine: 40, other: 50 }, pick: "aloneHot", expect: 0 },       // alone, your 40 tokens all fit
    { args: { T: 256, E: 8, C: 64, mine: 40, other: 50 }, pick: "mean", expect: 32 },          // the average load is half the capacity
    { args: { T: 512, E: 8, C: 64, mine: 64, other: 0 }, pick: "dropped", expect: 0 },         // edge: perfectly balanced at capacity
    { args: { T: 520, E: 8, C: 64, mine: 65, other: 0 }, pick: "dropped", expect: 8 },         // edge: T > E·C drops even when balanced
    { args: { T: 400, E: 2, C: 400, mine: 300, other: 0 }, pick: "pushRatio", expect: 3, from: "lecture_04:load-balancing-loss:predict" },
  ],
};
WIDGETS["fixture:lecture_04--capacity-drop"] = (root) => {
  const s = { T: 256, E: 8, C: 48, mine: 40, other: 0 };
  const pic = el("div"), read = el("div", { class: "readout" });
  const draw = () => {
    s.mine = Math.min(s.mine, s.T); s.other = Math.min(s.other, s.T - s.mine);
    const m = capacityDrop(s), W = 640, x0 = 50, y0 = 26, h = 170, top = Math.max(...m.loads, s.C, 1) * 1.08;
    const bw = Math.min(64, 560 / s.E), Y = v => y0 + h - h * v / top;
    let b = text(x0, 14, `${s.T} tokens, top-1, ${s.E} experts, ${s.C} slots each · blue: your request · orange: other requests · pale: dropped`, { fill: C.muted, size: 11 });
    b += line(x0, y0 + h, x0 + s.E * bw, y0 + h, C.rule);
    m.loads.forEach((l, i) => {
      const x = x0 + i * bw, kept = Math.min(l, s.C), mineKept = i === 0 ? Math.min(s.mine, kept) : 0, w = bw - 8;
      if (i === 0) { b += rect(x, Y(mineKept), w, h * mineKept / top, C.a) + rect(x, Y(kept), w, h * (kept - mineKept) / top, C.b); }
      else b += rect(x, Y(kept), w, h * kept / top, C.rule);
      if (m.drops[i]) b += rect(x, Y(l), w, h * m.drops[i] / top, C.b, 'opacity="0.3"') + text(x + w / 2, Y(l) - 4, `−${m.drops[i]}`, { fill: C.b, size: 11, anchor: "middle", weight: "bold" });
      b += text(x + w / 2, y0 + h + 13, i === 0 ? "E0 hot" : `E${i}`, { fill: C.muted, size: 10, anchor: "middle" }) + text(x + w / 2, y0 + h + 25, `${l}`, { fill: C.ink, size: 10, anchor: "middle" });
    });
    b += line(x0 - 4, Y(s.C), x0 + s.E * bw, Y(s.C), C.ink, 'stroke-dasharray="5 3"') + text(x0 - 6, Y(s.C) + 4, `cap ${s.C}`, { size: 9, anchor: "end" });
    b += line(x0 - 4, Y(m.mean), x0 + s.E * bw, Y(m.mean), C.ok, 'stroke-dasharray="2 3"') + text(x0 - 6, Y(m.mean) + 4, `mean`, { size: 9, anchor: "end", fill: C.ok });
    pic.innerHTML = svg(W, 230, b);
    const regime = m.dropped === 0 ? "no drops" : s.T > m.slots ? `over-subscribed: ${s.T} tokens > ${m.slots} slots, drops even if routing were perfectly balanced`
      : `drops at the hot expert although the mean load (${fmt(m.mean, 1)}) is below capacity: dropping is a tail event, not an average`;
    read.innerHTML = `hot expert E0: ${s.mine} yours + ${s.other} others = ${m.loads[0]} for ${s.C} slots → <span class="big">${m.droppedHot} dropped</span> there, ${m.dropped} dropped in the batch<br>
      your request alone at E0: ${m.aloneHot} dropped · so ${m.droppedHot > m.aloneHot ? "<b>who else is in the batch decides whether your tokens are dropped</b>" : "co-batched traffic changes nothing here"}<br>
      regime: <b>${regime}</b> · capacity factor C·E/T = ${fmt(m.capFactor, 2)} · slots ${m.slots}<br>
      balancing-loss push on p_i ∝ tokens routed to i: E0 vs an average other expert = <b>${fmt(m.pushRatio, 3)}×</b><br>
      <span class="muted small">provenance: fixture:lecture_04--capacity-drop · lecture_04.pdf:p47 (txt L379-L380, dropping is batch-level); video 1:15:42-1:16:18; p40 (L332-L333, the push is ∝ Σ 1[argmax = i]). Top-1 routing with the rest spread evenly is the prompts' setup, not a learned router.</span>`;
  };
  root.append(el("div", { class: "widget" }, el("div", { class: "controls" },
    slider("tokens in batch T", 64, 1024, s.T, 8, v => { s.T = v; draw(); }), slider("experts E", 2, 16, s.E, 1, v => { s.E = v; draw(); }),
    slider("capacity C per expert", 8, 512, s.C, 4, v => { s.C = v; draw(); }),
    slider("your tokens to E0", 0, 512, s.mine, 2, v => { s.mine = v; draw(); }), slider("other requests' tokens to E0", 0, 512, s.other, 2, v => { s.other = v; draw(); }),
    read), pic)); draw();
};
