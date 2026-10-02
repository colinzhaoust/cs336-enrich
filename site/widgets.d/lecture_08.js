// Widgets for thread lecture_08 (CS336 L8, parallelism basics). Register as "fixture:<id>" -> (root, notice) => void.
// Each widget has a pure model in MODELS (no DOM) that tools/check_widgets.mjs tests against the KPs' stored answers.
// Sources: lectures/lecture_08/lecture_08.txt (slide text layer, cited as pdf page + txt line),
//          lectures/lecture_08/transcript.json (cited as video M:SS), and the KP filtering notes that already
//          record which numbers are the author's (marked "author" below). No constant here is new.
import { el, fmt, slider } from "../widgets_lib.js";

export const MODELS = {};
export const WIDGETS = {};

const C = { ink: "var(--ink)", muted: "var(--muted)", rule: "var(--rule)", a: "var(--accent)", b: "var(--accent2)", ok: "var(--ok)", hi: "var(--hilite)" };
const rect = (x, y, w, h, fill, extra = "") => `<rect x="${x.toFixed(1)}" y="${y}" width="${Math.max(0, w).toFixed(1)}" height="${h}" style="fill:${fill}" ${extra}/>`;
const text = (x, y, s, opts = {}) => `<text x="${x.toFixed(1)}" y="${y}" style="fill:${opts.fill || C.ink};font:${opts.size || 12}px var(--sans)" text-anchor="${opts.anchor || "start"}">${s}</text>`;
const svg = (w, h, body) => `<svg viewBox="0 0 ${w} ${h}" width="100%" style="max-width:${w}px;border:1px solid var(--rule);border-radius:6px;background:#fff" role="img">${body}</svg>`;
const check = (label, value, onChange) => { const i = el("input", { type: "checkbox" }); i.checked = value; i.addEventListener("change", () => onChange(i.checked)); return el("label", {}, i, ` ${label}`); };

// ============================================================ 1. ZeRO ledger ==
// Per-GPU bytes per parameter by ZeRO stage, the largest model that fits, and per-step communication.
//   lecture_08.pdf:p17 (txt L157-L163): 2 B BF16 params + 2 B BF16 grads + 4 B FP32 master + 4 (or 2) + 4 (or 2) B Adam moments.
//   K = optimizer-state bytes: 12 with FP32 moments (p17, p21's "(4+K)"), 8 with BF16 moments (p28 "12 bytes per param", L271-L272).
//   p21 (L190-L195): DDP memory (4+K)·Ψ, ZeRO-1 (4+K/N_gpu)·Ψ; communication 2·#params for both.
//   p28 (L277-L282): stage 2 = 2 (param) + (10 (grad+state))/8, stage 3 = 12/8, max size = 80 GB / (bytes per param per GPU).
//   p27 (L257-L265): stage 1 and 2 are 2·#params, stage 3 is 3·#params ("1.5x comm cost").
// GB = 1e9 bytes, as the p28 table uses (80/12 = 6.66B).
function zeroLedger({ psi, N, K, mem }) {
  const bpp = [4 + K, 4 + K / N, 2 + (2 + K) / N, (4 + K) / N];           // bytes / param / GPU, stages 0-3
  const floor = [4 + K, 4, 2, 0];                                           // the part that does not shrink with N
  const comm = [2, 2, 2, 3];                                                // × #params per step
  const o = { bpp, floor, comm };
  bpp.forEach((x, i) => {
    o[`bpp${i}`] = x;
    o[`gb${i}`] = x * psi / 1e9;                                            // per-GPU GB of model state
    o[`max${i}`] = mem * 1e9 / x;                                           // parameters that fit in `mem` GB
    o[`max${i}B`] = mem / x;                                                // same, in billions
    o[`comm${i}`] = comm[i] * psi;                                          // parameter-units per step
  });
  o.commRatio3 = comm[3] / comm[0];
  return o;
}
MODELS["fixture:lecture_08--zero-ledger"] = {
  fn: zeroLedger,
  cases: [
    { args: { psi: 1e9, N: 8, K: 8, mem: 80 }, pick: "max1B", expect: 16, tol: 0.03, from: "lecture_08:zero-fit-table:predict" },
    { args: { psi: 1e9, N: 4, K: 8, mem: 80 }, pick: "max1", expect: 13.3e9, tol: 0.03, from: "lecture_08:zero-fit-table:check" },
    { args: { psi: 1e9, N: 8, K: 8, mem: 80 }, pick: "max0B", expect: 6.667 },            // p28 baseline 6.66..B
    { args: { psi: 1e9, N: 8, K: 8, mem: 80 }, pick: "max2B", expect: 24.62 },            // p28 stage 2
    { args: { psi: 1e9, N: 8, K: 8, mem: 80 }, pick: "max3B", expect: 53.33 },            // p28 stage 3
    { args: { psi: 1e9, N: 16, K: 8, mem: 80 }, pick: "max2B", expect: 30.48 },           // zero-fit-table transfer
    { args: { psi: 1e9, N: 16, K: 8, mem: 80 }, pick: "max3B", expect: 106.67 },          // zero-fit-table transfer
    { args: { psi: 1e9, N: 1e9, K: 8, mem: 80 }, pick: "max2B", expect: 40 },             // edge: stage 2 saturates at 80/2
    { args: { psi: 10e9, N: 16, K: 12, mem: 80 }, pick: "gb0", expect: 160 },             // zero-stages-1-2 transfer, DDP
    { args: { psi: 10e9, N: 16, K: 12, mem: 80 }, pick: "gb1", expect: 47.5 },            // zero-stages-1-2 transfer, ZeRO-1
    { args: { psi: 20e9, N: 1e9, K: 12, mem: 80 }, pick: "gb1", expect: 80, tol: 0.05, from: "lecture_08:zero-stages-1-2:check" },
    { args: { psi: 7e9, N: 32, K: 12, mem: 80 }, pick: "comm0", expect: 14e9, tol: 0.05, from: "lecture_08:naive-data-parallel-accounting:check" },
    { args: { psi: 10e9, N: 8, K: 12, mem: 80 }, pick: "commRatio3", expect: 1.5, from: "lecture_08:zero-stage-3-fsdp:predict" },
    { args: { psi: 10e9, N: 8, K: 12, mem: 80 }, pick: "comm3", expect: 30e9, tol: 0.05, from: "lecture_08:zero-stage-3-fsdp:check" },
    { args: { psi: 7e9, N: 1, K: 12, mem: 80 }, pick: "gb0", expect: 112 },               // naive DDP: N does not enter
    { args: { psi: 7e9, N: 8, K: 12, mem: 80 }, pick: "gb0", expect: 112 },               // naive-data-parallel predict: unchanged
  ],
};
WIDGETS["fixture:lecture_08--zero-ledger"] = (root) => {
  const s = { psiB: 10, logN: 1, K: 12, mem: 80 };
  const pic = el("div"), read = el("div", { class: "readout" });
  const names = ["naive DDP", "ZeRO-1 (opt. state)", "ZeRO-2 (+ grads)", "ZeRO-3 / FSDP (+ params)"];
  const forms = K => [`${4 + K}`, `4 + ${K}/N`, `2 + ${2 + K}/N`, `${4 + K}/N`];
  const draw = () => {
    const N = 2 ** s.logN, m = zeroLedger({ psi: s.psiB * 1e9, N, K: s.K, mem: s.mem });
    const W = 640, x0 = 170, span = 330, top = 26, row = 46;
    const scale = Math.max(m.gb0, s.mem * 1.15);                            // the DDP bar or the GPU line, whichever is longer
    let b = text(10, 16, `per-GPU model state · dark: replicated on every GPU · light: sharded 1/N · dashed: ${s.mem} GB GPU`, { fill: C.muted, size: 11 });
    m.bpp.forEach((x, i) => {
      const y = top + i * row, rep = m.floor[i] * s.psiB, tot = m[`gb${i}`], fits = tot <= s.mem;
      b += text(10, y + 14, names[i], { size: 12 });
      b += text(10, y + 29, `${forms(s.K)[i]} B/param · comm ${m.comm[i]}Ψ`, { fill: C.muted, size: 11 });
      b += rect(x0, y + 2, span * Math.min(rep, tot) / scale, 20, C.a);
      b += rect(x0 + span * Math.min(rep, tot) / scale, y + 2, span * Math.max(0, tot - rep) / scale, 20, C.a, 'opacity="0.35"');
      b += text(Math.min(x0 + span * tot / scale, x0 + span) + 6, y + 17, `${fmt(tot, 1)} GB ${fits ? "fits" : "does not fit"}`, { fill: fits ? C.ok : C.b, size: 12 });
    });
    const lx = x0 + span * s.mem / scale;
    b += `<line x1="${lx.toFixed(1)}" y1="${top - 2}" x2="${lx.toFixed(1)}" y2="${top + 4 * row - 18}" style="stroke:${C.b};stroke-width:1.5;stroke-dasharray:4 3"/>`;
    pic.innerHTML = svg(W, top + 4 * row - 6, b);
    read.innerHTML = `Ψ = ${s.psiB} B params · N = ${N} GPUs · K = ${s.K} B optimizer state (${s.K === 12 ? "FP32 master + FP32 moments, 16 B/param" : "FP32 master + BF16 moments, 12 B/param"})<br>
      <pre>${names.map((n, i) => `${n.padEnd(26)} ${String(fmt(m[`bpp${i}`], 3)).padStart(7)} B/param/GPU  ${String(fmt(m[`gb${i}`], 1)).padStart(8)} GB   max on ${s.mem} GB: ${String(fmt(m[`max${i}B`], 2)).padStart(7)} B params   comm ${m.comm[i]}Ψ = ${fmt(m[`comm${i}`] / 1e9, 1)} B`).join("\n")}</pre>
      communication: stages 0-2 move 2Ψ per step, stage 3 moves 3Ψ = <b>${m.commRatio3}× DDP</b><br>
      <span class="muted small">provenance: fixture:lecture_08--zero-ledger · lecture_08.pdf:p17 (L157-L163, 16 B/param; moments 4 or 2 B), p21 (L190-L195, (4+K) vs (4+K/N_gpu), both 2·#params), p27 (L257-L265, stage 3 is 3·#params), p28 (L271-L282, 12 B/param on 8× A100-80G: 6.66 / 16 / 24.62 / 53.33 B). Ignores activations, buffers and fragmentation, as the slides do.</span>`;
  };
  root.append(el("div", { class: "widget" }, el("div", { class: "controls" },
    slider("parameters Ψ (billions)", 1, 200, s.psiB, 1, v => { s.psiB = v; draw(); }),
    slider("GPUs N", 0, 10, s.logN, 1, v => { s.logN = v; draw(); }, v => `${2 ** v}`),
    slider("Adam moments", 8, 12, s.K, 4, v => { s.K = v; draw(); }, v => v === 12 ? "FP32 (K = 12, p17)" : "BF16 (K = 8, p28)"),
    slider("GPU memory (GB)", 16, 192, s.mem, 1, v => { s.mem = v; draw(); }),
    read), pic)); draw();
};

// ====================================================== 2. global-batch budget ==
// GPUs = TP × PP × DP (lecture_08.pdf:p57, txt L541-L547: model parallel until it fits, "then until you run out of GPUs"
// scale with data parallel). Each replica gets B/DP sequences, i.e. n_micro = B/(DP·b) micro-batches; bubble/compute
// ratio = (n_stages−1)/n_micro (p34, txt L339 "so we need a big batch size!"); idle fraction (p−1)/(m+p−1) is the
// shared pipeline-bubbles animation's readout. The link "batch is spent twice" is spoken (video 29:08-29:25, 33:52-34:13);
// n_micro = B/(d·b) is the author's (Narayanan et al. 2021 accounting; KP global-batch-budget filtering note).
function batchBudget({ G, TP, PP, B, b }) {
  const dp = Math.floor(G / (TP * PP));
  const perReplica = dp >= 1 ? B / dp : NaN, nMicro = perReplica / b;
  return { dp, used: dp * TP * PP, perReplica, nMicro, bubble: PP > 1 ? (PP - 1) / nMicro : 0, idle: (PP - 1) / (nMicro + PP - 1) };
}
MODELS["fixture:lecture_08--batch-budget"] = {
  fn: batchBudget,
  cases: [
    { args: { G: 512, TP: 1, PP: 8, B: 512, b: 1 }, pick: "bubble", expect: 0.875, tol: 0.02, from: "lecture_08:global-batch-budget:predict" },
    { args: { G: 128, TP: 1, PP: 8, B: 512, b: 1 }, pick: "bubble", expect: 0.21875 },     // the predict's starting state (16 replicas)
    { args: { G: 512, TP: 1, PP: 8, B: 512, b: 2 }, pick: "bubble", expect: 1.75 },        // global-batch-budget check: b 2 -> 1 halves this
    { args: { G: 256, TP: 1, PP: 16, B: 1024, b: 1 }, pick: "bubble", expect: 0.234375 },  // transfer: DP 16 meets <= 0.25
    { args: { G: 512, TP: 1, PP: 16, B: 1024, b: 1 }, pick: "bubble", expect: 0.46875 },   // transfer: DP 32 does not
    { args: { G: 4, TP: 1, PP: 4, B: 4, b: 1 }, pick: "bubble", expect: 0.75, tol: 0.02, from: "lecture_08:pipeline-bubble:predict" },
    { args: { G: 4, TP: 1, PP: 4, B: 4, b: 1 }, pick: "idle", expect: 3 / 7 },            // the animation's 43% idle for the same case
    { args: { G: 7, TP: 1, PP: 7, B: 24, b: 1 }, pick: "bubble", expect: 0.25 },          // pipeline-bubble check: 7 stages is the edge
    { args: { G: 8, TP: 1, PP: 8, B: 24, b: 1 }, pick: "bubble", expect: 7 / 24 },        // 8 stages crosses 0.25
    { args: { G: 8, TP: 1, PP: 8, B: 32, b: 1 }, pick: "bubble", expect: 0.219, tol: 0.01 }, // pipeline-bubble transfer
    { args: { G: 1024, TP: 8, PP: 4, B: 1024, b: 1 }, pick: "dp", expect: 32, from: "lecture_08:three-d-parallelism-recipe:predict" },
    { args: { G: 512, TP: 8, PP: 8, B: 1024, b: 1 }, pick: "perReplica", expect: 128, tol: 0.05, from: "lecture_08:three-d-parallelism-recipe:check" },
    { args: { G: 1024, TP: 8, PP: 4, B: 256, b: 1 }, pick: "perReplica", expect: 8 },     // three-d transfer: 8 sequences per replica
  ],
};
WIDGETS["fixture:lecture_08--batch-budget"] = (root) => {
  const s = { logG: 7, logTP: 0, PP: 8, logB: 9, logb: 0 };
  const pic = el("div"), read = el("div", { class: "readout" });
  const draw = () => {
    const a = { G: 2 ** s.logG, TP: 2 ** s.logTP, PP: s.PP, B: 2 ** s.logB, b: 2 ** s.logb }, m = batchBudget(a);
    const W = 640, x0 = 20, span = 600;
    let body = text(x0, 16, `global batch B = ${a.B} sequences, shared by ${Math.max(m.dp, 0)} data-parallel replicas`, { fill: C.muted, size: 11 });
    let regime;
    if (m.dp < 1) regime = `model does not fit: TP × PP = ${a.TP * a.PP} > ${a.G} GPUs`;
    else if (m.perReplica < a.b) regime = `infeasible: ${m.dp} replicas × micro-batch ${a.b} > B = ${a.B} (#replicas must stay below the batch, p29)`;
    else if (a.PP === 1) regime = "no pipeline: no bubble (data parallel only)";
    else regime = m.bubble >= 1 ? `bubble-dominated: idle time ≥ useful compute (ratio ${fmt(m.bubble, 3)})` : `compute-dominated: bubble is ${fmt(m.bubble, 3)} of useful compute`;
    if (m.dp >= 1) {
      const rw = span / Math.min(m.dp, 64);                                  // draw at most 64 replica blocks
      for (let i = 0; i < Math.min(m.dp, 64); i++) body += rect(x0 + i * rw, 24, rw - (rw > 3 ? 1.5 : 0), 18, i === 0 ? C.a : C.rule);
      if (m.dp > 64) body += text(x0 + span, 56, `(first 64 of ${m.dp} replicas)`, { fill: C.muted, size: 10, anchor: "end" });
      body += text(x0, 56, `replica 1 gets ${fmt(m.perReplica, 2)} sequences = ${fmt(m.nMicro, 2)} micro-batches of ${a.b}`, { fill: C.a, size: 11 });
      if (m.nMicro >= 1 && a.PP > 1) {
        const units = m.nMicro + a.PP - 1, u = span / units;
        body += text(x0, 82, `one pipeline's step: ${fmt(m.nMicro, 2)} units of work + ${a.PP - 1} units of bubble (fill + drain)`, { fill: C.muted, size: 11 });
        body += rect(x0, 90, u * m.nMicro, 22, C.a) + rect(x0 + u * m.nMicro, 90, u * (a.PP - 1), 22, C.b, 'opacity="0.75"');
        body += text(x0 + 4, 105, "useful", { fill: "#fff", size: 11 }) + text(x0 + span - 4, 105, "bubble", { fill: "#fff", size: 11, anchor: "end" });
      }
    }
    body += text(x0, 134, regime, { fill: m.dp >= 1 && m.bubble < 1 ? C.ok : C.b, size: 12 });
    pic.innerHTML = svg(W, 144, body);
    read.innerHTML = `DP = ⌊G / (TP × PP)⌋ = ⌊${a.G} / (${a.TP} × ${a.PP})⌋ = <b>${m.dp}</b>${m.dp >= 1 && m.used < a.G ? ` <span class="muted">(${a.G - m.used} GPUs left over)</span>` : ""}<br>
      sequences per replica B / DP = ${fmt(m.perReplica, 3)} · n_micro = B / (DP · b) = ${fmt(m.nMicro, 3)}${Number.isInteger(m.nMicro) || !Number.isFinite(m.nMicro) ? "" : " (not a whole number)"}<br>
      bubble / compute = (PP − 1) / n_micro = <span class="big">${a.PP > 1 && m.dp >= 1 ? fmt(m.bubble, 3) : "–"}</span>
      idle share of the step (PP − 1)/(n_micro + PP − 1) = ${a.PP > 1 && m.dp >= 1 ? fmt(m.idle, 3) : "–"}<br>
      <span class="muted small">provenance: fixture:lecture_08--batch-budget · lecture_08.pdf:p34 (L339, bubble/useful = (n_stages−1)/n_micro, "we need a big batch size!"), p29 (L284, #machines < batch size), p57 (L541-L547, model parallel until it fits, then data parallel); video 29:08-29:25 and 33:52-34:13 (batch size is a resource data parallel and pipelining both spend); n_micro = B/(DP·b) is Narayanan et al. 2021 accounting (author).</span>`;
  };
  root.append(el("div", { class: "widget" }, el("div", { class: "controls" },
    slider("GPUs G", 0, 14, s.logG, 1, v => { s.logG = v; draw(); }, v => `${2 ** v}`),
    slider("tensor parallel TP", 0, 3, s.logTP, 1, v => { s.logTP = v; draw(); }, v => `${2 ** v}`),
    slider("pipeline stages PP", 1, 32, s.PP, 1, v => { s.PP = v; draw(); }),
    slider("global batch B (sequences)", 0, 12, s.logB, 1, v => { s.logB = v; draw(); }, v => `${2 ** v}`),
    slider("micro-batch size b", 0, 4, s.logb, 1, v => { s.logb = v; draw(); }, v => `${2 ** v}`),
    read), pic)); draw();
};

// ================================================ 3. activation memory per layer ==
// lecture_08.pdf:p46 (txt ~L440, video 46:44-47:08): storing everything, sbh(34 + 5as/h) bytes per layer.
// p47 (L446-L447, video 48:11-48:27): with tensor parallel t, sbh(10 + 24/t + 5as/(ht)); the 10 = LayerNorm 4 + dropout 2 + sublayer inputs 4.
// p48-p49 (L459, video 51:21-51:53): sequence parallel splits the 10 too: sbh(34 + 5as/h)/t; recomputing attention drops 5as/h, leaving sbh·34/t.
// Recompute cost (selective-recomputation KP, author, after Korthikanti et al. 2022): forward FLOPs per layer 24bsh² + 4bs²h;
// attention core 4bs²h → share s/(6h+s); MLP 16bsh² → share 16h/(24h+4s).
function activationMemory({ s, h, a, t = 1, sp = false, recompute = false }) {
  const attn = 5 * a * s / h, keptAttn = recompute ? 0 : attn;
  const parts = { pointwise: sp ? 10 / t : 10, matmul: 24 / t, attention: keptAttn / t };
  const coef = parts.pointwise + parts.matmul + parts.attention;
  return { ...parts, attn, coef, full: 34 + attn, gbPerSeq: s * h * coef / 1e9, recompShare: s / (6 * h + s), mlpShare: 16 * h / (24 * h + 4 * s), attnShareStored: attn / (34 + attn) };
}
MODELS["fixture:lecture_08--activation-memory"] = {
  fn: activationMemory,
  cases: [
    { args: { s: 4096, h: 8192, a: 64, t: 4 }, pick: "coef", expect: 56, tol: 0.05, from: "lecture_08:activation-memory-tp:check" },
    { args: { s: 2048, h: 4096, a: 32, t: 1 }, pick: "coef", expect: 114 },                // activation-memory-tp predict
    { args: { s: 2048, h: 4096, a: 32, t: 8 }, pick: "coef", expect: 23 },                 // activation-memory-tp transfer
    { args: { s: 2048, h: 4096, a: 32, t: 8, recompute: true }, pick: "coef", expect: 13 }, // ... with attention recomputed (10 is 77%)
    { args: { s: 2048, h: 4096, a: 32, t: 8, sp: true }, pick: "coef", expect: 14.25, tol: 0.02, from: "lecture_08:sequence-parallel:predict" },
    { args: { s: 2048, h: 4096, a: 32, t: 8, sp: true, recompute: true }, pick: "coef", expect: 4.25 }, // the 34/t lower bound
    { args: { s: 2048, h: 12288, a: 96 }, pick: "recompShare", expect: 0.027, tol: 0.05, from: "lecture_08:selective-recomputation:predict" },
    { args: { s: 2048, h: 12288, a: 96 }, pick: "mlpShare", expect: 0.6486, tol: 0.03, from: "lecture_08:selective-recomputation:check" },
    { args: { s: 32768, h: 12288, a: 96 }, pick: "recompShare", expect: 0.3077 },         // selective-recomputation transfer
    { args: { s: 32768, h: 12288, a: 96 }, pick: "attnShareStored", expect: 1280 / 1314 }, // ... attention is 97% of storage
  ],
};
WIDGETS["fixture:lecture_08--activation-memory"] = (root) => {
  const s = { logS: 10, h: 4096, a: 32, logT: 1, sp: false, recompute: false };
  const pic = el("div"), read = el("div", { class: "readout" });
  const draw = () => {
    const a = { s: 2 ** s.logS, h: s.h, a: s.a, t: 2 ** s.logT, sp: s.sp, recompute: s.recompute }, m = activationMemory(a);
    const W = 640, x0 = 140, span = 420, scale = m.full;
    const bar = (y, label, segs) => {
      let x = x0, out = text(10, y + 15, label, { size: 12 });
      for (const [v, col, op] of segs) { out += rect(x, y, span * v / scale, 22, col, `opacity="${op}"`); x += span * v / scale; }
      return out + text(x + 6, y + 15, `${fmt(segs.reduce((q, z) => q + z[0], 0), 2)} sbh`, { size: 12 });
    };
    let b = text(10, 16, "per-layer activation bytes, in units of sbh · blue: pointwise 10 · green: matmul 24 · orange: attention 5as/h", { fill: C.muted, size: 11 });
    b += bar(26, "1 GPU, store all", [[10, C.a, 1], [24, C.ok, 1], [m.attn, C.b, 1]]);
    b += bar(62, `per GPU, t = ${a.t}`, [[m.pointwise, C.a, 1], [m.matmul, C.ok, 1], [m.attention, C.b, 1]]);
    b += bar(98, "store all ÷ t", [[m.full / a.t, C.rule, 1]]);
    const share = m.pointwise / m.coef;
    const regime = a.t === 1 ? "no tensor parallel: nothing is split" : a.sp ? `TP + SP: every term divides by t (full linear scaling, p49)` : `TP only: the replicated 10 sbh is ${fmt(100 * share, 1)}% of per-GPU memory, so ${fmt(m.full / m.coef, 2)}× saving, not ${a.t}×`;
    b += text(10, 144, regime, { fill: a.t > 1 && !a.sp ? C.b : C.ok, size: 12 });
    pic.innerHTML = svg(W, 154, b);
    read.innerHTML = `s = ${a.s}, h = ${a.h}, a = ${a.a}, t = ${a.t} · 5as/h = ${fmt(m.attn, 2)} (attention is ${fmt(100 * m.attnShareStored, 1)}% of the stored layer)<br>
      per-GPU coefficient = ${fmt(m.pointwise, 3)} + ${fmt(m.matmul, 3)} + ${fmt(m.attention, 3)} = <span class="big">${fmt(m.coef, 3)} sbh</span>
      = ${fmt(m.gbPerSeq, 3)} GB per layer per sequence (b = 1)<br>
      recompute cost, share of the layer's forward FLOPs: attention core s/(6h+s) = <b>${fmt(m.recompShare, 4)}</b> · MLP instead 16h/(24h+4s) = <b>${fmt(m.mlpShare, 4)}</b><br>
      <span class="muted small">provenance: fixture:lecture_08--activation-memory · lecture_08.pdf:p46 (sbh(34 + 5as/h); "we can drop this term via recomputation"), p47 (L446-L447, sbh(10 + 24/t + 5as/(ht)), 10 = LayerNorm 4 + dropout 2 + inputs 4), p48-p49 (L459, SP: full linear scaling); video 46:44-47:08, 48:11-48:27, 51:21-51:53 (34/t lower bound), 52:44-53:15 (recompute attention, not the MLP). Recompute FLOP shares follow Korthikanti et al. 2022 (author).</span>`;
  };
  root.append(el("div", { class: "widget" }, el("div", { class: "controls" },
    slider("sequence s", 9, 15, s.logS, 1, v => { s.logS = v; draw(); }, v => `${2 ** v}`),
    slider("hidden h", 1024, 16384, s.h, 1024, v => { s.h = v; draw(); }),
    slider("heads a", 8, 128, s.a, 8, v => { s.a = v; draw(); }),
    slider("tensor parallel t", 0, 4, s.logT, 1, v => { s.logT = v; draw(); }, v => `${2 ** v}`),
    check("sequence parallel (split the 10 along s)", s.sp, v => { s.sp = v; draw(); }),
    check("recompute the attention core", s.recompute, v => { s.recompute = v; draw(); }),
    read), pic)); draw();
};

// ================================================ 4. FSDP tokens-per-chip threshold ==
// Spoken rule (video 1:04:41-1:05:59): compute must outlast communication; FSDP alone is fine with a big per-chip batch,
// becomes communication-bound as it shrinks, and tensor parallel "push[es] that curve out". The accounting is the author's
// (KP fsdp-batch-per-chip-threshold filtering note, after the scaling book): compute 6·N·T FLOPs per chip, FSDP traffic
// 3 collectives × N/t params × 2 bytes per chip; so comm/compute = C / (t·T·W), threshold T* = C / (t·W), independent of N.
// Default hardware C = 1e15 FLOP/s and W = 4.5e11 B/s are the KP's own example numbers (check prompt).
function fsdpThreshold({ C, W, T, N, t = 1 }) {
  const compute = 6 * N * T / C, comm = 3 * (N / t) * 2 / W;
  return { compute, comm, ratio: comm / compute, threshold: C / (t * W) };
}
MODELS["fixture:lecture_08--fsdp-threshold"] = {
  fn: fsdpThreshold,
  cases: [
    { args: { C: 1e15, W: 4.5e11, T: 1024, N: 7e9 }, pick: "ratio", expect: 2.17, tol: 0.03, from: "lecture_08:fsdp-batch-per-chip-threshold:check" },
    { args: { C: 1e15, W: 4.5e11, T: 1024, N: 14e9 }, pick: "ratio", expect: 2.17, tol: 0.03 },  // predict: doubling N leaves it unchanged
    { args: { C: 1e15, W: 4.5e11, T: 1024, N: 7e9 }, pick: "threshold", expect: 2222.2 },        // C/W ≈ 2,222 tokens per chip
    { args: { C: 1e15, W: 5e10, T: 4096, N: 7e9 }, pick: "threshold", expect: 20000 },           // transfer: inter-node links
    { args: { C: 1e15, W: 5e10, T: 4096, N: 7e9 }, pick: "ratio", expect: 4.8828 },              // ... about 5× communication-bound
    { args: { C: 1e15, W: 5e10, T: 4096, N: 7e9, t: 8 }, pick: "threshold", expect: 2500 },      // ... TP = 8 pulls it to 2,500
    { args: { C: 1e15, W: 5e10, T: 4096, N: 7e9, t: 8 }, pick: "ratio", expect: 0.6104 },        // ... and 4,096 is now compute-bound
  ],
};
WIDGETS["fixture:lecture_08--fsdp-threshold"] = (root) => {
  const s = { NB: 70, logT: 13, logC: 15, logW: Math.log10(4.5e11), logt: 0 };
  const pic = el("div"), read = el("div", { class: "readout" });
  const draw = () => {
    const a = { C: 10 ** s.logC, W: 10 ** s.logW, T: 2 ** s.logT, N: s.NB * 1e9, t: 2 ** s.logt }, m = fsdpThreshold(a);
    // plot comm/compute vs tokens per chip, log-log: x = log2 T in [6, 17], y = log10 ratio in [-2, 2]
    const W = 640, H = 200, px = 50, py = 14, pw = 560, ph = 150;
    const X = lt => px + pw * (lt - 6) / 11, Y = lr => py + ph * (2 - Math.max(-2, Math.min(2, lr))) / 4;
    let b = rect(px, py, pw, Y(0) - py, C.b, 'opacity="0.08"') + rect(px, Y(0), pw, py + ph - Y(0), C.ok, 'opacity="0.08"');
    b += text(px + pw - 4, py + 14, "communication-bound", { fill: C.b, size: 11, anchor: "end" }) + text(px + 6, py + ph - 6, "compute-bound (traffic hidden)", { fill: C.ok, size: 11 });
    b += `<line x1="${px}" y1="${Y(0)}" x2="${px + pw}" y2="${Y(0)}" style="stroke:${C.muted};stroke-dasharray:3 3"/>`;
    let d = "";
    for (let lt = 6; lt <= 17.001; lt += 0.25) d += `${d ? "L" : "M"}${X(lt).toFixed(1)},${Y(Math.log10(a.C / (a.t * 2 ** lt * a.W))).toFixed(1)}`;
    b += `<path d="${d}" style="fill:none;stroke:${C.a};stroke-width:2"/>`;
    const lth = Math.log2(m.threshold);
    if (lth >= 6 && lth <= 17) b += `<line x1="${X(lth).toFixed(1)}" y1="${py}" x2="${X(lth).toFixed(1)}" y2="${py + ph}" style="stroke:${C.a};stroke-dasharray:4 3"/>` + text(X(lth) + 4, py + 28, `T* = ${fmt(m.threshold, 0)}`, { fill: C.a, size: 11 });
    b += `<circle cx="${X(s.logT).toFixed(1)}" cy="${Y(Math.log10(m.ratio)).toFixed(1)}" r="5" style="fill:${C.hi};stroke:${C.ink}"/>`;
    for (const lt of [6, 8, 10, 12, 14, 16]) b += text(X(lt), py + ph + 14, `${2 ** lt}`, { fill: C.muted, size: 10, anchor: "middle" });
    for (const lr of [-2, -1, 0, 1, 2]) b += text(px - 6, Y(lr) + 4, `${10 ** lr}`, { fill: C.muted, size: 10, anchor: "end" });
    b += text(px + pw / 2, H - 2, "tokens per chip per step (log scale) · y: communication time / compute time", { fill: C.muted, size: 11, anchor: "middle" });
    pic.innerHTML = svg(W, H, b);
    const bound = m.ratio > 1;
    read.innerHTML = `per chip per step: compute 6·N·T / C = ${fmt(m.compute, 3)} s · FSDP traffic 3·(N/t)·2 B / W = ${fmt(m.comm, 3)} s<br>
      communication / compute = C / (t·T·W) = <span class="big">${fmt(m.ratio, 3)}</span>
      <b style="color:${bound ? "var(--bad)" : "var(--ok)"}">${bound ? "communication-bound: FSDP alone cannot hide its traffic" : "compute-bound: the all-gathers hide under compute"}</b><br>
      threshold T* = C / (t·W) = ${fmt(m.threshold, 0)} tokens per chip (N does not appear)<br>
      <span class="muted small">provenance: fixture:lecture_08--fsdp-threshold · video 1:04:41-1:04:57 (compute must outlast communication), 1:05:20-1:05:43 (FSDP-only is compute-bound at a big per-chip batch, communication-bound as it shrinks), 1:05:44-1:05:59 (add tensor parallel to push the curve out); lecture_08.pdf:p56 ("Global batch size (divided by GPU)"). Accounting 6·N·T FLOPs and 3·N·2 bytes, and the example C and W, are the KP author's (after the scaling book); ignores (M−1)/M, latency and TP's own in-node traffic.</span>`;
  };
  root.append(el("div", { class: "widget" }, el("div", { class: "controls" },
    slider("parameters N (billions)", 1, 1000, s.NB, 1, v => { s.NB = v; draw(); }),
    slider("tokens per chip T", 6, 17, s.logT, 1, v => { s.logT = v; draw(); }, v => `${2 ** v}`),
    slider("chip compute C (FLOP/s)", 14, 16, s.logC, 0.05, v => { s.logC = v; draw(); }, v => (10 ** v).toExponential(2)),
    slider("FSDP link W (bytes/s per chip)", 10, 12, s.logW, 0.05, v => { s.logW = v; draw(); }, v => (10 ** v).toExponential(2)),
    slider("tensor parallel t (in node)", 0, 3, s.logt, 1, v => { s.logt = v; draw(); }, v => `${2 ** v}`),
    read), pic)); draw();
};
