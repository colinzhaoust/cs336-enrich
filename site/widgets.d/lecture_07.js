// Widgets for thread lecture_07 (CS336 L7, parallelism). Register as "fixture:<id>" -> (root, notice) => void.
// Every constant is from official/lectures/lecture_07.py (line anchors below) or the transcript (video M:SS).
// Each widget's numeric model is a pure function in MODELS, tested by tools/check_widgets.mjs.
import { el, fmt, slider } from "../widgets_lib.js";

export const MODELS = {};
export const WIDGETS = {};

const SVGNS = "http://www.w3.org/2000/svg";
const svg = (tag, attrs = {}, text) => {
  const e = document.createElementNS(SVGNS, tag);
  for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v);
  if (text !== undefined) e.textContent = text;
  return e;
};
const select = (label, options, value, onChange) => {
  const s = el("select", { onchange: e => onChange(e.target.value) }, ...options.map(([v, t]) => { const o = el("option", { value: v }, t); if (v === value) o.selected = true; return o; }));
  return el("label", {}, `${label} `, s);
};

// ===================================================================== collective grid ==
// Rank × slot grid for the lecture's collectives (lecture_07.py:L91-L205). Inputs mirror the lecture's examples:
//   "r+j"  rank r holds [r, r+1, ..., r+W-1]  (all-reduce / reduce-scatter input, L171-L174, L263)
//   "Wr+j" rank r holds [W·r, ..., W·r+W-1]   (all-to-all input, L187-L190)
//   "10r+j" rank r holds [10r, ..., 10r+W-1] (decimal layout: tens digit = source rank)
//   "r*n"  rank r holds n copies of r          (gather/reduce input [r], L117-L135; all-gather [r, r])
//   "custom" rows typed by the learner.
// Semantics: gather/reduce go to rank 0 (L113-L137); all-gather concatenates in rank order (L139-L150);
// reduce-scatter reduces slot block i across ranks and delivers it to rank i (L154-L165); all-reduce = full reduction
// on every rank (L169-L180); rs+ag runs reduce-scatter then all-gather on its output (L253-L282);
// all-to-all sends chunk j of rank i to rank j, stored at position i (L185-L200).
const PRESETS = [["r+j", "r + j (lecture's all-reduce input)"], ["Wr+j", "W·r + j (lecture's all-to-all input)"], ["10r+j", "10·r + j"], ["r*n", "[r, r, …] (n copies of the rank)"], ["custom", "custom rows"]];
const OPS = [["gather", "gather → rank 0"], ["reduce", "reduce → rank 0"], ["all-gather", "all-gather"], ["reduce-scatter", "reduce-scatter"], ["all-reduce", "all-reduce"], ["rs+ag", "reduce-scatter, then all-gather"], ["all-to-all", "all-to-all"]];

export function buildInputs({ W = 4, preset = "r+j", n = 1, rows } = {}) {
  if (preset === "custom") return rows.map(r => r.slice());
  const L = preset === "r*n" ? n : W;
  return Array.from({ length: W }, (_, r) => Array.from({ length: L }, (_, j) =>
    preset === "r+j" ? r + j : preset === "Wr+j" ? W * r + j : preset === "10r+j" ? 10 * r + j : r));
}

// Returns { out: rank -> array|null, src: rank -> [list of [r, j] input cells per output cell], mid?, error? }.
export function runCollective(op, inputs, red = "sum") {
  const W = inputs.length, L = inputs[0].length;
  const f = red === "max" ? Math.max : (a, b) => a + b;
  const reduceCol = j => inputs.reduce((acc, row, r) => r === 0 ? row[j] : f(acc, row[j]), 0);
  const all = (fn) => Array.from({ length: W }, (_, r) => fn(r));
  if (inputs.some(row => row.length !== L)) return { error: "all ranks must hold the same length" };
  if (op === "gather") return { out: all(r => r === 0 ? inputs.flat() : null), src: all(r => r === 0 ? inputs.flatMap((row, q) => row.map((_, j) => [[q, j]])) : null) };
  if (op === "reduce") return { out: all(r => r === 0 ? Array.from({ length: L }, (_, j) => reduceCol(j)) : null), src: all(r => r === 0 ? Array.from({ length: L }, (_, j) => inputs.map((_, q) => [q, j])) : null) };
  if (op === "all-gather") return { out: all(() => inputs.flat()), src: all(() => inputs.flatMap((row, q) => row.map((_, j) => [[q, j]]))) };
  if (op === "all-reduce") return { out: all(() => Array.from({ length: L }, (_, j) => reduceCol(j))), src: all(() => Array.from({ length: L }, (_, j) => inputs.map((_, q) => [q, j]))) };
  if (L % W) return { error: `${op} needs the length (${L}) to be a multiple of world size (${W})` };
  const c = L / W;
  if (op === "reduce-scatter" || op === "rs+ag") {
    const rs = all(r => Array.from({ length: c }, (_, k) => reduceCol(r * c + k)));
    const rsSrc = all(r => Array.from({ length: c }, (_, k) => inputs.map((_, q) => [q, r * c + k])));
    if (op === "reduce-scatter") return { out: rs, src: rsSrc };
    return { mid: rs, out: all(() => rs.flat()), src: all(() => rsSrc.flat()) };
  }
  if (op === "all-to-all") return { out: all(r => inputs.flatMap(row => row.slice(r * c, r * c + c))), src: all(r => inputs.flatMap((_, q) => Array.from({ length: c }, (_, k) => [[q, r * c + k]]))) };
  return { error: `unknown op ${op}` };
}

const collectiveModel = (a) => {
  const res = runCollective(a.op, buildInputs(a), a.red || "sum");
  if (res.error) return { error: res.error };
  const row = res.out[a.rank ?? 0];
  return { value: row ? row[a.index ?? 0] : NaN, length: row ? row.length : 0 };
};
const LECTURE_RS_ROWS = [[1, 2, 3], [10, 20, 30], [100, 200, 300]];
MODELS["fixture:lecture_07--collective-grid"] = {
  fn: collectiveModel,
  cases: [
    { args: { op: "reduce-scatter", W: 4, preset: "r+j", rank: 3, index: 0 }, pick: "value", expect: 18, tol: 0.001, from: "lecture_07:reduce-scatter:predict" },
    { args: { op: "reduce-scatter", preset: "custom", rows: LECTURE_RS_ROWS, rank: 1, index: 0 }, pick: "value", expect: 222, tol: 0.001, from: "lecture_07:reduce-scatter:transfer" },
    { args: { op: "rs+ag", preset: "custom", rows: [[1, 0, 4], [0, 3, 1], [2, 1, 0]], rank: 0, index: 2 }, pick: "value", expect: 5, tol: 0.001, from: "lecture_07:all-reduce-decomposition:check" },
    { args: { op: "all-to-all", W: 5, preset: "10r+j", rank: 1, index: 3 }, pick: "value", expect: 31, tol: 0.001, from: "lecture_07:all-to-all:check" },
    { args: { op: "reduce", W: 6, preset: "r*n", n: 3, rank: 0 }, pick: "length", expect: 3, tol: 0.001, from: "lecture_07:collective-taxonomy:check" },
    // edges: all-gather multiplies the length by W; rs+ag equals all-reduce; reduce MAX keeps the shape
    { args: { op: "all-gather", W: 6, preset: "r*n", n: 3, rank: 4 }, pick: "length", expect: 18, tol: 0.001 },
    { args: { op: "all-reduce", W: 4, preset: "r+j", rank: 2, index: 3 }, pick: "value", expect: 18, tol: 0.001 },
    { args: { op: "rs+ag", W: 4, preset: "r+j", rank: 2, index: 3 }, pick: "value", expect: 18, tol: 0.001 },
    { args: { op: "reduce", red: "max", W: 8, preset: "r*n", n: 1, rank: 0, index: 0 }, pick: "value", expect: 7, tol: 0.001 },
    { args: { op: "all-to-all", W: 4, preset: "Wr+j", rank: 1, index: 2 }, pick: "value", expect: 9, tol: 0.001 },
  ],
};

function drawGrid(title, rows, { hiRow = -1, hiCells = new Set(), pickCell = null, onPick = null, x0 = 0 } = {}) {
  const g = svg("g", { transform: `translate(${x0},0)` });
  g.append(svg("text", { x: 0, y: 14, "font-size": 12, fill: "var(--muted)" }, title));
  const cw = 34, ch = 24;
  rows.forEach((row, r) => {
    const y = 22 + r * (ch + 4);
    g.append(svg("text", { x: 0, y: y + 16, "font-size": 11, fill: r === hiRow ? "var(--ink)" : "var(--muted)", "font-weight": r === hiRow ? 700 : 400 }, `rank ${r}`));
    if (!row) { g.append(svg("text", { x: 52, y: y + 16, "font-size": 11, fill: "var(--muted)" }, "— (no output)")); return; }
    row.forEach((v, j) => {
      const key = `${r},${j}`, picked = pickCell && pickCell[0] === r && pickCell[1] === j;
      const rect = svg("rect", { x: 48 + j * cw, y, width: cw - 2, height: ch, rx: 3,
        fill: picked ? "var(--hilite)" : hiCells.has(key) ? "#f3dccf" : r === hiRow ? "#dce8f0" : "#fff",
        stroke: picked || hiCells.has(key) ? "var(--accent2)" : "var(--rule)", "stroke-width": picked ? 2 : 1, style: onPick ? "cursor:pointer" : "" });
      if (onPick) rect.addEventListener("click", () => onPick(r, j));
      g.append(rect, svg("text", { x: 48 + j * cw + (cw - 2) / 2, y: y + 16, "font-size": 12, "text-anchor": "middle", fill: "var(--ink)", "pointer-events": "none" }, fmt(v, 2)));
    });
  });
  return g;
}

export function collectiveGrid(root, notice) {
  // Default is deliberately none of the prompts' cases (they use W = 4, 5, 6, 8 or custom rows).
  const s = { op: "reduce-scatter", red: "sum", W: 3, preset: "r+j", n: 2, rank: 0, index: 0, custom: "1 0 4\n0 3 1\n2 1 0" };
  const pic = svg("svg", { width: "100%", viewBox: "0 0 640 300", style: "background:#fff;border:1px solid var(--rule);border-radius:6px" });
  const read = el("div", { class: "readout" });
  const parseCustom = () => s.custom.trim().split(/\n|\|/).map(line => line.trim().split(/[\s,]+/).map(Number)).filter(r => r.length && r.every(Number.isFinite));
  const draw = () => {
    const inputs = s.preset === "custom" ? parseCustom() : buildInputs(s);
    while (pic.firstChild) pic.firstChild.remove();
    if (!inputs.length) { read.textContent = "type one row of numbers per rank"; return; }
    const W = inputs.length;
    s.rank = Math.min(s.rank, W - 1);
    const res = runCollective(s.op, inputs, s.red);
    if (res.error) { pic.append(drawGrid("input (rank × slot)", inputs)); read.innerHTML = `<b>${res.error}</b>`; return; }
    const out = res.out[s.rank];
    if (out) s.index = Math.min(s.index, out.length - 1);
    const srcCells = out ? new Set(res.src[s.rank][s.index].map(([q, j]) => `${q},${j}`)) : new Set();
    const pick = (r, j) => { if (res.out[r]) { s.rank = r; s.index = j; draw(); } };
    pic.append(drawGrid("input (rank × slot)", inputs, { hiCells: srcCells }));
    let x = 48 + 34 * inputs[0].length + 30;
    if (res.mid) { pic.append(drawGrid("after reduce-scatter", res.mid, { x0: x })); x += 48 + 34 * res.mid[0].length + 30; }
    const outRows = res.out;
    pic.append(drawGrid(res.mid ? "after the all-gather" : `after ${s.op}`, outRows, { hiRow: s.rank, pickCell: out ? [s.rank, s.index] : null, onPick: pick, x0: x }));
    const h = 22 + W * 28 + 10; pic.setAttribute("viewBox", `0 0 ${Math.max(640, x + 48 + 34 * Math.max(...outRows.map(r => r ? r.length : 4)) + 10)} ${Math.max(140, h)}`);
    const opName = s.red === "max" ? "max" : "+";
    const contrib = out ? res.src[s.rank][s.index].map(([q, j]) => inputs[q][j]) : [];
    const expr = contrib.length > 1 ? `${contrib.map(v => fmt(v, 2)).join(` ${opName} `)} = ` : contrib.length === 1 ? `copied from rank ${res.src[s.rank][s.index][0][0]}, slot ${res.src[s.rank][s.index][0][1]}: ` : "";
    const same = s.op === "rs+ag" ? (JSON.stringify(res.out) === JSON.stringify(runCollective("all-reduce", inputs, s.red).out) ? "identical to all-reduce on the same inputs" : "differs from all-reduce") : "";
    read.innerHTML = out
      ? `rank ${s.rank} holds [${out.map(v => fmt(v, 2)).join(", ")}] · length <b>${out.length}</b> (input length ${inputs[0].length}, W = ${W})<br>
         slot ${s.index}: ${expr}<span class="big">${fmt(out[s.index], 2)}</span>
         ${contrib.length > 1 ? "reduced: one value per contributing rank, length unchanged by the op" : "moved, nothing summed"}${same ? `<br><b>${same}</b>` : ""}<br>
         <span class="muted small">click any output cell; the input cells that produced it are outlined</span><br>
         <span class="muted small">provenance: fixture:lecture_07--collective-grid · lecture_07.py:L91-L205 (definitions and the lecture's inputs), L253-L282 (reduce-scatter then all-gather reproduces all-reduce)</span>`
      : `rank ${s.rank} receives nothing from ${s.op}; pick rank 0`;
  };
  const customBox = el("textarea", { rows: 3, style: "width:100%;font-family:var(--mono);font-size:13px", oninput: e => { s.custom = e.target.value; draw(); } });
  customBox.value = s.custom;
  const ctl = el("div", { class: "controls" },
    select("collective", OPS, s.op, v => { s.op = v; draw(); }),
    select("reduce op", [["sum", "SUM"], ["max", "MAX"]], s.red, v => { s.red = v; draw(); }),
    select("inputs", PRESETS, s.preset, v => { s.preset = v; draw(); }),
    slider("world size W (presets)", 2, 8, s.W, 1, v => { s.W = v; draw(); }),
    slider("n (for [r, r, …])", 1, 4, s.n, 1, v => { s.n = v; draw(); }),
    el("label", {}, "custom rows (one rank per line)", customBox),
    slider("read rank", 0, 7, s.rank, 1, v => { s.rank = v; draw(); }),
    slider("read slot", 0, 31, s.index, 1, v => { s.index = v; draw(); }),
    read);
  root.append(el("div", { class: "widget" }, ctl, pic)); draw();
}
WIDGETS["fixture:lecture_07--collective-grid"] = collectiveGrid;

// ======================================================================= link ladder ==
// Bandwidth tiers named in the lecture, in bytes/s. time = bytes / bandwidth (no latency term; the lecture gives none).
const TIERS = [
  { id: "hbm", name: "HBM (B200)", bw: 8e12, where: "on one GPU", src: "L219 'HBM was 8 TB/s'; spoken 24:03" },
  { id: "nvlink", name: "NVLink 5.0 / NVSwitch", bw: 1.8e12, where: "8 GPUs in a node", src: "L219; spoken 23:40-24:13 ('about 4x slower')", tag: "tensor parallelism stays here (L69)" },
  { id: "ib", name: "Infiniband", bw: 0.05e12, where: "256 nodes in a pod", src: "L220 '~0.05 TB/s'", tag: "pipeline parallelism can live here (L70)" },
  { id: "pcie", name: "PCIe v7.0 ×16 (classic, home)", bw: 242e9, where: "GPUs in one home box", src: "L212 '242 GB/s'", home: true },
  { id: "eth", name: "Ethernet (classic, home)", bw: 200e6, where: "between home machines", src: "L213 '~200 MB/s'", home: true },
];
// Data-center Ethernet between pods (L221) has no number in the lecture, so it has no bar.
const ladderModel = ({ gb }) => {
  const out = {};
  for (const t of TIERS) out[`${t.id}_ms`] = gb * 1e9 / t.bw * 1e3;
  out.nvlink_over_hbm = TIERS[0].bw / TIERS[1].bw;
  out.ib_over_nvlink = TIERS[1].bw / TIERS[2].bw;
  return out;
};
MODELS["fixture:lecture_07--link-ladder"] = {
  fn: ladderModel,
  cases: [
    { args: { gb: 7.2 }, pick: "nvlink_ms", expect: 4, tol: 0.01, from: "lecture_07:interconnect-hierarchy:check" },   // 0.9 ms on HBM = 7.2 GB
    { args: { gb: 7.2 }, pick: "hbm_ms", expect: 0.9, tol: 0.001 },
    { args: { gb: 16 }, pick: "nvlink_ms", expect: 8.889, tol: 0.01, from: "lecture_07:interconnect-hierarchy:transfer" },
    { args: { gb: 16 }, pick: "ib_ms", expect: 320, tol: 0.001 },
    { args: { gb: 1 }, pick: "nvlink_over_hbm", expect: 4.444, tol: 0.01 },
    { args: { gb: 1 }, pick: "ib_over_nvlink", expect: 36, tol: 0.001 },
  ],
};

export function linkLadder(root, notice) {
  const s = { gb: 1 };   // default 1 GB: not the check's 0.9 ms (7.2 GB) or the transfer's 16 GB
  const pic = svg("svg", { width: "100%", viewBox: "0 0 640 270", style: "background:#fff;border:1px solid var(--rule);border-radius:6px" });
  const read = el("div", { class: "readout" });
  const gbIn = el("input", { type: "number", min: 0.001, step: "any", value: s.gb });
  const hbmIn = el("input", { type: "number", min: 0.0001, step: "any" });
  const draw = (from) => {
    const m = ladderModel(s);
    if (from !== "gb") gbIn.value = +s.gb.toPrecision(4);
    if (from !== "hbm") hbmIn.value = +m.hbm_ms.toPrecision(4);
    while (pic.firstChild) pic.firstChild.remove();
    const lo = Math.log10(1e8), hi = Math.log10(1e13), X0 = 200, XW = 300;
    const xOf = bw => X0 + XW * (Math.log10(bw) - lo) / (hi - lo);
    pic.append(svg("text", { x: 10, y: 16, "font-size": 12, fill: "var(--muted)" }, "bandwidth (log scale) · time to move the payload once"));
    TIERS.forEach((t, i) => {
      const y = 30 + i * 44 + (t.home ? 12 : 0);
      pic.append(svg("text", { x: 10, y: y + 14, "font-size": 12, fill: t.home ? "var(--muted)" : "var(--ink)" }, t.name));
      pic.append(svg("text", { x: 10, y: y + 28, "font-size": 10, fill: "var(--muted)" }, t.where));
      pic.append(svg("rect", { x: X0, y: y + 2, width: Math.max(2, xOf(t.bw) - X0), height: 16, rx: 2, fill: t.home ? "#cfc9bd" : t.id === "nvlink" ? "var(--accent2)" : "var(--accent)" }));
      pic.append(svg("text", { x: xOf(t.bw) + 6, y: y + 15, "font-size": 11, fill: "var(--ink)" }, `${fmt(t.bw / 1e12, 4)} TB/s`));
      pic.append(svg("text", { x: 630, y: y + 15, "font-size": 12, "text-anchor": "end", "font-weight": 700, fill: "var(--ink)" }, `${fmt(m[`${t.id}_ms`], 3)} ms`));
      if (t.tag) pic.append(svg("text", { x: X0, y: y + 32, "font-size": 10, fill: "var(--accent2)" }, t.tag));
    });
    pic.append(svg("line", { x1: 10, x2: 630, y1: 30 + 3 * 44 + 2, y2: 30 + 3 * 44 + 2, stroke: "var(--rule)", "stroke-dasharray": "4 3" }));
    read.innerHTML = `payload ${fmt(s.gb, 4)} GB<br>
      HBM <b>${fmt(m.hbm_ms, 3)} ms</b> → NVLink <b>${fmt(m.nvlink_ms, 3)} ms</b> (×${fmt(m.nvlink_over_hbm, 2)}) → Infiniband <span class="big">${fmt(m.ib_ms, 3)} ms</span> (another ×${fmt(m.ib_over_nvlink, 1)})<br>
      the ratios between tiers do not depend on the payload: same bytes, each hop down costs a fixed factor<br>
      <span class="muted small">pod-to-pod Ethernet (L221) has no number in the lecture, so it has no bar. Time = bytes / bandwidth; no latency term.</span><br>
      <span class="muted small">provenance: fixture:lecture_07--link-ladder · ${TIERS.map(t => `${t.name}: ${t.src}`).join(" · ")} · tags from the summary L69-L70</span>`;
  };
  gbIn.addEventListener("input", () => { const v = +gbIn.value; if (v > 0) { s.gb = v; draw("gb"); } });
  hbmIn.addEventListener("input", () => { const v = +hbmIn.value; if (v > 0) { s.gb = v * 1e-3 * TIERS[0].bw / 1e9; draw("hbm"); } });
  root.append(el("div", { class: "widget" }, el("div", { class: "controls" },
    el("label", {}, "payload (GB, 1 GB = 10^9 B)", gbIn),
    el("label", {}, "or: time to read it from HBM (ms)", hbmIn),
    read), pic)); draw();
}
WIDGETS["fixture:lecture_07--link-ladder"] = linkLadder;

// ===================================================================== TP vs PP bytes ==
// The lecture's toy MLP (lecture_07.py:L390-L392 batch 128, num_dim 1024, fp32; L444 TP world_size 4, 4 layers;
// L489 PP world_size 2, 4 layers, 4 micro-batches).
// Tensor parallelism (L447-L479): rank r holds num_dim × num_dim/W per layer, every layer ends with an all-gather of the
//   batch × num_dim/W slices, so each rank receives (W−1)/W of a batch × num_dim activation per layer.
//   Backward (homework, L479; spoken 1:08:12-1:08:33): the reduce-scatter leaves each rank one batch × num_dim/W slice.
// Pipeline parallelism (L492-L536): stage r holds num_layers/W consecutive layers; per micro-batch (batch/m rows) one
//   dist.send/recv of micro × num_dim at each stage boundary, so stage s receives m · (batch/m) · num_dim = batch × num_dim.
// 1 KiB = 1024 B, as in the KPs.
export const tpPpModel = ({ batch = 128, dim = 1024, bytes = 4, layers = 4, tp = 4, pp = 2, micro = 4 }) => {
  const slice = batch * (dim / tp) * bytes;
  const tpRecvLayer = (tp - 1) * slice;
  return {
    tp_shard_cols: dim / tp,
    tp_recv_layer_kib: tpRecvLayer / 1024,
    tp_recv_fwd_kib: layers * tpRecvLayer / 1024,
    tp_rs_keep_bytes: slice,
    pp_layers_per_stage: layers / pp,
    pp_micro_rows: batch / micro,
    pp_sends_per_boundary: micro,
    pp_recv_kib: micro * (batch / micro) * dim * bytes / 1024,
    valid: Number.isInteger(dim / tp) && Number.isInteger(layers / pp) && Number.isInteger(batch / micro) ? 1 : 0,
  };
};
MODELS["fixture:lecture_07--tp-pp-traffic"] = {
  fn: tpPpModel,
  cases: [
    { args: { batch: 64, dim: 4096, bytes: 2, layers: 4, tp: 8 }, pick: "tp_recv_fwd_kib", expect: 1792, tol: 0.001, from: "lecture_07:tensor-parallelism:transfer" },
    { args: { batch: 256, dim: 2048, bytes: 2, layers: 8, pp: 4, micro: 8 }, pick: "pp_recv_kib", expect: 1024, tol: 0.001, from: "lecture_07:pipeline-sharding:transfer" },
    { args: { batch: 128, dim: 1024, bytes: 4, tp: 4 }, pick: "tp_rs_keep_bytes", expect: 131072, tol: 0.001, from: "lecture_07:tp-backward-reduce-scatter:check" },
    // the lecture's own settings (predict prompts are short-answer, so no `from`): 384 KiB per layer, shard 1024 × 256; 32-row micro-batches, 4 sends
    { args: { batch: 128, dim: 1024, bytes: 4, tp: 4 }, pick: "tp_recv_layer_kib", expect: 384, tol: 0.001 },
    { args: { batch: 128, dim: 1024, bytes: 4, tp: 4 }, pick: "tp_shard_cols", expect: 256, tol: 0.001 },
    { args: { batch: 128, micro: 4, pp: 2, layers: 4 }, pick: "pp_micro_rows", expect: 32, tol: 0.001 },
    { args: { batch: 128, micro: 4, pp: 2, layers: 4 }, pick: "pp_sends_per_boundary", expect: 4, tol: 0.001 },
    // edges: PP bytes do not depend on m; TP bytes scale with the number of layers
    { args: { batch: 256, dim: 2048, bytes: 2, layers: 8, pp: 4, micro: 1 }, pick: "pp_recv_kib", expect: 1024, tol: 0.001 },
    { args: { batch: 64, dim: 4096, bytes: 2, layers: 8, tp: 8 }, pick: "tp_recv_fwd_kib", expect: 3584, tol: 0.001 },
  ],
};

export function tpPpTraffic(root, notice) {
  // Defaults differ from both predicts (TP W = 4; PP 2 stages with 4 micro-batches).
  const s = { batch: 128, dim: 1024, bytes: 4, layers: 4, tp: 2, pp: 4, micro: 8 };
  const pic = svg("svg", { width: "100%", viewBox: "0 0 640 220", style: "background:#fff;border:1px solid var(--rule);border-radius:6px" });
  const read = el("div", { class: "readout" });
  const kib = x => `${fmt(x, 1)} KiB`;
  const draw = () => {
    const m = tpPpModel(s);
    while (pic.firstChild) pic.firstChild.remove();
    // left: TP column blocks per layer; right: PP stages stacked by depth
    pic.append(svg("text", { x: 10, y: 16, "font-size": 12, fill: "var(--muted)" }, `tensor parallel: W = ${s.tp} column blocks per layer`));
    const L = Math.min(s.layers, 8), bw = Math.min(40, 240 / s.tp);
    for (let l = 0; l < L; l++) {
      for (let r = 0; r < s.tp; r++) pic.append(svg("rect", { x: 10 + r * bw, y: 26 + l * 22, width: bw - 2, height: 16, fill: r === 0 ? "var(--accent)" : "#dce8f0", stroke: "var(--rule)" }));
      pic.append(svg("text", { x: 14 + s.tp * bw, y: 38 + l * 22, "font-size": 10, fill: "var(--accent2)" }, "all-gather"));
    }
    if (s.layers > 8) pic.append(svg("text", { x: 10, y: 26 + 8 * 22 + 12, "font-size": 10, fill: "var(--muted)" }, `… ${s.layers} layers`));
    pic.append(svg("text", { x: 340, y: 16, "font-size": 12, fill: "var(--muted)" }, `pipeline: ${s.pp} stages × ${fmt(m.pp_layers_per_stage, 2)} layers`));
    const sh = Math.min(40, 180 / s.pp);
    for (let st = 0; st < s.pp; st++) {
      pic.append(svg("rect", { x: 340, y: 26 + st * sh, width: 120, height: sh - 4, fill: st === 1 ? "var(--accent)" : "#dce8f0", stroke: "var(--rule)" }));
      if (st > 0) pic.append(svg("text", { x: 468, y: 26 + st * sh + 4, "font-size": 10, fill: "var(--accent2)" }, `send/recv × ${s.micro}`));
    }
    const max = Math.max(m.tp_recv_fwd_kib, m.pp_recv_kib, 1);
    pic.append(svg("rect", { x: 10, y: 200, width: Math.max(2, 300 * m.tp_recv_fwd_kib / max), height: 8, fill: "var(--accent)" }));
    pic.append(svg("rect", { x: 340, y: 200, width: Math.max(2, 290 * m.pp_recv_kib / max), height: 8, fill: "var(--accent)" }));
    pic.append(svg("text", { x: 10, y: 196, "font-size": 10, fill: "var(--ink)" }, `rank 0 receives ${kib(m.tp_recv_fwd_kib)} per forward`));
    pic.append(svg("text", { x: 340, y: 196, "font-size": 10, fill: "var(--ink)" }, `stage 1 receives ${kib(m.pp_recv_kib)} per forward`));
    const warn = m.valid ? "" : `<br><b>not a valid split: num_dim/W, layers/stages and batch/m must be integers (int_divide, L453, L501, L509)</b>`;
    read.innerHTML = `<b>tensor parallel</b> (W = ${s.tp}): weight shard per layer ${s.dim} × ${fmt(m.tp_shard_cols, 2)}; per layer each rank receives (W−1)/W of ${s.batch} × ${s.dim} = <b>${kib(m.tp_recv_layer_kib)}</b>; over ${s.layers} layers <span class="big">${kib(m.tp_recv_fwd_kib)}</span>
      backward reduce-scatter leaves each rank ${s.batch} × ${fmt(m.tp_shard_cols, 2)} = ${m.tp_rs_keep_bytes.toLocaleString("en-US")} B<br>
      <b>pipeline</b> (${s.pp} stages, m = ${s.micro}): micro-batch ${fmt(m.pp_micro_rows, 2)} × ${s.dim}, ${m.pp_sends_per_boundary} sends per boundary; each stage receives <span class="big">${kib(m.pp_recv_kib)}</span>
      TP traffic is paid at every layer; PP traffic only at stage boundaries, and changing m does not change it${warn}<br>
      <span class="muted small">provenance: fixture:lecture_07--tp-pp-traffic · lecture_07.py:L390-L392 (batch 128, num_dim 1024, fp32), L447-L479 (TP), L492-L536 (PP), spoken 1:08:12-1:08:33 (backward reduce-scatter) · 1 KiB = 1024 B</span>`;
  };
  root.append(el("div", { class: "widget" }, el("div", { class: "controls" },
    slider("batch", 32, 512, s.batch, 32, v => { s.batch = v; draw(); }),
    slider("num_dim", 512, 8192, s.dim, 512, v => { s.dim = v; draw(); }),
    slider("bytes / element", 2, 4, s.bytes, 2, v => { s.bytes = v; draw(); }, v => v === 2 ? "bf16 (2)" : "fp32 (4)"),
    slider("layers", 2, 16, s.layers, 1, v => { s.layers = v; draw(); }),
    slider("TP world size W", 2, 16, s.tp, 1, v => { s.tp = v; draw(); }),
    slider("PP stages", 2, 8, s.pp, 1, v => { s.pp = v; draw(); }),
    slider("micro-batches m", 1, 16, s.micro, 1, v => { s.micro = v; draw(); }),
    read), pic)); draw();
}
WIDGETS["fixture:lecture_07--tp-pp-traffic"] = tpPpTraffic;

// ===================================================================== axes composer ==
// Composition of the axes (lecture_07.py:L67-L71 summary; Megatron-style W = W_d·W_t·W_p as in KP parallelism-axes):
// parameters are split by W_t·W_p and replicated W_d times; TP wants the NVLink domain: 8 GPUs per node (L219), or 72 with
// GB200/GB300 NVL72 (L229); spoken 1:17:14-1:17:56: TP inside a node, then DP/FSDP, then PP.
export const axesModel = ({ wt = 8, wp = 4, wd = 32, params_b = 70, bytes = 2, domain = 8 }) => ({
  gpus: wt * wp * wd,
  frac: 1 / (wt * wp),
  gb_per_gpu: params_b * 1e9 / (wt * wp) * bytes / 1e9,
  replicas: wd,
  tp_crosses_domain: wt > domain ? 1 : 0,
});
MODELS["fixture:lecture_07--axes-composer"] = {
  fn: axesModel,
  cases: [
    { args: { wt: 8, wp: 8, wd: 16, params_b: 160, bytes: 2 }, pick: "gb_per_gpu", expect: 5, tol: 0.001, from: "lecture_07:parallelism-axes:transfer" },
    { args: { wt: 8, wp: 4, wd: 32 }, pick: "frac", expect: 1 / 32, tol: 0.001, from: "lecture_07:parallelism-axes:predict" },
    { args: { wt: 8, wp: 4, wd: 32 }, pick: "gpus", expect: 1024, tol: 0.001 },
    // regime edge: TP of 16 leaves an 8-GPU NVLink node, but fits an NVL72 domain
    { args: { wt: 16, wp: 4, wd: 16 }, pick: "tp_crosses_domain", expect: 1, tol: 0.001 },
    { args: { wt: 16, wp: 4, wd: 16, domain: 72 }, pick: "tp_crosses_domain", expect: 0, tol: 0.001 },
  ],
};

export function axesComposer(root, notice) {
  const s = { lt: 2, lp: 1, ld: 3, params_b: 70, bytes: 2, domain: 8 };   // W_t 4, W_p 2, W_d 8: not a prompt's case
  const pic = svg("svg", { width: "100%", viewBox: "0 0 640 240", style: "background:#fff;border:1px solid var(--rule);border-radius:6px" });
  const read = el("div", { class: "readout" });
  const draw = () => {
    const wt = 2 ** s.lt, wp = 2 ** s.lp, wd = 2 ** s.ld, m = axesModel({ wt, wp, wd, params_b: s.params_b, bytes: s.bytes, domain: s.domain });
    while (pic.firstChild) pic.firstChild.remove();
    pic.append(svg("text", { x: 10, y: 16, "font-size": 12, fill: "var(--muted)" }, `one model replica = W_t × W_p = ${wt} × ${wp} GPUs; ${wd} replicas`));
    const cols = Math.min(wt, 32), rows = Math.min(wp, 8), cw = Math.min(18, 560 / cols), rh = Math.min(22, 180 / rows);
    for (let p = 0; p < rows; p++) {
      for (let t = 0; t < cols; t++) pic.append(svg("rect", { x: 40 + t * cw, y: 30 + p * rh, width: cw - 2, height: rh - 4, fill: "#dce8f0", stroke: "var(--rule)" }));
      pic.append(svg("text", { x: 10, y: 30 + p * rh + rh / 2 + 2, "font-size": 10, fill: "var(--muted)" }, `PP ${p}`));
      // NVLink domain outlines: groups of `domain` GPUs along the TP axis
      for (let d0 = 0; d0 < cols; d0 += s.domain) pic.append(svg("rect", { x: 39 + d0 * cw, y: 29 + p * rh, width: Math.min(s.domain, cols - d0) * cw, height: rh - 2, fill: "none", stroke: "var(--accent2)", "stroke-width": 1.5 }));
    }
    if (wt > cols || wp > rows) pic.append(svg("text", { x: 40, y: 30 + rows * rh + 12, "font-size": 10, fill: "var(--muted)" }, `(drawing truncated to ${cols} × ${rows})`));
    pic.append(svg("text", { x: 40, y: 228, "font-size": 11, fill: "var(--accent2)" }, `orange outline = one NVLink domain (${s.domain} GPUs)`));
    const link = m.tp_crosses_domain ? `<b>TP group of ${wt} is larger than the ${s.domain}-GPU NVLink domain: its per-layer all-gathers cross Infiniband (~36× slower, L219-L220)</b>` : `TP group of ${wt} fits inside one ${s.domain}-GPU NVLink domain; PP stages and DP replicas cross nodes`;
    read.innerHTML = `W = W_d · W_t · W_p = ${wd} · ${wt} · ${wp} = <b>${m.gpus}</b> GPUs<br>
      each GPU holds 1/(W_t·W_p) = 1/${wt * wp} of the parameters = <span class="big">${fmt(m.gb_per_gpu, 3)} GB</span> (${s.params_b}B × ${s.bytes} B, before any ZeRO), replicated ${wd}×<br>
      ${link}<br>
      <span class="muted small">provenance: fixture:lecture_07--axes-composer · lecture_07.py:L67-L71 (axes, which link each needs), L219 (8 GPUs per node), L229 (NVL72: 72 GPUs in one NVLink domain), spoken 1:17:14-1:17:56</span>`;
  };
  root.append(el("div", { class: "widget" }, el("div", { class: "controls" },
    slider("tensor parallel W_t", 0, 7, s.lt, 1, v => { s.lt = v; draw(); }, v => 2 ** v),
    slider("pipeline W_p", 0, 6, s.lp, 1, v => { s.lp = v; draw(); }, v => 2 ** v),
    slider("data parallel W_d", 0, 10, s.ld, 1, v => { s.ld = v; draw(); }, v => 2 ** v),
    slider("parameters (B)", 1, 1000, s.params_b, 1, v => { s.params_b = v; draw(); }),
    slider("bytes / parameter", 2, 4, s.bytes, 2, v => { s.bytes = v; draw(); }, v => v === 2 ? "bf16 (2)" : "fp32 (4)"),
    select("NVLink domain", [["8", "8 GPUs (typical node, L219)"], ["72", "72 GPUs (NVL72, L229)"]], String(s.domain), v => { s.domain = +v; draw(); }),
    read), pic)); draw();
}
WIDGETS["fixture:lecture_07--axes-composer"] = axesComposer;
