// Widgets for thread lecture_05 (CS336 L5, GPUs). Register as "fixture:<id>" -> (root, notice) => void.
// Each widget's numeric model is a pure function in MODELS, tested by tools/check_widgets.mjs against
// the KP prompts that use it. Every constant carries its source (file:line, slide page or video time).
import { el, fmt, slider } from "../widgets_lib.js";
export const MODELS = {};
export const WIDGETS = {};

// palette = site/styles.css :root tokens
const K = { ink: "#25313d", muted: "#6b7680", rule: "#d9d3c7", accent: "#24668d", accent2: "#b8582a", hilite: "#e3b23c", ok: "#3c8d5a", track: "#eeeae0" };
const canvas = (w, h) => el("canvas", { width: w, height: h });
function select(label, opts, value, on) {
  const s = el("select", { onchange: e => on(e.target.value) }, ...opts.map(([v, t]) => el("option", { value: v }, t)));
  s.value = value;
  return el("label", {}, `${label} `, s);
}
function setRange(lab, v, fmtv = x => x) { const i = lab.querySelector("input"), o = lab.querySelector(".readout"); i.value = v; o.textContent = fmtv(+i.value); }
const presets = (label, items) => el("div", { style: "margin-top:8px" }, el("span", { class: "small muted" }, `${label} `), ...items.flatMap(([t, f]) => [el("button", { class: "ghost small", onclick: f }, t), " "]));
const src = s => `<br><span class="muted small">${s}</span>`;
const regime = (txt, good) => `<span style="color:${good ? K.ok : K.accent2};font-weight:700">${txt}</span>`;

// =============================================================== roofline + memory wall ==
// H100 numbers as lecture 2 uses them: lecture_02.py:L74/L350 peak = 1979e12 / 2 FLOP/s (bf16, no sparsity),
// L351 bandwidth = 3.35e12 B/s, so the kink (accelerator intensity) is ~295 FLOPs/byte (L385).
// The FLOP/s and bandwidth multipliers are the learner's what-if controls (slide p18 / video 25:52-26:43:
// compute grows fast, memory bandwidth slowly); the widget invents no generation numbers.
const H100_PEAK = 1979e12 / 2, H100_BW = 3.35e12;
const N_VEC = 1024 * 1024, N_MAT = 1024; // lecture_02.py:L364, L435, L450
const ROOF_OPS = {
  relu: { name: "ReLU", n: N_VEC, flops: n => n, bytes: (n, b) => 2 * b * n },                          // lecture_02.py:L368-L369; slide p25
  gelu: { name: "GeLU", n: N_VEC, flops: n => 20 * n, bytes: (n, b) => 2 * b * n },                     // L404-L405
  dot: { name: "dot product", n: N_VEC, flops: n => 2 * n - 1, bytes: (n, b) => 2 * b * n + b },         // L422-L423
  matvec: { name: "matvec 1024", n: N_MAT, flops: n => n * (2 * n - 1), bytes: (n, b) => b * (n + n * n + n) }, // L437-L438
  matmul: { name: "matmul 1024", n: N_MAT, flops: n => n * n * (2 * n - 1), bytes: (n, b) => 3 * b * n * n },  // L453-L454
};
function roofModel({ op = "matvec", bytes = 2, flops_x = 1, bw_x = 1, t_fp32_us } = {}) {
  const P = H100_PEAK * flops_x, B = H100_BW * bw_x, kink = P / B;
  const o = ROOF_OPS[op], F = o.flops(o.n), Y = o.bytes(o.n, bytes), ai = F / Y;
  const time = by => Math.max(o.bytes(o.n, by) / B, F / P); // perfect overlap, lecture_02.py:L374-L375
  const ratio = time(bytes) / time(4);
  return {
    kink, kink_ratio: kink / (H100_PEAK / H100_BW), ai, frac: Math.min(P, ai * B) / P,
    bound: ai < kink ? "memory" : "compute", bytes_per_flop: Y / F, t_us: time(bytes) * 1e6,
    time_vs_fp32: ratio, t_scaled_us: (t_fp32_us ?? NaN) * ratio,
  };
}
MODELS["fixture:lecture_05--roofline-wall"] = {
  fn: roofModel,
  cases: [
    { args: { op: "relu", bytes: 2 }, pick: "frac", expect: 0.00085, tol: 0.01, from: "lecture_05:memory-wall-roofline:transfer" },
    { args: { op: "relu", bytes: 2 }, pick: "kink", expect: 295.4, tol: 0.001 },
    { args: { flops_x: 10, bw_x: 3 }, pick: "kink_ratio", expect: 10 / 3, tol: 1e-6 },          // predict: kink moves right 3.3x
    { args: { op: "matmul", bytes: 2 }, pick: "frac", expect: 1 },                                 // 1024 matmul: compute-bound on H100
    { args: { op: "matmul", bytes: 2, flops_x: 2 }, pick: "frac", expect: (2047 / 6) / (2 * H100_PEAK / H100_BW), tol: 1e-6 }, // ...memory-bound at 2x FLOP/s
    { args: { op: "relu", bytes: 4 }, pick: "bytes_per_flop", expect: 8 },                         // slide p25 fp32
    { args: { op: "relu", bytes: 2 }, pick: "bytes_per_flop", expect: 4, from: "lecture_05:low-precision-intensity:predict" },
    { args: { op: "relu", bytes: 2, t_fp32_us: 10 }, pick: "t_scaled_us", expect: 5, from: "lecture_05:low-precision-intensity:transfer" },
  ],
};
WIDGETS["fixture:lecture_05--roofline-wall"] = (root) => {
  const s = { op: "matvec", bytes: 2, flops_x: 1, bw_x: 1 };
  const cv = canvas(640, 340), read = el("div", { class: "readout" });
  const X0 = 64, X1 = 620, Y0 = 300, Y1 = 24;
  const x = ai => X0 + (Math.log10(ai) + 2) / 6 * (X1 - X0);          // 1e-2 .. 1e4 FLOPs/byte
  const y = f => Y0 - (Math.log10(f) - 10) / 6 * (Y0 - Y1);           // 1e10 .. 1e16 FLOP/s
  const draw = () => {
    const m = roofModel(s), ctx = cv.getContext("2d"); ctx.clearRect(0, 0, 640, 340);
    ctx.strokeStyle = K.rule; ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(X0, Y1); ctx.lineTo(X0, Y0); ctx.lineTo(X1, Y0); ctx.stroke();
    ctx.fillStyle = K.muted; ctx.font = "12px sans-serif";
    for (let e = -2; e <= 4; e++) ctx.fillText(`1e${e}`, x(10 ** e) - 10, Y0 + 15);
    for (let e = 10; e <= 16; e += 2) ctx.fillText(`1e${e}`, 22, y(10 ** e) + 4);
    ctx.fillText("arithmetic intensity (FLOPs / byte)", 380, 334);
    ctx.save(); ctx.translate(12, 210); ctx.rotate(-Math.PI / 2); ctx.fillText("attainable FLOP/s", 0, 0); ctx.restore();
    const roof = (P, B, col, dash, w) => { ctx.strokeStyle = col; ctx.setLineDash(dash); ctx.lineWidth = w; ctx.beginPath(); ctx.moveTo(x(1e-2), y(1e-2 * B)); ctx.lineTo(x(P / B), y(P)); ctx.lineTo(x(1e4), y(P)); ctx.stroke(); ctx.setLineDash([]); ctx.lineWidth = 1; };
    const P = H100_PEAK * s.flops_x, B = H100_BW * s.bw_x;
    if (s.flops_x !== 1 || s.bw_x !== 1) { roof(H100_PEAK, H100_BW, K.rule, [5, 4], 2); ctx.fillStyle = K.muted; ctx.fillText("dashed: H100 roof", X1 - 110, Y0 - 8); }
    roof(P, B, K.ink, [], 2);
    ctx.strokeStyle = K.hilite; ctx.setLineDash([2, 3]); ctx.beginPath(); ctx.moveTo(x(m.kink), y(P)); ctx.lineTo(x(m.kink), Y0); ctx.stroke(); ctx.setLineDash([]);
    ctx.fillStyle = K.ink; ctx.fillText(`kink ${m.kink.toFixed(0)} FLOPs/B`, Math.min(x(m.kink) + 6, X1 - 130), Math.max(14, y(P) - 14));
    ctx.fillStyle = K.muted; ctx.fillText("memory-bound: slope = bandwidth", x(0.012), y(0.012 * B) - 30); ctx.fillText("compute-bound: flat = peak", X1 - 170, y(P) + 28);
    for (const [k, o] of Object.entries(ROOF_OPS)) {
      const mo = roofModel({ ...s, op: k }), att = mo.frac * P, sel = k === s.op;
      ctx.fillStyle = mo.bound === "memory" ? K.accent2 : K.ok;
      ctx.beginPath(); ctx.arc(x(mo.ai), y(att), sel ? 7 : 4, 0, 7); ctx.fill();
      ctx.fillStyle = sel ? K.ink : K.muted; ctx.font = sel ? "bold 12px sans-serif" : "12px sans-serif";
      ctx.fillText(o.name, x(mo.ai) + 8, y(att) + (k === "gelu" || k === "matvec" ? 16 : 4)); ctx.font = "12px sans-serif";
    }
    read.innerHTML = `kink = peak ÷ bandwidth = <b>${m.kink.toFixed(0)}</b> FLOPs/byte (${m.kink_ratio.toFixed(2)}× the H100's 295)<br>
      ${ROOF_OPS[s.op].name}: ${m.ai < 1 ? m.ai.toPrecision(2) : m.ai.toFixed(1)} FLOPs/byte = ${m.bytes_per_flop < 1 ? m.bytes_per_flop.toPrecision(2) : m.bytes_per_flop.toFixed(2)} bytes/FLOP → ${regime(m.bound + "-bound", m.bound === "compute")}
      <span class="big">${m.frac < 0.01 ? m.frac.toExponential(2) : (100 * m.frac).toFixed(1) + "%"} of peak</span>
      time ≈ ${fmt(m.t_us, 3)} µs · ${s.bytes === 4 ? "fp32 baseline" : `${m.time_vs_fp32.toFixed(2)}× the fp32 time`} (peak held fixed: dtype changes bytes, not FLOPs)
      ${src("H100: 1979e12/2 FLOP/s, 3.35e12 B/s (lecture_02.py:L74, L350-L351); op FLOP and byte counts lecture_02.py:L363-L467; ReLU bytes/FLOP slide p25; roofline shape slide p21, video 31:24-32:09")}`;
  };
  const fx = v => `${v}×`;
  root.append(el("div", { class: "widget" }, el("div", { class: "controls" },
    select("operation", Object.entries(ROOF_OPS).map(([k, o]) => [k, o.name]), s.op, v => { s.op = v; draw(); }),
    select("element type", [["4", "fp32 (4 B)"], ["2", "bf16 (2 B)"], ["1", "fp8 (1 B)"]], "2", v => { s.bytes = +v; draw(); }),
    slider("peak FLOP/s (× H100)", 1, 10, 1, 0.5, v => { s.flops_x = v; draw(); }, fx),
    slider("memory bandwidth (× H100)", 1, 10, 1, 0.5, v => { s.bw_x = v; draw(); }, fx),
    read), cv));
  draw();
};

// ==================================================== warp access: coalescing, row-major, alignment ==
const BURST = 128; // lecture_06.py:L127-L130: a warp's accesses combine into 128-byte transactions (cache lines); slide p37
const WARP = 32;   // lecture_05.pdf:p11, lecture_06.py:L83
const PAD_TO = 32; // video 1:05:15-1:07:38: "powers of 2, ideally divisible also by 32"
function warpAddrs({ pattern = "stride", k = 1, C = 1024, s = 4, r = 0, col = 0 }) {
  return Array.from({ length: WARP }, (_, t) => pattern === "stride" ? k * t * s : pattern === "row" ? (r * C + t) * s : (t * C + col) * s);
}
function accessModel(args = {}) {
  const { C = 1024, s = 4 } = args, addrs = warpAddrs({ ...args, C, s });
  const lines = new Set();
  for (const a of addrs) for (let b = Math.floor(a / BURST); b <= Math.floor((a + s - 1) / BURST); b++) lines.add(b);
  const rowStats = cols => { let tot = 0, mis = 0; for (let r = 0; r < 64; r++) { const a0 = r * cols * s, a1 = a0 + WARP * s - 1; tot += Math.floor(a1 / BURST) - Math.floor(a0 / BURST) + 1; if (a0 % BURST) mis++; } return [tot / 64, mis / 64]; };
  const padded_C = Math.ceil(C / PAD_TO) * PAD_TO, [avg_row_tx, misaligned] = rowStats(C), [avg_row_tx_padded] = rowStats(padded_C);
  return { transactions: lines.size, lines: [...lines].sort((a, b) => a - b), addrs, stride_bytes: addrs[1] - addrs[0], useful: WARP * s / (lines.size * BURST), padded_C, avg_row_tx, misaligned, avg_row_tx_padded };
}
MODELS["fixture:lecture_05--warp-access"] = {
  fn: accessModel,
  cases: [
    { args: { pattern: "stride", k: 1, s: 4 }, pick: "transactions", expect: 1, from: "lecture_05:memory-coalescing:predict" },
    { args: { pattern: "stride", k: 2, s: 4 }, pick: "transactions", expect: 2, from: "lecture_05:memory-coalescing:check" },
    { args: { pattern: "col", C: 32, s: 4 }, pick: "transactions", expect: 32, from: "lecture_05:memory-coalescing:transfer" },
    { args: { pattern: "col", C: 1024, s: 4 }, pick: "stride_bytes", expect: 4096, from: "lecture_05:supp-row-major-layout:predict" },
    { args: { pattern: "col", C: 1024, s: 4 }, pick: "transactions", expect: 32, from: "lecture_05:supp-row-major-layout:transfer" },
    { args: { pattern: "row", C: 1024, s: 4, r: 5 }, pick: "transactions", expect: 1 },   // reading along a row is coalesced
    { args: { C: 3000 }, pick: "padded_C", expect: 3008, from: "lecture_05:dim-divisibility-padding:transfer" },
    { args: { C: 50257 }, pick: "padded_C", expect: 50272 },                                // next multiple of 32 (Karpathy went to 50304 = 64·786)
    { args: { C: 50257, s: 4 }, pick: "avg_row_tx", expect: 1.96875, tol: 1e-6 },         // 63 of 64 rows straddle a burst boundary
    { args: { C: 50304, s: 4 }, pick: "avg_row_tx", expect: 1 },                           // every row starts on a boundary
    { args: { C: 3008, s: 4 }, pick: "avg_row_tx", expect: 1 },
    { args: { C: 3008, s: 2 }, pick: "avg_row_tx", expect: 1 },                             // 32-divisible also aligns bf16 row reads (64 B)
    { args: { C: 3072, s: 4 }, pick: "avg_row_tx", expect: 1 },                             // 64-divisible: no further gain (dim-divisibility check)
  ],
};
WIDGETS["fixture:lecture_05--warp-access"] = (root) => {
  const s = { pattern: "stride", k: 4, C: 1000, s: 4, r: 1, col: 0 };
  const cv = canvas(640, 330), read = el("div", { class: "readout" });
  const cInput = el("input", { type: "number", min: 8, max: 100000, value: s.C, style: "width:110px" });
  cInput.addEventListener("input", () => { const v = Math.max(8, Math.min(100000, Math.round(+cInput.value || 8))); s.C = v; draw(); });
  const setC = v => { s.C = v; cInput.value = v; draw(); };
  const draw = () => {
    const m = accessModel(s), ctx = cv.getContext("2d"); ctx.clearRect(0, 0, 640, 330); ctx.font = "12px sans-serif";
    // (1) what the warp reads, in the matrix picture (first 8 rows and columns)
    ctx.fillStyle = K.muted; ctx.fillText(s.pattern === "stride" ? `1-D array: thread t reads element ${s.k}·t` : s.pattern === "row" ? `row-major matrix: thread t reads A[${s.r}][t] (along a row)` : `row-major matrix: thread t reads A[t][${s.col}] (down a column)`, 10, 14);
    const cs = 11;
    if (s.pattern === "stride") {
      for (let i = 0; i < 48; i++) { const hit = i % s.k === 0 && i / s.k < WARP; ctx.fillStyle = hit ? K.accent : "#fff"; ctx.strokeStyle = K.rule; ctx.fillRect(10 + i * cs, 26, cs, cs); ctx.strokeRect(10 + i * cs, 26, cs, cs); }
      ctx.fillStyle = K.muted; ctx.fillText("elements 0..47 (blue = read by some thread)", 10, 56);
    } else {
      for (let i = 0; i < 8; i++) for (let j = 0; j < 8; j++) { const hit = s.pattern === "row" ? i === Math.min(s.r, 7) : j === s.col; ctx.fillStyle = hit ? K.accent : "#fff"; ctx.strokeStyle = K.rule; ctx.fillRect(10 + j * cs, 24 + i * cs, cs, cs); ctx.strokeRect(10 + j * cs, 24 + i * cs, cs, cs); }
      ctx.fillStyle = K.muted; ctx.fillText(`corner of a ${s.C}-column matrix · in memory, row i starts at byte i·C·s = i·${s.C * s.s}`, 110, 40);
      ctx.fillText(`neighbours along a row: ${s.s} B apart · down a column: ${s.C * s.s} B apart`, 110, 58);
    }
    // (2) the 128-byte transactions the 32 loads touch
    const y0 = 128, nb = Math.min(m.lines.length, 32), bw = Math.min(140, 600 / nb);
    ctx.fillStyle = K.muted; ctx.fillText(`128-byte transactions touched: ${m.transactions}${m.lines.length > 32 ? " (first 32 drawn)" : ""}`, 10, y0 - 10);
    m.lines.slice(0, 32).forEach((b, i) => {
      const x0 = 10 + i * bw; ctx.fillStyle = "rgba(184,88,42,.14)"; ctx.fillRect(x0, y0, bw - 2, 44);
      for (const a of m.addrs) if (Math.floor(a / BURST) === b) { ctx.fillStyle = K.accent; ctx.fillRect(x0 + (a % BURST) / BURST * (bw - 2), y0 + 6, Math.max(1.5, s.s / BURST * (bw - 2)), 32); }
      if (i > 0 && b !== m.lines[i - 1] + 1) { ctx.fillStyle = K.ink; ctx.fillText("…", x0 - 6, y0 + 58); }
    });
    // (3) alignment: where 32-element row reads fall on the 128-byte grid, current C vs padded C
    const y1 = 214;
    ctx.fillStyle = K.muted; ctx.fillText(`row reads (32 consecutive elements) on the burst grid · rows 0-3`, 10, y1 - 10);
    [[s.C, 10], [m.padded_C, 330]].forEach(([cols, xo], panel) => {
      const scale = 290 / (3 * BURST);
      ctx.fillStyle = K.ink; ctx.fillText(panel ? `padded C = ${cols}` : `C = ${cols}`, xo, y1 + 8);
      for (let r = 0; r < 4; r++) {
        const a0 = (r * cols * s.s) % BURST, len = WARP * s.s, yy = y1 + 16 + r * 22;
        for (let g = 0; g <= 3; g++) { ctx.strokeStyle = K.rule; ctx.beginPath(); ctx.moveTo(xo + g * BURST * scale, yy - 2); ctx.lineTo(xo + g * BURST * scale, yy + 16); ctx.stroke(); }
        const crosses = Math.floor((a0 + len - 1) / BURST) > 0;
        ctx.fillStyle = crosses ? K.accent2 : K.accent; ctx.fillRect(xo + a0 * scale, yy, Math.min(len, 3 * BURST - a0) * scale, 14);
      }
      ctx.fillStyle = K.muted; ctx.fillText(`avg ${(panel ? m.avg_row_tx_padded : m.avg_row_tx).toFixed(2)} transactions per row read`, xo, y1 + 112);
    });
    const coal = m.transactions <= Math.ceil(WARP * s.s / BURST);
    read.innerHTML = `consecutive threads are <b>${m.stride_bytes} bytes</b> apart<br><span class="big">${m.transactions} transaction${m.transactions > 1 ? "s" : ""}</span>
      ${regime(coal ? "coalesced" : "not coalesced", coal)} · ${(100 * m.useful).toFixed(m.useful < 0.1 ? 1 : 0)}% of moved bytes used<br>
      rows starting off a 128-byte boundary: ${(100 * m.misaligned).toFixed(0)}% · pad C to a multiple of 32 → ${m.padded_C}
      ${src("128-byte transactions lecture_06.py:L127-L130; warp = 32 (p11); row-major stride p39; padding rule and vocab 50257 → 50304 video 1:05:15-1:07:38. Toy model: one warp reading 32 consecutive elements. The lecture's A100 plot shows divisors 16 and 32 already tie; this model needs 32, because real tile loads are wider than one element per thread.")}`;
  };
  const kSl = slider("stride k (array pattern)", 1, 32, s.k, 1, v => { s.k = v; draw(); });
  root.append(el("div", { class: "widget" }, el("div", { class: "controls" },
    select("access pattern", [["stride", "array: thread t reads element k·t"], ["row", "matrix: A[r][t] (along a row)"], ["col", "matrix: A[t][k] (down a column)"]], s.pattern, v => { s.pattern = v; draw(); }),
    kSl,
    el("label", {}, "columns C ", cInput),
    presets("C =", [["1000", () => setC(1000)], ["1024", () => setC(1024)], ["3000", () => setC(3000)], ["50257", () => setC(50257)], ["50304", () => setC(50304)]]),
    select("element type", [["4", "fp32 (4 B)"], ["2", "bf16 (2 B)"], ["1", "fp8 (1 B)"]], "4", v => { s.s = +v; draw(); }),
    slider("row r (row pattern)", 0, 7, s.r, 1, v => { s.r = v; draw(); }),
    read), cv));
  draw();
};

// ================================================================ wave quantization ==
const SM_COUNT = { A100: 108, H100: 132, B200: 148 }; // lecture_06.py:L44
const TILE_M = 256, TILE_N = 128;                     // lecture_05.pdf:p48 ("a tile size of 256 × 128")
function waveModel({ M = 1536, sms = 108, tm = TILE_M, tn = TILE_N } = {}) {
  const tr = Math.ceil(M / tm), tc = Math.ceil(M / tn), tiles = tr * tc, waves = Math.ceil(tiles / sms), last_busy = tiles - (waves - 1) * sms;
  return { tiles_rows: tr, tiles_cols: tc, tiles, waves, last_busy, last_idle: sms - last_busy, util: (M * M) / (tiles * tm * tn) };
}
MODELS["fixture:lecture_05--wave-grid"] = {
  fn: waveModel,
  cases: [
    { args: { M: 1792, sms: 108 }, pick: "tiles", expect: 98 },   // p48: 7 × 14 = 98
    { args: { M: 1793, sms: 108 }, pick: "tiles", expect: 120 },  // p48: 8 × 15 = 120
    { args: { M: 1792, sms: 108 }, pick: "waves", expect: 1 },
    { args: { M: 1793, sms: 108 }, pick: "waves", expect: 2 },    // predict: the jump
    { args: { M: 1793, sms: 108 }, pick: "last_busy", expect: 12 }, // video 1:08:12: "I've got 12 more of these guys"
    { args: { M: 1793, sms: 132 }, pick: "waves", expect: 1, from: "lecture_05:tile-wave-quantization:transfer" },
    { args: { M: 1793, sms: 132 }, pick: "last_idle", expect: 12 },
    { args: { M: 2048, sms: 108 }, pick: "last_busy", expect: 20, from: "lecture_05:tile-wave-quantization:check" },
  ],
};
WIDGETS["fixture:lecture_05--wave-grid"] = (root) => {
  const s = { M: 1536, sms: 108 };
  const cv = canvas(640, 320), read = el("div", { class: "readout" });
  const draw = () => {
    const m = waveModel(s), ctx = cv.getContext("2d"); ctx.clearRect(0, 0, 640, 320); ctx.font = "12px sans-serif";
    const cols = 12, cell = 10, shown = Math.min(m.waves, 4), pw = 152;
    const partialR = s.M % TILE_M !== 0, partialC = s.M % TILE_N !== 0;
    for (let w = 0; w < shown; w++) {
      const x0 = 8 + w * pw; ctx.fillStyle = K.ink; ctx.fillText(`wave ${w + 1}`, x0, 14);
      for (let i = 0; i < s.sms; i++) {
        const tIdx = w * s.sms + i, busy = tIdx < m.tiles;
        const partial = busy && ((partialR && Math.floor(tIdx / m.tiles_cols) === m.tiles_rows - 1) || (partialC && tIdx % m.tiles_cols === m.tiles_cols - 1));
        ctx.fillStyle = !busy ? K.track : partial ? K.hilite : K.accent;
        ctx.fillRect(x0 + (i % cols) * (cell + 1), 22 + Math.floor(i / cols) * (cell + 1), cell, cell);
      }
      const busyHere = Math.min(s.sms, m.tiles - w * s.sms); ctx.fillStyle = K.muted; ctx.fillText(`${busyHere} / ${s.sms} SMs busy`, x0, 22 + Math.ceil(s.sms / cols) * (cell + 1) + 14);
    }
    if (m.waves > 4) { ctx.fillStyle = K.muted; ctx.fillText(`+ ${m.waves - 4} more waves`, 8 + 3 * pw, 190); }
    ctx.fillStyle = K.accent; ctx.fillRect(8, 196, 10, 10); ctx.fillStyle = K.muted; ctx.fillText("full tile", 22, 205);
    ctx.fillStyle = K.hilite; ctx.fillRect(90, 196, 10, 10); ctx.fillStyle = K.muted; ctx.fillText("partial tile (edge)", 104, 205);
    ctx.fillStyle = K.track; ctx.fillRect(222, 196, 10, 10); ctx.fillStyle = K.muted; ctx.fillText("idle SM", 236, 205);
    // staircase: waves vs M (time ∝ waves for a fixed tile)
    const X0 = 40, X1 = 620, Y0 = 306, Y1 = 222, lo = 1024, hi = 4096, wmax = waveModel({ M: hi, sms: s.sms }).waves;
    const x = M => X0 + (M - lo) / (hi - lo) * (X1 - X0), y = w => Y0 - w / wmax * (Y0 - Y1);
    ctx.strokeStyle = K.rule; ctx.beginPath(); ctx.moveTo(X0, Y1); ctx.lineTo(X0, Y0); ctx.lineTo(X1, Y0); ctx.stroke();
    ctx.strokeStyle = K.accent; ctx.lineWidth = 2; ctx.beginPath();
    for (let M = lo; M <= hi; M += 1) { const w = waveModel({ M, sms: s.sms }).waves; M === lo ? ctx.moveTo(x(M), y(w)) : ctx.lineTo(x(M), y(w)); }
    ctx.stroke(); ctx.lineWidth = 1;
    ctx.fillStyle = K.accent2; ctx.beginPath(); ctx.arc(x(s.M), y(m.waves), 5, 0, 7); ctx.fill();
    ctx.fillStyle = K.muted; ctx.fillText("waves", 2, Y1 + 4); ctx.fillText(`${lo}`, X0 - 10, Y0 + 12); ctx.fillText(`M = N = K → ${hi}`, X1 - 92, Y0 + 12);
    const one = m.waves === 1;
    read.innerHTML = `${m.tiles_rows} × ${m.tiles_cols} = <b>${m.tiles} tiles</b> of 256 × 128 on ${s.sms} SMs<br>
      <span class="big">${m.waves} wave${one ? "" : "s"}</span>${regime(one ? "fits in one wave" : `last wave: ${m.last_busy} busy, ${m.last_idle} idle`, one || m.last_busy / s.sms > 0.8)}<br>
      time ∝ waves; useful share of tile work ${(100 * m.util).toFixed(1)}%
      ${src("tile 256 × 128 and 1792 → 1793 (98 → 120 tiles) slide p48, video 1:07:50-1:09:20; SM counts lecture_06.py:L44; waves lecture_06.py:L134-L137. Tile-to-SM order is schematic.")}`;
  };
  const mSl = slider("M (= N = K)", 1024, 4096, s.M, 1, v => { s.M = v; draw(); });
  const smSl = slider("SMs", 80, 160, s.sms, 1, v => { s.sms = v; draw(); });
  const setM = v => { s.M = v; setRange(mSl, v); draw(); }, setS = v => { s.sms = v; setRange(smSl, v); draw(); };
  root.append(el("div", { class: "widget" }, el("div", { class: "controls" }, mSl,
    presets("M =", [["1792", () => setM(1792)], ["1793", () => setM(1793)], ["2048", () => setM(2048)]]),
    smSl, presets("GPU", Object.entries(SM_COUNT).map(([k, v]) => [`${k} (${v})`, () => setS(v)])), read), cv));
  draw();
};

// ================================================= memory hierarchy: who can share, what reuse buys ==
// H100 column of lecture_06.py:L46-L54 (sizes L46-L49, bandwidths L51-L54). Latencies are the
// professor's spoken A100 figures (video 11:00-11:20): L1/shared 20-30 cycles, global ~10x that.
const LEVELS = [
  { id: "reg", name: "registers", bw: 401, cap: "256 KB / SM", scope: "one thread" },
  { id: "smem", name: "L1 / shared memory", bw: 33, cap: "256 KB / SM", scope: "one block (its SM)" },
  { id: "l2", name: "L2 cache", bw: 12, cap: "50 MB", scope: "whole GPU (on die)" },
  { id: "hbm", name: "HBM (global)", bw: 3.35, cap: "80 GB", scope: "whole GPU (off chip)" },
];
const SCOPE_LEVEL = { thread: "reg", block: "smem", grid: "hbm" }; // p12: registers per thread, shared per block, across blocks via global
function memModel({ scope = "thread", reuse = 8, staged = false } = {}) {
  const L = LEVELS.find(l => l.id === SCOPE_LEVEL[scope]);
  const global_reads = staged ? 1 : reuse;
  return { level: L.name, bw: L.bw, ratio_to_hbm: L.bw / 3.35, global_reads, shared_reads: staged ? reuse : 0, saved: reuse - global_reads };
}
MODELS["fixture:lecture_05--memory-levels"] = {
  fn: memModel,
  cases: [
    { args: { scope: "block" }, pick: "bw", expect: 33, from: "lecture_05:gpu-memory-hierarchy:predict" },
    { args: { scope: "block" }, pick: "ratio_to_hbm", expect: 9.85, tol: 0.01 },   // "~10x"; slide p10 says "~8x"
    { args: { scope: "grid" }, pick: "bw", expect: 3.35 },                          // transfer: across blocks = global
    { args: { reuse: 64, staged: true }, pick: "saved", expect: 63, from: "lecture_05:shared-memory-vs-l1:check" },
    { args: { reuse: 1, staged: true }, pick: "saved", expect: 0 },                 // no reuse, nothing to gain (shared-memory-vs-l1 transfer)
  ],
};
WIDGETS["fixture:lecture_05--memory-levels"] = (root) => {
  const s = { scope: "thread", reuse: 8, staged: false };
  const cv = canvas(640, 330), read = el("div", { class: "readout" });
  const draw = () => {
    const m = memModel(s), ctx = cv.getContext("2d"); ctx.clearRect(0, 0, 640, 330); ctx.font = "12px sans-serif";
    const box = (x, y, w, h, fill, stroke, lw = 1) => { ctx.fillStyle = fill; ctx.fillRect(x, y, w, h); ctx.strokeStyle = stroke; ctx.lineWidth = lw; ctx.strokeRect(x, y, w, h); ctx.lineWidth = 1; };
    const hl = id => SCOPE_LEVEL[s.scope] === id || (s.scope === "grid" && id === "l2");
    const fillOf = id => hl(id) ? "rgba(227,178,60,.35)" : "#fbfaf6";
    // die outline
    ctx.strokeStyle = K.rule; ctx.setLineDash([4, 3]); ctx.strokeRect(6, 6, 628, 226); ctx.setLineDash([]); ctx.fillStyle = K.muted; ctx.fillText("GPU die", 14, 22);
    for (let sm = 0; sm < 2; sm++) {
      const x0 = 24 + sm * 312; box(x0, 30, 280, 118, "#fff", K.ink);
      ctx.fillStyle = K.ink; ctx.fillText(`SM ${sm}${sm === 0 ? " · block A" : " · block B"}`, x0 + 8, 46);
      for (let t = 0; t < 4; t++) { box(x0 + 8 + t * 66, 54, 60, 34, sm === 0 && t === 0 ? fillOf("reg") : "#fbfaf6", K.rule); ctx.fillStyle = K.muted; ctx.fillText(`regs t${t}`, x0 + 14 + t * 66, 75); }
      box(x0 + 8, 96, 264, 42, sm === 0 && s.scope !== "thread" ? fillOf("smem") : "#fbfaf6", K.rule);
      ctx.fillStyle = K.ink; ctx.fillText("L1 (cache, automatic) + shared (you load it)", x0 + 14, 114);
      ctx.fillStyle = K.muted; ctx.fillText("33 TB/s · 256 KB · ~20-30 cycles", x0 + 14, 130);
    }
    box(24, 162, 592, 56, fillOf("l2"), K.rule); ctx.fillStyle = K.ink; ctx.fillText("L2 cache (on die, shared by all SMs)", 34, 184); ctx.fillStyle = K.muted; ctx.fillText("12 TB/s · 50 MB", 34, 202);
    box(24, 248, 592, 56, fillOf("hbm"), K.ink); ctx.fillStyle = K.ink; ctx.fillText("HBM / global memory (off chip)", 34, 270); ctx.fillStyle = K.muted; ctx.fillText("3.35 TB/s · 80 GB · ~10× the L1 latency", 34, 288);
    ctx.fillStyle = K.muted; ctx.fillText("registers 401 TB/s · 256 KB per SM", 420, 22);
    // the path the shared data takes
    ctx.strokeStyle = K.accent2; ctx.lineWidth = 2.5; ctx.beginPath();
    if (s.scope === "thread") { ctx.arc(62, 71, 22, 0, 7); }
    else if (s.scope === "block") { ctx.moveTo(62, 88); ctx.lineTo(62, 100); ctx.moveTo(128, 88); ctx.lineTo(128, 100); }
    else { ctx.moveTo(140, 138); ctx.lineTo(140, 276); ctx.lineTo(460, 276); ctx.lineTo(460, 138); }
    ctx.stroke(); ctx.lineWidth = 1;
    ctx.fillStyle = K.ink; ctx.fillText(`staged reads of one element: ${s.staged ? `1 from HBM + ${s.reuse} from shared` : `${s.reuse} from HBM (L1 may or may not keep it)`}`, 14, 322);
    read.innerHTML = `data shared by <b>${{ thread: "one thread", block: "warps of one block", grid: "different blocks" }[s.scope]}</b> → fastest common level:<br>
      <span class="big">${m.level}</span>${m.bw} TB/s, ${m.ratio_to_hbm.toFixed(1)}× HBM ${s.scope === "grid" ? "(L2 may serve it at 12 TB/s)" : ""}<br><br>
      element used ${s.reuse}× by the block · ${s.staged ? regime("staged in shared memory", true) : regime("read from global each time", false)}<br>
      global reads per element: <b>${m.global_reads}</b> · saved: <b>${m.saved}</b>
      ${src("H100 sizes and bandwidths lecture_06.py:L46-L54; slide p10 states SRAM ~8× faster than DRAM (the table gives 33 / 3.35 ≈ 10×); latencies video 11:00-11:20 (A100); scopes slide p12; L1 vs shared video 12:54-13:18")}`;
  };
  root.append(el("div", { class: "widget" }, el("div", { class: "controls" },
    select("data is shared by", [["thread", "one thread only"], ["block", "warps of the same block"], ["grid", "threads in different blocks"]], s.scope, v => { s.scope = v; draw(); }),
    slider("uses of each element by the block", 1, 128, s.reuse, 1, v => { s.reuse = v; draw(); }),
    el("label", {}, el("input", { type: "checkbox", onchange: e => { s.staged = e.target.checked; draw(); } }), " stage the tile in shared memory first"),
    read), cv));
  draw();
};

// ====================================================== warps, register file and occupancy ==
// Warp = 32 consecutively numbered threads (p11; lecture_06.py:L83). SM limits from lecture_06.py:L104-L105:
// 65536 registers, 64 resident warps; blocks fit = registers // (threads × regs/thread) (L108-L112).
const SM_REGS = 65536, SM_WARPS = 64;
function occModel({ threads = 160, regs = 64 } = {}) {
  const warps = Math.ceil(threads / WARP), idle = warps * WARP - threads, regs_block = threads * regs;
  const by_regs = Math.floor(SM_REGS / regs_block), by_warps = Math.floor(SM_WARPS / warps), blocks = Math.min(by_regs, by_warps);
  const resident = blocks * warps;
  return { warps, idle, regs_block, blocks, resident_warps: resident, occupancy: resident / SM_WARPS, limiter: blocks === 0 ? "does not fit" : by_regs <= by_warps ? "registers" : "warp slots" };
}
MODELS["fixture:lecture_05--warps-occupancy"] = {
  fn: occModel,
  cases: [
    { args: { threads: 96, regs: 32 }, pick: "warps", expect: 3, from: "lecture_05:execution-hierarchy:predict" },
    { args: { threads: 100, regs: 32 }, pick: "warps", expect: 4, from: "lecture_05:execution-hierarchy:transfer" },
    { args: { threads: 100, regs: 32 }, pick: "idle", expect: 28 },
    { args: { threads: 200, regs: 32 }, pick: "idle", expect: 24, from: "lecture_05:execution-hierarchy:check" },
    { args: { threads: 128, regs: 160 }, pick: "blocks", expect: 3, from: "lecture_05:supp-latency-hiding:transfer" },
    { args: { threads: 128, regs: 160 }, pick: "occupancy", expect: 0.1875 },
    { args: { threads: 256, regs: 96 }, pick: "occupancy", expect: 0.25, from: "lecture_05:supp-latency-hiding:check" },
    { args: { threads: 256, regs: 32 }, pick: "occupancy", expect: 1 },     // warp slots, not registers, bind
    { args: { threads: 1024, regs: 255 }, pick: "blocks", expect: 0 },     // block needs more registers than the SM has
  ],
};
WIDGETS["fixture:lecture_05--warps-occupancy"] = (root) => {
  const s = { threads: 160, regs: 64 };
  const cv = canvas(640, 320), read = el("div", { class: "readout" });
  const BLK = [K.accent, K.accent2, K.ok, K.hilite, "#7a6aa8", "#4f8fa8", "#a8744f", "#8a9a5b"];
  const draw = () => {
    const m = occModel(s), ctx = cv.getContext("2d"); ctx.clearRect(0, 0, 640, 320); ctx.font = "12px sans-serif";
    // one block, drawn warp by warp (32 lanes per row)
    ctx.fillStyle = K.ink; ctx.fillText(`one block: ${s.threads} threads → ${m.warps} warps of 32 (hollow = idle lane)`, 8, 14);
    const cw = 6, rows = Math.ceil(m.warps / 2);
    for (let w = 0; w < m.warps; w++) {
      const col = w < rows ? 0 : 1, row = w % rows, x0 = 8 + col * 316, y0 = 22 + row * 9;
      for (let l = 0; l < WARP; l++) { const live = w * WARP + l < s.threads; ctx.fillStyle = live ? K.accent : "#fff"; ctx.strokeStyle = live ? K.accent : K.accent2; ctx.fillRect(x0 + l * (cw + 1), y0, cw, 7); if (!live) ctx.strokeRect(x0 + l * (cw + 1) + 0.5, y0 + 0.5, cw - 1, 6); }
    }
    // register file of one SM
    const yR = 188; ctx.fillStyle = K.ink; ctx.fillText(`SM register file: 65,536 registers · one block needs ${s.threads} × ${s.regs} = ${m.regs_block.toLocaleString()}`, 8, yR - 6);
    ctx.fillStyle = K.track; ctx.fillRect(8, yR, 620, 18);
    for (let b = 0; b < m.blocks; b++) { ctx.fillStyle = BLK[b % BLK.length]; ctx.fillRect(8 + b * m.regs_block / SM_REGS * 620, yR, m.regs_block / SM_REGS * 620 - 1, 18); }
    // warp slots
    const yW = 246; ctx.fillStyle = K.ink; ctx.fillText(`resident warp slots: ${m.resident_warps} of 64 used`, 8, yW - 6);
    for (let i = 0; i < SM_WARPS; i++) { const b = Math.floor(i / m.warps); ctx.fillStyle = i < m.resident_warps ? BLK[b % BLK.length] : K.track; ctx.fillRect(8 + i * 9.7, yW, 8.5, 18); }
    ctx.fillStyle = K.muted; ctx.fillText("each colour = one resident block; when a warp stalls on HBM the SM issues from another resident warp", 8, yW + 36);
    read.innerHTML = `${m.warps} warp${m.warps > 1 ? "s" : ""} per block · <b>${m.idle}</b> idle lane${m.idle === 1 ? "" : "s"} in the last warp<br>
      blocks per SM: <b>${m.blocks}</b> (limited by ${regime(m.limiter, m.limiter === "warp slots")})<br>
      <span class="big">occupancy ${m.resident_warps}/64 = ${m.occupancy.toFixed(4).replace(/0+$/, "").replace(/\.$/, "")}</span>
      fewer resident warps = fewer to switch to while one waits on memory
      ${src("warp of 32 slide p11, lecture_06.py:L83; 65536 registers and 64 warps per SM, blocks = registers // (threads × regs) lecture_06.py:L100-L112; warp switching lecture_06.py:L90, video 23:24-23:51. Only the two limits the lecture names are modelled.")}`;
  };
  root.append(el("div", { class: "widget" }, el("div", { class: "controls" },
    slider("threads per block", 32, 1024, s.threads, 4, v => { s.threads = v; draw(); }),
    slider("registers per thread", 16, 255, s.regs, 1, v => { s.regs = v; draw(); }),
    read), cv));
  draw();
};

// ============================================================ bf16 vs fp32 accumulator ==
// Video 36:34-36:57: tensor cores downcast the inputs but sum partial products in full precision and may
// emit fp32. bf16 = fp32 with the low 16 bits rounded off (8 significant bits), round to nearest even.
const f32 = new Float32Array(1), u32 = new Uint32Array(f32.buffer);
function toBf16(x) { f32[0] = x; const b = u32[0]; u32[0] = ((b + 0x7fff + ((b >>> 16) & 1)) & 0xffff0000) >>> 0; return f32[0]; }
function accumulate(n, acc, addend = 1) {
  const rnd = acc === "bf16" ? toBf16 : Math.fround; let sum = 0; const trace = [0];
  for (let i = 0; i < n; i++) { sum = rnd(sum + addend); trace.push(sum); }
  return { sum, trace };
}
function accModel({ n = 1024, acc = "bf16" } = {}) {
  const { sum } = accumulate(n, acc);
  const e = sum > 0 ? Math.floor(Math.log2(sum)) : 0;
  return { sum, out_bf16: toBf16(sum), exact: n, spacing_at_sum: 2 ** (e - 7) };
}
MODELS["fixture:lecture_05--bf16-accumulator"] = {
  fn: accModel,
  cases: [
    { args: { n: 4096, acc: "bf16" }, pick: "sum", expect: 256, from: "lecture_05:tensor-core-fp32-accumulate:transfer" },
    { args: { n: 4096, acc: "fp32" }, pick: "out_bf16", expect: 4096, from: "lecture_05:tensor-core-fp32-accumulate:check" },
    { args: { n: 256, acc: "bf16" }, pick: "sum", expect: 256 },           // exact up to 256
    { args: { n: 255, acc: "bf16" }, pick: "sum", expect: 255 },
    { args: { n: 300, acc: "bf16" }, pick: "spacing_at_sum", expect: 2 },  // between 256 and 512 values are 2 apart
    { args: { n: 8192, acc: "fp32" }, pick: "sum", expect: 8192 },
  ],
};
WIDGETS["fixture:lecture_05--bf16-accumulator"] = (root) => {
  const s = { n: 1024, acc: "bf16" };
  const cv = canvas(640, 300), read = el("div", { class: "readout" });
  const draw = () => {
    const ctx = cv.getContext("2d"); ctx.clearRect(0, 0, 640, 300); ctx.font = "12px sans-serif";
    const X0 = 56, X1 = 620, Y0 = 270, Y1 = 20, x = i => X0 + i / s.n * (X1 - X0), y = v => Y0 - v / s.n * (Y0 - Y1);
    ctx.strokeStyle = K.rule; ctx.beginPath(); ctx.moveTo(X0, Y1); ctx.lineTo(X0, Y0); ctx.lineTo(X1, Y0); ctx.stroke();
    ctx.fillStyle = K.muted; ctx.fillText("0", X0 - 12, Y0 + 4); ctx.fillText(`${s.n}`, X0 - 40, Y1 + 4); ctx.fillText(`ones added → ${s.n}`, X1 - 110, Y0 + 18); ctx.fillText("running sum", 4, Y1 - 6);
    const line = (tr, col, w) => { ctx.strokeStyle = col; ctx.lineWidth = w; ctx.beginPath(); const step = Math.max(1, Math.floor(tr.length / 600)); for (let i = 0; i < tr.length; i += step) i ? ctx.lineTo(x(i), y(tr[i])) : ctx.moveTo(x(i), y(tr[i])); ctx.lineTo(x(tr.length - 1), y(tr[tr.length - 1])); ctx.stroke(); ctx.lineWidth = 1; };
    const a32 = accumulate(s.n, "fp32"), a16 = accumulate(s.n, "bf16");
    line(a32.trace, s.acc === "fp32" ? K.accent : "rgba(36,102,141,.3)", s.acc === "fp32" ? 3 : 2);
    line(a16.trace, s.acc === "bf16" ? K.accent2 : "rgba(184,88,42,.3)", s.acc === "bf16" ? 3 : 2);
    if (s.n > 256) { ctx.strokeStyle = K.hilite; ctx.setLineDash([3, 3]); ctx.beginPath(); ctx.moveTo(X0, y(256)); ctx.lineTo(X1, y(256)); ctx.stroke(); ctx.setLineDash([]); ctx.fillStyle = K.muted; ctx.fillText("256: next value 257 ties, rounds to even (256)", X0 + 6, y(256) - 6); }
    ctx.fillStyle = K.accent; ctx.fillText("fp32 accumulator", X1 - 120, y(a32.sum) + 16 < Y0 ? y(a32.sum) + 16 : Y0 - 8);
    ctx.fillStyle = K.accent2; ctx.fillText("bf16 accumulator", X1 - 120, y(a16.sum) - 8);
    const m = accModel(s), stuck = m.sum < m.exact;
    read.innerHTML = `add 1.0 (exact in bf16) ${s.n} times, rounding the running sum to <b>${s.acc}</b> after every add<br>
      <span class="big">sum = ${m.sum}</span>${regime(stuck ? `stuck: true sum is ${m.exact}` : "exact", !stuck)}<br>
      spacing of representable ${s.acc} values at the sum: ${s.acc === "bf16" ? m.spacing_at_sum : "1 or finer (fp32 is exact to 2^24)"}<br>
      stored output after casting to bf16: <b>${m.out_bf16}</b>
      ${src("video 36:34-36:57: downcast inputs, accumulate partial sums in full precision, may emit fp32. bf16 has 8 significant bits; rounding is round-to-nearest-even.")}`;
  };
  root.append(el("div", { class: "widget" }, el("div", { class: "controls" },
    slider("number of ones (dot-product length)", 16, 8192, s.n, 16, v => { s.n = v; draw(); }),
    select("accumulator", [["bf16", "bf16 (same as the inputs)"], ["fp32", "fp32 (what tensor cores do)"]], s.acc, v => { s.acc = v; draw(); }),
    read), cv));
  draw();
};

// =========================================================== FP8 matmuls: diluted saving ==
// Video 42:27-42:46: "20% to 30% savings, maybe more ... not going to be a 2x speedup because you have to do all
// these quantization operations"; 45:31-45:43: "basically linear improvements for just multiplying the quantized
// numbers ... quantize and dequantize means that the benefits are more diluted"; 40:50-42:27: MXFP8 keeps a
// separately quantized transposed copy. raw = 0.5 is "half the bits, linear improvement"; overhead is the control.
function fp8Model({ share = 0.5, raw = 0.5, overhead = 0 } = {}) {
  const mm = raw + overhead, mm_saving = 1 - mm;
  return { mm_time: mm, mm_saving, mm_speedup: 1 / mm, step_saving: share * mm_saving, step_time: 1 - share * mm_saving };
}
MODELS["fixture:lecture_05--fp8-step-saving"] = {
  fn: fp8Model,
  cases: [
    { args: { raw: 0.5, overhead: 0.25 }, pick: "mm_saving", expect: 0.25, from: "lecture_05:low-bit-speedup-diluted:check" },
    { args: { share: 0.6, raw: 0.5, overhead: 0.25 }, pick: "step_saving", expect: 0.15, from: "lecture_05:low-bit-speedup-diluted:predict" },
    { args: { share: 0.9, raw: 0.5, overhead: 0.25 }, pick: "step_saving", expect: 0.225, from: "lecture_05:low-bit-speedup-diluted:transfer" },
    { args: { share: 0.6, raw: 0.5, overhead: 0 }, pick: "step_saving", expect: 0.3 },    // the "half the bits, half the time" belief
    { args: { raw: 0.5, overhead: 0 }, pick: "mm_speedup", expect: 2 },
    { args: { raw: 0.5, overhead: 0.5 }, pick: "mm_saving", expect: 0 },                  // overhead eats the whole gain
  ],
};
WIDGETS["fixture:lecture_05--fp8-step-saving"] = (root) => {
  const s = { share: 0.5, raw: 0.5, overhead: 0 };
  const cv = canvas(640, 230), read = el("div", { class: "readout" });
  const draw = () => {
    const m = fp8Model(s), ctx = cv.getContext("2d"); ctx.clearRect(0, 0, 640, 230); ctx.font = "12px sans-serif";
    const X0 = 110, W = 500, bar = (yy, parts) => { let x0 = X0; for (const [v, col, t] of parts) { const w = v * W; ctx.fillStyle = col; ctx.fillRect(x0, yy, Math.max(0, w - 1), 34); ctx.fillStyle = "#fff"; if (w > 60) ctx.fillText(t, x0 + 6, yy + 21); x0 += w; } };
    ctx.fillStyle = K.ink; ctx.fillText("bf16 step", 10, 52); ctx.fillText("FP8 step", 10, 122);
    bar(32, [[s.share, K.accent, "matmuls"], [1 - s.share, K.muted, "everything else"]]);
    bar(102, [[s.share * s.raw, K.accent, "multiply"], [s.share * s.overhead, K.hilite, "quantize + T copy"], [1 - s.share, K.muted, "everything else"]]);
    // professor's 20-30% band, on the matmul part
    const bx = v => X0 + s.share * (1 - v) * W;
    ctx.fillStyle = "rgba(60,141,90,.18)"; ctx.fillRect(bx(0.3), 146, bx(0.2) - bx(0.3), 10);
    ctx.fillStyle = K.ok; ctx.fillText("matmul end at 20-30% saving (video 42:27)", Math.min(bx(0.3), 360), 172);
    ctx.strokeStyle = K.ink; ctx.setLineDash([3, 3]); ctx.beginPath(); ctx.moveTo(X0 + W * m.step_time, 96); ctx.lineTo(X0 + W * m.step_time, 142); ctx.moveTo(X0 + W, 26); ctx.lineTo(X0 + W, 142); ctx.stroke(); ctx.setLineDash([]);
    ctx.fillStyle = K.muted; ctx.fillText(`step time ${m.step_time.toFixed(3)} of bf16`, X0, 200); ctx.fillText("bar length = time; the multiply alone is 0.5 of the bf16 matmul (half the bits)", X0, 218);
    const inBand = m.mm_saving >= 0.2 - 1e-9 && m.mm_saving <= 0.3 + 1e-9;
    read.innerHTML = `matmul time: ${s.raw} multiply + ${s.overhead.toFixed(2)} overhead = <b>${m.mm_time.toFixed(2)}</b> of bf16 → ${m.mm_speedup.toFixed(2)}× faster<br>
      matmul saving ${regime((100 * m.mm_saving).toFixed(0) + "%", inBand)} ${m.mm_saving >= 0.5 - 1e-9 ? "(the naive 2×)" : inBand ? "(the quoted 20-30%)" : ""}<br>
      <span class="big">step time −${(100 * m.step_saving).toFixed(1)}%</span>= matmul share ${s.share} × matmul saving ${m.mm_saving.toFixed(2)}
      ${src("video 42:27-42:46 (20-30%, not 2x), 45:31-45:43 (linear for the multiply, diluted by quantize/dequantize), 40:50-42:27 (MXFP8 transposed copy; only safe layers quantized). The 0.5 multiply time is the 'half the bits' assumption, not a measurement.")}`;
  };
  root.append(el("div", { class: "widget" }, el("div", { class: "controls" },
    slider("matmul share of step time", 0.1, 1, s.share, 0.05, v => { s.share = v; draw(); }),
    slider("quantize + transposed-copy overhead (× bf16 matmul time)", 0, 0.5, s.overhead, 0.05, v => { s.overhead = v; draw(); }, v => v.toFixed(2)),
    read), cv));
  draw();
};

// ============================================== matrix-unit granularity: few big vs many small ==
// Video 20:53-22:03: H100-class GPU ~528 matrix-multiply units vs a TPU's 8; the TPU's big units want big
// matmuls ("the tensor core refuses to accept anything smaller than a 64-dimensional input").
// Utilization = useful MACs / MACs the padded passes occupy. Unit sizes and the 4 × 256² = 256 × 32² budget are
// the learner's controls (the latter is the KP transfer's design question), not chip specifications.
function unitModel({ U = 128, m = 64, n = 128 } = {}) {
  const pm = Math.ceil(m / U), pn = Math.ceil(n / U);
  return { util: (m * n) / (pm * U * pn * U), passes: pm * pn, units_in_budget: (4 * 256 * 256) / (U * U) };
}
MODELS["fixture:lecture_05--matrix-unit-fill"] = {
  fn: unitModel,
  cases: [
    { args: { U: 128, m: 16, n: 128 }, pick: "util", expect: 0.125, from: "lecture_05:tpu-matmul-unit-granularity:check" },
    { args: { U: 256, m: 32, n: 32 }, pick: "util", expect: 1 / 64 },   // transfer: big unit on a 32×32 problem
    { args: { U: 32, m: 32, n: 32 }, pick: "util", expect: 1 },         // transfer: small units fill exactly
    { args: { U: 32 }, pick: "units_in_budget", expect: 256 },          // same MAC budget as 4 units of 256²
    { args: { U: 128, m: 129, n: 128 }, pick: "util", expect: 129 / 256 }, // one row over: a second, nearly empty pass
  ],
};
WIDGETS["fixture:lecture_05--matrix-unit-fill"] = (root) => {
  const s = { U: 128, m: 64, n: 128 };
  const cv = canvas(640, 300), read = el("div", { class: "readout" });
  const draw = () => {
    const r = unitModel(s), ctx = cv.getContext("2d"); ctx.clearRect(0, 0, 640, 300); ctx.font = "12px sans-serif";
    const pm = Math.ceil(s.m / s.U), pn = Math.ceil(s.n / s.U), kk = Math.max(pm, pn), gap = kk > 8 ? 1 : 4, side = (250 - gap * (kk - 1)) / kk, sc = side / s.U;
    ctx.fillStyle = K.ink; ctx.fillText(`${s.m} × ${s.n} operand on ${s.U} × ${s.U} unit(s): ${r.passes} pass${r.passes > 1 ? "es" : ""}`, 10, 16);
    for (let i = 0; i < pm; i++) for (let j = 0; j < pn; j++) {
      const x0 = 10 + j * (side + gap), y0 = 26 + i * (side + gap);
      ctx.fillStyle = K.track; ctx.fillRect(x0, y0, side, side);
      const fh = Math.min(s.U, s.m - i * s.U), fw = Math.min(s.U, s.n - j * s.U);
      ctx.fillStyle = K.accent; ctx.fillRect(x0, y0, fw * sc, fh * sc);
      ctx.strokeStyle = K.ink; ctx.strokeRect(x0, y0, side, side);
    }
    // design comparison at a fixed MAC budget (KP transfer): many independent m×n matmuls
    const designs = [[256, "4 units of 256²"], [32, "256 units of 32²"], [s.U, `${r.units_in_budget < 1 ? r.units_in_budget.toFixed(2) : Math.round(r.units_in_budget)} units of ${s.U}²`]];
    ctx.fillStyle = K.ink; ctx.fillText(`same MAC budget, many independent ${s.m}×${s.n} matmuls:`, 300, 40);
    designs.forEach(([U, name], k) => { const u = unitModel({ U, m: s.m, n: s.n }).util, yy = 56 + k * 44; ctx.fillStyle = K.muted; ctx.fillText(name + (k === 2 ? " (current)" : ""), 300, yy); ctx.fillStyle = K.track; ctx.fillRect(300, yy + 6, 300, 16); ctx.fillStyle = u > 0.5 ? K.ok : K.accent2; ctx.fillRect(300, yy + 6, 300 * u, 16); ctx.fillStyle = K.ink; ctx.fillText(`${(100 * u).toFixed(u < 0.1 ? 2 : 1)}% useful`, 300 + Math.min(300 * u + 6, 230), yy + 19); });
    ctx.fillStyle = K.muted; ctx.fillText("blue = real operand, grey = padding the unit computes anyway", 10, 290);
    read.innerHTML = `useful fraction = ${s.m}·${s.n} / (${pm * s.U}·${pn * s.U})<br><span class="big">${(100 * r.util).toFixed(r.util < 0.1 ? 2 : 1)}% useful</span>
      ${regime(r.util >= 0.999 ? "fills the unit" : "padding wasted", r.util >= 0.999)}<br>an operand smaller than the unit leaves its array idle; a GPU's many small units can each take one small matmul
      ${src("video 20:53-22:03: 528 matrix-multiply units on the GPU vs 8 on a TPU; the batch sweep that stops at 64. Unit and operand sizes are what-if controls.")}`;
  };
  const p2 = v => 2 ** v;
  root.append(el("div", { class: "widget" }, el("div", { class: "controls" },
    slider("unit size U", 4, 8, 7, 1, v => { s.U = p2(v); draw(); }, v => `${p2(v)} × ${p2(v)}`),
    slider("operand rows m (batch)", 1, 512, s.m, 1, v => { s.m = v; draw(); }),
    slider("operand columns n", 1, 512, s.n, 1, v => { s.n = v; draw(); }),
    read), cv));
  draw();
};
