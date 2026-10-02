// Widgets for thread lecture_06. Register as "fixture:<id>" -> (root, notice) => void.
// Each widget's numeric model is a pure function in MODELS so tools/check_widgets.mjs can test it
// against the knowledge points' own answers. Sources: official/lectures/lecture_06.py (L…) and the
// human-made captions in lectures/lecture_06/transcript.json (video:M:SS).
import { el, fmt, slider } from "../widgets_lib.js";

export const MODELS = {};
export const WIDGETS = {};

const C = { ink: "var(--ink)", muted: "var(--muted)", rule: "var(--rule)", accent: "var(--accent)", accent2: "var(--accent2)", hilite: "var(--hilite)", ok: "var(--ok)", card: "var(--card)", paper: "var(--paper)" };
const PID = ["#24668d", "#b8582a", "#3c8d5a", "#7a5a99", "#8d6a24", "#3f7f86"];
const cdiv = (a, b) => Math.ceil(a / b);
const svg = (w, h, body) => `<svg viewBox="0 0 ${w} ${h}" width="100%" style="max-width:${w}px;display:block" role="img">${body}</svg>`;
const rect = (x, y, w, h, fill, extra = "") => `<rect x="${x.toFixed(1)}" y="${y.toFixed(1)}" width="${Math.max(0.5, w).toFixed(1)}" height="${h.toFixed(1)}" fill="${fill}" ${extra}/>`;
const text = (x, y, s, extra = "") => `<text x="${x.toFixed(1)}" y="${y.toFixed(1)}"${/font-size=/.test(extra) ? "" : ' font-size="12"'}${/fill=/.test(extra) ? "" : ` fill="${C.muted}"`} ${extra}>${s}</text>`;
const buttons = (opts, cur, on) => el("div", { style: "margin:6px 0" }, ...opts.map(([v, lab]) => el("button", { class: v === cur ? "primary" : "", onclick: () => on(v) }, lab)), " ");

// ======================================================= launch grid ==
// What the launch grid counts. Elementwise (GeLU): host num_blocks = triton.cdiv(num_elements, BLOCK_SIZE)
// (lecture_06.py:L345-L353), offsets = pid*BLOCK_SIZE + arange, mask = offsets < num_elements (L369-L376).
// Row reduction (softmax / row sum): grid (M,), one program per row (L448-L457, L510-L513); a row
// longer than BLOCK_SIZE is looped over in tiles, for start in range(0, N, BLOCK_SIZE), other=0.0,
// one tl.sum at the end (L517-L535). "These are not blocks. These are tiles." (video:1:11:06-1:11:28).
// Column reduction: the professor's "multiply by the row stride" variant (video:1:05:05-1:05:21).
// Matmul+ReLU: grid (cdiv(M, BLOCK_M), cdiv(N, BLOCK_N)), K loop in steps of BLOCK_K (L620-L621, L659-L667).
// The row values for the accumulator are 1..N, as in the KP's own prompts.
export function launchGridModel(a) {
  const mode = a.mode || "elementwise";
  if (mode === "elementwise") {
    const n = a.n, B = a.block, programs = a.grid === "floor" ? Math.floor(n / B) : cdiv(n, B);
    const covered = Math.min(n, programs * B);
    const lastTrue = programs ? Math.min(B, n - (programs - 1) * B) : 0;
    return { programs, tiles: 1, lastTrue, masked: programs ? B - lastTrue : 0, uncomputed: n - covered, firstUncomputed: covered < n ? covered : n, pieceKind: "block" };
  }
  if (mode === "row" || mode === "col") {
    const rows = a.rows, cols = a.cols, B = a.block;
    const programs = mode === "row" ? rows : cols, len = mode === "row" ? cols : rows;
    const tiles = cdiv(len, B), lastTrue = len - (tiles - 1) * B;
    const acc = Array.from({ length: B }, (_, l) => { let s = 0; for (let t = 0; t < tiles; t++) { const c = t * B + l; if (c < len) s += c + 1; } return s; });
    return { programs, tiles, lastTrue, masked: B - lastTrue, uncomputed: 0, acc, accLane: acc[a.lane ?? 0], sum: len * (len + 1) / 2, pieceKind: "tile" };
  }
  // matmul
  const gm = cdiv(a.M, a.BM), gn = cdiv(a.N, a.BN);
  return { programs: gm * gn, gridM: gm, gridN: gn, tiles: cdiv(a.K, a.BK), uncomputed: 0, pieceKind: "tile" };
}

function launchGrid(root) {
  // defaults deliberately differ from every prompt's case
  const s = { mode: "elementwise", n: 6000, block: 1024, grid: "cdiv", rows: 256, cols: 2500, rblock: 1024, M: 512, N: 768, K: 512, BM: 64, BN: 64, BK: 32 };
  const ctl = el("div", { class: "controls" }), fig = el("div"), read = el("div", { class: "readout" });
  const build = () => {
    ctl.replaceChildren(buttons([["elementwise", "elementwise (GeLU)"], ["row", "row reduction (row sum)"], ["col", "column sum"], ["matmul", "matmul + ReLU"]], s.mode, v => { s.mode = v; build(); draw(); }));
    const add = (...xs) => xs.forEach(x => ctl.append(x));
    const p2 = v => 2 ** v;
    if (s.mode === "elementwise") add(
      slider("num_elements", 256, 20000, s.n, 8, v => { s.n = v; draw(); }),
      slider("BLOCK_SIZE", 6, 12, Math.log2(s.block), 1, v => { s.block = p2(v); draw(); }, v => p2(v)),
      buttons([["cdiv", "grid = cdiv(n, BLOCK_SIZE)"], ["floor", "grid = n // BLOCK_SIZE (bug)"]], s.grid, v => { s.grid = v; build(); draw(); }));
    else if (s.mode === "row" || s.mode === "col") add(
      slider("rows M", 1, 4096, s.rows, 1, v => { s.rows = v; draw(); }),
      slider("cols N", 1, 8192, s.cols, 1, v => { s.cols = v; draw(); }),
      slider(s.mode === "row" ? "BLOCK_SIZE (tile along the row)" : "tile along the column", 2, 12, Math.log2(s.rblock), 1, v => { s.rblock = p2(v); draw(); }, v => p2(v)));
    else add(
      slider("M", 64, 4096, s.M, 64, v => { s.M = v; draw(); }), slider("N", 64, 4096, s.N, 64, v => { s.N = v; draw(); }), slider("K", 64, 4096, s.K, 64, v => { s.K = v; draw(); }),
      slider("BLOCK_M = BLOCK_N", 4, 8, Math.log2(s.BM), 1, v => { s.BM = s.BN = p2(v); draw(); }, v => p2(v)),
      slider("BLOCK_K", 3, 7, Math.log2(s.BK), 1, v => { s.BK = p2(v); draw(); }, v => p2(v)));
  };
  const draw = () => {
    const W = 640;
    if (s.mode === "elementwise") {
      const r = launchGridModel({ mode: "elementwise", n: s.n, block: s.block, grid: s.grid });
      const slots = Math.max(cdiv(s.n, s.block), r.programs) * s.block, px = (W - 40) / slots;
      let body = text(20, 16, `the vector, ${s.n} elements · one coloured window per program (block) · all programs independent`);
      for (let p = 0; p < r.programs; p++) {
        const x = 20 + p * s.block * px, real = Math.min(s.block, s.n - p * s.block);
        body += rect(x, 30, real * px, 46, PID[p % PID.length], `opacity="0.75"`) + (real < s.block ? rect(x + real * px, 30, (s.block - real) * px, 46, "#e6e1d6") : "");
        body += rect(x, 30, 1, 46, C.card);
      }
      if (r.uncomputed) body += rect(20 + r.programs * s.block * px, 30, r.uncomputed * px, 46, "none", `stroke="${C.accent2}" stroke-width="2" stroke-dasharray="4 3"`) + text(W - 20, 92, `never computed: ${r.uncomputed}`, `fill="${C.accent2}" text-anchor="end"`);
      if (r.masked && s.grid === "cdiv") body += text(W - 20, 92, `last program (pid ${r.programs - 1}): mask True ${r.lastTrue} / ${s.block}`, `text-anchor="end"`);
      fig.innerHTML = svg(W, 104, body);
      read.innerHTML = `<span class="big">${r.programs} programs</span> = ${s.grid === "cdiv" ? `cdiv(${s.n}, ${s.block})` : `${s.n} // ${s.block}`} · each runs once (1 piece = 1 block)<br>
        last program: <b>${r.lastTrue}</b> of ${s.block} mask entries True${r.uncomputed ? ` · <b style="color:var(--accent2)">${r.uncomputed} elements never computed</b> (from index ${r.firstUncomputed}; they keep whatever torch.empty_like left)` : " · no element missed"}<br>
        <span class="muted small">lecture_06.py:L345-L353 (cdiv grid), L369-L376 (offsets, mask)</span>`;
    } else if (s.mode === "row" || s.mode === "col") {
      const r = launchGridModel({ mode: s.mode, rows: s.rows, cols: s.cols, block: s.rblock, lane: 0 });
      const len = s.mode === "row" ? s.cols : s.rows, shown = Math.min(r.programs, 6), px = (W - 60) / (r.tiles * s.rblock);
      let body = text(20, 14, `${shown < r.programs ? `first ${shown} of ` : ""}${r.programs} ${s.mode === "row" ? "rows" : "columns"}: one colour = one program; numbers = loop step inside that program`);
      for (let p = 0; p < shown; p++) {
        const y = 24 + p * 22;
        for (let t = 0; t < r.tiles; t++) {
          const x = 20 + t * s.rblock * px, real = Math.min(s.rblock, len - t * s.rblock);
          body += rect(x, y, real * px - 1, 18, PID[p % PID.length], `opacity="${0.35 + 0.5 * (t + 1) / r.tiles}"`) + (real < s.rblock ? rect(x + real * px, y, (s.rblock - real) * px - 1, 18, "#e6e1d6") : "");
          if (s.rblock * px > 18 && r.tiles <= 40) body += `<text x="${(x + 4).toFixed(1)}" y="${(y + 13).toFixed(1)}" font-size="11" fill="#fff">${t + 1}</text>`;
        }
      }
      const small = len <= 32 && s.rblock <= 16;
      fig.innerHTML = svg(W, 34 + shown * 22, body);
      read.innerHTML = `<span class="big">${r.programs} programs</span> (one per ${s.mode === "row" ? "row" : "column"}) · each loops over <span class="big">${r.tiles} tiles</span> of ${s.rblock}${r.masked ? `; the last tile masks ${r.masked} lanes (other = 0.0)` : ""}<br>
        ${r.tiles > 1 ? "the pieces of one row are <b>tiles</b> visited by the same program, not extra blocks: blocks share no memory, so the partial sums must live in one program" : "the whole row fits in one tile: one load, then the reduction"}<br>
        ${small ? `row = [1, 2, …, ${len}] → acc after the loop = [${r.acc.join(", ")}] · tl.sum(acc) = ${r.sum}<br>` : `<span class="muted small">(make the row ≤ 32 and the tile ≤ 16 to see acc for the row [1, 2, …, N])</span><br>`}
        <span class="muted small">lecture_06.py:L448-L457 (grid (M,)), L517-L535 (tile loop, acc, tl.sum); video:1:11:06-1:11:28 ("not blocks … tiles")${s.mode === "col" ? "; column variant: video:1:05:05-1:05:21" : ""}</span>`;
    } else {
      const r = launchGridModel({ mode: "matmul", M: s.M, N: s.N, K: s.K, BM: s.BM, BN: s.BN, BK: s.BK });
      const cs = Math.min(16, 300 / Math.max(r.gridM, r.gridN));
      let body = text(20, 14, `output C: ${r.gridM} × ${r.gridN} tiles, one program each`) + text(360, 14, `inside one program: K loop`);
      for (let i = 0; i < Math.min(r.gridM, 40); i++) for (let j = 0; j < Math.min(r.gridN, 40); j++) body += rect(20 + j * cs, 24 + i * cs, cs - 1, cs - 1, PID[(i + j) % PID.length], `opacity="0.6"`);
      const kw = Math.min(28, 260 / r.tiles);
      for (let t = 0; t < Math.min(r.tiles, 128); t++) body += rect(360 + t * kw, 30, kw - 1, 22, C.accent, `opacity="${0.25 + 0.7 * (t + 1) / r.tiles}"`);
      body += text(360, 72, `${r.tiles} steps: acc += tl.dot(a, b), then ReLU once`);
      fig.innerHTML = svg(W, Math.max(90, 30 + Math.min(r.gridM, 40) * cs), body);
      read.innerHTML = `<span class="big">${r.programs} programs</span> = cdiv(${s.M}, ${s.BM}) × cdiv(${s.N}, ${s.BN}) = ${r.gridM} × ${r.gridN} · each runs a K loop of <span class="big">${r.tiles} iterations</span> = cdiv(${s.K}, ${s.BK})<br>
        the grid counts output tiles; the K reduction stays inside one program, so K never changes the grid<br>
        <span class="muted small">lecture_06.py:L620-L621 (2-D grid), L659-L667 (K loop), L669-L674 (ReLU, store)</span>`;
    }
  };
  root.append(el("div", { class: "widget" }, ctl, fig, read)); build(); draw();
}
MODELS["fixture:lecture_06--launch-grid"] = {
  fn: launchGridModel,
  cases: [
    { args: { mode: "elementwise", n: 10000, block: 1024, grid: "cdiv" }, pick: "programs", expect: 10, from: "lecture_06:triton-program-grid:predict" },
    { args: { mode: "elementwise", n: 10000, block: 1024, grid: "cdiv" }, pick: "lastTrue", expect: 784 },
    { args: { mode: "elementwise", n: 10000, block: 1024, grid: "floor" }, pick: "firstUncomputed", expect: 9216, from: "lecture_06:triton-program-grid:transfer" },
    { args: { mode: "elementwise", n: 5000, block: 512, grid: "floor" }, pick: "uncomputed", expect: 392, from: "lecture_06:triton-program-grid:check" },
    { args: { mode: "elementwise", n: 8192, block: 1024, grid: "floor" }, pick: "uncomputed", expect: 0 },
    { args: { mode: "row", rows: 512, cols: 4096, block: 1024 }, pick: "programs", expect: 512, from: "lecture_06:reduction-stays-in-one-block:predict" },
    { args: { mode: "row", rows: 512, cols: 4096, block: 1024 }, pick: "tiles", expect: 4 },
    { args: { mode: "col", rows: 512, cols: 4096, block: 128 }, pick: "programs", expect: 4096, from: "lecture_06:reduction-stays-in-one-block:transfer" },
    { args: { mode: "col", rows: 512, cols: 4096, block: 128 }, pick: "tiles", expect: 4 },
    { args: { mode: "row", rows: 1, cols: 12, block: 4, lane: 0 }, pick: "accLane", expect: 15, from: "lecture_06:triton-row-sum-kernel:predict" },
    { args: { mode: "row", rows: 1, cols: 12, block: 4 }, pick: "sum", expect: 78 },
    { args: { mode: "row", rows: 1, cols: 11, block: 4, lane: 3 }, pick: "accLane", expect: 12, from: "lecture_06:triton-row-sum-kernel:check" },
    { args: { mode: "row", rows: 1, cols: 10, block: 4 }, pick: "tiles", expect: 3, from: "lecture_06:triton-row-sum-kernel:transfer" },
    { args: { mode: "matmul", M: 1024, N: 1024, K: 1024, BM: 64, BN: 64, BK: 32 }, pick: "programs", expect: 256, from: "lecture_06:triton-matmul-relu-kernel:predict" },
    { args: { mode: "matmul", M: 1024, N: 1024, K: 1024, BM: 64, BN: 64, BK: 32 }, pick: "tiles", expect: 32 },
  ],
};
WIDGETS["fixture:lecture_06--launch-grid"] = launchGrid;

// ==================================================== softmax padding ==
// triton_softmax_kernel (lecture_06.py:L462-L484): BLOCK_SIZE = triton.next_power_of_2(N) (L448),
// grid (M,) (L449-L452); x_row = tl.load(..., mask=col_offsets < num_cols, other=float("-inf")) (L473);
// x_row - tl.max(x_row) over the whole block (L476), exp, tl.sum (L477-L478), masked store (L484).
// The row is num_cols copies of one value v, as in the KP's predict ([5, 5, 5]).
const np2 = n => 2 ** Math.ceil(Math.log2(Math.max(1, n)));
export function softmaxPaddingModel(a) {
  const block = np2(a.cols), padded = block - a.cols, other = a.other === "-inf" ? -Infinity : +a.other;
  const m = padded ? Math.max(a.v, other) : a.v;
  const real = Math.exp(a.v - m), pad = padded ? Math.exp(other - m) : 0;
  const denom = a.cols * real + padded * pad, out = real / denom;
  return { programs: a.rows ?? 1, block, padded, denom, padShare: padded * pad / denom, out, correct: 1 / a.cols, relErr: out * a.cols - 1 };
}
function softmaxPadding(root) {
  const s = { cols: 6, v: 2, other: "-inf", rows: 64 };
  const ctl = el("div", { class: "controls" }), fig = el("div"), read = el("div", { class: "readout" });
  const build = () => ctl.replaceChildren(
    slider("num_cols N", 1, 4096, s.cols, 1, v => { s.cols = v; draw(); }),
    slider("every entry of the row = v", -10, 10, s.v, 0.5, v => { s.v = v; draw(); }),
    slider("rows M", 1, 4096, s.rows, 1, v => { s.rows = v; draw(); }),
    buttons([["-inf", "other = -inf (lecture)"], ["0", "other = 0.0"]], s.other, v => { s.other = v; build(); draw(); }));
  const draw = () => {
    const r = softmaxPaddingModel(s), W = 640, lanes = Math.min(r.block, 64), px = (W - 40) / lanes;
    let body = text(20, 14, `one program's block: ${r.block} lanes${r.block > 64 ? " (drawn at 64 for legibility, proportions kept)" : ""}`);
    const realLanes = Math.round(lanes * s.cols / r.block);
    for (let i = 0; i < lanes; i++) {
      const live = i < realLanes;
      body += rect(20 + i * px, 22, px - 1, 30, live ? C.accent : "#e6e1d6", live ? `opacity="0.75"` : "");
      if (px > 26) body += `<text x="${(20 + i * px + 4).toFixed(1)}" y="42" font-size="11" fill="${live ? "#fff" : C.muted}">${live ? fmt(s.v, 1) : s.other}</text>`;
    }
    const bw = W - 40, realW = bw * (1 - r.padShare);
    body += text(20, 72, "softmax denominator, by where it comes from");
    body += rect(20, 78, realW, 18, C.accent, `opacity="0.75"`) + rect(20 + realW, 78, bw - realW, 18, C.accent2);
    body += text(20, 112, `real entries ${(100 * (1 - r.padShare)).toFixed(1)}%`) + (r.padShare > 0 ? text(Math.min(W - 170, 20 + realW + 4), 112, `padding ${(100 * r.padShare).toFixed(r.padShare < 0.01 ? 3 : 1)}%`, `fill="${C.accent2}"`) : "");
    fig.innerHTML = svg(W, 120, body);
    const ok = r.padShare === 0;
    read.innerHTML = `${r.programs} programs (one per row) · BLOCK_SIZE = next_power_of_2(${s.cols}) = <b>${r.block}</b> · <b>${r.padded}</b> padded lanes loaded as other = ${s.other}<br>
      <span class="big">each output = ${r.out.toPrecision(4)}</span> vs correct 1/${s.cols} = ${r.correct.toPrecision(4)} · ${ok ? `<b style="color:var(--ok)">padding invisible</b>: exp(−inf − max) = 0 adds nothing` : `<b style="color:var(--accent2)">padding corrupts the row</b>: each padded lane adds exp(${s.other} − max) to the denominator (outputs ${(100 * Math.abs(r.relErr)).toFixed(2)}% too small)`}<br>
      <span class="muted small">lecture_06.py:L448-L452 (grid, BLOCK_SIZE), L473 (mask, other = -inf), L476-L479 (max, exp, sum)</span>`;
  };
  root.append(el("div", { class: "widget" }, ctl, fig, read)); build(); draw();
}
MODELS["fixture:lecture_06--softmax-padding"] = {
  fn: softmaxPaddingModel,
  cases: [
    { args: { cols: 3, v: 5, other: "0" }, pick: "out", expect: 0.3326, tol: 0.001, from: "lecture_06:triton-softmax-kernel:predict" },
    { args: { cols: 3, v: 5, other: "-inf" }, pick: "out", expect: 1 / 3, tol: 1e-9 },
    { args: { cols: 1000, v: 0, other: "-inf", rows: 2048 }, pick: "programs", expect: 2048, from: "lecture_06:triton-softmax-kernel:transfer" },
    { args: { cols: 1000, v: 0, other: "-inf" }, pick: "padded", expect: 24 },
    { args: { cols: 3000, v: 0, other: "-inf" }, pick: "padded", expect: 1096, from: "lecture_06:triton-softmax-kernel:check" },
    { args: { cols: 1024, v: 0, other: "0" }, pick: "padded", expect: 0 },
  ],
};
WIDGETS["fixture:lecture_06--softmax-padding"] = softmaxPadding;

// ================================================== matmul time curve ==
// lecture_06.py:L172-L176 benchmarks a @ b for dim in [256 … 8192] and notes "time is roughly constant
// when dimension is small, then cubic scaling". The lecture prints no times, so the model is
// dimensionless: time / fixed cost = 1 + (dim / D0)^3 / speed, with the crossover D0 = 2048 taken from
// the professor's "up until you get up to almost 2,000 dimensional matrices, things are basically a
// constant" (video:25:55-26:06). The cubic term is the 2·dim³ FLOPs (L172 comment, KP statement).
const D0 = 2048;
export function matmulTimeModel(a) {
  const t = d => 1 + (d / D0) ** 3 / (a.speed || 1);
  const work = t(a.dim) - 1;
  return { t: t(a.dim), ratio: t(a.dim) / t(a.dim / 2), workShare: work / t(a.dim), regime: work < 1 ? "fixed-cost" : "FLOP" };
}
function matmulTime(root) {
  const s = { lg: 10, speed: 1 };
  const ctl = el("div", { class: "controls" }), fig = el("div"), read = el("div", { class: "readout" });
  const build = () => ctl.replaceChildren(
    slider("dim (square a @ b)", 7, 14, s.lg, 1, v => { s.lg = v; draw(); }, v => 2 ** v),
    buttons([[1, "GPU as benchmarked"], [2, "arithmetic 2× faster, same fixed cost"]], s.speed, v => { s.speed = v; build(); draw(); }));
  const draw = () => {
    const dim = 2 ** s.lg, r = matmulTimeModel({ dim, speed: s.speed }), W = 640, H = 220;
    const lx = lg => 50 + (lg - 7) / 7 * 560, ly = t => 190 - Math.log2(t) / Math.log2(600) * 170;
    let body = rect(50, 20, 1, 170, C.rule) + rect(50, 190, 560, 1, C.rule) + text(4, 24, "time") + text(560, 212, "dim");
    for (let lg = 7; lg <= 14; lg++) body += text(lx(lg) - 12, 206, String(2 ** lg), `font-size="10"`);
    body += `<line x1="50" x2="610" y1="${ly(1)}" y2="${ly(1)}" stroke="${C.muted}" stroke-dasharray="4 4"/>` + text(56, ly(1) - 4, "fixed per-call cost");
    let path = ""; for (let lg = 7; lg <= 14.001; lg += 0.05) { const d = 2 ** lg; path += `${path ? "L" : "M"}${lx(lg).toFixed(1)},${ly(matmulTimeModel({ dim: d, speed: s.speed }).t).toFixed(1)}`; }
    body += `<path d="${path}" fill="none" stroke="${C.accent}" stroke-width="2.5"/>`;
    for (let lg = 7; lg <= 14; lg++) body += `<circle cx="${lx(lg)}" cy="${ly(matmulTimeModel({ dim: 2 ** lg, speed: s.speed }).t)}" r="3" fill="${C.accent}"/>`;
    body += `<circle cx="${lx(s.lg)}" cy="${ly(r.t)}" r="6" fill="${C.hilite}" stroke="${C.ink}"/>` + `<circle cx="${lx(s.lg - 1)}" cy="${ly(matmulTimeModel({ dim: dim / 2, speed: s.speed }).t)}" r="5" fill="none" stroke="${C.accent2}" stroke-width="2"/>`;
    fig.innerHTML = svg(W, H, body);
    read.innerHTML = `dim ${dim / 2} → ${dim}: FLOPs × 8, <span class="big">time × ${r.ratio.toFixed(2)}</span> · time = ${fmt(r.t, 2)} × the fixed cost<br>
      ${r.regime === "fixed-cost" ? `<b>fixed-cost regime</b>: the arithmetic is only ${(100 * r.workShare).toFixed(0)}% of the time; a faster GPU or 8× the FLOPs barely moves it` : `<b>FLOP regime</b>: the 2·dim³ arithmetic is ${(100 * r.workShare).toFixed(0)}% of the time, so doubling dim heads toward × 8`}<br>
      <span class="muted small">lecture_06.py:L172-L176 (dims 256…8192, "constant … then cubic"); crossover D0 = 2048 from video:25:55-26:06. Times are in units of the fixed cost; the lecture prints none.</span>`;
  };
  root.append(el("div", { class: "widget" }, ctl, fig, read)); build(); draw();
}
MODELS["fixture:lecture_06--matmul-time"] = {
  fn: matmulTimeModel,
  cases: [
    // "roughly 8": the smooth model gives 65/9 = 7.22 at 4096 → 8192, within 10% of the KP's 8
    { args: { dim: 8192 }, pick: "ratio", expect: 8, tol: 0.1, from: "lecture_06:matmul-time-scaling:predict" },
    { args: { dim: 8192 }, pick: "ratio", expect: 65 / 9, tol: 1e-9 },
    { args: { dim: 256 }, pick: "ratio", expect: 1, tol: 0.01 },
    { args: { dim: 2048 }, pick: "t", expect: 2, tol: 1e-9 },
    { args: { dim: 256, speed: 2 }, pick: "t", expect: 1, tol: 0.01 },
    { args: { dim: 16384 }, pick: "ratio", expect: 513 / 65, tol: 1e-9 },
  ],
};
WIDGETS["fixture:lecture_06--matmul-time"] = matmulTime;

// ========================================================== stride map ==
// lecture_06.py:L591-L596: a matrix is linearized in memory, element (row, col) at
// row * stride(0) + col * stride(1); a contiguous R×C tensor has strides (C, 1). x.T is a view with the
// strides swapped (KP tensor-strides transfer; video:1:19:26-1:19:46 "flipped"). Values are torch.arange.
export function strideModel(a) {
  const [s0, s1] = a.transpose ? [1, a.cols] : [a.cols, 1];
  const shape = a.transpose ? [a.cols, a.rows] : [a.rows, a.cols];
  const index = a.i * s0 + a.j * s1;
  return { stride0: s0, stride1: s1, shape0: shape[0], shape1: shape[1], index, value: index };
}
function strideMap(root) {
  const s = { rows: 2, cols: 4, transpose: false, i: 1, j: 2 };
  const ctl = el("div", { class: "controls" }), fig = el("div"), read = el("div", { class: "readout" });
  // the i/j sliders are built once; their ranges follow the shape without rebuilding mid-drag
  const iL = slider("index i (dim 0)", 0, 5, s.i, 1, v => { s.i = v; draw(); });
  const jL = slider("index j (dim 1)", 0, 7, s.j, 1, v => { s.j = v; draw(); });
  const fit = (lab, max, key) => { const inp = lab.querySelector("input"), out = lab.querySelector(".readout"); inp.setAttribute("max", max); s[key] = Math.min(s[key], max); inp.value = s[key]; out.textContent = s[key]; };
  const reshape = () => { const r = strideModel(s); fit(iL, r.shape0 - 1, "i"); fit(jL, r.shape1 - 1, "j"); draw(); };
  const tBtns = el("div");
  const setT = v => { s.transpose = v; tBtns.replaceChildren(buttons([[false, "x (contiguous)"], [true, "x.T (view, no copy)"]], s.transpose, setT)); reshape(); };
  const build = () => { ctl.replaceChildren(
    slider("rows of x", 1, 6, s.rows, 1, v => { s.rows = v; reshape(); }),
    slider("cols of x", 1, 8, s.cols, 1, v => { s.cols = v; reshape(); }),
    tBtns, iL, jL); setT(s.transpose); };
  const draw = () => {
    const r = strideModel(s), n = s.rows * s.cols, cs = 34, W = 640;
    const name = s.transpose ? "x.T" : "x";
    let body = text(20, 14, `${name}: shape (${r.shape0}, ${r.shape1}), the logical view`);
    for (let i = 0; i < r.shape0; i++) for (let j = 0; j < r.shape1; j++) {
      const idx = i * r.stride0 + j * r.stride1, hit = i === s.i && j === s.j, nextRow = i === s.i + 1 && j === s.j, nextCol = i === s.i && j === s.j + 1;
      body += rect(20 + j * cs, 22 + i * cs, cs - 2, cs - 2, hit ? C.hilite : nextRow ? "rgba(184,88,42,.35)" : nextCol ? "rgba(36,102,141,.35)" : C.card, `stroke="${C.rule}"`);
      body += `<text x="${20 + j * cs + 9}" y="${22 + i * cs + 21}" font-size="13" fill="${C.ink}">${idx}</text>`;
    }
    const y0 = 34 + r.shape0 * cs, mw = Math.min(34, (W - 40) / n);
    body += text(20, y0 + 6, "memory (storage), one cell per element, index 0 … " + (n - 1));
    for (let k = 0; k < n; k++) {
      const hit = k === r.index, nr = s.i + 1 < r.shape0 && k === r.index + r.stride0, nc = s.j + 1 < r.shape1 && k === r.index + r.stride1;
      body += rect(20 + k * mw, y0 + 14, mw - 2, 28, hit ? C.hilite : nr ? "rgba(184,88,42,.35)" : nc ? "rgba(36,102,141,.35)" : C.card, `stroke="${C.rule}"`);
      if (mw > 16) body += `<text x="${20 + k * mw + 5}" y="${y0 + 33}" font-size="12" fill="${C.ink}">${k}</text>`;
    }
    body += text(20, y0 + 60, `orange: one step in i jumps ${r.stride0} cell${r.stride0 === 1 ? "" : "s"} · blue: one step in j jumps ${r.stride1} cell${r.stride1 === 1 ? "" : "s"}`);
    fig.innerHTML = svg(W, y0 + 70, body);
    read.innerHTML = `${name}.stride() = <b>(${r.stride0}, ${r.stride1})</b> · ${name}[${s.i}, ${s.j}] → ${s.i}·${r.stride0} + ${s.j}·${r.stride1} = <span class="big">index ${r.index}</span>, value ${r.value} (x = arange)<br>
      ${s.transpose ? "same storage as x, strides swapped: the transpose moves no data, only the pointer arithmetic" : "contiguous: the last dim has stride 1, the row stride is the number of columns"}<br>
      <span class="muted small">lecture_06.py:L591-L596 (linear index from strides), L623-L630 (kernels take strides as arguments)</span>`;
  };
  root.append(el("div", { class: "widget" }, ctl, fig, read)); build();
}
MODELS["fixture:lecture_06--stride-map"] = {
  fn: strideModel,
  cases: [
    { args: { rows: 3, cols: 4, transpose: false, i: 2, j: 1 }, pick: "stride0", expect: 4, from: "lecture_06:tensor-strides:predict" },
    { args: { rows: 3, cols: 4, transpose: false, i: 2, j: 1 }, pick: "index", expect: 9 },
    { args: { rows: 3, cols: 4, transpose: true, i: 1, j: 2 }, pick: "stride0", expect: 1, from: "lecture_06:tensor-strides:transfer" },
    { args: { rows: 3, cols: 4, transpose: true, i: 1, j: 2 }, pick: "index", expect: 9 },
    { args: { rows: 4, cols: 6, transpose: true, i: 3, j: 2 }, pick: "value", expect: 15, from: "lecture_06:tensor-strides:check" },
    { args: { rows: 2, cols: 4, transpose: false, i: 1, j: 2 }, pick: "index", expect: 6 },
  ],
};
WIDGETS["fixture:lecture_06--stride-map"] = strideMap;
