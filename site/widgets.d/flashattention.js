// Widgets for thread flashattention (Dao et al. 2022, arXiv:2205.14135v2). Register as "fixture:<id>" -> (root, notice) => void.
// Each widget has a pure model in MODELS (no DOM) that tools/check_widgets.mjs tests against the KPs' stored answers.
// Sources: papers/flashattention/paper.txt (cited as paper.txt:L<line> plus the paper's own anchor, e.g. Algorithm 1 line 1).
// Counting conventions are the paper's (Algorithm 0 lines, Theorem 2 / Theorem 5 / Proposition 4 proofs, Appendix C) or the
// KP prompts' (marked "KP"); measured numbers are Fig. 2 (left) as printed. No constant here is new.
import { el, fmt, slider } from "../widgets_lib.js";

export const MODELS = {};
export const WIDGETS = {};

const C = { ink: "var(--ink)", muted: "var(--muted)", rule: "var(--rule)", a: "var(--accent)", b: "var(--accent2)", ok: "var(--ok)", hi: "var(--hilite)" };
const rect = (x, y, w, h, fill, extra = "") => `<rect x="${x.toFixed(1)}" y="${y.toFixed(1)}" width="${Math.max(0, w).toFixed(1)}" height="${Math.max(0, h).toFixed(1)}" style="fill:${fill}" ${extra}/>`;
const text = (x, y, s, opts = {}) => `<text x="${x.toFixed(1)}" y="${y.toFixed(1)}" style="fill:${opts.fill || C.ink};font:${opts.weight || ""} ${opts.size || 12}px var(--sans)" text-anchor="${opts.anchor || "start"}">${s}</text>`;
const line = (x1, y1, x2, y2, stroke, extra = "") => `<line x1="${x1.toFixed(1)}" y1="${y1.toFixed(1)}" x2="${x2.toFixed(1)}" y2="${y2.toFixed(1)}" style="stroke:${stroke};${extra}"/>`;
const svg = (w, h, body) => `<svg viewBox="0 0 ${w} ${h}" width="100%" style="max-width:${w}px;border:1px solid var(--rule);border-radius:6px;background:#fff" role="img">${body}</svg>`;
const select = (label, options, value, onChange) => {
  const s = el("select", {}); for (const [v, t] of options) { const o = el("option", { value: v }, t); if (String(v) === String(value)) o.selected = true; s.append(o); }
  s.addEventListener("change", () => onChange(s.value)); return el("label", {}, `${label} `, s);
};
const button = (label, onClick, cls = "") => el("button", { class: cls, onclick: onClick }, label);
const M_ = x => fmt(x / 1e6, 2) + "M";                                        // element counts in millions
const pow2 = v => `${2 ** v}`;

// =================================================================== shared accounting ==
// Algorithm 1 line 1 (paper.txt:L313): B_c = ⌈M/4d⌉, B_r = min(⌈M/4d⌉, d); line 3 (L316-L320): T_r = ⌈N/B_r⌉, T_c = ⌈N/B_c⌉.
// Loop order (L331-L375, lines 5-15): outer j over K/V blocks (line 6 loads K_j, V_j), inner i over Q blocks (line 8 loads
// Q_i, O_i, ℓ_i, m_i; lines 12-13 write O_i, ℓ_i, m_i). So Q-block loads = T_c · T_r.
// Theorem 2 proof (paper.txt:L2602-L2606): K and V loaded once; T_c passes over Q and O, each loading all of Q and O:
//   Θ(Nd + Nd·T_c). The KP prompts (fa-io-complexity predict/check) fix the constant: K,V once = 2Nd, Q and O per pass = 2Nd.
// Algorithm 0 (paper.txt:L202-L222): line 1 reads Q, K and writes S; line 2 reads S, writes P; line 3 reads P, V and writes O:
//   4Nd + 4N² elements (fa-standard-attention-hbm-traffic predict counts the 4N² part).
// SRAM size M in elements: per-SM SRAM bytes / bytes per element (A100: 192 KB per SM, paper.txt:L130; "around 100KB", L404).
function blocks(N, d, M) {
  const raw = Math.ceil(M / (4 * d)), Bc = Math.max(1, raw), Br = Math.min(Bc, d);
  return { Bc, Br, capBinds: raw > d, Tc: Math.ceil(N / Bc), Tr: Math.ceil(N / Br) };
}
function ioCore(N, d, M) {
  const b = blocks(N, d, M);
  const flashKV = 2 * N * d, flashQO = 2 * b.Tc * N * d, flash = flashKV + flashQO;
  // line by line instead of the proof's 2Nd per pass: line 2 writes O, ℓ, m once; line 8 reads Q_i, O_i, ℓ_i, m_i and
  // lines 12-13 write O_i, ℓ_i, m_i on every inner iteration, i.e. 3Nd + 4N per pass.
  const flashLines = (N * d + 2 * N) + flashKV + b.Tc * (3 * N * d + 4 * N);
  const stdNN = 4 * N * N, stdNd = 4 * N * d, std = stdNN + stdNd;
  return {
    ...b, M, qLoads: b.Tc * b.Tr, kvLoads: b.Tc, flashKV, flashQO, flash, flashLines,
    stdNN, stdNd, std, inputs3: 3 * N * d, nnOverInputs: stdNN / (3 * N * d),
    stdOverFlash: std / flash, lowerBound: 4 * N * d,                       // Prop. 3 proof (L2680-L2684): Q, K, V and O have size Nd
    flashOverLB: flash / (4 * N * d), d2OverM: d * d / M,
    sram: { K: b.Bc * d, V: b.Bc * d, Q: b.Br * d, O: b.Br * d, S: b.Br * b.Bc },
    inRange: d <= M && M <= N * d,
  };
}
const elementsOf = a => a.M ?? (a.kb * 1024 / a.bytes);

// ========================================================= 1. IO ledger (forward pass) ==
// Inputs N, d and SRAM as M elements (or kb + bytes per element). Optional ref = another {N, d, M}: ratios to it.
function ioLedger(a) {
  const M = elementsOf(a), o = ioCore(a.N, a.d, M);
  if (a.ref) {
    const r = ioCore(a.ref.N, a.ref.d, elementsOf(a.ref));
    o.flashVsRef = o.flash / r.flash; o.qoVsRef = o.flashQO / r.flashQO; o.stdVsRef = o.std / r.std;
  }
  return o;
}
MODELS["fixture:flashattention--io-ledger"] = {
  fn: ioLedger,
  cases: [
    // fa-forward-loop-structure: predict (d = 128, M = 32768, N = 2048), check (M = 16384, d = 32, N = 2048), transfer (d = 32)
    { args: { N: 2048, d: 128, M: 32768 }, pick: "Bc", expect: 64, from: "flashattention:fa-forward-loop-structure:predict" },
    { args: { N: 2048, d: 128, M: 32768 }, pick: "Tc", expect: 32 },
    { args: { N: 2048, d: 128, M: 32768 }, pick: "Tr", expect: 32 },
    { args: { N: 2048, d: 32, M: 16384 }, pick: "qLoads", expect: 1024, tol: 0.02, from: "flashattention:fa-forward-loop-structure:check" },
    { args: { N: 2048, d: 32, M: 16384 }, pick: "Br", expect: 32 },                 // ... the cap at d binds
    { args: { N: 2048, d: 32, M: 32768 }, pick: "Bc", expect: 256 },                // transfer: ⌈M/4d⌉ = 256, B_r = 32
    { args: { N: 2048, d: 32, M: 32768 }, pick: "Br", expect: 32 },
    // fa-io-complexity: predict (N = 4096, d = 64, M = 32768: 17.3M), check (d 64 -> 128 at fixed M: Q/O term ×4), transfer (2M halves it)
    { args: { N: 4096, d: 64, M: 32768 }, pick: "flash", expect: 17.3e6, tol: 0.01 },
    { args: { N: 4096, d: 64, M: 32768 }, pick: "Tc", expect: 32 },
    { args: { N: 4096, d: 64, M: 32768 }, pick: "Bc", expect: 128 },   // fa-io-complexity predict: B_c = 128, T_c = 32
    { args: { N: 4096, d: 128, M: 32768, ref: { N: 4096, d: 64, M: 32768 } }, pick: "qoVsRef", expect: 4, tol: 0.1, from: "flashattention:fa-io-complexity:check" },
    { args: { N: 4096, d: 64, M: 65536, ref: { N: 4096, d: 64, M: 32768 } }, pick: "qoVsRef", expect: 0.5 },
    { args: { N: 4096, d: 64, M: 65536, ref: { N: 4096, d: 64, M: 32768 } }, pick: "stdVsRef", expect: 1 },  // standard has no M
    { args: { N: 4096, d: 64, M: 32768 }, pick: "stdOverFlash", expect: (67108864 + 1048576) / 17301504 },
    // fa-standard-attention-hbm-traffic: predict (4N² = 67.1M vs 3Nd = 0.79M, ~85×), transfer (double d: total barely moves)
    { args: { N: 4096, d: 64, M: 32768 }, pick: "stdNN", expect: 67.1e6, tol: 0.01 },
    { args: { N: 4096, d: 64, M: 32768 }, pick: "nnOverInputs", expect: 85, tol: 0.01 },
    { args: { N: 4096, d: 128, M: 32768, ref: { N: 4096, d: 64, M: 32768 } }, pick: "stdVsRef", expect: 1.0154, tol: 0.002 },
    // supp-sram-size-m: predict (192 KB fp16, d = 64: M = 98304, B_c = 384), check (192 KB fp32, d = 128: B_c = 96)
    { args: { N: 4096, d: 64, kb: 192, bytes: 2 }, pick: "M", expect: 98304, from: "flashattention:supp-sram-size-m:predict" },
    { args: { N: 4096, d: 64, kb: 192, bytes: 2 }, pick: "Bc", expect: 384 },
    { args: { N: 4096, d: 128, kb: 192, bytes: 4 }, pick: "Bc", expect: 96, tol: 0.05, from: "flashattention:supp-sram-size-m:check" },
    // fa-lower-bound: at M = Nd the count is Θ(Nd): T_c = 4, 10Nd, i.e. 2.5× the 4Nd input/output floor; at M = d it is Θ(N²d)
    { args: { N: 4096, d: 64, M: 4096 * 64 }, pick: "Tc", expect: 4 },
    { args: { N: 4096, d: 64, M: 4096 * 64 }, pick: "flashOverLB", expect: 2.5 },
    { args: { N: 4096, d: 64, M: 64 }, pick: "flash", expect: 2 * 4096 * 64 + 2 * 4096 * 4096 * 64 },
    // regime edge (§3.2, d² ≪ M): with M below ~2d² FlashAttention moves more elements than standard attention
    { args: { N: 4096, d: 64, M: 4096 }, pick: "stdOverFlash", expect: (67108864 + 1048576) / (524288 + 2 * 256 * 4096 * 64) },
  ],
};
WIDGETS["fixture:flashattention--io-ledger"] = (root) => {
  const s = { logN: 10, d: 64, kb: 96, bytes: 2, ref: null };
  const pic = el("div"), read = el("div", { class: "readout" });
  const args = () => ({ N: 2 ** s.logN, d: s.d, kb: s.kb, bytes: s.bytes });
  const draw = () => {
    const a = args(), N = a.N, d = a.d, m = ioLedger({ ...a, ref: s.ref });
    const M = m.M;
    // ---- plot: HBM accesses vs M on log-log axes, M from d to Nd (Theorem 2's range)
    const W = 640, px = 64, py = 28, pw = 420, ph = 190;
    const lo = Math.log10(d), hi = Math.log10(N * d);
    const ys = [m.lowerBound, m.std, ioCore(N, d, d).flash];
    const ylo = Math.floor(Math.log10(Math.min(...ys)) - 0.2), yhi = Math.ceil(Math.log10(Math.max(...ys)) + 0.1);
    const X = lm => px + pw * (lm - lo) / (hi - lo), Y = v => py + ph * (yhi - Math.log10(v)) / (yhi - ylo);
    let b = rect(px, py, pw, ph, "#fafafa");
    for (let e = ylo; e <= yhi; e++) b += line(px, Y(10 ** e), px + pw, Y(10 ** e), C.rule, "stroke-width:0.5") + text(px - 6, Y(10 ** e) + 4, `1e${e}`, { fill: C.muted, size: 10, anchor: "end" });
    for (let e = Math.ceil(lo); e <= Math.floor(hi); e++) b += text(X(e), py + ph + 14, `1e${e}`, { fill: C.muted, size: 10, anchor: "middle" });
    b += text(X(lo), py + ph + 26, "M = d", { fill: C.muted, size: 10, anchor: "start" }) + text(X(hi), py + ph + 26, "M = Nd", { fill: C.muted, size: 10, anchor: "end" });
    let path = "";
    for (let k = 0; k <= 160; k++) { const lm = lo + (hi - lo) * k / 160; path += `${path ? "L" : "M"}${X(lm).toFixed(1)},${Y(ioCore(N, d, 10 ** lm).flash).toFixed(1)}`; }
    b += line(px, Y(m.std), px + pw, Y(m.std), C.b, "stroke-width:2");
    b += line(px, Y(m.lowerBound), px + pw, Y(m.lowerBound), C.muted, "stroke-width:1.5;stroke-dasharray:5 4");
    b += `<path d="${path}" style="fill:none;stroke:${C.a};stroke-width:2"/>`;
    b += text(px + pw + 6, Y(m.std) + 4, "standard", { fill: C.b, size: 11 });
    b += text(px + pw + 6, Y(m.lowerBound) + 4, "floor", { fill: C.muted, size: 11 });
    b += text(px - 44, 14, "— FlashAttention 2Nd + 2·T_c·Nd", { fill: C.a, size: 11 }) + text(px + 170, 14, "— standard 4N² + 4Nd", { fill: C.b, size: 11 })
       + text(px + 310, 14, "- - floor 4Nd: read Q, K, V, write O", { fill: C.muted, size: 11 });
    if (s.ref) {
      const rM = elementsOf(s.ref);
      if (s.ref.N === N && s.ref.d === d && rM >= d && rM <= N * d) b += `<circle cx="${X(Math.log10(rM)).toFixed(1)}" cy="${Y(ioCore(N, d, rM).flash).toFixed(1)}" r="5" style="fill:none;stroke:${C.ink};stroke-width:1.5"/>`;
    }
    const cm = Math.min(hi, Math.max(lo, Math.log10(M)));
    b += line(X(cm), py, X(cm), py + ph, C.ink, "stroke-width:1;stroke-dasharray:2 3");
    b += `<circle cx="${X(cm).toFixed(1)}" cy="${Y(m.flash).toFixed(1)}" r="5" style="fill:${C.hi};stroke:${C.ink}"/>`;
    b += text(px + pw / 2, py + ph + 40, "SRAM size M in elements (log scale) · y: HBM accesses in elements (log scale)", { fill: C.muted, size: 11, anchor: "middle" });
    // ---- SRAM occupancy bar: what Algorithm 1 keeps on chip at once
    const by = py + ph + 58, bw = 560, used = m.sram.K + m.sram.V + m.sram.Q + m.sram.O + m.sram.S, scale = Math.max(M, used);
    let x = px - 44;
    b += text(px - 44, by - 6, `on chip at once (line 9): K_j, V_j are B_c×d = ${m.Bc}×${d}; Q_i, O_i are B_r×d = ${m.Br}×${d}; S_ij is B_r×B_c = ${m.Br}×${m.Bc}`, { fill: C.muted, size: 11 });
    for (const [k, col, op] of [["K", C.b, 0.8], ["V", C.b, 0.5], ["Q", C.a, 0.8], ["O", C.a, 0.5], ["S", C.hi, 0.9]]) {
      const w = bw * m.sram[k] / scale; b += rect(x, by, w, 22, col, `opacity="${op}"`);
      if (w > 22) b += text(x + w / 2, by + 15, k === "S" ? "S_ij" : `${k}_${k === "K" || k === "V" ? "j" : "i"}`, { fill: "#fff", size: 11, anchor: "middle" });
      x += w;
    }
    const mx = px - 44 + bw * M / scale;
    b += line(mx, by - 3, mx, by + 25, C.ink, "stroke-width:1.5;stroke-dasharray:4 3") + text(mx, by + 38, `M = ${fmt(M, 0)}`, { size: 11, anchor: mx > px + bw - 90 ? "end" : "middle" });
    pic.innerHTML = svg(W, by + 46, b);
    // ---- readout
    const range = M < d ? `<b style="color:var(--bad)">M < d: below Theorem 2's range (not one row of K fits)</b>` : M > N * d ? `<b style="color:var(--bad)">M > Nd: all of K fits; above Theorem 2's range</b>` : "inside Theorem 2's range d ≤ M ≤ Nd";
    const cap = m.capBinds ? `<b>the cap at d binds</b>: ⌈M/4d⌉ = ${Math.ceil(M / (4 * d))} > d, so B_r = d = ${m.Br} and S_ij is ${m.Br}×${m.Bc}, not square` : `the cap at d does not bind: B_r = B_c = ${m.Bc}`;
    const versus = m.stdOverFlash >= 1 ? `FlashAttention moves <b>${fmt(m.stdOverFlash, 2)}× fewer</b> elements` : `<b style="color:var(--bad)">FlashAttention moves ${fmt(1 / m.stdOverFlash, 2)}× more</b> (M is too small: d²/M = ${fmt(m.d2OverM, 3)} is not ≪ 1)`;
    const pin = s.ref ? `<br>vs pinned (N = ${s.ref.N}, d = ${s.ref.d}, M = ${fmt(elementsOf(s.ref), 0)}): Flash Q/O term <b>×${fmt(m.qoVsRef, 3)}</b> · Flash total ×${fmt(m.flashVsRef, 3)} · standard ×${fmt(m.stdVsRef, 4)}` : "";
    read.innerHTML = `M = ${s.kb} KB × 1024 / ${s.bytes} B = <b>${fmt(M, 0)} elements</b> · ${range}<br>
      line 1: B_c = ⌈M/4d⌉ = <b>${m.Bc}</b>, B_r = min(⌈M/4d⌉, d) = <b>${m.Br}</b> · ${cap}<br>
      line 3: T_c = ⌈N/B_c⌉ = <b>${m.Tc}</b> K/V blocks (each loaded once, line 6) · T_r = ⌈N/B_r⌉ = <b>${m.Tr}</b> Q blocks, each loaded T_c times (line 8): <b>${fmt(m.qLoads, 0)}</b> Q-block loads<br>
      <pre>standard (Alg. 0)   S, P written + read 4N² = ${M_(m.stdNN).padStart(9)}   Q, K, V, O 4Nd = ${M_(m.stdNd).padStart(8)}   total ${M_(m.std)}
FlashAttention      Q, O × T_c  2·T_c·Nd = ${M_(m.flashQO).padStart(9)}   K, V once  2Nd = ${M_(m.flashKV).padStart(8)}   total ${M_(m.flash)}</pre>
      ${versus} · Flash is ${fmt(m.flashOverLB, 2)}× the 4Nd floor${pin}<br>
      <span class="muted small">Counting every line of Alg. 1 (Q_i, O_i, ℓ_i, m_i read and O_i, ℓ_i, m_i written per pass, plus line 2's initial O, ℓ, m) gives ${M_(m.flashLines)}: same Θ(N²d²/M), a larger constant. The plot uses the proof's count.<br>
      provenance: fixture:flashattention--io-ledger · paper.txt:L313 (Alg. 1 line 1), L331-L375 (loop order, lines 5-15), L202-L222 (Alg. 0), L396-L404 (Theorem 2; d² ≪ M for d 64-128, M around 100KB), L2602-L2656 (proof: K, V once, T_c passes over Q and O; B_rB_c = O(M) at L2633), L2663-L2686 (Prop. 3 proof, Ω(Nd) at M = Θ(Nd)), L130 (A100: 192 KB SRAM per SM). Constants 2Nd and 2·T_c·Nd are the KP prompts' (fa-io-complexity).</span>`;
  };
  root.append(el("div", { class: "widget" }, el("div", { class: "controls" },
    slider("sequence N", 8, 16, s.logN, 1, v => { s.logN = v; draw(); }, pow2),
    slider("head dim d", 16, 256, s.d, 16, v => { s.d = v; draw(); }),
    slider("SRAM per SM (KB)", 4, 1024, s.kb, 4, v => { s.kb = v; draw(); }, v => `${v} KB${v === 192 ? " (A100)" : v > 192 ? " (hypothetical)" : ""}`),
    select("element", [[2, "fp16 / bf16 (2 B)"], [4, "fp32 (4 B)"]], s.bytes, v => { s.bytes = +v; draw(); }),
    el("div", { style: "margin-top:8px" }, button("pin this as reference", () => { const a = args(); s.ref = { N: a.N, d: a.d, M: elementsOf(a) }; draw(); }), " ",
      button("clear pin", () => { s.ref = null; draw(); }, "ghost")),
    read), pic)); draw();
};

// ======================================================= 2. block-sparse IO vs N ==
// Proposition 4 (paper.txt:L513-L519): Θ(Nd + N²d²M⁻¹s), s = fraction of nonzero blocks. Proof (L2848-L2856): only
// nonzero blocks are loaded, so the pass term scales by s, but O must still be written. Schedules (L522-L526):
// s = N^(−1/2) gives Θ(N√N), s = N^(−1)·log N gives Θ(N log N). Constants follow ioCore; the unscaled Θ(Nd) term is the
// KP's 2Nd (fa-block-sparse predict: "plus the unchanged ≈0.5M for K, V (and O)").
const sOf = (schedule, N, s) => Math.min(1, schedule === "dense" ? 1 : schedule === "fixed" ? s : schedule === "sqrt" ? N ** -0.5 : Math.log2(N) / N);
function blockSparse({ N, d, M, schedule = "fixed", s = 1, N0 }) {
  const c = ioCore(N, d, M), sv = sOf(schedule, N, s);
  const sparseTerm = sv * c.flashQO, total = c.flashKV + sparseTerm;
  const o = { s: sv, dense: c.flash, denseTerm: c.flashQO, sparseTerm, total, ioSaving: c.flash / total, Tc: c.Tc, Tr: c.Tr, keptBlocks: sv * c.Tc * c.Tr };
  if (N0) {
    const c0 = ioCore(N0, d, M), s0 = sOf(schedule, N0, s);
    o.growth = sparseTerm / (s0 * c0.flashQO); o.denseGrowth = c.flashQO / c0.flashQO;
  }
  return o;
}
MODELS["fixture:flashattention--block-sparse"] = {
  fn: blockSparse,
  cases: [
    { args: { N: 4096, d: 64, M: 32768, schedule: "fixed", s: 0.25 }, pick: "total", expect: 4.7e6, tol: 0.02, from: "flashattention:fa-block-sparse:predict" },
    { args: { N: 4096, d: 64, M: 32768, schedule: "fixed", s: 0.25 }, pick: "ioSaving", expect: 17301504 / 4718592 },   // < 4: the Nd term does not shrink
    { args: { N: 16384, d: 64, M: 32768, schedule: "sqrt", N0: 1024 }, pick: "growth", expect: 64, from: "flashattention:fa-block-sparse:transfer" },
    { args: { N: 16384, d: 64, M: 32768, schedule: "sqrt", N0: 1024 }, pick: "denseGrowth", expect: 256 },
    { args: { N: 4096, d: 64, M: 32768, schedule: "log", N0: 1024 }, pick: "growth", expect: 4.8, tol: 0.05, from: "flashattention:fa-block-sparse:check" },
    { args: { N: 4096, d: 64, M: 32768, schedule: "log", N0: 1024 }, pick: "denseGrowth", expect: 16 },
    { args: { N: 4096, d: 64, M: 32768, schedule: "dense" }, pick: "total", expect: 17301504 },                       // s = 1 is dense FlashAttention
    { args: { N: 4096, d: 64, M: 32768, schedule: "sqrt" }, pick: "s", expect: 1 / 64 },
  ],
};
WIDGETS["fixture:flashattention--block-sparse"] = (root) => {
  const s = { logN: 11, logN0: 10, schedule: "fixed", s: 0.5, d: 64, logM: 15 };
  const pic = el("div"), read = el("div", { class: "readout" });
  const names = { dense: "dense (s = 1)", fixed: "fixed s", sqrt: "s = N^(−1/2)", log: "s = log₂N / N" };
  const draw = () => {
    const N = 2 ** s.logN, N0 = 2 ** s.logN0, M = 2 ** s.logM, a = { N, d: s.d, M, schedule: s.schedule, s: s.s, N0 }, m = blockSparse(a);
    const W = 640, px = 64, py = 20, pw = 440, ph = 200, lo = 8, hi = 17;
    const at = (n, sch) => blockSparse({ N: 2 ** n, d: s.d, M, schedule: sch, s: s.s });
    const vals = []; for (let n = lo; n <= hi; n += 0.25) vals.push(at(n, "dense").total, at(n, s.schedule).total);
    const ylo = Math.floor(Math.log10(Math.min(...vals))), yhi = Math.ceil(Math.log10(Math.max(...vals)));
    const X = n => px + pw * (n - lo) / (hi - lo), Y = v => py + ph * (yhi - Math.log10(v)) / (yhi - ylo);
    let b = rect(px, py, pw, ph, "#fafafa");
    for (let e = ylo; e <= yhi; e++) b += line(px, Y(10 ** e), px + pw, Y(10 ** e), C.rule, "stroke-width:0.5") + text(px - 6, Y(10 ** e) + 4, `1e${e}`, { fill: C.muted, size: 10, anchor: "end" });
    for (let n = lo; n <= hi; n += 2) b += text(X(n), py + ph + 14, `${2 ** n}`, { fill: C.muted, size: 10, anchor: "middle" });
    const curve = (sch, col, dash) => { let p = ""; for (let n = lo; n <= hi + 1e-9; n += 0.125) p += `${p ? "L" : "M"}${X(n).toFixed(1)},${Y(at(n, sch).total).toFixed(1)}`; return `<path d="${p}" style="fill:none;stroke:${col};stroke-width:2;${dash ? "stroke-dasharray:5 4" : ""}"/>`; };
    b += curve("dense", C.a, false);
    if (s.schedule !== "dense") b += curve(s.schedule, C.b, false);
    b += text(px + pw + 6, Y(at(hi, "dense").total) + 4, "dense", { fill: C.a, size: 11 });
    if (s.schedule !== "dense") b += text(px + pw + 6, Y(at(hi, s.schedule).total) + 4, names[s.schedule], { fill: C.b, size: 11 });
    b += line(X(s.logN0), py, X(s.logN0), py + ph, C.muted, "stroke-dasharray:2 3") + text(X(s.logN0) + 3, py + ph - 6, "N₀", { fill: C.muted, size: 11 });
    b += `<circle cx="${X(s.logN).toFixed(1)}" cy="${Y(m.total).toFixed(1)}" r="5" style="fill:${C.hi};stroke:${C.ink}"/>`;
    b += `<circle cx="${X(s.logN).toFixed(1)}" cy="${Y(m.dense).toFixed(1)}" r="4" style="fill:none;stroke:${C.a};stroke-width:1.5"/>`;
    b += text(px + pw / 2, py + ph + 30, "sequence length N (log scale) · y: HBM accesses in elements (log scale), d and M fixed", { fill: C.muted, size: 11, anchor: "middle" });
    pic.innerHTML = svg(W, py + ph + 40, b);
    read.innerHTML = `${names[s.schedule]}: s = <b>${fmt(m.s, 4)}</b> of the T_r × T_c = ${m.Tr} × ${m.Tc} blocks kept (≈ ${fmt(m.keptBlocks, 1)} nonzero blocks)<br>
      <pre>dense FlashAttention   2Nd + 2·T_c·Nd     = ${M_(m.dense)}
block-sparse           2Nd + s·2·T_c·Nd   = ${M_(m.total)}   (only the pass term is scaled by s)</pre>
      IO saving dense / sparse = <span class="big">${fmt(m.ioSaving, 3)}×</span> ${m.s < 1 ? `(not 1/s = ${fmt(1 / m.s, 3)}×: the 2Nd term does not shrink)` : ""}<br>
      from N₀ = ${N0} to N = ${N}: sparse term N²d²M⁻¹·s grows <b>×${fmt(m.growth, 4)}</b> · dense term grows ×${fmt(m.denseGrowth, 4)}<br>
      <span class="muted small">s is treated as a continuous fraction; a real mask keeps a whole number of blocks, and which blocks (the paper uses a fixed butterfly pattern, L527) does not enter the count.
      Block-sparse attention is an approximation: it changes the function, unlike dense FlashAttention.<br>
      provenance: fixture:flashattention--block-sparse · paper.txt:L513-L519 (Prop. 4), L522-L526 (s = N^(−1/2) → Θ(N√N); s = N^(−1)log N → Θ(N log N)), L2848-L2856 (proof: scaled by s, O still written), L313 (block sizes); the unscaled 2Nd is the KP's convention (fa-block-sparse predict).</span>`;
  };
  root.append(el("div", { class: "widget" }, el("div", { class: "controls" },
    select("mask density", Object.entries(names), s.schedule, v => { s.schedule = v; draw(); }),
    slider("fixed s", 0.05, 1, s.s, 0.05, v => { s.s = v; draw(); }, v => v.toFixed(2)),
    slider("sequence N", 8, 17, s.logN, 1, v => { s.logN = v; draw(); }, pow2),
    slider("reference N₀", 8, 17, s.logN0, 1, v => { s.logN0 = v; draw(); }, pow2),
    slider("head dim d", 16, 256, s.d, 16, v => { s.d = v; draw(); }),
    slider("SRAM M (elements)", 12, 18, s.logM, 1, v => { s.logM = v; draw(); }, pow2),
    read), pic)); draw();
};

// ========================================== 3. backward: store P, or recompute it ==
// §3.1 Recomputation (paper.txt:L286-L296): store O and (m, ℓ), recompute S and P from blocks of Q, K, V in SRAM; "even with
// more FLOPs, our recomputation speeds up the backward pass due to reduced HBM accesses (Fig. 2)".
// Saved state: standard keeps P (N×N); FlashAttention keeps O (N×d), ℓ and m (N each) (Algorithm 4 inputs, L2264-L2272).
// D_i: textbook Σ_j P_ij dP_ij runs over N and needs P_i:, dP_i: on chip ("they might not fit into SRAM", L2253-L2258);
// Eq. 4 (L2018) / Alg. 4 line 19 (L2361): D_i = rowsum(dO_i ∘ O_i), d products.
// Backward HBM: Algorithm 3 (L2206-L2240) line by line: L1 P, dO in, dV out; L2 dO, V in, dP out; L3 P, dP in, dS out;
// L4 dS, K in, dQ out; L5 dS, Q in, dK out = 7N² + 8Nd. Theorem 5 proof (L2706-L2716): K, V once, dK, dV written once,
// T_c passes over Q, O, dO and T_c read/writes of dQ = 4Nd + 5·T_c·Nd.
// Measured (Fig. 2 left, L450-L475; GPT-2 medium fwd + bwd, A100): 66.6 vs 75.2 GFLOPs, 40.3 vs 4.4 GB, 41.7 vs 7.3 ms.
const FIG2 = { gflops: [66.6, 75.2], gb: [40.3, 4.4], ms: [41.7, 7.3] };
function savedState({ N, d, M }) {
  const b = blocks(N, d, M);
  const stdSaved = N * N, flashSaved = N * d + 2 * N;
  const bwdStd = 7 * N * N + 8 * N * d, bwdFlash = 4 * N * d + 5 * b.Tc * N * d;
  return {
    stdSaved, flashSaved, savedRatio: stdSaved / flashSaved, textbookTerms: N, eq4Terms: d, textbookRowOnChip: 2 * N,
    textbookRowFits: 2 * N <= M, textbookBlockOnChip: 2 * b.Br * N, Tc: b.Tc, Br: b.Br, bwdStd, bwdFlash, bwdRatio: bwdStd / bwdFlash,
    fig2FlopRatio: FIG2.gflops[1] / FIG2.gflops[0], fig2ByteRatio: FIG2.gb[0] / FIG2.gb[1], fig2Speedup: FIG2.ms[0] / FIG2.ms[1],
  };
}
MODELS["fixture:flashattention--saved-state"] = {
  fn: savedState,
  cases: [
    { args: { N: 16384, d: 64, M: 49152 }, pick: "stdSaved", expect: 268435456 },          // fa-backward-recomputation predict: 268M
    { args: { N: 16384, d: 64, M: 49152 }, pick: "flashSaved", expect: 1081344 },          // ... 1.08M
    { args: { N: 16384, d: 64, M: 49152 }, pick: "savedRatio", expect: 248, tol: 0.01 },   // ... ≈ 248×
    { args: { N: 32768, d: 96, M: 49152 }, pick: "eq4Terms", expect: 96, tol: 0.02, from: "flashattention:fa-backward-recomputation:check" },
    { args: { N: 32768, d: 96, M: 49152 }, pick: "textbookTerms", expect: 32768 },          // the length-N reduction it replaces
    { args: { N: 16384, d: 64, M: 32768 }, pick: "textbookRowOnChip", expect: 32768 },     // transfer: 2N reaches M at N = M/2
    { args: { N: 4096, d: 64, M: 32768 }, pick: "bwdRatio", expect: (7 * 4096 * 4096 + 8 * 4096 * 64) / (4 * 4096 * 64 + 5 * 32 * 4096 * 64) },
    { args: { N: 1024, d: 64, M: 49152 }, pick: "fig2Speedup", expect: 5.71, tol: 0.01, from: "flashattention:fa-io-aware-framing:predict" },   // fa-io-aware-framing predict: ≈ 5.7×
    { args: { N: 4096, d: 256, M: 8192 }, pick: "bwdRatio", expect: (7 * 4096 * 4096 + 8 * 4096 * 256) / (4 * 4096 * 256 + 5 * 512 * 4096 * 256) },  // regime edge: < 1
    { args: { N: 1024, d: 64, M: 49152 }, pick: "fig2FlopRatio", expect: 1.129, tol: 0.01 }, // +13% FLOPs from recomputation
  ],
};
WIDGETS["fixture:flashattention--saved-state"] = (root) => {
  const s = { logN: 12, d: 64, kb: 96 };   // not the predict case (N = 16384)
  const pic = el("div"), read = el("div", { class: "readout" });
  const draw = () => {
    const N = 2 ** s.logN, M = s.kb * 1024 / 2, m = savedState({ N, d: s.d, M });
    const W = 640, x0 = 200, span = 380;
    const bars = (y, title, rows, unit) => {
      const mx = Math.max(...rows.map(r => r[1]));
      let out = text(10, y, title, { fill: C.muted, size: 11 });
      rows.forEach(([name, v, col, note], i) => {
        const yy = y + 8 + i * 26, w = Math.max(1.5, span * v / mx);
        out += text(10, yy + 14, name, { size: 12 }) + rect(x0, yy, w, 18, col) + text(Math.min(x0 + w + 6, x0 + span - 4), yy + 14, `${note ?? fmt(v, 3)}${unit}`, { size: 11, anchor: x0 + w + 6 > x0 + span - 60 ? "end" : "start", fill: x0 + w + 6 > x0 + span - 60 ? "#fff" : C.ink });
      });
      return out;
    };
    let b = bars(16, `saved for the backward pass, per head per layer (elements, linear scale) · N = ${N}, d = ${s.d}`, [
      ["standard: P (N×N)", m.stdSaved, C.b, M_(m.stdSaved)], ["Flash: O, ℓ, m (Nd + 2N)", m.flashSaved, C.a, M_(m.flashSaved)]], "");
    b += bars(90, `backward HBM accesses (elements): Alg. 3 line by line vs Theorem 5 proof, M = ${fmt(M, 0)}`, [
      ["standard (Alg. 3)", m.bwdStd, C.b, M_(m.bwdStd)], ["Flash (Alg. 4)", m.bwdFlash, C.a, M_(m.bwdFlash)]], "");
    b += bars(164, "measured, GPT-2 medium attention fwd + bwd on A100 (Fig. 2 left, as printed)", [
      ["GFLOPs  std / Flash", FIG2.gflops[0], C.b, "66.6"], ["", FIG2.gflops[1], C.a, "75.2  (+13%: recomputation)"]], "");
    b += bars(232, "", [["HBM GB  std / Flash", FIG2.gb[0], C.b, "40.3"], ["", FIG2.gb[1], C.a, "4.4  (÷9.2)"]], "");
    b += bars(290, "", [["runtime ms  std / Flash", FIG2.ms[0], C.b, "41.7"], ["", FIG2.ms[1], C.a, "7.3  (÷5.7)"]], "");
    pic.innerHTML = svg(W, 350, b);
    const fits = m.textbookRowFits;
    read.innerHTML = `saved state: P has N² = ${M_(m.stdSaved)} elements; O, ℓ, m have Nd + 2N = ${M_(m.flashSaved)} · ratio <span class="big">${fmt(m.savedRatio, 1)}×</span> ≈ N/(d + 2)<br>
      softmax gradient D_i for one row: textbook Σ_j P_ij dP_ij sums <b>${fmt(m.textbookTerms, 0)}</b> products and needs the rows P_i:, dP_i: (2N = ${fmt(m.textbookRowOnChip, 0)} elements; a B_r = ${m.Br} row block needs ${fmt(m.textbookBlockOnChip, 0)}) on chip:
      <b style="color:${fits ? "var(--ok)" : "var(--bad)"}">${fits ? "one row still fits in M" : "one row alone exceeds M"}</b> · Eq. 4, D_i = do_iᵀ o_i, sums <b>${m.eq4Terms}</b> products of two saved length-d rows<br>
      backward HBM: standard ${M_(m.bwdStd)} vs Flash ${M_(m.bwdFlash)} (T_c = ${m.Tc}) · ${m.bwdRatio >= 1 ? `<b>${fmt(m.bwdRatio, 2)}× fewer</b> accesses, although Flash recomputes S_ij and P_ij for every block` : `<b style="color:var(--bad)">Flash moves ${fmt(1 / m.bwdRatio, 2)}× more</b>: M is too small for this d (d²/M = ${fmt(s.d * s.d / M, 3)} is not ≪ 1), so the T_c passes cost more than the N² it avoids`}<br>
      <span class="muted small">Element counts ignore dropout (Flash stores the RNG state instead of the N×N mask, L2244-L2249). The Fig. 2 bars are one measured configuration (N = 1024, d = 64) and do not move with the sliders.<br>
      provenance: fixture:flashattention--saved-state · paper.txt:L286-L296 (§3.1 recomputation), L2206-L2240 (Alg. 3), L2253-L2258 (B.4 observation 2), L2018 (Eq. 4), L2361 (Alg. 4 line 19), L2706-L2716 (Theorem 5 proof), L450-L475 (Fig. 2 left).</span>`;
  };
  root.append(el("div", { class: "widget" }, el("div", { class: "controls" },
    slider("sequence N", 9, 17, s.logN, 1, v => { s.logN = v; draw(); }, pow2),
    slider("head dim d", 16, 256, s.d, 16, v => { s.d = v; draw(); }),
    slider("SRAM per SM (KB, fp16)", 16, 192, s.kb, 16, v => { s.kb = v; draw(); }, v => `${v} KB → M = ${v * 512}`),
    read), pic)); draw();
};
