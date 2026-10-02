// Widgets for thread grpo (DeepSeekMath, Shao et al. 2024, arXiv:2402.03300v3). Register as "fixture:<id>" -> (root, notice) => void.
// Each widget has a pure model in MODELS (no DOM) that tools/check_widgets.mjs tests against the KPs' stored answers.
// Source: papers/grpo/paper.txt (cited as paper.txt:L<n>, with the equation or section as printed). The paper prints
// no clip ε; where a widget needs one its default 0.2 is the band [0.8, 1.2] that this thread's prompts use (marked "prompts").
import { el, fmt, slider } from "../widgets_lib.js";

export const MODELS = {};
export const WIDGETS = {};

const C = { ink: "var(--ink)", muted: "var(--muted)", rule: "var(--rule)", a: "var(--accent)", b: "var(--accent2)", ok: "var(--ok)", hi: "var(--hilite)" };
const rect = (x, y, w, h, fill, extra = "") => `<rect x="${x.toFixed(1)}" y="${y.toFixed(1)}" width="${Math.max(0, w).toFixed(1)}" height="${Math.max(0, h).toFixed(1)}" style="fill:${fill}" ${extra}/>`;
const text = (x, y, s, o = {}) => `<text x="${x.toFixed(1)}" y="${y.toFixed(1)}" style="fill:${o.fill || C.ink};font:${o.weight || ""} ${o.size || 12}px var(--sans)" text-anchor="${o.anchor || "start"}">${s}</text>`;
const line = (x1, y1, x2, y2, stroke, extra = "") => `<line x1="${x1.toFixed(1)}" y1="${y1.toFixed(1)}" x2="${x2.toFixed(1)}" y2="${y2.toFixed(1)}" style="stroke:${stroke};stroke-width:1.5" ${extra}/>`;
const path = (pts, stroke, extra = "") => `<path d="${pts.map((p, i) => `${i ? "L" : "M"}${p[0].toFixed(1)},${p[1].toFixed(1)}`).join("")}" style="fill:none;stroke:${stroke};stroke-width:2" ${extra}/>`;
const svg = (w, h, body) => `<svg viewBox="0 0 ${w} ${h}" width="100%" style="max-width:${w}px;border:1px solid var(--rule);border-radius:6px;background:#fff" role="img">${body}</svg>`;
const check = (label, value, onChange) => { const i = el("input", { type: "checkbox" }); i.checked = value; i.addEventListener("change", () => onChange(i.checked)); return el("label", {}, i, ` ${label}`); };
const sgn = (x, d = 3) => (x > 0 ? "+" : x < 0 ? "−" : "") + fmt(Math.abs(x), d);
const EPS0 = 1e-12;

// ======================================================== 1. advantage map ==
// Outcome supervision, §4.1.2 (paper.txt:L933): Â_{i,t} = r̃_i = (r_i − mean(r)) / std(r) for every token t of o_i.
// Process supervision, §4.1.3 (paper.txt:L937-L938): all step rewards of all G outputs are pooled into R and normalized by
//   mean(R), std(R); a token's advantage is Σ_{index(j) ≥ t} r̃_i^{index(j)} (its own step and every later step).
// Online RFT's coefficient is I(o) ∈ {0, 1} (Eq. 10/11, Table 10; paper.txt:L1445 "does not penalize incorrect responses").
// The paper does not say which std; ddof 0 (divide by G) is this thread's convention, ddof 1 is FeynRL's (rollouts/base.py:L135-L136).
// std = 0 (a uniform group) leaves Eq. §4.1.2 at 0/0; implementations add a small ε (FeynRL 1e-8), so every Â is 0.
function stats(xs, ddof) {
  const n = xs.length, mean = xs.reduce((a, x) => a + x, 0) / n;
  const ss = xs.reduce((a, x) => a + (x - mean) ** 2, 0);
  return { n, mean, std: n - ddof > 0 ? Math.sqrt(ss / (n - ddof)) : 0 };
}
function advantageMap({ groups, mode = "outcome", ddof = 0 }) {
  const o = { groups: [], nNeg: 0, nPos: 0, nZero: 0, nUnits: 0, rftNeg: 0, rftPos: 0, uniformGroups: 0 };
  groups.forEach((g, gi) => {
    const flat = mode === "process" ? g.flat() : g;
    const st = stats(flat, ddof), uniform = st.std < EPS0;
    const norm = r => (uniform ? 0 : (r - st.mean) / st.std);
    const binary = mode === "outcome" && g.every(r => r === 0 || r === 1);
    const outs = g.map((out, i) => {
      if (mode !== "process") {
        const A = norm(out), I = binary ? out : null;
        o[`A${gi + 1}.${i + 1}`] = A; if (I !== null) o[`I${gi + 1}.${i + 1}`] = I;
        return { r: out, A, I, steps: null };
      }
      const rt = out.map(norm), steps = out.map((r, j) => ({ r, rt: rt[j], A: rt.slice(j).reduce((a, x) => a + x, 0) }));
      steps.forEach((s, j) => { o[`A${gi + 1}.${i + 1}.${j + 1}`] = s.A; o[`rt${gi + 1}.${i + 1}.${j + 1}`] = s.rt; });
      return { r: out, steps };
    });
    for (const u of outs.flatMap(x => (x.steps ? x.steps : [x]))) { o.nUnits++; if (u.A < -1e-9) o.nNeg++; else if (u.A > 1e-9) o.nPos++; else o.nZero++; }
    for (const x of outs) if (x.I === 1) o.rftPos++;
    o[`mean${gi + 1}`] = st.mean; o[`std${gi + 1}`] = st.std;
    const sum = mode === "process" ? outs.flatMap(x => x.steps).reduce((a, s) => a + s.rt, 0) : outs.reduce((a, x) => a + x.A, 0);
    o[`sum${gi + 1}`] = Math.round(sum * 1e9) / 1e9;                         // 0 up to float noise
    if (uniform) o.uniformGroups++;
    o.groups.push({ ...st, uniform, binary, outs });
  });
  return o;
}
MODELS["fixture:grpo--advantage-map"] = {
  fn: advantageMap,
  cases: [
    { args: { groups: [[1, 0, 0, 0, 0]] }, pick: "A1.1", expect: 2.0 },                     // outcome predict: correct output +2.0
    { args: { groups: [[1, 0, 0, 0, 0]] }, pick: "A1.2", expect: -0.5 },                    // outcome predict: each wrong one −0.5
    { args: { groups: [[1, 0, 0, 0, 0]], ddof: 1 }, pick: "A1.1", expect: 1.789 },          // same with G−1 (FeynRL): +1.79
    { args: { groups: [[1, 0, 0, 0, 0]] }, pick: "sum1", expect: 0, tol: 1e-9 },            // advantages sum to zero
    { args: { groups: [[0.9, 0.1, 0.5, 0.5]] }, pick: "A1.1", expect: 1.414 },              // outcome transfer: +1.41
    { args: { groups: [[0.9, 0.1, 0.5, 0.5]] }, pick: "A1.3", expect: 0, tol: 1e-9 },       // outcome transfer: score = mean gets 0
    { args: { groups: [[1, 1, 1, 0.8]] }, pick: "A1.4", expect: -1.732, tol: 0.01, from: "grpo:group-relative-advantage-outcome:check" },
    { args: { groups: [[[0.9, 0.5], [0.2, 0.6, 0.8]]], mode: "process" }, pick: "A1.2.1", expect: -0.8165, tol: 0.01, from: "grpo:process-supervision-advantage:check" },
    { args: { groups: [[[0.9, 0.5], [0.2, 0.6, 0.8]]], mode: "process" }, pick: "rt1.2.1", expect: -1.633 },  // step 1 alone (the distractor)
    { args: { groups: [[[0.9, 0.5], [0.2, 0.6, 0.8]]], mode: "process" }, pick: "A1.2.3", expect: 0.8165 },   // last step: one term
    { args: { groups: [[1, 1, 1, 1], [0, 0, 0, 0], [1, 0, 0, 0]] }, pick: "nNeg", expect: 3, from: "grpo:gradient-coefficient-comparison:check" },
    { args: { groups: [[1, 1, 1, 1], [0, 0, 0, 0], [1, 0, 0, 0]] }, pick: "rftNeg", expect: 0 },             // Online RFT never pushes down
    { args: { groups: [[1, 1, 1, 1]] }, pick: "A1.1", expect: 0, tol: 1e-9 },              // coefficient predict: all correct, GRPO 0
    { args: { groups: [[1, 1, 1, 1]] }, pick: "I1.1", expect: 1 },                         // ... while Online RFT puts 1 on every token
    { args: { groups: [[0, 0, 0, 0]] }, pick: "uniformGroups", expect: 1 },                // coefficient transfer: all wrong, zero signal
  ],
};
WIDGETS["fixture:grpo--advantage-map"] = (root) => {
  const s = { mode: "outcome", ddof: 0, text: { outcome: "1, 0, 1, 1, 0, 0", process: "0.7 0.4 0.9, 0.3 0.8" } };
  const pic = el("div"), read = el("div", { class: "readout" }), err = el("div", { class: "muted small" });
  const inp = el("input", { type: "text", style: "width:100%;max-width:520px" });
  const parse = () => {
    const gs = s.text[s.mode].split("|").map(g => g.split(",").map(o => o.trim().split(/\s+/).filter(Boolean).map(Number)).filter(o => o.length));
    if (!gs.length || gs.some(g => g.length < 2)) return { err: "each question needs at least two outputs (separate outputs with commas)" };
    if (gs.some(g => g.some(o => o.some(x => !Number.isFinite(x))))) return { err: "rewards must be numbers" };
    if (s.mode === "outcome" && gs.some(g => g.some(o => o.length !== 1))) return { err: "outcome supervision: one reward per output (switch to process for step rewards)" };
    return { groups: s.mode === "outcome" ? gs.map(g => g.map(o => o[0])) : gs };
  };
  const draw = () => {
    const p = parse(); err.textContent = p.err || "";
    if (p.err) { pic.innerHTML = ""; read.innerHTML = ""; return; }
    const m = advantageMap({ groups: p.groups, mode: s.mode, ddof: s.ddof });
    const all = m.groups.flatMap(g => g.outs.flatMap(x => (x.steps ? x.steps : [x])).map(u => Math.abs(u.A)));
    const amax = Math.max(1e-9, ...all), cw = 15, per = s.mode === "outcome" ? 8 : 3, x0 = 118;
    const maxCells = Math.max(...m.groups.flatMap(g => g.outs.map(x => (x.steps ? x.steps.length * per : per))));
    const xr = x0 + maxCells * cw + 12, W = Math.max(640, xr + 150);
    const fill = A => (Math.abs(A) < 1e-9 ? "#fff" : A > 0 ? C.ok : C.b), op = A => (0.15 + 0.85 * Math.abs(A) / amax).toFixed(2);
    let y = 18, b = text(10, 14, `token cells (${per} per ${s.mode === "outcome" ? "output, schematic length" : "step, schematic"}) coloured by Â: green pushed up, red pushed down, white zero`, { fill: C.muted, size: 11 });
    let rows = 0;
    m.groups.forEach((g, gi) => {
      if (rows > 26) return;
      y += gi ? 30 : 18;
      b += text(10, y, `question ${gi + 1}: G = ${g.outs.length}, mean = ${fmt(g.mean, 3)}, std = ${fmt(g.std, 3)}${g.uniform ? "  →  all rewards equal: every Â = 0, zero signal" : ""}`, { fill: g.uniform ? C.b : C.muted, size: 11, weight: "bold" });
      g.outs.forEach((x, i) => {
        if (++rows > 26) return;
        y += s.mode === "process" ? (i ? 30 : 26) : (i ? 20 : 10);
        b += text(10, y + 11, `o${i + 1}  r = ${x.steps ? x.r.map(v => fmt(v, 2)).join(" ") : fmt(x.r, 3)}`, { size: 11 });
        const units = x.steps ? x.steps : [x];
        units.forEach((u, j) => {
          for (let c = 0; c < per; c++) b += rect(x0 + (j * per + c) * cw, y, cw - 2, 15, fill(u.A), `fill-opacity="${op(u.A)}" stroke="${Math.abs(u.A) < 1e-9 ? "#cfc9bd" : "none"}"`);
          if (x.steps) b += rect(x0 + (j * per + per - 1) * cw + cw - 4, y - 2, 2, 19, C.ink) + text(x0 + (j * per + per / 2) * cw, y - 3, sgn(u.A, 2), { size: 9, anchor: "middle", fill: C.muted });
        });
        b += text(xr, y + 11, x.steps ? `step r̃: ${x.steps.map(st => sgn(st.rt, 2)).join(" ")}` : `Â = ${sgn(x.A)}${x.I !== null ? ` · Online RFT I(o) = ${x.I}` : ""}`, { size: 11, fill: x.steps ? C.muted : C.ink });
      });
      y += 6;
    });
    if (rows > 26) b += text(10, y + 14, "(first 26 outputs shown)", { fill: C.muted, size: 10 });
    pic.innerHTML = svg(W, y + 22, b);
    const binaryAll = s.mode === "outcome" && m.groups.every(g => g.binary);
    read.innerHTML = `${s.mode === "outcome" ? "outcome supervision: Â<sub>i,t</sub> = (r<sub>i</sub> − mean(r)) / std(r), the <b>same value on every token</b> of o<sub>i</sub>" : "process supervision: step rewards of all outputs pooled into R, r̃ = (r − mean(R)) / std(R); a token's Â = sum of r̃ of its step <b>and every later step</b> (tick = step end, number above = that step's tokens' Â)"}<br>
      std divides by ${s.ddof ? "G − 1 (FeynRL)" : "G (population; the paper does not say)"} · per question Σ ${s.mode === "outcome" ? "Â" : "r̃"} = ${m.groups.map((g, gi) => fmt(m[`sum${gi + 1}`], 6)).join(", ")}<br>
      <span class="big">GRPO pushes down ${m.nNeg} of ${m.nUnits} ${s.mode === "outcome" ? "outputs" : "steps"}, pushes up ${m.nPos}, leaves ${m.nZero} at zero</span>${binaryAll ? ` · Online RFT (I(o)): pushes up ${m.rftPos}, pushes down 0` : ""}<br>
      ${m.uniformGroups ? `${m.uniformGroups} question(s) with all-equal rewards: r − mean = 0 for every output, so the ratio term gives no gradient (only the KL term acts)<br>` : ""}
      <span class="muted small">provenance: fixture:grpo--advantage-map · paper.txt:L933 (§4.1.2 outcome Â), L937-L938 (§4.1.3 pooled normalization, Σ<sub>index(j) ≥ t</sub>), L1445 (Eq. 10 vs 21: Online RFT does not penalize incorrect responses; GRPO reinforces and penalizes by magnitude). Token counts per output/step are schematic; std = 0 is handled as Â = 0 (implementations add ε).</span>`;
  };
  inp.addEventListener("input", () => { s.text[s.mode] = String(inp.value ?? ""); draw(); });
  const modeSel = el("select", {}, el("option", { value: "outcome" }, "outcome supervision (§4.1.2)"), el("option", { value: "process" }, "process supervision (§4.1.3)"));
  modeSel.addEventListener("change", () => { s.mode = modeSel.value; inp.value = s.text[s.mode]; draw(); });
  const presets = { outcome: [["one question, mixed", "1, 0, 1, 1, 0, 0"], ["three questions", "1, 1, 1, 1 | 0, 0, 0, 0 | 1, 1, 0, 0"], ["graded scores", "0.9, 0.6, 0.2, 0.3"]], process: [["two outputs", "0.7 0.4 0.9, 0.3 0.8"], ["one bad middle step", "0.8 0.1 0.9, 0.8 0.8 0.9"]] };
  const pbox = el("div", { style: "margin-top:6px" });
  const fillPresets = () => { pbox.replaceChildren(...presets[s.mode].map(([n, t]) => el("button", { onclick: () => { s.text[s.mode] = t; inp.value = t; draw(); } }, n)), " "); };
  modeSel.addEventListener("change", fillPresets);
  inp.value = s.text[s.mode]; fillPresets();
  root.append(el("div", { class: "widget" }, el("div", { class: "controls" },
    el("label", {}, "supervision ", modeSel),
    el("label", {}, "rewards (outputs separated by commas, questions by |, step rewards by spaces) ", inp), err,
    check("std with G − 1 (FeynRL) instead of G", false, v => { s.ddof = v ? 1 : 0; draw(); }), pbox,
    read), pic)); draw();
};

// ================================================== 2. per-token coefficient ==
// One token of Eq. 3 (paper.txt:L861-L864): min[ρÂ, clip(ρ, 1−ε, 1+ε)Â] − β·D_KL, ρ = π_θ/π_old, D_KL by Eq. 4 (L869-L873).
// Its gradient is GC·∇log π_θ. Ratio part: ρÂ when the unclipped branch is active (inside the band, or outside it on the side
// where ρÂ is the smaller), 0 when the clipped constant wins. KL part: d/dlog π_θ of −β(u − log u − 1), u = π_ref/π_θ, is
// β(u − 1), the term of Eq. 21 (L2205-L2207). With one update per sampling stage (§4.2, L956-L957; App. A.1.5, L2156) ρ = 1 at the
// gradient and GC reduces to Eq. 21's Â + β(π_ref/π_θ − 1). β default 0.04 is §4.2's KL coefficient (L955); ε has no printed
// value (default 0.2 = the prompts' band [0.8, 1.2]).
function tokenCoef({ lold, lp, lref, A, eps = 0.2, beta = 0.04 }) {
  const rho = Math.exp(lp - lold), lo = 1 - eps, hi = 1 + eps;
  const unc = rho * A, clp = Math.min(Math.max(rho, lo), hi) * A;
  const clipped = clp < unc - 1e-12;                                         // min picks the flat (clipped) branch
  const cRatio = clipped ? 0 : rho * A;
  const u = Math.exp(lref - lp), k3 = u - Math.log(u) - 1, cKL = beta * (u - 1);
  const inBand = rho >= lo - 1e-12 && rho <= hi + 1e-12;
  return { rho, unc, clp, obj: Math.min(unc, clp), clipped, inBand, cRatio, u, k3, k1: lp - lref, cKL, GC: cRatio + cKL,
    eq21: A + cKL, total: Math.min(unc, clp) - beta * k3 };
}
MODELS["fixture:grpo--token-coefficient"] = {
  fn: tokenCoef,
  cases: [
    { args: { lold: -1, lp: -1, lref: -1, A: 0.8, beta: 0 }, pick: "GC", expect: 0.8 },             // supp-importance-ratio predict: ρ = 1, 0.8 ∇log π
    { args: { lold: -1, lp: -1, lref: -1, A: 0.8, beta: 0 }, pick: "obj", expect: 0.8 },            // ... and the value 0.8
    { args: { lold: -1, lp: -0.5, lref: -1, A: -1, beta: 0 }, pick: "GC", expect: -1.6487 },        // transfer: unclipped branch, −1.65
    { args: { lold: -2, lp: -2.3, lref: -2, A: 1, beta: 0 }, pick: "GC", expect: 0.741, tol: 0.01, from: "grpo:supp-importance-ratio:check" },
    { args: { lold: -1, lp: -0.5, lref: -1, A: 1, beta: 0 }, pick: "GC", expect: 0, tol: 1e-9 },    // edge: A > 0 above 1+ε, clipped, 0
    { args: { lold: 0, lp: Math.log(0.7), lref: 0, A: -1, beta: 0 }, pick: "GC", expect: 0, tol: 1e-9 }, // edge: A < 0 below 1−ε, 0
    { args: { lold: -Math.log(3), lp: -Math.log(3), lref: 0, A: -0.6, beta: 0.1 }, pick: "GC", expect: -0.4, tol: 0.01, from: "grpo:grpo-objective-clip-kl:check" },
    { args: { lold: -Math.log(3), lp: -Math.log(3), lref: 0, A: -0.6, beta: 0.1 }, pick: "eq21", expect: -0.4 }, // ρ = 1: same as Eq. 21
    { args: { lold: Math.log(0.5), lp: Math.log(0.5), lref: Math.log(0.25), A: 0, beta: 0.04 }, pick: "GC", expect: -0.02 }, // transfer: π_θ = 2π_ref
    { args: { lold: Math.log(0.25), lp: Math.log(0.25), lref: Math.log(0.5), A: 0, beta: 0.04 }, pick: "GC", expect: 0.04 },  // π_θ = 0.5π_ref
    { args: { lold: Math.log(0.25), lp: Math.log(0.25), lref: Math.log(0.5), A: 0, beta: 0 }, pick: "GC", expect: 0, tol: 1e-9 }, // kl_coeff 0
  ],
};
WIDGETS["fixture:grpo--token-coefficient"] = (root) => {
  const s = { lold: -1.0, lp: -0.8, lref: -1.2, A: 0.5, eps: 0.2, beta: 0.04 };
  const pic = el("div"), read = el("div", { class: "readout" });
  const sl = {};
  const draw = () => {
    const m = tokenCoef(s), W = 640, H = 250, L = 50, R = 610, T = 24, B = 210, rmax = 2.5;
    const u0 = Math.exp(s.lref - s.lold), fr = r => Math.min(r * s.A, Math.min(Math.max(r, 1 - s.eps), 1 + s.eps) * s.A);
    const ft = r => { const u = u0 / r; return fr(r) - s.beta * (u - Math.log(u) - 1); };
    const ymax = Math.max(1, Math.abs(s.A) * rmax) * 1.1, X = r => L + (r / rmax) * (R - L), Y = v => T + (ymax - Math.max(-ymax, Math.min(ymax, v))) / (2 * ymax) * (B - T);
    let b = rect(X(1 - s.eps), T, X(1 + s.eps) - X(1 - s.eps), B - T, C.hi, 'fill-opacity="0.15"');
    b += line(L, Y(0), R, Y(0), C.rule) + line(X(1), T, X(1), B, C.rule, 'stroke-dasharray="3 3"');
    const pts = f => { const o = []; for (let r = 0.02; r <= rmax + 1e-9; r += 0.01) o.push([X(r), Y(f(r))]); return o; };
    b += path(pts(r => r * s.A), C.a, 'stroke-opacity="0.3"') + path(pts(fr), C.a) + (s.beta > 0 ? path(pts(ft), C.b, 'stroke-dasharray="5 3"') : "");
    const px = X(Math.min(rmax, m.rho)), py = Y(s.beta > 0 ? m.total : m.obj);
    b += `<circle cx="${px.toFixed(1)}" cy="${py.toFixed(1)}" r="5" style="fill:${m.clipped ? C.b : C.ok}"/>`;
    b += text(X(1) + 4, T + 12, "ρ = 1", { fill: C.muted, size: 11 }) + text(X(1 - s.eps), B + 14, `1−ε`, { fill: C.muted, size: 11, anchor: "middle" }) + text(X(1 + s.eps), B + 14, `1+ε`, { fill: C.muted, size: 11, anchor: "middle" });
    b += text(R, B + 30, "ρ = π_θ / π_old (π_old and π_ref held fixed)", { fill: C.muted, size: 11, anchor: "end" });
    b += text(L, 14, `solid: min[ρÂ, clip(ρ)Â] · faint: ρÂ${s.beta > 0 ? " · dashed: minus β·D_KL (Eq. 4)" : ""} · band shaded`, { fill: C.muted, size: 11 });
    pic.innerHTML = svg(W, H, b);
    const regime = m.clipped ? `<b>clipped</b>: ρ = ${fmt(m.rho, 3)} is ${s.A > 0 ? "above 1+ε with Â > 0" : "below 1−ε with Â < 0"}, the min is the flat clip(ρ)Â, ratio part of the gradient = 0`
      : m.inBand ? `<b>unclipped</b>: ρ = ${fmt(m.rho, 3)} is inside [${fmt(1 - s.eps, 2)}, ${fmt(1 + s.eps, 2)}], ratio part = ρÂ`
      : `<b>unclipped although outside the band</b>: with Â ${s.A >= 0 ? "> 0 the clip only binds above 1+ε" : "< 0 the clip only binds below 1−ε"}, so ρÂ is the min and keeps its gradient`;
    read.innerHTML = `ρ = exp(log π_θ − log π_old) = <b>${fmt(m.rho, 4)}</b> · ρÂ = ${fmt(m.unc, 4)} · clip(ρ)Â = ${fmt(m.clp, 4)} · min = ${fmt(m.obj, 4)}<br>${regime}${Math.abs(m.rho - 1) < 1e-9 ? " (ρ = 1: the first gradient after sampling; with one update per sampling stage, §4.2, the clip never binds)" : ""}<br>
      u = π_ref/π_θ = ${fmt(m.u, 4)} · D_KL estimate (Eq. 4) = u − log u − 1 = ${fmt(m.k3, 4)} · KL part of the coefficient β(u − 1) = ${sgn(m.cKL, 4)} (${m.cKL > 1e-12 ? "pulls this token up toward π_ref" : m.cKL < -1e-12 ? "pushes this token down toward π_ref" : "no pull"})<br>
      <span class="big">GC = ${sgn(m.cRatio, 4)} (ratio) ${m.cKL < 0 ? "−" : "+"} ${fmt(Math.abs(m.cKL), 4)} (KL) = ${sgn(m.GC, 4)}</span> · gradient = GC·∇log π_θ · Eq. 21 at ρ = 1: Â + β(u − 1) = ${sgn(m.eq21, 4)}<br>
      Â comes from the group's rewards alone; β changes only the KL part, never Â<br>
      <span class="muted small">provenance: fixture:grpo--token-coefficient · paper.txt:L861-L864 (Eq. 3), L869-L873 (Eq. 4, "guaranteed to be positive"), L2156 (App. A.1.5: one update ⇒ π_old = π_θ, min and clip removed), L2205-L2207 (Eq. 21), L955 (β = 0.04). ε: no value printed; 0.2 = this thread's prompts.</span>`;
  };
  const set = (k, v) => { s[k] = v; draw(); };
  const lpf = v => v.toFixed(2);
  root.append(el("div", { class: "widget" }, el("div", { class: "controls" },
    slider("log π_old(o_t) (sampling policy)", -4, 0, s.lold, 0.01, v => set("lold", v), lpf),
    slider("log π_θ(o_t) (current policy)", -4, 0, s.lp, 0.01, v => set("lp", v), lpf),
    slider("log π_ref(o_t) (reference)", -4, 0, s.lref, 0.01, v => set("lref", v), lpf),
    slider("advantage Â", -2, 2, s.A, 0.05, v => set("A", v), v => v.toFixed(2)),
    slider("clip ε", 0, 0.5, s.eps, 0.05, v => set("eps", v), v => v.toFixed(2)),
    slider("KL coefficient β", 0, 0.2, s.beta, 0.01, v => set("beta", v), v => v.toFixed(2)),
    read), pic)); draw();
};

// ===================================================== 3. KL estimators k1, k3 ==
// Eq. 4 (paper.txt:L869-L873): D_KL[π_θ || π_ref] ≈ u − log u − 1 at a sampled token, u = π_ref/π_θ, "unbiased" (Schulman, 2020,
// L866) and "guaranteed to be positive". k1 = log(π_θ/π_ref) = −log u is the plain log-ratio (Eq. 2's per-token penalty, L850).
// To show the expectation, the vocabulary is collapsed to two outcomes: the sampled token (log-probs set by the sliders) and
// "every other token" carrying the rest of the mass. Under o ~ π_θ: E[k1] = KL exactly, and E[u] = Σ π_ref = 1 so E[k3] = KL too.
function klEst({ lp, lref }) {
  const p = Math.exp(lp), q = Math.exp(lref);
  const k1f = u => -Math.log(u), k3f = u => u - Math.log(u) - 1;
  const outs = [{ w: p, u: q / p }, { w: 1 - p, u: (1 - q) / (1 - p) }].filter(o => o.w > 1e-12);
  const E = f => outs.reduce((a, o) => a + o.w * f(o.u), 0);
  const kl = outs.reduce((a, o) => a + o.w * Math.log(1 / o.u), 0);                // Σ π_θ log(π_θ/π_ref)
  const Ek1 = E(k1f), Ek3 = E(k3f), Vk1 = E(u => (k1f(u) - Ek1) ** 2), Vk3 = E(u => (k3f(u) - Ek3) ** 2);
  const u = q / p, uo = (1 - q) / (1 - p);
  return { p, q, u, k1: k1f(u), k3: k3f(u), uOther: uo, k1Other: k1f(uo), k3Other: k3f(uo), kl, Ek1, Ek3, Vk1, Vk3,
    sdk1: Math.sqrt(Vk1), sdk3: Math.sqrt(Vk3), negMass: outs.filter(o => k1f(o.u) < -1e-12).reduce((a, o) => a + o.w, 0) };
}
MODELS["fixture:grpo--kl-estimators"] = {
  fn: klEst,
  cases: [
    { args: { lp: Math.log(0.25), lref: Math.log(0.5) }, pick: "k3", expect: 0.307, tol: 0.01, from: "grpo:kl-k3-estimator:predict" },
    { args: { lp: Math.log(0.5), lref: Math.log(0.25) }, pick: "k3", expect: 0.193, tol: 0.01 },     // transfer: u = 0.5, smaller
    { args: { lp: -2.0, lref: -1.0 }, pick: "k3", expect: 0.718, tol: 0.01, from: "grpo:kl-k3-estimator:check" },
    { args: { lp: -2.0, lref: -1.0 }, pick: "k1", expect: -1.0 },                                     // the check's k1 distractor
    { args: { lp: -0.1, lref: -0.15 }, pick: "k3", expect: 0.00123, tol: 0.01 },                     // FeynRL's test value (predict why)
    { args: { lp: -2.0, lref: -1.0 }, pick: "Ek3", expect: 0.1355, tol: 0.01 },                      // unbiased: E[k3] = KL ...
    { args: { lp: -2.0, lref: -1.0 }, pick: "kl", expect: 0.1355, tol: 0.01 },                       // ... the exact two-outcome KL
    { args: { lp: -2.0, lref: -1.0 }, pick: "Ek1", expect: 0.1355, tol: 0.01 },                      // k1 is unbiased too
    { args: { lp: -1.2, lref: -1.2 }, pick: "k3", expect: 0, tol: 1e-9 },                            // u = 1: zero
  ],
};
WIDGETS["fixture:grpo--kl-estimators"] = (root) => {
  const s = { lp: -1.5, lref: -0.9 };
  const pic = el("div"), read = el("div", { class: "readout" });
  const draw = () => {
    const m = klEst(s), W = 640, H = 240, L = 40, R = 400, T = 22, B = 200, xm = 2.5, ymin = -2.5, ymax = 4;
    const X = lu => L + (lu + xm) / (2 * xm) * (R - L), Y = v => T + (ymax - Math.max(ymin, Math.min(ymax, v))) / (ymax - ymin) * (B - T);
    let b = line(L, Y(0), R, Y(0), C.rule) + line(X(0), T, X(0), B, C.rule, 'stroke-dasharray="3 3"');
    const pts = f => { const o = []; for (let lu = -xm; lu <= xm + 1e-9; lu += 0.02) { const v = f(Math.exp(lu)); if (v >= ymin && v <= ymax) o.push([X(lu), Y(v)]); } return o; };
    b += path(pts(u => -Math.log(u)), C.a) + path(pts(u => u - Math.log(u) - 1), C.b);
    const dot = (u, w, col, lab) => { const lu = Math.log(u); if (Math.abs(lu) > xm) return ""; return [[-Math.log(u), C.a], [u - Math.log(u) - 1, C.b]].map(([v, c]) => `<circle cx="${X(lu).toFixed(1)}" cy="${Y(v).toFixed(1)}" r="${(3 + 6 * Math.sqrt(w)).toFixed(1)}" style="fill:${c};fill-opacity:0.75"/>`).join("") + text(X(lu), B + 14, lab, { anchor: "middle", size: 10, fill: col }); };
    b += dot(m.u, m.p, C.ink, "sampled") + dot(m.uOther, 1 - m.p, C.muted, "others");
    b += text(L, 14, "per-sample value vs log u (u = π_ref/π_θ): blue k1, orange k3", { fill: C.muted, size: 11 });
    b += text(R, B + 30, "log u →  (dot area ∝ π_θ mass)", { fill: C.muted, size: 10, anchor: "end" }) + text(L - 4, Y(0) + 4, "0", { fill: C.muted, size: 10, anchor: "end" });
    const bx = 440, bw = 50, top = Math.max(m.Ek1 + m.sdk1, m.Ek3 + m.sdk3, 1e-6) * 1.15, YB = v => B - Math.max(0, v) / top * (B - T - 34);
    [["k1", m.Ek1, m.sdk1, C.a], ["k3", m.Ek3, m.sdk3, C.b]].forEach(([n, e, sd, c], i) => {
      const x = bx + i * (bw + 30);
      b += rect(x, YB(e), bw, B - YB(e), c, 'fill-opacity="0.8"') + line(x + bw / 2, YB(e - sd), x + bw / 2, YB(e + sd), C.ink);
      b += text(x + bw / 2, B + 14, `E[${n}]`, { anchor: "middle", size: 11 }) + text(x + bw / 2, YB(e + sd) - 4, `sd ${fmt(m[`sd${n}`], 3)}`, { anchor: "middle", size: 10, fill: C.muted });
    });
    b += line(bx - 6, YB(m.kl), bx + 2 * bw + 36, YB(m.kl), C.ok, 'stroke-dasharray="4 3"') + text(bx - 8, T + 12, "mean ± sd under o ~ π_θ; green = true KL", { size: 10, fill: C.muted });
    pic.innerHTML = svg(W, H, b);
    read.innerHTML = `sampled token: π_θ = ${fmt(m.p, 4)}, π_ref = ${fmt(m.q, 4)}, u = π_ref/π_θ = <b>${fmt(m.u, 4)}</b> · ${m.u > 1 ? "the policy has shrunk this token" : m.u < 1 ? "the policy has grown this token" : "no change"}<br>
      <span class="big">k3 = u − log u − 1 = ${fmt(m.k3, 4)}</span> · k1 = log(π_θ/π_ref) = ${sgn(m.k1, 4)}${m.k1 < -1e-12 ? " (<b>negative</b>: a KL estimate below zero on this sample)" : ""}<br>
      collapsed vocabulary (sampled token vs the rest): true KL = ${fmt(m.kl, 4)} · E[k1] = ${fmt(m.Ek1, 4)} · E[k3] = ${fmt(m.Ek3, 4)} (both unbiased) · sd k1 = ${fmt(m.sdk1, 3)}, sd k3 = ${fmt(m.sdk3, 3)} · π_θ-mass of samples where k1 &lt; 0: ${fmt(m.negMass, 3)}, where k3 &lt; 0: 0<br>
      <span class="muted small">provenance: fixture:grpo--kl-estimators · paper.txt:L866 (unbiased estimator, Schulman 2020), L869-L873 (Eq. 4, "guaranteed to be positive"), L850 (Eq. 2's log-ratio penalty = k1). The two-outcome vocabulary is a demo device: the paper estimates per token, the expectation is over the full vocabulary.</span>`;
  };
  root.append(el("div", { class: "widget" }, el("div", { class: "controls" },
    slider("log π_θ(o_t) (policy)", -4, -0.05, s.lp, 0.01, v => { s.lp = v; draw(); }, v => v.toFixed(2)),
    slider("log π_ref(o_t) (reference)", -4, -0.05, s.lref, 0.01, v => { s.lref = v; draw(); }, v => v.toFixed(2)),
    read), pic)); draw();
};

// ========================================================= 4. free baseline ==
// Eq. 5 (unified gradient, paper.txt:L1353-L1358): every method moves θ along E[GC · ∇log π_θ]; §4.1.1 (L858) calls the
// value function "a baseline ... for variance reduction" and GRPO uses the group's mean reward instead (L858, L933).
// Smallest concrete policy: one question, two outputs a, b with rewards R(a) = 1, R(b) = 0 (the supp-policy-gradient predict),
// and one logit θ with π(a) = σ(θ), so ∇_θ log π(a) = π(b) and ∇_θ log π(b) = −π(a) are numbers. The single-sample estimate is
// g = (R − b)/s · ∇log π(o). E[g] = π(a)π(b)/s for every b (the score-function identity); Var[g] depends on b and is 0 at
// b* = π(b). The one-logit policy is the widget's device, not the paper's.
function baseline({ pa, b, s = 1 }) {
  const pb = 1 - pa, ga = (1 - b) * pb / s, gb = (0 - b) * (-pa) / s;
  const E = pa * ga + pb * gb, V = pa * ga * ga + pb * gb * gb - E * E;
  return { ga, gb, E, V: Math.max(0, V), sd: Math.sqrt(Math.max(0, V)), coefA: pa * (1 - b) / s, coefB: pb * (0 - b) / s, bStar: pb, bMean: pa, E0: pa * pb };
}
MODELS["fixture:grpo--baseline"] = {
  fn: baseline,
  cases: [
    { args: { pa: 0.75, b: 0 }, pick: "coefA", expect: 0.75 },            // predict, before: 0.75 ∇log π(a)
    { args: { pa: 0.75, b: 0.5 }, pick: "coefA", expect: 0.375 },         // predict, after: 0.75·0.5 ∇log π(a) ...
    { args: { pa: 0.75, b: 0.5 }, pick: "coefB", expect: -0.125 },        // ... − 0.25·0.5 ∇log π(b)
    { args: { pa: 0.75, b: 0 }, pick: "E", expect: 0.1875 },              // the two are equal as a number on the logit
    { args: { pa: 0.75, b: 0.5 }, pick: "E", expect: 0.1875 },
    { args: { pa: 0.75, b: 0, s: 0.4 }, pick: "E", expect: 0.46875 },     // transfer: divide by fixed s = 0.4, magnitude × 2.5
    { args: { pa: 0.75, b: 0 }, pick: "V", expect: 0.01171875 },          // variance without a baseline ...
    { args: { pa: 0.75, b: 0.25 }, pick: "V", expect: 0, tol: 1e-9 },     // ... is 0 at b* = π(b)
    { args: { pa: 0.75, b: 0.75 }, pick: "V", expect: 0.046875 },         // b = mean reward π(a) is not b* unless π(a) = 0.5
  ],
};
WIDGETS["fixture:grpo--baseline"] = (root) => {
  const st = { pa: 0.6, b: 0, s: 1 };
  const pic = el("div"), read = el("div", { class: "readout" });
  const draw = () => {
    const m = baseline(st), W = 640, H = 230, L = 50, R = 420, T = 24, B = 190, b0 = -1, b1 = 2;
    const curve = bb => baseline({ ...st, b: bb });
    const ymax = Math.max(...Array.from({ length: 61 }, (_, k) => { const c = curve(b0 + k * 0.05); return Math.max(c.E, c.sd); })) * 1.1;
    const X = bb => L + (bb - b0) / (b1 - b0) * (R - L), Y = v => B - v / ymax * (B - T);
    let g = line(L, B, R, B, C.rule) + line(L, T, L, B, C.rule);
    const pts = f => { const o = []; for (let bb = b0; bb <= b1 + 1e-9; bb += 0.02) o.push([X(bb), Y(f(curve(bb)))]); return o; };
    g += path(pts(c => c.E), C.a) + path(pts(c => c.sd), C.b);
    g += line(X(m.bMean), T, X(m.bMean), B, C.ok, 'stroke-dasharray="4 3"') + text(X(m.bMean), B + 14, "mean reward π(a)", { size: 10, fill: C.ok, anchor: "middle" });
    g += line(X(m.bStar), T, X(m.bStar), B, C.muted, 'stroke-dasharray="2 3"') + text(X(m.bStar), B + 26, "b* = π(b)", { size: 10, fill: C.muted, anchor: "middle" });
    g += `<circle cx="${X(st.b).toFixed(1)}" cy="${Y(m.E).toFixed(1)}" r="5" style="fill:${C.a}"/><circle cx="${X(st.b).toFixed(1)}" cy="${Y(m.sd).toFixed(1)}" r="5" style="fill:${C.b}"/>`;
    g += text(L, 14, "vs baseline b: blue E[g] (expected gradient), orange sd[g] (one-sample noise)", { size: 11, fill: C.muted });
    g += text(R, B + 14, "baseline b →", { size: 10, fill: C.muted, anchor: "end" });
    const gm = Math.max(Math.abs(m.ga), Math.abs(m.gb), m.E, 1e-6), bx = 470, YG = v => 110 - v / gm * 70;
    [["a sampled", m.ga, st.pa], ["b sampled", m.gb, 1 - st.pa]].forEach(([n, v, w], i) => { const x = bx + i * 70;
      g += rect(x, Math.min(YG(v), 110), 44, Math.abs(YG(v) - 110), C.b, `fill-opacity="${(0.3 + 0.7 * w).toFixed(2)}"`) + text(x + 22, 196, n, { size: 10, anchor: "middle" }) + text(x + 22, 210, `p = ${fmt(w, 2)}`, { size: 10, anchor: "middle", fill: C.muted }); });
    g += line(bx - 6, YG(m.E), bx + 120, YG(m.E), C.a, 'stroke-dasharray="4 3"') + line(bx - 6, 110, bx + 120, 110, C.rule) + text(bx - 8, T + 10, "g per sample; dashed = mean", { size: 10, fill: C.muted });
    pic.innerHTML = svg(W, H, g);
    read.innerHTML = `two outputs, R(a) = 1, R(b) = 0, π(a) = ${fmt(st.pa, 2)} · baseline b = ${fmt(st.b, 2)}${st.s !== 1 ? ` · coefficient divided by s = ${fmt(st.s, 2)}` : ""}<br>
      E[(R − b)∇log π]${st.s !== 1 ? "/s" : ""} = ${fmt(m.coefA, 4)}·∇log π(a) ${m.coefB < 0 ? "−" : "+"} ${fmt(Math.abs(m.coefB), 4)}·∇log π(b) = <span class="big">${fmt(m.E, 4)}</span> on the logit (b = 0, s = 1 gives ${fmt(m.E0, 4)})<br>
      one-sample noise sd = ${fmt(m.sd, 4)} · zero at b* = π(b) = ${fmt(m.bStar, 2)}, where both samples give the same g · ${Math.abs(m.E * st.s - m.E0) < 1e-9 ? (st.s === 1 ? "expectation unchanged by b" : `expectation scaled by 1/s = ${fmt(1 / st.s, 3)}, direction unchanged`) : ""}<br>
      <span class="muted small">provenance: fixture:grpo--baseline · paper.txt:L1353-L1358 (Eq. 5, GC·∇log π), L858 (value function as a baseline for variance reduction; GRPO's group average as the baseline). Rewards and π(a) come from the supp-policy-gradient prompts; the one-logit policy σ(θ) is the widget's device. A per-question s fixed in advance only rescales; GRPO's std is recomputed from each sampled group (course p23).</span>`;
  };
  root.append(el("div", { class: "widget" }, el("div", { class: "controls" },
    slider("π(a), probability of the correct output", 0.05, 0.95, st.pa, 0.05, v => { st.pa = v; draw(); }, v => v.toFixed(2)),
    slider("baseline b (subtracted from both rewards)", -1, 2, st.b, 0.05, v => { st.b = v; draw(); }, v => v.toFixed(2)),
    slider("divide the coefficient by s", 0.1, 2, st.s, 0.05, v => { st.s = v; draw(); }, v => v.toFixed(2)),
    read), pic)); draw();
};
