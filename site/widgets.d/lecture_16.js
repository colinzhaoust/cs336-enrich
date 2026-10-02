// Widgets for thread lecture_16 (CS336 L16, post-training 2: RL from verifiable rewards). Register as "fixture:<id>" -> (root, notice) => void.
// Each widget has a pure model in MODELS (no DOM) that tools/check_widgets.mjs tests against this thread's stored answers.
// Sources: official/lectures/lecture_16.pdf (cited as p<n>, text layer in lectures/lecture_16/lecture_16.txt) and the lecture video
// (cited as video:M:SS, lectures/lecture_16/transcript.json). Formulas that are images on the slides are cited to the slide and the
// KP filtering_note that recorded them. Where a widget needs a magnitude no source prints, it uses a prompt's number or says "demo".
import { el, fmt, slider } from "../widgets_lib.js";

export const MODELS = {};
export const WIDGETS = {};

const C = { ink: "var(--ink)", muted: "var(--muted)", rule: "var(--rule)", a: "var(--accent)", b: "var(--accent2)", ok: "var(--ok)", hi: "var(--hilite)" };
const rect = (x, y, w, h, fill, extra = "") => `<rect x="${x.toFixed(1)}" y="${y.toFixed(1)}" width="${Math.max(0, w).toFixed(1)}" height="${Math.max(0, h).toFixed(1)}" style="fill:${fill}" ${extra}/>`;
const text = (x, y, s, o = {}) => `<text x="${x.toFixed(1)}" y="${y.toFixed(1)}" style="fill:${o.fill || C.ink};font:${o.weight || ""} ${o.size || 12}px var(--sans)" text-anchor="${o.anchor || "start"}">${s}</text>`;
const line = (x1, y1, x2, y2, stroke, extra = "") => `<line x1="${x1.toFixed(1)}" y1="${y1.toFixed(1)}" x2="${x2.toFixed(1)}" y2="${y2.toFixed(1)}" style="stroke:${stroke};stroke-width:1.5" ${extra}/>`;
const path = (pts, stroke, extra = "") => `<path d="${pts.map((p, i) => `${i ? "L" : "M"}${p[0].toFixed(1)},${p[1].toFixed(1)}`).join("")}" style="fill:none;stroke:${stroke};stroke-width:2" ${extra}/>`;
const dot = (x, y, r, fill, extra = "") => `<circle cx="${x.toFixed(1)}" cy="${y.toFixed(1)}" r="${r}" style="fill:${fill}" ${extra}/>`;
const svg = (w, h, body) => `<svg viewBox="0 0 ${w} ${h}" width="100%" style="max-width:${w}px;border:1px solid var(--rule);border-radius:6px;background:#fff" role="img">${body}</svg>`;
const check = (label, value, onChange) => { const i = el("input", { type: "checkbox" }); i.checked = value; i.addEventListener("change", () => onChange(i.checked)); return el("label", {}, i, ` ${label}`); };
const sgn = (x, d = 3) => (x > 0 ? "+" : x < 0 ? "−" : "") + fmt(Math.abs(x), d);
const ZERO = 1e-12;

// ========================================================== 1. group weights ==
// One prompt's group of G sampled responses and their checker rewards r_i. Each estimator multiplies grad log p(o_i) by a weight:
//   raw REINFORCE       r_i                         p5: ∇E[R] = E[R(z) ∇log p(z)] (no baseline)
//   RFT                 1 if correct (r_i > 0) else 0   p21 "reinforcing correct answers", p48 "learn from positives" (SFT on the correct ones)
//   mean baseline       r_i − mean(r)               p22 "subtract any state-dependent term"; p23 Dr. GRPO's unbiased version; Kimi's
//                                                   gradient "a policy gradient with a baseline R bar ... the mean" (video:45:55-46:08)
//   GRPO z-score        (r_i − mean(r)) / (std(r) + ε)   p18 "z-score within group"; p20 "1e-4 stability factor in the std"
// The GRPO weight is the mean-baseline weight times one per-prompt factor 1/(std + ε): "not a valid baseline" (p23), "upweights too easy
// or hard questions" (p24). std divides by G (population), the thread's convention in every prompt; the G − 1 toggle is torch.std's default.
// A uniform group (std = 0) has r − mean = 0 for every sample, so both baselined estimators give 0 whatever ε is.
function groupWeights({ rewards, ddof = 0, eps = 0 }) {
  const n = rewards.length, mean = rewards.reduce((a, x) => a + x, 0) / n;
  const ss = rewards.reduce((a, x) => a + (x - mean) ** 2, 0), std = n - ddof > 0 ? Math.sqrt(ss / (n - ddof)) : 0;
  const uniform = std < ZERO, den = std + eps;
  const o = { n, mean, std, uniform, scale: uniform ? 0 : 1 / den, nzR: 0, nzF: 0, nzB: 0, nzZ: 0, negR: 0, negB: 0, negZ: 0, rows: [] };
  rewards.forEach((r, i) => {
    const R = r, F = r > 0 ? 1 : 0, B = uniform ? 0 : r - mean, Z = uniform ? 0 : (r - mean) / den;
    Object.assign(o, { [`R${i + 1}`]: R, [`F${i + 1}`]: F, [`B${i + 1}`]: B, [`Z${i + 1}`]: Z });
    for (const [k, v] of [["R", R], ["F", F], ["B", B], ["Z", Z]]) if (Math.abs(v) > 1e-9) { o[`nz${k}`]++; if (v < 0 && k !== "F") o[`neg${k}`]++; }
    o.rows.push({ r, R, F, B, Z });
  });
  return o;
}
MODELS["fixture:lecture_16--group-weights"] = {
  fn: groupWeights,
  cases: [
    { args: { rewards: [1, -1, -1, 1, -1, -1, 1, -1] }, pick: "nzR", expect: 8, from: "lecture_16:policy-gradient-variance:check" },
    { args: { rewards: [1, 0, 0, 1, 0, 0, 1, 0] }, pick: "nzR", expect: 3 },                    // same batch coded 1/0: only the 3 correct count
    { args: { rewards: [1, 0] }, pick: "R2", expect: 0, tol: 1e-9 },                            // PG transfer: the 0-reward sample adds nothing
    { args: { rewards: [1, 0, 0, 0, 0] }, pick: "Z1", expect: 2.0, tol: 0.03, from: "lecture_16:grpo-group-zscore-advantage:check" },
    { args: { rewards: [1, 1, 1, 1] }, pick: "Z1", expect: 0, tol: 1e-9 },                      // z-score predict: all solved, every Â = 0
    { args: { rewards: [1, 0, 0, 1] }, pick: "Z1", expect: 1 },                                 // z-score transfer: +1, −1, −1, +1
    { args: { rewards: [1, 0, 0, 0] }, pick: "Z1", expect: 1.732 },                             // ... and 1,0,0,0: +1.73 against
    { args: { rewards: [1, 0, 0, 0] }, pick: "Z2", expect: -0.577 },                            //     −0.58 for each wrong one
    { args: { rewards: [1, 0, 0, 0, 0, 0, 0, 0] }, pick: "scale", expect: 3.024, tol: 0.02, from: "lecture_16:baseline-validity-std-division:check" },
    { args: { rewards: [1, 0, 0, 0] }, pick: "scale", expect: 2.309 },                          // std-division transfer: prompt A ×2.3
    { args: { rewards: [1, 1, 0, 0] }, pick: "scale", expect: 2.0 },                            //                        prompt B ×2
    { args: { rewards: [1, 0, 0, 0] }, pick: "B1", expect: 0.75, tol: 0.02, from: "lecture_16:kimi-loss-mean-baselined-pg:predict" },
    { args: { rewards: [1, 1, 0, 0, 0] }, pick: "B1", expect: 0.6, tol: 0.02, from: "lecture_16:kimi-loss-mean-baselined-pg:check" },
    { args: { rewards: [1, 1, 0, 0, 0] }, pick: "Z1", expect: 1.2247 },                         // ... where GRPO gives 1.22 (the check's contrast)
    { args: { rewards: [1, 0, 0, 0, 0, 0, 0, 0] }, pick: "B1", expect: 0.875 },                 // Kimi transfer: 0.875 and
    { args: { rewards: [1, 0, 0, 0, 0, 0, 0, 0] }, pick: "B2", expect: -0.125 },                //                −0.125
    { args: { rewards: [1, 1, 1, 0, 0] }, pick: "B4", expect: -0.6, tol: 0.02, from: "lecture_16:negative-gradients-vs-positive-only:check" },
    { args: { rewards: [1, 0, 0, 0] }, pick: "nzF", expect: 1 },                                // RFT transfer: 1 sample, positive
    { args: { rewards: [1, 0, 0, 0] }, pick: "negB", expect: 3 },                               // ... GRPO (mean-subtracted): 3 pushed down
    { args: { rewards: [1, 1, 1, 1, 1, 1, 1, 1] }, pick: "nzZ", expect: 0 },                    // difficulty predict: 8/8 solved, no contrast
    { args: { rewards: [0, 0, 0, 0, 0, 0, 0, 0] }, pick: "nzB", expect: 0 },                    // edge: all failed, also zero signal
    { args: { rewards: [1, 0, 0, 0, 0], ddof: 1 }, pick: "Z1", expect: 1.789 },                 // G − 1 convention (torch.std): +1.79, not +2.0
  ],
};
WIDGETS["fixture:lecture_16--group-weights"] = (root) => {
  const s = { text: "1, 1, 0, 1, 0, 0, 0", ddof: 0, eps: 0 };
  const pic = el("div"), read = el("div", { class: "readout" }), err = el("div", { class: "muted small" });
  const inp = el("input", { type: "text", value: s.text, style: "width:100%;max-width:420px" });
  const ROWS = [["R", "raw REINFORCE: r", "p5"], ["F", "RFT: 1 if correct", "p21, p48"], ["B", "mean baseline: r − mean", "p22-p23, Kimi"], ["Z", "GRPO: (r − mean)/std", "p18-p20"]];
  const draw = () => {
    const r = s.text.split(/[,\s]+/).filter(Boolean).map(Number);
    if (r.length < 2 || r.length > 16 || r.some(x => !Number.isFinite(x))) { err.textContent = "enter 2 to 16 numeric rewards separated by commas"; pic.innerHTML = ""; read.innerHTML = ""; return; }
    err.textContent = "";
    const m = groupWeights({ rewards: r, ddof: s.ddof, eps: s.eps });
    const amax = Math.max(1, ...m.rows.flatMap(x => [x.R, x.F, x.B, x.Z].map(Math.abs)));
    const x0 = 200, cw = Math.min(56, 440 / r.length), W = Math.max(660, x0 + r.length * cw + 20), rh = 76, half = 28;
    let b = text(10, 16, `${r.length} responses to one prompt; each bar is the weight on grad log p(o_i). All four rows share one scale.`, { fill: C.muted, size: 11 });
    r.forEach((v, i) => { b += text(x0 + i * cw + cw / 2, 36, `o${i + 1}`, { size: 11, anchor: "middle", fill: C.muted }) + text(x0 + i * cw + cw / 2, 50, `r=${fmt(v, 2)}`, { size: 11, anchor: "middle" }); });
    ROWS.forEach(([k, name, src], j) => {
      const y0 = 60 + j * rh + rh / 2;
      b += line(x0 - 6, y0, x0 + r.length * cw, y0, C.rule) + text(10, y0 - 4, name, { size: 12, weight: "bold" }) + text(10, y0 + 12, `${m[`nz${k}`]} of ${r.length} nonzero · ${src}`, { size: 10, fill: C.muted });
      m.rows.forEach((x, i) => {
        const v = x[k], h = (v / amax) * half, bx = x0 + i * cw + 4;
        if (Math.abs(v) > 1e-9) b += rect(bx, h > 0 ? y0 - h : y0, cw - 8, Math.abs(h), v > 0 ? C.ok : C.b);
        else b += rect(bx, y0 - 1, cw - 8, 2, "#cfc9bd");
        b += text(bx + (cw - 8) / 2, h >= 0 ? y0 - Math.abs(h) - 3 : y0 + Math.abs(h) + 11, Math.abs(v) > 1e-9 ? sgn(v, 2) : "0", { size: 10, anchor: "middle", fill: C.muted });
      });
    });
    pic.innerHTML = svg(W, 64 + ROWS.length * rh, b);
    read.innerHTML = `mean = ${fmt(m.mean, 4)} · std (÷ ${s.ddof ? "G − 1" : "G"}) = ${fmt(m.std, 4)}${s.eps ? ` + ${s.eps}` : ""}<br>
      ${m.uniform ? `<span class="big">all rewards equal: r − mean = 0 for every response, so the mean-baselined and GRPO rows are all 0: this prompt gives no signal</span>${m.nzR ? " (only raw REINFORCE and RFT still move it)" : ""}`
        : `<span class="big">GRPO row = mean-baseline row × ${fmt(m.scale, 3)}</span> (1/std, one factor for the whole prompt: it rescales this prompt's gradient, it is not subtracted, so it is not a baseline)`}<br>
      pushed down: raw ${m.negR} · RFT 0 (it never pushes down) · mean baseline ${m.negB} · GRPO ${m.negZ}<br>
      <span class="muted small">provenance: fixture:lecture_16--group-weights · lecture_16.pdf p5 (∇E[R] = E[R ∇log p]), p18 ("z-score within group"), p20 ("1e-4 stability factor"), p21/p48 (RFT, positives only), p22 ("subtract any state-dependent term"), p23 ("division by the stdev ... not a valid baseline"; Dr. GRPO), p24 ("upweights too easy or hard questions"); Kimi's r̄ baseline: video:45:55-46:15. "Correct" = reward > 0. std ÷ G is this thread's convention.</span>`;
  };
  inp.addEventListener("input", () => { s.text = String(inp.value ?? ""); draw(); });
  const presets = [["one success in 8", "1, 0, 0, 0, 0, 0, 0, 0"], ["±1 coding, 3 of 8 correct", "1, -1, -1, 1, -1, -1, 1, -1"], ["all solved (8/8)", "1, 1, 1, 1, 1, 1, 1, 1"], ["half solved", "1, 1, 0, 0"], ["graded", "0.9, 0.6, 0.2, 0.3"]];
  root.append(el("div", { class: "widget" }, el("div", { class: "controls" },
    el("label", {}, "rewards of one prompt's group ", inp), err,
    el("div", { style: "margin-top:6px" }, ...presets.map(([n, t]) => el("button", { onclick: () => { s.text = t; inp.value = t; draw(); } }, n)), " "),
    check("std with G − 1 (torch.std default) instead of G", false, v => { s.ddof = v ? 1 : 0; draw(); }),
    check("add the 1e-4 stability term to the std (p20)", false, v => { s.eps = v ? 1e-4 : 0; draw(); }),
    read), pic)); draw();
};

// ============================================================ 2. length lens ==
// One prompt's group of responses, each with a length |o_i| in tokens and a checker verdict (reward 1 correct, 0 wrong).
//   GRPO (p18, p24): every token of o_i gets weight Â_i / |o_i|, Â_i the group z-score; the per-response 1/|o_i| is the length normalizer.
//     "dividing longer sequences with a bigger number" (video:23:50); a sure-to-fail model can "divide by infinity" (video:24:14).
//   Dr. GRPO (p23-p24): Â_i = r_i − mean(r) (no std) and every token is divided by one constant instead of |o_i|. The constant's value
//     does not change the comparison; the slider's default 2,000 is a demo value (grpo-length-bias transfer: "a fixed maximum length").
//   Kimi k1.5 length reward (p43, image; formula recorded in the kimi-length-reward filtering_note): λ_i = 0.5 − (|o_i| − min)/(max − min),
//     correct responses receive λ_i, incorrect ones min(0, λ_i). All lengths equal leaves λ at 0/0; shown as 0.
function lengthLens({ resp, Lmax = 2000, ddof = 0 }) {
  const n = resp.length, r = resp.map(x => (x.ok ? 1 : 0)), mean = r.reduce((a, x) => a + x, 0) / n;
  const std = Math.sqrt(r.reduce((a, x) => a + (x - mean) ** 2, 0) / Math.max(1, n - ddof)), uniform = std < ZERO;
  const lens = resp.map(x => x.len), lo = Math.min(...lens), hi = Math.max(...lens), centre = (lo + hi) / 2;
  const o = { n, mean, std, uniform, lo, hi, centre, rows: [] };
  resp.forEach((x, i) => {
    const A = uniform ? 0 : (r[i] - mean) / std, Ad = r[i] - mean;
    const gw = A / x.len, dw = Ad / Lmax, lam = hi > lo ? 0.5 - (x.len - lo) / (hi - lo) : 0, pay = x.ok ? lam : Math.min(0, lam);
    Object.assign(o, { [`A${i + 1}`]: A, [`gw${i + 1}`]: gw, [`dw${i + 1}`]: dw, [`gtot${i + 1}`]: A, [`dtot${i + 1}`]: Ad * x.len / Lmax, [`lam${i + 1}`]: lam, [`pay${i + 1}`]: pay });
    o.rows.push({ ...x, A, Ad, gw, dw, lam, pay });
  });
  o.ratio12 = Math.abs(o.gw2) > ZERO ? o.gw1 / o.gw2 : NaN;                   // per-token GRPO weight of response 1 over response 2
  o.dratio12 = Math.abs(o.dw2) > ZERO ? o.dw1 / o.dw2 : NaN;                  // same under Dr. GRPO
  return o;
}
const LL = (spec) => spec.map(([len, ok]) => ({ len, ok }));
MODELS["fixture:lecture_16--length-lens"] = {
  fn: lengthLens,
  cases: [
    { args: { resp: LL([[300, 1], [1200, 1], [800, 0], [2000, 0]]) }, pick: "A1", expect: 1 },              // check's setup: both correct at Â = +1
    { args: { resp: LL([[300, 1], [1200, 1], [800, 0], [2000, 0]]) }, pick: "ratio12", expect: 4, tol: 0.01, from: "lecture_16:grpo-length-bias:check" },
    { args: { resp: LL([[300, 1], [1200, 1], [800, 0], [2000, 0]]) }, pick: "dratio12", expect: 1 },        // Dr. GRPO: equal per-token weight
    { args: { resp: LL([[100, 0], [1000, 0], [500, 1], [700, 1]]) }, pick: "gw1", expect: -0.01 },          // predict: 100-token wrong, −1/100
    { args: { resp: LL([[100, 0], [1000, 0], [500, 1], [700, 1]]) }, pick: "gw2", expect: -0.001 },         //          1,000-token wrong, −1/1000
    { args: { resp: LL([[100, 0], [1000, 0], [500, 1], [700, 1]]) }, pick: "gtot2", expect: -1 },           // GRPO total push is −1 at any length
    { args: { resp: LL([[100, 0], [1000, 0], [500, 1], [700, 1]]), Lmax: 2000 }, pick: "dtot2", expect: -0.25 }, // Dr. GRPO: grows with length
    { args: { resp: LL([[200, 0], [1500, 0], [600, 1], [900, 1]]) }, pick: "pay1", expect: 0, tol: 1e-9 },  // Kimi predict: shortest wrong gets 0
    { args: { resp: LL([[200, 0], [1500, 0], [600, 1], [900, 1]]) }, pick: "lam1", expect: 0.5 },           //   (its λ is +0.5, withheld)
    { args: { resp: LL([[200, 0], [1500, 0], [600, 1], [900, 1]]) }, pick: "pay2", expect: -0.5 },          //   longest wrong gets −0.5
    { args: { resp: LL([[200, 1], [1500, 1], [600, 0], [900, 0]]) }, pick: "pay1", expect: 0.5 },           // Kimi transfer: shortest correct +0.5
    { args: { resp: LL([[200, 1], [1500, 1], [600, 0], [900, 0]]) }, pick: "pay2", expect: -0.5 },          //                longest correct −0.5
    { args: { resp: LL([[200, 1], [1000, 1], [500, 0], [800, 0]]) }, pick: "pay3", expect: 0, tol: 1e-9 },  // Kimi check: wrong below the centre (600): 0
    { args: { resp: LL([[200, 1], [1000, 1], [500, 0], [800, 0]]) }, pick: "pay4", expect: -0.25 },         //             wrong above the centre: < 0
    { args: { resp: LL([[400, 1], [400, 1], [400, 0]]) }, pick: "lam1", expect: 0, tol: 1e-9 },             // edge: equal lengths, λ shown as 0
    { args: { resp: LL([[300, 1], [900, 1]]) }, pick: "gw1", expect: 0, tol: 1e-9 },                        // edge: uniform group, no weight at all
  ],
};
WIDGETS["fixture:lecture_16--length-lens"] = (root) => {
  const s = { text: "400c, 900c, 700w, 1600w", Lmax: 2000 };
  const pic = el("div"), read = el("div", { class: "readout" }), err = el("div", { class: "muted small" });
  const inp = el("input", { type: "text", value: s.text, style: "width:100%;max-width:420px" });
  const parse = () => {
    const toks = s.text.split(",").map(t => t.trim()).filter(Boolean), out = [];
    for (const t of toks) { const m = /^(\d+(?:\.\d+)?)\s*([cw])$/i.exec(t); if (!m || +m[1] < 1) return null; out.push({ len: +m[1], ok: m[2].toLowerCase() === "c" }); }
    return out.length >= 2 && out.length <= 12 ? out : null;
  };
  const draw = () => {
    const resp = parse();
    if (!resp) { err.textContent = "enter 2 to 12 responses as <length><c|w>, e.g. 300c, 1200w (c = correct, w = wrong)"; pic.innerHTML = ""; read.innerHTML = ""; return; }
    err.textContent = "";
    const m = lengthLens({ resp, Lmax: s.Lmax });
    const W = 660, L = 70, R = 630, xmax = m.hi * 1.08, X = v => L + (v / xmax) * (R - L);
    // top panel: per-token weight (×10³) against length
    const T1 = 46, B1 = 196, ymax = 1.15 * Math.max(...m.rows.map(x => Math.abs(x.gw) * 1e3), ...m.rows.map(x => Math.abs(x.dw) * 1e3), 0.05);
    const Y1 = v => T1 + (ymax - Math.max(-ymax, Math.min(ymax, v))) / (2 * ymax) * (B1 - T1);
    let b = text(10, 16, "top: per-token weight on grad log p (×10³) · bottom: Kimi length reward · x: response length in tokens", { fill: C.muted, size: 11 });
    b += line(L, Y1(0), R, Y1(0), C.rule) + text(L - 6, Y1(0) + 4, "0", { anchor: "end", size: 10, fill: C.muted });
    b += text(L - 6, T1 + 8, sgn(ymax, 2), { anchor: "end", size: 10, fill: C.muted }) + text(L - 6, B1, sgn(-ymax, 2), { anchor: "end", size: 10, fill: C.muted });
    const xs0 = Math.max(10, xmax * 0.03);
    for (const Aval of [...new Set(m.rows.map(x => +x.A.toFixed(9)))].filter(a => Math.abs(a) > 1e-9)) {
      const vStart = Math.max(xs0, (Math.abs(Aval) * 1e3) / ymax), pts = []; for (let v = vStart; v <= xmax; v += xmax / 300) pts.push([X(v), Y1((Aval / v) * 1e3)]);
      b += path(pts, Aval > 0 ? C.ok : C.b, 'stroke-opacity="0.55"');
    }
    for (const Ad of [...new Set(m.rows.map(x => +x.Ad.toFixed(9)))].filter(a => Math.abs(a) > 1e-9)) b += line(L, Y1((Ad / s.Lmax) * 1e3), R, Y1((Ad / s.Lmax) * 1e3), Ad > 0 ? C.ok : C.b, 'stroke-dasharray="6 4"');
    m.rows.forEach((x, i) => {
      b += dot(X(x.len), Y1(x.gw * 1e3), 5, x.ok ? C.ok : C.b) + text(X(x.len) + 7, Y1(x.gw * 1e3) + (x.gw >= 0 ? -4 : 12), `o${i + 1}`, { size: 10 });
      b += dot(X(x.len), Y1(x.dw * 1e3), 4, "#fff", `stroke="${x.ok ? "var(--ok)" : "var(--accent2)"}" stroke-width="2"`);
    });
    b += text(R, 32, "solid curve + filled dot: GRPO Â/|o| · dashed line + open dot: Dr. GRPO (r − mean)/constant", { size: 10, anchor: "end", fill: C.muted });
    // bottom panel: Kimi λ
    const T2 = 236, B2 = 340, Y2 = v => T2 + (0.6 - v) / 1.2 * (B2 - T2);
    b += line(L, Y2(0), R, Y2(0), C.rule) + text(L - 6, Y2(0.5) + 4, "+0.5", { anchor: "end", size: 10, fill: C.muted }) + text(L - 6, Y2(-0.5) + 4, "−0.5", { anchor: "end", size: 10, fill: C.muted }) + text(L - 6, Y2(0) + 4, "0", { anchor: "end", size: 10, fill: C.muted });
    if (m.hi > m.lo) {
      b += line(X(m.lo), Y2(0.5), X(m.hi), Y2(-0.5), C.ok, 'stroke-opacity="0.6"') + path([[X(m.lo), Y2(0)], [X(m.centre), Y2(0)], [X(m.hi), Y2(-0.5)]], C.b, 'stroke-dasharray="5 3" stroke-opacity="0.8"');
      b += line(X(m.centre), T2, X(m.centre), B2, C.muted, 'stroke-dasharray="2 3"') + text(X(m.centre) + 4, T2 + 10, "centre of the range", { size: 10, fill: C.muted });
    }
    m.rows.forEach((x, i) => { b += dot(X(x.len), Y2(x.pay), 5, x.ok ? C.ok : C.b) + text(X(x.len) + 7, Y2(x.pay) - 5, `o${i + 1}`, { size: 10 }); });
    b += text(R, T2 - 8, "green line: λ paid to correct · red dashed: min(0, λ) paid to wrong", { size: 10, anchor: "end", fill: C.muted });
    for (let v = 0; v <= xmax; v += xmax > 3000 ? 1000 : 500) b += text(X(v), B2 + 16, fmt(v, 0), { size: 10, anchor: "middle", fill: C.muted });
    pic.innerHTML = svg(W, B2 + 26, b);
    const rows = m.rows.map((x, i) => `<tr><td>o${i + 1}</td><td>${x.len}</td><td>${x.ok ? "correct" : "wrong"}</td><td>${sgn(x.A, 3)}</td><td>${sgn(x.gw * 1e3, 3)}</td><td>${sgn(x.dw * 1e3, 3)}</td><td>${sgn(x.pay, 3)}${!x.ok && x.lam > 1e-9 ? ` (λ = ${sgn(x.lam, 2)} withheld)` : ""}</td></tr>`).join("");
    const g1 = m.rows[0], g2 = m.rows[1];
    read.innerHTML = `<table class="small"><tr><th></th><th>|o|</th><th></th><th>GRPO Â</th><th>GRPO per token ×10³</th><th>Dr. GRPO per token ×10³</th><th>Kimi length reward</th></tr>${rows}</table>
      ${m.uniform ? `<b>all responses ${g1.ok ? "correct" : "wrong"}: every Â is 0, no length pressure from the task reward</b><br>` : ""}
      <span class="big">o1 vs o2 per-token weight: GRPO ${Number.isFinite(m.ratio12) ? `${fmt(m.ratio12, 3)}×` : "–"} · Dr. GRPO ${Number.isFinite(m.dratio12) ? `${fmt(m.dratio12, 3)}×` : "–"}</span>
      ${g1.ok === g2.ok && !m.uniform ? ` (same verdict, so same Â; GRPO's ratio is just |o2|/|o1| = ${fmt(g2.len / g1.len, 3)})` : ""}<br>
      GRPO: a response's total push Σ<sub>t</sub> Â/|o| = Â whatever its length, so a long wrong answer spreads the same penalty over more tokens; Dr. GRPO's total (r − mean)·|o|/constant grows with length<br>
      <span class="muted small">provenance: fixture:lecture_16--length-lens · lecture_16.pdf p18 (GRPO objective, 1/|o_i|), p23 (Dr. GRPO, "modification on the length-normalizer term"), p24 ("Length biases of GRPO"), p43 (λ in [0.5, −0.5], correct short, incorrect "shorter than the center of the range"; formula image, read in the KP's filtering_note); video:23:50-24:29 ("divide by infinity"). The Dr. GRPO constant is a slider (default 2,000 is a demo value). Rewards: 1 correct, 0 wrong; std ÷ G.</span>`;
  };
  inp.addEventListener("input", () => { s.text = String(inp.value ?? ""); draw(); });
  const presets = [["two wrong, 10× length apart", "150w, 1500w, 600c, 800c"], ["two correct, 4× apart", "250c, 1000c, 600w, 1800w"], ["mostly wrong", "300c, 900w, 1400w, 2200w, 3000w"], ["all wrong", "500w, 1500w, 2500w"]];
  root.append(el("div", { class: "widget" }, el("div", { class: "controls" },
    el("label", {}, "responses (length + c/w) ", inp), err,
    el("div", { style: "margin-top:6px" }, ...presets.map(([n, t]) => el("button", { onclick: () => { s.text = t; inp.value = t; draw(); } }, n)), " "),
    slider("Dr. GRPO's constant divisor (tokens)", 500, 8000, s.Lmax, 100, v => { s.Lmax = v; draw(); }, v => fmt(v, 0)),
    read), pic)); draw();
};

// ============================================================ 3. length drift ==
// The batch-level consequence of the length normalizer (video:38:41-39:58): under GRPO correct responses are pushed shorter but "there's a
// lower bound to how small your COT can go to solve a particular problem", incorrect ones are pushed longer with no ceiling, so the mean
// "is really being driven by the incorrect ones" (Dr. GRPO's split plot, p30). Model, per training stretch j (the check's own framing):
//   correct length  Lc(j) = max(floor, c0 − s·j)      incorrect length  Li(j) = i0 + g·j      mean(j) = q·Li(j) + (1 − q)·Lc(j)
// q = share of incorrect responses. The per-stretch magnitudes g = 300 and s = 50 are the check's; floor 200 and i0 = 2,000 are the transfer's;
// c0 = 300 is a demo value. Dr. GRPO removes the length preference from the objective ("cap off at a constant", video:24:33-24:49), so its
// objective-driven drift is drawn as 0; length changes the task itself causes are not modelled.
function lengthDrift({ q, g, s, c0, floor, i0, k = 1, obj = "grpo" }) {
  const dr = obj === "dr", fl = Math.min(floor, c0);                          // a floor above the start means "already at the floor"
  const Lc = j => (dr ? c0 : Math.max(fl, c0 - s * j)), Li = j => (dr ? i0 : i0 + g * j), mean = j => q * Li(j) + (1 - q) * Lc(j);
  const room = c0 - fl, shrink1 = dr ? 0 : Math.min(s, room), grow1 = dr ? 0 : g;
  const series = Array.from({ length: 11 }, (_, j) => ({ j, c: Lc(j), i: Li(j), m: mean(j) }));
  const floorAt = dr ? null : s > 0 ? Math.ceil(room / s) : (room > 0 ? null : 0);
  return { incPart: q * grow1, corPart: -(1 - q) * shrink1, dmean1: mean(1) - mean(0), dmeanK: mean(k) - mean(0), meanK: mean(k), cK: Lc(k), iK: Li(k), floorAt, series };
}
MODELS["fixture:lecture_16--length-drift"] = {
  fn: lengthDrift,
  cases: [
    { args: { q: 0.6, g: 300, s: 50, c0: 300, floor: 200, i0: 2000 }, pick: "dmean1", expect: 160, tol: 0.02, from: "lecture_16:incorrect-responses-drive-length-growth:check" },
    { args: { q: 0.6, g: 300, s: 50, c0: 300, floor: 200, i0: 2000 }, pick: "corPart", expect: -20 },      // the correct share's −0.4 × 50
    { args: { q: 0.5, g: 300, s: 50, c0: 200, floor: 200, i0: 2000 }, pick: "corPart", expect: 0, tol: 1e-9 }, // transfer: correct at its floor
    { args: { q: 0.5, g: 300, s: 50, c0: 200, floor: 200, i0: 2000 }, pick: "dmean1", expect: 150 },       //   ... mean rises, carried by the wrong half
    { args: { q: 0.8, g: 300, s: 50, c0: 300, floor: 200, i0: 2000 }, pick: "dmean1", expect: 230 },       // predict run B (80% wrong): +230
    { args: { q: 0.1, g: 300, s: 50, c0: 300, floor: 200, i0: 2000 }, pick: "dmean1", expect: -15 },       // predict run A (10% wrong): −15 at first,
    { args: { q: 0.1, g: 300, s: 50, c0: 300, floor: 200, i0: 2000, k: 4 }, pick: "dmeanK", expect: 30 },  //   +30 after 4 stretches (floor hit at 2)
    { args: { q: 0.1, g: 300, s: 50, c0: 300, floor: 200, i0: 2000 }, pick: "floorAt", expect: 2 },
    { args: { q: 0.6, g: 300, s: 50, c0: 300, floor: 200, i0: 2000, obj: "dr" }, pick: "dmean1", expect: 0, tol: 1e-9 }, // Dr. GRPO: no objective drift
  ],
};
WIDGETS["fixture:lecture_16--length-drift"] = (root) => {
  const s = { q: 0.5, g: 300, s: 50, c0: 300, floor: 200, i0: 2000, k: 4, obj: "grpo" };
  const pic = el("div"), read = el("div", { class: "readout" });
  const draw = () => {
    const m = lengthDrift(s), W = 640, L = 64, R = 520, T = 26, B = 220;
    const ymax = Math.max(...m.series.map(p => Math.max(p.i, p.c, p.m))) * 1.08, X = j => L + (j / 10) * (R - L), Y = v => B - (v / ymax) * (B - T);
    let b = text(10, 16, `mean response length (tokens) over training stretches · ${s.obj === "dr" ? "Dr. GRPO: objective-driven drift removed" : "GRPO with per-response length normalization"}`, { fill: C.muted, size: 11 });
    b += line(L, B, R, B, C.rule) + line(L, T, L, B, C.rule);
    for (let v = 0; v <= ymax; v += ymax > 5000 ? 1000 : 500) b += text(L - 6, Y(v) + 4, fmt(v, 0), { size: 10, anchor: "end", fill: C.muted });
    for (let j = 0; j <= 10; j += 2) b += text(X(j), B + 14, String(j), { size: 10, anchor: "middle", fill: C.muted });
    b += text(R, B + 28, "training stretch", { size: 10, anchor: "end", fill: C.muted });
    b += line(L, Y(s.floor), R, Y(s.floor), C.ok, 'stroke-dasharray="2 3" stroke-opacity="0.7"') + text(R + 4, Y(s.floor) + 12, "floor (correct)", { size: 10, fill: C.ok });
    b += line(X(s.k), T, X(s.k), B, C.hi, 'stroke-width="3" stroke-opacity="0.5"');
    const ser = (key, col, w) => path(m.series.map(p => [X(p.j), Y(p[key])]), col, `stroke-width="${w}"`);
    b += ser("i", C.b, 2) + ser("c", C.ok, 2) + ser("m", C.ink, 3);
    const last = m.series[10];
    b += text(R + 4, Y(last.i) + 4, "incorrect", { size: 11, fill: C.b }) + text(R + 4, Y(last.c) - 4, "correct", { size: 11, fill: C.ok }) + text(R + 4, Y(last.m) + 4, "batch mean", { size: 11, weight: "bold" });
    pic.innerHTML = svg(W, B + 36, b);
    const atFloor = s.obj !== "dr" && (s.c0 - s.s * s.k <= Math.min(s.floor, s.c0));
    read.innerHTML = `first stretch: Δmean = q·g − (1 − q)·min(s, room above floor) = <b>${sgn(m.incPart, 1)}</b> (incorrect) <b>${m.corPart < 0 ? "−" : "+"} ${fmt(Math.abs(m.corPart), 1)}</b> (correct) = <span class="big">${sgn(m.dmean1, 1)} tokens</span><br>
      after ${s.k} stretch${s.k === 1 ? "" : "es"}: mean ${fmt(m.meanK, 0)} tokens (${sgn(m.dmeanK, 0)}) · correct ${fmt(m.cK, 0)}, incorrect ${fmt(m.iK, 0)}<br>
      ${s.obj === "dr" ? "<b>Dr. GRPO</b>: no length preference in the objective, so neither group drifts" : atFloor ? `<b>correct responses at their floor</b>${m.floorAt !== null ? ` (from stretch ${m.floorAt})` : ""}: they can no longer respond, only the incorrect ones move` : `<b>correct responses still shrinking</b>: they reach the floor at stretch ${m.floorAt ?? "–"}, after which only the incorrect ones move`}<br>
      <span class="muted small">provenance: fixture:lecture_16--length-drift · video:38:41-39:58 ("a lower bound to how small your COT can go"; Dr. GRPO's plot "really being driven by the incorrect ones"), video:24:33-24:49 (fixed objective: length "cap[s] off"), lecture_16.pdf p30 (Dr. GRPO's length plots, image). Linear per-stretch drift is the prompts' framing, not a measured curve: g = 300 and s = 50 from the check, floor 200 and incorrect start 2,000 from the transfer, correct start 300 is a demo value.</span>`;
  };
  const set = (k, v) => { s[k] = v; draw(); };
  const objSel = el("select", {}, el("option", { value: "grpo" }, "GRPO (per-response length normalizer)"), el("option", { value: "dr" }, "Dr. GRPO (constant normalizer)"));
  objSel.addEventListener("change", () => set("obj", objSel.value));
  root.append(el("div", { class: "widget" }, el("div", { class: "controls" },
    el("label", {}, "objective ", objSel),
    slider("share of responses that are incorrect q", 0, 1, s.q, 0.05, v => set("q", v), v => v.toFixed(2)),
    slider("incorrect: growth per stretch g (tokens)", 0, 600, s.g, 10, v => set("g", v), v => fmt(v, 0)),
    slider("correct: shrink per stretch s (tokens)", 0, 200, s.s, 10, v => set("s", v), v => fmt(v, 0)),
    slider("correct: starting length", 50, 1500, s.c0, 50, v => set("c0", v), v => fmt(v, 0)),
    slider("correct: floor (shortest CoT that still solves it)", 50, 1500, s.floor, 50, v => set("floor", v), v => fmt(v, 0)),
    slider("stretches of training k", 0, 10, s.k, 1, v => set("k", v)),
    read), pic)); draw();
};

// ======================================================= 4. difficulty signal ==
// A prompt the current model solves with per-sample probability p, sampled G times (binary rewards). The group gives GRPO no signal exactly
// when all G rewards are equal ("if your problems are too hard, you get no rewards ... no signal", video:42:02-42:10; an all-solved group
// has every advantage 0, grpo-group-zscore-advantage): P(no signal) = p^G + (1 − p)^G. The expected contrast is E[std of the group's
// rewards] = Σ_k C(G,k) p^k (1 − p)^(G−k) √(k/G · (1 − k/G)), which peaks at medium difficulty ("filter on both sides", video:43:26-43:44).
// Kimi's filter (p41): "Select only examples that models fail on best-of-8": survival = (1 − p)^8 (the k1.5 paper applies it to no-CoT
// guesses, see the KP's filtering_note). Kimi's curriculum (p44): "Sample problems proportional to (1-success_rate)".
function difficultySignal({ p, G = 8, pP = 0.8, pQ = 0.4 }) {
  let Estd = 0, c = 1;
  for (let k = 0; k <= G; k++) { if (k) c = c * (G - k + 1) / k; const f = k / G; Estd += c * p ** k * (1 - p) ** (G - k) * Math.sqrt(f * (1 - f)); }
  const pUniform = p ** G + (1 - p) ** G;
  return { pUniform, pAllSolved: p ** G, pAllFailed: (1 - p) ** G, Estd, survive8: (1 - p) ** 8, weight: 1 - p, ratioQP: pP < 1 ? (1 - pQ) / (1 - pP) : Infinity };
}
MODELS["fixture:lecture_16--difficulty-signal"] = {
  fn: difficultySignal,
  cases: [
    { args: { p: 0.5, pP: 0.9, pQ: 0.3 }, pick: "ratioQP", expect: 7, tol: 0.01, from: "lecture_16:difficulty-filtering-curriculum:transfer" },
    { args: { p: 1, G: 8 }, pick: "pUniform", expect: 1 },                 // predict: solved 8/8, the group is certainly uniform
    { args: { p: 1, G: 8 }, pick: "Estd", expect: 0, tol: 1e-9 },          //   ... and carries no contrast
    { args: { p: 0, G: 8 }, pick: "pUniform", expect: 1 },                 // self-explain: always failed wastes rollouts the same way
    { args: { p: 0.5, G: 8 }, pick: "pUniform", expect: 0.0078125 },      // medium difficulty: 2/256
    { args: { p: 0.5, G: 2 }, pick: "Estd", expect: 0.25 },                // G = 2: half the time {0,1} with std 0.5
    { args: { p: 0.5 }, pick: "survive8", expect: 0.00390625 },           // best-of-8 filter keeps 1/256 of p = 0.5 problems
    { args: { p: 0.3 }, pick: "weight", expect: 0.7 },
  ],
};
WIDGETS["fixture:lecture_16--difficulty-signal"] = (root) => {
  const s = { p: 0.6, G: 8, pP: 0.8, pQ: 0.4 };
  const pic = el("div"), read = el("div", { class: "readout" });
  const draw = () => {
    const m = difficultySignal(s), W = 640, L = 50, R = 610, T = 26, B = 220, X = p => L + p * (R - L), Y = v => B - v * (B - T);
    let b = text(10, 16, `x: the model's success rate p on this prompt · G = ${s.G} samples per group`, { fill: C.muted, size: 11 });
    b += line(L, B, R, B, C.rule) + line(L, T, L, B, C.rule);
    for (const v of [0, 0.5, 1]) b += text(L - 6, Y(v) + 4, String(v), { size: 10, anchor: "end", fill: C.muted }) + text(X(v), B + 14, String(v), { size: 10, anchor: "middle", fill: C.muted });
    const curve = (f, col, extra = "") => { const pts = []; for (let i = 0; i <= 200; i++) { const p = i / 200; pts.push([X(p), Y(f(p))]); } return path(pts, col, extra); };
    b += curve(p => difficultySignal({ p, G: s.G }).pUniform, C.b) + curve(p => 2 * difficultySignal({ p, G: s.G }).Estd, C.ok);
    b += curve(p => 1 - p, C.a, 'stroke-dasharray="6 4"') + curve(p => (1 - p) ** 8, C.muted, 'stroke-dasharray="2 3"');
    b += line(X(s.p), T, X(s.p), B, C.hi, 'stroke-width="3" stroke-opacity="0.6"') + dot(X(s.p), Y(m.pUniform), 5, C.b) + dot(X(s.p), Y(2 * m.Estd), 5, C.ok);
    const lg = [[C.b, "", "P(all G rewards equal): zero GRPO signal"], [C.ok, "", "expected group std × 2 (contrast)"], [C.a, 'stroke-dasharray="6 4"', "Kimi curriculum weight 1 − p"], [C.muted, 'stroke-dasharray="2 3"', "survives the best-of-8 filter (1 − p)^8, p = no-CoT guess rate"]];
    lg.forEach(([col, ex, t], i) => { const lx = L + (i % 2) * 290, ly = B + 34 + Math.floor(i / 2) * 18; b += line(lx, ly - 4, lx + 20, ly - 4, col, ex) + text(lx + 26, ly, t, { size: 10 }); });
    pic.innerHTML = svg(W, B + 70, b);
    const regime = m.pUniform > 0.5 ? (s.p > 0.5 ? "too easy: most groups are all solved" : "too hard: most groups are all failed") : "medium difficulty: most groups have contrast";
    read.innerHTML = `p = ${s.p.toFixed(2)}: P(all ${s.G} solved) = ${fmt(m.pAllSolved, 4)}, P(all failed) = ${fmt(m.pAllFailed, 4)} → <span class="big">P(no signal) = ${fmt(m.pUniform, 4)}</span> · expected std ${fmt(m.Estd, 3)} · <b>${regime}</b><br>
      best-of-8 filter (p41, as specified in the k1.5 paper: drop a prompt the model guesses within 8 tries without CoT; here p is read as that no-CoT guess rate): survives with probability ${fmt(m.survive8, 4)} · Kimi curriculum (p44): sampled ∝ 1 − p = ${fmt(m.weight, 2)}<br>
      two problems: P at success ${s.pP.toFixed(2)}, Q at ${s.pQ.toFixed(2)} → <span class="big">Q is sampled ${Number.isFinite(m.ratioQP) ? fmt(m.ratioQP, 3) : "∞"}× as often as P</span> (${fmt(1 - s.pQ, 2)} / ${fmt(1 - s.pP, 2)})<br>
      <span class="muted small">provenance: fixture:lecture_16--difficulty-signal · lecture_16.pdf p41 ("Select only examples that models fail on best-of-8"), p44 ("Sample problems proportional to (1-success_rate)"), p51 (Qwen 3: filtering by best-of-n, 3,995 examples); video:42:02-42:10 ("no rewards ... no signal"), 43:26-43:44 ("filter on both sides"). Binary rewards with independent samples are the widget's assumption.</span>`;
  };
  const set = (k, v) => { s[k] = v; draw(); };
  root.append(el("div", { class: "widget" }, el("div", { class: "controls" },
    slider("success rate p of this prompt", 0, 1, s.p, 0.01, v => set("p", v), v => v.toFixed(2)),
    slider("group size G", 2, 16, s.G, 1, v => set("G", v)),
    slider("problem P success rate", 0, 0.99, s.pP, 0.01, v => set("pP", v), v => v.toFixed(2)),
    slider("problem Q success rate", 0, 0.99, s.pQ, 0.01, v => set("pQ", v), v => v.toFixed(2)),
    read), pic)); draw();
};

// ========================================================== 5. rollout budget ==
// p45 "Why is RL hard to make efficient? On policy = rollouts, which means (slow) inference; switching from training to rollouts often
// means different frameworks; long CoTs can make batches very uneven." Two pictures of the same loop:
//   (a) schedule: synchronous on-policy loop = rollout R, then train T, then weight sync S (nothing trains during S, the predict's framing);
//       training-GPU busy fraction = T / (R + T + S). Overlap: rollouts for step e+1 run while step e trains, period max(R, T) + S, but each
//       rollout comes from weights one update old: "reuse these rollouts ... will lead to off-policy problems" (video:53:09-53:31).
//   (b) one rollout batch decoded together "waiting on this one rollout to complete" (video:52:05-52:12): it runs to its longest sequence,
//       so the share of generation slots holding real tokens = Σ len / (B · max len) (the check's framing).
// Times and lengths are the prompts' (4, 3, 0.5; 7 × 1,000 and 1 × 8,000); the slides print no GPU counts or throughputs (p46 is an image).
function rolloutBudget({ roll, train, sync, overlap = false, B = 8, nLong = 1, short = 1000, long = 8000 }) {
  const period = overlap ? Math.max(roll, train) + sync : roll + train + sync;
  const nl = Math.min(B, Math.max(0, nLong)), maxLen = nl > 0 ? Math.max(short, long) : short;
  const real = (B - nl) * short + nl * long;
  return { period, trainBusy: train / period, rollBusy: roll / period, idleTrain: 1 - train / period, staleness: overlap ? 1 : 0,
    fill: real / (B * maxLen), real, slots: B * maxLen };
}
MODELS["fixture:lecture_16--rollout-budget"] = {
  fn: rolloutBudget,
  cases: [
    { args: { roll: 4, train: 3, sync: 0.5 }, pick: "trainBusy", expect: 0.4, tol: 0.05, from: "lecture_16:rl-rollout-cost:predict" },
    { args: { roll: 4, train: 3, sync: 0 }, pick: "trainBusy", expect: 0.4286 },                         // without the sync the share is 3/7
    { args: { roll: 4, train: 3, sync: 0.5, overlap: true }, pick: "trainBusy", expect: 0.6667 },        // overlap: 3 / (4 + 0.5), one update stale
    { args: { roll: 2, train: 3, sync: 0.5, overlap: true }, pick: "trainBusy", expect: 0.8571 },        // edge: training is the longer phase
    { args: { roll: 4, train: 3, sync: 0.5, B: 8, nLong: 1, short: 1000, long: 8000 }, pick: "fill", expect: 0.234, tol: 0.03, from: "lecture_16:rl-rollout-cost:check" },
    { args: { roll: 4, train: 3, sync: 0.5, B: 8, nLong: 0, short: 1000, long: 8000 }, pick: "fill", expect: 1 },   // even batch: no waste
    { args: { roll: 4, train: 3, sync: 0.5, B: 8, nLong: 8, short: 1000, long: 8000 }, pick: "fill", expect: 1 },   // all long: no waste either
  ],
};
WIDGETS["fixture:lecture_16--rollout-budget"] = (root) => {
  const s = { roll: 6, train: 3, sync: 0.5, overlap: false, B: 8, nLong: 1, short: 1000, long: 4000 };
  const pic = el("div"), read = el("div", { class: "readout" });
  const draw = () => {
    const m = rolloutBudget(s), W = 640, L = 110, R = 620, steps = s.overlap ? 4 : 3;
    const tot = steps * m.period, px = (R - L) / tot, P = m.period;
    let b = text(10, 16, `(a) ${s.overlap ? "overlapped: batch e is generated while batch e−1 trains (period 0 fills the pipeline)" : "synchronous: rollout, train, sync, repeat; each pool waits for the other"}`, { fill: C.muted, size: 11 });
    const lane = (y, name) => text(10, y + 15, name, { size: 11 });
    b += lane(30, "rollout GPUs") + lane(66, "training GPUs");
    const bar = (y, t0, len, fill, lab) => rect(L + t0 * px, y, len * px - 1, 22, fill) + (lab && len * px > lab.length * 5.8 + 8 ? text(L + t0 * px + 4, y + 15, lab, { size: 10, fill: "#fff" }) : "");
    b += rect(L, 30, R - L, 22, "#f1eee7") + rect(L, 66, R - L, 22, "#f1eee7");
    for (let e = 0; e < steps; e++) {
      const p0 = e * P, ys = p0 + P - s.sync;
      if (s.overlap) { b += bar(30, p0, s.roll, C.b, `b${e} from w${Math.max(0, e - 1)}`); if (e) b += bar(66, p0, s.train, C.a, `train b${e - 1} → w${e}`); }
      else b += bar(30, p0, s.roll, C.b, `b${e} from w${e}`) + bar(66, p0 + s.roll, s.train, C.a, `train b${e} → w${e + 1}`);
      b += bar(30, ys, s.sync, C.hi) + bar(66, ys, s.sync, C.hi);
    }
    b += text(R, 106, `grey = idle · yellow = weight sync · time axis spans ${fmt(tot, 1)} units`, { size: 10, anchor: "end", fill: C.muted });
    // (b) uneven batch
    const yb = 130, rowH = Math.min(14, 110 / s.B), maxLen = Math.max(s.short, s.nLong > 0 ? s.long : 0);
    b += text(10, yb, `(b) one rollout batch of ${s.B} sequences decoded together: it ends when the longest one does`, { fill: C.muted, size: 11 });
    for (let i = 0; i < s.B; i++) {
      const len = i >= s.B - s.nLong ? s.long : s.short, y = yb + 10 + i * rowH;
      b += rect(L, y, R - L, rowH - 2, "#f1eee7") + rect(L, y, (len / maxLen) * (R - L), rowH - 2, len === s.long && s.nLong ? C.b : C.a, 'fill-opacity="0.75"');
    }
    b += text(10, yb + 10 + (s.B * rowH) / 2 + 4, "sequences", { size: 11 });
    const Hb = yb + 10 + s.B * rowH + 18;
    b += text(R, Hb - 4, `filled = real tokens · grey = slots spent waiting (padding) · width = ${fmt(maxLen, 0)} decode steps`, { size: 10, anchor: "end", fill: C.muted });
    pic.innerHTML = svg(W, Hb + 4, b);
    const bound = s.roll > s.train ? "rollout-bound: inference is the longer phase" : s.roll < s.train ? "training-bound" : "balanced";
    read.innerHTML = `(a) <span class="big">training GPUs busy ${fmt(100 * m.trainBusy, 1)}%</span> = ${fmt(s.train, 2)} / ${s.overlap ? `(max(${fmt(s.roll, 2)}, ${fmt(s.train, 2)}) + ${fmt(s.sync, 2)})` : `(${fmt(s.roll, 2)} + ${fmt(s.train, 2)} + ${fmt(s.sync, 2)})`} · rollout GPUs busy ${fmt(100 * m.rollBusy, 1)}% · <b>${bound}</b><br>
      ${s.overlap ? "<b>overlap</b>: utilization goes up, but each step trains on rollouts from the previous weights (off-policy by one update), the trade-off of video:52:51-53:31" : "<b>synchronous, on-policy</b>: every rollout comes from the current weights; each pool idles while the other works"}<br>
      (b) <span class="big">${fmt(100 * m.fill, 1)}% of generation slots hold real tokens</span> (${fmt(m.real, 0)} / ${fmt(m.slots, 0)})<br>
      <span class="muted small">provenance: fixture:lecture_16--rollout-budget · lecture_16.pdf p45 (three inefficiencies), p46 (Kimi's RL infrastructure, image); video:51:42-52:17 (the Riemann-hypothesis rollout everyone waits on), 52:24-52:51 (separate machines vs switching frameworks), 52:51-53:31 (reusing rollouts goes off-policy). Times are abstract units; the predict uses 4 / 3 / 0.5 and the check 7 × 1,000 + 1 × 8,000 tokens.</span>`;
  };
  const set = (k, v) => { s[k] = v; draw(); };
  root.append(el("div", { class: "widget" }, el("div", { class: "controls" },
    slider("rollout time R", 0.5, 10, s.roll, 0.5, v => set("roll", v), v => fmt(v, 1)),
    slider("training time T", 0.5, 10, s.train, 0.5, v => set("train", v), v => fmt(v, 1)),
    slider("weight sync S", 0, 3, s.sync, 0.25, v => set("sync", v), v => fmt(v, 2)),
    check("overlap rollouts with training (reuse rollouts from the previous weights)", false, v => set("overlap", v)),
    slider("batch size B", 2, 16, s.B, 1, v => { s.B = v; s.nLong = Math.min(s.nLong, v); draw(); }),
    slider("long sequences in the batch", 0, 16, s.nLong, 1, v => set("nLong", Math.min(v, s.B))),
    slider("short length (tokens)", 100, 4000, s.short, 100, v => set("short", v), v => fmt(v, 0)),
    slider("long length (tokens)", 100, 16000, s.long, 100, v => set("long", v), v => fmt(v, 0)),
    read), pic)); draw();
};
