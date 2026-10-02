// Widgets for thread lecture_15 (CS336 L15, post-training: SFT, RLHF data, reward models, PPO, DPO). Register as "fixture:<id>" -> (root, notice) => void.
// Each widget has a pure model in MODELS (no DOM) that tools/check_widgets.mjs tests against the KPs' stored answers.
// Sources: lectures/lecture_15/lecture_15.txt (cited lecture_15.pdf:p<n>, text layer) and transcript.json (cited video:M:SS).
// The deck's equations (p51-p52, p56-p58) and plots (p17, p43-p45, p63) are images; formulas named from the papers in
// sources.json are marked so. Where a widget needs a distribution the lecture never prints (per-response rewards, annotator
// rates, refusal scores), its values are schematic, user-movable controls, and the provenance line says so.
import { el, fmt, slider } from "../widgets_lib.js";

export const MODELS = {};
export const WIDGETS = {};

const C = { ink: "var(--ink)", muted: "var(--muted)", rule: "var(--rule)", a: "var(--accent)", b: "var(--accent2)", ok: "var(--ok)", hi: "var(--hilite)" };
const rect = (x, y, w, h, fill, extra = "") => `<rect x="${x.toFixed(1)}" y="${y.toFixed(1)}" width="${Math.max(0, w).toFixed(1)}" height="${Math.max(0, h).toFixed(1)}" style="fill:${fill}" ${extra}/>`;
const text = (x, y, s, o = {}) => `<text x="${x.toFixed(1)}" y="${y.toFixed(1)}" style="fill:${o.fill || C.ink};font:${o.weight || ""} ${o.size || 12}px var(--sans)" text-anchor="${o.anchor || "start"}">${s}</text>`;
const line = (x1, y1, x2, y2, stroke, extra = "") => `<line x1="${x1.toFixed(1)}" y1="${y1.toFixed(1)}" x2="${x2.toFixed(1)}" y2="${y2.toFixed(1)}" style="stroke:${stroke};stroke-width:1.5" ${extra}/>`;
const path = (pts, stroke, extra = "") => pts.length ? `<path d="${pts.map((p, i) => `${i ? "L" : "M"}${p[0].toFixed(1)},${p[1].toFixed(1)}`).join("")}" style="fill:none;stroke:${stroke};stroke-width:2" ${extra}/>` : "";
const dot = (x, y, r, fill, extra = "") => `<circle cx="${x.toFixed(1)}" cy="${y.toFixed(1)}" r="${r}" style="fill:${fill}" ${extra}/>`;
const svg = (w, h, body) => `<svg viewBox="0 0 ${w} ${h}" width="100%" style="max-width:${w}px;border:1px solid var(--rule);border-radius:6px;background:#fff" role="img">${body}</svg>`;
const check = (label, value, onChange) => { const i = el("input", { type: "checkbox" }); i.checked = value; i.addEventListener("change", () => onChange(i.checked)); return el("label", {}, i, ` ${label}`); };
const select = (label, opts, value, onChange) => { const s = el("select", {}, ...opts.map(([v, n]) => { const o = el("option", { value: v }, n); if (v === value) o.selected = true; return o; })); s.addEventListener("change", () => onChange(s.value)); return el("label", {}, `${label} `, s); };
const sig = x => 1 / (1 + Math.exp(-x));
const logit = p => Math.log(p / (1 - p));
const pct = (x, d = 1) => `${fmt(100 * x, d)}%`;
const sgn = (x, d = 3) => (x > 1e-12 ? "+" : x < -1e-12 ? "−" : "") + fmt(Math.abs(x), d);

// ================================================== 1. Bradley-Terry fit, three responses ==
// The Stiennon reward-model objective (lecture_15.pdf:p52 image; named "the Stiennon objective" at p57; spoken 1:06:39-1:06:55
// "a binary classifier on which of a pair of examples is better"): P(i ≻ j) = σ(r_i − r_j), fit by maximum likelihood on the
// observed choices (Bradley-Terry form from stiennon_2020_summarize §3.4 / the DPO paper, sources.json; not in the text layer).
// Three responses A, B, C to one prompt, the same number of comparisons on every observed pair, so the log-likelihood is
// Σ_pairs [p_obs log σ(d) + (1 − p_obs) log σ(−d)], concave in the ratings; its gradient for r_i is Σ_j (p_obs,ij − σ(r_i − r_j)).
// Ratings are identified only up to a shared constant, so the fit is anchored at mean 0 (a convention) and the widget adds a
// shift the learner controls. Observed win rates are controls; nothing here is a lecture number.
function btFit({ pAB, pBC, pCA, obsCA = true, shift = 0 }) {
  const pairs = [[0, 1, pAB], [1, 2, pBC]]; if (obsCA) pairs.push([2, 0, pCA]);
  const r = [0, 0, 0];
  for (let it = 0; it < 6000; it++) {
    const g = [0, 0, 0];
    for (const [i, j, p] of pairs) { const e = p - sig(r[i] - r[j]); g[i] += e; g[j] -= e; }
    for (let k = 0; k < 3; k++) r[k] += 1.0 * g[k];
    const m = (r[0] + r[1] + r[2]) / 3; for (let k = 0; k < 3; k++) r[k] -= m;
  }
  const rs = r.map(x => x + shift), P = (i, j) => sig(rs[i] - rs[j]);
  const ll = pairs.reduce((a, [i, j, p]) => a + p * Math.log(P(i, j)) + (1 - p) * Math.log(P(j, i)), 0) / pairs.length;
  const res = pairs.map(([i, j, p]) => p - P(i, j)), maxRes = Math.max(...res.map(Math.abs));
  return { rA: rs[0], rB: rs[1], rC: rs[2], fitAB: P(0, 1), fitBC: P(1, 2), fitAC: P(0, 2), fitCA: P(2, 0), ll, maxRes,
    exact: maxRes < 1e-4 ? 1 : 0, gapAC: rs[0] - rs[2], nPairs: pairs.length };
}
MODELS["fixture:lecture_15--bt-fit"] = {
  fn: btFit,
  cases: [
    { args: { pAB: 0.8, pBC: 0.8, obsCA: false }, pick: "fitAC", expect: 0.94, tol: 0.01, from: "lecture_15:bradley-terry-reward-model:check" },
    { args: { pAB: 0.8, pBC: 0.8, obsCA: false }, pick: "fitAC", expect: 16 / 17, tol: 1e-4 },     // exact: odds 4 × 4 = 16
    { args: { pAB: 0.8, pBC: 0.8, obsCA: false }, pick: "gapAC", expect: 2 * Math.log(4), tol: 1e-4 }, // logits add: 2 ln 4
    { args: { pAB: 0.8, pBC: 0.8, obsCA: false }, pick: "exact", expect: 1 },                    // two pairs: always fit exactly
    { args: { pAB: 0.8, pBC: 0.8, obsCA: false, shift: 10 }, pick: "fitAC", expect: 16 / 17, tol: 1e-4 }, // shared shift: P unchanged
    { args: { pAB: 0.9, pBC: 0.9, pCA: 0.9 }, pick: "fitAB", expect: 0.5, tol: 1e-4 },           // transfer: cyclic data, fit says 50%
    { args: { pAB: 0.9, pBC: 0.9, pCA: 0.9 }, pick: "rA", expect: 0, tol: 1e-6 },                // ... all three ratings equal
    { args: { pAB: 0.9, pBC: 0.9, pCA: 0.9 }, pick: "exact", expect: 0, tol: 0 },                // ... and no rating fits the data
    { args: { pAB: 0.8, pBC: 0.8, pCA: 1 - 16 / 17 }, pick: "exact", expect: 1 },                // consistent third pair: still exact
  ],
};
WIDGETS["fixture:lecture_15--bt-fit"] = (root) => {
  const s = { pAB: 0.7, pBC: 0.6, pCA: 0.3, obsCA: false, shift: 0 };
  const pic = el("div"), read = el("div", { class: "readout" });
  const draw = () => {
    const m = btFit(s), W = 640, H = 150, L = 40, R = 600, span = 6, Y0 = 62;
    const X = v => L + (Math.max(-span, Math.min(span, v)) + span) / (2 * span) * (R - L);
    let b = text(10, 16, "fitted ratings r(x, y) on one line (only differences carry information)", { fill: C.muted, size: 11 });
    b += line(L, Y0, R, Y0, C.rule);
    for (let t = -span; t <= span; t += 2) b += line(X(t), Y0 - 4, X(t), Y0 + 4, C.rule) + text(X(t), Y0 + 18, t, { fill: C.muted, size: 10, anchor: "middle" });
    const pts = [["A", m.rA, C.a], ["B", m.rB, C.ok], ["C", m.rC, C.b]].sort((a, c) => a[1] - c[1]);
    pts.forEach(([n, v, c], i) => { b += dot(X(v), Y0, 7, c); b += text(X(v), Y0 - 14 - (i % 2) * 14, `${n} ${fmt(v, 2)}`, { anchor: "middle", size: 11, weight: "bold" }); });
    const row = (y, lab, obs, fit) => text(L, y, lab, { size: 11 }) + text(L + 130, y, obs === null ? "not observed" : pct(obs), { size: 11, fill: obs === null ? C.muted : C.ink }) + text(L + 240, y, pct(fit), { size: 11 }) + text(L + 330, y, obs === null ? "implied by the other two" : `residual (= gradient) ${sgn(obs - fit, 3)}`, { size: 11, fill: obs !== null && Math.abs(obs - fit) > 1e-3 ? C.b : C.muted });
    b += text(L, 100, "pair", { fill: C.muted, size: 10 }) + text(L + 130, 100, "observed", { fill: C.muted, size: 10 }) + text(L + 240, 100, "σ(r_i − r_j)", { fill: C.muted, size: 10 });
    b += row(114, "A beats B", s.pAB, m.fitAB) + row(128, "B beats C", s.pBC, m.fitBC) + row(142, "C beats A", s.obsCA ? s.pCA : null, m.fitCA);
    pic.innerHTML = svg(W, H, b);
    const regime = m.nPairs === 2 ? "<b>two pairs observed</b>: they pin down r_A − r_B and r_B − r_C exactly, so the third probability is implied: the logits add, odds multiply"
      : m.exact ? "<b>three pairs, consistent</b>: one set of ratings reproduces every observed rate"
      : `<b>three pairs, ${(s.pAB - 0.5) * (s.pBC - 0.5) > 0 && (s.pBC - 0.5) * (s.pCA - 0.5) > 0 ? "cyclic (A ≻ B ≻ C ≻ A): transitivity is violated" : "not consistent with one scalar per response"}</b>: no ratings reproduce the data (largest residual ${fmt(m.maxRes, 3)}); the maximum-likelihood fit compromises${Math.abs(m.rA - m.rB) < 0.05 && Math.abs(m.rB - m.rC) < 0.05 ? " all the way to equal ratings, 50% on every pair" : ""}`;
    read.innerHTML = `${regime}<br>
      <span class="big">implied P(A ≻ C) = σ(r_A − r_C) = ${pct(m.fitAC, 2)}</span> · r_A − r_C = ${fmt(m.gapAC, 3)} · mean log-likelihood per comparison ${fmt(m.ll, 4)}<br>
      shared shift ${sgn(s.shift, 1)} moves every rating, and no probability or the likelihood changes: a reward of 2.0 means nothing by itself<br>
      <span class="muted small">provenance: fixture:lecture_15--bt-fit · lecture_15.pdf:p52 (Stiennon reward-model loss, image), p57 ("the Stiennon objective") · video 1:06:39-1:06:55 · Bradley-Terry form σ(r_w − r_l) from stiennon_2020_summarize / the DPO paper (sources.json), not the text layer · equal comparison counts per pair; ratings anchored at mean 0 by convention; win rates are your inputs.</span>`;
  };
  const pf = v => pct(v, 0);
  root.append(el("div", { class: "widget" }, el("div", { class: "controls" },
    slider("observed: A beats B", 0.02, 0.98, s.pAB, 0.01, v => { s.pAB = v; draw(); }, pf),
    slider("observed: B beats C", 0.02, 0.98, s.pBC, 0.01, v => { s.pBC = v; draw(); }, pf),
    check("C vs A also observed", s.obsCA, v => { s.obsCA = v; draw(); }),
    slider("observed: C beats A (when observed)", 0.02, 0.98, s.pCA, 0.01, v => { s.pCA = v; draw(); }, pf),
    slider("shared shift added to every rating", -4, 4, s.shift, 0.5, v => { s.shift = v; draw(); }, v => v.toFixed(1)),
    read), pic)); draw();
};

// ============================================ 2. DPO margin plane: implied rewards and step weight ==
// DPO (lecture_15.pdf:p56-p58, all equations image-only; named from the DPO paper, sources.json): implied reward
// r̂(y) = β·log(π_θ(y)/π_ref(y)) (+ β log Z(x), which cancels in a pair's difference, p57), loss −log σ(r̂(y_w) − r̂(y_l)),
// gradient β·(1 − σ(margin))·[∇log π(y_w) − ∇log π(y_l)]: "pos gradient on good, neg gradient on bad ... scaled by
// 'prediction error' of the implied reward model" (p58; spoken 1:13:22-1:14:10 "how much my implied reward model is wrong").
// Controls are the two log-ratios Δ_w = log π_θ(y_w) − log π_ref(y_w) and Δ_l (nats) and β; only Δ_w − Δ_l enters the loss.
function dpoMargin({ beta, dw, dl }) {
  const rw = beta * dw, rl = beta * dl, margin = rw - rl, P = sig(margin);
  return { rw, rl, margin, P, loss: -Math.log(P), weight: beta * (1 - P), weightFrac: 1 - P, chosenFell: dw < -1e-12 ? 1 : 0, rawGap: dw - dl };
}
MODELS["fixture:lecture_15--dpo-margin"] = {
  fn: dpoMargin,
  cases: [
    { args: { beta: 0.1, dw: -1, dl: -4 }, pick: "P", expect: 0.574, tol: 0.002, from: "lecture_15:dpo-objective:check" },
    { args: { beta: 0.1, dw: 0, dl: 0 }, pick: "rw", expect: 0, tol: 0 },              // predict: policy = reference, r̂ = 0
    { args: { beta: 0.1, dw: 0, dl: 0 }, pick: "weightFrac", expect: 0.5 },             // ... weight = β/2 whatever the responses
    { args: { beta: 0.3, dw: 0, dl: 0 }, pick: "weight", expect: 0.15 },                // ... β/2 at any β
    { args: { beta: 0.1, dw: 5, dl: 0 }, pick: "loss", expect: 0.4741, tol: 0.001 },    // transfer: run A (+5 on chosen) ...
    { args: { beta: 0.1, dw: 0, dl: -5 }, pick: "loss", expect: 0.4741, tol: 0.001 },   // ... run B (−5 on rejected): same loss
    { args: { beta: 0.1, dw: -2, dl: -10 }, pick: "loss", expect: 0.3711, tol: 0.001 }, // both lowered: loss below start's 0.693 ...
    { args: { beta: 0.1, dw: -2, dl: -10 }, pick: "chosenFell", expect: 1 },            // ... while the chosen response lost likelihood
    { args: { beta: 0.1, dw: 30, dl: -30 }, pick: "weightFrac", expect: 0.00247, tol: 0.01 }, // well separated: step nearly 0
    { args: { beta: 1, dw: 1, dl: -1 }, pick: "margin", expect: 2 },                    // β scales the margin for the same log-ratios
  ],
};
WIDGETS["fixture:lecture_15--dpo-margin"] = (root) => {
  const s = { beta: 0.1, dw: 1, dl: -1 };
  const pic = el("div"), read = el("div", { class: "readout" }), sl = {};
  const draw = () => {
    const m = dpoMargin(s), W = 660, H = 290;
    // left: the (Δ_l, Δ_w) plane, iso-loss lines are the diagonals Δ_w − Δ_l = const
    const L = 46, T = 30, S = 220, span = 12, X = v => L + (v + span) / (2 * span) * S, Y = v => T + (span - v) / (2 * span) * S;
    let b = text(10, 16, "pair log-ratios (nats); diagonals: equal loss (value at top)", { fill: C.muted, size: 11 });
    b += rect(L, Y(0), S, S / 2, C.b, 'fill-opacity="0.08"') + text(L + 4, T + S - 6, "chosen less likely than under π_ref", { fill: C.b, size: 10 });
    b += `<rect x="${L}" y="${T}" width="${S}" height="${S}" style="fill:none;stroke:${C.rule}"/>`;
    b += line(L, Y(0), L + S, Y(0), C.rule, 'stroke-dasharray="3 3"') + line(X(0), T, X(0), T + S, C.rule, 'stroke-dasharray="3 3"');
    for (const k of [-12, -6, 0, 6, 12, 18]) {
      const x1 = Math.max(-span, -span - k), x2 = Math.min(span, span - k); if (x2 <= x1) continue;
      b += line(X(x1), Y(x1 + k), X(x2), Y(x2 + k), C.a, `stroke-opacity="${k === 0 ? 0.6 : 0.25}"`);
      if (k > 0) b += text(X(x2), T - 3, fmt(-Math.log(sig(s.beta * k)), 2), { fill: C.a, size: 9, anchor: "middle" });
    }
    b += text(X(-span) + 2, Y(-span) - 30, "loss 0.69 (margin 0)", { fill: C.a, size: 9 });
    b += text(L + S / 2, T + S + 27, "Δ_l = log π_θ(y_l) − log π_ref(y_l)", { fill: C.muted, size: 10, anchor: "middle" });
    b += `<text transform="translate(${L - 30},${T + S / 2}) rotate(-90)" style="fill:${C.muted};font:10px var(--sans)" text-anchor="middle">Δ_w (chosen)</text>`;
    b += dot(X(0), Y(0), 4, C.muted) + text(X(0) + 6, Y(0) + 12, "start", { fill: C.muted, size: 9 });
    for (const t of [-12, 0, 12]) b += text(X(t), T + S + 12, t, { fill: C.muted, size: 9, anchor: "middle" }) + text(L - 4, Y(t) + 3, t, { fill: C.muted, size: 9, anchor: "end" });
    const cx = Math.max(-span, Math.min(span, s.dl)), cy = Math.max(-span, Math.min(span, s.dw));
    b += dot(X(cx), Y(cy), 6, m.chosenFell ? C.b : C.ok, `stroke="${C.ink}" stroke-width="1.5"`);
    // right: loss and step weight against the margin β(Δ_w − Δ_l)
    const L2 = 340, R2 = 640, T2 = 30, B2 = 250, mm = 4, ym = 2, X2 = v => L2 + (Math.max(-mm, Math.min(mm, v)) + mm) / (2 * mm) * (R2 - L2), Y2 = v => B2 - Math.min(ym, v) / ym * (B2 - T2);
    b += text(L2, 16, "loss −log σ(margin) (blue), step weight 1 − σ (orange)", { fill: C.muted, size: 11 });
    b += line(L2, B2, R2, B2, C.rule) + line(X2(0), T2, X2(0), B2, C.rule, 'stroke-dasharray="3 3"');
    for (let t = -mm; t <= mm; t += 2) b += text(X2(t), B2 + 14, t, { fill: C.muted, size: 10, anchor: "middle" });
    for (const v of [0.5, 1, 1.5, 2]) b += line(L2, Y2(v), R2, Y2(v), C.rule, 'stroke-opacity="0.4"') + text(L2 - 4, Y2(v) + 4, v, { fill: C.muted, size: 10, anchor: "end" });
    b += text((L2 + R2) / 2, B2 + 28, "margin = r̂(y_w) − r̂(y_l) = β(Δ_w − Δ_l)", { fill: C.muted, size: 10, anchor: "middle" });
    const pl = [], pw = []; for (let v = -mm; v <= mm + 1e-9; v += 0.05) { if (-Math.log(sig(v)) <= ym) pl.push([X2(v), Y2(-Math.log(sig(v)))]); pw.push([X2(v), Y2(1 - sig(v))]); }
    b += path(pl, C.a) + path(pw, C.b, 'stroke-dasharray="5 3"');
    b += dot(X2(m.margin), Y2(m.loss), 5, C.a) + dot(X2(m.margin), Y2(m.weightFrac), 5, C.b);
    pic.innerHTML = svg(W, H, b);
    read.innerHTML = `implied rewards r̂(y_w) = β·Δ_w = ${sgn(m.rw, 3)} · r̂(y_l) = β·Δ_l = ${sgn(m.rl, 3)} · margin ${sgn(m.margin, 3)}<br>
      <span class="big">σ(margin) = ${fmt(m.P, 4)} · loss ${fmt(m.loss, 4)} · step weight β(1 − σ) = ${fmt(m.weight, 4)} (= ${fmt(m.weightFrac, 3)}·β)</span><br>
      ${Math.abs(m.margin) < 1e-12 ? "margin 0: the implied reward model says 50/50, so every pair gets weight β/2 however likely its responses are" : m.weightFrac < 0.05 ? "pair already well separated: the implied reward model is right, so this pair barely teaches" : m.margin < 0 ? "implied reward model ranks the pair the wrong way: large weight" : "the step pushes y_w up and y_l down, weighted by how wrong the implied reward model still is"}${m.chosenFell ? `<br><b>chosen response is less likely than under π_ref</b> (Δ_w = ${sgn(s.dw, 1)}), yet the loss is ${m.loss < Math.log(2) ? "below" : "not below"} its start value 0.693: DPO raises the margin, not the chosen likelihood` : ""}<br>
      <span class="muted small">provenance: fixture:lecture_15--dpo-margin · lecture_15.pdf:p56-p58 (derivation and gradient, images; forms named from the DPO paper, sources.json) · video 1:13:22-1:14:10 · log π_ref's absolute value never enters, only the log-ratios; β log Z(x) cancels in the difference (p57).</span>`;
    for (const k of ["dw", "dl"]) { const inp = sl[k].querySelector("input"), out = sl[k].querySelector(".readout"); inp.value = s[k]; out.textContent = s[k].toFixed(1); }
  };
  sl.dw = slider("Δ_w = log π_θ(y_w) − log π_ref(y_w)", -12, 12, s.dw, 0.1, v => { s.dw = v; draw(); }, v => v.toFixed(1));
  sl.dl = slider("Δ_l = log π_θ(y_l) − log π_ref(y_l)", -12, 12, s.dl, 0.1, v => { s.dl = v; draw(); }, v => v.toFixed(1));
  const presets = [["start: π_θ = π_ref", 0, 0], ["chosen +5 only", 5, 0], ["rejected −5 only", 0, -5], ["both lowered, rejected more", -2, -10]];
  root.append(el("div", { class: "widget" }, el("div", { class: "controls" },
    slider("β", 0.01, 1, s.beta, 0.01, v => { s.beta = v; draw(); }, v => v.toFixed(2)), sl.dw, sl.dl,
    el("div", { style: "margin-top:6px" }, ...presets.map(([n, w, l]) => el("button", { onclick: () => { s.dw = w; s.dl = l; draw(); } }, n)), " "),
    read), pic)); draw();
};

// ============================================ 3. KL-tilted optimum vs best-of-n: proxy vs gold reward ==
// RLHF objective max E_π[r] − β·KL(π ‖ π_ref) (InstructGPT's, p51 image; said aloud 1:06:10-1:06:39 "subject to this second term,
// which is really just a KL divergence"). Over all policies its maximizer is π*(y) ∝ π_ref(y)·exp(r(y)/β) (p56 image; spoken
// 1:11:18-1:13:04, the reference "exponentially tilt[ed]" by exp(r/β)); best-of-n (p54 "1024 outputs, take the best one")
// returns the highest-proxy of n samples from π_ref. Both are driven by the learned (proxy) reward r̂; the true (gold)
// preference r* is what a fresh judge would give. p63: "Optimizing for reward overfits past a point. Holds true for human pref
// (left), noisy LM pref (mid) but not noiseless LM pref (right)"; spoken 1:16:45-1:17:27 "overfitting to your learned reward
// model ... the KL regularizer ... is really critical".
// The five responses, their π_ref masses and gold rewards are SCHEMATIC (the lecture prints no per-response rewards; p63 is an
// image). The reward-model error is put where the lecture says annotators and judges err: length (p17, "very strong length
// effects") and assertiveness over factuality (p44, Hosking et al.). A slider scales that error; 0 is the noiseless judge.
const RESP = [
  { n: "short, correct", ref: 0.30, gold: 1.0, err: 0 },
  { n: "detailed, correct", ref: 0.20, gold: 1.5, err: 0.1 },
  { n: "padded, repetitive", ref: 0.15, gold: 0.8, err: 0.9 },
  { n: "assertive, wrong", ref: 0.10, gold: -1.0, err: 2.9 },
  { n: "off-topic", ref: 0.25, gold: -2.0, err: 0 },
];
function tiltDist(proxy, beta) {
  const w = RESP.map((r, i) => Math.log(r.ref) + proxy[i] / beta), mx = Math.max(...w), e = w.map(x => Math.exp(x - mx)), z = e.reduce((a, x) => a + x, 0);
  return e.map(x => x / z);
}
function bonDist(proxy, n) {                         // P(argmax of n draws from π_ref = y), ties share their group's mass by π_ref
  const idx = RESP.map((_, i) => i).sort((a, b) => proxy[a] - proxy[b]), out = new Array(RESP.length).fill(0);
  let F = 0;
  for (let k = 0; k < idx.length;) {
    let j = k, g = 0; while (j < idx.length && Math.abs(proxy[idx[j]] - proxy[idx[k]]) < 1e-12) { g += RESP[idx[j]].ref; j++; }
    const pg = Math.pow(Math.min(1, F + g), n) - Math.pow(F, n);
    for (let t = k; t < j; t++) out[idx[t]] = pg * RESP[idx[t]].ref / g;
    F += g; k = j;
  }
  return out;
}
function summarize(p, proxy) {
  let kl = 0, H = 0, gold = 0, prx = 0;
  p.forEach((q, i) => { if (q > 1e-300) { kl += q * Math.log(q / RESP[i].ref); H -= q * Math.log(q); } gold += q * RESP[i].gold; prx += q * proxy[i]; });
  return { kl: Math.max(0, kl), H, gold, proxy: prx };
}
function klTilt({ mode = "tilt", beta = 1, n = 1, errScale = 1 }) {
  const proxy = RESP.map(r => r.gold + errScale * r.err);
  const dist = x => (mode === "bon" ? bonDist(proxy, x) : tiltDist(proxy, x));
  const p = dist(mode === "bon" ? n : beta), cur = summarize(p, proxy);
  const sweep = [];                                  // weakest to strongest optimization
  if (mode === "bon") for (let t = 0; t <= 14; t += 0.05) sweep.push({ x: Math.pow(2, t), ...summarize(dist(Math.pow(2, t)), proxy) });
  else for (let t = 2; t >= -2.5; t -= 0.02) sweep.push({ x: Math.pow(10, t), ...summarize(dist(Math.pow(10, t)), proxy) });
  const peak = sweep.reduce((a, q) => (q.gold > a.gold ? q : a), sweep[0]);
  const pastPeak = cur.kl > peak.kl + 1e-6 && cur.gold < peak.gold - 1e-4 ? 1 : 0;
  const o = { ...cur, p, proxyR: proxy, sweep, peakGold: peak.gold, peakKL: peak.kl, peakX: peak.x, pastPeak, goldBelowPeak: peak.gold - cur.gold };
  p.forEach((q, i) => { o[`p${i + 1}`] = q; });
  return o;
}
MODELS["fixture:lecture_15--kl-tilt"] = {
  fn: klTilt,
  cases: [
    { args: { beta: 1e6 }, pick: "p1", expect: 0.30, tol: 1e-4 },                        // β → ∞: π* = π_ref (KL 0)
    { args: { beta: 1 }, pick: "kl", expect: 0.2985, tol: 0.002 },                      // ppo check: β = 1 ...
    { args: { beta: 10 }, pick: "kl", expect: 0.01023, tol: 0.01 },                     // ... ×10 → KL down (0.30 → 0.010)
    { args: { beta: 0.05 }, pick: "proxy", expect: 1.8932, tol: 0.002 },                 // overopt predict: proxy keeps rising ...
    { args: { beta: 0.5 }, pick: "proxy", expect: 1.6276, tol: 0.002 },                  // ... (higher at smaller β)
    { args: { beta: 0.05 }, pick: "pastPeak", expect: 1 },                              // ... while gold has passed its peak
    { args: { mode: "bon", n: 1024 }, pick: "pastPeak", expect: 1 },                    // overopt check: best-of-1024 past the gold peak
    { args: { mode: "bon", n: 1024 }, pick: "gold", expect: -1.0, tol: 0.001 },         // ... picks "assertive, wrong" almost surely
    { args: { mode: "bon", n: 4 }, pick: "gold", expect: 0.345, tol: 0.01 },            // ... moderate n: higher gold than n = 1024
    { args: { mode: "bon", n: 1024, errScale: 0 }, pick: "pastPeak", expect: 0, tol: 0 }, // noiseless judge (p63 right): no fall
    { args: { beta: 0.01, errScale: 0 }, pick: "gold", expect: 1.5, tol: 0.001 },       // ... the tilt converges on the best gold response
    { args: { mode: "bon", n: 1 }, pick: "p4", expect: 0.10, tol: 1e-6 },                // best-of-1 = π_ref
  ],
};
WIDGETS["fixture:lecture_15--kl-tilt"] = (root) => {
  const s = { mode: "tilt", lb: 0.5, ln: 1, errScale: 1 };
  const pic = el("div"), read = el("div", { class: "readout" }), sl = {};
  const draw = () => {
    const beta = Math.pow(10, s.lb), n = Math.pow(2, s.ln), m = klTilt({ mode: s.mode, beta, n, errScale: s.errScale });
    const W = 680, H = 280, L = 44, R = 380, T = 30, B = 236;
    const xmax = Math.max(0.5, ...m.sweep.map(q => q.kl)) * 1.05;
    const ys = m.sweep.flatMap(q => [q.gold, q.proxy]), ymin = Math.min(-0.5, ...ys) - 0.2, ymax = Math.max(1, ...ys) + 0.3;
    const X = v => L + v / xmax * (R - L), Y = v => B - (v - ymin) / (ymax - ymin) * (B - T);
    let b = text(10, 16, `${s.mode === "bon" ? "best-of-n" : "KL-regularized optimum"}: expected reward vs KL from π_ref`, { fill: C.muted, size: 11 });
    b += line(L, B, R, B, C.rule) + line(L, T, L, B, C.rule) + line(L, Y(0), R, Y(0), C.rule, 'stroke-dasharray="3 3"');
    for (let t = 0; t <= xmax + 1e-9; t += 0.5) b += text(X(t), B + 14, fmt(t, 1), { fill: C.muted, size: 10, anchor: "middle" });
    for (let t = Math.ceil(ymin); t <= ymax; t++) b += text(L - 5, Y(t) + 4, t, { fill: C.muted, size: 10, anchor: "end" });
    b += text((L + R) / 2, B + 28, "KL from π_ref (nats) →", { fill: C.muted, size: 10, anchor: "middle" });
    b += path(m.sweep.map(q => [X(q.kl), Y(q.proxy)]), C.a) + path(m.sweep.map(q => [X(q.kl), Y(q.gold)]), C.ok);
    const lastP = m.sweep[m.sweep.length - 1];
    const same = m.sweep.every(q => Math.abs(q.gold - q.proxy) < 1e-9);
    if (same) b += text(R - 2, Y(lastP.proxy) - 8, "proxy r̂ = gold r* (noiseless judge)", { fill: C.ok, size: 10, anchor: "end" });
    else b += text(R - 2, Y(lastP.proxy) - 6, "proxy r̂ (what is optimized)", { fill: C.a, size: 10, anchor: "end" }) + text(R - 2, Y(lastP.gold) - 8, "gold r* (fresh judge)", { fill: C.ok, size: 10, anchor: "end" });
    if (m.peakKL < lastP.kl - 0.02) b += line(X(m.peakKL), T, X(m.peakKL), B, C.ok, 'stroke-dasharray="2 3" stroke-opacity="0.7"') + text(X(m.peakKL) + 3, T + 10, "gold peak", { fill: C.ok, size: 10 });
    b += dot(X(m.kl), Y(m.proxy), 5, C.a, `stroke="${C.ink}"`) + dot(X(m.kl), Y(m.gold), 5, C.ok, `stroke="${C.ink}"`);
    // right: the distribution over the five schematic responses
    const L2 = 410, bw = 44, B2 = 200, T2 = 40, hs = (B2 - T2);
    b += text(L2, 16, "π_ref (outline) vs current π (filled)", { fill: C.muted, size: 11 });
    RESP.forEach((r, i) => {
      const x = L2 + i * (bw + 10);
      b += rect(x, B2 - m.p[i] * hs, bw, m.p[i] * hs, r.err > 0.5 && s.errScale > 0 ? C.b : C.ok, 'fill-opacity="0.75"');
      b += `<rect x="${x}" y="${(B2 - r.ref * hs).toFixed(1)}" width="${bw}" height="${(r.ref * hs).toFixed(1)}" style="fill:none;stroke:${C.ink};stroke-dasharray:3 2"/>`;
      b += text(x + bw / 2, B2 - m.p[i] * hs - 4, pct(m.p[i], 0), { size: 10, anchor: "middle" });
      r.n.split(", ").forEach((w, k) => { b += text(x + bw / 2, B2 + 13 + k * 11, w, { size: 9, anchor: "middle", fill: C.muted }); });
      b += text(x + bw / 2, B2 + 40, `r* ${sgn(r.gold, 1)}`, { size: 9, anchor: "middle", fill: C.ok }) + text(x + bw / 2, B2 + 51, `r̂ ${sgn(m.proxyR[i], 1)}`, { size: 9, anchor: "middle", fill: C.a });
    });
    pic.innerHTML = svg(W, H, b);
    const knob = s.mode === "bon" ? `n = ${fmt(n, 0)} (slider: n)` : `β = ${fmt(beta, 3)} (slider: β)`;
    const regime = s.errScale === 0 ? "<b>noiseless judge</b>: proxy = gold, so optimizing harder only helps (p63 right panel)"
      : m.pastPeak ? `<b>past the gold peak</b>: the proxy is still rising while the gold reward has fallen ${fmt(m.goldBelowPeak, 3)} below its peak (${fmt(m.peakGold, 3)} at KL ${fmt(m.peakKL, 2)}): the optimizer is exploiting the reward model's error`
      : "<b>before the gold peak</b>: proxy and gold rise together";
    read.innerHTML = `${knob} · KL(π ‖ π_ref) = <b>${fmt(m.kl, 3)}</b> nats · entropy ${fmt(m.H, 3)} nats (π_ref: 1.544)<br>
      <span class="big">proxy E[r̂] = ${fmt(m.proxy, 3)} · gold E[r*] = ${fmt(m.gold, 3)}</span><br>${regime}<br>
      ${s.mode === "bon" ? "best-of-n changes no weights: n = 1 is π_ref again" : "π*(y) ∝ π_ref(y)·exp(r̂(y)/β): smaller β, stronger tilt, larger KL; larger β holds π near π_ref"}<br>
      <span class="muted small">provenance: fixture:lecture_15--kl-tilt · lecture_15.pdf:p51 (InstructGPT objective with KL, image; said 1:06:10-1:06:39), p56 (maximizer π_ref·exp(r/β), image; said 1:11:18-1:13:04), p54 (best-of-n), p63 (overoptimization; human / noisy / noiseless panels) · video 1:16:45-1:17:27 · the five responses, π_ref and r* are schematic; r̂ − r* is put on length (p17) and assertiveness (p44) and scaled by the error slider.</span>`;
    sl.err.querySelector("input").value = s.errScale; sl.err.querySelector(".readout").textContent = s.errScale.toFixed(2);
  };
  sl.err = slider("reward-model error scale (0 = noiseless judge)", 0, 1.5, s.errScale, 0.05, v => { s.errScale = v; draw(); }, v => v.toFixed(2));
  root.append(el("div", { class: "widget" }, el("div", { class: "controls" },
    select("optimizer", [["tilt", "KL-regularized optimum (PPO/DPO target)"], ["bon", "best-of-n against the reward model"]], s.mode, v => { s.mode = v; draw(); }),
    slider("β (KL strength, log scale; used by the optimum)", -2, 2, s.lb, 0.05, v => { s.lb = v; draw(); }, v => fmt(Math.pow(10, v), 3)),
    slider("n (samples, log scale; used by best-of-n)", 0, 12, s.ln, 1, v => { s.ln = v; draw(); }, v => fmt(Math.pow(2, v), 0)),
    sl.err,
    el("div", { style: "margin-top:6px" }, el("button", { onclick: () => { s.errScale = 0; draw(); } }, "noiseless judge"), " ", el("button", { onclick: () => { s.errScale = 1; draw(); } }, "noisy reward model")),
    read), pic)); draw();
};

// ============================================ 4. safety: violation vs false-refusal frontier ==
// Spoken 29:31-30:11 (the safety slides p23-p25 are screenshots): every safety-tuning approach balances "how often am I allowing
// bad queries to get through ... the violation rate" against over-refusal ("how do I kill a Python process ... no, I can't let you
// kill anything"), "the false refusal rate also needs to be low ... have a good Pareto trade between the two", navigated by
// "various kinds of tailored data". The widget's mechanism is a SCHEMATIC signal-detection model, the author's device: the model
// gives each request a harm-likeness score, benign ~ N(0, 1), harmful ~ N(d, 1); it refuses when the score exceeds t.
// violation = Φ(t − d) (harmful answered), false refusal = 1 − Φ(t) (benign refused). Moving t slides along one frontier
// (refusal-only data mainly does this); raising d (data that teaches harmful from look-alike benign, the author's reading of
// "tailored data", filtering note) moves the whole frontier toward the origin. d and t are controls, not lecture numbers.
const erf = x => { const t = 1 / (1 + 0.3275911 * Math.abs(x)), y = 1 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-x * x); return x >= 0 ? y : -y; };
const Phi = x => 0.5 * (1 + erf(x / Math.SQRT2));
const PhiInv = p => { let lo = -10, hi = 10; for (let i = 0; i < 100; i++) { const mid = (lo + hi) / 2; if (Phi(mid) < p) lo = mid; else hi = mid; } return (lo + hi) / 2; };
function refusal({ d = 2.5, t = 1.6, targetV = null, baseV = null, models = [] }) {
  const o = { v: Phi(t - d), fr: 1 - Phi(t) };
  const frAt = v => 1 - Phi(d + PhiInv(v));                       // threshold that gives violation v on this frontier
  if (targetV !== null) o.frAtTarget = frAt(targetV);
  if (baseV !== null) { o.frAtBase = frAt(baseV); if (targetV !== null) o.frRises = o.frAtTarget > o.frAtBase ? 1 : 0; }
  const ms = models.map(([n, v, fr]) => ({ n, v, fr, dominatedBy: models.filter(([n2, v2, f2]) => n2 !== n && v2 <= v && f2 <= fr && (v2 < v || f2 < fr)).map(x => x[0]) }));
  o.models = ms; o.nDominated = ms.filter(x => x.dominatedBy.length).length;
  if (ms.length) { const lo = ms.reduce((a, x) => (x.v < a.v ? x : a)); o.lowestVName = lo.n; o.lowestVfr = lo.fr; }
  return o;
}
MODELS["fixture:lecture_15--refusal-frontier"] = {
  fn: refusal,
  cases: [
    { args: { d: 2.5, targetV: 0.01, baseV: 0.12 }, pick: "frRises", expect: 1 },          // predict: violation 12% → 1% by refusing more ...
    { args: { d: 2.5, targetV: 0.12 }, pick: "frAtTarget", expect: 0.0926, tol: 0.01 },    // ... false refusal at 12% violation ...
    { args: { d: 2.5, targetV: 0.01 }, pick: "frAtTarget", expect: 0.431, tol: 0.01 },     // ... rises to 43% at 1%
    { args: { d: 4, targetV: 0.01 }, pick: "frAtTarget", expect: 0.0471, tol: 0.01 },      // a better-separating model: same 1%, 4.7%
    { args: { models: [["X", 0.02, 0.25], ["Y", 0.04, 0.03], ["Z", 0.05, 0.06]] }, pick: "nDominated", expect: 1 },  // transfer: Z only
    { args: { models: [["X", 0.02, 0.25], ["Y", 0.04, 0.03], ["Z", 0.05, 0.06]] }, pick: "lowestVfr", expect: 0.25 }, // lowest violation = X, 25% refused
    { args: { d: 2.5, t: 2.5 }, pick: "v", expect: 0.5 },                                  // threshold at the harmful mean: half get through
    { args: { d: 2.5, t: 0 }, pick: "fr", expect: 0.5 },                                   // threshold at the benign mean: half refused
  ],
};
WIDGETS["fixture:lecture_15--refusal-frontier"] = (root) => {
  const s = { d: 2.5, t: 1.6, text: "M1 3 12, M2 6 5, M3 8 9" };
  const pic = el("div"), read = el("div", { class: "readout" }), err = el("div", { class: "muted small" });
  const parse = () => {
    const ms = []; for (const chunk of s.text.split(",").map(x => x.trim()).filter(Boolean)) {
      const [n, v, f] = chunk.split(/\s+/); if (!n || !Number.isFinite(+v) || !Number.isFinite(+f)) return { err: `could not read "${chunk}" (name, violation %, false-refusal %)` };
      ms.push([n, +v / 100, +f / 100]);
    } return { ms };
  };
  const draw = () => {
    const p = parse(); err.textContent = p.err || "";
    const m = refusal({ d: s.d, t: s.t, models: p.ms || [] }), W = 680, H = 270;
    // left: score densities and the threshold
    const L = 20, R = 330, T = 34, B = 200, lo = -3.5, hi = 7.5, X = z => L + (z - lo) / (hi - lo) * (R - L), pk = 0.42, Y = v => B - v / pk * (B - T);
    const pdf = (z, mu) => Math.exp(-0.5 * (z - mu) ** 2) / Math.sqrt(2 * Math.PI);
    let b = text(10, 16, "harm-likeness score the model assigns (schematic)", { fill: C.muted, size: 11 });
    const area = (mu, a, z1, col) => { const pts = [[X(a), Y(0)]]; for (let z = a; z <= z1 + 1e-9; z += 0.05) pts.push([X(z), Y(pdf(z, mu))]); pts.push([X(z1), Y(0)]); return `<path d="${pts.map((q, i) => `${i ? "L" : "M"}${q[0].toFixed(1)},${q[1].toFixed(1)}`).join("")}Z" style="fill:${col};fill-opacity:0.35"/>`; };
    const tt = Math.max(lo, Math.min(hi, s.t));
    b += area(s.d, lo, tt, C.b) + area(0, tt, hi, C.hi);
    const curve = mu => { const o = []; for (let z = lo; z <= hi + 1e-9; z += 0.05) o.push([X(z), Y(pdf(z, mu))]); return o; };
    b += path(curve(0), C.ok) + path(curve(s.d), C.b) + line(L, B, R, B, C.rule);
    b += text(X(0), Y(0.4) - 4, "benign", { fill: C.ok, size: 10, anchor: "middle" }) + text(X(s.d), Y(0.4) - 4 + (Math.abs(s.d) < 1.2 ? -11 : 0), "harmful", { fill: C.b, size: 10, anchor: "middle" });
    b += line(X(tt), T - 6, X(tt), B, C.ink, 'stroke-dasharray="4 3"') + text(X(tt) + 3, T - 8, "refuse →", { size: 10 }) + text(X(tt) - 3, T - 8, "← answer", { size: 10, anchor: "end" });
    b += text(L, B + 16, "shaded red: harmful answered (violation) · shaded yellow: benign refused (false refusal)", { fill: C.muted, size: 9 });
    // right: the frontier and the models
    const L2 = 400, R2 = 660, T2 = 34, B2 = 220, vx = 0.5, fy = 0.6, X2 = v => L2 + Math.min(vx, v) / vx * (R2 - L2), Y2 = f => B2 - Math.min(fy, f) / fy * (B2 - T2);
    b += text(L2 - 20, 16, "false-refusal rate vs violation rate", { fill: C.muted, size: 11 });
    b += line(L2, B2, R2, B2, C.rule) + line(L2, T2, L2, B2, C.rule);
    for (const v of [0, 0.1, 0.2, 0.3, 0.4, 0.5]) b += text(X2(v), B2 + 13, `${Math.round(v * 100)}%`, { fill: C.muted, size: 9, anchor: "middle" });
    for (const f of [0, 0.2, 0.4, 0.6]) b += text(L2 - 4, Y2(f) + 3, `${Math.round(f * 100)}%`, { fill: C.muted, size: 9, anchor: "end" });
    b += text((L2 + R2) / 2, B2 + 26, "violation rate (harmful answered)", { fill: C.muted, size: 10, anchor: "middle" });
    const fr = []; for (let t = -2; t <= 9; t += 0.02) { const v = Phi(t - s.d), f = 1 - Phi(t); if (v <= vx && f <= fy) fr.push([X2(v), Y2(f)]); }
    b += path(fr, C.a) + text(R2, T2 + 4, `frontier at separation d = ${fmt(s.d, 1)}`, { fill: C.a, size: 10, anchor: "end" });
    for (const x of m.models) { const dom = x.dominatedBy.length > 0; b += dot(X2(x.v), Y2(x.fr), 5, dom ? "#fff" : C.ink, `stroke="${C.ink}" stroke-width="1.5"`) + text(X2(x.v) + 7, Y2(x.fr) + 4, x.n + (dom ? ` (dominated by ${x.dominatedBy.join(", ")})` : ""), { size: 10, fill: dom ? C.muted : C.ink }); }
    b += dot(X2(m.v), Y2(m.fr), 6, C.b, `stroke="${C.ink}" stroke-width="1.5"`);
    pic.innerHTML = svg(W, H, b);
    read.innerHTML = `threshold t = ${fmt(s.t, 2)}: <span class="big">violation ${pct(m.v)} · false refusal ${pct(m.fr)}</span><br>
      lowering t (refusing more) cuts violations and raises false refusals along the same curve; only a larger separation d moves the curve itself toward (0, 0)<br>
      ${m.models.length ? `your models: ${m.models.map(x => `${x.n} (${pct(x.v, 0)}, ${pct(x.fr, 0)})${x.dominatedBy.length ? " dominated" : ""}`).join(" · ")} · lowest violation: ${m.lowestVName}, which refuses ${pct(m.lowestVfr, 0)} of benign requests` : ""}<br>
      <span class="muted small">provenance: fixture:lecture_15--refusal-frontier · video 29:31-30:11 (violation rate, false-refusal rate, the Python-process example, "a good Pareto trade") · the score distributions, d and t are a schematic signal-detection device, not from the lecture; model points are your inputs (% violation, % false refusal).</span>`;
  };
  const inp = el("input", { type: "text", value: s.text, style: "width:100%;max-width:420px" });
  inp.addEventListener("input", () => { s.text = String(inp.value ?? ""); draw(); });
  root.append(el("div", { class: "widget" }, el("div", { class: "controls" },
    slider("refusal threshold t (lower = refuse more)", -1, 5, s.t, 0.05, v => { s.t = v; draw(); }, v => v.toFixed(2)),
    slider("separation d (how well the model tells harmful from look-alike benign)", 0.5, 5, s.d, 0.1, v => { s.d = v; draw(); }, v => v.toFixed(1)),
    el("label", {}, "models to compare (name violation% false-refusal%, comma-separated) ", inp), err,
    read), pic)); draw();
};

// ============================================ 5. annotator pool: agreement (variance) vs shared bias ==
// Spoken 56:38-57:59: inter-annotator agreement is "a statement about a population of annotators, whether they have high
// variance or not ... it doesn't tell you what the bias is. It tells you the variance"; "if they're all using ChatGPT, the
// variance will also be zero". p43-p44 (images; Santurkar+ 2023, Hosking, Blunsom, Bartolo 2024): annotators matter "a lot",
// "true even for many annotators", style and assertiveness over factuality. SCHEMATIC model (the author's device; the
// lecture prints no rates): on each response pair a style cue (e.g. the more assertive answer) points at the truly better
// response on a fraction c of pairs and at the worse one otherwise. Each annotator independently follows the cue with
// probability b (the population's shared bias), else judges content, correct with probability a. Per pair the chance a label
// is right is p1 = b + (1 − b)a when the cue is right, p0 = (1 − b)a when it is wrong. Agreement = chance two annotators give
// the same label; the reward model's label is the N-annotator majority. "Copy one chatbot" makes every label the bot's.
function majority(p, N) {                          // P(Binomial(N, p) > N/2), ties split evenly
  if (p <= 0) return 0; if (p >= 1) return 1;
  const lp = Math.log(p), lq = Math.log(1 - p); let logC = 0, acc = 0;
  for (let k = 0; k <= N; k++) {
    if (k > 0) logC += Math.log((N - k + 1) / k);
    const term = Math.exp(logC + k * lp + (N - k) * lq);
    if (2 * k > N) acc += term; else if (2 * k === N) acc += term / 2;
  }
  return Math.min(1, acc);
}
function annotators({ b = 0.3, a = 0.8, c = 0.5, N = 5, copy = false, botAcc = 0.75 }) {
  if (copy) return { p1: botAcc, p0: botAcc, agreement: 1, single: botAcc, majAcc: botAcc, limit: botAcc, cueWins: null, variance: 0 };
  const p1 = b + (1 - b) * a, p0 = (1 - b) * a, ag = p => p * p + (1 - p) * (1 - p), step = p => (p > 0.5 ? 1 : p < 0.5 ? 0 : 0.5);
  const agreement = c * ag(p1) + (1 - c) * ag(p0);
  return { p1, p0, agreement, single: c * p1 + (1 - c) * p0, majAcc: c * majority(p1, N) + (1 - c) * majority(p0, N),
    limit: c * step(p1) + (1 - c) * step(p0), cueWins: 1 - majority(p0, N), variance: 1 - agreement };
}
MODELS["fixture:lecture_15--annotator-pool"] = {
  fn: annotators,
  cases: [
    { args: { copy: true, botAcc: 0.75 }, pick: "agreement", expect: 1 },                    // agreement check/predict: same chatbot, ~100% ...
    { args: { copy: true, botAcc: 0.75 }, pick: "majAcc", expect: 0.75 },                    // ... and accuracy is just the bot's
    { args: { b: 0.6, a: 0.8, c: 0.5, N: 5000 }, pick: "majAcc", expect: 0.5, tol: 0.001 },  // annotator check: 5,000 from one population ...
    { args: { b: 0.6, a: 0.8, c: 0.5, N: 50 }, pick: "majAcc", expect: 0.5022, tol: 0.002 }, // ... vs 50: the same bias survives
    { args: { b: 0.6, a: 0.8, c: 0.5, N: 5000 }, pick: "cueWins", expect: 1, tol: 0.001 },   // ... majority follows the cue where it is wrong
    { args: { b: 0, a: 0.8, c: 0.5, N: 5000 }, pick: "majAcc", expect: 1, tol: 0.001 },      // no shared bias: independent noise averages out
    { args: { b: 0, a: 0.6, c: 0.5, N: 1 }, pick: "agreement", expect: 0.52 },               // agreement transfer (a): a hard/subjective call, low agreement, no bias
    { args: { b: 0.6, a: 0.8, c: 0.5, N: 1 }, pick: "agreement", expect: 0.7088, tol: 0.002 }, // bias b = 0.6: agreement 71% ...
    { args: { b: 0.9, a: 0.8, c: 0.5, N: 1 }, pick: "agreement", expect: 0.9068, tol: 0.002 }, // ... b = 0.9: 91% agreement ...
    { args: { b: 0.9, a: 0.8, c: 0.5, N: 1 }, pick: "single", expect: 0.53, tol: 0.002 },      // ... with 53% of labels right
  ],
};
WIDGETS["fixture:lecture_15--annotator-pool"] = (root) => {
  const s = { b: 0.3, a: 0.8, c: 0.5, lN: 2, copy: false, botAcc: 0.75 };
  const pic = el("div"), read = el("div", { class: "readout" });
  const Nof = lN => 2 * Math.round(Math.pow(10, lN) / 2) + 1;      // odd pool sizes 1 … ~5,001
  const draw = () => {
    const N = Nof(s.lN), m = annotators({ ...s, N }), W = 660, H = 250;
    const L = 46, R = 400, T = 30, B = 200, X = lN => L + lN / 3.7 * (R - L), Y = v => B - (v - 0.3) / 0.7 * (B - T);
    let b = text(10, 16, "accuracy of the majority label (what the reward model is trained on) vs pool size", { fill: C.muted, size: 11 });
    b += line(L, B, R, B, C.rule) + line(L, T, L, B, C.rule);
    for (const v of [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]) b += line(L, Y(v), R, Y(v), C.rule, 'stroke-opacity="0.35"') + text(L - 4, Y(v) + 4, `${Math.round(v * 100)}%`, { fill: C.muted, size: 9, anchor: "end" });
    for (const [lN, lab] of [[0, "1"], [1, "11"], [2, "101"], [3, "1,001"], [3.7, "5,001"]]) b += text(X(lN), B + 14, lab, { fill: C.muted, size: 10, anchor: "middle" });
    b += text((L + R) / 2, B + 28, "annotators per pair, N (log scale)", { fill: C.muted, size: 10, anchor: "middle" });
    const pts = []; for (let t = 0; t <= 3.7 + 1e-9; t += 0.05) pts.push([X(t), Y(Math.max(0.3, annotators({ ...s, N: Nof(t) }).majAcc))]);
    b += path(pts, C.a) + line(L, Y(Math.max(0.3, m.limit)), R, Y(Math.max(0.3, m.limit)), C.b, 'stroke-dasharray="5 3"') + text(R, Y(Math.max(0.3, m.limit)) - 5, `limit as N → ∞: ${pct(m.limit, 0)}`, { fill: C.b, size: 10, anchor: "end" });
    b += dot(X(s.lN), Y(Math.max(0.3, m.majAcc)), 5, C.a, `stroke="${C.ink}"`);
    const bx = 450, bw = 70, YB = v => B - v * (B - T);
    [["agreement", m.agreement, C.ok], ["one label right", m.single, C.a]].forEach(([n, v, col], i) => {
      const x = bx + i * (bw + 30); b += rect(x, YB(v), bw, v * (B - T), col, 'fill-opacity="0.7"') + text(x + bw / 2, YB(v) - 4, pct(v, 0), { size: 11, anchor: "middle", weight: "bold" }) + text(x + bw / 2, B + 14, n, { size: 10, anchor: "middle", fill: C.muted });
    });
    b += line(bx - 10, B, bx + 2 * bw + 40, B, C.rule) + line(bx - 10, YB(1), bx + 2 * bw + 40, YB(1), C.rule, 'stroke-dasharray="3 3"') + text(bx - 14, YB(1) + 4, "100%", { fill: C.muted, size: 9, anchor: "end" }) + text(bx - 14, B + 4, "0%", { fill: C.muted, size: 9, anchor: "end" }) + text(bx - 10, 16, "per single annotator (own 0-100% scale)", { fill: C.muted, size: 10 });
    pic.innerHTML = svg(W, H, b);
    read.innerHTML = s.copy ? `<b>every annotator pastes the pair into the same chatbot</b>: <span class="big">agreement 100% (variance 0) · labels right ${pct(m.single, 0)}</span> (the bot's accuracy, set by its slider), and more annotators add nothing<br>`
      : `per pair, P(label right) = ${pct(m.p1, 0)} where the cue points the right way, ${pct(m.p0, 0)} where it points the wrong way<br>
      <span class="big">agreement ${pct(m.agreement, 1)} · one label right ${pct(m.single, 1)} · majority of N = ${N} right ${pct(m.majAcc, 1)}</span><br>
      ${m.p0 < 0.5 ? `<b>shared bias wins</b>: where the cue is wrong most annotators follow it, so the majority picks the cue's answer ${pct(m.cueWins, 1)} of the time and more annotators make that <i>more</i> certain (limit ${pct(m.limit, 0)})` : `<b>noise, not bias</b>: on every pair most annotators are right, so independent errors average out as N grows (limit ${pct(m.limit, 0)})`}<br>`;
    read.innerHTML += `agreement measures how much annotators differ (variance); whether their shared answer is right (bias) is a separate number<br>
      <span class="muted small">provenance: fixture:lecture_15--annotator-pool · video 56:38-57:59 ("It tells you the variance", "all using ChatGPT, the variance will also be zero") · lecture_15.pdf:p43-p44 (annotators matter "a lot", "even for many annotators", images) · the cue, b, a, c and the bot's accuracy are a schematic device with your values, not the lecture's.</span>`;
  };
  const pf = v => pct(v, 0);
  root.append(el("div", { class: "widget" }, el("div", { class: "controls" },
    slider("shared bias b: chance an annotator just follows the style cue", 0, 1, s.b, 0.01, v => { s.b = v; draw(); }, pf),
    slider("content accuracy a when judging content", 0.5, 1, s.a, 0.01, v => { s.a = v; draw(); }, pf),
    slider("pairs where the cue points at the better response, c", 0, 1, s.c, 0.01, v => { s.c = v; draw(); }, pf),
    slider("annotators per pair N (log)", 0, 3.7, s.lN, 0.05, v => { s.lN = v; draw(); }, v => fmt(Nof(v), 0)),
    check("all annotators copy one chatbot's verdict", s.copy, v => { s.copy = v; draw(); }),
    slider("that chatbot's accuracy", 0.5, 1, s.botAcc, 0.01, v => { s.botAcc = v; draw(); }, pf),
    read), pic)); draw();
};
