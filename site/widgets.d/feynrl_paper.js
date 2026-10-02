// Widgets for thread feynrl_paper (Fakoor, Aubry, Stranges, Smola, "Trust the Batch, On- or Off-Policy", arXiv:2605.12380v1).
// Register as "fixture:<id>" -> (root, notice) => void. Each widget has a pure model in MODELS (no DOM) that
// tools/check_widgets.mjs tests against this thread's stored answers.
// Sources (papers/feynrl/sources.json): the paper's page images 4blue2brown/tmp/pdfs/feynrl/page-0N.png / results-0N.png,
// cited as feynrl:p<N> with the equation as printed; the FeynRL code at external/FeynRL @ dfe85351, cited as path:L<n>.
// The paper prints no ε for its derivations; the default 0.2 is the band [0.8, 1.2] this thread's prompts use ("prompts"),
// and 0.4 is the GRPO clip of the temperature and FP8 runs (Table 3, read from the arXiv HTML; unverified in discovered.json).
import { el, fmt, slider } from "../widgets_lib.js";

export const MODELS = {};
export const WIDGETS = {};

const C = { ink: "var(--ink)", muted: "var(--muted)", rule: "var(--rule)", a: "var(--accent)", b: "var(--accent2)", ok: "var(--ok)", hi: "var(--hilite)" };
const rect = (x, y, w, h, fill, extra = "") => `<rect x="${x.toFixed(1)}" y="${y.toFixed(1)}" width="${Math.max(0, w).toFixed(1)}" height="${Math.max(0, h).toFixed(1)}" style="fill:${fill}" ${extra}/>`;
const text = (x, y, s, o = {}) => `<text x="${x.toFixed(1)}" y="${y.toFixed(1)}" style="fill:${o.fill || C.ink};font:${o.weight || ""} ${o.size || 12}px var(--sans)" text-anchor="${o.anchor || "start"}">${s}</text>`;
const line = (x1, y1, x2, y2, stroke, extra = "") => `<line x1="${x1.toFixed(1)}" y1="${y1.toFixed(1)}" x2="${x2.toFixed(1)}" y2="${y2.toFixed(1)}" style="stroke:${stroke};stroke-width:1.5" ${extra}/>`;
const path = (pts, stroke, extra = "", w = 2) => pts.length ? `<path d="${pts.map((p, i) => `${i ? "L" : "M"}${p[0].toFixed(1)},${p[1].toFixed(1)}`).join("")}" style="fill:none;stroke:${stroke};stroke-width:${w}" ${extra}/>` : "";
const svg = (w, h, body) => `<svg viewBox="0 0 ${w} ${h}" width="100%" style="max-width:${w}px;border:1px solid var(--rule);border-radius:6px;background:#fff" role="img">${body}</svg>`;
const check = (label, value, onChange) => { const i = el("input", { type: "checkbox" }); i.checked = value; i.addEventListener("change", () => onChange(i.checked)); return el("label", {}, i, ` ${label}`); };
const sgn = (x, d = 3) => (x > 1e-12 ? "+" : x < -1e-12 ? "−" : "") + fmt(Math.abs(x), d);
const TINY = 1e-12;

// ===================================================== shared per-token rules ==
// The weight w that multiplies A·∇log π_θ(y_t|c_<t) in each objective's gradient (the loss is −w·log π·A, so the loss
// gradient is c·g with c = −w·A and g = ∇log π_θ). ρ_t = π_θ/π_b (Eq. 7, feynrl:p4).
//  GRPO/DAPO, Eq. 4 / Eqs. 9-10 (feynrl:p4-p5; algs/GRPO/grpo.py:L174-L176): −min(ρA, clip(ρ, 1−ε_l, 1+ε_h)A). When the clipped
//    constant is the min (A > 0 and ρ > 1+ε_h, or A < 0 and ρ < 1−ε_l) the token's gradient is exactly 0; otherwise ∇(ρA) = ρA∇log π,
//    so w = ρ (the identity ∇ρ = ρ∇log π, supp-score-function-weighting).
//  CISPO (algs/CISPO/cispo.py:L174-L175): −sg(clamp(ρ, 1−ε_l, 1+ε_h))·log π·A, so w = clamp(ρ) and no token is ever zeroed.
//  P3O, Eq. 12 (feynrl:p5; algs/P3O/p3o.py:L207-L208): −sg(min{ρ, e_B})·log π·A (code: clamp(ratio, 0, ess_factor)), so w = min(ρ, e_B).
//  Behavioral KL of Eq. 12, weight (1 − e_B) (p3o.py:L200-L202, L223), with FeynRL's per-token estimator
//    k = log ρ + π_b/π_θ − 1, exponent clamped to ±10 (algs/RL/common.py:L131-L138). Its derivative in log π_θ is 1 − 1/ρ
//    (0 for the exp term once the clamp binds), so the KL adds (1 − e_B)(1 − 1/ρ) to c: it pushes ρ back toward 1.
function grpoClipped(rho, A, epsL, epsH) { return (A > TINY && rho > 1 + epsH + TINY) || (A < -TINY && rho < 1 - epsL - TINY); }
function weights(rho, A, epsL, epsH, eB) {
  const clipped = grpoClipped(rho, A, epsL, epsH);
  return { clipped, grpo: clipped ? 0 : rho, cispo: Math.min(Math.max(rho, 1 - epsL), 1 + epsH), p3o: Math.min(Math.max(rho, 0), eB), ratio: rho };
}
function klTerm(rho) {
  const lr = Math.log(rho), ex = Math.min(10, Math.max(-10, -lr)), bind = -lr > 10 || -lr < -10;
  return { k: lr + Math.exp(ex) - 1, dk: bind ? 1 : 1 - Math.exp(ex) };
}

// ========================================================= 1. token weight ==
function tokenWeight({ rho, A, epsL = 0.2, epsH = 0.2, eB = 1, kl = false }) {
  const w = weights(rho, A, epsL, epsH, eB), K = rho > 0 ? klTerm(rho) : { k: Infinity, dk: -Infinity };
  const klc = (1 - eB) * K.dk;
  return { ...w, wGRPO: w.grpo, wCISPO: w.cispo, wP3O: w.p3o, wRatio: rho,
    cGRPO: -w.grpo * A, cCISPO: -w.cispo * A, cP3O: -w.p3o * A, cRatio: -rho * A, klCoef: 1 - eB, kHat: K.k, cKL: klc,
    cP3Ototal: -w.p3o * A + (kl ? klc : 0), deadGRPO: w.clipped || Math.abs(rho) < TINY || Math.abs(A) < TINY ? 1 : 0 };
}
MODELS["fixture:feynrl_paper--token-weight"] = {
  fn: tokenWeight,
  cases: [
    { args: { rho: 1.5, A: 1 }, pick: "deadGRPO", expect: 1 },                         // fixed-clip predict: (1.5, +1) zero
    { args: { rho: 0.3, A: 1 }, pick: "wGRPO", expect: 0.3 },                          // ... (0.3, +1) non-zero, weight ρ
    { args: { rho: 0.3, A: -1 }, pick: "deadGRPO", expect: 1 },                        // ... (0.3, −1) zero
    { args: { rho: 1.5, A: 1 }, pick: "cRatio", expect: -1.5 },                        // supp-score predict (a): −1.5 g
    { args: { rho: 1.5, A: 1, eB: 1.5 }, pick: "cP3O", expect: -1.5 },                 // (b) detached sg(ρ) = 1.5: same −1.5 g
    { args: { rho: 1.5, A: 1, eB: 0.645 }, pick: "cP3O", expect: -0.645 },                      // supp-score transfer: −0.645 g (short prompt)
    { args: { rho: 1.5, A: 1, eB: 0.645 }, pick: "cGRPO", expect: 0, tol: 1e-9 },     // ... PPO at ε = 0.2 gives 0
    { args: { rho: 2.5, A: -0.6 }, pick: "cRatio", expect: 1.5, tol: 0.05, from: "feynrl_paper:supp-score-function-weighting:check" },
    { args: { rho: 4, A: 1, eB: 0.645 }, pick: "wP3O", expect: 0.645 },                 // p3o-objective predict: ρ = 4 → 0.645
    { args: { rho: 1.0, A: 1, eB: 0.645 }, pick: "wP3O", expect: 0.645 },               // ... ρ = 1 is capped too
    { args: { rho: 0.3, A: 1, eB: 0.645 }, pick: "wP3O", expect: 0.3 },                 // ... ρ = 0.3 passes through
    { args: { rho: 4, A: 1, eB: 0.645 }, pick: "klCoef", expect: 0.355 },               // ... KL coefficient 0.355
    { args: { rho: 1.1, A: 1, eB: 0.995 }, pick: "wP3O", expect: 0.995 },               // p3o-objective transfer: 0.995
    { args: { rho: 1.1, A: 1, eB: 0.995, epsL: 0.05, epsH: 0.05 }, pick: "wGRPO", expect: 0, tol: 1e-9 }, // ... PPO ε = 0.05: zero
    { args: { rho: 0.5, A: -1, eB: 0.645 }, pick: "wGRPO", expect: 0, tol: 1e-9 },     // vs-clip-family transfer: GRPO 0
    { args: { rho: 0.5, A: -1, eB: 0.645 }, pick: "wCISPO", expect: 0.8 },              // ... CISPO 0.8
    { args: { rho: 0.5, A: -1, eB: 0.645 }, pick: "wP3O", expect: 0.5 },                // ... P3O 0.5
    { args: { rho: 1.3, A: 1, eB: 0.9 }, pick: "wGRPO", expect: 0, tol: 1e-9 },        // vs-clip-family check: GRPO 0
    { args: { rho: 1.3, A: 1, eB: 0.9 }, pick: "wCISPO", expect: 1.2 },                 // ... CISPO 1.2
    { args: { rho: 1.3, A: 1, eB: 0.9 }, pick: "wP3O", expect: 0.9 },                   // ... P3O 0.9
    { args: { rho: 1.6, A: 1, epsL: 0.6, epsH: 0.6 }, pick: "deadGRPO", expect: 0 },    // two-concerns check: ε = 0.6, ρ = 1.6 still live
    { args: { rho: 1.61, A: 1, epsL: 0.6, epsH: 0.6 }, pick: "deadGRPO", expect: 1 },   // ... and just past 1.6 it stops
    { args: { rho: 1.35, A: -1, epsL: 0.2, epsH: 0.28 }, pick: "deadGRPO", expect: 0 }, // edge: A < 0 above the band is NOT clipped
    { args: { rho: 4, A: 1, eB: 0.645 }, pick: "cKL", expect: 0.26625 },                // KL pull: 0.355·(1 − 1/4), pushes log π down
    { args: { rho: 4, A: 1, eB: 0.645, kl: true }, pick: "cP3Ototal", expect: -0.37875 }, // net: −0.645 + 0.266, still up but less
    { args: { rho: 1, A: 1, eB: 0.645 }, pick: "cKL", expect: 0, tol: 1e-9 },          // KL gradient vanishes at ρ = 1
    { args: { rho: 4, A: 1, eB: 1 }, pick: "cKL", expect: 0, tol: 1e-9 },              // and for e_B = 1 (fresh batch): no pull
  ],
};
WIDGETS["fixture:feynrl_paper--token-weight"] = (root) => {
  const s = { rho: 0.6, A: -1, epsL: 0.2, epsH: 0.2, eB: 0.85, kl: false };   // default: no prompt's case (a GRPO-dead token with A < 0)
  const pic = el("div"), read = el("div", { class: "readout" });
  const draw = () => {
    const m = tokenWeight(s), W = 660, H = 300, L = 40, R = 520, T = 40, B = 250, rmax = 4.5;
    const push = r => { const t = tokenWeight({ ...s, rho: r }); return { g: t.wGRPO * s.A, c: t.wCISPO * s.A, p: t.wP3O * s.A, pk: -t.cP3Ototal, dead: t.clipped }; };
    let ymax = Math.max(0.5, Math.abs(s.A) * rmax) * 1.08;
    if (s.kl) for (let r = 0.25; r <= rmax; r += 0.05) ymax = Math.max(ymax, Math.abs(push(r).pk) * 1.08);   // below ρ ≈ 0.25 the KL pull (1 − 1/ρ) runs off the axis; it is clipped there
    const X = r => L + (r / rmax) * (R - L), Y = v => T + (ymax - Math.max(-ymax, Math.min(ymax, v))) / (2 * ymax) * (B - T);
    let b = rect(X(1 - s.epsL), T, X(1 + s.epsH) - X(1 - s.epsL), B - T, C.hi, 'fill-opacity="0.15"');
    b += line(L, Y(0), R, Y(0), C.rule) + line(L, T, L, B, C.rule) + line(X(1), T, X(1), B, C.rule, 'stroke-dasharray="3 3"');
    b += line(X(s.eB), T, X(s.eB), B, C.ok, 'stroke-dasharray="2 3"') + text(X(s.eB), T - 6, `e_B = ${fmt(s.eB, 3)}`, { fill: C.ok, size: 11, anchor: "middle" });
    const seg = (f, keep = () => true) => { const out = []; let cur = []; for (let r = 0.01; r <= rmax + 1e-9; r += 0.01) { if (keep(r)) cur.push([X(r), Y(f(r))]); else if (cur.length) { out.push(cur); cur = []; } } if (cur.length) out.push(cur); return out; };
    b += path(seg(r => r * s.A)[0], C.a, 'stroke-opacity="0.25"');
    for (const p of seg(r => push(r).g, r => !push(r).dead)) b += path(p, C.a);
    for (const p of seg(() => 0, r => push(r).dead)) b += path(p, C.b, 'stroke-opacity="0.6"', 5);
    b += path(seg(r => push(r).c)[0], C.muted, 'stroke-dasharray="6 3"') + path(seg(r => push(r).p)[0], C.ok);
    if (s.kl) b += path(seg(r => push(r).pk)[0], C.ok, 'stroke-dasharray="2 2" stroke-opacity="0.8"');
    const dot = (v, col) => `<circle cx="${X(Math.min(rmax, s.rho)).toFixed(1)}" cy="${Y(v).toFixed(1)}" r="4.5" style="fill:${col};stroke:#fff;stroke-width:1"/>`;
    b += dot(m.wCISPO * s.A, C.muted) + dot(m.wGRPO * s.A, m.clipped ? C.b : C.a) + dot(s.kl ? -m.cP3Ototal : m.wP3O * s.A, C.ok);
    b += text(X(1 - s.epsL), B + 14, "1−ε_l", { fill: C.muted, size: 10, anchor: "end" }) + text(X(1 + s.epsH), B + 14, "1+ε_h", { fill: C.muted, size: 10 }) + text(X(1), B + 26, "ρ = 1", { fill: C.muted, size: 10, anchor: "middle" });
    for (const r of [0, 2, 3, 4]) b += text(X(r), B + 14, String(r), { fill: C.muted, size: 10, anchor: "middle" });
    b += text(R, B + 42, "ρ_t = π_θ / π_b", { fill: C.muted, size: 11, anchor: "end" });
    b += text(L - 6, Y(0) + 4, "0", { fill: C.muted, size: 10, anchor: "end" });
    const key = [["GRPO/DAPO (Eq. 4)", C.a, ""], ["… zero gradient", C.b, ""], ["CISPO", C.muted, "6 3"], ["P3O (Eq. 12)", C.ok, ""]].concat(s.kl ? [["P3O + (1−e_B)·KL", C.ok, "2 2"]] : []);
    key.forEach(([n, col, da], i) => { const y = T + 8 + i * 18; b += line(R + 14, y, R + 36, y, col, da ? `stroke-dasharray="${da}"` : "") + text(R + 40, y + 4, n, { size: 10 }); });
    b += text(L, 16, `push on log π_θ = w·A per unit ∇log π (curves for A = ${sgn(s.A, 2)}; dots at ρ = ${fmt(s.rho, 3)}); band shaded`, { fill: C.muted, size: 11 });
    pic.innerHTML = svg(W, H, b);
    const side = s.A > 0 ? `ρ > 1+ε_h = ${fmt(1 + s.epsH, 3)} with A > 0` : `ρ < 1−ε_l = ${fmt(1 - s.epsL, 3)} with A < 0`;
    const g = Math.abs(s.A) < TINY ? "A = 0: no token gets a policy-gradient push under any objective"
      : m.clipped ? `<b style="color:var(--accent2)">zero gradient</b>: ${side.replace("<", "&lt;")}, so the min picks the flat clipped constant (Eqs. 9-10)`
      : `live, w = ρ = ${fmt(m.wGRPO, 4)} (${(m.rho >= 1 - s.epsL - TINY && m.rho <= 1 + s.epsH + TINY) ? "inside the band" : `outside the band, but A ${s.A > 0 ? "> 0 only clips above" : "&lt; 0 only clips below"}`})`;
    read.innerHTML = `<span class="big">GRPO w = ${fmt(m.wGRPO, 4)} · CISPO w = ${fmt(m.wCISPO, 4)} · P3O w = ${fmt(m.wP3O, 4)}</span><br>
      loss gradient c·g with c = −w·A: GRPO ${sgn(m.cGRPO, 4)} g · CISPO ${sgn(m.cCISPO, 4)} g · P3O ${sgn(m.cP3O, 4)} g · unclipped ratio surrogate −ρA: ${sgn(m.cRatio, 4)} g<br>
      GRPO: ${g}<br>
      CISPO: w = clamp(ρ, 1−ε_l, 1+ε_h) ${m.wCISPO !== m.rho ? "= the band edge (scaled, not zeroed)" : "= ρ (inside the band)"} · P3O: w = min(ρ, e_B) ${m.rho > s.eB ? "= e_B (capped; every ρ above e_B gets the same weight, ρ = 1 included)" : "= ρ (below the cap, passed through)"}<br>
      behavioral KL weight 1 − e_B = ${fmt(m.klCoef, 4)}${s.kl ? ` · per-token k̂ = log ρ + 1/ρ − 1 = ${fmt(m.kHat, 4)} · its term in c: (1 − e_B)(1 − 1/ρ) = ${sgn(m.cKL, 4)} g (${m.cKL > TINY ? "pulls log π down, toward π_b" : m.cKL < -TINY ? "pulls log π up, toward π_b" : "no pull"}) · <b>P3O total c = ${sgn(m.cP3Ototal, 4)} g</b>` : " (tick the box to add the KL term to P3O's gradient)"}<br>
      <span class="muted small">provenance: fixture:feynrl_paper--token-weight · Eq. 4 and Eqs. 9-10 (feynrl:p4-p5), Eq. 12 (feynrl:p5), Table 1 (feynrl:p6); algs/GRPO/grpo.py:L174-L176, algs/CISPO/cispo.py:L174-L175, algs/P3O/p3o.py:L207-L208 and L223, algs/RL/common.py:L131-L138 (KL estimator, exponent clamp ±10). e_B is an input here; the batch-ESS widget computes it from a batch (Eq. 11). Default ε = 0.2 is the prompts' band, not a paper value.</span>`;
  };
  root.append(el("div", { class: "widget" }, el("div", { class: "controls" },
    slider("ratio ρ_t", 0, 4.5, s.rho, 0.01, v => { s.rho = v; draw(); }, v => v.toFixed(2)),
    slider("advantage A", -1.5, 1.5, s.A, 0.1, v => { s.A = v; draw(); }, v => v.toFixed(1)),
    slider("ε_l", 0, 0.8, s.epsL, 0.01, v => { s.epsL = v; draw(); }, v => v.toFixed(2)),
    slider("ε_h", 0, 0.8, s.epsH, 0.01, v => { s.epsH = v; draw(); }, v => v.toFixed(2)),
    slider("batch e_B", 0.05, 1, s.eB, 0.005, v => { s.eB = v; draw(); }, v => v.toFixed(3)),
    check("add P3O's behavioral KL (1 − e_B)·KL(π_θ‖π_b) to its gradient", s.kl, v => { s.kl = v; draw(); }),
    read), pic)); draw();
};

// ============================================================ 2. batch ESS ==
// Eq. 11 (feynrl:p5): ESS(B; θ) = Ê_B[ρ_t]² / Ê_B[ρ_t²] = (Σw)² / (n·Σw²) ∈ [1/|B|, 1] over the valid response tokens B; e_B = sg(ESS).
// FeynRL (algs/P3O/p3o.py:L107-L135): ess = sum_w² / (sum_w_2 + 1e-8) / total, the three sums all-reduced across ranks; returns 1.0
// if no valid token. The cap and the KL weight (1 − e_B) then follow Eq. 12 as in the token-weight widget.
// Batch syntax: tokens separated by commas; a token is a ratio ρ ("1.5") or π_θ/π_b ("0.3/0.6"), optionally "@A" for its
// advantage (default +1). Two transforms act on every log-ratio: a common shift δ (ρ → ρ·e^δ; the paper's account of
// temperature, "shifting the per-token ratio ρ_t by a constant factor across the entire batch", feynrl:p7 §4.3) and a spread s
// (log ρ → s·log ρ, i.e. ρ → ρ^s: ratios that move unevenly); "flip" inverts every ratio (log-ratio sign flipped).
function parseBatch(str) {
  const toks = String(str ?? "").split(",").map(t => t.trim()).filter(Boolean);
  if (!toks.length) return { err: "type at least one token" };
  const out = [];
  for (const t of toks) {
    const [rs, as] = t.split("@").map(x => x.trim());
    let rho;
    if (rs.includes("/")) { const [a, b] = rs.split("/").map(Number); if (!(a >= 0) || !(b > 0)) return { err: `"${t}": π_θ/π_b needs π_θ ≥ 0 and π_b > 0` }; rho = a / b; }
    else rho = Number(rs);
    const A = as === undefined || as === "" ? 1 : Number(as.replace("−", "-"));
    if (!Number.isFinite(rho) || rho < 0) return { err: `"${t}": a ratio must be a number ≥ 0` };
    if (!Number.isFinite(A)) return { err: `"${t}": the advantage after @ must be a number` };
    out.push({ rho, A });
  }
  return { tokens: out };
}
function batchEss({ batch, epsL = 0.2, epsH = 0.2, shift = 0, spread = 1, flip = false }) {
  const p = parseBatch(batch); if (p.err) return { err: p.err };
  if (flip && p.tokens.some(t => t.rho === 0)) return { err: "a ratio of 0 cannot be inverted" };
  const toks = p.tokens.map(t => {
    const r0 = flip ? 1 / t.rho : t.rho;
    const rho = r0 === 0 ? (spread > 0 ? 0 : 1) : Math.exp(spread * Math.log(r0) + shift);
    return { ...t, rho };
  });
  const n = toks.length, sumW = toks.reduce((a, t) => a + t.rho, 0), sumW2 = toks.reduce((a, t) => a + t.rho * t.rho, 0);
  const ess = sumW2 > 0 ? (sumW * sumW) / (n * sumW2) : NaN, essCode = (sumW * sumW) / (sumW2 + 1e-8) / n;
  const eB = Number.isFinite(ess) ? ess : essCode;                                     // all-zero batch: Eq. 11 is 0/0; the code gives 0
  const o = { n, sumW, sumW2, ess, essCode, eB, klCoef: 1 - eB, deadGRPO: 0, deadCISPO: 0, deadP3O: 0, rows: [] };
  toks.forEach((t, i) => {
    const w = weights(t.rho, t.A, epsL, epsH, eB), noA = Math.abs(t.A) < TINY;
    const dG = w.clipped || t.rho === 0 || noA, dC = noA, dP = t.rho === 0 || noA;
    o.deadGRPO += dG; o.deadCISPO += dC; o.deadP3O += dP;
    const share = sumW > 0 ? t.rho / sumW : 0;
    Object.assign(o, { [`rho.${i + 1}`]: t.rho, [`share.${i + 1}`]: share, [`w_grpo.${i + 1}`]: w.grpo, [`w_cispo.${i + 1}`]: w.cispo, [`w_p3o.${i + 1}`]: w.p3o });
    o.rows.push({ ...t, ...w, share, dG, dP });
  });
  o.liveFracGRPO = (n - o.deadGRPO) / n; o.liveFracP3O = (n - o.deadP3O) / n; o.deadFracGRPO = o.deadGRPO / n;
  o.shareMax = Math.max(...o.rows.map(r => r.share));
  if (n >= 2) o.r21 = toks[0].rho > 0 ? toks[1].rho / toks[0].rho : Infinity;
  return o;
}
MODELS["fixture:feynrl_paper--batch-ess"] = {
  fn: batchEss,
  cases: [
    { args: { batch: "4, 1, 1, 1" }, pick: "ess", expect: 0.645, tol: 0.01, from: "feynrl_paper:feynrl-normalized-ess:predict" },
    { args: { batch: "8, 2, 2, 2" }, pick: "ess", expect: 0.645, tol: 0.01, from: "feynrl_paper:feynrl-normalized-ess:transfer" },
    { args: { batch: "4, 1, 1, 1", shift: Math.log(2) }, pick: "ess", expect: 0.6447 },          // same batch via the common shift
    { args: { batch: "4, 1, 1, 1" }, pick: "essCode", expect: 0.6447 },                         // FeynRL's 1e-8 changes nothing visible
    { args: { batch: "1, 0, 0, 0" }, pick: "ess", expect: 0.25, from: "feynrl_paper:supp-ess-bounds:predict" },
    { args: { batch: "0.1, 0.1, 0.1, 0.1" }, pick: "ess", expect: 1, from: "feynrl_paper:supp-ess-bounds:transfer" },
    { args: { batch: "0.05, 0.05, 0.05, 0, 0, 0, 0, 0" }, pick: "ess", expect: 0.375, tol: 0.05, from: "feynrl_paper:supp-ess-bounds:check" },
    { args: { batch: "4, 1, 1, 1" }, pick: "share.1", expect: 0.5714 },                         // importance-ratio predict: 4/7 ≈ 57%
    { args: { batch: "4, 1, 1, 1", flip: true }, pick: "share.1", expect: 0.0769 },             // ... transfer: 0.25/3.25 ≈ 7.7%
    { args: { batch: "4, 1, 1, 1", flip: true }, pick: "rho.1", expect: 0.25 },
    { args: { batch: "0.3/0.6, 0.6/0.3" }, pick: "r21", expect: 4, tol: 0.02, from: "feynrl_paper:feynrl-importance-ratio:check" },
    { args: { batch: "3, 1, 1, 1" }, pick: "w_p3o.2", expect: 0.75, tol: 0.05, from: "feynrl_paper:feynrl-p3o-objective:check" },
    { args: { batch: "3, 1, 1, 1" }, pick: "eB", expect: 0.75 },
    { args: { batch: "1.1, 1.0, 0.9, 1.0" }, pick: "eB", expect: 0.995, tol: 0.001 },           // p3o-objective transfer: e_B ≈ 0.995
    { args: { batch: "1.1, 1.0, 0.9, 1.0" }, pick: "klCoef", expect: 0.004975, tol: 0.02 },     // ... KL coefficient ≈ 0.005
    { args: { batch: "1.1, 1.0, 0.9, 1.0" }, pick: "w_p3o.1", expect: 0.995, tol: 0.001 },      // ... ρ = 1.1 capped just below 1.1
    { args: { batch: "1.25@1, 1.35@1, 0.75@1, 0.75@-1, 1.35@-1, 0.85@-1", epsL: 0.2, epsH: 0.28 }, pick: "deadGRPO", expect: 2, tol: 0.05, from: "feynrl_paper:feynrl-fixed-clip-regimes:check" },
    { args: { batch: "1.25@1, 1.35@1, 0.75@1, 0.75@-1, 1.35@-1, 0.85@-1", epsL: 0.2, epsH: 0.28 }, pick: "deadP3O", expect: 0 }, // P3O zeroes none
    { args: { batch: "1, 1, 1, 1", shift: Math.log(1.5) }, pick: "deadGRPO", expect: 4 },       // common shift past 1+ε with A > 0: all dead
    { args: { batch: "1, 1, 1, 1", shift: Math.log(1.5) }, pick: "ess", expect: 1 },            // ... while ESS stays exactly 1
    { args: { batch: "1@1, 1@-1", shift: Math.log(1.5) }, pick: "deadGRPO", expect: 1 },        // ... only the guarded sign stops
    { args: { batch: "1@1, 1@-1", shift: Math.log(1.39), epsL: 0.4, epsH: 0.4 }, pick: "deadGRPO", expect: 0 }, // Fig. 2 clip 0.4: inside
    { args: { batch: "1@1, 1@-1", shift: Math.log(1.41), epsL: 0.4, epsH: 0.4 }, pick: "deadGRPO", expect: 1 }, // ... just past ln 1.4
    { args: { batch: "1@1, 1@-1, 1@1, 1@-1", shift: 0.4, epsL: 0.4, epsH: 0.4 }, pick: "liveFracGRPO", expect: 0.5 },  // experiments notice: halves
    { args: { batch: "1@1, 1@-1, 1@1, 1@-1", shift: -0.6, epsL: 0.4, epsH: 0.4 }, pick: "liveFracGRPO", expect: 0.5 }, // ... on the low side too
    { args: { batch: "1@1, 1@-1, 1@1, 1@-1", shift: 0.4, epsL: 0.4, epsH: 0.4 }, pick: "eB", expect: 1 },             // ... e_B stays 1
    { args: { batch: "4, 1, 1, 1", spread: 0 }, pick: "ess", expect: 1 },                       // spread 0: all ratios 1
    { args: { batch: "4, 1, 1, 1", spread: 2 }, pick: "ess", expect: 0.3485, tol: 0.01 },       // (16,1,1,1): 361/(4·259)
    { args: { batch: "0, 0" }, pick: "eB", expect: 0, tol: 1e-9 },                              // all-zero: Eq. 11 undefined, code gives 0
  ],
};
const setSlider = (lab, v, f) => { const i = lab.querySelector("input"), o = lab.querySelector(".readout"); i.value = v; o.textContent = f ? f(v) : v; };
WIDGETS["fixture:feynrl_paper--batch-ess"] = (root) => {
  const s = { batch: "1.6, 1.1, 0.9, 0.7@-1, 1.3@-1, 1.0", epsL: 0.2, epsH: 0.2, shift: 0, spread: 1, flip: false };  // default: no prompt's batch
  const pic = el("div"), sweep = el("div", { style: "margin-top:6px" }), read = el("div", { class: "readout" }), err = el("div", { class: "muted small" });
  const inp = el("input", { type: "text", style: "width:100%;max-width:560px" }); inp.value = s.batch;
  const f2 = v => v.toFixed(2);
  const draw = () => {
    const m = batchEss(s); err.textContent = m.err || "";
    if (m.err) { pic.innerHTML = ""; sweep.innerHTML = ""; read.innerHTML = ""; return; }
    // --- per-token rows: ρ on a log2 axis from 1/16 to 16, ρ = 0 parked at the left edge
    const rows = m.rows.slice(0, 16), RH = 20, W = 680, T = 44, H = T + rows.length * RH + 34, xa = 100, xb = 380;
    const lx = r => r <= 0 ? xa - 22 : xa + (Math.max(-4, Math.min(4, Math.log2(r))) + 4) / 8 * (xb - xa);
    let b = text(10, 14, `tokens (A after @, default +1) · ρ_t on a log axis · band [1−ε_l, 1+ε_h] shaded · e_B dashed`, { fill: C.muted, size: 11 });
    b += rect(lx(1 - s.epsL), T - 6, lx(1 + s.epsH) - lx(1 - s.epsL), rows.length * RH + 4, C.hi, 'fill-opacity="0.18"');
    b += line(lx(1), T - 8, lx(1), T + rows.length * RH, C.rule, 'stroke-dasharray="3 3"') + line(lx(m.eB), T - 8, lx(m.eB), T + rows.length * RH, C.ok, 'stroke-dasharray="2 3"');
    const cols = [["share", 420], ["GRPO w", 490], ["CISPO w", 560], ["P3O w", 630]];
    cols.forEach(([n, x]) => { b += text(x, T - 12, n, { fill: C.muted, size: 10, anchor: "end" }); });
    rows.forEach((r, i) => {
      const y = T + i * RH + 10;
      b += text(10, y + 4, `t${i + 1} (A ${sgn(r.A, 2)})`, { size: 11 }) + line(xa, y, xb, y, C.rule, 'stroke-opacity="0.4"');
      b += `<circle cx="${lx(r.rho).toFixed(1)}" cy="${y.toFixed(1)}" r="5" style="fill:${r.dG ? C.b : C.a}"/>`;
      if (r.rho > 16 || (r.rho > 0 && r.rho < 1 / 16)) b += text(lx(r.rho) + (r.rho > 1 ? 7 : -7), y + 4, r.rho > 1 ? "▶" : "◀", { fill: C.muted, size: 9, anchor: r.rho > 1 ? "start" : "end" });
      b += text(420, y + 4, `${fmt(100 * r.share, 1)}%`, { size: 11, anchor: "end" });
      b += text(490, y + 4, r.dG ? "0 (stopped)" : fmt(r.grpo, 3), { size: 11, anchor: "end", fill: r.dG ? C.b : C.ink, weight: r.dG ? "bold" : "" });
      b += text(560, y + 4, fmt(r.cispo, 3), { size: 11, anchor: "end" }) + text(630, y + 4, r.dP ? "0" : fmt(r.p3o, 3), { size: 11, anchor: "end", fill: r.dP ? C.b : C.ok });
    });
    const yb = T + rows.length * RH + 14;
    for (const [r, t] of [[0, "0"], [0.25, "1/4"], [1, "1"], [4, "4"], [16, "16"]]) b += text(lx(r), yb, t, { fill: C.muted, size: 10, anchor: "middle" });
    if (m.rows.length > 16) b += text(10, yb + 14, `(first 16 of ${m.rows.length} tokens drawn; the sums use all)`, { fill: C.muted, size: 10 });
    pic.innerHTML = svg(W, H, b);
    // --- sweeps: the same batch under a common shift δ (left) and a spread s (right)
    const PW = 320, PH = 150, pad = 34;
    const panel = (x0, title, lo, hi, cur, f, xl) => {
      const X = v => x0 + pad + (v - lo) / (hi - lo) * (PW - pad - 12), Y = v => 22 + (1 - v) * (PH - 52);
      let g = text(x0 + 8, 14, title, { fill: C.muted, size: 11 }) + line(X(lo), Y(0), X(hi), Y(0), C.rule) + line(X(lo), Y(0), X(lo), Y(1), C.rule);
      g += text(X(lo) - 4, Y(1) + 4, "1", { fill: C.muted, size: 10, anchor: "end" }) + text(X(lo) - 4, Y(0) + 4, "0", { fill: C.muted, size: 10, anchor: "end" });
      const pts = { g: [], p: [], e: [] };
      for (let k = 0; k <= 120; k++) { const v = lo + (hi - lo) * k / 120, q = f(v); if (q.err) continue; pts.g.push([X(v), Y(q.liveFracGRPO)]); pts.p.push([X(v), Y(q.liveFracP3O)]); pts.e.push([X(v), Y(q.eB)]); }
      g += path(pts.p, C.ok, 'stroke-width="3" stroke-opacity="0.5"') + path(pts.g, C.a) + path(pts.e, C.ok, 'stroke-dasharray="5 3"');
      g += line(X(cur), Y(0), X(cur), Y(1), C.ink, 'stroke-dasharray="2 2" stroke-opacity="0.6"');
      g += text(X(lo), PH - 12, fmt(lo, 2), { fill: C.muted, size: 10, anchor: "middle" }) + text(X(hi), PH - 12, fmt(hi, 2), { fill: C.muted, size: 10, anchor: "middle" }) + text(X((lo + hi) / 2), PH - 12, xl, { fill: C.muted, size: 10, anchor: "middle" });
      return g;
    };
    let sb = panel(0, "common shift δ: every ρ × e^δ (feynrl:p7)", -1, 1, s.shift, v => batchEss({ ...s, shift: v }), "δ");
    sb += panel(PW + 20, "spread s: every log ρ × s", 0, 3, s.spread, v => batchEss({ ...s, spread: v }), "s");
    [["GRPO: share of tokens with gradient", C.a, ""], ["P3O: share of tokens with gradient", C.ok, ""], ["e_B (Eq. 11)", C.ok, "5 3"]].forEach(([n, col, da], i) => {
      sb += line(10 + i * 230, PH + 10, 30 + i * 230, PH + 10, col, da ? `stroke-dasharray="${da}"` : "") + text(34 + i * 230, PH + 14, n, { size: 10 }); });
    sweep.innerHTML = svg(2 * PW + 20, PH + 22, sb);
    const fresh = m.eB > 0.95;
    read.innerHTML = `n = ${m.n} · Σw = ${fmt(m.sumW, 4)} · Σw² = ${fmt(m.sumW2, 4)} · <span class="big">ESS = (Σw)² / (n·Σw²) = ${Number.isFinite(m.ess) ? fmt(m.ess, 4) : "not defined (all ratios 0; FeynRL gives 0)"}</span> · bounds [1/n, 1] = [${fmt(1 / m.n, 4)}, 1]<br>
      e_B = ${fmt(m.eB, 4)} → P3O caps every weight at ${fmt(m.eB, 4)} and puts ${fmt(m.klCoef, 4)} on the behavioral KL · <b>${fresh ? "e_B near 1: the cap barely binds and the KL pull is small" : "e_B below 1: ratios above " + fmt(m.eB, 3) + " are capped and the KL pull is " + fmt(m.klCoef, 3)}</b><br>
      tokens with exactly zero gradient: <b style="color:var(--accent2)">GRPO ${m.deadGRPO} of ${m.n}</b> (ε_l = ${f2(s.epsL)}, ε_h = ${f2(s.epsH)}) · CISPO ${m.deadCISPO} · P3O ${m.deadP3O}${m.deadP3O ? " (ρ = 0 or A = 0)" : ""} · largest weight share ${fmt(100 * m.shareMax, 1)}%${m.n >= 2 && Number.isFinite(m.r21) ? ` · ρ₂/ρ₁ = ${fmt(m.r21, 4)}` : ""}<br>
      ${Math.abs(s.shift) > 1e-9 ? `common shift: every ρ multiplied by e^δ = ${fmt(Math.exp(s.shift), 4)}; Σw and Σw² changed, <b>ESS did not</b> (Eq. 11 is scale-free), while GRPO's band sees the shifted ratios<br>` : ""}${s.flip ? "flip: every log-ratio negated, ρ → 1/ρ<br>" : ""}
      <span class="muted small">provenance: fixture:feynrl_paper--batch-ess · Eq. 11 (feynrl:p5) and algs/P3O/p3o.py:L107-L135 (calculate_ess, + 1e-8); weights as in token-weight (Eq. 4, Eq. 12, CISPO). The δ sweep is the paper's own description of temperature (§4.3, feynrl:p7: a constant factor on every ρ_t); the s sweep is a device for ratios that move unevenly. Neither simulates training. Tokens are typed, not taken from any run.</span>`;
  };
  inp.addEventListener("input", () => { s.batch = String(inp.value ?? ""); draw(); });
  const sl = {
    epsL: slider("ε_l", 0, 0.8, s.epsL, 0.01, v => { s.epsL = v; draw(); }, f2), epsH: slider("ε_h", 0, 0.8, s.epsH, 0.01, v => { s.epsH = v; draw(); }, f2),
    shift: slider("common shift δ (ρ × e^δ)", -1, 1, s.shift, 0.01, v => { s.shift = v; draw(); }, f2), spread: slider("spread s (log ρ × s)", 0, 3, s.spread, 0.05, v => { s.spread = v; draw(); }, f2),
  };
  const flipBox = check("flip every log-ratio (ρ → 1/ρ)", false, v => { s.flip = v; draw(); });
  const presets = [["one 4× token", "4, 1, 1, 1"], ["…all ×2", "8, 2, 2, 2"], ["one-hot", "1, 0, 0, 0"], ["near-fresh", "1.1, 1.0, 0.9, 1.0"],
    ["six tokens, both signs", "1.25@1, 1.35@1, 0.75@1, 0.75@-1, 1.35@-1, 0.85@-1"], ["π pairs", "0.3/0.6, 0.6/0.3"], ["fresh, both signs", "1@1, 1@-1, 1@1, 1@-1"]];
  const reset = () => { Object.assign(s, { shift: 0, spread: 1, flip: false }); setSlider(sl.shift, 0, f2); setSlider(sl.spread, 1, f2); flipBox.querySelector("input").checked = false; };
  const pbox = el("div", { style: "margin-top:6px" }, ...presets.map(([n, t]) => el("button", { onclick: () => { s.batch = t; inp.value = t; reset(); draw(); } }, n)),
    el("button", { onclick: () => { s.epsL = s.epsH = 0.4; setSlider(sl.epsL, 0.4, f2); setSlider(sl.epsH, 0.4, f2); draw(); } }, "GRPO clip 0.4 (Table 3, temperature runs)"));
  root.append(el("div", { class: "widget" }, el("div", { class: "controls" },
    el("label", {}, "batch: ratios ρ or π_θ/π_b, optional @A, separated by commas ", inp), err, pbox,
    sl.epsL, sl.epsH, sl.shift, sl.spread, flipBox, read), pic, sweep)); draw();
};
