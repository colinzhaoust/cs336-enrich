// Widgets for thread lecture_12 (CS336 L12, evaluation). Register as "fixture:<id>" -> (root, notice) => void.
// Each widget has a pure model in MODELS (no DOM) that tools/check_widgets.mjs tests against the KPs' stored answers.
// Sources: official/lectures/lecture_12.py (cited as lecture_12.py:L<n>), lectures/lecture_12/transcript.json
// (cited as video M:SS), official/lectures/images/*.png (the lecture's figures), and the KP prompts themselves, whose
// worked numbers are used as presets (marked "KP" below). Every other default is a learner control with no lecture
// value, and is labelled "demo" where it is shown.
import { el, fmt, slider } from "../widgets_lib.js";

export const MODELS = {};
export const WIDGETS = {};

const C = { ink: "var(--ink)", muted: "var(--muted)", rule: "var(--rule)", a: "var(--accent)", b: "var(--accent2)", ok: "var(--ok)", hi: "var(--hilite)" };
const rect = (x, y, w, h, fill, extra = "") => `<rect x="${x.toFixed(1)}" y="${(+y).toFixed(1)}" width="${Math.max(0, w).toFixed(1)}" height="${Math.max(0, h).toFixed(1)}" style="fill:${fill}" ${extra}/>`;
const text = (x, y, s, opts = {}) => `<text x="${x.toFixed(1)}" y="${(+y).toFixed(1)}" style="fill:${opts.fill || C.ink};font:${opts.weight || ""} ${opts.size || 12}px var(--sans)" text-anchor="${opts.anchor || "start"}">${s}</text>`;
const line = (x1, y1, x2, y2, stroke, extra = "") => `<line x1="${x1.toFixed(1)}" y1="${y1.toFixed(1)}" x2="${x2.toFixed(1)}" y2="${y2.toFixed(1)}" style="stroke:${stroke};stroke-width:1.5" ${extra}/>`;
const svg = (w, h, body) => `<svg viewBox="0 0 ${w} ${h}" width="100%" style="max-width:${w}px;border:1px solid var(--rule);border-radius:6px;background:#fff" role="img">${body}</svg>`;
const check = (label, value, onChange) => { const i = el("input", { type: "checkbox" }); i.checked = value; i.addEventListener("change", () => onChange(i.checked)); return el("label", {}, i, ` ${label}`); };
const select = (label, options, value, onChange) => {
  const s = el("select", {}, ...options.map(([v, n]) => { const o = el("option", { value: v }, n); if (v === value) o.selected = true; return o; }));
  s.addEventListener("change", () => onChange(s.value)); return el("label", {}, `${label} `, s);
};
const setSlider = (lab, v, fmtv = (x) => x) => { const inp = lab.querySelector("input"), out = lab.querySelector(".readout"); inp.value = v; out.textContent = fmtv(+inp.value); };
const esc = (s) => String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");

// ================================================= 1. conditional perplexity ==
// Perplexity (1/p(D))^(1/|D|) (lecture_12.py:L62) over the scored tokens only. With the first k tokens treated as the
// prompt, this is the conditional perplexity of L90, written there as p(response | prompt)^(1/|response|) (the inverse
// is implied, as the KP's check makes explicit). "Charges you bits for every bit of a deviation" is video 12:29-12:41,
// so the readout gives the same number as bits: ppl = 2^(mean −log2 p over scored tokens).
// Presets (KP): perplexity-sufficiency-vs-conditional predict (Stanford, X and Y) and check (France). Tokens per L88.
function condPpl({ p, k = 0 }) {
  const scored = p.slice(k), n = scored.length;
  const totalBits = scored.reduce((s, x) => s - Math.log2(x), 0);
  const bits = totalBits / n;
  return { n, totalBits, bits, nats: bits * Math.LN2, ppl: 2 ** bits, pScored: scored.reduce((s, x) => s * x, 1) };
}
const PX = [0.5, 0.5, 0.5, 0.5, 0.1], PY = [0.25, 0.25, 0.25, 0.25, 0.4], PF = [0.1, 0.1, 0.1, 0.1, 0.1, 0.5, 0.8];
MODELS["fixture:lecture_12--cond-perplexity"] = {
  fn: condPpl,
  cases: [
    { args: { p: PF, k: 5 }, pick: "ppl", expect: 1.58, from: "lecture_12:perplexity-sufficiency-vs-conditional:check" },
    { args: { p: PX, k: 0 }, pick: "ppl", expect: 2.7595, tol: 0.002 },   // predict: X full sentence, (1/0.00625)^(1/5)
    { args: { p: PY, k: 0 }, pick: "ppl", expect: 3.6411, tol: 0.002 },   // predict: Y full sentence, (1/0.0015625)^(1/5)
    { args: { p: PX, k: 4 }, pick: "ppl", expect: 10 },                   // predict: X conditional on '1885'
    { args: { p: PY, k: 4 }, pick: "ppl", expect: 2.5 },                  // predict: Y conditional on '1885'
    { args: { p: PX, k: 0 }, pick: "pScored", expect: 0.00625 },          // predict's why: p_X(D)
    { args: { p: PF, k: 0 }, pick: "ppl", expect: 5.9024, tol: 0.002 },   // check's why: full sequence dominated by the 0.1s
    { args: { p: [0.5, 0.5], k: 0 }, pick: "bits", expect: 1 },           // edge: one bit per token is perplexity 2
    { args: { p: PX, k: 2 }, pick: "ppl", expect: 3.4200, tol: 0.002 },   // the widget's default boundary: X still wins
    { args: { p: PY, k: 2 }, pick: "ppl", expect: 3.4200, tol: 0.002 },   // ... and ties Y exactly: the crossover
  ],
};
WIDGETS["fixture:lecture_12--cond-perplexity"] = (root) => {
  const presets = {
    stanford: { name: "Stanford, models X and Y (KP predict)", toks: ["Stanford", "was", "founded", "in", "1885"], X: PX, Y: PY },
    france: { name: "France, one model (KP check)", toks: ["The", "capital", "of", "France", "is", "Par", "is"], X: PF, Y: null },
  };
  const s = { pre: "stanford", k: 2, tok: 4, X: [...PX], Y: [...PY] };
  const pic = el("div"), read = el("div", { class: "readout" });
  const kWrap = el("div"), tokWrap = el("div");
  const build = () => {
    const P = presets[s.pre], n = P.toks.length;
    kWrap.replaceChildren(slider("prompt tokens k (conditioned on, not scored)", 0, n - 1, s.k, 1, v => { s.k = v; draw(); }, v => `${v} of ${n}`));
    const pxs = slider("p_X of this token", 0.01, 0.99, s.X[s.tok], 0.01, v => { s.X[s.tok] = v; draw(); });
    const pys = P.Y ? slider("p_Y of this token", 0.01, 0.99, s.Y[s.tok], 0.01, v => { if (s.Y) { s.Y[s.tok] = v; draw(); } }) : el("span");
    tokWrap.replaceChildren(select("edit token", P.toks.map((t, i) => [String(i), `${i + 1}: ${t}`]), String(s.tok), v => { s.tok = +v; build(); draw(); }), pxs, pys);
  };
  const draw = () => {
    const P = presets[s.pre], n = P.toks.length, models = [["X", s.X, C.a], ...(P.Y ? [["Y", s.Y, C.b]] : [])];
    const W = 640, x0 = 40, top = 34, H = 150, colW = (W - x0 - 20) / n, maxBits = 7, yb = v => top + H - H * Math.min(v, maxBits) / maxBits;
    let b = text(10, 16, "bits charged per token, −log2 p · grey: prompt (conditioned on, not scored) · coloured: scored", { fill: C.muted, size: 11 });
    for (let t = 0; t <= maxBits; t += 1) b += text(x0 - 6, yb(t) + 4, t, { fill: C.muted, size: 10, anchor: "end" }) + line(x0, yb(t), W - 20, yb(t), C.rule, 'opacity="0.5"');
    P.toks.forEach((tk, i) => {
      const cx = x0 + i * colW, bw = (colW - 14) / models.length;
      models.forEach(([nm, p, col], j) => {
        const bits = -Math.log2(p[i]), x = cx + 7 + j * bw;
        b += rect(x, yb(bits), bw - 3, top + H - yb(bits), i < s.k ? C.rule : col, i === s.tok ? `stroke="${C.ink}" stroke-width="1"` : "");
        b += text(x + (bw - 3) / 2, yb(bits) - 4, `${nm} ${fmt(bits, 2)}`, { size: 10, anchor: "middle", fill: i < s.k ? C.muted : C.ink });
      });
      b += text(cx + colW / 2, top + H + 16, esc(tk), { size: 12, anchor: "middle", weight: i === s.tok ? "bold" : "" });
    });
    if (s.k > 0) b += line(x0 + s.k * colW, top - 6, x0 + s.k * colW, top + H + 4, C.ink, 'stroke-dasharray="4 3"') + text(x0 + s.k * colW + 4, top + 4, "response starts", { size: 10, fill: C.muted });
    pic.innerHTML = svg(W, top + H + 26, b);
    const rows = models.map(([nm, p]) => {
      const f = condPpl({ p, k: 0 }), c = condPpl({ p, k: s.k });
      return { nm, f, c, txt: `${nm}: whole text ${fmt(f.bits, 3)} bits/token → ppl <b>${fmt(f.ppl, 3)}</b> · scored tokens only (${c.n}) ${fmt(c.bits, 3)} bits/token = ${fmt(c.nats, 3)} nats → ppl <b>${fmt(c.ppl, 3)}</b>` };
    });
    let verdict = "";
    if (rows.length === 2) {
      const win = (a, b2) => Math.abs(a - b2) < 1e-9 ? "tie" : a < b2 ? "X lower (better)" : "Y lower (better)";
      verdict = `<span class="big">whole text: ${win(rows[0].f.ppl, rows[1].f.ppl)} · scored from token ${s.k + 1}: ${win(rows[0].c.ppl, rows[1].c.ppl)}</span>`;
    }
    read.innerHTML = `${rows.map(r => r.txt).join("<br>")}<br>${verdict}
      perplexity = 2^(average bits per scored token); the k prompt tokens leave the average entirely<br>
      <span class="muted small">provenance: fixture:lecture_12--cond-perplexity · lecture_12.py:L62 ((1/p(D))^(1/|D|)), L88-L90 (the 'founded' example, conditional perplexity over the response) · video 12:29-12:41 ('charge you bits') · probabilities are the KP's own items (predict: X, Y; check: France), and the sliders let you change them.</span>`;
  };
  const go = (k) => { s.pre = k; const P = presets[k]; s.X = [...P.X]; s.Y = P.Y ? [...P.Y] : null; s.k = k === "stanford" ? 2 : 0; s.tok = P.toks.length - 1; build(); draw(); };
  root.append(el("div", { class: "widget" }, el("div", { class: "controls" },
    select("text", Object.entries(presets).map(([k, v]) => [k, v.name]), s.pre, go), kWrap, tokWrap, read), pic));
  build(); draw();
};

// ============================================== 2. Elo fit on a battle graph ==
// p(A wins against B) = 1 / (1 + 10^((ELO_B − ELO_A)/400)) (lecture_12.py:L172), fit "to maximize probability of
// pairwise comparisons" (L173). Six models A-F; the learner switches battle sets (edges) on and off and sets each one's
// observed win fraction. The fit is Newton's method on the log-likelihood, one connected component at a time, with the
// same number of battles on every edge (demo assumption). "Sparse sample ... as long as the graph is connected" is
// video 37:50-38:02: a component's level is not fixed by its own battles, so the fit pins each component's mean at
// 1000, and the learner can add a shift to every rating and an offset to every component that does not contain A.
// Presets (KP): arena-pairwise-elo predict (gap 400, all shifted by −100), its check (A +100 over B, B +200 over C),
// elo-needs-connected-comparisons check (two groups ABC / DEF) and transfer (a chain of checkpoints, then lose 3-4).
const NODES = ["A", "B", "C", "D", "E", "F"];
const EDGES = ["AB", "BC", "CD", "DE", "EF", "BE"];
const winFrac = (gap) => 1 / (1 + 10 ** (-gap / 400));
function eloFit({ w = {}, shift = 0, level = 0, q = "AB" }) {
  const c = Math.LN10 / 400, idx = Object.fromEntries(NODES.map((n, i) => [n, i]));
  const edges = Object.entries(w).filter(([, v]) => v !== null && v !== undefined).map(([e, v]) => [idx[e[0]], idx[e[1]], Math.min(0.999, Math.max(0.001, v))]);
  const comp = NODES.map((_, i) => i), find = i => comp[i] === i ? i : (comp[i] = find(comp[i]));
  for (const [i, j] of edges) comp[find(i)] = find(j);
  const roots = [...new Set(NODES.map((_, i) => find(i)))], r = NODES.map(() => 0);
  for (const root of roots) {
    const mem = NODES.map((_, i) => i).filter(i => find(i) === root);
    if (mem.length > 1) for (let it = 0; it < 60; it++) {
      const g = NODES.map(() => 0), H = NODES.map(() => NODES.map(() => 0));
      for (const [i, j, y] of edges) {
        if (find(i) !== root) continue;
        const p = 1 / (1 + 10 ** ((r[j] - r[i]) / 400)), h = c * c * p * (1 - p);
        g[i] += c * (y - p); g[j] -= c * (y - p); H[i][i] += h; H[j][j] += h; H[i][j] -= h; H[j][i] -= h;
      }
      const red = mem.slice(1), m = red.length, M = red.map((a, x) => [...red.map(b2 => H[a][b2]), g[a]]);
      for (let col = 0; col < m; col++) {                                  // Gaussian elimination, partial pivoting
        let piv = col; for (let row = col + 1; row < m; row++) if (Math.abs(M[row][col]) > Math.abs(M[piv][col])) piv = row;
        [M[col], M[piv]] = [M[piv], M[col]];
        for (let row = 0; row < m; row++) if (row !== col) { const f = M[row][col] / M[col][col]; for (let k = col; k <= m; k++) M[row][k] -= f * M[col][k]; }
      }
      let step = 0; red.forEach((a, x) => { const d = M[x][m] / M[x][x]; r[a] += d; step = Math.max(step, Math.abs(d)); });
      if (step < 1e-9) break;
    }
    const mean = mem.reduce((s2, i) => s2 + r[i], 0) / mem.length;
    const off = 1000 + shift + (root === find(0) ? 0 : level);
    for (const i of mem) r[i] += off - mean;
  }
  let ll = 0; for (const [i, j, y] of edges) { const p = 1 / (1 + 10 ** ((r[j] - r[i]) / 400)); ll += y * Math.log(p) + (1 - y) * Math.log(1 - p); }
  const a = idx[q[0]], b = idx[q[1]];
  const o = { ratings: r, comps: roots.length, identified: find(a) === find(b) ? 1 : 0, gap: r[a] - r[b], p: 1 / (1 + 10 ** ((r[b] - r[a]) / 400)), ll: edges.length ? ll / edges.length : 0, connected: roots.length === 1 ? 1 : 0 };
  NODES.forEach((n, i) => { o[`r${n}`] = r[i]; });
  return o;
}
const W100 = winFrac(100), W200 = winFrac(200);
const TWO_GROUPS = { AB: W100, BC: W200, DE: W100, EF: W200 };
const CHAIN = { AB: W100, BC: W100, CD: W100, DE: W100, EF: W100 };
MODELS["fixture:lecture_12--elo-graph"] = {
  fn: eloFit,
  cases: [
    { args: { w: { AB: 10 / 11 }, shift: -100, q: "AB" }, pick: "p", expect: 0.909, from: "lecture_12:arena-pairwise-elo:predict" },
    { args: { w: { AB: 10 / 11 }, shift: 0, q: "AB" }, pick: "gap", expect: 400 },                // 10/11 observed is a 400-point gap
    { args: { w: { AB: W100, BC: W200 }, q: "AC" }, pick: "p", expect: 0.85, from: "lecture_12:arena-pairwise-elo:check" },
    { args: { w: { AB: W100, BC: W200 }, q: "AC" }, pick: "gap", expect: 300 },                   // check's why: A is 300 above C
    { args: { w: TWO_GROUPS, q: "AD" }, pick: "identified", expect: 0, tol: 0 },                  // elo-needs-connected predict: no link
    { args: { w: TWO_GROUPS, level: 0, q: "AD" }, pick: "ll", expect: -0.6024, tol: 0.002 },      // ... the likelihood at offset 0
    { args: { w: TWO_GROUPS, level: -200, q: "AD" }, pick: "ll", expect: -0.6024, tol: 0.002 },   // ... is the same at offset −200
    { args: { w: TWO_GROUPS, level: -200, q: "AD" }, pick: "p", expect: 0.7597, tol: 0.002 },     // ... which prints the predict's lure 0.76
    { args: { w: { ...TWO_GROUPS, BE: 0.5 }, q: "AD" }, pick: "identified", expect: 1 },          // elo-needs-connected check: any cross pair
    { args: { w: { ...TWO_GROUPS, BE: 0.5 }, q: "AD" }, pick: "gap", expect: 0, tol: 0 },          // ... B-E tied: A = B+100 = E+100 = D
    { args: { w: CHAIN, q: "AF" }, pick: "identified", expect: 1 },                               // transfer: consecutive chain is connected
    { args: { w: CHAIN, q: "AF" }, pick: "gap", expect: 500 },                                    // ... five 100-point links
    { args: { w: { ...CHAIN, CD: null }, q: "AF" }, pick: "identified", expect: 0, tol: 0 },      // ... lose 3-vs-4 (C-D): split
    { args: { w: { ...CHAIN, CD: null }, q: "AF" }, pick: "comps", expect: 2 },
    { args: { w: { AB: 0.5, BC: 0.5 }, q: "AC" }, pick: "p", expect: 0.5 },                       // edge: all ties, one rating
  ],
};
WIDGETS["fixture:lecture_12--elo-graph"] = (root) => {
  const s = { on: { AB: true, BC: true, CD: false, DE: true, EF: true, BE: false }, w: { AB: W100, BC: W200, CD: 0.5, DE: W100, EF: W200, BE: 0.5 }, shift: 0, level: 0, q: "AD" };
  const pic = el("div"), read = el("div", { class: "readout" }), edgeBox = el("div"), shiftBox = el("div");
  const args = () => ({ w: Object.fromEntries(EDGES.filter(e => s.on[e]).map(e => [e, s.w[e]])), shift: s.shift, level: s.level, q: s.q });
  const buildEdges = () => edgeBox.replaceChildren(...EDGES.map(e => el("div", { style: "display:flex;gap:10px;align-items:center;flex-wrap:wrap" },
    check(`${e[0]}–${e[1]} battles`, s.on[e], v => { s.on[e] = v; draw(); }),
    slider(`${e[0]} wins`, 0.02, 0.98, s.w[e], 0.01, v => { s.w[e] = v; draw(); }, v => `${Math.round(v * 100)}% (gap ${fmt(400 * Math.log10(v / (1 - v)), 0)})`))));
  const buildShift = () => shiftBox.replaceChildren(
    slider("shift every rating", -300, 300, s.shift, 10, v => { s.shift = v; draw(); }),
    slider("offset of every component without A", -400, 400, s.level, 10, v => { s.level = v; draw(); }));
  const draw = () => {
    const m = eloFit(args()), r = m.ratings, W = 640, H = 230, x0 = 60, top = 30, bot = H - 30;
    const lo = Math.min(...r) - 60, hi = Math.max(...r) + 60, y = v => bot - (v - lo) / (hi - lo) * (bot - top), xs = i => x0 + i * 100;
    const idx = Object.fromEntries(NODES.map((n, i) => [n, i]));
    const sameAsA = NODES.map(n => eloFit({ ...args(), q: "A" + n }).identified);
    let b = text(10, 16, "height = fitted Elo rating · solid: battles on · dashed: no battles · orange: not linked to A (its level is free)", { fill: C.muted, size: 11 });
    for (let t = Math.ceil(lo / 100) * 100; t <= hi; t += 100) b += line(x0 - 30, y(t), W - 20, y(t), C.rule, 'opacity="0.5"') + text(x0 - 34, y(t) + 4, t, { fill: C.muted, size: 10, anchor: "end" });
    for (const e of EDGES) {
      const i = idx[e[0]], j = idx[e[1]], on = s.on[e];
      const mid = e === "BE" ? `Q ${(xs(i) + xs(j)) / 2} ${Math.min(y(r[i]), y(r[j])) - 50}` : "L";
      b += `<path d="M ${xs(i)} ${y(r[i]).toFixed(1)} ${mid} ${xs(j)} ${y(r[j]).toFixed(1)}" style="fill:none;stroke:${on ? C.ink : C.rule};stroke-width:${on ? 2 : 1.2}" ${on ? "" : 'stroke-dasharray="4 4"'}/>`;
    }
    NODES.forEach((n, i) => {
      const col = sameAsA[i] ? C.a : C.b, hl = s.q.includes(n);
      b += `<circle cx="${xs(i)}" cy="${y(r[i]).toFixed(1)}" r="${hl ? 13 : 11}" style="fill:${col};stroke:${hl ? C.ink : "none"};stroke-width:2"/>`;
      b += text(xs(i), y(r[i]) + 4, n, { fill: "#fff", size: 12, anchor: "middle", weight: "bold" });
      b += text(xs(i), y(r[i]) + 26, fmt(r[i], 0), { size: 10, anchor: "middle", fill: C.muted });
    });
    pic.innerHTML = svg(W, H, b);
    const [qa, qb] = s.q.split("");
    const res = m.identified
      ? `<span class="big">p(${qa} beats ${qb}) = ${fmt(m.p, 3)} (gap ${fmt(m.gap, 0)})</span>${qa} and ${qb} are linked through battles, so the fitted gap is fixed by the data`
      : `<span class="big">p(${qa} beats ${qb}): not identified</span>the formula prints ${fmt(m.p, 3)} at this offset, but the offset slider changes it while the log-likelihood stays ${fmt(m.ll, 4)}: no battle connects ${qa} to ${qb}`;
    read.innerHTML = `${m.comps === 1 ? "comparison graph connected: one scale" : `comparison graph has ${m.comps} components: ${m.comps} separate scales`} · log-likelihood ${fmt(m.ll, 4)} nats per battle (equal battles per edge)<br>${res}<br>
      shifting every rating changes no win probability and no likelihood: only differences enter p<br>
      <span class="muted small">provenance: fixture:lecture_12--elo-graph · lecture_12.py:L172 (the formula), L173 (fit to maximize the probability of the comparisons) · video 37:42-38:02 (sparse battles suffice 'as long as the graph is connected') · win fractions default to the KP check's gaps (100, 200); each component's mean is pinned at 1000, an arbitrary choice, which is the point.</span>`;
  };
  const presets = [
    ["two separate pools", () => { Object.assign(s.on, { AB: 1, BC: 1, CD: 0, DE: 1, EF: 1, BE: 0 }); Object.assign(s.w, { AB: W100, BC: W200, DE: W100, EF: W200 }); s.q = "AD"; }],
    ["chain of checkpoints", () => { Object.assign(s.on, { AB: 1, BC: 1, CD: 1, DE: 1, EF: 1, BE: 0 }); for (const e of ["AB", "BC", "CD", "DE", "EF"]) s.w[e] = W100; s.q = "AF"; }],
    ["400-point gap", () => { Object.assign(s.on, { AB: 1, BC: 0, CD: 0, DE: 0, EF: 0, BE: 0 }); s.w.AB = 10 / 11; s.q = "AB"; }],
  ];
  const pairs = []; NODES.forEach((a, i) => NODES.slice(i + 1).forEach(b2 => pairs.push([a + b2, `${a} vs ${b2}`])));
  const qSel = el("div");
  const buildQ = () => qSel.replaceChildren(select("win probability for", pairs, s.q, v => { s.q = v; draw(); }));
  root.append(el("div", { class: "widget" }, el("div", { class: "controls" }, edgeBox, shiftBox, qSel,
    el("div", { style: "margin-top:6px" }, ...presets.map(([n, f]) => el("button", { onclick: () => { f(); s.level = 0; s.shift = 0; buildEdges(); buildShift(); buildQ(); draw(); } }, n))), read), pic));
  buildEdges(); buildShift(); buildQ(); draw();
};

// ======================================= 3. two contracts: log-prob vs text ==
// lecture_12.py:L92-L101: cloze and multiple-choice completion are "perplexity in disguise", scored by the probabilities the
// submitted LM assigns (log_prob = LM(test_data), L99), so the leaderboard must trust that they are valid, summing to 1
// (L100); downstream tasks only read response = LM(prompt) and grade it (L101). Video 15:51-16:14: an LM that returns
// log-prob 0 everywhere "would get very good perplexity". Video 31:27-31:43: the model either samples a letter or writes a
// chain of thought, and the way the answer is extracted "does matter".
// Panel 1: one 4-option item. Log-prob scoring takes the argmax of the model's option probabilities. Generation samples the
// letter at temperature T (T = 0 greedy) and fails extraction with probability e. The option probabilities, T and e are
// demo controls (no lecture values). Panel 2: a submission that returns probability q for every token (KP predict, q = 0.9).
function mcScore({ p, key = 0, T = 1, e = 0, q = 0.9 }) {
  const z = p.reduce((s2, x) => s2 + x, 0), pn = p.map(x => x / z);
  const arg = pn.indexOf(Math.max(...pn)), lp = arg === key ? 1 : 0;
  const tw = T > 0 ? pn.map(x => x ** (1 / T)) : pn.map((x, i) => (i === arg ? 1 : 0)), tz = tw.reduce((s2, x) => s2 + x, 0), samp = tw.map(x => x / tz);
  return { pn, samp, lp, pKey: pn[key], gen: (1 - e) * samp[key], pplConst: 1 / q, sum2: 2 * q, maxValid: 0.5 };
}
const PDEMO = [0.4, 0.35, 0.15, 0.1];
MODELS["fixture:lecture_12--mc-scoring"] = {
  fn: mcScore,
  cases: [
    { args: { p: PDEMO, q: 0.9 }, pick: "pplConst", expect: 1.11, tol: 0.005 },          // logprob-vs-generation predict: 1/0.9 ≈ 1.11
    { args: { p: PDEMO, q: 0.9 }, pick: "sum2", expect: 1.8 },                            // ... already 1.8 over just two tokens: invalid
    { args: { p: PDEMO, key: 0, T: 1 }, pick: "lp", expect: 1 },                          // transfer: log-prob scoring takes the argmax
    { args: { p: PDEMO, key: 0, T: 1 }, pick: "gen", expect: 0.4 },                       // ... sampling the letter gets it 40% of the time
    { args: { p: PDEMO, key: 0, T: 0 }, pick: "gen", expect: 1 },                         // edge: greedy decoding equals the argmax
    { args: { p: PDEMO, key: 0, T: 2 }, pick: "gen", expect: 0.3282, tol: 0.002 },       // hotter sampling moves further from it
    { args: { p: PDEMO, key: 0, T: 1, e: 0.1 }, pick: "gen", expect: 0.36 },              // extraction failures cost generation only
    { args: { p: PDEMO, key: 1, T: 1 }, pick: "lp", expect: 0, tol: 0 },                  // key B: argmax wrong, log-prob scores 0 ...
    { args: { p: PDEMO, key: 1, T: 1 }, pick: "gen", expect: 0.35 },                      // ... while sampling still earns 0.35
    { args: { p: [2, 1, 1, 0] }, pick: "pKey", expect: 0.5 },                             // weights are renormalised to sum to 1
  ],
};
WIDGETS["fixture:lecture_12--mc-scoring"] = (root) => {
  const s = { p: [...PDEMO], key: 0, T: 1, e: 0, q: 0.5 };
  const pic = el("div"), read = el("div", { class: "readout" }), read2 = el("div", { class: "readout" }), L = ["A", "B", "C", "D"];
  const draw = () => {
    const m = mcScore(s), W = 640, H = 170, x0 = 40, top = 28, bh = 110, y = v => top + bh - bh * v;
    let b = text(10, 16, "dark: the model's probability (log-prob scoring reads it) · light: sampling distribution at T (generation draws it)", { fill: C.muted, size: 11 });
    for (const t of [0, 0.5, 1]) b += line(x0, y(t), W - 20, y(t), C.rule, 'opacity="0.5"') + text(x0 - 6, y(t) + 4, t, { fill: C.muted, size: 10, anchor: "end" });
    L.forEach((n, i) => {
      const cx = x0 + 20 + i * 145;
      b += rect(cx, y(m.pn[i]), 50, bh * m.pn[i], C.a) + rect(cx + 54, y(m.samp[i]), 50, bh * m.samp[i], C.a, 'opacity="0.35"');
      b += text(cx + 25, y(m.pn[i]) - 4, fmt(m.pn[i], 3), { size: 10, anchor: "middle" }) + text(cx + 79, y(m.samp[i]) - 4, fmt(m.samp[i], 3), { size: 10, anchor: "middle" });
      b += text(cx + 52, top + bh + 18, i === s.key ? `${n} (key)` : n, { size: 12, anchor: "middle", weight: i === s.key ? "bold" : "", fill: i === s.key ? C.ok : C.ink });
    });
    pic.innerHTML = svg(W, H, b);
    const lpTxt = m.lp ? "argmax is the key: scores 1" : "argmax is not the key: scores 0";
    read.innerHTML = `log-prob scoring (needs the model's probabilities, L99-L100): ${lpTxt} · deterministic<br>
      generation (needs only text, L101): P(sampled letter is the key) = ${fmt(m.samp[s.key], 3)} × (1 − extraction failure ${fmt(s.e, 2)}) = <b>${fmt(m.gen, 3)}</b> expected score${s.T === 0 ? " (greedy)" : ""}<br>
      <span class="big">${Math.abs(m.gen - m.lp) < 1e-9 ? "the two modes agree on this item" : `the two modes disagree by ${fmt(Math.abs(m.lp - m.gen), 3)} on the same model`}</span>
      <span class="muted small">option probabilities, T and the extraction failure rate are demo controls; the lecture gives no values for them.</span>`;
    read2.innerHTML = `a submission that returns probability ${fmt(s.q, 3)} for every token: reported perplexity 1/${fmt(s.q, 3)} = <b>${fmt(m.pplConst, 3)}</b><br>
      validity check: over just two candidate tokens its probabilities sum to ${fmt(m.sum2, 3)}; ${m.sum2 > 1 + 1e-9 ? "<b>more than 1: not a distribution</b>, and over a whole vocabulary of |V| tokens the sum is q·|V|" : "at most 1 over two tokens, but q·|V| over a whole vocabulary of |V| tokens exceeds 1 as soon as |V| > 1/q"}<br>
      a generation benchmark never sees this number: it grades response = LM(prompt) (L101)<br>
      <span class="muted small">provenance: fixture:lecture_12--mc-scoring · lecture_12.py:L92-L101 · video 15:51-16:22 (the log-prob 0 submission), 31:27-31:43 (sample a letter or chain of thought, then extract) · q = 0.9 is the KP predict's submission; the default here is 0.5.</span>`;
  };
  const opt = L.map((n, i) => slider(`model weight on ${n}`, 0.01, 1, s.p[i], 0.01, v => { s.p[i] = v; draw(); }));
  root.append(el("div", { class: "widget" }, el("div", { class: "controls" },
    ...opt, select("correct option", L.map((n, i) => [String(i), n]), "0", v => { s.key = +v; draw(); }),
    slider("sampling temperature T (0 = greedy)", 0, 2, s.T, 0.05, v => { s.T = v; draw(); }),
    slider("answer-extraction failure rate", 0, 0.5, s.e, 0.01, v => { s.e = v; draw(); }), read), pic,
    el("div", { class: "controls", style: "margin-top:10px" }, slider("constant per-token probability q", 0.05, 0.99, s.q, 0.01, v => { s.q = v; draw(); }), read2)));
  draw();
};

// ============================================= 4. judge length bias and LC ==
// AlpacaEval: win rate against a GPT-4-preview baseline as judged by GPT-4 preview (lecture_12.py:L186); "LLM judges favor
// longer responses, resulted in leaderboard gaming" (L187); "Alpaca Eval 2.0 used regression to debias the metric" (L188);
// spoken at video 39:21-39:50. Model: the judge's log-odds that a model's answer beats the baseline's is a quality term q
// plus a length term φ·tanh(Δ), with Δ the length difference in standard deviations. This follows the shape of the
// length-controlled GLM of Dubois et al. 2024 (sources.json alpacaeval_lc_2024) as the author recalls it, without its
// per-instruction term; the repo does not hold the PDF, so treat the form as the author's. The length-controlled (LC) win
// rate is the same fit read at Δ = 0, i.e. σ(q), assuming the regression recovered φ. q_A, q_B and φ are demo controls.
const sig = (x) => 1 / (1 + Math.exp(-x));
function lenBias({ qA, qB, phi, d }) {
  const rawA = sig(qA), rawB = sig(qB + phi * Math.tanh(d)), lcA = sig(qA), lcB = sig(qB), t = phi > 0 ? (qA - qB) / phi : Infinity;
  return { rawA, rawB, lcA, lcB, rawGain: rawB - lcB, cross: Math.abs(t) < 1 ? Math.atanh(t) : NaN, rawBahead: rawB > rawA ? 1 : 0, lcBahead: lcB > lcA ? 1 : 0 };
}
MODELS["fixture:lecture_12--length-bias"] = {
  fn: lenBias,
  cases: [
    { args: { qA: 0.4, qB: 0, phi: 1, d: 0 }, pick: "rawB", expect: 0.5 },                 // no padding: raw = LC
    { args: { qA: 0.4, qB: 0, phi: 1, d: 2 }, pick: "rawB", expect: 0.7240, tol: 0.002 },  // llm-judge-length-bias predict: original goes up
    { args: { qA: 0.4, qB: 0, phi: 1, d: 2 }, pick: "lcB", expect: 0.5 },                  // ... LC win rate does not move
    { args: { qA: 0.4, qB: 0, phi: 1, d: 2 }, pick: "rawBahead", expect: 1 },              // gaming: padded B overtakes A on the raw board
    { args: { qA: 0.4, qB: 0, phi: 1, d: 2 }, pick: "lcBahead", expect: 0, tol: 0 },       // ... but not on the LC board
    { args: { qA: 0.4, qB: 0, phi: 1, d: 0 }, pick: "cross", expect: 0.4236, tol: 0.002 }, // the padding where B overtakes A: atanh(0.4)
    { args: { qA: 0.4, qB: 0, phi: 0.3, d: 3 }, pick: "rawBahead", expect: 0, tol: 0 },    // edge: weak bias (φ < q_A − q_B) cannot flip it
    { args: { qA: 0.4, qB: 0, phi: 0, d: 3 }, pick: "rawGain", expect: 0, tol: 0 },        // unbiased judge: padding buys nothing
    { args: { qA: 0.4, qB: 0, phi: 1, d: -1 }, pick: "rawB", expect: 0.3184, tol: 0.002 }, // shorter than the baseline costs win rate
  ],
};
WIDGETS["fixture:lecture_12--length-bias"] = (root) => {
  const s = { qA: 0.4, qB: 0, phi: 1, d: 0 };
  const pic = el("div"), read = el("div", { class: "readout" });
  const draw = () => {
    const m = lenBias(s), W = 640, H = 230, x0 = 50, x1 = W - 20, top = 26, bot = H - 34, dMin = -2, dMax = 3;
    const x = d => x0 + (d - dMin) / (dMax - dMin) * (x1 - x0), y = v => bot - v * (bot - top);
    let b = text(10, 16, "win rate against the baseline, as the judge scores it, vs how much longer B's answers are", { fill: C.muted, size: 11 });
    for (const t of [0, 0.25, 0.5, 0.75, 1]) b += line(x0, y(t), x1, y(t), C.rule, 'opacity="0.5"') + text(x0 - 6, y(t) + 4, `${t * 100}%`, { fill: C.muted, size: 10, anchor: "end" });
    for (let d = dMin; d <= dMax; d++) b += text(x(d), bot + 14, d > 0 ? `+${d}` : d, { fill: C.muted, size: 10, anchor: "middle" });
    b += text((x0 + x1) / 2, bot + 28, "B's length difference from the baseline (standard deviations)", { fill: C.muted, size: 10, anchor: "middle" });
    let pth = ""; for (let i = 0; i <= 100; i++) { const d = dMin + (dMax - dMin) * i / 100; pth += `${i ? "L" : "M"} ${x(d).toFixed(1)} ${y(sig(s.qB + s.phi * Math.tanh(d))).toFixed(1)} `; }
    b += `<path d="${pth}" style="fill:none;stroke:${C.b};stroke-width:2.5"/>`;
    b += line(x0, y(m.lcB), x1, y(m.lcB), C.b, 'stroke-dasharray="6 4"') + line(x0, y(m.rawA), x1, y(m.rawA), C.a);
    b += text(x1 - 4, y(m.rawA) - 5, `A (same length as baseline): ${fmt(m.rawA * 100, 1)}%, raw = LC`, { fill: C.a, size: 10, anchor: "end" });
    b += text(x0 + 4, y(m.lcB) + 13, `B length-controlled: ${fmt(m.lcB * 100, 1)}%`, { fill: C.b, size: 10 });
    if (Number.isFinite(m.cross) && m.cross <= dMax && m.cross >= dMin) b += line(x(m.cross), top, x(m.cross), bot, C.muted, 'stroke-dasharray="2 3"') + text(x(m.cross) + 4, top + 10, "raw: B overtakes A", { fill: C.muted, size: 10 });
    b += `<circle cx="${x(s.d).toFixed(1)}" cy="${y(m.rawB).toFixed(1)}" r="6" style="fill:${C.b};stroke:${C.ink};stroke-width:1.5"/>`;
    pic.innerHTML = svg(W, H, b);
    read.innerHTML = `B raw win rate σ(q_B + φ·tanh Δ) = <b>${fmt(m.rawB * 100, 1)}%</b> · B length-controlled σ(q_B) = <b>${fmt(m.lcB * 100, 1)}%</b> · padding bought ${fmt(m.rawGain * 100, 1)} points<br>
      <span class="big">raw leaderboard: ${m.rawBahead ? "B ahead of A (gamed by length)" : "A ahead of B"} · length-controlled: ${m.lcBahead ? "B ahead of A" : "A ahead of B"}</span>
      ${s.phi === 0 ? "unbiased judge (φ = 0): length moves nothing, raw and LC coincide" : Number.isFinite(m.cross) ? `with this bias, padding beyond Δ = ${fmt(m.cross, 2)} lets B pass A on the raw board` : "this bias is too weak for any padding to lift B past A"}<br>
      <span class="muted small">provenance: fixture:lecture_12--length-bias · lecture_12.py:L186-L188 · video 39:21-39:50 · log-odds q + φ·tanh(Δ) after the length-controlled GLM of Dubois et al. 2024 as the author recalls it (no per-instruction term); LC reads the fit at Δ = 0 and assumes the regression found φ. q_A = 0.4, q_B = 0 and φ = 1 are demo values, not AlpacaEval's.</span>`;
  };
  root.append(el("div", { class: "widget" }, el("div", { class: "controls" },
    slider("B's padding Δ (std of length difference)", -2, 3, s.d, 0.05, v => { s.d = v; draw(); }),
    slider("judge length coefficient φ (0 = unbiased)", 0, 2, s.phi, 0.05, v => { s.phi = v; draw(); }),
    slider("A's quality term q_A (logits)", -1, 1.5, s.qA, 0.05, v => { s.qA = v; draw(); }),
    slider("B's quality term q_B (logits)", -1, 1.5, s.qB, 0.05, v => { s.qB = v; draw(); }), read), pic));
  draw();
};

// ================================== 5. order test for train-test overlap ==
// Route 1: "try to infer train-test overlap from model ... Exploit exchangeability of data points" (lecture_12.py:L346-L347,
// Oren et al. 2023, figure images/contamination-exchangeability.png at L348; spoken at video 1:09:56-1:10:26). The
// figure's mechanism: a model that saw the benchmark file gives an example high log-probability when it follows its
// published predecessor, and low log-probability otherwise. Toy model (the author's, after that figure): log p(order) =
// base + b per published transition (x_i then x_{i+1}) that the model memorized and that the order contains. With
// exchangeable items and no contamination every order scores the same. p-value = share of all n! orders scoring at least
// the published one (exact enumeration, n ≤ 8). The default n = 4 and the item texts are the figure's four questions.
// The toy has no noise, so b does not change the p-value; the real test compares against noisy shuffled scores.
function perms(n) {
  const out = [], a = [...Array(n).keys()], c = Array(n).fill(0); out.push([...a]);
  for (let i = 1; i < n;) { if (c[i] < i) { const j = i % 2 ? c[i] : 0; [a[j], a[i]] = [a[i], a[j]]; out.push([...a]); c[i]++; i = 1; } else { c[i] = 0; i++; } }
  return out;
}
function orderTest({ n = 4, k = 0, dependent = false }) {
  const sig2 = dependent ? n - 1 : Math.min(k, n - 1);           // transitions the model rewards: memorized, or built into the items
  const hits = (pm) => { let h = 0; for (let t = 0; t + 1 < n; t++) if (pm[t] + 1 === pm[t + 1] && pm[t] < sig2) h++; return h; };
  const all = perms(n), counts = Array(n).fill(0); let ge = 0, sum = 0;
  for (const pm of all) { const h = hits(pm); counts[h]++; sum += h; if (h >= sig2) ge++; }
  return { orders: all.length, canonical: sig2, counts, pValue: ge / all.length, meanShuffled: sum / all.length, flagged: ge / all.length < 0.05 ? 1 : 0 };
}
MODELS["fixture:lecture_12--order-test"] = {
  fn: orderTest,
  cases: [
    { args: { n: 4, k: 0 }, pick: "pValue", expect: 1 },                          // exchangeable and clean: every order ties
    { args: { n: 4, k: 3 }, pick: "pValue", expect: 1 / 24 },                     // train-test-overlap predict: the published order wins
    { args: { n: 4, k: 3 }, pick: "flagged", expect: 1 },
    { args: { n: 4, k: 3 }, pick: "meanShuffled", expect: 0.75 },                 // a random shuffle keeps k/n memorized transitions
    { args: { n: 4, k: 1 }, pick: "pValue", expect: 0.25 },                       // partial memorization: (n−k)!/n!
    { args: { n: 4, k: 2 }, pick: "pValue", expect: 1 / 12 },
    { args: { n: 6, k: 5 }, pick: "pValue", expect: 1 / 720 },
    { args: { n: 8, k: 7 }, pick: "pValue", expect: 1 / 40320 },
    { args: { n: 4, k: 0, dependent: true }, pick: "pValue", expect: 1 / 24 },    // self-explain: non-exchangeable items, a clean model is flagged too
  ],
};
WIDGETS["fixture:lecture_12--order-test"] = (root) => {
  const Q = ["Does a frog jump out of boiling water?", "Is it possible to create mass from energy?", "Is there a movie with 0 on rotten tomatoes?", "Is the jaguar S type rear wheel drive?",
    "item 5", "item 6", "item 7", "item 8"];
  const s = { n: 4, k: 0, dep: false, shuf: [1, 0, 3, 2] };
  const pic = el("div"), read = el("div", { class: "readout" }), kBox = el("div");
  const reshuffle = () => { const a = [...Array(s.n).keys()]; for (let i = a.length - 1; i > 0; i--) { const j = Math.floor(Math.random() * (i + 1)); [a[i], a[j]] = [a[j], a[i]]; } s.shuf = a; };
  const buildK = () => kBox.replaceChildren(slider("published transitions the model memorized", 0, s.n - 1, Math.min(s.k, s.n - 1), 1, v => { s.k = v; draw(); }, v => `${v} of ${s.n - 1}`));
  const draw = () => {
    s.k = Math.min(s.k, s.n - 1);
    const m = orderTest({ n: s.n, k: s.k, dependent: s.dep }), sigT = m.canonical, W = 640, H = 120 + s.n * 22;
    const mark = (ord, t) => t > 0 && ord[t - 1] + 1 === ord[t] && ord[t - 1] < sigT;
    let b = text(10, 16, "published order (left) vs a shuffle (right) · ✓ follows its published predecessor and the model rewards that: high score · · ordinary score", { fill: C.muted, size: 11 });
    [[[...Array(s.n).keys()], 10, "published order"], [s.shuf, 330, "shuffled order"]].forEach(([ord, x, nm]) => {
      b += text(x, 36, nm, { weight: "bold", size: 12 });
      ord.forEach((it, t) => { const ok = mark(ord, t); b += rect(x, 44 + t * 22, 270, 18, "#f1ede3") + text(x + 4, 57 + t * 22, esc(Q[it].length > 45 ? Q[it].slice(0, 44) + "…" : Q[it]), { size: 11 }) + text(x + 286, 58 + t * 22, t === 0 ? "" : ok ? "✓" : "·", { fill: ok ? C.ok : C.muted, size: 13, weight: "bold" }); });
    });
    const hy = 58 + s.n * 22, hh = 40, mx = Math.max(...m.counts), bw = 560 / s.n;
    b += text(10, hy, `all ${m.orders} orders, by how many rewarded transitions they keep (the published order keeps ${sigT})`, { fill: C.muted, size: 11 });
    m.counts.forEach((c, h) => { const hgt = hh * c / mx; b += rect(40 + h * bw, hy + 8 + hh - hgt, bw - 8, hgt, h >= sigT ? C.b : C.a, h >= sigT ? "" : 'opacity="0.5"') + text(40 + h * bw + (bw - 8) / 2, hy + hh + 22, `${h}: ${c}`, { size: 10, anchor: "middle", fill: C.muted }); });
    pic.innerHTML = svg(W, H, b);
    const verdict = sigT === 0 ? "every order scores the same: the published order is not special, p = 1, no evidence of overlap"
      : s.dep ? `flagged (p = ${fmt(m.pValue, 4)}) although the model never saw the file: the items depend on each other, so they are not exchangeable and the test's null is false`
        : `the published order beats ${fmt((1 - m.pValue) * 100, 2)}% of orders: p = ${fmt(m.pValue, 4)}${m.flagged ? ", evidence that this file, in this order, was in the training data" : ", too weak to call at 5%"}`;
    read.innerHTML = `published order: ${sigT} rewarded transitions · a random shuffle keeps ${fmt(m.meanShuffled, 3)} on average<br><span class="big">p-value = ${fmt(m.pValue, 5)}</span>${verdict}<br>
      <span class="muted small">provenance: fixture:lecture_12--order-test · lecture_12.py:L346-L348 (Route 1, exchangeability; figure contamination-exchangeability.png, whose four questions are used here) · video 1:09:56-1:10:26 · the per-transition reward is the author's toy reading of the figure; it has no noise, so any reward size gives the same p-value.</span>`;
  };
  root.append(el("div", { class: "widget" }, el("div", { class: "controls" },
    slider("benchmark items n", 3, 8, s.n, 1, v => { s.n = v; reshuffle(); buildK(); draw(); }), kBox,
    check("items depend on their predecessor (not exchangeable), model never saw the file", s.dep, v => { s.dep = v; if (v) s.k = 0; buildK(); draw(); }),
    el("div", { style: "margin-top:6px" }, el("button", { onclick: () => { reshuffle(); draw(); } }, "draw another shuffle")), read), pic));
  buildK(); draw();
};
