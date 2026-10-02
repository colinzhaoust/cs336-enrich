// Widgets for thread lecture_14 (CS336 L14, data II: filtering, dedup, mixing). Register as "fixture:<id>" -> (root, notice) => void.
// Each widget has a pure model in MODELS (no DOM) that tools/check_widgets.mjs tests against the KPs' stored answers.
// Sources: official/lectures/lecture_14.py (cited as L<line>), lectures/lecture_14/transcript.json (cited as video M:SS),
//          and the KP filtering notes, which record the thread's own evaluations of the lecture's formulas (marked "KP note").
// No constant here is new: every number is the lecture's, a prompt's, or computed by the lecture's own formula.
import { el, fmt, slider } from "../widgets_lib.js";

export const MODELS = {};
export const WIDGETS = {};

const C = { ink: "var(--ink)", muted: "var(--muted)", rule: "var(--rule)", a: "var(--accent)", b: "var(--accent2)", ok: "var(--ok)", hi: "var(--hilite)" };
const rect = (x, y, w, h, fill, extra = "") => `<rect x="${x.toFixed(1)}" y="${y.toFixed(1)}" width="${Math.max(0, w).toFixed(1)}" height="${h}" style="fill:${fill}" ${extra}/>`;
const text = (x, y, s, opts = {}) => `<text x="${x.toFixed(1)}" y="${y.toFixed(1)}" style="fill:${opts.fill || C.ink};font:${opts.weight || ""} ${opts.size || 12}px var(--sans)${opts.halo ? ";paint-order:stroke;stroke:#fff;stroke-width:4px" : ""}" text-anchor="${opts.anchor || "start"}">${s}</text>`;
const line = (x1, y1, x2, y2, stroke, extra = "") => `<line x1="${x1.toFixed(1)}" y1="${y1.toFixed(1)}" x2="${x2.toFixed(1)}" y2="${y2.toFixed(1)}" style="stroke:${stroke};stroke-width:1.5" ${extra}/>`;
const path = (pts, stroke, extra = "") => `<path d="${pts.map((p, i) => `${i ? "L" : "M"}${p[0].toFixed(1)},${p[1].toFixed(1)}`).join("")}" style="fill:none;stroke:${stroke};stroke-width:2" ${extra}/>`;
const svg = (w, h, body) => `<svg viewBox="0 0 ${w} ${h}" width="100%" style="max-width:${w}px;border:1px solid var(--rule);border-radius:6px;background:#fff" role="img">${body}</svg>`;
const esc = (s) => String(s).replace(/[&<>"]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]);
const button = (label, onclick) => el("button", { onclick }, label);
const setSlider = (lab, v) => { const inp = lab.querySelector("input"); inp.value = v; inp.dispatchEvent(new Event("input")); };

// MurmurHash3 x86_32, the function behind Python's mmh3.hash(x, seed) (L188, L243): UTF-8 bytes in, signed 32-bit out.
// Checked against the reference vectors ("", 1) = 0x514e28b7, ("Hello, world!", 0x9747b28c) = 0x24884cba, and mmh3's README
// mmh3.hash("foo") = -156908512. With it, minhash over seeds 0-99 on the lecture's A, B gives 60 matches, inside L263's assert.
const ENC = new TextEncoder();
export function mmh3(str, seed = 0) {
  const d = ENC.encode(String(str)), n = d.length, nb = n >> 2, c1 = 0xcc9e2d51, c2 = 0x1b873593;
  let h = seed >>> 0, k;
  for (let i = 0; i < nb; i++) {
    k = d[4 * i] | (d[4 * i + 1] << 8) | (d[4 * i + 2] << 16) | (d[4 * i + 3] << 24);
    k = Math.imul(k, c1); k = (k << 15) | (k >>> 17); k = Math.imul(k, c2);
    h ^= k; h = (h << 13) | (h >>> 19); h = (Math.imul(h, 5) + 0xe6546b64) | 0;
  }
  k = 0; const t = nb * 4;
  switch (n & 3) { case 3: k ^= d[t + 2] << 16; // falls through
    case 2: k ^= d[t + 1] << 8; // falls through
    case 1: k ^= d[t]; k = Math.imul(k, c1); k = (k << 15) | (k >>> 17); k = Math.imul(k, c2); h ^= k; }
  h ^= n; h ^= h >>> 16; h = Math.imul(h, 0x85ebca6b); h ^= h >>> 13; h = Math.imul(h, 0xc2b2ae35); h ^= h >>> 16;
  return h | 0;
}
const minhash = (S, seed) => { let m = Infinity; for (const x of S) { const v = mmh3(x, seed); if (v < m) m = v; } return m; };   // L242-L243

// ===================================================== 1. MinHash permutation ==
// L222-L230: Jaccard = |A ∩ B| / |A ∪ B|; A = {"1","2","3","4"}, B = {"1","2","3","5"} → 0.6.
// L237: Pr[h(A) = h(B)] = Jaccard(A, B). L242-L243: minhash(S, seed) = min(mmh3.hash(x, seed) for x in S).
// L253-L257: the seeded hash induces a permutation; the minima agree iff the first item of A ∪ B is in A ∩ B.
// L260-L263: 100 seeds, estimate within 0.01 of 0.6. Video 34:38-36:08 (permutation argument), 36:30-37:16 (100 seeds, "you get 0.6").
// Items are named "1", "2", ... with the shared ones first, so (shared 3, A-only 1, B-only 1) is exactly the lecture's A and B.
function setsOf({ shared, onlyA, onlyB }) {
  const A = [], B = []; let i = 1;
  for (let j = 0; j < shared; j++, i++) { A.push(String(i)); B.push(String(i)); }
  for (let j = 0; j < onlyA; j++, i++) A.push(String(i));
  for (let j = 0; j < onlyB; j++, i++) B.push(String(i));
  return { A, B, U: Array.from({ length: i - 1 }, (_, j) => String(j + 1)) };
}
function minhashModel({ shared, onlyA, onlyB, seeds = 100, seed0 = 0, simulate = true }) {
  const inter = shared, union = shared + onlyA + onlyB, jaccard = union ? inter / union : NaN;
  const o = { inter, union, sizeA: shared + onlyA, sizeB: shared + onlyB, jaccard, expected: seeds * jaccard };
  if (simulate && o.sizeA && o.sizeB) {
    const { A, B } = setsOf({ shared, onlyA, onlyB }); let m = 0;
    o.hits = [];
    for (let s = seed0; s < seed0 + seeds; s++) { const hit = minhash(A, s) === minhash(B, s); o.hits.push(hit); if (hit) m++; }
    o.matches = m; o.estimate = m / seeds;
  }
  return o;
}
MODELS["fixture:lecture_14--minhash-permutation"] = {
  fn: minhashModel,
  cases: [
    { args: { shared: 3, onlyA: 1, onlyB: 1 }, pick: "jaccard", expect: 0.6 },                         // L230 jaccard
    { args: { shared: 3, onlyA: 1, onlyB: 1, seeds: 100 }, pick: "estimate", expect: 0.6, tol: 0.0167 },  // L263 assert: |estimate − 0.6| < 0.01
    { args: { shared: 3, onlyA: 3, onlyB: 3 }, pick: "jaccard", expect: 1 / 3 },                       // minhash-collision-probability predict (choice 1/3)
    { args: { shared: 5, onlyA: 4, onlyB: 2, seeds: 200, simulate: false }, pick: "expected", expect: 90.9, tol: 0.01, from: "lecture_14:minhash-collision-probability:check" },
    { args: { shared: 4, onlyA: 4, onlyB: 0 }, pick: "jaccard", expect: 0.5 },                         // jaccard-similarity predict (choice: below, 0.5)
    { args: { shared: 900, onlyA: 100, onlyB: 300, simulate: false }, pick: "jaccard", expect: 0.692, tol: 0.01, from: "lecture_14:jaccard-similarity:transfer" },
    { args: { shared: 900, onlyA: 100, onlyB: 100, simulate: false }, pick: "jaccard", expect: 0.818, tol: 0.01 }, // jaccard transfer's starting pair
    { args: { shared: 0, onlyA: 5, onlyB: 5, seeds: 100 }, pick: "matches", expect: 0, tol: 0 },        // edge: disjoint sets never collide
    { args: { shared: 6, onlyA: 0, onlyB: 0, seeds: 100 }, pick: "matches", expect: 100 },             // edge: identical sets always collide
  ],
};
WIDGETS["fixture:lecture_14--minhash-permutation"] = (root) => {
  const s = { shared: 3, onlyA: 1, onlyB: 1, seed: 0, seeds: 100 };
  const pic = el("div"), read = el("div", { class: "readout" });
  const kind = (x, sh, a) => (+x <= sh ? "shared" : +x <= sh + a ? "A" : "B");
  const col = { shared: C.ok, A: C.a, B: C.b };
  const draw = () => {
    const m = minhashModel(s), { A, B, U } = setsOf(s), W = 640;
    let b = "";
    if (!A.length || !B.length) { pic.innerHTML = svg(W, 40, text(10, 24, "A or B is empty: minhash of an empty set is undefined (min of nothing)", { fill: C.b })); read.innerHTML = ""; return; }
    // the permutation that seed k induces: union items sorted by mmh3.hash(x, k)
    const perm = U.map(x => [x, mmh3(x, s.seed)]).sort((p, q) => p[1] - q[1]);
    const first = perm[0][0], firstA = perm.find(p => kind(p[0], s.shared, s.onlyA) !== "B")[0], firstB = perm.find(p => kind(p[0], s.shared, s.onlyA) !== "A")[0];
    const show = perm.slice(0, 26), bw = 23;
    b += text(10, 16, `A ∪ B (${m.union} items) in the order mmh3.hash(x, seed = ${s.seed}) puts them: the permutation this seed induces${perm.length > show.length ? ` (first ${show.length} shown)` : ""}`, { fill: C.muted, size: 11 });
    show.forEach(([x], i) => {
      const k = kind(x, s.shared, s.onlyA), X = 10 + i * bw;
      b += rect(X, 26, bw - 3, 24, col[k], i === 0 ? 'stroke="#222" stroke-width="2"' : 'opacity="0.8"');
      b += text(X + (bw - 3) / 2, 42, x, { fill: "#fff", size: x.length > 2 ? 9 : 11, anchor: "middle" });
    });
    b += text(10, 66, "green: in A and B · blue: only A · orange: only B · outlined: the first item of the union", { fill: C.muted, size: 11 });
    const same = firstA === firstB;
    b += text(10, 86, `minhash(A) = hash of item ${firstA} · minhash(B) = hash of item ${firstB} → ${same ? "EQUAL: the first item is shared" : `DIFFERENT: the first item (${first}) is only in ${kind(first, s.shared, s.onlyA)}`}`, { fill: same ? C.ok : C.b, size: 12 });
    // seeds strip and running estimate
    const n = s.seeds, x0 = 40, span = 580, top = 104, Hh = 90, xs = i => x0 + span * i / n, ys = v => top + Hh - Hh * v;
    b += line(x0, top, x0, top + Hh, C.rule) + line(x0, top + Hh, x0 + span, top + Hh, C.rule);
    b += text(x0 - 4, top + 4, "1", { fill: C.muted, size: 10, anchor: "end" }) + text(x0 - 4, top + Hh + 3, "0", { fill: C.muted, size: 10, anchor: "end" });
    b += line(x0, ys(m.jaccard), x0 + span, ys(m.jaccard), C.ok, 'stroke-dasharray="4 3"');
    b += text(x0 + span, ys(m.jaccard) - 4, `Jaccard = ${fmt(m.jaccard, 4)}`, { fill: C.ok, size: 11, anchor: "end" });
    let run = 0; const pts = m.hits.map((h, i) => { run += h; return [xs(i + 1), ys(run / (i + 1))]; });
    b += path(pts, C.a);
    const cw = span / n;
    m.hits.forEach((h, i) => { if (h) b += rect(x0 + i * cw, top + Hh + 6, Math.max(cw - (cw > 3 ? 1 : 0), 0.6), 10, C.ok); });
    b += text(x0, top + Hh + 30, `seeds 0…${n - 1}: a green tick = minhashes equal · line: running fraction of equal seeds (${m.matches} of ${n} = ${fmt(m.estimate, 3)})`, { fill: C.muted, size: 11 });
    pic.innerHTML = svg(W, top + Hh + 40, b);
    read.innerHTML = `|A| = ${m.sizeA}, |B| = ${m.sizeB}, |A ∩ B| = ${m.inter}, |A ∪ B| = ${m.union} → <b>Jaccard = ${m.inter}/${m.union} = ${fmt(m.jaccard, 4)}</b><br>
      seed ${s.seed}: ${same ? "minhashes equal" : "minhashes differ"} (one seed is one coin flip with P = Jaccard)<br>
      over ${n} seeds: <span class="big">${m.matches} equal</span> estimate ${fmt(m.estimate, 3)}; expected ${n} × ${fmt(m.jaccard, 4)} = ${fmt(m.expected, 1)}<br>
      <span class="muted small">provenance: fixture:lecture_14--minhash-permutation · lecture_14.py:L222-L230 (Jaccard, A = {1,2,3,4}, B = {1,2,3,5}), L237 (Pr[h(A) = h(B)] = Jaccard), L242-L243 (minhash = min of mmh3.hash(x, seed)), L253-L257 (induced permutation; first item shared ⇔ equal), L260-L263 (100 seeds, within 0.01); video 34:38-36:08. Hashes are MurmurHash3 x86_32 computed in the page, the same function as Python's mmh3.hash; items are the strings "1", "2", …</span>`;
  };
  const sliders = {
    shared: slider("in both (A ∩ B)", 0, 40, s.shared, 1, v => { s.shared = v; draw(); }),
    onlyA: slider("only in A", 0, 40, s.onlyA, 1, v => { s.onlyA = v; draw(); }),
    onlyB: slider("only in B", 0, 40, s.onlyB, 1, v => { s.onlyB = v; draw(); }),
  };
  const preset = (sh, a, bb) => { setSlider(sliders.shared, sh); setSlider(sliders.onlyA, a); setSlider(sliders.onlyB, bb); };
  root.append(el("div", { class: "widget" }, el("div", { class: "controls" },
    sliders.shared, sliders.onlyA, sliders.onlyB,
    slider("seed (which hash function)", 0, 999, s.seed, 1, v => { s.seed = v; draw(); }),
    slider("number of seeds", 1, 1000, s.seeds, 1, v => { s.seeds = v; draw(); }),
    el("div", { style: "margin-top:8px" }, button("L224: A = {1,2,3,4}, B = {1,2,3,5}", () => preset(3, 1, 1))),
    read), pic)); draw();
};

// ========================================================= 2. LSH bands and rows ==
// L281-L288: n = b · r hash functions in b bands of r; A and B collide if for *some* band *all* its hash functions agree.
// L292-L295 get_prob_collision (verbatim): prob_match = sim ** r; prob_collision = 1 - (1 - prob_match) ** b.
// L298: s = 0.8, b = 5, r = 10 → 0.4333. L302-L309: sims [0.7 … 0.98] at (b, r) = (10, 10), (10, 20), (20, 20).
// L312-L321: Lee et al. n = 9000, b = 20, r = 450; threshold (1/b) ** (1/r) = 0.99336; a fixed band matches with 1/b; P = 1 − (1 − 1/b)^b.
// Video 41:40-42:35 (0.8^r, "about 0.4"), 45:13-46:38 (r moves right 0.25 → 0.008; b moves left 0.72 → 0.92), 46:59-48:10 (threshold, 0.64).
// The inverse b = target^(−r) is the threshold formula solved for b (lsh-threshold-one-over-b transfer).
// The concrete pair below uses the minhash of panel 1 (real mmh3) on a 100-item union with round(100·s) shared items, so its Jaccard is s to 2 places.
function lshModel({ b, r, s, target = 0.8 }) {
  const band = Math.pow(s, r), none = Math.pow(1 - band, b), P = 1 - none, thr = Math.pow(1 / b, 1 / r);
  return { n: b * r, band, none, P, thr, pAtThr: 1 - Math.pow(1 - 1 / b, b), bForTarget: Math.pow(target, -r), above: s > thr };
}
const LSH_SIMS = [0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.98];                     // L302
MODELS["fixture:lecture_14--lsh-bands"] = {
  fn: lshModel,
  cases: [
    { args: { b: 5, r: 10, s: 0.8 }, pick: "P", expect: 0.4333, tol: 0.001 },          // L298 (KP note 0.4333)
    { args: { b: 5, r: 10, s: 0.8 }, pick: "band", expect: 0.1074, tol: 0.001 },       // L293 prob_match
    { args: { b: 20, r: 10, s: 0.8 }, pick: "P", expect: 0.897, tol: 0.005 },          // lsh-band-sharpening predict (choice: up, about 0.9)
    { args: { b: 20, r: 5, s: 0.6 }, pick: "P", expect: 0.80, tol: 0.05, from: "lecture_14:lsh-band-sharpening:transfer" },
    { args: { b: 5, r: 20, s: 0.6 }, pick: "P", expect: 1.828e-4, tol: 0.01 },        // the transfer's stricter layout
    { args: { b: 10, r: 10, s: 0.7 }, pick: "P", expect: 0.249, tol: 0.005 },          // L303 (KP note)
    { args: { b: 10, r: 20, s: 0.7 }, pick: "P", expect: 0.008, tol: 0.05 },           // L306: r moved the curve right (video 45:13-45:48)
    { args: { b: 10, r: 20, s: 0.9 }, pick: "P", expect: 0.726, tol: 0.005 },          // L306, "0.72" at 0.9 (video 45:51)
    { args: { b: 20, r: 20, s: 0.9 }, pick: "P", expect: 0.925, tol: 0.005 },          // L309, "0.92": b moved it left
    { args: { b: 20, r: 450, s: 0.9 }, pick: "thr", expect: 0.99336, tol: 0.0001 },    // L316 Lee et al. threshold
    { args: { b: 20, r: 450, s: 0.9 }, pick: "pAtThr", expect: 0.6415, tol: 0.001 },   // L321 (KP note), "0.64" aloud
    { args: { b: 90, r: 100, s: 0.9 }, pick: "thr", expect: 0.956, tol: 0.001 },       // lsh-threshold-one-over-b predict (choice: down, 0.956)
    { args: { b: 1, r: 10, s: 0.9, target: 0.8 }, pick: "bForTarget", expect: 9.3, tol: 0.01, from: "lecture_14:lsh-threshold-one-over-b:transfer" },
    { args: { b: 1, r: 1, s: 0.37 }, pick: "P", expect: 0.37 },                        // edge: one MinHash is the diagonal P = s (L272)
    { args: { b: 1e4, r: 1, s: 0.5 }, pick: "pAtThr", expect: 1 - 1 / Math.E, tol: 0.001 }, // edge: 1 − (1 − 1/b)^b → 1 − 1/e
  ],
};
WIDGETS["fixture:lecture_14--lsh-bands"] = (root) => {
  const s = { b: 5, r: 10, s: 0.8, target: 0.9, trial: 0 };
  const pic = el("div"), pair = el("div"), read = el("div", { class: "readout" });
  const draw = () => {
    const m = lshModel(s), W = 640;
    // S-curve
    const x0 = 46, y0 = 20, w = 360, h = 200, X = v => x0 + w * v, Y = v => y0 + h - h * v;
    let b = line(x0, y0, x0, y0 + h, C.rule) + line(x0, y0 + h, x0 + w, y0 + h, C.rule);
    for (let t = 0; t <= 10; t += 2) b += text(X(t / 10), y0 + h + 14, (t / 10).toFixed(1), { fill: C.muted, size: 10, anchor: "middle" }) + text(x0 - 6, Y(t / 10) + 4, (t / 10).toFixed(1), { fill: C.muted, size: 10, anchor: "end" });
    b += text(x0 + w, y0 + h + 28, "Jaccard similarity s", { fill: C.muted, size: 11, anchor: "end" });
    b += text(x0 + 4, y0 - 6, "P(collide) = 1 − (1 − s^r)^b", { fill: C.muted, size: 11 });
    b += line(X(0), Y(0), X(1), Y(1), C.rule, 'stroke-dasharray="4 4"') + text(X(0.55), Y(0.48), "b = r = 1: P = s", { fill: C.muted, size: 10 });
    const pts = []; for (let i = 0; i <= 600; i++) { const v = i / 600; pts.push([X(v), Y(lshModel({ b: s.b, r: s.r, s: v }).P)]); }
    b += path(pts, C.a);
    b += line(X(m.thr), y0, X(m.thr), y0 + h, C.hi, 'stroke-dasharray="3 2"');
    b += text(Math.min(X(m.thr), x0 + w) - 4, y0 + 12, `s* = ${m.thr.toFixed(4)}`, { fill: C.ink, size: 11, anchor: "end" });
    for (const v of LSH_SIMS) b += `<circle cx="${X(v).toFixed(1)}" cy="${Y(lshModel({ b: s.b, r: s.r, s: v }).P).toFixed(1)}" r="2.5" style="fill:${C.a}"/>`;
    b += `<circle cx="${X(s.s).toFixed(1)}" cy="${Y(m.P).toFixed(1)}" r="6" style="fill:${C.b}"/>`;
    // and-or panel: each band is an AND over r rows (s^r); the pair collides on an OR over b bands
    const px = 440, pw = 180, shown = Math.min(s.b, 12), bh = Math.min(12, 150 / shown);
    b += text(px, y0 - 6, `AND: one band, all ${s.r} rows equal`, { fill: C.muted, size: 11 });
    for (let i = 0; i < shown; i++) { const yy = y0 + 4 + i * (bh + 2); b += rect(px, yy, pw, bh, C.rule, 'opacity="0.5"') + rect(px, yy, pw * m.band, bh, C.a); }
    const yb = y0 + 4 + shown * (bh + 2);
    if (s.b > shown) b += text(px, yb + 10, `… ${s.b - shown} more bands, each the same s^r`, { fill: C.muted, size: 10 });
    b += text(px, yb + 26, `each band: s^r = ${fmt(m.band, 4)}`, { size: 11 });
    b += text(px, yb + 44, `OR over ${s.b} bands:`, { fill: C.muted, size: 11 });
    b += rect(px, yb + 50, pw, 14, C.rule, 'opacity="0.5"') + rect(px, yb + 50, pw * m.P, 14, C.b);
    b += text(px, yb + 78, `1 − (1 − s^r)^b = ${fmt(m.P, 4)}`, { size: 11, weight: "bold" });
    const regime = s.b === 1 && s.r === 1 ? "one MinHash: P = s, no threshold at all" : m.above ? `s = ${s.s.toFixed(3)} is above s* = ${m.thr.toFixed(4)}: the pair usually collides` : `s = ${s.s.toFixed(3)} is below s* = ${m.thr.toFixed(4)}: the pair usually does not collide`;
    b += text(10, y0 + h + 50, regime, { fill: m.above ? C.ok : C.b, size: 12 });
    pic.innerHTML = svg(W, y0 + h + 60, b);
    drawPair();
    read.innerHTML = `n = b · r = ${m.n} MinHashes in ${s.b} bands of ${s.r} rows<br>
      at s = ${s.s.toFixed(3)}: one band matches with s^r = ${fmt(m.band, 4)}; no band matches with (1 − s^r)^b = ${fmt(m.none, 4)} → <span class="big">P(collide) = ${fmt(m.P, 4)}</span><br>
      threshold s* = (1/b)^(1/r) = <b>${m.thr.toFixed(4)}</b>: there a band matches with 1/b = ${fmt(1 / s.b, 4)} and P = 1 − (1 − 1/b)^b = ${m.pAtThr.toFixed(4)} (1 − 1/e = ${(1 - 1 / Math.E).toFixed(4)})<br>
      to put s* at ${s.target.toFixed(3)} with r = ${s.r}: b = s*^(−r) = <b>${fmt(m.bForTarget, 3)}</b> bands<br>
      small dots: the professor's sims ${LSH_SIMS.join(", ")} → ${LSH_SIMS.map(v => lshModel({ b: s.b, r: s.r, s: v }).P.toFixed(3)).join(", ")}<br>
      <span class="muted small">provenance: fixture:lecture_14--lsh-bands · lecture_14.py:L281-L288 (b bands of r; some band, all rows), L292-L295 (get_prob_collision), L298 (0.8, 5, 10), L302-L309 (sims; (10,10), (10,20), (20,20)), L312-L321 (Lee et al. b = 20, r = 450; (1/b)^(1/r); 1 − (1 − 1/b)^b); video 41:40-48:10. Assumes the n hashes are independent (video 38:53).</span>`;
  };
  // one concrete pair, hashed for real: b·r MinHashes with seeds trial·n … trial·n + n − 1
  const drawPair = () => {
    const n = s.b * s.r;
    if (s.r > 60 || s.b > 24) { pair.innerHTML = `<div class="muted small">concrete pair: shown for b ≤ 24 and r ≤ 60 (now ${s.b} × ${s.r})</div>`; return; }
    const k = Math.round(100 * s.s), { A, B } = setsOf({ shared: k, onlyA: Math.ceil((100 - k) / 2), onlyB: Math.floor((100 - k) / 2) });
    const W = 640, x0 = 70, cw = Math.min(14, 480 / s.r), ch = Math.min(12, 150 / s.b);
    let body = text(10, 14, `one pair with Jaccard ${(k / 100).toFixed(2)} (100-item union, ${k} shared), hash draw #${s.trial}: filled = this row's MinHashes agree`, { fill: C.muted, size: 11 });
    let anyBand = -1;
    for (let i = 0; i < s.b; i++) {
      let all = true; const y = 22 + i * (ch + 2);
      for (let j = 0; j < s.r; j++) {
        const seed = s.trial * n + i * s.r + j, eq = minhash(A, seed) === minhash(B, seed);
        all = all && eq; body += rect(x0 + j * cw, y, cw - 1, ch, eq ? C.a : C.rule, eq ? "" : 'opacity="0.6"');
      }
      if (all && anyBand < 0) anyBand = i;
      body += text(x0 - 6, y + ch - 2, `band ${i + 1}`, { fill: all ? C.ok : C.muted, size: Math.min(10, ch), anchor: "end" });
      if (all) body += text(x0 + s.r * cw + 6, y + ch - 2, "all equal", { fill: C.ok, size: Math.min(10, ch) });
    }
    const yy = 22 + s.b * (ch + 2) + 16;
    body += text(10, yy, anyBand >= 0 ? `collide: band ${anyBand + 1} has all ${s.r} rows equal (one band is enough)` : `no collision: every band has at least one row that differs`, { fill: anyBand >= 0 ? C.ok : C.b, size: 12 });
    pair.innerHTML = svg(W, yy + 10, body);
  };
  const sb = slider("bands b", 1, 200, s.b, 1, v => { s.b = v; draw(); });
  const sr = slider("rows per band r", 1, 500, s.r, 1, v => { s.r = v; draw(); });
  const ss = slider("similarity s", 0, 1, s.s, 0.001, v => { s.s = v; draw(); }, v => v.toFixed(3));
  const presets = [["L298: b=5, r=10", 5, 10, 0.8], ["L303: b=10, r=10", 10, 10, 0.85], ["L306: b=10, r=20", 10, 20, 0.85], ["L309: b=20, r=20", 20, 20, 0.85], ["Lee et al.: b=20, r=450", 20, 450, 0.99], ["one MinHash: b=r=1", 1, 1, 0.8]];
  root.append(el("div", { class: "widget" }, el("div", { class: "controls" }, sb, sr, ss,
    slider("target threshold s* (solve for b)", 0.5, 0.999, s.target, 0.001, v => { s.target = v; draw(); }, v => v.toFixed(3)),
    el("div", { style: "margin-top:8px" }, ...presets.map(([lab, b2, r2, s2]) => button(lab, () => { setSlider(sb, b2); setSlider(sr, r2); setSlider(ss, s2); })),
      button("new hash draw for the pair", () => { s.trial++; drawPair(); })),
    read), pic, pair)); draw();
};

// ============================================ 3. exact dedup: hash, group, keep one; scope ==
// L163-L170: design space (item, match, action). L183-L188: fast non-cryptographic MurmurHash, h = mmh3.hash("hello").
// L196-L204: items = ["Hello!", "hello", "hello there", "hello", "hi", "bye"]; groupby(sorted(items, key=mmh3.hash), key=mmh3.hash);
//            keep next(group) of each group. L206-L214: high precision, misses near duplicates, MapReduce; C4 3-sentence spans.
// Scope (dedup-across-sources): video 48:55-49:21, dedup "will happen within a data set ... but you actually have to do deduplication
// across your entire data set because often data sets can be redundant"; decontamination is the same deal (video 26:58-27:14).
// A line "name: text" puts the item in source "name"; a line with no "name:" prefix is in source "(none)".
const parseLines = (txt) => txt.split("\n").filter(l => l.length).map(l => { const m = /^([A-Za-z0-9_-]{1,16}):\s?(.*)$/.exec(l); return m ? { src: m[1], item: m[2] } : { src: "(none)", item: l }; });
function dedupModel({ lines, scope = "union" }) {
  // lines: [{src, item}]. Group key = mmh3.hash(item), plus the source when dedup runs within each source.
  const groups = new Map();
  for (const it of lines) {
    const h = mmh3(it.item), key = scope === "union" ? `${h}` : `${it.src}\u0000${h}`;
    if (!groups.has(key)) groups.set(key, { h, members: [] });
    groups.get(key).members.push(it);
  }
  const kept = groups.size, sorted = [...groups.values()].sort((p, q) => p.h - q.h);
  return { items: lines.length, kept, removed: lines.length - kept, groups: sorted, deduped: sorted.map(g => g.members[0].item) };
}
const LEC_ITEMS = ["Hello!", "hello", "hello there", "hello", "hi", "bye"];   // L198
const asLines = (arr, src = "(none)") => arr.map(item => ({ src, item }));
// dedup-across-sources check, built as documents: A 2,000; B 800 of which 300 are A's; C 500 of which 100 are B's own (not in A).
const CHECK_SOURCES = (() => {
  const A = Array.from({ length: 2000 }, (_, i) => `a${i}`);
  const B = [...A.slice(0, 300), ...Array.from({ length: 500 }, (_, i) => `b${i}`)];
  const Cs = [...B.slice(300, 400), ...Array.from({ length: 400 }, (_, i) => `c${i}`)];
  return [...asLines(A, "A"), ...asLines(B, "B"), ...asLines(Cs, "C")];
})();
MODELS["fixture:lecture_14--exact-dedup"] = {
  fn: dedupModel,
  cases: [
    { args: { lines: asLines(LEC_ITEMS) }, pick: "kept", expect: 5, tol: 0 },              // exact-dedup-hash-groupby predict (choice: 5)
    { args: { lines: asLines(["hello"]) }, pick: "items", expect: 1 },                      // sanity
    { args: { lines: CHECK_SOURCES, scope: "union" }, pick: "kept", expect: 2900, tol: 0.001, from: "lecture_14:dedup-across-sources:check" },
    { args: { lines: CHECK_SOURCES, scope: "source" }, pick: "kept", expect: 3300, tol: 0 },  // each source alone: nothing removed
    { args: { lines: [...asLines(["x", "y"], "cc"), ...asLines(["x"], "wiki")], scope: "source" }, pick: "removed", expect: 0, tol: 0 }, // dedup-across-sources predict: the cross-source copy survives
    { args: { lines: [...asLines(["x", "y"], "cc"), ...asLines(["x"], "wiki")], scope: "union" }, pick: "removed", expect: 1, tol: 0 },
  ],
};
WIDGETS["fixture:lecture_14--exact-dedup"] = (root) => {
  const DEFAULT = ["cc: The Moon is Earth's only natural satellite.", "cc: Permission is hereby granted, free of charge", "cc: Permission is hereby granted, free of charge",
    "cc: Permission is hereby granted, free of charge.", "wiki: The Moon is Earth's only natural satellite.", "hqweb: The Moon is Earth's only natural satellite.",
    "hqweb: the Moon is Earth's only natural satellite.", "hqweb: Permission is hereby granted, free of charge"].join("\n");
  const s = { scope: "source" };
  const ta = el("textarea", { rows: 9, style: "width:100%;max-width:640px;font:12px var(--mono, monospace)" }); ta.value = DEFAULT;
  const pic = el("div"), read = el("div", { class: "readout" });
  const draw = () => {
    const lines = parseLines(ta.value), m = dedupModel({ lines, scope: s.scope }), other = dedupModel({ lines, scope: s.scope === "union" ? "source" : "union" });
    const W = 640, rowH = 20, shown = m.groups.slice(0, 16);
    let b = text(10, 16, `groups after sorting by mmh3.hash(item)${s.scope === "union" ? "" : " within each source"} · keep the first of each group`, { fill: C.muted, size: 11 });
    const srcOfHash = new Map(); for (const it of lines) { const h = mmh3(it.item); if (!srcOfHash.has(h)) srcOfHash.set(h, new Set()); srcOfHash.get(h).add(it.src); }
    shown.forEach((g, i) => {
      const y = 26 + i * rowH, dup = g.members.length > 1, twin = s.scope !== "union" && srcOfHash.get(g.h).size > 1;
      b += rect(8, y, W - 16, rowH - 3, dup ? C.hi : twin ? C.b : C.rule, twin && !dup ? 'opacity="0.18"' : 'opacity="0.35"');
      b += text(14, y + 13, String(g.h).padStart(11), { fill: C.muted, size: 11 });
      b += text(100, y + 13, esc(g.members[0].item.length > 52 ? g.members[0].item.slice(0, 50) + "…" : g.members[0].item), { size: 11 });
      b += text(W - 14, y + 13, `${g.members.map(x => x.src).join(", ")}${dup ? ` → keep 1, drop ${g.members.length - 1}` : ""}${twin ? " · same bytes in another source, kept" : ""}`, { fill: dup || twin ? C.b : C.muted, size: 11, anchor: "end" });
    });
    const yy = 26 + shown.length * rowH + 14;
    if (m.groups.length > shown.length) b += text(10, yy, `… ${m.groups.length - shown.length} more groups`, { fill: C.muted, size: 10 });
    pic.innerHTML = svg(W, yy + 8, b);
    const srcs = [...new Set(lines.map(x => x.src))];
    const within = s.scope === "union" ? other : m, union = s.scope === "union" ? m : other;
    read.innerHTML = `${m.items} items from ${srcs.length} source${srcs.length === 1 ? "" : "s"} (${srcs.map(esc).join(", ")})<br>
      dedup within each source: keep <b>${within.kept}</b> · dedup over the union: keep <b>${union.kept}</b><br>
      <span class="big">${within.kept - union.kept}</span> copies survive per-source dedup because their twin sits in another source<br>
      exact match: two items fall in one group only if every byte is equal (case, punctuation and whitespace included)<br>
      <span class="muted small">provenance: fixture:lecture_14--exact-dedup · lecture_14.py:L163-L170 (item, match, action), L183-L188 (MurmurHash), L196-L204 (the hello list; groupby on sorted hashes; keep one), L206-L214 (misses near duplicates; C4 spans); video 48:55-49:21 (dedup across the entire data set). Hashes are MurmurHash3 x86_32 = Python mmh3.hash, computed in the page. The default lines are this widget's own (the MIT-license sentence is L151's example of a near duplicate).</span>`;
  };
  ta.addEventListener("input", draw);
  const scopeBtn = el("select", {}, el("option", { value: "source" }, "dedup within each source"), el("option", { value: "union" }, "dedup over the union of all sources"));
  scopeBtn.addEventListener("change", () => { s.scope = scopeBtn.value; draw(); });
  root.append(el("div", { class: "widget" }, el("div", { class: "controls" },
    el("div", { class: "small muted" }, "one item per line; prefix \"source: \" to say which dataset it came from"), ta,
    el("label", {}, "scope ", scopeBtn),
    el("div", { style: "margin-top:8px" }, button("L198 list (one source)", () => { ta.value = LEC_ITEMS.join("\n"); draw(); }), button("three sources (default)", () => { ta.value = DEFAULT; draw(); })),
    read), pic)); draw();
};

// ================================================ 4. GPT-3's stochastic keep rule ==
// L104-L110: GPT-3 (Appendix A) keeps a document when np.random.pareto(9) > 1 - score. L83: "keep examples x with score(x) >= threshold
// (stochastically)". numpy's pareto(a) is the Lomax law, P(X > x) = (1 + x)^−a for x ≥ 0 (stated in the pareto-stochastic-keep check),
// so P(keep | score) = (2 − score)^−a; the KP note's 2e6-draw Monte Carlo gave 0.0020, 0.026, 0.424, 1.0 at scores 0, 0.5, 0.9, 1.
// Score where P(keep) = q: 2 − q^(−1/a) (pareto-stochastic-keep check, q = 0.5 → 0.92). The draws below are X = U^(−1/a) − 1, U uniform.
const mulberry32 = seed => () => { seed |= 0; seed = seed + 0x6D2B79F5 | 0; let t = Math.imul(seed ^ seed >>> 15, 1 | seed); t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t; return ((t ^ t >>> 14) >>> 0) / 4294967296; };
function paretoKeep({ a = 9, score, t = 0.5, q = 0.5 }) {
  const x = 1 - score, keep = x <= 0 ? 1 : Math.pow(1 + x, -a);
  return { keep, keep0: Math.pow(2, -a), scoreAtQ: 2 - Math.pow(q, -1 / a), hard: score >= t ? 1 : 0, vsShape9: keep / (x <= 0 ? 1 : Math.pow(1 + x, -9)) };
}
MODELS["fixture:lecture_14--pareto-keep"] = {
  fn: paretoKeep,
  cases: [
    { args: { a: 9, score: 0.9 }, pick: "keep", expect: 0.424, tol: 0.005 },          // pareto predict (choice: about 0.42 vs 0.026)
    { args: { a: 9, score: 0.5 }, pick: "keep", expect: 0.026, tol: 0.01 },
    { args: { a: 9, score: 0 }, pick: "keep", expect: 0.00195, tol: 0.01 },           // transfer: 2^−9 ≈ 0.002
    { args: { a: 3, score: 0 }, pick: "keep", expect: 0.125 },                         // transfer: 2^−3 with shape 3
    { args: { a: 3, score: 0 }, pick: "vsShape9", expect: 64 },                        // transfer: "about 64x more junk"
    { args: { a: 9, score: 0.5, q: 0.5 }, pick: "scoreAtQ", expect: 0.92, tol: 0.01, from: "lecture_14:pareto-stochastic-keep:check" },
    { args: { a: 9, score: 0.25 }, pick: "keep", expect: 0.0065, tol: 0.02 },         // KP note Monte Carlo
    { args: { a: 9, score: 0.75 }, pick: "keep", expect: 0.134, tol: 0.01 },
    { args: { a: 9, score: 1 }, pick: "keep", expect: 1 },                             // edge: score 1 is always kept
  ],
};
WIDGETS["fixture:lecture_14--pareto-keep"] = (root) => {
  const s = { a: 9, score: 0.7, t: 0.5, log: false, draw: 0 };
  const pic = el("div"), read = el("div", { class: "readout" });
  const draw = () => {
    const m = paretoKeep(s), W = 640, x0 = 56, y0 = 18, w = 540, h = 200, X = v => x0 + w * v;
    const lo = 1e-4, Y = v => s.log ? y0 + h - h * (Math.log10(Math.max(v, lo)) - Math.log10(lo)) / -Math.log10(lo) : y0 + h - h * v;
    let b = line(x0, y0, x0, y0 + h, C.rule) + line(x0, y0 + h, x0 + w, y0 + h, C.rule);
    for (let i = 0; i <= 10; i += 2) b += text(X(i / 10), y0 + h + 14, (i / 10).toFixed(1), { fill: C.muted, size: 10, anchor: "middle" });
    (s.log ? [1e-4, 1e-3, 1e-2, 0.1, 1] : [0, 0.25, 0.5, 0.75, 1]).forEach(v => { b += text(x0 - 6, Y(v) + 4, String(v), { fill: C.muted, size: 10, anchor: "end" }); });
    b += text(x0 + w, y0 + h + 28, "classifier score", { fill: C.muted, size: 11, anchor: "end" });
    b += text(x0 + 4, y0 - 4, `P(keep) ${s.log ? "(log scale)" : ""}`, { fill: C.muted, size: 11 });
    const curve = (a) => { const p = []; for (let i = 0; i <= 200; i++) { const v = i / 200; p.push([X(v), Y(paretoKeep({ a, score: v }).keep)]); } return p; };
    b += path([[X(0), Y(0 + lo * s.log)], [X(s.t), Y(0 + lo * s.log)], [X(s.t), Y(1)], [X(1), Y(1)]], C.muted, 'stroke-dasharray="5 3"');
    b += text(X(s.t) + 4, Y(1) + 14, `hard threshold t = ${s.t.toFixed(2)}`, { fill: C.muted, size: 10 });
    if (s.a !== 9) b += path(curve(9), C.rule, 'stroke-dasharray="2 2"');
    b += path(curve(s.a), C.a);
    const half = m.scoreAtQ;
    if (half >= 0 && half <= 1) b += `<circle cx="${X(half).toFixed(1)}" cy="${Y(0.5).toFixed(1)}" r="3.5" style="fill:${C.ink}"/>` + text(X(half) - 6, Y(0.5) + 4, `½ at ${half.toFixed(3)}`, { size: 10, anchor: "end" });
    b += `<circle cx="${X(s.score).toFixed(1)}" cy="${Y(m.keep).toFixed(1)}" r="6" style="fill:${C.b}"/>`;
    // 1,000 documents at this score, each with its own Pareto draw
    const rnd = mulberry32(1000 * s.draw + Math.round(1000 * s.score) + 7 * s.a); let kept = 0;
    const cells = []; for (let i = 0; i < 1000; i++) { const X1 = Math.pow(1 - rnd(), -1 / s.a) - 1, k = X1 > 1 - s.score; kept += k; if (i < 200) cells.push(k); }
    const yy = y0 + h + 44;
    b += text(10, yy, `200 of 1,000 documents at score ${s.score.toFixed(2)}, one Pareto(${s.a}) draw each (filled = kept)`, { fill: C.muted, size: 11 });
    cells.forEach((k, i) => { b += rect(10 + (i % 100) * 6.2, yy + 8 + Math.floor(i / 100) * 9, 5, 7, k ? C.ok : C.rule, k ? "" : 'opacity="0.6"'); });
    pic.innerHTML = svg(W, yy + 30, b);
    read.innerHTML = `shape a = ${s.a} · score ${s.score.toFixed(2)}: keep when Pareto(${s.a}) > 1 − score = ${(1 - s.score).toFixed(2)}<br>
      <span class="big">P(keep) = (2 − score)^−a = ${fmt(m.keep, 4)}</span> · this draw of 1,000: ${kept} kept · hard threshold at ${s.t.toFixed(2)} would ${m.hard ? "always keep" : "never keep"} it<br>
      score 0 is kept with 2^−a = ${fmt(m.keep0, 4)}${s.a !== 9 ? ` (${fmt(m.vsShape9, 3)}× the shape-9 rate at this score)` : ""} · kept more often than not above score 2 − 2^(1/a) = ${fmt(half, 4)}<br>
      <span class="muted small">provenance: fixture:lecture_14--pareto-keep · lecture_14.py:L104-L110 (GPT-3 Appendix A; keep_document = np.random.pareto(9) > 1 − score), L83 (threshold, stochastically); numpy pareto(a) tail (1 + x)^−a as stated in the pareto-stochastic-keep check; the KP note's Monte Carlo agrees to three digits. The lecture defines keep_document but never runs it, so it shows no keep probabilities: every number here is this formula evaluated. The hard threshold line is for contrast (L83, LLaMA's "classified positive", L115); the lecture gives no value for t.</span>`;
  };
  const chk = el("input", { type: "checkbox" }); chk.addEventListener("change", () => { s.log = chk.checked; draw(); });
  root.append(el("div", { class: "widget" }, el("div", { class: "controls" },
    slider("Pareto shape a", 1, 20, s.a, 0.5, v => { s.a = v; draw(); }),
    slider("classifier score", 0, 1, s.score, 0.01, v => { s.score = v; draw(); }, v => v.toFixed(2)),
    slider("hard threshold t (for contrast)", 0, 1, s.t, 0.01, v => { s.t = v; draw(); }, v => v.toFixed(2)),
    el("label", {}, chk, " log scale for P(keep)"),
    el("div", { style: "margin-top:8px" }, button("draw 1,000 again", () => { s.draw++; draw(); })),
    read), pic)); draw();
};

// ========================================================= 5. mixture → epochs per source ==
// L339-L348: p(s) over sources; vibes, uniform p ∝ 1, proportional p ∝ num_tokens(s). L369: in between, p ∝ num_tokens(s)^α, α in [0, 1].
// L357-L365: low 10T, high 10B, p = 0.5/0.5, 1T training tokens → epochs = p(s)·N / num_tokens(s) = 0.05 and 50 ("50x epochs ... overfitting").
// L367-L371 UniMax: sample uniformly with a hard cap C on epochs of any source, written p(s)·N ≤ C. Here the cap is in epochs
// (p(s)·N ≤ C·num_tokens(s), the professor's gloss "cap C on number of epochs", L370), and the budget a capped source cannot take is
// shared uniformly by the rest, smallest source first (the UniMax paper's procedure, linked at L367).
// L383-L390: a small-run optimum p = 0.1/0.9 epochs a ton at large scale. L392-L401 simulated epoching: ratio = small_run_tokens /
// large_run_tokens = 10B / 1T = 0.01; downsample every source by it (10T → 100B, 10B → 100M); L401's 0.7/0.3 then means 30 epochs (KP note).
// Video 54:23-55:37 (0.05 and 50 epochs), 58:51-59:28 (power α, then UniMax), 1:07:24-1:09:36 (cap the epochs; simulated epoching).
const TOK = { "": 1, K: 1e3, M: 1e6, B: 1e9, T: 1e12 };
const parseTok = (x) => { const m = /^([\d.]+(?:e[+-]?\d+)?)([KMBT]?)$/i.exec(String(x).trim()); return m ? +m[1] * TOK[m[2].toUpperCase()] : NaN; };
const showTok = (x) => !Number.isFinite(x) ? "–" : x >= 1e12 ? `${fmt(x / 1e12, 3)}T` : x >= 1e9 ? `${fmt(x / 1e9, 3)}B` : x >= 1e6 ? `${fmt(x / 1e6, 3)}M` : fmt(x, 3);
function mixtureModel({ sizes, weights = [], mode = "custom", alpha = 1, cap = 1, N, pilot }) {
  const K = sizes.length; let p;
  if (mode === "alpha") { const w = sizes.map(z => Math.pow(z, alpha)), t = w.reduce((a, b) => a + b, 0); p = w.map(x => x / t); }
  else if (mode === "unimax") {
    p = new Array(K).fill(0); let left = N; const order = sizes.map((z, i) => i).sort((i, j) => sizes[i] - sizes[j]);
    order.forEach((i, j) => { const take = Math.min(left / (K - j), cap * sizes[i]); p[i] = take / N; left -= take; });
  } else p = weights.slice(0, K);
  const sumP = p.reduce((a, b) => a + b, 0), drawn = p.map(x => x * N), epochs = drawn.map((d, i) => d / sizes[i]);
  const ratio = pilot / N, down = sizes.map(z => z * ratio), pilotEpochs = p.map((x, i) => x * pilot / down[i]), rawPilotEpochs = p.map((x, i) => x * pilot / sizes[i]);
  const o = { p, sumP, drawn, epochs, ratio, down, pilotEpochs, rawPilotEpochs, maxEpochs: Math.max(...epochs) };
  for (let i = 0; i < K; i++) Object.assign(o, { [`p${i}`]: p[i], [`epochs${i}`]: epochs[i], [`down${i}`]: down[i], [`pilotEpochs${i}`]: pilotEpochs[i], [`rawPilotEpochs${i}`]: rawPilotEpochs[i] });
  return o;
}
const LOWHIGH = [1e13, 1e10];                                                      // L358-L359
MODELS["fixture:lecture_14--mixture-epochs"] = {
  fn: mixtureModel,
  cases: [
    { args: { sizes: LOWHIGH, weights: [0.5, 0.5], N: 1e12, pilot: 1e10 }, pick: "epochs0", expect: 0.05 },   // L363
    { args: { sizes: LOWHIGH, weights: [0.5, 0.5], N: 1e12, pilot: 1e10 }, pick: "epochs1", expect: 50 },     // L364
    { args: { sizes: [1e13, 3e10], weights: [0.7, 0.3], N: 2e12, pilot: 1e10 }, pick: "epochs1", expect: 20, tol: 0.02, from: "lecture_14:mixing-baselines-epoching:predict" },
    { args: { sizes: [2e12, 5e11, 5e10, 5e9, 5e8], mode: "alpha", alpha: 0, N: 1e12, pilot: 1e10 }, pick: "epochs4", expect: 400, tol: 0.05, from: "lecture_14:mixing-baselines-epoching:transfer" },
    { args: { sizes: LOWHIGH, weights: [0.1, 0.9], N: 1e12, pilot: 1e10 }, pick: "epochs1", expect: 90, tol: 0.02, from: "lecture_14:regression-mixing-simulated-epoching:predict" },
    { args: { sizes: LOWHIGH, weights: [0.1, 0.9], N: 1e12, pilot: 1e10 }, pick: "rawPilotEpochs1", expect: 0.9 }, // the pilot itself barely repeats
    { args: { sizes: LOWHIGH, weights: [0.1, 0.9], N: 4e12, pilot: 2e10 }, pick: "ratio", expect: 0.005, tol: 0.02, from: "lecture_14:regression-mixing-simulated-epoching:transfer" },
    { args: { sizes: LOWHIGH, weights: [0.5, 0.5], N: 1e12, pilot: 1e10 }, pick: "ratio", expect: 0.01 },     // L397
    { args: { sizes: LOWHIGH, weights: [0.5, 0.5], N: 1e12, pilot: 1e10 }, pick: "down0", expect: 1e11 },     // L398: 10T → 100B
    { args: { sizes: LOWHIGH, weights: [0.5, 0.5], N: 1e12, pilot: 1e10 }, pick: "down1", expect: 1e8 },      // L398: 10B → 100M
    { args: { sizes: LOWHIGH, weights: [0.7, 0.3], N: 1e12, pilot: 1e10 }, pick: "pilotEpochs1", expect: 30 }, // L401 on the downsampled pool (KP note)
    { args: { sizes: LOWHIGH, mode: "alpha", alpha: 1, N: 1e12, pilot: 1e10 }, pick: "epochs1", expect: 1e12 / 1.001e13 }, // proportional: every source equal epochs
    { args: { sizes: LOWHIGH, mode: "unimax", cap: 1, N: 1e12, pilot: 1e10 }, pick: "epochs1", expect: 1 },   // UniMax cap 1: small source capped
    { args: { sizes: LOWHIGH, mode: "unimax", cap: 1, N: 1e12, pilot: 1e10 }, pick: "epochs0", expect: 0.099 }, // ... the rest goes to the big one
  ],
};
WIDGETS["fixture:lecture_14--mixture-epochs"] = (root) => {
  const s = { mode: "custom", alpha: 1, cap: 1, logN: 12, logPilot: 10 };
  const ta = el("textarea", { rows: 4, style: "width:100%;max-width:640px;font:12px var(--mono, monospace)" }); ta.value = "low 10T 0.5\nhigh 10B 0.5";
  const pic = el("div"), read = el("div", { class: "readout" });
  const draw = () => {
    const rows = ta.value.split("\n").map(l => l.trim().split(/\s+/)).filter(r => r.length >= 2 && r[0]);
    const names = rows.map(r => r[0]), sizes = rows.map(r => parseTok(r[1])), weights = rows.map(r => +(r[2] ?? NaN));
    const N = Math.pow(10, s.logN), pilot = Math.pow(10, Math.min(s.logPilot, s.logN));
    if (!rows.length || sizes.some(z => !(z > 0)) || (s.mode === "custom" && weights.some(w => !(w >= 0)))) { pic.innerHTML = ""; read.innerHTML = `<span class="muted">each line: name, tokens (e.g. 10T, 500B, 30B, 500M)${s.mode === "custom" ? ", weight" : ""}</span>`; return; }
    const m = mixtureModel({ sizes, weights, mode: s.mode, alpha: s.alpha, cap: s.cap, N, pilot });
    const W = 640, x0 = 110, span = 360, row = 34, lo = -3, hi = 3, X = e => x0 + span * (Math.min(Math.max(Math.log10(Math.max(e, 1e-9)), lo), hi) - lo) / (hi - lo);
    let b = text(10, 16, `epochs of each source = p(s) · N / tokens(s), log scale · production run N = ${showTok(N)} tokens · dashed: 1 epoch`, { fill: C.muted, size: 11 });
    [-3, -2, -1, 0, 1, 2, 3].forEach(k => { b += text(X(10 ** k), 30 + rows.length * row + 8, k >= 0 ? `${10 ** k}` : `${10 ** k}`, { fill: C.muted, size: 10, anchor: "middle" }); });
    b += line(X(1), 22, X(1), 26 + rows.length * row - 6, C.ink, 'stroke-dasharray="4 3"');
    names.forEach((nm, i) => {
      const y = 26 + i * row, e = m.epochs[i];
      b += text(10, y + 13, esc(nm), { size: 12 }) + text(10, y + 27, `${showTok(sizes[i])} · p = ${fmt(m.p[i], 4)}`, { fill: C.muted, size: 10 });
      b += rect(x0, y + 3, X(e) - x0, 20, e > 1 ? C.b : C.a);
      b += text(Math.min(X(e), x0 + span) + 6, y + 17, `${fmt(e, 3)} epochs${e < 1 ? ` (${fmt(100 * e, 3)}% seen once)` : ""}`, { fill: e > 1 ? C.b : C.ink, size: 11, halo: true });
    });
    pic.innerHTML = svg(W, 30 + rows.length * row + 14, b);
    const warn = Math.abs(m.sumP - 1) > 1e-6 ? `<br><b style="color:${C.b}">weights sum to ${fmt(m.sumP, 4)}${s.mode === "unimax" ? `: the cap leaves ${showTok(N * (1 - m.sumP))} of the budget unplaceable (N > Σ C · tokens)` : ", not 1"}</b>` : "";
    read.innerHTML = `${s.mode === "alpha" ? `p(s) ∝ tokens(s)^α, α = ${s.alpha} (${s.alpha === 0 ? "uniform" : s.alpha === 1 ? "proportional" : "between uniform and proportional"})` : s.mode === "unimax" ? `UniMax: uniform, but no source above C = ${s.cap} epochs` : "custom weights (third column)"}${warn}<br>
      most-repeated source: <span class="big">${fmt(m.maxEpochs, 3)} epochs</span> (${names[m.epochs.indexOf(m.maxEpochs)]})<br>
      simulated epoching: pilot run ${showTok(pilot)} tokens → ratio pilot / production = <b>${fmt(m.ratio, 4)}</b>; downsampled sources ${names.map((nm, i) => `${esc(nm)} ${showTok(m.down[i])}`).join(", ")}<br>
      pilot epochs on the downsampled pool: ${m.pilotEpochs.map((e, i) => `${esc(names[i])} ${fmt(e, 3)}`).join(", ")} (equal to production) · on the full pool they would be ${m.rawPilotEpochs.map((e, i) => `${esc(names[i])} ${fmt(e, 3)}`).join(", ")}<br>
      <span class="muted small">provenance: fixture:lecture_14--mixture-epochs · lecture_14.py:L339-L348 (mixture, uniform, proportional), L357-L365 (10T / 10B, 0.5/0.5, 1T → 0.05 and 50 epochs), L367-L371 (UniMax cap; α power), L383-L390 (0.1/0.9 small-scale optimum), L392-L401 (simulated epoching: ratio 0.01, 100B and 100M, 0.7/0.3); video 54:23-55:37, 1:07:24-1:09:36. The UniMax cap is applied in epochs, per the professor's gloss at L370; L371 writes it as p(s)·N ≤ C. Default sources and presets are the lecture's; the slider positions are not.</span>`;
  };
  ta.addEventListener("input", draw);
  const modeSel = el("select", {}, el("option", { value: "custom" }, "custom weights (third column)"), el("option", { value: "alpha" }, "p ∝ tokens^α (α = 0 uniform, 1 proportional)"), el("option", { value: "unimax" }, "UniMax: uniform with an epoch cap C"));
  modeSel.addEventListener("change", () => { s.mode = modeSel.value; draw(); });
  const preset = (txt) => { ta.value = txt; modeSel.value = s.mode = "custom"; draw(); };
  root.append(el("div", { class: "widget" }, el("div", { class: "controls" },
    el("div", { class: "small muted" }, "one source per line: name, tokens, weight"), ta,
    el("label", {}, "mixture ", modeSel),
    slider("α (p ∝ tokens^α mode)", 0, 1, s.alpha, 0.05, v => { s.alpha = v; draw(); }),
    slider("UniMax cap C (epochs)", 0.1, 50, s.cap, 0.1, v => { s.cap = v; draw(); }),
    slider("production tokens N", 9, 14, s.logN, 0.1, v => { s.logN = v; draw(); }, v => showTok(10 ** v)),
    slider("pilot run tokens", 7, 14, s.logPilot, 0.1, v => { s.logPilot = v; draw(); }, v => showTok(10 ** v)),
    el("div", { style: "margin-top:8px" }, button("L362: 0.5 / 0.5", () => preset("low 10T 0.5\nhigh 10B 0.5")), button("L389: 0.1 / 0.9", () => preset("low 10T 0.1\nhigh 10B 0.9")), button("L401: 0.7 / 0.3", () => preset("low 10T 0.7\nhigh 10B 0.3"))),
    read), pic)); draw();
};
