// Widgets for thread lecture_13 (CS336 L13, data I: Common Crawl, filtering, dataset history, licensing).
// Register as "fixture:<id>" -> (root, notice) => void. Each widget has a pure model in MODELS (no DOM) that
// tools/check_widgets.mjs tests against the KPs' stored answers.
// Sources: official/lectures/lecture_13.py (cited as L<n>), lectures/lecture_13/transcript.json (cited as video M:SS),
//          lectures/lecture_09/transcript.json for the four-epoch remark (cited as L9 video M:SS), and the KPs' own
//          prompts (marked "author"). No constant here is new.
import { el, fmt, slider } from "../widgets_lib.js";

export const MODELS = {};
export const WIDGETS = {};

const C = { ink: "var(--ink)", muted: "var(--muted)", rule: "var(--rule)", a: "var(--accent)", b: "var(--accent2)", ok: "var(--ok)", hi: "var(--hilite)" };
const rect = (x, y, w, h, fill, extra = "") => `<rect x="${x.toFixed(1)}" y="${y.toFixed(1)}" width="${Math.max(0, w).toFixed(1)}" height="${h}" style="fill:${fill}" ${extra}/>`;
const text = (x, y, s, opts = {}) => `<text x="${x.toFixed(1)}" y="${y.toFixed(1)}" style="fill:${opts.fill || C.ink};font:${opts.weight || ""} ${opts.size || 12}px var(--sans)" text-anchor="${opts.anchor || "start"}">${s}</text>`;
const line = (x1, y1, x2, y2, stroke, extra = "") => `<line x1="${x1.toFixed(1)}" y1="${y1.toFixed(1)}" x2="${x2.toFixed(1)}" y2="${y2.toFixed(1)}" style="stroke:${stroke};${extra}"/>`;
const svg = (w, h, body) => `<svg viewBox="0 0 ${w} ${h}" width="100%" style="max-width:${w}px;border:1px solid var(--rule);border-radius:6px;background:#fff" role="img">${body}</svg>`;
const T = 1e12;
const pct = (x) => `${fmt(100 * x, x < 0.1 ? 2 : 1)}%`;
const size = (n, unit) => unit === "tokens" ? (n >= T ? `${fmt(n / T, 2)}T tokens` : `${fmt(n / 1e9, 0)}B tokens`)
  : unit === "bytes" ? `${fmt(n / T, 1)} TB` : `${n >= 1e9 ? `${fmt(n / 1e9, 0)}B` : `${fmt(n / 1e6, 0)}M`} ${unit}`;
const buttons = (items) => el("div", { style: "margin-top:6px" }, ...items.map(([label, fn]) => el("button", { style: "margin:0 4px 4px 0;padding:3px 8px;font-size:12px", onclick: fn }, label)));
const setSlider = (wrap, v, f = (x) => x) => { wrap.querySelector("input").value = v; wrap.querySelector(".readout").textContent = f(v); };

// ================================================================ 1. the funnel ==
// Each dataset is the list of checkpoints the lecture sizes, in order, with the stages between them. A checkpoint with
// size null is one the lecture names but does not size: the widget draws it unsized and computes no fraction across it.
// Fractions are computed only between consecutive sized checkpoints in the same unit (one division each).
//   C4: L385-L396 (1.4T tokens of one April 2019 snapshot → 806 GB, 156B tokens); video 51:17.
//   RefinedWeb: L504-L511; video 1:02:24-1:02:47 ("They did some deduplication, and they had five trillion tokens. They released
//     about 600 billion of them"): 600B of 5T is a release share, not a filter's retention.
//   FineWeb: L513-L520 (95 dumps, stages, 15T); the pool is not sized.
//   DCLM: L539-L554 (DCLM-pool 240T, classifier, DCLM-baseline 3.8T); video 1:05:14-1:05:45 ("completely unfiltered", English,
//     rules, dedupe, model-based filtering, "you end up with something that's 1.4%"; the text's 3.8/240 is 1.6%).
//   SlimPajama: L498-L501 (RedPajama v1 1.2T, MinHashLSH dedup, 627B).
//   The Stack: L578-L584 (137M repos, 51B files, 5B unique, permissive licences, minhash + Jaccard, 3.1 TB of code).
const FUNNELS = {
  c4: { name: "C4 (T5)", kind: "rules", main: 0, pts: [
    { label: "one Common Crawl snapshot, April 2019", n: 1.4e12, unit: "tokens", ref: "L387" },
    { label: "C4 (806 GB of text)", n: 156e9, unit: "tokens", ref: "L396" }],
    stages: [["lines: end in punctuation, ≥ 5 words (L390)", "pages: ≥ 3 sentences (L391)", "no 'bad words' (L392)", "no '{', 'lorem ipsum', 'terms of use' (L393)", "langdetect English ≥ 0.99 (L394)"]],
    short: ["manual rules + langdetect"] },
  refinedweb: { name: "RefinedWeb", kind: "rules", main: null, release: 1, pts: [
    { label: "Common Crawl WARC", n: null, unit: "tokens", ref: "L508" },
    { label: "after Gopher rules + dedup", n: 5e12, unit: "tokens", ref: "L511; video 1:02:43" },
    { label: "publicly released", n: 600e9, unit: "tokens", ref: "L511; video 1:02:47" }],
    stages: [["trafilatura HTML→text from WARC (L508)", "Gopher rules, no ML filter 'to avoid biases' (L509)", "MinHash over 5-grams (L510)"], ["release a subset (not a filter)"]],
    short: ["trafilatura, Gopher rules, MinHash (pool not sized)", "release only (not a filter)"] },
  fineweb: { name: "FineWeb", kind: "rules", main: null, pts: [
    { label: "95 Common Crawl dumps", n: null, unit: "tokens", ref: "L515" },
    { label: "FineWeb", n: 15e12, unit: "tokens", ref: "L520" }],
    stages: [["URL filtering (L516)", "language ID, keep p(en) > 0.65 (L516)", "Gopher, C4 and more manual rules (L517)", "MinHash fuzzy dedup (L518)", "anonymise emails and IPs (L519)"]],
    short: ["URL filter, langID, rules, MinHash, PII (pool not sized)"] },
  dclm: { name: "DCLM-baseline", kind: "classifier", main: 0, pts: [
    { label: "DCLM-pool (unfiltered Common Crawl)", n: 240e12, unit: "tokens", ref: "L542; video 1:05:19" },
    { label: "DCLM-baseline", n: 3.8e12, unit: "tokens", ref: "L552" }],
    stages: [["English only, rules, dedup (video 1:05:39-1:05:45)", "fastText classifier: OpenHermes-2.5 + ELI5 vs RefinedWeb (L547-L554)"]],
    short: ["English, rules, dedup, fastText quality classifier"] },
  slimpajama: { name: "SlimPajama", kind: "dedup", main: 0, pts: [
    { label: "RedPajama v1", n: 1.2e12, unit: "tokens", ref: "L498, L500" },
    { label: "SlimPajama", n: 627e9, unit: "tokens", ref: "L501" }],
    stages: [["deduplication with MinHashLSH (L501)"]],
    short: ["MinHashLSH dedup"] },
  stack: { name: "The Stack", kind: "dedup", main: 1, pts: [
    { label: "repositories cloned (names from GitHub Archive)", n: 137e6, unit: "repos", ref: "L580-L581" },
    { label: "files in those repositories", n: 51e9, unit: "files", ref: "L581" },
    { label: "unique files", n: 5e9, unit: "files", ref: "L581 '5B unique!'" },
    { label: "The Stack (3.1 TB of code)", n: 3.1e12, unit: "bytes", ref: "L584" }],
    stages: [["git clone (L581)"], ["count distinct files (L581)"], ["keep permissive licences (MIT, Apache) via go-license-detector (L582)", "near-dedup with minhash + Jaccard (L583)"]],
    short: ["git clone (repos → files: units change)", "duplicates removed: only distinct files", "licence filter + near-dedup (units change: files → bytes)"] },
};
function funnel({ ds }) {
  const F = FUNNELS[ds], steps = [];
  for (let i = 0; i + 1 < F.pts.length; i++) {
    const a = F.pts[i], b = F.pts[i + 1], ok = a.n != null && b.n != null && a.unit === b.unit;
    steps.push(ok ? { keep: b.n / a.n, removedPer10: 10 * (1 - b.n / a.n), shrink: a.n / b.n } : { keep: NaN, removedPer10: NaN, shrink: NaN });
  }
  const m = F.main == null ? { keep: NaN, removedPer10: NaN, shrink: NaN } : steps[F.main];
  return { steps, keep: m.keep, removedPer10: m.removedPer10, shrink: m.shrink, releaseShare: F.release == null ? NaN : steps[F.release].keep };
}
MODELS["fixture:lecture_13--funnel"] = {
  fn: funnel,
  cases: [
    { args: { ds: "c4" }, pick: "keep", expect: 0.11, tol: 0.02, from: "lecture_13:rule-based-heuristic-filters:predict" },
    { args: { ds: "stack" }, pick: "removedPer10", expect: 9, tol: 0.01, from: "lecture_13:fuzzy-deduplication-methods:predict" },
    { args: { ds: "stack" }, pick: "shrink", expect: 10, tol: 0.03, from: "lecture_13:code-data-licensing-pipeline:predict" },
    { args: { ds: "c4" }, pick: "keep", expect: 0.1114, tol: 0.002 },        // 156B / 1.4T exactly
    { args: { ds: "dclm" }, pick: "keep", expect: 0.01583, tol: 0.002 },     // 3.8T / 240T (spoken: "1.4%")
    { args: { ds: "slimpajama" }, pick: "keep", expect: 0.5225, tol: 0.002 }, // 627B / 1.2T: dedup alone halves RedPajama
    { args: { ds: "refinedweb" }, pick: "releaseShare", expect: 0.12 },      // 600B of 5T is a release share, not a filter
    { args: { ds: "stack" }, pick: "keep", expect: 0.098, tol: 0.01 },       // 5B / 51B
  ],
};
WIDGETS["fixture:lecture_13--funnel"] = (root) => {
  const s = { ds: "slimpajama" };
  const pic = el("div"), read = el("div", { class: "readout" });
  const NOTE = {
    dclm: "Aloud the professor rounds this to \"1.4%\" (video 1:05:45); the text's 3.8T ÷ 240T is 1.6%.",
    refinedweb: "600B of 5T is how much RefinedWeb released (video 1:02:43-1:02:47), not what its filter kept: the pool before Gopher rules and dedup is not sized.",
    fineweb: "Only the output is sized: the lecture gives no pool size, so no retained fraction can be read off.",
    stack: "Repos → files → bytes change units, so only the files → unique files step is a fraction.",
  };
  const draw = () => {
    const F = FUNNELS[s.ds], m = funnel({ ds: s.ds }), W = 640, x0 = 20, span = 600, ROW = 62, top = 22;
    // bar widths are linear within a run of consecutive sized checkpoints in the same unit
    const ref = F.pts.map((p, i) => { let j = i; while (j > 0 && F.pts[j - 1].n != null && F.pts[j - 1].unit === p.unit) j--; return j; });
    let b = text(x0, 14, `${F.name}: bar length ∝ size, within one unit · dashed = named but not sized in the lecture`, { fill: C.muted, size: 11 });
    F.pts.forEach((p, i) => {
      const y = top + i * ROW;
      b += text(x0, y + 10, `${p.label}`, { size: 12, weight: "600" }) + text(x0 + span, y + 10, `${p.n == null ? "not sized" : size(p.n, p.unit)} · ${p.ref}`, { fill: C.muted, size: 11, anchor: "end" });
      if (p.n == null) b += `<rect x="${x0}" y="${y + 15}" width="${span}" height="16" style="fill:none;stroke:${C.muted};stroke-dasharray:4 3"/>`;
      else { const w = Math.max(2, span * p.n / F.pts[ref[i]].n); b += rect(x0, y + 15, w, 16, i === 0 || ref[i] === i ? C.rule : C.a); }
      if (i + 1 < F.pts.length) {
        const st = m.steps[i], sized = Number.isFinite(st.keep);
        b += text(x0 + 6, y + 48, `↓ ${F.short[i]}`, { fill: C.muted, size: 11 });
        b += text(x0 + span, y + 48, !sized ? "no fraction (not sized or units change)" : i === F.release ? `releases ${pct(st.keep)} · not a filter` : `keeps ${pct(st.keep)} · removes ${fmt(st.removedPer10, 1)} of every 10`, { fill: sized ? (i === F.main ? C.b : C.ink) : C.muted, size: 11, anchor: "end", weight: i === F.main ? "600" : "" });
      }
    });
    pic.innerHTML = svg(W, top + F.pts.length * ROW - 22, b);
    read.innerHTML = m.steps.map((st, i) => Number.isFinite(st.keep)
      ? `${F.pts[i].label} → ${F.pts[i + 1].label}: ${size(F.pts[i + 1].n, F.pts[i + 1].unit)} ÷ ${size(F.pts[i].n, F.pts[i].unit)} = <b>${pct(st.keep)}</b> ${i === F.release ? "released (a release choice, not a filter)" : `kept · ${fmt(st.removedPer10, 2)} of every 10 removed · ×${fmt(st.shrink, 1)} smaller`}`
      : `${F.pts[i].label} → ${F.pts[i + 1].label}: no fraction`).join("<br>") +
      `<br>stages the lecture lists (in order, none sized on its own):<ol style="margin:4px 0 4px 18px;padding:0">${F.stages.map((g, i) => g.map(x => `<li>${x}${F.stages.length > 1 ? ` <span class="muted small">(step ${i + 1})</span>` : ""}</li>`).join("")).join("")}</ol>` +
      (NOTE[s.ds] ? `<span class="muted">${NOTE[s.ds]}</span><br>` : "") +
      `<span class="muted small">provenance: fixture:lecture_13--funnel · lecture_13.py L385-L396 (C4), L504-L511 (RefinedWeb), L513-L520 (FineWeb), L539-L554 (DCLM), L498-L501 (SlimPajama), L578-L584 (The Stack); video 1:02:43-1:02:47, 1:05:19-1:05:45. The lecture sizes endpoints, never a single stage, so a stage's own share is not drawn.</span>`;
  };
  root.append(el("div", { class: "widget" }, el("div", { class: "controls" },
    buttons(Object.entries(FUNNELS).map(([k, F]) => [`${F.name} (${F.kind})`, () => { s.ds = k; draw(); }])), read), pic)); draw();
};

// ====================================================== 2. filter vs token budget ==
// A filter keeps a fraction f of a pool P: kept = P·f unique tokens. A training run of D tokens then needs e = D / (P·f) epochs.
// The epoch count at which a filter becomes too strict is the point: f* = D/P is the strictest filter that needs no repeats, and
// f₄ = D/(4P) the strictest that stays within four epochs.
//   Keep presets (one division each, see the funnel): DCLM 3.8T of 240T (L542, L552); Nemotron-CC's "remove 90% of data" for
//   FineWebEdu and DCLM (L561) = 10%; C4 156B of 1.4T (L387, L396); no filter = 1.
//   Pool preset: DCLM-pool 240T (L542). Budget presets: Llama 3 15T, Qwen3 36T (L574).
//   Four epochs: L9 video 25:42-25:53 ("up to four epochs with standard training recipes, you just don't get hurt at all. But
//   if you go past that point ... much worse than the projected scaling law if you had fresh data"). The size of the loss past
//   four epochs is lecture_09's effective-data widget; this widget only says which side of the line a filter lands on.
//   The 50T pool, 30T budget, 200B and 20T budgets are the KPs' own prompt numbers (author).
const KEEP = { dclm: 3.8e12 / 240e12, nemo: 0.10, c4: 156e9 / 1.4e12, none: 1 };
function filterBudget({ pool, keep, budget }) {
  const kept = pool * keep, epochs = budget / kept;
  return { kept, epochs, fStar: budget / pool, f4: budget / (4 * pool), regime: epochs <= 1 ? 0 : epochs <= 4 ? 1 : 2 };
}
MODELS["fixture:lecture_13--filter-budget"] = {
  fn: filterBudget,
  cases: [
    { args: { pool: 50 * T, keep: 0.1, budget: 20 * T }, pick: "epochs", expect: 4, from: "lecture_13:filter-aggressiveness-vs-tokens:check" },
    { args: { pool: 50 * T, keep: 0.1, budget: 20 * T }, pick: "kept", expect: 5 * T },          // ... 5T kept
    { args: { pool: 50 * T, keep: 0.1, budget: 20 * T }, pick: "regime", expect: 1 },            // ... exactly at the four-epoch edge
    { args: { pool: 240 * T, keep: KEEP.dclm, budget: 30 * T }, pick: "epochs", expect: 7.89, tol: 0.01 }, // predict: 30T over 3.8T
    { args: { pool: 240 * T, keep: KEEP.dclm, budget: 30 * T }, pick: "regime", expect: 2 },     // ... past four epochs
    { args: { pool: 240 * T, keep: KEEP.dclm, budget: 200e9 }, pick: "epochs", expect: 0.0526, tol: 0.01 }, // transfer: small team, one pass covers it
    { args: { pool: 240 * T, keep: KEEP.dclm, budget: 20 * T }, pick: "epochs", expect: 5.26, tol: 0.01 },  // transfer: frontier team, past four
    { args: { pool: 240 * T, keep: KEEP.dclm, budget: 20 * T }, pick: "f4", expect: 0.02083, tol: 0.01 },  // ... loosen to keep ≥ 2.1%
    { args: { pool: 240 * T, keep: KEEP.dclm, budget: 15 * T }, pick: "epochs", expect: 3.95, tol: 0.01 },  // Llama 3's 15T: just inside four
    { args: { pool: 240 * T, keep: KEEP.dclm, budget: 36 * T }, pick: "epochs", expect: 9.47, tol: 0.01 },  // Qwen3's 36T: past four
    { args: { pool: 1.4 * T, keep: KEEP.c4, budget: 0.1 * T }, pick: "kept", expect: 156e9 },              // C4 preset reproduces L396
  ],
};
WIDGETS["fixture:lecture_13--filter-budget"] = (root) => {
  const s = { logP: Math.log10(240), logf: -1, logD: Math.log10(15) };    // pool and budget in T tokens; f = keep fraction
  const pic = el("div"), read = el("div", { class: "readout" });
  const ZONES = ["≤ 1 epoch: every training token is new", "1-4 epochs: repeats, within the four epochs that \"just don't get hurt\" (L9 video 25:42)", "past 4 epochs: repeats are worth much less than fresh data (L9 video 25:53); loosen the filter"];
  const draw = () => {
    const P = 10 ** s.logP * T, f = 10 ** s.logf, D = 10 ** s.logD * T, m = filterBudget({ pool: P, keep: f, budget: D });
    const px = 60, py = 26, pw = 540, ph = 200, fx0 = -2.5, fx1 = 0, ey0 = -2, ey1 = 2;
    const X = lf => px + pw * (lf - fx0) / (fx1 - fx0), Y = le => py + ph * (1 - (Math.max(ey0, Math.min(ey1, le)) - ey0) / (ey1 - ey0));
    let b = text(px, 16, `epochs needed = budget ÷ (pool × kept fraction) · pool ${fmt(P / T, 1)}T · budget ${fmt(D / T, 2)}T`, { fill: C.muted, size: 11 });
    b += rect(px, Y(ey1), pw, Y(Math.log10(4)) - Y(ey1), C.b, 'opacity="0.10"') + rect(px, Y(Math.log10(4)), pw, Y(0) - Y(Math.log10(4)), C.hi, 'opacity="0.18"') + rect(px, Y(0), pw, Y(ey0) - Y(0), C.ok, 'opacity="0.10"');
    b += `<rect x="${px}" y="${py}" width="${pw}" height="${ph}" style="fill:none;stroke:${C.rule}"/>`;
    for (const e of [0.01, 0.1, 1, 4, 10, 100]) b += text(px - 5, Y(Math.log10(e)) + 4, `${e}`, { fill: C.muted, size: 10, anchor: "end" }) + line(px, Y(Math.log10(e)), px + pw, Y(Math.log10(e)), C.rule, "stroke-dasharray:2 3");
    for (const v of [0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1]) b += text(X(Math.log10(v)), py + ph + 13, `${fmt(100 * v, 1)}%`, { fill: C.muted, size: 10, anchor: "middle" });
    b += text(px + pw / 2, py + ph + 27, "fraction of the pool the filter keeps (log) · stricter ←", { fill: C.muted, size: 11, anchor: "middle" });
    b += `<text transform="translate(16 ${py + ph / 2}) rotate(-90)" style="fill:${C.muted};font:11px var(--sans)" text-anchor="middle">epochs (log)</text>`;
    const pts = []; for (let lf = fx0; lf <= fx1 + 1e-9; lf += 0.02) pts.push(`${X(lf).toFixed(1)},${Y(Math.log10(D / (P * 10 ** lf))).toFixed(1)}`);
    b += `<polyline points="${pts.join(" ")}" style="fill:none;stroke:${C.a};stroke-width:2.5"/>`;
    [["DCLM", KEEP.dclm], ["10%", KEEP.nemo], ["C4", KEEP.c4]].forEach(([n, v], i) => { const x = X(Math.log10(v)); b += line(x, py, x, py + ph, C.muted, "stroke-width:0.8") + text(x + (i === 2 ? 3 : -3), py + 11, n, { fill: C.muted, size: 10, anchor: i === 2 ? "start" : "end" }); });
    for (const [v, n] of [[m.fStar, "1 epoch"], [m.f4, "4 epochs"]]) if (Math.log10(v) > fx0 && v <= 1) { const x = X(Math.log10(v)); b += line(x, py, x, py + ph, C.b, "stroke-dasharray:4 3") + text(x + 3, py + ph - 6, n, { fill: C.b, size: 10 }); }
    b += `<circle cx="${X(s.logf).toFixed(1)}" cy="${Y(Math.log10(m.epochs)).toFixed(1)}" r="6" style="fill:${C.hi};stroke:${C.ink}"/>`;
    pic.innerHTML = svg(640, 262, b);
    const clipped = Math.log10(m.epochs) > ey1 || Math.log10(m.epochs) < ey0 ? " (off the plotted range)" : "";
    read.innerHTML = `kept = ${fmt(P / T, 1)}T × ${pct(f)} = <b>${size(m.kept, "tokens")}</b> unique · budget ${size(D, "tokens")} → <span class="big">${fmt(m.epochs, 2)} epochs</span>${clipped}<br>
      <b style="color:${m.regime === 0 ? "var(--ok)" : m.regime === 1 ? "var(--ink)" : "var(--bad)"}">${ZONES[m.regime]}</b><br>
      strictest filter with no repeats: keep ≥ ${pct(Math.min(1, m.fStar))}${m.fStar > 1 ? " (impossible: the budget exceeds the whole pool)" : ""} · strictest within four epochs: keep ≥ ${pct(Math.min(1, m.f4))}${m.f4 > 1 ? " (impossible)" : ""}<br>
      <span class="muted small">Only the epoch count is computed. A stricter filter's tokens are better per token, but the lecture gives no number for that, so the best filter is not computed: what moves is which side of the four-epoch line a given filter lands on as the budget grows. Nemotron-CC's 10% ("remove 90%", L561) and DCLM's own 3.8T of 240T (1.6%) are measured from different starting points.</span><br>
      <span class="muted small">provenance: fixture:lecture_13--filter-budget · lecture_13.py L387/L396 (C4), L542/L552 (DCLM), L561 (remove 90%), L574 (Llama 3 15T, Qwen3 36T); lecture_09 video 25:42-25:53 (four epochs).</span>`;
  };
  const f2T = v => `${fmt(10 ** v, 2)}T`, wP = slider("pool (tokens)", 0, Math.log10(300), s.logP, 0.01, v => { s.logP = v; draw(); }, f2T);
  const wf = slider("filter keeps", -2.5, 0, s.logf, 0.01, v => { s.logf = v; draw(); }, v => pct(10 ** v));
  const wD = slider("training budget (tokens)", -1, Math.log10(50), s.logD, 0.01, v => { s.logD = v; draw(); }, f2T);
  const setF = v => () => { s.logf = Math.log10(v); setSlider(wf, s.logf, x => pct(10 ** x)); draw(); };
  const setD = v => () => { s.logD = Math.log10(v); setSlider(wD, s.logD, f2T); draw(); };
  root.append(el("div", { class: "widget" }, el("div", { class: "controls" }, wP,
    buttons([["DCLM-pool 240T", () => { s.logP = Math.log10(240); setSlider(wP, s.logP, f2T); draw(); }]]), wf,
    buttons([["DCLM 1.6%", setF(KEEP.dclm)], ["\"remove 90%\": 10%", setF(KEEP.nemo)], ["C4 11.1%", setF(KEEP.c4)], ["no filter", setF(1)]]), wD,
    buttons([["Llama 3: 15T", setD(15)], ["Qwen3: 36T", setD(36)]]), read), pic)); draw();
};

// ================================================== 3. reported tokens count repeats ==
// Reported training tokens = Σ_i e_i·U_i over the components of a run (U_i unique tokens seen e_i times); unique = Σ_i U_i.
//   video 1:09:30-1:09:56: "Llama 3 was trained on 15 trillion tokens, Qwen3 ... 36 trillion. ... it's not clear how big these
//   unique tokens are ... some of it is repeated. Like if you do two epochs, that's twice the number of tokens".
//   L573-L574: Nemotron-CC 6.3T (HQ subset 1.1T); Llama 3 15T, Qwen3 36T.
//   The Σ e·U bookkeeping and the 6ND compute check are the KP's (author, reported-tokens-count-repeats filtering note);
//   C = 6ND with D = tokens processed is lecture_02's six-nd rule.
function repeatLedger({ U, e, N = 0, target = 0 }) {
  const unique = U.reduce((a, x, i) => a + (e[i] > 0 ? x : 0), 0), reported = U.reduce((a, x, i) => a + x * e[i], 0);
  return { unique, reported, inflation: reported / unique, flopsReported: 6 * N * reported, flopsUnique: 6 * N * unique,
    flopsRatio: reported / unique, epochsForTarget: target / unique };
}
MODELS["fixture:lecture_13--repeat-ledger"] = {
  fn: repeatLedger,
  cases: [
    { args: { U: [6.3 * T], e: [1], target: 15 * T }, pick: "epochsForTarget", expect: 2.4, tol: 0.02, from: "lecture_13:reported-tokens-count-repeats:predict" },
    { args: { U: [10 * T, 5 * T], e: [1, 4] }, pick: "unique", expect: 15 * T, from: "lecture_13:reported-tokens-count-repeats:check" },
    { args: { U: [10 * T, 5 * T], e: [1, 4] }, pick: "reported", expect: 30 * T },               // ... and the card's 30T
    { args: { U: [1 * T], e: [4], N: 7e9 }, pick: "flopsReported", expect: 1.68e23 },            // transfer: 6ND with D = 4T
    { args: { U: [1 * T], e: [4], N: 7e9 }, pick: "flopsUnique", expect: 4.2e22 },               // ... the colleague's D = 1T
    { args: { U: [1 * T], e: [4], N: 7e9 }, pick: "flopsRatio", expect: 4 },                     // ... underestimates by 4x
    { args: { U: [1 * T], e: [2] }, pick: "inflation", expect: 2 },                              // video 1:09:47: two epochs, twice the tokens
    { args: { U: [6.3 * T], e: [1], target: 36 * T }, pick: "epochsForTarget", expect: 5.71, tol: 0.01 }, // Qwen3's 36T from Nemotron-CC alone
    { args: { U: [1.1 * T], e: [1], target: 15 * T }, pick: "epochsForTarget", expect: 13.6, tol: 0.01 }, // the HQ subset alone
    { args: { U: [3 * T, 2 * T], e: [1, 1] }, pick: "inflation", expect: 1 },                    // edge: one pass, reported = unique
    { args: { U: [3 * T, 2 * T], e: [1, 0] }, pick: "unique", expect: 3 * T },                    // edge: a component never seen adds no unique tokens
  ],
};
WIDGETS["fixture:lecture_13--repeat-ledger"] = (root) => {
  const s = { U1: 3.8, e1: 2, U2: 1.1, e2: 1, NB: 8, D: 15 };            // T tokens, epochs, billions of params, target T tokens
  const pic = el("div"), read = el("div", { class: "readout" });
  const sci = x => x.toExponential(2).replace("e+", "e");
  const draw = () => {
    const U = [s.U1 * T, s.U2 * T], e = [s.e1, s.e2], m = repeatLedger({ U, e, N: s.NB * 1e9, target: s.D * T });
    const W = 640, x0 = 130, span = 480, scale = Math.max(m.reported, m.unique, s.D * T) * 1.02, X = n => x0 + span * n / scale;
    let b = text(10, 16, "each segment is one pass over a component · solid = first pass (unique) · light = a repeat", { fill: C.muted, size: 11 });
    const row = (y, label, segs) => { b += text(10, y + 15, label, { size: 12 }); let x = 0; for (const [n, col, op] of segs) { b += rect(X(x), y, span * n / scale - (span * n / scale > 3 ? 1.5 : 0), 22, col, `opacity="${op}"`); x += n; } return x; };
    const passes = (u, ep, col) => { const out = []; for (let k = 0; k < Math.ceil(ep - 1e-9); k++) out.push([u * Math.min(1, ep - k), col, k === 0 ? 1 : 0.35]); return out; };
    row(28, "reported", [...passes(U[0], e[0], C.a), ...passes(U[1], e[1], C.b)]);
    row(64, "unique", [[e[0] > 0 ? U[0] : 0, C.a, 1], [e[1] > 0 ? U[1] : 0, C.b, 1]]);
    b += text(X(m.reported) + 4 > x0 + span - 60 ? x0 + span : X(m.reported) + 4, 44, `${fmt(m.reported / T, 2)}T`, { size: 12, anchor: X(m.reported) + 4 > x0 + span - 60 ? "end" : "start", fill: C.ink });
    b += text(X(m.unique) + 4, 80, `${fmt(m.unique / T, 2)}T`, { size: 12 });
    const tx = X(s.D * T); b += line(tx, 22, tx, 94, C.hi, "stroke-width:2;stroke-dasharray:5 3") + text(Math.min(tx, x0 + span - 2), 106, `target run ${fmt(s.D, 1)}T`, { fill: "#b07d12", size: 11, anchor: "middle" });
    pic.innerHTML = svg(W, 114, b);
    read.innerHTML = `reported = ${fmt(s.U1, 2)}T × ${fmt(s.e1, 1)} + ${fmt(s.U2, 2)}T × ${fmt(s.e2, 1)} = <span class="big">${fmt(m.reported / T, 2)}T</span>unique = ${[[s.U1, s.e1], [s.U2, s.e2]].filter(([, ep]) => ep > 0).map(([u]) => `${fmt(u, 2)}T`).join(" + ") || "0"} = <b>${fmt(m.unique / T, 2)}T</b> · reported ÷ unique = <b>${fmt(m.inflation, 2)}×</b>${m.inflation > 1 + 1e-9 ? " (the count includes repeats)" : " (one pass: reported = unique)"}<br>
      a ${fmt(s.D, 1)}T-token run over these ${fmt(m.unique / T, 2)}T unique tokens needs <b>${fmt(m.epochsForTarget, 2)} epochs</b> on average${m.epochsForTarget > 4 ? " (past the four epochs lecture 9 says barely hurt)" : ""}<br>
      compute 6·N·D with N = ${fmt(s.NB, 0)}B: D = reported → ${sci(m.flopsReported)} FLOPs; D = unique → ${sci(m.flopsUnique)} FLOPs (${fmt(m.flopsRatio, 2)}× too low). D in 6ND counts tokens processed, so repeats count.<br>
      <span class="muted small">provenance: fixture:lecture_13--repeat-ledger · video 1:09:30-1:09:56 (reported counts include repeats; two epochs double the count); lecture_13.py L573-L574 (Nemotron-CC 6.3T, HQ 1.1T; Llama 3 15T, Qwen3 36T); lecture_09 video 25:42 (four epochs). The Σ e·U ledger and 6ND check are the KP's arithmetic.</span>`;
  };
  const fT = v => `${fmt(v, 2)}T`;
  const w1 = slider("component A: unique tokens", 0.1, 20, s.U1, 0.1, v => { s.U1 = v; draw(); }, fT);
  const wD = slider("target run (reported tokens)", 1, 40, s.D, 0.5, v => { s.D = v; draw(); }, fT);
  root.append(el("div", { class: "widget" }, el("div", { class: "controls" },
    w1, buttons([["Nemotron-CC 6.3T", () => { s.U1 = 6.3; setSlider(w1, 6.3, fT); draw(); }], ["its HQ subset 1.1T", () => { s.U1 = 1.1; setSlider(w1, 1.1, fT); draw(); }], ["DCLM-baseline 3.8T", () => { s.U1 = 3.8; setSlider(w1, 3.8, fT); draw(); }]]),
    slider("component A: epochs", 0.1, 10, s.e1, 0.1, v => { s.e1 = v; draw(); }),
    slider("component B: unique tokens", 0, 20, s.U2, 0.1, v => { s.U2 = v; draw(); }, fT),
    slider("component B: epochs", 0, 10, s.e2, 0.1, v => { s.e2 = v; draw(); }),
    slider("parameters N (billions, for 6ND)", 1, 400, s.NB, 1, v => { s.NB = v; draw(); }),
    wD, buttons([["Llama 3: 15T", () => { s.D = 15; setSlider(wD, 15, fT); draw(); }], ["Qwen3: 36T", () => { s.D = 36; setSlider(wD, 36, fT); draw(); }]]),
    read), pic)); draw();
};
