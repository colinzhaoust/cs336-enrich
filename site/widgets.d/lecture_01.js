// Widgets for thread lecture_01 (CS336 L1, overview and tokenization). Register as "fixture:<id>" -> (root, notice) => void.
// Each widget has a pure model in MODELS (no DOM) that tools/check_widgets.mjs tests against the KPs' stored answers.
// Sources: official/lectures/lecture_01.py (cited as file:line), its trace official/lectures/var/traces/lecture_01.json
// (@inspect values), and lectures/lecture_01/transcript.json (cited as video M:SS). No constant here is new: the tokenizers
// are the lecture's own code re-implemented line for line, and the only stored output is the GPT-5 trace for one string.
import { el, fmt, slider } from "../widgets_lib.js";

export const MODELS = {};
export const WIDGETS = {};

const C = { ink: "var(--ink)", muted: "var(--muted)", rule: "var(--rule)", a: "var(--accent)", b: "var(--accent2)", ok: "var(--ok)", hi: "var(--hilite)" };
const esc = s => String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");
const rect = (x, y, w, h, fill, extra = "") => `<rect x="${x.toFixed(1)}" y="${y}" width="${Math.max(0, w).toFixed(1)}" height="${h}" style="fill:${fill}" ${extra}/>`;
const text = (x, y, s, opts = {}) => `<text x="${x.toFixed(1)}" y="${y}" style="fill:${opts.fill || C.ink};font:${opts.size || 12}px ${opts.mono ? "var(--mono)" : "var(--sans)"}" text-anchor="${opts.anchor || "start"}">${s}</text>`;
const svg = (w, h, body) => `<svg viewBox="0 0 ${w} ${h}" width="100%" style="max-width:${w}px;border:1px solid var(--rule);border-radius:6px;background:#fff" role="img">${body}</svg>`;
const check = (label, value, onChange) => { const i = el("input", { type: "checkbox" }); i.checked = value; i.addEventListener("change", () => onChange(i.checked)); return el("label", {}, i, ` ${label}`); };
const buttons = (opts, onPick) => el("div", { style: "margin-top:6px;display:flex;flex-wrap:wrap;gap:4px" }, ...opts.map(([lab, v]) => el("button", { onclick: () => onPick(v) }, lab)));
const ENC = new TextEncoder();
const utf8 = s => [...ENC.encode(s)];
// label for a token's bytes: the text if it is valid UTF-8 (space shown as ␣), else hex
const DEC = new TextDecoder("utf-8", { fatal: true });
const show = bytes => { try { return DEC.decode(new Uint8Array(bytes)).replace(/ /g, "␣").replace(/\n/g, "⏎"); } catch { return bytes.map(x => x.toString(16).padStart(2, "0")).join(" "); } };

// ======================================================== 1. tokenizer lens ==
// The lecture's three hand-built tokenizers on one string, on a shared byte axis.
//   compression ratio = len(string.encode("utf-8")) / len(indices)        lecture_01.py:L567-L571
//   character tokenizer: list(map(ord, string)); vocab lower bound max+1   L505-L511, L644 (127,758 on the lecture string)
//   byte tokenizer: list(string.encode("utf-8")), vocab 256, ratio == 1    L514-L524, L668, L671
//   word tokenizer: regex.findall(r"\w+|.", string)                        L680 (5.5 on L678's string, trace at L689)
//   GPT-5 (tiktoken o200k_base): cannot run in the page; the trace stores its output for the lecture string only:
//   [13225, 11, 130321, 235, 0, 220, 177519, 0] (trace at L604; video 1:07:25-1:07:44, "20 divided by 8 ... 2.5").
// \w follows the Python `regex` module's Unicode definition (letters, marks, digits, connector punctuation);
// "." matches any character but a newline, as in Python.
const LECTURE_STRING = "Hello, 🌍! 你好!";                                       // lecture_01.py:L581, L601, L638, L662
const GPT5_STORED = { [LECTURE_STRING]: [13225, 11, 130321, 235, 0, 220, 177519, 0] };
const WORD_RE = /[\p{Alphabetic}\p{M}\p{Nd}\p{Pc}\p{Join_Control}]+|./gu;
function tokenizerLens({ text: s }) {
  const chars = Array.from(s), charBytes = chars.map(c => ENC.encode(c).length);
  const bytes = charBytes.reduce((a, b) => a + b, 0), cps = chars.map(c => c.codePointAt(0));
  const words = s.match(WORD_RE) || [], g = GPT5_STORED[s];
  return {
    chars: chars.length, bytes, charBytes,
    byteTokens: bytes, byteRatio: bytes / bytes, byteVocab: 256,
    charTokens: chars.length, charRatio: bytes / chars.length, charVocab: cps.length ? Math.max(...cps) + 1 : NaN,
    words, wordTokens: words.length, wordRatio: bytes / words.length, wordVocab: new Set(words).size,
    gpt5Ids: g || null, gpt5Tokens: g ? g.length : NaN, gpt5Ratio: g ? bytes / g.length : NaN, gpt5Vocab: 200019, // n_vocab, L612
  };
}
MODELS["fixture:lecture_01--tokenizer-lens"] = {
  fn: tokenizerLens,
  cases: [
    { args: { text: LECTURE_STRING }, pick: "gpt5Ratio", expect: 2.5, tol: 0.02, from: "lecture_01:compression-ratio:predict" },
    { args: { text: LECTURE_STRING }, pick: "bytes", expect: 20, from: "lecture_01:utf8-variable-length:predict" },
    { args: { text: LECTURE_STRING }, pick: "chars", expect: 13 },                       // 13 characters, 20 bytes
    { args: { text: LECTURE_STRING }, pick: "charRatio", expect: 20 / 13 },              // L647 trace 1.538
    { args: { text: LECTURE_STRING }, pick: "charVocab", expect: 127758 },               // L644 trace, ord('🌍') + 1
    { args: { text: LECTURE_STRING }, pick: "byteRatio", expect: 1 },                    // L671 assert
    { args: { text: "你好" }, pick: "charRatio", expect: 3, tol: 0.02, from: "lecture_01:character-tokenizer:predict" },
    { args: { text: "Hello, world!" }, pick: "charVocab", expect: 120, from: "lecture_01:character-tokenizer:transfer" },
    { args: { text: "Hello, world!" }, pick: "charRatio", expect: 1 },                   // ASCII only: characters = bytes
    { args: { text: "héllo" }, pick: "byteTokens", expect: 6, from: "lecture_01:utf8-variable-length:transfer" },
    { args: { text: "中文文档，每个汉字三个字节。" }, pick: "byteRatio", expect: 1, tol: 0.001, from: "lecture_01:byte-tokenizer:predict" },
    { args: { text: "I'll say supercalifragilisticexpialidocious!" }, pick: "wordTokens", expect: 8, from: "lecture_01:word-tokenizer:predict" },
    { args: { text: "I'll say supercalifragilisticexpialidocious!" }, pick: "wordRatio", expect: 5.5 }, // L689 trace
    { args: { text: "I'll say supercalifragilisticexpialidocious!" }, pick: "bytes", expect: 44 },
    { args: { text: "the cat sat on the mat." }, pick: "wordVocab", expect: 7 },        // word-tokenizer check: 7 distinct chunks
  ],
};
WIDGETS["fixture:lecture_01--tokenizer-lens"] = (root) => {
  const s = { text: "the cat in the hat" };
  const pic = el("div"), read = el("div", { class: "readout" });
  const ta = el("textarea", { rows: 2, style: "width:100%;font:inherit;padding:6px" }); ta.value = s.text;
  ta.addEventListener("input", () => { s.text = ta.value; draw(); });
  const draw = () => {
    const m = tokenizerLens(s), maxB = 160, nB = Math.min(m.bytes, maxB);
    const W = 640, x0 = 112, span = 520, u = nB ? Math.min(34, span / nB) : 34, lab = u >= 9;
    const bcol = { 1: C.rule, 2: C.hi, 3: C.b, 4: C.a };
    let b = text(10, 16, `one box per token, drawn on the string's UTF-8 bytes · ${m.chars} characters, ${m.bytes} bytes`, { fill: C.muted, size: 11 });
    const rows = [["UTF-8 bytes", 28], ["characters", 64], ["word chunks", 100]];
    rows.forEach(([n, y]) => { b += text(10, y + 16, n, { size: 12 }); });
    // byte row: one box per byte; colour = the byte length of the character it belongs to
    let bi = 0;
    Array.from(s.text).forEach((ch, ci) => {
      const nb = m.charBytes[ci], bs = utf8(ch);
      bs.forEach(v => {
        if (bi < maxB) { b += rect(x0 + bi * u, 28, u - 1.5, 24, bcol[nb] || C.a, nb > 1 ? 'opacity="0.55"' : ""); if (u >= 20) b += text(x0 + bi * u + (u - 1.5) / 2, 44, v.toString(16).padStart(2, "0"), { size: 10, anchor: "middle", mono: true }); }
        bi++;
      });
      // character row: one box spanning the character's bytes
      const start = bi - nb;
      if (start < maxB) {
        b += rect(x0 + start * u, 64, Math.min(nb, maxB - start) * u - 1.5, 24, bcol[nb] || C.a, `opacity="${nb > 1 ? 0.85 : 0.5}"`);
        if (lab) b += text(x0 + (start + nb / 2) * u - 0.75, 80, esc(ch === " " ? "␣" : ch), { size: 12, anchor: "middle" });
      }
    });
    // word row: chunk boxes from \w+|.
    let wb = 0;
    m.words.forEach((w, i) => {
      const n = ENC.encode(w).length;
      if (wb < maxB) {
        b += rect(x0 + wb * u, 100, Math.min(n, maxB - wb) * u - 1.5, 24, i % 2 ? C.ok : C.a, 'opacity="0.35"');
        if (n * u > 10) b += text(x0 + (wb + n / 2) * u - 0.75, 116, esc(Array.from(w.replace(/ /g, "␣")).slice(0, Math.max(1, Math.floor(n * u / 7))).join("")), { size: 11, anchor: "middle" });
      }
      wb += n;
    });
    if (m.bytes > maxB) b += text(W - 8, 140, `(first ${maxB} of ${m.bytes} bytes drawn)`, { fill: C.muted, size: 10, anchor: "end" });
    b += text(10, 148, "byte colour = length of its character in UTF-8: grey 1 · yellow 2 · orange 3 · blue 4", { fill: C.muted, size: 11 });
    pic.innerHTML = svg(W, 158, b);
    const g = m.gpt5Ids
      ? `GPT-5 (o200k)  ${String(m.gpt5Tokens).padStart(5)} tokens   ratio <b>${fmt(m.gpt5Ratio, 3)}</b>   vocab 200,019 · stored trace [${m.gpt5Ids.join(", ")}]`
      : `GPT-5 (o200k)      –           (tiktoken cannot run here; the trace stores only the lecture string)`;
    read.innerHTML = `<pre style="white-space:pre-wrap">tokenizer        tokens        bytes/token   vocabulary
byte           ${String(m.byteTokens).padStart(7)}        <b>${fmt(m.byteRatio, 3).toString().padEnd(6)}</b>       256 (fixed)
character      ${String(m.charTokens).padStart(7)}        <b>${fmt(m.charRatio, 3).toString().padEnd(6)}</b>       ≥ ${fmt(m.charVocab, 0)} (max code point + 1)
word \\w+|.     ${String(m.wordTokens).padStart(7)}        <b>${fmt(m.wordRatio, 3).toString().padEnd(6)}</b>       ${m.wordVocab} distinct chunks here; unbounded in general
${g}</pre>
      ${m.bytes > m.chars ? `bytes − characters = ${m.bytes - m.chars}: multi-byte characters raise every bytes-per-token ratio, including the character tokenizer's` : "ASCII only: characters = bytes, so the character tokenizer's ratio is exactly 1"}<br>
      <span class="muted small">provenance: fixture:lecture_01--tokenizer-lens · lecture_01.py:L567-L571 (ratio = UTF-8 bytes / tokens), L505-L511 and L644 (ord; vocab ≥ max + 1), L514-L524 and L671 (bytes; ratio 1), L680 (regex \\w+|.), trace at L604 (GPT-5 ids for the lecture string); video 1:07:25-1:07:44. The word tokenizer's real vocabulary is every distinct chunk of the training data (L688), not of this string.</span>`;
  };
  root.append(el("div", { class: "widget" }, el("div", { class: "controls" }, ta,
    buttons([["lecture string", LECTURE_STRING], ["你好", "你好"], ["Hello, world!", "Hello, world!"], ["word example (L678)", "I'll say supercalifragilisticexpialidocious!"], ["the cat in the hat", "the cat in the hat"]],
      v => { s.text = v; ta.value = v; draw(); }), read), pic)); draw();
};

// ============================================================ 2. BPE lab ==
// The lecture's train_bpe and BPETokenizer.encode, line for line:
//   indices = list(string.encode("utf-8")); counts = count_adjacent_pairs(indices)   L729-L736, L753-L757
//   pair = max(counts, key=counts.get)  -> first pair in insertion order on a tie   L740 (video 1:13:57 "we'll just take the first one")
//   new_index = 256 + i; vocab[new] = vocab[a] + vocab[b]; indices = merge(...)       L743-L746
//   merge(): left to right, non-overlapping, replaces every occurrence                L527-L538
//   encode(): bytes, then every merge in training (insertion) order                   L554-L559
// The lecture stops at num_merges = 3 on "the cat in the hat" (L711-L712): 18 bytes -> 12 tokens, ratio 1.5 (video 1:15:07).
// Python's max() would raise on an empty counts dict (one token left); here training simply stops.
const BPE_PARAGRAPH = "Basic idea: train the tokenizer on raw text to construct a vocabulary tailored to the data. Intuition: common sequences of bytes are represented by a single token, rare sequences are represented by many tokens. Sketch: start with each byte as a token, and successively merge the most common pair of adjacent tokens."; // lecture_01.py:L705-L708 text() lines, markdown removed
function mergeIds(ids, a, b, n) {
  const out = [];
  for (let i = 0; i < ids.length;) { if (i + 1 < ids.length && ids[i] === a && ids[i + 1] === b) { out.push(n); i += 2; } else { out.push(ids[i]); i += 1; } }
  return out;
}
function trainBpe(str, k) {
  let ids = utf8(str); const vocab = new Map(); for (let x = 0; x < 256; x++) vocab.set(x, [x]);
  const merges = [], curve = [ids.length];
  for (let i = 0; i < k; i++) {
    const counts = new Map();
    for (let j = 0; j + 1 < ids.length; j++) { const key = ids[j] * 65536 + ids[j + 1]; counts.set(key, (counts.get(key) || 0) + 1); }
    if (!counts.size) break;
    let best = -1, bc = -1; for (const [key, v] of counts) if (v > bc) { best = key; bc = v; }   // strict > keeps the first on ties
    const a = Math.floor(best / 65536), bb = best % 65536, id = 256 + i, before = ids.length;
    vocab.set(id, [...vocab.get(a), ...vocab.get(bb)]);
    ids = mergeIds(ids, a, bb, id);
    merges.push({ a, b: bb, id, count: bc, saved: before - ids.length });
    curve.push(ids.length);
  }
  return { ids, merges, vocab, curve };
}
function encodeBpe(str, merges) {
  let ids = utf8(str); const curve = [ids.length], fired = [];
  for (const m of merges) { const n = ids.length; ids = mergeIds(ids, m.a, m.b, m.id); curve.push(ids.length); if (ids.length < n) fired.push(m.id); }
  return { ids, curve, fired };
}
function bpeLab({ train, merges, encode = "" }) {
  const t = trainBpe(train, merges), e = encodeBpe(encode, t.merges), tb = utf8(train).length, eb = utf8(encode).length;
  const last = t.merges[t.merges.length - 1];
  return {
    trainBytes: tb, trainTokens: t.ids.length, trainRatio: tb / t.ids.length, mergesDone: t.merges.length, vocabSize: 256 + t.merges.length,
    lastCount: last ? last.count : NaN, lastSaved: last ? last.saved : NaN,
    encBytes: eb, encTokens: e.ids.length, encRatio: eb / e.ids.length, fired: e.fired.length, unused: t.merges.length - e.fired.length,
    t, e,
  };
}
MODELS["fixture:lecture_01--bpe-lab"] = {
  fn: bpeLab,
  cases: [
    { args: { train: "the cat in the hat", merges: 1 }, pick: "trainTokens", expect: 16, from: "lecture_01:bpe-merge-step:predict" },
    { args: { train: "the cat in the hat", merges: 1 }, pick: "vocabSize", expect: 257 },                 // predict: 257 entries
    { args: { train: "the cat in the hat", merges: 3 }, pick: "trainTokens", expect: 12 },                // L746 trace
    { args: { train: "the cat in the hat", merges: 3 }, pick: "trainRatio", expect: 1.5 },                // L748 trace, video 1:15:07
    { args: { train: "the cat in the hat", merges: 4 }, pick: "trainTokens", expect: 10 },                // bpe-merge-step transfer ('at')
    { args: { train: "aaabdaaabac", merges: 3 }, pick: "trainTokens", expect: 5, from: "lecture_01:bpe-merge-step:check" },
    { args: { train: "aaabdaaabac", merges: 1 }, pick: "lastCount", expect: 4 },                          // (a,a) counted with overlaps ...
    { args: { train: "aaabdaaabac", merges: 1 }, pick: "lastSaved", expect: 2 },                          // ... but merged non-overlapping
    { args: { train: "the cat in the hat", merges: 3, encode: "the quick brown fox" }, pick: "encTokens", expect: 16, from: "lecture_01:bpe-encode-apply-merges:predict" },
    { args: { train: "the cat in the hat", merges: 3, encode: "hat the" }, pick: "encTokens", expect: 5, from: "lecture_01:bpe-encode-apply-merges:transfer" },
    { args: { train: "the cat in the hat", merges: 3, encode: "hat the" }, pick: "fired", expect: 2 },     // 'the ' (258) does not fire
    { args: { train: "the cat in the hat", merges: 3, encode: "bathe the" }, pick: "encTokens", expect: 4, from: "lecture_01:bpe-encode-apply-merges:check" },
    { args: { train: "the cat in the hat", merges: 3, encode: "你好" }, pick: "encTokens", expect: 6, from: "lecture_01:bpe-idea:check" },
    { args: { train: "the cat in the hat", merges: 3, encode: "你好" }, pick: "unused", expect: 3 },        // no merge fires: all 3 entries idle
    { args: { train: "the cat in the hat", merges: 40 }, pick: "trainTokens", expect: 1 },                // edge: training runs out of pairs
    { args: { train: "the cat in the hat", merges: 40 }, pick: "mergesDone", expect: 13 },
  ],
};
WIDGETS["fixture:lecture_01--bpe-lab"] = (root) => {
  const s = { train: "the cat in the hat", k: 2, encode: "the quick brown fox" }, KMAX = 80;
  const pic = el("div"), read = el("div", { class: "readout" });
  const taT = el("textarea", { rows: 2, style: "width:100%;font:inherit;padding:6px" }); taT.value = s.train;
  const taE = el("input", { type: "text", style: "width:100%;font:inherit;padding:4px" }); taE.value = s.encode;
  taT.addEventListener("input", () => { s.train = taT.value; draw(); });
  taE.addEventListener("input", () => { s.encode = taE.value; draw(); });
  const box = (id, vocab, hot) => `<span style="display:inline-block;margin:1px;padding:1px 4px;border-radius:3px;font:12px var(--mono);border:1px solid var(--rule);background:${id >= 256 ? (hot ? "var(--hilite)" : "var(--accent)") : "#fff"};color:${id >= 256 && !hot ? "#fff" : "var(--ink)"}" title="id ${id}">${esc(show(vocab.get(id)))}<sub style="opacity:.7">${id}</sub></span>`;
  const draw = () => {
    const m = bpeLab({ train: s.train, merges: s.k, encode: s.encode });
    const full = trainBpe(s.train, KMAX), encFull = encodeBpe(s.encode, full.merges), V = m.t.vocab;
    // curve: tokens vs vocabulary size, training text (solid) and the encoded text (dashed), both as a share of their bytes
    const W = 640, H = 184, px = 46, py = 26, pw = 570, ph = 120, kTop = Math.max(1, full.merges.length);
    const X = k => px + pw * k / kTop, Y = f => py + ph * (1 - f);
    let b = `<line x1="${px}" y1="${py + ph}" x2="${px + pw}" y2="${py + ph}" style="stroke:${C.rule}"/><line x1="${px}" y1="${py}" x2="${px}" y2="${py + ph}" style="stroke:${C.rule}"/>`;
    const path = (curve, n, extra) => { if (!n) return ""; let d = ""; curve.forEach((v, k) => { d += `${d ? "L" : "M"}${X(k).toFixed(1)},${Y(v / n).toFixed(1)}`; }); return `<path d="${d}" style="fill:none;${extra}"/>`; };
    b += path(full.curve, m.trainBytes, `stroke:${C.a};stroke-width:2`);
    b += path(encFull.curve, m.encBytes, `stroke:${C.b};stroke-width:2;stroke-dasharray:5 3`);
    const kk = m.mergesDone;
    b += `<line x1="${X(kk).toFixed(1)}" y1="${py}" x2="${X(kk).toFixed(1)}" y2="${py + ph}" style="stroke:${C.muted};stroke-dasharray:2 3"/>`;
    if (m.trainBytes) b += `<circle cx="${X(kk).toFixed(1)}" cy="${Y(m.trainTokens / m.trainBytes).toFixed(1)}" r="4.5" style="fill:${C.hi};stroke:${C.ink}"/>`;
    for (const f of [0, 0.5, 1]) b += text(px - 6, Y(f) + 4, f.toFixed(1), { fill: C.muted, size: 10, anchor: "end" });
    for (const k of [0, Math.round(kTop / 2), kTop]) b += text(X(k), py + ph + 14, `${256 + k}`, { fill: C.muted, size: 10, anchor: "middle" });
    b += text(px, 15, "y: tokens ÷ bytes  · solid: training text  · dashed: new text  · yellow: num_merges", { fill: C.muted, size: 11 });
    b += text(px + pw / 2, H - 6, "vocabulary size (256 bytes + one entry per merge)", { fill: C.muted, size: 11, anchor: "middle" });
    pic.innerHTML = svg(W, H, b) +
      `<div style="margin-top:8px;font-size:12px;color:var(--muted)">training sequence after ${kk} merge${kk === 1 ? "" : "s"} (${m.trainTokens} tokens; blue = merged token, subscript = id)</div><div>${m.t.ids.slice(0, 240).map(id => box(id, V, false)).join("")}${m.t.ids.length > 240 ? " …" : ""}</div>` +
      `<div style="margin-top:8px;font-size:12px;color:var(--muted)">new text encoded with these merges, applied in training order (${m.encTokens} tokens; yellow = merged)</div><div>${m.e.ids.slice(0, 240).map(id => box(id, V, true)).join("")}${m.e.ids.length > 240 ? " …" : ""}</div>`;
    const lines = m.t.merges.map(q => `${String(q.id).padEnd(4)} (${q.a},${q.b})  ${show(V.get(q.a))} + ${show(V.get(q.b))} → ${show(V.get(q.id))}   count ${q.count}, saved ${q.saved}${m.e.fired.includes(q.id) ? "" : "   · unused on new text"}`);
    const stopped = m.mergesDone < s.k ? ` <b>training stopped after ${m.mergesDone}: one token left, no pairs to count</b>` : "";
    read.innerHTML = `training: ${m.trainBytes} bytes → <b>${m.trainTokens}</b> tokens, ratio ${fmt(m.trainRatio, 3)} · vocabulary <b>${m.vocabSize}</b>${stopped}<br>
      new text: ${m.encBytes} bytes → <span class="big">${m.encTokens} tokens</span> ratio ${fmt(m.encRatio, 3)} · ${m.fired} of ${m.mergesDone} merges fired, ${m.unused} vocabulary entries unused<br>
      ${m.mergesDone ? `last merge: count ${m.lastCount}, saved ${m.lastSaved} tokens${m.lastCount === 1 ? " (a count of 1: the merge only memorises the training text)" : ""}` : "no merges yet: every byte is its own token"}<br>
      <pre style="max-height:150px;overflow:auto;font-size:12px">${esc(lines.join("\n")) || "(no merges)"}</pre>
      <span class="muted small">provenance: fixture:lecture_01--bpe-lab · lecture_01.py:L729-L750 (train_bpe), L753-L757 (count_adjacent_pairs), L740 (max: first pair on a tie; video 1:13:57), L527-L538 (merge: every occurrence, left to right), L554-L559 (encode: merges in training order). The long preset is the lecture's own text at L705-L708. Counts overlap ('aaa' counts (a,a) twice), merges do not.</span>`;
  };
  root.append(el("div", { class: "widget" }, el("div", { class: "controls" },
    el("div", { class: "small muted" }, "training text"), taT,
    buttons([["the cat in the hat (L711)", "the cat in the hat"], ["aaabdaaabac", "aaabdaaabac"], ["lecture text L705-L708", BPE_PARAGRAPH]], v => { s.train = v; taT.value = v; draw(); }),
    slider("num_merges", 0, KMAX, s.k, 1, v => { s.k = v; draw(); }),
    el("div", { class: "small muted", style: "margin-top:8px" }, "new text to encode"), taE,
    buttons([["the quick brown fox (L717)", "the quick brown fox"], ["hat the", "hat the"], ["bathe the", "bathe the"], ["你好", "你好"]], v => { s.encode = v; taE.value = v; draw(); }),
    read), pic)); draw();
};

// ======================================================= 3. vocabulary trade ==
// Two tokenizers on the same document. Sequence length = bytes / (bytes per token) (L567-L571), attention cost is
// quadratic in it and the linear layers are linear in it (L610 "attention is quadratic in sequence length", L673;
// video 1:07:46). Bigger vocabulary raises the ratio "leading to sparsity" (L611) and costs V·d parameters in the
// input embedding, plus V·d in an untied output projection (supp-vocab-embedding-params, author: standard Transformer).
// Presets: bytes (ratio 1, L671; vocab 256, L668); GPT-5 on the lecture string (ratio 2.5, trace L609; vocab 200,019, L612);
// "1000 bytes → ~250 tokens" (L276). The page does not invent a ratio for a given vocabulary size: the learner sets both.
function vocabTrade({ bytes, r1, r2, V1, V2, d, N, tied = false }) {
  const tokens1 = bytes / r1, tokens2 = bytes / r2, lenRatio = tokens2 / tokens1, k = tied ? 1 : 2;
  const p1 = k * V1 * d, p2 = k * V2 * d;
  return { tokens1, tokens2, lenRatio, attnRatio: lenRatio ** 2, attnRatioInv: 1 / lenRatio ** 2, linRatio: lenRatio,
    embed1: V1 * d, embed2: V2 * d, vocabParams1: p1, vocabParams2: p2, share1: p1 / N, share2: p2 / N, extraParams: p2 - p1 };
}
MODELS["fixture:lecture_01--vocab-trade"] = {
  fn: vocabTrade,
  cases: [
    { args: { bytes: 1000, r1: 1, r2: 4, V1: 256, V2: 200019, d: 4096, N: 7e9 }, pick: "attnRatioInv", expect: 16, from: "lecture_01:tokenization-efficiency-lens:predict" },
    { args: { bytes: 1000, r1: 1, r2: 4, V1: 256, V2: 200019, d: 4096, N: 7e9 }, pick: "tokens2", expect: 250 },  // L276
    { args: { bytes: 1000, r1: 3, r2: 4.8, V1: 32000, V2: 64000, d: 4096, N: 7e9 }, pick: "attnRatio", expect: 0.390625, tol: 0.03, from: "lecture_01:tokenization-efficiency-lens:check" },
    { args: { bytes: 8192, r1: 4, r2: 1, V1: 50000, V2: 256, d: 4096, N: 7e9 }, pick: "tokens2", expect: 8192 },    // lens transfer: 4× longer
    { args: { bytes: 1000, r1: 3.6, r2: 4.0, V1: 32000, V2: 64000, d: 4096, N: 7e9 }, pick: "lenRatio", expect: 0.9, from: "lecture_01:vocab-size-vs-sequence-length:check" },
    { args: { bytes: 1000, r1: 4.0, r2: 4.3, V1: 50000, V2: 100000, d: 4096, N: 7e9 }, pick: "lenRatio", expect: 0.9302 }, // vocab transfer, ≈ 7% shorter
    { args: { bytes: 1000, r1: 4.0, r2: 2.0, V1: 50000, V2: 50000, d: 4096, N: 7e9 }, pick: "tokens2", expect: 500 },  // compression-ratio transfer
    { args: { bytes: 1000, r1: 1, r2: 2.5, V1: 256, V2: 200019, d: 4096, N: 7e9 }, pick: "embed1", expect: 1048576 },    // supp predict, bytes
    { args: { bytes: 1000, r1: 1, r2: 2.5, V1: 256, V2: 200019, d: 4096, N: 7e9 }, pick: "embed2", expect: 819277824 },  // supp predict, GPT-5
    { args: { bytes: 1000, r1: 1, r2: 2.5, V1: 256, V2: 128000, d: 4096, N: 7e9 }, pick: "share2", expect: 0.1498, tol: 0.05, from: "lecture_01:supp-vocab-embedding-params:check" },
    { args: { bytes: 1000, r1: 1, r2: 2.5, V1: 256, V2: 256000, d: 2048, N: 1e9 }, pick: "share2", expect: 1.048576 },  // supp transfer: all of a "1B" model
    { args: { bytes: 1000, r1: 1, r2: 2.5, V1: 256, V2: 256000, d: 8192, N: 70e9 }, pick: "share2", expect: 0.0599, tol: 0.02 }, // ... about 6% at 70B
  ],
};
const VSTOPS = [256, 32000, 50000, 64000, 100000, 127758, 128000, 200019, 256000];   // 256 L668, 127,758 L644, 200,019 L612; the rest are the prompts' sizes
const NSTOPS = [1e9, 3e9, 7e9, 8e9, 13e9, 34e9, 70e9, 175e9, 405e9];
WIDGETS["fixture:lecture_01--vocab-trade"] = (root) => {
  const s = { bytes: 1000, r1: 1, v1: 0, r2: 2.5, v2: 7, d: 2048, n: 3, tied: false };
  const pic = el("div"), read = el("div", { class: "readout" });
  const draw = () => {
    const a = { bytes: s.bytes, r1: s.r1, r2: s.r2, V1: VSTOPS[s.v1], V2: VSTOPS[s.v2], d: s.d, N: NSTOPS[s.n], tied: s.tied }, m = vocabTrade(a);
    const W = 640, x0 = 150, span = 400;
    const group = (y, title, v1, v2, unit, max) => {
      let o = text(10, y, title, { fill: C.muted, size: 11 });
      [[v1, "tokenizer 1", C.a], [v2, "tokenizer 2", C.b]].forEach(([v, n, col], i) => {
        const yy = y + 6 + i * 24;
        o += text(10, yy + 14, n, { size: 12 }) + rect(x0, yy, span * Math.min(1, v / max), 18, col, 'opacity="0.8"');
        o += text(x0 + span * Math.min(1, v / max) + 6, yy + 14, `${fmt(v, v < 10 ? 3 : 0)}${unit}`, { size: 12 });
      });
      return o;
    };
    const t1 = m.tokens1, t2 = m.tokens2, q1 = t1 * t1, q2 = t2 * t2;
    let b = group(16, `sequence length for a ${s.bytes}-byte document (tokens = bytes ÷ bytes-per-token)`, t1, t2, " tokens", Math.max(t1, t2));
    b += group(90, "attention cost ∝ length² (shown relative to the larger)", q1 / Math.max(q1, q2), q2 / Math.max(q1, q2), "", 1);
    b += group(164, `vocabulary matrices ${s.tied ? "V·d (tied)" : "2·V·d (untied)"}, share of a ${fmt(a.N / 1e9, 0)}B-parameter model`, 100 * m.share1, 100 * m.share2, "%", Math.max(100 * m.share1, 100 * m.share2, 1e-9));
    pic.innerHTML = svg(W, 222, b);
    const shorter = m.lenRatio < 1 ? 2 : m.lenRatio > 1 ? 1 : 0;
    read.innerHTML = `tokens: ${fmt(t1, 1)} vs ${fmt(t2, 1)} · length 2 / 1 = <b>${fmt(m.lenRatio, 4)}</b> (linear layers scale the same)<br>
      attention 2 / 1 = <span class="big">${fmt(m.attnRatio, 4)}</span> (1 / 2 = ${fmt(m.attnRatioInv, 4)})<br>
      vocabulary parameters: ${fmt(m.vocabParams1, 3)} vs ${fmt(m.vocabParams2, 3)} = ${fmt(100 * m.share1, 2)}% vs ${fmt(100 * m.share2, 2)}% of N
      ${m.share2 > 1 || m.share1 > 1 ? " <b style='color:var(--bad)'>(more than the whole model: the vocabulary does not fit this N)</b>" : ""}<br>
      <b>${shorter ? `tokenizer ${shorter} gives ${fmt(1 / Math.min(m.lenRatio, 1 / m.lenRatio), 3)}× shorter sequences and ${fmt(1 / Math.min(m.attnRatio, m.attnRatioInv), 3)}× cheaper attention` : "same compression: same length and attention cost"}</b><br>
      <span class="muted small">provenance: fixture:lecture_01--vocab-trade · lecture_01.py:L567-L571 (ratio), L610 and L673 (attention quadratic in length), L611 ("increasing vocabulary size ... leading to sparsity"), L276 (1000 bytes → ~250 tokens), L668 (256), L612 (200,019); video 1:07:46-1:08:22. The page does not know which ratio a vocabulary size buys; you set both. V·d per matrix is the standard embedding count (author). Sparsity (rare ids get few updates) is not drawn.</span>`;
  };
  root.append(el("div", { class: "widget" }, el("div", { class: "controls" },
    slider("document bytes", 100, 10000, s.bytes, 100, v => { s.bytes = v; draw(); }),
    slider("tokenizer 1: bytes per token", 1, 8, s.r1, 0.1, v => { s.r1 = v; draw(); }, v => v.toFixed(1)),
    slider("tokenizer 1: vocabulary V", 0, VSTOPS.length - 1, s.v1, 1, v => { s.v1 = v; draw(); }, v => VSTOPS[v].toLocaleString()),
    slider("tokenizer 2: bytes per token", 1, 8, s.r2, 0.1, v => { s.r2 = v; draw(); }, v => v.toFixed(1)),
    slider("tokenizer 2: vocabulary V", 0, VSTOPS.length - 1, s.v2, 1, v => { s.v2 = v; draw(); }, v => VSTOPS[v].toLocaleString()),
    slider("model width d", 512, 16384, s.d, 512, v => { s.d = v; draw(); }),
    slider("model parameters N", 0, NSTOPS.length - 1, s.n, 1, v => { s.n = v; draw(); }, v => `${NSTOPS[v] / 1e9}B`),
    check("tie input and output embeddings", s.tied, v => { s.tied = v; draw(); }),
    read), pic)); draw();
};

// ============================================================ 4. compute napkin ==
// C = 6·N·D (lecture_01.py:L325: 6 × 70e9 × 1e12 = 4.2e23; L393 "given a FLOPs budget (C = 6 N D)") and the
// compute-optimal rule D = 20·N (L398: 70B → ~1.4T tokens), with its caveat that inference cost favours a smaller
// model (L399; video 50:27-51:09). Substituting gives C = 120·N², N_opt = sqrt(C / 120).
function napkin({ C, N, D }) {
  const c = C ?? 6 * N * D, dd = D ?? c / (6 * N), Nopt = Math.sqrt(c / 120);
  return { C: c, D: dd, tpp: dd / N, Nopt, NoptB: Nopt / 1e9, Dopt: 20 * Nopt, D20: 20 * N, sizeVsOpt: N / Nopt };
}
MODELS["fixture:lecture_01--compute-napkin"] = {
  fn: napkin,
  cases: [
    { args: { N: 70e9, D: 1e12 }, pick: "C", expect: 4.2e23, tol: 0.05, from: "lecture_01:six-nd-preview:retrieval.1" },
    { args: { N: 8e9, D: 15e12 }, pick: "C", expect: 7.2e23, tol: 0.05, from: "lecture_01:six-nd-preview:predict" },
    { args: { C: 4.2e23, N: 35e9 }, pick: "D", expect: 2e12, tol: 0.05, from: "lecture_01:six-nd-preview:transfer" },
    { args: { N: 70e9, D: 1e12 }, pick: "D20", expect: 1.4e12, tol: 0.1, from: "lecture_01:chinchilla-d-20n:retrieval.0" },
    { args: { C: 1e25, N: 70e9 }, pick: "NoptB", expect: 290, tol: 0.1, from: "lecture_01:chinchilla-d-20n:predict" },
    { args: { C: 1e25, N: 70e9 }, pick: "Dopt", expect: 5.77e12, tol: 0.02 },            // predict's why: D ≈ 5.8T
    { args: { C: 1e25, N: 70e9 }, pick: "D", expect: 2.38e13, tol: 0.02 },               // transfer: ≈ 24T tokens
    { args: { C: 1e25, N: 70e9 }, pick: "tpp", expect: 340, tol: 0.02 },                 // ... ≈ 340 tokens per parameter
    { args: { C: 6e23, N: 10e9 }, pick: "tpp", expect: 1000, tol: 0.05, from: "lecture_01:chinchilla-d-20n:check" },
    { args: { C: 4.2e23, N: 70e9 }, pick: "tpp", expect: 14.29, tol: 0.02 },             // the lecture's 70B/1T run is below 20: slightly too big for its budget
    { args: { C: 120 * 70e9 ** 2, N: 70e9 }, pick: "tpp", expect: 20 },                  // edge: on the D = 20N line exactly
  ],
};
const MANT = [1, 1.2, 1.4, 1.5, 2, 2.5, 3, 3.5, 4, 4.2, 5, 6, 7, 8, 9];   // slider ladder; 4.2 (L325), 1.4 (L398) land exactly
const ladder = (e0, e1) => { const o = []; for (let e = e0; e <= e1; e++) for (const m of MANT) o.push(+(m + "e" + e)); return o; };
let napUid = 0;
const NL = ladder(8, 12), DL = ladder(9, 14), CL = ladder(20, 26);
WIDGETS["fixture:lecture_01--compute-napkin"] = (root) => {
  const near = (L, x) => L.reduce((bi, v, i) => Math.abs(Math.log(v / x)) < Math.abs(Math.log(L[bi] / x)) ? i : bi, 0);
  const s = { fixC: false, n: near(NL, 70e9), d: near(DL, 1e12), c: near(CL, 4.2e23) };
  const pic = el("div"), read = el("div", { class: "readout" }), ctl = el("div"), cid = `nap${++napUid}`;
  const draw = () => {
    const N = NL[s.n], m = s.fixC ? napkin({ C: CL[s.c], N }) : napkin({ N, D: DL[s.d] });
    // log-log plane: x = log10 N in [8, 13], y = log10 D in [9, 15]
    const W = 640, H = 244, px = 54, py = 26, pw = 560, ph = 180;
    const X = n => px + pw * (Math.log10(n) - 8) / 5, Y = d => py + ph * (15 - Math.log10(d)) / 6;
    let b = `<rect x="${px}" y="${py}" width="${pw}" height="${ph}" style="fill:#fff;stroke:${C.rule}"/>`;
    for (let e = 8; e <= 13; e++) b += text(X(10 ** e), py + ph + 14, `1e${e}`, { fill: C.muted, size: 10, anchor: "middle" });
    for (let e = 9; e <= 15; e++) b += text(px - 6, Y(10 ** e) + 4, `1e${e}`, { fill: C.muted, size: 10, anchor: "end" });
    b += `<line x1="${X(1e8)}" y1="${Y(2e9)}" x2="${X(1e13)}" y2="${Y(2e14)}" style="stroke:${C.ok};stroke-width:1.5"/>`;   // D = 20N spans the plot
    b += text(X(3e12) + 4, Y(6e13) + 18, "D = 20·N", { fill: C.ok, size: 11 });
    // iso-FLOP line D = C / (6N)
    const iso = n => m.C / (6 * n);
    b += `<path d="M${X(1e8).toFixed(1)},${Y(iso(1e8)).toFixed(1)} L${X(1e13).toFixed(1)},${Y(iso(1e13)).toFixed(1)}" style="fill:none;stroke:${C.a};stroke-width:2;stroke-dasharray:6 3" clip-path="url(#${cid})"/>`;
    b += `<clipPath id="${cid}"><rect x="${px}" y="${py}" width="${pw}" height="${ph}"/></clipPath>`;
    const inPlot = (n, d) => n >= 1e8 && n <= 1e13 && d >= 1e9 && d <= 1e15;
    if (inPlot(m.Nopt, m.Dopt)) b += `<circle cx="${X(m.Nopt).toFixed(1)}" cy="${Y(m.Dopt).toFixed(1)}" r="5" style="fill:#fff;stroke:${C.ok};stroke-width:2"/>`;
    if (inPlot(N, m.D)) b += `<circle cx="${X(N).toFixed(1)}" cy="${Y(m.D).toFixed(1)}" r="5.5" style="fill:${C.hi};stroke:${C.ink}"/>`;
    b += text(px, 15, `dashed: every (N, D) with C = ${fmt(m.C, 2)} FLOPs · green ring: compute-optimal · yellow: your run`, { fill: C.muted, size: 11 });
    b += text(px + pw / 2, H - 6, "parameters N (log) · y: training tokens D (log)", { fill: C.muted, size: 11, anchor: "middle" });
    pic.innerHTML = svg(W, H, b);
    const r = m.tpp, regime = Math.abs(r / 20 - 1) < 0.05 ? ["on the D = 20·N line: compute-optimal by the lecture's rule", C.ok]
      : r > 20 ? [`${fmt(r / 20, 3)}× more tokens per parameter than 20: a smaller-than-optimal model trained longer ("over-trained"; cheaper to serve, L399)`, C.b]
      : [`${fmt(20 / r, 3)}× fewer tokens per parameter than 20: a bigger-than-optimal model for this budget`, C.b];
    read.innerHTML = `C = 6 · N · D = 6 · ${fmt(N, 2)} · ${fmt(m.D, 3)} = <span class="big">${fmt(m.C, 3)} FLOPs</span>
      D = C / (6N) = ${fmt(m.D, 3)} tokens · D / N = <b>${fmt(r, 4)}</b> tokens per parameter<br>
      compute-optimal for this C: N = √(C/120) = <b>${fmt(m.NoptB, 4)}B</b>, D = 20·N = ${fmt(m.Dopt, 3)} tokens<br>
      <b style="color:${regime[1]}">${regime[0]}</b><br>
      <span class="muted small">provenance: fixture:lecture_01--compute-napkin · lecture_01.py:L325 (6 × 70e9 × 1e12 = 4.2e23), L393 (C = 6 N D), L398 (D = 20 N, 70B → ~1.4T), L399 (caveat: inference cost wants a smaller model); video 36:39-36:53, 50:27-51:09 ("quite crude"). 6ND is deferred to lecture 2; D = 20N ignores data and architecture differences.</span>`;
  };
  const build = () => {
    ctl.replaceChildren(
      slider("parameters N", 0, NL.length - 1, s.n, 1, v => { s.n = v; draw(); }, v => fmt(NL[v], 2)),
      s.fixC ? slider("budget C (FLOPs)", 0, CL.length - 1, s.c, 1, v => { s.c = v; draw(); }, v => fmt(CL[v], 2))
             : slider("tokens D", 0, DL.length - 1, s.d, 1, v => { s.d = v; draw(); }, v => fmt(DL[v], 2)));
  };
  const fix = check("hold the budget C fixed (moving N moves D)", s.fixC, v => {
    if (v) s.c = near(CL, 6 * NL[s.n] * DL[s.d]); else s.d = near(DL, CL[s.c] / (6 * NL[s.n]));
    s.fixC = v; build(); draw();
  });
  build();
  root.append(el("div", { class: "widget" }, el("div", { class: "controls" }, fix, ctl, read), pic)); draw();
};
