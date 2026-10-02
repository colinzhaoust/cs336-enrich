// Widgets for thread lecture_17 (CS336 L17, multimodal models). Register as "fixture:<id>" -> (root, notice) => void.
// Each widget has a pure model in MODELS (no DOM) that tools/check_widgets.mjs tests against the KPs' stored answers.
// Sources: official/lectures/lecture_17.py (cited as Lnn), the figures it shows (official/lectures/images/*.png, cited by
// file name), lectures/lecture_17/transcript.json (cited as video M:SS), and the cited papers where a figure the lecture
// shows does not carry a number (marked "paper, not the lecture"). Arithmetic on those numbers is this thread's and says so.
import { el, fmt, slider } from "../widgets_lib.js";

export const MODELS = {};
export const WIDGETS = {};

const C = { ink: "var(--ink)", muted: "var(--muted)", rule: "var(--rule)", a: "var(--accent)", b: "var(--accent2)", ok: "var(--ok)", hi: "var(--hilite)" };
const n1 = (x) => (Number.isFinite(x) ? x.toFixed(1) : "0");
const rect = (x, y, w, h, fill, extra = "") => `<rect x="${n1(x)}" y="${n1(y)}" width="${n1(Math.max(0, w))}" height="${n1(Math.max(0, h))}" style="fill:${fill}" ${extra}/>`;
const line = (x1, y1, x2, y2, stroke, extra = "") => `<line x1="${n1(x1)}" y1="${n1(y1)}" x2="${n1(x2)}" y2="${n1(y2)}" style="stroke:${stroke}" ${extra}/>`;
const text = (x, y, s, o = {}) => `<text x="${n1(x)}" y="${n1(y)}" style="fill:${o.fill || C.ink};font:${o.size || 12}px var(--sans)${o.bold ? ";font-weight:600" : ""}" text-anchor="${o.anchor || "start"}">${s}</text>`;
const outline = (x, y, w, h, stroke, sw = 1) => `<rect x="${n1(x)}" y="${n1(y)}" width="${n1(w)}" height="${n1(h)}" style="fill:none;stroke:${stroke};stroke-width:${sw}"/>`;
const svg = (w, h, body) => `<svg viewBox="0 0 ${w} ${h}" width="100%" style="max-width:${w}px;border:1px solid var(--rule);border-radius:6px;background:#fff" role="img">${body}</svg>`;
const check = (label, value, onChange) => { const i = el("input", { type: "checkbox" }); i.checked = value; i.addEventListener("change", () => onChange(i.checked)); return el("label", {}, i, ` ${label}`); };
// A slider whose position can be set from a preset.
const ctl = (label, min, max, value, step, on, f = (v) => v) => {
  const node = slider(label, min, max, value, step, on, f);
  return { node, set: (v) => { const i = node.querySelector("input"), o = node.querySelector(".readout"); if (i) i.value = String(v); if (o) o.textContent = f(v); } };
};
// A row of mutually exclusive buttons; the active one is solid, the rest are ghost buttons.
const choice = (title, opts, value, on) => {
  const btns = opts.map(([v, lab]) => el("button", { class: v === value ? "" : "ghost", onclick: () => { set(v); on(v); } }, lab));
  const set = (v) => btns.forEach((b, i) => { b.className = opts[i][0] === v ? "" : "ghost"; });
  return { node: el("div", { style: "margin:4px 0" }, el("span", { class: "muted small" }, `${title} `), ...btns), set };
};

// ============================================================ 1. image tokens ==
// How many tokens one image (or one video frame) becomes, under each scheme the lecture describes.
//   crop:     resize shorter side to `enc`, center crop enc x enc (L70-L72); ViT cuts `patch`-px patches (L78, "14x14
//             patches", "@336px"), one token per patch, so (enc/patch)^2 = 576 for ViT-L/14@336px (this thread's arithmetic).
//             LLaVA's per-patch linear W (L138-L139) keeps that count: one LM token per patch.
//   anyres:   a x b pieces at the encoder's resolution, encode each, concatenate (L162): a*b*(enc/patch)^2. LLaVA-1.5's
//             extra downsampled whole-image view is not in the lecture's text (it is in llava-onevision-anyres.png and
//             spoken at 39:02-40:37) and is not added.
//   native:   Qwen2-VL encodes at native resolution with ViT/14 and merges every 2x2 (L219-L220): ceil(W/14)*ceil(H/14)
//             patches / 4, "=> 66 tokens" at 224x224. The +2 is the vision start/end delimiters (qwen2_vl_2024, not the
//             lecture; KP filtering note). A side that is not a multiple of patch*merge is padded up (this thread's choice).
//   fixed256: Qwen-VL's one-layer cross-attention adaptor "maps to fixed length of 256" (L200), whatever the patch count.
//   vq:       Chameleon's VQ tokenizer: "512 x 512 image into 1024 tokens (codebook of size 8192)" (L284): a 32 x 32 grid of
//             16-px cells and log2(8192) = 13 bits per token (this thread's arithmetic, recorded in sources.json).
// Budget: OneVision makes single image, multi-image and video "roughly the same length" (L166-L170; bilinear interpolation
// when there are too many tokens, L163); Qwen2-VL video is "2 frames/sec, max 16384 tokens" (L221).
//   allowed per unit = floor(budget / units); a crop grid of floor(sqrt(allowed))^2 (81 -> 9 x 9);
//   for native, the square side that fits is patch * floor(sqrt(merge^2 * (allowed - delim))) (136 -> 23 patches, 322 px).
function imageTokens({ scheme = "crop", W = 336, H = 336, patch = 14, enc = 336, merge = 2, delim = 2, units = 1, budget = 0, codebook = 8192 }) {
  const o = { scheme };
  const perCrop = Math.floor(enc / patch) ** 2;
  o.perCrop = perCrop;
  if (scheme === "crop" || scheme === "vq") { o.g = Math.floor(enc / patch); o.perUnit = o.g * o.g; }
  else if (scheme === "anyres") { o.a = Math.ceil(W / enc); o.b = Math.ceil(H / enc); o.tiles = o.a * o.b; o.perUnit = o.tiles * perCrop; }
  else if (scheme === "native") {
    const pad = (n) => Math.ceil(n / merge) * merge;
    o.gw0 = Math.ceil(W / patch); o.gh0 = Math.ceil(H / patch); o.gw = pad(o.gw0); o.gh = pad(o.gh0); o.padded = o.gw !== o.gw0 || o.gh !== o.gh0;
    o.patches = o.gw * o.gh; o.perUnit = o.patches / (merge * merge) + delim;
  } else { o.perUnit = 256; o.patches = Math.ceil(W / patch) * Math.ceil(H / patch); }
  o.units = units; o.total = units * o.perUnit;
  o.capped = budget > 0; o.over = o.capped && o.total > budget;
  o.allowed = o.capped ? Math.floor(budget / units) : o.perUnit;
  o.allowedGrid = Math.floor(Math.sqrt(o.allowed));
  o.sidePx = scheme === "native" && o.allowed > delim ? patch * Math.floor(Math.sqrt(merge * merge * (o.allowed - delim))) : null;
  o.shrink = o.over ? o.perUnit / o.allowed : 1;
  if (scheme === "vq") {
    o.bitsPerToken = Math.log2(codebook); o.tokenBits = o.perUnit * o.bitsPerToken;
    o.rawBits = enc * enc * 3 * 8; o.ratio = o.rawBits / o.tokenBits; o.cellPx = patch;
  }
  return o;
}
MODELS["fixture:lecture_17--image-tokens"] = {
  fn: imageTokens,
  cases: [
    { args: { scheme: "crop", W: 4000, H: 3000, patch: 14, enc: 336 }, pick: "perUnit", expect: 576, from: "lecture_17:clip-preprocess-vit-patches:predict" },
    { args: { scheme: "crop", W: 4096, H: 14, patch: 14, enc: 336 }, pick: "perUnit", expect: 576 },                       // edge: any shape, still 576
    { args: { scheme: "crop", W: 336, H: 336, patch: 8, enc: 336 }, pick: "perUnit", expect: 1764 },                       // clip transfer, ViT-H/8
    { args: { scheme: "crop", W: 1920, H: 1080, patch: 16, enc: 224 }, pick: "perUnit", expect: 196, from: "lecture_17:clip-preprocess-vit-patches:check" },
    { args: { scheme: "crop", W: 448, H: 448, patch: 14, enc: 448 }, pick: "perUnit", expect: 1024, from: "lecture_17:vlm-template-encoder-projector-lm:check" },
    { args: { scheme: "fixed256", W: 1344, H: 1344, patch: 14, enc: 336 }, pick: "perUnit", expect: 256 },                 // vlm transfer, model B
    { args: { scheme: "anyres", W: 672, H: 672, patch: 14, enc: 336 }, pick: "perUnit", expect: 2304 },                    // vlm predict: 576 -> 2304 patches
    { args: { scheme: "anyres", W: 1344, H: 1344, patch: 14, enc: 336 }, pick: "perUnit", expect: 9216, from: "lecture_17:anyres-tiling-token-budget:predict" },
    { args: { scheme: "anyres", W: 337, H: 336, patch: 14, enc: 336 }, pick: "perUnit", expect: 1152 },                    // edge: one pixel over adds a tile
    { args: { scheme: "anyres", W: 1008, H: 1008, patch: 14, enc: 336 }, pick: "perUnit", expect: 5184 },                  // anyres check: 3 x 3 budget
    { args: { scheme: "anyres", W: 1008, H: 1008, patch: 14, enc: 336, units: 64, budget: 5184 }, pick: "allowed", expect: 81, from: "lecture_17:anyres-tiling-token-budget:check" },
    { args: { scheme: "anyres", W: 1008, H: 1008, patch: 14, enc: 336, units: 64, budget: 5184 }, pick: "allowedGrid", expect: 9 },
    { args: { scheme: "crop", W: 336, H: 336, patch: 14, enc: 336, units: 32, budget: 2304 }, pick: "allowed", expect: 72 }, // anyres transfer: 2304 / 32
    { args: { scheme: "native", W: 224, H: 224 }, pick: "perUnit", expect: 66 },                                            // L220
    { args: { scheme: "native", W: 448, H: 448 }, pick: "perUnit", expect: 258, from: "lecture_17:qwen2-dynamic-resolution-tokens:predict" },
    { args: { scheme: "native", W: 672, H: 336 }, pick: "perUnit", expect: 290, from: "lecture_17:qwen2-dynamic-resolution-tokens:check" },
    { args: { scheme: "native", W: 896, H: 896 }, pick: "perUnit", expect: 1026 },
    { args: { scheme: "native", W: 238, H: 238 }, pick: "perUnit", expect: 83 },                                            // edge: 17 patches padded to 18
    { args: { scheme: "native", W: 448, H: 448, units: 120, budget: 16384 }, pick: "total", expect: 30960 },               // qwen2 transfer
    { args: { scheme: "native", W: 448, H: 448, units: 120, budget: 16384 }, pick: "allowed", expect: 136 },
    { args: { scheme: "native", W: 448, H: 448, units: 120, budget: 16384 }, pick: "sidePx", expect: 322 },
    { args: { scheme: "vq", W: 512, H: 512, patch: 16, enc: 512, codebook: 8192 }, pick: "perUnit", expect: 1024 },        // L284
    { args: { scheme: "vq", W: 512, H: 512, patch: 16, enc: 512, codebook: 8192 }, pick: "bitsPerToken", expect: 13 },
    { args: { scheme: "vq", W: 256, H: 256, patch: 16, enc: 256, codebook: 16384 }, pick: "ratio", expect: 438.86, from: "lecture_17:discrete-image-tokens-vqvae:check" },
  ],
};
const SCHEMES = [
  ["crop", "CLIP crop (L70-L72)"], ["anyres", "AnyRes tiles (L162)"], ["native", "Qwen2-VL native + 2×2 (L219-L220)"],
  ["fixed256", "Qwen-VL adaptor → 256 (L200)"], ["vq", "Chameleon VQ (L284)"],
];
const SCHEME_SHORT = { crop: "crop", anyres: "AnyRes", native: "native", fixed256: "fixed 256", vq: "VQ" };
WIDGETS["fixture:lecture_17--image-tokens"] = (root) => {
  const s = { scheme: "native", W: 560, H: 420, patch: 14, enc: 336, merge: 2, delim: true, codebook: 8192, nImg: 1, sec: 0, fps: 2, budget: "none" };
  const pic = el("div"), read = el("div", { class: "readout" });
  const budgetOf = () => {
    const perCrop = Math.floor(s.enc / s.patch) ** 2;
    return { none: 0, qwen: 16384, any2: 4 * perCrop, any3: 9 * perCrop }[s.budget];
  };
  const args = (scheme) => ({ scheme, W: s.W, H: s.H, patch: s.patch, enc: s.enc, merge: s.merge, delim: s.delim ? 2 : 0, codebook: s.codebook,
    units: Math.max(s.nImg, Math.round(s.sec * s.fps)), budget: budgetOf() });
  const draw = () => {
    const o = imageTokens(args(s.scheme)), frames = Math.round(s.sec * s.fps);
    // left: the input image and what the scheme does to it
    const exW = s.scheme === "anyres" ? Math.max(s.W, o.a * s.enc) : s.W, exH = s.scheme === "anyres" ? Math.max(s.H, o.b * s.enc) : s.H; // tiles may overhang
    const bw = 300, bh = 226, sc = Math.min(bw / exW, bh / exH), w = s.W * sc, h = s.H * sc, x0 = 14 + (bw - exW * sc) / 2, y0 = 28 + (bh - exH * sc) / 2;
    let b = text(14, 18, `${s.W} × ${s.H} px input · ${SCHEME_SHORT[s.scheme]}`, { bold: true });
    b += rect(x0, y0, w, h, "#f1ede3", `stroke="${"#25313d"}"`);
    const grid = (gx, gy, cell, nx, ny, stroke, every = 1) => {
      let g = ""; if (cell * every < 3) return text(gx + 4, gy + 14, "grid too fine to draw at this scale", { fill: C.muted, size: 11 });
      for (let i = 0; i <= nx; i += every) g += line(gx + i * cell, gy, gx + i * cell, gy + ny * cell, stroke, 'stroke-width="0.6"');
      for (let j = 0; j <= ny; j += every) g += line(gx, gy + j * cell, gx + nx * cell, gy + j * cell, stroke, 'stroke-width="0.6"');
      return g;
    };
    if (s.scheme === "crop" || s.scheme === "vq") {
      const side = Math.min(w, h), cx = x0 + (w - side) / 2, cy = y0 + (h - side) / 2;
      b += rect(cx, cy, side, side, "rgba(36,102,141,.18)", `stroke="${"#24668d"}"`);
      b += grid(cx, cy, side / o.g, o.g, o.g, C.a);
      b += text(14, 272, `shorter side → ${s.enc} px, center crop; borders cut · ${o.g}×${o.g} ${s.scheme === "vq" ? "codes" : "patches"}`, { fill: C.a, size: 11 });
    } else if (s.scheme === "anyres") {
      const t = s.enc * sc; b += grid(x0, y0, t, o.a, o.b, C.b);
      b += text(14, 272, `${o.a} × ${o.b} pieces of ${s.enc} px, ${o.perCrop} tokens each (edge pieces padded)`, { fill: C.b, size: 11 });
    } else {
      const p = s.patch * sc; b += grid(x0, y0, p, o.gw, o.gh, "rgba(36,102,141,.35)");
      if (s.scheme === "native" && s.merge > 1) b += grid(x0, y0, p, o.gw, o.gh, C.a, s.merge);
      b += text(14, 272, s.scheme === "native" ? `${o.gw} × ${o.gh} patches of ${s.patch} px${o.padded ? " (padded)" : ""}; ${s.merge}×${s.merge} block → 1 token`
        : `${o.patches} patches → 1 cross-attention layer → 256 tokens`, { fill: C.a, size: 11 });
    }
    // right: same image, every scheme (log scale)
    const X = 340, span = 200, lg = (v) => span * Math.log10(Math.max(v, 1)) / 5;
    b += text(X, 18, "tokens per image, same input, each scheme", { fill: C.muted, size: 11 });
    SCHEMES.forEach(([k], i) => {
      const v = imageTokens(k === "vq" && s.scheme !== "vq" ? { ...args(k), enc: 512, patch: 16 } : args(k)).perUnit, y = 30 + i * 44, on = k === s.scheme;
      b += text(X, y + 12, SCHEME_SHORT[k] + (k === "vq" && s.scheme !== "vq" ? " (Chameleon 512 px, 16-px cells)" : ""), { bold: on, fill: on ? C.ink : C.muted });
      b += rect(X, y + 17, lg(v), 14, on ? C.a : C.rule);
      b += text(X + lg(v) + 6, y + 29, fmt(v, 0), { bold: on });
    });
    [[10, "10"], [100, "100"], [1000, "1k"], [10000, "10k"], [100000, "100k"]].forEach(([t, lab]) => { b += line(X + lg(t), 252, X + lg(t), 256, C.muted) + text(X + lg(t), 268, lab, { fill: C.muted, size: 10, anchor: "middle" }); });
    b += text(X, 284, "tokens, log scale", { fill: C.muted, size: 10 });
    pic.innerHTML = svg(640, 292, b);
    const unit = frames > s.nImg ? "frame" : "image";
    const formula = { crop: `(${s.enc}/${s.patch})² = ${o.g}² = ${o.perUnit}, whatever the input size`,
      anyres: `a·b·(${s.enc}/${s.patch})² = ⌈${s.W}/${s.enc}⌉·⌈${s.H}/${s.enc}⌉·${o.perCrop} = ${o.a}·${o.b}·${o.perCrop} = ${fmt(o.perUnit, 0)}`,
      native: `⌈${s.W}/${s.patch}⌉·⌈${s.H}/${s.patch}⌉ = ${o.gw}·${o.gh} = ${o.patches} patches; /${s.merge * s.merge}${s.delim ? " + 2 delimiters" : ""} = ${fmt(o.perUnit, 0)}`,
      fixed256: `${o.patches} patches in, 256 out: the adaptor fixes the count`,
      vq: `(${s.enc}/${s.patch})² = ${o.perUnit} codes × log₂(${s.codebook}) = ${o.bitsPerToken} bits`,
    }[s.scheme];
    const B = budgetOf();
    read.innerHTML = `${formula}<br>
      <span class="big">${fmt(o.perUnit, 0)} tokens / ${unit} · ${fmt(o.total, 0)} total</span>${o.units} ${unit}${o.units > 1 ? "s" : ""}${frames ? ` (${s.sec} s × ${s.fps} frames/s)` : ""}<br>
      ${!B ? "no budget set" : o.over ? `<b style="color:${C.b}">over the ${fmt(B, 0)}-token budget</b>: each ${unit} may have ${fmt(o.allowed, 0)} tokens, ${fmt(o.shrink, 2)}× fewer than it wants${s.scheme === "native" && o.sidePx ? `, i.e. a square ${unit} of about ${o.sidePx} px (${s.patch}·⌊√(${s.merge * s.merge}·(${o.allowed} − ${s.delim ? 2 : 0}))⌋)` : `, about a ${o.allowedGrid}×${o.allowedGrid} grid: the ${unit}'s grid is shrunk (bilinear interpolation, L163)`}` : `fits the ${fmt(B, 0)}-token budget (${fmt(o.allowed, 0)} allowed per ${unit})`}
      ${s.scheme === "vq" ? `<br>raw ${s.enc}×${s.enc} RGB at 8 bits = ${Math.round(o.rawBits).toLocaleString()} bits; tokens = ${Math.round(o.tokenBits).toLocaleString()} bits: <b>${fmt(o.ratio, 1)}× smaller</b>, each token one of ${s.codebook} codes for a ${s.patch}×${s.patch}-px cell` : ""}
      ${s.scheme === "crop" || s.scheme === "anyres" ? `<br>a per-patch linear projector W (LLaVA, L139) passes this count to the LM unchanged; W's shape (encoder width × LM width) does not contain it` : ""}<br>
      <span class="muted small">provenance: fixture:lecture_17--image-tokens · lecture_17.py L70-L72 (resize shorter side, center crop), L78 (ViT-L/14@336px), L138-L139 (per-patch W), L162-L163 (AnyRes a × b, bilinear interpolation), L166-L170 (same length across inputs), L200 (fixed 256), L219-L221 (Qwen2-VL 2×2 → 66; 2 frames/s, max 16384), L284 (512² → 1024 tokens, codebook 8192). Patch counts and bits are this thread's arithmetic; the +2 delimiters are from the Qwen2-VL paper, not the lecture; AnyRes's extra whole-image view is not added; budgets "one 2×2 / 3×3 AnyRes image" are the KP prompts' settings.</span>`;
  };
  const sch = choice("scheme", SCHEMES, s.scheme, (v) => { s.scheme = v; if (v === "vq" && s.enc === 336) { s.enc = 512; s.patch = 16; encC.set(512); patchC.set(16); } draw(); });
  const patchC = choice("patch / cell (px)", [[8, "8"], [14, "14"], [16, "16"]], s.patch, (v) => { s.patch = v; draw(); });
  const encC = choice("encoder side (px)", [[224, "224"], [256, "256"], [336, "336"], [448, "448"], [512, "512"]], s.enc, (v) => { s.enc = v; draw(); });
  const mergeC = choice("merge (native)", [[1, "none"], [2, "2×2"]], s.merge, (v) => { s.merge = v; draw(); });
  const cbC = choice("codebook (VQ)", [[8192, "8192 (L284)"], [16384, "16384"]], s.codebook, (v) => { s.codebook = v; draw(); });
  const budC = choice("token budget", [["none", "none"], ["qwen", "16384 (Qwen2-VL video, L221)"], ["any2", "one 2×2 AnyRes image"], ["any3", "one 3×3 AnyRes image"]], s.budget, (v) => { s.budget = v; draw(); });
  const sl = {
    W: ctl("width (px)", 16, 4096, s.W, 2, (v) => { s.W = v; draw(); }), H: ctl("height (px)", 16, 4096, s.H, 2, (v) => { s.H = v; draw(); }),
    nImg: ctl("images in the prompt", 1, 64, s.nImg, 1, (v) => { s.nImg = v; draw(); }),
    sec: ctl("video seconds (0 = not a video)", 0, 600, s.sec, 1, (v) => { s.sec = v; draw(); }),
    fps: ctl("frames per second", 0.25, 8, s.fps, 0.25, (v) => { s.fps = v; draw(); }),
  };
  const presets = [
    ["Qwen2-VL 224 (L220)", { scheme: "native", W: 224, H: 224, patch: 14, merge: 2, nImg: 1, sec: 0, budget: "none" }],
    ["Chameleon 512 (L284)", { scheme: "vq", W: 512, H: 512, patch: 16, enc: 512, codebook: 8192, nImg: 1, sec: 0, budget: "none" }],
    ["video at a 2×2 image's budget (L166, L170)", { scheme: "anyres", W: 336, H: 336, patch: 14, enc: 336, nImg: 1, sec: 10, fps: 2, budget: "any2" }],
  ];
  const apply = (p) => { Object.assign(s, p); sch.set(s.scheme); patchC.set(s.patch); encC.set(s.enc); mergeC.set(s.merge); cbC.set(s.codebook); budC.set(s.budget); for (const k of Object.keys(sl)) sl[k].set(s[k]); draw(); };
  root.append(el("div", { class: "widget" }, el("div", { class: "controls" },
    sch.node, sl.W.node, sl.H.node, patchC.node, encC.node, mergeC.node, check("2 delimiter tokens (native)", s.delim, (v) => { s.delim = v; draw(); }), cbC.node,
    sl.nImg.node, sl.sec.node, sl.fps.node, budC.node,
    el("div", { style: "margin-top:8px" }, el("span", { class: "muted small" }, "lecture examples "), ...presets.map(([n, p]) => el("button", { class: "ghost", onclick: () => apply(p) }, n))), read), pic));
  draw();
};
// ====================================================== 2. contrastive matrix ==
// One batch of N (image, text) pairs scored as an N x N matrix of cosine similarities times a learned scale.
//   CLIP (clip-code.png, shown at L61): logits = cos * exp(t); cross-entropy along rows and along columns, averaged.
//     "For each image, prefer its aligned text over other texts / for each text ... other images" (L59-L60): N row and
//     N column softmaxes, "2 times N different softmax classification problems" (video 7:18). Negatives per image: N - 1.
//   SigLIP (siglip-code.png, shown at L104): logits = cos * t + b; labels +1 on the diagonal, -1 off it;
//     l = -sum(log_sigmoid(labels * logits)) / n: N^2 independent binary terms (L103 "aligned or not?").
// The similarities are the learner's settings (every matched pair gets s+, every unmatched pair s-); they are not data.
// Initial scales are the papers', not the lecture's: CLIP's temperature starts at 0.07, i.e. scale 1/0.07 = 14.3
// (CLIP paper §2.5); SigLIP starts at t = exp(log 10) = 10, b = -10 (SigLIP paper §3.3).
const sigmoid = (z) => 1 / (1 + Math.exp(-z));
function contrastive({ N = 4, sPos = 0.6, sNeg = 0.1, scaleClip = 1 / 0.07, scaleSig = 10, bias = -10 }) {
  const o = { N, negatives: N - 1, softmaxProblems: 2 * N, binaryTerms: N * N };
  o.pClip = 1 / (1 + (N - 1) * Math.exp(scaleClip * (sNeg - sPos)));       // row softmax at the matched caption
  o.pClipNeg = (1 - o.pClip) / Math.max(N - 1, 1);
  o.lossClip = -Math.log(o.pClip);                                          // one row; rows and columns agree here
  o.zPos = scaleSig * sPos + bias; o.zNeg = scaleSig * sNeg + bias;
  o.pSigPos = sigmoid(o.zPos); o.pSigNegRight = sigmoid(-o.zNeg);
  o.lossSigPos = -Math.log(o.pSigPos); o.lossSigNeg = -Math.log(o.pSigNegRight);
  o.lossSigPerImage = o.lossSigPos + (N - 1) * o.lossSigNeg;                // Algorithm 1: sum over N^2 terms / N
  return o;
}
MODELS["fixture:lecture_17--contrastive-matrix"] = {
  fn: contrastive,
  cases: [
    { args: { N: 8 }, pick: "negatives", expect: 7, from: "lecture_17:clip-contrastive-objective:predict" },
    { args: { N: 32768 }, pick: "negatives", expect: 32767 },                                  // L57's example batch
    { args: { N: 32768 }, pick: "softmaxProblems", expect: 65536 },                            // "2 times N" (7:18)
    { args: { N: 4096 }, pick: "binaryTerms", expect: 16777216 },                              // siglip predict's batch, N^2
    { args: { N: 1, sPos: 0.6, sNeg: 0.1 }, pick: "pClip", expect: 1 },                        // edge: batch of 1 ranks nothing (22:24)
    { args: { N: 8, sPos: 0.6, sNeg: 0.1, scaleClip: 1 / 0.07 }, pick: "pClip", expect: 0.99450, tol: 0.001 },
    { args: { N: 32768, sPos: 0.6, sNeg: 0.1, scaleClip: 1 / 0.07 }, pick: "pClip", expect: 0.03717, tol: 0.01 }, // same pair, bigger batch
    { args: { N: 8, sPos: 0.6, sNeg: 0.1 }, pick: "pSigPos", expect: 0.01799, tol: 0.002 },   // SigLIP at init: t = 10, b = -10
    { args: { N: 32768, sPos: 0.6, sNeg: 0.1 }, pick: "pSigPos", expect: 0.01799, tol: 0.002 }, // ... unchanged by N (siglip check: "same")
    { args: { N: 8, sPos: 0.9, sNeg: 0.1, scaleSig: 20, bias: -10 }, pick: "pSigPos", expect: 0.99966, tol: 0.001 },
  ],
};
WIDGETS["fixture:lecture_17--contrastive-matrix"] = (root) => {
  const s = { mode: "clip", logN: 2, sPos: 0.3, sNeg: 0.1, scaleClip: 14.3, scaleSig: 10, bias: -2 };
  const pic = el("div"), read = el("div", { class: "readout" });
  const draw = () => {
    const N = 2 ** s.logN, o = contrastive({ N, sPos: s.sPos, sNeg: s.sNeg, scaleClip: s.scaleClip, scaleSig: s.scaleSig, bias: s.bias });
    const shown = Math.min(N, 10), cell = 22, gx = 40, gy = 60;
    let b = s.mode === "clip" ? text(14, 18, "CLIP: shade = softmax probability along each row", { size: 11, fill: C.muted }) + text(14, 32, "row and column of image 1 outlined: 2N softmaxes", { size: 11, fill: C.muted })
      : text(14, 18, "SigLIP: shade = σ(logit) of each cell on its own", { size: 11, fill: C.muted }) + text(14, 32, "label +1 on the diagonal, −1 elsewhere: N² yes/no", { size: 11, fill: C.muted });
    b += text(gx + shown * cell / 2, gy - 8, `texts T1…T${N}`, { size: 11, anchor: "middle", fill: C.muted }) + text(gx - 6, gy + 12, "I1", { size: 10, anchor: "end", fill: C.muted });
    for (let i = 0; i < shown; i++) for (let j = 0; j < shown; j++) {
      const diag = i === j, p = s.mode === "clip" ? (diag ? o.pClip : o.pClipNeg) : (diag ? o.pSigPos : 1 - o.pSigNegRight);
      b += rect(gx + j * cell, gy + i * cell, cell - 1, cell - 1, `rgba(36,102,141,${(0.06 + 0.94 * p).toFixed(3)})`);
      if (shown <= 8) b += text(gx + j * cell + cell / 2, gy + i * cell + 15, p >= 0.995 ? "1" : p < 0.005 ? "0" : p.toFixed(2).slice(1), { size: 9, anchor: "middle", fill: p > 0.55 ? "#fff" : C.ink });
    }
    if (N > shown) b += text(gx + shown * cell + 4, gy + shown * cell / 2, "…", { size: 14 }) + text(gx + 4, gy + shown * cell + 14, `(${shown} of ${N} rows drawn)`, { size: 10, fill: C.muted });
    if (s.mode === "clip") {
      b += outline(gx - 1, gy - 1, shown * cell + 1, cell + 1, "#b8582a", 2) + outline(gx - 1, gy - 1, cell + 1, shown * cell + 1, "#b8582a", 2);
    }
    // right: probability given to the matched pair as the batch grows, same similarities
    const PX = 320, PW = 300, PY = 48, PH = 182, xs = (l) => PX + PW * (l / 20), ys = (p) => PY + PH * (1 - p);
    b += outline(PX, PY, PW, PH, C.rule);
    b += text(PX, 18, "probability the model gives the matched pair (I1, T1)", { size: 11, fill: C.muted });
    let dC = "", dS = "";
    for (let l = 0; l <= 20; l += 0.25) { const q = contrastive({ N: 2 ** l, sPos: s.sPos, sNeg: s.sNeg, scaleClip: s.scaleClip, scaleSig: s.scaleSig, bias: s.bias }); dC += `${l ? "L" : "M"}${n1(xs(l))} ${n1(ys(q.pClip))} `; dS += `${l ? "L" : "M"}${n1(xs(l))} ${n1(ys(q.pSigPos))} `; }
    b += `<path d="${dC}" style="fill:none;stroke:${"#b8582a"};stroke-width:2"/><path d="${dS}" style="fill:none;stroke:${"#24668d"};stroke-width:2;stroke-dasharray:5 3"/>`;
    b += text(PX, 34, "CLIP softmax (solid)", { size: 11, fill: "#b8582a" }) + text(PX + 150, 34, "SigLIP sigmoid (dashed)", { size: 11, fill: "#24668d" });
    [[14, "16K (L120)", "end", -3], [15, "32K (L121)", "start", 3]].forEach(([l, lab, anchor, dx]) => { b += line(xs(l), PY, xs(l), PY + PH, C.muted, 'stroke-dasharray="2 3"') + text(xs(l) + dx, PY + 12, lab, { size: 9, fill: C.muted, anchor }); });
    [[0, "1"], [5, "32"], [10, "1K"], [15, "32K"], [20, "1M"]].forEach(([l, lab]) => { b += text(xs(l), PY + PH + 14, lab, { size: 10, anchor: "middle", fill: C.muted }); });
    b += text(PX + PW / 2, PY + PH + 28, "batch size N (log scale)", { size: 10, anchor: "middle", fill: C.muted });
    b += `<circle cx="${n1(xs(s.logN))}" cy="${n1(ys(o.pClip))}" r="4" style="fill:${"#b8582a"}"/><circle cx="${n1(xs(s.logN))}" cy="${n1(ys(o.pSigPos))}" r="4" style="fill:${"#24668d"}"/>`;
    pic.innerHTML = svg(640, 270, b);
    read.innerHTML = `batch N = <b>${fmt(N, 0)}</b>: each image is ranked against <b>${fmt(o.negatives, 0)}</b> wrong captions · CLIP solves ${fmt(o.softmaxProblems, 0)} softmax classifications (N rows + N columns) · SigLIP scores ${fmt(o.binaryTerms, 0)} independent yes/no pairs<br>
      CLIP: p(T1 | I1) = e^{${fmt(s.scaleClip * s.sPos, 2)}} / (e^{${fmt(s.scaleClip * s.sPos, 2)}} + ${fmt(N - 1, 0)}·e^{${fmt(s.scaleClip * s.sNeg, 2)}}) = <span class="big">${o.pClip.toPrecision(3)}</span>loss −ln p = ${fmt(o.lossClip, 3)}: <b>depends on N</b> through the denominator<br>
      SigLIP: σ(${s.scaleSig}·${s.sPos} ${s.bias < 0 ? "−" : "+"} ${Math.abs(s.bias)}) = <span class="big">${o.pSigPos.toPrecision(3)}</span>for this pair at any N; an unmatched pair is called "not aligned" with σ(−z) = ${o.pSigNegRight.toPrecision(3)}; loss per image ${fmt(o.lossSigPerImage, 3)} = own pair + (N−1) independent terms<br>
      <span class="muted small">The curves show how one pair's score depends on the rest of the batch, not downstream quality: "better than CLIP for &lt;16K" and "32K is enough" (L120-L121) are the paper's measurements, marked as lines only. provenance: fixture:lecture_17--contrastive-matrix · clip-code.png (L61: cos·exp(t), cross-entropy on both axes, averaged), siglip-code.png (L104: cos·t + b, labels 2·eye(n) − ones(n), −Σ log σ / n), L57-L60, L95, L102-L103, video 7:18 ("2 times N"), 27:51 ("if you change the batch size, it's a different loss function"). Initial scales are the papers' (CLIP: τ = 0.07, scale capped at 100; SigLIP: t = 10, b = −10), not the lecture's. Similarities s+ and s− are your settings, the same for every pair.</span>`;
  };
  const modeC = choice("matrix", [["clip", "CLIP softmax"], ["sig", "SigLIP sigmoid"]], s.mode, (v) => { s.mode = v; draw(); });
  root.append(el("div", { class: "widget" }, el("div", { class: "controls" }, modeC.node,
    slider("batch size N", 0, 20, s.logN, 1, (v) => { s.logN = v; draw(); }, (v) => fmt(2 ** v, 0)),
    slider("cosine of matched pairs s+", -1, 1, s.sPos, 0.05, (v) => { s.sPos = v; draw(); }),
    slider("cosine of unmatched pairs s−", -1, 1, s.sNeg, 0.05, (v) => { s.sNeg = v; draw(); }),
    slider("CLIP scale exp(t) (init 1/0.07 ≈ 14.3, cap 100)", 1, 100, s.scaleClip, 0.1, (v) => { s.scaleClip = v; draw(); }),
    slider("SigLIP scale t (init 10)", 1, 100, s.scaleSig, 1, (v) => { s.scaleSig = v; draw(); }),
    slider("SigLIP bias b (init −10)", -20, 10, s.bias, 0.5, (v) => { s.bias = v; draw(); }),
    read), pic));
  draw();
};
// =========================================================== 3. MRoPE axes ==
// MRoPE gives each visual token three position ids (temporal, height, width) instead of one (qwen2-vl-mrope.png, shown at
// L223: a 3-frame video of 3 x 4 patches has ids (t, h, w) from (0,0,0) to (2,2,3), and the text after it starts at (4,4,4)).
// The rotary dimensions are split among the axes. RoPE's frequencies fall with the dimension index (lecture_03 / rope
// thread), so the slot index below is a frequency rank: slot 0 rotates fastest. Layouts (L246-L247):
//   blocked      [t t t t w w w w h h h h]: axis k owns one contiguous run, so h gets only the slowest slots;
//   interleaved  [t w h t w h t w h t w h]: slot i belongs to axis i mod 3, so every axis spans fast and slow.
// The lecture's 12-slot schematic is kept as the default; larger counts repeat the same pattern (this thread's generalisation).
// Temporal ids count frames: frame k of a video gets t = k whatever the sampling rate, so two clips with the same frame count
// get the same ids; Qwen3-VL adds "explicit video timestamps (as separate tokens rather in positional embeddings)" (L248).
// Not modelled: Qwen2-VL's grouping of 2 consecutive frames into one temporal patch (paper, not the lecture).
const AX = ["t", "w", "h"], AXNAME = { t: "temporal", w: "width", h: "height" }, AXCOL = { t: "#b8582a", w: "#3c8d5a", h: "#24668d" };
function mrope({ layout = "blocked", slots = 12, T = 3, rows = 3, cols = 4, frames = 10, fpsA = 2, fpsB = 1 }) {
  const per = slots / 3;
  const axisOf = (i) => (layout === "blocked" ? AX[Math.min(2, Math.floor(i / per))] : AX[i % 3]);
  const o = { layout, slots, axes: Array.from({ length: slots }, (_, i) => axisOf(i)) };
  for (const a of AX) {
    const idx = o.axes.map((x, i) => (x === a ? i : -1)).filter((i) => i >= 0);
    o[`${a}Count`] = idx.length; o[`${a}Fast`] = Math.min(...idx); o[`${a}Slow`] = Math.max(...idx); o[`${a}Idx`] = idx;
  }
  o.textStart = Math.max(T - 1, rows - 1, cols - 1) + 1;
  o.idLast = frames - 1; o.durA = frames / fpsA; o.durB = frames / fpsB; o.sameIds = true;
  return o;
}
MODELS["fixture:lecture_17--mrope-axes"] = {
  fn: mrope,
  cases: [
    { args: { layout: "blocked", slots: 12 }, pick: "hFast", expect: 8 },          // predict: blocked h owns only slots 8-11 (slowest)
    { args: { layout: "interleaved", slots: 12 }, pick: "hFast", expect: 2 },      // predict: interleaved h reaches a fast slot
    { args: { layout: "interleaved", slots: 12 }, pick: "hSlow", expect: 11 },     // ... and the slowest one
    { args: { layout: "blocked", slots: 12 }, pick: "tSlow", expect: 3 },          // blocked t never gets a slow slot
    { args: { layout: "blocked", slots: 12 }, pick: "hCount", expect: 4 },         // predict distractor: count per axis unchanged
    { args: { layout: "interleaved", slots: 12 }, pick: "hCount", expect: 4 },
    { args: { T: 3, rows: 3, cols: 4 }, pick: "textStart", expect: 4 },            // qwen2-vl-mrope.png: text begins at (4,4,4)
    { args: { frames: 40, fpsA: 2, fpsB: 0.5 }, pick: "idLast", expect: 39 },      // mrope transfer: ids 0..39 for both clips
    { args: { frames: 40, fpsA: 2, fpsB: 0.5 }, pick: "durA", expect: 20 },        // ... covering 20 s
    { args: { frames: 40, fpsA: 2, fpsB: 0.5 }, pick: "durB", expect: 80 },        // ... and 80 s
  ],
};
WIDGETS["fixture:lecture_17--mrope-axes"] = (root) => {
  const s = { layout: "blocked", slots: 12, T: 3, rows: 3, cols: 4, frame: 0, frames: 10, fpsA: 2, fpsB: 1, stamps: false };
  const pic = el("div"), read = el("div", { class: "readout" });
  const draw = () => {
    const o = mrope(s);
    // 1. frequency slots
    const X0 = 20, SW = 600 / s.slots;
    let b = text(X0, 16, `rotary slots by frequency: fast (left) → slow (right) · ${s.layout}`, { size: 11, fill: C.muted });
    o.axes.forEach((a, i) => { b += rect(X0 + i * SW, 24, SW - 1, 20, AXCOL[a]); if (SW >= 14) b += text(X0 + i * SW + SW / 2, 38, a, { size: 11, anchor: "middle", fill: "#fff" }); });
    AX.forEach((a, k) => {
      const lo = X0 + o[`${a}Fast`] * SW, hi = X0 + (o[`${a}Slow`] + 1) * SW, y = 52 + k * 8;
      b += line(lo, y, hi, y, AXCOL[a], 'stroke-width="3"') + text(X0 + k * 200, 90, `${AXNAME[a]}: spans slots ${o[`${a}Fast`]}–${o[`${a}Slow`]}`, { size: 11, fill: AXCOL[a] });
    });
    // 2. position ids of one frame, then the text that follows
    const gy = 112, cw = 50, ch = 20;
    b += text(X0, gy - 6, `ids (t, h, w) of frame ${s.frame} of ${s.T}, ${s.rows} × ${s.cols} patches; text after the video →`, { size: 11, fill: C.muted });
    for (let r = 0; r < s.rows; r++) for (let c = 0; c < s.cols; c++) {
      b += rect(X0 + c * cw, gy + r * ch, cw - 2, ch - 2, "#f1ede3") + text(X0 + c * cw + cw / 2 - 1, gy + r * ch + 13, `(${s.frame},${r},${c})`, { size: 10, anchor: "middle" });
    }
    const tx = X0 + s.cols * cw + 16;
    for (let k = 0; k < 3; k++) { const p = o.textStart + k; b += rect(tx + k * 62, gy, 58, ch - 2, "#e8eef3") + text(tx + k * 62 + 29, gy + 13, `(${p},${p},${p})`, { size: 10, anchor: "middle" }); }
    b += text(tx, gy + 34, `text resumes at max id + 1 = ${o.textStart}; text tokens use (p, p, p)`, { size: 10, fill: C.muted });
    // 3. same frame count, two sampling rates
    const ty = gy + s.rows * ch + 26, maxDur = Math.max(o.durA, o.durB), xs = (sec) => X0 + 450 * sec / maxDur;
    b += text(X0, ty, `${s.frames} frames each · tick = a frame at its time in seconds · number = its temporal id`, { size: 11, fill: C.muted });
    [["A", s.fpsA, o.durA], ["B", s.fpsB, o.durB]].forEach(([n, fps, dur], k) => {
      const y = ty + 22 + k * 46;
      b += line(X0, y, xs(dur), y, C.ink) + text(xs(dur) + 4, y + 4, `${n}: ${fps} fps → ${fmt(dur, 2)} s`, { size: 11, bold: true });
      const every = Math.ceil(s.frames / 12);
      for (let f = 0; f < s.frames; f++) {
        const x = xs(f / fps); b += line(x, y - 5, x, y + 5, C.a);
        if (f % every === 0) b += text(x, y + 17, s.stamps ? `${fmt(f / fps, 1)}s` : String(f), { size: 9, anchor: "middle", fill: s.stamps ? C.b : C.a });
      }
    });
    pic.innerHTML = svg(640, ty + 120, b);
    read.innerHTML = `${AX.map((a) => `${AXNAME[a]}: ${o[`${a}Count`]} slots, ranks ${o[`${a}Fast`]}–${o[`${a}Slow`]}`).join(" · ")}<br>
      ${s.layout === "blocked" ? `blocked: height owns only the ${o.hCount} slowest slots and temporal only the ${o.tCount} fastest; each axis sees one end of the spectrum` : `interleaved: every axis has a slot near the fast end (rank ≤ 2) and one near the slow end; same count per axis`}<br>
      clip A and clip B: temporal ids 0…<b>${o.idLast}</b> in both, yet A covers <span class="big">${fmt(o.durA, 2)} s</span>and B <span class="big">${fmt(o.durB, 2)} s</span>${s.stamps ? "timestamp tokens (shown in red) carry the seconds; the ids still count frames" : "the ids alone cannot tell them apart: tick the timestamp tokens"}<br>
      <span class="muted small">provenance: fixture:lecture_17--mrope-axes · qwen2-vl-mrope.png (L223: ids (t, h, w), text resumes at (4,4,4) after a 3-frame 3 × 4 video), L246-L247 (interleave all axes over low and high frequency bands; [t w h …] vs [t t t t w w w w h h h h]), L248 (explicit timestamps as separate tokens). Slot index = frequency rank (RoPE's falling frequencies); the 12-slot schematic is the lecture's, larger counts are this thread's repetition of it. Qwen2-VL's 2-frame temporal patches are not modelled.</span>`;
  };
  const layC = choice("layout", [["blocked", "blocked [t t t t w w w w h h h h]"], ["interleaved", "interleaved [t w h t w h …]"]], s.layout, (v) => { s.layout = v; draw(); });
  const slotC = choice("rotary slots", [[12, "12 (lecture schematic)"], [24, "24"], [48, "48"]], s.slots, (v) => { s.slots = v; draw(); });
  root.append(el("div", { class: "widget" }, el("div", { class: "controls" }, layC.node, slotC.node,
    slider("video frames (grid)", 1, 4, s.T, 1, (v) => { s.T = v; s.frame = Math.min(s.frame, v - 1); draw(); }),
    slider("show frame", 0, 3, s.frame, 1, (v) => { s.frame = Math.min(v, s.T - 1); draw(); }),
    slider("patch rows", 1, 4, s.rows, 1, (v) => { s.rows = v; draw(); }), slider("patch columns", 1, 6, s.cols, 1, (v) => { s.cols = v; draw(); }),
    slider("frames per clip (A and B)", 1, 64, s.frames, 1, (v) => { s.frames = v; draw(); }),
    slider("clip A frames/s", 0.25, 8, s.fpsA, 0.25, (v) => { s.fpsA = v; draw(); }), slider("clip B frames/s", 0.25, 8, s.fpsB, 0.25, (v) => { s.fpsB = v; draw(); }),
    check("Qwen3-VL timestamp tokens (L248)", s.stamps, (v) => { s.stamps = v; draw(); }), read), pic));
  draw();
};
// ========================================================= 4. loss share ==
// How much of a batch's loss (and so of its gradient) one long video example contributes next to k text examples.
//   per-token mean: every token weighs the same, so an example's share is its token count / all tokens (the
//     modality-balance predict: 8000 / 8500 = 94%). This is the "video examples are long, don't want to dominate" of L249.
//   per-example mean: every example weighs the same, whatever its length.
//   square-root: each example's share proportional to sqrt(its token count). L249 names a "square-root-normalized per-token
//     loss" without a formula; reading it as this interpolation between the two is this thread's inference, not the lecture's.
// Equal per-token loss is assumed for every token, so the shares are pure weights.
function lossShare({ textLen = 500, videoLen = 8000, nText = 1 }) {
  const all = videoLen + nText * textLen;
  return {
    perToken: videoLen / all, perExample: 1 / (1 + nText),
    sqrt: Math.sqrt(videoLen) / (Math.sqrt(videoLen) + nText * Math.sqrt(textLen)),
    tokenRatio: videoLen / textLen,
  };
}
MODELS["fixture:lecture_17--loss-share"] = {
  fn: lossShare,
  cases: [
    { args: { textLen: 500, videoLen: 8000, nText: 1 }, pick: "perToken", expect: 0.9412, tol: 0.01, from: "lecture_17:modality-balance-and-stability:predict" },
    { args: { textLen: 500, videoLen: 8000, nText: 1 }, pick: "perExample", expect: 0.5 },   // predict distractor "about 50%"
    { args: { textLen: 500, videoLen: 8000, nText: 1 }, pick: "sqrt", expect: 0.8 },         // sqrt(16) = 4: 4/5 (inference)
    { args: { textLen: 500, videoLen: 500, nText: 1 }, pick: "perToken", expect: 0.5 },      // edge: equal lengths, rules agree
    { args: { textLen: 500, videoLen: 8000, nText: 16 }, pick: "perToken", expect: 0.5 },    // 16 text examples match one video
  ],
};
WIDGETS["fixture:lecture_17--loss-share"] = (root) => {
  const s = { textLen: 500, videoLen: 2000, nText: 1 };
  const pic = el("div"), read = el("div", { class: "readout" });
  const draw = () => {
    const o = lossShare(s), rows = [["per-token mean", o.perToken, "every token equal"], ["square-root (inference)", o.sqrt, "share ∝ √length"], ["per-example mean", o.perExample, "every example equal"]];
    let b = text(14, 18, `share of the batch loss from the one video (${fmt(s.videoLen, 0)} tokens) vs ${s.nText} text example${s.nText > 1 ? "s" : ""} (${fmt(s.textLen, 0)} tokens each)`, { size: 11, fill: C.muted });
    rows.forEach(([n, f, why], i) => {
      const y = 30 + i * 40, X = 170, Wd = 380;
      b += text(14, y + 14, n, { bold: i === 0 }) + text(14, y + 28, why, { size: 10, fill: C.muted });
      b += rect(X, y + 2, Wd * f, 22, "#b8582a") + rect(X + Wd * f, y + 2, Wd * (1 - f), 22, "#24668d");
      b += text(X + 4, y + 17, `video ${(100 * f).toFixed(1)}%`, { size: 11, fill: "#fff" }) + text(X + Wd + 6, y + 17, `text ${(100 * (1 - f)).toFixed(1)}%`, { size: 11 });
    });
    pic.innerHTML = svg(640, 152, b);
    read.innerHTML = `per-token mean: ${fmt(s.videoLen, 0)} / (${fmt(s.videoLen, 0)} + ${s.nText}·${fmt(s.textLen, 0)}) = <span class="big">${(100 * o.perToken).toFixed(1)}%</span>from the video, which is ${fmt(o.tokenRatio, 2)}× longer than one text example<br>
      ${o.perToken > 0.8 ? "<b>the video dominates the gradient</b>" : o.perToken > 0.5 ? "the video outweighs the text" : "the text side holds its own"} · per-example: ${(100 * o.perExample).toFixed(1)}% · √-weighted: ${(100 * o.sqrt).toFixed(1)}%<br>
      <span class="muted small">provenance: fixture:lecture_17--loss-share · lecture_17.py L249 ("square-root-normalized per-token loss: balance text and multimodal data (video examples are long, don't want to dominate)"), video 56:05-56:46. The per-token and per-example shares are arithmetic; the square-root row is this thread's reading of L249's name, not a formula the lecture gives. Every token's loss is taken as equal so the bars are pure weights. QK-norm and z-loss (L292-L293) address a different problem (norm growth, logit drift) and are not modelled.</span>`;
  };
  root.append(el("div", { class: "widget" }, el("div", { class: "controls" },
    slider("video example tokens", 100, 32000, s.videoLen, 100, (v) => { s.videoLen = v; draw(); }),
    slider("text example tokens", 50, 4000, s.textLen, 50, (v) => { s.textLen = v; draw(); }),
    slider("text examples in the batch", 1, 32, s.nText, 1, (v) => { s.nText = v; draw(); }),
    read), pic));
  draw();
};
