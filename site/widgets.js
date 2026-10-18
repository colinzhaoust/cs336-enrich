// Interactive presentations. Each widget = pure model + canvas/readout. Numbers mirror the fixtures
// and official/lectures/lecture_02.py; no widget invents constants (see provenance strings).
const H100_FLOPS_BF16 = 1979e12 / 2;      // lecture_02.py:L74 (half without sparsity)
const H100_BYTES = 3.35e12;                // lecture_02.py:L349 / facts.py

const el = (tag, attrs = {}, ...kids) => {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "class") e.className = v; else if (k.startsWith("on")) e.addEventListener(k.slice(2), v); else if (k === "html") e.innerHTML = v; else e.setAttribute(k, v);
  }
  for (const k of kids) e.append(k instanceof Node ? k : document.createTextNode(String(k)));
  return e;
};
const fmt = (x, d = 2) => Number.isFinite(x) ? (Math.abs(x) >= 1e5 || (Math.abs(x) < 1e-2 && x !== 0) ? x.toExponential(d) : x.toLocaleString(undefined, { maximumFractionDigits: d })) : "–";
function slider(label, min, max, value, step, onInput, fmtv = (v) => v) {
  const out = el("span", { class: "readout" }, fmtv(value));
  const inp = el("input", { type: "range", min, max, value, step });
  inp.addEventListener("input", () => { out.textContent = fmtv(+inp.value); onInput(+inp.value); });
  return el("label", {}, `${label} `, out, inp);
}
// Move a slider() when a preset changes the state it shows.
const setSlider = (lab, v, fmtv = (x) => x) => { const inp = lab.querySelector("input"), out = lab.querySelector(".readout"); inp.value = v; out.textContent = fmtv(+inp.value); };

// ---------------------------------------------------------------- napkin --
// pure model (MODELS below): days = 6·N·D ÷ (P·u·G·86400); largest model = GPUs × memory ÷ bytes/param
function napkinModel({ N, D, G, mfu, bpp = 12, gpuGB = 80, ngpu = 1 }) {
  return { days: (6 * N * D) / (H100_FLOPS_BF16 * mfu * G * 86400), maxN: (gpuGB * 1e9 * ngpu) / bpp, maxNB: (gpuGB * ngpu) / bpp };
}
export function napkin(root, notice) {
  const s = { N: 30e9, D: 5e12, G: 512, mfu: 0.5, bpp: 12, gpuGB: 80, ngpu: 4 };  // not the predict cases (70B/15T/1024; 8 GPUs)
  const days = () => napkinModel(s).days;
  const maxN = () => napkinModel(s).maxN;
  const read = el("div", { class: "readout" });
  const draw = () => {
    read.innerHTML = `<span class="big">${fmt(days(), 1)} days</span> = 6·N·D ÷ (P·u·G·86400)<br>
      6·${fmt(s.N, 0)}·${fmt(s.D, 0)} = ${fmt(6 * s.N * s.D, 2)} FLOPs<br><br>
      <span class="big">${fmt(maxN() / 1e9, 1)} B params fit</span> = ${s.ngpu}×${s.gpuGB} GB ÷ ${s.bpp} bytes/param<br>
      <span class="muted small">provenance: lecture_02.py:L72-L82</span>`;
  };
  const controls = el("div", { class: "controls" },
    slider("parameters N (B)", 1, 1000, 30, 1, v => { s.N = v * 1e9; draw(); }),
    slider("tokens D (T)", 0.1, 50, 5, 0.1, v => { s.D = v * 1e12; draw(); }),
    slider("GPUs", 8, 16384, 512, 8, v => { s.G = v; draw(); }),
    slider("MFU", 0.1, 1, 0.5, 0.05, v => { s.mfu = v; draw(); }),
    slider("bytes / param", 4, 20, 12, 2, v => { s.bpp = v; draw(); }, v => `${v}  (2+2+${v - 4} opt)`),
    slider("GPUs for memory", 1, 64, 4, 1, v => { s.ngpu = v; draw(); }),
  );
  root.append(el("div", { class: "widget" }, controls, read));
  draw();
}

// ------------------------------------------------------------ contraction --
export function contraction(root, notice) {
  const s = { B: 3, D: 4, K: 2, step: 0 };
  const cv = el("canvas", { width: 640, height: 300 });
  const read = el("div", { class: "readout" });
  const cells = () => s.B * s.K;
  const draw = () => {
    const ctx = cv.getContext("2d"); ctx.clearRect(0, 0, cv.width, cv.height);
    const cs = Math.min(34, 240 / Math.max(s.B, s.D, s.K), 530 / (s.D + 2 * s.K)); // x, w, y side by side: right edge 100 + (D + 2K)·cs ≤ 630
    const done = Math.min(s.step, cells());
    const cur = done < cells() && s.step > 0 ? done : -1; // cell being computed = next
    const active = s.step > 0 ? Math.min(s.step - 1, cells() - 1) : -1;
    const ai = active >= 0 ? Math.floor(active / s.K) : -1, ak = active >= 0 ? active % s.K : -1;
    const grid = (x0, y0, r, c, name, hl) => {
      ctx.font = "13px sans-serif"; ctx.fillStyle = "#6b7680"; ctx.fillText(name, x0, y0 - 8);
      for (let i = 0; i < r; i++) for (let j = 0; j < c; j++) {
        const h = hl(i, j);
        ctx.fillStyle = h || "#fff"; ctx.strokeStyle = "#d9d3c7";
        ctx.fillRect(x0 + j * cs, y0 + i * cs, cs, cs); ctx.strokeRect(x0 + j * cs, y0 + i * cs, cs, cs);
      }
    };
    grid(20, 40, s.B, s.D, `x  (B×D = ${s.B}×${s.D})`, (i, j) => i === ai ? "rgba(36,102,141,.35)" : null);
    grid(20 + s.D * cs + 40, 40, s.D, s.K, `w  (D×K = ${s.D}×${s.K})`, (i, j) => j === ak ? "rgba(184,88,42,.35)" : null);
    grid(20 + (s.D + s.K) * cs + 80, 40, s.B, s.K, `y  (B×K)`, (i, j) => (i * s.K + j) < s.step ? (i === ai && j === ak ? "rgba(227,178,60,.6)" : "rgba(227,178,60,.25)") : null);
    read.innerHTML = `<span class="big">${fmt(2 * s.D * Math.min(s.step, cells()), 0)} FLOPs</span> after ${Math.min(s.step, cells())} of ${cells()} cells · each cell 2·D = ${2 * s.D}<br>
      total 2·B·D·K = ${fmt(2 * s.B * s.D * s.K, 0)}<br><span class="muted small">provenance: lecture_02.py:L306-L310</span>`;
  };
  const reset = () => { s.step = 0; draw(); };
  const controls = el("div", { class: "controls" },
    slider("B", 1, 8, 3, 1, v => { s.B = v; reset(); }), slider("D", 1, 8, 4, 1, v => { s.D = v; reset(); }), slider("K", 1, 8, 2, 1, v => { s.K = v; reset(); }),
    el("div", { style: "margin-top:10px" }, el("button", { onclick: () => { s.step = Math.min(s.step + 1, cells()); draw(); } }, "next cell"), " ",
      el("button", { onclick: () => { s.step = cells(); draw(); } }, "all"), " ", el("button", { class: "ghost", onclick: reset }, "reset")),
    read);
  root.append(el("div", { class: "widget" }, controls, cv));
  draw();
}

// ------------------------------------------------------------ memory ledger --
// pure model (MODELS below): MiB per component for the toy MLP (N = D²·L), bytes per parameter, and activation
// memory as a fraction of the no-accumulation case (activations scale with the micro-batch, 2·micro·D·L)
function memoryLedgerModel({ B, micro, D = 1024, L = 16, opt = 4 }) {
  const N = D * D * L, mib = 2 ** 20;
  return { params: 2 * N / mib, grads: 2 * N / mib, opt: opt * N / mib, act: 2 * micro * D * L / mib, bytesPerParam: 2 + 2 + opt, actFraction: micro / B };
}
export function memoryLedger(root, notice) {
  const s = { B: 64, micro: 64, D: 1024, L: 16, opt: 4 };  // micro = B means no accumulation; the batch slider then drives micro too
  const cv = el("canvas", { width: 640, height: 220 });
  const read = el("div", { class: "readout" });
  const vals = () => { const m = memoryLedgerModel(s); return { "params (bf16)": m.params, "grads (bf16)": m.grads, [`optimizer (fp32 ×${s.opt / 4})`]: m.opt, "activations (bf16)": m.act }; };
  const draw = () => {
    const v = vals(), ctx = cv.getContext("2d"); ctx.clearRect(0, 0, cv.width, cv.height);
    const max = Math.max(...Object.values(v)); let y = 20;
    for (const [k, x] of Object.entries(v)) {
      ctx.fillStyle = "#6b7680"; ctx.font = "13px sans-serif"; ctx.fillText(k, 10, y + 14);
      ctx.fillStyle = k.startsWith("act") ? "#b8582a" : "#24668d"; ctx.fillRect(190, y, Math.max(2, 400 * x / max), 18);
      ctx.fillStyle = "#25313d"; ctx.fillText(`${fmt(x, 1)} MiB`, 196 + Math.max(2, 400 * x / max), y + 14); y += 46;
    }
    read.innerHTML = `micro-steps per update: <b>${Math.round(s.B / s.micro)}</b> · FLOPs per update = 6·B·N (unchanged by micro)<br><span class="muted small">lecture_02.py:L636-L646, L719-L731 (B=64, D=1024, L=16)</span>`;
  };
  const sMicro = slider("micro-batch", 16, 8192, 64, 16, v => { s.micro = Math.min(v, s.B); setSlider(sMicro, s.micro); draw(); });
  const controls = el("div", { class: "controls" },
    slider("batch B", 16, 8192, 64, 16, v => { const tied = s.micro === s.B; s.B = v; s.micro = tied ? v : Math.min(s.micro, v); setSlider(sMicro, s.micro); draw(); }),
    sMicro,
    slider("optimizer bytes/param", 4, 8, 4, 4, v => { s.opt = v; draw(); }, v => v === 4 ? "4 (AdaGrad)" : "8 (Adam)"),
    read);
  root.append(el("div", { class: "widget" }, controls, cv));
  draw();
}

// -------------------------------------------------------------- rope pair --
export function ropePair(root, notice) {
  const s = { m: 2, n: 3, shift: 2, theta: Math.PI / 6, q: [1, 0], k: [0.7, 0.7], only: false };  // m=1 would open at the supp-rotation predict (30°, 90°)
  const rot = (v, a) => [Math.cos(a) * v[0] - Math.sin(a) * v[1], Math.sin(a) * v[0] + Math.cos(a) * v[1]];
  const dot = (a, b) => a[0] * b[0] + a[1] * b[1];
  const cv = el("canvas", { width: 360, height: 360 });
  const read = el("div", { class: "readout" });
  const draw = () => {
    const ctx = cv.getContext("2d"); ctx.clearRect(0, 0, 360, 360);
    const c = 180, R = 140; ctx.strokeStyle = "#d9d3c7"; ctx.beginPath(); ctx.arc(c, c, R, 0, 7); ctx.stroke();
    const qm = rot(s.q, s.m * s.theta), kn = rot(s.k, s.n * s.theta);
    const qs = rot(s.q, (s.m + s.shift) * s.theta), ks = s.only ? kn : rot(s.k, (s.n + s.shift) * s.theta);
    const arrow = (v, col, w) => { ctx.strokeStyle = col; ctx.lineWidth = w; ctx.beginPath(); ctx.moveTo(c, c); ctx.lineTo(c + v[0] * R, c - v[1] * R); ctx.stroke(); };
    arrow(qm, "rgba(36,102,141,.35)", 2); arrow(kn, "rgba(184,88,42,.35)", 2); arrow(qs, "#24668d", 4); arrow(ks, "#b8582a", 4); ctx.lineWidth = 1;
    const before = dot(qm, kn), after = dot(qs, ks);
    read.innerHTML = `q<sub>m</sub>·k<sub>n</sub> = <b>${before.toFixed(4)}</b> (faint arrows)<br>after shift by s: <span class="big" style="color:${Math.abs(before - after) < 1e-9 ? "#3c8d5a" : "#b8582a"}">${after.toFixed(4)}</span>
      ${Math.abs(before - after) < 1e-9 ? "unchanged" : "changed"}<br>rotation difference: (n−m)θ = ${((s.n - s.m) * s.theta * 180 / Math.PI).toFixed(0)}°<br><span class="muted small">model: 4blue2brown/cs336/models/rope.mjs rotaryPair()</span>`;
  };
  const controls = el("div", { class: "controls" },
    slider("m (query position)", 0, 8, s.m, 1, v => { s.m = v; draw(); }), slider("n (key position)", 0, 8, 3, 1, v => { s.n = v; draw(); }),
    slider("shared shift s", 0, 8, 2, 1, v => { s.shift = v; draw(); }),
    el("label", {}, el("input", { type: "checkbox", onchange: e => { s.only = e.target.checked; draw(); } }), " control: shift only q"),
    read);
  root.append(el("div", { class: "widget" }, controls, cv));
  draw();
}

// -------------------------------------------------------------- ratio clip --
export function ratioClip(root, notice) {
  const s = { eps: 0.3, A: 0.5 };  // eps 0.2, A 1 is the ppo-ratio-clip predict
  const cv = el("canvas", { width: 640, height: 300 });
  const read = el("div", { class: "readout" });
  const draw = () => {
    const ctx = cv.getContext("2d"); ctx.clearRect(0, 0, 640, 300);
    const x = r => 40 + (r / 2.5) * 580, y = v => 260 - (v + 1.5) / 4.5 * 240;
    ctx.strokeStyle = "#d9d3c7"; ctx.beginPath(); ctx.moveTo(40, 260); ctx.lineTo(620, 260); ctx.moveTo(x(1), 20); ctx.lineTo(x(1), 260); ctx.stroke();
    ctx.fillStyle = "#6b7680"; ctx.font = "12px sans-serif"; ctx.fillText("ratio r = π_new/π_old", 470, 285); ctx.fillText("r=1", x(1) + 4, 32);
    ctx.fillText(`1−ε=${(1 - s.eps).toFixed(2)}`, x(1 - s.eps) - 50, 250); ctx.fillText(`1+ε=${(1 + s.eps).toFixed(2)}`, x(1 + s.eps) + 4, 250);
    const draw1 = (f, col) => { ctx.strokeStyle = col; ctx.lineWidth = 2; ctx.beginPath(); for (let r = 0; r <= 2.5; r += 0.01) { const v = f(r); r === 0 ? ctx.moveTo(x(r), y(v)) : ctx.lineTo(x(r), y(v)); } ctx.stroke(); ctx.lineWidth = 1; };
    draw1(r => r * s.A, "rgba(36,102,141,.4)");
    draw1(r => Math.min(r * s.A, Math.min(Math.max(r, 1 - s.eps), 1 + s.eps) * s.A), "#24668d");
    read.innerHTML = `objective = min(r·A, clip(r, 1−ε, 1+ε)·A)<br>with A ${s.A >= 0 ? "> 0" : "< 0"}: gradient is <b>zero</b> for r ${s.A >= 0 ? ">" : "<"} ${(s.A >= 0 ? 1 + s.eps : 1 - s.eps).toFixed(2)} (clipped side), unclipped on the other side<br><span class="muted small">faint: unclipped r·A. See FeynRL algs/PPO for the code lines.</span>`;
  };
  const controls = el("div", { class: "controls" },
    slider("clip ε", 0, 1, 0.3, 0.05, v => { s.eps = v; draw(); }), slider("advantage A", -1.5, 1.5, 0.5, 0.1, v => { s.A = v; draw(); }), read);
  root.append(el("div", { class: "widget" }, controls, cv));
  draw();
}

export const WIDGETS = {
  "fixture:napkin": napkin,
  "fixture:matmul-contraction": contraction,
  "fixture:memory-ledger": memoryLedger,
  "fixture:rope-shared-shift": ropePair,
  "fixture:ratio-clip": ratioClip,
};

// ======================================================= lecture 5 widgets ==

export function tilingSweep(root, notice) {
  const s = { N: 8, T: 2, step: 0 };
  const cv = el("canvas", { width: 640, height: 300 }); const read = el("div", { class: "readout" });
  const phases = () => Math.ceil(s.N / s.T); // last tile is partial (masked) when T does not divide N
  const draw = () => {
    const ctx = cv.getContext("2d"); ctx.clearRect(0, 0, 640, 300); const cs = Math.min(22, 180 / s.N);
    const grid = (x0, y0, name, hl) => { ctx.fillStyle = "#6b7680"; ctx.font = "12px sans-serif"; ctx.fillText(name, x0, y0 - 6); for (let i = 0; i < s.N; i++) for (let j = 0; j < s.N; j++) { const h = hl(i, j); ctx.fillStyle = h || "#fff"; ctx.strokeStyle = "#e4dfd4"; ctx.fillRect(x0 + j * cs, y0 + i * cs, cs, cs); ctx.strokeRect(x0 + j * cs, y0 + i * cs, cs, cs); } };
    const ph = Math.min(s.step, phases()); const tile = { r: 0, c: 0 }; // output tile (0,0)
    grid(20, 30, "A", (i, j) => i < s.T && Math.floor(j / s.T) < ph ? (Math.floor(j / s.T) === ph - 1 ? "rgba(36,102,141,.5)" : "rgba(36,102,141,.15)") : null);
    grid(20 + s.N * cs + 40, 30, "B", (i, j) => j < s.T && Math.floor(i / s.T) < ph ? (Math.floor(i / s.T) === ph - 1 ? "rgba(184,88,42,.5)" : "rgba(184,88,42,.15)") : null);
    grid(20 + 2 * (s.N * cs + 40), 30, "C (one T×T tile)", (i, j) => i < s.T && j < s.T ? `rgba(227,178,60,${0.15 + 0.6 * ph / phases()})` : null);
    const naive = s.N * s.N * 2 * s.N, tiled = 2 * s.N * s.N * phases(); // each A and B element is read once per output tile column / row
    read.innerHTML = `phase ${ph} / ${phases()} · each phase loads one T×T tile of A and B into shared memory and does T³ multiply-adds${s.N % s.T ? " (fewer in the partial last tile)" : ""}<br>
      <span class="big">global reads ÷ ${fmt(naive / tiled, 2)}</span> naive ${naive} → tiled ${tiled} element reads for the whole N×N product${s.N % s.T ? ` (T does not divide N: ⌈N/T⌉ = ${phases()} tiles per row)` : ""}<br>
      total multiply-adds N³ = ${s.N ** 3} for every T: tiling changes the reads, not the FLOPs<br><span class="muted small">lecture_05.pdf:p40-p42 ("factor of T reduction")</span>`;
  };
  const controls = el("div", { class: "controls" }, slider("N", 4, 16, 8, 4, v => { s.N = v; s.T = Math.min(s.T, v); s.step = 0; draw(); }), slider("tile T", 1, 8, 2, 1, v => { s.T = Math.min(v, s.N); s.step = 0; draw(); }),
    el("div", { style: "margin-top:8px" }, el("button", { onclick: () => { s.step = Math.min(s.step + 1, phases()); draw(); } }, "next phase"), " ", el("button", { class: "ghost", onclick: () => { s.step = 0; draw(); } }, "reset")), read);
  root.append(el("div", { class: "widget" }, controls, cv)); draw();
}

export function onlineSoftmax(root, notice) {
  const s = { scores: [1.0, 3.0, 0.5, 2.0, 4.0, 1.5, 2.5, 0.0], block: 2, step: 0 };
  const read = el("div", { class: "readout" }); const cv = el("canvas", { width: 640, height: 120 });
  const run = () => { let m = -Infinity, l = 0; const trace = []; for (let b = 0; b < s.scores.length; b += s.block) { const blk = s.scores.slice(b, b + s.block); const mNew = Math.max(m, ...blk); const scale = Math.exp(m - mNew); l = l * scale + blk.reduce((a, x) => a + Math.exp(x - mNew), 0); trace.push({ blk, mOld: m, mNew, scale, l }); m = mNew; } return { trace, m, l }; };
  const draw = () => {
    const { trace, m, l } = run(); const k = Math.min(s.step, trace.length); const ctx = cv.getContext("2d"); ctx.clearRect(0, 0, 640, 120);
    s.scores.forEach((v, i) => { const inBlk = Math.floor(i / s.block) < k; ctx.fillStyle = Math.floor(i / s.block) === k - 1 ? "#e3b23c" : inBlk ? "rgba(36,102,141,.4)" : "#eee"; ctx.fillRect(20 + i * 74, 100 - v * 20, 60, v * 20 + 2); ctx.fillStyle = "#25313d"; ctx.font = "12px sans-serif"; ctx.fillText(v.toFixed(1), 40 + i * 74, 114); });
    const rows = trace.slice(0, k).map((t, i) => `block ${i + 1} ${JSON.stringify(t.blk)}: m ${t.mOld === -Infinity ? "−∞" : t.mOld.toFixed(1)} → ${t.mNew.toFixed(1)}, scale=exp(m_old−m_new)=${t.mOld === -Infinity ? "0" : t.scale.toFixed(3)}, l=${t.l.toFixed(3)}`);
    const exact = s.scores.reduce((a, x) => a + Math.exp(x - m), 0);
    read.innerHTML = `<pre>${rows.join("\n") || "press next block"}</pre>${k === trace.length ? `final l = ${l.toFixed(4)} vs one-pass Σexp(x−max) = ${exact.toFixed(4)} ✓` : ""}<br><span class="muted small">Milakov &amp; Gimelshein 2018; lecture_05.pdf:p52-p53</span>`;
  };
  root.append(el("div", { class: "widget" }, el("div", { class: "controls" }, slider("block size", 1, 4, 2, 1, v => { s.block = v; s.step = 0; draw(); }), el("div", { style: "margin-top:8px" }, el("button", { onclick: () => { s.step++; draw(); } }, "next block"), " ", el("button", { class: "ghost", onclick: () => { s.step = 0; draw(); } }, "reset")), read), cv)); draw();
}

export function flashForwardTiles(root, notice) {
  const s = { N: 1024, d: 64, Bc: 64, bytes: 2 };
  const read = el("div", { class: "readout" });
  const draw = () => {
    const tilesK = Math.ceil(s.N / s.Bc), tilesQ = Math.ceil(s.N / s.Bc);
    const hbmNaive = s.bytes * (3 * s.N * s.d + 2 * s.N * s.N); // Q,K,V + write S, read P (ignoring O)
    const hbmFlash = s.bytes * (s.N * s.d + tilesQ * (2 * s.N * s.d) + s.N * s.d); // Q once, K,V per Q-tile, O once
    read.innerHTML = `S = QKᵀ is ${s.N}×${s.N} = ${(s.N * s.N * s.bytes / 1e6).toFixed(1)} MB; one tile is ${s.Bc}×${s.Bc} = ${(s.Bc * s.Bc * s.bytes / 1e3).toFixed(1)} KB in shared memory<br>
      <span class="big">HBM traffic ${(hbmFlash / 1e6).toFixed(1)} MB</span> tiled (O(N·d) per pass × ${tilesQ} Q-tiles) vs <b>${(hbmNaive / 1e6).toFixed(1)} MB</b> materializing S and P<br>
      FLOPs are identical (${(4 * s.N * s.N * s.d / 1e9).toFixed(2)} GFLOP) — only memory traffic changes<br><span class="muted small">Dao et al. 2022 (FlashAttention); lecture_05.pdf:p50-p54</span>`;
  };
  root.append(el("div", { class: "widget" }, el("div", { class: "controls" }, slider("N (sequence)", 256, 8192, 1024, 256, v => { s.N = v; draw(); }), slider("head dim d", 32, 128, 64, 32, v => { s.d = v; draw(); }), slider("tile Bc", 16, 256, 64, 16, v => { s.Bc = v; draw(); })), read)); draw();
}

// ========================================================= FeynRL widgets ==
export function rolloutTimeline(root, notice) {
  const s = { rollout: 4, train: 3, sync: 0.5, epochs: 4, mode: "sync", maxLag: 1 };
  const cv = el("canvas", { width: 640, height: 200 }); const read = el("div", { class: "readout" });
  const draw = () => {
    const ctx = cv.getContext("2d"); ctx.clearRect(0, 0, 640, 200); const px = 560 / (s.epochs * (s.rollout + s.train + s.sync) + 1);
    const bar = (y, x0, len, col, txt) => { ctx.fillStyle = col; ctx.fillRect(40 + x0 * px, y, Math.max(1, len * px), 26); if (txt) { ctx.fillStyle = "#fff"; ctx.font = "11px sans-serif"; ctx.fillText(txt, 44 + x0 * px, y + 17); } };
    ctx.fillStyle = "#6b7680"; ctx.font = "12px sans-serif"; ctx.fillText("rollout GPUs", 40, 40); ctx.fillText("training GPUs", 40, 100);
    let busyR = 0, busyT = 0, total = 0;
    if (s.mode === "sync") {
      let t = 0; for (let e = 0; e < s.epochs; e++) { bar(48, t, s.rollout, "#b8582a", `v${e} rollout`); bar(108, t, s.rollout, "#eee"); t += s.rollout; bar(48, t, s.train, "#eee"); bar(108, t, s.train, "#24668d", `train v${e}`); t += s.train; if (e < s.epochs - 1) { bar(48, t, s.sync, "#e3b23c"); bar(108, t, s.sync, "#e3b23c"); t += s.sync; } busyR += s.rollout; busyT += s.train; } total = t;
    } else {
      const period = Math.max(s.rollout, s.train) + s.sync; let t = 0; for (let e = 0; e < s.epochs; e++) { bar(48, t, period - s.sync, "#b8582a", `rollout (v${Math.max(0, e - s.maxLag)}…v${e})`); bar(108, t, s.train, "#24668d", `train v${e}`); if (s.train < period - s.sync) bar(108, t + s.train, period - s.sync - s.train, "#eee"); bar(48, t + period - s.sync, s.sync, "#e3b23c"); bar(108, t + period - s.sync, s.sync, "#e3b23c"); t += period; busyR += period - s.sync; busyT += s.train; } total = t;
    }
    read.innerHTML = `<span class="big">${s.mode}</span> ${s.epochs} epochs in ${total.toFixed(1)} time units · rollout GPUs busy ${(100 * busyR / total).toFixed(0)}% · training GPUs busy ${(100 * busyT / total).toFixed(0)}%<br>
      ${s.mode === "sync" ? "each pool idles while the other works; weight sync (yellow) after every epoch but the last (no sync once training ends)" : `pools overlap; replay may hold up to max_lag+1 = ${s.maxLag + 1} policy versions`}<br><span class="muted small">FeynRL run_rl_sync.py / run_rl_async.py (see KP anchors)</span>`;
  };
  const controls = el("div", { class: "controls" }, el("label", {}, el("input", { type: "checkbox", onchange: e => { s.mode = e.target.checked ? "overlap" : "sync"; draw(); } }), " overlap (async) mode"),
    slider("rollout time", 1, 8, 4, 0.5, v => { s.rollout = v; draw(); }), slider("train time", 1, 8, 3, 0.5, v => { s.train = v; draw(); }), slider("max_lag", 0, 4, 1, 1, v => { s.maxLag = v; draw(); }), read);
  root.append(el("div", { class: "widget" }, controls, cv)); draw();
}

Object.assign(WIDGETS, {
  "fixture:tiling-sweep": tilingSweep,
  "fixture:online-softmax-blocks": onlineSoftmax,
  "fixture:flash-forward-tiles": flashForwardTiles,
  "fixture:rollout-train-timeline": rolloutTimeline,
});

// ======================================================= lecture 2 extras ==
// backward = two contractions of the same upstream gradient (lecture_02.py:L533-L541)
export function backwardTwoContractions(root, notice) {
  const s = { B: 3, D: 4, which: 0 };
  const cv = el("canvas", { width: 640, height: 260 }); const read = el("div", { class: "readout" });
  const draw = () => {
    const ctx = cv.getContext("2d"); ctx.clearRect(0, 0, 640, 260); const cs = 26;
    const grid = (x0, y0, r, c, name, hl) => { ctx.fillStyle = "#6b7680"; ctx.font = "12px sans-serif"; ctx.fillText(name, x0, y0 - 6); for (let i = 0; i < r; i++) for (let j = 0; j < c; j++) { const h = hl(i, j); ctx.fillStyle = h || "#fff"; ctx.strokeStyle = "#d9d3c7"; ctx.fillRect(x0 + j * cs, y0 + i * cs, cs, cs); ctx.strokeRect(x0 + j * cs, y0 + i * cs, cs, cs); } };
    const G = "G = ∂L/∂h2  (B×D)";
    if (s.which === 0) { // h1.grad = G · w2ᵀ : contract over "out"
      grid(20, 40, s.B, s.D, G, (i, j) => i === 0 ? "rgba(36,102,141,.35)" : null);
      grid(20 + s.D * cs + 40, 40, s.D, s.D, "w2ᵀ  (out×in)", (i, j) => j === 0 ? "rgba(184,88,42,.35)" : null);
      grid(20 + 2 * s.D * cs + 80, 40, s.B, s.D, "h1.grad  (B×in)", (i, j) => i === 0 && j === 0 ? "rgba(227,178,60,.6)" : null);
      read.innerHTML = `<b>∂L/∂h1 = G · w2ᵀ</b> contracts the <b>out</b> axis: 2·B·D·D FLOPs<br>einsum("batch out, in out -> batch in")`;
    } else { // w2.grad = h1ᵀ · G : contract over "batch"
      grid(20, 40, s.D, s.B, "h1ᵀ  (in×B)", (i, j) => i === 0 ? "rgba(36,102,141,.35)" : null);
      grid(20 + s.B * cs + 40, 40, s.B, s.D, G, (i, j) => j === 0 ? "rgba(184,88,42,.35)" : null);
      grid(20 + (s.B + s.D) * cs + 80, 40, s.D, s.D, "w2.grad  (in×out)", (i, j) => i === 0 && j === 0 ? "rgba(227,178,60,.6)" : null);
      read.innerHTML = `<b>∂L/∂w2 = h1ᵀ · G</b> contracts the <b>batch</b> axis: 2·B·D·D FLOPs<br>einsum("batch out, batch in -> in out")`;
    }
    read.innerHTML += `<br>forward 2BD² + backward (2BD² + 2BD²) → backward = 2× forward<br><span class="muted small">lecture_02.py:L522, L533-L541</span>`;
  };
  root.append(el("div", { class: "widget" }, el("div", { class: "controls" }, el("div", {}, el("button", { onclick: () => { s.which = 0; draw(); } }, "∂L/∂h1"), " ", el("button", { onclick: () => { s.which = 1; draw(); } }, "∂L/∂w2")), read), cv)); draw();
}

Object.assign(WIDGETS, {
  "fixture:backward-two-contractions": backwardTwoContractions,
});

// ========================================================= lecture 3 widgets ==
// d_ff ratio / block parameter calculator (lecture_03/fixtures/block_params.json)
export function blockParams(root, notice) {
  const s = { d: 4096, ratio: 4.0, gated: false, heads: 32, hd: 128 };
  const read = el("div", { class: "readout" });
  const calc = () => { const dff = Math.round(s.ratio * s.d); const ffn = (s.gated ? 3 : 2) * s.d * dff; const attn = s.d * (s.heads * s.hd) * 3 + (s.heads * s.hd) * s.d; return { dff, ffn, attn, wr: s.heads * s.hd / s.d }; };
  const draw = () => { const c = calc(); read.innerHTML = `d_ff = ${c.dff} · FFN params = ${s.gated ? 3 : 2}·d·d_ff = <span class="big">${(c.ffn / 1e6).toFixed(1)} M</span>attention params = 4·d·(h·d_head) = ${(c.attn / 1e6).toFixed(1)} M · h·d_head/d = ${c.wr.toFixed(2)}<br>block total ${((c.ffn + c.attn) / 1e6).toFixed(1)} M<br><span class="muted small">lecture_03.pdf:p37-p43; gated FFN has a third matrix, so 8/3 × d matches 2 × 4 × d</span>`; };
  const base = { d: 4096, heads: 32, hd: 128 }; // every preset sets every field, so none inherits another's d or heads
  const presets = [["ReLU 4×", { ...base, ratio: 4, gated: false }], ["SwiGLU 8/3×", { ...base, ratio: 8 / 3, gated: true }], ["Mistral-style 3.5× gated", { ...base, ratio: 3.5, gated: true }], ["T5-11B 64×", { d: 1024, ratio: 64, gated: false, heads: 128, hd: 128 }]];
  const r2 = v => v.toFixed(2);
  const sd = slider("d_model", 256, 32768, 4096, 256, v => { s.d = v; draw(); }), sr = slider("d_ff / d_model", 1, 64, 4, 1 / 12, v => { s.ratio = v; draw(); }, r2);
  const gate = el("input", { type: "checkbox", onchange: e => { s.gated = e.target.checked; draw(); } });
  const sh = slider("heads", 1, 256, 32, 1, v => { s.heads = v; draw(); }), shd = slider("head_dim", 16, 512, 128, 16, v => { s.hd = v; draw(); });
  const apply = p => { Object.assign(s, p); setSlider(sd, s.d); setSlider(sr, s.ratio, r2); gate.checked = s.gated; setSlider(sh, s.heads); setSlider(shd, s.hd); draw(); };
  root.append(el("div", { class: "widget" }, el("div", { class: "controls" },
    sd, sr, el("label", {}, gate, " gated (SwiGLU: third matrix V)"), sh, shd,
    el("div", { style: "margin-top:8px" }, ...presets.map(([n, p]) => el("button", { onclick: () => apply(p) }, n)))), read)); draw();
}
Object.assign(WIDGETS, { "fixture:block-params": blockParams });

// ========================================================= lecture 9 widgets ==
// Chinchilla parametric fit L(N,D)=E + A/N^a + B/D^b (Hoffmann et al. 2022, §3.3 constants)
const CH = { E: 1.69, A: 406.4, B: 410.7, a: 0.34, b: 0.28 };
const chLoss = (N, D) => CH.E + CH.A / Math.pow(N, CH.a) + CH.B / Math.pow(D, CH.b);
export function isoflopSweep(root, notice) {
  const s = { logC: 19 };
  const cv = el("canvas", { width: 640, height: 300 }); const read = el("div", { class: "readout" });
  const draw = () => {
    const ctx = cv.getContext("2d"); ctx.clearRect(0, 0, 640, 300); const x = e => 50 + (e - 7) / 4 * 560, y = L => 260 - (L - 1.8) / 1.6 * 230;
    ctx.strokeStyle = "#d9d3c7"; ctx.beginPath(); ctx.moveTo(50, 260); ctx.lineTo(610, 260); ctx.moveTo(50, 260); ctx.lineTo(50, 20); ctx.stroke(); ctx.fillStyle = "#6b7680"; ctx.font = "12px sans-serif"; for (let e = 7; e <= 11; e++) ctx.fillText(`1e${e}`, x(e) - 10, 278); ctx.fillText("parameters N", 520, 295); ctx.fillText("fitted loss", 4, 16);
    const cols = ["#24668d", "#b8582a", "#3c8d5a", "#e3b23c"]; const mins = [];
    [18, 19, 20, 21].forEach((lc, i) => { const C = Math.pow(10, lc); ctx.strokeStyle = cols[i]; ctx.lineWidth = lc === Math.round(s.logC) ? 3 : 1.2; ctx.beginPath(); let first = true, best = null; for (let e = 7; e <= 11; e += 0.02) { const N = Math.pow(10, e), D = C / (6 * N); if (D < 1e6) continue; const L = chLoss(N, D); if (L > 3.4) continue; if (!best || L < best.L) best = { e, L }; first ? ctx.moveTo(x(e), y(L)) : ctx.lineTo(x(e), y(L)); first = false; } ctx.stroke(); if (best) { ctx.fillStyle = cols[i]; ctx.beginPath(); ctx.arc(x(best.e), y(best.L), 4, 0, 7); ctx.fill(); mins.push([lc, best]); } });
    ctx.lineWidth = 1;
    read.innerHTML = mins.map(([lc, b]) => `C=1e${lc}: N_opt ≈ 1e${b.e.toFixed(2)}, loss ${b.L.toFixed(3)}`).join("<br>") + `<br><span class="muted small">minima move right as the budget grows (Chinchilla Approach 2)</span>`;
  };
  root.append(el("div", { class: "widget" }, el("div", { class: "controls" }, slider("highlight budget 1e", 18, 21, 19, 1, v => { s.logC = v; draw(); }), read), cv)); draw();
}
Object.assign(WIDGETS, { "fixture:isoflop-sweep": isoflopSweep });

// ========================================================= lecture 7 widgets ==
// Byte accounting for the collectives (lecture_07 KP collective-bytes; lecture_07.py:L325, L363).
// S = the per-rank tensor the lecture passes to the benchmark (size_bytes = numel × element size).
// all-reduce: sent_bytes = 2·S·(W−1) (send + receive), reduce-scatter: S·(W−1) (no 2×); per-rank = sent_bytes / W.
// all-gather is the mirror of reduce-scatter (shards of S/W → S on every rank); all-to-all is not counted in the
// lecture (shown only as a transpose, L201), so its row uses the transpose: rank i keeps 1/W and ships (W−1)/W.
// pure model (MODELS below): bytes per collective for numel elements of `bytes` each on W ranks
function collectiveModel({ W, numel, bytes }) {
  const S = numel * bytes, f = (W - 1) / W, mib = 2 ** 20;
  return { S, oneTotal: S * (W - 1), oneRank: S * f, arTotal: 2 * S * (W - 1), arRank: 2 * S * f, arTotalMiB: 2 * S * (W - 1) / mib, arRankMiB: 2 * S * f / mib };
}
export function collectiveBytes(root, notice) {
  const LECT = Math.log2(100 * 1024 ** 2); // the lecture's num_elements = 100·1024², not a power of two
  const s = { W: 4, logN: LECT, bytes: 4 };
  const cv = el("canvas", { width: 640, height: 230 }); const read = el("div", { class: "readout" });
  const rows = () => {
    const m = collectiveModel({ W: s.W, numel: 2 ** s.logN, bytes: s.bytes }), S = m.S;
    return [
      { name: "all-gather (shards S/W → S)", total: m.oneTotal, rank: m.oneRank, formula: "S·(W−1)/W received per rank" },
      { name: "reduce-scatter (S → S/W)", total: m.oneTotal, rank: m.oneRank, formula: "S·(W−1) total, no 2×  (L363)" },
      { name: "all-reduce = RS + AG", total: m.arTotal, rank: m.arRank, formula: "2·S·(W−1) total  (L325)" },
      { name: "all-to-all (transpose)", total: m.oneTotal, rank: m.oneRank, formula: "S·(W−1)/W sent and received" },
    ].map(r => ({ ...r, S }));
  };
  const mib = x => `${fmt(x / 2 ** 20, 1)} MiB`;
  const draw = () => {
    const r = rows(), ctx = cv.getContext("2d"); ctx.clearRect(0, 0, 640, 230);
    const ref = 2 * 2 ** s.logN * 4, bw = 300; let y = 14; // fixed scale: 2·S at the fp32 payload, so bars move with W and with bytes/element
    ctx.strokeStyle = "#6b7680"; ctx.setLineDash([4, 4]); ctx.beginPath(); ctx.moveTo(220 + bw, 8); ctx.lineTo(220 + bw, 198); ctx.stroke(); ctx.setLineDash([]);
    ctx.fillStyle = "#6b7680"; ctx.font = "11px sans-serif"; ctx.fillText("2·S (fp32)", 220 + bw - 22, 210);
    for (const x of r) {
      const w = Math.max(2, bw * x.rank / ref);
      ctx.fillStyle = "#6b7680"; ctx.font = "12px sans-serif"; ctx.fillText(x.name, 10, y + 13);
      ctx.fillStyle = x.name.startsWith("all-reduce") ? "#b8582a" : "#24668d"; ctx.fillRect(220, y, w, 18);
      ctx.fillStyle = "#25313d"; ctx.fillText(`${mib(x.rank)} / rank`, 226 + w, y + 13); y += 52;
    }
    ctx.fillStyle = "#6b7680"; ctx.fillText("bars: bytes each rank sends + receives, fixed scale; they creep toward S or 2·S as W grows", 10, 226);
    const S = r[0].S;
    read.innerHTML = `S = ${Math.round(2 ** s.logN).toLocaleString()} elements × ${s.bytes} B = <b>${mib(S)}</b> per rank, W = ${s.W}<br>
      <pre>${r.map(x => `${x.name.padEnd(28)} total ${mib(x.total).padStart(12)}   per rank ${mib(x.rank).padStart(12)}   ${x.formula}`).join("\n")}</pre>
      all-reduce per rank = 2·S·(W−1)/W = <span class="big">${mib(2 * S * (s.W - 1) / s.W)}</span> → effective bandwidth ≈ 2·S/duration, independent of W and of ring vs tree (L330-L333)<br>
      <span class="muted small">provenance: fixture:collective-bytes · lecture_07.py:L322-L333 (all-reduce), L360-L370 (reduce-scatter); all-gather mirror from KP all-gather (L152, L276-L279); all-to-all bytes are not counted in the lecture (L201)</span>`;
  };
  const nfmt = v => `${Number.isInteger(v) ? `2^${v}` : "100·1024²"} = ${Math.round(2 ** v).toLocaleString()}`;
  const sN = slider("log2 num_elements", 10, 32, LECT, 1, v => { s.logN = v; draw(); }, nfmt);
  const lect = () => { s.logN = LECT; sN.querySelector("input").value = Math.round(LECT); sN.querySelector(".readout").textContent = nfmt(LECT); draw(); };
  root.append(el("div", { class: "widget" }, el("div", { class: "controls" },
    slider("world size W", 2, 64, 4, 1, v => { s.W = v; draw(); }),
    sN, el("button", { class: "ghost", onclick: lect }, "lecture: 100·1024² elements"),
    slider("bytes / element", 1, 4, 4, 1, v => { s.bytes = v; draw(); }, v => ({ 1: "fp8", 2: "bf16", 4: "fp32" })[v] || v),
    read), cv)); draw();
}

// ZeRO per-device memory (lecture_07 KP zero-sharding, basis=inference; ZeRO paper §5.1-§5.3, Fig. 1).
// Mixed-precision Adam: 2Ψ fp16 params + 2Ψ grads + KΨ (K = 12: fp32 master + two moments).
// stage 0: 16Ψ · stage 1: 4Ψ + 12Ψ/N · stage 2: 2Ψ + 14Ψ/N · stage 3: 16Ψ/N.
// pure model (MODELS below): per-device GB for ZeRO stages 0-3, Ψ parameters on N devices, K optimizer bytes/param
function zeroModel({ psi, N, K = 12 }) {
  return { stage0: (4 + K) * psi / 1e9, stage1: (4 * psi + K * psi / N) / 1e9, stage2: (2 * psi + (2 + K) * psi / N) / 1e9, stage3: (4 + K) * psi / N / 1e9 };
}
export function zeroMemory(root, notice) {
  const s = { psi: 13e9, logN: 4 };  // 7.5B at N = 64 is the zero-sharding predict
  const K = 12;
  const cv = el("canvas", { width: 640, height: 230 }); const read = el("div", { class: "readout" });
  const stages = () => { const z = zeroModel({ psi: s.psi, N: 2 ** s.logN, K }); return [
    { name: "stage 0 (DDP, replicated)", gb: z.stage0, f: "16Ψ" },
    { name: "stage 1 (optimizer state sharded)", gb: z.stage1, f: "4Ψ + 12Ψ/N" },
    { name: "stage 2 (+ gradients)", gb: z.stage2, f: "2Ψ + 14Ψ/N" },
    { name: "stage 3 (+ parameters, FSDP)", gb: z.stage3, f: "16Ψ/N" },
  ]; };
  const draw = () => {
    const st = stages(), ctx = cv.getContext("2d"); ctx.clearRect(0, 0, 640, 230);
    const max = st[0].gb; let y = 14;
    for (const x of st) {
      const w = Math.max(2, 360 * x.gb / max);
      ctx.fillStyle = "#6b7680"; ctx.font = "12px sans-serif"; ctx.fillText(x.name, 10, y + 13);
      ctx.fillStyle = x.name.startsWith("stage 3") ? "#3c8d5a" : "#24668d"; ctx.fillRect(240, y, w, 18);
      ctx.fillStyle = "#25313d"; ctx.fillText(`${fmt(x.gb, 2)} GB  = ${x.f}`, 246 + w, y + 13); y += 52;
    }
    ctx.fillStyle = "#6b7680"; ctx.fillText("bar = per-device bytes; only stage 3 has no N-independent term", 10, 226);
    const N = 2 ** s.logN;
    read.innerHTML = `Ψ = ${fmt(s.psi / 1e9, 2)} B parameters, N = ${N} devices, K = ${K} bytes (fp32 master copy + two Adam moments)<br>
      <span class="big">${fmt(st[0].gb, 1)} → ${fmt(st[1].gb, 1)} → ${fmt(st[2].gb, 1)} → ${fmt(st[3].gb, 2)} GB</span><br>
      <span class="muted small">provenance: fixture:zero-memory · lecture_07 KP zero-sharding (status=supplement, basis=inference) · zero_2019:§5.1-§5.3, Fig. 1 · lecture_07.py:L68, L152, L183, L387 name the collectives only</span>`;
  };
  root.append(el("div", { class: "widget" }, el("div", { class: "controls" },
    slider("parameters Ψ (B)", 0.5, 1000, 13, 0.5, v => { s.psi = v * 1e9; draw(); }),
    slider("log2 devices N", 0, 12, 4, 1, v => { s.logN = v; draw(); }, v => `N = ${2 ** v}`),
    read), cv)); draw();
}

Object.assign(WIDGETS, { "fixture:collective-bytes": collectiveBytes, "fixture:zero-memory": zeroMemory });

// ========================================================= lecture 8 widgets ==
// Communication per step under data / tensor / pipeline parallelism.
// The formulas below are the ones printed on the lecture_08 slides:
//   DDP: "transmits 2x # params every batch" (p15, p21 'Communication cost 2* # params'); ZeRO-3: "2 all gather (#param), 1 reduce-scatter (#param)" (p25)
//   pipeline: "bsh point-to-point communication per microbatch" (p43); tensor: "8bsh (n_devices−1)/n_devices per layer and all-reduce" (p43)
//   bubble ratio (n_stages−1)/n_micro (p34).
// Standard Megatron accounting fills the gaps the slides leave: counts are in parameters/activations, so bytes = count × dtype bytes;
// the pipeline total multiplies the per-microbatch figure by the number of microbatches and stage boundaries.
// pure model (MODELS below); bytes = 1 gives element counts
function parallelismModel(s) {
  const act = s.b * s.seq * s.h * s.bytes;                                      // one micro-batch activation b·s·h
  const dp = (s.zero === 3 ? 3 : 2) * s.psi * s.bytes;                            // 2·#params (DDP, ZeRO-1) or 3·#params (ZeRO-3)
  const tpLayer = 8 * act * (s.tp - 1) / s.tp;                                    // 8bsh (n−1)/n per layer
  const tp = tpLayer * s.L * s.m;  // every micro-batch runs every layer's all-reduces
  const ppMicro = act;                                                             // bsh per microbatch per boundary
  const pp = 2 * ppMicro * s.m * (s.pp - 1);  // activation forward + gradient backward per boundary, matching TP's 8bsh which counts both
  return { act, dp, tpLayer, tp, ppMicro, pp, bubble: (s.pp - 1) / s.m, tpOverPp: tp / pp, tpLayerOverAct: tpLayer / act };
}
export function parallelismComm(root, notice) {
  const s = { psi: 70e9, b: 8, seq: 4096, h: 8192, L: 80, bytes: 2, tp: 8, pp: 4, m: 16, zero: 0 };
  const cv = el("canvas", { width: 640, height: 200 }); const read = el("div", { class: "readout" });
  const calc = () => parallelismModel(s);
  const gb = x => `${fmt(x / 1e9, 2)} GB`;
  const draw = () => {
    const c = calc(), ctx = cv.getContext("2d"); ctx.clearRect(0, 0, 640, 200);
    const bars = [[`data parallel${s.zero === 3 ? " (ZeRO-3)" : s.zero === 1 ? " (ZeRO-1)" : " (DDP)"}`, c.dp, "#24668d"], [`tensor parallel (TP=${s.tp}, ${s.L} layers)`, c.tp, "#b8582a"], [`pipeline parallel (PP=${s.pp}, ${s.m} micro)`, c.pp, "#3c8d5a"]];
    const max = Math.max(...bars.map(b => b[1])); let y = 16;
    for (const [n, v, col] of bars) { const w = Math.max(2, 340 * v / max); ctx.fillStyle = "#6b7680"; ctx.font = "12px sans-serif"; ctx.fillText(n, 10, y + 13); ctx.fillStyle = col; ctx.fillRect(250, y, w, 18); ctx.fillStyle = "#25313d"; ctx.fillText(gb(v), 256 + w, y + 13); y += 56; }
    ctx.fillStyle = "#6b7680"; ctx.fillText("bytes per training step (all micro-batches, forward + backward); counts × dtype bytes", 10, 196);
    read.innerHTML = `<pre>DP   ${s.zero === 3 ? "3" : "2"}·#params·bytes          = ${s.zero === 3 ? "3" : "2"}·${fmt(s.psi, 0)}·${s.bytes}            = ${gb(c.dp)}
TP   8·b·s·h·(n−1)/n per layer = 8·${fmt(c.act, 0)}·${s.tp - 1}/${s.tp} = ${gb(c.tpLayer)} × ${s.L} layers × ${s.m} micro = ${gb(c.tp)}
PP   b·s·h per micro-batch     = ${gb(c.ppMicro)} × 2 (fwd + bwd) × ${s.m} micro × ${s.pp - 1} boundaries = ${gb(c.pp)}   bubble (p−1)/m = ${c.bubble.toFixed(2)}</pre>
      <span class="muted small">provenance: fixture:parallelism-comm · lecture_08.pdf:p15, p21 (DDP 2·#params), p25 (ZeRO-3: 2 all-gather + 1 reduce-scatter), p34 (bubble), p43 (pipeline bsh / tensor 8bsh·(n−1)/n). × dtype bytes, × layers, × micro-batches × boundaries are standard Megatron accounting (megatron_lm_2019), not stated on the slides. lecture_07.py:L325 would add a (W−1)/W factor to DP.</span>`;
  };
  root.append(el("div", { class: "widget" }, el("div", { class: "controls" },
    slider("parameters (B)", 1, 1000, 70, 1, v => { s.psi = v * 1e9; draw(); }),
    slider("micro-batch b", 1, 64, 8, 1, v => { s.b = v; draw(); }), slider("sequence s", 512, 32768, 4096, 512, v => { s.seq = v; draw(); }),
    slider("hidden h", 1024, 16384, 8192, 1024, v => { s.h = v; draw(); }), slider("layers", 4, 128, 80, 4, v => { s.L = v; draw(); }),
    slider("bytes / element", 1, 4, 2, 1, v => { s.bytes = v; draw(); }, v => ({ 1: "fp8", 2: "bf16", 4: "fp32" })[v] || v),
    slider("ZeRO stage", 0, 3, 0, 1, v => { s.zero = v === 2 ? 1 : v; draw(); }, v => v === 0 ? "0 (DDP)" : v === 3 ? "3 (FSDP)" : "1"),
    slider("TP devices", 2, 16, 8, 1, v => { s.tp = v; draw(); }), slider("PP stages", 2, 32, 4, 1, v => { s.pp = v; draw(); }), slider("micro-batches m", 1, 128, 16, 1, v => { s.m = v; draw(); }),
    read), cv)); draw();
}
Object.assign(WIDGETS, { "fixture:parallelism-comm": parallelismComm });

// ========================================================= lecture 4 widgets ==
// KV cache bytes. Formula is the executable one in lecture_10.py:L299-L302 (kv_cache_size_per_seq = S·(K·H)·L·2·2, × B);
// the GQA reduction N/K is lecture_10.py:L378-L387. Used by the lecture 10 widgets below.
const KV_BYTES = (L, K, H, bytes, S, B) => 2 * L * K * H * bytes * S * B;

// Top-k token-choice routing with a capacity factor. Capacity per expert = ceil(c · k · T / E) (Switch Transformer / Fedus et al. 2022,
// cited on lecture_04.pdf:p31; "token dropping from routing happens at a batch level", p47). Assignments come from a seeded PRNG with a
// skew knob, so the run is deterministic; a token whose chosen expert is full is dropped (passes through the residual).
const mulberry32 = seed => () => { seed |= 0; seed = seed + 0x6D2B79F5 | 0; let t = Math.imul(seed ^ seed >>> 15, 1 | seed); t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t; return ((t ^ t >>> 14) >>> 0) / 4294967296; };
export function moeRouting(root, notice) {
  const s = { T: 32, E: 8, k: 1, c: 1.25, skew: 0.5, step: 0 };
  const cv = el("canvas", { width: 640, height: 240 }); const read = el("div", { class: "readout" });
  const plan = () => { // deterministic routing choices for all T tokens
    const rnd = mulberry32(42 + s.T * 7 + s.E * 13 + s.k); const weights = Array.from({ length: s.E }, (_, e) => Math.pow(1 - s.skew, e));
    return Array.from({ length: s.T }, () => { const w = [...weights]; const picks = []; for (let j = 0; j < s.k; j++) { const tot = w.reduce((a, x) => a + x, 0); let r = rnd() * tot, e = 0; while (e < s.E - 1 && r > w[e]) { r -= w[e]; e++; } picks.push(e); w[e] = 0; } return picks; });
  };
  const cap = () => Math.ceil(s.c * s.k * s.T / s.E);
  const sim = () => { const p = plan(), C = cap(), load = Array(s.E).fill(0), dropped = []; for (let t = 0; t < Math.min(s.step, s.T); t++) for (const e of p[t]) { if (load[e] < C) load[e]++; else dropped.push([t, e]); } return { p, C, load, dropped }; };
  const draw = () => {
    const { p, C, load, dropped } = sim(), ctx = cv.getContext("2d"); ctx.clearRect(0, 0, 640, 240);
    const bw = Math.min(60, 560 / s.E), maxH = 160, unit = maxH / Math.max(C * 1.3, 1);
    for (let e = 0; e < s.E; e++) { const x = 40 + e * bw; ctx.strokeStyle = "#d9d3c7"; ctx.strokeRect(x, 200 - C * unit, bw - 6, C * unit); ctx.fillStyle = load[e] >= C ? "#b8582a" : "#24668d"; ctx.fillRect(x, 200 - load[e] * unit, bw - 6, load[e] * unit); ctx.fillStyle = "#6b7680"; ctx.font = "11px sans-serif"; ctx.fillText(`E${e}`, x, 214); ctx.fillText(`${load[e]}/${C}`, x, 228); }
    ctx.fillStyle = "#6b7680"; ctx.font = "12px sans-serif"; ctx.fillText("outline = capacity per expert; red = full (further tokens dropped)", 40, 20);
    const t = Math.min(s.step, s.T), last = t > 0 ? `token ${t - 1} → expert${s.k > 1 ? "s" : ""} ${p[t - 1].join(", ")}${dropped.some(d => d[0] === t - 1) ? " (dropped: full)" : ""}` : "press next token";
    read.innerHTML = `capacity = ceil(c·k·T/E) = ceil(${s.c}·${s.k}·${s.T}/${s.E}) = <b>${C}</b> slots per expert · routed ${t}/${s.T} tokens · ${last}<br>
      <span class="big">${dropped.length} dropped</span> assignments of ${t * s.k} · total slots ${C * s.E} ≥ ${s.k * s.T} = k·T ${C * s.E >= s.k * s.T ? "(enough if perfectly balanced)" : ""}<br>
      <span class="muted small">provenance: fixture:moe-routing · lecture_04.pdf:p31 (top-k token-choice routing), p47 (token dropping at the batch level), p15 (FLOPs fixed by k); capacity formula from Fedus et al. 2022 (Switch Transformer) as cited on those slides. Routing draws are a seeded PRNG for the demo, not a learned router.</span>`;
  };
  const reset = () => { s.step = 0; draw(); };
  root.append(el("div", { class: "widget" }, el("div", { class: "controls" },
    slider("tokens T", 8, 128, 32, 8, v => { s.T = v; reset(); }), slider("experts E", 2, 16, 8, 1, v => { s.E = v; reset(); }), slider("top-k", 1, 2, 1, 1, v => { s.k = v; reset(); }),
    slider("capacity factor c", 0.5, 2.5, 1.25, 0.25, v => { s.c = v; reset(); }, v => v.toFixed(2)), slider("router skew", 0, 0.9, 0.5, 0.1, v => { s.skew = v; reset(); }, v => v.toFixed(1)),
    el("div", { style: "margin-top:8px" }, el("button", { onclick: () => { s.step = Math.min(s.step + 1, s.T); draw(); } }, "next token"), " ", el("button", { onclick: () => { s.step = s.T; draw(); } }, "all"), " ", el("button", { class: "ghost", onclick: reset }, "reset")),
    read), cv)); draw();
}

Object.assign(WIDGETS, { "fixture:moe-routing": moeRouting });

// ========================================================= lecture 10 widgets ==
// Decode is memory-bound: each step reads all parameters and the KV cache (lecture_10.py:L297-L308):
//   memory = B·kv_cache_size_per_seq + parameter_size; latency = memory / memory_bandwidth; throughput = B / latency.
// H100: memory_bandwidth 3.35e12 (L322), h100_memory = 80e9 (L351). Defaults are llama2_13b_config (L313-L323) with its 13B parameters.
// pure model (MODELS below): ceiling = bandwidth / KV bytes per sequence, the large-B throughput limit
function decodeModel({ params, bytes = 2, S, B, L = 40, K = 40, H = 128, bw = 3.35e12 }) {
  const p = params * bytes, kv = KV_BYTES(L, K, H, bytes, S, B), mem = p + kv, lat = mem / bw;
  return { p, kv, mem, lat, tps: B / lat, ceiling: bw / (kv / B) };
}
export function decodeBound(root, notice) {
  const s = { params: 13e9, bytes: 2, S: 1024, B: 1, L: 40, K: 40, H: 128, bw: 3.35e12 };
  const cv = el("canvas", { width: 640, height: 270 }); const read = el("div", { class: "readout" });
  const calc = () => decodeModel(s);
  const draw = () => {
    const c = calc(), ctx = cv.getContext("2d"); ctx.clearRect(0, 0, 640, 270);
    // bytes bar on a fixed absolute scale (80 GB, or more if the parameters alone need it), so the parameter segment does not move with B
    const w = 500, scale = Math.max(80e9, 1.25 * c.p), px = w / scale, pw = Math.min(w, c.p * px), kw = Math.min(w - pw, c.kv * px);
    ctx.fillStyle = "#24668d"; ctx.fillRect(40, 30, pw, 28); ctx.fillStyle = "#b8582a"; ctx.fillRect(40 + pw, 30, kw, 28);
    if (c.mem > scale) { ctx.fillStyle = "#b8582a"; ctx.font = "12px sans-serif"; ctx.fillText("→", 40 + w + 4, 49); }
    ctx.strokeStyle = "#25313d"; ctx.setLineDash([3, 3]); ctx.beginPath(); ctx.moveTo(40 + 80e9 * px, 26); ctx.lineTo(40 + 80e9 * px, 62); ctx.stroke(); ctx.setLineDash([]);
    ctx.fillStyle = "#6b7680"; ctx.font = "11px sans-serif"; ctx.fillText("80 GB", 40 + 80e9 * px + 4, 70);
    // throughput and latency against B (1..512) at the current shape; B* = parameter bytes / KV bytes per sequence
    const kvs = c.kv / s.B, thr = B => B * kvs / (c.p + B * kvs), latB = B => (c.p + B * kvs) / s.bw, lmax = latB(512);
    const gx = B => 50 + (B - 1) / 511 * 560, gy = f => 250 - f * 120;
    ctx.strokeStyle = "#d9d3c7"; ctx.beginPath(); ctx.moveTo(50, 130); ctx.lineTo(50, 250); ctx.lineTo(610, 250); ctx.stroke();
    ctx.strokeStyle = "#24668d"; ctx.setLineDash([4, 4]); ctx.beginPath(); ctx.moveTo(50, gy(1)); ctx.lineTo(610, gy(1)); ctx.stroke(); ctx.setLineDash([]);
    [[thr, "#24668d"], [B => latB(B) / lmax, "#b8582a"]].forEach(([f, col]) => { ctx.strokeStyle = col; ctx.lineWidth = 2; ctx.beginPath(); for (let B = 1; B <= 512; B += 1) B === 1 ? ctx.moveTo(gx(B), gy(f(B))) : ctx.lineTo(gx(B), gy(f(B))); ctx.stroke(); ctx.lineWidth = 1; ctx.fillStyle = col; ctx.beginPath(); ctx.arc(gx(s.B), gy(f(s.B)), 4, 0, 7); ctx.fill(); });
    const bstar = c.p / kvs;
    if (bstar >= 1 && bstar <= 512) { ctx.strokeStyle = "#3c8d5a"; ctx.setLineDash([3, 3]); ctx.beginPath(); ctx.moveTo(gx(bstar), 130); ctx.lineTo(gx(bstar), 250); ctx.stroke(); ctx.setLineDash([]); ctx.fillStyle = "#3c8d5a"; ctx.fillText(`B* = ${fmt(bstar, 0)}`, Math.min(gx(bstar) + 4, 570), 142); }
    ctx.fillStyle = "#6b7680"; ctx.fillText("1", 46, 262); ctx.fillText("512", 592, 262); ctx.fillText("batch B →", 280, 262);
    ctx.fillStyle = "#24668d"; ctx.fillText(`throughput ÷ ceiling bandwidth / KV-per-seq = ${fmt(s.bw / kvs, 0)} tokens/s (dashed)`, 56, 124);
    ctx.fillStyle = "#b8582a"; ctx.fillText(`latency ÷ ${fmt(lmax * 1e3, 1)} ms (its value at B = 512)`, 360, 240);
    ctx.fillStyle = "#6b7680"; ctx.font = "12px sans-serif"; ctx.fillText(`bytes read per decode step: parameters ${fmt(c.p / 1e9, 1)} GB (blue) + KV cache ${fmt(c.kv / 1e9, 2)} GB (red)`, 40, 20);
    ctx.fillStyle = c.mem > 80e9 ? "#b8582a" : "#3c8d5a"; ctx.fillText(c.mem > 80e9 ? `memory ${fmt(c.mem / 1e9, 1)} GB > h100_memory 80 GB: does not fit (L351)` : `memory ${fmt(c.mem / 1e9, 1)} GB fits in 80 GB`, 40, 80);
    ctx.fillStyle = "#6b7680"; ctx.fillText(`latency = memory / bandwidth = ${fmt(c.lat * 1e3, 2)} ms per step`, 40, 104);
    read.innerHTML = `<span class="big">≤ ${fmt(c.tps, 1)} tokens/s</span> = B / latency = ${s.B} / (${fmt(c.mem, 2)} B ÷ ${fmt(s.bw, 2)} B/s)<br>
      per-request latency ${fmt(c.lat * 1e3, 2)} ms/token (${fmt(1 / c.lat, 1)} tokens/s for one stream) · larger B amortizes the parameter read but grows the KV term (L355-L357)<br>
      B* = parameter bytes / KV bytes per sequence = ${fmt(c.p / (c.kv / s.B), 1)}: below it throughput rises almost linearly in B, above it the KV read dominates and throughput flattens toward ${fmt(s.bw / (c.kv / s.B), 0)} tokens/s while latency keeps growing linearly<br>
      <span class="muted small">provenance: fixture:decode-bound · lecture_10.py:L297-L308 (memory, latency, throughput), L299-L302 (KV cache), L322 (3.35e12 B/s), L351 (80e9), L335 (assumes perfect overlap, no overhead), L333 (generation is memory-bound). Upper bound only: compute and overhead ignored.</span>`;
  };
  root.append(el("div", { class: "widget" }, el("div", { class: "controls" },
    slider("parameters (B)", 1, 500, 13, 1, v => { s.params = v * 1e9; draw(); }), slider("bytes / element", 1, 4, 2, 1, v => { s.bytes = v; draw(); }, v => ({ 1: "fp8 / int8", 2: "bf16", 4: "fp32" })[v] || v),
    slider("context S (tokens)", 128, 131072, 1024, 128, v => { s.S = v; draw(); }), slider("batch B", 1, 512, 1, 1, v => { s.B = v; draw(); }),
    slider("layers L", 1, 128, 40, 1, v => { s.L = v; draw(); }), slider("kv heads K", 1, 128, 40, 1, v => { s.K = v; draw(); }), slider("head dim H", 32, 256, 128, 32, v => { s.H = v; draw(); }),
    slider("bandwidth (TB/s)", 0.5, 10, 3.35, 0.05, v => { s.bw = v * 1e12; draw(); }, v => v.toFixed(2)),
    read), cv)); draw();
}

// KV cache growth with generated tokens, against GPU memory left after the parameters (lecture_10.py:L299-L302, L351).
// pure model (MODELS below): KV bytes per token per sequence 2·L·K·H·bytes; fitB = sequences of `prompt` tokens that fit
function kvModel({ params = 13e9, bytes = 2, B = 1, L = 40, K = 40, H = 128, prompt = 1024, gpuGB = 80 }) {
  const pbytes = params * bytes, budget = gpuGB * 1e9 - pbytes, perTokSeq = KV_BYTES(L, K, H, bytes, 1, 1);
  return { pbytes, budget, perTokSeq, perTok: perTokSeq * B, fitB: budget / (perTokSeq * prompt) };
}
export function kvCacheGrowth(root, notice) {
  const s = { params: 13e9, bytes: 2, B: 16, L: 40, K: 40, H: 128, prompt: 1024, maxGen: 8192, gpuGB: 80 };  // B = 64 at S = 1024 is the kv-cache-size predict
  const cv = el("canvas", { width: 640, height: 260 }); const read = el("div", { class: "readout" });
  const draw = () => {
    const ctx = cv.getContext("2d"); ctx.clearRect(0, 0, 640, 260);
    const { pbytes, budget, perTok } = kvModel(s);
    const kvAt = g => perTok * (s.prompt + g), top = Math.max(kvAt(s.maxGen), s.gpuGB * 1e9) * 1.05;
    const x = g => 50 + g / s.maxGen * 560, y = v => 220 - v / top * 190;
    ctx.strokeStyle = "#d9d3c7"; ctx.beginPath(); ctx.moveTo(50, 220); ctx.lineTo(610, 220); ctx.moveTo(50, 220); ctx.lineTo(50, 20); ctx.stroke();
    const n = 32; for (let i = 0; i <= n; i++) { const g = s.maxGen * i / n, v = kvAt(g); ctx.fillStyle = v > budget ? "#b8582a" : "#24668d"; ctx.fillRect(x(g) - 6, y(v), 12, 220 - y(v)); }
    if (budget > 0) { ctx.strokeStyle = "#3c8d5a"; ctx.setLineDash([5, 4]); ctx.beginPath(); ctx.moveTo(50, y(budget)); ctx.lineTo(610, y(budget)); ctx.stroke(); ctx.setLineDash([]); ctx.fillStyle = "#3c8d5a"; ctx.font = "12px sans-serif"; ctx.fillText(`${s.gpuGB} GB − parameters ${fmt(pbytes / 1e9, 1)} GB = ${fmt(budget / 1e9, 1)} GB for KV`, 56, y(budget) - 6); }
    ctx.fillStyle = "#6b7680"; ctx.font = "12px sans-serif"; ctx.fillText("generated tokens →", 500, 240); ctx.fillText("0", 46, 236); ctx.fillText(String(s.maxGen), 590, 236); ctx.fillText("KV cache bytes (prompt + generated, × batch)", 56, 16);
    const limit = budget > 0 ? Math.floor(budget / perTok) - s.prompt : -s.prompt;
    read.innerHTML = `KV grows ${fmt(perTok / 1e6, 2)} MB per generated token (2·L·K·H·bytes·B) from ${fmt(kvAt(0) / 1e9, 2)} GB at the prompt<br>
      <span class="big">${budget <= 0 ? "parameters alone exceed the GPU" : limit < 0 ? `prompt alone overflows by ${fmt(-limit, 0)} tokens` : `overflows after ${fmt(limit, 0)} generated tokens`}</span>${budget > 0 && limit >= 0 ? ` (max context ${fmt(s.prompt + limit, 0)} tokens at B = ${s.B})` : ""}<br>
      <span class="muted small">provenance: fixture:kv-cache-growth · lecture_10.py:L299-L302 (KV cache size), L351 (h100_memory = 80e9, "doesn't fit"), L356 (larger KV cache with B), L313-L323 (Llama 2 13B defaults).</span>`;
  };
  root.append(el("div", { class: "widget" }, el("div", { class: "controls" },
    slider("parameters (B)", 1, 500, 13, 1, v => { s.params = v * 1e9; draw(); }), slider("bytes / element", 1, 4, 2, 1, v => { s.bytes = v; draw(); }, v => ({ 1: "fp8 / int8", 2: "bf16", 4: "fp32" })[v] || v),
    slider("batch B", 1, 512, 16, 1, v => { s.B = v; draw(); }), slider("layers L", 1, 128, 40, 1, v => { s.L = v; draw(); }), slider("kv heads K", 1, 128, 40, 1, v => { s.K = v; draw(); }), slider("head dim H", 32, 256, 128, 32, v => { s.H = v; draw(); }),
    slider("prompt tokens", 0, 32768, 1024, 128, v => { s.prompt = v; draw(); }), slider("max generated", 256, 65536, 8192, 256, v => { s.maxGen = v; draw(); }), slider("GPU memory (GB)", 16, 192, 80, 8, v => { s.gpuGB = v; draw(); }),
    read), cv)); draw();
}

Object.assign(WIDGETS, { "fixture:decode-bound": decodeBound, "fixture:kv-cache-growth": kvCacheGrowth });

// ========================================================= lecture 15 widgets ==
// Bradley-Terry preference probability (lecture_15/fixtures/bradley_terry.json)
// pure model (MODELS below): p = σ(Δ), loss = −log p, ∂loss/∂Δ = −(1−p)
function btModel({ rw, rl, shift = 0 }) { const d = (rw + shift) - (rl + shift), p = 1 / (1 + Math.exp(-d)); return { d, p, loss: -Math.log(p), grad: -(1 - p) }; }
export function bradleyTerry(root, notice) {
  const s = { rw: 1.0, rl: 0.0, shift: 0.0 };
  const read = el("div", { class: "readout" });
  const draw = () => { const { d, p } = btModel(s); read.innerHTML = `Δ = r(y_w) − r(y_l) = <b>${d.toFixed(2)}</b> nats<br><span class="big">p(chosen ≻ rejected) = σ(Δ) = ${p.toFixed(4)}</span>loss = −log p = ${(-Math.log(p)).toFixed(4)} · ∂loss/∂Δ = −(1−p) = ${(-(1 - p)).toFixed(4)}<br><span class="muted small">the shared shift never moves p: only the difference is identified · lecture_15.pdf:p52-p57</span>`; };
  const presets = [["tie", { rw: 0, rl: 0, shift: 0 }], ["1 nat gap", { rw: 1, rl: 0, shift: 0 }], ["3 nat gap", { rw: 3, rl: 0, shift: 0 }], ["mislabeled", { rw: 0, rl: 2, shift: 0 }]];
  const sw = slider("r(y_w) chosen", -10, 10, 1, 0.1, v => { s.rw = v; draw(); }), sl = slider("r(y_l) rejected", -10, 10, 0, 0.1, v => { s.rl = v; draw(); }), ss = slider("shared shift (added to both)", -10, 10, 0, 0.5, v => { s.shift = v; draw(); });
  const apply = p => { Object.assign(s, p); setSlider(sw, s.rw); setSlider(sl, s.rl); setSlider(ss, s.shift); draw(); };
  root.append(el("div", { class: "widget" }, el("div", { class: "controls" }, sw, sl, ss, el("div", { style: "margin-top:8px" }, ...presets.map(([n, p]) => el("button", { onclick: () => apply(p) }, n)))), read)); draw();
}
// DPO implicit reward r̂ = β·(log π_θ − log π_ref) (lecture_15/fixtures/dpo_implicit_reward.json)
export function dpoImplicitReward(root, notice) {
  const s = { beta: 0.1, lw: -20, rw: -20, ll: -20, rl: -20 };
  const read = el("div", { class: "readout" });
  const draw = () => { const rhw = s.beta * (s.lw - s.rw), rhl = s.beta * (s.ll - s.rl), m = rhw - rhl, p = 1 / (1 + Math.exp(-m)); read.innerHTML = `r̂(y_w) = β·(log π_θ − log π_ref) = <b>${rhw.toFixed(3)}</b> · r̂(y_l) = <b>${rhl.toFixed(3)}</b><br><span class="big">margin ${m.toFixed(3)} → p = σ(margin) = ${p.toFixed(4)}</span>loss = −log p = ${(-Math.log(p)).toFixed(4)} · gradient weight β(1−p) = ${(s.beta * (1 - p)).toFixed(4)}<br><span class="muted small">the step is β(1−p)·[∇log π(y_w) − ∇log π(y_l)]: large when the model ranks the pair wrongly, vanishing when it is already right · lecture_15.pdf:p55-p58</span>`; };
  root.append(el("div", { class: "widget" }, el("div", { class: "controls" }, slider("β", 0.01, 1, 0.1, 0.01, v => { s.beta = v; draw(); }), slider("log π_θ(y_w)", -60, 0, -20, 0.5, v => { s.lw = v; draw(); }), slider("log π_ref(y_w)", -60, 0, -20, 0.5, v => { s.rw = v; draw(); }), slider("log π_θ(y_l)", -60, 0, -20, 0.5, v => { s.ll = v; draw(); }), slider("log π_ref(y_l)", -60, 0, -20, 0.5, v => { s.rl = v; draw(); })), read)); draw();
}
Object.assign(WIDGETS, { "fixture:bradley-terry": bradleyTerry, "fixture:dpo-implicit-reward": dpoImplicitReward });

// ================================================= lectures 11-13 widgets ==
const frame = (ctx, W, Hh, x0, y0, x1, y1) => { ctx.clearRect(0, 0, W, Hh); ctx.strokeStyle = "#d9d3c7"; ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(x0, y0); ctx.lineTo(x0, y1); ctx.lineTo(x1, y1); ctx.stroke(); };

// Filter retention (lecture_13). Every size is the professor's text; retained fraction = kept / pool is one division (the KP's arithmetic).
export function filterRetention(root, notice) {
  const ROWS = [
    { name: "C4", filter: "rules", pool: 1.4e12, kept: 156e9, ref: "L387, L396" },
    { name: "RefinedWeb", filter: "released share (after rules + dedup)", pool: 5e12, kept: 600e9, ref: "L509-L511 (\"Released 600B (out of 5T)\")" },
    { name: "FineWebEdu / DCLM per Nemotron-CC", filter: "classifier", frac: 0.10, ref: "L561 \"remove 90% of data\"" },
    { name: "DCLM-baseline", filter: "classifier", pool: 240e12, kept: 3.8e12, ref: "L542, L552" },
  ];
  const ABS = [["FineWeb", 15e12, "L520"], ["Nemotron-CC", 6.3e12, "L573 (HQ subset 1.1T)"], ["DCLM-baseline", 3.8e12, "L552"], ["RefinedWeb (released)", 600e9, "L511"], ["C4", 156e9, "L396"]];
  const s = { budget: 15e12, guess: 0.05, reveal: false };
  const frac = r => r.frac ?? r.kept / r.pool;
  const cv = el("canvas", { width: 640, height: 250 }); const read = el("div", { class: "readout" });
  const draw = () => {
    const ctx = cv.getContext("2d"); ctx.clearRect(0, 0, 640, 250); ctx.font = "12px sans-serif";
    ctx.fillStyle = "#6b7680"; ctx.fillText("retained fraction of the pool (bar length, 0 – 15%)", 10, 16);
    ROWS.forEach((r, i) => {
      const y = 30 + i * 40, f = frac(r), w = f / 0.15 * 380;
      ctx.fillStyle = "#25313d"; ctx.fillText(r.name, 10, y + 14); ctx.fillStyle = "#6b7680"; ctx.fillText(r.filter, 10, y + 28);
      if (s.reveal) { ctx.fillStyle = r.filter === "classifier" ? "#b8582a" : "#24668d"; ctx.fillRect(230, y + 2, w, 20); ctx.fillStyle = "#25313d"; ctx.fillText(`${(100 * f).toFixed(1)}%`, 236 + w, y + 17); }
      else { ctx.strokeStyle = "#d9d3c7"; ctx.strokeRect(230, y + 2, 380, 20); ctx.fillStyle = "#6b7680"; ctx.fillText("?", 236, y + 17); }
    });
    const gx = 230 + s.guess / 0.15 * 380; ctx.strokeStyle = "#e3b23c"; ctx.lineWidth = 2; ctx.beginPath(); ctx.moveTo(gx, 26); ctx.lineTo(gx, 190); ctx.stroke(); ctx.lineWidth = 1;
    ctx.fillStyle = "#b07d12"; ctx.fillText(`your guess for DCLM: ${(100 * s.guess).toFixed(1)}%`, Math.min(gx + 4, 480), 204);
    ctx.fillStyle = "#6b7680"; ctx.fillText("Gopher (L486): '10.5 TB of text (though Gopher only trained on 300B tokens - 12%)' — bytes vs tokens, a training-use share, not drawn", 10, 236);
    const dclm = ROWS[3];
    read.innerHTML = `${s.reveal ? `DCLM keeps 3.8T ÷ 240T = <span class="big">${(100 * frac(dclm)).toFixed(2)}%</span>your guess was ${(100 * s.guess).toFixed(1)}% · C4's rules kept ${(100 * frac(ROWS[0])).toFixed(1)}%; RefinedWeb released ${(100 * frac(ROWS[1])).toFixed(1)}% of its deduplicated 5T (a release choice, not a filter rate); the classifier keeps about 1.6%` : "guess DCLM's retained fraction, then reveal"}<br>
      absolute tokens still grow because the pools grew: ${ABS.map(([n, t]) => `${n} ${fmt(t / 1e12, 2)}T`).join(" · ")}<br>
      training budget ${fmt(s.budget / 1e12, 1)}T tokens needs: ${ABS.map(([n, t]) => `${n} ×${(s.budget / t).toFixed(1)}`).join(" · ")} epochs (Llama 3 15T, Qwen3 36T: L574)<br>
      <span class="muted small">provenance: fixture:filter-retention-ratios · lecture_13.py ${[...ROWS.map(r => `${r.name} ${r.ref}`), ...ABS.slice(0, 2).map(([n, , r]) => `${n} ${r}`)].join("; ")}; L574 (Llama 3 15T, Qwen3 36T). Nemotron-CC's "remove 90%" (10% kept) and DCLM's own 240T → 3.8T (1.6% kept) are measured from different starting points; the lecture does not reconcile them.</span>`;
  };
  const sg = slider("your guess: DCLM keeps what % of its pool?", 0, 15, 5, 0.1, v => { s.guess = v / 100; draw(); }, v => `${v}%`);
  const sbud = slider("training budget (T tokens)", 0.5, 40, 15, 0.5, v => { s.budget = v * 1e12; draw(); });
  root.append(el("div", { class: "widget" }, el("div", { class: "controls" }, sg,
    el("div", { style: "margin-top:8px" }, el("button", { onclick: () => { s.reveal = true; draw(); } }, "reveal"), " ", el("button", { class: "ghost", onclick: () => { s.reveal = false; draw(); } }, "hide")),
    sbud, el("div", { style: "margin-top:8px" }, el("button", { onclick: () => { s.budget = 15e12; setSlider(sbud, 15); draw(); } }, "Llama 3: 15T"), " ", el("button", { onclick: () => { s.budget = 36e12; setSlider(sbud, 36); draw(); } }, "Qwen3: 36T")),
    read), cv)); draw();
}

// Perplexity per token vs per byte. lecture_12.py:L62 defines (1/p(D))^(1/|D|) with |D| in tokens; converting to bytes with the
// lecture_01 compression ratio (bytes/token, lecture_01.py:L567-L571) is the KP's author-inference, not lecture content.
// ln ppl_token = L (nats/token); per-byte loss = L / c; bits per byte = L / (c·ln 2); ppl_byte = e^{L/c}.
export function perplexityPerByte(root, notice) {
  const s = { pplA: 8, cA: 2.5, cB: 4, bytes: 2500, tie: false, lossB: 3 };  // tie on at c_B = 5 would be the perplexity-definition predict
  const cv = el("canvas", { width: 640, height: 280 }); const read = el("div", { class: "readout" });
  const calc = () => {
    const LA = Math.log(s.pplA), total = LA * s.bytes / s.cA; // total −log p(D) in nats, fixed by A
    const LB = s.tie ? total / (s.bytes / s.cB) : s.lossB;
    const m = (L, c) => ({ L, c, tokens: s.bytes / c, ppl: Math.exp(L), perByte: L / c, bpb: L / (c * Math.LN2), pplByte: Math.exp(L / c) });
    return { A: m(LA, s.cA), B: m(LB, s.cB), total };
  };
  const draw = () => {
    const o = calc(), ctx = cv.getContext("2d"), x = c => 50 + (c - 0.5) / 6 * 560;
    const yMax = Math.max(o.A.ppl, o.B.ppl, 2) * 1.15, y = v => 250 - Math.log(v) / Math.log(yMax) * 220;
    frame(ctx, 640, 280, 50, 20, 610, 250); ctx.font = "12px sans-serif"; ctx.fillStyle = "#6b7680";
    for (const c of [1, 2, 3, 4, 5, 6]) ctx.fillText(String(c), x(c) - 3, 266); ctx.fillText("bytes per token c (tokenizer granularity)", 380, 278);
    ctx.fillText("perplexity (log scale)", 56, 14);
    ctx.strokeStyle = "#24668d"; ctx.lineWidth = 2; ctx.beginPath(); // same per-byte loss as A, every tokenizer granularity
    for (let i = 0; i <= 200; i++) { const c = 0.5 + 6 * i / 200, v = Math.exp(o.A.perByte * c); i ? ctx.lineTo(x(c), y(Math.min(v, yMax))) : ctx.moveTo(x(c), y(Math.min(v, yMax))); } ctx.stroke();
    ctx.strokeStyle = "#3c8d5a"; ctx.beginPath(); ctx.moveTo(x(0.5), y(o.A.pplByte)); ctx.lineTo(x(6.5), y(o.A.pplByte)); ctx.stroke(); ctx.lineWidth = 1;
    ctx.fillStyle = "#24668d"; ctx.fillText("per-token ppl at A's per-byte loss: e^{(L/c)·c}", x(0.6), 34);
    ctx.fillStyle = "#3c8d5a"; ctx.fillText(`per-byte ppl ${o.A.pplByte.toFixed(3)}`, x(5.2), y(o.A.pplByte) - 6);
    [["A", o.A, "#b8582a"], ["B", o.B, "#e3b23c"]].forEach(([n, m, col]) => { ctx.fillStyle = col; ctx.beginPath(); ctx.arc(x(m.c), y(Math.min(m.ppl, yMax)), 6, 0, 7); ctx.fill(); ctx.fillStyle = "#25313d"; ctx.fillText(`${n}: ${fmt(m.ppl, 2)}`, x(m.c) + 9, y(Math.min(m.ppl, yMax)) + 4); });
    const line = (n, m) => `${n}: ${fmt(m.tokens, 1)} tokens · loss ${m.L.toFixed(4)} nats/token · token ppl e^L = <b>${fmt(m.ppl, 3)}</b> · ${m.perByte.toFixed(4)} nats/byte = <b>${m.bpb.toFixed(4)} bits/byte</b> · byte ppl ${m.pplByte.toFixed(4)}`;
    const better = Math.abs(o.A.bpb - o.B.bpb) < 1e-9 ? "tie: identical probability of the text" : o.A.bpb < o.B.bpb ? "A is better per byte" : "B is better per byte";
    read.innerHTML = `${line("A", o.A)}<br>${line("B", o.B)}<br>
      <span class="big">${better}</span>${s.tie ? `same total −log p(D) = ${o.total.toFixed(1)} nats over ${s.bytes} bytes, so B's token ppl = ${s.pplA}^(c_B/c_A) = ${fmt(o.B.ppl, 3)}: the per-token number moves only because |D| changed` : `per-token: ${o.A.ppl < o.B.ppl ? "A" : "B"} looks better; per-byte: ${better}`}<br>
      <span class="muted small">provenance: fixture:perplexity-per-byte · lecture_12.py:L62 ((1/p(D))^(1/|D|), |D| counted in tokens) · bytes per token = lecture_01.py:L567-L571 get_compression_ratio (presets: bytes 1, GPT-5 tokenizer 2.5 and word tokenizer 5.5 on the lecture_01 examples) · the per-byte renormalisation is the KP's author-inference; the lecture does not make it.</span>`;
  };
  const sA = slider("A: token perplexity", 1.1, 100, s.pplA, 0.1, v => { s.pplA = v; draw(); });
  const cA = slider("A: bytes per token c_A", 1, 6, s.cA, 0.1, v => { s.cA = v; draw(); }), cB = slider("B: bytes per token c_B", 1, 6, s.cB, 0.1, v => { s.cB = v; draw(); });
  const lB = slider("B: own loss (nats/token, when untied)", 0.05, 8, +s.lossB.toFixed(2), 0.01, v => { s.lossB = v; draw(); });
  const presets = [["byte tokenizer: 1", 1], ["GPT-5 example: 2.5", 2.5], ["word tokenizer: 5.5", 5.5]];
  root.append(el("div", { class: "widget" }, el("div", { class: "controls" }, sA, cA, cB,
    el("div", { style: "margin-top:6px" }, "B ← ", ...presets.map(([n, c]) => el("button", { onclick: () => { s.cB = c; setSlider(cB, c); draw(); } }, n))),
    el("label", {}, el("input", { type: "checkbox", onchange: e => { s.tie = e.target.checked; draw(); } }), " B assigns the same total probability to the text as A"), lB,
    slider("text length (bytes)", 100, 10000, s.bytes, 100, v => { s.bytes = v; draw(); }), read), cv)); draw();
}

// WSD vs cosine with branched decays (lecture_11.pdf:p14 warmup / stable / decay, restart from the stable phase; p15 "Decay ~ 10%").
// The deck gives no warmup length, decay shape or final LR: warmup and decay are linear to 0 here and the 5% warmup default is a demo
// choice. A branched run of length L shares the stable prefix up to (1 − d)·L and decays over the last d·L.
export function wsdBranch(root, notice) {
  const s = { warm: 0.05, d: 0.10, branch: 0.5, k: 6 };
  const wsd = (t, L) => t < s.warm ? t / s.warm : t < (1 - s.d) * L ? 1 : t <= L ? Math.max(0, (L - t) / (s.d * L)) : NaN;
  const cos = (t, L) => t < s.warm ? t / s.warm : t <= L ? 0.5 * (1 + Math.cos(Math.PI * (t - s.warm) / (L - s.warm))) : NaN;
  const cv = el("canvas", { width: 640, height: 280 }); const read = el("div", { class: "readout" });
  const draw = () => {
    const ctx = cv.getContext("2d"), x = t => 50 + t * 560, y = v => 240 - v * 200; const L = s.branch, start = (1 - s.d) * L;
    frame(ctx, 640, 280, 50, 30, 610, 240); ctx.font = "12px sans-serif"; ctx.fillStyle = "#6b7680";
    for (let t = 0; t <= 10; t += 2) ctx.fillText(`${t * 10}%`, x(t / 10) - 10, 256); ctx.fillText("training progress (fraction of the longest run)", 340, 274); ctx.fillText("LR / peak", 6, 24);
    const curve = (f, Lr, col, w, dash) => { ctx.strokeStyle = col; ctx.lineWidth = w; ctx.setLineDash(dash || []); ctx.beginPath(); let first = true; for (let i = 0; i <= 1000; i++) { const t = i / 1000 * Lr, v = f(t, Lr); if (!Number.isFinite(v)) continue; first ? ctx.moveTo(x(t), y(v)) : ctx.lineTo(x(t), y(v)); first = false; } ctx.stroke(); ctx.setLineDash([]); ctx.lineWidth = 1; };
    curve(cos, 1, "rgba(107,118,128,.7)", 2); curve(cos, L, "rgba(107,118,128,.7)", 1, [5, 4]);
    curve(wsd, 1, "#24668d", 2); curve((t) => t >= start ? wsd(t, L) : NaN, L, "#b8582a", 3);
    ctx.fillStyle = "rgba(36,102,141,.08)"; ctx.fillRect(x(s.warm), 30, x(1 - s.d) - x(s.warm), 210);
    ctx.strokeStyle = "#b8582a"; ctx.setLineDash([3, 3]); ctx.beginPath(); ctx.moveTo(x(start), 30); ctx.lineTo(x(start), 240); ctx.stroke(); ctx.setLineDash([]);
    ctx.fillStyle = "#b8582a"; ctx.fillText(`branch starts at ${(100 * start).toFixed(1)}%`, Math.min(x(start) + 4, 500), 44);
    ctx.fillStyle = "#24668d"; ctx.fillText("WSD (stable phase shaded)", x(s.warm) + 6, y(1) - 6); ctx.fillStyle = "#6b7680"; ctx.fillText("cosine, full length / dashed: cosine for the branch length", x(0.3), y(0.18));
    const Ls = Array.from({ length: s.k }, (_, i) => (i + 1) / s.k);
    const wsdCost = 1 + Ls.slice(0, -1).reduce((a, l) => a + s.d * l, 0), cosCost = Ls.reduce((a, l) => a + l, 0);
    const diff = Math.max(...Array.from({ length: 200 }, (_, i) => { const t = s.warm + (L - s.warm) * i / 199; return Math.abs(cos(t, L) - cos(t, 1)); }));
    read.innerHTML = `branch of length ${(100 * L).toFixed(0)}%: reuse the WSD stable prefix to ${(100 * start).toFixed(1)}%, then decay for ${(100 * s.d * L).toFixed(1)}% → <b>${(100 * s.d * L).toFixed(1)}% extra compute</b><br>
      a cosine run of that length differs from the long cosine by up to ${diff.toFixed(3)}× peak LR after warmup, so it must be trained from step 0 (${(100 * L).toFixed(0)}%)<br>
      <span class="big">${s.k} data sizes: WSD ${wsdCost.toFixed(2)} runs vs cosine ${cosCost.toFixed(2)} runs</span>sizes ${Ls.map(l => `${(100 * l).toFixed(0)}%`).join(", ")}; WSD = one full run + d·L per shorter branch<br>
      <span class="muted small">provenance: fixture:wsd-branch-schedule · lecture_11.pdf:p14 (warmup, stable, decay; restart at the end of the stable phase), p15 ("Decay ~ 10%", read as 10% of the run), p22 (DeepSeek: two 10% step decays, not drawn). Loss curves are figure-only (p15) and not modelled; linear warmup, linear decay to 0 and the 5% warmup default are demo choices the deck does not specify.</span>`;
  };
  root.append(el("div", { class: "widget" }, el("div", { class: "controls" },
    slider("decay fraction d", 0.02, 0.5, s.d, 0.01, v => { s.d = v; draw(); }, v => `${Math.round(v * 100)}%`),
    slider("branch run length L (fraction of longest)", 0.1, 1, s.branch, 0.01, v => { s.branch = v; draw(); }, v => `${Math.round(v * 100)}%`),
    slider("warmup fraction (deck: unspecified)", 0.01, 0.2, s.warm, 0.01, v => { s.warm = v; draw(); }, v => `${Math.round(v * 100)}%`),
    slider("number of data sizes", 1, 12, s.k, 1, v => { s.k = v; draw(); }), read), cv)); draw();
}

// Muon: B = U S V^T → U V^T (lecture_11.pdf:p42). 2×2 SVD in closed form; Newton–Schulz coefficients (3.4445, −4.7750, 2.0315), 5 steps
// and the Frobenius pre-normalisation (eps 1e-7) are from the Muon reference implementation (Jordan 2024), not the slide.
const svd2 = ([a, b, c, d]) => { // M = [[a, b], [c, d]] = Rot(phi) · diag(s1, s2) · Rot(theta), s2 signed
  const E = (a + d) / 2, F = (a - d) / 2, G = (c + b) / 2, H = (c - b) / 2, Q = Math.hypot(E, H), R = Math.hypot(F, G);
  const a1 = Math.atan2(G, F), a2 = Math.atan2(H, E); return { s1: Q + R, s2: Q - R, theta: (a2 - a1) / 2, phi: (a2 + a1) / 2 };
};
const mul2 = (X, Y) => [X[0] * Y[0] + X[1] * Y[2], X[0] * Y[1] + X[1] * Y[3], X[2] * Y[0] + X[3] * Y[2], X[2] * Y[1] + X[3] * Y[3]];
const rot2 = a => [Math.cos(a), -Math.sin(a), Math.sin(a), Math.cos(a)];
export function muonOrthogonalize(root, notice) {
  const s = { m: [2, 1, 0.5, 0.5], k: 5 };
  const NS = [3.4445, -4.7750, 2.0315];
  const polar = (M) => { const v = svd2(M); return mul2(mul2(rot2(v.phi), [1, 0, 0, Math.sign(v.s2) || 1]), rot2(v.theta)); };
  const newtonSchulz = (M, k) => { const nf = Math.hypot(...M) + 1e-7; let X = M.map(v => v / nf); for (let i = 0; i < k; i++) { const A = mul2(X, [X[0], X[2], X[1], X[3]]), B = mul2(A, A).map((v, j) => NS[1] * A[j] + NS[2] * v); X = mul2(B, X).map((v, j) => NS[0] * X[j] + v); } return X; };
  const cv = el("canvas", { width: 640, height: 260 }); const read = el("div", { class: "readout" });
  const draw = () => {
    const M = s.m, v = svd2(M), O = polar(M), X = newtonSchulz(M, s.k), vo = svd2(O), vx = svd2(X), nf = Math.hypot(...M);
    const ctx = cv.getContext("2d"); ctx.clearRect(0, 0, 640, 260); ctx.font = "12px sans-serif";
    const panel = (cx, title, T, scale, col) => {
      ctx.strokeStyle = "#d9d3c7"; ctx.beginPath(); ctx.arc(cx, 140, 60, 0, 7); ctx.stroke();
      ctx.strokeStyle = col; ctx.lineWidth = 2; ctx.beginPath(); for (let i = 0; i <= 120; i++) { const a = i / 120 * 2 * Math.PI, p = [Math.cos(a), Math.sin(a)], q = [T[0] * p[0] + T[1] * p[1], T[2] * p[0] + T[3] * p[1]]; const px = cx + q[0] * 60 * scale, py = 140 - q[1] * 60 * scale; i ? ctx.lineTo(px, py) : ctx.moveTo(px, py); } ctx.stroke(); ctx.lineWidth = 1;
      ctx.fillStyle = "#25313d"; ctx.fillText(title, cx - 90, 24);
    };
    const gs = 1.6 / Math.max(v.s1, 1e-9);
    panel(110, `G maps the unit circle (×${gs.toFixed(2)})`, M, gs, "#24668d"); panel(320, "U·Vᵀ: circle → circle", O, 1, "#3c8d5a"); panel(530, `Newton–Schulz, ${s.k} steps`, X, 1, "#b8582a");
    ctx.fillStyle = "#6b7680"; ctx.fillText("faint circle = unit circle", 270, 250);
    const sv = (o) => `${Math.abs(o.s1).toFixed(4)}, ${Math.abs(o.s2).toFixed(4)}`;
    read.innerHTML = `G = [[${M.map(x => x.toFixed(2)).slice(0, 2).join(", ")}], [${M.map(x => x.toFixed(2)).slice(2).join(", ")}]] · singular values σ = <b>${sv(v)}</b> (ratio ${Math.abs(v.s2) > 1e-12 ? (v.s1 / Math.abs(v.s2)).toFixed(2) : "∞"})<br>
      G / ‖G‖_F keeps the ratio: σ = ${(v.s1 / nf).toFixed(4)}, ${(Math.abs(v.s2) / nf).toFixed(4)} (normalising is not orthogonalising)<br>
      <span class="big">U·Vᵀ: σ = ${sv(vo)}</span>U·Vᵀ = [[${O.slice(0, 2).map(x => x.toFixed(3)).join(", ")}], [${O.slice(2).map(x => x.toFixed(3)).join(", ")}]]${Math.abs(v.s2) < 1e-9 ? " — G is singular, so U·Vᵀ is not unique" : ""}<br>
      Newton–Schulz after ${s.k} steps: σ = <b>${sv(vx)}</b> (approximate: the quintic is tuned for speed, so it pulls the σ toward 1 and nearly equal, but not exactly to 1)<br>
      <span class="muted small">provenance: fixture:muon-orthogonalize · lecture_11.pdf:p42 ("NewtonSchultz (approximately) orthogonalizes the matrix B_t = U S V^T → U V^T"); the 2×2 SVD is closed form. The slide prints no Newton–Schulz coefficients: X ← aX + (bA + cA²)X with A = XXᵀ, (a, b, c) = (3.4445, −4.7750, 2.0315), 5 steps and X₀ = G/(‖G‖_F + 1e-7) come from the Muon reference implementation (Jordan 2024; Muon reference cited as arXiv:2502.16982 §2). Momentum and the update scale factor are not modelled; the default entries are arbitrary.</span>`;
  };
  const names = ["G₁₁", "G₁₂", "G₂₁", "G₂₂"];
  root.append(el("div", { class: "widget" }, el("div", { class: "controls" },
    ...names.map((n, i) => slider(n, -3, 3, s.m[i], 0.05, v => { s.m[i] = v; draw(); })),
    slider("Newton–Schulz steps k", 0, 10, s.k, 1, v => { s.k = v; draw(); }), read), cv)); draw();
}

// muP vs SP for a hidden linear layer (lecture_11.pdf:p47-p50). Θ-scalings only, constants dropped:
// SP init 1/√n_{l−1}, LR Θ(1); muP init (1/√n_{l−1})·min(1, √(n_l/n_{l−1})), SGD LR n_l/n_{l−1}, Adam LR 1/n_{l−1}.
const MUP_ROWS = [
  ["SP", "init std", "1/√n_{l−1}", (i, o) => 1 / Math.sqrt(i)], ["SP", "LR (SGD and Adam)", "Θ(1)", () => 1],
  ["muP", "init std", "(1/√n_{l−1})·min(1, √(n_l/n_{l−1}))", (i, o) => Math.min(1, Math.sqrt(o / i)) / Math.sqrt(i)],
  ["muP", "LR, SGD", "n_l / n_{l−1}", (i, o) => o / i], ["muP", "LR, Adam", "1 / n_{l−1}", (i, o) => 1 / i],
];
const mupWidths = ({ nIn, nOut, scale }, m) => ({ i: nIn * (scale !== "out" ? m : 1), o: nOut * (scale !== "in" ? m : 1) });
// pure model (MODELS below): × change of each row at width multiplier m against m = 1
function mupModel(s) {
  const b = mupWidths(s, 1), w = mupWidths(s, s.m), r = MUP_ROWS.map(([, , , fn]) => fn(w.i, w.o) / fn(b.i, b.o));
  return { spInit: r[0], spLr: r[1], mupInit: r[2], mupSgdLr: r[3], mupAdamLr: r[4] };
}
export function mupVsSpTable(root, notice) {
  const s = { nIn: 1024, nOut: 1024, m: 2, scale: "both" };
  const rows = MUP_ROWS;
  const widths = (m) => mupWidths(s, m);
  const cv = el("canvas", { width: 640, height: 260 }); const read = el("div", { class: "readout" });
  const tbl = el("table", { style: "border-collapse:collapse;font-size:13px;width:100%;margin-top:8px" });
  const draw = () => {
    const b = widths(1), w = widths(s.m);
    tbl.innerHTML = `<tr style="text-align:left;color:#6b7680"><th>param.</th><th>quantity</th><th>scaling</th><th>at base (${b.i}→${b.o})</th><th>at m = ${s.m} (${fmt(w.i, 0)}→${fmt(w.o, 0)})</th><th>× change</th></tr>` +
      rows.map(([p, q, f, fn]) => { const r = fn(w.i, w.o) / fn(b.i, b.o); return `<tr style="border-top:1px solid #d9d3c7"><td>${p}</td><td>${q}</td><td>${f}</td><td>${fmt(fn(b.i, b.o), 4)}</td><td>${fmt(fn(w.i, w.o), 4)}</td><td style="color:${Math.abs(r - 1) < 1e-12 ? "#3c8d5a" : "#b8582a"}"><b>${fmt(r, 4)}</b></td></tr>`; }).join("");
    const ctx = cv.getContext("2d"), lx = m => 50 + (Math.log2(m) + 2) / 6 * 560, ly = r => 130 - Math.log2(r) / 6 * 100;
    frame(ctx, 640, 260, 50, 20, 610, 240); ctx.font = "12px sans-serif"; ctx.fillStyle = "#6b7680";
    for (const m of [0.25, 0.5, 1, 2, 4, 8, 16]) ctx.fillText(String(m), lx(m) - 6, 254); for (const r of [1 / 64, 1 / 8, 1, 8, 64]) ctx.fillText(r < 1 ? `1/${1 / r}` : String(r), 14, ly(r) + 4);
    ctx.fillText("× change vs base, against width multiplier m (log-log)", 56, 14);
    const cols = ["#6b7680", "#25313d", "#3c8d5a", "#24668d", "#b8582a"];
    rows.forEach(([p, q, , fn], j) => { ctx.strokeStyle = cols[j]; ctx.lineWidth = 2; ctx.setLineDash(p === "SP" ? [5, 4] : []); ctx.beginPath(); for (let i = 0; i <= 120; i++) { const m = Math.pow(2, -2 + 6 * i / 120), ww = widths(m), r = fn(ww.i, ww.o) / fn(b.i, b.o); const yy = Math.max(20, Math.min(240, ly(r))); i ? ctx.lineTo(lx(m), yy + j * 0.8) : ctx.moveTo(lx(m), yy + j * 0.8); } ctx.stroke(); ctx.setLineDash([]); ctx.lineWidth = 1; ctx.fillStyle = cols[j]; ctx.fillText(`${p} ${q}`, 470, 40 + j * 15); });
    ctx.strokeStyle = "#e3b23c"; ctx.beginPath(); ctx.moveTo(lx(s.m), 20); ctx.lineTo(lx(s.m), 240); ctx.stroke();
    read.innerHTML = `${s.scale === "both" ? "square-layer widening: SP LR and muP-SGD LR stay flat, muP-Adam LR falls as 1/m, init std is the same in SP and muP" : s.scale === "out" ? "fan-out only: muP init departs from SP once n_l &lt; n_{l−1}; muP-SGD LR follows n_l" : "fan-in only: muP-SGD LR falls as 1/m, Adam as 1/m, muP init falls faster than SP's once n_l &lt; n_{l−1}"}<br>
      <span class="muted small">provenance: fixture:mup-vs-sp-table · lecture_11.pdf:p47 (σ = Θ((1/√n_{l−1})·min(1, √(n_l/n_{l−1})))), p49 (η_l = Θ(n_l/n_{l−1}); "with Adam ‖ΔW_l‖_* √n_{l−1} = Θ(η_l)"), p50 recap (muP LR n_l/n_{l−1}, Adam 1/n_{l−1}; SP init 1/√n_{l−1}, LR Θ(1); "LR changes for Adam, also init diffs when fanout n_l &lt; fanin"). The text layer drops the square roots; these are the KP's reconstruction. Θ means constants are dropped: values are scaling expressions, not tuned hyperparameters. The KP predict asks about 1024 → ×4; the default opens at ×2 so the learner moves to it.</span>`;
  };
  root.append(el("div", { class: "widget" }, el("div", { class: "controls" },
    slider("width multiplier m", 0.25, 16, s.m, 0.25, v => { s.m = v; draw(); }),
    slider("base fan-in n_{l−1}", 64, 8192, s.nIn, 64, v => { s.nIn = v; draw(); }), slider("base fan-out n_l", 64, 8192, s.nOut, 64, v => { s.nOut = v; draw(); }),
    el("div", { style: "margin-top:8px" }, ...[["both", "scale n_l and n_{l−1}"], ["out", "scale fan-out n_l only"], ["in", "scale fan-in n_{l−1} only"]].map(([k, n]) => el("button", { onclick: () => { s.scale = k; draw(); } }, n))),
    read), el("div", {}, cv, tbl))); draw();
}

Object.assign(WIDGETS, {
  "fixture:filter-retention-ratios": filterRetention,
  "fixture:perplexity-per-byte": perplexityPerByte,
  "fixture:wsd-branch-schedule": wsdBranch,
  "fixture:muon-orthogonalize": muonOrthogonalize,
  "fixture:mup-vs-sp-table": mupVsSpTable,
});

// ================================================================ MODELS ==
// Pure models behind the legacy widgets above, checked by tools/check_widgets.mjs against the prompts they
// serve (`from` = thread:kp:prompt, "#n" = the answer's n-th number). Each widget calls the same function.
const LLAMA13 = { params: 13e9, bytes: 2, L: 40, K: 40, H: 128, bw: 3.35e12 };  // lecture_10.py:L313-L323, L322
export const MODELS = {
  "fixture:napkin": { fn: napkinModel, cases: [
    { args: { N: 70e9, D: 15e12, G: 1024, mfu: 0.5 }, pick: "days", expect: 144, from: "lecture_02:napkin-training-time:predict" },
    { args: { N: 70e9, D: 15e12, G: 4096, mfu: 0.4 }, pick: "days", expect: 45, from: "lecture_02:napkin-training-time:transfer" },
    { args: { gpuGB: 80, ngpu: 8, bpp: 12 }, pick: "maxNB", expect: 53, from: "lecture_02:napkin-max-model-size:predict" },
    { args: { gpuGB: 80, ngpu: 8, bpp: 8 }, pick: "maxNB", expect: 80, from: "lecture_02:napkin-max-model-size:transfer" },
  ] },
  "fixture:memory-ledger": { fn: memoryLedgerModel, cases: [
    { args: { B: 64, micro: 64, opt: 8 }, pick: "bytesPerParam", expect: 12, from: "lecture_02:optimizer-state-memory:predict" },
    { args: { B: 1024, micro: 128 }, pick: "actFraction", expect: 0.125, from: "lecture_02:gradient-accumulation:predict" },
    { args: { B: 64, micro: 64, D: 1024, L: 16, opt: 4 }, pick: "act", expect: 2, tol: 1e-9 },  // 2·64·1024·16 B = 2 MiB
  ] },
  "fixture:collective-bytes": { fn: collectiveModel, cases: [
    { args: { W: 8, numel: 100 * 1024 ** 2, bytes: 2 }, pick: "arTotalMiB", expect: 2800, from: "lecture_07:collective-bytes:transfer" },
    { args: { W: 4, numel: 100 * 1024 ** 2, bytes: 4 }, pick: "arRankMiB", expect: 600, tol: 1e-9 },  // predict's W = 4 side: 2·400·3/4
    { args: { W: 8, numel: 100 * 1024 ** 2, bytes: 4 }, pick: "arRankMiB", expect: 700, tol: 1e-9 },  // W = 8 side: 700/600 = 1.17
  ] },
  "fixture:zero-memory": { fn: zeroModel, cases: [
    { args: { psi: 7.5e9, N: 64 }, pick: "stage0", expect: 120, from: "lecture_07:zero-sharding:predict#1" },
    { args: { psi: 7.5e9, N: 64 }, pick: "stage1", expect: 31.4, from: "lecture_07:zero-sharding:predict#2" },
    { args: { psi: 70e9, N: 1024 }, pick: "stage3", expect: 1.09, from: "lecture_07:zero-sharding:transfer" },
  ] },
  "fixture:parallelism-comm": { fn: parallelismModel, cases: [
    { args: { psi: 70e9, b: 8, seq: 4096, h: 8192, L: 32, bytes: 2, tp: 8, pp: 8, m: 1, zero: 0 }, pick: "tpOverPp", expect: 16, from: "lecture_08:tp-vs-pp-communication:predict" },
    { args: { psi: 70e9, b: 8, seq: 4096, h: 8192, L: 32, bytes: 2, tp: 4, pp: 4, m: 1, zero: 0 }, pick: "tpLayerOverAct", expect: 6, from: "lecture_08:tp-vs-pp-communication:check" },
    { args: { psi: 70e9, b: 4, seq: 4096, h: 8192, L: 32, bytes: 1, tp: 8, pp: 8, m: 1, zero: 0 }, pick: "ppMicro", expect: 134217728, from: "lecture_08:tp-vs-pp-communication:transfer#4" },
    { args: { psi: 70e9, b: 4, seq: 4096, h: 8192, L: 32, bytes: 1, tp: 8, pp: 8, m: 1, zero: 0 }, pick: "tpLayer", expect: 939524096, from: "lecture_08:tp-vs-pp-communication:transfer#9" },
  ] },
  "fixture:decode-bound": { fn: decodeModel, cases: [
    { args: { ...LLAMA13, params: 70e9 / 8, S: 0, B: 1 }, pick: "tps", expect: 190, from: "lecture_10:decode-bandwidth-bound:transfer" },
    { args: { ...LLAMA13, S: 1024, B: 256 }, pick: "tps", expect: 3600, tol: 0.02, from: "lecture_10:batch-latency-throughput-tradeoff:predict" },
    { args: { ...LLAMA13, S: 4096, B: 64 }, pick: "ceiling", expect: 1000, from: "lecture_10:batch-latency-throughput-tradeoff:transfer" },
    { args: { ...LLAMA13, S: 1024, B: 1 }, pick: "tps", expect: 125 },   // the "roughly 125 tokens/s" in decode-bandwidth-bound:predict
    { args: { ...LLAMA13, S: 1024, B: 64 }, pick: "tps", expect: 2690 },  // the "about 2690 tokens/s" in batch-latency-throughput-tradeoff:predict
  ] },
  "fixture:kv-cache-growth": { fn: kvModel, cases: [
    { args: { params: 13e9, prompt: 1024, gpuGB: 80 }, pick: "fitB", expect: 64, from: "lecture_10:kv-cache-size:predict" },
    { args: { L: 32, K: 8, H: 128, bytes: 2 }, pick: "perTokSeq", expect: 131072, tol: 1e-9, from: "lecture_10:kv-cache-size:transfer" },
    { args: { L: 80, K: 8, H: 128, bytes: 2 }, pick: "perTokSeq", expect: 327680, tol: 1e-9, from: "lecture_10:gqa-kv-reduction:transfer" },
  ] },
  "fixture:bradley-terry": { fn: btModel, cases: [
    { args: { rw: 3, rl: 1 }, pick: "p", expect: 0.88, from: "lecture_15:bradley-terry-reward-model:predict#4" },
    { args: { rw: 3, rl: 1, shift: 10 }, pick: "p", expect: 0.88, from: "lecture_15:bradley-terry-reward-model:predict#6" },
    { args: { rw: 2 * Math.log(4), rl: 0 }, pick: "p", expect: 0.941, from: "lecture_15:bradley-terry-reward-model:check" },  // logit 0.8 = ln 4, twice
  ] },
  "fixture:mup-vs-sp-table": { fn: mupModel, untied: "the predict/transfer answers write the factors as 1/4 and Θ(1), which the number parser cannot tie; the cases check those rows directly", cases: [
    { args: { nIn: 1024, nOut: 1024, m: 4, scale: "both" }, pick: "mupAdamLr", expect: 0.25, tol: 1e-9 },  // mup-lr-derivation-and-recap:predict, muP + Adam
    { args: { nIn: 1024, nOut: 1024, m: 4, scale: "both" }, pick: "spLr", expect: 1, tol: 1e-9 },         // same predict, SP unchanged
    { args: { nIn: 1024, nOut: 1024, m: 4, scale: "both" }, pick: "mupSgdLr", expect: 1, tol: 1e-9 },     // transfer: SGD unchanged for square layers
  ] },
};
