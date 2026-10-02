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

// ---------------------------------------------------------------- napkin --
export function napkin(root, notice) {
  const s = { N: 70e9, D: 15e12, G: 1024, mfu: 0.5, bpp: 12, gpuGB: 80, ngpu: 8 };
  const days = () => (6 * s.N * s.D) / (H100_FLOPS_BF16 * s.mfu * s.G * 86400);
  const maxN = () => (s.gpuGB * 1e9 * s.ngpu) / s.bpp;
  const read = el("div", { class: "readout" });
  const draw = () => {
    read.innerHTML = `<span class="big">${fmt(days(), 1)} days</span> = 6·N·D ÷ (P·u·G·86400)<br>
      6·${fmt(s.N, 0)}·${fmt(s.D, 0)} = ${fmt(6 * s.N * s.D, 2)} FLOPs<br><br>
      <span class="big">${fmt(maxN() / 1e9, 1)} B params fit</span> = ${s.ngpu}×${s.gpuGB} GB ÷ ${s.bpp} bytes/param<br>
      <span class="muted small">provenance: lecture_02.py:L72-L82</span>`;
  };
  const controls = el("div", { class: "controls" },
    slider("parameters N (B)", 1, 1000, 70, 1, v => { s.N = v * 1e9; draw(); }),
    slider("tokens D (T)", 0.1, 50, 15, 0.1, v => { s.D = v * 1e12; draw(); }),
    slider("GPUs", 8, 16384, 1024, 8, v => { s.G = v; draw(); }),
    slider("MFU", 0.1, 1, 0.5, 0.05, v => { s.mfu = v; draw(); }),
    slider("bytes / param", 2, 20, 12, 2, v => { s.bpp = v; draw(); }, v => `${v}  (2+2+${v - 4} opt)`),
    slider("GPUs for memory", 1, 64, 8, 1, v => { s.ngpu = v; draw(); }),
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
    const cs = Math.min(34, 240 / Math.max(s.B, s.D, s.K));
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

// --------------------------------------------------------------- roofline --
export function roofline(root, notice) {
  const s = { n: 1024, dtype: 2 };
  const cv = el("canvas", { width: 640, height: 340 });
  const read = el("div", { class: "readout" });
  const ops = () => {
    const n = s.n, b = s.dtype;
    return [
      { name: "relu (n²)", flops: n * n, bytes: 2 * b * n * n },                // lecture_02.py:L363-L399
      { name: "gelu (n²)", flops: 20 * n * n, bytes: 2 * b * n * n },           // 20 flops/elt (lecture_02.py:L404)
      { name: "dot (n)", flops: 2 * n, bytes: 2 * b * n },                      // L418-L433
      { name: "matvec (n×n·n)", flops: 2 * n * n, bytes: b * (n * n + 2 * n) }, // L434-L448
      { name: "matmul (n×n·n×n)", flops: n * n * (2 * n - 1), bytes: 3 * b * n * n }, // L449-L467
    ].map(o => ({ ...o, ai: o.flops / o.bytes }));
  };
  const draw = () => {
    const ctx = cv.getContext("2d"); ctx.clearRect(0, 0, cv.width, cv.height);
    const accAI = H100_FLOPS_BF16 / H100_BYTES; // 295
    const x = ai => 60 + (Math.log10(Math.max(ai, 1e-2)) + 2) / 6 * 540; // 1e-2..1e4
    const y = f => 300 - (Math.log10(Math.max(f, 1e9)) - 9) / 6.5 * 260; // 1e9..~3e15
    ctx.strokeStyle = "#d9d3c7"; ctx.beginPath(); ctx.moveTo(60, 300); ctx.lineTo(600, 300); ctx.moveTo(60, 300); ctx.lineTo(60, 30); ctx.stroke();
    ctx.fillStyle = "#6b7680"; ctx.font = "12px sans-serif";
    for (let e = -2; e <= 4; e++) ctx.fillText(`1e${e}`, x(10 ** e) - 10, 315);
    ctx.fillText("arithmetic intensity (FLOPs/byte)", 380, 335); ctx.save(); ctx.translate(14, 200); ctx.rotate(-Math.PI / 2); ctx.fillText("attainable FLOP/s", 0, 0); ctx.restore();
    // roofline
    ctx.strokeStyle = "#25313d"; ctx.lineWidth = 2; ctx.beginPath();
    ctx.moveTo(x(1e-2), y(1e-2 * H100_BYTES)); ctx.lineTo(x(accAI), y(H100_FLOPS_BF16)); ctx.lineTo(x(1e4), y(H100_FLOPS_BF16)); ctx.stroke(); ctx.lineWidth = 1;
    ctx.fillStyle = "#25313d"; ctx.fillText(`kink = ${accAI.toFixed(0)} FLOPs/byte`, x(accAI) + 6, y(H100_FLOPS_BF16) - 8);
    const rows = [];
    for (const o of ops()) {
      const bound = o.ai < accAI ? "memory" : "compute";
      const attain = Math.min(H100_FLOPS_BF16, o.ai * H100_BYTES);
      ctx.fillStyle = bound === "memory" ? "#b8582a" : "#3c8d5a";
      ctx.beginPath(); ctx.arc(x(o.ai), y(attain), 5, 0, 7); ctx.fill();
      ctx.fillText(o.name, x(o.ai) + 8, y(attain) + 4);
      rows.push(`${o.name.padEnd(18)} AI=${o.ai.toFixed(1).padStart(7)} → ${bound}-bound, MFU ≤ ${Math.min(1, o.ai / accAI).toFixed(2)}`);
    }
    read.innerHTML = `<pre>${rows.join("\n")}</pre><span class="muted small">H100: ${fmt(H100_FLOPS_BF16, 2)} FLOP/s (no sparsity), ${fmt(H100_BYTES, 2)} B/s · lecture_02.py:L338-L483</span>`;
  };
  const controls = el("div", { class: "controls" },
    slider("n", 16, 16384, 1024, 16, v => { s.n = v; draw(); }),
    slider("bytes/element", 1, 4, 2, 1, v => { s.dtype = v; draw(); }, v => ({ 1: "fp8", 2: "bf16", 4: "fp32" })[v] || v),
    read);
  root.append(el("div", { class: "widget" }, controls, cv));
  draw();
}

// ------------------------------------------------------------ memory ledger --
export function memoryLedger(root, notice) {
  const s = { B: 64, micro: 64, D: 1024, L: 16, opt: 4 };
  const cv = el("canvas", { width: 640, height: 220 });
  const read = el("div", { class: "readout" });
  const vals = () => { const N = s.D * s.D * s.L, mb = 2 ** 20; return { "params (bf16)": 2 * N / mb, "grads (bf16)": 2 * N / mb, [`optimizer (fp32 ×${s.opt / 4})`]: s.opt * N / mb, "activations (bf16)": 2 * s.micro * s.D * s.L / mb }; };
  const draw = () => {
    const v = vals(), ctx = cv.getContext("2d"); ctx.clearRect(0, 0, cv.width, cv.height);
    const max = Math.max(...Object.values(v)); let y = 20;
    for (const [k, x] of Object.entries(v)) {
      ctx.fillStyle = "#6b7680"; ctx.font = "13px sans-serif"; ctx.fillText(k, 10, y + 14);
      ctx.fillStyle = k.startsWith("act") ? "#b8582a" : "#24668d"; ctx.fillRect(190, y, Math.max(2, 400 * x / max), 18);
      ctx.fillStyle = "#25313d"; ctx.fillText(`${fmt(x, 1)} MB`, 196 + Math.max(2, 400 * x / max), y + 14); y += 46;
    }
    read.innerHTML = `micro-steps per update: <b>${Math.round(s.B / s.micro)}</b> · FLOPs per update = 6·B·N (unchanged by micro)<br><span class="muted small">lecture_02.py:L636-L646, L719-L731 (B=64, D=1024, L=16)</span>`;
  };
  const controls = el("div", { class: "controls" },
    slider("batch B", 16, 8192, 64, 16, v => { s.B = v; s.micro = Math.min(s.micro, v); draw(); }),
    slider("micro-batch", 16, 8192, 64, 16, v => { s.micro = Math.min(v, s.B); draw(); }),
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
  const s = { eps: 0.2, A: 1.0 };
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
    slider("clip ε", 0, 1, 0.2, 0.05, v => { s.eps = v; draw(); }), slider("advantage A", -1.5, 1.5, 1, 0.1, v => { s.A = v; draw(); }), read);
  root.append(el("div", { class: "widget" }, controls, cv));
  draw();
}

export const WIDGETS = {
  "fixture:napkin": napkin,
  "fixture:matmul-contraction": contraction,
  "fixture:roofline": roofline,
  "fixture:memory-ledger": memoryLedger,
  "fixture:rope-shared-shift": ropePair,
  "fixture:ratio-clip": ratioClip,
};

// ======================================================= lecture 5 widgets ==
// Numbers: H100 table from lecture_06.py L46-L54 (bandwidth TB/s, capacity) as recorded in lecture_05 KPs.
export function memoryHierarchy(root, notice) {
  const levels = [
    { name: "registers", bw: 401, cap: "256 KB / SM" }, { name: "L1 / shared", bw: 33, cap: "256 KB / SM" },
    { name: "L2", bw: 12, cap: "50 MB" }, { name: "HBM", bw: 3.35, cap: "80 GB" },
  ];
  const cv = el("canvas", { width: 640, height: 240 }); const read = el("div", { class: "readout" });
  const s = { bytes: 64 };
  const draw = () => {
    const ctx = cv.getContext("2d"); ctx.clearRect(0, 0, 640, 240); let y = 16;
    for (const l of levels) {
      const w = Math.max(3, 420 * Math.log10(l.bw * 10) / Math.log10(4010));
      ctx.fillStyle = "#6b7680"; ctx.font = "13px sans-serif"; ctx.fillText(`${l.name}  (${l.cap})`, 10, y + 14);
      ctx.fillStyle = l.name === "HBM" ? "#b8582a" : "#24668d"; ctx.fillRect(190, y, w, 18);
      ctx.fillStyle = "#25313d"; ctx.fillText(`${l.bw} TB/s · ${(s.bytes * 1e6 / (l.bw * 1e12) * 1e6).toFixed(2)} µs per ${s.bytes} MB`, 196 + w, y + 14); y += 50;
    }
    ctx.fillStyle = "#6b7680"; ctx.fillText("bar length is log-scaled bandwidth", 190, 232);
    read.innerHTML = `<span class="muted small">lecture_06.py:L46-L54 (H100), lecture_05.pdf:p10-p12</span>`;
  };
  root.append(el("div", { class: "widget" }, el("div", { class: "controls" }, slider("bytes to move (MB)", 1, 1024, 64, 1, v => { s.bytes = v; draw(); }), read), cv)); draw();
}

export function coalescing(root, notice) {
  const s = { stride: 1, warp: 32, elt: 4, burst: 128 };
  const cv = el("canvas", { width: 640, height: 200 }); const read = el("div", { class: "readout" });
  const draw = () => {
    const ctx = cv.getContext("2d"); ctx.clearRect(0, 0, 640, 200);
    const addrs = Array.from({ length: s.warp }, (_, t) => t * s.stride * s.elt);
    const bursts = new Set(addrs.map(a => Math.floor(a / s.burst)));
    const span = Math.max(...addrs) + s.elt; const px = 600 / Math.max(span, s.burst);
    for (let b = 0; b * s.burst < span; b++) { ctx.fillStyle = bursts.has(b) ? "rgba(184,88,42,.18)" : "#f5f2ea"; ctx.fillRect(20 + b * s.burst * px, 40, s.burst * px - 1, 60); }
    for (const a of addrs) { ctx.fillStyle = "#24668d"; ctx.fillRect(20 + a * px, 50, Math.max(2, s.elt * px - 1), 40); }
    ctx.fillStyle = "#6b7680"; ctx.font = "13px sans-serif"; ctx.fillText(`${s.warp} threads · ${s.elt}-byte elements · stride ${s.stride} → ${bursts.size} × ${s.burst}-byte transactions`, 20, 130);
    read.innerHTML = `<span class="big">${bursts.size} transactions</span> for ${s.warp * s.elt} useful bytes (${(100 * s.warp * s.elt / (bursts.size * s.burst)).toFixed(0)}% of moved bytes used)<br><span class="muted small">lecture_05.pdf:p37-p39; 128-byte transactions from lecture_06.py:L127-L130</span>`;
  };
  root.append(el("div", { class: "widget" }, el("div", { class: "controls" }, slider("stride (elements)", 1, 64, 1, 1, v => { s.stride = v; draw(); }), slider("element bytes", 1, 8, 4, 1, v => { s.elt = v; draw(); }), read), cv)); draw();
}

export function tilingSweep(root, notice) {
  const s = { N: 8, T: 2, step: 0 };
  const cv = el("canvas", { width: 640, height: 300 }); const read = el("div", { class: "readout" });
  const phases = () => s.N / s.T;
  const draw = () => {
    const ctx = cv.getContext("2d"); ctx.clearRect(0, 0, 640, 300); const cs = Math.min(22, 180 / s.N);
    const grid = (x0, y0, name, hl) => { ctx.fillStyle = "#6b7680"; ctx.font = "12px sans-serif"; ctx.fillText(name, x0, y0 - 6); for (let i = 0; i < s.N; i++) for (let j = 0; j < s.N; j++) { const h = hl(i, j); ctx.fillStyle = h || "#fff"; ctx.strokeStyle = "#e4dfd4"; ctx.fillRect(x0 + j * cs, y0 + i * cs, cs, cs); ctx.strokeRect(x0 + j * cs, y0 + i * cs, cs, cs); } };
    const ph = Math.min(s.step, phases()); const tile = { r: 0, c: 0 }; // output tile (0,0)
    grid(20, 30, "A", (i, j) => i < s.T && Math.floor(j / s.T) < ph ? (Math.floor(j / s.T) === ph - 1 ? "rgba(36,102,141,.5)" : "rgba(36,102,141,.15)") : null);
    grid(20 + s.N * cs + 40, 30, "B", (i, j) => j < s.T && Math.floor(i / s.T) < ph ? (Math.floor(i / s.T) === ph - 1 ? "rgba(184,88,42,.5)" : "rgba(184,88,42,.15)") : null);
    grid(20 + 2 * (s.N * cs + 40), 30, "C (one T×T tile)", (i, j) => i < s.T && j < s.T ? `rgba(227,178,60,${0.15 + 0.6 * ph / phases()})` : null);
    const naive = s.N * s.N * 2 * s.N, tiled = naive / s.T;
    read.innerHTML = `phase ${ph} / ${phases()} · each phase loads one T×T tile of A and B into shared memory and does T³ multiply-adds<br>
      <span class="big">global reads ÷ ${s.T}</span> naive ${naive} → tiled ${tiled} element reads for the whole N×N product<br><span class="muted small">lecture_05.pdf:p40-p42 ("factor of T reduction")</span>`;
  };
  const controls = el("div", { class: "controls" }, slider("N", 4, 16, 8, 4, v => { s.N = v; s.T = Math.min(s.T, v); s.step = 0; draw(); }), slider("tile T", 1, 8, 2, 1, v => { s.T = Math.min(v, s.N); s.step = 0; draw(); }),
    el("div", { style: "margin-top:8px" }, el("button", { onclick: () => { s.step = Math.min(s.step + 1, phases()); draw(); } }, "next phase"), " ", el("button", { class: "ghost", onclick: () => { s.step = 0; draw(); } }, "reset")), read);
  root.append(el("div", { class: "widget" }, controls, cv)); draw();
}

export function waveQuantization(root, notice) {
  const s = { M: 1792, tileM: 256, tileN: 128, sms: 108 };
  const cv = el("canvas", { width: 640, height: 220 }); const read = el("div", { class: "readout" });
  const draw = () => {
    const ctx = cv.getContext("2d"); ctx.clearRect(0, 0, 640, 220);
    const pts = []; for (let m = 1024; m <= 2560; m += 8) { const tiles = Math.ceil(m / s.tileM) * Math.ceil(m / s.tileN); pts.push([m, Math.ceil(tiles / s.sms)]); }
    const maxW = Math.max(...pts.map(p => p[1])); const x = m => 40 + (m - 1024) / 1536 * 580, y = w => 190 - w / maxW * 160;
    ctx.strokeStyle = "#24668d"; ctx.lineWidth = 2; ctx.beginPath(); pts.forEach((p, i) => i ? ctx.lineTo(x(p[0]), y(p[1])) : ctx.moveTo(x(p[0]), y(p[1]))); ctx.stroke(); ctx.lineWidth = 1;
    const tiles = Math.ceil(s.M / s.tileM) * Math.ceil(s.M / s.tileN), waves = Math.ceil(tiles / s.sms);
    ctx.fillStyle = "#b8582a"; ctx.beginPath(); ctx.arc(x(s.M), y(waves), 5, 0, 7); ctx.fill();
    ctx.fillStyle = "#6b7680"; ctx.font = "12px sans-serif"; ctx.fillText("M = N = K (square matmul)", 460, 210); ctx.fillText("waves", 4, 20);
    read.innerHTML = `M=${s.M}: ${Math.ceil(s.M / s.tileM)} × ${Math.ceil(s.M / s.tileN)} = <b>${tiles} tiles</b> on ${s.sms} SMs → <span class="big">${waves} waves</span> last wave ${(100 * (tiles - (waves - 1) * s.sms) / s.sms).toFixed(0)}% full<br><span class="muted small">lecture_05.pdf:p43-p48 (1792 → 1793: 98 → 120 tiles); SM count lecture_06.py:L134-L137</span>`;
  };
  root.append(el("div", { class: "widget" }, el("div", { class: "controls" }, slider("M", 1024, 2560, 1792, 1, v => { s.M = v; draw(); }), slider("SMs", 80, 148, 108, 1, v => { s.sms = v; draw(); }), read), cv)); draw();
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
    read.innerHTML = `S = QKᵀ is ${s.N}×${s.N} = ${(s.N * s.N * s.bytes / 2 ** 20).toFixed(1)} MB; one tile is ${s.Bc}×${s.Bc} = ${(s.Bc * s.Bc * s.bytes / 1024).toFixed(0)} KB in shared memory<br>
      <span class="big">HBM traffic ${(hbmFlash / 2 ** 20).toFixed(1)} MB</span> tiled (O(N·d) per pass × ${tilesQ} Q-tiles) vs <b>${(hbmNaive / 2 ** 20).toFixed(1)} MB</b> materializing S and P<br>
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
  "fixture:memory-hierarchy": memoryHierarchy,
  "fixture:coalescing": coalescing,
  "fixture:tiling-sweep": tilingSweep,
  "fixture:wave-quantization": waveQuantization,
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

// gradient accumulation timeline (lecture_02.py:L719-L731)
export function gradAccumTimeline(root, notice) {
  const s = { B: 64, micro: 16, D: 1024, L: 16 };
  const cv = el("canvas", { width: 640, height: 220 }); const read = el("div", { class: "readout" });
  const draw = () => {
    const ctx = cv.getContext("2d"); ctx.clearRect(0, 0, 640, 220); const n = Math.max(1, Math.round(s.B / s.micro)); const w = 560 / n;
    ctx.fillStyle = "#6b7680"; ctx.font = "12px sans-serif"; ctx.fillText("micro-steps (forward + backward, grads accumulate)", 40, 30); ctx.fillText("activation memory alive", 40, 120); ctx.fillText("gradient buffer", 40, 175);
    for (let i = 0; i < n; i++) { ctx.fillStyle = "#24668d"; ctx.fillRect(40 + i * w, 40, w - 3, 26); ctx.fillStyle = "#fff"; ctx.fillText(`fwd+bwd ${i + 1}`, 44 + i * w, 58); ctx.fillStyle = "#b8582a"; ctx.fillRect(40 + i * w, 130, w - 3, 8 + 60 * s.micro / s.B); ctx.fillStyle = "#3c8d5a"; ctx.fillRect(40 + i * w, 185, w - 3, 14); }
    ctx.fillStyle = "#e3b23c"; ctx.fillRect(40 + n * w - 2, 40, 6, 26); ctx.fillStyle = "#6b7680"; ctx.fillText("optimizer.step(); zero_grad()", 40 + n * w - 120, 84);
    const mb = 2 ** 20;
    read.innerHTML = `${n} micro-steps per update · activation memory 2·micro·D·L = <b>${(2 * s.micro * s.D * s.L / mb).toFixed(1)} MB</b> (full batch: ${(2 * s.B * s.D * s.L / mb).toFixed(1)} MB)<br>FLOPs per update unchanged: 6·B·N · gradient buffer keeps size 2·N throughout<br><span class="muted small">lecture_02.py:L719-L731 (B=64, D=1024, L=16, micro=B/4)</span>`;
  };
  root.append(el("div", { class: "widget" }, el("div", { class: "controls" }, slider("batch B", 16, 512, 64, 16, v => { s.B = v; s.micro = Math.min(s.micro, v); draw(); }), slider("micro-batch", 4, 512, 16, 4, v => { s.micro = Math.min(v, s.B); draw(); }), read), cv)); draw();
}

// activation checkpointing sweep (lecture_02.py:L751-L755, L770-L773)
export function checkpointSweep(root, notice) {
  const s = { L: 16, every: 4, step: 0 };
  const cv = el("canvas", { width: 640, height: 240 }); const read = el("div", { class: "readout" });
  const draw = () => {
    const ctx = cv.getContext("2d"); ctx.clearRect(0, 0, 640, 240); const w = 560 / s.L;
    const ck = l => l % s.every === 0; // checkpointed layers
    // backward position: layers > s.L - step are done; current segment being recomputed
    const cur = s.L - s.step; // layer whose backward is happening (1-indexed from the top)
    const segStart = cur > 0 ? Math.floor((cur - 1) / s.every) * s.every : 0;
    let alive = 0, recompute = 0;
    for (let l = 0; l < s.L; l++) {
      const stored = ck(l), inSeg = s.step > 0 && l >= segStart && l < cur, done = l >= cur && s.step > 0;
      ctx.fillStyle = done ? "#eee" : inSeg ? "#e3b23c" : stored ? "#24668d" : "#f5f2ea";
      ctx.fillRect(40 + l * w, 60, w - 2, 60); ctx.strokeStyle = "#d9d3c7"; ctx.strokeRect(40 + l * w, 60, w - 2, 60);
      if (stored) { ctx.fillStyle = "#fff"; ctx.font = "11px sans-serif"; ctx.fillText("ckpt", 44 + l * w, 95); }
      if (!done && (stored || inSeg)) alive++;
      if (inSeg && !stored) recompute++;
    }
    ctx.fillStyle = "#6b7680"; ctx.font = "12px sans-serif"; ctx.fillText("layers →  (blue = stored at forward, yellow = recomputed for the current backward segment, grey = backward done)", 40, 45);
    if (s.step > 0) { ctx.fillStyle = "#b8582a"; ctx.fillText(`backward at layer ${cur}`, 40 + (cur - 1) * w, 140); }
    const peak = Math.ceil(s.L / s.every) + s.every - 1;
    read.innerHTML = `checkpoint every ${s.every} layers: stored ${Math.ceil(s.L / s.every)} + recompute window ${s.every - 1} → <b>peak ≈ ${peak} layers alive</b> (store-all: ${s.L}; every √L=${Math.round(Math.sqrt(s.L))}: ≈ 2√L)<br>extra compute: about one more forward pass (≈ +33% on 6ND)<br>alive now: ${alive} · recomputed in this segment: ${recompute}<br><span class="muted small">lecture_02.py:L751-L755, L770-L773</span>`;
  };
  root.append(el("div", { class: "widget" }, el("div", { class: "controls" }, slider("layers L", 4, 32, 16, 4, v => { s.L = v; s.step = 0; draw(); }), slider("checkpoint every", 1, 16, 4, 1, v => { s.every = v; s.step = 0; draw(); }), el("div", { style: "margin-top:8px" }, el("button", { onclick: () => { s.step = Math.min(s.step + 1, s.L); draw(); } }, "backward one layer"), " ", el("button", { class: "ghost", onclick: () => { s.step = 0; draw(); } }, "reset")), read), cv)); draw();
}

Object.assign(WIDGETS, {
  "fixture:backward-two-contractions": backwardTwoContractions,
  "fixture:grad-accum-timeline": gradAccumTimeline,
  "fixture:checkpoint-sweep": checkpointSweep,
});

// ========================================================= lecture 3 widgets ==
// d_ff ratio / block parameter calculator (lecture_03/fixtures/block_params.json)
export function blockParams(root, notice) {
  const s = { d: 4096, ratio: 4.0, gated: false, heads: 32, hd: 128 };
  const read = el("div", { class: "readout" });
  const calc = () => { const dff = Math.round(s.ratio * s.d); const ffn = (s.gated ? 3 : 2) * s.d * dff; const attn = s.d * (s.heads * s.hd) * 3 + (s.heads * s.hd) * s.d; return { dff, ffn, attn, wr: s.heads * s.hd / s.d }; };
  const draw = () => { const c = calc(); read.innerHTML = `d_ff = ${c.dff} · FFN params = ${s.gated ? 3 : 2}·d·d_ff = <span class="big">${(c.ffn / 1e6).toFixed(1)} M</span>attention params = 4·d·(h·d_head) = ${(c.attn / 1e6).toFixed(1)} M · h·d_head/d = ${c.wr.toFixed(2)}<br>block total ${((c.ffn + c.attn) / 1e6).toFixed(1)} M<br><span class="muted small">lecture_03.pdf:p37-p43; gated FFN has a third matrix, so 8/3 × d matches 2 × 4 × d</span>`; };
  const presets = [["ReLU 4×", { ratio: 4, gated: false }], ["SwiGLU 8/3×", { ratio: 8 / 3, gated: true }], ["Mistral-style 3.5× gated", { ratio: 3.5, gated: true }], ["T5-11B 64×", { d: 1024, ratio: 64, gated: false, heads: 128, hd: 128 }]];
  root.append(el("div", { class: "widget" }, el("div", { class: "controls" },
    slider("d_model", 256, 32768, 4096, 256, v => { s.d = v; draw(); }), slider("d_ff / d_model", 1, 64, 4, 1 / 12, v => { s.ratio = v; draw(); }, v => v.toFixed(2)),
    el("label", {}, el("input", { type: "checkbox", onchange: e => { s.gated = e.target.checked; draw(); } }), " gated (SwiGLU: third matrix V)"),
    slider("heads", 1, 256, 32, 1, v => { s.heads = v; draw(); }), slider("head_dim", 16, 512, 128, 16, v => { s.hd = v; draw(); }),
    el("div", { style: "margin-top:8px" }, ...presets.map(([n, p]) => el("button", { onclick: () => { Object.assign(s, p); draw(); } }, n)))), read)); draw();
}
Object.assign(WIDGETS, { "fixture:block-params": blockParams });

// ========================================================= lecture 9 widgets ==
// Chinchilla parametric fit L(N,D)=E + A/N^a + B/D^b (Hoffmann et al. 2022, §3.3 constants)
const CH = { E: 1.69, A: 406.4, B: 410.7, a: 0.34, b: 0.28 };
const chLoss = (N, D) => CH.E + CH.A / Math.pow(N, CH.a) + CH.B / Math.pow(D, CH.b);
function chOpt(C) { let best = null; for (let e = 7; e <= 12; e += 0.01) { const N = Math.pow(10, e), D = C / (6 * N); if (D < 1e6) continue; const L = chLoss(N, D); if (!best || L < best.L) best = { N, D, L }; } return best; }
export function chinchillaAllocator(root, notice) {
  const s = { logC: 21, N: null };
  const read = el("div", { class: "readout" });
  const draw = () => { const C = Math.pow(10, s.logC); const o = chOpt(C); const N = s.N || o.N; const D = C / (6 * N); read.innerHTML = `C = 1e${s.logC.toFixed(1)} FLOPs<br><span class="big">N_opt ≈ ${o.N.toExponential(2)} · D_opt ≈ ${o.D.toExponential(2)}</span> D/N ≈ ${(o.D / o.N).toFixed(0)} tokens per parameter · fitted loss ${o.L.toFixed(3)}<br>your N = ${N.toExponential(2)} → D = ${D.toExponential(2)}, loss ${chLoss(N, D).toFixed(3)} (${chLoss(N, D) > o.L + 1e-4 ? "+" + (chLoss(N, D) - o.L).toFixed(3) + " above optimum" : "at optimum"})<br><span class="muted small">fit constants from Hoffmann et al. 2022 §3.3; C = 6ND (lecture_02:six-nd)</span>`; };
  root.append(el("div", { class: "widget" }, el("div", { class: "controls" }, slider("log10 compute budget C", 18, 25, 21, 0.1, v => { s.logC = v; s.N = null; draw(); }, v => `1e${v.toFixed(1)}`), slider("your log10 N (0 = optimal)", 0, 12, 0, 0.1, v => { s.N = v > 0 ? Math.pow(10, v) : null; draw(); }, v => v > 0 ? `1e${v.toFixed(1)}` : "optimal")), read)); draw();
}
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
Object.assign(WIDGETS, { "fixture:chinchilla-allocator": chinchillaAllocator, "fixture:isoflop-sweep": isoflopSweep });

// ========================================================= lecture 7 widgets ==
// Byte accounting for the collectives (lecture_07 KP collective-bytes; lecture_07.py:L325, L363).
// S = the per-rank tensor the lecture passes to the benchmark (size_bytes = numel × element size).
// all-reduce: sent_bytes = 2·S·(W−1) (send + receive), reduce-scatter: S·(W−1) (no 2×); per-rank = sent_bytes / W.
// all-gather is the mirror of reduce-scatter (shards of S/W → S on every rank); all-to-all is not counted in the
// lecture (shown only as a transpose, L201), so its row uses the transpose: rank i keeps 1/W and ships (W−1)/W.
export function collectiveBytes(root, notice) {
  const s = { W: 4, logN: Math.log2(100 * 1024 ** 2), bytes: 4 };
  const cv = el("canvas", { width: 640, height: 230 }); const read = el("div", { class: "readout" });
  const rows = () => {
    const S = 2 ** s.logN * s.bytes, W = s.W, f = (W - 1) / W;
    return [
      { name: "all-gather (shards S/W → S)", total: S * (W - 1), rank: S * f, formula: "S·(W−1)/W received per rank" },
      { name: "reduce-scatter (S → S/W)", total: S * (W - 1), rank: S * f, formula: "S·(W−1) total, no 2×  (L363)" },
      { name: "all-reduce = RS + AG", total: 2 * S * (W - 1), rank: 2 * S * f, formula: "2·S·(W−1) total  (L325)" },
      { name: "all-to-all (transpose)", total: S * (W - 1), rank: S * f, formula: "S·(W−1)/W sent and received" },
    ].map(r => ({ ...r, S }));
  };
  const mib = x => `${fmt(x / 2 ** 20, 1)} MiB`;
  const draw = () => {
    const r = rows(), ctx = cv.getContext("2d"); ctx.clearRect(0, 0, 640, 230);
    const max = Math.max(...r.map(x => x.rank)); let y = 14;
    for (const x of r) {
      const w = Math.max(2, 380 * x.rank / max);
      ctx.fillStyle = "#6b7680"; ctx.font = "12px sans-serif"; ctx.fillText(x.name, 10, y + 13);
      ctx.fillStyle = x.name.startsWith("all-reduce") ? "#b8582a" : "#24668d"; ctx.fillRect(220, y, w, 18);
      ctx.fillStyle = "#25313d"; ctx.fillText(`${mib(x.rank)} / rank`, 226 + w, y + 13); y += 52;
    }
    ctx.fillStyle = "#6b7680"; ctx.fillText("bars: bytes each rank sends + receives; saturate at 2·S as W grows", 10, 226);
    const S = r[0].S;
    read.innerHTML = `S = ${fmt(2 ** s.logN, 0)} elements × ${s.bytes} B = <b>${mib(S)}</b> per rank, W = ${s.W}<br>
      <pre>${r.map(x => `${x.name.padEnd(28)} total ${mib(x.total).padStart(12)}   per rank ${mib(x.rank).padStart(12)}   ${x.formula}`).join("\n")}</pre>
      all-reduce per rank = 2·S·(W−1)/W = <span class="big">${mib(2 * S * (s.W - 1) / s.W)}</span> → effective bandwidth ≈ 2·S/duration, independent of W and of ring vs tree (L330-L333)<br>
      <span class="muted small">provenance: fixture:collective-bytes · lecture_07.py:L322-L333 (all-reduce), L360-L370 (reduce-scatter); all-gather mirror from KP all-gather (L152, L276-L279); all-to-all bytes are not counted in the lecture (L201)</span>`;
  };
  root.append(el("div", { class: "widget" }, el("div", { class: "controls" },
    slider("world size W", 2, 64, 4, 1, v => { s.W = v; draw(); }),
    slider("log2 num_elements", 10, 32, Math.log2(100 * 1024 ** 2), 1, v => { s.logN = v; draw(); }, v => `2^${v} = ${fmt(2 ** v, 0)}`),
    slider("bytes / element", 1, 4, 4, 1, v => { s.bytes = v; draw(); }, v => ({ 1: "fp8", 2: "bf16", 4: "fp32" })[v] || v),
    read), cv)); draw();
}

// ZeRO per-device memory (lecture_07 KP zero-sharding, basis=inference; ZeRO paper §5.1-§5.3, Fig. 1).
// Mixed-precision Adam: 2Ψ fp16 params + 2Ψ grads + KΨ (K = 12: fp32 master + two moments).
// stage 0: 16Ψ · stage 1: 4Ψ + 12Ψ/N · stage 2: 2Ψ + 14Ψ/N · stage 3: 16Ψ/N.
export function zeroMemory(root, notice) {
  const s = { psi: 7.5e9, logN: 6 };
  const K = 12;
  const cv = el("canvas", { width: 640, height: 230 }); const read = el("div", { class: "readout" });
  const stages = () => { const P = s.psi, N = 2 ** s.logN; return [
    { name: "stage 0 (DDP, replicated)", gb: (4 + K) * P / 1e9, f: "16Ψ" },
    { name: "stage 1 (optimizer state sharded)", gb: (4 * P + K * P / N) / 1e9, f: "4Ψ + 12Ψ/N" },
    { name: "stage 2 (+ gradients)", gb: (2 * P + (2 + K) * P / N) / 1e9, f: "2Ψ + 14Ψ/N" },
    { name: "stage 3 (+ parameters, FSDP)", gb: (4 + K) * P / N / 1e9, f: "16Ψ/N" },
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
      <span class="muted small">provenance: fixture:zero-memory · lecture_07 KP zero-sharding (status=supplement, basis=inference) · zero_2019:§5.1-§5.3, Fig. 1 (7.5B, N=64: 120 → 31.4 → 16.6 → 1.9 GB) · lecture_07.py:L68, L152, L183, L387 name the collectives only</span>`;
  };
  root.append(el("div", { class: "widget" }, el("div", { class: "controls" },
    slider("parameters Ψ (B)", 0.5, 1000, 7.5, 0.5, v => { s.psi = v * 1e9; draw(); }),
    slider("log2 devices N", 0, 12, 6, 1, v => { s.logN = v; draw(); }, v => `N = ${2 ** v}`),
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
export function parallelismComm(root, notice) {
  const s = { psi: 70e9, b: 8, seq: 4096, h: 8192, L: 80, bytes: 2, tp: 8, pp: 4, m: 16, zero: 0 };
  const cv = el("canvas", { width: 640, height: 200 }); const read = el("div", { class: "readout" });
  const calc = () => {
    const act = s.b * s.seq * s.h * s.bytes;                                      // one micro-batch activation b·s·h
    const dp = (s.zero === 3 ? 3 : 2) * s.psi * s.bytes;                            // 2·#params (DDP, ZeRO-1) or 3·#params (ZeRO-3)
    const tpLayer = 8 * act * (s.tp - 1) / s.tp;                                    // 8bsh (n−1)/n per layer
    const tp = tpLayer * s.L * s.m;  // every micro-batch runs every layer's all-reduces
    const ppMicro = act;                                                             // bsh per microbatch per boundary
    const pp = 2 * ppMicro * s.m * (s.pp - 1);  // activation forward + gradient backward per boundary, matching TP's 8bsh which counts both
    return { act, dp, tpLayer, tp, ppMicro, pp, bubble: (s.pp - 1) / s.m };
  };
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
// the GQA reduction N/K is lecture_10.py:L378-L387. Defaults are llama2_13b_config (lecture_10.py:L313-L323).
const KV_BYTES = (L, K, H, bytes, S, B) => 2 * L * K * H * bytes * S * B;
export function kvCacheBytes(root, notice) {
  const s = { L: 40, N: 40, K: 40, H: 128, bytes: 2, S: 1024, B: 1 };
  const cv = el("canvas", { width: 640, height: 150 }); const read = el("div", { class: "readout" });
  const draw = () => {
    const ctx = cv.getContext("2d"); ctx.clearRect(0, 0, 640, 150);
    const mha = KV_BYTES(s.L, s.N, s.H, s.bytes, s.S, s.B), cur = KV_BYTES(s.L, s.K, s.H, s.bytes, s.S, s.B), mqa = KV_BYTES(s.L, 1, s.H, s.bytes, s.S, s.B);
    const bars = [[`MHA  (K = N = ${s.N})`, mha, "#b8582a"], [`yours (K = ${s.K})`, cur, "#24668d"], [`MQA  (K = 1)`, mqa, "#3c8d5a"]]; let y = 10;
    for (const [n, v, col] of bars) { const w = Math.max(2, 400 * v / mha); ctx.fillStyle = "#6b7680"; ctx.font = "12px sans-serif"; ctx.fillText(n, 10, y + 13); ctx.fillStyle = col; ctx.fillRect(170, y, w, 18); ctx.fillStyle = "#25313d"; ctx.fillText(`${fmt(v / 1e9, 3)} GB`, 176 + w, y + 13); y += 44; }
    ctx.fillStyle = "#6b7680"; ctx.fillText(`regime: ${s.K === s.N ? "MHA (K = N)" : s.K === 1 ? "MQA (K = 1)" : "GQA (1 < K < N)"} · N/K = ${(s.N / s.K).toFixed(2)}× smaller than MHA`, 10, 146);
    read.innerHTML = `KV bytes = 2 (K and V) · L · K · H · bytes · S · B = 2·${s.L}·${s.K}·${s.H}·${s.bytes}·${s.S}·${s.B} = <span class="big">${fmt(cur / 1e9, 3)} GB</span><br>
      per token per sequence: 2·L·K·H·bytes = ${fmt(2 * s.L * s.K * s.H * s.bytes / 1e6, 3)} MB · GQA reduces the cache by N/K = ${(s.N / s.K).toFixed(2)}×<br>
      <span class="muted small">provenance: fixture:kv-cache-bytes · lecture_10.py:L172 (store an H-vector per B, S, L, K), L299-L302 (S·(K·H)·L·2·2 × B), L378-L387 (MHA K=N, MQA K=1, GQA reduces by N/K), L313-L323 (Llama 2 13B defaults).</span>`;
  };
  root.append(el("div", { class: "widget" }, el("div", { class: "controls" },
    slider("layers L", 1, 128, 40, 1, v => { s.L = v; draw(); }), slider("query heads N", 1, 128, 40, 1, v => { s.N = v; s.K = Math.min(s.K, v); draw(); }),
    slider("kv heads K", 1, 128, 40, 1, v => { s.K = Math.min(v, s.N); draw(); }), slider("head dim H", 32, 256, 128, 32, v => { s.H = v; draw(); }),
    slider("bytes / element", 1, 4, 2, 1, v => { s.bytes = v; draw(); }, v => ({ 1: "fp8", 2: "bf16", 4: "fp32" })[v] || v),
    slider("tokens S", 128, 131072, 1024, 128, v => { s.S = v; draw(); }), slider("batch B", 1, 256, 1, 1, v => { s.B = v; draw(); }),
    read), cv)); draw();
}

// MoE parameter / FLOP budget. lecture_04.pdf:p15 "increase the # experts without affecting FLOPs", p16 "Same FLOP, more param does better",
// p32 shared experts always on. Forward FLOPs/token = 2 × active parameters (lecture_02:six-nd forward share), so a dense model
// with the same active count has identical FLOPs/token and E/k-fold fewer total FFN parameters.
export function moeBudget(root, notice) {
  const s = { E: 64, k: 8, shared: 1, d: 4096, ratio: 1.0, L: 32, attn: 4 };
  const cv = el("canvas", { width: 640, height: 130 }); const read = el("div", { class: "readout" });
  const calc = () => {
    const dff = Math.round(s.ratio * s.d), expert = 3 * s.d * dff, attn = s.attn * s.d * s.d; // gated FFN per expert (lecture_03 block-params); attention 4·d² (lecture_03.pdf:p37-p43)
    const totalL = attn + (s.E + s.shared) * expert, activeL = attn + (s.k + s.shared) * expert;
    return { dff, expert, attn, total: totalL * s.L, active: activeL * s.L, flops: 2 * activeL * s.L };
  };
  const draw = () => {
    const c = calc(), ctx = cv.getContext("2d"); ctx.clearRect(0, 0, 640, 130);
    const bars = [["MoE total params", c.total, "#24668d"], ["MoE active params / token", c.active, "#b8582a"], ["dense of equal active", c.active, "#3c8d5a"]]; let y = 10;
    for (const [n, v, col] of bars) { const w = Math.max(2, 380 * v / c.total); ctx.fillStyle = "#6b7680"; ctx.font = "12px sans-serif"; ctx.fillText(n, 10, y + 13); ctx.fillStyle = col; ctx.fillRect(200, y, w, 18); ctx.fillStyle = "#25313d"; ctx.fillText(`${fmt(v / 1e9, 2)} B`, 206 + w, y + 13); y += 40; }
    read.innerHTML = `per layer: attention ${fmt(c.attn / 1e6, 1)} M + experts (${s.E} routed + ${s.shared} shared) × ${fmt(c.expert / 1e6, 1)} M each (3·d·d_ff, d_ff = ${c.dff})<br>
      <span class="big">${fmt(c.total / 1e9, 1)} B total · ${fmt(c.active / 1e9, 1)} B active</span> → total/active = ${(c.total / c.active).toFixed(1)}× · forward FLOPs/token = 2·active = ${fmt(c.flops, 2)}<br>
      dense model with the same FLOPs/token has ${fmt(c.active / 1e9, 1)} B parameters — the MoE holds ${(c.total / c.active).toFixed(1)}× more at equal FLOPs<br>
      <span class="muted small">provenance: fixture:moe-budget · lecture_04.pdf:p15-p16 (FLOPs unchanged by # experts; same FLOP, more params), p32 (shared experts always on), p54-p56 quote DeepSeek V1 16B/2.8B, V2 236B/21B, V3 671B/37B total/active. FLOPs = 2·params per token from lecture_02:six-nd; per-expert 3·d·d_ff from lecture_03 block-params.</span>`;
  };
  root.append(el("div", { class: "widget" }, el("div", { class: "controls" },
    slider("experts E", 1, 256, 64, 1, v => { s.E = v; s.k = Math.min(s.k, v); draw(); }), slider("top-k", 1, 16, 8, 1, v => { s.k = Math.min(v, s.E); draw(); }),
    slider("shared experts", 0, 4, 1, 1, v => { s.shared = v; draw(); }), slider("d_model", 512, 16384, 4096, 512, v => { s.d = v; draw(); }),
    slider("expert d_ff / d_model", 0.25, 8, 1, 0.25, v => { s.ratio = v; draw(); }, v => v.toFixed(2)), slider("layers", 4, 128, 32, 4, v => { s.L = v; draw(); }),
    read), cv)); draw();
}

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

Object.assign(WIDGETS, { "fixture:kv-cache-bytes": kvCacheBytes, "fixture:moe-budget": moeBudget, "fixture:moe-routing": moeRouting });

// ========================================================= lecture 10 widgets ==
// Decode is memory-bound: each step reads all parameters and the KV cache (lecture_10.py:L297-L308):
//   memory = B·kv_cache_size_per_seq + parameter_size; latency = memory / memory_bandwidth; throughput = B / latency.
// H100: memory_bandwidth 3.35e12 (L322), h100_memory = 80e9 (L351). Defaults are llama2_13b_config (L313-L323) with its 13B parameters.
export function decodeBound(root, notice) {
  const s = { params: 13e9, bytes: 2, S: 1024, B: 1, L: 40, K: 40, H: 128, bw: 3.35e12 };
  const cv = el("canvas", { width: 640, height: 120 }); const read = el("div", { class: "readout" });
  const calc = () => { const p = s.params * s.bytes, kv = KV_BYTES(s.L, s.K, s.H, s.bytes, s.S, s.B), mem = p + kv, lat = mem / s.bw; return { p, kv, mem, lat, tps: s.B / lat }; };
  const draw = () => {
    const c = calc(), ctx = cv.getContext("2d"); ctx.clearRect(0, 0, 640, 120);
    const w = 500, pw = w * c.p / c.mem; ctx.fillStyle = "#24668d"; ctx.fillRect(40, 30, pw, 28); ctx.fillStyle = "#b8582a"; ctx.fillRect(40 + pw, 30, w - pw, 28);
    ctx.fillStyle = "#6b7680"; ctx.font = "12px sans-serif"; ctx.fillText(`bytes read per decode step: parameters ${fmt(c.p / 1e9, 1)} GB (blue) + KV cache ${fmt(c.kv / 1e9, 2)} GB (red)`, 40, 20);
    ctx.fillStyle = c.mem > 80e9 ? "#b8582a" : "#3c8d5a"; ctx.fillText(c.mem > 80e9 ? `memory ${fmt(c.mem / 1e9, 1)} GB > h100_memory 80 GB: does not fit (L351)` : `memory ${fmt(c.mem / 1e9, 1)} GB fits in 80 GB`, 40, 80);
    ctx.fillStyle = "#6b7680"; ctx.fillText(`latency = memory / bandwidth = ${fmt(c.lat * 1e3, 2)} ms per step`, 40, 104);
    read.innerHTML = `<span class="big">≤ ${fmt(c.tps, 1)} tokens/s</span> = B / latency = ${s.B} / (${fmt(c.mem, 2)} B ÷ ${fmt(s.bw, 2)} B/s)<br>
      per-request latency ${fmt(c.lat * 1e3, 2)} ms/token (${fmt(1 / c.lat, 1)} tokens/s for one stream) · larger B amortizes the parameter read but grows the KV term (L355-L357)<br>
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
export function kvCacheGrowth(root, notice) {
  const s = { params: 13e9, bytes: 2, B: 64, L: 40, K: 40, H: 128, prompt: 1024, maxGen: 8192, gpuGB: 80 };
  const cv = el("canvas", { width: 640, height: 260 }); const read = el("div", { class: "readout" });
  const draw = () => {
    const ctx = cv.getContext("2d"); ctx.clearRect(0, 0, 640, 260);
    const pbytes = s.params * s.bytes, budget = s.gpuGB * 1e9 - pbytes, perTok = 2 * s.L * s.K * s.H * s.bytes * s.B;
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
    slider("batch B", 1, 512, 64, 1, v => { s.B = v; draw(); }), slider("layers L", 1, 128, 40, 1, v => { s.L = v; draw(); }), slider("kv heads K", 1, 128, 40, 1, v => { s.K = v; draw(); }), slider("head dim H", 32, 256, 128, 32, v => { s.H = v; draw(); }),
    slider("prompt tokens", 0, 32768, 1024, 128, v => { s.prompt = v; draw(); }), slider("max generated", 256, 65536, 8192, 256, v => { s.maxGen = v; draw(); }), slider("GPU memory (GB)", 16, 192, 80, 8, v => { s.gpuGB = v; draw(); }),
    read), cv)); draw();
}

// Speculative sampling: draft proposes k tokens, target verifies in one pass (lecture_10.py:L513-L522). With i.i.d. per-token acceptance
// probability p, tokens per target pass = 1 + p + … + p^k = (1 − p^{k+1}) / (1 − p) (Leviathan et al. 2022, Eq. 1; the +1 is the token the
// target supplies after the first rejection or at the end). The seeded run draws k uniforms per pass.
export function speculativeDecode(root, notice) {
  const s = { k: 4, p: 0.8, pass: 0, seed: 7 };
  const cv = el("canvas", { width: 640, height: 200 }); const read = el("div", { class: "readout" });
  const expected = () => s.p >= 1 ? s.k + 1 : (1 - Math.pow(s.p, s.k + 1)) / (1 - s.p);
  const run = () => { const rnd = mulberry32(s.seed); const passes = []; for (let i = 0; i < s.pass; i++) { const u = Array.from({ length: s.k }, rnd); let acc = 0; while (acc < s.k && u[acc] < s.p) acc++; passes.push({ u, acc, tokens: acc + 1 }); } return passes; };
  const draw = () => {
    const passes = run(), ctx = cv.getContext("2d"); ctx.clearRect(0, 0, 640, 200);
    const show = passes.slice(-6); const cw = 24;
    show.forEach((ps, row) => { const y = 20 + row * 28; ctx.fillStyle = "#6b7680"; ctx.font = "11px sans-serif"; ctx.fillText(`pass ${passes.length - show.length + row + 1}`, 10, y + 14);
      for (let j = 0; j < s.k; j++) { ctx.fillStyle = j < ps.acc ? "#3c8d5a" : j === ps.acc ? "#b8582a" : "#eee"; ctx.fillRect(70 + j * cw, y, cw - 3, 20); }
      ctx.fillStyle = "#e3b23c"; ctx.fillRect(70 + s.k * cw + 6, y, cw - 3, 20); ctx.fillStyle = "#25313d"; ctx.fillText(`${ps.acc} accepted + 1 target token = ${ps.tokens}`, 70 + (s.k + 1) * cw + 14, y + 14); });
    ctx.fillStyle = "#6b7680"; ctx.font = "12px sans-serif"; ctx.fillText("green = draft token accepted, red = first rejection (target resamples), yellow = token from the target pass", 10, 196);
    const tot = passes.reduce((a, x) => a + x.tokens, 0), acc = passes.reduce((a, x) => a + x.acc, 0);
    read.innerHTML = `expected tokens per target pass = (1 − p^{k+1}) / (1 − p) = (1 − ${s.p}^${s.k + 1}) / ${(1 - s.p).toFixed(2)} = <span class="big">${expected().toFixed(3)}</span> (of at most k+1 = ${s.k + 1}) · expected accepted draft tokens = ${(expected() - 1).toFixed(3)}<br>
      seeded run: ${passes.length} passes → ${tot} tokens (${acc} draft accepted) · mean ${passes.length ? (tot / passes.length).toFixed(3) : "–"} per pass vs ${expected().toFixed(3)} expected<br>
      <span class="muted small">provenance: fixture:speculative-decode · lecture_10.py:L513-L522 (draft p guesses a few tokens, target q verifies in parallel, modified rejection sampling, exact sample), L538-L539 (70B/8B, 8B/1B pairs); the closed form is Leviathan et al. 2022 Eq. 1 under an i.i.d. acceptance rate, a demo simplification (real acceptance is per-position).</span>`;
  };
  root.append(el("div", { class: "widget" }, el("div", { class: "controls" },
    slider("draft length k", 1, 12, 4, 1, v => { s.k = v; s.pass = 0; draw(); }), slider("acceptance prob p", 0, 1, 0.8, 0.05, v => { s.p = v; s.pass = 0; draw(); }, v => v.toFixed(2)),
    slider("seed", 1, 99, 7, 1, v => { s.seed = v; s.pass = 0; draw(); }),
    el("div", { style: "margin-top:8px" }, el("button", { onclick: () => { s.pass++; draw(); } }, "next pass"), " ", el("button", { onclick: () => { s.pass += 20; draw(); } }, "+20"), " ", el("button", { class: "ghost", onclick: () => { s.pass = 0; draw(); } }, "reset")),
    read), cv)); draw();
}

Object.assign(WIDGETS, { "fixture:decode-bound": decodeBound, "fixture:kv-cache-growth": kvCacheGrowth, "fixture:speculative-decode": speculativeDecode });

// ========================================================= lecture 16 widgets ==
// GRPO group-relative advantage: A_i = (r_i − mean(r)) / std(r) (DeepSeekMath §4.1.2, papers/grpo/paper.txt:L933; lecture_16.pdf:p18 "z-score
// within group", p20 "1e-4 stability factor in the std"). std is the unbiased sample std (ddof = 1, torch.std default); the paper does not say
// which, numpy's default is ddof = 0 — the toggle shows both. When std = 0 the lecture's code adds eps; the other option zeroes the advantages.
export function groupAdvantage(root, notice) {
  const s = { text: "1, 0, 0, 1, 1, 0, 0, 0", ddof: 1, zero: "eps", eps: 1e-4 };
  const cv = el("canvas", { width: 640, height: 200 }); const read = el("div", { class: "readout" });
  const parse = () => s.text.split(/[,\s]+/).map(Number).filter(Number.isFinite);
  const stats = (r) => { const n = r.length, mean = r.reduce((a, x) => a + x, 0) / n; const ss = r.reduce((a, x) => a + (x - mean) ** 2, 0); const std = n - s.ddof > 0 ? Math.sqrt(ss / (n - s.ddof)) : 0; const den = std === 0 ? (s.zero === "eps" ? s.eps : 0) : std + (s.zero === "eps" ? s.eps : 0); return { n, mean, std, den, adv: r.map(x => den === 0 ? 0 : (x - mean) / den) }; };
  const draw = () => {
    const r = parse(), ctx = cv.getContext("2d"); ctx.clearRect(0, 0, 640, 200);
    if (r.length < 2) { read.innerHTML = "enter at least two rewards"; return; }
    const st = stats(r), max = Math.max(1, ...st.adv.map(Math.abs)), bw = Math.min(60, 580 / r.length), y0 = 100;
    ctx.strokeStyle = "#d9d3c7"; ctx.beginPath(); ctx.moveTo(30, y0); ctx.lineTo(620, y0); ctx.stroke();
    st.adv.forEach((a, i) => { const h = a / max * 80, x = 40 + i * bw; ctx.fillStyle = a >= 0 ? "#3c8d5a" : "#b8582a"; ctx.fillRect(x, h >= 0 ? y0 - h : y0, bw - 6, Math.abs(h)); ctx.fillStyle = "#6b7680"; ctx.font = "11px sans-serif"; ctx.fillText(`r=${r[i]}`, x, 192); ctx.fillStyle = "#25313d"; ctx.fillText(a.toFixed(2), x, h >= 0 ? y0 - h - 4 : y0 + Math.abs(h) + 12); });
    ctx.fillStyle = "#6b7680"; ctx.font = "12px sans-serif"; ctx.fillText("advantage per rollout (bars) · zero line = group mean", 30, 16);
    read.innerHTML = `G = ${st.n} rewards · mean = ${st.mean.toFixed(4)} · std (ddof = ${s.ddof}${s.ddof === 1 ? ", unbiased, torch.std default" : ", population, numpy default"}) = ${st.std.toFixed(4)}${st.std === 0 ? ` → all rewards equal: ${s.zero === "eps" ? `divide by eps = ${s.eps}, advantages are 0 anyway since r − mean = 0` : "advantages set to 0"}` : s.zero === "eps" ? ` (+ eps ${s.eps} in the denominator)` : ""}<br>
      A_i = (r_i − mean) / std = <span class="big">[${st.adv.map(a => a.toFixed(3)).join(", ")}]</span> · Σ A_i = ${st.adv.reduce((a, x) => a + x, 0).toExponential(1)}<br>
      the /std is not a valid baseline (changes the gradient's scale per group, upweights all-easy / all-hard groups) — lecture_16.pdf:p23-p24, Dr. GRPO drops it<br>
      <span class="muted small">provenance: fixture:group-advantage · papers/grpo/paper.txt:L933 (DeepSeekMath §4.1.2, Â_i = (r_i − mean(r))/std(r)) · lecture_16.pdf:p18 (z-score within group), p19 (mean/var normalization per group), p20 (1e-4 stability factor), p23-p24 (std not an unbiased baseline). No knowledge_points.json for lecture_16 / papers/grpo yet.</span>`;
  };
  const inp = el("input", { type: "text", value: s.text, style: "width:360px" }); inp.addEventListener("input", () => { s.text = inp.value; draw(); });
  root.append(el("div", { class: "widget" }, el("div", { class: "controls" },
    el("label", {}, "group rewards ", inp),
    el("label", {}, el("input", { type: "checkbox", checked: "", onchange: e => { s.ddof = e.target.checked ? 1 : 0; draw(); } }), " unbiased std (ddof = 1)"),
    el("label", {}, el("input", { type: "checkbox", checked: "", onchange: e => { s.zero = e.target.checked ? "eps" : "zero"; draw(); } }), " std = 0 handling: add eps = 1e-4 (unchecked: set advantages to 0)"),
    el("div", { style: "margin-top:8px" }, ...[["binary 3/8", "1, 0, 0, 1, 1, 0, 0, 0"], ["all correct", "1, 1, 1, 1"], ["one outlier", "0, 0, 0, 0, 0, 0, 0, 1"], ["graded", "0.2, 0.9, 0.5, 0.7"]].map(([n, t]) => el("button", { onclick: () => { s.text = t; inp.value = t; draw(); } }, n))),
    read), cv)); draw();
}
Object.assign(WIDGETS, { "fixture:group-advantage": groupAdvantage });

// ========================================================= lecture 15 widgets ==
// Bradley-Terry preference probability (lecture_15/fixtures/bradley_terry.json)
export function bradleyTerry(root, notice) {
  const s = { rw: 1.0, rl: 0.0, shift: 0.0 };
  const read = el("div", { class: "readout" });
  const draw = () => { const d = (s.rw + s.shift) - (s.rl + s.shift); const p = 1 / (1 + Math.exp(-d)); read.innerHTML = `Δ = r(y_w) − r(y_l) = <b>${d.toFixed(2)}</b> nats<br><span class="big">p(chosen ≻ rejected) = σ(Δ) = ${p.toFixed(4)}</span>loss = −log p = ${(-Math.log(p)).toFixed(4)} · ∂loss/∂Δ = −(1−p) = ${(-(1 - p)).toFixed(4)}<br><span class="muted small">the shared shift never moves p: only the difference is identified · lecture_15.pdf:p52-p57</span>`; };
  const presets = [["tie", { rw: 0, rl: 0 }], ["1 nat gap", { rw: 1, rl: 0 }], ["3 nat gap", { rw: 3, rl: 0 }], ["mislabeled", { rw: 0, rl: 2 }]];
  root.append(el("div", { class: "widget" }, el("div", { class: "controls" }, slider("r(y_w) chosen", -10, 10, 1, 0.1, v => { s.rw = v; draw(); }), slider("r(y_l) rejected", -10, 10, 0, 0.1, v => { s.rl = v; draw(); }), slider("shared shift (added to both)", -10, 10, 0, 0.5, v => { s.shift = v; draw(); }), el("div", { style: "margin-top:8px" }, ...presets.map(([n, p]) => el("button", { onclick: () => { Object.assign(s, p); draw(); } }, n)))), read)); draw();
}
// DPO implicit reward r̂ = β·(log π_θ − log π_ref) (lecture_15/fixtures/dpo_implicit_reward.json)
export function dpoImplicitReward(root, notice) {
  const s = { beta: 0.1, lw: -20, rw: -20, ll: -20, rl: -20 };
  const read = el("div", { class: "readout" });
  const draw = () => { const rhw = s.beta * (s.lw - s.rw), rhl = s.beta * (s.ll - s.rl), m = rhw - rhl, p = 1 / (1 + Math.exp(-m)); read.innerHTML = `r̂(y_w) = β·(log π_θ − log π_ref) = <b>${rhw.toFixed(3)}</b> · r̂(y_l) = <b>${rhl.toFixed(3)}</b><br><span class="big">margin ${m.toFixed(3)} → p = σ(margin) = ${p.toFixed(4)}</span>loss = −log p = ${(-Math.log(p)).toFixed(4)} · gradient weight β(1−p) = ${(s.beta * (1 - p)).toFixed(4)}<br><span class="muted small">the step is β(1−p)·[∇log π(y_w) − ∇log π(y_l)]: large when the model ranks the pair wrongly, vanishing when it is already right · lecture_15.pdf:p55-p58</span>`; };
  root.append(el("div", { class: "widget" }, el("div", { class: "controls" }, slider("β", 0.01, 1, 0.1, 0.01, v => { s.beta = v; draw(); }), slider("log π_θ(y_w)", -60, 0, -20, 0.5, v => { s.lw = v; draw(); }), slider("log π_ref(y_w)", -60, 0, -20, 0.5, v => { s.rw = v; draw(); }), slider("log π_θ(y_l)", -60, 0, -20, 0.5, v => { s.ll = v; draw(); }), slider("log π_ref(y_l)", -60, 0, -20, 0.5, v => { s.rl = v; draw(); })), read)); draw();
}
Object.assign(WIDGETS, { "fixture:bradley-terry": bradleyTerry, "fixture:dpo-implicit-reward": dpoImplicitReward });

// ========================================================= lecture 6 widgets ==
// Triton program grid: cdiv(num_elements, BLOCK_SIZE) programs (lecture_06.py L345-L376)
export function tritonProgramGrid(root, notice) {
  const s = { n: 8192, block: 1024, pid: 0 };
  const cv = el("canvas", { width: 640, height: 200 }); const read = el("div", { class: "readout" });
  const draw = () => {
    const ctx = cv.getContext("2d"); ctx.clearRect(0, 0, 640, 200); const nb = Math.ceil(s.n / s.block); const w = 600 / nb;
    for (let b = 0; b < nb; b++) { ctx.fillStyle = b === s.pid ? "#e3b23c" : "#24668d"; ctx.fillRect(20 + b * w, 60, Math.max(1, w - 2), 50); if (w > 26) { ctx.fillStyle = "#fff"; ctx.font = "11px sans-serif"; ctx.fillText(`pid ${b}`, 24 + b * w, 90); } }
    ctx.fillStyle = "#6b7680"; ctx.font = "12px sans-serif"; ctx.fillText(`grid = cdiv(${s.n}, ${s.block}) = ${nb} programs`, 20, 40);
    const start = s.pid * s.block, end = Math.min(s.n, start + s.block);
    ctx.fillText(`program ${s.pid}: offsets = ${start} + arange(${s.block}); mask = offsets < ${s.n} keeps ${end - start} of ${s.block}`, 20, 140);
    read.innerHTML = `each program loads one BLOCK_SIZE slice, computes, stores; the last block is partial when ${s.n} % ${s.block} = ${s.n % s.block} ≠ 0<br><span class="muted small">lecture_06.py:L345-L376</span>`;
  };
  root.append(el("div", { class: "widget" }, el("div", { class: "controls" }, slider("num_elements", 1024, 16384, 8192, 256, v => { s.n = v; s.pid = Math.min(s.pid, Math.ceil(v / s.block) - 1); draw(); }), slider("BLOCK_SIZE", 128, 4096, 1024, 128, v => { s.block = v; s.pid = 0; draw(); }), slider("program id", 0, 63, 0, 1, v => { s.pid = Math.min(v, Math.ceil(s.n / s.block) - 1); draw(); }), read), cv)); draw();
}

// Triton mask: one row per program, BLOCK_SIZE = next_power_of_2(num_cols) (lecture_06.py L448-L484)
export function tritonMask(root, notice) {
  const s = { cols: 6, other: "-inf" };
  const cv = el("canvas", { width: 640, height: 150 }); const read = el("div", { class: "readout" });
  const np2 = n => 1 << Math.ceil(Math.log2(Math.max(1, n)));
  const draw = () => {
    const ctx = cv.getContext("2d"); ctx.clearRect(0, 0, 640, 150); const B = np2(s.cols); const w = Math.min(60, 600 / B);
    for (let i = 0; i < B; i++) { const live = i < s.cols; ctx.fillStyle = live ? "#24668d" : "#f1ede3"; ctx.fillRect(20 + i * w, 40, w - 3, 44); ctx.fillStyle = live ? "#fff" : "#6b7680"; ctx.font = "12px sans-serif"; ctx.fillText(live ? `x${i}` : s.other, 24 + i * w, 66); }
    ctx.fillStyle = "#6b7680"; ctx.fillText(`row of ${s.cols} values · BLOCK_SIZE = next_power_of_2(${s.cols}) = ${B} · mask = col_offsets < ${s.cols}`, 20, 30);
    const why = s.other === "-inf" ? "exp(−inf) = 0, so padded lanes add nothing to the softmax sum" : s.other === "0.0" ? "0 adds nothing to a row sum" : "a wrong `other` corrupts the reduction";
    read.innerHTML = `masked lanes load <b>other = ${s.other}</b>: ${why}<br><span class="muted small">lecture_06.py:L448-L484 (softmax), L510-L535 (row sum uses other=0.0)</span>`;
  };
  root.append(el("div", { class: "widget" }, el("div", { class: "controls" }, slider("num_cols", 1, 64, 6, 1, v => { s.cols = v; draw(); }), el("div", {}, ...["-inf", "0.0", "1.0"].map(o => el("button", { onclick: () => { s.other = o; draw(); } }, `other=${o}`))), read), cv)); draw();
}

// Compression ratio (lecture_01): UTF-8 bytes per token
export function compressionRatio(root, notice) {
  const s = { text: "the cat sat on the mat and the hat sat on the cat", vocab: "bytes" };
  const read = el("div", { class: "readout" });
  const tok = () => { const b = new TextEncoder().encode(s.text).length; if (s.vocab === "bytes") return { tokens: b, bytes: b }; if (s.vocab === "words") return { tokens: s.text.split(/\s+/).filter(Boolean).length, bytes: b }; const words = s.text.split(/\s+/).filter(Boolean); return { tokens: words.reduce((a, w) => a + Math.max(1, Math.ceil(w.length / 3)), 0), bytes: b }; };
  const draw = () => { const t = tok(); read.innerHTML = `<span class="big">${(t.bytes / t.tokens).toFixed(2)} bytes / token</span> ${t.bytes} bytes → ${t.tokens} tokens (${s.vocab})<br>compression ratio = bytes ÷ tokens; higher means shorter sequences for the same text<br><span class="muted small">lecture_01.py:L567-L571 · "bpe-like" is a crude 3-chars-per-token stand-in, not a real tokenizer</span>`; };
  const ta = el("textarea", { rows: 3, style: "width:100%;font:inherit;padding:6px" }); ta.value = s.text; ta.addEventListener("input", () => { s.text = ta.value; draw(); });
  root.append(el("div", { class: "widget" }, el("div", { class: "controls" }, ta, el("div", { style: "margin-top:8px" }, ...["bytes", "bpe-like", "words"].map(v => el("button", { onclick: () => { s.vocab = v; draw(); } }, v)))), read)); draw();
}

Object.assign(WIDGETS, {
  "fixture:triton-program-grid": tritonProgramGrid,
  "fixture:triton-mask": tritonMask,
  "fixture:compression-ratio": compressionRatio,
});

// ============================================== lectures 11-14, 17 widgets ==
// Shared by the seven widgets below: move a slider() made above when a preset changes the state it shows.
const setSlider = (lab, v, fmtv = (x) => x) => { const inp = lab.querySelector("input"), out = lab.querySelector(".readout"); inp.value = v; out.textContent = fmtv(+inp.value); };
const frame = (ctx, W, Hh, x0, y0, x1, y1) => { ctx.clearRect(0, 0, W, Hh); ctx.strokeStyle = "#d9d3c7"; ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(x0, y0); ctx.lineTo(x0, y1); ctx.lineTo(x1, y1); ctx.stroke(); };

// MinHash + LSH S-curve (lecture_14/fixtures/minhash_lsh.json). get_prob_collision is lecture_14.py:L292-L295 verbatim:
// prob_match = sim ** r, prob_collision = 1 - (1 - prob_match) ** b; threshold (1/b) ** (1/r) and 1 - (1 - 1/b) ** b are L316, L321.
export function minhashLsh(root, notice) {
  const s = { b: 5, r: 10, s1: 0.8, s2: 0.9 };
  const SIMS = [0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.98]; // lecture_14.py:L302
  const P = (sim) => 1 - Math.pow(1 - Math.pow(sim, s.r), s.b);
  const thr = () => Math.pow(1 / s.b, 1 / s.r), pAtThr = () => 1 - Math.pow(1 - 1 / s.b, s.b);
  const cv = el("canvas", { width: 640, height: 320 }); const read = el("div", { class: "readout" });
  const draw = () => {
    const ctx = cv.getContext("2d"), x = v => 50 + v * 560, y = v => 280 - v * 250;
    frame(ctx, 640, 320, 50, 30, 610, 280);
    ctx.fillStyle = "#6b7680"; ctx.font = "12px sans-serif";
    for (let t = 0; t <= 10; t += 2) { ctx.fillText((t / 10).toFixed(1), x(t / 10) - 8, 296); ctx.fillText((t / 10).toFixed(1), 22, y(t / 10) + 4); }
    ctx.fillText("Jaccard similarity s", 500, 314); ctx.fillText("P(collide) = 1 − (1 − s^r)^b", 56, 22);
    ctx.strokeStyle = "rgba(107,118,128,.35)"; ctx.setLineDash([4, 4]); ctx.beginPath(); ctx.moveTo(x(0), y(0)); ctx.lineTo(x(1), y(1)); ctx.stroke(); ctx.setLineDash([]);
    ctx.fillText("b = r = 1: P = s", x(0.62), y(0.62) - 6);
    ctx.strokeStyle = "#24668d"; ctx.lineWidth = 2; ctx.beginPath();
    for (let i = 0; i <= 2000; i++) { const v = i / 2000; i ? ctx.lineTo(x(v), y(P(v))) : ctx.moveTo(x(v), y(P(v))); } ctx.stroke(); ctx.lineWidth = 1;
    const t = thr(); ctx.strokeStyle = "#e3b23c"; ctx.beginPath(); ctx.moveTo(x(t), 30); ctx.lineTo(x(t), 280); ctx.stroke();
    ctx.fillStyle = "#b07d12"; ctx.fillText(`s* = (1/b)^(1/r) = ${t.toFixed(4)}`, Math.min(x(t) + 4, 470), 44);
    ctx.fillStyle = "#24668d"; for (const v of SIMS) { ctx.beginPath(); ctx.arc(x(v), y(P(v)), 3, 0, 7); ctx.fill(); }
    [[s.s1, "#b8582a"], [s.s2, "#3c8d5a"]].forEach(([v, col]) => { ctx.fillStyle = col; ctx.beginPath(); ctx.arc(x(v), y(P(v)), 6, 0, 7); ctx.fill(); });
    const row = (v) => `s = ${v.toFixed(2)}: one band matches with s^r = ${fmt(Math.pow(v, s.r), 4)} → <b>P(collide) = ${P(v).toFixed(4)}</b>`;
    read.innerHTML = `n = b·r = ${s.b * s.r} hash functions in ${s.b} bands of ${s.r}<br>
      <span style="color:#b8582a">●</span> ${row(s.s1)}<br><span style="color:#3c8d5a">●</span> ${row(s.s2)}<br>
      <span class="big">threshold s* = ${t.toFixed(4)}</span>at s*: a fixed band matches with 1/b = ${(1 / s.b).toFixed(4)}; P = 1 − (1 − 1/b)^b = ${pAtThr().toFixed(4)} (1 − 1/e = ${(1 - 1 / Math.E).toFixed(4)})<br>
      small dots: the professor's sims ${SIMS.join(", ")} → ${SIMS.map(v => P(v).toFixed(3)).join(", ")}<br>
      <span class="muted small">provenance: fixture:minhash-lsh · lecture_14.py:L237 (Pr[h(A) = h(B)] = Jaccard), L278-L295 (b bands of r rows, get_prob_collision), L298 (s = 0.8, b = 5, r = 10), L302-L309 (sims and the three (b, r) sweeps), L312-L321 (n = 9000, b = 20, r = 450 from dedup_lee_2021; threshold and 1 − (1 − 1/b)^b). Assumes the n hashes are independent, which the lecture does not state.</span>`;
  };
  const sb = slider("bands b", 1, 50, s.b, 1, v => { s.b = v; draw(); }), sr = slider("rows per band r", 1, 500, s.r, 1, v => { s.r = v; draw(); });
  const f2 = v => v.toFixed(3);
  const s1 = slider("readout similarity s₁", 0, 1, s.s1, 0.001, v => { s.s1 = v; draw(); }, f2), s2 = slider("readout similarity s₂", 0, 1, s.s2, 0.001, v => { s.s2 = v; draw(); }, f2);
  const presets = [["L298: b=5, r=10", 5, 10, 0.8], ["L303: b=10, r=10", 10, 10, 0.85], ["L306: b=10, r=20", 10, 20, 0.85], ["L309: b=20, r=20", 20, 20, 0.85], ["Lee et al.: b=20, r=450", 20, 450, 0.99], ["one MinHash: b=r=1", 1, 1, 0.8]];
  const go = ([, b, r, sim]) => { Object.assign(s, { b, r, s1: sim }); setSlider(sb, b); setSlider(sr, r); setSlider(s1, sim, f2); draw(); };
  root.append(el("div", { class: "widget" }, el("div", { class: "controls" }, sb, sr, s1, s2,
    el("div", { style: "margin-top:8px" }, ...presets.map(p => el("button", { onclick: () => go(p) }, p[0]))), read), cv)); draw();
}

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

// Image token budget (lecture_17/fixtures/image_token_budget.json). All formulas are the fixture's: 14-px patches, 336-px encoder
// (lecture_17.py:L71-L72, L78), AnyRes a·b·576 (L162), Qwen2-VL patches/4 + 2 (L220), 2 frames/s and a 16384 cap (L221).
export function imageTokenBudget(root, notice) {
  const s = { mode: "qwen2-dynamic-14px-2x2", W: 224, H: 224, nImg: 1, sec: 0, fps: 2, cap: 16384 };
  const PATCH = 14, ENC = 336, TILE = (ENC / PATCH) ** 2; // 576
  const calc = () => {
    const o = { mode: s.mode };
    if (s.mode === "clip-fixed-336") o.perImg = TILE;
    else if (s.mode === "anyres-tiles-336") { o.a = Math.ceil(s.W / ENC); o.b = Math.ceil(s.H / ENC); o.perImg = o.a * o.b * TILE; }
    else { o.gw = Math.ceil(s.W / PATCH); o.gh = Math.ceil(s.H / PATCH); o.patches = o.gw * o.gh; o.perImg = Math.ceil(o.patches / 4) + 2; }
    o.frames = s.sec * s.fps; o.units = Math.max(s.nImg, o.frames); o.total = o.units * o.perImg; o.over = o.total > s.cap;
    o.allowed = Math.floor(s.cap / o.units);
    o.side = s.mode === "qwen2-dynamic-14px-2x2" && o.allowed > 2 ? PATCH * Math.floor(Math.sqrt(4 * (o.allowed - 2))) : null;
    return o;
  };
  const cv = el("canvas", { width: 640, height: 300 }); const read = el("div", { class: "readout" });
  const draw = () => {
    const o = calc(), ctx = cv.getContext("2d"); ctx.clearRect(0, 0, 640, 300);
    const sc = Math.min(600 / s.W, 260 / s.H), w = s.W * sc, h = s.H * sc, x0 = 20, y0 = 26;
    ctx.fillStyle = "#f1ede3"; ctx.fillRect(x0, y0, w, h); ctx.strokeStyle = "#25313d"; ctx.strokeRect(x0, y0, w, h);
    ctx.font = "12px sans-serif"; ctx.fillStyle = "#6b7680"; ctx.fillText(`${s.W} × ${s.H} px input`, x0, 16);
    if (o.mode === "clip-fixed-336") {
      const side = Math.min(s.W, s.H) * sc; ctx.fillStyle = "rgba(36,102,141,.25)"; ctx.fillRect(x0 + (w - side) / 2, y0 + (h - side) / 2, side, side);
      ctx.strokeStyle = "#24668d"; ctx.strokeRect(x0 + (w - side) / 2, y0 + (h - side) / 2, side, side);
      ctx.fillStyle = "#24668d"; ctx.fillText("center crop → resized to 336 × 336 → 24 × 24 patches", x0 + 6, y0 + h - 6);
    } else if (o.mode === "anyres-tiles-336") {
      ctx.strokeStyle = "#b8582a"; for (let i = 0; i < o.a; i++) for (let j = 0; j < o.b; j++) ctx.strokeRect(x0 + i * ENC * sc, y0 + j * ENC * sc, ENC * sc, ENC * sc);
      ctx.fillStyle = "#b8582a"; ctx.fillText(`${o.a} × ${o.b} tiles of 336 px, 576 tokens each`, x0 + 6, y0 + 16);
    } else {
      const step = PATCH * sc; ctx.strokeStyle = "rgba(36,102,141,.25)";
      if (step >= 3) { for (let i = 0; i <= o.gw; i++) { ctx.beginPath(); ctx.moveTo(x0 + i * step, y0); ctx.lineTo(x0 + i * step, y0 + h); ctx.stroke(); } for (let j = 0; j <= o.gh; j++) { ctx.beginPath(); ctx.moveTo(x0, y0 + j * step); ctx.lineTo(x0 + w, y0 + j * step); ctx.stroke(); } }
      ctx.strokeStyle = "#24668d"; if (step >= 2) for (let i = 0; i < o.gw; i += 2) for (let j = 0; j < o.gh; j += 2) ctx.strokeRect(x0 + i * step, y0 + j * step, 2 * step, 2 * step);
      ctx.fillStyle = "#24668d"; ctx.fillText(`${o.gw} × ${o.gh} patches of 14 px; each 2 × 2 block → 1 token`, x0 + 6, y0 + 16);
    }
    const unitName = o.frames > s.nImg ? "frames" : s.nImg > 1 ? "images" : "image";
    read.innerHTML = `${o.mode === "qwen2-dynamic-14px-2x2" ? `patches = ⌈${s.W}/14⌉·⌈${s.H}/14⌉ = ${o.patches}; tokens = ${o.patches}/4 + 2 = ${o.perImg}${o.patches % 4 ? " (rounded up: odd patch grid)" : ""}` : o.mode === "anyres-tiles-336" ? `a·b·576 = ${o.a}·${o.b}·576` : "fixed (336/14)² = 576 whatever the input size"}<br>
      <span class="big">${fmt(o.perImg, 0)} tokens / ${unitName === "frames" ? "frame" : "image"} · ${fmt(o.total, 0)} total</span>${fmt(o.units, 2)} ${unitName}${o.frames ? ` (${s.sec} s × ${s.fps} frames/s)` : ""} · budget ${fmt(s.cap, 0)}<br>
      ${o.over ? `<b style="color:#b8582a">over budget by ${fmt(o.total - s.cap, 0)}</b>: ${fmt(o.allowed, 0)} tokens allowed per ${unitName === "frames" ? "frame" : "image"}${o.side ? `, i.e. a square frame of about ${o.side} px (14·⌊√(4·(allowed − 2))⌋)` : " — the lecture shrinks tokens by bilinear interpolation (L163)"}` : "fits the budget"}<br>
      <span class="muted small">provenance: fixture:image-token-budget · lecture_17.py:L71-L72 (resize shorter side to 336, center crop), L78 (ViT-L/14@336px), L162-L163 (AnyRes a × b pieces, bilinear interpolation if too many tokens), L166-L170 (same length across single / multi-image / video), L220 (224 × 224, ViT/14, 2 × 2 → 66 tokens), L221 (2 frames/s, max 16384). Patch counts are the fixture's arithmetic; the +2 is the vision start/end delimiters from qwen2_vl_2024, not the lecture; AnyRes's extra thumbnail image (LLaVA-1.5) is not modelled.</span>`;
  };
  const sliders = {
    W: slider("width (px)", 14, 4096, s.W, 14, v => { s.W = v; draw(); }), H: slider("height (px)", 14, 4096, s.H, 14, v => { s.H = v; draw(); }),
    nImg: slider("images in the prompt", 1, 64, 1, 1, v => { s.nImg = v; draw(); }), sec: slider("video seconds (0 = not a video)", 0, 600, 0, 1, v => { s.sec = v; draw(); }),
    fps: slider("frames per second", 0.25, 8, 2, 0.25, v => { s.fps = v; draw(); }), cap: slider("token budget", 64, 262144, 16384, 64, v => { s.cap = v; draw(); }),
  };
  const presets = [["CLIP crop, any input", { mode: "clip-fixed-336", W: 1344, H: 1344, nImg: 1, sec: 0 }], ["AnyRes 2×2", { mode: "anyres-tiles-336", W: 672, H: 672, nImg: 1, sec: 0 }], ["AnyRes 4×4", { mode: "anyres-tiles-336", W: 1344, H: 1344, nImg: 1, sec: 0 }],
    ["Qwen2-VL 224 (L220)", { mode: "qwen2-dynamic-14px-2x2", W: 224, H: 224, nImg: 1, sec: 0 }], ["Qwen2-VL 448", { mode: "qwen2-dynamic-14px-2x2", W: 448, H: 448, nImg: 1, sec: 0 }], ["Qwen2-VL 60 s video at 448", { mode: "qwen2-dynamic-14px-2x2", W: 448, H: 448, nImg: 1, sec: 60, fps: 2, cap: 16384 }]];
  const apply = (p) => { Object.assign(s, p); for (const k of Object.keys(sliders)) setSlider(sliders[k], s[k]); draw(); };
  root.append(el("div", { class: "widget" }, el("div", { class: "controls" },
    el("div", {}, ...["clip-fixed-336", "anyres-tiles-336", "qwen2-dynamic-14px-2x2"].map(m => el("button", { onclick: () => { s.mode = m; draw(); } }, m))),
    ...Object.values(sliders), el("div", { style: "margin-top:8px" }, ...presets.map(([n, p]) => el("button", { onclick: () => apply(p) }, n))), read), cv)); draw();
}

// Perplexity per token vs per byte. lecture_12.py:L62 defines (1/p(D))^(1/|D|) with |D| in tokens; converting to bytes with the
// lecture_01 compression ratio (bytes/token, lecture_01.py:L567-L571) is the KP's author-inference, not lecture content.
// ln ppl_token = L (nats/token); per-byte loss = L / c; bits per byte = L / (c·ln 2); ppl_byte = e^{L/c}.
export function perplexityPerByte(root, notice) {
  const s = { pplA: 8, cA: 2.5, cB: 5, bytes: 2500, tie: true, lossB: Math.log(64) };
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
      <span class="muted small">provenance: fixture:perplexity-per-byte · lecture_12.py:L62 ((1/p(D))^(1/|D|), |D| counted in tokens) · bytes per token = lecture_01.py:L567-L571 get_compression_ratio (presets: bytes 1, GPT-5 tokenizer 2.5 and word tokenizer 5.5 on the lecture_01 examples) · default A ppl 8, 1000 vs 500 tokens is the KP's predict item. The per-byte renormalisation is the KP's author-inference; the lecture does not make it.</span>`;
  };
  const sA = slider("A: token perplexity", 1.1, 100, s.pplA, 0.1, v => { s.pplA = v; draw(); });
  const cA = slider("A: bytes per token c_A", 1, 6, s.cA, 0.1, v => { s.cA = v; draw(); }), cB = slider("B: bytes per token c_B", 1, 6, s.cB, 0.1, v => { s.cB = v; draw(); });
  const lB = slider("B: own loss (nats/token, when untied)", 0.05, 8, +s.lossB.toFixed(2), 0.01, v => { s.lossB = v; draw(); });
  const presets = [["byte tokenizer: 1", 1], ["GPT-5 example: 2.5", 2.5], ["word tokenizer: 5.5", 5.5]];
  root.append(el("div", { class: "widget" }, el("div", { class: "controls" }, sA, cA, cB,
    el("div", { style: "margin-top:6px" }, "B ← ", ...presets.map(([n, c]) => el("button", { onclick: () => { s.cB = c; setSlider(cB, c); draw(); } }, n))),
    el("label", {}, el("input", { type: "checkbox", checked: "", onchange: e => { s.tie = e.target.checked; draw(); } }), " B assigns the same total probability to the text as A"), lB,
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
export function mupVsSpTable(root, notice) {
  const s = { nIn: 1024, nOut: 1024, m: 2, scale: "both" };
  const rows = [
    ["SP", "init std", "1/√n_{l−1}", (i, o) => 1 / Math.sqrt(i)], ["SP", "LR (SGD and Adam)", "Θ(1)", () => 1],
    ["muP", "init std", "(1/√n_{l−1})·min(1, √(n_l/n_{l−1}))", (i, o) => Math.min(1, Math.sqrt(o / i)) / Math.sqrt(i)],
    ["muP", "LR, SGD", "n_l / n_{l−1}", (i, o) => o / i], ["muP", "LR, Adam", "1 / n_{l−1}", (i, o) => 1 / i],
  ];
  const widths = (m) => ({ i: s.nIn * (s.scale !== "out" ? m : 1), o: s.nOut * (s.scale !== "in" ? m : 1) });
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
  "fixture:minhash-lsh": minhashLsh,
  "fixture:filter-retention-ratios": filterRetention,
  "fixture:image-token-budget": imageTokenBudget,
  "fixture:perplexity-per-byte": perplexityPerByte,
  "fixture:wsd-branch-schedule": wsdBranch,
  "fixture:muon-orthogonalize": muonOrthogonalize,
  "fixture:mup-vs-sp-table": mupVsSpTable,
});
