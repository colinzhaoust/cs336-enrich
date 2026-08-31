const kvOptions = [1, 2, 4, 8];
const states = [
  { id: "prefill-vs-decode", mode: "Decode adds one token at a time", kicker: "Deployment changes the operation", title: "Prefill is parallel; generation is serial", visual: "prefill", note: "A decode step produces one new query token and attends to the existing context." },
  { id: "growing-kv-read", mode: "Each decode step rereads a growing cache", kicker: "KV cache", title: "Save compute, repeat memory movement", visual: "cache", note: "Caching avoids recomputing old keys and values, but each next token reads the growing prior K/V history." },
  { id: "exact-byte-ledger", mode: "Count cache bytes before comparing heads", kicker: "Declared calculation", title: "Separate per-token/per-layer from model total", visual: "formula", note: "Model-wide cache multiplies by sequence length, layer count, and batch. Other inference memory traffic is not included." },
  { id: "mha-repeated-read", mode: "Query heads stay fixed · KV heads control traffic", kicker: "Multi-head attention", title: "8 query heads · 8 K/V heads", visual: "heads", kvHeads: 8, note: "This declared MHA point is the 1× cache-byte reference." },
  { id: "mqa-extreme", mode: "Query heads stay fixed · KV heads control traffic", kicker: "Multi-query attention", title: "8 query heads · 1 shared K/V head", visual: "heads", kvHeads: 1, note: "MQA is the one-K/V-head capacity extreme, not an automatic quality win." },
  { id: "gqa-knob", mode: "Query heads stay fixed · KV heads control traffic", kicker: "Grouped-query attention", title: "KV groups are a capacity/traffic knob", visual: "heads", kvHeads: 2, note: "Move the slider among 1, 2, 4, and 8 K/V heads. Query-head count remains fixed at eight." },
  { id: "evidence-caveat", mode: "Return to official slide 63 · quality is empirical", kicker: "Evidence boundary", title: "Sharing trades capacity for traffic", visual: "evidence", kvHeads: 2, note: "Read the official plots and tables in their cited settings; do not turn them into a universal prescription." },
];

const el = Object.fromEntries(["mode", "kicker", "state-title", "visual", "state-note", "kv-heads", "kv-value", "per-layer", "model-total", "ratio", "evidence", "previous", "replay", "next", "progress"].map((id) => [id, document.getElementById(id)]));
let current = Math.max(0, states.findIndex((state) => state.id === new URLSearchParams(location.search).get("state")));

function bytesFor(kvHeads) {
  const perLayer = 2 * kvHeads * 128 * 2;
  const model = perLayer * 4096 * 32;
  return { perLayer, modelMiB: model / 2 ** 20, ratio: kvHeads / 8 };
}

function setKvHeads(kvHeads) {
  const optionIndex = kvOptions.indexOf(kvHeads);
  el["kv-heads"].value = String(optionIndex < 0 ? 3 : optionIndex);
  const values = bytesFor(kvHeads);
  el["kv-value"].textContent = String(kvHeads);
  el["per-layer"].textContent = values.perLayer >= 1024 ? `${values.perLayer / 1024} KiB` : `${values.perLayer} B`;
  el["model-total"].textContent = `${values.modelMiB} MiB`;
  el.ratio.textContent = values.ratio === 1 ? "1×" : `1/${1 / values.ratio}×`;
}

function token(label, className = "token") {
  const span = document.createElement("span");
  span.className = className;
  span.textContent = label;
  return span;
}

function renderVisual(state) {
  el.visual.replaceChildren();
  if (state.visual === "prefill") {
    const row = document.createElement("div"); row.className = "token-row";
    row.append(...[1,2,3,4,5,6].map(String).map((item) => token(item)));
    const decode = document.createElement("div"); decode.className = "read-meter"; decode.textContent = "new query token → read prior K/V cache";
    el.visual.append(row, decode);
  } else if (state.visual === "cache") {
    const row = document.createElement("div"); row.className = "token-row";
    row.append(...[1,2,3,4].map((item) => token(`K,V ${item}`)));
    const meter = document.createElement("div"); meter.className = "read-meter"; meter.textContent = "step t reads t cached K/V units";
    el.visual.append(row, meter);
  } else if (state.visual === "formula") {
    const equation = document.createElement("div"); equation.className = "formula";
    equation.innerHTML = "<code>per token/layer × sequence × layers × batch</code><span>4 KiB × 4096 × 32 × 1 = 512 MiB for declared MHA</span>";
    el.visual.append(equation);
  } else if (state.visual === "heads") {
    const kvHeads = state.kvHeads ?? kvOptions[Number(el["kv-heads"].value)];
    const kvRow = document.createElement("div"); kvRow.className = "head-row";
    kvRow.append(...Array.from({length: kvHeads}, (_, index) => token(`KV${index + 1}`, `head kv ${kvHeads < 8 ? "shared" : ""}`)));
    const qRow = document.createElement("div"); qRow.className = "head-row";
    qRow.append(...Array.from({length: 8}, (_, index) => token(`Q${index + 1}`, "head")));
    el.visual.append(kvRow, qRow);
  }
}

function render() {
  const state = states[current];
  el.mode.textContent = state.mode;
  el.mode.classList.toggle("evidence-mode", state.visual === "evidence");
  el.kicker.textContent = state.kicker;
  el["state-title"].textContent = state.title;
  el["state-note"].textContent = state.note;
  el.evidence.hidden = state.visual !== "evidence";
  const heads = state.kvHeads ?? 8;
  setKvHeads(heads);
  el["kv-heads"].disabled = state.id !== "gqa-knob";
  renderVisual(state);
  el.previous.disabled = current === 0;
  el.next.disabled = current === states.length - 1;
  el.next.textContent = current === states.length - 1 ? "Complete" : "Next";
  el.progress.textContent = `${current + 1} / ${states.length}`;
  history.replaceState(null, "", `${location.pathname}?state=${state.id}`);
}

el["kv-heads"].addEventListener("input", () => { const heads = kvOptions[Number(el["kv-heads"].value)]; setKvHeads(heads); renderVisual({...states[current], kvHeads: heads}); });
el.previous.addEventListener("click", () => { if (current > 0) { current -= 1; render(); } });
el.next.addEventListener("click", () => { if (current < states.length - 1) { current += 1; render(); } });
el.replay.addEventListener("click", render);
addEventListener("keydown", (event) => {
  if (event.key === "ArrowLeft" && current > 0) { current -= 1; render(); }
  if (event.key === "ArrowRight" && current < states.length - 1) { current += 1; render(); }
});
render();
