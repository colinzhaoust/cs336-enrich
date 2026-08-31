const states = [
  {
    id: "training-result",
    mode: "Training learns an ordered merge table",
    kicker: "Final training state",
    title: "“the cat in the hat”",
    tokens: ["18 bytes → 12 tokens", "vocab 256 → 259"],
    status: "Counts, vocabulary, rules, and sequence changed.",
    active: null,
    frozen: false,
  },
  {
    id: "freeze-rules",
    mode: "Training has finished",
    kicker: "Learned tokenizer parameters",
    title: "Ordered rules are now frozen",
    tokens: ["th → 256", "256+e → 257", "257+32 → 258"],
    status: "The rule order is part of the learned tokenizer.",
    active: null,
    frozen: true,
  },
  {
    id: "held-out-bytes",
    mode: "Encoding replays the frozen rules",
    kicker: "Held-out UTF-8 bytes",
    title: "“the quick brown fox”",
    tokens: ["116", "104", "101", "32", "+15 unchanged"],
    status: "No pair counts are computed.",
    metric: "19 tokens",
    active: null,
    frozen: true,
    prediction: "Predict: which frozen rule can apply first?",
  },
  {
    id: "replay-rank-0",
    mode: "Encoding replays the frozen rules",
    kicker: "Replay rank 0",
    title: "t + h → 256",
    tokens: ["256", "101", "32", "+15 unchanged"],
    status: "Only the held-out sequence changed.",
    metric: "18 tokens",
    active: 0,
    frozen: true,
    prediction: "Predict: can rank 1 now apply?",
  },
  {
    id: "replay-rank-1",
    mode: "Encoding replays the frozen rules",
    kicker: "Replay rank 1",
    title: "256 + e → 257",
    tokens: ["257", "32", "+15 unchanged"],
    status: "Earlier rule order remains visible.",
    metric: "17 tokens",
    active: 1,
    frozen: true,
    prediction: "Predict: can rank 2 now apply?",
  },
  {
    id: "replay-rank-2",
    mode: "Encoding replays the frozen rules",
    kicker: "Replay rank 2",
    title: "257 + space → 258",
    tokens: ["258", "+15 unchanged"],
    status: "Unmatched bytes pass through unchanged.",
    metric: "16 tokens",
    active: 2,
    frozen: true,
  },
  {
    id: "decode-round-trip",
    mode: "Decode uses the same frozen vocabulary",
    kicker: "Round trip",
    title: "“the quick brown fox” ✓",
    tokens: ["258", "+15 unchanged"],
    status: "Decoded text exactly matches the input.",
    metric: "16 tokens",
    active: 2,
    frozen: true,
  },
  {
    id: "state-ledger",
    mode: "Same learned tokenizer · different mutable state",
    kicker: "State ledger",
    title: "Training versus encoding",
    tokens: [],
    status: "",
    active: null,
    frozen: true,
    ledger: true,
  },
];

const rules = ["t + h → 256", "256 + e → 257", "257 + space → 258"];
const elements = Object.fromEntries(
  ["mode", "state-kicker", "state-title", "tokens", "status-row", "rules", "frozen", "ledger", "prediction", "previous", "replay", "next", "progress"]
    .map((id) => [id, document.getElementById(id)]),
);

let current = Math.max(0, states.findIndex((state) => state.id === new URLSearchParams(location.search).get("state")));
const visited = new Set();
let unlockTimer = null;

function tokenClass(label) {
  if (["256", "257", "258"].includes(label)) return "token merged";
  if (label.startsWith("+")) return "token unchanged";
  return "token";
}

function render({ replay = false } = {}) {
  const state = states[current];
  clearTimeout(unlockTimer);
  elements.mode.textContent = state.mode;
  elements.mode.classList.toggle("encoding", current >= 2);
  elements["state-kicker"].textContent = state.kicker;
  elements["state-title"].textContent = state.title;
  elements.tokens.replaceChildren(...state.tokens.map((label) => {
    const span = document.createElement("span");
    span.className = tokenClass(label);
    span.textContent = label;
    return span;
  }));
  elements["status-row"].replaceChildren();
  if (state.status) {
    const status = document.createElement("span");
    status.textContent = state.status;
    elements["status-row"].append(status);
  }
  if (state.metric) {
    const metric = document.createElement("span");
    metric.className = "metric";
    metric.textContent = state.metric;
    elements["status-row"].append(metric);
  }
  elements.rules.replaceChildren(...rules.map((label, rank) => {
    const item = document.createElement("li");
    item.textContent = label;
    if (rank === state.active) item.className = "active";
    if (state.active !== null && rank < state.active) item.className = "passed";
    return item;
  }));
  elements.frozen.hidden = !state.frozen;
  elements.ledger.hidden = !state.ledger;
  elements.ledger.innerHTML = state.ledger ? `
    <section><h2>Training</h2><ul><li>Pair counts change</li><li>Vocabulary grows</li><li>Merge table grows</li><li>Corpus sequence changes</li></ul></section>
    <section><h2>Encoding</h2><ul><li>No new pair counts</li><li class="frozen-item">Vocabulary is frozen</li><li class="frozen-item">Merge order is frozen</li><li>Held-out sequence changes</li></ul></section>` : "";
  elements.prediction.textContent = state.prediction || "";
  elements.previous.disabled = current === 0;
  elements.next.disabled = current === states.length - 1;
  elements.next.textContent = current === states.length - 1 ? "Complete" : "Next";
  elements.progress.textContent = `${current + 1} / ${states.length}`;
  history.replaceState(null, "", `${location.pathname}?state=${state.id}`);

  const needsPause = state.prediction && !visited.has(state.id) && !replay;
  if (needsPause) {
    elements.next.disabled = true;
    elements.prediction.textContent = `${state.prediction} Reveal unlocks in 3 seconds.`;
    unlockTimer = setTimeout(() => {
      elements.next.disabled = false;
      elements.prediction.textContent = state.prediction;
      visited.add(state.id);
    }, 3000);
  }
}

elements.previous.addEventListener("click", () => { if (current > 0) { current -= 1; render(); } });
elements.next.addEventListener("click", () => { if (current < states.length - 1) { current += 1; render(); } });
elements.replay.addEventListener("click", () => render({ replay: true }));
addEventListener("keydown", (event) => {
  if (event.key === "ArrowLeft" && current > 0) { current -= 1; render(); }
  if (event.key === "ArrowRight" && !elements.next.disabled && current < states.length - 1) { current += 1; render(); }
});

render();
