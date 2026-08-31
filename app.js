(function () {
  "use strict";

  const model = window.CS336_LECTURE_RUNS;
  const legacy = window.CS336_LECTURES || [];
  const artifactRegistry = window.CS336_AUGMENTATION_REGISTRY_V2 || { artifacts: [], byId: {}, bySlot: {}, byAnchor: {}, supersededSlots: [] };
  const config = window.CS336_SITE_CONFIG || { discussions: { enabled: false } };
  if (!model?.lectures?.length) throw new Error("Lecture run data did not load");

  const state = { lecture: 1, runId: "", anchorId: "", activeRail: null, artifactCleanup: null, scrollScheduled: false, mobile: null, settlingRunId: "", pinnedRunId: "", navigationToken: 0 };
  const assetPromises = new Map();
  const supersededSlots = new Set(artifactRegistry.supersededSlots || []);
  const el = {
    kicker: document.querySelector("#lecture-kicker"),
    title: document.querySelector("#lecture-title"),
    thesis: document.querySelector("#lecture-thesis"),
    links: document.querySelector("#lecture-links"),
    player: document.querySelector("#official-player"),
    runNav: document.querySelector("#run-nav"),
    runs: document.querySelector("#lecture-runs"),
    contextRail: document.querySelector("#context-rail"),
    rail: document.querySelector("#rail-content"),
    nav: document.querySelector("#course-nav"),
    navToggle: document.querySelector(".nav-toggle"),
    discussionStatus: document.querySelector("#discussion-status"),
    toast: document.querySelector("#feedback-toast")
  };

  const anchorSlotMap = {
    "L01-ABSTRACTION-LADDER": "L01-R02-ABSTRACTION", "L01-KNOWLEDGE-TYPES": "L01-R02-TRANSFER", "L01-EFFICIENCY-EQUATION": "L01-R02-EFFICIENCY",
    "L01-TOKENIZER-QUIRKS": "L01-R07-TOKENIZER", "L01-COMPRESSION-RATIO": "L01-R07-RATIO", "L01-UTF8-BYTES": "L01-R08-UTF8",
    "L01-BPE-PAIR-COUNT": "L01-R09-REPLAY", "L01-BPE-MERGE": "L01-R09-REPLAY", "L01-BPE-MERGE-CODE": "L01-R09-REPLAY", "L01-BPE-TRAIN-VS-USE": "L01-R10-TRAIN-USE",
    "L02-FP16-UNDERFLOW": "L02-R02-UNDERFLOW", "L02-BF16-RANGE": "L02-R02-DTYPES", "L02-REARRANGE": "L02-R03-INSPECTOR",
    "L02-MATMUL-2BDK": "L02-R04-STRIP", "L02-MFU": "L02-R04-STRIP", "L02-ROOFLINE": "L02-R05-ROOFLINE", "L02-6ND": "L02-R06-DERIVATION",
    "L02-OPTIMIZER-MEMORY": "L02-R07-BYTE-STACK", "L02-CHECKPOINTING": "L02-R08-CHECKPOINT", "L02-CHECKPOINT-FREQUENCY": "L02-R08-CHECKPOINT",
    "L03-PRE-POST-NORM": "L03-R02-TOPOLOGY", "L03-RMSNORM": "L03-R03-FORMULAS", "L03-NORM-RUNTIME": "L03-R03-MOVEMENT",
    "L03-GLU-GATE": "L03-R04-GATE", "L03-POSITION-FAMILIES": "L03-R05-FAMILIES", "L03-ROPE-RELATIVE": "L03-R05-ROPE", "L03-ROPE-FREQUENCIES": "L03-R05-ROPE", "L03-ROPE-CODE": "L03-R05-ROPE",
    "L03-GLU-DIMENSION": "L03-R06-FF", "L03-Z-LOSS": "L03-R07-STABILITY", "L03-QK-NORM": "L03-R07-STABILITY", "L03-SOFT-CAP": "L03-R07-STABILITY",
    "L03-KV-CACHE": "L03-R08-CACHE", "L03-MQA": "L03-R09-KV", "L03-GQA": "L03-R09-KV", "L03-SLIDING-WINDOW": "L03-R10-WINDOW", "L03-INTERLEAVED-ATTN": "L03-R10-WINDOW"
  };

  const authoritativeLinks = {
    "L01-R03-SOURCE": [["Official executable lecture", "https://github.com/stanford-cs336/lectures/blob/main/lecture_01.py"]],
    "L01-R09-BACKGROUND": [["Gage, A New Algorithm for Data Compression", "https://dl.acm.org/doi/10.5555/177910.177914"], ["Sennrich et al., Neural Machine Translation of Rare Words", "https://arxiv.org/abs/1508.07909"], ["GPT-2 tokenizer", "https://github.com/openai/gpt-2/blob/master/src/encoder.py"]],
    "L02-R02-LOWBITS": [["NVIDIA FP8 primer", "https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/examples/fp8_primer.html"], ["FP8 formats paper", "https://arxiv.org/abs/2209.05433"], ["NVIDIA NVFP4 introduction", "https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/"]],
    "L03-R05-QA": [["RoPE", "https://arxiv.org/abs/2104.09864"], ["Longformer", "https://arxiv.org/abs/2004.05150"]]
  };

  function escapeHTML(value) {
    return String(value ?? "").replaceAll("&", "&amp;").replaceAll("<", "&lt;").replaceAll(">", "&gt;").replaceAll('"', "&quot;").replaceAll("'", "&#039;");
  }

  function formatTime(seconds) {
    const hours = Math.floor(seconds / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    const secs = seconds % 60;
    return hours ? `${hours}:${String(minutes).padStart(2, "0")}:${String(secs).padStart(2, "0")}` : `${String(minutes).padStart(2, "0")}:${String(secs).padStart(2, "0")}`;
  }

  function lectureByNumber(number) { return model.lectures.find((item) => item.lecture === number) || model.lectures[0]; }
  function runById(id) { for (const lecture of model.lectures) { const run = lecture.runs.find((item) => item.id === id); if (run) return { lecture, run }; } return null; }
  function slotForAnchor(id) { return artifactRegistry.byAnchor?.[id]?.slotId || anchorSlotMap[id] || ""; }
  function youtubeId(url) { return new URL(url).searchParams.get("v"); }
  function watchUrl(lecture, seconds) { return `${lecture.videoUrl}&t=${seconds}s`; }
  function sourceLabel(lecture) { return lecture.sourceUrl.endsWith(".pdf") ? "Official slide deck" : "Official lecture code"; }
  function slotTypeLabel(type) { return { "formula-comparison": "Formula comparison", table: "Comparison table", "background-link": "Background", "interactive-demo": "Interactive", "slow-manim": "Slow demonstration" }[type] || type; }

  function table(headers, rows) {
    return `<div class="table-wrap"><table><thead><tr>${headers.map((item) => `<th>${escapeHTML(item)}</th>`).join("")}</tr></thead><tbody>${rows.map((row) => `<tr>${row.map((cell) => `<td>${cell}</td>`).join("")}</tr>`).join("")}</tbody></table></div>`;
  }

  function readySupplement(slot) {
    const id = slot.id;
    if (id === "L01-R02-ABSTRACTION") return table(["Level", "Fast path", "Editable surface", "If it leaks"], [["Prompt/API", "Hosted model", "Prompt, tools, context", "Change prompt or provider"], ["Fine-tune", "Released weights", "Data and selected parameters", "Change the training recipe"], ["From scratch", "Build the stack", "Tokenizer through systems", "Change the mechanism"]]);
    if (id === "L01-R02-TRANSFER") return table(["Knowledge", "How checked", "Accessible-scale value", "Frontier caveat"], [["Mechanics", "Executable invariants", "High", "Implementations still vary"], ["Mindset", "Accounting and scaling", "High", "The binding resource can change"], ["Intuition", "Empirical behavior", "Mixed", "May not transfer across scale"]]);
    if (id === "L01-R02-EFFICIENCY") return `<div class="formula"><code>accuracy = efficiency × resources</code><p>A framing, not a predictive law. Data, compute, memory, bandwidth, and evaluation are multidimensional.</p></div>`;
    if (id === "L01-R04-WORKFLOW") return `<ol class="process"><li>Implement locally</li><li>Run unit tests</li><li>Run on cluster</li><li>Benchmark or train</li><li>Compare on leaderboard</li></ol>`;
    if (id === "L01-R05-6ND") return `<div class="formula"><code>C ≈ 6ND</code><dl><div><dt>C</dt><dd>training FLOPs</dd></div><div><dt>N</dt><dd>parameters</dd></div><div><dt>D</dt><dd>training tokens</dd></div></dl><p>Lecture 2 derives the approximation. Long-context and inference costs can change the choice.</p></div>`;
    if (id === "L01-R05-RESOURCE-MATRIX") return table(["Assignment", "Main decision", "Scarce-resource view"], [["1 · Basics", "Make the LM correct and stable", "Compute and memory"], ["2 · Systems", "Move data efficiently", "Bandwidth"], ["3 · Scaling", "Choose a recipe before the large run", "Compute budget"], ["4 · Data", "Build and evaluate the mixture", "Useful data"], ["5 · Alignment", "Learn from weaker feedback", "Samples and systems"]]);
    if (id === "L01-R07-RATIO") return `<div class="formula"><code>compression ratio = UTF-8 bytes / emitted tokens</code><p><strong>Lecture example:</strong> 20 / 8 = 2.5 bytes per token. Shorter sequences help attention, but a larger vocabulary increases embedding cost and sparsity.</p></div>`;
    if (id === "L01-R08-BASELINES") return table(["Atom", "Coverage", "Vocabulary", "Main failure"], [["Character", "Unicode code points", "Large, sparse", "Rare entries"], ["Byte", "Universal", "256", "Long sequences"], ["Word", "Corpus-dependent", "Large", "Unseen words collapse to UNK"], ["BPE", "Byte fallback", "Learned", "Heuristic boundaries"]]);
    if (id === "L01-R10-PRODUCTION") return table(["Toy reference", "Assignment concern"], [["Scan every merge", "Index applicable merges"], ["One full string", "Pretokenize into chunks"], ["Ordinary bytes", "Preserve special tokens"], ["Clear Python loops", "Efficient data structures"]]);
    if (id === "L02-R01-LEDGERS") return table(["Opening estimate", "Known now", "Still to derive"], [["70B × 15T on 1,024 H100s", "N, D, GPU count", "6ND, dense BF16 peak, MFU"], ["Model capacity on 8 × 80 GB", "HBM capacity", "2+2+4+4 bytes/parameter; activations separate"]]);
    if (id === "L02-R02-DTYPES") return table(["Format", "S / E / F", "Bytes", "Smallest subnormal, approx.", "Lecture use"], [["FP32", "1 / 8 / 23", "4", "1.40e-45", "Range and resolution baseline"], ["FP16", "1 / 5 / 10", "2", "5.96e-8", "Fine resolution, narrow range"], ["BF16", "1 / 8 / 7", "2", "9.18e-41", "FP32-like range, coarse resolution"]]);
    if (id === "L02-R02-UNDERFLOW") return `<div class="underflow-demo"><code>value = 1e-8</code><button type="button" data-reveal-underflow>Predict, then reveal</button><div class="demo-result" hidden><span>FP16 → <strong>0</strong></span><span>BF16 → <strong>≈ 1.0e-8</strong></span></div></div>`;
    if (id === "L02-R03-INSPECTOR") return `<div class="stepper" data-stepper="einops"><div class="stepper-stage"><code>input: [batch, total_hidden=8]</code></div><div class="stepper-controls"><button type="button" data-step="-1">Previous</button><span>1 / 3</span><button type="button" data-step="1">Next</button></div><script type="application/json">["input: [batch, total_hidden=8]","split: [batch, heads=2, hidden1=4]","join: [batch, heads×hidden2=8]"]</script></div>`;
    if (id === "L02-R04-STRIP") return `<div class="formula-stack"><code>work ≈ 2BDK FLOPs</code><code>rate = work / synchronized seconds</code><code>MFU = measured rate / dense dtype peak</code><p>The denominator must name dtype and dense versus sparse hardware mode.</p></div>`;
    if (id === "L02-R04-TIMING") return `<ol class="process compact"><li>Synchronize</li><li>Run</li><li>Synchronize</li><li>Repeat</li></ol>`;
    if (id === "L02-R07-BYTE-STACK") return table(["Persistent tensor", "Dtype", "Bytes / parameter"], [["Parameter", "BF16", "2"], ["Gradient", "BF16", "2"], ["Adam first moment", "FP32", "4"], ["Adam second moment", "FP32", "4"], ["Subtotal", "", "12"]]) + `<p class="caveat">Activation memory is a separate function of batch, sequence length, layers, and hidden sizes.</p>`;
    if (id === "L02-R08-COMPARE") return table(["Technique", "What changes", "What persists", "Cost"], [["Gradient accumulation", "Logical batch is split", "Gradient buffer", "Sequential microbatches"], ["Checkpointing", "Some forward states discarded", "Selected checkpoints", "Recompute in backward"]]);
    if (id === "L03-R01-BEFORE-AFTER") return table(["Original Transformer", "A1 modern baseline"], [["Post-norm", "Pre-norm"], ["Sin/cos absolute", "RoPE"], ["ReLU", "SwiGLU"], ["Bias", "No bias"]]);
    if (id === "L03-R03-FORMULAS") return `<div class="formula-stack"><code>LayerNorm(x) = γ ⊙ (x − μ) / √(σ² + ε) + β</code><code>RMSNorm(x) = γ ⊙ x / √(mean(x²) + ε)</code><p>RMSNorm removes centering. Runtime still depends on reductions and memory movement, not FLOP count alone.</p></div>`;
    if (id === "L03-R04-ACTIVATIONS") return table(["Function", "Expression", "Role"], [["ReLU", "max(0,x)", "Baseline nonlinearity"], ["GELU", "xΦ(x)", "Smooth gate-like weighting"], ["Swish", "xσ(x)", "Self-gated activation"]]);
    if (id === "L03-R04-PARALLEL") return table(["Claimed win", "Evidence gap", "Observed caveat"], [["Shared norm and fused matmul", "Few clean controlled ablations", "Later adoption retreated; expressiveness/depth may trade off"]]);
    if (id === "L03-R05-FAMILIES") return table(["Method", "Where position enters", "Behavior"], [["Sin/cos", "Input embedding", "Absolute basis"], ["Learned absolute", "Input embedding", "Learned positions"], ["Relative bias", "Attention logits", "Relative offset"], ["RoPE", "Q and K", "Relative inner product through rotation"]]);
    if (id === "L03-R06-FF") return `<div class="calculator"><label>Model width <input type="range" min="256" max="4096" step="256" value="1024" data-ff-width></label><p><code>d_model = <span>1024</span></code></p><p>Standard 4×: <strong data-ff-standard>4096</strong><br>Parameter-matched GLU ≈8/3×: <strong data-ff-glu>2731</strong></p></div>`;
    if (id === "L03-R06-REG") return table(["Intuition", "Observed practice", "Mechanism caveat"], [["One pass over huge data suggests little classical overfitting", "Many models still use weight decay", "Benefit may interact with optimizer and LR schedule, not train/validation gap"]]);
    if (id === "L03-R07-STABILITY") return table(["Location", "Intervention", "Controls", "Caveat"], [["Output softmax", "z-loss", "log normalizer growth", "Regularization strength matters"], ["Attention softmax", "QK norm", "Q/K scale", "Architecture-specific adoption"], ["Attention logits", "Soft cap", "logit magnitude", "More conservative; may reduce quality"]]);
    if (id === "L03-R08-ROADMAP") return table(["Branch now", "Cost addressed", "Deferred"], [["MQA / GQA", "Decode KV-cache traffic", ""], ["Local / sliding attention", "Long-context attention", "SSM and alternatives → Lecture 4"]]);
    if (id === "L03-R08-CACHE") return `<div class="calculator"><label>Cached tokens <input type="range" min="256" max="8192" step="256" value="2048" data-cache-tokens></label><p>Each decoded token rereads cached K/V for <strong><span data-cache-value>2,048</span></strong> prior positions. Cache avoids recomputing K/V, but shifts the bottleneck toward memory traffic.</p></div>`;
    if (id === "L03-R09-EVIDENCE") return table(["Structure", "KV groups", "Cache traffic", "Quality interpretation"], [["MHA", "one per Q head", "Highest", "Most KV expressiveness"], ["MQA", "one total", "Lowest", "Extreme may lose quality"], ["GQA", "intermediate", "Tunable", "Read with slide 63 latency/quality evidence"]]);
    if (id === "L03-R10-WINDOW") return `<div class="window-demo"><label>Window <input type="range" min="1" max="6" value="2" data-window></label><label>Layers <input type="range" min="1" max="6" value="3" data-layers></label><p>Local receptive reach: <strong data-window-result>6 tokens</strong>. A periodic full-attention layer enables global mixing instead of only gradual expansion.</p></div>`;
    const links = authoritativeLinks[id] || slot.content?.links?.map((url, index) => [`Official resource ${index + 1}`, url]);
    if (links) return `<ul class="resource-list">${links.map(([label, url]) => `<li><a href="${url}" target="_blank" rel="noreferrer">${escapeHTML(label)} ↗</a></li>`).join("")}</ul>`;
    if (slot.type === "table" && slot.content?.rows?.length) return `<ul class="plain-list">${slot.content.rows.map((item) => `<li>${escapeHTML(item)}</li>`).join("")}</ul>`;
    return "";
  }

  function artifactSources(artifact) {
    return `<ul class="artifact-sources">${artifact.sources.map((source) => `<li><a href="${escapeHTML(source.href)}" target="_blank" rel="noreferrer">${escapeHTML(source.label)} ↗</a></li>`).join("")}</ul>`;
  }

  function artifactTemplate(slot, artifact, activeSlotId) {
    const video = artifact.optionalVideo ? `<details class="artifact-video" data-artifact-video-details="${escapeHTML(artifact.id)}">
      <summary>${escapeHTML(artifact.optionalVideo.title)} <span>${escapeHTML(artifact.optionalVideo.durationLabel)}</span></summary>
      <div class="artifact-video-stage" data-artifact-video-stage="${escapeHTML(artifact.id)}">
        <img src="${escapeHTML(artifact.optionalVideo.poster)}" alt="${escapeHTML(artifact.optionalVideo.alt)}" loading="lazy" width="1920" height="1080">
        <button type="button" data-load-artifact-video="${escapeHTML(artifact.id)}">Load video controls</button>
      </div>
    </details>` : "";
    return `<section id="slot-${escapeHTML(slot.id)}" class="supplement artifact-shell is-ready${activeSlotId === slot.id ? " is-targeted" : ""}" data-slot-id="${escapeHTML(slot.id)}" data-artifact-id="${escapeHTML(artifact.id)}">
      <div class="supplement-heading"><span>${escapeHTML(artifact.typeLabel)}</span><strong>Finished</strong></div>
      <div class="artifact-overview" data-artifact-overview>
        <p class="artifact-placement"><time>${escapeHTML(artifact.placement.insertionAfter)}</time> ${escapeHTML(artifact.placement.label)}</p>
        <h4>${escapeHTML(artifact.title)}</h4>
        <p class="artifact-caption">${escapeHTML(artifact.caption)}</p>
        <p class="artifact-alt"><strong>Visual description.</strong> ${escapeHTML(artifact.alt)}</p>
        <p class="artifact-provenance"><strong>Evidence.</strong> ${escapeHTML(artifact.provenance)}</p>
        ${artifactSources(artifact)}
      </div>
      <button class="artifact-load" type="button" data-load-artifact="${escapeHTML(artifact.id)}">Open ${escapeHTML(artifact.typeLabel.toLowerCase())}</button>
      <div class="artifact-mount" data-artifact-mount="${escapeHTML(artifact.id)}" aria-live="polite"></div>
      ${video}
      <p class="artifact-caveat"><strong>Caveat.</strong> ${escapeHTML(artifact.caveat)}</p>
    </section>`;
  }

  function supplementTemplate(slot, activeSlotId) {
    const artifact = artifactRegistry.bySlot?.[slot.id];
    if (artifact?.status === "finished") return artifactTemplate(slot, artifact, activeSlotId);
    const body = readySupplement(slot);
    const planned = !body;
    return `<section id="slot-${escapeHTML(slot.id)}" class="supplement ${planned ? "is-planned" : "is-ready"}${activeSlotId === slot.id ? " is-targeted" : ""}" data-slot-id="${escapeHTML(slot.id)}">
      <div class="supplement-heading"><span>${slotTypeLabel(slot.type)}</span><strong>${planned ? "Planned" : "Augmentation"}</strong></div>
      <p class="supplement-reason">${escapeHTML(slot.pedagogicalReason)}</p>
      ${planned ? `<div class="planned-note"><strong>Not published yet.</strong><p>${slot.type === "slow-manim" ? `The former short render is intentionally hidden. Rebuild target: ${escapeHTML(slot.content?.targetSeconds || "paced, source-aligned delivery")}.` : "This slot records an approved teaching need, not finished learner material."}</p></div>` : body}
    </section>`;
  }

  function railTemplate(lecture, run, activeSlotId = "") {
    const slots = run.augmentationSlots.filter((slot) => !supersededSlots.has(slot.id));
    if (!slots.length) return `<div class="rail-empty"><strong>No detour here.</strong><p>This moment stays uninterrupted so the original transition remains intact.</p></div>`;
    return `<div class="rail-run-head"><span>${escapeHTML(run.id)} · ${formatTime(run.startSeconds)}–${formatTime(run.endSeconds)}</span><strong>${escapeHTML(run.title)}</strong></div>${slots.map((slot) => supplementTemplate(slot, activeSlotId)).join("")}`;
  }

  function sourceLinks(lecture, run) {
    return `<div class="source-links"><a href="${watchUrl(lecture, run.startSeconds)}" target="_blank" rel="noreferrer">Watch ${formatTime(run.startSeconds)}–${formatTime(run.endSeconds)} ↗</a><a href="${lecture.sourceUrl}" target="_blank" rel="noreferrer">${sourceLabel(lecture)} ↗</a></div>`;
  }

  function feedbackTemplate(run) {
    const artifactAnchors = artifactRegistry.artifacts.filter((artifact) => artifact.runId === run.id).flatMap((artifact) => Object.keys(artifact.anchorTargets || {}));
    const anchors = [...new Set([...run.anchors, ...artifactAnchors])];
    if (!anchors.length) return `<span class="no-anchor-feedback">No legacy anchor in this transition.</span>`;
    return `<div class="run-feedback"><button type="button" data-open-feedback="${run.id}">Comment on this run</button><div class="feedback-panel" data-feedback-panel="${run.id}" hidden><label>Closest stable anchor<select>${anchors.map((id) => `<option value="${id}">${id}</option>`).join("")}</select></label><p>Which source sentence, code line, or slide did this supplement clarify or obscure?</p><button type="button" data-mount-discussion="${run.id}">Open discussion</button><div class="giscus-slot"></div></div></div>`;
  }

  function runTemplate(lecture, run, index) {
    const next = lecture.runs[index + 1];
    const legacyAnchors = run.anchors.map((id) => `<span id="${escapeHTML(id)}" class="legacy-anchor" data-slot-target="${escapeHTML(slotForAnchor(id))}" aria-hidden="true"></span>`).join("");
    const evidence = run.restoredEvidence?.length ? `<details class="restored-evidence"><summary>Original transitions and evidence restored</summary><ul>${run.restoredEvidence.map((item) => `<li>${escapeHTML(item)}</li>`).join("")}</ul></details>` : "";
    return `<article id="${run.id}" class="lecture-run" data-run-id="${run.id}" tabindex="-1">
      ${legacyAnchors}
      <header class="run-heading"><p><span>Run ${String(index + 1).padStart(2, "0")}</span><time>${formatTime(run.startSeconds)}–${formatTime(run.endSeconds)}</time></p><h3>${escapeHTML(run.title)}</h3></header>
      <div class="source-body">
        <p class="content-label professor-label">Professor's teaching move</p>
        <p class="professor-intent">${escapeHTML(run.professorIntent)}</p>
        ${sourceLinks(lecture, run)}
        <div class="source-reference"><span>Original source position</span>${run.sourceRefs.map((ref) => `<code>${escapeHTML(ref)}</code>`).join("")}</div>
        <div class="paraphrase"><p class="content-label">Our source-faithful paraphrase</p><p>${escapeHTML(run.originalSummary)}</p></div>
        ${run.correctedClaim ? `<p class="live-caveat"><strong>Delivery caveat.</strong> ${escapeHTML(run.correctedClaim)}</p>` : ""}
        ${evidence}
        <div class="mobile-rail-slot" data-mobile-rail="${run.id}"></div>
      </div>
      <footer class="run-footer"><div class="transition"><span>${next ? "Transition" : "Lecture return"}</span><p>${next ? `Next, the lecture moves to “${escapeHTML(next.title)}”.` : "The lecture closes by returning to its opening frame and handing off to the next topic."}</p></div>${feedbackTemplate(run)}</footer>
    </article>`;
  }

  function renderPlayer(lecture, run) {
    el.player.innerHTML = `<div class="player-copy"><span>Official Stanford Online recording</span><strong>${escapeHTML(run.title)}</strong><p>${formatTime(run.startSeconds)}–${formatTime(run.endSeconds)} · ${escapeHTML(lecture.instructor)}</p></div><button type="button" data-load-official="${lecture.lecture}" data-start="${run.startSeconds}">Load video at ${formatTime(run.startSeconds)}</button><a href="${watchUrl(lecture, run.startSeconds)}" target="_blank" rel="noreferrer">Watch on YouTube ↗</a>`;
  }

  function resolveAsset(path) { return new URL(path, document.baseURI).href; }

  function loadStyle(path) {
    const href = resolveAsset(path);
    const key = `style:${href}`;
    if (assetPromises.has(key)) return assetPromises.get(key);
    const existing = [...document.styleSheets].find((sheet) => sheet.href === href);
    if (existing) return Promise.resolve();
    const promise = new Promise((resolve, reject) => {
      const link = document.createElement("link");
      link.rel = "stylesheet";
      link.href = href;
      link.dataset.augmentationAsset = path;
      link.addEventListener("load", resolve, { once: true });
      link.addEventListener("error", () => reject(new Error(`Could not load ${path}`)), { once: true });
      document.head.append(link);
    });
    assetPromises.set(key, promise);
    return promise;
  }

  function loadScript(path) {
    const src = resolveAsset(path);
    const key = `script:${src}`;
    if (assetPromises.has(key)) return assetPromises.get(key);
    const promise = new Promise((resolve, reject) => {
      const script = document.createElement("script");
      script.src = src;
      script.async = false;
      script.dataset.augmentationAsset = path;
      script.addEventListener("load", resolve, { once: true });
      script.addEventListener("error", () => reject(new Error(`Could not load ${path}`)), { once: true });
      document.head.append(script);
    });
    assetPromises.set(key, promise);
    return promise;
  }

  function cleanupArtifact() {
    state.artifactCleanup?.();
    state.artifactCleanup = null;
  }

  function activeRail() { return state.activeRail || el.rail; }

  async function loadArtifact(artifactId, requestedState = "", focus = true) {
    const artifact = artifactRegistry.byId?.[artifactId];
    const shell = activeRail().querySelector(`[data-artifact-id="${CSS.escape(artifactId)}"]`);
    const mount = shell?.querySelector(`[data-artifact-mount="${CSS.escape(artifactId)}"]`);
    const button = shell?.querySelector(`[data-load-artifact="${CSS.escape(artifactId)}"]`);
    if (!artifact || !mount || mount.dataset.loaded === "true") return;
    const initialState = requestedState || artifact.primary.defaultState || "";
    button?.setAttribute("aria-busy", "true");
    if (button) button.textContent = "Loading…";
    mount.innerHTML = `<p class="artifact-loading" role="status">Loading finished augmentation…</p>`;
    try {
      cleanupArtifact();
      if (artifact.primary.kind === "iframe") {
        const iframe = document.createElement("iframe");
        const source = new URL(resolveAsset(artifact.primary.path));
        if (initialState) source.searchParams.set("state", initialState);
        iframe.src = source.href;
        iframe.title = artifact.primary.title;
        iframe.loading = "lazy";
        iframe.dataset.artifactFrame = artifact.id;
        mount.replaceChildren(iframe);
        let detachFrameListeners = () => {};
        await new Promise((resolve, reject) => {
          iframe.addEventListener("load", () => {
            const frameDocument = iframe.contentDocument;
            const resize = () => {
              const height = frameDocument?.documentElement?.scrollHeight;
              if (height) iframe.style.height = `${height + 2}px`;
            };
            const scheduleResize = () => requestAnimationFrame(resize);
            resize();
            frameDocument?.addEventListener("click", scheduleResize);
            frameDocument?.addEventListener("keydown", scheduleResize);
            detachFrameListeners = () => {
              frameDocument?.removeEventListener("click", scheduleResize);
              frameDocument?.removeEventListener("keydown", scheduleResize);
            };
            if (focus) iframe.focus();
            resolve();
          }, { once: true });
          iframe.addEventListener("error", () => reject(new Error(`Could not load ${artifact.primary.path}`)), { once: true });
        });
        state.artifactCleanup = () => detachFrameListeners();
      } else if (artifact.primary.kind === "mount") {
        await Promise.all((artifact.primary.styles || []).map(loadStyle));
        for (const path of artifact.primary.scripts || []) await loadScript(path);
        const api = window[artifact.primary.global];
        if (!api || typeof api[artifact.primary.method] !== "function") throw new Error(`${artifact.primary.global}.${artifact.primary.method} is unavailable`);
        mount.replaceChildren();
        const instance = api[artifact.primary.method](mount, { stepId: initialState });
        state.artifactCleanup = () => instance?.destroy?.();
        if (artifact.selfDescribing) shell.classList.add("is-self-describing-loaded");
        if (focus) mount.querySelector("button")?.focus();
      }
      mount.dataset.loaded = "true";
      if (button) button.hidden = true;
    } catch (error) {
      mount.innerHTML = `<p class="artifact-error" role="alert"><strong>Could not load this augmentation.</strong> ${escapeHTML(error.message)}</p>`;
      if (button) { button.hidden = false; button.removeAttribute("aria-busy"); button.textContent = "Try again"; }
    }
  }

  function loadArtifactVideo(artifactId) {
    const artifact = artifactRegistry.byId?.[artifactId];
    const stage = activeRail().querySelector(`[data-artifact-video-stage="${CSS.escape(artifactId)}"]`);
    if (!artifact?.optionalVideo || !stage || stage.querySelector("video")) return;
    const video = document.createElement("video");
    video.controls = true;
    video.preload = "metadata";
    video.poster = resolveAsset(artifact.optionalVideo.poster);
    video.setAttribute("aria-label", artifact.optionalVideo.alt);
    const source = document.createElement("source");
    source.src = resolveAsset(artifact.optionalVideo.path);
    source.type = "video/mp4";
    video.append(source);
    if (artifact.optionalVideo.descriptionTrack) {
      const track = document.createElement("track");
      track.kind = "descriptions";
      track.src = resolveAsset(artifact.optionalVideo.descriptionTrack);
      track.srclang = "en";
      track.label = "Visual description";
      video.append(track);
    }
    stage.replaceChildren(video);
    video.focus();
  }

  function placeRail(run) {
    const lecture = lectureByNumber(state.lecture);
    const activeSlotId = state.anchorId ? slotForAnchor(state.anchorId) : "";
    const html = railTemplate(lecture, run, activeSlotId);
    const mobile = window.matchMedia("(max-width: 920px)").matches;
    cleanupArtifact();
    document.querySelectorAll(".mobile-rail-slot").forEach((slot) => slot.replaceChildren());
    state.mobile = mobile;
    if (mobile) {
      el.rail.replaceChildren();
      const mobileSlot = document.querySelector(`[data-mobile-rail="${run.id}"]`);
      if (!mobileSlot) return;
      const wrapper = document.createElement("section");
      wrapper.className = "mobile-context";
      wrapper.setAttribute("aria-label", "Current lecture augmentations");
      wrapper.innerHTML = `<p class="rail-label">At this moment</p><div class="mobile-rail-content">${html}</div>`;
      mobileSlot.append(wrapper);
      state.activeRail = wrapper.querySelector(".mobile-rail-content");
    } else {
      el.rail.innerHTML = html;
      state.activeRail = el.rail;
    }
    document.querySelectorAll("[data-active-rail]").forEach((node) => node.removeAttribute("data-active-rail"));
    state.activeRail.dataset.activeRail = "true";
    if (activeSlotId) requestAnimationFrame(() => activeRail().querySelector(`[data-slot-id="${CSS.escape(activeSlotId)}"]`)?.scrollIntoView({ block: "nearest" }));
  }

  function setCurrentRun(runId, options = {}) {
    const found = runById(runId);
    if (!found || found.lecture.lecture !== state.lecture) return;
    const changed = state.runId !== runId;
    state.runId = runId;
    document.documentElement.dataset.currentRun = runId;
    document.querySelectorAll(".lecture-run").forEach((node) => node.classList.toggle("is-current", node.dataset.runId === runId));
    document.querySelectorAll("[data-run-target]").forEach((node) => { const active = node.dataset.runTarget === runId; node.classList.toggle("is-current", active); active ? node.setAttribute("aria-current", "location") : node.removeAttribute("aria-current"); });
    if (changed || options.force || state.anchorId) placeRail(found.run);
    if (options.updatePlayer !== false) renderPlayer(found.lecture, found.run);
  }

  function renderLecture(number, initialRunId = "") {
    state.lecture = number;
    const lecture = lectureByNumber(number);
    state.runId = "";
    el.kicker.textContent = `Lecture ${lecture.lecture} · ${lecture.instructor}`;
    el.title.textContent = lecture.title;
    el.thesis.textContent = lecture.spineThesis;
    el.links.innerHTML = `<a href="${lecture.videoUrl}" target="_blank" rel="noreferrer">Official video ↗</a><a href="${lecture.sourceUrl}" target="_blank" rel="noreferrer">${sourceLabel(lecture)} ↗</a>`;
    el.runNav.innerHTML = lecture.runs.map((run, index) => `<a href="#${run.id}" data-run-target="${run.id}"><span>${formatTime(run.startSeconds)}</span>${String(index + 1).padStart(2, "0")} · ${escapeHTML(run.title)}</a>`).join("");
    el.runs.innerHTML = lecture.runs.map((run, index) => runTemplate(lecture, run, index)).join("");
    document.querySelectorAll("[data-lecture]").forEach((node) => { const active = Number(node.dataset.lecture) === number; node.classList.toggle("is-active", active); active ? node.setAttribute("aria-current", "page") : node.removeAttribute("aria-current"); });
    document.title = `Lecture ${lecture.lecture}: ${lecture.title} · CS336 Enriched`;
    setCurrentRun(initialRunId || lecture.runs[0].id);
    scheduleRunSync();
  }

  function syncRunFromViewport() {
    if (state.anchorId || state.settlingRunId || state.pinnedRunId) return;
    const marker = Number.parseFloat(getComputedStyle(document.documentElement).getPropertyValue("--header-h")) + 34;
    const runs = [...document.querySelectorAll(".lecture-run")];
    const containing = runs.find((node) => { const rect = node.getBoundingClientRect(); return rect.top <= marker && rect.bottom > marker; });
    const nearest = containing || runs.sort((a, b) => Math.abs(a.getBoundingClientRect().top - marker) - Math.abs(b.getBoundingClientRect().top - marker))[0];
    if (nearest && nearest.dataset.runId !== state.runId) setCurrentRun(nearest.dataset.runId, { updatePlayer: false });
  }

  function scheduleRunSync() {
    if (state.scrollScheduled || state.anchorId || state.settlingRunId || state.pinnedRunId) return;
    state.scrollScheduled = true;
    requestAnimationFrame(() => { state.scrollScheduled = false; syncRunFromViewport(); });
  }

  function loadOfficial(button) {
    const lecture = lectureByNumber(Number(button.dataset.loadOfficial));
    const start = Number(button.dataset.start);
    const iframe = document.createElement("iframe");
    iframe.title = `Official Lecture ${lecture.lecture} video`;
    iframe.loading = "lazy";
    iframe.allow = "accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share";
    iframe.allowFullscreen = true;
    iframe.src = `https://www.youtube-nocookie.com/embed/${youtubeId(lecture.videoUrl)}?start=${start}&rel=0`;
    el.player.replaceChildren(iframe);
  }

  function showToast(message) {
    el.toast.textContent = message; el.toast.hidden = false; clearTimeout(showToast.timer); showToast.timer = setTimeout(() => { el.toast.hidden = true; }, 3000);
  }

  function mountGiscus(panel, term) {
    const slot = panel.querySelector(".giscus-slot");
    slot.replaceChildren();
    if (!(config.discussions.enabled && config.discussions.repoId && config.discussions.categoryId)) { slot.innerHTML = `<p class="discussion-pending">Discussion setup is not configured. Use stable anchor <code>${escapeHTML(term)}</code>.</p>`; return; }
    const script = document.createElement("script");
    script.src = "https://giscus.app/client.js";
    Object.assign(script.dataset, { repo: config.discussions.repo, repoId: config.discussions.repoId, category: config.discussions.category, categoryId: config.discussions.categoryId, mapping: "specific", term, strict: "1", reactionsEnabled: "1", emitMetadata: "0", inputPosition: "top", theme: "light", lang: "en", loading: "lazy" });
    script.crossOrigin = "anonymous"; script.async = true; slot.append(script);
  }

  function nextLayout() { return new Promise((resolve) => requestAnimationFrame(() => requestAnimationFrame(resolve))); }

  async function settleHashNavigation(target, artifactTarget, hash) {
    const token = ++state.navigationToken;
    state.settlingRunId = target.run.id;
    state.pinnedRunId = target.run.id;
    const alignTarget = () => document.getElementById(target.run.id)?.scrollIntoView({ block: "start", behavior: "auto" });
    alignTarget();
    const feedbackSelect = document.querySelector(`[data-feedback-panel="${CSS.escape(target.run.id)}"] select`);
    if (feedbackSelect?.querySelector(`option[value="${CSS.escape(hash)}"]`)) feedbackSelect.value = hash;
    if (artifactTarget) await loadArtifact(artifactTarget.artifactId, artifactTarget.stateId, false);
    await document.fonts?.ready;
    await nextLayout();
    if (token !== state.navigationToken) return;
    alignTarget();
    await new Promise((resolve) => setTimeout(resolve, 140));
    if (token !== state.navigationToken) return;
    alignTarget();
    await nextLayout();
    if (token !== state.navigationToken) return;
    state.anchorId = "";
    state.settlingRunId = "";
  }

  function releasePinnedRun() {
    if (!state.pinnedRunId) return;
    state.pinnedRunId = "";
    scheduleRunSync();
  }

  function handleHash() {
    const hash = decodeURIComponent(location.hash.slice(1));
    const lectureMatch = hash.match(/^lecture-(\d)$/);
    if (lectureMatch) { state.anchorId = ""; state.settlingRunId = ""; state.pinnedRunId = ""; renderLecture(Number(lectureMatch[1])); scrollTo({ top: 0, behavior: "auto" }); return; }
    const artifactTarget = artifactRegistry.byAnchor?.[hash];
    const anchorRunId = artifactTarget?.runId || model.anchorToRun[hash];
    const directRun = runById(hash);
    const target = directRun || (anchorRunId ? runById(anchorRunId) : null);
    if (!target) return;
    state.anchorId = anchorRunId ? hash : "";
    state.settlingRunId = target.run.id;
    if (target.lecture.lecture !== state.lecture) renderLecture(target.lecture.lecture, target.run.id);
    else setCurrentRun(target.run.id, { force: true });
    requestAnimationFrame(() => { settleHashNavigation(target, artifactTarget, hash); });
  }

  document.addEventListener("click", (event) => {
    const lectureButton = event.target.closest("[data-lecture]");
    if (lectureButton) { event.preventDefault(); state.anchorId = ""; state.pinnedRunId = ""; history.pushState(null, "", `#lecture-${lectureButton.dataset.lecture}`); renderLecture(Number(lectureButton.dataset.lecture)); scrollTo({ top: 0, behavior: "auto" }); el.navToggle.setAttribute("aria-expanded", "false"); el.nav.classList.remove("is-open"); return; }
    const load = event.target.closest("[data-load-official]"); if (load) { loadOfficial(load); return; }
    const loadArtifactButton = event.target.closest("[data-load-artifact]");
    if (loadArtifactButton) { const artifact = artifactRegistry.byId?.[loadArtifactButton.dataset.loadArtifact]; const requestedState = artifactRegistry.byAnchor?.[state.anchorId]?.stateId || artifact?.primary.defaultState || ""; loadArtifact(loadArtifactButton.dataset.loadArtifact, requestedState); return; }
    const loadArtifactVideoButton = event.target.closest("[data-load-artifact-video]");
    if (loadArtifactVideoButton) { loadArtifactVideo(loadArtifactVideoButton.dataset.loadArtifactVideo); return; }
    const openFeedback = event.target.closest("[data-open-feedback]");
    if (openFeedback) { const panel = document.querySelector(`[data-feedback-panel="${CSS.escape(openFeedback.dataset.openFeedback)}"]`); const opening = panel.hidden; document.querySelectorAll(".feedback-panel").forEach((item) => { item.hidden = true; }); panel.hidden = !opening; if (opening) panel.querySelector("select")?.focus(); return; }
    const discuss = event.target.closest("[data-mount-discussion]");
    if (discuss) { const panel = document.querySelector(`[data-feedback-panel="${CSS.escape(discuss.dataset.mountDiscussion)}"]`); const term = panel.querySelector("select").value; mountGiscus(panel, term); showToast(`Discussion anchored to ${term}`); return; }
    const reveal = event.target.closest("[data-reveal-underflow]"); if (reveal) { reveal.nextElementSibling.hidden = false; reveal.textContent = "Result revealed"; reveal.disabled = true; return; }
    const step = event.target.closest("[data-step]");
    if (step) { const root = step.closest("[data-stepper]"); const states = JSON.parse(root.querySelector("script").textContent); const label = root.querySelector(".stepper-controls span"); let index = Number(root.dataset.index || 0); index = Math.max(0, Math.min(states.length - 1, index + Number(step.dataset.step))); root.dataset.index = index; root.querySelector(".stepper-stage code").textContent = states[index]; label.textContent = `${index + 1} / ${states.length}`; root.querySelector('[data-step="-1"]').disabled = index === 0; root.querySelector('[data-step="1"]').disabled = index === states.length - 1; }
  });

  document.addEventListener("input", (event) => {
    if (event.target.matches("[data-ff-width]")) { const root = event.target.closest(".calculator"); const width = Number(event.target.value); root.querySelector("p span").textContent = width.toLocaleString(); root.querySelector("[data-ff-standard]").textContent = (width * 4).toLocaleString(); root.querySelector("[data-ff-glu]").textContent = Math.round(width * 8 / 3).toLocaleString(); }
    if (event.target.matches("[data-cache-tokens]")) event.target.closest(".calculator").querySelector("[data-cache-value]").textContent = Number(event.target.value).toLocaleString();
    if (event.target.matches("[data-window], [data-layers]")) { const root = event.target.closest(".window-demo"); const reach = Number(root.querySelector("[data-window]").value) * Number(root.querySelector("[data-layers]").value); root.querySelector("[data-window-result]").textContent = `${reach} tokens`; }
  });

  el.navToggle.addEventListener("click", () => { const open = el.navToggle.getAttribute("aria-expanded") === "true"; el.navToggle.setAttribute("aria-expanded", String(!open)); el.nav.classList.toggle("is-open", !open); });
  document.addEventListener("keydown", (event) => {
    if (["ArrowUp", "ArrowDown", "PageUp", "PageDown", "Home", "End", " "].includes(event.key)) releasePinnedRun();
    if (event.key === "Escape" && el.navToggle.getAttribute("aria-expanded") === "true") { el.navToggle.setAttribute("aria-expanded", "false"); el.nav.classList.remove("is-open"); el.navToggle.focus(); }
  });
  window.addEventListener("hashchange", handleHash);
  window.addEventListener("scroll", scheduleRunSync, { passive: true });
  window.addEventListener("wheel", releasePinnedRun, { passive: true });
  window.addEventListener("touchstart", releasePinnedRun, { passive: true });
  window.addEventListener("pointerdown", releasePinnedRun, { passive: true });
  window.addEventListener("resize", () => {
    const mobile = window.matchMedia("(max-width: 920px)").matches;
    const found = runById(state.runId);
    if (found && mobile !== state.mobile) placeRail(found.run);
    scheduleRunSync();
  });

  function initialize() {
    const hash = decodeURIComponent(location.hash.slice(1));
    const anchorRunId = artifactRegistry.byAnchor?.[hash]?.runId || model.anchorToRun[hash] || "";
    const runId = anchorRunId || (runById(hash) ? hash : "");
    const target = runId ? runById(runId) : null;
    const lectureMatch = hash.match(/^lecture-(\d)$/);
    state.anchorId = anchorRunId ? hash : "";
    state.settlingRunId = target?.run.id || "";
    state.pinnedRunId = target?.run.id || "";
    renderLecture(target?.lecture.lecture || Number(lectureMatch?.[1]) || 1, target?.run.id || "");
    el.discussionStatus.innerHTML = config.discussions.enabled ? `<strong>Live:</strong> comments are stored in GitHub Discussions under the selected legacy anchor.` : `<strong>Prepared:</strong> stable anchors work now; Giscus can be enabled in <code>config.js</code>.`;
    if (target) setTimeout(handleHash, 0);
  }

  initialize();
})();
