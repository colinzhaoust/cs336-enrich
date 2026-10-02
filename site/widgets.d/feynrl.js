// Widgets for thread feynrl (FeynRL RL post-training library, local clone at sha dfe85351, see repos/feynrl/sources.json).
// Register as "fixture:<id>" -> (root, notice) => void. Each widget has a pure model in MODELS (no DOM) that
// tools/check_widgets.mjs tests against the KPs' stored answers.
// Anchors are repo-relative (core/rl_engines.py:L150 = external/FeynRL/core/rl_engines.py line 150). Defaults are the
// gsm8k recipe, examples/llm/rl/gsm8k/qwen2.5-1.5b-instruct/train_sync.yaml (and train_async.yaml), cited as yaml:L<n>.
import { el, fmt, slider } from "../widgets_lib.js";

export const MODELS = {};
export const WIDGETS = {};

const C = { ink: "var(--ink)", muted: "var(--muted)", rule: "var(--rule)", a: "var(--accent)", b: "var(--accent2)", ok: "var(--ok)", hi: "var(--hilite)" };
const rect = (x, y, w, h, fill, extra = "") => `<rect x="${x.toFixed(1)}" y="${y.toFixed(1)}" width="${Math.max(0, w).toFixed(1)}" height="${Math.max(0, h).toFixed(1)}" style="fill:${fill}" ${extra}/>`;
const text = (x, y, s, o = {}) => `<text x="${x.toFixed(1)}" y="${y.toFixed(1)}" style="fill:${o.fill || C.ink};font:${o.weight || ""} ${o.size || 12}px var(--sans)" text-anchor="${o.anchor || "start"}">${s}</text>`;
const line = (x1, y1, x2, y2, stroke, extra = "") => `<line x1="${x1.toFixed(1)}" y1="${y1.toFixed(1)}" x2="${x2.toFixed(1)}" y2="${y2.toFixed(1)}" style="stroke:${stroke};stroke-width:1.5" ${extra}/>`;
const path = (pts, stroke, extra = "") => `<path d="${pts.map((p, i) => `${i ? "L" : "M"}${p[0].toFixed(1)},${p[1].toFixed(1)}`).join("")}" style="fill:none;stroke:${stroke};stroke-width:2" ${extra}/>`;
const svg = (w, h, body) => `<svg viewBox="0 0 ${w} ${h}" width="100%" style="max-width:${w}px;border:1px solid var(--rule);border-radius:6px;background:#fff" role="img">${body}</svg>`;
const check = (label, value, onChange) => { const i = el("input", { type: "checkbox" }); i.checked = value; i.addEventListener("change", () => onChange(i.checked)); return el("label", {}, i, ` ${label}`); };
const select = (label, opts, value, onChange) => { const s = el("select", {}, ...opts.map(([v, t]) => el("option", { value: v }, t))); s.value = value; s.addEventListener("change", () => onChange(s.value)); return el("label", {}, `${label} `, s); };
const sgn = (x, d = 3) => (x > 0 ? "+" : x < 0 ? "−" : "") + fmt(Math.abs(x), d);
const HATCH = `<defs><pattern id="frl-hatch" width="6" height="6" patternUnits="userSpaceOnUse" patternTransform="rotate(45)"><rect width="6" height="6" style="fill:#fff"/><line x1="0" y1="0" x2="0" y2="6" style="stroke:var(--accent2);stroke-width:2"/></pattern></defs>`;

// ===================================================== 1. pipeline sizes ==
// What one YAML sets and how many of each thing the code then builds.
//   rollout engines: num_engines = max(1, rollout_gpus // tp), each actor .options(num_gpus=tp)  (core/rl_engines.py:L150, L174-L183)
//   prompts per epoch: bsz = num_engines * rollout_batch_size_per_gpu (L198); num_batches = ceil(samples / bsz) (L200);
//     total_prompts = num_batches * bsz, completions = total_prompts * n_samples (L362-L370). An upper bound: empty responses
//     are skipped by add_batch_seqs (rollouts/replay_buffer.py:L74-L75).
//   training: DataLoader(batch_size=train_batch_size_per_gpu) over the buffer, drop_last False, so ceil(items / bsz) micro-batches;
//     batches_per_engine = ceil(num_batches / num_engines), padding repeats the last batch (core/rl_engines.py:L459-L482);
//     shard_and_put deals them round-robin, engine e gets batches[e::num_engines] (L493). One training engine per training GPU.
//   normalize_loss False with update_after_full_replay: pi_loss * (ga_pi / num_micro) (algs/GRPO/grpo.py:L398-L400).
//   overlap: replay_buffer_size = prompt_per_pass * n_samples * max_lag (run_rl_async.py:L236-L238), the deque maxlen
//     (rollouts/replay_buffer.py:L25-L26); prompt_queue_maxsize = engines * max(2, max_lag) (L244);
//     results_queue_maxsize = max(prompt_queue_maxsize, replay_buffer_size // (rollout_batch_size_per_gpu * n_samples)) (L248-L249);
//     max_lag < 1 is rejected by load_and_verify (configs/load.py:L812-L813).
//   NCCL weight-sync group: world_size = 1 + num_rollout_engines * tp, engine i's TP worker w has rank 1 + i*tp + w
//     (core/rl_engines.py:L645-L654, L675).
// Recipe values: rollout_gpus 2, tp 1, rollout_batch_size_per_gpu 64, rollout_samples_per_epoch 512, n_samples 4
// (yaml:L8, L71, L80-L82), training_gpus 6 (yaml:L7), train_batch_size_per_gpu 8, gradient_accumulation_steps 1 (yaml:L59-L60),
// max_lag 1 (train_async.yaml:L29).
function pipelineSizes({ rolloutGpus = 2, tp = 1, perGpu = 64, samples = 512, n = 4, trainGpus = 6, trainBsz = 8, maxLag = 1, ga = 1, qe = 0, qw = 0 }) {
  const engines = Math.max(1, Math.floor(rolloutGpus / tp)), gpusUsed = engines * tp;
  const bsz = engines * perGpu, batches = Math.ceil(samples / bsz), prompts = batches * bsz, completions = prompts * n;
  const micro = Math.ceil(completions / trainBsz), perEngine = Math.ceil(micro / trainGpus), padded = perEngine * trainGpus;
  const bufCap = prompts * n * maxLag, pq = engines * Math.max(2, maxLag), itemsPerShard = perGpu * n;
  const rq = Math.max(pq, Math.floor(bufCap / Math.max(1, itemsPerShard)));
  return { engines, gpusPerEngine: tp, gpusUsed, idleGpus: rolloutGpus - gpusUsed, bsz, batches, prompts, roundUp: prompts - samples, completions,
    micro, perEngine, padded, dup: padded - micro, noNormScale: ga / perEngine, bufCap, bufCapNoCeil: samples * n * maxLag,
    promptQueue: pq, resultsQueue: rq, shardsPerPass: batches * engines, lagError: maxLag < 1 ? 1 : 0,
    world: 1 + engines * tp, rankQ: 1 + qe * tp + qw };
}
const RECIPE = { rolloutGpus: 2, tp: 1, perGpu: 64, samples: 512, n: 4, trainGpus: 6, trainBsz: 8, maxLag: 1, ga: 1 };
MODELS["fixture:feynrl--pipeline-sizes"] = {
  fn: pipelineSizes,
  cases: [
    { args: RECIPE, pick: "completions", expect: 2048, tol: 0, from: "feynrl:config-recipe-gsm8k:predict" },
    { args: { ...RECIPE, samples: 500 }, pick: "completions", expect: 2048, tol: 0, from: "feynrl:config-recipe-gsm8k:transfer" },
    { args: { ...RECIPE, samples: 500 }, pick: "roundUp", expect: 12, tol: 0 },                        // 500 prompts asked, 512 built
    { args: { ...RECIPE, samples: 513 }, pick: "completions", expect: 2560, tol: 0 },                  // edge: one prompt over adds a whole batch
    { args: { ...RECIPE, rolloutGpus: 8, tp: 4 }, pick: "engines", expect: 2, tol: 0 },                 // rollout-engine-generate predict: 2 actors
    { args: { ...RECIPE, rolloutGpus: 8, tp: 4 }, pick: "gpusPerEngine", expect: 4, tol: 0 },           // ... each num_gpus=4
    { args: { ...RECIPE, rolloutGpus: 3, tp: 2 }, pick: "idleGpus", expect: 1, tol: 0 },                // edge: 3 // 2 = 1 engine, one GPU unused
    { args: RECIPE, pick: "perEngine", expect: 43, tol: 0 },                                           // replay-batching predict: 43 per engine
    { args: RECIPE, pick: "dup", expect: 2, tol: 0 },                                                  // ... 256 real padded to 258
    { args: RECIPE, pick: "noNormScale", expect: 0.02326, tol: 0.02, from: "feynrl:replay-batching:transfer" },
    { args: { ...RECIPE, trainGpus: 8 }, pick: "dup", expect: 0, tol: 0 },                             // edge: 256 / 8 divides, no padding
    { args: { ...RECIPE, maxLag: 2 }, pick: "bufCap", expect: 4096, tol: 0 },                          // staleness transfer: capacity 4096
    { args: { rolloutGpus: 4, tp: 1, perGpu: 32, samples: 300, n: 8, trainGpus: 6, trainBsz: 8, maxLag: 3 }, pick: "bufCap", expect: 9216, tol: 0.02, from: "feynrl:staleness-budget-max-lag:check" },
    { args: { rolloutGpus: 4, tp: 1, perGpu: 32, samples: 300, n: 8, trainGpus: 6, trainBsz: 8, maxLag: 3 }, pick: "bufCapNoCeil", expect: 7200, tol: 0 }, // the check's distractor
    { args: RECIPE, pick: "resultsQueue", expect: 8, tol: 0 },                                         // 2048 // 256 = 8 shards, one pass
    { args: { ...RECIPE, maxLag: 0 }, pick: "lagError", expect: 1, tol: 0 },                           // staleness predict: rejected
    { args: { ...RECIPE, rolloutGpus: 4, tp: 2 }, pick: "world", expect: 5, tol: 0 },                  // weight-sync-nccl-async predict: world_size 5
    { args: { ...RECIPE, rolloutGpus: 4, tp: 2, qe: 1, qw: 1 }, pick: "rankQ", expect: 4, tol: 0 },    // ... engine 1, TP worker 1 -> rank 4
  ],
};
WIDGETS["fixture:feynrl--pipeline-sizes"] = (root) => {
  const s = { ...RECIPE, samples: 384, overlap: false };          // 384 prompts: not the recipe's 512, so the predict needs a move
  const pic = el("div"), read = el("div", { class: "readout" });
  const draw = () => {
    const m = pipelineSizes(s), W = 640, L = 150, R = 620;
    let b = HATCH, y = 22;
    // A. prompts per epoch: whole dataloader batches, the rounded-up tail hatched
    b += text(10, y + 14, "prompts / epoch", { fill: C.muted, size: 11 });
    const span = Math.max(m.prompts, s.samples), px = (R - L) / span;
    for (let k = 0; k < m.batches; k++) {
      const x0 = L + k * m.bsz * px, inReq = Math.max(0, Math.min(m.bsz, s.samples - k * m.bsz));
      b += rect(x0, y, inReq * px, 20, C.a, 'fill-opacity="0.75"') + rect(x0 + inReq * px, y, (m.bsz - inReq) * px, 20, "url(#frl-hatch)");
      b += line(x0, y - 3, x0, y + 23, C.ink);
    }
    b += line(L + m.prompts * px, y - 3, L + m.prompts * px, y + 23, C.ink);
    b += text(L, y + 36, `${m.batches} batch${m.batches > 1 ? "es" : ""} × ${m.bsz} prompts${m.roundUp ? ` · hatched: +${m.roundUp} prompts from rounding up` : ""}`, { fill: C.muted, size: 11 });
    // B. rollout engines (and NCCL ranks)
    y += 56;
    b += text(10, y + 14, "rollout engines", { fill: C.muted, size: 11 });
    const ew = Math.min(110, (R - L) / Math.max(1, m.engines) - 6);
    for (let e = 0; e < Math.min(m.engines, 16); e++) {
      const x0 = L + e * (ew + 6);
      b += rect(x0, y, ew, 22, C.hi, 'fill-opacity="0.45" stroke="var(--rule)"');
      b += text(x0 + ew / 2, y + 15, ew > 70 ? (s.overlap ? `engine ${e} · ranks ${1 + e * s.tp}${s.tp > 1 ? `–${e * s.tp + s.tp}` : ""}` : `engine ${e} · ${s.tp} GPU${s.tp > 1 ? "s" : ""}`) : `${e}`, { size: 10, anchor: "middle" });
    }
    if (m.engines > 16) b += text(R, y + 36, `… ${m.engines} engines`, { fill: C.muted, size: 11, anchor: "end" });
    // C. training micro-batches, one column per training engine, round-robin; padded duplicates orange
    y += 44;
    b += text(10, y + 14, "micro-batches", { fill: C.muted, size: 11 }) + text(10, y + 28, "per training engine", { fill: C.muted, size: 11 });
    const G = Math.min(s.trainGpus, 16), cw = Math.min(30, (R - L) / G - 4), H = 120, ch = H / Math.max(1, m.perEngine);
    for (let g = 0; g < G; g++) {
      const x0 = L + g * (cw + 4);
      for (let k = 0; k < m.perEngine; k++) {
        const idx = k * s.trainGpus + g, dupe = idx >= m.micro;
        b += rect(x0, y + H - (k + 1) * ch, cw, Math.max(0.6, ch - (ch > 3 ? 1 : 0)), dupe ? C.b : C.a, `fill-opacity="${dupe ? 0.9 : 0.55}"`);
      }
      b += text(x0 + cw / 2, y + H + 13, `${g}`, { fill: C.muted, size: 10, anchor: "middle" });
    }
    b += text(L + G * (cw + 4) + 8, y + 12, `${m.perEngine} each`, { size: 12, weight: "bold" });
    if (m.dup) b += text(L + G * (cw + 4) + 8, y + 28, `orange: ${m.dup} repeat${m.dup > 1 ? "s" : ""} of the last batch`, { fill: C.muted, size: 11 });
    // D. overlap: replay buffer capacity in passes
    y += H + 26;
    if (s.overlap) {
      b += text(10, y + 14, "async buffer maxlen", { fill: C.muted, size: 11 });
      const pw = Math.min(110, (R - L) / Math.max(1, s.maxLag) - 6);
      for (let k = 0; k < s.maxLag; k++) b += rect(L + k * (pw + 6), y, pw, 20, C.ok, 'fill-opacity="0.4" stroke="var(--rule)"') + text(L + k * (pw + 6) + pw / 2, y + 14, `pass ${k + 1}: ${m.completions}`, { size: 10, anchor: "middle" });
      if (s.maxLag < 1) b += text(L, y + 14, "max_lag < 1: load_and_verify raises, no buffer is built", { fill: C.b, size: 12, weight: "bold" });
      y += 34;
    }
    pic.innerHTML = svg(W, y, b);
    read.innerHTML = `rollout: max(1, ${s.rolloutGpus} // ${s.tp}) = <b>${m.engines}</b> engine${m.engines > 1 ? "s" : ""}, each a Ray actor with num_gpus = ${s.tp}${m.idleGpus > 0 ? ` · <b>${m.idleGpus} rollout GPU${m.idleGpus > 1 ? "s" : ""} unused</b>` : m.idleGpus < 0 ? ` · <b>asks for ${m.gpusUsed} GPUs but only ${s.rolloutGpus} exist</b> (max(1, …) still builds one engine)` : ""}<br>
      prompts: ceil(${s.samples} / ${m.bsz}) = ${m.batches} batches × ${m.bsz} = <b>${m.prompts}</b>${m.roundUp ? ` (${m.roundUp} more than asked: rollout_samples_per_epoch counts prompts and is rounded up to whole batches)` : " (no rounding)"}<br>
      <span class="big">${m.prompts} × ${s.n} samples = ${m.completions} completions</span> per epoch, at most (empty responses are skipped)<br>
      training: ceil(${m.completions} / ${s.trainBsz}) = ${m.micro} micro-batches → ceil(/${s.trainGpus}) = <b>${m.perEngine} per engine</b>${m.dup ? `, padded to ${m.padded} with <b>${m.dup}</b> duplicate${m.dup > 1 ? "s" : ""} (unequal shards would deadlock ZeRO-3)` : ", no padding"} · micro-batches mix every prompt's samples<br>
      normalize_loss False + full replay: each micro-batch mean × ga/num_micro = ${s.ga}/${m.perEngine} = ${fmt(m.noNormScale, 5)}<br>
      ${s.overlap ? (m.lagError ? `<b>overlap.max_lag = ${s.maxLag}: ValueError "must be >= 1" before any engine starts</b>` : `overlap: buffer maxlen = ${m.prompts} × ${s.n} × max_lag ${s.maxLag} = <b>${m.bufCap}</b>${m.bufCap !== m.bufCapNoCeil ? ` (not ${s.samples} × ${s.n} × ${s.maxLag} = ${m.bufCapNoCeil}: the prompts are rounded up first)` : ""} · prompt queue ${m.promptQueue} shards · results queue ${m.resultsQueue} shards of ${s.perGpu * s.n}<br>
      NCCL weight-sync group: world_size = 1 + ${m.engines} × ${s.tp} = <b>${m.world}</b> (rank 0 = training rank 0; engine i, TP worker w → rank 1 + i·${s.tp} + w)`) : "tick overlap to see the async buffer, queues and NCCL ranks"}<br>
      <span class="muted small">provenance: core/rl_engines.py:L150 (engines), L198-L200 and L362-L370 (prompts), L459-L493 (micro-batches, padding, round-robin); algs/GRPO/grpo.py:L398-L400; run_rl_async.py:L236-L249 (buffer, queues); configs/load.py:L812-L813; core/rl_engines.py:L654, L675 (NCCL ranks). Defaults: train_sync.yaml:L7-L8, L59-L60, L71-L82, except rollout_samples_per_epoch (384, recipe 512).</span>`;
  };
  const set = (k, v) => { s[k] = v; draw(); };
  root.append(el("div", { class: "widget" }, el("div", { class: "controls" },
    slider("run.rollout_gpus", 1, 16, s.rolloutGpus, 1, v => set("rolloutGpus", v)),
    slider("rollout.tensor_parallel_size", 1, 8, s.tp, 1, v => set("tp", v)),
    slider("rollout.rollout_batch_size_per_gpu", 8, 128, s.perGpu, 8, v => set("perGpu", v)),
    slider("rollout.rollout_samples_per_epoch (prompts)", 16, 2048, s.samples, 1, v => set("samples", v)),
    slider("rollout.n_samples", 1, 16, s.n, 1, v => set("n", v)),
    slider("run.training_gpus", 1, 16, s.trainGpus, 1, v => set("trainGpus", v)),
    slider("train.train_batch_size_per_gpu", 1, 32, s.trainBsz, 1, v => set("trainBsz", v)),
    slider("train.gradient_accumulation_steps", 1, 8, s.ga, 1, v => set("ga", v)),
    check("overlap.enabled (async)", s.overlap, v => set("overlap", v)),
    slider("overlap.max_lag", 0, 4, s.maxLag, 1, v => set("maxLag", v)),
    read), pic)); draw();
};

// ======================================================= 2. epoch ledger ==
// What one sync-mode run calls, epoch by epoch (run_rl_sync.py):
//   for epoch in range(start_epoch, number_of_epochs) with start_epoch 0 (L260, L339); is_last_epoch = epoch == N - 1 (L341).
//   run_epoch_sync: replay_buffer.reset (L35), collect_rollouts once (L40-L47), prepare_training_batches + shard_and_put once
//     (L54-L68), run_training_step steps_per_epoch times on the same shard_refs (L72-L76), policy_version += 1 once (L90).
//   main: sync_weights_direct only `if not sync_success and not is_last_epoch and weight_sync_method == "direct"` (L417-L426);
//     an exception sets sync_success False (L427-L429); need_disk_for_rollout = (disk and not last) or (no sync succeeded and
//     method in nccl/direct/disk and not last) (L443-L445); save_checkpoint if need_disk or (epoch+1) % interval == 0 or last
//     (L436-L447); refresh_rollout_engine if need_disk and not last (L463-L470).
//   overlap (run_rl_async.py): policy_version += 1 then perform_inline_sync unless the last epoch (L841-L847); saves when
//     (epoch+1) % interval == 0 or last (L1333-L1335).
//   validation: sync mode + nccl raises (configs/load.py:L722-L724); overlap + direct/disk raises (L706-L709).
//   direct-sync verification on one engine (rollouts/vllm_engine.py:L263-L284): drop None/0 → none left: raise (all failed);
//     fewer than results: raise (partial); unequal counts: raise; else loaded_version = version.
//   optimizer steps inside one train_step: is_boundary = is_last if update_after_full_replay else (step+1) % ga == 0 or is_last
//     (algs/GRPO/grpo.py:L299-L307). A micro-batch is on-policy while no optimizer step has happened since the weights that
//     generated it were loaded, i.e. in pass 1 up to and including the first boundary (supp-on-off-policy).
// Recipe: total_number_of_epochs 100, train_steps_per_epoch 1, update_after_full_replay True, gradient_accumulation_steps 1,
// weight_sync_method "direct", checkpoint_save_interval 15 (train_sync.yaml:L13-L14, L47, L55-L56, L60); 43 micro-batches
// per engine per pass is what the recipe builds (fixture:feynrl--pipeline-sizes).
function verifyTp(results) {
  if (!results || !results.length) return 0;
  const valid = results.filter(r => r !== null && r !== 0);
  if (!valid.length) return 1;                       // all workers returned None/0
  if (valid.length < results.length) return 2;        // partial failure
  if (new Set(valid).size > 1) return 3;              // different param counts
  return 0;
}
const VERDICT = ["accepted: loaded_version advances, prefix cache reset", "RuntimeError: all TP workers loaded nothing", "RuntimeError: partial failure (some workers returned 0/None)", "RuntimeError: TP workers loaded different param counts"];
function epochLedger({ epochs = 100, steps = 1, micro = 43, fullReplay = true, ga = 1, method = "direct", overlap = false, failEpoch = 0, results = null, saveInterval = 15 }) {
  const errNccl = !overlap && method === "nccl" ? 1 : 0, errOverlap = overlap && method !== "nccl" ? 1 : 0;
  const verdict = verifyTp(results);
  const o = { errNccl, errOverlap, error: errNccl || errOverlap, verdict, rollouts: 0, trainSteps: 0, optSteps: 0, bumps: 0,
    directCalls: 0, directFails: 0, saves: 0, refreshes: 0, inlineSyncs: 0, epochsLog: [] };
  const optPerPass = fullReplay ? 1 : Math.ceil(micro / ga);
  o.passesPerEpoch = steps; o.bumpsPerEpoch = 1; o.optPerPass = optPerPass; o.microPerEpoch = steps * micro;
  o.onPolicy = Math.min(micro, fullReplay ? micro : ga);   // pass 1, up to and including the first optimizer step
  if (o.error) return o;
  for (let e = 0; e < epochs; e++) {
    const last = e === epochs - 1, row = { e, sync: "none", save: false, refresh: false };
    o.rollouts++; o.trainSteps += steps; o.optSteps += steps * optPerPass; o.bumps++;
    const periodic = saveInterval > 0 && (e + 1) % saveInterval === 0;
    if (overlap) {
      if (!last) { o.inlineSyncs++; row.sync = "nccl"; }
      row.save = periodic || last;
    } else {
      let ok = false;
      if (!last && method === "direct") {
        o.directCalls++;
        ok = !(failEpoch === e + 1 && verdict !== 0);
        if (!ok) o.directFails++;
        row.sync = ok ? "direct" : "direct failed";
      }
      const needDisk = !last && (method === "disk" || !ok);
      row.save = needDisk || periodic || last;
      if (needDisk) { o.refreshes++; row.refresh = true; if (method === "disk") row.sync = "disk"; }
    }
    if (row.save) o.saves++;
    o.epochsLog.push(row);
  }
  return o;
}
MODELS["fixture:feynrl--epoch-ledger"] = {
  fn: epochLedger,
  cases: [
    { args: { epochs: 6, steps: 4, method: "direct" }, pick: "directCalls", expect: 5, tol: 0.05, from: "feynrl:sync-epoch-loop:check" },
    { args: { epochs: 6, steps: 4, method: "direct" }, pick: "trainSteps", expect: 24, tol: 0 },          // the check's distractor (one per step)
    { args: { epochs: 6, steps: 3 }, pick: "bumpsPerEpoch", expect: 1, tol: 0 },                         // sync-epoch-loop predict: one bump ...
    { args: { epochs: 6, steps: 3 }, pick: "passesPerEpoch", expect: 3, tol: 0 },                        // ... three passes
    { args: { epochs: 6, steps: 3, method: "nccl" }, pick: "errNccl", expect: 1, tol: 0 },               // sync-epoch-loop transfer: ValueError
    { args: { epochs: 6, method: "direct", overlap: true }, pick: "errOverlap", expect: 1, tol: 0 },     // overlap needs nccl
    { args: { epochs: 6, method: "nccl", overlap: true }, pick: "inlineSyncs", expect: 5, tol: 0 },      // overlap: also N - 1
    { args: { epochs: 1, steps: 2, micro: 4, fullReplay: false, ga: 2 }, pick: "onPolicy", expect: 2, tol: 0.05, from: "feynrl:supp-on-off-policy:check" },
    { args: { epochs: 1, steps: 2, micro: 4, fullReplay: true }, pick: "onPolicy", expect: 4, tol: 0 },  // one step per pass: all of pass 1
    { args: { epochs: 1, steps: 2, micro: 4, fullReplay: false, ga: 2 }, pick: "optSteps", expect: 4, tol: 0 },
    { args: {}, pick: "saves", expect: 7, tol: 0 },                                                      // recipe: epochs 15, 30, ..., 90 and 100
    { args: { epochs: 10, failEpoch: 8, results: [150, 0] }, pick: "refreshes", expect: 1, tol: 0 },     // weight-sync-direct predict: epoch 7 (0-based) falls back
    { args: { epochs: 10, failEpoch: 8, results: [150, 0] }, pick: "verdict", expect: 2, tol: 0 },       // weight-sync-direct transfer: partial failure
    { args: { epochs: 10, failEpoch: 5, results: [150, 150, 148, 150] }, pick: "verdict", expect: 3, tol: 0 },   // weight-sync-direct check: raises ...
    { args: { epochs: 10, failEpoch: 5, results: [150, 150, 148, 150] }, pick: "refreshes", expect: 1, tol: 0 }, // ... and main falls back to disk
    { args: { epochs: 10, failEpoch: 5, results: [150, 150, 150, 150] }, pick: "refreshes", expect: 0, tol: 0 }, // equal counts: accepted
    { args: { epochs: 10, failEpoch: 10, results: [0, 0] }, pick: "directCalls", expect: 9, tol: 0 },    // edge: no sync is attempted after the last epoch
    { args: { epochs: 10, method: "disk" }, pick: "refreshes", expect: 9, tol: 0 },                      // disk: save + refresh every epoch but the last
  ],
};
WIDGETS["fixture:feynrl--epoch-ledger"] = (root) => {
  const s = { epochs: 4, steps: 1, micro: 6, fullReplay: true, ga: 1, method: "direct", overlap: false, failEpoch: 0, resultsText: "150, 150", saveInterval: 15 };
  const pic = el("div"), read = el("div", { class: "readout" });
  const parse = t => t.split(/[,\s]+/).filter(Boolean).map(x => (/^none$/i.test(x) ? null : Number(x))).filter(x => x === null || Number.isFinite(x));
  const draw = () => {
    const results = parse(s.resultsText), m = epochLedger({ ...s, results });
    const W = 640, L = 110, R = 620;
    let b = "", y = 18;
    if (m.error) {
      b += text(W / 2, 40, m.errNccl ? "load_and_verify: ValueError — 'nccl' requires overlap.enabled=True" : "load_and_verify: ValueError — overlap.enabled=True requires weight_sync_method='nccl'", { fill: C.b, size: 13, weight: "bold", anchor: "middle" });
      b += text(W / 2, 62, "no engine is started", { fill: C.muted, size: 12, anchor: "middle" });
      pic.innerHTML = svg(W, 80, b);
    } else {
      // (1) one epoch on one training engine: passes x micro-batches, optimizer steps as ticks
      b += text(10, y + 4, s.overlap ? "epoch 1 (sync mode only)" : "epoch 1, one engine", { fill: C.muted, size: 11 });
      const cw = Math.min(16, (R - L) / s.micro), ch = 14;
      if (!s.overlap) {
        let seen = 0, stepped = false;
        for (let p = 0; p < s.steps; p++) {
          const yy = y + 12 + p * (ch + 8);
          b += text(L - 6, yy + 11, `pass ${p + 1}`, { fill: C.muted, size: 10, anchor: "end" });
          for (let k = 0; k < s.micro; k++) {
            const on = !stepped;
            b += rect(L + k * cw, yy, cw - (cw > 4 ? 1 : 0), ch, on ? C.ok : C.b, `fill-opacity="${on ? 0.75 : 0.55}"`);
            seen++;
            const boundary = s.fullReplay ? k === s.micro - 1 : ((k + 1) % s.ga === 0 || k === s.micro - 1);
            if (boundary) { stepped = true; b += rect(L + (k + 1) * cw - 1.5, yy - 3, 3, ch + 6, C.ink); }
          }
        }
        y += 12 + s.steps * (ch + 8) + 4;
        b += text(L, y + 8, "green: weights still the ones that generated the samples · orange: after ≥ 1 optimizer step · bar: optimizer step", { fill: C.muted, size: 10 });
      } else { b += text(L, y + 4, "overlap: samples come from earlier versions; see the overlap-rounds animation", { fill: C.muted, size: 11 }); }
      // (2) the whole run: rollout | train ×steps | sync, one block per epoch
      y += 26;
      b += text(10, y + 14, "whole run", { fill: C.muted, size: 11 });
      const ew = (R - L) / s.epochs;
      m.epochsLog.forEach((row, e) => {
        const x0 = L + e * ew, w = ew - (ew > 6 ? 2 : 0);
        b += rect(x0, y, w * 0.35, 20, C.hi, 'fill-opacity="0.8"') + rect(x0 + w * 0.35, y, w * 0.45, 20, C.a, 'fill-opacity="0.7"');
        const col = row.sync === "direct" || row.sync === "nccl" ? C.ok : row.sync === "none" ? "#fff" : C.b;
        b += rect(x0 + w * 0.8, y, w * 0.2, 20, col, row.sync === "none" ? 'stroke="var(--rule)"' : "");
        if (row.save) b += `<circle cx="${(x0 + w * 0.9).toFixed(1)}" cy="${(y - 5).toFixed(1)}" r="2.5" style="fill:var(--ink)"/>`;
        if (ew >= 40) b += text(x0 + w / 2, y + 34, `epoch ${e + 1}`, { fill: C.muted, size: 10, anchor: "middle" });
      });
      b += text(L, y + 50, `yellow rollout · blue ${s.steps} training pass${s.steps > 1 ? "es" : ""} · green weight sync · orange disk save + refresh`, { fill: C.muted, size: 10 });
      b += text(L, y + 64, "white: no sync (after the last epoch) · dot: checkpoint saved", { fill: C.muted, size: 10 });
      pic.innerHTML = svg(W, y + 74, b);
    }
    const fe = s.failEpoch && !s.overlap && s.method === "direct" ? (s.failEpoch > s.epochs ? `epoch ${s.failEpoch} is beyond this ${s.epochs}-epoch run` : s.failEpoch === s.epochs ? `epoch ${s.failEpoch} is the last: no sync is attempted, the results are never read` : `epoch ${s.failEpoch}: [${results.map(r => r ?? "None").join(", ")}] → <b>${VERDICT[m.verdict]}</b>${m.verdict ? " → main() catches it, saves to disk and refreshes the engines" : ""}`) : "";
    read.innerHTML = m.error ? `<b>${m.errNccl ? "ValueError: weight_sync_method 'nccl' requires overlap.enabled=True (sync rollout engine does not support nccl)" : "ValueError: overlap.enabled=True requires weight_sync_method='nccl'"}</b>` :
      `per epoch: 1 collect_rollouts, ${s.steps} run_training_step pass${s.steps > 1 ? "es" : ""} over the <b>same</b> micro-batches, ${m.optPerPass} optimizer step${m.optPerPass > 1 ? "s" : ""} per pass, policy_version += 1 once<br>
      <span class="big">${s.overlap ? `perform_inline_sync × ${m.inlineSyncs}` : s.method === "direct" ? `sync_weights_direct × ${m.directCalls}` : `disk refresh × ${m.refreshes}`}</span> over ${s.epochs} epoch${s.epochs > 1 ? "s" : ""} (none after the last epoch) · run_training_step × ${m.trainSteps} · optimizer steps ${m.optSteps} · final policy_version ${m.bumps}<br>
      ${s.overlap ? "" : `on-policy micro-batches in an epoch: <b>${m.onPolicy}</b> of ${m.microPerEpoch} (before the first optimizer step; policy_version still reads the same integer until the epoch ends)<br>`}
      checkpoints saved: ${m.saves}${!s.overlap ? ` · disk refreshes: ${m.refreshes}` : ""}${fe ? `<br>${fe}` : ""}<br>
      <span class="muted small">provenance: run_rl_sync.py:L27-L99 (epoch), L339-L341, L411-L471 (sync, fallback, save, refresh); run_rl_async.py:L841-L847, L1333-L1335; configs/load.py:L706-L709, L722-L724; rollouts/vllm_engine.py:L263-L284 (verification); algs/GRPO/grpo.py:L299-L307 (optimizer boundaries). Recipe: train_sync.yaml:L13-L14, L47, L55-L60.</span>`;
  };
  const set = (k, v) => { s[k] = v; draw(); };
  const res = el("input", { type: "text", value: s.resultsText, size: 18 });
  res.addEventListener("input", () => set("resultsText", res.value));
  root.append(el("div", { class: "widget" }, el("div", { class: "controls" },
    slider("train.total_number_of_epochs", 1, 100, s.epochs, 1, v => set("epochs", v)),
    slider("train.train_steps_per_epoch", 1, 6, s.steps, 1, v => set("steps", v)),
    slider("micro-batches per engine per pass", 1, 48, s.micro, 1, v => set("micro", v)),
    check("train.update_after_full_replay", s.fullReplay, v => set("fullReplay", v)),
    slider("gradient_accumulation_steps (used when the box is off)", 1, 8, s.ga, 1, v => set("ga", v)),
    select("run.weight_sync_method", [["direct", "direct"], ["disk", "disk"], ["nccl", "nccl"]], s.method, v => set("method", v)),
    check("overlap.enabled", s.overlap, v => set("overlap", v)),
    slider("direct sync gets these TP results at epoch (0 = never)", 0, 100, s.failEpoch, 1, v => set("failEpoch", v)),
    el("label", {}, "collective_rpc results (one per TP worker) ", res),
    slider("run.checkpoint_save_interval", 0, 20, s.saveInterval, 1, v => set("saveInterval", v)),
    read), pic)); draw();
};

// ====================================================== 3. mixed sampler ==
// MixedDatasetSampler (data_feeds/mixed_sampler.py). The rollout dataloader passes local_batch_size = num_rollout_engines *
// rollout_batch_size_per_gpu (core/rl_engines.py:L198, L222) and dynamic_ratio_every_step = train.dynamic_ratio_every_step
// (L226; recipe False, yaml:L57).
//   probs = ratios / ratios.sum() in float32 (L50-L51).
//   redo_ratio_every_step = num_datasets > local_batch_size or dynamic_ratio_every_step (L70).
//   fixed: _fixed_sample_count once (L72): raw = probs * B; base = floor(raw); the B - sum(base) leftover slots go to the
//     largest fractional parts, order = argsort(frac)[::-1] (L80-L90). numpy sorts these short arrays stably, so after the
//     reversal a tie goes to the later dataset. A dataset left at 0 is never sampled (warning, L93-L99).
//   dynamic: rng.multinomial(B, probs) every batch (L110). The draws shown here use a small seeded JS generator, not numpy's
//     stream: the counts are illustrative, their distribution is the same.
//   indices: rng.integers(0, len) per dataset, with replacement (L127-L128).
const f32 = Math.fround;
function mixedCounts({ ratios, batch, dynamic = false, nBatches = 1, seed = 42 }) {
  const n = ratios.length;
  let sum = f32(0); for (const r of ratios) sum = f32(sum + f32(r));
  const probs = ratios.map(r => f32(f32(r) / sum));
  const redo = n > batch || dynamic;
  const o = { n, probs, redo: redo ? 1 : 0, raw: probs.map(p => f32(p * batch)) };
  if (!redo) {
    const base = o.raw.map(Math.floor), frac = o.raw.map((r, i) => f32(r - base[i]));
    const left = batch - base.reduce((a, x) => a + x, 0);
    const order = frac.map((f, i) => i).sort((i, j) => frac[i] - frac[j] || i - j).reverse();
    const counts = base.slice(); const bonus = new Array(n).fill(0);
    for (let k = 0; k < left; k++) { counts[order[k]] += 1; bonus[order[k]] = 1; }
    Object.assign(o, { counts, bonus, base, frac, left, zero: counts.filter(c => c === 0).length });
    o.batches = Array.from({ length: nBatches }, () => counts.slice());
  } else {
    let st = seed >>> 0;
    const rnd = () => { st = (st + 0x6D2B79F5) >>> 0; let t = st; t = Math.imul(t ^ (t >>> 15), t | 1); t ^= t + Math.imul(t ^ (t >>> 7), t | 61); return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
    o.batches = Array.from({ length: nBatches }, () => {
      const c = new Array(n).fill(0);
      for (let k = 0; k < batch; k++) { let u = rnd(), i = 0; while (i < n - 1 && u >= probs[i]) { u -= probs[i]; i++; } c[i]++; }
      return c;
    });
    o.counts = o.batches[0];
    o.zero = 0;
  }
  o.totals = probs.map((p, i) => o.batches.reduce((a, c) => a + c[i], 0));
  o.expected = probs.map(p => p * batch * nBatches);
  probs.forEach((p, i) => { o[`c${i}`] = o.counts[i]; o[`t${i}`] = o.totals[i]; });
  return o;
}
MODELS["fixture:feynrl--mixed-sampler"] = {
  fn: mixedCounts,
  cases: [
    { args: { ratios: [0.5, 0.3, 0.2], batch: 10 }, pick: "c0", expect: 5, tol: 0 },     // predict: a=5, b=3, c=2
    { args: { ratios: [0.5, 0.3, 0.2], batch: 10 }, pick: "c1", expect: 3, tol: 0 },
    { args: { ratios: [0.5, 0.3, 0.2], batch: 10 }, pick: "c2", expect: 2, tol: 0 },
    { args: { ratios: [0.5, 0.3, 0.2], batch: 7 }, pick: "c0", expect: 4, tol: 0 },      // transfer: a=4 (3.5 rounds up: largest fraction) ...
    { args: { ratios: [0.5, 0.3, 0.2], batch: 7 }, pick: "c1", expect: 2, tol: 0 },      // ... b=2
    { args: { ratios: [0.5, 0.3, 0.2], batch: 7 }, pick: "c2", expect: 1, tol: 0 },      // ... c=1
    { args: { ratios: [0.6, 0.3, 0.1], batch: 7, nBatches: 10 }, pick: "t0", expect: 40, tol: 0.01, from: "feynrl:mixed-dataset-sampling:check" },
    { args: { ratios: [0.6, 0.3, 0.1], batch: 7, nBatches: 10 }, pick: "t2", expect: 10, tol: 0 },   // the leftover slot goes to c every batch
    { args: { ratios: [0.6, 0.3, 0.1], batch: 7, nBatches: 10 }, pick: "c0", expect: 4, tol: 0 },    // not 4.2 x 10 = 42 (expected value)
    { args: { ratios: [0.9, 0.06, 0.04], batch: 7 }, pick: "zero", expect: 1, tol: 0 },               // edge: c rounds to 0, never sampled
    { args: { ratios: [0.5, 0.5], batch: 3 }, pick: "c1", expect: 2, tol: 0 },                         // edge: a tie goes to the later dataset
    { args: { ratios: [0.25, 0.25, 0.25, 0.25], batch: 3 }, pick: "redo", expect: 1, tol: 0 },         // edge: 4 datasets > batch 3 forces multinomial
    { args: { ratios: [1.0], batch: 128 }, pick: "c0", expect: 128, tol: 0 },                          // the gsm8k recipe: one dataset, 2 x 64
  ],
};
const DCOL = ["var(--accent)", "var(--accent2)", "var(--ok)", "var(--hilite)", "var(--muted)", "var(--ink)"];
WIDGETS["fixture:feynrl--mixed-sampler"] = (root) => {
  const s = { text: "a: 0.5, b: 0.3, c: 0.2", batch: 12, dynamic: false, nBatches: 4 };
  const pic = el("div"), read = el("div", { class: "readout" });
  const parse = t => t.split(",").map(x => x.split(":").map(y => y.trim())).filter(p => p.length === 2 && p[0] && Number(p[1]) > 0).slice(0, 6).map(([nm, v]) => ({ nm, v: Number(v) }));
  const draw = () => {
    const ds = parse(s.text);
    if (!ds.length) { read.innerHTML = "type ratios as name: value, name: value (all positive)"; pic.innerHTML = ""; return; }
    const m = mixedCounts({ ratios: ds.map(d => d.v), batch: s.batch, dynamic: s.dynamic, nBatches: s.nBatches });
    const W = 640, L = 110, R = 620, px = (R - L) / s.batch;
    let b = "", y = 16;
    // 1. ideal shares probs x B
    b += text(10, y + 13, "probs × B", { fill: C.muted, size: 11 });
    let x = L;
    m.raw.forEach((r, i) => { b += rect(x, y, r * px, 18, DCOL[i], 'fill-opacity="0.3"') + (r * px > 34 ? text(x + r * px / 2, y + 13, fmt(r, 2), { size: 10, anchor: "middle" }) : ""); x += r * px; });
    // 2. integer slots of the first batch
    y += 30;
    b += text(10, y + 13, m.redo ? "batch 1 (drawn)" : "every batch", { fill: C.muted, size: 11 });
    let slot = 0;
    m.counts.forEach((c, i) => {
      for (let k = 0; k < c; k++) {
        const bonus = !m.redo && m.bonus[i] && k === c - 1;
        b += rect(L + slot * px, y, px - (px > 4 ? 1 : 0), 18, DCOL[i], `fill-opacity="0.85"${bonus ? ' stroke="var(--ink)" stroke-width="2.5"' : ""}`);
        slot++;
      }
    });
    if (!m.redo && m.left) b += text(L, y + 32, `outlined: the ${m.left} leftover slot${m.left > 1 ? "s" : ""} after flooring, handed to the largest fractional part${m.left > 1 ? "s" : ""}`, { fill: C.muted, size: 10 });
    // 3. N batches
    y += 46;
    b += text(10, y + 13, `${s.nBatches} batches`, { fill: C.muted, size: 11 });
    const cw = Math.min(26, (R - L) / s.nBatches - 4), H = 90, sh = H / s.batch;
    m.batches.forEach((c, j) => {
      let yy = y + H;
      c.forEach((cnt, i) => { yy -= cnt * sh; b += rect(L + j * (cw + 4), yy, cw, cnt * sh, DCOL[i], 'fill-opacity="0.75"'); });
    });
    ds.forEach((d, i) => { b += rect(L + s.nBatches * (cw + 4) + 14, y + 4 + i * 16, 10, 10, DCOL[i]) + text(L + s.nBatches * (cw + 4) + 30, y + 13 + i * 16, `${d.nm}: total ${m.totals[i]} (expected ${fmt(m.expected[i], 2)})`, { size: 11 }); });
    pic.innerHTML = svg(W, y + H + 12, b);
    const regime = m.redo ? `<b>multinomial every batch</b> (${ds.length > s.batch ? `${ds.length} datasets > batch ${s.batch} forces it` : "dynamic_ratio_every_step True"}): counts vary batch to batch; only their mean is probs × B. Draws shown come from a seeded JS generator, not numpy's stream`
      : `<b>fixed counts</b> (largest remainder, computed once): floor(${m.raw.map(r => fmt(r, 2)).join(", ")}) = [${m.base.join(", ")}], ${m.left} left over → <b>[${m.counts.join(", ")}]</b> in every batch`;
    read.innerHTML = `${regime}<br>
      <span class="big">${ds.map((d, i) => `${d.nm} = ${m.counts[i]}`).join(" · ")}</span> per batch of ${s.batch}${s.nBatches > 1 ? ` · over ${s.nBatches} batches: ${ds.map((d, i) => `${d.nm} ${m.totals[i]}`).join(", ")}` : ""}<br>
      ${m.zero ? `<b>${m.zero} dataset${m.zero > 1 ? "s" : ""} rounded to 0: never sampled</b> (the sampler prints a warning)<br>` : ""}
      within a dataset, indices are drawn with replacement: a prompt can repeat inside an epoch or never appear<br>
      <span class="muted small">provenance: data_feeds/mixed_sampler.py:L50-L51 (float32 probs), L70-L72 (fixed vs redo), L80-L90 (largest remainder), L93-L99 (zero warning), L110 (multinomial), L127-L128 (with replacement); local batch = engines × rollout_batch_size_per_gpu (core/rl_engines.py:L198, L222; gsm8k: 2 × 64, one dataset, train_sync.yaml:L107).</span>`;
  };
  const inp = el("input", { type: "text", value: s.text, size: 34 });
  inp.addEventListener("input", () => { s.text = inp.value; draw(); });
  root.append(el("div", { class: "widget" }, el("div", { class: "controls" },
    el("label", {}, "data.train_ratios ", inp),
    slider("local batch size B", 1, 128, s.batch, 1, v => { s.batch = v; draw(); }),
    check("train.dynamic_ratio_every_step", s.dynamic, v => { s.dynamic = v; draw(); }),
    slider("batches", 1, 20, s.nBatches, 1, v => { s.nBatches = v; draw(); }),
    read), pic)); draw();
};

// ======================================================= 4. group zscore ==
// From per-token rewards to the stored advantage, inside the rollout engine (rollouts/vllm_engine.py, rollouts/base.py):
//   rewards[prompt_len:] = compute_score's tensor, one value per response token (vllm_engine.py:L405-L407; length checked at
//     base.py:L180-L181). An empty response is counted but not added to the group (vllm_engine.py:L411-L417).
//   the group statistic is the SUM of a response's rewards: group_stats['rewards'].append(rewards_resp.sum()) (L416).
//   normalize_rewards (base.py:L121-L163): n > 1: mean = Σ/n, std = sqrt(Σ(r - mean)² / max(n - 1, 1)) (Bessel, L131-L136);
//     n = 1: mean 0, std 1, so the raw reward is kept (L138-L142). The numerator reads only the LAST token:
//     zscore[-1] = (token_rewards[-1] - mean) / (std + 1e-8) (L151-L152). reward.broadcast True copies it to every response
//     token (L154-L155; recipe True, yaml:L65). Prediction-aligned: pred_zscores[prompt_len-1 : T-1] = token_zscores[prompt_len:]
//     (L159-L163), because the logit at position t predicts token t+1.
//   the trainer reads micro_batch['zscore'][:, :-1] as the advantage (algs/GRPO/grpo.py:L316); micro-batches are a uniform
//     shuffle of the whole buffer (core/rl_engines.py:L457-L467), so the advantage is never recomputed from the batch.
function groupZ({ groups, promptLen = 3, broadcast = true }) {
  const o = { groups: [], mismatch: 0, kept: 0, empty: 0 };
  groups.forEach((g, gi) => {
    const kept = g.filter(r => r.length > 0);
    o.empty += g.length - kept.length; o.kept += kept.length;
    const sums = kept.map(r => r.reduce((a, x) => a + x, 0)), n = kept.length;
    let mean = 0, std = 1;
    if (n > 1) { mean = sums.reduce((a, x) => a + x, 0) / n; std = Math.sqrt(sums.reduce((a, x) => a + (x - mean) ** 2, 0) / Math.max(n - 1, 1)); }
    const rows = g.map(r => {
      if (!r.length) return { r, empty: true };
      const T = promptLen + r.length, last = r[r.length - 1], sum = r.reduce((a, x) => a + x, 0);
      const z = (last - mean) / (std + 1e-8);
      const tok = new Array(T).fill(0); tok[T - 1] = z; if (broadcast) for (let t = promptLen; t < T; t++) tok[t] = z;
      const pred = new Array(T).fill(0); for (let t = promptLen; t < T; t++) pred[t - 1] = tok[t];
      if (Math.abs(sum - last) > 1e-9) o.mismatch++;
      return { r, z, sum, last, tok, pred, T };
    });
    let k = 0; rows.forEach(row => { if (!row.empty) { k++; o[`z${gi + 1}.${k}`] = row.z; } });
    o[`mean${gi + 1}`] = mean; o[`std${gi + 1}`] = std;
    o[`sumZ${gi + 1}`] = Math.round(rows.filter(r => !r.empty).reduce((a, r) => a + r.z, 0) * 1e6) / 1e6;
    o[`nonzeroPred${gi + 1}`] = rows.filter(r => !r.empty).reduce((a, r) => a + r.pred.filter(v => Math.abs(v) > 1e-12).length, 0);
    o.groups.push({ n, mean, std, rows });
  });
  return o;
}
MODELS["fixture:feynrl--group-zscore"] = {
  fn: groupZ,
  cases: [
    { args: { groups: [[[1], [0], [0], [0]]] }, pick: "z1.1", expect: 1.5 },                       // group-relative-advantage predict
    { args: { groups: [[[1], [0], [0], [0]]] }, pick: "z1.2", expect: -0.5 },
    { args: { groups: [[[1], [0], [0], [0]]] }, pick: "std1", expect: 0.5 },                       // Bessel: sqrt(0.75 / 3)
    { args: { groups: [[[1], [1], [1], [1]]] }, pick: "z1.1", expect: 0, tol: 0 },                 // transfer: uniform group, all zero
    { args: { groups: [[[0.7]]] }, pick: "z1.1", expect: 0.7, tol: 0.05, from: "feynrl:group-relative-advantage:check" },
    { args: { groups: [[[0, 0.6, 0.4], [0, 0, 0]]] }, pick: "z1.1", expect: -0.1414, tol: 0.05, from: "feynrl:reward-interface:check" },
    { args: { groups: [[[0, 0, 1.0], [0, 0, 0]]] }, pick: "z1.1", expect: 0.7071 },                // all mass on the last token: consistent
    { args: { groups: [[[0.5, 0.5], [0, 0]]] }, pick: "mean1", expect: 0.5 },                      // reward-interface transfer: the mean uses the sum 1.0
    { args: { groups: [[[0.5, 0.5], [0, 0]]] }, pick: "mismatch", expect: 1, tol: 0 },
    { args: { groups: [[[0, 1], [0, 0], []]] }, pick: "empty", expect: 1, tol: 0 },                // edge: an empty response leaves the group
    { args: { groups: [[[0, 1], [0, 0], []]] }, pick: "std1", expect: 0.7071 },                    // ... so n = 2, not 3
    { args: { groups: [[[0, 0, 1], [0, 0, 0]]], broadcast: false }, pick: "nonzeroPred1", expect: 2, tol: 0 }, // no broadcast: last token only
    { args: { groups: [[[0, 0, 1], [0, 0, 0]]], broadcast: true }, pick: "nonzeroPred1", expect: 6, tol: 0 },  // broadcast: every response token
    { args: { groups: [[[1], [0], [0], [0]], [[1], [1], [1], [1]]] }, pick: "sumZ1", expect: 0, tol: 0 },     // z sums to 0 inside a group
  ],
};
WIDGETS["fixture:feynrl--group-zscore"] = (root) => {
  const s = { text: "0 0 1, 0 1, 0 0 0, 0 0 ; 0 1, 0 0 1, 0 1, 0 1", broadcast: true, promptLen: 3, mb: 4 };
  const pic = el("div"), read = el("div", { class: "readout" });
  const parse = t => t.split(";").map(g => g.split(",").map(r => r.trim().split(/\s+/).filter(Boolean).map(Number).filter(Number.isFinite))).filter(g => g.length).slice(0, 3).map(g => g.slice(0, 6).map(r => r.slice(0, 12)));
  const zc = v => Math.abs(v) < 1e-12 ? "#fff" : v > 0 ? `rgba(60,141,90,${Math.min(0.85, 0.2 + Math.abs(v) / 2.5)})` : `rgba(184,88,42,${Math.min(0.85, 0.2 + Math.abs(v) / 2.5)})`;
  const draw = () => {
    const groups = parse(s.text), m = groupZ({ groups, promptLen: s.promptLen, broadcast: s.broadcast });
    const W = 640, L = 100, cs = 22;
    let b = "", y = 14;
    m.groups.forEach((g, gi) => {
      b += text(10, y + 12, `prompt ${gi + 1}`, { weight: "bold", size: 12 }) + text(L, y + 12, g.n > 1 ? `n = ${g.n} · mean of sums ${fmt(g.mean, 4)} · std (n−1) ${fmt(g.std, 4)}${g.std < 1e-12 ? " · all equal: every z = 0" : ""}` : g.n === 1 ? "n = 1: mean 0, std 1, the raw reward is kept" : "no non-empty response", { fill: C.muted, size: 11 });
      y += 20;
      g.rows.forEach((row, ri) => {
        if (row.empty) { b += text(L, y + 14, "empty response: counted, not added to the group or the buffer", { fill: C.muted, size: 11 }); y += 24; return; }
        b += text(L - 8, y + 15, "reward", { fill: C.muted, size: 10, anchor: "end" }) + text(L - 8, y + cs + 15, "pred z", { fill: C.muted, size: 10, anchor: "end" });
        for (let t = 0; t < row.T; t++) {
          const isP = t < s.promptLen, x0 = L + t * cs;
          b += rect(x0, y, cs - 2, cs - 2, isP ? "#eee" : t === row.T - 1 ? C.hi : "#fff", 'stroke="var(--rule)"' + (t === row.T - 1 ? ' fill-opacity="0.5"' : ""));
          if (!isP) b += text(x0 + cs / 2 - 1, y + 15, fmt(row.r[t - s.promptLen], 2), { size: 9, anchor: "middle" });
          b += rect(x0, y + cs, cs - 2, cs - 2, zc(row.pred[t]), 'stroke="var(--rule)"');
          if (Math.abs(row.pred[t]) > 1e-12 && cs > 20) b += text(x0 + cs / 2 - 1, y + cs + 15, fmt(row.pred[t], 2), { size: 8, anchor: "middle" });
        }
        b += text(L + row.T * cs + 8, y + 15, `Σ = ${fmt(row.sum, 3)}${Math.abs(row.sum - row.last) > 1e-9 ? ` ≠ last ${fmt(row.last, 3)}` : ""}`, { size: 11, fill: Math.abs(row.sum - row.last) > 1e-9 ? C.b : C.ink });
        b += text(L + row.T * cs + 8, y + cs + 15, `z = ${sgn(row.z, 4)}`, { size: 11, weight: "bold" });
        y += 2 * cs + 8;
      });
      y += 6;
    });
    // the trainer's first micro-batch: a seeded uniform shuffle of every stored sample
    const pool = []; m.groups.forEach((g, gi) => { let k = 0; g.rows.forEach(r => { if (!r.empty) pool.push({ g: gi + 1, i: ++k, z: r.z }); }); });
    let st = 7; const rnd = () => (st = (st * 16807) % 2147483647) / 2147483647;
    for (let i = pool.length - 1; i > 0; i--) { const j = Math.floor(rnd() * (i + 1)); [pool[i], pool[j]] = [pool[j], pool[i]]; }
    const mb = pool.slice(0, s.mb);
    b += text(10, y + 14, "micro-batch 1", { weight: "bold", size: 12 }) + text(L, y + 14, mb.map(p => `p${p.g}·s${p.i}: z ${sgn(p.z, 3)}`).join("   "), { size: 11 });
    pic.innerHTML = svg(W, y + 26, b);
    read.innerHTML = `${m.groups.map((g, gi) => `prompt ${gi + 1}: z = [${g.rows.filter(r => !r.empty).map(r => sgn(r.z, 3)).join(", ")}] · Σz = ${fmt(m[`sumZ${gi + 1}`], 3)}`).join("<br>")}<br>
      ${m.mismatch ? `<b>${m.mismatch} response${m.mismatch > 1 ? "s" : ""} with reward off the last token</b>: the group mean uses the sum, the numerator only the last token, so they disagree. Put the whole scalar on r[−1]<br>` : "every response's reward sits on its last token: the sum and the last token agree<br>"}
      ${s.broadcast ? "broadcast on: every response token carries the response's z" : "broadcast off: only the last response token carries z, the rest are 0"} · shifted one left (prediction-aligned)<br>
      the micro-batch mixes prompts and reads the stored z; nothing is normalized over it<br>
      <span class="muted small">provenance: rollouts/vllm_engine.py:L405-L417 (rewards, sum, empty responses), L491-L494; rollouts/base.py:L131-L142 (mean, Bessel std, n = 1), L151-L163 (last token, broadcast, pred-aligned); algs/GRPO/grpo.py:L316; core/rl_engines.py:L457-L467 (shuffle). Recipe: reward.broadcast True, n_samples 4 (train_sync.yaml:L65, L71). The shuffle shown is a seeded JS stand-in for torch's generator.</span>`;
  };
  const inp = el("input", { type: "text", value: s.text, size: 48 });
  inp.addEventListener("input", () => { s.text = inp.value; draw(); });
  root.append(el("div", { class: "widget" }, el("div", { class: "controls" },
    el("label", {}, "per-token rewards (spaces: tokens, commas: responses, ; : prompts; an empty slot is an empty response) ", inp),
    check("reward.broadcast", s.broadcast, v => { s.broadcast = v; draw(); }),
    slider("prompt length (tokens)", 1, 4, s.promptLen, 1, v => { s.promptLen = v; draw(); }),
    slider("train_batch_size_per_gpu", 1, 8, s.mb, 1, v => { s.mb = v; draw(); }),
    read), pic)); draw();
};

// ========================================================= 5. clip loss ==
// compute_policy_loss, one row per token (sums, before normalization).
//   GRPO standard (algs/GRPO/grpo.py:L169-L176): ratio = exp(logprobs - old_logprobs) in float32, masked positions get
//     log-ratio 0 (L170); pi_sum = -Σ min(ratio·A, clamp(ratio, 1 - clip_low, 1 + clip_high)·A)·mask.
//   GRPO decoupled, overlap mode (L145-L159): r_prox = exp(logprobs - prox_logprobs) is clipped instead, and the surrogate is
//     multiplied by behave_w = exp(prox_logprobs - old_logprobs), clamped to [0, behave_imp_weight_cap] when the cap is set.
//   CISPO standard (algs/CISPO/cispo.py:L164-L175): pi_sum = -Σ clamp(ratio).detach() · logprobs · A · mask; decoupled
//     (L146-L159) adds behave_w the same way.
//   metrics (grpo.py:L190-L213, cispo.py:L189-L200): denom = Σmask clamped to ≥ 1; pi_loss = pi_sum / denom;
//     clipfrac = share of masked tokens with ratio > 1 + clip_high or < 1 - clip_low.
//   gradient d(pi_sum)/d(logprobs) per token (d ratio / d logprobs = ratio): GRPO takes the min's branch; the clamp branch is
//     flat outside the band, so a token whose min is the clamped value contributes 0; CISPO's weight is detached, so its
//     gradient is -clamp(ratio)·A·w, never 0 unless A = 0.
// Recipe: clip_low = clip_high = 0.4 (yaml:L44-L45), behave_imp_weight_cap 5.0 (train_async.yaml:L30).
function clipLoss({ tokens, variant = "grpo", decoupled = false, clipLow = 0.4, clipHigh = 0.4, cap = 5.0 }) {
  const lo = 1 - clipLow, hi = 1 + clipHigh, clamp = r => Math.min(Math.max(r, lo), hi);
  const o = { rows: [], piSum: 0, denom: 0, clipped: 0, capped: 0 };
  tokens.forEach((t, i) => {
    const mask = t.mask === 0 ? 0 : 1;
    const lr = mask ? (decoupled ? t.lp - t.prox : t.lp - t.old) : 0, r = Math.exp(lr);
    let w = 1;
    if (decoupled) { w = mask ? Math.exp(t.prox - t.old) : 1; if (cap !== null && cap !== undefined) { if (w >= cap) o.capped += mask; w = Math.min(Math.max(w, 0), cap); } }
    const A = t.A, cr = clamp(r), unc = r * A, cl = cr * A;
    let term, grad, branch;
    if (variant === "cispo") { term = -cr * t.lp * A * w * mask; grad = -cr * A * w * mask; branch = r > hi || r < lo ? "weight clamped" : "weight = ratio"; }
    else {
      const mn = Math.min(unc, cl); term = -mn * w * mask;
      branch = unc < cl ? "unclipped" : cl < unc ? "clipped" : "equal";
      grad = branch === "clipped" && (r >= hi || r <= lo) ? 0 : -A * r * w * mask;
    }
    const isClipped = mask && (r > hi || r < lo) ? 1 : 0;
    o.piSum += term; o.denom += mask; o.clipped += isClipped;
    o.rows.push({ r, w, unc, cl, term, grad, branch, isClipped, mask });
    o[`r${i + 1}`] = r; o[`w${i + 1}`] = w; o[`grad${i + 1}`] = grad; o[`term${i + 1}`] = term;
  });
  o.denom = Math.max(o.denom, 1);
  o.piLoss = o.piSum / o.denom; o.clipfrac = o.clipped / o.denom; o.capfrac = o.capped / o.denom;
  return o;
}
const LN = Math.log;
MODELS["fixture:feynrl--clip-loss"] = {
  fn: clipLoss,
  cases: [
    { args: { tokens: [{ lp: 10, old: 0, A: 1 }], clipLow: 0.2, clipHigh: 0.2 }, pick: "piLoss", expect: -1.2 },        // grpo predict (test_grpo_loss_clipping)
    { args: { tokens: [{ lp: 10, old: 0, A: 1 }], clipLow: 0.2, clipHigh: 0.2 }, pick: "clipfrac", expect: 1 },
    { args: { tokens: [{ lp: 10, old: 0, A: 1 }], clipLow: 0.2, clipHigh: 0.2 }, pick: "grad1", expect: 0, tol: 0 },    // clipped: loss -1.2, gradient 0
    { args: { tokens: [{ lp: LN(0.5), old: 0, A: -1 }], clipLow: 0.2, clipHigh: 0.2 }, pick: "piLoss", expect: 0.8 },   // grpo transfer
    { args: { tokens: [{ lp: LN(0.5), old: 0, A: -1 }], clipLow: 0.2, clipHigh: 0.2 }, pick: "grad1", expect: 0, tol: 0 },
    { args: { tokens: [{ lp: LN(1.5), old: 0, A: -2 }, { lp: LN(1.5), old: 0, A: 2 }], clipLow: 0.2, clipHigh: 0.28 }, pick: "piLoss", expect: 0.22, tol: 0.05, from: "feynrl:grpo-clipped-loss:check" },
    { args: { tokens: [{ lp: LN(1.5), old: 0, A: -2 }, { lp: LN(1.5), old: 0, A: 2 }], clipLow: 0.2, clipHigh: 0.28 }, pick: "grad1", expect: 3.0 },     // A < 0 above the band: unclipped, keeps -ratio·A
    { args: { tokens: [{ lp: -0.5, old: -10.5, A: 1 }], variant: "cispo", clipLow: 0.2, clipHigh: 0.2 }, pick: "piLoss", expect: 0.6, tol: 0.01, from: "feynrl:cispo-vs-ppo-clip:predict" },
    { args: { tokens: [{ lp: -0.5, old: -10.5, A: 1 }], variant: "cispo", clipLow: 0.2, clipHigh: 0.2 }, pick: "grad1", expect: -1.2 },  // cispo transfer: -1.2 ...
    { args: { tokens: [{ lp: -0.5, old: -10.5, A: 1 }], clipLow: 0.2, clipHigh: 0.2 }, pick: "grad1", expect: 0, tol: 0 },               // ... GRPO 0
    { args: { tokens: [{ lp: LN(0.6), old: 0, A: 1 }], variant: "cispo", clipLow: 0.2, clipHigh: 0.2 }, pick: "grad1", expect: -0.8, tol: 0.05, from: "feynrl:cispo-vs-ppo-clip:check" },
    { args: { tokens: [{ lp: LN(0.6), old: 0, A: 1 }], clipLow: 0.2, clipHigh: 0.2 }, pick: "grad1", expect: -0.6 },                     // GRPO: unclipped branch below the band
    { args: { tokens: [{ lp: 0, prox: -LN(1.5), old: -LN(1.5) - LN(2), A: 1 }], decoupled: true, clipLow: 0.2, clipHigh: 0.2, cap: 5 }, pick: "piSum", expect: -2.4, tol: 0.05, from: "feynrl:decoupled-loss-prox:check" },
    { args: { tokens: [{ lp: LN(3), old: 0, A: 1 }], clipLow: 0.2, clipHigh: 0.2 }, pick: "piSum", expect: -1.2 },                         // the check's distractor: sync formula on pi/pi_old = 3
    { args: { tokens: [{ lp: 0, prox: 0, old: -2, A: 1 }], decoupled: true, cap: 5 }, pick: "w1", expect: 5 },                           // decoupled transfer: e^2 capped at 5
    { args: { tokens: [{ lp: 0, prox: 0, old: -2, A: 1 }], decoupled: true, cap: null }, pick: "w1", expect: 7.389 },                    // ... cap null
    { args: { tokens: [{ lp: LN(0.2), old: LN(0.1), A: 1 }] }, pick: "r1", expect: 2 },                                                  // importance ratio 0.2/0.1
    { args: { tokens: [{ lp: 5, old: 0, A: 1, mask: 0 }, { lp: 0, old: 0, A: 1 }] }, pick: "piLoss", expect: -1 },                      // edge: a masked token adds nothing, denom 1
  ],
};
WIDGETS["fixture:feynrl--clip-loss"] = (root) => {
  const s = { variant: "grpo", decoupled: false, clipLow: 0.4, clipHigh: 0.4, cap: 5.0, capOn: true, sel: 0,
    tokens: [{ lp: -1.0, old: -1.5, prox: -1.2, A: 1.0, mask: 1 }, { lp: -2.0, old: -1.9, prox: -1.95, A: -0.5, mask: 1 }, { lp: -0.7, old: -0.7, prox: -0.7, A: 0.5, mask: 0 }] };
  const pic = el("div"), read = el("div", { class: "readout" }), table = el("div");
  const args = () => ({ tokens: s.tokens, variant: s.variant, decoupled: s.decoupled, clipLow: s.clipLow, clipHigh: s.clipHigh, cap: s.capOn ? s.cap : null });
  const draw = () => {
    const m = clipLoss(args()), t = s.tokens[s.sel];
    // gradient of this token's term against its log-ratio, the other inputs held: GRPO solid, CISPO dashed
    const W = 640, H = 230, L = 50, R = 610, T = 24, B = 190, xmin = -1.2, xmax = 1.2;
    const base = s.decoupled ? t.prox : t.old;
    const gAt = (variant, x) => clipLoss({ ...args(), variant, tokens: [{ ...t, mask: 1, lp: base + x }] }).grad1;
    const xs = []; for (let x = xmin; x <= xmax + 1e-9; x += 0.01) xs.push(x);
    const all = xs.flatMap(x => [gAt("grpo", x), gAt("cispo", x)]), ymax = Math.max(0.5, ...all.map(Math.abs)) * 1.1;
    const X = x => L + (x - xmin) / (xmax - xmin) * (R - L), Y = v => T + (ymax - v) / (2 * ymax) * (B - T);
    let b = rect(X(LN(1 - s.clipLow)), T, X(LN(1 + s.clipHigh)) - X(LN(1 - s.clipLow)), B - T, C.hi, 'fill-opacity="0.15"');
    b += line(L, Y(0), R, Y(0), C.rule) + line(X(0), T, X(0), B, C.rule, 'stroke-dasharray="3 3"');
    b += path(xs.map(x => [X(x), Y(gAt("grpo", x))]), C.a) + path(xs.map(x => [X(x), Y(gAt("cispo", x))]), C.b, 'stroke-dasharray="6 4"');
    const cx = Math.min(xmax, Math.max(xmin, t.lp - base)), cur = gAt(s.variant, cx);
    b += `<circle cx="${X(cx).toFixed(1)}" cy="${Y(cur).toFixed(1)}" r="5" style="fill:${m.rows[s.sel].isClipped ? C.b : C.ok}"/>`;
    b += text(L, 14, `token ${s.sel + 1}: d(pi_sum)/d(logprobs) against log(π/${s.decoupled ? "π_prox" : "π_old"}) · solid GRPO · dashed CISPO · band shaded`, { fill: C.muted, size: 11 });
    b += text(X(LN(1 - s.clipLow)), B + 14, `1−${s.clipLow}`, { fill: C.muted, size: 10, anchor: "middle" }) + text(X(LN(1 + s.clipHigh)), B + 14, `1+${s.clipHigh}`, { fill: C.muted, size: 10, anchor: "middle" });
    b += text(R, B + 30, `x = log-ratio (ratio = e^x, 0.30 … 3.32) · A = ${fmt(t.A, 2)}${s.decoupled ? ` · w = ${fmt(m.rows[s.sel].w, 3)}` : ""}`, { fill: C.muted, size: 10, anchor: "end" });
    b += text(L - 6, Y(ymax * 0.9), fmt(ymax * 0.9, 2), { fill: C.muted, size: 10, anchor: "end" }) + text(L - 6, Y(-ymax * 0.9), fmt(-ymax * 0.9, 2), { fill: C.muted, size: 10, anchor: "end" }) + text(L - 6, Y(0) + 4, "0", { fill: C.muted, size: 10, anchor: "end" });
    pic.innerHTML = svg(W, H + 12, b);
    const rows = m.rows.map((r, i) => r.mask ? `token ${i + 1}: ${s.decoupled ? "r_prox" : "ratio"} ${fmt(r.r, 4)}${s.decoupled ? ` · w ${fmt(r.w, 3)}` : ""} · ${s.variant === "grpo" ? `ratio·A ${fmt(r.unc, 4)}, clip·A ${fmt(r.cl, 4)} → <b>${r.branch === "equal" ? "inside the band (both equal)" : r.branch}</b>` : `clamped weight ${fmt(Math.min(Math.max(r.r, 1 - s.clipLow), 1 + s.clipHigh), 4)} (${r.branch}, detached)`} · term ${sgn(r.term, 4)} · <b>gradient ${sgn(r.grad, 4)}</b>${s.variant === "grpo" && r.grad === 0 && Math.abs(r.term) > 1e-12 ? " (loss not 0, gradient 0)" : ""}` : `token ${i + 1}: masked, adds nothing`).join("<br>");
    read.innerHTML = `${rows}<br>
      <span class="big">pi_sum = ${sgn(m.piSum, 4)} · pi_loss = pi_sum / ${m.denom} = ${sgn(m.piLoss, 4)}</span> · clipfrac ${fmt(m.clipfrac, 3)}${s.decoupled && s.capOn ? ` · behave_w capped on ${fmt(m.capfrac, 3)}` : ""}<br>
      ${s.variant === "grpo" ? "GRPO: once the min picks the clamped value (ratio past the band in A's direction) the token's gradient is 0, though its loss term is not" : "CISPO: the clamp only sets a detached weight on log π, so every masked token keeps gradient −clamp(ratio)·A"}${s.decoupled ? "; decoupled: the band is around π_prox, and w = π_prox/π_old rescales the term" : ""}<br>
      <span class="muted small">provenance: algs/GRPO/grpo.py:L140-L176 (standard and decoupled), L190-L213 (clipfrac, pi_loss); algs/CISPO/cispo.py:L141-L175; use_decoupled_loss = overlap.enabled (core/rl_engines.py:L59). Defaults: clip 0.4/0.4 (train_sync.yaml:L44-L45), cap 5.0 (train_async.yaml:L30); the token values are made up.</span>`;
  };
  const buildTable = () => {
    const cols = [["lp", "log π"], ["old", "log π_old"], ...(s.decoupled ? [["prox", "log π_prox"]] : []), ["A", "advantage"]];
    const num = (i, k) => { const x = el("input", { type: "number", step: "0.01", value: s.tokens[i][k], style: "width:70px" }); x.addEventListener("input", () => { const v = Number(x.value); if (Number.isFinite(v)) { s.tokens[i][k] = v; draw(); } }); return x; };
    table.replaceChildren(...s.tokens.map((tk, i) => el("div", { class: "small" },
      el("label", {}, el("input", { type: "radio", name: "frl-clip-sel", ...(i === s.sel ? { checked: "" } : {}), onchange: () => { s.sel = i; draw(); } }), ` token ${i + 1} `),
      ...cols.flatMap(([k, nm]) => [` ${nm} `, num(i, k)]),
      " ", check("mask", tk.mask === 1, v => { tk.mask = v ? 1 : 0; draw(); }))));
  };
  const set = (k, v) => { s[k] = v; draw(); };
  root.append(el("div", { class: "widget" }, el("div", { class: "controls" },
    select("train.alg_name", [["grpo", "grpo (min-clip)"], ["cispo", "cispo (detached clamp weight)"]], s.variant, v => set("variant", v)),
    check("decoupled loss (overlap.enabled)", s.decoupled, v => { s.decoupled = v; buildTable(); draw(); }),
    slider("clip_low", 0.05, 0.5, s.clipLow, 0.01, v => set("clipLow", v), v => v.toFixed(2)),
    slider("clip_high", 0.05, 0.5, s.clipHigh, 0.01, v => set("clipHigh", v), v => v.toFixed(2)),
    check("behave_imp_weight_cap set", s.capOn, v => set("capOn", v)),
    slider("behave_imp_weight_cap", 1.1, 10, s.cap, 0.1, v => set("cap", v), v => v.toFixed(1)),
    table, read), pic)); buildTable(); draw();
};
