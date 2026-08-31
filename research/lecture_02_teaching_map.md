# CS336 Spring 2026 Lecture 2 teaching map

This document reconstructs the lecture as Percy Liang actually teaches it, rather than treating `lecture_02.py` as 39 independent content cards. It is a design brief for augmenting the original lecture spine with formulas, tables, links, and deliberately paced demonstrations.

No transcript is stored here. All descriptions below are paraphrases derived from the public subtitle track and checked against the official executable lecture source. Subtitle timecodes can drift by a few seconds.

## Verified recording and source boundary

| Field | Verified value |
|---|---|
| Offering | Stanford CS336, Spring 2026 |
| Schedule row | Lecture 2, Wednesday April 1 |
| Instructor | Percy Liang |
| Official schedule description | PyTorch (einops), resource accounting (FLOPs, memory, arithmetic intensity) |
| Official video | <https://www.youtube.com/watch?v=kuYAsz7zspQ> |
| Playlist position | 2 in <https://www.youtube.com/playlist?list=PLoROMvodv4rMqXOcazWaTUHhq-yembLCV> |
| YouTube title | `Stanford CS336 Language Modeling from Scratch \| Spring 2026 \| Lecture 2: PyTorch (einops)` |
| Recording duration | 4,645 seconds, approximately 1:17:25 |
| Spoken lecture boundary | First speech at about 00:05; conclusion finishes at about 1:17:24 |
| Official executable source | <https://github.com/stanford-cs336/lectures/blob/main/lecture_02.py> |
| Source revision inspected | `8b59b50730766695c2ffedd1a79c50cd09b9eb91` (2026-05-27) |
| Schedule evidence | <https://cs336.stanford.edu/> |

The public recording exposes an English subtitle track. It was downloaded without audio or video and used only as temporary research input. The raw subtitle file is not part of the repository.

The short YouTube title is incomplete as a description of the class. Einops occupies only about nine minutes. The actual lecture is a resource-accounting argument that proceeds from storage, through computation, to the combined memory/compute cost of training.

## The lecture's real thesis

The lecture is one continuous accounting exercise:

> Given finite compute and memory, learn to estimate what a training computation costs, diagnose which resource limits it, and recognize the standard trades that change the bill.

Percy opens two unpaid “ledger entries” before teaching the mechanics:

1. How long does a 70B-parameter, 15T-token run take on 1,024 H100s?
2. How large a model fits on eight 80 GB H100s with AdamW?

The rest of the lecture supplies the missing terms. The first question is paid off by matmul FLOPs, measured throughput, MFU, backward-pass accounting, and finally `6ND`. The second is paid off by dtype sizes, parameters, gradients, optimizer state, the omitted activation term, and finally gradient accumulation/checkpointing.

The intended mindset is more important than any isolated fact: whenever a line of tensor code is written, ask about shapes, bytes moved, FLOPs performed, and the hardware bottleneck.

## Argument graph: promises and later payoffs

| Opening promise | Deferred terms | Where Percy pays it off |
|---|---|---|
| `training FLOPs = 6 × parameters × tokens` at 02:05 | Why 2? Why 6? Why tokens and parameters? | A matmul becomes `2 × data points × parameters` at 34:00; two backward matmuls make backward twice forward at 62:00–65:20; the result closes as `2ND + 4ND = 6ND` at 65:30–66:35. |
| 143-day estimate at 02:05–02:45 | H100 dense BF16 peak, number of GPUs, seconds/day, MFU | Dense peak is separated from the sparse marketing figure at 29:00–30:00; MFU is defined at 35:00–39:30; memory bottlenecks explain why realized throughput is lower at 40:30–55:00. |
| `2 + 2 + 4 + 4 = 12 bytes/parameter` at 02:45–03:35 | Which tensor owns each term? Why mixed precision? What is missing? | BF16/FP32 roles at 11:00–13:10; parameters, gradients, AdaGrad/Adam states and activations at 69:15–71:45. Activations remain a separate, batch/sequence-dependent term. |
| “Memory affects speed too” at 08:00–08:45 | Why should fewer bytes reduce wall time? | HBM-to-compute movement at 40:30–42:25; `max(memory time, compute time)` at 43:45–45:10; arithmetic intensity through 57:00. |
| MFU is often about 0.5 at 37:00–39:25 | Why not close to 1? | Percy explicitly defers the answer until the memory-bottleneck section, then connects low arithmetic intensity to idle compute at 40:30–55:00. |

These returns are the lecture's narrative glue. A redesigned site should visibly preserve them, for example with “opened at 02:05 / resolved at 66:10” links or a small persistent resource ledger.

## Detailed timecoded teaching map

### 00:05–01:00 — Current event as a bridge from Lecture 1

- Percy reports that the Marin `1e23`-FLOP run finished close to its scaling-law forecast.
- This is not a disposable announcement: it puts a concrete compute scale on the screen before the resource lecture begins and connects to the course's “best model under a budget” framing.
- He adds an extrapolation caveat: scaling-law behavior is not guaranteed to remain identical farther out.

**Augment lightly:** a link to the run/forecast and a one-line definition of an IsoFLOPs curve are enough. Do not insert a standalone scaling-laws explainer here; the point is continuity, not a new topic.

### 01:00–04:50 — The contract: two napkin questions and three kinds of learning

- 01:00–01:55: last lecture was overview/tokenization; today is the systems side, with compute and memory as finite resources.
- 01:55–03:35: two back-of-the-envelope questions establish the target intuitions. Percy shows the answers before deriving them: roughly 143 days for the 70B/15T run, and roughly 53B parameters as a naive eight-H100 upper bound.
- 03:00–03:45: he immediately marks activations as absent from the capacity estimate and says the goal is the rough shape, not perfect accounting.
- 03:45–04:50: he distinguishes mechanics (PyTorch/tensor semantics), mindset (remember resource accounting), and intuitions (where resources go). “No ML magic” is a deliberate scope boundary.

**Best augmentation:** one two-tab “open ledger” using the exact assumptions, with unknown terms visibly tagged “derive later.” It should remain referentially available throughout the lecture and receive checkmarks at 37:00, 66:10, and 70:30.

**Do not interrupt:** 01:20–04:50 is one motivational unit. Splitting the two questions into independent cards loses the promise that the lecture will explain them.

### 04:50–08:00 — Bottom-up storage: tensors, shapes, FP32, bytes

- 04:50–05:35: all model artifacts are tensors: data, parameters, gradients, optimizer states, and activations. DeepSeek is used as a real model-file example.
- 05:35–06:55: floating-point values, FP32 bit allocation, dynamic range versus local precision, and the historical meaning of single/double precision.
- 06:55–08:00: a `4 × 8` FP32 tensor becomes `4 × 8 × 4 = 128` bytes; the GPT-3 FFN matrix scales the same formula to about 2.3 GB.

**Best augmentation:** a persistent tensor ledger table with columns `role`, `shape`, `dtype`, `lifetime`, `numel`, and `bytes`. Animate the 4-by-8 count once, then let the HTML table do the static comparison work.

**Transition:** wanting efficiency leads directly to reducing precision; this is not a fresh chapter reset.

### 08:00–17:00 — Precision as a memory/range trade, then a live digression

- 08:00–09:45: fewer bits can reduce storage and may improve speed, but Percy explicitly says “twice as fast” is not universal. FP16's five exponent bits make both underflow and overflow practical training risks; `1e-8` becomes zero.
- 09:45–10:55: BF16 reallocates the same 16-bit budget toward the exponent. It keeps FP32-like range at worse resolution, a trade that often fits noisy deep-learning workloads.
- 10:55–13:10: the operational consequence is mixed precision: BF16 for parameters/activations/gradients, FP32 for optimizer state; AMP is operation-aware, with matmul versus exponentiation used only as examples.
- 13:10–14:55: FP8 has range/precision variants; NVFP4 combines a tiny local codebook with a shared block scale. Much of this is implemented below normal user-level tensor APIs.
- 15:00–17:00: student questions expand the block-scale explanation and distinguish low-bit inference quantization from actually training a credible one-bit model.

**Best augmentation:** a comparison table before any animation:

| format | sign / exponent / fraction | bytes | smallest positive subnormal (approx.) | main lecture takeaway |
|---|---:|---:|---:|---|
| FP32 | 1 / 8 / 23 | 4 | `1.40e-45` | baseline range and resolution |
| FP16 | 1 / 5 / 10 | 2 | `5.96e-8` | more local resolution than BF16, much less range |
| BF16 | 1 / 8 / 7 | 2 | `9.18e-41` | FP32-like range, coarse local resolution |

Then use a slow, step-controlled number-line demonstration for the single claim `torch.tensor([1e-8], dtype=torch.float16) == 0`, followed immediately by the mixed-precision routing table. FP8/FP4 should primarily be background links and a compact table; a long animation would over-weight Percy's digression.

**Do not interrupt:** FP16 → BF16 → mixed precision is one causal unit. FP8/FP4 and the student Q&A are a natural optional/expandable branch.

### 17:00–18:00 — CPU/GPU placement closes memory basics

- Tensors default to CPU memory and must be placed on the GPU to use GPU parallelism.
- Percy cannot execute GPU examples because the lecture is running on his laptop; he shows the code and moves on.

**Best augmentation:** a small CPU RAM → interconnect → GPU HBM diagram plus direct-on-GPU allocation. Avoid invented benchmark numbers; the live lecture does not measure transfer latency here.

### 18:00–27:25 — Einops as the bookkeeping language for the rest of the lecture

- 18:00–19:05: negative axis indices and implicit transposes are framed as error-prone; named dimensions reduce cognitive bookkeeping.
- 19:05–20:50: simple matrix multiplication in einsum form. A dimension absent from the output is contracted/summed.
- 20:50–22:45: batched attention-like scores preserve batch while contracting hidden; ellipsis stands for shared prefix dimensions.
- 22:45–24:00: `reduce` generalizes named sum/mean/max/min; a student asks about speed and Percy answers that it is semantic sugar over the same primitives.
- 24:00–26:25: `rearrange` splits a flattened `total_hidden=8` into `heads=2 × hidden1=4`, applies the matrix operation, then joins dimensions again.
- 26:25–27:25: student question about flattening order; Percy points to pattern order, then closes by saying the notation changes how one thinks about tensor code.

**Best augmentation:** one continuous code-and-shape inspector, not six independent articles. Keep the same tensor visible while the source line changes. The learner should be able to step between `split`, `einsum transform`, and `join`, with named dimensions colored consistently.

**Do not interrupt:** 18:00–27:25 is a compact language tutorial. The concepts are individually linkable but depend on the same motivation and notation.

### 27:25–40:30 — FLOPs, matmul accounting, benchmarking, and MFU

- 27:25–28:55: FLOPs (amount of work) is separated from FLOP/s (hardware rate); Percy treats the ambiguous pronunciation/spelling as a practical source of confusion.
- 28:55–30:35: model-training orders of magnitude and the H100 spec sheet. The advertised 1,979 TFLOP/s sparse figure must be halved for dense BF16. A quick GPU-week computation builds scale intuition.
- 30:35–34:45: the running linear model uses `x[B,D] @ w[D,K]`. One multiply plus approximately one add for every `(b,d,k)` yields `2BDK`. Recasting `DK` as parameter count produces `2 × data points × parameters`.
- 34:45–36:45: actual wall time requires benchmarking. GPU operations are asynchronous, so synchronization before and after the timed operation matters; repeated trials are used.
- 36:45–39:25: MFU is measured FLOP/s divided by dtype-specific promised FLOP/s. About 0.5 is a good modern-model result, but Percy defers “why only 0.5?” until the memory section.
- 39:25–40:30: matrix multiplications dominate by design, dtype changes peak rate, and the lecture pivots to the missing memory-bandwidth term.

**Best augmentation:** preserve a single `B,D,K` strip across the formula, benchmark code, and MFU. Useful additions are:

- a units table separating FLOPs, FLOP/s, seconds, and utilization;
- a slow cell-to-matrix derivation of `2BDK`;
- a benchmark checklist (`synchronize → run → synchronize → repeat`);
- a formula comparison: `MFU = measured FLOP/s ÷ dense dtype peak FLOP/s`.

**Do not interrupt:** `2BDK → measured rate → MFU` is one chain. A generic tensor-shape or hardware card inserted between them makes MFU appear unmotivated.

### 40:30–57:10 — Data movement, arithmetic intensity, and roofline

- 40:30–42:25: the hardware cartoon adds HBM. Inputs move to compute units, computation happens, and outputs move back. Memory capacity and memory bandwidth are distinct constraints, but tensor size affects both.
- 42:25–45:10: ReLU becomes the worked example. BF16 reads `2n` bytes and writes `2n` bytes for about `n` comparisons. Assuming perfect overlap, total time is the maximum of memory-transfer time and compute time, with an explicit caveat that real overlap has overhead.
- 45:10–47:50: accelerator intensity is peak FLOP/s divided by bytes/s, around 295 FLOP/byte for the lecture's H100 constants. Workload arithmetic intensity is FLOPs/bytes. ReLU is about 0.25 and therefore memory-bound.
- 47:50–49:35: GELU performs roughly 20 operations for the same read/write traffic, giving intensity around 5. It is still far below the knee; in the isolated-kernel model it need not take longer than ReLU.
- 49:35–50:35: dot product is about 0.5 FLOP/byte and memory-bound.
- 50:35–51:25: matrix-vector multiplication is about 1 FLOP/byte and still memory-bound.
- 51:25–53:55: square matrix multiplication is about `n/3`; for `n=1024`, roughly 341 FLOP/byte, just beyond the H100 knee. Reuse grows compute faster than moved bytes, so sufficiently large matrices can saturate compute.
- 53:00–54:05: training's large matrices contrast with one-token-at-a-time inference's matrix-vector behavior. Precision also changes the intensity calculation.
- 54:05–55:10: Percy returns to the deferred MFU question: memory stalls reduce realized arithmetic throughput.
- 55:10–57:10: only after every example is on the ledger does he introduce the roofline plot as a visualization of the same relationship.

**Best augmentation:** maintain one live table and add each row in teaching order:

| operation (lecture assumptions) | approximate FLOPs | approximate bytes | intensity | diagnosis vs H100 knee ≈295 |
|---|---:|---:|---:|---|
| BF16 ReLU | `n` | `4n` | `0.25` | memory-bound |
| BF16 GELU | `20n` | `4n` | `5` | memory-bound |
| BF16 dot | `2n` | `4n + 2` | `≈0.5` | memory-bound |
| BF16 matvec | `≈2n²` | `≈2n² + 4n` | `≈1` | memory-bound |
| BF16 `n×n` matmul | `≈2n³` | `≈6n²` | `≈n/3` | compute-bound once above knee |

The roofline should be the final view of this table, not a standalone explanation. Add points one at a time, retain the same colors, and let the learner scrub or step back to the byte/FLOP row that produced each point.

**Do not interrupt:** 40:30–57:10 is the strongest indivisible chain in the lecture. ReLU → GELU → dot → matvec → matmul is a deliberate ladder of reuse. The roofline is its summary.

### 57:10–66:45 — A deep network turns local matmul accounting into `6ND`

- 57:10–59:05: Percy introduces the running deep network: `L` layers, `D×D` weights, a linear operation followed by ReLU, and `D²L` parameters.
- 59:05–60:10: a minimal regression example reviews how PyTorch attaches gradients to the computation graph.
- 60:10–61:40: a two-layer linear network is rewritten with named einsum dimensions, explicitly reusing the notation taught earlier.
- 61:40–65:25: zooming into one layer, backward must compute both an input/activation gradient and a parameter gradient. Both are matmuls with the same dimension product, so backward is about twice forward.
- 65:25–66:45: summing across parameters gives `2ND` forward and `4ND` backward, finally closing the opening `6ND` promise. Percy limits the Transformer approximation: long contexts add important context-length-squared work.

**Best augmentation:** keep the deep-network diagram, forward equation, and two backward equations in one visual field. Use matching dimension colors and a three-row cost table:

| pass | matmuls being counted | cost |
|---|---|---:|
| forward | activation × weight | `2ND` |
| backward | activation gradient + weight gradient | `4ND` |
| total | forward + backward | `6ND` |

At the last row, visibly return to the 02:05 napkin calculation. This callback is more important than a separate generic autograd animation.

**Do not interrupt:** the deep-network setup is essential context. The current site jumps directly to `AUTOGRAD`, which makes the later cost formulas float free of the model being counted.

### 66:45–72:10 — Optimizer state closes the model-memory ledger

- 66:45–68:55: AdaGrad is used as a small concrete optimizer because Assignment 1 uses Adam. Percy explicitly declines a detailed optimizer lecture; the purpose is to expose persistent state attached to every parameter.
- 68:55–71:10: parameters, activations, gradients, and optimizer state are totaled. BF16 model tensors use two-byte elements; optimizer accumulators conventionally use FP32 for stability. AdaGrad has one accumulator; Adam has first- and second-moment buffers.
- 71:10–72:10: large optimizer state mainly limits what fits in HBM; it is not necessarily the dominant arithmetic kernel. Transformer accounting is deferred to Assignment 1.
- The training-loop walkthrough is explicitly skipped in the recording.

**Best augmentation:** a per-parameter byte stack next to a separate activation formula:

| persistent tensor in the opening AdamW estimate | lecture dtype | bytes per parameter |
|---|---:|---:|
| parameter | BF16 | 2 |
| gradient | BF16 | 2 |
| Adam first moment | FP32 | 4 |
| Adam second moment | FP32 | 4 |
| subtotal |  | 12 |

Then show `activation memory = f(batch, sequence length, layers, hidden sizes)` outside the 12-byte stack. It was deliberately omitted from the opening 53B upper bound.

**Do not interrupt:** optimizer-family details should not become a full branch. Percy uses the family only to explain how many buffers exist.

### 72:10–76:20 — Two memory reductions with different mechanisms

- 72:10–73:25: gradient accumulation splits a large logical batch into microbatches, keeps the gradient buffer, and performs an optimizer update only after accumulating the intended batch.
- 73:25–75:35: training retains intermediate activations for backward, unlike layer-by-layer inference. Activation checkpointing stores selected boundaries and recomputes missing forward activations during backward: memory is traded for extra compute.
- 75:35–76:20: storing every layer gives `O(L)` saved activation memory and no recomputation; storing no intermediates can give `O(1)` memory but `O(L²)` recomputation in the lecture's simple schedule; spacing checkpoints around `√L` balances the terms.

**Best augmentation:** first place gradient accumulation and checkpointing in a comparison table so they are not conflated:

| technique | what is sliced/discarded | what persists | cost paid |
|---|---|---|---|
| gradient accumulation | logical batch becomes microbatches | gradient buffer | more sequential microbatch passes; not a reduction in mathematical training FLOPs |
| activation checkpointing | selected forward activations are discarded | checkpoints | missing forward work is recomputed during backward |

Then show checkpointing as a 40–60 second step-controlled sequence: ordinary forward shelf, discard policy, backward request, recomputation from nearest checkpoint, and final placement tradeoff.

**Do not interrupt:** activation premise → selective retention → recomputation → checkpoint frequency is one continuous story.

### 76:20–77:24 — Summary closes every open ledger

Percy returns to the exact lecture-level claims: all artifacts are tensors; einops helps reason about tensor operations; `6ND` comes from forward plus backward; roofline distinguishes memory- from compute-bound work; and gradient accumulation/checkpointing reduce activation-memory pressure so larger effective batches are possible.

**Best augmentation:** show the two opening napkin questions again with each once-unknown term linked to the point at which it was derived. Do not add a new recap card for every one of the 39 segment IDs.

## Live-delivery caveats the augmented version should preserve

The video is a live executable lecture, and several self-corrections are pedagogically useful. The augmented site should clarify them, not silently pretend the delivery was a polished paper.

- Around 02:10 Percy corrects the displayed accelerator label to H100. The current official source uses H100.
- Around 08:10 he says half-width arithmetic might be twice as fast and immediately qualifies that it depends. Do not present a universal 2× claim.
- Around 10:45 he skips/recompiles stale material before mixed precision. The augmentation should follow the final verbal explanation, not the temporary screen state.
- Around 17:20 GPU code is shown but not executed because the lecture runs on a laptop. Any measured GPU timing added by us must be labeled as an augmentation.
- Around 29:15 the sparse H100 marketing number is halved for dense BF16. A hardware table must name sparsity and dtype.
- Around 30:05 Percy notices that the displayed duration arithmetic appears to use one week while he had said two. Use explicit units rather than inheriting the live slip.
- Around 32:00 `2BDK` ignores the exact `D-1` addition correction; it is deliberately a scaling approximation.
- Around 44:00 perfect overlap makes total time `max(communication, computation)`; Percy says real overlap has overhead.
- Around 46:40 he corrects ReLU intensity from “half” to `0.25` FLOP/byte.
- Around 49:10 “GELU is the same speed as ReLU” is bounded to the isolated, bandwidth-bound model with comparable traffic. It is not a universal end-to-end claim.
- Around 66:15 `6ND` is described as a useful Transformer approximation only while long-context quadratic terms are not dominant.
- Around 69:55 Percy verbally corrects an on-screen memory formula: bytes must multiply `num_parameters`, not an already-byte-valued `parameter_memory`. The inspected source still contains this unit-confusing construction. Our table should use bytes/parameter and annotate the verbal correction.
- Around 73:15 the spoken phrase suggests gradient accumulation “saves compute.” The surrounding explanation and source make clear that its primary purpose here is lower activation memory for a large logical batch; it does not reduce the mathematical FLOPs of processing that batch.

## How the current 39-segment model fractures the lecture

The segment inventory is useful as an index and coverage checklist. It is not a faithful presentation structure. The current app renders every segment as a fresh two-column article with repeated labels (“Read the source claim,” “Inspect the augmentation”), a new learning goal, a three-beat sequence, pager controls, and often duplicated media. That repeated reset turns dependent steps into peers.

| Current segment group | Where it belongs in the real lecture | What is lost when each ID becomes a full row |
|---|---|---|
| `L02-TRAINING-TIME`, `L02-MODEL-CAPACITY` | 01:55–03:45, then callbacks at 66:10 and 69:15 | They look like completed mini-lessons rather than questions whose unexplained terms organize the lecture. |
| `L02-TENSOR-RANK`, `L02-BSHD`, `L02-FP32`, `L02-TENSOR-MEMORY` | 04:50–08:00 | The single storage ledger is reset four times. BSHD is a useful inline inspection, not a narrative branch. |
| `L02-FP16-UNDERFLOW`, `L02-BF16-RANGE`, `L02-MIXED-PRECISION`, `L02-AMP-CASTING`, `L02-FP8`, `L02-FP4` | 08:00–17:00 | The causal path from underflow to BF16 to mixed precision is broken; FP8/FP4 receive the same visual weight despite being a digression. |
| `L02-CPU-GPU` | 17:00–18:00 | Appropriate as a small bridge, but too thin for a full repeated card. |
| `L02-NAMED-DIMS`, `L02-EINSUM-MATMUL`, `L02-BATCHED-ATTENTION`, `L02-EINSUM-ELLIPSIS`, `L02-REDUCE`, `L02-REARRANGE` | 18:00–27:25 | One notation tutorial becomes six restarts. The learner loses the persistent tensor and color mapping that make named dimensions valuable. |
| `L02-FLOPS-UNITS`, `L02-MATMUL-2BDK`, `L02-MFU` | 27:25–40:30 | The same `B,D,K` example no longer visibly connects work, measured wall time, and utilization. |
| `L02-COMPUTE-MEMORY`, `L02-RELU-INTENSITY`, `L02-GELU-INTENSITY`, `L02-DOT-INTENSITY`, `L02-MATVEC-INTENSITY`, `L02-MATMUL-INTENSITY`, `L02-ROOFLINE` | 40:30–57:10 | This is the most damaging split. The operation ladder builds data reuse monotonically, and roofline is its final synthesis. Seven cards make the plot look like a new fact instead of the summary of the ledger. |
| `L02-AUTOGRAD`, `L02-BACKWARD-2X`, `L02-6ND` | 57:10–66:45 | The actual deep-network setup is missing from the segment list, so the gradient equations lack a running model. The callback to the opening estimate is also invisible. |
| `L02-OPTIMIZER-FAMILY`, `L02-OPTIMIZER-MEMORY`, `L02-TRAIN-LOOP` | 66:45–72:10 | Percy explicitly skips detailed optimizer mechanics and the training loop. The site promotes both to equal narrative status while the important 12-byte ledger is one among several cards. |
| `L02-GRAD-ACCUM`, `L02-ACTIVATION-MEMORY`, `L02-CHECKPOINTING`, `L02-CHECKPOINT-FREQUENCY` | 72:10–76:20 | Gradient accumulation and checkpointing need comparison because they reduce peak activation memory differently. The last three IDs are one continuous animation, not three repetitions of the same clip. |

The right column repeats even more than the segment count suggests. `renderedAugmentation()` injects a generic three-step sequence, a “shared clip” navigation block, the video, media metadata, caveat, and segment-specific notes for every mapped ID. Thus `ActivationCheckpointingTradeoff.mp4` is rendered again under all three checkpoint-related segments; the same happens for the two floating-point IDs. This is implementation-level duplication, not merely similar prose.

## Audit of the four current Lecture 2 clips

All four current MP4s are 854×480 at 15 fps and last between 9.3 and 9.9 seconds. Each tries to communicate multiple sequential claims, leaving roughly one second or less for many transitions and almost no reading dwell. None should remain unchanged in the primary lecture path.

| Clip | Current duration | Decision | Why | Rebuild target |
|---|---:|---|---|---|
| `FloatingPointRange.mp4` | 9.87 s | **Keep the concept and poster as reference; replace the primary MP4.** | It is aligned with Percy's range-versus-resolution argument, but bit layouts, log-scale range, `1e-8`, two results, and the takeaway arrive too quickly. The tiny 480p captions compete with the plot. It also stops before the mixed-precision consequence. | 25–40 s or step mode. Hold bit allocations, then range, then the `1e-8` experiment, then route BF16/FP32 roles. Use the HTML comparison table for dense static facts. |
| `EinopsRearrangeHeads.mp4` | 9.33 s | **Remove from the primary path until rebuilt.** | The clip compresses split, per-head transform, and join. Its adapted `w[4,3]` produces width 6, while the official source uses `w[4,4]` and returns width 8; the adaptation is not clearly framed as such. Some code is clipped at the left/bottom in the poster. | 30–45 s or a stepper tied to exact source code. Prefer the source's exact shapes; if adaptation is retained, label it before it happens. Keep one tensor and one color per named axis across all steps. |
| `RooflineModel.mp4` | 9.53 s | **Remove as a standalone introduction; rebuild as the end of the intensity sequence.** | ReLU, dot, matvec, GELU, the knee, matmul, axes, equations, and two advice boxes compete on a small frame. Labels overlap and the bottom annotation collides with axis material. “Compute-bound: use faster arithmetic” is too loose; reducing FLOPs or increasing effective compute throughput is the real lever. | 45–60 s, driven by the shared operation table. Add one point after its bytes/FLOPs derivation. Reveal the roof only after the learner has seen the ladder. |
| `ActivationCheckpointingTradeoff.mp4` | 9.60 s | **Keep the storyboard idea; replace and render only once.** | Four conceptual frames—ordinary activation shelf, selective checkpoints, backward recomputation, asymptotic schedule table—are compressed into under ten seconds. Small labels are unreadable at normal embedded size. Mapping the same file to three full rows multiplies the repetition. | 45–75 s or step mode. Begin with training-versus-inference retention, then show one backward request and exact recomputation path, then expose the placement table. One media block may carry three internal anchors. |

Highest-priority missing augmentations are not more isolated clips. They are the connective visuals the lecture itself relies on:

1. the opening napkin ledger with unresolved terms and later callbacks;
2. the `B,D,K → 2BDK → measured FLOP/s → MFU` strip;
3. the continuous operation-intensity table that culminates in roofline;
4. the deep-network forward/two-backward-matmul derivation that closes `6ND`;
5. the per-parameter memory stack that closes `2+2+4+4` and separates activation memory;
6. the gradient-accumulation versus checkpointing comparison.

## Recommended site factorization

The redesign should preserve Percy's lecture as the dominant object and attach augmentations to it.

### One lecture spine, eight continuous units

1. Motivation and the two open napkin questions
2. Tensor storage and precision
3. Einops as named bookkeeping
4. Matmul work, timing, and MFU
5. Data movement, arithmetic intensity, and roofline
6. Deep-network forward/backward and `6ND`
7. Optimizer/training memory and the 12-byte stack
8. Gradient accumulation and checkpointing

Keep all 39 stable IDs, but demote them to timecoded anchors inside these units. They remain useful for deep links, search, and Discussions without becoming 39 full page templates.

### Original lecture on the left, genuinely additive modules on the right

The left column should follow the official recording and executable source in their actual order:

- embedded official video with timecode links;
- compact timecoded outline/paraphrase;
- the exact source/code range currently being discussed;
- Percy's live caveat or self-correction when relevant.

The right column should appear only when it adds one of four things:

- **formula comparison:** same variables/units shown side by side;
- **demonstration:** motion or interaction reveals a state change that prose cannot;
- **table:** dense static comparisons such as dtypes or arithmetic intensity;
- **background:** one-sentence annotation plus an authoritative link.

Remove the generic “learning goal,” three repeated storyboard beats, “one clip, N source segments,” media provenance, and per-row slide-notes wrapper from the normal reading path. Put provenance and implementation links in one unobtrusive details panel per artifact.

### A single augmentation can span several anchors

An artifact should be placed once at its actual teaching location, with internal time/step anchors. Examples:

- the precision module owns `FP16-UNDERFLOW → BF16-RANGE → MIXED-PRECISION`;
- the named-dimension inspector owns all six einops anchors;
- the arithmetic-intensity ledger owns the six operation anchors and hands off to roofline;
- the checkpoint module owns `ACTIVATION-MEMORY → CHECKPOINTING → CHECKPOINT-FREQUENCY`.

This removes duplicated videos without losing addressability.

## Concrete insertion plan

| Lecture time | Augmentation | Format | Guardrail |
|---|---|---|---|
| 02:00 | Open 70B/15T training-time ledger | calculator + formula table | Show `6ND` and MFU as unexplained inputs; link forward to their derivations. |
| 02:45 | Open eight-H100 capacity ledger | byte stack | Mark it “upper bound; activations omitted.” Do not add unmentioned master weights to Percy's 12-byte estimate. |
| 05:00 | Tensor roles and lifetimes | compact table | Keep it visible as later rows are filled in. |
| 06:00 | FP32/FP16/BF16 comparison | HTML table | Separate range from resolution. |
| 08:50 | `1e-8` representability | slow stepper/animation | Allow a pause before revealing FP16 zero and BF16 nonzero. |
| 11:55 | Mixed-precision routing | diagram + table | Matmul/exp are examples, not a universal AMP whitelist. |
| 13:10 | FP8/NVFP4 context | collapsible background links | Preserve their digression-level weight. |
| 17:05 | CPU RAM ↔ GPU HBM | small diagram | No invented live benchmark. |
| 18:15 | Named-axis tensor inspector | stepper | Use one persistent tensor through einsum, reduce, and rearrange. |
| 27:35 | FLOPs versus FLOP/s | units table | Always show units, including seconds and utilization. |
| 31:35 | `2BDK` | slow formula animation | Count one output cell before scaling to `B×K`. |
| 35:05 | Safe GPU timing | code checklist | Highlight synchronization and repeated trials. |
| 37:05 | MFU | formula comparison | Denominator is dense, dtype-specific peak. Link forward to the memory explanation. |
| 40:40 | HBM move/compute/move | two-clock demonstration | Preserve `max(t_memory,t_compute)` as the ideal-overlap model plus caveat. |
| 42:30–55:00 | Operation intensity ledger | persistent table + optional micro-animation | Add operations in Percy's order; never reset the table. |
| 55:40 | Roofline | interactive plot | Derive points from the visible ledger; do not introduce all labels simultaneously. |
| 57:35 | Running deep network | schematic | Keep `B,D,L` visible until `6ND`. |
| 62:00 | Two backward matmuls | equation comparison + animation | Use the exact named dimensions and show which axis contracts. |
| 66:05 | Close training-time ledger | callback | Substitute the now-derived `6ND`, not a new standalone calculator. |
| 69:15 | Training memory ledger | byte stack + activation formula | Annotate the live code typo and separate bytes/parameter from batch-dependent activations. |
| 72:20 | Gradient accumulation | microbatch stepper | Emphasize peak activation memory, not reduced mathematical FLOPs. |
| 73:30 | Activation checkpointing | 45–75 s stepper/video | One artifact, three anchors, generous reading holds. |
| 76:20 | Close both napkin questions | summary ledger | Show the callbacks; do not enumerate 39 cards. |

## Background resources, attached where Percy uses them

These already appear in the official source and should be contextual links rather than a generic resource dump:

- DeepSeek V3.2 tensor index at 05:00: <https://huggingface.co/deepseek-ai/DeepSeek-V3.2?show_file_info=model.safetensors.index.json>
- Mixed Precision Training at 11:30: <https://arxiv.org/abs/1710.03740>
- PyTorch AMP at 12:30: <https://pytorch.org/docs/stable/amp.html>
- NVIDIA FP8 primer at 13:10: <https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/examples/fp8_primer.html>
- FP8 Formats for Deep Learning at 13:10: <https://arxiv.org/abs/2209.05433>
- NVIDIA NVFP4 introduction at 13:40: <https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/>
- Einops basics at 19:00: <https://einops.rocks/1-einops-basics/>
- NVIDIA H100 data sheet at 29:00: <https://resources.nvidia.com/en-us-gpu-resources/h100-datasheet-24306>
- JAX Scaling Book roofline chapter at 55:40: <https://jax-ml.github.io/scaling-book/roofline/>
- Transformer training-memory accounting at 71:45: <https://erees.dev/transformer-memory/>
- Transformer FLOPs accounting at 71:45: <https://www.adamcasson.com/posts/transformer-flops>

When a resource is not needed to understand Percy's next sentence, keep it collapsed. Background links should deepen the original lecture, not compete with it.

## Animation and review pipeline changes

The problem is not merely playback speed. The current scenes are authored as title-card summaries, while the lecture needs contextual demonstrations.

1. **Script from a lecture beat, not a segment title.** Every scene brief should name the preceding sentence, the misconception being resolved, and the next sentence it hands back to.
2. **Use motion only for change.** Bit-layout and operation tables should be HTML. Animate underflow, contraction, reuse, recomputation, and other state transitions.
3. **Design for silent reading.** A simple new label needs a 2.5–4 second hold; a formula/table state needs roughly 6–10 seconds or explicit step controls. A three- or four-stage concept should usually be 30–75 seconds, not 9 seconds.
4. **Prefer step mode for teaching-critical derivations.** Autoplay can be a preview, but the learner should control `next`, `previous`, and `replay this step` for einsum, roofline, and checkpointing.
5. **Render text at a web-embedded size.** Move from 480p proof renders to at least 720p delivery, with a minimum readable type size validated in the actual two-column layout.
6. **Reuse exact source variables and shapes.** Adaptations must be announced in-frame. Mathematical and unit audits should compare source, transcript, scene, poster, and supplement.
7. **Review contact sheets plus real-time playback.** A contact sheet finds clipping and density; it cannot detect insufficient reading dwell. Review at 1× on both desktop and mobile.
8. **Do not duplicate shared media.** Render once per continuous teaching unit; stable IDs point to internal steps.
9. **Acceptance test the handoff.** After an augmentation finishes, the original lecture's next sentence should make immediate sense. If it feels like returning from a different lesson, the augmentation is too detached.

## Bottom line

Lecture 2 should not be represented as 39 equally weighted explanations. It is a single resource ledger with two opening questions, a bottom-up derivation, and explicit callbacks. The enriched site will feel like a genuine supplement when the original video/source order remains intact and our additions appear exactly where they discharge a confusion: formula comparisons beside formulas, tables beside accounting, links beside digressions, and slow demonstrations only where something changes over time.
