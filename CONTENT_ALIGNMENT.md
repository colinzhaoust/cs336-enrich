# CS336 Lecture 1-3 content alignment audit

Audit date: 2026-08-31
Scope: 88 site segments, 12 segment-to-render mappings, and the Spring 2026 official materials for Lectures 1-3.
Constraint: this audit changes no site data, application code, Manim source, or media.

## 1. Evidence base and locator rules

The audit used the local shallow clone at:

- `/Users/colin/Desktop/cs336_enrich/official/lectures/`
- upstream repository: `https://github.com/stanford-cs336/lectures`
- audited commit: `8b59b50730766695c2ffedd1a79c50cd09b9eb91`
- Lecture 1: `lecture_01.py`, 762 lines, executable lecture, Percy Liang
- Lecture 2: `lecture_02.py`, 856 lines, executable lecture, Percy Liang
- Lecture 3: `lecture_03.pdf`, 67 PDF pages, Tatsu Hashimoto

The local official course schedule independently identifies Lecture 1 as "Overview, tokenization [Percy]", Lecture 2 as "PyTorch (einops), resource accounting ... [Percy]", and Lecture 3 as "Architectures, hyperparameters [Tatsu]" in `official/course-site/index.html` lines 672-697.

Line locators below are 1-based source lines at the audited commit. PDF page locators are 1-based PDF pages. The rendered PDF was checked visually; for this deck the instructional slide number and PDF page number agree.

Use commit-pinned source URLs when the left column claims an exact line range:

```text
https://github.com/stanford-cs336/lectures/blob/8b59b50730766695c2ffedd1a79c50cd09b9eb91/lecture_01.py#L65-L80
https://github.com/stanford-cs336/lectures/blob/8b59b50730766695c2ffedd1a79c50cd09b9eb91/lecture_02.py#L71-L78
https://github.com/stanford-cs336/lectures/blob/8b59b50730766695c2ffedd1a79c50cd09b9eb91/lecture_03.pdf
```

For the PDF, display the page number next to the deck link because GitHub's PDF viewer does not provide a dependable page fragment. Do not pretend that a deck link is already page-deep-linked.

### Verified official video metadata

The official YouTube descriptions contain no chapter timestamps. Therefore the site may show the verified video link and total duration, but it should not publish per-segment start/end times until a human has timestamped the recording.

| Lecture | Verified direct video | Verified duration | Segment timecodes |
| --- | --- | ---: | --- |
| 1 | `https://www.youtube.com/watch?v=JuoVZkPBiKk` | 1:19:22 | Not supplied by the official description; do not guess |
| 2 | `https://www.youtube.com/watch?v=kuYAsz7zspQ` | 1:17:25 | Not supplied by the official description; do not guess |
| 3 | `https://www.youtube.com/watch?v=lVynu4bo1rY` | 1:29:14 | Not supplied by the official description; do not guess |

The official playlist is `https://www.youtube.com/playlist?list=PLoROMvodv4rMqXOcazWaTUHhq-yembLCV`.

## 2. Highest-priority findings

### P0. Preserve the official source order in Lecture 2

The source introduces FP32 and its representation at lines 116-121, then uses FP32's four-byte element size in the tensor-memory calculation at lines 123-132. The site currently orders `L02-TENSOR-MEMORY` before `L02-FP32`, which asks the learner to use an element size before the dtype has been introduced.

Required order:

```text
L02-TENSOR-RANK -> L02-BSHD -> L02-FP32 -> L02-TENSOR-MEMORY
-> L02-FP16-UNDERFLOW -> L02-BF16-RANGE -> L02-MIXED-PRECISION
```

### Verified during the audit: the einops render was corrected

The official `einops_rearrange()` example at lines 261-276 starts with `x.shape == (seq=3, total_hidden=8)`, splits `total_hidden` into `heads=2` and `hidden1=4`, applies `w.shape == (4,4)` independently to every head, and rejoins to `total_hidden=8`.

The first render inspected during this audit had changed the example to an attention-specific B,S,H,D reshape and omitted the source's central `w` operation. The latest render has restored the 8-value, 2-head split and the intervening `einsum`, so the structural P0 is resolved. It deliberately uses `w[4,3]` and joins to 6 values to make the transform visible; that differs from Percy's executed `w[4,4]` example. Either switch to 4-by-4 for literal reproduction, or put "adapted width: hidden2=3" inside the frame. This remaining issue is P1.

### P0. Finish wiring the corrected KV-sharing argument to all three source segments

The official Lecture 3 sequence is:

```text
pages 58-60: full-sequence vs incremental attention, KV-cache traffic,
             poor incremental arithmetic intensity
page 61:     MQA as the extreme cache-reduction move
pages 62-63:GQA as the compromise and evidence about quality
```

The latest clip has been corrected to begin from incremental KV-cache traffic and now follows MQA -> GQA. However, its manifest still maps the video only to `L03-MQA` and `L03-GQA`; the prerequisite row `L03-KV-CACHE` does not receive the rendered explanation. The clip also summarizes the memory problem verbally rather than deriving the page-60 incremental arithmetic-intensity expression or closing on the page-63 quality evidence.

Required completion: map the continuous clip to `L03-KV-CACHE -> L03-MQA -> L03-GQA`, add one inspectable bytes/FLOPs beat from pages 58-60, and close with page 63's empirical caveat.

### P0/P1. Finish the current clips that still stop before the source's conclusion

- Extend `L01-BPE-PAIR-COUNT` + `L01-BPE-MERGE` through `L01-BPE-TRAIN-VS-USE`. The current clip accurately trains three merges, but the source immediately freezes those merges and encodes `"the quick brown fox"` at lines 714-720. Without that beat, the learner may not distinguish training statistics from deterministic inference.
- P1: the latest `L03-SLIDING-WINDOW` clip now ends with the correct statement that periodic full-attention layers restore long-range mixing. It still does not animate the layer stack, and the manifest maps it only to `L03-SLIDING-WINDOW`. Add a short local/local/local/full stack beat and share the asset with `L03-INTERLEAVED-ATTN`.

### P0. Stop calling the tokenizer comparison a measured Pareto frontier

`L01-TOKENIZER-PARETO` is a useful synthesis, but there is no single "tokenization unit" source anchor and its plotted points are schematic. Character tokenization is also dominated in the diagram, so the word "Pareto" can imply a mathematical frontier that the source did not establish.

Rename the visible concept to "Tokenizer tradeoff map" or "Vocabulary-size / sequence-length tradeoff". Cite the exact evidence bundle: lines 608-612, 643-648, 667-673, 686-695, and 705-708. Keep the schematic caveat next to the plot, not only in the expandable notes.

## 3. Recommended continuous-explanation boundaries

These boundaries follow the official teaching transitions. A shared video may be linked from several segment rows, but each row should open at a named chapter inside the clip or clearly highlight the current segment.

### Lecture 1

1. `L01-TOKENIZER-INTERFACE -> L01-TOKENIZER-QUIRKS -> L01-COMPRESSION-RATIO`
2. `L01-UNICODE-CODEPOINTS -> L01-CHAR-TOKENIZER` (same source function; do not make two redundant animations)
3. `L01-UTF8-BYTES -> L01-BYTE-TOKENIZER` (the rendered UTF-8 clip should be shared with the byte-tokenizer row)
4. `L01-WORD-TOKENIZER` stays separate, then the tradeoff map summarizes all baselines
5. `L01-BPE-PAIR-COUNT -> L01-BPE-MERGE -> L01-BPE-TRAIN-VS-USE`; use `L01-BPE-MERGE-CODE` as a code magnifier inside this sequence, then close with `L01-PRETOKENIZATION`

### Lecture 2

1. `L02-FP32 -> L02-TENSOR-MEMORY -> L02-FP16-UNDERFLOW -> L02-BF16-RANGE -> L02-MIXED-PRECISION -> L02-AMP-CASTING`
2. `L02-EINSUM-MATMUL -> L02-BATCHED-ATTENTION -> L02-EINSUM-ELLIPSIS`
3. `L02-REARRANGE` stays a focused source-code animation because it includes a split, transform, and join
4. `L02-FLOPS-UNITS -> L02-MATMUL-2BDK -> L02-MFU`
5. `L02-COMPUTE-MEMORY -> L02-RELU-INTENSITY -> L02-GELU-INTENSITY -> L02-DOT-INTENSITY -> L02-MATVEC-INTENSITY -> L02-MATMUL-INTENSITY -> L02-ROOFLINE`
6. Add the source's `deep_network()` setup (lines 559-600) immediately before `L02-AUTOGRAD`; then continue `L02-AUTOGRAD -> L02-BACKWARD-2X -> L02-6ND`
7. `L02-ACTIVATION-MEMORY -> L02-CHECKPOINTING -> L02-CHECKPOINT-FREQUENCY`

### Lecture 3

1. `L03-PRE-POST-NORM -> L03-DOUBLE-NORM`
2. `L03-RMSNORM -> L03-NORM-RUNTIME -> L03-NO-BIAS`
3. `L03-ACTIVATION-ZOO -> L03-GLU-GATE`; defer the width calculation in `L03-GLU-DIMENSION` until the hyperparameter unit
4. `L03-POSITION-FAMILIES -> L03-ROPE-RELATIVE -> L03-ROPE-FREQUENCIES -> L03-ROPE-CODE`
5. `L03-FF-RATIO -> L03-GLU-DIMENSION -> L03-FF-BASIN`
6. `L03-Z-LOSS -> L03-QK-NORM -> L03-SOFT-CAP`
7. `L03-KV-CACHE -> L03-MQA -> L03-GQA`
8. `L03-SLIDING-WINDOW -> L03-INTERLEAVED-ATTN`

Do not merge `L03-SERIAL-PARALLEL` into the GLU sequence: it changes block topology, not activation mechanics. Do not merge `L02-GRAD-ACCUM` into checkpoint scheduling: one changes the batch schedule, the other rematerializes layer activations.

## 4. Segment-by-segment audit

Legend: `[R]` currently has a rendered asset; `OK` is source-aligned; `P0` blocks a faithful teaching flow; `P1` is a worthwhile refinement.

### Lecture 1: Overview and tokenization

Official executable order is `why_this_course_exists()` -> `current_lm_landscape()` -> syllabus -> `tokenization()`. The site intentionally omits logistics and the executable-lecture demo; that omission is acceptable if the page says it is a curated learning path rather than a complete transcript.

| Segment | Verified source anchor | Finding and action |
| --- | --- | --- |
| `L01-ABSTRACTION-LADDER` | `lecture_01.py` 65-80 | OK. The source's three historical levels and "understanding via building" are preserved. |
| `L01-INDUSTRIAL-SCALE` | 82-100 | OK. Keep the distinction between public facts and scale-dependent examples. |
| `L01-KNOWLEDGE-TYPES` | 101-113 | OK. Mechanics/mindset transfer; intuitions only partially transfer. |
| `L01-EFFICIENCY-EQUATION` | 115-123 | OK. Do not turn the pedagogical equation into a measured physical law. |
| `L01-LM-TIMELINE` | 126-185 | P1: use the source's eras as section headers; avoid implying a single causal chain between every named model. |
| `L01-COURSE-STACK` | 235-251 | OK. This is a syllabus synthesis, not a literal end-to-end execution trace. |
| `L01-TOKENIZER-INTERFACE` | 256-263 and 579-588 | P1: cite both the abstract `Tokenizer` interface and the narrative introduction. Current locator omits the actual method signatures. |
| `L01-TOKENIZER-QUIRKS` | 591-612 | P1: leading spaces, word position, and digit grouping are source claims. Emoji/language comparisons are an augmentation based on the executable example, so label them "try this" rather than source claims. |
| `L01-COMPRESSION-RATIO` | 567-571 and 608-612 | OK. Preserve the exact definition `UTF-8 bytes / tokens`; show that attention length is a consequence, not part of the definition. |
| `L01-UNICODE-CODEPOINTS` | 627-648 | OK. Share its animation with the character-tokenizer row. |
| `L01-UTF8-BYTES` `[R]` | 651-673 | OK with P1 refinement: the source demonstrates `a` and `🌍`; `中` is a correct added example and should be labeled as such. Share this exact clip with `L01-BYTE-TOKENIZER`. |
| `L01-CHAR-TOKENIZER` | class 505-511; lesson 627-648 | P1: add the class anchor so the left column can show the actual `ord`/`chr` implementation. Do not duplicate the code-point animation. |
| `L01-BYTE-TOKENIZER` | class 514-524; lesson 651-673 | P1: add the class anchor and reuse the UTF-8 clip. The source says default byte compression ratio is exactly 1. |
| `L01-WORD-TOKENIZER` | 676-695 | OK. The source does not implement an actual vocabulary lookup, so mark the UNK collapse visualization as a faithful consequence rather than executed output. |
| `L01-TOKENIZER-PARETO` `[R]` | 608-612, 643-648, 667-673, 686-695, 705-708 | P0: replace the vague "tokenization unit" locator and visible "Pareto" claim. Plot is schematic; exact counts require a fixed tokenizer/corpus. |
| `L01-BPE-PAIR-COUNT` `[R]` | 729-758 | OK. The four initial pairs really tie at 2 for `"the cat in the hat"`; Python insertion order selects `(t,h)` first in this run. |
| `L01-BPE-MERGE` `[R]` | 705-713 and 729-750 | OK. The shown 18 -> 16 -> 14 -> 12 token counts and `th`, `the`, `the␠` merges match the executable code. |
| `L01-BPE-TRAIN-VS-USE` | tokenizer class 549-564; lesson 710-720 | P0: make this the final beat of the existing BPE clip. Show the learned ordered merge map being replayed on `"the quick brown fox"`. |
| `L01-BPE-MERGE-CODE` | 527-538; calls at 558 and 746 | OK as an inset, not a disconnected lesson. Its two-step/one-step cursor behavior supports both training and encoding. |
| `L01-PRETOKENIZATION` | 722-726 | P1: use as the BPE epilogue. The source lists efficient merge selection, special tokens, pretokenization, and speed; do not reduce it to regex boundaries alone. |

#### Lecture 1 best next animation work

1. Extend the existing BPE clip through train-vs-use and the production-boundary epilogue. This is more valuable than starting a new topic because it closes the source's central algorithm.
2. Replace the schematic tokenizer "Pareto" scene with an exact, same-string comparison. Show UTF-8 byte count, emitted token count, bytes/token, vocabulary size, and OOV count; retain a clearly separated schematic summary only after the exact table.

### Lecture 2: PyTorch, einops, and resource accounting

The official `main()` order is memory accounting -> GPU placement -> einops -> FLOP accounting -> arithmetic intensity -> deep network -> gradients -> optimizer -> train loop -> memory optimizations. The definition order in the Python file is not always the teaching order; the site should follow the calls in `main()`.

| Segment | Verified source anchor | Finding and action |
| --- | --- | --- |
| `L02-TRAINING-TIME` | `lecture_02.py` 71-78 | OK. Keep all assumptions visible: 70B, 15T, 6ND, 1024 H100s, dense BF16 peak, 50% MFU. |
| `L02-MODEL-CAPACITY` | 79-86 | OK. It is explicitly an upper bound because activations are omitted. |
| `L02-TENSOR-RANK` | 89-110 | OK. The B,S,H,D tensor follows the rank examples. |
| `L02-BSHD` | 105-110 | OK as an augmentation. Source defines the axes but does not perform the interactive slices. |
| `L02-TENSOR-MEMORY` | 123-132; helper 791-792 | P0: move after `L02-FP32`. Include `get_memory_usage = numel * element_size` as the source-code closure. |
| `L02-FP32` | 116-121 and source image `images/fp32.png` | P0: move before tensor-memory arithmetic. The memory example depends on FP32's four-byte element size. |
| `L02-FP16-UNDERFLOW` `[R]` | 134-143 | Numerically aligned. P0 copy fix: "no later optimizer step can recover that signal" is too absolute; say "that individual contribution is zero for this operation/step unless scaling or a higher-precision path prevents underflow." |
| `L02-BF16-RANGE` `[R]` | 145-152 | OK. The scene correctly separates dynamic range from local resolution and uses the smallest-subnormal scale only as an augmentation. |
| `L02-MIXED-PRECISION` | 154-162 | P1: use this as the closing consequence of the rendered float clip: BF16 parameters/activations/gradients, FP32 optimizer state. |
| `L02-AMP-CASTING` | 163-167 | OK. "matmuls, not exp" is the source's explicit example, not a universal whitelist. |
| `L02-FP8` | 168-173 | OK. Preserve the exact E4M3 and E5M2 ranges if numbers are animated. |
| `L02-FP4` | 174-181 | OK. Explicitly distinguish the 15-value local codebook from the per-block scale. |
| `L02-CPU-GPU` | 184-199 | P1: say tensors are on CPU "by default"; source also demonstrates direct allocation on the GPU. Transfer timing is an added measurement, not shown by this source. |
| `L02-NAMED-DIMS` | 214-220 | OK. This motivates einops before introducing syntax. |
| `L02-EINSUM-MATMUL` | 222-233 | P1: narrow the current 222-244 anchor; the simple matrix case ends at 233. |
| `L02-BATCHED-ATTENTION` | 234-244 | OK. It is the second, more complex einsum example. |
| `L02-EINSUM-ELLIPSIS` | 246-247 | OK, but it is a shorthand for a shared prefix, not arbitrary broadcasting without compatibility constraints. |
| `L02-REDUCE` | 250-259 | OK. Source executes `sum`; mean/max/min are listed alternatives and should not be presented as executed results unless the augmentation computes them. |
| `L02-REARRANGE` `[R]` | 261-276 | P1 in the latest render. The missing `w` operation is restored, but the animation uses adapted `w[4,3]` and output width 6 while the source executes `w[4,4]` and keeps width 8. Label the adaptation in-frame or use the exact source shape. Preserve the distinction that rearrange changes axis semantics while `einsum` changes values/width. |
| `L02-FLOPS-UNITS` | 279-297 | OK. Keep FLOPs and FLOP/s units visually distinct. |
| `L02-MATMUL-2BDK` | 298-325 | OK. One multiply plus one addition for each `(i,j,k)` gives the lecture's `2BDK` approximation. |
| `L02-MFU` | 327-335 | OK. It is measured FLOP/s divided by dtype-specific promised FLOP/s, ignoring communication/overhead in the stated definition. |
| `L02-COMPUTE-MEMORY` | 338-360 and 363-375 | P1: current anchor omits the `max(communication_time, computation_time)` overlap argument at 374-375. Add the second range. |
| `L02-RELU-INTENSITY` | 363-398 | OK. BF16 reads 2 bytes, performs about one comparison, writes 2 bytes, giving about 0.25 FLOP/byte. |
| `L02-GELU-INTENSITY` | 400-415 | OK. "No more wall time" is under the source's isolated-kernel and memory-bound assumptions; retain that boundary. |
| `L02-DOT-INTENSITY` | 418-431 | OK. The source includes the final 2-byte scalar write and obtains about 0.5 FLOP/byte. |
| `L02-MATVEC-INTENSITY` | 434-447 | OK. The source computes about 1 FLOP/byte. Repeated model-weight streaming during decoding is a useful explanation, but it is an augmentation. |
| `L02-MATMUL-INTENSITY` | 449-468 | OK. For 1024-square BF16 matmul the lecture formula is about `n/3`, or 341 FLOP/byte. |
| `L02-ROOFLINE` `[R]` | 471-481; H100 constants 350-351 | Numerically aligned: dense BF16 989.5 TFLOP/s and 3.35 TB/s give a knee near 295 FLOP/byte. P1: the clip is the culmination of the six preceding intensity segments, not a standalone introduction. Change "compute-bound: use faster arithmetic" to "reduce FLOPs or raise effective compute throughput." |
| `L02-AUTOGRAD` | 484-500 | P0 sequence gap: the official lecture calls `deep_network()` first. Add a short prelude from lines 559-600 before introducing forward/backward. |
| `L02-BACKWARD-2X` | 502-545 | OK. The two backward matmuls compute activation and weight gradients. |
| `L02-6ND` | 547-556 | OK with the source caveat: this is exact for the simplified MLP accounting and a useful approximation for Transformers at short context. |
| `L02-OPTIMIZER-FAMILY` | 602-615 | OK. The executed optimizer is AdaGrad; the family tree is explanatory text. |
| `L02-OPTIMIZER-MEMORY` | 633-646 | P1: visibly separate the executed AdaGrad example (one FP32 second-moment buffer, 4 bytes/parameter) from the Adam extension (two FP32 moments, 8 bytes/parameter). |
| `L02-TRAIN-LOOP` | 683-715 | OK. Preserve the exact step order and `zero_grad(set_to_none=True)` after the optimizer step. |
| `L02-GRAD-ACCUM` | 718-730 | OK. Keep it separate from checkpointing: it changes microbatch scheduling while the gradient buffer persists. |
| `L02-ACTIVATION-MEMORY` | 733-749 | OK. This is the premise for checkpointing and should share its continuous animation. |
| `L02-CHECKPOINTING` | 751-763 | OK. The source's checkpointed class begins at 776; add that code anchor when source code is shown. |
| `L02-CHECKPOINT-FREQUENCY` | 764-773 | OK. Preserve all three regimes and their different memory/recomputation costs. |

#### Lecture 2 best next animation work

1. Correct the FP16 supplement's over-absolute optimizer sentence, then extend the float clip through the mixed-precision consequence. This is a small edit with high conceptual value.
2. Build one continuous activation-memory/checkpointing animation for `L02-ACTIVATION-MEMORY -> L02-CHECKPOINTING -> L02-CHECKPOINT-FREQUENCY`. It completes the lecture's final resource tradeoff and currently has no rendered coverage.

The existing roofline clip is already the right visual summary. Its next improvement should be a short lead-in that derives each plotted point from bytes and FLOPs, not another independent roofline diagram.

### Lecture 3: Architecture and hyperparameters

The PDF has a strong sequence: modern baseline -> normalization -> activations -> block topology -> positions -> hyperparameters -> stability -> inference-efficient attention. The site largely preserves this, except that `L03-GLU-DIMENSION` pulls the page-38 width rule forward into the page-23 activation unit.

| Segment | Verified PDF pages | Finding and action |
| --- | ---: | --- |
| `L03-MODERN-TRANSFORMER` | 3-4 | OK. The four changes match page 4 exactly. |
| `L03-ARCH-MATRIX` | 5-9 | OK. Treat pages 5-8 as setup and page 9 as the actual matrix. |
| `L03-PRE-POST-NORM` `[R]` | 10-12 | Aligned. The formulas and untouched pre-norm identity path match the deck. P1: replace "normalization bottleneck" with the more precise "every direct residual route crosses LN." |
| `L03-DOUBLE-NORM` | 13 | P1: extend the current norm-topology clip to this page so learners can see exactly which extra norm is outside the residual stream. |
| `L03-RMSNORM` | 14-17 | OK. The core comparison is page 14; pages 15-17 motivate and validate runtime/performance. |
| `L03-NORM-RUNTIME` | 15-17 | OK. Share with RMSNorm; page 16 explicitly says FLOPs are not runtime because of data movement. |
| `L03-NO-BIAS` | 18-19 | P1 copy correction: the deck gives memory and optimization-stability reasons, not a measured claim that bias has "little modeling capacity." Phrase that as a course-design tradeoff or augmentation. |
| `L03-ACTIVATION-ZOO` | 20-23 | P1: current pages 20-21 cover the zoo plus ReLU/GELU; Swish appears through SwiGLU on page 23. If all three curves are animated, mark the Swish curve as an added comparison. |
| `L03-GLU-GATE` | 22-23 | OK. Make clear that `xV` is the multiplicative branch; a GLU gate is not necessarily a sigmoid gate. |
| `L03-GLU-DIMENSION` | 23 and 38 | P0 order: page 23 only notes the extra parameter and smaller dimensions; the explicit two-thirds/8-over-3 rule is page 38. Move this segment into the hyperparameter chapter beside `L03-FF-RATIO`. |
| `L03-SERIAL-PARALLEL` | 27-28 | OK. Keep separate from activation mechanics. The source's claimed benefit is shared LN and fusable matmuls if implemented correctly. |
| `L03-POSITION-FAMILIES` | 30 | OK. Four strategies are visibly compared on one page. |
| `L03-ROPE-RELATIVE` | 31-32 | OK. Preserve the target invariant that the inner product depends on `i-j`, then introduce rotation as the construction. |
| `L03-ROPE-FREQUENCIES` | 33-34 | OK. Pair coordinates first, then assign frequencies; do not show a single 2D rotation as the whole high-dimensional algorithm. |
| `L03-ROPE-CODE` | 35 | OK. The deck's key closure is applying RoPE at every attention operation, not once at input embedding. |
| `L03-FF-RATIO` | 37-41 | OK. This is the correct home for the standard 4x, gated 8/3x, T5 exception, and conclusion. |
| `L03-FF-BASIN` | 40 | OK, but share the FF-ratio animation; the figure is evidence inside that argument, not a new topic. |
| `L03-HEAD-RATIO` | 42-43 | OK. The equality is common but not required; preserve the table's overcomplete exceptions. |
| `L03-ASPECT-RATIO` | 44-46 | OK. The source discusses latency/parallelism as well as empirical quality; fixed-parameter morphing is an augmentation. |
| `L03-VOCAB-SIZE` | 47 | OK. Present 30-50K and 100-250K as observed ranges in this table, not universal rules. |
| `L03-REGULARIZATION` | 48-50 | OK. Weight decay is connected to optimization dynamics and the learning-rate schedule, not classical overfitting control. |
| `L03-Z-LOSS` | 52-54 | OK. Pages 52-53 are the softmax-stability premise; page 54 gives z-loss. |
| `L03-QK-NORM` | 55 | OK. Share with the stability sequence; the location is before the attention softmax. |
| `L03-SOFT-CAP` | 56 | OK. Retain the deck's caveat that soft-capping may have performance issues. |
| `L03-KV-CACHE` | 58-60 | Partial P0: the corrected MQA/GQA clip now opens with incremental cache traffic, but the manifest does not map that rendered asset to this row and the page-60 arithmetic-intensity derivation is still absent. |
| `L03-MQA` `[R]` | 61 | OK in the latest render. MQA is now presented first as the one-KV-head extreme. Share the clip with the prerequisite KV-cache row. |
| `L03-GQA` `[R]` | 62-63 | Mostly aligned in the latest render: GQA follows MQA as a capacity/efficiency compromise. P1: close on page-63 quality evidence, not only the cache ratio. |
| `L03-SLIDING-WINDOW` `[R]` | 64 | The 12-token width-4 count (78 -> 42 cells) is a correct added example. The deck itself does not supply these numbers, so label them "worked example." |
| `L03-INTERLEAVED-ATTN` | 65-66 | P1: the latest sliding-window clip states the periodic-full-attention conclusion, but does not animate the stack and is not mapped to this row. Add a local/local/local/full layer stack and share the asset. |

#### Lecture 3 best next animation work

1. Finish the now-corrected KV sequence with page-60 arithmetic intensity, page-63 evidence, and a manifest mapping for `L03-KV-CACHE`. This is the largest conceptual gain because it connects Lecture 2 resource accounting to Lecture 3 architecture.
2. Build a continuous RMSNorm/runtime/no-bias animation. If prioritizing current assets first, add the short interleaved-layer stack and map the sliding-window video to `L03-INTERLEAVED-ATTN`.

## 5. Left-column source-card specification

The current left column labels a paraphrase as "From the source claim" and derives a URL by parsing only the first `lines N-M` pattern. That is insufficient for multi-range concepts and PDFs. A later implementation should store explicit structured provenance instead of parsing display text.

Recommended data shape:

```js
source: {
  edition: "Spring 2026",
  instructor: "Percy Liang",
  kind: "executable-python",
  file: "lecture_02.py",
  commit: "8b59b50730766695c2ffedd1a79c50cd09b9eb91",
  anchors: [
    { label: "dtype premise", startLine: 116, endLine: 121 },
    { label: "memory calculation", startLine: 123, endLine: 132 }
  ],
  video: {
    url: "https://www.youtube.com/watch?v=kuYAsz7zspQ",
    duration: "1:17:25",
    start: null,
    end: null,
    timingStatus: "not-verified"
  }
}
```

For every source card:

1. Show edition, instructor, exact file/page, and audited commit.
2. Separate "source says" from "augmentation adds". This is especially important for multilingual tokenizer quirks, transfer-time measurement, repeated weight streaming, bias capacity, and worked numeric examples.
3. Support multiple anchors. `L01-TOKENIZER-INTERFACE`, the tokenizer tradeoff map, `L02-COMPUTE-MEMORY`, and `L03-GLU-DIMENSION` cannot be represented faithfully by the current single-range parser.
4. For executable lectures, optionally show the containing function name, such as `train_bpe()` or `einops_rearrange()`, beside the line range.
5. For PDFs, show `lecture_03.pdf, page N` as visible text even when the external viewer cannot deep-link.
6. Do not show a clock icon or timestamp field when `timingStatus` is not verified.

## 6. Acceptance checks for the next refinement pass

- Segment order matches the official execution/deck order unless an explicit "enriched detour" label explains the change.
- Every rendered clip begins with the prerequisite named in this audit and ends with the source's immediate conclusion.
- A shared clip's internal chapters appear in the same order as its source segments.
- Exact source examples use the official shapes, strings, formulas, and code before introducing an adapted example.
- Schematic charts say "schematic" inside the visual frame.
- Added examples say "worked example" or "augmentation" inside the left/right comparison, not only in hidden notes.
- No segment-level YouTube timestamp is published until independently checked against the official video.
