# Lecture-first architecture for CS336 Enrich

## Decision

The official CS336 lecture is the only narrative spine. The official recording,
source program or slide deck, lecturer's transitions, live caveats, evidence,
and recap determine the order and visible units of the site.

Augmentations are optional **timed slots** attached to that spine. They may
clarify a formula, make a comparison inspectable, repair a failed or skipped
demonstration, or provide authoritative background. They may not become a
replacement syllabus, reset the learner after every legacy segment ID, or
interrupt an atomic claim/evidence/caveat chain.

This document is grounded in the copied teaching maps:

- [Lecture 1 teaching map](lecture_01_teaching_map.md)
- [Lecture 2 teaching map](lecture_02_teaching_map.md)
- [Lecture 3 teaching map](lecture_03_teaching_map.md)

The executable canonical model is
[`data/lecture_runs.js`](../data/lecture_runs.js). The current page still uses
`data/lectures.js`; switching renderers is a later implementation step.

## Page model

A lecture page should be read from top to bottom in the official order:

1. A sticky official video and current source/slide range establish the primary
   object.
2. One visible `lectureRun` represents a continuous teaching move, not one
   concept card.
3. The source rail contains the lecturer's question, paraphrased original
   explanation, current source references, live correction, evidence, and
   recap.
4. An augmentation slot is shown only at its recorded insertion point and is
   collapsed by default unless it repairs a demonstrated gap.
5. A shared artifact mounts once. Multiple stable IDs point to the run or an
   internal step; they do not duplicate video, notes, provenance, or takeaway.
6. Discussions retain the legacy ID as their stable term. The comment prompt
   should ask which source sentence/slide the augmentation helped or obscured.

The model deliberately has no generic per-anchor learning goal, three repeated
storyboard beats, repeated media metadata, or permanent second-course right
rail. Provenance belongs to one details panel per artifact.

## Executable schema

Each lecture declares the verified recording, official source, duration,
lecturer, lecture-wide thesis, explicitly restored source beats, and ordered
runs. Each run contains:

- `startSeconds` / `endSeconds`: continuous position in the official video;
- `professorIntent`: the question or teaching move being performed;
- `sourceRefs`: exact source lines or slide pages plus video interval;
- `originalSummary`: a paraphrase of the original content, not our rewrite;
- `anchors[]`: hidden/internal legacy IDs used for deep links, search, and
  Discussion terms;
- `uninterruptedRanges[]`: claim chains where no augmentation may be mounted;
- `augmentationSlots[]`: optional, timed, reasoned additions;
- `restoredEvidence`: transitions, evidence slides, and recaps previously
  absent from the 88-card inventory.

Allowed slot types are intentionally small:

| Type | Use | Avoid |
| --- | --- | --- |
| `formula-comparison` | Align formulas, variables, units, assumptions, and caveats | Animating a static equation merely to create motion |
| `table` | Compare categorical evidence or maintain a persistent ledger | Repeating the lecturer's prose in rows |
| `background-link` | Attach an authoritative source at the moment it is invoked | A generic resource dump |
| `interactive-demo` | Repair a failed/skipped interaction or allow hypothesis testing | A passive copy of an existing live demo |
| `slow-manim` | Reveal one causal state/topology change over time | Multi-concept title-card micro-lectures |

Every slot has a non-empty `pedagogicalReason`, is optional, and is placed at
a safe boundary rather than strictly inside an uninterrupted range.

## Continuous run inventory

The 88 previous cards become 28 visible runs: Lecture 1 has 10, Lecture 2 has
8, and Lecture 3 has 10.

| Run | Time range | Visible teaching move | Internal anchors |
| --- | --- | --- | ---: |
| L01-R01 | 00:04–03:23 | Welcome and the learning contract | 0 |
| L01-R02 | 03:23–11:38 | Why this course exists | 4 |
| L01-R03 | 11:38–20:03 | History, openness, and executable lectures | 1 |
| L01-R04 | 20:03–27:16 | Logistics: watching is not doing | 0 |
| L01-R05 | 27:16–64:46 | Five assignments, one efficiency lens | 1 |
| L01-R06 | 64:46–65:06 | The real unit boundary | 0 |
| L01-R07 | 65:06–68:29 | Tokenizer interface, quirks, and compression | 3 |
| L01-R08 | 68:29–72:00 | Three baseline tokenizers and their failure modes | 5 |
| L01-R09 | 72:00–75:22 | BPE: from bytes to learned chunks | 3 |
| L01-R10 | 75:22–79:22 | Use the tokenizer, then return to invariants | 3 |
| L02-R01 | 00:05–04:50 | Open the compute and memory ledgers | 2 |
| L02-R02 | 04:50–18:00 | Tensor storage and precision | 11 |
| L02-R03 | 18:00–27:25 | Einops as named bookkeeping | 6 |
| L02-R04 | 27:25–40:30 | From 2BDK to measured MFU | 3 |
| L02-R05 | 40:30–57:10 | Data movement to roofline | 7 |
| L02-R06 | 57:10–66:45 | A deep network closes 6ND | 3 |
| L02-R07 | 66:45–72:10 | Optimizer state closes the memory ledger | 3 |
| L02-R08 | 72:10–77:25 | Two activation-memory trades and the final callback | 4 |
| L03-R01 | 00:05–07:41 | Why these modern defaults? | 2 |
| L03-R02 | 07:41–14:15 | Keep the residual stream clean | 2 |
| L03-R03 | 14:15–20:15 | Normalization as systems design | 3 |
| L03-R04 | 20:15–31:11 | Gated FFNs, then block topology | 3 |
| L03-R05 | 31:11–43:40 | Position strategies to RoPE | 4 |
| L03-R06 | 43:40–65:10 | Which hyperparameters actually matter? | 7 |
| L03-R07 | 65:10–74:06 | Stability: two softmax danger zones | 3 |
| L03-R08 | 74:06–79:45 | Why decoding changes attention | 1 |
| L03-R09 | 79:45–85:08 | MQA to GQA: an efficiency-quality knob | 2 |
| L03-R10 | 85:08–89:10 | Long-context hybrids and the return to the matrix | 2 |

## Restored source evidence and recaps

The run data makes omissions explicit rather than inventing replacement IDs.
The most important restorations are:

- **Lecture 1:** staff/update continuity, logistics and learning contract, the
  complete five-assignment question sequence, the 64:46 unit boundary, internal
  versus external evaluation, the data pipeline, the assignment/resource
  synthesis, and the final tokenizer-free/adaptive-computation callback.
- **Lecture 2:** the Marin callback, both unresolved opening ledgers, the
  running deep-network setup, callbacks that derive and close `6ND` and the
  12-byte stack, live unit corrections/caveats, and the final recap that closes
  both ledgers.
- **Lecture 3:** slides 1–2 (survey method), 24–26 (GLU controlled evidence and
  recap), 29 (architecture recap), 36 (hyperparameter questions), 51
  (hyperparameter recap), 57 (efficient-attention roadmap), and 67 (return to
  common versus variable axes).

These are source-rail beats, not new top-level segments. This distinction is
what makes the site an augmentation of the original lecture.

## Existing artifact disposition

`keep` means the artifact remains suitable in its stated supporting role;
`rebuild` keeps the teaching target but replaces the delivery; `remove`
means it should leave the primary path until a different form is built.

| Existing artifact | Status | Intended next form |
| --- | --- | --- |
| L01 UTF-8 bytes MP4 | rebuild | 22–30 s optional byte-expansion replay |
| L01 BPE pair-count/merge MP4 | rebuild | Prefer a user-controlled state stepper after the official trace |
| L01 BPE train-vs-use MP4 | rebuild | 24–35 s or click-stepped skipped-code repair |
| L01 tokenizer Pareto MP4 | remove | Renamed, reproducible static toy comparison |
| L02 floating-point MP4 | rebuild | Table plus 25–40 s/step-mode underflow demonstration |
| L02 einops MP4 | remove | Exact-source shape inspector; rebuild before returning |
| L02 roofline MP4 | remove | Persistent operation ledger culminating in an interactive plot |
| L02 checkpointing MP4 | rebuild | 45–75 s/step-mode, mounted once for three anchors |
| L03 pre/post norm MP4 | rebuild | 25–35 s topology trace with source evidence beside it |
| L03 runtime-norm MP4 | remove | Formula table + focused read/reduce/write interaction |
| L03 KV-sharing MP4 | rebuild | Cache accounting first, then 45–70 s MHA→MQA→GQA morph and evidence |
| L03 sliding-window MP4 | remove | Window slider plus local/local/local/full propagation interaction |
| Existing posters and Manim source | keep | Archival storyboard/source reference, not proof of publish quality |

A future render gate should require source pages and video interval, one claim,
one evidence boundary, one caveat, 720p minimum (1080p target), 30 fps,
3–5 second key-state dwell, 1× real-time review, and embedded-size typography QA.

## Stable ID to run mapping

All 88 IDs remain addressable. They are not visible page units. This mapping is
also exported as `CS336_LECTURE_RUNS.anchorToRun`.

| Stable ID / Discussion term | Run | Continuous teaching move |
| --- | --- | --- |
| `L01-ABSTRACTION-LADDER` | `L01-R02` | Why this course exists |
| `L01-INDUSTRIAL-SCALE` | `L01-R02` | Why this course exists |
| `L01-KNOWLEDGE-TYPES` | `L01-R02` | Why this course exists |
| `L01-EFFICIENCY-EQUATION` | `L01-R02` | Why this course exists |
| `L01-LM-TIMELINE` | `L01-R03` | History, openness, and executable lectures |
| `L01-COURSE-STACK` | `L01-R05` | Five assignments, one efficiency lens |
| `L01-TOKENIZER-INTERFACE` | `L01-R07` | Tokenizer interface, quirks, and compression |
| `L01-TOKENIZER-QUIRKS` | `L01-R07` | Tokenizer interface, quirks, and compression |
| `L01-COMPRESSION-RATIO` | `L01-R07` | Tokenizer interface, quirks, and compression |
| `L01-UNICODE-CODEPOINTS` | `L01-R08` | Three baseline tokenizers and their failure modes |
| `L01-CHAR-TOKENIZER` | `L01-R08` | Three baseline tokenizers and their failure modes |
| `L01-UTF8-BYTES` | `L01-R08` | Three baseline tokenizers and their failure modes |
| `L01-BYTE-TOKENIZER` | `L01-R08` | Three baseline tokenizers and their failure modes |
| `L01-WORD-TOKENIZER` | `L01-R08` | Three baseline tokenizers and their failure modes |
| `L01-BPE-PAIR-COUNT` | `L01-R09` | BPE: from bytes to learned chunks |
| `L01-BPE-MERGE` | `L01-R09` | BPE: from bytes to learned chunks |
| `L01-BPE-MERGE-CODE` | `L01-R09` | BPE: from bytes to learned chunks |
| `L01-BPE-TRAIN-VS-USE` | `L01-R10` | Use the tokenizer, then return to invariants |
| `L01-PRETOKENIZATION` | `L01-R10` | Use the tokenizer, then return to invariants |
| `L01-TOKENIZER-PARETO` | `L01-R10` | Use the tokenizer, then return to invariants |
| `L02-TRAINING-TIME` | `L02-R01` | Open the compute and memory ledgers |
| `L02-MODEL-CAPACITY` | `L02-R01` | Open the compute and memory ledgers |
| `L02-TENSOR-RANK` | `L02-R02` | Tensor storage and precision |
| `L02-BSHD` | `L02-R02` | Tensor storage and precision |
| `L02-FP32` | `L02-R02` | Tensor storage and precision |
| `L02-TENSOR-MEMORY` | `L02-R02` | Tensor storage and precision |
| `L02-FP16-UNDERFLOW` | `L02-R02` | Tensor storage and precision |
| `L02-BF16-RANGE` | `L02-R02` | Tensor storage and precision |
| `L02-MIXED-PRECISION` | `L02-R02` | Tensor storage and precision |
| `L02-AMP-CASTING` | `L02-R02` | Tensor storage and precision |
| `L02-FP8` | `L02-R02` | Tensor storage and precision |
| `L02-FP4` | `L02-R02` | Tensor storage and precision |
| `L02-CPU-GPU` | `L02-R02` | Tensor storage and precision |
| `L02-NAMED-DIMS` | `L02-R03` | Einops as named bookkeeping |
| `L02-EINSUM-MATMUL` | `L02-R03` | Einops as named bookkeeping |
| `L02-BATCHED-ATTENTION` | `L02-R03` | Einops as named bookkeeping |
| `L02-EINSUM-ELLIPSIS` | `L02-R03` | Einops as named bookkeeping |
| `L02-REDUCE` | `L02-R03` | Einops as named bookkeeping |
| `L02-REARRANGE` | `L02-R03` | Einops as named bookkeeping |
| `L02-FLOPS-UNITS` | `L02-R04` | From 2BDK to measured MFU |
| `L02-MATMUL-2BDK` | `L02-R04` | From 2BDK to measured MFU |
| `L02-MFU` | `L02-R04` | From 2BDK to measured MFU |
| `L02-COMPUTE-MEMORY` | `L02-R05` | Data movement to roofline |
| `L02-RELU-INTENSITY` | `L02-R05` | Data movement to roofline |
| `L02-GELU-INTENSITY` | `L02-R05` | Data movement to roofline |
| `L02-DOT-INTENSITY` | `L02-R05` | Data movement to roofline |
| `L02-MATVEC-INTENSITY` | `L02-R05` | Data movement to roofline |
| `L02-MATMUL-INTENSITY` | `L02-R05` | Data movement to roofline |
| `L02-ROOFLINE` | `L02-R05` | Data movement to roofline |
| `L02-AUTOGRAD` | `L02-R06` | A deep network closes 6ND |
| `L02-BACKWARD-2X` | `L02-R06` | A deep network closes 6ND |
| `L02-6ND` | `L02-R06` | A deep network closes 6ND |
| `L02-OPTIMIZER-FAMILY` | `L02-R07` | Optimizer state closes the memory ledger |
| `L02-OPTIMIZER-MEMORY` | `L02-R07` | Optimizer state closes the memory ledger |
| `L02-TRAIN-LOOP` | `L02-R07` | Optimizer state closes the memory ledger |
| `L02-GRAD-ACCUM` | `L02-R08` | Two activation-memory trades and the final callback |
| `L02-ACTIVATION-MEMORY` | `L02-R08` | Two activation-memory trades and the final callback |
| `L02-CHECKPOINTING` | `L02-R08` | Two activation-memory trades and the final callback |
| `L02-CHECKPOINT-FREQUENCY` | `L02-R08` | Two activation-memory trades and the final callback |
| `L03-MODERN-TRANSFORMER` | `L03-R01` | Why these modern defaults? |
| `L03-ARCH-MATRIX` | `L03-R01` | Why these modern defaults? |
| `L03-PRE-POST-NORM` | `L03-R02` | Keep the residual stream clean |
| `L03-DOUBLE-NORM` | `L03-R02` | Keep the residual stream clean |
| `L03-RMSNORM` | `L03-R03` | Normalization as systems design |
| `L03-NORM-RUNTIME` | `L03-R03` | Normalization as systems design |
| `L03-NO-BIAS` | `L03-R03` | Normalization as systems design |
| `L03-ACTIVATION-ZOO` | `L03-R04` | Gated FFNs, then block topology |
| `L03-GLU-GATE` | `L03-R04` | Gated FFNs, then block topology |
| `L03-SERIAL-PARALLEL` | `L03-R04` | Gated FFNs, then block topology |
| `L03-POSITION-FAMILIES` | `L03-R05` | Position strategies to RoPE |
| `L03-ROPE-RELATIVE` | `L03-R05` | Position strategies to RoPE |
| `L03-ROPE-FREQUENCIES` | `L03-R05` | Position strategies to RoPE |
| `L03-ROPE-CODE` | `L03-R05` | Position strategies to RoPE |
| `L03-FF-RATIO` | `L03-R06` | Which hyperparameters actually matter? |
| `L03-GLU-DIMENSION` | `L03-R06` | Which hyperparameters actually matter? |
| `L03-FF-BASIN` | `L03-R06` | Which hyperparameters actually matter? |
| `L03-HEAD-RATIO` | `L03-R06` | Which hyperparameters actually matter? |
| `L03-ASPECT-RATIO` | `L03-R06` | Which hyperparameters actually matter? |
| `L03-VOCAB-SIZE` | `L03-R06` | Which hyperparameters actually matter? |
| `L03-REGULARIZATION` | `L03-R06` | Which hyperparameters actually matter? |
| `L03-Z-LOSS` | `L03-R07` | Stability: two softmax danger zones |
| `L03-QK-NORM` | `L03-R07` | Stability: two softmax danger zones |
| `L03-SOFT-CAP` | `L03-R07` | Stability: two softmax danger zones |
| `L03-KV-CACHE` | `L03-R08` | Why decoding changes attention |
| `L03-MQA` | `L03-R09` | MQA to GQA: an efficiency-quality knob |
| `L03-GQA` | `L03-R09` | MQA to GQA: an efficiency-quality knob |
| `L03-SLIDING-WINDOW` | `L03-R10` | Long-context hybrids and the return to the matrix |
| `L03-INTERLEAVED-ATTN` | `L03-R10` | Long-context hybrids and the return to the matrix |

## Validation contract

The architecture is valid only when:

- the 88 IDs parsed from `data/lectures.js` are unique and match the mapping
  exactly, with no missing or extra ID;
- runs are ordered and each range fits inside the verified recording duration;
- every slot uses an allowed type, has a pedagogical reason, lies inside its
  run, and does not lie strictly inside an uninterrupted range;
- every referenced existing artifact status is one of `keep`, `rebuild`,
  or `remove`;
- the three video IDs resolve to the Stanford Online Lecture 1, Lecture 2, and
  Lecture 3 recordings named in the teaching maps.

Run `node research/check_lecture_runs.mjs` from `cs336_site` to enforce the
local structural checks. Video identity was additionally checked against the
public YouTube page titles on 2026-08-31; the teaching maps record the caption,
duration, schedule, and source evidence used for the content reconstruction.

