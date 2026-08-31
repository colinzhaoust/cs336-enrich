# CS336 augmentation animation pipeline v2

Status: design gate for the next rebuild. This document changes no published
media and no site behavior.

## 1. The reset

The purpose of an augmentation is to make one moment in the official lecture
easier to understand, then return the learner to the professor's next move. It
is not a compressed second lecture and it is not a video-shaped summary card.

The source of truth for this reset is:

- `lecture_01_teaching_map.md`
- `lecture_02_teaching_map.md`
- `lecture_03_teaching_map.md`
- the public caption intervals and official source/deck cited by those maps;
- the 12 existing scenes under `scenes/cs336/lecture_01` through
  `scenes/cs336/lecture_03`.

The three maps reconstruct different teaching grammars:

- Lecture 1 is one efficiency argument, followed by a late, real unit boundary
  into tokenization. Percy's executable BPE trace is already the main teaching
  object.
- Lecture 2 is one resource ledger. Definitions open entries that are paid off
  later by `2BDK`, MFU, roofline, `6ND`, and the memory stack.
- Lecture 3 is an empirical architecture survey. Its recurring unit is
  `question -> comparison -> evidence -> caveat -> recap`.

V2 must preserve those grammars. Stable segment IDs remain anchors and comment
targets; they are not instructions to manufacture one clip per ID.

## 2. Why v1 produced fast, crowded, context-free clips

### 2.1 A proof render became the delivery format

All 12 current MP4s are H.264/YUV420p at `854x480`, `15 fps`. Their measured
durations are 6.60-12.87 seconds. The manifests explicitly call Lecture 1's
render quality `low`, and several duration targets are 6-13 seconds. Those are
useful animatic settings, but they were promoted to publication settings.

Fast-start has already been applied (`moov` precedes `mdat`) and is not the
problem. Encoding hygiene cannot compensate for unreadable content.

### 2.2 The authoring unit was “segment,” not “teaching gap”

The site has 88 stable IDs. V1 treated many of them as independent mini-lessons
with a title, goal, three beats, source line, takeaway, notes, and discussion.
Shared assets were then mounted under multiple IDs. This caused both kinds of
repetition the user noticed:

- the page repeats its right-rail scaffold;
- the video repeats the title, segment label, provenance, and takeaway that the
  surrounding page already states.

In the real lectures, several IDs often belong to one indivisible argument.
For example, pair counting and merging are phases of one BPE training loop;
ReLU through matmul are rows in one arithmetic-intensity ladder; KV cache,
MQA, GQA, and the quality evidence are one systems-to-evidence argument.

### 2.3 “Show every fact” replaced a single teaching question

The scenes contain many small labels (`font_size` 13-20 is common), while most
transitions last 0.35-1.25 seconds and many final waits last only 0.25-1.3
seconds. Examples:

- the 7-second train-versus-use scene shows the corpus, three learned rules,
  freezing, held-out bytes, three applications, round trip, and takeaway;
- the 9.53-second roofline scene introduces axes, both roof segments, knee,
  several operations, regions, advice, and the equation;
- the 10.07-second normalization scene crosses formula, runtime/data movement,
  no-bias, and an empirical caveat;
- the 10.53-second KV scene starts at a head diagram and reaches MQA and GQA,
  without first establishing decode traffic or ending on the lecture's
  evidence.

This is not repaired by globally slowing playback. Some assets have the wrong
medium and some have several lessons inside one file.

### 2.4 The visual opened without the professor's question and closed without
the handoff

Scene chrome encodes a segment title and source reference, but not the previous
lecture sentence, the precise gap, the evidence boundary, or the next lecture
move. Consequently a clip can be individually true and still interrupt the
lecture's logic. Roofline looks like a new topic instead of the payoff of the
operation ledger; GQA looks like a head-count trick instead of a response to
decode memory traffic.

### 2.5 Review optimized for coverage and posters

The old pipeline successfully checked that clips existed, rendered, and fit a
frame. It did not require a real-time 1x watch beside the relevant transcript,
a minimum settled-state dwell, a mobile embedded-size reading test, or a check
that the professor's next sentence still made sense. Contact sheets catch
clipping and density; they do not catch a formula disappearing before it can
be read.

## 3. Choose the medium before storyboarding

Use the lightest additive form. Motion is justified only by change over time.

| Learner need | Default medium | Not Manim because... |
| --- | --- | --- |
| Compare definitions, assumptions, units, model families, or evidence | HTML table | Simultaneous scanning and revisiting matter more than time |
| Inspect one equation and its caveat | Formula card with aligned notation | A static relation should not be forced through a timeline |
| Vary a parameter and observe a result | Browser interactive | The learner needs control and counterfactuals |
| Replay discrete program state | Stepper with previous/next | The learner should predict and revisit an exact state |
| See one topology, flow, contraction, reuse, or recomputation evolve | Manim or canvas motion | Motion exposes the causal/spatial relation |
| Follow a claim, derivation, and evidence without the source lecture | Narrated micro-lecture | It is a standalone explanation, not an inline silent loop |
| Read papers, docs, or historical context | One-line annotation + link | Background should remain a detour |

An inline illustration loop may be 8-15 seconds only when it has one spatial
relation, almost no prose, and can loop or scrub cleanly. A teaching-critical
asset follows the 25-60 second standard below or becomes a stepper. More than
60 seconds is a warning that the artifact is attempting to replace the
lecture.

## 4. Mandatory context card

No storyboard begins until this card is reviewed. It is authoring metadata and
site chrome, not text to repeat inside the video.

```yaml
artifactId: L02-ROOFLINE-PAYOFF-V2
lecture: 2
source:
  videoUrl: https://www.youtube.com/watch?v=kuYAsz7zspQ
  interval: 40:30-57:10
  insertionAfter: 55:10
  sourcePages: []
  sourceLines: [lecture_02.py:<verified range before production>]
context:
  professorBefore: "Paraphrase of the immediately preceding move"
  professorAfter: "Paraphrase of the sentence to which we return"
  uninterruptedRun: "ReLU -> GELU -> dot -> matvec -> matmul -> roofline"
teachingQuestion: "One question, ending in a question mark"
additiveClaim: "One sentence describing what the source leaves hard to see"
evidenceBoundary:
  lectureEvidence: "What Percy/Tatsu actually shows"
  augmentationEvidence: "Any calculation or toy example we add"
caveat: "The local condition under which the claim holds"
medium: interactive-stepper
audio: silent
expectedLearnerTimeSeconds: 55
```

Gate rules:

1. `teachingQuestion` contains exactly one question.
2. `additiveClaim` must say what is newly visible, not summarize the source.
3. Every number is tagged as lecture value, reproduced calculation, cited
   evidence, or declared toy value.
4. The five seconds before and after the insertion are reviewed.
5. If `professorAfter` feels like a topic switch after the artifact, reject or
   shorten the artifact.

## 5. Delivery standard

### 5.1 Content and pacing

- One artifact answers one teaching question.
- Target duration: 25-60 seconds at 1x for linear video/animatic. Prefer a
  user-controlled stepper for exact derivations and code traces.
- One causal change per beat. Do not move an object while asking the learner to
  read a new formula on it.
- After motion settles, use these minimum dwells:

  | State | Minimum dwell at 1x |
  | --- | ---: |
  | One short label or changed number | 3 s |
  | Prediction pause before revealing an answer | 3-4 s |
  | One principal formula | 6 s |
  | Two formulas or two columns being compared | 8 s |
  | Dense evidence/table state | 10 s or convert to HTML/step mode |
  | Final state needed for the lecture handoff | 5 s |

- No more than seven visible labels at one time. A label should normally be no
  more than eight words. Longer explanation belongs in the adjacent caption
  or transcript.
- A beat budget includes entrance, settle, dwell, and exit separately. A
  0.7-second `FadeIn` is not reading time.

### 5.2 Picture and text

- Delivery master: `1920x1080`, square pixels, `30 fps`, H.264, YUV420p.
- Internal animatic: `854x480`, `15 fps` is permitted, but its file path and
  manifest status must contain `animatic`; it must never be mounted as the
  production source.
- Keep essential content inside a 72 px horizontal and 54 px vertical safe
  area on the 1080p master. Keep captions out of the browser controls area.
- At a 640 px desktop embed, essential labels must measure at least 18 CSS px.
  At a 360 px mobile embed, they must measure at least 16 CSS px. This implies
  roughly 54 and 86 source pixels respectively; design for the stricter mobile
  case rather than trusting Manim's `font_size` value.
- A principal formula should have a rendered glyph height of roughly 96 source
  pixels and never share the frame with a dense paragraph.
- Code in video is limited to two focused lines. Longer code is HTML with line
  highlighting synchronized to the visual.
- The final browser test, not a Manim font constant, decides readability.

### 5.3 CS336 visual language

- Preserve the source-adjacent light background, cardinal accent, restrained
  typography, and engineering-diagram tone.
- The site owns artifact title, stable IDs, source link, and discussion. Do not
  burn them into every frame.
- Use color as a persistent variable/axis identity, not decoration. The same
  `B`, `D`, `K`, token, or KV group keeps the same color throughout a run.
- Do not imitate a 3Blue1Brown camera style. Prefer stable frames, local
  transformations, and a visible accounting ledger.

### 5.4 Source, audio, caption, and transcript policy

- Every asset declares official video interval, slide pages or source lines,
  insertion point, evidence boundary, and caveat.
- Inline augmentation defaults to silent because it sits beside the professor.
  It must not autoplay while the official lecture is playing.
- Use voiceover only when the artifact must be understandable away from the
  lecture and contains more than two inferential steps. A voiced export is a
  separate variant; it does not replace the quiet inline version.
- Every asset ships a short adjacent caption answering “why this is here” and a
  structured Markdown transcript of visual states.
- A voiced asset also ships WebVTT captions matching the narration. A silent
  asset ships a WebVTT description track or equivalent step labels plus the
  full Markdown visual transcript for screen readers.
- Do not fabricate a verbatim professor quote. Context cards use labeled
  paraphrases unless the exact caption text has been verified.

### 5.5 Poster, contact sheet, and web encoding

- Poster: `1920x1080`; choose the decisive settled state, never a title card.
  Poster alt text states the visible relation and the result.
- Contact sheet: at least opening, every settled beat, and closing state, each
  stamped with its video time. Six to ten frames is typical. It is a review
  artifact, not the poster.
- Production MP4 must have `moov` before `mdat` (`-movflags +faststart`).
- Video-only: H.264 High/Main, YUV420p. With audio: AAC-LC, 48 kHz. Preserve a
  visually lossless master outside the published site if generation is costly.
- The manifest records exact duration, dimensions, fps, source hash, caption,
  transcript, poster, contact sheet, and review sign-offs.

## 6. Gated production workflow

No later stage can waive an earlier failure by saying the render looks good.

### Gate 0: storyboard gate

Deliverables:

- completed context card;
- medium decision and rejected alternatives;
- beat table with entrance/settle/dwell/exit seconds;
- exact values, formulas, state snapshots, and source provenance;
- one rough wireframe at desktop and mobile embed size.

Review questions:

- Does this answer a gap in the professor's run?
- Is motion necessary?
- Is there one question, one additive claim, and one evidence boundary?
- Does the artifact appear once even if it serves several stable IDs?

Exit: content owner approves the storyboard before scene code begins.

### Gate 1: low-resolution animatic

Render `854x480/15fps` with final beat timing but temporary styling. For an
interactive, record one deterministic 480p traversal. This file is stored
under a non-published `animatics/` path.

Exit:

- total duration is 25-60 seconds, or step mode has an estimated 25-60 second
  first pass;
- every required dwell is present;
- no header/takeaway duplicates site chrome;
- previous/next/replay work for teaching-critical steps.

### Gate 2: content review against professor transcript

The reviewer watches/reads the entire official interval, not only the named
slide. They complete a four-column ledger:

| Beat | Professor/deck claim | Augmentation claim | Evidence/caveat |
| --- | --- | --- | --- |

Reject if the augmentation:

- changes order within an uninterrupted run;
- turns a conditional or empirical statement into a universal prescription;
- imports a result without marking it as augmentation;
- omits the evidence or exception that the professor uses to qualify a claim;
- makes the next professor sentence harder to follow.

Exit: all formulas, values, units, source pages/lines, and contextual
paraphrases are signed off.

### Gate 3: visual QA

Use the animatic plus full-resolution still prototypes. Review the contact
sheet and a real-time 1x playback.

Exit:

- no overlap, clipping, or unsafe edge text;
- persistent color/variable mapping;
- formulas readable without fullscreen at 640 px;
- a 390 px viewport either keeps essential video text at 16 CSS px or replaces
  dense video text with responsive HTML;
- motion has stopped before every timed dwell;
- poster shows the decisive state;
- reduced-motion behavior and keyboard step controls are specified.

### Gate 4: 1080p production render

Render only after Gates 0-3 pass. Encode `1920x1080/30`, H.264/YUV420p, apply
fast-start, generate poster/contact sheet, and attach transcript/captions.

Automated exit criteria:

- exact dimensions and fps;
- duration in the approved range;
- H.264/YUV420p and `moov < mdat`;
- no missing sidecars;
- optional OCR warnings for small or unsafe text have been manually resolved.

### Gate 5: browser-context review

Mount the artifact once at the real insertion point. Test with the official
lecture paused immediately before it, then resume immediately after it.

Required viewports: 390, 768, and 1440 CSS px.

Exit:

- original video/deck remains the dominant object;
- no duplicated right-rail wrapper or duplicate shared video;
- source timestamp seeks correctly;
- video never starts over the official lecture audio;
- captions, transcript, playback rate, pause/scrub, and keyboard controls work;
- deep links to every served stable ID land on the one artifact's internal
  step rather than remounting it;
- the professor's next move is immediately intelligible.

## 7. Disposition matrix for the current 12 clips

“Keep concept” means preserve the teaching opportunity, not the existing MP4.
No current 480p file passes the primary-path delivery gate unchanged.

| Existing clip | Keep concept? | V2 disposition | Why / replacement |
| --- | --- | --- | --- |
| L1 `L01-UTF8-BYTES` | Yes | **Rebuild**, secondary priority | Variable-length byte expansion benefits from motion, but three expansions plus decode need 25-35 s and fewer labels. Static byte/code-point facts stay in HTML. |
| L1 `L01-BPE-PAIR-COUNT--L01-BPE-MERGE` | Yes | **Replace interactive**; absorb into L1 flagship | Percy already performs the two-minute code trace. Put one optional after-trace stepper at 75:22, not a dense replay during it. |
| L1 `L01-BPE-TRAIN-VS-USE` | Yes, high value | **Replace interactive**; final states of L1 flagship | It fills the exact skipped step, but the 7 s MP4 is a finished slide flashed in motion. |
| L1 `L01-TOKENIZER-PARETO` | Keep the reproducible toy data, not the concept name | **Remove video** | Motion adds nothing and “Pareto” overstates one declared toy comparison. Use a static table after BPE with provenance and caveat. |
| L2 `FloatingPointRange` | Yes | **Rebuild** or stepper, secondary priority | Separate dtype fact table from a 25-40 s `1e-8` reveal and mixed-precision handoff. |
| L2 `EinopsRearrangeHeads` | Yes | **Replace interactive** | Exact source shapes, named axes, split/transform/join, and reversible inspection require previous/next. Do not use the unlabeled adapted shape. |
| L2 `RooflineModel` | Yes, flagship | **Replace interactive / rebuild as ledger payoff** | Roofline belongs after the operation ladder. It is not a standalone plot intro. |
| L2 `ActivationCheckpointingTradeoff` | Yes | **Replace interactive** or 45-60 s rebuild | One backward request and its recomputation path should be replayable. Mount once for three anchors. |
| L3 `L03PrePostNorm` | Yes | **Rebuild**, secondary priority | Use 25-35 s for post -> pre morph and direct gradient route; leave evidence on the source slide beside it. |
| L3 `L03RuntimeNorm` | Partly | **Remove composite video** | Formula is a compare card; no-bias is an annotated equation; only read/reduce/write might merit a short motion loop. One 10 s micro-lecture is the wrong unit. |
| L3 `L03KVSharing` | Yes, flagship | **Rebuild as browser-coupled motion + evidence** | Begin with decode traffic, then MQA extreme, GQA knob, and the page 63 evidence/caveat. |
| L3 `L03SlidingWindow` | Yes | **Replace interactive** | A window-width scrubber and local/local/local/full propagation explain the lecture. A `78 -> 42` cell-count clip covers only the first half. |

## 8. Exactly three flagship rebuilds

The first V2 batch contains one flagship per lecture. Other clips remain
offline or explicitly marked legacy until these three pass the whole pipeline.

### 8.1 Lecture 1 flagship: after-trace BPE replay and train/use stepper

**Source context:** Lecture 1, 73:16-76:01; insert after Percy's uninterrupted
training trace at 75:22. Verify the relevant official `lecture_01.py` ranges;
the current manifest points to 549-564 and 705-720.

**One question:** During encoding of held-out text, which state changes, and
which state is frozen from training?

**Medium:** HTML/canvas stepper with a 55-second deterministic animatic for
review. Silent; no autoplay. Persistent panes: corpus/held-out sequence, ordered
merge rules, active pair, and length/vocabulary ledger.

| Animatic time | Beat | Settle + dwell |
| ---: | --- | ---: |
| 0-5 | Start on the final state Percy just produced; label it “replay after source trace,” outside the canvas | 3 s |
| 5-12 | Separate mutable training state from the ordered merge map; freeze rules 256, 257, 258 | 5 s |
| 12-19 | Replace corpus with held-out UTF-8 bytes; pair-count panel is visibly absent | 5 s |
| 19-27 | Apply rule 256 to all eligible non-overlapping matches; show before/after token count | 5 s |
| 27-35 | Apply rule 257; keep earlier rule order visible | 5 s |
| 35-43 | Apply rule 258; unmatched bytes remain unchanged | 5 s |
| 43-49 | Decode the resulting tokens back to the identical string | 4 s |
| 49-55 | Final two-column state ledger: training updates counts/vocab/rules; use updates only the token sequence | 6 s |

Final stepper controls expose one state per row above. `Previous` restores the
exact prior array; `Next` cannot skip the prediction pause on first traversal;
`Replay transition` repeats only the local merge.

Acceptance:

1. Exact sequences and non-overlapping merge behavior match official code.
2. Encoding never recomputes pair frequencies or changes the merge map.
3. Rule order, token IDs, sequence length, vocabulary size, and round trip are
   independently testable from a checked-in fixture.
4. The source trace remains uninterrupted; the stepper appears once after it.
5. `L01-BPE-PAIR-COUNT`, `L01-BPE-MERGE`, and
   `L01-BPE-TRAIN-VS-USE` deep-link to internal states of this one artifact.
6. No in-canvas title, source footer, or takeaway duplicates the page.

### 8.2 Lecture 2 flagship: arithmetic-intensity ledger pays off as roofline

**Source context:** Lecture 2, 40:30-57:10; the persistent ledger accumulates
from 42:25, and the roofline payoff appears only when Percy introduces the plot
at 55:10.

**One question:** How does reuse move an operation from memory-bound to
compute-bound on the same H100?

**Medium:** responsive HTML table plus interactive SVG/canvas roofline. A
60-second silent animatic records the canonical traversal. The table is the
source of plot points; selecting a point selects its derivation row.

| Animatic time | Beat | Settle + dwell |
| ---: | --- | ---: |
| 0-7 | ReLU row: about `n` FLOPs, `4n` BF16 bytes, intensity `0.25` | 5 s |
| 7-14 | GELU keeps traffic comparable but raises work to about `20n`; intensity `5` remains far below knee | 5 s |
| 14-20 | Dot row, about `0.5` FLOP/byte | 4 s |
| 20-27 | Matvec row, about `1` FLOP/byte; highlight limited reuse | 5 s |
| 27-37 | Square matmul row; reuse changes scaling to about `n/3`, `341` at `n=1024` | 7 s |
| 37-44 | Reveal the lecture's dense BF16 H100 roof: `989.5 TFLOP/s / 3.35 TB/s ≈ 295 FLOP/byte` | 6 s |
| 44-53 | Morph the visible intensity column into x positions, then reveal the sloped bandwidth roof and flat compute roof | 7 s |
| 53-60 | Select matmul, then GELU, and trace each point back to its bytes/FLOPs row; retain the real-overlap caveat | 7 s |

Acceptance:

1. Rows appear in Percy's order: ReLU, GELU, dot, matvec, matmul.
2. Every plotted x value is computed from the currently visible ledger row; no
   hand-positioned explanatory point.
3. The dense-versus-sparse and BF16 assumptions are visible next to the H100
   knee. Units remain on every quantity.
4. `max(t_memory, t_compute)` is labeled ideal overlap; real overlap overhead
   remains adjacent.
5. Copy never claims “compute-bound means use faster arithmetic.” It says to
   reduce required FLOPs or raise effective compute throughput, subject to the
   operation and hardware.
6. At 390 px, the table becomes the primary view and the plot is horizontally
   full-width, not a shrunken desktop diagram.
7. The final state hands back to Percy's MFU explanation and roofline summary.

### 8.3 Lecture 3 flagship: decode traffic to MQA/GQA, ending on evidence

**Source context:** Lecture 3 slides 57-63, 74:06-84:59. The artifact begins
only after slide 57 establishes that this unit concerns inference and
long-context interventions.

**One question:** Why does token-by-token decoding make KV sharing valuable,
and what evidence limits the claim that more sharing is always better?

**Medium:** one 55-60 second silent motion asset coupled to responsive HTML
accounting and the official page 63 evidence. The source slide remains visible;
the animation does not redraw or detach the paper result from it.

| Time | Beat | Settle + dwell |
| ---: | --- | ---: |
| 0-7 | Contrast one prefill pass with the first single-token decode step | 5 s |
| 7-16 | Append three decode tokens; each step rereads the growing prior K/V cache while doing little new work | 6 s |
| 16-24 | Freeze on the model-wide byte meter and one formula: `2 × n_kv_heads × d_head × bytes/value × n_layers` per cached token | 7 s |
| 24-32 | MHA state: query heads keep separate K/V streams; trace the repeated read once | 6 s |
| 32-40 | Morph to MQA's one K/V stream; byte meter falls, while an “expressiveness extreme” marker appears | 6 s |
| 40-48 | Morph back to GQA groups; expose group count as the capacity/traffic knob | 6 s |
| 48-58 | Stop motion. Browser highlights official page 63 and an adjacent evidence note: MQA has a reported small perplexity hit; cited GQA result reports low/no hit; neither is a universal guarantee | 10 s |

Acceptance:

1. The order is systems problem -> MQA extreme -> quality cost -> GQA
   compromise -> evidence. MQA is never introduced after GQA.
2. Cache byte arithmetic is exact for a declared configuration and distinguishes
   per token, per layer, model-wide, batch, and sequence multipliers.
3. Query-head count remains fixed while KV-head count changes.
4. The animation does not imply the KV cache is unnecessary; it shows the
   compute saved and repeated memory read it introduces.
5. Page 63 evidence and its caveat remain visible for 10 seconds or until the
   learner advances. No paper result is generalized beyond the cited setting.
6. `L03-KV-CACHE`, `L03-MQA`, and `L03-GQA` point to internal chapters of one
   mounted artifact.
7. A timecoded cross-link returns to Lecture 2's arithmetic-intensity ledger
   instead of redefining roofline in the right rail.

## 9. Content that should not become Manim

### Lecture 1

- abstraction ladder; mechanics/mindset/intuition; efficiency assumptions:
  compact comparison tables;
- five assignments and scarce-resource synthesis: persistent course-spine
  table;
- tokenizer interface and compression ratio: formula cards;
- tokenizer quirks: editable tokenizer-boundary interactive, repairing the
  failed live demo;
- character/byte/word/BPE comparison and the former “Pareto” asset: static,
  reproducible table;
- production tokenizer concerns: checklist plus Assignment 1 links.

### Lecture 2

- FP32/FP16/BF16 bit/range/resolution facts: HTML table; animate only the
  representability consequence;
- named dimensions, einsum, reduce, and rearrange: one persistent tensor
  inspector/stepper;
- FLOPs, FLOP/s, seconds, peak, and MFU: units/formula table;
- operation-intensity ledger and roofline: interactive, not a standalone MP4;
- optimizer memory `2+2+4+4`: byte stack/table;
- gradient accumulation versus checkpointing: comparison table plus stepper.

### Lecture 3

- architecture adoption matrix: filterable table preserving year and
  exceptions;
- LayerNorm/RMSNorm formulas, no-bias, activation functions, z-loss/QK
  norm/soft-cap: aligned formula/evidence tables;
- GLU parameter matching, FF ratio, head ratio, aspect ratio, vocabulary size:
  exact calculators and comparison tables;
- regularization and architecture prescriptions: evidence/caveat tables;
- sliding-window width: interactive causal matrix with layered propagation;
- papers and Q&A: collapsed timestamped background links.

## 10. QA tool and sign-off record

`../tools/qa_animation_v2.py` is a non-mutating validator for production
candidates. It checks ffprobe metadata, duration, codec/pixel format, MP4 atom
order, required sidecars, and optionally OCR-sampled text size/safe-area
warnings. It can generate a timestamped contact sheet only when an output path
is explicitly supplied.

Example:

```bash
python3 tools/qa_animation_v2.py \
  media/lecture_02/roofline-v2.mp4 \
  --transcript media/lecture_02/roofline-v2.md \
  --captions media/lecture_02/roofline-v2.vtt \
  --poster media/lecture_02/roofline-v2-poster.png \
  --ocr --contact-sheet /tmp/roofline-v2-contact.png
```

OCR is advisory: it cannot reliably parse every TeX formula. A clean automated
report never replaces Gate 3 or the 390 px browser review.

## 11. Definition of done for the first V2 batch

- Only the three flagships in Section 8 are eligible for the first production
  batch.
- Each has passed all six gates with named content and visual reviewers.
- Each asset is mounted once and connected to internal stable-ID chapters.
- Legacy 480p clips are not silently presented as V2; they are either hidden or
  explicitly labeled legacy reference.
- The site can be read with all augmentations collapsed and still reproduces
  the professor's lecture order and transitions.
- Opening an augmentation answers “why is this here now?” in one sentence;
  closing it makes the official next sentence easier, not harder, to follow.
