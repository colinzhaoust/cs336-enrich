# CS336 Spring 2026 Lecture 1 teaching map

## Purpose

This document reconstructs the teaching logic of the actual Spring 2026
Lecture 1 before proposing any augmentation. The aim is not to turn Percy's
lecture into a new concept taxonomy. It is to preserve his argument, pacing,
code demonstrations, and transitions, then add a small number of formula
comparisons, tables, background links, interactives, and slow visual demos at
the moments where they answer a question the lecture leaves open.

The timestamps below are paraphrases derived from the public caption track.
They are not a transcript. No complete transcript, video, or audio is stored in
the repository.

## Verified lecture identity and evidence

- Official offering: CS336, Spring 2026.
- Official schedule entry: Lecture 1, Monday March 30, “Overview,
  tokenization [Percy].”
- Lecturer: Percy Liang. The teaching staff introduce themselves during the
  opening, but Percy delivers the lecture.
- Official course page: <https://cs336.stanford.edu/>
- Official lecture source:
  <https://github.com/stanford-cs336/lectures/blob/main/lecture_01.py>
- Source revision inspected locally from the official lecture repository.
- Official recording:
  <https://www.youtube.com/watch?v=JuoVZkPBiKk>
- Recording title: “Stanford CS336 Language Modeling from Scratch | Spring
  2026 | Lecture 1: Overview, Tokenization.”
- Recording publisher: Stanford Online.
- Recording duration: 79:22 (`lengthSeconds = 4762`).
- Captions: a public English (United States) caption track and a public English
  auto-generated track are available. The English (United States) VTT was used
  for the map. Temporary caption and page files were kept in `/tmp` only.

The official source and the recording agree on the major order:

1. welcome and why the course exists;
2. current language-model landscape;
3. executable-lecture interlude;
4. logistics;
5. a tour of all five assignments, unified by efficiency;
6. the first actual unit, tokenization;
7. next lecture: resource accounting.

This ordering matters. Tokenization begins at **65:06**, not near the start of
the lecture. The first 65 minutes establish why the course is organized as it
is and how every later technical choice should be evaluated.

## Lecture-wide teaching thesis

Percy is not mainly trying to enumerate language-model components. He builds
one argument:

> Researchers need enough mechanics and systems mindset to tear open a leaky
> abstraction; because frontier scale is inaccessible, the transferable lens
> is efficiency under fixed resources; the five assignments repeatedly apply
> that lens; tokenization is the first concrete example.

The final tokenization section therefore works as an instance of the lecture's
efficiency thesis, not as a separate mini-course that can be moved ahead of the
syllabus tour.

## Timecoded teaching map

### 00:04–02:18 — welcome, staff, and the third offering

- **Question being answered:** Who is teaching this course, and why are they
  excited to teach “from scratch” when coding agents can already generate a
  great deal of code?
- **Sequence:** Percy introduces the staff and the third offering; Tatsu,
  Marcel, Herman, and Steven briefly introduce themselves; the introductions
  make the course feel like an active, evolving project rather than a frozen
  recording.
- **Emphasis:** Curiosity about how models work is still valuable even when
  automation can produce implementations.
- **What is passed over:** No technical definitions are intended yet.
- **Augmentation:** Keep only unobtrusive staff/course links. Do not put a
  concept card, formula, or animation between introductions.
- **Do not interrupt:** The whole opening is a human continuity block.

### 02:18–03:23 — what changed and what did not

- **Question:** What is new in the third offering?
- **Sequence:** The “from scratch” philosophy remains; the staff have refined
  what is high value per unit time; this year adds more current ingredients,
  especially MoE, long context, and agents.
- **Transition intent:** The update list leads into the deeper reason the
  course exists.
- **Augmentation:** A compact “stable philosophy / changing coverage” table can
  sit after this phase. It should not become its own lecture chapter.

### 03:23–04:54 — the abstraction ladder and fundamental research

- **Question:** Why build models when researchers can fine-tune or prompt
  existing ones?
- **Sequence:** Percy moves historically from implementing and training, to
  fine-tuning pretrained models, to prompting remote systems. He first grants
  that higher abstraction increases productivity, then introduces the failure
  mode: the abstractions leak and constrain the research design space.
- **Key transition intent:** “Understanding via building” follows as the
  remedy, not as craftsmanship for its own sake.
- **Repeated emphasis:** Prompting is useful; the claim is specifically about
  access to fundamental research questions.
- **Augmentation:** One small comparison table is appropriate:

  | Level | Fast path | What remains editable | Typical failure recourse |
  | --- | --- | --- | --- |
  | Prompt/API | Prompt a hosted model | Prompt, tools, context | Change prompt/provider |
  | Fine-tune | Adapt released weights | Data and selected parameters | Change training recipe |
  | From scratch | Build the stack | Tokenizer through systems | Change the underlying mechanism |

  This should be an annotation beside the continuous argument, not three
  navigable segments.
- **Do not interrupt:** 03:23–04:54 is a single concession → limitation →
  conclusion argument.

### 04:54–07:16 — industrial scale and the small-to-frontier gap

- **Question:** If understanding comes from building, why not simply learn all
  frontier-model intuitions by training a small model?
- **Sequence:** Frontier cost and lack of public construction details make
  direct replication impossible; Percy then gives two counterexamples to
  naive transfer: the MLP/attention FLOP mix changes with model scale, and some
  behavior appears only after reaching larger scales.
- **Emphasis:** Small models are useful but not automatically representative.
- **Augmentation:** Preserve the two examples as a paired evidence panel.
  Add source/background links and a tiny “changes with scale / might transfer”
  table. An animation is unnecessary unless it directly overlays the changing
  FLOP proportions on the original plot.
- **Do not interrupt:** Do not insert media between the FLOP-mix example and
  the emergence example; their pairing motivates the next classification.

### 07:16–09:15 — mechanics, mindset, and intuitions

- **Question:** What knowledge can this course honestly teach at accessible
  scale?
- **Sequence:** Percy classifies knowledge as mechanics, mindset, or intuition;
  mechanics and the efficiency/scaling mindset transfer better; empirical
  modeling and data intuitions may not. The SwiGLU anecdote demonstrates that
  some successful choices lack a satisfying derivation.
- **Emphasis:** The transfer boundary is part of the lesson, not a disclaimer
  to hide.
- **Augmentation:** This is well served by a three-row comparison table with
  columns for “how verified,” “small-scale value,” and “frontier caveat.” A
  three-box Manim flow would add little.
- **Do not interrupt:** Let the classification and SwiGLU example remain one
  unit.

### 09:15–11:38 — the bitter lesson reframed as efficiency

- **Question:** Does scaling imply that algorithms no longer matter?
- **Sequence:** Percy rejects “scale is all that matters,” replaces it with
  “algorithms that scale matter,” introduces the informal relation between
  accuracy, efficiency, and resources, explains why even small percentage
  gains matter at frontier cost, and ends with the fixed-budget question.
- **Repeated emphasis:** Efficiency becomes the mindset Percy wants students
  to carry through the quarter.
- **Best augmentation:** Put the formula and its assumptions in one compact
  comparison immediately after 11:29:

  | Quantity | In this lecture's framing | Important caveat |
  | --- | --- | --- |
  | Resources | Data, compute, memory, bandwidth | Which resource binds can change |
  | Efficiency | Useful model quality per resource | Not one universal scalar in practice |
  | Accuracy | Desired evaluation outcome | Depends on the evaluation and scale |

  The page should label `accuracy = efficiency × resources` as a framing, not
  a physical law or a predictive equation.
- **Do not interrupt:** 09:15–11:29 is the central argument of the lecture. A
  user may expand the formula after the conclusion, not during it.

### 11:38–19:25 — history as dependency context, not a second syllabus

- **Question:** Which durable ingredients produced modern language models, and
  why can an open course teach them?
- **Sequence:** pre-neural language modeling → neural sequence ingredients →
  pretrained foundation models → scaling → early open replications → credible
  open-weight models → more fully open models. Percy then changes the view of
  an LM from fine-tuned object, to prompted system, to conversational system,
  to agent, while arguing that kernels, optimization, Transformer-like
  mechanisms, and attention remain durable.
- **Emphasis:** Open research artifacts are what make reconstruction and this
  course possible. Product specifications change faster than fundamentals.
- **What is deliberately fast:** This is contextual panorama, not a lecture on
  every model family.
- **Augmentation:** An expandable dependency timeline is useful after 19:25.
  It should default to collapsed and distinguish “ingredient,” “scale event,”
  and “release/open ecosystem.” Do not make every named model a top-level
  segment.
- **Do not interrupt:** 11:38–19:10. The speed and accumulation are rhetorical;
  repeated media inserts would destroy that effect.

### 19:25–20:03 — executable-lecture interlude

- **Question:** Why does the displayed material look and behave like a Python
  program?
- **Sequence:** Percy explains that stepping through the rendered lecture is
  executing its program and demonstrates returning from a function to `main`.
- **Augmentation:** None beyond a link to the source/trace and a “run locally”
  note. Recreating this as a video would duplicate the live demonstration.
- **Do not interrupt:** This is only about 38 seconds and is itself a demo.

### 20:03–27:16 — logistics and the learning contract

- **Question:** Who should take the course, what work is expected, and how may
  students use AI and compute?
- **Sequence:** workload and five units → reasons to take/not take → following
  from home → assignment structure and unit tests → local correctness then
  cluster benchmarking/training → leaderboards → pedagogical AI policy → Modal
  compute → question pause.
- **Repeated emphasis:** Watching is not equivalent to doing; assignments are
  where the learning happens.
- **Augmentation:** Use links and at most one compact workflow diagram
  (`implement locally → test → run on cluster → benchmark/leaderboard`). This
  is support material, not a visual-explanation target.
- **Do not interrupt:** The AI-policy explanation and the compute handoff are
  administrative continuity blocks.

### 27:16–35:53 — Assignment 1: the complete basic LM loop

- **Question:** What must be built before one can honestly say “I trained a
  language model from scratch”?
- **Sequence:** Percy announces the five-part course and enters Basics:
  tokenization preview → architecture refinements → training decisions →
  Assignment 1 deliverables → the expressivity/stability/efficiency triad.
- **Transition intent:** Tokenization at 28:06 is a preview inside the syllabus,
  not yet the tokenization unit. It introduces “atoms,” compression, adaptive
  computation, and the tokenizer-free dream so later architecture and training
  decisions can be seen in the same pipeline.
- **Repeated emphasis:** Apparently small hyperparameters can determine whether
  a large run is stable and useful.
- **Best augmentation:** A single end-to-end strip can remain visible:
  `raw bytes → tokens → model → loss/optimizer → trained model`. Add a compact
  table for expressivity, stability, and efficiency at 34:36.
- **Do not interrupt:** 28:06–29:45 is a complete tokenization preview. Do not
  splice the later UTF-8 or BPE animation into it.

### 35:53–45:12 — Assignment 2: data movement is the systems story

- **Question:** After a correct model exists, what makes it run efficiently?
- **Sequence:** resource accounting and `6ND` preview → memory and compute are
  physically separated → roofline idea → GPU/node topology → kernels and
  fusion → multi-GPU collectives and sharding → prefill/decode inference →
  Assignment 2 → scaling-book recommendation.
- **Repeated emphasis:** Minimize data movement. This same principle links
  kernels, distributed training, and inference.
- **Augmentation:** Add the recommended background book link where Percy gives
  it. A table contrasting training, prefill, and decode can sit after 43:59;
  detailed roofline animation belongs in Lecture 2, not here.
- **Do not interrupt:** The fusion example and the move from single GPU to
  multi-GPU are a continuous abstraction-scale transition.

### 45:12–53:28 — Assignment 3: replace a single run with a scaling recipe

- **Question:** How can one choose a very expensive training configuration
  without tuning at full scale?
- **Sequence:** the one-shot large-budget problem → scaling recipe maps FLOPs
  to hyperparameters → smaller runs produce loss observations → fit and
  extrapolate → hyperparameter transfer and predictability → compute-optimal
  model/data balance → Chinchilla rule of thumb and inference caveat → live
  Marin preregistration → simulated assignment API.
- **Repeated emphasis:** Scaling laws must be constructed carefully; they are
  not automatic laws of nature. Predictability can matter as much as local
  optimality.
- **Augmentation:** A formula/assumption panel is valuable:
  `C ≈ 6ND`, plus a table distinguishing FLOPs budget `C`, parameters `N`,
  tokens `D`, and an explicit inference-cost caveat. A link to the live Marin
  result belongs at the original timestamp.
- **Do not interrupt:** 45:12–48:55 builds one conceptual shift; 49:00–51:56 is
  one example. Do not insert a clip in the middle of either.

### 53:28–60:20 — Assignment 4: capability starts with data and evaluation

- **Question:** What should the model learn, and where does usable training text
  come from?
- **Sequence:** trained/fast/scalable model still lacks a target data mixture →
  desired capabilities → internal versus external evaluation → diverse evals
  and contamination → active collection and legal questions → transform,
  filter, deduplicate, mix, and synthesize → pre/mid/post-training data → dirty
  work of the assignment.
- **Emphasis:** Data does not arrive as a clean dataset, and evaluation has
  different roles during development and external reporting.
- **Augmentation:** Two compact tables are enough: internal vs external eval,
  and raw source → transformation → filtering → deduplication → mixing. Avoid
  a Manim clip; the lecture is a rapid syllabus tour.
- **Do not interrupt:** Preserve Percy's question-driven handoff at 53:28–54:02.

### 60:20–64:46 — Assignment 5 and the efficiency synthesis

- **Question:** How can weaker feedback improve an already trained model, and
  what ties all five assignments together?
- **Sequence:** supervised next-token training → generate and score responses →
  preference/RL update → DPO/PPO/GRPO → instability and RL systems → possible
  Assignment 5 scope → final inventory of data, compute, memory, and bandwidth
  → tokenization, architecture, data filtering, and scaling laws reinterpreted
  as efficiency choices.
- **Repeated emphasis:** Binding constraints can change; the efficiency mindset
  survives the change.
- **Augmentation:** After 64:46, show a course-wide table mapping each assignment
  to its main scarce resource and decision. This is the right expanded version
  of the current `L01-COURSE-STACK` idea.
- **Do not interrupt:** 62:54–64:46 is the lecture's first large synthesis and
  should remain uninterrupted.

### 64:46–65:06 — question pause and a real unit boundary

This is a genuine chapter break. The revised site should represent it as such:
“Course overview” ends; “Tokenization” begins. Augmentation should never erase
this boundary.

### 65:06–65:50 — tokenizer interface

- **Question:** What type conversion is required between raw text and a
  language model?
- **Sequence:** Karpathy background recommendation → Unicode string → integer
  token sequence → encode/decode round trip.
- **Augmentation:** Keep the background link and a tiny reversible mapping:
  `string --encode--> list[int] --decode--> string`. No standalone animation is
  needed.

### 65:50–67:00 — tokenizer quirks; the failed live interaction is a real gap

- **Question:** Why are production token boundaries unintuitive?
- **Sequence:** Percy tries to use an interactive tokenizer site, notes the
  classroom connection will not support it, then verbally explains leading
  spaces, position-dependent token IDs, and digit grouping.
- **Emphasis:** These quirks motivate dissatisfaction with tokenizers.
- **Highest-value augmentation:** Embed a small editable tokenizer comparison
  at exactly this point. It repairs a live demonstration that did not work in
  the recording. Suggested presets: `hello hello`, leading-space variants, a
  long number, and the multilingual lecture string. It should show boundaries,
  IDs, UTF-8 bytes, and token count without navigating away.
- **Do not substitute:** A prerecorded Manim animation is less useful here than
  the interaction Percy intended.

### 67:00–68:29 — GPT tokenizer, round trip, and compression ratio

- **Question:** How do we quantify sequence shortening without forgetting
  vocabulary cost?
- **Sequence:** executable encode → executable decode → round-trip invariant →
  count 20 UTF-8 bytes and 8 tokens → compute 2.5 bytes/token → relate shorter
  sequences to quadratic attention → warn that increasing vocabulary creates
  sparsity → skip the full vocabulary because of time.
- **Augmentation:** Put the exact formula next to the observed values:
  `compression ratio = UTF-8 byte count / emitted token count`. Add a small
  two-axis table for sequence length versus vocabulary/embedding cost. The full
  vocabulary should be an optional background drawer, honoring Percy's skip.
- **Do not interrupt:** Keep encode, decode, invariant, and ratio as one code
  demonstration.

### 68:29–69:45 — character tokenizer

- **Question:** Why not use one Unicode character per token?
- **Sequence:** `ord`/`chr` code → round trip → approximately 150K assigned
  characters → rare slots and weak compression.
- **Emphasis:** The deeper issue is sparse, rarely trained vocabulary entries,
  not merely the headline vocabulary size.
- **Augmentation:** Add one row to a running tokenizer comparison table. If the
  page mentions Python's possible code-point ID domain (1,114,112), distinguish
  it explicitly from Percy's approximate count of assigned characters; do not
  make the two numbers appear contradictory.

### 69:45–70:50 — byte tokenizer and UTF-8

- **Question:** Can a fixed 256-entry vocabulary provide universal coverage?
- **Sequence:** one-byte ASCII versus multi-byte Unicode → encode to integers
  in `[0,255]` → longer sequence → compression ratio 1 → small vocabulary but
  poor sequence length.
- **Best use of existing animation:** `L01-UTF8-BYTES` can be retained here,
  after Percy establishes that a character may use multiple bytes. It should
  be an optional “slow down the byte expansion” demonstration, not the primary
  lecture content.
- **Required refinement:** The current 8.6-second clip presents three rows, a
  flattened byte stream, decoding, and a takeaway too quickly. Target roughly
  22–30 seconds or a click-stepper. Hold each character expansion long enough
  to read, then assemble/decode the stream. Remove the repeated lecture title,
  segment ID, source line, and takeaway from inside the video because the page
  already supplies them.
- **Do not interrupt:** Insert only after the original executable conversion,
  not between the premise and the observed byte IDs.

### 70:50–72:00 — word tokenizer and the OOV failure

- **Question:** Why not make meaningful human words the atoms?
- **Sequence:** regex chunking → semantic appeal and good compression → large
  training-derived vocabulary → unseen word → `UNK` → lost structure and
  perplexity problems.
- **Augmentation:** A static before/after row showing an unseen word collapsing
  to `UNK` is enough. It should join the character/byte/BPE comparison table.
  No animation is necessary.

### 72:00–73:16 — BPE motivation and algorithm sketch

- **Question:** How can a tokenizer keep byte-level coverage while learning
  useful larger chunks?
- **Sequence:** compression origin → NLP adaptation → GPT-2 usage → train on
  corpus → common sequences receive single tokens, rare sequences decompose →
  start from bytes and repeatedly merge the most frequent adjacent pair.
- **Augmentation:** Place the original Gage, Sennrich, and GPT-2 links here.
  A two-column “training learns ordered merges / encoding replays them” preview
  can prepare the following code without replacing it.

### 73:16–75:22 — BPE training, continuous executable code trace

- **Question:** What state changes during each BPE training round?
- **Sequence:** initialize bytes → count overlapping adjacent pairs → choose a
  maximum (Percy explicitly notes a tie and takes the first) → allocate token
  256 → add vocabulary bytes → replace all non-overlapping occurrences → repeat
  to 257 and 258 → observe corpus length shrink and vocabulary grow → compute
  toy compression ratio.
- **Emphasis:** This is the longest detailed code demonstration in the
  tokenization unit. The evolving program state is the teaching medium.
- **Best use of existing animation:** The current shared pair-count/merge scene
  contains useful material, especially the explicit tie table and the
  length/vocabulary ledger. But the 12.87-second rendering repeats a sequence
  Percy already spends over two minutes tracing. It should not be inserted
  during the trace.
- **Required refinement:** Convert it to either:
  1. a 30–45 second optional replay after 75:22, synchronized with the official
     source lines; or
  2. preferably, a user-controlled stepper with corpus tokens, pair-count
     table, winning pair, merge table, and sequence/vocabulary sizes visible at
     once.
  Keep the pair-count and merge phases together; splitting them into separate
  navigation stops would break the training loop.
- **Do not interrupt:** 73:16–75:22. This must remain one uninterrupted code
  trace in the original lecture column.

### 75:22–76:01 — using a trained tokenizer; the second genuine gap

- **Question:** What changes when applying the learned tokenizer to new text?
- **Sequence:** Percy states that new text begins from bytes and replays the
  learned merge order; he deliberately does **not** step through this code;
  decode reconstructs the original string.
- **Highest-value existing animation:** `L01-BPE-TRAIN-VS-USE` belongs exactly
  after the skipped step-through. Its conceptual distinction is genuinely
  additive.
- **Required refinement:** The current 7-second clip is essentially a dense
  final slide appearing in motion. Slow it to roughly 24–35 seconds or make it
  click-stepped: learn rules on the corpus → freeze ordered rules → initialize
  held-out bytes → replay rule 1/2/3 → decode. Visually distinguish “no new pair
  counts” without repeating a large title and takeaway.

### 76:01–77:29 — from teaching implementation to Assignment 1 tokenizer

- **Question:** What separates the correct toy implementation from a usable
  production implementation?
- **Sequence:** the code is correct but slow → avoid scanning irrelevant merges
  → special-token handling → pretokenize into chunks → optimize implementation,
  possibly beyond Python.
- **Augmentation:** Use a concise comparison table:

  | Toy reference | Assignment/production concern |
  | --- | --- |
  | Scan every learned merge | Index only applicable merges |
  | Treat full string as one sequence | Pretokenize into chunks |
  | Ordinary byte content | Preserve special tokens |
  | Clear Python loops | Efficient data structures / implementation |

  Link directly to the relevant Assignment 1 problem and test fixtures. This
  is more useful than another animation.
- **Do not interrupt:** The list is Percy's handoff from lecture to assignment;
  keep it as one checklist.

### 77:29–79:15 — summary, tokenizer-free invariants, and next lecture

- **Question:** What should survive even if BPE is eventually replaced?
- **Sequence:** recap the tokenizer interface and baseline failures → BPE as a
  data-driven heuristic → models still need abstractions/chunks → chunks should
  be variable to support adaptive computation → next lecture is resource
  accounting, then architectures.
- **Emphasis:** The final claim returns to the efficiency/adaptive-computation
  idea introduced in the Assignment 1 preview at 28:45. That long-range return
  should be visible in the revised site.
- **Augmentation:** After the lecture ends, offer a final comparison table and
  links to byte-level/tokenizer-free work. Do not put a video before Percy's
  closing synthesis or next-lecture handoff.

## What the current site gets structurally wrong

The current Lecture 1 data defines 20 equally addressable concept segments in
four chapters: `Course frame`, `Representation`, `Tokenizer baselines`, and
`Byte Pair Encoding`. That is a useful content inventory, but it is not the
lecture's teaching structure.

### 1. It promotes supporting concepts into a replacement syllabus

`L01-ABSTRACTION-LADDER`, `L01-INDUSTRIAL-SCALE`,
`L01-KNOWLEDGE-TYPES`, and `L01-EFFICIENCY-EQUATION` are successive steps in
one 03:23–11:38 argument. Making each an independent destination encourages
readers to enter or leave in the middle of the proof.

They should remain anchorable annotations inside one official chapter,
“Why this course exists,” rather than four co-equal lecture units.

### 2. It compresses 37 minutes of syllabus logic into one generic stack card

`L01-COURSE-STACK` reduces Basics, Systems, Scaling Laws, Data, Alignment, and
the final efficiency synthesis to three beats. In the real lecture, 27:16–64:46
is not filler: Percy repeatedly asks what is still missing after the previous
assignment and uses the answer to introduce the next one.

The revised site should restore the five-assignment narrative spine and place
small augmentation tables at the original transitions.

### 3. It detaches tokenization from the efficiency argument

The current chapter change suggests that tokenization follows directly after
the course-frame concept cards. In the lecture, tokenization is previewed at
28:06, related to adaptive computation and efficiency, then embedded in a tour
of the entire stack, synthesized again at 62:54, and only formally taught from
65:06. This repeated setup is pedagogically important.

### 4. It orders a derived comparison before the BPE mechanism that justifies it

`L01-TOKENIZER-PARETO` appears with the baseline tokenizers before pair
counting, merging, and train-vs-use. Its BPE numbers require knowledge of the
three learned merges, so the site asks the learner to accept evidence from a
mechanism not yet taught. It is also not a mathematical Pareto frontier.

Move it after the BPE explanation, rename it to something like “One declared
toy comparison,” and render it as a static, reproducible table.

### 5. Several segments duplicate the same official moment

- `L01-UNICODE-CODEPOINTS` and `L01-CHAR-TOKENIZER` both derive from the same
  68:29–69:45 character-tokenizer demonstration.
- `L01-UTF8-BYTES` and `L01-BYTE-TOKENIZER` both derive from the same
  69:45–70:50 byte-tokenizer demonstration.
- `L01-BPE-PAIR-COUNT`, `L01-BPE-MERGE`, and `L01-BPE-MERGE-CODE` are phases of
  one training loop, not three independent lecture chapters.

Stable IDs can remain for discussion links, but the reading UI should group
them under one original lecture moment and avoid repeating summary/goal/source
text for each ID.

### 6. The right rail repeats metadata instead of supplying missing evidence

The surrounding card already provides a title, segment ID, source location,
summary, goal, beats, and takeaway. Each current Manim clip repeats a large
title, segment ID, source line, and final takeaway inside the video. The slide
copy then restates much of the same explanation again.

This creates the impression of a second lecture competing with the original.
The augmentation pane should contain only the incremental artifact and one
short “why this is here” caption. Title, provenance, and discussion controls
belong in shared page chrome.

## Disposition of current Lecture 1 augmentations

| Current item | Disposition | Correct placement | Reason |
| --- | --- | --- | --- |
| `L01-ABSTRACTION-LADDER` | Keep as compact table/note | After 04:54 | Clarifies the argument but should not split it |
| `L01-INDUSTRIAL-SCALE` | Keep sources and paired evidence | After 07:16 | The two scale counterexamples work as a pair |
| `L01-KNOWLEDGE-TYPES` | Keep as table, not Manim | After 09:15 | Comparison is spatial, not temporal |
| `L01-EFFICIENCY-EQUATION` | Keep with assumptions/caveat | After 11:29 | Central framing; avoid presenting it as a literal predictive law |
| `L01-LM-TIMELINE` | Keep as collapsed background timeline | After 19:25 | Contextual panorama, not the lecture's main navigation |
| `L01-COURSE-STACK` | Replace with official five-assignment spine | 27:16–64:46 | Current card erases the actual syllabus argument |
| `L01-TOKENIZER-INTERFACE` | Keep inline | 65:21–65:50 | Small reversible mapping is sufficient |
| `L01-TOKENIZER-QUIRKS` | Prioritize as an interactive | 65:50–67:00 | Repairs the failed live demonstration |
| `L01-COMPRESSION-RATIO` | Keep as formula + two-axis comparison | 67:00–68:29 | Formula and tradeoff need simultaneous reading |
| `L01-UNICODE-CODEPOINTS` + `L01-CHAR-TOKENIZER` | Merge in UI | 68:29–69:45 | Same source moment and argument |
| `L01-UTF8-BYTES` + `L01-BYTE-TOKENIZER` | Merge in UI; retain a much slower optional demo | 69:45–70:50 | Variable-length expansion benefits from motion, but not at 8.6 seconds |
| `L01-WORD-TOKENIZER` | Keep as a static OOV example | 70:50–72:00 | No evolving state requires video |
| `L01-BPE-PAIR-COUNT` + `L01-BPE-MERGE` + `L01-BPE-MERGE-CODE` | Group as one optional replay/stepper | After 75:22 | Preserve Percy's uninterrupted code trace |
| `L01-BPE-TRAIN-VS-USE` | Keep and substantially slow | After 75:44 | Fills the exact step-through Percy skips |
| `L01-PRETOKENIZATION` | Keep as production checklist/table | 76:01–77:29 | Best paired with Assignment 1 links |
| `L01-TOKENIZER-PARETO` | Delete the 6.6-second video; retain a renamed static table | After 77:29 | Motion adds nothing; current name overstates the toy evidence |

## Clip-level quality findings

The four current Lecture 1 MP4s are low-quality Manim renders at 854×480 and
15 fps. Their measured durations are:

| Clip | Duration | Finding |
| --- | ---: | --- |
| `L01-UTF8-BYTES.mp4` | 8.60 s | Too fast for three code-point/byte expansions plus reassembly and decode |
| `L01-BPE-PAIR-COUNT--L01-BPE-MERGE.mp4` | 12.87 s | Compresses a two-minute source trace into a dense replay; useful facts, wrong pacing/placement |
| `L01-BPE-TRAIN-VS-USE.mp4` | 7.00 s | Strong augmentation target, but currently behaves like a dense slide flashed briefly |
| `L01-TOKENIZER-PARETO.mp4` | 6.60 s | Static table presented as video; remove the video form |

The core problem is not only resolution. Each scene treats “show all required
facts” as completion, while the lecture requires time to establish a question,
make one state change, inspect the result, and connect it back to Percy's
claim.

For retained motion:

- render at least 1280×720, 30 fps for published material;
- remove redundant scene headers, source footers, and takeaway sentences;
- show one causal change per beat;
- leave essential text readable for at least roughly 2.5–4 seconds;
- use 4–6 second holds on states the learner must compare;
- prefer click-to-advance for code-aligned token operations;
- do not autoplay augmentation while the original lecture is playing;
- review the animation beside the relevant 30–120 seconds of source video,
  not as an isolated MP4.

## Recommended augmentation pipeline for Lecture 1

1. **Locate the exact teaching gap.** Record the official time range, source
   lines, question Percy is answering, and what he demonstrates versus skips.
2. **Choose the lightest artifact.** Link for background; table for comparison;
   formula block for assumptions; interactive for exploration; Manim only for
   state change over time.
3. **Write an additive claim.** One sentence stating what the augmentation adds
   that the original lecture does not already show.
4. **Storyboard against the source.** Include the five seconds before and after
   the insertion point so the opening and closing frames connect to Percy's
   words.
5. **Pace for reading and prediction.** Insert a pause or user step before the
   answer appears. Do not make all labels arrive simultaneously.
6. **Remove duplicated chrome.** The site owns title, ID, provenance, and
   discussion. The artifact owns only the visual evidence.
7. **Review in context.** Validate both artifact quality and whether the
   original teaching flow remains intelligible with every augmentation
   collapsed.

## Minimal Lecture 1 augmentation set

If the goal is to augment rather than refactor, the first high-quality release
does not need twenty right-column experiences. It needs approximately these
seven:

1. 03:23–04:54: abstraction-level comparison table.
2. 07:16–09:15: mechanics/mindset/intuition transfer table.
3. 09:15–11:38: efficiency framing with assumptions.
4. 27:16–64:46: original five-assignment spine with a few background links and
   one final resource/efficiency matrix.
5. 65:50–67:00: embedded tokenizer-boundary interactive.
6. 69:45–70:50: slow UTF-8 byte expansion.
7. 75:22–76:01: slow or step-controlled train-versus-use BPE demonstration,
   followed by a static tokenizer comparison and production checklist.

Everything else can begin as a source link or collapsed note. This preserves
Percy's 79-minute lecture as the main object and makes the enriched material
feel like carefully placed marginalia rather than a competing reconstruction.
