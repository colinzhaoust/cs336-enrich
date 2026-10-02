# Atlas: philosophy

Goal: help a human learn a technical subject faster than the raw lecture allows,
without lying about what the lecture said.

## One object, two lines

The atlas is a graph of **knowledge points** (KPs). Everything else is a view.

- A KP is the smallest thing a learner can be tested on with **one changed-case
  question**. If it needs two questions, split it. If it cannot be tested alone,
  it is a presentation, not a KP.
- A **lecture** is a path through the graph chosen by the professor.
- The **teacher line** produces the graph's *provenance*: which KPs the professor
  touched, which were said, computed, cited, shown, deferred to an assignment,
  or dropped. Teaching order is itself evidence of prerequisite structure
  (InstructKG, 2026).
- The **student line** produces the graph's *presentations*: for each KP on the
  path, the learner action it demands and the cheapest medium that supports it,
  plus the supplementary KPs the professor assumed.

The teacher line answers "what did the professor actually filter and why".
The student line answers "what do I feed a learner so they can act". Both
attach to the same node.

## What the evidence says (see research/03-learning-science.md)

1. **Commit before reveal, offered, not enforced.** (Colin, 2026-10-02: nothing is locked; a prediction is an optional pause, and the closed-book changed case is an opt-in self-test.) A prediction typed before the answer is the
   single largest lever (Crouch 2004; Brod 2021; ICAP Active vs Constructive).
   Sliders alone do not teach.
2. **Retrieval is the unit.** Delayed recall beats rereading by ~20 points at a
   week (Roediger & Karpicke 2006). Every KP carries a prompt that is re-asked
   later. Quantum Country is the only interactive format with published
   retention data.
3. **Worked, then faded.** First derivation fully worked with "why this step";
   later instances hide the last step (Renkl; Kalyuga expertise reversal).
4. **Animate only change over time.** Animation's average gain is small
   (g≈0.2–0.4) and vanishes with captions or decoration (Tversky 2002;
   Höffler & Leutner 2007). Every animation is segmented, steppable, and
   carries one *notice*: the one thing the learner must observe.
5. **No open chat box.** Unguarded tutors raise practice scores and lower
   exam scores (Bastani 2025). A tutor, if any, is invoked after a failed
   transfer question with the KP's solution and misconception list in context.

## What the engineering evidence says (see research/02-*.md)

- **Separate semantics from rendering** (ALGOGEN 99.8% vs 82.5%). An author or
  model emits a typed fixture; a deterministic primitive library renders it.
  The LLM never writes a Manim scene directly.
- **Render success is not education.** TheoremExplainAgent's 94% success hid
  pervasive layout faults and zero learner data. LLM2Manim is the only Manim
  system with a human post-test (d=0.67), and it used Mayer's principles and an
  expert review gate.
- **Lint before render.** Symbolic bounds/overlap checks catch most layout
  faults for free (SGA; See-Before-You-Code).
- **Simulated students are not evidence.** Use them for coverage checks only
  (competence paradox, 2601.05473).

## Sequencing and mastery (see research/07-sequencing-and-transfer-items.md)

- Show the whole map; gate on prerequisites; let the learner choose only among *ready*
  knowledge points (ALEKS outer-fringe success ≈ 0.93; free learner control g ≈ 0.05).
- Block first exposure, interleave retrieval only across knowledge points already passed.
- A knowledge point is *mastered* when its changed case is answered, and *secured* only
  after a retrieval prompt is remembered at least two days later. Three changed cases,
  one of them far, is the bar for calling anything learned; one is a start.
- Changed-case items declare what varies and what is invariant, and choice distractors come
  from named misconceptions. Items that leak the answer by wording are defects.

## Grading (see research/06, 08, 09)

- Three blind audits answered all 752 predict/transfer prompts without seeing the key, then compared.
  They found prompt faults (about 6%) and, more importantly, grader faults.
- Numbers, choices and up/down/same trends are graded automatically by `site/grade.js`, and
  `tools/test_grader.mjs` proves every official answer passes its own grader and that x2, x0.5 and the
  opposite trend fail. A number prompt asks for exactly one quantity.
- Free-text answers are self-graded after the reveal. Token overlap rejected 61% of correct
  paraphrases, so the learner compares their answer with the official one and its reason and marks it.
  The automatic verdict is still logged, so an export shows where the two disagree.
- Every knowledge point has an auto-graded changed case: either the transfer itself (number, choice,
  trend) or a separate `check`. Mastery needs the self-graded transfer *and* the check; a knowledge
  point is only *secured* after a retrieval prompt is remembered two days later.
- Choice options are shown in a stable per-item shuffle: authors list the right answer first.
- No LLM grades learners. If one is added later it must be audited the same way first.

**The changed case is closed book.** A widget can compute any case, and the source quotes and retrieval
answers state facts the check asks for, so presentations, provenance and retrieval cards are hidden from the
moment the learner starts the transfer until transfer and check are answered. Provenance is also hidden
before the prediction. Seeing the mechanism comes before the test, never during it.

**Retrieval has layers.** A card is `core` (the concept a practitioner keeps) or `detail` (an identifier,
path, exact constant, source wording). Both exist, but they are not shown together: detail cards are folded on
the KP page and join reviews only after the KP's core cards are remembered twice (Colin, 2026-10-02). Card
questions carry no slide/line citations; answers may.

## The spoken layer (see research/10)

The professor's speech is evidence too. Human-made captions become timestamped transcripts; spoken
anchors (`video:M:SS-M:SS`) are verified like line anchors. Speech adds reasons, caveats and oral
filtering (skips, deferrals) that slides never show, and also contains slips. When speech and the
written source disagree, the disagreement is recorded, and neither silently wins.

## Provenance discipline

Every claim in a KP points to a line in the official source, an assignment
problem, or a paper. Every status is `evidence` or `inference`. Teaching order
from `lecture_XX.py` is evidence; a prerequisite edge the author adds is
inference until a learner attempt confirms it matters.

## Non-goals

- Not a universal course compiler. One lecture done deeply beats seventeen
  done shallowly.
- Not a video generator. Video is one presentation among six.
- Not a chat tutor.
- Not a metric dashboard. Artifact counts are not coverage.

## Definition of done for a KP

A KP is done when a learner can: state a prediction, observe the mechanism,
answer a changed-case question, and be re-asked a week later. Until a real
learner has done this for at least one KP, every count in this repo is a
description of inputs, not outcomes.
