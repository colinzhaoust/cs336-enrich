(function (root) {
  "use strict";

  const slotTypes = [
    "formula-comparison",
    "table",
    "background-link",
    "interactive-demo",
    "slow-manim"
  ];
  const artifactStatuses = ["keep", "rebuild", "remove"];

  const official = {
    course: "https://cs336.stanford.edu/",
    lecturesRepo: "https://github.com/stanford-cs336/lectures",
    playlist: "https://www.youtube.com/playlist?list=PLoROMvodv4rMqXOcazWaTUHhq-yembLCV"
  };

  const artifactInventory = [
    { path: "media/lecture_01/L01-UTF8-BYTES.mp4", status: "rebuild", role: "optional slow byte-expansion replay" },
    { path: "media/lecture_01/L01-BPE-PAIR-COUNT--L01-BPE-MERGE.mp4", status: "rebuild", role: "post-trace replay; prefer stepper" },
    { path: "media/lecture_01/L01-BPE-TRAIN-VS-USE.mp4", status: "rebuild", role: "repair the code step-through Percy skips" },
    { path: "media/lecture_01/L01-TOKENIZER-PARETO.mp4", status: "remove", role: "replace with a renamed reproducible static table" },
    { path: "media/lecture_02/FloatingPointRange.mp4", status: "rebuild", role: "underflow/range demonstration after the dtype table" },
    { path: "media/lecture_02/EinopsRearrangeHeads.mp4", status: "remove", role: "replace with an exact-source shape stepper" },
    { path: "media/lecture_02/RooflineModel.mp4", status: "remove", role: "replace with a persistent ledger culminating in an interactive roofline" },
    { path: "media/lecture_02/ActivationCheckpointingTradeoff.mp4", status: "rebuild", role: "single slow checkpoint/recompute artifact mounted once" },
    { path: "media/lecture_03/videos/lecture_03_scenes/480p15/L03PrePostNorm.mp4", status: "rebuild", role: "slow topology trace beside official evidence" },
    { path: "media/lecture_03/videos/lecture_03_runtime_norm/480p15/L03RuntimeNorm.mp4", status: "remove", role: "split into formula table and focused movement demo" },
    { path: "media/lecture_03/videos/lecture_03_scenes/480p15/L03KVSharing.mp4", status: "rebuild", role: "cache problem first, then MHA/MQA/GQA and evidence" },
    { path: "media/lecture_03/videos/lecture_03_scenes/480p15/L03SlidingWindow.mp4", status: "remove", role: "replace with window and interleaving controls" },
    { path: "media/**/posters/* and media/**/*-poster.png", status: "keep", role: "archival storyboard reference only" },
    { path: "scenes/cs336/lecture_0*/", status: "keep", role: "source and reconstruction reference; not proof of publish quality" }
  ];

  function slot(id, type, atSeconds, pedagogicalReason, content, extra = {}) {
    return { id, type, atSeconds, pedagogicalReason, content, optional: true, ...extra };
  }

  function run(id, title, startSeconds, endSeconds, professorIntent, sourceRefs,
    originalSummary, anchors, uninterruptedRanges, augmentationSlots, extra = {}) {
    return {
      id,
      title,
      startSeconds,
      endSeconds,
      professorIntent,
      sourceRefs,
      originalSummary,
      anchors,
      uninterruptedRanges,
      augmentationSlots,
      ...extra
    };
  }

  const lectures = [
    {
      lecture: 1,
      title: "Overview and tokenization",
      instructor: "Percy Liang",
      videoUrl: "https://www.youtube.com/watch?v=JuoVZkPBiKk",
      videoDurationSeconds: 4762,
      videoVerifiedFrom: "public English (United States) caption track and Stanford Online recording metadata",
      sourceUrl: "https://github.com/stanford-cs336/lectures/blob/main/lecture_01.py",
      spineThesis: "Build enough mechanics and systems mindset to open leaky abstractions; use efficiency under fixed resources as the transferable lens; treat tokenization as the first worked instance.",
      restoredSourceBeats: [
        "00:04–03:23 staff continuity and what changed in the third offering",
        "20:03–27:16 logistics, assignment workflow, AI policy, and compute handoff",
        "27:16–64:46 the complete five-assignment question-and-answer spine",
        "64:46–65:06 the genuine overview-to-tokenization boundary",
        "77:29–79:15 recap, tokenizer-free invariants, adaptive computation, and next-lecture handoff"
      ],
      runs: [
        run("L01-R01", "Welcome and the learning contract", 4, 203,
          "Establish the course as a living third offering and explain why from-scratch understanding still matters when coding can be automated.",
          ["lecture_01.py · opening/main flow", "video 00:04–03:23"],
          "Staff introductions lead into a stable from-scratch philosophy and a changing set of current topics such as MoE, long context, and agents.",
          [], [{ startSeconds: 4, endSeconds: 203, reason: "The human opening and update list form one continuity block." }],
          [slot("L01-R01-LINKS", "background-link", 203, "The opening needs provenance and staff context, not a competing concept explanation.", { links: [official.course, official.lecturesRepo] })],
          { restoredEvidence: ["Staff introductions and the stable-philosophy/changing-coverage transition are retained even though no legacy segment ID covered them."] }),

        run("L01-R02", "Why this course exists", 203, 698,
          "Move from the productivity of higher abstraction to its limits, identify what transfers from accessible scale, and make efficiency the course-wide mindset.",
          ["lecture_01.py · lines 65–123", "video 03:23–11:38"],
          "The abstraction ladder leaks for fundamental research; small models do not reproduce every frontier intuition; mechanics and mindset transfer better than empirical taste; algorithms that absorb resources efficiently still matter.",
          ["L01-ABSTRACTION-LADDER", "L01-INDUSTRIAL-SCALE", "L01-KNOWLEDGE-TYPES", "L01-EFFICIENCY-EQUATION"],
          [
            { startSeconds: 203, endSeconds: 294, reason: "Concession → limitation → building conclusion." },
            { startSeconds: 294, endSeconds: 436, reason: "The two scale counterexamples motivate the transfer classification together." },
            { startSeconds: 436, endSeconds: 555, reason: "Mechanics/mindset/intuition and the SwiGLU caveat are one unit." },
            { startSeconds: 555, endSeconds: 698, reason: "Efficiency is the central conclusion and should not be interrupted before its fixed-budget framing." }
          ],
          [
            slot("L01-R02-ABSTRACTION", "table", 294, "A spatial comparison makes the editable surface and failure recourse at each abstraction level inspectable without restating the argument.", { columns: ["level", "fast path", "editable surface", "failure recourse"] }),
            slot("L01-R02-TRANSFER", "table", 555, "The lecture explicitly distinguishes three knowledge types; a table keeps their evidence and scale caveats aligned.", { columns: ["knowledge type", "how verified", "small-scale value", "frontier caveat"] }),
            slot("L01-R02-EFFICIENCY", "formula-comparison", 698, "The informal equation needs its variables and binding-resource caveat visible so it is not mistaken for a physical law.", { formula: "accuracy = efficiency × resources", caveat: "A framing; resources and evaluations are multidimensional." })
          ]),

        run("L01-R03", "History, openness, and executable lectures", 698, 1203,
          "Give fast dependency context for durable LM ingredients, show why open artifacts make reconstruction possible, then explain the executable lecture medium.",
          ["lecture_01.py · lines 126–185", "video 11:38–20:03"],
          "A rapid history connects n-grams, neural sequence models, foundation models, scaling, open replications, chat systems, and agents; the source-code interlude then reveals that the displayed lecture is itself executing Python.",
          ["L01-LM-TIMELINE"],
          [
            { startSeconds: 698, endSeconds: 1165, reason: "The speed and accumulation of the history are rhetorical and should remain continuous." },
            { startSeconds: 1165, endSeconds: 1203, reason: "The executable-lecture interlude is already a live demo." }
          ],
          [
            slot("L01-R03-TIMELINE", "interactive-demo", 1165, "A collapsed dependency timeline lets learners revisit names after the panorama without turning them into a second syllabus.", { defaultState: "collapsed", categories: ["ingredient", "scale event", "open ecosystem"] }),
            slot("L01-R03-SOURCE", "background-link", 1203, "The best supplement to an executable-lecture demo is the actual source and a run-local note.", { links: ["https://github.com/stanford-cs336/lectures/blob/main/lecture_01.py"] })
          ]),

        run("L01-R04", "Logistics: watching is not doing", 1203, 1636,
          "Define who the course is for, how assignments are evaluated, how local and cluster work connect, and the pedagogical contract for AI use.",
          ["lecture_01.py · logistics section", "video 20:03–27:16"],
          "Workload, prerequisites, home viewing, unit tests, benchmarking, leaderboards, AI policy, and Modal compute form the operational learning contract.",
          [], [{ startSeconds: 1203, endSeconds: 1636, reason: "Administrative explanations depend on uninterrupted handoffs, especially AI policy to compute." }],
          [slot("L01-R04-WORKFLOW", "table", 1636, "A compact workflow preserves the distinction between correctness and systems evaluation without adding a new technical unit.", { rows: ["implement locally", "run unit tests", "run on cluster", "benchmark or train", "compare on leaderboard"] })],
          { restoredEvidence: ["The logistics and learning-contract section was absent from the 20-card presentation model and is now restored to the spine."] }),

        run("L01-R05", "Five assignments, one efficiency lens", 1636, 3886,
          "Repeatedly ask what is still missing after each assignment, building from a correct model to systems, scaling, data, alignment, and a final scarce-resource synthesis.",
          ["lecture_01.py · lines 235–251 plus assignment tour", "video 27:16–64:46"],
          "Basics establishes the LM loop; Systems minimizes data movement; Scaling replaces one large guess with a recipe; Data connects capability to collection and evaluation; Alignment uses feedback; the closing synthesis maps every decision back to data, compute, memory, or bandwidth.",
          ["L01-COURSE-STACK"],
          [
            { startSeconds: 1686, endSeconds: 1785, reason: "The tokenization preview is a complete preview and must not receive the later UTF-8/BPE media." },
            { startSeconds: 2153, endSeconds: 2712, reason: "Systems moves continuously from resource accounting through hardware, fusion, collectives, and inference." },
            { startSeconds: 2712, endSeconds: 3208, reason: "Scaling recipe setup and worked example each rely on deferred payoff." },
            { startSeconds: 3208, endSeconds: 3620, reason: "Capability → evaluation → collection → data pipeline is one question-driven handoff." },
            { startSeconds: 3774, endSeconds: 3886, reason: "Alignment closes into the lecture-wide efficiency synthesis." }
          ],
          [
            slot("L01-R05-PIPELINE", "table", 2153, "A persistent end-to-end strip keeps the five assignments connected without replacing Percy's question sequence.", { rows: ["raw bytes → tokens → model → optimizer → trained model"] }),
            slot("L01-R05-6ND", "formula-comparison", 3208, "The scaling-law preview uses C≈6ND; a units-and-assumptions panel makes the later Lecture 2 derivation discoverable.", { formula: "C ≈ 6ND", variables: ["C: training FLOPs", "N: parameters", "D: training tokens"], caveat: "Inference cost can change the compute-optimal choice." }),
            slot("L01-R05-RESOURCE-MATRIX", "table", 3886, "A five-assignment resource matrix restores the detailed synthesis erased by the former three-beat course-stack card.", { columns: ["assignment", "decision", "scarce resource", "efficiency question"] })
          ],
          { restoredEvidence: ["All five assignment transitions", "expressivity/stability/efficiency triad", "internal vs external evaluation", "raw-data processing pipeline", "final resource synthesis"] }),

        run("L01-R06", "The real unit boundary", 3886, 3906,
          "Pause after the course synthesis and explicitly reset from overview to the first technical unit.",
          ["video 64:46–65:06"], "A question pause separates Course overview from Tokenization.", [],
          [{ startSeconds: 3886, endSeconds: 3906, reason: "This pause is itself the boundary; no augmentation belongs inside it." }], []),

        run("L01-R07", "Tokenizer interface, quirks, and compression", 3906, 4109,
          "Define the reversible string↔IDs interface, reveal production-tokenizer quirks, then quantify sequence shortening while keeping vocabulary cost visible.",
          ["lecture_01.py · lines 567–612", "video 65:06–68:29"],
          "Encode/decode establishes the contract; a failed classroom web demo leaves token-boundary behavior unexplored; executable GPT tokenization then produces a round trip and a 20-byte/8-token compression example.",
          ["L01-TOKENIZER-INTERFACE", "L01-TOKENIZER-QUIRKS", "L01-COMPRESSION-RATIO"],
          [
            { startSeconds: 3906, endSeconds: 3950, reason: "Encode and decode form one reversible mapping." },
            { startSeconds: 4020, endSeconds: 4109, reason: "Encode, decode, invariant, and ratio are one code demonstration." }
          ],
          [
            slot("L01-R07-KARPATHY", "background-link", 3950, "Percy recommends background here; attaching it at the original moment preserves his flow.", { label: "Tokenizer background recommended in lecture" }),
            slot("L01-R07-TOKENIZER", "interactive-demo", 4020, "This directly repairs the live tokenizer interaction that failed in the classroom and lets learners test leading spaces, digits, and multilingual text.", { presets: ["hello hello", " leading space", "123456789", "多语言 🌍"], outputs: ["boundaries", "IDs", "UTF-8 bytes", "token count"] }),
            slot("L01-R07-RATIO", "formula-comparison", 4109, "The exact ratio and vocabulary/sequence tradeoff need simultaneous inspection after the code demonstration.", { formula: "compression ratio = UTF-8 bytes / emitted tokens", example: "20 / 8 = 2.5 bytes/token" })
          ]),

        run("L01-R08", "Three baseline tokenizers and their failure modes", 4109, 4320,
          "Test character, byte, and word atoms in turn so their coverage, sequence-length, sparsity, and OOV failures motivate learned subwords.",
          ["lecture_01.py · lines 627–695", "video 68:29–72:00"],
          "Character tokens are readable but sparse; bytes guarantee coverage with a fixed 256-entry vocabulary but lengthen sequences; words compress well but collapse unseen forms to UNK.",
          ["L01-UNICODE-CODEPOINTS", "L01-CHAR-TOKENIZER", "L01-UTF8-BYTES", "L01-BYTE-TOKENIZER", "L01-WORD-TOKENIZER"],
          [
            { startSeconds: 4109, endSeconds: 4185, reason: "ord/chr, assigned-character scale, sparsity, and compression constitute one character-tokenizer argument." },
            { startSeconds: 4185, endSeconds: 4250, reason: "UTF-8 premise, observed byte IDs, vocabulary size, and sequence length are one byte-tokenizer demonstration." },
            { startSeconds: 4250, endSeconds: 4320, reason: "Semantic appeal leads directly to OOV collapse." }
          ],
          [
            slot("L01-R08-UTF8", "slow-manim", 4250, "Variable-length character-to-byte expansion is a real temporal state change; a slow optional replay helps after Percy's executable conversion.", { targetSeconds: "22–30", artifact: "media/lecture_01/L01-UTF8-BYTES.mp4", artifactStatus: "rebuild", guardrail: "One character expansion at a time; no repeated title/source/takeaway chrome." }),
            slot("L01-R08-BASELINES", "table", 4320, "A running comparison is more readable than separate cards and preserves the successive failure argument.", { columns: ["tokenizer", "coverage", "vocabulary", "sequence length", "failure mode"] })
          ]),

        run("L01-R09", "BPE: from bytes to learned chunks", 4320, 4522,
          "Motivate BPE, then let the executable program—not a competing animation—show the changing corpus, pair counts, tie choice, new IDs, and non-overlapping replacement.",
          ["lecture_01.py · lines 705–758", "video 72:00–75:22"],
          "BPE starts with universal byte coverage and repeatedly counts adjacent pairs, selects a maximum, allocates a new token ID, and merges occurrences; the toy corpus shrinks while the vocabulary grows.",
          ["L01-BPE-PAIR-COUNT", "L01-BPE-MERGE", "L01-BPE-MERGE-CODE"],
          [{ startSeconds: 4396, endSeconds: 4522, reason: "The evolving program state is the teaching medium; pair counting, tie resolution, merge allocation, and cursor movement are one trace." }],
          [
            slot("L01-R09-BACKGROUND", "background-link", 4396, "The original Gage, Sennrich, and GPT-2 context deepens the motivation without interrupting code.", { labels: ["Gage BPE", "Sennrich et al.", "GPT-2 tokenizer"] }),
            slot("L01-R09-REPLAY", "interactive-demo", 4522, "A user-controlled state table is useful only after the official trace; it allows replay without compressing Percy's two-minute explanation.", { states: ["corpus tokens", "pair counts", "winner", "merge table", "sequence length", "vocabulary size"], artifact: "media/lecture_01/L01-BPE-PAIR-COUNT--L01-BPE-MERGE.mp4", artifactStatus: "rebuild" })
          ]),

        run("L01-R10", "Use the tokenizer, then return to invariants", 4522, 4762,
          "Distinguish learning ordered merges from replaying them on new text, hand off from teaching code to Assignment 1 production concerns, and close on variable chunks and adaptive computation.",
          ["lecture_01.py · lines 505–571, 710–726", "video 75:22–79:22"],
          "Encoding begins from held-out bytes and replays frozen rules without recounting pairs; production code adds indexing, pretokenization, and special tokens; the recap compares tokenizer failures and asks what abstractions should survive beyond BPE.",
          ["L01-BPE-TRAIN-VS-USE", "L01-PRETOKENIZATION", "L01-TOKENIZER-PARETO"],
          [
            { startSeconds: 4561, endSeconds: 4649, reason: "Production concerns are one assignment handoff checklist." },
            { startSeconds: 4649, endSeconds: 4755, reason: "Recap, tokenizer-free invariants, adaptive computation, and next-lecture handoff form the closing synthesis." }
          ],
          [
            slot("L01-R10-TRAIN-USE", "slow-manim", 4561, "Percy deliberately skips this step-through, so a slow replay is genuinely additive rather than duplicative.", { targetSeconds: "24–35 or click-stepped", artifact: "media/lecture_01/L01-BPE-TRAIN-VS-USE.mp4", artifactStatus: "rebuild", steps: ["learn rules", "freeze order", "initialize held-out bytes", "replay rules", "decode"] }),
            slot("L01-R10-PRODUCTION", "table", 4649, "Toy-versus-production concerns are categorical and belong in one checklist beside the assignment handoff.", { columns: ["toy reference", "assignment/production concern"] }),
            slot("L01-R10-COMPARISON", "table", 4762, "A declared toy table can compare exact counts after the mechanism is known, without overstating one example as a Pareto frontier.", { rename: "One declared toy comparison", artifact: "media/lecture_01/L01-TOKENIZER-PARETO.mp4", artifactStatus: "remove" })
          ],
          { restoredEvidence: ["Final tokenizer recap", "tokenizer-free variable-chunk invariant", "adaptive-computation callback", "next Lecture 2 handoff"] })
      ]
    },

    {
      lecture: 2,
      title: "PyTorch (einops) and resource accounting",
      instructor: "Percy Liang",
      videoUrl: "https://www.youtube.com/watch?v=kuYAsz7zspQ",
      videoDurationSeconds: 4645,
      videoVerifiedFrom: "public English subtitle track, Stanford Online recording metadata, and playlist position 2",
      sourceUrl: "https://github.com/stanford-cs336/lectures/blob/main/lecture_02.py",
      spineThesis: "Open two compute/memory ledger questions, derive every missing term from tensor storage and operations, diagnose bottlenecks, then close both ledgers with 6ND, optimizer bytes, and activation-memory trades.",
      restoredSourceBeats: [
        "00:05–01:00 Marin scaling-law callback from Lecture 1",
        "57:10–61:40 the running deep-network setup before local backward equations",
        "76:20–77:24 the callback summary that closes both opening ledgers",
        "Live self-corrections on H100 dense peak, ReLU intensity, unit arithmetic, and gradient accumulation"
      ],
      runs: [
        run("L02-R01", "Open the compute and memory ledgers", 5, 290,
          "Connect to the Marin run, reveal two napkin answers before their derivations, and define mechanics, mindset, and intuition as the lecture's learning contract.",
          ["lecture_02.py · lines 71–86", "video 00:05–04:50"],
          "The 70B/15T/1024-H100 time estimate and the eight-H100 AdamW capacity estimate are promises whose unexplained terms organize the rest of the lecture; activations are explicitly omitted from the latter.",
          ["L02-TRAINING-TIME", "L02-MODEL-CAPACITY"],
          [{ startSeconds: 60, endSeconds: 290, reason: "Both questions and the three learning types form one motivational contract." }],
          [
            slot("L02-R01-MARIN", "background-link", 60, "A direct run/forecast link preserves the Lecture 1 scaling-law callback without opening a new scaling lesson.", { label: "Marin 1e23-FLOP result" }),
            slot("L02-R01-LEDGERS", "interactive-demo", 290, "Keeping unknown terms visible makes later callbacks legible and preserves the deferred-payoff structure.", { ledgers: ["6ND training time", "2+2+4+4 bytes/parameter"], state: "open/unresolved", caveat: "Capacity estimate omits activations." })
          ]),

        run("L02-R02", "Tensor storage and precision", 290, 1080,
          "Build memory accounting from tensor roles, shapes, dtypes, and element size; then treat lower precision as a range/resolution trade with operational mixed-precision consequences.",
          ["lecture_02.py · lines 89–199", "video 04:50–18:00"],
          "Ranks and named B,S,H,D roles feed numel×element-size accounting; FP32 leads to FP16 underflow, BF16's reallocated exponent budget, mixed precision, lower-bit context, student Q&A, and finally CPU/GPU placement.",
          ["L02-TENSOR-RANK", "L02-BSHD", "L02-FP32", "L02-TENSOR-MEMORY", "L02-FP16-UNDERFLOW", "L02-BF16-RANGE", "L02-MIXED-PRECISION", "L02-AMP-CASTING", "L02-FP8", "L02-FP4", "L02-CPU-GPU"],
          [
            { startSeconds: 290, endSeconds: 480, reason: "Tensor roles, FP32 anatomy, and byte count are one bottom-up storage derivation." },
            { startSeconds: 480, endSeconds: 790, reason: "FP16 → BF16 → mixed precision is one causal chain." },
            { startSeconds: 790, endSeconds: 1020, reason: "FP8/FP4 and their Q&A are one optional digression, not peer chapters." }
          ],
          [
            slot("L02-R02-TENSOR-LEDGER", "table", 480, "A persistent role/shape/dtype/lifetime ledger carries the same accounting object forward without four card resets.", { columns: ["role", "shape", "dtype", "lifetime", "numel", "bytes"] }),
            slot("L02-R02-DTYPES", "formula-comparison", 790, "Bit allocation, range, resolution, and the 1e-8 result need side-by-side comparison before motion.", { formats: ["FP32", "FP16", "BF16"], artifact: "media/lecture_02/FloatingPointRange.mp4", artifactStatus: "rebuild" }),
            slot("L02-R02-UNDERFLOW", "interactive-demo", 790, "A step-controlled representability test lets the learner predict FP16 zero before revealing the result, then connects directly to BF16.", { value: "1e-8", reveal: ["FP16 → 0", "BF16 → nonzero"] }),
            slot("L02-R02-LOWBITS", "background-link", 1020, "FP8/FP4 receive only digression-level weight while preserving authoritative primers.", { defaultState: "collapsed" }),
            slot("L02-R02-DEVICE", "table", 1080, "A small CPU RAM/interconnect/GPU HBM diagram fills the laptop-demo gap without inventing timing measurements.", { kind: "device-placement diagram" })
          ]),

        run("L02-R03", "Einops as named bookkeeping", 1080, 1645,
          "Replace error-prone anonymous axes with named dimensions, reusing one tensor while introducing contraction, shared prefixes, reduction, splitting, transforming, and joining.",
          ["lecture_02.py · lines 214–276", "video 18:00–27:25"],
          "Negative axes motivate names; einsum contracts hidden while preserving batch; ellipsis abbreviates shared prefixes; reduce removes named axes; rearrange splits total_hidden into heads×hidden1 and joins it back.",
          ["L02-NAMED-DIMS", "L02-EINSUM-MATMUL", "L02-BATCHED-ATTENTION", "L02-EINSUM-ELLIPSIS", "L02-REDUCE", "L02-REARRANGE"],
          [{ startSeconds: 1080, endSeconds: 1645, reason: "This is a compact language tutorial using one motivation, notation, tensor, and color mapping." }],
          [slot("L02-R03-INSPECTOR", "interactive-demo", 1645, "A single stepper tied to exact source shapes preserves dimensional identity through split, contraction, reduction, and join.", { sourceShape: "total_hidden=8 → heads=2 × hidden1=4 → output width=8", artifact: "media/lecture_02/EinopsRearrangeHeads.mp4", artifactStatus: "remove", rebuildCondition: "Use exact source w[4,4], persistent colors, and unclipped code at 30–45 s or step mode." })]),

        run("L02-R04", "From 2BDK to measured MFU", 1645, 2430,
          "Separate work from rate, derive matmul cost from one output cell, benchmark asynchronous GPU work correctly, and compare measured throughput with the dense dtype-specific peak.",
          ["lecture_02.py · lines 279–335", "video 27:25–40:30"],
          "FLOPs and FLOP/s are given units; H100 sparse marketing numbers are separated from dense BF16 peak; x[B,D]@w[D,K] yields approximately 2BDK; synchronization makes timing meaningful; MFU exposes the remaining gap before memory bandwidth explains it.",
          ["L02-FLOPS-UNITS", "L02-MATMUL-2BDK", "L02-MFU"],
          [{ startSeconds: 1645, endSeconds: 2430, reason: "2BDK → measured rate → MFU is one dependency chain with a deferred bottleneck explanation." }],
          [
            slot("L02-R04-STRIP", "formula-comparison", 2430, "Keeping B,D,K, units, timing, and the MFU denominator in one strip prevents each formula from becoming an isolated fact.", { formulas: ["FLOPs ≈ 2BDK", "measured FLOP/s = FLOPs / synchronized seconds", "MFU = measured / dense dtype peak"] }),
            slot("L02-R04-TIMING", "table", 2430, "A checklist captures the easily missed synchronization requirement without duplicating Percy's code.", { rows: ["synchronize", "run", "synchronize", "repeat"] })
          ]),

        run("L02-R05", "Data movement to roofline", 2430, 3430,
          "Add HBM traffic to compute accounting, build an ordered ladder of arithmetic intensity from ReLU to matmul, and introduce roofline only as the final summary.",
          ["lecture_02.py · lines 338–481", "video 40:30–57:10"],
          "Move/compute/move yields an ideal max(memory time, compute time) model; fixed-traffic ReLU and GELU, then dot, matvec, and tiled matmul progressively increase reuse; the H100 knee diagnoses the bottleneck and pays off the earlier MFU question.",
          ["L02-COMPUTE-MEMORY", "L02-RELU-INTENSITY", "L02-GELU-INTENSITY", "L02-DOT-INTENSITY", "L02-MATVEC-INTENSITY", "L02-MATMUL-INTENSITY", "L02-ROOFLINE"],
          [{ startSeconds: 2430, endSeconds: 3430, reason: "ReLU → GELU → dot → matvec → matmul is a deliberate monotonic reuse ladder; roofline summarizes it." }],
          [
            slot("L02-R05-LEDGER", "table", 3430, "A persistent operations table makes every roofline point auditable from bytes and FLOPs and preserves Percy's teaching order.", { rows: ["BF16 ReLU", "BF16 GELU", "BF16 dot", "BF16 matvec", "BF16 n×n matmul"], columns: ["FLOPs", "bytes", "intensity", "diagnosis"] }),
            slot("L02-R05-ROOFLINE", "interactive-demo", 3430, "The learner should reveal the roof only after deriving the points, then scrub back to the ledger row that produced each one.", { artifact: "media/lecture_02/RooflineModel.mp4", artifactStatus: "remove", rebuildCondition: "45–60 s or interactive; add one point at a time and preserve exact caveats." })
          ]),

        run("L02-R06", "A deep network closes 6ND", 3430, 4005,
          "Set up one L-layer D-wide network, reuse named dimensions to derive the two backward matmuls, and return to the opening training-time estimate.",
          ["lecture_02.py · lines 484–556", "video 57:10–66:45"],
          "A running network with D²L parameters grounds autograd; each layer has one forward matmul and two same-sized backward matmuls, producing 2ND+4ND=6ND under the stated Transformer approximation.",
          ["L02-AUTOGRAD", "L02-BACKWARD-2X", "L02-6ND"],
          [{ startSeconds: 3430, endSeconds: 4005, reason: "Network setup, local gradients, cost aggregation, and the opening callback form one derivation." }],
          [slot("L02-R06-DERIVATION", "slow-manim", 4005, "Motion is justified for tracing values forward and two distinct gradient contractions backward while the cost table remains visible.", { targetSeconds: "45–70 or step-controlled", rows: ["forward 2ND", "backward 4ND", "total 6ND"], caveat: "Long contexts add quadratic attention work." })],
          { restoredEvidence: ["L-layer running network setup", "explicit callback to the 02:05 estimate", "context-length caveat"] }),

        run("L02-R07", "Optimizer state closes the memory ledger", 4005, 4330,
          "Use optimizer families only to count persistent buffers, distinguish their dtype from model tensors, and separate the 12-byte stack from activation memory.",
          ["lecture_02.py · lines 602–715", "video 66:45–72:10"],
          "AdaGrad supplies one accumulator; Adam supplies first and second moments; BF16 parameter and gradient plus FP32 moments total 12 bytes per parameter; activations depend on batch, sequence, layers, and hidden sizes. The detailed train-loop walkthrough is skipped in the recording.",
          ["L02-OPTIMIZER-FAMILY", "L02-OPTIMIZER-MEMORY", "L02-TRAIN-LOOP"],
          [{ startSeconds: 4005, endSeconds: 4330, reason: "Optimizer family is introduced only to explain buffer count and close the opening ledger." }],
          [slot("L02-R07-BYTE-STACK", "formula-comparison", 4330, "Putting per-parameter bytes beside, but not inside, the activation formula preserves the lecture's omitted-term caveat and verbal unit correction.", { stack: ["parameter BF16: 2", "gradient BF16: 2", "Adam m FP32: 4", "Adam v FP32: 4"], separate: "activation memory = f(batch, sequence, layers, hidden sizes)" })],
          { restoredEvidence: ["The train-loop code is labeled as skipped rather than promoted to an equal lecture unit.", "The on-screen unit-confusing formula is annotated with Percy's verbal correction."] }),

        run("L02-R08", "Two activation-memory trades and the final callback", 4330, 4645,
          "Contrast microbatch accumulation with selective activation retention, derive checkpoint placement tradeoffs, and close both original napkin questions.",
          ["lecture_02.py · lines 718–773", "video 72:10–77:25"],
          "Gradient accumulation reuses activation memory across microbatches while retaining gradients; checkpointing discards selected forward states and recomputes them; every-layer, no-intermediate, and √L schedules trade memory for recomputation; the recap closes all open ledger terms.",
          ["L02-GRAD-ACCUM", "L02-ACTIVATION-MEMORY", "L02-CHECKPOINTING", "L02-CHECKPOINT-FREQUENCY"],
          [{ startSeconds: 4405, endSeconds: 4580, reason: "Activation premise → selective retention → recomputation → checkpoint frequency is one continuous story." }],
          [
            slot("L02-R08-COMPARE", "table", 4580, "The table prevents gradient accumulation and checkpointing from being conflated and makes the different paid costs explicit.", { columns: ["technique", "discarded/sliced", "persists", "cost paid"] }),
            slot("L02-R08-CHECKPOINT", "slow-manim", 4580, "One backward request and its exact recomputation path are temporal and benefit from a slow, user-controlled demonstration.", { targetSeconds: "45–75 or step mode", artifact: "media/lecture_02/ActivationCheckpointingTradeoff.mp4", artifactStatus: "rebuild", mountOnceFor: ["L02-ACTIVATION-MEMORY", "L02-CHECKPOINTING", "L02-CHECKPOINT-FREQUENCY"] }),
            slot("L02-R08-CLOSE", "table", 4645, "Returning to both opening estimates makes the lecture's deferred-payoff structure visible without 39 recap cards.", { state: "all ledger terms resolved" })
          ],
          { restoredEvidence: ["76:20–77:24 final summary and ledger callbacks"] })
      ]
    },

    {
      lecture: 3,
      title: "Architectures and hyperparameters",
      instructor: "Tatsunori Hashimoto",
      videoUrl: "https://www.youtube.com/watch?v=lVynu4bo1rY",
      videoDurationSeconds: 5350,
      videoVerifiedFrom: "public human English captions, Stanford Online recording, and the 67-page official deck",
      sourceUrl: "https://github.com/stanford-cs336/lectures/blob/main/lecture_03.pdf",
      spineThesis: "Survey many trained models to infer conservative defaults, repeatedly moving from a design choice to formula/topology, adoption evidence, systems or stability rationale, exceptions, and recap.",
      restoredSourceBeats: [
        "Slides 1–2 survey method",
        "Slides 24–26 controlled GLU evidence and activation recap",
        "Slide 29 architecture recap",
        "Slide 36 hyperparameter question list",
        "Slide 51 hyperparameter recap",
        "Slide 57 efficient-attention roadmap",
        "Slide 67 lecture recap returning to common versus variable axes"
      ],
      runs: [
        run("L03-R01", "Why these modern defaults?", 5, 461,
          "Explain the empirical survey method, compare the original Transformer with the Assignment 1 baseline, establish the architecture matrix, and name the three evaluation lenses.",
          ["lecture_03.pdf · slides 1–9", "video 00:05–07:41"],
          "Tatsu starts from what one can learn by training, then uses cross-model evidence as a second-best method; original and modern blocks frame the choices; a model survey and roadmap distinguish common defaults from variable axes under learn-well/run-efficiently/not-blow-up criteria.",
          ["L03-MODERN-TRANSFORMER", "L03-ARCH-MATRIX"],
          [{ startSeconds: 5, endSeconds: 461, reason: "Survey method → baseline comparison → matrix → evaluation lenses is the setup for the entire lecture." }],
          [
            slot("L03-R01-BEFORE-AFTER", "table", 461, "A four-row comparison keeps the block visible and lets each changed component be highlighted without turning it into four cards.", { rows: ["post-norm → pre-norm", "sin/cos → RoPE", "ReLU → SwiGLU", "bias → no bias"] }),
            slot("L03-R01-MATRIX", "interactive-demo", 461, "A chronological matrix helps inspect evidence while preserving the lecturer's order and common/variable distinction.", { defaultOrder: "year and lecturer sequence", labels: ["common", "variable", "next lecture"] })
          ],
          { restoredEvidence: ["Slides 1–2 survey method", "Slide 8 roadmap"] }),

        run("L03-R02", "Keep the residual stream clean", 461, 855,
          "State the pre-norm consensus, support it with convergence and gradient evidence, then show newer norm placements as extensions that preserve the identity route.",
          ["lecture_03.pdf · slides 10–13", "video 07:41–14:15"],
          "Post-norm and pre-norm are compared through residual topology, warmup/convergence data, clean-path gradient reasoning, and spikes; non-residual post norm and double norm are presented as empirical variations on the same principle.",
          ["L03-PRE-POST-NORM", "L03-DOUBLE-NORM"],
          [
            { startSeconds: 461, endSeconds: 784, reason: "Claim → data → gradient explanation is an atomic argument." },
            { startSeconds: 785, endSeconds: 855, reason: "Double/non-residual post norm extends the same principle rather than starting a new lesson." }
          ],
          [slot("L03-R02-TOPOLOGY", "slow-manim", 855, "Tracing the identity path while morphing post to pre to double norm is a genuine topology change; evidence remains beside the official slides.", { targetSeconds: "25–35", artifact: "media/lecture_03/videos/lecture_03_scenes/480p15/L03PrePostNorm.mp4", artifactStatus: "rebuild", evidenceBoundary: "Do not animate away slides 11–12 evidence or overstate final-quality guarantees." })]),

        run("L03-R03", "Normalization as systems design", 855, 1215,
          "Move from LayerNorm/RMSNorm formulas to data movement and runtime evidence, then generalize the same systems and optimization rationale to no-bias before recapping.",
          ["lecture_03.pdf · slides 14–19", "video 14:15–20:15"],
          "RMSNorm removes centering and bias; fewer FLOPs do not alone explain runtime because normalization moves and reduces data; paper evidence limits the claim; no-bias is presented through memory and optimization-stability reasoning.",
          ["L03-RMSNORM", "L03-NORM-RUNTIME", "L03-NO-BIAS"],
          [{ startSeconds: 855, endSeconds: 1215, reason: "Formula → systems reason → evidence → generalization → recap is a six-slide argument." }],
          [
            slot("L03-R03-FORMULAS", "formula-comparison", 1215, "LayerNorm and RMSNorm are dense static formulas whose changed operations must be compared simultaneously.", { compare: ["center + scale", "scale only"] }),
            slot("L03-R03-MOVEMENT", "interactive-demo", 1215, "Read/reduce/write traffic is the missing systems explanation and should be isolated from the static formula and no-bias equation.", { steps: ["read", "reduce", "normalize", "write"], artifact: "media/lecture_03/videos/lecture_03_runtime_norm/480p15/L03RuntimeNorm.mp4", artifactStatus: "remove" }),
            slot("L03-R03-EVIDENCE", "table", 1215, "Keeping the course's runtime/performance evidence and adoption caveat next to the claim prevents a universal faster-and-equal prescription.", { includeSlides: [16, 17, 18, 19] })
          ],
          { restoredEvidence: ["Slides 16–17 runtime/performance evidence", "Slide 19 normalization/no-bias recap"], correctedClaim: "No-bias is justified here by memory movement and optimization stability, not a measured claim that bias has negligible modeling capacity." }),

        run("L03-R04", "Gated FFNs, then block topology", 1215, 1871,
          "Narrow the activation zoo to gating, construct and parameter-match GLU variants, read controlled evidence and caveats, then compare serial and parallel blocks before recapping architecture choices.",
          ["lecture_03.pdf · slides 20–29", "video 20:15–31:11"],
          "ReLU/GELU/Swish lead to content×gate branches and GLU variants; parameter fairness and two controlled studies show a consistent small gain without necessity; parallel blocks offer fusion potential but adoption and controlled evidence remain mixed.",
          ["L03-ACTIVATION-ZOO", "L03-GLU-GATE", "L03-SERIAL-PARALLEL"],
          [
            { startSeconds: 1290, endSeconds: 1633, reason: "GLU construction → parameter fairness → controlled evidence → caveat must remain together." },
            { startSeconds: 1634, endSeconds: 1871, reason: "Parallel-block systems claim, evidence uncertainty, adoption retreat, and recap form one unit." }
          ],
          [
            slot("L03-R04-ACTIVATIONS", "formula-comparison", 1290, "A static formula/curve comparison clears the vocabulary so motion can focus only on gating.", { functions: ["ReLU", "GELU", "Swish"] }),
            slot("L03-R04-GATE", "slow-manim", 1633, "Elementwise content×gate interaction is temporal and spatial; a single slow demo can explain it without swallowing the parameter/evidence discussion.", { targetSeconds: "25–40", oneClaim: "content branch × learned gate" }),
            slot("L03-R04-EVIDENCE", "table", 1633, "The missing controlled results and recap are necessary to calibrate a small, consistent benefit rather than a theorem.", { restoreSlides: [24, 25, 26] }),
            slot("L03-R04-PARALLEL", "table", 1871, "Claimed systems benefit, evidence gap, and current adoption need to be visible together.", { columns: ["claimed win", "evidence gap", "current adoption"] })
          ],
          { restoredEvidence: ["Slides 24–26 GLU evidence and recap", "Slide 29 architecture recap"] }),

        run("L03-R05", "Position strategies to RoPE", 1871, 2620,
          "Compare where position enters, derive RoPE's relative-inner-product condition from 2D rotation through coordinate pairs and code, then preserve the natural Q&A as an optional detour.",
          ["lecture_03.pdf · slides 30–35", "video 31:11–43:40"],
          "Sine, learned absolute, relative bias, and rotary injection are compared; coupled rotations make Q·K depend on relative angle; coordinate pairs receive frequencies; sin/cos tables implement the geometry; Q&A limits the conclusions.",
          ["L03-POSITION-FAMILIES", "L03-ROPE-RELATIVE", "L03-ROPE-FREQUENCIES", "L03-ROPE-CODE"],
          [{ startSeconds: 1871, endSeconds: 2349, reason: "Target invariant → 2D geometry → high-dimensional pairing → code is one derivation." }],
          [
            slot("L03-R05-FAMILIES", "table", 2349, "The four strategies differ categorically by injection point and should be compared before the derivation.", { columns: ["method", "where position enters", "absolute/relative behavior"] }),
            slot("L03-R05-ROPE", "slow-manim", 2349, "Coupled rotation and relative-angle invariance are inherently dynamic; a pauseable demo can bridge formula, coordinate pairs, and code in lecture order.", { targetSeconds: "45–60", chapters: ["relative distance", "coupled rotation", "coordinate pairs", "code highlight"] }),
            slot("L03-R05-QA", "background-link", 2620, "The post-RoPE Q&A is the safe place for papers and uncertain extensions, which should not be upgraded into main claims.", { defaultState: "collapsed" })
          ]),

        run("L03-R06", "Which hyperparameters actually matter?", 2620, 3910,
          "Open the model-instantiation question list, compare defaults and exceptions for FF width, heads, depth/width, vocabulary, and regularization, then close with a recap rather than isolated prescriptions.",
          ["lecture_03.pdf · slides 36–51", "video 43:40–65:10"],
          "The 4× FF default, GLU 8/3 parameter match, LLaMA variants, T5 exception and broad basin lead to head and aspect ratios, vocabulary observations, then the dropout/weight-decay optimization caveat; FLOPs often dominate exact ratios.",
          ["L03-FF-RATIO", "L03-GLU-DIMENSION", "L03-FF-BASIN", "L03-HEAD-RATIO", "L03-ASPECT-RATIO", "L03-VOCAB-SIZE", "L03-REGULARIZATION"],
          [
            { startSeconds: 2620, endSeconds: 2993, reason: "Default → GLU correction → exception → empirical basin → retreat is one evidence story." },
            { startSeconds: 3561, endSeconds: 3910, reason: "Regularization intuition → observed practice → optimizer interaction → recap is one counterintuitive turn." }
          ],
          [
            slot("L03-R06-FF", "interactive-demo", 2993, "An exact parameter-count calculator makes 4× versus 8/3 fair without separating the formula from its exceptions and evidence.", { compare: ["standard FFN", "GLU FFN"], contextRows: ["4×", "8/3×", "LLaMA", "T5"] }),
            slot("L03-R06-SHAPE", "table", 3561, "The head/aspect ratios are empirical bands with systems constraints and exceptions, best kept in the source model table.", { columns: ["observed ratio", "exception", "systems constraint", "evidence caveat"] }),
            slot("L03-R06-VOCAB", "table", 3561, "Adding coverage, embedding/output cost, and sequence length reconnects this observation to Lecture 1 without inventing a universal recommendation.", { crossLink: "L01-R08" }),
            slot("L03-R06-REG", "table", 3910, "The contradiction between classical intuition and observed weight decay only makes sense with the optimization-mechanism caveat visible.", { columns: ["intuition", "observed practice", "mechanism caveat"] })
          ],
          { restoredEvidence: ["Slide 36 hyperparameter question list", "Slide 51 hyperparameter recap"] }),

        run("L03-R07", "Stability: two softmax danger zones", 3910, 4446,
          "Motivate stability by failure cost, locate output and attention softmaxes separately, and compare z-loss, QK norm, and soft-capping with adoption and quality caveats.",
          ["lecture_03.pdf · slides 52–56", "video 65:10–74:06"],
          "Output-logit growth is addressed with z-loss on the log normalizer; attention-logit growth is addressed with QK normalization; soft-capping is a stronger bounded intervention that can trade quality for conservatism.",
          ["L03-Z-LOSS", "L03-QK-NORM", "L03-SOFT-CAP"],
          [{ startSeconds: 3910, endSeconds: 4446, reason: "Failure cost → two danger zones → three interventions → comparative caveat is one argument." }],
          [slot("L03-R07-STABILITY", "formula-comparison", 4446, "One model-location and formula table prevents three techniques from becoming context-free prescriptions.", { columns: ["model location", "formula", "controlled variable", "adoption", "quality caveat"] })]),

        run("L03-R08", "Why decoding changes attention", 4446, 4785,
          "Use the efficient-attention roadmap to establish the deployment problem, then compare prefill with token-by-token decode and show why a KV cache saves compute yet creates repeated memory traffic.",
          ["lecture_03.pdf · slides 57–60", "video 74:06–79:45"],
          "Dense attention interventions split into inference-cost and long-context branches; incremental generation appends Q/K/V while rereading cached K/V, changing arithmetic intensity relative to training and prefill.",
          ["L03-KV-CACHE"],
          [{ startSeconds: 4501, endSeconds: 4785, reason: "Deployment problem → prefill/decode accounting → cache traffic must precede head-sharing solutions." }],
          [
            slot("L03-R08-ROADMAP", "table", 4501, "Restoring slide 57 prevents the site from jumping directly into head diagrams and preserves the Lecture 4 boundary.", { branches: ["GQA/MQA: inference cost", "local/sliding: long-context cost"], deferred: "SSM and alternatives → Lecture 4" }),
            slot("L03-R08-CACHE", "interactive-demo", 4785, "Growing the cache and counting repeated reads makes the systems bottleneck visible before MQA/GQA appears.", { compare: ["training/prefill", "decode"], crossLink: "L02-R05" })
          ],
          { restoredEvidence: ["Slide 57 roadmap", "Slide 60 prefill/decode accounting"] }),

        run("L03-R09", "MQA to GQA: an efficiency-quality knob", 4785, 5108,
          "Present MQA as the single-K/V extreme, show its expressiveness/quality cost, then introduce GQA as a tunable compromise and read the latency-quality evidence.",
          ["lecture_03.pdf · slides 61–63", "video 79:45–85:08"],
          "Many Q heads first collapse to one shared K/V pair under MQA; grouping restores some K/V diversity; cache bytes and latency improve along the knob, but page 63 evidence and caveats determine how to interpret the trade.",
          ["L03-MQA", "L03-GQA"],
          [{ startSeconds: 4785, endSeconds: 5108, reason: "MQA extreme → quality cost → GQA compromise → evidence is the lecture's strongest efficient-attention argument." }],
          [
            slot("L03-R09-KV", "slow-manim", 5108, "The MHA→MQA→GQA morph is a real structural change, but only after the cache problem and with enough time to read each state.", { targetSeconds: "45–70 or linked steppers", artifact: "media/lecture_03/videos/lecture_03_scenes/480p15/L03KVSharing.mp4", artifactStatus: "rebuild", mountOnceFor: ["L03-MQA", "L03-GQA"] }),
            slot("L03-R09-EVIDENCE", "table", 5108, "Exact cache bytes and page 63 quality/latency evidence prevent the morph from implying that more sharing is unconditionally better.", { sourceSlide: 63 })
          ]),

        run("L03-R10", "Long-context hybrids and the return to the matrix", 5108, 5350,
          "Move from full causal attention to a local band, then show how periodic full layers propagate distant information, compare current hybrid patterns, and return to common versus variable architecture axes.",
          ["lecture_03.pdf · slides 64–67", "video 85:08–89:10"],
          "Sliding windows reduce the attended cells; layered local receptive fields expand only gradually; interleaved full attention enables global communication; model families choose different hybrids, so long context remains an active design axis.",
          ["L03-SLIDING-WINDOW", "L03-INTERLEAVED-ATTN"],
          [{ startSeconds: 5108, endSeconds: 5350, reason: "Full → local → interleave → current examples → lecture recap is one conclusion." }],
          [
            slot("L03-R10-WINDOW", "interactive-demo", 5350, "Window width and layered propagation are user-controlled relationships; an interactive can show both halves of the lecturer's conclusion.", { artifact: "media/lecture_03/videos/lecture_03_scenes/480p15/L03SlidingWindow.mp4", artifactStatus: "remove", controls: ["window width", "layer count", "insert full layer"] }),
            slot("L03-R10-MODELS", "table", 5350, "A model-family table grounds the hybrid claim and makes the final return to the architecture matrix explicit.", { finishAt: "common versus variable axes", restoreSlide: 67 })
          ],
          { restoredEvidence: ["Interleaved full-attention propagation", "Current model-family hybrid patterns", "Slide 67 global recap"] })
      ]
    }
  ];

  const anchorToRun = {};
  for (const lecture of lectures) {
    for (const lectureRun of lecture.runs) {
      for (const anchor of lectureRun.anchors) {
        if (anchorToRun[anchor]) throw new Error(`Duplicate lecture anchor: ${anchor}`);
        anchorToRun[anchor] = lectureRun.id;
      }
    }
  }

  const model = {
    version: 1,
    principle: "The official lecture video and source are the only narrative spine. Augmentations are optional, time-attached slots and never replacement chapters.",
    official,
    slotTypes,
    artifactStatuses,
    artifactInventory,
    lectures,
    anchorToRun
  };

  root.CS336_LECTURE_RUNS = model;
  if (typeof module !== "undefined" && module.exports) module.exports = model;
})(typeof window !== "undefined" ? window : globalThis);
