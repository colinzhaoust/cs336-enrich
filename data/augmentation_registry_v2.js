(function (root) {
  "use strict";

  const artifacts = [
    {
      id: "L01-BPE-AFTER-TRACE-V2",
      status: "finished",
      lecture: 1,
      runId: "L01-R10",
      slotId: "L01-R10-TRAIN-USE",
      supersedesSlots: ["L01-R09-REPLAY"],
      placement: {
        sourceInterval: "73:16–76:01",
        insertionAfter: "75:22",
        label: "After Percy's uninterrupted BPE training trace"
      },
      title: "Replay frozen BPE rules on held-out text",
      typeLabel: "Interactive stepper",
      caption: "Percy traces how BPE learns three ordered rules, then intentionally skips the held-out encode steps. This replay shows that encoding reuses the frozen rule order; it does not train again.",
      alt: "An ordered three-rule BPE merge table is learned on ‘the cat in the hat’ and frozen. On held-out ‘the quick brown fox,’ the encoder replays the rules, reducing 19 byte tokens to 16 without recounting pairs or changing the vocabulary.",
      provenance: "Official teaching logic reproduced from lecture_01.py lines 527–564, 705–720, and 729–757; fixture assertions cover rule order, token lengths, vocabulary growth, and decode round trip.",
      caveat: "This is the deliberately slow teaching encoder. Production implementations optimize applicable-merge lookup and add pretokenization and special-token handling while preserving merge priority.",
      sources: [
        { label: "Watch the original trace", href: "https://www.youtube.com/watch?v=JuoVZkPBiKk&t=4396s" },
        { label: "Official lecture code", href: "https://github.com/stanford-cs336/lectures/blob/main/lecture_01.py#L527-L564" },
        { label: "Visual transcript", href: "augmentations_v2/lecture_01/L01-BPE-AFTER-TRACE-V2.transcript.md" }
      ],
      primary: {
        kind: "iframe",
        path: "augmentations_v2/lecture_01/L01-BPE-AFTER-TRACE-V2.stepper.html",
        title: "Interactive BPE training versus encoding replay",
        defaultState: "held-out-bytes"
      },
      optionalVideo: {
        path: "media_v2/lecture_01/L01-BPE-AFTER-TRACE-V2.mp4",
        poster: "media_v2/lecture_01/L01-BPE-AFTER-TRACE-V2-poster.png",
        descriptionTrack: "media_v2/lecture_01/L01-BPE-AFTER-TRACE-V2.vtt",
        durationLabel: "40 seconds · silent · 1080p",
        title: "Optional slow linear replay",
        alt: "Slow linear version of the same frozen-rule replay shown in the interactive stepper."
      },
      anchorTargets: {
        "L01-BPE-PAIR-COUNT": "training-result",
        "L01-BPE-MERGE": "replay-rank-0",
        "L01-BPE-MERGE-CODE": "replay-rank-0",
        "L01-BPE-TRAIN-VS-USE": "state-ledger"
      }
    },
    {
      id: "L02-ROOFLINE-PAYOFF-V2",
      status: "finished",
      lecture: 2,
      runId: "L02-R05",
      slotId: "L02-R05-ROOFLINE",
      supersedesSlots: ["L02-R05-LEDGER"],
      placement: {
        sourceInterval: "40:30–57:10",
        insertionAfter: "55:10",
        label: "One continuous arithmetic-intensity run; roofline appears only at step 6"
      },
      title: "Arithmetic intensity, kept on one ledger",
      typeLabel: "Interactive ledger",
      caption: "Accumulate ReLU, GELU, dot, matvec, and matmul from the same FLOPs and bytes ledger. The dense BF16 H100 roofline remains hidden until Percy's 55:10 payoff.",
      alt: "A six-step arithmetic-intensity ledger adds five operations in lecture order, then reveals a roofline plot whose markers are computed from the visible FLOPs and bytes values.",
      provenance: "Official formulas and hardware values from lecture_02.py lines 338–481. Ratios, bottleneck classifications, and plot coordinates are reproduced calculations, not benchmark measurements.",
      caveat: "Counts model isolated BF16 operations and simplified HBM traffic. max(memory time, compute time) assumes ideal overlap; real kernels have additional overhead.",
      sources: [
        { label: "Watch from 40:30", href: "https://www.youtube.com/watch?v=kuYAsz7zspQ&t=2430s" },
        { label: "Official lecture code", href: "https://github.com/stanford-cs336/lectures/blob/main/lecture_02.py#L338-L481" },
        { label: "Artifact notes", href: "augmentations_v2/lecture_02/README.md" }
      ],
      primary: {
        kind: "mount",
        styles: ["augmentations_v2/lecture_02/roofline_ledger.css"],
        scripts: [
          "augmentations_v2/lecture_02/roofline_ledger.data.js",
          "augmentations_v2/lecture_02/roofline_ledger.js"
        ],
        global: "CS336RooflineLedger",
        method: "mount",
        defaultState: "relu"
      },
      selfDescribing: true,
      anchorTargets: {
        "L02-COMPUTE-MEMORY": "relu",
        "L02-RELU-INTENSITY": "relu",
        "L02-GELU-INTENSITY": "gelu",
        "L02-DOT-INTENSITY": "dot",
        "L02-MATVEC-INTENSITY": "matvec",
        "L02-MATMUL-INTENSITY": "matmul",
        "L02-ROOFLINE": "roofline"
      }
    },
    {
      id: "L03-KV-DECODE-EVIDENCE-V2",
      status: "finished",
      lecture: 3,
      runId: "L03-R09",
      slotId: "L03-R09-KV",
      supersedesSlots: ["L03-R08-CACHE", "L03-R09-EVIDENCE"],
      placement: {
        sourceInterval: "74:06–84:59",
        insertionAfter: "85:08",
        label: "After the uninterrupted slides 57–63 argument"
      },
      title: "Decode traffic, KV sharing, and the evidence boundary",
      typeLabel: "Interactive companion",
      caption: "Slides 57–63 form one causal chain: serial decode rereads a growing KV cache, fewer KV heads reduce those bytes, and slide 63 bounds the systems argument with empirical quality evidence.",
      alt: "A seven-state companion moves from serial decode and growing KV-cache reads to an exact byte ledger, then keeps eight query heads fixed while K/V heads change from eight to one or grouped values; the final state returns to slide 63's setting-specific quality evidence.",
      provenance: "Official claims and evidence order follow lecture_03.pdf slides 57–63. Declared cache totals are reproduced dimensional calculations with fixture checks; slide 63 quality values remain labeled empirical.",
      caveat: "Declared cache bytes scale with head count, head dimension, dtype, sequence, layers, and batch. Real latency also depends on kernels, batching, hardware, layout, and other traffic; slide 63 observations are not universal guarantees.",
      sources: [
        { label: "Watch the original run", href: "https://www.youtube.com/watch?v=lVynu4bo1rY&t=4446s" },
        { label: "Official deck · slides 57–63", href: "https://github.com/stanford-cs336/lectures/blob/main/lecture_03.pdf" },
        { label: "Visual transcript", href: "augmentations_v2/lecture_03/L03-KV-DECODE-EVIDENCE-V2.transcript.md" }
      ],
      primary: {
        kind: "iframe",
        path: "augmentations_v2/lecture_03/L03-KV-DECODE-EVIDENCE-V2.companion.html",
        title: "Interactive decode traffic and KV-sharing evidence companion",
        defaultState: "prefill-vs-decode"
      },
      optionalVideo: {
        path: "media_v2/lecture_03/L03-KV-DECODE-EVIDENCE-V2.mp4",
        poster: "media_v2/lecture_03/L03-KV-DECODE-EVIDENCE-V2-poster.png",
        descriptionTrack: "media_v2/lecture_03/L03-KV-DECODE-EVIDENCE-V2.vtt",
        durationLabel: "55 seconds · silent · 1080p",
        title: "Optional slow linear replay",
        alt: "Slow linear version of the decode traffic, KV-sharing, and empirical-evidence sequence shown in the interactive companion."
      },
      anchorTargets: {
        "L03-KV-CACHE": "growing-kv-read",
        "L03-MQA": "mqa-extreme",
        "L03-GQA": "gqa-knob"
      }
    }
  ];

  const byId = Object.fromEntries(artifacts.map((artifact) => [artifact.id, artifact]));
  const bySlot = Object.fromEntries(artifacts.map((artifact) => [artifact.slotId, artifact]));
  const byAnchor = {};
  for (const artifact of artifacts) {
    for (const [anchorId, stateId] of Object.entries(artifact.anchorTargets || {})) {
      byAnchor[anchorId] = { artifactId: artifact.id, runId: artifact.runId, slotId: artifact.slotId, stateId };
    }
  }

  root.CS336_AUGMENTATION_REGISTRY_V2 = {
    version: 2,
    artifacts,
    byId,
    bySlot,
    byAnchor,
    supersededSlots: [...new Set(artifacts.flatMap((artifact) => artifact.supersedesSlots || []))]
  };
})(typeof window !== "undefined" ? window : globalThis);
