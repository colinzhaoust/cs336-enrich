(function (root) {
  "use strict";

  const nVector = 1024 * 1024;
  const nMatrix = 1024;
  const hardware = {
    name: "NVIDIA H100",
    dtype: "BF16",
    mode: "dense",
    peakFlopsPerSecond: 1979e12 / 2,
    bandwidthBytesPerSecond: 3.35e12,
    provenance: "Lecture value",
    source: "lecture_02.py lines 350–351"
  };

  function operation(id, label, sourceTime, sourceLines, n, flops, bytes, formulas, caveat) {
    return { id, label, sourceTime, sourceLines, n, flops, bytes, formulas, caveat, intensity: flops / bytes, provenance: "Reproduced from official source arithmetic" };
  }

  const operations = [
    operation("relu", "ReLU", "42:25–47:50", "363–398", nVector,
      nVector, 4 * nVector,
      { flops: "n", bytes: "2n read + 2n write = 4n", intensity: "n / 4n = 0.25" },
      "Isolated BF16 elementwise operation. The perfect-overlap timing model is idealized."),
    operation("gelu", "GELU", "47:50–49:35", "400–415", nVector,
      20 * nVector, 4 * nVector,
      { flops: "≈20n", bytes: "2n read + 2n write = 4n", intensity: "20n / 4n ≈ 5" },
      "The lecture uses about 20 operations; the tanh approximation and implementation can vary."),
    operation("dot", "Dot product", "49:35–50:35", "418–431", nVector,
      2 * nVector - 1, 4 * nVector + 2,
      { flops: "2n − 1", bytes: "2n + 2n read + 2 write", intensity: "(2n − 1) / (4n + 2) ≈ 0.5" },
      "The scalar output is counted as one BF16 write, matching the official teaching model."),
    operation("matvec", "Matrix–vector", "50:35–51:25", "434–447", nMatrix,
      nMatrix * (2 * nMatrix - 1), 2 * nMatrix + 2 * nMatrix * nMatrix + 2 * nMatrix,
      { flops: "n(2n − 1)", bytes: "2n vector + 2n² matrix + 2n output", intensity: "n(2n − 1) / (2n² + 4n) ≈ 1" },
      "This single-vector case has little weight reuse across outputs and models decode-like work."),
    operation("matmul", "Square matmul", "51:25–55:10", "449–468", nMatrix,
      nMatrix * nMatrix * (2 * nMatrix - 1), 6 * nMatrix * nMatrix,
      { flops: "n²(2n − 1)", bytes: "2n² + 2n² read + 2n² write", intensity: "n²(2n − 1) / 6n² ≈ n/3" },
      "The ledger assumes each matrix is moved once from HBM. Real kernels tile and reuse data; precision also changes the knee.")
  ];

  const data = {
    artifactId: "L02-ROOFLINE-PAYOFF-V2",
    version: 1,
    source: {
      videoUrl: "https://www.youtube.com/watch?v=kuYAsz7zspQ",
      interval: "40:30–57:10",
      insertionAfter: "55:10",
      sourceFile: "lecture_02.py",
      sourceLines: "338–481",
      sourceUrl: "https://github.com/stanford-cs336/lectures/blob/main/lecture_02.py#L338-L481"
    },
    context: {
      professorBefore: "Percy has built an ordered ladder of operations whose data reuse increases from elementwise work to large matrix multiplication.",
      professorAfter: "Percy introduces roofline as the visualization of the same ledger and relates it back to MFU.",
      uninterruptedRun: "ReLU → GELU → dot → matvec → matmul → roofline"
    },
    teachingQuestion: "How does reuse move an operation from memory-bound to compute-bound on the same H100?",
    additiveClaim: "The same visible FLOPs and bytes ledger computes every x-position, making the roofline payoff traceable rather than a new unexplained plot.",
    evidenceBoundary: {
      lectureEvidence: "Official source formulas, H100 dense BF16 peak, HBM bandwidth, operation order, and ideal-overlap model.",
      augmentationEvidence: "Exact recomputation of the displayed ratios and SVG coordinates; no measured kernel benchmarks."
    },
    caveats: [
      "max(t_memory, t_compute) assumes perfect overlap; real overlap has overhead.",
      "Counts model isolated BF16 operations and simplified HBM traffic, not measured end-to-end kernels.",
      "Compute-bound means reduce required FLOPs or raise effective compute throughput when the operation and hardware allow it; it does not imply a universal optimization recipe."
    ],
    hardware,
    kneeFlopsPerByte: hardware.peakFlopsPerSecond / hardware.bandwidthBytesPerSecond,
    operations,
    revealAtStep: 5,
    steps: [
      { id: "relu", label: "1 · ReLU", operationIndex: 0, rooflineVisible: false },
      { id: "gelu", label: "2 · GELU", operationIndex: 1, rooflineVisible: false },
      { id: "dot", label: "3 · Dot", operationIndex: 2, rooflineVisible: false },
      { id: "matvec", label: "4 · Matvec", operationIndex: 3, rooflineVisible: false },
      { id: "matmul", label: "5 · Matmul", operationIndex: 4, rooflineVisible: false },
      { id: "roofline", label: "6 · Roofline payoff", operationIndex: 4, rooflineVisible: true }
    ]
  };

  root.CS336_ROOFLINE_LEDGER_DATA = data;
  if (typeof module !== "undefined" && module.exports) module.exports = data;
})(typeof window !== "undefined" ? window : globalThis);
