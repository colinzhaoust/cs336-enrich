# Lecture 2 arithmetic-intensity ledger

This is a standalone, silent browser interactive for the continuous Lecture 2 run from 40:30 to 57:10. It does not replace the official lecture and is not mounted by the shared site yet.

## Context card

```yaml
artifactId: L02-ROOFLINE-PAYOFF-V2
lecture: 2
source:
  videoUrl: https://www.youtube.com/watch?v=kuYAsz7zspQ
  interval: 40:30-57:10
  insertionAfter: 55:10
  sourcePages: []
  sourceLines: [lecture_02.py:338-481]
context:
  professorBefore: Percy has accumulated ReLU, GELU, dot, matvec, and matmul on one arithmetic-intensity ledger.
  professorAfter: Percy introduces roofline as the visualization of that ledger and relates it back to MFU.
  uninterruptedRun: ReLU -> GELU -> dot -> matvec -> matmul -> roofline
teachingQuestion: How does reuse move an operation from memory-bound to compute-bound on the same H100?
additiveClaim: The same visible FLOPs and bytes ledger computes every plot position, so the roofline is a traceable payoff rather than a new topic.
evidenceBoundary:
  lectureEvidence: Official source formulas, operation order, dense BF16 H100 peak, bandwidth, and ideal-overlap model.
  augmentationEvidence: Recomputed ratios, classifications, ceilings, and SVG coordinates; no benchmark measurements.
caveat: Counts model isolated BF16 operations and simplified HBM traffic; max(memory time, compute time) assumes ideal overlap and real overlap has overhead.
medium: responsive-interactive-ledger
audio: silent
expectedLearnerTimeSeconds: 60
```

## Source values

All arithmetic follows the inspected Spring 2026 `lecture_02.py`:

- overall run and hardware: lines 338–351;
- ReLU: lines 363–398;
- GELU: lines 400–415;
- dot product: lines 418–431;
- matrix–vector: lines 434–447;
- square matrix multiplication: lines 449–468;
- roofline payoff and MFU return: lines 471–481.

The hardware knee is a reproduced calculation:

```text
dense BF16 H100 peak = 1979 / 2 = 989.5 TFLOP/s
HBM bandwidth         = 3.35 TB/s
knee                  = 989.5 / 3.35 = 295.373... FLOP/byte
```

No value in this artifact is a measured kernel benchmark. The table labels formulas and source locations, and the evidence drawer preserves the ideal-overlap and isolated-operation caveats.

## Files

- `roofline_ledger.data.js`: source values, formulas, provenance, context, and step order.
- `roofline_ledger.js`: dependency-free renderer and mount API.
- `roofline_ledger.css`: component-scoped responsive styling.
- `roofline_ledger.demo.html`: standalone review page, not a shared-site integration.
- `roofline_ledger.test.mjs`: deterministic calculation and contract checks.

## Mount API

Load the data before the renderer:

```html
<link rel="stylesheet" href="roofline_ledger.css">
<div id="roofline-ledger"></div>
<script src="roofline_ledger.data.js"></script>
<script src="roofline_ledger.js"></script>
<script>
  const ledger = CS336RooflineLedger.mount("#roofline-ledger", {
    stepId: "relu",
    onStepChange({ step, operation, result }) {
      console.log(step.id, operation.id, result.intensity);
    }
  });

  ledger.setStep("roofline");
</script>
```

`mount()` returns `setStep(idOrIndex)`, `getState()`, and `destroy()`. The component does not modify global site styles or fetch assets.

## Interaction and accessibility

- `Tab` reaches the step tabs, operation-row controls, Previous/Next, source links, and evidence details.
- `Left`/`Right` move between adjacent steps; `Home`/`End` jump to the first/last step.
- The table is sufficient without the plot. At 390 px it becomes labeled vertical rows, while the SVG remains full width.
- The roofline is not rendered as the explanation until step 6, matching the 55:10 source payoff.
- No autoplay, motion dependency, canvas-only text, audio, or pointer-only gesture is used.

## Test

From `cs336_site`:

```bash
node augmentations_v2/lecture_02/roofline_ledger.test.mjs
```

The test recomputes every operation, the H100 knee, bottleneck classification, plot-coordinate bounds, source order, and the delayed roofline reveal.
