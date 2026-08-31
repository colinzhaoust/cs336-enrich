(function (root) {
  "use strict";

  const SVG_NS = "http://www.w3.org/2000/svg";
  const CHART = { width: 720, height: 380, left: 72, right: 26, top: 28, bottom: 58, xMin: 0.2, xMax: 600, yMin: 0.5, yMax: 1500 };

  function formatNumber(value, maximumFractionDigits = 2) {
    return new Intl.NumberFormat("en-US", { maximumFractionDigits }).format(value);
  }

  function formatSI(value, unit) {
    const scales = [[1e12, "T"], [1e9, "G"], [1e6, "M"], [1e3, "K"]];
    const scale = scales.find(([size]) => Math.abs(value) >= size);
    return scale ? `${formatNumber(value / scale[0], 3)} ${scale[1]}${unit}` : `${formatNumber(value, 3)} ${unit}`;
  }

  function calculateOperation(operation, hardware) {
    const intensity = operation.flops / operation.bytes;
    const memoryRoofFlopsPerSecond = intensity * hardware.bandwidthBytesPerSecond;
    const roofFlopsPerSecond = Math.min(hardware.peakFlopsPerSecond, memoryRoofFlopsPerSecond);
    return {
      intensity,
      memoryRoofFlopsPerSecond,
      roofFlopsPerSecond,
      bottleneck: intensity < hardware.peakFlopsPerSecond / hardware.bandwidthBytesPerSecond ? "memory-bound" : "compute-bound"
    };
  }

  function logScale(value, min, max, outputMin, outputMax) {
    const ratio = (Math.log10(value) - Math.log10(min)) / (Math.log10(max) - Math.log10(min));
    return outputMin + ratio * (outputMax - outputMin);
  }

  function plotPoint(intensity, performanceFlopsPerSecond) {
    const plotRight = CHART.width - CHART.right;
    const plotBottom = CHART.height - CHART.bottom;
    return {
      x: logScale(intensity, CHART.xMin, CHART.xMax, CHART.left, plotRight),
      y: logScale(performanceFlopsPerSecond / 1e12, CHART.yMin, CHART.yMax, plotBottom, CHART.top)
    };
  }

  function escapeHTML(value) {
    return String(value ?? "").replaceAll("&", "&amp;").replaceAll("<", "&lt;").replaceAll(">", "&gt;").replaceAll('"', "&quot;").replaceAll("'", "&#039;");
  }

  function rowTemplate(operation, index) {
    const result = calculateOperation(operation, data().hardware);
    return `<tr data-operation-row="${index}">
      <th scope="row" data-label="Operation"><button type="button" data-select-operation="${index}">${escapeHTML(operation.label)}</button><small>${escapeHTML(operation.sourceTime)}</small></th>
      <td data-label="FLOPs"><code>${escapeHTML(operation.formulas.flops)}</code><small>${formatSI(operation.flops, "FLOPs")}</small></td>
      <td data-label="Bytes moved"><code>${escapeHTML(operation.formulas.bytes)}</code><small>${formatSI(operation.bytes, "B")}</small></td>
      <td data-label="Intensity"><strong>${formatNumber(result.intensity, result.intensity < 10 ? 3 : 1)}</strong><small>FLOP / byte</small></td>
      <td data-label="Diagnosis"><span class="rl-diagnosis ${result.bottleneck}">${result.bottleneck}</span></td>
    </tr>`;
  }

  function chartTemplate(dataset) {
    const knee = dataset.kneeFlopsPerByte;
    const kneePoint = plotPoint(knee, dataset.hardware.peakFlopsPerSecond);
    const startPoint = plotPoint(CHART.xMin, CHART.xMin * dataset.hardware.bandwidthBytesPerSecond);
    const endPoint = plotPoint(CHART.xMax, dataset.hardware.peakFlopsPerSecond);
    const xTicks = [0.25, 0.5, 1, 5, 10, 100, 295, 500];
    const yTicks = [1, 10, 100, 1000];
    const points = dataset.operations.map((operation, index) => {
      const result = calculateOperation(operation, dataset.hardware);
      const point = plotPoint(result.intensity, result.roofFlopsPerSecond);
      return `<g class="rl-point" data-chart-point="${index}" transform="translate(${point.x.toFixed(2)} ${point.y.toFixed(2)})"><circle r="7"/><text x="0" y="-13" text-anchor="middle">${escapeHTML(operation.label)}</text></g>`;
    }).join("");
    return `<svg class="rl-chart" viewBox="0 0 ${CHART.width} ${CHART.height}" role="img" aria-labelledby="rl-chart-title rl-chart-desc">
      <title id="rl-chart-title">Dense BF16 H100 roofline derived from the visible ledger</title>
      <desc id="rl-chart-desc">A logarithmic plot. Each operation marker is computed from its FLOPs divided by bytes moved. The roofline knee is ${formatNumber(knee, 1)} FLOP per byte.</desc>
      <path class="rl-axis" d="M${CHART.left} ${CHART.top}V${CHART.height - CHART.bottom}H${CHART.width - CHART.right}"/>
      ${xTicks.map((tick) => { const x = logScale(tick, CHART.xMin, CHART.xMax, CHART.left, CHART.width - CHART.right); return `<g class="rl-tick"><path d="M${x} ${CHART.height - CHART.bottom}v7"/><text x="${x}" y="${CHART.height - 31}" text-anchor="middle">${tick}</text></g>`; }).join("")}
      ${yTicks.map((tick) => { const y = logScale(tick, CHART.yMin, CHART.yMax, CHART.height - CHART.bottom, CHART.top); return `<g class="rl-tick"><path d="M${CHART.left - 7} ${y}h7"/><text x="${CHART.left - 11}" y="${y + 4}" text-anchor="end">${tick}</text></g>`; }).join("")}
      <path class="rl-memory-roof" d="M${startPoint.x} ${startPoint.y}L${kneePoint.x} ${kneePoint.y}"/>
      <path class="rl-compute-roof" d="M${kneePoint.x} ${kneePoint.y}L${endPoint.x} ${endPoint.y}"/>
      <path class="rl-knee" d="M${kneePoint.x} ${kneePoint.y}V${CHART.height - CHART.bottom}"/>
      <text class="rl-roof-label" x="${(startPoint.x + kneePoint.x) / 2}" y="${(startPoint.y + kneePoint.y) / 2 - 13}">bandwidth roof</text>
      <text class="rl-roof-label" x="${kneePoint.x + 74}" y="${kneePoint.y - 13}">dense BF16 compute roof</text>
      <text class="rl-axis-label" x="${(CHART.left + CHART.width - CHART.right) / 2}" y="${CHART.height - 7}" text-anchor="middle">arithmetic intensity (FLOP / byte, log scale)</text>
      <text class="rl-axis-label" transform="translate(17 ${(CHART.top + CHART.height - CHART.bottom) / 2}) rotate(-90)" text-anchor="middle">roofline performance (TFLOP/s, log scale)</text>
      ${points}
    </svg>`;
  }

  function data() {
    const dataset = root.CS336_ROOFLINE_LEDGER_DATA;
    if (!dataset) throw new Error("Load roofline_ledger.data.js before roofline_ledger.js");
    return dataset;
  }

  function shellTemplate(dataset) {
    return `<section class="roofline-ledger" aria-labelledby="rl-title">
      <header class="rl-header">
        <div><p class="rl-eyebrow">Lecture 2 · 40:30–57:10 · interactive augmentation</p><h2 id="rl-title">Arithmetic intensity, kept on one ledger</h2><p>${escapeHTML(dataset.teachingQuestion)}</p></div>
        <div class="rl-source"><span>Official source</span><a href="${dataset.source.videoUrl}&t=2430s" target="_blank" rel="noreferrer">Watch from 40:30 ↗</a><a href="${dataset.source.sourceUrl}" target="_blank" rel="noreferrer">lecture_02.py lines 338–481 ↗</a></div>
      </header>
      <p class="rl-why"><strong>Why this is here:</strong> ${escapeHTML(dataset.additiveClaim)}</p>
      <nav class="rl-steps" role="tablist" aria-label="Lecture operation sequence">${dataset.steps.map((step, index) => `<button type="button" role="tab" aria-selected="${index === 0}" tabindex="${index === 0 ? 0 : -1}" data-step-index="${index}">${escapeHTML(step.label)}</button>`).join("")}</nav>
      <div class="rl-table-wrap"><table><caption>Rows accumulate in Percy's lecture order. Values are calculations, not benchmarks.</caption><thead><tr><th>Operation</th><th>FLOPs</th><th>Bytes moved</th><th>Intensity</th><th>Diagnosis</th></tr></thead><tbody>${dataset.operations.map(rowTemplate).join("")}</tbody></table></div>
      <div class="rl-inspector">
        <section class="rl-derivation" aria-labelledby="rl-derivation-title"><p class="rl-label">Selected derivation</p><h3 id="rl-derivation-title"></h3><div class="rl-equations"></div><dl class="rl-values"></dl><p class="rl-operation-source"></p><p class="rl-caveat"></p></section>
        <section class="rl-payoff" aria-labelledby="rl-payoff-title"><p class="rl-label">55:10 payoff</p><h3 id="rl-payoff-title">Roofline stays hidden until the ledger is complete</h3><div class="rl-locked"><strong>Five rows first.</strong><p>The plot will reuse the visible ratios instead of introducing hand-positioned points.</p></div><div class="rl-chart-wrap" hidden>${chartTemplate(dataset)}<div class="rl-hardware"><span><strong>${formatNumber(dataset.hardware.peakFlopsPerSecond / 1e12, 1)} TFLOP/s</strong> dense BF16 peak</span><span><strong>${formatNumber(dataset.hardware.bandwidthBytesPerSecond / 1e12, 2)} TB/s</strong> HBM bandwidth</span><span><strong>${formatNumber(dataset.kneeFlopsPerByte, 1)} FLOP/B</strong> knee</span></div></div></section>
      </div>
      <div class="rl-controls"><button type="button" data-previous>Previous</button><p aria-live="polite"><span>Step 1 of ${dataset.steps.length}</span><strong>ReLU joins the ledger.</strong></p><button type="button" data-next>Next</button></div>
      <details class="rl-method"><summary>Assumptions and evidence boundary</summary><div><p><strong>Lecture evidence.</strong> ${escapeHTML(dataset.evidenceBoundary.lectureEvidence)}</p><p><strong>Our calculation.</strong> ${escapeHTML(dataset.evidenceBoundary.augmentationEvidence)}</p><ul>${dataset.caveats.map((caveat) => `<li>${escapeHTML(caveat)}</li>`).join("")}</ul></div></details>
    </section>`;
  }

  function mount(target, options = {}) {
    const element = typeof target === "string" ? document.querySelector(target) : target;
    if (!element) throw new Error("CS336RooflineLedger.mount requires a target element");
    const dataset = data();
    element.innerHTML = shellTemplate(dataset);
    const rootElement = element.querySelector(".roofline-ledger");
    let currentStep = Math.max(0, dataset.steps.findIndex((step) => step.id === options.stepId));
    if (currentStep < 0) currentStep = 0;
    let selectedOperationIndex = dataset.steps[currentStep].operationIndex;

    function render() {
      const step = dataset.steps[currentStep];
      const selectedIndex = step.rooflineVisible ? selectedOperationIndex : step.operationIndex;
      const operation = dataset.operations[selectedIndex];
      const result = calculateOperation(operation, dataset.hardware);
      rootElement.querySelectorAll("[data-step-index]").forEach((button, index) => {
        const selected = index === currentStep;
        button.setAttribute("aria-selected", String(selected));
        button.tabIndex = selected ? 0 : -1;
      });
      rootElement.querySelectorAll("[data-operation-row]").forEach((row, index) => {
        row.hidden = !step.rooflineVisible && index > selectedIndex;
        row.classList.toggle("is-selected", index === selectedIndex);
      });
      rootElement.querySelectorAll("[data-chart-point]").forEach((point, index) => point.classList.toggle("is-selected", index === selectedIndex));
      rootElement.querySelector("#rl-derivation-title").textContent = operation.label;
      rootElement.querySelector(".rl-equations").innerHTML = `<code>${escapeHTML(operation.formulas.flops)} FLOPs</code><code>${escapeHTML(operation.formulas.bytes)}</code><code>${escapeHTML(operation.formulas.intensity)}</code>`;
      rootElement.querySelector(".rl-values").innerHTML = `<div><dt>Computed intensity</dt><dd>${formatNumber(result.intensity, 3)} FLOP / byte</dd></div><div><dt>H100 diagnosis</dt><dd>${result.bottleneck}</dd></div><div><dt>Roofline ceiling</dt><dd>${formatNumber(result.roofFlopsPerSecond / 1e12, 2)} TFLOP/s</dd></div>`;
      rootElement.querySelector(".rl-operation-source").innerHTML = `<strong>Source:</strong> ${escapeHTML(operation.sourceTime)}, lecture_02.py lines ${escapeHTML(operation.sourceLines)}. ${escapeHTML(operation.provenance)}.`;
      rootElement.querySelector(".rl-caveat").innerHTML = `<strong>Caveat:</strong> ${escapeHTML(operation.caveat)}`;
      rootElement.querySelector("#rl-payoff-title").textContent = step.rooflineVisible ? `Dense BF16 H100 roofline, knee ≈ ${formatNumber(dataset.kneeFlopsPerByte, 1)} FLOP/byte` : "Roofline stays hidden until the ledger is complete";
      rootElement.querySelector(".rl-locked").hidden = step.rooflineVisible;
      rootElement.querySelector(".rl-chart-wrap").hidden = !step.rooflineVisible;
      rootElement.querySelector("[data-previous]").disabled = currentStep === 0;
      rootElement.querySelector("[data-next]").disabled = currentStep === dataset.steps.length - 1;
      const status = rootElement.querySelector(".rl-controls p");
      status.querySelector("span").textContent = `Step ${currentStep + 1} of ${dataset.steps.length}`;
      status.querySelector("strong").textContent = step.rooflineVisible ? `Roofline revealed at 55:10; ${operation.label} traces back to its ledger row.` : `${operation.label} joins the persistent ledger.`;
      options.onStepChange?.({ step, operation, result });
    }

    function setStep(value, focus = false) {
      const index = typeof value === "number" ? value : dataset.steps.findIndex((step) => step.id === value);
      if (index < 0 || index >= dataset.steps.length) return false;
      currentStep = index;
      selectedOperationIndex = dataset.steps[index].operationIndex;
      render();
      if (focus) rootElement.querySelector(`[data-step-index="${currentStep}"]`)?.focus();
      return true;
    }

    function onClick(event) {
      const stepButton = event.target.closest("[data-step-index]");
      if (stepButton) { setStep(Number(stepButton.dataset.stepIndex)); return; }
      if (event.target.closest("[data-previous]")) { setStep(currentStep - 1); return; }
      if (event.target.closest("[data-next]")) { setStep(currentStep + 1); return; }
      const rowButton = event.target.closest("[data-select-operation]");
      if (rowButton) {
        const operationIndex = Number(rowButton.dataset.selectOperation);
        if (dataset.steps[currentStep].rooflineVisible) { selectedOperationIndex = operationIndex; render(); }
        else setStep(operationIndex);
      }
    }

    function onKeydown(event) {
      const tab = event.target.closest('[role="tab"]');
      if (!tab) return;
      const map = { ArrowRight: currentStep + 1, ArrowLeft: currentStep - 1, Home: 0, End: dataset.steps.length - 1 };
      if (!(event.key in map)) return;
      event.preventDefault();
      setStep(Math.max(0, Math.min(dataset.steps.length - 1, map[event.key])), true);
    }

    rootElement.addEventListener("click", onClick);
    rootElement.addEventListener("keydown", onKeydown);
    render();
    return { setStep, getState: () => ({ stepIndex: currentStep, step: dataset.steps[currentStep], selectedOperation: dataset.operations[selectedOperationIndex] }), destroy: () => { rootElement.removeEventListener("click", onClick); rootElement.removeEventListener("keydown", onKeydown); element.replaceChildren(); } };
  }

  const api = { mount, calculateOperation, plotPoint, formatNumber, formatSI };
  root.CS336RooflineLedger = api;
  if (typeof module !== "undefined" && module.exports) module.exports = api;
})(typeof window !== "undefined" ? window : globalThis);
