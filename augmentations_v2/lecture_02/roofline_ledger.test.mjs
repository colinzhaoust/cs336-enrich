import assert from "node:assert/strict";
import fs from "node:fs";
import vm from "node:vm";

const directory = new URL("./", import.meta.url);
const context = { globalThis: {}, Intl };
context.globalThis = context;
vm.createContext(context);
vm.runInContext(fs.readFileSync(new URL("roofline_ledger.data.js", directory), "utf8"), context);
vm.runInContext(fs.readFileSync(new URL("roofline_ledger.js", directory), "utf8"), context);

const data = context.CS336_ROOFLINE_LEDGER_DATA;
const api = context.CS336RooflineLedger;
assert.ok(data, "data module should expose the ledger fixture");
assert.ok(api, "renderer should expose its calculation API without mounting");

assert.equal(data.source.interval, "40:30–57:10");
assert.equal(data.source.insertionAfter, "55:10");
assert.equal(data.source.sourceLines, "338–481");
assert.deepEqual(Array.from(data.operations, (operation) => operation.id), ["relu", "gelu", "dot", "matvec", "matmul"]);
assert.equal(data.steps.at(-1).id, "roofline");
assert.equal(data.steps.at(-1).rooflineVisible, true);
assert.ok(data.steps.slice(0, -1).every((step) => step.rooflineVisible === false), "roofline must stay hidden until the payoff step");

const knee = data.hardware.peakFlopsPerSecond / data.hardware.bandwidthBytesPerSecond;
assert.ok(Math.abs(knee - 295.3731343283582) < 1e-10);
assert.equal(data.kneeFlopsPerByte, knee);

const results = Object.fromEntries(data.operations.map((operation) => [operation.id, api.calculateOperation(operation, data.hardware)]));
assert.equal(results.relu.intensity, 0.25);
assert.equal(results.gelu.intensity, 5);
assert.ok(Math.abs(results.dot.intensity - 0.49999952316306917) < 1e-12);
assert.ok(Math.abs(results.matvec.intensity - 0.9975633528265108) < 1e-12);
assert.ok(Math.abs(results.matmul.intensity - 341.1666666666667) < 1e-12);
assert.deepEqual([results.relu.bottleneck, results.gelu.bottleneck, results.dot.bottleneck, results.matvec.bottleneck, results.matmul.bottleneck], ["memory-bound", "memory-bound", "memory-bound", "memory-bound", "compute-bound"]);
assert.equal(results.matmul.roofFlopsPerSecond, data.hardware.peakFlopsPerSecond);
assert.ok(results.gelu.roofFlopsPerSecond < data.hardware.peakFlopsPerSecond);

for (const operation of data.operations) {
  const result = api.calculateOperation(operation, data.hardware);
  const point = api.plotPoint(result.intensity, result.roofFlopsPerSecond);
  assert.ok(Number.isFinite(point.x) && Number.isFinite(point.y), `${operation.id} needs finite plot coordinates`);
  assert.ok(point.x >= 72 && point.x <= 694, `${operation.id} x coordinate should be inside the plot`);
  assert.ok(point.y >= 28 && point.y <= 322, `${operation.id} y coordinate should be inside the plot`);
  assert.match(operation.sourceTime, /^\d{2}:\d{2}–\d{2}:\d{2}$/);
  assert.match(operation.sourceLines, /^\d+–\d+$/);
  assert.equal(operation.provenance, "Reproduced from official source arithmetic");
}

const rendererSource = fs.readFileSync(new URL("roofline_ledger.js", directory), "utf8");
assert.doesNotMatch(rendererSource, /\.mp4|<video|autoplay/i);
assert.match(rendererSource, /ArrowRight/);
assert.match(rendererSource, /ArrowLeft/);
assert.match(rendererSource, /Home/);
assert.match(rendererSource, /End/);

console.log("Validated the Lecture 2 arithmetic-intensity ledger: five source-ordered calculations, delayed roofline payoff, H100 knee, classifications, plot coordinates, provenance, and keyboard contract.");
