import fs from "node:fs";
import vm from "node:vm";
import { spawnSync } from "node:child_process";
import path from "node:path";
import { createHash } from "node:crypto";

const requiredFiles = [
  "index.html", "styles.css", "app.js", "config.js", "data/lectures.js",
  "data/lecture_runs.js", "data/augmentation_registry_v2.js", "README.md", ".gitignore", ".github/workflows/pages.yml",
  "augmentations_v2/lecture_01/L01-BPE-AFTER-TRACE-V2.stepper.html",
  "augmentations_v2/lecture_02/roofline_ledger.data.js",
  "augmentations_v2/lecture_02/roofline_ledger.js",
  "augmentations_v2/lecture_02/roofline_ledger.css",
  "augmentations_v2/lecture_03/L03-KV-DECODE-EVIDENCE-V2.companion.html"
];
for (const file of requiredFiles) if (!fs.existsSync(file)) throw new Error(`Missing required file: ${file}`);

const context = { window: {}, globalThis: {} };
context.globalThis = context;
vm.createContext(context);
vm.runInContext(fs.readFileSync("data/lectures.js", "utf8"), context);
vm.runInContext(fs.readFileSync("data/lecture_runs.js", "utf8"), context);
vm.runInContext(fs.readFileSync("data/augmentation_registry_v2.js", "utf8"), context);

const legacy = context.window.CS336_LECTURES;
const model = context.window.CS336_LECTURE_RUNS;
const registry = context.window.CS336_AUGMENTATION_REGISTRY_V2;
if (!Array.isArray(legacy) || legacy.length !== 3) throw new Error("Expected three legacy lectures");
if (!model || model.version !== 1 || model.lectures.length !== 3) throw new Error("Expected the lecture-run model");
if (!registry || registry.version !== 2 || registry.artifacts.length !== 3) throw new Error("Expected three finished V2 augmentations");

for (const manifestPath of ["augmentations_v2/lecture_01/manifest.json", "augmentations_v2/lecture_03/manifest.json"]) {
  const manifest = JSON.parse(fs.readFileSync(manifestPath, "utf8"));
  if (manifest.status !== "locally-integrated-awaiting-user-acceptance") throw new Error(`${manifest.artifactId} has stale integration status`);
  for (const [pathField, shaField] of [["scene", "sceneSha256"], ["fixtureSource", "fixtureSourceSha256"]]) {
    const relative = manifest.reproducibility?.[pathField];
    const expectedSha = manifest.reproducibility?.[shaField];
    if (!relative || !expectedSha) throw new Error(`${manifest.artifactId} is missing ${pathField} reproducibility metadata`);
    const sourcePath = path.resolve(path.dirname(manifestPath), relative);
    if (!fs.existsSync(sourcePath) || !sourcePath.includes(`${path.sep}scenes${path.sep}cs336_v2${path.sep}`)) throw new Error(`${manifest.artifactId} ${pathField} is not self-contained`);
    const actualSha = createHash("sha256").update(fs.readFileSync(sourcePath)).digest("hex");
    if (actualSha !== expectedSha) throw new Error(`${manifest.artifactId} ${pathField} SHA mismatch`);
  }
}

function isGitIgnored(path) {
  return spawnSync("git", ["check-ignore", "--no-index", "-q", path], { stdio: "ignore" }).status === 0;
}

const legacyIds = new Set(legacy.flatMap((lecture) => lecture.segments.map((segment) => segment.id)));
const runAnchors = new Set(Object.keys(model.anchorToRun));
if (legacyIds.size !== 88) throw new Error(`Expected 88 legacy IDs, found ${legacyIds.size}`);
if (runAnchors.size !== legacyIds.size || [...legacyIds].some((id) => !runAnchors.has(id))) {
  throw new Error("Lecture-run anchors must match the 88 legacy IDs exactly");
}

const runIndex = new Map(model.lectures.flatMap((lecture) => lecture.runs.map((run) => [run.id, run])));
const artifactIds = new Set();
const artifactSlots = new Set();
for (const artifact of registry.artifacts) {
  if (artifact.status !== "finished") throw new Error(`${artifact.id} must be explicitly finished`);
  if (artifactIds.has(artifact.id) || artifactSlots.has(artifact.slotId)) throw new Error(`Duplicate artifact or slot: ${artifact.id}`);
  artifactIds.add(artifact.id); artifactSlots.add(artifact.slotId);
  const run = runIndex.get(artifact.runId);
  if (!run || !run.augmentationSlots.some((slot) => slot.id === artifact.slotId)) throw new Error(`${artifact.id} points to a missing run/slot`);
  if (!artifact.caption || !artifact.alt || !artifact.provenance || !artifact.caveat || !artifact.sources?.length) throw new Error(`${artifact.id} is missing visible metadata`);
  for (const path of [...(artifact.primary.styles || []), ...(artifact.primary.scripts || []), artifact.primary.path, artifact.optionalVideo?.path, artifact.optionalVideo?.poster, artifact.optionalVideo?.descriptionTrack].filter(Boolean)) {
    if (!fs.existsSync(path)) throw new Error(`${artifact.id} references missing asset ${path}`);
    if (path.startsWith("media_v2/") && isGitIgnored(path)) throw new Error(`${artifact.id} publication asset is ignored by git: ${path}`);
  }
  for (const source of artifact.sources.filter((item) => !/^https?:/.test(item.href))) if (!fs.existsSync(source.href)) throw new Error(`${artifact.id} references missing source ${source.href}`);
  for (const anchor of Object.keys(artifact.anchorTargets || {})) if (!runAnchors.has(anchor)) throw new Error(`${artifact.id} references unknown stable anchor ${anchor}`);
}
for (const qaPath of [
  "media_v2/lecture_01/L01-BPE-AFTER-TRACE-V2-contact-sheet.png",
  "media_v2/lecture_01/animatics/L01-BPE-AFTER-TRACE-V2-animatic.mp4",
  "media_v2/lecture_03/L03-KV-DECODE-EVIDENCE-V2-contact-sheet.png",
  "media_v2/lecture_03/animatics/L03-KV-DECODE-EVIDENCE-V2-animatic.mp4"
]) if (!isGitIgnored(qaPath)) throw new Error(`QA-only artifact must remain ignored: ${qaPath}`);
if (registry.bySlot["L01-R10-TRAIN-USE"]?.placement.insertionAfter !== "75:22") throw new Error("L1 BPE replay must be mounted after 75:22");
if (registry.bySlot["L02-R05-ROOFLINE"]?.placement.sourceInterval !== "40:30–57:10") throw new Error("L2 ledger must cover the continuous 40:30–57:10 run");
if (registry.byAnchor["L02-ROOFLINE"]?.stateId !== "roofline") throw new Error("L2 roofline deep link must reveal step 6");
if (registry.bySlot["L03-R09-KV"]?.placement.insertionAfter !== "85:08") throw new Error("L3 companion must follow the uninterrupted slides 57–63 run");
if (registry.byAnchor["L03-KV-CACHE"]?.runId !== "L03-R09" || registry.byAnchor["L03-GQA"]?.stateId !== "gqa-knob") throw new Error("L3 KV deep links must converge on the single finished companion");
for (const slot of ["L01-R09-REPLAY", "L02-R05-LEDGER", "L03-R08-CACHE", "L03-R09-EVIDENCE"]) if (!registry.supersededSlots.includes(slot)) throw new Error(`${slot} must be superseded to avoid duplicate teaching UI`);

const expectedRunCounts = new Map([[1, 10], [2, 8], [3, 10]]);
const allowedSlots = new Set(model.slotTypes);
for (const lecture of model.lectures) {
  if (lecture.runs.length !== expectedRunCounts.get(lecture.lecture)) throw new Error(`Lecture ${lecture.lecture} has the wrong run count`);
  let previousEnd = 0;
  for (const run of lecture.runs) {
    for (const field of ["id", "title", "professorIntent", "originalSummary"]) if (!run[field]) throw new Error(`${run.id || "run"} is missing ${field}`);
    if (run.startSeconds < previousEnd || run.endSeconds <= run.startSeconds || run.endSeconds > lecture.videoDurationSeconds) throw new Error(`${run.id} has an invalid or unordered time range`);
    previousEnd = run.endSeconds;
    if (!run.sourceRefs?.length) throw new Error(`${run.id} needs original source references`);
    for (const slot of run.augmentationSlots) {
      if (!allowedSlots.has(slot.type) || !slot.pedagogicalReason) throw new Error(`${slot.id} has an invalid type or empty reason`);
      if (slot.atSeconds < run.startSeconds || slot.atSeconds > run.endSeconds) throw new Error(`${slot.id} is outside ${run.id}`);
    }
  }
}

const html = fs.readFileSync("index.html", "utf8");
if (!html.includes("A source-aligned companion to Stanford CS336") || html.includes("Keep the lecture. Add the missing view.")) throw new Error("The course header must describe the companion rather than use the retired slogan");
for (const ref of ["styles.css", "config.js", "data/lectures.js", "data/lecture_runs.js", "data/augmentation_registry_v2.js", "app.js"]) if (!html.includes(ref)) throw new Error(`index.html does not reference ${ref}`);
for (const id of ["run-nav", "lecture-runs", "context-rail", "rail-content", "discussion-status"]) if (!html.includes(`id="${id}"`)) throw new Error(`index.html is missing ${id}`);
if (!html.includes("lecture-media-note")) throw new Error("The lecture intro must explain transcript-timed official excerpts");
if (/<video\b/i.test(html)) throw new Error("Legacy local MP4s must not appear in the primary HTML path");

const app = fs.readFileSync("app.js", "utf8");
for (const behavior of ["CS336_LECTURE_RUNS", "CS336_AUGMENTATION_REGISTRY_V2", "model.anchorToRun", "data-mobile-rail", "data-active-rail", "originalMaterials", "Transcript-indexed excerpt", "Stop at", "mountGiscus", "loadArtifact", "loadStyle", "loadScript", "syncRunFromViewport", "renderLecture(number, initialRunId", "loading = \"lazy\""]) if (!app.includes(behavior)) throw new Error(`app.js is missing lecture-first behavior: ${behavior}`);
if (/youtube(?:-nocookie)?\.com\/embed/.test(app)) throw new Error("Stanford Online disables third-party playback; excerpt cards must link to the official timestamp instead of presenting a broken embed");
for (const removed of ["Learning goal", "Storyboard", "One clip,", "Segment-specific notes", "data-lazy-video", "augmentationManifestUrls"]) if (app.includes(removed)) throw new Error(`Old repeated segment UI remains in app.js: ${removed}`);
if (app.includes("relocateRail")) throw new Error("The desktop sticky rail must not be reparented into a short mobile run container");
if (/media\/lecture_|RooflineModel\.mp4|L01-BPE-TRAIN-VS-USE\.mp4/i.test(app + fs.readFileSync("data/augmentation_registry_v2.js", "utf8"))) throw new Error("The V2 loader must not restore retired or rebuild MP4s");
if (/\.autoplay\s*=|setAttribute\(\s*["']autoplay/i.test(app)) throw new Error("Augmentation videos must not autoplay");
if (!/video\.preload\s*=\s*["']metadata["']/.test(app)) throw new Error("Optional video must load only after learner request");

const styles = fs.readFileSync("styles.css", "utf8");
if (!styles.includes("grid-template-columns: 176px minmax(0, 1fr) minmax(380px, 0.54fr)")) throw new Error("Desktop lecture/source/supplement grid must leave the finished artifact enough reading width");
if (!styles.includes("align-items: stretch") || !styles.includes(".run-outline, .context-rail { align-self: stretch")) throw new Error("Sticky rail parents must span the full lecture-run flow");
if (!styles.includes("@media (max-width: 920px)")) throw new Error("Mobile source-to-augmentation flow is missing");
if (!styles.includes(".mobile-context")) throw new Error("Mobile augmentation needs an independent inline container");
if (!app.includes("pinnedRunId") || !app.includes('window.addEventListener("wheel", releasePinnedRun') || !app.includes('window.addEventListener("touchstart", releasePinnedRun')) throw new Error("Deep-linked runs must remain pinned through programmatic layout scroll and release on learner navigation");

const workflow = fs.readFileSync(".github/workflows/pages.yml", "utf8");
if (!/^\s+path:\s+\.\s*$/m.test(workflow)) throw new Error("Pages workflow must upload repository root");

console.log(`Validated ${model.lectures.length} lecture spines, ${model.lectures.reduce((sum, lecture) => sum + lecture.runs.length, 0)} continuous runs, ${legacyIds.size} stable anchors, and ${registry.artifacts.length} on-demand V2 artifacts. Retired/rebuild MP4s remain outside the primary path.`);
