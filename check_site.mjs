import fs from "node:fs";
import path from "node:path";
import { spawnSync } from "node:child_process";
import vm from "node:vm";

const requiredFiles = [
  "index.html",
  "styles.css",
  "app.js",
  "slide_parser.js",
  "config.js",
  "data/lectures.js",
  "README.md",
  ".gitignore",
  ".github/workflows/pages.yml"
];
for (const file of requiredFiles) {
  if (!fs.existsSync(file)) throw new Error(`Missing required file: ${file}`);
}

const context = { window: {} };
vm.createContext(context);
vm.runInContext(fs.readFileSync("data/lectures.js", "utf8"), context);
vm.runInContext(fs.readFileSync("slide_parser.js", "utf8"), context);

const lectures = context.window.CS336_LECTURES;
if (!Array.isArray(lectures) || lectures.length !== 3) {
  throw new Error("Expected exactly three lectures");
}

const ids = new Set();
let total = 0;
for (const lecture of lectures) {
  if (!lecture.sourceUrl || !lecture.videoUrl || !lecture.summary) {
    throw new Error(`Lecture ${lecture.number} is missing required metadata`);
  }
  for (const segment of lecture.segments) {
    total += 1;
    if (ids.has(segment.id)) throw new Error(`Duplicate segment ID: ${segment.id}`);
    ids.add(segment.id);
    for (const field of ["chapter", "title", "source", "summary", "goal", "priority", "visual"]) {
      if (!segment[field]) throw new Error(`${segment.id} is missing ${field}`);
    }
    if (!Array.isArray(segment.beats) || segment.beats.length !== 3) {
      throw new Error(`${segment.id} must have exactly three storyboard beats`);
    }
    if (!lecture.chapters.includes(segment.chapter)) {
      throw new Error(`${segment.id} uses unknown chapter: ${segment.chapter}`);
    }
  }
}

function lectureByNumber(number) {
  const lecture = lectures.find((candidate) => candidate.number === number);
  if (!lecture) throw new Error(`Missing lecture ${number}`);
  return lecture;
}

function assertContiguousSegmentOrder(lectureNumber, expectedIds) {
  const actualIds = lectureByNumber(lectureNumber).segments.map((segment) => segment.id);
  const start = actualIds.indexOf(expectedIds[0]);
  if (start === -1 || expectedIds.some((id, offset) => actualIds[start + offset] !== id)) {
    throw new Error(`Lecture ${lectureNumber} must preserve segment order: ${expectedIds.join(" -> ")}`);
  }
}

assertContiguousSegmentOrder(2, [
  "L02-TENSOR-RANK",
  "L02-BSHD",
  "L02-FP32",
  "L02-TENSOR-MEMORY",
  "L02-FP16-UNDERFLOW",
  "L02-BF16-RANGE",
  "L02-MIXED-PRECISION"
]);
assertContiguousSegmentOrder(3, ["L03-FF-RATIO", "L03-GLU-DIMENSION", "L03-FF-BASIN"]);
const gluDimension = lectureByNumber(3).segments.find((segment) => segment.id === "L03-GLU-DIMENSION");
if (gluDimension?.chapter !== "Hyperparameters") {
  throw new Error("L03-GLU-DIMENSION must belong to the Hyperparameters chapter");
}

const tokenizerToy = lectureByNumber(1).segments.find((segment) => segment.id === "L01-TOKENIZER-PARETO");
const tokenizerToyCopy = [
  tokenizerToy?.title,
  tokenizerToy?.source,
  tokenizerToy?.summary,
  tokenizerToy?.goal,
  ...(tokenizerToy?.beats || [])
].join(" ");
if (/\bpareto\b|\bfrontier\b/i.test(tokenizerToyCopy)) {
  throw new Error("L01-TOKENIZER-PARETO copy must not claim a Pareto frontier");
}
for (const provenancePhrase of ["toy", "the cat in the hat", "the quick brown fox", "three BPE merges", "OOV"]) {
  if (!tokenizerToyCopy.toLowerCase().includes(provenancePhrase.toLowerCase())) {
    throw new Error(`L01-TOKENIZER-PARETO must retain toy provenance: ${provenancePhrase}`);
  }
}

const html = fs.readFileSync("index.html", "utf8");
for (const ref of ["styles.css", "config.js", "data/lectures.js", "slide_parser.js", "app.js"]) {
  if (!html.includes(ref)) throw new Error(`index.html does not reference ${ref}`);
}
if (html.indexOf("slide_parser.js") > html.indexOf("app.js")) {
  throw new Error("index.html must load slide_parser.js before app.js");
}
for (const id of ["reading-position", "reading-chapter", "reading-segment", "reading-count", "reading-progress"]) {
  if (!html.includes(`id="${id}"`)) throw new Error(`index.html is missing reading-position control: ${id}`);
}

const app = fs.readFileSync("app.js", "utf8");
for (const behavior of ["sourceUrlForSegment", "data-copy-link", "data-lazy-video", "shared-sequence", "Manim source"]) {
  if (!app.includes(behavior)) throw new Error(`app.js is missing source/augmentation behavior: ${behavior}`);
}

const pagesWorkflow = fs.readFileSync(".github/workflows/pages.yml", "utf8");
if (!/^\s+path:\s+\.\s*$/m.test(pagesWorkflow)) {
  throw new Error("GitHub Pages workflow must upload the repository root with path: .");
}

const markdownSlides = context.window.CS336_PARSE_MARKDOWN_SLIDES(fs.readFileSync("augmentations/lecture_01/slides.md", "utf8"));
const utf8Slide = markdownSlides.get("L01-UTF8-BYTES");
if (!utf8Slide || utf8Slide.title !== "One character is not necessarily one byte") {
  throw new Error("Production Markdown parser did not preserve the explicit UTF-8 slide title");
}
for (const [segmentId, slide] of markdownSlides) {
  if (/Slide title:/i.test(slide.html)) throw new Error(`${segmentId} repeats Slide title metadata in rendered notes`);
}
if (!/Learning check:/i.test(utf8Slide.html)) {
  throw new Error("Production Markdown parser removed real UTF-8 slide body content");
}

const manifestPaths = [1, 2, 3].map((lecture) => `augmentations/lecture_${String(lecture).padStart(2, "0")}/manifest.json`);
const renderedIds = new Set();
const renderedVideos = new Set();
const renderedPosters = new Set();
const repositoryRoot = path.resolve(".");

function resolveManifestFile(rawPath, manifestPath) {
  const cleanPath = rawPath.split("#")[0];
  if (cleanPath.startsWith("media/") || cleanPath.startsWith("augmentations/") || cleanPath.startsWith("scenes/")) {
    return path.resolve(cleanPath);
  }
  return path.resolve(path.dirname(manifestPath), cleanPath);
}

function isGitIgnored(artifactPath) {
  const repositoryRelativePath = path.relative(repositoryRoot, artifactPath);
  const result = spawnSync("git", ["check-ignore", "--no-index", "-q", "--", repositoryRelativePath], {
    cwd: repositoryRoot,
    stdio: "ignore"
  });
  if (result.status === 0) return true;
  if (result.status === 1) return false;
  throw new Error(`Could not verify git-ignore status for ${repositoryRelativePath}`);
}

function requireArtifact(rawPath, manifestPath, label) {
  if (!rawPath) throw new Error(`${manifestPath} is missing ${label}`);
  const artifactPath = resolveManifestFile(rawPath, manifestPath);
  if (!fs.existsSync(artifactPath)) throw new Error(`${manifestPath} references missing ${label}: ${rawPath}`);
  if (fs.statSync(artifactPath).size < 1024) throw new Error(`${label} is unexpectedly small: ${rawPath}`);
  const repositoryRelativePath = path.relative(repositoryRoot, artifactPath);
  if (isGitIgnored(artifactPath)) throw new Error(`${manifestPath} references git-ignored ${label}: ${repositoryRelativePath}`);
  return artifactPath;
}

for (const manifestPath of manifestPaths) {
  const manifest = JSON.parse(fs.readFileSync(manifestPath, "utf8"));
  const assets = manifest.segments
    ? Object.entries(manifest.segments).map(([segmentId, asset]) => ({ ...asset, segmentIds: [segmentId] }))
    : (manifest.assets || []).map((asset) => ({ ...asset, segmentIds: asset.segmentIds || asset.segments || [] }));
  const qaContactSheets = [manifest.render?.contactSheet, ...(manifest.assets || []).map((asset) => asset.contactSheet)].filter(Boolean);

  for (const contactSheet of qaContactSheets) {
    const contactSheetPath = resolveManifestFile(contactSheet, manifestPath);
    if (fs.existsSync(contactSheetPath) && !isGitIgnored(contactSheetPath)) {
      throw new Error(`${manifestPath} QA contact sheet must remain git-ignored: ${path.relative(repositoryRoot, contactSheetPath)}`);
    }
  }

  if (!assets.length) throw new Error(`${manifestPath} has no rendered assets`);
  for (const asset of assets) {
    renderedVideos.add(requireArtifact(asset.video, manifestPath, "video"));
    renderedPosters.add(requireArtifact(asset.poster, manifestPath, "poster"));
    const manimSource = asset.source || manifest.render?.source;
    if (!manimSource?.startsWith("scenes/") || manimSource.includes("..")) {
      throw new Error(`${manifestPath} Manim source must be repository-root-relative: ${manimSource || "missing"}`);
    }
    requireArtifact(manimSource, manifestPath, "Manim source");
    asset.segmentIds.forEach((segmentId) => {
      if (!ids.has(segmentId)) throw new Error(`${manifestPath} references unknown segment: ${segmentId}`);
      renderedIds.add(segmentId);
    });
    if (asset.slides) requireArtifact(asset.slides, manifestPath, "slide supplement");
  }

  if (manifest.supplement) requireArtifact(manifest.supplement, manifestPath, "slide supplement");

  if (manifest.lecture === 1) {
    const tokenizerToyAsset = manifest.segments?.["L01-TOKENIZER-PARETO"];
    const expectedProvenance = {
      trainingText: "the cat in the hat",
      heldOutText: "the quick brown fox",
      bpeMerges: 3,
      wordVocabulary: "five distinct training chunks plus UNK"
    };
    if (!tokenizerToyAsset || JSON.stringify(tokenizerToyAsset.provenance) !== JSON.stringify(expectedProvenance)) {
      throw new Error(`${manifestPath} must retain the declared tokenizer toy provenance`);
    }
    if (tokenizerToyAsset.data !== "scenes/cs336/lecture_01/reproduce_tokenizer_tradeoff.py") {
      throw new Error(`${manifestPath} must link the tokenizer toy to its reproduction script`);
    }
  }

  if (manifest.lecture === 3) {
    const kvAsset = (manifest.assets || []).find((asset) => asset.id === "l03-kv-sharing");
    const expectedKvSegments = ["L03-KV-CACHE", "L03-MQA", "L03-GQA"];
    if (!kvAsset || JSON.stringify(kvAsset.segments) !== JSON.stringify(expectedKvSegments)) {
      throw new Error(`${manifestPath} must map l03-kv-sharing to ${expectedKvSegments.join(" -> ")}`);
    }
    const expectedSourcePages = [58, 59, 60, 61, 62, 63];
    if (JSON.stringify(kvAsset.sourcePages) !== JSON.stringify(expectedSourcePages)) {
      throw new Error(`${manifestPath} must cite official Lecture 3 slides 58–63 for l03-kv-sharing`);
    }
    if (
      JSON.stringify(kvAsset.arithmeticIntensity?.incrementalMHA?.sourcePages) !== JSON.stringify([59, 60]) ||
      kvAsset.qualityEvidence?.sourcePage !== 63
    ) {
      throw new Error(`${manifestPath} must retain incremental-decoding motivation and page-63 quality evidence`);
    }
  }
}

if (renderedVideos.size !== 12 || renderedPosters.size !== 12) {
  throw new Error(`Expected exactly 12 published clips and 12 posters; found ${renderedVideos.size} clips and ${renderedPosters.size} posters`);
}

function filesUnder(directory) {
  return fs.readdirSync(directory, { withFileTypes: true }).flatMap((entry) => {
    const entryPath = path.resolve(directory, entry.name);
    return entry.isDirectory() ? filesUnder(entryPath) : [entryPath];
  });
}

for (const mediaPath of filesUnder("media")) {
  if (!renderedVideos.has(mediaPath) && !renderedPosters.has(mediaPath) && !isGitIgnored(mediaPath)) {
    throw new Error(`Unreviewed media must remain git-ignored: ${path.relative(repositoryRoot, mediaPath)}`);
  }
}

console.log(`Validated ${lectures.length} lectures, ${total} unique segments, and ${renderedIds.size} rendered segment augmentations across ${renderedVideos.size} unique clips with ${renderedPosters.size} posters. The Pages workflow deploys the repository root; all published manifest assets are present and trackable; extra media and QA sheets are ignored; Markdown slide-title metadata is not repeated in rendered notes.`);
