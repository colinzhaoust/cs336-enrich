import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { createRequire } from "node:module";

const here = path.dirname(fileURLToPath(import.meta.url));
const siteRoot = path.resolve(here, "..");
const require = createRequire(import.meta.url);
const model = require(path.join(siteRoot, "data/lecture_runs.js"));
const legacySource = fs.readFileSync(path.join(siteRoot, "data/lectures.js"), "utf8");
const legacyIds = [...legacySource.matchAll(/segment\("(L\d{2}-[A-Z0-9-]+)"/g)].map((match) => match[1]);
const mappedIds = Object.keys(model.anchorToRun);
const expectedVideos = new Map([
  [1, "https://www.youtube.com/watch?v=JuoVZkPBiKk"],
  [2, "https://www.youtube.com/watch?v=kuYAsz7zspQ"],
  [3, "https://www.youtube.com/watch?v=lVynu4bo1rY"]
]);

const failures = [];
const assert = (condition, message) => {
  if (!condition) failures.push(message);
};

assert(legacyIds.length === 88, `Expected 88 legacy IDs; found ${legacyIds.length}.`);
assert(new Set(legacyIds).size === legacyIds.length, "Legacy segment IDs are not unique.");
assert(mappedIds.length === 88, `Expected 88 mapped IDs; found ${mappedIds.length}.`);

for (const id of legacyIds) assert(model.anchorToRun[id], `Missing ID→run mapping for ${id}.`);
for (const id of mappedIds) assert(legacyIds.includes(id), `Mapping contains unknown legacy ID ${id}.`);

const artifactPaths = new Set();
for (const artifact of model.artifactInventory) {
  assert(Boolean(artifact.path?.trim()), "Artifact inventory contains a blank path.");
  assert(!artifactPaths.has(artifact.path), `Artifact inventory repeats ${artifact.path}.`);
  assert(model.artifactStatuses.includes(artifact.status), `${artifact.path} has invalid status ${artifact.status}.`);
  assert(Boolean(artifact.role?.trim()), `${artifact.path} has no intended role.`);
  artifactPaths.add(artifact.path);
}
for (const status of model.artifactStatuses) {
  assert(model.artifactInventory.some((artifact) => artifact.status === status), `Artifact inventory does not exercise status ${status}.`);
}

let slotCount = 0;
for (const lecture of model.lectures) {
  const prefix = `L${String(lecture.lecture).padStart(2, "0")}-`;
  assert(lecture.videoUrl === expectedVideos.get(lecture.lecture), `Lecture ${lecture.lecture} video URL does not match the verified Stanford recording.`);
  assert(lecture.sourceUrl.startsWith("https://github.com/stanford-cs336/lectures/"), `Lecture ${lecture.lecture} source is not in the official lectures repository.`);
  assert(lecture.restoredSourceBeats?.length > 0, `Lecture ${lecture.lecture} does not declare restored evidence/recap beats.`);

  let previousEnd = -1;
  const seenRunIds = new Set();
  for (const lectureRun of lecture.runs) {
    assert(!seenRunIds.has(lectureRun.id), `Duplicate run ID ${lectureRun.id}.`);
    seenRunIds.add(lectureRun.id);
    assert(previousEnd === -1 || lectureRun.startSeconds === previousEnd,
      `${lectureRun.id} does not continue exactly from the previous run boundary.`);
    assert(lectureRun.startSeconds < lectureRun.endSeconds, `${lectureRun.id} has an invalid time range.`);
    assert(lectureRun.endSeconds <= lecture.videoDurationSeconds, `${lectureRun.id} extends beyond the verified recording duration.`);
    assert(Boolean(lectureRun.professorIntent?.trim()), `${lectureRun.id} has no professor intent.`);
    assert(Boolean(lectureRun.originalSummary?.trim()), `${lectureRun.id} has no original-content summary.`);
    assert(lectureRun.sourceRefs?.length > 0, `${lectureRun.id} has no source references.`);
    previousEnd = lectureRun.endSeconds;

    for (const anchor of lectureRun.anchors) {
      assert(anchor.startsWith(prefix), `${anchor} is attached to the wrong lecture run ${lectureRun.id}.`);
      assert(model.anchorToRun[anchor] === lectureRun.id, `${anchor} does not round-trip through anchorToRun.`);
    }

    let previousBoundaryEnd = lectureRun.startSeconds;
    for (const boundary of lectureRun.uninterruptedRanges) {
      assert(boundary.startSeconds >= lectureRun.startSeconds && boundary.endSeconds <= lectureRun.endSeconds,
        `${lectureRun.id} has an uninterrupted range outside its run.`);
      assert(boundary.startSeconds < boundary.endSeconds, `${lectureRun.id} has an invalid uninterrupted range.`);
      assert(boundary.startSeconds >= previousBoundaryEnd, `${lectureRun.id} has unordered/overlapping uninterrupted ranges.`);
      assert(Boolean(boundary.reason?.trim()), `${lectureRun.id} has an uninterrupted range without a reason.`);
      previousBoundaryEnd = boundary.endSeconds;
    }

    for (const augmentation of lectureRun.augmentationSlots) {
      slotCount += 1;
      assert(model.slotTypes.includes(augmentation.type), `${augmentation.id} uses unsupported type ${augmentation.type}.`);
      assert(Boolean(augmentation.pedagogicalReason?.trim()), `${augmentation.id} has no pedagogical reason.`);
      assert(augmentation.optional === true, `${augmentation.id} is not marked optional.`);
      assert(augmentation.atSeconds >= lectureRun.startSeconds && augmentation.atSeconds <= lectureRun.endSeconds,
        `${augmentation.id} lies outside ${lectureRun.id}.`);

      for (const boundary of lectureRun.uninterruptedRanges) {
        const interrupts = augmentation.atSeconds > boundary.startSeconds && augmentation.atSeconds < boundary.endSeconds;
        assert(!interrupts, `${augmentation.id} interrupts ${lectureRun.id} at ${augmentation.atSeconds}s.`);
      }

      const status = augmentation.content?.artifactStatus;
      if (status) assert(model.artifactStatuses.includes(status), `${augmentation.id} has invalid artifact status ${status}.`);
    }
  }
}

if (failures.length) {
  console.error(`Lecture-run validation failed (${failures.length}):`);
  for (const failure of failures) console.error(`- ${failure}`);
  process.exit(1);
}

const runCounts = Object.fromEntries(model.lectures.map((lecture) => [
  `L${lecture.lecture}`,
  lecture.runs.length
]));

console.log(JSON.stringify({
  ok: true,
  legacyIds: legacyIds.length,
  mappedIds: mappedIds.length,
  runCounts,
  augmentationSlots: slotCount,
  artifactInventory: model.artifactInventory.length,
  verifiedVideoUrls: Object.fromEntries(expectedVideos)
}, null, 2));
