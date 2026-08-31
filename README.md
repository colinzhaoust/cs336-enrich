# CS336 Enriched

An independent visual companion to Stanford CS336: Language Modeling from Scratch.

The site keeps the original lecture sequence as its spine. Each of its 28 continuous teaching runs combines a transcript-timed excerpt from the original Stanford Online embed, the exact official slide or executable-source location, and only then our formulas, comparisons, interactives, or optional slow demonstrations. Video is never downloaded, cut, or rehosted. Three reviewed V2 artifacts are integrated on demand; the former 12 short renders remain out of the primary learning path. The repository is a self-contained static site that works at the repository root locally and under the GitHub Pages project path `/cs336-enrich/`.

## Preview locally

From this directory:

```bash
python3 -m http.server 8000
```

Then open `http://localhost:8000`.

Opening `index.html` directly is useful for a quick layout check, but use a local HTTP server to exercise interactive sidecars and the optional video reliably.

## Repository layout

```text
data/lectures.js                 source map and 88 stable discussion IDs
data/lecture_runs.js             28 continuous lecture-first teaching runs
data/augmentation_registry_v2.js reviewed artifact placement and lazy-load contract
augmentations_v2/lecture_01..03/ responsive BPE, roofline, and KV-decode companions
media_v2/lecture_01,lecture_03/  optional reviewed slow replays and sidecars
augmentations/ and media/        legacy material retained outside the primary path
.github/workflows/pages.yml      repository-root GitHub Pages deployment
```

Registry paths are repository-root-relative and resolved from the current document, so the same loader works at localhost and `https://colinzhaoust.github.io/cs336-enrich/`.

## Render the Manim source locally

The primary Manim path is the reviewed V2 source in `scenes/cs336_v2/`. Its production target is 1920×1080 at 30 fps. Install [Manim's system dependencies](https://docs.manim.community/en/stable/installation.html), including ffmpeg, then create a local environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install "manim==0.19.0"
```

From the repository root, verify the deterministic fixtures first:

```bash
python scenes/cs336_v2/lecture_01/bpe_fixture.py
python scenes/cs336_v2/lecture_03/kv_decode_fixture.py
```

Then render the reviewed V2 scenes to ignored local build output:

```bash
python -m manim -qh --fps 30 --format mp4 \
  --media_dir build/manim/v2/lecture_01 \
  scenes/cs336_v2/lecture_01/bpe_after_trace.py \
  BPEAfterTraceV2

python -m manim -qh --fps 30 --format mp4 \
  --media_dir build/manim/v2/lecture_03 \
  scenes/cs336_v2/lecture_03/kv_decode_evidence.py \
  KVDecodeEvidenceV2
```

`-qh --fps 30` is the production setting: 1920×1080, 30 fps. The older 854×480/15 fps clips under `scenes/cs336/` and `media/` are legacy references only and are not mounted in the primary learning path. The V2 scene sources use Helvetica Neue and Menlo for deterministic macOS renders; on another platform, install compatible fonts or change the `SANS` and `MONO` constants.

Before publishing changed media, review representative frames, generate a poster, preserve MP4 fast-start metadata, and update the matching manifest. The media allowlist in `.gitignore` deliberately excludes Manim caches, partial renders, QA contact sheets, and any unreviewed media filename.

## Content model

`data/lecture_runs.js` groups the 88 legacy IDs into the professor's continuous teaching runs: 10 for Lecture 1, 8 for Lecture 2, and 10 for Lecture 3. Each run preserves the time range, professor intent, official source location, source-faithful paraphrase, and transition. Augmentations live only in declared slots inside those runs.

The run start/end times were reconstructed from public caption timelines and checked against the official recording and source. Stanford Online disables playback on third-party websites, so each excerpt card opens the official YouTube recording at `start` and keeps `end` visible as the stop/return cue; it does not create or publish a derivative video file. Lecture 3 links the corresponding official slide ranges, while Lectures 1–2 link the executable lecture source that generated the on-screen material.

`data/augmentation_registry_v2.js` maps a finished artifact to one slot, its internal deep-link state, source interval, caption, visual description, provenance, caveat, and lazy assets. Planned slots remain labels rather than simulated content. The 88 stable IDs are also GitHub Discussion terms; do not rename one after feedback exists unless the related Discussion is migrated deliberately.

## Enable GitHub Discussions and Giscus

The public repository, Discussions, and the Giscus GitHub App are enabled.
`config.js` contains the verified repository ID and the ID of the default
`General` category.

The embedded comment setup uses:

- **Discussion title contains a specific term**
- strict matching
- the `General` category
- each stable segment ID as its term

Each run has one lightweight feedback control. The learner chooses the closest stable anchor, and the site mounts one Giscus thread using that ID as its specific term. Artifact anchors stay available even when a finished replay is mounted after the original run that introduced the mechanism.

## Feedback workflow

```text
learner comment
→ GitHub Discussion keyed by segment ID
→ maintainer triage
→ storyboard / Manim / HTML revision
→ browser and frame inspection
→ commit and Pages deployment
→ reply with the new URL and commit
```

Recommended comment structure:

```text
Segment: L02-FLOATING-POINT
Type: confusion | correction | visual idea
I expected:
I got lost when:
A clearer visual might:
```

## Publish with GitHub Pages

The included workflow publishes the repository root whenever `main` is updated. In GitHub:

1. Go to **Settings → Pages**.
2. Under **Build and deployment**, choose **GitHub Actions**.
3. Push to `main` or manually run the `Deploy CS336 Enriched` workflow.

The resulting URL is expected to be:

```text
https://colinzhaoust.github.io/cs336-enrich/
```

## Upstream course material

Original CS336 materials are linked, not mirrored. Source links in the site point to the official [`stanford-cs336/lectures`](https://github.com/stanford-cs336/lectures) repository, and video links point to the original course recordings. This repository contains only the independent segment map, explanatory copy, Manim source, and derived visual supplements.

## Disclaimer

This is an independent educational companion. It is not affiliated with or endorsed by Stanford University. Original course material remains linked to the official CS336 website and the `stanford-cs336/lectures` repository.

No license has been selected for this repository yet.
