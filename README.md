# CS336 Enriched

An independent visual companion to Stanford CS336: Language Modeling from Scratch.

The site pairs original course material with source-faithful visual explanations, interactive diagrams, and short Manim sequences. It currently covers Lectures 1–3 with 12 rendered clips. The repository is a self-contained static site: it works at the repository root locally and under the GitHub Pages project path `/cs336-enrich/`.

## Preview locally

From this directory:

```bash
python3 -m http.server 8000
```

Then open `http://localhost:8000`.

Opening `index.html` directly is useful for a quick layout check, but the augmentation manifests are fetched at runtime, so use a local HTTP server to load videos and slide supplements reliably.

## Repository layout

```text
data/lectures.js                 source map and 88 stable segment IDs
augmentations/lecture_01..03/    manifests and segment-specific slide notes
media/lecture_01..03/            12 published MP4s and 12 posters
scenes/cs336/lecture_01..03/     executable Manim and reproduction source
.github/workflows/pages.yml      repository-root GitHub Pages deployment
```

Each rendered augmentation includes a **Manim source** link. Manifest paths are repository-root-relative, so those links continue to work at `https://colinzhaoust.github.io/cs336-enrich/`.

## Render the Manim source locally

The checked-in preview renders were made with Manim Community 0.19.0 at 854×480 and 15 fps. Install [Manim's system dependencies](https://docs.manim.community/en/stable/installation.html), including ffmpeg, then create a local environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install "manim==0.19.0"
```

From the repository root, render each lecture to ignored local build output:

```bash
python -m manim -ql --media_dir build/manim/lecture_01 \
  scenes/cs336/lecture_01/lecture_01_tokenization.py \
  CS336UTF8Bytes CS336BPEPairCountAndMerge \
  CS336BPETrainVsUse CS336TokenizerTradeoffToy

python -m manim -ql --media_dir build/manim/lecture_02 \
  scenes/cs336/lecture_02/lecture_02_augmented.py \
  FloatingPointRange EinopsRearrangeHeads RooflineModel
python -m manim -ql --media_dir build/manim/lecture_02_checkpointing \
  scenes/cs336/lecture_02/lecture_02_checkpointing.py \
  ActivationCheckpointingTradeoff

python -m manim -ql --media_dir build/manim/lecture_03 \
  scenes/cs336/lecture_03/lecture_03_scenes.py \
  L03PrePostNorm L03KVSharing L03SlidingWindow
python -m manim -ql --media_dir build/manim/lecture_03_runtime_norm \
  scenes/cs336/lecture_03/lecture_03_runtime_norm.py \
  L03RuntimeNorm
```

Replace `-ql` with `-qm` for a medium-quality render. The scene sources use Helvetica Neue and Menlo for deterministic macOS previews; on another platform, install compatible fonts or change the `FONT`/`SANS` and `MONO` constants. `scenes/cs336/lecture_01/reproduce_tokenizer_tradeoff.py` independently reproduces the Lecture 1 tokenizer toy counts.

Before publishing changed media, review representative frames, generate a poster, preserve MP4 fast-start metadata, and update the matching manifest. The media allowlist in `.gitignore` deliberately excludes Manim caches, partial renders, QA contact sheets, and any unreviewed media filename.

## Content model

Lecture and segment content lives in `data/lectures.js`. Every segment has:

- a stable ID such as `L02-BF16-RANGE`;
- an original source locator;
- a source summary and learning goal;
- planned output formats;
- an implementation priority;
- three storyboard beats;
- a visual family used by the current HTML preview.

The stable ID is also the GitHub Discussion term. Do not rename an ID after feedback exists unless the related Discussion is migrated deliberately.

## Enable GitHub Discussions and Giscus

1. Create a public GitHub repository named `cs336-enrich`.
2. Open the repository's **Settings** page and enable **Discussions** under Features.
3. Create a category named `Course feedback`. The Announcements category type works well because maintainers and Giscus control topic creation while visitors can comment.
4. Install the [Giscus GitHub App](https://github.com/apps/giscus) for the repository.
5. Open [giscus.app](https://giscus.app/) and enter `colinzhaoust/cs336-enrich`.
6. Select **Discussion title contains a specific term**, enable strict matching, and choose `Course feedback`.
7. Copy the generated repository ID and category ID into `config.js`.
8. Set `enabled: true` in `config.js`.

Each segment button then mounts one Giscus thread using the segment ID as its specific term. Before Giscus is configured, the same button copies a structured feedback template and explains the pending setup instead of opening a dead link.

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
