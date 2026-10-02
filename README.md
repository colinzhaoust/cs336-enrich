# Atlas · CS336

Live: https://colinzhaoust.github.io/cs336-enrich/

Stanford CS336 (all 17 lectures), five papers (Chinchilla, FlashAttention, RoPE, DeepSeekMath/GRPO, FeynRL/P3O) and three repositories (FeynRL, edtrace, TheoremExplainAgent), cut into 418 knowledge points. Each knowledge point carries:

- **provenance:** anchors to lecture code lines, slide pages, paper sections and lecture-video timestamps, including what the professor said but did not write;
- **a prediction you commit first;**
- **a mechanism to watch or drive:** 25 rendered Manim animations and 111 widgets whose numbers are tested against the questions;
- **a closed-book changed case;**
- **spaced-repetition cards in two layers** (core, then detail).

- `index.html`: overview and rendered animations
- `site/`: the app (course map, sessions, knowledge points, per-thread widget galleries at `#/t/<thread>/_gallery`)
- `lectures/`, `papers/`, `repos/`: knowledge-point JSON per thread
- `build/`: rendered animations (video, step frames, manifest)
- `PHILOSOPHY.md`: the design rules and the evidence behind them

Lecture figures load from [stanford-cs336/lectures](https://github.com/stanford-cs336/lectures); caption transcripts and paper texts are not published here. This is a generated build; the previous version of this site is tagged `v1-codex`.
