# CS336 Lecture 3 augmented scenes

These scenes use a light academic visual system derived from the CS336 course
site. Each class is a short, silent insert designed to sit beside the original
lecture material.

## Render

From the repository root:

```bash
python -m manim -ql \
  --media_dir build/manim/lecture_03 \
  scenes/cs336/lecture_03/lecture_03_scenes.py \
  L03PrePostNorm L03KVSharing L03SlidingWindow

python -m manim -ql \
  --media_dir build/manim/lecture_03_runtime_norm \
  scenes/cs336/lecture_03/lecture_03_runtime_norm.py \
  L03RuntimeNorm
```

The source avoids LaTeX so the preview build needs only Manim and ffmpeg.

## Scene mapping

| Scene | Segment IDs | Main visual claim |
| --- | --- | --- |
| `L03PrePostNorm` | `L03-PRE-POST-NORM` | Pre-norm leaves the identity residual path unnormalized. |
| `L03RuntimeNorm` | `L03-RMSNORM`, `L03-NORM-RUNTIME`, `L03-NO-BIAS` | Lower normalization and bias traffic can matter even when these operations are a small share of FLOPs. |
| `L03KVSharing` | `L03-KV-CACHE`, `L03-MQA`, `L03-GQA` | Number of KV heads is a direct cache-size knob. |
| `L03SlidingWindow` | `L03-SLIDING-WINDOW` | Local attention keeps a causal band rather than the full triangle. |
