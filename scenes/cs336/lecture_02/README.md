# CS336 Lecture 2 augmented scenes

The scenes in `lecture_02_augmented.py` are short, source-aligned visual
supplements for stable IDs already used by the site.

```bash
python -m manim -ql --media_dir build/manim/lecture_02 \
  scenes/cs336/lecture_02/lecture_02_augmented.py FloatingPointRange

python -m manim -ql --media_dir build/manim/lecture_02 \
  scenes/cs336/lecture_02/lecture_02_augmented.py EinopsRearrangeHeads

python -m manim -ql --media_dir build/manim/lecture_02 \
  scenes/cs336/lecture_02/lecture_02_augmented.py RooflineModel

python -m manim -ql --media_dir build/manim/lecture_02_checkpointing \
  scenes/cs336/lecture_02/lecture_02_checkpointing.py \
  ActivationCheckpointingTradeoff
```

Preview renders use Manim's low-quality preset (854×480, 15 fps). Production
renders can replace `-ql` with `-qm` without changing the source.

The checkpointing scene renders to ignored local build output first so the
final MP4 can be fast-start remuxed and copied into `media/lecture_02/`
without leaving Manim cache directories in the published site tree.

Its representative poster is `ActivationCheckpointingTradeoff.png`; the
four-beat QA contact sheet is stored under `media/lecture_02/contact-sheets/`.
