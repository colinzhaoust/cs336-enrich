"""CS336 Lecture 2 augmented diagrams.

These scenes intentionally follow the course's restrained visual language:
light paper, Stanford cardinal, compact engineering labels, and inspectable
quantities.  They are short visual supplements for the matching stable segment
IDs in ``data/lectures.js`` rather than standalone lectures.

Render all scenes at preview quality with:

    python -m manim -ql --media_dir build/manim/lecture_02 \
      scenes/cs336/lecture_02/lecture_02_augmented.py

Individual scene classes may be passed after the file path.
"""

from __future__ import annotations

import math

from manim import (
    AnimationGroup,
    Arrow,
    Axes,
    Create,
    DOWN,
    FadeIn,
    FadeOut,
    GrowFromCenter,
    LEFT,
    Line,
    ORIGIN,
    Rectangle,
    ReplacementTransform,
    RIGHT,
    RoundedRectangle,
    Scene,
    Text,
    Transform,
    UP,
    VGroup,
    WHITE,
    Write,
)


# A course-site palette, deliberately quieter than a cinematic explainer.
PAPER = "#FAFAF7"
INK = "#252525"
MUTED = "#626262"
RULE = "#D8D5CF"
CARDINAL = "#8C1515"
CARDINAL_LIGHT = "#F3E4E2"
BLUE = "#276A8A"
BLUE_LIGHT = "#DCEBF1"
GOLD = "#A06400"
GOLD_LIGHT = "#F3E8CD"
GREEN = "#39734D"
GREEN_LIGHT = "#E1EFE5"
PURPLE = "#6C5A8E"

# Roboto is the site font.  Helvetica is bundled on macOS and has matching
# neutral grotesque metrics for deterministic offline Manim renders.
FONT = "Helvetica Neue"
MONO = "Menlo"


class CS336Scene(Scene):
    """Shared scene chrome for the Lecture 2 supplements."""

    segment_id = "L02"
    title = "Lecture 2"
    source_ref = "lecture_02.py"

    def setup(self) -> None:
        self.camera.background_color = PAPER

    def add_header(self) -> VGroup:
        marker = Text(
            f"CS336 · LECTURE 2 · {self.segment_id}",
            font=MONO,
            font_size=15,
            color=CARDINAL,
            weight="BOLD",
        )
        heading = Text(self.title, font=FONT, font_size=34, color=INK, weight="BOLD")
        title_group = VGroup(marker, heading).arrange(DOWN, aligned_edge=LEFT, buff=0.08)
        title_group.to_edge(UP, buff=0.26).to_edge(LEFT, buff=0.42)
        rule = Line(LEFT * 6.25, RIGHT * 6.25, color=RULE, stroke_width=1)
        rule.next_to(title_group, DOWN, buff=0.14)
        source = Text(self.source_ref, font=MONO, font_size=15, color=MUTED)
        source.to_edge(DOWN, buff=0.18).to_edge(RIGHT, buff=0.38)
        # Add the source first so the rule remains the final mobject. Several
        # scenes position their first body element relative to that rule.
        self.add(source, title_group, rule)
        return VGroup(title_group, rule, source)

    @staticmethod
    def label(text: str, *, size: int = 21, color: str = INK, bold: bool = False) -> Text:
        return Text(text, font=FONT, font_size=size, color=color, weight="BOLD" if bold else "NORMAL")

    @staticmethod
    def mono(text: str, *, size: int = 18, color: str = INK) -> Text:
        return Text(text, font=MONO, font_size=size, color=color)

    @staticmethod
    def callout(text: str, color: str = CARDINAL) -> VGroup:
        body = Text(text, font=FONT, font_size=20, color=color, weight="BOLD")
        box = RoundedRectangle(
            width=body.width + 0.42,
            height=body.height + 0.30,
            corner_radius=0.08,
            fill_color=WHITE,
            fill_opacity=1,
            stroke_color=color,
            stroke_width=1.5,
        )
        body.move_to(box)
        return VGroup(box, body)


def bit_field(label: str, widths: tuple[int, int, int], colors: tuple[str, str, str]) -> VGroup:
    """Return a compact sign/exponent/fraction allocation row."""

    sign, exponent, fraction = widths
    total = sign + exponent + fraction
    row_width = 6.0
    blocks = VGroup()
    names = (("sign", sign), ("exponent", exponent), ("fraction", fraction))
    for (name, count), color in zip(names, colors):
        width = row_width * count / total
        box = Rectangle(
            width=width,
            height=0.54,
            fill_color=color,
            fill_opacity=0.92,
            stroke_color=PAPER,
            stroke_width=1.2,
        )
        text = Text(f"{name} · {count}", font=FONT, font_size=15, color=WHITE, weight="BOLD")
        if text.width > width - 0.08:
            text.scale_to_fit_width(max(0.08, width - 0.08))
        text.move_to(box)
        blocks.add(VGroup(box, text))
    blocks.arrange(RIGHT, buff=0)
    name = Text(label, font=MONO, font_size=18, color=INK)
    slot = Rectangle(width=1.0, height=0.54, stroke_opacity=0, fill_opacity=0)
    name.move_to(slot).align_to(slot, RIGHT)
    return VGroup(VGroup(slot, name), blocks).arrange(RIGHT, buff=0.18)


class FloatingPointRange(CS336Scene):
    """L02-FP16-UNDERFLOW + L02-BF16-RANGE."""

    segment_id = "L02-FP16-UNDERFLOW · L02-BF16-RANGE"
    title = "Sixteen bits: choose range or local resolution"
    source_ref = "lecture_02.py · lines 134–152"

    def construct(self) -> None:
        self.add_header()

        fp32 = bit_field("FP32", (1, 8, 23), (INK, CARDINAL, BLUE))
        fp16 = bit_field("FP16", (1, 5, 10), (INK, CARDINAL, BLUE))
        bf16 = bit_field("BF16", (1, 8, 7), (INK, CARDINAL, BLUE))
        formats = VGroup(fp32, fp16, bf16).arrange(DOWN, aligned_edge=LEFT, buff=0.16)
        formats.scale(0.88).next_to(self.mobjects[-1], DOWN, buff=0.24).to_edge(LEFT, buff=0.62)

        self.play(FadeIn(fp32, shift=RIGHT * 0.18), run_time=0.8)
        self.play(FadeIn(fp16, shift=RIGHT * 0.18), run_time=0.7)
        self.play(FadeIn(bf16, shift=RIGHT * 0.18), run_time=0.7)

        bracket = Line(LEFT * 0.01, RIGHT * 2.98, color=CARDINAL, stroke_width=3)
        bracket.move_to(bf16[1][1].get_bottom() + DOWN * 0.17).align_to(bf16[1][1], LEFT)
        range_note = self.label("BF16 reuses FP32's 8 exponent bits", size=18, color=CARDINAL, bold=True)
        range_note.next_to(bracket, DOWN, buff=0.08).align_to(bracket, LEFT)
        self.play(Create(bracket), FadeIn(range_note, shift=UP * 0.1), run_time=0.8)

        # A log10 magnitude axis: the visual question is whether 10^-8 is inside
        # a format's non-zero representable interval.
        axis = Line(LEFT * 5.25, RIGHT * 5.25, color=INK, stroke_width=2)
        axis.shift(DOWN * 1.25)
        exponent_values = [-40, -20, -8, 0, 20, 40]

        def point_for(exp: float):
            return axis.get_start() + RIGHT * (10.5 * (exp + 40) / 80)

        ticks = VGroup()
        for exponent in exponent_values:
            tick = Line(DOWN * 0.08, UP * 0.08, color=INK, stroke_width=1.5).move_to(point_for(exponent))
            text = self.mono(f"10^{exponent}", size=14, color=MUTED).next_to(tick, DOWN, buff=0.07)
            ticks.add(VGroup(tick, text))
        axis_group = VGroup(axis, ticks)
        self.play(Create(axis), FadeIn(ticks), run_time=1.0)

        fp16_min = point_for(math.log10(5.96e-8))
        full_min = point_for(-40)
        zero_point = point_for(0)
        fp16_range = Line(fp16_min, zero_point, color=GOLD, stroke_width=8).shift(UP * 0.25)
        bf16_range = Line(full_min, zero_point, color=BLUE, stroke_width=8).shift(UP * 0.55)
        fp16_tag = self.mono("FP16 min subnormal = 5.96e-8", size=16, color=GOLD).next_to(fp16_range, UP, buff=0.04).align_to(fp16_range, LEFT)
        bf16_tag = self.mono("BF16 min subnormal ≈ 9.18e-41", size=16, color=BLUE).next_to(bf16_range, UP, buff=0.04).align_to(bf16_range, LEFT)
        self.play(Create(fp16_range), FadeIn(fp16_tag), run_time=0.8)
        self.play(Create(bf16_range), FadeIn(bf16_tag), run_time=0.8)

        signal_x = point_for(-8)
        signal_line = Line(signal_x + UP * 1.0, signal_x + DOWN * 0.52, color=CARDINAL, stroke_width=3)
        signal = self.callout("training signal = 1e-8")
        signal.next_to(signal_line, UP, buff=0.04)
        self.play(Create(signal_line), GrowFromCenter(signal), run_time=0.8)

        fp16_result = self.callout("FP16 → 0", GOLD)
        bf16_result = self.callout("BF16 → nonzero", BLUE)
        results = VGroup(fp16_result, bf16_result).arrange(RIGHT, buff=0.28).to_edge(DOWN, buff=0.45)
        self.play(FadeIn(fp16_result, shift=UP * 0.12), run_time=0.6)
        self.play(FadeIn(bf16_result, shift=UP * 0.12), run_time=0.6)

        tradeoff = self.label(
            "same 16-bit storage · FP16 keeps more fraction bits · BF16 keeps the dynamic range",
            size=16,
            color=MUTED,
        ).next_to(results, UP, buff=0.13)
        self.play(Write(tradeoff), run_time=1.0)
        self.wait(1.2)


class EinopsRearrangeHeads(CS336Scene):
    """L02-REARRANGE: make the hidden-axis factorization explicit."""

    segment_id = "L02-REARRANGE"
    title = "Rearrange: split, transform, then join"
    source_ref = "lecture_02.py · lines 261–276"

    def make_cells(
        self,
        colors: list[str],
        *,
        values: list[int | str] | None = None,
        cell_width: float = 0.55,
    ) -> VGroup:
        cells = VGroup()
        labels = values if values is not None else list(range(len(colors)))
        for value, color in zip(labels, colors):
            cell = Rectangle(
                width=cell_width,
                height=0.48,
                fill_color=color,
                fill_opacity=0.82,
                stroke_color=PAPER,
                stroke_width=1,
            )
            text = Text(str(value), font=MONO, font_size=15, color=WHITE)
            text.move_to(cell)
            cells.add(VGroup(cell, text))
        cells.arrange(RIGHT, buff=0)
        return cells

    def construct(self) -> None:
        self.add_header()

        code = self.mono(
            "rearrange(x, '... (heads hidden1) -> ... heads hidden1', heads=2)",
            size=17,
            color=INK,
        )
        code_box = RoundedRectangle(
            width=code.width + 0.5,
            height=0.62,
            corner_radius=0.08,
            fill_color="#F0F1F1",
            fill_opacity=1,
            stroke_color=RULE,
            stroke_width=1,
        )
        code.move_to(code_box)
        code_group = VGroup(code_box, code).next_to(self.mobjects[-1], DOWN, buff=0.24).to_edge(LEFT, buff=0.54)
        self.play(FadeIn(code_group, shift=DOWN * 0.08), run_time=0.7)

        head_colors = [CARDINAL, BLUE]
        colors = [head_colors[index // 4] for index in range(8)]
        flat = self.make_cells(colors, cell_width=0.48)
        flat.move_to(LEFT * 3.35 + UP * 0.42)
        flat_label = self.mono("x[seq, total_hidden=8]", size=19).next_to(flat, UP, buff=0.16)
        factor = self.label("8 = 2 heads × hidden1=4", size=18, color=MUTED).next_to(flat, DOWN, buff=0.15)
        self.play(FadeIn(flat_label), Create(flat), FadeIn(factor), run_time=1.0)

        split_rows = VGroup()
        for head in range(2):
            row = self.make_cells(
                [head_colors[head]] * 4,
                values=list(range(head * 4, (head + 1) * 4)),
            )
            tag = self.mono(f"head {head}", size=16, color=head_colors[head])
            split_rows.add(VGroup(tag, row).arrange(RIGHT, buff=0.18))
        split_rows.arrange(DOWN, aligned_edge=LEFT, buff=0.18).move_to(RIGHT * 3.35 + UP * 0.38)
        split_label = self.mono("x[seq, heads=2, hidden1=4]", size=18).next_to(split_rows, UP, buff=0.16)
        arrow = Arrow(flat.get_right() + RIGHT * 0.2, split_rows.get_left() + LEFT * 0.2, color=CARDINAL, buff=0.08)
        arrow_text = self.label("name the two axes", size=16, color=CARDINAL, bold=True).next_to(arrow, UP, buff=0.05)
        self.play(Create(arrow), FadeIn(arrow_text), run_time=0.5)
        self.play(
            AnimationGroup(
                *[FadeIn(row, shift=RIGHT * 0.25) for row in split_rows],
                lag_ratio=0.18,
            ),
            FadeIn(split_label),
            run_time=1.2,
        )

        # A named axis is a bookkeeping change, not a data copy.  Highlight
        # one whole head to show what becomes independently addressable.
        head_highlight = RoundedRectangle(
            width=split_rows[1].width + 0.2,
            height=split_rows[1].height + 0.15,
            corner_radius=0.06,
            stroke_color=BLUE,
            stroke_width=3,
            fill_opacity=0,
        ).move_to(split_rows[1])
        head_note = self.callout("hidden1 is now independently addressable", BLUE)
        head_note.move_to(LEFT * 2.45 + DOWN * 1.1)
        self.play(Create(head_highlight), FadeIn(head_note, shift=UP * 0.1), run_time=0.8)

        transformed_rows = VGroup()
        for head in range(2):
            row = self.make_cells(
                [head_colors[head]] * 3,
                values=[f"y{head * 3 + idx}" for idx in range(3)],
            )
            tag = self.mono(f"head {head}", size=16, color=head_colors[head])
            transformed_rows.add(VGroup(tag, row).arrange(RIGHT, buff=0.18))
        transformed_rows.arrange(DOWN, aligned_edge=LEFT, buff=0.18).move_to(split_rows)
        transformed_label = self.mono("x[seq, heads=2, hidden2=3]", size=18).move_to(split_label)
        einsum_code = self.mono(
            "einsum(x, w, '... hidden1, hidden1 hidden2 -> ... hidden2')",
            size=15,
            color=INK,
        ).move_to(LEFT * 2.7 + DOWN * 1.72)
        w_note = self.callout("w[hidden1=4, hidden2=3] acts inside each head", CARDINAL)
        w_note.next_to(einsum_code, UP, buff=0.14)
        self.play(
            FadeOut(head_highlight),
            FadeOut(head_note),
            FadeIn(einsum_code),
            FadeIn(w_note),
            Transform(split_rows, transformed_rows),
            Transform(split_label, transformed_label),
            run_time=1.2,
        )

        join_code = self.mono("rearrange(x, '... heads hidden2 -> ... (heads hidden2)')", size=15, color=INK)
        join_code.move_to(LEFT * 3.0 + DOWN * 2.56)
        joined = self.make_cells(
            [head_colors[index // 3] for index in range(6)],
            values=[f"y{idx}" for idx in range(6)],
            cell_width=0.46,
        ).move_to(RIGHT * 3.38 + DOWN * 2.44)
        joined_label = self.mono("x[seq, total_hidden=6]", size=16).next_to(joined, UP, buff=0.09)
        join_arrow = Arrow(
            split_rows.get_bottom() + DOWN * 0.12,
            joined.get_top() + UP * 0.12,
            color=MUTED,
            buff=0.08,
        )
        join_arrow.shift(RIGHT * 0.52)
        join_label = self.label("join heads", size=14, color=MUTED).next_to(join_arrow, LEFT, buff=0.06)
        self.play(FadeIn(join_code), Create(join_arrow), FadeIn(join_label), run_time=0.7)
        self.play(Create(joined), FadeIn(joined_label), run_time=0.9)

        invariant = self.label(
            "Take-away: rearrange names axes; einsum changes hidden1 → hidden2; rearrange joins heads.",
            size=17,
            color=INK,
        ).to_edge(DOWN, buff=0.68)
        self.play(Write(invariant), run_time=1.0)
        self.wait(1.2)


class RooflineModel(CS336Scene):
    """L02-ROOFLINE: the lecture's H100 BF16 roofline example."""

    segment_id = "L02-ROOFLINE"
    title = "H100 BF16 roofline: bandwidth → compute"
    source_ref = "lecture_02.py · lines 471–481"

    def construct(self) -> None:
        self.add_header()

        axes = Axes(
            x_range=[-1, 3.1, 1],
            y_range=[-0.3, 3.2, 1],
            x_length=8.8,
            y_length=4.7,
            axis_config={"color": INK, "stroke_width": 2, "include_ticks": False},
            tips=True,
        ).shift(DOWN * 0.45 + LEFT * 0.25)
        x_label = self.label("arithmetic intensity I (FLOP/byte) · log scale →", size=17)
        x_label.next_to(axes, DOWN, buff=0.12)
        y_label = self.label("throughput (TFLOP/s) · log scale", size=16)
        y_label.rotate(math.pi / 2).next_to(axes, LEFT, buff=0.12)

        tick_labels = VGroup()
        for exponent, value in [(-1, "0.1"), (0, "1"), (1, "10"), (2, "100"), (3, "1000")]:
            tick_labels.add(self.mono(value, size=13, color=MUTED).next_to(axes.c2p(exponent, -0.3), DOWN, buff=0.05))
        for exponent, value in [(0, "1"), (1, "10"), (2, "100"), (3, "1000")]:
            tick_labels.add(self.mono(value, size=13, color=MUTED).next_to(axes.c2p(-1, exponent), LEFT, buff=0.05))
        self.play(Create(axes), FadeIn(tick_labels), FadeIn(x_label), FadeIn(y_label), run_time=0.9)

        # Both axes hold log10 values. H100 BF16: 3.35 TB/s bandwidth and
        # 989.5 TFLOP/s peak imply a knee at 989.5 / 3.35 ≈ 295 FLOP/byte.
        bandwidth_tb_s = 3.35
        peak_tflop_s = 989.5
        knee_intensity = peak_tflop_s / bandwidth_tb_s
        knee_x = math.log10(knee_intensity)
        peak_y = math.log10(peak_tflop_s)
        memory_start_x = -0.75
        memory_start_y = math.log10(bandwidth_tb_s) + memory_start_x
        memory_line = Line(
            axes.c2p(memory_start_x, memory_start_y),
            axes.c2p(knee_x, peak_y),
            color=BLUE,
            stroke_width=5,
        )
        compute_line = Line(axes.c2p(knee_x, peak_y), axes.c2p(3.05, peak_y), color=CARDINAL, stroke_width=5)
        memory_label = self.label("3.35 TB/s × I", size=17, color=BLUE, bold=True)
        memory_label.rotate(memory_line.get_angle()).next_to(memory_line.get_center(), UP, buff=0.16)
        compute_label = self.label("989.5 TFLOP/s", size=17, color=CARDINAL, bold=True).next_to(compute_line, UP, buff=0.12)
        self.play(Create(memory_line), FadeIn(memory_label), run_time=1.0)
        self.play(Create(compute_line), FadeIn(compute_label), run_time=0.8)

        knee = Line(axes.c2p(knee_x, -0.3), axes.c2p(knee_x, peak_y), color=MUTED, stroke_width=1.5)
        knee.set_stroke(opacity=0.65)
        knee_label = self.mono("knee I* ≈ 295", size=14, color=MUTED)
        knee_label.next_to(knee.get_bottom() + UP * 0.22, RIGHT, buff=0.08)
        self.play(Create(knee), FadeIn(knee_label), run_time=0.7)

        # Arithmetic intensities follow the lecture's BF16 examples.
        operations = [
            ("ReLU", 0.25, CARDINAL),
            ("dot", 0.50, GOLD),
            ("matvec", 1.0, PURPLE),
            ("GELU", 5.0, GREEN),
            ("1024x1024\nmatmul", 341.0, BLUE),
        ]

        points = VGroup()
        for index, (name, intensity, color) in enumerate(operations):
            x = math.log10(intensity)
            y = min(math.log10(bandwidth_tb_s * intensity), peak_y)
            dot = RoundedRectangle(
                width=0.17,
                height=0.17,
                corner_radius=0.04,
                fill_color=color,
                fill_opacity=1,
                stroke_color=PAPER,
                stroke_width=1,
            ).move_to(axes.c2p(x, y))
            tag = self.mono(f"{name}\nI≈{intensity:g}", size=14, color=color)
            direction = DOWN if index in (0, 2, 4) else UP
            tag.next_to(dot, direction, buff=0.08)
            points.add(VGroup(dot, tag))

        self.play(
            AnimationGroup(*[GrowFromCenter(point[0]) for point in points], lag_ratio=0.18),
            run_time=1.3,
        )
        self.play(AnimationGroup(*[FadeIn(point[1]) for point in points], lag_ratio=0.12), run_time=1.0)

        memory_region = self.callout("memory-bound: move fewer bytes", BLUE)
        compute_region = self.callout("compute-bound: use faster arithmetic", CARDINAL)
        region_notes = VGroup(memory_region, compute_region).arrange(RIGHT, buff=0.35).to_edge(DOWN, buff=0.42)
        self.play(FadeIn(memory_region, shift=UP * 0.1), run_time=0.7)
        self.play(FadeIn(compute_region, shift=UP * 0.1), run_time=0.7)

        equation = self.mono("H100 BF16 · log-log · knee: 989.5 / 3.35 ≈ 295 FLOP/byte", size=16, color=INK)
        equation.next_to(self.mobjects[1], DOWN, buff=0.12).to_edge(RIGHT, buff=0.48)
        self.play(Write(equation), run_time=1.0)
        self.wait(1.3)
