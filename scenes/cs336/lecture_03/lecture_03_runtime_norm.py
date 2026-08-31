"""CS336 Lecture 3: RMSNorm, runtime, and bias removal.

Render the site-preview clip with::

    python -m manim -ql --media_dir build/manim/lecture_03_runtime_norm \
      scenes/cs336/lecture_03/lecture_03_runtime_norm.py L03RuntimeNorm

The scene follows official Lecture 3 slides 14--19.  It deliberately frames
RMSNorm and bias removal as runtime-engineering choices, not quality laws.
"""

from __future__ import annotations

from manim import (
    AnimationGroup,
    Arrow,
    Create,
    DOWN,
    FadeIn,
    FadeOut,
    GrowArrow,
    LEFT,
    Line,
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
    config,
)


PAPER = "#FAFAF7"
INK = "#252525"
MUTED = "#626262"
RULE = "#D8D5CF"
CARDINAL = "#8C1515"
CARDINAL_SOFT = "#F1E3E3"
GREEN = "#2F6B4F"
GREEN_SOFT = "#E5EFE9"
BLUE = "#3B6E8F"
BLUE_SOFT = "#E5EDF2"
AMBER = "#A86412"
AMBER_SOFT = "#F3E9D7"
PANEL = "#F4F3EF"
FONT = "Helvetica Neue"
MONO = "Menlo"

config.background_color = PAPER


def label(
    text: str,
    size: int = 24,
    color: str = INK,
    weight: str = "NORMAL",
    font: str = FONT,
) -> Text:
    return Text(text, font=font, font_size=size, color=color, weight=weight)


def box(
    width: float,
    height: float,
    *,
    fill: str = WHITE,
    stroke: str = RULE,
    radius: float = 0.08,
    stroke_width: float = 1.2,
) -> RoundedRectangle:
    return RoundedRectangle(
        width=width,
        height=height,
        corner_radius=radius,
        fill_color=fill,
        fill_opacity=1,
        stroke_color=stroke,
        stroke_width=stroke_width,
    )


def chip(text: str, color: str, fill: str, width: float = 1.2) -> VGroup:
    body = box(width, 0.42, fill=fill, stroke=color, radius=0.05, stroke_width=1.3)
    copy = label(text, 13, color, "BOLD", MONO).move_to(body)
    return VGroup(body, copy)


class L03RuntimeNorm(Scene):
    """Connect normalization arithmetic to data movement and bias removal."""

    def setup(self) -> None:
        self.camera.background_color = PAPER

    def chrome(self) -> VGroup:
        kicker = label(
            "CS336 · LECTURE 3 · L03-RMSNORM → L03-NORM-RUNTIME → L03-NO-BIAS",
            14,
            CARDINAL,
            "BOLD",
            MONO,
        )
        heading = label("Normalization is a runtime-engineering choice", 33, INK, "BOLD")
        title = VGroup(kicker, heading).arrange(DOWN, aligned_edge=LEFT, buff=0.09)
        title.to_edge(UP, buff=0.27).to_edge(LEFT, buff=0.48)
        rule = Line(LEFT * 6.18, RIGHT * 6.18, color=RULE, stroke_width=1)
        rule.next_to(title, DOWN, buff=0.15)
        source = label("lecture_03.pdf · slides 14–19", 14, MUTED, font=MONO)
        source.to_edge(DOWN, buff=0.18).to_edge(RIGHT, buff=0.42)
        return VGroup(title, rule, source)

    @staticmethod
    def step_strip(active: int) -> VGroup:
        names = ("1  NORMALIZE", "2  RUNTIME", "3  DROP BIAS")
        steps = VGroup()
        for idx, name in enumerate(names):
            is_active = idx == active
            item = chip(
                name,
                CARDINAL if is_active else MUTED,
                CARDINAL_SOFT if is_active else PANEL,
                2.25,
            )
            steps.add(item)
        steps.arrange(RIGHT, buff=0.16).move_to(UP * 2.1)
        return steps

    @staticmethod
    def norm_panel(*, rms: bool) -> VGroup:
        panel = box(5.82, 3.45, fill=WHITE, stroke=GREEN if rms else BLUE)
        name = label("RMSNORM" if rms else "LAYERNORM", 19, GREEN if rms else BLUE, "BOLD", MONO)
        name.move_to(panel.get_top() + DOWN * 0.34 + LEFT * 1.98)

        if rms:
            formula = label("y = x / √(mean(x²) + ε) · γ", 18, INK, font=MONO)
            ops = VGroup(
                chip("x²", GREEN, GREEN_SOFT, 0.75),
                chip("mean", GREEN, GREEN_SOFT, 1.02),
                chip("√", GREEN, GREEN_SOFT, 0.66),
                chip("× γ", GREEN, GREEN_SOFT, 0.84),
            ).arrange(RIGHT, buff=0.12)
            removed = VGroup(
                label("removed", 13, MUTED, "BOLD", MONO),
                chip("mean(x)", CARDINAL, CARDINAL_SOFT, 1.24),
                chip("+ β", CARDINAL, CARDINAL_SOFT, 0.84),
            ).arrange(RIGHT, buff=0.12)
            for item in removed[1:]:
                strike = Line(item.get_left() + RIGHT * 0.08, item.get_right() + LEFT * 0.08, color=CARDINAL, stroke_width=2)
                item.add(strike)
            count = label("one reduction · γ only · d learned values", 15, GREEN, "BOLD")
        else:
            formula = label("y = (x − μ) / √(var(x) + ε) · γ + β", 17, INK, font=MONO)
            ops = VGroup(
                chip("mean μ", BLUE, BLUE_SOFT, 1.15),
                chip("center", BLUE, BLUE_SOFT, 1.08),
                chip("variance", BLUE, BLUE_SOFT, 1.24),
                chip("× γ + β", BLUE, BLUE_SOFT, 1.15),
            ).arrange(RIGHT, buff=0.1)
            removed = VGroup()
            count = label("mean + variance work · γ and β · 2d learned values", 15, BLUE, "BOLD")

        formula.move_to(panel.get_top() + DOWN * 0.93)
        ops.move_to(panel.get_center() + DOWN * 0.13)
        count.move_to(panel.get_bottom() + UP * 0.38)
        if rms:
            removed.move_to(panel.get_center() + DOWN * 0.82)
        return VGroup(panel, name, formula, ops, removed, count)

    @staticmethod
    def runtime_panel() -> VGroup:
        panel = box(11.2, 3.58, fill=WHITE)
        floops_title = label("SCHEMATIC FLOP VIEW", 15, MUTED, "BOLD", MONO)
        floops_title.move_to(panel.get_top() + DOWN * 0.36 + LEFT * 3.98)
        wall_title = label("WALL-CLOCK VIEW", 16, MUTED, "BOLD", MONO)
        wall_title.move_to(panel.get_top() + DOWN * 0.36 + RIGHT * 2.96)

        track = box(4.25, 0.55, fill=PANEL, stroke=RULE, radius=0.04)
        matmul = Rectangle(
            width=3.86,
            height=0.55,
            fill_color=BLUE,
            fill_opacity=1,
            stroke_width=0,
        ).align_to(track, LEFT)
        norm = Rectangle(
            width=0.39,
            height=0.55,
            fill_color=CARDINAL,
            fill_opacity=1,
            stroke_width=0,
        ).next_to(matmul, RIGHT, buff=0)
        flop_bar = VGroup(track, matmul, norm).move_to(LEFT * 3.0 + UP * 0.38)
        bar_labels = VGroup(
            label("matrix multiplies", 13, WHITE, "BOLD").move_to(matmul),
            label("norm", 11, WHITE, "BOLD").move_to(norm),
        )
        flop_note = label("few FLOPs does not imply zero time", 15, INK, "BOLD")
        flop_note.next_to(flop_bar, DOWN, buff=0.25)

        read = chip("read x", AMBER, AMBER_SOFT, 1.12)
        reduce = chip("reduce", AMBER, AMBER_SOFT, 1.12)
        write = chip("write y", AMBER, AMBER_SOFT, 1.12)
        lane = VGroup(read, reduce, write).arrange(RIGHT, buff=0.62).move_to(RIGHT * 2.95 + UP * 0.38)
        arrows = VGroup(
            Arrow(read.get_right(), reduce.get_left(), buff=0.07, color=AMBER, stroke_width=2.5, max_tip_length_to_length_ratio=0.15),
            Arrow(reduce.get_right(), write.get_left(), buff=0.07, color=AMBER, stroke_width=2.5, max_tip_length_to_length_ratio=0.15),
        )
        move_note = label("data movement + reductions can still matter", 15, AMBER, "BOLD")
        move_note.next_to(lane, DOWN, buff=0.25)

        lesson = box(7.55, 0.65, fill=CARDINAL_SOFT, stroke=CARDINAL, radius=0.06)
        lesson_copy = label("FLOPs ≠ runtime", 22, CARDINAL, "BOLD", MONO).move_to(lesson)
        lesson_group = VGroup(lesson, lesson_copy).move_to(panel.get_bottom() + UP * 0.58)
        return VGroup(
            panel,
            floops_title,
            wall_title,
            flop_bar,
            bar_labels,
            flop_note,
            lane,
            arrows,
            move_note,
            lesson_group,
        )

    @staticmethod
    def bias_panel() -> VGroup:
        panel = box(11.2, 3.62, fill=WHITE)
        context = label("UNGATED FFN EXAMPLE", 15, MUTED, "BOLD", MONO)
        context.move_to(panel.get_top() + DOWN * 0.35 + LEFT * 4.08)

        with_bias = VGroup(
            label("with bias", 16, BLUE, "BOLD", MONO),
            label("FFN(x) = σ(xW₁ + b₁)W₂ + b₂", 22, INK, font=MONO),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.15)
        with_bias.move_to(panel.get_center() + UP * 0.62 + LEFT * 2.82)

        no_bias = VGroup(
            label("common modern choice", 16, GREEN, "BOLD", MONO),
            label("FFN(x) = σ(xW₁)W₂", 22, GREEN, "BOLD", MONO),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.15)
        no_bias.move_to(panel.get_center() + UP * 0.62 + RIGHT * 3.0)

        arrow = Arrow(
            with_bias.get_right() + RIGHT * 0.1,
            no_bias.get_left() + LEFT * 0.1,
            buff=0,
            color=CARDINAL,
            stroke_width=3,
            max_tip_length_to_length_ratio=0.12,
        )

        resource = VGroup(
            chip("− (d_ff + d_model) bias values / layer", CARDINAL, CARDINAL_SOFT, 4.6),
            chip("fewer bias loads + adds", GREEN, GREEN_SOFT, 3.2),
        ).arrange(RIGHT, buff=0.28).move_to(panel.get_center() + DOWN * 0.66)

        boundary = label(
            "Course rationale: memory + optimization stability · realized speed depends on fusion and hardware",
            15,
            INK,
        ).move_to(panel.get_bottom() + UP * 0.42)
        return VGroup(panel, context, with_bias, no_bias, arrow, resource, boundary)

    def construct(self) -> None:
        chrome = self.chrome()
        self.play(FadeIn(chrome, shift=DOWN * 0.07), run_time=0.55)

        step1 = self.step_strip(0)
        self.play(FadeIn(step1), run_time=0.35)
        layer = self.norm_panel(rms=False).scale(0.93).shift(LEFT * 3.08 + DOWN * 0.42)
        rms = self.norm_panel(rms=True).scale(0.93).shift(RIGHT * 3.08 + DOWN * 0.42)
        self.play(FadeIn(layer[0:3]), FadeIn(rms[0:3]), run_time=0.55)
        self.play(
            AnimationGroup(*[FadeIn(item, shift=UP * 0.05) for item in layer[3]], lag_ratio=0.08),
            AnimationGroup(*[FadeIn(item, shift=UP * 0.05) for item in rms[3]], lag_ratio=0.08),
            run_time=0.8,
        )
        self.play(FadeIn(layer[5]), FadeIn(rms[4]), FadeIn(rms[5]), run_time=0.6)
        self.wait(0.25)

        step2 = self.step_strip(1)
        runtime = self.runtime_panel().shift(DOWN * 0.39)
        self.play(
            ReplacementTransform(step1, step2),
            FadeOut(layer, shift=LEFT * 0.12),
            FadeOut(rms, shift=RIGHT * 0.12),
            run_time=0.55,
        )
        self.play(FadeIn(runtime[0:3]), run_time=0.35)
        self.play(FadeIn(runtime[3]), FadeIn(runtime[4]), FadeIn(runtime[6]), GrowArrow(runtime[7][0]), GrowArrow(runtime[7][1]), run_time=0.8)
        self.play(FadeIn(runtime[5]), FadeIn(runtime[8]), run_time=0.4)
        self.play(FadeIn(runtime[9], shift=UP * 0.08), run_time=0.45)
        self.wait(0.35)

        step3 = self.step_strip(2)
        bias = self.bias_panel().shift(DOWN * 0.39)
        self.play(ReplacementTransform(step2, step3), FadeOut(runtime), run_time=0.5)
        self.play(FadeIn(bias[0:3]), run_time=0.55)
        self.play(GrowArrow(bias[4]), FadeIn(bias[3], shift=RIGHT * 0.08), run_time=0.55)
        self.play(FadeIn(bias[5], shift=UP * 0.08), FadeIn(bias[6]), run_time=0.55)

        takeaway = label(
            "Take-away: simplify traffic and elementwise work; verify quality and wall-clock empirically.",
            18,
            INK,
            "BOLD",
        ).to_edge(DOWN, buff=0.72)
        self.play(FadeIn(takeaway, shift=UP * 0.06), run_time=0.5)
        self.wait(1.0)
