"""Activation checkpointing supplement for CS336 Lecture 2.

This is a continuous, source-aligned explanation for the existing segments
``L02-ACTIVATION-MEMORY``, ``L02-CHECKPOINTING``, and
``L02-CHECKPOINT-FREQUENCY``.  It deliberately keeps the visual language close
to the CS336 course site: light paper, cardinal accents, compact engineering
labels, and explicit accounting.

Render the site preview with:

    python -m manim -ql --media_dir build/manim/lecture_02_checkpointing \
      scenes/cs336/lecture_02/lecture_02_checkpointing.py \
      ActivationCheckpointingTradeoff
"""

from __future__ import annotations

from manim import (
    AnimationGroup,
    Arrow,
    Create,
    DOWN,
    FadeIn,
    FadeOut,
    LEFT,
    Line,
    Rectangle,
    RIGHT,
    RoundedRectangle,
    Scene,
    Text,
    UP,
    VGroup,
    WHITE,
    Write,
)


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

FONT = "Helvetica Neue"
MONO = "Menlo"


class ActivationCheckpointingTradeoff(Scene):
    """Store every activation, rematerialize, then choose a frequency."""

    def setup(self) -> None:
        self.camera.background_color = PAPER

    @staticmethod
    def label(text: str, *, size: int = 20, color: str = INK, bold: bool = False) -> Text:
        return Text(
            text,
            font=FONT,
            font_size=size,
            color=color,
            weight="BOLD" if bold else "NORMAL",
        )

    @staticmethod
    def mono(text: str, *, size: int = 17, color: str = INK) -> Text:
        return Text(text, font=MONO, font_size=size, color=color)

    def add_header(self) -> None:
        marker = self.mono(
            "CS336 · LECTURE 2 · L02-ACTIVATION-MEMORY → CHECKPOINT-FREQUENCY",
            size=13,
            color=CARDINAL,
        )
        heading = self.label("Activation checkpointing: memory ↔ recomputation", size=31, bold=True)
        title = VGroup(marker, heading).arrange(DOWN, aligned_edge=LEFT, buff=0.06)
        title.to_edge(UP, buff=0.23).to_edge(LEFT, buff=0.40)
        rule = Line(LEFT * 6.28, RIGHT * 6.28, color=RULE, stroke_width=1)
        rule.next_to(title, DOWN, buff=0.12)
        source = self.mono("lecture_02.py · lines 733–773; checkpoint call at 786", size=13, color=MUTED)
        source.to_edge(DOWN, buff=0.16).to_edge(RIGHT, buff=0.36)
        self.add(title, rule, source)

    def layer(self, index: int) -> VGroup:
        box = RoundedRectangle(
            width=0.76,
            height=0.52,
            corner_radius=0.06,
            fill_color=WHITE,
            fill_opacity=1,
            stroke_color=INK,
            stroke_width=1.4,
        )
        text = self.mono(f"L{index}", size=15)
        text.move_to(box)
        return VGroup(box, text)

    def activation(self, index: int, *, checkpoint: bool = False) -> VGroup:
        color = CARDINAL if checkpoint else BLUE
        box = Rectangle(
            width=0.60,
            height=0.25,
            fill_color=CARDINAL_LIGHT if checkpoint else BLUE_LIGHT,
            fill_opacity=1,
            stroke_color=color,
            stroke_width=1.3,
        )
        text = self.mono(f"h{index}", size=11, color=color)
        text.move_to(box)
        return VGroup(box, text)

    def schedule_row(
        self,
        label: str,
        memory: str,
        recompute: str,
        *,
        fill: str,
        accent: str,
    ) -> VGroup:
        background = RoundedRectangle(
            width=11.3,
            height=0.70,
            corner_radius=0.06,
            fill_color=fill,
            fill_opacity=1,
            stroke_color=RULE,
            stroke_width=1,
        )
        name = self.label(label, size=18, color=accent, bold=True)
        mem = self.mono(memory, size=16, color=INK)
        work = self.mono(recompute, size=16, color=INK)
        name.move_to(background.get_left() + RIGHT * 1.55)
        mem.move_to(background.get_left() + RIGHT * 5.10)
        work.move_to(background.get_left() + RIGHT * 8.92)
        return VGroup(background, name, mem, work)

    def construct(self) -> None:
        self.add_header()

        premise = self.label(
            "Training backward needs intermediate activations",
            size=21,
            color=INK,
            bold=True,
        ).move_to(UP * 2.0)
        self.play(FadeIn(premise, shift=DOWN * 0.08), run_time=0.45)

        layers = VGroup(*[self.layer(i) for i in range(1, 10)])
        layers.arrange(RIGHT, buff=0.38).move_to(UP * 0.85)
        connectors = VGroup(
            *[
                Arrow(
                    layers[i].get_right(),
                    layers[i + 1].get_left(),
                    buff=0.05,
                    color=MUTED,
                    stroke_width=1.5,
                    max_tip_length_to_length_ratio=0.22,
                )
                for i in range(len(layers) - 1)
            ]
        )
        forward_tag = self.mono("forward →", size=15, color=MUTED)
        forward_tag.next_to(layers, UP, buff=0.12).align_to(layers, LEFT)
        self.play(
            Create(connectors),
            AnimationGroup(*[FadeIn(layer) for layer in layers], lag_ratio=0.07),
            FadeIn(forward_tag),
            run_time=1.15,
        )

        activations = VGroup(*[self.activation(i) for i in range(1, 10)])
        for activation, layer in zip(activations, layers):
            activation.next_to(layer, DOWN, buff=0.22)
        shelf_line = Line(
            activations.get_left() + LEFT * 0.15 + DOWN * 0.21,
            activations.get_right() + RIGHT * 0.15 + DOWN * 0.21,
            color=RULE,
            stroke_width=2,
        )
        shelf_tag = self.label("activation shelf", size=15, color=BLUE, bold=True)
        shelf_tag.next_to(shelf_line, DOWN, buff=0.08).align_to(shelf_line, LEFT)
        self.play(
            AnimationGroup(*[FadeIn(cell, shift=DOWN * 0.08) for cell in activations], lag_ratio=0.08),
            Create(shelf_line),
            FadeIn(shelf_tag),
            run_time=1.15,
        )

        formula_box = RoundedRectangle(
            width=6.95,
            height=0.76,
            corner_radius=0.06,
            fill_color=WHITE,
            fill_opacity=1,
            stroke_color=BLUE,
            stroke_width=1.4,
        ).move_to(DOWN * 1.25 + LEFT * 1.7)
        formula = self.mono("BF16 shelf: 2 × B × D × L = 2 MiB", size=18, color=BLUE)
        formula.move_to(formula_box)
        scope = self.label("B=64, D=1024, L=16 · simplified activation term", size=14, color=MUTED)
        scope.next_to(formula_box, DOWN, buff=0.08).align_to(formula_box, LEFT)
        peak = VGroup(
            RoundedRectangle(
                width=2.55,
                height=0.76,
                corner_radius=0.06,
                fill_color=CARDINAL_LIGHT,
                fill_opacity=1,
                stroke_color=CARDINAL,
                stroke_width=1.4,
            ),
            self.label("peak grows as O(L)", size=17, color=CARDINAL, bold=True),
        )
        peak[1].move_to(peak[0])
        peak.move_to(DOWN * 1.25 + RIGHT * 3.25)
        self.play(FadeIn(VGroup(formula_box, formula, scope)), FadeIn(peak), run_time=0.75)

        checkpoint_indices = {2, 5, 8}  # visual h3, h6, h9 in zero-based indexing
        discard = [activation for i, activation in enumerate(activations) if i not in checkpoint_indices]
        keep = [activation for i, activation in enumerate(activations) if i in checkpoint_indices]
        checkpoint_tag = self.label(
            "selective: keep h3, h6, h9 · discard intermediates",
            size=18,
            color=CARDINAL,
            bold=True,
        ).move_to(premise)
        self.play(
            FadeOut(premise),
            FadeIn(checkpoint_tag),
            *[cell.animate.set_opacity(0.10) for cell in discard],
            *[
                cell[0].animate.set_fill(CARDINAL_LIGHT).set_stroke(CARDINAL)
                for cell in keep
            ],
            *[cell[1].animate.set_color(CARDINAL) for cell in keep],
            FadeOut(VGroup(formula_box, formula, scope, peak)),
            run_time=0.9,
        )

        recompute = Arrow(
            layers[5].get_bottom() + DOWN * 0.62,
            layers[8].get_bottom() + DOWN * 0.62,
            buff=0.05,
            color=GOLD,
            stroke_width=3,
        )
        recompute_label = self.label("recompute h7 → h8", size=16, color=GOLD, bold=True)
        recompute_label.next_to(recompute, DOWN, buff=0.08)
        backward = Arrow(
            layers[8].get_top() + UP * 0.32,
            layers[5].get_top() + UP * 0.32,
            buff=0.05,
            color=CARDINAL,
            stroke_width=3,
        )
        backward_label = self.mono("← backward", size=15, color=CARDINAL)
        backward_label.next_to(backward, UP, buff=0.06)
        code = self.mono("checkpoint(layer, x)  # rematerialize on backward", size=15, color=INK)
        code.move_to(DOWN * 1.55)
        self.play(Create(recompute), FadeIn(recompute_label), run_time=0.65)
        self.play(Create(backward), FadeIn(backward_label), FadeIn(code), run_time=0.75)

        network = VGroup(
            checkpoint_tag,
            layers,
            connectors,
            forward_tag,
            activations,
            shelf_line,
            shelf_tag,
            recompute,
            recompute_label,
            backward,
            backward_label,
            code,
        )
        self.play(FadeOut(network), run_time=0.55)

        column_headers = VGroup(
            self.label("schedule", size=15, color=MUTED, bold=True),
            self.label("saved activation memory", size=15, color=MUTED, bold=True),
            self.label("recomputation", size=15, color=MUTED, bold=True),
        )
        column_headers[0].move_to(LEFT * 4.18 + UP * 1.78)
        column_headers[1].move_to(LEFT * 0.65 + UP * 1.78)
        column_headers[2].move_to(RIGHT * 3.25 + UP * 1.78)

        rows = VGroup(
            self.schedule_row("every layer", "O(L)", "none", fill=BLUE_LIGHT, accent=BLUE),
            self.schedule_row("no layers", "O(1)", "O(L²)", fill=CARDINAL_LIGHT, accent=CARDINAL),
            self.schedule_row("every √L", "O(√L)", "O(L)", fill=GREEN_LIGHT, accent=GREEN),
        ).arrange(DOWN, buff=0.16).move_to(DOWN * 0.10)
        self.play(FadeIn(column_headers), run_time=0.35)
        self.play(
            AnimationGroup(*[FadeIn(row, shift=RIGHT * 0.12) for row in rows], lag_ratio=0.18),
            run_time=1.15,
        )

        takeaway = self.label(
            "Take-away: checkpoint placement is a resource schedule, not free memory.",
            size=19,
            color=INK,
            bold=True,
        ).to_edge(DOWN, buff=0.56)
        self.play(Write(takeaway), run_time=0.8)
        self.wait(0.55)
