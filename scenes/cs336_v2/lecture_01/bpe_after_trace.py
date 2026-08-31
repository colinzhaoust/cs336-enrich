"""Lecture 1 V2 flagship: replay frozen BPE rules on held-out text.

This is the deterministic, silent traversal of a browser stepper.  It begins
after Percy's uninterrupted training trace and therefore does not recount pair
frequencies or re-teach the training loop.

Animatic:
    python -m manim -ql --fps 15 --media_dir build/manim/v2/lecture_01_animatic \
      scenes/cs336_v2/lecture_01/bpe_after_trace.py BPEAfterTraceV2

Production source render:
    python -m manim -qh --fps 30 --media_dir build/manim/v2/lecture_01 \
      scenes/cs336_v2/lecture_01/bpe_after_trace.py BPEAfterTraceV2
"""

from __future__ import annotations

from manim import (
    BOLD,
    DOWN,
    FadeIn,
    FadeOut,
    Indicate,
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
)


PAPER = "#FAFAF7"
INK = "#252525"
MUTED = "#5F6368"
RULE = "#D7D4CD"
PANEL = "#F2F0EB"
CARDINAL = "#8C1515"
CARDINAL_PALE = "#F6E8E7"
BLUE = "#2E657D"
BLUE_PALE = "#E8F0F3"
GREEN = "#356548"
GREEN_PALE = "#E8F1EA"
SANS = "Helvetica Neue"
MONO = "Menlo"


def text(label: str, size: int, color: str = INK, *, weight: str = "NORMAL") -> Text:
    return Text(label, font=SANS, font_size=size, color=color, weight=weight)


def mono(label: str, size: int, color: str = INK, *, weight: str = "NORMAL") -> Text:
    return Text(label, font=MONO, font_size=size, color=color, weight=weight)


def fit_width(mobject, maximum: float):
    if mobject.width > maximum:
        mobject.scale_to_fit_width(maximum)
    return mobject


def pill(
    label: str,
    *,
    width: float,
    fill: str = PANEL,
    stroke: str = RULE,
    color: str = INK,
    size: int = 56,
) -> VGroup:
    box = RoundedRectangle(
        width=width,
        height=0.70,
        corner_radius=0.10,
        fill_color=fill,
        fill_opacity=1,
        stroke_color=stroke,
        stroke_width=2,
    )
    label_mob = mono(label, size, color, weight=BOLD)
    if label_mob.width > width - 0.20:
        label_mob.scale_to_fit_width(width - 0.20)
    label_mob.move_to(box)
    return VGroup(box, label_mob)


def card(width: float, height: float) -> RoundedRectangle:
    return RoundedRectangle(
        width=width,
        height=height,
        corner_radius=0.13,
        fill_color="#FFFFFF",
        fill_opacity=1,
        stroke_color=RULE,
        stroke_width=2,
    )


class BPEAfterTraceV2(Scene):
    """A 41-second canonical traversal of the after-trace stepper."""

    def setup(self) -> None:
        self.camera.background_color = PAPER

    def mode_banner(self, label: str, *, encoding: bool = False) -> VGroup:
        banner = RoundedRectangle(
            width=12.95,
            height=0.92,
            corner_radius=0.12,
            fill_color=BLUE_PALE if encoding else CARDINAL_PALE,
            fill_opacity=1,
            stroke_color=BLUE if encoding else CARDINAL,
            stroke_width=2,
        )
        heading = text(
            label,
            58,
            BLUE if encoding else CARDINAL,
            weight=BOLD,
        )
        fit_width(heading, 12.25)
        heading.move_to(banner)
        return VGroup(banner, heading).to_edge(UP, buff=0.47)

    def rule_row(self, rank: int, left: str, right: str) -> VGroup:
        rank_chip = pill(
            str(rank + 1),
            width=0.72,
            fill=PANEL,
            stroke=RULE,
            color=MUTED,
            size=42,
        )
        number = pill(
            str(256 + rank),
            width=1.18,
            fill=CARDINAL_PALE,
            stroke=CARDINAL,
            color=CARDINAL,
            size=50,
        )
        relation = mono(f"{left}+{right}", 46, INK, weight=BOLD)
        arrow = text("→", 48, CARDINAL, weight=BOLD)
        row = VGroup(rank_chip, relation, arrow, number).arrange(RIGHT, buff=0.11)
        fit_width(row, 3.55)
        return row

    def rules_panel(self) -> tuple[VGroup, list[VGroup], VGroup]:
        shell = card(4.35, 4.70)
        label = text("ORDERED MERGE RULES", 46, MUTED, weight=BOLD)
        fit_width(label, 3.55)
        label.next_to(shell.get_top(), DOWN, buff=0.30)
        line = Line(LEFT * 1.75, RIGHT * 1.75, color=RULE, stroke_width=2)
        line.next_to(label, DOWN, buff=0.20)
        rows = [
            self.rule_row(0, "t", "h"),
            self.rule_row(1, "256", "e"),
            self.rule_row(2, "257", "32"),
        ]
        row_group = VGroup(*rows).arrange(DOWN, aligned_edge=LEFT, buff=0.35)
        row_group.next_to(line, DOWN, buff=0.35)
        frozen = pill(
            "FROZEN",
            width=2.35,
            fill=GREEN_PALE,
            stroke=GREEN,
            color=GREEN,
            size=48,
        )
        frozen.next_to(shell.get_bottom(), UP, buff=0.26)
        panel = VGroup(shell, label, line, row_group, frozen)
        return panel, rows, frozen

    def training_panel(self) -> VGroup:
        shell = card(8.10, 4.70)
        label = text("FINAL TRAINING STATE", 48, MUTED, weight=BOLD)
        fit_width(label, 6.75)
        corpus = mono('"the cat in the hat"', 66, INK, weight=BOLD)
        fit_width(corpus, 7.35)
        stats = VGroup(
            pill("18 bytes  →  12 tokens", width=5.50, fill=PANEL, size=48),
            pill("vocab 256  →  259", width=4.75, fill=PANEL, size=48),
        ).arrange(DOWN, buff=0.30)
        changed = text("counts · vocabulary · rules · sequence", 48, CARDINAL)
        fit_width(changed, 7.20)
        content = VGroup(label, corpus, stats, changed).arrange(DOWN, buff=0.34)
        content.move_to(shell)
        return VGroup(shell, content)

    def held_out_panel(self, prefix: list[str], length: int, active: int | None) -> VGroup:
        shell = card(8.10, 4.70)
        label = text("HELD-OUT TEXT", 48, MUTED, weight=BOLD)
        fit_width(label, 6.75)
        sample = mono('"the quick brown fox"', 61, INK, weight=BOLD)
        fit_width(sample, 7.30)
        chips = VGroup(
            *[
                pill(
                    value,
                    width=1.05,
                    fill=CARDINAL_PALE if value in {"256", "257", "258"} else PANEL,
                    stroke=CARDINAL if value in {"256", "257", "258"} else RULE,
                    color=CARDINAL if value in {"256", "257", "258"} else INK,
                    size=50,
                )
                for value in prefix
            ],
            pill("+15 unchanged", width=2.55, fill=BLUE_PALE, stroke=BLUE, color=BLUE, size=34),
        ).arrange(RIGHT, buff=0.12)
        status_text = "no pair counts · rules stay frozen" if active is None else f"rank {active} replayed · rules stay frozen"
        status = text(status_text, 48, GREEN if active is None else BLUE, weight=BOLD)
        fit_width(status, 4.75)
        count = pill(f"{length} tokens", width=2.55, fill=GREEN_PALE, stroke=GREEN, color=GREEN, size=48)
        bottom = VGroup(status, count).arrange(RIGHT, buff=0.35)
        content = VGroup(label, sample, chips, bottom).arrange(DOWN, buff=0.37)
        content.move_to(shell)
        return VGroup(shell, content)

    def decode_panel(self) -> VGroup:
        shell = card(8.10, 4.70)
        label = text("DECODE WITH FROZEN VOCAB", 48, MUTED, weight=BOLD)
        fit_width(label, 6.75)
        tokens = VGroup(
            pill("258", width=1.25, fill=CARDINAL_PALE, stroke=CARDINAL, color=CARDINAL, size=50),
            pill("+15 unchanged", width=2.55, fill=BLUE_PALE, stroke=BLUE, color=BLUE, size=34),
        ).arrange(RIGHT, buff=0.16)
        arrow = text("↓", 64, CARDINAL, weight=BOLD)
        result = pill(
            '"the quick brown fox"  ✓',
            width=6.55,
            fill=GREEN_PALE,
            stroke=GREEN,
            color=GREEN,
            size=52,
        )
        content = VGroup(label, tokens, arrow, result).arrange(DOWN, buff=0.30)
        content.move_to(shell)
        return VGroup(shell, content)

    def final_ledger(self) -> VGroup:
        left_shell = card(6.15, 5.55)
        right_shell = card(6.15, 5.55)
        left_title = text("TRAINING", 62, CARDINAL, weight=BOLD)
        right_title = text("ENCODING", 62, BLUE, weight=BOLD)
        left_rows = VGroup(
            text("pair counts change", 52, INK),
            text("vocabulary grows", 52, INK),
            text("merge table grows", 52, INK),
            text("corpus sequence changes", 52, INK),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.30)
        right_rows = VGroup(
            text("no new pair counts", 52, INK),
            text("vocabulary is frozen", 52, GREEN),
            text("merge order is frozen", 52, GREEN),
            text("held-out sequence changes", 52, BLUE),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.30)
        for row in [*left_rows, *right_rows]:
            fit_width(row, 5.05)
        left_content = VGroup(left_title, left_rows).arrange(DOWN, buff=0.48)
        right_content = VGroup(right_title, right_rows).arrange(DOWN, buff=0.48)
        left_content.move_to(left_shell)
        right_content.move_to(right_shell)
        columns = VGroup(
            VGroup(left_shell, left_content),
            VGroup(right_shell, right_content),
        ).arrange(RIGHT, buff=0.40)
        banner = pill(
            "same learned tokenizer · different mutable state",
            width=8.15,
            fill=GREEN_PALE,
            stroke=GREEN,
            color=GREEN,
            size=46,
        )
        banner.next_to(columns, UP, buff=0.22)
        return VGroup(columns, banner).shift(DOWN * 0.32)

    def construct(self) -> None:
        banner = self.mode_banner("TRAINING LEARNS AN ORDERED MERGE TABLE")
        training = self.training_panel().shift(LEFT * 2.30 + DOWN * 0.20)
        rules, rule_rows, frozen = self.rules_panel()
        rules.shift(RIGHT * 4.15 + DOWN * 0.20)
        frozen.set_opacity(0)

        self.play(FadeIn(banner), FadeIn(training), FadeIn(rules), run_time=0.8)
        self.wait(3.0)

        frozen.set_opacity(1)
        self.play(FadeIn(frozen, shift=UP * 0.10), run_time=0.8)
        self.wait(4.3)

        encoding_banner = self.mode_banner("ENCODING REPLAYS THE FROZEN RULES", encoding=True)
        held = self.held_out_panel(["116", "104", "101", "32"], 19, None)
        held.shift(LEFT * 2.30 + DOWN * 0.20)
        self.play(
            ReplacementTransform(banner, encoding_banner),
            ReplacementTransform(training, held),
            run_time=0.8,
        )
        banner = encoding_banner
        self.wait(4.0)

        states = [
            (["256", "101", "32"], 18),
            (["257", "32"], 17),
            (["258"], 16),
        ]
        for rank, (prefix, length) in enumerate(states):
            next_panel = self.held_out_panel(prefix, length, rank)
            next_panel.shift(LEFT * 2.30 + DOWN * 0.20)
            self.play(
                Indicate(rule_rows[rank], color=BLUE, scale_factor=1.04),
                ReplacementTransform(held, next_panel),
                run_time=0.9,
            )
            held = next_panel
            self.wait(4.0)

        decoded = self.decode_panel().shift(LEFT * 2.30 + DOWN * 0.20)
        self.play(ReplacementTransform(held, decoded), run_time=1.0)
        self.wait(4.0)

        ledger = self.final_ledger()
        self.play(
            FadeOut(decoded),
            FadeOut(rules),
            FadeOut(banner),
            FadeIn(ledger),
            run_time=1.0,
        )
        self.wait(6.0)
