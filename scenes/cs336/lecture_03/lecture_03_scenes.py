"""CS336 Lecture 3 augmented micro-lectures.

Render all three scenes at the site-preview quality with::

    python -m manim -ql --media_dir build/manim/lecture_03 \
      scenes/cs336/lecture_03/lecture_03_scenes.py \
      L03PrePostNorm L03KVSharing L03SlidingWindow

The visual language intentionally follows the spare CS336 course-site style:
light paper, Stanford cardinal, compact diagrams, and source-slide references.
"""

from __future__ import annotations

from manim import (
    AnimationGroup,
    Arrow,
    BLACK,
    Create,
    DOWN,
    FadeIn,
    FadeOut,
    GrowArrow,
    LEFT,
    Line,
    MovingCameraScene,
    ORIGIN,
    Rectangle,
    ReplacementTransform,
    RIGHT,
    RoundedRectangle,
    Scene,
    Square,
    Text,
    Transform,
    UP,
    VGroup,
    WHITE,
    Write,
    config,
)


PAPER = "#FAFAF7"
INK = "#252525"
MUTED = "#626262"
RULE = "#D8D5CF"
CARDINAL = "#8C1515"
CARDINAL_DARK = "#651111"
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


def label(text: str, size: int = 24, color: str = INK, weight: str = "NORMAL", font: str = FONT) -> Text:
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


def operation(name: str, color: str = CARDINAL, width: float = 0.88) -> VGroup:
    body = box(width, 0.54, fill=WHITE, stroke=color, radius=0.06, stroke_width=1.8)
    text = label(name, 17, color, "BOLD", MONO).move_to(body)
    return VGroup(body, text)


class CS336Scene(Scene):
    """Shared title bar and closing convention for short lecture inserts."""

    segment_id = "L03"
    title = "Lecture 3 augmentation"
    source_ref = "CS336 · Lecture 3"

    def setup(self) -> None:
        self.camera.background_color = PAPER

    def header(self, kicker: str | None = None) -> VGroup:
        segment = label(
            f"CS336 · LECTURE 3 · {kicker or self.segment_id}",
            15,
            CARDINAL,
            "BOLD",
            MONO,
        )
        heading = label(self.title, 34, INK, "BOLD")
        title_group = VGroup(segment, heading).arrange(DOWN, aligned_edge=LEFT, buff=0.1)
        title_group.to_edge(UP, buff=0.28).to_edge(LEFT, buff=0.48)
        rule = Line(LEFT * 6.18, RIGHT * 6.18, color=RULE, stroke_width=1).next_to(title_group, DOWN, buff=0.16)
        return VGroup(title_group, rule)

    def source(self) -> Text:
        return label(self.source_ref, 15, MUTED, font=MONO).to_edge(DOWN, buff=0.18).to_edge(RIGHT, buff=0.42)

    def show_chrome(self, kicker: str | None = None) -> VGroup:
        chrome = VGroup(self.header(kicker), self.source())
        self.play(FadeIn(chrome, shift=DOWN * 0.08), run_time=0.65)
        return chrome


class L03PrePostNorm(CS336Scene):
    """Contrast where normalization sits relative to the identity path."""

    segment_id = "L03-PRE-POST-NORM"
    title = "Pre-norm keeps an identity path"
    source_ref = "lecture_03.pdf · slides 10–12"

    def topology(self, *, pre_norm: bool) -> VGroup:
        panel = box(5.72, 3.65, fill=WHITE)
        heading_text = "PRE-NORM" if pre_norm else "POST-NORM"
        heading_color = GREEN if pre_norm else CARDINAL
        heading = label(heading_text, 19, heading_color, "BOLD", MONO)
        heading.move_to(panel.get_top() + DOWN * 0.35 + LEFT * 1.92)

        x_in = panel.get_left()[0] + 0.47
        x_out = panel.get_right()[0] - 0.47
        y_main = panel.get_center()[1] - 0.42
        input_dot = Square(0.16, fill_color=INK, fill_opacity=1, stroke_width=0).move_to([x_in, y_main, 0])
        output_dot = Square(0.16, fill_color=INK, fill_opacity=1, stroke_width=0).move_to([x_out, y_main, 0])
        x_label = label("x", 18, INK, font=MONO).next_to(input_dot, DOWN, buff=0.13)
        out_label = label("x'", 18, INK, font=MONO).next_to(output_dot, DOWN, buff=0.13)

        if pre_norm:
            main = Arrow(input_dot.get_right(), output_dot.get_left(), buff=0.02, color=GREEN, stroke_width=5, max_tip_length_to_length_ratio=0.06)
            ln = operation("LN", BLUE)
            fn = operation("F", CARDINAL)
            branch_y = y_main + 0.72
            ln.move_to([x_in + 1.36, branch_y, 0])
            fn.move_to([x_in + 2.75, branch_y, 0])
            branch = VGroup(
                Arrow(input_dot.get_center() + UP * 0.03, ln.get_left(), buff=0.03, color=MUTED, stroke_width=2.3, max_tip_length_to_length_ratio=0.08),
                Arrow(ln.get_right(), fn.get_left(), buff=0.05, color=MUTED, stroke_width=2.3, max_tip_length_to_length_ratio=0.08),
                Arrow(fn.get_right(), output_dot.get_center() + UP * 0.03, buff=0.03, color=MUTED, stroke_width=2.3, max_tip_length_to_length_ratio=0.08),
            )
            status = VGroup(
                label("gradient highway", 17, GREEN, "BOLD"),
                label("identity path stays untouched", 15, MUTED),
            ).arrange(DOWN, aligned_edge=LEFT, buff=0.07)
            status.move_to(panel.get_bottom() + UP * 0.38)
            formula = label("x' = x + F(LN(x))", 18, INK, font=MONO).move_to(panel.get_top() + DOWN * 0.86)
            diagram = VGroup(main, branch, ln, fn)
        else:
            merge = operation("+", MUTED, 0.58)
            ln = operation("LN", CARDINAL)
            merge.move_to([x_out - 1.75, y_main, 0])
            ln.move_to([x_out - 0.73, y_main, 0])
            main = VGroup(
                Arrow(input_dot.get_right(), merge.get_left(), buff=0.02, color=MUTED, stroke_width=3, max_tip_length_to_length_ratio=0.07),
                Arrow(merge.get_right(), ln.get_left(), buff=0.04, color=CARDINAL, stroke_width=4.5, max_tip_length_to_length_ratio=0.08),
                Arrow(ln.get_right(), output_dot.get_left(), buff=0.03, color=CARDINAL, stroke_width=4.5, max_tip_length_to_length_ratio=0.08),
            )
            fn = operation("F", CARDINAL)
            branch_y = y_main + 0.72
            fn.move_to([x_in + 1.45, branch_y, 0])
            branch = VGroup(
                Arrow(input_dot.get_center() + UP * 0.03, fn.get_left(), buff=0.03, color=MUTED, stroke_width=2.3, max_tip_length_to_length_ratio=0.08),
                Arrow(fn.get_right(), merge.get_top(), buff=0.03, color=MUTED, stroke_width=2.3, max_tip_length_to_length_ratio=0.08),
            )
            status = VGroup(
                label("normalization bottleneck", 17, CARDINAL, "BOLD"),
                label("every path crosses LN", 15, MUTED),
            ).arrange(DOWN, aligned_edge=LEFT, buff=0.07)
            status.move_to(panel.get_bottom() + UP * 0.38)
            formula = label("x' = LN(x + F(x))", 18, INK, font=MONO).move_to(panel.get_top() + DOWN * 0.86)
            diagram = VGroup(main, branch, merge, ln, fn)

        return VGroup(panel, heading, formula, input_dot, output_dot, x_label, out_label, diagram, status)

    def construct(self) -> None:
        self.show_chrome()
        post = self.topology(pre_norm=False).scale(0.93).shift(LEFT * 3.05 + DOWN * 0.35)
        pre = self.topology(pre_norm=True).scale(0.93).shift(RIGHT * 3.05 + DOWN * 0.35)
        self.play(FadeIn(post[0]), FadeIn(pre[0]), Write(post[1]), Write(pre[1]), run_time=0.65)
        self.play(FadeIn(post[2]), FadeIn(pre[2]), run_time=0.45)
        self.play(
            AnimationGroup(*[FadeIn(mob) for mob in post[3:7]], lag_ratio=0.07),
            AnimationGroup(*[FadeIn(mob) for mob in pre[3:7]], lag_ratio=0.07),
            run_time=1.3,
        )
        self.play(Create(post[7]), Create(pre[7]), run_time=1.8)
        self.wait(0.4)
        self.play(FadeIn(post[8], shift=UP * 0.08), FadeIn(pre[8], shift=UP * 0.08), run_time=0.7)

        gradient = Arrow(RIGHT * 5.25 + DOWN * 2.82, RIGHT * 0.72 + DOWN * 2.82, buff=0, color=GREEN, stroke_width=4, max_tip_length_to_length_ratio=0.06)
        gradient_label = label("backward: direct gradient route", 17, GREEN, "BOLD").next_to(gradient, UP, buff=0.1)
        self.play(GrowArrow(gradient), FadeIn(gradient_label), run_time=1.0)
        takeaway = label(
            "Take-away: pre-norm leaves the residual identity route outside LayerNorm.",
            18,
            INK,
            "BOLD",
        ).to_edge(DOWN, buff=0.72)
        self.play(FadeIn(takeaway, shift=UP * 0.06), run_time=0.6)
        self.wait(1.1)


class L03KVSharing(CS336Scene):
    """Morph MHA into GQA and MQA while keeping eight query heads."""

    segment_id = "L03-GQA · L03-MQA"
    title = "KV sharing is a cache-size knob"
    source_ref = "lecture_03.pdf · slides 58–63"

    @staticmethod
    def stage(name: str, n_kv: int, note: str) -> VGroup:
        panel = box(10.9, 4.02, fill=WHITE)
        stage_name = label(name, 22, CARDINAL, "BOLD", MONO).move_to(panel.get_top() + DOWN * 0.36 + LEFT * 4.55)
        query_title = label("8 query heads", 15, MUTED, "BOLD").move_to(panel.get_top() + DOWN * 0.83 + LEFT * 3.78)
        kv_title = label(f"{n_kv} key/value head{'s' if n_kv > 1 else ''}", 15, MUTED, "BOLD").move_to(panel.get_top() + DOWN * 0.83 + RIGHT * 1.95)

        q_heads = VGroup()
        for idx in range(8):
            q_box = box(0.63, 0.29, fill=BLUE_SOFT, stroke=BLUE, radius=0.04)
            q_text = label(f"Q{idx + 1}", 11, BLUE, "BOLD", MONO).move_to(q_box)
            q_heads.add(VGroup(q_box, q_text))
        q_heads.arrange(DOWN, buff=0.04).move_to(LEFT * 3.78 + DOWN * 0.38)

        kv_heads = VGroup()
        for idx in range(n_kv):
            kv_box = box(1.2, 0.29, fill=CARDINAL_SOFT, stroke=CARDINAL, radius=0.04)
            kv_label = f"K{idx + 1} · V{idx + 1}" if n_kv > 1 else "shared K · V"
            kv_text = label(kv_label, 11, CARDINAL_DARK, "BOLD", MONO).move_to(kv_box)
            kv_heads.add(VGroup(kv_box, kv_text))
        kv_heads.arrange(DOWN, buff=0.04).move_to(RIGHT * 1.95 + DOWN * 0.38)

        links = VGroup()
        group_size = 8 // n_kv
        for idx, q_head in enumerate(q_heads):
            kv_idx = min(n_kv - 1, idx // group_size)
            links.add(Line(q_head.get_right(), kv_heads[kv_idx].get_left(), color=RULE, stroke_width=1.7))

        cache_track = box(2.0, 0.34, fill=PANEL, stroke=RULE, radius=0.04)
        cache_fill = RoundedRectangle(
            width=max(0.22, 2.0 * n_kv / 8),
            height=0.34,
            corner_radius=0.04,
            fill_color=GREEN,
            fill_opacity=1,
            stroke_width=0,
        ).align_to(cache_track, LEFT)
        cache = VGroup(cache_track, cache_fill).move_to(RIGHT * 4.23 + DOWN * 1.08)
        cache_label = label(f"relative KV cache  {n_kv}/8", 15, GREEN, "BOLD", MONO).next_to(cache, UP, buff=0.11)
        note_text = label(note, 16, INK).move_to(panel.get_bottom() + UP * 0.23)

        return VGroup(panel, stage_name, query_title, kv_title, links, q_heads, kv_heads, cache, cache_label, note_text)

    def construct(self) -> None:
        self.show_chrome("L03-GQA · L03-MQA")
        formula = label("KV bytes / token / layer = 2 × n_kv_heads × d_head × bytes_per_value", 16, INK, font=MONO)
        formula.to_edge(DOWN, buff=0.52)

        mha = self.stage("KV CACHE", 8, "Incremental decoding repeatedly reads stored keys and values.").shift(DOWN * 0.22)
        mqa = self.stage("MQA", 1, "One shared K/V stream minimizes cache traffic.").shift(DOWN * 0.22)
        gqa = self.stage("GQA", 2, "GQA spends some cache back for more K/V capacity.").shift(DOWN * 0.22)

        self.play(FadeIn(mha[0]), FadeIn(mha[1:4]), run_time=0.55)
        self.play(FadeIn(mha[4]), FadeIn(mha[5]), FadeIn(mha[6]), run_time=1.1)
        self.play(FadeIn(mha[7:10]), FadeIn(formula), run_time=0.75)
        self.wait(0.8)

        self.play(ReplacementTransform(mha, mqa), run_time=1.25)
        self.wait(1.0)
        self.play(ReplacementTransform(mqa, gqa), run_time=1.25)
        self.wait(1.0)

        takeaway = label(
            "Take-away: MQA cuts KV traffic to 1/8; GQA spends some back (2/8 here) for capacity.",
            18,
            INK,
            "BOLD",
        ).to_edge(DOWN, buff=0.72)
        self.play(FadeOut(formula), FadeIn(takeaway, shift=UP * 0.06), run_time=0.8)
        self.wait(1.2)


class L03SlidingWindow(CS336Scene):
    """Reduce causal attention to a moving local band and count the cells."""

    segment_id = "L03-SLIDING-WINDOW"
    title = "Sliding-window attention keeps a causal band"
    source_ref = "lecture_03.pdf · slides 64–66"

    N = 12
    WINDOW = 4
    CELL = 0.31

    def grid(self) -> tuple[VGroup, list[list[Square]]]:
        rows: list[list[Square]] = []
        group = VGroup()
        for row in range(self.N):
            row_cells: list[Square] = []
            for col in range(self.N):
                cell = Square(
                    self.CELL,
                    fill_color=WHITE,
                    fill_opacity=1,
                    stroke_color=RULE,
                    stroke_width=0.75,
                )
                cell.move_to([col * self.CELL, -row * self.CELL, 0])
                row_cells.append(cell)
                group.add(cell)
            rows.append(row_cells)
        group.center().shift(LEFT * 2.1 + DOWN * 0.3)
        return group, rows

    @staticmethod
    def fill_cells(rows: list[list[Square]], mode: str) -> None:
        for row, cells in enumerate(rows):
            for col, cell in enumerate(cells):
                if mode == "full" and col <= row:
                    cell.set_fill(BLUE_SOFT, opacity=1).set_stroke(BLUE, width=0.8)
                elif mode == "window" and col <= row and col >= row - L03SlidingWindow.WINDOW + 1:
                    cell.set_fill(CARDINAL_SOFT, opacity=1).set_stroke(CARDINAL, width=0.95)
                else:
                    cell.set_fill(WHITE, opacity=1).set_stroke(RULE, width=0.7)

    def counter(self, heading: str, value: str, detail: str, color: str) -> VGroup:
        card = box(3.35, 1.22, fill=WHITE)
        title = label(heading, 15, MUTED, "BOLD", MONO)
        number = label(value, 28, color, "BOLD", MONO)
        foot = label(detail, 14, INK)
        copy = VGroup(title, number, foot).arrange(DOWN, buff=0.08).move_to(card)
        return VGroup(card, copy)

    def construct(self) -> None:
        self.show_chrome()
        grid, rows = self.grid()
        q_label = label("query token", 15, MUTED, "BOLD").rotate(1.5708).next_to(grid, LEFT, buff=0.28)
        k_label = label("key token", 15, MUTED, "BOLD").next_to(grid, UP, buff=0.2)
        self.play(FadeIn(grid), FadeIn(q_label), FadeIn(k_label), run_time=0.75)

        full_state = grid.copy()
        full_rows = [list(full_state[i * self.N : (i + 1) * self.N]) for i in range(self.N)]
        self.fill_cells(full_rows, "full")
        full_count = self.counter("FULL CAUSAL", "78 cells", "n(n + 1) / 2", BLUE).shift(RIGHT * 3.75 + UP * 0.72)
        self.play(Transform(grid, full_state), FadeIn(full_count), run_time=1.15)
        self.wait(0.7)

        window_state = grid.copy()
        window_rows = [list(window_state[i * self.N : (i + 1) * self.N]) for i in range(self.N)]
        self.fill_cells(window_rows, "window")
        window_count = self.counter("WINDOW = 4", "42 cells", "46% fewer cells", CARDINAL).shift(RIGHT * 3.75 + DOWN * 0.86)
        self.play(Transform(grid, window_state), FadeIn(window_count), run_time=1.25)
        self.wait(0.65)

        row_marker = Rectangle(
            width=self.CELL * self.N + 0.1,
            height=self.CELL + 0.09,
            stroke_color=CARDINAL,
            stroke_width=2.5,
            fill_opacity=0,
        ).move_to(grid[7 * self.N : 8 * self.N])
        row_note = label("one query reads at most 4 recent keys", 16, CARDINAL, "BOLD").next_to(grid, DOWN, buff=0.25)
        self.play(Create(row_marker), FadeIn(row_note), run_time=0.65)
        self.play(row_marker.animate.move_to(grid[10 * self.N : 11 * self.N]), run_time=1.15)
        self.play(row_marker.animate.move_to(grid[11 * self.N : 12 * self.N]), run_time=0.6)
        takeaway = VGroup(
            label("Most layers: O(nw) local attention.", 18, INK, "BOLD"),
            label("Periodic full-attention layers restore long-range mixing.", 16, GREEN, "BOLD"),
        ).arrange(DOWN, buff=0.06).to_edge(DOWN, buff=0.62)
        self.play(FadeIn(takeaway, shift=UP * 0.06), run_time=0.6)
        self.wait(1.0)
