"""CS336 Lecture 1 tokenization animations.

Render the complete first batch at low quality:

    python -m manim -ql \
      --media_dir build/manim/lecture_01 \
      scenes/cs336/lecture_01/lecture_01_tokenization.py \
      CS336UTF8Bytes CS336BPEPairCountAndMerge \
      CS336BPETrainVsUse CS336TokenizerTradeoffToy

The scenes intentionally use a restrained, source-adjacent CS336 visual
language: light paper background, Stanford cardinal accents, Roboto text, and
small monospace labels. They do not depend on LaTeX.
"""

from __future__ import annotations

from manim import (
    Arrow,
    BOLD,
    Circle,
    Circumscribe,
    Create,
    DOWN,
    Ellipse,
    FadeIn,
    FadeOut,
    GrowArrow,
    Indicate,
    LEFT,
    Line,
    Rectangle,
    RIGHT,
    RoundedRectangle,
    Scene,
    Text,
    UP,
    VGroup,
    Write,
)


PAPER = "#FAFAF7"
INK = "#252525"
MUTED = "#626262"
RULE = "#D8D5CF"
PANEL = "#F1F0EC"
CARDINAL = "#8C1515"
CARDINAL_DARK = "#65120F"
ACCENT = "#2E657D"
SUCCESS = "#3F6B4F"
WARNING = "#9A5A16"
MONO = "Menlo"
# Roboto is the website font; Helvetica Neue is the bundled macOS metric/style
# fallback used for deterministic local renders.
SANS = "Helvetica Neue"


class CS336LectureScene(Scene):
    """Shared, deliberately quiet CS336 lecture chrome."""

    def setup(self) -> None:
        self.camera.background_color = PAPER

    def header(self, segment_id: str, title: str) -> VGroup:
        kicker = Text(
            f"CS336 · LECTURE 1 · {segment_id}",
            font=MONO,
            font_size=15,
            color=CARDINAL,
            weight=BOLD,
        )
        heading = Text(title, font=SANS, font_size=34, color=INK, weight=BOLD)
        heading_group = VGroup(kicker, heading).arrange(
            DOWN, aligned_edge=LEFT, buff=0.12
        )
        heading_group.to_edge(UP, buff=0.3).to_edge(LEFT, buff=0.5)
        rule = Line(
            LEFT * 6.2,
            RIGHT * 6.2,
            stroke_color=RULE,
            stroke_width=1.5,
        ).next_to(heading_group, DOWN, buff=0.18)
        return VGroup(heading_group, rule)

    def source_note(self, text: str) -> Text:
        return (
            Text(text, font=MONO, font_size=15, color=MUTED)
            .to_edge(DOWN, buff=0.2)
            .to_edge(RIGHT, buff=0.38)
        )

    def caption(self, text: str, *, color: str = MUTED) -> Text:
        caption = Text(text, font=SANS, font_size=20, color=color)
        if caption.width > 11.8:
            caption.scale_to_fit_width(11.8)
        return caption


def _chip(
    label: str,
    *,
    width: float = 0.62,
    height: float = 0.58,
    fill: str = PANEL,
    stroke: str = RULE,
    text_color: str = INK,
    font_size: int = 21,
) -> VGroup:
    box = RoundedRectangle(
        width=width,
        height=height,
        corner_radius=0.07,
        fill_color=fill,
        fill_opacity=1,
        stroke_color=stroke,
        stroke_width=1.25,
    )
    label_mob = Text(label, font=MONO, font_size=font_size, color=text_color)
    if label_mob.width > width - 0.12:
        label_mob.scale_to_fit_width(width - 0.12)
    label_mob.move_to(box)
    return VGroup(box, label_mob)


def _globe(*, radius: float = 0.24, color: str = CARDINAL_DARK) -> VGroup:
    """A Cairo-safe vector stand-in for U+1F30D (color emoji is raster-only)."""

    outline = Circle(radius=radius, stroke_color=color, stroke_width=2)
    meridian = Ellipse(
        width=radius * 0.72,
        height=radius * 1.9,
        stroke_color=color,
        stroke_width=1.4,
    )
    equator = Line(
        LEFT * radius * 0.9,
        RIGHT * radius * 0.9,
        stroke_color=color,
        stroke_width=1.4,
    )
    return VGroup(outline, meridian, equator)


def _token_row(
    labels: list[str],
    *,
    merged_labels: set[str] | None = None,
    width: float = 0.58,
    buff: float = 0.055,
) -> VGroup:
    merged_labels = merged_labels or set()
    chips = VGroup()
    for label in labels:
        merged = label in merged_labels
        chips.add(
            _chip(
                label,
                width=max(width, 0.31 + 0.22 * len(label)),
                fill="#F6E8E7" if merged else PANEL,
                stroke=CARDINAL if merged else RULE,
                text_color=CARDINAL_DARK if merged else INK,
                font_size=18,
            )
        )
    chips.arrange(RIGHT, buff=buff)
    return chips


class CS336UTF8Bytes(CS336LectureScene):
    """L01-UTF8-BYTES: exact code point-to-byte expansion and round trip."""

    def construct(self) -> None:
        header = self.header("L01-UTF8-BYTES", "UTF-8 uses a variable number of bytes")
        source = self.source_note("lecture_01.py · lines 651–673")
        self.play(
            FadeIn(header[0][0]),
            FadeIn(header[0][1]),
            Create(header[1]),
            FadeIn(source),
            run_time=0.12,
        )

        sample = VGroup(
            Text("A", font=SANS, font_size=52, color=INK, weight=BOLD),
            Text("中", font=SANS, font_size=52, color=INK, weight=BOLD),
            _globe(radius=0.34, color=INK),
        ).arrange(RIGHT, buff=0.56)
        sample.next_to(header, DOWN, buff=0.46)
        prompt = self.caption("same interface: character → code point → UTF-8 bytes")
        prompt.next_to(sample, DOWN, buff=0.18)
        self.play(Write(sample), FadeIn(prompt), run_time=0.9)

        entries = [
            ("A", "U+0041", ["41"]),
            ("中", "U+4E2D", ["E4", "B8", "AD"]),
            ("globe", "U+1F30D", ["F0", "9F", "8C", "8D"]),
        ]
        rows = VGroup()
        byte_groups: list[VGroup] = []
        for character, codepoint, byte_values in entries:
            character_chip = _chip(
                "" if character == "globe" else character,
                width=0.92,
                height=0.68,
                fill="#F6E8E7",
                stroke=CARDINAL,
                text_color=CARDINAL_DARK,
                font_size=28,
            )
            if character == "globe":
                character_chip.add(_globe(radius=0.2).move_to(character_chip[0]))
            codepoint_text = Text(codepoint, font=MONO, font_size=19, color=MUTED)
            arrow = Arrow(
                LEFT * 0.25,
                RIGHT * 0.25,
                buff=0,
                color=RULE,
                stroke_width=2,
                max_tip_length_to_length_ratio=0.22,
            )
            byte_group = VGroup(
                *[
                    _chip(
                        value,
                        width=0.66,
                        height=0.58,
                        fill="#E9F0F3",
                        stroke=ACCENT,
                        text_color=ACCENT,
                        font_size=17,
                    )
                    for value in byte_values
                ]
            ).arrange(RIGHT, buff=0.08)
            byte_groups.append(byte_group)
            row = VGroup(character_chip, codepoint_text, arrow, byte_group).arrange(
                RIGHT, buff=0.36
            )
            rows.add(row)
        rows.arrange(DOWN, aligned_edge=LEFT, buff=0.24)
        rows.next_to(prompt, DOWN, buff=0.33)

        for index, row in enumerate(rows):
            self.play(FadeIn(row, shift=RIGHT * 0.12), run_time=0.45)
            self.play(Circumscribe(byte_groups[index], color=ACCENT, fade_out=True), run_time=0.55)

        byte_counts = VGroup(
            *[
                Text(label, font=MONO, font_size=14, color=color, weight=BOLD)
                for label, color in [
                    ("1 byte", SUCCESS),
                    ("3 bytes", WARNING),
                    ("4 bytes", CARDINAL),
                ]
            ]
        )
        for count, row in zip(byte_counts, rows):
            count.next_to(row, RIGHT, buff=0.28)
        self.play(FadeIn(byte_counts), run_time=0.6)

        flat_values = ["41", "E4", "B8", "AD", "F0", "9F", "8C", "8D"]
        flat_stream = VGroup(
            *[
                _chip(
                    value,
                    width=0.62,
                    height=0.52,
                    fill="#E9F0F3",
                    stroke=ACCENT,
                    text_color=ACCENT,
                    font_size=16,
                )
                for value in flat_values
            ]
        ).arrange(RIGHT, buff=0.06)
        flat_stream.to_edge(LEFT, buff=0.7).shift(DOWN * 2.62)
        decode_arrow = Arrow(
            flat_stream.get_right() + RIGHT * 0.18,
            flat_stream.get_right() + RIGHT * 1.08,
            buff=0,
            color=CARDINAL,
            stroke_width=3,
        )
        decoded = VGroup(
            Text("A中", font=SANS, font_size=29, color=CARDINAL_DARK, weight=BOLD),
            _globe(radius=0.2),
        ).arrange(RIGHT, buff=0.12)
        decoded.next_to(decode_arrow, RIGHT, buff=0.18)

        self.play(
            FadeOut(sample),
            FadeOut(prompt),
            rows.animate.shift(UP * 0.48),
            byte_counts.animate.shift(UP * 0.48),
            run_time=0.7,
        )
        self.play(FadeIn(flat_stream, shift=UP * 0.12), GrowArrow(decode_arrow), run_time=0.7)
        self.play(Write(decoded), run_time=0.55)
        takeaway = self.caption(
            "Vocabulary stays fixed at 256 byte values; sequence length depends on the text.",
            color=INK,
        ).to_edge(DOWN, buff=0.48)
        self.play(FadeIn(takeaway), run_time=0.6)
        self.wait(1.1)


class CS336BPEPairCountAndMerge(CS336LectureScene):
    """L01-BPE-PAIR-COUNT + L01-BPE-MERGE as one continuous lesson."""

    corpus = "the cat in the hat"

    def construct(self) -> None:
        header = self.header("L01-BPE-PAIR-COUNT", "BPE learns by counting adjacent pairs")
        source = self.source_note("lecture_01.py · lines 705–758")
        self.play(
            FadeIn(header[0][0]),
            FadeIn(header[0][1]),
            Create(header[1]),
            FadeIn(source),
            run_time=0.12,
        )

        corpus_label = Text(
            f'corpus:  "{self.corpus}"', font=MONO, font_size=24, color=INK
        )
        corpus_label.next_to(header, DOWN, buff=0.42).to_edge(LEFT, buff=0.72)
        labels = ["␠" if char == " " else char for char in self.corpus]
        tokens = _token_row(labels, width=0.54, buff=0.04)
        tokens.next_to(corpus_label, DOWN, buff=0.34).to_edge(LEFT, buff=0.72)
        self.play(Write(corpus_label), FadeIn(tokens), run_time=0.8)

        window = RoundedRectangle(
            width=tokens[0].width + tokens[1].width + 0.13,
            height=0.72,
            corner_radius=0.09,
            stroke_color=CARDINAL,
            stroke_width=3,
            fill_opacity=0,
        ).move_to(VGroup(tokens[0], tokens[1]))
        scan_label = Text("scan overlapping windows", font=SANS, font_size=17, color=CARDINAL)
        scan_label.next_to(window, DOWN, buff=0.12)
        self.play(Create(window), FadeIn(scan_label), run_time=0.45)
        for index in [1, 2, 3, 4, 5, 10, 11, 12, 14, 15, 16]:
            self.play(
                window.animate.move_to(VGroup(tokens[index], tokens[index + 1])),
                scan_label.animate.next_to(
                    VGroup(tokens[index], tokens[index + 1]), DOWN, buff=0.12
                ),
                run_time=0.18,
            )

        count_rows = [
            ("(t, h)", "2"),
            ("(h, e)", "2"),
            ("(e, ␠)", "2"),
            ("(a, t)", "2"),
            ("all other pairs", "1"),
        ]
        count_panel = RoundedRectangle(
            width=4.1,
            height=2.42,
            corner_radius=0.1,
            fill_color=PANEL,
            fill_opacity=1,
            stroke_color=RULE,
            stroke_width=1,
        )
        count_panel.to_edge(RIGHT, buff=0.72).shift(DOWN * 1.03)
        count_title = Text("pair counts", font=MONO, font_size=18, color=MUTED, weight=BOLD)
        count_title.move_to(count_panel.get_top() + DOWN * 0.32).align_to(count_panel, LEFT).shift(RIGHT * 0.28)
        count_mobs = VGroup()
        for pair, value in count_rows:
            pair_text = Text(pair, font=MONO, font_size=16, color=INK)
            value_text = Text(value, font=MONO, font_size=17, color=CARDINAL, weight=BOLD)
            row = VGroup(pair_text, value_text).arrange(RIGHT, buff=0.55)
            if row.width > 3.48:
                row.scale_to_fit_width(3.48)
            count_mobs.add(row)
        count_mobs.arrange(DOWN, aligned_edge=LEFT, buff=0.14)
        count_mobs.next_to(count_title, DOWN, buff=0.22).align_to(count_title, LEFT)
        self.play(FadeOut(window), FadeOut(scan_label), FadeIn(count_panel), FadeIn(count_title), run_time=0.5)
        self.play(FadeIn(count_mobs, lag_ratio=0.1), run_time=0.8)

        tie_note = self.caption(
            "Four pairs tie at 2; this toy run chooses the first encountered pair.",
            color=CARDINAL_DARK,
        )
        tie_note.to_edge(DOWN, buff=0.58)
        self.play(Indicate(count_mobs[0], color=CARDINAL), FadeIn(tie_note), run_time=0.8)
        self.wait(0.6)

        new_header = self.header("L01-BPE-MERGE", "Each merge creates one new token")
        round0 = _token_row(labels, width=0.46, buff=0.025).scale(0.88)
        round1_labels = ["th", "e", "␠", "c", "a", "t", "␠", "i", "n", "␠", "th", "e", "␠", "h", "a", "t"]
        round2_labels = ["the", "␠", "c", "a", "t", "␠", "i", "n", "␠", "the", "␠", "h", "a", "t"]
        round3_labels = ["the␠", "c", "a", "t", "␠", "i", "n", "␠", "the␠", "h", "a", "t"]
        rounds = [
            round0,
            _token_row(round1_labels, merged_labels={"th"}, width=0.46, buff=0.035).scale(0.88),
            _token_row(round2_labels, merged_labels={"the"}, width=0.46, buff=0.035).scale(0.88),
            _token_row(round3_labels, merged_labels={"the␠"}, width=0.46, buff=0.035).scale(0.88),
        ]
        labels_left = VGroup(
            *[
                Text(text, font=MONO, font_size=15, color=color, weight=BOLD)
                for text, color in [
                    ("start · 18 tokens", MUTED),
                    ("256 = th · 16", CARDINAL),
                    ("257 = the · 14", CARDINAL),
                    ("258 = the␠ · 12", CARDINAL),
                ]
            ]
        )
        body_rows = VGroup()
        for label_mob, token_mob in zip(labels_left, rounds):
            row = VGroup(label_mob, token_mob).arrange(RIGHT, buff=0.34)
            body_rows.add(row)
        body_rows.arrange(DOWN, aligned_edge=LEFT, buff=0.24)
        body_rows.next_to(new_header, DOWN, buff=0.38).to_edge(LEFT, buff=0.55)

        self.play(
            FadeOut(header),
            FadeIn(new_header),
            FadeOut(corpus_label),
            FadeOut(tokens),
            FadeOut(count_panel),
            FadeOut(count_title),
            FadeOut(count_mobs),
            FadeOut(tie_note),
            run_time=0.75,
        )
        self.play(FadeIn(body_rows[0]), run_time=0.45)
        for row in body_rows[1:]:
            self.play(FadeIn(row, shift=DOWN * 0.08), run_time=0.62)
            self.play(Circumscribe(row[1], color=CARDINAL, fade_out=True), run_time=0.48)

        takeaway = self.caption(
            "The ordered merge table is the tokenizer; encoding later replays these rules.",
            color=INK,
        ).to_edge(DOWN, buff=0.48)
        self.play(FadeIn(takeaway), run_time=0.65)
        self.wait(1.0)


class CS336BPETrainVsUse(CS336LectureScene):
    """L01-BPE-TRAIN-VS-USE: freeze learned rules, then replay them in order."""

    def construct(self) -> None:
        header = self.header(
            "L01-BPE-TRAIN-VS-USE", "Training learns rules; encoding only replays them"
        )
        source = self.source_note("lecture_01.py · lines 549–564, 705–720")
        self.play(
            FadeIn(header[0][0]),
            FadeIn(header[0][1]),
            Create(header[1]),
            FadeIn(source),
            run_time=0.12,
        )

        train_panel = RoundedRectangle(
            width=5.75,
            height=3.8,
            corner_radius=0.11,
            fill_color=PANEL,
            fill_opacity=1,
            stroke_color=RULE,
            stroke_width=1.2,
        ).shift(LEFT * 3.05 + DOWN * 0.38)
        use_panel = train_panel.copy().shift(RIGHT * 6.1)
        train_title = Text("TRAIN · count on corpus", font=MONO, font_size=18, color=CARDINAL, weight=BOLD)
        use_title = Text("USE · no new counts", font=MONO, font_size=18, color=ACCENT, weight=BOLD)
        train_title.move_to(train_panel.get_top() + DOWN * 0.34)
        use_title.move_to(use_panel.get_top() + DOWN * 0.34)
        self.play(
            FadeIn(train_panel), FadeIn(use_panel), FadeIn(train_title), FadeIn(use_title), run_time=0.55
        )

        corpus = Text('"the cat in the hat"', font=MONO, font_size=20, color=INK)
        corpus.next_to(train_title, DOWN, buff=0.3)
        learned = VGroup(
            Text("1   (t, h)       → 256 = th", font=MONO, font_size=16, color=INK),
            Text("2   (256, e)     → 257 = the", font=MONO, font_size=16, color=INK),
            Text("3   (257, space) → 258 = the+space", font=MONO, font_size=16, color=INK),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.18)
        learned.next_to(corpus, DOWN, buff=0.34)
        freeze = _chip(
            "freeze ordered merge map",
            width=4.45,
            height=0.55,
            fill="#F6E8E7",
            stroke=CARDINAL,
            text_color=CARDINAL_DARK,
            font_size=16,
        ).next_to(learned, DOWN, buff=0.32)
        self.play(Write(corpus), FadeIn(learned, lag_ratio=0.12), run_time=0.9)
        self.play(FadeIn(freeze), Circumscribe(learned, color=CARDINAL, fade_out=True), run_time=0.75)

        held_out = Text('"the quick brown fox"', font=MONO, font_size=20, color=INK)
        held_out.next_to(use_title, DOWN, buff=0.3)
        start_label = Text("UTF-8 bytes", font=MONO, font_size=14, color=MUTED)
        start_row = _token_row(["t", "h", "e", "SP"], width=0.5, buff=0.04).scale(0.78)
        start_more = Text("+ 15 unchanged bytes", font=MONO, font_size=14, color=MUTED)
        start = VGroup(start_label, start_row, start_more).arrange(RIGHT, buff=0.16)
        start.next_to(held_out, DOWN, buff=0.36)
        self.play(Write(held_out), FadeIn(start), run_time=0.7)

        replay_rows = VGroup(
            Text("apply 1  th | e | space  → 18 tokens", font=MONO, font_size=15, color=INK),
            Text("apply 2  the | space     → 17 tokens", font=MONO, font_size=15, color=INK),
            Text("apply 3  the+space       → 16 tokens", font=MONO, font_size=15, color=CARDINAL_DARK, weight=BOLD),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.22)
        replay_rows.next_to(start, DOWN, buff=0.36).align_to(held_out, LEFT)
        for row in replay_rows:
            self.play(FadeIn(row, shift=RIGHT * 0.1), run_time=0.52)

        lossless = Text(
            "decode(encode(text)) == text",
            font=MONO,
            font_size=16,
            color=SUCCESS,
            weight=BOLD,
        ).next_to(replay_rows, DOWN, buff=0.34)
        self.play(FadeIn(lossless), run_time=0.5)
        takeaway = self.caption(
            "Statistics stop after training; inference is deterministic for a fixed ordered merge map.",
            color=INK,
        ).to_edge(DOWN, buff=0.75)
        self.play(FadeIn(takeaway), run_time=0.55)
        self.wait(1.1)


class CS336TokenizerTradeoffToy(CS336LectureScene):
    """L01-TOKENIZER-PARETO: exact counts for one declared toy setup."""

    def construct(self) -> None:
        header = self.header(
            "L01-TOKENIZER-PARETO", "One toy setup, exact tokenizer trade-offs"
        )
        source = self.source_note("lecture_01.py · lines 505–571, 676–720")
        self.play(
            FadeIn(header[0][0]),
            FadeIn(header[0][1]),
            Create(header[1]),
            FadeIn(source),
            run_time=0.12,
        )

        setup = VGroup(
            Text('train: "the cat in the hat"', font=MONO, font_size=17, color=MUTED),
            Text('held out: "the quick brown fox" · 19 UTF-8 bytes', font=MONO, font_size=17, color=INK),
            Text('toy rules: 3 BPE merges · word vocab = observed chunks + UNK', font=MONO, font_size=15, color=CARDINAL_DARK),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.1)
        setup.next_to(header, DOWN, buff=0.3).to_edge(LEFT, buff=0.66)
        self.play(FadeIn(setup, lag_ratio=0.08), run_time=0.7)

        columns = [
            ("tokenizer", 0.0),
            ("V / ID domain", 2.5),
            ("tokens", 5.25),
            ("bytes/token", 7.0),
            ("OOV", 9.4),
        ]
        table_left = -5.65
        table_top = 1.22
        header_cells = VGroup()
        for label, offset in columns:
            cell = Text(label, font=MONO, font_size=15, color=MUTED, weight=BOLD)
            cell.move_to([table_left + offset, table_top, 0]).align_to(
                [table_left + offset, table_top, 0], LEFT
            )
            header_cells.add(cell)
        header_rule = Line(
            LEFT * 5.72,
            RIGHT * 5.72,
            stroke_color=RULE,
            stroke_width=1.4,
        ).move_to([0, table_top - 0.28, 0])
        self.play(FadeIn(header_cells), Create(header_rule), run_time=0.5)

        rows_data = [
            ("byte", "256", "19", "1.00", "0", SUCCESS),
            ("character*", "1,114,112", "19", "1.00", "0", ACCENT),
            ("word toy", "5 + UNK", "7", "2.71", "3", WARNING),
            ("BPE · 3 merges", "259", "16", "1.19", "0", CARDINAL),
        ]
        rows = VGroup()
        for row_index, values in enumerate(rows_data):
            cells = VGroup()
            for value, (_, offset) in zip(values[:5], columns):
                color = values[5] if offset == 0 else (CARDINAL_DARK if value == "3" else INK)
                cell = Text(
                    value,
                    font=MONO,
                    font_size=17,
                    color=color,
                    weight=BOLD if offset == 0 else "NORMAL",
                )
                anchor = [table_left + offset, table_top - 0.72 - row_index * 0.62, 0]
                cell.move_to(anchor).align_to(anchor, LEFT)
                cells.add(cell)
            rows.add(cells)
        for row in rows:
            self.play(FadeIn(row, shift=RIGHT * 0.1), run_time=0.48)

        formula = Text(
            "bytes/token = 19 ÷ emitted tokens",
            font=MONO,
            font_size=15,
            color=MUTED,
        ).to_edge(LEFT, buff=0.66).shift(DOWN * 2.05)
        char_note = Text(
            "* CharacterTokenizer uses Python ord/chr's full code-point ID domain.",
            font=MONO,
            font_size=13,
            color=MUTED,
        ).next_to(formula, DOWN, buff=0.12).align_to(formula, LEFT)
        self.play(FadeIn(formula), FadeIn(char_note), run_time=0.55)
        self.play(
            Circumscribe(rows[2], color=WARNING, fade_out=True),
            Circumscribe(rows[3], color=CARDINAL, fade_out=True),
            run_time=0.75,
        )
        takeaway = self.caption(
            "Shorter is not automatically better: word tokens pay here with three unknown chunks.",
            color=INK,
        ).to_edge(DOWN, buff=0.75)
        self.play(FadeIn(takeaway), run_time=0.55)
        self.wait(1.1)
