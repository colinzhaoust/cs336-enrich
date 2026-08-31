"""Lecture 3 V2 flagship: decode traffic -> MQA extreme -> GQA -> evidence.

This silent deterministic traversal is a fallback/review capture for the
responsive companion.  It follows slides 57-63 rather than opening with an
attention-family glossary.
"""

from __future__ import annotations

from manim import (
    Arrow,
    BOLD,
    Circle,
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
AMBER = "#9A5A16"
AMBER_PALE = "#F7EDDE"
SANS = "Helvetica Neue"
MONO = "Menlo"


def txt(label: str, size: int, color: str = INK, *, weight: str = "NORMAL") -> Text:
    return Text(label, font=SANS, font_size=size, color=color, weight=weight)


def mono(label: str, size: int, color: str = INK, *, weight: str = "NORMAL") -> Text:
    return Text(label, font=MONO, font_size=size, color=color, weight=weight)


def fit(mobject, width: float):
    if mobject.width > width:
        mobject.scale_to_fit_width(width)
    return mobject


def card(width: float, height: float, *, fill: str = "#FFFFFF", stroke: str = RULE):
    return RoundedRectangle(
        width=width,
        height=height,
        corner_radius=0.13,
        fill_color=fill,
        fill_opacity=1,
        stroke_color=stroke,
        stroke_width=2,
    )


def pill(label: str, width: float, *, fill=PANEL, stroke=RULE, color=INK, size=48):
    shell = RoundedRectangle(
        width=width,
        height=0.70,
        corner_radius=0.10,
        fill_color=fill,
        fill_opacity=1,
        stroke_color=stroke,
        stroke_width=2,
    )
    label_mob = mono(label, size, color, weight=BOLD)
    fit(label_mob, width - 0.20)
    label_mob.move_to(shell)
    return VGroup(shell, label_mob)


class KVDecodeEvidenceV2(Scene):
    def setup(self) -> None:
        self.camera.background_color = PAPER

    def banner(self, label: str, color: str = CARDINAL, fill_color: str = CARDINAL_PALE):
        shell = RoundedRectangle(
            width=12.85,
            height=0.92,
            corner_radius=0.12,
            fill_color=fill_color,
            fill_opacity=1,
            stroke_color=color,
            stroke_width=2,
        )
        label_mob = txt(label, 58, color, weight=BOLD)
        fit(label_mob, 12.15)
        label_mob.move_to(shell)
        return VGroup(shell, label_mob).to_edge(UP, buff=0.47)

    def prefill_decode(self):
        banner = self.banner("DECODE ADDS ONE TOKEN AT A TIME")
        left_shell = card(5.95, 5.25)
        right_shell = card(5.95, 5.25)
        left_title = txt("PREFILL", 62, GREEN, weight=BOLD)
        right_title = txt("DECODE STEP", 62, CARDINAL, weight=BOLD)
        fit(right_title, 4.95)
        prefill_tokens = VGroup(
            *[pill(str(i), 0.72, fill=GREEN_PALE, stroke=GREEN, color=GREEN, size=42) for i in range(1, 7)]
        ).arrange(RIGHT, buff=0.10)
        prefill_note = txt("many tokens in one parallel pass", 48, INK)
        fit(prefill_note, 4.95)
        decode_token = pill("new", 1.35, fill=CARDINAL_PALE, stroke=CARDINAL, color=CARDINAL, size=44)
        history = pill("prior K/V cache", 3.75, fill=BLUE_PALE, stroke=BLUE, color=BLUE, size=44)
        read_arrow = Arrow(history.get_bottom(), history.get_bottom() + DOWN * 0.75, buff=0.05, color=BLUE, stroke_width=4)
        read_note = txt("read the existing history", 48, INK)
        fit(read_note, 4.95)
        left_content = VGroup(left_title, prefill_tokens, prefill_note).arrange(DOWN, buff=0.65)
        right_content = VGroup(right_title, decode_token, history, read_arrow, read_note).arrange(DOWN, buff=0.31)
        left_content.move_to(left_shell)
        right_content.move_to(right_shell)
        panels = VGroup(VGroup(left_shell, left_content), VGroup(right_shell, right_content)).arrange(RIGHT, buff=0.38)
        panels.shift(DOWN * 0.30)
        return VGroup(banner, panels)

    def cache_growth(self):
        banner = self.banner("EACH DECODE STEP REREADS A GROWING CACHE", BLUE, BLUE_PALE)
        shell = card(12.70, 5.45)
        label = txt("KV CACHE IN MEMORY", 50, MUTED, weight=BOLD)
        cache_slots = VGroup(
            *[
                pill(f"K,V {i}", 1.45, fill=BLUE_PALE, stroke=BLUE, color=BLUE, size=38)
                for i in range(1, 5)
            ]
        ).arrange(RIGHT, buff=0.18)
        meter = pill("read 1 cached unit", 4.20, fill=CARDINAL_PALE, stroke=CARDINAL, color=CARDINAL, size=44)
        arrow = Arrow(cache_slots.get_bottom(), cache_slots.get_bottom() + DOWN * 0.86, buff=0.08, color=CARDINAL, stroke_width=5)
        saved = txt("cache saves recomputing old K/V", 48, GREEN, weight=BOLD)
        cost = txt("but decode moves prior K/V again", 48, CARDINAL, weight=BOLD)
        fit(saved, 5.10)
        fit(cost, 5.10)
        note = VGroup(saved, cost).arrange(RIGHT, buff=0.45)
        content = VGroup(label, cache_slots, arrow, meter, note).arrange(DOWN, buff=0.38)
        content.move_to(shell)
        group = VGroup(banner, VGroup(shell, content).shift(DOWN * 0.30))
        return group, cache_slots, meter

    def byte_ledger(self):
        banner = self.banner("COUNT CACHE BYTES BEFORE COMPARING HEADS", BLUE, BLUE_PALE)
        shell = card(12.70, 5.45)
        formula = mono("2 × n_kv × d_head × bytes/value", 68, CARDINAL, weight=BOLD)
        fit(formula, 11.35)
        qualifier = txt("per cached token · per layer", 54, MUTED, weight=BOLD)
        multiply = mono("× sequence × layers × batch", 58, BLUE, weight=BOLD)
        fit(multiply, 10.60)
        config = pill("8 Q heads · d_head 128 · BF16 · 32 layers · 4096 tokens · batch 1", 10.65, fill=PANEL, size=40)
        result_left = VGroup(
            txt("PER TOKEN · PER LAYER", 36, MUTED, weight=BOLD),
            mono("4 KiB", 66, CARDINAL, weight=BOLD),
        ).arrange(DOWN, buff=0.14)
        result_right = VGroup(
            txt("MODEL-WIDE CACHE", 36, MUTED, weight=BOLD),
            mono("512 MiB", 66, CARDINAL, weight=BOLD),
        ).arrange(DOWN, buff=0.14)
        results = VGroup(result_left, result_right).arrange(RIGHT, buff=1.55)
        content = VGroup(formula, qualifier, multiply, config, results).arrange(DOWN, buff=0.28)
        content.move_to(shell)
        return VGroup(banner, VGroup(shell, content).shift(DOWN * 0.30))

    def sharing_state(self, name: str, kv_heads: int, cache_mib: int, ratio: str, note: str, color: str, pale: str):
        banner = self.banner("QUERY HEADS STAY FIXED · KV HEADS CONTROL TRAFFIC", BLUE, BLUE_PALE)
        diagram_shell = card(8.45, 5.35)
        meter_shell = card(3.85, 5.35)
        q_label = txt("8 QUERY HEADS · FIXED", 44, BLUE, weight=BOLD)
        queries = VGroup(
            *[pill(f"Q{i+1}", 0.78, fill=BLUE_PALE, stroke=BLUE, color=BLUE, size=34) for i in range(8)]
        ).arrange(RIGHT, buff=0.12)
        kv_label = txt(f"{kv_heads} K/V HEAD{'S' if kv_heads > 1 else ''}", 50, color, weight=BOLD)
        kvs = VGroup(
            *[pill(f"KV{i+1}", 0.92, fill=pale, stroke=color, color=color, size=34) for i in range(kv_heads)]
        ).arrange(RIGHT, buff=max(0.14, min(0.60, 5.7 / max(kv_heads, 1) - 0.92)))
        connectors = VGroup()
        for query_index, query in enumerate(queries):
            kv_index = min(kv_heads - 1, query_index * kv_heads // 8)
            connectors.add(Line(kvs[kv_index].get_bottom(), query.get_top(), color=RULE, stroke_width=2))
        stack = VGroup(kv_label, kvs, connectors, q_label, queries)
        kv_label.move_to(diagram_shell.get_center() + UP * 1.70)
        kvs.move_to(diagram_shell.get_center() + UP * 0.88)
        q_label.move_to(diagram_shell.get_center() + DOWN * 0.65)
        queries.move_to(diagram_shell.get_center() + DOWN * 1.52)
        for connector, query_index in zip(connectors, range(8)):
            kv_index = min(kv_heads - 1, query_index * kv_heads // 8)
            connector.put_start_and_end_on(kvs[kv_index].get_bottom(), queries[query_index].get_top())
        name_text = txt(name, 72, color, weight=BOLD)
        cache_label = txt("DECLARED TOY CACHE", 38, MUTED, weight=BOLD)
        fit(cache_label, 3.05)
        cache_value = mono(f"{cache_mib} MiB", 66, color, weight=BOLD)
        ratio_pill = pill(ratio, 2.65, fill=pale, stroke=color, color=color, size=46)
        note_text = txt(note, 44, INK, weight=BOLD)
        fit(note_text, 3.20)
        meter_content = VGroup(name_text, cache_label, cache_value, ratio_pill, note_text).arrange(DOWN, buff=0.38)
        meter_content.move_to(meter_shell)
        panels = VGroup(VGroup(diagram_shell, stack), VGroup(meter_shell, meter_content)).arrange(RIGHT, buff=0.38)
        panels.shift(DOWN * 0.30)
        return VGroup(banner, panels)

    def evidence(self):
        banner = self.banner("RETURN TO SLIDE 63 · QUALITY IS EMPIRICAL", CARDINAL, CARDINAL_PALE)
        left_shell = card(5.95, 4.85)
        right_shell = card(5.95, 4.85)
        badge_left = pill("EMPIRICAL · SHAZEER 2019", 4.95, fill=PANEL, color=MUTED, size=36)
        badge_right = pill("EMPIRICAL · AINSLIE 2023", 4.95, fill=PANEL, color=MUTED, size=36)
        left_title = txt("MQA: small PPL hit", 58, CARDINAL, weight=BOLD)
        fit(left_title, 5.05)
        left_value = mono("29.9  →  30.2", 58, CARDINAL, weight=BOLD)
        left_scope = txt("Billion Word LM rows shown", 42, MUTED)
        right_title = txt("GQA: low / no hit", 58, GREEN, weight=BOLD)
        right_scope = txt("in the cited comparisons", 48, MUTED)
        fit(left_scope, 5.05)
        fit(right_title, 5.05)
        fit(right_scope, 5.05)
        left_content = VGroup(badge_left, left_title, left_value, left_scope).arrange(DOWN, buff=0.42)
        right_content = VGroup(badge_right, right_title, right_scope).arrange(DOWN, buff=0.52)
        left_content.move_to(left_shell)
        right_content.move_to(right_shell)
        panels = VGroup(VGroup(left_shell, left_content), VGroup(right_shell, right_content)).arrange(RIGHT, buff=0.38)
        caveat = pill(
            "not a universal guarantee across models, scales, data, or training recipes",
            11.75,
            fill=AMBER_PALE,
            stroke=AMBER,
            color=AMBER,
            size=42,
        )
        body = VGroup(panels, caveat).arrange(DOWN, buff=0.25).shift(DOWN * 0.25)
        return VGroup(banner, body)

    def construct(self) -> None:
        current = self.prefill_decode()
        self.play(FadeIn(current), run_time=0.8)
        self.wait(4.2)

        growth, slots, meter = self.cache_growth()
        for slot in slots[1:]:
            slot.set_opacity(0.18)
        self.play(ReplacementTransform(current, growth), run_time=0.8)
        self.wait(1.0)
        for index in range(1, 4):
            slots[index].set_opacity(1)
            next_meter = pill(
                f"read {index + 1} cached units",
                4.20,
                fill=CARDINAL_PALE,
                stroke=CARDINAL,
                color=CARDINAL,
                size=44,
            ).move_to(meter)
            self.play(
                FadeIn(slots[index]),
                ReplacementTransform(meter, next_meter),
                run_time=0.5,
            )
            meter = next_meter
            self.wait(0.8)
        self.wait(2.3)

        ledger = self.byte_ledger()
        self.play(ReplacementTransform(growth, ledger), run_time=1.0)
        self.wait(7.0)

        mha = self.sharing_state("MHA", 8, 512, "1× KV bytes", "one KV stream per Q", CARDINAL, CARDINAL_PALE)
        self.play(ReplacementTransform(ledger, mha), run_time=1.0)
        self.wait(6.0)

        mqa = self.sharing_state("MQA", 1, 64, "1/8 KV bytes", "capacity extreme", AMBER, AMBER_PALE)
        self.play(ReplacementTransform(mha, mqa), run_time=1.0)
        self.wait(6.0)

        gqa = self.sharing_state("GQA", 2, 128, "1/4 KV bytes", "capacity / traffic knob", GREEN, GREEN_PALE)
        self.play(ReplacementTransform(mqa, gqa), run_time=1.0)
        self.wait(6.0)

        evidence = self.evidence()
        self.play(ReplacementTransform(gqa, evidence), run_time=1.0)
        self.wait(11.8)
