"""Deterministic toy states for the Lecture 1 BPE after-trace augmentation.

The implementation mirrors the intentionally slow reference code in the
official Spring 2026 ``lecture_01.py``.  It is deliberately dependency-free so
that both the Manim scene and the browser stepper can be checked against the
same fixture.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass
import json
from pathlib import Path


TRAINING_CORPUS = "the cat in the hat"
HELD_OUT_TEXT = "the quick brown fox"


def merge(indices: list[int], pair: tuple[int, int], new_index: int) -> list[int]:
    """Replace every eligible non-overlapping occurrence, left to right."""

    merged: list[int] = []
    index = 0
    while index < len(indices):
        if (
            index + 1 < len(indices)
            and indices[index] == pair[0]
            and indices[index + 1] == pair[1]
        ):
            merged.append(new_index)
            index += 2
        else:
            merged.append(indices[index])
            index += 1
    return merged


def train_bpe(text: str, rounds: int) -> tuple[dict[tuple[int, int], int], dict[int, bytes], list[int]]:
    indices = list(text.encode("utf-8"))
    merges: dict[tuple[int, int], int] = {}
    vocab = {value: bytes([value]) for value in range(256)}
    for round_index in range(rounds):
        counts: dict[tuple[int, int], int] = defaultdict(int)
        for pair in zip(indices, indices[1:]):
            counts[pair] += 1
        pair = max(counts, key=counts.get)
        new_index = 256 + round_index
        merges[pair] = new_index
        vocab[new_index] = vocab[pair[0]] + vocab[pair[1]]
        indices = merge(indices, pair, new_index)
    return merges, vocab, indices


@dataclass(frozen=True)
class ReplayState:
    step: str
    active_rule: int | None
    indices: list[int]


def build_fixture() -> dict:
    merges, vocab, training_indices = train_bpe(TRAINING_CORPUS, 3)
    held_out_bytes = list(HELD_OUT_TEXT.encode("utf-8"))
    states = [ReplayState("held-out-bytes", None, held_out_bytes)]
    indices = held_out_bytes
    for rank, (pair, new_index) in enumerate(merges.items()):
        indices = merge(indices, pair, new_index)
        states.append(ReplayState(f"replay-rank-{rank}", rank, indices))
    decoded = b"".join(vocab[index] for index in indices).decode("utf-8")
    return {
        "provenance": "reproduced calculation from official lecture_01.py:527-564,705-720,729-757",
        "trainingCorpus": TRAINING_CORPUS,
        "heldOutText": HELD_OUT_TEXT,
        "initialVocabularySize": 256,
        "finalVocabularySize": len(vocab),
        "trainingInitialLength": len(TRAINING_CORPUS.encode("utf-8")),
        "trainingFinalIndices": training_indices,
        "rules": [
            {"rank": rank, "pair": list(pair), "newIndex": new_index}
            for rank, (pair, new_index) in enumerate(merges.items())
        ],
        "states": [asdict(state) for state in states],
        "decoded": decoded,
    }


def assert_fixture(fixture: dict) -> None:
    assert fixture["rules"] == [
        {"rank": 0, "pair": [116, 104], "newIndex": 256},
        {"rank": 1, "pair": [256, 101], "newIndex": 257},
        {"rank": 2, "pair": [257, 32], "newIndex": 258},
    ]
    assert fixture["trainingInitialLength"] == 18
    assert len(fixture["trainingFinalIndices"]) == 12
    assert fixture["finalVocabularySize"] == 259
    assert [len(state["indices"]) for state in fixture["states"]] == [19, 18, 17, 16]
    assert fixture["states"][1]["indices"][:3] == [256, 101, 32]
    assert fixture["states"][2]["indices"][:2] == [257, 32]
    assert fixture["states"][3]["indices"][0] == 258
    assert fixture["decoded"] == fixture["heldOutText"]


def main() -> None:
    fixture = build_fixture()
    assert_fixture(fixture)
    target = Path(__file__).with_name("bpe_fixture.json")
    target.write_text(json.dumps(fixture, indent=2) + "\n", encoding="utf-8")
    print(f"PASS: verified and wrote {target}")


if __name__ == "__main__":
    main()
