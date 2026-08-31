"""Reproduce the exact toy counts shown in CS336TokenizerTradeoffToy.

This intentionally mirrors the simple Lecture 1 implementations. It is not a
tokenizer benchmark: one tiny training string teaches both the word vocabulary
and three ordered BPE merges, then every tokenizer sees one held-out string.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict


TRAIN_TEXT = "the cat in the hat"
EVAL_TEXT = "the quick brown fox"
NUM_MERGES = 3


def merge(indices: list[int], pair: tuple[int, int], new_index: int) -> list[int]:
    output: list[int] = []
    i = 0
    while i < len(indices):
        if i + 1 < len(indices) and tuple(indices[i : i + 2]) == pair:
            output.append(new_index)
            i += 2
        else:
            output.append(indices[i])
            i += 1
    return output


def train_bpe(text: str, num_merges: int) -> dict[tuple[int, int], int]:
    indices = list(text.encode("utf-8"))
    merges: dict[tuple[int, int], int] = {}
    for i in range(num_merges):
        counts: dict[tuple[int, int], int] = defaultdict(int)
        for pair in zip(indices, indices[1:]):
            counts[pair] += 1
        pair = max(counts, key=counts.get)
        new_index = 256 + i
        merges[pair] = new_index
        indices = merge(indices, pair, new_index)
    return merges


def encode_bpe(text: str, merges: dict[tuple[int, int], int]) -> list[int]:
    indices = list(text.encode("utf-8"))
    for pair, new_index in merges.items():
        indices = merge(indices, pair, new_index)
    return indices


def main() -> None:
    num_bytes = len(EVAL_TEXT.encode("utf-8"))
    merges = train_bpe(TRAIN_TEXT, NUM_MERGES)
    bpe_tokens = encode_bpe(EVAL_TEXT, merges)

    word_pattern = r"\w+|."
    train_chunks = re.findall(word_pattern, TRAIN_TEXT)
    eval_chunks = re.findall(word_pattern, EVAL_TEXT)
    word_vocab = set(train_chunks)
    word_oov = sum(chunk not in word_vocab for chunk in eval_chunks)

    rows = [
        {
            "tokenizer": "byte",
            "vocab_or_id_domain": 256,
            "tokens": num_bytes,
            "bytes_per_token": num_bytes / num_bytes,
            "oov_chunks": 0,
        },
        {
            "tokenizer": "character",
            "vocab_or_id_domain": 0x110000,
            "tokens": len(EVAL_TEXT),
            "bytes_per_token": num_bytes / len(EVAL_TEXT),
            "oov_chunks": 0,
        },
        {
            "tokenizer": "word toy",
            "vocab_or_id_domain": len(word_vocab) + 1,
            "vocab_note": f"{len(word_vocab)} observed chunks + UNK",
            "tokens": len(eval_chunks),
            "bytes_per_token": num_bytes / len(eval_chunks),
            "oov_chunks": word_oov,
        },
        {
            "tokenizer": f"BPE · {NUM_MERGES} merges",
            "vocab_or_id_domain": 256 + NUM_MERGES,
            "tokens": len(bpe_tokens),
            "bytes_per_token": num_bytes / len(bpe_tokens),
            "oov_chunks": 0,
        },
    ]

    payload = {
        "provenance": {
            "official_source": "lecture_01.py lines 505–571, 676–720, 729–758",
            "training_text": TRAIN_TEXT,
            "held_out_text": EVAL_TEXT,
            "held_out_utf8_bytes": num_bytes,
            "num_bpe_merges": NUM_MERGES,
            "word_pattern": word_pattern,
        },
        "ordered_bpe_merges": [
            {"pair": list(pair), "new_id": new_id}
            for pair, new_id in merges.items()
        ],
        "rows": rows,
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
