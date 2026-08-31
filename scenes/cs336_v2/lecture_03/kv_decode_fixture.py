"""Checked arithmetic for the Lecture 3 KV decode/GQA companion."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path


@dataclass(frozen=True)
class DeclaredConfig:
    query_heads: int = 8
    head_dim: int = 128
    bytes_per_value: int = 2  # BF16
    layers: int = 32
    sequence_tokens: int = 4096
    batch: int = 1


def cache_bytes_per_token_per_layer(kv_heads: int, config: DeclaredConfig) -> int:
    """K and V elements stored for one cached token in one layer."""

    return 2 * kv_heads * config.head_dim * config.bytes_per_value


def model_cache_bytes(kv_heads: int, config: DeclaredConfig) -> int:
    return (
        cache_bytes_per_token_per_layer(kv_heads, config)
        * config.sequence_tokens
        * config.layers
        * config.batch
    )


def build_fixture() -> dict:
    config = DeclaredConfig()
    variants = [
        ("MHA", 8, "one KV stream per query head"),
        ("MQA", 1, "one shared KV stream: the capacity extreme"),
        ("GQA", 2, "two KV groups: an intermediate capacity/traffic setting"),
        ("GQA-4", 4, "four KV groups: another point on the same knob"),
    ]
    records = []
    for name, kv_heads, interpretation in variants:
        per_layer = cache_bytes_per_token_per_layer(kv_heads, config)
        total = model_cache_bytes(kv_heads, config)
        records.append(
            {
                "name": name,
                "queryHeads": config.query_heads,
                "kvHeads": kv_heads,
                "bytesPerCachedTokenPerLayer": per_layer,
                "modelCacheBytes": total,
                "modelCacheMiB": total / 2**20,
                "relativeKvTrafficVsMha": kv_heads / config.query_heads,
                "interpretation": interpretation,
            }
        )
    per_step = [
        {
            "cachedTokens": cached_tokens,
            "mhaKvReadBytesPerLayer": cached_tokens
            * cache_bytes_per_token_per_layer(8, config),
        }
        for cached_tokens in [1, 2, 3, 4]
    ]
    return {
        "provenance": {
            "formula": "reproduced calculation; dimensional accounting derived from slides 58-62",
            "qualityEvidence": "course slide 63; empirical results from Shazeer 2019 and Ainslie et al. 2023",
        },
        "declaredConfiguration": asdict(config),
        "formula": "2 * n_kv_heads * d_head * bytes_per_value per cached token per layer",
        "modelFormula": "per_token_per_layer * sequence_tokens * layers * batch",
        "variants": records,
        "toyDecodeRead": per_step,
        "slide63Evidence": [
            {
                "source": "Shazeer 2019 table shown on course slide 63",
                "setting": "Billion Word LM; h=8 rows displayed by the course",
                "mhaDevPpl": 29.9,
                "mqaDevPpl": 30.2,
                "courseCharacterization": "small perplexity hit",
                "evidenceType": "empirical result in one cited setting",
            },
            {
                "source": "Ainslie et al. 2023 plots shown on course slide 63",
                "courseCharacterization": "low/no quality hit with GQA in the cited comparison",
                "evidenceType": "empirical result in the paper's evaluated settings",
            },
        ],
        "evidenceCaveat": "Neither cited result guarantees that more KV sharing preserves quality for every model, scale, dataset, or training recipe.",
    }


def assert_fixture(fixture: dict) -> None:
    by_name = {item["name"]: item for item in fixture["variants"]}
    assert {item["queryHeads"] for item in fixture["variants"]} == {8}
    assert by_name["MHA"]["bytesPerCachedTokenPerLayer"] == 4096
    assert by_name["MQA"]["bytesPerCachedTokenPerLayer"] == 512
    assert by_name["GQA"]["bytesPerCachedTokenPerLayer"] == 1024
    assert by_name["MHA"]["modelCacheMiB"] == 512
    assert by_name["MQA"]["modelCacheMiB"] == 64
    assert by_name["GQA"]["modelCacheMiB"] == 128
    assert by_name["MQA"]["relativeKvTrafficVsMha"] == 1 / 8
    assert by_name["GQA"]["relativeKvTrafficVsMha"] == 1 / 4
    assert [row["mhaKvReadBytesPerLayer"] for row in fixture["toyDecodeRead"]] == [
        4096,
        8192,
        12288,
        16384,
    ]
    assert fixture["slide63Evidence"][0]["mqaDevPpl"] > fixture["slide63Evidence"][0]["mhaDevPpl"]
    assert all(item["evidenceType"].startswith("empirical") for item in fixture["slide63Evidence"])


def main() -> None:
    fixture = build_fixture()
    assert_fixture(fixture)
    output = Path(__file__).with_name("kv_decode_fixture.json")
    output.write_text(json.dumps(fixture, indent=2) + "\n", encoding="utf-8")
    print(f"PASS: verified and wrote {output}")


if __name__ == "__main__":
    main()
