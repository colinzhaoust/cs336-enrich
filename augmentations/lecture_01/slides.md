# Lecture 1 augmented slide copy

These are short companion slides, not replacements for Percy's originals. Each
section is keyed by the stable site segment ID so it can be linked to one video
and one GitHub Discussion.

## L01-UTF8-BYTES

**Slide title:** One character is not necessarily one byte

**Learning check:** Given a short Unicode string, can you predict which
characters expand the byte-token sequence?

| Character | Code point | UTF-8 bytes | Length |
| --- | --- | --- | ---: |
| `A` | `U+0041` | `41` | 1 byte |
| `中` | `U+4E2D` | `E4 B8 AD` | 3 bytes |
| `🌍` | `U+1F30D` | `F0 9F 8C 8D` | 4 bytes |

**Narration:** UTF-8 is a variable-length encoding. A byte tokenizer therefore
has a fixed vocabulary of 256 possible byte values and universal coverage, but
the number of model positions depends on the input language and characters.
Decoding groups the byte stream according to UTF-8's prefix rules and recovers
the original Unicode string.

**Pause question:** For the fixed string `A中🌍`, how many tokens does a pure
byte tokenizer emit? Answer: 8.

**Accessibility note:** Byte groups use both spacing and labels; color is not
the only indication of group membership.

## L01-BPE-PAIR-COUNT

**Slide title:** BPE statistics come from overlapping adjacent windows

**Toy corpus:** `the cat in the hat`

**Exact initial counts:** `(t,h)`, `(h,e)`, `(e,space)`, and `(a,t)` each occur
twice; all other adjacent pairs occur once. The animation chooses `(t,h)`
because it is the first maximum encountered. A production implementation must
make its tie-breaking convention reproducible.

**Narration:** Start from the current tokenization, scan every adjacent pair,
and accumulate a frequency table. The pair windows overlap: the second token of
one window is the first token of the next. Select one maximum only after the
entire corpus has been counted.

**Pause question:** Why can two implementations trained on the same corpus
learn different vocabularies? Answer: ties can be resolved differently unless
the ordering rule is fixed.

## L01-BPE-MERGE

**Slide title:** A merge is a new vocabulary entry plus a corpus rewrite

| Round | New token | Corpus length |
| ---: | --- | ---: |
| Start | — | 18 |
| 1 | `256 = th` | 16 |
| 2 | `257 = the` | 14 |
| 3 | `258 = the␠` | 12 |

**Narration:** Each selected pair receives one new token ID. Replacing every
non-overlapping occurrence shortens this toy corpus by two positions in each of
the first three rounds. The ordered list of learned merges is the tokenizer
model: later encoding starts from bytes and replays the rules in order.

**Boundary:** Real tokenizers add pretokenization, special-token handling, and
efficient data structures. This scene isolates only the pair-count and merge
invariant.

## L01-BPE-TRAIN-VS-USE

**Slide title:** Learn the merge order once; replay it on every new string

**Official source:** `lecture_01.py` lines 549–564 define `encode()` as an
ordered loop over `params.merges`; lines 705–720 train on `the cat in the hat`,
freeze the returned parameters, and encode `the quick brown fox`.

| Phase | Input | Operation | Result |
| --- | --- | --- | --- |
| Train | `the cat in the hat` | Count current adjacent pairs after every round | `(t,h)→256`, `(256,e)→257`, `(257,space)→258` |
| Use | `the quick brown fox` | Start from 19 UTF-8 bytes; replay the three rules in order | 16 token IDs |
| Decode | Those 16 IDs | Concatenate each ID's byte value, then UTF-8 decode | Original held-out string |

**Key distinction:** Pair frequencies belong to tokenizer training. They are
not recomputed for each input at inference. With fixed parameters and a fixed
merge order, encoding is deterministic.

**Pause question:** Why must the merge map preserve order rather than merely
store a set of pairs? Answer: later pairs can contain token IDs created by
earlier merges; `(256,e)` is meaningful only after `(t,h)→256` has run.

## L01-TOKENIZER-PARETO

**Slide title:** Exact counts for one declared toy setup

This is a reproducible teaching example, not a measured tokenizer benchmark or
a mathematical Pareto frontier.

**Toy provenance:** Train the word vocabulary and three-merge BPE on
`the cat in the hat`; evaluate every tokenizer on the held-out ASCII string
`the quick brown fox`, which contains exactly 19 UTF-8 bytes. The word splitter
is the source regex `\w+|.`. Its five observed training chunks plus one `UNK`
give six IDs. BPE starts with 256 bytes and adds three learned IDs, giving 259.

| Tokenizer | Vocabulary / ID domain | Emitted tokens | Bytes/token (`19 ÷ tokens`) | OOV chunks |
| --- | ---: | ---: | ---: | ---: |
| Byte | 256 | 19 | 1.00 | 0 |
| Character | 1,114,112 Python code-point IDs | 19 | 1.00 | 0 |
| Word toy | 5 observed chunks + `UNK` | 7 | 2.71 | 3 |
| BPE, 3 merges | 259 | 16 | 1.19 | 0 |

**Narration:** The word toy emits the fewest tokens only by collapsing `quick`,
`brown`, and `fox` to three unknowns. The BPE toy shortens the byte sequence
without an unknown token because rare text remains decomposable into bytes.
Character and byte tokenization tie on this all-ASCII held-out string; a
multibyte Unicode string would separate their token counts.

**Reproduce:** Run
`python3 scenes/cs336/lecture_01/reproduce_tokenizer_tradeoff.py`. It prints the
ordered merges, setup, and all table values as JSON.

**Learning check:** Why isn't “use the shortest possible token sequence” a
complete tokenizer objective? Answer: vocabulary/ID-domain cost, coverage, and
the treatment of unseen text remain separate constraints.
