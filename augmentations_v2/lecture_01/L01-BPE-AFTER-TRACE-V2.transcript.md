# Visual transcript: BPE after the training trace

This silent optional replay is inserted after 75:22, once Percy has finished
the continuous BPE training code trace. It does not repeat the pair-count
demonstration. Its purpose is to make the training/use boundary visible before
the lecture resumes with the reference encoder's performance limitations.

The context descriptions below are paraphrases, not quotations.

## Visual states

1. **Final training state (0:00–0:04).** The corpus is `the cat in the hat`.
   The checked calculation shows 18 input bytes becoming 12 tokens after three
   merges, while vocabulary size grows from 256 to 259.
2. **Freeze learned parameters (0:04–0:09).** The ordered rules are shown as
   `t+h→256`, `256+e→257`, and `257+space→258`. A “Frozen” marker appears.
3. **Initialize held-out text (0:09–0:14).** `the quick brown fox` begins as 19
   UTF-8 byte tokens. The first four are `116, 104, 101, 32`; the remaining 15
   bytes are marked unchanged. Pair counts are deliberately absent.
4. **Replay rank 0 (0:14–0:19).** Applying `t+h→256` changes the prefix to
   `256, 101, 32` and reduces the sequence from 19 to 18 tokens.
5. **Replay rank 1 (0:19–0:24).** Applying `256+e→257` changes the prefix to
   `257, 32` and reduces the sequence from 18 to 17 tokens.
6. **Replay rank 2 (0:24–0:29).** Applying `257+space→258` changes the prefix to
   `258` and reduces the sequence from 17 to 16 tokens. Unmatched bytes remain.
7. **Decode (0:29–0:34).** The frozen vocabulary maps the tokens back to bytes;
   UTF-8 decoding reconstructs `the quick brown fox` exactly.
8. **State ledger (0:34–0:40.40).** Training changes pair counts, vocabulary,
   the merge table, and the corpus sequence. Encoding computes no new pair
   counts and freezes the vocabulary and merge order; only the held-out token
   sequence changes.

## Claim and evidence ledger

| Beat | Professor/source claim | Augmentation claim | Evidence and caveat |
| --- | --- | --- | --- |
| Training result | Three rounds learn three merges on `the cat in the hat`. | The final training lengths are 18 bytes to 12 tokens and vocabulary 256 to 259. | Reproduced by `bpe_fixture.py` from the official slow reference logic. |
| Freeze | `BPETokenizerParams` contains vocabulary and ordered merges. | Both become fixed inputs to encoding. | Official lines 541–558; Python dictionary insertion order is used by this teaching implementation. |
| Held-out input | Encoding starts from UTF-8 bytes of `the quick brown fox`. | There are 19 initial byte tokens and no pair-count stage. | Official lines 554–558 and 714–718; reproduced fixture. |
| Rule replays | Encoding loops through learned merges and calls `merge`. | The three states have lengths 18, 17, and 16. | Official lines 527–538 and 557–558; exact arrays are checked in the fixture. |
| Decode | Decoding joins vocabulary bytes and checks equality. | The final sequence reconstructs the identical string. | Official lines 561–564 and 719–720. |
| Caveat | Percy calls this encoder very slow and next discusses Assignment 1 extensions. | The state distinction is semantic, not a prescription for an efficient implementation. | Production tokenizers optimize lookup and handle pretokenization and special tokens. |

## Browser behavior

The HTML stepper exposes Previous, Next, and Replay transition controls. The
first pass pauses for prediction before each merge reveal. Left and right arrow
keys move between available states. Reduced-motion users receive immediate
state replacement. At a narrow viewport, the sequence and rule panes stack so
all essential text remains at least 16 CSS pixels.
