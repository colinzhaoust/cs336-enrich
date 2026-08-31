# Visual transcript: decode traffic to MQA/GQA evidence

This silent companion follows the official Lecture 3 run on slides 57-63. It
does not introduce attention variants as an isolated glossary. It begins with
the deployment problem Tatsu establishes, keeps query-head count fixed while
K/V sharing changes, and ends by returning to the empirical evidence on slide
63. Context descriptions are paraphrases, not quotations.

## Visual states

1. **Prefill versus decode (0:00-0:05).** Prefill processes many input tokens
   in parallel. Autoregressive generation adds one query token at a time, and
   each new query attends to the prior K/V history.
2. **Growing repeated read (0:05-0:13).** Four K/V cache blocks appear one by
   one. A meter changes from reading one cached unit to four. The cache avoids
   recomputing old keys and values, but decode moves the growing history again
   at each step.
3. **Exact byte ledger (0:13-0:21).** The formula is `2 * n_kv_heads * d_head
   * bytes_per_value` per cached token per layer. Only afterward is it
   multiplied by sequence length, layer count, and batch. For the declared toy
   configuration, MHA is 4 KiB per token per layer and 512 MiB model-wide.
4. **MHA reference (0:21-0:28).** Eight query heads connect to eight K/V heads.
   The declared 512 MiB cache is the 1x traffic reference.
5. **MQA extreme (0:28-0:35).** Query heads remain eight while K/V heads
   collapse to one. Declared cache bytes fall to 64 MiB, one eighth of MHA.
   The frame labels this as a capacity extreme rather than an automatic win.
6. **GQA knob (0:35-0:42).** Query heads remain eight while two K/V groups are
   shown. The declared cache is 128 MiB, one quarter of MHA. The responsive
   companion also permits one, two, four, or eight K/V heads.
7. **Evidence boundary (0:42-0:54.80).** Motion stops and the learner is sent
   back to official slide 63. The course characterizes the displayed Shazeer
   2019 result as a small MQA perplexity hit; the shown rows are 29.9 for MHA
   and 30.2 for MQA. It characterizes the displayed Ainslie 2023 comparisons
   as low/no GQA quality hit. Both are labeled empirical and setting-specific,
   not universal guarantees.

## Claim and evidence ledger

| Beat | Professor/deck claim | Augmentation claim | Evidence and caveat |
| --- | --- | --- | --- |
| Decode | Incremental generation cannot parallelize across future tokens and uses a KV cache. | A simple growing strip makes repeated reads visible. | Slides 58-60; strip length is illustrative. |
| Bytes | Incremental attention has poor arithmetic intensity and the `n/d` term is difficult. | Exact KV storage is `2 * n_kv_heads * d_head * bytes/value` per token per layer. | Dimensional calculation; model total explicitly multiplies sequence, layers, and batch. It is not total inference memory traffic. |
| MQA | Multiple queries can share one K/V stream to reduce memory movement. | The declared toy cache falls from 512 MiB to 64 MiB when K/V heads go 8 to 1. | Reproduced calculation. Real latency depends on kernels, batch, hardware, and layout. |
| GQA | Fewer than eight but more than one K/V heads form an expressiveness/efficiency knob. | Two K/V groups give 128 MiB in the declared configuration. | Reproduced calculation; query heads remain fixed at eight. |
| Quality | Slide 63 says MQA sometimes has a small PPL hit and GQA has low/no hit in the cited evidence. | The companion keeps the empirical labels and values adjacent to the caveat. | Shazeer 2019 and Ainslie et al. 2023 as presented on official slide 63; no universal guarantee. |

## Responsive behavior

At 390 CSS pixels the visual and calculator panes stack, retain a 16-pixel base
font, and avoid horizontal scrolling. Previous, Next, Replay transition, left
arrow, and right arrow controls expose the seven states. The GQA state enables
the K/V-head slider. Reduced-motion users receive immediate state changes.
