---
title: L10 · Inference, read through
minutes: 45
---
This lecture is about running a trained model: given a prompt, produce the response as accurately and as quickly as possible. It shows, with the same byte-and-FLOP accounting as [L2](#/read/lecture_02), why generating text leaves a GPU waiting on memory, and then walks through the ways to fix that: shrink the KV cache, quantize, prune and distill, draft and verify, and schedule live traffic well. After it you can estimate a model's decode latency and throughput from its shape, say which stage of inference is compute-bound and which memory-bound, and explain what each speed-up technique trades away.

## Why does inference deserve its own lecture? {#why}
source: lecture_10.py:L16-L97 · video 0:05-8:55

Once a model is trained, everyone except the researcher who plots its loss wants to *use* it. Inference shows up in many places:
- **actual use**: chatbots, code completion, agents, batch data processing;
- **evaluation**, whenever a benchmark needs the model to generate;
- **reinforcement learning**, which samples many generations (rollouts), scores them and updates the weights, so inference sits inside training.

::figure official/lectures/images/inference-schema.png | the whole job in one picture: a prompt goes in, tokens come out one after another

Efficiency matters even more here than in training, because the two costs have different shapes. Training is a one-time cost: very expensive, but once it is done, it is done. Inference is a repeated cost, paid every day. The professor gave the arithmetic aloud. OpenAI is estimated to produce about 8.6 trillion tokens a day; DeepSeek-V4 was trained on 32 trillion tokens. So in $32 / 8.6 \approx 3.7$ days, OpenAI generates as many tokens as a frontier training run consumed.

The importance has also grown with agents. A chatbot's tokens are meant to be read by a human, and humans read slowly, so beyond some speed faster inference stops mattering. An agent instead goes from a query through a long internal trace (thinking, tool calls, introspection) to an output, and most of the tokens it produces are "not for reading". For an agent, tokens generated *are* compute spent, and there is no ceiling on how much value a faster model unlocks.

::code lecture_10.py:L69-L76 | one-time against repeated cost, and why agents make the token count unbounded

Inference is big business, both for providers serving closed models and for those serving open-weight ones, and there are four open-source stacks worth knowing by name: **vLLM** (from Berkeley, pioneered PagedAttention, the "go-to"), **SGLang** (also Berkeley, RadixAttention, good for agentic workloads), **TensorRT-LLM** (NVIDIA, very fast but narrower) and **llama.cpp** (C++, runs on a CPU, locally). Two of their ideas, PagedAttention and continuous batching, close this lecture.

### What does "fast" mean?

There is no single number. The lecture uses three:
- **Time-to-first-token (TTFT)**: how long the user waits before any output appears. It matters for interactive use: the dead wait before the first token is what users feel. Once text starts streaming, it "doesn't maybe have to be that fast", because the reader cannot read that fast anyway.
- **Latency**, in **seconds per token**: how fast tokens appear for *one* query. Also an interactive metric; it is the speed of the stream. Note the unit: in this lecture latency is per token, not per request.
- **Throughput**, in **tokens per second**: how fast tokens appear summed over *many* queries. This is the batch-processing metric: if you have a petabyte of documents to score, you only care when the whole job finishes.

Latency and throughput usually move together, and many interventions improve both. But there is a lever that moves them in opposite directions, and finding it is the point of the next few sections.

::predict inference-metrics
::code lecture_10.py:L90-L93 | three metrics, each attached to a kind of workload

### Why is inference harder to make efficient than training?

The professor called this the one high-level bit to remember from the lecture. In training (including supervised fine-tuning) you see all the tokens at once. In a Transformer the sequence is just another dimension of a big tensor, so attention and the MLP process every position in one big matrix multiplication. In inference you cannot do that: generation is **autoregressive**, so token $t+1$ cannot be computed before token $t$ has been sampled. You cannot parallelize across the sequence dimension, and that is why, as the next sections show, it is hard to reach high arithmetic intensity and fully use the GPU.

::code lecture_10.py:L95-L97 | training parallelizes over the sequence; generation cannot
::video 7:18-8:07 | the one idea to keep: inference cannot parallelize over the sequence the way training does

The lecture has three parts, and so does this read-through: understanding the workload (the math of intensity, latency and throughput), taking shortcuts (lossy: smaller KV caches, quantization, pruning) and using shortcuts but double-checking (lossless: speculative sampling), and finally handling dynamic workloads (continuous batching, PagedAttention).

::note aside 0:31 | Inference gets "only one lecture" in the course, though it is "of growing importance". Much of the math and several figures come from the [scaling book's inference chapter](https://jax-ml.github.io/scaling-book/inference/), which the professor recommends reading.
::kp inference-metrics

## How do we write down a Transformer's shapes? {#notation}
source: lecture_10.py:L100-L115 · video 8:55-14:22

All the accounting below is done symbolically, in the notation of the scaling book's [Transformer chapter](https://jax-ml.github.io/scaling-book/transformers/). It works like einops (see [L2's einsum](#/read/lecture_02)): each letter names a dimension and also stands for its length.
- **B**: batch, the number of sequences;
- **T**: sequence, the number of tokens;
- **D**: model dimension;
- **H**: head dimension.

A product is written by listing each operand's dimensions. In $BTD \times DH \to BTH$, D appears in both operands and disappears from the result: it is a **contracting** dimension (red in the lecture), summed over. B, T and H appear in one operand and stay: regular dimensions. A third kind appears in both operands *and* stays in the result; that is a **batching** dimension (blue). For example $BD \times BD \to B$ takes one dot product per batch element. Remember this case: a batching dimension shared by both operands is exactly what will make attention slow at inference.

::code lecture_10.py:L103-L108 | contracting dimensions vanish, batching dimensions survive in both operands and the result

With that notation, the scaling book draws a whole Transformer block, attention followed by a gated MLP, with every tensor's shape.

::figure https://jax-ml.github.io/scaling-book/assets/img/transformer-diagram.png | follow the letters: Q carries N heads, K and V carry K heads; in attention B appears in both operands (batching), in the MLP the weights carry no B

The professor finds this "probably the most crisp definition of what a Transformer is", because it tells you exactly the shape of every tensor. Reading it:
- **Attention.** X (the layer's activations) is projected by a query matrix to $Q$ of shape $B\,T\,N\,H$, N heads of dimension H, and by key and value matrices to $K$ and $V$ with **K** heads instead of N; K can be smaller than N (grouped-query attention, below). In the attention product, B appears in both Q and K: a batching dimension. The head dimension is contracted.
- **MLP.** A gate matrix and an up projection take D to F; a down projection takes F back to D. Here the weight matrices have no B or T at all.

The conventions used throughout:
- $F = 4D$: the MLP up-projects into four times the model dimension;
- $D = NH$: the model dimension is split across N heads;
- $N = KG$: with grouped-query attention, the N query heads are split into K groups of G heads each;
- **S** and **T** both measure sequence length: S is the number of tokens we condition on, T the number we produce. In training they are equal ($S = T$); in generation, T = 1.

::note skip 13:05 | The block review is kept to notation, because students implemented the Transformer in Assignment 1: "I don't want to belabor it, but just to swap in the notation."
::note slip 20:11 | A student pointed out that the diagram swaps K and G. The professor agreed: K should be the number of KV groups and G the heads per group ("I'll fix that later"). The lecture's own convention, $N = KG$, is the corrected one.

## When is a matrix multiply compute-bound? {#matmul-intensity}
source: lecture_10.py:L118-L159 · video 14:22-18:46

A warm-up from [L2's arithmetic intensity](#/read/lecture_02). Recall the idea: an operation's **arithmetic intensity** is FLOPs per byte moved between HBM and the compute units; the **accelerator intensity** is the hardware's FLOP/s divided by its memory bandwidth. Above the accelerator's intensity you are compute-bound (good); below it, memory-bound (bad: the compute units wait for bytes).

Multiply X ($B \times D$) by W ($D \times F$). Think of B as the batch of tokens, D as the model dimension and F as the MLP's up-projection dimension. Everything is in bf16, 2 bytes per number, which the professor says is always the case for inference.
1. Read X: $2BD$ bytes.
2. Read W: $2DF$ bytes.
3. Compute $Y = XW$: $2BDF$ FLOPs.
4. Write Y ($B \times F$): $2BF$ bytes.

::code lecture_10.py:L127-L137 | four steps, each adding to a FLOP or byte counter; the asserts check the totals

$$ \text{intensity} = \frac{2BDF}{2BD + 2DF + 2BF} $$

The numerator is cubic in the dimensions, while every read and write is quadratic, "and that's how you're going to get high arithmetic intensity". Now assume the batch is much smaller than both D and F. The code makes this precise by substituting $D = cB$, $F = cB$ and letting $c \to \infty$:

$$ \frac{2B \cdot c^2B^2}{2cB^2 + 2c^2B^2 + 2cB^2} \;\xrightarrow{c\to\infty}\; B $$

The $DF$ read dominates the bytes, and $D$ and $F$ cancel. **The intensity is the number of tokens, B.** This is the non-square analogue of L2's square-matrix result, $n/3$.

::code lecture_10.py:L140-L144 | sympy takes the limit and asserts intensity == B

The H100 does 989e12 FLOP/s (dense bf16) and moves 3.35e12 bytes/s, an accelerator intensity of $989 / 3.35 \approx 295$. So this matmul is compute-bound if and only if $B > 295$.

The extreme case is $B = 1$, a matrix-vector product: you read the whole $D \times F$ matrix to do only $2DF$ FLOPs, intensity 1, about 300 times short of the hardware. "This is basically what happens with inference": instead of full matrices you get "very thin matrices".

::code lecture_10.py:L146-L159 | 989e12 / 3.35e12 ≈ 295, so compute-bound iff B > 295; B = 1 is intensity 1

A bigger model does not help. Widening F raises the FLOPs and the dominant byte term $2DF$ by the same factor, so the intensity stays at B. "Memory-bound" here means **bandwidth**-bound: the GPU may have plenty of memory capacity and still spend its time waiting for the weights to arrive.

## How do you avoid recomputing the whole history? The KV cache {#kv-cache}
source: lecture_10.py:L162-L176 · video 20:27-25:22

The most naive way to generate: a Transformer is a black box that takes a sequence and returns a distribution over the next token. So feed it the prompt, sample a token, append it, feed the whole longer sequence again, and repeat.

::figure https://jax-ml.github.io/scaling-book/assets/img/naive-inference-1400.webp | each new token reruns the whole history through the model

This works and is terrible. One forward pass over $t$ tokens costs $O(t^2)$, because attention compares every position with every other. Doing it once per generated token makes generating T tokens $O(T^3)$ FLOPs.

The observation that fixes it: a lot of the work is shared across prefixes. When the model generates the fifth token, the keys and values it computes for the first four tokens are exactly the ones it computed a step earlier. So store them. The **KV cache** keeps, in HBM, every layer's key and value vectors for every token already processed, and each step reuses them instead of recomputing.

::figure https://jax-ml.github.io/scaling-book/assets/img/cached-inference-1400.webp | the cache is filled once by the prompt; each step adds one token's keys and values and reads all the earlier ones
::code lecture_10.py:L169-L172 | for every sequence (B), token (S), layer (L) and KV head (K), store an H-dimensional vector

"An H-dimensional vector" is really two of them, a key and a value; the byte count later has a factor 2 for that. What the cache saves is the old tokens' K and V projections, the only thing earlier positions contribute to a new token's attention. The new token itself still runs through every layer's attention and MLP.

### Why is reusing the cache exact?

The code says the work "can be shared" but not why that is legitimate. The professor said it aloud: "this is because it's a causal transformer." In a causal (decoder-only) model a token attends only to itself and earlier tokens, so a token's hidden states, and so its keys and values in every layer, cannot depend on anything appended later. Appending a token leaves every cached K and V *exactly* unchanged. The cache is not an approximation.

In a bidirectional model (BERT-style), "if you attach a token, then everything changes": every earlier token attends to the new one, so from the second layer up its hidden states, and hence its K and V, change. (Layer 1's keys and values are projections of each token's own embedding, so those would survive.)

::predict kv-cache-needs-causality
::widget fixture:lecture_10--kv-recompute | under the causal mask only the new token's column is computed; switch to bidirectional and every earlier token's cells turn orange from layer 2 up, while layer 1 stays grey
::video 22:24-22:41 | "this is because it's a causal transformer. If it was bidirectional ... everything changes"

### Two stages: prefill and generation

With a cache, inference splits into two stages that behave very differently:
1. **Prefill**: take the prompt and encode it, populating the cache. You see the entire prompt, so this is parallelizable "just like in training". It also produces the distribution for the first output token.
2. **Generation** (decode): produce response tokens one at a time. Each step runs one new token through the model, reading the cache and appending that token's K and V to it.

The cache does not make generation parallel; it only spares you from paying again for tokens already seen. With it, step $t$ attends over $t$ cached tokens, $O(t)$ work, so the attention for generating T tokens totals $O(T^2)$ instead of $O(T^3)$: doubling the length quadruples it rather than multiplying it by eight.

::predict prefill-vs-decode
::code lecture_10.py:L174-L176 | prefill is parallel like training; generation is sequential

Time-to-first-token is, essentially, prefill time: nothing can be emitted until the whole prompt has been encoded. Seconds per token afterwards is set by generation.

Where the cache lives matters. It is stored in HBM, so every decode step must read it back; how large it gets, and what reading it costs, is the subject of the latency section below.

::kp prefill-vs-decode
::kp kv-cache-needs-causality

## Which stage is memory-bound, and where exactly? {#stage-intensity}
source: lecture_10.py:L178-L260 · video 25:22-35:07

Now the warm-up's accounting is applied to the two kinds of layer in a Transformer, for both stages at once. S is the number of tokens conditioned on and T the number of tokens processed now; afterwards, prefill is $T = S$ and generation $T = 1$. Only the matrix multiplications are counted, because they hold the work; everything else is few FLOPs and can be fused into them.

### The MLP: intensity is the number of tokens

The gated MLP reads X ($B \times T \times D$) and three weight matrices (up, gate, down, $2DF$ bytes each), computes the up projection U and the gate G and writes both to HBM, then computes $Y = \text{GeLU}(G) \cdot U$ times the down projection and writes Y. The totals:

$$ \text{FLOPs} = 6BTDF, \qquad \text{bytes} = 4BTD + 4BTF + 6DF $$

::code lecture_10.py:L187-L205 | read X and the three weights, write U and G, write Y; the asserts give 6BTDF FLOPs and 4BTD + 4BTF + 6DF bytes

With the same assumption as before, $BT \ll D, F$, the weight reads dominate and the intensity is $BT$. That makes sense: an MLP "is basically a big matmul", and the batch and sequence dimensions "don't interact" in it, so together they act as one token dimension.

::code lecture_10.py:L208-L211 | the limit gives intensity == B*T

What this means for the two stages:
- **Prefill**: easy to make compute-bound. Make $BT$ large, with large batches or long prompts. One 1024-token prompt alone has intensity 1024 in the limit (about 865 by the exact formula for Llama 2 13B's D and F), far above 295.
- **Generation**: two problems. $T = 1$, one token at a time, so the intensity is just B. And B is now the **number of concurrent requests**. In a batch job you control it; serving a chatbot, it is however many users happen to be talking to you, which can be high or low and changes over time: "a bit unpredictable". Getting a large B anyway is the job of continuous batching, at the end.

::predict mlp-intensity-batch
::widget fixture:lecture_10--intensity | in generation the MLP bar sits at about B whatever the width F; it crosses the 295 line only once B·T passes about 300 tokens
::note skip 26:12 | The step-by-step gated-MLP accounting was not walked aloud ("I'm not going to maybe belabor step through every single detail"); only the result, B·T, was discussed.

### Attention: no B at all

With FlashAttention (see [L5](#/read/lecture_05)), the $S \times T$ score matrix never touches HBM, so attention reads Q ($B \times T \times D$), K and V ($B \times S \times D$ each) and writes Y ($B \times T \times D$). Two matrix products, $QK^\top$ and softmax times V, cost $2BSTD$ FLOPs each.

$$ \text{FLOPs} = 4BSTD, \qquad \text{bytes} = 4BSD + 4BTD $$

::code lecture_10.py:L225-L240 | read Q, K, V; two products; write Y; the intensity simplifies to S·T / (S + T)

Divide, and B and D cancel from every term:

$$ \text{intensity} = \frac{4BSTD}{4BSD + 4BTD} = \frac{ST}{S+T} $$

::worked attention-decode-intensity

For **prefill**, $T = S$ gives $S/2$: good, as long as prompts are long. For **generation**, $T = 1$ gives $S/(S+1)$, which is below 1 however long the context ("or even let's just call it 1"). At $S = 4096$ it is 0.9998, against the 295 the H100 needs. A longer context brings more work and proportionally more KV bytes, so the ratio only creeps toward 1 from below.

::predict attention-decode-intensity
::code lecture_10.py:L242-L248 | prefill: S/2, "Good!"; generation: below 1, "Bad!"

### Why batching rescues the MLP but not attention

The intensity of attention has no B in it, so adding concurrent requests cannot help. The reason is which tensors are shared:
- In the MLP, **every sequence hits the same weights**. $W_\text{up}$, $W_\text{gate}$ and $W_\text{down}$ have no B. A large batch loads them once and uses them for every sequence; that reuse is where the intensity comes from.
- In attention, **every sequence has its own KV cache**. Q, K and V all carry B. For each sequence you are doing a separate small product against its own cache, and doing more of them is not "helpful".

This is the blue B from the notation section: B is a *batching* dimension of the attention product, in both operands. Batching over a dimension that appears in both operands is like taking many dot products, which has terrible intensity.

::code lecture_10.py:L250-L253 | MLP weights don't depend on B; Q, K, V all do
::video 32:09-32:58 | MLP weights are loaded once for the whole batch, but each sequence's KV cache is its own
::note slip 32:23 | Aloud: "So these all depend on MLP." The code line says Q, K and V "all depend on B", which is what the argument needs. He also read the attention FLOPs as "B times S times T times D"; the code's count is 4·B·S·T·D. The intensity, ST/(S+T), is unaffected either way.

### The scorecard

| | MLP | attention |
|---|---|---|
| prefill ($T = S$) | $BS$: great | $S/2$: workable |
| generation ($T = 1$) | $B$: workable, needs many concurrent requests | $S/(S+1) < 1$: the bottleneck |

Prefill is compute-bound; generation is memory-bound. Of the four entries, generation attention is the fundamental bottleneck: "If you're sticking with a transformer, you can't really improve this." The code says "impossible to improve", meaning the ratio itself, for the standard attention accounting above. What the rest of the lecture does instead is cut the *bytes*: if the time is bytes divided by bandwidth, fewer bytes per step means faster steps, even at the same poor ratio.

::code lecture_10.py:L255-L260 | the four intensities in one list
::widget fixture:lecture_10--intensity | move B: the attention bar does not move at all while the MLP bar follows B; switch to prefill and attention jumps to S/2

"So now whenever you hear people say, oh, inference is memory bound, you know why."

::note aside | Put in whole-model terms: a decode step at B = 1 does about 2 FLOPs per parameter and reads 2 bytes per parameter, an intensity of 1, so on an H100 the arithmetic would finish about 300 times sooner than the weights arrive.
::kp mlp-intensity-batch
::kp attention-decode-intensity

## How fast can one GPU generate? Latency and throughput {#latency-throughput}
source: lecture_10.py:L263-L368 · video 35:07-45:47

Because generation is memory-bound, timing it becomes simple: count the bytes each step must move and divide by bandwidth. "In some ways it's nice because it's simpler. But in other ways, it's frustrating that your accelerators are sitting there not doing anything." The assumption, stated in the code, is that compute and communication overlap perfectly and overhead is ignored, so what comes out is a best case: a lower bound on latency, an upper bound on throughput.

### What sits in memory

The lecture's running example is Llama 2 13B on one H100. Its shape: context $S = 1024$, $D = 5120$, $F = 13824$, $N = K = 40$ heads (plain multi-head attention, "there's no GQA here"), $H = 128$, $L = 40$ layers, vocabulary $V = 32000$, and bandwidth 3.35e12 bytes/s.

::code lecture_10.py:L317-L329 | the Llama 2 13B shape and the H100's bandwidth

Two things occupy HBM during decoding.
- **The parameters.** Embedding and unembedding ($2VD$), the gated MLP ($3DF$ per layer) and the attention projections ($2DNH$ for Q and the output, $2DKH$ for K and V, per layer). With this config that is 13.0e9 parameters, "a good sanity check" against the model's name. In bf16 they take $2 \times 13.0\text{e}9 = 26.0$ GB. (The comment notes that training needs a larger multiple; see [L2's memory ledger](#/read/lecture_02).)
- **The KV cache.** Per sequence: S tokens × K heads × H dimensions × L layers × 2 (key and value) × 2 (bytes in bf16).

$$ \text{KV bytes per sequence} = S \cdot K H \cdot L \cdot 2 \cdot 2, \qquad \text{memory} = B \cdot \text{KV per sequence} + 2 \cdot \text{params} $$

For Llama 2 13B that is $40 \times 128 \times 40 \times 4 = 819{,}200$ bytes per token, and 0.839 GB per 1024-token sequence, the "about 838 million times B" the professor read off the trace.

::code lecture_10.py:L291-L300 | parameter count, 2 bytes per parameter, the KV cache per sequence, and the total
::predict kv-cache-size
::widget fixture:lecture_10--kv-ledger | the orange KV bar grows by 0.84 GB per sequence while the 26 GB weight bar stays put; sequences that fit = (80 − 26) / 0.84

The cache grows linearly in batch and in context, while the weights are fixed. At $B = 64$ the cache is $64 \times 0.839 = 53.7$ GB, already twice the weights: the KV cache "could even be larger than the number of parameters" at a large enough batch.

### Latency and throughput from bytes

Each decode step reads all the parameters and every sequence's whole KV cache, and produces one token for each of the B sequences:

$$ \text{latency} = \frac{2\cdot\text{params} + B \cdot \text{KV per sequence}}{\text{bandwidth}}, \qquad \text{throughput} = \frac{B}{\text{latency}} $$

Latency is in seconds per token (per step); throughput has the extra factor B because a step produces B tokens.

::code lecture_10.py:L302-L306 | latency is memory IO; throughput is B over latency

At $B = 1$: 26.87 GB per step, $26.87\text{e}9 / 3.35\text{e}12 = 8.0$ ms per token, about 125 tokens per second (the professor read "0.008 seconds per token" and "124 tokens per second"). No FLOP rate appears anywhere: one user's tokens per second is set by how fast the weights can be streamed past the compute units, so a model with fewer bytes is faster in proportion.

::predict decode-bandwidth-bound
::widget fixture:lecture_10--batch-sweep | at B = 1 the latency is almost all weight read: halve the bytes (fewer parameters, int4 weights, more GPUs sharing the read) and the latency halves
::video 35:27-35:58 | decode time is bytes over bandwidth, with the compute idle
::kp kv-cache-size
::kp decode-bandwidth-bound

### The trade-off: batch size

The code's `b1`, `b64` and `b256` evaluate the same formula at three batch sizes. They are marked `@stepover`, so the values are not printed in the source; the numbers here are the formula evaluated (the B = 1 values match what was read aloud).

| batch B | memory | latency | throughput |
|---|---|---|---|
| 1 | 26.9 GB | 8.0 ms/token | 125 tokens/s |
| 64 | 79.7 GB | 23.8 ms/token | 2,690 tokens/s |
| 256 | 240.8 GB (does not fit in 80 GB) | 71.9 ms/token | 3,562 tokens/s |

::code lecture_10.py:L341-L353 | B = 1, 64, 256; the assert confirms 256 overflows the H100's 80e9 bytes

Raising B does two things at once:
- it **worsens latency**, because each step must read a KV cache that is B times larger;
- it **improves throughput**, because the cost of reading the parameters is amortized over B tokens.

So "fast" has two meanings that, depending on which you care about, are "completely opposite", and the batch size decides between them. The professor's picture is a bus. Each rider waits for everyone to board and ride together, so the latency is poor; but the bus moves everyone at once, so its throughput is good.

The throughput gains also diminish. Latency is a linear function of B whose intercept is the parameter bytes and whose slope is the KV cache per sequence, so throughput is "proportional to B over B plus something" and levels off.

Going from 64 to 256 is four times the batch for 1.32 times the throughput. The curve bends where $B$ × KV bytes equals the parameter bytes, $26.03 / 0.839 \approx 31$, and approaches a ceiling of bandwidth over KV bytes per sequence, $3.35\text{e}12 / 0.839\text{e}9 \approx 3{,}990$ tokens/s. In practice "you'll never get to the asymptote because you'll hit the memory": B = 256 needs 241 GB.

::predict batch-latency-throughput-tradeoff
::widget fixture:lecture_10--batch-sweep | throughput rises steeply until B ≈ 31, where the KV read equals the weight read, then bends toward the dashed ceiling while latency keeps growing linearly
::video 43:40-43:59 | the bus: each rider waits for the batch, the bus moves everyone at once

Two more rules close the section.
- **Copies.** Launch M copies of the model, on M times the hardware: latency stays the same and throughput rises M-fold. The harder kind of parallelism, sharding one model and its KV cache across devices, is left to the scaling book.
- **Per stage.** Time-to-first-token is essentially prefill time, so use smaller batches during prefill for fast TTFT and larger batches during generation for throughput.

::code lecture_10.py:L363-L368 | M copies scale throughput; TTFT is prefill; small batches for prefill, large for generation
::note skip 44:30 | "I'm not going to really talk about parallelism too much": sharding a model across devices for inference is left to the [scaling book's inference chapter](https://jax-ml.github.io/scaling-book/inference/).
::kp batch-latency-throughput-tradeoff

## How do you make the KV cache smaller? {#kv-reduction}
source: lecture_10.py:L371-L446 · video 45:47-1:04:28

Memory is the bottleneck, and at realistic batch sizes the KV cache is most of the memory. So the first family of shortcuts shrinks the cache, with one condition: do not lose too much accuracy. These are lossy changes to the model, and each must be checked.

### Grouped-query attention (GQA)

Keep all N query heads, but compute only K key and value heads, each shared by $N/K$ query heads.
- **Multi-head attention (MHA)**: $K = N$, no sharing.
- **Multi-query attention (MQA)**: $K = 1$, one key and value for all heads.
- **Grouped-query attention (GQA)**: K in between, "hopefully where we'll find a balance between accuracy and speed".

::figure https://jax-ml.github.io/scaling-book/assets/img/gmqa.png | same number of query heads in all three; only the number of key/value heads changes

Since the cache stores $K \cdot H$ numbers per token per layer, GQA divides it by $N/K$. Attention still does the same work, since each query head still attends over the whole context. The gain is purely in bytes, and because decode is memory-bound, fewer bytes is faster. The GQA paper's speed plot shows this: time per sample falls sharply from MHA to $K = 1$, stays low up to around $K = 8$, then rises.

::figure official/lectures/images/gqa-speed.png | time per sample against the number of KV groups: MHA at the top, small K far faster
::predict gqa-kv-reduction

Rerun the Llama 2 13B numbers with $K = 8$ (a 1:5 ratio of KV heads to query heads):

| | memory | latency | throughput |
|---|---|---|---|
| MHA ($K = 40$), B = 64 | 79.7 GB | 23.8 ms | 2,690 tokens/s |
| GQA ($K = 8$), B = 64 | 33.4 GB | 10.0 ms | 6,417 tokens/s |
| GQA ($K = 8$), B = 256 | 65.6 GB | 19.6 ms | 13,068 tokens/s |

::code lecture_10.py:L390-L402 | the same formula at K = 40 and K = 8; then the batch is raised to 256, which now fits

Read across the first two rows: cutting the memory improved latency *and* throughput. "It's not that latency and throughput are always at odds. If you reduce the amount of memory, then it improves both"; batch size is "the point of tension". And the saved memory can be spent: at K = 8 a batch of 256, which overflowed before, fits in 65.6 GB and roughly doubles throughput again, at a somewhat higher latency. You play these knobs jointly.

::note slip 49:36 | The code's comment for the K = 8, B = 64 run says "Worse latency, but better throughput (and it fits in memory now!)". Against the MHA run at the same B, latency improves (10.0 against 23.8 ms), as the professor said aloud. And the MHA B = 64 run already fit, at 79.7 GB, by 0.3 GB; the run that did not fit was B = 256, which is the comparison he made aloud.
::note aside | K also enters the parameter count (the K and V projections), so the K = 8 model has 11.3e9 parameters, 22.7 GB of weights. Part of its latency gain comes from fewer weights, not only from the smaller cache.

The accuracy check: on the GQA paper's evaluations, GQA does about as well as MHA.

::figure official/lectures/images/gqa-accuracy.png | the paper's accuracy across its evaluations: GQA holds up against MHA
::note aside 51:14 | "Take it with a grain of salt": accuracy results are for a particular model, and the DeepSeek-V2 paper, next, found that GQA does hurt. "Take everything that's not just math with a grain of salt here." MQA, which "no one uses because it's really bad", is the extreme that shows why.
::kp gqa-kv-reduction

### Multi-head latent attention (MLA)

DeepSeek's idea keeps a full set of keys and values for every head but caches them in compressed form. Normally the cache holds $K = W_K h$ and $V = W_V h$, $N \cdot H$ dimensions each, from a token's activation $h$. MLA instead stores one low-dimensional latent

$$ c = W_c\, h \quad (C \text{ dimensions}), \qquad K = W_K\, c, \quad V = W_V\, c \ \text{ when needed.} $$

::figure official/lectures/images/mla-schema.png | MHA, GQA, MQA and MLA side by side: MLA caches one compressed vector per token and projects keys and values up from it
::code lecture_10.py:L409-L413 | cache c instead of K and V; DeepSeek-V2 goes from N·H = 16384 to 512, plus 64 for RoPE

DeepSeek-V2 compresses $N \cdot H = 16384$ to $C = 512$, "quite aggressive".

One wrinkle: MLA is not compatible with RoPE (the rotary position embedding of [L3](#/t/lecture_03)), so 64 extra dimensions carry the rotary part, $512 + 64 = 576$ in all. Per layer per token, MHA caches both K and V, $2 \times 16384 = 32768$ numbers, so MLA's 576 is about 57 times smaller. The speed-up "follow[s] by just simple math": the smaller the cache, the faster, almost linearly while decode stays memory-bound.

::predict mla-latent-kv
::widget fixture:lecture_10--kv-ledger | switch the attention to MLA with N = H = 128: the bar of 512 + 64 values per layer is about 57× shorter than the MHA bar, which holds both K and V

The accuracy check reverses the GQA paper's story. In DeepSeek-V2's Table 8, MHA beats GQA (at higher cost); in Table 9, MLA is even a bit better than MHA while much cheaper. The professor's own reading: "let's just say it's about the same."

::figure official/lectures/images/mla-accuracy.png | Table 8: MHA above GQA on the same benchmarks
::figure official/lectures/images/mla-accuracy2.png | Table 9: MLA matches or slightly beats MHA
::note slip 53:14 | RoPE was said to operate "directly on the keys and values"; it rotates queries and keys, not values. Why exactly MLA and RoPE clash (the rotation depends on position, so it cannot be folded into the shared projection) is not explained in the lecture.
::note spoken 55:02 | Asked how MLA compares with simply shrinking the model dimension: the ablations don't show it. His guess is that uniform shrinking is worse, because it reduces everything indiscriminately: "the trick in all of this kind of is to find places in the model where you can squeeze."
::kp mla-latent-kv
::kp lecture_04:mla-latent-kv-cache

### Cross-layer attention (CLA)

GQA shares keys and values across heads; CLA shares them across **layers**. Only some layers compute their own K and V; the others reuse a previous layer's cache. The paper finds this improves the Pareto frontier of accuracy against KV-cache size: for the same cache size, it is more accurate than shrinking the cache by changing K or the head dimension.

::figure official/lectures/images/cla-diagram.png | some layers compute keys and values, the layers after them reuse that cache
::figure official/lectures/images/cla-results.png | accuracy against KV-cache size: the CLA points sit on a better frontier

### Local (sliding-window) attention

An old and natural idea. Instead of attending to every earlier token, each token attends only to the last W tokens: a sliding window. Then a layer's cache never holds more than W tokens, so the KV cache is **independent of the sequence length**, "which is great", especially for long contexts.

::figure official/lectures/images/longformer-attention.png | full attention against a sliding window, a dilated window, and a window plus a few global positions (Longformer)

The model can still use information from further back, because it propagates one window per layer. After $\ell$ layers a position can be influenced by tokens up to about $\ell W$ back, so the effective context grows linearly with depth. The figure also shows fancier variants: dilated windows, or a window plus attention to a fixed set of global positions.

The catch is that local attention "still hurts accuracy": "There's no free lunch here." The fix used in practice is **hybrid** models that interleave local layers with global (full-attention) layers. Only the local layers' caches are capped.

::code lecture_10.py:L429-L433 | the cache is independent of length; accuracy suffers; interleave with global layers
::predict sliding-window-kv
::widget fixture:lecture_10--kv-ledger | set a window: the green local layers stop at W tokens however long S gets, while the blue global layers keep growing with S

For example, a 32-layer model with a 4096-token window, one global layer after every three local ones, at a 32k context: $(8 \times 32768 + 24 \times 4096) / (32 \times 32768) \approx 0.34$ of the all-global cache.

::note spoken 59:24 | Asked about linear attention against sliding windows: linear attention, gated DeltaNet and Mamba keep a compressed summary of the whole history instead of a cache, and are more expressive than a window; but compressing everything into a fixed state can lose what a needle-in-a-haystack question needs. "I'm not going to talk about linear attention" here; see the [L4 thread](#/t/lecture_04).
::kp sliding-window-kv

### DeepSeek-V4's attention

DeepSeek keeps inventing attention variants. V4 supports a 1M-token context with three mechanisms, which the professor walked through on the figure without dwelling on the acronyms:
- **Compressed Sparse Attention (CSA)** compresses every m tokens of keys and values into one;
- **DeepSeek Sparse Attention (DSA)** selects the top k of those to keep, using a small, cheap attention (an indexer) to score them, "a lightning fast way to figure out what tokens you need to keep";
- **Heavily Compressed Attention (HCA)** compresses even more.

::figure official/lectures/images/deepseek-v4-attention.png | follow the KV tokens up the left: compressed, then top-k selected by the lightning indexer's scores, then concatenated with a sliding window of recent entries before the query attends
::note skip 1:03:40 | "In the interest of time, I'll move on": the rest of DeepSeek-V4's design is not covered. See [L4's DSA](#/t/lecture_04).

### The section in one line

Reduce the KV cache (inference is memory-bound, so this directly buys latency and throughput) without hurting accuracy:
- a **lower-dimensional cache**: share across heads (GQA), across layers (CLA), or compress (MLA);
- **local attention**, truncating the cache, on some of the layers;
- other ideas: linear attention and state-space models (Mamba 2, Gated DeltaNet), and diffusion models, a non-autoregressive way to generate that "can be much faster".

::code lecture_10.py:L442-L446 | the goal and the three families
::kp lecture_04:hybrid-attention-ratios

## Can the numbers themselves be smaller? Quantization {#quantization}
source: lecture_10.py:L449-L487 · video 1:04:28-1:07:39

Quantization is less an architecture change than a systems one: reduce the precision of the numbers. Fewer bytes per weight means fewer bytes per decode step, and since decode is memory-bound, that means lower latency and higher throughput. As always, the price may be accuracy.

### The mechanics

Pick a **scale** (the real-valued size of one integer step) and a **zero point** (which integer stands for 0.0). Then

$$ x_\text{quant} = \mathrm{round}(x / \text{scale}) + \text{zero\_point}, \qquad x_\text{approx} = (x_\text{quant} - \text{zero\_point}) \cdot \text{scale} $$

::code lecture_10.py:L455-L459 | quantize x = 5.2342 with scale 0.1 and zero point 4, then dequantize
::worked quantization-inference

With $x = 5.2342$: $5.2342 / 0.1 = 52.342$, rounded to 52, plus 4 gives 56; dequantized, $(56 - 4) \times 0.1 = 5.2$. Only the rounding is irreversible, so the error is at most half a step (here 0.034, under 0.05). A smaller scale buys precision at the cost of range.

::predict quantization-inference

### The formats

::figure https://www.datocms-assets.com/104802/1709770809-twitter-post-20.png | Baseten's comparison of the number formats listed below

- **fp32** (4 bytes): needed for parameters and optimizer states during training (see [L2's mixed precision](#/read/lecture_02));
- **bf16** (2 bytes): the default for inference;
- **fp8** (1 byte), e4m3 on H100s: you can train in it "if you dare";
- **int8** (1 byte, $[-128, 127]$): less accurate but cheaper than fp8, for inference only;
- **int4** (0.5 bytes, $[-8, 7]$): cheaper still, even less accurate.

::code lecture_10.py:L462-L466 | bytes per number, from fp32 down to int4
::note slip | The code gives fp8 e4m3 a range of [−240, 240] "on H100s". The H100's E4M3 format, the one [L2](#/read/lecture_02) describes, reaches ±448; ±240 is the maximum of a different e4m3 variant used by some other accelerators. The byte count, 1, is right either way.

In the napkin model of the previous sections, int4 weights for Llama 2 13B at B = 1 would cut the per-token bound from about 7.8 ms to about 1.9 ms ($0.5 \times 13.0\text{e}9 / 3.35\text{e}12$), four times less. The real gain is smaller, because the KV cache and other overheads are not quantized away.

::widget fixture:lecture_10--batch-sweep | switch the weights from bf16 to int4 at B = 1: the latency falls 4×; at a large B the KV read dominates and the same switch helps little

### Quantization-aware training and post-training quantization

There are two places to do it.
- **Quantization-aware training (QAT)**: during training, quantize and dequantize in the forward pass, simulating the quantization error, so the weights adapt to it. It works better, but "requires expensive large-scale training".
- **Post-training quantization (PTQ)**: quantize a model after it is trained, which is much cheaper and what people typically do. The naive version runs the model on sample data to pick one scale and zero point per layer or tensor; that "generally doesn't work as well". **GPTQ** quantizes layer by layer using Hessian (second-order) information, and pushes each step's quantization error into the weights not yet quantized, so later weights correct for earlier errors.

::code lecture_10.py:L470-L478 | QAT simulates the error in training; PTQ calibrates afterwards; GPTQ compensates with the Hessian

### Activation-aware quantization (AWQ)

The observation: some activation channels are much larger than others, and the weights those channels multiply matter more. So spend precision where it matters, choosing by activation statistics which 0.1-1% of the weights to keep in high precision (fp16) while the rest go to int3. The paper reports fp16 to int3 giving 4 times less memory and a 3.2 times speed-up. Note how this differs from GPTQ: GPTQ compensates for the rounding error, AWQ decides which weights to protect from it.

::figure official/lectures/images/awq-schema.png | (a) plain rounding to int3 hurts badly; (b) keeping the 1% of weights on large-activation channels in fp16 recovers it, at poor hardware efficiency; (c) scaling those channels before quantizing does the same without mixed precision
::kp quantization-inference

## Can you just make the model smaller? Pruning and distillation {#pruning}
source: lecture_10.py:L490-L504, L32-L41 · video 1:07:39-1:11:43

The crudest shortcut: "rip out parts of an expensive model to make it cheaper, and then fix it up". It turns out to work. The lecture follows a paper from NVIDIA:
1. **Identify** the important layers, heads and hidden dimensions, using a small calibration set (1024 samples).
2. **Remove** the unimportant ones to get a smaller model.
3. **Distill** the original model into the pruned one, to repair it.

::figure official/lectures/images/pruning-kd-loop.png | estimate importance of embedding channels, heads and MLP channels, rank them, trim, then distill from the original; the loop can repeat
::code lecture_10.py:L496-L499 | identify, remove, distill

Asked how importance is measured, the professor said: pass the calibration set through the model and look at the magnitude of the activations. Units that are nearly always close to zero ("dead units") can go; large ones stay. This works because of an empirical fact about trained models: some channels are much larger than others.

Right after removal the model "is not going to be very good"; the distillation step, training the small model to imitate the original, heals it, at a fraction of the cost of training it from scratch. The paper took a 15B model down to 8B without hurting accuracy much.

::figure official/lectures/images/pruning-kd.png | the paper's results for the pruned and distilled models
::predict prune-then-distill

The lecture's summary of all the lossy shortcuts names two recipes for a faster model:
- **from scratch**: define a faster architecture, then train it;
- **distillation**: define a faster architecture, initialize its weights from the original model (even though the architecture differs: "you just make this Frankenstein thing"), then repair it by distillation.

::code lecture_10.py:L32-L41 | the two recipes for a faster model
::note slip | Step 2 of the code says "Remove unimportant layers" though step 1 scored layers, heads and hidden dimensions; aloud, the professor said it removes "different hidden units and different even layers", which matches the paper.
::note spoken 1:11:03 | Asked about a neuron that is always 100: you cannot just remove it, or everything breaks; if it has a high mean and low variance, "maybe there's another way to just incorporate the bias" (hedged).
::note aside 1:11:43 | The pruning section ends with a "# TODO" in the code: it is a stub, and the results are only in the paper's figure.
::kp prune-then-distill

## Can you go faster without changing the output? Speculative sampling {#speculative}
source: lecture_10.py:L507-L550 · video 1:11:43-1:16:55

Everything so far was lossy: it changes the model and may cost accuracy. Speculative sampling (also called speculative decoding) is lossless: it produces exact samples from the original model, only faster.

### The asymmetry it exploits

Recall the two stages. Prefill encodes a given sequence in parallel; it is compute-bound, and as a side effect it gives the model's probability for every position of that sequence. Generation produces one token at a time and is memory-bound. So **checking is faster than generation**: given a sequence, the model can say how likely it is much faster than it could have produced it.

::code lecture_10.py:L508-L511 | prefill also gives you probabilities; checking is faster than generation

The idea, from two papers of the same period ([Leviathan et al. 2022](https://arxiv.org/abs/2211.17192), [Chen et al. 2023](https://arxiv.org/abs/2302.01318)):
- a cheap **draft model** $p$ guesses a few tokens, say 4;
- the **target model** $q$, the one you actually want samples from, scores all of them in one parallel pass and accepts them if they "look good".

The costs are balanced on purpose. The draft is small, so even though it is memory-bound and generates one token at a time, its steps are cheap. The target is big and expensive, but it is asked to process a batch of positions in parallel, which costs about one ordinary decode step: a decode step's time is set by reading the weights and the cache, and that happens once per pass whether the pass scores one position or five.

::animation fixture:lecture_10--spec-rounds | four short sequential draft bars, then one target bar about one decode step long; the target emits the accepted prefix plus one token of its own: 2 tokens after a rejection, 5 when all four are accepted
::video 1:12:50-1:13:11 | why both costs are acceptable: the small draft runs sequentially, the big target checks in parallel

The algorithm, from Chen et al.:

::figure official/lectures/images/speculative-sampling-algorithm.png | draft K tokens one by one; compute K + 1 sets of target logits in parallel; accept each draft with min(1, q/p); at the first rejection, sample from (q − p)₊ and stop; if all are accepted, sample one extra token from q

Walk through it. The draft samples $\tilde{x}_1, \dots, \tilde{x}_K$ autoregressively. The target computes its distribution at all $K+1$ positions in one pass. Then, left to right, each draft token $\tilde{x}_t$ is accepted with probability $\min(1, q(\tilde{x}_t)/p(\tilde{x}_t))$. At the first rejection, a replacement token is drawn from the residual distribution $\max(q - p, 0)$, normalized, and the round ends. If all K are accepted, the target's extra $(K+1)$-th distribution supplies one more token for free. Either way, every round emits **the accepted prefix plus one token**, so at least one.

::predict speculative-sampling-mechanism
::code lecture_10.py:L514-L522 | draft p guesses, target q verifies in parallel; always at least one token; an exact sample

### Why the samples are exact

This is "modified rejection sampling" with proposal $p$ and target $q$. In plain rejection sampling a rejection yields nothing and you try again, which "will keep looping". Here a rejection still yields a token, drawn from the residual, and the result is still "guaranteed to be an **exact sample** from the target model".

The lecture's proof is by example, with a two-token vocabulary {A, B}. Suppose the draft oversamples A: $p(A) > q(A)$. Since both distributions sum to 1, it undersamples B: $p(B) < q(B)$. The residual $\max(q - p, 0)$ is then $[0,\ q(B) - p(B)]$, which normalizes to $[0, 1]$: on rejection, always B.

$$ P[A] = p(A)\cdot\frac{q(A)}{p(A)} + p(B)\cdot 1\cdot 0 = q(A) $$
$$ P[B] = p(B)\cdot 1 + p(A)\left(1 - \frac{q(A)}{p(A)}\right)\cdot 1 = p(B) + p(A) - q(A) = q(B) $$

The only way to end on A is to propose A and accept it. B is reached by proposing it (always accepted, since $q(B)/p(B) > 1$) or by rejecting A and drawing from the residual. The mass rejected from A, $p(A) - q(A)$, is exactly B's deficit, $q(B) - p(B)$. That is why the residual must be $q - p$ and not $q$ itself: it adds back exactly what the draft under-proposed.

::worked speculative-sampling-exactness
::code lecture_10.py:L524-L532 | the two-token proof: the two routes sum to q(A) and q(B)

With numbers: $q = [0.3, 0.7]$, $p = [0.9, 0.1]$. A is proposed 90% of the time and accepted with probability $0.3/0.9 = 1/3$, so $P[A] = 0.3$. B arrives directly with 0.1, or via a rejected A with $0.9 \times 2/3 = 0.6$: $P[B] = 0.7$. Note the counter-intuitive part: the token the draft is most "confident" about is the one that gets rejected most, because over-proposed tokens are exactly the ones whose rate must be pulled down.

::predict speculative-sampling-exactness
::note skip 1:15:20 | "I'm going to skip this simple proof": the two-token argument above is in the lecture code but was not presented in class ("basically the same arguments as rejection sampling").
::kp speculative-sampling-mechanism
::kp speculative-sampling-exactness

### How many tokens to draft?

The professor showed the results of Chen et al., whose target is the 70B Chinchilla with a 4-token draft: the time per token falls substantially, at essentially the same benchmark scores.

::figure official/lectures/images/speculative-sampling-results.png | ordinary (ArS) against speculative (SpS) sampling: the scores barely move, the time per token falls
::figure official/lectures/images/speculative-sampling-stats.png | against the draft length K: sampling time drops then flattens or rises, acceptance falls steadily, the loop time grows linearly

Why there is a sweet spot was only said aloud: "if you have too few draft tokens, you're not really leveraging the batching on the target model side. And if you have too many, then you're going to reject more often." In that paper the best length was "around three or four".

The reason more drafts stop paying: draft token $i$ counts only if all $i - 1$ before it were accepted. If each is accepted independently with probability $\alpha$ (a simplifying model from Leviathan et al., not from the lecture), the expected tokens per target pass is

$$ E(k) = 1 + \alpha + \alpha^2 + \dots + \alpha^k = \frac{1 - \alpha^{k+1}}{1 - \alpha} \;\le\; \frac{1}{1-\alpha} $$

which saturates, while the draft's sequential work grows linearly in $k$. At $\alpha = 0.6$, $k = 4$ gives 2.31 tokens per pass and $k = 16$ gives 2.50: four times the draft work for 8% more tokens.

::predict spec-draft-length-tradeoff
::widget fixture:lecture_10--spec-sweet-spot | the expected-tokens bars flatten under the dashed ceiling 1/(1 − α) while the cost per pass keeps rising with k; a higher α moves the best k right, a compute-bound target (large B) moves it to k = 1

The best k is not a constant. A draft closer to the target (higher $\alpha$) keeps later tokens alive longer and moves the sweet spot up. A target run at such a large batch that its pass is compute-bound loses the "free" extra positions, and the gain fades.

### Building a good draft

In practice the target is, for example, 70B with an 8B draft, or 8B with a 1B draft. You want the draft as close to the target as possible, which means distilling it, so all of the lossy tricks above apply here too: shrink the KV cache, quantize, prune. "If you end up with a model you're happy with, just serve that. If you're not happy with it, then it at least can be a draft model", and the target fixes things up.

::code lecture_10.py:L537-L540 | draft and target sizes; distill the draft toward the target

Later methods improve the draft itself: **Medusa** lets the draft generate several tokens in parallel, and **EAGLE** feeds the draft high-level features from the target model.

::figure official/lectures/images/medusa-eagle.png | how the two extensions change what the draft sees and produces
::note skip 1:16:43 | The literature after the original papers, Medusa and EAGLE included, was skipped ("which I'll skip for now").

The lecture's summary: exact sampling from the target "thanks to math"; it exploits the asymmetry between checking and generation; and there is much room for innovation on the draft model, which involves training.
::kp spec-draft-length-tradeoff

## How do you batch requests that arrive at random? Continuous batching {#continuous-batching}
source: lecture_10.py:L46-L52, L553-L573 · video 1:16:55-1:19:34

The last part is about live traffic: a website where users come and chat. Batching sequences there is tricky for three reasons:
1. requests **arrive at different times**, and waiting to fill a batch is bad for the early ones;
2. sequences have **shared prefixes**, such as a common system prompt, or several samples drawn for one prompt;
3. sequences have **different lengths**, and padding them is inefficient.

It is "far from this very simple training where you have these blocks of the same number of tokens all at once". Training gets a dense block of batch × sequence length; inference gets a ragged array, with requests arriving and finishing at different times. And recall why batching matters at all: the decode MLP's intensity is B, the number of sequences decoded together.

::code lecture_10.py:L46-L50 | the three reasons live batching is hard
::figure https://images.ctfassets.net/xjan103pcp94/1LJioEsEdQQpDCxYNWirU6/82b9fbfc5b78b10c1d4508b60e72fdcf/cb_02_diagram-static-batching.png | static batching: slots whose request has ended sit idle until the longest one finishes

### Iteration-level scheduling

The system **Orca** ([Yu et al., OSDI 2022](https://www.usenix.org/system/files/osdi22-yu.pdf)) introduced the fix, now called continuous batching. Schedule at the level of a single decode step, not a whole request:
- every step, decode one token for every sequence in the batch;
- a sequence that ends is ejected immediately;
- a new request joins the batch as soon as it arrives, without waiting for the others to finish.

The batch is "dynamically being updated with either old finished sequences being evicted and new ones coming in".

::code lecture_10.py:L561-L563 | decode step by step; add new requests as they arrive
::animation fixture:lecture_10--continuous-batching | under static batching the slots of B and C sit idle until A ends while D, E, F wait; iteration-level scheduling refills each slot the next step, so the same six requests finish in 13 steps instead of 18
::predict continuous-batching

For a sense of the waste: a static batch of 4 requests generating 50, 100, 150 and 400 tokens runs 400 steps, but only $50 + 100 + 150 + 400 = 700$ of its $4 \times 400 = 1600$ slot-steps do work, 44% occupancy.

### Selective batching

The second problem: batching only works when all sequences have the same shape (or so it seems), and each request has a different length. Suppose three sequences of 3, 9 and 5 tokens, with hidden size H.
- **Attention** depends on each sequence's own length: a $3 \times 3$ and a $9 \times 9$ score matrix cannot share one tensor. So attention is run for each sequence separately.
- **Everything else**, the MLP layers ("which takes up a lot of FLOPs") and the projections, works token by token. So concatenate all the sequences into one "mega sequence" of shape $[3 + 9 + 5, H] = [17, H]$ and process it at once, with no padding.

::code lecture_10.py:L569-L573 | attention per sequence; the rest on the concatenation [3 + 9 + 5, H]
::widget fixture:lecture_10--batch-schedule | switch static to iteration-level: freed slots fill at the next step and occupancy rises; the selective-batching panel concatenates the sequences to [Σ tokens, H] and runs attention once per sequence

::note aside | How a new request's prefill is mixed into the ongoing decode steps (Orca runs it in the same iteration; later systems split long prefills into chunks) is not discussed in the lecture.
::kp continuous-batching

## Where does the KV cache live? PagedAttention {#paged}
source: lecture_10.py:L576-L607 · video 1:19:34-1:23:36

The final idea comes from the paper that introduced vLLM ([Kwon et al. 2023](https://arxiv.org/pdf/2309.06180.pdf)). The question is how the KV cache of many requests is laid out in memory.

### The problem: fragmentation

The previous status quo: when a request comes in, reserve one contiguous section of KV-cache memory for its prompt and response, up to a maximum length, because you do not know when it will stop. This suffers from fragmentation, "what happens to your hard drive" (back when you had to defragment one):
- **internal fragmentation**: with a limit of, say, 1,024 tokens, the whole reservation is held while the request runs, though it may generate far fewer tokens;
- **external fragmentation**: gaps between reserved sections that are too small to use.

::figure official/lectures/images/paged-attention-fragmentation.png | two requests in contiguous slabs: reserved slots never used inside each (internal fragmentation) and a gap between them (external)

Wasted memory is not harmless: it caps how many sequences fit, and so the batch size, and so the throughput.

### The solution: pages

"These are systems people, so they know their operating systems": the same problem was solved once before, with virtual memory. **PagedAttention** divides each sequence's KV cache into fixed-size, non-contiguous **blocks**. A table maps each sequence's logical blocks to physical blocks anywhere in memory; a block is allocated only when the tokens to fill it arrive. Waste is then at most the unfilled part of each sequence's last block.

::figure official/lectures/images/paged-attention-blocks.png | one sequence's cache cut into blocks of 4 tokens, scattered in memory
::figure official/lectures/images/paged-attention-logical.png | two requests: contiguous logical blocks, mapped through a block table to physical blocks interspersed in memory

For example, a request with a 37-token prompt that generates 60 tokens, under a 2048-token reservation, wastes $2048 - 97 = 1951$ slots. With 16-token blocks it occupies 7 blocks, 112 slots, and wastes 15.

::predict paged-attention
::widget fixture:lecture_10--paged-blocks | the slab's grey tail is the whole unused reservation; with blocks the waste is under one block per sample, and with sharing on every sample points to the same prompt blocks

### Sharing prefixes

Blocks also make sharing easy, which is the second of the live-traffic problems. Two cases are common:
- **a shared system prompt**: its cache is computed and stored once and used by every request, which saves compute as well as memory;
- **several samples per prompt** (for example, program synthesis that draws many candidate programs): all samples share the prompt's blocks.

::figure official/lectures/images/paged-attention-sharing.png | two requests whose prompts begin with the same few-shot prefix; only the task input differs

Sharing is kept safe with **copy-on-write** at the block level. Samples point to the same physical block until one of them has to write into it; then that block is copied for the writer. In the lecture's example, two samples continue a shared prefix: if they sample the same next token they keep sharing; when they differ, the block is split and each continues on its own copy.

::figure official/lectures/images/paged-attention-parallel.png | samples A1 and A2 share physical block 7 ("Four score and seven"); when they write different tokens ("fathers", "mothers") into the shared block 1, it is copied on write and its reference count drops from 2 to 1
::code lecture_10.py:L594-L599 | share system prompts and samples; copy-on-write at the block level

vLLM adds other optimizations: a kernel that fuses the block reads with attention (fewer kernel launches), the latest attention kernels (FlashAttention, FlashDecoding), and CUDA graphs to avoid launch overhead. The section's summary: use ideas from operating systems, paging, to make good use of memory under dynamic workloads.

::note skip 1:23:28 | "A bunch of other optimizations like kernels that I'm not going to have time to go over": the fused block-read kernel, FlashDecoding and CUDA graphs are only listed.
::kp paged-attention

## What should you carry away? {#summary}
source: lecture_10.py:L55-L60 · video 1:23:36-1:25:24

- **Inference matters**: actual use, evaluation and reinforcement learning, a repeated cost where training is a one-time one.
- **It behaves differently from training**: generation is memory-bound (decode attention has intensity below 1, the decode MLP only B) and the workload is dynamic. Its napkin math is bytes over bandwidth: about 8 ms per token for Llama 2 13B on one H100 at B = 1.
- **Batch size trades latency for throughput**; reducing memory improves both.
- **Techniques**: new architectures (GQA, MLA, CLA, local attention), quantization, pruning and distillation, speculative sampling. Almost all of them follow one principle: "reduce your KV cache but don't hurt accuracy too much".
- **Ideas from systems**: speculative execution (speculative sampling) and paging (PagedAttention, with continuous batching to keep batches full).
- **New architectures have huge potential.** The KV cache and the way attention is built make the Transformer "an inference unfriendly kind of architecture"; state-space models, linear attention or diffusion could unlock a lot.

::code lecture_10.py:L55-L60 | the lecture's own summary
::note deferred 1:24:40 | Inference-friendly architectures were "discussed briefly" but not covered; linear attention and state-space models are in the [L4 thread](#/t/lecture_04). Next lecture: scaling laws, part 2.
