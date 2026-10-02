---
title: L4 · Attention alternatives and mixtures of experts, read through
minutes: 45
---
This lecture covers two upgrades to the basic transformer of [L3](#/read/lecture_03). The first replaces or thins out attention so that cost grows linearly, not quadratically, with context length: linear attention, its gated descendants (Mamba-2, Gated DeltaNet), the hybrids that ship them, and DeepSeek's sparse attention. The second replaces the feed-forward block with a mixture of experts (MoE), which adds parameters without adding FLOPs per token. After reading it you can count what each design costs, say why every production "linear" model is still a hybrid, and explain how a router that cannot be differentiated still gets trained.

## Why does attention become the bottleneck? {#why}
source: lecture_04.pdf p2-p3 · video 0:05-5:58

::slide 2 | left: context windows of released models on a log scale, from GPT-1's few hundred tokens to the first 1M-token window (Gemini 1.5) and the first 10M window (Llama 4 Scout); right: milliseconds per step against sequence length, split into feed-forward (blue) and attention (orange)

Vendors are racing to longer contexts, for agents and long documents. The right-hand plot shows why that hurts. The feed-forward cost grows linearly with sequence length; attention, where every position interacts with every other, grows quadratically. At short lengths the feed-forward part dominates. By 16K tokens attention is roughly 450 of the 610 ms, about three quarters of the step.

There are really two costs here, and the rest of the attention half attacks both.
- **Compute.** Scoring $n$ queries against $n$ keys is $n^2$ dot products.
- **Memory at decode time.** To generate token $t$, softmax attention needs the key and value of every earlier token, in every layer. These are kept in the **KV cache**, which grows by a fixed amount with every token.

The lecture assumes you know how big that cache gets. Here is the count, from [L10's KV cache](#/read/lecture_10) and [L3's GQA section](#/read/lecture_03). Per token you store one key and one value of `head_dim` numbers, for every KV head of every layer:

$$ \text{KV bytes per token} = 2 \cdot n_\text{layers} \cdot n_\text{kv heads} \cdot d_\text{head} \cdot \text{bytes per value} $$

::worked supp-kv-cache-bytes
::predict supp-kv-cache-bytes
::widget fixture:lecture_04--kv-cache-model | start from the Llama-2-13B defaults and move only the context slider: bytes per token never change, but the total doubles with every doubling of context
::kp supp-kv-cache-bytes

::slide 3 | top left: attention masks for a full transformer and for strided and fixed sparse transformers; bottom left: a stack of sliding-window (local) layers with a global-attention layer every few layers; right: attention forward+backward TFLOP/s on an A100, PyTorch at 36-46 against FlashAttention-2 at 132-176, with PyTorch out of memory at 16K

The "basic toolkit" has two tools, both covered elsewhere in the course.

**Mix local and global attention.** Most layers attend only within a sliding window, and a full global layer appears every so often. "If you're only doing global attention once every eight layers", the professor says, "you've very much controlled the cost." Information still travels the whole context, because each local layer extends the reach of the one before. This is [L3's sliding-window interleave](#/read/lecture_03).

::animation fixture:lecture_03--swa-reach | each local layer extends the reach by one window, and the single full-attention layer jumps it to the whole context

**Systems engineering.** The right panel is FlashAttention (the [L5 read-through](#/read/lecture_05) takes it apart). Plain PyTorch attention runs at 36-46 TFLOP/s and runs out of memory at 16K tokens. FlashAttention-2 runs the same arithmetic at 132-176 TFLOP/s, about 4× faster, and handles 16K, because it never materializes the $n \times n$ attention matrix. The professor's lesson: "This doesn't fix any of the quadratic cost issues, but constant factors are very, very powerful." Big-O is not the whole story, a theme that returns with DSA.

::note deferred 3:48 | How FlashAttention works is left to the systems lectures and the systems assignment. Here only its constant-factor speedup matters. (He says "factors of 2" aloud; on the slide's chart FlashAttention-1 is about 2.5× and FlashAttention-2 about 4× over PyTorch.)

The banner asks the real question: at 5 or 10 million tokens these tricks are not enough. This is the first year the professor teaches linear-time attention, because in the last two years a set of recipes has become "battle-tested at scale". All of them grow from one idea, the associativity of matrix multiplication.

::video 2:05-2:41 | the cost plot read aloud: the feed-forward cost grows linearly, attention quadratically, and attention "quickly outpaces" it

## Linear attention: what if we just move the parentheses? {#linear}
source: lecture_04.pdf p4 · video 5:58-8:04, 21:17-21:36, 29:57-30:23

::slide 4 | attention written as Attn(Q, K, V) = ρ(QKᵀ)V with Q, K of shape n × d_k and V of shape n × d_v; the highlighted identity (QKᵀ)V = Q(KᵀV); the cost falls from n²d_k + n²d_v to 2nd_vd_k

Write attention for one head compactly. $Q$ and $K$ are $n \times d_k$ (one row per position), $V$ is $n \times d_v$, and $\rho$ is the row-wise softmax:

$$ \mathrm{Attn}(Q, K, V) = \rho(QK^\top)\,V $$

Where does the quadratic come from? $QK^\top$ is an $n \times n$ matrix, each entry a $d_k$-long dot product, so it costs $n^2 d_k$ multiply-adds. Multiplying that $n \times n$ matrix by $V$ costs another $n^2 d_v$. Both terms have $n^2$.

Now, the professor says, "let's forget that attention has a softmax". If $\rho$ were the identity, the expression would be a plain product of three matrices, and matrix products are associative, so we may group them the other way:

$$ (QK^\top)V = Q\,(K^\top V) $$

$K^\top V$ is $(d_k \times n)(n \times d_v)$, a small $d_k \times d_v$ matrix, costing $n d_k d_v$. Multiplying $Q$ by it costs another $n d_k d_v$. The total is $2 n d_k d_v$, and no $n \times n$ object ever appears. The slide calls this "very silly, but surprisingly important".

Why is that a better bargain? Because $n$ is the context length, which can be in the millions, while $d_k$ and $d_v$ are head widths, in the hundreds; even full hidden sizes are "on the order of thousands, tens of thousands. No one has a million coordinates in their hidden dimensions." Trading an $n^2$ for a $d^2$ is a large win exactly when contexts are long.

::worked linear-attention-reassociation

**Worked example.** One head with $d_k = d_v = 128$ and $n = 8192$. Softmax attention costs $8192^2 \cdot 256 \approx 1.7 \times 10^{10}$ multiply-adds; the reordered form costs $2 \cdot 8192 \cdot 128^2 \approx 2.7 \times 10^8$. The ratio is $n^2(2d) / (2nd^2) = n/d = 64$. The crossover is at $n = d$: for contexts shorter than 128 tokens the quadratic form is actually cheaper.

::widget fixture:lecture_04--attention-cost | slide the context n: the softmax and linear FLOP curves cross at n = d (128), and past it the gap grows as n/d, 64× at 8,192 tokens

The catch is in the word *if*. With the softmax in place the reordering is not allowed: each row of $\rho(QK^\top)$ is divided by a sum over all $n$ scores in that row, so the $n \times n$ matrix must exist before $V$ is applied. Dropping $\rho$ changes the *model*, not just the algorithm: linear attention is a different layer, not a faster softmax attention. The professor makes this explicit in a later answer: "The first step is lossy, where we drop the row, and then we become linear." (The captions write ρ, rho, as "row".) The "kernel version" cited on the slide (Katharopoulos et al. 2020) replaces $\rho$ by a feature map $\phi$ applied to queries and keys, $\phi(Q)(\phi(K)^\top V)$, which keeps the associativity trick while making the scores nonnegative.

::predict linear-attention-reassociation

Two points that the slide leaves out. First, a language model uses a **causal mask**: token $t$ may only read tokens $j \le t$. Then there is no single $K^\top V$ for the whole sequence; each position needs the sum over its own prefix. That is precisely what turns linear attention into a recurrence, the subject of the next slide. Second, a student asked whether removing the softmax makes training unstable. The professor did not think so: "If anything, softmaxes are usually more of the problems". He expects linear layers to *improve* stability.

::video 6:52-7:44 | moving the parentheses, and why n·d_v·d_k is the better term: n is millions, d is thousands
::kp linear-attention-reassociation

## Linear attention is an RNN, and that is the point {#recurrent}
source: lecture_04.pdf p5-p6 · video 8:04-11:11, 20:41-21:47

::slide 5 | the same reordering, then the recurrence S_t = S_{t−1} + k_t v_tᵀ and y_t = q_tᵀ S_t; "This 'duality' allows us to train efficiently (using the parallel, quadratic form) and inference efficiently (using the serial, linear form)"; a footnote: weighting S_{t−1} by γ gives RetNet

Apply the causal mask to the reordered product. Row $t$ of the output reads only keys and values up to $t$, so

$$ y_t = q_t^\top \sum_{j \le t} k_j v_j^\top . $$

Call the prefix sum $S_t$. A prefix sum grows by its newest term, so it can be updated one token at a time:

$$ S_t = S_{t-1} + k_t v_t^\top, \qquad y_t = q_t^\top S_t . $$

Each $k_t v_t^\top$ is an outer product, a $d_k \times d_v$ matrix, so the **state** $S_t$ is one $d_k \times d_v$ matrix per head, the same size whatever $t$ is. "As I sweep left to right on my context", the professor narrates, "I can multiply my Ks and Vs, and I can update this thing S, which is a state." That is exactly the shape of a recurrent neural network: a fixed-size state carried forward and read at every step.

::worked linear-attention-recurrent-duality

Why is this "even nicer" than the linear cost? RNNs are "very nice for inference reasons": the state is fixed in size, so generating token one million costs the same as generating token ten. They are "not nice for training reasons": step $t$ waits for step $t-1$, so a sequence cannot be processed in parallel. Linear attention has both forms of one function. Training computes all positions at once as matrix multiplies; the slide calls this the "parallel, quadratic form" because, with the causal mask, it multiplies by the masked $n \times n$ matrix $QK^\top$, quadratic but pure matmul work (practical kernels process the sequence in chunks). Inference runs the serial form and carries one small $S$.

The two forms compute the same outputs exactly. A student asked: if the forms are equivalent, why do models with more recurrent layers lose quality? Because the loss happened earlier. The professor: dropping $\rho$ is lossy, "and then after that, this linear form to this recurrent form, this equivalence, that is exact."

::predict linear-attention-recurrent-duality

**The state against the cache.** For one head with $d_k = d_v = 128$, the state holds $128 \times 128 = 16{,}384$ numbers. The KV cache for the same head holds $128 + 128 = 256$ numbers per token. They are equal at $16{,}384 / 256 = 64$ tokens. At $2^{20}$ (about a million) tokens the cache is $2^{20} \cdot 256 / 16{,}384 = 16{,}384\times$ bigger than the state. The fixed state is what makes long-context decoding cheap, and, as we will see, also what limits it.

::widget fixture:lecture_04--attention-cost | in the memory panel the green linear-state line stays flat at d_k × d_v while the KV-cache line climbs with every token; they cross at 64 tokens

The footnote on the slide is the first elaboration: multiply the old state by a constant $\gamma < 1$ before adding, $S_t = \gamma S_{t-1} + k_t v_t^\top$, and you get **RetNet**. Old key-value pairs now fade geometrically: a pair written $a$ steps ago carries weight $\gamma^a$. With $\gamma = 0.95$, a pair 20 steps old keeps $0.95^{20} \approx 0.36$ of its weight.

::widget fixture:lecture_04--state-memory | untick the same-key write and lower γ: the step-0 value fades as γ^age along the green curve, while plain linear attention (grey) never forgets it

::note slip 9:41 | Right after praising the linear form, the professor says "Unfortunately, this is linear, which is not very good." The slide says "linear time (great)". He most likely means that plain linear attention is too simple to be expressive enough, which is how the next part opens ("linear attention is too naive and simple for me").
::kp linear-attention-recurrent-duality

::slide 6 | MiniMax-M1: benchmark bars against o3, Gemini 2.5 Pro, Claude 4 Opus, DeepSeek-R1 and Qwen3-235B; inference FLOPs against generation length (MiniMax-M1 grows far more slowly than DeepSeek-R1); the block diagram, M lightning-attention blocks per softmax-attention block, each followed by an MoE layer; a small table where the hybrid matches or beats softmax attention

Does something this simple work at scale? **MiniMax-M1** (and MiniMax-Text-01), a large Chinese open model, uses a **7-to-1 hybrid**: seven linear-attention layers ("lightning attention") for each full softmax-attention layer. Its benchmark scores are competitive with o3 and DeepSeek-R1, for example 86.0 on AIME 2024. The FLOPs curve shows the payoff: at 128K generated tokens, DeepSeek-R1 needs about $7 \times 10^{16}$ FLOPs and MiniMax-M1 under $2 \times 10^{16}$.

It is "not fully linear, because you've got these softmaxes, but the dependence is much, much milder." That qualification is the rule for everything in this half of the lecture: "No one has, thus far, really proven out fully linear time attention mechanisms at scale. Everything that I'm going to talk about in the next couple of slides is a hybrid."

::predict hybrid-attention-ratios

## Mamba-2: add a forget gate, keep the duality {#mamba2}
source: lecture_04.pdf p7-p8 · video 11:11-14:56, 21:47-22:26, 33:01-33:35

::slide 7 | the linear-attention recurrence, then Mamba-2's: S_t = γ_t S_{t−1} + k_t v_tᵀ and y_t = q_tᵀ S_t + v_tᵀD with γ_t = f(x_t), the new terms in red; "we can make linear attention more expressive via gating (gating is good!)"; "This continues to have duality properties (compute γ in parallel, apply duality)"; Mamba block diagrams below

Plain linear attention always carries everything forward: every pair is added with weight 1 and never leaves. We know from LSTMs, the professor says, that it matters "when to pass information forward and when to just not pass information forward, to forget things and send them to 0." So add a gate. **Mamba-2** (from the state-space-model family of Albert Gu, Tri Dao and colleagues) multiplies the old state by $\gamma_t$ before writing:

$$ S_t = \gamma_t\, S_{t-1} + k_t v_t^\top, \qquad y_t = q_t^\top S_t + v_t^\top D, \qquad \gamma_t = f(x_t) $$

Unlike RetNet's constant $\gamma$, here $\gamma_t$ is computed from the current token, so the model decides per token how much of its memory to keep. Mamba-2 was derived from state-space theory, and the slide declines to repeat the justification ("go read the mamba 2 paper"), but mechanically it is this small elaboration of linear attention.

The extra term $v_t^\top D$ does not touch the state. The professor first set it aside ("you can ignore the vt for now") and explained it in Q&A: it acts "a little bit like a residual connection", letting the current token's value pass straight to the output, and "D is another modulated gate" that sets how much passes through.

### Why the gate must read only the input

The crucial property is in the last line of the slide. "Gamma of t is not stateful. Gamma of t only depends on my current inputs x of t." That is what keeps the parallel form. Unroll the recurrence and each pair is weighted by the product of the gates applied after it:

$$ S_t = \sum_{j \le t} \Big(\prod_{i=j+1}^{t} \gamma_i\Big)\, k_j v_j^\top $$

If every $\gamma_i$ comes from $x_i$ alone, all of them can be computed first, in parallel, and the products become a decay-weighted causal mask on $QK^\top$. Training is again one big masked matrix multiply; inference is again the small recurrence. The professor states the general rule of thumb: as long as the terms in your RNN are "only input-dependent terms, so no state dependence", you keep the duality between parallel operations for training and serial ones for inference.

An LSTM fails this test. Its gates read the previous hidden state $h_{t-1}$, so step $t$'s gate cannot be computed until step $t-1$ is finished. That is why LSTMs had to be trained step by step. Asked later what state-space models give up compared with transformers, the professor put the history this way: attention used to have "the very strong advantage of hardware efficiency", and state-space models caught on "despite their similarity to LSTMs" because the duality hands that efficiency back.

::predict gate-input-only-duality
::video 14:31-14:57 | the rule of thumb: "only input-dependent terms, so no state dependence" keeps the parallel/serial duality
::kp gate-input-only-duality

::slide 8 | Nemotron-3-Nano-30B-A3B's layer pattern: blocks of Mamba-2 and MoE layers with an occasional attention layer (×5, ×3, ×1, ×4); benchmark bars against Qwen3-30B-A3B-Thinking and GPT-OSS-20B; relative output throughput on the right

NVIDIA's **Nemotron 3** uses Mamba-2 as its cheap layer, with "your big softmax attention every now and then". Count the layers in the figure. The first block (×5) has three Mamba-2 layers and one attention layer, the third block (×1) has one of each, and the other blocks are Mamba-2 only:
- Mamba-2 layers: $5 \cdot 3 + 3 + 1 + 4 = 23$;
- attention layers: $5 + 1 = 6$.

That is about 3.8 Mamba layers per attention layer, the slide's "3-1 ish" (each followed by an MoE layer, the second half of this lecture). The model is competitive with Qwen3-30B-A3B and GPT-OSS-20B, at a relative output throughput of about 3.3 against their 1.0 and 1.5. These are "small frontier models", but they show the recipe working.

## Gated DeltaNet: gate the write, erase before you overwrite {#deltanet}
source: lecture_04.pdf p9-p11 · video 14:56-20:41, 22:26-22:54

::slide 9 | Mamba-2's update again, then Gated DeltaNet's: S_t = γ_t (I − β_t k_t k_tᵀ) S_{t−1} + β_t k_t v_tᵀ and y_t = q_tᵀ S_t, with γ_t and β_t both functions of x_t; it adds a "no input operation" gate (β = 0) and erases anything in the direction of the current key

**Gated DeltaNet** is, in the professor's estimate, probably the most widely used state-space-style layer today. It makes two changes to Mamba-2, both functions of the input only, so the duality survives:

$$ S_t = \gamma_t\,(I - \beta_t k_t k_t^\top)\,S_{t-1} + \beta_t\, k_t v_t^\top, \qquad \gamma_t, \beta_t = f(x_t) $$

**An input gate $\beta_t$.** It scales the write. "If beta t is 0, that basically means, don't take any of my current information, don't add it into my state." Together with the forget gate $\gamma_t$, this is very reminiscent of an LSTM: one gate decides what to forget, one decides what to let in.

**An erase term $(I - \beta_t k_t k_t^\top)$.** This comes from DeltaNet. When you write a new value under key $k_t$, you would like to also "erase any previous keys that have gone into it", rather than piling the new value on top of the old one. The matrix $I - k_t k_t^\top$ projects out the $k_t$ direction, so it removes whatever the state stores along the current key and leaves every other direction alone. The professor's caveat: "That's not exactly right, because you're not doing things like unit normalization." The projector reading holds exactly when $\|k_t\| = 1$.

::worked gated-linear-attention-variants

**Worked example.** Take unit-norm keys, $\gamma_t = 1$, a value 4 stored earlier under key $k$, and a new token that writes value 10 under the same key. Read the state with $q = k$:
- plain linear attention adds, and returns $4 + 10 = 14$, a blend of old and new;
- Gated DeltaNet with $\beta = 1$ erases, then writes, and returns 10: the old value is *replaced*;
- with $\beta = 0.5$ it returns $0.5 \cdot 4 + 0.5 \cdot 10 = 7$;
- with $\beta = 0$ nothing is written or erased, and it still returns 4.

Expanding the update shows where the name comes from: $S_t = S_{t-1} + \beta_t k_t\,(v_t - S_{t-1}^\top k_t)^\top$ (with $\gamma_t = 1$). The state is corrected by the error between the value you want under $k_t$ and the value it currently returns, which is the classic delta rule of online least squares.

::widget fixture:lecture_04--state-memory | tick the same-key write and move β: at β = 1 Gated DeltaNet's read jumps to the new value, at β = 0 it keeps the old one, while plain linear attention always adds the two
::predict gated-linear-attention-variants

::note aside 17:46 | The slide's "close relationships to fast weight programming / test time training" is said aloud as a recurring rediscovery: this projector update appears if you solve certain meta-learning least-squares problems, and researchers in fast-weight programming and test-time training reached "the exact same kinds of solutions" from very different design principles.

Stepping back after the Q&A, the professor observes that the methods that survived testing "converged to almost very LSTM-like objects": gated, linear recurrences, now trainable in parallel.

::kp gated-linear-attention-variants

::slide 10 | Qwen3.5 / Qwen3-Next: the block diagram, three (Gated DeltaNet → MoE) layers per one (Gated Attention → MoE) layer; benchmark bars; decode throughput against context length, normalized to Qwen3-32B, with Qwen3-Next-80B-A3B rising from about 3× at 4K to about 11× at 128K while Qwen3-30B-A3B stays near 3×

The newest Qwen models, among the best open models available, use exactly this layer in a **3-to-1 Gated DeltaNet/attention hybrid**. The benchmark panel shows little or no loss against the previous generation and closed models. The throughput panel shows the gain growing with context, because only a quarter of the layers keep a KV cache and pay the quadratic cost. Qwen3-30B-A3B, an ordinary transformer MoE, is already about 3× faster than the dense 32B at every length (it activates about 3B parameters); the hybrid's advantage keeps widening, to about 11× at 128K.

::slide 11 | ByteDance Seed and UC Santa Cruz, "A Systematic Analysis of Hybrid Linear Attention": a table of update rules (RetNet, GLA, Mamba-2, RWKV-6, DeltaNet, Gated DeltaNet, …); RULER sub-task accuracy and average recall against the linear:full ratio (3-1, 6-1, 12-1, 24-1, pure), with a dashed full-transformer baseline

How many softmax layers can you remove? Controlled studies are rare; the professor points to this one. Read the plots from left to right, adding more linear layers per full layer:
- at low ratios (3-to-1, 6-to-1) the best variants, the Gated DeltaNet family among them, sit at the dashed transformer line: "basically no hit";
- past some point, long-context performance degrades more noticeably;
- fully recurrent ("pure") is clearly worse in every architecture; in the recall panel some variants fall from about 0.43 to 0.15–0.2.

Be careful which task you look at. Single-key retrieval is something "all of these long context architectures explicitly optimize for", so it flatters them. Question answering declines steadily as the ratio grows.

**What the hybrid saves.** In a 7-to-1 model only one layer in eight keeps a KV cache, so the cache is one eighth of an all-softmax model's at every context length; in a 3-to-1 model, one quarter. It still grows linearly with context. The hybrid lowers the constant, it does not remove the growing part.

::widget fixture:lecture_04--kv-cache-model | set the linear:softmax ratio to 7-to-1 and then 3-to-1: only the blue softmax layers keep a cache, so the total is 1/8 and then 1/4 of the all-softmax bar, and it still grows with context
::video 19:41-20:31 | no loss at low hybrid ratios, degradation past a point, worst at pure RNN; single-key retrieval flatters these models
::kp hybrid-attention-ratios

## Sparse attention: keep softmax, read fewer tokens {#dsa}
source: lecture_04.pdf p12-p13 · video 22:54-34:22

::slide 12 | the DSA prototype from DeepSeek-V3.2: a lightning indexer scores each preceding token, I_{t,s} = Σ_j w_{t,j} · ReLU(q_{t,j} · k_s), over a few indexer heads in FP8; the top-k scores select which key-value entries the query attends to; "Can be 'post hoc' adapted after dense short context pretraining"

A completely different answer to the cost of attention: do not replace softmax attention, *sparsify* it. In **DeepSeek Sparse Attention** (DSA), from DeepSeek-V3.2, each query first runs a cheap **lightning indexer** over the long context, picks a small subset of tokens, and then does ordinary full attention on that subset only.

The forward pass, as the professor walks through it:
1. For query token $t$ and each earlier token $s$, the indexer takes small query-key inner products $q_{t,j} \cdot k_s$ over a few indexer heads $j$, applies a ReLU, and sums them with weights $w_{t,j}$ derived from the query token. That gives an index score $I_{t,s}$.
2. A **top-k** keeps the $k$ highest-scoring positions.
3. Normal softmax attention runs between query $t$ and those $k$ key-value entries.

The surprising practical point is that you do not have to pretrain with it. Open models usually train in three stages: "short context pre-training, long context extension, then post-training." You train a normal dense transformer, and when you do the long-context extension, which you were going to do anyway, you "drop in this lightning indexer" and train the model to use it. The professor finds it "surprising that it works", given that a top-k is a "frankly scary-looking" non-differentiable object. The second half of the lecture will make top-k look less scary.

::slide 13 | DeepSeek-V3.2's benchmarks next to GPT-5-High, Claude 4.5 Sonnet and Gemini 3 Pro; cost per million tokens against token position for prefill and decode, V3.1 rising steeply and V3.2 nearly flat; GLM-5 results and a RULER table for GLM-4.7-Flash with and without DSA

The results: V3.2 roughly matches the frontier models of its time. The cost plots, read at 128K tokens, show prefill falling from about 0.65 to about 0.2 US dollars per million tokens, and decode from about 2.2 to about 0.3 dollars. GLM-5, which the professor calls one of the best open models available, adopted DSA too. Its RULER table shows what the extension stage buys. At 128K, GLM-4.7-Flash scores 79.21. Training only the indexer on top of the frozen model drops it to 71.35. Training indexer and model jointly for 150B tokens brings it back to 78.86. So little is lost relative to full attention, "even at long context retrieval tasks that are fairly difficult to do with RNN style architectures."

::predict sparse-attention-dsa
::kp sparse-attention-dsa

### DSA is not linear time

"Notice, I'll go back one slide, this is not linear time." To decide which $k$ tokens to keep, the indexer must score every query against every earlier token: "it does have to look at everything ... It is really brute force inner products." So the indexer is still $O(n^2)$. The savings are constant factors, on two fronts:
- **the indexer is made very cheap per pair**: few heads, low dimension (the q's and k's are projected down just for indexing), small weights, and low precision (FP8 on the slide);
- **the expensive attention runs on $k$ tokens, not $n$**: its total is $n \cdot k$, and $k$ is chosen near short-context lengths and bounded whatever the input length. The lecture gives no value; DeepSeek-V3.2 uses $k = 2{,}048$.

**Worked example.** Let $k = 2{,}048$ and grow the context from 64K to 128K tokens. The indexer's pairs grow as $n^2$, so its cost quadruples. The attention part grows as $n \cdot k$, so it only doubles. If one indexer pair costs 1/64 of an attention pair, the two parts cost the same at $n = 64 \cdot 2{,}048 = 131{,}072$ tokens; past that, the cheap quadratic indexer is most of the bill.

The KV cache does not shrink at all: every token's key and value must be kept so that the indexer can choose among them. DSA saves reads and FLOPs, not storage.

::widget fixture:lecture_04--attention-cost | watch the DSA lines: the dashed indexer climbs with the same n² slope as softmax, the top-k part grows like n, and past n = R·k the indexer dominates; in the memory panel DSA sits on the KV-cache line
::predict dsa-indexer-still-quadratic

The professor's moral is the same as for FlashAttention: "sometimes, don't get too stuck up on the quadratic versus not. Sometimes, the constant factors are really, really important."

::note skip 27:15 | Further DSA details, beyond the indexer-then-top-k forward pass, are skipped ("I won't talk much about this"). What he flags instead is that top-k selection will be "core to the next part of this lecture".
::kp dsa-indexer-still-quadratic

### Questions that closed the attention half

::note aside 31:54 | Is FP4 attention possible? "Definitely possible", but softmaxes are sensitive to underflow and overflow at low precision. The indexer suggests a sensible split: select in low precision, then attend in full precision so the value vectors are weighted finely.
::note aside 30:51 | Asked to predict future attention architectures, the professor expects more combining of all these tricks, plus a higher layer in which post-training teaches the model to manage its own context (compaction, retrieval).

**What do state-space layers give up?** A student asked the obvious question: if these layers are so cheap, why not use them everywhere? "The downside, I think, is expressive power. The all to all connection in softmax attention is incredibly powerful." The training-efficiency objection to recurrences has been answered by the duality. What remains is the fixed state: "if you have a finite state and you have to carry everything through, you're going to be losing some information relative to just carrying everything."

Put numbers on it with the head from before. The state holds 16,384 numbers whatever the context. At 1,024 tokens that is 16 numbers per context token; at 2,048 it is 8; at a million, 1/64 of a number. Somewhere along that line a fact stated once near the start has to be squeezed out. You could make the state as big as the context, "but then you're paying these very large costs." This is why the production models keep a softmax layer every few layers: those layers hold every token's key and value and can look a fact up directly. The professor hedges that "It might be possible one day to not have any trade-offs", but so far this is where they show up.

::predict finite-state-recall-tradeoff
::video 33:38-34:15 | the fixed state against carrying everything: "a state the size of your context" is fine, but costs as much as the context
::kp finite-state-recall-tradeoff

## Mixtures of experts: more parameters, same FLOPs {#moe}
source: lecture_04.pdf p14-p19 · video 34:22-44:26, 1:10:19-1:11:05

::slide 14 | the MoE gallery: a scaling chart with "GPT-MoE-1.8T" at the top ("GPT4 (?)"), Mistral's torrent link for Mixtral 8x22B, Grok, the DeepSeekMoE and DeepSeek-V3 papers, Llama 4, and OLMoE

The second half modifies the other big block of the transformer, the feed-forward network (FFN, or MLP). "Conceptually, they don't really change the game", the professor says: "One way of thinking about mixture of experts is, they are just a more efficient MLP." He gives two reasons to learn them anyway. Past a certain size, nearly every model you can download is an MoE. And their mechanical parts, top-k selection plus auxiliary losses to make it trainable, keep reappearing elsewhere, as they already did in DSA.

::slide 15 | Switch Transformer's figure: a dense block with one FFN, against a sparse block where a router sends the token "The" to FFN 2 and "Dog" to FFN 1 out of four; "Replace big feedforward with (many) big feedforward networks and a selector layer"; "You can increase the # experts without affecting FLOPs"

The mental model the professor wants you to keep: take the FFN, make four copies of it, each the *same size* as the original, and add a small **router** that picks one of them for every input. You now hold "4x the parameters", but "on any forward or backward pass, I'm only going to pay one FFN worth of cost". Each copy is an **expert**. The parameters a token actually passes through are the model's **active parameters**; everything else is stored but idle for that token.

In general, with $E$ experts of equal size, each token routed to $k$ of them:
- FFN parameters grow with $E$;
- FFN FLOPs per token grow with $k$ (times the expert size), and not with $E$ at all.

**Worked example.** An MoE layer with 8 full-size experts and top-2 routing runs 2 FFNs' worth of compute per token and stores 8. Raise the count to 64 experts, still top-2: FLOPs per token are unchanged (factor 1), parameters go up 8×. The router itself is one small matrix multiply and is negligible.

::predict moe-params-without-flops
::widget fixture:lecture_04--moe-ffn-budget | drag the routed-expert count E: only the parameters bar moves; top-k, expert size and shared experts are the only controls that move the FLOPs bar

### Routing is per token, and experts are not subject experts

Two student questions sharpen the picture. What is the granularity of routing? "At the token level. So every token gets an expert." And the router is "super naive": "a single matrix multiply between your input" and one vector per expert. Nothing in it knows that a question is about medicine; it sees one token's hidden state, and might decide "this is a token that looks like it's in, I don't know, Japanese. Let's route it to expert 7."

So the tokens of one sequence spread over many experts. Later in the lecture someone asked whether experts end up specializing in domains. Papers that visualize the assignments find shallow, lexical patterns: punctuation goes to one expert, other symbols to another, non-English character sets to another. "They're not like medical experts, legal experts ... There's no semantics." The name "mixture of experts" invites the wrong picture.

::predict expert-routing-token-level
::video 1:10:31-1:11:05 | what experts actually pick up: punctuation, symbols, scripts, "no semantics"
::kp expert-routing-token-level

### Why MoEs took over

::slide 16 | Switch Transformer: test loss falls from about 6.0 with 1 expert to about 4.85 with 256 experts as sparse parameters grow at fixed FLOPs; right: negative log perplexity over training for Switch-Base with 16 to 128 experts, all well above the dense T5-Base

"Same FLOP, more param does better." In the left plot every model has the same active compute; only the number of experts changes. Test loss keeps falling, from about 6.0 at one expert to about 4.85 at 256. "If you keep the total compute the same, but you just increase the number of sparse parameters, somehow, the models are generally getting better." The right plot shows the same ordering throughout training.

::slide 17 | left: Switch-Base reaches T5-Base's final quality about 7× sooner; right: OLMoE (1.3B active, 6.9B total) against a 1.3B dense model on 128 H100s, reaching the same HellaSwag score with about 3× fewer tokens and about 2× less training time

They are also faster to train. Switch-Base reaches the dense T5-Base's final perplexity about 7× sooner. OLMoE trained a 1.3B-active, 6.9B-total MoE against a 1.3B dense model: the MoE matches the dense model's HellaSwag with about 3× fewer FLOPs or tokens, and about 2× less wall-clock time. The professor summarizes: "something like two times faster".

::slide 18 | MMLU against activated parameters for many open models; DeepSeek-V2 (a red star at about 21B activated, MMLU about 78.5) sits near LLaMA 3 70B and above Mixtral 8x22B, DBRX and Qwen1.5 72B

What matters for serving cost is the active parameter count, and here the MoEs dominate. DeepSeek-V2, with about 21B active parameters, scores about 78.5 on MMLU, close to LLaMA 3 70B (79.5) at less than a third of the active compute. This plot, the professor says, was "really a big shift" when the early DeepSeek MoEs came out.

::slide 19 | Switch Transformer's MoE encoder with device placement: attention replicated on every device, the MoE layer split so that FFN_1 … FFN_E live on devices 1 … E, connected by an all-to-all dispatch before the experts and an all-to-all combine after

The last reason is systems. Each expert is "a natural chunk" of the model: put different experts on different devices and send each token's activation to the device that holds its expert. This is **expert parallelism**, an extra axis for cutting up a model too big for one device.

::note deferred 39:58 | Parallelism in general comes "in three more lectures" ([L7](#/read/lecture_07), [L8](#/read/lecture_08)); expert parallelism returns at the end of this lecture.

Two questions probed the cost. Isn't shipping activations a bottleneck? Yes: "You're getting more aggregate flops. You're reducing your memory use. And in exchange, you're going to pay for comms." Whether that is a net win "is highly dependent on your topology". Is there an upper limit to parallelizing? Yes, "as you shard over more and more devices, the communication cost explodes."

::note deferred 44:09 | Choosing a sharding for a given network topology, and the limits of parallelizing MoEs, are left to the assignment.
::kp moe-params-without-flops

## Who ships MoEs, and why it took until 2024 {#landscape}
source: lecture_04.pdf p20-p25 · video 40:42-41:54, 44:26-47:23

::slide 20 | Llama 4 Maverick's table against Gemini 2.0 Flash, DeepSeek v3.1 and GPT-4o, with inference cost per million tokens of 0.19–0.49 against 4.38 for GPT-4o; GPT-OSS on Humanity's Last Exam; "MoEs are most of the highest-performance open models, and are quite quick"

The big open MoEs from the West are Llama 4 and OpenAI's GPT-OSS, both "top tier models in their own right". Maverick's table makes the economic case: comparable scores to GPT-4o at a tenth of the quoted price per token.

::note aside 40:55 | "Unfortunately, I guess, open source model releases in the West have stalled, for the most part." Much of the MoE research and training action, he says, has moved to Chinese labs (Qwen, DeepSeek, MiniCPM and others).

::slide 21 | Qwen1.5-MoE-A2.7B (14.3B total, 2.7B activated) against Mistral-7B, Gemma-7B, Qwen1.5-7B and DeepSeekMoE 16B on MMLU, GSM8K, HumanEval, multilingual and MT-Bench

Chinese labs also did a lot of MoE work "on the smaller end". Qwen1.5-MoE-A2.7B activates 2.7B of its 14.3B parameters. Against 7B dense models it is in the same range: MMLU 62.5 against 61.0 (Qwen1.5-7B), 64.1 (Mistral-7B) and 64.6 (Gemma-7B), and GSM8K 61.5 against 62.5, 47.5 and 50.9. The professor's "doing better than many of the 7B models" is generous on MMLU; the fair summary is that it matches 7B dense models with about a third of their active parameters.

::slide 22 | DeepSeekMoE's controlled comparison at 0.2B activated parameters and 2.9T FLOPs per 2K tokens, trained on 100B tokens: a dense model (0.2B total), a hash-routed MoE and a Switch-routed MoE (2.0B total each); Pile loss 2.060, 1.932 and 1.881, with the MoEs ahead on almost every benchmark

This is the kind of ablation the professor admires DeepSeek for. All three models have the same active parameters and the same FLOPs; the two MoEs hold 10× the parameters. Even routing by a fixed hash lowers the Pile loss from 2.060 to 1.932, and a learned Switch router to 1.881.

::note aside 45:04 | DeepSeek's early papers ablate dense layers, hash routing, Switch routers and more. "If you're interested in a lot of these questions, like architecture, design and so on, I would suggest you read the earlier DeepSeek papers."

::slide 23 | DeepSeek-V3's benchmarks against DeepSeek-V2.5, Qwen2.5-72B, Llama-3.1-405B, GPT-4o and Claude-3.5-Sonnet: MMLU-Pro 75.9, GPQA-Diamond 59.1, MATH 500 90.2, AIME 2024 39.2, Codeforces 51.6, SWE-bench 42.0

DeepSeek-V3, taken apart at the end of the lecture, is competitive with the best closed models of its generation (90.2 on MATH 500 against 74.6 for GPT-4o). The professor expects MoEs to dominate big models for at least the next couple of years.

::slide 24 | "Why haven't MoEs been more popular?": Fedus et al.'s paragraph that sparsity pays when many accelerators can host the extra parameters; and Zoph et al.'s "sparse models often suffer from training instabilities", with a training-loss curve that spikes to about 350 near step 14,000

If MoEs are so good, why did they catch on only "in 2024 onwards", when Google was pushing them in 2022? The slide gives two reasons and the professor adds a third aloud:
- **Infrastructure.** It is "hard to parallelize experts in ways that are efficient in utilization", and an MoE has so many parameters that "it's hard to fit them on a single device". The advantage appears mainly when many machines can host the extra parameters (the Fedus quote).
- **Training.** The objectives are heuristic, and "the MoEs can really blow up on you": the left curve on the slide diverges late in training.
- **Fine-tuning**, which comes up near the end of the lecture (slide 50): sparse MoEs overfit small fine-tuning sets.

None of these is a law of nature. Each has a fix, and most of the remaining lecture is those fixes.

::slide 25 | typical: the MLP replaced by an MoE layer; less common: MoE over attention heads (ModuleFormer, JetMoE), with a router choosing among attention modules

Experts can also be made of attention heads, but these are "much less common" and "haven't been quite as easy to tame". Big models put the experts in the FFN only.

::note skip 47:10 | "So I'm only going to talk about the left": attention-head MoEs are set aside for the rest of the lecture.

## The router: who chooses, and how the gate is computed {#routing}
source: lecture_04.pdf p26-p31 · video 47:23-54:20

::slide 26 | MoE – what varies? Routing function; expert sizes; training objectives

Designing an MoE comes down to three choices: how tokens are routed to experts, how big the experts are (for a fixed budget, many small experts or a few large ones), and how to train the router. This part is the first, the next part the second, and the training part the third.

::slide 27 | a 5-expert × 3-token score matrix, three ways: token chooses expert (top-k down each token's column), expert chooses token (top-k along each expert's row), and a global assignment over the whole matrix

Every routing scheme starts from a score for each (token, expert) pair and keeps some top-k. They differ in which direction the top-k runs.
- **Token choice**: each token picks its $k$ favourite experts.
- **Expert choice**: "each expert picks their favorite tokens". The load per expert is fixed by construction, but a token may be picked by many experts or by none.
- **Global routing**: "this very complicated router that globally decides" the assignment for the whole batch, as an optimization problem.

**Worked example on the slide's numbers.** Token T1's column of scores over experts E1–E5 is 3.13, 0.51, −1.32, 2.25, −2.81. With token choice and $k = 2$, T1 goes to E1 and E4. T2's column is 0.14, −0.25, 1.97, 2.61, −0.68, so T2 goes to E4 and E3. Expert E4 is now serving two of the three tokens. Under expert choice with one token per expert, E1 would take T1 (3.13) and E4 would take T2 (2.61), and no expert can be oversubscribed.

::slide 28 | OLMoE's ablation, token choice (TC) against expert choice (EC) over 200B tokens: training loss, C4 validation loss and HellaSwag all favour TC (HellaSwag about 58 against 53), MMLU roughly tied

"Almost all the MoE do token choice TopK." In OLMoE's controlled comparison, token choice reaches lower validation loss and higher downstream scores than expert choice. Expert choice "also trains fine", and successful expert-choice models exist, but token choice has been much easier to get working.

::note aside 49:36 | The professor recalls that "one of the unreleased Llama 4 models" may have used expert choice, and adds that this is not "a strong vote of confidence".

::slide 29 | top-2 routing (router probabilities such as 0.65 and 0.3 weighting two of four FFNs) is "used in most MoEs": Switch Transformer (k = 1), GShard, Grok, Mixtral (2), Qwen, DBRX (4), DeepSeek (7); hash routing, a fixed hash of the token choosing the FFN, is a "common baseline"

The router is "a small, small linear projection": each expert has a vector, and the experts whose vectors have the largest inner products with the token's hidden state win. The $k$ varies by model, from Switch's single expert to DeepSeek's 7. (DeepSeek's 7 is the small DeepSeekMoE ablation of slide 33, 1 shared plus 7 routed experts; DeepSeek v1 itself routes to 6, and v3 to 8, as the table on slide 35 shows.)

One mystery the professor admits to: you do not strictly need a learned router. Hashing tokens to experts "gives gains, not as much as TopK" (slide 22 showed it). It is a common baseline in papers, not used in deployment.

::slide 30 | other routing methods: reinforcement learning on the router (used in the earliest work, Bengio 2013, "not common now"), and BASE routing, which solves a linear assignment between tokens and experts (Clark '22)

Two more principled alternatives exist. To a learning theorist, routing is a bandit problem: you pick $k$ of $N$ arms and only observe the ones you picked, so use RL. That was done in early work, but RL adds "a lot of overhead in terms of both the RL algorithm and the stochasticity", while heuristics on plain top-k work. Or solve the global assignment exactly as a linear assignment problem, an idea the professor loves "as a person that likes things that make sense", but it "hasn't been seen at scale at all", because it is "extremely expensive relative to the others".

::slide 31 | the top-k router in equations (from DeepSeekMoE): h_t = Σ_i g_{i,t} FFN_i(u_t) + u_t; g_{i,t} = s_{i,t} if s_{i,t} is among the top K, else 0; s_{i,t} = Softmax_i(u_tᵀ e_i), "gates selected by a logistic regressor"; this is the DeepSeek v1-2 router (Grok and Qwen too), while Mixtral, DBRX and DeepSeek v3 softmax after the top-k

Written out, the MoE layer for token $t$ with input $u_t$ is

$$ h_t = u_t + \sum_{i=1}^{N} g_{i,t}\,\mathrm{FFN}_i(u_t), \qquad g_{i,t} = \begin{cases} s_{i,t} & \text{if } s_{i,t} \in \mathrm{TopK}(\{s_{j,t}\}, K) \\ 0 & \text{otherwise} \end{cases}, \qquad s_{i,t} = \mathrm{Softmax}_i(u_t^\top e_i) $$

$u_t$ passes through the residual; $e_i$ is expert $i$'s learned vector, so the scores are a softmax over $N$ logits, a logistic regression. Only the $K$ experts with nonzero gates run.

The slide's side notes mark one real variation: *where* the softmax goes.
- **Softmax, then top-k** (DeepSeek v1-2, Grok, Qwen): normalize over all $N$ experts, then zero the rest. The kept gates sum to less than 1.
- **Top-k, then softmax** (Mixtral, DBRX, DeepSeek v3): pick the top $K$ logits and softmax over only those. The kept gates sum to exactly 1.

**Worked example.** Use T1's logits from slide 27, top-2. Softmax over all five gives 0.666 for E1 and 0.276 for E4, summing to 0.942; the other 0.058 went to experts that do not run. Softmaxing over only the two selected logits gives 0.707 and 0.293. Same experts, different mixing weights.

::predict topk-token-choice-routing

Look at the structure once more: score, take a top-k, run only the winners. "If you were paying attention in the DSA slide, this looks a lot like DSA", and the same pattern appears in H-Nets and elsewhere. It is worth recognizing on sight.

::video 48:32-49:01 | token picks experts, expert picks tokens, or a global assignment; and "almost all the MoE do token choice TopK"
::kp topk-token-choice-routing

## Expert sizes: fine-grained experts and shared experts {#fine-grained}
source: lecture_04.pdf p32-p35 · video 54:20-58:26

::slide 32 | (a) conventional top-2 routing over N experts; (b) fine-grained segmentation: each expert split in two, 2N experts, top-4; (c) shared expert isolation (DeepSeekMoE): expert 1 always on (green) plus top-3 of the rest; "originally from DeepSpeed MoE"

Two refinements, now in nearly every big MoE, change the second design axis.

**Fine-grained experts.** Cut each expert into smaller pieces and route to proportionally more of them. In panel (b), each expert is split in two and $k$ doubles from 2 to 4. Compute per token is unchanged (4 half-size experts cost the same as 2 full ones), but the number of different expert combinations a token can choose from explodes, so each piece can specialize more narrowly.

**Shared experts.** Some processing is useful for every token. With only routed experts, "you were just reusing a lot of these weights to do common modeling": each expert re-learns the same common function. So make a few experts **shared**: they bypass the router and process every token, and the routed experts are freed to specialize. In panel (c), one of the fine-grained experts becomes shared and the router picks 3 of the rest, so the token still runs 4 small experts.

::note slip 54:27 | The professor says "DeepSeekMoE pioneered this idea ... called Shared Experts" and later that DeepSeek v1 "comes up with both the fine-grained and shared expert design". The slide credits shared experts to DeepSpeed-MoE; DeepSeekMoE popularized the combination.

::slide 33 | DeepSeekMoE's ablation, all with the same total and active parameters, normalized to the best: 0 shared + 2 of 16 routed (GShard), 1 shared + 1 of 15, 1 shared + 3 of 31 (fine-grained), 1 shared + 7 of 63 (finer); the finest wins nearly everywhere, and on TriviaQA the scores climb from about 0.61 to 0.85, 0.93 and 1.0

This is a careful ablation, and checking its bookkeeping shows why. In units of one GShard expert:

| configuration | expert size | parameters | compute per token | routed choices |
|---|---|---|---|---|
| 0 shared + 2 of 16 | 1 | 16 | 2 | C(16, 2) = 120 |
| 1 shared + 1 of 15 | 1 | 16 | 2 | C(15, 1) = 15 |
| 1 shared + 3 of 31 | 1/2 | 16 | 2 | C(31, 3) = 4,495 |
| 1 shared + 7 of 63 | 1/4 | 16 | 2 | C(63, 7) ≈ 5.5 × 10⁸ |

Parameters and compute are identical in every row; only the structure changes. Adding the shared expert (row 1 to row 2) helps most on the knowledge benchmarks, TriviaQA and NaturalQuestions (about 0.61 to 0.85 and 0.56 to 0.79), and each finer split helps further. "More experts, shared experts all seem to generally help."

::slide 34 | OLMoE's ablations: 32 routed experts against 31 routed + 1 shared give nearly identical curves; 8, 32 and 64 experts at equal compute show 8 clearly worse and 64 slightly best

OLMoE, "the nice Western carefully controlled MoE study", agrees on fine-graining (8 experts are clearly worse than 32 or 64) but disagrees on shared experts: in their setting, one shared expert makes no difference. Why the two studies differ is left open.

::slide 35 | the routing setups of recent MoEs: routed, active and shared expert counts and the fine-grained ratio, from GShard (2048 routed, 2 active) and Switch (64, 1) through Mixtral (8, 2), DBRX (16, 4), DeepSeek v1 (64, 6, 2 shared, 1/4), Qwen 1.5 (60, 4, 4 shared, 1/8), DeepSeek v3 (256, 8, 1 shared, 1/14), OLMoE (64, 8, 1/8), MiniMax (32, 2, ~1/4) and Llama 4 Maverick (128, 1, 1 shared, 1/2)

The table shows the field's history in one column. The early Google models and the first Western MoEs (Mixtral, DBRX, Grok) use a few full-size experts and no shared ones. From DeepSeek v1 on, experts are a fraction of a standard FFN (the "fine-grained ratio") and most models add shared experts. The professor compares DeepSeekMoE and DeepSeek-V3 to the Llama architecture for dense models: the standard design that everyone copied. Qwen 3.5 and GLM still use both features.

::worked fine-grained-shared-experts

**Worked comparison.** Mixtral routes 2 of 8 full-size experts, so its routed FFN compute per token is 2 standard FFNs, and 2/8 = 25% of its routed parameters are active. DeepSeek v3 routes 8 of 256 experts at 1/14 size, so its routed compute is $8/14 \approx 0.57$ standard FFNs, about 0.29× Mixtral's, while only $8/256 \approx 3.1\%$ of its routed parameters are active. "8 experts" sounds four times more expensive than "2 experts"; it is less than a third.

::predict fine-grained-shared-experts
::widget fixture:lecture_04--moe-ffn-budget | click the Mixtral preset and pin it as the baseline, then click DeepSeek v3: parameters grow while FLOPs per token fall to about 0.3×, and the count of distinct top-k sets jumps by many orders of magnitude

A student asked how shared experts interact with expert parallelism. They get no parallelism savings, since every token needs them. Instead you copy the shared expert onto every device, spending memory to avoid communication.

::video 55:27-55:44 | why a shared expert helps: it absorbs the common processing, so the routed experts specialize more
::kp fine-grained-shared-experts

## Training the router: why it is hard, and the first two fixes {#train-routing}
source: lecture_04.pdf p36-p39 · video 42:33-43:19, 58:26-1:03:13

::slide 36 | "Major challenge: we need sparsity for training-time efficiency… But sparse gating decisions are not differentiable!"; solutions: 1. reinforcement learning on the gating policy; 2. stochastic perturbations; 3. heuristic "balancing" losses; "Guess which one people use in practice?"

If every expert ran on every token during training, learning to route would be easy: you would see which expert helps each input. But then you pay the FLOPs of all experts, and the whole point was not to. So training is sparse too, and that creates two problems.
- **The choice is not differentiable.** Which $k$ experts win is a step function of the router's scores. Gradients flow through the gate values $g_{i,t}$ of the experts that ran, but not through the decision itself.
- **You never see the counterfactual.** "You only know the ones that activated." An expert that was not chosen produces no output for that token, so the loss says nothing about whether it would have been better.

"So it's got this RL bandit flavor to the problem, but we're not going to solve it with either RL or bandits. We're going to solve it with the power of heuristics and deep learning magic." The answer to the slide's question is number 3, heuristic balancing losses. The other two are worth knowing first.

::predict nondifferentiable-routing-solutions

::slide 37 | Clark et al.'s scaling curves: validation loss against expert count (1 to 512) for six model sizes, with S-BASE, RL-R (REINFORCE) and Hash routing, and an overlay comparing them; "RL is the 'right solution' but gradient variances and complexity means it's not widely used"

**RL.** Treat the router as a policy and train it with REINFORCE. It works: the RL-R curves fall with expert count much like the others. But "not so much better that it's a clear win", and its gradient variance and complexity mean that simpler routers proposed in the same paper beat it.

::note slip 1:00:04 | Both the slide and the professor date the paper "Clark et al 2020". The unified scaling-laws-for-routed-models paper by Clark et al. is from 2022, which is also the year slide 30 gives ("Clark '22").

::slide 38 | Shazeer et al. 2017's noisy top-k gating: G(x) = Softmax(KeepTopK(H(x), k)), H(x)_i = (x·W_g)_i + StandardNormal()·Softplus((x·W_noise)_i), KeepTopK setting all but the top k to −∞; "routing decisions are stochastic", "experts that are a bit more robust", "the model learns how to rank K experts"

**Noise.** The first modern MoE paper (Shazeer et al. 2017) adds Gaussian noise to the router logits during training, with a learned, input-dependent scale (the softplus term):

$$ H(x)_i = (x W_g)_i + \varepsilon_i \cdot \mathrm{Softplus}\big((x W_\text{noise})_i\big), \qquad \varepsilon_i \sim \mathcal{N}(0, 1), \qquad G(x) = \mathrm{Softmax}\big(\mathrm{KeepTopK}(H(x), k)\big) $$

What does that buy? If two experts score almost the same, "then stochastically, you'll pick one of them and not the other." Near-ties are broken at random, so the router explores, and experts that turn out to help accumulate weight. The softmax over the kept $k$ means the model also learns how to *rank* its chosen experts, rather than making a hard pick.

::slide 39 | Fedus et al.'s router code, adding a random_uniform(1−ε, 1+ε) jitter to the router logits during training and casting them to float32 before the softmax; a table from Zoph et al.: Baseline 4/6 runs stable, quality −1.755; input jitter 3/3, −1.777; dropout 3/3, −1.822

The Switch Transformer used a cheaper variant, a uniform multiplicative jitter of about $1 \pm \varepsilon$, "for the same goal of getting less brittle experts". Later Google work (Zoph et al. 2022) removed it. Read the table: with jitter or dropout every run was stable, but the quality (log perplexity, higher is better) was worse, −1.777 and −1.822 against the baseline's −1.755. The professor's conclusion is that the stochastic exploration terms are not needed, and that simple top-k with balancing is what you actually do.

::note slip 1:02:51 | The professor says the later ablation found that dropping the stochastic tricks "actually helps with both stability and overall quality". The table on the slide shows a better quality without them, but fewer stable runs (4 of 6 against 3 of 3). Their removal traded some stability for quality, and the stability problem was then handled another way (float32 routers and the z-loss, later in this lecture).
::note aside 1:02:25 | Two details hiding in the code. The jitter is written as an addition of a factor near 1, although the text and the table call it multiplicative input jitter; either way the logits wobble by about ε. And the next line casts the router logits from bfloat16 to float32 "for stability", the fix that the stability part of the lecture returns to.
::kp nondifferentiable-routing-solutions

## Balancing losses: the heuristic that makes MoEs train {#balancing}
source: lecture_04.pdf p40-p43 · video 1:03:13-1:11:37

Start with what goes wrong if you simply run gradient descent through a top-k router. The experts that happen to be chosen get gradient signal and improve. Improved experts get stronger router weights, "strong weights means that they're selected more often", and they get still more signal. Experts that are not chosen get no gradient at all, so they never improve and never win. "So you get this rich gets richer effect ... And they run away, taking on everything." This **expert collapse** (or expert starvation) is, in the professor's words, "a very, very real problem" and "the core issue that you have to solve with heuristic training of MoEs."

::predict expert-collapse-without-balancing
::video 1:03:35-1:04:06 | the loop: chosen, stronger weights, chosen more often, "they run away, taking on everything"

::slide 40 | the Switch Transformer's auxiliary loss: loss = α·N·Σ_i f_i·P_i, with f_i the fraction of the T tokens in the batch dispatched to expert i (by argmax) and P_i the router probability allocated to expert i, averaged over the batch; "The derivative with respect to p_i(x) is αN/T² Σ 1[argmax p(x) = i], so more frequent use = stronger downweighting"

The fix is an auxiliary loss added to the language-modelling loss. For $N$ experts and a batch of $T$ tokens, the Switch Transformer uses

$$ \mathcal{L}_\text{aux} = \alpha N \sum_{i=1}^{N} f_i P_i, \qquad f_i = \frac{1}{T}\sum_{x} \mathbb{1}[\arg\max p(x) = i], \qquad P_i = \frac{1}{T}\sum_x p_i(x) $$

$f_i$ is the fraction of tokens actually dispatched to expert $i$ (a count, with no gradient). $P_i$ is the router probability given to expert $i$, averaged over the batch (smooth, with a gradient). The professor admits this is "not a thing, where you would derive it from first principles". The way to understand it is through its gradient. Only $P_i$ contains $p_i(x)$, with coefficient $1/T$, so

$$ \frac{\partial \mathcal{L}_\text{aux}}{\partial p_i(x)} = \frac{\alpha N}{T}\, f_i = \frac{\alpha N}{T^2} \sum_{x'} \mathbb{1}[\arg\max p(x') = i] . $$

The push on an expert's probability is proportional to how many tokens it is already getting. "The more tokens you get, the more negative gradient you get." Popular experts are pushed down hardest, which directly counters the rich-get-richer loop.

::worked load-balancing-loss

**Worked example.** In a batch of $T = 400$ tokens, expert A is the argmax for 300 tokens and expert B for 100. The push on $p_A(x)$ is $(\alpha N / 400^2) \cdot 300$ and on $p_B(x)$ it is $(\alpha N / 400^2) \cdot 100$: three times stronger on the popular expert. And if the router's probabilities match its loads ($P_i = f_i$), the loss is $\alpha N \sum f_i^2$, which is smallest, equal to $\alpha$, when every expert gets exactly $1/N$ of the tokens.

::predict load-balancing-loss

::slide 41 | DeepSeek v1-2: per-expert balancing L_ExpBal = α₁ Σ f_i P_i, the Switch loss with f_i normalized by N′/(K′T); per-device balancing L_DevBal = α₂ Σ f′_i P′_i, the same objective aggregated over the experts on each device

DeepSeek v1 and v2 rebuild exactly this. Backpropagate straight through the experts, "ignoring all this nondifferentiability and exploration concerns", and add the Switch-style per-expert loss. Then, being "very savvy with their systems design", add a second copy computed per **device**. If one machine holds four experts and another holds the other four, you want both machines busy: "you want those two machines to be balanced, so both of them are running at full utilization."

A student asked why the device term is needed, since perfectly balanced experts also balance devices. The answer: you do not want to turn the per-expert loss up until the load is fully uniform, "because that has deleterious effects on training dynamics". Device balance matters enough to pay a little extra loss for it specifically.

::kp load-balancing-loss

::slide 42 | DeepSeek v3: a per-expert bias b_i added to the score only for top-k selection, g′_{i,t} = s_{i,t} if s_{i,t} + b_i is in the top K_r, else 0; "auxiliary loss free balancing"; but a complementary sequence-wise balance loss L_Bal = α Σ f_i P_i "to prevent extreme imbalance within any single sequence"; "(but the approach is not fully aux loss free..)"

DeepSeek v3 tries to get rid of these "uglier" auxiliary losses. Each expert gets a bias $b_i$ that is added to its score *only when choosing the top-k*. The gate value that multiplies the expert's output is still the unbiased $s_{i,t}$. The bias is adjusted by online learning rather than by gradient: after each step, lower it for experts that received too many tokens and raise it for those that received too few (the update rule is in the DeepSeek-V3 report). Balancing therefore changes *which* experts are chosen, never how their outputs are weighted.

It is not fully aux-loss free, as the slide notes: v3 keeps a small sequence-wise balance loss "to ensure extreme imbalances don't happen". "There's no solution that fully gets rid of them so far."

::predict aux-loss-free-bias

::slide 43 | OLMoE with and without the load-balancing loss (LBL): training loss and C4 and Pile validation losses are all higher without it; expert assignment in the first MoE layer, without balancing one expert takes nearly 100% of tokens at first and then two experts split them, while with balancing all eight experts stay near 12.5%

Do we really need all this? OLMoE ran the ablation, and without the balancing loss "the effects are pretty catastrophic". The losses are clearly higher. The bottom panels show why: without balancing, one expert briefly takes almost every token, then two experts split them roughly half and half, and the other six get essentially nothing for the rest of training. With balancing, all eight sit near 1/8. "Without it, we've thrown away a ton of parameters. Those experts are doing absolutely nothing."

So balancing is not only about hardware utilization, as slide 40's motivation suggests; it decides whether most of the model's parameters are trained at all. The professor's summary of MoE training: two dynamics that cancel. Useful experts reinforce themselves, "a positive reinforcement cycle that's nicely balanced out by evening things out", and the rest of the system can be treated "as if you can just pump gradients through".

::note aside 1:09:51 | "That same trick is used in DSA", and in H-Nets (an attempt to remove tokenizers): top-k selection made trainable by auxiliary losses is becoming a general ingredient of architecture design.
::kp expert-collapse-without-balancing
::kp aux-loss-free-bias

## The systems side: expert parallelism, sparse matmuls, dropped tokens {#systems}
source: lecture_04.pdf p44-p47 · video 1:11:37-1:16:35

::slide 44 | left: the MoE encoder with device placement, FFN_1 … FFN_E on different devices, all-to-all dispatch and combine; right: how model weights and data are split over a 4×4 grid of cores under data, model, model+data, expert+data and expert+model+data parallelism

The two standard ways to spread training over many machines both run out ([L8](#/read/lecture_08) covers them in depth). **Data parallelism** gives each machine different examples, and stops once you have used up your batch size. **Model parallelism** cuts the model itself, and stops once you have used up its natural cut points. MoEs add a third axis. Because each expert FFN fits on one device, you can place different experts on different devices: **expert parallelism**.

In the right-hand grid, under expert+data parallelism, each core holds a different expert (a different colour) while the data is still split as in data parallelism. The weights stay where they are. What moves is the tokens: before the MoE layer, an all-to-all **dispatch** sends each token's activation to the devices holding its $k$ experts; after it, an all-to-all **combine** brings the outputs back. That is why communication grows with the number of tokens, the width of each activation, and the number of distinct devices a token touches.

::predict expert-parallelism-communication

::slide 45 | MegaBlocks: (A) batched matrix multiplication, one equal-sized product per expert with a fixed expert capacity; (B) the same as one block-diagonal matmul; (C) block-sparse matmul, which allows imbalanced routing and variable-sized experts; "Modern libraries like MegaBlocks (used in many open MoEs) use smarter sparse MMs"

On a single GPU holding several experts, the naive implementation runs one small matmul per expert. "This is not nice, because you ideally want these bigger matrix multiplies", where caches are reused and the hardware stays busy ([L5](#/read/lecture_05) explains why). Worse, each expert receives a different, data-dependent number of tokens, so equal-sized batched matmuls (panel A) force you to pad short batches or cut long ones. Writing all experts as one block-diagonal product (panel B) turns the many small matmuls into one big one, but the blocks are still equal-sized. MegaBlocks' **block-sparse** products (panel C) let the blocks differ in size from expert to expert, so no token has to be padded or dropped. The professor sees a hardware-architecture co-design here: MoE computation maps onto structured matrix multiplications that hardware runs efficiently.

::note slip 1:12:42 | He says this uses "sparse matrix multiply that has been built into GPUs" and structured sparsity "natively supported in hardware". MegaBlocks' block-sparse kernels are software kernels that run dense tensor-core tiles; they are not the GPU's 2:4 structured-sparsity feature. The slide says only "smarter sparse MMs".

::slide 46 | Nemotron 3's LatentMoE: in the standard MoE, the full-width token goes through all-to-all dispatch to experts E1-E4 and a shared expert SE; in LatentMoE, a latent down-projection precedes the dispatch and an up-projection follows the combine, and there are more, smaller-width routed experts (E1-E8)

A recent trick, from Nemotron 3, attacks the communication directly. The shared expert, which never leaves the device, keeps the full hidden width. The routed experts, whose inputs must be shipped, work in a smaller latent width: project the residual stream down, *then* do the all-to-all, then project back up after the combine. Shipping, say, a quarter-width vector instead of a full one would cut dispatch traffic 4× (the slide gives no ratio), and the professor says it saves a lot of communication "without fully having the drawbacks of having a smaller hidden dimension size". In the figure the saving is spent on more routed experts.

::note skip 42:13 | Asked earlier about the communication cost of expert parallelism, the professor said "I will not explain, but I will show you one trick of how you reduce that": this down-projection is the one trick shown.
::kp expert-parallelism-communication

::slide 47 | "Fun side issue – stochasticity of MoE models": routing, permutation, computation and un-permutation of six tokens over three experts at capacity factor 1; one expert is sent three tokens, has two slots, and drops one (red cross), whose output comes back as 0, while another expert leaves a slot unused; "other people's queries can drop your token!"

Each expert has a fixed **capacity**, a number of token slots per batch. In the Switch Transformer it is the capacity factor times the tokens an expert would get under perfect balance, $c \cdot kT/E$. In the slide's example, 6 tokens over 3 experts at capacity factor 1 gives 2 slots each. One expert is chosen by 3 tokens, so the last one is **dropped**: it skips that expert, and only its residual passes through, while another expert sits with an empty slot.

The professor tells it as a queue. A popular expert's queue of tokens builds up "until my queue is so long, I have to start dropping tokens", and older MoE inference code would "silently drop" them: "You just send a zero back and pretend that that was fine." Routing and capacity are decided per batch, and a serving batch mixes requests from many users. So "if other users sent in queries that hit the experts that you're using", they could "bump you out of the expert queue", and you would get a worse answer because of someone else's prompt. That is randomness a dense model does not have.

**Worked example.** A batch of 512 tokens, 8 experts, top-1, 96 slots per expert: 768 slots in total, far more than 512. But the router sends 150 tokens to expert 2 and spreads the other 362 evenly, about 52 per expert. Expert 2 overflows by $150 - 96 = 54$ tokens, and those 54 are dropped although average capacity was plentiful. Dropping is a tail event of the load, which is one more reason balance matters.

::predict token-dropping-stochasticity
::widget fixture:lecture_04--capacity-drop | raise only the other requests' tokens to the hot expert E0: your own tokens start being dropped although your request and the mean load are unchanged
::widget fixture:moe-routing | route tokens one at a time with a skewed router: the popular experts fill their capacity c·k·T/E first and further tokens are dropped even when the total slots would suffice

This was "trivia", in the professor's word, and it has been solved: today's "drop less" architectures, MegaBlocks and other common open-source MoE frameworks, do not drop tokens.

::video 1:15:30-1:16:18 | the overflowing queue, and how other users' tokens can push yours out
::kp token-dropping-stochasticity

## Stability, fine-tuning, and upcycling {#stability}
source: lecture_04.pdf p48-p53 · video 1:16:35-1:22:11

::slide 48 | Zoph et al.'s instability example: 10 logits of 128 and one of 128.5; a bfloat16 roundoff of 0.5 changes the softmax output from about 0.142 to 0.091 (36%) and makes all logits equal; "Solution: Use Float 32 just for the expert router (sometimes with an aux z-loss)"; the z-loss L_z(x) = (1/B) Σ_i (log Σ_j e^{x_j^{(i)}})²

Recall the rule from [L3's stability section](#/read/lecture_03): "exponentials are bad, divisions are bad, which means softmaxes are danger zone for stability issues." An MoE adds "yet another softmax", in the router.

The slide's footnote shows how fragile it is. Give a softmax ten logits of 128 and one of 128.5. In exact arithmetic the largest output is $e^0 / (e^0 + 10\,e^{-0.5}) \approx 0.142$. But bfloat16 keeps only 8 significant bits, so between 128 and 256 its representable numbers are spaced 1 apart, and 128.5 rounds to 128. Now all eleven logits are equal and the output is $1/11 \approx 0.091$, a 36% change from one roundoff. Large router logits make the routing itself noisy.

The fixes are cheap:
- **Compute the router in float32**, and only the router. It is one small linear layer per token (hidden width × number of experts), negligible next to the experts, which stay in bf16.
- **Add a z-loss**, which penalizes the squared log of the softmax normalizer and so keeps the logits small. It is the same z-loss [L3](#/read/lecture_03) introduced for the output softmax; in MoEs it was "quite popular ... even in the early days".

::predict router-z-loss-fp32

::slide 49 | OLMoE with a router z-loss (weight 0.001) against none: without it the training loss, C4 validation loss, HellaSwag and MMLU curves are full of spikes

OLMoE's ablation is unambiguous: "It's quite clear, from these very spiky training loss curves, that z-loss on the router can be quite helpful."

::video 1:16:42-1:17:10 | why the router is fragile: another softmax, and softmaxes are the danger zone
::kp router-z-loss-fp32

::slide 50 | "Sparse MoEs can overfit on smaller fine-tuning data": on the SuperGLUE CB task the sparse model's training score reaches 100 while its validation score stays near 91, below the dense model's (about 94); Zoph et al.'s fix, updating only some parameters (all ≈ 86.2, non-MoE ≈ 86.2, MoE only ≈ 82.7, attention ≈ 85.7, FFN ≈ 86.4 on SuperGLUE); DeepSeek's fix, 1.4M SFT examples

The third barrier from slide 24. MoEs "have so many parameters that if you're trying to fine tune the experts, you actually end up with very serious overfitting issues." On the slide, the dense model's train and validation curves stay close, while the sparse model memorizes its training set and validates worse.

There are two kinds of fix. One is to fine-tune only part of the model. Zoph et al.'s bar chart shows that updating only the non-MoE parameters scores as well as updating everything, while updating only the MoE layers is clearly worst. Fine-tuning just the attention layers is, the professor says, something he sees "quite often with MoE works recently". The other is "the bitter lesson version": "maybe you should just use 1.4 million examples instead of whatever number you have", as DeepSeek did for its chat model, so that even the experts have enough data to fine-tune without overfitting.

::predict moe-adoption-barriers
::kp moe-adoption-barriers

::slide 51 | sparse upcycling: the dense block's layer norms and attention are copied, its MLP is copied E times into the experts, and the router is trained from scratch; right: C4 validation accuracy against extra pretraining time for Base, Large and XL, where upcycled models (orange) pull ahead of continued dense training (blue)

Can you start an MoE from a dense model you already trained? **Upcycling** copies everything, makes $E$ copies of each MLP to serve as the experts, and adds a randomly initialized router. Then you keep training. All the experts start identical, so right after upcycling the MoE computes the dense model's function (up to the gate weights). What breaks the symmetry is the random router plus "the stochasticity of which inputs go to where": different experts see different tokens, receive different gradients, and "start to specialize". The original paper's plot shows the upcycled model beating the same dense model trained for the same extra compute.

::predict upcycling

::slide 52 | MiniCPM-MoE (13.6B), upcycled from MiniCPM-2.4B with top-2 of 8 experts and about 4B active parameters, trained on about 520B tokens: MMLU 53.46 to 58.90, GSM8K 53.83 to 61.56, HumanEval 50.00 to 56.71

MiniCPM, a Chinese lab the professor likes for its controlled ablations, upcycled its 2.4B dense model into an 8-expert, top-2 MoE. The numbers on the slide are consistent with simple copying. If the FFNs are a fraction $f$ of the 2.4B parameters, the upcycled model has $2.4(1 - f) + 8 \cdot 2.4 f$ parameters in total and $2.4(1 - f) + 2 \cdot 2.4 f$ active. With $f = 2/3$ these give $0.8 + 12.8 = 13.6$B total and $0.8 + 3.2 = 4.0$B active, matching the slide's 13.6B and "~4B active". After about 520B more tokens, it gains on every benchmark shown.

::note slip 1:21:15 | The professor says MiniCPM was upcycled "to a 13.4 billion parameter model". The slide's table says MiniCPM-MoE (13.6B).

::slide 53 | Qwen1.5-MoE, initialized from Qwen 1.8B, top-4 of 60 experts with 4 shared: 14.3B parameters, 2.7B activated; "one of the first (confirmed) upcycling successes"

Qwen's first MoE was initialized from Qwen 1.8B and became Qwen1.5-MoE-A2.7B (2.7B active of 14.3B), with a DeepSeekMoE-like design of 60 routed and 4 shared fine-grained experts. It was one of the first confirmed large-scale upcycling successes.

::note aside 1:21:24 | Slide 53's table pairs the wrong benchmark rows with Qwen1.5-7B and Gemma-7B. Slide 21 shows the same numbers correctly: Gemma-7B has MMLU 64.6 and GSM8K 50.9, Qwen1.5-7B has 61.0 and 62.5. This is a deck error, not a spoken one.

Upcycling has nearly vanished: "I don't think I've seen a single upcycled model this year." Labs no longer train a big dense model first and convert it. "You might as well just train your big hero run on an MoE to start with."

::video 1:20:00-1:20:32 | copy the MLP into every expert, a random router, and the stochasticity that makes experts specialize
::kp upcycling

## DeepSeek v1 to v3: the whole recipe in one model family {#deepseek}
source: lecture_04.pdf p54-p60 · video 1:22:11-1:26:14

The professor ends by walking through DeepSeek's MoEs, because almost every idea of the lecture shows up in them, and recommends the DeepSeek papers as "very well written".

::slide 54 | DeepSeek MoE v1 (16B total, 2.8B active): 2 shared plus fine-grained (64/4) routed experts; standard top-k routing (softmax over all experts, then top-k); standard auxiliary balancing, per expert and per device

**v1** is "the prototypical platonic ideal of MoE model": shared plus fine-grained experts, the standard top-k router of slide 31, and the expert-level and device-level balancing losses of slide 41.

**Worked bookkeeping**, in units of a standard FFN, from the table on slide 35 (64 routed experts of 1/4 size, 6 active, 2 shared):
- routed compute per token: $6 \times 1/4 = 1.5$; shared: $2 \times 1/4 = 0.5$; total 2 standard FFNs;
- routed parameters: $64 \times 1/4 = 16$ standard FFNs, of which $6/64 \approx 9.4\%$ are active for any token;
- distinct routed sets a token can pick: $\binom{64}{6} = 74{,}974{,}368$.

Across the whole model, attention included, 2.8B of 16B parameters are active, about 18%.

::slide 55 | DeepSeek MoE v2 (236B total, 21B active): 2 shared plus fine-grained (160/10) experts, 6 active; new: top-M device routing (first pick the M devices holding the highest-affinity experts, then the top-K experts on them; M ≥ 3 performs about as well as unrestricted routing) and a communication balancing loss for traffic both into and out of each device

**v2** scales the design up and adds two systems pieces, both of them "adding auxiliary losses to optimize systems".
- **Top-M device routing.** Each token's experts may live on at most $M$ devices: first choose the $M$ devices whose experts score highest, then the top-$K$ experts among them. This caps how many devices each token's activation must be sent to. The paper finds $M \ge 3$ about as good as unrestricted routing.
- **Communication balancing loss.** Device-limited routing bounds what each device sends; this extra loss encourages each device to also *receive* about the same amount, so no link becomes the bottleneck.

"Successful language model training is not just about deep learning, it's also about really respecting your systems."

::slide 56 | DeepSeek MoE v3 (671B total, 37B active): 1 shared plus fine-grained routed experts, 8 active; new: sigmoid gating normalized over the selected top-K (s_{i,t} = Sigmoid(u_tᵀ e_i), g_{i,t} = g′_{i,t} / Σ_j g′_{j,t}), still with top-M device routing; auxiliary-loss-free biases plus a sequence-wise auxiliary loss

**v3** keeps shared and fine-grained experts (256 routed at 1/14 size, 8 active, 1 shared) and changes how balancing and gating are done. Balancing is now the bias trick of slide 42 plus the small sequence-wise loss. Gating replaces the softmax over all experts with a sigmoid score per expert, renormalized over the $K$ selected ones so the kept gates sum to 1: in effect the "softmax after top-k" style. The professor calls it "a different way of weighting their experts, but really, mostly similar." The active fraction keeps falling across the family: 2.8/16 ≈ 18% for v1, 21/236 ≈ 9% for v2, 37/671 ≈ 5.5% for v3.

::note slip 1:23:25 | Slide 56 labels the model "V2 (671B – 37 active)" and writes "Fine-grained (258)". Both are typos: this is v3, which has 256 routed experts, as the table on slide 35 says.
::kp aux-loss-free-bias

### The rest of DeepSeek-V3: MLA and multi-token prediction

::slide 57 | Multi-head latent attention (MLA): the input h_t is compressed to a latent c_t^KV (cached during inference, hatched) from which every head's keys k^C and values v^C are produced, plus a separate small key k^R_t with RoPE applied, also cached; queries come from their own latent c_t^Q, split into a plain part and a rotated part

**MLA** attacks the KV cache from section 1 in yet another way. Instead of producing each head's keys and values directly from the hidden state $h_t$, first compress $h_t$ into a low-dimensional latent, and produce keys and values from that:

$$ c_t^{KV} = W^{DKV} h_t, \qquad k_t^{C} = W^{UK} c_t^{KV}, \qquad v_t^{C} = W^{UV} c_t^{KV} $$

"Instead of KV caching all your Ks and Vs, I only need to store these Cs", and the $c$'s are "hopefully, lower dimensional". Unlike [L3's MQA and GQA](#/read/lecture_03), which save memory by sharing keys and values across heads, MLA keeps every head; the saving comes from the latent being much narrower than heads × head width × 2.

::slide 58 | the MLA equations; benefits: only c_t^KV is cached, which can be much smaller, and W^UK can be merged into the Q projection (queries are compressed too, for training memory); complexity: without RoPE, ⟨hW^Q, W^UK c⟩ = ⟨h W^Q W^UK, c⟩; with RoPE, the rotations R_q, R_k sit between W^Q and W^UK; "The solution – have a few non-latent key dimensions that can be rotated"

Better still, you never have to rebuild the keys. Without position encoding, a score is $\langle h W^Q,\, W^{UK} c \rangle = \langle h\, W^Q W^{UK},\, c \rangle$, and $W^Q W^{UK}$ is one fixed matrix that can be precomputed: queries are projected straight into latent space and scored against the cached $c$'s.

RoPE breaks this ("this is going to conflict with rope when you do KV caching"). RoPE rotates queries and keys by position-dependent matrices, so the score becomes $\langle h W^Q R_q,\, R_k W^{UK} c \rangle$. The rotations sit *between* $W^Q$ and $W^{UK}$ and differ for every pair of positions, so no fixed product can be precomputed. DeepSeek's fix is to carry position in a small side channel: a few extra, non-latent key dimensions that are rotated (cached alongside $c$), while the latent part stays unrotated and mergeable.

::worked mla-latent-kv-cache

**Worked numbers**, using DeepSeek-V2's published sizes (not on the slide): 128 heads of width 128 would cache $2 \cdot 128 \cdot 128 = 32{,}768$ values per token per layer. MLA caches a 512-wide latent plus a 64-wide rotated key, 576 values: about 57× smaller, at every context length and every dtype.

::predict mla-latent-kv-cache
::widget fixture:lecture_04--kv-cache-model | set 128 KV heads of width 128, then tick MLA (latent 512, rotated key 64): bytes per token shrink about 57× with no head removed, and setting the rotated dims to 0 shows the 64× an un-rotated latent would give

::note skip 1:24:56 | The decoupled-RoPE construction is named but skipped: "you have nonlatent dimensions that encode positions, but I'm not going to go into too much more details about that." The merge of W^UK into the query side is on the slide but not explained aloud.
::kp mla-latent-kv-cache

::slide 59 | multi-token prediction (MTP): small sequential modules, each a transformer block on top of the main model's hidden state and the next token's embedding, predict one more token ahead with their own cross-entropy loss; "(But they only do MTP with one token ahead)"; next to it, EAGLE's draft-model diagram

The last ingredient is **multi-token prediction**. Lightweight modules on top of the main model predict tokens further ahead; DeepSeek-V3 uses a single extra token. There is a statistical argument (predicting further ahead may teach the model more), and a systems one: "you now have a speculative decoder built in", a cheap draft of the next tokens that the big model can verify, as in EAGLE.

::note deferred 1:25:30 | Speculative decoding is left to Percy's inference lecture; see [L10's speculative sampling](#/read/lecture_10).

::slide 60 | MoE summary: MoEs take advantage of sparsity, not all inputs need the full model; discrete routing is hard, but top-k heuristics seem to work; lots of empirical evidence now that MoEs work and are cost-effective

The professor's summary: MoEs let you have "more parameters than you're paying for", the parameter benefit without the compute cost. Routing looks like a hard problem, but "very simple things work well, even at scale": a linear router, a top-k, and balancing losses. And "MoEs are here to stay." The attention half ends in the same place: pure designs lose to pragmatic hybrids of a few expensive, expressive components and many cheap ones.
