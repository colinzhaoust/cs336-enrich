---
title: L3 · Architectures and hyperparameters, read through
minutes: 45
---
This lecture surveys what the language models of 2017-2025 actually chose: where the norm goes, which norm, which activation, how position enters attention, how wide the feed-forward layer is, how many heads, how deep, how big a vocabulary, whether to regularize, how to keep training from blowing up, and how to make attention cheaper. After it you can read a model card and say which choices are consensus, which still vary, and the reason (usually a systems reason) behind each default.

## How do you learn architecture without training a hundred models? {#survey}
source: lecture_03.pdf p2-p9 · video 0:05-7:29

::slide 2 | the outline (recap of the modern transformer, what large LMs share, what varies) and the theme: hands-on experience is the best teacher, other people's experience the second best

The professor opens with a confession: architecture, to him, "has always been pretty inscrutable". There is no small theory that tells you which choices are right. The best way to learn is to train models and try variants, which is what the assignments are for, but no course has the compute to cover the whole design space. So the lecture takes the second-best route, a **survey**: look at what every serious model has done, find the choices that are fixed across all effective models, and find the ones that can vary without hurting.

::slide 3 | the 2017 transformer (Vaswani et al.): sinusoidal position encodings added at the input, a ReLU feed-forward layer FFN(x) = max(0, xW₁ + b₁)W₂ + b₂, and "Add & Norm" after every sub-block (post-norm LayerNorm)

The starting point is the original transformer. Three of its choices matter today, because every one of them has since been replaced:
- **Position:** sines and cosines of the position, added to the token embedding once, at the input.
- **Feed-forward layer (FFN):** a ReLU between two linear layers, both with bias terms.
- **Normalization:** LayerNorm applied *after* each residual addition ("post-norm").

::slide 4 | the variant you implement in assignment 1: the norm moved in front of each sub-block, RoPE inside attention, a SwiGLU feed-forward layer, and no bias terms in linear layers or norms

The transformer of assignment 1 differs in exactly four places: the LayerNorm sits in front of each block, positions enter through rotary embeddings (RoPE), the feed-forward layer uses SwiGLU instead of ReLU, and no linear layer or norm has a bias. Why these? The professor's honest first answer: "we've copied a lot of this over from Llama, so did everyone else." The rest of the lecture is the better answer, the evidence and the reasoning behind each change.

::predict arch-consensus-vs-variation

::slide 5 | a collage of 2024 technical reports (Llama 3, Nemotron-4 340B, Qwen, InternLM2, Gemma, Reka, Falcon2): "over 19 new dense model releases, many of them with minor architecture tweaks"

::slide 6 | the 2025 releases: gpt-oss, Llama 4, DeepSeek-V3.2, Kimi K2, GLM-4.7, MiniMax, Step-3, Nemotron 3 and others, most of them mixtures of experts

There is plenty of data to survey. Last year the professor counted 19 new dense models. This year he expected fewer, and there are indeed fewer *dense* ones (he names Qwen3, Gemma 4, released "last Thursday", and OLMo 3), but there are many more releases overall. Most of them are **mixtures of experts**, which are the topic of the next lecture. Today is about dense models.

::slide 9 | the professor's table of models by year, one row per model and one column per choice (norm type, serial or parallel block, pre- or post-norm, position embedding, activation), next to the high-level view: "dominance of LLaMA-like architectures" and "trends over the years (QK-norm, hybrid attention)"

The table is a spreadsheet of every notable autoregressive model since 2017 and the choices it made. We return to it at the end. The lecture has three parts: common **architecture variations** (norms, activations, position embeddings, attention), **hyperparameters** (feed-forward width, head count, depth, vocabulary, regularization), and **stability tricks**, which are closely tied to the architecture.

Why is architecture so messy? Because an architecture has to satisfy three requirements at once, and each leaves a mark. It has to learn from data and generalize. It has to "train efficiently on GPUs". And "it has to not blow up": a loss curve that falls and then suddenly explodes halfway through a run is worthless. Many of the choices below are explained by the second or the third requirement, not the first.

The professor also gives a short history that the slides only imply:
- **Until about GPT-3:** open experimentation, no gold standard.
- **After Llama 2:** "everyone's like, wow, Llama 2 is great, I want my own Llama 2", so most models became Llama 2 look-alikes with minor variations.
- **Last year:** a trend toward modifications that make training more **stable**.
- **This year:** modifications that make **long context** cheaper.

::video 6:25-7:13 | the four phases: experimentation, Llama 2 convergence, stability tweaks, long-context tweaks
::note deferred 4:07 | Mixture-of-experts models, which make up most of this year's releases, are left to the next lecture, as are state-space models and other attention alternatives.
::kp arch-consensus-vs-variation

## Where should the LayerNorm go? {#norm-placement}
source: lecture_03.pdf p10-p13 · video 7:29-13:56

::slide 10 | Xiong et al.'s two block diagrams: post-LN (left) with a LayerNorm on the main line after each addition, pre-LN (right) with the LayerNorm inside each branch before attention and the FFN, plus a final LayerNorm; the equations of both

"There is one thing that everyone agrees on", the professor says: the original transformer put the LayerNorm in the wrong place. To see the two options, picture the **residual stream**: the vector x that runs from the embedding to the output, to which each sub-block (attention, then the FFN) adds its contribution.

- **Post-norm** (the original transformer, BERT) normalizes *after* the addition: $x \leftarrow \mathrm{LN}(x + \mathrm{Attn}(x))$. The norm sits on the residual stream itself, so the signal the skip connection carries is rescaled at every sub-block.
- **Pre-norm** normalizes a copy of x on its way *into* the branch: $x \leftarrow x + \mathrm{Attn}(\mathrm{LN}(x))$. The stream is never normalized inside the stack; one final norm sits after the last block.

The slide's rule is to set up the norm "so that it doesn't affect the main residual signal path". Since the newer models also put norms *after* sub-blocks (see below), the professor suggests calling the original placement **residual norm** rather than post-norm, because what matters is that it sits on the residual path. Almost every modern LM uses pre-norm. The one "somewhat funny exception" is OPT-350M, which is post-norm for no known reason.

::widget fixture:lecture_03--residual-path | switch between pre-norm and post-norm: only post-norm puts LN boxes on the straight residual line, and the count of norms on the stream climbs by 2 per layer, while pre-norm's stays at 0 however deep the stack

::slide 11 | the evidence: Nguyen and Salazar's dev BLEU curves on English-Vietnamese (the PostNorm+LayerNorm curve lowest), and Xiong's validation loss and BLEU on IWSLT and BERT, where Pre-LN without warmup converges faster than Post-LN with warmup

Why did everyone converge? The early motivation was **warmup**. Post-norm transformers need a learning-rate warmup at the start of training, and researchers hoped pre-norm would remove it. In the plots, pre-norm runs without warmup converge faster than post-norm runs with it, and post-norm without warmup barely trains. But the professor adds a caveat the slides lack: "modern transformer training still does warm up as well". Dropping warmup is not why pre-norm survived.

::slide 12 | left (Xiong 2020): expected gradient of the FFN's W¹ in each of 6 layers at initialization, flat at about 0.2 for Pre-LN, rising from about 0.05 at layer 1 to about 1.3 at layer 6 for Post-LN; right (Nguyen and Salazar): gradient global norm over training, with tall spikes for PostNorm+LayerNorm; bottom: original stated advantage removing warmup, today stability and larger learning rates

The reasons that kept it are on this slide. The slogan people who design architectures repeat is **"keep your residual stream clean"**. With pre-norm, the input x reaches the top of the network untouched, so in the backward pass the gradient flows straight down the skip connections. At initialization the gradient sizes stay the same in every layer, the flat blue bars on the left. With post-norm, every block's LayerNorm rescales the gradient on its way back. In Xiong's measurement the gradient at layer 6 is about 25 times the gradient at layer 1 (orange bars). That is **gradient attenuation**: the deeper you go from the output, the weaker the signal. The professor calls this the clearest of the explanations.

The second effect is empirical: pre-norm shows smaller and less frequent **gradient spikes** (right plot). Together they explain the bottom line. Today pre-norm is chosen for stability and for tolerating larger learning rates in large, deep networks, and "stability and the ability to go deep are both very, very important for modern large language models."

::video 11:06-11:47 | why a clean residual path keeps gradient sizes equal across layers at initialization
::note aside 12:13 | The slides cite "Salazar and Ngyuen 2019"; the paper is Nguyen and Salazar (2019), "Transformers without Tears". Similar small citation slips recur in the deck: Narang et al. (slides 17, 25) is a 2021 paper, Ivanov et al. (slides 15-16) a 2020 one, and "Dehgani" and "Idefcs" on slide 55 are Dehghani and Idefics.
::predict pre-norm-vs-post-norm
::kp pre-norm-vs-post-norm

::slide 13 | two newer layouts: the "double norm" with a LayerNorm both before and after each sub-block (inside the branch), and the layout with only the after-norm; "Recent models: Grok, Gemma 2. Olmo 2 only does non-residual post norm"

If the problem is a norm *on the residual stream*, nothing forbids a norm *after* the computation, as long as it stays inside the branch. That is exactly what several recent models do:
- **Double norm** (Grok, Gemma 2): $x \leftarrow x + \mathrm{LN}(\mathrm{Attn}(\mathrm{LN}(x)))$, a norm on the branch's input and another on its output, before the add.
- **Non-residual post-norm** (OLMo 2): $x \leftarrow x + \mathrm{LN}(\mathrm{Attn}(x))$, only the norm on the branch's output.

Neither touches x on the skip path, so both are in the same family as pre-norm. The second norm rescales only what a block *adds*, which controls the size of each update.

::worked non-residual-double-norm
::widget fixture:lecture_03--residual-path | pick the double-norm (Grok, Gemma 2) or OLMo 2 wiring: more LN boxes per layer, but all of them sit on the branch, so the residual-stream count stays 0, as with pre-norm

The professor states the rule of thumb behind this aloud, calling it "so ridiculous" and yet "proven right": **if you have stability issues, sprinkle in LayerNorms.** "Every time people have encountered stability issues, they say, oh, but what if we just throw a LayerNorm into attention? Turns out, that works too." He comes back to this with QK-norm in the stability section. Neither the slides nor the professor give evidence for the double norm itself, or say why OLMo 2 dropped the pre-norm.

::predict non-residual-double-norm
::note deferred 13:23 | "I'll get to this later as we talk about stability": why extra norms help is picked up again at QK-norm (1:09:48).
::kp non-residual-double-norm

## LayerNorm or RMSNorm, and why drop the biases? {#rmsnorm}
source: lecture_03.pdf p14-p19 · video 13:56-20:14

::slide 14 | LayerNorm, y = (x − E[x]) / √(Var[x] + ε) · γ + β (GPT-1/2/3, OPT, GPT-J, BLOOM), against RMSNorm, y = x / √(‖x‖² + ε) · γ (LLaMA family, PaLM, Chinchilla, T5)

The second near-universal change is the norm itself. **LayerNorm** takes the d_model entries of a vector x, subtracts their mean, divides by their standard deviation, then scales by a learned gain γ and shifts by a learned bias β. **RMSNorm** skips the mean and the shift: it divides x by its root-mean-square and multiplies by γ. "It's just a scaling down and scaling back up."

::worked rmsnorm-vs-layernorm

LayerNorm is the more expressive of the two, so there is no representational reason to prefer RMSNorm. The case for it is practical. In practice RMSNorm loses nothing ("RMSNorm models just as well as LayerNorm"), "but more importantly, it is faster."

::note aside 14:14 | Our note on the slide's formula: it divides by the l2 norm ‖x‖₂, not by the root-mean-square √(mean(x²)). The two differ by the constant √d_model, which the learned gain γ absorbs, so the models are equivalent; the usual definition, and the one you implement, uses the mean.

::slide 15 | "Modern explanation: it's faster (and just as good)": fewer operations (no mean), fewer parameters (no bias); "Does this explanation make sense?" Ivanov et al.'s table: tensor contractions 99.80% of FLOPs, statistical normalization 0.17%, element-wise 0.03%

The usual explanation is fewer operations and fewer parameters. The slide immediately questions it. Matrix multiplies (tensor contractions) are 99.8% of a transformer's FLOPs, and normalization is 0.17%. Saving a fraction of 0.17% cannot matter, if FLOPs were what mattered.

::slide 16 | "FLOPS are not runtime!": the same operator classes by runtime: tensor contractions 61.0%, statistical normalization 25.5%, element-wise 13.5%; the data-flow graph of an attention block annotated with FLOPs (left, "43G" for the attention) and FLOP-to-memory ratio (right, "153" for attention, 3.5 for LayerNorm)

They are not. Normalization is 0.17% of the FLOPs and 25.5% of the runtime. The reason is **arithmetic intensity**, FLOPs per byte moved, which the professor recalls from [L2's arithmetic intensity](#/read/lecture_02). A matmul does many multiply-adds for each number it reads, so its time is set by compute. A norm reads the whole activation tensor, does a few operations per element, and writes it back, so its time is set by memory traffic. In the graph on the right, the attention block does 153 FLOPs per memory access and the LayerNorm 3.5. A student asked why the gap is so large. The answer: for tensor contractions "the majority of the workload is multiplying", for normalization "the majority of the workload is memory movement", and activations are large.

So an RMSNorm saves wall-clock time by moving less data: one fewer pass over the activations for the mean, and one fewer parameter vector to load. The professor qualifies the 25%: it is "quite extreme", measured on tiny models whose matrices "don't really generally make sense in modern workloads". The direction of the effect is what carries over, and it gets stronger with each GPU generation, because compute grows faster than memory bandwidth (see [L5](#/read/lecture_05)).

::video 15:37-16:13 | the two numbers: about 0.17% of FLOPs, up to 25% of runtime
::widget fixture:lecture_03--norm-runtime | double the matmul FLOP/s with bandwidth fixed: the FLOP-share bar stays at 0.17% but the runtime share rises from 25% to 40%; cutting the norm's FLOPs moves nothing, cutting its bytes does

**Worked example.** Take the slide's split: 75% of the step in matmuls, 25% in norms. A new GPU doubles matmul throughput and keeps the same memory bandwidth. Matmul time becomes 0.375; norm time stays 0.25, because it is bandwidth-bound. The step takes 0.625 of the old time, and norms are now 0.25 / 0.625 = 40% of it.

::predict rmsnorm-vs-layernorm

::slide 17 | Narang et al.'s comparison on a 223M-parameter transformer: vanilla 3.50 steps/s and final loss 1.838; RMS Norm 3.68 steps/s and final loss 1.821, with better SGLUE, XSum, WebQ and WMT scores

The validation: in Narang et al.'s controlled comparison, swapping in RMSNorm gives about 5% more steps per second (3.68 against 3.50), and a slightly lower final loss (1.821 against 1.838). The speed is the "free systems win". The quality gain is a bonus the professor says "I don't think is something that you're guaranteed".

::slide 18 | dropping bias terms: the original FFN(x) = max(0, xW₁ + b₁)W₂ + b₂ against most implementations' FFN(x) = σ(xW₁)W₂; "Reasons: memory (similar to RMSnorm) and optimization stability"

The same reasoning extends to every **bias term**. The original transformer's linear layers all had biases; most modern implementations drop them, in the linear layers and in the norm. A bias is a tiny number of parameters, so its compute and parameter cost is negligible, but it is one more tensor to load, add and store every step: "not very arithmetically intense, but fairly memory intensive".

::worked bias-removal

The slide names two reasons, memory and optimization stability, and the professor ranks them. Biases "can also induce stability issues", he says, mentioning it "offhand", but "the primary reason these are dropped is just to simplify things from a systems perspective." He is candid about the epistemics: you cannot reason out beforehand that dropping biases is safe. It is "collectively acquired knowledge" from many training runs, with "no guarantee".

::predict bias-removal

::slide 19 | LayerNorm recap: everyone does non-residual norm (often pre-norm), for nicer gradient propagation and fewer spikes, some add a second norm; most use RMSNorm, which works as well and has fewer parameters to move; biases are dropped since the compute/param trade-off is not great

The recap adds one more piece of honesty: everyone keeps the norm off the residual stream, "but I think this might partially be because Llama 2 did that."

::note aside 18:35 | Bias terms as a source of instability are mentioned offhand and not developed; no experiment is cited for the stability reason.
::kp rmsnorm-vs-layernorm
::kp bias-removal

## Which activation, and what does a gate add? {#glu}
source: lecture_03.pdf p20-p26 · video 20:14-27:09

::slide 20 | "a whole zoo of activations": ReLU, GeLU, Swish, ELU, GLU, GeGLU, ReGLU, SeLU, SwiGLU, LiGLU; "What are these things? What do people use? Does it matter?"

The professor once made it "a point of pride to never know what a SwiGLU is". Now it is worth knowing which parts of these names matter. The answer, in short: the nonlinearity matters little, and the **gate** matters.

::slide 21 | ReLU, FF(x) = max(0, xW₁)W₂ (original transformer, T5, Gopher, Chinchilla, OPT); GeLU, FF(x) = GELU(xW₁)W₂ with GELU(x) = xΦ(x) (GPT-1/2/3, GPT-J, GPT-NeoX, BLOOM); SwiGLU/GeGLU (Llama, PaLM, T5 v1.1, most models after 2023)

Plain activations are fine. A ReLU FFN can train "a reasonably performant language model" (Chinchilla used one). **GeLU**, $x\,\Phi(x)$ with Φ the standard normal CDF, differs from ReLU only by "this tiny divot" just below zero, which changes the gradients near zero and little else; GPT-3 used it and was "a perfectly good large language model". But almost every credible modern model uses a gated unit.

::slide 22 | the gated construction: the "first part" max(0, xW₁) becomes max(0, xW₁) ⊗ (xV), giving FF_ReGLU(x) = (max(0, xW₁) ⊗ xV)W₂; "note that we have an extra parameter (V)"

A **gated linear unit (GLU)** changes only the first half of the FFN. Next to the activated branch $\max(0, xW_1)$, compute a second, purely linear projection $xV$ of the same size, and multiply the two entrywise. Each hidden unit is then scaled by a learned, input-dependent factor before $W_2$ projects back down. The motivation is a general heuristic of architecture design, "gating is often very helpful".

$$ \mathrm{FF}_{\mathrm{ReGLU}}(x) = \big(\max(0,\, xW_1) \otimes xV\big)\,W_2 $$

::slide 23 | GeGLU, FFN = (GELU(xW) ⊗ xV)W₂ (T5 v1.1, mT5, LaMDA, Phi-3, Gemma 2/3/4); SwiGLU, FFN = (Swish(xW) ⊗ xV)W₂ with swish(x) = x·sigmoid(x) (LLaMA 1/2/3, PaLM, Mistral, OLMo, most models post 2023); "Gated models use smaller dimensions for the d_ff by 2/3"

The names compose: the activation on the first branch, plus "GLU". ReLU gives ReGLU, GeLU gives **GeGLU**, and **swish**, $x \cdot \mathrm{sigmoid}(x)$, gives **SwiGLU**. The split follows lineage. Google's models (T5, Gemma) use GeGLU; "everything that's a Llama descendant", and PaLM, uses SwiGLU. SwiGLU is the more common, "but honestly, amongst the gated units, doesn't really matter."

The footnote is the piece of trivia the professor flags as important later. The gate adds a third matrix V of the same shape as $W_1$, so at equal $d_{ff}$ a gated FFN has 3/2 the parameters of a plain one. To compare fairly, gated models shrink $d_{ff}$ by 2/3: three matrices of width $\tfrac23 d_{ff}$ hold as many parameters as two of width $d_{ff}$.

::worked gated-activations
::widget fixture:lecture_03--layer-budget | tick the gate: a third green FFN matrix appears and at the same d_ff the FFN grows to 1.5× its two-matrix size, while the attention segments do not move
::predict gated-activations
::note slip 23:33 | The professor says "if you take a SwiGLU, which is x times a sigmoid, then you will get a SwiGLU". The function x·sigmoid(x) is swish; SwiGLU is the gated unit built from it (slide 23).

::slide 24 | Shazeer 2020's table on T5-style fine-tuning: average score ReLU 83.80, GELU 83.86, Swish 83.60, GLU 84.20, GEGLU 84.12, SwiGLU 84.36, ReGLU 84.67, against a run-to-run standard deviation of 0.235

Does it work? "Yes, fairly consistently so." In Noam Shazeer's original paper the gains are small, but consistent: GLU, GEGLU, SwiGLU and ReGLU each beat all three plain activations on the average score, by about 0.3-0.9 points over ReLU, against a standard deviation of 0.235 across replicate runs. (The one gate without any activation, the bilinear row at 83.79, does not.) Two things make the comparison credible, the professor says. Shazeer trains replicates and reports error bars, and every comparison is **parameter-matched** through the 2/3 adjustment, so the gate is not winning by having more weights.

::slide 25 | Narang et al.'s comparison at 223M parameters: final loss vanilla (ReLU) 1.838, GeLU 1.838, Swish 1.847, GLU 1.814, GeGLU 1.792, ReGLU 1.803, SwiGLU 1.789, LiGLU 1.798, with matching gains on SGLUE, XSum and WebQ

A second, larger comparison from Google corroborates it: the gated variants reach final losses of 1.79-1.81 against 1.84-1.85 for GeLU, Swish and ReLU, and score higher downstream. (These are encoder-decoder T5-style models, not autoregressive LMs, the professor notes.) Together with "a lot of model training runs" since, the conclusion is that gating "gives you a nice boost without much of a computational cost".

::slide 26 | gating summary: many variations; *GLU isn't necessary for a working model (see GPT-3), but it's rare to see others; an outlier, Nemotron 340B, uses squared ReLU; evidence points toward somewhat consistent gains from Swi/GeGLU

Gates are not *necessary*: GPT-3 used a plain GeLU, and Nemotron 340B uses a squared ReLU, "a crazy choice, but that works too". It is just rare now to see a model without one.

::note slip 30:46 | On the summary table the professor says the activation column is "almost always" GLUs "with the exception of things like Falcon, which use a gated linear unit". The sentence contradicts itself; the exception he means is a non-gated model (Falcon uses GeLU).
::kp gated-activations

## Should attention and the MLP run side by side? {#parallel}
source: lecture_03.pdf p27-p29 · video 27:09-31:04, 40:20-41:25

::slide 27 | a standard serial block: Norm → causal multi-head self-attention → Dropout → Add, then Norm → position-wise feed-forward → Dropout → Add; "Could we parallelize the transformer block?"

A standard block is **serial**: attention first, then the MLP, which reads the stream *after* attention's update. A systems-minded reader sees a bottleneck: the MLP has to wait for attention. If the two ran side by side, new optimizations would open up.

::slide 28 | PaLM's description: serial y = x + MLP(LayerNorm(x + Attention(LayerNorm(x)))) against parallel y = x + MLP(LayerNorm(x)) + Attention(LayerNorm(x)); "roughly 15% faster training speed at large scales, since the MLP and Attention input matrix multiplications can be fused"; a small quality degradation at 8B, none at 62B; recent adopters Cohere Command A, Falcon 2 11B, Command R+

The **parallel** block, first used in GPT-J and then in PaLM and GPT-NeoX, computes both branches from the same normalized input and adds both to the stream:

$$ \text{serial: } y = x + \mathrm{MLP}\big(\mathrm{LN}(x + \mathrm{Attn}(\mathrm{LN}(x)))\big) \qquad \text{parallel: } y = x + \mathrm{MLP}(\mathrm{LN}(x)) + \mathrm{Attn}(\mathrm{LN}(x)) $$

Implemented right, it saves two things. The two branches read the same LN(x), so one LayerNorm can be shared. And the first matrix multiply of each branch (the Q, K, V projections and the MLP's up-projection) read the same input, so they can be **fused** into one larger matmul. PaLM's report, pasted on the slide, gives about 15% faster training at large scale, with a small quality loss at 8B parameters that disappears at 62B.

::widget fixture:lecture_03--residual-path | pick the parallel wiring: one LN feeds both Attn and MLP (1 norm per layer instead of 2), and "MLP reads this layer's attention output" turns to no
::predict serial-vs-parallel-blocks

The professor calls it "a really fun idea", but one the test of time has not favoured. Command A and other Cohere models (Cohere was founded by a former transformer author and follows many Google ideas) use it; few others do. His reasons, given aloud:
- **The serial block got fast enough.** Optimization of the serial form "has gotten sufficiently good" that the systems gain is no longer worth the cost.
- **The cost is depth.** The MLP never sees this layer's attention output, so "you've lost half of your depth".

In the Q&A a student asked how big the quality cost is. The honest answer is that nobody knows precisely: PaLM reported "no performance drop, 15% systems utilization improvement", but later Google models stopped using parallel layers, which he reads as "an implicit signal that actually, there might be some losses". As far as he knows, no one has run controlled ablations.

::video 28:59-29:23 | the two reasons parallel blocks faded: serial blocks got fast enough, and parallel halves effective depth
::note aside 41:13 | "This one is a little bit hard to get precise numbers on, because no one's done the ablations." PaLM's "no performance drop" is a simplification of its own report, which shows a small drop at 8B that vanishes at 62B.
::kp serial-vs-parallel-blocks

::slide 29 | architecture summary: everyone does non-residual norm (except OPT-350M); RMSNorm has clear compute wins, sometimes even performance; GLUs are consensus; most models use serial layers; beside it, the model table coloured by these columns

The summary table makes the trends visible: RMSNorm (blue) dominates recent rows, the parallel column is mostly serial, almost every recent activation is a GLU. The professor's takeaway is that the original transformer "has somewhat stood the test of time": within dense attention models, the changes so far are where the norm goes, which norm, biases, and the gate. Changes to attention itself, such as state-space layers, wait for the next lecture.

::note deferred 30:04 | Transformer alternatives that change attention (state-space models) are left to the next lecture: "today, I'm just only going to cover core attention-based methods".

## How does a transformer know where a token is? {#position}
source: lecture_03.pdf p30-p35 · video 31:04-43:36

The one architectural component that is "still in flux" across models, the professor says, is how position enters attention. Position has to be added somehow, because attention on its own is **position-independent**: it is a set of inner products between tokens, so shuffling the tokens leaves every score unchanged.

::slide 30 | four families: sine embeddings, Embed(x, i) = v_x + PE_pos (original transformer); absolute embeddings, Embed(x, i) = v_x + u_i (GPT-1/2/3, OPT); relative embeddings, a vector added inside attention, e_ij = x_i W^Q (x_j W^K + a^K_ij)ᵀ / √d_z (T5, Gopher, Chinchilla); RoPE (GPT-J, PaLM, LLaMA, most 2024+ models)

Four families have been used:
- **Sinusoidal** (the original transformer): add a fixed vector of sines and cosines of the position to the token embedding, with a "Fourier transform intuition" that position can always be recovered from it.
- **Absolute** (GPT-1/2/3, OPT): add a *learned* vector $u_i$ for each position $i$.
- **Relative** (T5, Gopher, Chinchilla): leave the embeddings alone and add a term to the attention computation itself that depends on the offset, so "if you're three positions off, the attention matrix gets a different offset added to it".
- **RoPE** (rotary position embeddings): rotate the queries and keys by an angle proportional to their position. It is now dominant, used by most models since 2024, and "in some ways, came out of nowhere": a GPT-J-era idea from a little-known blog post and paper.

::slide 31 | the goal: a relative embedding f(x, i) with ⟨f(x, i), f(y, j)⟩ = g(x, y, i − j); sine has cross-terms that are not relative, ⟨Embed(x, i), Embed(y, i)⟩ = ⟨v_x, v_y⟩ + ⟨PE_i, v_y⟩ + …; absolute is "obviously not relative"; the relative scheme's e_ij "is not an inner product"

RoPE starts from an opinionated stance: the model should not care about the *absolute* position of words, only about how far apart they are. If "a" and "apple" appear together, the attention between them should be the same at the start of a document or at the end. Written as a requirement: there is a function $f$ that embeds a word with its position, and the inner product of two such embeddings depends on the positions only through their difference,

$$ \langle f(x, i),\, f(y, j)\rangle = g(x, y,\, i - j). $$

None of the earlier schemes meets it. An additive sinusoid expands into four terms, $\langle v_x, v_y\rangle + \langle v_x, PE_j\rangle + \langle PE_i, v_y\rangle + \langle PE_i, PE_j\rangle$. The last depends only on $i-j$ for sinusoids, but the two cross-terms each depend on one absolute position. Absolute embeddings are absolute by construction. The relative scheme is relative, "but they're not embeddings": it adds a term to the attention score, so there is no $f(x,i)$ whose inner product produces it.

::predict position-embedding-taxonomy

::slide 32 | the rotation idea in 2D: "we know that" with "we" rotated by 0 positions and "know" by 1; "of course we know" with "we" rotated by 2 and "know" by 3; the angle between the two arrows is the same

The trick uses a fact about angles: **inner products are invariant to rotation**. Rotate both vectors by the same angle and their inner product does not change. So start from position-free word vectors and rotate each one by an angle proportional to its position. In "we know that", "we" is at position 0 (no rotation) and "know" at position 1 (rotated by θ). In "of course we know", "we" is at position 2 and "know" at 3, rotated by 2θ and 3θ. Both words moved, but the angle between them is still θ, so their inner product is the same as before. Only the offset survives.

::animation fixture:rope-shared-shift | shifting both positions by the same amount rotates both arrows together and the dot product does not change; rotating only one of them changes it

::slide 33 | the block-diagonal choice (Su et al. 2021): split the d-dimensional query or key into d/2 pairs (x₁, x₂), (x₃, x₄), …, rotate pair i by mθᵢ at position m; at the right, a partially rotated query that rotates only the high-frequency pair ("Gemma 4 alternative: just first 2")

In two dimensions there is only one way to rotate. In $d$ dimensions there are infinitely many, so RoPE does "the simplest possible thing": cut the vector into $d/2$ pairs of coordinates and rotate each pair in its own plane, pair $i$ by angle $m\theta_i$ at position $m$. The paper motivates this with complex numbers, but the professor finds it easier to think of it as many independent 2D rotations. The frequencies differ by pair, $\theta_i = 10000^{-2(i-1)/d}$. Fast pairs capture whether two tokens are neighbours; slow ones capture long-range offsets.

**Worked numbers.** With head dimension $d = 128$ there are 64 pairs. Pair 1 has $\theta_1 = 1$ radian per position, so it turns a full circle every $2\pi \approx 6.3$ positions. Pair 64 has $\theta_{64} = 10000^{-126/128} \approx 1.15\times10^{-4}$, a full circle every $2\pi/\theta_{64} \approx 54{,}000$ positions. Each pair's frequency is $10000^{-2/128} \approx 0.866$ times the previous one's.

::widget fixture:rope--frequency-dials | raise the position m: the first pairs spin many turns while the last ones have barely moved; the frequencies fall geometrically from pair to pair

The small figure on the right is a curiosity from Gemma 4, released the week of the lecture: **proportional RoPE** (p-RoPE), which rotates only part of the vector, in the figure the high-frequency pair, and leaves the rest carrying "only semantic information". In the Q&A the professor gave the argument: the low-frequency pairs "just aren't rotating very much" anyway, so you can leave them out "if you're really strapped for extra space", which makes it "really an optimization for teeny tiny models".

::slide 34 | the actual math: f_{q,k}(x_m, m) = R^d_{Θ,m} W_{q,k} x_m with R a block-diagonal matrix of 2×2 blocks [cos mθᵢ, −sin mθᵢ; sin mθᵢ, cos mθᵢ]; "Difference with sine embeddings: not additive, no cross terms"

In matrix form the rotation is a block-diagonal matrix of 2×2 rotation blocks, applied to the projected query and key: $q_m = R_{\Theta,m} W_q x_m$ and $k_n = R_{\Theta,n} W_k x_n$. Since $R_m^\top R_n = R_{n-m}$, the score $q_m^\top k_n = x_m^\top W_q^\top R_{n-m} W_k x_n$ depends on positions only through $n - m$. It uses sines and cosines like the original encoding, but *multiplies* by them rather than adding them, so there are no cross-terms and no absolute position can be read out of an inner product.

::slide 35 | the implementation (Hugging Face style): the usual q/k/v projections and reshapes into heads, then cos, sin = rotary_emb(value_states, position_ids) and query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin), then the usual attention; "embedding at each attention operation to enforce position invariance"

In code it is two lines inside the attention function: compute the cosines and sines for each token's position id, and apply them to the queries and keys (as a sparse matrix multiply or, cheaper, elementwise). Two details matter. It is applied **in every attention layer**, not once at the input, so position invariance holds at every layer. And it is applied to queries and keys only, never to values, because only the $q \cdot k$ score needs to see position.

::note slip 42:13 | Answering a question, the professor says relative position is "applied both to the keys and values". RoPE is applied to queries and keys, as he said at 38:31 and as slide 35's code shows.
::kp position-embedding-taxonomy
::kp rope:rope-relative-position
::kp rope:rope-per-layer-application

### Questions after the architecture part
- **Higher-dimensional rotations?** The professor knows of no paper that uses them. Any rotation within a 2D plane of the space is a variant of RoPE; one could imagine other closed loops, but he has not seen it done.
- **Why is additive relative attention a problem?** It "can't be factorized as a product" $f(x, i)\cdot f(y, j)$. He calls that "more of an aesthetic problem": schemes that inject offsets into the attention matrix "do reasonably well"; they just have not become dominant.
- **How do you distil architecture knowledge from papers?** Look broadly enough to see a pattern, which is what this lecture does, and try things yourself at small scale. Reading any single report in isolation is hard, because "no single paper seems to give any full detail" any more.

::note aside 39:19 | "Higher dimensional rotation. It's a good question. I don't think so." The answers in this Q&A are the professor's impressions, not results.

## How wide should the feed-forward layer be? {#dff}
source: lecture_03.pdf p36-p41 · video 43:36-49:53

::slide 36 | hyperparameter questions from an intro NLP course: how much bigger is the feed-forward layer than the hidden size, how many heads and must they divide the hidden size, what vocabulary size; and: do people regularize these huge LMs, do they scale deep or wide?

Hyperparameters only start to matter when you actually have to instantiate a model. Then the space looks daunting, a high-dimensional search with no theory. The surprise of this part is how small the space of things people actually use is, and how forgiving most of these settings are.

::slide 37 | the feed-forward layer FFN(x) = max(0, xW₁ + b₁)W₂ + b₂ has two dimensions, d_ff and d_model; "d_ff = 4 d_model. This is almost always true. There's just a few exceptions."

The first consensus hyperparameter is the ratio of the FFN's hidden width $d_{ff}$ (the output dimension of $W_1$) to the model width $d_{model}$. It controls how rich the MLPs are. The rule is $d_{ff} = 4\,d_{model}$, "for whatever reason", and it "works remarkably well".

::slide 38 | exception #1, GLU variants: scaling 4 by 2/3 gives d_ff = 8/3 d_model; ratios PaLM 4, Mistral 7B 3.5, LLaMA-2 70B 3.5, LLaMA 70B 2.68, Qwen 14B 2.67, DeepSeek 67B 2.68, Yi 34B 2.85, T5 v1.1 2.5; "PaLM, LLaMA2 and Mistral are slightly larger"

The first exception is the one we already met: gated FFNs have a third matrix, so the 2/3 correction turns 4 into $\tfrac83 \approx 2.67$. The table shows exactly that. LLaMA 65B (the slide's "LLaMA 70B"), Qwen 14B and DeepSeek 67B sit at 2.67-2.68, Yi at 2.85, T5 v1.1 at 2.5. The arithmetic for a concrete width:

::worked dff-ratio
::widget fixture:block-params | switch the gate on at a fixed d_ff and the FFN parameters jump by 3/2; scale d_ff by 2/3 (the "SwiGLU 8/3×" preset) and they return to the ReLU count, while attention never moves
::predict dff-ratio

A real model rounds the result: LLaMA-7B, with $d_{model} = 4096$, uses $d_{ff} = 11008$ instead of 10923, a multiple of 256 slightly above the exact value (alignment matters for GPU kernels, see [L5](#/read/lecture_05)).

Why are LLaMA-2 70B and Mistral 7B at 3.5? The slide only says they are "slightly larger". The professor gives a reason aloud: their attention is cheaper. Both use grouped-query attention (GQA, taught at the end of this lecture), in which many query heads share a few key and value heads, so the K and V projections shrink. With the parameters saved in attention, "we can multiply this ratio by an arbitrary 1.33. And we'll get roughly 3.5", which "emphasizes the MLPs a little bit more". In the papers, he sums up, you find "either 2.6-ish or 3.5 for GLUs, or 4 if you're doing non-GLU models."

The accounting behind that reason, per layer and in units of $d_{model}^2$ (biases and norms ignored):
- **Attention** with $h$ query heads and $g$ key/value heads: Q and O stay $d \times d$, K and V shrink to $d \times \tfrac{g}{h}d$, so $2 + 2g/h$. Full multi-head attention ($g = h$) is 4.
- **A gated FFN** at ratio $r$: three $d \times rd$ matrices, so $3r$. At $r = 8/3$ that is 8.

LLaMA-2 70B has 64 query heads and 8 KV heads, so its attention costs $2 + 2\cdot 8/64 = 2.25$ instead of 4, freeing 1.75. A multi-head layer with SwiGLU at 8/3 totals $4 + 8 = 12$; the GQA layer with SwiGLU at 3.5 totals $2.25 + 10.5 = 12.75$, about 6% more. So the move is roughly, not exactly, parameter-neutral.

::widget fixture:lecture_03--layer-budget | lower the K/V heads g: only the orange K and V segments shrink; tick "spend the freed budget" and the green FFN widens until the layer is back at the dashed reference, and the readout gives that FFN ratio
::predict gqa-frees-ffn-budget
::note spoken 46:01 | The reason for LLaMA-2's 3.5 is only in the audio. The professor says "MQA"; LLaMA-2 70B and Mistral 7B use grouped-query attention with 8 KV heads (their papers, not the slides). LLaMA-2 7B and 13B keep full multi-head attention and use about 2.7, which fits his reading. 3.5 / (8/3) is 1.31, rounded aloud to 1.33.
::kp gqa-frees-ffn-budget

::slide 39 | exception #2, T5 11B: d_ff = 65,536 with d_model = 1024, "an astounding 64-times multiplier"; the T5 paper's reason: modern accelerators such as TPUs "are most efficient for large dense matrix multiplications"; other recent exceptions: Gemma 2 (8×), SmolLM / Gemma 3 / Gemma 4 (4×, GLU)

Most technical reports are "very boring": "we did Llama, but we changed one thing". Google is the exception, sometimes "very bold". T5 11B uses $d_{ff} = 65{,}536$ with $d_{model} = 1024$, a 64× multiplier with 128-headed attention. Its reason is a systems argument: "the bigger my matrix multiplies, the more efficient I can keep my hardware." Gemma 2 also pushes higher, to 8×. No other model has gone near 64.

::slide 40 | Kaplan et al. 2020, a 50M-parameter model: loss increase against d_ff / d_model on a log axis: about 0.7% at 0.5, essentially 0% from 1 to 4, about 1.7% at 8, about 5% near 25, about 8% near 50

Is 4 special? Kaplan's scaling-law paper swept the ratio, as a side panel. There is a **basin**: from about 1 to about 10, the loss is within a percent or two of the best, "very, very flat". Past 10 "your loss starts really shooting up". Every ratio in use, 2.5 to 4, sits comfortably in the basin, so the choice barely matters for quality.

::slide 41 | lessons: 4× and 2.66× have worked well for nearly all modern LLMs; T5 shows even 64× can work, so the choice is not written in stone; but T5 v1.1, the "improved" follow-up, went back to a standard 2.5 multiplier on GeGLU, so 64× is likely suboptimal

The lesson. The defaults are safe. Radical choices "can technically work" (T5 was "a fine model"), "but it's probably going to be compute-inefficient". The professor's favourite punchline: when Google built the improved T5 v1.1, they quietly went back to a standard 2.5 multiplier.

::video 47:17-47:36 | T5's systems argument for a 64× multiplier: bigger matmuls use the hardware better
::kp dff-ratio

## How many heads, and how wide is each? {#heads}
source: lecture_03.pdf p42-p43 · video 49:53-51:31

::slide 42 | consensus hyperparameter 2: head dim × num heads = model dim; a reminder slide from CS224n: we compute XQ ∈ ℝ^{n×d}, reshape to ℝ^{n×h×d/h} and transpose to ℝ^{h×n×d/h}, so the head axis acts like a batch axis and "the matrices are the same sizes"; "This doesn't have to be true: we can have head-dimensions > model-dim / num-heads. But most models do follow this guideline"

The canonical choice for multi-head attention: with $h$ heads, give each head dimension $d/h$, so the heads together are exactly as wide as one big head, $h \cdot d/h = d$. Then splitting into heads is a reshape, every projection is a square $d \times d$ matrix, and multi-head attention costs the same as single-head attention. The professor always found this "very strange" to teach, because nothing requires it. The query projection $W_Q$ maps $d_{model}$ to $h \times d_{head}$, whatever those are. Only the convenience of square matrices argues for ratio 1.

::slide 43 | num heads, head dim, model dim and ratio: GPT-3 96, 128, 12288, 1; T5 128, 128, 1024, 16; T5 v1.1 64, 64, 4096, 1; LaMDA 128, 128, 8192, 2; PaLM 48, 258, 18432, 1.48; LLaMA-2 64, 128, 8192, 1; Qwen 3.5 (27B) 24, 256, 5120, 1.2; "most models have ratios around 1, notable exceptions by some Google models"

Almost every model follows it. The exceptions are mostly Google's: T5 at 16 (128 heads of 128 on a 1024-wide stream) and LaMDA at 2. Recomputing each row from its own columns is a good habit, and it catches one entry:

::worked head-dim-times-heads
::predict head-dim-times-heads

How much does the ratio matter? The slides' summary says the convention has "low to no validation". Aloud, the professor is milder: it is "yet another forgiving hyperparameter", with "a couple ablations" showing "a pretty wide basin around one". Neither names the ablations.

::widget fixture:lecture_03--layer-budget | set T5's 128 heads × 128 with d_model 1024: the Q and O segments follow h × head_dim, not d_model, so each becomes 16 d² and h·d_head/d reads 16
::kp head-dim-times-heads

## Deep or wide? The aspect ratio {#aspect}
source: lecture_03.pdf p44-p46 · video 51:31-55:08

::slide 44 | d_model / n_layer by model: BLOOM 205, T5 v1.1 171, PaLM (540B) 156, GPT-3 / OPT / Mistral / Qwen / OLMo 3 128 ("sweet spot?"), LLaMA / LLaMA-2 102, Gemma 3 87, Gemma 4 61, T5 (11B) 33

The **aspect ratio** $d_{model}/n_{layers}$ is, to the professor, one of the most interesting hyperparameters conceptually. When you scale a model family up or down, you usually fix the aspect ratio and grow both dimensions, so this one number controls the whole depth-versus-width trade-off. Recent talk about reasoning might suggest models should be very deep, or very shallow for utilization. In fact they cluster: GPT-3, OPT, Mistral, Qwen and OLMo 3 all sit at 128, most others between about 100 and 200. GPT-3, for example, has $d_{model} = 12288$ and 96 layers: $12288/96 = 128$.

::predict aspect-ratio
::note slip 52:29 | The slide's T5 (11B) value of 33 does not match T5 11B, which has d_model 1024 and 24 layers per stack, a ratio of about 43; the professor's own table on slide 51 lists T5 (11B) at 43 and GPT-2 at 33 (1600 / 48). The 33 on this slide appears to be GPT-2's row. Aloud, the professor rounds the whole band to "about 100".

::slide 45 | "Extremely deep models are harder to parallelize and have higher latency": Tay et al. 2021 on the limits of depth (every layer must wait for the previous one; width parallelizes over thousands of devices), with a diagram of four layers on four GPUs passing activations forward and gradients backward

Why not go very deep? Because depth is sequential. Every layer has to wait for the previous one, so per-token latency grows with the number of layers. And parallelizing a deep model across GPUs means cutting it by depth, layer 0 on GPU 0, layer 1 on GPU 1 and so on. That is **pipeline parallelism**, "something that most people really, really do not want to deal with". Width is "much easier to parallelize": each matrix can be split across GPUs (**tensor parallelism**). Both are taught in [L7](#/read/lecture_07). So there are systems reasons to go wide, and maybe expressiveness reasons to go deep, and models end up around 100.

::slide 46 | the evidence: Kaplan et al. 2020's loss against aspect ratio for 50M, 274M and 1.5B-parameter models, flat from roughly 10 to a few hundred ("a wide range of architectures achieve similar performance"); Tay et al. 2021's loss and downstream accuracy against FLOPs for many depth/width variants

The evidence says the choice barely matters for quality. In Kaplan's sweep, for each model size the loss is flat over a wide band of aspect ratios, with a similar optimum around a hundred regardless of size. In Tay et al.'s experiments, as you sweep depth against width, "the only thing that matters, in some sense, is flops": models get better as FLOPs grow, whatever their shape. So the professor's conclusion: there is a forgiving band, and inside it you choose for systems utilization, not for expressiveness, which is hard to reason about anyway.

**Worked example.** Two 7B configurations reach the same loss: (A) 32 layers of width 4096, ratio 128; (B) 128 layers of width 2048, ratio 16. With loss tied, pick A. Every token in B passes through four times as many sequential layers, and splitting B across GPUs needs a deeper pipeline.

::video 53:00-53:29 | depth splits need pipeline parallelism, width splits need only tensor parallelism
::note deferred 53:02 | "If you cut up your layers, we'll talk about this in the systems lecture": pipeline and tensor parallelism are taught in the parallelism lectures.
::kp aspect-ratio

## How big should the vocabulary be? {#vocab}
source: lecture_03.pdf p47 · video 55:08-59:18

::slide 47 | monolingual models, 30-50k: original transformer 37000, GPT 40257, GPT-2/3 50257, T5/T5 v1.1 32128, LLaMA 32000; multilingual / production systems, 100-250k: mT5 250000, PaLM 256000, GPT-4 100276, Gemma 4 262144, DeepSeek 100000, Qwen 152064, Yi 64000; "Monolingual vocabs don't need to be huge, but multilingual ones do"

Vocabulary size (the number of distinct tokens, see [L1's tokenizers](#/read/lecture_01)) splits models cleanly into two groups. The early open models aimed to be good at English and used about 30,000-50,000 tokens. After LLaMA, many teams built multilingual or production systems, including closed ones like GPT-4, and those use 100,000-250,000. Google's models have the largest vocabularies; later LLaMA derivatives sit around 100,000 or more.

The obvious reason is coverage: a vocabulary has to cover many scripts and word forms, so multilingual models "really do need much larger vocab". The professor adds a second reason the slide omits. Scaling-law studies find that bigger models can make use of bigger vocabularies, and the right-hand column is also the column of bigger models: "No one's training large monolingual models anymore." Coverage and scale are confounded in this table.

A vocabulary is not free. The embedding and output matrices each have $V \times d_{model}$ entries. At $d_{model} = 4096$, going from 32,000 to 256,000 tokens adds $224{,}000 \times 4096 \approx 0.92$ billion parameters to the output matrix alone, and as many again to the input embedding if the two are not tied.

::predict vocab-size-trends
::note aside 56:48 | Asked about multimodal models: if images are tokenized, you need many more tokens, and open releases usually add a separate image tokenizer with its own, quite large vocabulary.
::note aside 56:30 | Our note on the table: "GPT 40257" looks like a typo. GPT-1 used 40,000 BPE merges and a vocabulary of about 40,478 tokens. Yi's 64,000 also sits between the two bands without comment.
::kp vocab-size-trends

### Comparing models with different tokenizers: bits per byte
A student asked how valid it is to compare models whose tokenizers differ. "That's not a hyperparameter question, but that's a good question." A model with a bigger vocabulary cuts the same text into fewer, longer tokens, so its **per-token** loss or perplexity is measured in different units. Comparing per-token numbers across tokenizers is meaningless: a tokenizer with short tokens produces many easy-to-predict tokens.

The fix starts from what a language model is: a probability distribution over strings. The total log-probability a model assigns to a fixed text can always be compared. Dividing it by a normaliser both models share, the number of **bytes** in the text, gives **bits per byte**. The professor names two conditions:
- **The tokenizer must not touch the sequence.** Old word-level tokenizers that dropped words or mapped them to an unknown token score a different, shorter string. Modern byte-level tokenizers are complete and can encode any string.
- **The normaliser must be the same.** For bits per byte it always is: the byte count of the text.

::predict bpb-tokenizer-comparison

Concretely, bits per byte = (bits per token) ÷ (bytes per token). A model at 3.2 bits per token with 4.0 bytes per token scores 0.80 bits per byte; one at 2.7 bits per token with 3.0 bytes per token scores 0.90. The second model has the lower per-token loss but is the worse model of the same text.

::video 58:02-58:31 | the two conditions: the tokenizer does not drop text, and the normaliser is the byte count
::note deferred 59:10 | A follow-up about perplexity and downstream performance across tokenizers was left unresolved: "We'll have to talk later, because I'm not sure I understand the question."
::kp bpb-tokenizer-comparison

## Do these huge models need regularization? {#regularization}
source: lecture_03.pdf p48-p51 · video 59:18-1:05:00, 1:24:14-1:24:54

::slide 48 | "Do we need regularization during pretraining?" Arguments against: there is a lot of data (trillions of tokens), more than parameters; SGD does a single pass over the corpus, which makes it hard to memorize

Machine Learning 101 says big models overfit and need dropout and weight decay. Pretraining breaks the premise. There is more internet text than any lab has FLOPs to train on, so a run rarely sees the same document twice, and a single pass of SGD is very unlikely to memorize its data. Overfitting is "not really a problem, almost ever" in compute-limited language modeling. Some people only look at training loss for this reason.

::slide 49 | dropout and weight decay by model: original transformer 0.1 / 0; GPT-2 0.1 / 0.1; T5 0.1 / 0; GPT-3 0.1 / 0.1; T5 v1.1 0 / 0; PaLM 0 / variable; OPT 0.1 / 0.1; LLaMA 0 / 0.1; Qwen 14B 0.1 / 0.1; "many older models used dropout during pretraining; newer models (except Qwen) rely only on weight decay"; footnote: papers often don't discuss dropout, which on open models closely matches not using it

What do people do anyway? Many older models used dropout 0.1. Newer ones mostly drop it, Qwen being the exception on the slide, but **weight decay** survives, typically 0.1 (GPT-2, GPT-3, OPT, LLaMA). That is "very mystifying": why penalize weight size in a model that cannot overfit? (Recent technical reports often do not mention these settings at all; the footnote warns that the table may not hold for closed models.)

::predict regularization-weight-decay

::slide 50 | Andriushchenko et al. 2023: left, validation loss against training loss for weight decay 0, 0.1 and 0.3, all on the diagonal ("it's not to control overfitting"); middle, training loss under a 10× cosine learning-rate decay, where the strongest weight decay (0.3) starts highest and ends lowest; right, the same with a constant learning rate

The answer, from Andriushchenko et al.: in LLM training, weight decay is **not a regularizer**, it is an optimization intervention. The left plot shows validation loss equal to training loss for every weight-decay setting: there is no overfitting gap to close. The middle plot shows what it does do. With a cosine learning-rate decay, the runs with stronger weight decay "start out slow, but they essentially end up converging to a much better minimum later". The right plot shows that the effect depends on the schedule: with a constant learning rate it largely goes away, which is closer to where Machine Learning 101 intuition comes from.

::video 1:02:10-1:02:49 | stronger weight decay is worse early and better late, and only under a decaying learning rate

This is the kind of interaction the professor says makes architecture "very strange and hard" to reason about from scratch, and why the course makes you train models. In the Q&A he sketched the mechanism loosely: weight decay, "shrinkage to 0", "might allow you to use a higher learning rate, or it might allow you to decay faster", and dropout has gone because "it doesn't really interact well with optimization". Later, asked whether any hyperparameter is changed during training, he named this one: people schedule weight decay "in concert with learning rate". Architecture hyperparameters cannot change mid-run without breaking the model.

::slide 51 | hyperparameter summary: the factor-of-4 rule (8/3 for GLUs) is standard, with some evidence; head dim × num heads = d_model is standard, but with low to no validation; aspect ratio has a wide range of good values (100-200), and systems concerns dictate the value; you still "regularize" LMs, but the effect is on optimization dynamics; beside it, the model table with MLP factor, aspect ratio, weight decay and dropout columns

The summary of the hyperparameter part: for the hairy-looking settings there are standard choices that "have worked well for everybody". Factor 4 for the FFN (8/3 if gated), heads that multiply out to $d_{model}$, an aspect ratio around 100-200, and weight decay because it helps optimization, while "you don't need the regularization at all".

::note aside 1:04:12 | Asked whether diffusion language models need different architectures, the professor said he had "not looked into enough": few large ones exist, many are retrofitted from Llama-like models, and he does not know what the optimal architecture from scratch would be.
::kp regularization-weight-decay

## How do you keep training from blowing up? {#stability}
source: lecture_03.pdf p52-p56 · video 1:04:59-1:13:52

::slide 52 | "Recently, lots of attention on stable training": training loss and gradient L2 norm over 600k steps for OLMo 0424 7B (blue, with frequent loss spikes and gradient-norm spikes up to 3) and OLMo 2 1124 7B (orange, smooth and lower); "Don't train models that look like the blue curve!"

Over the last few years the emphasis has moved from performance alone to **stability**. Most of the choices so far are forgiving: change them and quality barely moves. A run that blows up is different. Spikes like the blue curve can leave you with a worse model, or one that cannot be trained further, after millions of dollars of compute. The smooth orange run, OLMo 2, appears on the slides' lists for three of the tricks in this lecture: the non-residual post-norm (slide 13), the z-loss and QK-norm (below).

::slide 53 | "Beware of softmaxes! Softmaxes can be ill-behaved due to exponentials / division by zero", next to the transformer diagram with its two softmaxes: inside attention and at the output

When a run is unstable, the usual suspects are the **softmaxes**. A softmax combines the two most dangerous operations in numerics: an exponential, which blows up quickly, and a division, which blows up when the denominator approaches zero. A language model has two of them: the **output softmax** that turns logits into a distribution over the vocabulary, and the **attention softmax** in every layer. The attention one is "especially" a danger zone.

### The output softmax: z-loss
::slide 54 | the softmax as log P(x) = U_r(x) − log Z(x) with Z(x) = Σ_r' e^{U_r'(x)}; Devlin 2014's loss L = Σᵢ [log P(xᵢ) − α (log Z(xᵢ) − 0)²]; PaLM's report: "an auxiliary loss of z_loss = 10⁻⁴ · log² Z to encourage the softmax normalizer log(Z) to be close to 0, which we found increases the stability of training"; other examples: Baichuan 2, DCLM, OLMo 2, OLMo 3

The professor's argument, given aloud while the formula is an image: the log-probability of token r is $U_r - \log Z$. The first term, the logit, is the output of the residual stream, and is well behaved if the model is. The second, the log of the normaliser $Z = \sum_{r'} e^{U_{r'}}$, "might not be so OK": if Z gets very big, or very close to 0, it blows up even when the logits look reasonable. Ideally Z stays near 1, $\log Z$ near 0.

The softmax is **over-parametrized**: adding the same constant to every logit changes Z but not a single probability. So the model can be asked to keep $\log Z$ near 0 at no cost to its predictions. The **z-loss** (Devlin 2014) adds a penalty $\alpha \log^2 Z$, which PaLM used with $\alpha = 10^{-4}$:

::worked z-loss
::widget fixture:lecture_03--z-loss | shift every logit: log Z, and the dot on the penalty parabola, slides along while the probabilities under the bars never change; the penalty bottoms out at log Z = 0
::predict z-loss

It became popular again through open models. Baichuan was, the professor thinks, the first open model to use it, then DCLM, OLMo and others. (OLMo 3's settings on slide 66 list a z-loss weight of $10^{-5}$.)

::note slip 1:08:40 | "This is from Jacob Devlin's paper, 2024, sorry, 2014": self-corrected.
::kp z-loss

### The attention softmax: QK-norm
::slide 55 | QK-norm: in the multi-head attention diagram, LN boxes on the queries and keys after the QKV projection, before the softmax; "the query and keys are Layer (RMS) normed before going into the softmax operation"; over it, a cartoon: "STACK MORE LAYER Norms"; other examples DCLM, OLMo 2, Gemma 2, Qwen3, OLMo 3, Gemma 4; originally from vision and multimodal models (Dehghani 2023, Idefics, Chameleon)

For attention, the professor returns to the heuristic from the norm section: "if you have instability, if you can throw a layer norm in there, somehow, it might control it." (The cartoon on the slide says the same thing more rudely.) Normally the block's input is pre-normed, projected into Q and K, and their dot products go into the softmax. **QK-norm** adds an RMSNorm (or LayerNorm) on the queries and keys themselves, after the projection and right before the dot product. Then the inputs to the softmax always have a scale of roughly one, because the norm has divided out the size of the Qs and Ks.

::worked qk-norm

The pre-norm alone does not do this. It bounds the block input, but $W_Q$ and $W_K$ can still grow and produce huge q and k, and then attention can collapse onto a single position (a near one-hot softmax). QK-norm came from the multimodal world (Idefics, Chameleon, and ViT-22B before them) and is now "a very standard intervention that most of the large models now introduce". It "doesn't seem to affect performance" across many runs, but it "does definitely prevent the kinds of attention degeneracies".

The professor sees one line of development in the three norm tricks: first a norm before each sub-block (pre-norm), then norms after each sub-block (double norm), and now norms on the queries and keys.

::predict qk-norm
::kp qk-norm

### Logit soft-capping
::slide 56 | Gemma 2's report: logits are capped in each attention layer and the final layer by logits ← soft_cap · tanh(logits / soft_cap), with soft_cap 50.0 for self-attention and 30.0 for the final layer; "Prevents logits from blowing up, but also might have perf issues?"; below, a perplexity table (±0.1 at 95%): bf16 baseline 11.19, soft_cap 11.24, QKV_norm 10.85, QK_norm_cap 11.00, QK_norm 10.84, QK_FC_norm 10.87

The third trick is a harder intervention and "more of a Google-specific trick". QK-norm controls the *inputs* to the softmax "and hopes that the outputs are well-behaved". **Soft-capping** caps the logits that go into the softmax directly: $z \mapsto c \tanh(z/c)$, which is close to the identity for small logits and saturates at $\pm c$. It is "almost a hard constraint", only smooth.

::worked logit-soft-capping
::widget fixture:lecture_03--soft-cap | small logits stay on the dashed identity line; past about c the tanh curve flattens, so two large logits end up almost level: a small gap survives, where a hard clip would leave none
::predict logit-soft-capping

The slide asks whether it "might have perf issues". The professor answers yes. The table at the bottom is from the systematic comparison he credits to "some NVIDIA folks". Against a baseline perplexity of 11.19, soft-capping alone is slightly worse (11.24), while QK-norm is better (10.84), because it lets you "crank up the learning rate a little bit". The reason soft-capping costs quality: "you can never express very confident signals in your softmax beyond a certain point."

::video 1:13:28-1:13:41 | the answer to the slide's question: soft-capping alone costs quality, because confidence is capped
::note slip 1:12:54 | "I think Gemma 2, 3, and 4 all use the logit soft cap trick." Gemma 2 does (slide 56); the Gemma 3 report says it replaced soft-capping with QK-norm.
::kp logit-soft-capping

## Why share keys and values across heads? MQA and GQA {#gqa}
source: lecture_03.pdf p57-p63 · video 1:13:52-1:25:08

::slide 57 | attention heads: most models barely touch them, with a few exceptions: GQA/MQA (saving inference costs by reducing the number of heads), sparse or sliding-window attention (restricting the pattern to cut compute), and "exotic SSM stuff (Jamba, Falcon 3, Qwen 3.5, etc): next lecture!"

The last part is about the attention operation itself, but only within dense, "all by all" attention. Two changes are now standard: **grouped-query attention**, which cuts inference cost, and **sliding-window attention**, which cuts long-context cost. State-space models and linear-time attention wait for the next lecture.

::slide 58 | the compute in attention for training: XQ times KᵀXᵀ gives h score matrices of n × n, softmax, times XV, output n × d; with d = hidden dim, b = batch, n = length (< d), h = heads, k = head dim (d/h); total arithmetic bnd², total memory accesses bnd + bhn² + d² (activations, softmax, projections); arithmetic intensity O((1/k + 1/(bn))⁻¹), high

To see why heads matter, switch from training to **serving**. Serving pays for two resources: the FLOPs, and the memory accesses, which set latency and utilization. Their ratio is the **arithmetic intensity** from [L2](#/read/lecture_02), and a GPU is only busy when it is high.

In training (or **prefill**, processing a user's prompt), all n positions are processed at once. The arithmetic is of order $bnd^2$. The memory traffic is $bnd$ for the activations, $bhn^2$ for the softmax's score matrices, and $d^2$ for the projection weights. Dividing:

$$ \frac{bnd^2}{bnd + bhn^2 + d^2} = \Big(\frac1d + \frac{n}{dk} + \frac{1}{bn}\Big)^{-1} = O\Big(\Big(\frac1k + \frac{1}{bn}\Big)^{-1}\Big) $$

using $k = d/h$ and $n < d$. Read the terms: head dimensions need to be big enough to multiply "reasonably sized matrices", and batch times sequence length needs to be large. Both are true in training, so "your GPUs are going to be fully utilized".

::slide 59 | the incremental case, generating text: "can't parallelize the generation process, needs to be step by step"; attention is updated incrementally through the KV cache; an animation frame of one step with and without the cache: the new query token against cached keys and values

Generation is different. Tokens come out one at a time, each conditioned on the last: "the curse of autoregressive language modeling". The efficient way is a **KV cache**: keep the keys and values of every past token, and at each step compute only the new token's query, key and value, then attend from the new query over the cache ([L10](#/read/lecture_10) covers the mechanics). This saves a lot of compute. But every step now has to read the whole cache, and the weights, once per generated token.

::slide 60 | the incremental arithmetic intensity: total arithmetic bnd², total memory accesses bn²d + nd² (K, V and projections); intensity O((n/d + 1/b)⁻¹), "not good": it needs large batches and short sequences, or big model dimensions; "The n/d term is difficult to reduce"

Over a whole sequence the arithmetic is the same, $bnd^2$, done incrementally. The memory traffic is not: the cache, of size $bnd$, is re-read at each of the n steps, giving $bn^2d$, and the weights are re-read at each step, giving $nd^2$. Now

$$ \frac{bnd^2}{bn^2d + nd^2} = \Big(\frac{n}{d} + \frac1b\Big)^{-1}. $$

The $1/b$ term says batch more users together. The $n/d$ term is the problem: it does not shrink with batch size, and it means "if we want to serve a small model efficiently, this is not so good."

::worked mqa-gqa-kv-cache

::slide 61 | MQA: "have multiple queries, but just one dimension for keys and values"; the diagram: one shared K and one shared V per layer, broadcast to all query heads, so the K and V caches are a single head wide; total memory access bnd + bn²k + nd², arithmetic intensity O((1/d + n/(dh) + 1/b)⁻¹)

**Multi-query attention** (MQA, Shazeer 2019) attacks the $n/d$ term. Keep all h query heads, but give the whole layer a single key head and a single value head, shared by every query head. The cache shrinks h-fold, to one head of width k, the term becomes $bn^2k$, and intensity becomes $O((1/d + n/(dh) + 1/b)^{-1})$: the problem term is divided by the number of heads.

**Worked numbers**, in the slides' units (FLOPs per value moved, constants dropped), for $d = 4096$, $h = 32$, $k = 128$, $n = 2048$, $b = 64$:
- training or prefill: $(1/4096 + 2048/(4096\cdot128) + 1/(64\cdot2048))^{-1} \approx 240$;
- decoding with multi-head attention: $(0.5 + 1/64)^{-1} \approx 1.9$, and $(0.5 + 1)^{-1} \approx 0.67$ for a single user;
- decoding with MQA: $(1/4096 + 2048/(4096\cdot32) + 1/64)^{-1} \approx 32$.

Decoding drops intensity by two orders of magnitude, and sharing K and V recovers a factor of about 16.

::predict mqa-gqa-kv-cache
::note slip 1:17:23 | The professor describes the KV cache as keeping "all of the passkeys and queries" and "this matrix of Q dot K over the past". The cache stores the past keys and values (slide 59). Past queries are never needed again: each step computes one new query and its row of scores against the cached keys. He also self-corrects "number of heads, sorry, head dims" (1:16:38) and "reduces, sorry, increase the arithmetic intensity" (1:20:10).

::slide 62 | GQA: "don't go all the way to one dimension of KV, have fewer dims": multi-head (8 K/V heads for 8 queries), grouped-query (4 K/V heads, each shared by 2 queries), multi-query (1 K/V head for all 8); "a simple knob to control expressiveness (key-query ratio) and inference efficiency"; more recently, MLA (multi-head latent attention) from DeepSeek V2

The catch is that MQA loses expressive power: one key and one value for all queries is a strong restriction. **Grouped-query attention** (GQA) is the middle ground. Keep h query heads and use g key/value heads, each shared by a group of h/g queries. With g = h it is ordinary multi-head attention; with g = 1, MQA. The KV cache shrinks by h/g, and the cache term in the intensity by the same factor. In the example above, g = 8 gives intensity $\approx 7$. For LLaMA-2 70B (80 layers, 64 query heads of 128, 8 KV heads) the cache holds $2 \times 8 \times 128$ values per layer per token, 320 KB in bf16 across all layers, against 2.5 MB for full multi-head attention. At 4096 tokens that is about 1.3 GB per sequence instead of 10.7 GB.

::widget fixture:lecture_03--layer-budget | as g drops, the K/V-read bar shrinks by h/g, but the q·k FLOPs bar does not move: GQA and MQA cut the bytes read per decoded token, not the score FLOPs

::slide 63 | "Does MQA hurt? Sometimes": Shazeer 2019, billion-word dev perplexity: multi-head 29.9, multi-query 30.2 (with d_ff raised from 8192 to 9088), fewer or narrower heads 30.9-31.2; Ainslie 2023: performance against time per sample, MHA-XXL about 47.2 at 1.5 ms, GQA-XXL about 47.1 at under 0.3 ms, MQA-XXL about 46.6, MHA-Large about 46.0; time per sample against GQA groups, flat from 1 to 8 groups and rising to the MHA level at 64

The evidence. MQA costs a small perplexity hit, 30.2 against 29.9 in Shazeer's table. Notice the $d_{ff}$ column: the MQA model gets a wider FFN (9088 instead of 8192) to make up the parameters saved in K and V, the same move as LLaMA-2's 3.5 ratio. GQA does better: in Ainslie et al., GQA-XXL is nearly as good as full multi-head XXL at about a fifth of the time per sample, and much better than a smaller multi-head model of similar cost. The time plot shows why a small number of groups is enough: inference time stays near MQA's up to about 8 groups. In the professor's words, "a small reduction in the number of heads" keeps most of the quality while "getting significant inference improvements", which is why almost every model today uses GQA.

::video 1:22:36-1:22:54 | a small reduction in KV heads keeps most of the quality and most of the inference gain
::note spoken 1:24:55 | Asked whether MQA/GQA is only an inference-time switch: no, "you train with a certain number of repeats". The KV-head count is fixed at training time, and the K and V projections are smaller, which is why GQA saves parameters as well as cache.
::note deferred 1:21:40 | Multi-head latent attention (MLA, DeepSeek V2), a different factorization with different trade-offs, is left to the next lecture; inference mechanics to Percy's inference lecture.
::kp mqa-gqa-kv-cache

::note aside 1:23:14 | Asked how much labs still search hyperparameters rather than reuse rules of thumb: a mix, but reports usually change one architecture thing at a time. Google "seems to really spice things up"; Gemma 4, for example, adds a separate embedding for every layer to trade memory against FLOPs.

## How do you afford long context? Sliding windows and interleaving {#swa}
source: lecture_03.pdf p64-p66 · video 1:25:08-1:28:37

::slide 64 | "Attending to the entire context can be expensive (quadratic)": sparse or structured attention trading expressiveness against runtime (GPT-3, GPT-OSS, Gemma 4); Child et al. 2019's attention masks: (a) full causal attention, (b) strided sparse attention, (c) fixed sparse attention

Full attention compares every position with every earlier one, so its cost grows with the square of the context length. **Sparse** attention restricts which positions each query may see. The simplest restriction is a **sliding window** (SWA): each position attends only to the last w positions, a band along the diagonal. Its cost per layer grows linearly with the context. It is an old idea. GPT-3's paper already alternates full attention, "where every position can attend to everyone in the past", with banded attention within a fixed window, following OpenAI's earlier work on sparse patterns.

::slide 65 | Cohere Command A: transformer blocks 1-3 sliding-window self-attention (grouped-query attention, RoPE positional embeddings), block 4 full self-attention (grouped-query attention, no positional embeddings), "interleaved SWA and full attention (3:1 ratio)", MLPs with SwiGLU and no bias terms; "Long-range info via NoPE, short-range info via RoPE + SWA"; other models: LLaMA 4, Gemma 3, Gemma 4, OLMo 3 do SWA + full RoPE

What has become "really, really popular over the past year" is to **interleave** the two. Command A, the first recent open model the professor saw doing it, makes every fourth layer full attention and the three in between sliding-window. This "hits a sweet spot" between long-context quality and inference cost, without resorting to a state-space model.

::predict sliding-window-interleave

Command A adds a twist in the position embeddings. The local layers use RoPE, so within the window they know exactly how far apart tokens are. The full layers use **NoPE**, no position embedding at all: they see the long context "almost, at bags", matching by content alone, while "the short range information still gets position information". LLaMA 4, Gemma 3, Gemma 4 and OLMo 3 interleave the same way but keep RoPE in the full layers too.

How does a local layer use anything outside its window? Through depth. A sliding-window layer reads only its window, but each vector in that window is the residual stream, which already contains what earlier layers gathered. Once a full-attention layer has written long-range information into every position, the local layers above it can use it: "as you go down these blocks, you're aggregating local information into global ones. So the local attentions at the end can, of course, access more global information".

::animation fixture:lecture_03--swa-reach | each local layer pushes token 1,000's information one window further while the window it reads stays one window wide; the full layer at layer 4 lights the whole row, so layer 5's window at 60,000 is made only of carriers
::predict swa-reach-through-depth

Even a stack with *no* full layers reaches beyond one window: each local layer can move information one window further back, so k layers of window w reach about k × w tokens. With a 4,096-token window, three local layers reach about 12,000 tokens past a given token, while each layer's attention still costs only a window's worth. The full layers make the reach immediate instead of gradual.

::video 1:26:17-1:26:44 | local information aggregated into global through the blocks, so the last local layers see more global information
::kp swa-reach-through-depth

::slide 66 | more interleaved designs (from Maarten Grootendorst's visual guide): Gemma 4 with several local sliding-window layers per global layer, p-RoPE, shared keys and values, and a last layer that is always global; OLMo 3's settings: sliding-window attention on 3/4 of layers with 4,096 tokens, YaRN RoPE scaling on the full-attention layers, z-loss weight 10⁻⁵; Qwen 3.5 / Qwen3-Next: three Gated DeltaNet layers for every gated-attention layer, with mixture-of-experts MLPs

The other designs on the slide all follow the pattern, with a cheap layer and an expensive one alternating. OLMo 3 puts a 4,096-token window on three quarters of its layers. Qwen 3.5 keeps one full-attention layer in every four, but its cheap layers are not sliding windows: they are a state-space layer called **Gated DeltaNet**, explained next lecture. Long context is, the professor says, "still an active area of investigation", and the place where most architecture changes are happening now.

::note deferred 1:28:08 | Qwen 3.5's Gated DeltaNet state-space layers are left to the next lecture.
::kp sliding-window-interleave

## What should you carry away? {#summary}
source: lecture_03.pdf p67 · video 1:28:37-1:29:10

::slide 67 | the full model table again: "Many aspects (arch, hparams) of transformers are in common across the big LMs. Major differences? Position embeddings, activations, tokenization"

Across all these models, a lot is shared, and the shared choices make good defaults:

| choice | the default | the main reason |
|---|---|---|
| norm placement | off the residual stream (pre-norm, sometimes plus a post-branch norm) | clean gradient path, stability, larger learning rates |
| norm type | RMSNorm, no bias | less data movement, same quality |
| FFN | gated (SwiGLU or GeGLU), $d_{ff} \approx \tfrac83 d_{model}$ | small, consistent gains at equal parameters |
| block | serial | parallel saves ~15% but costs depth |
| position | RoPE, in every attention layer | scores depend only on the offset |
| head width | $h \cdot d_{head} = d_{model}$ | convention; the basin is wide |
| aspect ratio | $d_{model}/n_{layers}$ about 100-200 | loss is flat; depth hurts latency and parallelism |
| vocabulary | 30-50k monolingual, 100-250k multilingual | coverage, and bigger models use bigger vocabularies |
| regularization | weight decay ~0.1, no dropout | helps optimization under a decaying learning rate |
| stability | z-loss, QK-norm | keep both softmaxes numerically tame |
| attention | GQA; interleaved sliding-window and full layers | smaller KV cache; long context at linear cost in most layers |

The differences that remain, per the slide and the professor's last words, are how models handle **context and position**, the **activation**, and the **tokenizer**. The other lesson is about method. Almost none of these defaults was derived. They were found by many teams training many models, and the reasons, often systems reasons, were understood afterwards. That is why the course asks you to train your own.

::kp arch-consensus-vs-variation
