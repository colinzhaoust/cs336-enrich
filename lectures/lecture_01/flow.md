---
title: L1 · Overview and tokenization, read through
minutes: 40
---
This lecture does two jobs. The first hour lays out what the course is for and the ten-week map: why you would build a language model from scratch, what knowledge carries over to frontier scale, and why every unit ends up being about efficiency. The last twenty minutes start the first unit, tokenization: you build four tokenizers, see why three of them fail, and train a byte-pair-encoding (BPE) tokenizer by hand. After it you can compute a tokenizer's compression ratio, explain the trade between vocabulary size and sequence length, and run BPE training and encoding on paper.

## Why build language models from scratch? {#why}
source: lecture_01.py:L52-L99 · video 0:04-7:09

This is the third offering of CS336, taught by Percy Liang and Tatsu Hashimoto with three course assistants. The 2025 lectures are on YouTube. What is new this year: the same "from scratch" philosophy, a stronger focus on the concepts with the most value per hour of your time, and more on modern ingredients such as mixture of experts, long context and agents.

::note aside 2:44 | "From scratch" is qualified at once: "we don't actually build everything up from scratch because that wouldn't fit in the quarter". Two years of refinement went into choosing which pieces are worth building yourself.

### The problem: researchers lost contact with the technology

The professor tells the last decade as three steps up a ladder of abstraction:
- **2016**: researchers implemented and trained their own models;
- **2018**: they downloaded a pretrained model such as BERT and fine-tuned it;
- **today**: they prompt an API model (GPT, Claude, Gemini).

Each step raised productivity. But these abstractions are *leaky*, unlike a programming language or an operating system: when a prompted model cannot do what you want, there is no recourse below the prompt. And fundamental research needs to "tear up the whole stack"; prompting alone confines you to a small part of the design space. Hence the course's thesis: full understanding is necessary for fundamental research, and the way to understand is to build.

### The catch: industrialization

There is one small problem. Building the real thing has become industrial:
- GPT-4 (2023) supposedly cost 100 million dollars to train; aloud, the professor guessed today's frontier runs cost on the order of a billion, flagging it as speculation.
- In 2025 xAI built a cluster of 230,000 GPUs to train Grok.
- And the details are secret. The GPT-4 technical report says outright that, given the competitive landscape and safety implications, it contains nothing about architecture, hardware, compute or data.

::figure official/lectures/images/gpt4-no-details.png | the GPT-4 report's own statement that it discloses nothing about how the model was built

So frontier models are out of reach. You can build small models (under 1B parameters), but a small model may not behave like a large one. The lecture gives two examples.

**Where the FLOPs go changes with scale.** Stephen Roller's table for the OPT model family shows the share of compute spent in the feedforward (MLP) layers rising from 44% at 760M parameters to 80% at 175B, while attention's share falls. Optimize attention at small scale and you are optimizing something that hardly matters at large scale.

::figure official/lectures/images/roller-flops.png | read the "% FLOPS FFN" column: 44% at 760M, 80% at 175B
::note deferred 6:01 | Why the MLP share grows with scale is left to the FLOP-counting lectures. In short: per layer, the MLP's FLOPs grow with the square of the hidden size, while the attention-score FLOPs grow with hidden size times context length; see [L2's FLOP counting](#/read/lecture_02).
::note slip 5:59 | Aloud the table is dated "back in 2021"; the tweet in the figure is from October 2022, and the OPT models it covers were released in 2022.

**Some behaviors only appear at scale.** In Wei et al. (2022), small models score at chance on tasks such as modular arithmetic or multi-task question answering, and only past a critical training compute (between about $10^{22}$ and $10^{23}$ FLOPs in several panels) does accuracy suddenly rise. Work only at small scale and you never see such phenomena.

::figure official/lectures/images/wei-emergence-plot.png | each panel is flat at the dashed "random" line, then jumps once training FLOPs pass about 1e22

## What can a small-scale course teach that transfers? {#efficiency}
source: lecture_01.py:L101-L123 · video 7:09-11:36

The professor splits knowledge into three kinds:
- **Mechanics**: how things work. What a Transformer is, how model parallelism works.
- **Mindset**: how to approach building a model. Squeeze the most out of the hardware, and take scaling seriously. Aloud: "profile and benchmark everything".
- **Intuitions**: which data and modeling decisions give good accuracy.

The first two transfer to large scale, and the course teaches them. Intuitions only partly do: what works at 100M parameters may not work at 100B, and "for that, you actually have to go somewhere where you can do things at scale".

Some intuitions are not even justified. Noam Shazeer's paper introducing the SwiGLU activation, now standard, ends by offering no explanation and attributing the success "to divine benevolence". Mechanics can be checked by construction; such design choices can only be found by experiment.

::figure official/lectures/images/divine-benevolence.png | the SwiGLU paper's conclusion: no explanation, only experiments

### The bitter lesson, read correctly

Rich Sutton's "bitter lesson" is usually summarized as: general methods that exploit computation win. The professor names a wrong and a right reading.
- **Wrong**: scale is all that matters, so algorithms do not matter.
- **Right**: algorithms *that scale* are what matter.

His slogan is

$$ \text{accuracy} = \text{efficiency} \times \text{resources} $$

where, aloud, "efficiency is output over input, and resources is the input". Nothing here is measured; it is a way to think. Efficiency multiplies resources: a recipe that is 4× more compute-efficient, given $10^{25}$ FLOPs, behaves like the old recipe given $4 \times 10^{25}$.

::predict efficiency-framing

Efficiency matters *more* at scale, not less. In a small experiment, a run that takes twice as long means you "just wait twice as long". At scale the same waste "could be hundreds of millions of dollars", so "even like a 5% improvement might be a big deal". Empirically, a 2020 OpenAI paper measured a 44× gain in algorithmic efficiency on ImageNet between 2012 and 2019: the compute needed to reach AlexNet's accuracy fell 44-fold, separately from the hardware getting faster. Multiply algorithmic and hardware gains together and you get the jumps the field has seen.

::code lecture_01.py:L115-L123 | the two readings, the slogan, and the framing question every later unit answers
::video 9:53-10:22 | why waste is tolerable at small scale but a 5% gain matters at scale

The course's organizing question follows: **what is the best model one can build given a certain compute and data budget?** In other words, maximize efficiency. For pretraining, the course mostly talks about the compute budget, "because we're going to assume that we have a lot more data than we have compute". If you had a fixed small dataset, or a warehouse of idle GPUs, you would be data-bound instead, and some answers change; the syllabus returns to this at its end.

::kp efficiency-framing

## How did language models get here? {#landscape}
source: lecture_01.py:L126-L185 · video 11:36-19:26

A short history, to place the course's ingredients.

**Pre-neural (before the 2010s).** Shannon used a language model in 1950 to measure the entropy of English. For decades n-gram models (probabilities of the next word given the previous few) were a component of machine translation and speech recognition, not the whole system but the part that kept output fluent.

**Neural ingredients (2010s).** The LSTM (1997); Bengio's first neural language model (2003), which was a feedforward network over a small context, not an LSTM; sequence-to-sequence models (2014), which boldly compressed a whole sentence into a vector; the Adam optimizer (2014); the attention mechanism (2015) and then the Transformer (2017), both developed for machine translation; mixture of experts (2017); and model parallelism (GPipe, ZeRO, Megatron-LM).

**Early foundation models (late 2010s).** ELMo (LSTMs) and BERT (Transformer) were pretrained on lots of text and then fine-tuned on downstream tasks such as question answering, with large gains. Google's T5 (11B) cast every task as text-to-text, foreshadowing "prompt in, response out".

**Embracing scaling.** OpenAI scaled GPT to GPT-2 (1.5B: fluent text, first signs of zero-shot ability), took scaling laws seriously (Kaplan et al., 2020: loss is predictable from scale), and trained GPT-3 (175B, more than 10× the largest model at the time), which showed in-context learning. Google answered with PaLM (540B), massive but *undertrained*: DeepMind's Chinchilla (70B) showed with compute-optimal scaling laws that, for the same compute, a smaller model on more tokens does better. That rule returns in the scaling section below.

**Open models.** After GPT-3, early replication attempts came from EleutherAI (The Pile dataset, GPT-J; small, for lack of compute), Meta's OPT (175B, a replication with many hardware problems) and BigScience's BLOOM (176B, focused on data sourcing). None was very strong. Since then Meta's Llama series, Mistral and a wave of Chinese models (DeepSeek, Qwen, Kimi, GLM, MiniMax, MiMo) have brought open weights close to closed models.

A further tier releases not just weights but paper, code and data: AI2's OLMo, NVIDIA's Nemotron, and Marin, the professor's own project, which develops in the open. The course depends on this ecosystem: its papers are the only glimpse into how frontier models are built, though even they omit much, notably the data mixture.

### What a language model is, and what has not changed

The answer to "what is a language model" kept moving:
- 2018 (BERT): something you fine-tune;
- 2020 (GPT-3): something you prompt;
- 2022 (ChatGPT): something you talk to;
- 2026 (agents): something that acts autonomously, given a page of instructions and left to run a long coding task.

::note skip 17:52 | The example agent trace was a link the classroom had no internet to open ("I guess I don't have internet"), so agents were described, not shown.

The fundamentals are the same: GPUs and kernels, stochastic-gradient-style optimization, the Transformer and attention. The specs differ: longer contexts are demanded, so inference efficiency matters even more. That is why the course barely had to change.

## How does the course work? {#logistics}
source: lecture_01.py:L188-L232 · video 19:26-27:17

### An executable lecture

The lecture is a Python program whose execution delivers the lecture. Its `text()` and `image()` calls render as the slides; the code between them really runs, and a viewer steps through it line by line, showing variable values (the `# @inspect` comments mark which). You can also see the program's structure: each section is a function called from `main()`, and this read-through follows that order.

::code lecture_01.py:L188-L195 | the lecture's own description of itself, with a tiny loop to step through

### Logistics

- It is a 5-unit class with five intense assignments. A course evaluation claimed the first assignment alone was as much work as all five CS 224n assignments plus the final project; the professor was told this is exaggerated, "but better to be conservative".
- **Take it** if you have an obsessive need to understand how things work, or want to build research-engineering muscle: enough systems depth that "everything else seems kind of easy".
- **Do not take it** if you need research done this quarter (talk to your advisor), want the hottest techniques (no multimodality, no RAG, no agents in any depth), or want good results in your application domain: then prompt or fine-tune a model, and pretrain only as a last resort.
- Following at home: all materials are online and lectures are recorded and later posted.

::note skip 22:20 | "We don't do multimodality. We don't talk about agents in any depth."

**Assignments.** Five of them: basics, systems, scaling laws, data, alignment. There is no scaffolding code, but there are unit tests and adapter interfaces, so you are not in a "sparse reward setting" where a submission is simply right or wrong. Implement and test locally on a laptop, then run on a cluster to measure accuracy and speed. Most assignments have a leaderboard of the form "minimize perplexity given a training budget".

**AI policy.** Coding agents can solve every assignment, and you would learn nothing by pasting the Assignment 1 PDF into one. AI is very useful for answering questions and tutoring, so the rule is: if you use AI, use it with the course's AGENTS.md prompt, which asks it to be pedagogically minded (explain, clarify, but not write the Transformer for you). It is a first-year experiment.

**Compute** is provided by Modal.

## Unit 1, basics: what does it take to train a language model? {#basics}
source: lecture_01.py:L235-L316 · video 27:17-35:53

The syllabus has five units, mirroring the five assignments: basics, systems, scaling laws, data, alignment. The professor gives "a taste" of each. The basics unit takes about the first two weeks, and its goal is to train a basic language model you built yourself, from three components: tokenization, model architecture, training.

### Tokenization

Tokenization answers: what are the atoms the model operates on? Formally, a tokenizer converts between raw inputs (bytes) and sequences of integers (tokens); conceptually, "it's a segmentation of the text".

::figure official/lectures/images/tokenized-example.png | encode turns "Stanford was founded in 1885." into integer ids, each covering one coloured chunk; decode turns them back

The popular choice is **Byte-Pair Encoding** (BPE), whose intuition is to break the input into frequently occurring chunks. Through the course's efficiency lens, tokenization buys two things:
- **Shorter sequences.** Roughly, 1000 bytes become about 250 tokens.
- **Adaptive computation.** A long, predictable stretch of bytes can become one token, while rare or interesting parts stay as several tokens, so the model spends more positions, and so more compute, where the content is.

Aloud the professor ranked these: the second is "more subtle, but maybe more importantly". The code only names it.

Why shorter sequences matter: a Transformer's attention compares every position with every other, so its cost grows with the square of the sequence length $n$. Going from 250 tokens to 1000 (raw bytes) makes the sequence 4× longer and the attention scores $4^2 = 16$ times more expensive. The rest of the model (projections, MLPs) grows only linearly, 4×, and at short context that linear part dominates, so 16× is the attention term only.

::predict tokenization-efficiency-lens
::widget fixture:lecture_01--vocab-trade | set tokenizer 1 to 1 byte per token and tokenizer 2 to 4: the length bar shrinks 4× and the attention bar 16×

The dream is a tokenizer-free architecture that operates directly on bytes (ByT5, MegaByte, BLT, T-Free, H-Net). These are promising but "have not yet been scaled up to the frontier", and since frontier models still use tokenizers, the course still teaches them. The professor adds that every year he hopes he will not have to.

::code lecture_01.py:L275-L280 | the efficiency lens and the tokenizer-free dream
::video 28:55-29:15 | adaptive computation called "maybe more important" than shorter sequences
::note deferred 29:15 | Tokenization is taught in full at the end of this lecture; the sections below carry it.
::kp tokenization-efficiency-lens

### Model architecture

The starting point is the original Transformer (2017), which you may know from CS 224n. Since then many refinements matter:
- activation functions: ReLU, then SwiGLU;
- positional encodings: sinusoidal, then RoPE;
- normalization: LayerNorm, RMSNorm, QK norm, pre-norm versus post-norm;
- attention: full attention is quadratic in sequence length, "and that gets really expensive", so there are sparse or local attention, grouped-query attention (GQA) and multi-head latent attention (MLA);
- recurrence, state-space models and linear attention (Mamba, Gated DeltaNet), often in a hybrid with attention, which "seems to work quite well";
- MLPs: dense, or mixture of experts, now "the dominant paradigm for building compute efficient transformers", which also needs its own training techniques;
- the shape: hidden dimension, depth, number of heads and experts. It sounds like a mere hyperparameter, but at scale it has "a huge, huge implication".

::figure official/lectures/images/transformer-architecture.png | the 2017 encoder-decoder Transformer; language models keep the decoder stack, and nearly every box here has since been refined
::note deferred 30:18 | Architecture is Tatsu's subject in the next lectures; see [L3](#/read/lecture_03).

### Training

How do you set the parameters? The decisions:
- the loss: next-token prediction by default; predicting several tokens ahead (multi-token prediction, used by DeepSeek-V3) helps;
- the optimizer: AdamW, and increasingly Muon (used, aloud, in the Kimi K2 models), or SOAP;
- initialization scale (Xavier, muP), which "sounds kind of boring" but strongly affects the stability of larger models;
- the learning-rate schedule (cosine, WSD), regularization (dropout, weight decay), batch size (critical batch size), and MoE load balancing.

You might treat these as knobs to try at random. The professor's point is the opposite: setting them in a principled way is "the difference between a run that just blows up and is useless" and one that is state of the art.

::note deferred 33:42 | How to set hyperparameters in a principled way returns with scaling laws, below.

### Assignment 1 and the three-way balance

Assignment 1: implement a BPE tokenizer, a Transformer, cross-entropy loss, the AdamW optimizer and the training loop; do resource accounting of where your FLOPs go; train on TinyStories and OpenWebText; and compete on a leaderboard to minimize OpenWebText perplexity in 45 minutes on one B200 GPU (like the nanoGPT speedruns).

The unit's high-level principle: tokenizer, model and training look like separate pieces, but every decision balances three things:
- **expressivity**: the model can represent complex dependencies in the data;
- **stability**: parameter and gradient norms stay in a "goldilocks zone", neither blowing up nor vanishing; "a lot of training language models is about just stability";
- **efficiency**: it runs fast on hardware, in training and inference. Many architecture changes are speed-ups, such as projecting to a lower-dimensional space, and the question is always whether they still work as well.

## Unit 2, systems: how do you get the most out of the hardware? {#systems}
source: lecture_01.py:L319-L374 · video 35:53-45:12

The systems unit's goal is to squeeze the most out of the hardware (GPU or TPU). Its components are kernels, parallelism and inference, preceded by some basics.

### Resource accounting and a first formula

The basics, which start next lecture, are **resource accounting**: keeping track of where all the FLOPs go and where all the memory is spent. One formula comes up constantly. Training a model with $N$ parameters on $D$ tokens costs about

$$ C \approx 6ND \ \text{FLOPs} $$

(a FLOP is one floating-point addition or multiplication). For a 70B-parameter model on 1T tokens, $6 \times 70\text{e}9 \times 10^{12} = 4.2 \times 10^{23}$ FLOPs. The professor posed the formula and pointedly left open "and where does that come from?" The answer is [L2](#/read/lecture_02): 2 FLOPs per parameter per token in the forward pass, 4 in the backward pass.

::code lecture_01.py:L325 | the one-line estimate, 6 × 70e9 × 1e12
::predict six-nd-preview

The formula is linear in both factors: doubling the tokens doubles the compute exactly as doubling the parameters does. So an 8B model on 15T tokens ($7.2 \times 10^{23}$) costs more than the 70B model on 1T, although it is almost 9× smaller. Conversely, at a fixed budget of $4.2 \times 10^{23}$, halving the model to 35B lets you train on $4.2\text{e}23 / (6 \times 35\text{e}9) = 2 \times 10^{12}$ tokens.

::widget fixture:lecture_01--compute-napkin | move N or D and watch C move in proportion; tick "hold the budget fixed" and moving N slides D the other way along the dashed line
::note deferred 36:12 | The derivation of 6ND, and what it leaves out (attention FLOPs that grow with context length), is the next lecture's.
::kp six-nd-preview

### Memory is not where compute is

The professor's cartoon of a GPU: memory (high-bandwidth memory, HBM) and compute (the streaming multiprocessors, SMs) are in different places. To compute anything you move parameters or activations from memory to compute, compute, and move the results back, and "that often is the bottleneck".

::figure official/lectures/images/compute-memory.png | two boxes and the link between them; the link is usually the slow part

A B200, the GPU students will use, does 2.25 PFLOP/s (petaFLOPs per second) in bf16 and has 8 TB/s of memory bandwidth. Divide one by the other: the chip can do about $2.25\text{e}15 / 8\text{e}12 \approx 281$ FLOPs in the time it takes to move one byte. Any operation doing fewer FLOPs per byte than that leaves the compute units waiting. **Roofline analysis** formalizes this test of whether a computation is compute-bound or memory-bound, and "in general, it is memory". Benchmarking and profiling (NVIDIA's nsight) show what happens in practice.

::note deferred 37:35 | "What does that mean? ... I'll do this next lecture." The ratio above and the roofline are worked out in [L2](#/read/lecture_02), where the H100's ratio is about 295.

A DGX B200 node holds 8 GPUs connected by NVLink; a thousand GPUs means many such nodes, linked by InfiniBand or Ethernet.

::figure https://docs.nvidia.com/dgx/dgxb200-user-guide/_images/dgx-b200-system-topology.png | eight GPUs per node joined by NVLink switches; network cards lead out to other nodes

### Kernels

A kernel is a function that runs on the GPU. Plain PyTorch already launches one standard kernel per primitive operation, "whether you know it or not". For some computations you can write custom kernels that run faster. The principle: **organize computation to minimize data movement**.
- Naive: read from HBM, compute A, write to HBM; read again, compute B, write again. The data crosses the slow link twice.
- Fused: read once, compute A and B, write once.

::animation fixture:fusion-timeline | the naive version's compute lane sits idle between kernels while the same tensor is written to HBM and read back; the fused one is a single block

That is **operator fusion** (for example a matmul with its activation). **Tiling**, the idea behind FlashAttention, is a more sophisticated variant. Modern GPUs add peculiarities (warp divergence, memory coalescing, bank conflicts, occupancy, bulk-asynchronous memory transfers), and kernels are written in CUDA, Triton (the course's choice), CUTLASS or ThunderKittens.

::note deferred 40:03 | Not every GPU peculiarity will be covered in depth: "I'm not sure how many of these details we'll have time to get into."

### Parallelism

With 1024 GPUs the principle is the same, but moving data between GPUs is even slower than within one. The tools are classic collective operations (gather, reduce, all-reduce). The parameters, activations, gradients and optimizer states are sharded across GPUs, and computation can be split by data, tensor, pipeline (layers), sequence or expert. Bringing the right data to the right GPU at the right time is a whole orchestration problem.

::note deferred 41:34 | The trade-offs between the parallelism strategies are left to the parallelism lectures.

### Inference

Inference means generating tokens from a prompt. It is needed to use a model at all, and also inside reinforcement learning (rollouts), test-time compute, synthetic-data generation and evaluation. It has two phases:
- **prefill**: the prompt's tokens are all given, so they are processed at once, like training. Compute-bound.
- **decode**: tokens are generated one at a time, each needing the previous one. This "quickly becomes memory bound", which is why inference is hard.

::figure official/lectures/images/prefill-decode.png | iteration 1 processes the whole prompt and fills the KV cache; each later iteration emits one token and reads the cache again

Ways to speed up decoding:
- a cheaper model, by pruning, quantization or distillation;
- **speculative decoding**: a cheap draft model guesses several tokens ahead, and the full model scores them all in one parallel pass; accepted tokens come several at a time, and the output is exactly what the full model would have produced;
- systems work: fused kernels, and continuous batching, because a service receives queries at unpredictable times, whereas training batches are fixed in advance.

::animation fixture:lecture_10--spec-rounds | four short sequential draft steps, then one target pass checks all four at once and keeps the accepted prefix plus one token of its own

::note skip 42:16 | Inference gets less time than the professor would like, "because the course is already kind of filled up"; whether you write inference from scratch was undecided.

**Assignment 2**: a fused RMSNorm kernel in Triton, distributed data-parallel training, optimizer-state sharding, and benchmarking and profiling of each; details may change this year. Recommended reading: [How to Scale Your Model](https://jax-ml.github.io/scaling-book/), a conceptual book on roofline analysis and Transformer math. It is from Google, so it centers on TPUs, but the concepts carry over, and it now has a chapter on GPUs.

## Unit 3, scaling laws: what would you train with $10^{25}$ FLOPs? {#scaling}
source: lecture_01.py:L377-L411 · video 45:12-53:28

Suppose you have $10^{25}$ FLOPs, "tens of millions of dollars of compute". What model should you train? You cannot tune hyperparameters at that scale the usual way, because you only get to train one model. This is the central problem of large-scale training, and it does not arise when you fine-tune or work small.

### From one model to a scaling recipe

The conceptual shift: stop thinking about a single model and think about a **scaling recipe**, a mapping from a FLOP budget to a full set of hyperparameters (aloud, "basically a config file"). For a given recipe:
1. run it at several small budgets (say up to $10^{24}$) and record the losses;
2. fit a scaling law to those points and extrapolate to the target (say $10^{25}$).

Two things become possible. You can optimize the recipe for the large scale using only small experiments. And you can predict the loss before running the big experiment, which, the professor notes, "allows you to go raise money".

But scaling laws "are not laws of nature". They do not happen automatically; "you have to will them into existence" by building a recipe that extrapolates. That means knowing how each hyperparameter should move with scale (does the learning rate stay constant or drop? how fast does the batch size grow?). It also means parameterizing the model for **hyperparameter transfer** (muP is one way): the best hyperparameters at small scale are the same, or a predictable function of them, at large scale. If the best learning rate jumps between $10^{-5}$ and $10^{-4}$ from one scale to the next, nothing can be predicted. Hence the slogan: **predictability is at least as important as optimality**.

### Bigger model or more tokens?

The classic question: with budget $C = 6ND$, should you spend it on a bigger model (larger $N$) or on more tokens (larger $D$)? Kaplan et al. (2020) and Chinchilla (2022) answer it with **IsoFLOP curves**:
- for each of several small budgets (in the figure, $6 \times 10^{18}$ to $3 \times 10^{21}$ FLOPs), train models of different sizes, each on as many tokens as the budget allows, and find the size with the lowest loss;
- fit a line through those optimal sizes and extrapolate to large budgets.

::figure official/lectures/images/chinchilla-isoflop.png | left: one U-shaped curve per budget, its minimum the best size; middle and right: the minima fall on lines, extrapolated to 63B parameters and 1.4T tokens
::animation fixture:isoflop-sweep | each larger budget's U-curve sits lower and its minimum further right; both the best N and the best D grow with C

If the optima fall on a line you can extrapolate; "if you're unlucky, it's going to be all over the place", and then you should have no confidence in the prediction.

The upshot, "quite crude": train on about **20 tokens per parameter**, $D \approx 20N$. A 70B model should see about 1.4T tokens. The exact ratio varies with the dataset and architecture.

::code lecture_01.py:L393-L399 | the question in terms of C = 6ND, the IsoFLOP method, and the 20:1 rule with its caveat

Put the two rules together and you can size a model from a budget alone: $C = 6N \cdot 20N = 120N^2$, so $N = \sqrt{C/120}$.

::predict chinchilla-d-20n

For $10^{25}$ FLOPs that gives $N \approx 290$B parameters on about 5.8T tokens. Notice that the lecture's own example, 70B on 1T tokens, is only 14 tokens per parameter: by this rule it is slightly too big a model for its budget.

**The caveat**: the rule ignores inference cost. A model is trained once but served many times, and a smaller model is cheaper to serve. "A lot of models these days are small, but they're trained on way more tokens than is compute optimal." Fix a 10B model at a budget of $6 \times 10^{23}$ FLOPs and the budget buys $10^{13}$ tokens, 1000 tokens per parameter, fifty times the rule. This is also why PaLM is called undertrained: it was big for its compute, with too few tokens per parameter.

::widget fixture:lecture_01--compute-napkin | hold C fixed and slide N: only one point meets the green D = 20·N line; every smaller N buys more than 20 tokens per parameter
::video 50:27-51:09 | "quite crude", varies with data and architecture, and why today's small models are over-trained
::kp chinchilla-d-20n

### A live prediction

The Marin project pre-registers its scaling predictions: it fitted a scaling law on runs at several budgets and published the predicted loss of a much larger run before training it. At the time of the lecture that run was finishing ("should be done maybe as early as tonight").

::note deferred 51:37 | The result is promised for the next lecture. [L2](#/read/lecture_02) opens with it: the final loss landed within 0.05 of the forecast.

**Assignment 3** simulates the high-stakes setting without the cost. The staff trained many models offline and expose a training API: you submit a configuration (a "training job") under a FLOP budget and get back a loss. You fit scaling laws to your data points, then submit extrapolated hyperparameters and a loss prediction for a larger budget, with a leaderboard for the lowest loss.

## Units 4 and 5, data and alignment, and the thread that ties them {#data}
source: lecture_01.py:L414-L478, L242-L253 · video 53:28-1:04:46

### Data: what do you train on?

You can now train a model, make it fast and scale it up. What is missing is what to train it on, arguably the most important question, because data quality largely decides how good the model is. One framing: data reflects what you want the model to do. Speak many languages? Hold a conversation? Run long agentic coding tasks?

**Evaluation** comes first, because it defines those capabilities. It serves two distinct purposes that are often conflated:
- **internal**, guiding model development: what matters is smoothness across scales (predictability again) and relative performance. A perplexity of 1.2 means little in absolute terms, but it is still a very good measure of a model's intrinsic quality. Ideally run it on private documents that are not on the internet, to avoid contamination.
- **external**, reporting quality on real use cases to customers or reviewers: here ecological validity matters. Examples: GPQA, Humanity's Last Exam (HLE), SWE-Bench, Terminal-Bench.

Language models are general-purpose, so they need a diverse set of evaluations; an average over them conflates many different things.

::note deferred 54:30 | "Evaluation is a fairly deep topic", left to the data unit.

**Curation.** "Data does not just fall from the sky." It is crawled web pages, books (controversial now), arXiv papers, GitHub code and more. Copyright raises legal questions (fair use, or licensing, as Google did with Reddit), and raw data is HTML, PDFs and directories, not text.

::figure https://ar5iv.labs.arxiv.org/html/2101.00027/assets/pile_chart2.png | The Pile's sources (2021), sized by share: web, academic papers, books, code and more

**Processing**:
- *transformation*: convert HTML or PDF to text, extracting the main content;
- *filtering*: keep high-quality data and remove harmful content, with classifiers; a random Common Crawl document is usually bad;
- *deduplication*: saves compute and avoids memorization, using Bloom filters or MinHash;
- *mixing*: how much to upweight or downweight each source;
- *rewriting and synthetic data*: use a language model to rewrite real data into forms closer to the downstream tasks.

Data also comes in stages: large, diverse pretraining data; high-quality **mid-training** data at the end of pretraining, including long-context data such as code repositories and books; and post-training data (conversations, agentic traces with tool calls).

**Assignment 4**: start from raw Common Crawl HTML, convert it to text, train quality and harmful-content classifiers, deduplicate with MinHash, and minimize perplexity given a token budget. "A lot of what people would call dirty work", and part of the full experience.

### Alignment: improving the model from weak supervision

Up to here the model is trained with full supervision: predict the next token. Once it is reasonable, it can be improved further with **weak supervision**, which helps whenever it is easier to critique a response than to write the right one. The template:
1. generate responses from the model;
2. score them with a human, a verifier or an LM judge;
3. update the model to prefer the better responses.

Algorithms: PPO from reinforcement learning (as in InstructGPT); DPO, simpler, for preference data; GRPO, which removes PPO's value function. The challenges: RL algorithms are unstable and hard to tune (the professor prefers to stay with supervised learning as long as possible), and at scale RL needs new infrastructure, an inference server producing rollouts and a training server updating weights. Rollout workers that lag make the data off-policy, so throughput trades against on-policyness: "a big, wonderful mess".

::note deferred 1:02:28 | RL systems are left to the alignment unit; Assignment 5 was still being decided (last year: implement DPO and GRPO).

### It is all about efficiency

The syllabus closes where the lecture began. Resources are data plus hardware (compute, memory, communication bandwidth), and the question is how to train the best model given a fixed set of them. Today we are **compute-constrained**, and every unit reads as a way to squeeze more out of the hardware:
- **systems**: clearly about efficiency;
- **tokenization**: raw bytes are elegant, but compute-inefficient with today's architectures;
- **architecture**: many changes reduce memory or FLOPs (sharing KV caches, sliding-window attention), often for faster inference;
- **data filtering**: do not spend precious compute on bad or irrelevant data. Even if bad data does not hurt directly, under a fixed budget "more time on bad data means less time on good data";
- **scaling laws**: spend compute on small models to tune the hyperparameters of the big one.

"Tomorrow, we will become data-constrained", and then "the calculus of what design decisions you should take might change". Filtering is the clearest case: once tokens are scarce and compute is plentiful, throwing tokens away becomes the expensive move.

::code lecture_01.py:L242-L253 | the resources, the compute-constrained reading of every unit, and the data-constrained teaser
::kp efficiency-framing

## What exactly is a tokenizer? {#tokenizer}
source: lecture_01.py:L256-L262, L484-L606 · video 1:05:06-1:07:25

The first unit proper starts here. It was inspired by Andrej Karpathy's [video on tokenization](https://www.youtube.com/watch?v=zduSFxRajkE), which the professor recommends for depth.

On one side is raw text, which is generally a Unicode string, for example `"Hello, 🌍! 你好!"`. On the other side, a language model places a probability distribution over *sequences of tokens*, usually represented by integer indices such as `[15496, 11, 995, 0]`. So we need two procedures:
- **encode**: string → list of integers;
- **decode**: list of integers → string.

A tokenizer is a class implementing both. The lecture writes it as an abstract interface that every tokenizer below fills in.

::code lecture_01.py:L256-L262 | the whole contract: encode str → list[int], decode list[int] → str

The model never sees characters, only ids, and ids mean something only relative to the vocabulary that produced them. The example `[15496, 11, 995, 0]` is GPT-2's encoding of "Hello, world!"; decode the same four integers with another tokenizer and you get an unrelated string. The GPT-5 tokenizer, for instance, encodes "Hello" as 13225, not 15496.

The one property every tokenizer must have: it **round-trips**, `decode(encode(s)) == s` for every string. "If you implement a tokenizer and it doesn't round trip, you have a problem." A lowercasing encoder, for example, would fail it on any capital letter.

::predict tokenizer-interface

### A production tokenizer, observed

To get a feel, the lecture points to an [interactive site](https://tiktokenizer.vercel.app/?encoder=gpt2) where you type text and see it tokenized. Three observations, which the professor offers as reasons "tokenizers are kind of annoying and why people want to get rid of them":
- **A word and its preceding space form one token** (e.g. `" world"`). Many tokens are "space, word".
- **The same word at the beginning and in the middle is represented differently.** In `"hello hello"` the first is `hello`, the second ` hello`: "two completely different indices that have nothing to do with each other".
- **Numbers are split into chunks of a few digits.** Sometimes predictably, sometimes not. Some tokenizers make every digit a token, "but then you're blowing up the number of tokens you have".

::predict gpt-tokenizer-observations

So `"red red red"` encodes as `[a, b, b]`: the first `red` has no space before it, every later one is ` red`. Token identity depends on the character before the word.

::note skip 1:05:54 | The interactive site did not load in class ("this is not going to work"), so the observations were stated, not demonstrated. Aloud the first one also came out garbled ("a word conglomerate with its preceding space are different tokens"); the next sentence agrees with the code.
::note deferred 1:06:03 | Why these patterns arise is not explained. They come from pre-tokenization: a regular expression splits text into chunks before BPE runs, and it attaches the leading space to each word and controls how digits group. Assignment 1 uses the GPT-2 version of that regex.

Then the lecture runs the real thing: the GPT-5 tokenizer (`o200k_base`, from OpenAI's tiktoken library) on the lecture's test string. It produces 8 tokens, `[13225, 11, 130321, 235, 0, 220, 177519, 0]`, and decoding them gives the string back exactly. Note the emoji costs two tokens (130321, 235), while `你好` costs one.

::code lecture_01.py:L599-L606 | load o200k_base, encode, decode, assert the round trip
::kp tokenizer-interface
::kp gpt-tokenizer-observations

## How do you measure a tokenizer? Compression ratio and vocabulary size {#ratio}
source: lecture_01.py:L567-L571, L608-L615 · video 1:07:25-1:08:28

The lecture's measure is the **compression ratio**: the number of UTF-8 bytes per token.

::code lecture_01.py:L567-L571 | bytes of the string, divided by the number of tokens

$$ \text{compression ratio} = \frac{\text{UTF-8 bytes of the string}}{\text{number of tokens}} $$

The numerator is **bytes**, not characters. `"Hello, 🌍! 你好!"` has 13 characters but 20 bytes, because the emoji takes 4 bytes and each Chinese character 3 (the byte section below explains why). So GPT-5's ratio on it is $20 / 8 = 2.5$; dividing characters instead would give $13 / 8 \approx 1.6$, a different and wrong number.

::predict compression-ratio
::widget fixture:lecture_01--tokenizer-lens | on the lecture string the byte row has 20 boxes and the character row 13: the GPT-5 ratio 2.5 is 20 bytes ÷ 8 tokens

A larger ratio means a shorter sequence for the same text, which is good "because attention is quadratic". Ratios also differ by language with the same tokenizer: if a 1000-byte English document becomes 250 tokens (ratio 4.0) and a 1000-byte Chinese document 500 tokens (ratio 2.0), the Chinese document uses twice the context window.

::video 1:07:25-1:07:56 | 20 bytes ÷ 8 tokens = 2.5, and why higher is better
::kp compression-ratio

### The vocabulary-size trade

How do you raise the compression ratio? Increase the **vocabulary size**, the number of distinct token values: with more entries, longer chunks get their own token. GPT-5's vocabulary has 200,019 entries; "these days, tokenizers, especially multilingual tokenizers, have 100k or 200k tokens".

The cost is **sparsity**. Aloud: "every element of vocab is treated like a distinct element". Each id is learned on its own, from the occasions it occurs, so a huge vocabulary of rarely seen ids learns slowly. And the gains shrink: the entries added last are the rarest chunks, so each saves less. Suppose growing a vocabulary from 32K to 64K lifts the ratio from 3.6 to 4.0 then sequences get only 10% shorter ($3.6/4.0 = 0.9$), for 32K new ids to learn.

::predict vocab-size-vs-sequence-length

The three vocabularies of this lecture make the point: bytes 256, characters at least 127,758 (computed below), GPT-5 200,019. The character vocabulary is almost as large as GPT-5's yet compresses far worse, so size alone buys nothing; what matters is whether the entries are frequent chunks.

::widget fixture:lecture_01--vocab-trade | set ratios 3.6 and 4.0 and vocabularies 32,000 and 64,000: the length bar shrinks only 10% while the vocabulary bar doubles with V
::note skip 1:08:22 | Browsing the GPT-5 vocabulary file was skipped "in the interest of time".
::kp vocab-size-vs-sequence-length

### What sparsity costs in parameters (supplement)

The lecture names sparsity but not its concrete price, which follows from how a Transformer uses ids. Each id owns one row of the input embedding matrix, a vector of the model's width $d$, and (if the output layer is not tied to it) one row of the output projection. So a vocabulary of $V$ tokens costs

$$ V \cdot d \ \text{parameters per matrix} $$

At $d = 4096$: $256 \times 4096 \approx 1.05$M parameters for bytes, and $200{,}019 \times 4096 \approx 819$M for GPT-5's vocabulary, about 780× more. A row is updated only when its id occurs, which is the sparsity: rare ids get few gradient updates.

::predict supp-vocab-embedding-params

This cost is independent of depth, so it weighs most on small models. A 7B model with $d = 4096$, a 128K vocabulary and untied matrices spends $2 \times 128{,}000 \times 4096 \approx 1.05$B parameters, 15% of its total, on the vocabulary. For a "1B" model with $d = 2048$ and a 256K vocabulary, those two matrices alone would be the whole budget.

::widget fixture:lecture_01--vocab-trade | the vocabulary bars depend only on V and d: the same 2·V·d is a large share of a small model and a small share of a large one
::kp supp-vocab-embedding-params

## Why not use characters, or bytes? {#chars-bytes}
source: lecture_01.py:L505-L524, L627-L673 · video 1:08:28-1:10:48

Now the lecture builds tokenizers itself, starting with the two most obvious ones. The professor went "through this fast"; each takes only a few lines.

### Characters: Unicode code points

A Unicode string is a sequence of Unicode characters, and each character has an integer **code point**, which Python's `ord` returns: `ord("a") = 97`, `ord("🌍") = 127757`. `chr` converts back. So a character tokenizer is just `ord` to encode and `chr` to decode, and it round-trips.

::code lecture_01.py:L505-L511 | encode is map(ord), decode is map(chr)

On the lecture string it gives 13 tokens, one per character. What is its vocabulary? There are about 150K assigned Unicode characters. The code can only compute a lower bound from the string it has, the largest code point plus one: $127757 + 1 = 127{,}758$, set by the emoji. Its compression ratio is $20/13 \approx 1.54$, above 1 only because some characters take several bytes; on pure ASCII text it is exactly 1. On `"你好"` alone it is $6/2 = 3$.

::predict character-tokenizer
::code lecture_01.py:L636-L648 | 13 tokens, vocabulary ≥ 127,758, ratio 1.54: "the worst of both worlds"

The code lists two problems: (1) a very large vocabulary, (2) many characters are rare (e.g. 🌍), an inefficient use of the vocabulary. Aloud the professor ranked them: 150K is "not crazy" (GPT-5's vocabulary is larger), and rarity is "the bigger problem". Most vocabulary slots are rarely used, while the common text still costs one token per character. A large vocabulary and a low compression ratio: "the worst of both worlds".

::note aside | The code's bound comes from one string. Unicode code points actually run up to 0x10FFFF (1,114,112 values), so a vocabulary covering every possible code point would be far larger than the 150K assigned characters.
::video 1:09:05-1:09:41 | 150k is "not crazy"; rarity is the bigger problem
::kp character-tokenizer

### Bytes: UTF-8

A Unicode string can also be stored as a sequence of bytes, integers from 0 to 255. The most common encoding is **UTF-8**, which is *variable length*: a character takes 1 to 4 bytes depending on its code point.
- ASCII (U+0000 to U+007F) takes 1 byte: `a` is the single byte 0x61.
- U+0080 to U+07FF takes 2 bytes (é is `c3 a9`).
- U+0800 to U+FFFF takes 3 bytes, which covers most Chinese characters.
- Code points above U+FFFF take 4 bytes: `"🌍"` is `f0 9f 8c 8d`.

::code lecture_01.py:L652-L658 | "a" is one byte, the emoji four

So `ord` gives one integer for 🌍 while UTF-8 encoding gives four bytes: one is the character's number, the other its stored form. The lecture string's byte count, step by step:

::worked utf8-variable-length
::predict utf8-variable-length

A **byte tokenizer** emits one token per UTF-8 byte. Its indices for the lecture string are `[72, 101, 108, 108, 111, 44, 32, 240, 159, 140, 141, 33, 32, 228, 189, 160, 229, 165, 189, 33]`: 20 tokens, all between 0 and 255.

::code lecture_01.py:L514-L524 | encode is string.encode("utf-8"), decode is bytes(indices).decode("utf-8")

The vocabulary is nice and small: exactly 256, fixed, and every possible string is covered. But the compression ratio is bytes ÷ bytes, so it is **exactly 1 for every input**; the code asserts it. A 5,000-character Chinese document is about 15,000 tokens, still at ratio 1. Sequences are as long as the text's byte count, and since a Transformer's context is limited because attention is quadratic, "this is not looking great".

::predict byte-tokenizer
::widget fixture:lecture_01--tokenizer-lens | type any script: the byte row is one token per byte box, so its ratio stays 1.0 while its token count follows the byte count
::code lecture_01.py:L667-L673 | vocabulary 256, assert compression_ratio == 1, sequences too long

One more property the lecture does not show: byte ids are not characters. `[228, 189]`, the first two bytes of 你, opens a three-byte sequence and never completes it, so `bytes([228, 189]).decode("utf-8")` raises an error. A model emitting byte ids can produce sequences that no string maps to.

::kp utf8-variable-length
::kp byte-tokenizer

## Why not use words? {#words}
source: lecture_01.py:L676-L695 · video 1:10:48-1:11:58

Characters and bytes are "both really bad". The classical NLP approach goes the other way: split the string into words. The lecture uses the regular expression `\w+|.`, which keeps each run of letters and digits together and makes every other single character (space, apostrophe, punctuation) its own chunk.

::code lecture_01.py:L678-L680 | split with \w+|.

`"I'll say supercalifragilisticexpialidocious!"` (44 bytes) becomes 8 chunks: `I`, `'`, `ll`, ` `, `say`, ` `, `supercalifragilisticexpialidocious`, `!`. To make this a tokenizer you would map each distinct chunk to an integer; the lecture does not build that mapping.

::predict word-tokenizer

What is good: each token is meaningful, "because humans invented words, and words tend to have a stable semantic meaning". And the compression ratio is high: $44 / 8 = 5.5$, against GPT-5's 2.5 on the earlier string.

What is bad is the vocabulary: it is the number of distinct chunks in the training data, which can be huge. Worse:
- many words are rare, and the model will not learn much about them;
- there is no fixed vocabulary size, and in fact it is unbounded: "at test time, you might get some sequence ... and then you have a token you've never seen before";
- such unseen words get a special **UNK** token, which "is ugly and can mess up perplexity calculations". Every unknown word collapses to the same id, so the text loses information, and a model can look better than it is by confidently predicting UNK.

::widget fixture:lecture_01--tokenizer-lens | pick the word example: letter runs stay whole, but every space, apostrophe and punctuation mark is its own chunk, 8 in all
::code lecture_01.py:L686-L695 | meaningful tokens and a good ratio, but a vocabulary of every training chunk, and UNK for the rest

So the failure is not that words are too long; long tokens are the good part. It is the open-ended vocabulary. A word tokenizer trained on `"the cat sat on the mat."` encodes `"the dog sat on the rug."` with two UNKs, for `dog` and `rug`.

::kp word-tokenizer

## How does BPE learn a vocabulary from data? {#bpe-train}
source: lecture_01.py:L527-L547, L698-L712, L729-L758 · video 1:11:58-1:15:21

The fix is **Byte Pair Encoding**. Its history in three steps: Philip Gage introduced it in 1994 as a data-compression algorithm, "way before language models were really on the scene"; Sennrich et al. (2016) adapted it to NLP for neural machine translation, where papers had been using word tokenization; and GPT-2 (2019) was the first to use it for large language models.

The basic idea: *train* the tokenizer on raw text, to build a vocabulary tailored to the data. Common byte sequences become a single token; rare ones are represented by many tokens. And because it starts from bytes, everything can be tokenized: a rare string "just breaks up into smaller units rather than having this UNK token". BPE combines the byte tokenizer's full coverage with the word tokenizer's long chunks, deciding by frequency which chunks earn a token.

The sketch: start with each byte as a token, and repeatedly merge the most common pair of adjacent tokens.

::code lecture_01.py:L705-L708 | train on the data; common sequences become one token; merge the most frequent pair

Because the merges come from pair counts, the vocabulary depends on the training text. Two tokenizers of the same size, one trained on English Wikipedia and one on Python code, tokenize the same file differently, and the code-trained one compresses Python better.

::predict bpe-idea
::note aside | The lecture collapses two variants. Sennrich's BPE merged characters inside words, after splitting text into words; the byte-level version with a 256-byte base vocabulary is GPT-2's.

### The training loop

Training takes a string and a number of merges. The professor simplifies: "let's assume it's one long sequence", with no splitting into words first. Each step:
1. **count** every adjacent pair of tokens in the current sequence (`count_adjacent_pairs` zips the list with itself shifted by one);
2. **pick** the most frequent pair, `max(counts, key=counts.get)`;
3. **create** a new token with the next free index, $256 + i$ (0 to 255 are the bytes), and record its bytes: `vocab[new] = vocab[a] + vocab[b]`;
4. **merge**: replace every occurrence of the pair in the sequence with the new token.

::code lecture_01.py:L735-L746 | count, pick the max, new index 256 + i, extend the vocabulary, merge
::code lecture_01.py:L527-L538 | merge scans left to right and replaces every occurrence of the pair, without overlap

The two data structures are the whole learned tokenizer: `merges`, mapping a pair of ids to its new id, and `vocab`, mapping each id to its bytes.

::code lecture_01.py:L541-L545 | a BPE tokenizer is fully specified by its vocab and its merges

### "the cat in the hat", three merges

Before reading the trace, try the first step.

::predict bpe-merge-step

The 18 bytes start as 18 tokens: `[116, 104, 101, 32, 99, 97, 116, 32, 105, 110, 32, 116, 104, 101, 32, 104, 97, 116]`. The worked steps:

::worked bpe-merge-step

::animation fixture:bpe-merge-tree | every occurrence of the chosen pair is replaced at once and the vocabulary grows by exactly one entry; the third token 'the␣' crosses a word boundary

Three things to notice.
- **Merged tokens join later pairs.** 256 (`th`) pairs with `e` to make 257 (`the`), which pairs with a space to make 258. That is how BPE builds long tokens from bytes.
- **Each merge adds exactly one vocabulary entry** and shortens the sequence by the pair's count (here 2 each time): $18 \to 16 \to 14 \to 12$, a compression ratio of $18/12 = 1.5$. "Over time, the sequence is shrinking and the vocabulary size is growing."
- **Every pick was a tie.** In the first step four pairs (`th`, `he`, `e␣`, `at`) occur twice. Python's `max` returns the first maximal key, and the counts dictionary is in order of first occurrence, so `th` wins because it appears first in the string. The professor said only "I guess there's a few ties, but we'll just take the first one"; steps 2 and 3 were ties as well.

::note aside 1:13:57 | The tie-break is an artefact of Python's `max` and dictionary order, not part of the algorithm. Assignment 1 specifies its own rule, so check the handout rather than relying on this one.
::note aside | Without pre-tokenization, merges cross word boundaries: token 258 is `the␣`, including the space. A real tokenizer splits text into word-like chunks first, which is also why production tokens carry a *leading* space (` world`) rather than a trailing one.

A fourth merge would pick `at` (in "cat" and "hat"), now the only pair occurring twice, giving 10 tokens. Further merges have count 1: they only memorize the training text. The widget lets you watch this on longer text, and on a trickier string where pairs overlap: in `aaabdaaabac`, the pair (a, a) is *counted* 4 times (overlapping), but `merge` replaces it only twice, left to right.

::widget fixture:lecture_01--bpe-lab | slide num_merges: the training curve falls fast at first and then flattens once the chosen pair's count drops to 1
::kp bpe-idea
::kp bpe-merge-step

## How does a trained BPE tokenizer encode new text? {#bpe-encode}
source: lecture_01.py:L549-L564, L714-L726 · video 1:15:21-1:17:29

To encode a new string: convert it to bytes, then **apply every learned merge, in training order**. Each merge is the same `merge` function run over the whole current sequence. Decoding maps each id to its bytes through `vocab`, concatenates them and decodes the result as UTF-8.

::code lecture_01.py:L554-L564 | encode replays the merges in order; decode joins vocab bytes and UTF-8-decodes

Order matters because later merges refer to ids made by earlier ones: (256, 101) can only fire once 256 exists. Encoding is not a lookup of the longest matching vocabulary entry; a long token is reached only through its chain of merges.

::predict bpe-encode-apply-merges

With the three merges learned above, `"the quick brown fox"` (19 bytes) becomes:

::worked bpe-encode-apply-merges

Merges match exact adjacent pairs anywhere, not whole words. `"bathe the"` encodes to `[98, 97, 258, 257]`: the `th` inside "bathe" is followed by `e` and a space, so it reaches 258, while the final `the` has no space after it and stops at 257. And a string none of whose pairs was learned, such as `"你好"`, simply stays as its 6 raw bytes: more tokens, but never UNK.

::widget fixture:lecture_01--bpe-lab | encode "hat the": the merges replay in order, and the final 'the' stops at 257 because no space follows it
::note skip 1:15:44 | "Let me actually not step through that code": encode was not traced in class, only its output shown. The worked example above is that trace.

### What Assignment 1 adds

The lecture's implementation is "a full-blown BPE implementation" and "extremely slow". Assignment 1 asks you to:
- **loop only over the merges that matter.** `encode` makes one pass over the text per merge, and "the number of merges you have is essentially the vocab size minus 256", so a 200K vocabulary means about 200K passes. You need some index structures to skip the rest.
- **detect and preserve special tokens** such as `<|endoftext|>`: "conceptually not deep, but important".
- **use pre-tokenization** (e.g. the GPT-2 regex): break the text into chunks first and run BPE on each chunk. Aloud the professor gave speed as the reason, "that's going to be much faster"; it also keeps merges inside words.
- **make it as fast as possible.** "At some point you might realize that Python is just not very fast", and Rust or C is welcome.

::code lecture_01.py:L722-L726 | the four extensions Assignment 1 asks for
::video 1:16:06-1:16:44 | one pass per merge, and merges = vocabulary size − 256
::kp bpe-encode-apply-merges

## What should you carry away? {#summary}
source: lecture_01.py:L494-L502, L49 · video 1:17:29-1:19:17

The tokenization summary, with the numbers behind each line:
- **A tokenizer converts strings to tokens (indices) and back**, and must round-trip. Its compression ratio is UTF-8 bytes per token; higher means shorter sequences, and attention cost grows with the square of length.
- **Character, byte and word tokenizers are each suboptimal.** Characters: a vocabulary of 150K-plus, mostly rare, and a ratio near 1 (1.54 on the lecture string). Bytes: a perfect 256-entry vocabulary but a ratio of exactly 1. Words: a ratio of 5.5 but an unbounded vocabulary and UNK.
- **BPE is an effective, data-driven heuristic**: start from bytes, repeatedly merge the most frequent adjacent pair, and replay the merges to encode. It keeps full coverage and gets long tokens where the data is repetitive.
- **Tokenization is a separate step**, and maybe one day models will work end to end from bytes.

Whatever replaces it, the professor argues, must keep two properties:
1. **The model operates on chunks**, abstractions of the sequence. This is clearest beyond text: in video or DNA, individual bytes or bases carry little signal, and some abstraction must lift them to a level where modeling works.
2. **Chunks are variable**, so more model capacity goes to the interesting parts: "Not all bytes are treated the same."

That second property is the adaptive computation from the basics section, now with the reason it matters. With it the lecture's two halves meet: tokenization is one more place where the course asks how to spend a fixed compute budget well.

::video 1:18:41-1:18:53 | chunks should be variable: adaptive computation
::kp tokenization-efficiency-lens
::note deferred 1:19:01 | Next lecture: resource accounting, "sort of a baby system", where 6ND and the hardware numbers above are derived. See [L2](#/read/lecture_02).
