---
title: L8 · Parallelism basics, read through
minutes: 45
---
This lecture is the systems view of training one model on thousands of GPUs. It explains why one GPU is not enough, how the machines are wired, and the standard ways of splitting the work: data parallelism with ZeRO and FSDP, pipeline, tensor, sequence and expert parallelism. It then shows how real training runs combine them. After it you can say, for any of these strategies, what it splits, what it sends over the network and how much, which link it belongs on, and why big runs use several of them at once. [L7](#/read/lecture_07) builds the same strategies in code; this lecture adds the accounting and the practice.

## Why is one GPU not enough? {#why}
source: lecture_08.pdf p1-p6 · video 0:05-2:38

The professor frames the lecture as "knowledge, details, and trivia" about how modern parallelism works, and promises a slide near the end called "4D parallelism": there are four (or more) things you can split at once, and at the largest scale you need most of them. Choosing the best combination for a given model and network is part of the assignment.

::note deferred 0:58 | "Part of your assignment will be figuring out, given a particular network topology and a particular model, what is the optimal parallelization strategy for that." The lecture gives the rules of thumb; the optimization itself is homework.

There are two reasons to leave the single GPU, and the whole lecture is organized around them.

::slide 4 | left: Bill Dally's single-chip plot from lecture 5 (about 1000× in ten years); right: the TOP500 "projected performance development", supercomputers climbing past 1 EFlop/s

**Compute.** Even after a 1000× rise in ten years (the left plot, met in [L5](#/read/lecture_05)), one chip is far from what a training run needs. The fastest supercomputers have exaflops, $10^{18}$ FLOP/s. An H100 does about $10^{15}$ dense bf16 FLOP/s (see [L2's FLOP accounting](#/read/lecture_02)), so an exaflop is roughly a thousand of them working together. The only way to get there is many chips linked together.

::slide 5 | model size in billions of parameters, log scale, 2018-2022: ELMo (94M), BERT-Large (340M), GPT-2 (1.5B), Megatron-LM (8.3B), T5 (11B), Turing-NLG (17.2B), GPT-3 (175B), Megatron-Turing NLG (530B)

**Memory.** Model sizes grew about 10× a year on this plot, from 94 million parameters to 530 billion in four years. A preview of a number we derive later: training needs about 16 bytes per parameter before activations, so the 530B model needs about 8.5 TB of state against 80 GB on one GPU. The model has to be cut into pieces.

::slide 6 | a DGX A100 node: eight GPUs at the bottom, all wired to six NVSwitches; above them PCIe switches (PLX) connect pairs of GPUs to four InfiniBand cards (HCA) that lead out of the box; two CPUs at the top

The picture to keep for the rest of the lecture is this box. Inside it, eight GPUs are wired through NVSwitches with very fast links: this is **intra-node** communication, and it is fast enough that you can afford communication-heavy schemes. To reach a GPU in another box, data goes out through a network card (an InfiniBand HCA) onto the much slower **inter-node** network. In the professor's words, slow links force us to use "processes that respect the communication constraints of our channel more". Every strategy in Part 2 will be placed according to which of these two links it stresses. [L7's hardware section](#/read/lecture_07) gives the speeds: about 1.8 TB/s for NVLink 5.0 against about 0.05 TB/s for InfiniBand.

## Collectives, and the one identity we need {#collectives}
source: lecture_08.pdf p7-p8 · video 2:38-4:21

Throughout the lecture nobody sends packets. All communication is counted at the level of **collective operations**, the operations every GPU in a group calls together. "Our discussion today is going to be algorithmic", the professor says; making collectives fast needs work all the way down to the hardware, and that is skipped.

::slide 7 | five collectives on four ranks: all-reduce (every rank ends with the sum), broadcast (rank 2's input copied to all), reduce (the sum lands on one root), all-gather (every rank ends with every rank's piece), reduce-scatter (rank i ends with the sum of slot i)

A one-line recap of each, with $M$ ranks:
- **reduce**: sum everyone's tensor onto one root rank;
- **broadcast**: copy one rank's tensor to everyone;
- **all-reduce**: everyone ends with the full sum;
- **all-gather**: everyone holds one piece; afterwards everyone holds all $M$ pieces;
- **reduce-scatter**: everyone holds a full tensor; afterwards rank $i$ holds only slot $i$ of the sum.

[L7](#/read/lecture_07) runs each of these in PyTorch and works them on small integer tensors. This lecture needs just one fact about them.

::slide 8 | an all-reduce of A, B, C, D (left) done as a reduce-scatter (middle: rank i gets the sum of slot i, A_i + B_i + C_i + D_i) followed by an all-gather (right: every rank collects every summed slot)

::predict collective-primitives

$$ \text{all-reduce} = \text{reduce-scatter} \;+\; \text{all-gather} $$

Reduce-scatter leaves rank $i$ with the summed slot $i$; all-gathering those slots puts the full sum everywhere, which is exactly an all-reduce. The slide's second claim is the important one: in the bandwidth-limited regime, this two-step version is the best you can do. It is how the standard ring all-reduce actually works, so splitting an all-reduce into its two halves costs nothing extra. For a gradient of #params elements, each half moves about $\#\text{params} \cdot (M-1)/M$ per rank, and the whole all-reduce about twice that, which is where the deck's "2× #params" for data parallelism will come from.

Why care? Because the split opens a seam. Between the two halves, each rank holds a fully summed *slice* and nothing else, and it can do work on that slice before the all-gather runs. The professor flags this as the one thing to remember from the review: since the two forms have the same cost, "we can do the algorithm on the right for free". It is the whole trick behind ZeRO, two sections from now.

::video 3:14-3:38 | why this one decomposition matters: equal cost means the split algorithm comes for free
::widget fixture:collective-bytes | at every world size W the all-reduce bar is exactly the reduce-scatter bar plus the all-gather bar; slide W up and each bar saturates at S or 2S per rank

::note aside 4:07 | Asked why this decomposition in particular: there are others, but this one is the one the ZeRO algorithm needs.
::kp collective-primitives

## How are the chips wired? Mesh, tree, and why not connect everything {#networks}
source: lecture_08.pdf p9-p13 · video 4:21-11:05

Most of the lecture is hardware-agnostic: once you know the concepts you can apply them to any accelerator. But the professor spends a few minutes on the network itself, because it explains why different companies parallelize differently.

::slide 9 | top: TPU chips in a 2D grid whose edges wrap around (a toroidal mesh); bottom: NVIDIA's DGX A100 and H100 256-GPU SuperPODs (InfiniBand leaf and spine switches against an NVLink switch fabric) with a table of dense PFLOP/s, bisection and reduce bandwidth for 1 DGX and for 32 DGXs

**TPUs: a toroidal mesh.** Each TPU chip is wired only to its neighbours, and the neighbours wrap around at the edges. (The real network is a 3D torus; the professor pointed to a web visualization and said you can keep the 2D picture.) The number of neighbours per chip is the same "no matter how large your network gets", so the topology scales up simply and cheaply, and each link can be made "more beefy" for the same power. The cost: talking to a chip far away takes many hops.

**GPUs: all-to-all, built as a fat tree.** GPUs in a node are connected very fast; nodes form a pod through leaf switches; pods talk through spine switches. Any GPU can reach any other, but the tree, and the cost of keeping it fast, grows as you add nodes.

The table shows the trade in numbers. On the A100 SuperPOD, the reduce bandwidth is 150 GB/s inside one 8-GPU box and drops to 100 GB/s across 32 boxes, because leaving the box means InfiniBand. The H100 SuperPOD extends the NVLink switch fabric across all 256 GPUs and keeps 450 GB/s at both scales, 4.5× the A100's cross-node figure; its bisection bandwidth across 32 DGXs is 57,600 GB/s against 6,400, a 9× jump. That is the slide's "All-to-all (up to 256)".

::slide 10 | "Why mesh? Why tree?": pro mesh, cheaper and fast "(and just do tensor parallel)"; pro tree/all-to-all, better for less structured communication "(expert parallel)"; below, Bill Dally and Jeff Dean agreeing that the best network depends on the traffic pattern

Which is better depends on the traffic. If chips only talk to neighbours in a predictable pattern, the mesh is ideal and cost-effective. If traffic goes anywhere, "very stochastic, unpredictable ways", the switched tree is more flexible. The slide's quote from Bill Dally (NVIDIA) and Jeff Dean (Google) says the same: a mixture-of-experts model scatters tokens to experts all over the machine, which suits switches; a dense model with predictable partitions suits a torus. Both terms in brackets, tensor parallel and expert parallel, are defined later in this lecture.

::video 6:25-6:43 | which traffic each topology suits: neighbour-only (TPU mesh) versus unpredictable (GPU fat tree)

::slide 11 | Google's TPU8i, with fully connected groups of chips rather than a torus, and TPU8t's "Virgo" switched scale-out network

Then the workloads moved. Google announced TPU8i and TPU8t the morning of the lecture, and the professor added this slide that day. TPU8i, an inference chip, connects groups of chips all-to-all, closer to a tree. That makes sense: modern LLMs are MoEs, and serving one sends tokens to experts everywhere. The training chip TPU8t gets a switched cross-rack network called Virgo that "looks a lot more like a GPU". He calls it "a little bit of a convergent evolution": the workloads are defining the network.

::slide 12 | SemiAnalysis's table of Huawei's CloudMatrix 384 (Ascend 910C) against NVIDIA's GB200 NVL72: per chip and per system FLOPs, HBM, scale-up and scale-out bandwidth, scale-up domain size and power

If all-to-all is so good, why not connect everything? Huawei's CloudMatrix 384 nearly does. The numbers on the slide:

| | GB200 NVL72 | CloudMatrix 384 | ratio |
|---|---|---|---|
| BF16 dense TFLOPS per chip | 2,500 | 780 | 0.3× |
| chips in the fast (scale-up) domain | 72 | 384 | 5.3× |
| BF16 dense PFLOPS per system | 180 | 300 | 1.7× |
| all-in system power (W) | 145,000 | 599,821 | 4.1× |
| power per BF16 dense TFLOP (W) | 0.81 | 2.00 | 2.5× |

Each Ascend chip is much weaker; the professor compares it to an H200, the slide to a GB200, at about a third of the FLOPs. But Huawei links 384 of them in one rack-scale domain over optical switches, so the system out-computes NVIDIA's 72-GPU rack, at about four times the power. "If you're willing to pay the power cost, you can solve a lot of communication problems brute force." That is why domains stop at 8, 72 or 384: the fast all-to-all domain costs power and money that grows quickly with its size.

::note aside 8:43 | A callback to the GPU lecture's question "if SRAM is so good, why not only SRAM?": that chip exists (Groq). Like CloudMatrix, it is the brute-force corner of the design space: efficient designs in power and manufacturing end up in one place, brute-force designs in another.
::note slip 10:02 | Aloud the domain becomes "300 chips"; the slide's table says 384.

::slide 13 | Part 1 recap: the new unit of compute is the datacenter; we want linear memory scaling (max model parameters grow with the number of GPUs) and linear compute scaling (model FLOPs grow with the number of GPUs); simple collective primitives

The recap sets the goals for Part 2. The unit of compute is no longer the GPU but the whole datacenter. From it we want two kinds of linear scaling: the largest model we can hold should grow with the number of GPUs (**memory**), and so should the FLOPs we can apply (**compute**). And we want to get there "lossless", wasting none of the hardware, using only the collectives above.

::predict bandwidth-hierarchy
::kp bandwidth-hierarchy

## Data parallelism, and why its memory is terrible {#data-parallel}
source: lecture_08.pdf p14-p17 · video 11:05-15:23

::slide 14 | Part 2 outline: data parallelism (naive, ZeRO levels 1-3), model parallelism (pipeline, tensor), activation parallelism (sequence parallel)

Part 2 is the bulk of the lecture: the algorithms. They come in two families. In **data parallelism** the data moves around: each GPU gets different examples. In **model parallelism** the model is chopped up and its pieces live on different GPUs. The professor admits the boundary is "a little leaky", because one data-parallel algorithm (ZeRO stage 3) also cuts up the parameters. The grouping is still useful, and we will see why.

::slide 15 | naive SGD, θ_{t+1} = θ_t − η Σ_{i=1}^{B} ∇f(x_i); split the B-sized batch across M machines and exchange gradients; compute scaling B/M per GPU, communication 2× #params every batch, memory scaling none

Start with plain SGD (forget Adam for the moment). One step sums the gradients of $B$ examples:

$$ \theta_{t+1} = \theta_t - \eta \sum_{i=1}^{B} \nabla f(x_i) $$

The sum splits naturally. Give each of $M$ machines $B/M$ examples, let each compute the gradient on its share, then add the $M$ partial sums with an all-reduce so every machine applies the same update.

::predict naive-data-parallel-accounting

Score it on the three things we care about:
- **Compute** scales perfectly: each GPU does $B/M$ of the work, "as long as there's enough examples per GPU".
- **Communication** is an all-reduce of the gradients every step, about 2 × #params (the reduce-scatter plus the all-gather from the last section). That count does not depend on $B$, so it is "OK if batches are big": a bigger batch means more compute per step for the same bytes.
- **Memory** does not scale at all. Every GPU holds a full copy of the model.

::worked naive-data-parallel-accounting
::note slip 13:13 | Aloud: "You need to have the same sized activations so you don't save any memory." That holds at a fixed batch per GPU; at a fixed global batch each GPU's activations shrink with B/M. The slide's point, that parameters and optimizer state are not split, is unaffected.

::slide 16 | four GPUs, each with its own full model copy, fed from one dataset

"Memory seems like it'd be a problem." It is worse than it looks, because a model copy is much more than its weights.

::slide 17 | "We need 5 copies of weights and 16 bytes per param!": 2 bytes BF16 parameters, 2 bytes BF16 gradients, 4 bytes FP32 master weights, 4 (or 2) bytes for each Adam moment, the last three bracketed as "optimizer state"

"Our memory situation is actually terrible." In mixed-precision Adam training (see [L2's memory accounting](#/read/lecture_02)) each parameter carries five weight-sized copies:
- 2 bytes: the BF16 parameters that forward and backward read;
- 2 bytes: the BF16 gradients;
- 4 bytes: an FP32 master copy, "the thing you accumulate into in SGD", because small updates added to a BF16 weight would round away;
- 4 + 4 bytes: Adam's first and second moments, in FP32 or, if training is stable enough, BF16.

That is 16 bytes per parameter. The three copies the slide brackets as optimizer state, 12 of the 16 bytes, are, as the professor stresses, "most of the cost memory wise". Activations come on top and are left until much later.

::predict training-memory-per-param
::worked training-memory-per-param
::video 14:21-15:04 | which copies are optimizer state, and why they may need high precision

So a 13B model needs 208 GB of state before it has seen a single token, and with naive data parallelism every GPU holds all 208 GB. The next three sections remove that redundancy step by step.

::kp naive-data-parallel-accounting
::kp training-memory-per-param

## ZeRO stages 1 and 2: shard the optimizer state, then the gradients, for free {#zero}
source: lecture_08.pdf p18-p23 · video 15:23-20:25

::slide 18 | the ZeRO paper's figure: per-GPU memory as blue parameters, orange gradients and green optimizer state across gpu_0 … gpu_{N−1}; baseline (2+2+K)Ψ = 120 GB, P_os 2Ψ + 2Ψ + KΨ/N_d = 31.4 GB, P_os+g 2Ψ + (2+K)Ψ/N_d = 16.6 GB, P_os+g+p (2+2+K)Ψ/N_d = 1.9 GB, for K = 12, Ψ = 7.5B, N_d = 64

**ZeRO** (the Zero Redundancy Optimizer, Rajbhandari et al.) attacks the redundancy directly. Its core idea, per the slide: split up the expensive parts, the state, and use the reduce-scatter equivalence. The figure, from the ZeRO paper, uses its own notation, which we keep from now on: $\Psi$ is the parameter count, and $K$ is the bytes of optimizer state per parameter. With FP32 master weights and FP32 moments, $K = 4 + 4 + 4 = 12$, so the ledger reads $(2 + 2 + K)\Psi = 16\Psi$.

The figure's example is a 7.5B model on 64 GPUs. Replicated, every GPU holds $16 \times 7.5 = 120$ GB. Then shard more and more:
- shard the optimizer state (stage 1): $4 \times 7.5 + 12 \times 7.5/64 = 30 + 1.4 = 31.4$ GB;
- also shard the gradients (stage 2): $2 \times 7.5 + 14 \times 7.5/64 = 15 + 1.6 = 16.6$ GB;
- also shard the parameters (stage 3): $16 \times 7.5 / 64 = 1.9$ GB.

From 120 GB to 1.9 GB, "quite dramatic". The question for the next two sections is what each step costs in communication. "You might imagine there's no free lunch", the professor says; the remarkable thing is that much of it turns out to be free.

### Stage 1: shard the optimizer state

::slide 19 | stage 1: each GPU keeps the full blue parameters and orange gradients but only a 1/N slice of the green optimizer state; each worker updates only the parameters of its slice

Every GPU keeps the full BF16 parameters and gradients, but holds only $1/N$ of the optimizer state (master weights and both moments). Each GPU is then **responsible for updating one slice** of the parameters, the slice whose state it owns. GPU 0 updates the first slice and nothing else.

::slide 20 | the four steps of stage 1: compute a full gradient on your share of the batch; reduce-scatter the gradients (#params of communication); update your slice with its gradient and state; all-gather the parameters (#params)

The schedule is the all-reduce, cut open:
- **Step 1.** Each GPU computes a full gradient on its own examples, exactly as in naive data parallelism.
- **Step 2.** Reduce-scatter the gradients. GPU $i$ receives the sum over all GPUs of slice $i$ only. That is all it needs: in the professor's words, the other workers "only need the gradients associated with their update". Cost: #params.
- **Step 3.** Each GPU updates its slice, using the summed gradient for that slice and the optimizer state it owns.
- **Step 4.** All-gather the updated parameters, so every GPU has the full new model for the next step. Cost: #params.

::predict zero-stages-1-2

::slide 21 | table: naive DDP uses one all-reduce of gradients; ZeRO stage 1 uses one reduce-scatter (gradients) plus one all-gather (parameters); both cost 2 × #params; memory (4+K)·#params against (4+K/N_gpu)·#params; "ZeRO stage 1 is free (in the bandwidth limited regime)"

Naive data parallelism did one all-reduce: 2 × #params. Stage 1 does a reduce-scatter and an all-gather: also 2 × #params, because that pair *is* an all-reduce. The update has simply moved into the middle. So stage 1 has "the exact same communication characteristics as naive DDP" and saves memory for free: per GPU, $(4 + K)\Psi$ becomes $(4 + K/N)\Psi$.

Notice what is left over, though. The 4 bytes of BF16 parameters and gradients stay replicated. On 8 GPUs with $K = 12$, stage 1 gives $4 + 12/8 = 5.5$ bytes per parameter: a 2.9× saving, not 8×, and no number of GPUs takes it below 4.

::video 18:43-19:04 | two collectives in place of one all-reduce, at the same cost, so the memory saving is free

### Stage 2: shard the gradients too

::slide 22 | stage 2: P_os+g, 2Ψ + (2+K)Ψ/N_d = 16.6 GB; keep the gradients sharded too; "we can never instantiate a full gradient vector, but each worker must compute a full gradient (since we're data parallel)"

"Emboldened by our success, let's shard even more stuff." Now each GPU should keep only its slice of the gradients as well. That looks contradictory. A data-parallel worker computes the gradient of the *whole* model on its examples, yet it is never allowed to hold the whole gradient vector at once.

::slide 23 | stage 2's steps: walk backward through the graph; after computing a layer's gradients, immediately reduce them to the worker that owns them (a reduce diagram); once a gradient is no longer needed, free it; update; all-gather the parameters

The way out is "just a systems trick". The backward pass produces gradients one layer at a time, from the last layer to the first. So as soon as a layer's gradient is computed, reduce it straight to the GPU that owns that slice, and free it as soon as backpropagation no longer needs it. No GPU ever holds more than a layer or so of full gradients. Then update and all-gather the parameters as in stage 1.

The total traffic is the same: the gradients are still reduced once and the parameters gathered once, 2 × #params, just sent in many pieces instead of one. Memory falls to $2\Psi + (2 + K)\Psi/N$: only the 2 bytes of BF16 parameters stay replicated. The worked steps put both stages side by side with DDP:

::worked zero-stages-1-2
::widget fixture:lecture_08--zero-ledger | slide N up: the ZeRO-1 bar shrinks only to its dark replicated floor of 4 bytes per parameter (2 for ZeRO-2), while every data-parallel row still communicates 2Ψ

::note slip 25:09 | Aloud, stages 1 and 2 are both "free in the literal sense". The slide (p27) says stage 2 is "(almost) free (ignoring overhead)". The bytes are the same, but many small per-layer collectives pay more latency and launch overhead than one big one.
::kp zero-stages-1-2

## ZeRO stage 3 (FSDP): shard everything and hide the cost {#fsdp}
source: lecture_08.pdf p24-p28 · video 20:25-28:53

::slide 24 | stage 3, P_os+g+p: every GPU holds a thin 1/N slice of parameters, gradients and optimizer state, (2+2+K)Ψ/N_d = 1.9 GB; "send and request parameters on demand while stepping through the compute graph"; "Is it possible to do this with low overhead?"

The last step, which the professor says first seemed to him like "total magic": shard the parameters as well. Now every GPU holds only $1/N$ of everything. ZeRO stage 3 is better known by PyTorch's name, **FSDP** (Fully Sharded Data Parallel); if you have parallelized anything in PyTorch you have probably used it. The idea is stage 2's, applied to the weights: fetch each layer's parameters on demand just before you need them, and throw them away right after.

::slide 25 | the PyTorch FSDP tutorial's diagram for two GPUs: load the model shard; all-gather the weights; forward (local); free the full weights; all-gather again; backward (local); reduce-scatter the gradients; free; repeat for every FSDP unit; update the local weights; "2 all gather (#param), 1 reduce-scatter (#param)"

Walk one layer through it on two GPUs:
- **Forward.** All-gather the layer's weights so both GPUs hold the full layer. Run its forward on your own examples. Then free the gathered weights: "I don't need these parameters anymore."
- **Backward.** To backpropagate through the layer you need its activations, which you kept, and its weights, which you freed. So all-gather them again, "on demand". Run the backward, reduce-scatter the layer's gradients to their owners, and free the weights again.
- **Update.** Each GPU updates its own slice, as in stages 1 and 2.

::predict zero-stage-3-fsdp

Count the collectives: two all-gathers of the parameters and one reduce-scatter of the gradients. That is 3 × #params against DDP's 2 × #params, so one extra all-gather. The second all-gather exists only because the weights were freed after the forward pass; keeping them would save the communication and give back the memory.

::worked zero-stage-3-fsdp

Two students' questions sharpen the picture. Is this pipelining, passing things from one GPU to the next? No: under FSDP "every GPU goes through the entire model", start to finish, on its own examples. It is the same computation as naive data parallelism; only the parameters are fetched and freed in between. And why doesn't the cost multiply by the number of layers, if every layer does its own collectives? The number of operations does, but each one moves only that layer's parameters, so they add up to the same total as one whole-model collective.

::slide 26 | from the PyTorch FSDP paper: CPU issue stream, GPU compute stream and GPU communication stream for (W1W0 + W2W0)x = y; the all-gathers AG0, AG1, AG2 run on the communication stream ahead of and alongside FWD0, FWD1, FWD2; in backward, AG2 and AG1 again, then RS2, RS1, RS0

Doing communication at every layer sounds terribly slow. Two ideas make it nearly overhead-free. The first we have seen: sweep through the graph, communicate what is needed, and immediately free the memory. The second is **overlapping communication with computation**. Look at the timeline. The GPU has separate streams for compute and for communication. While layer 0's forward runs, the communication stream is already all-gathering layer 1's weights; while layer 1 computes, layer 2's arrive. In backward the same happens with the re-gathers and the reduce-scatters. The example computes $(W_1 W_0 + W_2 W_0)x$, so $W_0$'s forward is used twice, and the gaps show where FSDP frees parameters.

If each layer's compute takes longer than fetching the next layer's weights, the communication hides entirely underneath it. That needs a fast network and enough computation per layer, a condition Part 3 makes precise as a minimum number of tokens per GPU. The professor's verdict from practice: FSDP's GPU utilization is "very close to just the single GPU performance".

::video 23:41-24:11 | the next layer's all-gather is issued while the current layer computes
::note slip 24:36 | "Without paying for almost any amount of memory use": he means communication. FSDP saves memory; the overlap hides the communication.

::slide 27 | "What's the point?": DDP costs 2 × #params; ZeRO stage 1 is 2 × #params, free, "you might as well always do it"; stage 2 is 2 × #params, (almost) free (ignoring overhead); stage 3 is 3 × #params, "1.5x comm cost, but that's not bad! (ignoring latency..)"; FSDP is conceptually simple, a block wrapper

The scorecard: stage 1 is free, so always use it; stage 2 is almost free; stage 3 costs 1.5× DDP's bytes, which overlap mostly hides. The bracket "(ignoring latency..)" is the honest caveat. Each on-demand fetch must arrive before the layer that needs it, so FSDP is sensitive to how *quickly* a collective completes, not only to how many bytes it moves. That is why slow, high-latency links push us toward other strategies later.

FSDP is also conceptually simple: "all you're going to do is do a bunch of all-gathers, compute, free, and then repeat on the backwards pass", wrapped around each block of the model.

::note deferred 25:51 | "You will actually have to write an FSDP implementation as part of your assignment": a wrapper that turns any module into its FSDP version.
::widget fixture:lecture_08--zero-ledger | ZeRO-3 is the only bar with no dark replicated floor, so it keeps falling as (4 + K)/N; its communication reads 3Ψ against DDP's 2Ψ
::kp zero-stage-3-fsdp

### Will it fit?

::slide 28 | "Pure BF16 training (with Kahan summation) is viable … BF16 for everything but the master weights – 12 bytes per param"; on 8× A100 80G: baseline 6.66 B params (12 B/param), stage 1 16 B (5), stage 2 24.62 B (2 + 10/8), stage 3 53.33 B (12/8)

A concrete node: 8 A100s with 80 GB each, and a leaner ledger. With BF16 Adam moments the bill is $2 + 2 + 4 + 2 + 2 = 12$ bytes per parameter. (The slide names **Kahan summation**, compensated summation that carries the rounding error of each addition forward, as what makes pure BF16 training viable; it is not explained further.) The largest model is the GPU's 80 GB divided by the bytes each GPU holds per parameter.

::predict zero-fit-table
::worked zero-fit-table

The pattern to see: only the sharded terms divide by the GPU count. Stage 1 gains only 2.4×, because 4 bytes stay replicated. Stage 2 can never exceed $80 / 2 = 40$B on 80 GB GPUs however many you add. Only stage 3 grows linearly with $N$, which is the "linear memory scaling" Part 1 asked for. Every row ignores activations, buffers and fragmentation, so these are upper bounds.

::widget fixture:lecture_08--zero-ledger | set the Adam moments to BF16 (K = 8), N = 8 and 80 GB: the max-model column reproduces slide 28; push N to 16 and only the ZeRO-3 column doubles

A student asked about other GPUs. The table scales linearly with per-GPU memory: for an H200's 141 GB multiply by 141/80, so ZeRO-3 on 8 H200s fits about $53.33 \times 141/80 \approx 94$B parameters.

::note slip 27:55 | Aloud the table is for "a A100 GPU" and stage 3 fits "50 billion"; the slide is for an 8× A100 node and says 53.33B.
::kp zero-fit-table

## Where data parallelism runs out {#limits}
source: lecture_08.pdf p29-p31 · video 28:53-31:42

"FSDP, I think, is very clean. It's very elegant. I like it a lot. I wish I could end the lecture here." Two problems force us into "uglier and more hairy things".

::slide 29 | "#machines < batch size (and near this, comm overhead is high)"; "diminishing returns to batch sizes"; the predicted-training-speed curve: ε_opt(B)/ε_max against batch size over noise scale B/𝓑, rising linearly ("perfect scaling") below B/𝓑 = 1 and flattening ("ineffective scaling") above it

**Problem 1: data parallelism consumes the batch.** The professor calls batch size "an important resource". With a batch of 8 you can never use more than 8 data-parallel GPUs, because there is nothing left to split. Near that limit, each GPU's share is small and its fixed communication dominates.

Can't we just grow the batch along with the GPU count? Not forever. Past the **critical batch size**, an extra example in the batch is worth less than taking another SGD step on that example. The curve on the slide (it has the shape of the large-batch model of McCandlish et al., 2018) shows it: below the noise scale, doubling the batch nearly halves the number of steps needed ("perfect scaling"); above it, the gains flatten ("ineffective scaling"). "An infinitely large batch size is not infinitely better." So we face a hard trade: small batches and idle GPUs, or big batches and worse optimization.

::video 29:30-29:57 | the definition of the critical batch size: an extra batch element versus an extra SGD step

::slide 30 | "ZeRO stages 1 and 2 don't let you scale memory; ZeRO stage 3 is nice in principle, but does not reduce activation memory"; plot of achieved TFLOP/s per GPU against 768-2,000+ GPUs: PTD-P (Megatron's combined parallelism) stays near 150-170 for 175B and 530B models, while ZeRO-3 falls to about 50

**Problem 2: activations.** Stages 1 and 2 leave a replicated floor. Stage 3 removes it for parameters, gradients and optimizer state, but every GPU still runs whole layers on its examples and keeps all their **activations** for the backward pass. ZeRO does nothing about those, and for big models they dominate, as we will see. The plot (Figure 10 of Narayanan et al. 2021, shown again on slide 60) previews the outcome: at fixed global batch, ZeRO-3 alone degrades badly as GPUs are added, while a combination of model-parallel strategies holds its throughput.

::note slip 30:19 | Aloud: "ZeRO stage 2 lets you cut up the parameter memory." Parameters are sharded only at stage 3, as the slide and his own walk-through say.

::slide 31 | model parallelism "splits up the parameters across GPUs (like zero3).. but communicate activations (while zero3 sends params)"; three types: pipeline, tensor (+ sequence), expert parallel

::predict why-model-parallel

**Model parallelism** splits the parameters across GPUs like ZeRO-3, but in a fundamentally different way. In FSDP, "it was just a wrapper": parameters flew around and the computation was the normal one. In model parallelism each GPU *permanently owns* a piece of the model and computes only with that piece, so what crosses the network are **activations**. If layer 1 lives on GPU 0 and layer 2 on GPU 1, GPU 0 sends layer 1's output to GPU 1. Parameter traffic scales with the model; activation traffic scales with batch × sequence × hidden. And because each GPU runs only part of each example's computation, model parallelism scales memory *without changing the batch size*.

There are three ways to cut a model:
- **pipeline parallel**: cut along depth, by layers;
- **tensor parallel** (plus sequence parallel): cut along width, inside each matrix multiply;
- **expert parallel**: put different MoE experts on different GPUs.

::kp why-model-parallel

## Pipeline parallelism: cut the depth, fill the bubble {#pipeline}
source: lecture_08.pdf p32-p38 · video 31:42-39:27

::slide 32 | layer-wise parallel: layers 0-3 on GPUs 0-3; activations pass forward, partial gradients pass backward

The simplest cut: put layers 0-3 on GPUs 0-3. In the forward pass each GPU sends its output activations to the next; in the backward pass each sends the gradient with respect to its input back to the previous one. Each GPU stores a quarter of the parameters and a quarter of the activations.

::slide 33 | timeline of layer-wise parallelism on 4 GPUs: F0 climbs from GPU 0 to GPU 3, then B0 descends, then all update; "with n GPUs, each GPU is active 1/n of the time"

The timeline is "very, very depressing". Only one GPU works at any moment: the forward pass climbs through the stages, the backward pass climbs back down, and every other GPU waits. With $n$ GPUs each is busy only $1/n$ of the time. The idle region is called the **bubble**.

::slide 34 | the pipelined schedule: the batch cut into 4 micro-batches, F_{i,j} and B_{i,j} for stage i and micro-batch j, staggered so stages overlap, with the bubble between the forward ramp and the backward ramp; "the ratio of bubble time to useful compute is (n_stages − 1)/n_micro, so we need a big batch size!"

The fix is to pipeline. Cut the batch into **micro-batches**. As soon as stage 0 finishes micro-batch 0 it hands it on and starts micro-batch 1, so after a short ramp-up all stages are busy on different micro-batches; the backward pass drains the same way. What remains idle is the ramp-up and ramp-down. With $p$ stages and $m$ micro-batches, the slide's accounting (the GPipe one, with equal-cost steps) is

$$ \frac{\text{bubble time}}{\text{useful compute}} = \frac{n_\text{stages} - 1}{n_\text{micro}} = \frac{p-1}{m} $$

::predict pipeline-bubble

In the slide's picture $p = 4$ and $m = 4$: the bubble is $3/4$ of the useful compute, or $3/(4+3) \approx 43\%$ of the whole step. The numerator is fixed by the depth of the pipeline; the only lever is $m$. "So we need a big batch size!"

::animation fixture:pipeline-bubbles | with 4 stages and one batch, three stages idle at every moment; at m = 4 and m = 8 the micro-batches fill the middle, and only the ramp-up and ramp-down stay idle: the idle fraction is (p−1)/(m+p−1)

::note slip 33:41 | Aloud the bubble ratio becomes "the number of stages divided by this micro-batches" and "1 over the microbatch size". The slide's ratio is (n_stages − 1)/n_micro, and it falls as one over the *number* of micro-batches, not their size.

### The batch is spent twice

Here the professor connects two things the slides keep apart. Batch size is the resource data parallelism consumes, and now the pipeline wants it too: "we can spend it in a different way". At a fixed global batch $B$ (in sequences), with micro-batch size $b$ and $d$ data-parallel replicas, each pipeline gets

$$ n_\text{micro} = \frac{B}{d \cdot b} $$

micro-batches. So adding data-parallel replicas shrinks every pipeline's share and *grows* its bubble. And the critical batch size stops you from simply raising $B$ to compensate.

::predict global-batch-budget
::widget fixture:lecture_08--batch-budget | raise the GPU count G at a fixed global batch: the GPUs become extra replicas, each replica's slice of the batch and with it n_micro shrinks, and the bubble part of the bar grows
::video 33:52-34:13 | batch size as a resource spent twice: by data-parallel replicas and by pipeline micro-batches
::kp global-batch-budget

### Why use pipelines at all?

::slide 35 | "Pipelines seem terrible. Why do we do it?": they save memory compared to DDP; they have good communication properties compared to FSDP, depending only on activations (b × s × h) and point to point; use them on slower links (inter-node)

Folklore, the professor says, holds that parallelization code stays understandable until you implement pipeline parallelism. Two reasons make it worth it anyway:
- **Memory.** Each GPU holds only its stages' layers, and it composes with data parallelism.
- **Communication.** A stage boundary sends one activation tensor of size $b \times s \times h$ (micro-batch × sequence length × hidden size) forward, and its gradient back. It is **point to point**, one GPU to the next, not a collective among everyone. Asked why that is small, the professor answered that $b \cdot s \cdot h$ is "almost always a smaller amount of data" than a whole parameter matrix, which is what FSDP moves.

That makes pipeline parallelism the most communication-efficient strategy, so it goes on the **slowest links**: between nodes, between pods, even between data centers.

::slide 36 | Narayanan et al. 2021: achieved TFLOP/s per GPU against pipeline-parallel size 1-8; at batch size 8 it falls from about 165 to about 87, at batch size 128 it stays near 160-175

The measured version of the bubble formula, from the Megatron paper. With a global batch of 128, going from 1 to 8 pipeline stages costs little. With a batch of 8 it almost halves throughput (about 165 to 87 TFLOP/s per GPU), because 8 stages with a handful of micro-batches are mostly bubble.

::slide 37 | "Trading communication bandwidth for utilization": a 4-device 1F1B schedule (forward in blue, backward in green) above an interleaved schedule where each device holds several smaller stages, with a shorter idle region

People have tried "much more clever ways" to shrink the bubble by scheduling. In the schedule at the bottom, each device is assigned several smaller, non-adjacent stages instead of one big one, so micro-batches enter and leave the pipeline in smaller steps and the ramps shrink. The price is in the slide's title: more stages means more stage boundaries, so more activation traffic. (In the Megatron paper, $v$ chunks per device cut the bubble by $v$ and multiply the communication by $v$.)

::note aside 37:01 | The professor attributes this figure to "the DeepSeek paper". Its layout (a default 1F1B schedule above an interleaved schedule with multiple stages per device) matches Figure 4 of Narayanan et al. 2021, the Megatron paper; DeepSeek-V3's DualPipe schedule looks different. Either way, the idea on the slide is interleaving.

::slide 38 | zero-bubble pipelining: split the backward pass into B (backpropagating activation gradients, z and x) and W (computing weight gradients); "the second part can be done whenever"; the MLP computation graph with F, B and W, and the 1F1B against ZB-H1/ZB-H2 schedules

The cleverest trick is not about scheduling as such but about the structure of the backward pass. At every node of the graph, backprop does two things:
- **B**: propagate the gradient to the node's *input*, so the previous stage can continue. That is urgent: the previous stage "can't do any work until you've propagated that signal back".
- **W**: compute the gradient for the node's *weights*. Nothing downstream waits for it; the weight gradient is "a leaf node, so to speak". It can be done whenever.

So run all the B's as early as possible and push the W's into the gaps the bubble would have left. Depending on how the B and W work balance, this fills the pipeline almost completely. It is also, in his words, "much more complicated than you would normally like to deal with".

::video 38:11-38:50 | why the backward-propagation half must run promptly but the weight-gradient half can wait
::note skip 38:56 | The zero-bubble schedules themselves (ZB-H1, ZB-H2) are not walked through; only the B/W split is taught.
::kp pipeline-bubble

## Tensor parallelism: cut the width {#tensor}
source: lecture_08.pdf p39-p43 · video 39:27-44:55

::slide 39 | a 2×4 matrix X times a 4×2 matrix A gives Y = [[74, 98], [258, 346]]; split X into column blocks X1, X2 and A into row blocks A1, A2: X1·A1 = [[11, 15], [95, 131]] and X2·A2 = [[63, 83], [163, 215]] add up to the same Y

Pipelining cuts the network along depth. The other axis is width: cut *inside* each matrix multiply. This is **tensor parallelism**, which [L7](#/read/lecture_07) implemented on an MLP. It rests on one observation: a matrix multiply splits into smaller matrix multiplies whose partial sums add back up.

The slide's numbers check out. Row 0 of X is [0, 1, 2, 3] and column 0 of A is [10, 11, 12, 13], so $Y_{00} = 0 + 11 + 24 + 39 = 74$. Split the inner dimension in half: $X_1 A_1$ uses only the first two terms ($0 \cdot 10 + 1 \cdot 11 = 11$), $X_2 A_2$ the last two ($2 \cdot 12 + 3 \cdot 13 = 63$), and $11 + 63 = 74$. Two GPUs can each compute one half, and a sum puts them together. The professor notes it is "the same idea as tiling" from [L5](#/read/lecture_05): this core primitive "just appears many, many times".

::slide 40 | Megatron-LM's MLP on two GPUs: X enters f and is used by both; GPU i computes Y_i = GeLU(X A_i) with A = [A1, A2] split by columns, then Z_i = Y_i B_i with B split by rows; g combines Z_1 + Z_2, then dropout; "in the forward pass, f is the identity and g is an all-reduce; in the backward pass f is an all-reduce, g is the identity"

Now a real layer, the MLP $Z = \text{Dropout}(\text{GeLU}(XA)\,B)$. Megatron-LM's recipe splits the first matrix $A$ by **columns** and the second $B$ by **rows**.

::predict tensor-parallel-megatron
::worked tensor-parallel-megatron

Why this order? Splitting $A$ by columns gives each GPU complete columns of the hidden layer, so the GeLU, which is elementwise, can be applied locally with no communication. Splitting $B$ by rows then matches: GPU $i$ owns exactly the rows of $B$ that multiply its columns of $Y$. The one unavoidable sum comes at the very end. The reverse order would put a sum in front of the GeLU, because the GeLU of a sum is not the sum of GeLUs, and cost two collectives instead of one.

The backward pass mirrors the forward. Gradients flow from $Z$ back toward $X$. At $g$ the incoming gradient is already whole and each GPU just uses it: identity. At $f$ each GPU has computed only its part of $\partial L/\partial X$, through its own columns of $A$, and those parts must be summed: all-reduce. "This duality is important if you're going to write tensor parallel."

::video 40:51-41:31 | f and g swap roles between the forward and backward pass

::slide 41 | the same pattern for (a) the MLP and (b) self-attention, where the heads are split across GPUs (Q = [Q1, Q2], K = [K1, K2], V = [V1, V2]) and the output projection is split by rows; columnwise: QKV, up-projection; rowwise: attention output, down-projection; replicated: norms, routers

Attention follows the same pattern. The QKV projection is split by columns, which means by heads: each GPU computes its own heads' attention completely locally. The output projection is split by rows, followed by one all-reduce. So a Transformer block has:
- **column-wise**: the QKV projection and the MLP up-projection;
- **row-wise**: the attention output projection and the MLP down-projection;
- **replicated**: small things like LayerNorms and MoE routers, which are not worth the overhead of cutting.

Each block thus pays two all-reduces in forward (one per sublayer) and two in backward.

::slide 42 | "On GPUs, tensor parallel within a node (up to 8 GPUs)"; bar chart of tokens/s per GPU for a 3B model against TP = 2, 4, 8, 16, 32: drops of 10.8% and 12.2% up to TP = 8, then 42.7% at 16 and 65.6% at 32

When should we use it? Every matrix multiply now comes with an activation-sized all-reduce, many times per layer. Tensor parallelism is, after a self-correction from "computation hungry", "very communication hungry". So it belongs where communication is cheapest: inside the 8-GPU NVLink node. The chart shows the cliff. Up to TP = 8 each doubling costs 11-12% of throughput per GPU; at TP = 16, once the all-reduces must cross nodes, it costs 43%, and at 32 another 66%.

TPUs are the exception. A TPU pod has no "8 fast, then slow" boundary, just one big mesh with high bandwidth for this regular, neighbour-friendly pattern. So TPU users "can tensor parallel very large numbers compared to the GPU world", and the balance of tensor against pipeline parallelism differs between the two kinds of hardware (slide 10's "just do tensor parallel").

::video 43:26-43:57 | why TPUs can run much more tensor parallelism than GPUs

::slide 43 | tensor parallel against pipeline parallel: pros (no bubble, low complexity, no need for large batches) and cons ("much larger communication"): pipeline bsh point-to-point per micro-batch; tensor 8bsh (n_devices − 1)/n_devices per layer, all-reduce; "use tensor parallel whenever we have low-latency, high-bandwidth interconnects"

The comparison with pipelining, which is also a model-splitting strategy:
- **Pros of tensor parallel.** No bubble: if the network is fast enough, nobody waits. Low complexity: it is "just cutting up some matmuls", and you can wrap a model without major infrastructure changes. And it does not need large batches.
- **Cons.** Much more communication. A pipeline stage boundary moves $bsh$ per micro-batch, point to point. Tensor parallelism moves $8bsh\,(n-1)/n$ **per layer**, in blocking all-reduces.

The 8 is not derived on the slide. It is 2 all-reduces per layer (attention and MLP) × 2 passes (forward and backward) × 2, because a ring all-reduce moves about twice its tensor; $(n-1)/n$ is the ring's share.

::predict tp-vs-pp-communication
::worked tp-vs-pp-communication
::widget fixture:parallelism-comm | for fixed b, s and h, the tensor-parallel bar (8bsh(n−1)/n per layer, for every layer and micro-batch) stays far above the pipeline bar (bsh per boundary per micro-batch); only the data-parallel bar moves when you change the parameter count

So: tensor parallelism on the fast links, pipeline parallelism everywhere else.

::note slip 44:33 | The captions read the tensor-parallel cost as "a times b times s times h"; the slide says 8bsh(n−1)/n, so "a" is probably "eight" mis-captioned.
::kp tensor-parallel-megatron
::kp tp-vs-pp-communication

## Activation memory, and sequence parallelism {#activations}
source: lecture_08.pdf p44-p49 · video 44:55-53:21

::slide 44 | a PyTorch memory profile over a few training steps: a flat green band of parameters, a yellow band of optimizer state appearing after the first step, red humps of activations that rise through each forward pass, and blue gradients that grow as the backward sweep begins

The naive view is "memory is just parameters", the green band. We have added the optimizer state, the yellow band. But the profile shows a large *dynamic* part. Activations (red) pile up during each forward pass, and gradients (blue) appear as the backward pass starts. The peak comes "a little bit after the maximum activation point": early in the backward sweep, when most activations are still held and gradients have begun to accumulate.

::slide 45 | Korthikanti et al. 2022: memory per GPU for 22B, 175B, 530B and 1T models, baseline against "present work"; blue parameters and optimizer state, green activations, an 80 GB line that every baseline bar crosses and every present-work bar stays under

For big models at moderately long sequences, activations "just dwarf the parameter memory": in every baseline bar green is the larger part, every baseline total is over the 80 GB line, and at 530B and 1T the activations alone are over it. Tensor and pipeline parallelism already split the parameter memory linearly. Any memory strategy that ignores activations cannot be fully effective.

::note deferred 46:17 | "Forget the column that says present work": the paper's improved bars are what the next slides build up to (tensor + sequence parallel + selective recomputation).

::slide 46 | "Activations memory per layer = sbh(34 + 5as/h)", storing everything; the 5as/h terms come from the quadratic attention terms including dropout; "as with flash attention, we can drop this term via recomputation"; legend: a heads, b micro-batch, h hidden, L layers, p pipeline size, s sequence, t tensor-parallel size, v vocabulary

The rule of thumb from Korthikanti et al., in bytes with 2-byte activations:

$$ \text{activation memory per layer} = sbh\left(34 + \frac{5as}{h}\right) $$

The $sbh$ factor is "fundamental": we expect to store something for every position, every batch element and every hidden unit. The 34 counts the linear terms: inputs of each sublayer, Q, K and V, the MLP's hidden activations, dropout masks and so on. The $5as/h$ term is the attention scores, softmax and its dropout, $a$ heads × $s$ positions per token, so it grows quadratically with sequence length. As in [FlashAttention](#/read/lecture_05), it can be dropped by recomputing it in the backward pass.

::predict activation-memory-tp

::slide 47 | "Activations memory per layer = sbh(10 + 24/t + 5as/(ht))" under tensor parallelism; the remaining 10 is LayerNorm (4sbh), dropout (2sbh) and the inputs to attention and MLP (4sbh); "these terms alone will continue to grow with size"

Tensor parallelism splits the matrix multiplies in attention and the MLP, so it divides their activations by $t$: the 24 linear terms and the attention term. What it cannot divide are the 10 $sbh$ of LayerNorm (4), dropout (2) and the inputs to attention and the MLP (4). These sit outside the split regions, and every tensor-parallel rank holds them whole, since the layer inputs "need to be stored as residuals for the backwards pass". Even with "1,000 different GPUs in tensor parallel because you're Google", you still pay 10 $sbh$.

::worked activation-memory-tp
::note slip 48:11 | Aloud, "the MLPs are 24 of those 34". In Korthikanti et al. the 24 is 16 sbh from the MLP plus 8 sbh from attention's matmul inputs (Q, K, V and the output projection). The formula is unaffected.
::kp activation-memory-tp

### Sequence parallelism: split the rest along the sequence

::slide 48 | one Transformer layer as alternating regions: LayerNorm and dropout regions marked "Sequence Parallel", attention and MLP regions marked "Tensor Parallel", with g and ḡ at each boundary; "all the 10sbh terms are pointwise ops over the sequence"; in the forward pass g is an all-gather and ḡ a reduce-scatter, reversed in backward

The remaining 10 $sbh$ belong to cheap operations, LayerNorm and dropout, that act on each position independently. So split them along the **sequence** axis instead of the hidden axis: each of the $t$ GPUs keeps $s/t$ of the positions. (The professor calls the name "extremely misleading"; context parallelism, later, would deserve it more.)

The boundaries change accordingly. Entering a tensor-parallel region, the attention or MLP needs every position's full hidden vector, so $g$ all-gathers the sequence shards. Leaving it, the partial sums that plain TP would all-reduce are instead reduce-scattered by $\bar g$, so each GPU gets the summed result for just its positions. In backward the two swap. The professor finds it "very reminiscent of FSDP": store sharded, materialize on demand.

::predict sequence-parallel

It is also free in bandwidth, by the identity from the start of the lecture: plain TP did one all-reduce at $g$; sequence parallelism does an all-gather and a reduce-scatter, which together are an all-reduce.

::worked sequence-parallel
::animation fixture:lecture_07--tp-gather-scatter | the same duality L7 animates: a forward all-gather takes each rank from its slice to the whole tensor, and the backward reduce-scatter runs it in reverse, from partial sums back to one summed slice per rank

::slide 49 | Korthikanti et al.'s table of activation memory per layer: no parallelism sbh(34 + 5as/h); tensor parallel sbh(10 + 24/t + 5as/(ht)); tensor + sequence parallel sbh(34/t + 5as/(ht)); tensor parallel + selective recomputation sbh(10 + 24/t); tensor + sequence parallel + selective recomputation sbh(34/t)

The table puts it together. With tensor and sequence parallelism every term divides by $t$: "fully linear dependence". Recompute the attention term as well and the bill is

$$ sbh \cdot \frac{34}{t} $$

which the professor recommends remembering as "the lower bound of what you can achieve, reasonably speaking" for activation memory in normal training. To check by hand whether a model fits, add this to the parameter, gradient and optimizer bytes from the ZeRO sections.

::widget fixture:lecture_08--activation-memory | raise t with tensor parallel alone: green and orange shrink but the blue 10 sbh stays, so the per-GPU bar stays above store-all ÷ t; tick sequence parallel and the bar becomes exactly store-all ÷ t
::video 51:21-51:59 | sbh × 34/t as the activation-memory lower bound to remember
::kp sequence-parallel

### Why recompute attention and not the MLP?

A student asked why only the attention term is recomputed. You *can* recompute the MLP too, the professor said, but that means "running the MLP again in the backward pass, which you probably do not want to do". Recomputing attention "is generally cheaper", and saves the quadratic cost. Korthikanti et al. call this **selective recomputation**, and the numbers show why it is the right choice.

::predict selective-recomputation

A layer's forward pass costs about $24bsh^2 + 4bs^2h$ FLOPs. The MLP's two matmuls are $16bsh^2$ of that; the attention core ($QK^\top$ and attention × V) is $4bs^2h$. For a GPT-3-like layer ($h = 12288$, $a = 96$, $s = 2048$):
- recomputing the attention core adds $s/(6h + s) = 2048/75776 \approx 2.7\%$ of the forward FLOPs, and frees the $5as/h = 80$ term, which was $80/114 \approx 70\%$ of the layer's stored activations;
- recomputing the MLP would add $16h/(24h + 4s) \approx 65\%$ of the forward FLOPs, to free only part of the linear 34.

The cheapness rests on $s$ being small next to $6h$. At $s = 32768$ the attention core is 97% of the stored activations, but its recomputation now costs about 31% of the forward FLOPs.

::widget fixture:lecture_08--activation-memory | tick "recompute the attention core": the orange segment disappears for a FLOP share s/(6h+s); raise s and watch both the memory share and the recompute share grow
::video 52:44-53:15 | why the MLP is not recomputed: it means running the MLP again, while attention is cheaper
::kp selective-recomputation

## Expert parallelism, and the strategies the lecture skips {#experts}
source: lecture_08.pdf p50-p55 · video 53:21-1:03:58

::slide 50 | GShard's MoE Transformer encoder with device placement: attention layers replicated on every device, the MoE feed-forward layer's experts FFN_1 … FFN_E spread one per device, with an "All-to-All Dispatch" before and an "All-to-All Combine" after; "Instead of splitting up the matmul, split up the experts and route activations"

The last standard ingredient. Most big models are now **mixtures of experts** ([L4](#/read/lecture_04)): each MLP is replaced by many expert MLPs, and a router sends each token to a few of them. That structure offers a natural cut. **Expert parallelism** (EP) puts different experts on different GPUs and sends each token to the GPU that holds its expert: an all-to-all **dispatch** before the expert layer and an all-to-all **combine** after it to bring the outputs home. The experts' matrices themselves stay whole.

::slide 51 | "EP is roughly like TP in behavior for MLPs – high bandwidth, reduces activation"; Megatron's Guideline 4, "Prefer EP over TP for Expert Layers": better GEMM efficiency (larger local matrices), lower communication, a simpler graph that overlaps more easily, no token permutation when EP = number of experts; example: for Mixtral 8x7B, EP8×TP1 outperforms EP4×TP2

The professor thinks of EP as an analogue of tensor parallelism for the MLPs: a high-bandwidth strategy that splits the MLP across devices and also cuts activation memory. But for MoE layers "you almost always prefer" EP over TP, as NVIDIA's Megatron guidelines say. The main reason is matmul size. Cut a matrix "too finely" and "your GPU utilization will suffer", so "you want your matmuls as big as possible". TP = 8 makes every expert's GEMM 8 times narrower; EP = 8 keeps each expert's GEMM at full size and moves the tokens instead. Routing sparse token activations is also cheaper than all-reducing the dense outputs of sliced matmuls.

::predict expert-parallel
::worked expert-parallel
::video 54:53-55:22 | why routing tokens beats cutting matrices for MoE layers: small matmuls waste the GPU

That makes EP sound easy. It is not. The all-to-all happens at every MoE layer, and it is "very latency sensitive": "your computation is waiting for your tokens to arrive". Getting it fast takes heroic engineering. DeepSeek wrote its own dispatch library (DeepEP) that leans on low-level GPU networking primitives, and NVIDIA has a similar effort called Hybrid EP: "seriously complicated business".

::note skip 56:06 | The two libraries are named to show the difficulty, not explained. The trivia that goes with them: to squeeze out the last bit of networking speed, DeepSeek's engineers found and used undocumented PTX instructions (GPU machine-code-like instructions).

::slide 52 | "DP usually shares replicas with EP splits (so EP<DP)"; "DP and TP can interact badly to lower utilization"; a survey figure of MoE layouts: data + expert, data + expert + tensor, data + expert + pipeline, expert + tensor parallelism

Most strategies compose like LEGO blocks; EP has extra constraints. The usual (and older libraries') way to combine it with data parallelism is to reuse the data-parallel groups. With data parallelism 8, the experts are sharded across those same 8 replicas, and each replica's tokens are routed among them. So the EP group lives inside the DP group, and EP can be at most DP: the slide's "EP<DP" (read it as EP ≤ DP; equality is the limiting case).

::slide 53 | "MoEs apply to the MLPs, not the attention"; high TP is useful for attention, low TP for the MLPs; Megatron's "MoE Parallel Folding" decouples them: attention layers use TP × CP × DP × PP, MoE layers use ETP × EP × EDP × PP

The final complexity is an imbalance inside the model. MoE changes only the MLPs. Attention cannot use EP, so to split attention you want a high TP. But a high TP would also slice the experts, and high TP plus high EP leaves "really tiny pieces" with very bad utilization. You want high TP for attention and low TP for the MLPs.

Megatron's answer is to **decouple** them. The same GPUs are grouped one way for the attention layers (tensor × context × data × pipeline) and another way for the MoE layers (expert-tensor × expert × expert-data × pipeline). Both products must equal the GPU count. A worked check from slide 69's table: DeepSeek-V3 in Megatron runs on 1,024 GPUs with TP 2, PP 16, CP 1, EP 64. For attention, $\text{DP} = 1024/(2 \times 16) = 32$. For the experts, with expert-TP 1, $\text{EDP} = 1024/(64 \times 16) = 1$. So EP = 64 exceeds the attention-side DP of 32: folding lets the expert layers use the GPUs that attention spends on TP, which slide 52's naive rule would not allow.

::kp expert-parallel

::slide 54 | context parallelism / ring attention: two devices each holding a query block, passing key-value blocks around a ring

**Context parallelism** (or ring attention) splits one long sequence across GPUs. Each device keeps a block of queries, and the key/value blocks are passed around a ring so every query block eventually sees every key; the original ring-attention work showed it running well on TPU meshes. It is used in long-context extension and in serving. The professor skips the details because they overlap with what came before.

::note skip 1:01:44 | Context parallelism / ring attention is named and skipped: "it overlaps in concept to a lot of what we've talked about already".

::slide 55 | recap table, drawbacks in red: for DDP/ZeRO-1, FSDP/ZeRO-3, pipeline, tensor, sequence/context and expert parallel, what each communicates, its parameter and activation memory per rank, its main bandwidth cost, whether it scales the global batch, and how easy it is to use

The recap table is the professor's own, with what he subjectively sees as drawbacks in red. The point of the red is that "there is no one strictly dominant parallelization strategy":

| method | communicates | splits | consumes the global batch? | ease |
|---|---|---|---|---|
| DDP / ZeRO-1 | gradient all-reduce per step | only optimizer state (ZeRO-1) | yes | very easy |
| FSDP / ZeRO-3 | parameter all-gathers, overlapped | params, grads, state, ~1/DP | yes | moderate |
| pipeline | activations between stages; bubbles | ~1/PP of the layers | no, but needs micro-batches | hard |
| tensor | blocking activation-sized collectives every block | ~1/TP of weights and matmul activations | no | hard |
| sequence / context | per-layer sequence-shard exchange | sequence-side activations / KV | no | hard |
| expert | token all-to-all per MoE layer | ~1/EP of expert weights | no, but needs enough tokens per expert | hard |

Read along the batch column. FSDP is wonderful, but it does not help with activations and it consumes the global batch. Tensor parallelism cuts activations and "doesn't touch global batch size", but needs fast, high-bandwidth links. Pipelining uses the slow links. Each strategy's weakness is covered by another, which is why large runs combine them.

A question afterwards: do pipeline parallelism and FSDP still apply to MoEs? Yes, extensively, in every frontier model. The old advice to keep EP within the 8 fast GPUs, like TP, is one "people don't really follow anymore".

## Putting it together: 3D (or 4D) parallelism {#recipe}
source: lecture_08.pdf p56-p62 · video 1:03:58-1:12:26

One nice thing about parallelism, the professor says, is that "you can do some math". For each strategy, count the compute and the communication per layer; then see how a combination scales. You will do something like this in the assignment.

::note deferred 1:03:58 | The per-layer compute and communication math for combined strategies is left to the assignment.

::slide 56 | from the JAX scaling book: per-layer compute and communication for DP, FSDP, model parallelism (MP) and FSDP + MP; plot of FLOPs time ÷ communication time against B/N, the batch per chip, on a 4×4×4 TPU mesh: FSDP-only rises and crosses the compute-bound line near 850, MP-only is flat below it, FSDP + MP crosses near 400; "no scheme works when B < 400", "only mixed FSDP + MP works when B < 850", "both work when B > 850"

The key quantity is the **global batch divided by the number of chips**. Communication can be hidden under compute only if compute takes longer, so plot the ratio compute time ÷ communication time: above 1 the job is compute-bound, below 1 it waits on the network.

::predict fsdp-batch-per-chip-threshold

Why the batch per chip? Under FSDP each chip's communication per step is set by the parameter count: gather the parameters twice, reduce-scatter the gradients once, whatever the batch. Its compute is set by parameters × the tokens it processes. So:

$$ \frac{\text{compute time}}{\text{communication time}} \approx \frac{6N \cdot T / C}{3 \cdot 2N / W} = \frac{T \cdot W}{C} $$

with $N$ parameters, $T$ tokens per chip per step, $C$ the chip's FLOP/s and $W$ its link bandwidth in bytes/s. The parameter count cancels; only the tokens per chip and the hardware remain. FSDP alone is compute-bound when $T > C/W$. (This derivation is ours, following the scaling book; the slide shows only the plot and its table.)

Below that threshold FSDP cannot hide its traffic. Then you add model parallelism, which here means tensor parallelism: with tensor parallelism of degree $t$ inside fast links, each chip gathers only $1/t$ of the parameters, so the threshold falls to about $C/(tW)$, and the combined curve stays compute-bound at smaller per-chip batches. The plot shows exactly this. FSDP-only crosses into compute-bound territory at about 850 per chip, FSDP + MP at about 400, and model parallelism alone never gets there: its communication is activations, which grow with the batch just as its compute does, so its line is flat. "You keep adding strategies to push this curve out."

::widget fixture:lecture_08--fsdp-threshold | change N and both time readouts move but their ratio does not: only tokens per chip, the link, the chip and t move the point across the line
::video 1:04:41-1:05:59 | compute must outlast communication; FSDP alone fails as the per-chip batch shrinks, and tensor parallel pushes the curve out
::note slip 1:05:20 | The professor's example, "a batch size of 2,000 per chip", is a point where FSDP alone is comfortably compute-bound, at the right edge of the plot. It is not the threshold, which on this plot is about 850 per chip.
::kp fsdp-batch-per-chip-threshold

::slide 57 | "3D (4D) parallelism", simple rules of thumb: until the model fits, tensor/expert parallel up to the GPUs per machine and pipeline parallel across machines (or ZeRO-3, depending on bandwidth); then scale the rest of the way with data parallel; if the batch is small, gradient-accumulate; figure of two data-parallel ranks, each a 3-stage pipeline whose stages are 4-way tensor-parallel

After all the generalities, the prescription is "very simple":
1. **Until the model fits in memory**, cut it up "by whatever means necessary": tensor or expert parallelism over the fast links, up to the GPUs per machine (8), then pipeline parallelism, or ZeRO-3 depending on bandwidth, across machines.
2. **Then, until you run out of GPUs**, use data parallelism for the rest.

If the batch ends up too small for good communication efficiency, use gradient accumulation (see [L2](#/read/lecture_02)) to trade a larger effective batch for fewer synchronizations.

::predict three-d-parallelism-recipe
::worked three-d-parallelism-recipe
::widget fixture:lecture_08--batch-budget | set TP = 8 and enough pipeline stages to fit, then raise G: every further GPU becomes a data-parallel replica, DP = G/(TP × PP), and each replica's share of the batch B/DP falls
::video 1:06:27-1:07:07 | the order: model-parallel inside the node until it fits, pipeline or ZeRO-3 across nodes, then data parallel

::slide 58 | Megatron's current guidelines: (1) minimize model parallelism, maximize data parallelism; (2) keep EP and TP within the NVLink domain; (3) use pipeline parallelism for multi-node scaling; (4) prefer EP over TP for expert layers; (5) enable context parallelism for long sequences (≥ 8K tokens)

NVIDIA's guide to parallelizing MoEs says "exactly the thing that I said, but in reverse order", and it applies to dense models too: maximize data parallelism, keep EP and TP inside the NVLink box, cross nodes with pipelines, prefer EP for MoE layers, and use context parallelism for long sequences.

::note aside 1:08:05 | Two questions here. Sequence parallelism "is more of an add on" to tensor parallelism, not a standalone strategy. And would a looped (weight-shared) Transformer, as rumours suggest for some frontier models, change things? Probably for FSDP, which is built on "get weights, discard weights"; a loop reuses the same weights, and its parameter efficiency might make much of the model parallelism less important. The professor is sceptical of the rumours.

::slide 59 | Narayanan et al. 2021, Table 1: models from 1.7B to 1008B parameters; tensor-parallel size 1, 2, 4, 8, then 8 throughout; pipeline size 1 to 64; 32 to 3,072 GPUs; 43-52% of theoretical peak; DP size 32, 32, 32, 32, 32, 32, 24, 15, 9, 6

The recipe as it played out in a real scaling study (Narayanan et al. 2021, by NVIDIA and then-Stanford researchers including Deepak Narayanan and Matei Zaharia; old, the professor says, but still one of the best resources, since networking fundamentals have not changed much). Data parallelism starts maxed out at 32. Tensor parallelism grows with the model until it hits 8, then stops. From there pipeline parallelism grows (2, 4, 8, 16, 35, 64) to make the model fit. At the largest scale data parallelism shrinks, down to 6 for the 1-trillion-parameter model, because that much tensor and pipeline parallelism is needed just to fit it. Check the last row: $3072 / (8 \times 64) = 6$.

::slide 60 | Figure 10 of the same paper: TFLOP/s per GPU against 768-2,000+ GPUs; PTD-P holds 150-170 for the 175B and 530B models while ZeRO-3 without model parallelism falls toward 50

"More GPUs, same, flat utilization!" Combined carefully, the strategies keep per-GPU throughput almost constant even at "ludicrously large numbers of GPUs", while ZeRO-3 alone collapses. The professor sees this as the reason behind today's gigantic data-center build-outs: the strategies and the communication hardware are good enough to use them, even across data centers.

::slide 61 | Figure 13: TFLOP/s per GPU for a 162.2B GPT on 64 A100s over (pipeline, tensor) = (2, 32), (4, 16), (8, 8), (16, 4), (32, 2), at batch 32 and batch 128; both peak at (8, 8), about 165 at batch 128

On 64 GPUs, the best split is "clearly tensor parallel size of 8", the (8, 8) configuration. More tensor parallelism than 8 crosses the node boundary, "you're going to get into trouble". More pipeline parallelism needs bigger batches, which is why the gap between batch 32 (blue) and batch 128 (orange) widens as PP grows.

::note slip 1:11:20 | The slide says "64 machines"; the figure's caption says 64 A100 GPUs, which is eight 8-GPU machines.

::slide 62 | throughput in sequences per second against batch size 1-256, with and without activation recomputation (t = 8, p = 16): without recomputation the curve stops at batch 8, near 4; with it the curve continues to about 7.8 at batch 256

A counterintuitive one. Recomputation does *more* arithmetic, so at a given batch it is a little slower (3.0 against 3.9 sequences per second at batch 8). But it frees memory, and "memory can be turned into batch size". Without recomputation this configuration runs out of memory past batch 8; with it the batch can grow to 256, and throughput roughly doubles to about 7.8. Spending FLOPs on recomputation can raise utilization.

::video 1:11:51-1:12:26 | why spending FLOPs on recomputation can raise utilization: memory becomes batch size
::kp three-d-parallelism-recipe

## What recent models actually run {#in-the-wild}
source: lecture_08.pdf p63-p73 · video 1:12:26-1:20:05

The last part looks at published training runs, to see the recipe in practice and how it is changing.

::slide 63 | OLMo's "Distributed Training Framework" section: ZeRO via PyTorch FSDP, a micro-batch of 4096 tokens per GPU, a global batch of about 4M tokens (2048 sequences of 2048) for the 1B and 7B models; the caption "Dolma – 7B model, FSDP (probably fits intra-node)"

**OLMo (7B).** The slide labels it Dolma; aloud the professor corrects himself: OLMo, "which was trained on the Dolma data set". It was trained with plain FSDP "across a whole bunch of accelerators", an example of how FSDP "scales surprisingly well" as long as the model is small. Many 7B-class models are trained with FSDP alone.

::note why 1:12:57 | The slide says FSDP "(probably fits intra-node)", while aloud OLMo ran across many accelerators. Both can hold: a 7B model's sharded state fits in one 8-GPU node, so the sharding group can stay inside a node while data parallelism spans many nodes (hybrid sharding). The slide is about the shard group, the professor about the whole job.

::slide 64 | DeepSeek's infrastructure paragraph (HAI-LLM: data, tensor, sequence and 1F1B pipeline parallelism as in Megatron, ZeRO-1 for the optimizer state, overlap of communication and computation); "V3 – PP (16), EP (64-way, 8 nodes), ZeRO stage 1; EP uses 1F1B A2A overlap"

**DeepSeek.** The first DeepSeek LLM (dense) used ZeRO stage 1 with tensor, sequence and pipeline parallelism: the classic combination. DeepSeek-V3, an MoE, keeps the 16-way pipeline and ZeRO-1 but replaces tensor parallelism with **64-way expert parallelism** spanning 8 nodes, far beyond the "keep EP inside one box" advice. To make that work they reuse their pipelining tricks to overlap the expert all-to-all with computation, so the large EP does not leave idle periods.

::slide 65 | Yi's paper: ZeRO-1, tensor parallel within each node combined with pipeline parallel, kernel fusion, topology-aware allocation; "Yi-lightning (2025) – Tensor replaced by Expert parallelism"

**Yi.** Again ZeRO-1 + tensor + pipeline, "the classic" combination; and once Yi moved to MoE (Yi-Lightning), tensor parallelism was replaced by expert parallelism. They serve similar goals, but EP is "a little bit more efficient".

::slide 66 | Llama 3 405B's parallelism table: 8,192 GPUs at TP 8, CP 1, PP 16, DP 64, sequence 8,192, 430 TFLOP/s per GPU, 43% MFU; 16,384 GPUs at DP 128, 400 TFLOP/s, 41%; long-context stage 16,384 GPUs at CP 16, PP 16, DP 8, sequence 131,072, 380 TFLOP/s, 38%; the paper's ordering [TP, CP, PP, DP] from innermost (fastest links) to outermost

**Llama 3 405B**, a giant dense model, is one of the few reports with a full breakdown for every phase. Ignore the first row, a small-batch warm-up stage. The main pretraining row is "a very standard parallelization strategy": TP 8, CP 1, PP 16, DP 128.

::predict recent-lm-configs

For long-context extension (131K-token sequences) they raise context parallelism to 16 and lower data parallelism to 8: the GPU count is fixed, so the new CP comes out of DP. The paper's ordering, [TP, CP, PP, DP] from innermost to outermost, is the bandwidth hierarchy again. TP needs the fastest links; DP (here FSDP) is outermost because it tolerates latency by prefetching parameters asynchronously.

::slide 67 | Llama 3 405B's interruption table for a 54-day period: faulty GPU 148 (30.1%), GPU HBM3 memory 72 (17.2%), software bug 54, network switch/cable 35, …; about 78% attributed to confirmed or suspected hardware issues

An aside the professor wants infrastructure engineers to know: "GPUs fail all the time". Faulty GPUs caused 148 interruptions during Llama 3 405B pretraining, the largest single cause; about 78% of all unexpected interruptions were hardware. Fast parallelism is not enough; you also need redundancy and recovery. Fault tolerance is not covered in the course.

::slide 68 | Gemma 2's compute infrastructure: 2B on a 2×16×16 TPUv5e configuration (512 chips, 512-way data replication, 1-way model sharding); 9B on 8×16×32 TPUv4 (4,096 chips, 1024-way data, 4-way model); 27B on 8×24×32 TPUv5p (6,144 chips, 768-way data, 8-way model); optimizer state sharded like ZeRO-3; "ZeRO-3, MP (=TP+SP), DP"

**Gemma 2** (Google, on TPUs) uses only FSDP-style sharding plus model parallelism (tensor + sequence) and data parallelism; no pipelines. For the 27B model, $768 \times 8 = 6144$ chips. The professor reads it as the Google claim realized: on a TPU mesh "you don't need to do pipelines", you tensor-parallel over the big torus. Whether that "can scale out forever" is less clear to him, but at Gemma's scales it works.

::video 1:16:24-1:16:59 | Gemma 2 as the TPU strategy: FSDP plus tensor parallel over the mesh, no pipeline

::slide 69 | Megatron's recommended MoE configurations: Mixtral 8x7B on 64 GPUs (TP 1, PP 4, CP 1, EP 8), Mixtral 8x22B on 256 GPUs (TP 4, PP 4, CP 1, EP 8), DeepSeek-V3 671B on 1,024 GPUs (TP 2, PP 16, CP 1, EP 64); "DP likely 2 (to add up to 256 GPUs)"

**Mixtral 8x22B**, from the configurations NVIDIA publishes in its Megatron Bridge repository: EP 8, PP 4, and TP 4, the TP "for the attention layers". It follows the advice to keep EP around 8.

::note why 1:17:39 | The slide's "DP likely 2" multiplies all four degrees, 4 × 4 × 1 × 8 = 128, and doubles to 256. With experts sharing data-parallel replicas (slide 52) or folded as on slide 53, the attention layers would see DP = 256 / (4 × 4 × 1) = 16, and 2 is what remains for the expert layers only if their tensor parallelism is also 4: 256 / (4 × 8 × 4) = 2.

::slide 70 | Nemotron 3 Super's long-context phase: 64-way context parallelism, 2-way tensor parallelism and 64-way expert parallelism on GB200s, at 1M-token context; "TP / PP / CP / EP (2/0/64/64)"

**Nemotron 3 Super** follows the DeepSeek-V3 model: a lot of expert parallelism, and in its long-context extension phase 64-way context parallelism. (The paper's paragraph names no pipeline degree; the slide writes 0, the overview table "??".)

::slide 71 | Megatron's configurations for Qwen3: 30B-A3B on 8 GPUs (TP 1, PP 1, EP 8); 235B-A22B on 512 GPUs (TP 2, PP 8, EP 32) for pretraining and finetuning; a second table of benchmarked shapes for DSV3, Qwen3 and Qwen3-Next on H100, B200 and GB200

**Qwen 3** follows the DeepSeek recipe too: EP 32, PP 8 and TP 2 for the attention matrices. A check against slide 52's rule: on 512 GPUs, attention's DP is $512/(2 \times 8) = 32$, exactly the EP degree. The smaller 30B-A3B model fits on one 8-GPU node with EP 8 alone. The second table, which the professor does not go into, shows that even the choice among tensor-parallel layouts noticeably changes performance.

::note skip 1:18:32 | NVIDIA's Megatron Bridge benchmarking of different configurations is shown but not explained.

::slide 72 | overview table of DP, TP/SP, EP, PP and CP for DeepSeek, DeepSeek-V3, Yi, Llama 3 405B, Gemma 2, Mixtral 8x22B, Nemotron 3 120B (long context) and Qwen 3, with "??" where unpublished; "Patterns – TP generally <= 8. EP can be big (but hard!). Long context phases use large CP"

The patterns:
- every model uses as much **data parallelism** as it can;
- **TP is at most 8**, the size of the fast domain on GPUs (Gemma's 8-way model sharding on TPUs happens to match);
- **EP can be big**, 32 or 64, partly because DeepSeek-V3 and its infrastructure showed how;
- **long-context phases use large CP**.

::note slip 1:19:04 | Aloud, TP "almost always remains below eight"; on the slide several rows are exactly 8, so read it as at most 8.
::note why 1:18:49 | The table's first "Deepseek" row (TP 1, EP 8, PP 16, ZeRO-1) does not match slide 64's dense DeepSeek LLM, which used tensor parallelism. It matches DeepSeek-V2, which reported 8-way expert and 16-way pipeline parallelism with ZeRO-1.
::kp recent-lm-configs

::slide 73 | recap: scaling beyond a point needs multi-GPU, multi-node parallelism; there is no single solution (you probably want all three approaches); simple, interpretable rules of thumb combine them

Three sentences to keep. Past a certain scale you need many GPUs on many nodes, perhaps many data centers. No single strategy solves it: you have fast links and slow links, and techniques that spend batch size, memory or bandwidth, and you want to use all of them. And yet the rules for combining them are simple: model-parallel inside the fast domain until the model fits, pipelines (or FSDP) across the slow links, data parallelism for everything else. Done that way, the professor concludes, you get effectively full utilization of the hardware. The next lecture is about scaling laws.
