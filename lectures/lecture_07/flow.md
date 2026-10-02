---
title: L7 · Parallelism, read through
minutes: 40
---
This lecture takes training from one GPU to many. It first builds the vocabulary of distributed programming: the collective operations, the hardware that carries them, and how to call and time them in PyTorch. It then uses that vocabulary to write data, tensor and pipeline parallelism from scratch on a deep MLP. After it you can say what each strategy splits, which collective it needs, how many bytes that collective moves, and which link in the machine it should run on.

## Why train on more than one GPU? {#goal}
source: lecture_07.py:L19-L57 · video 0:05-5:50

The last two lectures ([L5 on GPUs](#/read/lecture_05) and [L6 on kernels](#/read/lecture_06)) looked inside one GPU. Their lesson was that the arithmetic units are far from the data: a kernel spends most of its time waiting for bytes from high-bandwidth memory (HBM), so you fuse operations and tile them to read each byte as few times as possible. This lecture zooms out. A training job may use four GPUs or a hundred thousand, and the piece of data a GPU needs next may sit in another GPU's memory, or in another building.

::figure official/lectures/images/gpu-node-overview.png | one box is last week's GPU; this week the boxes multiply and the wires between them become the bottleneck

The professor's framing is that nothing changes in principle. Compute is far from data at every scale, and the game is the same: "orchestrate the computation to try to avoid data transfer bottlenecks". As he put it, "it's very easy to use a ton of GPUs, but it's hard to use them effectively."

The lecture lays the levels out as one generalized hierarchy, fastest first:
- **single GPU, on chip:** L1 cache and shared memory, next to the compute units;
- **single GPU, off chip:** HBM;
- **one node, several GPUs:** NVLink and NVSwitch;
- **several nodes:** Infiniband or Ethernet.

HBM, which L5 "lamented was so slow", now counts as the *fast* level. Every level down is slower, and the remedy changes name with the level. Last week it was fusion and tiling (read into shared memory, do as much as you can, write back). This week it is **replication and sharding**: decide which tensors each GPU keeps a full copy of and which it keeps only a piece of, so that as little as possible crosses the slow wires.

::code lecture_07.py:L25-L35 | the same enemy at every level: compute is far from data

### Two reasons to go multi-GPU

The lecture gives exactly two.
1. **It does not fit.** The parameters, gradients, optimizer state and activations exceed one GPU's HBM. A B200 has 192 GB. With [L2's accounting](#/read/lecture_02) of 12 bytes per parameter for AdamW training, a 1-trillion-parameter model needs about 12 TB of state before any activations, the HBM of about 63 B200s. Even its bf16 weights alone, 2 TB, need 11.
2. **It is too slow.** Even when the model fits, more GPUs mean more FLOP/s. L2's training time, $6ND$ divided by the cluster's throughput, falls linearly with the GPU count, as long as the GPUs are not left waiting on each other.

::code lecture_07.py:L37-L39 | the two reasons: memory capacity and speed

The second reason hides a trade-off the professor named aloud. If a model fits on fewer GPUs, spreading it over more buys FLOPs but costs communication bandwidth, and "that's some calculation you're going to have to do" before choosing how to parallelize. The byte counts in this lecture are that calculation.

### How the lecture is organized, and how it runs

Part 1 builds the blocks: the collective operations (the programming model), the hardware that connects GPUs, PyTorch's `torch.distributed`, and a benchmark of real bandwidth. Part 2 uses them to train a deep MLP three ways, each cutting the model along a different axis: **data parallelism** cuts the batch, **tensor parallelism** the width, **pipeline parallelism** the depth.

::code lecture_07.py:L55-L57 | three strategies, three axes of the same MLP

The lecture is an executable Python file, but distributed code runs as several processes at once, and the lecture's tracer can only step through one. So when traced, every distributed call is replaced by a no-op and the function runs once as rank 0. When run directly, it really spawns the processes, and the output of one such run (four GPUs on a cloud machine) is linked from the file. This read-through shows that recorded output where it matters.

::note skip 5:34 | Everything is shown on MLPs, not a full Transformer. MLPs are where a Transformer's compute goes, so the core is the same; real models "just require a lot more bookkeeping" (1:15:27).
::note deferred 5:13 | Implementing these pieces in torch.distributed is Assignment 2.

## What is a collective operation? The four warm-ups {#collectives}
source: lecture_07.py:L75-L137, L203-L206 · video 5:50-12:32, 19:44-21:53

How do you tell eight GPUs what to send to whom? You could write every point-to-point transfer yourself: GPU 3 sends this slice to GPU 5, GPU 5 adds it and passes it on. Instead, distributed programming uses **collective operations**: you name one communication pattern over all devices at once, and the library works out the individual transfers. These primitives go back to parallel computing in the 1980s; as the professor put it, "it wasn't invented for LLM training." Naming the whole pattern is easier to write and lets the system do the scheduling, which is often faster than hand-managed sends.

Two terms set up every example:
- a **rank** is one device (here, one GPU; "for this class, the rank is the GPU"), numbered 0, 1, 2, 3;
- the **world size** is the number of devices, here 4.

::figure official/lectures/images/ranks.png | four ranks, 0 to 3; world size 4

The lecture sorts eight collectives into three groups:
- **foundations:** broadcast, scatter, gather, reduce. The professor calls them "really just warm ups"; they are stepping stones and rarely show up in training itself;
- **workhorses:** all-gather, reduce-scatter, all-reduce, which "show up again and again" in distributed training;
- **all-to-all**, the most general, needed for mixture-of-experts models.

Each is written in the code as literal tensors: what each rank holds before, what it holds after. Nothing runs yet.

### Broadcast and scatter: one rank sends

**Broadcast** copies rank 0's tensor to every rank. Rank 0 holds [0, 1, 2, 3]; afterwards all four ranks hold [0, 1, 2, 3]. Its training use is minor and happens once: rank 0 loads the initial checkpoint and broadcasts it.

**Scatter** splits rank 0's tensor into world-size pieces and gives piece $i$ to rank $i$: rank 0 ends with [0], rank 1 with [1], and so on. Each GPU can then work on its own part. Scatter is the stepping stone to reduce-scatter.

::note spoken 12:06 | Asked whether this is NumPy broadcasting: only in spirit ("one thing that goes to many things"); the collective is a communication pattern between devices.

### Gather and reduce: everything comes to one rank

**Gather** is the inverse of scatter. Each rank holds one piece; rank 0 receives them all and **concatenates** them: [0], [1], [2], [3] become [0, 1, 2, 3] on rank 0.

**Reduce** starts from the same pieces but **combines** them with an operation instead of concatenating. With sum, [0], [1], [2], [3] become [6] on rank 0. The operation can be any associative and commutative one: sum, min, max. The professor tied the two together: you can think of gather as "a reduction where the operation is concatenation".

The difference shows in the shapes. Gather's output is world size times as long as one input; reduce's output has the input's length.

::predict collective-taxonomy
::code lecture_07.py:L115-L135 | gather ends with [0, 1, 2, 3] on rank 0; reduce ends with [6]
::widget fixture:lecture_07--collective-grid | pick gather, then reduce, on the same inputs: gather's row on rank 0 is W cells long, reduce's keeps the input length, and each reduced cell outlines one input cell from every rank

The destination rank is not fixed in advance. Rank 0 is the example; in code you pass the destination as an argument when you make the call.

### How to remember the names

The lecture closes the vocabulary with three rules that decode every name:
- **reduce** means applying an associative operation (sum, min, max) across ranks;
- **scatter** is the inverse of **gather**: scatter distributes, gather centralizes;
- **all-** means the destination is every rank, not just one.

So all-gather is a gather whose result lands everywhere, and all-reduce is a reduce whose result lands everywhere. Reduce-scatter combines two rules: reduce, then scatter the result.

::code lecture_07.py:L203-L206 | the three rules that name every collective
::note slip 10:13 | Introducing gather, the professor said "the inverse of scatter is scatter"; he meant gather, as the code (L205) and his own mnemonic at 20:07 say.
::kp collective-taxonomy

## The workhorses: all-gather, reduce-scatter, all-reduce, all-to-all {#workhorses}
source: lecture_07.py:L139-L201 · video 12:32-19:44

Three collectives carry almost all the communication of distributed training. The pattern to expect, in the professor's words, is "gather to do something and then scatter and then gather and scatter again."

### All-gather: every rank gets every piece

**All-gather** is gather with the result delivered to every rank. Rank $r$ holds the one-element tensor [$r$]; afterwards every rank holds [0, 1, 2, 3]. Nothing is summed. With world size $W$ and pieces of length $n$, each rank ends with the same tensor of length $W \cdot n$, in rank order.

::code lecture_07.py:L139-L152 | four one-element shards become [0, 1, 2, 3] on every rank

Its training use, foreshadowed here: each rank holds only a **shard** (a slice) of the parameters, and before the forward pass it all-gathers them to get the full parameters. Of the gathered tensor, only $1/W$ was already local; the other $(W-1)/W$ had to cross the wires. If 8 ranks each hold 1M bf16 parameters (2 MB), after the all-gather each holds 16 MB, of which 14 MB arrived from other ranks.

::note deferred 13:16 | "It's not important that you understand this statement precisely": the parameter-shard use returns with ZeRO and FSDP in the next lecture.
::kp all-gather

### Reduce-scatter: sum each slot, deliver slot $i$ to rank $i$

Now every rank holds a full-length vector. Rank $r$ holds $[r, r+1, r+2, r+3]$, so the four rows are [0, 1, 2, 3], [1, 2, 3, 4], [2, 3, 4, 5] and [3, 4, 5, 6]. **Reduce-scatter** reduces each slot across the ranks and gives the result for slot $i$ only to rank $i$:
- slot 0: $0 + 1 + 2 + 3 = 6$, to rank 0;
- slot 1: $1 + 2 + 3 + 4 = 10$, to rank 1;
- slot 2: $14$, to rank 2;
- slot 3: $3 + 4 + 5 + 6 = 18$, to rank 3.

::predict reduce-scatter
::code lecture_07.py:L154-L167 | each rank keeps one slot's sum: 6, 10, 14, 18

In general, $W$ inputs of length $L$ become one output of length $L/W$ per rank. The use case: after the backward pass each rank has gradients computed from its own data, and they must be summed. Reduce-scatter sums them but **distributes the storage**: no rank holds the whole summed gradient, only its share.

::widget fixture:lecture_07--collective-grid | switch to reduce-scatter and click rank 3's output: it outlines slot 3 on every input row, one value from each rank, summed
::kp reduce-scatter

### All-reduce = reduce-scatter + all-gather

**All-reduce** leaves every rank with the full elementwise reduction. On the same four inputs, every rank ends with [6, 10, 14, 18]. It is the simplest to describe: reduce, then replicate everywhere. The lecture states the identity that matters for the rest of the course:

$$ \text{all-reduce} = \text{reduce-scatter} \;\text{then}\; \text{all-gather} $$

After the reduce-scatter, ranks 0 to 3 hold 6, 10, 14 and 18. An all-gather of those four numbers puts [6, 10, 14, 18] on every rank, which is exactly the all-reduce.

::code lecture_07.py:L169-L183 | the same inputs, every rank ends with [6, 10, 14, 18]
::animation fixture:lecture_07--ring-allreduce | after the first W − 1 ring steps each rank owns one fully summed slot (the reduce-scatter result 6, 10, 14, 18); the next W − 1 steps only copy those slots around

The animation shows one way a library actually runs an all-reduce, the **ring**: each rank passes one slot to its neighbor per step and adds what arrives. The lecture itself does not describe the ring (the library chooses ring or tree for you), but it makes the identity visible. The first half is literally a reduce-scatter, the second literally an all-gather.

All-reduce's use case is the same as reduce-scatter's, with one word changed: sum the gradients from the different data shards, but **replicate** the result on every rank. Data parallelism, later in this lecture, does exactly that. Why bother splitting it? Because between the two halves, each rank holds only its own summed slot. Systems that keep state sharded (ZeRO and FSDP) stop there, do work on their shard, and run the all-gather later. In the professor's words, splitting it means "you can intervene and you can manage things a bit more."

::video 16:22-16:41 | why all-reduce is worth splitting into two halves
::kp all-reduce-decomposition

### All-to-all: the transpose

In **all-to-all**, each rank sends a different piece to each other rank. Position $j$ in rank $i$'s tensor is addressed to rank $j$. With rank $r$ holding $[4r, 4r+1, 4r+2, 4r+3]$:
- rank 0 holds [0, 1, 2, 3] and sends 0 to rank 0 (keeps it), 1 to rank 1, 2 to rank 2, 3 to rank 3;
- rank 1 holds [4, 5, 6, 7] and sends 4 to rank 0, 5 to rank 1, and so on.

Rank 0 then collects column 0 of everyone, [0, 4, 8, 12], and rank 1 collects column 1, [1, 5, 9, 13]. Laid out as a rank-by-chunk matrix, with balanced pieces, all-to-all is a **transpose**.

::code lecture_07.py:L185-L201 | the 4 × 4 layout transposed; the MoE use case; unbalanced splits allowed

Its use is mixture-of-experts (MoE) models (see [L4](#/t/lecture_04)). Each rank holds a split of the tokens and a subset of the experts. Which expert a token goes to is decided by the router after looking at the token, "the key idea of the MOE is that it's dynamic routing", so no fixed pattern like all-gather fits. Each rank must send each other rank exactly the tokens routed to the experts there. All-to-all also handles unbalanced splits, any number of bytes to any rank, but balanced is the goal; that is what MoE load balancing is for.

::widget fixture:lecture_07--collective-grid | pick all-to-all: the output grid is the input grid transposed, and rank i's slot j outlines rank j's slot i
::note skip 8:05 | All-to-all is "important for MOEs, but we're not going to actually spend too much time on this": it gets this static example only, is never run, and no bytes are counted. Unbalanced splits are mentioned and not shown (19:13).
::kp all-to-all

## How are GPUs actually connected? {#hardware}
source: lecture_07.py:L209-L236 · video 21:53-36:13

The collectives are patterns; the hardware decides how fast each one runs. The lecture describes two worlds.

### At home: PCIe and Ethernet

The classic computer has CPUs on a PCIe bus (the bus that once carried your mouse and keyboard), with GPUs and RAM hanging off it. Two GPUs in the same box talk over PCIe; the lecture quotes about 242 GB/s for PCIe v7.0 with 16 lanes. Two GPUs in different boxes talk over Ethernet, about 200 MB/s on a home network. This is what you would get, the professor said, if you hooked your gaming GPU up with a friend's to train a big model.

### In the data center: NVLink, NVSwitch, Infiniband

::figure official/lectures/images/gpu-node-overview.png | eight GPUs per node on NVLink to an NVSwitch; nodes joined into pods by Infiniband

A typical training cluster is built in layers:
- **node:** 8 GPUs, each connected by NVIDIA's NVLink to an NVSwitch. The switch routes traffic, so from a programming view any GPU can talk directly to any other in the node. On B200s, NVLink 5.0 gives 1.8 TB/s, against 8 TB/s for the B200's HBM;
- **pod:** many nodes (the figure's 256 is "kind of made up"), connected by Infiniband, about 0.05 TB/s. The path now leaves the GPU through PCIe, a network card (HCA, the Infiniband NIC) and an Infiniband cable;
- **cluster or data center:** several pods, connected by Ethernet, which goes through PCIe and the CPU.

Why not one big switch? Because "you can't have an NVSwitch handling 100,000 GPUs." Like the memory hierarchy inside a GPU, the bigger the domain, the slower the link.

The numbers turn into ratios. The professor used them once aloud: NVLink is "about 4x slower" than HBM. Infiniband is slower again by much more.

::predict interconnect-hierarchy
::worked interconnect-hierarchy
::widget fixture:lecture_07--link-ladder | type any payload: each hop down multiplies the time by the same factor, about 4.4 from HBM to NVLink and 36 from NVLink to Infiniband, whatever the size

A 16 GB gradient tensor crosses NVLink 5.0 in about 9 ms ($16 / 1800$ s) and Infiniband in 320 ms ($16 / 50$ s): same bytes, 36 times the time. The rest of the lecture keeps returning to this gap. The strategy that communicates in every layer belongs inside the 8-GPU NVLink domain; the one that communicates rarely can cross the slow links.

::note spoken 35:00 | Asked what happens with nine GPUs: if eight share a node and the ninth sits on another without NVLink, that is "really bad": one GPU's worth of extra compute, very expensive to reach.

### Bypassing the CPU: RDMA

Over standard Ethernet, data from a GPU goes through the CPU: it is copied into the operating system kernel's socket buffer ("kernel" here is the CPU's operating-system kernel, not a GPU kernel), wrapped into TCP packets, copied to the network card's ring buffer, and shipped. Every copy adds latency.

**Remote direct memory access (RDMA)** lets one GPU read or write another GPU's memory without involving the CPU at all. NVLink and NVSwitch provide it; Infiniband supports it; standard Ethernet does not. The professor separated the two kinds of names in answer to a question: RDMA is "more of a desiderata", the capability, while NVLink, NVSwitch and Infiniband are the cables and switches that provide it.

::code lecture_07.py:L223-L230 | Ethernet goes through the CPU; RDMA skips it; Infiniband has it, RoCE adds it to Ethernet

### Two recent advances

- **GB200/GB300 NVL72** puts 72 GPUs into a single NVLink domain: trays of GPUs stacked in a rack, all on one NVSwitch fabric. Normally, "if you're mortal", the fast domain is 8 GPUs; "if you have a lot of money", it is 72.
- **RoCE** (RDMA over Converged Ethernet) brings RDMA to Ethernet. It is cheaper but weaker than Infiniband, which "generally is very expensive, as is a lot of NVIDIA products". Meta has used it.

::note aside 31:41 | The lecture counts NVL72 as 8 GPUs per tray × 9 trays, and aloud describes a tray as two CPUs with four GPUs each. NVIDIA's published GB200 NVL72 layout is 18 compute trays of 4 GPUs (two Grace CPUs, each paired with two GPUs). The total, 72 GPUs in one NVLink domain, is the same either way.

### NCCL: collectives as GPU kernels

At the lowest software level sits the **NVIDIA Collective Communication Library (NCCL**, pronounced "nickel"). You ask it for an all-reduce; it does three things:
- detects the topology of the hardware (how many nodes, which switches, NVLink or PCIe);
- optimizes the path between GPUs;
- launches GPU kernels that send and receive the data. Communication is a kernel too, because "everything that runs on a GPU is a kernel."

::code lecture_07.py:L232-L236 | NCCL turns a collective into packets: topology, path, kernels
::note skip 30:56 | NCCL's internals are skipped: "just know that it exists". Asked whether NCCL is optimized for multi-node clusters: "I don't know the details", though NVIDIA optimizes its whole stack for exactly these customers (33:59).
::note skip 36:03 | A question on how TPU interconnects differ was taken offline.
::kp interconnect-hierarchy
::kp rdma-nccl

## How do you call collectives from PyTorch? {#torch-distributed}
source: lecture_07.py:L239-L284, L540-L591 · video 36:13-46:38

You do not program NCCL directly. PyTorch's `torch.distributed` library gives a clean interface to the collectives, for example `all_gather_into_tensor`, and supports several **backends**: `nccl` for GPUs and `gloo` for CPUs. Parallel processing predates GPUs, so the same collectives run on CPUs too. The library also offers higher-level wrappers such as `FullyShardedDataParallel`, which this course does not use because it builds things from scratch.

### One process per rank, the same code on every rank

A distributed program is one function launched `world_size` times, once per rank, each in its own process. The lecture's `spawn` helper calls PyTorch's `mp.spawn` to start them; each process receives its `rank` and the `world_size`.

::code lecture_07.py:L577-L591 | spawn starts world_size processes; when traced it runs rank 0 alone with distributed calls disabled

Each process first calls `setup`, which joins the group with `init_process_group`. The `MASTER_ADDR` and `MASTER_PORT` it sets are only a rendezvous: rank 0's address, used to coordinate metadata. They are not how the GPUs talk. "The actual data goes through NCCL. Otherwise, it would be very, very slow." On a GPU machine the backend is `nccl`; on the professor's laptop it is `gloo`.

::code lecture_07.py:L540-L549 | the master address is for coordination only; nccl if CUDA is available, else gloo

Every rank then runs the same code and must call the same collectives in the same order; a collective completes only when every rank has joined it. The processes otherwise run asynchronously: one might finish before another starts, and their prints interleave in whatever order the hardware feels like. `dist.barrier()` makes every process wait until all have reached that line. The lecture uses more barriers than usual so the printed output stays readable, but the professor named the cost: "you end up kind of waiting potentially unnecessarily."

::predict torch-distributed-setup

### The three workhorses, run for real

`collective_operations_main` runs on four ranks the examples from the start of the lecture.

**All-reduce.** Rank $r$ creates `[0, 1, 2, 3] + r` and calls `dist.all_reduce(tensor=data, op=SUM, async_op=False)`. The call writes its result **in place** into `data`.

::code lecture_07.py:L253-L260 | each rank's input is [0, 1, 2, 3] + rank; all_reduce overwrites it
::code lecture_07_stdout.txt:L28-L31 | in the recorded four-GPU run every rank prints [6, 10, 14, 18]

**Reduce-scatter.** Here the call takes a separate output: `reduce_scatter_tensor(output, input)` leaves the input untouched and writes rank $r$'s one summed slot into a freshly allocated one-element `output`. Whatever happens to be in the allocated buffer beforehand does not matter.

::code lecture_07.py:L262-L270 | input of length world_size, output of length 1
::code lecture_07_stdout.txt:L36-L39 | inputs unchanged; outputs 6, 10, 18 and 14, printed in arrival order

**All-gather.** The reduce-scatter's output becomes the all-gather's input, and the output is allocated with length `world_size`. Before the call the buffer holds leftovers ("don't worry about it"); after it, every rank holds [6, 10, 14, 18].

::code lecture_07.py:L272-L282 | the reduce-scatter output is gathered back: all-reduce = reduce-scatter + all-gather
::code lecture_07_stdout.txt:L44-L47 | every rank ends with [6, 10, 14, 18], the all-reduce result

The professor called it "proof via example that all reduce is equal to reduce scatter plus all gather." Then `cleanup()` destroys the process group, which is good practice.

### Asynchronous collectives

With `async_op=True` the call returns immediately and the communication proceeds in the background. CUDA kernels are already asynchronous with respect to the Python process; now the processes are asynchronous with respect to each other too. The point is **overlap**: launch the all-reduce, load the next batch (work that does not depend on it), and wait only when you need the result.

::note skip 44:29 | Overlapping communication with computation is "a typical thing, which I'm not going to talk about this class"; it returns as the first item of the lecture's "what's missing" list.
::note aside 38:54 | The traced lecture ran on a laptop with gloo; the linked stdout is the real four-GPU NCCL run.
::kp torch-distributed-setup

## How fast is a collective? Timing and byte accounting {#benchmark}
source: lecture_07.py:L287-L372 · video 46:38-55:07

Part 1 ends by measuring real communication speed: an all-reduce, then a reduce-scatter, each on $100 \times 1024^2$ (about 105 million) fp32 elements per rank, 400 MiB, over four GPUs.

### Timing: warm up, synchronize, barrier

The timing follows [L2's benchmarking rule](#/read/lecture_02) with one addition. There are now "two forms of asynchrony": CUDA kernels run asynchronously from the Python process, and the processes run asynchronously from each other. So:
- **warm up:** run the all-reduce once untimed, so one-time setup costs (creating communicators, opening connections) are not counted;
- **`torch.cuda.synchronize()`:** wait until this rank's GPU has finished its queued kernels;
- **`dist.barrier()`:** wait until every rank has reached this point.

Both waits come before starting the clock and again before stopping it.

::code lecture_07.py:L301-L320 | warmup, then synchronize + barrier on both sides of the timed call

::predict benchmark-method

Remove the closing waits and the measured time drops: `all_reduce` returns once the work is queued on the GPU, so the clock stops before the transfer is done. Remove the warmup instead and the measured time rises, because the timed call now pays the first-call setup.

Why synchronize *before* barrier? A student asked, and the professor first said "I'm not sure", then reasoned it out. If a rank reaches the barrier while its kernels are still running, the barrier only lines up the Python processes, and "the barrier doesn't really do anything" for the GPU work. Each rank must first wait for its own GPU, then all ranks wait for each other.

::video 54:39-55:00 | why a barrier before synchronize would not synchronize anything

Each rank is a separate process and reports its own time; in the recorded run the four times differ a little. "If you want to report one number, you can take the average."

::note skip 46:44 | Benchmarking is kept short ("since I want to actually move on to part two"): one timed call per collective, no repetitions or statistics.
::kp benchmark-method

### Effective bandwidth: the MFU of communication

A time on its own does not say whether it is good. The professor's analogy is MFU from L2: compare what you achieved with what the hardware could do. For communication that means **effective bandwidth**: the bytes that had to be sent, divided by the time.

For an all-reduce of a tensor of $S$ bytes per rank on $W$ ranks, the lecture counts:
- $W - 1$ steps, because combining $W$ values takes $W - 1$ additions;
- a factor 2, for sending plus receiving (in the ring picture, the reduce-scatter half and the all-gather half);
- so $\text{sent\_bytes} = 2 \cdot S \cdot (W - 1)$ in total;
- and $\text{total\_duration} = W \times \text{duration}$, "the total amount that all the ranks have waited".

$$ \text{bandwidth} = \frac{2\,S\,(W-1)}{W \cdot \text{duration}} = \frac{W-1}{W}\cdot\frac{2S}{\text{duration}} $$

::code lecture_07.py:L322-L333 | sent_bytes = size × 2 × (W − 1); total duration = W × duration
::video 49:29-50:21 | where the W − 1, the 2 and the W in the denominator come from

As $W$ grows, $(W-1)/W$ tends to 1, so effective bandwidth is about $2S/\text{duration}$, **independent of the world size**, "which is good". The bytes each rank moves are $2S(W-1)/W$, which saturates at $2S$: an all-reduce over 1000 GPUs moves barely more per GPU than one over 8. It is also independent of whether NCCL passes the messages around a ring or down a tree. At the lecture's $W = 4$, the factor is 0.75.

::predict collective-bytes

For the lecture's tensor, $S = 400$ MiB: each rank sends and receives $2 \times 400 \times 3/4 = 600$ MiB; at $W = 8$ it would be $2 \times 400 \times 7/8 = 700$ MiB, a factor 1.17, not 2.

::widget fixture:collective-bytes | slide the world size from 4 to 64: per-rank bytes creep toward 2·S and stop; switch the payload to bf16 and every bar halves

### Reduce-scatter: no factor 2

The reduce-scatter benchmark gives each rank an input of $W \times$ num_elements and an output of num_elements. Its accounting has no factor 2: each rank only sends away its contributions, and nothing comes back, because the all-gather half is missing.

$$ \text{sent\_bytes} = S_\text{in}\,(W-1) $$

::code lecture_07.py:L360-L370 | data_bytes × (W − 1), "no 2x here"; the closing note compares it with all-reduce

The lecture's closing note: all-reduce is reduce-scatter plus all-gather, so it moves twice the data in twice the time, and "the two cancel out, so you get the same kind of bandwidth." The recorded run agrees: the two bandwidth readouts are similar.

::note aside | Read the "2x the data" per unit of output. The two benchmarks are not the same size: the reduce-scatter's input is $W$ times the all-reduce's tensor (L342 against L305), so by the lecture's own formulas it sent $4S \times 3 = 12S$ bytes against the all-reduce's $2S \times 3 = 6S$, twice as many. That is why it took longer in the recorded run while reporting a similar bandwidth.
::note slip 49:51 | Aloud the factor 2 is "because you need to both send and reduce"; the code's comment says send + receive.
::note aside | The run's millisecond times and GB/s readouts are one run on unspecified GPUs, so they are not restated here; the professor's point is the method and the ratio.
::kp collective-bytes

## Data parallelism: split the batch, all-reduce the gradients {#ddp}
source: lecture_07.py:L51-L53, L375-L436 · video 55:07-1:02:32

Part 2 trains a deep MLP: a stack of `num_dim × num_dim` weight matrices, each followed by GELU. MLPs are where a Transformer spends its compute, so this is representative. The sample data is a batch of 128 examples of dimension 1024. The three strategies cut this one computation three ways.

::figure official/lectures/images/data-parallelism.png | a schematic: the data is cut by rows, every GPU keeps all of the model

### The algorithm

**Data parallelism** gives each rank a slice of the data and a full copy of everything else.
- **Shard the batch.** The local batch size is $128 / 4 = 32$; rank $r$ takes rows $32r$ to $32r + 31$. (In practice each rank should load its own data rather than slicing one batch; the slicing is for illustration.)
- **Replicate the model.** Every rank creates all four 1024 × 1024 parameter matrices and its own AdamW optimizer.
- **Train as usual on the local batch:** forward, loss, `loss.backward()`.
- **Synchronize the gradients.** For every parameter, `dist.all_reduce(param.grad, op=AVG)`. After this, every rank holds the same gradient: the average over all 128 examples.
- **Step.** `optimizer.step()` runs on every rank.

::code lecture_07.py:L405-L414 | each rank takes its 32 rows and builds all the parameters and its own optimizer
::code lecture_07.py:L416-L432 | ordinary training, plus one all-reduce of every gradient between backward and step

The code's comment calls that all-reduce the "ONLY difference between standard training and DDP" (distributed data parallel). The professor found it elegant: a one-line change, after which "each rank is basically performing parameter updates as if it had all the data on it", while only processing a quarter of it.

### Why the copies never drift apart

What differs across ranks after a step, and what is the same? Guess before reading on.

::predict data-parallelism-ddp

The **losses differ**, since each is computed on a different 32-example slice. The **gradients** start different and are averaged to be identical. The **parameters** then stay identical: they started identical (every rank seeds its initialization the same way; in practice rank 0 would broadcast a checkpoint), and identical optimizers applying identical gradients to identical parameters produce identical results. No periodic resynchronization is needed.

::code lecture_07_stdout.txt:L64-L67 | four ranks, four different losses, the same parameter summaries

### What it costs

Communication per step is one all-reduce of the full gradient. With the byte accounting from the benchmark, the lecture's toy has 4 fp32 matrices of $1024^2$ elements, 16 MiB of gradients, so each rank sends and receives $2 \times 16 \times 3/4 = 24$ MiB per step. For a 7B-parameter model with bf16 gradients (14 GB) on 8 ranks it is $2 \times 14 \times 7/8 = 24.5$ GB per rank per step.

Memory per rank does **not** shrink, except for activations, which follow the local batch. Every rank holds full parameters, gradients and optimizer state. The professor said so directly: all-reduce "does require holding all the models, parameters in memory."

And the optimizer step is done redundantly on every rank. That is deliberate. In the closing summary the professor explained the trade: every rank repeats the same update "but the reason you're doing that is that you don't have to move the optimizer state across."

::video 1:20:12-1:20:33 | why every rank repeats the optimizer step instead of sharing it
::note spoken 1:00:57 | The batch must be at least the world size, and preferably a multiple of it; otherwise pad it, "but it's just easier for everyone if it is."
::note spoken 1:01:31 | For a Transformer, DDP "would actually be basically the same": it is modular and never touches the forward pass.
::note slip 1:01:55 | "DDP just averages the parameters here": it averages the gradients (L429), as his own summary a few seconds later says.
::note deferred 1:14:51 | A real implementation starts sending each gradient as soon as the backward pass produces it, overlapping the all-reduce with the rest of backward. That is explored in Assignment 2.
::kp data-parallelism-ddp

## What if the model does not fit? A preview of ZeRO {#zero}
source: lecture_07.py:L68, L183, L387 · video 1:02:32-1:02:59

DDP's simplicity has a price: every rank holds everything. "But what if the model parameters don't fit in memory? Then you're going to have to be more clever, and that's the topic for the next class." The lecture names the answer three times without working it out: **ZeRO** and **FSDP** (fully sharded data parallel), which replace DDP's all-reduce with its two halves. The numbers below come from the ZeRO paper (Rajbhandari et al., 2019), not from this lecture; the next lecture covers them.

::code lecture_07.py:L387 | the one-line pointer: all-gather and reduce-scatter instead of holding all parameters

The idea follows from the decomposition. After the reduce-scatter half, each rank owns the summed gradient for one shard of the parameters. If it also owns that shard's optimizer state, it can update that shard alone, and all-gather the updated parameters afterwards. Nobody needed the whole optimizer state.

The paper's accounting is for mixed-precision Adam with $\Psi$ parameters on $N$ devices: 2 bytes per parameter for bf16/fp16 weights, 2 for gradients, and $K = 12$ for the fp32 state (a master copy of the weights plus Adam's two moments), 16 bytes in total. ZeRO shards these in three stages:

$$ \underbrace{16\Psi}_{\text{DDP}} \;\to\; \underbrace{4\Psi + \tfrac{12\Psi}{N}}_{\text{stage 1: optimizer state}} \;\to\; \underbrace{2\Psi + \tfrac{14\Psi}{N}}_{\text{stage 2: + gradients}} \;\to\; \underbrace{\tfrac{16\Psi}{N}}_{\text{stage 3: + parameters}} $$

::predict zero-sharding
::worked zero-sharding
::widget fixture:zero-memory | raise N: stages 1 and 2 level off at 4Ψ and 2Ψ, and only stage 3, with no N-independent term, keeps shrinking

Stage 3 shards even the parameters, so a rank cannot run a layer until it all-gathers that layer's weights, just before use, and frees them afterwards. This is the use case the lecture gave for all-gather at the start: "each rank holds parameter shard, gather to get full parameters for forward pass."

::note aside | The paper's 16 bytes include an fp32 master copy of the weights. [L2's napkin count](#/read/lecture_02) of 12 bytes per parameter (2 + 2 + 4 + 4) has none; the two are different recipes, so do not mix their totals.
::note deferred 1:02:32 | FSDP and ZeRO are the next lecture's topic; see [L8](#/t/lecture_08).
::kp zero-sharding

## Tensor parallelism: split every layer's width {#tensor}
source: lecture_07.py:L439-L481 · video 1:02:59-1:09:38

Data parallelism leaves the model whole. **Tensor parallelism** cuts the model itself: each rank gets part of *each* layer, and every rank sees the whole batch. The cost, the professor warned at once, is that "we're going to have to transfer a lot more data."

::figure official/lectures/images/tensor-parallelism.png | the cut now runs through every layer, not through the data

### The algorithm

With `num_dim = 1024` and 4 ranks, `local_num_dim = 1024 / 4 = 256`. Each rank holds, for every layer, a `1024 × 256` block of columns of the weight matrix: a quarter of the parameters. This is **column tensor parallelism**.

::code lecture_07.py:L450-L459 | every rank gets all 128 × 1024 of the data, and a 1024 × 256 column block of each layer

Why does that work? A matrix multiply splits by output columns: column block $r$ of $x W$ is $x$ times column block $r$ of $W$. So rank $r$ computes `x @ params[layer]`, a `128 × 256` slice of the layer's output, with no help from anyone. GELU is elementwise, so it applies to the slice directly. But the next layer needs the *full* `128 × 1024` input. So every layer ends with an all-gather:

::code lecture_07.py:L463-L475 | matmul on this rank's columns, GELU, all-gather the four slices, concatenate along dim 1

`dist.all_gather` fills a list of four `128 × 256` buffers, one per rank, and `torch.cat(..., dim=1)` lays them side by side into `128 × 1024`. Every rank now holds the full activation and moves on to the next layer.

::code lecture_07_stdout.txt:L68-L71 | after four layers, all four ranks hold the same 128 × 1024 activation

The professor contrasted it with data parallelism. DDP is "very elegant, because it's splitting by data. The model is treated as a module." Tensor parallelism has to "muck around with the model", because it rests on splitting one matrix multiplication into smaller ones.

::note skip 1:04:38 | Splitting by rows is possible too, "but we're not going to talk about that right now". Megatron-LM pairs a column split with a row split, so that an MLP block needs one collective per direction instead of one per layer; the lecture's toy gathers after every layer.

### What it costs

How much does a rank receive per layer? It already has its own slice and gets the other three. In fp32, one slice is $128 \times 256 \times 4$ B $= 128$ KiB, so each rank receives $3 \times 128 = 384$ KiB per layer.

::predict tensor-parallelism

Two things follow. The traffic scales with the **activation**, batch × num_dim, not with the parameters: the weight blocks never move. And it is paid **at every layer**. For 4 layers that is $4 \times 384$ KiB $= 1.5$ MiB per rank per forward pass; a real model multiplies it by dozens of layers, larger batches and long sequences, and the backward pass pays again. Worse, the activations travel on the critical path: the next layer cannot start until the gather is done. So the summary's rule: tensor parallelism "requires very fast interconnects". In the professor's words later, "you wouldn't do tensor parallelism past an NVLink domain."

::widget fixture:lecture_07--tp-pp-traffic | add layers and TP traffic grows with them, a (W−1)/W share of one batch × num_dim activation each time; the weight shard never moves
::video 1:16:21-1:16:40 | per-layer activation traffic is why TP stays inside a node
::kp tensor-parallelism

### The backward pass: reduce-scatter

The code stops at the forward pass: "Backward pass: homework exercise". A student asked what happens in backprop, and the professor gave the answer aloud: "in forward, you're all gathering, in the backward, you're reduce scattering."

Here is why. In the forward pass, every rank used the *whole* gathered activation $x$ as input to its own column block of the next layer. So in the backward pass, each rank computes a gradient with respect to all of $x$, but only from its own block's contribution: a **partial** gradient. The true gradient of $x$ is the **sum** of the four partials. And each rank needs only the slice of that sum that its own weight block produced in the previous layer. Sum, then keep only your slice: a reduce-scatter.

::animation fixture:lecture_07--tp-gather-scatter | the forward all-gather takes each rank from 1 slice to all 4; the backward reduce-scatter runs it in reverse, from 4 partial slices to 1 summed slice

After the reduce-scatter each rank holds $128 \times 256 \times 4 = 131{,}072$ bytes of activation gradient. An all-reduce would leave the full $524{,}288$ bytes on every rank, four times what its weight block can use. The duality runs both ways: a forward pass that ends in a reduce-scatter needs an all-gather in its backward pass.

::predict tp-backward-reduce-scatter
::note spoken 1:08:33 | Asked whether autograd does this automatically: plain `.backward()` will not, "because there's no parallelism in that". In this course you would "manage and call the reduce scatter yourself", by design; in practice libraries do it for you.
::note deferred 1:09:14 | Writing the tensor-parallel backward pass, shapes and hooks, is still the homework; only which collective it needs was said in class.
::kp tp-backward-reduce-scatter

## Pipeline parallelism: split the depth {#pipeline}
source: lecture_07.py:L484-L536 · video 1:09:38-1:14:23

The third cut runs between layers. In **pipeline parallelism** each rank gets a contiguous subset of the layers, whole: all their dimensions, nothing split inside a layer. The data enters at rank 0 and flows through the ranks in order, like an assembly line.

::figure official/lectures/images/pipeline-parallelism.png | the cut now runs across the depth: each GPU owns a run of consecutive layers

### The algorithm

The lecture runs 4 layers on 2 ranks, so `local_num_layers = 4 / 2 = 2`: rank 0 holds layers 1–2, rank 1 holds layers 3–4, each a full `1024 × 1024` matrix. The batch of 128 is cut into 4 **micro-batches** of 32. For each micro-batch:
- rank 0 takes it from the data; every later rank **receives** it from rank $r - 1$ with `dist.recv`;
- the rank runs its own layers on it;
- every rank but the last **sends** the result to rank $r + 1$ with `dist.send`.

::code lecture_07.py:L509-L515 | rank 0 chunks the data; later ranks allocate empty buffers to receive into
::code lecture_07.py:L517-L530 | receive from the previous rank, run the local layers, send to the next

`send` and `recv` are not collectives. They are **point-to-point** operations, one sender and one receiver, which the professor introduced on the spot ("fairly explanatory"). Rank 1 pre-allocates its receive buffers because `recv` writes into a tensor that must already exist with the right shape.

::code lecture_07_stdout.txt:L72-L75 | rank 0 sends four 32 × 1024 tensors to rank 1, one per micro-batch

### What it costs

Communication happens only at **stage boundaries**, and only between neighbors. With two stages there is one boundary, crossed once per micro-batch: four sends of $32 \times 1024$, which together are just the batch's activation, $128 \times 1024$, once. Tensor parallelism moved an activation's worth after *every* layer; pipeline parallelism moves one per stage boundary, however many layers each stage holds.

::predict pipeline-sharding

That is why the summary says pipeline parallelism "can work with slow interconnects". The professor gave a concrete case: decentralized training, where "GPUs are halfway across the world", uses pipeline parallelism, and "you wouldn't want to do tensor parallel in that setting."

::widget fixture:lecture_07--tp-pp-traffic | compare the PP row with the TP row: a stage receives batch × num_dim per forward pass whatever m is, while TP pays at every layer
::kp pipeline-sharding

### The bubble, and why micro-batches

The catch is idleness. If the whole batch went through at once, rank 1 would wait while rank 0 computed, and rank 0 would sit idle while rank 1 computed. The idle stretches are **pipeline bubbles**: "while you're not processing, you're kind of waiting around for other tensors to process." The fix, in the code's comment, is to "break up into micro batches to minimize the bubble". Rank 0 hands off the first micro-batch as soon as it is done and starts the second, so both ranks work at once.

How much do micro-batches help? The lecture gives no formula; the standard one is GPipe's (Huang et al., 2018). With $p$ stages and $m$ equal micro-batches, the last micro-batch leaves the last stage after $m + p - 1$ time slots, and each stage is busy for $m$ of them:

$$ \text{idle fraction} = \frac{p-1}{m+p-1} $$

::worked pipeline-bubble
::animation fixture:pipeline-bubbles | with 4 stages and one batch, three stages idle at every moment; at m = 4 and m = 8 the micro-batches fill the middle, and only the ramp-up and ramp-down stay idle

For the lecture's $p = 2$: one micro-batch leaves each stage idle half the time; four leave it idle $1/5 = 20\%$. Micro-batching does not reduce anyone's work, it only fills the idle slots. The price is smaller matrix multiplies per micro-batch, which run less efficiently (L2's arithmetic intensity). The bubble shrinks slowly: for $p = 8$, 32 micro-batches still leave $7/39 \approx 18\%$ idle, and getting below 10% takes $m = 64$.

::predict pipeline-bubble

### Overlap: what the naive version leaves out

The code's `send` and `recv` block: a rank computing a micro-batch is not sending the previous one or receiving the next. "If you put an I before these" (`isend`, `irecv`), "then it becomes kind of async", and a stage can communicate while it computes. Overlapping communication with computation "is actually very important to pipeline parallelism"; the lecture marks it "not handled".

::code lecture_07.py:L532 | the naive pipeline does not overlap communication with computation
::note deferred 1:10:48 | Micro-batch schedules and bubbles get their full treatment in the next lecture ("will talk more about this on Wednesday"); the backward pass is homework (L534).
::kp pipeline-bubble

## How do the strategies fit together? What's missing, and the summary {#summary}
source: lecture_07.py:L59-L72 · video 1:14:23-1:20:57

### What the toy versions leave out

The lecture lists what its bare-bones implementations skip, to be filled in next time:
- **overlapping communication with computation.** It matters most in pipelines, but in data parallelism too: the toy does the whole backward pass and then all the all-reduces, while a clever version sends each gradient as soon as it is computed;
- **more general models**, with attention and so on. The MLP shows the core; larger models only add bookkeeping that hides the algorithm;
- **other forms of parallelism.** *Sequence parallelism* chops a sequence into pieces across ranks, which parallelizes attention. *Expert parallelism* puts different experts of an MoE on different ranks, which is where all-to-all comes in. And *combinations* of all of these.
- **JAX and TPUs**, where you only define the model and how each tensor is sharded, and the compiler decides which collectives to insert (the code links Stanford's [Levanter](https://crfm.stanford.edu/2023/06/16/levanter-1_0-release.html)). The course stays with PyTorch's primitives on purpose: the compiler route "would take a lot of the joy out of actually building things from scratch."

::code lecture_07.py:L59-L64 | the lecture's own list of what is missing
::note deferred 1:15:57 | Combining strategies "will show up in the assignment".

### Four axes, chosen by the hardware

The summary sorts the strategies by the axis of the computation they cut:
- **data** parallelism cuts the **batch**;
- **tensor** (and **expert**) parallelism cuts the **width**;
- **pipeline** parallelism cuts the **depth**;
- **sequence** parallelism cuts the **length**.

Which to use is "strongly dependent on the hardware". Tensor parallelism communicates a full activation in every layer, so it lives inside an NVLink domain. Pipeline parallelism tolerates slow links. Data parallelism needs one all-reduce per step. A typical large job uses tensor parallelism within a node, then data parallelism or FSDP across nodes, then pipeline parallelism if needed.

They compose because they cut different axes. Give every GPU a coordinate on each: $W = W_d \cdot W_t \cdot W_p$. Only the width and depth splits shrink the model on a GPU, so each GPU holds $1/(W_t W_p)$ of the parameters, and that share is replicated $W_d$ times. This accounting is ours (it matches Megatron-LM's 2021 paper); the lecture names combinations without counting them.

::predict parallelism-axes
::worked parallelism-axes
::widget fixture:lecture_07--axes-composer | the parameter share per GPU depends only on W_t·W_p; push W_t past the NVLink domain and the readout warns that tensor parallelism now crosses Infiniband

For 160B parameters in bf16 with $W_t = 8$, $W_p = 8$, $W_d = 16$ (1024 GPUs), each GPU holds $160\text{e}9 / 64 = 2.5\text{e}9$ parameters, 5 GB of weights.

Data parallelism has a limit of its own. Growing it means growing the batch, and past the **critical batch size** "it doesn't actually help you": you are wasting compute and are better off using tensor parallelism.

::video 1:17:14-1:17:56 | TP inside a node, then DP or FSDP, then PP; and where data parallelism stops paying
::note deferred 1:17:56 | The critical batch size and other strategy trade-offs come later in the course ("we'll talk more about as we go through the class"); see [L9](#/t/lecture_09).

### The lecture's summary

- **Many ways to parallelize:** data (batch), tensor or expert (width), pipeline (depth), sequence (length).
- **Data parallelism:** DDP uses one all-reduce of the gradients per step and keeps a full copy of everything on every rank, about $2S$ bytes of traffic per rank whatever the world size. FSDP and ZeRO use the all-reduce's two halves, all-gather and reduce-scatter, to keep state sharded.
- **Tensor parallelism** requires very fast interconnects such as NVLink: it all-gathers an activation in every layer forward and reduce-scatters in backward.
- **Pipeline parallelism** can work with slow interconnects, since only stage boundaries communicate, but you must work to reduce the bubbles, $(p-1)/(m+p-1)$ of the time.
- **Recompute, store, or communicate.** Any intermediate you need later can be recomputed (as in [L2's activation checkpointing](#/read/lecture_02)), stored in your own memory, or stored in another GPU's memory and communicated when needed. DDP's redundant optimizer step is this choice made one way: compute it everywhere rather than move optimizer state.
- **The hierarchy is here to stay.** Hardware is getting faster, but "we'll always want bigger models", so there will always be a fast local level and slow distant ones, and the same accounting.

::code lecture_07.py:L66-L72 | the lecture's six-line summary
::note deferred 1:20:49 | Next lecture: a deeper dive into parallelism, including FSDP/ZeRO and pipeline schedules. See [L8](#/t/lecture_08).
::kp parallelism-axes
