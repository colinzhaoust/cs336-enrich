---
title: L2 · Resource accounting, read through
minutes: 40
---
This lecture teaches you to price a training run before you launch it: how many bytes each tensor takes, how many FLOPs each operation costs, why the GPU is usually waiting on memory, and what a full training step stores and computes. After it you can estimate training days and the largest model that fits from a few multiplications, and say whether an operation is limited by compute or by memory.

## What is this lecture for? {#goal}
source: lecture_02.py:L16-L38 · video 0:05-1:55, 3:43-4:43

The course's guiding question, from [L1](#/read/lecture_01), is: what is the best model you can train with a fixed budget of compute and memory? Answering it means maximizing *computational efficiency*, and you cannot optimize what you cannot measure. So before any modeling, this lecture builds the ledger: for any piece of PyTorch code, how much memory does it hold and how much compute does it spend?

The professor opened with a result that shows why such accounting matters. The Marin project's $10^{23}$-FLOP training run, mentioned in L1, had finished, and its final loss landed within 0.05 of the value forecast in advance. The forecast came from a scaling law fitted to many smaller runs (IsoFLOP curves, each a set of models trained with the same compute). Predicting a big run from small ones only works if you can count the compute each run spends.

He named three things to take away:
- **Mechanics**: straightforward; this is just how PyTorch tensors work.
- **Mindset**: resource accounting. Every time you write a line of code, think about what it costs.
- **Intuitions**: a feel for where the resources go. There is "no ML magic today"; architecture is the next lecture's subject.

The lecture is an executable Python file. Its `text()` calls are the slides, and the code between them actually runs (on the professor's laptop, as it turns out, which matters later). This read-through follows the order in which `main()` calls its sections: memory of tensors, compute of operations, the link between them (arithmetic intensity), then the memory and compute of a whole training step, and finally two tricks for saving memory.

::note deferred 1:05 | Tokenization, the previous lecture's topic, is left to Assignment 1. Modeling and architecture go to the next lecture ("I'll leave that to Tatsu"); today is accounting only.

## How long would training take, and how big a model fits? {#napkin}
source: lecture_02.py:L71-L86 · video 1:55-3:43

The lecture opens with two questions you should be able to answer by the end. They use formulas derived later, so for now take the constants on trust; the rest of the lecture explains each one.

### Question 1: training time

*How long would it take to train a 70B-parameter model on 15T tokens on 1024 H100 GPUs?*

The recipe has four parts:
- **Total compute.** Training costs about $6ND$ floating-point operations (FLOPs), where $N$ is the parameter count and $D$ the number of training tokens. Here $6 \times 70\text{e}9 \times 15\text{e}12 = 6.3\text{e}24$ FLOPs. The section on the backward pass, near the end, derives the 6.
- **Hardware speed.** One H100 does about $9.9\text{e}14$ FLOP/s in bf16 (a 16-bit number format covered below). The datasheet says 1979 teraFLOP/s, but that figure assumes a special sparse format, so the code divides it by 2.
- **Utilization.** Real code never reaches the peak. The lecture assumes a model FLOPs utilization (MFU) of 0.5, meaning half the peak; MFU is defined properly later.
- **Divide.** Daily throughput is $9.9\text{e}14 \times 0.5 \times 1024 \times 86400 \approx 4.4\text{e}22$ FLOPs per day, so the run takes $6.3\text{e}24 / 4.4\text{e}22 \approx 144$ days.

::predict napkin-training-time
::code lecture_02.py:L72-L77 | the whole estimate is four lines of arithmetic
$$ \text{days} = \frac{6ND}{\text{FLOP/s per GPU} \times \text{MFU} \times \#\text{GPUs} \times 86400} $$

Nothing in the formula is quadratic. Days grow linearly with parameters and tokens and shrink linearly with GPU count and utilization. Four times the GPUs at MFU 0.4 instead of 0.5 gives $144 / (4 \times 0.8) = 45$ days.

::note slip 2:46 | The professor reads the answer as 143 days; the exact quotient is 143.9, so the practice prompts use 144. Same arithmetic, different rounding.

### Question 2: the largest model that fits

*What is the largest model you can train on 8 H100s using the AdamW optimizer?*

Each H100 has 80 GB of high-bandwidth memory (HBM), so 8 of them hold 640 GB. Training keeps four things per parameter in memory for the whole step: the parameter itself, its gradient, and two numbers the optimizer tracks (AdamW's first and second moments). In the lecture's accounting the parameter and gradient are 2 bytes each (bf16) and the two moments 4 bytes each (fp32), so 12 bytes per parameter, and $640\text{e}9 / 12 \approx 53$B parameters.

::worked napkin-max-model-size
::code lecture_02.py:L79-L83 | 2 + 2 + (4 + 4): weight, gradient, two optimizer moments

This is an upper bound. It leaves out the activations (the intermediate results the backward pass needs), whose size depends on batch size and sequence length. The optimizer state is two thirds of the 12 bytes, so swapping AdamW for SGD with momentum (one fp32 buffer, 8 bytes per parameter in total) raises the limit to 80B.

::widget fixture:napkin | change the GPU count or MFU and watch days move inversely; change bytes per parameter and only the divisor of the memory answer moves
::note spoken 3:43 | Both answers are rough by design: "the point is not to precisely calculate every single thing, but just get the rough shape of things."
::note aside | 2 + 2 + 4 + 4 has no fp32 "master copy" of the weights. Many real recipes, including the blog the lecture cites later, keep one, which makes 16 bytes per parameter and about 40B here. The lecture's own number is 12 and 53B.
::kp napkin-training-time
::kp napkin-max-model-size

## What is a tensor, and how much memory does it take? {#tensors}
source: lecture_02.py:L89-L132 · video 4:43-8:46

Everything a training run stores is a **tensor**, a multi-dimensional array: the data, the parameters, the gradients, the optimizer state and the activations. A released model checkpoint is literally a bag of named tensors; the lecture points to the file index of [DeepSeek v3.2 on Hugging Face](https://huggingface.co/deepseek-ai/DeepSeek-V3.2?show_file_info=model.safetensors.index.json), where each entry has a shape and a precision. Because everything is a tensor, one rule prices all of it.

A tensor's **rank** is its number of dimensions (axes): `torch.zeros(4)` is rank 1 (a vector), `torch.zeros(4, 8)` rank 2 (a matrix), `torch.zeros(4, 8, 2)` rank 3. Rank counts axes, not elements, and it is not the linear-algebra rank of a matrix. Transformers routinely use rank 4, for example (batch, sequence, heads, head dimension).

::code lecture_02.py:L100-L110 | rank is the number of axes; the last tensor has the four axes a Transformer layer carries

The memory rule is: **bytes = number of elements × bytes per element**. The default element type in PyTorch is `float32`, 4 bytes, so a 4 × 8 matrix takes $32 \times 4 = 128$ bytes.

::code lecture_02.py:L125-L132 | numel times element_size, checked by assertions; then one GPT-3 matrix

Scale makes this rule matter. One feedforward weight matrix of GPT-3 has shape $(4 \cdot 12288) \times 12288$, about 604 million elements; in fp32 that is 2304 MiB, roughly 2.4 GB, for a single matrix. And GPT-3 is an old model.

### fp32: the default

A 32-bit float (fp32, "single precision") has 1 sign bit, 8 exponent bits and 23 mantissa (fraction) bits. The exponent sets the **dynamic range**, how large and how small a number can be; the mantissa sets the **resolution**, how finely nearby values are distinguished.

::figure official/lectures/images/fp32.png | 1 sign bit, 8 exponent bits, 23 mantissa bits

The name comes from scientific computing, where fp32 was the baseline you could expect and fp64 ("double") the upgrade for demanding simulations. Deep learning goes the other way: its computations do not need that much precision, so you can be "a lot sloppier" and use fewer bits.

::note spoken 8:20 | Fewer bits save more than memory. Operating on 16-bit numbers is also faster, "let's say twice as fast, but not always", and the professor promised that reducing memory saves time in a less obvious way too. The arithmetic-intensity section below is that explanation.
::kp tensor-rank

## How few bits can a number have? {#precision}
source: lecture_02.py:L134-L199 · video 8:46-18:05

Halving the bits halves the memory. The question is what you lose.

### fp16: half the bits, too little range

fp16 ("half precision") keeps 1 sign bit, cuts the exponent to 5 bits and keeps 10 mantissa bits. It takes 2 bytes per element, but its dynamic range is poor: it cannot represent very large numbers or very small ones.

::figure official/lectures/images/fp16.png | only 5 exponent bits, so the range is narrow

Try to store $10^{-8}$ in fp16. Before reading on, decide what you will get back (the prompt also asks about bf16, the format introduced next).

::predict dtype-bytes

You get exactly 0. The smallest positive fp16 value is the subnormal $2^{-24} \approx 5.96 \times 10^{-8}$, and $10^{-8}$ is closer to 0 than to it, so it rounds to 0. That is **underflow**. Training in fp16, which people did "back in the day", produces underflow, overflow and NaNs, and with them unstable training.

### bf16: same size, fp32's range

Google Brain designed bfloat16 (bf16, "brain floating point") in 2018 to fix this. It keeps 16 bits but moves bits from the mantissa to the exponent: 1 sign, 8 exponent, 7 mantissa. With the same 8-bit exponent as fp32 it has the **same dynamic range as fp32**, at the same memory cost as fp16. The price is resolution, since 7 mantissa bits distinguish nearby values far more coarsely. In bf16, $10^{-8}$ is stored as about $1.0012 \times 10^{-8}$, not 0.

::figure official/lectures/images/bf16.png | the exponent field is as wide as fp32's; the mantissa is what got cut
::worked dtype-bytes
::code lecture_02.py:L138-L152 | the same 1e-8 underflows in fp16 (assert x == 0) and survives in bf16

Why is the coarse resolution acceptable? Because, in the professor's words, deep learning is "sloppy and stochastic anyway". Gradients are noisy estimates, so an extra rounding error per value barely matters, while a value that overflows to infinity or underflows to zero breaks training.

::video 10:30-10:54 | why losing mantissa bits is an acceptable trade in deep learning
::note skip 10:54 | Right after explaining bf16, the professor said "let me actually skip over this part". The bf16 no-underflow demo (L151-L152) was most likely not shown live; the result is the one stated above.

### Mixed precision: bf16 where it is safe, fp32 where it is not

What should training use? The professor's spoken advice comes first: for a small model you do not want to fuss over, plain fp32 is fine; it just costs 4 bytes per value. Pure fp16 is too risky. bf16 is "the sweet spot", though even bf16 "can be risky as well" if used for everything.

The common practice is **mixed precision training** (Micikevicius et al., [2017](https://arxiv.org/pdf/1710.03740.pdf)): different tensors get different precisions according to their role.
- **bf16** for parameters, activations and gradients.
- **fp32** for the optimizer state.

The optimizer state stays in fp32 because of what it holds. Adam-style optimizers keep running averages of gradients and of squared gradients, built up over thousands of steps from tiny increments; at bf16's resolution those increments would be rounded away. The professor gave this reason later, at the optimizer: people have tried bf16 there, and squaring and averaging over many steps is not stable.

PyTorch implements a version of this automatically, AMP (automatic mixed precision): inside an `autocast` block it casts operations to bf16 where that is safe. Matrix multiplications are safe; exponentiation stays in higher precision.

::code lecture_02.py:L160-L166 | the rule by tensor role, then autocast deciding per operation
::video 1:10:24-1:10:45 | why optimizer state is the one thing kept in fp32
::note skip 12:40 | AMP itself was skipped ("We're not going to talk too much about this"); its casting policy beyond "matmuls safe, exp not" is not part of the lecture.
::note aside | The 2017 paper's own devices, an fp32 master copy of the weights and loss scaling, are not mentioned. The lecture keeps the rule and drops the mechanism, consistent with its napkin count of 12 bytes per parameter rather than 16.

### Going further: fp8 and fp4

The professor expects this class to stop at bf16, but hardware goes lower.

**fp8** (1 byte) was standardized in 2022 for machine learning. The H100 supports two variants, depending on whether you need range or resolution: E4M3 (4 exponent bits, range ±448) and E5M2 (5 exponent bits, range ±57344). NVIDIA's Transformer Engine library supports it.

::figure https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/_images/fp8_formats.png | E4M3 spends a bit on resolution, E5M2 on range
::note skip 13:37 | The trade-off between the two fp8 variants, and when to use each, was explicitly skipped.

**nvfp4** (NVIDIA, 2025) uses 4 bits per value. Four bits can only spell these numbers: −6, −4, −3, −2, −1.5, −1, −0.5, 0, 0.5, 1, 1.5, 2, 3, 4, 6. Training on that grid alone would fail, so values are grouped into **blocks** that share a separate scale factor. Each value effectively gets more than 4 bits of dynamic range, with one restriction: a value cannot be far larger or smaller than its neighbors in the same block. NVIDIA's Nemotron 3 Super was trained in NVFP4. Much of this happens inside NVIDIA's software stack; you do not simply create an fp4 tensor yourself.

::note spoken 15:19 | Asked how the block scaling works, the professor restated it: within a block each value varies over its 4 bits, and the whole block is scaled up or down; "you can't have this value be way over here and the neighboring value way down here."
::note deferred 16:30 | Asked about 1-bit models: low-bit numbers are mostly an inference story. You train in something like bf16 and then *quantize* to 1 or 2 bits, which is much easier than training a 1-bit model, which he does not think anyone has done credibly. Quantization is left to a later lecture.

### Where tensors live

PyTorch creates tensors in CPU memory. To use the GPU's parallelism you must move them to GPU memory with `x.to(device)` or create them there directly. Forgetting to do so is the classic way to lose your speed-up.

::figure official/lectures/images/cpu-gpu.png | CPU and GPU each have their own memory; data must be moved across
::code lecture_02.py:L186-L199 | move a tensor, or create it on the GPU directly
::note aside 17:27 | The lecture was executed on the professor's laptop, which has no GPU. GPU-dependent results (the timings, the measured MFU, the checkpointing memory) were shown but not run, so the numbers in the trace are not meaningful.
::kp dtype-bytes
::kp mixed-precision

## How do you write tensor operations without losing track of axes? {#einops}
source: lecture_02.py:L202-L276 · video 18:05-27:20

Before counting the compute of operations, the lecture takes a detour through notation, because the FLOP counting later is written in it. The problem it solves is code like this:

::code lecture_02.py:L216-L218 | which axes are -2 and -1? You have to reconstruct the shapes in your head

Positional axis numbers are easy to get wrong. **einops** is a library in which every axis has a name, inspired by Einstein summation notation. The lecture covers three of its functions and leaves the rest to the [einops tutorial](https://einops.rocks/1-einops-basics/).

### einsum: matrix multiplication with bookkeeping

The professor's one-line definition: einsum is "a generalized matrix multiplication with good bookkeeping". You name the axes of each input and of the output. Any axis that appears in an input but **not in the output is summed over**.

::code lecture_02.py:L232 | x is (seq1, hidden), y is (hidden, seq2); hidden is missing from the output, so it is summed

Read it as a loop. Enumerate every combination of the named indices (seq1, hidden, seq2), multiply `x[seq1, hidden] * y[hidden, seq2]`, and add the product into `z[seq1, seq2]`. With x of shape 3 × 5 and y of shape 5 × 7, z is 3 × 7 and each entry sums 5 products.

::animation fixture:matmul-contraction | the highlighted index walks along the contracted axis of both inputs and never appears in the output

The benefit shows with batches. For two tensors of shape (batch, seq, hidden), the old way is `x @ y.transpose(-2, -1)`, which relies on PyTorch implicitly batching over every axis but the last two. The einsum version just names everything:

::predict einsum
::code lecture_02.py:L243-L247 | no transpose: naming hidden on both sides does it; `...` stands for any number of batch axes

The transpose disappears because the naming does it. Writing `...` instead of `batch` lets the same line work for any number of leading axes (batch, sequence, heads), so code stays modular whatever shape comes in. The professor admits he always gets confused by transposes, and not having to think about them "makes me happy".

::note spoken 23:50 | Asked whether einops is faster: no, it reduces to the same primitive operations; "you can think about it as just sugar." The gain is readability.

### reduce: sum, mean, max or min over a named axis

`reduce` generalizes `sum`, `mean`, `max` and `min`. `x.sum(dim=-1)` becomes `reduce(x, "... hidden -> ...", "sum")`: the axis that disappears on the right is the one reduced, and the last argument names the operation.

::code lecture_02.py:L255-L258 | the old positional sum and the named reduce

### rearrange: split and merge axes

Sometimes one axis secretly holds two, because a matrix was flattened. Take x of shape (3, 8) where the 8 is really `heads × hidden1` = 2 × 4, and suppose you want to multiply each head's 4-vector by a 4 × 4 matrix w. `rearrange` with parentheses splits the axis (you give `heads=2` since 8 could also be 4 × 2), einsum transforms the new last axis, and a second rearrange merges the axes back.

::code lecture_02.py:L270-L276 | split (heads hidden1), transform hidden1 to hidden2, merge back

::note spoken 26:31 | Asked whether flattening is row-major or column-major: the order of the names inside the parentheses fixes it.
::note deferred 24:10 | rearrange "will come up in an assignment once". The professor's advice: einops takes time to get used to but is worth it, because once you think in einsum, transposes and reductions become fluid.
::kp einsum

## How much compute does an operation cost? {#flops}
source: lecture_02.py:L279-L335, L830-L848 · video 27:20-40:30

Memory was easy: elements times bytes. Compute is measured in **FLOPs**, floating-point operations, where one FLOP is a basic operation such as an addition or a multiplication. GPUs can do other things, but these are the bread and butter that eat most of the time, so everything else is ignored.

Two acronyms sound the same and are constantly confused:
- **FLOPs**: a *count* of operations, the amount of work. "GPT-3 took 3.14e23 FLOPs."
- **FLOP/s**: operations *per second*, the speed of hardware. Also written FLOPS with a capital S, which is exactly the confusion; the lecture always writes `/s`. "An H100 does 989 teraFLOP/s."

### Reference numbers

To calibrate: training GPT-3 (2020) took $3.14 \times 10^{23}$ FLOPs; GPT-4 (2023) is speculated to have taken $2 \times 10^{25}$.

On the hardware side, the H100 datasheet lists 1979 teraFLOP/s for bf16. The professor told this as a story: you benchmark, you fall short, you read the fine print, and the footnote says the figure assumes sparsity (a special structured-sparse matrix format). Dense matrices get half, so "you always have to take these numbers divide by 2": about $9.9 \times 10^{14}$ FLOP/s. That is the `/ 2` you saw in the napkin math.

::video 29:27-29:57 | why the datasheet FLOP/s is always halved before any napkin math

So 8 H100s (one node) for two weeks deliver $8 \times 2 \times 604800 \times 9.9\text{e}14 \approx 9.6 \times 10^{21}$ FLOPs, about 3% of GPT-3's training compute.

::code lecture_02.py:L296 | 8 GPUs × 2 weeks × seconds per week × FLOP/s
::note slip 30:15 | Reading this line aloud, the professor decided it "looks like it's one week" and said about 5e21. The code multiplies by 2 and its text says two weeks, so the code's value is about 9.6e21; one week would be about 4.8e21.

### The FLOPs of a matrix multiply

Most FLOP counting reduces to matrix multiplications, so the lecture takes a linear model: $B$ data points of dimension $D$, mapped to $K$ outputs. The data matrix x is $B \times D$, the weights w are $D \times K$, and `y = x @ w`.

Each output entry `y[i][k]` is a dot product of length $D$: $D$ multiplications and $D - 1$ additions, about $2D$ FLOPs. There are $B \cdot K$ output entries. Equivalently, for every triple $(i, j, k)$ there is one multiply `x[i][j] * w[j][k]` and one add.

$$ \text{FLOPs}(x_{B\times D}\, w_{D\times K}) = 2\,B\,D\,K $$

::predict matmul-flops
::code lecture_02.py:L308-L314 | the GPU-sized case is B=16384, D=32768, K=8192, about 8.8e12 FLOPs; the CPU fallback is much smaller
::widget fixture:matmul-contraction | step through the cells: each costs 2·D, there are B·K of them, and sliding any one of B, D, K scales the total linearly

The professor dropped the $-1$ on purpose: for $D$ in the thousands, $2D - 1$ and $2D$ are indistinguishable.

Now reread $2BDK$. $B$ is the number of data points, and $D \cdot K$ is the number of parameters in w. So the forward pass of a linear layer costs **2 × (data points) × (parameters)** FLOPs. This is the first piece of the $6ND$ from the opening question; the backward pass supplies the other 4.

::video 33:59-34:45 | 2·B·D·K re-read as 2 × data points × parameters
::note spoken 33:00 | Two student questions. Sub-cubic matrix multiplication algorithms do not matter in practice; real gains come from co-designing algorithms with the hardware, not from asymptotics. And adds and multiplies cost the same: "the way the hardware is built, the two are basically the same."

### Why counting only matmuls is safe

The code goes straight from "what is a FLOP" to the matmul and never says why other operations can be ignored. The professor said it aloud. An elementwise operation on an $m \times n$ matrix (adding two matrices, applying ReLU) costs about $mn$ FLOPs, a small constant per element. The matmul that produced that matrix from an inner dimension $k$ cost $2mnk$. So for large enough matrices, nothing else you will meet is as expensive as matrix multiplication.

Take a layer `h = x @ W` with x and W both 4096 × 4096, followed by GELU (an activation function costing about 20 FLOPs per element, the lecture's own estimate, used later). GELU's share of the FLOPs is $20 / (20 + 2 \cdot 4096) \approx 0.24\%$. With a tiny hidden size of 64 the share rises to $20/148 \approx 13.5\%$: the rule holds only "for large enough matrices".

::widget fixture:lecture_02--elementwise-share | GELU's FLOP share falls like 10/D as D grows, but its share of the *time* stays far larger, because GELU is limited by memory, not compute
::video 32:31-32:58 | the condition "for large enough matrices" and the caveat "when we talk about memory"

The professor attached a caveat in the same breath: this is about FLOPs, not time. Elementwise operations are cheap in FLOPs and still slow, for a reason the next section explains.

::kp matmul-flops
::kp elementwise-flops-negligible

### Timing an operation on a GPU

FLOPs are a property of the computation, independent of hardware. To learn how long it takes, you time it. On a GPU there is a trap: PyTorch launches GPU work **asynchronously**. `x @ w` puts a kernel on the GPU's queue and returns immediately, while the GPU is still working. A timer around that line measures only the launch.

So the lecture's `benchmark()` helper calls `torch.cuda.synchronize()` before starting the clock, so earlier queued work is not counted, and again after the operation, so the clock waits until the GPU has finished. It runs several trials and averages them.

::code lecture_02.py:L830-L848 | synchronize before, synchronize after each run, average over trials
::animation fixture:lecture_02--sync-timeline | without the closing synchronize the timer window shuts when the launch returns, while the kernel is still running

::predict gpu-timing-synchronize

If you omit it, the professor warned, "you're going to find that wow, your timings are really fast", because the call is non-blocking. The giveaway is a FLOP/s figure above the datasheet: a 1.1e12-FLOP matmul "measured" at 20 µs would run at about 56 times the H100's peak, which no GPU can do. The timer is wrong, not the hardware.

::note deferred 35:18 | Benchmarking proper, with this as "a little preview", is the subject of a later lecture.
::kp gpu-timing-synchronize

### Model FLOPs utilization

Dividing the FLOP count by the measured time gives the **actual** FLOP/s. The datasheet gives the **promised** FLOP/s. Their ratio is the model FLOPs utilization:

$$ \text{MFU} = \frac{\text{actual FLOP/s}}{\text{promised FLOP/s}} $$

The definition ignores communication between GPUs and other overhead. You never get more than promised, and often get much less. A bf16 matmul doing 3.4e10 FLOPs in 50 µs runs at 6.8e14 FLOP/s, an MFU of about 0.69.

The promised number depends heavily on the **data type**. The lecture's `get_promised_flop_per_sec` takes the dtype as an argument: an H100 promises 67.5 teraFLOP/s in fp32 against about 989 in bf16, and an A100 19.5 against 312. MFU is always measured against the peak for the dtype you are using.

::code lecture_02.py:L325-L330 | the promised figure is looked up by dtype, then MFU is a single division
::widget fixture:lecture_02--mfu | switching bf16 to fp32 changes only the promised denominator; pushing the measured time low enough drives MFU above 1, which means the timer is broken

How good is good? The code says only that MFU of at least 0.5 is "quite good". Aloud the professor gave three bands: about 0.5 for a modern model and "you should be pretty happy with yourself"; a bare matmul might reach about 0.8; and about 0.1 means something is wrong.

::video 37:52-38:21 | the three MFU bands: about 0.1 broken, 0.5 good, 0.8 for a bare matmul
::note spoken 38:53 | A student asked which number is "promised": the spec-sheet figure already divided by 2 (989), and "on top of that, you only get 0.5 of that in general".
::note spoken 40:00 | The dtype dependence has a practical edge: "if you try to do float32 nowadays, it's going to be really, really slow", because hardware is optimized for bf16 and fp8.
::note aside 34:46 | The timing and MFU in the lecture's trace came from a laptop CPU, so the professor called them "not very meaningful".

Why is a good MFU only 0.5 and not close to 1? The professor deferred that question ("I'll come back to that when we talk about memory bottlenecks"), and the next section answers it.

::kp mfu

## Why is the GPU usually waiting? Arithmetic intensity {#intensity}
source: lecture_02.py:L338-L468 · video 40:30-54:45

Here is the professor's cartoon of the hardware. The tensors sit in high-bandwidth memory (HBM). The compute units sit elsewhere on the chip. To compute anything you (1) send the inputs from memory to the compute units, (2) compute, and (3) send the outputs back.

::figure official/lectures/images/compute-memory.png | two boxes and the link between them: the link has its own speed limit

So how long an operation takes depends on two hardware speeds, not one:
- **accelerator speed**, in FLOP/s: about $9.9 \times 10^{14}$ for an H100 in bf16;
- **memory bandwidth**, in bytes/s: about $3.35 \times 10^{12}$ for an H100.

This is why the lecture began with memory. Size matters for capacity (a model too big for memory will not run) and also for speed, because every byte has to be moved, and moving takes time.

### ReLU: the measuring stick

Take a vector of $n = 2^{20}$ (about a million) bf16 values and apply ReLU, $\max(x, 0)$ elementwise.
- **Bytes moved:** read x, $2n$ bytes (bf16 is 2 bytes per value), and write y, another $2n$: $4n$ in total.
- **FLOPs:** one comparison per element, $n$.
- **Communication time:** $4n / 3.35\text{e}12 \approx 1.3 \times 10^{-6}$ s.
- **Computation time:** $n / 9.9\text{e}14 \approx 1.1 \times 10^{-9}$ s.

The lecture assumes communication and computation **overlap perfectly**: values start being computed as soon as they arrive, and results go back while the rest is still in flight. So the total time is the larger of the two, about a microsecond. In practice overlap is imperfect and there is some overhead, but the max is good enough here.

::code lecture_02.py:L368-L375 | count bytes and FLOPs, convert each to time, take the max

The comparison names the bottleneck:
- **memory-bound**: communication time is larger; the compute units sit waiting for bytes to arrive;
- **compute-bound**: computation time is larger; the compute units are the bottleneck.

ReLU is memory-bound by a factor of about a thousand.

### The same test as one ratio

Divide both sides of that comparison by bytes and it becomes a comparison of two ratios.
- **Accelerator intensity**: how many FLOPs the hardware can do per byte it moves, $9.9\text{e}14 / 3.35\text{e}12 \approx 295$ for an H100 in bf16. The professor's number to keep in your head: "about 300".
- **Arithmetic intensity**: how many FLOPs this operation actually does per byte it moves, FLOPs ÷ bytes. For ReLU, $n / 4n = 0.25$.

$$ \text{memory-bound} \iff \frac{\text{FLOPs}}{\text{bytes}} < \frac{\text{peak FLOP/s}}{\text{bandwidth (bytes/s)}} $$

An operation is memory-bound when its arithmetic intensity is below the accelerator's, compute-bound when above. If someone tells you an intensity is 0.25, the professor said, you should think "this is really bad". In general you will find yourself memory-bound, because data movement is expensive. The way out is to do more work per byte moved. The lecture now walks through five operations in increasing intensity so you can watch the crossover happen.

### GELU: more work, same time

GELU is a smoother activation, $\text{GELU}(x) = 0.5\,x\,(1 + \tanh(\sqrt{2/\pi}\,(x + 0.044715\,x^3)))$. The bytes are unchanged, $4n$, but the tanh costs work; the lecture's crude estimate is 20 FLOPs per element. Intensity: $20n / 4n = 5$. Twenty times ReLU's, and still far below 295, so still memory-bound.

The consequence is counterintuitive. GELU looks far more expensive than ReLU, but run on its own it takes the same time, because the time is set by the bytes moved, which are identical. As the professor put it, "it's exactly the same, because that's not where the bottleneck is." This is the caveat from the previous section: elementwise operations are negligible in FLOPs but not in time.

### Dot product and matrix-vector: still memory-bound

A dot product of two length-$n$ bf16 vectors reads $2n + 2n$ bytes and writes a 2-byte scalar; it does $n$ multiplications and $n - 1$ additions, $2n - 1$ FLOPs. Intensity: $(2n-1)/(4n+2) \approx 0.5$, whatever $n$ is, because FLOPs and bytes both grow linearly in $n$.

::code lecture_02.py:L424-L427 | 4n + 2 bytes, 2n − 1 FLOPs, intensity about 1/2
::kp supp-dot-product-cost

A matrix-vector product, an $n \times n$ matrix times a length-$n$ vector, is $n$ dot products. The professor polled the room on this one before revealing it. It reads $2n + 2n^2 + 2n$ bytes and does $n(2n - 1)$ FLOPs, so for $n = 1024$ its intensity is 0.998, about 1. "Barely, barely higher", and memory-bound. Each matrix entry is read once and used in a single multiply-add, so intensity stays near 1 at any $n$.

### Matrix multiplication: finally compute-bound

Multiply two $n \times n$ matrices. Read $2n^2 + 2n^2$ bytes, write $2n^2$: $6n^2$ bytes. There are $n^2$ output entries, each a dot product of length $n$: $n^2(2n - 1)$ FLOPs. Intensity:

$$ \frac{n^2(2n-1)}{6n^2} \approx \frac{n}{3} $$

::predict arithmetic-intensity

For $n = 1024$ that is about 341, above 295: **compute-bound**. The professor gave the intuition the code leaves as a bare comment: you send $n^2$ numbers but compute $n^3$ products, so work per byte grows like $n$, and every input value is reused $n$ times. Intensity crosses 295 at about $n = 887$.

::code lecture_02.py:L455-L458 | 6n² bytes, n²(2n − 1) FLOPs, intensity about n/3
::widget fixture:lecture_02--roofline | raise n and the matmul point slides right (about n/3) across the kink near n ≈ 887, while the matvec point stays near intensity 1 for every n
::video 52:01-52:29 | why matmul intensity grows like n, and why that is the reason people ask for big batches

This is why practitioners ask for large batch sizes and large matrices. Below the accelerator intensity, making an operation smaller does not make it faster; it is all the same, waiting on memory. Above it, you are actually saturating the GPU.

### What this means for Transformers

Training Transformers is mostly big matrix multiplications "with some things sprinkled in between", which is good news for intensity; the architecture is designed that way. Inference is different. Generating text produces one token at a time, so each weight matrix multiplies a single vector: a matrix-vector product, memory-bound. In training the whole sequence is processed at once, which makes these matrix-matrix products.

The precision matters too: an fp32 value is 4 bytes instead of 2, and the H100's fp32 peak is far lower (67.5e12 / 3.35e12 puts its kink near 20 FLOPs per byte), so both intensities change with the dtype.

Now the deferred question has its answer. MFU is actual FLOP/s over promised FLOP/s, and the promise assumes the compute units never wait. Every memory-bound operation in a model (the elementwise operations, the normalizations, the small products) runs well below peak. That is why 0.5 counts as good.

::note deferred 53:03 | The intensity of an actual Transformer is left to Assignment 1 and the next lecture. Why inference is memory-bound is left to the inference lecture.
::note deferred 44:19 | How communication and computation really overlap is left to the GPU lecture; here it is assumed perfect.
::note slip 54:03 | The professor said "by default everything we're doing here is fp16"; the code uses bf16 throughout. Both are 2 bytes per value, so every number above holds.
::kp arithmetic-intensity

## How do you see all of this in one picture? {#roofline}
source: lecture_02.py:L471-L481 · video 54:45-57:04

A **roofline plot** puts arithmetic intensity on the x-axis and achieved FLOP/s on the y-axis, usually both on log scales.
- Each vertical slice is one computation, with its own intensity.
- Each piecewise-linear line is one piece of hardware (an H100, a B200, and so on).
- Left of the kink, the line rises: achievable FLOP/s = intensity × bandwidth, so the slope is set by memory bandwidth. You are memory-bound.
- Right of the kink, the line is flat at the peak FLOP/s. You are compute-bound, and "obviously you can't exceed the peak FLOPs".
- The kink sits exactly at the accelerator intensity, where the two limits meet.

::figure https://jax-ml.github.io/scaling-book/assets/img/roofline-improved-1400.webp | the rising part is the memory limit, the flat part the compute limit; the kink is the accelerator intensity

The figure is borrowed from the [jax-ml scaling book](https://jax-ml.github.io/scaling-book/roofline/); the lecture reads it rather than plotting its own five operations. The widget in the previous section plots those five on an H100's roofline.

The plot ties back to MFU. An operation's MFU can be no better than the roof allows:

$$ \text{MFU} \le \min\!\left(1,\ \frac{\text{arithmetic intensity}}{\text{accelerator intensity}}\right) $$

::predict roofline

An operation with intensity 30 on an H100 (kink 295) is capped at an MFU of about 0.10 no matter how good the kernel is. Past the kink the bound saturates at 1, so extra intensity buys nothing. Doubling the memory bandwidth halves the accelerator intensity and moves the kink left, from 295 to about 148, so operations in between turn from memory-bound to compute-bound.

::widget fixture:lecture_02--roofline | double the memory bandwidth and the kink moves left; points between the two kinks change from memory-bound to compute-bound
::note deferred 55:36 | A student asked why accelerators have so much more compute than memory bandwidth, so that they sit idle waiting. The professor deferred it to the GPU lecture: "if you have an answer, you should tell Jensen" (NVIDIA's CEO).
::kp roofline

## What does training add on top of a forward pass? {#network}
source: lecture_02.py:L484-L499, L559-L599 · video 57:04-1:00:15

So far: tensors, operations on them, and how memory and compute interact. Now the lecture asks what it takes to *train*, using one running example for the rest of the lecture.

### The running example: a deep network

The network has $L$ layers. The input is a $B \times D$ batch. Each layer multiplies by a $D \times D$ weight matrix, producing a pre-activation, then applies ReLU elementwise, producing that layer's activation. Inputs, activations and outputs all have dimension $D$. The parameter count is $D^2 L$; with $D = 8$ and $L = 3$ that is 192.

::figure official/lectures/images/deep-network.png | x, then for each layer a matmul (pre-activation g) and a ReLU (activation h); this figure returns for the memory tricks
::code lecture_02.py:L577-L599 | a Block is one weight matrix and a ReLU; DeepNetwork applies L blocks in sequence

### Gradients in PyTorch

The *forward pass* computes the loss from the inputs; the *backward pass* computes the gradient of the loss with respect to every tensor marked `requires_grad=True`. The lecture's toy is a one-weight-vector regression: x = (1, 2, 3), w = (1, 1, 1), prediction $x \cdot w = 6$, and loss $\tfrac{1}{2}(6 - 5)^2$. `loss.backward()` fills `w.grad` with $(6 - 5)\,x = (1, 2, 3)$.

::code lecture_02.py:L492-L499 | forward, then loss.backward() sets w.grad; the assert checks it equals (1, 2, 3)

The professor treated this as familiar mechanics. The question that matters for accounting is how much compute `loss.backward()` costs.

## What does the backward pass cost? Where 6ND comes from {#backward}
source: lecture_02.py:L502-L556 · video 1:00:15-1:06:50

To count backward FLOPs, simplify the network to two linear layers with no ReLU (elementwise work is negligible anyway): $x$ is $B \times D$ with $B = 1024$, $D = 256$, and $h_1 = x\,w_1$, $h_2 = h_1 w_2$, with both weight matrices $D \times D$. The loss is an arbitrary scalar of $h_2$. The code writes each matmul as an einsum, for a reason that will become clear.

::code lecture_02.py:L516-L523 | two einsum matmuls, a dummy loss, retain_grad so the intermediate gradients can be checked

### Zoom in on one layer

Look at the second layer, $h_2 = h_1 w_2$. Its forward cost is one $B \times D$ by $D \times D$ matmul: $2BD^2$ FLOPs, about 134 million here. (The professor briefly said "bf16, that's 2 bytes" and corrected himself: the 2 is one multiply plus one add, not bytes.)

In the backward pass, backpropagation hands this layer $\partial L / \partial h_2$, the gradient of the loss with respect to its output, a $B \times D$ matrix. From it the layer must compute **two** things:
- $\partial L / \partial w_2$, the gradient for its own parameters, which the optimizer needs;
- $\partial L / \partial h_1$, the gradient with respect to its input, the "backward message" that the earlier layers need.

Before reading on, guess the cost.

::predict backward-2x-forward

Written in einsum, each is one matmul:

::code lecture_02.py:L537-L543 | h1.grad sums over out; w2.grad sums over batch; each is checked against autograd with allclose

`h1_grad` contracts $\partial L/\partial h_2$ (batch, out) with $w_2$ (in, out) over **out**. `w2_grad` contracts $\partial L/\partial h_2$ (batch, out) with $h_1$ (batch, in) over **batch**. The `allclose` assertions confirm both match what `loss.backward()` computed, so the hand-written versions really are the backward pass.

::widget fixture:backward-two-contractions | the same upstream gradient is contracted twice, once against w2 (summing out) and once against h1 (summing batch); both have the same B·D·D size

Each involves the same three dimensions $B$, $D$, $D$. And "the number of FLOPs is essentially the product of all the dimensions. It doesn't matter which ones you're batching or not": enumerate every index triple, multiply, accumulate. Only the axis you sum over changes. So each backward matmul costs $2BD^2$, and

$$ \text{backward} = 2BD^2 + 2BD^2 = 4BD^2 = 2 \times \text{forward} $$

::worked backward-2x-forward
::video 1:04:40-1:05:27 | each backward matmul has the same three dimensions, so the same FLOPs as the forward one
::note why 1:03:16 | Why einsum here: the professor always forgets which factor gets the transpose in the matrix-calculus version. With named axes, the scalar case tells you the shape (h1.grad is h2.grad times w2), and the names tell you how to index.

### All layers: 6ND

The same holds for every weight matrix in the network. Per data point, every parameter costs 2 FLOPs forward (one multiply, one add) and 4 FLOPs backward (two matmuls of forward size). Summed over the network, with $N$ parameters and, as in the opening question, $D$ data points (the $D$ of 6ND, not the hidden size used above):

$$ \underbrace{2ND}_{\text{forward}} + \underbrace{4ND}_{\text{backward}} = 6ND $$

This is where the 6ND of the opening question comes from, "just by counting forward and backward". The optimizer update is only a few FLOPs per parameter per step, negligible next to work proportional to $N \times D$.

::worked six-nd
::widget fixture:lecture_02--backward-flops | each layer adds one forward block and two backward blocks of the same size; ticking "layer 1's input is data" removes exactly one block

One small overcount: the first layer's input is data, not a parameter's output, so nobody needs $\partial L / \partial x$. A careful count for the two-layer net is $10BD^2$, against 6ND's $12BD^2$.

6ND is exact for multilayer perceptrons and a good approximation for Transformers **at short context lengths**. The reason for the caveat was said aloud: attention compares every position with every other, adding FLOPs that grow with context length squared and are not proportional to the parameter count. At long context, 6ND undercounts.

::video 1:06:28-1:06:47 | where 6ND stops being accurate: context length squared
::note deferred 1:11:59 | The FLOP and memory accounting for an actual Transformer is "more complicated, but the same idea", and is left to Assignment 1. The lecture cites a [blog post on Transformer FLOPs](https://www.adamcasson.com/posts/transformer-flops).
::kp backward-2x-forward
::kp six-nd

## What does a training step store? The optimizer and the memory ledger {#optimizer}
source: lecture_02.py:L602-L715 · video 1:06:50-1:12:15

The last piece of training is the optimizer, which turns gradients into parameter updates.

### AdaGrad, as a stand-in for Adam

The lecture lists a family tree in four lines:
- **momentum** = SGD plus an exponential average of the gradient (the first moment);
- **AdaGrad** = SGD plus scaling by the accumulated squared gradient (the second moment);
- **RMSProp** = AdaGrad with an exponential average of the squared gradient instead of a sum;
- **Adam** = RMSProp plus momentum.

The code implements AdaGrad (2011), which the professor placed "somewhere in between SGD and Adam". For each parameter it keeps one extra tensor of the same shape, `g2`, the running sum of squared gradients. Each step adds the current squared gradient to `g2`, then moves the parameter by the learning rate times the gradient divided by $\sqrt{g_2}$. Parameters that have seen large gradients take smaller steps.

::code lecture_02.py:L664-L680 | optimizer state lives in self.state[p]; g2 accumulates squared gradients; the update divides by sqrt(g2)
::note why 1:07:01 | Why AdaGrad and not Adam: "so that I'm not just giving you what's in assignment 1." Students implement Adam in Assignment 1. The optimizer details beyond the four-line list were skipped for time.

The point for accounting is that `g2` is **optimizer state**: a full parameter-shaped tensor that lives for the whole of training. After `optimizer.step()`, `optimizer.zero_grad(set_to_none=True)` frees the gradients until the next backward.

### The memory ledger

For the toy network ($B = 2$, $D = 4$, $L = 3$, so $N = D^2 L = 48$ parameters), the lecture totals four terms:
- **parameters**: $2N$ bytes (bf16), 96 bytes;
- **gradients**: $2N$ bytes (bf16), one per parameter, 96 bytes;
- **optimizer state**: $4N$ bytes for AdaGrad's `g2` (fp32), 192 bytes;
- **activations**: $2 \cdot B \cdot D \cdot L$ bytes (bf16, one $B \times D$ activation per layer), 48 bytes.

The total is 432 bytes. Only the activation term depends on the batch size; the other three scale with the parameter count. The optimizer state is fp32 for the stability reason from the precision section: it accumulates squares over many steps. AdaGrad stores one moment, 4 bytes per parameter; Adam stores two (first and second moments), 8 bytes per parameter. That 8 is the "4 + 4" of the opening AdamW question.

::code lecture_02.py:L635-L646 | four terms; optimizer state is 4 bytes per parameter for AdaGrad
::widget fixture:memory-ledger | only the activation bar responds to batch size; switch AdaGrad to Adam and the optimizer bar doubles
::video 1:10:24-1:10:56 | fp32 for optimizer state, and 4 bytes per parameter for AdaGrad against 8 for Adam
::note slip 1:10:04 | In class the optimizer-state line was written as a multiple of the parameter *memory*, and the professor flagged it as a typo: it should be a multiple of the parameter *count*. The current file already reads `4 * num_parameters`.

The step's compute is the 6ND rule with $B$ data points: $6 \times 2 \times 48 = 576$ FLOPs. For a Transformer the accounting is more involved but the idea is the same; the lecture cites a [blog post on Transformer training memory](https://erees.dev/transformer-memory/), and Assignment 1 asks you to do it.

::kp optimizer-state-memory

### Memory has two costs

Optimizer state is a large share of the memory: with Adam, 8 of the 12 bytes per parameter. Does that make it a big cost? The professor separated two things memory does.
- **Capacity**: the tensor has to fit in HBM.
- **Traffic**: the tensor has to be shipped to the compute units, and that takes time.

Optimizer state costs mainly capacity. It is read and written once per step, by a cheap elementwise update, whereas forward and backward work grows with the number of tokens in the step. So it limits the largest model that fits but is "not really the bottleneck for compute".

::predict optimizer-state-capacity-not-speed
::video 1:10:56-1:11:42 | "memory serves two purposes", and which one optimizer state uses

The rule needs each step to process many tokens. For a 1B-parameter Adam model, the update moves about 22 bytes per parameter, about 6.6 ms at 3.35 TB/s. With 1M tokens per step, forward and backward take $6 \times 10^9 \times 10^6 = 6 \times 10^{15}$ FLOPs, about 12 s at an achieved $5 \times 10^{14}$ FLOP/s, so the update is negligible. With only 1,000 tokens per step it would be about a third of the step.

::kp optimizer-state-capacity-not-speed

### The training loop

The lecture's `train_loop()` puts the pieces together: get a batch, forward to a loss, `loss.backward()`, `optimizer.step()`, `optimizer.zero_grad()`, repeat.

::code lecture_02.py:L702-L715 | the five lines every training loop repeats
::note skip 1:12:04 | The professor skipped the training loop ("this is just a general review"); he had started to skip it once at 1:09, then went back to the memory ledger first.

## How do you fit a bigger batch? Gradient accumulation {#accumulation}
source: lecture_02.py:L718-L730 · video 1:12:10-1:13:20

Memory affects both what fits and, through traffic, how fast it runs, so you generally want to reduce it. The lecture closes with the two standard tricks, both aimed at the one term that grows with batch size: activations.

Large batches improve training stability. But activation memory is $2 \cdot B \cdot D \cdot L$ bytes and grows with $B$, so a large enough batch runs out of memory. For $B = 64$, $D = 1024$, $L = 16$ the lecture's formula gives 2 MiB, and it grows linearly from there.

**Gradient accumulation** splits the batch into micro-batches:
- run forward and backward on one micro-batch at a time;
- **do not zero the gradients** between micro-batches, so each backward adds into `.grad`;
- every batch_size / micro_batch_size micro-steps, take one optimizer step and then zero the gradients.

The sum of the micro-batch gradients equals the full-batch gradient (up to a constant factor if the loss is averaged per micro-batch), so the update is the same. Only one micro-batch's activations are alive at a time, so with micro-batches of 16 the activation memory falls from 2 MiB to 0.5 MiB. The gradient buffer does not shrink: it is parameter-shaped and simply keeps accumulating. And the FLOPs are unchanged, since every example is still processed once.

::code lecture_02.py:L721-L730 | the same activation formula with B replaced by the micro-batch size
::predict gradient-accumulation
::animation fixture:lecture_02--grad-accum-timeline | four micro-step humps, each a quarter of the full-batch peak, cover the same 64 examples while the fixed-size gradient buffer fills until the single optimizer step
::widget fixture:memory-ledger | set the micro-batch below the batch: the activation bar tracks the micro-batch while FLOPs per update stay fixed

Gradient accumulation shrinks only the activation term. If optimizer state is what does not fit, it does not help.

::note slip 1:13:12 | The professor called accumulation "a very simple code change which allows you to save on compute". It saves activation memory; total FLOPs are unchanged. His own closing summary says these are "ways to reduce the memory".
::note deferred 1:12:34 | Bigger batches help stability only "up to a critical batch size", a topic for a later lecture.
::kp gradient-accumulation

## Can you store fewer activations? Activation checkpointing {#checkpointing}
source: lecture_02.py:L733-L787 · video 1:13:20-1:16:16

Why must training store activations at all? Each layer's backward needs that layer's input (recall that $\partial L/\partial w_2$ was computed from $h_1$). So the forward pass keeps every layer's activations until the backward pass reaches it. Inference computes no gradients, so it only needs the current layer's activation; this cost is specific to training.

::figure official/lectures/images/deep-network.png | every g (pre-ReLU) and h (post-ReLU) in this chain is kept for the backward pass by default

For $B = 64$, $D = 1024$, $L = 16$ the lecture's formula $2 \cdot B \cdot D \cdot L$ gives 2 MiB. The code then measures the actual peak memory of a forward plus backward on a GPU; on the laptop that measurement could not run.

**Activation checkpointing** (also called gradient checkpointing or rematerialization) keeps activations only at a subset of layers, the checkpoints, during the forward pass. During the backward pass it recomputes the missing activations from the nearest earlier checkpoint. It trades compute for memory, a general systems trick: if you want to save memory, recompute things.

In PyTorch it is one wrapper: `torch.utils.checkpoint.checkpoint(layer, x)` runs a block without keeping its intermediate results, only what is needed to rerun it.

::code lecture_02.py:L782-L787 | the same network with each block wrapped in checkpoint
::code lecture_02.py:L757-L758 | by default x, g1, h1, g2, h2, … are kept; checkpointing each block keeps only x, h1, h2, …

The professor gave the number the text leaves out. By default each block keeps both its pre-ReLU tensor g and its post-ReLU tensor h. Checkpointing each linear+ReLU block drops the g's, which saves "basically half" of the activation memory. When backward needs $g_3$, it recomputes it from $h_2$ with one matmul.

::video 1:15:12-1:15:22 | checkpointing each block halves activation memory by not storing the pre-ReLU tensor

### How often to checkpoint

The lecture pushes the idea further for deep networks, with three schedules:
- **Store every layer**: $O(L)$ activation memory and no recomputation. This is the default.
- **Store nothing**: $O(1)$ memory, but the backward pass through each layer must rebuild that layer's input from the network input, so compute grows like $O(L^2)$.
- **Keep a checkpoint every $\sqrt{L}$ layers**: $O(\sqrt{L})$ memory and $O(L)$ recomputation.

::code lecture_02.py:L770-L773 | the three schedules and their costs

Why $\sqrt{L}$? Checkpoint every $k$ layers. During backward you hold all $L/k$ checkpoints, plus the up to $k$ layers recomputed inside the one segment currently being refilled. Peak memory is about $L/k + k$, smallest when $L/k = k$, that is $k = \sqrt{L}$.

::predict activation-checkpointing
::animation fixture:lecture_02--checkpoint-sweep | backward refills one segment at a time from its own checkpoint: resident layers peak at checkpoints + one segment (4 + 4 = 8 of 16), half of store-all
::worked activation-checkpointing
::widget fixture:lecture_02--checkpoint-tradeoff | peak memory L/k + k is U-shaped in k with its minimum at √L while recomputation stays one forward pass; "store nothing" makes recomputation grow like L²

The recomputation is $O(L)$ because each segment is recomputed once, from its own checkpoint: about one extra forward pass in total. One forward pass is $2ND$ of the $6ND$ budget, so checkpointing costs about a third more FLOPs. That is the price of the memory saving.

::note slip 1:15:57 | Aloud, the professor said the √L schedule's "recomputation overhead is also square root of L. So that's balanced." The written line (L773) says O(L) recomputation, and that is the right total: √L is the length of each recomputed segment, and there are √L segments.
::kp activation-checkpointing

## What should you carry away? {#summary}
source: lecture_02.py:L62-L68 · video 1:16:16-1:17:20

The lecture's own summary, with the numbers that go with each line:
- **Everything is operations on tensors**: parameters, gradients, activations, optimizer state, data. Memory is elements × bytes per element: 4 for fp32, 2 for bf16, 1 for fp8.
- **einops** names axes; axes missing from the output are summed. It is notation, not a speed-up.
- **6 × (data points) × (parameters) FLOPs per training step**: 2 forward, 4 backward, because each layer computes two gradients of forward size.
- **Arithmetic intensity and roofline analysis** tell you whether a computation is compute-bound or memory-bound: compare FLOPs per byte with the accelerator's, about 295 for an H100 in bf16, and MFU can be no higher than $\min(1, \text{intensity}/295)$.
- **Matrix multiplications are compute-bound**, with intensity about $n/3$; elementwise operations, dot products and matrix-vector products are memory-bound.
- **Gradient accumulation and activation checkpointing** reduce activation memory, which lets you use bigger batches: the first by processing micro-batches, the second by recomputing.

With these you can redo the two opening questions yourself: $6ND$ over (peak × MFU × GPUs) for days, and memory over 12 bytes per parameter (an upper bound that ignores activations) for model size.

::note spoken 1:16:41 | Reading the third line aloud, the professor corrected it: for one training step the "data points" are the batch, not the whole dataset. Over a whole run they are all training tokens, which is how the opening question used 6ND.
::note deferred 1:17:16 | Next lecture: architectures and hyperparameters, where these counts get applied to a real Transformer. See [L3](#/read/lecture_03).
