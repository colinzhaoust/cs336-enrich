---
title: L6 · Benchmarking, profiling and Triton kernels, read through
minutes: 45
---
This lecture turns the GPU picture from [L5](#/read/lecture_05) into code. First you measure (benchmark and profile) to find where the time goes; then you write your own GPU kernels in Triton, in four steps of increasing difficulty: an elementwise GeLU, a row-wise softmax, a row sum whose row is too long for one block, and a tiled matrix multiply with a fused ReLU. After it you can time GPU code correctly, read a profiler table, explain why a fused kernel beats a chain of PyTorch ops, and write and read a Triton kernel: its launch grid, its offsets and masks, and its tile loop.

## What is this lecture for? {#goal}
source: lecture_06.py:L13-L36 · video 0:05-0:30

Last lecture gave a high-level overview of GPUs and of what makes them fast or slow. This one is its continuation, "where we're going to dive more deeply into the code". The lecture is an executable Python file: its `text()` calls are the slides and the code between them runs, here on the professor's GPU, a B200 (NVIDIA's Blackwell generation).

`main()` runs four stages, and this read-through follows them:
- **a recap of the GPU**: the hardware, the programming model, and the hardware details that decide performance;
- **benchmarking and profiling**: how long code takes and where the time goes;
- **GeLU three ways**: the measuring tools applied to one activation function, which reveals kernel fusion;
- **Triton kernels**: GeLU (elementwise), softmax (a reduction whose row fits in a block), row sum (a reduction whose row does not), and matmul + ReLU (tiling).

::code lecture_06.py:L17-L26 | the lecture's order, with the professor's own one-line labels for each Triton example

A word that recurs everywhere: a **kernel** is one function that runs on the GPU. Every PyTorch operation you call (an add, a matmul, a tanh) launches one or more kernels. Writing kernels yourself means deciding how one computation is spread over the GPU's many threads.

::note aside | Every timing and every profiler table in the lecture is a measurement from the professor's machine. They exist only in his trace, and the professor called the GeLU timings "not terribly optimized". This read-through gives their direction or ratio, never a millisecond value.

## What does the GPU look like, and how do you program it? {#gpu-review}
source: lecture_06.py:L39-L73 · video 0:30-7:04

### The hardware

A GPU is a chip of **streaming multiprocessors (SMs)** next to a large off-chip memory, the **HBM** (high-bandwidth memory). Each SM has its own **registers** and a block of fast on-chip memory that serves both as an **L1 cache** and as **shared memory**. These are the same physical memory; the difference is that you control shared memory, while L1 is managed by the hardware. An **L2 cache** is shared by the whole chip.

::figure official/lectures/images/gpu-hardware.png | each SM carries its own registers and L1/shared memory; L2 is chip-wide; HBM sits off-chip behind one link

The lecture's table, across three NVIDIA generations (A100 / H100 / B200):
- **SMs**: 108 / 132 / 148.
- **registers per SM**: 256 KB on all three (65,536 four-byte registers).
- **L1 + shared memory per SM**: 192 KB / 256 KB / 256 KB.
- **L2**: 40 MB / 50 MB / 96–126 MB.
- **HBM**: 80 GB / 80 GB / 192 GB.
- **bandwidth**, from fastest to slowest level: registers ~116 / ~401 / ~447 TB/s; L1 and shared memory ~19 / ~33 / ~19 TB/s; L2 ~5–8 / ~12 / ~9 TB/s; HBM 2 / 3.35 / 8 TB/s.

The professor's reading of the table: the SM count (between 100 and 200) and the per-SM memories have barely changed across generations; HBM is the number "actually going up quite a bit". Size and speed are inversely related. Registers and shared memory are on the SM, "local and fast, but small"; HBM is "slow and far but big". On an H100, registers are about 120 times faster than HBM and shared memory about 10 times. "This is the main hierarchy you should have in your head."

::note skip 4:22 | Two newer pieces are named and set aside: thread block clusters on H100 and B200, which let blocks share memory across SMs, and the B200's tensor memory (TMEM) for tensor cores, which sits between registers and shared memory and is invisible to the programmer.

### The programming model

The software side has three levels:
- a **thread** runs the kernel's code on a small part of the data;
- a **thread block** (also called a CTA, cooperative thread array) is a group of threads;
- the **grid** is the collection of thread blocks.

"When you launch a kernel, you're basically launching a grid of threads and thread blocks to all parallel simultaneously do some computation."

::figure https://docs.nvidia.com/cuda/parallel-thread-execution/_images/grid-with-CTAs.png | a grid of thread blocks (CTAs), each a grid of threads

Why the middle level? For an elementwise operation such as GeLU, threads alone are natural: thread $i$ computes $f(x_i)$. But softmax and matrix multiplication need threads to **communicate**: a softmax needs a row's maximum and sum, which no single thread owns. The professor was careful about the reason. Threads *could* communicate by writing to HBM and reading each other's results, "if you were willing to pay the cost". HBM is too slow for that. So threads communicate through **shared memory**, which is local to one SM, and a thread block is exactly "a collection of threads that access the same shared memory". Consequently a block is scheduled on a single SM. The typical block reads data from HBM, processes it with the threads cooperating through shared memory, and writes the result back.

::code lecture_06.py:L66-L73 | blocks exist because threads must communicate, and shared memory is the fast place to do it

The mapping to remember: **grid ↔ HBM, block ↔ shared memory, thread ↔ registers.** Triton, the language this lecture writes kernels in, is built around the middle level: "In Triton, think natively in terms of thread blocks." Tiling, which closes the lecture, "is the whole game here".

::note aside | This part is the professor's recap of [L5](#/read/lecture_05), where the execution hierarchy and the memory hierarchy were taught on slides; here it fixes the vocabulary the code will use.
::kp lecture_05:execution-hierarchy
::kp lecture_05:gpu-memory-hierarchy

## Why does the same correct code run fast or slow? Five hardware details {#hardware-details}
source: lecture_06.py:L75-L141 · video 7:04-21:44

The programming model is clean: threads, blocks, a grid. If all you care about is **correctness**, that is all you need to know, and writing the per-block computation is "like writing Python". But **performance** is very sensitive to the hardware underneath, "and the whole reason we're talking about GPUs and kernels is that you're trying to squeeze out performance". The professor went through five details, partly a review of L5, to give the flavor.

::code lecture_06.py:L76-L78 | the model is enough for correctness; speed needs the hardware

### Warps and control divergence

Inside a block, threads are grouped into **warps** of 32. A block of 64 threads is two warps. All 32 threads of a warp execute the **same instruction in lockstep**. If they need different instructions, say an `if A else B` where some threads take A and others B, the warp runs the A threads while the others wait, then the B threads: the branches are serialized. That is **control divergence**, and it is why branching inside a kernel is best avoided.

::code lecture_06.py:L82-L90 | 32 threads per warp, lockstep, and the A-then-B picture of divergence

Warps also hide latency. An SM keeps several warps resident and can switch between them **at zero cost**, unlike a CPU switching threads. A read from HBM can take "100 cycles or something"; instead of waiting, the SM switches to a warp that has work ready.

### Register-limited occupancy

Each thread may use up to 255 registers, and an SM has a fixed register file (65,536 on these GPUs) and a fixed number of warp slots (64). The more registers each thread uses, the fewer threads fit. **Occupancy** is the fraction of warp slots in use. The lecture computes one case:
- a block of 128 threads, each using 160 registers, needs $128 \times 160 = 20{,}480$ registers;
- $\lfloor 65{,}536 / 20{,}480 \rfloor = 3$ blocks fit on one SM, limited by registers;
- 3 blocks of 4 warps each are 12 warps, out of 64 slots: occupancy $12/64 = 0.1875$, about 18%.

::code lecture_06.py:L99-L112 | registers per block, blocks per SM, warps, occupancy
::widget fixture:lecture_05--warps-occupancy | set 128 threads and 160 registers per thread: 3 blocks fit, 12 of 64 warp slots fill; lower the registers and occupancy rises until warp slots, not registers, become the limit

Low occupancy is not automatically bad. Fewer warps means fewer to switch to while one waits on memory, but if each thread does more work, the trade can pay. The example is **thread coarsening**: instead of one thread per element, each thread handles several elements, "maybe eight". This comes back when we read the compiled GeLU kernel.

::note slip | The lecture text introducing the example (L97) says the block has 64 threads; the code just below uses 128, and the professor said 128 aloud (12:33). The numbers above use 128. With 64 threads the block would need 10,240 registers, 6 blocks would fit, and occupancy would be 12/64 all the same.

### Bank conflicts (shared memory)

Shared memory is divided into **32 banks**, each 4 bytes wide; consecutive 4-byte words go to consecutive banks. Each cycle, a bank can serve one thread (unless several threads want the very same word). If several threads of a warp hit the same bank at different addresses, the accesses are serialized: a **bank conflict**. The worst case is a matrix whose rows span exactly the 32 banks, read down a column by 32 threads: every element of the column is in the same bank, a 32-way conflict, and the threads "just wait in line".

You cannot always read rows instead: a matmul $A B$ reads rows of $A$ and columns of $B$, and sometimes transposes. The fix is **swizzling**, rearranging where data is placed in shared memory (for example by XOR-ing the row and column index) so a column no longer lands in one bank.

::code lecture_06.py:L114-L124 | 32 banks of 4 bytes; a column read is a 32-way conflict; swizzling rearranges the layout
::note skip 16:24 | How swizzling works is not explained ("which I won't get into"). The profiler can show you bank conflicts and occupancy.

### Memory coalescing (HBM)

When the 32 threads of a warp read HBM, their accesses are combined into transactions of **128 bytes** (cache lines). The best case, **full coalescing**, is 32 threads reading 32 consecutive 4-byte values: $32 \times 4 = 128$ bytes, one transaction. Reading down a column instead fetches a whole cache line for each thread and uses 4 bytes of it. It sounds like bank conflicts but is a different constraint: bank conflicts are about shared memory, coalescing about HBM.

::code lecture_06.py:L126-L130 | a warp's 32 four-byte reads form one 128-byte transaction when they are consecutive
::widget fixture:lecture_05--warp-access | stride 1 gives one 128-byte transaction per warp; each doubling of the stride doubles the transactions

### Block occupancy and wave quantization

Logically you can launch as many blocks as you like; physically there are only so many SMs, and blocks run in **waves**. A B200 has 148 SMs. Launch 160 blocks and the first wave runs 148; the second runs the remaining 12 while 136 SMs sit idle. Two waves of time do 160 blocks of work where 296 would fit, about 54% use. This is **wave quantization**, and the fix is to choose the block count so that it divides evenly into the SMs.

::code lecture_06.py:L132-L137 | 160 blocks on 148 SMs: a full wave, then a wave of 12
::figure https://developer-blogs.nvidia.com/wp-content/uploads/2019/06/pasted-image-0.png | blocks scheduled onto SMs in waves; the last wave is partly empty
::widget fixture:lecture_05--wave-grid | push the tile count just past the SM count: a second, nearly empty wave appears and most SMs idle through it

::note spoken 20:35 | A student asked whether two blocks could share an SM to fill the tail. Blocks "have to stay together", and a block that already uses most of its SM gains nothing from a neighbour; the remedy is to change the block size so the tail disappears.

The summary of the recap: the programming model is grid (HBM) → block (shared memory) → thread (registers), and the hardware details (warps, bank conflicts, coalescing, occupancy) determine performance. Many of them are hard to know in advance; you need the exact SM count and memory sizes, and "sometimes the scheduler does something you don't really have control over". That is the argument for the next part: measure.

::kp lecture_05:control-divergence
::kp lecture_05:supp-latency-hiding
::kp lecture_05:memory-coalescing
::kp lecture_05:tile-wave-quantization

## How long does it take? Benchmarking {#benchmarking}
source: lecture_06.py:L144-L203 · video 21:44-26:25

Before writing any kernel, the professor gives "a recipe for success":
- benchmark and profile your code;
- make changes;
- benchmark and profile again.

He teaches measurement before Triton on purpose: "you should always just measure what's going on in your code and figure out what the bottlenecks are before you start writing kernels."

**Benchmarking** measures the wall-clock time of an operation. It gives only the end-to-end time, not where it is spent (that is profiling). It is still useful for two things: comparing implementations (which is faster?) and seeing how performance scales, for example with matrix dimension. PyTorch has a tool for this, [torch.utils.benchmark](https://pytorch.org/tutorials/recipes/recipes/benchmark.html), but "because this class is language models from scratch", the lecture writes its own to make the gotchas visible.

### The benchmark function

The operation under test comes from a small wrapper: `run_operation2(dim, operation)` creates two random dim × dim matrices on the GPU and returns a zero-argument function that applies the operation to them, so creating the inputs is not timed. Then `benchmark(run)` does four things:
- **Warm up.** Run once before timing. First calls can be slow because of compilation and caches ("some things are lazily compiled"), and what matters is the steady state, since a kernel is run over and over.
- **Synchronize.** A kernel launch is **asynchronous**: Python enqueues the work and returns while the GPU is still computing. `torch.cuda.synchronize()` waits until the GPU is done, so the warmup's work does not leak into the first timed trial.
- **Time with CUDA events.** A start event and an end event are recorded on the GPU around `run()`, then the code synchronizes and reads the elapsed time between them. The events measure on the GPU's own timeline, which avoids counting CPU-side overhead.
- **Repeat and average.** Times vary, so it runs several trials (3 by default) and returns the mean.

::code lecture_06.py:L179-L203 | warmup, synchronize, CUDA events around run(), synchronize, mean of the trials

If you forget to synchronize and time with Python's clock, the stopwatch stops when the launch returns, long before the GPU finishes, and the measured time is mostly launch overhead. [L2](#/read/lecture_02) previewed exactly this trap.

::predict benchmark-recipe
::note spoken 25:24 | The mean is a simplification: "If you are being very particular, you probably want to maybe look at the whole distribution, the P95 or whatever." The 95th percentile tells you about the slow tail that a mean hides.
::kp benchmark-recipe
::kp lecture_02:gpu-timing-synchronize

### What a kernel launch costs

The lecture uses "kernel" and "launch" constantly; aloud the professor defined a launch in one sentence: launching a kernel means "launching a grid of threads and thread blocks" to compute in parallel. Two facts about launches matter for everything that follows. A launch returns immediately (that is why benchmarking must synchronize), and each launch has a roughly fixed overhead, no matter how little work the kernel does. For tiny inputs that overhead is the whole time, so eight tiny kernels take about eight times as long as one, even if they do the same total work.

::kp supp-kernel-launch

### How matmul time scales with dimension

The lecture then times a square matmul at dim = 256, 512, 1024, 2048, 4096 and 8192. A matmul of two dim × dim matrices does $2\,\text{dim}^3$ FLOPs (from [L2](#/read/lecture_02)), so you would expect time to grow cubically, 8× for every doubling of dim.

::code lecture_06.py:L172-L176 | six sizes, then the lecture's one-line conclusion

::predict matmul-time-scaling

Above the crossover it does: from 4096 to 8192 the time goes up about 8×. Below it, the time is **roughly constant**: on the professor's B200, flat "up until you get up to almost 2,000 dimensional matrices". His reason: GPUs "are built for fairly large matrix multiplications", and a small matmul cannot fill the hardware, so it finishes in about the same short time whatever its size. The fixed cost of launching and finishing a kernel adds to the same floor. Either way, below about 2048, halving dim saves almost nothing.

::widget fixture:lecture_06--matmul-time | below about dim 2048 the curve sits on the fixed-cost floor and doubling dim barely moves it; above, each doubling multiplies the time by close to 8; a 2× faster GPU only shrinks the cubic part
::note aside | The crossover near 2,000 is the professor's reading of his own B200 plot. It is not a portable number; on another GPU or dtype it moves.
::kp matmul-time-scaling

## Where does the time go? Profiling {#profiling}
source: lecture_06.py:L206-L259 · video 26:25-30:12

Benchmarking gives one number. **Profiling** shows where the time is spent, and, even if you don't care about time, "what's actually happening under the hood". With high-level code you write `a + b` and get a result; the profiler tells you which GPU kernels actually ran.

The lecture uses PyTorch's built-in profiler. Its `profile(run)` helper warms up, synchronizes, then runs the operation once inside a `torch.profiler.profile` context that records GPU (CUDA) activity, synchronizes again, and prints a table of the kernels sorted by total GPU time.

::code lecture_06.py:L244-L252 | record CUDA activity around one run, then a table sorted by cuda_time_total

### Three profiles

- **`a + b` at dim 2048.** The table shows one kernel with a long name containing `CUDA functor add`: a kernel that adds two tensors. It takes 100% of the time, since it is the only thing running.
- **`a @ b` at dim 2048.** A different long name, a CUTLASS matmul kernel with `64x64x16` in it.
- **`a @ b` at dim 128.** The same Python line runs a *different* kernel, with `32x32x16` in its name.

So which CUDA kernel runs depends on the tensor dimensions, and the long names are the real kernels. The name is a recipe for the implementation. The professor decoded his example, `cutlass3x_sm100_simt_sgemm_f32_f32_f32_f32_f32_64x64x16_...`:
- **cutlass**: NVIDIA's CUDA library for linear algebra;
- **sm100**: built for the Blackwell architecture (B200);
- **f32**: float32 throughout;
- **64x64x16**: the tile shape, which the matmul section explains.

::code lecture_06.py:L225-L234 | which kernels ran, chosen per shape, and how to read the name
::predict profiler-kernels

Two tokens were left undecoded: `simt` means the kernel runs on the ordinary cores rather than tensor cores (a tensor-core kernel says `tensorop` instead), and `sgemm` is single-precision general matrix multiply. By the same recipe, a name with `sm90` and `bf16` is a Hopper (H100) kernel in bfloat16.

::note deferred 26:55 | In the assignment you will use NVIDIA's Nsight profiler, which gives more detail; it was skipped in lecture "in the interest of time". And on practice: "I think we make you do it on the assignment, so you have no choice."
::kp profiler-kernels

## Same function, three speeds: what GeLU reveals {#gelu}
source: lecture_06.py:L262-L302, L694-L702 · video 30:12-36:51

Now apply both tools to one function. GeLU is a common activation function, usually computed with a tanh approximation that is friendlier to hardware:

$$ \text{GeLU}(x) \approx 0.5\,x\,\Bigl(1 + \tanh\bigl(\sqrt{2/\pi}\,(x + 0.044715\,x^3)\bigr)\Bigr), \qquad \sqrt{2/\pi} \approx 0.79788456 $$

The lecture puts three implementations in the race:
- **naive**: the formula typed directly as PyTorch operations;
- **built-in**: `torch.nn.functional.gelu(x, approximate="tanh")`;
- **compiled**: `torch.compile(naive_gelu)`. `torch.compile` takes any PyTorch function and returns another function that computes the same thing, generated by a compiler.

::code lecture_06.py:L694-L702 | the same tanh approximation, written by hand and called from the library

All three are checked to give the same answer on a random input (`allclose` with tolerance 1e-6). The compiled one is checked too, because "compilation shouldn't change semantics". Then each is benchmarked on a 16384 × 16384 matrix. The result: built-in fastest, compiled close behind, naive far slower. The code's verdict: "The builtin and compiled versions are significantly faster!"

::code lecture_06.py:L267-L283 | three versions, two correctness checks, three benchmarks

Same formula, same answer, same FLOPs, "wildly different performance characteristics". The profiler shows why.

::predict gelu-fusion

### What the profiler shows

- **naive_gelu**: many kernels. Each primitive in the PyTorch expression (a multiply, an add, the tanh) is its own kernel; the table lists unary and binary functor kernels and a tanh kernel. Counting the expression, that is roughly eight or nine elementwise kernels.
- **builtin_gelu**: one kernel, a GeLU kernel in CUDA. Why does it exist? "Because people use GeLU. So someone wrote a kernel for it and put it in the standard library."
- **compiled_gelu**: one kernel, and it is a **Triton** kernel. The compiler read naive_gelu's computation graph and wrote a single Triton kernel for it.

Why many kernels are slow: each kernel reads its input from HBM, brings it to the SMs, computes, and writes the result back to HBM; the next kernel reads it from HBM again. "Between kernel invocations, things have to go back to HBM." The fused kernel reads each element from HBM once and writes once, with every intermediate kept on-chip. That is **kernel fusion**.

::code lecture_06.py:L299-L302 | no fusion: many HBM round trips; fusion: one read, one write; the compiled kernel is Triton
::animation fixture:fusion-timeline | the unfused chain (drawn as five launches) writes and re-reads the tensor between kernels while compute idles; fused, one launch does one read and one write, and the arithmetic is unchanged

### Why fusion buys so much: elementwise ops are memory-bound

The lecture uses a fact from [L2](#/read/lecture_02) without restating it. GeLU does about 20 FLOPs per element against 8 bytes moved in fp32 (one read, one write), an arithmetic intensity of a few FLOPs per byte, far below the hundreds a GPU needs to be compute-bound. So its time is set by **bytes through HBM**, not FLOPs. A chain of $k$ unfused elementwise kernels moves the tensor through HBM about $k$ times; the fused kernel moves it once. The speed-up is of the order of the round trips removed, and the extra arithmetic inside a fused kernel is nearly free.

For scale: the benchmark tensor has $16384^2 \approx 2.7 \times 10^8$ fp32 values, 1.07 GB. One read plus one write is 2.1 GB, about a quarter of a millisecond at a B200's 8 TB/s. Nine unfused kernels move about nine times that. (This is napkin math from the table above, not the lecture's measurement.)

::predict supp-memory-bound-fusion
::kp lecture_02:arithmetic-intensity
::kp lecture_05:operator-fusion

::note spoken 36:03 | Asked why the Triton kernel is faster: it is not. The compiled Triton kernel "is slower than the built-in" here; last year it "was actually closer", and "it's very hardware dependent. And I think none of this is terribly optimized."
::note skip 34:24 | How torch.compile turns a computation graph into a Triton kernel is "a fascinating topic", not covered: "I'm not going to say too much about how it works."
::kp gelu-fusion
::kp supp-memory-bound-fusion

## CUDA or Triton: what do you program? {#triton}
source: lecture_06.py:L305-L314 · video 36:51-39:44

The compiled GeLU was a Triton kernel. Now write kernels yourself. There are two main languages, and they differ in what you describe.

- **CUDA** (NVIDIA): you specify what **each thread** does. Your code gets a thread ID and executes on its piece of data. This is close to what the hardware actually does and gives fine-grained control. The cost is that you manage more yourself, notably shared memory. When threads must cooperate (a softmax's row maximum, a matmul's shared tiles), they have to synchronize, and "you have to basically do that bookkeeping".
- **Triton** (OpenAI): you specify what **each thread block** does. The frame is always the same: load data into shared memory, operate on it, write back to global memory (HBM). The compiler decides how the block's threads split the work, where values live, and how the threads synchronize.

::code lecture_06.py:L308-L314 | CUDA programs threads, Triton programs blocks

Where each one pays: for purely elementwise work, "this CUDA is just fine", even simpler, because each thread just processes its element. Triton earns its place once threads must communicate. In the professor's closing words, blocks are "easier to think about" than threads "because you don't have to think about explicitly synchronizing threads or doing shared memory". Triton thinks at a level between PyTorch's whole-tensor operations and CUDA's single elements: you operate on a block-sized vector of data as if it were a small tensor.

::predict cuda-vs-triton

Triton is "generally powerful enough", especially when getting started. Its limit: if you want to exploit "every single new feature of the latest hardware, it might not give you the full flexibility".

::note spoken 1:24:33 | Asked about alternatives: every language "has an inductive bias". Triton "was built by people who train transformers", so Transformer workloads come relatively easy. At the extreme you can write PTX (the GPU assembly, below) by hand, "but I wouldn't advise that as a first step"; ThunderKittens and CuTe DSL are other options with different trade-offs, not ranked above or below.
::kp cuda-vs-triton

## Your first Triton kernel: GeLU {#triton-gelu}
source: lecture_06.py:L317-L389 · video 39:44-50:41

Every Triton kernel has two halves: a Python **host** function that prepares memory and launches the kernel, and the **kernel** itself, a Python function decorated with `@triton.jit` that the Triton compiler turns into GPU code. Take an input vector of 8192 elements.

### The host side: allocate, cut into blocks, launch

The host first allocates the output, `y = torch.empty_like(x)`. In Triton "we're not thinking functionally anymore": a kernel returns nothing, it reads and writes memory explicitly, so the output must exist before the kernel writes into it. The host also asserts that `x` is on the GPU and contiguous, because the kernel will index it by flat offset.

The vector is too big for one SM in general, so it is cut into blocks. With `BLOCK_SIZE = 1024`, `num_blocks = triton.cdiv(8192, 1024) = 8`. `cdiv` is ceiling division: if the length were not a multiple of the block size, the last partial block still needs a program. The launch syntax puts the grid in square brackets: `triton_gelu_kernel[(num_blocks,)](x, y, num_elements, BLOCK_SIZE=BLOCK_SIZE)` runs the kernel once for each of the 8 blocks. Triton calls each of these instances a **program**; one program is one thread block. Note that `x` and `y` arrive in the kernel as **pointers**, the memory addresses of the first element: "you can think about these as just integers".

::code lecture_06.py:L337-L353 | allocate y, cdiv the elements into blocks, launch a 1-D grid of num_blocks programs

### The kernel side: who am I, which elements, read, compute, write

Each program wakes up and asks "who am I?":
- `pid = tl.program_id(axis=0)` is this program's index in the grid, 0 to 7;
- `start = pid * BLOCK_SIZE` is its first element: 0 for program 0, 1024 for program 1, 2048 for program 2;
- `offsets = start + tl.arange(0, BLOCK_SIZE)` is the vector of the 1024 element indices it owns;
- `mask = offsets < num_elements` is True for real elements. Here 8192 divides evenly, so every mask is all True; in general only the last program has False entries, and they stop it from reading or writing past the end of the tensor.

::code lecture_06.py:L369-L376 | program id, start, the vector of offsets, and the mask for the ragged end

Take 10,000 elements instead. `cdiv(10000, 1024) = 10` programs. The last starts at $9 \times 1024 = 9216$ and owns offsets 9216 to 10,239; only $10000 - 9216 = 784$ of them are real, so 784 mask entries are True and 240 are False.

::predict triton-program-grid
::widget fixture:lecture_06--launch-grid | in elementwise mode, changing num_elements moves only the mask boundary inside the last program; switch the grid to n // BLOCK_SIZE and the last program disappears, leaving a tail that nothing computes

Now the body. `tl.load(x_ptr + offsets, mask=mask)` is pointer arithmetic: the address of `x` plus each offset, loaded where the mask allows. The result is a 1024-element vector, and from here "you do your normal computation" on it as if it were a small tensor. Triton has no `tl.tanh`, so the kernel builds it from the exponential, $\tanh a = (e^{2a} - 1)/(e^{2a} + 1)$. Finally `tl.store(y_ptr + offsets, y, mask=mask)` writes the block back.

::code lecture_06.py:L378-L389 | one masked load, the GeLU arithmetic with tanh built from exp, one masked store

Count the memory traffic. One program does exactly one `tl.load` and one `tl.store`. The intermediates `a`, `exp` and `tanh` are block-local values that never go to HBM. So each element crosses HBM once in and once out, the fused behaviour that `torch.compile` produced automatically. Correctness is checked the same way as before: `check_equal_1d(triton_gelu, naive_gelu)`.

::predict triton-gelu-kernel

The professor wanted one thing remembered: "all the kernels are going to look something like this". You wake up, you figure out which indices you own, you read, you do some stuff, and you write to HBM.

::note skip 42:18 | The kernel runs on the GPU, so the executable lecture could not step through it line by line ("I'm not going to be able to trace through this"); the code was shown instead.
::note spoken 48:38 | Asked where a loaded `x` actually lives: `x_ptr` is an address in HBM, and the local `x` is "generally a register or shared memory. Triton figures out what to do there." Likewise whether tensor cores are used: "you don't control that. The hardware figures out where to put things."
::note warning | The comment at L349 calls BLOCK_SIZE the "Number of threads", and the professor repeats it later (1:08:59). In Triton it is the number of *elements* one program handles; the compiler picks the thread count. The PTX in the next section shows 8 elements per thread, so this block of 1024 elements runs on 128 threads.
::kp triton-program-grid
::kp triton-gelu-kernel

## What does the GPU actually run? Reading PTX {#ptx}
source: lecture_06.py:L325-L334, L735-L740 · video 50:41-57:15

The Triton code is, in the professor's words, "in some sense a lie". The GPU does not run Python. Triton compiles the kernel to **PTX** (Parallel Thread Execution), an assembly language for NVIDIA GPUs, which is then compiled further into machine code. The lecture writes the PTX out from `kernel.asm['ptx']` and reads it.

::code lecture_06.py:L735-L740 | the compiled kernel's PTX text is available from kernel.asm and saved to a file

The crucial change of viewpoint: PTX describes what **one thread** does. The thread block has been compiled away. The same PTX is compiled once and run by every thread; each thread tells itself apart by two special registers. The professor's observations:
- `ld.global.*` and `st.global.*` are the reads from and writes to global memory (HBM), the `tl.load` and `tl.store`;
- `%ctaid.x` is the block index (what `tl.program_id` becomes) and `%tid.x` is the thread index within the block;
- `%f*` are floating-point registers and `%r*` integer registers;
- one thread processes **8 elements**: thread coarsening, chosen by the compiler because "this thread is pretty lightweight".

::code lecture_06.py:L330-L334 | the four things to look for in the PTX

::predict ptx-inspection

The 8 explains the warning above. Triton's default is 4 warps per program, $4 \times 32 = 128$ threads, and $1024 / 128 = 8$ elements per thread. The lecture states the 8, not this arithmetic.

PTX still does not fix everything. Which SM runs a block, how warps are scheduled: "A lot of those are hardware controlled, so you don't even see."

::note spoken 56:22 | The earlier question, does `tl.load` make the thread sit idle while data arrives from HBM? Yes, the load blocks that warp for some cycles, but the SM runs many warps, "so when you get to that point, it can just find another warp to run". This is the zero-cost warp switching of the hardware recap.
::note spoken 55:03 | Do people write PTX by hand? Some do, "if you really think you're better than the compiler", and on less mature accelerators you sometimes have to; on NVIDIA, "generally, yeah, you shouldn't need to do that".
::note skip 51:08 | The PTX listing itself was skimmed ("I am obviously not going to go through all of this"); the file is generated at run time and is not in the course repository. A follow-up question, why an SM has four warp schedulers, went unanswered.
::kp ptx-inspection

## A reduction that fits in a block: softmax {#softmax}
source: lecture_06.py:L392-L484 · video 57:15-1:05:23

GeLU is "the simplest form": however messy its arithmetic, every element is independent. The remaining examples climb in difficulty. Softmax aggregates over many values (a **reduction**), first in the easy case where a whole row fits in one block; then the row sum handles a row that does not fit; then matmul. "By that point, you'll have all the ingredients that you need to do the assignment and implement flash attention."

Softmax exponentiates and normalizes each row of a matrix, so that the row is positive and sums to 1:

$$ \text{softmax}(x)_j = \frac{e^{x_j - \max_k x_k}}{\sum_i e^{x_i - \max_k x_k}} $$

So `[0, 0, 0]` becomes `[1/3, 1/3, 1/3]`, and `[1, 1, -inf]` becomes `[1/2, 1/2, 0]`. It is used in attention and to produce output probabilities. Subtracting the row maximum does not change the result (it cancels between numerator and denominator), but it keeps every exponent at most 0, which is "for numerical stability": the lecture's toy row `[0, 0, 100]` would overflow fp32 if you computed $e^{100} \approx 2.7 \times 10^{43}$ directly, since fp32 tops out near $3.4 \times 10^{38}$.

### Naive softmax: count the reads and writes

The naive version is five PyTorch operations on an M × N matrix. In plain PyTorch, unless you call `torch.compile`, each one is a separate kernel that reads its input from HBM and writes its output back.

::code lecture_06.py:L419-L440 | five ops, each annotated with its reads and writes, and the total

Count them per operation:
- row max: reads $MN$, writes $M$;
- subtract the max: reads $MN + M$, writes $MN$;
- exponentiate: reads $MN$, writes $MN$;
- row sum: reads $MN$, writes $M$;
- divide: reads $MN$, writes $MN$.

Total: $5MN + M$ reads and $3MN + 2M$ writes. A fused kernel reads the matrix once and writes the result once, $MN$ and $MN$. For large $N$ the ratio is about $8MN / 2MN = 4$, the lecture's "speedup of 4x!", "in principle". The arithmetic is identical in both; only the HBM round trips differ.

::worked naive-softmax-traffic
::predict naive-softmax-traffic

On the toy input `[[5, 5, 5], [0, 0, 100]]`, the naive softmax returns `[[1/3, 1/3, 1/3], [0, 0, 1]]`: the second row's first two entries are $e^{-100}/(1 + 2e^{-100})$, about $4 \times 10^{-44}$.

::note deferred 58:55 | The naive softmax is an Assignment 1 exercise.
::note aside | The code's tally for the divide counts $MN$ reads and leaves out the $M$ denominators it also reads, so a stricter count is $5MN + 2M$ reads. For large $N$ it changes nothing.
::kp naive-softmax-traffic

### The Triton softmax: one program per row

The kernel design follows from one question: which parts of the work need each other? Each row must be normalized by its own sum, so softmax is not elementwise, but it is **row-wise**: different rows never interact. So each row gets its own program. "And the blocks don't have shared memory, so that's fine. There's no shared memory across blocks."

On the host:
- `BLOCK_SIZE = triton.next_power_of_2(N)`, so one block covers all the columns. (`tl.arange` needs a power-of-two length; the professor just rounded up "for good luck".)
- The grid is `(M,)`, one program per row.
- The host also passes the **row strides**: how many elements to skip to go from one row to the next.

::code lecture_06.py:L443-L459 | block size = next power of 2 above N, grid = number of rows, strides passed explicitly

::figure official/lectures/images/triton-softmax.png | program pid = 1 owns row 1: load the row, subtract its max, exp and sum, normalize and store, all inside one program

In the kernel, program `row_idx` finds its row at `x_ptr + row_idx * x_row_stride`, adds the column offsets, and loads with the mask `col_offsets < num_cols`. The padding lanes are filled with `other=float("-inf")`, "because that's going to be the equivalent of a 0 for the Softmax operation": $-\infty$ never wins the max, and $e^{-\infty} = 0$ adds nothing to the sum. Then the four lines of the formula, `tl.max`, subtract, `tl.exp`, `tl.sum`, divide, and one masked store. The reductions `tl.max` and `tl.sum` across the block are single calls; Triton handles the cross-thread communication.

::code lecture_06.py:L462-L484 | find the row by stride, load with -inf padding, max/exp/sum/divide, store with the same mask

::predict triton-softmax-kernel

Why not pad with 0? For `[5, 5, 5]` with $N = 3$, BLOCK_SIZE is 4 and one lane is padding. With `other=0.0` the block is `[5, 5, 5, 0]`; the max is still 5, but the padded lane contributes $e^{0-5} \approx 0.0067$ to the denominator, so each real entry becomes $1/3.0067 \approx 0.3326$ instead of $1/3$. With a row of negative values the damage is much larger, because then the padded 0 *becomes* the max.

::widget fixture:lecture_06--softmax-padding | with other = -inf the padding's share of the denominator is exactly 0; switch to other = 0.0 and the outputs fall below 1/N, badly when the row's values are negative

Once the scaffolding is done, the core "looks very much like the naive version": "if you can fit it into a block, you can just write normal PyTorch, almost." Both implementations are checked against PyTorch's softmax on a 2048 × 2048 matrix.

::note spoken 1:04:52 | Asked about a softmax over columns: since the kernel only follows pointers, the professor's hedged answer was to change the strides, stepping by the row stride along a column.
::note deferred 1:04:37 | Asked what happens when a row has more columns than a block can hold: "we'll come back to that". The next section answers it.
::kp triton-softmax-kernel

## What if the row does not fit? Row sum and baby tiling {#row-sum}
source: lecture_06.py:L487-L535 · video 1:05:23-1:11:53

In the softmax, a whole row fit in one block, so the reduction happened inside a block "(handled by Triton)". Real rows are often longer. Say a row has 4096 columns but the block size is 1024.

The strategy:
- break the row into **tiles** of BLOCK_SIZE (4 tiles here);
- the program loops over the tiles, and each lane keeps its own running sum in an accumulator vector;
- after the loop, one final reduction sums the accumulators into the row's total (Triton does it with shared memory or warp shuffles, register-to-register exchanges within a warp).

::code lecture_06.py:L492-L495 | tiles, per-thread accumulation, one final reduction

The lecture switches from softmax to **row sum**, "because it's just easier to think about". (A long-row softmax needs a running maximum that changes as tiles arrive, the *online softmax* of [L5](#/read/lecture_05), which is not shown here.)

### The kernel

Still one program per row, grid `(M,)`. Inside:
- `acc = tl.zeros([BLOCK_SIZE])`: one accumulator per lane, in fp32;
- `for start in range(0, N, BLOCK_SIZE)`: the tile loop; each iteration loads the tile `start + tl.arange(0, BLOCK_SIZE)` with mask `cols < N` and `other=0.0` (here 0 is the right padding, since it adds nothing to a sum), and does `acc += x`, an elementwise vector add;
- after the loop, `tl.sum(acc, axis=0)` collapses the vector to one number, stored at `out_ptr + row`.

::code lecture_06.py:L517-L535 | an accumulator vector, the tile loop with masked loads, one sum after the loop

### A trace, by hand

The professor traced a row of ten values, `[3, 1, 4, 1, 5, 9, 2, 6, 5, 3]`, with BLOCK_SIZE 4:
- tile 0, columns 0–3: acc = [3, 1, 4, 1];
- tile 1, columns 4–7: add [5, 9, 2, 6], acc = [8, 10, 6, 7];
- tile 2, columns 8–11: columns 10 and 11 are past the end, masked, and load 0.0; add [5, 3, 0, 0], acc = [13, 13, 6, 7];
- `tl.sum(acc)` = 39.

::figure official/lectures/images/triton-row-sum.png | one block owns row 1; its four lanes walk three tiles, the last partly masked, then one tree reduction gives 39
::animation fixture:lecture_06--blocks-vs-tiles | the same 10 values in pieces of 4 are three independent blocks in GeLU, but one block visiting three tiles in the row sum; the acc lanes go 3,1,4,1 then 8,10,6,7, and only after the last tile does tl.sum give 39

Lane $i$ accumulates columns $i$, $i + 4$, $i + 8$. Each loop iteration is a vector add; the one vector-to-scalar reduction happens once, after the loop. The lecture's own demo input, `[[1, 2, 3, 4], [5, 6, 7, 8]]` with BLOCK_SIZE 1024, has a row shorter than a block, so the loop runs once and the result is `[10, 26]`, the same as `x.sum(dim=1)`.

::predict triton-row-sum-kernel

This is "a little bit more complicated than before, because now we have a for loop within a thread", and it is needed whenever the data does not fit in a block.

::note spoken 1:10:29 | Where does `acc` live? You don't say; "this is up to the Triton compiler". The professor added that if the block size is large enough, "it has to go in shared memory". The lecture's figure labels the accumulators as registers, the usual place for a small per-lane vector; both are the compiler's choice.
::kp triton-row-sum-kernel

### Tiles are not blocks

The professor stopped to make sure of one distinction. In GeLU we also cut a long vector into pieces, "but those were blocks", each processed independently. "These are not blocks. These are tiles. The block corresponds to this whole row." The difference has a reason. GeLU's pieces never need each other. A row sum's pieces must be added together, and separate blocks share no memory in which to combine them. So the pieces become tiles that the same program visits in turn, and the grid counts **independent reductions** (rows), not pieces.

A 512 × 4096 matrix with BLOCK_SIZE 1024 therefore launches 512 programs, each looping over 4 tiles, not 2048 programs.

::predict reduction-stays-in-one-block
::widget fixture:lecture_06--launch-grid | in row-reduction mode the program count follows the number of rows and never the row length; a longer row adds tiles to each program's loop, not programs
::video 1:11:06-1:11:28 | "These are not blocks. These are tiles."
::note aside | This is the lecture's design, not a hardware law. Real kernels can split one long reduction across blocks and combine the partial sums with atomic adds or a second pass; the lecture does not go there.
::kp reduction-stays-in-one-block

## How should a matmul kernel read memory? Tiling {#tiling}
source: lecture_06.py:L538-L587 · video 1:11:53-1:18:58

Matrix multiplication is "the bread and butter of deep learning", "optimized to death". The lecture adds a twist, a matmul followed by a ReLU, "just for kicks", though it is also simply one linear layer plus its activation. The reason for the twist comes at the end.

Notation: $A$ is $M \times K$, $B$ is $K \times N$, and $C = AB$ is $M \times N$. Picture each as a 3 × 3 grid of entries, $A_1 \dots A_9$, $B_1 \dots B_9$, $C_1 \dots C_9$, row by row.

::code lecture_06.py:L546-L549 | the 3 × 3 picture of A · B = C used in the argument below

### Naive: one output at a time, everything from HBM

Fix one output $(m, n)$. For each $k$, read $A[m,k]$ and $B[k,n]$ from HBM, multiply, accumulate; finally write $C[m,n]$. It is a correct kernel. But count: every $(m, n, k)$ triple reads two numbers from HBM, so about $MKN$ reads (twice that, counting both matrices) for $2MKN$ FLOPs. The arithmetic intensity, FLOPs per value moved, is a constant, $O(1)$, "which is not good": by [L2's roofline](#/read/lecture_02), that is deep in memory-bound territory.

The waste is visible in the picture. Computing $C_4$ needs $A_4, A_5, A_6$, the second row of $A$. Computing $C_5$ needs the same three again, read from HBM a second time. Can you read them once and use them for both? Yes, with shared memory.

### Idealized: load everything into shared memory

Load all of $A$ and $B$ into shared memory once, then compute $C$. Now the reads are $MK + KN$, quadratic instead of cubic, and the writes $MN$. For square $n \times n$ matrices that is $2n^3$ FLOPs over $2n^2$ values read, an intensity of $O(n)$: the ideal from L2. The catch: $A$ and $B$ "are usually too large to fit into shared memory". An SM has 256 KB; two 4096 × 4096 fp32 matrices are 128 MB.

::code lecture_06.py:L551-L569 | naive: MKN reads, O(1) intensity; idealized: MK + KN reads, O(N), but it does not fit

### Tiling: globally naive, locally idealized

The classic answer is **tiling**. In the professor's framing, it "globally look[s] like the naive approach, but locally ... like the idealized approach".

Divide $C$ into output tiles, and give each tile to one thread block. Fix an output tile. For each pair (a row tile of $A$, a column tile of $B$) along the $K$ dimension:
- load the $A$ tile and the $B$ tile from HBM into shared memory;
- multiply the two tiles;
- accumulate into the tile's partial sum.

When the sweep along $K$ is done, write the output tile to HBM. A different tile of $C$ is computed completely separately, by another block.

::figure official/lectures/images/gemm_tiled.png | one orange tile of C: the outer loop walks the purple row tiles of A and column tiles of B, the inner loop works inside the current pair
::code lecture_06.py:L574-L582 | the tiled algorithm in six lines, and its intensity
::animation fixture:tiling-sweep | each pair of input tiles is loaded into shared memory once and reused for all T×T partial products before the next pair loads; the global-read counter moves only on tile loads

Why the intensity becomes $O(\text{tile size})$: with square tiles of side $T$, each element of $A$ that is loaded is reused by all $T$ columns of the output tile, and each element of $B$ by all $T$ rows. Total reads fall by a factor of $T$ while the FLOPs stay $2MKN$.

::worked matmul-tiling-intensity
::predict matmul-tiling-intensity

With the lecture's own sizes, $M = N = K = 1024$ and 64 × 64 output tiles, each element of $A$ is read from HBM $1024 / 64 = 16$ times, once per column tile of $C$, instead of 1024 times in the naive scheme. You "can generally not reach order n", which would need everything in shared memory, "but if your tiles are big, then that's still not too bad". Bigger tiles mean more reuse, up to what shared memory (and registers) can hold. The `64x64x16` in the profiler's kernel name is exactly such a tile shape.

### Bonus: fuse the activation

Often a matmul is followed by an elementwise activation, $\text{GeLU}(AB)$ or $\text{ReLU}(AB)$. Since the kernel holds each finished output tile on-chip before writing it, it can apply the activation right there, at no extra HBM traffic. "This is kernel fusion." That is why the example is matmul + ReLU.

::code lecture_06.py:L584-L587 | an elementwise activation on the finished tile is fusion for free
::note spoken 1:16:38 | The professor pointed out that this is the same tiling picture Tatsu showed in [L5](#/read/lecture_05), now about to become code.
::kp matmul-tiling-intensity
::kp lecture_05:tiling-shared-memory

## Writing the tiled matmul in Triton {#matmul-kernel}
source: lecture_06.py:L589-L674 · video 1:18:58-1:22:06

### First, strides

The kernel navigates matrices by pointer arithmetic, so recall how a matrix sits in memory. A tensor is multi-dimensional, but memory is a line. The **strides** say how far to jump: element (row, col) is at

$$ \text{index} = \text{row} \times \text{stride}_{\text{row}} + \text{col} \times \text{stride}_{\text{col}} $$

elements from the start. The lecture's example is the contiguous 2 × 4 tensor `[[0, 1, 2, 3], [4, 5, 6, 7]]`: its strides are (4, 1), since moving down a row skips 4 elements and moving right skips 1. Element (1, 2) is at $1 \times 4 + 2 \times 1 = 6$, and indeed holds 6. "If it were to transpose, it would be flipped": `x.T` is the same memory with strides (1, 4). Strides count elements, not bytes.

::code lecture_06.py:L591-L596 | strides (4, 1), and (row 1, col 2) maps to index 6
::predict tensor-strides
::widget fixture:lecture_06--stride-map | toggling x.T leaves the memory strip unchanged and swaps the two jumps: a step in i now moves 1 cell and a step in j a whole row of x

Because kernels receive strides as arguments instead of assuming row-major layout, the same kernel works on transposed or sliced views. The softmax kernel took a row stride for this reason; the row-sum kernel's `row * N` quietly assumed contiguous rows.

::kp tensor-strides

### The host: a 2-D grid of output tiles

The host checks that the inner dimensions match, allocates $C$, and fixes the tile sizes `BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, 32`. The grid is now two-dimensional: `(cdiv(M, BLOCK_M), cdiv(N, BLOCK_N))`, one program per 64 × 64 output tile. For the 1024 × 1024 demo that is $16 \times 16 = 256$ programs. All six strides are passed in.

::code lecture_06.py:L607-L632 | 64 × 64 output tiles, a 2-D grid, and every stride passed to the kernel

### The kernel

Program `(pid_m, pid_n)` wakes up responsible for output tile $(m, n)$. Then:
- **indices**: `indices_m` are its 64 rows of $A$ (and $C$), `indices_n` its 64 columns of $B$ (and $C$), and `indices_k` the 32 positions of the current $K$ step;
- **pointer tiles**: from the strides it builds a 64 × 32 matrix of addresses into $A$ and a 32 × 64 matrix into $B$, using broadcasting (`[:, None]` and `[None, :]`) exactly as you would build index grids in PyTorch;
- **accumulator**: `acc = tl.zeros([64, 64])` in fp32;
- **the K loop**: for each step of 32, load the $A$ tile and the $B$ tile (masked, with `other=0.0`, so ragged edges contribute zero products), do `acc += tl.dot(a, b)`, and advance both pointer tiles by `BLOCK_K` times the stride along $K$;
- **fused ReLU**: after the loop, `acc = tl.maximum(acc, 0.0)` once, on the finished tile;
- **store**: one masked write of the 64 × 64 tile to $C$.

::code lecture_06.py:L646-L674 | pointer tiles from strides, the K loop of tl.dot, ReLU on the finished tile, one masked store

It "is going to look like the row reduction": the same loop over tiles accumulating into `acc`, except it walks along a row tile of $A$ and down a column tile of $B$ at once. Inside, `tl.dot` multiplies the two small tiles: "whenever things are in shared memory, things look like PyTorch". For the demo each program runs $1024 / 32 = 32$ loop steps. The $K$ reduction stays inside one program, as the row sum did, so $K$ changes the loop length and never the grid.

::predict triton-matmul-relu-kernel
::widget fixture:lecture_06--launch-grid | in matmul mode the grid counts output tiles, cdiv(M, BLOCK_M) × cdiv(N, BLOCK_N); changing K lengthens each program's loop without adding programs

The ReLU must wait until the loop ends. ReLU of a sum is not a sum of ReLUs: if one output's partial sums along $K$ are −8 and then +5, the correct value is $\text{ReLU}(-3) = 0$, but clamping inside the loop would give $\max(-8, 0) + 5 = 5$.

::note skip 1:20:09 | The index and pointer construction was glossed over aloud: "it's straightforward", you just have to track the indices.
::note aside | The professor (and the comment at L579) says the accumulator sits in shared memory. For a 64 × 64 fp32 tile Triton normally keeps it in registers spread across the block's threads; as with the row sum, the compiler decides. Also, unlike the earlier examples, nothing in this section checks the Triton result against `naive_matmul_relu`, and the tile sizes are fixed rather than tuned.
::kp triton-matmul-relu-kernel

## What should you carry away? {#summary}
source: lecture_06.py:L28-L36 · video 1:22:06-1:26:36

The lecture's summary, with what goes with each line:
- **Know the programming model** (PyTorch, Triton, PTX) for correctness. It is what you control: grid, blocks, threads, and in Triton one block's load–compute–store.
- **Understand the hardware** for performance: SMs, warps, occupancy, bank conflicts, coalescing. Everything is finite, "a finite number of SMs, a finite number of banks", and your big matrices have to fit those constraints.
- **Benchmark** to see scaling: warm up, synchronize, time with CUDA events, repeat. Small matmuls sit on a fixed-cost floor; large ones grow like $\text{dim}^3$.
- **Profile** to see which kernels run and for how long. A chain of PyTorch ops is a chain of kernels, each a round trip to HBM.
- **Triton: think in thread blocks.** Read from HBM into shared memory, do the work (fusing whatever you can), write back to HBM once.
- **Four examples of increasing difficulty**: GeLU (elementwise: grid = pieces of the vector), softmax (a row-wise reduction that fits a block: grid = rows), row sum (a row that does not fit: grid = rows, a loop over tiles), matmul (tiling: grid = output tiles, a loop over $K$ tiles, reuse $O(T)$, fused ReLU).

::code lecture_06.py:L28-L34 | the lecture's own six-line summary

These are the ingredients of FlashAttention, which you implement in the assignment: a tiled matmul, a row-wise softmax whose rows are too long for one block (hence the online softmax), and fusion of everything in between.

::note deferred 1:24:15 | Next lecture: more than one GPU. See [L7](#/read/lecture_07).
::note spoken 1:26:06 | A last question, whether to load whole tensors at once or process pieces, was taken offline: "it depends on the nature of the computation."
