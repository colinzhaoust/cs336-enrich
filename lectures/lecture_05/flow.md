---
title: L5 · GPUs, read through
minutes: 45
---
This lecture builds a working mental model of a GPU: what it is made of, how code runs on it, and why memory, not arithmetic, is usually the limit. It then turns that model into six tricks for making GPU code fast and uses them to take FlashAttention apart. After it you can explain why a matrix multiply can get much slower when the matrix grows by one row, and why FlashAttention is several times faster while doing slightly *more* arithmetic.

## Why GPUs, and how is a GPU different from a CPU? {#why-gpus}
source: lecture_05.pdf p2-p8 · video 0:05-9:10

::slide 2 | left: matmul throughput (TFLOP/s) against matrix size, with the professor's labels "compute intensity", "tiling" and "wave quantization"; right: FlashAttention's tiling diagram and its runtime against PyTorch on GPT-2

The lecture opens with a puzzle. The left plot (from Horace He's post on matrix shapes) is the TFLOP/s an A100 achieves multiplying two N×N matrices. You might expect a smooth rise with N. Instead the points split into bands, from about 75 TF/s to about 250 TF/s, with saw-teeth and sudden drops. The hand-drawn labels name the three causes this lecture explains: **compute intensity** (the overall rise), **tiling** (the gaps between bands) and **wave quantization** (the periodic drops). We come back to the plot near the end, once every piece is in place.

The right half is the goal of Part 3. FlashAttention is a single fused kernel that computes GPT-2's attention in about 2 ms, against about 17 ms for PyTorch's chain of separate ops.

The lecture has three parts:
- **Part 1, the hardware.** What a GPU is physically and how its programming model works.
- **Part 2, six tricks.** Each one is a way of making a workload run fast on that hardware.
- **Part 3, FlashAttention.** It is built entirely from Part 2's tricks. The professor calls it "our victory lap".

::note deferred 3:09 | For practice the professor points to the exercises in the JAX "scaling book" (originally about TPUs, now also GPUs). He says the assignment "will have things that look a little bit like those exercises". His other recommended sources are Horace He's blog and the GPU Mode (formerly CUDA Mode) community.

::slide 5 | Kaplan et al.'s scaling plot: validation loss falls along a straight line in log-log as compute grows, across model sizes

Compute is, in the professor's words, "the currency by which we operate". The Kaplan plot shows loss falling predictably as training compute rises, so faster hardware, better utilization and better parallelization each drive progress on their own. In the last year or so the same is true of inference.

::slide 6 | 42 years of processor data: transistor counts keep climbing, while single-thread performance and clock frequency flatten after about 2005 ("End of Dennard Scaling")

Where did compute come from historically? Through the 1980s and 1990s it came from **Dennard scaling**. Transistors shrank, clocks got faster, and a serial CPU simply ran every instruction quicker. In the plot the green frequency curve flattens in the mid-2000s while the orange transistor count keeps rising. The professor adds the reason aloud: making transistors smaller no longer makes them switch faster, for fundamental physical reasons. Once clocks stopped speeding up, the only way to get more compute was to scale *horizontally*, with more units running instructions in parallel. That is the GPU's story.

::slide 7 | Bill Dally's plot of single-chip int8 throughput (TOPS): K20X 3.94, M40 6.84, P100 21.2, V100 125, Q8000 261, A100 1248, H100 4000; on the left, where the gains came from

The result is about 1000× in ten years. The curve is almost flat until the P100 and takes off with the V100 (2017). Most of the gain comes from **number representation** (about 16×) and **complex instructions**, the matrix instructions behind tensor cores (about 12.5×), not from process shrinks (about 2.5×). Tensor cores, sparsity and lower number formats all return later in the lecture.

::note deferred 6:03 | Scaling across many GPUs belongs to the same horizontal-scaling story, but it is left to the parallelism lectures after Percy's next lecture.

::slide 8 | left: die area of a CPU (big control unit, big cache, four ALUs) against a GPU (a grid of small ALUs with thin strips of control and cache); right: four threads on each, the GPU interleaving them with waits, the CPU running them one after another

A CPU is designed for fast serial execution of code with complex branches and control flow. So it spends most of its die on a big **control unit** (branch prediction, instruction scheduling) and **cache**, with a few ALUs (arithmetic logic units) that do the actual math. The goal is **latency**: each thread should finish as soon as possible.

A GPU makes the opposite trade. It fills the die with many small ALUs and keeps little control logic and cache per core. The goal is **throughput**, the total amount of data processed. The timeline on the right shows what that means. On the GPU, each thread T1–T4 alternates between processing (green), waiting for data (white) and ready (gold). Every thread takes a long time to finish, but while one waits another runs, so the hardware stays busy. The CPU instead finishes T1 before starting T2. The professor puts it as "even though your tasks don't finish quickly, you're getting large aggregate throughput".

::video 8:26-8:48 | the trade in one sentence: single tasks finish slowly, aggregate throughput is what the GPU optimizes

The practical consequence is that a GPU wins only when there is enough **independent** work to keep thousands of ALUs busy. A long chain where each step needs the previous result stays CPU work. Code with many unpredictable branches also runs badly on a GPU, for a reason we will see under control divergence.

::predict gpu-vs-cpu-throughput
::kp gpu-vs-cpu-throughput

## What is inside a GPU: SMs and the memory hierarchy {#anatomy}
source: lecture_05.pdf p9-p10 · video 9:10-13:20, 28:00-28:40, 1:10:00-1:10:40

::slide 9 | left: one SM, split into four quarters, each with its own warp scheduler, register file, INT32/FP32/FP64 units and a large "TENSOR CORE" block; right: the full GA100 chip, a grid of SMs around a central L2 cache

The basic unit of a GPU is the **SM (streaming multiprocessor)**. Think of it as a core: an independent compute unit that runs its own jobs, with its own small memories. An SM is not a single ALU, though. Inside it are many **SPs (streaming processors)**, the green INT32/FP32/FP64 lanes in the left picture, which execute threads in parallel. It also contains **tensor cores**, the big green blocks, which we meet properly two sections from now. The SM is the discrete unit you program against. A GPU has many of them, and each executes **blocks** (jobs, defined in the next section) independently of the others.

How many? The right picture is the full GA100 die, which has 128 SMs. The A100 product enables 108 of them. The professor's own table from the next lecture gives 108 (A100), 132 (H100) and 148 (B200).

::note slip 10:08 | The professor says "A100 will have 128 SMs". That is the slide's full GA100 die. A shipping A100 has 108 SMs enabled, which is the number he uses himself in the wave-quantization example (slide 48) and in lecture 6's table. Later (20:53) he says an "A100 might have something 132", but 132 is the H100.

::slide 10 | top left: measured A100 latencies in cycles (global 290, L2 200, L1 33, shared memory 23 load / 19 store); bottom left: a GPU board with the GPU package and memory chips; right: the GA100 die shot, SMs in green around the purple L2 partitions, HBM interfaces at the edges

The second half of the picture, and the one that dominates the rest of the lecture, is **memory**. "Modern hardware and LLM optimization is really defined by the memory", as the professor put it. From fastest and smallest to slowest and largest:
- **Registers** belong to one thread. They are the fastest storage and hold things like loop indices and the address where an array starts.
- **L1 cache and shared memory** sit *inside* each SM. In the latency table they answer in about 20–33 cycles.
- **L2 cache** is on the die but outside the SMs, shared by all of them. It takes about 200 cycles.
- **Global memory** (HBM, high-bandwidth memory) is the set of DRAM chips next to the GPU, off the compute die. It takes about 290 cycles, roughly 10× slower than L1, and it is what a spec sheet means by GPU memory. When an H200 advertises 144 GB, that is global memory.

The reason is physical distance: the closer the memory sits to the SM, the faster it is. The die shot shows the L2 in the middle of the chip and the HBM interfaces at the edges. The slide's caption gives the economics: SRAM, the on-chip memory used for L1, shared memory and L2, is about 100× more expensive than DRAM but about 8× faster.

Speed comes in two flavours, latency (how long one access takes) and bandwidth (bytes per second). The professor's table in lecture 6 gives the H100's bandwidths, which fall by more than 100× from registers to HBM:

::code lecture_06.py:L42-L54 | lecture 6's hardware table: per-SM sizes and per-level bandwidths for A100, H100, B200

On an H100, about 401 TB/s for registers, 33 TB/s for L1/shared, 12 TB/s for L2, and 3.35 TB/s for HBM. The sizes move the other way: 256 KB of registers and 256 KB of L1/shared per SM, 50 MB of L2, 80 GB of HBM. Every trick in Part 2 is a way of serving more of a kernel's reads from the top of this list.

**Why not build the whole chip from SRAM?** Students asked this twice. The answers: SRAM costs hundreds of times more per byte. It has to sit physically close to the compute, and signals are hard to propagate over long distances. And SRAM needs power the whole time to hold its value, so a giant SRAM chip is energy-hungry. Groq's design does use huge amounts of SRAM and suits some inference workloads, but most accelerators keep the hierarchy, and you have to respect it to make anything fast.

::note aside 28:31 | The size of the latency gaps is a design choice, not a law. The professor notes that the TPU's equivalent of L2 is much faster, because Google made different trade-offs on silicon.

::widget fixture:lecture_05--memory-levels | pick who shares the data: one thread keeps it in registers (401 TB/s); warps of one block meet in shared memory (33 TB/s, about 10× HBM); different blocks can only meet through L2/HBM (3.35 TB/s)

::predict gpu-memory-hierarchy
::kp gpu-memory-hierarchy

### Shared memory versus L1: same silicon, different owner
Slide 10 lists "L1 and shared memory" together. A student asked what the difference is. Both are SRAM inside the SM ("if I remember right, they're both SRAM", the professor hedged). The difference is who controls them. **L1 is a cache**: the hardware decides what recently used data to keep, and "you don't get to control the L1 cache". **Shared memory is a programmable scratchpad**: your kernel explicitly loads data into it and reads it back, and every thread of a block can address it.

That difference matters later. Tiling works only because a kernel can *guarantee* that a tile stays on the SM while all the block's threads reuse it. A cache might evict it. The flip side is that staging data in shared memory pays off only through reuse. An element read once by one thread gains nothing from being copied first.

::video 12:54-13:18 | who controls each memory: the cache "operates on its own", shared memory you put things in and out of
::widget fixture:lecture_05--memory-levels | tick "stage the tile in shared memory": global reads per element fall from k to 1, so the saving is k − 1; with a single use (k = 1) staging saves nothing
::kp shared-memory-vs-l1

## How does code run on it? Threads, warps, blocks {#execution}
source: lecture_05.pdf p11-p12 · video 13:20-17:05, 29:55-30:05

::slide 11 | a CUDA program of 4096 blocks with 256 threads each; each block is assigned to one SM and divided into 8 warps of 32 threads; each SM has 4 warp schedulers issuing to its INT32/FP32 lanes

With the hardware in place, the next question is how a program is laid out on it. Three names matter.

- **Thread.** The unit that does the work. GPU threads follow **SIMT** (single instruction, multiple threads): every thread executes *the same instructions*, each on its own inputs. Thread 7 adds element 7, thread 8 adds element 8. The professor frames SIMT as a trade between programmability and efficiency: if every thread could do something different, the hardware would be much harder to program and to keep efficient.
- **Block.** A group of threads. A block is guaranteed to run on a **single SM**, and it gets that SM's shared memory. That guarantee is what will let a block's threads cooperate on a tile in shared memory later.
- **Warp.** The scheduling unit. Threads always execute in groups of **32 consecutively numbered threads**, called a warp. The professor gives the reason aloud: scheduling 32 threads as one "decreases the overhead of a scheduler deciding which threads to run".

The slide's example: blocks of 256 threads, so each block is 256 / 32 = 8 warps, and each SM has 4 warp schedulers that pick which ready warp issues its next instruction.

::video 14:51-15:09 | why threads are scheduled in groups of 32: less scheduler overhead

A student asked whether "all threads execute the same instruction" applies to the block or the warp. It is the **warp**. Different warps of the same block can be at different points in the program, and the scheduler decides which warp runs next. One more level, not named on the slide, is the **grid**: all the blocks of one kernel launch, here the 4096 blocks. Because a block never spans SMs, a GPU runs faster on a big problem only by running *many blocks at once*, one or more per SM.

::code lecture_06.py:L83-L85 | lecture 6's recap: threads grouped into warps of 32, so a 64-thread block is 2 warps

Since warps are always 32 wide, a block size that is not a multiple of 32 wastes lanes. A 100-thread block becomes ⌈100/32⌉ = 4 warps, and the last warp has 4 live threads and 28 idle lanes.

::widget fixture:lecture_05--warps-occupancy | threads are grouped in 32-wide warps: any block size that is not a multiple of 32 leaves hollow (idle) lanes in its last warp

::note slip 29:55 | Asked "how many warps are in a block?", the professor answers that it is "hardware-dependent". In CUDA the programmer chooses the block size, and the warp count follows as block size ÷ 32 (rounded up), as in lecture 6's 64-thread example.

::predict execution-hierarchy
::kp execution-hierarchy

::slide 12 | the CUDA memory model: each thread has registers (and local memory), each block has its shared memory, and all blocks of the grid share global and constant memory, which the host (CPU) can also read and write

The memory model maps the memory levels onto those software units:
- per **thread**: registers, plus "local" memory for what does not fit in registers;
- per **block**: shared memory, for data that several threads reuse or exchange;
- per **grid**: global memory, readable and writable by every block, and constant memory, which is read-only;
- the **host** (CPU memory) can copy data in and out of global memory, for example to offload beyond the GPU's capacity.

The bold line on the slide is the rule to remember: information that crosses blocks has to be written to and read from global memory, which is slow. In the professor's words, "as soon as you go outside of your shared memory, things are going to be slow". So the game for the rest of the lecture is to arrange the work so that the reads a block needs come from its own shared memory as often as possible.

::note aside 16:36 | Constant memory is on the slide, but the professor says he has rarely seen it used.

## Side thread: are TPUs different? {#tpu}
source: lecture_05.pdf p13-p14 · video 17:05-22:45

::slide 13 | the abstract layout of one TPU "TensorCore": a scalar unit dispatching instructions, a vector unit (VPU) with vector memory, and a large matrix-multiply unit (MXU), all fed from HBM

Google's TPUs are the other branch of accelerator evolution, and they look strikingly similar. The professor calls it convergent evolution: "if you want to build an energy-efficient accelerator that does machine learning, you end up at the same place". A TPU core has:
- a lightweight control unit (the **scalar unit**, which plays the CPU's dispatching role);
- a **vector unit** (VPU) for elementwise work such as activations;
- a big **matrix-multiply unit** (MXU), which supplies almost all the chip's FLOP/s;
- slow **HBM** plus fast on-chip memory (vector memory, also called SMEM).

So the same mental model applies: a special matmul circuit, vector units, a small amount of control, slow big memory and fast small memory. The slide notes the two real differences. One is how the chips are networked to each other, which the professor calls the biggest difference and leaves to the parallelism lecture. The other is that TPUs have no warps, only blocks.

::slide 14 | the JAX scaling book's GPU↔TPU dictionary: SM ↔ TensorCore, warp scheduler ↔ VPU, CUDA core ↔ VPU ALU, SMEM ↔ VMEM, tensor core ↔ MXU, HBM ↔ HBM; and counts for an H100 against a TPU v5p: 132 SMs vs 2 TensorCores, 528 tensor cores vs 8 MXUs

Every GPU concept has a TPU counterpart, and the mapping is precise: a GPU tensor core and a TPU MXU are even built from the same kind of circuit, a **systolic array** that streams operands through a grid of multiply-adders. Watch out for one naming clash. On a TPU a "TensorCore" is a whole *processor*, the analogue of an SM. On a GPU a "tensor core" is the *matrix-multiply circuit inside* an SM.

The real difference is **granularity**. An H100 has 132 SMs with 528 tensor cores in total (4 per SM). A TPU v5p has 2 TensorCores with 8 MXUs. The two chips are sized for comparable matmul throughput, so each TPU unit is far bigger. Many small units are flexible: you can spread them over many small problems. A few huge units lock you into big matmuls, because a small operand fills only a corner of each unit. The professor gave an example from his own paper. A batch-size sweep on TPUs "stops at 64", because the matrix unit "refuses to accept anything smaller than a 64-dimensional input". Real hardware pads small operands rather than refusing them, so read "refuses" as "cannot use efficiently".

::video 21:23-22:03 | the cost of a few big matrix units: a batch sweep that cannot go below 64
::widget fixture:lecture_05--matrix-unit-fill | shrink the batch rows m below the unit size: grey padding grows and the useful fraction falls as m/U; on many 32×32 matmuls the 256 small units stay full while the 4 big ones use 1/64 of their arrays

A worked number: if a unit works on 128-wide blocks and your batch dimension is 16, then 16 of every 128 rows are real and the useful fraction is 16/128 = 0.125.

::note slip 20:53 | The professor attributes the 132 SMs and 528 matmul units to "a typical GPU like A100". They are the H100 column of the slide's table (132 × 4 = 528). An A100 has 108 SMs and 432 tensor cores.

::predict tpu-matmul-unit-granularity
::kp tpu-matmul-unit-granularity

## Why the model works, and why matmuls are special {#strengths}
source: lecture_05.pdf p15-p17 · video 22:45-25:50, 27:35-28:00

::slide 15 | three strengths: scaling by adding SMs; the SIMT model (one instruction decoder and warp scheduler driving many CUDA cores); lightweight threads, shown again as the interleaved GPU timeline from slide 8

Three properties made GPUs so successful:
- **They scale by adding SMs**, as long as memory bandwidth keeps up.
- **SIMT makes them "deceptively easy" to program.** You do not program every thread. You write one instruction stream and say "run it on all of these inputs", much like `map` in functional programming. Even hand-writing low-level PTX (NVIDIA's assembly-like language) is doable for that reason.
- **Threads are lightweight.** They can be stopped and started at any time. The professor explains what that buys aloud: when one warp is stalled, the scheduler swaps in another warp that is ready, and the swap is very quick. That raises throughput whenever some warps are waiting.

That third point is the mechanism behind slide 8's timeline, and it deserves one more step, which lecture 6 supplies. An SM keeps many warps **resident** at once and switches between them at zero cost. A global-memory read takes hundreds of cycles, as the latency table on slide 10 showed. While one warp waits for it, another computes. This is **latency hiding**, and it works only if enough warps fit on the SM. What limits that is mostly registers. The SM has a fixed register file, so the more registers each thread uses, the fewer blocks fit. The fraction of the SM's warp slots that are actually filled is called **occupancy**.

::code lecture_06.py:L100-L112 | lecture 6's occupancy example: 128 threads × 160 registers = 20,480 registers per block, so 65,536 // 20,480 = 3 blocks fit, which is 12 warps of 64, occupancy 0.1875

Low occupancy is not automatically bad. Lecture 6 notes it is fine if each thread does more work ("thread coarsening"). But a kernel with very few resident warps has nothing to switch to while it waits for HBM.

::widget fixture:lecture_05--warps-occupancy | raise registers per thread: fewer blocks fit in the 65,536-register file, so fewer of the 64 warp slots fill and fewer warps are left to switch to while one waits on memory
::video 23:24-23:51 | the scheduler swaps in another warp when one is stalled
::predict supp-latency-hiding
::kp supp-latency-hiding

::slide 16 | "Fast Matrix Multiplies using Graphics Hardware" (Larsen and McAllister): early researchers packing four numbers into a pixel's RGBA channels and tuning refresh rate and texture formats to make shaders multiply matrices

Even before GPUs had any matmul hardware, people saw that commodity graphics hardware's massive parallelism suited scientific computing. This early paper programmed graphics *shaders* to multiply matrices. The implementation notes on the slide read like hacker folklore: pack 4 numbers into one pixel's red, green, blue and alpha channels, run full screen instead of in a window, lower the monitor's refresh rate for about 10%. The professor calls it "really cool hacker work" that you no longer have to do.

::slide 17 | matmul versus non-matmul TFLOP/s across GPU generations (log scale): equal at about 10 TFLOP/s on the K80 through P100, then diverging: V100 roughly 125 vs 15, A100 roughly 300 vs 20, H100 roughly 1000 vs 60

With the V100 (Volta, 2017; the slide says "V, T series", meaning Volta and Turing), NVIDIA added **tensor cores**, circuits that do nothing but small matrix multiplies. A student checked: before the V100 there was no specialized matmul unit, only many ALUs. The plot shows what happened next. Up to the P100, a FLOP was a FLOP, about 10 TFLOP/s whatever you computed. From the V100 on, matmul FLOP/s pull away, and by the H100 the gap is more than 10×. Lecture 2's H100 figure is about 990 TFLOP/s for dense bf16 matmuls. NVIDIA's datasheet lists about 67 TFLOP/s for ordinary fp32 arithmetic.

So matmuls became "the one privileged operation in machine learning". A FLOP inside a matmul is more than 10× cheaper than a FLOP anywhere else. The professor draws the design consequence aloud: any near-future architecture that scales with compute will be built around matrix multiplies, because that is the only way to get that throughput. It also means FLOP counts can mislead. A softmax or activation that is 1% of a layer's FLOPs can take a much larger share of its time.

**Worked example.** A layer's FLOPs are 99% matmul at 990 TFLOP/s and 1% elementwise at 67 TFLOP/s. Time is FLOPs divided by rate: 0.99/990 = 0.0010 and 0.01/67 = 0.00015 (arbitrary units). So the elementwise part takes 0.00015 / 0.00115 ≈ 13% of the time, for 1% of the FLOPs.

::video 24:48-25:31 | the architectural consequence: anything that scales with compute will be built around matmuls
::predict tensor-cores-matmul-special
::kp tensor-cores-matmul-special

## Compute outruns memory: the roofline and the six tricks {#memory-wall}
source: lecture_05.pdf p18-p22 · video 25:50-32:55

::slide 18 | normalized scaling over 1996-2023: peak hardware FLOP/s 60,000× in 20 years (3.0× every 2 years), DRAM bandwidth 100× (1.6× every 2 years), interconnect bandwidth 30× (1.4× every 2 years)

This is the last hardware fact, and the one that sets the agenda for Part 2. The parts of a GPU scale at different rates. The professor reads the three curves aloud. Compute (grey) has grown about 60,000× in 20 years. Memory bandwidth (green), how fast data reaches the chip, has grown only about 100×. Interconnect bandwidth between devices (blue) has grown only about 30×. That last curve is the seed of the parallelism lectures.

In the early days, compute was not much faster than memory transfer, so nobody thought hard about memory. Today the gap is enormous, and it keeps widening: per byte of bandwidth, today's chips have about 60,000 / 100 = 600× more FLOP/s than those of 20 years ago. It is hard to keep the compute units fed with data. "That's why a lot of the optimization I'm talking about today are memory optimizations."

::video 25:52-26:43 | three curves: compute grows fast; memory bandwidth and GPU-to-GPU interconnect grow slowly

::slide 19 | recap of Part 1: GPUs are massively parallel (the same instructions applied across many workers); compute, especially matmuls, has scaled faster than memory; we have to respect the memory hierarchy

The professor is candid that this is "a flavor" of the hardware, not a full course on it. The consequence of the three lines: keep as much as possible in shared memory rather than global memory.

::note aside 28:56 | Asked whether the growing compute-memory gap will change chip design, the professor describes inference hardware. Inference is even more memory-bound than training, which has led to **prefill/decode disaggregation**: the matmul-heavy prefill runs on one kind of chip and the bandwidth-limited decode on another. Some models go further (he names a Chinese open model, apparently StepFun's Step-3) and send attention and MLP layers to different accelerators.

::slide 20 | the square-matmul throughput plot from slide 2 again: "performance on a GPU can be complex, even for something as simple as a square matmul"

Part 2 opens on the same plot: the bands, saw-teeth and drops are what it has to explain.

::slide 21 | the roofline model: throughput (GFLOP/s) against operational intensity (FLOPs/byte), log-log; each memory level gives a sloped roof (GPU registers, shared memory, main memory, CPU main memory) that meets the flat ALU-throughput ceiling; dense matmuls sit far right on the flat part, sparse matmuls on the sloped part

The tool is the **roofline model**, already met in [L2's roofline](#/read/lecture_02). A quick recap. **Arithmetic intensity** is FLOPs performed per byte moved from memory. If a kernel does few FLOPs per byte, its speed is set by bandwidth: throughput = intensity × bandwidth, the sloped part of the roof. Past a certain intensity, the compute units are saturated and throughput is flat at peak FLOP/s. The kink sits at the accelerator's own intensity, peak FLOP/s ÷ bandwidth, about 295 FLOP/byte for an H100 in bf16. Below the kink an operation is **memory-bound**; above it, **compute-bound**.

The slide's version draws one sloped roof per memory level. Data served from registers or shared memory reaches the flat ceiling at a much lower intensity than data from main memory, which is the hierarchy from slide 10 restated. Dense matmuls (diamonds) sit far right on the flat part. Sparse matmuls (circles) sit on the slope at well under 1 FLOP/byte.

The goal of Part 2 is stated in bold: **how do we avoid being memory-bound?** On the flat part "there's nothing else to do". So each trick either moves fewer bytes for the same FLOPs or serves the bytes from a faster level.

::widget fixture:lecture_05--roofline-wall | raise peak FLOP/s with bandwidth fixed: the kink slides right (10× FLOP/s with 3× bandwidth moves it 3.3×), and at only 2× FLOP/s the 1024 matmul already drops under the sloped, memory-bound roof

**Worked example.** An elementwise kernel with intensity 0.25 FLOP/byte on an H100 (kink ≈ 295) can reach at most 0.25 / 295 ≈ 0.00085 of peak, under 0.1%. No amount of extra compute helps it. Only moving fewer bytes does.

::predict memory-wall-roofline
::kp memory-wall-roofline

::slide 22 | how do we make GPUs go fast: 1. control divergence (not a memory bottleneck); 2. low precision; 3. operator fusion; 4. recomputation; 5. coalescing memory; 6. tiling

The six tricks, in the order the lecture takes them. The first is the odd one out, a consequence of SIMT rather than of memory. The other five all reduce or relocate memory traffic.

::note aside 32:31 | The numbering drifts between slides. This list counts control divergence as 1, but the trick slides that follow start their own count from low precision ("Trick 1") and end at tiling ("Trick 5, the big one"). Control divergence is the unnumbered extra.

## Control divergence: what an `if` costs inside a warp {#divergence}
source: lecture_05.pdf p23 · video 32:55-34:50

::slide 23 | code `if (threadIdx.x < 4) { A; B; } else { X; Y; } Z;` and its timeline: after "diverge", the threads taking the if-branch run A and B while the others idle, the else-threads run X and Y while the first group idles, and only Z runs on all threads together

On a CPU an `if` picks one branch, runs it and moves on. On a GPU every thread of a warp executes the *same instruction*. So when the threads of one warp disagree on a condition, the warp runs **both** branches one after the other. During each branch, the threads on the other side are **masked off**: they sit idle. In the slide's code, threads 0–3 take the if-branch and the rest the else-branch. The timeline shows the warp issuing A, B, X and Y in turn, with part of the warp idle at every step, and reconverging only at Z. This is **control divergence**.

The cost is the *sum* of the branch lengths, whatever the split. Eight threads on the slow branch do not cost 8/32 of it: the warp pays for the whole branch.

::code lecture_06.py:L87-L89 | lecture 6's version: the A lanes run while the others are masked, then the B lanes
::worked control-divergence

The professor gives the practical fix aloud. GPU code tends to replace branches with arithmetic. A ReLU, for example, is written as a multiply by a 0/1 mask rather than an `if`, because the multiply runs on all lanes at once, while an if/else issues both sides. Divergence also happens only *within* a warp. If the condition depends only on the block index, so that all 32 threads of a warp agree, only one branch is issued.

::video 34:21-34:44 | the fix: multiply by a mask instead of branching
::note aside 34:46 | The professor sets divergence aside as "a separate story from all this memory stuff, which is the main narrative". The remaining tricks are all about memory.
::predict control-divergence
::kp control-divergence

## Trick 1: low precision, fewer bits to move {#low-precision}
source: lecture_05.pdf p24-p26 · video 34:50-38:00

::slide 24 | Dally's plot again: the largest single source of the 1000× is number representation (FP32 → FP16 → int8, about 16×); "if you have fewer bits, you have fewer bits to move"

The professor calls low precision the trick where NVIDIA invests the most hardware effort, and "a pretty nontrivial part" of Dally's curve. The first reason it helps: half the bits means half the bytes to move, and moving bytes is the bottleneck.

::slide 25 | elementwise ReLU on a vector of size n; fp32: 1 read and 1 write of 4 bytes for 1 comparison FLOP, 8 bytes/FLOP; fp16: the same at 2 bytes each, 4 bytes/FLOP

The example is deliberately "dumb", in the professor's word. The work does not change; the traffic halves, and since ReLU is memory-bound, so does its time.

::worked low-precision-intensity

::note slip 36:11 | The slide writes intensity as bytes per FLOP. The professor catches it himself: "Sorry, this is the inverse of arithmetic intensity." In the roofline's FLOPs-per-byte terms, fp32 ReLU is 1/8 FLOP/byte and fp16 is 1/4, and higher is better. (The slide also says the write happens "if x < 0" but counts it every time; the 8 bytes is the worst case.)

::widget fixture:lecture_05--roofline-wall | pick ReLU and switch fp32 → bf16: bytes per FLOP fall from 8 to 4, the dot moves right along the sloped roof and the time halves, while the FLOP count and the roof stay put

::kp low-precision-intensity

::slide 26 | left: a tensor core multiplies two 16-bit inputs into a full-precision product and adds it into an FP32 accumulator; right: which operations can use 16-bit storage (matmuls, most pointwise ops), which need more precision (adding small values to large sums; reductions such as sum, softmax, normalization), and which need more range (exp, log, pow; loss functions)

The second reason is that tensor cores run low-precision matmuls faster. How a low-precision matmul works is subtler than "everything in 16 bits", and the left diagram shows it. The *inputs* are downcast to 16 bits. Each product is formed at full precision, and the partial sums are added in an **FP32 accumulator**. The output can then be emitted in FP32. The precision given up is in the operands, not in the running sum.

Why does the accumulator matter? bf16 keeps only 8 significant bits, so between 256 and 512 adjacent representable values are 2 apart. Add 1.0 to a bf16 running sum of 256 and you get 257, which rounds back to 256. A bf16 accumulator summing 4096 ones gets stuck at 256. An FP32 accumulator represents every integer up to 2²⁴ exactly, so it reaches 4096. Long dot products, thousands of terms in a big matmul, need a wide accumulator.

::widget fixture:lecture_05--bf16-accumulator | with a bf16 accumulator the running sum climbs exactly to 256 and then stalls (256 + 1 rounds back to 256); the fp32 accumulator stays on the diagonal
::video 36:34-36:57 | what is low precision and what is not: downcast the inputs, accumulate in full precision

The right half is the craft. Anyone can lower precision, the professor says; the black art, as he calls it, is deciding *which operation gets which format*. Matmul weights and activations can usually be low precision. A softmax, or an exponential, probably needs FP32, or maybe gets away with BF16. The table's logic: reductions and adding small values to large sums lose accuracy, and functions like exp blow up range. Getting low-precision training to today's state took years of "slow, incremental, empirical work" deciding which operations can be downcast and how, without breaking training stability.

::predict tensor-core-fp32-accumulate
::kp tensor-core-fp32-accumulate

## The low-precision frontier: FP8, MXFP8, MXFP4 {#fp8}
source: lecture_05.pdf p27-p29 · video 38:00-47:35

::slide 27 | left: the same number in four formats (FP16 0.395264, BF16 0.394531, FP8 E4M3 0.40625, FP8 E5M2 0.375); right: plain FP8 with one FP32 scaling factor for a whole matrix, against MXFP8 with one E8M0 scale per block; bottom: forward and backward passes casting the weights rowwise and columnwise

Once you see how expensive fast memory is, the temptation is obvious: "if only I can cut my precision by another factor of 2". **FP8** was the next step, and at 8 bits there is no longer one canonical format. **E4M3** has 4 exponent bits and 3 mantissa bits, so more precision and less range. **E5M2** has 5 and 2, so more range and less precision. The left column shows what that costs: one number stored as 0.395264 in FP16 becomes 0.40625 in E4M3 and 0.375 in E5M2.

With only 4 or 5 exponent bits, values quickly overflow or underflow, so FP8 tensors carry a **scaling factor**. Classic FP8 training uses one FP32 scale for a whole tensor. But one matrix can contain very different magnitudes; some parts of a sequence have much bigger activations than others. So **MXFP8** (microscaling, supported in hardware from Blackwell) gives each small block its own scale. Per the slide:
- one scale per 32 elements;
- the scales are themselves 8-bit **E8M0**, all exponent and no mantissa, so every scale is a power of 2;
- with that many scales, range matters less, so the elements use E4M3, with more mantissa.

The professor then paused the class with a question: what is the problem with this design? The answer is the **transpose**. The 32-element scale blocks run along one axis of the matrix. Transposing it, which the backward pass needs, puts the blocks along the other axis, where the existing scales no longer fit. Transposing used to be free; now the matrix may have to be requantized. What MXFP8 training actually does, which he found "crazy but also kind of cool", is keep **two quantized copies** of every matrix, one for the original layout and one for the transpose.

::slide 28 | MXFP8 training in practice: in a transformer block only the QKV, output projection, FC1 and FC2 matmuls are MXFP8 (green), while layer norms, softmax, the attention batched matmuls and the activation stay BF16; below, weight, activation and gradient are each quantized for FPROP, and separately transposed and quantized for DGRAD and WGRAD, whose output goes to the optimizer in FP32

The training recipe on this slide (from arXiv 2506.08027) shows both points. Not everything is quantized: only the linear layers' matmuls run in MXFP8, and the layers judged unsafe stay in BF16, which again is trial and error. And every tensor is quantized twice, once as-is and once transposed. The forward matmul (FPROP) and the two backward matmuls (DGRAD for activation gradients, WGRAD for weight gradients) each read the layout they need.

What does it buy? The professor's figure: when those matmuls run in FP8 "you're probably going to get 20% to 30% savings, maybe more, depending on the size of your matrix". Not 2×, because of all the quantizing, dequantizing and transposed copies. The raw multiply gets roughly linearly faster with fewer bits, but the conversion overhead dilutes the gain. And a saving on the matmuls is only a saving on the matmul share of the step.

**Worked example.** Suppose the FP8 multiply takes 0.5 of the bf16 matmul time, and quantizing plus building the transposed copy adds 0.25. The new time is 0.75, a saving of 25%. If matmuls are 60% of the step time, the step gets 0.60 × 0.25 = 15% faster. "Half the bits, half the time" would have predicted 30%.

::video 42:27-42:46 | the quoted figure (20-30%, not 2x) and the stated reason (quantization overhead)
::widget fixture:lecture_05--fp8-step-saving | raise the quantize/transposed-copy overhead from 0 to 0.25: the matmul saving falls from the naive 50% into the quoted 20–30% band, and the step saving is that times the matmul share
::predict low-bit-speedup-diluted
::kp low-bit-speedup-diluted

The Q&A after this slide added several points.
- **First and last layers.** A student had read that they are hard to quantize. The professor had no intuition for the first layer. For the last one, he said it drives a lot of the loss, so quantizing it brings instability and big loss increases.
- **Why mostly matmuls?** You can quantize anything, such as activations after a ReLU, and it does cut memory traffic. But the matmul is where the big throughput gains are, so the quantize/dequantize overhead is usually not worth it elsewhere. The hardware supports all of it.
- **How scales are chosen.** They are not trained with gradients. Depending on the library, they come from scanning a block for its max and min, or from running statistics. At inference they can be fitted more carefully.

::slide 29 | MXFP4: all sixteen 4-bit values, 1 sign, 2 exponent and 1 mantissa bit: ±0, 0.5, 1, 1.5, 2, 3, 4, 6; one scale per 16 elements, the scales in E4M3

The professor showed **MXFP4** "almost because I think it's so ridiculous". Every value the element format can hold fits on one slide: 0, 0.5, 1, 1.5, 2, 3, 4, 6 and their negatives. The block scale does the rest; per the slide, one scale per 16 elements, the scale stored in E4M3. He knows of one paper that trained in FP4 but no big, serious model trained that way in the wild yet, and expects the next generation of models to use it.

::note slip 43:15 | Introducing this slide, the professor says "the entirety of MXFP8 can be shown to you in this slide". He means MXFP4; the slide shows 4-bit values.
::note aside 43:31 | Our note on the format: one scale per 16 elements with E4M3 scales is NVIDIA's NVFP4 layout. The OCP microscaling standard's MXFP4 uses one E8M0 scale per 32 elements, like MXFP8. The element values shown are the same in both.
::note aside 46:15 | Two more student questions were answered briefly. Structured sparsity has been tried a lot (the professor counts mixture-of-experts as one successful structured-sparse operation, and mentions Chris Ré's work on structured matrices), but in his view the compute benefits against the representation losses have not panned out. On current practice for small deployable models: you probably train a bigger model and quantize down, combining some quantization-aware training with post-training quantization. Even industry teams, he says, are "still doing science of quantization".

## Trick 2: operator fusion, one trip to memory instead of many {#fusion}
source: lecture_05.pdf p30-p33 · video 47:35-50:05

::slide 30 | Horace He's factory analogy: a memory warehouse connected to a compute factory by a conveyor belt; when the factory doubles in size, the belt stays the same, so "compute scales up, memory doesn't"

Fusion is the most conceptual of the tricks: a simple idea, but "you'll be surprised at how much this doesn't already happen". Think of the GPU as a factory (the compute) fed by a warehouse (global memory) through a conveyor belt (memory bandwidth). Making the factory bigger, which is what each GPU generation does, only helps if the belt keeps up. It does not, as slide 18 showed.

::slide 31 | naive: each of three operations ships its input from memory to compute and its result back to memory; fused: the input ships once, all three operations run in compute, and only the final result ships back

Now suppose a computation is a chain of several operations. Run naively, each operation is its own trip: load the input from the warehouse, process it, send the result back, then load it again for the next step. Shipping intermediates back and forth is "somewhat silly". A **fused kernel** loads the input once, runs the whole chain inside the SM with intermediates in registers, and writes only the final product back. You pay for the belt twice instead of twice per operation.

::slide 32 | the PyTorch FX graph of sin²x + cos²x: x feeds sin and cos, each is squared (pow), and an add combines them; "naively launches 5 CUDA kernels (back and forth)"

The concrete case is $\sin^2 x + \cos^2 x$, applied elementwise. PyTorch's computation graph has five pointwise operations: sin, cos, two squares and an add. Run eagerly, each is a separate CUDA **kernel** (one function launched on the GPU), and each reads its inputs from global memory and writes its output back.

::slide 33 | the same graph before fusion (five nodes inside a red box) and after TorchInductor's fusion: a single fused node between input and output

All five are pointwise, so they fuse into one kernel that reads x once and writes the result once. Counting tensor-sized trips makes the saving concrete:

::worked operator-fusion
::animation fixture:lecture_05--sincos-fusion | naive sin²x + cos²x: the compute lane idles while HBM writes each intermediate and the next kernel reads it back (6 reads + 5 writes); fused, HBM sees one read of x and one write of y

The saving is in **bytes moved**, not in FLOPs: every sin, cos, square and add still runs. It is not mainly about launch overhead either, though that is real. For a memory-bound chain, 11 tensor passes falling to 2 means roughly a 5.5× speedup.

Easy fusions like this, a graph you can "squish down into a single unit", are done automatically by compilers: `torch.compile` for PyTorch (its TorchInductor backend made the fused node on slide 33) and XLA for JAX. "The advanced version sometimes require manual intervention." Fusing across a reduction such as softmax is the hard case. FlashAttention, in Part 3, is exactly such a hand-written fusion, and [L6](#/read/lecture_06) writes fused kernels in Triton.

::note deferred 50:01 | "I'll talk a little bit more about fusion later." The harder fusions return in FlashAttention and in the next lecture's kernels.
::predict operator-fusion
::kp operator-fusion

## Trick 3: recomputation, throwing work away to save memory traffic {#recomputation}
source: lecture_05.pdf p34-p36 · video 50:05-52:50

::slide 34 | the backpropagation example from CS221: loss = (w·φ(x) − y)² with w = [3, 1], φ(x) = [1, 2], y = 2; the forward pass stores the values in yellow (score 5, residual 3, loss 9), the backward pass computes the gradients in green, ending in ∇w = [6, 12]

A reminder of what backprop stores. The forward pass computes each node's value, from the leaves to the root: score = 3·1 + 1·2 = 5, residual = 5 − 2 = 3, loss = 3² = 9. The backward pass then computes each node's gradient from the root down, and it *uses the stored forward values*: ∇w = 2 · residual · φ(x) = 2 · 3 · [1, 2] = [6, 12]. So training normally writes every activation to memory in the forward pass and reads it back in the backward pass. Mathematically that is just how gradients work. With a systems mindset, the professor says, it is a lot of memory traffic.

::slide 35 | three sigmoids stacked: x → sigmoid → s2 → sigmoid → s1 → sigmoid → out. Old forward pass: 1 memory read, 3 writes. Old backward pass reads s2, s1 and dout and writes dx: 3 reads, 1 write. "8 mem read/writes, very low arithmetic intensity"

Take three sigmoids stacked on top of each other. Storing activations, the forward pass reads x and writes s2, s1 and the output: 1 read and 3 writes. The backward pass reads s2, s1 and the incoming gradient dout, and writes dx: 3 reads and 1 write. That is 8 memory accesses for a handful of sigmoid FLOPs, a very low arithmetic intensity.

::slide 36 | the new version: the forward pass reads x and writes only out (1 read, 1 write); the backward pass reads x and dout, recomputes the three sigmoids on the fly, and writes dx (2 reads, 1 write); "5/8th the memory accesses"

Now throw the activations away. The forward pass reads x and writes only the output: 1 read, 1 write. The backward pass reads x and dout, **recomputes** the sigmoids on the fly, right where it needs them, and writes dx: 2 reads, 1 write. That is 5 accesses instead of 8, for the same gradients plus one extra forward computation. The professor's precondition: "imagine you're in a world where computation is super, super cheap". For memory-bound ops that is exactly the world we are in, so recomputing is *faster*, not just smaller.

::video 51:19-52:33 | the itemized count: 8 accesses when storing activations, 5 when recomputing

This is a second motive for a trick you may know from [L2's activation checkpointing](#/read/lecture_02). There, recomputation traded compute for memory *capacity*: store only every few layers' activations and recompute the rest to fit a bigger model. Here it trades compute for memory *traffic*. The deciding property is arithmetic intensity. Recomputing a cheap elementwise op (a sigmoid, a GeLU) costs a few FLOPs and saves a write and a read of a large tensor. Recomputing a big matmul re-pays the expensive, compute-bound part. PyTorch's "min-cut" recomputation, the source of these slides, chooses what to save and what to recompute on exactly this basis.

::predict recomputation-memory-traffic
::kp recomputation-memory-traffic

## Trick 4: memory coalescing, reading DRAM the way it wants to be read {#coalescing}
source: lecture_05.pdf p37-p39 · video 52:50-57:50

::slide 37 | a 16-byte address space cut into four 4-byte "burst sections" (0–3, 4–7, 8–11, 12–15); accessing one location delivers its whole section; in practice burst sections are 128 bytes or more; bottom left, a DRAM cell array where one row is selected and copied to the sense amplifiers

The professor introduces this one, with a question mark on the slide, as "another memory trivia thing" that turns out to matter. Global memory is **DRAM**, and DRAM is read in **bursts**. The address space is cut into burst sections. When you access one location, every other location in its section is delivered too, essentially for free. The toy example has 4-byte sections; real ones are about 128 bytes, a figure the professor also gives aloud.

Why bursts? A student asked. DRAM cells sit in a grid. Reading starts by *activating* a whole row, copying it into the sense amplifiers, and that step is slow. Once the row is open, reading neighbouring elements of it is comparatively cheap. You get the row for free, the professor says, not a T shape (a row plus a column).

::slide 38 | coalesced loads: four threads T0–T3 reading locations 0–3 all fall in one burst section, and so do T0–T3 reading 8–11; "if all accessed locations fall into the same burst section, only one DRAM request will be made"

A warp's 32 threads issue their loads together. If all the addresses fall in the same burst, the hardware merges them into a single DRAM transaction: the access is **coalesced**. Lecture 6 states it concretely: a warp's accesses are combined into 128-byte transactions, so 32 threads each reading a consecutive 4-byte float use exactly one. If the addresses scatter, the warp needs up to 32 transactions and uses 4 bytes of each 128.

::code lecture_06.py:L127-L130 | lecture 6: 128-byte transactions; full coalescing is 32 threads × 4 bytes = 128 bytes

::widget fixture:lecture_05--warp-access | with stride 1 the warp's 32 fp32 loads land in one 128-byte transaction; each doubling of the stride doubles the transactions, and reading down a column of a row-major matrix costs one transaction per thread
::predict memory-coalescing

::slide 39 | (A) not coalesced: in a row-major matrix, thread 1 and thread 2 each walk along their own row; (B) coalesced: threads each walk down their own column; right: a 4×4 row-major matrix laid out in memory as four colour-coded rows, with threads T0–T3 at load iteration 0 reading M0,0, M1,0, M2,0, M3,0, one element in each row's burst

Why does a DRAM detail matter for ML? Because matrices are read in big blocks for matmuls, and some access orders are far better than others. To see which, you need the layout. In **row-major** order a matrix is stored row after row. Neighbours along a row are adjacent in memory; neighbours down a column are a whole row apart.

::worked supp-row-major-layout
::kp supp-row-major-layout

Now picture a kernel where each thread handles one row and walks along it (picture A). At each step, the warp's threads read the *same column position* in *different rows*. Those addresses are a row apart, so each lands in a different burst. In the 4×4 example on the right, iteration 0 touches all four colour-coded bursts to get one column; iteration 1 touches all four again. You read the whole matrix to get each column, "even though all I wanted was a single column". Flip the assignment (picture B), so each thread owns a column, and at each step the warp reads consecutive elements of one row: one burst, fully coalesced. The professor's mnemonic: for a row-major matrix, threads that move along the rows, the major axis, are *not* coalesced.

::video 57:00-57:37 | reading one column of a row-major matrix touches every burst
::note why 56:22 | The slide's phrase "threads that move along rows are not coalesced" describes each thread's path over time. What decides coalescing is where the warp's 32 *simultaneous* addresses fall, and in picture A they form a column.
::kp memory-coalescing

## Trick 5: tiling, the big one {#tiling}
source: lecture_05.pdf p40-p42 · video 57:50-1:02:15

::slide 40 | a 4×4 matmul P = M·N with one thread per output element; the access-order table shows thread(0,0) reading M0,0·N0,0, M0,1·N1,0, …; thread(0,1) reads M0,0 again and thread(1,0) reads N1,0 again

The professor saves the most important trick for last; he thinks it has the biggest impact on performance. **Tiling** means grouping and ordering the work so that a block reads each piece of global memory once and reuses it from shared memory.

Start from a naive matmul, P = M·N, with one thread per output element. Thread (i, j) computes P[i][j] by reading row i of M and column j of N. The access table shows the waste. Thread (0,0) and thread (0,1) both read M0,0. Thread (0,0) and thread (1,0) both read N1,0. Every element of M is needed by all the threads in its output row, so in an N×N matmul each input element is fetched from global memory **N times**. The reads are not coalesced either.

::slide 41 | the matrices cut into 2×2 tiles (thick borders); the matmul runs in phases: 1. load the M0,0 and N0,0 tiles into shared memory; 2. compute partial sums for P; 3. load the next pair; 4. …; "repeated reads now access shared, not global memory, and memory access can be coalesced"

Since the same element is read many times, why not fetch it once and keep it close? Matrices have a natural grouping: cut them into square sub-matrices, **tiles**. In the 4×4 example with 2×2 tiles, the block computing the top-left output tile works in **phases**:
- **Phase 1.** Load the top-left tile of M (M0,0 to M1,1) and the top-left tile of N into shared memory. This is the only slow part: one transfer from global memory.
- **Phase 2.** Multiply the two tiles and add the result into the output tile's partial sums, all reading from fast shared memory.
- **Phase 3.** Load the next pair: the next tile along M's rows (columns 2–3) and the next tile down N's columns (rows 2–3), and add their product in.
- **Phase 4.** Continue until the whole row of M-tiles and column of N-tiles has been used, then write the finished output tile back.

Two gains: the repeated reads now hit shared memory, and loading a tile, consecutive elements of each row, can be coalesced.

::note slip 59:32 | The slide's step 3 reads "Load the M0,0 and N2,0 tile". N2,0 is the corner element of N's second tile, but the second M tile is the one starting at M0,2, not M0,0 again. Phase 2 pairs M's tile (columns 2–3) with N's tile (rows 2–3).

::animation fixture:tiling-sweep | each pair of input tiles is loaded into shared memory once and then reused for all T×T partial products before the next pair loads; the global-read counter advances only on tile loads

::slide 42 | tiling math: A · B = C with tile size T and matrix size N; the outer loop runs over tiles (purple), the inner loop over elements inside the current tile (green), accumulating into a temporary result tile; non-tiled, each input is read N times from global memory; tiled, N/T times, and T times within each tile

The arithmetic. Untiled, each input element is read $N$ times from global memory. Tiled with tile size $T$, each element is loaded into shared memory once per output tile that needs it, which is $N/T$ times, and each load is then used $T$ times from shared memory. The total number of uses is still $(N/T) \times T = N$. Only where they come from changes:

$$ \text{global reads per element: } N \;\rightarrow\; \frac{N}{T}, \qquad \text{a factor of } T $$

The professor adds the limiting case aloud: if $T = N$, the whole matrix is one tile, read once from global memory and used $N$ times from shared memory. FLOPs are unchanged, $2N^3$ for a square matmul (see [L2's matmul FLOPs](#/read/lecture_02)). Tiling reorders the same sums.

::worked tiling-shared-memory
::video 1:00:19-1:00:48 | "N over T" said aloud, and the limiting case T = N
::predict tiling-shared-memory

Why not just use $T = N$? Because the tiles must fit in shared memory: about 256 KB per SM on an H100, shared with L1. With two fp32 tiles of $T \times T$ in a 128 KB budget, $2 \cdot T^2 \cdot 4 \le 131{,}072$ gives $T \le 128$. A student checked that this is the whole reason, and the professor agreed. An all-SRAM chip like Groq can hold the whole matrix and be very fast, but it is very expensive.

Two more questions rounded this out. Is splitting work across GPUs the same idea? In a way: data and tensor parallelism cut matrices along an axis across devices, so "all these parallelism look a little bit like tiling". Does PyTorch already tile? Yes, every library matmul kernel tiles. Tiling matters to you when you write your own kernels or exotic operations, and even library kernels are imperfect, as the next section shows.

::kp tiling-shared-memory

## When tiles do not fit the matrix: the matrix mystery solved {#mystery}
source: lecture_05.pdf p43-p49 · video 1:02:15-1:11:10

::slide 43 | NVIDIA's tile-quantization figure with 128×128 thread-block tiles: (a) a 256×256 matrix splits into exactly 4 tiles; (b) at 257 columns, six thread blocks are launched, two of which waste most of their work; factors affecting tile sizes: coalesced memory access, shared memory size, divisibility of the matrix dimension

Tiling is powerful, but it brings subtle effects. With 128×128 tiles, a 256×256 output is exactly four tiles: "life is good". Add one column and the output is 256×257. Now there are 2 × 3 = 6 tiles, and the extra column of tiles holds one real column out of 128: two "very skinny tiles with basically nothing inside of them". Each partial tile still costs a whole thread block. So tile size is something you *optimize*, trading memory reuse (bigger tiles) against wasted partial tiles, under three constraints: coalesced access, shared memory size, and the divisibility of the matrix dimensions.

In practice nobody derives tile sizes by hand. The professor mentions PyTorch's `max-autotune` compiler mode. Turn it on, and for the next 15 minutes or so PyTorch benchmarks one tile configuration after another on your shapes. It gives "nontrivial benefits".

::slide 44 | memory comes in bursts; "aligned layout": a tile whose width matches the burst sections loads as one nice tile; "unaligned layout": the same tile straddles burst boundaries and costs two bad tiles; "coalesced accesses may be impossible depending on the dimension of the matrix (have to do padding)"

The second complication ties tiling back to coalescing. Suppose each row of a tile is exactly one burst wide. If rows start on burst boundaries, loading the tile takes one burst per row: "one nice tile". Now make the matrix one element wider. Each row starts one element later than the previous one, so the tile's rows straddle burst boundaries, and loading it touches two bursts per row: "two bad tiles". No access order fixes this; the dimension itself is the problem. The fix is **padding**: round the dimension up so rows line up with bursts again.

That explains a famously weird observation. Andrej Karpathy, speeding up nanoGPT, increased the vocabulary size from 50,257 to 50,304 and got about a **25% speedup**. 50,304 = 64 × 786. It is odd to pad a matrix and get faster, until you see that an odd width misaligns every row read, and a width divisible by a large power of 2 aligns them all.

::widget fixture:lecture_05--warp-access | try C = 50257, then 50304: with the odd width almost every row starts off a 128-byte boundary and a 32-element row read needs about 2 transactions; padded to a multiple of 32, every row read needs 1

The professor's practical rule, given when a student asked about tile sizes: don't think about tile sizes unless you write your own kernels, but do think about **matrix sizes**. Make them powers of 2, or at least "ideally divisible also by 32". For a hidden size of 3000, that means padding to 94 × 32 = 3008.

::video 1:05:15-1:05:35 | the worked case: vocab 50,257 → 50,304, about 25% faster

::slide 46 | the square-matmul plot once more: "we understand some of this (compute intensity, tiling); let's take a closer look"

The overall rise from the left is the roofline: small matmuls do too little work per byte to saturate the tensor cores. That leaves the bands and the drops.

::slide 47 | the same points colour-coded by the largest K dividing N: K = 2 orange, 8 green, 16 red, 32 purple, and blue for odd N; next to it, the aligned and unaligned layouts

The bands are **divisibility**. The professor reads the legend aloud. Blue points are sizes divisible only by 1, not even even. They are the bottom band, under 100 TF/s even at N = 4096. Orange (divisible by 2) is next, at roughly 125–150 TF/s. Green (8) reaches about 200–230. Red (16) and purple (32) form the top band at up to about 250 TF/s, and they perform the same. That is not because powers of two are magic. It is because 16 and 32 are large enough to give the right burst-window property: tiles load with coalesced reads. "Tiling has a major impact through alignment."

A student asked whether divisible by 64 would beat 32. No: "as long as it divides by your burst size, then you're good to go. It's a divisor property." Other phenomena can reward larger divisors, but not this one.

::video 1:06:40-1:07:38 | the divisibility legend read aloud (1, 2, 8, 16, 32) and why 16 and 32 tie
::predict dim-divisibility-padding
::kp dim-divisibility-padding

::slide 48 | a zoom of the plot around 1536–2048 showing sudden drops; "this happens at 1792 to 1793": with 256 × 128 tiles there are 1792/256 × 1792/128 = 7 × 14 = 98 tiles; at 1793 there are 8 × 15 = 120; "an A100 has 108 SMs, so it cannot execute all 120"

Last, the periodic drops. Each 256 × 128 output tile is one thread block, and each block runs on one SM.

::worked tile-wave-quantization

Growing both dimensions by one adds 22 tiles, almost all of them nearly empty. That is bad, but not this bad. The real cause is the SM count. Blocks are scheduled onto SMs in **waves**. 98 tiles fit in one wave on 108 SMs. 120 do not: after the first 108 finish, a second wave runs the remaining 12 while 96 SMs sit idle. Time follows the number of waves, so about 0.17% more FLOPs roughly doubles the runtime. This is **wave quantization**.

::widget fixture:lecture_05--wave-grid | step M from 1792 to 1793: tiles jump 98 → 120, more than the A100's 108 SMs, so a second wave runs 12 tiles while 96 SMs sit idle; on the H100 (132 SMs) the same 120 tiles fit in one wave
::code lecture_06.py:L134-L137 | lecture 6's statement of the same effect on a B200: 160 blocks on 148 SMs leave a second wave of 12

Where the cliff falls depends on the GPU, not only the matrix. On an H100 with 132 SMs, the same 120 tiles fit in one wave, and that cliff disappears (the partial tiles still waste work). Lecture 6's fix is to make the number of thread blocks divide the number of SMs. Asked why 108 SMs, the professor answered: "You will have to ask Jensen for that."

::note aside 1:08:03 | "Percy calls this GPU trivia. So I think he hates this one, but I love this one." The two instructors disagree on how much wave quantization matters to a modeller.
::predict tile-wave-quantization
::kp tile-wave-quantization

::slide 49 | recap of Part 2: reduce memory accesses (coalescing, fusion); move memory to shared memory (tiling); trade memory for compute or accuracy (quantization, recomputation)

The professor's summary of Part 2 is "memory, memory, memory": read less, read from closer, or spend accuracy or compute to move fewer bytes.

::note aside 1:11:46 | Asked about Cerebras' wafer-scale chips, the professor said compilation there is much harder (he mentions interference effects across the wafer) and that "that's a world that I'm not familiar with".

## Part 3: FlashAttention from the tricks we know {#flashattention}
source: lecture_05.pdf p50-p55 · video 1:12:00-1:18:35

::slide 50 | FlashAttention (Dao et al.): GPT-2 attention takes about 17 ms in PyTorch (matmul, mask, softmax, dropout, matmul) and about 2 ms as one fused kernel; the table: standard 66.6 GFLOPs, 40.3 GB of HBM reads/writes, 41.7 ms against FlashAttention's 75.2 GFLOPs, 4.4 GB, 7.3 ms; HBM accesses and runtime fall as the block size grows from 64 to 512

FlashAttention was one of the biggest improvements to attention in years, and "it's all systems": it replaces PyTorch's chain of separate attention ops with one cleverly fused kernel. The table says where the speed comes from. FlashAttention does *more* arithmetic, 75.2 against 66.6 GFLOPs, yet runs 41.7 / 7.3 ≈ 5.7× faster, because it moves about 9× fewer bytes through HBM (40.3 GB against 4.4 GB). The paper's own summary, quoted on the slide, names two techniques, tiling and recomputation, used to compute exact attention with sub-quadratic HBM accesses. We now know both.

::slide 51 | recap of attention: XQ times KᵀXᵀ gives the n×n score matrices (one per head, 3 here); softmax of the scores times XV mixes the values into an n×d output

Attention, per head: the scores are $S = QK^\top$, an $n \times n$ matrix for sequence length $n$. Then $P = \mathrm{softmax}(S)$ row by row, and the output is $O = PV$, which is $n \times d$. The slide counts "3 matrix multiplies (K, Q, V)"; the attention core itself is two, $QK^\top$ and $PV$, with the Q, K, V projections before them. Run naively, $S$ and $P$, each $n \times n$, are written to HBM and read back.

**Worked size.** One head, $n = 8192$, bf16. The score matrix is $8192^2 \times 2$ bytes ≈ 134 MB, written and read at least twice (scores, then probabilities). The output, with $d = 128$, is $8192 \times 128 \times 2$ bytes ≈ 2 MB. Doubling $n$ quadruples the first and only doubles the second.

::slide 52 | left: the memory pyramid with the paper's A100 numbers: GPU SRAM 19 TB/s (20 MB), GPU HBM 1.5 TB/s (40 GB), CPU DRAM 12.8 GB/s (>1 TB); right: FlashAttention's Figure 1: blocks of Kᵀ and V copied to SRAM in the outer loop, blocks of Q in the inner loop, each score block computed on SRAM, output written to HBM

The paper's Figure 1 is "literally just tiling" for the attention matmuls: copy blocks of Q, K and V into SRAM, multiply them there, move on. The pyramid is slide 10's hierarchy in the paper's A100 numbers.

The hard part is the softmax in the middle. A softmax normalizes each row by a sum over the *whole* row, so it seems to tie all the tiles of a row together: you cannot finish any tile's probabilities until you have seen every score in the row. This is the piece you will implement in the assignment.

::note deferred 1:13:46 | "You'll see this in the assignment": implementing the tiled global softmax is part of the assignment. The forward pass is taught here in full.

::slide 53 | from Milakov and Gimelshein (2018): the safe softmax makes three passes (find the max, sum the exponentials, normalize); the online softmax updates the max and the sum together in one pass, d_j = d_{j−1}·e^{m_{j−1} − m_j} + e^{x_j − m_j}, then normalizes

The standard "safe" softmax subtracts the row maximum before exponentiating, so no exponential overflows. That needs one pass to find the max, a second to sum $e^{x_j - m}$, and a third to divide. The **online softmax** merges the first two. Keep a running max $m$ and a running sum $d$. When a new element raises the max, every term already in the sum was computed against the old max and is too large by $e^{m_{old} - m_{new}}$, so multiply the sum by that factor before adding the new terms:

$$ m_j = \max(m_{j-1}, x_j), \qquad d_j = d_{j-1}\, e^{\,m_{j-1} - m_j} + e^{\,x_j - m_j} $$

The corrections telescope: at the end, $d$ equals $\sum_j e^{x_j - m_{final}}$ exactly, in one pass. The same update works a *tile* at a time, with the tile's max and its sum of exponentials. In the professor's words, "I don't need to see the rest of the tiles in order to compute this."

::worked online-softmax
::animation fixture:online-softmax-blocks | when a tile raises the running max, the accumulated denominator is multiplied by exp(m_old − m_new) rather than recomputed from scratch; when the max is unchanged the factor is 1
::predict online-softmax
::note slip 1:14:00 | The slide credits "Mikailov and Gimelshein 2018". The paper is by Milakov and Gimelshein (arXiv 1805.02867).
::kp online-softmax

::slide 54 | the FlashAttention-2 forward pass for one Q block and two K/V blocks: S(1) = Q K(1)ᵀ and S(2) = Q K(2)ᵀ are computed in SRAM (dashed, never written to HBM), exponentiated to A(1), A(2), summed into l(1) and l(2) = l(1) + Σ exp(S(2)); the output starts as O(1) = A(1)V(1)/l(1) and is rescaled, O(2) = (l(1)/l(2))·O(1) + A(2)V(2)/l(2); Q, K, V and O live in HBM (blue)

Now put it together, using the slide's diagram from FlashAttention-2 (the slide cites Dao 2023). Take one block of Q rows and stream the K and V blocks past it:
- **Step 1, tiled matmul.** Compute the score tile $S^{(1)} = Q K^{(1)\top}$ in SRAM. It is never written to HBM.
- **Step 2, fused exponential.** Exponentiate it in the same kernel: $A^{(1)} = \exp(S^{(1)})$, and add its row sums into the running normalizer $l^{(1)}$.
- **Step 3, multiply by V.** Add $A^{(1)} V^{(1)}$ into the output accumulator.
- **Step 4, next tile, with the telescoping correction.** Compute $S^{(2)}$, $A^{(2)}$, update $l^{(2)} = l^{(1)} + \sum \exp(S^{(2)})$, and rescale the output already accumulated, as the slide's arrow says, "to correct denominator": $O^{(2)} = \frac{l^{(1)}}{l^{(2)}} O^{(1)} + \frac{A^{(2)}}{l^{(2)}} V^{(2)}$. The full algorithm also subtracts a running max, and rescales both $l$ and $O$ by $e^{m_{old} - m_{new}}$ whenever it rises, exactly as in the online softmax. The slide's diagram leaves the max out for clarity. FlashAttention-2 actually keeps the output unnormalized and divides by $l$ once at the very end.

The per-row state is tiny, a max and a sum per row, so it lives in registers or shared memory. Only Q, K and V tiles are read from HBM and only the $n \times d$ output is written. K and V are streamed once for every block of Q rows, so the traffic is a multiple of $n \cdot d$ that shrinks as SRAM, and with it the block size, grows; the paper bounds it by $O(n^2 d^2 / M)$ for SRAM size $M$, which the block-size panel on slide 50 shows. Nothing of size $n \times n$ ever reaches HBM, while the matmul FLOPs stay $O(n^2 d)$. For long sequences that moves attention from memory-bound toward compute-bound, which is the roofline goal from Part 2.

::animation fixture:flash-forward-tiles | only one S tile exists at a time, in shared memory; the HBM traffic counter advances for Q, K, V tile loads and the final O write, never for an N×N matrix
::predict flash-attention-forward

The slide's three bullets name the tricks: tiling (of the inner products $S$), **fusion** (of the exponential into the matmul kernel), and the **online, telescoping softmax**. The fourth, recomputation, lives in the backward pass. Storing the $n^2$ attention probabilities for backward would bring the quadratic memory back, so FlashAttention throws them away and recomputes them tile by tile from Q, K and the saved per-row statistics during backward. That is slide 36's trick, applied where it pays most.

::note skip 1:16:49 | The backward pass is skipped in class ("I'm not going to cover this or talk about it"), apart from that one sentence: recompute tile by tile instead of saving the N²-sized activations. The [FlashAttention paper read-through](#/read/flashattention) covers it, along with the paper's I/O-complexity analysis.
::kp flash-attention-forward

::slide 55 | recap of the lecture: hardware powers scale, and low-level details decide what scales; current GPU compute strongly encourages thinking about matmuls and data movement; careful thinking about the GPU (coalescing, tiling, fusion) leads to good performance

The professor adds one warning to the slide: understand *why*, down to the hardware. "I don't want you guys to be the kind of people that cargo cult, 32 multipliers for your matrices." Hardware-aware architecture design, he says, is critical for future systems. The next lecture, [L6](#/read/lecture_06), puts this into practice: benchmarking, profiling, and writing these kernels in Triton.
