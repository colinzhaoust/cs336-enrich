---
title: FlashAttention · IO-aware exact attention, read through
minutes: 40
---
In 2022 Tri Dao and colleagues at Stanford made attention several times faster without approximating anything. The trick was to stop counting arithmetic and start counting trips to GPU memory. This read-through follows the paper in its own order: the IO model, why standard attention is slow, the tiled algorithm with its running softmax, recomputation in the backward pass, the IO theorem and its lower bound, the block-sparse extension, and the experiments. Afterwards you can trace Algorithm 1 by hand, count its memory traffic, and say which of the paper's headline numbers mean what.

## Why is attention slow, and what were faster methods missing? {#why}
source: fa:abstract · fa:§1 · Fig. 1 · Fig. 2 (left)

Self-attention compares every token with every other token. For a sequence of $N$ tokens the score matrix is $N \times N$, so time and memory grow with $N^2$: going from 1K to 2K tokens of context quadruples them. That is why long context was hard in 2022.

Many papers had attacked the $N^2$ with **approximate attention**. Sparse methods (Reformer, local attention) compute only some scores. Low-rank methods (Linformer, Performer) factor the attention matrix through a smaller one. These cut the FLOPs to linear or near-linear in $N$. Yet the paper observes that many of them show no wall-clock speedup over standard attention, and few were adopted. Its diagnosis: they reduce FLOPs, which "may not correlate with wall-clock speed", and ignore the cost of memory access, which the paper calls **IO**.

The paper's thesis is that attention algorithms should be **IO-aware**: they should count reads and writes between the levels of fast and slow memory, as IO-complexity analysis has done for sorting and database joins since Aggarwal and Vitter (1988). On a GPU the two levels that matter are the small on-chip **SRAM** and the large off-chip **HBM**. The next section puts numbers on both.

The goal follows directly: never write the $N \times N$ attention matrix to HBM, and never read it back. Two things stand in the way.
- **The softmax.** Each row of attention is normalized by a sum over the *whole* row, which seems to require the full row at once.
- **The backward pass.** Gradients normally need the attention matrix saved from the forward pass.

The paper answers each with an established technique. **Tiling** splits the inputs into blocks and computes the softmax incrementally, block by block. **Recomputation** saves only the softmax's normalization statistics and rebuilds the attention matrix on chip during the backward pass. Everything runs inside a single CUDA kernel, so the intermediate values never leave the chip.

::slide lecture_05:50 | left, the paper's Fig. 1 (right): PyTorch's GPT-2 attention, about 17 ms, split into matmul, mask, softmax, dropout, matmul, against about 2 ms for the fused kernel; middle, Fig. 2 (left); right, Fig. 2 (middle), which returns in the IO-complexity section

Look at the left bar first. The two matrix multiplies, where the FLOPs are, are thin slices. Most of the 17 ms goes to masking, softmax and dropout, operations that do almost no arithmetic but each read and write a full $N \times N$ matrix. The fused kernel's caption claims a 7.6× speedup on the attention computation.

::figure fa:fig1 | the paper's Figure 1. Left: tiling keeps the N×N attention matrix (dotted box) off HBM; the outer loop (red) walks over K and V blocks, the inner loop (blue) over Q blocks. Right: the 7.6× speedup on GPT-2 attention

The table in the middle of the slide is the whole paper in three rows. It measures GPT-2 medium's attention, forward plus backward, on an A100 (sequence length 1024, head dimension 64, 16 heads, batch 64).

| | Standard | FlashAttention |
|---|---|---|
| GFLOPs | 66.6 | 75.2 |
| HBM reads/writes (GB) | 40.3 | 4.4 |
| Runtime (ms) | 41.7 | 7.3 |

::predict fa-io-aware-framing

FlashAttention does 75.2 / 66.6 ≈ 1.13, so 13% *more* arithmetic (the recomputation in the backward pass) and still runs 41.7 / 7.3 ≈ 5.7× faster, because it moves 40.3 / 4.4 ≈ 9.2× fewer bytes through HBM. Runtime follows the bytes, not the FLOPs. The speedup is smaller than 9.2× because not all of the runtime was HBM traffic to begin with.

FlashAttention is also **exact**: it returns the same $\mathrm{softmax}(QK^\top)V$ as the standard implementation, up to floating point. The speed comes from where intermediate values live, not from dropping terms. That property carries the experiment section, and we return to it there.

::kp fa-io-aware-framing

## What does the hardware look like, and what is M? {#hardware}
source: fa:§2.1 · lecture_05.pdf p10, p52

The paper's background section is the memory hierarchy that [L5](#/read/lecture_05) builds up in detail. In the paper's A100 numbers:

| Memory | Size | Bandwidth |
|---|---|---|
| on-chip SRAM | 192 KB per SM, 108 SMs | about 19 TB/s |
| HBM (off-chip DRAM) | 40–80 GB | 1.5–2.0 TB/s |

Summed over all 108 streaming multiprocessors (SMs), SRAM is 192 × 108 = 20,736 KB, about 20 MB. So SRAM is roughly 10× faster than HBM (19 against 1.5–2.0 TB/s) and about 2,000–4,000× smaller (20 MB against 40–80 GB). The slide redraws this as a pyramid, with CPU DRAM at 12.8 GB/s below it.

::slide lecture_05:52 | left: the memory pyramid with the paper's A100 numbers (SRAM 19 TB/s, 20 MB; HBM 1.5 TB/s, 40 GB); right: the paper's Figure 1, which the professor calls "literally just tiling" for the attention matmuls

A GPU **kernel** is one operation launched on the GPU. It loads its inputs from HBM into registers and SRAM, computes, and writes its outputs back to HBM. Whether a kernel is fast then depends on its **arithmetic intensity**, FLOPs per byte moved (see [L2's arithmetic intensity](#/read/lecture_02)):
- **Compute-bound** kernels spend their time doing arithmetic: large matmuls, convolutions with many channels.
- **Memory-bound** kernels spend their time waiting for HBM: elementwise operations (activations, dropout, masking) and reductions (sum, softmax, layer norm).

The standard fix for memory-bound chains is **kernel fusion**. If several operations apply to the same input, load it once, apply all of them in SRAM, and write once. Compilers fuse elementwise chains automatically ([L5's fusion trick](#/read/lecture_05)). The paper points out the catch in training. The intermediate values still have to be written to HBM so that the backward pass can use them, which undoes much of the benefit. Fusing attention therefore needs both tiling and recomputation.

### What M means in the theorems
The analysis later uses $M$ for "the size of SRAM". The paper never states the unit, but its range $d \le M \le Nd$ only makes sense counted in **elements**, and $M$ is what one kernel's block can use: one SM's share, not the whole chip's 20 MB. Converting one SM's 192 KB to fp16 elements and feeding it into Algorithm 1's block-size rule, which we meet below:

::worked supp-sram-size-m

So $M$ is tens of thousands of elements, and the blocks are tens to hundreds of rows. In practice a kernel cannot use all 192 KB (registers and other state compete for it), which is why the paper speaks of M "around 100KB".

::note warning | §3.2 writes "M (around 100KB)" in bytes right next to $d^2$ in elements. The comparison $d^2 \ll M$ holds either way for $d$ = 64–128 (4,096–16,384 against tens of thousands), but when you compute block sizes, convert to elements first.
::widget fixture:flashattention--io-ledger | set SRAM to 192 KB and switch fp16 to fp32: M in elements halves, and so does the block size B_c; nothing is ever summed over the 108 SMs
::kp supp-sram-size-m

## How much memory traffic does standard attention cause? {#standard}
source: fa:§2.2 · fa:alg0 · fa:§C:thm2-proof

For one attention head, the inputs are three $N \times d$ matrices: queries $Q$, keys $K$ and values $V$, where $N$ is the sequence length and $d$ the head dimension (64 for GPT-2). Attention computes

$$ S = QK^\top \in \mathbb{R}^{N\times N}, \qquad P = \mathrm{softmax}(S) \in \mathbb{R}^{N\times N}, \qquad O = PV \in \mathbb{R}^{N\times d} $$

with the softmax applied to each row. (In practice $S$ is also scaled by $1/\sqrt{d}$ and may be masked; Appendix B carries those along, and they change nothing below.) The paper's **Algorithm 0** is how a framework like PyTorch runs this, as three separate kernels:
- **line 1:** load $Q$ and $K$ by blocks from HBM, compute $S = QK^\top$, write $S$ to HBM.
- **line 2:** read $S$ from HBM, compute $P = \mathrm{softmax}(S)$, write $P$ to HBM.
- **line 3:** load $P$ and $V$ by blocks from HBM, compute $O = PV$, write $O$ to HBM.

The matmuls in lines 1 and 3 are tiled internally, as every library matmul is. The waste is between the kernels. The $N \times N$ matrix makes a round trip through HBM between each pair of steps, because the softmax runs as its own pass. Count the elements moved, line by line:

::worked fa-standard-attention-hbm-traffic

The paper states the result as **Theorem 2**'s first half: standard attention needs $\Theta(Nd + N^2)$ HBM accesses. Theorem 2 counts elements, not bytes, and $\Theta$ hides the constant 4. Since usually $N \gg d$, the $N^2$ term dominates. For GPT-2, $N = 1024$ and $d = 64$, so the four crossings of $S$ and $P$ move $N/d = 16$× more elements than reading the inputs and writing the output.

::predict fa-standard-attention-hbm-traffic

The quadratic traffic is a choice of implementation, not of mathematics. Nothing in $O = \mathrm{softmax}(QK^\top)V$ requires $S$ to exist in HBM. It is there because the softmax kernel needs the whole row, and a separate kernel can only receive it through HBM.

Scale makes this concrete. In Fig. 2's GPT-2 medium setting (64 sequences × 16 heads, $N = 1024$), one copy of $S$ has 64 × 16 × 1024² ≈ 1.07 billion entries, about 2.1 GB at 2 bytes each, so the four crossings of the forward pass move about 8.6 GB. Masking applied to $S$ and dropout applied to $P$ each add one more memory-bound pass over $N^2$ entries: the tall bars in Fig. 1 (right). Fusing masking into the softmax helped, but the matrix itself still went to HBM.

::widget fixture:flashattention--io-ledger | read the "standard" row: its 4N² part dwarfs its 4Nd part, so doubling d barely moves it, while doubling N quadruples it
::kp fa-standard-attention-hbm-traffic

## How can a softmax be computed one block at a time? {#softmax-tiling}
source: fa:§3.1 (Tiling) · fa:alg1:L10-L11 · lecture_05.pdf p53

To avoid the round trips, the kernel must work on blocks of $S$ that fit in SRAM, and never hold a full row. The obstacle is that softmax "couples columns of K", in the paper's phrase: every entry of a row is divided by a sum over all of the row's columns.

Start with how a single softmax is computed safely. Exponentials of large scores overflow, so implementations subtract the row maximum first. For a vector $x \in \mathbb{R}^B$:

$$ m(x) = \max_i x_i, \qquad f(x) = \big[e^{x_1 - m(x)}\ \cdots\ e^{x_B - m(x)}\big], \qquad \ell(x) = \sum_i f(x)_i, \qquad \mathrm{softmax}(x) = \frac{f(x)}{\ell(x)} $$

Now split a row into two blocks, $x = [x^{(1)}\ x^{(2)}]$, and suppose each block was processed on its own, against its own max. The max of the whole row is the larger of the two block maxes. Each block's exponentials were taken against the wrong reference, but they are off by exactly one constant factor per block, so they can be corrected:

$$ m(x) = \max\big(m(x^{(1)}),\, m(x^{(2)})\big), \qquad \ell(x) = e^{m(x^{(1)}) - m(x)}\,\ell(x^{(1)}) + e^{m(x^{(2)}) - m(x)}\,\ell(x^{(2)}) $$

The same factors fix the vector $f(x)$, block by block. So if we carry two numbers per row, the running max $m$ and the running sum $\ell$, we can compute the softmax one block at a time. The paper calls this "algebraic aggregation" and credits the rescaling idea to earlier work, including Milakov and Gimelshein's 2018 online softmax, which [L5](#/read/lecture_05) teaches on slide 53 as a "telescoping sum".

::slide lecture_05:53 | Milakov and Gimelshein's two algorithms: the safe softmax's three passes (max, sum, normalize) against the online softmax's single pass that updates the max and the sum together

Algorithm 1 applies the rule with the running statistics $(m_i, \ell_i)$ of a block of rows playing $x^{(1)}$ and the new score tile playing $x^{(2)}$. Line 10 computes the new tile's own statistics on chip: $\tilde m_{ij} = \mathrm{rowmax}(S_{ij})$, $\tilde P_{ij} = \exp(S_{ij} - \tilde m_{ij})$ and $\tilde\ell_{ij} = \mathrm{rowsum}(\tilde P_{ij})$. Line 11 merges them:

$$ m_i^{new} = \max(m_i, \tilde m_{ij}), \qquad \ell_i^{new} = e^{m_i - m_i^{new}}\,\ell_i + e^{\tilde m_{ij} - m_i^{new}}\,\tilde\ell_{ij} $$

::predict fa-softmax-decomposition
::worked fa-softmax-decomposition

Both factors are at most 1, and at least one of them is exactly 1. If the new tile raises the max, the old sum shrinks by $e^{m_{old} - m_{new}}$. If it does not, the old sum is untouched and it is the new tile's terms that get scaled down. Forgetting a factor does not cause a small error: adding sums taken against different maxes mixes units, and the error is $e^{m_{new} - m_{old}}$, which can be arbitrarily large.

::animation fixture:online-softmax-blocks | block 3 raises the running max from 3.0 to 4.0, so the accumulated denominator is multiplied by exp(m_old − m_new) instead of being recomputed; at block 2 the max is unchanged and the factor is 1
::note why | Appendix B.1 derives the memory-efficient forward pass without the max-shift "for simplicity". Mathematically the max cancels; it is there only so that no exponential overflows. Every line of Algorithm 1 carries it, so the bookkeeping above is what a real kernel does.
::kp fa-softmax-decomposition

## Algorithm 1: which blocks are loaded, in which order? {#algorithm1}
source: fa:§3.1 · fa:alg1 · fa:fig1 · fa:§C:thm2-proof

With the softmax decomposed, the algorithm is a tiled matmul with bookkeeping. Here is **Algorithm 1** as printed, condensed. The input is $Q, K, V$ in HBM and an SRAM of $M$ elements.
- **1.** set block sizes $B_c = \lceil M/4d \rceil$ and $B_r = \min(\lceil M/4d \rceil, d)$.
- **2.** in HBM, initialize $O = 0$ ($N \times d$), $\ell = 0$ and $m = -\infty$ (length $N$ each).
- **3.** split $Q$ into $T_r = \lceil N/B_r \rceil$ row blocks $Q_i$ of size $B_r \times d$, and $K$, $V$ into $T_c = \lceil N/B_c \rceil$ blocks $K_j$, $V_j$ of size $B_c \times d$.
- **4.** split $O$, $\ell$ and $m$ into the same $T_r$ row blocks $O_i$, $\ell_i$, $m_i$.
- **5.** **Outer loop**, for $j = 1, \dots, T_c$:
- **6.** load $K_j$ and $V_j$ from HBM into SRAM, once per outer step.
- **7.** **Inner loop**, for $i = 1, \dots, T_r$:
- **8.** inside the inner loop: load $Q_i$, $O_i$, $\ell_i$, $m_i$ from HBM into SRAM;
- **9.** on chip, $S_{ij} = Q_i K_j^\top$, a $B_r \times B_c$ tile;
- **10.** on chip, the tile's statistics $\tilde m_{ij}$, $\tilde P_{ij}$, $\tilde \ell_{ij}$ (previous section);
- **11.** on chip, merge into $m_i^{new}$ and $\ell_i^{new}$ (previous section);
- **12.** write the updated $O_i$ to HBM (next section);
- **13.** write $\ell_i \leftarrow \ell_i^{new}$ and $m_i \leftarrow m_i^{new}$ to HBM.

(Lines 14–16 close the loops and return $O$.) Nothing of size $N \times N$ is ever written: each score tile $S_{ij}$ is born in SRAM in line 9 and dies there after line 12.

### The block sizes
Line 1 is sized so that everything one inner step needs fits on chip together. The paper does not explain the constants, but Appendix C's proof gives the conditions they satisfy. $K_j$ and $V_j$ ($B_c \times d$ each) must fit, so do $Q_i$ and $O_i$ ($B_r \times d$ each), and so does the score tile $S_{ij}$ ($B_r \times B_c$). Read this way, the 4 in $M/4d$ leaves room for four blocks of $d$-wide rows, $K_j$, $V_j$, $Q_i$, $O_i$. The cap $B_r \le d$ keeps the tile at $B_r B_c \le d \cdot M/4d = M/4$. Without the cap, a small head dimension would make $M/4d$ huge and the square tile would overflow.

::widget fixture:flashattention--io-ledger | the bar under the plot is what sits in SRAM during one inner step; push d down to 32 and watch the cap bind: B_r stops at d and S_ij turns from a square into a wide rectangle

### The loop order, and a warning about pictures you may have seen
Read lines 5–8 again. The **outer** loop walks over the K/V blocks, and each $K_j, V_j$ is loaded once and stays on chip while the **inner** loop sweeps over *every* Q block. So each $Q_i$ is loaded $T_c$ times, once per K/V block. Its output block $O_i$ and statistics $\ell_i, m_i$ are read and written back to HBM on every one of those visits, because the next visit to row block $i$ comes only after all the other row blocks have had their turn with $K_j$. Figure 1's arrows say the same thing: red "outer loop" arrows run along $K^\top$ and $V$, blue "inner loop" arrows along $Q$ and the output.

::animation fixture:flashattention--alg1-loop | the paper's order on a 4 × 3 grid of tiles (T_r = 4, T_c = 3): when the outer loop moves to K₂, V₂, Q₁ and O₁ cross HBM a second time; at the end K and V show 1 load each, every Q_i and O_i show 3, and "S written to HBM" stays at 0

::note warning | This is the reverse of the picture most lectures draw. L5's slide 54, and the animation below, hold one Q block on chip and stream the K and V blocks past it, so each output block is finished before the next starts. That diagram comes from the FlashAttention-2 paper (Dao, 2023). The paper we are reading prints K/V outer and Q inner, which is why $O_i$ round-trips through HBM $T_c$ times here. Both orders compute the same exact result, and both keep $S$ off HBM. They differ in which matrices are re-read. Hold on to the printed order: the IO count in Theorem 2 is read straight off it.
::animation fixture:flash-forward-tiles | for contrast, the lecture's order: one Q block stays put while K tiles stream past; HBM traffic grows by one K tile per step and never by an S tile

::predict fa-forward-loop-structure

Take the predict's setting, $d = 128$, $M = 32{,}768$, $N = 2048$. Then $B_c = B_r = 64$ and there are 32 K/V blocks and 32 Q blocks. The K/V blocks are loaded 32 times in total, once each. The Q blocks are loaded 32 × 32 = 1,024 times, and every one of those loads also reads and writes an $O_i$. Now keep $M$ and shrink the head to $d = 32$: $\lceil M/4d \rceil = 256$ exceeds $d$, so the cap binds. $B_c = 256$ but $B_r = 32$, and the score tile is 32 × 256 = 8,192 elements instead of a 64 × 64 square.

::kp fa-forward-loop-structure

## Line 12: how the output is corrected, and why the result is exact {#line12}
source: fa:alg1:L12 · fa:thm1 · fa:§C:thm1-proof · fa:§B.5

The statistics $m$ and $\ell$ are only half the job. The output rows themselves are sums of $P_{ij} V_j$ over all key blocks, and every partial sum was weighted by the probabilities known *at the time*. Line 12 repairs that every time a new block arrives:

$$ O_i \leftarrow \mathrm{diag}(\ell_i^{new})^{-1}\Big(\mathrm{diag}(\ell_i)\, e^{m_i - m_i^{new}}\, O_i \;+\; e^{\tilde m_{ij} - m_i^{new}}\, \tilde P_{ij} V_j\Big) $$

Here $\mathrm{diag}(\ell)$ is just "multiply row $r$ by $\ell_r$". Read the formula from the inside out. The stored $O_i$ is already *normalized*, a proper weighted average, divided by the old sum $\ell_i$. So the update:
- multiplies by $\ell_i$ to recover the unnormalized numerator, $\sum e^{s - m_i} v$ over the keys seen so far;
- rescales it by $e^{m_i - m_i^{new}}$ to the new max, exactly as line 11 does for $\ell$;
- adds the new block's contribution $\tilde P_{ij} V_j$, itself rescaled from the block's max $\tilde m_{ij}$ to $m_i^{new}$;
- divides by the new sum $\ell_i^{new}$, so the stored value is again a normalized average.

One row with $d = 1$, continuing the softmax example ($m_i = 1.0$, $\ell_i = 2.0$, new block max 3.0):

::worked fa-output-rescale-update
::predict fa-output-rescale-update

The earlier keys' share of this output row fell from all of it to 0.1353 / 1.0353 ≈ 13%, because the new block holds a much larger score. If an implementation forgets $\mathrm{diag}(\ell_i)$, it divides the old part by a normalizer twice and undercounts it by exactly $\ell_i$. If it forgets the factor on $\tilde P_{ij} V_j$, the new block is overweighted whenever the running max is larger than the block's own.

::note aside | Keeping $O_i$ normalized at every step is a choice, not a necessity. Accumulating the unnormalized numerator and dividing by $\ell$ once at the very end is also exact and saves work. That is what FlashAttention-2 does ([L5](#/read/lecture_05), slide 54). In Algorithm 1 as printed, though, $O_i$ goes back to HBM after every inner step, and storing it normalized means whatever is in HBM is always a valid partial answer.

### Theorem 1: exact, with the same FLOPs and linear extra memory
**Theorem 1** states that Algorithm 1 returns $O = \mathrm{softmax}(QK^\top)V$ with $O(N^2 d)$ FLOPs and $O(N)$ additional memory beyond the inputs and output. The FLOP count is direct. Each inner step does two $B_r \times B_c \times d$ matmuls (lines 9 and 12), and there are $T_r T_c = (N/B_r)(N/B_c)$ inner steps, so the block sizes cancel and the total is $O(N^2 d)$, the same order as standard attention. The extra memory is the statistics $\ell$ and $m$, $2N$ numbers.

Correctness is an induction on the outer loop. Claim: after outer iteration $j$, HBM holds the exact row max, row sum and attention output over the first $j$ key blocks. Line 2's initialization is the claim for zero blocks. Line 11 extends the statistics by one block (previous section). In line 12, $\mathrm{diag}(\ell^{(j)})$ cancels the stored normalization and $e^{m^{(j)} - m^{(j+1)}}$ moves the old numerator to the new max, so old and new parts add up to the unnormalized output over the first $j+1$ key blocks, and $\ell^{(j+1)}$ is exactly its normalizer. At $j = T_c$, $O = \mathrm{softmax}(QK^\top)V$. Every rescale is an exact identity; nothing is approximated.

::note why | Appendix B.5 compares this with Rabe and Staats (2021), who also tiled attention to avoid storing the attention matrix. They aimed at *memory footprint* and run at about the speed of standard attention or slightly slower; FlashAttention aims at *memory accesses* and runs 2–4× faster. They also keep a temporary output per block and combine them at the end, where line 12's incremental update needs only one copy of the output.
::kp fa-output-rescale-update

## Recomputation: what does the backward pass do without the attention matrix? {#backward}
source: fa:§3.1 (Recomputation) · fa:§B.2 · fa:§B.4 · fa:alg4 · fa:thm5

Training needs gradients, and the standard backward pass (the paper's Algorithm 3) reads the saved $N \times N$ matrix $P$ and writes and reads two more, $dP$ and $dS$. FlashAttention never stored $P$. It saves only the output $O$, the statistics $\ell$ and $m$, and the random-number-generator state used for dropout. Then, in the backward pass, it rebuilds each block of $P$ on chip from $Q_i$ and $K_j$:

$$ P_{ij} = \mathrm{diag}(\ell_i)^{-1} \exp(S_{ij} - m_i), \qquad S_{ij} = Q_i K_j^\top $$

Because the final $m_i$ and $\ell_i$ are known by now, this needs no rescaling: one matmul and one exponential per tile. The dropout mask, $N \times N$ too, is regenerated from the saved generator state instead of being stored.

That handles $P$. The gradient of the softmax needs one more thing. Write $dX$ for the gradient of the loss with respect to $X$, and lower-case $o_i$, $do_i$, $v_j$ for rows. Back through $O = PV$ gives $dV = P^\top dO$ and $dP = dO\, V^\top$. Back through the row-wise softmax gives $dS_{ij} = P_{ij}(dP_{ij} - D_i)$, where $D_i = \sum_j P_{ij}\, dP_{ij}$ is a sum over the *whole* row, the same coupling problem as the forward softmax. Appendix B's Eq. 4 removes it:

::worked fa-backward-recomputation

So $D_i = do_i^\top o_i$ is a dot product of length $d$ between two rows that are already saved, $O$ from the forward pass and $dO$ from upstream. The backward pass never needs a length-$N$ row of $P$ or $dP$ on chip. The paper names this its second observation about the backward pass. It is easy to overlook, and without it the block-wise backward would break as soon as a row no longer fits in SRAM.

**Algorithm 4** then has the same shape as Algorithm 1. The outer loop loads $K_j$, $V_j$ and accumulates $dK_j$, $dV_j$ on chip. The inner loop loads $Q_i$, $O_i$, $dO_i$, $dQ_i$, $\ell_i$, $m_i$, recomputes $P_{ij}$, forms $dS_{ij}$, and adds its contribution to $dQ_i$ in HBM. **Theorem 5** gives the backward pass the same IO counts as the forward: $\Theta(Nd + N^2)$ for standard attention, $\Theta(N^2 d^2 M^{-1})$ for FlashAttention.

::predict fa-backward-recomputation
::widget fixture:flashattention--saved-state | saved state P (N²) against O, ℓ, m (Nd + 2N): the ratio is about N/(d + 2); below, the backward HBM count stays lower even though every P_ij is recomputed, and Fig. 2's measured bars sit underneath

The paper calls this "a form of selective gradient checkpointing": saving less and recomputing during the backward pass (see [L2's activation checkpointing](#/read/lecture_02)). Ordinary checkpointing trades speed for memory. Here the recomputation happens on chip, from blocks that were being loaded anyway, and it replaces reading $P$ from HBM. Since the pass is memory-bound, it is a net win on both counts. That is where Fig. 2's extra 13% of FLOPs comes from, and why the runtime still drops 5.7×.

::note skip | [L5](#/read/lecture_05) skips the backward pass entirely, saying only that FlashAttention recomputes tile by tile. Everything in this section is from the paper's Appendix B.
::note slip | Appendix B.4 says the backward pass "performs $O(N^2)$ FLOPs". By the same count as Theorem 1 (several $B_r \times B_c \times d$ matmuls per inner step), it is $O(N^2 d)$; the $d$ was dropped.
::kp fa-backward-recomputation

## Theorem 2: how many HBM accesses does FlashAttention make? {#io}
source: fa:§3.2 · fa:thm2 · fa:§C:thm2-proof · Fig. 2 (middle) · fa:§E.5

Algorithm 1's traffic can be counted straight off its loops. **Theorem 2**: with SRAM of size $M$, $d \le M \le Nd$,

$$ \text{standard attention: } \Theta(Nd + N^2), \qquad \text{FlashAttention: } \Theta\!\left(\frac{N^2 d^2}{M}\right) \text{ HBM accesses.} $$

The proof has two counts. $K$ and $V$ are loaded once each, $2Nd$ elements, because the outer loop visits each block once. $Q$ and $O$ are swept once per outer iteration, $T_c$ passes of $\Theta(Nd)$ elements each. The number of passes comes from the block size: K/V blocks of $B_c \times d$ must fit in $M$, so $B_c = \Theta(M/d)$ and $T_c = N/B_c = \Theta(Nd/M)$. Multiplying:

::worked fa-io-complexity

No $N^2$ term appears, because $S$ never reaches HBM. The $N^2$ in the result comes from re-reading $Q$ and $O$, and it is divided by $M/d^2$. With the proof's constants the two quadratic terms compare as $4N^2$ against $8N^2 d^2/M$, a ratio of $M/2d^2$. For $d = 64$ that is 6 at $M = 49{,}152$ (96 KB of fp16) and 12 at $M = 98{,}304$ (192 KB). Hence the paper's condition: for typical $d$ (64–128) and $M$, $d^2$ is many times smaller than $M$, and FlashAttention makes many times fewer HBM accesses. The measured 9.2× in Fig. 2 is the same order.

::predict fa-io-complexity

Three consequences are worth seeing move.
- **Traffic is still quadratic in N.** For a fixed chip, doubling $N$ quadruples FlashAttention's accesses too. What is linear in $N$ is its *extra memory* (Theorem 1), not its traffic. It reaches $\Theta(Nd)$ only at $M = \Theta(Nd)$, when all of $K$ and $V$ fit on chip at once.
- **Bigger SRAM, fewer passes.** Doubling $M$ doubles $B_c$, halves $T_c$, and halves the dominant term. Standard attention's count never depends on $M$.
- **Head dimension enters squared.** Doubling $d$ halves the number of rows per block (twice as many passes) and doubles the size of each pass, so traffic grows 4×. The appendix's hardware runs agree: at $d = 128$ on an A100 the speedup is smaller (up to 3× with a causal mask), and on a T4, whose SRAM is smaller, block sizes must shrink and the speedup drops, which the paper says "matches the IO complexity analysis".

::widget fixture:flashattention--io-ledger | pin a configuration, then double M (the Q/O term halves) or double d (it grows ×4); drag SRAM down to 8 KB at d = 64 and the readout turns red: with d² not ≪ M, FlashAttention moves about 2× more than standard
::animation fixture:flashattention--alg1-loop | the counters are the proof: K and V stop at 1 load each, Q and O climb to T_c = 3, and the S counter never leaves 0

The proof assumes the block size is as large as SRAM allows. Fig. 2 (middle) checks what happens when it is not. It runs the forward pass at block sizes 64 to 512 in the same GPT-2 medium setting.

::figure fa:fig2 | Fig. 2 (middle), the right-hand plot on slide 50: as the block size B_c grows from 64 to 512, HBM accesses fall steadily (fewer passes over Q); runtime falls with them up to 256 and then flattens

Fewer passes mean fewer HBM accesses and a faster forward pass. Beyond a block size of 256 the runtime is bottlenecked by other factors, such as arithmetic, and still larger blocks would not fit in SRAM anyway. IO-awareness gets the kernel off the memory roof. After that, the compute roof takes over ([L2's roofline](#/read/lecture_02)).

::note warning | The intro writes the result as $O(N^2d^2M^{-1})$ against $\Omega(Nd + N^2)$ for standard attention; Theorem 2 states both as $\Theta$. The $\Theta$ version is the one proved, and it counts elements with the constants hidden.
::kp fa-io-complexity

## Can anyone do better? The lower bound {#lower-bound}
source: fa:prop3 · fa:§3.2 · fa:§C:prop3-proof

The abstract says FlashAttention is "optimal for a range of SRAM sizes". **Proposition 3** is the precise statement: no algorithm computes exact attention with $o(N^2 d^2 M^{-1})$ HBM accesses for *all* $M$ in $[d, Nd]$. (Little-$o$ means asymptotically smaller, by more than a constant factor.)

::predict fa-lower-bound
::worked fa-lower-bound

The whole proof is one value of $M$. At $M = Nd$, FlashAttention's count is $N^2d^2/Nd = Nd$, and any exact algorithm must at least read $Q$, $K$, $V$ and write $O$, $Nd$ elements each. There FlashAttention already sits on the trivial floor, so no algorithm beats it everywhere.

Read the quantifier carefully. The proposition says nothing about a fixed, small $M$, which is the realistic case: a cleverer algorithm might beat $\Theta(N^2d^2/M)$ at $M$ = 100K elements without contradicting anything. The authors say so: lower bounds over a subrange of $M$ are common in the streaming-algorithms literature, and they leave per-$M$ (parameterized) bounds "as exciting future work".

::widget fixture:flashattention--io-ledger | at N = 1024, d = 64, set SRAM to 128 KB of fp16, which makes M = Nd, the right end of the axis: the FlashAttention curve comes down to 2.5× the dashed 4Nd floor, T_c = 4; further left it sits far above the floor, where Proposition 3 says nothing
::note warning | §5 goes further than Proposition 3 and calls the implementation "optimal within constants" on a single GPU. What is proved is the weaker, all-$M$ statement above.
::kp fa-lower-bound

## Block-sparse FlashAttention: skipping blocks entirely {#block-sparse}
source: fa:§3.3 · fa:prop4 · fa:alg5 · Fig. 2 (right)

Once attention is computed block by block, skipping blocks is easy. Block-sparse attention fixes a mask in advance and sets masked scores to $-\infty$ before the softmax, so they get zero probability:

$$ P = \mathrm{softmax}\big(S \odot \mathbb{1}_{\tilde M}\big), \qquad (S \odot \mathbb{1}_{\tilde M})_{kl} = \begin{cases} S_{kl} & \tilde M_{kl} = 1 \\ -\infty & \tilde M_{kl} = 0 \end{cases} $$

The mask must have **block form**: it is constant on each $B_r \times B_c$ tile, so it is described by a small block mask $\mathbf{M} \in \{0,1\}^{N/B_r \times N/B_c}$. (The paper reuses the letter M here; this is the mask, not the SRAM size.) **Algorithm 5** is Algorithm 1 with one change: inner iterations whose block is zero are skipped, with no loads and no compute.

**Proposition 4**: with $s$ the fraction of nonzero blocks,

$$ \Theta\!\left(Nd + \frac{N^2 d^2}{M}\, s\right) \text{ HBM accesses.} $$

::worked fa-block-sparse

Only the pass term scales with $s$. The $Nd$ term stays written out because $O$ must still be written however sparse the mask. Theorem 2 could absorb it, Proposition 4 cannot. So keeping 1 block in 4 cuts traffic by less than 4×. If the density falls as $N$ grows, the exponent changes. With $s = N^{-1/2}$ the sparse term grows like $N\sqrt N$, and with $s = (\log N)/N$ like $N \log N$, both schedules taken from earlier sparse-attention work. For its experiments the paper uses a fixed **butterfly** pattern, which earlier work showed can approximate arbitrary sparsity.

::predict fa-block-sparse
::widget fixture:flashattention--block-sparse | the saving stays below 1/s because the 2Nd term does not shrink; switch the schedule to s = N^−1/2 and the sparse curve's slope drops from N² to N√N
::figure fa:fig2 | Fig. 2 (right): at sequence length 4K, block-sparse FlashAttention's runtime falls in proportion to the sparsity

::note warning | Block-sparse FlashAttention is *not* exact. It computes a different function, attention under a fixed mask, and the abstract calls it "an approximate attention algorithm". Its accuracy has to be measured separately, as the experiments do.
::kp fa-block-sparse

## Faster training of the same models {#faster}
source: fa:§4.1 · Tables 1–3 · fa:§E.1–E.3

The experiments test two claims: the same models train faster, and longer context gives better models. Speed first. All runs are on 8 A100s.

**BERT-large** (sequence length 512), trained from the MLPerf initialization to 72.0% masked-language-modelling accuracy, averaged over 10 runs: 17.4 ± 1.4 minutes against 20.0 ± 1.5 for Nvidia's MLPerf 1.1 record, 20.0 / 17.4 ≈ 1.15, so 15% faster. Table 1 is the only result in the paper that reports run-to-run spread.

**GPT-2** on OpenWebText, sequence length 1K, Table 2 as printed:

| Model | Implementation | Perplexity | Training time (speedup) |
|---|---|---|---|
| GPT-2 small | HuggingFace | 18.2 | 9.5 days (1.0×) |
| GPT-2 small | Megatron-LM | 18.2 | 4.7 days (2.0×) |
| GPT-2 small | FlashAttention | 18.2 | 2.7 days (3.5×) |
| GPT-2 medium | HuggingFace | 14.2 | 21.0 days (1.0×) |
| GPT-2 medium | Megatron-LM | 14.3 | 11.5 days (1.8×) |
| GPT-2 medium | FlashAttention | 14.3 | 6.9 days (3.0×) |

Against Megatron-LM, already an optimized implementation, the gain is 4.7 / 2.7 ≈ 1.7× (small) and 11.5 / 6.9 ≈ 1.7× (medium). The perplexity column is the point. FlashAttention matches the other implementations "as we do not change the model definition", and Appendix E's validation curves overlap.

::figure fa:table2 | read the perplexity column before the time column: the kernel changes how long training takes, not what is learned

::note slip | The headline numbers drift between places. The abstract says "3× speedup on GPT-2", while the small-model row prints 3.5× (3× is the medium row). §4.1's text says 1.7× over Megatron, which matches both rows, but the §4 summary bullet says 1.8×, which is Megatron's own speedup over HuggingFace on GPT-2 medium. And "the same perplexity" is exact for small (18.2 for all three) but not for medium, where HuggingFace prints 14.2 against 14.3.

**Long-Range Arena** (sequence lengths 1K–4K): FlashAttention runs 2.4× faster than standard attention at 59.8 average accuracy against 59.3. Block-sparse FlashAttention runs 2.8× at 59.6. The approximate baselines are slower or worse. Linformer reaches 2.5× but drops to 54.9; Performer is 1.8× at 58.9.

### Why "exact" is what makes these numbers easy to read
An exact kernel is a drop-in replacement. It computes the same function, so the same checkpoint, the same gradients and the same perplexity at the same context follow (Table 2), and its speedup composes with everything else. Linformer's 2.5× is a different kind of number: it comes from changing the model, and its 4.4-point accuracy loss has to be re-measured on every task. FlashAttention's 59.8 against 59.3 is the same model run twice.

::predict supp-exact-vs-approximate
::figure fa:table3 | compare the speedup column with the average column: only the exact kernel gets its speedup without moving the accuracy
::kp supp-exact-vs-approximate

## Better models from longer context, and what the benchmarks show {#longer}
source: fa:§4.2 · fa:§4.3 · Tables 4–6 · Fig. 3 · fa:§E.5 · fa:§5

Speed is also a budget. Spend it on context instead:

| GPT-2 small | Context | Perplexity | Training time (speedup over Megatron 1K) |
|---|---|---|---|
| Megatron-LM | 1K | 18.2 | 4.7 days (1.0×) |
| FlashAttention | 1K | 18.2 | 2.7 days (1.7×) |
| FlashAttention | 2K | 17.6 | 3.0 days (1.6×) |
| FlashAttention | 4K | 17.5 | 3.6 days (1.3×) |

At 4K context, 4× longer, FlashAttention still trains 4.7 / 3.6 ≈ 1.3× faster than Megatron at 1K and reaches 0.7 lower perplexity. Because the 1K rows match exactly, the gain is attributable to context alone.

::predict fa-experimental-claims

**Long documents.** A pretrained RoBERTa is fine-tuned at longer sequence lengths on MIMIC-III (ICU discharge summaries) and ECtHR (European Court of Human Rights cases). Micro-F1 rises from 52.8 at 512 tokens to 57.1 at 16K on MIMIC (+4.3) and from 72.2 to 80.7 at 8K on ECtHR (+8.5). The abstract's "6.4 points of lift" is the average of the two, (4.3 + 8.5) / 2. The curves are not monotone, though. MIMIC dips to 50.7 at 1K before rising, and ECtHR falls back to 79.2 at 16K. The paper suggests distribution shift in document length as the cause.

::figure fa:table5 | follow each row left to right: both datasets improve overall, but neither improves at every step
::note slip | The text says "Table 6 shows" the 4.3 and 8.5 point gains; the numbers are in Table 5.

**Path-X and Path-256.** These Long-Range Arena tasks ask whether two points in a 128 × 128 (or 256 × 256) black-and-white image are connected by a path, with the image fed in one pixel at a time: sequences of 16K and 64K. Every earlier Transformer either ran out of memory or scored at chance. FlashAttention reaches 61.4% on Path-X. Block-sparse FlashAttention scales to 64K and reaches 63.1% on Path-256, where dense FlashAttention does not get past chance. These were the first better-than-chance Transformers on either task.

**The attention kernel alone** (Fig. 3, one A100 40 GB, with dropout and a padding mask). Runtime still grows quadratically in $N$, but FlashAttention is up to 3× faster than PyTorch attention at common lengths (128–2K). The approximate methods grow linearly, and their runtimes cross FlashAttention's somewhere between 512 and 1024. Block-sparse FlashAttention is faster than every implementation tested at every length. Memory grows linearly in $N$: up to 20× less than exact baselines, and at 64K, where everything except Linformer has run out of memory, still 2× less than Linformer.

::figure fa:fig3 | left: forward + backward runtime against sequence length, where the approximate methods cross FlashAttention between 512 and 1024; right: memory, linear for FlashAttention

**Other hardware** (Appendix E.5). The speedup depends on the gap the paper exploits. On an A100 it is generally 2–4×, more with dropout and masking because more operations get fused. On an RTX 3090, whose HBM is slower (about 900 GB/s against 1.5 TB/s), it is 2.5–4.5×. On a T4 the smaller SRAM forces smaller blocks and the speedup shrinks. Each case is the IO model at work: the cheaper HBM traffic is relative to compute, the less there is to save.

### What the paper says it cannot do
- **Every variant needs a hand-written CUDA kernel.** That is far lower-level than PyTorch, takes significant engineering, and may not transfer across GPU architectures. The authors call for compiling high-level attention code to IO-aware kernels, as Halide does for image processing.
- **The analysis covers one GPU.** Attention split across GPUs adds another memory level, the other GPUs' HBM.
- **Beyond attention** the same thinking could apply to any memory-bound layer. Appendix D names sparse MLPs, which are often memory-bound, and kernel methods, whose $N \times N$ kernel matrix is, like $QK^\top$, a function of low-rank inputs and can be recomputed block by block.

The first limitation is where the course goes next. [L6](#/read/lecture_06) writes tiled kernels in Triton, a Python-level language that compiles to GPU code, and the assignment has you implement the tiled softmax and forward pass yourself. Either loop order is correct; know which one you are writing, because it decides what gets re-read from HBM.

::kp fa-experimental-claims
