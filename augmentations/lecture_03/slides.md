# Lecture 3 · augmented slide copy

The copy below is intentionally short. The animation carries the topology;
the slide text names the invariant, the tradeoff, and a useful check.

## L03-PRE-POST-NORM · Pre-norm keeps an identity path

**Claim.** In a pre-norm block, the residual stream can travel from input to
output through an exact identity path. The normalization operation lives only
on the learned branch.

**Compare.** Post-norm computes `LN(x + F(x))`, so both the residual and learned
branch cross LayerNorm after the merge. Pre-norm computes `x + F(LN(x))`, so
the main residual route remains untouched.

**Check yourself.** Trace the backward path from `x'` to `x` without entering
`F`. Which topology still has a derivative equal to one along that route?

**Caveat.** “Improves gradient behavior” is not “always gives better final
quality.” Initialization, depth scaling, and newer norm variants still matter.

## L03-RMSNORM / L03-NORM-RUNTIME / L03-NO-BIAS · Count movement, not only FLOPs

**Source.** Official Lecture 3, slides 14–19. Slide 14 gives the LayerNorm to
RMSNorm simplification; slides 15–17 ask why it can matter for runtime despite
the small FLOP share; slides 18–19 broaden the same engineering logic to bias
terms.

**Normalization work.** LayerNorm computes a mean and variance over the model
dimension, then applies learned scale `γ` and shift `β`. RMSNorm does not center
the vector: `RMSNorm(x) = x / sqrt(mean(x²) + ε) × γ`. In this course-level
accounting, that changes mean-plus-variance work to one second-moment reduction
and changes `γ + β` (2d learned values) to `γ` alone (d learned values).

**Runtime lesson.** Matrix multiplications are the vast majority of Transformer
FLOPs, so removing a small amount of normalization arithmetic does not explain
runtime by FLOP count alone. Normalization still reads activations, performs a
reduction, and writes activations. The course's point is therefore `FLOPs ≠
runtime`: data movement and implementation details can make a low-FLOP operation
visible in wall-clock time.

**Bias accounting.** For the ungated example
`FFN(x) = σ(xW₁ + b₁)W₂ + b₂`, dropping both biases removes `d_ff + d_model`
bias values per layer and the corresponding logical bias loads/adds. Actual
kernel savings depend on fusion, tensor shape, dtype, and hardware. Optimizer
state savings also depend on the optimizer, so they are not folded into this
parameter count.

**Check yourself.** For `d_model = 4096` and `d_ff = 11008`, how many bias
values does one ungated FFN omit? Why is that parameter count not, by itself, a
prediction of the speedup?

**Caveat.** The deck cites memory and optimization stability as reasons for
dropping bias and reports RMSNorm runtime/performance evidence from specific
papers. It does not establish a universal quality law. Verify both model quality
and end-to-end wall-clock behavior in the actual implementation.

## L03-KV-CACHE / L03-MQA / L03-GQA · Incremental decoding makes KV traffic the bottleneck

**Source.** Official Lecture 3, slides 58–63. Slides 58–60 contrast
full-sequence attention with step-by-step generation and derive the incremental
arithmetic-intensity problem; slide 61 introduces MQA; slides 62–63 introduce
GQA and qualify the quality tradeoff.

**Incremental decoding.** Generation cannot parallelize future tokens. At each
step it adds the new key and value, then rereads prior keys and values from the
growing KV cache. The course writes the total incremental attention work as
`O(b n d²)` arithmetic operations and `O(b n² d + n d²)` memory accesses, so
the arithmetic intensity is `O((n/d + 1/b)⁻¹)`. This is unfavorable for small
batches or long sequences because repeated cache traffic grows relative to the
useful arithmetic. Here `b` is batch size, `n` sequence length, and `d` hidden
dimension. (Official slides 59–60.)

**KV-sharing response.** Query-head count can stay fixed while key/value-head
count shrinks. The KV cache per generated token scales with `n_kv_heads`, not
`n_q_heads`. MQA shares one K/V stream across all query heads; the course gives
its memory accesses as `O(b n d + b n² k + n d²)` and arithmetic intensity as
`O((1/d + n/(d h) + 1/b)⁻¹)`, with `h` query heads and `k = d/h`. GQA keeps
more than one K/V group, providing a knob between sharing and expressiveness.
(Official slides 61–62.)

**Course sequence.** Incremental decoding must repeatedly read the stored KV
cache, which makes memory traffic the immediate problem. MQA first collapses
the cache to one shared K/V stream; GQA then turns sharing into a knob by using
more than one K/V group when extra capacity is worth the traffic.

**Resource equation.** Ignoring batch and sequence dimensions,
`KV bytes/token/layer = 2 × n_kv_heads × d_head × bytes/value`. The leading 2
stores both keys and values. Multiply by `n_layers` for the model-wide cache
per token.

**Check yourself.** With 8 query heads and 2 KV heads, how many query heads
share each KV stream? How much smaller is the head-dependent part of the cache
than 8-head MHA?

**Caveat.** KV sharing reduces memory movement during decoding; it does not make
the attention-score matrix disappear. The course reports a small perplexity hit
for MQA in one cited result and low or no hit for GQA in another. Treat this as
empirical evidence from those studies, not as a guarantee for every model or
training setting. (Official slide 63.)

## L03-SLIDING-WINDOW · Keep the causal band

**Claim.** A causal window of width `w` lets query `t` read only keys
`max(0, t-w+1) … t`. The active cells form a diagonal band inside the causal
triangle.

**Concrete frame.** For 12 tokens, full causal attention uses 78 cells. A
width-4 causal window uses `1 + 2 + 3 + 9×4 = 42` cells, 46% fewer in this
small example.

**Scaling.** Full causal attention grows as `O(n²)`; fixed-width local attention
grows as `O(nw)`. The saving becomes more pronounced as context length grows.

**Practical pattern.** A local window alone cannot directly connect distant
tokens. The course's next step is therefore to interleave mostly local layers
with periodic full-attention layers: bounded work in most layers, plus explicit
long-range mixing every few layers.
