---
title: RoPE · Rotary position embedding, read through
minutes: 35
---
In 2021 Jianlin Su and colleagues at Zhuiyi Technology proposed a way to give a transformer word order: rotate each query and key by an angle proportional to its position. The attention score then depends only on how far apart two tokens are. This read-through follows the RoFormer paper in its own order: the earlier position encodings, the requirement RoPE starts from, the 2D derivation, the general block-diagonal form, the two properties the paper claims (long-term decay and linear attention), the cheap elementwise implementation, and the experiments. Afterwards you can rotate a query by hand, prove that only the offset survives, implement it in two lines, and say what the paper's evidence does and does not show.

## Why does attention need position at all? {#why}
source: rope:§1 · rope:§2.1 · rope:eq1 · rope:eq2 · lecture_03.pdf p30

Self-attention compares tokens by inner products. Token $m$'s query $q_m$ meets token $n$'s key $k_n$, and the score is $q_m^\top k_n$. Nothing in that formula knows where the tokens sit in the sentence. Shuffle the input words and every score moves with its word, unchanged. On its own, attention treats a sentence as a bag of words, so "dog bites man" and "man bites dog" look the same.

The paper's §2.1 sets up the notation the rest of it uses. A sequence of $N$ tokens has word embeddings $x_1, \dots, x_N \in \mathbb{R}^d$ that carry no position. Position enters through three functions that build the query, key and value from a token's embedding *and* its position:

$$ q_m = f_q(x_m, m), \qquad k_n = f_k(x_n, n), \qquad v_n = f_v(x_n, n) \qquad \text{(Eq. 1)} $$

Attention then does what it always does (Eq. 2): softmax over the scaled scores, and a weighted sum of the values.

$$ a_{m,n} = \frac{\exp\!\big(q_m^\top k_n / \sqrt{d}\big)}{\sum_{j=1}^{N} \exp\!\big(q_m^\top k_j / \sqrt{d}\big)}, \qquad o_m = \sum_{n=1}^{N} a_{m,n}\, v_n $$

So every position-encoding scheme is a choice of $f_q$, $f_k$, $f_v$. Hold on to one detail of Eq. 1: $f_q$ sees only the query's own position $m$, and $f_k$ only the key's own position $n$. That restriction is what lets a decoder compute each key once, cache it, and reuse it for every later query.

::slide lecture_03:30 | the four families the course lists: sine embeddings added to the input (original transformer), learned absolute embeddings (GPT-1/2/3, OPT), relative terms added inside attention (T5, Gopher, Chinchilla), and RoPE (GPT-J, PaLM, LLaMA, most 2024+ models)

The introduction sorts earlier work into two camps. **Absolute** encodings add a vector for position $i$ to the word embedding, either a fixed sinusoid (Vaswani et al., 2017) or a learned vector per position (BERT, GPT). **Relative** encodings (Shaw et al., Transformer-XL, T5, DeBERTa and others) put the offset between two tokens into the attention computation itself. The paper's complaint about both camps is the same: they *add* position to the context representation. It claims, and later argues, that this makes them unsuitable for linear self-attention.

RoPE instead encodes the absolute position with a rotation matrix, and the relative position then appears by itself inside the inner product. The paper advertises three properties:
- **sequence-length flexibility**: a rotation is defined for any position, so there is no table of positions that runs out;
- **long-term decay**: the dependency between tokens is supposed to weaken as they move apart;
- **linear attention**: unlike earlier relative schemes, RoPE can be combined with kernelized (linear-cost) attention.

The second and third get their own sections below. The first is claimed but, as the experiments section shows, never tested.

::note aside | The course reaches RoPE in [L3](#/read/lecture_03), slides 30–35, as the one architecture component "still in flux". The professor says it "came out of nowhere": a blog post and a little-known paper that GPT-J picked up, and that most models since 2024 now use. The paper's own README points to the original Chinese blog post on kexue.fm and to EleutherAI's write-up.

## How did earlier methods add position? {#additive}
source: rope:§2.2 · rope:§2.3 · rope:eq3–eq10

The baseline is the original transformer's choice of Eq. 1: add a position vector $p_i$ to the word embedding, then project.

$$ f_{t}(x_i, i) = W_{t}\,(x_i + p_i), \qquad t \in \{q, k, v\} \qquad \text{(Eq. 3)} $$

The vector $p_i$ is either learned, one vector per position up to a maximum length $L$, or the fixed sinusoid of Vaswani et al.:

$$ p_{i,2t} = \sin\!\big(i / 10000^{2t/d}\big), \qquad p_{i,2t+1} = \cos\!\big(i / 10000^{2t/d}\big) \qquad \text{(Eq. 4)} $$

Each pair of coordinates $(2t, 2t+1)$ holds a sine and a cosine at frequency $10000^{-2t/d}$. Remember these frequencies; RoPE reuses exactly them.

::note slip | Eq. 4 as printed writes $\sin(k/10000^{2t/d})$ with a $k$ that is never defined; the argument should be the position $i$ of $p_i$.

Now expand the attention score that Eq. 3 produces. The query is $W_q(x_m + p_m)$ and the key $W_k(x_n + p_n)$, and a bilinear form distributes over both sums:

$$ q_m^\top k_n = x_m^\top W_q^\top W_k x_n + x_m^\top W_q^\top W_k p_n + p_m^\top W_q^\top W_k x_n + p_m^\top W_q^\top W_k p_n \qquad \text{(Eq. 6)} $$

::predict rope-position-goal

Four terms. Only the first, content against content, is free of position. The two middle ones pair one token's *content* with the *other* token's absolute position. The last pairs the two positions.

The interesting subtlety is the last term. For pure sinusoids, with $\theta_t = 10000^{-2t/d}$, the dot product $p_m \cdot p_n = \sum_t \cos\!\big((m-n)\,\theta_t\big)$ depends only on $m-n$: sinusoids are relative on their own. It is the matrix $W_q^\top W_k$ sitting between them that mixes frequencies and brings back terms in $m+n$.

§2.3 then walks through the relative schemes, and every one of them is an edit of Eq. 6:
- **Shaw et al. (2018)** (Eq. 5) leave the query alone and add a trainable vector $\tilde p_r$ to the key and value, indexed by the clipped offset $r = \mathrm{clip}(m-n, r_{\min}, r_{\max})$, on the hypothesis that exact distances stop mattering beyond some range.
- **Transformer-XL (Dai et al., 2019)** (Eq. 7) replaces $p_n$ by a sinusoid of the offset, $\tilde p_{m-n}$, replaces $p_m$ in the last two terms by two trainable vectors $u$ and $v$, and gives the position key its own projection $\widetilde W_k$.
- **T5 (Raffel et al., 2020)** (Eq. 8) keeps only the content term and adds a learned scalar bias: $q_m^\top k_n = x_m^\top W_q^\top W_k x_n + b_{i,j}$.
- **TUPE** (Eq. 9) drops the middle two terms after finding little correlation between words and absolute positions, and projects positions with their own matrices: $x_m^\top W_q^\top W_k x_n + p_m^\top U_q^\top U_k p_n + b_{i,j}$.
- **DeBERTa (He et al., 2020)** (Eq. 10) keeps the two middle terms with relative embeddings $\tilde p_{m-n}$ and drops the position-position term.

::note slip | The paper credits Eq. 9 to "Raffel et al. [2020]". The scheme with separate position projections $U_q, U_k$ is TUPE (Ke et al., 2020), whom the paper cites in the sentence before it.

The paper's diagnosis, which is the whole motivation for RoPE: all of these start from the additive decomposition of Eq. 3 and then delete, swap or re-parametrize terms of Eq. 6 until the score looks relative. RoPE asks instead what $f_q$ and $f_k$ would have to be for the score to come out relative *without* any editing.

## What exactly should the score depend on? {#requirement}
source: rope:§3.1 · rope:eq11 · lecture_03.pdf p31

§3.1 turns the wish into an equation. Keep Eq. 1's structure, a query built from $(x_m, m)$ alone and a key built from $(x_n, n)$ alone, and demand that their inner product depend on the two positions only through the offset:

$$ \big\langle f_q(x_m, m),\; f_k(x_n, n) \big\rangle = g(x_m, x_n, m - n) \qquad \text{(Eq. 11)} $$

The paper says it hopes the inner product "encodes position information only in the relative form". Read Eq. 11 as two conditions at once:
- **(a) an inner product of per-token vectors.** The score must be $\langle f_q, f_k \rangle$, with each side computed from one token.
- **(b) relative.** Absolute positions must cancel, leaving only $m - n$.

The second condition is a stance about language: the attention between "a" and "apple" should be the same at the start of a document and at its end. A common misreading is that a relative encoding means $f$ takes $m - n$ as input. It does not, and cannot: $f_q$ never sees $n$. Relativity is a property of the inner product, not of either function. That is what keeps keys cacheable and, later, makes linear attention possible.

::slide lecture_03:31 | the course states the same requirement as ⟨f(x, i), f(y, j)⟩ = g(x, y, i − j), then marks why each earlier family fails: sine has cross-terms that are not relative, absolute is not relative, relative "is not an inner product"

The slide compresses §2's two pages into three bullets, and each earlier family fails exactly one of the two conditions:

::worked rope-position-goal

Can anything pass both? Adding never will: any additive position term leaves cross terms, and a product of two absolute positions such as $m \cdot n$ can never be rewritten as a function of $m - n$. Multiplying has a chance, if the query's factor and the key's factor carry the position with *opposite signs*, so that the positions subtract when the two meet. Growing and shrinking scale factors would do that, but they blow up with position. What is needed is a unit-size factor with the opposite-sign property. In two dimensions that is a rotation.

::kp rope-position-goal

## Two facts about rotations the paper takes for granted {#rotations}
source: rope:§3.2.1 · rope:§3.2.2 · lecture_03.pdf p32-p33

The paper moves between complex numbers and rotation matrices in a single sentence and assumes two facts. Both are standard; here they are with numbers, because every later step uses them.

**Fact 1: a rotation keeps lengths and dot products, and rotations add angles.** The 2D rotation by $\alpha$ is

$$ R(\alpha) = \begin{pmatrix} \cos\alpha & -\sin\alpha \\ \sin\alpha & \cos\alpha \end{pmatrix}, \qquad R(\alpha)^\top R(\alpha) = I, \qquad R(\alpha) R(\beta) = R(\alpha + \beta), \qquad R(\alpha)^\top = R(-\alpha). $$

So rotating both vectors by the *same* angle leaves their dot product unchanged. Rotating them by *different* angles does change it, but only through the difference:

$$ \big(R(\alpha) q\big)^\top \big(R(\beta) k\big) = q^\top R(\alpha)^\top R(\beta)\, k = q^\top R(\beta - \alpha)\, k. $$

Example: $q = (2, 0)$ and $k = (0, 3)$ are perpendicular, so $q \cdot k = 0$. Rotate $q$ by 80° and $k$ by 50°. Now $q$ points at 80° and $k$ at 90° + 50° = 140°, a gap of 60°, and both lengths are unchanged: the dot product is $2 \cdot 3 \cdot \cos 60° = 3$. Only the 30° difference between the two rotations mattered.

::widget fixture:rope--rotation-pair | drag α and β: the dashed circles show the lengths never change, and R(α)q · R(β)k always equals q · R(β−α)k, so only the difference β − α moves the readout

**Fact 2: complex multiplication is rotation, and $\mathrm{Re}[z w^*]$ is a dot product.** Write the 2-vector $(a, b)$ as $z = a + ib$. Multiplying by $e^{i\varphi}$ rotates it by $\varphi$; the real and imaginary parts of $(a + ib)e^{i\varphi}$ are exactly $R(\varphi)$ applied to $(a, b)$. And for a second vector $w = c + id$, the conjugate $w^* = c - id$ turns the product into the dot product: $\mathrm{Re}[z w^*] = ac + bd$. Put the two together and the key's phase enters with a minus sign:

::worked supp-complex-rotation

Without the conjugate, $\mathrm{Re}[z w]$ is $ac - bd$, not a dot product, and the phases would add to $(m + n)\theta$. The conjugate on the key is the complex-number version of the opposite-sign coupling from the previous section. Tick the conjugate off in the widget above to watch the readout stop matching the dot product.

::kp supp-rotation-preserves-dot
::kp supp-complex-rotation

## The 2D case: rotate the projected vector by its position {#2d}
source: rope:§3.2.1 · rope:eq12 · rope:eq13 · rope:§3.4.1 · rope:eq20–eq33 · lecture_03.pdf p32

Start with head dimension $d = 2$, where a vector is a point in the plane, or a complex number. §3.2.1 states a solution to Eq. 11:

$$ f_q(x_m, m) = (W_q x_m)\, e^{im\theta}, \qquad f_k(x_n, n) = (W_k x_n)\, e^{in\theta}, \qquad g = \mathrm{Re}\big[(W_q x_m)(W_k x_n)^*\, e^{i(m-n)\theta}\big] \qquad \text{(Eq. 12)} $$

Here $\theta$ is a fixed non-zero constant. In words: project the word embedding as usual, then rotate the result by angle $m\theta$, where $m$ is the token's position. The query at position 5 is turned by $5\theta$; the key at position 2 by $2\theta$. The score $g$ follows from Fact 2: rotating by $m\theta$ and $n\theta$ and taking $\mathrm{Re}[\,\cdot\,(\cdot)^*]$ leaves the phase $e^{i(m-n)\theta}$. As a real matrix, the same thing reads

$$ f_{\{q,k\}}(x_m, m) = \begin{pmatrix} \cos m\theta & -\sin m\theta \\ \sin m\theta & \cos m\theta \end{pmatrix} W_{\{q,k\}}\, x_m \qquad \text{(Eq. 13)} $$

The paper's one-line summary is the name: rotate the "affine-transformed word embedding vector by amount of angle multiples of its position index".

::slide lecture_03:32 | the course's picture of the same idea: in "we know that", "we" is rotated by 0 positions and "know" by 1; in "of course we know", "we" by 2 and "know" by 3; both arrows moved, the angle between them did not

Try it on numbers before the general proof.

::predict rope-2d-solution

With $\theta = \pi/6$ (30°), a query $W_q x_m = (1, 0)$ at $m = 1$ turns by 30° to $(0.866, 0.5)$. A key $W_k x_n = (0.7, 0.7)$ at $n = 3$ turns by 90° to $(-0.7, 0.7)$. The score is $0.866 \times (-0.7) + 0.5 \times 0.7 = -0.606 + 0.35 = -0.256$. Swap the positions (query at 3, key at 1) and the score becomes $+0.956$: the net rotation is now $-60°$ instead of $+60°$. RoPE depends on the *sign* of the offset, so it can tell "the key came before the query" from "the key came after".

::widget fixture:rope-shared-shift | set m = 1 and n = 3 to reproduce −0.256, then raise the shared shift s: both arrows turn together and the readout stays put; tick "shift only q" and it changes

### Why a rotation, and not something else? (§3.4.1)
§3.2.1 calls Eq. 12 "a solution" and defers the derivation to §3.4.1, which tries to show that, under some extra assumptions, the solution is forced. It is the paper's most interesting piece of reasoning, so here it is, in the order the paper gives it. Two ingredients: write each function in polar form, length times $e^{i \cdot \text{angle}}$, so that multiplying complex numbers multiplies lengths and adds angles; and impose an initial condition (Eq. 22) that at position 0 nothing has been encoded yet, $q = f_q(x_q, 0)$ and $k = f_k(x_k, 0)$.

::worked rope-2d-solution

The result (Eq. 31, then Eq. 33 with $\gamma = 0$ and $q = W_q x_m$) is Eq. 12 again. The length of the vector carries no position at all; the only thing position adds is the angle $m\theta$. Note also what does not depend on content: Eq. 28 shows the position part of the angle, $\varphi(m)$, is the same for every word, so the rotation angle is never a function of $x_m$.

::note why | The derivation is weaker than "rotation is the only answer". Setting $m = n$ only gives the product $R_q(x_q, m)\,R_k(x_k, m) = \lVert q\rVert\lVert k\rVert$ (Eq. 26a); the paper then picks the "straightforward solution" in which each length is position-free (Eq. 27). And the step from a constant difference $\varphi(m+1) - \varphi(m)$ to $\varphi(m) = m\theta + \gamma$ uses integer positions. So the paper proves that rotation is sufficient, and necessary only under its chosen assumptions. The course (slide 32) gives just the picture and never argues necessity.

::kp rope-2d-solution

## From 2 to d dimensions: many rotations at different speeds {#general}
source: rope:§3.2.2 · rope:eq14 · rope:eq15 · rope:fig1 · rope:§3.3 · lecture_03.pdf p33-p34

Real heads have $d = 64$ or $128$ dimensions, and in high dimensions there are infinitely many rotations. §3.2.2 makes the simplest choice: cut the $d$-dimensional vector into $d/2$ adjacent pairs $(x_1, x_2), (x_3, x_4), \dots$ and apply the 2D solution to each pair in its own plane, each with its own frequency.

$$ f_{\{q,k\}}(x_m, m) = R^d_{\Theta,m}\, W_{\{q,k\}}\, x_m \qquad \text{(Eq. 14)} $$

$$ R^d_{\Theta,m} = \mathrm{blockdiag}\big(R(m\theta_1), R(m\theta_2), \dots, R(m\theta_{d/2})\big), \qquad \theta_i = 10000^{-2(i-1)/d},\; i = 1, \dots, d/2 \qquad \text{(Eq. 15)} $$

The matrix is block-diagonal: $2 \times 2$ rotation blocks down the diagonal and zeros everywhere else, so pair $i$ is turned by $m\theta_i$ and nothing mixes between pairs. Because the inner product is a sum over coordinates, it is also a sum over pairs, and each pair's contribution obeys the 2D result separately. The frequencies are the sinusoidal encoding's frequencies from Eq. 4, "following Vaswani et al.", with no further motivation.

::figure rope:fig1 | the paper's Figure 1. Top (dashed box, d = 2): a query/key pair (x₁, x₂) with constant θ₁ and position m is rotated by mθ₁ into (x′₁, x′₂). Bottom: the d-dimensional query/key of every token (rows for positions 1–6) split into pairs coloured θ₁, θ₂, …, θ_{d/2}; each pair is rotated by its own angle, giving the position-encoded query/key on the right, whose colours shift a little from row to row

::predict rope-block-diagonal

::worked rope-block-diagonal

::note warning | The paper writes the exponent two ways. Eq. 15 counts pairs from 1, $10000^{-2(i-1)/d}$; §3.3 and §3.4.3 write $10000^{-2i/d}$ with $i$ counted from 0. Same set of frequencies. Code (and the widgets here) use the 0-indexed form.

What does a spread of frequencies buy? Each pair is a clock hand that turns $\theta_i$ radians per position, so it returns to where it started every $2\pi/\theta_i = 2\pi \cdot 10000^{2i/d}$ positions (0-indexed). For $d = 128$, with 64 pairs:

| pair (0-indexed $i$) | $\theta_i$ | positions per full turn $2\pi/\theta_i$ |
|---|---|---|
| 0 | 1 | 6.28 |
| 16 | 0.1 | 62.8 |
| 32 | 0.01 | 628 |
| 48 | 0.001 | 6,283 |
| 63 | $1.15 \times 10^{-4}$ | 54,410 |

Each pair is $10000^{-2/128} \approx 0.866$ times as fast as the one before, so on a log axis the frequencies lie on a straight line. The fast pairs distinguish neighbours: one position apart is already a 57° turn for pair 0. The slow pairs barely move over thousands of tokens, so they distinguish large offsets without wrapping around. With a single shared $\theta$, offsets that differ by about a multiple of $2\pi/\theta$ would look nearly the same, and the model could resolve distance at only one scale.

::widget fixture:rope--frequency-dials | raise the position m: the first dials spin many turns while the last pairs have barely moved; the log strip of θ_i is a straight line

::slide lecture_03:33 | the course's picture of the block-diagonal choice: pairs (x₁, x₂), (x₃, x₄), … each rotated by mθᵢ; on the right, Gemma 4's partial variant that rotates only some pairs

::note aside | Two course additions from [L3](#/read/lecture_03). The professor calls pairing coordinates "the simplest possible thing" and prefers thinking of independent 2D rotations over the paper's complex numbers. And Gemma 4's "p-RoPE" rotates only part of the vector, on the argument that the slow pairs hardly rotate anyway. The paper discusses neither, nor why the pairs are adjacent coordinates rather than, say, $i$ and $i + d/2$.

::kp rope-block-diagonal

## Why only the offset survives {#relative}
source: rope:§3.2.2 · rope:eq16 · lecture_03.pdf p34

Apply Eq. 14 to the query and the key and multiply. The paper's Eq. 16 is the central identity of RoPE:

$$ q_m^\top k_n = \big(R^d_{\Theta,m} W_q x_m\big)^\top \big(R^d_{\Theta,n} W_k x_n\big) = x_m^\top W_q^\top\, R^d_{\Theta,n-m}\, W_k\, x_n, \qquad R^d_{\Theta,n-m} = (R^d_{\Theta,m})^\top R^d_{\Theta,n} \qquad \text{(Eq. 16)} $$

The two rotations meet in the middle and merge into one rotation by the offset. The paper asserts the merge without proof; it is Fact 1, applied block by block. Before the proof, a guess on the 2D example from above.

::predict rope-relative-position

::worked rope-relative-position

The score stays at $-0.256$: both arrows gain the same extra $4\theta$, and the 60° gap between them is untouched. Move only the query by 4 (to $m = 5$, $n = 3$) and the net rotation becomes $R_{-2}$, the score $0.956$. Shared shifts cancel; one-sided shifts do not.

::animation fixture:rope-shared-shift | the readout q_m · k_n stays fixed while both arrows rotate together by a shared shift, and changes the moment only one arrow moves

::note slip | Eq. 16 as printed reads $x^\top W_q R^d_{\Theta,n-m} W_k x_n$: the query's subscript $m$ and the transpose on $W_q$ are missing. The correct form is the one above.

Two misreadings to avoid. "Rotation preserves dot products, so RoPE leaves the score unchanged": only a *shared* rotation preserves it; here the rotations differ, and the net $R_{n-m}$ that remains is the point. "The score depends on $|m - n|$": $R_{n-m}$ and $R_{m-n}$ are transposes, not equal, as the $-0.256$ against $+0.956$ example showed.

### Multiplicative, not additive
§3.2.2 ends with the contrast that motivated the whole paper. Eqs. 3–10 are additive and produce cross terms that each scheme then edits away. RoPE is multiplicative: it never creates them.

::worked rope-multiplicative-vs-additive

So RoPE is not "the sinusoidal encoding applied to $q$ and $k$ instead of $x$". It uses the same frequencies, but it multiplies by them. Adding sinusoids to $q$ and $k$ would bring back exactly the cross terms of Eq. 6.

::slide lecture_03:34 | the course's summary of the math: f_{q,k}(x_m, m) = R^d_{Θ,m} W_{q,k} x_m with 2×2 blocks of cos mθᵢ and sin mθᵢ, and the line "difference with sine embeddings: not additive, no cross terms"

::kp rope-relative-position
::kp rope-multiplicative-vs-additive

## Property 1: does attention fade with distance? {#decay}
source: rope:§3.3 · rope:§3.4.3 · rope:eq35–eq37 · rope:fig2

§3.3 claims that with $\theta_i = 10000^{-2i/d}$ "the inner-product will decay when the relative position increase", which "coincides with the intuition" that distant tokens should be less connected. The proof is in §3.4.3. It is worth reading carefully, because what it proves is weaker than what §3.3 says.

The setup reads each pair of coordinates as one complex number, so the $d$-dimensional score becomes a sum of $d/2$ rotated 2D scores, one per frequency:

$$ \big(R^d_{\Theta,m} W_q x_m\big)^\top \big(R^d_{\Theta,n} W_k x_n\big) = \mathrm{Re}\Big[\sum_{i=0}^{d/2-1} q_{[2i:2i+1]}\, k^*_{[2i:2i+1]}\, e^{i(m-n)\theta_i}\Big] \qquad \text{(Eq. 35)} $$

Call the content part of each term $h_i = q_{[2i:2i+1]} k^*_{[2i:2i+1]}$: it depends only on the two vectors. The distance lives only in the unit phases $e^{i(m-n)\theta_i}$. A single phase never shrinks, but a *sum* of phases turning at different rates can cancel. Abel summation (summation by parts, the discrete version of integrating by parts) moves the distance dependence into the partial sums $S_j = \sum_{l<j} e^{i(m-n)\theta_l}$:

::predict rope-long-term-decay

::worked rope-long-term-decay

So the bound is (content factor) × (distance factor). The paper does not prove that the distance factor decays. It plots its average, $\frac{1}{d/2}\sum_{i=1}^{d/2} |S_i|$, against the offset.

::figure rope:fig2 | the paper's Figure 2, "long-term decay of RoPE": the relative upper bound (1/(d/2)) Σ|S_i| on the vertical axis against relative distance 0–256; it falls steeply from about 20 near the start to around 10–12 by distance 50, then keeps falling slowly while oscillating, with ripples of several units, reaching about 7 near 250. The figure does not state d

At distance 0 every phase is 1, so $|S_j| = j$ and the average is $(d/2 + 1)/2$, which is 32.5 for $d = 128$. Recomputing the plotted quantity for $d = 128$ (an inference; the figure prints no $d$, but this matches its scale) gives about 18.0 at distance 10, 12.6 at 50, 10.2 at 100 and 6.5 at 250. Further out the curve stops falling steadily: about 4.5 at distance 1,000 and 5.0 at 2,000.

::widget fixture:rope--decay-bound | the bound drops from 32.5 at distance 0 and ripples downward; tick "all θ_i equal" and it goes flat; show the aligned pair and one real score peaks at r0 under the decaying bound

The "all equal" switch shows where the decay comes from. With one frequency, $S_j = j\, e^{i(m-n)\theta}$ has size $j$ at every distance, so nothing decays at all. The decay comes entirely from summing phases that turn at many different rates and drift out of step.

::note warning | Three cautions the paper skips. (1) What is proved, Eq. 37, is an upper bound, and its content factor $\max_i |h_{i+1} - h_i|$ depends on the learned $q$ and $k$. A trained head can still put a large score on a token far away; the widget's aligned pair does exactly that. (2) That the distance factor decays is shown by a plot, not proved, and the plot oscillates; it is not monotone, as 4.5 at 1,000 against 5.0 at 2,000 shows. (3) §3.3's wording, "the inner-product will decay", overstates Eq. 37. The course leaves this property out entirely.

::kp rope-long-term-decay

## Property 2: RoPE inside linear attention {#linear}
source: rope:§3.3 · rope:eq17–eq19 · rope:§1

Standard attention computes $q_m^\top k_n$ for every pair, $O(N^2)$ work. **Linear attention** (Katharopoulos et al., 2020) rewrites attention in a general form with a similarity function (Eq. 17) and then picks a similarity that factors into a query part and a key part, with non-negative feature maps $\phi$ and $\varphi$ (for example $\mathrm{elu}(x) + 1$):

$$ \mathrm{Attention}(Q, K, V)_m = \frac{\sum_{n} \phi(q_m)^\top \varphi(k_n)\, v_n}{\sum_{n} \phi(q_m)^\top \varphi(k_n)} \qquad \text{(Eq. 18)} $$

Because the score splits, $\phi(q_m)^\top$ can be pulled out of the sums: $\sum_n \varphi(k_n) v_n^\top$ and $\sum_n \varphi(k_n)$ are computed once and shared by every query, which makes the cost linear in $N$. This is why the paper says earlier relative schemes are incompatible with it. A T5 bias $b_{m-n}$ added to the score belongs to neither the query nor the key, so the sums cannot be precomputed. RoPE's rotations act on each token separately, so they can.

The paper inserts them into the numerator only:

$$ \mathrm{Attention}(Q, K, V)_m = \frac{\sum_{n} \big(R^d_{\Theta,m}\phi(q_m)\big)^\top \big(R^d_{\Theta,n}\varphi(k_n)\big)\, v_n}{\sum_{n} \phi(q_m)^\top \varphi(k_n)} \qquad \text{(Eq. 19)} $$

Rotation keeps the norm of the feature vectors, so it injects position without changing their size. But it does not keep the *sign* of inner products: two non-negative vectors rotated apart by more than 90° have a negative dot product. Hence the unrotated denominator, kept "to avoid the risk of dividing zero".

::predict rope-linear-attention

::worked rope-linear-attention

A concrete instance shows what is given up. In 2D with $\theta = \pi/3$ (60°), put the query at $m = 2$ with $\phi(q) = (1, 0)$, one key at $n = 0$ with $\varphi(k) = (1, 0)$ and another at $n = 1$ with $\varphi(k) = (0.6, 0.8)$. The rotated numerator terms are $\phi(q)^\top R\big((n - m)\theta\big)\varphi(k_n)$. For $n = 0$ the key turns by $-120°$ to $(-0.5, -0.866)$, so the term is $-0.5$. For $n = 1$ it turns by $-60°$ to $(0.993, -0.120)$, so the term is $0.993$. The unrotated denominator is $1 + 0.6 = 1.6$. The effective weights are $-0.5/1.6 = -0.313$ and $0.993/1.6 = 0.620$: one is negative, and together they sum to 0.308, not 1.

::widget fixture:rope--linear-attention | the unrotated denominator stays positive, but once the query sits at a different position from the keys the weights stop summing to 1, and a key rotated past 90° gets a negative weight

The paper acknowledges that the weights in Eq. 19 are "not strictly probabilistic normalized" and argues, without proof, that they can still model the importance of values. Its only support is the Performer convergence curve in the experiments. The course does not cover linear attention.

::kp rope-linear-attention

## Computing it cheaply, and where it goes {#efficient}
source: rope:§3.4.2 · rope:eq34 · external/RoFormer/README.md:L22-L32 · rope:§4.2.2 · lecture_03.pdf p35

Written as Eq. 16, RoPE is a $d \times d$ matrix times a vector for every query and key. §3.2.2 already notes that, "due to the sparsity" of $R^d_{\Theta}$, this is wasteful: each row has only two non-zero entries. §3.4.2 gives the realization everyone actually uses. Look at what one pair does under rotation by $\varphi$: $(x_1, x_2) \mapsto (x_1\cos\varphi - x_2\sin\varphi,\; x_1\sin\varphi + x_2\cos\varphi)$. Every output coordinate is (its own value) × cos plus (its partner, one sign flipped) × sin. Collect that over all pairs:

$$ R^d_{\Theta,m}\, x = \begin{pmatrix} x_1 \\ x_2 \\ x_3 \\ x_4 \\ \vdots \end{pmatrix} \otimes \begin{pmatrix} \cos m\theta_1 \\ \cos m\theta_1 \\ \cos m\theta_2 \\ \cos m\theta_2 \\ \vdots \end{pmatrix} + \begin{pmatrix} -x_2 \\ x_1 \\ -x_4 \\ x_3 \\ \vdots \end{pmatrix} \otimes \begin{pmatrix} \sin m\theta_1 \\ \sin m\theta_1 \\ \sin m\theta_2 \\ \sin m\theta_2 \\ \vdots \end{pmatrix} \qquad \text{(Eq. 34)} $$

Here $\otimes$ is elementwise multiplication. Each frequency appears twice because it serves both coordinates of its pair.

::predict rope-efficient-implementation

::worked rope-efficient-implementation

The cost is two elementwise products and one addition: $2d$ multiplications per vector instead of $d^2$. For $d = 128$ that is 256 against 16,384. The cos and sin vectors depend only on the position, so they are computed once per position and shared by every head, layer and batch element. RoPE is far cheaper than the projection $W_q x$ that comes before it.

The official repository does not contain the layer itself (its training script selects `model='roformer'` from the bert4keras library), but its README gives the authors' pseudocode, which is Eq. 34 line for line:

::code external/RoFormer/README.md:L22-L32 | cos_pos and sin_pos repeat each frequency twice; qw2 is the companion (−x₂, x₁, −x₄, x₃, …) built by stacking the negated odd coordinates with the even ones; after the two rotations the attention is the ordinary einsum

`sinusoidal_pos` is the Eq. 4 table, sines in the even slots and cosines in the odd ones, so `1::2` picks the cosines and `::2` the sines. RoPE reuses the original transformer's position table; it just multiplies by it instead of adding it.

::note aside | Many later implementations, Hugging Face's Llama code among them, pair coordinate $i$ with $i + d/2$ ("rotate_half") instead of adjacent coordinates. That is the same operator under a fixed permutation of dimensions. Since the same permutation applies to $q$ and $k$, every score is unchanged, but weights trained with one convention need their dimensions reordered for the other.

### Every layer, queries and keys only
Eq. 14 rotates $W_q x_m$ and $W_k x_n$, the vectors that enter the dot product, and §4.2.2 confirms that the rotation lives "in the self-attention block". Two consequences follow, which the course puts on a slide and the paper leaves implicit.

::slide lecture_03:35 | the course's implementation slide (Hugging Face style): the usual q/k/v projections, then cos, sin = rotary_emb(...) and apply_rotary_pos_emb(query, key, cos, sin), then the usual attention; the note "embedding at each attention operation to enforce position invariance"

- **It is applied inside every attention layer**, not once at the input. If you rotated the embedding once instead, $x'_m = R_m x_m$, even the first layer's score would be $x_m^\top R_m^\top W_q^\top W_k R_n x_n$. The projections now sit *between* the two rotations, and a general $W_q^\top W_k$ does not commute with rotations, so $R_m^\top$ and $R_n$ can no longer merge into $R_{n-m}$.
- **Values are never rotated.** Eq. 2 uses $v_n$ directly, and the README rotates only `qw` and `kw`. Rotating $v_n$ by $n\theta$ would leak absolute position into the output $o_m = \sum_n a_{m,n} v_n$, where nothing cancels it.

::predict rope-per-layer-application

::kp rope-efficient-implementation
::kp rope-per-layer-application

## What did the experiments show? {#experiments}
source: rope:§4 · rope:table1 · rope:fig3 · rope:table2 · rope:table5 · lecture_03.pdf p30, p65

§4 runs five experiments, all on two cloud servers with 4 V100 GPUs each. They are small by today's standards, and the claims in the introduction run ahead of them, so it pays to read each one for what it measures.

**Machine translation (§4.1).** WMT 2014 English–German, about 4.5 million sentence pairs, in the fairseq toolkit with a 37k joint BPE vocabulary. The baseline transformer and RoFormer are trained under the same settings: Adam with $\beta_1 = 0.9$, $\beta_2 = 0.98$, the learning rate raised linearly from $10^{-7}$ to $5 \times 10^{-4}$ and then decayed with the inverse square root of the step, label smoothing 0.1. Evaluation uses one model made by averaging the last 5 checkpoints, beam size 4, length penalty 0.6.

::figure rope:table1 | the paper's Table 1: BLEU on WMT14 En-De, Transformer-base 27.3, RoFormer 27.5 (bold); one number per model

::predict rope-experiments

A 0.2 BLEU gap from one run per model, with no seeds or variance, is smaller than the run-to-run noise usually seen on this benchmark. It shows RoPE does not hurt translation; it does not show it helps.

**Pre-training (§4.2).** BERT (bert-base-uncased) against the same model with RoPE in its self-attention, on BookCorpus plus Wikipedia split 8:2 into train and validation, batch 64, sequence length 512, 100k steps, AdamW at learning rate $10^{-5}$. The evidence is the masked-language-model loss curve.

::figure rope:fig3 | the paper's Figure 3. Left: MLM training loss against steps for BERT and RoFormer, where RoFormer's curve falls faster. Right: training loss for Performer with and without RoPE, where the RoPE variant converges faster. Curves only, no table of numbers

::note slip | §4.2 says it replaces "the original sinusoidal position encoding of BERT". BERT's position embeddings are learned, as the paper's own §2.2 says when it lists Devlin et al. among the trainable absolute encodings.

**GLUE fine-tuning (§4.3).** The pre-trained models are fine-tuned on six GLUE tasks. Table 2 is mixed: RoFormer is higher on MRPC, STS-B and QQP and lower on SST-2, QNLI and MNLI. That sits awkwardly with the introduction's claim that RoFormer "consistently achieves better performance".

::figure rope:table2 | the paper's Table 2: BERT against RoFormer on six GLUE tasks; RoFormer is ahead on MRPC, STS-B and QQP and behind on SST-2, QNLI and MNLI

**Linear attention (§4.4).** Performer (Choromanski et al., 2020) with and without RoPE, inserted as in Eq. 19. The right panel of Figure 3 shows faster convergence with RoPE. This is the only experiment touching a claimed property, and it is a loss curve, not a downstream score.

**Long Chinese documents (§4.5).** On CAIL2019-SCM, a legal case-matching task with long documents, RoFormer reaches 68.29 accuracy at maximum length 512 and 69.79 at 1024, a gain of 1.50 points.

::figure rope:table5 | the paper's Table 5: CAIL2019-SCM accuracy for RoFormer at maximum sequence length 512 (68.29) and 1024 (69.79)

### What the experiments did not test
Of the three properties in §1, **sequence-length flexibility** is never tested. No model is trained at one length and evaluated at a longer one. Table 5 compares two separately trained length settings, which shows that more context helps on long documents, not that RoPE extrapolates. **Long-term decay** has only the plot of a bound. **Linear attention** has one convergence curve. Nothing in §4 reports seeds or variance.

::note aside | The course does not cite these results at all. Its evidence for RoPE is adoption: GPT-J, PaLM, LLaMA and most models since 2024 (slide 30). Later in [L3](#/read/lecture_03), Cohere's Command A keeps RoPE only in its sliding-window layers and uses no position embedding at all (NoPE) in the full-attention layers: short-range information through RoPE, long-range through content alone.

::slide lecture_03:65 | Cohere Command A: three sliding-window attention blocks with RoPE for every full-attention block with no positional embeddings, "long-range info via NoPE, short-range info via RoPE + SWA"

::kp rope-experiments

## The whole idea on one page {#summary}
source: rope:§3 · rope:§4

| step | what it says |
|---|---|
| requirement (Eq. 11) | $\langle f_q(x_m, m), f_k(x_n, n)\rangle$ may depend on positions only through $m - n$, with each side built from its own token |
| 2D solution (Eq. 12–13) | rotate the projected vector by $m\theta$; the conjugate on the key makes positions subtract |
| general form (Eq. 14–15) | $d/2$ independent pair rotations at frequencies $\theta_i = 10000^{-2i/d}$, from 1 radian per position down to about $10^{-4}$ |
| key identity (Eq. 16) | $q_m^\top k_n = x_m^\top W_q^\top R_{n-m} W_k x_n$: multiplicative, no cross terms |
| decay (Eq. 35–37) | an upper bound whose distance factor shrinks on average, not a guarantee for every score |
| linear attention (Eq. 19) | rotations act per token, so they fit kernelized attention; the weights lose normalization |
| implementation (Eq. 34) | $x \otimes \cos + \text{companion} \otimes \sin$, $2d$ multiplications, on $q$ and $k$ in every layer |
| evidence (§4) | small, single-run gains and faster loss curves; length extrapolation untested |

The lasting part of the paper is the first four rows. Rotating queries and keys by position gives a relative encoding that is still an inner product of per-token vectors. That one structural fact is why RoPE caches cleanly in a decoder, why it fits linear attention, and why it became the default position encoding of the models in [L3](#/read/lecture_03).
