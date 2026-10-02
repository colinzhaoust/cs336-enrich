---
title: L11 · Scaling case studies and muP, read through
minutes: 45
---
[L9](#/read/lecture_09) taught scaling laws as a clean procedure: train small models, fit a power law, extrapolate. This lecture asks what that procedure looks like when real teams build real models. It walks through the published recipes of MiniCPM and DeepSeek, then a tour of newer reports, then two ways of handling the hyperparameters that change with scale: fit their scaling (DeepSeek, StepFun) or reparametrize the network so they stop changing (muP, derived from scratch at the end). After it you can explain why a cosine schedule makes a Chinchilla sweep cost $n^2$ and how WSD fixes it, read an optimal-batch or optimal-LR law, say what a sloppy optimizer comparison gets wrong, and derive muP's initialization and learning-rate rules for a linear layer.

## Does the textbook procedure survive contact with real models? {#why}
source: lecture_11.pdf p2-p5 · video 0:05-3:48

The professor calls the lecture "a bit of a grab bag": the advanced details of how you would scale up a language model in practice. Three questions frame it.

::slide 2 | the three questions: does Chinchilla's approach to scaling actually work, can we save compute when training and fitting, and should we pick architectures or parametrizations that scale nicely

The first is the one [L9](#/read/lecture_09) left open: does the Chinchilla machinery "actually work at real building open-source model scales?" The second is about cost: a scaling study is itself expensive, and there are tricks to make it cheaper. The third is about the network: can we choose a parametrization so that some hyperparameters stop depending on scale at all?

::slide 3 | "the newest model we talked about with scaling details - 2022" (DeepMind's Chinchilla paper), then the newer reports this lecture uses: DeepSeek LLM, MiniCPM and Llama 3 (2024), Hunyuan-Large (2024), MiniMax-01 (2025), Kimi K2 (2026)

The classical canon from L9 (Kaplan, Hestness, Chinchilla) stops around 2022. Since then, the scaling papers from teams that train big models have become fewer, and most of the detailed ones have come from the Chinese open-source community. The lecture uses a curated set of them to show what scaling at the open frontier looks like, from Chinchilla up to Kimi K2, "the most recent release that has some scaling details".

::slide 4 | "Initialization, optimizers, and various hyperparams (LR/batch) can be scale sensitive": top, StepFun's optimal learning rate and batch size against data size D for models from 59M to 1B parameters; bottom, losses of AdamW, NAdamW, Muon and Soap against the Chinchilla ratio for 1.2B and 300M models, and how many more tokens AdamW needs to match them

The second half of the lecture is previewed on this slide. Optimizer behaviour is "pretty scale sensitive", so initializations, learning rates and batch sizes all have to be set as a function of scale. In the top-left plot each colour is one model size, and the optimal learning rate sits on a different line for each; in the top-right plot the optimal batch size climbs with the data size. The bottom row compares optimizers at different tokens-per-parameter ratios. Every one of these plots returns later in the lecture.

::slide 5 | the plan: two detailed public scaling recipes, MiniCPM and DeepSeek LLM

Why spend most of the time on two 2024 papers rather than the newest ones? Because the newest ones no longer write this down. How to set learning rates, how to run a Chinchilla analysis: these are now "taken for granted", so recent reports skip them. MiniCPM and DeepSeek are serious scaling investigations by teams that built good models, and they take *different* approaches to the key question of this lecture, which is what to do about hyperparameters that drift with scale.

## MiniCPM: muP, a scaling ladder, and the optimal batch {#minicpm}
source: lecture_11.pdf p6-p12 · video 3:33-9:59

::slide 6 | MiniCPM (2024), a small high-performance LM from a Tsinghua group: "careful, extensive scaling computations + muP to stabilize and simplify scaling", "not really 'sota' model (even in 2024) but many interesting lessons on scaling"

MiniCPM came out of an industry-academic partnership in the Chinese open-source community. When it appeared it was roughly state of the art in the 1-2B bracket; today Gemma and others at that size are better. The lecture studies it for its method, not its scores.

::slide 7 | benchmark table: MiniCPM-1.2B and MiniCPM-2.4B against 7B, 13B-40B and 1-3B models on C-Eval, CMMLU, MMLU, HumanEval, MBPP, GSM8K and MATH (MiniCPM-2.4B: MMLU 53.46, HumanEval 50.00)

The table backs the slide's claim that the 1-2.5B models "beat most out 2Bs and match many modern 7B models". For example, MiniCPM-2.4B's MMLU of 53.46 is above Llama2-7B's 44.32 and Deepseek-7B's 47.82, though below Mistral-7B's 62.69.

### Technique 1: muP to keep the learning rate still

The point of the study is not to brute-force a 1.5B model by training hundreds of them. It is to "get the hyperparameters exactly right the first time" with small models. Initialization is the first lever, because the right initialization can make certain hyperparameters less scale dependent. MiniCPM uses **muP** (the maximal update parametrization, met briefly in [L9](#/read/lecture_09)), whose whole promise is that the optimal learning rate stays the same as the model is scaled up or down.

::slide 8 | "Scale_emb = 12, scale_depth = 1.4, init_std = 0.1, lr = 0.01", above MiniCPM's Table 7 of operations: multiply the embedding output by scale_emb; scale each residual increment by scale_depth/√num_layers; init std of each 2-D tensor init_std/√(d_m/d_base), other parameters 0.1; LR of each 2-D tensor 1/(d_m/d_base) times the base LR; output logits times 1/(d_m/d_base)

The professor read this table aloud. Here $d_m$ is the model width and $d_{base}$ the width of the small proxy model on which the constants were tuned, so $d_m/d_{base}$ is how many times wider the model is than its proxy. Each row is a width-dependent rescaling:
- **embedding output** multiplied by a constant, scale_emb = 12;
- **residual connections**: each layer's increment scaled by scale_depth$/\sqrt{\text{num\_layers}}$, so deeper models add smaller increments (with 24 layers, $1.4/\sqrt{24} \approx 0.29$);
- **matrix initialization**: std init_std$/\sqrt{d_m/d_{base}}$, smaller for wider models;
- **per-tensor learning rates**: every matrix gets the base LR divided by $d_m/d_{base}$. "You should pay attention to this one because this is exotic if you aren't used to per parameter learning rates";
- **LM head**: logits divided by $d_m/d_{base}$.

A model 4× wider than its proxy thus gets matrices initialized with half the std and trained with a quarter of the learning rate. Why these particular rules? That was deferred to the derivation at the end of the lecture, where the $1/\text{width}$ learning rate for Adam falls out.

::note deferred 6:16 | "We'll talk about why you do these things, or how you derive this particular set of scalings." The answer is the muP derivation in the last third of the lecture.
::note aside 7:13 | MiniCPM also replicates a Chinchilla analysis, "much less important to us now", covered here only for the trick it uses (WSD, in the next section).

### The ladder

::predict minicpm-mup-fixed-aspect-recipe

::slide 9 | the study models, 9M to 0.5B: name, N (B), d_m, d_ff, d_h, n_h, L, from 9M (0.009B, d_m 320, 8 layers) to 0.5B (0.499B, d_m 1344, 24 layers); "the gap between the largest model here and the actual model they train is ~5x"; "optimal batch, LR, token-to-size ratios are directly fitted via scaling analysis"

The recipe has three parts: use muP for initialization, fix the shape, and scale up the overall size. The table on the slide image shows what "fix the shape" means. Every model uses heads of size $d_h = 64$ (so $n_h = d_m/64$: 320/64 = 5 heads, 1344/64 = 21) and an MLP width of exactly $2.5\,d_m$ (800 = 2.5 × 320, 3360 = 2.5 × 1344), while width and depth grow together. The width-to-depth ratio is not quite constant, it drifts from 320/8 = 40 to 1344/24 = 56, but every model comes from one family, so a single number, the size, moves along the ladder.

The largest rung is 0.5B, and the model they release is MiniCPM-2.4B, a factor of $2.4/0.499 \approx 4.8$, the "~5x" gap. Over that gap they want the most sensitive quantities right: the optimal batch size, the learning rate, and the token-to-size ratio from Chinchilla. muP does not make the fitting unnecessary. It keeps the learning rate still, and the rest is still fitted from the ladder.

::slide 10 | "According to muP – optimal learning rate should be (roughly) stable. Is it?"; loss against learning rate from 10⁻³ to 10⁻¹ for models of 0.04B, 0.1B, 0.3B, 0.5B and 2.1B, each curve bottoming out near 10⁻²; "after applying for the Tensor Program, the learning rate shift becomes minimal"

Is it? "At least in the MiniCPM case, the answer is yes", and very cleanly. Each curve is one model size; every minimum sits "basically right on 10 to the negative 2", the lr = 0.01 of slide 8, with only the smallest model's minimum slightly shifted.

Notice also the cliff on the right: past about $3 \times 10^{-2}$ the loss jumps from below 3.5 to between 6 and 8 for most sizes. The curves are flat near the optimum and steep past it, so the safe error is to be a little low. If you get a plot like this, "you've removed the need to tune the learning rate".

::video 7:26-8:16 | the answer to "Is it?": the optimum sits at about 1e-2 for every size, shifting only slightly for the smallest
::kp minicpm-mup-fixed-aspect-recipe

### The optimal batch still moves

Even with a parametrization that pins the learning rate to one optimum, the optimal batch size is "still not going to be the same". It varies with both the data size and the model size, so MiniCPM measures it.

::predict optimal-batch-vs-loss

::slide 11 | three heat maps for 9M, 30M and 170M models: tokens processed (y) against batch size (x), loss as colour; vertical columns of points are single training runs; a red line traces the minimum-loss batch at each data size

Read the plot one column at a time. A training run has a fixed batch size (one x position), and as it trains it processes more tokens, so its logged losses climb straight up the plot: one run is one vertical column of points. Now read across instead. At a fixed number of tokens processed (one y value), the columns give the loss reached with each batch size. The professor says they fit "quadratic curves" along these slices to find each minimum; the red line joins the minima. It is the optimal batch for that model and that amount of data, and in every panel it leans right as the data grows: more tokens processed, larger optimal batch.

::slide 12 | optimal batch size (log scale, about 3×10⁴ to 6×10⁵) against final loss from about 3.3 to 5.4, with the fitted line log(BS) = −6.24·log(L) + 20.91; "polynomially increase the batch size as loss decreases"

Following Kaplan 2020's critical-batch analysis ([L9](#/read/lecture_09)), plot the optimal batch against the loss reached, and the points fall on one clean line. The text layer gives no exponent, but the fit is printed on the slide image:

$$ \log B_\text{opt} = -6.24 \log L + 20.91 \quad\Longleftrightarrow\quad B_\text{opt} = e^{20.91}\, L^{-6.24} $$

The logs are natural: at $L = 3.5$ the formula gives $e^{20.91 - 6.24 \times 1.253} = e^{13.09} \approx 4.9 \times 10^5$, and at $L = 5.0$ it gives $e^{10.87} \approx 5.3 \times 10^4$, both where the line sits on the plot. So going from loss 5.0 to loss 3.5 multiplies the optimal batch by $(5.0/3.5)^{6.24} \approx 9.3$. Lower loss, bigger batch: the critical-batch scaling of L9.

The practical recipe, which the slide leaves implicit, has two steps. Back out the loss you expect to reach from the scaling law, then read the batch off this line. One consequence: a batch tuned early in training is too small later, because the loss keeps falling. A batch that was optimal at loss 3.5 would, by this formula, be about $(3.5/2.8)^{6.24} \approx 4.0$ times too small at loss 2.8 (an extrapolation: the plotted range ends near 3.3).

::video 8:37-9:44 | how one run becomes a column of points, and how the loss target then sets the batch size

The professor's verdict on MiniCPM's first part: batch size and learning rate "are going to come back over and over again". They are arguably the most important and most sensitive parameters to do scaling laws on.

::kp optimal-batch-vs-loss

## Why a Chinchilla sweep costs $n^2$, and how WSD makes it cheap {#wsd}
source: lecture_11.pdf p13-p15 · video 9:59-14:09

What remains after batch and learning rate is the model-size versus data trade-off of [L9's Chinchilla analysis](#/read/lecture_09). To run it you train many models: for an IsoFLOP sweep, say, you fix a FLOP budget and trade the number of tokens against model size, which means training each model on several different amounts of data.

::slide 13 | Chinchilla's appendix figure: cosine schedules whose cycle is 1.0×, 1.1×, 1.25×, 1.5×, 2.0× and 5.0× the number of training steps (LR curves left), with training and C4 loss over 8 and 12.5 million sequences; the longer the cycle, the worse the loss at the end of training; "this turns the cost of fitting a scaling law from n to n^2.. Can we avoid this?"

The trouble is the learning-rate schedule. A **cosine schedule** decays the learning rate along a half cosine that ends exactly at the planned end of training, so "you need to know the terminal total budget before you train". Look at the left panels: a schedule planned for 5× the steps has barely started decaying when training stops, and its loss (middle and right panels) ends up clearly worse than the schedule whose cycle matches the run, 1.0×. So the loss at step $t$ of a long cosine run is *not* the loss a run planned to stop at $t$ would reach; it is pessimistic.

Hence the rule from Chinchilla: to fit a scaling law you must train from scratch at every data size, not just stop one long run early. In the professor's example, "if you want to train a 8 million sequence model, you can't restart from the end of a 4-million sequence model". Each data size is a new run from step 0.

::predict chinchilla-fit-cost-n-squared
::worked chinchilla-fit-cost-n-squared

The slide never defines $n$. Read it as the number of data sizes per model: one long run whose intermediate losses you could read off would cost the length of the longest run, while $n$ separate runs cost the sum of their lengths, about $n^2/2$ units for evenly spaced sizes. The professor calls it "a quadratic cost in some sense".

::widget fixture:lecture_11--sweep-cost | each cosine row draws its own LR curve over its own length, so the cosine bill is the sum of all the D_i; raise k with even spacing and watch it grow like k², while the WSD bar barely moves

### Warmup, stable, decay

::slide 14 | MiniCPM's Figure 15: Cosine(40N) against WSD(40N, 4N) and WSD(80N, 8N): a short warmup to a peak LR of 0.02, a flat stable phase, then a straight drop to about 0.002; "the WSD LRS with different end steps share the same stable training stage"; paper excerpt: scaling law "with linear cost (O(mC))", 6 model sizes from 0.04B to 2B, each with 6 decayed models branched from checkpoints at 10N to 60N tokens

MiniCPM's fix, now known as the **warmup-stable-decay (WSD)** schedule, is "basically a big trapezoid":
- **warmup**: a fixed number of steps, not a fraction of the planned run, so it does not depend on the training horizon;
- **stable**: a long phase at constant learning rate, the vast majority of training;
- **decay**: a fast drop at the end, "something like 10% to 20%" of the run.

The stable phase is the key. A constant learning rate does not know when training will end, so every run of every length shares the same stable prefix. To get a decayed model at a shorter data size, branch off the stable run near that point and add only a short decay. "If I want to reuse a run and train it for longer, I'll just roll back my training to the last stable checkpoint", then continue the stable phase and decay. "You do have to redecay every time, but that's maybe 10% of the total cost": much better than rerunning pre-training from scratch.

In the figure the notation WSD(80N, 8N) means a run of 80N tokens ($N$ is the parameter count) whose last 8N are the decay, 10% of the run. The excerpt shows the payoff at MiniCPM's scale: six model sizes, each with six decayed branches from its single stable run, 36 data points without 36 from-scratch runs. For one model with points at 10N, 20N, …, 60N tokens, cosine would cost $10 + 20 + \dots + 60 = 210N$ tokens of training, while WSD costs the 60N stable run plus a decay of 10% of each branch's length on top, $60 + 0.1 \times 210 = 81N$: about 2.6× cheaper, and the gap grows with every extra data size.

::note spoken 11:37 | What the decay ends at is said two ways: "decay fairly rapidly down to 0" and "down to usually about 10% of your maximum learning rate". The slide image's Figure 15 sides with the second: its decays drop from 0.02 to about 0.002, 10% of the peak.
::widget fixture:wsd-branch-schedule | slide the branch length: the orange branch reuses the blue WSD stable run up to 90% of its own length and pays only its decay, while the dashed cosine for the same length leaves the full-length cosine right after warmup, so it needs a run from step 0
::widget fixture:lecture_11--sweep-cost | switch to doubling sizes and slide the decay fraction d: WSD's bill is D_max + d·ΣD_i, so it wins whenever there is more than one data size, until d passes the break-even shown in the readout

::slide 15 | C4 loss against tokens for WSD runs (40N, 2N), (60N, 2N), (80N, 2N), (40N, 4N), (60N, 6N), (80N, 8N) and Cosine(80N): the WSD curves lag above the cosine curve during the stable phase, then drop steeply in their decays; "Decay ~ 10%"

::predict wsd-schedule-branching

The loss curves show the price and the payoff. During the stable phase a WSD run looks "way underperforming", above the cosine curve the whole time. Then the decay starts and the loss plunges: "suddenly you reclaim all of your gains". At 80N tokens the WSD(80N, 8N) branch ends at about 3.62, slightly below the cosine curve's 3.65 at the same point. Shorter decays (2N) recover less. The professor's anecdotal summary: many people find cosine slightly better in many cases, but WSD is "basically as good" and much more versatile, since a run can always be continued.

He adds a lesson of its own: "just how much of an impact learning rate decay has". The stable-phase loss hides a large improvement that only the final annealing releases. So a stable-phase loss is never an $L(N, D)$ data point; only the end of a decayed branch is.

::note aside 12:54 | An external WSD-versus-cosine study is credited loosely ("maybe it was Martin Jaggi and others ... over at ETH") and not shown.
::video 12:06-12:44 | rolling back to the last stable checkpoint and redecaying, at "maybe 10% of the total cost"

With WSD, a Chinchilla analysis becomes "actually very easy": one long run per model size, rewind the checkpoints, decay repeatedly, and you have the sweep along the data axis. The professor's two takeaways from MiniCPM: remember muP as a trick to stabilize the learning rate across scales, and remember WSD, "a basic thing to know".

::kp chinchilla-fit-cost-n-squared
::kp wsd-schedule-branching

## MiniCPM's Chinchilla fits: about 100 tokens per parameter {#minicpm-chinchilla}
source: lecture_11.pdf p16-p18 · video 14:09-15:57

::slide 16 | MiniCPM's fitting excerpt: L(N, D) = C_N·N^(−α) + C_D·D^(−β) + L_0 fitted with scipy curvefit, and the optimum at fixed compute C = 6ND written as N_opt/D_opt = K²·(C/6)^η; "MiniCPM authors choose method 1 (lower envelope) and method 3 (joint fit)"

Equipped with WSD runs, MiniCPM looks for the optimal data-to-model ratio. Of [L9's three Chinchilla methods](#/read/lecture_09), it uses method 1, the lower envelope of loss against compute across all runs, and method 3, a joint parametric fit of the loss surface. The professor finds the choice odd: methods 1 and 3 are "to me, the least reliable of the Chinchilla methods". The parametric form is Chinchilla's: a power law in parameters, a power law in data, and a floor. From it, the compute-optimal ratio at budget $C = 6ND$ comes out as $N_\text{opt}/D_\text{opt} = K^2 (C/6)^\eta$: a constant $K^2$ times a power of compute.

::slide 17 | method 1, "real loss w.r.t. compute" on code, English (Wikihow) and Chinese (Wikihow) evaluation sets: one coloured segment per model size, together forming a lower envelope; "fairly clear (though maybe not linear?) trends"; "their runs suggest relatively low diminishing returns due to data"

Each coloured segment is one model trained on increasing data (its WSD branches); the lower envelope of all segments is the best loss at each compute. The professor reads the envelopes more kindly than the slide does, as "mostly linear" on these log-log axes, which is what a power law in compute should look like.

::slide 18 | method 3 on Ultratext: loss contours over non-embedding parameters (10⁹) and compute (10¹⁸ FLOPs), black dots for the runs; the fit 7.54×10⁻²/N^0.30 + 2.92×10⁻¹/D^0.30 + 0.25, K² = 0.01, η = −0.00, and D_opt/N_opt at C = 10²¹ equal to 95.60

The text layer only says "very high data-model ratios". The slide image prints the fitted surface, and with it we can see where the ratio comes from:

$$ L(N, D) = \frac{7.54\times10^{-2}}{N^{0.30}} + \frac{2.92\times10^{-1}}{D^{0.30}} + 0.25 $$

The two exponents are equal, both 0.30. That is why $\eta = 0$: the optimal ratio does not change with compute at all. With equal exponents, the ratio is set by the two coefficients alone. At fixed $C = 6ND$, the optimum balances the marginal returns of the two terms, $\alpha\,C_N N^{-\alpha} = \beta\,C_D D^{-\beta}$, which for $\alpha = \beta$ gives

$$ \frac{D_\text{opt}}{N_\text{opt}} = \left(\frac{C_D}{C_N}\right)^{1/0.30} = \left(\frac{0.292}{0.0754}\right)^{3.33} \approx 3.87^{3.33} \approx 91 $$

That is close to the printed 95.60 (the difference is rounding in the displayed exponents), and to $K^2 = 0.01$, i.e. $N/D = 1/100$. So MiniCPM's joint fit says about 90-100 tokens per parameter, roughly five times Chinchilla's 20 ([L9](#/read/lecture_09)). In this fit the data term's coefficient is about four times the parameter term's, so data stays the more valuable thing to buy for longer: slide 17's "low diminishing returns due to data".

The paper concludes that small models should be trained on "way more pieces of text than Chinchilla". The professor is not convinced: "It is highly unclear to me whether this is a really true or whether their Chinchilla fits are a little bit strange relative to the Chinchilla paper." He points to "pretty different exponents" from Chinchilla's; the methods are the least reliable two; and the decayed WSD branches are a different kind of data point from Chinchilla's full cosine runs. Any of these could move the fitted ratio.

::note spoken 15:23 | The professor doubts MiniCPM's conclusion and suspects the fit rather than the world: "their Chinchilla fits are a little bit strange relative to the Chinchilla paper."
::video 15:07-15:33 | the exponents differ from Chinchilla's, and the "way more pieces of text" conclusion is doubted
::kp minicpm-high-data-ratio

## DeepSeek: no muP, fit the batch and learning rate instead {#deepseek}
source: lecture_11.pdf p19-p21 · video 15:57-20:02

::slide 19 | DeepSeek LLM (2024), "another LM with careful scaling analysis": 7B and 67B models, "generally high performance compared to other open LM"

This is the original DeepSeek LLM paper, before the MoE work. Even now the professor rates it as "one of the more nicely executed scaling analyses in the open world", with "great taste in the experiments".

DeepSeek takes the opposite stance from MiniCPM. No muP, no special rescalings. They accept that the learning rate and batch size change with scale, and bet that the change is predictable: "if the way in which they change are predictable, just like a scaling law, then we're good to go." So they measure the optimum at small scales and fit a power law through it.

::slide 20 | "don't use any muP, directly estimate optimal batch / LR": two grids of final loss over batch size (2^15 to 2^20 tokens, and 2^19 to 2^23.5) and learning rate (around 2^−10 to 2^−7.75, and 2^−11.25 to 2^−8.5), at compute budgets of 1e17 FLOPs (177M FLOPs/token) and 1e20 FLOPs (2.94B FLOPs/token); a star marks the fitted optimum on the second

Each grid fixes the compute budget and sweeps learning rate against batch size; the colour is the final loss and the lightest cell is the optimum. The paper's excerpt makes a point worth noticing: the loss "remains stable across a wide range of choices of batch sizes and learning rates". At 1e20 FLOPs, for example, a whole block of cells sits between about 2.474 and 2.49. A flat basin is good news for training (being a little off costs little) and bad news for fitting (it is hard to say exactly where the minimum is). Notice also how the grid moved between the two budgets: at 1000× the compute, the batch range starts about 16× higher and the learning rates sit lower.

::predict deepseek-direct-batch-lr-fit

::slide 21 | "small scale runs + collect 'near optimal' (within 0.25% of min) models"; the fits η_opt = 0.3118·C^(−0.1250) and B_opt = 0.2920·C^(0.3271) against non-embedding training FLOPs from 10¹⁷ to 10²⁴, grey dots for near-optimal runs, stars for 7B (batch 9.2M, LR 4.2e−4) and 67B (batch 19.7M, LR 3.2e−4); "learning rate fit looks a bit questionable.."

To turn grids into a law, they keep every run within 0.25% of the best loss at its budget as "near optimal" (the grey dots) and fit a line through the cloud on log-log axes. The slide image prints the result, which the text layer drops:

$$ \eta_\text{opt} = 0.3118 \cdot C^{-0.1250}, \qquad B_\text{opt} = 0.2920 \cdot C^{0.3271} $$

with $C$ in non-embedding training FLOPs and $B$ in tokens. At higher compute, much larger batches and somewhat lower learning rates. Each 10× in compute multiplies the optimal batch by $10^{0.3271} \approx 2.1$ and the learning rate by $10^{-0.125} \approx 0.75$.

The stars are the settings actually used for the big runs, and the formulas reproduce them. The 7B star's batch of 9.2M tokens corresponds to $C^{0.3271} = 9.2\times10^6/0.292$, so $C \approx 8.4\times10^{22}$; at that compute the LR formula gives $0.3118 \times (8.4\times10^{22})^{-0.125} \approx 4.2\times10^{-4}$, the starred value. The 67B star (19.7M tokens) sits at $C \approx 8.6\times10^{23}$, where the formula gives $3.2\times10^{-4}$, again the starred value.

::note warning | The StepFun comparison table on slide 33 quotes DeepSeek's LR law as 0.3188·C^−0.1250; slide 21, from the DeepSeek paper itself, prints 0.3118. The formula above uses 0.3118, which reproduces both stars.

The batch fit "does look like a line". The learning-rate fit is the shakier one: "I'm not quite sure that constitutes the best linear fits that I've seen in my life, but I guess it works ... The models do train." The reason he gives is the grid itself: "unless your grid is in the right place, you've got a lot of quantization error", and the near-optimal LRs at one budget come in a few discrete values, so the line through them is poorly pinned. The flat basin of slide 20 adds to this: a 0.25% band admits a wide range of learning rates.

::note aside 19:33 | Asked why one FLOP budget shows a large spread of optimal learning rates, he answers that the points at one budget come from different model sizes.
::video 18:52-19:30 | why the LR fit is shakier than the batch fit: grid quantization error

His one-line summary of the two papers: there are two ways to deal with the sensitive hyperparameters. "You can either attempt to stabilize them, or you can attempt to just scaling law fit them and then just go with the scaling law." MiniCPM stabilizes with muP; DeepSeek fits.

::kp deepseek-direct-batch-lr-fit

## DeepSeek's Chinchilla replication, and the test that matters {#deepseek-isoflop}
source: lecture_11.pdf p22-p24 · video 20:02-22:34

In 2024 "every open model builder was replicating the entire scaling law stack", and DeepSeek complements its hyperparameter laws with a Chinchilla replication. Like MiniCPM, it needs a schedule that makes the sweep affordable.

::slide 22 | DeepSeek's multi-step scheduler: peak LR after 2000 warmup steps, then 31.6% of the peak after 80% of the training tokens and 10% after 90%; loss curves of the multi-step schedule (80% + 10% + 10%) against cosine over 100B tokens, and of the 80/10/10, 70/15/15 and 60/20/20 splits; "generally seems to match performance of cosine learning rates"

DeepSeek's variant of WSD replaces the linear decay with two steps. The slide's "two decay steps of 10% each" means two stages of 10% of the tokens each; the excerpt on the slide image gives the levels: the learning rate drops to 31.6% of its peak at 80% of training and to 10% at 90%. Since $0.316 \approx \sqrt{0.1}$, each step cuts the rate by the same factor of about 3.2. The same logic holds as for WSD: everything before the first drop is shared, so a shorter run can branch off it. In the left plot the multi-step curve runs above cosine until the 80% mark, then drops and ends level with it; the right plot shows the 70/15/15 and 60/20/20 splits ending in about the same place.

::note aside 20:32 | Why two decay phases instead of one? "I am not really sure why this is. It hasn't really been a thing that has taken off afterwards."

::slide 23 | IsoFLOP curves of bits-per-byte on the validation set against non-embedding FLOPs per token M, for budgets from 1e17 to 3e20; fitted optimal model scale M and data D against compute C = MD, extrapolated to DeepSeek LLM 67B at 4.5e23 FLOPs: 4.3e11 FLOPs/token and 1.04e12 tokens

For the model-data trade-off DeepSeek uses method 2, IsoFLOPs ([L9](#/read/lecture_09)): fix a budget, train several model sizes at that budget, find the bottom of the U-shaped curve, repeat at more budgets, and fit how the optimal model and data grow with compute. DeepSeek measures model scale as non-embedding FLOPs per token, $M$, rather than parameters, so compute is simply $C = MD$. The professor finds these curves "much nicer" than MiniCPM's methods 1 and 3, and the conclusion more trustworthy: "the Chinchilla laws have been replicated by other people at large-ish compute ranges. So that analysis is generally sound." That is the lecture's first answer to slide 2's "does Chinchilla's approach actually work?".

The extrapolation lines on the slide check out: at $C = 4.5\times10^{23}$, $M \times D = 4.3\times10^{11} \times 1.04\times10^{12} \approx 4.5\times10^{23}$.

::animation fixture:isoflop-sweep | the same method-2 picture as slide 23(a): each larger budget's U-curve sits lower and its minimum moves right, and the line through the minima is the fitted optimal model size

::predict deepseek-isoflop-loss-prediction

::slide 24 | "the fitted scaling models (generally) accurately predict the final model losses": bits-per-byte on the validation set against training FLOPs C = MD from 10¹⁶ to past 10²⁴; a power law fitted to small models (grey circles) and the 7B and 67B models trained on 2T tokens (blue stars, at about 0.72 and 0.63) lying on its extension

This plot is "the final punchline of a lot of scaling law work".

The grey circles are small, carefully trained models at low compute; the dashed line is the power law fitted to them; the two stars are the 7B and 67B models DeepSeek actually trained, far to the right, and they land on the line. The out-of-sample prediction is the real test: a good fit to the small models is in-sample and proves nothing, while predicting the loss of the model you then train is evidence the whole recipe works. The professor's assessment is more hedged than the slide's "accurately": "it could be better, but it's not bad at all."

::note spoken 21:25 | He regrets that plots like this are now "somewhat rare in open-source release papers".
::video 21:25-22:08 | the out-of-sample test: the two stars are the trained models, and "it could be better, but it's not bad at all"
::kp deepseek-isoflop-loss-prediction

## The newer reports: less detail, more MoE {#tour}
source: lecture_11.pdf p25-p30 · video 22:34-30:26

The newer reports are a quick tour. Their scaling sections have shrunk, because the core machinery is "all known to everybody". What remains shows what labs now care about.

::slide 25 | Qwen 2.5's "Scaling Law for Hyper-parameters": optimal learning rate and batch size as functions of model size N and data size D, for dense models from 44M to 14B parameters and MoE models from 44M to 1B activated parameters, on 0.8B to 600B tokens; Qwen 3: "similar to Qwen2.5", for each dense or MoE model

**Qwen 2.5 and 3** do DeepSeek's analysis: scaling experiments for the optimal batch size and learning rate, a law, a prediction, now extended to MoE models. By Qwen 3 the report essentially says "we did the same as in Qwen 2.5". It has become "part of a very standard recipe".

::slide 26 | Kimi K2's sparsity scaling law: sparsity = total experts / activated experts; at constant activated parameters, more experts lowers training and validation loss; at a validation loss of 1.5, sparsity 48 needs 1.69×, 1.39× and 1.15× fewer FLOPs than sparsity 8, 16 and 32; K2 activates 8 of 384 experts; a second plot: doubling attention heads lowers validation loss by about 0.5% to 1.2%

**Kimi K2** shows what is new: MoE scaling laws. With most labs switching to MoEs, the question becomes how sparse to be. K2 varies FLOPs and sparsity (total experts over active experts, here with 8 active) and finds the expected result, "the more sparse your network, the better your validation loss, given a FLOP". The value is in the numbers, which the slide image prints: to reach a validation loss of 1.5, sparsity 48 saves a factor 1.69 in FLOPs over sparsity 8, but only 1.15 over sparsity 32. Returns are diminishing, and more experts cost infrastructure complexity, so K2 stops at 48: 8 of 384 experts per token. This is the kind of decision a scaling law is for.

::slide 27 | Hunyuan-Large: IsoFLOP curves of training loss against activated parameters for budgets from 5.0e18 to 9.5e19, fitted by quadratics, and the optimal activated parameters against compute extrapolated to 58.1B; "Optimal ratio – 96-1 (data to active param)"

**Hunyuan** does IsoFLOPs over the *activated* parameters of an MoE at a fixed sparsity (sparsity is not searched), and lands on 96 tokens per active parameter.

::predict isoflop-ratios-in-the-wild
::worked isoflop-ratios-in-the-wild

::slide 28 | Llama 3: IsoFLOP curves of validation loss against training tokens from 6e18 to 1e22 FLOPs ("39-1 ratio"), and "compute-to-downstream scaling": normalized NLL per character of the correct answer against compute, then benchmark accuracy against that NLL, a sigmoid through scaling-law models, Llama 2 models and the Llama 3 405B prediction

**Llama 3** also runs IsoFLOPs and gets 39 tokens per parameter, "slightly different but similar" to the others. 

The more interesting panel is the right one, a two-step map from compute to a benchmark. Step one: more compute lowers the negative log-likelihood of the correct answer, a clean line. Step two: lower NLL means higher accuracy, along a sigmoid fitted to the points. Chained, they predict the 405B model's accuracy. The professor is cautious: there are "systematic deviations from the curve", and he does not "buy that this curve is like the one truth". Still, it shows that log loss and downstream accuracy are tightly coupled "in many cases", which matters since everything so far has been about log loss.

::slide 29 | MiniMax-01: loss, optimal model size and optimal tokens against compute (PFLOP/s-days) for softmax attention, lightning (linear) attention and a hybrid, from 70M to 7B parameters

**MiniMax-01** uses scaling laws for an architecture decision, one of the uses [L9](#/read/lecture_09) promised. Lightning attention (a linear attention), full softmax attention and a hybrid of the two are scaled side by side: do any of them get worse with scale, or need very different model sizes? "For the most part, no": the three are comparable and want similar sizes. That result justified the hybrid architecture in the deployed model.

::slide 30 | recipes so far: DeepSeek (assume most transformer hyperparameters are invariant to scale; scaling analysis on batch and LR; IsoFLOPs for sizing with a piecewise-linear schedule); MiniCPM (muP to make the transformer and LR scale invariant; a piecewise-linear schedule to get samples for Chinchilla method 3); and the less detailed recent reports: Qwen (LR/batch), Kimi K2 (MoE scaling), Llama 3 and Hunyuan (just IsoFLOPs), MiniMax (architecture decisions)

The recap puts it side by side. DeepSeek's key move is the scaling analysis of batch and learning rate; MiniCPM's is muP. Both use a piecewise-linear schedule to make Chinchilla cheap. The newer reports add MoE scaling, architecture decisions, and replicated IsoFLOPs. Why so little detail now? The professor's guess: Chinchilla and learning-rate scaling are "fairly well understood by everybody", so there is no extra value in printing them.

The ratios themselves span a wide range: Chinchilla's 20, Llama 3's 39, MiniCPM's and Hunyuan's roughly 96. Note that Hunyuan's is per *active* parameter of an MoE, so it is not on the same footing as the dense ones.

::note skip 23:39 | Qwen, Kimi K2, Hunyuan, Llama 3 and MiniMax-01 get "quickly" covered, as an overview of what open releases do.
::note aside 29:47 | Asked how scaling changes once post-training is accounted for: "I think that's still a big, open question." Post-training can change what pre-training you should do; the closest work, on coverage or diversity in pre-training as a predictor of post-training, is "still pretty nascent".
::kp isoflop-ratios-in-the-wild

## What should the learning rate and batch scale with? StepFun's grid {#stepfun}
source: lecture_11.pdf p31-p37 · video 30:26-41:39

The rest of the lecture maps onto the two examples. First the DeepSeek approach, scaling laws for learning rates, batch sizes and optimizers. Then the MiniCPM approach: "can we just reparameterize everything to make it so that we don't have to worry about scale?"

::slide 31 | "Optimizers choices / tuning can be tricky and scale sensitive": StepFun's optimal LR and batch against D (from slide 4), and the optimizer comparisons at several Chinchilla ratios; "How should we pick different optimizers?"

::slide 32 | the StepFun paper, "Predictable Scale: Part I, Step Law – Optimal Hyperparameter Scaling Law in Large Language Model Pre-training"; core question: how do we set LR and batch as we scale? (the DeepSeek / Qwen approach)

The hyperparameter story leans on a recent StepFun preprint. StepFun trains credible large models, and for this study they "burned a ton of compute" grid-searching hyperparameters. The professor does not think there is "a great, really, truly reliable, robust study on hyperparameters", but this one "is probably getting somewhat close".

::slide 33 | StepFun's table of published laws: OpenAI (LR 3.239×10⁻³ − 1.395×10⁻⁴·log N; batch 2e18·L^−4.76190), Microsoft (LR 1.3192e−5·N^−0.23·D^−0.32), DeepSeek (0.3188·C^−0.1250; 0.2920·C^0.3271), Porian (3.7·N^−0.36; 0.7576·N^0.703), MiniCPM (batch 2e18/L^6.24), MeiTuan, and Step Law (1.79·N^−0.713·D^0.307; 0.58·D^0.571), with relative errors 9.51‰, 9.25‰, 9.26‰, 3.71‰ and 0.94‰ for Step Law; "Critical batch: batch as a function of loss (OpenAI); Compute power law: poly function of compute (DeepSeek) .. Or something else?"

The table shows how little agreement there is. The proposed laws "don't even have the same inputs":
- **OpenAI (Kaplan)**: batch as a function of the loss, the critical batch of [L9](#/read/lecture_09). The exponent $-4.76 = -1/0.21$ is Kaplan's.
- **DeepSeek**: both as power laws of compute, as we just saw.
- **Porian**: both as functions of model size $N$.
- **MiniCPM**: batch as a function of loss again, with the 6.24 exponent of slide 12.
- **Step Law**: learning rate as a function of model size *and* data size, batch as a function of data size alone.

The last column is each law's error at predicting the optimum on StepFun's runs, in parts per thousand: Step Law's 0.94‰ against 3.7-9.5‰ for the others. That is in-sample for StepFun, so the professor warns: "not to take even this last row as gospel. A lot of these things are pretty brittle."

::predict lr-batch-functional-form-choice

::note spoken 32:18 | He reads the OpenAI batch law as "2eE18, capital L to the exponent", which matches what the table prints, 2e18·L^−4.76190. Kaplan's paper puts the constant B* at about 2×10⁸ tokens, so the 2e18 looks like the table's own error; the same 2e18 also appears in the MiniCPM row, whose own fit on slide 12 gives e^20.91 ≈ 1.2×10⁹.

::slide 34 | the approach, "purely empirical – grid search the space": a loss contour over learning rate and batch size with the global minimum (red x), the Step Law prediction (yellow star) beside it, DeepSeek's and Porian's predictions further out, and Microsoft's and OpenAI's LR laws as vertical lines far to the left; contours at +0.125%, +0.25%, +0.5%, +1% and +2% above the minimum; table of 18 dense models from 2.15×10⁸ to 1.07×10⁹ parameters on 4×10⁹ to 10¹¹ tokens

The design is DeepSeek's at larger scale: train a lot of models on a fine grid of learning rate and batch size, across a range of model sizes and data sizes, and find the optimum of each. The contour plot is one $(N, D)$ cell. Here the Step Law star sits almost on the measured minimum, while the other laws' predictions fall outside the inner contours.

::slide 35 | "Observation 1: loss over batch/LR are convex": loss against batch size at four fixed learning rates (3.45e−4 to 9.77e−4) and loss against learning rate at four fixed batch sizes (262,144 to 1,048,576), each a smooth U, cut from the 3-D loss surface

The first observation is about the landscape. The slice the professor describes is about 1B parameters and 100B tokens ("I believe"). Fix the batch and vary the learning rate, or the other way round, and every slice is a smooth, convex U. This matters more than it looks. Had the surface been "very jaggedy", you could have "very serious doubts about whether this kind of program of gridding the space is even viable". Convexity is what makes "the minimum" a well-defined thing to fit a law to.

::predict stepfun-convexity-batch-vs-data

::slide 36 | "Observation 2": optimal learning rate (left) and batch size (right) against data size D, one colour per model size from 59M to 1B; "batch is primarily dependent on dataset size"; "higher optimal LR with D (for fixed M), but this is likely more fragile if swapping to WSD – see e.g. InternLM scaling law paper (Zhou+ 2026)"

The right panel is the one to look at. Each colour is a model size, and all the colours fall on the same line: the optimal batch depends on the amount of training data and essentially nothing else. "I don't know if I've seen any contradicting evidence to this observation." On log-log axes the line is a power law in $D$: the table's $0.58\,D^{0.571}$, roughly the square root of the data.

The left panel is different. Each model size has its own line, lower for bigger models: bigger models want smaller learning rates. And along each line the optimal learning rate *rises* with data, which the professor calls "counterintuitive, actually". The slide adds that this trend is "likely more fragile" under a WSD schedule, and he notes that other papers argue the dependence on $D$ should be reversed.

::widget fixture:lecture_11--hp-laws | switch to "set N and D separately": moving right (bigger model, same data) leaves the optimal batch unchanged, moving up (more data) raises it as D^0.571, while the LR responds to both directions with opposite signs

::slide 37 | "Observation 3 (?) robustness": loss landscapes for MoE models at sparsity ratios N_a/N = 0.27 and 0.58, and for three other datasets (bilingual, code integration, code-dominant), each with the global minimum and the Step Law prediction marked

Do the laws carry over? To MoEs, "to a certain extent": the predicted optima land near the measured ones "as long as you're roughly controlling for active parameters". To other data, less so: change the training data and both optima "shift a little bit", so "all of these phenomena are likely contingent" on the data. The slide's own "(?)" says the same. He adds a reassuring aside: practitioners already know roughly where a good learning rate lies (1e-3 or 1e-4 for a model you would train tomorrow), and these laws agree that the right range is fairly stable.

::kp lr-batch-functional-form-choice
::kp stepfun-convexity-batch-vs-data

### Along the Chinchilla path, StepFun becomes DeepSeek

Before moving on, the professor brings the final law back up and resolves an apparent contradiction. StepFun says the learning rate goes *up* with data. DeepSeek says it goes *down* with compute. Both can be right:

$$ \eta_\text{opt} = 1.79\, N^{-0.713} D^{0.307}, \qquad B_\text{opt} = 0.58\, D^{0.571} $$

In his words, the batch should be "roughly square root of the number of data times some constant", and the learning rate should go "upwards with data but downwards with the model size". Now remember that these experiments mostly assume Chinchilla-style scaling, where "N and D are both driven by compute": each grows like $C^{0.5}$. Then the two opposite pulls on the learning rate combine.

::predict stepfun-laws-along-chinchilla-path

With $N, D \propto C^{0.5}$:

$$ \eta_\text{opt} \propto C^{0.5(-0.713 + 0.307)} = C^{-0.203}, \qquad B_\text{opt} \propto C^{0.5 \times 0.571} = C^{0.286} $$

The learning rate falls and the batch grows with compute, which is DeepSeek's law, "albeit with different exponents" ($-0.125$ and $0.327$ for DeepSeek). He says the two effects "would cancel"; with these exponents they cancel only partly, and the model-size term wins. A 100× compute increase along the path (10× in $N$ and in $D$) multiplies the optimal learning rate by $10^{-0.713} \times 10^{0.307} = 10^{-0.406} \approx 0.39$.

Off the path the two laws part ways, and this is why the choice of variable matters. Keep a 1B model fixed and train it on 10× more data. StepFun predicts a learning rate $10^{0.307} \approx 2.0$ times *higher* and a batch $10^{0.571} \approx 3.7$ times larger. A compute-only law sees 10× more compute and predicts a *lower* learning rate: wrong direction. A law in compute is only valid along the allocation it was fitted on. As a sanity check on the absolute numbers: at 1B parameters and 100B tokens the Step Law gives $\eta \approx 1.6\times10^{-3}$ and $B \approx 1.1\times10^{6}$ tokens, in the region of slide 34's minimum.

::widget fixture:lecture_11--hp-laws | along the Chinchilla path, slide compute up: the path (slope 1) is flatter than the same-LR line (slope 0.713/0.307 ≈ 2.3), so moving out along it crosses into lower LR, while moving straight up (more data, same N) crosses into higher LR
::video 38:04-39:03 | how "up with data, down with model size" becomes "down with compute" once N and D are both driven by compute

In some sense, he says, this is "the newest and most large-scale version of the DeepSeek analysis".

Should you then just use these published laws instead of running your own grid? It depends on your regime. If you are close to where StepFun ran its grid, they are "probably really good defaults", better than "something I pull out of my hat". For your own big pre-training run there will be differences: someone believes in "a big weight decay", the architecture differs, and nobody knows whether those scale the same. That is why the reports in the tour kept redoing Chinchilla and the hyperparameter fits: to check they are "roughly first-order correct".

::note aside 41:22 | "Scaling laws have this very scientific feel to them ... But ultimately, a big part of scaling laws is still vibes": whether another team's experimental setting is close enough to yours to transfer is a judgement call.
::kp stepfun-laws-along-chinchilla-path

## Optimizers: why small-scale wins often vanish {#optimizers}
source: lecture_11.pdf p38-p41 · video 39:22-49:21

Optimizers did not fit into the architecture lecture, and they belong here anyway: their behaviour depends on scale. The section is less a prescription than a story about what we can and cannot know from small experiments.

::slide 38 | left: NanoGPT speedrun validation loss against wall-clock time on 8×H100 for Adam (139 ms/step), DistributedShampoo (179 and 154 ms/step), SOAP (301 ms/step) and Muon (142 ms/step), Muon reaching 3.28 first; right: speedup over AdamW against model size (130M, 300M, 520M, 1.2B) at 8× Chinchilla: Muon from about 1.38 to 1.10, Soap from about 1.40 to 1.10, NAdamW from about 1.18 to 1.09; "optimizers' speedup w.r.t. AdamW decreases with model size"

The left plot is the NanoGPT speedrun, which inspired Assignment 1: a tiny model, trained as fast as possible to a fixed loss of about 3.28. Muon (purple) beat Adam (blue) by a wide margin, at almost the same cost per step (142 against 139 ms). "Wow, lots of big gains from just changing the optimizer." Then the right plot, from a larger study: Muon's speedup over AdamW is about 1.38× at 130M parameters and shrinks to about 1.10× at 1.2B. The gain fades with scale. "This is a really tricky and important part of how we do research": a scale-dependent change looks great small, and then?

::predict optimizer-comparison-confounders

::slide 39 | "Fantastic Pretraining Optimizers and Where to Find Them" (Wen, Hall, Ma, Liang): left, loss of 130M models where AdamW with LR 6e−4 needs about 10,000 steps to reach the loss that AdamW with LR 8e−3, Mars and Nesterov AdamW reach at about 5,000, "tuning LR of AdamW leads to 2x speedup"; right, loss against weight decay for Lion (optimum "wd ≈ 0.6") and AdamW (optimum near 0.1)

The study is the one by Kaiyue Wen, David Hall, Tengyu Ma and Percy Liang, and its first lesson is "technically not a scaling thing, but a very important general empirical machine learning thing". **Problem 1: tuning.** Tune the AdamW baseline badly (LR 6e-4) and every new optimizer looks great; tune its learning rate properly (8e-3) and the baseline gets a 2× speedup and "all my gains went away". The right plot makes the same point for weight decay: Lion's best weight decay is about 0.6, AdamW's about 0.1. Run both at one shared value and one of them is handicapped. Different optimizers need different hyperparameters, and likely different hyperparameter *scaling*.

::slide 40 | "Problem 2 – fairly significant scale dependence": the speedup-versus-model-size plot, and loss of 520M models against Chinchilla ratio 1 to 8, where the matrix-based optimizers (Muon, Soap, Kron, solid) consistently beat the scalar-based ones (AdamW, Mars, NAdamW, dashed); "always check scaling with respect to compute and chinchilla ratios. These are often major confounders to performance!"

**Problem 2: scale dependence.** Two axes always matter when you develop an algorithm.
- **Compute.** Hold the data-to-model ratio fixed and scale up; this is the left plot, and the gains shrink.
- **The Chinchilla ratio**, tokens per parameter. Some algorithms work well when the model is much bigger than the data (an implicit regularizer helps, or a data-inefficient method wins on compute grounds). Others work well with lots of data per parameter, because "you can pack knowledge more efficiently into the parameters".

Even papers with good scaling hygiene often vary only model size and never the ratio. In this study the ratio turned out not to matter ("In this case, it's not"): at 520M the gaps between optimizers stay about the same from 1× to 8× Chinchilla. "But this is not always the case."

::video 45:58-47:20 | why the Chinchilla ratio is its own confounder, and that in this study the gains did not depend on it

::slide 41 | Problem 2.5, "establishing scaling is nontrivial!": IsoFLOP parabolas of Paloma macro loss from 3e18 to 3e20 FLOPs; the fitted scaling law with held-out validation: fit left of a dotted line between 10²⁰ and 10²¹, extrapolated points "0.8% worse" at 10²¹ and "2.5% worse" at 10²², and "Run Diverged" at 10²³; "Cautious AdamC + Sqrt batch-size scaling of learning rates (Fixed with some more careful parametrization / scaling / optimizer changes)"

**Problem 2.5: even a beautiful trend can break.** This comes from Will Held's work with Marin, Percy Liang's open training project, which publishes its failed runs too. They ran a standard Chinchilla analysis with a recipe built on Cautious AdamC (an Adam variant) and square-root batch-size scaling of the learning rate. The IsoFLOP parabolas are clean and the fitted line is "really beautiful-looking". Then the extrapolation: the held-out run at $10^{21}$ FLOPs is 0.8% worse than predicted, the one at $10^{22}$ is 2.5% worse, and at $10^{23}$ the run diverges. "Even good-looking scaling trends for many orders of magnitude can suddenly bite you." The fix was more careful muP-style parametrization and a change of optimizer, which gave clean scaling over more orders of magnitude.

So the advice to run scaling experiments and scale your learning rates is easier said than done: a "fairly nontrivial fraction of the time" you end up with a plot like this, unsure what went wrong.

::note aside 48:16 | Aloud the clean trend holds "up to this dashed vertical line at 10 to the 20 something"; the slide image places the end of the fitted region between 10²⁰ and 10²¹ and the divergence at 10²³.
::kp optimizer-comparison-confounders

## Muon: an optimizer for matrices {#muon}
source: lecture_11.pdf p42-p43 · video 49:21-57:22

Muon gets a few slides because it now passes the course's inclusion test for new research: "Has it been in a big training run?" It has. It is also an interesting idea.

::slide 42 | Muon's algorithm: initialize B₀ = 0; for each step compute the gradient G_t, accumulate momentum B_t ← μB_{t−1} + G_t, compute O_t ← NewtonSchulz5(B_t), update θ_t ← θ_{t−1} − ηO_t; beside it the NanoGPT speedrun plot; "Optimizer for 'matrix valued' parameters. NewtonSchultz (approximately) orthogonalizes the matrix B_t = USVᵀ → UVᵀ"

Ignore line 5 for a moment and this is plain SGD with momentum: take the gradient, add it into the running buffer $B_t$ with factor $\mu$, step along $B_t$. The new idea starts from an observation. A gradient step treats all parameters alike, but a language model's parameters are not alike. Some are vectors, such as RMSNorm gains. Others are matrices, in attention and the MLPs, and a matrix has a spectrum: singular values and singular directions. Muon works on the spectrum.

Line 5 takes the update matrix, writes it as its singular value decomposition $B_t = U S V^\top$, and replaces it with

$$ B_t = U S V^\top \;\longrightarrow\; U V^\top $$

Every singular value becomes 1. Big singular values are shrunk back to unit, small ones are expanded up to unit; the singular directions are kept.

::predict muon-orthogonalized-update
::worked muon-orthogonalized-update

The intuition is the professor's analogy with Adam. AdaGrad and Adam divide by the size of the gradient, so "every coordinate is roughly going to be the same size". Muon works in the spectral norm instead, so "every kind of direction is going to be unit size". Note that this changes the update's direction, not just its length: a normalization would keep the 100:10:1 proportions of singular values 10, 1 and 0.1; orthogonalization makes them 1:1:1, so the weak directions get relatively much more push.

It only makes sense for a matrix, since only a matrix has a singular value decomposition. So Muon is applied to the matrix parameters, and the vector parameters still get "something like AdamW". And the orthogonalization is not computed by an SVD. **Newton-Schulz** (here five iterations, "NewtonSchultz5") approximates it using only matrix multiplies, so it is fast on GPUs and "not exactly the orthogonalization".

::widget fixture:muon-orthogonalize | change the entries of the 2×2 update: the raw update's ellipse axes (singular values) stretch and shrink, but U Vᵀ always maps the circle to the circle; lower the Newton-Schulz steps to see the approximation before it converges
::video 51:33-52:40 | singular values pushed to 1, and the analogy: Adam equalizes coordinates, Muon equalizes directions

::slide 43 | "Muon and scaling": the NanoGPT speedrun ("very small!"), the scaling study where the speedup decreases with model size, and Kimi K2's training loss over about 15 trillion tokens; "scaling gains are tricky to measure, but clearly muon 'works' at scale"

The story so far ran: great on the speedrun, then scaling studies found the gain shrinking with scale, and the professor thought "the story was closed here", since nobody would spend the compute to scale it fully. Then Kimi K2 came out, trained "fully with muon", with "a few bells and whistles to prevent muon from blowing up on you" after they hit instability issues. K2 is an outstanding model and its loss curve looks reasonable. So Muon works at scale. Is it better than Adam there? "Kimi K2 certainly doesn't have any ablation, so we don't know at that scale."

The point of the story: it is very hard to know whether something works at scale, and yet small experiments transferred upward are how the field does science. Respect the speedrun; good ideas start there and eventually reach the big models.

::note aside 54:40 | Whether Muon beats AdamW at large scale: "Those are interesting questions to which we don't yet have the answers."
::note aside 55:03 | Asked whether SVD is fast on GPUs: it is not, but Muon does not do an SVD; Newton-Schulz is "a matrix multiply only finite iteration algorithm". Three ideas combine: treat the parameter as a matrix, orthogonalize, and do it cheaply.
::note aside 55:43 | Asked about hyperparameters: they differ from Adam's. Jeremy Bernstein's line of work argues every layer should have its own learning rate, or even its own optimizer; "maybe the future of optimization", though tuning each one is not appealing.
::kp muon-orthogonalized-update

### Which hyperparameters get the grid?

A student asked the obvious question about all these grids: everyone grids learning rate and batch size, but surely they interact with all the other hyperparameters?

::predict grid-sensitive-sweep-the-rest

They do, and "in general, yes, you are worried about hyperparameter interactions". But the number of combinations grows exponentially with the number of hyperparameters, so nobody can grid the whole space. What labs do is grid "the stuff that's most concerning and most sensitive", learning rate first ("the most important thing to tune"), together with batch size. Then for things like weight decay they "might do a univariate sweep afterwards to be locally optimal". A 6 × 4 grid of learning rate and batch plus four one-at-a-time sweeps of five values each costs $24 + 20 = 44$ runs; a full joint grid over all six would cost $6 \times 4 \times 5^4 = 15{,}000$. The price of the cheap version is in the word "locally": a coordinate-wise search can miss an interaction, such as AdamW's coupling of learning rate and weight decay.

::video 56:34-57:19 | why labs do not grid everything ("they grow exponentially"), and what they do for weight decay instead
::kp grid-sensitive-sweep-the-rest

## muP from scratch: the goal and the initialization {#mup-init}
source: lecture_11.pdf p44-p47 · video 57:22-1:05:15

The final part is the MiniCPM approach in depth. muP "is a bit of a mysterious object": many papers and implementations exist, and "I don't think all of them even agree on what the underlying math is". What follows is the professor's interpretation of the core ideas they share.

::slide 44 | "Recall – the maximum update parametrization makes appealing claims": training loss against log₂ learning rate for widths 128 to 8192; under standard practice the optimum shifts left as width grows, under muP ("our work") all widths share one optimum; Table 2 of the muP paper: for a model r times wider, AdamW LR l/r for matrix-like weights and l for others, init variance σ/r for matrix-like weights, output multiplier τ/r

The goal is the picture. Under standard practice, widening the model shifts the optimal learning rate (left: the minimum moves from about $2^{-10}$ toward $2^{-14}$ as width goes from 128 to 8192). Under muP the minimum stays put (right). To get there, muP allows three kinds of knobs: per-layer initialization, per-parameter learning rates, and scaling of things like residual connections with model size. The table is the muP paper's rule for a model $r$ times wider: matrix-like weights get learning rate $l/r$ and init variance $\sigma/r$, the output layer's multiplier becomes $\tau/r$, and everything else is unchanged. MiniCPM's Table 7 (slide 8) is an implementation of exactly this.

::slide 45 | CerebrasGPT, 0.1B to 13B models trained with the Chinchilla recipe: Pile test loss against training FLOPs for Cerebras-GPT, its scaling law, and a μP variant (with Pythia, GPT-J and GPT-NeoX for comparison); percentage deviation from the scaling law, where the standard models swing between about −0.9% and +1.0% and the μP models stay near −0.5%; hyperparameters tuned on a 40M μP model and transferred up to 2.7B

Does it work in practice? Cerebras trained 0.1B to 13B models with the usual Chinchilla recipe, plus a muP variant whose hyperparameters were tuned on a 40M proxy and transferred. The core finding: with muP, the scaling fits are more stable. The right plot shows it as deviation from the fitted law. The standard models zig-zag between about $-0.9\%$ and $+1.0\%$; the muP models stay on their projected trend, "almost right on the money".

::note skip 59:46 | More examples of models trained with muP are skipped in favour of the conceptual foundations and the math.
::note aside 1:00:12 | Reading pointers: the paper on the next slide ("for me, was the clearest"), Greg Yang's tensor programs series ("somewhat inscrutable"), Jeremy Bernstein's review in a different framework ("very accessible"), and accounts by physicists.

::slide 46 | "A Spectral Condition for Feature Learning" (Greg Yang, James B. Simon, Jeremy Bernstein), "a very accessible 'muP for babies' paper"; "muP is based off the following assertion. As a function of the width of the network n_l.. A1: The activations at initialization should remain Θ(1). A2: After one gradient step, the change in activation should be Θ(1)"; "if individual activations are Θ(1), then the norm should be Θ(√n_l)"

The derivation follows that paper, and its style is "a very physicist way of thinking": take a scaling limit (make the width $n_l$ large), assert invariants that should hold in the limit, and see what they force. Two invariants, with $\Theta(1)$ meaning "neither growing nor shrinking with width":
- **A1.** At initialization, each activation stays $\Theta(1)$. If activations "blow up with the network size, or they shrink to 0 with the network size, I have chosen a wrong parameterization."
- **A2.** After one gradient step, the change in each activation is also $\Theta(1)$.

A note on bookkeeping used throughout: if each of the $n_l$ coordinates of a layer's activation vector $h_l$ is $\Theta(1)$, the vector's length is $\|h_l\|_2 = \Theta(\sqrt{n_l})$.

### A2 is feature learning

Why insist on A2? The professor adds an aside "in case some of you are theorists". A2 is "what people call feature learning": a gradient step should change the network's internal representations by a significant amount, the same at every width. The contrast is the **neural tangent kernel** (NTK) regime, where "the change in activations actually vanish[es] as a function of the network width". A very wide network in that regime barely moves its hidden features and fits the data with essentially the random features it started with, like a kernel method. Its loss can still fall, through many tiny weight changes, but it learns no new features. muP rules this out by demanding a $\Theta(1)$ change at every width.

::predict a2-is-feature-learning
::video 1:01:23-1:02:05 | A2 named as "feature learning", and the NTK contrast where activation changes vanish with width
::kp a2-is-feature-learning

### Deriving the initialization from A1

::slide 47 | "Deriving muP (condition A1)": deep linear network h_l = W_l h_{l−1}, W_l ~ N(0, σ² I_{n_l×n_{l−1}}), matrix concentration ‖W_l‖_* → σ(√n_{l−1} + √n_l), ‖h_l‖₂ ≈ ‖W_l‖_*‖h_{l−1}‖₂; pick σ = (√n_l/√n_{l−1})(√n_l + √n_{l−1})^{−1} = Θ((1/√n_{l−1}) min(1, √(n_l/n_{l−1}))); inductive assumption ‖h_{l−1}‖₂ = Θ(√n_{l−1}); inductive case ‖W_l‖_* → √n_l/√n_{l−1}, so ‖h_l‖₂ = √n_l + o(√n_l); "this is a kind of 'worst case' derivation – the ≈ is an upper bound"

Take the simplest possible network: a deep linear network, $h_l = W_l h_{l-1}$, no biases, no nonlinearities. Layer $l$ maps width $n_{l-1}$ (fan-in) to width $n_l$ (fan-out), and its weights are initialized as independent Gaussians with standard deviation $\sigma$. The question is what $\sigma$ keeps A1 true at every layer and every width. Here $\|W\|_*$ is the operator (spectral) norm, the largest factor by which $W$ can stretch a vector.

::worked mup-two-conditions-and-init

Two remarks on the steps. First, the operator-norm fact is "standard matrix concentration", asserted without proof: a Gaussian matrix's largest stretch grows like the *sum* $\sqrt{n_{l-1}} + \sqrt{n_l}$, not the product. Second, $\sigma$ was "drawn out of a hat": guess it, plug it into the induction, and check that it returns $\|h_l\| = \sqrt{n_l}$ at every layer, up to lower-order terms he does not track. The slide calls this a worst case because $\|W h\| \le \|W\|_* \|h\|$ is an upper bound: the chosen $\sigma$ guarantees activations cannot grow, rather than matching their typical size.

What does the result change compared with the standard initialization $1/\sqrt{n_{l-1}}$ ("1 over square root fan-in")? When fan-out is at least fan-in, the $\min$ is 1 and the two agree up to a constant. When fan-out is smaller, muP shrinks the init by $\sqrt{n_l/n_{l-1}}$.

::predict mup-two-conditions-and-init

For a layer from width 4096 down to 1024, $\sqrt{1024/4096} = 1/2$: the $\Theta$ form gives $\sigma = (1/64)(1/2) = 1/128$, half the standard $1/64$. With the exact constant of the slide, $\sigma = (1/2)/(32 + 64) = 1/192$, a third of it.

::widget fixture:lecture_11--mup-init | with the exact σ the worst-case output size per coordinate is exactly 1 at every fan-out/fan-in ratio; standard 1/√fan-in matches muP's Θ form once fan-out ≥ fan-in, but its bound climbs like √(fan-in/fan-out) when fan-out is smaller
::note aside 1:03:18 | Said with hedging: the ≈ holds "where the fan-out is, I think, not much bigger than the fan-in". For a Gaussian matrix the typical stretch is about σ√n_l, against the operator norm σ(√n_l + √n_{l−1}), so the two are closest when fan-out is much *larger* than fan-in; either way the slide's point, that the ≈ is an upper bound, stands.
::video 1:03:39-1:05:07 | the induction: assume √n_{l−1} in, apply the operator-norm bound, and the chosen σ returns √n_l
::kp mup-two-conditions-and-init

## muP's learning rate, from condition A2 {#mup-lr}
source: lecture_11.pdf p48-p50 · video 1:05:15-1:12:38

"Condition A2 is a little bit hairier, to be honest." The derivation is restricted to the simplest case: SGD, one example per batch, a deep linear network.

::slide 48 | "Deriving muP (condition A2)": for SGD on a linear layer the update is a rank-one outer product, ΔW_l = −η_l ∇_{h_l}ℓ h_{l−1}ᵀ, so ‖ΔW_l h_{l−1}‖₂ = ‖ΔW_l‖_*‖h_{l−1}‖₂; the change in activation Δh_l = W_l Δh_{l−1} + ΔW_l(h_{l−1} + Δh_{l−1}); "assuming that the leading order terms don't cancel": W_lΔh_{l−1} = Θ(√n_l) by induction and the A1 argument, ΔW_l h_{l−1} = ‖ΔW_l‖_*√n_{l−1}, so ‖ΔW_l‖_* = Θ(√n_l/√n_{l−1}), and ΔW_lΔh_{l−1} = O(‖ΔW_l‖_*√n_{l−1})

Two facts set up the argument.

**The update is rank one.** Backpropagation through $h_l = W_l h_{l-1}$ gives the weight gradient as an outer product of the output gradient and the input activation. So the SGD update $\Delta W_l = -\eta_l \nabla_{h_l}\ell\; h_{l-1}^\top$ is rank one and points along $h_{l-1}$, which means it stretches $h_{l-1}$ by exactly its operator norm: $\|\Delta W_l h_{l-1}\| = \|\Delta W_l\|_* \|h_{l-1}\|$.

**The change in activation has three terms.** Expanding $h_l + \Delta h_l = (W_l + \Delta W_l)(h_{l-1} + \Delta h_{l-1})$:

$$ \Delta h_l = W_l\,\Delta h_{l-1} + \Delta W_l\, h_{l-1} + \Delta W_l\, \Delta h_{l-1} $$

The first term is the old weights acting on the change from below; the others are what the weight update itself adds. A2 asks for $\|\Delta h_l\| = \Theta(\sqrt{n_l})$. Assuming the terms do not cancel, each must be of that size. The first is by induction plus the A1 argument. The second is $\|\Delta W_l\|_* \sqrt{n_{l-1}}$, so

$$ \|\Delta W_l\|_* = \Theta\!\left(\sqrt{n_l / n_{l-1}}\right) $$

The update's operator norm must scale as the square root of fan-out over fan-in. That is the target; the question is which learning rate delivers it.

He is candid about the style: "the right way to think about this is not like, OK, Tatsu is sitting here doing rigorous mathematics ... It's physicists math." The argument tracks orders of magnitude, and "you can see the sketchy pieces adding up", such as assuming no cancellations.

::slide 49 | "Deriving muP (condition A2) part 2": pick the LR so that ‖ΔW_l‖_*√n_{l−1} = Θ(√n_l); "suppose that the loss update also scales O(1)": Δℓ ≈ Θ(⟨ΔW_l, ∇_{W_l}ℓ⟩) = Θ(‖ΔW_l‖_F‖∇_{W_l}ℓ‖_F) = Θ(‖ΔW_l‖_*‖∇_{W_l}ℓ‖_*); with Δℓ = O(1) and ‖ΔW_l‖_* = Θ(√n_l/√n_{l−1}), ‖∇_{W_l}ℓ‖_* = Θ(√n_{l−1}/√n_l); so η_l = Θ(n_l/n_{l−1}); "[with Adam, ‖ΔW_l‖_*√n_{l−1} = Θ(√η_l)]"

The missing piece is the size of the gradient, and for that he makes one more assumption, "probably the least palatable of the assumptions, in my opinion": the loss changes by $\Theta(1)$ per step at every width, i.e. the model makes appreciable progress whatever its size. ("Not quite sure that's palatable to me.") Then a first-order Taylor expansion, plus the fact that for rank-one matrices the Frobenius and operator norms coincide, turns the loss change into a product of norms:

::worked mup-lr-derivation-and-recap

In words: A2 fixes how big the update must be, $\sqrt{n_l/n_{l-1}}$; the $\Theta(1)$ loss change fixes how big the gradient is, $\sqrt{n_{l-1}/n_l}$; and for SGD the learning rate is the ratio, $\eta_l = \Theta(n_l/n_{l-1})$, "a fan-out over fan-in ratio for a learning rate for a layer". For a square hidden layer ($n_l = n_{l-1}$) that is $\Theta(1)$: plain SGD needs no width correction there.

**Adam is different**, and the lecture gives only the result: $\eta_l = \Theta(1/n_{l-1})$, "I'm not going to even touch the derivation". The slide's bracketed line for Adam does not by itself produce that result, so here is the standard argument in the same style (ours, not the lecture's). Adam divides each coordinate of the gradient by its own running size, so the update's entries are all of size about $\eta$ regardless of how big the gradient was. For the rank-one gradient here, the update is then roughly $\eta$ times a rank-one matrix of $\pm 1$ entries, whose operator norm is $\sqrt{n_l\, n_{l-1}}$. So $\|\Delta W_l\|_* \sqrt{n_{l-1}} \approx \eta\, n_{l-1} \sqrt{n_l}$, and setting this to A2's $\sqrt{n_l}$ gives $\eta = 1/n_{l-1}$. Because Adam throws away the gradient's size, the width factor that SGD got for free from the shrinking gradient has to be carried by $\eta$ itself.

::video 1:08:49-1:10:13 | the "least palatable" assumption Δℓ = Θ(1), and how the SGD learning rate falls out as fan-out over fan-in

::slide 50 | "muP mini recap": init std Θ((1/√n_{l−1}) min(1, √(n_l/n_{l−1}))), learning rate n_l/n_{l−1} (for Adam 1/n_{l−1}); standard parametrizations: init 1/√n_{l−1}, learning rate Θ(1); "Differences – LR changes for Adam, also init diffs when fanout n_l < fanin"

The recap, in one table:

| | init std | SGD learning rate | Adam learning rate |
|---|---|---|---|
| standard (SP) | $1/\sqrt{n_{l-1}}$ | $\Theta(1)$ | $\Theta(1)$ |
| muP | $\frac{1}{\sqrt{n_{l-1}}}\min\!\left(1, \sqrt{n_l/n_{l-1}}\right)$ | $n_l/n_{l-1}$ | $1/n_{l-1}$ |

Two differences matter in practice. The init differs only when fan-out is smaller than fan-in. And under Adam the learning rate is per layer, $1/\text{fan-in}$: "layers that have big fanins for Adam get smaller learning rates". This is MiniCPM's "LR of each 2-D tensor times $1/(d_m/d_{base})$" from slide 8.

::predict mup-lr-derivation-and-recap

Widen every hidden layer from 1024 to 4096 and train with Adam: muP multiplies the hidden-layer learning rate by $1024/4096 = 1/4$, while the standard parametrization leaves it unchanged. With plain SGD and square layers, muP leaves it unchanged too, since $n_l/n_{l-1} = 1$ before and after.

::widget fixture:mup-vs-sp-table | slide the width multiplier with both widths tied: the SP learning rate and the SGD-muP rate stay flat, the Adam-muP rate falls as 1/m, and the init std departs from SP's only once you make the fan-out smaller than the fan-in
::note slip 1:10:49 | Aloud the muP init becomes "1 over square root nl minus 1 times a component that is 1 over the square root fanout over fanin", and different "if this ratio is not smaller than 1". The slide's factor is min(1, √(fan-out/fan-in)), not its inverse, and it changes the init only when that ratio is below 1. His next words, "closer to the square root fanout over fanin", agree with the slide.

What is the high-level point? Beyond the rules themselves, the method: take a scaling limit of the network, assert invariants, add assumptions, and read off the constraints they put on the hyperparameters. It is "a general principle for coming up with different kinds of algorithms or hyperparameter scalings", and quite unlike the usual way of working in CS or ML.

::kp mup-lr-derivation-and-recap

## Does muP survive a real language model? {#mup-practice}
source: lecture_11.pdf p51-p57 · video 1:12:38-1:15:52

The derivation covered one linear layer. A real transformer has embeddings, attention, gated MLPs, norms and fancy optimizers, most of which the theory does not cover. The closing evidence is a stress test of muP by an independent researcher.

::slide 51 | "A Large-Scale Exploration of μ-Transfer" (Lucas Dax Lingle): per-parameter muP rules for a transformer of width M, H heads of size D, MLP width F; init variance and Adam LR (Θ and exact) for the embedding W^E (1, 1), the attention W^AQ, W^AK, W^AV (1/M, 1/M) and W^AO (1/(HD)), the MLP input W^FI (1/M) and output W^FO (1/F), and the unembedding W^U (init 1/M², LR 1/M); attention scale 1/D instead of 1/√D

The slide image names the paper and gives muP's rules for a whole transformer, one row per weight. The hidden matrices follow the recap: init variance and Adam learning rate both $1/\text{fan-in}$ ($1/M$ for the matrices reading the width-$M$ residual stream, $1/F$ for the MLP's output matrix, which reads the width-$F$ hidden layer). The embedding is the exception, with variance and learning rate both constant, since its fan-in is the vocabulary, which does not grow with width. The output (softmax) layer gets the smaller init variance $1/M^2$. One more change is in the text box: attention logits are scaled by $1/D$ instead of the usual $1/\sqrt{D}$.

::slide 52 | "Q1: Does muP work as claimed? When we scale widths, is optimal LR constant?": validation loss at base LR 2⁻¹⁰ to 2⁻² for widths 128, 512 and 2048; baseline μP optimal at 2⁻⁶ for every width (3.695, 2.953, 2.511), and likewise with projection biases (3.705, 2.947, 2.529); "each model being 4x wider (and 16x larger) than the last"

The headline replicates. Each model is 4× wider and so 16× larger than the last, and under muP the best base learning rate is $2^{-6}$ at all three widths, with or without biases on the projections: "exactly the learning rate optimality invariance".

::slide 53 | "What is muP robust to?": SwiGLU and squared-ReLU activations, large or small batches, initialization variations such as zero attention, RMSNorm gains, exotic optimizers (Lion), regularizers; "Which of these (if any) break muP?"

Technically, SwiGLU does not fit the theory, nor do the initialization variants, nor RMSNorm, nor a heavily modified optimizer. Which of these "really break muP"? "For the most part, these mostly work with muP." Three things do not.

::predict mup-robustness-in-practice

::slide 54 | "In our arch – RMSNorm has learnable gains. This turns out to break muP": with vector gains the optimum moves from 2⁻⁴ (widths 128, 512) to 2⁻⁸ (2048), with scalar gains from 2⁻⁴ to 2⁻⁶, both marked as not transferring; trainable gains also hurt the largest models; "But these gains can be removed with little loss of perf.."

**Learnable RMSNorm gains** break it: with per-coordinate gains the best learning rate jumps from $2^{-4}$ at widths 128 and 512 to $2^{-8}$ at 2048. A gain is a learned per-coordinate multiplier, and it can undo the width scaling that the init and learning-rate rules were designed to keep. The remedy is cheap, because the gains can be removed with little loss. ("In our arch" is the paper's phrase.)

::slide 55 | "There are other, exotic optimizers based on just gradient signs. Do they transfer?": the AdamW and Lion algorithms side by side (Lion updates with sign(c_t)); with Lion the optimum is 2⁻¹⁰ at width 128 but 2⁻⁸ at 512 and 2048, and large learning rates diverge (losses above 10); marked as not transferring

**Sign-based optimizers** such as Lion break it too. The text layer only asks "Do they transfer?", but the slide image answers with a red cross, and so does the professor: "It also seems to break if you use more exotic optimizers like Lion." A sign update is "spiritually similar to muon", so the same caution may apply there.

::slide 56 | "What about strong (0.1) weight decay? – this is maybe the only significant muP failure": with decoupled weight decay the optimum is 2⁻⁸ at width 128 but 2⁻⁶ at 512 and 2048, and width 2048 at 2⁻² reaches 6.594; marked as not transferring; the SGD with L2 regularization versus SGD with decoupled weight decay algorithm

**Strong decoupled weight decay** (0.1) is "the thing that seems most concerning": "maybe the one stress test that it does fail".

::note spoken 1:14:58 | The slide calls strong weight decay "maybe the only significant muP failure", but aloud three things break muP: learned RMSNorm gains, Lion, and large decoupled weight decay. Weight decay is the most concerning of the three.

::slide 57 | "Overall, muP generally seems useful – insofar that SP is quite a bit more unstable": standard parametrization's optimum moving from 2⁻⁶ (width 128) to 2⁻⁸ (512) to 2⁻¹⁰ (2048), with losses above 7 at large LRs; the large-scale muP experiment, 2M to 10B parameters (widths 128 to 8192), optimal at 2⁻⁶ every time (3.766, 2.983, 2.459, 2.167)

Is muP useful, then? The comparison on this slide is the strongest case. Under the standard parametrization the optimal learning rate moves from $2^{-6}$ to $2^{-8}$ to $2^{-10}$ as width grows 16×, a 16× drop, and the learning rate that was best for the small model gives a loss above 7 at width 2048. (That is L9's rule of thumb that wider networks want smaller learning rates, roughly $1/\text{width}$; see [L9](#/read/lecture_09).) Under muP one base learning rate, $2^{-6}$, is best all the way from 2M to 10B parameters. In the professor's words, with SP the optimum shifts "very predictably, but very greatly"; with muP it does not.

So muP is "one of the many tools in the toolkit to try to control hyperparameter drift", not done and dusted. There is promise in the muP-style program, and also in just fitting scaling laws, the DeepSeek way.

::video 1:14:15-1:15:11 | which deviations break muP aloud: learned RMSNorm gains, sign-based Lion, and large decoupled weight decay
::kp mup-robustness-in-practice

## Recap: scaling in the wild {#recap}
source: lecture_11.pdf p58 · video 1:15:52-1:16:58

::predict scaling-in-practice-challenges

::slide 58 | challenges in scaling "in practice": 1. setting model architecture hyperparameters (width, etc.), 2. setting optimizer hyperparameters (LR, batch), 3. the compute needed to fit the big Chinchilla sweep; some solutions: 1. assume stability (or use muP), 2. search for the optimal LR/batch at small scale and either keep it fixed or predict its scaling, 3. use alternative learning-rate schedules (WSD-like)

A Chinchilla fit tells you how to split a budget between parameters and tokens. It does not tell you the rest, and the rest is scale sensitive. The slide pairs three challenges with three answers, and the lecture has shown each one at work:

| challenge | answer | where we saw it |
|---|---|---|
| architecture hyperparameters (width and its init) | assume they are stable, or make them so with muP | DeepSeek assumes; MiniCPM and CerebrasGPT use muP; derived in the last part |
| optimizer hyperparameters (learning rate, batch) | grid them at small scale, then hold them fixed or fit their scaling | MiniCPM's batch-vs-loss line, DeepSeek's laws in compute, StepFun's laws in N and D |
| the compute cost of the Chinchilla sweep | a WSD-like schedule, so one run branches into many decays | MiniCPM's WSD, DeepSeek's two-step variant |

The professor's closing is a caution. The first presentation of scaling laws "makes it sound like a science": draw this line, follow these procedures, and you will know what happens at scale. In reality it is "a lot more messy and a lot more unknown". People do use scaling laws to choose architectures, optimizers and hyperparameters, "but there is really an art to it", since you never know whether a trend will extrapolate forever (slide 41's diverged run is the reminder). muP and learning-rate searches are ways of controlling hyperparameter drift, "but there's no silver bullet yet".

::note aside 1:16:48 | "Maybe next year, there will be a module that's like, we solved it, but not quite yet."
::video 1:15:52-1:16:53 | why scaling "in the wild" is messier than the procedure suggests, and the two families of fixes
::kp scaling-in-practice-challenges
