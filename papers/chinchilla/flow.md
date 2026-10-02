---
title: Chinchilla · Training compute-optimal language models, read through
minutes: 35
---
In 2022 a DeepMind team (Hoffmann et al.) asked a planning question: if your training budget in FLOPs is fixed in advance, how big should the model be, and how many tokens should it see? Three different fits over 400+ training runs gave the same answer, grow both equally, and a 70B model trained on that rule (Chinchilla) beat the 280B Gopher at the same cost. After this read-through you can set up the question, run each of the three estimation methods in your head, turn a budget into a model size and token count, and say what the evidence does and does not show.

## Why ask again how to spend a training budget? {#question}
source: chinchilla:§abstract · chinchilla:§1 · Table 1 · Eq. 1 · Fig. 1

Training a large language model is usually a one-shot affair. You know how many accelerators you have and for how long; that fixes a compute budget before the first step. Because the run is too expensive to repeat, the choices made before it starts matter a great deal, and the biggest two are the number of parameters $N$ and the number of training tokens $D$.

By 2022 the field had an answer from Kaplan et al. (2020), the first big scaling-law study (recapped in [L9](#/read/lecture_09)). Their fits said that when compute grows 10×, the model should grow about 5.5× and the data only about 1.8×. In other words, spend new compute mostly on parameters. The models of the time followed that advice.

::figure chinchilla:table1 | the column to read is the last one: almost every model, from 175B to 530B parameters, saw about 300B tokens

| Model | Parameters | Training tokens | Tokens per parameter |
|---|---|---|---|
| LaMDA | 137B | 168B | 1.2 |
| GPT-3 | 175B | 300B | 1.7 |
| Jurassic | 178B | 300B | 1.7 |
| Gopher | 280B | 300B | 1.1 |
| MT-NLG 530B | 530B | 270B | 0.5 |
| Chinchilla | 70B | 1.4T | 20 |

(The last column is ours: tokens divided by parameters.) Model size had grown threefold in two years while the token count stood still at about 300 billion. The paper's claim is that this was a mistake: these models are "significantly undertrained" (abstract).

::predict compute-optimal-question

The question gets a precise form. Write $L(N, D)$ for the final pre-training loss of a model with $N$ parameters trained on $D$ tokens. The compute it costs is a fixed function $\text{FLOPs}(N, D)$. For a budget $C$, the best split is

$$ N_{opt}(C),\ D_{opt}(C) = \operatorname*{argmin}_{N,D \;\text{s.t.}\; \text{FLOPs}(N,D) = C} L(N,D) \qquad (1) $$

Two things are held fixed here, and both matter later. First, the *training* compute: a bigger model trained to convergence would reach a lower loss, but it would cost more, so it is not a candidate. Second, nothing about serving the model: inference cost is not in the objective at all.

A footnote settles which loss: the smoothed *training* loss. The runs use fewer tokens than the corpus holds (an "infinite data" regime), so the model is always scored on text it has not yet trained on, and the training loss is an unbiased estimate of the test loss.

To estimate $N_{opt}$ and $D_{opt}$ the authors trained over 400 models, from 70M to over 16B parameters, on 5B to over 400B tokens (the abstract says "5 to 500 billion"), many of them at several training lengths. Their headline, in Fig. 1, overlays the three resulting fits with Kaplan's.

::figure chinchilla:fig1 | optimal parameters against training FLOPs: the three new fits form a shallower line than Kaplan's, and GPT-3, Gopher and MT-NLG sit far above the new lines (too big for their compute), with Chinchilla on them

From that frontier the paper predicts that at Gopher's budget the optimal model is about 4× smaller and should see about 4× more tokens. They then trained it: Chinchilla, 70B parameters on 1.4T tokens. It beat Gopher, and being 4× smaller it is also cheaper to fine-tune and serve, a benefit the paper stresses although Eq. 1 never counts it.

::note why | "Compute-optimal" means best loss for a fixed *training* budget, not cheapest to deploy. [L9](#/read/lecture_09) returns to this: once a model is served heavily, labs deliberately train smaller models on more tokens than Chinchilla's rule.
::kp compute-optimal-question

## What does a FLOP budget count? {#budget}
source: chinchilla:§3.3 · chinchilla:§F · Table A4

Eq. 1 needs a formula for $\text{FLOPs}(N, D)$. The paper uses the standard approximation from Kaplan et al. and from [L2's FLOP counting](#/read/lecture_02):

$$ C \approx 6ND $$

The reason is short. In the forward pass each parameter takes part in one multiply and one add per token, 2 FLOPs, so the forward pass costs about $2ND$. The backward pass costs twice the forward pass, because each weight matrix needs one matmul for its own gradient and one to pass the gradient down to the layer below: $4ND$. Together, $6ND$.

::worked supp-six-nd-budget

Appendix F checks the approximation. The authors count FLOPs layer by layer: embeddings, the query, key and value projections, the attention logits and softmax, the feed-forward block and the final logits, still with backward = 2 × forward. They count the embedding matrices both in $N$ and in the FLOPs, so the two stay consistent. Table A4 compares their exact count with $6ND$ for six model shapes.

::figure chinchilla:tableA4 | the ratio column, exact FLOPs over 6ND, runs from 0.99 (6.8B) to 1.10 (305M): 6ND is within 10% at the paper's sizes

The gap comes from the terms that do not scale with parameters, mostly the attention logits, which grow with sequence length squared. At the paper's context lengths and widths they are a small correction; at much longer contexts they would not be. The paper calls the difference "very small" and says it does not affect the analysis.

::predict supp-six-nd-budget

The 10% matters for one sentence later. The paper says Chinchilla and Gopher used "the same number of FLOPs". By $6ND$ they did not: $6 \times 2.8\times10^{11} \times 3\times10^{11} = 5.04\times10^{23}$ for Gopher and $6 \times 7\times10^{10} \times 1.4\times10^{12} = 5.88\times10^{23}$ for Chinchilla, 17% more. The "same" rests on the exact count, not on $6ND$.

::note slip | The paper prints two values for Gopher's budget. Table 3 and the Fig. 2 caption use $5.76\times10^{23}$ FLOPs (the "Gopher unit"); Appendix F says its more careful count gives $6.3\times10^{23}$. This read-through uses $5.76\times10^{23}$ wherever the paper's tables do.

One consequence of $C = 6ND$ is used everywhere below. If $N_{opt} \propto C^a$ and $D_{opt} \propto C^b$, then $6 N_{opt} D_{opt} = C$ forces $a + b = 1$. Whatever share of each new factor of compute goes to parameters, the rest goes to tokens. So the whole debate is one number: how $a$ compares with $\tfrac12$.
::kp supp-six-nd-budget

## What did the earlier study do differently? {#kaplan}
source: chinchilla:§2 · chinchilla:§B · Fig. A1 · chinchilla:§D.1

Kaplan et al. also trained many models and fitted power laws to pick the best model size for a budget, so on the surface the two studies did the same thing. The paper names two differences in how the runs were set up, and both push Kaplan's answer toward bigger models.

**Difference 1: one learning-rate schedule for every run.** Some background first. Language models are trained with a learning rate that starts high and is lowered toward the end; this paper uses a *cosine* schedule, which follows half a cosine wave from the peak learning rate down to a tenth of it over a chosen number of steps, the *cycle length*. The loss drops noticeably in the last part of the cycle, when the learning rate gets small. So a run's loss at step $t$ depends not only on $t$ but on how far along its own schedule it is.

Kaplan et al. used a fixed number of training tokens and a fixed schedule for all models, and read losses at intermediate points to stand for shorter training. Take a cosine schedule set for 130B tokens and read its loss after, say, 20B. At that point the learning rate is still near its peak, so the loss is higher than a run whose schedule was set to *end* at 20B would reach. The paper calls those intermediate losses "overestimates". Every short-training point looks worse than it really is, so training on fewer tokens looks less effective, and the fit concludes that model size should grow faster than data.

Appendix B measures how much the schedule matters. One model is trained with the cosine cycle set to 1, 1.1, 1.25, 1.5, 2 and 5 times the number of steps actually taken, and the loss is read at the target step.

::figure chinchilla:figA1 | the six curves at the target step: cycles up to 1.25× the run length end close together; 1.5×, 2× and 5× end clearly higher, because their learning rate never came down

::predict cosine-cycle-length

The rule the paper takes from this: decay the learning rate 10× over approximately the number of tokens you will actually train on. Overshooting the run length by more than about 25% clearly hurts. A footnote adds two details: decaying 10× and decaying to zero over the same steps differ little (10× is slightly better), while decaying only 5× is clearly worse. Another footnote draws the practical lesson: you must fix not only the model size but also the *training length* before training begins, because the schedule depends on it. All three approaches below follow this rule; every run's schedule matches its own horizon.

**Difference 2: much smaller models.** Most of this paper's runs have more than 500M parameters and go up to 16B, while many of Kaplan's had fewer than 100M. That matters because the paper sees slight curvature in the frontier (Appendix E, at the end of this read-through): fits from very small models predict differently from fits on larger ones.

The section also positions the paper among other work, briefly. Other hyperparameters (learning rate, batch size, optimiser, width-to-depth ratio) are set from existing heuristics, not studied; the paper uses slightly shallower models than one proposal recommends, for better wall-clock speed on its hardware. Mixture-of-experts models and retrieval-augmented models are mentioned as alternatives to dense scaling. Retrieval effectively multiplies the training tokens a model sees (about 10× in one cited system), a first hint that data may matter more than people assumed.

::note why | Pay attention to the direction of the bias: a long fixed schedule makes small-data points look *worse*, not better. That is what tilts the fit toward big models on few tokens. The paper tests the consequence directly in Appendix D.4, in the Approach 1 section below.
::kp cosine-cycle-length

## What does it mean to fit a scaling law? {#fitting}
source: chinchilla:§3 · Table 2

Section 3 opens with the shared plan of all three approaches: train a range of models, varying both model size and token count, and fit an estimator of how the best choices scale with compute. All three assume the answer is a power law,

$$ N_{opt} \propto C^{a}, \qquad D_{opt} \propto C^{b} $$

and the paper admits up front that future work may need to allow for curvature at large sizes.

A power law $y = k\,x^a$ becomes a straight line once you take logs: $\log y = \log k + a \log x$. The exponent is the slope on log-log axes, and the constant $k$ only shifts the line up or down. So "fitting a scaling law" means putting points on log-log axes and fitting a line, as in [L9](#/read/lecture_09). Two habits follow. Sample the points evenly in $\log x$, so each decade counts equally in the slope (Approach 1 uses 1,500 logarithmically spaced FLOP values for this reason). And remember that using the line beyond the data assumes the slope stays the same out there.

::worked supp-power-law-fitting
::predict supp-power-law-fitting

The exponent is all you need to read the rule. A ratio of two points on a power law depends only on the slope: 10× the compute multiplies $N_{opt}$ by $10^a$. With $a = 0.50$ that is $3.16\times$; with Kaplan's $a = 0.73$ it is $5.37\times$ (the paper's "5.5×"), and the data then grows only $10^{0.27} = 1.86\times$ (the paper's "1.8×").

::widget fixture:chinchilla--allocation-table | read the "×10 budget" lines: Approach 1 multiplies N and D by 3.16 each, Kaplan's line multiplies N by 5.37; on the top panel those exponents are the slopes of the coloured lines and the grey dashed one
::kp supp-power-law-fitting

## Approach 1: what is the lowest loss reachable at each FLOP count? {#approach-1}
source: chinchilla:§3.1 · Fig. 2 · chinchilla:§D.1 · chinchilla:§D.4 · Fig. A4

The first approach fixes a family of model sizes, from 70M to over 10B parameters, and varies only how long each one trains. Every size is trained four times, with horizons (token counts) that span a factor of 16, and each run decays its learning rate 10× over its *own* horizon. Peak learning rates run from $2\times10^{-4}$ for the smallest models to $1.25\times10^{-4}$ for the largest (Appendix D.1).

Now plot all the training curves against FLOPs instead of steps. Each curve starts high and falls; curves of small models fall fast and then flatten, big ones start slower and keep going. At any FLOP count, one run is lowest. That lowest run is the best split of that much compute between parameters and tokens, which is exactly Eq. 1. Joining the lowest points across all FLOP counts gives the *envelope*: the lowest loss reachable for each budget.

::figure https://arxiv.org/html/2203.15556v1/scaling_11.png | left: all runs against FLOPs with the envelope of minimal loss along their bottom; centre and right: the model size and token count of each envelope point against FLOPs, straight lines on log-log axes, with the green projection at Gopher's budget of 5.76·10²³ FLOPs

::worked approach-1

Two details make the envelope honest. Each training curve is smoothed (a Gaussian window of 10 steps) and interpolated, so the loss is known at every FLOP count and the envelope can be read at 1,500 log-spaced values, not only at run endings. And a footnote reports where the selected points fall.

::predict approach-1

All selected points lie within the last 15% of their runs. That is the schedule effect from the last section seen from the other side: a run in the middle of its schedule still has a high learning rate, so at that FLOP count a shorter run that has already finished its decay is lower. The envelope is traced by run *endings*, which is why the paper concludes that the cosine cycle should match the planned token count.

Fitting lines through the envelope points gives $a = 0.50$ and $b = 0.50$: parameters and tokens should grow at the same rate. At Gopher's budget the Approach 1 line gives 67B parameters and 1.5T tokens (Table 3, below).

### A head-to-head test against Kaplan's rule

Approach 1's section points to a direct test in Appendix D.4. At $10^{21}$ FLOPs, Kaplan's fit says the optimal model has 4.68B parameters; Approach 1 says 2.86B. The authors trained a 4.74B and a 2.80B model, with the same batch size (0.5M tokens), the same peak learning rate ($1.5\times10^{-4}$, decayed 10×) and the same depth-to-width ratio, so that the split of compute is the only real difference.

::predict why-kaplan-differed

At the same budget the tokens follow from $D = C/6N$: about $1\times10^{21} / (6 \times 4.74\times10^9) \approx 35$B tokens for the bigger model and $1\times10^{21} / (6 \times 2.80\times10^9) \approx 60$B for the smaller.

::figure chinchilla:figA4 | the two training curves at 10²¹ FLOPs: the 2.80B model, trained on more tokens, ends with the lower loss

The smaller model wins at the end of training. This is the cleanest evidence in the paper that the disagreement is real and not a quirk of curve fitting: two runs, everything matched except the allocation.

::widget fixture:chinchilla--allocation-table | pick the 1B row and multiply its budget by 9 (about 1.1·10²¹ FLOPs): Approach 1 gives 3.0B parameters on about 61B tokens, while the Kaplan readout at the same budget gives about 5.0B on about 36B; the same budget split two opposite ways, and the gap widens every decade because 0.73 ≠ 0.50

So the paper's account of the gap is about the *data fed to the fit*, not its form: both studies fit power laws, but Kaplan's short-training points were biased upward by a fixed long schedule, and its models were mostly small, where the frontier bends.

::note aside | [L9](#/read/lecture_09) tells the story differently. The course leads with later reanalyses: Kaplan counted parameters without the embedding (and output) layers, and tuned warmup and batch size poorly for tiny models; correcting those moves Kaplan's exponent to about 0.5. The professor doubts that the decay schedule is the main cause. The paper itself never raises parameter counting; it counts embeddings in $N$ and in the FLOPs (Appendix F) and blames the schedule and the model range.
::kp approach-1
::kp why-kaplan-differed

## Approach 2: at a fixed budget, which model size is best? {#approach-2}
source: chinchilla:§3.2 · Fig. 3

The second approach asks Eq. 1's question head-on. Pick a FLOP budget. Train models of many sizes, each on exactly as many tokens as the budget allows, $D = C/6N$, with its cosine schedule matched to that token count. Plot each model's final loss against its size. That curve is an *IsoFLOP profile*: every point on it cost the same.

The paper uses 9 budgets from $6\times10^{18}$ to $3\times10^{21}$ FLOPs and sizes up to 16B parameters. Only final losses are used, unlike Approach 1, which used points along whole training curves.

::predict approach-2

Along one profile, moving right doubles $N$ and halves $D$. Too far left, the model is too small to use the compute well; too far right, it sees too few tokens to learn its parameters. So the loss has a valley, and its bottom is $N_{opt}(C)$.

::figure chinchilla:fig3 | left: nine U-shaped loss curves, one per budget, each with a clear minimum, the minima moving right as the budget grows; centre and right: the minima's model size and token count against FLOPs, projected in green to Gopher's budget

::animation fixture:isoflop-sweep | along one IsoFLOP curve N rises as D = C/6N falls and the loss has one minimum; each larger budget's minimum sits further right and lower, so both N_opt and D_opt grow with C

To locate each bottom, the authors fit a parabola to each profile (in log model size) and take its vertex. They also make sure each budget was trained on a wide enough range of sizes to see a clear minimum. That condition is not a formality: if all the sampled sizes sit on one side of the valley, the parabola's vertex is an extrapolation.

::widget fixture:chinchilla--isoflop-parabola | slide the centre of the sampled sizes until the lowest loss is at the edge of the range: the readout switches to "not bracketed" and the vertex drifts away from the true valley; centre them again and the vertex lands on it (the samples here are the Approach 3 fitted surface, a stand-in for the unprinted Fig. 3 losses)

Lines through the nine minima give $a = 0.49$ and $b = 0.51$, almost the same as Approach 1. At Gopher's budget, the Fig. 3 projection is about 63B parameters on 1.4T tokens (read from the figure, as shown on the CS336 slides; the text does not print it).

::note aside | [L9](#/read/lecture_09) calls IsoFLOP profiles the professor's favourite method, "very easy and very robust", and the one that outlived the paper: labs now run IsoFLOP sweeps for any trade-off, from diffusion language models to mixture-of-experts sparsity.
::kp approach-2

## Approach 3: can one formula describe every run? {#approach-3}
source: chinchilla:§3.3 · Eq. 2 · Eq. 3 · chinchilla:§D.2 · Eq. 9-11

The third approach stops reading minima off plots. It takes every final loss from Approaches 1 and 2 and fits one formula for the loss as a function of both $N$ and $D$:

$$ \hat L(N,D) = E + \frac{A}{N^{\alpha}} + \frac{B}{D^{\beta}} \qquad (2) $$

Each term has a meaning.
- $E$ is the loss of an ideal model of the data: the *entropy of natural text*, the part of next-token prediction that no model can remove.
- $A/N^{\alpha}$ is the price of finite size: even a perfectly trained transformer with $N$ parameters falls short of the ideal.
- $B/D^{\beta}$ is the price of finite training: the model takes only a limited number of optimisation steps on a sample of the data, so it never reaches the best transformer of its size.

Appendix D.2 derives this from a classical *risk decomposition*. Call $f^\star$ the best possible predictor, $f_N$ the best transformer with $N$ parameters, and $\bar f_{N,D}$ what training actually produces after one pass over $D$ tokens. Then the loss splits exactly into three gaps:

$$ L(\bar f_{N,D}) = L(f^\star) + \big(L(f_N) - L(f^\star)\big) + \big(L(\bar f_{N,D}) - L(f_N)\big) $$

The first is $E$; the second depends only on $N$; the third depends on how far optimisation got, which the paper models as depending only on $D$. Theory suggests the shapes. For simple two-layer networks the approximation gap shrinks like $1/N^{1/2}$, and stochastic gradient methods converge no faster than $1/D^{1/2}$. So power laws are the natural guess for both terms, with exponents to be fitted.

::note why | The additive form is a modelling choice, and it carries the result. Because the $N$ term and the $D$ term are separate, the best split of a budget can be solved in closed form (next section). It is the same "sum of power laws" form as the joint data-model law in [L9](#/read/lecture_09).

**How it is fitted.** The five constants $(A, B, E, \alpha, \beta)$ are chosen to minimise a Huber loss between the predicted and observed *log* loss:

$$ \min_{A,B,E,\alpha,\beta} \sum_{\text{runs } i} \text{Huber}_\delta\Big(\log \hat L(N_i, D_i) - \log L_i\Big) \qquad (3) $$

A Huber loss is squared error for small residuals and absolute error for large ones, so a few runs that fit badly cannot dominate. The paper uses $\delta = 10^{-3}$, with L-BFGS started from a grid of initial values to avoid local minima. Larger $\delta$ overfits the small-compute runs and predicts held-out large runs poorly; smaller $\delta$ changes nothing.

::worked approach-3-parametric-fit

The fitted constants appear only in Appendix D.2:

$$ \hat L(N,D) = 1.69 + \frac{406.4}{N^{0.34}} + \frac{410.7}{D^{0.28}} \qquad (10) $$

Both exponents are below the theoretical $\tfrac12$, and the paper adds that future methods should try to raise them.

::predict approach-3-parametric-fit

Here is Eq. 10 applied to the paper's own two big models (our arithmetic). Chinchilla, 70B on 1.4T tokens: $406.4/(7\times10^{10})^{0.34} = 0.083$ and $410.7/(1.4\times10^{12})^{0.28} = 0.163$, so $\hat L = 1.69 + 0.08 + 0.16 = 1.94$. Gopher, 280B on 300B tokens: $0.052$ and $0.251$, so $\hat L = 1.99$. Gopher's extra parameters buy back only 0.03 on the first term, while its short training costs 0.09 on the second. The fit predicts Chinchilla's win before the run is made.
::kp approach-3-parametric-fit

## What allocation does the formula imply? {#frontier}
source: chinchilla:§3.3 (Efficient frontier) · Eq. 4 · Fig. 4

With $\hat L$ in hand, Eq. 1 becomes a calculus problem. Impose $C \approx 6ND$, so that $D = C/6N$, and minimise over $N$ alone:

$$ \hat L(N) = E + A N^{-\alpha} + B\left(\frac{6N}{C}\right)^{\beta} $$

A bigger model lowers the second term but, at a fixed budget, leaves fewer tokens and raises the third. The optimum is where the two slopes cancel. Solving gives power laws "by construction":

$$ N_{opt}(C) = G\left(\frac{C}{6}\right)^{a}, \quad D_{opt}(C) = G^{-1}\left(\frac{C}{6}\right)^{b}, \quad G = \left(\frac{\alpha A}{\beta B}\right)^{\frac{1}{\alpha+\beta}}, \quad a = \frac{\beta}{\alpha+\beta}, \quad b = \frac{\alpha}{\alpha+\beta} \qquad (4) $$

::worked approach-3-frontier

The exponents are crossed: the share of compute that goes to $N$ is set by the *data* exponent $\beta$, and vice versa. The term that decays more slowly is the one worth spending more on. Here $\beta = 0.28 < \alpha = 0.34$: the data term is the stubborn one, so tokens get slightly more than half of each new factor of compute.

::predict approach-3-frontier

The paper reports $a = 0.46$, $b = 0.54$: the only approach with $b$ noticeably above $a$. A small difference with a long reach: since $D_{opt}/N_{opt} \propto C^{b-a} = C^{0.08}$, tokens per parameter keep rising with compute instead of staying constant.

::figure https://arxiv.org/html/2203.15556v1/approach_3_v2.png | left: contours of equal fitted loss over model size and tokens, with the efficient frontier in blue, a straight line in log-log space that crosses each contour at its cheapest point, and dashed IsoFLOP lines; right: the matching IsoFLOP slices of the fit; the caption projects 40B parameters at Gopher's budget

::widget fixture:chinchilla--parametric-frontier | press "N = N_opt" and compare the two bars: at the optimum the parameter term over the data term is 0.82 = β/α at every budget, not 1; then set α = β and the bottom panel's tokens-per-parameter line goes flat

A common mistake is to think the two correction terms are equal at the optimum. The first-order condition equates $\alpha \cdot A/N^\alpha$ with $\beta \cdot B/D^\beta$, so the terms themselves stand in the ratio $\beta/\alpha \approx 0.82$ at every budget.

::note slip | Two printed numbers of Approach 3 do not follow exactly from the printed constants. From the rounded exponents, $a = 0.28/0.62 = 0.452$ and $b = 0.548$, which round to 0.45 and 0.55, not the printed 0.46 and 0.54. And Eq. 4 with Eq. 10's rounded constants gives $N_{opt} \approx 32$B at Gopher's $5.76\times10^{23}$ FLOPs (on about 3.0T tokens), not the 40B of the Fig. 4 caption. Small changes in the exponents move the projection a lot over five decades; the paper's own figures presumably use unrounded values.
::note aside | Later work ([L9](#/read/lecture_09)) re-extracted the data points from the paper's plots and refitted Approach 3: the original fit's residuals were biased, and the refit stays near a constant 20 tokens per parameter, in line with Approaches 1 and 2.
::kp approach-3-frontier

## Do the three approaches agree, and what does the rule say? {#scaling}
source: chinchilla:§3.4 · Table 2 · Table 3 · chinchilla:§D.3 · Table A3 · chinchilla:§C · Table A2

Three methods, different fitting procedures, partly different runs. Table 2 puts their exponents side by side, with 10th-90th percentile ranges from bootstrapping (refitting on 80% of the data, 100 times).

::figure chinchilla:table2 | the three rows of new exponents all sit near 0.5, far from Kaplan's 0.73 / 0.27 in the last row

| Approach | $a$ ($N_{opt} \propto C^a$) | $b$ ($D_{opt} \propto C^b$) |
|---|---|---|
| 1. Minimum over training curves | 0.50 (0.488, 0.502) | 0.50 (0.501, 0.512) |
| 2. IsoFLOP profiles | 0.49 (0.462, 0.534) | 0.51 (0.483, 0.529) |
| 3. Parametric modelling of the loss | 0.46 (0.454, 0.455) | 0.54 (0.542, 0.543) |
| Kaplan et al. (2020) | 0.73 | 0.27 |

The headline, in the abstract's words: "for every doubling of model size the number of training tokens should also be doubled." Approaches 1 and 2 give nearly the same model sizes. Approach 3 predicts even smaller models at large budgets, and the paper explains why. Its residuals are larger for the low-compute runs ($C \le 10^{21}$), so the Huber loss treats many of them as outliers and effectively weights the high-compute runs more. Since the frontier bends (it is slightly concave in $\log N_{opt}$ against $\log C$, Appendix E), a fit dominated by the large runs has a lower slope and predicts smaller models.

Table 3 turns Approach 1 into a lookup table: for each model size, the budget and token count at which that size is compute-optimal.

::figure chinchilla:table3 | read the tokens column against the parameters column: the ratio stays near 20 from 400M to 10T parameters

| Parameters | FLOPs | Gopher units | Tokens | Tokens per parameter |
|---|---|---|---|---|
| 400M | 1.92e19 | 1/29,968 | 8.0B | 20.0 |
| 1B | 1.21e20 | 1/4,761 | 20.2B | 20.2 |
| 10B | 1.23e22 | 1/46 | 205.1B | 20.5 |
| 67B | 5.76e23 | 1 | 1.5T | 22.4 |
| 175B | 3.85e24 | 6.7 | 3.7T | 21.1 |
| 280B | 9.90e24 | 17.2 | 5.9T | 21.1 |
| 520B | 3.43e25 | 59.5 | 11.0T | 21.2 |
| 1T | 1.27e26 | 221.3 | 21.2T | 21.2 |
| 10T | 1.30e28 | 22,515.9 | 216.2T | 21.6 |

(The last column is ours.) This is where the famous **20 tokens per parameter** comes from. The phrase is not printed in the paper; it is the CS336 slides' summary of this table. Each row also satisfies $C \approx 6ND$: for the 1B row, $6 \times 10^9 \times 2.02\times10^{10} = 1.21\times10^{20}$. (The 67B row's 1.5T is rounded; $5.76\times10^{23}/(6 \times 6.7\times10^{10})$ is 1.43T.)

::predict equal-scaling-rule

With $a = b = 0.5$, a budget $k$ times larger multiplies both $N$ and $D$ by $\sqrt{k}$. For 4 Gopher units: $67\text{B} \times 2 = 134$B parameters on $1.5\text{T} \times 2 = 3$T tokens, and $6 \times 1.34\times10^{11} \times 3\times10^{12} = 2.4\times10^{24} \approx 4 \times 5.76\times10^{23}$. Equal scaling means equal *growth factors*, so the ratio $D/N$ stays near 20; it does not mean $N = D$.

::widget fixture:chinchilla--allocation-table | step the approach slider from 1 to 3: Approach 1's tokens-per-parameter line stays flat near 20 across nine decades of budget, Approach 3's printed rows climb from 23 to 143, and Kaplan's dashed line falls

The table's conclusions are blunt. The models of Table 1 are badly oversized for their budgets. A 175B model should be trained on trillions of tokens, not 300 billion. A 1-trillion-parameter model is unlikely to be the right choice unless the budget is about $10^{26}$ FLOPs. (The paper calls that "over 250×" Gopher's compute; by its own Gopher unit, $10^{26}/5.76\times10^{23} \approx 174$, and Table 3's 1T row is 221 units.) And the token counts needed are far beyond what anyone was training on, so collecting data becomes as important as scaling hardware. The paper is open about the uncertainty of extrapolating many orders of magnitude, but holds that for many current budgets smaller models on more tokens would have been better.

::note slip | The prose under Table 3 does not match the table. It says a 175B model should use $4.41\times10^{24}$ FLOPs and over 4.2T tokens, and a 280B model about $10^{25}$ FLOPs and 6.8T tokens; Table 3's rows say $3.85\times10^{24}$ and 3.7T, and $9.90\times10^{24}$ and 5.9T. The prose numbers satisfy $6ND$ themselves ($6 \times 1.75\times10^{11} \times 4.2\times10^{12} = 4.41\times10^{24}$), and they sit closer to Approach 2's rows in Table A3 (4.3T and 7.1T), so they were probably taken from a different version of the fit. Trust the table.

Appendix D.3 gives the same lookup for Approaches 2 and 3 (Table A3). Approach 2 tracks Approach 1 closely (67B on 1.7T tokens at about $6.9\times10^{23}$ FLOPs); Approach 3 asks for far more data (67B on 4.1T tokens at $1.71\times10^{24}$ FLOPs, and 10T parameters on 1,425.5T tokens).

::note slip | Table A3 has a typo in Approach 3's 175B row: FLOPs printed as $1.26\times10^{24}$, but its 12.0T tokens give $6 \times 1.75\times10^{11} \times 1.2\times10^{13} = 1.26\times10^{25}$, which also fits between the neighbouring rows ($1.71\times10^{24}$ for 67B, $3.52\times10^{25}$ for 280B). The 280B row is slightly off too: $3.52\times10^{25}$ FLOPs implies about 21.0T tokens under $6ND$, not the printed 20.1T.

Is the result specific to the training data? Appendix C repeats the IsoFLOP analysis (4 budgets) on two other datasets: C4, a filtered web crawl, gives $a = b = 0.50$; GitHub code gives $a = 0.53$, $b = 0.47$. Both are close to equal scaling, which the paper reads as independence from the dataset, as long as training stays under one epoch.
::kp equal-scaling-rule

## Building the test model: what exactly is Chinchilla? {#chinchilla}
source: chinchilla:§4 · chinchilla:§4.1 · Table 4 · chinchilla:§A · Table A1 · chinchilla:§G

A frontier fitted on models of at most 16B parameters (the IsoFLOP budgets stop at $3\times10^{21}$ FLOPs) makes a claim about $5.76\times10^{23}$ FLOPs, more than two decades further out. The paper tests it with one large run. For Gopher's budget, the analysis puts the optimal size between 40B (Approach 3) and about 70B (Approaches 1 and 2). The authors chose the large end, 70B parameters on 1.4T tokens, citing dataset and computational-efficiency considerations: a 40B model at the same budget would need even more tokens.

Chinchilla keeps Gopher's architecture and training setup, with a few changes:

::figure chinchilla:table4 | the two rows differ only in width and heads: both have 80 layers, Gopher with d_model 16,384 and 128 heads, Chinchilla with 8,192 and 64; Chinchilla's peak learning rate is 1·10⁻⁴ against Gopher's 4·10⁻⁵, and its batch doubles from 1.5M to 3M tokens midway (Gopher's from 3M to 6M)

- **Data mix.** Same corpus, MassiveText, with slightly different sampling proportions to supply 1.4T tokens.
- **Optimiser.** AdamW instead of Adam, which improves the language-modelling loss and downstream results after fine-tuning. A footnote notes that an AdamW model only overtakes an Adam one about 80% of the way through the cosine cycle, but ends clearly better.
- **Tokenizer.** A slightly modified SentencePiece tokenizer without NFKC normalisation; 94.15% of the vocabulary is shared with Gopher's, and the change helps with mathematics and chemistry.
- **Precision.** The forward and backward passes run in bfloat16, but the optimiser state keeps a float32 copy of the weights.

::note warning | These changes are confounds. Appendix G trains small models to measure them: at 680M parameters, Chinchilla's optimiser setup (AdamW plus the high-precision weight copy) clearly beats Gopher's, and at 417M and 1.4B AdamW alone beats Adam. So the Chinchilla-Gopher comparison tests the allocation together with these improvements; there is no 70B ablation that separates them. The cleanest test of the allocation alone is the matched two-run comparison at $10^{21}$ FLOPs in Appendix D.4.

Being 4× smaller than Gopher, Chinchilla also has a smaller memory footprint and costs about 4× less per token to serve. The paper treats this as a bonus, since training compute is only part of a model's lifetime cost: fine-tuning and inference add up too.

## Did the prediction hold? {#results}
source: chinchilla:§4.2 · Fig. 5 · Table 6 · Fig. 6 · Fig. 7 · Tables 7-10

Chinchilla is evaluated on largely the same suite as Gopher, so the two can be compared directly: 20 language-modelling sets, 57 MMLU subjects, 62 BIG-bench tasks, and reading comprehension, question answering and common-sense benchmarks.

**Language modelling.** Chinchilla has lower bits-per-byte than Gopher on every subset of The Pile, and beats Jurassic-1 (178B) on all but two (dm_mathematics and ubuntu_irc). On Wikitext103 its perplexity is 7.16 against Gopher's 7.75.

::figure chinchilla:fig5 | one bar per Pile subset, each showing how much lower Chinchilla's bits-per-byte is than Gopher's: every bar points the same way

The paper itself urges caution here. Chinchilla saw 4× more data, so overlap between training and test sets could inflate these numbers, and the authors put more weight on benchmarks where leakage is less of a concern.

::predict chinchilla-vs-gopher

**MMLU** (exam-style questions in 57 academic subjects, 5-shot):

| Model | Average accuracy |
|---|---|
| Random | 25.0% |
| Average human rater | 34.5% |
| GPT-3 | 43.9% |
| Gopher | 60.0% |
| **Chinchilla** | **67.6%** |
| Average human expert | 89.8% |
| Forecast for June 2022 / June 2023 | 57.1% / 63.4% |

Chinchilla gains 7.6 points over Gopher and beats what expert forecasters had predicted for the state of the art in June 2023. It passes 90% on four subjects. By task it is better on 51 of 57, equal on 2, and worse on 4 (college mathematics, econometrics, moral scenarios, formal logic).

::figure chinchilla:fig6 | per-subject difference, Chinchilla minus Gopher, sorted: almost all bars are positive, with four small negative ones at the end

::note slip | The abstract says 67.5% on MMLU; Table 6, §4.2.2 and the Fig. 6 caption say 67.6%. Use 67.6%.

**BIG-bench.** Average accuracy rises from 54.4% to 65.1% (+10.7 points), and Chinchilla is worse on only 4 of 62 tasks.

::figure chinchilla:fig7 | per-task difference on 62 BIG-bench tasks: four negative bars, the rest positive

**Reading comprehension, common sense, closed-book QA.** LAMBADA zero-shot: 77.4% against Gopher's 74.5% and MT-NLG's 76.6%. RACE-h and RACE-m: gains of more than 10 points (82.3 vs 71.6, 86.8 vs 75.1). On five common-sense benchmarks Chinchilla matches or beats Gopher and GPT-3 everywhere and beats MT-NLG 530B on all but PIQA. On TruthfulQA it reaches 43.6% zero-shot against Gopher's 29.5%, which the paper reads as evidence that better modelling of the pre-training data alone can help there. On Natural Questions it sets a closed-book state of the art, 31.5% 5-shot and 35.5% 64-shot; on TriviaQA it beats Gopher in every setting.

::note slip | For Natural Questions the prose gives Gopher's scores as 21% (5-shot) and 28% (64-shot); Table 9 prints 24.5% and 28.2%.

**Bias and toxicity.** On Winogender (does the model resolve a pronoun to the right occupation regardless of gender?) Chinchilla is more accurate than Gopher for every group, but unevenly: +3.2 points for male pronouns against +8.3 for female and +9.2 for neutral. Its unprompted samples are about as toxic as Gopher's (mean PerspectiveAPI score 0.087 against 0.081), consistent with the earlier finding that toxicity barely tracks language-modelling quality.

Put together: a model with a quarter of the parameters, trained on 4.7× the tokens (1.4T against 300B) for roughly the same compute, is better almost everywhere. That is the allocation prediction passing its one large-scale test, subject to the confounds noted above.
::kp chinchilla-vs-gopher

## What can the paper not show? {#limits}
source: chinchilla:§5 · chinchilla:§E · Fig. A5 · chinchilla:§C

The discussion restates the diagnosis: the race to bigger models at a fixed 300B tokens left those models well short of what their compute could buy, and all three approaches say Gopher was oversized. Then the authors list three limitations of their own.

**1. Two large runs.** Training at this scale is expensive, so there are only two comparable large-scale runs, Chinchilla and Gopher, and nothing at intermediate scales. The 400+ runs span 70M to 16B parameters, and the largest IsoFLOP budget is $3\times10^{21}$ FLOPs; the claim at $5.76\times10^{23}$ is an extrapolation checked by a single pair.

**2. A bent frontier.** All three approaches assume the frontier is a power law, a straight line on log-log axes. But the authors see some concavity in $\log N_{opt}$ at high compute. Appendix E shows it by fitting lines to the first, middle and last thirds of the frontier points separately.

::figure https://arxiv.org/html/2203.15556v1/curvature_v3.png | three straight-line fits to the training-curve envelope, through its first (orange), middle (green) and last (blue) thirds of points: the lines differ, so the frontier is not one straight line

::predict paper-limitations

Concave means the slope falls as compute grows, so a line fitted over the whole range overshoots at the top. The paper's conclusion: "we may still be overestimating the optimal size of large models." At a fixed budget a smaller model means more tokens, so if anything the 20 tokens per parameter is a floor at large scale, not a ceiling. (The same concavity is why Approach 3, which leans on the large runs, gives smaller models.)

**3. Less than one epoch.** All training runs in the analysis use less than one epoch of the corpus, so the paper says nothing about the regime where data runs out and must be repeated. Appendix C's dataset-independence claim carries the same restriction.

::note slip | The one-epoch statement is looser than it sounds. Table A1 shows that in Chinchilla's own 1.4T tokens two subsets of MassiveText are repeated: MassiveWeb for 1.24 epochs and Wikipedia for 3.40 (Books, C4, News and GitHub stay below one epoch). The corpus as a whole is not exhausted, but not every token is fresh. For the scaling runs (same mix, at most about 400B tokens) only Wikipedia comes near one epoch, by our arithmetic: one Wikipedia epoch is $1.4\times10^{12} \times 0.01 / 3.40 \approx 4.1$B tokens, and 1% of 400B is 4B.

The paper ends with what follows from the result. If models should grow no faster than their data, then dataset scaling deserves as much attention as model scaling, and the authors expect it to help only with high-quality data. Bigger datasets need more care about train-test overlap, and trillions of web tokens bring more toxic, biased and private content, so inspecting the data matters more. They expect a similar size-versus-data trade-off in other modalities, and stress that the methods are easy to reproduce.

One limitation the paper does not list is built into Eq. 1: the objective counts training compute only. A model that will serve billions of requests is cheaper over its life if it is smaller and trained longer than the compute-optimal size. [L9](#/read/lecture_09) shows how far practice moved past the rule for that reason (Llama 3 70B at about 215 tokens per parameter), and [L11](#/read/lecture_11) shows how scaling studies are run in practice today.

::note deferred | The multi-epoch regime the paper leaves open is taken up by later data-constrained scaling laws (repeating data for up to about 4 epochs is almost as good as new data), covered in [L9](#/read/lecture_09).
::kp paper-limitations
