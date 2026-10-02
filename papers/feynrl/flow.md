---
title: FeynRL · Trust the batch, on- or off-policy (P3O), read through
minutes: 30
---
In 2026 Fakoor, Aubry, Stranges and Smola asked why RL post-training of language models is so sensitive to one number chosen before training starts: the clip range of PPO and GRPO. Their answer is that the clip does two different jobs with one constant. They replace it with a statistic measured on every batch, the effective sample size of the importance ratios, and plug it into an objective called P3O. After this read-through you can compute importance ratios and the batch ESS by hand, say what PPO's clip and P3O's cap each do to a single token, follow the objective into the FeynRL code, and judge which of the paper's experimental claims the figures support.

## Why is RL post-training fragile? {#question}
source: feynrl:abstract · feynrl:§1 · feynrl:p2

RL post-training works the same way in every recent reasoning recipe ([L16](#/read/lecture_16)): sample several completions for a prompt, score them with a reward (here, whether the final math answer is right), and raise the probability of the completions that scored above average. Almost every recipe does the raising with PPO's clipped objective or its value-free descendant GRPO. Both carry a clip range $\epsilon$, fixed before training, and the results depend on it.

The paper's diagnosis starts by separating two things we want from an update.

- A **trust-region** concern: one gradient update should not move the policy too far from where it is now, *whatever* the data are. A large step on perfectly fresh data can still wreck a model.
- An **off-policy** concern: data produced by an older or different policy should influence the update only to the extent that it is still reliable. Stale data are dangerous even when the step is tiny.

In the paper's summary, the trust region controls the size of an update, while off-policy correction weights data by how reliably it can be used. The two concerns are independent axes: a fresh batch with a big learning rate triggers only the first; a stale batch with a tiny step triggers mostly the second.

::predict feynrl-two-concerns

PPO-style objectives handle both with one fixed clip range: it bounds how far a step may move each token's probability, and it also caps how much weight a stale token can get. The paper's premise is that fixed clipping itself, not the value of the range, causes the fragility, so iterating on the clip (symmetric, asymmetric, sequence-level) leaves the problem in place. Neither concern is a constant you can set in advance, the abstract says; how severe each is shows up in the policy ratios of the current batch.

So the proposal is to *measure* how on-policy the current batch is and let the measurement drive the update. The measurement is the effective sample size (ESS) of the per-token importance ratios. The objective that uses it is P3O ("Policy-on Policy-off Policy Optimization"), taken from an earlier paper by Fakoor, Chaudhari and Smola (2019) and applied here, the authors say, to large-model post-training for the first time. It introduces no new hyper-parameter and removes the clip range.

::note why | The objective is not new. The original P3O already set its off-policy cap to the ESS and its regularizer weight to one minus the ESS. What this paper contributes is the per-token formulation for language models, the argument against fixed clipping, an implementation in the FeynRL library, and experiments on 1.5B and 4B models.
::kp feynrl-two-concerns

## Where does off-policy data come from? {#sources}
source: feynrl:§1 · feynrl:§2 (From on-policy to off-policy) · feynrl:§4.3

"Off-policy" means the completions you train on were not sampled from the policy you are updating. Call the policy that actually produced them the **behavior policy** $\pi_b$ and the one being trained $\pi_\theta$. In textbook on-policy RL they are the same. In large-model post-training they almost never are, for three kinds of reasons the paper lists.

1. **Reuse.** Rollouts are expensive, so each batch of completions is used for several optimizer steps. After the first step, $\pi_\theta$ has moved and the batch is stale.
2. **Older or different policies.** Demonstrations, rollouts from a previous run (the batch-RL setting), or an asynchronous pipeline in which the rollout engines keep generating with last version's weights while the trainer updates.
3. **Engine mismatch.** Rollouts come from an inference engine (vLLM in FeynRL), training from a different stack. Even with identical weights the two can assign different probabilities to the same token: different numerical precision (training in BF16, generating in FP8), a different sampling strategy (temperature other than 1), or different kernels and architecture details.

The third is the easiest to miss: regenerating rollouts after every step does *not* make the ratios exactly one if the sampling engine computes probabilities differently.

::predict feynrl-off-policy-sources

Why not discard stale data? Because then each expensive rollout feeds only a few updates. The paper says current recipes commit to one regime before training, either fresh on-policy rollouts after every update or a fixed off-policy correction tuned for one level of staleness, and so force a choice between sample efficiency and stability.

The asynchronous case is concrete in FeynRL, the library behind the paper (its practice thread is [FeynRL](#/t/feynrl)). In sync mode the rollout GPUs and the training GPUs take turns. In overlap mode rollout keeps running while training runs, so the replay buffer can hold samples from several policy versions at once; a parameter `max_lag` bounds how many.

::animation fixture:rollout-train-timeline | in overlap mode (s1, s2) the buffer holds samples from more than one policy version when training reads it; that version gap is the staleness source, and it does not exist in sync mode (s0)

::kp feynrl-off-policy-sources

## Background: which objective is being fixed? {#background}
source: feynrl:§2 · Eq. 1 · Eq. 2 · Eq. 3 · Eq. 4

The setting is a pre-trained model fine-tuned so that its completions score well. The model generates a completion $y = (y_1, \dots, y_T)$ to a prompt $x$ one token at a time:

$$ \pi_\theta(y \mid x) = \prod_{t=1}^{T} \pi_\theta(y_t \mid x, y_{<t}) \qquad (1) $$

The paper abbreviates the conditioning context as $c_{<t} = (x, y_{<t})$. Generation is treated as a finite-horizon decision process: the state is the prompt plus the prefix, the action is the next token, and a single scalar reward $r(x, y)$ arrives at the end. The goal is to maximize $J(\pi_\theta) = \mathbb{E}[r(x,y)]$ with $y$ sampled from $\pi_\theta$. The likelihood-ratio trick gives the **REINFORCE** gradient, recapped in [L16](#/read/lecture_16):

$$ \nabla_\theta J(\pi_\theta) = \mathbb{E}_{x,\, y\sim\pi_\theta(\cdot\mid x)}\Big[\sum_{t=1}^{T} r(x,y)\,\nabla_\theta \log \pi_\theta(y_t \mid x, y_{<t})\Big] \qquad (2) $$

Two facts about Eq. 2 matter later. It is **on-policy**: each rollout serves the gradient step of the policy that produced it, then is thrown away. And it is noisy, so the reward is replaced by an **advantage** $A = r - b$, the reward minus a baseline that does not depend on the sampled token; the gradient stays unbiased and its variance drops.

**GRPO's baseline.** Learning a value network as baseline is expensive at this scale, since it is about as large as the policy. GRPO samples a group of $G$ completions per prompt $p$ and uses the group's own statistics:

$$ A_{p,j} = \frac{r_{p,j} - \frac{1}{G}\sum_{k=1}^{G} r_{p,k}}{\sqrt{\frac{1}{G}\sum_{k=1}^{G}\big(r_{p,k} - \frac{1}{G}\sum_{l=1}^{G} r_{p,l}\big)^2} + \epsilon} \qquad (3) $$

The group mean is a per-prompt baseline; the population standard deviation rescales advantages so they are comparable across prompts. For example, with $G = 4$ and rewards $(1, 0, 0, 0)$ the mean is 0.25 and the standard deviation is $\sqrt{0.1875} = 0.433$, so the correct completion gets $A = 1.73$ and each wrong one $-0.58$. Every token of a completion shares its completion's advantage.

**PPO's clip.** A single update can move $\pi_\theta$ too far even on on-policy data. PPO samples from a snapshot $\pi_{\theta_{old}}$ taken at the start of the optimizer epoch and clips the per-token ratio:

$$ \mathcal{L}_{\text{PPO}}(\theta) = -\mathbb{E}_{\pi_{\theta_{old}}}\big[\min\big(\rho_t(\theta)A,\ \operatorname{clip}(\rho_t(\theta), 1-\epsilon_\ell, 1+\epsilon_h)A\big)\big], \quad \rho_t(\theta) = \frac{\pi_\theta(y_t\mid x, y_{<t})}{\pi_{\theta_{old}}(y_t\mid x, y_{<t})} \qquad (4) $$

The range $[1-\epsilon_\ell, 1+\epsilon_h]$ is fixed before training. In PPO's intended setting $\pi_{\theta_{old}}$ is close to $\pi_\theta$, so $\rho_t$ stays near one for most tokens and the clip plays a clean trust-region role. The rest of the paper asks what happens to Eq. 4 when that assumption fails.

::note aside | [L16](#/read/lecture_16) teaches the clip as the answer to "a batch cannot move the policy far" and never raises the off-policy concern; importance sampling, ESS and P3O do not appear in the course. This paper starts where the lecture stops.
::kp lecture_16:ppo-ratio-clip
::kp lecture_16:grpo-group-zscore-advantage

## Importance sampling: how do you learn from someone else's samples? {#importance}
source: feynrl:§2 (From on-policy to off-policy) · Eq. 5 · Eq. 6 · Eq. 7 · Eq. 8 · feynrl:p4

When the data come from $\pi_b$ but the loss is evaluated under $\pi_\theta$, the empirical gradient on those samples is a biased estimate of $\nabla_\theta J(\pi_\theta)$. The standard fix is **importance sampling**: change the measure, and pay for it with a weight.

$$ \mathbb{E}_{x\sim\pi_\theta}[f(x)] = \int \pi_\theta(x) f(x)\,dx = \int \pi_b(x)\frac{\pi_\theta(x)}{\pi_b(x)} f(x)\,dx = \mathbb{E}_{x\sim\pi_b}\Big[\frac{\pi_\theta(x)}{\pi_b(x)} f(x)\Big] \qquad (5) $$

The middle step multiplies and divides by $\pi_b(x)$, which needs $\pi_b(x) > 0$ wherever $\pi_\theta(x) > 0$ (the paper's "provided that $d\pi_\theta/d\pi_b$ is well defined"). A sample that $\pi_b$ produced rarely but $\pi_\theta$ likes counts more; one that $\pi_b$ over-produced counts less. Applied to the whole objective:

$$ J(\pi_\theta) = \mathbb{E}_{x,\,y\sim\pi_\theta(\cdot\mid x)}[r(x,y)] = \mathbb{E}_{x,\,y\sim\pi_b(\cdot\mid x)}\Big[\frac{\pi_\theta(y\mid x)}{\pi_b(y\mid x)}\, r(x,y)\Big] \qquad (6) $$

The weight in Eq. 6 is a ratio of whole-sequence probabilities, a product of $T$ per-token ratios. Over thousands of tokens such a product explodes or vanishes. So the paper, like PPO, works with the **per-token ratio** and applies it inside the per-token gradient:

$$ \rho_t(\theta) = \frac{\pi_\theta(y_t \mid x, y_{<t})}{\pi_b(y_t \mid x, y_{<t})} \qquad (7) $$

This is PPO's ratio with the snapshot $\pi_{\theta_{old}}$ replaced by whatever policy generated the data. In code it is computed in log space, from the log-probabilities the rollout engine stored and the ones the trainer just computed:

::code algs/P3O/p3o.py:L192-L196 | the ratio is exp of a log-probability difference, in float32, with padded positions set to log-ratio 0 before the exponential

::note warning | Eq. 7 is not just Eq. 6 written per token. The per-token weight is a different estimator, lower in variance but biased for the sequence-level objective. The paper switches to it without saying so; every PPO-family method makes the same switch.

**The catch.** The paper's remark is that importance sampling is unbiased but high-variance. A ratio is bounded below by 0 and unbounded above. A token that $\pi_b$ gave probability $10^{-4}$ and $\pi_\theta$ now gives $0.1$ gets weight 1,000. A few such tokens can dominate the whole gradient.

The problem is one-sided. Take four tokens with log-ratios $(\ln 4, 0, 0, 0)$: the ratios are $(4, 1, 1, 1)$ and the first token carries $4/7 = 57\%$ of the total weight. Flip the sign of its log-ratio and the ratios become $(0.25, 1, 1, 1)$: it now carries $0.25/3.25 = 7.7\%$. Being disliked by $\pi_\theta$ can only shrink a token's weight toward zero; being under-sampled by $\pi_b$ can inflate it without limit.

::predict feynrl-importance-ratio
::widget fixture:feynrl_paper--batch-ess | type the batch 4, 1, 1, 1 and read the share column: the ratio-4 token carries 57% of the weight; tick the flip box and the same token, now at 1/4, carries 7.7%

**A second KL, not to be confused with the first.** The section closes with the language-model objective used in practice. Fine-tuning is regularized toward a frozen reference policy $\pi_{\theta_0}$, usually the model you started from:

$$ J_{\text{LM}}(\pi_\theta) = \mathbb{E}_{x,\,y\sim\pi_\theta(\cdot\mid x)}\Big[r(x,y) - \eta\sum_{t=1}^{T}\mathrm{KL}\big(\pi_\theta(\cdot\mid x, y_{<t})\,\|\,\pi_{\theta_0}(\cdot\mid x, y_{<t})\big)\Big] \qquad (8) $$

The coefficient $\eta > 0$ controls how far the model may drift from the reference, to keep its language-modeling abilities (the per-token KL penalty of [L16](#/read/lecture_16)). Keep this anchor apart from what comes next: P3O adds a *second* KL whose anchor is the behavior policy that produced the batch, not the reference model.
::kp feynrl-importance-ratio

## What does a fixed clip do to each token? {#clip}
source: feynrl:§3 · Eq. 9 · Eq. 10 · feynrl:p5

Section 3 opens by taking Eq. 4 apart by the sign of the advantage. For a token with $A > 0$ the min picks

$$ -\min\Big(\frac{\pi(y_t\mid c_{<t})}{\pi_{old}(y_t\mid c_{<t})},\ 1+\epsilon_h\Big)A \qquad (A > 0) \qquad (9) $$

Raising the token's probability raises $\rho_t$ and lowers the loss, until $\rho_t$ passes $1 + \epsilon_h$. From there the term is the constant $-(1+\epsilon_h)A$, and the policy gains no more credit for raising that token further. For $A < 0$ the min becomes a max:

$$ -\max\Big(\frac{\pi(y_t\mid c_{<t})}{\pi_{old}(y_t\mid c_{<t})},\ 1-\epsilon_\ell\Big)A \qquad (A < 0) \qquad (10) $$

Now the loss falls as the probability falls, until $\rho_t$ drops below $1 - \epsilon_\ell$, where it goes flat.

The paper does not write out the consequence, so here it is. Where the term is constant, its gradient is **exactly zero**. The token is not down-weighted; it drops out of the update. (The loss *value* is not zero there, only its gradient.) And only one edge applies per token: a positive-advantage token is cut off above $1+\epsilon_h$, a negative-advantage token below $1-\epsilon_\ell$. A token with $\rho_t = 0.3$ is live if $A > 0$ and dead if $A < 0$.

::predict feynrl-fixed-clip-regimes
::widget fixture:feynrl_paper--token-weight | push ρ past the edge that A's sign guards (above 1+ε when A > 0, below 1−ε when A < 0): GRPO's weight drops to exactly 0, the thick orange segment; on the other side it keeps growing with ρ

In FeynRL's GRPO this is three lines: the unclipped term, the clamped term, and the elementwise minimum.

::code algs/GRPO/grpo.py:L169-L176 | torch.minimum of ratio·adv and clamp(ratio)·adv is Eq. 4; where the clamped branch wins, nothing in it depends on θ

**Why no single value works.** The effect of $(\epsilon_\ell, \epsilon_h)$ depends on the regime. On a fresh, on-policy batch $\rho_t \approx 1$ for most tokens and the clip is rarely active. On a stale or foreign batch many ratios are far from one, and the clip range now decides what *fraction of the batch* contributes at all. Too small, and many tokens are clipped before they contribute. Too large, and high-variance importance-weighted gradients destabilize training. The paper's conclusion: no single value suits both regimes, so in practice the range has to be tuned per task, model scale and degree of mismatch.

::note why | Watch what a fix for one regime does to the other. Suppose stale batches lose most tokens at $\epsilon = 0.2$ and you widen the clip to 0.6. On fresh batches, where the off-policy side barely acts, you have just loosened the trust region: a token's ratio can now reach 1.6 before its gradient stops. One knob, two jobs.

**Prior work re-parameterizes the clip without removing it.** GRPO uses Eq. 4 with $\epsilon_\ell = \epsilon_h$. DAPO lets them differ, to treat positive and negative advantages separately. GSPO moves the clip from the token to the sequence, clipping a geometric mean of the token ratios, but then needs sequence-level ranges $\epsilon^{seq}_\ell, \epsilon^{seq}_h$, still chosen in advance. In every variant, some $\epsilon$ survives.
::kp feynrl-fixed-clip-regimes

## Effective sample size: how on-policy is this batch? {#ess}
source: feynrl:§3 (Effective Sample Size to the Rescue) · Eq. 11 · feynrl:p5

If the clip range is the problem, something must take its place, and it should read the batch rather than be set before training. The paper's choice is the **effective sample size** of the policy ratios in the current batch:

$$ \mathrm{ESS}(\mathcal{B};\theta) = \frac{\Big(\widehat{\mathbb{E}}_{\mathcal{B}}\Big[\frac{\pi(y_t\mid c_{<t})}{\pi_{old}(y_t\mid c_{<t})}\Big]\Big)^2}{\widehat{\mathbb{E}}_{\mathcal{B}}\Big[\Big(\frac{\pi(y_t\mid c_{<t})}{\pi_{old}(y_t\mid c_{<t})}\Big)^2\Big]} = \frac{\widehat{\mathbb{E}}_{\mathcal{B}}[\rho_t]^2}{\widehat{\mathbb{E}}_{\mathcal{B}}[\rho_t^2]} \in \Big[\frac{1}{|\mathcal{B}|}, 1\Big] \qquad (11) $$

Here $\mathcal{B}$ is the set of valid response tokens in the training batch (padding and prompt tokens excluded), $|\mathcal{B}|$ its size, and $\widehat{\mathbb{E}}_{\mathcal{B}}$ the plain average over them. With weights $w_i = \rho_i$ and $n = |\mathcal{B}|$ it is $(\sum w)^2 / (n \sum w^2)$.

The name comes from classical importance sampling: $(\sum w)^2/\sum w^2$ estimates how many *unweighted* samples would be as informative as $n$ weighted ones, and dividing by $n$ makes it a fraction. Equal weights give 1; one weight carrying everything gives $1/n$.

Work a toy batch, ratios $(4, 1, 1, 1)$: $\sum w = 7$, $\sum w^2 = 19$, so $\mathrm{ESS} = 49/(4 \times 19) = 49/76 = 0.645$. One token with four times the weight of the others drags the batch from "4 effective tokens" to about 2.6.

::predict feynrl-normalized-ess
::widget fixture:feynrl_paper--batch-ess | raise the spread s, which makes the ratios unequal, and ESS falls toward 1/n; then move the common shift δ: every ratio, Σw and Σw² change, but ESS does not move

**Why the bounds hold.** The paper states $[1/|\mathcal{B}|, 1]$ without proof. The upper bound is Cauchy-Schwarz applied to $w$ and the all-ones vector: $(\sum w)^2 \le n \sum w^2$, with equality exactly when all $w_i$ are equal. The lower bound: for non-negative weights $(\sum w)^2 \ge \sum w^2$, since the expansion only adds non-negative cross terms, with equality when a single weight is non-zero. So $(1, 0, 0, 0)$ gives $1/4$. More generally, $k$ equal non-zero weights out of $n$ give exactly $k/n$, whatever their size.

::predict supp-ess-bounds

That last remark is the property to remember. Multiply every weight by the same constant $c$ and both numerator and denominator gain a factor $c^2$: **ESS is scale-free**. $(8, 2, 2, 2)$ has the same ESS as $(4, 1, 1, 1)$, 0.645, and $(0.1, 0.1, 0.1, 0.1)$ has ESS 1. ESS measures how *concentrated* the weight is across tokens, not how far the ratios are from one. Two consequences:

- ESS = 1 does not mean the batch is on-policy; it means the ratios are all equal. A batch with every ratio 0.5 has ESS 1.
- A mismatch that multiplied every ratio by the same factor would be invisible to ESS. This matters in Section 4.

Why this gauges on-policyness: under $\pi_b$ the expected ratio is exactly 1 (Eq. 5 with $f = 1$), so ESS is roughly $1/\widehat{\mathbb{E}}[\rho^2]$, the second moment that controls importance-sampling variance.

**Detached.** P3O uses the value $e_{\mathcal{B}} = \operatorname{sg}(\mathrm{ESS}(\mathcal{B};\theta))$, where $\operatorname{sg}$ is the stop-gradient: for backpropagation it is a constant. In the paper's words, the ESS is close to one when the batch is on-policy and falls when a few large ratios dominate, as with stale or mismatched data.

::code algs/P3O/p3o.py:L107-L135 | three sums over valid tokens (Σw, Σw², count), all-reduced across data-parallel ranks, then (Σw)² / (Σw² + 1e-8) / n; returned as a Python float, which is what detaches it

Two details the paper does not mention: $\mathcal{B}$ is one micro-batch's valid tokens pooled across GPUs, so $e_{\mathcal{B}}$ is recomputed every micro-batch step; and with no valid tokens or a non-finite sum the function returns 1.0 (no cap, no extra KL).

::note slip | Eq. 11 writes the ratio as $\pi/\pi_{old}$, Eq. 7 as $\pi_\theta/\pi_b$, and Table 1 uses $\pi_b$ throughout. The paper treats $\pi_{old}$ and $\pi_b$ as the same thing in Section 3: the policy that produced the batch, whose log-probabilities were stored with it. That is also what the code uses (`old_logprobs`).
::kp feynrl-normalized-ess
::kp supp-ess-bounds

## The P3O objective: one statistic, two controls {#p3o}
source: feynrl:§3 (Policy-on Policy-off Policy Optimization) · Eq. 12 · feynrl:p5-p6

P3O replaces the fixed clip of Eq. 4 with two terms, both set by the batch ESS:

$$ \mathcal{L}_{\text{P3O}}(\theta) = \mathbb{E}_{x,\,y\sim\pi_{\theta_{old}}(\cdot\mid x)}\Big[\sum_{t=1}^{T}\Big(-\operatorname{sg}(\min\{\rho_t, e_{\mathcal{B}}\})\log\pi_\theta(y_t\mid c_{<t})\,A + (1-e_{\mathcal{B}})\,\mathrm{KL}\big(\pi_\theta(\cdot\mid c_{<t})\,\|\,\pi_{\theta_{old}}(\cdot\mid c_{<t})\big)\Big)\Big] \qquad (12) $$

**Term 1, a capped policy gradient.** Each token's $\log\pi_\theta$ is weighted by its advantage times a detached coefficient, its importance ratio capped at $e_{\mathcal{B}}$. Without the cap this is importance-weighted REINFORCE. The cap is one-sided and absolute: ratios above $e_{\mathcal{B}}$ are cut to $e_{\mathcal{B}}$, ratios below pass through unchanged. There is no band around 1 and no lower edge.

**Term 2, a pull toward the behavior policy.** A per-token KL between the current policy and the policy that produced the batch, weighted by $1 - e_{\mathcal{B}}$. It vanishes on a fresh batch ($e_{\mathcal{B}} \approx 1$) and grows as the batch drifts. The anchor is $\pi_{\theta_{old}}$, not the reference model of Eq. 8.

Back to the toy batch, $e_{\mathcal{B}} = 0.645$ and $A = +1$. Tokens with $\rho_t = 4$, $1.5$ and $1.0$ all get weight 0.645; a token with $\rho_t = 0.3$ keeps 0.3. The KL weight is $1 - 0.645 = 0.355$.

::predict feynrl-p3o-objective

Notice the token with $\rho_t = 1.0$. It is exactly on-policy, yet its weight is cut to 0.645, because the cap is the batch's ESS, which is below 1 whenever the ratios are unequal. With ratios $(3, 1, 1, 1)$, $e_{\mathcal{B}} = 36/48 = 0.75$, and every ratio-1 token gets 0.75. The paper says an ESS-driven objective "behaves like an on-policy update on fresh data"; that holds only approximately. On a nearly uniform batch $(1.1, 1.0, 0.9, 1.0)$: $\sum w = 4.0$, $\sum w^2 = 4.02$, $e_{\mathcal{B}} = 16/16.08 = 0.995$, so the cap barely binds and the KL weight is 0.005.

::widget fixture:feynrl_paper--token-weight | P3O's green line is flat at e_B for every ρ above e_B, ρ = 1 included, and never reaches 0; tick the KL box to add the (1 − e_B) pull, which pushes a token with ρ > 1 back down and vanishes at e_B = 1

**Why the detached weight still gives a policy gradient.** The paper moves from Eq. 4's ratio form to Eq. 12's $\operatorname{sg}(\cdot)\log\pi$ form without the one-line identity that makes them comparable:

$$ \nabla_\theta \rho_t = \rho_t\,\nabla_\theta\log\pi_\theta(y_t\mid c_{<t}) $$

So the gradient of the ratio surrogate $-\rho_t A$ is $-\rho_t A\,\nabla\log\pi_\theta$, and the gradient of $-\operatorname{sg}(w)\log\pi_\theta A$ is $-w A\,\nabla\log\pi_\theta$. With $w = \rho_t$ they are identical. Detaching changes nothing about the direction; it only stops gradient flowing through the weight itself. What differs between PPO-style objectives and P3O is the *value* of the weight, and whether there is a flat region where it is effectively zero. For one token with $\rho_t = 1.5$, $A = +1$ and $g = \nabla\log\pi_\theta$: the ratio surrogate gives $-1.5\,g$; P3O with $e_{\mathcal{B}} = 0.645$ gives $-0.645\,g$, a shorter step in the same direction; PPO with $\epsilon = 0.2$ gives 0, because 1.5 is above 1.2.

::predict supp-score-function-weighting

This is the paper's central contrast: the clip in Eq. 4 discards the gradient of every token outside the band, while the cap in Eq. 12 only scales it by a positive factor. Every token contributes, and Eq. 12 has no clipping range, no trust-region coefficient and no staleness budget.

**What the KL does to one token.** FeynRL estimates the KL at the sampled token with the non-negative estimator $\log(\pi/\pi_{old}) + \pi_{old}/\pi - 1$, whose derivative with respect to $\log\pi$ is $1 - 1/\rho_t$. For the $\rho_t = 4$ token: the policy term contributes $-0.645\,g$ to the loss gradient, the KL term $0.355 \times (1 - 1/4)\,g = 0.266\,g$. Net $-0.379\,g$: the token is still pushed up, but less. At $\rho_t = 1$ the KL gradient is zero.

::code algs/P3O/p3o.py:L204-L208 | rho = clamp(ratio, 0, ess_factor) is min{ρ_t, e_B}, and .detach() is the sg
::code algs/P3O/p3o.py:L223 | the total loss: policy term, optional entropy bonus and reference KL (fixed coefficients), and (1 − ess_factor) times the behavioral KL

::note warning | Two gaps between Eq. 12 and the code. The paper writes an exact per-token KL "in the same form as Eq. (8)"; the code uses the single-sample estimator above, a Monte-Carlo estimate rather than a sum over the vocabulary. And Eq. 12 has no entropy or reference-KL term; the code adds both as optional terms with their own coefficients, which Table 1 also says it omits.
::kp feynrl-p3o-objective
::kp supp-score-function-weighting

## Table 1: what does each objective fix before training? {#family}
source: feynrl:§3 · feynrl:table1 · feynrl:p6

Table 1 colours each quantity by who chooses it: red and green for hyper-parameters fixed before training, purple for auxiliary choices, blue for batch statistics. Entropy and reference-KL terms are omitted.

::figure feynrl:table1 | count the coloured symbols per row: GRPO/DAPO carry ε_ℓ and ε_h; GSPO carries ε^seq; the Decoupled row carries ε_ℓ, ε_h, a cap c_w and the purple π_prox; P3O's only coloured symbol is the blue e_B

| Method | Policy objective (per token) | Fixed before training |
|---|---|---|
| GRPO / DAPO | $-\min(\rho_t A,\ \operatorname{clip}(\rho_t, 1-\epsilon_\ell, 1+\epsilon_h)A)$ | $\epsilon_\ell, \epsilon_h$ (one value if symmetric) |
| GSPO | $-\min(S_\theta A,\ \operatorname{clip}(S_\theta, 1-\epsilon^{seq}, 1+\epsilon^{seq})A)$, $S_\theta(y) = \exp\big(\tfrac{1}{T}\sum_t \log\rho_t\big)$ | sequence-level $\epsilon^{seq}$ |
| Decoupled | $-\operatorname{sg}\big(\operatorname{clip}(\tfrac{\pi_{prox}}{\pi_b}, 0, c_w)\big)\min\big(\tfrac{\pi_\theta}{\pi_{prox}}A,\ \operatorname{clip}(\tfrac{\pi_\theta}{\pi_{prox}}, 1-\epsilon_\ell, 1+\epsilon_h)A\big)$ | $\epsilon_\ell, \epsilon_h, c_w$, plus how $\pi_{prox}$ is built |
| P3O | $-\operatorname{sg}(\min\{\rho_t, e_{\mathcal{B}}\})\log\pi_\theta A + (1-e_{\mathcal{B}})\mathrm{KL}(\pi_\theta \Vert \pi_b)$ | nothing |

(The last column is ours, read off the table's colours.) GSPO's $S_\theta$ is the geometric mean of the token ratios, so one number per completion is clipped. The Decoupled row is explained in the next section.

::predict feynrl-p3o-vs-clip-family

The table leaves out one relative that FeynRL implements and that is closer to P3O than any row: **CISPO**. Its loss has P3O's shape exactly, a detached clipped ratio times $\log\pi_\theta$ times $A$, so it too keeps a gradient on every token. But its clip is a fixed band $[1-\epsilon_\ell, 1+\epsilon_h]$, and it has no behavior KL.

::code algs/CISPO/cispo.py:L174-L175 | clamp(ratio, 1 − clip_low, 1 + clip_high).detach() · log π · A: the same line as P3O's, with a band fixed before training where P3O has [0, e_B]

Put one token through all three to see the difference. Take $\rho_t = 1.3$, $A = +1$, $\epsilon = 0.2$ for GRPO and CISPO, and $e_{\mathcal{B}} = 0.9$ for P3O. GRPO: 1.3 is above 1.2 with $A > 0$, so the clamped branch wins and the weight on $\nabla\log\pi$ is 0. CISPO: weight $\operatorname{clamp}(1.3, 0.8, 1.2) = 1.2$. P3O: weight $\min(1.3, 0.9) = 0.9$. Now a token the policy already moved away from, $\rho_t = 0.5$ with $A = -1$ and $e_{\mathcal{B}} = 0.645$: GRPO drops it (0.5 is below 0.8 with $A < 0$); CISPO lifts its weight to the band edge, 0.8; P3O leaves it at 0.5, the actual ratio, because its cap is one-sided.

::widget fixture:feynrl_paper--token-weight | compare the three weight curves at one ρ: GRPO has a zero segment, CISPO is clamped to a band fixed before training, P3O is capped at e_B; only P3O's cap moves when you change the batch statistic

::note aside | In FeynRL, P3O still accepts `clip_low` and `clip_high`. The loss never reads them; they only define a `clipfrac` metric, which reports how many tokens a PPO range would have clipped.
::kp feynrl-p3o-vs-clip-family

## Off-policy training at scale: one loss for sync and async {#async}
source: feynrl:§3 (Remark; Off-policy data in large-model training) · feynrl:p6

At large-model scale the optimizer rarely sees strictly on-policy data: engines differ, and reusing rollouts across optimizer epochs widens the gap. The standard fix is a **decoupled loss** (the Decoupled row of Table 1). It separates two policies that PPO merges into $\pi_{old}$:

- a **proximal** policy $\pi_{prox}$, a snapshot of the current weights taken at the start of each epoch, which the clip is measured against (the trust region);
- the behavior policy $\pi_b$ that actually generated the data, which an outer importance weight $\pi_{prox}/\pi_b$ corrects for, capped at $c_w$ (the off-policy correction).

This works, but it adds hyper-parameters: how $\pi_{prox}$ is built, how long it lives, and the cap $c_w$, all to be tuned per off-policy regime. P3O, the paper says, sidesteps this: the same $e_{\mathcal{B}}$ that sets the cap on fresh data falls when the batch becomes off-policy, so Eq. 12 covers both regimes with no extra hyper-parameter or code path.

This is the point where the paper meets the code most directly ([FeynRL practice thread](#/t/feynrl)). FeynRL switches every algorithm to the decoupled loss automatically when overlap mode is on:

::code core/rl_engines.py:L56-L60 | use_decoupled_loss is simply overlap.enabled, passed to every algorithm, with the behavior-weight cap c_w beside it
::code algs/P3O/p3o.py:L67-L69 | P3O stores both arguments under the comment that they are not used in P3O "for now", and its loss never reads them

So switching a run from sync to overlap mode changes GRPO's loss (it now clips $\pi/\pi_{prox}$, weights by $\pi_{prox}/\pi_{old}$, and runs an extra no-gradient forward pass per epoch to take the snapshot) and leaves P3O's loss untouched; only $e_{\mathcal{B}}$ will come out lower when the buffer mixes versions.

::predict feynrl-same-loss-sync-async
::animation fixture:feynrl_paper--overlap-loss | from s0 to s1 (overlap on) a purple π_prox snapshot block appears before every GRPO training block while the P3O lane gains nothing; at s2 (max_lag = 2) the buffer holds three versions and both lanes repeat their pattern unchanged

"No staleness budget" is true of the *objective* only. FeynRL's asynchronous runner still has `max_lag`, which sizes the replay buffer and evicts samples more than `max_lag` versions behind. P3O makes the loss safe for whatever mix arrives; very stale data simply get a low $e_{\mathcal{B}}$ and contribute little.

::animation fixture:feynrl--overlap-rounds | from round 2 the buffer holds shards older than the trainer, and at s7 evict_stale drops exactly the shards stamped below T − max_lag; that bound lives in the scheduler, whichever loss is used

::note warning | The paper's off-policy evidence (Section 4) comes from temperature and precision mismatches, not from replay staleness. Whether any run used overlap mode is not reported, so the async claim rests on the argument and the code, not on a plotted experiment.
::kp feynrl-same-loss-sync-async

## The paper's own caveat: one ESS, two drifts {#two-anchor}
source: feynrl:§3.1 · Eq. 13 · feynrl:§B · feynrl:§D

Section 3.1 raises a weakness of P3O itself. Within one epoch there are two different drifts in a batch:

- between $\pi_\theta$ and the policy that generated the data, $\pi_b$ (staleness, engine mismatch);
- between $\pi_\theta$ and its own snapshot at the start of the epoch, $\pi_{prox}$ (how far this epoch's optimizer steps have already moved it).

A single $e_{\mathcal{B}}$ mixes the two. When both are large but in different directions, one anchor cannot tell them apart. The natural extension adds a second anchor at $\pi_{prox}$ and pulls $\pi_\theta$ toward a mixture of the two:

$$ \mathcal{L}_{\text{ext}}(\theta) = -\mathbb{E}_{\pi_b}\big[\operatorname{sg}(\min\{r_b, e_{\text{mix}}\})\log\pi_\theta(y_t\mid c_{<t})\,A\big] + (1-e_{\text{mix}})\,\mathrm{KL}(\pi_\theta\,\|\,\pi_{\text{mix}}) \qquad (13) $$

Here $r_b = \pi_\theta/\pi_b$ is the per-token behavior ratio, $e_{\text{mix}}$ a joint ESS computed from both anchors, and $\pi_{\text{mix}}$ a per-token mixture of $\pi_b$ and $\pi_{prox}$ with weights $(1 - e_b)$ and $(1 - e_{prox})$, each anchor's own ESS. The full formulation is in Appendix B.

The weights are the clever part. An anchor the policy has *not* drifted from has ESS near 1, hence weight near 0, and drops out of the mixture; the anchor with the larger drift dominates. Say the batch came from an old behavior policy ($e_b = 0.5$) but the optimizer has not stepped yet this epoch ($e_{prox} \approx 1$): the weights are 0.5 and about 0, $\pi_{\text{mix}}$ collapses onto $\pi_b$, and Eq. 13 is single-anchor P3O around the behavior policy. That is what the paper means by "reduces to P3O when either anchor is uninformative". The extension adds no hyper-parameter either: both mixture weights are batch statistics.

::predict feynrl-two-anchor-issue

Then the surprise: empirically the single-anchor P3O matches or slightly outperforms the extension across the regimes of Section 4. The authors read this as evidence that the behavior-axis ESS already captures most of the drift that matters. So the extension is rejected on results, not on complexity.

::note aside | The appendix figures for this comparison (Figs. 8 and 9) are not in the page images used here; the following comes from the arXiv HTML version. At temperature 1.2 the two-anchor variant reportedly peaks near 0.67 average reward and then collapses after about step 16 while P3O stays stable; at the default temperature it is stable and slightly ahead. The paper offers no mechanism for the collapse.
::kp feynrl-two-anchor-issue

## Experiments I: setup and the clip sweep {#clip-sweep}
source: feynrl:§4 · feynrl:§4.1 · feynrl:§4.2 · Fig. 1 · feynrl:table3

Section 4 tests three things: (i) sensitivity to the clip range that P3O removes, (ii) robustness to off-policy data from precision and temperature mismatches, and (iii) performance on held-out math benchmarks. The only baseline is GRPO, with all other hyper-parameters identical within each experiment family.

**Setup.** Two open models, Qwen3-4B-Thinking-2507 and Qwen2.5-1.5B, trained on DeepScaleR-Preview: 40,000 math problem-answer pairs collected from AIME (1984-2023), AMC (before 2023), Omni-MATH and Still. The reward is binary, 1 if the final answer matches the reference, with no shaping or bonuses, so that only the objective differs. All runs use 8 NVIDIA H100 GPUs, with optimizer workers separated from the rollout engines and weights synchronized under a shared scheduler; that is FeynRL's architecture.

::note aside | Appendix Table 3 (read from the arXiv HTML, not checked against a page image) gives the per-experiment configuration. Clip runs: Qwen3-4B, 6 training and 2 rollout GPUs, 512 rollout samples per epoch, 1,024 maximum tokens, temperature 1.0. Temperature runs: 4,096 maximum tokens, GRPO clip 0.4. FP8 runs: 5 training and 3 rollout GPUs, 256 samples, 16,384 maximum tokens, GRPO clip 0.4. Optimizer AdamW, learning rate $10^{-5}$.

**The clip sweep.** GRPO is run with symmetric $\epsilon \in \{0.2, 0.4, 0.6\}$ and compared with a single P3O run.

::figure feynrl:fig1 | average training reward against steps; blue is GRPO's mean over the three clip values with a shaded ±1 std band across clip values, red is the single P3O run. (a) Qwen3-4B: both climb from about 0.07 to about 0.5 by step 20; after step 25 the GRPO mean sags to about 0.37 and its band widens, while P3O stays near 0.5. (b) Qwen2.5-1.5B: both wander between about 0.2 and 0.28, overlapping throughout

The paper reads Fig. 1 as GRPO's trajectory varying substantially with $\epsilon$ while P3O stays stable. It adds, without a plot, that DAPO and GSPO, which also keep a fixed range, show the same sensitivity, so the variable isolated is the presence or absence of a fixed clip.

::note warning | Read the shaded band carefully. It is the spread *over the three clip values*, not over random seeds; P3O was "run once". No seed count is reported anywhere, so neither method's run-to-run variance is measured. The caption also claims P3O matches or exceeds "the best GRPO variant", but the figure shows only the mean and band, not the individual variants. And panel (b) shows no separation at all.
::kp feynrl-experiments

## Experiments II: temperature and FP8 mismatch {#mismatch}
source: feynrl:§4.3 · Fig. 2 · Fig. 3 · feynrl:p7-p8

Section 4.3 creates off-policy data with no reuse or staleness at all, purely through the engine, and asks whether the batch ESS handles it "organically".

**Temperature.** Rollouts are sampled at temperature $T \ne 1$ while the trainer scores tokens at $T = 1$. The paper's account: a temperature rescales token log-probabilities uniformly, shifting every $\rho_t$ by a constant factor across the batch. For GRPO that offset lands inside or outside the fixed band regardless of how on-policy the batch otherwise is, a bias you cannot correct without retuning $\epsilon$. The ESS, it says, measures the shift as ratio concentration and tightens the cap and regularizer accordingly.

Those two sentences cannot both be right. If the shift really were a constant factor on every ratio, the ESS would not move at all: Eq. 11 is scale-free (the [ESS section](#ess)). What actually happens is not a constant factor. Sampling at temperature $T$ uses $\pi_T(y) = \pi(y)^{1/T}/Z_T$, so the ratio $\pi/\pi_T = \pi^{1 - 1/T} Z_T$ depends on each token's own probability (and $Z_T$ on its context). Take a two-token choice with trainer probabilities 0.9 and 0.1.

| Rollout temperature | Rollout probabilities | Ratio $\rho$, common token | Ratio $\rho$, rare token |
|---|---|---|---|
| 0.6 | 0.975 / 0.025 | 0.92 | 3.99 |
| 1.2 | 0.862 / 0.138 | 1.04 | 0.72 |

At $T = 0.6$ the rare token is under-sampled, so when it does appear its ratio is about 4: the upper tail. A batch of 40 tokens in the rollout proportions, 39 common and 1 rare, has ESS 0.81. At $T = 1.2$ the ratios straddle 1 more mildly; 6 common and 1 rare give ESS 0.99. ESS reacts to the uneven part of the change, not to a common scale.

Now the curves, for Qwen3-4B with GRPO at clip 0.4 (Table 3).

::predict feynrl-experiments

::figure feynrl:fig2 | average training reward. (b) T = 1.2: both rise to about 0.7 by step 10; after step 20 GRPO (blue) falls to about 0.2 and to about 0.1 by step 32, while P3O (red) holds near 0.6-0.65 to step 36. (a) T = 0.6: both peak near step 13 (GRPO about 0.67, P3O about 0.62), and *both* fall to about zero by step 20

The text says GRPO degrades at both temperatures "relative to the on-policy baseline" while P3O matches or exceeds standard-temperature performance, and that the same holds for Qwen2.5-1.5B (Fig. 7, appendix, not checked here). Panel (b) supports that. Panel (a) does not.

::note slip | Fig. 2(a) shows P3O collapsing at $T = 0.6$ just as GRPO does, from about 0.62 to about zero by step 20. The caption's "P3O is robust to off-policy data introduced through the varied sampling temperature" and the prose's "P3O matches or exceeds standard-temperature performance" hold for $T = 1.2$ only. No $T = 1.0$ curve is plotted in Fig. 2, so the "on-policy baseline" it is compared against is not on the page.

::widget fixture:feynrl_paper--batch-ess | press "fresh, both signs" and "GRPO clip 0.4", then drag the common shift δ: once e^δ leaves [0.6, 1.4], GRPO's share of tokens with gradient halves (the sign that edge guards stops), P3O keeps every token, and e_B stays exactly 1; only the spread slider s, the uneven part, moves ESS

That widget shows the mechanism the paper offers for Fig. 2(b), not a simulation of training. A clip of 0.4 means the band $[0.6, 1.4]$, which in log space is $[-0.51, 0.34]$: a log-ratio shift of just 0.34 on every token already pushes all positive-advantage tokens out of it.

**BF16 training, FP8 rollouts.** Mixed-precision pipelines generate with an FP8-quantized copy of the policy because it is much faster, and train in BF16. The quantized model assigns slightly different token probabilities, so $\rho_t$ moves away from one with no staleness at all. This is known to cause reward collapse at long generation lengths, so the experiment uses 16,384-token rollouts. The question is whether P3O's adaptive regularizer handles the mismatch without changing the training configuration.

::figure feynrl:fig3 | (a) Qwen3-4B: both climb from about 0.3 to about 0.5; GRPO peaks near 0.55 around step 19, then falls to about 0.03 by step 30, while P3O stays around 0.5 to step 35. (b) Qwen2.5-1.5B: the red and blue curves interleave between about 0.18 and 0.28, with no separation

The paper's reading: P3O stays stable when FP8 quantization pushes importance ratios away from one, while GRPO collapses later in training under the same mismatch. That is clear in panel (a). Panel (b), the smaller model, shows no difference, and the text does not comment on it.
::kp feynrl-experiments
::kp feynrl-normalized-ess

## Experiments III: held-out benchmarks {#benchmarks}
source: feynrl:§4.4 · Fig. 4 · feynrl:table7

To check that the training-reward differences carry over, checkpoints of Qwen3-4B-Thinking-2507 trained with each method are evaluated on five held-out math benchmarks: AIME24, AIME25, AIME26, AMO-Bench and AMC. The paper notes that DeepScaleR-Preview contains no samples from them. The metric is **pass@k**: the fraction of problems solved by at least one of $k$ sampled answers.

::figure feynrl:fig4 | pass@k for k = 1 to 16, averaged over the five benchmarks. (a) clip study at 4K-token evaluation: P3O (red) from about 0.21 to 0.42, GRPO averaged over ε ∈ {0.2, 0.4, 0.6} (blue) from about 0.16 to 0.33, the curve labelled Baseline (black) from about 0.06 to 0.19. (b) FP8 runs at 16K-token evaluation, dashed iter 15 and solid iter 30: P3O at both checkpoints and GRPO at iter 15 lie together around 0.2-0.23 at k = 1 and 0.45-0.49 at k = 16; GRPO at iter 30 has fallen to about 0.01-0.06; Baseline lies above everything, about 0.37 to 0.54

In the clip study P3O is competitive with or better than the averaged GRPO sweep at every $k$, without choosing a clip. In the FP8 setting P3O keeps its benchmark performance much later into training, while GRPO by iteration 30 has pass@k near zero on all benchmarks.

The appendix's Table 7 gives per-benchmark numbers (read from the arXiv HTML, not checked against a page image). On AMC, pass@1 under BF16 training with FP8 rollouts: GRPO 0.499 at iteration 15 and 0.029 at iteration 30; P3O 0.478 and then 0.529. At iteration 15 the two are close, with GRPO slightly ahead; the gap opens only as GRPO collapses.

::note warning | Two things in Fig. 4 the text does not discuss. In (a), GRPO is the *average* over three clip values, so the comparison is with a typical clip choice, not a tuned one. In (b), the curve labelled Baseline (presumably the starting checkpoint; the caption does not say) is above both trained models at every $k$: by this figure, neither method's FP8 training improved on the starting point at 16K-token evaluation. The FP8 result is about P3O not collapsing, not about it gaining.
::kp feynrl-experiments

## What does the paper show, and what not? {#discussion}
source: feynrl:§5 · feynrl:§4 · feynrl:§D

The discussion restates the stance. Recent methods respond to RL's fragility by adding or re-parameterizing fixed choices (asymmetric clips, staleness budgets), each making the algorithm more sensitive to its configuration. The paper takes the opposite path: the trust placed in an update should be adaptive, not committed to before training.

The authors also say what it does not do. It does not make arbitrary off-policy data reliable: the token-level ratios must be meaningful and the behavior data must have adequate support (Eq. 5's condition that $\pi_b$ covers what $\pi_\theta$ wants). What it does is turn mismatch into a measured property of the batch, removing clip ranges, behavior-weight caps and staleness budgets from the objective. They present the work as a starting point for methods that shrink, rather than expand, the set of knobs fixed in advance.

Here is the evidence, sorted.

**Shown.**
- On Qwen3-4B, GRPO's training reward depends on the clip value and its average sags late in training, while one P3O run does not (Fig. 1a).
- At temperature 1.2 and under BF16/FP8 mismatch with 16K-token rollouts, GRPO collapses on Qwen3-4B while P3O holds (Figs. 2b, 3a); on held-out benchmarks GRPO's FP8 run falls to near-zero pass@k by iteration 30 and P3O's does not (Fig. 4b).
- P3O does this with no clip range, no behavior-weight cap and no staleness parameter in its loss.

**Not shown.**
- Seed variance: no seed count, and the only spread plotted is over clip values.
- Robustness at $T = 0.6$: both methods collapse (Fig. 2a).
- Any effect on Qwen2.5-1.5B in the clip and FP8 experiments: the curves overlap (Figs. 1b, 3b).
- An ESS or ratio trace over training, so the mechanism (ESS falls, the cap tightens) is argued, not observed.
- An ablation separating the cap from the $(1 - e_{\mathcal{B}})$ KL term.
- A plotted comparison with DAPO, GSPO, CISPO or the decoupled loss; their sensitivity is asserted.
- Replay staleness from asynchronous training, the case the single-objective argument is mostly about.

None of this undercuts the idea, which is simple and checkable: a clip range is a guess about the batch made before seeing it, and the batch's ratios can be measured instead. The experiments establish that, on one 4B model under two engine mismatches, measuring beats guessing. Seeds, sizes and genuinely stale data are the obvious next experiment, easy to run in FeynRL: switch `alg_name` between `grpo` and `p3o`, turn on overlap mode, and log `ess_factor` and `clipfrac` side by side ([FeynRL practice thread](#/t/feynrl)).

::note deferred | [L16](#/read/lecture_16) ends its algorithm half at GRPO and its variants. This paper is one answer to the question the lecture leaves open: what to do when rollouts are too expensive to use only once.
::kp feynrl-experiments
