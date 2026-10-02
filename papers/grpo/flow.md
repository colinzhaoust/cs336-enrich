---
title: DeepSeekMath · GRPO, read through
minutes: 40
---
In early 2024 a DeepSeek team (Shao et al.) built a 7B open model that scored over 50% on competition mathematics, and introduced along the way GRPO, PPO without the value model, the RL algorithm DeepSeek's later R1 reasoning model was trained with. The paper has two halves, data and RL. After this read-through you can describe how they mined 120B math tokens from the web, compute GRPO's group advantage and KL term by hand, read every training method from SFT to GRPO as one gradient formula, and say what the paper's RL did and did not improve.

## What does the paper set out to do? {#question}
source: grpo:§abstract · grpo:§1 · grpo:§1.1 · Fig. 1

In 2024 the best models at mathematics, GPT-4 and Gemini Ultra, were closed. Open models trailed far behind on the standard competition benchmark, MATH (problems from high-school maths competitions, graded on the final answer). The paper's goal is to close that gap with a small model, and its claim is that two things do it: better pre-training data, and a cheaper reinforcement-learning algorithm.

The model is built in four stages, and the paper follows them in order, as does this read-through:

| Stage | Starts from | What is added | Result |
|---|---|---|---|
| Corpus | Common Crawl | a classifier loop that finds math pages | DeepSeekMath Corpus, 120B tokens |
| Pre-training | DeepSeek-Coder-Base-v1.5 7B | 500B tokens, 56% of them from the corpus | DeepSeekMath-Base 7B |
| SFT | Base | 776K worked solutions | DeepSeekMath-Instruct 7B |
| RL | Instruct | GRPO on about 144K questions | DeepSeekMath-RL 7B |

::figure grpo:fig1 | top-1 MATH accuracy of open models over time, without tools or voting: DeepSeekMath-RL 7B sits alone above 50%

The headline numbers: 51.7% on MATH with a single answer per problem and no tools, and 60.9% with self-consistency over 64 samples (sample 64 solutions, take the most common final answer). GPT-4 scored 52.9% and Gemini Ultra 53.2% in the paper's own table, so a 7B model came within about 1.5 points.

Side findings get almost as much space: starting from a *code* model helps math, arXiv papers did not, SFT, rejection sampling, DPO, PPO and GRPO can be written as one gradient formula, and RL helped in a more modest way than the headline suggests.

::note why | If you came here for GRPO, the RL half starts at the part titled "Why drop PPO's value model?"; the four parts before it are short. They matter because GRPO's gains are measured on top of this base model and this SFT model.

## How do you find 120B math tokens in Common Crawl? {#corpus}
source: grpo:§2.1 · Fig. 2

Common Crawl is a public scrape of billions of web pages. Somewhere in it are forums like MathOverflow, lecture notes, and solution blogs, mixed with everything else. The question is how to pull out the math pages at scale, cheaply, without a human looking at each one.

The paper's answer is a loop around a cheap text classifier, **fastText** (a bag-of-n-grams linear classifier that scores a page in microseconds). It is the same quality-classifier pattern as [L14's filtering](#/read/lecture_14): train on a small set of pages you trust, score everything, keep the top.

::figure grpo:fig2 | the loop: seed corpus → train fastText → recall math pages from Common Crawl → find math-related domains → annotate URLs by hand → add the new pages to the seed, and around again

**Iteration 1.** The seed is OpenWebMath, an existing 13.6B-token collection of math web text. The classifier gets 500,000 seed pages as positives and 500,000 random Common Crawl pages as negatives (vector dimension 256, learning rate 0.1, word n-grams up to 3, minimum word count 3, 3 epochs). Common Crawl is first shrunk by URL deduplication and near-deduplication to 40B HTML pages. The classifier scores all of them and the pages are ranked by score. How many to keep is decided by pre-training: small runs on the top 40B, 80B, 120B and 160B tokens. The first iteration keeps the top 40B tokens.

**Why iterate.** A classifier trained on OpenWebMath recalls pages that look like OpenWebMath. Math that looks different is missed, because the positives were not diverse enough. So the authors widen the seed using site structure. They group Common Crawl into *domains* (pages sharing a base URL) and compute, for each domain, the share of its pages already collected. A domain with over 10% collected is called math-related (mathoverflow.net, for example). Inside those domains, humans mark which URL patterns hold math (mathoverflow.net/questions), and pages under those patterns that the classifier missed join the seed. A retrained classifier then recalls more.

After four iterations the corpus holds 35.5M pages and 120B tokens, about 3,400 tokens per page on average (our division). The loop stops because in iteration 4 nearly 98% of what was collected had already been found in iteration 3.

**Decontamination.** A corpus scraped from the web contains benchmark questions with their answers. Any text segment that shares an exact 10-gram with GSM8K, MATH, CMATH or AGIEval questions or answers is removed; benchmark texts shorter than 10 grams but at least 3 grams long are matched exactly.

::note aside | Only one human step sits in the loop, and it works at the level of URL patterns, not pages: one annotation can admit thousands of pages. That is what makes the loop cheap. The paper notes the same recipe should work for other domains, code for example.

## Is the new corpus actually better? {#corpus-quality}
source: grpo:§2.2 · grpo:§2.2.1 · Table 1 · Fig. 3

The paper trains the same 1.3B general model (DeepSeek-LLM 1.3B) for 150B tokens on each of four math corpora and evaluates it few-shot on eight benchmarks. The setting is fixed across runs: AdamW ($\beta_1 = 0.9$, $\beta_2 = 0.95$, weight decay 0.1), a peak learning rate of $5.3\times10^{-4}$ reached after 2,000 warmup steps, dropped to 31.6% of the peak at 80% of training and to 10% at 90%, with 4M-token batches of 4K-token sequences.

::figure grpo:table1 | read the columns down the rows: the arXiv-heavy MathPile is below no math training on most columns, and the DeepSeekMath Corpus row leads all eight columns

| Corpus | Size | GSM8K | MATH | CMATH |
|---|---|---|---|---|
| No math training | – | 2.9% | 3.0% | 12.3% |
| MathPile | 8.9B | 2.7% | 3.3% | 1.2% |
| OpenWebMath | 13.6B | 11.5% | 8.9% | 16.8% |
| Proof-Pile-2 | 51.9B | 14.3% | 11.2% | 19.9% |
| DeepSeekMath Corpus | 120.2B | 23.8% | 13.6% | 41.5% |

Three readings follow. **Quality:** a fair comparison needs equal exposure, and Fig. 3 gives it. At 50B tokens, one full epoch of Proof-Pile-2, the DeepSeekMath model is already ahead, so the lead is not only from size. **Multilinguality:** the corpus is mostly English and Chinese, and the Chinese benchmark CMATH jumps to 41.5%, while the English-centric corpora barely move it, and MathPile pushes it down from 12.3% to 1.2%. **Scale:** 150B tokens is 1.25 passes over the new corpus but about 11 over OpenWebMath and 17 over MathPile (our division); the small corpora are repeated many times and their curves flatten early.

::figure grpo:fig3 | benchmark accuracy against training tokens for the four corpora: the DeepSeekMath curve keeps climbing while the others plateau

::note warning | Table 1 compares corpora on one 1.3B model. The paper does not test whether the ranking holds at 7B, though the 7B base model below is consistent with it.

## What does the 7B base model learn from it? {#base}
source: grpo:§2.3 · Tables 2-4 · grpo:§5.1 · Tables 6-9

DeepSeekMath-Base 7B starts from DeepSeek-Coder-Base-v1.5 7B, a code model, and continues pre-training for 500B tokens with the §2.2.1 recipe, except for a peak learning rate of $4.2\times10^{-4}$ and 10M-token batches. The mix:

| Source | Share | Tokens (our arithmetic) |
|---|---|---|
| DeepSeekMath Corpus | 56% | 280B, about 2.3 passes over 120B |
| AlgebraicStack (math code) | 4% | 20B |
| arXiv | 10% | 50B |
| GitHub code | 20% | 100B |
| Common Crawl text, English and Chinese | 10% | 50B |

The code and general text are there to keep what the starting model already knew. With few-shot chain-of-thought prompting (worked examples in the prompt, the model writes its reasoning before the answer), the base model reaches 64.2% on GSM8K (grade-school word problems) and 36.2% on MATH. Minerva 540B, a closed PaLM model further trained on math and 77 times larger, scores 58.8% and 33.6%. The best open base model before it, Llemma 34B, scores 54.0% and 25.3%.

::figure grpo:table2 | DeepSeekMath-Base 7B's row beats every open base model on all eight benchmarks, and Minerva 540B on GSM8K and MATH; Minerva 540B stays ahead on OCW (17.6% vs 15.4%) and MMLU-STEM (63.9% vs 56.5%)

The same base model also writes Python to solve problems (66.9% on GSM8K+Python, 31.4% on MATH+Python) and drafts formal proofs in the Isabelle proof assistant (25.8% and 24.6% on the Olympiad-level miniF2F, valid and test), both above Llemma 34B (Table 3). Math training lifted general reasoning too: MMLU (exam questions in 57 subjects) rises from 49.1% to 54.9% and BBH (23 hard multi-step reasoning tasks) from 55.2% to 59.5% over the code model it started from (Table 4).

::note slip | §2.3 says the code tokens let DeepSeekMath-Base "maintain" the code model's coding scores. Against the checkpoint it was actually trained from (taken just before learning-rate decay) that holds: HumanEval 40.2% → 40.9%, MBPP 52.6% → 52.6%. Against the released, fully decayed DeepSeek-Coder-Base-v1.5 it is a drop: 43.2% → 40.9% and 60.4% → 52.6%.

### Two lessons from the pre-training ablations

The paper's discussion (§5.1, mostly on the 1.3B model, with an 89B-token iteration-2 version of the corpus) answers two questions that explain the choices above.

**Does code help math?** Train 400B tokens of code, then 150B of math, and compare with 400B general tokens then the same math. Code first ends best without tools (GSM8K 21.9% vs 19.1%, MATH 15.3% vs 14.4%) and with tools (GSM8K+Python 17.4% vs 14.3%). Mixing code and math in one stage keeps coding from being forgotten and gives the best tool-use scores (19.7% and 13.5%), but costs math without tools (17.6% and 12.1%); the authors guess a 1.3B model lacks the capacity for both at once. This is why the 7B model starts from a code model.

**Do arXiv papers help?** Training on arXiv alone, either MathPile or the 28.0B-token ArXiv-RedPajama, gives, in the paper's words, "no notable improvements or even deterioration", at 1.3B (150B tokens) and at 7B (40B tokens). At 7B, ArXiv-RedPajama moves miniF2F-test from 21.7% down to 11.9%. The authors scope it themselves: they did not test arXiv mixed with other data, at larger scale, or on tasks such as turning formal proofs back into prose.

::note slip | Table 8 has one exception the prose skips: at 7B, SAT rises from 40.6% to 46.9% with MathPile and to 50.0% with ArXiv-RedPajama, though the same corpora lower GSM8K, MATH and CMATH.

## Instruction tuning: what does SFT add? {#sft}
source: grpo:§3.1 · grpo:§3.2 · Table 5

A base model continues text; it does not reliably answer a question in a fixed format. **Supervised fine-tuning (SFT)**, recapped in [L15](#/read/lecture_15), trains it on question-solution pairs. Here the 776K examples, English and Chinese, come in three solution formats:
- **chain-of-thought (CoT)**: the reasoning in prose, ending with the answer;
- **program-of-thought (PoT)**: a Python program whose output is the answer;
- **tool-integrated reasoning**: prose that calls Python for the computations and continues from the results.

The English part annotates GSM8K and MATH problems with tool-integrated solutions and adds subsets of MathInstruct and Lila-OOD; the Chinese part covers K-12 problems across 76 sub-topics. Training is short: examples are packed into 4K-token sequences, 500 steps at batch size 256 with a constant learning rate of $5\times10^{-5}$, so at most about 128,000 packed sequences.

The result, DeepSeekMath-Instruct 7B, scores 82.9% on GSM8K and 46.8% on MATH with chain-of-thought, and 57.4% on MATH with tools. On MATH without tools it beats every open model by at least 9 points (the closest, InternLM2-Math 20B, has 37.7%), including Qwen 72B and the PPO-trained WizardMath-v1.1 7B, but stays below GPT-4 (52.9%) and Gemini Ultra (53.2%).

::figure grpo:table5 | the chain-of-thought block: Instruct's 46.8% on MATH beats every open model but trails GPT-4 and Gemini Ultra; the RL row below it is where the next parts lead

## Why drop PPO's value model? {#ppo}
source: grpo:§4.1 · grpo:§4.1.1 · Eq. 1 · Eq. 2 · Fig. 4

RL after SFT works like this: give the model a question, let it write answers, score them, and make high-scoring answers more likely. The paper runs it with a learned **reward model** $r_\varphi$ that scores an answer (trained from DeepSeekMath-Base on data labelled by answer correctness). To see what GRPO changes, start from the gradient every such method uses, recapped from [L16](#/read/lecture_16):

$$ \nabla_\theta\, \mathbb{E}_{o\sim\pi_\theta(\cdot\mid q)}[R(o)] = \mathbb{E}_{o}\Big[R(o)\sum_{t=1}^{|o|}\nabla_\theta\log\pi_\theta(o_t\mid q,o_{<t})\Big] $$

Read it as weighted SFT: $\nabla\log\pi_\theta$ is the step that imitates the sampled answer $o$, and the reward is its weight. The estimate is noisy, so one subtracts a **baseline** $b$ from the reward. Any $b$ that does not depend on the sampled answer is free: $\mathbb{E}_o[b\,\nabla\log\pi_\theta(o)] = b\,\nabla\sum_o\pi_\theta(o) = b\,\nabla 1 = 0$. The expected direction stays the same and only the noise changes. The paper never states this identity, but its whole design rests on it.

::worked supp-policy-gradient
::widget fixture:grpo--baseline | slide b anywhere: the expected-gradient line stays flat while the one-sample noise changes, reaching zero at b = π(b); dividing by s only rescales the flat line

**PPO**, the standard RL algorithm for LLMs since InstructGPT ([L15's RLHF](#/read/lecture_15)), maximizes a clipped surrogate. Answers $o$ are sampled from the policy as it stood before the update, $\pi_{\theta_{old}}$, and each token is reweighted by the probability ratio:

$$ \mathcal{J}_{PPO}(\theta) = \mathbb{E}_{q,\;o\sim\pi_{\theta_{old}}}\ \frac{1}{|o|}\sum_{t=1}^{|o|}\min\Big[\rho_t A_t,\ \operatorname{clip}(\rho_t, 1-\varepsilon, 1+\varepsilon)\,A_t\Big],\qquad \rho_t = \frac{\pi_\theta(o_t\mid q,o_{<t})}{\pi_{\theta_{old}}(o_t\mid q,o_{<t})} \qquad (1) $$

The clip stops a token's probability from moving more than a factor $1\pm\varepsilon$ in the direction its advantage favours, which makes it safe to take several gradient steps on one batch of samples. The **advantage** $A_t$, how much better token $t$ turned out than expected, comes from **GAE** (generalized advantage estimation) over the rewards and a learned **value function** $V_\psi$, a second network that predicts at every token the reward still to come. To keep the policy from over-optimizing the reward model, PPO for LLMs also folds a KL penalty (KL divergence measures how far one distribution is from another) against a frozen **reference model** $\pi_{ref}$ (usually the SFT model) into the reward of every token:

$$ r_t = r_\varphi(q, o_{\le t}) - \beta\log\frac{\pi_\theta(o_t\mid q,o_{<t})}{\pi_{ref}(o_t\mid q,o_{<t})} \qquad (2) $$

The paper's objection is to $V_\psi$, for two reasons. It is "typically another model of comparable size as the policy" (§4.1.1), so it doubles the trained weights, gradients and optimizer state. And it must be accurate at *every* token, although the reward model scores only the last one; learning a per-token value from one number at the end is hard.

::figure grpo:fig4 | top, PPO: policy, reference, reward and value models, with GAE combining reward and value into advantages; bottom, GRPO: the value model is gone, the policy samples a group o₁…o_G, and "group computation" turns the G rewards into advantages

GRPO's replacement uses something cheap in language modelling: you can sample the same question many times. For each question it samples a group of $G$ answers and uses their *average reward* as the baseline. The question is shared by all $G$ answers, so the baseline depends on the question and not on any one answer's luck: the identity above says it does not bias the direction (up to one detail in the next part).

::predict ppo-value-model-cost

Counting models makes the saving concrete. PPO holds four 7B-class networks: the trained policy and critic, the frozen reference and reward models. GRPO holds three. If a trained network costs 16 bytes per parameter (weights, gradients, Adam states) and a frozen one 2, PPO needs $112 + 112 + 14 + 14 = 252$ GB of model state for 7B models, GRPO $140$ GB. The 112 GB saved is exactly the critic; the reference model stays because GRPO keeps a KL term.

::note warning | The paper quantifies neither reason: there is no memory measurement and no PPO-versus-GRPO comparison at equal compute anywhere in it. Its saving is in memory and critic passes, not in generation (see the recipe below).
::kp supp-policy-gradient
::kp ppo-value-model-cost

## How does a group of answers replace the critic? {#advantage}
source: grpo:§4.1.2 · grpo:algorithm1 (line 9)

Here is the advantage GRPO uses under **outcome supervision**, where one reward scores each whole answer. For a question $q$, sample $G$ answers from $\pi_{\theta_{old}}$, score them, $\mathbf r = \{r_1,\dots,r_G\}$, and normalize within the group:

$$ \hat A_{i,t} = \tilde r_i = \frac{r_i - \operatorname{mean}(\mathbf r)}{\operatorname{std}(\mathbf r)} \quad\text{for every token } t \text{ of answer } o_i $$

Three properties follow at once. The advantages of one group sum to zero, so some answers are pushed up and others down. The sign is *relative*: an answer above its group's average is reinforced, one below is penalized, whatever its absolute score. And every token of an answer gets the same number, because the reward arrived only at the end.

::predict group-relative-advantage-outcome
::worked group-relative-advantage-outcome

The rare success gets the big push. With one correct answer in five, the correct one gets $+2.0$ and each wrong one $-0.5$; with four correct in five the numbers flip to $+0.5$ and $-2.0$. A group where every answer gets the same reward, all right or all wrong, has $r_i - \operatorname{mean} = 0$ everywhere: that question contributes no reward signal at all. (The formula is $0/0$ there; implementations add a small constant to the std so the result is 0.)

::widget fixture:grpo--advantage-map | press "three questions": the all-correct and all-wrong groups turn white (zero advantage, zero signal) and only the mixed group gets green and red rows, each output's cells one flat colour because the advantage ignores token position

The paper says the group-relative form "aligns well with the comparative nature of reward models" (§4.1.1): reward models are trained on comparisons between answers to the same question, so their scores are most meaningful relative to one another. It leaves two details open, and implementations differ on both. Is the std divided by $G$ or $G-1$? Is a stability constant added? FeynRL, an open RL library, picks $G-1$ and $10^{-8}$:

::code rollouts/base.py:L131-L136 | an implementation (FeynRL), not the paper's code: the group mean, then a Bessel-corrected std; with G − 1 the one-in-five example gives +1.79 and −0.45 instead of +2.0 and −0.5

::note aside | The course questions this advantage on two counts the paper does not raise ([L16](#/read/lecture_16), slides 22-24). Subtracting the group mean is a valid baseline; dividing by the group std is not, since it reweights each question by $1/\operatorname{std}$ and so gives near-uniform groups (too easy or too hard questions) extra weight. And the $1/|o_i|$ in the objective below makes a long wrong answer cheaper per token. Dr. GRPO (2025) removes both.
::kp group-relative-advantage-outcome

## The GRPO objective: PPO's clip with a KL term in the loss {#objective}
source: grpo:§4.1.1 · Eq. 3 · Eq. 4 · grpo:§A.1.5 · grpo:§A.1.6

With $\hat A_{i,t}$ in hand, GRPO keeps PPO's surrogate almost unchanged. Write $\rho_{i,t} = \pi_\theta(o_{i,t}\mid q,o_{i,<t})\,/\,\pi_{\theta_{old}}(o_{i,t}\mid q,o_{i,<t})$ for the ratio of token $t$ of answer $i$. GRPO maximizes

$$ \mathcal{J}_{GRPO}(\theta) = \mathbb{E}_{q,\;\{o_i\}_{i=1}^{G}\sim\pi_{\theta_{old}}}\ \frac{1}{G}\sum_{i=1}^{G}\frac{1}{|o_i|}\sum_{t=1}^{|o_i|}\Big\{\min\Big[\rho_{i,t}\hat A_{i,t},\ \operatorname{clip}(\rho_{i,t},1-\varepsilon,1+\varepsilon)\,\hat A_{i,t}\Big] - \beta\,\mathbb{D}_{KL}\big[\pi_\theta\,\|\,\pi_{ref}\big]\Big\} \qquad (3) $$

Read it from the inside out. Each token gets PPO's clipped term with the group advantage. The KL penalty to the reference model is subtracted *per token, in the objective*. The token terms are averaged over each answer's length, then over the $G$ answers of the group.

::worked grpo-objective-clip-kl

**Why move the KL out of the reward?** In PPO's Eq. 2 the penalty changes every token's reward, so it flows through GAE into the advantages and the value targets. GRPO's advantages are a pure function of the group's rewards; putting the KL anywhere else would mean re-normalizing a mix of task reward and penalty. In the paper's words, the change avoids "complicating the calculation of $\hat A_{i,t}$". The two terms now act separately: $\beta$ changes the KL term and never the advantage.

::predict grpo-objective-clip-kl

**When does the clip act?** At the moment the first gradient is taken after sampling, $\pi_\theta = \pi_{\theta_{old}}$, so every $\rho_{i,t} = 1$. One is inside the band $[1-\varepsilon, 1+\varepsilon]$, so the min picks the unclipped term. Its value is $\hat A_{i,t}$, and its gradient is $\hat A_{i,t}\,\nabla\rho = \hat A_{i,t}\,\nabla_\theta\log\pi_\theta$ (the ratio is constant in value but not in slope). The clip only starts to matter on a second, third... update of the same samples, once ratios have drifted out of the band.

::predict supp-importance-ratio
::widget fixture:grpo--token-coefficient | set log π_θ equal to log π_old: the point sits at ρ = 1 in the shaded band and the ratio part of the coefficient equals Â; drag log π_θ until ρ leaves the band on the side Â's sign favours and only then does that part drop to 0

::note why | The training recipe below uses "a single update following each exploration stage". Put together with the above (our inference, which the course states in one line on L16's slide 18): in the runs the paper reports, the clip never bound, and GRPO was a plain policy gradient with group-normalized rewards plus a KL term. The paper never even prints a value for $\varepsilon$.

### The KL term and its estimator

Computing the exact $\mathbb{D}_{KL}[\pi_\theta\|\pi_{ref}]$ would need a sum over the whole vocabulary at every position. GRPO instead estimates it from the one token that was sampled, using the two log-probabilities it already has. With $u = \pi_{ref}(o_{i,t}\mid q,o_{i,<t})\,/\,\pi_\theta(o_{i,t}\mid q,o_{i,<t})$:

$$ \hat{\mathbb{D}}_{KL}\big[\pi_\theta\,\|\,\pi_{ref}\big] = u - \log u - 1 \qquad (4) $$

The paper calls it unbiased (citing Schulman's 2020 note on KL approximations) and "guaranteed to be positive". The plain estimator is $-\log u = \log(\pi_\theta/\pi_{ref})$, the per-token term of Eq. 2; its average under $\pi_\theta$ is the true KL, but single samples can be negative. Eq. 4 adds $u - 1$, which has mean zero, $\mathbb{E}_{\pi_\theta}[u] = \sum\pi_{ref} = 1$, so the expectation is unchanged. And $f(u) = u - \log u - 1$ is convex with minimum $f(1) = 0$, so no sample is ever negative.

::predict kl-k3-estimator
::worked kl-k3-estimator
::widget fixture:grpo--kl-estimators | move log π_θ below log π_ref: the blue plain log-ratio goes negative on that token while the orange Eq. 4 estimate stays above zero, and both bars still average to the green true KL

The estimate is asymmetric: halving a token's probability relative to the reference ($u = 2$) costs $0.307$; doubling it ($u = 0.5$) costs $0.193$. The penalty leans harder against dropping tokens the reference liked than against inventing new ones.

::code algs/RL/common.py:L131-L139 | FeynRL's version of Eq. 4 in log space, log(π/π_ref) + π_ref/π − 1, with the exponent clamped to ±10 against overflow

::note slip | "Guaranteed to be positive" is slightly off: the estimate is zero when $\pi_\theta = \pi_{ref}$ on that token, so it is non-negative. Unbiasedness also needs samples from $\pi_\theta$; the samples come from $\pi_{\theta_{old}}$, which equals $\pi_\theta$ only on the first update of a batch. The paper does not discuss either point.

Differentiating Eq. 4 shows what the KL term does to each token. Since $\nabla u = -u\,\nabla\log\pi_\theta$, the gradient of $-\beta(u-\log u-1)$ is $\beta(u-1)\,\nabla\log\pi_\theta$. With the ratio term at $\rho = 1$, every token's update is $\nabla\log\pi_\theta$ times one number:

$$ GC_{GRPO} = \hat A_{i,t} + \beta\Big(\frac{\pi_{ref}(o_{i,t}\mid q,o_{i,<t})}{\pi_\theta(o_{i,t}\mid q,o_{i,<t})} - 1\Big) \qquad (21) $$

The KL part is a restoring force: positive when the policy has made the token less likely than the reference, negative when more likely. Example: $\hat A = -0.6$, $\beta = 0.1$, and the policy has cut the token's probability to a third of the reference's. Then $GC = -0.6 + 0.1\times(3-1) = -0.4$: the KL softens the push down. In a uniform group ($\hat A = 0$) the KL part is the *only* gradient. This coefficient, Eq. 21 from the appendix, returns in the unified paradigm.

::code algs/GRPO/grpo.py:L165-L188 | the whole Eq. 3 in an implementation (FeynRL): ratio from stored log-probs, min of unclipped and clipped terms negated into a loss, then + kl_coeff × the Eq. 4 sum; FeynRL normalizes by a global token count instead of the paper's 1/|o_i| and allows different low and high clip bounds
::kp supp-importance-ratio
::kp grpo-objective-clip-kl
::kp kl-k3-estimator

## Process supervision: can each reasoning step get its own reward? {#process}
source: grpo:§4.1.3

Outcome supervision gives an entire solution one number. A solution that reaches the right answer through a wrong step and a lucky fix gets the same credit on every token as a clean one. The paper argues this "may not be sufficient and efficient" for complex math, and also tries **process supervision**: a *process reward model* scores each reasoning step, placing a reward on that step's last token.

Answer $i$ has $K_i$ steps, and step $j$ ends at token $index(j)$. All step rewards of all $G$ answers go into one pool,

$$ \mathbf R = \big\{\{r_1^{index(1)},\dots,r_1^{index(K_1)}\},\ \dots,\ \{r_G^{index(1)},\dots,r_G^{index(K_G)}\}\big\},\qquad \tilde r_i^{index(j)} = \frac{r_i^{index(j)} - \operatorname{mean}(\mathbf R)}{\operatorname{std}(\mathbf R)} $$

and a token's advantage is the sum of the normalized rewards of every step that ends at or after it:

$$ \hat A_{i,t} = \sum_{index(j)\ge t}\tilde r_i^{index(j)} $$

Two details are easy to get wrong. The normalization pools *every* step of *every* answer; it does not compare step 2 of one answer with step 2 of another. And the advantage is an undiscounted reward-to-go: a token is credited with its own step and everything after it, so the first step's tokens collect $K_i$ terms and the last step's only one.

::predict process-supervision-advantage
::worked process-supervision-advantage

The reward-to-go can flip a step's sign. In the widget's "one bad middle step" preset, two answers have step rewards $(0.8, 0.1, 0.9)$ and $(0.8, 0.8, 0.9)$. The pool has mean $0.717$ and std $0.279$, so the first answer's steps normalize to $+0.30, -2.21, +0.66$. Its first step was fine on its own, yet its tokens get $0.30 - 2.21 + 0.66 = -1.25$: they led into the bad step. The second answer's first step gets $+1.25$.

::widget fixture:grpo--advantage-map | switch to process supervision and press "one bad middle step": the first answer's opening step turns red although its own reward is above average, because its tokens sum every later step, including the bad one

::note aside | Process supervision did slightly better than outcome supervision in the paper's 1.3B comparison (Fig. 5, in the unified-paradigm part below). [L16](#/read/lecture_16) follows the thread forward: DeepSeek's later R1 dropped process reward models and went back to outcome rewards from a rule-based checker.
::kp process-supervision-advantage

## Iterative GRPO: what if the reward model falls behind? {#iterative}
source: grpo:§4.1.4 · grpo:algorithm1 · Fig. 6

A learned reward model was trained on answers from the SFT model. As RL moves the policy, its answers drift into territory the reward model never saw, and its scores become less trustworthy: "the old reward model may not be sufficient to supervise the current policy model" (§4.1.4). The paper's fix wraps GRPO in an outer loop, Algorithm 1:

::figure grpo:algorithm1 | read lines 3, 6 and 12 together: the reference model is reset once per outer iteration, the old policy once per step, and the reward model at the end of each iteration

- **Line 3**, start of each outer iteration: set the reference model to the current policy, $\pi_{ref} \leftarrow \pi_\theta$.
- **Line 6**, every step: set $\pi_{\theta_{old}} \leftarrow \pi_\theta$, then sample $G$ answers per question from it.
- **Lines 8-9:** score them with $r_\varphi$ and compute $\hat A_{i,t}$ from the group.
- **Lines 10-11:** run $\mu$ updates on this batch by maximizing the GRPO objective.
- **Line 12**, end of the iteration: continue training $r_\varphi$ on new samples from the policy, with a replay buffer that keeps 10% historical data so it does not forget how to score older answers.

::animation fixture:grpo--alg1-iterations | π_old jumps to the current policy before every step's sampling, while π_ref moves only once per iteration, right after r_φ is retrained, and then points at θ₃ (the end of iteration 1), not θ₀

::worked iterative-grpo

So the KL anchor moves. Inside one iteration, $\pi_{ref}$ is fixed and the KL term restrains drift from where the iteration began. Across iterations the policy can travel arbitrarily far from the SFT model, as long as each hop is small. The paper motivates only the reward-model refresh; the reference reset is stated without a reason.

::predict iterative-grpo

The experiment runs two outer iterations from DeepSeekMath-Instruct 7B. Iterative RL "significantly improves the performance, especially at the first iteration" (§5.2.1).

::figure grpo:fig6 | GSM8K and MATH accuracy of iterative RL from DeepSeekMath-Instruct 7B; the text says the iterations help, the first one most, and the numbers exist only in the image
::kp iterative-grpo

## The DeepSeekMath-RL recipe, and what it bought {#recipe}
source: grpo:§4.2 · Table 5

The RL run starts from DeepSeekMath-Instruct 7B and uses a deliberately narrow question set: about 144K chain-of-thought questions about GSM8K and MATH, taken from the SFT data. Every other SFT question is excluded on purpose, so that every benchmark except chain-of-thought GSM8K and MATH counts as out-of-domain and tests whether RL generalizes. The reward model is trained from DeepSeekMath-Base 7B at learning rate $2\times10^{-5}$.

| GRPO setting | Value |
|---|---|
| policy learning rate | $1\times10^{-6}$ |
| KL coefficient $\beta$ | 0.04 |
| answers per question $G$ | 64 |
| max answer length | 1,024 tokens |
| batch size | 1,024 |
| policy updates per sampling stage | 1 |
| clip $\varepsilon$ | not given |

::predict grpo-training-recipe

The batch arithmetic shows where GRPO spends its compute. A batch of 1,024 questions with $G = 64$ is 65,536 sampled answers, up to $65{,}536 \times 1{,}024 \approx 67$M generated tokens, for *one* policy update. Generation, not the missing critic, dominates the cost; this is why efficient inference ([L10](#/read/lecture_10)) is part of RL. The paper does not print the number of RL steps.

::note warning | $G$ is answers per question, not questions per batch; and with one update per sampling stage, $\varepsilon$ would not act even if it were given. A reproduction has to choose it, and it does not matter for this recipe.

**Results.** Against the Instruct model it started from (Table 5):

| Benchmark | Setting | Instruct | RL | Change |
|---|---|---|---|---|
| GSM8K | chain-of-thought (in-domain) | 82.9% | 88.2% | +5.3 |
| MATH | chain-of-thought (in-domain) | 46.8% | 51.7% | +4.9 |
| MGSM-zh | chain-of-thought | 73.2% | 79.6% | +6.4 |
| CMATH | chain-of-thought | 84.6% | 88.8% | +4.2 |
| GSM8K | tool-integrated | 83.7% | 86.7% | +3.0 |
| MATH | tool-integrated | 57.4% | 58.8% | +1.4 |
| MGSM-zh | tool-integrated | 72.0% | 78.4% | +6.4 |
| CMATH | tool-integrated | 84.3% | 87.6% | +3.3 |

Every column improves, including the Chinese grade-school and exam benchmarks (MGSM-zh, CMATH) and the tool-integrated format, none of which had RL questions. The paper reads this as evidence that RL generalizes beyond its training set. With 51.7% on MATH the RL model beats every open model from 7B to 70B and most closed ones in the table, though not GPT-4 (52.9%) or Gemini Ultra (53.2%).

::figure grpo:table5 | the two DeepSeekMath-RL rows against the Instruct rows above them: higher in every column of both blocks
::note warning | No seeds or variance are reported, so small gains such as 1.4 points on tool-integrated MATH cannot be told apart from run-to-run noise. The out-of-domain gains are on math benchmarks in other languages and formats, not on other subjects.
::kp grpo-training-recipe

## One formula for SFT, RFT, DPO, PPO and GRPO {#paradigm}
source: grpo:§5.2.1 · Eq. 5 · Table 10 · grpo:§A.1 · Eq. 6-21 · Fig. 5

SFT, rejection sampling, DPO, PPO and GRPO are usually taught as different ideas. The paper shows that all of their gradients have one shape:

$$ \nabla_\theta\mathcal{J}_{\mathcal{A}}(\theta) = \mathbb{E}_{(q,o)\sim\mathcal{D}}\Big[\frac{1}{|o|}\sum_{t=1}^{|o|} GC_{\mathcal{A}}(q,o,t,\pi_{rf})\,\nabla_\theta\log\pi_\theta(o_t\mid q,o_{<t})\Big] \qquad (5) $$

Every method imitates some token sequences with some weight. They differ in three components:
- the **data source** $\mathcal{D}$: where the (question, answer) pairs come from. Fixed human data, answers sampled once from the SFT model (*offline*), or answers sampled from the policy as it trains (*online*);
- the **reward function** $\pi_{rf}$: where the signal comes from. Human selection, a correctness *rule*, or a learned reward *model*;
- the **algorithm** $\mathcal{A}$: how data and signal become the **gradient coefficient** $GC$, the per-token weight that decides how hard each token is reinforced or penalized.

::figure grpo:table10 | six rows, three columns; read down the gradient-coefficient column: 1 for SFT, a 0/1 indicator for both RFTs, a sigmoid for DPO, the GAE advantage for PPO, the group advantage plus a KL term for GRPO

| Method | Data source | Reward | Gradient coefficient |
|---|---|---|---|
| SFT | $q, o \sim P_{sft}$ (human data) | (human selection) | $1$ |
| RFT | $q \sim P_{sft}$, $o \sim \pi_{sft}$ (offline) | Rule | $\mathbb{I}(o)$, Eq. 10 |
| DPO | $q \sim P_{sft}$, $o^+, o^- \sim \pi_{sft}$ (offline) | Rule | sigmoid of a log-ratio gap, Eq. 14 |
| Online RFT | $q \sim P_{sft}$, $o \sim \pi_\theta$ (online) | Rule | $\mathbb{I}(o)$, Eq. 10 |
| PPO | $q \sim P_{sft}$, $o \sim \pi_\theta$ (online) | Model | $A_t$ from GAE, Eq. 18 |
| GRPO | $q \sim P_{sft}$, $\{o_i\} \sim \pi_\theta$ (online) | Model | $\hat A_{i,t} + \beta(\pi_{ref}/\pi_\theta - 1)$, Eq. 21 |

Appendix A.1 derives each row. SFT's coefficient is 1 on every token, with the human choice of data acting as the reward. **Rejection-sampling fine-tuning (RFT)** samples several answers per question from the SFT model, keeps the correct ones and fine-tunes on them, so its coefficient is the correctness indicator:

$$ GC_{RFT}(q,o,t) = \mathbb{I}(o) = \begin{cases}1 & \text{the answer of } o \text{ is correct}\\ 0 & \text{otherwise}\end{cases} \qquad (10) $$

Online RFT is the same with answers sampled from the live policy. DPO ([L15](#/read/lecture_15)) has two data terms, the preferred answer pushed up and the rejected one down, both weighted by a sigmoid of how far the policy already prefers the wrong one. PPO and GRPO fit only after assuming one update per sampling stage, so that the ratio is 1 and the min and clip disappear, exactly as in the objective part above.

::predict unified-paradigm

The paradigm's use is that it isolates one difference at a time. Fig. 5 trains DeepSeekMath-Instruct 1.3B further with four methods.

::figure grpo:fig5 | GSM8K and MATH accuracy over training steps for RFT, Online RFT, GRPO with outcome supervision and GRPO with process supervision: the online methods pull away from RFT late in training, and both GRPO curves sit above Online RFT

**Data source, holding the coefficient fixed.** RFT and Online RFT share $GC = \mathbb{I}(o)$ and differ only in where answers come from. Online RFT is comparable early but "gains an absolute advantage in the later stage" (§5.2.1). The paper's explanation: at first the policy is still close to the SFT model, so the two sample sets look alike; later the policy has drifted, and samples from the frozen SFT model describe a model that no longer exists.

**Coefficient, holding the online data fixed.** Online RFT and GRPO differ in the coefficient. $\mathbb{I}(o)$ is absolute and never negative: every correct answer gets the same push and wrong answers are ignored. $\hat A$ is relative: answers above the group mean are pushed up by how far above they are, answers below are pushed *down*. GRPO beats Online RFT, which the paper credits to "altering positive and negative gradient coefficients", and process-supervised GRPO beats outcome-supervised GRPO, credited to finer, step-aware coefficients.

::worked gradient-coefficient-comparison
::predict gradient-coefficient-comparison

The relative coefficient has a blind spot that the indicator does not. On a question the policy always solves, Online RFT keeps putting weight 1 on every answer, while GRPO's $\hat A$ is 0 for all of them. On a question it never solves, both give zero. GRPO's negative gradients exist only in mixed groups.

::widget fixture:grpo--advantage-map | press "three questions" and read the last line: GRPO pushes down only the two wrong answers of the mixed group and leaves both uniform groups at zero, while Online RFT's I(o) pushes up all six correct answers and pushes nothing down

::note aside | Read off Fig. 5 as shown on [L16's](#/read/lecture_16) slide 21 (the text prints no numbers): over about 9,000 steps GSM8K rises from about 56.5% to about 60% for RFT, 62% for Online RFT, 64% for GRPO with outcome and 65% with process supervision; on MATH, RFT reaches about 28% and GRPO about 30%. The course reads the RFT gap as "learning from negative gradients".
::note warning | The GRPO-versus-Online-RFT comparison changes two components at once: the coefficient and, per Table 10, the reward (a learned model for GRPO, a rule for Online RFT). The figure cannot say how much of the gap is the coefficient. All four curves are single runs of a 1.3B model.
::note slip | Small inconsistencies in this part. Fig. 5's caption names a "DeepSeekMath-Instruct 1.3B" model that the paper never describes (it trains DeepSeek-LLM 1.3B in §2.2 and §5.1, and instruction-tunes only the 7B). Table 10 lists DPO's reward as "Rule", while Appendix A.1.4 says human preference, which "can be Rule" in math. Its caption writes $\pi_{\theta_{sft}}$ where the table writes $\pi_{sft}$. In Eq. 12 the rejected answer's log-ratio is written over $o^-_{<t}$ where $o^-_t$ is meant, and Eqs. 20-21 drop $q$ from the conditioning. Algorithm 1 line 11 says to maximize "the GRPO objective (Equation 21)", but Eq. 21 is the gradient coefficient; the objective is Eq. 3.
::kp unified-paradigm
::kp gradient-coefficient-comparison

## Why does RL help: new ability or better aim? {#why-rl}
source: grpo:§5.2.2 · Fig. 7

The RL model gained about 5 points on GSM8K and MATH from a subset of the very questions the SFT model had already been trained on. Did RL teach the model to solve problems it could not solve before, or did it make the model more reliable on problems it could already solve? The paper separates the two with two metrics, computed from $K$ samples per problem at temperature 0.7:
- **Pass@K**: a problem counts as solved if *any* of the $K$ samples is correct. It measures what the model *can* reach.
- **Maj@K**: a problem counts as solved if the *most common* final answer among the $K$ samples is correct (majority voting, as in the abstract's self-consistency score). It measures how much probability the model puts on the right answer.

::figure grpo:fig7 | Maj@K and Pass@K against K for the Instruct and RL 7B models on GSM8K and MATH, at temperature 0.7; compare the gap between the two models on each metric

The finding, in the caption's words: "RL enhances Maj@K but not Pass@K." The authors' reading: RL makes "the output distribution more robust", boosting the correct response from the top-K rather than enhancing "fundamental capabilities" (§5.2.2). A toy case shows the mechanism. A model solves one problem in 10 of 64 samples before RL and 40 of 64 after. Pass@64 is 1 both times. Maj@64 can flip from wrong to right, because the correct answer now wins the vote.

::predict rl-findings-maj-not-pass

The same pattern over five problems, 8 samples each (our example). Correct samples before RL: 0, 2, 3, 6, 1; after: 0, 5, 7, 8, 3. Pass@8 is 4 of 5 = 80% both times: the problem with 0 correct stays unsolved. Maj@8, counting a problem solved when at least 5 of its 8 samples are correct, goes from 1 of 5 (20%) to 3 of 5 (60%). (Counting "at least 5 of 8" is a simplification: if the wrong answers disagree with each other, fewer correct samples can win the vote.)

This fits the gradient-coefficient analysis (our connection; the paper does not draw it). GRPO's coefficient is zero on a question whose $G$ samples all fail, so a problem the model never solves gives no signal to learn from. What RL can do is shift probability toward answers the model already produces sometimes, which is what Maj@K rewards.

::note warning | Fig. 7 is an image and the text gives no values, no K range and no error bars, so "not Pass@K" is the authors' description, not a measured difference. The paper also does not test whether Pass@K is unchanged at very large K or just not improved at the K plotted. The answers here were capped at 1,024 tokens; [L16](#/read/lecture_16) shows R1 learning far longer reasoning chains with RL, a setting this paper does not test.
::kp rl-findings-maj-not-pass

## What would make RL more effective, and what are the limits? {#limits}
source: grpo:§5.2.3 · grpo:§6

The paper ends its RL discussion with directions, one per component of Eq. 5.

**Data source.** The RL questions were the SFT questions, and the answers were sampled with "a naive nucleus sampling" (drawing from the smallest set of top tokens that covers most of the probability). The authors name this as a likely reason only Maj@K improved, and propose out-of-distribution questions and stronger search-based decoding, such as tree search. Efficient inference matters too, since it sets how much exploration a training budget buys.

**Algorithm.** Every method in Table 10 fully trusts the reward signal: the coefficient follows the reward, right or wrong. But rewards are noisy. Even the carefully annotated PRM800K process-supervision dataset has, by the paper's estimate, about 20% incorrect annotations. The authors want algorithms robust to noisy rewards, in the spirit of weak-to-strong generalization (a weak supervisor training a stronger model).

**Reward function.** Three open problems: reward models that generalize to new questions and new decoding methods (otherwise, they write, RL "may merely stabilize the distribution" rather than improve capability); reward models that report their own uncertainty; and cheaper ways to build high-quality process reward models.

**Limitations the paper states.** DeepSeekMath is strong on quantitative reasoning but weaker than closed models at geometry and theorem proving; in a dry run it could not handle problems about triangles and ellipses, which the authors attribute to data selection bias. And unlike GPT-4, it gains little from few-shot examples: its zero-shot and few-shot scores are similar, which they put down to model scale.

**What the paper does and does not show about GRPO.** It shows that a critic-free PPO variant, with one update per sample batch and a KL term in the loss, added about 5 points to a strong SFT model on in-domain math and also improved every out-of-domain benchmark it measured. It does not compare GRPO with PPO at equal compute or memory, does not ablate $G$, $\beta$ or the std division, and reports its method comparisons (Fig. 5) as single training curves of a 1.3B model. The course picks up the story there: [L16](#/read/lecture_16) shows the group advantage becoming the default in DeepSeek R1 and later recipes, and the critiques (std division, length normalization) that followed.

::note deferred | What the paper leaves open (GRPO's biases, rule-based rewards, much longer chains of thought) is where [L16](#/read/lecture_16) continues: Dr. GRPO, R1-Zero's rule rewards, Kimi k1.5's length control.
