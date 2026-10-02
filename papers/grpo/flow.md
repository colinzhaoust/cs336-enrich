---
title: DeepSeekMath · GRPO, read through
minutes: 40
---
In early 2024 a DeepSeek team (Shao et al.) built a 7B open model that scored over 50% on competition mathematics, and introduced along the way the RL algorithm most reasoning models now use: GRPO, PPO without the value model. The paper has two halves, data and RL. After this read-through you can describe how they mined 120B math tokens from the web, compute GRPO's group advantage and KL term by hand, read every training method from SFT to GRPO as one gradient formula, and say what the paper's RL did and did not improve.

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

Two side findings get as much space as the headline. On the data side, starting from a *code* model beats starting from a general one, and arXiv papers, a standard ingredient of math corpora, did not help. On the RL side, the paper offers a "unified paradigm" that writes SFT, rejection sampling, DPO, PPO and GRPO as one gradient formula, and an explanation of why RL helped that is more modest than the headline suggests.

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

Size is easy to claim; quality has to be measured. The paper trains the same 1.3B general model (DeepSeek-LLM 1.3B) for 150B tokens on each of four math corpora and evaluates it few-shot on eight benchmarks. The setting is fixed across runs: AdamW ($\beta_1 = 0.9$, $\beta_2 = 0.95$, weight decay 0.1), a peak learning rate of $5.3\times10^{-4}$ reached after 2,000 warmup steps, dropped to 31.6% of the peak at 80% of training and to 10% at 90%, with 4M-token batches of 4K-token sequences. (The two drops are factors of $\sqrt{10}$ each: $10^{-1/2} = 0.316$.)

::figure grpo:table1 | read the columns down the rows: the arXiv-heavy MathPile barely beats no math training (and loses on GSM8K), and the DeepSeekMath Corpus row leads all eight columns

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

The same base model also writes Python to solve problems (66.9% on GSM8K+Python, 31.4% on MATH+Python) and drafts formal proofs in Isabelle (25.8% and 24.6% on miniF2F valid and test), both above Llemma 34B (Table 3). Math training lifted general reasoning too: MMLU rises from 49.1% to 54.9% and BBH from 55.2% to 59.5% over the code model it started from (Table 4).

::note slip | §2.3 says the code tokens let DeepSeekMath-Base "maintain" the code model's coding scores. Against the checkpoint it was actually trained from (taken just before learning-rate decay) that holds: HumanEval 40.2% → 40.9%, MBPP 52.6% → 52.6%. Against the released, fully decayed DeepSeek-Coder-Base-v1.5 it is a drop: 43.2% → 40.9% and 60.4% → 52.6%.

### Two lessons from the pre-training ablations

The paper's discussion (§5.1, run with an 89B-token iteration-2 version of the corpus and the 1.3B model) answers two questions that explain the choices above.

**Does code help math?** Train 400B tokens of code, then 150B of math, and compare with 400B general tokens then the same math. Code first ends best without tools (GSM8K 21.9% vs 19.1%, MATH 15.3% vs 14.4%) and with tools (GSM8K+Python 17.4% vs 14.3%). Mixing code and math in one stage keeps coding from being forgotten and gives the best tool-use scores (19.7% and 13.5%), but costs math without tools (17.6% and 12.1%); the authors guess a 1.3B model lacks the capacity for both at once. This is why the 7B model starts from a code model.

**Do arXiv papers help?** Training on arXiv alone, either MathPile or the 28.0B-token ArXiv-RedPajama, gives no notable gain or even a loss on every math benchmark, at 1.3B (150B tokens) and at 7B (40B tokens). At 7B, ArXiv-RedPajama moves miniF2F-test from 21.7% down to 11.9%. The authors scope it themselves: they did not test arXiv mixed with other data, at larger scale, or on tasks such as turning formal proofs back into prose.

## Instruction tuning: what does SFT add? {#sft}
source: grpo:§3.1 · grpo:§3.2 · Table 5

A base model continues text; it does not reliably answer a question in a fixed format. **Supervised fine-tuning (SFT)**, recapped in [L15](#/read/lecture_15), trains it on question-solution pairs. Here the 776K examples, English and Chinese, come in three solution formats:
- **chain-of-thought (CoT)**: the reasoning in prose, ending with the answer;
- **program-of-thought (PoT)**: a Python program whose output is the answer;
- **tool-integrated reasoning**: prose that calls Python for the computations and continues from the results.

The English part annotates GSM8K and MATH problems with tool-integrated solutions and adds subsets of MathInstruct and Lila-OOD; the Chinese part covers K-12 problems across 76 sub-topics. Training is short: examples are packed into 4K-token sequences, 500 steps at batch size 256 with a constant learning rate of $5\times10^{-5}$, so at most about 128,000 packed sequences.

The result, DeepSeekMath-Instruct 7B, scores 82.9% on GSM8K and 46.8% on MATH with chain-of-thought, and 57.4% on MATH with tools. On MATH without tools it beats every open model by at least 9 points (the closest, InternLM2-Math 20B, has 37.7%), including Qwen 72B and the PPO-trained WizardMath-v1.1 7B, but stays below GPT-4 (52.9%) and Gemini Ultra (53.2%).

::figure grpo:table5 | the chain-of-thought block: Instruct's 46.8% on MATH beats every open model but trails GPT-4 and Gemini Ultra; the RL row below it is where the next parts lead

These SFT numbers are the baseline every RL number in the paper is measured against.

## Why drop PPO's value model? {#ppo}
source: grpo:§4.1 · grpo:§4.1.1 · Eq. 1 · Eq. 2 · Fig. 4

RL after SFT works like this: give the model a question, let it write answers, score them, and make high-scoring answers more likely. The paper runs it with a learned **reward model** $r_\varphi$ that scores an answer (trained from DeepSeekMath-Base on data labelled by answer correctness). To see what GRPO changes, start from the gradient every such method uses, recapped from [L16](#/read/lecture_16):

$$ \nabla_\theta\, \mathbb{E}_{o\sim\pi_\theta(\cdot\mid q)}[R(o)] = \mathbb{E}_{o}\Big[R(o)\sum_{t=1}^{|o|}\nabla_\theta\log\pi_\theta(o_t\mid q,o_{<t})\Big] $$

Read it as weighted SFT: $\nabla\log\pi_\theta$ is the step that imitates the sampled answer $o$, and the reward is its weight. The estimate is noisy, so one subtracts a **baseline** $b$ from the reward. Any $b$ that does not depend on the sampled answer is free: $\mathbb{E}_o[b\,\nabla\log\pi_\theta(o)] = b\,\nabla\sum_o\pi_\theta(o) = b\,\nabla 1 = 0$. The expected direction stays the same and only the noise changes. The paper never states this identity, but its whole design rests on it.

::worked supp-policy-gradient
::widget fixture:grpo--baseline | slide b anywhere: the expected-gradient line stays flat while the one-sample noise changes, reaching zero at b = π(b); dividing by s only rescales the flat line

**PPO**, the standard RL algorithm for LLMs since InstructGPT ([L15's RLHF](#/read/lecture_15)), maximizes a clipped surrogate. Answers $o$ are sampled from the policy as it stood before the update, $\pi_{\theta_{old}}$, and each token is reweighted by the probability ratio:

$$ \mathcal{J}_{PPO}(\theta) = \mathbb{E}_{q,\;o\sim\pi_{\theta_{old}}}\ \frac{1}{|o|}\sum_{t=1}^{|o|}\min\Big[\rho_t A_t,\ \operatorname{clip}(\rho_t, 1-\varepsilon, 1+\varepsilon)\,A_t\Big],\qquad \rho_t = \frac{\pi_\theta(o_t\mid q,o_{<t})}{\pi_{\theta_{old}}(o_t\mid q,o_{<t})} \qquad (1) $$

The clip stops a token's probability from moving more than a factor $1\pm\varepsilon$ in the direction its advantage favours, which makes it safe to take several gradient steps on one batch of samples. The **advantage** $A_t$, how much better token $t$ turned out than expected, comes from **GAE** (generalized advantage estimation) over the rewards and a learned **value function** $V_\psi$, a second network that predicts at every token the reward still to come. To keep the policy from over-optimizing the reward model, PPO for LLMs also folds a KL penalty against a frozen **reference model** $\pi_{ref}$ (usually the SFT model) into the reward of every token:

$$ r_t = r_\varphi(q, o_{\le t}) - \beta\log\frac{\pi_\theta(o_t\mid q,o_{<t})}{\pi_{ref}(o_t\mid q,o_{<t})} \qquad (2) $$

The paper's objection is to $V_\psi$, for two reasons. It is "typically another model of comparable size as the policy" (§4.1.1), so it doubles the trained weights, gradients and optimizer state. And it must be accurate at *every* token, although the reward model scores only the last one; learning a per-token value from one number at the end is hard.

::figure grpo:fig4 | top, PPO: policy, reference, reward and value models, with GAE combining reward and value into advantages; bottom, GRPO: the value model is gone, the policy samples a group o₁…o_G, and "group computation" turns the G rewards into advantages

GRPO's replacement uses something cheap in language modelling: you can sample the same question many times. For each question it samples a group of $G$ answers and uses their *average reward* as the baseline. The question is shared by all $G$ answers, so the baseline depends on the question and not on any one answer's luck: the identity above says it does not bias the direction (up to one detail in the next part).

::predict ppo-value-model-cost

Counting models makes the saving concrete. PPO holds four 7B-class networks: the trained policy and critic, the frozen reference and reward models. GRPO holds three. If a trained network costs 16 bytes per parameter (weights, gradients, Adam states) and a frozen one 2, PPO needs $112 + 112 + 14 + 14 = 252$ GB of model state for 7B models, GRPO $140$ GB. The 112 GB saved is exactly the critic; the reference model stays because GRPO keeps a KL term.

::note warning | The paper quantifies neither reason: there is no memory measurement and no PPO-versus-GRPO comparison at equal compute anywhere in it. GRPO also *samples more* than typical PPO (64 answers per question in the recipe below), so its saving is in memory and critic passes, not in generation.
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
