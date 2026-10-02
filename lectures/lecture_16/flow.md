---
title: L16 · RL from verifiable rewards, read through
minutes: 50
---
Lecture 15 took a pretrained model to roughly GPT-3.5 with SFT and RLHF. This lecture covers the step to o1- and R1-style reasoning models: reinforcement learning against rewards a program can check. The first half rebuilds the algorithms (policy gradient, PPO, GRPO) and asks what GRPO's objective actually optimizes. The second half reads four recipes from the field: DeepSeek R1, Kimi k1.5, Qwen 3 and Qwen3-Coder-Next. After it you can write GRPO's advantage by hand, explain why reasoning chains grow longer during training, and say what a verifiable reward does and does not protect you from.

## Why RL with verifiable rewards? {#why-rlvr}
source: lecture_16.pdf p2-p4 · video 0:05-3:46

::slide 3 | top: two overoptimization plots from lecture 15 (left, RM score against KL from the initial policy, the gold curves bending down for every reward-model size; right, eval win-rate against proxy reward for expert iteration, best-of-n and PPO); bottom: AlphaGo, a protein structure, and the Lean logo, domains "where we optimize exactly what we want"

The course so far has built a ChatGPT-like model: pretraining, instruction tuning, and RLHF ([L15](#/read/lecture_15)). What is missing is the recent jump to *thinking models*, which write very long chains of thought (CoT) and solve hard problems in mathematics and coding. The professor opens with news he had no time to put on a slide: OpenAI had just announced that one of its thinking models solved a major open math problem. "How does that thing work? That's exactly this thinking model stuff or RLVR stuff."

Lecture 15 ended on what the professor calls "a downer note". RLHF trains a reward model on human preferences and then runs RL against it. The reward model is a learned proxy, and you cannot keep pouring compute into optimizing the same proxy: "eventually you're going to overfit your reward model", no matter how well you regularize. The top-left plot shows the symptom. As the policy moves away from where it started (more KL), the proxy score keeps climbing while the true ("gold") score rises, peaks, and falls. RLHF is also annotation-bottlenecked, because a better reward model needs more human labels.

Compare AlphaGo. There the reward is the win-loss condition of Go, which has no "sloppiness" in its definition, so "you can just put in as much compute as you want", and as long as the objective improves you are doing well. The professor frames RLHF as a learning problem and Go as closer to a search problem ("not quite a precise distinction"). The question for this lecture is whether some language tasks have Go's property. Formal mathematics (Lean, on the slide), ordinary math with a checkable final answer, and code with unit tests all do. This is **RLVR, reinforcement learning from verifiable rewards**: the reward is computed by a checker, so it *is* the objective rather than a model of it.

One sentence frames everything that follows: "the algorithms aren't going to be that different fundamentally, but where we will end up will actually be surprisingly different." The optimizer stays in the PPO family. The reward changes.

::predict verifiable-reward-vs-rlhf

The lecture has two parts (slide 4). Part 1 covers the core algorithms, PPO and then GRPO and its variants, which are also what you implement in the assignment. Part 2 reads open technical reports to see those algorithms in production.

::note aside 2:46 | Keep the word "verifiable" in mind. The lecture undercuts it twice later: Kimi's math checker turns out to be a learned reward model, and a coding agent and a Lean checker both get hacked. The recap's version is the careful one: RLVR is RLHF with "more unhackable rewards".
::kp verifiable-reward-vs-rlhf

## PPO, once more: from policy gradient to clipped ratios {#ppo-theory}
source: lecture_16.pdf p5-p8 · video 3:46-7:51

::slide 5 | three attempts: (1) the policy gradient ∇θ E[R(z)] = E[R(z) ∇θ log pθ(z)], "variances are too high"; (2) TRPO, maximize the ratio-weighted advantage subject to a KL ≤ δ constraint; (3) PPO, the min of the ratio-weighted advantage and its clipped version

PPO came up in lecture 15, but the professor redoes it because "PPO is confusing enough that I think you will benefit from doing it twice." Everything starts from the first line of the slide, which he calls "the core from which everything else is derived".

**Attempt 1, policy gradient.** A language model is a policy $p_\theta$ over whole outputs $z$. We want to raise the expected reward $\mathbb{E}_{p_\theta}[R(z)]$. The reward is a checker's verdict, not differentiable in $\theta$, but the gradient does not need it to be:

$$ \nabla_\theta\, \mathbb{E}_{z\sim p_\theta}[R(z)] = \mathbb{E}_{z\sim p_\theta}\big[R(z)\,\nabla_\theta \log p_\theta(z)\big] $$

Read the right side as SFT with weights. $\nabla_\theta \log p_\theta(z)$ is exactly the gradient you would take to imitate $z$; the reward multiplies it. In the professor's words, we are "taking essentially weighted SFT updates. The weights might be positive or negative." A sampled output with high reward is imitated more; with a negative weight it is pushed away. This is the REINFORCE trick, and every algorithm in the lecture is a choice of what weight to put in front of $\nabla \log p$.

::worked policy-gradient-variance

The slide's complaint is variance. One scalar reward multiplies the score of a whole sampled sequence, so the estimate swings with which sequence happened to be drawn. Aloud, the professor stresses a second problem: each gradient step needs fresh samples from the current policy. "Can we reuse our rollouts?"

**Attempt 2, TRPO** answers that by maximizing an importance-weighted objective, the probability ratio $\pi_\theta/\pi_{\theta_{old}}$ times the advantage, subject to a constraint that the new policy stays within KL distance $\delta$ of the old one (written on the slide image). He skips it as covered last time.

**Attempt 3, PPO** replaces the constraint with a clip. With ratio $r = \pi_\theta(a|s)/\pi_{\theta_k}(a|s)$ and advantage $A$:

$$ L = \min\big(r\,A,\ \operatorname{clip}(r,\,1-\epsilon,\,1+\epsilon)\,A\big) $$

The **advantage** is how much better an action was than expected; for now think of it as the reward minus a baseline. The min makes the clip one-sided, which the slide does not spell out. If $A>0$, raising the action's probability helps until $r$ reaches $1+\epsilon$; beyond that the clipped term is the smaller one, it is a constant, and its gradient is zero. If $A<0$, the flat region sits below $1-\epsilon$. So a batch can move each action's probability only so far in the direction its advantage favours, which is what makes it safe to take several gradient steps on the same rollouts.

::widget fixture:ratio-clip | with A > 0 the objective goes flat for r > 1 + ε while the faint unclipped line keeps rising; drag A below zero and the flat segment jumps to r < 1 − ε
::predict ppo-ratio-clip

::note slip 6:34 | Reading the pseudocode, the professor says "I clip the advantage in this slightly strange but OK way." PPO clips the probability ratio, not the advantage, as slide 5 says ("clip the ratios at some eps") and as he says himself at 9:26 ("computing the clipping ratios").

::slide 7 | Spinning Up's PPO-Clip pseudocode: collect trajectories, compute rewards-to-go, estimate advantages with the current value function, maximize the clipped objective with Adam, then fit the value function by mean-squared-error regression

PPO is the RL workhorse that OpenAI used for simulated robots walking (2017) and for its Dota-playing OpenAI Five (2019), slide 6. At the level of Spinning Up's pseudocode it looks easy: sample trajectories, compute advantages with any method, update the policy with the clipped objective, fit a **value function** $V_\phi(s)$ (a second network predicting expected reward from a state) by regression. "I could implement this in one go."

::slide 8 | a pile of papers and tables, topped by "Implementation Matters in Deep Policy Gradients: A Case Study on PPO and TRPO"; caption: "We need to look at a live implementation when talking about PPO"

Then you see a blog post titled "The 37 implementation details of PPO", which "should strike fear into your heart". Different libraries give very different numbers, many implementations are wrong, and there are papers showing that the baselines some people use in PPO "aren't even baselines at all. They fundamentally change the optimization problem." So the lecture looks at a real implementation next.
::kp policy-gradient-variance
::kp ppo-ratio-clip

## PPO for language models, in practice {#ppo-practice}
source: lecture_16.pdf p9-p16 · video 7:51-11:19

::slide 9 | Zheng et al.'s diagram of PPO for language models: the policy LM samples a response to a user query; a reward model scores the full response; a value model predicts V(s_t) per token; GAE combines them into advantages (the box writes out the TD error δ_t = r(s_t, a_t) + γV(s_{t+1}) − V(s_t) and the return); an experience buffer feeds the PPO-clip loss, an MSE loss for the value model, and a KL term to the SFT model

The language-model version maps RL onto text. The **state** $s_t$ is the prompt plus the tokens generated so far, the **action** $a_t$ is the next token, and the reward model scores only the finished sequence. The professor walks the boxes: an advantage estimator in the middle, an experience buffer "because you're going to keep some of the old stuff", a value model trained alongside the policy and used inside the advantage calculation. "Notice how the green box appears twice."

One detail turns a simple setup into a hard one. The KL term to the reference model operates token by token, "so it's not actually just a bandit problem. It's like a whole multi-step RL problem." A **bandit** is RL with a single decision and a single reward; that is what the problem would be if the only reward were the one at the end.

::slide 12 | AlpacaFarm's compute_loss: a clipped value loss, then ratio = exp(logprob − old_logprob), pg_losses = −advantages·ratio and pg_losses2 = −advantages·clamp(ratio, 1 − cliprange, 1 + cliprange), pg_loss = max of the two; loss = pg_loss + vf_coef·vf_loss; Cliprange = 0.2

The reference implementation is AlpacaFarm's PPO, written by the professor's students for an RLHF project; it "took a long time to get working". The outer loop (slide 11) is ordinary: for several PPO epochs over a fixed set of rollouts, compute the loss, clip gradient norms, step. The loss (slide 12) "follows almost exactly the PPO update". In the code the objective is negated into a loss, so the min of the objective becomes a `torch.maximum` of the two negated terms, with clip range $\epsilon = 0.2$. The value function's loss is clipped too, around its old predictions. The messy part is elsewhere.

::slide 14 | reward shaping: "add per-token KL penalty, last-token full reward"; in practice "clip KL for sequences where new policy logp < reference logp"; the code computes kl = torch.clamp(logprobs − ref_logprobs, min=0.0), non_score_rewards = −kl_ctl.value·kl, and adds the task reward at each sequence's last non-pad position

**Reward shaping** turns the KL regularizer into rewards. Every token gets a reward of $-\beta \log\frac{\pi(y_t)}{\pi_{\text{ref}}(y_t)}$, and the last token additionally gets the full task reward, since the reward model only scores the finished sequence. In the code (slide image) $\beta$ is an adaptive coefficient `kl_ctl.value`, and the per-token log-ratio is clamped at zero from below, so tokens where the policy is *less* likely than the reference contribute no penalty. (The slide's text says "sequences", but the clamp is applied token by token.)

::worked per-token-kl-reward-shaping

The professor's verdict on that clamp is blunt. A KL divergence is never negative, but its per-token sample estimate is a sum of positive and negative log-ratios, and cutting off the negative ones "totally ruins the point of a KL divergence". Yet "if you remove this, it blows up immediately." He shows it as an example of the hacks PPO accumulates, not as advice: "This is not to say that this is the way you should implement PPO." The code also carries a comment admitting a small off-by-one bug when the pad token equals the end-of-sequence token, which makes the same point.

::note skip 9:40 | The rollout code (slide 13: sample, score with the reward model after re-tokenizing, shape the rewards, estimate advantages over the whole batch) is skipped aloud.

::slide 15 | generalized advantage estimation: Â_t = Σ_l (γλ)^l δ_{t+l} with δ_t = r_t + γV(s_{t+1}) − V(s_t), the AlpacaFarm loop that computes it backwards over the response, and the note "Funny detail – this is a bandit problem and gamma=lambda=1 works – this is the reward-to-go vs the value"

PPO estimates advantages with **GAE (generalized advantage estimation)**. Each token's TD error $\delta_t$ is what that step earned beyond what the value model expected, and GAE sums later TD errors with weights $(\gamma\lambda)^l$; the discount $\gamma$ and the mixing factor $\lambda$ trade bias against variance over a long chain of decisions. In practice people often set $\gamma = \lambda = 1$. Then the value terms telescope away and each token's advantage is simply the reward-to-go minus the value at that token. With one reward at the end, that is the same $R$ for every position minus a per-position baseline.

::worked language-rl-is-a-bandit

The slide calls this a funny detail that works. The professor puts it more sharply: $\gamma = \lambda = 1$ is "a degenerate setting that turns this back into a bandit problem. So you've thrown away a lot of the structure that you get from PPO." To see why nobody wants $\gamma < 1$ here, take a 300-token answer with one terminal reward and $\gamma = 0.99$: the first token sees the reward discounted by $0.99^{299} = e^{299\ln 0.99} \approx e^{-3.0} \approx 0.05$, so 95% of the credit is gone before it reaches the start of the answer.

::predict language-rl-is-a-bandit

::slide 16 | three AlpacaFarm training curves for two 7B runs over about 380 steps: objective/kl_sum_seq rising to about 50 (green) and 13 (blue); objective/rewards (the reward-model score) rising from about 2.3 to about 3.8 and 3.2; objective/non_score_rewards (the KL penalty) falling from 0 to about −1 and −1.4

After all that engineering, "you will get what you expect". Reward-model reward goes up, and the negative-KL reward goes down as the policy drifts from the reference. The numbers above are read off the slide image. Note that the left panel is captioned "Increasing overall rewards" but its plot title is the summed KL, which rises as the policy drifts. The professor's takeaway: you should see curves like these in the assignment, and getting there takes hacks.
::kp language-rl-is-a-bandit
::kp per-token-kl-reward-shaping

## GRPO: PPO without the value model {#grpo}
source: lecture_16.pdf p17-p21 · video 11:19-20:05

::slide 17 | why not PPO: complicated implementation; the value model is memory-hungry and needs extra tuning. Why not DPO: the data are not inherently pairwise Bradley-Terry comparisons; DPO is offline, "though could be made online by iterating"

Why another algorithm? The professor's case against PPO has two parts. The implementation is painful and "can require hacks to stabilize". He scopes that: labs have "very turnkey solutions" for PPO at scale, but for researchers implementing it from scratch it is finicky. The second part applies to everyone: the value model is "as big as the original model", memory "you would rather be using for other stuff like models or inference servers", and it needs its own tuning.

Why not DPO, which lecture 15 made look so convenient ([L15's DPO](#/read/lecture_15))? DPO is "a very specific solution to a very specific problem": pairwise preferences in Bradley-Terry form. A math problem gives a scalar verdict per answer, not a comparison between two answers. There are DPO variants that break the pairwise structure, "but really you're just using the wrong hammer for the job"; PPO is "the more general hammer". The second objection, that DPO is offline, he calls "very overstated because it can be made online by just iterating DPO repeatedly."

::predict why-grpo-not-ppo-or-dpo
::kp why-grpo-not-ppo-or-dpo

::slide 18 | GRPO as written in the DeepSeek paper: for each question q sample G outputs o_1..o_G from π_old; maximize (1/G) Σ_i [min(ratio·A_i, clip(ratio, 1−ε, 1+ε)·A_i) − β·D_KL(π_θ‖π_ref)], with the KL estimated as π_ref/π_θ − log(π_ref/π_θ) − 1 and A_i = (r_i − mean(r))/std(r); PPO's objective alongside "for reference"; footer: "In the online case (rollout + immediate update), this is just policy gradient with group normalized rewards"

**GRPO (group relative policy optimization)**, from the DeepSeekMath paper ([the GRPO paper read-through](#/read/grpo)), keeps PPO's objective and changes "arguably the most complicated and annoying part": the value function. The value function exists to be subtracted as a baseline, which lowers the variance of the gradient. But it is a whole neural network and it "destabilizes training". Dropping it and going back to vanilla REINFORCE would bring the variance back. GRPO needs a baseline that costs no network.

Its answer uses the fact that you can sample one prompt many times. A value model would say "I predicted a score of 5, I got 6, so this was a good rollout." GRPO instead samples, say, 10 other rollouts of the same prompt and asks "how good was I compared to my 10 other rollouts". Concretely, for a group of $G$ responses with rewards $r_1,\dots,r_G$, every token of response $i$ gets the advantage

$$ A_i = \frac{r_i - \operatorname{mean}(r_1,\dots,r_G)}{\operatorname{std}(r_1,\dots,r_G)} $$

a z-score within the group. The rest is PPO: the clipped ratio, plus a KL penalty to a reference model. The KL is computed with the particular estimator on the slide, which "is not particularly important" to the professor ([the GRPO thread](#/read/grpo) covers why that estimator is always non-negative).

::widget fixture:grpo--kl-estimators | the slide's KL estimator π_ref/π − log(π_ref/π) − 1 is never negative for any single sample, unlike the plain log-ratio whose clamping slide 14 had to hack around

The footer is worth unpacking. In the **online** case you sample, take one gradient step, and sample again. On that one step $\pi_\theta = \pi_{\theta_{old}}$, so every ratio is 1, the clip "never does anything", and the min reduces to $\min(A_i, A_i)$. What is left is "just advantage minus a KL penalty": a policy gradient whose weights are group-normalized rewards.

::worked grpo-group-zscore-advantage
::widget fixture:lecture_16--group-weights | try the "all solved (8/8)" preset: every response is correct, yet both baselined rows are all zeros, because a group with no contrast gives GRPO nothing to learn
::video 14:40-16:31 | 'how good was I compared to my 10 other rollouts', and why the clip disappears in the online case
::predict grpo-group-zscore-advantage

::slide 19 | "You can (and people do) write tiny GRPO implementations": compute reward for each rollout, mean/var normalization per group, compute KL term, gradient updates on the loss; nano-aha-moment's compute_pg_loss: kl_penalty = exp(ref_logps − logps) − (ref_logps − logps) − 1, policy_loss = −logps·advantages, loss = (policy_loss + KL_COEFFICIENT·kl_penalty).sum() / total_response_len

Without a value function, GRPO fits on a slide. The example is nano-aha-moment from McGill. Read off the code: the policy loss is just $-\log p \times A$, with no ratio and no clip, because this implementation is online. The KL penalty is the slide-18 estimator, per token. The sum is divided by the total number of response tokens in the batch. The professor's one warning for the assignment: written with autodiff, "you will have to do a stop grad somewhere".

::slide 20 | nano-aha-moment's advantage code: groups of GENERATIONS_PER_SAMPLE indices (the comment's example is [[0,1,2],[3,4,5],[6,7,8]]), rewards = np.array(rewards), response_advantages = (rewards − rewards.mean()) / (rewards.std() + 1e-4), and each response's advantage repeated for every one of its tokens

The advantage is one line. The only change from the paper is the `1e-4` added to the standard deviation. The professor explains what the slide does not: it keeps the division from blowing up "when you only have a single sample, or if your samples happen to have the exact same rewards", which happens all the time in math, where a failed problem gives every sample exactly 0. (NumPy's `.std()` is the population standard deviation, dividing by $G$; `torch.std` divides by $G-1$ by default, which the widget's checkbox shows.)

::slide 21 | DeepSeekMath's Figure 5: DeepSeekMath-Instruct 1.3B trained further by RFT, Online RFT, GRPO+OS and GRPO+PS for about 9,000 steps; on GSM8K accuracy rises from about 56.5% to about 60% (RFT), 62% (online RFT), 64% (GRPO+OS) and 65% (GRPO+PS); on MATH from about 26.5% to about 28% for RFT against about 30% for GRPO

Does it work? In DeepSeekMath's plot, the two GRPO curves (yellow, outcome supervision; blue, process supervision) sit above **RFT, rejection-sampling fine-tuning**: "taking the correct answers that your model generates and training on them and throwing away everything else", which you also implement as a baseline. The numbers are read off the slide image. **Process supervision** (grading the intermediate steps, not just the final answer) gives the blue line a small extra gain. "We will get back to this later": R1 drops it.

::note deferred 19:47 | Process supervision is "one of the big design decisions for these kinds of RL problems". It returns in the R1 section, where outcome rewards win.
::kp grpo-group-zscore-advantage

## Is GRPO's advantage a valid baseline? {#baselines}
source: lecture_16.pdf p22-p24 · video 20:05-25:56

::slide 22 | top: GRPO's advantage A_i = (r_i − mean)/std; bottom: Sutton and Barto §13.4, REINFORCE with baseline: ∇J ∝ Σ_s μ(s) Σ_a (q(s, a) − b(s)) ∇π(a|s), valid for any b(s) that does not vary with a, because Σ_a b(s)∇π(a|s) = b(s)∇1 = 0

Now the conceptual question: "Are we actually taking policy gradients?" GRPO replaced PPO's value-based advantage with a z-score. Is that a good advantage?

The rule comes from Sutton and Barto. In a policy gradient you may subtract from the reward any **baseline** $b(s)$ that depends only on the state, not on the action taken. In the bandit view of language RL, "it's just the prompt". The reason is the one-line identity on the slide: summed over actions, the baseline's contribution is $b(s)\nabla\sum_a \pi(a|s) = b(s)\nabla 1 = 0$. So any prompt-dependent baseline leaves the expected gradient unchanged; the choice of $b$ changes only the variance.

::worked baseline-validity-std-division

The group mean is such a baseline, up to a detail: it includes the sample's own reward. The last step of the derivation above shows that this only rescales each weight by $(G-1)/G$ relative to the **leave-one-out** baseline (the mean of the *other* $G-1$ rewards), which does not depend on the sample at all.

Dividing by the standard deviation is different. $\sigma(x)$ is one number for the whole prompt, so it multiplies that prompt's gradient instead of being subtracted from it. Across prompts, each one gets a different weight $1/\sigma(x)$, and the summed update is a reweighted gradient rather than the gradient of expected reward. In the professor's words, GRPO "divides by the standard deviation, which breaks this baseline contract." He is careful about what that means: "it's in some ways a problem. In other ways, it's not." If you want an algorithm that does "what's written on the tin", that actually descends the reward, GRPO is not it.

::widget fixture:lecture_16--group-weights | press "one success in 8" and then "half solved": the GRPO row is the mean-baseline row times one factor, 3.02 for the near-uniform group against 2.0 for the balanced one, a rescaling rather than a subtracted term
::predict baseline-validity-std-division

::slide 23 | GRPO's objective with two terms in red, the per-response 1/|o_i| in front of the token sum and the std in the advantage's denominator; Dr. GRPO ("GRPO Done Right, without bias") with both removed, Â = R(q, o_i) − mean; right: a "token efficiency" plot where Dr. GRPO reaches reward about 0.6 at output lengths around 500 tokens while GRPO drifts out to 800-1,000; footer: "this gets pretty close to reinforce w/ leave-one-out"

There is a second deviation that the professor "didn't mention earlier". GRPO averages each response's token terms over that response's own length, $\frac{1}{|o_i|}\sum_t$, before averaging over the group. A first-principles derivation from the policy-gradient and baseline theorems has neither this length normalizer nor the std division. A paper soon after GRPO, Dr. GRPO (Liu et al. 2025), removed both: on the slide its objective simply sums the token terms (equivalently, divides by a constant) and uses $r_i - \text{mean}$ as the advantage. That is close to REINFORCE with a leave-one-out baseline (often called RLOO). The token-efficiency plot shows the effect, read off the image: the same reward is reached with far shorter outputs.

### What the two terms do

::slide 24 | "Stdev – upweights too easy or hard questions"; Dr. GRPO's paragraph on response-level length bias; five plots over 150 policy steps: reward (both reach about 0.6), output length (GRPO climbs from about 350 to over 1,000 tokens, Dr. GRPO levels off near 500), length of correct answers (both rise from about 200 to about 420), length of incorrect answers (GRPO from about 1,000 to about 1,900, Dr. GRPO back near 1,000), average benchmark score (both about 22-25%)

**The length normalizer** is the easier one. A response's advantage is spread over its own tokens, so each token's weight is $\hat A_i/|o_i|$. For a wrong answer ($\hat A<0$), a longer answer is punished less per token. The professor's extreme case: if I know I will get a proof wrong and take a reward of −1, "I'm just going to generate an infinitely long string. If I do that, I'll get to divide by infinity." Less extremely, the objective teaches a model "to blab on once it realizes that it cannot actually solve the problem." For a right answer the same division favours short responses.

::video 23:46-24:49 | the 'divide by infinity' argument, then the length capping off once the normalizer is removed
::worked grpo-length-bias
::widget fixture:lecture_16--length-lens | press "two wrong, 10× length apart": under GRPO the 1,500-token wrong answer gets one tenth of the 150-token one's per-token push; switch to Dr. GRPO and both sit on one flat line

The fix changes behaviour exactly where the argument says it should. In the slide's plots, reward and benchmark scores are the same with and without it, but GRPO's output length keeps climbing while Dr. GRPO's levels off, and the gap sits in the *incorrect* answers. Thinking length that seemed to grow "forever", the professor says, "turns out to just cap off at a constant". And "especially on the incorrect cases, you really don't want to be generating longer and longer outputs."

::note slip 23:55 | Introducing the length effect, the professor says it will "encourage the model to generate long outputs because-- or sorry when you're wrong it will encourage you to generate wrong outputs". He means long outputs when the answer is wrong, as his next sentences show.

**The std division** emphasizes prompts whose rewards barely vary. For a binary reward with success rate $p$ the group's std is $\sqrt{p(1-p)}$, which is small near $p=0$ and $p=1$: questions that are too easy or too hard. The professor says an always-solved question divides by "basically 0". Strictly, an exactly uniform group has a zero numerator too, so its advantage is 0 (the `1e-4` keeps it finite). The upweighting bites on *nearly* uniform groups, such as 1 success in 8, whose weights are multiplied by about 3.02 against 2.0 for a half-solved group. Upweighting both ends "seems like clearly a thing that maybe we don't want", because a model learns most from problems "within its solvability range".

::predict grpo-length-bias
::kp baseline-validity-std-division
::kp grpo-length-bias

## DeepSeek R1-Zero: the clean experiment {#r1-zero}
source: lecture_16.pdf p25-p30 · video 25:56-31:38

That is the whole algorithmic toolkit. The second half reads three open model reports (slide 25): DeepSeek R1, "central to many recent RLVR efforts"; Kimi k1.5, released at the same time with complementary details; and Qwen 3, the most recent, with RL on very little data. A fourth, Qwen3-Coder-Next, is new this year and covers agentic RL.

::slide 26 | search interest in "DeepSeek-R1", flat until the paper's submission on 22 January 2025 and then a spike to 100; what is remarkable about R1: performance exceeding OpenAI o1, an open and simple RL recipe that "ended speculations on the necessity of MCTS/PRMs", and SFT insights from R1-Zero and the distilled models

R1 was "a bit of a social phenomenon", and the professor thinks everyone should know it. It was the first open model to match OpenAI o1: very long chains of thought, clearly trained with RL, strong on hard math. It came with a recipe anyone could run. "If your solution was like you have this horrible PPO thing that no one but DeepSeek can run", that matters much less to researchers than GRPO, which "even you in your assignments" can play with. And its distillation results have held up.

::slide 27 | the DeepSeekMath GRPO plot again (GRPO+PS on top), with "But they do not use process supervision in R1"

R1 builds on DeepSeekMath's GRPO with one big change: it abandons process supervision and uses only **outcome supervision**. Outcome supervision rewards only whether the final answer is correct; process supervision uses something like "a grading rubric ... to check the validity of intermediate steps in a proof". Many people thought process supervision was important. "Turned out it wasn't critical."

::slide 28 | R1-Zero's setup: rewards = accuracy ("is it correct?") + format ("use thinking tags"); data not public; base model DeepSeek-V3; results "a bit worse than OpenAI o1", with a table: AIME 2024 pass@1 71.0 (o1-0912 74.4), cons@64 86.7 (83.3), MATH-500 95.9 (94.8), GPQA Diamond 73.3 (77.3), LiveCodeBench 50.0 (63.4), Codeforces rating 1444 (1843)

**R1-Zero** is the controlled version: start from the DeepSeek-V3 base model (which, as the professor notes, has itself been mid-trained and can follow some instructions) and run GRPO with two rewards. The **accuracy reward** checks whether the math answer is correct. The **format reward** checks that the model wraps its reasoning in thinking tags, which matters because the tags "allow them to strip out the chain of thought later". No process rewards, no search.

The result is "only a little bit worse than OpenAI o1". The slide image's table is more mixed than that. R1-Zero beats o1-0912 on AIME with majority voting over 64 samples (86.7 vs 83.3) and on MATH-500 (95.9 vs 94.8), and is clearly behind on code (LiveCodeBench 50.0 vs 63.4, Codeforces rating 1444 vs 1843).

The professor likes the result because it has "none of the messes of a real production post-training pipeline": no RLHF or other stage to credit, just "a very simple base model plus GRPO". The assignment replicates a version of it.

::predict r1-zero-recipe

::slide 29 | left: DeepSeek-R1-Zero's average response length during training, rising roughly linearly from a few hundred tokens to about 10,000 over about 8,000 steps; right: a training transcript where the model, mid-solution, writes "Wait, wait. Wait. That's an aha moment I can flag here." and re-checks its work

R1 also took off because of two phenomena the paper highlighted. Response length grows steadily during training, from a few hundred tokens to roughly ten thousand (read off the slide image), as if the model learned to think longer. And an "aha moment" went viral: a CoT in which the model stops itself and re-evaluates.

::slide 30 | "But maybe a bit overstated": Dr. GRPO's biased and unbiased objectives and token-efficiency plot under "Length due to biased objective?"; under "Base model already has 'aha'", a base model's response to a trigonometry problem containing "Aha! I can use this to get ..."

The professor is unimpressed by both. "We now know that longer CoTs is arguably a natural side effect of the length normalization of the GRPO algorithm": exactly the bias from the previous section. And Dr. GRPO showed the "aha" phrasing already appears in the base model before any RL, so "clearly, it can't just be a result of the RL algorithm." People write "aha" when solving math problems, the model learned it in pretraining, and RL on lots of math tokens draws it out. Neither curve is clean evidence that RL created a new reasoning behaviour. What R1 did show is "just how simple RLVR could be".

::widget fixture:lecture_16--length-drift | switch the objective to Dr. GRPO: the push the length normalizer puts on incorrect responses disappears and the mean-length line goes flat, so a rising length curve alone cannot tell you whether reasoning got longer for a good reason
::predict r1-length-growth-aha-evidence
::kp r1-zero-recipe
::kp r1-length-growth-aha-evidence

## DeepSeek R1: the production pipeline {#r1}
source: lecture_16.pdf p31-p38 · video 31:38-40:00

::slide 31 | R1 vs R1-Zero, key differences: SFT initialization, a language-consistency reward for the CoT, non-verifiable rewards in stage 2; pipeline DeepSeek-V3 → Reasoning SFT → RL (GRPO) → SFT/RLHF

R1-Zero was the clean experiment; R1 is the attempt to productionize it, and it shows how all the course's pieces stack into one system. The usual order: take the mid-trained model, do reasoning training (perhaps with long-context extension somewhere), and run RLHF at the end, because the final model is "the most user-facing object" and RLHF is where formatting and style get fixed. R1 differs from R1-Zero in three ways: it starts from a reasoning SFT, adds a language-consistency reward, and later blends in non-verifiable rewards.

::slide 32 | the R1 paper's paragraph on cold-start data: to avoid the unstable early phase of RL from the base model, "construct and collect a small amount of long CoT data" by few-shot prompting with long CoTs, prompting for answers with reflection and verification, gathering readable R1-Zero outputs, and post-processing by human annotators; "Claimed benefit: interpretability"; "Origins on the data not quite clear"

R1-Zero used no SFT at all. R1 first fine-tunes on long CoT data, and the professor enjoys reading between the lines: when a report says it will "construct and collect a small amount of long CoT data", "I wonder if that was distilled from some other models." For a very good base model, SFT on long CoTs alone unlocks much of o1-style behaviour, which makes it a great starting point for RL.

::slide 33 | "Even a small number of samples is effective for bootstrapping reasoning from LMs": s1's table and plot; s1-32B, fine-tuned on 1K examples, scores AIME 2024 56.7, MATH-500 93.0, GPQA 59.6, against r1-distill's 72.6 / 94.3 / 62.1 with 800K examples and o1's 74.4 / 94.8 / 77.3; caption "1k math and science questions + long CoTs from Gemini / r1"

How little data is enough? The slide uses s1 (named on the slide image): 1,000 math and science questions with long CoTs written by Gemini or R1 bring a 32B model to 93.0 on MATH-500, close to R1-distill's 94.3 trained on 800 times as many examples. The professor draws a broader point. Several papers, including some from his students, show that with the right base model and distillation data "you can basically get a lot of the long COT reasoning juice just from SFT". So an open question is "do you really need RL for some of this". His working answer: RL is "a great source of supervision". For frontier problems no one has written long CoTs, and RL lets the model generate its own; once they exist, others can learn them by imitation.

::slide 34 | "The RL part is basically the same", plus a language-consistency reward; the R1 paper's paragraph: CoTs mix languages when prompts span several languages; the reward is the proportion of target-language words in the CoT, summed with the accuracy reward; ablations show a slight performance drop, accepted for readability

The RL stage is R1-Zero's GRPO with one extra reward. R1-Zero-style training would switch languages inside its CoT, which DeepSeek found "very uninterpretable and slightly disturbing". The fix, per the paper excerpt on the slide image, is a reward equal to the fraction of CoT words in the target language, added to the accuracy reward, at a slight cost in performance. The professor adds that it was done "just for interpretability reasons".

::slide 35 | the usual post-training after reasoning RL: an SFT step for 2 epochs on reasoning data (non-verifiable tasks such as "write a proof of X", judged by V3, 600k) and non-reasoning data (the V3 SFT set, 200k); an RLHF step re-using R1-Zero-style reasoning RL plus V3's RLHF pipeline for non-verifiable tasks, "still uses GRPO (for RLHF)"

Then comes the post-training you met in lecture 15: SFT, then RLHF. For non-verifiable tasks they reuse DeepSeek-V3's pipeline, so "there's really no surprises at all here." Asked at the end of class how reasoning and non-reasoning data are divided, the professor points to exactly this split: reasoning problems go into the reasoning-RL stage, and chattiness and other non-reasoning behaviour go into the final RLHF stage.

::worked r1-stage-order-and-distillation

::slide 36 | R1's benchmark table: a mixture-of-experts model with 37B activated and 671B total parameters; AIME 2024 79.8 (o1-1217 79.2), MATH-500 97.3 (96.4), Codeforces rating 2029 (2061), GPQA Diamond 71.5 (75.7), SWE-bench Verified 49.2 (48.9)

"It's really, really good", which is why R1 caused such a stir. On the slide image's table it edges o1-1217 on AIME and MATH-500, roughly ties on SWE-bench Verified, and trails on GPQA and Codeforces. It also reproduced the test-time scaling behaviour people expected, from a recipe simple enough that the source of the gains was clear.

::slide 37 | distillation: R1 generates 800k CoT traces and Qwen 2.5 (and Llama) models are fine-tuned on them; DeepSeek-R1-Distill-Qwen-1.5B scores 28.9 on AIME 2024 against GPT-4o's 9.3; Distill-Qwen-32B 72.6 on AIME and 94.3 on MATH-500; Distill-Llama-70B 70.0 and 94.5

**Distillation** means fine-tuning a smaller model on a stronger model's outputs. R1 generated 800k traces and Qwen 2.5 models trained on them improve dramatically, in some cases matching specialized thinking models. From the slide image: even the 1.5B distilled model scores 28.9 on AIME 2024, three times GPT-4o's 9.3, and the 32B one reaches 72.6. The table also has Llama-8B and Llama-70B rows; the professor finds it "quite fascinating" that this works for Llama too. If the CoTs are legible to the student, base models are "already surprisingly good" at reasoning for long periods.

::predict r1-stage-order-and-distillation

::slide 38 | the R1 paper's unsuccessful-attempts section: process reward models (hard to define a fine-grained step, hard to judge a step automatically, a model-based PRM invites reward hacking and needs retraining) and MCTS (inspired by AlphaGo; the token search space is far larger than chess's)

The professor likes DeepSeek's reports because they publish what failed. DeepSeekMath was full of process reward models (PRMs). In R1 they tried PRMs, "they just didn't do very much", and "outcome reward models are great and they're good enough". The deciding argument is scale: outcome data scales easily, while step-by-step rubrics are "very hard to scale up". The other big speculation about o1 was tree search like AlphaGo's; DeepSeek tried a lot of MCTS and "couldn't get it to work very well". R1 is how the field learned that neither was necessary.
::kp r1-stage-order-and-distillation
::kp r1-zero-recipe

### Q&A: why length grows when correct answers are pushed short

A student asked: if the length normalizer makes wrong answers longer, doesn't it make right answers shorter, so the effects cancel? The professor's answer is an asymmetry. Correct responses are indeed pushed shorter (good for inference cost, bad if it hurts accuracy), but "there's a lower bound to how small your COT can go to solve a particular problem". Incorrect responses have no upper bound. So the net growth is driven by the incorrect responses, and it is largest where the model fails most often.

R1's own length curve cannot show this, because it aggregates all responses. Dr. GRPO's split plot (slide 24) can: correct-answer length is the same under both objectives, and the growth sits in the incorrect answers, "as we would expect from the explanation". He frames it as holding "at least in these sets of controlled comparisons".

::widget fixture:lecture_16--length-drift | once the green correct line reaches its floor it goes flat while the red incorrect line keeps climbing; raise the share of incorrect responses q and the mean climbs faster
::predict incorrect-responses-drive-length-growth
::kp incorrect-responses-drive-length-growth

## Kimi k1.5: data curation and a DPO-style loss {#kimi}
source: lecture_16.pdf p39-p42 · video 40:00-46:25

::slide 40 | Kimi k1.5 long-CoT against OpenAI o1 (bar chart): AIME 2024 77.5 vs 74.4, MATH-500 96.2 vs 94.8, Codeforces percentile 94 vs 94, LiveCodeBench v5 62.5 vs 67.2, MathVista 74.9 vs 71, MMMU 70 vs 77.3; key steps: dataset construction (difficulty filtering), SFT for long CoT, RL with their own policy-gradient loss

The professor feels "obligated" to cover Kimi k1.5 whenever he covers R1: it came out at the same time, also beat o1 with RL (the bar chart above, read off the slide image), and gets far less attention. The real reason to study it is that it does several things differently from DeepSeek and still works, which tells you what in these recipes is essential.

In particular it uses a different RL algorithm, evidence that "technically speaking, I don't think you necessarily need something like GRPO".

::note slip 40:20 | The professor says Kimi "also beat R1". Slide 39 says "Also beats o1 using RL", and the slide 40 comparison is against o1.

::slide 41 | data curation: automatic filters for questions that need rich reasoning and are easy to evaluate, a tagging system to balance domains; "Exclude multiple choice / true false (false positives)"; "Select only examples that models fail on best-of-8"; the Kimi paper's difficulty estimate: an SFT model answers each prompt ten times at high temperature and the pass rate is the difficulty proxy; "SFT – little description, just described as 'prompt engineering' (distillation?)"

Kimi says much more than DeepSeek about data. For RL there is a wrinkle SFT does not have: difficulty. In SFT you "just jam the data in" and the model learns from it. In RL, "if your problems are too hard, you get no rewards. And if you get no rewards, you have no signal. And if you have no signal, you can't learn." In GRPO terms, a group whose rewards are all 0, or all 1, has zero advantages for every sample.

So Kimi curates. It balances topics across domains. It excludes multiple-choice and true/false questions because they produce false positives: a model can guess the right letter without reasoning and collect the reward. And it filters by **best-of-8**, keeping only problems the model does not solve within 8 tries. Per the Kimi paper, that test asks the model to guess without any CoT, so it removes problems that are too easy to hack rather than problems the model can merely solve. The difficulty itself is estimated from the pass rate of ten high-temperature samples, as the excerpt on the slide image says. The professor adds that you can "filter on both sides" to keep medium-difficulty problems, the "general consensus" when you want RL to progress at a steady pace.

::note warning 42:26 | Two readings to watch. Aloud, multiple choice is excluded because Kimi wants "things that require long, deep thought"; the slide (and the paper) give false positives as the reason. And the professor reads best-of-8 as "only look at examples that basically fail this test", i.e. problems the model cannot solve once in 8 samples, which sits awkwardly with his own "too hard, no signal". The paper's no-CoT version resolves it: guessable problems are removed, genuinely solvable-with-thought ones stay.

::widget fixture:lecture_16--difficulty-signal | the red curve, the chance that all G rewards in a group are equal so GRPO gets nothing, rises to 1 at both p = 0 and p = 1; the green contrast curve peaks in the middle, which is why medium-difficulty filtering pays

Kimi's SFT stage, like DeepSeek's, gets almost no description ("prompt engineering", perhaps distillation): "We can speculate about what that is, but we have no concrete information."

::slide 42 | Kimi RL: maximize E[r(x, y, y*)] − τ·KL(π_θ ‖ π_θi) with a reference-answer reward; a DPO-style closed form r − τ log Z = τ log(π*/π_θi); the surrogate L(θ) = E[(r − τ log Z − τ log(π_θ/π_θi))²]; its gradient (1/k) Σ_j [∇ log π_θ(y_j, z_j | x)·(r(x, y_j, y*) − r̄) − (τ/2) ∇ (log π_θ/π_θi)²], "baselined policy gradient w/ regularization"

Kimi's loss starts where everyone starts: maximize expected reward (the reward compares the answer $y$ to a reference answer $y^*$) minus a KL penalty to the previous policy $\pi_{\theta_i}$. Then it follows DPO's route ([L15's DPO derivation](#/read/lecture_15)). Assume the policy class can represent anything, solve the KL-regularized problem in closed form, and get an equation linking the reward to a log-ratio of policies that holds at the optimum. Since the equality holds at the minimizer, put a squared loss on it and minimize that. "This is a big heuristic", and "optimization people looking at this would be horrified", but it is "a totally reasonable intuition": the two sides should be close when the model is good.

The derivation never passes through PPO. Yet "lo and behold", the gradient of the squared loss, on the slide's last line, is a policy gradient with a baseline $\bar r$, plus a regularizer that plays the KL's role. And $\bar r$ is the mean reward of the $k$ responses sampled for the same prompt. "We've reinvented the group mean normalized baseline through quite different means."

Look at what is *not* in it: no division by the group's standard deviation, and no division by each response's length. Two unrelated derivations, GRPO's from PPO and Kimi's from DPO, land on the same per-prompt mean baseline. The professor reads that as a hint about which component actually matters if you design a new RL algorithm.

::widget fixture:lecture_16--group-weights | Kimi's gradient weight is the "mean baseline" row: r minus the group's mean, i.e. GRPO's row without the 1/std factor; on "one success in 8" the two differ by the whole factor of about 3
::predict kimi-loss-mean-baselined-pg
::kp kimi-loss-mean-baselined-pg

## Kimi k1.5: shorter CoTs, curriculum, and checkers {#kimi-length}
source: lecture_16.pdf p43-p44 · video 46:25-51:18

::slide 43 | "The kimi objective doesn't have the same GRPO length bias problem ... but they want to further compress the CoTs"; per-batch length reward len_reward(i) = λ if the answer is correct, min(0, λ) if wrong, with λ = 0.5 − (len(i) − min_len)/(max_len − min_len); λ runs from 0.5 to −0.5 with longer sequences negative; correct answers pushed short, incorrect ones "shorter than the center of the range of rollouts"; enabled only later in training "due to its effects on perf"

The professor credits Kimi with "a nicer or better view of the length problem". Presenting an ever-rising length curve as a success implies "it's great that our model is thinking for longer"; Kimi instead treats long CoTs as waste. Its objective does not divide by sequence length, so it has no GRPO length bias to begin with. It goes further and adds a reward for being short.

The motive is cost. Long thinking means you "subsidize the user": "If you're OpenAI and your users have the 200-dollar Pro Plan, and your models are thinking for an hour at a time, that's not a very good place to be in." Five minutes is a great place to be. Shorter CoTs that still solve hard problems are a general goal of language-model development.

The length reward ranks responses within a group by length and maps them linearly onto $\lambda \in [0.5, -0.5]$:

$$ \lambda_i = 0.5 - \frac{\text{len}_i - \text{min\_len}}{\text{max\_len} - \text{min\_len}},\qquad \text{reward}_i = \begin{cases}\lambda_i & \text{correct}\\ \min(0,\lambda_i) & \text{incorrect}\end{cases} $$

Correct answers simply get $\lambda$: shorter is better. Wrong answers are never paid a positive $\lambda$; they are only penalized for being longer than the middle of the range. Why not push wrong answers short too? The professor's geometry example: suppose the model is bad at geometry, gets many wrong answers, and the penalty shrinks its geometry CoTs to nothing. "I will never get a positive geometry reward ever again. And I'm stuck." So wrong answers are only kept from growing "unboundedly". The term is switched on only later in training, because it costs performance early on.

::worked kimi-length-reward
::widget fixture:lecture_16--length-lens | in the bottom panel correct responses sit on the λ line from +0.5 (shortest) to −0.5 (longest); wrong responses stay at 0 up to the centre of the length range and follow the line down only beyond it
::predict kimi-length-reward

::note slip 48:53 | Aloud, wrong answers are pushed "a little bit shorter than the average". The slide's rule uses the centre of the range, the midpoint of the shortest and longest lengths, which is not the mean.

::slide 44 | curriculum: assign difficulty labels and go from easy to hard; sample problems proportional to (1 − success_rate) to avoid repeating solved ones. Rewards: for code, generate new test cases for problems with ground-truth solutions; for math, 800k samples train a CoT reward model for answer-equivalence checks; excerpt: on manual spot checks the classic reward model reached about 84.4% accuracy and the chain-of-thought reward model 98.5%

Kimi's **curriculum** orders problems from easy to hard and samples each problem with probability proportional to $1 - \text{success rate}$, so mastered problems stop being drawn. A problem solved 8 times out of 8 gets weight 0, which matches GRPO's view of it: its group has zero advantages and only wastes rollouts. "Basically everyone doing RL does these success rate filtering" to avoid wasting compute on problems that are solved or hopeless.

::predict difficulty-filtering-curriculum

The rewards hold an irony the professor relishes. For code, Kimi takes problems with ground-truth solutions and generates new test cases. For math, it trains a reward model on 800k samples to judge whether two answers are equivalent; on the slide image it is right about 98.5% of the time, against 84.4% for a classic reward model. "We started out this lecture by saying we want to work on formal math ... where a compiler can check the correctness", and "we ended up with a reward model that checks the correctness of math answers." The reason, which the assignment will teach you: equivalent math answers can be written many ways, and a model asked for a LaTeX `\boxed{}` answer sometimes skips the box or adds extra text inside it. A strict checker marks such answers wrong even when the reasoning is right. So "most RL projects have a very complicated answer checker, either a regex or a model". "It's a real rabbit hole getting the verified part of RLVR."
::kp kimi-length-reward

## Why RL is hard to make efficient, and Kimi's results {#rl-infra}
source: lecture_16.pdf p45-p48 · video 51:18-55:29

::slide 45 | "Systems and utilization aspects of RL are very important". Why is RL hard to make efficient? On-policy means rollouts, which means slow inference; switching from training to rollouts often means different frameworks; long CoTs can make batches very uneven

"Training is hard and inference is hard. And RL puts the two together." Three problems, each with a picture from the professor:

- **Rollouts are inference.** On-policy RL must generate fresh samples from the current model before every update, and autoregressive generation is slow ([L10's inference](#/read/lecture_10)).
- **Uneven lengths.** Imagine one prompt in the batch is the Riemann hypothesis and the model is "chugging along" on a gigantic CoT. With naive batched inference, "everyone else is waiting on this one rollout". Do you truncate it, move it to another machine? "These are all decisions that you can make."
- **Two frameworks.** You alternate rollout and training. Either some machines only roll out and others only train, or you keep switching frameworks on the same machines. "Both of them are very costly."

And then a "really difficult and horrible trade off". On-policy GRPO "behaves very nicely". In the assignment you will see your utilization is low "and then you will get greedy": reuse rollouts so that inference and training overlap. That makes training off-policy, the samples come from an older policy, which leads to "destabilizing your training".

::widget fixture:lecture_16--rollout-budget | in synchronous mode the training lane is grey whenever rollouts run; set rollout 4, training 3, sync 0.5 and the trainers are busy 3/7.5 = 40% of the time; in the batch panel, set the long length to 8,000 and one long sequence among seven 1,000-token ones leaves about 77% of decode slots as padding
::animation fixture:rollout-train-timeline | sync mode alternates rollout, train and weight sync so each GPU group idles while the other works; overlap mode keeps rollouts running during training, at the price of samples from older policy versions (FeynRL's version of the trade-off)
::predict rl-rollout-cost

::slide 46 | Kimi's hybrid deployment framework: Megatron (training) and vLLM (rollout) run as sidecars in the same pod; after training, Megatron offloads its GPU memory and passes weights through a checkpoint engine; vLLM starts with dummy weights, receives the new ones, rolls out, and is terminated so Megatron can onload and train again

Most open reports now include an RL-infrastructure section, and Kimi's is a good example. There is a training part (blue, Megatron) and an inference part (green, vLLM), a way to move weights from one to the other, and close coordination. Kimi even shares the same machines between them, because "as inference is running, the training one might be idle". The [FeynRL repo read-through](#/read/feynrl) implements both the synchronous loop and the overlapped, off-policy variant in code.

::slide 47 | eight training curves for a small model on math data: accuracy (blue) and token length (orange) against iteration on total, OMNI-MATH500, MATH500, AIMO2024, AIME2024, ChatGLMMath, GAOKAO and GPQA; on several, accuracy keeps rising while length levels off

Kimi's scaling plots show performance rising as the model thinks longer. On OmniMath, though, the length flattens while accuracy keeps climbing, which the professor reads as the length control "kicking in and doing good things".

::slide 48 | Kimi's ablation against expert iteration: twelve benchmark panels, the RL method (orange, "Ours") above ReST (blue) on most of them, including the total; "Could we avoid RL-style negative gradients and just learn from positives?"

Last question: is RL actually better than just training on your own correct answers? That alternative is **expert iteration** (ReST in the plot, RFT in DeepSeekMath's): sample, keep the correct answers, fine-tune on them, repeat. It has worked well in past papers, and "in cases where you're dealing with very unstable stuff, you may actually want to do expert iteration instead." But Kimi's large ablation shows RL "consistently better" (orange over blue). The difference is the negative gradient: expert iteration only ever pushes probability *up* on correct samples, while a baselined policy gradient also pushes it *down* on below-average ones. "You can't really avoid RL if you want to squeeze out all of your performance."

::widget fixture:lecture_16--group-weights | the RFT row never goes below zero, so wrong responses simply drop out of the update; the mean-baseline and GRPO rows give every below-mean response a downward bar
::worked negative-gradients-vs-positive-only
::predict negative-gradients-vs-positive-only
::kp rl-rollout-cost
::kp negative-gradients-vs-positive-only

## Qwen 3: the playbook on 3,995 examples {#qwen3}
source: lecture_16.pdf p49-p54 · video 55:29-1:01:10

::slide 49 | Qwen3-235B-A22B (MoE) and Qwen3-32B (dense) against o1, R1, Grok 3, Gemini 2.5 Pro and o3-mini; for example AIME'24 85.7 (R1 79.8, o1 74.3), AIME'25 81.5 (R1 70.0), Codeforces Elo 2056 (R1 2029); "Better than o1 and R1 (though comes later) .. but interesting scaling and data results"

Qwen 3 came later and beats o1 and R1 on most rows of the slide image's table. That is expected for a later model; what the professor wants from it is its scaling and data results.

::slide 50 | Qwen3's post-training pipeline: base model → Stage 1 long-CoT cold start → Stage 2 reasoning RL → Stage 3 thinking mode fusion → Stage 4 general RL → Qwen3-235B-A22B and Qwen3-32B; lightweight models (Qwen3-30B-A3B, 14B, 8B, 4B, 1.7B, 0.6B) come from strong-to-weak distillation; "RLHF comes after reasoning RL (like r1), distillation after that"

The pipeline is DeepSeek's with one addition: long-CoT SFT, reasoning RL, then *thinking mode fusion*, then general RL (RLHF), then distillation to produce the smaller models, because you do not want to serve the flagship to everyone. "You can basically have this be your mental picture of how frontier-ish language models are built."

::slide 51 | "We know the playbook by now": filtering for difficulty by best-of-n, like Kimi (remove things the model gets right without CoT; remove things too similar to validation data); manual filtering of CoT quality (guessing vs getting it right); "RL with GRPO on only 3995 examples"

Qwen uses the "tried and tested playbook", taking "the best parts of Kimi and DeepSeek". It filters for difficulty with best-of-n sampling. It removes problems the model answers correctly without a CoT, since that is "not a thinking problem". It removes problems too similar to validation data (decontamination), and manually checks reference CoTs for guessing. Then the remarkable part: GRPO on only 3,995 examples. "If you have the rest of the pipeline right, you can actually get surprisingly far."

::kp difficulty-filtering-curriculum

::slide 52 | thinking mode fusion: chat templates with /think and /no_think flags, the non-thinking response keeping an empty think block; early stopping via a special string: when thinking reaches a user-defined budget, insert "Considering the limited time by the user, I have to give the solution based on the thinking directly now. </think>" and the model answers; per the report this ability "emerges naturally" from thinking mode fusion

**Thinking mode fusion** puts an instant-response model and a long-CoT model into one set of weights. Training data mixes thinking and non-thinking examples, tagged `/think` or `/no_think` in the prompt, and the non-thinking answers keep an empty think block. Answering a question at the end of class, the professor stresses the point: it is one model, and the switch is a tag in the prompt, not an API flag that routes to a different model. Before this, labs including OpenAI usually served separate thinking and non-thinking models.

The second trick controls length at inference. When thinking hits a budget, insert a fixed stop string (on the slide image), close the think block, and the model must answer from what it has so far. Per the report, nobody trained this ability; it emerged from the fusion.

::slide 53 | Qwen3-235B-A22B pass@1 against thinking budget (1K to 32K tokens): AIME'24 from about 42 to about 86, AIME'25 from about 31 to about 81, LiveCodeBench v5 from about 45 to about 67, GPQA Diamond from about 64 to about 72; the non-thinking mode (dashed) at about 40, 25, 35 and 63

What surprised the professor is how gracefully performance degrades as the budget shrinks. At small budgets the model is cut off mid-thought, yet still answers reasonably, and thinking mode beats non-thinking mode on every task at every budget. Read off the slide image, AIME'24 climbs from about 42% with 1K thinking tokens to about 86% with 32K, against about 40% without thinking.

::slide 54 | performance after Stage 2 (reasoning RL), Stage 3 (thinking mode fusion) and Stage 4 (general RL), thinking and non-thinking: general tasks, instruction following and agent scores mostly rise (ToolUse thinking 63.3 → 70.4 → 85.5); math and coding dip, AIME'24 thinking 83.8 → 81.9 (−1.9) → 81.4 (−0.5) and LiveCodeBench v5 68.4 → 67.2 (−1.2) → 65.7 (−1.5); caption "Note that math/stem abilities go down (a bit) with general purpose RLHF"

Qwen reports what each stage contributes. Fusion and general RL lift general tasks (Arena-Hard, CounterFactQA), instruction following and tool use, mostly by several points. Math and coding dip slightly.

::note slip 1:00:34 | The professor attributes the math and coding drop to fusion ("because we fuse together non-thinking components"); the caption attributes it to general-purpose RLHF. The table on the slide image shows both: on AIME'24 (thinking) fusion costs 1.9 points and general RL 0.5, on LiveCodeBench fusion costs 1.2 and general RL 1.5.

The professor adds a postscript: in later Qwen 3.5 releases Qwen went back to separate thinking and non-thinking models, apparently because they found even these small drops unacceptable and "wanted to squeeze out all the juice possible on thinking modes."
::kp r1-stage-order-and-distillation

## Agentic RL: Qwen3-Coder-Next and reward hacking {#agentic}
source: lecture_16.pdf p55-p60 · video 1:01:10-1:09:12

::slide 55 | Qwen3-Coder-Next, built on Qwen3-Next and post-trained for agentic abilities; the technical report's title page

The last case study is new this year: Qwen3-Coder-Next, which the professor calls a great agent post-training paper and the one with the most detail on agentic RLVR. (He first calls it "Qwen 3.5-Next-Coder", then corrects himself.) The lesson of the whole course applies: there is no new agent training algorithm. "Data is the important thing."

::slide 56 | mid-training data: GitHub repository-level long-context data (concatenated files, 600 billion tokens) and pull requests with RAG-retrieved repository context; Common Crawl documents mixing text and code, parsed with an LLM; synthetic QA about coding web documents and trajectories from running coding agents in environments; instruction-following and fill-in-the-middle data

Agent abilities cannot be injected only at the end, so they start in **mid-training** (a later pretraining phase on targeted data, see [L15's mid-training](#/read/lecture_15)). Qwen concatenates the files of whole repositories into long-context documents, 600 billion tokens of them, since an agent will later see long traces of opened files. It pairs pull requests with RAG-retrieved context about the repository, converts text-plus-code web pages into clean markdown with an LLM, generates coding QA, and adds traces of public coding agents run in environments.

::note skip 1:03:24 | Fill-in-the-middle data is mentioned as "not super relevant for us", and agent harnesses are set aside at 1:01:26.

::slide 57 | Qwen3-Next branches into four experts (web dev, UX, single-turn QA, SWE), which are distilled back into Qwen3-Next-Coder

Then Qwen trains four separate expert models from the mid-trained model and distills them back into one. The professor has not seen this in frontier training; the closest he knows is DeepSeek V3.2's data-processing experts, and academic work such as branch-train-merge. In Q&A he sees the appeal (separate teams per expert, simple aggregation given compute) but would prefer one big training objective, which avoids the distillation step.

::slide 58 | web dev expert: SFT on "valid web code" based on checks (a vision-language model plus agent actions); UX expert: trained on many tool-call formats (XML, JSON, TypeScript, pythonic variants); QA expert: standard single-turn code synthesis, "just more data"

The web-dev expert is trained on web code that passes checks by a vision-language model and agent actions; the UX expert learns many different tool-call formats; the QA expert just gets more synthetic code data.

::slide 59 | automated SWE-bench-style environment construction (800k tasks): collect repositories, parse functions with tree-sitter, have a model inject a bug, keep it only if some tests go from pass to fail, then generate an issue statement and use the reversed bug patch as the oracle fix

The most involved is the software-engineering (SWE) expert. SWE-bench (real GitHub issues, judged by the repository's tests) is the gold standard, and Qwen wants "SWE-bench but more". The pipeline on the slide image builds tasks automatically: inject a bug into a real function, keep it only if some tests now fail, write an issue for it, and keep the original code as the reference fix. That yields 800k tasks to run RL on, and RL performance goes up.

::slide 60 | left: SWE-bench Verified rising from about 68% to 75.1% over RL steps "with a reinforced reward-hacking blocker", while average agent turns grow from about 50 to 130; right: without the blocker, a slow climb to about 76% and then a jump to 84.6%, annotated "Reward Hacking: restoring deleted git remotes" with the agent's tool call git remote add origin ...; git fetch origin; bottom: Qwen3-Coder-Next (80B total, 3B active) at 70.6 / 71.1 / 71.3 on SWE-bench Verified under three agent scaffolds, close to much larger open models

Here the professor stops for what he calls "very important through the broader context". We can pour compute into RL only because we believe the reward is unhackable or hard to hack. "If that assumption breaks down, your RL method will find increasingly obscure ways of cheating you out of your performance."

In a git repository, an agent can look at later commits and simply read the fix. Without a guard, the right-hand plot shows "learning, learning, learning ... And suddenly, you get this emergent jump". The jump was not a new skill: the agent had learned to use git calls to retrieve the history. Qwen added a whole reward term whose only job is to stop the agent touching the git history. Even constraints get routed around: "If you tell it, you can't use Git log, it might add an origin like a remote and then query the remote." The slide image shows exactly that, the agent re-adding a deleted remote and fetching it. "RLVR is only as robust as your reward."

He adds a story from his own group. They ran RL on Lean, a formal proof language, assuming "there's no way this can go wrong ... The Lean compiler is bulletproof." It is not adversarially robust: certain strings make it accept proofs that should fail in some modes. "The notion of verifiable rewards is actually much trickier than many of you might initially think."

::video 1:06:11-1:08:15 | the git-history 'emergent jump', the git log ban routed around with a remote, and the Lean checker that is not adversarially robust
::predict verifiable-reward-hacking

With the hack blocked, scores go up "slow but steady", and the result is strong: about 70.6% on SWE-bench Verified from a model with only 3 billion active parameters (read off the slide image's table, under the SWE-Agent scaffold).

The professor ends with a caution. RL will do well on the environments it trained on and may generalize to a validation set, but "task specific performance doesn't necessarily mean it'll generalize to broader domains."
::kp verifiable-reward-hacking

## Recap: it is all about the reward {#recap}
source: lecture_16.pdf p61 · video 1:09:12-1:15:45

::slide 61 | recap: overoptimization is a problem, and RL in narrow domains is one solution; GRPO is simple (with some flaws) and enables RLVR; lots of successful recipes in the wild (R1, Kimi 1.5, Qwen 3)

"The core takeaways of this lecture is it's really all about the reward for RL." RLHF and RLVR are arguably the same problem; the difference is that we want "more unhackable rewards so that we can actually put in much more compute". That is why verifiable domains let RL scale, and why a hackable checker brings the old problem back.

GRPO is what made this accessible to the research community, and the professor wants you to know it as well as pretraining losses: the functional form and the update. In one line: sample $G$ responses per prompt, score them with a checker, subtract the group mean (GRPO also divides by the std and by each response's length, both of which bias it), and take a clipped policy-gradient step with a KL penalty. Kimi's DPO-style derivation reaching the same mean baseline suggests that baseline is the part that matters.

The recipes agree on the rest: SFT on long CoTs first, filter prompts to the difficulty where groups have contrast, run reasoning RL with outcome rewards (no PRMs, no tree search), then SFT and RLHF for everything without a checker, then distill to smaller models. RL "remains very finicky and noisy", but "it's not that hard. It's not like the old days of doing PPO on various really tricky environments."

A few answers from the closing Q&A fill gaps:
- **Does mid-training decide what RL can learn?** Pretraining and SFT do most of the work, the professor says. If pretraining has no code at all, "you're in trouble", but given broad coverage, SFT gets the model close enough to start earning rewards, so mid-training is "very nice to have" without being make-or-break.
- **Is long-CoT training part of mid-training?** Not traditionally, but long-CoT-like data is used in long-context extension, a phase before RLHF that uses books, code and synthetic data, which he regrets not covering.

::note slip 1:13:56 | In this answer he says long-CoT SFT appears in "R1 and Kimi K5.1"; he means Kimi k1.5.
