---
title: L15 · After pretraining (SFT and RLHF), read through
minutes: 45
---
Pretraining gets you a GPT-3: a strong next-word predictor that you can barely steer. This lecture covers the two stages that turn it into a ChatGPT. The first is supervised fine-tuning (SFT) on demonstrations. The second is reinforcement learning from human feedback (RLHF) on preferences, done with PPO or DPO. Most of the lecture is about the *data* each stage needs, because that is where the leverage and the trouble are. After it you can explain why a few hundred examples can change a model's behaviour while a few thousand correct facts can make it hallucinate, derive DPO from the RLHF objective, and say why pushing hard on a learned reward eventually makes the model worse.

## What is post-training for? {#why}
source: lecture_15.pdf p2-p6 · video 0:05-6:52

::slide 2 | left: GPT-3 used as a copywriting tool (three product blurbs for a skincare brand); right: ChatGPT, “Optimizing Language Models for Dialogue”; the question at the top is how to get from one to the other

Everything so far in the course gets you to a souped-up GPT-3, whose usefulness is limited to tasks where reliability does not matter, like the copywriting on the left. Going back to it after using ChatGPT, the professor says, feels like asking “what is the point of this thing?” This lecture covers the step from GPT-3 to ChatGPT. The next lecture covers the step from ChatGPT to reasoning models like o1.

::slide 3 | a long, fussy plotting prompt (four series, smoothed curves, 10–40% error bars, an animated pie chart in “continuous time”) and the GPT-4 output that satisfies it, from Bubeck et al. 2023

The capability being bought is **instruction following**. At the GPT-3 level the only way to steer a model was to pile up few-shot examples and hope. With GPT-3.5 and GPT-4 you could write a long, programmatic prompt like this one and get the whole thing right in one shot. The professor calls this a remarkable form of control.

::slide 4 | “Pretraining data isn't quite what we want (but it scales)”; can we collect data of behaviours we *do* want; three questions: what does that data look like, how do we best use it, do we need scale

Pretraining stays essential. It scales in a diverse, broad way, and if you skip it and “try to train our way to victory, we will get none of the things that we want”. But the wanted behaviours are buried in what the professor calls the “primordial soup” of pretraining. Getting them out takes explicit data collection, explicit steering and a lot of messy engineering. The lecture's three questions organize the rest: what the data looks like, how to use it, and whether it needs scale.

::slide 5 | rich information before the ChatGPT competition (Stiennon 2020, Bai 2022); poor information now; a news excerpt: Scale AI compared 1,729 Bard rewrites against ChatGPT in October 2023

A caveat before starting: information about frontier post-training is sparse. The detailed sources are old. Stiennon et al. 2020 (“Learning to summarize from human feedback”, the original RLHF-for-LMs paper) and Anthropic's HH paper (Bai et al. 2022) publish their annotation guidelines in their appendices, which the professor recommends reading; for anything resembling a modern model, no guidelines are public. Since the competition heated up, labs treat post-training data as a trade secret. Open-source recipes exist, but most of them rely on distillation from stronger models, which is a different thing from the human data collection frontier labs do.

The excerpt on the slide, from leaked Scale AI documents, shows the competition at work: in 2023 annotators were told to study why GPT-4 beat Bard and to write responses at least as good as GPT-4's.

::slide 6 | InstructGPT's three steps: (1) a labeler writes a demonstration and GPT-3 is fine-tuned on it (SFT, boxed in red); (2) several outputs are sampled, a labeler ranks them (D > C > A = B) and a reward model is trained on the ranking; (3) PPO optimizes the policy against the reward model

This is the standard recipe, from InstructGPT (Ouyang et al. 2022):
1. **SFT.** For each prompt an annotator writes a reference response, and the model is fine-tuned on these demonstrations. This is imitation.
2. **Reward model.** The model produces several outputs for each prompt, an annotator ranks them, and a reward model is trained to score outputs the way the annotator would.
3. **RL.** The policy is optimized to get high reward-model scores, here with PPO. This is reinforcement.

The professor's framing is that the algorithms are “not really the secret sauce ... It's going to be the data.” SFT in particular is “basically exactly the same as pre-training”: the same next-token loss on a different dataset. So the SFT half of the lecture is almost entirely about data, and the RLHF half splits between data and algorithms.

::video 5:42-6:25 | the two-part recipe, and data rather than algorithms as the leverage
::predict sft-then-rlhf
::kp sft-then-rlhf

## What is inside an SFT dataset, and how has it changed? {#sft-data}
source: lecture_15.pdf p7-p14 · video 6:52-19:08

::slide 9 | the open-world lineage, top left to bottom right: FLAN, Self-Instruct, Alpaca, ShareGPT/Vicuna, Open Assistant, WizardLM, Tulu 3, Nemotron (“Tool use etc.”)

The slide is roughly chronological. Each dataset stands for a different idea of where instruction data should come from:
- **FLAN.** NLP researchers had already collected many supervised datasets of inputs and outputs, so convert all of them into instructions and train on the lot. The professor calls it the oldest and in some ways most visionary of the set, and also “not the right thing to do”.
- **Self-Instruct.** Models are getting better, possibly better than some annotators, so let the model generate the instruction data itself.
- **Alpaca and Vicuna.** Distillation: take a strong model's responses as targets. Vicuna used prompts that users shared online (ShareGPT) as the inputs.
- **Open Assistant (OASST).** Fully human: volunteers write both prompts and responses, “kind of like Wikipedia”.
- **WizardLM and Tulu 3.** Increasingly elaborate pipelines that use language models to generate harder instruction data.
- **Nemotron.** The newest shift, away from chat and toward agents and tool use.

All of this is the open world. Closed labs also collect human data, and how they do it is a large ingredient that none of this reflects.

::note slip 8:07 | The professor says FLAN “was used to train the T5 model by Google”. The original FLAN (Wei et al. 2022) fine-tuned Google's 137B LaMDA-PT model; Flan-T5 came from the later FLAN collection (Chung et al. 2022).

A student asked how much the correctness of the input-output pairs matters. The professor called it a nuanced question. Collect the best responses you can, because bad examples teach bad behaviour. But models can be instruction-tuned on “all sorts of strange things”, even data without proper responses, because pretraining generalization carries much of the load.

::slide 10 | four random FLAN examples: an email plus “Write a subject line for this email” (target “Ronald Chisholm LOI”); a news item with “OPTIONS: - World - Sports - Business - Science/Tech” (target “Business”); a long travel article plus “Write highlights” (a three-sentence target about the city's history and art); a restaurant data-to-text task

Look at FLAN's actual examples and you can see where they came from. The email task is the Enron email corpus turned into subject-line prediction. The highlights task is a summarization dataset with the article on the left and the reference summary on the right. Nobody prompts a chatbot like this, with the instruction glued onto the end. The professor names three problems:
- **Inherited quality.** The source NLP datasets were not high quality. The summaries are short and often hallucinated, with details that are not in the input, and a model trained on them inherits those flaws.
- **Unnatural format.** The templated structure carries all the way into the model's behaviour. In his words, “the unnaturalness really gets you”.
- **The wrong point on the quality–quantity trade-off.** FLAN was built on the theory that post-training, like pretraining, needs scale, so it collected a gigantic dataset. Later datasets got much smaller and still worked, because a strong pretrained model needs only a few high-quality examples.

::video 12:45-14:20 | FLAN's three problems: hallucinated summaries, “the unnaturalness really gets you”, and “the wrong point ... on the quality, quantity trade off”

::slide 11 | three Alpaca examples: three health tips, a definition of “algorithm”, and an avg_list function whose comment says the average of [4, 3, 6, 10, 8] is 6.4

Alpaca, from the professor's students, distilled a strong OpenAI model's responses into input-output pairs. The inputs look natural and the outputs are longer and chattier. The finding was that this kind of data “reliably induced ChatGPT-like behavior”, but only on a good enough base model, the original Llama. Both pretraining and post-training have to be right. After Alpaca people realized that getting most of the way to a ChatGPT-style system was not that hard. Getting the details right still was.

Distilled data is not checked data, though. The code example's own comment is wrong: (4 + 3 + 6 + 10 + 8) / 5 = 31/5 = 6.2, not 6.4.

::note slip 14:26 | The professor says Alpaca “distilled ChatGPT traces”. Alpaca was generated with text-davinci-003 through the Self-Instruct pipeline, and slide 16's table lists it as “Generated w/ Davinci-003”.

::slide 12 | two Open Assistant examples: a long answer about “monopsony” in labour economics that ends with a formal reference (Bivens & Mishel 2013), and a list of cheap science projects for kids

After Alpaca there was great optimism that a large, high-quality, human-written instruction dataset would let open models catch up with the closed labs. Open Assistant was that crowdsourced effort, with long, detailed, expert-style responses. The professor recalls “like 10,000 or more examples” before the project stalled. The Wang et al. table on slide 16 lists 34,795 OASST-1 instances, so “or more” is right. The monopsony example returns in the knowledge section, because its reference turns out to be a problem.

::slide 13 | Nemotron-SFT-OpenCode-v1: user questions about JavaScript promises and calculator number types, answered by assistant turns that emit JSON tool_calls (“skill”, “todowrite” with a four-item to-do list) and cite AGENTS.md

Products have moved from chat to agents. Users want tool calls and to-do lists, like the list Claude Code or Codex writes and checks off as it works. So a large part of NVIDIA's open Nemotron SFT data is agentic: assistant turns with structured tool calls, supervised directly into the model.

::slide 14 | what varies across these datasets: chattiness, detail, tool use

The summary has three axes. **Chattiness**: FLAN is usually valid data, “but people don't want to talk to a NLP benchmark”, so later datasets give longer, more human-like responses. **Detail**: OASST packs in far more factual knowledge, which “can be both a pro and a con”. **Tool use**: the newest shift, toward agentic applications.

::predict sft-data-progression
::kp sft-data-progression

## Does the style of the data matter? Length and lists {#style}
source: lecture_15.pdf p15-p18 · video 19:08-21:58

::slide 15 | what varies across datasets: length and bullet points (style), references and complex knowledge; less visible but important: scale and safety

Suppose you are put in charge of collecting SFT data. You have to decide on style (length, bullet points), how much knowledge goes into the responses, how many examples you need, and what safety data to collect. The next three parts take these in turn, starting with style.

Style is not an accident. When people say Claude has a different tone from ChatGPT, or that ChatGPT is too chatty, those are, in the professor's words, “conscious decisions that are made by the data collection folks”.

::slide 16 | Wang et al. 2023, Table 1: twelve instruction datasets with instance counts, turns, and average prompt and completion lengths, e.g. Flan V2 31.2 tokens per completion, Alpaca 64.6, Open Assistant 1 212.5, ShareGPT 357.8

Datasets differ a lot in response length. The numbers are on the slide image: an average Flan V2 completion is 31.2 tokens and a ShareGPT completion is 357.8, so 357.8 / 31.2 ≈ 11.5 times longer. Open Assistant sits at 212.5. The table also records where each dataset came from. Self-Instruct was generated with vanilla GPT-3, Alpaca with Davinci-003, and ShareGPT is user prompts with outputs from various models.

::slide 17 | Dubois et al. 2023: the preference for lists (about 55–65%) and for longer outputs (about 55–80%) across annotators (humans, simulated annotators, GPT-4) and models

Length matters because of how preferences are judged. Put two responses side by side and ask which is better, and people tend to pick the one with bullet points and the one with more detail. The professor thinks this is partly natural: in a side-by-side comparison, detailed and list-like responses usually *are* better. But it distorts a chatbot's tone away from how people talk. The plot shows the effect is not only human. Reading the slide image, human annotators prefer the longer output roughly 70% of the time and the GPT-4 judge roughly 75%, against the 50% you would expect if length did not matter.

::slide 18 | Wang et al. 2023, Table 3: LLaMA-13B fine-tuned on each dataset, scored on MMLU, GSM, BBH, TydiQA, Codex-Eval and AlpacaEval (win rate against Davinci-003)

Now the trap. The AlpacaEval column (a preference-style win rate judged by a model) swings wildly with the SFT data: 3.2% after Flan V2, 58.1% after Open Assistant, 70.5% after ShareGPT. Over the same rows MMLU barely moves: 50.6, 43.3 and 49.3.

**Worked example.** Flan V2 and ShareGPT give nearly the same MMLU (50.6 against 49.3), but their AlpacaEval scores differ 70.5 / 3.2 ≈ 22-fold, and their completions differ about 11-fold in length (slide 16). If you only tracked AlpacaEval, you would think ShareGPT made a far better model. The long, chatty data won the preference contest without making the model more knowledgeable.

The professor's warning is about engagement signals, which most companies watch: “it's very, very easy to fool yourself into thinking you're getting better data, when in reality, your models capabilities are not changing.” Think about style control separately from capabilities control.

::widget fixture:lecture_12--length-bias | set B's quality equal to the baseline's (q_B = 0) and drag B's padding above 0: the judge's raw win rate climbs past 50% with no change in quality, while the length-controlled rate stays at 50%

The same confound comes back twice later. Model judges share it (slide 48), and RLHF amplifies it, because a policy optimized against a length-loving judge learns to write longer. [L12's evaluation lecture](#/read/lecture_12) shows the fix used by AlpacaEval 2, a length-controlled win rate.

::video 21:07-21:58 | “easy to fool yourself” with engagement signals, and style control separately from capabilities control
::predict length-bias-preferences
::kp length-bias-preferences
::kp lecture_12:llm-judge-length-bias

## Can SFT teach facts? Knowledge, citations and hallucination {#knowledge}
source: lecture_15.pdf p19-p21 · video 21:58-28:16

::slide 19 | the Open Assistant monopsony answer, ending in a bolded reference (Bivens & Mishel 2013, Journal of Economic Perspectives 27(3), 57-78); what is it teaching: the fact, or to output citations when asked? “But by what mechanism?”

Open Assistant was meant to be the highest-quality human data, and it contains answers like this one, which ends with a real citation. The professor's point is that SFT on this example teaches **two things at once**. Next-token prediction teaches the model the fact itself, that this Bivens and Mishel paper exists. It also teaches a behaviour: a good answer ends with a reference.

The second lesson is the dangerous one. Models do not reliably know whether a reference is true or false. If the model learns “emit a reference here” from examples whose content it does not actually know, it can misgeneralize and “hallucinate out a reference” instead of giving a proper one.

::slide 20 | the folklore: fine-tuning on facts the model doesn't know makes it hallucinate; left, Schulman's “Hallucination and Behavior Cloning” slide; right, Gekhman et al.: training accuracy on Known vs Unknown examples over 50 epochs, and dev accuracy peaking near epoch 10 (“Overfitting starts”) then falling

The professor calls the folklore “not wrong”, and the right-hand plot is the evidence. Gekhman et al. split fine-tuning examples into facts the base model already knows and facts it does not. Reading the slide image, the Known examples are fit fast: about 60% training accuracy after the first epoch and over 80% by epoch 10. The Unknown examples are at roughly 15% by epoch 10 and only reach 100% around epoch 50. Meanwhile accuracy on held-out questions peaks near 43% around epoch 10 and then falls toward 40% as the unknown facts get memorized. Training only on known facts does not show this decline. An example that pairs the citation format with knowledge the model lacks is, in the professor's words, teaching it “to forcibly emit unknown knowledge”.

The left panel is John Schulman's version. The professor notes that Schulman, who created PPO, is naturally “very pro reinforcement learning”, but calls the argument sound. What a model knows depends on the model, while demonstrations written by someone else are the same whatever it knows. In the professor's words, “You can't have an external person shoving knowledge down your throat” if you want calibration. So teaching a model what it knows has to be **policy-dependent**, which is a job for RL.

A follow-up question asked whether SFT's loss is doing something wrong here. It isn't: the reference is a correct sequence, and the loss penalizes any probability on anything else. The problem is generalization. One example teaches two entangled things, the reference template and the knowledge, and “if I generalize based on the template, I'm in big trouble.”

::video 22:30-24:18 | “teaching the model two different things at once”, and the folklore “which I think is not wrong”

::slide 21 | takeaways: you may not want to fine-tune on tail knowledge, even if that is the use case; in principle RL-style correctness feedback could help; knowledge storage and extraction in LMs is messy

So you might *not* want to train on the highest-quality data when the model doesn't already know its content. Tail knowledge can be actively harmful, especially when it comes with markers like “References:” that force the model to produce something afterwards.

What counts as tail knowledge? There is no formal definition. In the professor's past work, the length of a topic's Wikipedia article served as a proxy for how well known it is, and training on less-known material did make models hallucinate more.

::predict sft-tail-knowledge-hallucination
::kp sft-tail-knowledge-hallucination

### Why would RL help? The “I know” signal
A student asked why RL would help with hallucination when SFT does not. The professor offered what he called “a folk story”. Suppose the model's activations contain an “I know this” direction. SFT forced references regardless, so the model learned to cite whether or not that direction was active. Under RL the model samples its own answers and gets rewards. References produced while the “I know” direction is on tend to be right and earn good rewards, and references produced while it is off tend to be fabricated and earn bad ones. So the policy can learn to make citing depend on the internal signal.

The limit matters as much as the mechanism: “If you don't know what you know at all, at any level, RL cannot help you there.” RL can only extract a known/unknown distinction that already exists inside the model. It cannot create one. The deck's “in principle” on slide 21 stays in principle. The next lecture's verifiable rewards are where correctness rewards become concrete.

::video 26:31-27:21 | the “I know something” direction, and when RL cannot help
::predict rl-reads-internal-knowledge-signal
::kp rl-reads-internal-knowledge-signal

## How do you make a model safe, and how much data does it take? {#safety}
source: lecture_15.pdf p22-p27 · video 28:16-34:57

::slide 22 | why safety controls: a misinformation threat model (Goldstein et al. 2023) and a scam email requesting a deceased relative's social security number that ChatGPT writes anyway (Kang et al. 2023)

Post-training is where a model meets real users, so it is where misuse has to be handled: disinformation, spear-phishing, scams like the one on the right. Pretraining people, the professor jokes, “live in this ivory tower”. The post-training team is the last line of defence, and its main tool is training the model to refuse malicious inputs.

::slide 23 | “Details are pretty sparse”: an excerpt on safety fine-tuning data (adversarial and “borderline” prompts) and a plot of violation rate (%) against false refusal rate (%) for Llama 3 8B and 70B, each point a different safety/helpfulness data mix; “# Examples? ~ few thousand in Llama 2”

Public information about safety SFT is even sparser than about capabilities SFT. Even the more detailed descriptions omit how many examples were used, which the professor finds “kind of crazy”.

The plot shows the frame every safety-tuning approach works in. There are two error rates:
- the **violation rate**: how often harmful requests get answered;
- the **false-refusal rate**: how often benign requests get refused. The professor's example is “how do I kill a Python process”, answered with “no, I can't let you kill anything”.

You want both low, which means finding “a good Pareto trade between the two” with tailored data. The excerpt names the usual lever against over-refusal: alongside the adversarial prompts, the developers collect **borderline prompts**, which look like adversarial ones but should get helpful answers, “thereby reducing the false refusal rate”. Each point in the plot is a different data mix. Reading the slide image, violation rates run from about 70% down to about 12–20%, while false refusals stay between about 1.7% and 3.2%. The caption adds that the 8B model needs a higher share of safety data than the 70B one to reach comparable safety.

::note slip 29:18 | The professor introduces this slide as the Llama 2 safety description, and the slide's footer says “few thousand in Llama 2”. The screenshot itself is from the Llama 3 report: its legend reads Llama 3 8B and Llama 3 70B, and the text cites 2024 work. The “few thousand” count is the Llama 2 figure; the plot and excerpt are Llama 3's.

::widget fixture:lecture_15--refusal-frontier | lower the refusal threshold and the point slides along one curve, trading violations for false refusals; only raising the separation d (telling harmful requests from look-alike benign ones) moves the whole curve toward the origin
::slide 24 | the Tulu 3 paper (“Pipeline with most details”) and its safety and non-compliance data: CoCoNot 10,983; WildJailbreak 50,000; WildGuardMix 50,000

The professor's best public reference for a full post-training pipeline is Ai2's **Tulu 3**, the recipe behind the OLMo models. It is one of the few places a reasonably strong pipeline can be seen end to end. Typical safety sets have a few thousand to tens of thousands of examples, and he puts its safety component at “something like 50,000-ish examples”. The slide's table lists three safety sets: CoCoNot with 10,983 examples, and WildJailbreak and WildGuardMix with 50,000 each.

::slide 25 | “Main safety approach – extract scenarios from users”: the WildChat paper (1M ChatGPT interaction logs), a non-compliance taxonomy, and the WildTeaming framework that mines in-the-wild jailbreak tactics and composes them into new attacks

Where do the prompts come from? Ai2 had earlier run **WildChat**, which gave people free chat access in exchange for their logs. From those logs they filtered out the unsafe requests and the jailbreak attempts, mined them at scale, and wrote the preferred response for each: resist the jailbreak, or say no. Closed labs appear to do the same with their usage data, finding unsafe behaviours and having annotators “play whack-a-mole” with them.

::predict safety-violation-vs-false-refusal
::kp safety-violation-vs-false-refusal

::slide 26 | Bianchi et al.: mean harmfulness score on four test sets as 0, 100, 300, 500, 1000, 1500 or 2000 safety examples are added to Alpaca-style training data; “Significant improvements to safety with ~500 samples”

Here is the result that surprises people who have not done SFT before. With a capable model, you need very few examples to steer it. Add a few hundred refusals of unsafe requests to the training data and the rate of following malicious or hateful instructions “drops dramatically”. The bars, read from the slide image, on the malicious-instructions test set:

| added safety examples | 0 | 100 | 300 | 500 | 2000 |
|---|---|---|---|---|---|
| mean harmfulness (roughly) | 2.9 | 1.85 | 0.75 | 0.3 | 0.15 |

So 500 examples remove about 90% of the harmfulness score, and the next 1,500 add little. The Anthropic HH test set (Q-Harm) levels off higher, at about 1.0 from 300 examples on.

The professor's explanation: after pretraining, models already have a “safe or unsafe” axis inside them, so a very small push is enough to pull it out. The other side of the coin is that being able to steer with few examples does not mean more examples are useless. A lab that wants “really fine grained distinctions about what is safe and not safe” still needs very large-scale data collection.

::video 32:14-33:40 | “as little as 500 examples”, the safety axis already present after pretraining, and why fine-grained safety still needs scale

::slide 27 | putting it together: SFT works best when extracting pretraining behaviours, not adding new ones; adding (factually correct!) data can sometimes hurt; small amounts of the right behaviour (safety, instruction-following, style) make a big difference, but a long tail benefits from more data

The three SFT lessons, connected:
1. SFT works best when the behaviour is already somewhere in the pretrained model and you only need to pull out the right mode.
2. Adding correct data can hurt, through the hallucination mechanism of the last part.
3. So you can often focus on quality over quantity, with the long-tail caveat.

A student asked how we know whether a behaviour is “already in pretraining”. The professor admitted we don't, directly. We can show what is *not* there; for example, SFT struggles to generalize to very rare programming languages. We cannot show that safety *is* there. “Extracting” is a working hypothesis.

::predict small-data-big-behavior-change
::kp small-data-big-behavior-change

## How do you run SFT, and why has it moved into pretraining? {#midtraining}
source: lecture_15.pdf p28-p30 · video 35:53-41:58

::slide 28 | “Just do gradient descent”: a ten-line PyTorch training loop (forward, loss.backward(), optimizer.step()); “But what if ... you have tons of compute and data, you want to scale up instruction tuning”

Here the method is the most boring part: SFT is gradient descent on the next-token loss. The professor put the training loop up “almost as a joke”: call `loss.backward()` and that is basically it.

::slide 29 | turning instruction tuning into pretraining: (1) pretrain on web data, (2) mix instruction-tuning data into pretraining, (3) do an actual but short instruction-tuning round; “Lets you scale up instruction tuning w/o catastrophic forgetting”

The one nuance worth a slide is what happens when you have lots of compute and instruction data. Pretraining and post-training used to be separate phases. Now high-quality data, including instruction data, is mixed in at the tail end of pretraining, during the **decay phase**. That is the final stretch of a warmup-stable-decay schedule, where the learning rate is annealed toward zero ([L11's WSD schedule](#/read/lecture_11)).

Why bother? A long SFT run on a large, narrow instruction set, after pretraining has finished, tends to overwrite what pretraining learned. This is **catastrophic forgetting**. Mixing the instruction data into pretraining lets you scale it up while the model still sees web data alongside it. It also lets you emphasize higher-quality data in general. Then a short, real SFT round sets the final chat behaviour.

::worked midtraining-two-phase

The professor says “everyone, as far as I know, is doing it”. Model reports call it mid-training, or a second phase of pretraining with a different data mix.

::note aside 37:09 | His “pet peeve”: calling such a model a base model is “kind of a lie”. A base model used to mean a model trained to predict the next word on internet text. Today's base models have been trained on UltraChat and other synthetic chat data designed to make them good at chat.

::slide 30 | MiniCPM's data mixture for the stable stage (CommonCrawl Chinese 25%, code 25%, Dolma 24%, C4 15%, Pile 8%) against the decay stage, where Wikipedia, SFT_mixed, Code_SFT, Knowledge_SFT, UltraChat, Stack Exchange QA and other SFT sets appear; “Publicized in recent Chinese-derived LMs (miniCPM, jetMoE)”

The recipe is common knowledge among LLM companies but rarely documented. MiniCPM and JetMoE are among the few that published their mixes. The two pies on the slide image show what changes at the decay stage. General web text (CommonCrawl, Dolma, C4, Pile) is 25 + 24 + 15 + 8 = 72% of the stable-stage mix and 14.6 + 15.7 + 9.5 + 5.1 ≈ 45% of the decay mix. The freed share goes to Wikipedia (6.7%), Stack Exchange QA, UltraChat and slices explicitly labelled SFT, such as SFT_mixed (4.8%) and Code_SFT (3.8%). The boundary between pretraining, instruction tuning and “high-quality data” is blurring.

A student asked whether the prompts are masked out of the loss in this mixed-in data. They are not; it is pure pretraining, so the model also predicts the prompt. The professor says the difference is small, since some SFT recipes also train on the prompt.

::predict midtraining-two-phase
::kp midtraining-two-phase

### Why the decay phase gets the best data
Another student guessed that the decay phase might get *lower*-quality data. The professor said the intuition runs the other way. The decay is “the most important part of your training”, for two reasons: it is the part of pretraining closest to deployment, and it runs at the lowest learning rate. Both point to putting the highest-quality data there. Why not make all of pretraining high quality? Because “you'll run out of tokens”. You cannot make Wikipedia your whole pretraining set.

The decay phase also doubles as a cheap test bench. Data mixing is “much more trial and error than you think”, and mixture-optimization algorithms are fairly unreliable. Mid-training, though, is much shorter than full pretraining: “something like 10 of these for each one”. So the usual practice, as he has heard it, is to run many data ablations on the decay phase, rank which domains help, and carry those estimates back into the pretraining mix.

**Worked example.** Every ablation branches from the same stable-phase checkpoint and costs only its own decay. If the decay is 10% of a full run, one run's compute buys 1 / 0.10 = 10 ablations. At 5%, it buys 1 / 0.05 = 20.

::note aside 41:35 | An example of how this works in practice came out in court: documents from the lawsuit over Meta's use of books show researchers running ablations to estimate how useful each books subset was.

::predict decay-phase-data-ablations
::kp decay-phase-data-ablations
::kp lecture_11:wsd-schedule-branching

## Why go beyond imitation? {#optimization}
source: lecture_15.pdf p31-p34 · video 34:57-35:53, 41:58-47:07

The second half of the lecture is RLHF. Instead of predicting the next word of given text, we now up-weight or down-weight the model's own outputs according to how good a rater, or a reward model, judges them. Before any algorithm, the professor stops on a conceptual difference he calls “quite important”.

::slide 32 | Imitation (SFT): fit p̂(y|x) ≈ p*(y|x) for a reference distribution p*; pure generative modelling; requires samples from the reference policy. Optimization (RLHF): find p̂(y|x) that maximizes E_p[R(y, x)] for a reward R(y, x); maximize a reward we can measure; LMs are policies, not a model of some distribution

$$ \text{SFT:}\ \ \hat p(y\mid x) \approx p^*(y\mid x) \qquad\qquad \text{RLHF:}\ \ \max_{p}\ \mathbb{E}_{y\sim p(\cdot\mid x)}\big[R(y,x)\big] $$

Pretraining and SFT are both generative modelling: there is a distribution of sequences and we fit it. SFT only changes which distribution. RLHF plays a different game, “a maximize a reward game”. The model is still a distribution over outputs, but now it is a **policy**, judged only by the reward its outputs earn. The reward might be how long users stay engaged, or how many math problems get solved. Nothing asks the policy to match any distribution.

The consequence is striking: the policy “can totally collapse out my distribution onto a single point for every input”. One answer per prompt, no diversity, and that is fine as long as the reward is high. This is the root of the mode collapse and lost calibration in the last part.

The columns also need different inputs: imitation needs demonstrations someone wrote, optimization needs a reward you can measure on the policy's own outputs.

::video 42:46-43:50 | “a fit a distribution game” against “a maximize a reward game”, and collapsing “onto a single point for every input”

Earlier, a student had asked whether SFT and RL are really so different. The professor said the lines are “very blurry”. Methods like expert iteration (sample, keep the good outputs, fine-tune on them) look like “SFT with extra bells and whistles”. The real distinction is the feedback. SFT is dense supervision on someone else's text. RL is self-taught supervision on “your own output from your own policy”, so the model may not drift as far from what it already does.

::slide 33 | “Why optimize? G-V gap”: Zhang et al. 2023, six freelance writers comparing their own news summaries with Instruct Davinci's (total 50.4% vs 49.6%, agreement α = 0.07); left, the DeepSeekMath-V2 paper, “Towards Self-Verifiable Mathematical Reasoning”

Why not keep collecting SFT data forever? Because people do not always write the thing they would prefer. The professor's group asked freelance writers to summarize news articles, then to compare their own summaries with Instruct Davinci's (the predecessor of ChatGPT). Some preferred the model's. The numbers are on the slide image: overall the writers' summaries won 50.4% to 49.6%, and annotator 1 preferred the model's summaries 57.0% of the time. When interviewed, the writers said things like “actually, this is pretty good stuff.” They were good writers whose summaries passed quality checks. They simply had not thought of the better way until they saw it.

People “aren't really these optimal systems”, so asking them to *rate* outputs can capture quality that asking them to *write* demonstrations misses. That is the **generation–validation (G-V) gap**: what people can recognize as better outruns what they write. A second reason, which the DeepSeekMath-V2 paper on the left points to, is that in some domains verification is simply much easier than generation. Checking a proof is easier than writing one, and DeepSeek has used models to verify their own proofs. That thread, RL from verifiable rewards (RLVR), is the whole next lecture.

So an SFT model trained on these writers' summaries can at best reproduce them, while a method that optimizes the raters' judgement has room to go past them. Imitation is bounded by its demonstrations; optimization is bounded by what the reward can measure.

::predict imitation-vs-optimization
::kp imitation-vs-optimization

::slide 34 | overview of the RLHF half: data (how people collect it, what to worry about), algorithms (PPO, DPO), side effects

The loop, as narrated: start from the SFT model, which already follows instructions. For each prompt, sample a few outputs at temperature 1; the SFT model is still diverse, so this gives real variety. A rater ranks them, sometimes as a simple binary choice. A reward model is trained on the rankings, and standard RL maximizes its score. Why go through a reward model at all? Because “it might be easier to train a verifier than to train a model that does well directly.”

::note deferred 45:54 | RLVR, RL on verifiable rewards such as math answers and unit tests, is “its own universe” and gets the entire next lecture ([L16](#/read/lecture_16)).

## How is preference data collected, and by whom? {#pairwise}
source: lecture_15.pdf p35-p42 · video 47:07-52:57, 57:59-59:07

::slide 35 | the InstructGPT figure again, now with step 2 boxed: sample several outputs, a labeler ranks them (D > C > A = B), train the reward model

::slide 36 | a standard pairwise annotation interface: an instruction (“Tell me about self driving cars”), AI Response 1 and AI Response 2, and four choices from “Response 1 is better” to “Response 2 is better”, with “only slightly better” options in between

The standard format is **pairwise feedback**. An annotator sees one prompt and two responses and says which is better, sometimes with a “slightly better” grade. Most places use an interface like this one.

::slide 37 | an excerpt of InstructGPT's labeling instructions: evaluate outputs for being helpful, truthful and harmless, with examples of each; “For most tasks, being truthful and harmless is more important than being helpful”

The InstructGPT appendix is the last glimpse of an industry lab's data collection process. Annotators rate outputs on three criteria:
- **helpful**: clear writing, answering the question the user meant to ask, sensitivity to internationality (“football” need not mean American football), and no overly long or rambling answers;
- **truthful**: no hallucinated details, no false claims about the world;
- **harmless**: no harm to people, and preferring outputs that decline questionable prompts.

The slide image adds the tie-breaking rule: for most tasks, being truthful and harmless matters more than being helpful.

::slide 38 | the leaked Google Bard annotation guideline: two responses rated side by side, with a helpfulness scale (from “Not at All Helpful” to “Extremely Helpful”) and a presentation scale (Poor, Adequate, Excellent)

The only other closed-lab guideline in public view leaked from Google Bard's annotators. Its criteria are similar (helpfulness, presentation, accuracy), but it grades each response on a Likert scale, a fixed set of labelled levels, rather than asking only which of two is better. Its closing advice is that a concise response presenting the most helpful information is usually better than a longer one, which pushes against the length bias of the style section.

::slide 39 | a survey of one platform's annotators (Outlier, Scale AI; n ≈ 910): ages (35–44 the largest group, 34%), education (bachelor's 44%, master's 32%, doctorate 9%), and main domains (language, creative writing, generalist projects, coding, mathematics, ...)

The picture has changed since then. The annotator workforce has shifted toward experts and higher pay. On this one platform, which is not representative of the whole market, 44% of annotators hold a bachelor's degree and 32% a master's, so 76% have a bachelor's or master's. The professor says “like 70%”. The largest age group is 35–44. The tasks are mostly creative and technical writing.

::slide 40 | “Large variation in compensation”: a Business Insider report on a Handshake AI project for OpenAI (at least 50 dollars an hour, 3,000–4,000 freelancers) and a chart of US specialist annotator pay by field, with highly skilled experts in law, engineering and medicine at about 115–120 dollars an hour

Beyond that there is a fast-growing market for bespoke expert annotators. Labs want to deploy models in white-collar jobs, so they want doctors and lawyers writing SFT data and judging responses. Median wages are above 50 dollars an hour across fields, and some experts get more than 100 dollars. If your mental model of an annotator is low-cost pairwise clicks from overseas, the professor says, that is not the full picture. But it is a pyramid: cheap, scalable annotation has not gone away.

::slide 41 | complexities of crowdsourcing: hard to get really high-quality, verifiable annotators; hard to get them to really check correctness; have to be careful about AI use

Why so expensive? Because good preference data is hard to get:
- **Verifiable annotators.** High-quality annotators whose work you can actually verify are scarce.
- **Checking correctness.** It is very hard to get truly correct judgements under time pressure. The Bard annotators said, as part of a labour dispute, that they had to check the correctness of long chat responses in under a minute.
- **AI use.** Anyone who has run surveys recently knows people paste in ChatGPT answers, and preventing it is “extremely, extremely difficult”. Then the “human” preferences are really that chatbot's.

A student asked whether experts are hired because they label more carefully, or because only they can answer. Mostly the latter, the professor said: you need a lawyer to check whether a Blue Book legal citation is correct. Platforms are also moving to better-qualified annotators for general tasks, partly because unsupervised annotators “will use the cheapest LLM to generate plausible-looking answers”; many annotation shops now sell verified human work as their product.

::video 51:24-52:16 | why verifiable annotators are hard to get: hidden AI use, and checking long answers “in under a minute”

::slide 42 | crowdsourcing ethics: TIME's report that OpenAI used Kenyan workers on less than 2 dollars per hour to make ChatGPT less toxic, and The Atlantic's “America already has an AI underclass”

Alongside the growth in expert work, there is still a large amount of low-paid annotation. Scale AI's early outsourcing “got into quite a bit of trouble”. Like the wider economy, annotation has split into high- and low-wage work.

::predict pairwise-feedback-crowdsourcing
::kp pairwise-feedback-crowdsourcing

## Does it matter who the annotators are? {#annotators}
source: lecture_15.pdf p43-p44 · video 52:57-57:59

Annotators have “a surprising amount of influence” over the model. Post-training is the final shaping step before a model ships, so whatever the annotator pool prefers, including its biases, ends up in the product.

::slide 43 | Santurkar et al. 2023: left, InstructGPT's labeler demographics (52.6% Southeast Asian, 31.6% white; nationalities led by Filipino 22%, Bangladeshi 22%, American 17%); right, how closely each model's opinions match each religious group, with base models (AI21's J1 models, ada, davinci) next to OpenAI's instruction-tuned text-davinci-001/002/003

The professor's group, with Percy Liang and a postdoc, asked language models standard opinion-poll questions and measured which human groups each model's answers were closest to. Base models were closest to Protestant and Roman Catholic respondents and further from Buddhist or Hindu ones. After post-training the ranking moved toward Buddhists, Hindus and atheists. From the slide image, with higher numbers meaning closer:

| group | davinci (base) | text-davinci-003 (post-trained) |
|---|---|---|
| Protestant | 0.788 | 0.694 |
| Roman Catholic | 0.794 | 0.700 |
| Buddhist | 0.764 | 0.709 |
| Hindu | 0.776 | 0.707 |
| Atheist | 0.761 | 0.713 |

Among these five groups, the base model is closest to Catholics, while text-davinci-003 is furthest from Protestants and closest to atheists. Why these groups? The InstructGPT appendix lists the annotators, shown on the left: largely Southeast Asian (Filipino and Bangladeshi nationals were the two largest groups) plus people from the US West Coast. Roughly, those are the demographics the models moved toward.

::note slip 54:51 | The professor adds that subtle preferences can pass through innocuous data: a model trained on data generated by an owl-loving model inherits the preference for owls. He calls this “emergent misalignment”, then “subliminal transfer” a sentence later. The owl result is subliminal learning (Cloud et al. 2025). Emergent misalignment (Betley et al. 2025) is a different result: broad misalignment after narrow fine-tuning on insecure code.

He also warns that the whole study of LLM political opinions “can be quite fragile and sketchy at times”. The broader point stands: who annotates changes what the model does.

::slide 44 | Hosking, Blunsom, Bartolo 2024: how much more or less often crowdworkers flag each error type than expert annotators, for baseline, more assertive and less assertive (and more or less complex) outputs; caption: annotators underestimate inconsistency and factuality errors, especially in assertive outputs

The second axis is expertise rather than demographics. Hosking et al. compared careful expert annotators with crowdworkers on a platform. The cells give the crowd's error-flagging rate minus the experts'. Formatting errors are flagged slightly *more* by the crowd (+3.1 points at baseline). Factuality errors are flagged far *less* (−16.2 points at baseline), and the gap widens to −22.3 points when the output is written assertively. Inconsistency behaves the same way (−10.6, widening to −16.9). Factuality is much harder to check, so non-experts don't check it, and a confident tone makes them check even less. This is the slide's “not all annotators”: the experts do weigh factuality.

**Worked example.** Out of 100 baseline outputs, the crowd marks about 16 fewer as containing a factual error than the experts do. Rewrite the outputs assertively and the shortfall grows to about 22. A reward model trained on crowd labels learns that confident, well-formatted text is good. The policy then optimizes that, which is how RLHF amplifies style.

::predict annotator-distribution-effects
::kp annotator-distribution-effects

### What makes an annotator good? Agreement measures variance, not bias
A student asked what makes one annotator higher quality than another. There is no mechanical test, the professor said, but he offered two notions.

The first is a sufficiently detailed guideline with semi-objective criteria. “It should be factual”, with factuality defined, perhaps as nothing in the first three pages of a Google search contradicting it. Then you can catch an annotator who isn't following the rule.

The second is **inter-annotator agreement**, and here he added a warning. Agreement is “a statement about a population of annotators”. It “doesn't tell you what the bias is. It tells you the variance.” Some tasks have variance by nature: for “do you like this?”, low agreement is no sign of sloppiness. And if the annotators are all using ChatGPT, agreement is perfect, because the variance is zero.

Slide 33 shows inherent variance in practice: the six writers' agreement was α = 0.07, close to none, although each was a careful professional.

::widget fixture:lecture_15--annotator-pool | raise the shared bias b and agreement rises while the share of correct labels falls; tick “all annotators copy one chatbot” and agreement hits 100% whatever the chatbot's accuracy

::video 57:23-57:59 | “it doesn't tell you what the bias is. It tells you the variance”, and the case where everyone uses ChatGPT
::predict agreement-measures-variance-not-bias
::kp agreement-measures-variance-not-bias

## Can a model replace the human annotators? {#ai-feedback}
source: lecture_15.pdf p45-p48 · video 59:07-1:05:23

Asked about domain-specific models as annotators, the professor split the question. Model-based annotation is a good idea: for catching up to the frontier it is “very, very good”, and modern models beat a random crowdworker. Domain-specific models have no special advantage, because they are rarely better than the strongest general open models.

::slide 45 | “GPT4 is a surprisingly good pairwise feedback system”: left, human vs simulated win rates across systems (Spearman correlation 0.98, R² = 0.87); right, agreement with the human majority label against cost per 1,000 examples, with human annotators near 0.66 at a few hundred dollars and GPT-4 near 0.645 at about 15 dollars

When GPT-4 came out, the professor's students compared its pairwise judgements with carefully collected human ones. Both levels look good, and the numbers are on the slide image:
- **System level.** Rank a set of chatbots by GPT-4's win rates and by humans': the Spearman rank correlation is 0.98.
- **Per pair.** GPT-4 agrees with the majority human label about as often as a single human annotator does, roughly 65% against 66%.
- **Cost.** It is “an order of magnitude less”. On the plot, human labels cost a few hundred dollars per 1,000 examples against about 15 dollars for GPT-4.

Note what “agreement near human inter-annotator levels” means, given the last part: GPT-4's disagreement with humans is about as large as humans' disagreement with each other. Per-pair agreement can never be high when humans themselves disagree about a third of the time.

::slide 46 | AI feedback in open post-training: UltraFeedback (GPT-4 rates responses from a pool of models), a Zephyr-7b interview about choosing AI feedback, and Tulu 3's pipeline (prompts, responses from 22 models, GPT-4o ratings on helpfulness, instruction following, truthfulness and honesty, binarized into chosen and rejected)

At the time it was unclear whether open models would go the expensive-human-data route or become distillation from stronger models. Years later the professor thinks the answer is clear: “there's basically no space for human collected data if all you want to do is to catch up to the frontier.”

Hugging Face's Zephyr is the instructive case. The team wanted no model distillation, so they bought human preference data from the same vendors the big labs use. It was time-consuming and costly, and the results were “not actually better” than model-based feedback, so Zephyr used AI feedback. (This was at 7B scale; the professor would be surprised if larger models changed the answer.) Now UltraChat for SFT and UltraFeedback for preferences are standard, and Tulu 3 uses model-based annotation for its whole pipeline. Pushing the frontier out is different. There you still rely on human data.

::slide 47 | Constitutional AI (Bai et al.): a helpful RLHF model answers red-teaming prompts, critiques and revises its own answers, and is fine-tuned on the revisions (SL-CAI); then the model judges pairs of its own samples to train a preference model, and RL against it gives the final RL-CAI model

Model-generated data need not be pure distillation from a stronger model. In Anthropic's Constitutional AI, a model was prompted to generate safety data and then trained on it, an early self-training loop for post-training. Self-Instruct is the capability-centred version of the same idea. The slide shows both stages: response, critique, revision, then supervised fine-tuning on the revisions; and then AI preference labels that train a preference model for RL (“RLAIF”, RL from AI feedback). One limit remains: world knowledge that only lawyers or scientists have cannot be bootstrapped, so for that you still need human annotators.

::predict ai-feedback-rlaif
::kp ai-feedback-rlaif

::slide 48 | “Length effects are a very significant outcome of RLHF”: left, Chen et al. 2024, model-judged win score against response length for ReMax, PPO, DPO and their length-penalized Odin versions, with GPT-3.5-turbo far above the trend; right, Singhal et al. 2024, reward against output length, and an SFT answer of 59 tokens that becomes 243 tokens after RLHF

Model judges are “very susceptible to the same kinds of biases as humans”, sometimes more. After model-generated feedback became common, studies showed you could push response length out and keep winning model-judged comparisons. On the left, nearly every method's win score climbs with length. GPT-3.5-turbo is the outlier above the trend, a model that is actually better rather than just longer. On the right, reward correlates with length. In Singhal et al.'s example, a 59-token SFT answer becomes a 243-token RLHF answer, 243 / 59 ≈ 4.1 times longer, with “similar output, but much longer”. One paper even showed you can RLHF on length alone and do well on many of these benchmarks.

::kp length-bias-preferences

## How does RLHF optimize? The reward model and PPO {#ppo}
source: lecture_15.pdf p49-p53 · video 1:05:23-1:09:02

::slide 49 | how do we RLHF: PPO, “the original and very finicky approach (the brief version)”, then DPO, “the new, very accessible approach”

With the data in hand, how do we use it? The goal is the RLHF column of slide 32 (repeated as slide 50): maximize the expected reward of the policy's samples. The professor calls this “baby reinforcement learning”. Each episode is one prompt and one response with one reward at the end, a bandit rather than true multi-turn RL, so the algorithms can be simple too.

::slide 51 | InstructGPT's RL setup: PPO on a bandit environment, a per-token KL penalty from the SFT model “to mitigate over-optimization of the reward model”, the value function initialized from the reward model, and objective (2); “..this is very innocuous looking”

InstructGPT's equation 2, readable on the slide image:

$$ \text{objective}(\phi) = \mathbb{E}_{(x,y)\sim D_{\pi_\phi^{\mathrm{RL}}}}\Big[r_\theta(x,y) - \beta\log\frac{\pi_\phi^{\mathrm{RL}}(y\mid x)}{\pi^{\mathrm{SFT}}(y\mid x)}\Big] + \gamma\,\mathbb{E}_{x\sim D_{\mathrm{pretrain}}}\big[\log \pi_\phi^{\mathrm{RL}}(x)\big] $$

Term by term:
- $r_\theta(x,y)$: sample a response from the policy and score it with the learned reward model.
- $-\beta \log(\pi^{\mathrm{RL}}/\pi^{\mathrm{SFT}})$: averaged over samples this is a KL divergence, which keeps the policy close to the SFT model so it doesn't “go too far and become degenerate”.
- The $\gamma$ term: ordinary language-model likelihood on pretraining data, mixed in to fix regressions on public NLP benchmarks (the “PPO-ptx” models). With $\gamma = 0$ the models are plain “PPO”.

The deck calls it “very innocuous looking”. The last part shows why the KL term is not a detail.

::note slip 1:06:31 | The professor says the KL keeps the policy close to “my pre-trained model”. In InstructGPT the KL is to the SFT model, as the slide's own text says (“a per-token KL penalty from the SFT model”); pretraining data enters only through the separate γ term.

::slide 52 | Stiennon et al.: the reward model starts from the supervised baseline plus a linear head, is trained with loss(r) = −E[log σ(r(x, y_i) − r(x, y_{1−i}))], and is normalized so reference summaries score 0 on average; the RL reward is R(x, y) = r(x, y) − β log(π^RL/π^SFT), and the KL term acts as an entropy bonus against collapse to a single mode

Where does $r_\theta$ come from? Stiennon et al. train a reward model as “a binary classifier on which of a pair of examples is better”, then hill-climb on it. The loss on the slide is the **Bradley–Terry** model of pairwise choice. Each response gets a scalar score, and the probability that the annotator prefers $y_w$ (winner) over $y_l$ (loser) is a sigmoid of the score difference:

$$ P(y_w \succ y_l \mid x) = \sigma\big(r(x,y_w) - r(x,y_l)\big), \qquad \mathcal{L} = -\log\sigma\big(r(x,y_w) - r(x,y_l)\big) $$

Two consequences follow from the difference. Adding the same constant to every score changes nothing, so pairs identify rewards only up to a shift. That is why the slide's reward model is normalized afterwards so the reference summaries average 0. And the gradient is the remaining error, $1 - P$, so pairs the model already separates by a wide margin stop teaching.

::worked bradley-terry-reward-model

::widget fixture:bradley-terry | drag the shared shift: P(chosen) never moves, because only r_w − r_l enters; widen the gap and the gradient −(1 − P) shrinks toward 0

Bradley–Terry also assumes one scalar per response, so preferences must be transitive. With A ≻ B and B ≻ C observed at 80% each, each rating gap is ln(0.8/0.2) = ln 4. The gaps add, so the model implies P(A ≻ C) = σ(2 ln 4) = 16/17 ≈ 0.94. A cyclic dataset (A ≻ B ≻ C ≻ A, each 90%) has no consistent ratings at all.

::widget fixture:lecture_15--bt-fit | with only A–B and B–C observed, the implied P(A ≻ C) follows by adding the two rating gaps; tick “C vs A also observed” and make the three rates cyclic, and the fit pulls all ratings together toward 50%

The slide's last paragraph gives the KL term two jobs: it acts as an entropy bonus that deters the policy from “collapsing to a single mode”, and it keeps outputs close to what the reward model saw in training.

::predict bradley-terry-reward-model
::kp bradley-terry-reward-model

::slide 53 | PPO at a conceptual level: Attempt 1, policy gradients ∇E[R(z)] = E[R(z)∇log p(z)] (“variances are too high”); Attempt 2, TRPO, maximize E[(π_θ/π_old)·Â] subject to E[KL(π_old, π_θ)] ≤ δ; Attempt 3, PPO, min(ratio·A, clip(ratio, 1−ε, 1+ε)·A)

How do we hill-climb the reward? PPO, which the assignment requires you to understand. The professor gives the “baby description” as three attempts:

1. **Policy gradients.** The reward is a score on sampled text, so you cannot backpropagate through it. The log-derivative trick moves the gradient inside the expectation: take the gradient of each sample's log-probability and weight it by its reward. It “really just looks like SFT, but with weighted examples”. The slide's complaint is variance. The professor's complaint aloud is cost: you must sample fresh outputs for every step, and sampling (inference) is slow while training is efficient.
2. **TRPO.** So roll out once and reuse the rollout for several steps. This is off-policy. Reuse is only safe if the policy does not move far from the one that sampled the data, or the reward estimates blow up. TRPO corrects with an importance-weight ratio $\pi_\theta/\pi_{\text{old}}$ and constrains the KL between old and new policy to at most $\delta$.
3. **PPO.** The constraint is hard to handle, so PPO replaces it with “a heuristic clipping thing”: clip the ratio to $[1-\epsilon, 1+\epsilon]$, which removes any incentive to move further.

::worked ppo-for-rlhf-brief

::widget fixture:ratio-clip | with a positive advantage, the clipped objective goes flat once the ratio passes 1 + ε, so nothing pushes the policy further; flip the advantage negative and the flat side moves below 1 − ε

::note deferred 1:08:50 | The clipping details, advantages and the per-token KL get a proper treatment in the next lecture ([L16](#/read/lecture_16)), along with GRPO, which the assignment implements.

::widget fixture:lecture_15--kl-tilt | raise β and the policy's bars fall back onto the reference outlines and the KL readout drops; follow the curve and the proxy reward only rises with KL while the gold reward peaks and falls, which is the region the KL term keeps the policy out of

::predict ppo-for-rlhf-brief
::kp ppo-for-rlhf-brief
::kp lecture_16:ppo-ratio-clip

## Can we skip RL? From workarounds to DPO {#dpo}
source: lecture_15.pdf p54-p61 · video 1:09:02-1:16:33

::slide 54 | “Can we get rid of PPO?” Avoid on-policy RL: SFT on pairs with [GOOD]/[BAD] control tokens; train only on preferred outputs; train a reward model, sample, train on the preferred outputs; train a reward model, sample 1024 outputs, take the best

PPO's equations already look “a little bit gnarly”, and on-policy RL (sampling fresh rollouts from the current policy during training) is finicky to run. For years people asked whether they could use pairwise data without it. The professor lists the reasonable attempts “so that you do not necessarily repeat them in your own research”, with verdicts:
- **Control tokens.** SFT on both responses, with [GOOD] prepended to the chosen one and [BAD] to the rejected one, then generate after [GOOD]. “It does not work.”
- **SFT on preferred outputs only.** “Also does not work very well.” It also throws away the contrast with the rejected response, which is the signal pairwise data carries.
- **Reward-model-filtered SFT.** Sample outputs, keep the ones the reward model prefers, fine-tune on them. “Does not work as well, although it does somewhat work.” It survives as a component, as Llama's loop on slide 59 shows.
- **Best-of-n.** Sample 1024 outputs and return the one the reward model likes best. It is not mentioned aloud. It changes no weights, and it costs 1024 generations per query.

::predict rlhf-without-rl-alternatives
::kp rlhf-without-rl-alternatives

::slide 55 | DPO, “RLHF without tears?”: get rid of the reward model and of on-policy machinery (rollouts, outer loops); take gradient steps on the log-loss of good outputs and negative steps on bad ones, appropriately weighted; RLHF (preference data → reward model ⇄ policy) against DPO (preference data → final LM, by maximum likelihood)

What finally worked is **DPO** (direct preference optimization). It removes the two complicated parts of PPO: the reward model, and everything on-policy. What is left looks like SFT. Raise the log-likelihood of the good response, lower the log-likelihood of the bad one (“negative SFT”), and weight the two steps appropriately. Getting the weighting right is the whole derivation.

::slide 56 | the RLHF objective max E[r(x, y)] − β·KL(π_θ ‖ π_ref); assume π is the set of all policies; the maximizer π_r(y|x) = π_ref(y|x)·exp(r(x, y)/β)/Z(x); solved for the “implied reward” r(x, y) = β log(π_r/π_ref) + β log Z(x); “the equivalence also used in the kimi-think paper”

Start from the KL-regularized objective, the same shape as InstructGPT's minus the γ term. Make **one strong assumption**: the policy is not a neural network but can be *any* distribution (the nonparametric assumption). Then the objective can be solved per prompt in closed form. The optimum takes the reference policy and **exponentially tilts** it by the reward: each response's probability is multiplied by $e^{r/\beta}$, so high-reward responses are scaled up and low-reward ones down. $Z(x)$ renormalizes:

$$ \pi_r(y\mid x) = \frac{1}{Z(x)}\,\pi_{\mathrm{ref}}(y\mid x)\,\exp\!\Big(\frac{r(x,y)}{\beta}\Big) \quad\Longrightarrow\quad r(x,y) = \beta\log\frac{\pi_r(y\mid x)}{\pi_{\mathrm{ref}}(y\mid x)} + \beta\log Z(x) $$

Turn it around and solve for the reward. Every policy now defines an **implied reward**: how much more likely it makes a response than the reference does, in units of $\beta$.

::note slip 1:11:55 | The professor says “the minimizer is the following”. The objective is maximized; the slide says “The maximizer is then”.

::slide 57 | substitute the implied reward into the Stiennon pairwise loss to get L_DPO = −E log σ(β log π_θ(y_w)/π_ref(y_w) − β log π_θ(y_l)/π_ref(y_l)); key steps: nonparametric assumption, parametrize the reward via the policy, optimize the reward with supervised losses; “MLE on the pairwise rewards”

Now plug the implied reward into the Bradley–Terry loss from slide 52. $Z(x)$ depends only on the prompt, so $\beta\log Z(x)$ is the same for both responses and cancels in the difference. That removes the intractable sum over all possible responses:

$$ \mathcal{L}_{\mathrm{DPO}} = -\,\mathbb{E}_{(x,y_w,y_l)}\log\sigma\Big(\beta\log\frac{\pi_\theta(y_w\mid x)}{\pi_{\mathrm{ref}}(y_w\mid x)} - \beta\log\frac{\pi_\theta(y_l\mid x)}{\pi_{\mathrm{ref}}(y_l\mid x)}\Big) $$

The slide's summary: it is maximum likelihood on the pairwise preferences, with the reward parametrized through the policy. Fitting that reward model *is* training the policy, with an ordinary supervised loss on a fixed dataset. There are no rollouts and no separate reward model. The equivalence to RLHF holds exactly only under the nonparametric assumption. With a real network the two methods can land in different places, which is slide 61's point.

::worked dpo-objective

::slide 58 | the DPO gradient: −β·E[σ(r̂(x, y_l) − r̂(x, y_w)) · (∇log π(y_w|x) − ∇log π(y_l|x))], annotated “higher weight when reward estimate is wrong”, “increase likelihood of y_w”, “decrease likelihood of y_l”

$$ \nabla_\theta\mathcal{L}_{\mathrm{DPO}} = -\beta\,\mathbb{E}\Big[\underbrace{\sigma\big(\hat r_\theta(x,y_l)-\hat r_\theta(x,y_w)\big)}_{\text{how wrong the implied reward is}}\big(\nabla_\theta\log\pi_\theta(y_w\mid x) - \nabla_\theta\log\pi_\theta(y_l\mid x)\big)\Big] $$

The professor finds this the most intuitive form. For every annotated pair, raise the winner's likelihood and lower the loser's, and scale the step by how wrong the implied reward model currently is. If it already gives the winner a much higher reward, take a small step. If it rated the two about equal, take a big one.

**Worked example.** At the start the policy equals the reference, so both implied rewards are $\beta\log 1 = 0$. The implied preference is $\sigma(0) = 0.5$, the loss is $\ln 2 \approx 0.693$, and every pair gets weight $\beta \cdot 0.5$, however likely or unlikely its responses are. Later, with $\beta = 0.1$, suppose the chosen response's log-ratio has fallen to −1.0 and the rejected one's to −4.0. The implied rewards are −0.1 and −0.4, the margin is 0.3, and $\sigma(0.3) \approx 0.574$. The model now prefers the winner, *although the winner became less likely than under the reference*. DPO raises the margin, not necessarily the chosen response's probability.

::widget fixture:lecture_15--dpo-margin | the presets “chosen +5 only” and “rejected −5 only” give the same loss, because only Δ_w − Δ_l enters; at “start” the weight is exactly β/2; “both lowered” has a loss below 0.693 although the chosen response lost likelihood

::widget fixture:dpo-implicit-reward | set log π_θ equal to log π_ref for both responses: the margin is 0 and the weight is β/2; push the rejected response's log π_θ down and the weight falls as the implied reward model becomes confident

::slide 59 | DPO in Llama: collected prompts → K generations per prompt from the best models of earlier rounds → rejection sampling with a reward model → SFT data → SFT model → DPO training → final DPO model, repeated; “DPO + Expert iteration for post-training”

Is DPO better than PPO? In the professor's view it “maybe ... doesn't matter very much, unless you're like at the frontier”, and DPO is “good enough for Llama”. Llama's post-training is an outer loop: SFT the model, run DPO, use the result to generate candidates, rejection-sample the best with a reward model, and repeat. The core RLHF primitive inside that loop is DPO.

::slide 60 | two variants from the Tulu 3 paper: SimPO, which drops π_ref, divides each log-probability by the response length |y| and subtracts a margin γ; and length-normalized DPO, which divides each β log-ratio by the response length

There are “too many variants”. SimPO changes the weighting: it drops the reference policy and normalizes by length. Length-normalized DPO divides by length “to avoid certain length hacking issues”, the length bias of the style section showing up inside the loss. The professor's verdict: “none of these variants maybe seem to matter very much.”

::slide 61 | “But PPO does too (and sometimes better?)”: left, Ivison et al., aggregate score rising from initial SFT 56.8 through DPO runs (58.1, 61.0) to PPO runs (62.2, 62.8, 62.4); right, Tulu 3's table: SFT base 55.7, SimPO 51.8–52.9, DPO 55.2, PPO 54.5–55.5, length-normalized DPO from 46.8 to 57.3 depending on settings

The results are highly contingent on the experimental setup. One Ai2 paper (left) finds that moving from DPO to PPO helps: 61.0 with DPO on better data, 62.2 after switching to PPO, 62.8 with a bigger reward model. Another Ai2 table (right) finds that well-tuned length-normalized DPO beats PPO, 57.3 against 55.5. That same method scores 46.8 with a different β, so the spread *within* one method is larger than the gap *between* methods. The takeaway: the core idea, a step toward the good response and a step away from the bad one, “works reasonably well as long as you set the step sizes right.”

::note slip 1:15:46 | The professor attributes the “DPO done right beats PPO” result to the Tulu 2 paper. The table the slide shows for it is from the Tulu 3 paper.

::predict dpo-objective
::kp dpo-objective

## What goes wrong when you optimize too hard? {#pitfalls}
source: lecture_15.pdf p62-p65 · video 1:16:33-1:19:48

::slide 62 | things to watch out for: left, overoptimization (Gao et al.: reward-model score against the KL distance between the RL policy and the initial policy, dashed proxy curves rising while solid gold curves peak and fall, for reward models from 3M to 3B parameters); right, mode collapse / entropy (an RLHF calibration plot)

When InstructGPT came out, there was a real question whether one could “RLHF our way to superintelligent systems”: just collect enough thumbs up and thumbs down. It turns out that pushing RLHF hard makes the policy overfit the learned reward model. This is **overoptimization**.

The left plot shows the shape. The x-axis is how far the policy has moved from where it started, measured as KL. The dashed **proxy** curves are the learned reward model's scores, and they keep rising. The solid **gold** curves are scores from a reference judge standing in for true preference, and they rise, peak and fall. From the slide image: with a 3M-parameter reward model, gold peaks around 0.55 and falls to about 0 by KL ≈ 95. With 1.2B–3B reward models it climbs to about 1.0 and is still roughly flat at KL ≈ 80. A better reward model postpones overoptimization but does not remove it.

This is why the KL regularizer from InstructGPT's objective is, in the professor's words, “really critical in a lot of cases in order to prevent your optimization process from overfitting your reward model, at least if your optimization process is very good.” The stronger the optimizer, the more it finds the reward model's mistakes.

::slide 63 | AlpacaFarm: win rate under the evaluation preference against the proxy reward, for expert iteration, best-of-n and PPO, with proxies trained on (a) human preferences, (b) noisy simulated LM preferences, (c) single-prompt GPT-4 preferences; “Holds true for human pref (left), noisy LM pref (mid) but not noiseless LM pref (right)”

This slide tests the mechanism. Here the x-axis is the proxy reward itself, and the y-axis is the win rate under the true preference. Reading the slide image:
- **(a) Human preferences.** PPO's win rate peaks around 0.55 and then drops. Best-of-n peaks around 0.51, then falls to about 0.46 as n grows.
- **(b) Noisy simulated LM preferences.** Same shape: peaks around 0.44–0.47, then a fall to about 0.39.
- **(c) A noiseless LM judge.** No peak. The win rate rises steadily to about 0.51.

It happens across optimizers, gradient-free best-of-n included, so the culprit is not PPO. It disappears only when the proxy can match the target exactly: a reward model trained on a deterministic judge has no error to exploit. Overoptimization is the reward model's error being found and amplified.

::widget fixture:lecture_15--kl-tilt | for both the KL-regularized optimum and best-of-n, the proxy curve rises all the way while the gold curve peaks and falls as mass piles onto the response the reward model overrates; the “noiseless judge” button makes proxy and gold one curve with no peak

::note skip 1:18:00 | The professor says “I'm going to skip this because we're at time” while on the side-effect slides, and the human/noisy/noiseless contrast is never said aloud. This part reads slide 63 from its image and caption.

::predict rlhf-overoptimization
::kp rlhf-overoptimization

::slide 64 | mode collapse: an RLHF model's calibration on MMLU (52B, 5-shot) at temperature 1 (far from the diagonal) and 2.5 (on it); GPT-4's calibration before RLHF (accuracy 0.82, ECE 0.007) and after PPO (accuracy 0.78, ECE 0.074); the entropy of models' answers to opinion questions, with text-davinci-003 piled up at zero entropy

The second side effect follows from slide 32. RLHF models “have much less diversity”: they concentrate on a few outputs. A model that is no longer modelling a distribution has no inherent diversity. In the professor's words, “It's a policy that can collapse as long as it gets a good reward.”

The plots show the cost. Calibration means stated probabilities match frequencies: of the answers given 70% probability, about 70% are right. The two top-right panels, from OpenAI's GPT-4 report, compare the model before and after RLHF. The pretrained model's calibration curve hugs the diagonal, with an expected calibration error (ECE) of 0.007. After PPO the ECE is 0.074, about ten times worse. OpenAI listed this as an open problem, and the professor doesn't think anyone has really solved it. The entropy histogram (Santurkar et al.) shows collapse directly: on opinion questions, text-davinci-003's answers sit overwhelmingly at entropy near 0.

**Worked example.** Read the post-PPO panel. Answers the model gives about 0.95 probability are right about 87% of the time, and answers it gives about 0.75 are right only about half the time. So if the RLHF model says 0.95 on each of 100 questions, expect closer to 87 right than 95, and you could not have known that without measuring: confidence and correctness have come apart. The top-left panel shows a partial fix from Anthropic's work: the same RLHF model's probabilities, softened with temperature 2.5, land back on the diagonal. In the professor's summary of that argument, RLHF models are naturally uncalibrated: “You could recalibrate sometimes, but not always.”

This matters more in the next lecture. RLVR needs entropy: the model has to keep exploring different solutions to make progress on hard problems, and a collapsed policy cannot explore.

::predict mode-collapse-calibration
::kp mode-collapse-calibration

::slide 65 | recap: RLHF data collection is (also) hard, with many confounding factors; RLHF algorithms are more complex than SFT, especially PPO; be mindful of (over)optimizing for rewards

The recap: post-training is messy because most of it is getting good data. RLHF algorithms are more complex than SFT, PPO especially; a simpler relative, GRPO, is what the assignment uses. The biggest problem with RLHF is overoptimization, and it sets up the next lecture's question: are there rewards we *can't* overoptimize, where you can keep adding compute and performance keeps improving? Verifiable rewards are why RLVR has been so impactful.

::video 1:19:22-1:19:45 | overoptimization as the bridge to RLVR
::kp lecture_16:verifiable-reward-vs-rlhf
