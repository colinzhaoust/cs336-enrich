---
title: L12 · Evaluation, read through
minutes: 40
---
The course has covered everything needed to train a language model; before it turns to the data you train on, this lecture asks what behavior you want, which means asking how you would know a model is good. It tours the main families of evaluation in the order the field met them (perplexity, exams, chat, agents, pure reasoning, safety), then steps back to the questions that cut across all of them: realism, validity and purpose. After it you can say what construct a given benchmark actually measures, compute perplexity and an Elo win probability, spot the classic ways a benchmark number misleads (length bias, contamination, broken items, scaffold differences), and state the rules of the game for an evaluation of your own.

## What does it mean for a model to be good? {#good}
source: lecture_12.py:L6-L12, L33-L57 · video 0:05-5:16

So far the course has built the architecture, the optimizer and the training loop, made training fast with kernels and parallelism, fitted scaling laws and made inference fast. The missing piece is **data**, and data shapes behavior: train on code and the model is good at code; train only on DNA sequences and it "probably can't speak English". So before choosing data you need to know what behavior you want, and that is evaluation: given a model, how good is it?

Evaluation looks mechanical. Write some prompts, send them to the model, compute accuracy. The professor's point is that it is neither simple nor neutral: evaluation "sets North Stars". Every developer, open or closed, reads benchmarks as the measure of progress, so whoever designs the evaluation implicitly shapes what the next models will be able to do.

::code lecture_12.py:L34-L42 | the three mechanical steps, then the real problem: abstract construct → concrete metric

The hard part has a name. You start with an **abstract construct**, something like "good at conversation" or "good at reasoning", and you have to turn it into a **concrete metric**, computed from concrete prompts or environments. Every benchmark in this lecture is one such translation, and most of its failures are failures of the translation.

To make that tangible, the lecture lists four answers to "what makes a model good?", each with a real leaderboard behind it.

1. **It does well on benchmarks.** [Artificial Analysis](https://artificialanalysis.ai/) combines many datasets into one intelligence index and ranks models by it.
2. **It does well on benchmarks and is cheap to run**, because cost matters. Plot the same index against the dollar cost of running it.
3. **People prefer its responses.** [Arena AI](https://arena.ai/leaderboard) (formerly Chatbot Arena) ranks models by human votes; it gets its own section below.
4. **People choose to use, and pay for, it.** [OpenRouter](https://openrouter.ai/rankings) routes requests to many models through one endpoint, so it sees which models people actually spend tokens on. This is more of an economic lens. As he put it, "if people are paying for it, it must be good."

::figure official/lectures/images/artificial-analysis.png | notion 1: one aggregate benchmark index, one ranked list
::figure official/lectures/images/artificial-analysis-cost.png | notion 2: the index against the cost to run it (log scale); the shaded quadrant is high score at low cost
::figure official/lectures/images/openrouter.png | notion 4: token volume by model, revealed preference rather than a test

On the cost plot the professor reads only the pattern: the two axes are correlated, "the more you pay to run, the higher the intelligence index, but it's not exactly aligned". Models of similar score can differ several-fold in cost.

These four need not agree, and each answers a different question. A support-bot buyer who picks the top of the intelligence index has silently chosen notion 1, when their real question is closer to 2 or 3. "None of those are necessarily the correct answer"; the lecture comes back to this at the end, when it asks what evaluation is *for*.

::predict construct-to-metric
::note aside 4:43 | OpenRouter's statistics are "not representative of all model usage", only of what people route through OpenRouter. A usage ranking is a sample from one population of users.
::note deferred 0:38 | Training data, the "missing piece", is next week's topic; this lecture is its prerequisite.
::kp construct-to-metric

## What does perplexity measure, and how did its use change? {#perplexity}
source: lecture_12.py:L60-L79 · video 5:16-9:46

The most natural evaluation follows from what a language model *is*: a probability distribution $p(x)$ over sequences of tokens. A distribution is judged by how much probability it gives to real data. Take a test dataset $D$ of $|D|$ tokens; **perplexity** is

$$ \text{PPL}(D) = \left(\frac{1}{p(D)}\right)^{1/|D|} = \exp\!\Big(-\frac{1}{|D|}\sum_{i=1}^{|D|} \log p(x_i \mid x_{<i})\Big) $$

The first form is the lecture's; the second, the exponentiated average negative log-likelihood per token, is how it is computed in practice (Assignment 1's cross-entropy loss is the log of it). The $1/|D|$ normalization is there "so that the numbers are more interpretable": it turns a vanishingly small probability of a whole corpus into a per-token number. A perplexity of $k$ means the model is, on average, as uncertain as if it were choosing uniformly among $k$ tokens. Lower is better; perplexity, likelihood and log loss are "all kind of related concepts", the same information on different scales.

::code lecture_12.py:L61-L66 | a model is p(x); perplexity asks how much mass it gives D; training minimizes it on train, so measure it on test

### The classic paradigm: in-distribution

In pre-training you minimize perplexity on the training set, so the obvious evaluation is perplexity on a held-out test set. Through the 2010s that is what language-modeling papers did, on a few standard datasets:
- **Penn Treebank** (PTB), Wall Street Journal text;
- **WikiText-103**, Wikipedia;
- **One Billion Word Benchmark** (1BW), from WMT11 machine-translation data (EuroParl, UN, news).

Each comes with its own train and test split: **in-distribution evaluation**. Progress was measured purely in perplexity reduction. A famous 2016 paper applied CNNs and LSTMs to 1BW and took the perplexity from 51.3 to 30.0 (the paper's numbers, quoted in the lecture). At a time when people still debated n-gram and hybrid models, it was "the first definitive result that showed just pure neural is clearly the way to go".

::code lecture_12.py:L68-L73 | three standard datasets; train on the train split, test on the test split

### GPT-2: out-of-distribution evaluation

GPT-2 (OpenAI, 2019) changed the game. It trained on **WebText**, 40 GB of text from websites linked from Reddit, and then evaluated **zero-shot** on the standard datasets, without ever training on their train splits. That is **out-of-distribution evaluation**: the question is no longer "did you model this dataset" but "does what you learned from a large general corpus transfer to it?"

::figure official/lectures/images/gpt2-perplexity.png | four model sizes against the in-distribution state of the art; compare the PTB column (transfer wins) with the 1BW column (it does not)

The professor read one cell aloud: on PTB, a tiny dataset, the largest GPT-2 (1.5B parameters) reaches about 35 perplexity against a state of the art of 46. On large datasets like 1BW the result reverses: with enough in-distribution training data, the dataset-specific models stay ahead. The mechanism is simple. A small dataset's own train split is too small to learn much, so a large general corpus has a lot to add; a large dataset's own train split already covers its test distribution.

::code lecture_12.py:L75-L79 | WebText, zero-shot, and the reading: helps on small datasets, not on large ones

Back then this setup was "a bit of a novelty"; today, training on one big corpus and evaluating on many standard benchmarks is the norm for the whole field.

::note slip 8:53 | Right after the PTB example he calls it impressive "given that this model was not trained on One Billion Word's data set at all". The example was PTB; the dataset name is a slip.
::note aside 9:05 | A caveat the code lacks: GPT-2's zero-shot gains may partly be overlap, "because I don't know if they did a careful train test decontam." Contamination returns in the validity section.

### A trap the lecture does not mention: perplexity is per token

$|D|$ counts tokens, so perplexity depends on the tokenizer. Two models that give the same text exactly the same total probability, but cut it into different numbers of tokens, report different perplexities. Say model A uses 1000 tokens and reports perplexity 8. Model B's coarser tokenizer uses 500 tokens for the same text, with the same $p(D)$. Then $1/p(D) = 8^{1000}$, and B's perplexity is $8^{1000/500} = 64$. The models are equally good; only the denominator changed.

The fix is to normalize per byte (or per character) instead of per token, using the bytes-per-token **compression ratio** from [L1's tokenizer section](#/read/lecture_01). This is our addition, not the lecture's: the professor never raises it, but it matters whenever you read perplexities from models with different vocabularies.

::predict perplexity-definition
::widget fixture:perplexity-per-byte | hold the text's total log-probability fixed and change only the tokenizer: per-token perplexity moves, per-byte perplexity stays put
::kp perplexity-definition

## Is perplexity all you need, or more than you need? {#perplexity-limits}
source: lecture_12.py:L81-L90 · video 9:46-13:06

### The faith argument

The professor presents, half seriously, the argument that "perplexity is all you need". He labels it "more faith than science" and warns not to "take this too seriously", but calls it a mindset that drives a lot of language-modeling research.

There is a true distribution $t$ of text, and we train a model $p$. The best you can do is match it: the cross-entropy of $p$ against $t$ is smallest, equal to the entropy $H(t)$, exactly when $p = t$. And a model equal to the true distribution can do anything text can express: condition on a problem and generate the solution, $p(\text{solution} \mid \text{problem})$; condition on a question and generate the answer. So, the argument concludes, pushing perplexity down will eventually "reach AGI".

::code lecture_12.py:L81-L85 | true distribution t, best perplexity at p = t, then every task is a conditional sample

Why bother with an argument he does not defend? Because it was "a strong driver of many people who have been scaling and scaling language models at a time when, maybe, the gains weren't as obvious". Before GPT-3 it was not clear language models would matter; what kept people going was the belief that if you drove perplexity down, good things would happen.

Note what the argument does *not* say: it says the perfect model solves tasks, not that any finite reduction in perplexity buys task ability, and not that $p = t$ is reachable.

::note slip 10:14 | Line 83 and the spoken version both say the best possible *perplexity* is $H(t)$. Strictly, the minimum cross-entropy is $H(t)$; the minimum perplexity is $e^{H(t)}$ (or $2^{H(t)}$ with $H$ in bits). The argument is unaffected; the units are off by an exponential.

### The counter-argument: perplexity charges for every token

Perplexity may be *more* than you need. Take the sentence "Stanford was founded in 1885". Perplexity scores the prediction of every token. The probability of *1885* given the rest is interesting: it is a question-and-answer item "phrased as sentence completion", and it tests world knowledge. But *founded*, or the first word of the sentence, carries little of interest. "And perplexity doesn't care. It's going to charge you bits for every bit of a deviation from the true distribution."

The fix is **conditional perplexity**: condition on a prompt and score only the response,

$$ \text{PPL}(\text{response} \mid \text{prompt}) = \left(\frac{1}{p(\text{response} \mid \text{prompt})}\right)^{1/|\text{response}|} $$

so the tokens you consider incidental move into the prompt and leave both the product and the normalization.

::code lecture_12.py:L87-L90 | the 'founded' example and the conditional fix
::note slip | Line 90 writes the conditional version as p(response | prompt)^(1/|response|), with the exponent on p rather than on 1/p. Read together with line 62 it means the same quantity, written above with the reciprocal; as printed, larger would be better, the opposite of perplexity.

Does the choice of scored tokens matter in practice? Enough to reverse a ranking. Try it before reading the worked numbers.

::predict perplexity-sufficiency-vs-conditional
::worked perplexity-sufficiency-vs-conditional
::widget fixture:lecture_12--cond-perplexity | drag the prompt boundary: on the whole sentence X's cheap filler tokens win it the lower perplexity; once only '1885' is scored, Y wins, 2.5 against 10

Model X is confident about the filler and unsure of the date; Y is the reverse. On the whole sentence X wins ($160^{1/5} \approx 2.76$ against $640^{1/5} \approx 3.64$); scored on *1885* alone, Y wins (2.5 against 10). Same formula, different tokens, opposite verdict. Where to put the prompt/response boundary in free-running text is a choice the lecture leaves open, and it is exactly this choice that decides what the metric measures.

::kp perplexity-sufficiency-vs-conditional

## Which benchmarks are perplexity in disguise, and what must a perplexity leaderboard trust? {#disguise}
source: lecture_12.py:L92-L105 · video 13:06-18:14

Some benchmarks report accuracy but are, underneath, next-token prediction: "perplexity in disguise".

**LAMBADA** (2016) is a cloze task, fill in the blank. Given a passage, predict the final word of the target sentence. Accuracy is the reported number, but "it is really a next-token prediction problem". What makes it sharper than plain perplexity is the choice of positions: the target words were "carefully chosen" so that you need the long context to resolve them; the target sentence alone is not enough. Scoring only those positions focuses perplexity on long-range dependencies, which is why the early GPT papers "latched on to" it: they believed long-context modeling was a key to reasoning.

::figure official/lectures/images/lambada.png | in each passage the target word (miscarriage, Gabriel, chains) can only be recovered from earlier sentences, not from the target sentence

**HellaSwag** is "multiple choice, but it's really sentence completion". A context ("A woman is outside with a bucket and a dog. The dog is running around... She...") is followed by four candidate endings, and the model must pick the one that continues it best. Scoring each ending by the probability the model assigns to it is a perplexity computation: "not exactly perplexity, it's multiple choice. But in some ways, it's perplexity."

::figure official/lectures/images/hellaswag.png | two items, from ActivityNet captions and WikiHow; the wrong endings come from adversarial filtering, so they read as fluent but wrong

::code lecture_12.py:L92-L96 | cloze (LAMBADA) and multiple-choice completion (HellaSwag)

### The warning: two contracts

Suppose you are sold on perplexity and launch a perplexity leaderboard. What is the contract? People submit an `LM`; you send it the test data and it returns `log_prob = LM(test_data)`. Now you must **trust that the probabilities are valid**, that they sum to 1 over the vocabulary. Otherwise "I could just implement LM, and it always returns 1 or, I guess, logprob 0", and get a perfect perplexity from something that is not a distribution at all. Checking that a submission internally computes a real distribution is not easy: "Maybe you have to look at the code or something."

Downstream tasks have a different contract. You send `prompt`, get back `response = LM(prompt)`, and grade the response. The model is a black box: there is no number of its own for it to inflate.

::code lecture_12.py:L98-L101 | log-prob scoring needs valid probabilities; generation only needs the response
::predict logprob-vs-generation-benchmarks
::widget fixture:lecture_12--mc-scoring | the same four option probabilities give two scores: log-prob scoring takes the argmax (0 or 1), sampling the letter earns only the key's sampled share; below, a constant-probability cheat already sums past 1 over two tokens

The two contracts also give different numbers for the same model on the same multiple-choice item. Log-prob scoring compares the options' probabilities and is right whenever the correct option is the most likely. Generation samples an answer, so with option probabilities 0.4, 0.35, 0.15 and 0.1 on a correct option A, log-prob scoring scores 1 while sampling at temperature 1 is right only 40% of the time. A multiple-choice number is only comparable within one scoring mode.

::note aside 16:57 | For models that give only a bound on the likelihood (VAEs, for instance) a perplexity leaderboard must also "trust the math" that the bound is valid. Mentioned and dropped.
::note spoken 31:27 | Asked later how an exam answer is compared with the key, he describes the generation route: the model samples a letter or, "more commonly now", writes a chain of thought from which an answer is extracted, and "language model evaluations can be very sensitive to that" extraction. He then skips the topic.

### Summary of perplexity

Perplexity is still used heavily in language-model development. It is cheap, it needs no labels, and it is "smoothly varying with scale, which is important for getting scaling laws" (see [L9](#/read/lecture_09)). But for "the non-believers" you still need benchmarks that capture real-world situations: "For the believers, if I show you a remarkably [low] perplexity, they'll be convinced." The rest of the lecture is for the non-believers.

::code lecture_12.py:L103-L105 | perplexity for development and scaling laws; benchmarks for everyone else
::kp logprob-vs-generation-benchmarks

## Why test a model with exams? MMLU {#exams}
source: lecture_12.py:L108-L120 · video 18:14-22:03

Exams are how we test humans, and the same mentality works for language models. Exams have two useful properties:
- **control** over the subject and the difficulty;
- **unambiguous correct answers**, so they are easy to grade.

Much of the benchmarking culture of language models grew out of this idea.

::code lecture_12.py:L109-L111 | exams: control over subject and difficulty, easy to grade

**MMLU**, Massive Multitask Language Understanding (Hendrycks et al., 2020), is the canonical one: multiple-choice questions across 57 subjects (math, US history, law, morality and so on), "collected by graduate and undergraduate students from freely available sources online".

It was ahead of its time. It appeared around GPT-3, when many people "still thought of language models as, well, language models": machines that generate fluent English, not general task solvers. Building a broad exam for them was a bet that they would become more.

Two things to know about it:
- **Despite its name, it tests knowledge, not language understanding.** The questions are exam items in subjects; fluency alone does not answer them. (Aloud: "knowledge and reasoning and less so just pure language understanding".)
- **It was evaluated on GPT-3 with few-shot prompting.** The prompt reads, for example, "The following are questions about high school mathematics", then a few in-context question-answer pairs, then the test question. That looks mundane now, but at the time it was "pretty radical" to expect a language model to do anything sensible with it.

::code lecture_12.py:L113-L117 | 57 subjects, multiple choice, collected online; knowledge, not language understanding; GPT-3 few-shot
::figure official/lectures/images/mmlu.png | left, the few-shot prompt; right, GPT-3 by size: on commonsense and linguistic benchmarks even the small models score 60+, on MMLU (blue) they sit at chance (25%) and only the largest rises clearly above it. Knowledge, not language, is what MMLU separates

At the start the small models were "barely above chance" (25% on four options) and GPT-3 "well above chance". Since then the benchmark "has essentially saturated": reading a public tracker (llm-stats), the professor showed scores going from GPT-3.5 Turbo to GPT-4 to, now, the 90s.

::predict mmlu-exam-benchmark
::note aside 21:48 | The [HELM MMLU viewer](https://crfm.stanford.edu/helm/mmlu/latest/) shows individual predictions, which he recommends browsing: it is "always fun to see what models actually goof up on".

The way MMLU's questions were collected, from freely available online sources, also makes it a natural candidate for the contamination problem below: anything on the open web may be in a model's training data.

::kp mmlu-exam-benchmark

## What happens when exams saturate? MMLU-Pro, GPQA, HLE {#harder-exams}
source: lecture_12.py:L122-L152 · video 22:03-31:52

A pattern you will see again and again: a benchmark gets too easy, so someone builds a harder one. The three examples here use three different levers.

### MMLU-Pro: change the format

Around 2024, with MMLU nearly solved, MMLU-Pro made three changes:
1. **removed noisy and trivial questions** from MMLU;
2. **expanded 4 choices to 10**, which lowers the chance score from 25% to 10% and gives each question more distractors;
3. **evaluated with chain of thought**, which "gives model more of a chance".

The third is not a hardening at all; it raises scores. By then chain of thought was "much more in vogue", and some questions "could benefit from a few reasoning steps", whereas in the GPT-3 era it "wasn't really a thing". The paper reports that model accuracy dropped by 16% to 33% relative to MMLU, so the benchmark was "not as saturated". Within two years it was saturating too: the professor read the current top near 90.

::code lecture_12.py:L122-L126 | three edits to MMLU; accuracy drops by 16-33%
::figure official/lectures/images/mmlu-pro.png | left: the same models score lower on MMLU-Pro; middle: scores vary less across prompt styles; right: chain of thought helps far more on MMLU-Pro than on MMLU, a sign its questions need reasoning steps
::note slip 22:55 | "Now the accuracy of the model is back at 33" probably merges two numbers: the code says accuracy dropped *by* 16% to 33%. If he was reading the tracker's early score instead, it is a value off a chart; either way 33 is not an accuracy the code states.

### GPQA: make it Google-proof

GPQA (Graduate-Level Google-Proof Q&A) starts from an observation: if a question can be solved by searching Google, and language models are trained on the internet, it is probably too easy. So the questions should resist search, and really hard questions take "quite a bit of human labor". They were written by 61 PhD contractors from Upwork and passed through a pipeline: a writer drafts the question; an expert answers it and gives feedback; the writer revises; a second expert answers; then non-experts (experts in *other* fields) try it with Google.

::figure official/lectures/images/gpqa.png | the pipeline on one chemistry question; bottom right, the Diamond rule: both experts agree, and at most one of three non-experts gets it right

The numbers, read off the paper:
- PhD experts in the field: **65%** accuracy. "This is not a good grade if you're getting 65% on an exam."
- Non-experts with 30 minutes and Google: **34%**, "a bit over chance" on what is mostly four-way multiple choice (chance 25%). That gap is what "Google-proof" means operationally.
- GPT-4 at release: **39%**.

Leaderboards often report the **Diamond** subset, the questions that passed the strictest filter. GPQA, too, has since saturated: on the live tracker the professor read 94, far above the experts. "These benchmarks have a shelf life of not too high." They remain useful, though, for developing smaller models and fitting scaling laws.

::code lecture_12.py:L131-L136 | 61 PhD writers; experts 65%, non-experts with Google 34%, GPT-4 39%
::note slip 24:44 | In the captions the Diamond criterion comes out garbled ("if the two experts agree and at one of the non-experts is able to answer"). The paper's rule, printed in the figure, is that both experts agree and at most one of three non-experts answers correctly.

### Is the test set in the training data?

A student asked the obvious question: how do we know these questions are not in the training data? "The short answer is we don't know, because I don't know what's in the training set." Take the numbers "with a little bit of a grain of salt, because it's only trust." Contamination is also subtler than training on the test set: labs probably do not literally do that, but "questions can be derived from other sources. And those sources could be trained on." He deferred the remedies to the validity section.

::note deferred 26:37 | "I'll come back to this contamination point a little bit later and ways of addressing it": see the validity section below.

### Humanity's Last Exam: filter by what models get wrong

HLE ("ominously" named) tries to "give it all we can to create something that's really tough": 2500 questions, multimodal, many subjects, multiple choice plus short answer. It was crowdsourced with incentives for both kinds of contributor: a prize pool of 500,000 dollars "for people who were into money" and co-authorship for "those who are into that". Candidate questions were **filtered by frontier LLMs**, kept only if current models got them wrong, then went through multiple stages of expert review.

::figure official/lectures/images/hle-examples.png | sample questions: specialist, often multimodal, short-answer or multiple choice
::figure official/lectures/images/hle-pipeline.png | the funnel: submissions first pass an LLM difficulty check, then expert review, then approval; the survivors are split into a public set and a private set

Note the private set. HLE kept part of the benchmark unreleased so that it cannot enter training, though you still have to send the questions to a model's API to evaluate, and "let's hope" those prompts do not end up in training data.

Filtering by frontier models makes the difficulty relative to *those* models: a score on HLE says a model solves questions that the filtering models missed, not what fraction of a subject it knows.

::figure official/lectures/images/hle-results.png | the usual dataset-paper picture: old benchmarks near the top, the new one in single digits

"Every data set paper looks like this": previous benchmarks, look how well models do; my dataset, look how badly. HLE still has room, the professor said; on the live tracker even the strongest model he showed was in the mid-60s.

::code lecture_12.py:L140-L144 | 2500 questions, a 500K-dollar prize pool and co-authorship, filtered by frontier LLMs, multiple stages of review

### What exams can and cannot do

- **The trend is towards harder questions** as models improve and saturate the old ones.
- **Multiple choice can be as difficult as you want.** "I can give you a very hard multiple-choice question." Its real limitation is different: multiple choice "restricts the set of questions you can ask".
- **Exams do not capture real usage.** "No one asks HLE questions to a language model except for when you're evaluating on HLE." Real questions are open-ended, often have no single correct answer, and "might not be well formed".

Difficulty and realism are separate axes. A harder exam is still an exam.

::code lecture_12.py:L149-L152 | harder over time; multiple choice can be arbitrarily hard; it still misses real usage
::predict harder-exams-saturation
::kp harder-exams-saturation

## How do you evaluate an open-ended answer? Chatbot Arena {#arena}
source: lecture_12.py:L155-L182 · video 31:52-38:21

Exams still capture some notion of intelligence and difficulty, but "most people don't ask a multiple-choice exam question to their AI assistant". A real request looks like this: *I would like to make a beet salad with goat cheese. What kind of herbs would work well and what would not?* The response is a page of open-ended advice. You cannot check it by exact match; "there's no ground truth even". How do you score it?

::code lecture_12.py:L159-L163 | the beet-salad prompt and the challenge: evaluating an open-ended response

### The mechanism

**Chatbot Arena** (now Arena AI) asks humans, but in a clever way.
1. Anyone on the internet can come to the site and type a prompt.
2. They get responses from **two** random models, anonymized as assistant A and assistant B.
3. They vote: A is better, both are good, both are bad, or B is better.

::figure official/lectures/images/arena-beets.png | the beet-salad prompt in the Arena interface: two anonymous answers side by side, and the vote buttons below

This yields a large pile of pairwise outcomes: model X beat model Y on some prompt. To turn them into a ranking, Arena borrows **Elo ratings** from chess. Each model gets a rating, and the probability that A beats B is a smooth function of the rating difference:

$$ p(A \text{ beats } B) = \frac{1}{1 + 10^{(\text{ELO}_B - \text{ELO}_A)/400}} $$

The ratings are the parameters, fitted to **maximize the probability of the observed comparisons** (maximum likelihood).

::code lecture_12.py:L171-L173 | the Elo win probability, fitted by maximum likelihood to the pairwise votes

What the scale means, by evaluating the formula:

| rating gap | p(higher-rated model wins) |
|---|---|
| 0 | 0.50 |
| 100 | 0.64 |
| 200 | 0.76 |
| 300 | 0.85 |
| 400 | 0.91 (exactly 10/11) |

Two properties follow from the formula depending only on the *difference*. First, shifting every rating by the same amount changes no prediction, so a rating is meaningful only relative to the pool; 1300 is not an absolute quality. Second, gaps add: if A is 100 above B and B is 200 above C, the fit puts A 300 above C and predicts A wins 85% of A-vs-C votes, even if they never met.

This is the same model as the **Bradley–Terry** reward model of [L15's RLHF section](#/read/lecture_15), written in another base: $10^{x/400} = e^{x \ln 10/400}$, so an Elo gap of 400 is a logit gap of $\ln 10 \approx 2.30$. L15 uses it to learn a reward from preferences; here it ranks models.

::predict arena-pairwise-elo
::widget fixture:bradley-terry | slide both scores together and p does not move; only the gap matters (the Elo curve is this sigmoid with the gap rescaled by ln10/400)

::figure official/lectures/images/lmarena-leaderboard.png | the resulting leaderboard, one rating per model; the top ten sit within a few dozen points of each other, which by the table above means head-to-head votes close to a coin flip

::note aside 33:38 | The interface collects ties ("both are good, both are bad"), but the formula at line 172 has no tie term; how ties enter the fit is not discussed.

### What Arena buys

- **Real-world prompts.** The site gives free access to strong models, so people come to get something done; "there's this implicit assumption that they're actually trying to use it to do something useful."
- **No need to feed every model the same prompts.** This matters because humans do the rating: "you can't expect a human to rate all the models or even more than two models."
- **Dynamic.** New prompts and new models arrive continuously, and the ratings have "a natural story" for updating over time.

The second point rests on a property the professor stated only aloud, "the beauty of ELO". In chess, not everyone plays everyone; a sparse sample of games suffices, and "you can still derive rankings as long as the graph is connected". Draw models as nodes and battles as edges. Within a connected graph, every rating difference is pinned down through chains of battles. Two groups that never meet, directly or through intermediaries, have no defined offset: you could add 200 to every rating in one group and every observed vote would be explained exactly as well.

::predict elo-needs-connected-comparisons
::widget fixture:lecture_12--elo-graph | with two separate pools the offset slider moves one component freely while the likelihood stays put; switch on any one cross battle and every node joins one scale
::video 37:42-38:02 | "you can still derive rankings as long as the graph is connected"

(Connectivity is necessary, not quite sufficient: if one side wins every battle on a bridge, the fitted gap can run off to infinity. The lecture stays at the level of connectivity.)

### What Arena conflates

- **Who are these people?** A "random person on the internet who comes to" Arena is an unknown distribution. The paper reports demographics, "but demographics don't tell the whole story". There may be biases, spammers, even people who submitted a model and "want their model to look good". "It's a little bit of a Wild West."
- **A binary preference conflates style and correctness.** Elo suits chess because "the only thing that matters is, did you win?". Which of two chat answers is better is far less clear cut.
- **Can the rater judge correctness?** The person who asked has an intent and can say whether it was met, which is good. But "they're presumably asking the question because they don't know the answer". And there is **sycophancy**: pleasing answers may beat correct but honest ones.

These are raised as questions, not findings; the lecture cites no measurement of them for Arena.

::code lecture_12.py:L176-L182 | real prompts, unknown raters, style vs correctness, sycophancy; no shared prompts needed; dynamic
::note aside 36:01 | Rater demographics from the Arena paper are not discussed; "demographics don't tell the whole story".
::kp arena-pairwise-elo
::kp arena-properties-biases
::kp elo-needs-connected-comparisons

## Can a model be the judge? AlpacaEval and WildBench {#judges}
source: lecture_12.py:L184-L205 · video 38:21-44:58

Humans are slow and expensive. The alternative is **LLM-as-a-judge**: let a strong model decide which answer is better. It is "very popular these days", and the lecture's two examples show both its promise and its failure modes.

### AlpacaEval: win rate, length bias, and a regression fix

**AlpacaEval** (2023) uses a fixed set of 805 instructions from various sources. For each one, the model under test answers, a baseline model (GPT-4 preview) answers, and a judge (also GPT-4 preview) picks the better answer. The metric is the **win rate against the baseline**.

The professor immediately flags a "potential bias": the judge is the same model as the baseline it compares against. One mitigation is "multiple judges and ensembling". "But let's put that aside for now."

The problem that actually bit was different: **LLM judges favor longer responses**. Fine-tuned models were submitted that got "really high AlpacaEval performance just because the responses were longer", leaderboard gaming in plain sight. **AlpacaEval 2.0** responded with "a very simple regression method to de-bias the metric": model the judge's preference with a term for the length difference, then report the win rate with that term set to zero (length-controlled win rate).

::code lecture_12.py:L184-L188 | 805 instructions; win rate vs GPT-4 preview judged by GPT-4 preview; length bias, gaming, regression fix
::predict llm-judge-length-bias
::widget fixture:lecture_12--length-bias | pad B's answers: its raw win rate climbs the S-curve and passes A, while its length-controlled rate (dashed) does not move; set the length coefficient to 0 and padding buys nothing

The widget's curve is a toy (its coefficients are demo values, not AlpacaEval's), but it shows why the fix works. If the judge's preference is quality plus a length effect, reading the fitted model at zero length difference keeps the quality part and drops the padding part. A judge's bias, once a leaderboard rewards it, becomes a target that submissions optimize.

::note skip 39:03 | The self-preference concern (GPT-4 preview judging against GPT-4 preview) is raised and set aside; no other judge biases (position, self-enhancement) are named.

### How do you evaluate a metric?

This raises a question "not necessarily specific to AlpacaEval": we evaluate models with metrics, but how do you know a *metric* is any good? "This is a hard problem. And there's no answer here." One sanity check is **correlation with another metric** you trust. The professor read AlpacaEval's correlation with Chatbot Arena off the AlpacaEval figure: 0.98. So if you want an Arena-like ranking but do not want to wait for human votes (or are "too shy to put your model" on Arena), AlpacaEval is a cheap stand-in.

::code lecture_12.py:L189-L192 | how do we evaluate the metric? correlation with Chatbot Arena is high
::figure https://github.com/tatsu-lab/alpaca_eval/raw/main/figures/chat_correlations_no_ae.png | how well each automatic benchmark's ranking agrees with Chatbot Arena's human ranking

He then attached three limits to that reasoning, all said only aloud:
- **A correlation holds over the models it was measured on.** "This correlation, of course, is with respect to a certain set of models", and "might not hold for models that let's say are stronger than GPT-4 preview", the judge. A judge that cannot tell two answers apart better than it could write them is the risky case.
- **Higher correlation is better only if mimicking is the goal.** "Higher isn't necessarily better in all cases unless you are trying to mimic this other metric."
- **The reference may not be ground truth.** Validating against Arena "is a little bit circular because then you can ask why, well, is Chatbot Arena really the ground truth? But at least we're all in the same boat together."

::predict judge-correlation-validity-range
::video 40:46-41:22 | "this correlation, of course, is with respect to a certain set of models"
::figure official/lectures/images/alpacaeval-leaderboard.png | the AlpacaEval leaderboard as it stood; historical
::note aside 41:22 | The AlpacaEval leaderboard "hasn't been maintained in over a year"; the screenshot is what it looked like at the time.

### WildBench: give the judge a checklist

**WildBench** sourced 1024 examples from 1M human–chatbot conversations, collected, like Arena, from a free chat service. Like AlpacaEval it uses an LLM judge (GPT-4 Turbo, plus GPT-4). Its innovation is a **checklist** generated for each prompt or task, which the professor likens to chain of thought for judging. The reason it helps: asking a judge whether a response is good "is in some sense a very ill-defined task", since it depends on what you care about; a prompt-specific checklist "greatly scopes" it. It, too, is well correlated with Chatbot Arena, "the de facto sanity check".

::code lecture_12.py:L194-L197 | 1024 examples from 1M conversations; LLM judge with a checklist; correlated with Arena
::figure official/lectures/images/wildbench.png | the judge sees the conversation, the response(s) and a task-specific checklist, then either compares two answers (WB-Reward) or scores one (WB-Score); bottom right, both correlate with Arena Elo

### What the chat section concludes

- **Evaluating open-ended responses has no clean solution**, only ideas.
- **Pairwise comparisons provide higher signal**, especially between similar responses. "This one's slightly better than this" is directional information; deciding whether something is "a 7 out of 10 or a 8 out of 10 ... tends to be a lot less high signal."
- **Beware of biases from humans and from LLM judges.** They differ, but both exist; evaluating with multiple judges, human and model, is the main hedge: if all of them say your model is better, maybe it is.
- **A checklist or rubric improves reliability, whoever judges.** "If you just ask a human to rate something without giving some [kind] of rubric, you're probably going to get very nonsensical results."

::code lecture_12.py:L201-L205 | pairwise for signal, beware biases, rubrics for reliability
::kp llm-judge-length-bias
::kp judge-correlation-validity-range

## How do you evaluate what a model does? Agentic benchmarks {#agents}
source: lecture_12.py:L208-L254 · video 44:58-53:56

Chat benchmarks evaluate what language models **say**. Agentic benchmarks evaluate what they **do**. An **agent** is a language model plus an **agent scaffold**: the logic that decides how the language model is called, what tools it can use, and what it sees next. The tasks require tool use (running code, for instance) and iterating over a period of time, and they are graded by outcomes rather than by judging a single response.

::code lecture_12.py:L209-L214 | say vs do; agent = LM + scaffold; tasks with tools and iteration

### Four benchmarks

**SWE-bench**: 2294 tasks across 12 Python repositories. Given a codebase and a GitHub issue, submit a pull request; the metric is **unit tests**. The tests are chosen so that some fail before the fix; the PR must make them pass "and you don't break anything else". The appeal: "the evaluation is very straightforward." It pioneered evaluating an agent's coding in a realistic environment, where much of the attention on agents has gone. On the live tracker (for SWE-bench Verified, explained in the validity section) the professor read about 16% in 2024 and 93 now.

::figure official/lectures/images/swebench.png | the model gets the issue and the code; its patch is graded by the repository's tests (here 2 fail, so the patch does not count)

**Terminal-Bench**: the environment is a computer terminal, "very simple and universal": any task that can be done by typing commands. 229 tasks crowdsourced from 93 contributors; 89 of them make up Terminal-Bench 2.0. The tasks take a human from an hour to over a week, depending on expert or junior (his reading of the human-time figure).

::figure official/lectures/images/terminal-bench.png | one task: the agent gets a container and an instruction, works in the shell, and hidden tests decide pass or fail
::figure official/lectures/images/terminal-bench-human-time.png | estimated human time per task: experts mostly under a day, juniors stretching past a week
::figure official/lectures/images/terminal-bench-results.png | read the Model column: the same model appears in several rows with different agents, and different accuracies

That last figure carries the section's main point. "You can have two agents but the same model and they have different accuracies."

**Cybench**: 40 Capture the Flag tasks, the cybersecurity exercises humans compete in. The agent can read source code and access a server, and must hack in to extract a **flag**, a unique string that proves the break-in. Difficulty is measured by **first-solve time**: how long the first human team took to solve the task in the original competition. The early Cybench scaffold is the simplest possible loop: one continuous memory to which every action and observation is appended. "This history obviously grows quite a bit", which is why later scaffolds need ways of managing context. When the benchmark came out the best models solved around 10%; now, he said, it is "completely solved": the leaderboard's oldest and newest rows show the whole climb.

::figure official/lectures/images/cybench.png | a CTF task: the agent runs commands in a Kali Linux box against a task server and must submit the flag (optional subtask questions guide it)
::figure official/lectures/images/cybench-agent.png | the simple scaffold: act, execute in the environment, append the observation to one growing memory, repeat
::figure official/lectures/images/cybench-results.png | the leaderboard today, oldest models at the bottom; the last columns give the hardest task solved, by human first-solve time

**MLE-bench**: 75 Kaggle competitions. The agent reads the description and the data, writes code, processes data, trains models and submits, and gets a grade.

::figure official/lectures/images/mlebench.png | a competition's description and data go to the agent, which trains, debugs and submits a CSV; the grader scores it against the Kaggle leaderboard
::figure official/lectures/images/mlebench-results.png | the leaderboard: each row is an agent (scaffold) plus an LLM, and the same LLM appears under several agents

::code lecture_12.py:L216-L240 | SWE-bench (2294 tasks, unit tests), Terminal-Bench (229 / 89), Cybench (40 CTFs, first-solve time), MLE-bench (75 competitions)
::note skip | Cybench's first-solve time appears in the code but was not mentioned aloud.

### Scaffolds matter

On these leaderboards the models are "the same usual suspects", but there is "quite a bit of variation" across agent scaffolds. Solving complicated tasks now takes far more than the simple loop above. Four ingredients, from a post on "deep agents":
- **Explicit planning.** Stream-of-consciousness chain of thought builds up context until "the agent can easily lose track of where it is". Keep a to-do list instead, and check items off.
- **Hierarchical delegation.** Call sub-agents with a clean context; a sub-agent "returns only the result", so the main agent's context stays uncluttered (encapsulation).
- **Persistent memory.** Read and write files, since a growing history cannot all live in the context window.
- **Extreme context engineering.** Explicit instructions about the process: when to delegate, which strategy to try, what to write to memory.

::figure https://www.philschmid.de/static/blog/agents-2.0-deep-agents/overview.png | the deep-agent pattern: an orchestrator with a plan, sub-agents, and file-based memory around the model

And these features "are generally optimized for the particular language model". So the score on an agentic benchmark belongs to the pair:
- agents dramatically enhance the capability surface of language models;
- agent scaffolds are very important;
- **evaluating agents = evaluating the agent scaffold + the language model.**

::code lecture_12.py:L244-L254 | four scaffold ingredients; evaluating agents evaluates scaffold and model together
::predict agent-lm-plus-scaffold
::note deferred 47:00 | Why the tracker shows SWE-bench *Verified* rather than SWE-bench, "which later I'll show you": see dataset quality in the validity section.
::kp agent-lm-plus-scaffold

## Can reasoning be measured apart from knowledge? ARC-AGI {#reasoning}
source: lecture_12.py:L257-L283 · video 53:56-1:00:12

Every benchmark so far needs linguistic and world knowledge. Can we isolate **reasoning**, "pure fluid intelligence", from knowing facts? Arguably that is a purer form of intelligence, since it "isn't just about memorizing facts". The professor calls this family "pure reasoning, for lack of a better term".

**ARC-AGI** (started in 2019) is the main attempt. Its design rules:
- **100% solvable by humans, but challenging for AI.** The human ceiling anchors the construct; it is not a sign the tasks are easy.
- **Every task is unique** ("a special snowflake"), so memorizing facts or previous problems should not help.

Starting in 2019, in the GPT-2 era, before large language models took off, it was "prescient".

::code lecture_12.py:L258-L264 | knowledge vs reasoning; solvable by humans, hard for AI; every task unique

A task shows a few input-output grid pairs and asks for the output of a new input. In the first example "if you take 10 seconds, you can probably solve" it: fill in the yellow to complete the rectangle. ARC-AGI-2 (March 2025) needs more multi-step reasoning.

::figure https://arcprize.org/media/images/arc-task-grids.jpg | ARC-AGI-1: infer the rule from the example pairs, apply it to the test grid
::figure https://arcprize.org/media/images/blog/arc-agi-2-unsolved-1.png | ARC-AGI-2: several rules must be combined

### What moved the scores

::figure official/lectures/images/arc-agi-results.png | scores by model release date: ARC-AGI-1 (circles) sits near 0 until the "AI Reasoning" line, then climbs; ARC-AGI-2 (triangles) follows later

The trajectory reads like a controlled experiment. Pretrained language models, GPT-3 included, "didn't move the needle at all", exactly as the creators intended: pretraining teaches facts and linguistic patterns, and neither directly helps on grids. Then in 2024 OpenAI released the reasoning models o1 and o3, and scores "started taking off pretty abruptly". ARC-AGI-1 is now basically solved, and ARC-AGI-2 looks on its way. "It was really the reasoning capabilities that unlocked this."

One caveat: "without pre-training we wouldn't have this explosion of reasoning models". Pretraining was a precondition, just not "directly visible" in the chart.

::code lecture_12.py:L272-L274 | pretrained LMs didn't move the needle; reasoning models made it take off
::predict reasoning-vs-knowledge-arc

### ARC-AGI-3

The familiar pattern again: the creators saw the old version falling and made a new one. **ARC-AGI-3** (March 2026) is a set of **interactive environments**, small games you play online. Again there is no language; you work out the rules from patterns. Scores are "extremely low" for now: "next year when I teach this class, I'm sure I'll have to update this slide."

::figure official/lectures/images/arc-agi-3.png | a level of one ARC-AGI-3 game: arrow keys and clicks, no instructions; the goal and rules must be discovered by playing
::figure official/lectures/images/arc-agi-3-results.png | the frontier models' scores at launch: every one under one percent

Asked what the model actually receives, since the game looks so graphical: the frame is, he thinks, about 64 by 64, and it can be given as an image or as ASCII art. Either way the model must reason about something spatial, "really not English or natural language".

### Limits

- **Disentangling reasoning from knowledge is difficult.** ARC is "probably the best attempt", but it is not clear you can fully decouple them: the tasks "still come from some prior", and he is "not sure there's anything as quote unquote pure as reasoning". If people really cared about a benchmark, they could probably game it.
- **It is constrained to human reasoning.** Being 100% human-solvable in reasonable time, it says nothing about superhuman reasoning, such as winning more olympiad gold medals or solving open math problems, which "arguably are still very useful and important".
- **It clearly exposes gaps** in current models.

::code lecture_12.py:L280-L283 | disentangling is hard; human-level only; exposes gaps
::note aside 57:57 | ARC-AGI-3 scores are expected to change fast; the screenshot is a snapshot.
::kp reasoning-vs-knowledge-arc

## What does safety mean for AI? {#safety}
source: lecture_12.py:L286-L311 · video 1:00:12-1:05:09

For cars, safety is clear: crash tests smash a car into a wall and measure how the dummies fare. That clarity is the result of "decades of lobbying and figuring out what safety means for vehicles". For AI "there is not a great answer here", so the lecture gives examples of how people have operationalized it, and then the reasons no single number will do.

::figure https://www.team-bhp.com/forum/attachments/road-safety/2173645d1625144681-will-crash-test-rating-change-if-higher-variant-chosen-images-30.jpeg | a crash test: the kind of settled safety definition AI does not have

### Two benchmarks

- **HarmBench** is built from 510 harmful behaviors that violate laws or norms. Prompt the model with them and expect it to **refuse**. This is probably the dominant thing people mean by safety, preventing bad actors from using models, "but it turns out that's not the only thing that matters". (The [HELM HarmBench page](https://crfm.stanford.edu/helm/safety/latest/#/leaderboard/harm_bench) shows results, including a linked example of a safety failure.)
- **AIR-Bench** tries to be holistic. It starts from regulatory frameworks (in the EU, China and the US) and company policies, builds a taxonomy of 314 risk categories, and writes 5694 prompts against them.

::figure https://crfm.stanford.edu/helm/assets/air-overview-DpBbyagA.png | AIR-Bench's taxonomy: from regulations and policies down to fine-grained risk categories, each with prompts
::code lecture_12.py:L290-L297 | HarmBench: 510 harmful behaviors; AIR-Bench: regulations and policies, 314 categories, 5694 prompts

### Jailbreaks

Models are trained to refuse harmful instructions, but "if you're clever" you can get around it. **GCG** (Greedy Coordinate Gradient) does it automatically: a coordinate-wise optimization over the tokens of a suffix appended to a harmful request, until the model complies. The suffix is "basically gibberish". And, "remarkably", suffixes optimized on open-weight models (Llama) **transfer** to closed models (GPT-4): an attacker does not need the closed model's gradients. At the time, a gibberish suffix plus a request for a step-by-step plan to destroy humanity got a commercial model to comply.

::figure official/lectures/images/gcg-examples.png | one request plus an optimized gibberish suffix, sent to four chatbots from four companies (ChatGPT, Claude, Bard, Llama-2): all comply, which is what transfer looks like
::code lecture_12.py:L301-L304 | GCG optimizes prompts to bypass safety; transfers from Llama to GPT-4
::predict safety-contextual-dual-use
::note aside 1:02:51 | "I hope ... these attacks don't work anymore." Whether GCG still works today is left open.

### What is safety?

- **Safety is strongly contextual.** It involves politics, law and social norms, which vary across countries; what is safe in one jurisdiction may not be in another.
- **The risks are varied** and need different attention: hallucination (especially in medical, legal or financial settings), sycophancy, abetting crimes, inequality, loss of critical thinking. They also move differently with capability: hallucination falls as models get more accurate, while abetting crimes "might be counter to capabilities" (captions garbled here; presumably because a more capable model is more useful to a bad actor).
- **Capabilities are dual-use.** A capable cybersecurity agent (the professor names Mythos) can hack into a system or do penetration testing to make systems more secure, so whether it is a risk or a benefit cuts both ways: a double-edged sword. This is the same skill Cybench measured.

So one refusal rate is one slice of one notion of safety, not a safety score.

::code lecture_12.py:L307-L311 | contextual, varied risks, dual use
::note skip 1:04:30 | The broader view of safety as AI going well for people, its societal impact, "is actually a much more complex topic than I'll have time to talk about".
::kp safety-contextual-dual-use

## How close is an evaluation to real use? {#realism}
source: lecture_12.py:L314-L335 · video 1:05:09-1:08:43

The lecture now steps back from benchmark families to properties every evaluation has. The first is **ecological validity**: how well does an evaluation capture real-world use?

- **Exam benchmarks** such as GPQA are far from it, however hard they are.
- **Chatbot Arena** prompts come from real people, but the distribution is uncontrolled; "one could wonder whether this distribution is the right distribution of people or use cases".

The three projects below move closer, "at the use case level, at least not the individual query level".

::code lecture_12.py:L315-L317 | ecological validity; exams far from it; Arena real but uncontrolled

**GDPVal** (OpenAI) takes the top 9 sectors of the US economy by GDP and 44 occupations within them, and has professionals with about 14 years of experience write tasks from their own work: nurses, concierges, real-estate agents, film and video editors.

::figure official/lectures/images/gdpval.png | nine sample tasks, each a real deliverable (a 3D design, a consultation report, an itinerary, a sales brochure) rather than a question with a key

**MedHELM** starts from the observation that medical benchmarks had been standardized exams. Humans pass those exams too, then train for years more before treating patients; likewise a language model "should also not just pass medical exams" and be deployed. So its 121 clinical tasks were sourced from 29 clinicians, a mixture of private and public datasets, meant to represent what clinicians would actually ask a model.

::figure https://crfm.stanford.edu/helm/assets/medhelm-overview-CND0EIsy.png | MedHELM's overview: clinical tasks grouped by what clinicians do, instead of exam questions

**Clio** (Anthropic) asks who actually has the data on how people use models. The model developers. Privacy forbids reading people's conversations, but you can have language models read them and report only aggregate patterns: what kinds of things people use Claude for.

::figure official/lectures/images/clio-table4.png | Clio's check on synthetic data where the true categories are known: the categories it recovers closely match the ground truth

The section ends on a tension: **realism and privacy are sometimes at odds**. Ideally you would evaluate on "a sample from the actual query stream", which would show "deeply what the errors are"; but that stream is exactly what privacy protects. Note also what realism costs in gradability: a GDPVal deliverable or a clinical note has no answer key, so you are back to the judging problems of the chat section.

::code lecture_12.py:L319-L335 | GDPVal (44 occupations, ~14 years' experience), MedHELM (121 tasks, 29 clinicians), Clio; realism vs privacy
::predict ecological-validity-realism
::kp ecological-validity-realism

## How do we know an evaluation is valid? {#validity}
source: lecture_12.py:L338-L368 · video 1:08:43-1:15:41

The second cross-cutting property is scientific validity: does the number measure what we think? Two ways it fails are covered: the test data was in the training data, and the test items are themselves wrong.

### Train-test overlap

Machine learning 101: don't train on your test set. Before foundation models this was easy to honor. ImageNet, SQuAD and the like came with a train split and a test split, "and everyone played the same game". Today models are trained on the internet, "and much more than that", and the developers don't tell you what was in the data. The lecture offers four routes.

::code lecture_12.py:L341-L344 | ML 101; fixed splits before; today, train on the internet and don't tell

**Route 1: infer overlap from the model.** A neat black-box test from Tatsu Hashimoto's group exploits **exchangeability**. The order in which a benchmark lists its questions is arbitrary, so a model that never saw the file has no reason to prefer the published order over a shuffle. If it assigns noticeably higher log-probability to the questions in their published order, it probably saw that file, in that order, during training.

::figure official/lectures/images/contamination-exchangeability.png | in canonical order each question gets high log-probability because it follows its published predecessor; shuffled, those same questions drop

::predict train-test-overlap-routes
::widget fixture:lecture_12--order-test | with nothing memorized every one of the n! orders scores the same (p = 1); memorize all published transitions of 4 items and the published order beats every shuffle (p = 1/24); tick "items depend on their predecessor" and a clean model is flagged too

The last case is the test's assumption made visible. If items genuinely depend on each other (a benchmark of multi-part questions, say), the published order is more likely for any model, and the test would cry contamination falsely.

**Route 2: reporting norms.** In statistics you always report a confidence interval with an estimate; that is just standard practice. By the same logic, a position paper argues that model providers should report train-test overlap: if you claim a GPQA score, "you should always provide some justification that you didn't train on the test set."

**Route 3: fresh evaluations.** Give up and assume the worst case: everyone trained on every existing benchmark. What we have on our side is that we can always build new ones. LiveCodeBench and UncheatableEval scrape new web pages, arXiv papers or GitHub content dated after the models' training cutoff. Robust, but timestamps are not safe either: a repository posted after the cutoff may be "derived from some other repo".

**Route 4: private evaluations.** Companies evaluate on internal code bases that are not on the internet and that they would not train on. If you are not a company, use your own writing; the professor keeps rejected papers from grad school that he never put online. This route is easiest for **perplexity**, because perplexity needs no labels: "all you need in perplexity is just you need a good data set and you can evaluate log probabilities."

::code lecture_12.py:L346-L360 | route 1: exchangeability; route 2: report overlap; route 3: fresh evals (timestamps not safe); route 4: private evals, easiest for perplexity

Recall the subtlety from the exam Q&A: contamination is rarely literal. Test questions are often derived from sources that are on the web, so a model can have seen their substance without seeing the benchmark. HLE's private set and route 4 attack this by keeping the material off the web entirely.

::note aside 1:12:44 | His own private eval: rejected papers from grad school, "never put online".
::kp train-test-overlap-routes

### Dataset quality

Even with no contamination, a benchmark can be broken item by item.

- **SWE-bench Verified.** This is why the agent section showed SWE-bench *Verified*: the original had problems, "unit tests that weren't quite rigorous enough and other things", and the tasks were screened and fixed.
- **Platinum benchmarks.** Many benchmarks, GSM8K and MMLU among them, have been audited, with people finding that "something's broken around this question". One item asks about a curve that is never given; another asks whether a baby has socks on, when there is no way to tell. A Platinum version removes or repairs such items, so that the errors left are really the model's. When top models plateau below 100%, part of the gap may be label error.

::figure https://pbs.twimg.com/media/GjICXQlWkAAYnDS?format=jpg&name=4096x4096 | broken items found in popular benchmarks: questions that cannot be answered as written
::figure https://pbs.twimg.com/media/GjICcGQXYAAM4o1?format=jpg&name=4096x4096 | more examples from the Platinum audit

- **Agentic benchmarks are even harder to assess**, because an item is a whole environment, not a question you can read with its answer. Two failure modes: **insufficient test cases**, so "you might pass all the test cases but still not have a working solution"; and a **trivial agent can solve the task**. In one benchmark (τ-bench), an agent that outputs the empty response scores 38%.
- **Docent** uses a language model to inspect agent traces and flag problems: "a qualitative response to our very quantitative heavy way of benchmarking".

The advice that closes the section applies to anyone who builds or runs a benchmark: "always look at the output and try to audit it to make sure that you actually think you're measuring what you think you're measuring."

::code lecture_12.py:L362-L368 | SWE-bench Verified; Platinum benchmarks; insufficient tests and trivial agents; Docent
::predict dataset-quality-verified-platinum
::note slip 1:14:50 | The captions name the benchmark with the 38% empty-response score "TorchBench". The cited paper (arXiv 2507.02825) reports it for τ-bench, where intentionally impossible airline tasks accept an empty reply; a caption error, not the professor's.
::kp dataset-quality-verified-platinum

## What is evaluation for, and what are we evaluating? {#point}
source: lecture_12.py:L371-L390, L27-L30 · video 1:15:41-1:18:30

"There's no one evaluation to rule them all." Which one is right depends on the question you are trying to answer, and that question is "often ... not really stated clearly". Four purposes, each leading to different benchmarks or combinations:

1. **A purchase decision.** A user or company choosing model A or B for a use case, such as a customer-service chatbot.
2. **Raw capability.** A researcher with an intuitive notion of intelligence who wants to measure it.
3. **Benefits and harms**, for business or policy reasons.
4. **Developer feedback**, to improve the model.

Matching purpose to family: a purchase decision wants something close to the use case (a realistic or private task set); raw capability leans on exams and reasoning benchmarks; harms on safety benchmarks; developer feedback on cheap, smooth signals such as perplexity.

::code lecture_12.py:L373-L377 | no one true evaluation; four purposes

### Methods versus models

What is being evaluated has also changed. Before foundation models, researchers evaluated **methods**: with standardized train-test splits, "the only thing that was varying is the actual algorithm", so a better score meant a better method. Today we mostly evaluate **models and systems**, where "anything goes": a score reflects data, compute and scaffold as well as any method, which is useful because the end model "is the thing that is actually going to get shipped".

There are exceptions. The **nanoGPT speedrun** fixes the data and asks how fast you can train a model to a particular validation loss. It is "deliberately a way to evaluate an algorithm".

::figure official/lectures/images/karpathy-nanogpt-speedrun.png | the speedrun as a benchmark: train a 124M Transformer to a fixed validation loss; records are measured against a fixed target

Each kind has its use: evaluating methods encourages algorithmic innovation from researchers, evaluating models serves downstream users. "Either way, we need to define the rules of the game!" A field that only does the second loses the ability to say *why* a model is better.

::code lecture_12.py:L380-L390 | methods vs models/systems; the speedrun exception; define the rules of the game
::predict methods-vs-models-rules-of-game

### Takeaways

- **There is no one true evaluation**; choose it according to what you are trying to measure.
- **State the rules of the game**: are you evaluating a method, a model, or an agent (model plus scaffold)?
- **Consider difficulty, realism and validity.** They trade off: "It's hard to have really real and difficult and ecologically valid" and uncontaminated, so "you will have to choose which one you're willing to compromise on", depending on your goal.

The tour in one table:

| family | example | what it measures | main weakness |
|---|---|---|---|
| perplexity | held-out log-likelihood | fit to a text distribution | charges every token; per-token, so tokenizer-dependent |
| exams | MMLU, GPQA, HLE | knowledge on graded questions | saturates; far from real use |
| chat | Arena, AlpacaEval, WildBench | preference on open-ended answers | rater and judge biases (style, length) |
| agents | SWE-bench, Terminal-Bench | what a model plus scaffold can do | scaffold confounds the model; weak tests |
| reasoning | ARC-AGI | skill on novel puzzles | knowledge never fully removed; human-level only |
| safety | HarmBench, AIR-Bench | refusals, policy compliance | contextual; jailbreaks; dual use |
| realism | GDPVal, MedHELM, Clio | closeness to actual use | hard to grade; privacy |

::code lecture_12.py:L27-L30 | the lecture's three takeaways
::video 1:18:02-1:18:19 | difficulty, realism and validity trade off; choose which to compromise on
::note deferred 1:18:22 | Next week: training data, the "missing piece" the lecture opened with.
::kp methods-vs-models-rules-of-game
