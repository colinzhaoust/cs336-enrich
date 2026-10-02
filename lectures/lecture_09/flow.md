---
title: L9 · Scaling laws (basics), read through
minutes: 45
---
This lecture is about deciding what to train before you can afford to train it. It shows that loss falls as a power law in data, parameters and compute, explains why with a little statistics, and then uses such laws to choose architectures, optimizers, shapes, batch sizes, learning rates, and the split of a compute budget between model size and data. After it you can fit and read a scaling law, size a compute-optimal model from a FLOP budget, and say why Kaplan and Chinchilla disagreed and why production models ignore both.

## What do you do with ten thousand GPUs? {#scenario}
source: lecture_09.pdf p2-p4 · video 0:06-4:04

::slide 2 | the scenario: a friend gives you ten thousand B200s for a month to build a good open-source LM; infra (assignment 2) and data (assignment 4) are done, and the open question is which big model to run

The professor sets up the whole lecture with one scenario. A very wealthy friend hands you 10,000 B200s for a month and asks for a very good open-source language model. Your infrastructure team exists (assignment 2), and you have a good pre-training dataset (assignment 4, assumed done here). What remains is the run itself, and a big run is full of choices: which architecture, which hyperparameters, how big a model, how much data. A wrong choice on a run that costs millions is very expensive. "How do you make sure that your big run is actually successful?"

How big is that budget? Lecture 2's napkin math gives a lower bound. Take the H100's dense bf16 rate, $9.9 \times 10^{14}$ FLOP/s, at 50% utilization (a B200 is faster, so the real figure is higher):

$$ C \approx 10^4 \text{ GPUs} \times 30 \times 86400 \text{ s} \times 4.95\times10^{14} \text{ FLOP/s} \approx 1.3 \times 10^{25} \text{ FLOPs} $$

That is the kind of number this lecture will turn into "train an N-parameter model on D tokens". We come back to it at the end. ([L2's FLOP counting](#/read/lecture_02) derives $C \approx 6ND$, which this lecture assumes throughout.)

::slide 3 | "Scaling isn't easy": the professor's table of models (original transformer to Mistral 7B) with their tokenizer, vocabulary, norm, position embedding, activation, MLP factor and layer count; "We could cargo cult things from existing LMs... but how do these get optimized in the first place?"

The table is the one from [L3](#/read/lecture_03): every published model's choices side by side. You could copy them. For most choices that is fine, and the professor says you will probably pick well-adopted practices "out of a hat". But if you are at the frontier and want something better than the best models today, copying cannot get you there: "You can't just copy the choices of others and get something that's better than the state of the art." And the table's values were themselves chosen by someone. The question is how.

::slide 4 | "simple, predictive rules for model performance"; old and unpleasant: tune hyperparameters on big models; new (over?) optimism: tune on small models, extrapolate; left, Kaplan's validation loss against compute with the frontier L = 2.57·C^−0.048; right, test loss against non-embedding parameters for 1 to >6 layers

The answer is **scaling laws**: simple, predictive rules that connect a model's behaviour at small scale to its behaviour at large scale. The "old and unpleasant" way is to tune hyperparameters on the big runs themselves, which means paying for many big runs. The new way is to do all the optimization on small models, and rely on a simple, robust regularity to carry the result up. The slide's "(over?)" is a warning the lecture keeps returning to: the regularity has to be real, and it has to be measured carefully.

The left plot, from the slide image, is the regularity in its most famous form. Each coloured curve is one training run; colour is model size, from $10^5$ to $10^{11}$ parameters. The dashed line under all of them is the best loss reached at each compute budget, and it is straight on log-log axes: $L = 2.57 \cdot C^{-0.048}$, with $C$ in petaFLOP/s-days. Straight on log-log axes means each 100× of compute multiplies the best loss by the same factor, $100^{-0.048} \approx 0.80$.

::note spoken 2:53 | The professor describes scaling laws at the big labs as "almost kind of a way of life ... It's almost a belief", and warns that they "can sometimes be quite tricky objects". The rest of the lecture is about both halves of that sentence.
::note deferred 0:33 | This is the first of two scaling lectures. The advanced one (after the inference lecture) covers modern open-model tech reports, muP and other parametrizations, and optimizers.
::video 3:08-3:57 | the engineering view: do all your optimization at small scale, and extrapolate with a simple rule
::kp scaling-law-definition

## Is this new? A short history of data scaling {#history}
source: lecture_09.pdf p6-p11 · video 4:04-11:21

::slide 6 | two theory results: the finite-hypothesis bound ε(ĥ) ≤ min ε(h) + 2·sqrt((1/m)·log(2k/δ)), and a nonparametric density estimator converging at rate n^(−β/(2β+1)); "But these are upper bounds, not actual, realized loss values"

Machine learning theory has always asked how good a model will be, and its answers depend on sample size. The first bound says that if you pick the best of $k$ hypotheses on $m$ samples, your error is at most the best possible error plus a term that shrinks like $\sqrt{\log k / m}$. The second, for estimating a smooth density, gives a rate $n^{-\beta/(2\beta+1)}$ that depends on how smooth the density is ($\beta$). Both are polynomial in the sample size, which already looks like a scaling law. The catch is in the slide's last line: these are **upper bounds**. They say how bad things can get, not what loss you will actually see. A scaling law is a statement about realized loss.

::slide 7 | Cortes, Jackel, Solla, Vapnik and Denker (AT&T Bell Labs, 1993): test error modelled as a + b/l^α and training error as a − c/l^β, converging to a common value a; points from small training sets predict the learning curve out to 15,360 examples

The earliest data scaling law the professor knows is from Bell Labs in 1993. The motivation is exactly today's: training a classifier on the full dataset is expensive, so fit the error curve on smaller samples and predict the rest, which avoids "training poor classifiers on the whole training set". The functional form on the slide image is worth remembering: test error $a + b/l^{\alpha}$, a power law in training-set size $l$ *plus a constant floor* $a$. That floor comes back later in this lecture. In the professor's words, this is "almost literally a data scaling law back in 1993".

::slide 8 | Banko and Brill 2001: test accuracy of four learners (memory-based, Winnow, perceptron, naive Bayes) on confusion-set disambiguation, rising roughly linearly in log(words) from 0.1M to 1000M words; their remark that none of the learners is close to asymptoting

In NLP, Banko and Brill made the same observation at much larger scale: accuracy rose steadily with the log of the corpus size, up to a billion words, for every learner they tried. Their conclusion, on the slide, is that the field should reconsider spending money on algorithms instead of on data.

::slide 9 | Kolachina et al. 2012: BLEU against training sample size for machine translation, fitted with six curve families; the table defines them, including Pow3 = c − a·x^−α and Pow4 = c − (−ax + b)^−α

Kolachina et al. asked which *functional form* predicts a translation system's BLEU score as data grows. They tried exponentials, power laws and a log form. The professor's point is that they landed on the same forms we use today, the "pow 3 and pow 4" of the table. The digit is the number of fitted constants: Pow3 is a power law with an offset ($c$, $a$, $\alpha$), Pow4 adds a shift.

::slide 10 | Hestness et al. 2017: machine-translation learning curves with fits ε₂₀₈(m) = 41.2·m^−0.36 + 0.39 and ε₅₁₂(m) = 21.5·m^−0.30 + 0.32, a composite curve ε(m) = 3.87·m^−0.13, and the hypothesized shape: small-data region, power-law region, irreducible-error region

The professor dates neural scaling laws to **Hestness et al. 2017**, a paper he thinks is under-cited. Years before the OpenAI papers, it fitted data scaling laws across machine translation, language modelling and speech, and found the same polynomial trends everywhere. The fits on the slide image have exactly the 1993 form, a power law plus a constant: for the 512-hidden model, $21.5\,m^{-0.30} + 0.32$. On the right is the shape Hestness hypothesized, which the next section uses: a flat small-data region at best-guess error, a straight power-law region, and a flat irreducible-error region.

::slide 11 | "Very ahead of its time": three excerpts, on emergence (accuracy cliffs until enough data puts a model in the power-law region), scaling by compute (learning curves project compute requirements), and speed = accuracy (faster computation can buy back accuracy lost to quantization or sparsity)

The paper also anticipated three ideas that are central today. **Emergence**: a model can sit at best-guess accuracy and then suddenly improve. The professor's gloss is that accuracy "is a much more discontinuous measure than losses". **Scaling by compute**: if models scale predictably with data, compute becomes the bottleneck. And **speed is accuracy**: once compute is the bottleneck, systems speed-ups turn directly into better models, which is the premise of the systems half of this course.

::note spoken 10:33 | Asked whether scaling laws can be justified or are just empirical, the professor answers that they are "pure curve fitting exercises". Polynomials are "arguably a very natural class", but there is "no golden rule". Theory supplies candidate forms (how error rates decay), and so do physicists, who think in limits. The "(?)" on slide 16 below is meant seriously.

## What does a scaling law look like? {#power-laws}
source: lecture_09.pdf p12-p15 · video 11:21-15:45

::slide 13 | top: Kaplan et al.'s three fits, L = (C_min/2.3·10⁸)^−0.050 against compute, L = (D/5.4·10¹³)^−0.095 against dataset size, L = (N/8.8·10¹³)^−0.076 against non-embedding parameters; bottom: downstream tasks fitted by sigmoids in log FLOPs, and Epoch AI's capability index against release date

Part 2 of the lecture asks three questions in turn (slide 12): how data affects performance, whether to spend on data or on model size, and how to set hyperparameters on the big model. It starts with the striking fact that makes all three answerable: put a resource on a log x-axis and log test loss on the y-axis, and you get a straight line. In the top row, from Kaplan et al. 2020 ("OpenAI's neural scaling laws paper", which the lecture cites again and again), that resource is compute, then dataset size, then parameters, and each time "all of them linearize log test loss".

The slide image prints the three fitted exponents, which the text layer drops. Read them as "what 10× buys":
- compute: $10^{-0.050} \approx 0.89$, so 10× compute cuts loss by about 11%;
- data: $10^{-0.095} \approx 0.80$, about 20% per 10× tokens;
- parameters: $10^{-0.076} \approx 0.84$, about 16% per 10× parameters.

The bottom row shows how far the idea stretches. Downstream accuracy on a task is not a line but a **sigmoid** in log compute: flat at chance, then rising, then saturating. And forecasting work even puts release *dates* on the x-axis, where the upper envelope of capabilities also looks linear. None of these had to be regular. "Language model performance is often much more regular as a function of scale and resources than it initially might appear."

::slide 14 | "Data scaling laws: simple formula that maps dataset size (n) to error"; Hestness's monotonic, logistic-like curve on log-log axes: best-guess error, then the power-law region, then the irreducible error

Start with the simplest kind, the **data scaling law**: fix the model and the training procedure, grow the dataset, and measure the error. Unless the professor says otherwise, the model is much bigger than the dataset, so data is the only constraint. What should the curve look like? It should be monotone, "monotone-ish if we tune our hyperparameters well", because more data should not hurt. It starts at random guessing and ends at an entropy-like floor you cannot beat, the **irreducible error** of the task ("that's the noise floor"). In between is the part we care about.

::slide 15 | Kaplan's data law alone: test loss from about 4.2 down to 2.75 as tokens grow from about 2·10⁷ to 1.4·10⁹, on a straight line L = (D/5.4·10¹³)^−0.095; "Scale-free" or "Power law"

In the middle region, loss against dataset size is a straight line on log-log axes. You will make a plot like this in assignment 3. A line on a log-log plot is called **scale-free** or a **power law**:

$$ \log L = -\alpha \log n + c \quad\Longleftrightarrow\quad L = A\, n^{-\alpha} $$

What does that mean in practice? Two things, the professor says. First, the error decays *polynomially*: every multiplication of the data by the same factor multiplies the loss by the same factor. It does not subtract a fixed amount. Second, a straight line "usually means that I'm very far away from my asymptote". Close to the floor, the curve has to taper off.

**Worked example.** With the slide's fit $L = (D/5.4\times10^{13})^{-0.095}$: at $D = 10^8$ tokens, $\log_{10}(D/5.4\times10^{13}) = -5.73$, so $L = 10^{0.545} \approx 3.5$; at $D = 10^9$, $L = 10^{0.450} \approx 2.8$, matching the plotted points. To cut the loss by another 10%, the data must grow by $(1/0.9)^{1/0.095} = 10^{0.0458/0.095} \approx 3.0\times$.

::predict scaling-law-definition
::widget fixture:lecture_09--power-floor | set the floor E to 0: the blue line is straight and "next 10× of data multiplies L by" reads the same factor at every n, so 4 → 2 over one decade means 2 → 1 over the next, never a fixed amount subtracted
::video 15:13-15:45 | what a line on a log-log plot means, and why it also means you are "very far away from my asymptote"
::note aside 12:59 | Forecasting "scaling laws" with dates or capability indices on the x-axis are mentioned, not taught.
::kp scaling-law-definition

## Why would loss be a power law? {#why-power}
source: lecture_09.pdf p16-p20 · video 15:45-21:33

::slide 16 | Q: why do scaling laws show up? We know error should be monotone, but why linear in log-log? A (?): estimation error naturally decays polynomially; example: estimating the mean of a dataset

Monotone we expected. But why a *line*? The slide's tentative answer, with its question mark, is that estimation error naturally decays polynomially in the number of samples. To see what that means, the lecture becomes "a statistics class for the next ... three slides".

::slide 17 | mean estimation: x₁ … xₙ ~ N(μ, σ²), μ̂ = Σᵢxᵢ/n, E[(μ̂ − μ)²] = σ²/n; "This is a 'scaling law'": log(Error) = −log n + 2 log σ; "More generally, any polynomial rate 1/n^α is a scaling law"

The simplest estimation problem: draw $n$ samples from a Gaussian and estimate its mean by the sample average. The expected squared error is exactly the variance of the average, $\sigma^2/n$. Take logs and you have a line:

$$ \mathbb E(\hat\mu-\mu)^2 = \frac{\sigma^2}{n} \quad\Longrightarrow\quad \log \text{Error} = -\log n + 2\log\sigma $$

The slope is $-1$, set by the estimator. The noise $\sigma$ appears only in the intercept, so noisier data moves the line up without tilting it. The general lesson, said aloud: anything of the form $1/n^{\alpha}$ "plus some constant" is a scaling law once you subtract the constant and plot on log-log axes.

::worked power-law-from-estimation-rate
::widget fixture:lecture_09--power-floor | press "mean estimation, α = 1" and change A (σ²): the whole line moves up or down without changing its slope, so 100× more data cuts the error 100× whatever A is
::note spoken 16:19 | The variance step is justified in one sentence ("each of these guys is Gaussian, and ... mu hat is Gaussian"), not derived. The worked steps above fill it in: independent variances add to nσ², and the 1/n in front of the sum is squared.
::kp power-law-from-estimation-rate

::slide 18 | "most 'classical' models (regression, etc) have 1/n scaling. This means we should see y = −x + C"; measured neural laws: machine translation ε(m) = 3.87·m^−0.13, speech ε(m) = 0.95·m^−0.30 and 1.36·m^−0.30, language modelling L = (D/5.4·10¹³)^−0.095; "Very different from predictions.. Why might this be?"

Classical estimators, mean estimation or linear regression, all give error about $1/n$ (or $d/n$ for $d$ regression coefficients). So we should see slope $-1$. The measured neural slopes, printed on the slide image, are $-0.13$ for translation, $-0.30$ for speech and $-0.095$ for language modelling, which the professor rounds to "about −0.1, −0.3, −0.1". Still polynomial, but far slower. A slope of $-0.1$ means 10× more data cuts the loss by only 21%; at slope $-1$ it would cut it 10×.

::note spoken 17:39 | The professor attributes the two left panels to Hestness and the right one to Kaplan; the language-modelling panel is the same Kaplan fit as on slide 15.

::slide 19 | nonparametric detour: x uniform in the 2D unit box, yᵢ = f(xᵢ) + N(0,1); cut the box into squares of side n^(−1/4); √n boxes with √n samples each, Error ≈ 1/√n + (other smoothness terms); in d dimensions Error = n^(−1/d), slope −1/d

Where would such a slow rate come from? From flexibility. Neural networks can approximate arbitrary functions, so compare them with an estimator that assumes nothing about $f$ except smoothness. Cut the 2D input box into small squares of side $n^{-1/4}$ and, in each square, predict the average of the $y$'s that land there. There are $(n^{1/4})^2 = \sqrt n$ squares, each receiving about $n/\sqrt n = \sqrt n$ samples, so each square is a small mean-estimation problem with error about $1/\sqrt n$. In $d$ dimensions the same construction gives

$$ \text{Error} \approx n^{-1/d}, \qquad \text{slope } -\tfrac{1}{d}. $$

The trade is the one any flexible learner faces. Finer boxes fit a rougher $f$, but the same samples spread over more boxes, so each box learns more slowly, and the effect compounds with every dimension.

::worked nonparametric-dimension-exponent
::predict nonparametric-dimension-exponent
::widget fixture:lecture_09--power-floor | press the "boxes" presets: with α = 1/d the line flattens as d grows; 16× data halves the error at d = 4, and halving it at d = 6 needs 2⁶ = 64× the data
::note why | The slide's $n^{-1/d}$ is the rate of the box estimator's averaging step alone. Balancing the bias of wide boxes against the variance of narrow ones gives the textbook rate $n^{-2/(d+2)}$, which equals $n^{-1/2}$ at $d = 2$. Either way the exponent shrinks as $d$ grows, which is the only point the slide needs.

So one "mental model", in the professor's words, is that the networks on slide 18 are behaving like "non-parametric regressors in 10 dimensions", like a nearest-neighbour estimator in ten dimensions: slope $-1/d = -0.1$ gives $d = 10$.

::slide 20 | Bahri et al. 2021: scaling laws arise from polynomial rates 1/n^α, and the slope α is connected to the intrinsic dimensionality of the data; plot of 4/α_D against estimated dimension for teacher-student nets, CIFAR-10/100, SVHN, MNIST, FashionMNIST, with reference lines 4/α_D and 2/α_D; "But estimators of intrinsic dimension are sketchy, and this is not airtight.."

Some theorists push this further and argue that the exponent literally measures the **intrinsic dimension** of the data, the number of directions the data actually varies along, which can be far below the raw input dimension. The plot tests it: estimated dimension on the x-axis, $4/\alpha_D$ on the y-axis. Teacher-student points sit near the dashed $4/\alpha_D = d$ line; the image datasets scatter around it.

::note aside 20:07 | The professor keeps his distance: "I don't quite how much I truly, truly buy this argument", since the evidence "relies on estimators of intrinsic dimension", which may be "a little sketchy". He offers it as an interesting way to read exponents as a measure of how fast networks learn, not as a measurement.
::note aside | The plot's reference line is $\alpha_D \approx 4/d$, not the box estimator's $1/d$ (read from the slide image). Under that constant a language-model slope of 0.1 would mean $d \approx 40$, not 10. The qualitative story, higher dimension means a shallower slope, is the same; the number is only as good as the constant.
::kp nonparametric-dimension-exponent

### A straight line means you are far from the floor
A student asked what “the model is bigger than the data” means. The answer completes the picture of slide 14. When the model is small relative to its data, you enter the **irreducible error regime** for that model class: you have fitted the data as well as the model can, and more data does not help. To measure a clean data law you want to stay in the power-law regime, with a model “usually like 10 times bigger than the data, what have you”, or else “explicitly fit the asymptote and then you correct for it”.

What does "correct for it" mean? Write the law with its floor, as Cortes (1993) and Hestness (2017) both did:

$$ L(n) = E + A\,n^{-\alpha} $$

On raw log-log axes this curve bends. Far from the floor the reducible part $A n^{-\alpha}$ dominates and the line is straight with slope $-\alpha$. Near the floor the loss flattens toward $E$, and a straight-line fit to the raw points reports a slope much shallower than $\alpha$. Subtract $E$ and plot $\log(L - E)$, and the line comes back.

**Worked example.** Suppose the floor is $E = 2.0$, and the loss is 3.0 at 1B tokens and 2.2 at 10B. A naive line through the raw losses has slope $\log_{10}(3.0/2.2) \approx 0.13$ per decade. Subtract the floor first: the reducible part falls from 1.0 to 0.2, five-fold per decade, so $\alpha = \log_{10} 5 \approx 0.70$. The two readings also disagree about the future. The raw line predicts $2.2 \times 10^{-0.13} \approx 1.6$ at 100B tokens, below the floor; the floored law predicts $2.0 + 0.04 = 2.04$.

::predict irreducible-floor-bends-loglog
::widget fixture:lecture_09--power-floor | raise the floor E: the blue curve bends flat toward the dashed floor, the raw-slope readout falls below α, and the grey no-floor extrapolation drops under E; tick "plot L − E" and it is straight again with slope −α
::video 20:47-21:33 | "this irreducible error regime", and his two options: stay in the power-law regime, or fit the asymptote and correct for it
::kp irreducible-floor-bends-loglog

## Which data, and can you repeat it? {#data-engineering}
source: lecture_09.pdf p21-p27 · video 21:33-30:13

::slide 21 | "Other data scaling laws": beyond dataset size, how does dataset composition affect performance? Picking a data mixture with small models; deciding whether to repeat data; balancing quality against repetition rate

A data law by itself only tells you how fast your model learns. That is useful for forecasting, "not useful for very much else". The engineering questions are about *which* data: how much news versus Wikipedia, whether to repeat data or spend the compute elsewhere, whether to repeat only the high-quality part. Can scaling laws answer those?

::slide 22 | Hashimoto 2021: "Data composition affects the offset, not the slope"; left, excess error against training data size for three mixture proportions q = 0.00, 0.22, 0.56, parallel lines on log-log axes; right, the intercept log C(q) against the data-source proportion, U-shaped with its minimum near the middle

Go back to the view of data laws as empirical generalization bounds. There, the **slope** is set by the model class and the **intercept** by the data distribution. So, "for many models", changing the composition of the data moves the line up or down without changing its slope. The left plot shows it for a toy problem, a linear regression trained on a mixture of two sources: the three mixture proportions give three parallel lines. The right plot shows the intercept as a function of the mixture. Using only one source ($q = 0$ or $q = 1$) gives a high intercept, and a mixture near the middle is best. That is a quantitative argument for **diverse data**.

The practical consequence is strong. If composition changes only the offset, the lines for two mixtures never cross, so the mixture that wins at small scale also wins at large scale.

::slide 23 | "Natural idea: build data scaling laws": the Data Mixing Laws paper (Ye et al.), fitting training-step, model-size and mixture laws to predict unseen mixtures at large scale; "Empirical eval: just take the best small dataset": DataDecide (Magnusson et al.), where ranking 25 datasets with 150M-parameter models predicts the 1B ranking of pairs about 80% of the time

The natural idea is to fit a law on mixtures: train small models on a few news/Wikipedia proportions, fit how loss depends on the proportion, scale up step by step, and extrapolate to find the best mixture at full size. That is the data-mixing-laws paper on the left. In practice, the professor says, people who do this work report that "reality is a lot more noisy". What usually happens is simpler: train a bunch of small models, pick the best mixture, and scale it up, "no scaling law required". The DataDecide study on the right is a large empirical test of that recipe; per the slide image, pre-training on 25 datasets at 150M parameters predicts which of a pair wins at 1B about 80% of the time. And that it works is "consistent with the argument that the intercepts differ, but the slopes don't change".

::predict data-composition-offset-not-slope
::widget fixture:lecture_09--two-laws | press "B: same slope, +0.1 offset" and move the target scale: the gap readout is the same at every scale, so the small-scale ranking never flips
::note aside 24:51 | DataDecide is given as a pointer for anyone interested in data-mixture selection at scale.
::kp data-composition-offset-not-slope

::slide 24 | Muennighoff et al., "Scaling Data-Constrained Language Models": left, final test loss against tokens and epochs, with "up to ≈4 epochs repeating is almost as good as new data", "rapidly diminishing returns", "at ≈40 epochs, repeating is worthless"; right, at 10²² FLOPs the naive frontier picks 8.67B parameters on 178B tokens (loss 2.376), the data-constrained one 6.34B on 242B (loss 2.359); the formula D′ = U_D + U_D·R_D*·(1 − e^(−R_D/R_D*))

Compute keeps growing; the amount of good text does not. So what happens when you have to repeat data? The paper on this slide (named aloud and on the slide image) found that "up to four epochs with standard training recipes, you just don't get hurt at all". Past that, the real curve (solid) falls away from the fresh-data projection (dotted), and by about 40 epochs another pass is worth almost nothing.

Their fix is to count **effective data**. With $U_D$ unique tokens repeated $R_D$ extra times,

$$ D' = U_D + U_D R_D^{*}\left(1 - e^{-R_D/R_D^{*}}\right) $$

The first pass counts in full. Each repetition counts for less than the one before, and however many times you repeat, $D'$ never exceeds $U_D(1 + R_D^*)$. $R_D^*$ is a fitted constant. With the paper's value, about 15.4 (not on the slide), 4 epochs are still worth 93% of 4 fresh passes, while 40 epochs are worth about 38% of 40 fresh passes and the 40th epoch adds only 8% of a fresh one.

The right plot shows what this does to planning. At $10^{22}$ FLOPs with about 25B unique tokens, a planner who treats repeats as fresh picks 8.67B parameters on 178B tokens (7.1 epochs). The data-constrained law picks a smaller model, 6.34B, and more passes, 242B tokens (9.7 epochs), and it reaches the lower loss, 2.359 against 2.376. Both are about the same compute: $6 \times 8.67\times10^9 \times 178\times10^9 \approx 6 \times 6.34\times10^9 \times 242\times10^9 \approx 9.2\times10^{21}$.

::widget fixture:lecture_09--effective-data | add epochs: the yellow bar (what the last epoch added) shrinks while tokens seen grow linearly, and D′ flattens toward the dashed ceiling 1 + R_D*

::slide 25 | "Pre-training under infinite compute" (Kim, Kotha, Liang, Hashimoto): important notes, scaling laws can "break" if applied blindly, and are lower bounds; loss against epoch count (best near 8 epochs, worse beyond), against parameter count (nearly flat from 150M to 1.4B), and against seed tokens for the standard recipe (fit 1.30/D^0.23 + 1.89), regularized (1.03/D^0.23 + 1.96) and ensembled (0.88/D^0.24 + 1.90) models

Take repetition to the extreme: a fixed amount of data and unlimited compute. This is recent work of the professor's group with Percy Liang. You cannot just keep adding epochs (the left plot bottoms out near 8 epochs and then gets worse), and you cannot just keep growing the model (the middle plot is nearly flat). You end up reaching for other things, regularization and ensembling. The interesting part is the right plot. Each intervention lowers the curve, but the fitted exponents on the slide image are 0.23, 0.23 and 0.24: the slopes are "surprisingly similar", and only the intercepts move. "Very, very often your slopes don't change."

The two notes at the top apply to everything in this lecture. A law fitted in one regime can break when pushed outside it. And a law is a **lower bound** on what is achievable, in the sense that it describes one recipe; a better recipe can beat it.

::note spoken 29:22 | A student asked why the right plot looks linear rather than log-log. The x-axis is a bad axis: it doubles at each tick (209M, 419M, 839M, 1.67B), so it is log-scaled, and the y-range is so narrow that linear and log look the same. The fitted curves are still power laws.

::slide 26 | Goyal et al., "Scaling Laws for Data Filtering: Data Curation cannot be Compute Agnostic": "Given that repeated data is less valuable.. Data selection should then be adaptive to scale!"; quality buckets E (best) to F, the quality-quantity trade-off, and estimated error curves where highly aggressive filtering wins at small compute and less aggressive filtering at large compute

Repetition makes **data filtering** depend on scale. If you or I filtered data tomorrow, we would filter "very aggressively" and keep only the best, because with little compute we cannot train on the whole internet anyway. A lab with a lot of compute cannot do that: keeping only the top slice would mean repeating it many times, and a repeated high-quality token is worth less than an unseen mediocre one. So "as you get more and more compute, your filter has become looser and looser". On the right, the curve for the best bucket alone (green) wins at small compute and then flattens, as its few tokens get repeated, while the looser pools keep improving.

**Worked example.** A filter keeps 10% of a corpus and the budget is exactly that 10%: one pass, nothing repeated. Raise the budget 100×: the same filter now needs 100 epochs, and by the effective-data formula those tokens are worth only about 16% of fresh ones. Keeping the whole corpus instead needs 10 epochs, worth about 78%. Unless the filtered tokens are several times better per token, the looser filter wins.

::predict data-repetition-effective-data
::widget fixture:lecture_09--effective-data | in the right panel, grow the budget D with the filter fixed: the strict filter's share of tokens that still count collapses, while a looser filter keeps most of its value
::video 27:19-28:28 | why a small-compute lab filters hard and a large-compute lab loosens its filter
::note aside 26:17 | The infinite-compute study is shown only for its slopes; its regularization and ensembling methods are not explained.
::kp data-repetition-effective-data

::slide 27 | recap: a remarkably linear relationship between log data size and log error, across domains and models; theory: similar to generalization bounds (mean estimation); applications: data collection and curation

The data half ends with a recap the professor calls uncontroversial: more data improves performance in a very predictable way, and the only surprise is how polynomial it stays for such big models.

::note warning 29:49 | Over "a tiny slice of a compute range", polynomial and exponential growth are hard to tell apart, "because Taylor approximations are a thing. Everything looks linear if you zoom in enough." Before trusting a fitted form, make sure the data span a wide range.

## Transformer or LSTM, Adam or SGD? {#architecture}
source: lecture_09.pdf p28-p32 · video 30:13-35:38

::slide 28 | "Scaling laws for model engineering": how can we efficiently design huge LMs (LSTMs vs Transformers, Adam vs SGD)? How should we allocate limited resources (train longer vs bigger, more data vs more GPUs)? "Scaling laws provide a simple procedure to answer these."

Now the part promised at the start: using scaling laws to design the model. Maybe you are "a radical" who believes LSTMs are the future and wants to "blow my B200 run" on one. Maybe you want SGD instead of Adam. Or you have a fixed budget and must decide between a longer run and a bigger model. In [L3](#/read/lecture_03) the answer to such questions was "look at what others did". Here the answer is first-principles: try it at small scale and fit a law. The slide after this one lists the hyperparameters taken in turn, mostly through the Kaplan paper: architecture, optimizer, aspect ratio and depth, batch size.

::slide 30 | test loss against non-embedding parameters (10⁵ to 10⁹): Transformers on one straight line from about 5.2 to 2.3; LSTMs with 1, 2 and 4 layers above it, the gap widening with size

Are Transformers better than LSTMs? The brute-force answer is to spend tens of millions of dollars training an LSTM the size of GPT-3. The scaling-law answer is to train both families at a range of small sizes and compare the lines. Here the LSTMs have "definitely different intercept, maybe even different slope". That justifies scaling the Transformer.

The slope is the thing to watch. A worse intercept is a constant handicap. A worse slope means the gap *grows* with scale, so a model that looks competitive at 100M parameters falls further behind at every step up. You "certainly don't want something where your slopes are worse". Every new-architecture paper today (Mamba, Gated DeltaNet and so on) has a plot of exactly this kind, "vanilla transformer, our really cool model", to show its line sits below.

**Worked example.** Two options follow $\log_{10} L = 0.8 - 0.05\log_{10} N$ (A) and $\log_{10} L = 1.25 - 0.10\log_{10} N$ (B). At $N = 10^8$, A gives 0.40 and B gives 0.45, so A wins. Set them equal: $0.05x = 0.45$, $x = 9$. Above $N = 10^9$ the steeper B wins. A comparison at one size cannot tell you this; two fitted lines can.

::predict architecture-optimizer-via-scaling
::widget fixture:lecture_09--two-laws | press "B: steeper, starts worse": the gap shrinks by the same amount per decade and the lines cross at one marked scale; a ranking read on one side of the crossing is wrong on the other

::slide 31 | Tay et al. 2022, cross-architecture scaling of T5-style models: negative log-perplexity against FLOPs for many architectures (left), and per-architecture panels against the vanilla Transformer: ALBERT, DConv, Evolved, Funnel, Transformer-GLU, LConv, MLP Mixer, MoS Transformer, Performer, Switch Transformer, Universal Transformer

There are fewer good architecture scaling studies than one would like. The professor's favourite is this one from Google, which trained growing T5-style models in many architectures. What makes it worth showing is that its small-scale trends anticipated today's frontier choices. The efficient-attention **Performer** "doesn't scale very well", and nobody uses it. The **gated linear unit** (GLU) is better "throughout the scaling trends", and every modern model uses one. The **Switch Transformer** (a mixture of experts) scales well. One counter-example: mixture of softmaxes looks effective, yet is not used today. Hence the slogan many people go by: "if it doesn't show up in the scaling law, it's not a good intervention."

::slide 32 | Hestness et al. 2017, minimum validation loss against training characters for depth-10 recurrent highway networks: SGD ε(m) = 5.37·m^−0.094, Adam ε(m) = 5.25·m^−0.095; "Note, this is in 2017, so pre-transformers"

Is Adam better than SGD? The fits on the slide image answer precisely: SGD $5.37\,m^{-0.094}$, Adam $5.25\,m^{-0.095}$. Adam is lower by a roughly constant factor, $5.25/5.37 \approx 0.98$, and the exponents differ by 0.001. "The intercepts are different, but the slopes are very similar." The professor finds this mysterious every time: "It's rare to get different slopes, even with an intervention as big as SGD to ADAM." The model is a recurrent highway network from 2017, before Transformers, so the lesson is about the method, not the architecture.

::video 34:47-35:31 | SGD and Adam differ in intercept, not slope, and why he finds that surprising
::kp architecture-optimizer-via-scaling

## Deep or wide, and what counts as a parameter? {#shape}
source: lecture_09.pdf p33-p36 · video 35:38-41:03

::slide 33 | Kaplan: test loss against non-embedding parameters for 1, 2, 3, 6 and >6 layers; "1 vs 2 layers makes a huge difference"; "More layers have diminishing returns below 10⁷ params"

How should you trade depth for width? Sweep the number of layers and plot each depth's scaling line. A single layer is "a terrible, terrible scaling trend": its line is visibly flatter. From two layers on, the lines are "surprisingly more competitive" and nearly merge, although in this figure more layers are still a little better at every size.

::slide 34 | Kaplan Figure 5, loss increase against shape at fixed parameter count: feed-forward ratio d_ff/d_model, aspect ratio d_model/n_layer for 50M, 274M and 1.5B models ("A wide range of architectures achieve similar performance"), attention head dimension; caption: aspect ratio can vary by a factor of 40 with little impact, and (n_layer, d_model) = (6, 4288) reaches a loss within 3% of (48, 1600)

A finer study needs the right axis. The professor's point, which the slide does not spell out, is to look for **scale-invariant quantities**. The number of layers is not one: "As you make your model bigger, you do want more layers", so its optimum moves with size by construction, and a sweep over it cannot tell you whether your recipe is stable. The **aspect ratio** $d_{model}/n_{layer}$ might be one, since the optimal *shape* could stay the same as the model grows.

The middle panel checks it. For 50M, 274M and 1.5B parameters, loss is within a few percent of the minimum over aspect ratios from about 10 to a few hundred, and the minimum sits "around 100d model for every layer, or maybe a little bit less" at every size, with a slight drift. The caption on the slide image puts it strongly: the aspect ratio can vary by a factor of 40 at little cost, and a 6-layer model of width 4288 lands within 3% of a 48-layer model of width 1600. The same analysis works for the feed-forward ratio and the head dimension (left and right panels).

So if your recipe is "fix the aspect ratio and scale up", a plot like this is your evidence that it is safe: the optimum does not move as the model grows.

**Worked example.** A small sweep finds 6 layers of width 512 best, aspect ratio $512/6 \approx 85$. For a model of width 4096, carry the ratio, not the layer count: $4096/85 \approx 48$ layers. With $N \approx 12\,n_{layer}\,d_{model}^2$ and the aspect ratio fixed, $N \propto n_{layer}^3$, so 8× the parameters means $8^{1/3} = 2\times$ the layers and 2× the width.

::predict scale-invariant-hyperparameters
::widget fixture:lecture_09--scale-up | grow the parameter multiplier at a fixed aspect ratio: layers and width both grow as k^(1/3), so the small model's layer count does not carry over; tick "hold the layer count fixed" and the model stretches wide and the aspect readout turns red
::video 36:21-37:34 | "The number of layers is not a scale-invariant quantity", and what he sweeps instead
::note slip 35:38 | The professor says "in lecture 2, I told you about aspect ratios ... It's like four times the reasonable multiplier". Aspect ratios were in the architecture lecture ([L3](#/read/lecture_03)), where the common values are about 100-200; "four times" is the feed-forward multiplier $d_{ff}/d_{model}$ from the same lecture.
::kp scale-invariant-hyperparameters

::slide 35 | test loss against parameters with embeddings (0 to >6 layers: curves bend, the 0- and 1-layer lines almost flat) and against non-embedding parameters (the clean straight lines of slide 33); "Embedding layer parameters don't behave the same!"; related: scaling laws for mixtures of experts

"Not all parameters are created equal", and your scaling laws "may look good or bad depending on how you define what a parameter is". Kaplan noticed that counting the embedding parameters gave "very funky-looking scaling laws" (left: bent curves, and the 0-layer model, which is nearly all embedding, barely improves at all). So they counted only **non-embedding parameters**, on the grounds that those are the ones "doing computation", and got the clean lines on the right. That choice will matter later: it is part of why Kaplan and Chinchilla disagree.

The professor adds Percy Liang's point: predictable scaling "is engineered". It does not happen automatically. You get it by choosing the right x-axis and setting the hyperparameters right at every scale, and only then does it hold over many orders of magnitude.

::slide 36 | Abnar et al. (Apple, MIT), "Parameters vs FLOPs: Scaling Laws for Optimal Sparsity for Mixture-of-Experts Language Models": pretraining loss against total parameters and against active parameters, curves coloured by sparsity from 0% to 95% with their minima starred; an IsoFLOP surface over sparsity and active parameters

Mixtures of experts, now the dominant way to train large models ([L4](#/read/lecture_04)), make the question sharper, because total and active parameters are decoupled. This Apple/MIT study plots loss against both. As the total parameter count grows, the loss-minimizing model becomes sparser and sparser (the stars move to darker curves). And at a fixed number of active parameters, that is at a fixed compute cost, adding total parameters still lowers the loss: the inactive experts help.

**Worked example.** Model X has 100M parameters, 40M of them embeddings; model Y has 80M, 10M of them embeddings. On a Kaplan-style law counted in non-embedding parameters, X has 60M and Y has 70M, so Y is the bigger model where it matters and should reach the lower loss.

::predict depth-width-and-parameter-quality
::video 37:34-38:53 | why Kaplan dropped embedding parameters, and the warning that predictable scaling "is engineered"
::note skip 40:46 | The functional form of the MoE scaling law ("you can write down what the functional forms of this are") is skipped.
::note deferred 38:23 | The consequences of Kaplan's non-embedding count are promised for later; they appear in the Kaplan-versus-Chinchilla section below.
::kp depth-width-and-parameter-quality

## How big a batch, and what learning rate? {#batch-lr}
source: lecture_09.pdf p37-p40 · video 41:03-50:55

For a new big model, most of the choices above are already settled: you will not switch to an LSTM, you will pick "a DeepSeek-V4-inspired MoE or something". Two things, though, you really do have to re-derive for every run: the **batch size** and the **learning rate**. They are coupled ("You change one, you have to change the other"), and the batch size carries a systems stake too: data parallelism ([L7](#/read/lecture_07)) needs a large batch. So the relevant question is how large you can make the batch before you start to suffer, and how that changes with scale.

::slide 37 | McCandlish, Kaplan, Amodei, "An Empirical Model of Large-Batch Training": a loss landscape where a smaller batch's noisy step (red) and a larger batch's step (blue) both point along the local gradient, not at the minimum; predicted training speed against batch size / noise scale, "perfect scaling" below 1 and "ineffective scaling" above; "Critical batch = min number of examples before diminishing returns"

Increasing the batch has strong diminishing returns past a point. The professor explains the mechanism with the picture:
- **Noise-limited regime.** With a small batch, the gradient estimate is noisy. Each extra example in the batch reduces that noise, and since noise is what limits you, the returns are "perfect": doubling the batch roughly halves the steps you need.
- **Bias-limited regime.** Once the batch is big enough that the noise is small, a better gradient estimate stops helping. Gradient descent sees only the local slope, and the local descent direction (the arrow) does not point at the minimum. No amount of averaging fixes that disagreement, so more examples per step buy almost nothing.

The **critical batch size** is the crossover between the two, a convenient trade-off point: the largest batch you can use before the efficiency losses get large.

::slide 38 | the definition: pick a target loss, sweep batch sizes, record steps S and examples E needed; the curve follows S/S_min − 1 = (E/E_min − 1)^−1; fit S_min, E_min and pick B_crit = E_min/S_min; "This balances both sides of the equation, giving roughly 2x the steps / passes optimal"; claimed to be close to the trace of the gradient covariance over the squared norm of the gradient

The derivation is "complicated calculations about the local quadratic approximations to the objective", so the lecture gives only the mechanical recipe:
1. Pick a target loss you want to reach as quickly as possible.
2. Sweep the batch size $B$. For each, record the steps $S$ and the examples $E = B \cdot S$ needed to reach the target.
3. Fit the trade-off curve printed on the slide image,

$$ \left(\frac{S}{S_{min}} - 1\right)\left(\frac{E}{E_{min}} - 1\right) = 1 $$

where $S_{min}$ is the fewest steps possible (with a huge batch) and $E_{min}$ the fewest examples possible (with a tiny batch). You cannot have both.
4. Pick the balanced point, $B_{crit} = E_{min}/S_{min}$.

At the balanced point both brackets equal 1, so you pay $2 S_{min}$ steps and $2 E_{min}$ examples: twice the minimum of each, which the professor calls "a little bit more" than optimal on both sides. The slide adds that $B_{crit}$ is close to the **gradient noise scale**, the trace of the gradient covariance divided by the squared gradient norm: how big the noise is relative to the signal.

::worked critical-batch-size
::widget fixture:lecture_09--critical-batch | press "set B = B_crit": the point sits at 2× S_min steps and 2× E_min examples; far below B_crit doubling the batch nearly halves the steps, at B_crit it saves 25%, far above it almost nothing
::predict critical-batch-size
::note skip 46:37 | Estimating $B_{crit}$ from the gradient noise scale is left to the paper: "You should really read the critical batch size paper."

::slide 39 | Kaplan: critical batch size (tokens) against WebText2 training loss, for 3M and 85M models and noise-scale measurements, with the fit B_crit = 2.1·10⁸ tokens · L^−4.8; "The smaller the loss target, the bigger the batch"; C_min(C) = C / (1 + B/B_crit(L))

Why is this in a scaling lecture? Because the critical batch is a reasonable batch to pick, and it changes in a predictable way as you train a better model. The target loss stands in for compute: the better the model you want, the lower the target. The fit on the slide image is a power law in the loss:

$$ B_{crit} \approx 2.1\times10^{8}\ \text{tokens} \cdot L^{-4.8} $$

**Worked example.** At a target loss of 4, $4^{4.8} \approx 780$, so $B_{crit} \approx 2.7\times10^{5}$ tokens. At a target of 3, $3^{4.8} \approx 195$, so $B_{crit} \approx 1.1\times10^{6}$ tokens. Lowering the target from 4 to 3 multiplies the critical batch by $(4/3)^{4.8} \approx 4$. A large run that ends far down this curve can use a very large batch, which is good news for data parallelism.

The professor's reason: near the minimum you are trying to resolve "really, really tiny differences", so the noise matters more and variance reduction is worth more.

::video 42:34-43:59 | noise-limited versus bias-limited regimes, and why returns diminish past the critical batch
::note slip 47:48 | He says the relationship is "roughly inverse polynomial in batch size". It is the critical batch that is an inverse polynomial in the target *loss*, as his previous sentence and the slide's fit say.
::kp critical-batch-size

### The learning rate moves with width
The learning-rate story is "also a little bit complicated", and most of it is deferred to the advanced lecture. The mental picture he gives is for scaling the width of a standard network (he sets depth aside). The wider the model, the *smaller* the learning rate should be, because "I have bigger, more parameters, I'm changing more things at once. Maybe I should move less." A common rule of thumb scales the learning rate as $1/\text{width}$.

**Worked example.** Tuned at 6e-3 on width 512, the rule gives $6\times10^{-3} / (3072/512) = 10^{-3}$ at width 3072. Run the wider model at 6e-3 instead and you are 6× too high.

::predict lr-shrinks-with-width
::widget fixture:lecture_09--scale-up | grow the width: under standard parametrization the learning-rate readout divides by the width factor, and keeping the base rate is that many times too high; the muP value stays at the tuned rate
::video 48:57-49:32 | "the bigger my model, the smaller my learning rate should be", and the 1-over-width rule of thumb
::kp lr-shrinks-with-width

::slide 40 | Yang et al. 2022: training loss against log₂ learning rate for widths 128 to 8192, the optimum shifting under standard practice and staying put under muP ("optimum stable"); Yao et al. 2024's muP table for a model r times wider: matrix-like AdamW learning rate l → l/r, matrix-like init variance σ → σ/r, output multiplier τ → τ/r, others unchanged

There are two philosophies, and the plot shows both. **Standard practice** (left): sweep the learning rate at each width and watch the optimum move; it moves “pretty predictably”, so you can fit a law to where it goes and predict the optimum for your big model. **muP** and its relatives (right) rescale the network instead: change the initialization sizes and the optimizer's step sizes for different parts of the network so that the loss-minimizing learning rate stays the same at every width. Then you tune once at small scale and reuse the value. The table on the slide image is the recipe for a model $r$ times wider: divide the hidden (“matrix-like”) layers' AdamW learning rate and initialization variance by $r$, divide the output multiplier by $r$, leave embeddings and other parameters alone.

Some report great success with muP, others less; both approaches have been used in large runs, and "anecdotally" more people seem to favour fitting the moving optimum. Either way, a scaling study that uses one fixed learning rate at every size is biased: the larger models are increasingly mis-tuned, and the fitted slope partly measures that error.

::video 50:08-50:41 | the two philosophies: predict where the optimum moves, or reparametrize so it stays put
::note deferred 50:50 | "I'll talk in detail about both of these next lecture": he means the next *scaling* lecture, since the next lecture is inference.
::kp mup-scale-aware-lr

## The design procedure, and where it can mislead {#procedure}
source: lecture_09.pdf p41-p42 · video 50:55-56:36

::slide 41 | Tay et al. 2023: negative log-perplexity against parameters for many shapes (NL12 best, about −1.36 at about 1.7·10¹⁰ parameters), and SuperGLUE accuracy against parameters (NL32-XL best, about 79.9, while NL12 scores about 77.9); "downstream scaling can often be much less predictable"

One caution before the procedure. Upstream, loss or perplexity against parameters is a beautiful line. So you ship the model with the best perplexity, NL12 in the left plot. But on the downstream benchmark (right) the best model is NL32-XL, which is much worse in perplexity. This is "one of the worst correlations" between upstream and downstream the professor has seen, but the effect is common. Scaling laws are clean, regular and predictable on the perplexity side; "transfer from perplexity to downstream is a lot less certain than it might initially seem".

::note aside 52:13 | His former students in post-training "always complain" that pre-training teams hand over a model with good perplexity and say "It's all your problem now", when the problems started in pre-training.

::slide 42 | "The effect of hyperparameters on big LMs can be predicted before training!" (optimizer, depth, architecture); the scaling-law-based design procedure: 1. train a few smaller models, 2. establish a scaling law (e.g. Adam vs SGD), 3. select the optimal hyperparameter from the law's prediction

Before a big run you should know roughly what you will get, not just that it will train. With scaling laws, "you should be able to actually predict fairly precisely the numerical values" of how good the run will be, and what one optimizer gains over another. The procedure:
1. Train a few smaller models for each option.
2. Fit a scaling law per option, a line on log-log axes.
3. If the fit is good enough, trust "that the gap will persist", and run the winner at scale.

He calls this the naive version, but a reasonably accurate picture of practice: "What else are you going to do? Just deploy a run out of nowhere?"

::worked scaling-law-design-procedure
::predict scaling-law-design-procedure

The lecture has now collected four cautions for step 3:
- a law can break if you extrapolate it blindly (slide 25);
- a law is a lower bound: it describes one recipe, and a better recipe can land below it (slide 25);
- over a narrow range, polynomial and exponential look alike (the warning after slide 27);
- downstream metrics scale much less predictably than loss (slide 41).

**Worked example.** A law fitted on 10M-1B-parameter models predicts a loss of 2.10 at 100B; the real run reaches 2.02. That is not a failure of the law. It is the lower-bound caution: the big run's recipe was better than the small runs'.

### Questions from the class
**How many runs per point?** Usually one. Perplexities are very clean: with homogeneous training data and big eval sets, a rerun differs "in the second decimal place". Learning-rate and critical-batch scaling laws are different, "truly horrendous stuff", and people do less variance reduction there than they should.

**Why not fit downstream metrics directly?** People do, and it has the same problem as slide 41: a noisy, jagged trend gives no confidence that it continues. His philosophy: establish the regularity on a low-variance measurement first, then argue separately that it transfers downstream.

**Training loss or test loss?** Some plots show one, some the other. Careful science fits test loss, as Kaplan did. But almost all pre-training runs are **one-pass SGD**: each token is seen once, so the generalization gap is "very, very small" and train and validation loss nearly coincide. Some pre-training codebases do not even compute a validation loss. The exceptions are runs that repeat data, like the infinite-compute and repetition studies above, where the model can fit tokens it has seen before and training loss drops below held-out loss.

Why does one-pass training loss track test loss? Each batch's loss is measured before the model has trained on that batch, so it is an honest sample from the training distribution. (That reasoning is ours; the professor only states the rule.)

::predict one-pass-train-loss-is-test-loss
::video 55:55-56:35 | why one-pass SGD makes training and validation loss interchangeable, and which runs are the exception
::kp scaling-law-design-procedure
::kp one-pass-train-loss-is-test-loss

## More data or a bigger model? {#joint}
source: lecture_09.pdf p43-p44 · video 56:36-1:00:11

::slide 43 | "Clearly, lots of data is wasted on small models": Kaplan's loss against tokens for models from 393.2K to 708M parameters, each curve flattening at its own level; joint laws Error = n^−α + m^−β + C (Rosenfeld+ 2020) and Error = [m^−α + n^−1]^β (Kaplan+ 2020); Rosenfeld's error landscape over data and model fractions

Now the most famous use of scaling laws, whose rule of thumb you have probably used even if you never fit one. The resource you are handed is compute, and compute is roughly data times parameters ($C \approx 6ND$, from [L2](#/read/lecture_02)). So: do you want more data or a bigger model?

The plot shows why the question has an answer. The smallest model (393.2K parameters, the top curve on the slide image) is flat from about $10^8$ tokens onward: feeding it more data is "a complete waste of compute". The same tokens would have done far more for one of the big models at the bottom. To allocate compute well you need a law in both variables at once. Kaplan et al. and Rosenfeld et al. proposed, almost simultaneously, two roughly equivalent forms. Rosenfeld's is "really just the sum of two inverse terms":

$$ \text{Error} = n^{-\alpha} + m^{-\beta} + C $$

with $n$ the data, $m$ the model size and $C$ the irreducible error. Kaplan's is a little more complicated, $\text{Error} = [m^{-\alpha} + n^{-1}]^{\beta}$, but has the same behaviour.

A good habit, the professor says, for any law someone shows you: take each variable to infinity. With infinite data the data term vanishes and you are left with a pure model-size law. With an infinite model you are left with a pure data law. Both forms pass the test.

::worked joint-data-model-law

Because the terms add, the larger one acts as a floor. If the model term dominates, more data barely moves the total: that is "data wasted on small models" in one line of algebra.

**Worked example.** Take $\alpha = 0.5$, $\beta = 0.3$, $C = 0$ and start at $n = m = 10^4$: the data term is $10^{-2} = 0.0100$, the model term $10^{-1.2} = 0.0631$, total 0.073. Raise the data 100× to $n = 10^6$: the data term falls to 0.0010, the total only to 0.064. Raise the model 100× instead, to $m = 10^6$: the model term falls to $10^{-1.8} = 0.0158$, total 0.026. Now the two terms are comparable, and further gains need both.

::predict joint-data-model-law
::widget fixture:lecture_09--joint-law | when m^−β is much larger than n^−α, the blue curve for your model size is already flat on its dashed floor: 100× more data lowers the error by a few percent, 100× bigger model by more than half

::slide 44 | Rosenfeld: fit on small data and small models (green dots, model fraction up to 1/16, data fraction up to 1/8), extrapolate to the rest (red); estimated against measured top-1 error on ImageNet (mean −4.5%, σ 4.7%) and test loss on WikiText-103 (mean 0.5%, σ 1.7%); "optimize n^−α + m^−β + C with your costs"

Joint laws also extrapolate well. Rosenfeld fitted only the small corner of the grid (green: at most 1/16 of the full model and 1/8 of the full data) and predicted the large-model, large-data runs (red). On WikiText-103 the predictions are off by 0.5% on average, with a spread of 1.7% (from the slide image).

Once you trust the law, allocation is "a simple nonlinear optimization problem": minimize the joint law subject to a fixed FLOP budget, $6ND = C$.

::video 58:52-59:21 | taking each variable to infinity turns the joint law into a pure model or a pure data law
::kp joint-data-model-law
::kp lecture_02:six-nd

## Kaplan's answer, Chinchilla's answer, and three ways to fit {#chinchilla}
source: lecture_09.pdf p45-p49 · video 1:00:11-1:06:53

::slide 45 | "Kaplan claims: N_opt = C^0.73, D_opt = C^0.27 (tokens per param decreases w/ C)"; "Chinchilla [Hoffman et al] argue these fits are quite off": optimal parameters against FLOPs for Chinchilla's approaches 1-3 and the steeper Kaplan line, with Chinchilla (70B), Gopher (280B), GPT-3 (175B) and Megatron-Turing NLG (530B) starred

Solving that optimization, Kaplan et al. got a lopsided rule:

$$ N_{opt} \propto C^{0.73}, \qquad D_{opt} \propto C^{0.27} $$

The exponents sum to 1, because $C \propto ND$: every factor of compute is split between the model and the data. But the split is uneven. With 100× the compute, the model grows $100^{0.73} \approx 29\times$ and the data only $100^{0.27} \approx 3.5\times$, so tokens per parameter *fall* as compute grows.

::worked kaplan-compute-optimal-allocation
::predict kaplan-compute-optimal-allocation

This rule shaped an era. Around GPT-3, everyone was training "hundreds of billions of parameters", even "trillion-parameter dense models", and "part of that was driven by this". In 2022, Hoffmann et al. at DeepMind (the Chinchilla paper; "Hoffman" on the slide) said these predictions were "all terribly off". In the plot, the three starred giants, GPT-3, Gopher and Megatron-Turing NLG, sit on or near Kaplan's steep dashed line and are "way too big". Chinchilla's three fits form a much shallower line, and the teal star, Chinchilla itself at 70B parameters, is the better model for the same compute. Its famous multiplier is 20 tokens per parameter.

::widget fixture:lecture_09--allocation | multiply the budget by 100: the Kaplan point moves N ×29 but D only ×3.5, so its tokens-per-parameter readout falls; the Chinchilla point moves both ×10
::note slip 1:00:31 | He first reads the exponents the wrong way round (data $C^{0.73}$, model $C^{0.27}$; "they always flip this"), then corrects himself: "I think it's reversed, sorry." Parameters get 0.73, as on the slide.

The professor walks through the Chinchilla paper in detail, not for the ratio but because the disagreement shows that fitting a scaling law "is not just something where you turn the crank".

::slide 46 | Chinchilla's table: coefficient a (N_opt ∝ C^a) and b (D_opt ∝ C^b) with intervals; 1. minimum over training curves 0.50 / 0.50; 2. IsoFLOP profiles 0.49 / 0.51; 3. parametric modelling of the loss 0.46 / 0.54; Kaplan et al. 0.73 / 0.27

Chinchilla fits the trade-off three different ways, which the professor likes as a way of guarding against your own modelling assumptions. The table on the slide image gives the result. Methods 1 and 2 agree, about 0.5 and 0.5: parameters and tokens should grow at the same rate. Method 3 differs a little, 0.46 and 0.54; why comes later. Kaplan's 0.73 and 0.27 are far from all three.

::slide 47 | method 1, minimum over runs: training curves for models from 70M to 10B (each at four cosine cycle lengths) against FLOPs; the envelope of minimal loss per FLOP; optimal parameters and tokens against FLOPs, projected to Gopher's budget (5.76·10²³ FLOPs): 67B parameters, 1.5T tokens

**Method 1, the lower envelope.** Plot every training curve against FLOPs. At each FLOP count, the lowest point over all runs is the best loss anyone achieved with that much compute, and it belongs to one particular run with one model size. Collect those envelope points, scatter their model sizes against FLOPs, and you get a straight line on log-log axes. Kaplan used a version of this too. At Gopher's budget, $5.76\times10^{23}$ FLOPs, the line says 67B parameters and, from the slide image, 1.5T tokens. Simple, but defining "the envelope" has tricky parts.

::slide 48 | method 2, IsoFLOPs: training loss against parameters for nine budgets from 6·10¹⁸ to 3·10²¹ FLOPs, each a valley with a clear minimum; the minima against FLOPs give optimal parameters (63B at Gopher's budget) and tokens (1.4T)

**Method 2, IsoFLOPs**, the professor's personal favourite, "very easy and very robust". Pick a set of FLOP budgets. For each budget, train models of several sizes, each on exactly as many tokens as the budget allows, $D = C/6N$: double the data, halve the model. Each budget traces a valley of final losses. Too small a model is under-sized for the compute; too large a model sees too few tokens. Take the bottom of each valley (or fit a parabola and take its vertex), and fit a line through the bottoms on log-log axes. At Gopher's budget this gives 63B parameters and 1.4T tokens, close to method 1.

::animation fixture:isoflop-sweep | along one IsoFLOP curve N rises as D = C/6N falls, and the loss has one minimum; each larger budget's minimum sits further right, tracing N_opt ∝ C^a with a ≈ 0.5
::widget fixture:chinchilla--isoflop-parabola | slide the sampled sizes to one side: the vertex is trustworthy only when the lowest sampled loss has a higher neighbour on both sides, otherwise the readout says "not bracketed"

**Worked example.** Suppose your IsoFLOP minima land at 1B parameters for $10^{20}$ FLOPs and 4B for $10^{21}$. The exponent is the slope between them: $a = \log_{10}(4)/\log_{10}(10) \approx 0.60$. Before trusting it, check that each valley's minimum was inside the swept range; a minimum at the edge of a sweep is only a bound.

::predict chinchilla-three-methods

::slide 49 | method 3, joint fits: a parametric loss L̂(N, D) fitted by least squares to all runs on the size-data grid; iso-loss contours with the efficient frontier, projecting 40B parameters at Gopher's budget; IsoFLOP slices of the fit against the data

**Method 3, the joint fit**, the most natural one given the last section: hypothesize a joint form for loss as a function of $N$ and $D$, fit its constants to all your runs by least squares, and minimize it under the FLOP constraint. The fitted surface's frontier gives 40B parameters at Gopher's budget. It is the most brute-force approach, and its weak point is the curve fitting itself: a surface in several variables is "kind of tricky" to fit.

::video 1:04:37-1:05:47 | sweep N at a fixed FLOP budget, fit a quadratic per budget, then a line through the minima
::kp kaplan-compute-optimal-allocation
::kp chinchilla-three-methods

## Why did Kaplan and Chinchilla disagree? {#discrepancy}
source: lecture_09.pdf p50-p53 · video 1:06:53-1:14:21

::slide 50 | "Why such a big difference (when both fit joint scaling laws?)": the slide 45 plot again, Kaplan's steep line against Chinchilla's three

Nothing Chinchilla did was materially different from what Kaplan did. Both ran sensible studies and fitted joint laws. Yet the predictions are very different, and in hindsight most people side with Chinchilla: "If you do your own Chinchilla-style analysis, you will find probably scaling is much closer to Chinchilla." So where does the gap come from? This is "where we get into the messy realities of how scaling laws are made": the result depends on implementation details, hyperparameters, and even on what the x-axis is.

::slide 51 | "Resolving Discrepancies in Compute-Optimal Scaling of Language Models" (Porian, ..., Wortsman, Jitsev, Schmidt, Carmon): a reproduction of Kaplan (a = 0.835, N* = 3T at the target budget), then counting last-layer FLOPs (a = 0.706, 787B), correcting warmup (a = 0.602, 292B), and either cosine decay without tuning (a = 0.571, 183B) or optimizer tuning without decay (a = 0.497, 77B); bullets: last-layer parameters removed from the count, warmup too long at very small budgets, decay maybe not critical if batch and LR are tuned

**Explanation 1** reproduces Kaplan's result in Kaplan's setting and then makes small changes one at a time; the exponents in the panels are from the slide image:
- **Count the last layer.** Kaplan excluded the embedding parameters, which is "generally an OK thing to do". But they also excluded the final output (softmax) layer, because it has the same shape as the embedding (vocabulary by hidden size, transposed). Counting it moves the exponent $a$ from 0.835 to 0.706. Whether you include those parameters "has a big material impact on the shape of the scaling law".
- **Fix the warmup.** The learning-rate warmup was too long for the smallest budgets. Many of Kaplan's models were so small, and their runs so short, that they were not converging by the time warmup was done, so "their learning rates were set very suboptimally". Fixing it gives 0.602.
- **Tune the batch size and optimizer.** Kaplan used one big batch size for everything, too big for the small models. Tuning it per model, without even a decay schedule, gives 0.497: Chinchilla's answer.

Each change looks minor. Together they turn a 3T-parameter recommendation into a 77B one. This is also what "scaling laws are lower bounds" means in practice: a law says "if I continue this recipe and I scale it up, then this is what I will get". Scale up a recipe with "crazy" warmup or batch sizes and you get a bad law, so the small runs should be "as close to the proper full run as possible".

::note aside 1:07:55 | He covered the authors on his own slide, offers to post the citation in Slack, and names only the last author, Yair Carmon, a former student. The caption calls the paper "Resolving Discrepancy to ..."; the title on the slide image is "Resolving Discrepancies in Compute-Optimal Scaling of Language Models".

::slide 52 | Pearce and Song, "Reconciling Kaplan and Chinchilla Scaling Laws": start from a fitted Chinchilla loss, Loss(N_T, D) = 482/N_T^0.35 + 2085/D^0.37 + 1.82; simulate training curves for Kaplan's model sizes (1K to 1.5B parameters); in total parameters the frontier exponent is 0.51 (close to Chinchilla's 0.50), in non-embedding parameters the local exponent is 0.78 (close to Kaplan's 0.73); "Non-embedding vs total param choice + small nonlinearities"

**Explanation 2** trains no models at all, which the professor finds clever. It takes Chinchilla's fitted loss surface (on the slide image), simulates the training curves Kaplan would have seen at Kaplan's much smaller model sizes, and fits them both ways. Counted in total parameters, the compute-optimal exponent comes out 0.51, Chinchilla's answer. Counted in non-embedding parameters, the same simulated data give a local exponent of 0.78, close to Kaplan's 0.73. Their story is slightly different from explanation 1's: Kaplan worked at a much smaller compute scale, which is very sensitive to small changes, and at that scale dropping the embedding parameters introduces a slight nonlinearity that tilts the fit.

**Worked example.** Why does the parameter count matter most for small models? Take a model with a 50,000-token vocabulary and width 512. Its embedding matrix alone is about $50{,}000 \times 512 \approx 26$M parameters, twice what four Transformer blocks of that width hold ($12 \times 4 \times 512^2 \approx 12.6$M). Drop it and a small model's size shrinks by a big factor, while a 10B model's size barely changes. The small end of the axis moves, so the fitted slope changes. (Illustrative numbers, not from the lecture.)

::predict kaplan-chinchilla-discrepancy
::video 1:08:53-1:10:20 | the small fixes one at a time: count the last layer, fix warmup for small models, tune the batch size
::kp kaplan-chinchilla-discrepancy

::slide 53 | "Fun addendum: errors in Chinchilla method 3" (Besiroglu et al. 2024): residuals of Hoffmann et al.'s fit, centred near −0.05, against the refit's, centred at 0; optimal tokens per parameter against training compute: Hoffmann's method-3 policy rising from about 20 to well over 100, the refit staying near the D/N = 20 rule of thumb, with the Chinchilla model on it

One loose end, which the professor finds “amusing more than anything else”. Chinchilla's method 3 never agreed with methods 1 and 2, and the authors were unbothered. The numbers look close (0.46 against 0.5), but the implication is not small. Methods 1 and 2 say parameters and tokens grow together, at a fixed ratio; that is where the factor of 20 comes from. Method 3's unequal exponents say tokens per parameter keep growing with compute ($D/N \propto C^{0.54-0.46} = C^{0.08}$), “a very different scaling conclusion asymptotically”. The green line on the right shows how far that drifts.

Researchers at Epoch AI could get neither the raw data nor the code, so they extracted the data points from the paper's plots and refitted method 3. The residuals on the left show the original fit was biased: its errors centre around −0.05 instead of 0, so it had underfit. The refit's optimal policy stays near 20 tokens per parameter. The slide's "recovered the raw data" means this reconstruction from figures. The conclusion: the authors were "more right than they knew", and the method was robust after all.

::widget fixture:chinchilla--parametric-frontier | the readout "tokens per parameter along the frontier" with the paper's original method-3 exponents (α = 0.34, β = 0.28) is rising, about ×1.25 per decade of compute; set α = β and it goes flat, the fixed ratio that methods 1 and 2 imply
::video 1:12:42-1:13:09 | why method 3's slightly unequal exponents give "a very different scaling conclusion asymptotically"
::kp chinchilla-three-methods

## Should you train the Chinchilla-optimal model? {#overtraining}
source: lecture_09.pdf p54-p55 · video 1:14:21-1:16:47

Methods 1 and 2, and the refit of method 3, agree that parameters and tokens should grow at the same rate, at about **20 tokens per parameter**. With $C = 6ND$ and $D = 20N$, a budget fixes the model directly:

$$ C = 6N \cdot 20N = 120\,N^2 \quad\Longrightarrow\quad N = \sqrt{C/120}, \qquad D = 20N $$

::worked chinchilla-twenty-tokens-rule
::predict chinchilla-twenty-tokens-rule
::widget fixture:lecture_09--allocation | the Chinchilla point sits where the D = 20N line crosses the isoFLOP line 6ND = C; multiply the budget by 25 and N and D each grow 5×, while the Kaplan point leaves the line toward more parameters

::slide 54 | "Chinchilla aims to tell you what gives the best model for fixed training compute.. But most of the compute in a real deployment is inference.. So we should 'over' train": GPT-3 2 tokens/param, Chinchilla 20, LLaMA-65B 22, Llama 2 70B 29, Mistral 7B 110, Llama 3 70B 215; "The more usage we expect, the more it becomes worth it to pay the upfront cost"

And yet "you probably don't want the Chinchilla factor". Chinchilla answers one question: the best model for a fixed *training* budget. A production lab does not mostly spend compute on training. External surveys of frontier labs suggest most of it goes to R&D and **serving**. For serving you want "small models that are capable", not big bloated ones that are expensive on every request, even if the big one would have minimized training cost. So you train a smaller model on more tokens than Chinchilla would: "overtraining", in quotes, because for a model that will be served "that's the right amount of training".

The list reads as a history. GPT-3 was badly under-trained (175B parameters on 300B tokens, about 1.7 per parameter, which the slide rounds to 2). Chinchilla set the ratio at 20, and for a while models stayed near it (LLaMA-65B at 22, Llama 2 70B at 29); that was an era when models were "cool, but ... not being served at scale". Once serving became real, ratios jumped (Mistral 7B at 110, Llama 3 70B at 215), and labs moved to mixtures of experts and other inference-driven trade-offs. Llama 3 70B at 215 tokens per parameter means about $70\text{B} \times 215 \approx 15$T tokens, more than ten times the 1.4T that Chinchilla's rule gives for a 70B model.

Why is the trade cheap? Near its minimum the IsoFLOP valley is flat. Halving the model at a fixed budget doubles its tokens (from 20 to 80 per parameter) and raises the loss only slightly, while the cost of every generated token, about $2N$ FLOPs, halves.

::worked overtraining-for-inference
::predict overtraining-for-inference
::widget fixture:chinchilla--parametric-frontier | press "N = N_opt", then slide your N 0.30 decades to the left (half the parameters): the loss excess is under 1%, because the IsoFLOP valley is flat near its minimum, while the per-token serving cost halves
::widget fixture:lecture_09--allocation | press the p54 presets: at a fixed budget, raising tokens per parameter shrinks N and grows D along the same isoFLOP line; the serving-cost readout is 0.5 at 80 tokens per parameter, and the Llama 3 70B preset needs about 15T tokens
::note slip 1:15:14 | He says GPT-3 was "three tokens per parameter"; the slide prints 2, and 300B tokens over 175B parameters is about 1.7.
::note spoken 1:15:43 | Chinchilla matters "not because I think the 20-to-1 ratio is the one golden ratio" (it may be, for research runs), but because of what it teaches about how to fit scaling laws.

**Worked example: back to the ten thousand GPUs.** The opening budget was at least $1.3\times10^{25}$ FLOPs. Chinchilla's rule gives $N = \sqrt{1.3\times10^{25}/120} \approx 3.3\times10^{11}$: a 330B-parameter model on about 6.5T tokens. If the model will be served heavily, train it Llama-3 style at about 215 tokens per parameter instead: $C = 6 \cdot 215\,N^2$ gives $N \approx 10^{11}$, a 100B model on about 21T tokens, with a somewhat higher loss and about a third of the serving cost per token. (Whether 21T good unique tokens exist is the repetition question from earlier.)

::slide 55 | "Isoflops everywhere": IsoFLOP profiles for autoregressive and diffusion language models and their compute frontiers (Gulrajani+ 2023); IsoFLOP surfaces over MoE sparsity and total or active parameters (Abnar+ 2025)

The method outlived the ratio. IsoFLOP sweeps are "very easy to execute": fix a FLOP budget and sweep the other degrees of freedom to see the shape of the loss surface. They have been used for diffusion language models (one of the professor's students; top) and for the MoE sparsity study from slide 36 (bottom). "If you're ever in a situation where you're thinking, how am I going to decide all these trade-offs, IsoFLOP is always a good default."

::video 1:14:25-1:15:14 | why "overtrained" is in quotes: for a model that will be served it is the right amount of training
::kp chinchilla-twenty-tokens-rule
::kp overtraining-for-inference
::kp chinchilla-three-methods

## What should you carry away? {#summary}
source: lecture_09.pdf p56-p57 · video 1:16:47-1:17:52

::slide 57 | recap: data scaling (understand how data affects models, clean theory); model scaling (dramatically reduce costs for training); scaling as prediction (understand what problems can be "brute forced")

The lecture's single fact is a log-linear regularity between the resources you put in and the loss you get out, and it extends from data to parameters, compute and even MoE sparsity. That regularity lets you make choices that would otherwise be guesses, with evidence from small runs instead of big ones.

| question | what scaling laws say | where |
|---|---|---|
| why a power law? | estimation error decays polynomially; flexible learners have shallow, dimension-dependent slopes | slides 16-20 |
| near the floor? | fit $L = E + A n^{-\alpha}$ and plot $L - E$, or keep the model much bigger than the data | Q&A after slide 20 |
| which data mixture? | composition moves the offset, not the slope: the best small-scale mix stays best | slides 22-23 |
| repeat data? | about 4 epochs are nearly free; effective data saturates, so filters loosen with compute | slides 24-26 |
| architecture, optimizer? | fit a law per option; watch the slopes, since only different slopes cross | slides 30-32 |
| model shape? | carry the aspect ratio (about 100), not the layer count; count non-embedding parameters | slides 33-35 |
| batch size? | about $B_{crit} = E_{min}/S_{min}$, which grows as the target loss falls | slides 37-39 |
| learning rate? | falls about as 1/width under standard parametrization; or use muP and keep it fixed | slide 40 |
| more data or bigger model? | joint law plus $C = 6ND$; Chinchilla: about 20 tokens per parameter, $N = \sqrt{C/120}$ | slides 43-49 |
| why Kaplan said otherwise | parameter counting, warmup and batch size on small runs | slides 50-52 |
| what to actually train | a smaller model, over-trained, when serving dominates | slide 54 |

The other lesson is about method. Scaling laws are lower bounds on a recipe, predictable scaling is engineered, and tiny choices in how the small runs are set up can move a recommendation by more than an order of magnitude. Next comes Percy's lecture on inference, then the advanced scaling lecture.

::note deferred 1:17:41 | Advanced scaling topics (modern open-model reports, muP and other parametrizations, optimizers) return after the inference lecture.
::kp scaling-law-definition
::kp scaling-law-design-procedure
