---
title: L14 · Data II, read through
minutes: 45
---
The previous lecture ([L13](#/read/lecture_13)) asked where pretraining data comes from. This one follows the data through the pipeline that turns a raw crawl into a training set: extract text, filter it, deduplicate it, mix sources, and then, for post-training, generate data with a stronger model. After it you can build a filter from a small example set, compute what MinHash and locality-sensitive hashing will and will not catch, count how many epochs a mixture really asks for, and say how today's synthetic and agentic coding datasets are made.

## Raw data is not text: how do you get text out of it? {#transformation}
source: lecture_14.py:L12-L57 · video 0:05-6:42

A one-line recap of [L13](#/read/lecture_13): data does not "fall from the sky". The internet is live services (GitHub, websites); their content has to be dumped or crawled (GitHub Archive, Common Crawl); and then processed into a dataset (The Stack). Along the way you deal with terms of service and copyright, which means either a license or an argument for fair use.

This lecture picks up at the processing step. The plan is the pipeline in order (transformation, filtering, deduplication, mixing), all of it about **pretraining** data, then a quick look at post-training data and how people use synthetic data for mid-training and supervised fine-tuning.

::code lecture_14.py:L13-L19 | where L13 stopped and the five stages this lecture covers

### HTML to text

Even after you have scraped something, you do not have text. Open a Common Crawl archive and you find HTML, sometimes PDFs (arXiv), or whole directories (code repositories). Most of the web is HTML, so most of the effort goes there. Converting HTML to text means two things:
- **remove boilerplate**: navigation, ads, headers, footers, menus;
- **extract the content**, the main part of the page.

Neither is crisp. What counts as content "is not always clear": navigation elements are noise for most purposes, but they also teach a model what web pages look like.

The process is **inherently lossy**, because a page is hierarchical (the DOM tree) or visual (the rendered page), and you need one linear sequence of tokens. Tables show the problem. A simple table can be rendered as markdown; a nested table cannot, and at some point you "give up" or approximate. The code leaves "What about images, tables, etc.?" as an open question; images get no answer at all.

The tools are rule-based: trafilatura, resiliparse, jusText, lynx. Rules are used because they are very fast and the job does not need "too much intelligence". The professor allowed that there "could be a case for model-based interventions" here, if they were fast and did something smarter. Rules also have a failure rate, which is why anyone who looks at web data finds imperfections in it.

::code lecture_14.py:L43-L49 | strip boilerplate, keep content, accept the loss; the tool you pick matters

::predict html-to-text-lossy

Does the choice of extractor matter? The DCLM paper ran the comparison: the same Common Crawl pages, extracted three ways, each used to train a model and evaluated on DCLM's CORE and EXTENDED benchmark suites.

::figure official/lectures/images/dclm-wet.png | the WET files Common Crawl ships score 20.7 on CORE; re-extracting the raw HTML with resiliparse or trafilatura gives about 24

So **accuracy matters**: the extractor decides what the model will ever see, and no later stage can bring back content the extractor dropped. Resiliparse and trafilatura are close (trafilatura best on CORE, 24.5; resiliparse best on EXTENDED, 13.4), and both beat Common Crawl's own pre-extracted WET text by a wide margin. [L13](#/read/lecture_13) explains the WARC/WET distinction.


### PDFs

Hugging Face's FinePDFs dataset shows the same problem for PDFs. Opened as raw bytes instead of in a reader, a PDF is a stream of drawing commands that has to be rendered before it is text.

::figure https://huggingfacefw-finepdfsblog.hf.space/_astro/pdf-description.Cb49jXc6_Z17eX4E.webp | what a PDF holds before rendering: layout instructions, not running text

What the pipeline does:
- **Source:** Common Crawl, which does hold some PDFs. A URL with no extension may turn out to be a PDF only after you fetch it.
- **Recrawl:** many PDFs in Common Crawl are truncated, because PDFs are big, so they have to be fetched again.
- **Convert:** some PDFs are scans, effectively images, so conversion mostly means OCR with a vision-language model (RolmOCR) or a tool like Docling. That is "much more expensive" than HTML extraction, and making it fast is part of the work.
- **Clean up and filter**, a lot.

Why bother? PDFs are a very small fraction of the web, but valuable: "if you bother to make a PDF", you probably have something to say, so the average PDF is of higher quality than the average HTML page. The cost is structure. HTML tags such as `h1` and `p` carry some meaning; PDFs are "by design, all about layout", so much of the semantic structure is lost.

::code lecture_14.py:L51-L57 | recrawl, OCR, clean up; the layout is lost
::note skip 5:03 | The FinePDFs blog post has many more details, which the professor skipped ("which I'll spare you of").
::kp html-to-text-lossy

## What is filtering, in one abstraction? {#filtering}
source: lecture_14.py:L60-L87 · video 6:42-11:27

Once you have text, "you're far from done". Almost every kind of filtering fits one building block. You have:
- **target data T**: usually a small amount of data that looks like what you want;
- **raw data R**: a huge amount of everything, the "fresh shipment" from the transformation step.

The goal is to find a subset **T'** of R that is similar to T.

::figure official/lectures/images/raw-target-schema.png | T sits outside R; the filter's job is to find T', the part of R that resembles T

The same block serves three applications, which differ only in what T is:
- **language identification**: English (or German) against everything else;
- **quality filtering**, the main reason to filter: encyclopedic text rather than spam;
- **toxicity filtering**: the internet has plenty of nasty content you may not want to train on.

::predict filter-target-raw-framework

A filtering algorithm has two requirements.
1. **Generalize from T.** You already have T, so getting T back is useless. T' should be different from T: new documents that share what makes T good.
2. **Be extremely fast.** It runs over all of R, which "could be 100 trillion tokens", and what survives is typically "a single-digit fraction" of it.

::code lecture_14.py:L61-L72 | one block, three applications, two requirements

The general scheme has two steps: estimate some model from R and T and derive a **scoring function** from it; then keep the examples of R according to their score. A survey of the whole area of data selection is [linked in the lecture](https://arxiv.org/abs/2402.16827).

### Two kinds of scorer

The model can be one of two kinds.

| | generative | discriminative |
|---|---|---|
| score | $p_T(x)$: how likely x is under a model of T | $p(T \mid x)$: how likely x came from T rather than R |
| trained on | T only | T as positives, a random subset of R as negatives (maybe balanced) |
| typical tool | KenLM, a 5-gram language model | fastText, a linear bag-of-words classifier |

Both are chosen for speed: the scorer "has to be cheap, so probably you're not training a big language model". KenLM is an n-gram model; fastText is used "because it's fast". The classifier route is the more common one today.

To use either, score every document and keep those with $\text{score}(x) \ge$ threshold, where the threshold depends on how high you want the bar. Sometimes the keep is stochastic rather than a hard cut; GPT-3's version comes below.

::code lecture_14.py:L80-L83 | p_T(x) from KenLM, p(T | x) from fastText; keep above a threshold, sometimes stochastically

::predict kenlm-vs-fasttext-scorers

The difference matters. KenLM never sees R: it measures whether a page is fluent in the way T is fluent. fastText learns what separates T from R, so a page that is fluent but ordinary is a typical negative. A well-written cooking blog can get a moderate perplexity under a math KenLM, yet a math-versus-web classifier rejects it.

### To use a model or not

Not everyone filters with a model. C4, Gopher, RefinedWeb, FineWeb and Dolma deliberately use only rules (see [L13](#/read/lecture_13)); GPT-3, LLaMA and DCLM use a trained classifier. The older datasets avoided classifiers on purpose, not to save cost: they "wanted to not bias things too much", since a classifier pulls everything toward whatever T is.

The code says model-based filtering is "becoming the norm". Aloud the professor went further: "basically everyone does some amount of model-based filtering". The reason is compute. With plenty of compute you could train on everything, but "most people are compute-poor", and without careful filtering "you're just wasting flops on low-quality content".

::code lecture_14.py:L85-L87 | rule-only datasets versus model-filtered ones
::kp filter-target-raw-framework

## How do you find one language, or one subject? {#langid-math}
source: lecture_14.py:L89-L102 · video 11:27-14:13

### Language identification

The goal: given a piece of text, decide whether it is in a particular language. You rarely build this yourself. Meta trained a set of **fastText language-ID models** that you use off the shelf:
- they support 176 languages;
- they were trained on multilingual sites: Wikipedia (which has many languages), Tatoeba (a translation site) and SETimes (Southeast European news);
- Dolma keeps a page when the model gives $p(\text{English}) \ge 0.5$.

::code lecture_14.py:L89-L95 | an off-the-shelf p(language | x) and a 0.5 cut

Language ID is "a fairly easy problem": from a few words you can tell Spanish from Japanese, so a simple classifier works. It is not entirely solved. Code-switching (text that mixes languages) and dialects are hard, and the threshold is "generally fairly heuristic". But it is "not really the bottleneck for training a good language model".

::predict offtheshelf-classifier-domain-bias

One consequence is worth spelling out (the professor did not draw it). A classifier's idea of "English" is the English of its training data. A page in heavy dialect, slang, or mixed with another language looks less like Wikipedia, Tatoeba and SETimes, scores lower, and can fall under Dolma's 0.5 even though a person would call it English. The filter defines the language operationally.

::note spoken 12:19 | Aloud, the hard cases were named: "there are subtleties because of code-switching ... and there's some dialects". Tying those failures to the training sites is our reading.

### Finding math: OpenMathText

Filtering can also target a subject. If you want a model that is good at math, go and find math. The professor's example is the paper he calls OpenMathText (its own title is OpenWebMath, 2023), which built a large math corpus from Common Crawl with a pipeline of several filters, not a single classifier:
1. **rules**: for example, does the page contain LaTeX commands?
2. **a generative scorer**: a KenLM trained on ProofPile, a known math dataset; keep the page if its perplexity is below 15000;
3. **a discriminative scorer**: a fastText classifier for "mathematical writing", kept above 0.17 if the page has math and above 0.8 if it does not.

::code lecture_14.py:L97-L102 | rules, then KenLM perplexity, then a fastText classifier with two thresholds
::worked kenlm-vs-fasttext-scorers

The two thresholds say: if the rules already found LaTeX, the classifier only needs to be mildly convinced (a lower bar); if there is no LaTeX, it has to be very sure (a higher bar). The professor said it that way, after first saying it backwards and correcting himself; he gave no reason for the specific numbers.

The result, as he read it from the paper: 14.7B tokens ("15 billion" aloud), used to train 1.4B-parameter models that do better at math than models trained on 20 times as much unfiltered data. Targeted curation beat volume.

::note slip 13:38 | Aloud the LaTeX case was first called "a higher bar", then corrected in the same sentence: "sorry, it's a lower bar". The code's numbers agree with the correction: 0.17 if math, 0.8 if not.
::kp kenlm-vs-fasttext-scorers

## What is "quality", and who decides it? {#quality}
source: lecture_14.py:L104-L127 · video 14:13-16:32

Treat quality filtering as a tool. "You can define quality however you want. There's no universal notion of quality." If you define quality as math, you get math and you get better at math. Each famous quality filter is the same kind of p(T | x) classifier; what differs is the positive set, and the positive set *is* the definition of quality.

| | positives (T) | negatives | classifier | keep rule |
|---|---|---|---|---|
| GPT-3 | samples of Wikipedia, WebText2, Books1, Books2 | Common Crawl | linear, word features | stochastic, by score |
| LLaMA / RedPajama | pages **referenced by** Wikipedia | Common Crawl | linear | classified positive |
| phi-1 | files GPT-4 rates educational, from a 100K-file sample | files it rates not educational | random forest on code-model embeddings | classified positive |

::predict quality-positives-define-quality

- **GPT-3** (Appendix A of its paper). WebText is pages linked from highly rated Reddit posts, so its positives are a mix of Reddit-endorsed pages, encyclopedia and book prose.
- **LLaMA** and its open reproduction RedPajama took the pages that Wikipedia *cites*, not Wikipedia articles: news, reference and institutional sites rather than encyclopedia text.
- **phi-1** (Microsoft) is the interesting one. Its philosophy: really high-quality, textbook-like data for a small model. The raw data R is the Python subset of The Stack (see [L13](#/read/lecture_13)). GPT-4 is prompted to judge each file's educational value for "a student whose goal is to learn basic coding concepts" on a 100K-file subset of R; the files it rates positive become T. A cheap random forest, on embeddings from a pretrained code model (CodeGen), learns to imitate GPT-4 and classifies the rest of R.

The phi-1 recipe is a pattern worth remembering: the target is "the output of an expensive classifier", distilled into a cheap one that can run over everything. fastText "could have probably" replaced the random forest. Besides the filtered code, phi-1's training data includes synthetic data generated by GPT-3.5 (later GPT-4).

::code lecture_14.py:L117-L124 | R, a prompt and GPT-4 define T; a random forest generalizes it to all of R

Did it work? On HumanEval (a Python code-generation benchmark), a 1.3B model trained on the raw Python subset of The Stack reached 12.19% after 96K steps; trained on the filtered subset it reached 17.68% after only 36K steps. Fewer steps, higher score: filtering harder does not always cost performance just because there are fewer tokens.

::code lecture_14.py:L125-L127 | 12.19% after 96K steps against 17.68% after 36K steps
::note slip | The code calls phi-1 "a small model (1.5B)" at L118 but reports results for a 1.3B model at L126-L127. phi-1 has 1.3B parameters.

### Keeping documents stochastically

GPT-3 does not cut at a threshold. It draws a random number from a Pareto distribution and keeps the document if the draw beats one minus the score:

::code lecture_14.py:L108-L110 | keep when np.random.pareto(9) > 1 − score

::predict pareto-stochastic-keep

What does that do to keep probabilities? The lecture defines the function but never runs it, so the numbers below are ours. NumPy's `pareto(a)` draws X with tail $P(X > x) = (1+x)^{-a}$ for $x \ge 0$. With $x = 1 - \text{score}$:

$$ P(\text{keep} \mid s) = P\big(X > 1 - s\big) = (2 - s)^{-9} $$

::worked pareto-stochastic-keep

| score | 1.0 | 0.9 | 0.5 | 0.0 |
|---|---|---|---|---|
| P(keep) | 1.0 | 0.42 | 0.026 | 0.002 |

Two things follow. First, the score is not the keep probability: a document scored 0.5 is kept 2.6% of the time, not half the time, and a document is kept more often than not only above a score of $2 - 2^{1/9} \approx 0.92$. Second, nothing is ever thrown away for sure: even a score-0 document survives one time in 512. The lecture gives no reason for the stochastic rule; a plausible one is diversity, so that some of the low-scoring mass, which the classifier may be wrong about, still reaches training.

::widget fixture:lecture_14--pareto-keep | the curve drops steeply below score 1 but never touches 0, unlike the dashed hard threshold; the grid of 1,000 documents at your score shows how many actually get through
::note slip 14:47 | Aloud, GPT-3 was said to keep documents "if the linear classifier is scored highly enough", which sounds like a hard threshold. The code says "keep documents stochastically based on score", and keep_document itself was not discussed.
::kp quality-positives-define-quality
::kp pareto-stochastic-keep

## Toxicity, and why there is no best threshold {#scale}
source: lecture_14.py:L129-L142 · video 16:32-22:55

### Toxicity filtering

"Toxicity works the same way." Dolma's toxicity filter is a classifier trained on the **Jigsaw Toxic Comments** dataset (2018), from a project whose goal was to "help people have better discussions online". The data is comments on Wikipedia talk pages, which for controversial articles "get quite heated", each annotated with six labels: toxic, severe_toxic, obscene, threat, insult, identity_hate. Define positives and negatives from those labels and train a classifier.

::code lecture_14.py:L129-L132 | Jigsaw: Wikipedia talk-page comments, six labels

Like language ID, this filter brings its training domain with it. Its notion of toxic is that of talk-page annotators, in that register. Applied to a medical forum, anatomical vocabulary can look "obscene" to it, and posts that are not toxic at all get removed. Whatever an off-the-shelf filter removes from a new domain is defined by the corpus it learned from, not by your pretraining goals. (The professor stated the training data; the bias reading is ours.)

::kp offtheshelf-classifier-domain-bias

With these examples you have the whole toolkit: identify the type of data you want, build a classifier for it, and filter Common Crawl with it.

### The best threshold depends on how long you train

Now a subtlety. A classifier gives you a score, but you cannot pick, say, 0.9 and call it the best threshold, because what data you want depends on the model you will train, in particular on how many tokens you train on:
- train **shorter**, and you want less data of higher quality;
- train **longer**, and you can tolerate, and need, lower-quality data.

With a magic wand you would want more high-quality data for the long run too, "but that's not an option ... The data pool is what it is."

::code lecture_14.py:L134-L137 | no single optimal threshold: the right one moves with the training length
::figure official/lectures/images/data-filtering-scale.png | follow the blue DCLM curve and the purple resiliparse curve: DCLM is lower at first, bottoms out around 430M tokens, then rises once it repeats its 97.6M tokens many times; unfiltered resiliparse keeps falling, to about 3.5 by 25B tokens

The figure is a preliminary experiment by Michael Ryan: a 157M-parameter model, trained on a deliberately small pool (100 WARC files, a tiny fraction of Common Crawl) filtered in several ways, with evaluation loss plotted against tokens trained. The dashed vertical lines mark where each dataset completes an epoch.
- **DCLM** (blue), a strict quality filter, keeps only 97.6M tokens of this pool. Early on its loss is far below resiliparse's. Then it has to repeat the data; the second pass still teaches something, but soon it overfits and the loss turns up, to about 4.9 at 4B tokens.
- **Resiliparse** (purple) is "basically no filtering": text extraction only, 4.42B tokens per epoch. It starts much worse (about 5.3 against 4.6), but because it barely repeats, it keeps improving and ends around 3.5.

High-quality data wins "in this regime, where you're not epoching". Once you train on many tokens it no longer does. And stopping DCLM early, at its best point (about 3.8), still leaves it "worse than if you had trained for longer using low-quality data".

::predict filter-threshold-scale-dependent
::widget fixture:lecture_13--filter-budget | set the DCLM 1.6% filter and move the budget: at Llama 3's 15T it needs just under four epochs, at Qwen3's 36T over nine; the longer the run, the more of the pool a filter has to keep
::video 19:02-20:14 | high-quality data wins until it starts repeating; then unfiltered data overtakes it

In [L13](#/read/lecture_13) the same trade-off appears from the other side, as how aggressively each big dataset filtered for its token budget. It will come back in this lecture once more, in data mixing, where a small source given a large weight is repeated many times.

::note spoken 20:34 | Asked whether each point was a single training run and needed confidence intervals: yes, a single run each; repeats would be "good practice", but each run is expensive, and in their experience such pretraining ablations "tend to be stable".
::note spoken 21:30 | Asked whether a bigger high-quality pool would also hit diminishing returns: "every data set is going to have diminishing returns eventually", since it is finite, but its curve would sit lower and keep falling longer.

### Filtering, summed up

Filtering matters most when you are compute-constrained, which is almost everyone: "if you have infinite compute, you don't need to filter". The recipe is to define what good data looks like and train a classifier that extrapolates it to the rest of the pool. There are two ways to get the "good" examples:
- find an existing dataset that "I really like and I just want more of it" (Wikipedia, ProofPile, WebText);
- prompt a language model to pre-filter a pool, then train a smaller classifier on its choices and run that over everything (phi-1).

::code lecture_14.py:L140-L142 | define the target, extrapolate to the raw data
::kp filter-threshold-scale-dependent

## Why deduplicate, and how does exact deduplication work? {#exact-dedup}
source: lecture_14.py:L145-L214 · video 22:55-31:26

After filtering you have only what you deem high quality, but it still contains duplicates. They come in two kinds.
- **Exact duplicates.** Mirror sites exist to be copies, and a crawler does not always know two mirrors are the same, so it fetches both (the lecture links Project Gutenberg's list of mirrors). Forking a repository copies it too; even if you then change a few files, "99% of that repo might be the same".
- **Near duplicates**: the same text differing by a few tokens, from copying or from a common source.

Where near duplicates come from:
- **terms of service and licenses.** The MIT license appears on countless pages. The license text itself is often an exact copy (unless someone made a typo), but the page around it differs, so at the level of whole documents those pages are near duplicates. Shared headers and footers are the same story.
- **formulaic writing**, copy-pasted or generated from a template;
- **minor formatting differences** introduced by copying.

::figure https://d3i71xaburhd42.cloudfront.net/4566c0d22ebf3c31180066ab23b6c445aeec78d5/5-Table1-1.png | near-duplicate pairs from Lee et al.: one LM1B news pair differs only in punctuation; a templated ad differs only in the country it names

The professor's two examples from the table: an article from LM1B (the One Billion Word benchmark) that exists in two versions, one with a comma and one without ("I don't really know why, but that happens"), and a low-quality ad template in which someone replaced "Canada" with "USA". Training on many variations of the same template with different entities is "wasting your GPUs".

The extreme case shows why you should look at your data. An audit of C4 found one product description (for a gas mask on Amazon, beginning "by combining fantastic ideas, interesting arrangements ...") repeated **61,036 times**. "The web is weird."

::code lecture_14.py:L146-L157 | two kinds of duplicate; licenses, templates and copy-paste; one description 61,036 times in C4

### Why it helps

The paper [Deduplicating Training Data Makes Language Models Better](https://arxiv.org/pdf/2107.06499) (Lee et al., 2021) gives two reasons, which the professor ranked:
1. **Efficiency.** Fewer tokens with almost no loss of information, since you only remove copies. "Mostly, I think, it's just to make sure that you're not wasting flops."
2. **Less memorization.** Text repeated many times gets memorized, which matters for copyrighted content and for privacy.

::note aside 26:58 | **Decontamination** is the same operation aimed at evaluation: make sure your test set is not in your training set. The professor called it "arguably even more important" and did not develop it further.

### The design space

Every deduplication scheme makes three choices:
1. **What is an item?** A sentence, a paragraph, a document.
2. **How do two items match?** Exact match; the existence of a common sub-item; or the fraction of sub-items they share (near deduplication).
3. **What action?** Remove all copies, or all but one.

The key challenge is algorithmic. Filtering judges one item at a time ("Is this item good or not?"), which is linear time and parallelizes trivially. Deduplication is "fundamentally about comparing items to other items", and at web scale "you can't do the n-squared thing" of comparing everything with everything. You need linear-time algorithms, and the literature gets them from hash functions.

::code lecture_14.py:L163-L170 | item, match, action; then the n-squared problem

### Hash functions

A hash function h maps an item (a string) to a hash value, an integer or short string much smaller than the item. A **collision** is $h(x) = h(y)$ for $x \ne y$. Hash functions trade speed against collision resistance:
- **cryptographic** hashes such as SHA-256 are collision-resistant and slow (used in Bitcoin);
- **non-cryptographic** hashes such as DJB2, MurmurHash and CityHash are not collision-resistant but are fast. They are built for hash tables, where "hash collisions aren't the end of the world".

Deduplication needs the second kind: a rare collision merges two different items, a small loss, while speed matters on every item. The lecture uses MurmurHash (Python's `mmh3`). `mmh3.hash("hello")` is 613153351.

::code lecture_14.py:L183-L188 | cryptographic and slow, or fast and good enough; MurmurHash is the choice

### Exact deduplication

The simplest scheme: the item is a string, matching is exact, and the action is to keep one copy. Sort the items by hash, group equal hashes, keep the first of each group.

::code lecture_14.py:L198-L204 | hash, group by hash, keep one per group

::predict exact-dedup-hash-groupby

On the list `["Hello!", "hello", "hello there", "hello", "hi", "bye"]` only the two identical `"hello"` strings share a hash. Five strings survive, in hash order: `['hi', 'bye', 'hello', 'hello there', 'Hello!']`. A capital letter or an exclamation mark is enough to make a different item. (The lecture computes this with `mmh3`; the values here come from a reimplementation of the same MurmurHash3 function.)

::widget fixture:lecture_14--exact-dedup | press "L198 list (one source)": only byte-identical lines share a hash and a group, so "Hello!" and "hello there" stay next to the one surviving "hello"

The trade-off:
- **pro**: simple, clear semantics, high precision (whatever it removes really was a copy);
- **con**: it misses near duplicates entirely.
- The code is written MapReduce-style (map each item to its hash, reduce each group to one item), so it parallelizes and scales easily.

**C4** (from the T5 paper; see [L13](#/read/lecture_13)) applied exact dedup to Common Crawl with one twist in the design space: the item is a **3-sentence span**, matched exactly, all copies but one removed. That catches repeated boilerplate inside otherwise different pages, but it has a strange effect. Removing a span from the middle of a document rips three sentences out of it, and "it breaks the coherence".

::code lecture_14.py:L210-L214 | C4: 3-sentence spans, exact match, keep one; documents may lose coherence
::kp exact-dedup-hash-groupby

## How do you measure "nearly the same", and hash for it? Jaccard and MinHash {#minhash}
source: lecture_14.py:L217-L265 · video 31:26-37:29

To find near duplicates you first need a similarity measure. View each document as a **set** of items. The lecture leaves open what the items of a real document are; in practice they are usually short overlapping word or character n-grams ("shingles"), as in the [MMDS chapter](http://infolab.stanford.edu/~ullman/mmds/ch3n.pdf) the lecture links, and as [L13's MinHash over n-grams](#/read/lecture_13) describes.

### Jaccard similarity

$$ \text{Jaccard}(A, B) = \frac{|A \cap B|}{|A \cup B|} $$

For $A = \{1,2,3,4\}$ and $B = \{1,2,3,5\}$: the intersection is $\{1,2,3\}$, 3 items; the union is $\{1,2,3,4,5\}$, 5 items; Jaccard is $3/5 = 0.6$. It ranges from 0 (disjoint) to 1 (identical).

::code lecture_14.py:L222-L230 | intersection 3, union 5, Jaccard 0.6

::predict jaccard-similarity

Note the denominator is the union, not either set. If B sits wholly inside an A twice its size, Jaccard is only 0.5, even though every item of B is in A. And Jaccard 0.6 does not mean "60% of the text is identical"; it is a statement about shared items.

Two documents are **near duplicates** if their Jaccard similarity is at least some threshold; the professor's example threshold was 0.99. The algorithmic challenge is to find all such pairs in linear time.

::kp jaccard-similarity

### MinHash

The first step toward that is **MinHash**: a random hash function h with the property

$$ \Pr[h(A) = h(B)] = \text{Jaccard}(A, B). $$

This turns the usual goal of hashing upside down. Normally you want distinct items to hash to different values and you want no collisions. Here you *want* collisions, controlled so that similar sets collide more often than dissimilar ones, with probability exactly their similarity.

The construction is one line: hash every element of the set with a seeded hash, and take the minimum.

::code lecture_14.py:L242-L243 | minhash(S, seed) = min over x in S of mmh3.hash(x, seed)

Why the minimum works: write the two sets as a characteristic matrix, one row per item of the union, a 1 where the set contains it.

| item | A | B |
|---|---|---|
| 1 | 1 | 1 |
| 2 | 1 | 1 |
| 3 | 1 | 1 |
| 4 | 1 | 0 |
| 5 | 0 | 1 |

A random hash function induces a random **permutation** of these rows, for example 4, 3, 1, 5, 2. The minhash of a set is the hash of whichever of its items comes first in that order. Every row of the union is equally likely to be first overall.
- If row 1, 2 or 3 comes first, that item is in both sets, so it is first in A *and* first in B: the minhashes are equal.
- If row 4 or 5 comes first, it is in only one set; the other set's first item is a different one, and the minhashes differ.

So the minhashes agree exactly when the first row of the union is a shared row, which happens with probability $|A \cap B| / |A \cup B|$: 3 rows out of 5 here. The minimum is arbitrary: "You can take the max, too. It doesn't really matter." Any fixed rule for picking one element under the random order works.

::code lecture_14.py:L253-L257 | the hash induces a permutation; rows 1-3 make the minima agree, rows 4-5 make them differ
::predict minhash-collision-probability

The lecture checks the claim numerically: 100 seeds give 100 independent hash functions; count the seeds on which the two minhashes agree. The assert demands the estimate be within 0.01 of 0.6, and the run gives exactly 0.6: 60 matching seeds out of 100 (the professor read "you get 0.6"; our MurmurHash3 reimplementation agrees). With a strict "< 0.01" on 100 seeds, 60 is in fact the only count the assert accepts.

::code lecture_14.py:L259-L263 | 100 seeds; the fraction of equal minhashes estimates Jaccard
::widget fixture:lecture_14--minhash-permutation | step the seed: the minhashes are equal exactly when the outlined first item of A ∪ B is green (shared); over many seeds the running fraction settles on the dashed Jaccard line

The payoff: you never compare documents pairwise. Compute each set's minhash once and look for collisions, which is a hash-table lookup: linear time.

But one collision is weak evidence. Two documents with Jaccard 0.3 collide on a single minhash 30% of the time, and two with Jaccard 0.99 miss 1% of the time. "A collision doesn't tell us Jaccard(A, B) > threshold", which is what we want.

::code lecture_14.py:L265-L265 | a collision is only evidence in expectation
::kp minhash-collision-probability

## How do you turn "collide with probability s" into a threshold? Locality-sensitive hashing {#lsh}
source: lecture_14.py:L268-L321 · video 37:29-49:21

With one MinHash, $P(\text{A and B collide}) = \text{Jaccard}(A, B)$. More similar pairs collide more often on average, but the outcome is "very stochastic"; "the variance is quite large". What we want is a step: collide if Jaccard is above the threshold, don't collide if below. The probabilities have to be **sharpened**.

**Locality-sensitive hashing** (LSH), "a very classic idea in theoretical computer science", does it with more hash functions, independent of one another, arranged in bands:
- take $n$ MinHash functions and split them into $b$ **bands** of $r$ functions each, $n = b \cdot r$;
- A and B **collide if, for some band, all of its hash functions agree**.

With $n = 12$, $b = 3$, $r = 4$:

```
h1 h2 h3 h4  |  h5 h6 h7 h8  |  h9 h10 h11 h12
```

If $h_5(A) = h_5(B)$ and $h_6$, $h_7$ and $h_8$ also agree, the second band fires and the pair collides, whatever the other bands do. Inside a band it is an AND; across bands it is an OR. "This and-or structure is doing the lifting."

::code lecture_14.py:L278-L288 | n = b · r hash functions; collide if some band has all its hashes equal

### The collision probability

Each MinHash agrees with probability $s = \text{Jaccard}(A, B)$, independently. So:
- a fixed band matches (all r agree) with probability $s^r$, which is small: "exponential in r";
- a fixed band fails with probability $1 - s^r$, and all b bands fail with $(1 - s^r)^b$;
- some band matches with probability

$$ P(\text{collide}) = 1 - (1 - s^r)^b. $$

::code lecture_14.py:L292-L295 | prob_match = sim ** r; prob_collision = 1 − (1 − prob_match) ** b

The lecture's example, $s = 0.8$, $b = 5$, $r = 10$: one band matches with $0.8^{10} = 0.107$; the pair collides with $1 - 0.893^5 = 0.433$ ("0.4" aloud). That is higher than one band's chance because you get b tries.

Plotted against s, this function is an **S-curve**: 0 at s = 0, 1 at s = 1, flat at both ends and steep in between. That is the shape we wanted: low collision probability below some threshold, high above it, like a phase transition.

::predict lsh-band-sharpening

### Moving and sharpening the curve

The code evaluates the curve at similarities 0.7 to 0.98 for three settings:

| s | 0.7 | 0.75 | 0.8 | 0.85 | 0.9 | 0.95 | 0.98 |
|---|---|---|---|---|---|---|---|
| b = 10, r = 10 | 0.249 | 0.440 | 0.679 | 0.888 | 0.986 | 1.000 | 1.000 |
| b = 10, r = 20 | 0.008 | 0.031 | 0.109 | 0.327 | 0.726 | 0.988 | 1.000 |
| b = 20, r = 20 | 0.016 | 0.062 | 0.207 | 0.546 | 0.925 | 1.000 | 1.000 |

::code lecture_14.py:L302-L309 | the same similarities under three (b, r) settings

- **b = r = 10** spans 0.25 to 1. Not bad as a filter around 0.9, but pairs below it still collide fairly often: false positives, which you can check afterwards.
- **Raising r** to 20 sharpens the curve and **moves it right**: every band now needs twice as many agreements, so everything is harder to match. At s = 0.7 the collision probability falls from 0.25 to 0.008.
- **Raising b** to 20 **moves it left**: more bands, more chances, easier to match. At s = 0.9 it rises from 0.72 to 0.92, while the low end rises only a little.

You can make the transition as sharp as you like by raising both b and r, but each costs more hash functions, so it "can be more expensive".

::widget fixture:lecture_14--lsh-bands | click through the L303, L306 and L309 presets: raising r shrinks every band's AND bar and pushes the curve right; raising b adds OR chances and pulls it back left; the concrete pair below shows real hashes agreeing row by row
::video 45:13-46:38 | raising r: 0.25 becomes 0.008; raising b: 0.72 becomes 0.92 at s = 0.9
::kp lsh-band-sharpening

### Where is the threshold?

Lee et al. used $n = 9000$ hash functions as $b = 20$ bands of $r = 450$. Where does their curve switch from 0 to 1? At the similarity where a single band matches with probability $1/b$, so that about one of the b bands is expected to match:

$$ s^{*\,r} = \frac{1}{b} \quad\Longrightarrow\quad s^* = \left(\frac{1}{b}\right)^{1/r}. $$

For Lee et al., $s^* = (1/20)^{1/450} = 0.9934$. To filter at some Jaccard, say 0.9, you choose b and r so that $(1/b)^{1/r}$ equals 0.9, and then scale both up together to sharpen the transition around it.

At the threshold the collision probability is a constant, whatever b and r:

$$ P(\text{collide at } s^*) = 1 - \left(1 - \frac{1}{b}\right)^{b} \approx 1 - \frac{1}{e} \approx 0.63. $$

For $b = 20$ it is $1 - 0.95^{20} = 0.6415$ ("0.64" aloud); the $1 - 1/e = 0.6321$ limit is approached as b grows. So the centre of the transition sits near 0.63, not 0.5; as b and r grow, everything below $s^*$ goes to 0 and everything above goes to 1.

::code lecture_14.py:L312-L321 | Lee et al.'s b = 20, r = 450; threshold (1/b)^(1/r); collision probability ≈ 1 − 1/e there

::predict lsh-threshold-one-over-b

For a sense of how sharp 9000 hashes make it: under Lee et al.'s setting, pairs at Jaccard 0.98 collide with probability 0.002, at 0.99 with 0.20, at 0.995 with 0.89. A setting with thousands of hash functions is not catching loosely similar documents; it is catching pairs whose shared items make up over 99% of their union. Regrouping the same 9000 hashes as 90 bands of 100 would lower the threshold to $(1/90)^{1/100} \approx 0.956$.

::widget fixture:lecture_14--lsh-bands | press "Lee et al.: b=20, r=450" and put s on the dashed s* line: each band's AND bar is then exactly 1/b long and the OR bar reads about 0.64
::note why | Why Lee et al. chose a threshold as high as 0.993 is not said in the lecture.

The method as a whole is **MinHash LSH**. LSH works with any family of hash functions; for deduplication, MinHash is the one to use, because its collision probability is Jaccard.

::kp lsh-threshold-one-over-b

### Deduplicate across sources, not just within them

::predict dedup-across-sources

One last remark, said aloud only. Datasets often arrive already deduplicated, but deduplication only compares items within the collection it runs on. If a document is in two datasets (a page in both a Common Crawl extract and a curated web set filtered from Common Crawl), deduplicating each dataset separately never compares the two copies. "You actually have to do deduplication across your entire data set because often data sets can be redundant. So sometimes that's not done, but it should be."

::widget fixture:lecture_14--exact-dedup | with "three sources (default)" loaded, switch the scope from "dedup within each source" to "dedup over the union": lines that two sources share survive the first and are removed by the second
::video 48:55-49:21 | deduplication has to run across your entire data set
::kp dedup-across-sources

## How much of each source? Mixtures and the epoch trap {#mixing}
source: lecture_14.py:L331-L371 · video 49:21-1:00:00

So far the pipeline has turned raw HTML or PDFs into text, filtered it for quality and deduplicated it, usually one source at a time. But language models are trained on many sources at once. Marin, an open model-development project, keeps a live view of the sources its next model will train on:

::figure official/lectures/images/marin-token-viewer.png | tokens per dataset, in billions, colored by category: web (orange) dominates, with Nemotron-CC medium-quality web at about 2.1T; multilingual, code, math and specialized sets are each far smaller

Nemotron-CC web data at several quality levels, FinePDFs (from the first section), institutional books, code: a few hundred billion to two trillion tokens each. An older example is The Pile ([L13](#/read/lecture_13)), which assigned a weight to each of its components.

::figure https://stanford-cs324.github.io/winter2022/lectures/images/the-pile.png | The Pile's components, each given its own share of the mixture

The question: **what distribution over sources should we train on?** A **data mixture** is just that, a distribution $p(s)$ over sources s, for example $p = \{\text{Wikipedia}: 0.3,\ \text{CC}: 0.5,\ \text{GitHub}: 0.2\}$.

::code lecture_14.py:L339-L343 | a mixture is a distribution over sources

How is a mixture realized in training? (A student asked.) Fill each batch by sampling, for each sequence, which source it comes from; each sequence comes from one source, so every batch is mixed. Nothing is sampled per token, and you do not alternate whole steps between sources, which would add variance.

### Baselines

- **Vibes**: set $p(s)$ by hand from intuition. "More often than you might think, what people do"; even recent papers run some method and then tweak the result.
- **Uniform**: $p(s) \propto 1$, each source equally.
- **Proportional**: $p(s) \propto \text{num\_tokens}(s)$, each source by its size. Rational, but a huge low-quality source then "is going to eat up a lot of your tokens".

::code lecture_14.py:L345-L348 | vibes, uniform, proportional

The intuition is to upweight higher-quality sources. Two things limit it.
1. **Diversity.** Sources are often incomparable (literature, code, papers); you cannot say a paper is higher quality than a piece of code, and you do not want all your mass on papers.
2. **Finiteness.** Each source has a fixed number of tokens. Put too much weight on a small source and you run out of it, and then you **epoch**: train on literally the same tokens again.

### The epoch trap, in numbers

Training for N tokens draws $p(s) \cdot N$ tokens from source s. ("Train for 1 trillion tokens" means steps times batch size, not unique tokens.) Divided by the source's size, that is how many times each of its tokens is seen:

$$ \text{epochs}(s) = \frac{p(s) \cdot N}{\text{num\_tokens}(s)} $$

Take two sources: a low-quality one with 10T tokens and a high-quality one with 10B (high-quality sources are generally smaller). Mix them 50/50 and train for 1T tokens:
- low: $0.5 \times 10^{12} / 10^{13} = 0.05$ epochs, so only 5% of that source is ever touched, once;
- high: $0.5 \times 10^{12} / 10^{10} = 50$ epochs. The run needs 500B high-quality tokens, there are only 10B, so each is repeated 50 times.

::code lecture_14.py:L357-L365 | a 50/50 mix of 10T and 10B over 1T tokens: 0.05 epochs and 50 epochs
::predict mixing-baselines-epoching

A 50/50 mixture does not treat the two sources equally at all. "Some big model runs have messed this up": if you define a distribution by looking only at quality, you miss how many data points each source has. Why are 50 epochs bad? At best they waste compute; at worst the model overfits. Nobody chooses 50 epochs; you get them "without realizing it". The lesson: look at how many epochs you are actually doing.

::widget fixture:lecture_14--mixture-epochs | with the L362 preset, the same weight 0.5 is a twentieth of one pass over the 10T source and 50 passes over the 10B one; a bar past the dashed one-epoch line changes color
::video 54:23-55:37 | the same mix is 0.05 epochs of the big source and 50 of the small one

### UniMax: cap the epochs

The problem was noticed long ago in multilingual models, where some languages have very little data. Earlier work interpolated between uniform and proportional mixing with a power, $p(s) \propto \text{num\_tokens}(s)^\alpha$, $\alpha \in [0, 1]$, which flattens the distribution. **UniMax** is more explicit: sample sources uniformly, but with a hard **cap C on the number of epochs** of any source. Once a source has been used C times, "too bad. You don't get any more tokens." The rest of the budget moves to the other sources: a safety net.

$$ p(s) \cdot N \le C \cdot \text{num\_tokens}(s) \quad \text{for every source } s $$

::code lecture_14.py:L367-L371 | between uniform and proportional with a power α; UniMax caps the epochs instead
::note slip | The code writes the cap as "p(s) * num_training_tokens ≤ C", which reads as a cap on tokens, while the line before glosses C as a cap "on number of epochs". For C to count epochs, the right side must be C times the source's size, as written above. Aloud the professor read the token form, and illustrated C with "I'm only going to take 20 epochs", an example rather than UniMax's setting.
::note skip 59:51 | UniMax's "simple procedure" for finding the mixture that satisfies the cap was not described.
::kp mixing-baselines-epoching

## Can you fit the best mixture at small scale? Regression-based mixing {#regression-mixing}
source: lecture_14.py:L373-L406 · video 1:00:00-1:12:56

With 50 sources there are 50 numbers to fill in. Proportional mixing is not enough, and sampling in proportion to some estimated quality score is still a heuristic. The most principled approach, and the easiest to understand, is **regression-based mixing** (RegMix, Olmix and similar papers):
1. go to a small scale, say 300M-parameter models;
2. sample many mixtures and train "a swarm of small models", one per mixture;
3. measure each model's loss on a target: downstream evaluations, perplexity, "whatever you want";
4. fit a regression from mixture weights to that loss, a cheap model of "if I train on this mixture, what loss do I get?";
5. optimize the regression to find the best mixture, and train the large model on it.

It resembles [scaling laws](#/read/lecture_09): cheap experiments, a fitted function, then a jump to scale.

::figure official/lectures/images/regmix.png | RegMix with three sources (Hacker News, GitHub, PhilPapers): proxy runs give target losses, a linear or tree regression is fit, and its predicted minimum, 22.8% / 67.0% / 10.2% with a predicted 5.34, becomes the large run's mixture

The design decisions:
- **Which mixtures to try.** You need a distribution over distributions; a Dirichlet is common.
- **Which regression.** Linear models and boosted trees have been used; a log-linear model "tends to work pretty well".
- **Which target.** Usually downstream evaluations, and here you must be careful not to overfit. Pretraining is supposed to produce a general-purpose model. If your target is a set of code evaluations, "guess what? You're going to upweight all the code data", and the model may then be bad at poetry. Uniform and proportional mixing cannot overfit this way, "because you're not looking".
- **How small.** A cost-accuracy trade-off. Tiny proxies may not be representative; but proxies as large as the final model would be "hyperparameter tuning at the largest scale", which defeats the purpose.

::code lecture_14.py:L375-L378 | swarm distribution, regression, target, proxy scale

The Olmix paper tabulates how published methods fill in these choices:

::figure official/lectures/images/data-mixing-methods.png | proxy models from 1M to 410M parameters, swarms of 4 to 512 runs, Dirichlet or grid sampling, LightGBM, log-linear or power-law regressions; only Olmix's own method has a data-repetition constraint

::note slip 1:04:08 | Aloud the proxy models in this table were "generally ... tens of millions of parameters". The table's sizes run from 1M (RegMix) to 410M (DML's largest), with Olmix at 30M and others at 280M and 350M.
::note skip 1:04:42 | How to solve the resulting optimization problem ("different ways") was not covered.

### Two leaps of faith

The method rests on two hopes, marked with praying hands in the code.
1. **The regression is accurate at its minimizer.** For a random mixture it probably is, "because of classic generalization": it predicts in distribution. But optimization pushes toward the extremes of the mixture space, "where you might not have as much coverage".
2. **The optimal mixture transfers from small to large scale.** At the scales the open community works at, this "tends to be true or not plainly false". But "clearly, there are scale-dependent effects": the filtering section showed that a longer run tolerates lower-quality data, so the optimum cannot be the same.

::code lecture_14.py:L380-L381 | hope 1: accurate at the minimizer; hope 2: the optimum transfers

### One scale-dependent effect you can fix

::predict regression-mixing-simulated-epoching

Return to 10T tokens of low-quality data and 10B of high-quality. A small model trained on few tokens never repeats anything, so the regression may conclude that the high-quality data is wonderful and put, say, 0.9 on it ("a made-up example"): "Wikipedia is so great. Let's just train on Wikipedia." Train the large 1T-token model on that mixture and the high-quality source is repeated $0.9 \times 10^{12} / 10^{10} = 90$ times. You overfit.

::code lecture_14.py:L383-L390 | the small run puts 0.9 on the 10B source; the large run would epoch it heavily

Two fixes:
- **Cap the epochs**, as Olmix does (the only yes in its table's repetition row).
- **Simulated epoching**: make the small scale look like the large scale. That is a general theme of the course; [muP](#/read/lecture_11) parameterizes a model so that hyperparameters transfer, and the same principle applies to data. Not repeating at small scale but repeating at large scale are "qualitatively different operations on your data set".

The instantiation: **downsample every source proportionally** by the ratio of the small run to the large run.

::code lecture_14.py:L395-L398 | ratio = 10B / 1T = 0.01; the 10T source becomes 100B, the 10B source becomes 100M

| | low-quality | high-quality |
|---|---|---|
| full pool | 10T | 10B |
| downsampled ×0.01 | 100B | 100M |
| epochs of the large run (1T) at weight 0.9 on high | 0.01 | 90 |
| epochs of the small run (10B) on the downsampled pool, same weights | 0.01 | 90 |
| epochs of the small run (10B) on the full pool, same weights | 0.0001 | 0.9 |

Every epoch count in the small run on the downsampled pool equals the large run's, because both numerator and denominator were scaled by the same ratio. Now the "train on Wikipedia" mixture repeats a minuscule source 90 times *in the small run too*, gets a bad loss there, and the optimizer is pushed toward a more balanced mixture; the code types $p = \{\text{low}: 0.7,\ \text{high}: 0.3\}$ as an illustration. You are "simulating the data scarcity" of the large run at small scale.

::widget fixture:lecture_14--mixture-epochs | load the L389 preset: on the full pool the 10B pilot barely repeats the small source (0.9 epochs), but on the downsampled pool it repeats it 90 times, exactly as the 1T production run will
::video 1:07:44-1:09:36 | make small scale look like large scale: downsample every source by 1/100

The risk, raised in Q&A: a tiny source can be downsampled to almost nothing, and rounding can turn "train once on this data" into "train zero times by accident". The workaround is to always train at least once on every source.

::note aside 1:12:07 | Asked whether mixing applies only to given sources: no. The professor had "forgot[ten] to mention" that one source can be split, e.g. Common Crawl grouped by topic (with a tool like WebOrganizer) and by quality, giving a two-dimensional grid of cells, each a mixture component, as in the Nemotron work. Hand-picked extra sources are added on top.

### Mixing, summed up

The problem is how to weight sources (Wikipedia, Common Crawl, code, math scraped from somewhere). Regression-based mixing is a good framework: fit a function from mixture weights to loss at small scale, optimize it, generalize to large scale. Watch epoching and overfitting, and either cap the epochs or simulate them. The wider lesson: "if you're trying to optimize anything", you are "in danger of optimizing the wrong thing".

::code lecture_14.py:L403-L406 | weight the sources; fit at small scale; cap or simulate the epochs
::kp regression-mixing-simulated-epoching

## Where does post-training data come from? Teachers and synthetic data {#post-training}
source: lecture_14.py:L409-L423 · video 1:12:56-1:17:46

Everything so far was pretraining (or mid-training) data: fairly task-agnostic, meant to build basic skills, even when RegMix optimizes a loss. Post-training data is very task-dependent (see [L13's stages](#/read/lecture_13)). The professor went through it quickly, without a comprehensive review, picking recent datasets for coding, "since that's of great interest these days".

The general recipe:
1. define a set of **environments** (for code, GitHub repositories);
2. define a set of **tasks or prompts**;
3. collect **responses from a strong model**, the teacher.

So in the open community "almost all the post-training data ... is synthetically generated". You could use humans instead of a teacher model, but they are slow and expensive. A few years ago frontier labs paid many people for responses; now even they use "hybrid human-AI" data. Either way, some teacher provides the responses.

::code lecture_14.py:L410-L413 | environments, tasks, a teacher's responses

### OpenThoughts

OpenThoughts was motivated by OpenAI's o1 and the interest in reasoning, mostly for math and science: how do you build a really good open post-training dataset for it? The result: **1.2M examples**, with QwQ-32B as the teacher. The questions come from 27 human and synthetic sources (StackExchange, NuminaMath, chemistry and others). The coding sources alone range from coding puzzles to code review:

::figure official/lectures/images/openthoughts-sources.png | some of the coding question sources and their sizes, from StackExchange CodeGolf (85.9K questions) to fully synthetic sets such as KodCode (384K) and glaive-code-assistant (946K)

::predict synthetic-post-training-recipe

The project ran many ablations on how to turn questions into a dataset. Its findings, as the professor read them:
- **sample multiple responses per prompt** (16): helpful;
- **better models are not necessarily better teachers**: QwQ-32B ("now a very old and small model") was a better teacher than DeepSeek-R1, at the time one of the strongest open models;
- **answer filtering** (keeping only responses with verified answers) was not helpful;
- **a few small high-quality sources** (such as OpenMath-2-Math) beat many large diverse ones.

The paper's reasons for the teacher and filtering results were not given in the lecture.

::code lecture_14.py:L416-L422 | 1.2M examples from QwQ-32B; 16 samples help; the best model is not the best teacher; answer filtering did not help
::figure official/lectures/images/openthoughts-pipeline.png | read left to right: millions of source questions are filtered, deduplicated and randomly sampled down to 75K, then answered 16 times each for the final 1.2M

The 1.2M counts responses: "divided by 16 gives you the number of actual questions". The pipeline figure confirms it, 75K questions (53K math, 16K code, 6K science) × 16 = 1.2M.

::note slip 1:17:13 | In the pipeline walk the captions read "and then you duplicate"; the figure's stage is "Deduplicate Questions".
::kp synthetic-post-training-recipe

## How do you make data for coding agents? {#swe}
source: lecture_14.py:L425-L460 · video 1:17:46-1:23:44

The recent interest is in **agentic coding**: not a model that writes a function, but one that does software development in a real repository. The datasets below differ along two axes: where the tasks come from (synthetic, or real pull requests), and whether the teacher's trajectories get **execution feedback** (running the code and tests) or not.

### SWE-smith: synthetic tasks in real repositories

Given a repository, an agent first makes it usable (installs dependencies, runs the tests); then a language model generates tasks, generally by modifying the code to introduce bugs; the tasks whose tests confirm the bug become task instances. From 128 GitHub repositories this yields 50K tasks, "quite large" at the time (last year).

::figure official/lectures/images/swe-smith.png | real repository and its tests, an environment built by SWE-agent plus a developer, bugs introduced by several strategies, and verified synthetic task instances

::code lecture_14.py:L427-L428 | an LM injects bugs; 128 repositories give 50K tasks

### SWE-Zero: drop the execution

The observation behind SWE-Zero (from NVIDIA): unlike math or coding contests, software-engineering tasks have heavy dependencies. "Most GitHub repos ... don't even run"; dependencies are out of date, especially after rolling back to the commit a PR started from. Setting up thousands of Docker images, one per repository, is "an infrastructural nightmare".

But strong models can solve many tasks **without** execution feedback:

::figure official/lectures/images/swezero-noexec.png | SWE-bench Verified with and without execution: MiniMax-M2.5 scores 80.2 with it and 69.5 without; every model loses some points, none collapses

"If you were allowing execution, you get 80. If you don't allow execution, you get almost 70." So these models carry an internal "world model" of code semantics, enough to fix many bugs by reading alone.

::predict swe-synthetic-tasks-execution

SWE-Zero builds on that:
- **300K agent trajectories** that need no repository-specific execution, over **150K real GitHub PRs**, so the tasks are realistic;
- the **OpenHands** scaffold, with an execution-free prompt: the agent may explore, analyze and edit, but may not run Python, tests or package managers, only basic shell commands like sed and grep;
- **future git commits removed**, so the agent cannot "git hack": read the repository's later history, where the actual fix lives;
- **distilled** from Qwen3-Coder-480B, then **filtered** to drop trajectories in which the teacher ignored the instructions and "still tr[ied] to execute anyway".

::figure official/lectures/images/swezero-prompt.png | the standard OpenHands workflow (explore, test, implement, verify) next to SWE-Zero's: "CANNOT RUN PYTHON CODE", no tests, python, pytest, pip and apt prohibited
::code lecture_14.py:L431-L440 | heavy dependencies; solve without execution; 300K trajectories from 150K PRs; no future commits; distill and filter

A second set, **SWE-Hero**, has 13K trajectories that do use execution feedback. The models are fine-tuned on SWE-Zero first, then on SWE-Hero.

::figure official/lectures/images/swezero-results.png | SWE-bench Verified against model size: SWE-Zero gets a 32B model to 57.5%, and the execution-based SWE-Hero stage adds 4.7 to 6.3 points at each size (62.2% at 32B); the strongest large models still reach 70-78%

::note slip 1:21:26 | Aloud: "first fine-tune on these SWE-Zero examples, and then fine-tune again on these SWE-Zero examples". The second stage is SWE-Hero's 13K execution trajectories, named a sentence earlier.
::note skip 1:20:41 | "There's a lot of details how to prevent agent hacking": the anti-hacking measures, including removing future commits, were not explained aloud.

### SWE-rebench: mine real PRs

SWE-rebench grabs "tons and tons of PRs": 450K from GitHub and GitHub Archive. For each repository an LLM (Qwen2.5-72B-Instruct) hypothesizes a dependency-installation script, which is validated by actually installing and running the tests, and retried if it fails ("most of them probably failed and try harder"). An LLM also labels each task's quality. The result is 21K interactive Python tasks from 3.4K repositories.

::figure official/lectures/images/swe-rebench.png | preliminary filtering of GitHub and GH Archive, an install-and-validate loop driven by an LLM, then LLM labelling: 21,000+ tasks
::code lecture_14.py:L445-L447 | 450K PRs in, 21K executable tasks from 3.4K repositories out
::note slip 1:22:06 | Aloud, SWE-rebench uses "a language model to give you the responses". The code says the LLM installs dependencies and assesses PR quality; it does not write solutions.

### SWE-ZERO-12M: scale the execution-free idea

Released the day of the lecture: SWE-Zero's recipe scaled to **12M agent trajectories**. Being execution-free is what makes it cheap. It uses SWE-rebench-v2's tasks: 32K executable and 120K that could not be made to run. An execution-based pipeline must throw the 120K away; "SWE-Zero doesn't care. You can use all of them." The trajectories come from mini-coder-1.7b, a very small model (50.4 pass@100), in the mini-swe-agent scaffold.

::code lecture_14.py:L451-L453 | 12M trajectories over 32K executable and 120K non-executable tasks, from a 1.7B model

### Post-training data, summed up

The datasets are getting more sophisticated, from environment-free math to coding with real repositories, and the coding datasets are growing fast.
- **Prompts** come in three kinds: **fully synthetic**; **semi-synthetic**, a real environment with synthetic tasks (SWE-smith's injected bugs); or **real** (GitHub PRs, as in SWE-Zero and SWE-rebench).
- **Responses** come from capable models that are also good teachers, which is not the same thing (OpenThoughts).
- **Code environments are painful**, which is exactly what SWE-Zero sidesteps.
- There is "a lot of filtering and other details", which there was no time for.

::code lecture_14.py:L456-L460 | prompts fully synthetic, semi-synthetic or real; good teachers; painful environments
::kp swe-synthetic-tasks-execution

## What should you take away? {#summary}
source: lecture_14.py:L30-L36 · video 1:23:44-1:24:40

- **Filtering**: define what good looks like (an existing dataset, or a prompted LM's choices), train a lightweight classifier for it (language ID, quality, toxicity), and run it over the crawl to get the small subset you want. How hard to filter depends on how long you will train.
- **Deduplication**: hashing makes it linear time; MinHash makes collisions track Jaccard similarity; LSH's bands sharpen that into a threshold at $(1/b)^{1/r}$. Run it across all sources. It saves flops and reduces memorization.
- **Mixing**: try mixtures at small scale and extrapolate to the optimal mixture at large scale, while counting epochs (cap them, or simulate them by downsampling).
- **Post-training data** looks like evaluations, and today it is mostly synthetic: real or synthetic tasks, a good teacher, a lot of filtering.

The professor closed with a caveat: real data work is "very grungy", domain-specific, and driven by looking at concrete examples. "This lecture is not really representative of what data work is like"; it is a map of the landscape.

::code lecture_14.py:L30-L36 | the lecture's own summary
::video 1:24:13-1:24:33 | data work is grungy and example-driven
