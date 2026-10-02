---
title: L13 · Data I: sources and datasets, read through
minutes: 45
---
The lectures so far took the training data as given. This one asks where it comes from: what "trained on the Internet" really means, what the law lets you copy, how Common Crawl and a few curated sources are collected, and how a decade of open datasets, from BooksCorpus to Nemotron-CC, turned raw web pages into training tokens. After it you can say why a given page is or is not in a dataset, read a dataset recipe (extraction, language ID, rules or classifier, deduplication, mixture) and see what each step throws away, and judge a reported token count against a training budget.

## Why is data the thing to get right? {#why}
source: lecture_13.py:L6-L87 · video 0:04-4:42

So far the course has been about how to train a model *given data*, and, [last lecture](#/read/lecture_12), how to tell a good model from a bad one. This lecture and the next ask *what data* to train on. The professor's claim is that data is the most important thing to get right.

His evidence is what companies disclose. An open-weight model such as Llama 3 publishes its architecture in full (it has to, the weights are public) and even its training procedure, but says next to nothing about its data.

::figure official/lectures/images/llama3-data.png | the entire description of Llama 3's pre-training data: "a variety of data sources", deduplication and cleaning, PII and adult domains removed. No source is named.

There are two reasons for the secrecy:
1. **Competitive dynamics.** Data is "your competitive secret sauce"; you do not want competitors to know what you did.
2. **Copyright liability.** You do not want to be sued for saying you trained on certain data. This lecture returns to the point several times, and one dataset (Books3, below) shows exactly how disclosure leads to lawsuits.

::code lecture_13.py:L52-L59 | full transparency into architecture and training, none into data, and the two reasons

Data work has changed shape but not importance. Before foundation models it meant annotating labels for supervised learning; now there is less annotation, at least in pre-training, but a great deal of curation and cleaning. The professor's reason it stays a bottleneck: data is a long-tail problem that "scales with human effort". Only so many people can usefully work on an architecture or a training system, but the data for a model meant to do everything splits into thousands of independent pieces of work, so it parallelizes across people. That is why data teams at model developers are "actually quite big".

### Three stages, from a lot of rough data to a little good data

Data enters training at three stages:
1. **Pre-training**: raw text, mostly documents from the web.
2. **Mid-training**: more training on higher-quality data to enhance particular capabilities (aloud he added long context as an example).
3. **Post-training**: chat transcripts, or environments for reinforcement learning. It becomes task-specific.

The lines are blurry and there can be more than three stages, but the trend holds throughout: you go from **large amounts of lower-quality data to small amounts of high-quality data**.

The vocabulary follows the stages. A **base model** is the result after pre-training *plus* mid-training, not pre-training alone; an **instruct** or **chat model** is the result after post-training. Increasingly, base models are not released at all. Qwen3.5-397B-A17B ships only as an instruct model; "you don't see the intermediate checkpoints".

::predict training-stages-quality-gradient
::code lecture_13.py:L65-L77 | three stages, the large-to-small trend, and where "base" and "instruct" sit

The fully open OLMo 2 from AI2 shows all three stages. Read the three tables top to bottom and watch the size and kind of data change.

::figure official/lectures/images/olmo2-pretraining.png | pre-training: 3.90T tokens, of which DCLM-Baseline web pages are 3.71T; code, papers, math and Wikipedia are the remaining slivers
::figure official/lectures/images/olmo2-dolmino.png | mid-training (Dolmino): an 833B "high quality" subset (note DCLM again, now only its top 7% by fastText score) plus a 10.7B math mix, much of it synthetic
::figure official/lectures/images/tulu.png | post-training (Tülu 3): prompt datasets counted in thousands, about 0.94M prompts used for supervised fine-tuning, by category (general, knowledge, math, coding, safety, multilingual, instruction following)

From 3.9 trillion web-heavy tokens, to about 0.84 trillion selected ones, to under a million curated prompts: the trend in numbers. The lecture's question is now posed: what are these datasets, and how are they chosen and processed?

::video 3:02-3:11 | "from training on large amounts of low quality data to smaller amounts of high quality data"
::note skip 2:23 | "Today we're focusing mostly on pre-training." Mid-training gets the OLMo example only, and post-training data is deferred to the next lecture.
::kp training-stages-quality-gradient

## Can you train on "the whole Internet"? {#access}
source: lecture_13.py:L90-L147, L42 · video 4:42-14:20

You hear that language models are "trained on the entire Internet". The professor's first objection is that this "doesn't really quite type-check". The Internet is a set of live servers you can send a request to and get a response from. Training on *that* would mean an agent that goes online and acts, which is reinforcement learning, not pre-training. "The public world wide web" is closer, and still not right.

To train on the web, someone first has to turn it into files. That is a **crawler**: it discovers pages, starting from a seed set of URLs and following links, and downloads what it finds. You may not build it yourself; the lecture's next big section is Common Crawl, a crawl anyone can download.

::code lecture_13.py:L95-L101 | the web is live servers; a crawler discovers and downloads pages

Even a perfect crawler cannot download every page. The lecture sorts the obstacles into four kinds.

| obstacle | what it is | examples |
|---|---|---|
| dynamic content | sites that are apps: the URL does not change, content appears after clicking or submitting forms | Discord, wandb |
| authentication | login, usually payment: "walled gardens" | Facebook, X, LinkedIn, NYTimes |
| technical restrictions | robots.txt (voluntary); bot detection with CAPTCHAs (Cloudflare); IP or country blocks; rate limits | nytimes.com/robots.txt |
| legal restrictions | terms of service forbidding bots; no licence to copy the pages for training | |

Some details the professor added aloud:
- Dynamic content is "the deep web": the traditional model of crawling, follow the hyperlinks, never reaches it.
- Walled gardens favour their owners. Facebook does not need to crawl Facebook, and "if you're X AI, you can train on X"; everyone else cannot.
- **robots.txt** is a file at a site's root saying which crawlers may fetch what. The New York Times' file disallows bots such as OAI-SearchBot, PerplexityBot, ChatGPT-User and ClaudeBot. It is "not a legal restriction": obeying it is just being a good citizen, "not even a contract".
- Terms of service *are* the legal layer: a contract for using the site, which often says bots may not use the content, or not for AI training. And even when the terms are silent, the content itself may carry no licence for you to copy it.

::code lecture_13.py:L105-L123 | dynamic content, authentication, technical restrictions, legal restrictions
::predict web-is-not-downloadable

The restrictions are also growing. The *Consent in Crisis* study checked robots.txt and terms of service over time for the URLs behind common datasets (C4, RefinedWeb, Dolma, all met later).

::figure official/lectures/images/decline-consent.png | top: robots.txt restrictions, with the dark-red "full restrictions" band thin until 2023 and widening after ChatGPT and GPTBot (the shaded right part is a forecast); middle: terms of service, where pages with no terms (grey) give way to "no crawling" and "no AI" terms; bottom: restrictions by crawler, OpenAI's agent most restricted

His reading: until 2023 the picture was flat; then the share of sites with full robots.txt restrictions jumped. For terms of service, around 2016 almost no pages had any; now most do, and most forbid AI use. So even if the web could be crawled in 2020, what you can *legally* crawl today is much smaller.

::note aside 10:39 | Aloud the jump was read as reaching "almost 50%". In the figure the measured dark-red band reaches roughly a quarter of sites by mid-2024; it is the shaded forecast that approaches half by 2025. The direction of the trend is the point either way.

Crawlers do not always behave. In July 2024 the CEO of iFixit complained that Anthropic's crawler hit their servers "a million times in 24 hours"; Read the Docs replied that it was being hammered too.

::figure official/lectures/images/anthropic-crawling.png | the complaint and the reply; note the reply's prediction that abuse, "not even" copyright, will get AI crawlers blocked

This is a problem before copyright even enters: a crawler can violate terms of service or robots.txt, or simply generate so much load that it costs the site money and degrades service for its other users.

::note aside 11:48 | The professor did not remember the site's name ("I forget the name of this site"); the screenshot shows Kyle Wiens of iFixit.

### Shadow libraries

One more corner of the web: **shadow libraries** such as Library Genesis (LibGen), Z-Library, Anna's Archive and Sci-Hub. They collect copyrighted books and papers (by the lecture's cited counts, LibGen about 4M books in 2019, Sci-Hub about 88M papers in 2022), bypass paywalls such as Elsevier's, and survive takedown orders, lawsuits and country blocks by moving servers abroad. Their operators argue they free what should be free; legally, it is "piracy and copyright infringement". They are "technically part of the web", so a crawl can pick them up, and they come back twice: in the Anthropic lawsuit and in the Books3 dataset.

::code lecture_13.py:L136-L142 | part of the web, but piracy

The summary: the Internet is huge, and there are many technical and legal limits on what you can access. Every dataset in this lecture follows the same pipeline: **live service → raw data → processed data** (transformation, filtering, deduplication).

::video 4:45-4:57 | why "trained on the entire internet" does not type-check
::note deferred 13:45 | "We'll come back to this point later": shadow-library piracy returns in Bartz v. Anthropic and in Books3.
::kp web-is-not-downloadable

## What is copyrighted? Almost everything {#copyright}
source: lecture_13.py:L150-L177 · video 14:20-19:20

Suppose you obey every robots.txt, terms of service and rate limit. You still have to ask whether you are *allowed to train* on what you downloaded. That is an open legal question, though the last year brought several developments, and the frame for it is **intellectual property law**. Its goal is to *incentivize* the creation of intellectual goods; the professor stressed that this is the spirit of the law, not saying no to everything. Of its forms (copyright, patents, trademarks, trade secrets), copyright is the one that matters for training data.

Copyright goes back to 1709 in England (the Statute of Anne), the first time governments and courts regulated it. In the United States the current law is the Copyright Act of 1976, which protects "original works of authorship fixed in any tangible medium of expression". Five properties matter here:

- **Collections are not copyrightable**, unless there is creativity in the selection or arrangement. A telephone directory, listed alphabetically, is not protected.
- **Expression, not ideas.** You cannot copyright the quicksort algorithm; you can copyright a particular implementation of it.
- **"Fixed", not "published".** The 1909 law protected published works; since 1976, anything written down is protected. No registration is needed, in contrast with patents, which you must file for and pay for.
- **The threshold is extremely low.** Put something on your website and it is copyrighted. "That's it."
- **Registration matters only to sue**, and it costs 65 dollars, "much smaller than the lawyer fees", so it is no real barrier.

Copyright expires (the lecture says after 75 years) and the work enters the **public domain**, where anyone can use it: Shakespeare, Beethoven, most of Project Gutenberg. The rationale is again the incentive: protect creators for a while, but "after 75 years, presumably, it's not worth protecting, or they're passed away".

::code lecture_13.py:L160-L171 | fixed in a tangible medium; collections and ideas excluded; no registration; extremely low threshold; 75 years
::note warning | "75 years" is the lecture's simplification. Actual US terms depend on the work and its date (for example, the author's life plus 70 years for recent individual works); the point that old works eventually become free is what the lecture uses.

The summary follows: *basically everything on the Internet is copyrighted*. Does that make training on anything a violation? Not necessarily. There are two ways to use a copyrighted work:
1. get a **licence** for it;
2. appeal to **fair use**.

::predict copyright-low-threshold-expression
::video 17:32-17:41 | no registration needed: "you put something on your website, it's copyrighted"
::note slip 1:17:22 | Near the end, discussing Common Pile, he said "I think collections can be copyrighted", against his own "collections are not copyrightable" here (16:26). The lecture's exception reconciles them: a collection with creative selection or arrangement can be.
::kp copyright-low-threshold-expression

## How can you use a copyrighted work? Licences, fair use, terms of service {#licences}
source: lecture_13.py:L179-L217 · video 19:20-27:35

### Licences

A **licence**, from contract law, is granted by a licensor to a licensee. Effectively, "a license is a promise not to sue": you may use the work in the ways the licence permits. Ownership does not move.

The **Creative Commons** licences, created in 2001 by Lessig and Eldred to bridge the public domain and ordinary copyright, let a creator say "use this": the work then acts "like it's in the public domain" without waiting out the copyright term. Wikipedia, Open Courseware, Khan Academy, the Free Music Archive, and many images on Flickr and videos on YouTube carry them. Public-domain and Creative Commons material together is what the lecture calls **permissive**.

For everything else, if you have money, you buy a licence. Model developers do: Google with Reddit, OpenAI with Shutterstock, OpenAI with Stack Exchange.

::code lecture_13.py:L180-L190 | a licence is a promise not to sue; Creative Commons; paid licensing deals

### Fair use

If you do not want to pay, you can argue **fair use** (section 107 of the Copyright Act): use without a licence that the law permits. Four factors decide it, and none is a hard rule; they are "just tendencies which have to be weighed in court".
1. **Purpose and character of the use.** Educational is favoured over commercial; *transformative* over reproductive (rehosting an identical copy is the worst case).
2. **Nature of the work.** Factual is favoured over fictional, non-creative over creative: a page of facts about World War II is less protected than a poem.
3. **Amount used.** A snippet is favoured over the whole work.
4. **Effect on the market** for the original. Copyright exists to align economic incentives, so a use that substitutes for the original and cuts what its author can earn counts against you; one that goes to a new market counts for you.

::code lecture_13.py:L192-L197 | the four factors

Examples of fair use: writing a summary of a movie you watched; reimplementing an algorithm (the idea) rather than copying the code (the expression); and Google Books, which indexes copyrighted books and shows snippets. The lecture dates Authors Guild v. Google 2002-2013; it ended in Google's favour, which is why the feature exists, and it set precedent for thinking about training.

::note slip 24:12 | Aloud the case was "settled in favor of Google" after 11 years. It was decided, not settled: the authors sued in 2005, a proposed settlement was rejected in 2011, and the courts held Google Books to be fair use in 2013 (affirmed on appeal in 2015). The lecture's 2002 predates the suit (it is roughly when Google's book-scanning project began).

One point the professor flagged as new "for an ML audience": **copyright is not about verbatim memorization**. Many papers measure verbatim regurgitation, but that is only one way to infringe.

Plots and characters can be copyrighted (Harry Potter the character, not just the books), while parody, imitation to make fun of something, is likely fair use. Copyright is about semantics and economics, "definitely not about N-gram overlap".

### What this means for language models

The lecture's four considerations:
- **Copying is already the act.** Downloading the data, the first step of training, is potentially a violation even if you never do anything with it. The word is *copy*right.
- **Training should be transformative.** He hedged this aloud: "not necessarily a fact, but intuitively has a transformative flavor". A model uses the data as a means to an end, to extract general ideas about the world, which seems different from rehosting the work.
- **General idea, not concrete expression.** A model should learn about wizards, not reproduce Harry Potter.
- **Markets.** Regardless of copyright, language models can affect the market for writers and artists, and that is factor 4: harming the market makes fair use less likely.

::code lecture_13.py:L209-L213 | copying is already a violation; training should be transformative; idea over expression; market effects
::video 24:33-24:47 | copyright is not verbatim memorization, a point many ML papers miss

### Terms of service: a third layer

Even with a licence or a fair-use argument, the site's **terms of service** can add restrictions. YouTube hosts many Creative Commons videos, yet its terms forbid downloading videos with a bot. So there are several layers: copyright (answered by licence or fair use) and the contract you accepted by using the site.

::code lecture_13.py:L215-L217 | terms of service can restrict even licensed works
::predict license-and-terms-of-service
::video 27:07-27:32 | licence and terms of service are separate layers (YouTube)
::note spoken 31:40 | Asked what happens if a licence is later made stricter: "first, I'm not a lawyer", but he believes content obtained under the old licence can still be used, while for a living source such as Reddit the new licence governs the new content, so "you wouldn't be able to train on the later things".
::kp license-and-terms-of-service

## What have the courts said? {#lawsuits}
source: lecture_13.py:L219-L238 · video 27:35-32:12

Three cases frame the current landscape.

- **The New York Times v. OpenAI (2023).** The allegation: OpenAI trained on NYT articles and ChatGPT could be prompted to reproduce them almost verbatim. The lecture gives no outcome.
- **Bartz, Graeber and other authors v. Anthropic (2024).** The allegation: Anthropic pirated millions of books and trained Claude on them. The 2025 summary judgement split the question in two. *Training* on the plaintiffs' works was fair use. *Pirating* the copies was not, even copies never trained on: that has nothing to do with machine learning, piracy was already illegal. Anthropic had also bought physical books, cut off the bindings and scanned them, and the court found that fair use too, but buying a book afterwards does not cure having pirated it first. Anthropic settled for 1.5 billion dollars, which the professor put at about 3,000 dollars a book, so roughly 500,000 books.
- **Kadrey, Silverman and other authors v. Meta.** The allegation: Meta trained on the plaintiffs' books, something revealed by Meta's own LLaMA paper (see LLaMA's data, below). The 2025 judgement: training on the books was fair use, "in this instance". A separate allegation, that Meta torrented books, was still pending; his opinion was that if precedent holds, it will not go well for Meta.

::code lecture_13.py:L223-L233 | Anthropic: training fair use, piracy not, 1.5 billion dollars; Meta: training fair use, torrenting pending

The summary: so far training has been held fair use (or at least not held to be unfair), but the rulings "have so far been narrow", about specific instances, not any training on any copyrighted content. Pirating books is clearly illegal. The area is active and changing.

::predict fair-use-training-vs-piracy
::video 28:08-28:30 | the Anthropic ruling: training was fair use, pirating the books was not
::note spoken 44:08 | Asked how a crawl can avoid pirated books: "you can't". Common Crawl most likely contains copyrighted books, and developers appeal to fair use. A book is not legally different from your website, since both are copyrighted; a published author is just better placed to defend it in court. For those who want to be really careful, the end of the lecture (Common Pile) shows the alternative.
::note aside 27:55 | On NYT v. OpenAI the captions read "I don't think this is still pending"; the lecture text gives no outcome, and none is assumed here.
::note aside 30:59 | A question about voice data (ElevenLabs) was taken offline.
::kp fair-use-training-vs-piracy

## How is the web turned into files? Common Crawl {#common-crawl}
source: lecture_13.py:L241-L271 · video 32:12-36:08

Most model developers run their own crawler, "because they want to have full control" over the data. For everyone else there is **Common Crawl**, a non-profit founded in 2007. Roughly every month it runs a crawl that adds 3-5 billion pages; crawls overlap but try to diversify, and the total so far is about 300 billion pages.

For scale, by the figures the lecture links to: Google's search index is at least 100 PB, and the April 2026 crawl alone has 2.19 billion pages, 372.2 TB (about 170 KB of raw response per page, mostly text, no images).

::code lecture_13.py:L244-L251 | monthly crawls of 3-5 billion pages, 300 billion in all, and the April 2026 crawl
::note aside 33:01 | The professor doubted the 300 billion total ("if you multiply this number by 20, you don't quite get 300 billion. But that's what they say"). The intended arithmetic is unclear; the figure is Common Crawl's own.

### Crawling is graph traversal

Common Crawl uses Apache Nutch. The algorithm is simple, "all the gory details are in the implementation": start from a set of seed URLs (for Common Crawl, at least hundreds of millions), keep a queue, pop a URL, download the page, add its hyperlinks to the queue, repeat, in parallel over many machines.

::figure https://upload.wikimedia.org/wikipedia/commons/thumb/d/df/WebCrawlerArchitecture.svg/330px-WebCrawlerArchitecture.svg.png | the loop: a scheduler pulls URLs from the queue, a multi-threaded downloader fetches pages from the web, stores text and metadata, and sends the new URLs back to the queue

Three policies steer it:
- **selection**: which pages to download;
- **politeness**: respect robots.txt and do not overload a server;
- **re-visit**: how often to check whether a page changed, so frequently changing pages are fetched often and static ones are not refetched for nothing.

The hard part is that URLs do not map one-to-one to content. The same URL can return different content depending on browser state, and many URLs return the same content (tracking parameters, mirror sites, which exist precisely to duplicate). Without care, a crawl fills with duplicates, which is why every dataset below deduplicates.

::code lecture_13.py:L253-L262 | seed URLs, pop-download-enqueue, and the three policies
::predict common-crawl-mechanics
::video 33:41-33:55 | crawling as graph traversal: pop, download, enqueue the hyperlinks
::kp common-crawl-mechanics

### WARC or WET: the HTML-to-text step

Common Crawl ships each crawl in two formats:
- **WARC**: the raw HTTP response, i.e. the HTML exactly as the server sent it;
- **WET**: text that Common Crawl already extracted from it, a necessarily lossy conversion.

WET is convenient but "not necessarily the best way to use the web". Converting HTML to text is a choice, and tools such as [trafilatura](https://trafilatura.readthedocs.io/en/latest/) and [resiliparse](https://resiliparse.chatnoir.eu/en/stable/) do it differently. The DCLM paper (met again near the end) trained models on the same crawl extracted three ways and compared their downstream accuracy:

::figure official/lectures/images/dclm-wet.png | same data, three extractors: resiliparse and trafilatura score about 24 on DCLM's CORE benchmark, the WET files 20.7

So the extractor alone moves the trained model's accuracy by several points. That is why later datasets start from WARC and pick an extractor on purpose: The Pile used jusText, RefinedWeb trafilatura, and Nemotron-CC jusText because it returned more tokens. The criterion for judging an extractor is the one in this table, the accuracy of a model trained on its output, not how clean the text looks.

::code lecture_13.py:L264-L270 | WARC is raw HTML, WET is lossy text, and the conversion matters for downstream accuracy
::predict warc-wet-extraction-loss
::video 35:26-35:49 | WET is a lossy conversion, and the converter matters
::kp warc-wet-extraction-loss

## Which sources are worth getting separately? Wikipedia, GitHub, arXiv {#curated}
source: lecture_13.py:L274-L327 · video 36:08-45:00

The web is not uniform. It is not a set of interchangeable sites from which you take a random fraction; it has "specific pockets of really interesting, high quality content". Three of them get their own treatment, and each is obtained without a crawler.

### Wikipedia

A free encyclopedia since 2001 (by its own count, 67 million articles across 361 language editions as of May 2026). Two policies shape it:
- **no original thought**: no opinions, promotion or personal pages; everything is referenced and cited;
- **notability**: an article needs significant coverage in reliable sources.

You might conclude Wikipedia holds nothing that is not already on the web. Not quite: it also cites books, which a crawl cannot easily reach. Anyone can edit; vandalism is reverted by administrators (and bots); and, as in any peer-production system, a few people do most of the work (one editor, Steven Pruitt, has about 5M edits).

What matters for data: Wikipedia publishes **periodic dumps** every few weeks, the whole site in one archive. You download those instead of crawling, and Wikipedia prefers it ("they don't want you to crawl Wikipedia").

The dumps create a vulnerability, given as an aside. Vandalism normally gets rolled back, but the dumps happen on a known schedule. An attacker can make a malicious edit right *before* a dump; the edit is reverted afterwards, but the dump keeps it. Other work shows that injected examples can make a model attach negative sentiment to a trigger phrase such as "iPhone". The takeaway: even "high quality" sources can contain adversarial content. He thinks this particular hole has since been fixed.

::code lecture_13.py:L286-L294 | anyone edits, a few do most of it, periodic dumps, and poisoning timed before a dump
::predict curated-sources-bulk-dumps

### GitHub and Software Heritage

Code helps with programming, and also, the lecture says, with reasoning, a claim it marks as "(folklore)". **GitHub** (founded 2008, Microsoft since 2018; by the linked count 420M+ repositories, 28M public) hosts repositories that are not files but directories with commit history, issues, pull requests and comments. There are many duplicates, from copied code and forks. GitHub allows training on any public repository with a permissive licence such as MIT or Apache.

There are two kinds of data, obtained two ways:
- **repositories**: clone them with the git protocol, rather than scraping the GitHub website;
- **metadata** (issues, pull requests, comments): from the GitHub API, whose event stream the GitHub Archive records in hourly snapshots, "basically every single comment, or star, or action on GitHub".

**Software Heritage**, a non-profit founded in 2016 to preserve software, aggregates repositories from GitHub, GitLab, Bitbucket, PyPI and more, but keeps repositories, not metadata.

::code lecture_13.py:L303-L309 | repositories by git, metadata from the event stream
::note slip | In the lecture source the "GitHub Archive" link (L309) points at arXiv's S3 bulk-data page, the same URL used correctly for arXiv at L327. The GitHub Archive is a separate project (gharchive.org).

### arXiv

A free preprint server since 1991, starting with physics and now covering math, CS, statistics and more (about 3M submissions). Each submission has metadata, a PDF and, optionally, LaTeX source; approval is light, with no peer review. So "what does it mean to train on arXiv?" is already a design choice: convert the PDF to text, or use the LaTeX source (LLaMA, below, used LaTeX). Licensing is clear: authors choose all rights reserved or a Creative Commons licence such as CC-BY, and the metadata (title, abstract) is CC0. And again, no crawling: bulk download from Amazon S3.

::code lecture_13.py:L323-L327 | metadata, PDF, LaTeX; authors pick the licence; metadata is CC0; bulk download
::note slip 43:01 | "I think most arXiv papers are Creative Commons." The lecture text only says authors choose; by public tallies most choose arXiv's default non-exclusive distribution licence, which is not Creative Commons. For a dataset, check each paper's licence.

The common thread: these sources come as dumps or bulk downloads with structure and licence information attached, which you can use for filtering. Stack Exchange, met with The Pile below, is the same kind of source.

::video 38:09-38:27 | download the dump; Wikipedia does not want to be crawled
::note spoken 43:39 | Asked whether model-generated (synthetic) data may be used: "The short answer is probably yes", with the longer answer deferred to Common Pile at the end.
::kp curated-sources-bulk-dumps

## 2019: how do you get good text out of a messy crawl? {#2019}
source: lecture_13.py:L330-L404 · video 45:00-52:36

With the landscape set, the lecture walks through the datasets actually used to build models, from 2019 on. Watch the design choices: each one is a different answer to the question of what counts as "good" text.

### BERT: Wikipedia and books

BERT (2018) trained on Wikipedia and books. The books were **BooksCorpus**: in 2015 researchers took the self-published e-books priced at zero on Smashwords (a site where anyone can publish an e-book) and made a corpus of 7K books, 985M words. It circulated in academia for years, "back in the innocent days when no one was paying attention", and has since been taken down for violating Smashwords' terms of service. In one line: "Just because it was free and you can get it, doesn't mean it was legally allowed."

BERT also trained on **documents rather than sentences**. Earlier language-modeling benchmarks, such as the 1 Billion Word Benchmark (sentences from machine translation), were sentence-level; long documents like books teach structure that single sentences cannot.

::code lecture_13.py:L342-L350 | BooksCorpus: free Smashwords books, 7K books, 985M words, taken down over terms of service

### GPT-2's WebText: let Reddit vote

GPT-2 wanted high-quality web text, and Common Crawl was considered too messy. The trick: keep pages that are outgoing links from Reddit posts with at least 3 karma. "A good post must link to good websites": the karma of the referring post is a surrogate for the quality of the page. The result, **WebText**, was 8 million pages, 40 GB of text. OpenAI never released it, but **OpenWebText** replicated it: extract all URLs from a dataset of Reddit submissions, keep English with Facebook's fastText language classifier, remove near-duplicates.

::code lecture_13.py:L354-L361 | WebText: Reddit links with at least 3 karma; OpenWebText: the open replication
::note slip 47:20 | Aloud, "greater than three karma"; the lecture text and the GPT-2 paper say at least 3.

### CCNet: keep what looks like Wikipedia

CCNet (Facebook, 2019) wanted an *automatic* way to build large, high-quality pre-training sets, especially for low-resource languages such as Urdu, so nothing English-specific or manual. Its pipeline has three parts, which every later pipeline reuses in some form:
1. **deduplication**: remove duplicate paragraphs after light normalization;
2. **language identification**: a fastText classifier; keep the target language;
3. **quality filtering**: train a KenLM 5-gram language model on Wikipedia and keep documents that score as likely under it, i.e. that look like Wikipedia.

Because filtering Common Crawl yields far more text than Wikipedia itself, BERT models trained on CCNet outperformed those trained on Wikipedia. (CCNet names both the tool and the released dataset.)

::code lecture_13.py:L369-L375 | dedup, language ID, a Wikipedia 5-gram model as the quality filter

### C4: write rules instead

C4, the Colossal Clean Crawled Corpus, comes from the T5 paper (Google, 2019), better known for casting every NLP task as text-to-text, but the dataset was a major contribution. Its premise: "Common Crawl is mostly not useful natural language". At the time you had small clean datasets, or Common Crawl, "a mess", and training on it "led to junk results". Where WebText used Reddit and CCNet used a Wikipedia model, C4 used **hand-written rules**, applied to one Common Crawl snapshot (April 2019, 1.4 trillion tokens):

- keep only lines that end in punctuation and have at least 5 words;
- remove pages with fewer than 3 sentences;
- remove any page containing a word from a list of "bad words";
- remove pages containing `{` (no code), "lorem ipsum", "terms of use" and similar boilerplate;
- keep English, by langdetect with probability at least 0.99.

The result: 806 GB of text, 156 billion tokens, much larger than WebText's 40 GB.

::code lecture_13.py:L385-L396 | one snapshot, five rules, 806 GB / 156B tokens
::predict rule-based-heuristic-filters

The kept fraction is $156\text{B} / 1.4\text{T} \approx 11\%$: about one token in nine survives. And the two size units line up through the tokenizer's compression ratio ([L1](#/read/lecture_01): bytes per token): $806\text{ GB} / 156\text{B} \approx 5.2$ bytes per token.

::widget fixture:lecture_13--funnel | click "C4 (T5) (rules)": one sized step, 1.4T → 156B tokens, keeps 11.1%; the five rules are listed in order, but the lecture never sizes them one by one
::kp lecture_01:compression-ratio

The rules are not neutral. The `{` rule "filters out a lot of code", which shows that "at that time, they weren't thinking about code models"; the punctuation rule removes bullet lists, tables and equations. Whatever a rule finds un-prose-like, it deletes. Later datasets that wanted code or math had to add it back from separate sources.

::video 51:02-51:18 | the "{" rule silently removes code
::note slip 50:42 | Aloud the line rule was "more than five words" and C4 "800 gigabytes"; the text says at least 5 words and 806 GB.

An analysis of C4 (Dodge et al., 2021) shows where its text comes from.

::figure https://stanford-cs324.github.io/winter2022/lectures/images/c4-domains.png | tokens by top-level domain (left, .com far ahead) and by website (right): patents.google.com first, then Wikipedia, then news sites such as the NYT

C4 also built a **WebText-like** variant: filter to pages linked from Reddit posts with at least 3 karma. From 12 Common Crawl dumps it found only 17 GB, against WebText's 40 GB. The professor's reading: Common Crawl is incomplete, since a complete crawl should have reached 40. The variant still improved NLP benchmarks such as GLUE and SQuAD.

::code lecture_13.py:L401-L404 | 12 dumps give 17 GB of WebText-like pages against WebText's 40 GB

So by 2019 there were three ways to define quality: links from upvoted Reddit posts, resemblance to Wikipedia, and rules.

## 2020-2021: classifiers, mixtures and rules at scale {#2020}
source: lecture_13.py:L407-L486 · video 52:36-59:40

### GPT-3: a quality classifier

GPT-3's dataset (2020) combined four parts:
- **Common Crawl**, processed;
- **WebText2**: WebText expanded with more links. It overlaps the Common Crawl part, but it is "a more targeted distribution", so the mixture keeps both;
- **Books1 and Books2**, described only as "Internet-based books corpora", which "remains a mystery";
- **Wikipedia**.

The result was 570 GB, 400 billion tokens. For the Common Crawl part, GPT-3 trained a **quality classifier** to separate WebText, Wikipedia, Books1 and Books2 (the trusted sources) from the rest, and kept what the classifier scored as similar to them. It also did fuzzy deduplication of documents, including against WebText and against benchmarks (the second is decontamination; see [L12](#/read/lecture_12) on why test data in training corrupts evaluation).

::code lecture_13.py:L408-L418 | four components; a classifier trained on the trusted sources; fuzzy dedup including benchmarks
::note spoken 53:44 | Aloud, the reason for the fuzzy dedup was that "WebText and Common Crawl had dupes", i.e. overlap between parts of the mixture; the benchmark decontamination in the text was not mentioned. The size was read as "about 500 gigabytes"; the text says 570 GB.

All of 2019's proxies and GPT-3's classifier share one idea: **quality is defined by resemblance to a reference you already trust**.

::predict reference-set-quality-proxy

### The Pile: 22 curated domains

GPT-3 was a big event, and open efforts followed. **The Pile** (EleutherAI, 2021) was a grassroots project: volunteers on Discord "jamming" on high-quality sources. Instead of one filtered crawl, it curated 22 domains: 825 GB of text, about 275B tokens.

::figure https://stanford-cs324.github.io/winter2022/lectures/images/the-pile.png | the 22 components with size, sampling weight and epochs: Pile-CC is the largest but only 18% of the weight; PubMed Central, Books3, OpenWebText2 and arXiv follow; small sources are seen 2-3 times, so the "effective size" (1254 GiB) exceeds the raw 825 GiB

Some of the components:
- **Pile-CC**: Common Crawl, extracted from WARC with jusText, which beat the WET files;
- **PubMed Central**: 5 million papers, public because NIH-funded work must be;
- **arXiv**, using the LaTeX;
- **Enron emails**: 500K emails from 150 senior Enron employees, released during the 2002 investigation; "one of the few email datasets we have", if "a weird distribution" for email;
- Project Gutenberg, Books3 and Stack Exchange, next.

Note the Epochs column: a mixture is not just a list of sources but a weighting, and upweighting a small source means repeating it. The Pile's effective size is $1254 / 825 \approx 1.5$ times its raw size. Remember this when reading token counts at the end of the lecture.

::code lecture_13.py:L424-L433 | 22 domains, 825 GB; Pile-CC from WARC; PubMed Central; arXiv; Enron

### Books: from cleared to pirated

The Pile brings in two book sources that sit at opposite ends of the legal spectrum.
- **Project Gutenberg**, started in 1971 by Michael Hart to widen access to literature (about 75K books in 2025, mostly English). It only includes books that have received copyright clearance, most of them in the public domain. **PG-19** is its books from before 2019, which is most of them, since public domain means old.
- **Books3** (2020): 196K books from the shadow library Bibliotik, including books by authors such as Stephen King, Min Jin Lee and Zadie Smith. "No one was paying attention" then; people trained on it; it has since been taken down over copyright infringement and lawsuits. "You cannot use, or should not use Books3 anymore."

::code lecture_13.py:L441-L453 | Gutenberg: copyright-cleared; Books3: 196K books from a shadow library, taken down

With BooksCorpus (free but scraped against terms of service) and GPT-3's undescribed Books1/2, that makes a spectrum of provenance: cleared, ToS-violating, unknown, pirated.

::predict books-provenance-spectrum

### Stack Exchange

User-contributed questions and answers, starting with StackOverflow in 2008 and spreading to topics such as math and literature, with reputation points and badges to reward participation. Two things make it valuable:
- **The Q&A format is close to real use.** We think of pre-training data as raw text, but parts of the web already "look supervised", which may teach question-answering behaviour directly. "Not everything is super magically emergent."
- **Metadata** (users, votes, comments, badges, tags) for filtering, in anonymized XML dumps: again, no crawling.

::code lecture_13.py:L463-L465 | Q&A close to real applications; metadata for filtering; XML dumps

### Gopher's MassiveText: rules again

DeepMind's Gopher (2021) was never released and was subsumed by Chinchilla (see [L9](#/read/lecture_09)), but its data description is thorough, "except for the parts where they don't tell you what's in the data". MassiveText combined MassiveWeb, C4, and books, news, GitHub and Wikipedia, the last four with no details. MassiveWeb's filtering:

- keep English; deduplicate; remove train-test overlap;
- quality filtering with **manual rules, not a classifier**, e.g. at least 80% of words must contain an alphabetic character; the reason given aloud was that rules gave them "more control";
- toxicity by Google SafeSearch, not word lists.

The result was 10.5 TB of text, but Gopher trained on only 300B tokens of it, about 12%.

::code lecture_13.py:L481-L486 | manual rules, SafeSearch, 10.5 TB of which 12% was used

The professor framed the period as a split "between people who wanted to use rules, and people who wanted to use classifiers". Rules are transparent and controllable; classifiers can capture "quality" no rule spells out, but they inherit whatever the reference set looks like.

::kp books-provenance-spectrum
::kp rule-based-heuristic-filters

## 2023-2024: open recipes, and is the web all you need? {#open-recipes}
source: lecture_13.py:L489-L537 · video 59:40-1:04:37

### LLaMA and RedPajama

The first LLaMA paper described its data processing in detail, "probably one of the last" models from a non-fully-open developer to do so. Its mixture:
- **Common Crawl processed with CCNet**, but with a twist on the quality classifier: instead of "does this page look like Wikipedia?", it asked whether the page is *referenced by* Wikipedia. Maybe Wikipedia articles are "too stylized", while the pages they cite are "presumably good";
- **C4** (more diverse; recall, rule-filtered);
- **GitHub**, keeping permissive licences, filtered by manual rules;
- **Wikipedia**, June-August 2022, 20 languages, manually filtered;
- **Project Gutenberg and Books3** (from The Pile);
- **arXiv**, from the LaTeX: comments removed, macros expanded inline, bibliography removed;
- **Stack Exchange**: the 28 largest sites, answers sorted by score.

The lecture gives the result as 1.2T tokens.

::code lecture_13.py:L490-L498 | CCNet with "referenced by Wikipedia", C4, GitHub, Wikipedia, books, arXiv LaTeX, Stack Exchange
::predict multi-source-mixtures
::note slip | The 1.2T figure (L498, also said aloud at 1:01:06) is the size of the RedPajama v1 reproduction. The LLaMA paper itself reports roughly 1.4T tokens after tokenization. The professor also dated LLaMA 1 to 2022, following the source's comment; the paper is from February 2023.

Listing Books3 "really got them in a lot of trouble": announcing it let anyone trace it back through The Pile to a shadow library, and it is the basis of the Kadrey v. Meta allegation. "That's why people don't want to talk about their data anymore", which closes the loop with the secrecy at the start of the lecture.

Meta did not release the dataset, but the description was enough for Together to reproduce it as **RedPajama v1**, which others then trained on. RedPajama also contained Books3 at first, since stripped out: early copyright decisions propagate downstream. Cerebras's **SlimPajama** is a 627B-token subset of RedPajama v1 made by deduplication alone, with MinHashLSH.

::code lecture_13.py:L500-L501 | RedPajama v1 reproduces LLaMA; SlimPajama deduplicates it to 627B
::widget fixture:lecture_13--funnel | the default view, SlimPajama: dedup alone keeps 627B of 1.2T, 52%, so nearly half of RedPajama was near-duplicate text
::video 1:00:09-1:00:25 | why LLaMA scored "referenced by Wikipedia" instead of "looks like Wikipedia"

### RefinedWeb and FineWeb: web only

**RefinedWeb** (2023, used to train Falcon) made the opposite bet to the curated mixtures: "web data is all you need". What if you skip GitHub, arXiv and Stack Exchange and just process the web well?
- HTML to text with trafilatura, from WARC rather than WET;
- filtering with the Gopher rules, deliberately avoiding ML-based filtering "to avoid biases". Aloud the concern was narrowness: "I don't want to find an overly narrow subset of the web";
- fuzzy deduplication with MinHash over 5-grams.

It ended with 5T tokens and released 600B of them. That 12% is a release choice, not what the filter kept; the size of the pool before filtering is not given.

**FineWeb** (Hugging Face) began as a replication of RefinedWeb and improved on it: 95 Common Crawl dumps, URL filtering, language ID keeping pages with $p(\text{en}) > 0.65$ (far looser than C4's 0.99), Gopher rules plus C4 rules plus more manual rules, MinHash deduplication, and anonymization of email and public IP addresses (personally identifiable information, PII). The result: 15T tokens. Dataset sizes were "growing quite a bit".

::code lecture_13.py:L505-L520 | RefinedWeb: trafilatura, Gopher rules, MinHash, 600B of 5T released; FineWeb: 95 dumps, more rules, PII removal, 15T

### Dolma

AI2's **Dolma** (2024) is a mixture again, with its own Common Crawl processing:

::figure https://miro.medium.com/v2/resize:fit:1400/1*-0Qqhvu7JD6Y9JgsfKJdxw.png | Dolma's sources in bytes, documents and Llama tokens: Common Crawl 2,281B of the 3,059B total, then The Stack (411B), C4 (198B), Reddit, peS2o, Gutenberg and Wikipedia

- **Reddit** from the Pushshift project (2005-2023), submissions and comments kept separately; you could get it "before things got locked down";
- **peS2o**: 40M academic papers from AI2's Semantic Scholar;
- C4, Project Gutenberg, Wikipedia and Wikibooks, and The Stack (code, below).

Its Common Crawl processing: language ID with a fastText classifier (a model, but only for language), quality filtering with Gopher and C4 rules (still avoiding model-based quality filtering), toxicity filtering with rules plus the Jigsaw classifier, and deduplication with Bloom filters. The result: 3T tokens.

::code lecture_13.py:L527-L537 | Pushshift Reddit, peS2o; language ID, rules, toxicity, Bloom-filter dedup; 3T tokens

Every pipeline so far deduplicates, at different granularities: CCNet by paragraph, GPT-3 fuzzily by document, RefinedWeb and FineWeb with MinHash over n-grams, SlimPajama with MinHashLSH, Dolma with Bloom filters. The lecture names these tools without explaining them; how MinHash and locality-sensitive hashing find near-duplicates is worked through in [L14](#/t/lecture_14).

::note deferred | MinHash, LSH, Jaccard similarity and Bloom filters appear on screen but were never explained (nor said aloud); the next lecture and Assignment 4 cover them.
::kp multi-source-mixtures

## How hard should you filter? DCLM and Nemotron-CC {#classifiers}
source: lecture_13.py:L539-L575 · video 1:04:37-1:10:55

### DCLM: model-based filtering becomes the norm

DataComp-LM (DCLM, 2024) is where model-based quality filtering "started to really become the norm". Its stated goal was a standard benchmark for data processing: fix the pool and the training recipe, and let people compare filtering methods. In practice, most people use the dataset it released.

DCLM processed Common Crawl into **DCLM-pool**, completely unfiltered: 240 trillion tokens, "probably more than the number of tokens that anyone really trains on", much of it low quality. Then it filtered the pool down to **DCLM-baseline**.

::figure official/lectures/images/dclm-filter.png | follow the pool left to right: the English filter alone removes about half the documents (50.8%); all the heuristic cleaning together (a RefinedWeb reproduction) leaves 19.9%, Bloom-filter dedup leaves 13.7%, and the fastText classifier keeps the final 1.4% of the original documents

The classifier is the interesting part. It is a fastText model, "just ... a linear classifier", trained on:
- **positives** (200K): [OpenHermes-2.5](https://huggingface.co/datasets/teknium/OpenHermes-2.5), mostly GPT-4-generated instruction data, and [ELI5](https://www.reddit.com/r/explainlikeimfive/), a subreddit of curious questions and answers;
- **negatives** (200K): RefinedWeb, "basically the web".

Then it scored all of DCLM-pool and kept the top. The result: 3.8T tokens. That positive set is odd for a pre-training filter, instruction data and Q&A rather than encyclopaedic prose, "kind of weird, but somehow this works":

::code lecture_13.py:L546-L554 | 200K positives (OpenHermes-2.5, ELI5), 200K negatives (RefinedWeb), fastText over the whole pool
::figure official/lectures/images/dclm-quality.png | the same pipeline with different quality filters, scored on CORE: PageRank, SemDedup, BGE features, AskLLM and perplexity filtering all trail the fastText OH-2.5 + ELI5 classifier (30.2)

So the reference set *is* the definition of quality. Swap OpenHermes and ELI5 for Wikipedia and the same recipe keeps different pages. DCLM became, for a while, "a bit of a gold standard" for quality filtering in the open community, and OLMo 2's pre-training at the start of the lecture is 95% DCLM-baseline.

::kp reference-set-quality-proxy

The kept fraction is $3.8\text{T} / 240\text{T} \approx 1.6\%$ of tokens. Compare C4's rules, which kept 11%.

::widget fixture:filter-retention-ratios | guess DCLM's kept fraction before revealing: it is about 1.6%, against C4's 11%, yet the absolute sizes keep growing because the pools grew faster
::widget fixture:lecture_13--funnel | click "DCLM-baseline (classifier)": 240T → 3.8T, 1.6% kept, with English, rules, dedup and the classifier listed as unsized stages
::note aside 1:05:45 | Aloud, DCLM keeps "something that's 1.4%". That is the figure's number, a share of *documents*; 3.8T of 240T is a share of *tokens*, 1.6%. Both are right in their own units (kept documents are presumably longer than average).

### Nemotron-CC: filtering too hard

NVIDIA's **Nemotron-CC** (2024) pushed back: FineWebEdu and DCLM "filter too aggressively (remove 90% of data)". We "need moar tokens", while preserving quality. Its changes:

- **extraction**: jusText instead of trafilatura, because it returned more tokens;
- **classifier ensembling**: prompt Nemotron-340B-instruct to score FineWeb documents for educational value and distill those labels into a fast fastText model; combine it with the DCLM classifier;
- **synthetic rephrasing**: for documents the classifiers rate low, have a language model rewrite them (aloud, to "look like Wikipedia") instead of discarding them; for documents rated high, have it generate tasks from them, such as question-answer pairs, summaries and extracted key information.

It is one of the main datasets to lean into synthetic data for pre-training ("probably not the first", he allowed). The result: 6.3T tokens, with a high-quality subset of 1.1T.

::code lecture_13.py:L561-L574 | too aggressive; jusText for more tokens; two classifiers; rephrase low quality, generate tasks from high quality; 6.3T
::figure official/lectures/images/nemotron-results.png | average over ten benchmarks: FineWebEdu 53.2, DCLM 57.0, Nemotron-CC 57.8, and its high-quality subset 60.1

::note aside | "Remove 90%" is Nemotron-CC's characterization; by the lecture's own numbers DCLM keeps 1.6% of its pool, i.e. removes over 98%. The two are measured from different starting points, and the lecture does not reconcile them.

Why push for more tokens? For reference, Llama 3 trained on 15T tokens and Qwen3 on 36T. A 3.8T dataset cannot fill such a budget without repetition. [L9's data-repetition result](#/read/lecture_09) says how much that costs: up to about four epochs, repeated tokens are worth nearly as much as fresh ones; beyond that, much less.

::predict filter-aggressiveness-vs-tokens
::widget fixture:lecture_13--filter-budget | click "DCLM 1.6%" with the 240T pool: a 15T budget needs 3.95 epochs, just inside the four-epoch line; set the budget to Qwen3's 36T and it needs 9.5, so the filter must loosen
::kp lecture_09:data-repetition-effective-data

### Reported token counts include repeats

The professor added a caution aloud that the lecture text does not have. It is "not clear how big these unique tokens are", because training-token counts in papers include repeats: "if you do two epochs, that's twice the number of tokens". A reported count is $\sum_i e_i U_i$ over the mixture's components, $U_i$ unique tokens seen $e_i$ times, while the dataset's size is $\sum_i U_i$.

So Nemotron-CC's 6.3T and Llama 3's 15T are not directly comparable: 6.3T seen $15 / 6.3 \approx 2.4$ times is reported as 15T. The Pile's Epochs column above is the same bookkeeping inside one dataset. Compute estimates go the other way: in [L2's $6ND$](#/read/lecture_02), $D$ counts tokens *processed*, so it is the reported count, repeats included.

::predict reported-tokens-count-repeats
::widget fixture:lecture_13--repeat-ledger | each extra epoch lengthens the reported bar and leaves the unique bar unchanged; set component B to zero, then click "Nemotron-CC 6.3T" and "Llama 3: 15T": the run needs 2.4 epochs
::video 1:09:30-1:09:56 | "if you do two epochs, that's twice the number of tokens, so you have to be careful"
::kp reported-tokens-count-repeats

### The pattern across all of them

"All of these look very similar at some level": take a crawl, filter by rules or by a model; if a model, decide what good data looks like, train a classifier, score everything, select. And there is a trade-off: you can take 240 trillion tokens of mostly low-quality text, or about one trillion of high quality, "and there's some sweet spot in-between".

| dataset | year | text from | quality filter | dedup | size |
|---|---|---|---|---|---|
| WebText | 2019 | Reddit-linked pages | ≥ 3 karma | | 40 GB |
| C4 | 2019 | one CC snapshot | rules | | 156B tokens |
| GPT-3 | 2020 | CC + curated | classifier (WebText, Wikipedia, books) | fuzzy | 400B tokens |
| The Pile | 2021 | 22 domains | curation | | 825 GB |
| MassiveWeb (Gopher) | 2021 | web | rules, SafeSearch | yes | 10.5 TB |
| RefinedWeb | 2023 | CC WARC, trafilatura | Gopher rules | MinHash | 600B released |
| FineWeb | 2024 | 95 CC dumps | rules | MinHash | 15T tokens |
| Dolma | 2024 | CC + curated | rules | Bloom filter | 3T tokens |
| DCLM-baseline | 2024 | CC, resiliparse | fastText classifier | Bloom filter | 3.8T tokens |
| Nemotron-CC | 2024 | CC, jusText | two classifiers + rephrasing | | 6.3T tokens |

::video 1:10:35-1:10:56 | a lot of low-quality tokens or few high-quality ones, and a sweet spot in between
::note slip 1:09:08 | Nemotron-CC was rounded aloud to "6 trillion tokens"; the text says 6.3T.
::kp filter-aggressiveness-vs-tokens

## What does a code dataset need? The Stack {#code}
source: lecture_13.py:L578-L597 · video 1:10:55-1:14:44

By 2022 it was clear that code would matter a lot, and **The Stack** (BigCode) set out to build a really good code dataset:
- took repository names from the GitHub Archive (2015-2022);
- git-cloned 137M repositories, containing 51B files, of which only **5B were unique**;
- kept only permissively licensed repositories (MIT, Apache), detected with go-license-detector;
- removed near-duplicates with minhash and Jaccard similarity;
- result: 3.1 TB of code.

::code lecture_13.py:L579-L584 | 137M repositories, 51B files, 5B unique; permissive licences; minhash near-dedup; 3.1 TB

$5 / 51 \approx 9.8\%$: about nine of every ten files on GitHub are a copy of another file (forks, vendored libraries, copied code). Skipping deduplication would mean training mostly on repeats.

::predict fuzzy-deduplication-methods
::widget fixture:lecture_13--funnel | click "The Stack (dedup)": only the files → unique files step is a fraction (51B → 5B, keeps 9.8%); repositories, files and bytes are different units
::note slip 1:11:19 | Aloud, "they clone 137 repos", dropping the M (137M repositories), and "three terabytes" for 3.1 TB.
::kp fuzzy-deduplication-methods

**Stack v2** (2024) widened the sources and the processing:
- **sources**: issues, comments and pull requests from the GitHub Archive; repositories from Software Heritage; documentation crawled from sites such as PyPI, npm and devdocs.io;
- **processing**: remove binary files, malware and bot activity (many PRs are bots), deduplicate, redact PII, subsample PRs to keep the set representative and manageable;
- **low-resource languages**: compile code in rare languages (the example is Nim, which the professor "hadn't even heard of") to LLVM, the low-level intermediate representation that many compilers share, and train on the pair. The model can then learn the mapping between the rare language and a representation it has seen a lot of;
- **existing datasets**: GSM8K, code contests, StackOverflow, arXiv, Wikipedia, OpenWebMath. This list is the only place math data enters the lecture.

::code lecture_13.py:L586-L592 | Software Heritage and GitHub Archive; cleaning; LLVM pairing for low-resource languages; extra datasets

A pull request is a structured object (a description, files, diffs, comments, review states), not a sequence, so it must be **linearized** into tokens. The key decision is how much context to include: a PR event may change one line, and learning from it needs some surrounding lines, or the whole file around the diff.

::figure official/lectures/images/stackv2-pr1.png | a PR as tokens: title, status and repo name, then the base files, then the diff hunks per file, each field in an XML-like tag
::figure official/lectures/images/stackv2-pr2.png | the comment and review events that follow: who commented, the review state (approved, rejected, commented, changes required), and review comments anchored to a file and line

The payoff is that the model learns "not just how to generate code, but also the software development process around code".

::predict code-data-licensing-pipeline
::video 1:12:46-1:13:17 | pairing a low-resource language with LLVM IR to borrow data
::kp code-data-licensing-pipeline

## Can you train only on permissively licensed data? Common Pile {#common-pile}
source: lecture_13.py:L600-L618 · video 1:14:44-1:19:29

Recall: almost all data on the Internet is copyrighted, some of it is permissively licensed or public domain, and fair use for training is not settled. If you are very risk-averse, the rule is "if I don't know whether it is OK, that's a no". **Common Pile** took that attitude and asked: can you train a good model using only permissively licensed data, or rather, how far can you get?

The project scoured the Internet for permissively licensed data worth training on and collected 8 TB, "actually pretty good" for that constraint.

::figure official/lectures/images/commonpile.png | sizes on a log scale, by category: code (Stack v2) is largest at 4775 GB, then government and legal text (1172 GB, e.g. USPTO patents), wikis, web, academic papers, forums, public-domain books and educational resources

Government proceedings turn out to be a large permissive source. Assembling the set is harder than reading licence labels, because of three subtleties:
- **Licence laundering.** People are sloppy: anyone can take a copyrighted work and slap a CC-BY label on it, and it is hard to tell whether the label is real.
- **Collection licences.** Dolma is distributed under ODC-By, a licence on the *collection*; it does not extend to the individual documents inside, which keep their own status. Many datasets on Hugging Face show a permissive licence that, dug into, does not hold at the level of individual works.
- **Synthetic data.** Text generated by a language model trained on unlicensed data has unclear status. Common Pile skipped synthetic data entirely. It is "probably fine", since the open-weight models carry licences such as MIT, but they "were presumably also trained on unlicensed data", so "it's a little bit of data laundering if you are really honest".

::code lecture_13.py:L612-L615 | licence laundering, collection licences, synthetic data
::predict permissive-only-common-pile

The test was **Comma**, a model trained on Common Pile, compared with older models:

::figure official/lectures/images/comma-results.png | Comma v0.1-1T (gold) against LLaMA, MPT, RPJ-INCITE and Qwen3 (hatched): level with or ahead of the 2023 models on several benchmarks (stars), and clearly behind Qwen3, most of all on coding

His reading: "not as good as the Qwen models for sure", but better than the old 2023 baselines such as the first LLaMA and MPT. You can do decently, but it is "tough to compete without more tokens". He does not think it is the final word: "you can probably eke more out" of permissive data.

::video 1:17:17-1:17:53 | licence laundering, collection licences, and synthetic-data laundering
::kp permissive-only-common-pile

## What should you carry away? {#summary}
source: lecture_13.py:L40-L45 · video 1:19:29-1:21:51

- **Data does not fall from the sky.** You do not just download a dataset from Hugging Face; someone has to do the work. The Internet is live services; someone (a site with dumps, or a crawler) produces raw data; someone decides how to process it.
- **The pipeline**: live service → raw data → processed data (transformation, filtering, deduplication), and each step changes the final model. HTML extraction alone moved DCLM's accuracy by about four points.
- **Filtering is probably the most important step.** Going from 240 trillion tokens of pool to a few trillion kept "merits a lot of attention". Quality is defined by a reference set or by rules, and filtering harder trades tokens for quality.
- **Data differentiates models.** Most language models share roughly the same Transformer architecture; how the data was processed "can make a pretty big difference".
- **Legal and ethical issues** (copyright, privacy, and more than one lecture can cover): licences, fair use, terms of service, piracy.
- **Much of the pipeline is heuristic**, "a lot just based on vibes": you define a classifier, a rule, a threshold. There are many opportunities to improve, starting with Assignment 4.

::code lecture_13.py:L40-L45 | the lecture's own summary
::note slip 1:20:31 | Aloud: "from 200 trillion tokens to less than three trillion tokens". DCLM's own numbers are 240T to 3.8T.
::note deferred 1:21:47 | Next lecture continues with data: more on filtering (including how deduplication works) and post-training data. See the [L14 thread](#/t/lecture_14).
