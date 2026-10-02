---
title: L17 · Multimodal models, read through
minutes: 45
---
Everything in this course so far has been text in, text out. This lecture is about letting a language model see: how an image becomes something a Transformer can read, how the open vision-language models (LLaVA and Qwen) bolt an image encoder onto a pretrained language model, and how one model (Chameleon) tries to both read and draw images with a single vocabulary. After it you can count how many tokens an image costs under each scheme, say what CLIP-style encoders capture and what they lose, and explain why understanding an image and generating one are different problems.

## Why does multimodality come down to tokens? {#tokens}
source: lecture_17.py:L5-L24 · video 0:05-4:17

A language model already goes from any text to any text: any language, code, poetry, even DNA. But the world is not only text. It has images, audio and video, and the "North Star" the professor names is an **omni model**: give it any combination of modalities and get back any combination. Show it an image and a video and ask about both; ask it to draw; turn audio into a picture.

::figure official/lectures/images/multimodality.png | the four modalities the lecture names: text, images, audio, video. Only the first two get a mechanism today

The lecture does not build an omni model. It gives the pieces for one direction, getting images *in*, and the argument for how to do it fits in three lines.

1. **Transformers work really well.** "Despite the best efforts of people to try other things," across every modality they are still the best tool we have at scale. So we use them.
2. **Transformers speak tokens.** They take a sequence of tokens in and give tokens out. Here "token" is widened beyond the discrete ids of text: a token can also be **continuous**, an embedding vector that never passes through a vocabulary. What matters is that each token stands for roughly one *semantic unit* of information. A text subword is "somewhat meaningful"; a single pixel "is certainly not meaningful by itself".
3. **So everything must become tokens**, discrete or continuous.

::code lecture_17.py:L15-L20 | the whole design argument: Transformers work, Transformers speak tokens, so convert everything into tokens

Text needed this step too. In [L1's tokenization](#/read/lecture_01) the answer was BPE: not perfect, "not the worst thing in the world". For images, audio and video there is no obvious equivalent, and finding one is the subject of the lecture. It splits into two questions that need different answers:
- **Input** (understanding): how do you turn an image or a clip of audio into tokens the Transformer can consume?
- **Output** (generation): how do you turn the Transformer's output back into an image or audio?

The lecture "will mostly focus on question number one". Generation comes back only at the end, with Chameleon, and the summary explains why the two questions really are different: understanding needs semantics, generation needs fine detail.

::predict everything-into-tokens
::code lecture_17.py:L22-L24 | input and output of non-text data are posed as two separate questions
::note aside 0:05 | The lecture replaced a planned second lecture on reinforcement learning, because the course "would be a little bit incomplete" without multimodality, "so pervasive if you look at all the major models". The professor adds that this "could be a whole class in itself".
::video 2:21-3:04 | tokens can be continuous embeddings, but each should be a semantic unit; a pixel is not
::kp everything-into-tokens

## How can web captions train an image encoder? CLIP {#clip-objective}
source: lecture_17.py:L48-L61 · video 4:17-8:15

The story of modern vision-language models starts with CLIP (Contrastive Language-Image Pretraining, OpenAI, 2021). The context: GPT-2 and GPT-3 had already shown that you can scrape noisy text off the internet, train a large model on it, and get something useful. Computer vision was still built on annotated datasets such as ImageNet: humans label each image with a class, a ResNet is trained on the labels. The CLIP authors asked whether the far larger supply of (image, caption) pairs on the web could play the role for images that raw web text played for language.

### The objective

Take a batch of $N$ (image, text) pairs; the lecture's example is $N = 32768$.
1. Encode each image with an image encoder, giving vectors $I_1 \ldots I_N$.
2. Encode each text with a text encoder, giving $T_1 \ldots T_N$.
3. For each image, prefer its own caption over the other captions in the batch: the dot product $I_1 \cdot T_1$ should be much larger than $I_1 \cdot T_j$ for every other $j$.
4. Conversely, for each text, prefer its own image over the other images.

::figure official/lectures/images/clip.png | panel (1): the N×N grid of dot products between every image and every caption; the blue diagonal holds the matched pairs, everything off it is a negative. Panels (2) and (3) are zero-shot classification, used in the next sections

::code lecture_17.py:L56-L60 | four lines: encode both sides, then rank the aligned partner first in both directions

In the paper's pseudocode, both embeddings are projected to a shared width and L2-normalised, so every entry of the grid is a cosine similarity, multiplied by a learned temperature scale. Then a cross-entropy runs along every row (each image picks its caption among N) and along every column (each caption picks its image among N), and the two are averaged.

::figure official/lectures/images/clip-code.png | follow the last four lines: labels = arange(n) puts the right answer on the diagonal, one cross-entropy over axis 0, one over axis 1, averaged

Written out for one row, with $s_{ij}$ the cosine between image $i$ and text $j$ and $\tau$ the temperature:

$$ \ell_i^{\text{img}} = -\log \frac{e^{s_{ii}/\tau}}{\sum_{j=1}^{N} e^{s_{ij}/\tau}}, \qquad \mathcal{L} = \frac{1}{2N}\sum_{i=1}^{N}\left(\ell_i^{\text{img}} + \ell_i^{\text{txt}}\right) $$

So the objective is "2 times N different softmax classification problems", a multiclass classification with the examples laid out as an N × N matrix. The professor named the temperature aloud; the learned projections to the shared space are only in the pseudocode.

::predict clip-contrastive-objective
::widget fixture:lecture_17--contrastive-matrix | each row is one softmax over the N captions, so image 1 is ranked against N − 1 wrong captions; keep the cosines fixed and raise N, and the probability CLIP gives the same matched pair falls, because every extra caption adds to the denominator

### A worked example: the batch is the set of negatives

Fix the similarities: every matched pair has cosine 0.6, every unmatched pair 0.1, and the scale is the paper's starting value $1/\tau = 1/0.07 \approx 14.3$. The matched caption's probability in one row is

$$ p = \frac{1}{1 + (N-1)\,e^{-14.3 \times (0.6 - 0.1)}} = \frac{1}{1 + (N-1)\times 0.00079}. $$

- With $N = 8$: $p = 1/(1 + 7 \times 0.00079) = 0.9945$. The task is nearly solved; the gradient is tiny.
- With $N = 32768$: $p = 1/(1 + 32767 \times 0.00079) = 1/26.9 = 0.037$. The same pair, with the same embeddings, is now far from solved.

The embeddings did not change; only the number of competitors did. That is what "requires large batch sizes" means: with a batch of 8 each image has only 7 easy negatives, and "if you have a batch size of 1, clearly, it doesn't work, or even 10, it doesn't work". The batch is not a throughput knob, as it is for language modelling; it is part of the loss.

::note spoken 19:08 | A student asked whether another dog caption in the same batch confuses the model, since it is counted as a wrong answer. Such false negatives happen, and "in general, this process is going to be noisy", but it is tolerated because "on average, it's unlikely that there's always going to be a dog".
::video 6:39-7:30 | each image must outscore every other caption and each caption every other image: 2N softmax classifications
::kp clip-contrastive-objective

## What does CLIP train on, and what does its encoder see? {#clip-data-encoder}
source: lecture_17.py:L63-L82 · video 8:15-17:18

### The data

The paper gives "not too many details" about its data. Roughly: take about 500K search queries, search the web for each, and collect (image, text) pairs, keeping up to about 20K per query. The result was **400M** image-text pairs.

The two numbers do not multiply out: 500K × 20K would be 10 billion. The paper's own wording resolves it: up to 20,000 pairs were kept *per query* to balance the classes, so ~20K is a cap, not an average.

OpenAI did not release the dataset. **OpenCLIP** reproduced CLIP in the open, trained on **LAION-5B**, 5 billion images with text, so here at least "you can point to the data set that it was trained on and the code as well". One twist: LAION was filtered using CLIP itself, "so there's some bootstrapping happening".

::code lecture_17.py:L63-L67 | 500K queries, 400M pairs, not released; OpenCLIP on LAION-5B, which used CLIP for filtering

### Preprocessing: every image becomes 336 × 336

Web images come in every shape, long and skinny or tall. Neural networks "don't like things to be dynamic. They want things to be fixed size." So CLIP forces a fixed input:
1. resize with bicubic interpolation so that the **shorter side is 336** pixels (224 is the other common target);
2. **center-crop** the longer side to get a 336 × 336 square, cutting off the borders.

This was "obviously for convenience", and it fitted what the authors had in mind, ImageNet classification, where "usually, the object is in the middle and you're just trimming off some background". For a scanned document it throws away the edges of every page. "Later, we'll see that you can do better than this": that is AnyRes, below.

::code lecture_17.py:L69-L72 | resize the shorter side to 336, center crop to 336 × 336

### The vision encoder: a ViT, one token per patch

The authors tried ResNet-50s and Vision Transformers, which had just come out; the ViT won, and "when people say CLIP, they usually mean the ViT version".

A **Vision Transformer** (ViT) cuts the image into small square patches, flattens each patch into a vector, projects it linearly to the model width, adds a position embedding "just like you would if you were training a language model", and runs a standard Transformer encoder. Each patch is one token. This is the answer to the opening worry that a pixel is not a semantic unit: a patch is a bigger unit that the Transformer can work with.

::figure official/lectures/images/vit.png | the bottom row: the image is cut into a grid of patches, each flattened and linearly projected; positions 1–9 are added; the starred extra token is ViT's class token, which CLIP replaces with attention pooling

The Transformer outputs one vector per patch. To get one vector for the whole image you could average them; CLIP instead uses **attention pooling**: take the global average as a *query* and run one more attention step against the keys and values of every patch, giving a vector that is "maybe a little bit more informed" than a straight average.

The best model is **ViT-L/14@336px**: L for large (about 24 layers, which the professor offered tentatively and which is right), 14 × 14-pixel patches of 3 colour channels, trained at 336 × 336. Most of training ran at a lower resolution for speed, with 336 used "for the latter part of training".

### Worked example: how many tokens is one image?

The lecture never states the count; it follows from its numbers.
- Patches per side: $336 / 14 = 24$. Patches per image: $24^2 = 576$.
- Raw values per patch: $14 \times 14 \times 3 = 588$.

And because of the crop, **576 for every image**. A 4000 × 3000 photo is resized so its shorter side is 336 (to 448 × 336) and cropped to 336 × 336: still 576 patches. The extra pixels are discarded, not tokenised. Note also what "/14" means: the patch size in pixels, not the number of patches per side. Shrink the patches and the count rises with the square: 8-pixel patches on the same crop give $(336/8)^2 = 1764$.

::predict clip-preprocess-vit-patches
::widget fixture:lecture_17--image-tokens | in "CLIP crop" mode the count stays at 576 whatever width and height you set; only the patch size and the crop side change it (8-px patches give 1764, a 224 crop with 16-px patches gives 196)
::code lecture_17.py:L74-L78 | ResNet-50 vs ViT; attention pooling; ViT-L/14@336px

### The text encoder

A GPT-2-style Transformer, "since that was from the same group that developed GPT-2": 63M parameters and 12 layers. To get one vector from a sentence, wrap it in [BOS] … [EOS] and take the [EOS] token's activation at the top layer. Because the model is causal, [EOS] is the only position that has attended to the whole caption.

::code lecture_17.py:L80-L82 | 63M-parameter GPT-2 Transformer; the top-layer [EOS] activation is the text embedding
::note slip 15:51 | Asked whether 2D position embeddings would suit patches better than positions 0 through 9, the professor recalled that "in the CLIP paper, I believe they tried some 2D version" and that 1D "doesn't really matter that much", adding "for classification, maybe it doesn't matter". The 1D-versus-2D comparison this atlas knows of is in the ViT paper's appendix, so the attribution is probably a slip. Spatially aware positions come back with MRoPE.

### Why captions, and not just images?

A student asked why CLIP trains on image-text pairs at all, rather than on images alone. There is a line of work, SimCLR for example, that learns image representations from images only: take one image, augment it (crop it, perturb it, rotate it), and train the encoder to give the augmented views similar embeddings while pushing other images away. That teaches invariances that are "useful for low level details".

But its positives are only views of the *same* image. Nothing in the objective says that two different photos show the same kind of thing; in fact, two different dog photos are negatives and get pushed apart. "You won't data augment your way from one type of dog to another dog." A caption supplies exactly that link: two different photos both described as a dog are tied together through the text, which "gives you a higher level semantic representations of images".

The argument cuts both ways. If what you need is *instance* identity, say finding re-uploads of one photo after cropping and recompression, while different photos of similar scenes must not match, then augmentation is the right signal and captions would wrongly pull similar scenes together.

::predict text-pairs-vs-augmentation-ssl
::video 11:43-12:23 | "you won't data augment your way from one type of dog to another dog"
::kp clip-preprocess-vit-patches
::kp text-pairs-vs-augmentation-ssl

## What is CLIP good at, and what does it miss? {#clip-result}
source: lecture_17.py:L84-L95 · video 17:18-22:37

### Zero-shot classification

The result that "got a lot of people excited" in 2021: on ImageNet, **zero-shot** CLIP beat a ResNet-50 trained on the 1.2M labelled ImageNet images. Those labels cost "many, many hours of Amazon Mechanical Turk worker time"; CLIP used web data that already existed.

Zero-shot means no training on ImageNet at all. Panels (2) and (3) of the CLIP figure above show how: write each class name into a sentence ("A photo of a {object}."), encode all of them with the text encoder, encode the image, and pick the class whose text has the highest dot product with the image. Classification becomes the same ranking CLIP was trained on, with the class names as the candidate captions.

::code lecture_17.py:L84-L85 | zero-shot CLIP beat a ResNet-50 trained on 1.2M labelled ImageNet images

### The ablation: ranking beats predicting the caption

The obvious alternative to ranking is to make the image encoder *predict* its caption, with a language model or as a bag of words. The paper tried it.

::figure official/lectures/images/clip-efficiency.png | zero-shot ImageNet accuracy against images processed: the Transformer language model (blue) is the slowest, bag-of-words prediction (orange) is 3× more efficient, and the contrastive objective (green) 4× more efficient again

The surprise is the order: the stronger text model does *worse*, or at least is less efficient. For a representation judged by classification, "modeling the exact token sequences of the caption isn't so important for getting the rough representation of the image". Predicting a caption word by word spends capacity on phrasing, which the ranking never asks for. The professor flagged this as a first sign of an asymmetry he returns to in the summary: understanding and generating need different things.

::code lecture_17.py:L87-L89 | predicting text from images is much less compute efficient than CLIP-style ranking

### What the representation is, and isn't

The section's summary has three lines.
1. **The encoding captures semantics given by (noisy) text.** "Generally, text talks about the semantics." The captions are noisy because alt text rarely states the obvious: "if you have an image and it's a dog, you don't need to say a dog". A lot of data filtering was needed, and it is "maybe somewhat surprising" that it works at all.
2. **The design decisions were chosen for image classification**, so the representation is "not very fine-grained". The objective only has to pick the right caption out of a batch, so it learns whatever distinguishes captions and nothing finer; and the 336 crop has already thrown away detail. The lecture's own example, in the OneVision section, is OCR, where "a J looks like an I".
3. **Technically**, it needs large batches and a softmax over the whole batch, which is "not really very decomposable". In language-model training the sequences in a batch are independent until the loss is averaged at the end; here every image's loss depends on every caption in the batch.

Even so, CLIP is "a robust starting point for everything we're going to do later": every vision-language model in the rest of the lecture, except Chameleon, starts from a CLIP-style encoder.

::predict clip-semantics-vs-detail
::code lecture_17.py:L92-L95 | semantics from noisy text; tuned for classification, not fine-grained; large batches and a full-batch softmax
::video 21:29-22:01 | the semantics come from the paired text, and the design was tuned for classification, so it is not fine-grained
::kp clip-semantics-vs-detail
::kp clip-contrastive-objective

## Can you drop the batch-wide softmax? SigLIP {#siglip}
source: lecture_17.py:L98-L121 · video 22:37-28:37

SigLIP (Sigmoid Loss for Language Image Pre-training, Google, 2023) is "basically an improved version of CLIP" that removes the full-batch softmax.

### One question per pair

- **CLIP** asks, for each image, *which* of the N captions is its own: a multiclass classification over the batch.
- **SigLIP** asks, for each (image, text) pair separately, *are these two aligned or not?*: a binary classification.

Same N × N grid of similarities as before. The diagonal cells get label +1, every off-diagonal cell gets −1, and each cell is scored on its own by a log-sigmoid. With $z_{ij} = +1$ if $i = j$ and $-1$ otherwise, a learned scale $t$ and bias $b$:

$$ \mathcal{L} = -\frac{1}{N}\sum_{i=1}^{N}\sum_{j=1}^{N} \log \sigma\!\big(z_{ij}\,(t\, x_i \cdot y_j + b)\big) $$

::figure official/lectures/images/siglip-code.png | line 10 builds the labels: 2·eye(n) − ones(n) is +1 on the diagonal and −1 everywhere else; line 11 is one log-sigmoid per cell, summed, with no normalisation across the row

No term normalises over the batch, so a cell's loss does not depend on how many other captions are present. The negatives are still all there: the N − 1 off-diagonal cells in each row are each pushed toward "not aligned". Asked whether this needs clever negative sampling, the professor said no: "at least in the initial paper, they were just operating on literally the same type of matrix".

::code lecture_17.py:L101-L103 | CLIP: multiclass over the batch; SigLIP: binary, aligned or not

Rerun the worked example from CLIP with SigLIP's starting values $t = 10$, $b = -10$ and the same cosines. The matched pair's logit is $10 \times 0.6 - 10 = -4$, so its probability is $\sigma(-4) = 0.018$, and that number is the same at $N = 8$ and at $N = 32768$. Training raises it by moving the embeddings, $t$ and $b$, not by changing the batch.

::widget fixture:lecture_17--contrastive-matrix | switch to SigLIP and move the batch size: the dashed SigLIP line stays flat, because each cell is its own yes/no, while CLIP's solid line moves with N; the 16K and 32K markers are the paper's measured crossover and saturation points, not computed by the curves

### Data

SigLIP trained on **WebLI**, from Google's PaLI work: on the order of a billion (image, text) pairs scraped from the web. Images containing text were run through automatic OCR, a second source of captions besides alt text; the top 10% by quality were kept; it covers 100 languages.

::code lecture_17.py:L106-L111 | WebLI: O(billion) pairs, OCR text, top 10% kept, 100 languages

### Efficiency

- CLIP: 10 days on 256 TPUv3.
- SigLIP: 5 days on 32 TPUv4, "much faster".

That is $256 \times 10 = 2560$ chip-days against $32 \times 5 = 160$, sixteen times fewer. The code adds "(lower FLOP/s than TPUv3)", which is true only of the whole allocation, not per chip. A TPUv4 chip is faster than a TPUv3 chip (about 275 against 123 peak bf16 TFLOP/s), but 32 of them deliver $32 \times 275 = 8.8$ PFLOP/s against $256 \times 123 = 31.5$ for CLIP's 256: about 72% less. Aloud: "at this scale, it's really actually not faster. Actually, 60% slower or something". So SigLIP finished in half the time on about a quarter of the peak compute.

Part of the gain may be plain engineering: "CLIP probably did not try to optimize the code for the maximum throughput". The structural part is that a sum of independent per-cell terms splits across devices easily.

::figure official/lectures/images/siglip-parallelism.png | three devices, four pairs each. Panel 2: each device scores its own 4 × 4 block, + on the diagonal, − elsewhere. Panels 3–4: the text chunks rotate to the next device, which scores another off-diagonal block of − cells; only the per-device sums are added at the end

Think of it like [data parallelism](#/read/lecture_07): each device holds a slice of the pairs. Unlike language modelling, the examples interact, since every image must be scored against every text. So each device first computes the losses on its local pairs, then receives the next device's texts (device 1 gets $T_5 \ldots T_8$, then $T_9 \ldots T_{12}$) and scores those negatives, "until you cover all the off-diagonal block entries". No device ever holds the whole N × N matrix, and no softmax has to wait for a full row.

::code lecture_17.py:L113-L116 | 10 days on 256 TPUv3 against 5 days on 32 TPUv4
::video 26:24-27:38 | the pairwise loss split across devices: local pairs first, then text chunks rotate

### Batch size

Because the loss is decoupled from the batch, the batch size becomes a training choice again rather than part of the objective. In CLIP, "if you change the batch size, it's a different loss function". The paper's findings:
- SigLIP beats CLIP **below a 16K batch**, where CLIP's softmax degrades.
- You *can* go up to a **1M** batch, but **32K is enough**: beyond it the gains saturate. The professor tied this to [L9's critical batch size](#/read/lecture_09): 32K "was essentially their critical batch size".

A larger batch still gives each image more negatives per step, so batch size is not irrelevant to SigLIP; it just stops being the thing that makes the loss work.

::predict siglip-sigmoid-decouples-batch
::code lecture_17.py:L118-L121 | decouple batch size from loss; better than CLIP below 16K; up to 1M, but 32K is enough
::note slip 25:36 | Reading the efficiency line aloud he said "CLIP was five days on 32v4s", meaning SigLIP. Two more slips in this stretch: "the diagonal are negative examples" (23:23) should be the off-diagonal, which he put right in the code walk at 23:44; and "for smaller batch size for CLIP, you have more variance but it's the same in expectation" (28:10) is about SigLIP, and is loose even then, since a larger batch brings more negatives.
::kp siglip-sigmoid-decouples-batch

## How do you connect an image encoder to a language model? LLaVA {#llava}
source: lecture_17.py:L124-L145 · video 28:37-35:25

CLIP and SigLIP map an image to vectors that carry its semantics. Now the vision-language models (VLMs), in two families, LLaVA and Qwen, which share one template. The idea: take an existing image encoder and an existing pretrained language model and stitch them together, "rather than training something from scratch". In the professor's framing this is "more of a flavor of mid-training or post-training" than pretraining.

LLaVA (Large Language and Vision Assistant, 2023) caught attention because GPT-4 had just shown visual reasoning, and LLaVA showed an open model could do some of it, with everything visible "under the hood".

### The template: encoder, projector, language model

- **Vision encoder**: CLIP ViT-L/14, the best CLIP.
- **Language model**: Vicuna, a LLaMA fine-tuned on ShareGPT conversations, i.e. chats people had with ChatGPT and posted online (fine-tuning on chats like this is [L15's SFT](#/read/lecture_15)).
- **Projector**: one matrix $W$.

The encoder's output vectors are "not really in the same space, so to speak, as the text". So each patch feature $Z_v$ is multiplied by $W$ to give a vector $H_v$ of the LM's embedding width. The text is embedded as usual. Then the image vectors and the text vectors are put in one sequence and run through the language model, which answers in text.

::figure official/lectures/images/llava-architecture.png | the image Xv goes through the vision encoder (Zv) and W to become the white tokens Hv; they sit in the same row as the grey text tokens Hq, and the LM reads both to produce the green response

Two things to see here. First, the image enters as **continuous** tokens, the second kind from the opening section: vectors that bypass the LM's vocabulary table. The professor's shorthand, "in some sense converting these images into textual tokens", means they are made to look like token embeddings, not that they are vocabulary ids. Second, $W$ is applied to each patch separately, so the LM gets one token per encoder patch: 576 for a 336 crop. $W$'s shape is (encoder width × LM width); the number of patches is not in it. Feed a tiled 672 × 672 input (2304 patches) and the LM's sequence grows fourfold, while $W$ has exactly as many parameters as before.

Simpler than the alternatives: the code names Flamingo and Q-Former as "more complex" connectors without describing them.

::predict vlm-template-encoder-projector-lm
::code lecture_17.py:L137-L139 | CLIP ViT-L/14, then a linear projection W into embedding space
::note spoken 32:30 | Aloud, the encoder output is called "a vector", and W turns it into "another vector". What LLaVA projects is the grid of patch features, one vector per patch; the single pooled CLIP vector of the previous sections is not used.

### The data: GPT-4 writes conversations it never saw

There was no large dataset of images paired with instruction-style conversations, so LLaVA synthesised one.
1. Start from **MS COCO**: images annotated by Mechanical Turk workers with captions and bounding boxes of the objects.
2. Prompt (text-only) **GPT-4** with an image's captions or its object boxes, *not the pixels*, and ask it to write one of three things: a conversation (questions and answers about the image), a detailed description, or a complex reasoning question with its answer.
3. Pair each generation with the original image.

The result is **158K** examples.

::figure official/lectures/images/llava-gen.png | the top half is all GPT-4 gets: five human captions and a list of boxes (person, backpack, suitcase…). The bottom half is what it writes from them: a conversation, a detailed description, a complex-reasoning answer

Notice what this implies. GPT-4 can be faithful only to what the annotation says. "What is the dog catching? A frisbee" is grounded if the caption mentions a frisbee; "what breed is the dog?" or "what colour is the frisbee?" would be invented, since COCO boxes record classes and positions, not colours. Training on invented details teaches the VLM to assert things it cannot see. The pairs still help, because most of what GPT-4 writes is about what the captions say, and that is in the image.

::predict synthetic-instruction-data
::code lecture_17.py:L130-L134 | COCO captions and boxes, GPT-4 writes questions and conversations, paired with the images: 158K examples

### Training in two stages

1. **Alignment**: freeze the vision encoder *and* the language model; train only $W$. With a random $W$ the projected image vectors "are not embeddings, representing any natural language token". This stage trains $W$ until they "look like natural language token embeddings", so the frozen LM can read them.
2. **Fine-tuning**: keep the vision encoder frozen; train $W$ and the language model together on the 158K conversations. All of it is image plus text in, text out.

The LM has to be pretrained already: aligning to a random LM "doesn't make sense". Answering a student, the professor added that the alignment stage runs for a fixed token budget, "not an adaptive threshold".

::code lecture_17.py:L142-L144 | stage 1: only W trains; stage 2: W and the LM train, the encoder stays frozen
::video 33:24-34:22 | train only W until image vectors look like token embeddings, then unfreeze the LM

The paper's showcase is a photo of a man ironing clothes on an ironing board attached to the back of a yellow taxi.

::figure official/lectures/images/llava-example.png | asked "What is unusual about this image?", LLaVA explains why ironing on a moving vehicle is odd; GPT-4 answers in one line; BLIP-2 and OpenFlamingo miss the point ("sitting on the back of a yellow cab", "drying his clothes")

LLaVA's point, as the professor read it: even when the prompt does not ask about the unusualness ("What's happening in the scene?"), it still mentions it. GPT-4 could do this; other open models at the time could not.

::kp vlm-template-encoder-projector-lm
::kp synthetic-instruction-data

## How do you keep the detail a 336 crop throws away? LLaVA-OneVision and AnyRes {#anyres}
source: lecture_17.py:L148-L170 · video 35:25-41:11

After LLaVA came LLaVA 1.5 and LLaVA-NeXT; the lecture skips to the latest, **LLaVA-OneVision** (2024), which folds in their innovations. The ambition grows: not one image, but multiple images and video, a video being a sequence of sampled frames.

The template is the same, with every part upgraded: "it's like you have a system and you're just upgrading the parts".
- **Vision encoder**: SigLIP, using its grid of patch features from both before and after the last Transformer layer.
- **Language model**: Qwen-2 72B, probably the best open LM at the time.
- **Projector**: a 2-layer MLP instead of one linear map.

::figure official/lectures/images/llava-onevision.png | the same diagram as LLaVA's with the parts relabelled (SigLIP, 2-layer MLP, Qwen-2); the visual signal can now be a single image, several images or a video

::code lecture_17.py:L153-L156 | SigLIP grid features, Qwen-2 72B, 2-layer MLP projector

### AnyRes: tile instead of shrink

OneVision wants OCR, and OCR needs fine detail: at low resolution "a J looks like an I". CLIP resizes and crops everything to 336 × 336, and "if you have a document and you crop to 336 by 336, you can't read it".

The fix is **AnyRes**, which the lecture attributes to LLaVA 1.5. The encoder cannot take a high-resolution image, so rather than shrinking the image to fit the encoder, cut the image into encoder-sized pieces:
1. break the image into an $a \times b$ grid of pieces, each at the encoder's native resolution;
2. encode each piece separately;
3. concatenate all the pieces' tokens;
4. if that gives too many tokens (a very high-resolution image), shrink the token grid with **bilinear interpolation**.

In the paper's figure there is one more path: the whole image is also resized down to one piece and encoded, so the model gets an overview next to the detailed tiles.

::figure official/lectures/images/llava-onevision-anyres.png | top path: the page is split into pieces, each encoded (coloured squares), the grid is shrunk by bilinear interpolation and flattened into the LLM; bottom path: the whole page resized to one piece and encoded as an overview

The interpolation reduces "the number of patches", so it acts on the encoded feature grid, not on the pixels. AnyRes is adaptive, which suits the Transformer: sentences can be any length, and "it turns out images can be any resolution" too.

::code lecture_17.py:L158-L163 | high resolution matters for OCR; tile into a × b encoder-sized pieces, encode, concatenate; interpolate if too many tokens

### Worked example: tokens grow with area

The lecture gives no AnyRes numbers. Take a 336-pixel encoder with 14-pixel patches, so each piece costs 576 tokens, and ignore the overview piece.
- A 672 × 672 image: $672/336 = 2$ pieces per side, a 2 × 2 grid, $4 \times 576 = 2304$ tokens.
- A 1344 × 1344 image: $1344/336 = 4$ per side, a 4 × 4 grid, $16 \times 576 = 9216$ tokens: sixteen times a plain CLIP crop, not four. Doubling the side quadruples the tokens.
- One pixel too many, 337 × 336, needs a second column of pieces: 1152 tokens.

Detail is no longer free. Every doubling of resolution costs four times the sequence length, which is why step 4 exists.

::predict anyres-tiling-token-budget
::widget fixture:lecture_17--image-tokens | in AnyRes mode the count jumps by a whole 576-token piece each time width or height passes a multiple of 336; 4 × 4 pieces give 16×, not 4×

### One budget for three kinds of input

Single images, multiple images and video are all "technically all reducible to images", but OneVision puts "their thumb on the scale" so that each kind of input costs roughly the same number of tokens. Videos can be long, and they do not want their data "dominated by a bunch of repetitive frames".
- **Single image**: high resolution, the overview plus up to 9 pieces. "I get to look at it more carefully."
- **Multiple images**: each at the encoder's base resolution, one piece each. "I'm just going to look at it from afar."
- **Video**: even fewer tokens per frame, up to 32 frames.

::figure official/lectures/images/llava-onevision-modalities.png | the right column is the budget: single image (1 + 9) × 729 = 7,290 tokens, multi-image 12 × 729 = 8,748, video 32 × 196 = 6,272. Three different shapes, roughly one total

Reading the figure, OneVision's encoder makes 729 tokens per piece, a 27 × 27 grid, rather than the 576 of a 336 crop with 14-pixel patches, and a video frame is cut to 196 tokens, a 14 × 14 grid, about a quarter of a piece. The principle is the one in the code: a shared total, split by how carefully each input deserves to be looked at.

With the 576-token pieces of the worked example: if one 3 × 3-tiled image ($9 \times 576 = 5184$ tokens) sets the budget, a 64-frame video gets $5184 / 64 = 81$ tokens per frame, a 9 × 9 grid, seven times fewer than one crop.

::widget fixture:lecture_17--image-tokens | pick a budget equal to one tiled image and add video seconds: the same total is split across the frames, and each frame gets far fewer tokens than one 576-token crop
::code lecture_17.py:L165-L170 | single image: higher resolution; multiple images: base resolution; video: lower resolution per frame
::note deferred 41:01 | "Later, we'll see how a big part of being able to handle multimodal is to deal with longer context": video is what pushes Qwen3-VL to a 256K context.
::video 39:02-40:37 | the downsampled overview plus encoder-sized crops, then why single images, multiple images and video get different resolutions
::kp anyres-tiling-token-budget

## Where does OneVision's skill come from? Data, stages and transfer {#onevision-data}
source: lecture_17.py:L172-L192 · video 41:11-45:56

### Data: quality over quantity, and very targeted

The stated philosophy is "quality over quantity". In practice the data is "very targeted": most of it is built around specific tasks, such as visual question answering, questions about tables, or spotting the difference between two images. "This is definitely post-training territory": you decide which tasks the model should do and make data for each.

::figure official/lectures/images/llava-onevision-data-1.png | the 3.2M single-image examples by category: General 36.1%, Doc/Chart/Screen 20.6%, Math/Reasoning 20.1%, Language 14.3%, General OCR 8.9%. LLaVA's own 158K set is one small slice of "General"

Much of it is generated by GPT-4 class models. The professor called this "unabashedly, basically, distilling GPT-4", which is "not ideal", but "what you do if you don't have an annotation budget". The data generation is the LLaVA recipe at scale.

::figure official/lectures/images/llava-onevision-data-2.png | the second table: the multi-image and video data, again a long list of task-specific sets

::code lecture_17.py:L172-L175 | the data philosophy, then the paper's two data tables

### Training: easier to harder

The philosophy here is "easier to harder", and the stages extend LLaVA's: first an alignment stage that trains only the projector, then the full model.

::figure official/lectures/images/llava-onevision-training.png | read the "Trainable" row: Stage-1 trains only the projector (72.0M parameters next to the 72.7B LLM), Stage-1.5 and Stage-2 train the full model (73.2B). The resolution row grows from one 384-pixel piece (729 tokens) to grids up to 6 × 6

Why three stages rather than two? "I'm not sure there's any particular principled reason for this." The middle stage (1.5 in the figure) uses high-quality data focused on knowledge; the last one uses examples that look like the downstream tasks.

The same table answers a student's question about sizes. The projector is tiny, 72M against a 72B language model, and the ViT is "generally less than a billion" parameters. The encoder does "a very local operation": it looks at small patches and holds little knowledge, so "most of the capabilities of the model are still in the language model".

::code lecture_17.py:L177-L179 | easier to harder, then the stage table
::note aside 1:04:39 | The answer on parameter sizes (is the vision encoder much smaller? "the answer is yes, in general") came in the Q&A after Qwen3-VL, read off this table; the reason, a local operation on small patches, is the professor's own gloss ("I guess that's the easy way to say it").

### Transfer between kinds of input

A finding the professor called interesting: skills learned on one kind of input carry over to kinds never trained for that skill.

- **Charts to multiple images.** The diagram and chart data is all single-image. "At training time, you never saw an example where you have a table and a chart." Yet at test time, given a diagram in one image and a table in another, the model answers a question that needs both.

::figure official/lectures/images/llava-onevision-transfer-s1.png | a sector diagram (radius 11, angle 40°) in one image and an insurance price table in another; the model computes the area, 38.01, reads Allstate's $63, and multiplies

- **OCR plus relations to GUI agents.** OCR data is single-image; relational reasoning ("what changed between these two images") is multi-image. Together they generalise to reading a sequence of phone screenshots and describing the taps between them, which is what a GUI agent needs.

::figure official/lectures/images/llava-onevision-transfer-s2.png | four App Store screens in a row; the model infers the three taps (search bar, the TikTok result, "Open") from what changed between screens

- **Visual prompting to video.** Data with a circle drawn on part of the image ("answer about this") exists only for single images. It generalises to a video where a player is highlighted across frames.

::figure official/lectures/images/llava-onevision-transfer-s8.png | four frames of a football match with one player marked by a shadow ellipse; asked to "describe the player highlighted in the video", the model follows the mark across frames

At first sight the dataset is per-task supervised learning, "you're basically targeting each of these tasks". "But if you have enough tasks, these models seem to do some transfer, which is, I guess, reassuring." Why it works is not stated. A plausible reading: all three kinds of input reach the LM in the same token format under one budget, so a skill learned on one has no format barrier to the others.

::predict cross-modal-transfer-onevision
::code lecture_17.py:L181-L187 | charts to multi-image; OCR plus relations to GUI agents; circles to video
::video 43:30-44:09 | transfer means a combination never seen in training (a table and a chart in two images) works at test time

### The template, stated

The OneVision summary names what every VLM so far has been: **vision encoder + projector + LM**. Most of the work goes into data curation, heavy on synthetic, task-specific data. And the LLaVA line is "one of the few works that open sources ... not just the model weights, but also the data", so it can be replicated and studied.

::code lecture_17.py:L189-L192 | the standard VLM template; most work is data curation; weights and data released
::kp cross-modal-transfer-onevision
::kp synthetic-instruction-data
::kp vlm-template-encoder-projector-lm

## What do the stages train, and why in that order? Qwen-VL {#qwen-vl}
source: lecture_17.py:L195-L211 · video 45:56-49:16

The Qwen team started training multimodal models in 2023 too. The professor went "through this quickly because you hopefully see the pattern".

### Architecture: a fixed 256 tokens

- **Vision encoder**: OpenCLIP's large ViT (14 × 14 patches); the file spells it ViT-bigC, OpenCLIP's name is ViT-bigG. "So it's basically a CLIP encoder."
- **Adaptor**: one layer of **cross-attention** with 2D positional encodings that maps the image to a **fixed 256 tokens**. It is "definitely not very dynamic, but neither is the vision encoder at this point."
- **Special tokens** `<img>`, `<box>` and `<ref>` mark an image, a bounding box and the text a box refers to, so the model can read and write box coordinates as text.

The adaptor is the other kind of projector. A fixed set of 256 learned queries attends over all the patch features, so whatever the number of patches, 256 vectors come out. LLaVA's $W$ passes the patch count through; a query-based adaptor fixes it.

::code lecture_17.py:L198-L201 | OpenCLIP ViT, one cross-attention layer to a fixed 256, special tokens for images and boxes
::note aside 46:58 | "I guess this got cut off": the special-token line was clipped in the lecture's HTML rendering, "what happens when your slides are in HTML". He read the tokens out anyway.

### Training: three stages, different parts frozen

::figure official/lectures/images/qwen-vl-stages.png | flames train, snowflakes are frozen. Stage 1: LM frozen, ViT and cross-attention train, low-resolution image-text pairs. Stage 2: everything trains, high resolution, multi-task data. Stage 3: ViT frozen, adaptor and LM train on chat data

1. **Stage 1**: large-scale, low-quality image-text data. Freeze the LM; train the vision encoder and the adaptor. They call it pre-training, but it is "not pre-training from scratch". Unlike LLaVA, the encoder trains here.
2. **Stage 2**: higher-quality, task-specific data (visual question answering, chart questions), at a **higher resolution**. Train all parameters.
3. **Stage 3**: instruction-tuning data. Freeze the vision encoder; train the adaptor and the LM.

::figure official/lectures/images/qwen-vl-stage1.png | the stage-1 data: 5B web image-text pairs (LAION, DataComp, Coyo and others) cleaned down to 1.4B, 28% kept; this is the "1.4 billion examples" read aloud
::figure official/lectures/images/qwen-vl-stage2.png | the stage-2 data, now tasks: captioning 19.7M, VQA 3.6M, several grounding sets, OCR 24.8M, and 7.8M of pure text, so the LM keeps practising language while it trains

### Reading all the schedules together

| model | stage 1 | stage 2 | stage 3 |
|---|---|---|---|
| LLaVA | only $W$ (alignment) | $W$ + LM | none |
| OneVision | only the projector | full model | full model, task-like data |
| Qwen-VL | encoder + adaptor, LM frozen, noisy data | everything, higher resolution | adaptor + LM, encoder frozen, instructions |

The pattern: the first stage connects things, on the largest and cheapest data; later stages unfreeze more and use better data; instruction data comes last. Which parts train depends on which part is weak. LLaVA's CLIP was already strong, so only the connector learned at first. Qwen-VL trains its encoder in stage 1, on the noisy data, while the LM stays frozen. The lecture states these schedules, not the reasons; one plausible reason for the freeze is to keep large amounts of low-quality text from degrading the pretrained LM while the encoder side adapts. The highest-quality data is saved for the end, when everything is already connected.

::predict staged-freezing-training
::code lecture_17.py:L203-L209 | stage 1: freeze LM, train encoder + adaptor; stage 2: all, higher resolution; stage 3: freeze encoder, train adaptor + LM

The examples show the Qwen ambition: a VLM that subsumes a language model. Some answers are in Chinese; it writes code, because a language model can; and it can find Spider-Man and the Hulk in a picture, answering with **bounding boxes**. "It doesn't actually output an image. It just outputs a bounding box", as coordinates in text.

::figure official/lectures/images/qwen-vl-examples.png | top left, a `<box>(750,0),(999,999)</box>` in a Chinese prompt asks who is in that region; bottom middle, Spider-Man and the Hulk boxed from the model's text output; also a floor sign read, a buggy function fixed, an abstract OCR'd
::kp staged-freezing-training
::kp vlm-template-encoder-projector-lm

## How do you let the image decide its own token count? Qwen2-VL {#qwen2-vl}
source: lecture_17.py:L214-L233 · video 49:16-52:50

Qwen2-VL is "again, an upgrade". Two new ideas: dynamic resolution and multimodal RoPE.

### Dynamic resolution: tokens follow pixel area

The visual encoder is a larger ViT, 675M parameters, and "the main thing" is **dynamic resolution**. Going from LLaVA to OneVision already showed that images of different sizes must be handled; "if you try to do video, it's clear that you need some dynamic resolution". Instead of a fixed crop or a fixed number of tiles, the image is encoded at (close to) its native size, and its token count follows from its size.

The rule:
1. cut the image into 14 × 14-pixel patches, as many as its size gives;
2. merge every 2 × 2 block of neighbouring patch features into one LM token, to keep the context short.

So the token count is about (patches)/4, which grows with the image's pixel area. The lecture's example: a 224 × 224 image gives **66 tokens**.

::figure official/lectures/images/qwen2-vl-architecture.png | the token counts written above the encoder follow the sizes below it: the 1092 × 8204 blog page costs 11,427 tokens, the 224 × 28 equation strip only 8, the 1260 × 700 photo 1,125

::code lecture_17.py:L217-L221 | 675M ViT; dynamic resolution; ViT/14 with every 2×2 compressed gives 66 tokens; video at 2 frames/s, max 16384 tokens

### Worked example: where do 66 tokens come from?

- $224 / 14 = 16$ patches per side, $16^2 = 256$ patches.
- 2 × 2 merge: $256 / 4 = 64$.

That is 64, not 66, and the lecture does not say where the other two come from. The Qwen2-VL paper wraps every image's tokens in a `<|vision_start|>` and a `<|vision_end|>` token: $64 + 2 = 66$. The figure confirms the rule, counting without the two delimiters:
- the equation strip, 224 × 28: $16 \times 2 = 32$ patches, $/4 = 8$ tokens;
- the photo, 1260 × 700: $90 \times 50 = 4500$ patches, $/4 = 1125$;
- the blog page, 1092 × 8204: $78 \times 586 = 45{,}708$ patches, $/4 = 11{,}427$.

Since the count follows area, doubling the side quadruples it: a 448 × 448 image is $32^2 = 1024$ patches, 256 merged tokens, 258 with delimiters, not 132.

**Video** is sampled at 2 frames per second, capped at **16384 tokens**. A 60-second clip at 448 × 448 per frame is 120 frames × 258 = 30,960 tokens, almost twice the cap. To fit, each frame may have $16384 / 120 \approx 136$ tokens, about a 320 × 320 frame. The cap forces the same trade OneVision made by hand: lower resolution per frame.

::predict qwen2-dynamic-resolution-tokens
::widget fixture:lecture_17--image-tokens | in "native" mode 224 gives 66 and 448 gives 258; set 60 s of video at 2 frames/s with the 16384 budget and the readout goes over budget and prints the per-frame allowance and the frame size that fits
::note spoken 50:13 | Aloud, dynamic resolution is "the same idea AnyRes idea from LLaVA, where each 224 by 224 patch is encoded with a ViT". The code's mechanism is different: native resolution with 2 × 2 merging, not tiling into fixed pieces. He also said the ViT "they started with is the OpenCLIP" one, while the code says the encoder was initialised from DFN (Data Filtering Networks).

### Multimodal RoPE: a position in time, height and width

A student had asked, back at CLIP, whether patches deserved a smarter position than 0, 1, 2, …; this is the answer, "fancier positional embeddings that do take into account the spatial structure".

Recall [L3's RoPE](#/read/lecture_03): each pair of dimensions in the query and key is rotated by an angle proportional to the token's position, with a different frequency per pair, so that the inner product of two tokens depends only on their *distance*. In text, distance is one number, how many tokens apart.

**MRoPE** gives every visual token three coordinates instead: time (which frame), height (which row) and width (which column). The rotary dimensions are split into three groups; for each axis, "you compute the RoPE and then you concatenate". Text tokens get the same number on all three axes, so for text MRoPE reduces to ordinary RoPE.

::figure official/lectures/images/qwen2-vl-mrope.png | each patch of the 3-frame, 3 × 4 video carries (time, height, width), from (0,0,0) to (2,2,3); the text after it continues at (4,4,4), (5,5,5), one past the largest id used

::code lecture_17.py:L223-L224 | MRoPE, given as the paper's figure
::note deferred 52:09 | "Later, with Qwen-3, we'll see how this is actually a bit suboptimal": the blocked split of the rotary dimensions, fixed in Qwen3-VL.

### Initialisation and stages

The LM is initialised from Qwen2 and the vision encoder from DFN. Training follows the Qwen-VL shape: stage 1 trains only the visual encoder, stage 2 trains everything, stage 3 trains the LM on instruction-following data.

::code lecture_17.py:L226-L230 | LM from Qwen2, encoder from DFN; three stages as in Qwen-VL

::figure official/lectures/images/qwen2-vl-capabilities.png | the capability map: general chat, video understanding, grounding, multilingual OCR, math and code, formula recognition, function calling, UI interaction; every output is still text
::kp qwen2-dynamic-resolution-tokens
::kp staged-freezing-training

## What changed in a state-of-the-art VLM? Qwen3-VL {#qwen3-vl}
source: lecture_17.py:L236-L265 · video 52:50-59:21

Qwen3-VL (2025) is covered "quickly". The changes are not "structural big changes", but "they're probably changes that do impact the quality of the model". The diagram is the Qwen2-VL one with four additions: Qwen3 as the LM, timestamp tokens in the video, DeepStack on the right, and a new position layout you cannot see.

::figure official/lectures/images/qwen3-vl.png | compare with Qwen2-VL: the video's frame tokens are now separated by text tokens like <0.0 seconds>, <4.0 seconds>; and on the right, DeepStack feeds vision tokens into LLM blocks 1, 2 and 3, not only the input

- **Language model**: the Qwen-3 family, dense and mixture-of-experts up to 235B-A22B (235B parameters, 22B active per token; see [L4's MoE](#/read/lecture_04)). The Qwen-3 models "are really, really good, and it really helps the quality of the final model".
- **Long context**: up to **256K** tokens, "really important if you're trying to do long video". This is the long-context payoff deferred in the OneVision section.
- **Vision encoder**: SigLIP-2, the same architecture as SigLIP, designed to be backward compatible.

::code lecture_17.py:L240-L245 | Qwen-3 dense and MoE up to 235B-A22B, 256K context, SigLIP-2

### Interleaved MRoPE

In RoPE, each pair of rotary dimensions has its own frequency, falling steadily along the dimension index: the first pairs rotate fast, the last slowly. Qwen2-VL's MRoPE gave time the first block of dimensions, width the next, height the last: `[t t t t w w w w h h h h]`. So each axis lived at one end of the spectrum. With RoPE's usual ordering, time got only the fast dimensions and height only the slow ones (the professor guessed the reverse, temporal low and height high, but either way each axis is stuck at one end).

Why that hurts: slow frequencies change little between neighbours, which is good for long distances and bad for telling adjacent positions apart; fast ones are the reverse. An axis that only has slowly rotating dimensions has trouble telling adjacent rows apart.

Qwen3-VL **interleaves** the axes: `[t w h t w h t w h t w h]`. Each axis keeps the same number of dimensions, four of the twelve in the schematic, but now spread from fast to slow, so "all the axes are exposed to both low and high frequency".

::predict mrope-interleaved-axes
::widget fixture:lecture_17--mrope-axes | switch from blocked to interleaved: each axis still owns 4 of the 12 slots, but height's slots go from only the slowest (8–11) to a spread from 2 to 11

### Timestamps as tokens

In Qwen2-VL, time was "implicit in the positional encodings": a frame's temporal id says which frame it is, not *when* it is. Two clips of 40 frames, one sampled at 2 frames per second (20 seconds) and one at 0.5 (80 seconds), get identical temporal ids 0 to 39. Qwen3-VL adds **explicit timestamp tokens**, text such as "<0.0 seconds>" before each frame's tokens, so the model can "directly refer to, like what happened after two seconds" and tell a 20-second clip from an 80-second one.

::widget fixture:lecture_17--mrope-axes | give clips A and B different frame rates with the same frame count: their temporal ids are identical while the seconds differ, until the timestamp tokens are switched on
::code lecture_17.py:L246-L248 | interleave t, w, h across frequency bands; explicit timestamps as separate tokens
::video 54:04-54:48 | why interleave: a blocked layout gives each axis only low or only high frequencies
::kp mrope-interleaved-axes

### Balancing long videos against short text

A video example can be thousands of tokens; a single-image or text example a few hundred. The normal loss treats every token the same, "and this would mean that the video examples are going to dominate". With one 500-token text example and one 8000-token video in a batch, a per-token mean takes $8000 / 8500 = 94\%$ of the loss, and of the gradient, from the video.

Qwen3-VL uses a **square-root-normalized per-token loss**. "The details aren't too clear. But I believe they normalized each example by the square root of the length." Two extremes bracket it:
- **per-token mean**: an example weighs in proportion to its length $n$: the video gets 94%;
- **per-example mean**: every example weighs the same: 50%;
- **divide each example's summed loss by $\sqrt{n}$**: an example weighs in proportion to $\sqrt{n}$. Here $\sqrt{8000}/\sqrt{500} = 4$, so the video gets $4/5 = 80\%$.

Long examples still count for more, just much less than in proportion to their length.

::widget fixture:lecture_17--loss-share | under the per-token mean the video's share is its token count over all the batch's tokens, so it takes over as the video grows; the square-root and per-example rows shrink that share (the square-root row is this read-through's reading of L249, not a formula the lecture gives)
::code lecture_17.py:L249 | square-root-normalized per-token loss, so long video examples do not dominate
::note spoken 1:02:03 | In the Q&A he added that you can always down-weight a source, as in [L14's data mixtures](#/read/lecture_14), and that multimodal tokens do not "vastly" outnumber the tens of trillions of text tokens a frontier LM is trained on.

### DeepStack: inject vision into several layers

The connectors so far, in order: LLaVA's linear map, OneVision's MLP, Qwen-VL's cross-attention. **DeepStack** goes further. The vision encoder already computes a stack of representations, one per layer; DeepStack adds features from several of those layers "directly into the residual stream of the language model" at several of its layers, not only at the input. "A bit more of a deep fusion", instead of treating the encoder as a black box that emits one sequence.

::code lecture_17.py:L251-L252 | DeepStack: cross-layer fusion into multiple LM layers
::note slip 57:03 | DeepStack is called "a paper from the DeepSeek team". The linked paper (arXiv 2406.04334) is, as far as this atlas knows, by Meng et al. of Fudan University and Microsoft, not DeepSeek.

### Training and results

Pre-training now has four stages: train the adapter first ("the same as all of these models"), then train everything on progressively longer sequences, 8K, 32K and 256K. Post-training is SFT on long chain-of-thought data, knowledge distillation and RL, the pipeline of [L15](#/read/lecture_15) and [L16](#/read/lecture_16) applied to a VLM.

::figure official/lectures/images/qwen3-vl-pretraining.png | S0 trains only the merger (the adapter) on 67B tokens; S1 and S2 train everything on about 1T tokens each, at 8,192 and 32,768 tokens per sequence; S3 adapts to 262,144 (256K) on 100B. "Most of the tokens are in the stages 2 and 3" aloud; by the table most are in S1 and S2, the second and third rows

::code lecture_17.py:L254-L257 | four pre-training stages (adapter, then 8K, 32K, 256K); post-training: SFT on long CoT, distillation, RL

"At this point, it's really a systems paper": the core ideas are the few above, with "a lot of details that are different". The results table compares Qwen3-VL 235B-A22B with Gemini 2.5 Pro, GPT-5 and Claude Opus 4.1, and the open model holds the bold, best-in-row number in many rows.

::figure official/lectures/images/qwen3-vl-results.png | bold marks the best in each row; Qwen3-VL leads most of the document-understanding, grounding and agent rows, and trails on some STEM-puzzle and video rows

The summary: state-of-the-art performance; lots of data work "but not many details" (for data mixtures the professor points to the LLaVA papers instead); minor but potentially important architectural changes; and scale.

::code lecture_17.py:L261-L265 | SOTA; lots of data work but not many details; minor architectural changes; scale up
::kp staged-freezing-training

## Can one model read and draw images? Chameleon {#chameleon}
source: lecture_17.py:L268-L298 · video 59:21-1:00:39, 1:07:18-1:14:43

### Why the VLMs so far cannot draw

Every model so far encodes images with CLIP or SigLIP and injects the vectors into a language model. A student asked about generating video; the answer: "these models don't generate videos or images. All the multimodal stuff is on the input side. You're always generating text." The LM's output layer is a softmax over a text vocabulary, and the continuous image features have no decoder back to pixels. More training would not change that. To draw, you need something else, such as a diffusion model attached as a head.

::code lecture_17.py:L271-L272 | VLMs encode images and inject them into the LM; they can't generate images (need diffusion)

### Chameleon: make everything discrete tokens

Chameleon (Meta, 2024) takes the other road: map images into **discrete tokens** from a vocabulary, like words. Then one autoregressive model can read and generate images "in the same way", interleaving text and images freely in both its input and its output. "Maybe this reflects that I'm a language person", but the appeal is clear: text and images "truly live in the same space", by "making everything look like text".

::figure official/lectures/images/chameleon.png | (a) text tokens (green) and image tokens (blue) from an image tokenizer go into one mixed-modal autoregressive LM; (b) the same LM emits text and then image tokens, which an image de-tokenizer turns back into a picture

::figure official/lectures/images/chameleon-example.png | one response interleaving text and generated images: asked for quirky birds, it writes about a toucan, a puffin and a golden pheasant and draws each where the <img> marker sits

### The image tokenizer: a VQ-VAE

The encoder now has to produce discrete codes, "so we can generate them". The tool is the **VQ-VAE** (vector-quantized variational autoencoder, van den Oord et al., 2017):
1. an encoder maps the image to a grid of continuous vectors;
2. each vector is rounded to the nearest entry of a learned **codebook** of prototype vectors (about 8,000 here); the index of that entry is the token;
3. a decoder takes the codebook vectors and reconstructs the image;
4. train encoder, codebook and decoder to minimise the reconstruction loss.

::figure official/lectures/images/vq-vae.png | the encoder's continuous grid (green) is snapped to nearest codebook entries, giving a grid of integer indices (3, 12, 7, …): these are the tokens; the decoder rebuilds the image from them. The loss box shows the extra codebook and commitment terms the lecture skips

The rounding is not differentiable, so extra loss terms are needed, which the professor "won't have time to get into".

Chameleon's tokenizer turns a **512 × 512 image into 1024 tokens**, each from a codebook of **8192**. Then a **new BPE tokenizer** is trained over the combined data, since text plus image codes "looks different" from plain language.

::code lecture_17.py:L279-L285 | discrete tokens so they can be generated; VQ-VAE; 512 × 512 → 1024 tokens, codebook 8192; a new BPE tokenizer

### Worked example: what 1024 tokens can carry

- Grid: $\sqrt{1024} = 32$ tokens per side, so each token stands for a $512/32 = 16$-pixel square.
- Information per token: $\log_2 8192 = 13$ bits.
- The whole image as tokens: $1024 \times 13 = 13{,}312$ bits, about 1.7 kB.
- The raw image: $512 \times 512 \times 3 \times 8 = 6{,}291{,}456$ bits, about 786 kB.

The tokens are about 470 times smaller. Each 16 × 16-pixel cell, with its 6,144 bits of colour, becomes one choice among 8192 prototypes. A 6-point letter is smaller than that cell, so small print cannot survive. That is "discretization loses information - think OCR".

::predict discrete-image-tokens-vqvae
::widget fixture:lecture_17--image-tokens | in VQ mode at 512 px with 16-px cells and 8192 codes, 1024 tokens of 13 bits stand in for about 6.3 million bits, about 470× smaller

### Training, and why it was unstable

Training is "just normal language model training. There's no adapter", no separate encoder in the loop, two stages as for a language model:
- **Stage 1 (80% of training)**: large-scale, unsupervised: 2.9T text tokens, 1.5T text/image tokens, 400B interleaved text/image tokens.
- **Stage 2 (20%)**: half stage-1 data, half high-quality data.

::code lecture_17.py:L287-L289 | stage 1: 2.9T text, 1.5T text/image, 400B interleaved; stage 2: 50% stage-1 data, 50% high quality

The problem was stability. "Just calling things discrete tokens isn't hiding the fact that there's an image living there." Text tokens have **low entropy**: "most words are predictable". Image tokens have **high entropy**: "I don't know what shade of blue this exact token is going to be."

Training on the mixture made the norms grow and the loss unstable, **norm growth and logit drift**. The fixes are two tools from [L3's stability section](#/read/lecture_03): **QK-norm** (normalise queries and keys before their dot product, so attention logits cannot blow up) and **z-loss** (a penalty that keeps the softmax normaliser near 1, so output logits cannot drift). They controlled it "to some extent". Why the entropy gap drives norm growth is not explained in the lecture.

This is the same lesson as Qwen3-VL's square-root loss, in a sharper form: visual and text tokens differ in density and entropy, and a model that mixes them has to correct for it. Each imbalance has its own fix. Length imbalance is fixed in the loss weights; the entropy mismatch inside one softmax is fixed in the architecture.

::predict modality-balance-and-stability
::code lecture_17.py:L291-L293 | low-entropy text vs high-entropy image tokens leads to norm growth and logit drift; fixes: QK-norm, z-loss
::kp modality-balance-and-stability

### The verdict

- **Elegant**: just autoregressive modelling of discrete tokens, one model for every modality.
- **Not as performant**: discretisation loses information, OCR again. "If you discretize very small print, you're not going to be able to read it anymore."
- **Training with multiple modalities is tricky**, more so here than in the Qwen models.

VQ tokens were popular for image generation for a while, because a Transformer generates discrete symbols, so you put your images into discrete form. Then diffusion models became viable for generation, and "this flavor of method is less popular than it used to be".

::code lecture_17.py:L295-L298 | elegant; not as performant (discretization loses information); multimodal training is tricky
::note skip 1:13:16 | "I'm not even going to show the results": Chameleon is shown for its elegance, not its benchmark numbers.
::video 1:13:20-1:13:54 | the cost of discreteness: not as performant, and small print becomes unreadable
::kp discrete-image-tokens-vqvae

## What should you take away? {#summary}
source: lecture_17.py:L40-L45 · video 1:14:43-1:17:34

**Frontier models are expected to be multimodal**, "natively multimodal", omni. Gemini and GPT are announced that way and do handle all these modalities, "but of course, there's no details about how these are built". The professor's guess: a continuous encoder, "because you don't want to lose information", and diffusion for generation. "But that's my speculation."

**The fundamental challenge is encoding non-text modalities.** There is "no one universal encoder", because understanding and generation want different things. For classification, CLIP only needed high-level semantics, so its vectors "could be fairly small". OCR or generating an image needs "really fine-grained detail", which is why diffusion is good at generation: it can optimise the low- and high-frequency information directly. This is the asymmetry first seen in CLIP's ablation, where predicting the caption word by word lost to ranking.

**Balance the modalities.** Video "certainly has lower information density than text", so "you don't want a video to overwhelm your text": square-root loss weights, budgets per kind of input, stability fixes.

**The current recipe is continuous encoders + Transformer + diffusion for generation.** "Even CLIP, even though it's five years old, or similar ideas are still the go-to way to capture semantics of images."

::code lecture_17.py:L40-L45 | the lecture's five summary lines

The whole lecture in one table, by how an image becomes tokens:

| scheme | model | tokens for one image | can generate images? | cost |
|---|---|---|---|---|
| resize + center crop, ViT/14 | CLIP, LLaVA | 576 for any image | no | borders and fine detail lost |
| cross-attention to fixed queries | Qwen-VL | 256 for any image | no | count fixed whatever the detail |
| AnyRes tiles | LLaVA-OneVision | $a \cdot b$ pieces × tokens per piece | no | tokens grow with area; budgets by input type |
| native resolution, 2 × 2 merge | Qwen2-VL, Qwen3-VL | patches / 4 (+2): 66 at 224 × 224 | no | tokens grow with area; video capped |
| VQ-VAE codes | Chameleon | 1024 for 512 × 512, 13 bits each | yes | discretisation loses detail; unstable mix |

And the recurring choices of the VLM template:
- **encoder**: CLIP or SigLIP, trained contrastively on web captions, so it captures caption-level semantics;
- **projector**: linear, MLP, cross-attention, or DeepStack's fusion into many layers;
- **data**: mostly synthetic and task-specific, distilled from stronger models;
- **training**: connect first with the big parts frozen, then unfreeze, instruction data last;
- **positions**: MRoPE over time, height and width, interleaved across frequencies, with timestamps as tokens.

::video 1:15:39-1:16:09 | no universal encoder: semantics suffice for classification, OCR and generation need fine detail
::note skip 1:17:05 | "Diffusion models, which I didn't talk about, are great for generation": generation beyond Chameleon is left out entirely.
::note aside 1:17:22 | "I guess we don't have any homework on doing this." No assignment covers multimodal models; the professor encourages trying to train some of them.
::kp everything-into-tokens
::kp clip-semantics-vs-detail
::kp modality-balance-and-stability
