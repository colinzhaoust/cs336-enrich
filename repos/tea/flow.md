---
title: TEA · TheoremExplainAgent, read through
minutes: 35
---
TheoremExplainAgent (TEA) takes the name of a theorem and a one-line description and returns a narrated, multi-scene Manim video, written and debugged by language models. This read-through follows one run through the code in the order it executes, from the command line to the stitched video, and then reads the separate evaluator. After it you can say which model is called at each step, which file each step leaves on disk, what makes a scene count as "done", and which kinds of quality nobody ever checks.

## What does one run of TEA do? {#overview}
source: generate_video.py:L76-L133 · external/TheoremExplainAgent-main/README.md:L125-L145

TEA is the code release of an ACL 2025 paper (arXiv 2502.19400). According to Atlas's survey notes on that paper, about 94% of its videos were generated successfully. Keep that number in mind; the last third of this read-through asks what "successfully" means.

A recap of **Manim**, the library TEA writes code for: a Manim video is a Python class that subclasses `Scene`, whose `construct()` method creates objects and plays animations on them. `manim -qh file.py` runs it and writes an `.mp4` (`-qh`: 1080p at 60 frames per second). The add-on **manim-voiceover** lets the code wrap animations in `with self.voiceover(text="...")` blocks: the text is spoken by a text-to-speech service, the animation is timed to the audio, and a subtitle file (`.srt`) is written next to the video. TEA's speech service is a local model, Kokoro, wrapped in its own `KokoroService` class. Atlas, the site you are reading, also renders its animations with Manim, which is why this repository is worth reading closely.

A run passes through six stages. Each one is a few model calls followed by a file written to disk:

| Stage | Code | Model | What it leaves on disk |
|---|---|---|---|
| 1. Scene outline | `VideoPlanner.generate_scene_outline` | planner | `<prefix>_scene_outline.txt` |
| 2. Per-scene plan (3 calls) | `_generate_scene_implementation_single` | planner | `scene<i>/subplans/*.txt`, `scene<i>/<prefix>_scene<i>_implementation_plan.txt` |
| 3. Code | `CodeGenerator.generate_manim_code` | scene | `scene<i>/code/<prefix>_scene<i>_v0.py` |
| 4. Render | `VideoRenderer.render_scene` | none (the `manim` CLI) | `media/videos/...`, `scene<i>/succ_rendered.txt` |
| 5. Fix loop | `CodeGenerator.fix_code_errors` | scene | `v1.py`, `v2.py`, ... |
| 6. Combine | `VideoRenderer.combine_videos` | none (ffmpeg) | `<prefix>_combined.mp4` and `.srt` |

`<prefix>` is the topic lower-cased with every run of other characters replaced by `_`, so "Big O notation" becomes `big_o_notation`. A third model role, the **helper**, appears only when retrieval (RAG) is switched on: it picks Manim plugins and writes search queries.

The conductor is `VideoGenerator` in `generate_video.py`. Its constructor builds the three workers and a semaphore that caps how many scenes are processed at once.

::code generate_video.py:L92-L133 | the three workers the rest of the run calls into, and the scene semaphore

A theme runs through the whole trace. Planning prompts repeat careful layout rules, but the only test a scene must pass before it counts as rendered is that the `manim` process exits with code 0. Watch for where checks are *asked for* in prompt text and where they are *performed* by code.

::note aside | Everything in this read-through comes from reading the code statically. No model was called and no video was rendered, so statements about runtime behaviour ("this raises AttributeError") describe what the code must do, not something observed.

## What do you type, and what gets built? {#cli}
source: generate_video.py:L667-L744 · generate_video.py:L917-L954 · external/TheoremExplainAgent-main/README.md:L125-L145

There are two ways to say what to explain. A batch run passes `--theorems_path`, a JSON list of records such as `data/thb_easy/math.json` (20 theorems); each record's `theorem` and `description` become the topic and its description. A single run passes `--topic "Big O notation" --context "most common type of asymptotic notation..."`, the README's own example. Both forms need a `--model`, which must be on the whitelist in `src/utils/allowed_models.json`; the default is `gemini/gemini-1.5-pro-002`.

::code generate_video.py:L714-L735 | three wrappers, all at temperature 0.7; only the helper can name a different model

The `__main__` block builds three `LiteLLMWrapper` objects (LiteLLM calls OpenAI, Gemini, Azure, Bedrock and others behind one interface). The planner and the scene model get `--model`; the helper gets `--helper_model` if given, else `--model`.

::note aside | The wrapper overrides that temperature for OpenAI "o" models: any name matching `openai/o...` (such as the README's `openai/o3-mini`) is sent with no temperature and `reasoning_effort="medium"` instead (`mllm_tools/litellm.py` L156–L158). It also asks LiteLLM for up to 99 API retries per call.

Defaults matter here because most features are off. `--max_retries` is 5, `--max_scene_concurrency` is 1 (scenes one at a time), and RAG (`--use_rag`), the visual fix pass (`--use_visual_fix_code`), example-code prompting (`--use_context_learning`) and Langfuse tracing (`--use_langfuse`) all stay off unless you name them.

::code generate_video.py:L738-L744 | the batch branch; note that --sample_size slices the head of the list

`--sample_size 3` keeps `theorems[:3]`, the first three records in file order, not a random sample.

::predict cli-run-modes

The mode is decided by an `if / elif / else` chain: `if args.theorems_path` (L738), `elif args.topic and args.context` (L917), `else` print a usage message (L952–L954). Each flag is read only inside the branch that needs it. If you pass a theorems file *and* a topic, the file wins and `--topic` is never looked at. `--check_status` and `--peek_existing_videos` are read only inside the batch branch, so in single-topic mode they are silently ignored and the full pipeline runs.

::widget fixture:tea--cli-dispatch | turn on --topic next to --theorems_path: the first test is already true, so the topic flag lands in "never read", with no warning

::note warning | `--max_retries` does not count re-runs of `manim` on the same code. It counts how many times the scene model is asked to *rewrite* failing code; each rewrite is rendered once. The fix-loop section below works out the totals.
::kp cli-run-modes

## Where does a run write, and how does it pick up where it left off? {#resume}
source: generate_video.py:L489-L569 · generate_video.py:L279-L329 · src/core/video_renderer.py:L122-L124 · generate_video.py:L620-L650

Every topic is handled by `generate_video_pipeline`. Before it calls any model it decides what is already done, and it decides by looking for files. For the README example with `--output_dir output/my_exp_name`, a finished run leaves this tree:

- `output/my_exp_name/session_id.txt`: one id per output folder (reused by every later run there), also copied into each topic folder;
- `big_o_notation/big_o_notation_scene_outline.txt`: stage 1;
- `big_o_notation/scene1/subplans/`: the three stage-2 plans and a `scene_trace_id.txt`;
- `big_o_notation/scene1/big_o_notation_scene1_implementation_plan.txt`: the three plans joined;
- `big_o_notation/scene1/code/`: `..._v0.py`, `..._v1.py`, ... and their logs;
- `big_o_notation/scene1/succ_rendered.txt`: an empty file meaning "this scene rendered";
- `big_o_notation/media/videos/big_o_notation_scene1_v0/1080p60/`: what `manim` wrote;
- `big_o_notation/big_o_notation_combined.mp4` and `.srt`: the final video.

Each stage checks for its own output first. If the outline file exists it is read from disk and the planner is not called. Each scene's plan is loaded if its file exists, and only the missing ones are generated. For rendering, a scene is queued if and only if its `succ_rendered.txt` is absent.

::code generate_video.py:L489-L500 | the outline is generated only if its file is missing; delete the file to re-plan
::code generate_video.py:L556-L569 | normal mode keys on succ_rendered.txt; only --only_render keys on code files

So rerunning the same command is cheap: finished scenes are skipped. Code files do not count as done; a scene with `v0.py` to `v3.py` and no marker starts again at a fresh v0. Topics that normalise to the same prefix share a folder, so "Big-O Notation" reuses the files of "Big O notation".

::predict resume-by-artifact

The prediction contains a quirk worth spelling out. The queued scenes are passed on as a plain list, `render_video_fix_code` enumerates that list, and `process_scene` sets `curr_scene = i + 1` from the list index. If scenes 1–3 rendered and scenes 4 and 5 are retried, scene 4's plan is processed as "scene 1": its code goes into `scene1/code/`, the model is told it is writing `Scene1`, and its success marker is written to `scene1/`. `scene4/` never gets a marker, so the next rerun queues it again.

::note aside | Two more things follow from the same index, by reading the code (not observed). The new file is named `..._scene1_v0.py`, the same as scene 1's, so it overwrites scene 1's code and renders into scene 1's media folder. And with RAG on, the query cache is keyed by that folder, so the retried scene reuses scene 1's cached queries.

The marker is an empty file. Nothing records *which* code version succeeded, so the combine step later has to guess from folder names. `--check_status` (batch mode only) prints the same file checks as a table, one `PCR` triple per scene: plan file, at least one code file, success marker.

::code generate_video.py:L835-L845 | P, C and R are three file-existence tests, printed per scene
::widget fixture:tea--fix-loop | look at the rerun line: it queues exactly the scenes without succ_rendered.txt, whatever code they hold, and renumbers them by position, so a queued scene 4 is generated into folder scene1/
::kp resume-by-artifact

## Stage 1: how is a theorem split into scenes? {#outline}
source: src/core/video_planner.py:L135-L178 · task_generator/prompts_raw/__init__.py:L185-L233 · generate_video.py:L511-L535

The outline is one call to the planner model. The prompt, `_prompt_scene_plan`, casts the model as an expert in educational video production and in the topic, gives it the topic and description, and asks for a plan of individual scenes. Each scene needs four fields: a 2–5 word title, a purpose (its learning objective and link to the previous scene), a description, and a layout. The answer must be XML: an `<SCENE_OUTLINE>` block holding `<SCENE_1>`, `<SCENE_2>`, and so on.

::code task_generator/prompts_raw/__init__.py:L219-L231 | the spatial rules and the requirements list: 3 to 7 scenes, under 15 minutes, no external assets, no quizzes

The prompt also states two spatial rules that every later prompt repeats: a **safe-area margin** of 0.5 units on all sides of the frame and a **minimum spacing** of 0.3 units between any two objects, edge to edge. (Manim's frame is about 14.2 units wide and 8 units tall.) It ends by promising that these constraints "will be strictly enforced in subsequent planning stages". Keep that promise in mind.

The code that handles the answer is short. It looks for `<SCENE_OUTLINE>...</SCENE_OUTLINE>` with a regular expression and, if the search fails, keeps the *whole* response as the outline. Either way the result is written to `<prefix>_scene_outline.txt`.

::code src/core/video_planner.py:L161-L175 | one planner call, a regex with a keep-everything fallback, and a file write

Back in the pipeline, the number of scenes is counted, not read from any field. `extract_xml` strips an `xml` code fence if there is one, and `re.findall(r'<SCENE_(\d+)>[^<]', ...)` counts opening tags followed by any character other than `<`. Then, for each scene number, `<SCENE_n>(.*?)</SCENE_n>` cuts that scene's block out of the outline; if the closing tag is missing, the scene is skipped without a message.

::code generate_video.py:L515-L535 | count opening tags, then cut each block out; a block with no closing tag is silently not planned

::predict scene-outline-plan

The limits in the prompt (3 to 7 scenes, under 15 minutes) exist only as prose; no code checks them. And the fallback has a cost. A reply cut off by a token limit inside `<SCENE_5>` fails the outline regex, so it is saved whole; the count still finds five opening tags, and scene 5 is dropped at planning because its block never closes. A reply with the wrong tag names (say `<SLIDE_1>`) counts zero scenes, and nothing is planned or rendered, without an error.

::widget fixture:tea--outline-parse | edit the reply: a reply cut off inside <SCENE_5> still counts 5, is saved whole, and scene 5 is silently not planned

::note aside | The ban on "quiz sessions" (requirement 8, next to the ban on channel promotion) is a product choice, but it means the video never asks the viewer to commit to an answer. That is the opposite of Atlas's commit-before-reveal rule, the predictions you see on this page. The same prompt does allow "detailed example questions", and the narration-stage prompt asks the model to "pose questions to encourage active thinking", without pausing for an answer.
::kp scene-outline-plan

## Stage 2: how does each scene get a detailed plan? {#scene-plan}
source: src/core/video_planner.py:L180-L346 · task_generator/prompts_raw/__init__.py:L1407-L1459 · generate_video.py:L527-L535

The outline says *what* each scene teaches. Before anyone writes code, each scene gets a detailed plan from three more planner calls, made one after another, each with its own prompt template and its own XML tag:

1. **Vision and storyboard** (`<SCENE_VISION_STORYBOARD_PLAN>`): what appears on screen, in which order, and where. Input: the scene's block from the outline.
2. **Technical implementation** (`<SCENE_TECHNICAL_IMPLEMENTATION_PLAN>`): which Manim objects (`Tex`, `MathTex`, `VGroup`...), with which parameters, positioned how, animated with which methods. Input: the outline block plus the storyboard.
3. **Animation and narration** (`<SCENE_ANIMATION_NARRATION_PLAN>`): animation timings and the full narration script. Input: the outline block, the storyboard and the technical plan.

Each output is cut out by its tag (keeping the whole reply if the tag is missing, the same fallback as the outline), saved under `scene<i>/subplans/`, and appended to a running string. At the end the string is written as `scene<i>/<prefix>_scene<i>_implementation_plan.txt`. That concatenation is the "implementation plan" that code generation and every fix will receive.

::code src/core/video_planner.py:L241-L248 | one stage: call the planner, cut out the tag (or keep everything), append to the plan
::code src/core/video_planner.py:L338-L343 | the three stage outputs are written as one file under a heading

Although the method is `async`, the pipeline calls it inside a plain `for` loop with `await`, so scenes are planned one at a time. (Concurrent variants exist in `VideoPlanner` but are not on this path.)

::predict per-scene-three-stage-plan

The count generalises: with RAG off, planning costs $1 + 3n$ planner calls for $n$ scenes, so 16 for five scenes and 22 for the maximum of seven.

Why three stages instead of one? Each prompt is narrower, and each later stage builds on concrete decisions made earlier, which should reduce drift between what the storyboard imagines and what code can do. The templates are insistent about layout. The storyboard template repeats the 0.5 margin and 0.3 spacing as strictly enforced and bans absolute coordinates: every position must be relative (`next_to`, `align_to`, `shift` from another object or the frame edge). The technical template asks for a `buff` of at least 0.3 on every placement and ends with safety checks.

::code task_generator/prompts_raw/__init__.py:L1450-L1457 | the "Mandatory Safety Checks" are a paragraph addressed to the model; nothing below runs them

Those checks are instructions, not code. No program measures a margin or a gap between this stage and rendering. A storyboard that puts a title 0.2 units from the top edge flows unchanged into the joined plan and on into the code prompt.

Each template also tells the model that its scene is entirely self-contained, with no dependency on other scenes, while the narration should flow as one video. That makes scenes independent to render (one failure does not poison the others), but nothing keeps colours, positions or notation consistent across scenes; only the narration is asked to connect them.

::note aside | With `--use_context_learning`, example plans from `data/context_learning` are appended to the matching stage's prompt. With `--use_rag`, each stage's prompt also gets retrieved documentation, which the next section explains.
::widget fixture:tea--outline-parse | the call counter: each planned scene adds three planner calls to the one outline call; with --use_rag the helper adds one plugin call and one query call per stage
::kp per-scene-three-stage-plan

## What does retrieval add, and when? {#rag}
source: src/rag/vector_store.py:L56-L107 · src/rag/rag_integration.py:L57-L97 · src/rag/vector_store.py:L247-L356 · src/core/code_generator.py:L289-L309

Code models know Manim's API only as well as their training data, and Manim changes between versions. TEA's optional remedy is **retrieval-augmented generation (RAG)**: before a prompt is sent, search the Manim documentation and paste relevant passages in. The documentation is not in the repository; the README links a zip to unpack into `--manim_docs_path`.

### Setting up the stores

With `--use_rag`, the documentation is split into **Chroma** collections (Chroma is a local vector database). One collection, `manim_core`, holds the core docs from `manim_docs_path/manim_core`; each folder under `manim_docs_path/plugin_docs` gets its own `manim_plugin_<folder>` collection. Building a collection walks the folder recursively, keeps only `.md` and `.py` files, splits them with LangChain's Markdown and Python splitters, and embeds the pieces with the `--embedding_model` through LiteLLM. Collections are saved to `--chroma_db_path` and loaded on later runs.

::code src/rag/vector_store.py:L66-L85 | load the core store if it exists, else build it; then one store per plugin folder

Other file types (`.rst`, `.ipynb`, `.txt`) are skipped silently, and a folder with no `.md` or `.py` files yields an empty collection, not an error. The planner and the code generator each open their own `RAGVectorStore` on the same path.

The plugins are five community add-ons the README acknowledges: manim-physics, manim-Chemistry, ManimML, manim-dsa and manim-circuit. Which matter for a topic is decided once, before the outline, by the helper model: `detect_relevant_plugins` sends the topic, the description and the list in `plugin_docs/plugins.json`, and parses a JSON array from the reply. On any exception it returns an empty list.

::code src/rag/rag_integration.py:L79-L97 | plugin choice is an LLM call over plugin descriptions, not keyword matching; any failure means "no plugins"
::kp rag-plugin-detection-and-stores

### Five retrieval points per scene

Retrieval is not done once per video. It happens before each of the three planning stages, before code generation and before each error fix. Each time, the helper model turns that stage's input into JSON queries (at most 10, says the code-stage prompt), each tagged `manim-core` or a plugin name, cached per scene under `scene<i>/rag_cache/`.

Then `find_relevant_docs` routes each query to the core store or to its plugin's store (dropping it with a warning if there is none) and asks for $k = 2$ chunks with relevance score at least 0.5. Duplicate chunks are removed and the rest pasted into the prompt, each with its score.

::code src/rag/vector_store.py:L277-L291 | route queries by type, then k chunks per query above a 0.5 relevance score

So $k = 2$ is per query, not per prompt: ten queries can bring up to 20 chunks before de-duplication.

::predict rag-query-retrieve-k2

The cache key is the scene folder, not the input text. That is harmless for the planning and code stages, whose input for a scene does not change. It is wrong for error fixes: the queries written for a scene's *first* error are reused for every later fix of that scene, even for a completely different error.

::widget fixture:tea--fix-loop | with --use_rag on, the error-fix query calls count scenes that needed a fix, not fixes: a scene's second fix reuses rag_queries_error_fix.json

::note aside | The detected plugins do not reach code generation. `generate_manim_code` and `fix_code_errors` call their query generators without a plugin list, so those two query prompts are told "No plugins are relevant." and that they must not use plugins not listed (`code_generator.py` L289-L296, L356-L364). Yet, as the next section shows, the code template imports all five plugins in every scene.
::note warning | A static-reading hazard: `find_relevant_docs` reads `span.id` for every query (L286, L304), but `span` is only created when Langfuse is enabled (L263-L273). Run with `--use_rag` but without `--use_langfuse`, as in the README's RAG example, and the first search with a core query should raise `UnboundLocalError` and stop the run. This was not executed here.
::kp rag-query-retrieve-k2

## Stage 3: how does a plan become Manim code? {#codegen}
source: generate_video.py:L350-L381 · src/core/code_generator.py:L252-L335 · task_generator/prompts_raw/__init__.py:L1669-L1800 · task_generator/prompts_raw/__init__.py:L120-L165

Once plans exist, `render_video_fix_code` starts one `process_scene` task per queued scene and runs them with `asyncio.gather`; the semaphore lets at most `--max_scene_concurrency` of them (default 1) work at a time. A task begins by asking the scene model for the first version of the code.

::code generate_video.py:L362-L372 | the code prompt gets the whole outline, this scene's joined plan, and four fixed context strings

The four extra strings never change: a cheat sheet of Manim's class-inheritance diagrams, font-size advice (28 for titles, 24 for labels and formulas), coordinate limits for the frame, and an empty string. Example scenes (`--use_context_learning`) and retrieved chunks (`--use_rag`) are appended the same way.

The template, `_prompt_code_generation`, demands strict adherence to the plan and the spatial constraints and lists 20 guidelines. The ones that shape every file:
- the class is named `Scene{n}` and inherits at least from `VoiceoverScene`;
- speech comes from `KokoroService`, imported from `src.utils.kokoro_voiceover` exactly, with narration in `with self.voiceover(text=...)` blocks timed to the narration plan;
- positions are relative only, with the 0.5 margin and 0.3 spacing; the model must "implement explicit checks" for both and comment any violation "for manual review";
- no external files, no main function, default black background, a fixed palette of text colours.

It ends with a skeleton the reply must follow, inside `<CODE>` and a Python code fence.

::code task_generator/prompts_raw/__init__.py:L1735-L1744 | five plugin imports in every scene, with the instruction not to change them
::code task_generator/prompts_raw/__init__.py:L1789-L1793 | the class signature and the speech service every scene starts from

Two details cause trouble later. First, every scene imports all five plugins, whether the plan uses them or not, and the template forbids changing those lines. On a machine without, say, `manim_ml`, a core-only scene fails with `ModuleNotFoundError` and enters the fix loop, where the fix model has to decide to delete the import. RAG's plugin list does not change this block; the code prompt never receives it. Removing the imports means editing the template and rebuilding the prompts (see [where prompts come from](#prompts)).

Second, the coordinate limits contradict the margin. The frame is 14.22 units wide and 8 tall, centred on the origin, so a 0.5-unit margin leaves $x \in [-6.61, 6.61]$ and $y \in [-3.5, 3.5]$. `_code_limit` says $x$ within $\pm 7$ and $y$ within $\pm 4$, essentially the whole frame. One prompt, two safe areas.

### Getting the code out of the reply

The reply is free text, so the code has to be cut out of it. `_extract_code_with_retries` applies `re.search(r"```python(.*)```", ..., re.DOTALL)`. If there is no Python fence, it sends the reply back to the scene model with a request to return "the exact same code" in the right format, and tries again, for at most 10 attempts in all.

::code src/core/code_generator.py:L233-L250 | 10 attempts mean 9 re-asks; after the last miss it raises ValueError

::predict code-generation-prompt-contract

The re-ask goes to the scene model at temperature 0.7, so "the exact same code" is a request, not a guarantee. If all attempts fail, the `ValueError` propagates out of the scene's task and up through `asyncio.gather`.

Nowhere in this stage does Python code inspect positions. The "explicit checks" and review comments are things the model is asked to write into its own output, and nothing reads them. The only test the code meets is the next stage: does `manim` run it without error?
::kp code-generation-prompt-contract

## What happens when the code does not render? {#fix-loop}
source: generate_video.py:L360-L424 · src/core/video_renderer.py:L53-L126 · task_generator/prompts_raw/__init__.py:L1216-L1245

The first version is saved as `v0.py` and handed to `render_scene`, which runs `manim -qh <file> --media_dir <topic>/media --progress_bar none` in a worker thread and captures its output. If the exit code is non-zero, it raises an exception whose message is the captured standard error (stderr). The `except` branch appends that message to an `_error.log` file and returns `(code, error_text)`. On a zero exit it writes `succ_rendered.txt` and returns `(code, None)`.

::code src/core/video_renderer.py:L53-L68 | the whole gate: run manim, and treat a non-zero exit code as the error, with stderr as its text
::code src/core/video_renderer.py:L113-L124 | failure returns the error text at once; success writes the empty marker file

`render_scene` sits inside a `while retries < max_retries` loop of its own, but its `except` branch returns on the first exception, so that loop never repeats. All retrying happens one level up, in `process_scene`:

::code generate_video.py:L383-L422 | render; success breaks; the cap is tested next; only then the version is bumped and a fix requested

After each render the tests run in a fixed order: success breaks; else, if `curr_version >= max_retries`, print "Max retries reached" and break; else increment the version, call `fix_code_errors`, write `v<n>.py` and its `_fix_log.txt`, and render again.

::worked error-driven-fix-loop

So a scene that never renders costs `max_retries + 1` renders and `max_retries` fixes: 6 and 5 at the CLI default. `render_video_fix_code` has a default of 3 in its signature, but the pipeline always passes the CLI value.

::predict error-driven-fix-loop

The fix prompt (`_prompt_fix_error`) receives the joined plan, the failing code and the raw stderr. It asks for an error analysis, then the complete corrected file inside `<FULL_CORRECTED_CODE>` and a Python fence, and tells the model to remove external assets, keep any voiceover, and leave alone all code not causing the error. The same 10-attempt extractor cuts the code out.

Each fix sees only the current code and the current error. Earlier versions and their errors are not in the prompt, so the loop cannot notice it is oscillating between two broken versions. When the cap is reached the scene is left without a marker, the run moves on, and the combine step will later refuse to build the video.

The error signal is the exit code and nothing else. A scene that renders a blank frame, or two formulas stacked on each other throughout, exits 0 and is a success. A scene whose `manim` run writes a complete `.mp4` and then crashes during cleanup with exit code 1 is a failure, rewritten until the cap.

::widget fixture:tea--fix-loop | set which version first exits 0: a scene that never does is rendered max_retries + 1 times, because the cap is tested after that version's render
::note why | A loop that listens only to the exit code converges on "code that runs", not on "a scene that teaches". Getting the second would need another signal in the loop, for example a check of object positions before the render, or a look at the rendered frames after it. The next section is the repo's attempt at the second.
::kp error-driven-fix-loop

## Is there a check on what the scene looks like? {#visual-fix}
source: src/core/video_renderer.py:L67-L120 · src/core/code_generator.py:L392-L454 · task_generator/prompts_raw/__init__.py:L974-L1020 · src/core/parse_video.py:L9-L61

On paper, yes. With `--use_visual_fix_code`, a scene that renders cleanly is meant to be shown to the scene model acting as a vision-language model (VLM): Gemini and Vertex models get the video, other models one still, the frame (sampled every 5 seconds) with the most pixels brighter than 10 on a 0–255 grey scale. The prompt, `_prompt_visual_self_reflection`, asks about exactly what the exit-code gate cannot see: overlapping objects, objects outside the frame, cluttered layouts, narration out of sync with the picture. The model returns improved code, which becomes the next version, or replies `<LGTM>` ("looks good to me") to accept.

::code src/core/video_renderer.py:L70-L85 | the visual branch runs only after a zero exit; its first act is to read self.scene_model

As shipped, the branch cannot run. `VideoRenderer.__init__` stores only `output_dir`, `print_response` and `use_visual_fix_code`; nothing ever sets `self.scene_model`. The attribute access at L79 raises `AttributeError` before any model is called. Because the whole branch sits inside the same `try` as the render, the `except` catches it and returns the message as if it were a render error.

::predict visual-fix-path-broken

Follow that through the fix loop. Every clean render comes back with the error text "'VideoRenderer' object has no attribute 'scene_model'", and the scene model is asked to fix code with nothing wrong in it. With `--max_retries 2`, v0, v1 and v2 all render cleanly, all come back as errors, and the scene ends unmarked. The flag makes every scene fail, however good its code.

::code src/core/code_generator.py:L411-L415 | a second fault on the same path: the template's placeholders are {implementation} and {generated_code}, but it is formatted with code=

If someone fixed L79, the next call would fail too. `visual_self_reflection` reads the raw `.txt` template from disk (the only prompt the code reads at run time rather than from the compiled constants) and calls `.format(code=code)` on a text whose placeholders are `{implementation}` and `{generated_code}`. That raises `KeyError: 'implementation'`, again before any model call.

::note aside | Two more mismatches sit behind these, by reading the code. The Gemini branch builds the video path as `media/videos/<file>.mp4`, but `manim` writes into `media/videos/<file>/1080p60/`. And the reply format asks for code inside `<code>` tags or a bare `<LGTM>`, while the reply is passed through the Python-fence extractor *before* the `<LGTM>` test. A bare `<LGTM>` therefore triggers format re-asks rather than acceptance.

Even the intended design has a soft spot. The loop stops on `<LGTM>` *or* on any phrase from `_banned_reasonings`, a list of refusal phrases such as "can't assist" or "unable to evaluate". Both simply break out of the loop, so a model that refuses to judge the video is treated exactly like one that approves it. And the snapshot heuristic picks the busiest frame, not a representative one.

::widget fixture:tea--fix-loop | switch --use_visual_fix_code on: a clean render (exit 0) turns yellow and is rewritten like a failure, so the scene never gets its marker
::note why | This is the only place in generation where anything looks at the picture, and it is a VLM opinion, not a measurement. Because a broken check here shows up only as "render failed", the logs give no hint that the layout check never ran.
::kp visual-fix-path-broken

## How do scenes become one narrated video? {#combine}
source: src/core/video_renderer.py:L203-L276 · src/core/video_renderer.py:L391-L440 · generate_video.py:L900-L907 · src/utils/kokoro_voiceover.py

Speech is produced inside each scene's render. `KokoroService` is a manim-voiceover speech service backed by Kokoro, a local text-to-speech model run through ONNX, with voice, speed and language from `src/config/config.py`. Each `with self.voiceover(text=...)` block is synthesised, the animation inside it is timed to the audio, and manim-voiceover writes the subtitle cues. So each successful render leaves an `.mp4` with sound and an `.srt` in `media/videos/<prefix>_scene<i>_v<n>/1080p60/` (the folder is named after the code file, hence the version number).

After all scenes of a topic are processed, the pipeline calls `combine_videos` unless `--only_plan` or `--only_render` was given. It returns early if the combined files already exist; otherwise it re-reads the outline, recounts the scenes with the same regex, and looks for one video per scene.

::code src/core/video_renderer.py:L250-L276 | per scene, the folder with the highest _v number wins; one missing video and nothing is combined

Selection is by folder name only: sort a scene's media folders by the number after `_v` and take the last. Since the loop stops at the first success, the highest version is usually the one that rendered. But nothing checks that the chosen folder holds a video: if the newest version left only partial files, the scene counts as missing even when an older version rendered completely.

::predict render-and-concat

The combine step is all-or-nothing. If the number of videos found differs from the scene count, it prints an abort message and returns. One scene that exhausted its retries means no combined video for the topic.

If the counts match, ffmpeg concatenates the scenes; when any scene has audio, a silent scene gets a silent track of its own length, so the streams line up. The result is encoded with libx264 and AAC into `<prefix>_combined.mp4`.

::code src/core/video_renderer.py:L437-L440 | each scene's subtitle cues are shifted by the summed length of the scenes before it

The subtitles are merged the same way: each scene's `.srt` cues are shifted by the running total of the earlier scenes' durations, read with `ffprobe`, and written to `<prefix>_combined.srt`.

::note aside | Two path assumptions are hard-coded. The quality folder is always `1080p60`, matching `-qh`; change the render flag to `-ql` and the combiner finds nothing. And scene folders are matched by splitting on the word "scene", so a topic whose prefix itself contains "scene" confuses the matcher.
::kp render-and-concat

## What does evaluate.py measure? {#eval}
source: evaluate.py:L300-L346 · evaluate.py:L357-L395 · evaluate.py:L106-L208 · eval_suite/utils.py:L36-L81 · eval_suite/image_utils.py:L13-L99 · eval_suite/prompts_raw/__init__.py:L1-L120 · external/TheoremExplainAgent-main/README.md:L225-L228

Evaluation is a separate program. Point `evaluate.py` at a topic folder (`--file_path output/my_exp_name/big_o_notation`), and by default (`--eval_type all`) it scores the folder's single video with three judge models at temperature 0. The transcript is the first `.srt` there (failing that, a `.txt` other than the outline). The README asks for a video, an SRT file, and access to Gemini and GPT-4o.

::code evaluate.py:L384-L395 | three judges: a text model, a Gemini video model and an image model, all deterministic

The judges never see the generation inputs. The only thing that tells them the subject is the folder name: `big_o_notation` becomes "Big O Notation", and that string is passed to the video and image judges in the slot their prompts call "Description of the theorem". The original `description` from the theorem file does not reach the evaluator at all.

Each judge sees a different slice of the video:

| Judge | Default model | Input | Criteria (1–5 each) |
|---|---|---|---|
| Text | `azure/gpt-4o` | the subtitle text only | `accuracy_and_depth`, `logical_flow` |
| Video | `gemini/gemini-1.5-pro-002` | 10 equal-length clips, one at a time | `visual_consistency` |
| Image | `azure/gpt-4o` | 10 still frames, one at a time | `visual_relevance`, `element_layout` |

The **text judge** reads the `.srt` as plain text. If under 1% of its letters are capitals, the judge first rewrites it with punctuation (a step meant for auto-generated captions). It scores accuracy and depth (is the theorem explained correctly, and why it holds?) and logical flow (clear structure, coherent build-up).

The **video judge** scores each of 10 equal-length clips for visual consistency: consistent style, smooth motion and transitions. Its prompt is filled with the description only; the function receives the transcript path but never uses it.

The **image judge** samples one frame per second, splits them into 10 groups of `total // 10` frames, and keeps from each group the frame with the most non-black pixels (the visual fix's "busiest frame" again). Each still is scored for visual relevance (does it match the theorem's concepts?) and element layout (placement and size, unintended overlap, clarity).

::code eval_suite/image_utils.py:L28-L38 | one frame per second, 10 groups of total // 10 frames; the remainder at the end is never looked at

Integer division leaves a tail. A 125-second video gives 125 frames and groups of 12, so the last 5 seconds are never sampled by the image judge.

### From 32 scores to one number

That is 2 text scores, 10 clip scores and 20 frame scores (10 stills, two criteria each). Within each criterion, the clip or frame scores are combined by a **geometric mean**, the $n$-th root of their product. Then `calculate_overall_score` collects every remaining `score` field, skipping the per-chunk lists, and takes the geometric mean of those five criterion scores:

$$ \text{overall} = \Big(\prod_{c=1}^{5} s_c\Big)^{1/5} $$

::predict evaluator-metrics-judges
::worked evaluator-metrics-judges

A geometric mean punishes one weak criterion much more than an arithmetic mean does: four 5s and a 1 give 3.62, not 4.2. A video perfect on four criteria and judged 1 on layout lands 1.38 points below a perfect score; an arithmetic mean would have taken off only 0.8.

::code eval_suite/utils.py:L49-L58 | scores must be ints or digit strings; nothing checks that they lie between 1 and 5

Replies are parsed as JSON, and each `score` must be an integer or a string of digits; anything else (such as "4.5") raises, which the text and video judges retry (3 and 5 attempts) and the image judge, which has no retry loop, does not survive. The 1–5 range is only stated in the prompts. A judge that answered 0 would be accepted, and a single 0 makes every geometric mean it enters, including the overall score, 0.

::widget fixture:tea--eval-scores | drop one criterion to 1 and compare the geometric bar with the arithmetic one; follow each input to its judge: only the two text criteria take the transcript
::kp evaluator-metrics-judges

## What is never checked? {#blind-spots}
source: eval_suite/prompts_raw/__init__.py:L41-L88 · src/core/video_renderer.py:L53-L126 · generate_video.py:L746-L756 · 02-llm-education-systems-2024-2026.md:L5 · 02-llm-education-systems-2024-2026.md:L69

Put generation and evaluation side by side and ask, for each kind of fault, which stage could catch it.

| Fault | Generation | Evaluation |
|---|---|---|
| Code that crashes, bad LaTeX, missing import | caught: non-zero exit, sent to the fix loop | nothing to judge (no video) |
| Objects overlapping, outside the margins, off-screen | passes: exit code 0 | `element_layout`, a 1–5 opinion on 10 stills |
| Narration says one thing, picture shows another | passes | no judge sees both |
| Wrong mathematics in the picture | passes | `visual_relevance` on stills, at best |
| Does a learner understand it afterwards? | never asked | never asked |

**Geometry is never measured.** The 0.5 margin and 0.3 spacing appear in the outline, planning and code prompts, but no line of Python computes a bounding box, a gap or a distance to the frame edge. Nor does the evaluator: `element_layout` is an image model's opinion of a still, with no numbers in its criterion.

**Narration and picture are never compared.** The text judge is told that it has no access to the visuals and should "assume that there are reasonable visuals". The video judge never gets the transcript; the image judge gets one still and the topic name. A video whose narration describes a step the animation never shows can score well on every criterion.

::code eval_suite/prompts_raw/__init__.py:L65-L66 | the text judge is told not to consider the visuals at all

**No learner is involved.** None of the five scores asks whether anyone learned anything. Atlas's survey notes on the paper put it this way: TEA reported a success rate of about 94%, but layout problems appeared in most outputs and there was no learner study. Within the code, "success" can only mean what the pipeline counts: `--peek_existing_videos` reports how many topic folders contain a `_combined.mp4`, and a combined video exists exactly when every scene's `manim` run exited 0.

::code generate_video.py:L746-L756 | the success count is a count of combined video files

::predict evaluator-blind-spots

::note why | Where would a real layout check have to sit? Before the render, as code that asks each Manim object for its bounding box and compares it with the frame and with its neighbours, so a violation could go back into the fix loop like a crash does. The prompts ask the model to write such checks into its own scene; nothing guarantees it does, and nothing reads the result.
::note aside | Atlas's survey notes take one idea from the paper worth keeping, that a video "exposes reasoning flaws text hides", as a lens for reviewing material. The planner's ban on quizzes is the other contrast with Atlas: the generated video never asks the viewer to commit to a prediction before the answer appears.
::kp evaluator-blind-spots

## Where do the prompts and the theorems come from? {#prompts}
source: task_generator/parse_prompt.py:L30-L50 · task_generator/__init__.py:L28-L40 · external/TheoremExplainAgent-main/README.md:L268-L298 · generate_video.py:L893-L894 · src/config/config.py:L8-L9

Every prompt above has two copies. The source is a text file, `task_generator/prompts_raw/<name>.txt` (`eval_suite/prompts_raw/` for the judges). The code uses a Python constant in that folder's `__init__.py`, generated by `parse_prompt.py`, which writes `_<name> = """<file text>"""` for each `.txt`: `prompt_scene_plan.txt` becomes `_prompt_scene_plan`. Wrapper functions in `task_generator/__init__.py` (`get_prompt_scene_plan`, ...) fill the placeholders with `.format()`.

::code task_generator/parse_prompt.py:L42-L50 | each .txt becomes one triple-quoted constant named after its file
::code task_generator/__init__.py:L28-L40 | a wrapper: format the constant, return the string

The consequence is the README's instruction: after editing a `.txt` file, rerun `parse_prompt.py`. Until then the run uses the old constant. Ask the outline prompt for exactly 2 scenes in the `.txt` and change nothing else, and the planner still reads "between 3 and 7". The one exception is the visual self-reflection prompt, which `visual_self_reflection` reads straight from its `.txt` at run time.

::predict task-generation-prompts

Because the templates are format strings, a literal brace in a `.txt` must be doubled (`{{ }}`), hence the double braces in every JSON example. Of the 38 templates, 7 are never used by code outside their module.

The inputs are equally simple. A theorem file is a JSON list of records with the keys `theorem`, `description`, `difficulty`, `remark` and `subfield`; the repo ships `data/thb_{easy,medium,hard}/{math,physics,chemistry,comp_sci}.json` (20 records in `thb_easy/math.json`). The pipeline reads exactly two keys:

::code generate_video.py:L893-L894 | only theorem and description are read; a record with other key names fails with KeyError before any model call

The full benchmark, TheoremExplainBench, has 240 theorems and is hosted on Hugging Face, not in the repo (README L291–L298).

::note slip | `src/config/config.py` sets `THEOREMS_PATH` to `data/easy_20.json`, a file that does not exist in this copy. The CLI never uses that setting (`--theorems_path` defaults to none), so it is stale rather than harmful.
::kp task-generation-prompts

## Where can a run fail without telling you? {#silent}
source: mllm_tools/litellm.py:L154-L193 · src/core/video_planner.py:L166-L175 · generate_video.py:L934-L951 · src/rag/rag_integration.py:L95-L97 · src/core/code_generator.py:L137-L143 · external/TheoremExplainAgent-main/README.md:L317-L318

The last part cuts across all the stages. Several layers turn a failure into data that looks normal, and the next layer carries on.

It starts in the model wrapper. `LiteLLMWrapper.__call__` catches every exception from the API call and *returns the exception message* as if it were the model's reply.

::code mllm_tools/litellm.py:L186-L193 | no exception leaves the wrapper: an authentication error comes back as a string of "completion" text

Combine that with the outline's keep-everything fallback. Suppose `.env` has no API key and you run a single topic. The first planner call returns an authentication-error message; the outline regex finds no tags, so the message is saved as the outline. The scene count is 0, so nothing is planned or rendered. Then `combine_videos` calls `.group(1)` on its own failed outline search, with no guard, and raises `AttributeError`: a traceback about video combination, far from the missing key. With `--only_plan` there is no traceback at all.

::predict silent-degradation-points

The README's FAQ describes the symptom without the chain: missing plans or scenes "could be API-related issues".

::widget fixture:tea--outline-parse | load the error-text preset: the error string becomes the outline, 0 scenes are counted, and nothing raises until combine_videos

The RAG layer is inconsistent about failure:
- plugin detection catches everything and returns an empty list, so a broken helper silently means "no plugins";
- the storyboard, technical, narration and code query generators catch only `JSONDecodeError`. A reply with no `json` code fence (such as an error string from the wrapper) fails one step earlier, at `.group(1)`, with an `AttributeError` that stops the run;
- the error-fix query generator strips the fences first, so a non-JSON reply does return an empty list, and that fix proceeds with no documentation, after one printed line.

Two command-line paths are statically wrong. Both `--only_gen_vid` branches call `render_video_fix_code(topic, description, max_retries=...)`, but the method needs the outline and the list of plans as two more positional arguments.

::code generate_video.py:L934-L939 | single-topic --only_gen_vid: two arguments missing, and no await either

Calling a Python `async` function with missing arguments raises `TypeError` immediately, so both modes stop before anything renders. The single-topic branch would still do nothing even with the arguments supplied, because it never awaits the coroutine it creates.

::note aside | One more coupling, by reading the code: `generate_video_pipeline` reads the module-level `args` (for `--only_render`), so calling it from any entry point other than this script's `__main__` raises `NameError`.
::note why | Each fallback is locally reasonable: keep the reply if the tag is missing, keep going if retrieval fails. Together they move every failure downstream, away from its cause. Letting the wrapper raise, or treating a zero-scene outline as an error, would make the failures loud again.
::kp silent-degradation-points
