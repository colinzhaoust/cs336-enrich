---
title: edtrace · The tracer behind the executable lectures, read through
minutes: 35
---
Nine of CS336's seventeen lectures are Python files that you step through in a browser instead of slides. edtrace is the tool that makes that possible: it runs a lecture under Python's tracing hook, records every line as a step with the values and prose attached, and hands a JSON file to a small viewer. This read-through follows one lecture file through the whole pipeline, in the order things happen. After it you can predict what a trace will contain for any line of a lecture, read a trace file by hand, and say which claims in a published trace you should re-run rather than trust.

## What is edtrace, and what happens when you run it? {#entry}
source: README.md:L1-L30 · execute.py:L343-L387

Lectures 1, 2, 6, 7, 10, 12, 13, 14 and 17 of the course are **executable lectures**: a Python program replaces the lecture notes. The page you read on the course site is not a rendered document. It is a recording of that program running, which you replay one line at a time. Calls such as `text("...")` show up as prose in place of their line, and variables marked with a comment like `# @inspect x` appear in a side panel with their values at that moment. [L2](#/read/lecture_02) is the lecture where this matters most: every tensor shape and byte count it discusses is an `@inspect` value.

edtrace ("educational tracer", by Percy Liang) has two halves. The **backend** is a Python package that runs the file and writes a trace. The **frontend** is a React page (`TraceViewer.jsx`, about 950 lines) that loads the trace and lets you step through it. The two share nothing but the JSON format, so a trace can be made on one machine and viewed anywhere.

The README's whole example is a six-line lecture plus one command:

::code tools/edtrace/README.md:L13-L22 | a lecture in miniature: one inspected variable, one line of prose, one more inspected value
::code README.md:L24-L30 | one command runs the file and writes var/traces/hello.json

`python -m edtrace.execute` lands in the script block at the bottom of `execute.py`. It takes `-m` (one or more module names), `-o` (the output directory, default `var/traces`) and `-I` (inspect every local variable instead of only the marked ones, which the lectures never use). For each module it strips a stray `.py`, calls `execute()`, prints the number of steps, and dumps the trace as indented JSON.

::code execute.py:L370-L387 | the CLI: a loop over modules, one execute() and one json.dump each

The heart is the end of `execute()`, and most of edtrace's behaviour follows from the order of its lines:

::code execute.py:L343-L353 | import, register the one visible file, install the hook, run main(), uninstall, then package

1. **Import first, trace later.** `importlib.import_module` runs the module's top-level code before any hook is installed, so imports, constants and class definitions produce no steps. That is why every lecture puts its content inside a `main()` function.
2. **One visible file.** The module's own path goes into `visible_paths`, and nothing else ever does.
3. **Run for real.** `sys.settrace(trace_func)` installs the hook, `module.main()` executes the lecture, `sys.settrace(None)` removes the hook. Nothing is parsed or simulated: if `main()` downloads an image or multiplies two GPU matrices, that really happens.
4. **Package.** Only after the run does edtrace read the visible files back from disk, scan them for `@hide`, and build the `Trace` object that gets written.

I ran the README's `hello.py` through the local copy of edtrace (Python 3.11, nothing installed). The console printed one bracketed line per recorded step, `[0 hello.py:3] def main():`, then `[1 hello.py:4] x = 3  # @inspect x` with `env: x = Value(type='int', contents=3, ...)`, then lines 5 and 6, and finally `4 steps`. Four steps for three lines of body: the first one is the `def main():` line itself. The next section explains why.

## How does Python let a program watch itself run? {#settrace}
source: execute.py:L232-L245 · execute.py:L290-L341 · Python docs, sys.settrace

edtrace is built entirely on one interpreter hook, and never explains it, so here is the protocol.

`sys.settrace(f)` installs a **global trace function** for the current thread. Whenever a new frame starts (a function is called), the interpreter calls `f(frame, event, arg)` with event `'call'`. Whatever `f` returns becomes that frame's **local trace function**. From then on the interpreter calls the local function for the frame's own events:
- `'line'`, *before* each new line executes;
- `'return'`, when the frame is about to return;
- `'exception'`, when an exception is raised in it.

The local function's return value replaces it for the next event, and returning `None` switches local tracing off for that frame. At every call, `frame.f_locals` holds the frame's variables at that instant.

The key detail is timing. A `'line'` event fires before its line runs, so at that moment the line's results do not exist yet. They only become visible at the frame's *next* event, whether that is the next line or the return.

::predict supp-sys-settrace

edtrace splits its work along exactly that seam:
- `trace_func` is the global function. It runs when a line is about to execute, so it decides whether to record the line and opens a step for it, with the call stack but no values.
- It returns `local_trace_func`, a fresh closure that remembers that step. Python calls it at the frame's next event, by which time the line has finished, so it reads the values, attaches them to the step, and then calls `trace_func(frame, event, arg)` itself for that same event. That opens the step for the next line and returns a new closure.

So each event in a traced frame closes the previous line and opens the next one.

::code execute.py:L232-L245 | the docstring states the before/after split; the rules list at the end is terse
::code execute.py:L334-L341 | the closure hands control back to trace_func, which returns the next closure

One consequence surprises people. For a frame it does not care about, edtrace returns `trace_func` itself rather than `None`. That keeps the frame traced, so a helper with five plain lines still costs seven calls into edtrace: one `'call'`, five `'line'` and one `'return'`, each re-running the file filter before giving up.

The interactive below, used throughout this read-through, records a small template program with a line-for-line port of edtrace that was checked against real runs. In `lec.py`, `main()` loops `z = f(i)  # @inspect z` on line 6 and calls a function `f` defined in the same file. A checkbox swaps in `g`, an identical copy that lives in `helpers.py`. Other controls set the loop count, the length of the callee and the directives on line 6. It shows the files, the recorded steps, the viewer's keys and the raw settrace event log.

::widget fixture:edtrace--trace-run | follow one callee frame through the event log: one 'call', one 'line' per body line, one 'return', and edtrace's code runs at every one of them, even for a helpers.py frame that records nothing
::note why | Why the split? A tracer that only had the 'line' hook would see each line before it runs and could never show what a line produced. Reading values one event later is the only way, inside settrace, to see the result of a line.
::kp supp-sys-settrace

## Which frames become steps? {#filter}
source: execute.py:L215-L263

Every event in every frame reaches `trace_func`, including frames inside PyTorch, NumPy and the course's own helper modules. Four tests, in order, throw most of them away.

::code execute.py:L247-L263 | the file filter, the return filter and the comprehension filter, before anything is recorded

1. **Is the frame in the lecture file?** `frame.f_code.co_filename` must be in `visible_paths`, which holds exactly one path. Library code is never stepped, and neither is `lecture_util.py` or `references.py`, even though they live next to the lecture.
2. **What is the stack?** `get_stack()` walks Python's own call stack with `traceback.extract_stack()` and drops the frames of the runner and of edtrace itself by name.
3. **Is it a `'return'`?** Return events are ignored; a step means "this line is about to run".
4. **Is the innermost frame a list comprehension or a lambda?** Frames named `<listcomp>` or `<lambda>` are skipped, because, in the code's own comment, they are "redundant and just stay on the line".

::code execute.py:L215-L230 | the stack keeps every frame except seven named runner and tracer frames, innermost last

Notice what `get_stack()` does *not* do: it never filters by file. Once a frame passes test 1, its whole stack is stored, library frames included. The published `lecture_02.json` shows this. A step inside `Block.forward` has eight stack entries: `main`, `deep_network`, two PyTorch frames (`_wrapped_call_impl` and `_call_impl` in `torch/nn/modules/module.py`), `DeepNetwork.forward` at the `x = layer(x)` line, the same two PyTorch frames again, and finally `Block.forward`. The PyTorch path is stored as `../../usr/local/lib/python3.11/site-packages/...`, relative to the directory the trace was made in. So the file itself tells you it was produced in a container under Python 3.11.

::predict execute-settrace-loop

A concrete consequence: when the professor moves a helper into `lecture_util.py`, its lines silently vanish from the trace. Its *effects* do not vanish. A value it returns can still be inspected on the calling line, and any prose it emits lands on that line (see [renderings](#renderings)).

::widget fixture:edtrace--trace-run | tick "line 6 calls g from helpers.py": every g row in the event log ends at "not in visible_paths", and no step ever has a second stack entry
::note warning | The comprehension filter is narrower than its comment suggests. Only the names `<listcomp>` and `<lambda>` are tested, and only for the innermost frame. Dict comprehensions are recorded, and by the same test so are set comprehensions and generator expressions: the published lecture_14.json has 49 stack entries named `<dictcomp>`, and lecture_01.json records a step on the line that builds the byte vocabulary with one. And a function called *from* a list comprehension is recorded with the `<listcomp>` frame in its stack, as lecture_02.json does for each `Block(dim)` built inside `nn.ModuleList([...])`.
::note aside | The `<listcomp>` test only matters on Python 3.11. From 3.12 on, list comprehensions run inline in the enclosing frame, with no frame of their own. edtrace requires Python 3.11 or later. The published traces of lectures 1, 2, 7 and 14 contain comprehension frames, so they were made on 3.11; lecture_02.json also names python3.11 in its paths.
::kp execute-settrace-loop

## What is a step, and why do some lines get two? {#steps}
source: execute.py:L280-L341

A recorded line becomes a `Step`: the stack, an `env` dictionary of inspected values, and a list of renderings. The step is created in two moves, one on each side of the line.

::code execute.py:L280-L303 | open a step before the line (unless it repeats the previous stack); at close, reuse it or append a second one

**Open.** Before the line runs, `trace_func` builds a step with the current stack and an empty env, and appends it unless its stack is identical to the previous step's. The line number is part of each stack element, so the only steps merged are exact repeats of the same location.

**Close.** After the line runs, `local_trace_func` asks one question: is the step it opened still the last step in the list?
- **Yes.** Nothing was recorded in between, so it fills env and renderings into that same step. One line, one step.
- **No.** The line called a function in the lecture file, and that function's lines were appended in between. So it appends a *second* step for the same line, with the same stack, and fills that one.

I traced this small file locally to see it:
- `square(v)` (lines 3–5) computes `s = v * v` and returns it;
- `main()` (lines 7–10) runs `for i in range(2): z = square(i)  # @inspect z`, then `text("done")`.

The console listed fifteen steps:

| steps | line | what happened |
|---|---|---|
| 0 | 7 | `main`'s `'call'` event, on its `def` line |
| 1 | 8 | the `for` line, first iteration |
| 2 | 9 | open step for `z = square(i)`, empty env |
| 3, 4, 5 | 3, 4, 5 | `square`'s `def` line, its body, its `return` |
| 6 | 9 | close step for the same line, now with `z = 0` |
| 7–12 | 8, 9, 3–5, 9 | the same pattern again, ending with `z = 1` |
| 13 | 8 | the `for` line once more, when `range(2)` runs out |
| 14 | 10 | `text("done")` |

::predict step-open-close

Why keep the empty open step? Because the viewer needs somewhere to stand *before* the call. From step 2 you can step into `square` or step over it, and the highlighted line tells you which call is about to happen. The close step is where the result lives. A line that calls nothing traced gets just one step, because its open and close coincide. That includes a call into `lecture_util.py`.

::widget fixture:edtrace--trace-run | with the callee in lec.py, line 6 gets an open step with empty env and a close step carrying z; switch the callee to helpers.py and the two collapse into one
::note aside | The `def` line steps (0 and 3 above) come from `'call'` events: the frame starts, and its first reported location is the definition line. They are why hello.py had four steps for three body lines.
::kp step-open-close

## How do text(), image() and link() reach the page? {#renderings}
source: execute_util.py:L1-L159

In an executable lecture, the prose is code. Each `text("...")` call is a bullet of what would have been a slide, and the viewer shows it *in place of* the line that made it. The mechanism is a single module-level list.

::code execute_util.py:L22-L36 | a Rendering is a type, a data string, a style, and optionally an external or internal link

Every rendering function appends one or more `Rendering` records to `_current_renderings`. When a traced line closes, `local_trace_func` calls `pop_renderings()`, which copies the list into the step and empties it.

::code execute_util.py:L141-L148 | one global list, emptied into whichever step closes next

The functions differ only in what they append:
- **`text(message)`** appends one `markdown` rendering. With `verbatim=True` it splits on newlines and appends one rendering per line, styled monospace with preserved whitespace. There is no separate heading type: `text("# Heading")` is markdown that the viewer turns into a heading.
- **`image(url)`** appends an `image` rendering whose data is a *local path*. A URL is downloaded at trace time into `var/files/` (see [references](#references) for the cache), and a missing local file raises an error that aborts the run. `video()` works the same way.
- **`link(...)`** appends a `link` rendering. It has four forms, covered in the next section.
- **`plot(spec)`** carries a Vega-Lite chart specification. **`note(message)`** is a side note that the viewer shows only when notes are switched on with the `N` key.
- **`system_text(command)`** runs a shell command at trace time and emits its output as verbatim text.

::code execute_util.py:L40-L54 | verbatim mode emits one markdown rendering per line, with a monospace style merged in

Because the list is global, it does not matter *who* calls `text()`. The course's `lecture_util.py` has helpers such as `article_link(url)`, whose whole body is `link(title="article", url=url)`. Calling it contributes no steps, because its file is invisible, but the link it appends waits in the list and lands on the lecture line that called it. Several calls on one line all land on that line's step, in order.

::predict renderings-emit

For scale: the published `lecture_02.json` carries 210 markdown, 18 link and 11 image renderings across its 955 steps. In the viewer, a line that has renderings shows them instead of its code (press `R` for raw mode to see the code again).

::widget fixture:edtrace--trace-run | toggle text() in f, then move f to helpers.py or put @stepover on line 6: the "f got" text stops landing on f's line and lands on line 6's step instead
::kp renderings-emit

## Where do the citations come from? {#references}
source: execute_util.py:L94-L126 · reference.py · arxiv_util.py · file_util.py

In the viewer each citation is a short label like `[Hoffmann+ 2022]` with a hover card of title, authors, date and abstract, built at trace time from a scraped web page.

::code execute_util.py:L102-L126 | four forms of link(): keyword fields, an existing Reference, a function or class, or a URL string

The four forms of `link()`:
1. `link(title=..., url=...)` builds a new `Reference` from keyword fields.
2. `link(ref)` shows an existing `Reference`. The course keeps many of them in `references.py`.
3. `link(some_function)` is an *internal* link. It records the file and the `def` line, and the viewer jumps there when clicked.
4. `link("https://...")` routes the string through `url_reference`.

A `Reference` is a frozen record of title, authors, organization, date, url, description and notes. Its `label` property makes the anchor text:
- **Authors present:** the first author's last word, a `+` if there are more authors, and the year. A first author containing the word "Team", such as "Kimi Team", is kept whole and gets no `+`.
- **Otherwise:** the title; failing that, the URL; failing that, `?`.

::code reference.py:L15-L33 | the label: [LastName+ YYYY], with the Team exception and three fallbacks

`url_reference` scrapes only one kind of URL: one that starts with exactly `https://arxiv.org/`. For those, `arxiv_reference` pulls the paper id out with a regular expression and downloads the paper's abstract page. It accepts `abs/` and `pdf/` URLs, an optional version suffix and an optional `.pdf`. It then reads the page's HTML with BeautifulSoup:
- the title from the `citation_title` meta tag;
- the authors from the links in the authors block;
- the date from `citation_date`;
- the abstract from `og:description`.

No arXiv API is involved. Any other URL becomes a bare `Reference(url=...)`, labelled by the URL itself.

::code arxiv_util.py:L18-L53 | id regex, cached download of the abs page, four fields scraped from meta tags and the authors div

::predict references-arxiv

Every download, images included, goes through `cached()`. It names the file `var/files/<prefix>-<md5 of the URL>-<the URL with non-word characters replaced by _>` and downloads it only if that file is missing. The hash keeps names unique, and the readable tail lets a person find a file by eye. The downloader retries an HTTP 429 (rate limited) response up to 50 times, waiting 1, 2, 4… seconds and doubling each time. The course's own run log for lecture 7 shows one of these downloads happening: `Downloading https://media.springernature.com/... to var/files/image-b0641f11...`.

::code file_util.py:L33-L41 | the cache key: prefix, md5 of the URL, sanitized URL

::note warning | The prefix test is literal. `link("http://arxiv.org/abs/2104.09864")`, with http instead of https, is not scraped. Its label is the raw URL, not `[Su+ 2021]`.
::note aside | `label` is a property, not a field, so it is not part of the serialized Reference. The trace carries it only as the rendering's data string, which the viewer uses as the anchor text. The viewer has its own copy of the label rule (getReferenceAnchorText), used only when data is empty.
::kp references-arxiv

## How is a # @inspect comment read? {#directives}
source: execute.py:L73-L137 · execute.py:L305-L332

The lecture author controls the trace with comments. There are four directives:
- `@inspect` shows a variable's value and updates it;
- `@clear` stops showing it;
- `@stepover` records the line but not the calls beneath it;
- `@hide` keeps the line out of the viewer.

Comments are invisible to Python, so edtrace reads them as text. It takes the source line that `traceback` reports for the frame (stored as the stack element's `code`) and hands it to `parse_directives`.

::code execute.py:L88-L109 | split once on '#', then on whitespace; an @token opens a directive and later tokens are its arguments

The grammar is four rules:
1. Only the text between the first `#` and the next `#` is read.
2. It is split on whitespace.
3. A token starting with `@` opens a new directive. An unknown one, such as a typo like `@inpsect`, only prints a warning.
4. Every other token becomes an argument of the most recent directive. Tokens before the first `@` are dropped.

So `# @inspect x y` and `# @inspect x @inspect y` mean the same thing, and the lectures can mix prose into the comment. [L2](#/read/lecture_02) writes `z = x @ y   # seq1 seq2 @inspect z`. The words `seq1 seq2` annotate the shapes for the human reader, and the tracer drops them. The viewer, for its part, strips everything from the first `@` in the comment when it displays the line, so the reader sees `# seq1 seq2`.

After the line runs, `local_trace_func` turns the directives into env entries:

::code execute.py:L305-L332 | look each name up in f_locals, follow dotted attributes with getattr, serialize; then write None for @clear

- A plain name is looked up in `frame.f_locals`. A missing name prints `WARNING: variable ... not found in locals` and records nothing.
- A dotted name such as `w.grad` splits at the first dot, finds `w`, then follows `getattr` for each remaining part. The env key stays the full string `"w.grad"`. In the published `lecture_02.json`, the line `assert torch.equal(w.grad, torch.tensor([1, 2, 3]))  # @inspect w.grad` carries exactly that key, holding a float32 tensor `[1.0, 2.0, 3.0]`.
- `@clear` names are set to `None`, which becomes `null` in JSON. L2's `text("Let's try a more complex example...")  # @clear x y z` produces the env `{"x": null, "y": null, "z": null}`, and the viewer removes those three from its panel.

::predict directive-grammar

Two more uses show the timing rule at work:
- `@inspect` can sit on a `def` line. [L7](#/read/lecture_07) writes `def collective_operations_main(rank: int, world_size: int):  # @inspect rank world_size`. The `def` line's step is opened at the `'call'` event and closed at the first body line, when the arguments are already locals. The published trace shows `rank = 0` and `world_size = 4` on that step.
- Values are read *after* the line, so `x += 1  # @inspect x` shows the new value. In hello.py that is 4, not 3.

::widget fixture:edtrace--directive-parse | type a line and see which tokens are dropped, which open a directive and which become arguments; a second '#' cuts the rest off
::note warning | Because the split is on the first `#` anywhere in the line, `text("# Heading")  # @inspect x` yields no directives at all: the parsed text is ` Heading")  `. And because only the line that Python reports is read, a directive written on the continuation line of a multi-line statement is never seen by the recorder. Only `@hide`, which scans every line of the file (see [pruning](#pruning)), notices it.
::kp directive-grammar

## What does an inspected value look like in the file? {#values}
source: execute.py:L32-L46 · execute.py:L147-L200

JSON cannot hold a tensor, a datetime or a dataclass, so every inspected value passes through `to_serializable_value`, which wraps it as a `Value`: a `type` string, the `contents`, and for arrays a `dtype` and a `shape`.

::code execute.py:L147-L192 | an ordered type dispatch: primitives, numpy scalars, arrays and tensors, sympy, containers, asdict, and str() as the last resort

The branches, in order (the first match wins):
- `bool`, `int`, `float` and `str` pass through. A float `nan` or `inf` becomes the string `"nan"` or `"inf"`, because JSON has no such numbers.
- `np.int64` and `np.float64` become plain Python numbers.
- `np.ndarray` and `torch.Tensor` record `str(dtype)`, `list(shape)` and the *complete* `tolist()`.
- SymPy integers and floats become numbers; other SymPy expressions become their string.
- Lists, tuples, dicts and dataclasses recurse. Dict keys that are not primitives are turned into strings.
- An object with an `asdict()` method is serialized through it.
- Anything else becomes `str(value)`.

The `type` string is `module.ClassName`, except for builtins, which are just `int`, `str` and so on.

::worked inspect-serialization

A real one from the published L2 trace: the line `x = torch.zeros(4)        # rank 1 tensor (vector) @inspect x` closes with the env `{"x": {"type": "torch.Tensor", "contents": [0.0, 0.0, 0.0, 0.0], "dtype": "torch.float32", "shape": [4]}}`. The words before `@inspect` were dropped, as the grammar says.

The most consequential property of this function is something it lacks: **there is no size limit anywhere**. A tensor is always dumped in full. L2's GPT-3 feedforward matrix, 49152 × 12288, would be about 604 million numbers if anyone inspected it. Nobody does. The lectures inspect small tensors and derived numbers such as byte counts, and that discipline is the only thing keeping trace files small.

::widget fixture:edtrace--value-render | grow the shape and the stored number count grows with prod(shape): the record never switches to a summary, and dtype and shape sit beside the full list
::note warning | Only the 64-bit numpy scalars are special-cased. An `np.float32` or `np.int32` scalar, which is what you get by indexing a float32 or int32 array, falls through to `str(value)`, so it is stored and shown as a string.
::kp inspect-serialization

## How do @stepover and @hide prune the trace? {#pruning}
source: execute.py:L265-L278 · execute.py:L356-L366

Lectures often call something whose insides are not the point, such as building a model or a timing helper. Two directives prune such lines, at two different times.

**`@stepover` acts while recording.** `execute()` keeps a list called `stepovers` of (file, line) locations. When `trace_func` reaches a line carrying `@stepover`, it checks the *last* entry of that list:
- if the last entry is this same location, it pops it ("we're back to this line");
- otherwise it pushes the location.

Then, for every event, it skips the event if any stepover location appears among the *callers* of the current frame (`stack[:-1]`). The line itself is still recorded, with its `@inspect` values. Only what runs beneath it is not.

::code execute.py:L265-L278 | push on reaching the line, pop on reaching it again with it on top; skip anything whose callers include a pushed location

The pop happens when the line is *reached again*, not when its call returns. That design has a visible effect in loops. Take a `@stepover` line inside a `for` loop:
- the first iteration pushes it, and its callee is skipped;
- the second iteration finds it on top and pops it, so its callee is recorded;
- the third iteration pushes it again, and so on.

This was first predicted from reading the code, and running edtrace confirmed it: the callee is recorded on iterations 2 and 4 of a four-iteration loop. The published `lecture_02.json` shows it in the wild. `DeepNetwork.forward` loops `x = layer(x)  # @stepover` over three layers. In the first forward pass, `Block.forward`'s lines appear only on the second pass through the loop, between two steps on line 598.

::predict stepover-hide

A line that is reached only once is pushed and never popped. That stale entry is harmless until something else checks the top of the list, and [L7](#/read/lecture_07) has a case where it matters. Its `spawn()` helper runs `with DisableDistributed():  # @stepover` (shown in [limits](#limits)). Under Python 3.11 a `with` line is reported twice, once on entry and once when `__exit__` is called, so when nothing else is on the list the pair pushes and pops: `__enter__` is skipped but `__exit__` is recorded. In the published `lecture_07.json` that happens for the first `spawn` only, leaving 132 steps of `__exit__` restoring patched functions one by one.

From the second `spawn` on, the function being spawned has its own `setup(...)  # @stepover` and `cleanup()  # @stepover` lines. Each is reached once and left on the list, so when the `with` line comes back, the top entry is `cleanup`'s, not its own. The `with` line pushes again instead of popping, and `__exit__` is skipped. I reproduced the asymmetry with a small file: a single-visit `@stepover` call inside the `with` body turns `__exit__` off for that call and on for the next one.

`@hide` acts after recording. Once `main()` has finished, `compute_hidden_line_numbers` scans each saved file's text, line by line, and lists every line carrying `@hide` in `hidden_line_numbers`. No step is removed. The viewer simply does not draw those lines, and stepping passes through their steps without showing them.

::code execute.py:L356-L366 | @hide is a text scan of the saved file, after execution; it never touches steps

The demo file `examples.py` uses `@hide` on the *continuation* line of a two-line `text(...)` call. The line Python reports for the statement, and so the one that shows the rendering, is the first; the dangling second half would otherwise be drawn as a stray line of code.

::widget fixture:edtrace--trace-run | turn on @stepover with n ≥ 2 and the callee's steps return on every second iteration; turn on @hide instead and no step changes, only the listing strikes the line out
::note aside | The lectures lean on @stepover heavily: L1 on its BPE merge and tokenizer calls, L2 on model construction and `get_num_parameters(model)  # @inspect num_parameters @stepover`, and L7 on every setup, cleanup and per-rank print. None of the published traces uses `@hide`: their hidden_line_numbers lists are all empty.
::kp stepover-hide

## What file comes out? {#json}
source: execute.py:L17-L70 · execute.py:L350-L352 · execute.py:L384-L387

After `main()` returns, `execute()` builds a `Trace` and the CLI writes `asdict(trace)` to `var/traces/<module>.json` with two-space indentation. The format is three dataclasses deep:

::code execute.py:L17-L70 | StackElement, Value, Step, Trace: the entire contract between recorder and viewer

- **`files`**: a map from each visible file's path to its *full source text*. In practice that is one entry, the lecture.
- **`hidden_line_numbers`**: a map from path to the `@hide` line numbers.
- **`steps`**: a list. Each step has:
  - `stack`, a list of `{path, line_number, function_name, code}` with the innermost frame last;
  - `env`, a map from expression to a `Value` or `null`;
  - `renderings`, a list of `{type, data, style, external_link, internal_link}`, with the link objects nested in full.

There is no "current line" field. The current line is the last element of `stack`, and the nesting depth is the stack's length.

Here is the second step of the hello.py trace I produced, exactly as written: `{"stack": [{"path": "hello.py", "line_number": 4, "function_name": "main", "code": "x = 3  # @inspect x"}], "env": {"x": {"type": "int", "contents": 3, "dtype": null, "shape": null}}, "renderings": []}`. The step for line 5 is the mirror image: an empty env, and one rendering `{"type": "markdown", "data": "Welcome!", "style": {}, "external_link": null, "internal_link": null}`.

::worked trace-json-schema

Two properties of this format matter in practice.

**The trace is self-contained.** The viewer reads the lecture's lines from `files`, never from disk, so a trace still shows the code it was made from after the lecture file changes. That has already happened in the course: the source embedded in the published `lecture_02.json` differs from the `lecture_02.py` now in the repository on three lines, for example `num_train_steps = 10` against `3`. The only things not embedded are the images and videos, which are stored as `var/files/...` paths and must be served next to the trace.

**Paths are relative to where you ran the command.** `relativize()` turns every path into one relative to the working directory, so a trace made from another directory has different keys in `files` and in every stack. This is also where the `../../usr/local/lib/python3.11/...` paths of PyTorch frames come from.

For scale, the published traces run from 235 steps (lecture 17, 187 KB) to 955 steps (lecture 2, 704 KB). Lecture 1's 617 steps take 986 KB.

::widget fixture:edtrace--trace-run | step into the callee and read the raw steps[i] record: it has two stack entries, the caller's line first, and the highlighted line is the last one
::kp trace-json-schema

## How does the viewer walk through the steps? {#viewer}
source: TraceViewer.jsx:L10-L20 · TraceViewer.jsx:L209-L428 · TraceViewer.jsx:L578-L600

The viewer is a single React component, and its state lives in the URL: `?trace=lecture_02&step=44`. A bare name is expanded to `var/traces/<name>.json`. Other parameters switch modes: `raw` (show code instead of renderings), `animate`, `hideEnv` and `showNotes`. The alternative pair `source` and `line` selects a line instead of a step. Because the state is the URL, every position in a lecture is a shareable link.

The keys:
- **`l` / `h`** (or the arrow keys) step forward and back by one index. This goes into every traced call.
- **`j` / `k`** (shift-arrows) step over: they stay at the current level.
- **`u`** steps up and out of the current function.

None of these looks at function names. Everything is computed from stacks:
- **`inSameFunction(a, b)`** is true when the stacks have equal length and agree on the path and line of every element except the last.
- **`isStrictAncestorOf(a, b)`** is just `a.length < b.length`.

::code TraceViewer.jsx:L244-L275 | step over: the next step in the same function, or the first one that escaped to a shorter stack; step up: the first shorter stack outside this function

From an open step of a line that calls a traced function, `j` skips the callee's steps, which have longer stacks, and lands on the close step, which has the identical caller chain. This is the second half of why edtrace records the open/close pair: it gives step-over a place to land on either side of a call.

::predict viewer-stepping-model

**The variables panel.** Each step stores only the values inspected on *its own* line, so the panel has to rebuild the picture. `renderEnv` walks backwards from the current step:
- steps in the same function contribute their env;
- steps of called functions, with longer stacks, are passed over;
- the walk stops at the first strictly shorter stack, the caller.

It then merges the collected envs oldest first and deletes every key whose value is `null`, which is how `@clear` takes effect.

::code TraceViewer.jsx:L355-L402 | merge env backwards through same-function steps, stop at an ancestor, drop nulls

**Renderings are not step-dependent.** Before drawing, the viewer builds `lineNumberToRenderings` by looping over *all* steps and keying by line number, so the last write wins. A line that ran several times shows its last run's prose at every step, and lines after the current step already show their prose. The lines below the current one are hidden only in animate mode (`A`). That mode "cloaks" every line that no step up to the current one has reached, and reveals the lines above each reached line back to the nearest unindented one. Hidden lines from `@hide` are skipped when drawing, whatever the step.

::code TraceViewer.jsx:L578-L582 | one map over the whole trace, keyed by line number only

::widget fixture:edtrace--trace-run | stand on line 6's open step and press j: you land on the close step; inside the callee the env panel merges only steps with matching caller entries and stops at the first shorter stack
::note warning | Every step of `main` has a one-element stack, and the comparison skips the last element, so all of them count as "the same function". On a lecture's top level the panel therefore accumulates every value inspected in `main` so far, not just the recent ones. The key is the caller's path and line, not the function name: the same helper called from two different lines counts as two different functions.
::kp viewer-stepping-model

## How does the viewer draw a value? {#drawing}
source: TraceViewer.jsx:L434-L553

The variables panel turns each `Value` back into something readable. `renderValue` dispatches on the `type` string that the backend's `get_type_str` produced, so the two files have to agree on spellings such as `torch.nn.parameter.Parameter`:
- `NoneType` shows `None`, and `bool` shows `true` or `false`;
- `int` and `float` go through `renderNumber`;
- `torch.Tensor`, `torch.nn.parameter.Parameter` and `numpy.ndarray` go through `renderTensor`;
- SymPy values show their string;
- anything else is drawn by the shape of its contents: an array as a one-row table of recursively rendered items, an object as a key : value table, and the rest as JSON text.

::code TraceViewer.jsx:L434-L464 | the dispatch on Value.type; the tensor branch lists its three type strings explicitly

`renderNumber` applies four tests in a fixed order and uses the first that passes:
1. a string, such as `"nan"`, passes through unchanged;
2. a magnitude above 10¹² gets exponent notation with three decimals, so 3.5e13 shows as `3.500e+13`;
3. a magnitude above 10⁶ gets thousands separators;
4. a value that is an integer after multiplying by 1000, meaning at most three decimals, prints exactly, so 0.125 stays `0.125`. Anything else is rounded to four decimals, so 2.718281828 shows as `2.7183`.

::code TraceViewer.jsx:L466-L481 | four ordered tests; only long fractions are rounded

::predict viewer-value-rendering

`renderTensor` picks a layout from the *rank alone*:

| rank | drawn as |
|---|---|
| 0 | a single number |
| 1 | one table row |
| 2 | a table |
| 3 | its slices stacked vertically, with a blank separator row before every slice after the first |
| 4 and up | JSON text |

The dtype plays no part in the layout. It appears only in the hover title, which reads like `torch.Tensor torch.float32 [2 x 3]`.

::code TraceViewer.jsx:L483-L520 | layout by len(shape): scalar, row, table, stacked slices, JSON

::widget fixture:edtrace--value-render | try 0.125, 2.718281828 and 3.5e13 against the four tests; then give a tensor shape [3, 2, 4]: 6 data rows plus 2 separators, whatever the dtype
::note aside | A Python string has type `str`, matches no branch, and reaches `JSON.stringify`, so it is shown with quotation marks. In the published L2 trace, `properties.name` is displayed as `"NVIDIA H100 80GB HBM3"`, quotes included.
::kp viewer-value-rendering

## What can a trace not capture? {#limits}
source: execute.py:L343-L348 · execute_util.py:L151-L154 · lecture_07.py:L557-L591

Everything above adds up to one sentence: a trace is a recording of **one run, on one machine, of one file**. Each part of that sentence is a limit.

**One run, and it must finish.** There is no `try` around `module.main()`, no per-line error capture and no incremental saving. If any line raises an exception, for example a CUDA call on a laptop without a GPU or a failed download, the exception escapes `execute()` and the JSON is never written. A partial trace does not exist.

::code execute.py:L343-L348 | main() runs bare: an exception here means no trace file at all

**One machine, frozen.** Whatever varied between runs is baked in as it happened that time: random tensors, `system_text(["date"])`, timings, downloaded pages, and hardware queries. In the published `lecture_02.json`, `torch.cuda.get_device_properties(...)  # @inspect properties.name` recorded `NVIDIA H100 80GB HBM3`. The run used to make that file is not the one shown in class: as the [L2 read-through](#/read/lecture_02) notes, the live lecture ran on the professor's laptop. A trace's numbers belong to whatever machine produced it, so claims about speed or memory are better checked by re-running the lecture than by reading its trace.

::code execute_util.py:L151-L154 | system_text runs a real subprocess at trace time; its output is frozen into the trace

**One file, one process.** `sys.settrace` installs a hook for the current thread of the current process. A new process started with `torch.multiprocessing.spawn` is a fresh interpreter with no trace function, so nothing it does could be recorded. Its prints go to its own stdout, and its renderings go to its own copy of the list. [L7](#/read/lecture_07), the distributed-training lecture, needs four processes talking to each other, so it detects the tracer and changes its behaviour:

::code lecture_07.py:L557-L591 | sys.gettrace() is non-None only under the tracer; then rank 0 runs alone inside DisableDistributed

When `sys.gettrace()` returns nothing (a plain `python lecture_07.py`), `spawn` really starts `world_size` processes. Under edtrace it instead calls the function once, in-process, as rank 0, inside `DisableDistributed`. That context manager replaces every function in `torch.distributed` with `lambda *args, **kwargs: None`. Since those lambdas live in the lecture file, they are `<lambda>` frames, which the tracer skips. So the traced lecture walks through every line of the collective-operations code, but every all-reduce is a no-op on a single tensor. The trace cannot show what a collective computed. The lecture links instead the saved stdout of a real four-process run, where every rank prints `[6, 10, 14, 18]` after the all-reduce.

**No size control.** As the [values](#values) section showed, tensors are stored in full at every step that inspects them. Loops add a step per line per iteration. The source is embedded, and stacks repeat their full caller chain in every step. Inspecting a 4096 × 4096 tensor inside a 10-iteration loop would store about 168 million numbers. The published traces stay under a megabyte only because the authors inspect small things.

::predict limits-what-trace-cannot-capture

**Not stepped at all:**
- module-level code, which runs before the hook is installed;
- every file other than the lecture;
- list-comprehension and lambda bodies;
- `'return'` events;
- anything beneath a `@stepover` line while its location is on the list.

Values computed there can still surface through `@inspect` on a lecture line.

::widget fixture:edtrace--value-render | set the shape to 4096 × 4096 and the recording steps to 10: the stored numbers grow with both, because every recorded step keeps the full tolist()
::widget fixture:edtrace--trace-run | with g in helpers.py, line 6 still gets one step per iteration: the step count grows with n though nothing inside g is recorded
::note why | Why accept these limits? Because the alternative, a static outline of the code, could not show a single computed value, and computed values are the point of an executable lecture. edtrace gives up reproducibility in exchange: what you see really happened, once.
::kp limits-what-trace-cannot-capture
