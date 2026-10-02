// Widgets for thread tea (TheoremExplainAgent, local copy external/TheoremExplainAgent-main; see repos/tea/sources.json).
// Register as "fixture:<id>" -> (root, notice) => void. Each widget has a pure model in MODELS (no DOM) that
// tools/check_widgets.mjs tests against the KPs' stored answers.
// Every rule a model applies is a line of the repo, cited as <file>:L<n>. The models re-implement control flow, regexes
// and arithmetic; they never run the repo. Inputs a learner types (flags, planner replies, per-version outcomes, judge
// scores, video length) are the learner's, not data from the repo.
import { el, fmt, slider } from "../widgets_lib.js";

export const MODELS = {};
export const WIDGETS = {};

const C = { ink: "var(--ink)", muted: "var(--muted)", rule: "var(--rule)", a: "var(--accent)", b: "var(--accent2)", ok: "var(--ok)", hi: "var(--hilite)" };
const esc = s => String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
const rect = (x, y, w, h, fill, extra = "") => `<rect x="${x.toFixed(1)}" y="${y.toFixed(1)}" width="${Math.max(0, w).toFixed(1)}" height="${Math.max(0, h).toFixed(1)}" style="fill:${fill}" ${extra}/>`;
const text = (x, y, s, o = {}) => `<text x="${x.toFixed(1)}" y="${y.toFixed(1)}" style="fill:${o.fill || C.ink};font:${o.weight || ""} ${o.size || 12}px ${o.mono ? "var(--mono)" : "var(--sans)"}" text-anchor="${o.anchor || "start"}">${esc(s)}</text>`;
const line = (x1, y1, x2, y2, stroke, extra = "") => `<line x1="${x1.toFixed(1)}" y1="${y1.toFixed(1)}" x2="${x2.toFixed(1)}" y2="${y2.toFixed(1)}" style="stroke:${stroke};stroke-width:1.5" ${extra}/>`;
const svg = (w, h, body) => `<svg viewBox="0 0 ${w} ${h}" width="100%" style="max-width:${w}px;border:1px solid var(--rule);border-radius:6px;background:#fff" role="img">${body}</svg>`;
const check = (label, value, onChange) => { const i = el("input", { type: "checkbox" }); i.checked = value; i.addEventListener("change", () => onChange(i.checked)); return el("label", {}, i, ` ${label}`); };
const prov = s => `<span class="muted small">${s}</span>`;

// ======================================================== 1. CLI dispatch ==
// generate_video.py __main__ (L667-L954). Before any branch, three LiteLLMWrappers are built at temperature 0.7:
//   planner and scene from --model, helper from --helper_model if given else --model (L714-L735, L722).
// Then one if/elif/else over the input mode (L738, L917, L952); every mode flag is read only inside the branch that tests it.
// Batch branch (L738-L915): theorems = json.load(file); if --sample_size: theorems[:k] (L743-L744, a head slice, so min(k, n));
//   --peek_existing_videos prints two counts and exits (L746-L769); --debug_combine_topic combines that topic and exits (L788-L790);
//   --only_gen_vid calls render_video_fix_code(topic, description, max_retries=...) with 2 of its 4 required positional
//   parameters (L800 vs L291-L297): TypeError at the call, nothing renders; elif --check_status prints the P/C/R table and exits
//   (L810-L885); else per theorem: --only_combine -> combine_videos only (L896-L897), otherwise generate_video_pipeline(only_plan,
//   specific_scenes=args.scenes) then combine_videos unless --only_plan or --only_render (L899-L907).
// Topic branch (L917-L951, needs --topic AND --context): --only_gen_vid calls the async render_video_fix_code without await and
//   exits (L937-L939): a coroutine is created and dropped, nothing renders; --only_combine -> combine only (L941-L942); otherwise
//   generate_video_pipeline(only_plan) (no --scenes passed, L944-L949) then combine unless --only_plan or --only_render (L950-L951).
// --only_render is read inside generate_video_pipeline through the module-global args (L561, L584), so it acts in both branches.
// The shipped data/thb_easy/math.json has 20 entries (repos/tea/knowledge_points.json, task-generation-prompts filtering note).
const CLI_FLAGS = [
  ["theorems_path", "--theorems_path data/thb_easy/math.json"], ["sample_size", "--sample_size"], ["topic", "--topic"], ["context", "--context"],
  ["peek", "--peek_existing_videos"], ["debug_combine", "--debug_combine_topic"], ["only_gen_vid", "--only_gen_vid"], ["check_status", "--check_status"],
  ["only_combine", "--only_combine"], ["only_plan", "--only_plan"], ["only_render", "--only_render"], ["scenes", "--scenes"], ["helper_model", "--helper_model"],
];
function cliDispatch(a) {
  const f = k => !!a[k];
  const nFile = a.n_file ?? 20, rows = [], read = new Set(["helper_model"]);
  const row = (ln, code, st, depth = 0) => rows.push({ ln, code, st, depth });   // st: "true" | "false" | "skip"
  const o = { branch: 3, nTopics: 0, pipeline: 0, combine: 0, renders: 0, crash: 0, helperFallback: f("helper_model") ? 0 : 1, outcome: "" };
  const batch = f("theorems_path"), topicMode = !batch && f("topic") && f("context");
  // --- batch branch
  row("L738", "if args.theorems_path:", batch ? "true" : "false");
  let n = 0, stop = !batch;
  if (batch) { read.add("theorems_path"); read.add("sample_size"); n = a.sample_size > 0 ? Math.min(a.sample_size, nFile) : nFile; }
  row("L743", "if args.sample_size: theorems = theorems[:k]", !batch ? "skip" : a.sample_size > 0 ? "true" : "false", 1);
  const step = (ln, code, key, depth = 1) => { if (stop) { row(ln, code, "skip", depth); return false; } read.add(key); const t = f(key); row(ln, code, t ? "true" : "false", depth); if (t) stop = true; return t; };
  if (step("L746", "if args.peek_existing_videos: print counts; exit()", "peek")) { o.branch = 1; o.outcome = "prints how many topic folders hold <prefix>_combined.mp4 and how many scene folders hold succ_rendered.txt, then exits; nothing is generated (L746-L769)"; }
  if (step("L788", "if args.debug_combine_topic is not None: combine; exit()", "debug_combine")) { o.branch = 1; o.combine = 1; o.outcome = "combine_videos(debug topic) once, then exits (L788-L790)"; }
  if (step("L792", "if args.only_gen_vid: render_video_fix_code(topic, desc, max_retries=…)", "only_gen_vid")) { o.branch = 1; o.crash = 1; o.outcome = "TypeError: render_video_fix_code() missing 2 required positional arguments: 'scene_outline' and 'implementation_plans' (L800 vs L291-L297); nothing renders"; }
  if (step("L810", "elif args.check_status: print P/C/R table; exit()", "check_status")) { o.branch = 1; o.nTopics = n; o.outcome = `prints the P/C/R status table for ${n} theorem(s), then exits; nothing is generated (L810-L885)`; }
  const batchElse = batch && !stop;
  row("L887", "else:  # per theorem", batch ? (batchElse ? "true" : "skip") : "skip", 1);
  if (batchElse) {
    o.branch = 1; o.nTopics = n; read.add("only_combine");
    row("L896", "if args.only_combine: combine_videos(topic)", f("only_combine") ? "true" : "false", 2);
    if (f("only_combine")) { o.combine = 1; o.outcome = `combine_videos for each of ${n} theorem(s); no planning or rendering (L896-L897)`; row("L899", "await generate_video_pipeline(…, only_plan, specific_scenes=args.scenes)", "skip", 2); row("L906", "if not only_plan and not only_render: combine_videos(topic)", "skip", 2); }
    else {
      ["only_plan", "only_render", "scenes"].forEach(k => read.add(k));
      o.pipeline = 1; o.renders = f("only_plan") ? 0 : 1; o.combine = !f("only_plan") && !f("only_render") ? 1 : 0;
      row("L899", "await generate_video_pipeline(…, only_plan, specific_scenes=args.scenes)", "true", 2);
      row("L906", "if not only_plan and not only_render: combine_videos(topic)", o.combine ? "true" : "false", 2);
      o.outcome = `generate_video_pipeline for each of ${n} theorem(s)${f("only_plan") ? ", plans only (returns before rendering, L537-L539)" : f("only_render") ? ", rendering only scenes without code (L561-L566)" : ""}${f("scenes") ? ", planning restricted to --scenes" : ""}; ${o.combine ? "then combine_videos" : "combine_videos skipped"}`;
    }
  } else { row("L896", "if args.only_combine: combine_videos(topic)", "skip", 2); row("L899", "await generate_video_pipeline(…, only_plan, specific_scenes=args.scenes)", "skip", 2); row("L906", "if not only_plan and not only_render: combine_videos(topic)", "skip", 2); }
  // --- topic branch
  row("L917", "elif args.topic and args.context:", batch ? "skip" : topicMode ? "true" : "false");
  let tstop = !topicMode;
  const tstep = (ln, code, key) => { if (tstop) { row(ln, code, "skip", 1); return false; } read.add(key); const t = f(key); row(ln, code, t ? "true" : "false", 1); if (t) tstop = true; return t; };
  if (topicMode) { read.add("topic"); read.add("context"); o.branch = 2; o.nTopics = 1; }
  if (tstep("L937", "if args.only_gen_vid: render_video_fix_code(…)  # no await; exit()", "only_gen_vid")) o.outcome = "the async render_video_fix_code is called without await: a coroutine object is created and dropped, exit() follows, nothing renders (L937-L939)";
  if (tstep("L941", "if args.only_combine: combine_videos(args.topic)", "only_combine")) { o.combine = 1; o.outcome = "combine_videos for the one topic; no planning or rendering (L941-L942)"; }
  if (topicMode && !tstop) {
    ["only_plan", "only_render"].forEach(k => read.add(k));
    o.pipeline = 1; o.renders = f("only_plan") ? 0 : 1; o.combine = !f("only_plan") && !f("only_render") ? 1 : 0;
    o.outcome = `generate_video_pipeline for the one topic${f("only_plan") ? ", plans only (L537-L539)" : f("only_render") ? ", rendering only scenes without code (L561-L566)" : ""}; ${o.combine ? "then combine_videos" : "combine_videos skipped"}`;
  }
  row("L944", "asyncio.run(generate_video_pipeline(topic, context, only_plan))", topicMode && !tstop ? "true" : "skip", 1);
  row("L950", "if not only_plan and not only_render: combine_videos(topic)", topicMode && !tstop ? (o.combine ? "true" : "false") : "skip", 1);
  // --- usage
  const usage = !batch && !topicMode;
  row("L952", "else: print('Please provide either …'); exit()", usage ? "true" : "skip");
  if (usage) o.outcome = "prints \"Please provide either (--theorems_path) or (--topic and --context)\" and exits (L952-L954)";
  o.rows = rows;
  o.read = CLI_FLAGS.map(x => x[0]).filter(k => f(k) && read.has(k));
  o.ignored = CLI_FLAGS.map(x => x[0]).filter(k => f(k) && !read.has(k));
  o.nIgnored = o.ignored.length;
  return o;
}
MODELS["fixture:tea--cli-dispatch"] = {
  fn: cliDispatch,
  cases: [
    { args: { theorems_path: 1, sample_size: 3 }, pick: "nTopics", expect: 3 },                 // cli predict: first three theorems
    { args: { theorems_path: 1, sample_size: 3 }, pick: "helperFallback", expect: 1 },          // cli predict: helper gets --model
    { args: { theorems_path: 1, topic: 1 }, pick: "branch", expect: 1 },                        // cli transfer: batch branch wins
    { args: { theorems_path: 1, topic: 1 }, pick: "nTopics", expect: 20 },                      // ... over the whole file
    { args: { theorems_path: 1, topic: 1 }, pick: "nIgnored", expect: 1 },                      // ... and --topic is never read
    { args: { topic: 1, context: 1, check_status: 1 }, pick: "branch", expect: 2 },             // cli check: topic branch
    { args: { topic: 1, context: 1, check_status: 1 }, pick: "combine", expect: 1 },            // ... full pipeline then combine
    { args: { topic: 1, context: 1, check_status: 1 }, pick: "nIgnored", expect: 1 },           // ... --check_status ignored
    { args: { topic: 1 }, pick: "branch", expect: 3 },                                          // topic without context: usage message
    { args: { theorems_path: 1, sample_size: 50 }, pick: "nTopics", expect: 20 },               // slice past the end keeps all 20
    { args: { topic: 1, context: 1, only_gen_vid: 1 }, pick: "renders", expect: 0 },            // silent-degradation: un-awaited coroutine
    { args: { theorems_path: 1, only_gen_vid: 1 }, pick: "crash", expect: 1 },                  // silent-degradation: TypeError at L800
    { args: { topic: 1, context: 1, only_plan: 1 }, pick: "combine", expect: 0 },               // silent-degradation check: --only_plan skips combine
    { args: { theorems_path: 1, check_status: 1, only_gen_vid: 1 }, pick: "crash", expect: 1 }, // only_gen_vid is tested before check_status
  ],
};
WIDGETS["fixture:tea--cli-dispatch"] = (root) => {
  const s = { topic: true, context: true, n_file: 20, sample_size: 0 };
  const pic = el("div"), read = el("div", { class: "readout" });
  const draw = () => {
    const m = cliDispatch(s);
    const W = 660, rh = 21, y0 = 30;
    let b = text(10, 18, "the if/elif chain as written · green = tested true, taken · outline = tested false · grey = never reached", { fill: C.muted, size: 11 });
    m.rows.forEach((r, i) => {
      const y = y0 + i * rh, x = 14 + r.depth * 22;
      const fill = r.st === "true" ? C.ok : "#fff", op = r.st === "true" ? 0.18 : 1;
      b += rect(x, y, W - x - 96, rh - 4, fill, `fill-opacity="${op}" stroke="${r.st === "skip" ? "#e2ddd2" : r.st === "true" ? "var(--ok)" : "#b9b2a5"}"`);
      b += text(x + 6, y + 12, r.code, { mono: true, size: 11, fill: r.st === "skip" ? "#b4ada0" : C.ink });
      b += text(W - 88, y + 12, r.ln, { size: 10, fill: C.muted });
      b += text(W - 50, y + 12, r.st === "true" ? "taken" : r.st === "false" ? "false" : "—", { size: 10, fill: r.st === "true" ? C.ok : C.muted, weight: r.st === "true" ? "bold" : "" });
    });
    pic.innerHTML = svg(W, y0 + m.rows.length * rh + 8, b);
    const lab = k => CLI_FLAGS.find(x => x[0] === k)[1].split(" ")[0];
    read.innerHTML = `wrappers built before the chain (L714-L735): planner = --model, scene = --model, helper = ${s.helper_model ? "--helper_model" : "<b>--model</b> (no --helper_model, falls back, L722)"}; all at temperature 0.7<br>
      branch: <b>${["", "batch (--theorems_path)", "single topic (--topic and --context)", "usage message"][m.branch]}</b><br>
      <span class="big" style="font-size:18px">${esc(m.outcome)}</span>
      ${m.branch === 1 && m.nTopics ? `theorems used: ${m.nTopics} of ${s.n_file}${s.sample_size ? ` (theorems[:${s.sample_size}] keeps the first ${m.nTopics} in file order, L743-L744)` : ""}<br>` : ""}
      flags read on this path: ${m.read.length ? m.read.map(lab).join(", ") : "none"}<br>
      <span style="color:${m.nIgnored ? "var(--accent2)" : "inherit"}">flags you set that this path never reads: ${m.nIgnored ? m.ignored.map(lab).join(", ") + " (no warning is printed)" : "none"}</span><br>
      ${prov("provenance: fixture:tea--cli-dispatch · generate_video.py:L714-L735 (wrappers), L738-L954 (dispatch), L291-L297 (render_video_fix_code signature), L537-L539 and L561-L584 (only_plan, only_render inside the pipeline). File length 20 = data/thb_easy/math.json.")}`;
  };
  const boxes = CLI_FLAGS.filter(([k]) => k !== "sample_size").map(([k, lab]) => check(lab, !!s[k], v => { s[k] = v; draw(); }));
  const presets = [["batch run", { theorems_path: true }], ["topic + context", { topic: true, context: true }], ["topic + context + --only_gen_vid", { topic: true, context: true, only_gen_vid: true }], ["batch + --only_gen_vid", { theorems_path: true, only_gen_vid: true }]];
  const ctl = el("div", { class: "controls" }, ...boxes,
    slider("--sample_size (0 = not given)", 0, 30, s.sample_size, 1, v => { s.sample_size = v; draw(); }),
    el("div", { style: "margin-top:6px" }, ...presets.map(([nm, p]) => el("button", { onclick: () => { for (const [k] of CLI_FLAGS) if (k !== "sample_size") s[k] = !!p[k]; boxes.forEach((bx, i) => { bx.querySelector("input").checked = !!s[CLI_FLAGS.filter(([k]) => k !== "sample_size")[i][0]]; }); draw(); } }, nm)), " "),
    read);
  root.append(el("div", { class: "widget" }, ctl, pic)); draw();
};

// ======================================================== 2. outline parse ==
// What the pipeline does with the planner's reply, regex by regex (fresh topic, no files on disk):
//   video_planner.py:L166-L167  m = re.search(r'(<SCENE_OUTLINE>.*?</SCENE_OUTLINE>)', reply, re.DOTALL); saved = m.group(1) if m else reply
//   video_planner.py:L174-L175  saved is written to <prefix>_scene_outline.txt (no error on a miss)
//   utils.py:L125-L128          extract_xml: re.search(r'```xml\n(.*?)\n```', ., re.DOTALL).group(1), else the text unchanged
//   generate_video.py:L273,L516 count = len(re.findall(r'<SCENE_(\d+)>[^<]', content)): opening tags followed by a non-'<' char
//   generate_video.py:L527-L535 for n in 1..count: re.search(f'<SCENE_{n}>(.*?)</SCENE_{n}>'); no match -> scene silently not planned
//   video_planner.py:L241,L287,L325 three planner calls per planned scene (storyboard, technical, narration) + 1 outline call (L161)
//   with --use_rag: detect_relevant_plugins (1 helper call, L150-L153) + one helper query call per stage (rag_integration.py
//     L121-L263, each skipped only when its scene<i>/rag_cache file exists; a fresh topic has none)
//   generate_video.py:L542-L543,L568 unplanned scenes stay None in the plan list and are still queued for code generation
//   video_renderer.py:L237-L238 combine_videos (after rendering, unless --only_plan) re-runs the OUTLINE regex on the saved file
//     and calls .group(1) unguarded: no </SCENE_OUTLINE> -> AttributeError; else it re-counts with the same tag regex.
// The 3-7 scene rule is prompt prose only (task_generator/prompts_raw/__init__.py:L225); nothing in code checks it.
const sceneBlock = (n, title) => `    <SCENE_${n}>\n    Scene Title: ${title}\n    Scene Purpose: ...\n    Scene Description: ...\n    Scene Layout: ...\n    </SCENE_${n}>\n`;
const TITLES = ["Setting the stage", "The key construction", "Proof step by step", "A worked example", "Why it matters", "Edge cases", "Summary"];
const outlineOf = k => `<SCENE_OUTLINE>\n${TITLES.slice(0, k).map((t, i) => sceneBlock(i + 1, t)).join("\n")}</SCENE_OUTLINE>`;
const OUTLINE_PRESETS = {
  "well-formed, 3 scenes": outlineOf(3),
  "well-formed, 5 scenes": outlineOf(5),
  "```xml fence, 4 scenes, closing paragraph": "Here is the plan.\n```xml\n" + outlineOf(4) + "\n```\nLet me know if you want a different number of scenes.",
  "cut off inside <SCENE_5>": `<SCENE_OUTLINE>\n${TITLES.slice(0, 4).map((t, i) => sceneBlock(i + 1, t)).join("\n")}\n    <SCENE_5>\n    Scene Title: Why it ma`,
  "slide tags instead of scene tags": "<SLIDE_OUTLINE>\n    <SLIDE_1>\n    Slide Title: Setting the stage\n    </SLIDE_1>\n    <SLIDE_2>\n    Slide Title: The key construction\n    </SLIDE_2>\n</SLIDE_OUTLINE>",
  "error text instead of a reply (no API key)": "AuthenticationError: no API key was provided (example wording: LiteLLMWrapper returns str(e) as the reply, mllm_tools/litellm.py:L191-L193)",
  "an empty <SCENE_3></SCENE_3>": `<SCENE_OUTLINE>\n${sceneBlock(1, TITLES[0])}\n${sceneBlock(2, TITLES[1])}\n    <SCENE_3></SCENE_3>\n${sceneBlock(4, TITLES[3])}</SCENE_OUTLINE>`,
};
function outlineParse({ text = "", rag = false, only_plan = false }) {
  const m = /(<SCENE_OUTLINE>.*?<\/SCENE_OUTLINE>)/s.exec(text);
  const saved = m ? m[1] : text;
  const fx = /```xml\n(.*?)\n```/s.exec(saved), content = fx ? fx[1] : saved;
  const tags = [...content.matchAll(/<SCENE_(\d+)>[^<]/g)].map(x => +x[1]);
  const count = tags.length, scenes = [];
  for (let n = 1; n <= count; n++) scenes.push({ n, planned: new RegExp(`<SCENE_${n}>(.*?)</SCENE_${n}>`, "s").test(content) });
  const planned = scenes.filter(x => x.planned).length;
  const cm = /(<SCENE_OUTLINE>.*?<\/SCENE_OUTLINE>)/s.exec(saved);
  const combineCount = cm ? [...cm[1].matchAll(/<SCENE_(\d+)>[^<]/g)].length : null;
  return {
    outlineMatched: m ? 1 : 0, fenced: fx ? 1 : 0, savedWhole: m ? 0 : 1, count, tags, scenes, planned, unplanned: count - planned,
    plannerCalls: 1 + 3 * planned, helperCalls: rag ? 1 + 3 * planned : 0,
    combineReached: only_plan ? 0 : 1, combineError: !only_plan && !cm ? 1 : 0, combineCount,
    inRange: count >= 3 && count <= 7 ? 1 : 0, savedLen: saved.length,
  };
}
MODELS["fixture:tea--outline-parse"] = {
  fn: outlineParse,
  cases: [
    { args: { text: OUTLINE_PRESETS["```xml fence, 4 scenes, closing paragraph"] }, pick: "count", expect: 4, from: "tea:scene-outline-plan:predict" },
    { args: { text: OUTLINE_PRESETS["cut off inside <SCENE_5>"] }, pick: "count", expect: 5, from: "tea:scene-outline-plan:check" },
    { args: { text: OUTLINE_PRESETS["cut off inside <SCENE_5>"] }, pick: "planned", expect: 4 },           // scene 5: no </SCENE_5>, not planned
    { args: { text: OUTLINE_PRESETS["cut off inside <SCENE_5>"] }, pick: "combineError", expect: 1 },      // and combine_videos raises at L237
    { args: { text: OUTLINE_PRESETS["slide tags instead of scene tags"] }, pick: "count", expect: 0 },     // outline transfer: 0 scenes
    { args: { text: OUTLINE_PRESETS["slide tags instead of scene tags"] }, pick: "savedWhole", expect: 1 },// ... whole reply saved
    { args: { text: OUTLINE_PRESETS["well-formed, 5 scenes"] }, pick: "plannerCalls", expect: 16, from: "tea:per-scene-three-stage-plan:predict" },
    { args: { text: OUTLINE_PRESETS["well-formed, 5 scenes"], rag: true }, pick: "helperCalls", expect: 16 },   // RAG adds 1 + 3 per scene
    { args: { text: OUTLINE_PRESETS["error text instead of a reply (no API key)"] }, pick: "count", expect: 0 },          // silent-degradation predict
    { args: { text: OUTLINE_PRESETS["error text instead of a reply (no API key)"] }, pick: "combineError", expect: 1 },   // ... AttributeError later
    { args: { text: OUTLINE_PRESETS["error text instead of a reply (no API key)"], only_plan: true }, pick: "combineError", expect: 0 }, // check: no traceback
    { args: { text: OUTLINE_PRESETS["an empty <SCENE_3></SCENE_3>"] }, pick: "count", expect: 3 },        // empty tag not counted
    { args: { text: OUTLINE_PRESETS["an empty <SCENE_3></SCENE_3>"] }, pick: "planned", expect: 3 },      // ... yet n runs 1..3: scene 3 is planned from an empty block, scene 4 never
  ],
};
WIDGETS["fixture:tea--outline-parse"] = (root) => {
  const s = { text: OUTLINE_PRESETS["well-formed, 3 scenes"], rag: false, only_plan: false };
  const pic = el("div"), read = el("div", { class: "readout" });
  const ta = el("textarea", { rows: 9, style: "width:100%;font:12px var(--mono);padding:6px", spellcheck: "false" });
  const draw = () => {
    const m = outlineParse(s), W = 660;
    const stage = (x, y, w, title, res, good) => rect(x, y, w, 44, good ? C.ok : C.b, `fill-opacity="0.12" stroke="${good ? "var(--ok)" : "var(--accent2)"}" rx="4"`) + text(x + 6, y + 16, title, { size: 11, mono: true }) + text(x + 6, y + 34, res, { size: 11, weight: "bold", fill: good ? C.ok : C.b });
    let b = text(10, 16, "planner reply  →  saved outline  →  scene count  →  per-scene plans  →  combine_videos", { fill: C.muted, size: 11 });
    b += stage(10, 26, 205, "OUTLINE regex (planner L166)", m.outlineMatched ? "matched: block saved" : "miss: whole reply saved", m.outlineMatched);
    b += stage(225, 26, 200, "extract_xml (utils L126)", m.fenced ? "```xml fence stripped" : "no fence: text unchanged", true);
    b += stage(435, 26, 215, "<SCENE_(\\d+)>[^<] (L516)", `count = ${m.count}${m.count ? ` (tags ${m.tags.slice(0, 5).join(", ")}${m.count > 5 ? ", …" : ""})` : ""}`, m.count > 0);
    for (let i = 0; i < 2; i++) b += line(215 + i * 210, 48, 225 + i * 210, 48, C.muted);
    // per-scene cells
    const cw = 72, y1 = 96;
    b += text(10, y1 - 8, `the loop runs n = 1 … ${m.count} (not over the tag numbers found) and looks up <SCENE_n>(.*?)</SCENE_n> (L527-L535)`, { size: 11, fill: C.muted });
    if (!m.count) b += text(10, y1 + 22, "no scenes counted: no plan, no code, no render, and no error raised here", { size: 12, fill: C.b, weight: "bold" });
    m.scenes.slice(0, 8).forEach((sc, i) => {
      const x = 10 + i * (cw + 4);
      b += rect(x, y1, cw, 40, sc.planned ? C.ok : C.b, `fill-opacity="${sc.planned ? 0.15 : 0.1}" stroke="${sc.planned ? "var(--ok)" : "var(--accent2)"}" ${sc.planned ? "" : 'stroke-dasharray="4 3"'} rx="3"`);
      b += text(x + cw / 2, y1 + 16, `scene ${sc.n}`, { size: 11, anchor: "middle", weight: "bold" });
      b += text(x + cw / 2, y1 + 32, sc.planned ? "3 calls" : "no block", { size: 10, anchor: "middle", fill: sc.planned ? C.ok : C.b });
    });
    if (m.count > 8) b += text(10 + 8 * (cw + 4), y1 + 24, `+${m.count - 8}`, { size: 11, fill: C.muted });
    const y2 = 160;
    b += stage(10, y2, 640, "combine_videos: re.search(OUTLINE).group(1) on the saved file (video_renderer L237)",
      !m.combineReached ? "not reached (--only_plan returns before rendering and combine is skipped)" : m.combineError ? "AttributeError: 'NoneType' object has no attribute 'group' (the traceback, far from the cause)" : `re-counts ${m.combineCount} scene(s)`, !m.combineError);
    pic.innerHTML = svg(W, y2 + 54, b);
    read.innerHTML = `scenes counted: <b>${m.count}</b> · planned: <b>${m.planned}</b>${m.unplanned ? ` · not planned: ${m.scenes.filter(x => !x.planned).map(x => x.n).join(", ")} (stay None in the plan list and are still queued for code generation, L542-L543, L568)` : ""}<br>
      <span class="big" style="font-size:18px">planner-model calls in planning: 1 + 3 × ${m.planned} = ${m.plannerCalls}</span>
      ${s.rag ? `helper-model calls (RAG, fresh rag_cache): 1 plugin pick + 3 × ${m.planned} stage queries = ${m.helperCalls}<br>` : ""}
      ${m.count && !m.inRange ? `count ${m.count} is outside the prompt's "between 3 and 7": nothing in code checks it (prompts_raw/__init__.py L225)<br>` : ""}
      saved to &lt;prefix&gt;_scene_outline.txt: ${m.savedWhole ? `the <b>whole reply</b> (${m.savedLen} characters), because the OUTLINE regex needs both tags` : "the &lt;SCENE_OUTLINE&gt;…&lt;/SCENE_OUTLINE&gt; block"}<br>
      ${prov("provenance: fixture:tea--outline-parse · video_planner.py:L161-L175, L241/L287/L325 · src/utils/utils.py:L125-L128 · generate_video.py:L273, L516, L527-L543, L568 · video_renderer.py:L237-L238 · rag_integration.py:L79-L87, L121-L263. Regexes are re-implemented in JavaScript (DOTALL as the s flag). The reply texts are examples you can edit.")}`;
  };
  ta.value = s.text; ta.addEventListener("input", () => { s.text = String(ta.value ?? ""); draw(); });
  const ctl = el("div", { class: "controls" }, el("label", {}, "planner reply (edit it)", ta),
    el("div", { style: "margin-top:6px" }, ...Object.keys(OUTLINE_PRESETS).map(k => el("button", { onclick: () => { s.text = OUTLINE_PRESETS[k]; ta.value = s.text; draw(); } }, k)), " "),
    check("--use_rag", false, v => { s.rag = v; draw(); }), check("--only_plan", false, v => { s.only_plan = v; draw(); }), read);
  root.append(el("div", { class: "widget" }, ctl, pic)); draw();
};

// ======================================================== 3. fix loop ==
// process_scene (generate_video.py:L350-L424), one scene, as written:
//   curr_version = 0 (L351); generate_manim_code -> v0.py + v0_init_log.txt (L362-L378)
//   while True: render_scene(v<curr_version>)                                    (L383-L398)
//     success iff manim's returncode == 0 (video_renderer.py:L58-L68) and, when --use_visual_fix_code is on, the visual block
//       does not raise; it always raises: self.scene_model is never set on VideoRenderer (L20-L30 vs L79), and the except at
//       L113-L120 returns (code, str(e)) as if it were a render error. On success succ_rendered.txt is written (L122-L124).
//     if error is None: break (L399-L400); if curr_version >= max_retries: break (L402-L404)   <- tested AFTER the render
//     curr_version += 1; fix_code_errors(plan, code, error) -> v<n>.py + v<n>_fix_log.txt (L406-L422)
//   Each failed render appends to v<n>_error.log (video_renderer.py:L117-L118). CLI default max_retries = 5 (L682).
// With --use_rag: generate_manim_code makes one helper query call per scene (rag_queries_code.json, code_generator.py:L112-L121)
//   and fix_code_errors one per scene on its first fix only (rag_queries_error_fix.json is keyed by scene folder, not by the
//   error, code_generator.py:L170-L179, L188-L204). A fresh run has no rag_cache files.
// Next identical rerun (generate_video.py:L549-L582, L314, L329, L350): scenes without succ_rendered.txt are queued, the
//   filtered list is enumerated, and curr_scene = position + 1, so queued scene k is generated into folder scene<position+1>/.
function fixLoop(a) {
  const R = Math.max(0, a.max_retries ?? 5), scenes = a.scenes || [], o = { renders: 0, fixes: 0, rendered: 0, unrendered: 0, helperErrFix: 0, helperCode: 0, rows: [] };
  scenes.forEach((fc, i) => {
    const vs = []; let v = 0, ok = false;
    for (;;) {
      const exit = a.exit0 ? 0 : (fc !== null && fc !== undefined && v >= fc ? 0 : 1);
      const err = exit !== 0 ? 1 : a.visual ? 2 : 0;          // 0 none, 1 manim stderr, 2 AttributeError from the visual block
      vs.push({ v, exit, err, stop: false });
      if (!err) { ok = true; break; }
      if (v >= R) { vs[vs.length - 1].stop = true; break; }
      v++;
    }
    const n = i + 1, fixes = vs.length - 1;
    o.rows.push({ n, vs, ok, fixes });
    o[`renders${n}`] = vs.length; o[`err${n}v0`] = vs[0].err;
    o.renders += vs.length; o.fixes += fixes; ok ? o.rendered++ : o.unrendered++;
    if (a.rag) { o.helperCode++; if (fixes > 0) o.helperErrFix++; }
  });
  const queued = o.rows.filter(r => !r.ok).map(r => r.n);
  o.requeued = queued.length; o.queued = queued;
  queued.forEach((k, j) => { o[`as${k}`] = j + 1; });
  return o;
}
MODELS["fixture:tea--fix-loop"] = {
  fn: fixLoop,
  cases: [
    { args: { max_retries: 2, scenes: [null] }, pick: "renders", expect: 3 },                     // fix-loop predict: v0, v1, v2
    { args: { max_retries: 2, scenes: [null] }, pick: "fixes", expect: 2 },                       // ... 2 fix calls
    { args: { max_retries: 3, scenes: [null] }, pick: "renders", expect: 4, from: "tea:error-driven-fix-loop:check" },
    { args: { max_retries: 3, scenes: [null] }, pick: "rendered", expect: 0 },                    // ... no succ_rendered.txt
    { args: { scenes: [null] }, pick: "renders", expect: 6 },                                     // CLI default 5: 6 renders, 5 fixes
    { args: { max_retries: 0, scenes: [null] }, pick: "renders", expect: 1 },                     // budget 0: v0 only, no fix
    { args: { max_retries: 2, scenes: [2] }, pick: "rendered", expect: 1 },                       // v2 still renders when curr_version == max_retries
    { args: { max_retries: 2, scenes: [3] }, pick: "rendered", expect: 0 },                       // ... v3 is never reached
    { args: { scenes: [null], exit0: true }, pick: "renders", expect: 1 },                        // fix-loop transfer: exit 0 = success at v0
    { args: { max_retries: 2, scenes: [0], visual: true }, pick: "rendered", expect: 0 },         // visual-fix check: unrendered after v2
    { args: { max_retries: 2, scenes: [0], visual: true }, pick: "renders", expect: 3 },          // ... three clean renders, all "errors"
    { args: { max_retries: 5, scenes: [0, 0], visual: true }, pick: "err2v0", expect: 2 },        // visual-fix predict: v0's error is the AttributeError
    { args: { scenes: [0, 2, 0, 0, 2], rag: true }, pick: "helperErrFix", expect: 2, from: "tea:rag-query-retrieve-k2:check" },
    { args: { scenes: [0, 2, 0, 0, 2], rag: true }, pick: "fixes", expect: 4 },                   // ... 4 fixes, 2 query calls
    { args: { max_retries: 3, scenes: [null, 0, null, null, 0, null] }, pick: "requeued", expect: 4, from: "tea:resume-by-artifact:check" },
    { args: { scenes: [0, 0, 0, null, null] }, pick: "requeued", expect: 2 },                     // resume predict: scenes 4 and 5 queued
    { args: { scenes: [0, 0, 0, null, null] }, pick: "as4", expect: 1 },                          // ... scene 4's plan runs as scene 1
    { args: { scenes: [0, 0, 0, null, null] }, pick: "as5", expect: 2 },                          // ... scene 5's as scene 2
  ],
};
WIDGETS["fixture:tea--fix-loop"] = (root) => {
  const s = { max_retries: 5, text: "0, 2, x", visual: false, rag: false, exit0: false };
  const pic = el("div"), read = el("div", { class: "readout" }), err = el("div", { class: "muted small" });
  const inp = el("input", { type: "text", style: "width:100%;max-width:320px" });
  const parse = () => {
    const parts = s.text.split(",").map(t => t.trim()).filter(Boolean);
    if (!parts.length) return { err: "list one entry per scene" };
    if (parts.length > 8) return { err: "at most 8 scenes" };
    const sc = parts.map(t => (/^(x|never|-)$/i.test(t) ? null : /^\d+$/.test(t) ? +t : NaN));
    if (sc.some(v => Number.isNaN(v))) return { err: "each entry is a version number (0, 1, 2, …) or x for never" };
    return { scenes: sc };
  };
  const draw = () => {
    const p = parse(); err.textContent = p.err || "";
    if (p.err) { pic.innerHTML = ""; read.innerHTML = ""; return; }
    const m = fixLoop({ ...s, scenes: p.scenes }), R = s.max_retries;
    const cw = 50, gap = 12, x0 = 70, rh = 52, W = Math.max(660, x0 + (R + 1) * (cw + gap) + 190);
    const col = e => (e === 0 ? C.ok : e === 1 ? C.b : C.hi);
    let b = text(10, 16, "one row per scene; each cell is one manim run of v<n>.py, and the label under it is the stop test", { fill: C.muted, size: 11 });
    b += text(10, 30, "green = exit 0, accepted · red = exit ≠ 0, stderr is the error · yellow = exit 0, visual block raises", { fill: C.muted, size: 11 });
    m.rows.forEach((r, i) => {
      const y = 40 + i * rh;
      b += text(10, y + 20, `scene ${r.n}`, { size: 12, weight: "bold" });
      r.vs.forEach((c, j) => {
        const x = x0 + j * (cw + gap);
        b += rect(x, y + 4, cw, 24, col(c.err), `fill-opacity="0.85" rx="3"`) + text(x + cw / 2, y + 20, `v${c.v}`, { size: 12, anchor: "middle", fill: "#fff", weight: "bold" });
        if (c.err) b += text(x + cw / 2, y + 42, c.stop ? `${c.v} ≥ ${R}: stop` : `${c.v} < ${R}: fix`, { size: 9, anchor: "middle", fill: c.stop ? C.b : C.muted });
        if (c.err && !c.stop) b += line(x + cw + 1, y + 16, x + cw + gap - 1, y + 16, C.muted);
      });
      const xe = x0 + r.vs.length * (cw + gap);
      b += text(xe, y + 14, r.ok ? "succ_rendered.txt written" : "no succ_rendered.txt", { size: 11, fill: r.ok ? C.ok : C.b, weight: "bold" });
      b += text(xe, y + 28, `${r.vs.length} render${r.vs.length > 1 ? "s" : ""}, ${r.fixes} fix_code_errors`, { size: 10, fill: C.muted });
    });
    const yq = 40 + m.rows.length * rh + 6;
    b += line(10, yq - 4, W - 10, yq - 4, C.rule);
    b += text(10, yq + 12, `next identical rerun queues scenes without succ_rendered.txt (L568): ${m.requeued ? m.queued.join(", ") : "none"}`, { size: 11, weight: "bold" });
    const moved = m.queued.some(k => m["as" + k] !== k), maps = m.queued.map(k => `plan ${k} → scene${m["as" + k]}/`);
    let yl = yq + 12;
    if (m.requeued) {
      yl += 16; b += text(10, yl, `curr_scene = position in that list + 1 (L314, L350)${moved ? ", so a plan can land in another scene's folder:" : ":"}`, { size: 11, fill: moved ? C.b : C.muted });
      for (let i = 0; i < maps.length; i += 4) { yl += 15; b += text(22, yl, maps.slice(i, i + 4).join(" · "), { size: 11, mono: true, fill: moved ? C.b : C.muted }); }
    }
    pic.innerHTML = svg(W, yl + 10, b);
    read.innerHTML = `<span class="big" style="font-size:18px">manim runs: ${m.renders} · fix_code_errors calls: ${m.fixes} · scenes rendered: ${m.rendered} of ${m.rows.length}</span>
      a scene's worst case is ${R} + 1 = ${R + 1} runs: the stop test curr_version ≥ max_retries comes after the render of v${R} (L402)<br>
      ${s.rag ? `helper-model query calls (RAG, fresh cache): ${m.helperCode} for code generation + <b>${m.helperErrFix}</b> for error fixes (one per scene that needed a fix; later fixes reuse rag_queries_error_fix.json whatever the new error says)<br>` : ""}
      ${s.visual ? `--use_visual_fix_code: every exit-0 run comes back as "'VideoRenderer' object has no attribute 'scene_model'" (L79), which the loop treats as a render error; no VLM is ever called<br>` : ""}
      ${s.exit0 ? `renderer always exits 0: the first run is accepted; stderr is read only when returncode ≠ 0 (L67-L68)<br>` : ""}
      the loop reads one signal, whether render_scene returned an error; without the visual flag that is manim's exit code, so a scene with overlapping or off-screen objects that exits 0 is a success<br>
      ${prov("provenance: fixture:tea--fix-loop · generate_video.py:L350-L424 (loop), L682 (default 5), L549-L582 and L314/L329/L350 (rerun queue and numbering) · video_renderer.py:L20-L30, L53-L126 · code_generator.py:L112-L121, L170-L204 (RAG query caches). Which version first exits 0 is your input, not the repo's.")}`;
  };
  inp.value = s.text; inp.addEventListener("input", () => { s.text = String(inp.value ?? ""); draw(); });
  const presets = [["three scenes", "0, 2, x"], ["one scene, never clean", "x"], ["six scenes, two clean at v0", "x, 0, x, x, 0, x"], ["five scenes, two clean at v2", "0, 2, 0, 0, 2"]];
  const ctl = el("div", { class: "controls" },
    slider("--max_retries", 0, 8, s.max_retries, 1, v => { s.max_retries = v; draw(); }),
    el("label", {}, "per scene: the first version whose manim run exits 0 (x = never), comma separated ", inp), err,
    el("div", { style: "margin-top:6px" }, ...presets.map(([nm, t]) => el("button", { onclick: () => { s.text = t; inp.value = t; draw(); } }, nm)), " "),
    check("--use_visual_fix_code", false, v => { s.visual = v; draw(); }),
    check("--use_rag (count helper query calls)", false, v => { s.rag = v; draw(); }),
    check("replace manim with a wrapper that always exits 0", false, v => { s.exit0 = v; draw(); }), read);
  root.append(el("div", { class: "widget" }, ctl, pic)); draw();
};

// ======================================================== 4. evaluator scores ==
// evaluate.py process_theorem (L315-L346) with three judges at temperature 0 (L384-L395):
//   text judge (default azure/gpt-4o, L357-L360) reads the transcript only: accuracy_and_depth, logical_flow (one integer each,
//     eval_suite/prompts_raw/__init__.py:L57-L63, L73-L80); runs only if a .srt/.txt exists (L320-L324)
//   video judge (default gemini/gemini-1.5-pro-002, L362-L366): 10 equal-duration chunks, duration / 10 each (L115-L128);
//     its prompt is formatted with the description only, never the transcript (video_utils.py:L148): visual_consistency
//   image judge (default azure/gpt-4o, L367-L370): frames at 1 fps (image_utils.py:L29); frames_per_chunk = total // 10 (L37);
//     10 windows of that many frames, the one with most non-black pixels per window (L43-L51): visual_relevance, element_layout
//     So frames past 10 * (total // 10) are never candidates, and total < 10 makes frames_per_chunk 0 -> ZeroDivisionError at L38.
//   Every score is an integer 1-5 or the reply is rejected (eval_suite/utils.py:L52-L58).
//   Per criterion: geometric mean over chunks/frames (evaluate.py:L160-L162, image_utils.py:L97-L99).
//   overall_score (only for --eval_type all, L332-L341) = geometric mean of every non-"chunks" score field (L171-L208), i.e. of
//   the criterion scores of the judges that ran; calculate_geometric_mean drops None and returns 0.0 for an empty list (utils.py:L66-L81).
// Frame count n for a T-second video is taken as T (t = 0, 1, …, T−1 at 1 fps): an approximation of moviepy's iter_frames.
const gmean = xs => { const v = xs.filter(x => x !== null && x !== undefined); return v.length ? v.reduce((p, x) => p * x, 1) ** (1 / v.length) : 0.0; };
const amean = xs => xs.reduce((p, x) => p + x, 0) / xs.length;
const CRITS = [["accuracy_and_depth", "text"], ["logical_flow", "text"], ["visual_consistency", "video"], ["visual_relevance", "image"], ["element_layout", "image"]];
function evalScores(a) {
  const type = a.eval_type || "all", T = a.T ?? 95;
  const ran = { text: (type === "all" || type === "text") && a.transcript !== false, video: type === "all" || type === "video", image: type === "all" || type === "image" };
  const chunks = v => (Array.isArray(v) ? (v.length === 1 ? Array(10).fill(v[0]) : v) : [v]);
  const raw = { accuracy_and_depth: [a.ad ?? 4], logical_flow: [a.lf ?? 4], visual_consistency: chunks(a.vc ?? [4]), visual_relevance: chunks(a.vr ?? [4]), element_layout: chunks(a.el ?? [4]) };
  const o = { crit: {}, ran };
  for (const [k, j] of CRITS) { o.crit[k] = ran[j] ? gmean(raw[k]) : null; o[k] = o.crit[k] ?? NaN; }
  const present = CRITS.map(([k]) => o.crit[k]).filter(x => x !== null);
  o.nScores = present.length;
  o.overall = type === "all" ? gmean(present) : NaN;
  o.arith = type === "all" && present.length ? amean(present) : NaN;
  o.transcriptCriteria = ran.text ? 2 : 0;                    // criteria whose input includes the transcript
  const n = Math.floor(T), fpc = Math.floor(n / 10);
  o.frames = n; o.kfWindow = fpc; o.kfError = fpc === 0 ? 1 : 0; o.kfCovered = 10 * fpc; o.kfTail = fpc ? n - 10 * fpc : n; o.chunkSec = T / 10;
  return o;
}
MODELS["fixture:tea--eval-scores"] = {
  fn: evalScores,
  cases: [
    { args: { ad: 5, lf: 5, vc: [5], vr: [5], el: [1] }, pick: "overall", expect: 3.624, tol: 0.005, from: "tea:evaluator-metrics-judges:predict" },
    { args: { ad: 5, lf: 5, vc: [5], vr: [5], el: [1] }, pick: "arith", expect: 4.2 },                 // the misconception's arithmetic mean
    { args: { ad: 5, lf: 5, vc: [5], vr: [5], el: [1], transcript: false }, pick: "overall", expect: 2.924 },  // no transcript: 3 scores
    { args: { ad: 5, lf: 5, vc: [5], vr: [5], el: [1], transcript: false }, pick: "nScores", expect: 3 },
    { args: { el: [5, 5, 5, 5, 5, 5, 5, 5, 5, 1] }, pick: "element_layout", expect: 4.257 },           // one bad still of ten: 5^0.9
    { args: { ad: 4, lf: 4 }, pick: "transcriptCriteria", expect: 2 },                                // judges check: a swapped .srt moves 2 criteria
    { args: { eval_type: "video", transcript: false }, pick: "transcriptCriteria", expect: 0 },          // judges transfer: video judge only
    { args: { T: 95 }, pick: "kfWindow", expect: 9 },                                                 // 95 s: windows of 9 frames
    { args: { T: 95 }, pick: "kfTail", expect: 5 },                                                   // ... last 5 s never a key frame
    { args: { T: 19 }, pick: "kfTail", expect: 9 },                                                   // 19 s: 9 of 19 s never sampled
    { args: { T: 9 }, pick: "kfError", expect: 1 },                                                   // under 10 s: ZeroDivisionError
    { args: { T: 95 }, pick: "chunkSec", expect: 9.5 },                                               // video chunks cover all 95 s
  ],
};
WIDGETS["fixture:tea--eval-scores"] = (root) => {
  const s = { ad: 4, lf: 5, vcT: "4", vrT: "4", elT: "4, 4, 3, 4, 2, 4, 4, 3, 4, 4", transcript: true, eval_type: "all", T: 95 };
  const pic = el("div"), read = el("div", { class: "readout" }), err = el("div", { class: "muted small" });
  const listIn = key => { const i = el("input", { type: "text", style: "width:100%;max-width:300px" }); i.value = s[key]; i.addEventListener("input", () => { s[key] = String(i.value ?? ""); draw(); }); return i; };
  const parseList = t => { const v = t.split(",").map(x => x.trim()).filter(Boolean); if (!(v.length === 1 || v.length === 10)) return null; if (!v.every(x => /^[1-5]$/.test(x))) return null; return v.map(Number); };
  const draw = () => {
    const vc = parseList(s.vcT), vr = parseList(s.vrT), elx = parseList(s.elT);
    err.textContent = vc && vr && elx ? "" : "each list is one integer 1-5 (all ten the same) or exactly ten of them: the judges must answer integers 1-5 (eval_suite/utils.py L52-L58)";
    if (!(vc && vr && elx)) { pic.innerHTML = ""; read.innerHTML = ""; return; }
    const m = evalScores({ ad: s.ad, lf: s.lf, vc, vr, el: elx, transcript: s.transcript, eval_type: s.eval_type, T: s.T });
    const W = 720, x0 = 120, tw = 330, xj = x0 + tw + 14, sx = t => x0 + (t / m.frames) * tw;
    const dim = on => (on ? 1 : 0.25);
    let b = text(10, 16, `what each judge receives from one ${m.frames}-second video (time runs left to right)`, { fill: C.muted, size: 11 });
    // transcript row
    let y = 30;
    b += `<g opacity="${dim(m.ran.text)}">` + text(10, y + 13, ".srt transcript", { size: 11 }) + rect(x0, y + 2, tw, 16, C.a, 'fill-opacity="0.25"') + line(x0 + tw + 2, y + 10, xj - 2, y + 10, C.muted) +
      text(xj, y + 8, "text judge: accuracy_and_depth,", { size: 11, weight: "bold" }) + text(xj, y + 21, "logical_flow (no visuals, L65-L66)", { size: 11 }) + "</g>";
    if (!m.ran.text) b += text(x0 + 6, y + 14, s.eval_type === "all" || s.eval_type === "text" ? "no transcript file: text judge skipped (L321-L322)" : "not run for this --eval_type", { size: 10, fill: C.b });
    // video chunks row
    y = 66;
    b += `<g opacity="${dim(m.ran.video)}">` + text(10, y + 13, "10 video chunks", { size: 11 });
    for (let i = 0; i < 10; i++) b += rect(x0 + i * tw / 10 + 1, y + 2, tw / 10 - 2, 16, C.ok, `fill-opacity="${0.15 + 0.15 * ((vc.length === 1 ? vc[0] : vc[i]) - 1)}"`);
    b += line(x0 + tw + 2, y + 10, xj - 2, y + 10, C.muted) + text(xj, y + 8, "video judge: visual_consistency", { size: 11, weight: "bold" }) + text(xj, y + 21, `chunks of ${fmt(m.chunkSec, 1)} s, description only`, { size: 11 }) + "</g>";
    // key-frame row
    y = 102;
    b += `<g opacity="${dim(m.ran.image)}">` + text(10, y + 13, "frames at 1 fps", { size: 11 });
    if (m.kfError) b += rect(x0, y + 2, tw, 16, C.b, 'fill-opacity="0.15"') + text(x0 + 6, y + 14, `${m.frames} frames // 10 = 0: ZeroDivisionError (image_utils L38)`, { size: 10, fill: C.b, weight: "bold" });
    else {
      for (let i = 0; i < 10; i++) b += rect(sx(i * m.kfWindow) + 1, y + 2, sx(m.kfWindow) - x0 - 2, 16, C.hi, 'fill-opacity="0.45"');
      if (m.kfTail) b += rect(sx(m.kfCovered), y + 2, x0 + tw - sx(m.kfCovered), 16, "#d9d3c7", "") + text(x0 + tw - 3, y + 32, `last ${m.kfTail} s: never a candidate`, { size: 10, anchor: "end", fill: C.b });
    }
    b += line(x0 + tw + 2, y + 10, xj - 2, y + 10, C.muted) + text(xj, y + 8, "image judge: visual_relevance,", { size: 11, weight: "bold" }) + text(xj, y + 21, `element_layout: 10 stills${m.kfError ? "" : `, 1 per ${m.kfWindow} s`}`, { size: 11 }) + "</g>";
    // score bars
    y = 150;
    b += line(10, y - 6, W - 10, y - 6, C.rule) + text(10, y + 8, "criterion scores (geometric mean over chunks or stills) and the overall score", { fill: C.muted, size: 11 });
    const bx = 170, bw = 300, sc = v => bx + ((v - 0) / 5) * bw;
    const bars = [...CRITS.map(([k]) => [k, m.crit[k]]), ["overall_score (geometric)", s.eval_type === "all" ? m.overall : null], ["arithmetic mean (not used)", s.eval_type === "all" ? m.arith : null]];
    for (let t = 1; t <= 5; t++) b += line(sc(t), y + 14, sc(t), y + 18 + 7 * 20, "#eee9df") + text(sc(t), y + 18 + 7 * 20 + 10, String(t), { size: 9, anchor: "middle", fill: C.muted });
    bars.forEach(([k, v], i) => {
      const yy = y + 18 + i * 20, isO = i >= 5;
      b += text(10, yy + 11, k, { size: 11, weight: i === 5 ? "bold" : "", fill: i === 6 ? C.muted : C.ink });
      if (v === null || Number.isNaN(v)) { b += text(bx, yy + 11, i >= 5 ? "only for --eval_type all" : "judge did not run", { size: 10, fill: C.muted }); return; }
      b += rect(bx, yy + 1, sc(v) - bx, 14, i === 5 ? C.a : i === 6 ? "#cfc9bd" : C.ok, `fill-opacity="${isO ? 0.9 : 0.6}"`) + text(sc(v) + 6, yy + 12, fmt(v, 3), { size: 11, weight: i === 5 ? "bold" : "" });
    });
    pic.innerHTML = svg(W, y + 18 + 7 * 20 + 16, b);
    read.innerHTML = `${s.eval_type === "all" ? `<span class="big" style="font-size:18px">overall_score = geometric mean of ${m.nScores} criterion scores = ${fmt(m.overall, 3)}</span>arithmetic mean would be ${fmt(m.arith, 3)}${m.nScores < 5 ? ` · only ${m.nScores} scores enter because the text judge did not run` : ""}<br>` : `--eval_type ${s.eval_type}: one judge, no overall_score (L342-L343)<br>`}
      the transcript reaches ${m.transcriptCriteria ? "only accuracy_and_depth and logical_flow" : "no criterion here"}; nothing compares what is said with what is shown, and no criterion measures a margin or a bounding box<br>
      ${prov("provenance: fixture:tea--eval-scores · evaluate.py:L115-L162, L171-L208, L315-L346, L357-L395 · eval_suite/image_utils.py:L13-L60, L91-L99 · eval_suite/video_utils.py:L148 · eval_suite/utils.py:L52-L81 · eval_suite/prompts_raw/__init__.py:L57-L66. The scores are your input; frame count is approximated as one per second.")}`;
  };
  const typeSel = el("select", {}, ...["all", "text", "video", "image"].map(t => el("option", { value: t }, t)));
  typeSel.addEventListener("change", () => { s.eval_type = typeSel.value; draw(); });
  const ctl = el("div", { class: "controls" },
    el("label", {}, "--eval_type ", typeSel), check("transcript (.srt) present", true, v => { s.transcript = v; draw(); }),
    slider("text judge: accuracy_and_depth", 1, 5, s.ad, 1, v => { s.ad = v; draw(); }),
    slider("text judge: logical_flow", 1, 5, s.lf, 1, v => { s.lf = v; draw(); }),
    el("label", {}, "video judge: visual_consistency per chunk", listIn("vcT")),
    el("label", {}, "image judge: visual_relevance per still", listIn("vrT")),
    el("label", {}, "image judge: element_layout per still", listIn("elT")), err,
    slider("video length (s)", 5, 300, s.T, 1, v => { s.T = v; draw(); }, v => `${v} s`), read);
  root.append(el("div", { class: "widget" }, ctl, pic)); draw();
};
