// Widgets for thread edtrace (the tracer that turns CS336's executable lectures into traces, and its viewer).
// Register as "fixture:<id>" -> (root, notice) => void. Each widget has a pure model in MODELS (no DOM) that
// tools/check_widgets.mjs tests against the KPs' stored answers.
// Sources: tools/edtrace/backend/src/edtrace/execute.py (cited as execute.py:L<n>), execute_util.py, and
//          tools/edtrace/frontend/src/TraceViewer.jsx (cited as TraceViewer:L<n>).
// The models below are line-for-line ports of that code, not approximations. The trace-run port was checked
// against real edtrace runs: the same template program, 720 variants (n 1-10, k 1-5, text on/off, callee in
// this file or helpers.py, @stepover on/off, @hide on/off), run with python3.11 -m edtrace.execute; every
// variant's steps (stack, env, renderings) and hidden_line_numbers matched, and the generated settrace event
// stream matched a raw sys.settrace recording of the same program (see research/14-student-line.md, edtrace).
import { el } from "../widgets_lib.js";

export const MODELS = {};
export const WIDGETS = {};

const esc = (s) => String(s).replace(/[&<>"]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]);
const button = (label, onclick, title = "") => el("button", { onclick, title }, label);
const checkbox = (label, on, onchange) => {
  const inp = el("input", { type: "checkbox" }); inp.checked = on;
  inp.addEventListener("change", () => onchange(inp.checked));
  return el("label", {}, inp, " ", label);
};
const intSlider = (label, min, max, value, onInput) => {
  const out = el("span", { class: "readout" }, value);
  const inp = el("input", { type: "range", min, max, value, step: 1 });
  inp.addEventListener("input", () => { out.textContent = inp.value; onInput(+inp.value); });
  return el("label", {}, `${label} `, out, inp);
};
const MONO = "font-family:var(--mono);font-size:12.5px";
const box = (style = "") => el("div", { style: `border:1px solid var(--rule);border-radius:6px;background:#fff;padding:6px 8px;${style}` });

// ============================================================ directive parsing (execute.py:L73-L137) ==
const ACCEPTED_DIRECTIVES = ["@inspect", "@clear", "@stepover", "@hide"];   // L73-L77
// L88-L109 verbatim: split on the FIRST "#" only, whitespace tokens; "@x" opens a directive, later tokens are its args.
export function parseDirectives(line) {
  if (!line.includes("#")) return { directives: [], tokens: [], warnings: [] };
  const tokens = line.split("#")[1].split(/\s+/).filter(Boolean);   // Python str.split("#")[1] is the text between the 1st and 2nd "#"
  const directives = [], warnings = [], dropped = [];
  for (const token of tokens) {
    if (token.startsWith("@")) {
      if (!ACCEPTED_DIRECTIVES.includes(token)) warnings.push(`WARNING: ${token} is not a valid directive.`);   // L101-L102
      directives.push({ name: token, args: [] });
    } else if (directives.length > 0) directives[directives.length - 1].args.push(token);
    else dropped.push(token);   // L107: tokens before the first @ are ignored
  }
  return { directives, tokens, warnings, dropped };
}
const exprsOf = (directives, name) => directives.filter(d => d.name === name).flatMap(d => d.args);   // L112-L137

// ======================================================= 1. trace-run: program -> events -> steps -> viewer ==
// The template program (the one run for real, see header). main() comes first so its line numbers do not move
// when the callee's body length k changes. f is always defined in lec.py; with `other`, line 6 calls g from
// helpers.py instead, whose body is identical. Body: optional text() line, then assignments with @inspect, then return.
const ASSIGNS = [["a", "x + 1", v => v + 1], ["b", "a * 2", v => v * 2], ["c", "b - 1", v => v - 1], ["d", "c + 3", v => v + 3]];
export function buildProgram({ n = 2, k = 3, txt = true, other = false, stepover = false, hide = false }) {
  k = Math.max(txt ? 2 : 1, Math.min(5, k));
  const body = [];   // {code, effect(locals, out)}
  let v = "x";
  if (txt) body.push({ code: 'text(f"f got {x}")', effect: (L, out) => out.push({ type: "markdown", data: `f got ${L.x}` }) });
  for (const [name, expr, fn] of ASSIGNS.slice(0, k - 1 - body.length)) {
    const src = v;
    body.push({ code: `${name} = ${expr}  # @inspect ${name}`, effect: (L) => { L[name] = fn(L[src]); } });
    v = name;
  }
  const ret = v; body.push({ code: `return ${ret}`, effect: (L) => { L.__ret = L[ret]; } });
  const callee = other ? "g" : "f";
  const dir = "  # @inspect z" + (stepover ? " @stepover" : "") + (hide ? " @hide" : "");
  const lec = ["from edtrace import text", "from helpers import g", "", "def main():",
    `    for i in range(${n}):`, `        z = ${callee}(i)${dir}`, '    text("done")', "", "def f(x):", ...body.map(b => "    " + b.code)];
  const hel = ["from edtrace import text", "", "def g(x):", ...body.map(b => "    " + b.code)];
  return { n, k, txt, other, stepover, hide, body, callee, files: { "lec.py": lec, "helpers.py": hel },
    calleeFile: other ? "helpers.py" : "lec.py", calleeDef: other ? 3 : 9 };
}

// The settrace event stream CPython 3.11 delivers for the template (matched against a raw sys.settrace recording):
// main: call@4; per iteration line@5, line@6, then the callee's call@def, line@each body line, return@last;
// after the loop line@5 (the exhausted for-check), line@7, return@7. A line's effects happen after its 'line' event,
// so the next event sees them (that is when edtrace's local_trace_func reads f_locals, execute.py:L290-L306).
// text()'s own frames (execute_util.py) also get events; trace_func drops them at L249-L251, so they are not listed.
export function settraceEvents(P) {
  // Each event carries `adds`: the Renderings that executing its line appends to edtrace's module-global
  // _current_renderings (execute_util.py:L141). They are appended after the tracer has handled the event.
  const ev = [], stack = [];
  const snap = () => stack.map(f => ({ path: f.file, line_number: f.line, function_name: f.fn, code: P.files[f.file][f.line - 1].trim() }));
  const emit = (f, event) => { const e = { frame: f, event, line: f.line, stack: snap(), locals: { ...f.locals }, adds: [] }; ev.push(e); return e; };
  const main = { id: 0, fn: "main", file: "lec.py", line: 4, locals: {} };
  stack.push(main); emit(main, "call");
  for (let i = 0; i < P.n; i++) {
    main.line = 5; emit(main, "line"); main.locals.i = i;
    main.line = 6; emit(main, "line");
    const c = { id: i + 1, fn: P.callee, file: P.calleeFile, line: P.calleeDef, locals: { x: i } };
    stack.push(c); emit(c, "call");
    P.body.forEach((b, j) => { c.line = P.calleeDef + 1 + j; const e = emit(c, "line"); b.effect(c.locals, e.adds); });
    emit(c, "return"); stack.pop();
    main.locals.z = c.locals.__ret;
  }
  main.line = 5; emit(main, "line");
  main.line = 7; emit(main, "line").adds.push({ type: "markdown", data: "done" });
  emit(main, "return");
  return ev;
}

// Port of execute() / trace_func / local_trace_func (execute.py:L203-L353). One `log` row per tracer invocation says
// what that code path did. Python calls the global function on 'call' events and the frame's own local function
// (whatever the previous invocation returned) on that frame's later events.
const sameStack = (a, b) => a.length === b.length && a.every((x, i) => x.path === b[i].path && x.line_number === b[i].line_number && x.function_name === b[i].function_name && x.code === b[i].code);
export function recordTrace(P) {
  const events = settraceEvents(P), visible = ["lec.py"];   // L345: only the traced module's own file
  const steps = [], stepovers = [], log = [], calls = {};
  let current = [];   // _current_renderings
  const GLOBAL = { kind: "trace_func" };
  function traceFunc(F, e, ei, via) {
    calls[F.id] = (calls[F.id] || 0) + 1;
    const row = { ei, fn: via ? "local→trace_func" : "trace_func", frame: F };
    log.push(row);
    if (!visible.includes(F.file)) { row.what = `${F.file} not in visible_paths → return trace_func (L249-L251)`; row.kind = "skip"; return GLOBAL; }
    const stack = e.stack;
    if (e.event === "return") { row.what = "'return' event → return trace_func (L255-L256)"; row.kind = "skip"; return GLOBAL; }
    const item = stack[stack.length - 1];
    const { directives } = parseDirectives(item.code);
    let note = "";
    if (directives.some(d => d.name === "@stepover")) {   // L266-L274
      const top = stepovers[stepovers.length - 1];
      if (top && top[0] === item.path && top[1] === item.line_number) { stepovers.pop(); note = `@stepover: (${item.path}, ${item.line_number}) was on the stack → pop. `; }
      else { stepovers.push([item.path, item.line_number]); note = `@stepover: push (${item.path}, ${item.line_number}). `; }
    }
    if (stepovers.some(so => stack.slice(0, -1).some(it => it.path === so[0] && it.line_number === so[1]))) {   // L277-L278
      row.what = note + `a caller is on the stepovers stack → return trace_func (L277-L278)`; row.kind = "skip"; return GLOBAL;
    }
    const open = { stack, env: {}, renderings: [], ei };
    if (steps.length === 0 || !sameStack(open.stack, steps[steps.length - 1].stack)) {   // L286-L287
      steps.push(open); row.what = note + `open step ${steps.length - 1} (empty env)`; row.kind = "open"; row.step = steps.length - 1;
    } else { row.what = note + `same stack as step ${steps.length - 1} → no new step`; row.kind = "merge"; row.step = steps.length - 1; }
    const cl = { openIdx: steps.length - 1, stack, directives, item };
    return cl;
  }
  function localTraceFunc(cl, F, e, ei) {
    const row = { ei, fn: "local_trace_func", frame: F }; log.push(row);
    let close;
    if (cl.openIdx === steps.length - 1) { close = steps[steps.length - 1]; row.what = `line ${cl.item.line_number} done → fill step ${steps.length - 1}`; row.kind = "fill"; }   // L294-L295
    else { close = { stack: cl.stack, env: {}, renderings: [], ei, close: true }; steps.push(close); row.what = `line ${cl.item.line_number} done, steps were added since → append close step ${steps.length - 1}`; row.kind = "close"; }   // L296-L303
    row.step = steps.indexOf(close);
    const filled = [];
    for (const expr of exprsOf(cl.directives, "@inspect")) {   // L310-L328
      const v = expr.split(".")[0];
      if (v in e.locals) { close.env[expr] = e.locals[v]; filled.push(`${expr} = ${e.locals[v]}`); }
      else row.warn = `WARNING: variable ${v} not found in locals`;
    }
    for (const expr of exprsOf(cl.directives, "@clear")) close.env[expr] = null;   // L330-L332
    close.renderings = current; current = [];   // L335 pop_renderings()
    if (filled.length) row.what += `; env ${filled.join(", ")}`;
    if (close.renderings.length) row.what += `; renderings ${close.renderings.map(r => JSON.stringify(r.data)).join(", ")}`;
    return traceFunc(F, e, ei, true);   // L338
  }
  events.forEach((e, ei) => {
    const F = e.frame;
    const tracer = e.event === "call" ? GLOBAL : F.tracer;
    F.tracer = tracer === GLOBAL ? traceFunc(F, e, ei, false) : localTraceFunc(tracer, F, e, ei);
    current.push(...e.adds);   // the line now runs: text() appends to the global list
  });
  // compute_hidden_line_numbers over the files in visible_paths (L356-L366)
  const hidden = { "lec.py": P.files["lec.py"].map((l, i) => parseDirectives(l).directives.some(d => d.name === "@hide") ? i + 1 : 0).filter(Boolean) };
  return { P, events, steps, log, calls, hidden };
}

// ---- viewer (TraceViewer.jsx), on the recorded steps ----
const last = (a) => a[a.length - 1];
export function inSameFunction(s1, s2) {   // TraceViewer:L408-L421: equal length, all but the last element agree on path and line
  if (s1.length !== s2.length) return false;
  for (let i = 0; i < s1.length - 1; i++) if (s1[i].path !== s2[i].path || s1[i].line_number !== s2[i].line_number) return false;
  return true;
}
const isStrictAncestorOf = (s1, s2) => s1.length < s2.length;   // L426-L428: a pure length test
export function stepOverIndex(steps, cur, dir = 1) {   // L244-L260
  let i = cur + dir;
  while (i >= 0 && i < steps.length) {
    if (inSameFunction(steps[i].stack, steps[cur].stack)) return i;
    if (isStrictAncestorOf(steps[i].stack, steps[cur].stack)) return i;
    i += dir;
  }
  return i;
}
export function stepUpIndex(steps, cur) {   // L262-L275
  let i = cur + 1;
  while (i < steps.length) {
    if (!inSameFunction(steps[i].stack, steps[cur].stack) && isStrictAncestorOf(steps[i].stack, steps[cur].stack)) return i;
    i++;
  }
  return i;
}
export function mergedEnv(steps, cur) {   // renderEnv, L355-L385: walk back through same-function steps until an ancestor
  const envs = [], from = [];
  for (let i = cur; i >= 0; i--) {
    if (inSameFunction(steps[i].stack, steps[cur].stack)) { if (Object.keys(steps[i].env).length) { envs.push(steps[i].env); from.push(i); } }
    else if (isStrictAncestorOf(steps[i].stack, steps[cur].stack)) break;
  }
  envs.reverse(); from.reverse();
  const env = {}; for (const e of envs) Object.assign(env, e);
  for (const key in env) if (env[key] === null) delete env[key];
  return { env, from };
}
export function lineRenderings(steps) {   // renderLines, L579-L582: keyed by line number only, last step wins
  const m = {}; for (const s of steps) m[last(s.stack).line_number] = s.renderings; return m;
}

// Model: either run the template program (args = buildProgram's) or, with args.steps, take a hand-written trace
// (stacks as "fn@line", all in one file) and answer the viewer questions at step args.at.
const parseStack = (st) => st.map(s => { const [fn, ln] = s.split("@"); return { path: "lecture.py", line_number: +ln, function_name: fn, code: "" }; });
function traceRunModel(args) {
  const viewer = (steps, at) => {
    if (at === undefined || !steps[at]) return {};
    const m = mergedEnv(steps, at);
    return { envCount: Object.keys(m.env).length, envKeys: Object.keys(m.env).join(","), stepOver: stepOverIndex(steps, at, 1), stepUp: stepUpIndex(steps, at) };
  };
  if (args.steps) return viewer(args.steps.map(s => ({ stack: parseStack(s.stack), env: s.env || {}, renderings: [] })), args.at);
  const T = recordTrace(buildProgram(args)), S = T.steps, P = T.P;
  const atLine6 = S.filter(s => s.stack.length === 1 && last(s.stack).line_number === 6);
  const calleeIds = T.events.filter(e => e.frame.fn === P.callee).map(e => e.frame.id);
  const calleeCalls = [...new Set(calleeIds)].reduce((a, id) => a + (T.calls[id] || 0), 0);
  return {
    steps: S.length, callLineSteps: atLine6.length, calleeSteps: S.filter(s => s.stack.length > 1).length,
    calleeTraceFuncPerCall: calleeCalls / P.n, renderingsOnCallLine: atLine6.reduce((a, s) => a + s.renderings.length, 0),
    zSteps: S.filter(s => "z" in s.env).length, hiddenCount: T.hidden["lec.py"].length, ...viewer(S, args.at),
  };
}
// The viewer-stepping-model check's trace, verbatim from the prompt.
const VIEWER_CHECK = [{ stack: ["main@10"] }, { stack: ["main@11"] }, { stack: ["main@11", "g@3"], env: { s: 1 } }, { stack: ["main@11", "g@4"], env: { t: 1 } },
  { stack: ["main@11"] }, { stack: ["main@12"] }, { stack: ["main@12", "g@3"], env: { s: 1 } }, { stack: ["main@12", "g@4"], env: { r: 1 } }];
MODELS["fixture:edtrace--trace-run"] = {
  fn: traceRunModel,
  cases: [
    // prompts
    { args: { n: 3, k: 3, txt: false }, pick: "callLineSteps", expect: 6, tol: 0, from: "edtrace:step-open-close:check" },
    { args: { n: 4, k: 2, txt: false, stepover: true }, pick: "calleeSteps", expect: 6, tol: 0, from: "edtrace:stepover-hide:check" },
    { args: { n: 1, k: 5, txt: false, other: true }, pick: "calleeTraceFuncPerCall", expect: 7, tol: 0, from: "edtrace:supp-sys-settrace:check" },
    { args: { steps: VIEWER_CHECK, at: 7 }, pick: "envCount", expect: 2, tol: 0, from: "edtrace:viewer-stepping-model:check" },
    { args: { n: 1, k: 2, txt: false }, pick: "callLineSteps", expect: 2, tol: 0 },          // step-open-close predict: 6 (open), def, body 1, body 2, 6 (close)
    { args: { n: 1, k: 2, txt: false }, pick: "calleeSteps", expect: 3, tol: 0 },            //   ... the def line plus two body lines
    { args: { n: 1, k: 2, txt: false }, pick: "zSteps", expect: 1, tol: 0 },                 //   ... and only the close step carries z
    { args: { n: 1, k: 2, txt: false, other: true }, pick: "callLineSteps", expect: 1, tol: 0 }, // step-open-close transfer: one step, still with z
    { args: { n: 1, k: 2, txt: false, other: true }, pick: "zSteps", expect: 1, tol: 0 },
    { args: { n: 1, k: 3, txt: true, other: true }, pick: "calleeSteps", expect: 0, tol: 0 },          // execute-settrace-loop predict: zero steps for the helper ...
    { args: { n: 1, k: 3, txt: true, other: true }, pick: "renderingsOnCallLine", expect: 1, tol: 0 }, //   ... and its rendering lands on the calling line
    { args: { n: 1, k: 3, txt: false, stepover: true }, pick: "calleeSteps", expect: 0, tol: 0 },      // stepover-hide predict: no steps for the body ...
    { args: { n: 1, k: 3, txt: false, stepover: true }, pick: "callLineSteps", expect: 1, tol: 0 },    //   ... one step for the line ...
    { args: { n: 1, k: 3, txt: false, stepover: true }, pick: "zSteps", expect: 1, tol: 0 },           //   ... showing total (z here)
    { args: { n: 1, k: 3, txt: false, hide: true }, pick: "calleeSteps", expect: 4, tol: 0 },          // stepover-hide transfer: @hide removes no step (def + 3 body lines)
    { args: { n: 1, k: 3, txt: false, hide: true }, pick: "hiddenCount", expect: 1, tol: 0 },
    { args: { n: 10, k: 1, txt: false, other: true }, pick: "callLineSteps", expect: 10, tol: 0 },     // limits predict: one step per iteration for a line with no traced call
    { args: { n: 1, k: 3, txt: false, at: 2 }, pick: "stepOver", expect: 7, tol: 0 },                 // viewer predict: j from line 6's open step lands on its close step
    { args: { n: 1, k: 3, txt: false, at: 6 }, pick: "stepUp", expect: 7, tol: 0 },                   // viewer transfer: u from the last body line lands on the caller's close step
    // edges
    { args: { n: 2, k: 3, txt: false, stepover: true }, pick: "calleeSteps", expect: 4, tol: 0 },     // stepover toggles: iteration 2 is recorded (def + 3)
    { args: { n: 5, k: 3, txt: false, stepover: true }, pick: "calleeSteps", expect: 8, tol: 0 },     // iterations 2 and 4 recorded
    { args: { n: 1, k: 3, txt: false }, pick: "calleeTraceFuncPerCall", expect: 5, tol: 0 },         // same-file callee: also k + 2 (call; each line and the return reach it via local_trace_func's forward, L338)
    { args: { n: 3, k: 3, txt: false }, pick: "steps", expect: 24, tol: 0 },                                    // total, as the real run printed
    { args: { steps: VIEWER_CHECK, at: 7 }, pick: "stepUp", expect: 8, tol: 0 },                      // no later ancestor step: index runs off the end
  ],
};
WIDGETS["fixture:edtrace--trace-run"] = (root) => {
  const s = { n: 2, k: 3, txt: true, other: false, stepover: false, hide: false }, view = { cur: 0 };
  const listing = el("div"), viewer = el("div"), evlog = el("div"), read = el("div", { class: "readout" });
  let T;
  const KIND = { open: "var(--ok)", close: "var(--accent)", fill: "var(--accent)", merge: "var(--muted)", skip: "var(--muted)" };
  const rerun = () => { T = recordTrace(buildProgram(s)); view.cur = Math.min(view.cur, T.steps.length - 1); draw(); };
  const go = (i) => { if (i >= 0 && i < T.steps.length) { view.cur = i; draw(); } };
  const draw = () => {
    const S = T.steps, cur = S[view.cur], P = T.P, here = last(cur.stack), lr = lineRenderings(S), hid = T.hidden["lec.py"];
    // listing: both files; the current step's line is highlighted; renderings replace a line as the viewer does
    const fileBox = (file) => {
      const lines = P.files[file], vis = file === "lec.py";
      const rows = lines.map((code, i) => {
        const ln = i + 1, isCur = vis && here.path === file && here.line_number === ln, isHid = vis && hid.includes(ln);
        const rs = vis ? (lr[ln] || []) : [];
        const shown = rs.length ? ` <span style="color:var(--accent2)">⇒ viewer shows ${rs.map(r => `“${esc(r.data)}”`).join(" ")}</span>` : "";
        return `<div style="white-space:pre-wrap;${isCur ? "background:#fdf1cf;" : ""}${isHid ? "color:#aaa;text-decoration:line-through;" : ""}"><span style="color:var(--muted);display:inline-block;width:2.2em">${ln}</span>${esc(code)}${isHid ? ' <span style="color:var(--accent2);text-decoration:none">(@hide: not drawn)</span>' : ""}${shown}</div>`;
      }).join("");
      const head = vis ? `<b>${file}</b> <span class="muted">· in visible_paths (execute.py:L345)</span>` : `<b>${file}</b> <span class="muted">· not in visible_paths: its frames are never recorded${P.other ? "" : " (unused: line 6 calls f)"}</span>`;
      return `<div style="margin-bottom:6px">${head}</div><div style="${MONO}">${rows}</div>`;
    };
    listing.innerHTML = ""; const lb = box(); lb.innerHTML = fileBox("lec.py") + `<div style="height:8px"></div>` + fileBox("helpers.py"); listing.append(lb);
    // viewer: the current step record and the merged env panel
    const m = mergedEnv(S, view.cur), ov = stepOverIndex(S, view.cur, 1), ob = stepOverIndex(S, view.cur, -1), up = stepUpIndex(S, view.cur);
    const stackTxt = cur.stack.map(e => `${e.function_name}@${e.line_number}`).join(" → ");
    const envOwn = Object.keys(cur.env).length ? Object.entries(cur.env).map(([k2, v]) => `${k2}: ${v === null ? "null" : v}`).join(", ") : "{}";
    const merged = Object.keys(m.env).length ? Object.entries(m.env).map(([k2, v]) => `<b>${esc(k2)}</b> = ${esc(v)}`).join("<br>") : '<span class="muted">(empty: the viewer shows no env panel)</span>';
    viewer.innerHTML = "";
    const vb = box("margin-top:8px");
    vb.append(el("div", {}, button("h ←", () => go(view.cur - 1), "step backward"), " ", button("l →", () => go(view.cur + 1), "step forward"), " ",
      button("k", () => go(ob), "step over backward"), " ", button("j", () => go(ov), "step over forward"), " ", button("u", () => go(up), "step up"),
      el("span", { class: "muted small", style: "margin-left:8px" }, `step ${view.cur} of ${S.length - 1} · j → ${ov < S.length ? ov : "end"} · u → ${up < S.length ? up : "end"}`)));
    const rec = el("div", { style: `${MONO};margin-top:6px` });
    rec.innerHTML = `steps[${view.cur}] = { stack: [${esc(stackTxt)}], env: ${esc(envOwn)}, renderings: [${cur.renderings.map(r => `“${esc(r.data)}”`).join(", ")}] }${cur.close ? ' <span style="color:var(--accent)">(a close step)</span>' : ""}<br>
      current line = last element of stack: ${here.path}:${here.line_number} · env panel (renderEnv merges steps ${m.from.length ? m.from.join(", ") : "none"}):<br><div style="margin:4px 0 0 1em">${merged}</div>`;
    vb.append(rec); viewer.append(vb);
    // event log: every settrace event and what edtrace's tracer code did with it
    const rowsOf = {}; T.log.forEach(r => (rowsOf[r.ei] = rowsOf[r.ei] || []).push(r));
    const tr = T.events.map((e, ei) => {
      const rs = rowsOf[ei] || [], stepHit = rs.find(r => r.step === view.cur && (r.kind === "open" || r.kind === "close"));
      const what = rs.map(r => `<span style="color:${KIND[r.kind] || "inherit"}">${esc(r.fn)}: ${esc(r.what)}</span>${r.warn ? ` <span style="color:var(--bad)">${esc(r.warn)}</span>` : ""}`).join("<br>");
      const st = rs.find(r => r.step !== undefined && (r.kind === "open" || r.kind === "close"));
      return `<tr data-step="${st ? st.step : ""}" style="cursor:${st ? "pointer" : "default"};${stepHit ? "background:#fdf1cf" : ""}"><td>${ei}</td><td>${e.frame.fn} <span class="muted">${e.frame.file}</span></td><td>${e.event}</td><td>${e.line}</td><td>${what}</td></tr>`;
    }).join("");
    evlog.innerHTML = `<div class="small muted" style="margin-top:8px">settrace events in order (click a row that opened a step to jump to it; text()'s own frames in execute_util.py are left out, they are dropped at L249-L251):</div>
      <div style="max-height:280px;overflow:auto;border:1px solid var(--rule);border-radius:6px;background:#fff"><table style="border-collapse:collapse;width:100%;font-size:12px;font-family:var(--mono)"><thead><tr style="text-align:left;color:var(--muted)"><th>#</th><th>frame</th><th>event</th><th>line</th><th>tracer: what it did</th></tr></thead><tbody>${tr}</tbody></table></div>`;
    evlog.querySelectorAll("tr[data-step]").forEach(row => { if (row.dataset.step !== "") row.addEventListener("click", () => go(+row.dataset.step)); });
    // readout
    const R = traceRunModel(s);
    read.innerHTML = `<span class="big">${R.steps} steps</span>
      line 6 (the call line): <b>${R.callLineSteps}</b> step${R.callLineSteps === 1 ? "" : "s"} (${P.n} iteration${P.n === 1 ? "" : "s"}; ${R.callLineSteps === 2 * P.n ? "open + close each time" : R.callLineSteps === P.n ? "one each time: open and close coincide" : "pairs only on recorded iterations"})<br>
      steps inside ${P.callee}: <b>${R.calleeSteps}</b>${P.other ? " (helpers.py is not visible)" : P.stepover ? " (@stepover skips every other visit)" : ""}<br>
      trace_func calls per ${P.callee} frame: <b>${R.calleeTraceFuncPerCall}</b> (= k + 2: call, ${P.k} lines, return)<br>
      renderings on line 6's steps: <b>${R.renderingsOnCallLine}</b> · steps with z: ${R.zSteps}<br>
      hidden_line_numbers: [${hid.join(", ")}]${hid.length ? " (steps unchanged)" : ""}<br>
      <span class="muted small">provenance: fixture:edtrace--trace-run · execute.py:L232-L341 (trace_func, local_trace_func, stepovers), L343-L352 (execute), L356-L366 (hidden lines); execute_util.py:L40-L54, L141-L148 (text, pop_renderings); TraceViewer.jsx:L244-L275 (j, k, u), L355-L385 (env merge), L408-L428, L579-L582. The port reproduces real edtrace runs of this program (720 variants, python 3.11).</span>`;
  };
  const ctl = el("div", { class: "controls" },
    intSlider("loop iterations n", 1, 10, s.n, v => { s.n = v; rerun(); }),
    intSlider("callee body lines k (incl. return)", 1, 5, s.k, v => { s.k = v; rerun(); }),
    checkbox("callee's first line calls text()", s.txt, v => { s.txt = v; rerun(); }),
    checkbox("line 6 calls g from helpers.py (another file)", s.other, v => { s.other = v; rerun(); }),
    checkbox("@stepover on line 6", s.stepover, v => { s.stepover = v; rerun(); }),
    checkbox("@hide on line 6", s.hide, v => { s.hide = v; rerun(); }),
    el("div", { class: "small muted", style: "margin-top:4px" }, "k counts the return line; with text() on, k ≥ 2."), read);
  root.append(el("div", { class: "widget" }, ctl, el("div", {}, listing, viewer, evlog)));
  rerun();
};

// ============================================================ 2. directive-parse: one line's comment -> env ==
// parse_directives (L88-L109), get_inspect_expressions / get_clear_expressions (L112-L137), and what
// local_trace_func then writes into the close step's env (L310-L332): each @inspect expr whose first dotted part is a
// local gets a serialized value (dotted parts are followed with getattr), a missing one prints a warning and writes
// nothing, and every @clear expr is written as None afterwards (so @clear wins over @inspect for the same name).
function directiveModel({ line, locals = [] }) {
  const { directives, tokens, warnings, dropped } = parseDirectives(line);
  const L = new Set(locals), inspect = exprsOf(directives, "@inspect"), clear = exprsOf(directives, "@clear");
  const env = {}, warn = [...warnings];
  for (const expr of inspect) {
    const v = expr.split(".")[0];
    if (L.has(v)) env[expr] = expr.includes(".") ? `getattr chain on ${v}` : `value of ${v}`;
    else warn.push(`WARNING: variable ${v} not found in locals`);
  }
  for (const expr of clear) env[expr] = null;
  const values = Object.values(env).filter(x => x !== null).length;
  return { directives, tokens, dropped, inspect, clear, env, warn, values, nulls: Object.keys(env).length - values,
    nInspect: inspect.length, warnings: warn.length, stepover: directives.some(d => d.name === "@stepover") ? 1 : 0, hide: directives.some(d => d.name === "@hide") ? 1 : 0,
    afterHash: line.includes("#") ? line.split("#")[1] : null, nHash: line.split("#").length - 1 };
}
const CHECK_LINE = "loss = loss + 1  # note: watch @inspect loss w.real @clear old tmp @inspect lr";
MODELS["fixture:edtrace--directive-parse"] = {
  fn: directiveModel,
  cases: [
    { args: { line: CHECK_LINE, locals: ["loss", "w", "lr", "old", "tmp"] }, pick: "values", expect: 3, tol: 0, from: "edtrace:directive-grammar:check" },
    { args: { line: CHECK_LINE, locals: ["loss", "w", "lr", "old", "tmp"] }, pick: "nulls", expect: 2, tol: 0 },          // old, tmp -> None (real run: env had both as None)
    { args: { line: "a, b = 1, 2  # @inspect a @inspect b", locals: ["a", "b"] }, pick: "values", expect: 2, tol: 0 },     // predict: both a and b
    { args: { line: "a, b = 1, 2  # set both @inspect a b @clear c", locals: ["a", "b"] }, pick: "values", expect: 2, tol: 0 }, // transfer: prose dropped ...
    { args: { line: "a, b = 1, 2  # set both @inspect a b @clear c", locals: ["a", "b"] }, pick: "nulls", expect: 1, tol: 0 },  //   ... and c is None
    { args: { line: 'text("# Heading")  # @inspect x', locals: ["x"] }, pick: "nInspect", expect: 0, tol: 0 },          // KP note: split at the first '#', inside the string
    { args: { line: "x = 1  # @inpsect x", locals: ["x"] }, pick: "warnings", expect: 1, tol: 0 },                         // unknown @token: warning only (L101-L102)
    { args: { line: "assert torch.equal(w.grad, torch.tensor([1, 2, 3]))  # @inspect w.grad", locals: ["w"] }, pick: "values", expect: 1, tol: 0 },                      // lecture_02.py:L499, dotted
    { args: { line: 'text("Let\'s try a more complex example...")  # @clear x y z', locals: ["x", "y", "z"] }, pick: "nulls", expect: 3, tol: 0 },     // lecture_02.py:L234
    { args: { line: "x = 3  # @inspect x @clear x", locals: ["x"] }, pick: "values", expect: 0, tol: 0 },                  // clear runs after inspect
    { args: { line: "y = f(x)  # @inspect y @stepover", locals: ["y"] }, pick: "stepover", expect: 1, tol: 0 },
  ],
};
const DIRECTIVE_PRESETS = [
  ["docstring L92", '... # @inspect x y @hide', "x, y"],
  ["L116", 'x, y = str.split("a,b")  # @inspect x @inspect y', "x, y"],
  ["lecture_02 L234", 'text("Let\'s try a more complex example...")  # @clear x y z', "x, y, z"],
  ["lecture_02 L499", "assert torch.equal(w.grad, torch.tensor([1, 2, 3]))  # @inspect w.grad", "w"],
  ["'#' inside a string", 'text("# Heading")  # @inspect x', "x"],
  ["typo", "z = x + y  # @inpsect z", "x, y, z"],
];
WIDGETS["fixture:edtrace--directive-parse"] = (root) => {
  const s = { line: "w = w - lr * grad  # update @inspect w lr @clear grad", locals: "w, lr, grad" };
  const pic = el("div"), read = el("div", { class: "readout" });
  const lineIn = el("input", { type: "text", value: s.line, style: `width:100%;${MONO}` }), locIn = el("input", { type: "text", value: s.locals, style: `width:100%;${MONO}` });
  const draw = () => {
    const locals = s.locals.split(/[,\s]+/).filter(Boolean), m = directiveModel({ line: s.line, locals });
    const role = [];   // colour each token by what parse_directives does with it
    let open = null;
    for (const t of m.tokens) { if (t.startsWith("@")) { open = t; role.push([t, ACCEPTED_DIRECTIVES.includes(t) ? "var(--accent)" : "var(--bad)", ACCEPTED_DIRECTIVES.includes(t) ? "opens a directive" : "unknown: warning, still opens"]); } else role.push([t, open ? "var(--ok)" : "#aaa", open ? `arg of ${open}` : "before any @: dropped"]); }
    const parts = s.line.split("#");
    const lineHtml = parts.length < 2 ? esc(s.line) : `${esc(parts[0])}<b style="color:var(--accent2)">#</b><u>${esc(parts[1])}</u>${parts.length > 2 ? `<span style="color:#aaa">#${esc(parts.slice(2).join("#"))}</span>` : ""}`;
    pic.innerHTML = "";
    const b = box(); b.innerHTML = `<div class="small muted">the line; underlined = line.split("#")[1], the only text parse_directives reads (L97)${m.nHash > 1 ? "; grey text after a second '#' is never read" : ""}</div>
      <div style="${MONO};white-space:pre-wrap;margin:4px 0 8px">${lineHtml}</div>
      <div class="small muted">tokens, in order</div>
      <div style="display:flex;flex-wrap:wrap;gap:6px;margin:4px 0 8px">${role.length ? role.map(([t, c, why]) => `<span style="border:1px solid ${c};border-radius:4px;padding:1px 6px;${MONO}" title="${esc(why)}"><b style="color:${c}">${esc(t)}</b> <span class="muted" style="font-size:11px">${esc(why)}</span></span>`).join("") : '<span class="muted small">(no tokens: no directives)</span>'}</div>
      <div class="small muted">directives = [${m.directives.map(d => `Directive(${esc(d.name)}, [${d.args.map(esc).join(", ")}])`).join(", ")}]</div>
      <div class="small muted" style="margin-top:8px">close step's env after local_trace_func (inspect first, then clear)</div>
      <div style="${MONO}">${Object.keys(m.env).length ? Object.entries(m.env).map(([k2, v]) => `${esc(k2)}: ${v === null ? '<span style="color:var(--accent2)">None → the viewer deletes it</span>' : `<span style="color:var(--ok)">Value(${esc(v)})</span>`}`).join("<br>") : "{}"}</div>
      ${m.warn.length ? `<div style="${MONO};color:var(--bad);margin-top:6px">${m.warn.map(esc).join("<br>")}</div>` : ""}`;
    pic.append(b);
    read.innerHTML = `<span class="big">${m.values} value${m.values === 1 ? "" : "s"}</span>${m.nulls} set to None · ${m.inspect.length} @inspect expr${m.inspect.length === 1 ? "" : "s"}, ${m.clear.length} @clear<br>
      @stepover: ${m.stepover ? "yes (recording, not execution)" : "no"} · @hide: ${m.hide ? "yes (viewer only)" : "no"}<br>
      <span class="muted small">provenance: fixture:edtrace--directive-parse · execute.py:L88-L109 (parse_directives), L112-L137, L310-L332 (env fill, clear), TraceViewer.jsx:L380-L385 (null deleted). Values are not computed here: only which keys get a value.</span>`;
  };
  lineIn.addEventListener("input", () => { s.line = lineIn.value; draw(); });
  locIn.addEventListener("input", () => { s.locals = locIn.value; draw(); });
  const presets = el("div", { style: "margin-top:8px;display:flex;flex-wrap:wrap;gap:4px" }, ...DIRECTIVE_PRESETS.map(([lab, line, loc]) => button(lab, () => { lineIn.value = s.line = line; locIn.value = s.locals = loc; draw(); })));
  root.append(el("div", { class: "widget" }, el("div", { class: "controls" },
    el("label", {}, "source line", lineIn), el("label", {}, "names in f_locals after the line runs", locIn), presets, read), pic));
  draw();
};

// ======================================================= 3. value-render: Value record -> what the viewer draws ==
// renderNumber (TraceViewer:L466-L481), verbatim order of tests. nan/inf reach it as strings (execute.py:L154-L155).
export function renderNumber(x) {
  if (typeof x === "string") return { text: x, branch: 0, why: "a string (nan/inf were serialized as text, L154-L155): passed through" };
  if (Math.abs(x) > 1e12) return { text: x.toExponential(3), branch: 1, why: "|x| > 1e12 → x.toExponential(3)" };
  if (Math.abs(x) > 1e6) return { text: x.toLocaleString(), branch: 2, why: "|x| > 1e6 → x.toLocaleString() (thousands separators)" };
  if ((x * 1000) % 1 === 0) return { text: x.toString(), branch: 3, why: "x·1000 is an integer (≤ 3 decimals) → x.toString(), exact" };
  return { text: x.toFixed(4), branch: 4, why: "otherwise → x.toFixed(4)" };
}
// renderTensor's layout (L483-L520) as a count of <tr> rows, walking the loop on the shape: rank 0 a number, rank 1 one
// row, rank 2 one row per row, rank 3 the slices stacked with a separator row before every slice after the first
// (pushed only once some row exists), rank ≥ 4 JSON.stringify text. dtype plays no part except in the title (L544-L553).
export function tensorLayout(shape) {
  const r = shape.length;
  if (r === 0) return { rows: 0, kind: "scalar: renderNumber(contents)" };
  if (r === 1) return { rows: 1, kind: "rank 1: one-row table" };
  if (r === 2) return { rows: shape[0], kind: "rank 2: a table, one <tr> per row" };
  if (r === 3) { let rows = 0, sep = 0; for (let i = 0; i < shape[0]; i++) { if (rows > 0) { rows++; sep++; } rows += shape[1]; } return { rows, sep, kind: "rank 3: slices stacked, a blank separator row between them" }; }
  return { rows: 0, kind: `rank ${r}: no table, JSON.stringify(contents) as text` };
}
function valueModel({ x, shape = [2, 3], recorded = 1 }) {
  const o = {};
  if (x !== undefined) { const n = renderNumber(x); o.text = n.text; o.branch = n.branch; o.shownNumber = parseFloat(String(n.text).replace(/,/g, "")); }
  const L = tensorLayout(shape), numbers = shape.reduce((a, b) => a * b, 1);   // tolist() keeps every element (L164-L166)
  return { ...o, rows: L.rows, numbers, total: numbers * recorded, rank: shape.length };
}
MODELS["fixture:edtrace--value-render"] = {
  fn: valueModel,
  cases: [
    { args: { shape: [3, 2, 4] }, pick: "rows", expect: 8, tol: 0, from: "edtrace:viewer-value-rendering:check" },
    { args: { x: 0.125 }, pick: "shownNumber", expect: 0.125, tol: 0 },          // viewer-value-rendering predict: exact
    { args: { x: 2.718281828 }, pick: "shownNumber", expect: 2.7183, tol: 0 },   //   ... toFixed(4)
    { args: { x: 3.5e13 }, pick: "branch", expect: 1, tol: 0 },                  //   ... toExponential(3) = 3.500e+13
    { args: { x: 3.5e13 }, pick: "shownNumber", expect: 3.5e13, tol: 0 },
    { args: { x: "nan" }, pick: "branch", expect: 0, tol: 0 },                   // KP misconception: nan is the string 'nan', not null
    { args: { shape: [2, 3] }, pick: "numbers", expect: 6, tol: 0 },             // inspect-serialization predict: torch.zeros((2, 3)), contents [[0,0,0],[0,0,0]]
    { args: { shape: [2, 3] }, pick: "rows", expect: 2, tol: 0 },
    { args: { shape: [3, 50000] }, pick: "numbers", expect: 150000, tol: 0 },    // inspect-serialization check: 3 lists of 50000 numbers
    { args: { shape: [4096, 4096], recorded: 10 }, pick: "total", expect: 167772160, tol: 0 },   // limits predict: about 10 × 16.8 million
    { args: { shape: [] }, pick: "rows", expect: 0, tol: 0 },                    // edges: a 0-d tensor is drawn as a number
    { args: { shape: [5] }, pick: "rows", expect: 1, tol: 0 },
    { args: { shape: [1, 4, 4] }, pick: "rows", expect: 4, tol: 0 },             // one slice: no separator
    { args: { shape: [2, 0, 4] }, pick: "rows", expect: 0, tol: 0 },             // empty slices: the separator needs an earlier row
    { args: { shape: [2, 2, 2, 2] }, pick: "rows", expect: 0, tol: 0 },          // rank 4: JSON text
    { args: { x: 1234567 }, pick: "branch", expect: 2, tol: 0 },
    { args: { x: 0.00004 }, pick: "shownNumber", expect: 0, tol: 0 },          // x·1000 = 0.04 is not an integer → toFixed(4) = "0.0000"
  ],
};
WIDGETS["fixture:edtrace--value-render"] = (root) => {
  const s = { num: "1.5", shape: "2, 4", type: "torch.Tensor", dtype: "torch.float32", recorded: 1 };
  const numBox = el("div"), tenBox = el("div", { style: "margin-top:8px" }), read = el("div", { class: "readout" });
  const parseNum = (t) => { const u = t.trim(); return /^-?(nan|inf)$/i.test(u) ? u.toLowerCase() : Number(u); };
  const parseShape = (t) => t.trim() === "" ? [] : t.split(/[,x×\s]+/).filter(Boolean).map(v => Math.max(0, Math.floor(+v))).filter(Number.isFinite);
  const draw = () => {
    const x = parseNum(s.num), sh = parseShape(s.shape), valid = typeof x === "string" || Number.isFinite(x);
    const n = valid ? renderNumber(x) : null, L = tensorLayout(sh), m = valueModel({ shape: sh, recorded: s.recorded });
    numBox.innerHTML = ""; const nb = box();
    nb.innerHTML = `<div class="small muted">a Python ${typeof x === "string" ? "float (nan/inf)" : "int or float"} inspected → Value(type, contents) → renderNumber(contents)</div>` + (n ? `
      <div style="${MONO};margin-top:4px">contents: ${esc(typeof x === "string" ? `"${x}"` : String(x))} → <b style="font-size:16px">${esc(n.text)}</b></div>
      <div class="small" style="margin-top:4px">${[1, 2, 3, 4].map(b => `<span style="${b === n.branch ? "font-weight:700;color:var(--accent)" : "color:#aaa"}">${b}. ${["", "|x| > 1e12: toExponential(3)", "|x| > 1e6: toLocaleString()", "x·1000 integer: toString()", "else toFixed(4)"][b]}</span>`).join(" &nbsp; ")}${n.branch === 0 ? ' <b style="color:var(--accent)">string: passed through</b>' : ""}</div>` : '<div class="muted">not a number</div>');
    numBox.append(nb);
    // tensor: the Value record and the table renderTensor would build
    const prod = m.numbers, title = `${s.type} ${s.dtype}${sh.length ? ` [${sh.join(" x ")}]` : " []"}`;
    const small = sh.length <= 3 && sh.every(d => d <= 8) && prod <= 200;
    let table = "";
    if (sh.length >= 1 && sh.length <= 3 && small) {
      const cell = () => `<td style="border:1px solid var(--rule);padding:1px 6px">0</td>`;
      const rowsHtml = [];
      if (sh.length === 1) rowsHtml.push(`<tr>${Array(sh[0]).fill(0).map(cell).join("")}</tr>`);
      if (sh.length === 2) for (let i = 0; i < sh[0]; i++) rowsHtml.push(`<tr>${Array(sh[1]).fill(0).map(cell).join("")}</tr>`);
      if (sh.length === 3) for (let k2 = 0; k2 < sh[0]; k2++) { if (rowsHtml.length) rowsHtml.push(`<tr><td colspan="${sh[2]}" style="color:var(--accent2);font-size:10px">&nbsp;separator&nbsp;</td></tr>`); for (let i = 0; i < sh[1]; i++) rowsHtml.push(`<tr>${Array(sh[2]).fill(0).map(cell).join("")}</tr>`); }
      table = `<table title="${esc(title)}" style="border-collapse:collapse;${MONO};font-size:11px;margin-top:6px">${rowsHtml.join("")}</table>`;
    } else if (sh.length === 0) table = `<div style="${MONO};margin-top:6px">0 <span class="muted">(renderNumber(0): JSON 0.0 arrives as the number 0)</span></div>`;
    else if (sh.length >= 4) table = `<div style="${MONO};margin-top:6px;color:var(--muted)">[[[[0, 0, …]]]] <span>(JSON.stringify(contents, null, 2) as plain text)</span></div>`;
    else table = `<div class="muted small" style="margin-top:6px">(too large to draw here: ${L.rows.toLocaleString()} rows)</div>`;
    tenBox.innerHTML = ""; const tb = box();
    tb.innerHTML = `<div class="small muted">x = (a ${esc(s.type)} of shape [${sh.join(", ")}])  # @inspect x → the Value record (execute.py:L163-L166), then renderTensor (TraceViewer:L483-L520)</div>
      <div style="${MONO};margin-top:4px">Value(type="${esc(s.type)}", dtype="${esc(s.dtype)}", shape=[${sh.join(", ")}], contents=x.tolist() → <b>${prod.toLocaleString()}</b> number${prod === 1 ? "" : "s"}, all of them)</div>
      <div class="small" style="margin-top:6px">layout: <b>${esc(L.kind)}</b> → <b>${L.rows}</b> &lt;tr&gt; row${L.rows === 1 ? "" : "s"}${L.sep ? ` (${L.rows - L.sep} data + ${L.sep} separator)` : ""} · hover title: “${esc(title)}”</div>${table}`;
    tenBox.append(tb);
    read.innerHTML = `${n ? `<span class="big">${esc(n.text)}</span>` : ""}tensor: <b>${L.rows}</b> table rows · <b>${prod.toLocaleString()}</b> numbers per recorded step<br>
      recorded at ${s.recorded} step${s.recorded === 1 ? "" : "s"} (e.g. a loop): <b>${m.total.toLocaleString()}</b> numbers in the trace file<br>
      <span class="muted small">provenance: fixture:edtrace--value-render · execute.py:L147-L166 (to_serializable_value: nan/inf as strings, ndarray/Tensor keep dtype, shape and the full tolist(); no size limit), TraceViewer.jsx:L434-L520 (renderValue, renderNumber, renderTensor), L544-L553 (title). The zeros stand in for any values: the layout reads only the shape.</span>`;
  };
  const numIn = el("input", { type: "text", value: s.num, style: `width:100%;${MONO}` }), shIn = el("input", { type: "text", value: s.shape, style: `width:100%;${MONO}` });
  numIn.addEventListener("input", () => { s.num = numIn.value; draw(); }); shIn.addEventListener("input", () => { s.shape = shIn.value; draw(); });
  const sel = (opts, key) => { const e = el("select", {}, ...opts.map(o => el("option", { value: o }, o))); e.value = s[key]; e.addEventListener("change", () => { s[key] = e.value; draw(); }); return e; };
  const numPresets = el("div", { style: "display:flex;flex-wrap:wrap;gap:4px;margin-top:4px" }, ...["0.001", "3.14159", "123456789", "2e15", "nan", "0.00004"].map(v => button(v, () => { numIn.value = s.num = v; draw(); })));
  root.append(el("div", { class: "widget" }, el("div", { class: "controls" },
    el("label", {}, "an inspected number", numIn), numPresets,
    el("label", {}, "tensor shape (e.g. 2, 3 or 4, 2, 3; empty = 0-d)", shIn),
    el("label", {}, "type ", sel(["torch.Tensor", "torch.nn.parameter.Parameter", "numpy.ndarray"], "type")),
    el("label", {}, "dtype ", sel(["torch.float32", "torch.float16", "torch.bfloat16", "torch.int64", "float64", "int32"], "dtype")),
    intSlider("steps that record it", 1, 20, s.recorded, v => { s.recorded = v; draw(); }), read),
    el("div", {}, numBox, tenBox)));
  draw();
};
