import { WIDGETS } from "./widgets.js?v=3";
import { judge } from "./grade.js?v=1";
import { loadCards, saveCards, review, dueCards, retrievability, nextInterval } from "./fsrs.js";

// ------------------------------------------------------------------ util --
const $ = (s, r = document) => r.querySelector(s);
const el = (tag, attrs = {}, ...kids) => {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "class") e.className = v; else if (k.startsWith("on")) e.addEventListener(k.slice(2), v); else if (k === "html") e.innerHTML = v; else e.setAttribute(k, v);
  }
  for (const k of kids.flat()) if (k !== null && k !== undefined) e.append(k instanceof Node ? k : document.createTextNode(String(k)));
  return e;
};
const esc = s => String(s ?? "").replace(/[&<>]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;" })[c]);
const state = { threads: null, data: {}, cards: loadCards(), progress: loadProgress(), builds: {} };
function loadProgress() { try { return JSON.parse(localStorage.getItem("atlas.progress.v1") || "{}"); } catch { return {}; } }
function saveProgress() { try { localStorage.setItem("atlas.progress.v1", JSON.stringify(state.progress)); } catch {} }
function setting(k, dflt) { try { const v = localStorage.getItem("atlas.set." + k); return v === null ? dflt : v === "1"; } catch { return dflt; } }
function setSetting(k, v) { try { localStorage.setItem("atlas.set." + k, v ? "1" : "0"); } catch {} }
function participant() { try { return localStorage.getItem("atlas.participant") || ""; } catch { return ""; } }
function logEvent(type, payload = {}) {
  try {
    const log = JSON.parse(localStorage.getItem("atlas.events.v1") || "[]");
    log.push({ t: Date.now(), type, ...payload });
    localStorage.setItem("atlas.events.v1", JSON.stringify(log.slice(-5000)));
  } catch {}
}
function exportBundle() {
  let events = []; try { events = JSON.parse(localStorage.getItem("atlas.events.v1") || "[]"); } catch {}
  return { format: "atlas-learner-export/1", exported_at: new Date().toISOString(), participant: participant(),
    threads: Object.fromEntries(Object.entries(state.data).map(([id, d]) => [id, { version: d.kp.version, n_kps: d.kp.knowledge_points.length }])),
    progress: state.progress, cards: state.cards, events };
}
async function getJSON(p) { const r = await fetch(p, { cache: "no-cache" }); if (!r.ok) throw new Error(`${r.status} ${p}`); return r.json(); }
async function tryJSON(p) { try { return await getJSON(p); } catch { return null; } }

// -------------------------------------------------------------- sources --
function videoSecs(t) { const p = t.split(":").map(Number); while (p.length < 3) p.unshift(0); return p[0] * 3600 + p[1] * 60 + p[2]; }
const VIDEO_REF = /^video:((?:\d+:)?\d+:\d\d)(?:-((?:\d+:)?\d+:\d\d))?$/;
function anchorHref(sources, a) {
  const src = sources?.[a.source];
  const vm = VIDEO_REF.exec(a.ref || "");
  if (vm) { const url = src?.url || Object.values(sources || {}).find(x => /youtube\.com\/watch/.test(x.url || ""))?.url; return url ? `${url}&t=${videoSecs(vm[1])}s` : null; }
  if (!src) return null;
  const m = /^([\w.\-/]+):L(\d+)(?:-L(\d+))?/.exec(a.ref || "");
  if (src.upstream && m) return `${src.upstream.replace(/\/[^/]*$/, "/")}${m[1]}#L${m[2]}${m[3] ? "-L" + m[3] : ""}`;
  if (src.url) return src.url;
  return null;
}
function anchorText(a) { return a.ref; }

// ----------------------------------------------------------------- views --
async function boot() {
  state.threads = await getJSON("threads.json");
  // per-thread widgets: site/widgets.d/<thread>.js, each exporting WIDGETS (one writer per file)
  await Promise.all(state.threads.threads.map(t => import(`./widgets.d/${t.id}.js?v=${state.threads.widgets_v || 1}`)
    .then(m => Object.assign(WIDGETS, m.WIDGETS || {})).catch(e => console.warn(`widgets.d/${t.id}.js`, e))));
  await Promise.all(state.threads.threads.map(async t => {
    const [kp, sources, discovered, snippets] = await Promise.all([tryJSON(t.kp), tryJSON(t.sources), tryJSON(t.discovered),
      tryJSON(t.kp.replace(/knowledge_points\.json$/, "snippets.json"))]);
    if (!kp) { t.missing = true; return; }
    state.data[t.id] = { kp, sources: sources || {}, discovered, snippets: snippets || {} };
  }));
  window.addEventListener("hashchange", route); route();
}
function route() {
  updateDue();
  const h = location.hash.replace(/^#\/?/, "");
  const [a, b, c] = h.split("/");
  const app = $("#app"); app.innerHTML = "";
  if (!a) return renderHome(app);
  if (a === "review") return renderReview(app);
  if (a === "data") return renderData(app);
  if (a === "session") return renderSession(app, b);
  if (a === "map") return renderMap(app);
  if (a === "t" && state.data[b]) return renderThread(app, b, c);
  renderHome(app);
}
function updateDue() { const n = dueCards(state.cards).length; $("#due-pill").textContent = `reviews due: ${n}`; }

function renderHome(app) {
  app.className = "";
  const main = el("div", { class: "main", style: "max-width:1100px;margin:0 auto" },
    el("h1", {}, "Atlas"),
    el("p", { class: "muted" }, "One pipeline, four inputs. Each thread is a graph of knowledge points with provenance back to the source and a commit-before-reveal learning loop. Pick a thread; the side list is the professor's (or paper's, or call graph's) order; the page is one knowledge point."),
    el("p", { class: "small" }, el("a", { href: "#/map" }, "Course map"), " — all lecture threads in course order with cross-lecture prerequisite edges."),
    ...["lecture", "paper", "repo"].flatMap(kind => [el("h2", {}, { lecture: "Lectures", paper: "Papers", repo: "Repositories" }[kind]), el("div", { class: "threads" }, ...state.threads.threads.filter(t => t.kind === kind).map(t => {
      const d = state.data[t.id]; const n = d ? d.kp.knowledge_points.length : 0; const mastered = d ? d.kp.knowledge_points.filter(k => masteryOf(t.id, k.id) === "mastered").length : 0;
      return el("div", { class: `tcard ${t.missing ? "missing" : ""}`, onclick: () => { if (!t.missing) location.hash = `#/t/${t.id}`; } }, el("div", { class: "kind" }, t.kind), el("div", { style: "font-weight:700" }, t.title), el("div", { class: "small muted" }, t.subtitle), el("div", { class: "small", style: "margin-top:8px" }, t.missing ? "not built yet" : `${n} knowledge points · ${mastered} mastered`));
    }))]),
    el("div", { class: "threads", style: "display:none" }, ...state.threads.threads.map(t => {
      const d = state.data[t.id];
      const n = d ? d.kp.knowledge_points.length : 0;
      const mastered = d ? d.kp.knowledge_points.filter(k => masteryOf(t.id, k.id) === "mastered").length : 0;
      return el("div", { class: `tcard ${t.missing ? "missing" : ""}`, onclick: () => { if (!t.missing) location.hash = `#/t/${t.id}`; } },
        el("div", { class: "kind" }, t.kind), el("div", { style: "font-weight:700" }, t.title), el("div", { class: "small muted" }, t.subtitle),
        el("div", { class: "small", style: "margin-top:8px" }, t.missing ? "not built yet" : `${n} knowledge points · ${mastered} mastered`));
    })),
    el("h2", {}, "Sessions"), el("p", { class: "small" }, el("a", { href: "#/session" }, "Guided sessions"), " — a fixed sequence for one sitting, with strict gating and a participant id. Use these for a real learner."),
    el("h2", {}, "How to read a thread"),
    el("ul", { class: "small" },
      el("li", {}, el("b", {}, "Teacher line: "), "the ", el("i", {}, "filtering"), " tab compares everything the source touched (discovered evidence) with what became a knowledge point, so you can see what was said, computed, cited, shown, deferred, or dropped."),
      el("li", {}, el("b", {}, "Student line: "), "each knowledge point opens with a prediction you must commit before anything is revealed; then the mechanism (code, formula, interactive, or a steppable animation with one ", el("i", {}, "notice"), "), then a changed-case transfer question, then retrieval prompts that are scheduled with FSRS and come back in ", el("a", { href: "#/review" }, "reviews"), "."),
      el("li", {}, el("b", {}, "Evidence discipline: "), "every anchor links to a source line or page; dashed chips are author inference, solid chips are evidence from the source.")),
  );
  app.append(main);
}

function masteryOf(tid, kid) {
  const p = state.progress[`${tid}:${kid}`];
  if (!p) return "new";
  const kpDef = state.data[tid]?.kp.knowledge_points.find(k => k.id === kid);
  const needsCheck = !!kpDef?.prompts?.check;
  if (p.transfer === "ok" && (!needsCheck || p.check === "ok")) {
    // secured: a retrieval prompt remembered ≥2 days after the changed case
    const secured = Object.entries(state.cards).some(([id, c]) => id.startsWith(`${tid}:${kid}:`) && c.reps >= 2 && c.lapses === 0 && (c.last - (p.transfer_t || 0)) > 2 * 86400e3);
    return secured ? "secured" : "mastered";
  }
  if (p.predict) return "seen";
  return "new";
}
function readiness(tid, kp, kps) { const m = masteryOf(tid, kp.id); if (m === "mastered" || m === "secured") return m; return prereqsMet(tid, kp, kps) ? "ready" : "locked"; }
function prereqsMet(tid, kp, kps) {
  const local = (kp.prerequisites || []).every(p => ["mastered","secured"].includes(masteryOf(tid, p.kp)) || !kps.find(k => k.id === p.kp));
  const ext = (kp.external_prerequisites || []).every(e => { const [t2, k2] = String(e.kp || e).split(":"); return !state.data[t2] || ["mastered","secured"].includes(masteryOf(t2, k2)); });
  return local && ext;
}

function renderThread(app, tid, kid) {
  app.className = "layout";
  const t = state.threads.threads.find(x => x.id === tid), d = state.data[tid];
  const kps = d.kp.knowledge_points, byId = Object.fromEntries(kps.map(k => [k.id, k]));
  const stops = d.kp.path?.stops || kps.map(k => k.id), supps = d.kp.path?.supplements || [];
  const side = el("div", { class: "side" },
    el("div", { class: "small muted" }, t.kind), el("div", { style: "font-weight:700;margin-bottom:8px" }, t.title),
    el("div", { class: "small", style: "margin-bottom:8px" },
      el("a", { href: `#/t/${tid}/_filter` }, "filtering view"), " · ", t.kind === "lecture" ? [el("a", { href: `#/t/${tid}/_spoken` }, "spoken"), " · "] : null, el("a", { href: `#/t/${tid}/_graph` }, "graph"), " · ", el("a", { href: `#/t/${tid}/_gallery` }, "widgets"), d.kp.course_path ? [" · ", el("a", { href: `#/t/${tid}/_paths` }, "two orders")] : null),
    el("h3", {}, d.kp.path?.basis ? `path (${d.kp.path.basis})` : "path"),
    el("ul", { class: "path" }, ...stops.map((id, i) => byId[id] && el("li", { class: id === kid ? "active" : "", onclick: () => location.hash = `#/t/${tid}/${id}` },
      el("span", { class: "n" }, i + 1), el("span", { class: `dot ${masteryOf(tid, id)}` }), el("span", {}, byId[id].label)))),
    supps.length ? [el("h3", {}, "supplements (student line)"), el("ul", { class: "path" }, ...supps.map(id => byId[id] && el("li", { class: `supp ${id === kid ? "active" : ""}`, onclick: () => location.hash = `#/t/${tid}/${id}` }, el("span", { class: "n" }, "+"), el("span", { class: `dot ${masteryOf(tid, id)}` }), el("span", {}, byId[id].label))))] : null,
  );
  const main = el("div", { class: "main" });
  app.append(side, main);
  if (!kid) kid = stops.find(id => masteryOf(tid, id) !== "mastered") || stops[0];
  if (kid === "_filter") return renderFilter(main, tid);
  if (kid === "_graph") return renderGraph(main, tid);
  if (kid === "_paths") return renderPaths(main, tid);
  if (kid === "_spoken") return renderSpoken(main, tid);
  if (kid === "_gallery") return renderGallery(main, tid);
  renderKP(main, tid, byId[kid], kps);
  renderMath(main);
}

// ------------------------------------------------------------ KP page --
// Short free-text answers are self-graded after the reveal (Quantum Country style): the round-3
// audit showed token overlap rejects most correct paraphrases. number / choice / trend answers,
// and short prompts that carry an explicit `accept` list, are graded automatically.
function stableShuffle(arr, key) {
  let h = 2166136261; for (const ch of key) { h ^= ch.charCodeAt(0); h = Math.imul(h, 16777619) >>> 0; }
  const a = arr.slice();
  for (let i = a.length - 1; i > 0; i--) { h = Math.imul(h ^ (h >>> 15), 2246822519) >>> 0; const j = h % (i + 1); [a[i], a[j]] = [a[j], a[i]]; }
  return a;
}
function selfGraded(p) { return (p.format || "short") === "short" && !Array.isArray(p.accept); }
function promptBlock(tid, kp, name, p, opts = {}) {
  const key = `${tid}:${kp.id}`; state.progress[key] ||= {};
  const prog = state.progress[key];
  const box = el("div", { class: `block gate ${opts.locked ? "locked" : ""}` });
  box.append(el("div", { class: "eyebrow" }, opts.eyebrow || name));
  box.append(el("div", { class: "q" }, p.q));
  const reveal = el("div", { class: "reveal", style: "display:none" });
  const verdictLine = v => v === "pending" ? el("div", { class: "verdict" }, "Compare your answer with this one:")
    : el("div", { class: `verdict ${v === "ok" ? "ok" : "bad"}` }, v === "ok" ? (prog[`${name}_self`] ? "You marked this as matching." : "That matches.") : "Here is what actually happens:");
  const showReveal = () => {
    const v = prog[name];
    reveal.style.display = ""; reveal.innerHTML = "";
    // native append prints null as text, so drop absent parts
    reveal.append(...[verdictLine(v), el("div", { class: "yours" }, `you said: ${prog[`${name}_given`]}`), el("div", {}, el("b", {}, "answer: "), p.answer),
      p.why ? el("div", { class: "small", style: "margin-top:4px" }, p.why) : null,
      p.targets_misconception ? el("div", { class: "small muted" }, "targets: ", p.targets_misconception) : null].filter(Boolean));
    if (v === "pending") {
      const grade = ok => {
        prog[name] = ok ? "ok" : "miss"; prog[`${name}_self`] = true; saveProgress();
        logEvent("self_grade", { thread: tid, kp: kp.id, prompt: name, verdict: prog[name], auto: prog[`${name}_auto`] });
        showReveal();
        if (opts.onGrade) opts.onGrade(prog[name]);
      };
      reveal.append(el("div", { class: "selfgrade" }, el("span", { class: "small muted" }, "Be strict: does yours contain the same mechanism, not just the same words? "),
        el("button", { class: "primary", onclick: () => grade(true) }, "mine says the same thing"), " ",
        el("button", { onclick: () => grade(false) }, "mine is wrong or missing something")));
    }
  };
  const commit = (given) => {
    const auto = judge(p, given) ? "ok" : "miss";
    prog[`${name}_auto`] = auto; prog[`${name}_given`] = given; prog[`${name}_t`] = Date.now();
    prog[name] = selfGraded(p) ? "pending" : auto; saveProgress();
    logEvent("commit", { thread: tid, kp: kp.id, prompt: name, given, verdict: prog[name], auto });
    showReveal();
    form.querySelectorAll("button,input").forEach(x => x.disabled = true);
    if (opts.onCommit) opts.onCommit(prog[name]);
    if (prog[name] !== "pending" && opts.onGrade) opts.onGrade(prog[name]);
  };
  let form;
  if (p.format === "choice") {
    // stable per-item shuffle: authors tend to list the correct option first (round-4 audit: 92%)
    const order = stableShuffle(p.choices || [], `${tid}/${kp.id}/${name}`);
    form = el("div", { class: "choices" }, ...order.map(c => el("button", { onclick: () => commit(c) }, c)));
  } else {
    const inp = el("input", { type: "text", placeholder: p.format === "number" ? "number" : p.format === "trend" ? "up / down / same" : "short answer" });
    inp.addEventListener("keydown", e => { if (e.key === "Enter" && inp.value.trim()) commit(inp.value.trim()); });
    form = el("div", {}, inp, " ", el("button", { class: "primary", onclick: () => { if (inp.value.trim()) commit(inp.value.trim()); } }, "commit"));
  }
  box.append(form, reveal);
  if (prog[name]) { form.querySelectorAll("button,input").forEach(x => x.disabled = true); showReveal(); }
  return box;
}
function renderKP(main, tid, kp, kps) {
  const d = state.data[tid], key = `${tid}:${kp.id}`; state.progress[key] ||= {}; const prog = state.progress[key];
  const t = state.threads.threads.find(x => x.id === tid);
  const prov = kp.provenance || {};
  if (state.lastOpen !== `${tid}:${kp.id}`) { state.lastOpen = `${tid}:${kp.id}`; logEvent("open", { thread: tid, kp: kp.id }); }
  main.append(el("div", { class: "kp-head" },
    el("div", { class: "kind" }, kp.kind || "", " · ", el("span", { class: `chip ${prov.status} ${prov.basis === "inference" ? "inference" : ""}` }, prov.status), (kp.prerequisites || []).length ? el("span", { class: "small muted" }, ` needs: ${kp.prerequisites.map(p => p.kp).join(", ")}`) : null),
    el("h1", {}, kp.label)));
  if (!prereqsMet(tid, kp, kps) && setting("strict", false)) {
    const missing = [...(kp.prerequisites || []).filter(p => kps.find(k => k.id === p.kp) && masteryOf(tid, p.kp) !== "mastered").map(p => [tid, p.kp]),
      ...(kp.external_prerequisites || []).map(e => String(e.kp || e).split(":")).filter(([t2, k2]) => state.data[t2] && masteryOf(t2, k2) !== "mastered")];
    main.append(el("div", { class: "block gate" }, el("div", { class: "eyebrow" }, "locked · strict mode"), el("div", {}, "Master these first (answer their changed-case question):"),
      el("ul", {}, ...missing.map(([t2, k2]) => el("li", {}, el("a", { href: `#/t/${t2}/${k2}` }, `${t2 === tid ? "" : t2 + " · "}${(state.data[t2]?.kp.knowledge_points.find(k => k.id === k2) || {}).label || k2}`)))),
      el("div", { class: "small muted" }, "The one-sentence claim stays hidden until then. Turn strict mode off on the data page to browse freely.")));
    logEvent("blocked", { thread: tid, kp: kp.id, missing: missing.map(m => m.join(":")) });
    return;
  }
  if (!prereqsMet(tid, kp, kps)) {
    const missing = [...(kp.prerequisites || []).filter(p => kps.find(k => k.id === p.kp) && masteryOf(tid, p.kp) !== "mastered").map(p => [tid, p.kp]),
      ...(kp.external_prerequisites || []).map(e => String(e.kp || e).split(":")).filter(([t2, k2]) => state.data[t2] && masteryOf(t2, k2) !== "mastered")];
    main.append(el("div", { class: "warn" }, "Prerequisites not yet mastered: ", ...missing.map(([t2, k2]) => el("a", { href: `#/t/${t2}/${k2}`, style: "margin-right:8px" }, `${t2 === tid ? "" : t2 + " · "}${k2}`)), el("span", { class: "muted" }, " — you can continue; the gate is advisory in this demo.")));
  }

  // 1. predict gate
  const rest = el("div", { class: prog.predict ? "" : "locked-rest" });
  const unlock = () => { rest.querySelectorAll(".block.locked").forEach(b => b.classList.remove("locked")); statement.style.display = ""; };
  const predict = promptBlock(tid, kp, "predict", kp.prompts.predict, { eyebrow: "1 · predict before anything is revealed", onCommit: () => { unlock(); setTimeout(route, 0); } });
  main.append(predict);
  const statement = el("div", { class: "block", style: prog.predict ? "" : "display:none" }, el("div", { class: "eyebrow" }, "the claim"), el("div", { class: "statement" }, kp.statement));
  main.append(statement);

  // 2. mechanism: presentations
  const locked = prog.predict ? "" : "locked";
  const mech = el("div", { class: `block ${locked}` }, el("div", { class: "eyebrow" }, "2 · mechanism"));
  const interactiveRefs = new Set((kp.presentations || []).filter(p => p.type === "interactive").map(p => p.ref));
  for (const p of kp.presentations || []) mech.append(renderPresentation(tid, kp, p, d, interactiveRefs));
  if (kp.prompts.self_explain) {
    const ta = el("textarea", { rows: 2, style: "width:100%;max-width:640px;font:inherit;padding:6px", placeholder: "two sentences", onchange: e => { prog.self_explain = e.target.value; saveProgress(); logEvent("self_explain", { thread: tid, kp: kp.id, text: e.target.value }); } });
    if (prog.self_explain) ta.value = prog.self_explain;
    mech.append(el("div", { class: "pres" }, el("div", { class: "label" }, "self-explain: ", kp.prompts.self_explain), ta));
  }
  if (kp.misconceptions?.length) mech.append(el("div", { class: "pres" }, el("div", { class: "label" }, "misconceptions this KP targets"), el("ul", { class: "small" }, ...kp.misconceptions.map(m => el("li", {}, el("b", {}, m.belief), " — ", m.why_wrong)))));
  rest.append(mech);

  // 3. changed case, closed book: the mechanism (widgets can compute any case) is hidden from the
  // moment the learner starts the changed case until both transfer and check are answered.
  const answered = n => !kp.prompts[n] || prog[`${n}_given`] !== undefined;
  const caseDone = answered("transfer") && answered("check");
  const closedBook = !caseDone && !!prog.closed_book;
  const caseLocked = !prog.predict || (!caseDone && !prog.closed_book);
  if (closedBook) { mech.style.display = "none"; rest.append(el("div", { class: "block notice" }, "Mechanism hidden: answer the changed case from what you understood. It comes back once you have answered.")); }
  else if (prog.predict && !caseDone) rest.append(el("div", { class: "block" }, el("button", { onclick: () => { prog.closed_book = Date.now(); saveProgress(); logEvent("closed_book", { thread: tid, kp: kp.id }); route(); } }, "Start the changed case"), el("span", { class: "muted small", style: "margin-left:8px" }, "the mechanism above is hidden while you answer")));
  rest.append(promptBlock(tid, kp, "transfer", kp.prompts.transfer, { eyebrow: kp.prompts.check ? "3a · changed case (you grade it)" : "3 · changed case (counts for mastery)", locked: caseLocked, onGrade: () => route() }));
  if (kp.prompts.check) rest.append(promptBlock(tid, kp, "check", kp.prompts.check, { eyebrow: "3b · quick check (graded automatically; mastery needs 3a and 3b)", locked: caseLocked, onGrade: () => route() }));

  // 4. retrieval
  const ret = el("div", { class: `block retrieval ${locked}` }, el("div", { class: "eyebrow" }, "4 · retrieval prompts (scheduled; come back in reviews)"));
  // layers: core cards first; detail cards (identifiers, paths, exact thresholds) wait behind a fold
  const rs = (kp.prompts.retrieval || []).map((r, i) => [r, i]);
  rs.filter(([r]) => r.layer !== "detail").forEach(([r, i]) => ret.append(retrievalCard(`${tid}:${kp.id}:r${i}`, r)));
  const det = rs.filter(([r]) => r.layer === "detail");
  if (det.length) {
    const box = el("details", { class: "detail-layer" }, el("summary", { class: "small muted" }, `more detail (optional, ${det.length} card${det.length > 1 ? "s" : ""}; scheduled in reviews only after the core cards are remembered twice)`));
    det.forEach(([r, i]) => box.append(retrievalCard(`${tid}:${kp.id}:r${i}`, r)));
    ret.append(box);
  }
  if (closedBook) ret.style.display = "none";  // its answers can answer the changed case
  rest.append(ret);
  main.append(rest);

  // provenance
  // the teacher line quotes the source, which states the answer: hidden before the prediction and while the changed case is open
  if (!prog.predict) main.append(el("div", { class: "block prov muted small" }, "Provenance (teacher line: sources, quotes, what the professor filtered) opens after you commit a prediction."));
  else if (closedBook) main.append(el("div", { class: "block prov muted small" }, "Provenance is hidden while you answer the changed case."));
  else main.append(renderProvenance(kp, d.sources, t));

  // next up: outer fringe
  const stops = d.kp.path?.stops || kps.map(k => k.id);
  const next = stops.filter(id => id !== kp.id && masteryOf(tid, id) !== "mastered" && prereqsMet(tid, kps.find(k => k.id === id), kps));
  const sess = Object.entries(SESSIONS).find(([, sd]) => sd.thread === tid && (sd.kps || []).includes(kp.id));
  if (sess) { const ids = sess[1].kps; const i = ids.indexOf(kp.id); main.append(el("div", { class: "footer-nav" }, el("span", { class: "muted small" }, `session ${sess[0]} · ${i + 1}/${ids.length}`), i + 1 < ids.length ? el("a", { href: `#/t/${tid}/${ids[i + 1]}` }, "next in session →") : el("a", { href: "#/data" }, "session done → export data"))); }
  main.append(el("div", { class: "footer-nav" }, el("span", { class: "muted small" }, "next on the outer fringe:"), ...next.slice(0, 3).map(id => el("a", { href: `#/t/${tid}/${id}` }, kps.find(k => k.id === id).label))));
}

function retrievalCard(id, r, revealAlso = null) {
  const c = state.cards[id];
  const card = el("div", { class: "card" });
  const meta = el("div", { class: "meta" }, c ? `R=${retrievability(c).toFixed(2)} · S=${c.S.toFixed(1)}d · due ${new Date(c.due).toLocaleDateString()}` : "not yet reviewed");
  const ans = el("div", { class: "ans" }, r.answer, revealAlso ? el("div", { class: "muted small", style: "margin-top:4px" }, revealAlso) : null);
  const grade = (G) => { state.cards[id] = review(c, G); saveCards(state.cards); logEvent("retrieval", { card: id, grade: G, S: +state.cards[id].S.toFixed(2) }); meta.textContent = `next in ${nextInterval(state.cards[id].S)} d`; updateDue(); btns.querySelectorAll("button").forEach(b => b.disabled = true); };
  const btns = el("div", { style: "display:none;margin-top:6px" }, el("button", { onclick: () => grade(1) }, "didn't remember"), " ", el("button", { class: "primary", onclick: () => grade(3) }, "remembered"));
  card.append(el("div", {}, r.q), el("button", { class: "ghost small", style: "margin-top:6px", onclick: () => { card.classList.add("open"); btns.style.display = ""; } }, "reveal"), ans, btns, meta);
  return card;
}

function renderPresentation(tid, kp, p, d, interactiveRefs = new Set()) {
  const box = el("div", { class: "pres" });
  const lab = el("div", { class: "label" }, `${p.type} · ${p.action}${p.faded ? " · faded" : ""}`);
  box.append(lab);
  if (p.type === "animation") {
    const fid = (p.ref || "").replace("fixture:", "");
    const holder = el("div", {}); box.append(holder);
    loadBuild(fid).then(m => {
      if (!m) {
        const w = interactiveRefs.has(p.ref) ? null : WIDGETS[p.ref];
        holder.append(el("div", { class: "muted small" }, `Manim render for '${fid}' not built yet (atlas/engine/render.py). ${w ? "Interactive stand-in below drives the same fixture numbers." : interactiveRefs.has(p.ref) ? "The interactive below drives the same fixture numbers." : ""}`));
        if (w) { const h2 = el("div", {}); holder.append(h2); w(h2, p.notice); }
        if (p.notice) holder.append(el("div", { class: "notice" }, el("b", {}, "notice: "), p.notice));
        return;
      }
      const vid = el("video", { controls: "", preload: "metadata", src: `${state.threads.builds}${fid}/video.mp4` });
      const strip = el("div", { class: "steps" });
      m.steps.forEach(s => strip.append(el("div", { class: "step", onclick: () => { vid.currentTime = Math.max(0, s.time - 0.05); vid.pause(); logEvent("seek", { thread: tid, kp: kp.id, fixture: fid, step: s.id }); strip.querySelectorAll(".step").forEach(x => x.classList.remove("active")); event.currentTarget.classList.add("active"); } },
        el("img", { src: `${state.threads.builds}${fid}/${s.frame}`, alt: s.caption }), el("div", {}, s.caption))));
      holder.append(vid, el("div", { class: "small muted" }, `lint: ${m.lint_worst} · ${m.quality} · steps are seek points, not autoplay`), strip);
      if (p.notice) holder.append(el("div", { class: "notice" }, el("b", {}, "notice: "), p.notice));
      m.notices?.forEach(n => holder.append(el("div", { class: "notice small" }, el("b", {}, `${n.id} @ ${n.at_step}: `), n.claim)));
    });
  } else if (p.type === "interactive") {
    const w = WIDGETS[p.ref];
    if (w) { const holder = el("div", {}); box.append(holder); w(holder, p.notice); }
    else box.append(el("div", { class: "muted small" }, `no widget registered for ${p.ref}`));
    if (p.notice) box.append(el("div", { class: "notice" }, el("b", {}, "notice: "), p.notice));
  } else if (p.type === "code" || p.type === "formula-step" || p.type === "example") {
    if (Array.isArray(p.steps) && p.steps.length) box.append(workedSteps(tid, kp, p));
    const sn = d.snippets?.[p.ref];
    if (sn) box.append(el("pre", {}, numbered(sn)));
    else if (d.discovered?.sections) {
      const eqs = d.discovered.sections.flatMap(sec => sec.equations || []);
      const hits = eqs.filter(e => e.latex && (p.ref === e.ref || (p.ref || "").split(/[,\s]+/).includes(e.ref)));
      if (hits.length) box.append(el("div", { class: "latex-block" }, ...hits.map(e => { const d = el("div", { class: "latex-row" }, el("span", { class: "muted small", style: "margin-right:10px" }, `(${e.number})`)); const t = el("span", { class: "latex" }); t.textContent = e.latex; d.append(t); return d; })));
    }
    else { const ev = findEvidence(d, p.ref); if (ev.length) box.append(el("pre", {}, ev.map(e => e.text).join("\n"))); }
    box.append(el("div", { class: "small" }, "source: ", refLink(d.sources, kp, p.ref), sn ? el("span", { class: "muted" }, ` · ${sn.path.split("/").slice(-2).join("/")} @${sn.sha256}`) : null));
  } else if (p.type === "video-span") {
    const vm = VIDEO_REF.exec(p.ref || "");
    const holder = el("div", {}); box.append(holder);
    loadTranscript(tid).then(tr => {
      if (!vm) { holder.append(el("div", { class: "muted small" }, `spoken excerpt ${p.ref}`)); return; }
      if (!tr) {
        // no transcript in this build (published copies leave captions out): link the moment instead
        const url = videoUrl(d), a = videoSecs(vm[1]);
        holder.append(el("div", { class: "small" }, url ? el("a", { href: `${url}&t=${a}s`, target: "_blank" }, `watch ${vm[1]}${vm[2] ? "–" + vm[2] : ""} in the lecture video`) : `lecture video ${vm[1]}${vm[2] ? "–" + vm[2] : ""}`));
        if (p.notice) holder.append(el("div", { class: "notice" }, el("b", {}, "listen for: "), String(p.notice).replace(/^listen for:\s*/i, "")));
        return;
      }
      const a = videoSecs(vm[1]), b = vm[2] ? videoSecs(vm[2]) : a + 30;
      const segs = tr.segments.filter(s2 => s2.t1 >= a - 1 && s2.t0 <= b + 1);
      holder.append(el("blockquote", { class: "spoken" }, ...segs.map(s2 => el("div", {}, el("span", { class: "muted small", style: "margin-right:6px" }, s2.at), s2.text))),
        el("div", { class: "small" }, el("a", { href: `${tr.url}&t=${a}s`, target: "_blank" }, `watch ${vm[1]}${vm[2] ? "–" + vm[2] : ""} on YouTube`), el("span", { class: "muted" }, " · human-made captions")));
      if (p.notice) holder.append(el("div", { class: "notice" }, el("b", {}, "listen for: "), String(p.notice).replace(/^listen for:\s*/i, "")));
    });
  } else if (p.type === "figure") {
    if (/\.(png|jpe?g|gif|svg|webp)$/i.test(p.ref || "")) {
      // published builds point lecture figures at the upstream repo instead of copying them (threads.json figure_base)
      const fb = state.threads.figure_base, src = p.ref.startsWith("http") ? p.ref : fb && p.ref.startsWith("official/lectures/") ? fb + p.ref.slice("official/lectures/".length) : `../../${p.ref}`;
      box.append(el("img", { src, style: "max-width:100%;border:1px solid var(--rule);border-radius:6px" }), el("div", { class: "small muted" }, p.ref));
    } else {
      // a source anchor (e.g. "chinchilla:table3", "lecture_08.pdf:p26"): link to it, quote it if a KP anchor carries the quote
      const a = (kp.provenance?.anchors || []).find(x => x.ref === p.ref);
      box.append(el("div", { class: "small" }, "figure in the source: ", refLink(d.sources, kp, p.ref)));
      if (a?.quote) box.append(el("div", { class: "quote small" }, `“${a.quote}”`));
    }
    if (p.notice) box.append(el("div", { class: "notice" }, el("b", {}, "look for: "), p.notice));
  } else {
    box.append(el("div", { class: "small" }, p.ref));
  }
  return box;
}
// Worked example: every step with its reason. Faded (PHILOSOPHY 3, second instance): the last step is
// hidden until the learner writes their own version of it.
function workedSteps(tid, kp, p) {
  const ol = el("ol", { class: "worked" });
  const stepEl = s => el("li", {}, s.tex ? (() => { const t = el("span", { class: "latex" }); t.textContent = s.tex; return t; })() : null,
    s.do ? el("div", {}, s.do) : null, s.why ? el("div", { class: "why muted small" }, "why: ", s.why) : null);
  const shown = p.faded ? p.steps.slice(0, -1) : p.steps;
  shown.forEach(s => ol.append(stepEl(s)));
  if (p.faded) {
    const last = p.steps[p.steps.length - 1];
    const ta = el("textarea", { rows: 2, placeholder: "write the last step yourself, then reveal", style: "width:100%;max-width:640px;font:inherit;padding:6px" });
    const li = el("li", { class: "faded-step" }, ta);
    const btn = el("button", { class: "ghost small", onclick: () => { logEvent("faded_step", { thread: tid, kp: kp.id, ref: p.ref, text: ta.value }); li.replaceWith(stepEl(last)); renderMath(ol); btn.remove(); } }, "reveal the last step");
    ol.append(li); setTimeout(() => renderMath(ol), 0);
    return el("div", {}, ol, btn);
  }
  setTimeout(() => renderMath(ol), 0);
  return ol;
}

function renderMath(root = document) {
  if (!window.katex) { setTimeout(() => renderMath(root), 300); return; }
  root.querySelectorAll(".latex:not([data-done])").forEach(n => { try { window.katex.render(n.textContent, n, { throwOnError: false, displayMode: true }); } catch {} n.dataset.done = "1"; });
}
function numbered(sn) {
  const lines = sn.text.split("\n"); const w = String(sn.end).length;
  return lines.map((l, i) => `${String(sn.start + i).padStart(w)}  ${l}`).join("\n");
}
function refLink(sources, kp, ref) {
  const a = (kp.provenance?.anchors || []).find(x => x.ref === ref) || { source: Object.keys(sources)[0], ref };
  const href = anchorHref(sources, a);
  return href ? el("a", { href, target: "_blank" }, ref) : el("code", {}, ref);
}
function findEvidence(d, ref) {
  const m = /^([\w.\-/]+):L(\d+)(?:-L(\d+))?/.exec(ref || ""); if (!m || !d.discovered?.items) return [];
  const lo = +m[2], hi = m[3] ? +m[3] : lo;
  return d.discovered.items.filter(it => (it.kind === "code" || it.kind === "claim" || it.kind === "value") && it.line >= lo && (it.extra?.end_line || it.line) <= hi + 0).map(it => ({ text: it.kind === "claim" ? `# ${it.text}` : it.kind === "value" ? `# @inspect ${it.text}` : it.text }));
}
function videoUrl(d) { const k = Object.keys(d.sources || {}).find(x => /_video$/.test(x)); return k ? d.sources[k].url : null; }
async function loadTranscript(tid) {
  const d = state.data[tid]; if (!d) return null;
  if (d.transcript !== undefined) return d.transcript;
  const t = state.threads.threads.find(x => x.id === tid);
  d.transcript = await tryJSON(t.kp.replace(/knowledge_points\.json$/, "transcript.json"));
  return d.transcript;
}
async function loadBuild(fid) {
  if (state.builds[fid] !== undefined) return state.builds[fid];
  state.builds[fid] = await tryJSON(`${state.threads.builds}${fid}/manifest.json`);
  return state.builds[fid];
}

function renderProvenance(kp, sources, t) {
  const prov = kp.provenance || {};
  const box = el("div", { class: "block prov" }, el("div", { class: "eyebrow" }, "provenance (teacher line)"));
  box.append(el("div", {}, el("span", { class: `chip ${prov.status} ${prov.basis === "inference" ? "inference" : ""}` }, prov.status), el("span", { class: "chip" }, prov.basis || "evidence")));
  for (const a of prov.anchors || []) {
    const href = anchorHref(sources, a);
    box.append(el("span", { class: "anchor" }, el("span", { class: "chip" }, a.role || "ref"), href ? el("a", { href, target: "_blank" }, anchorText(a)) : el("code", {}, anchorText(a)), a.quote ? el("span", { class: "quote" }, ` “${a.quote}”`) : null));
  }
  if (prov.filtering_note) box.append(el("div", { class: "fnote" }, el("b", {}, "what the source filtered: "), prov.filtering_note));
  const st = spokenStretches(state.data[t.id] || { kp: {} }, kp.id), url = state.data[t.id]?.sources?.[Object.keys(state.data[t.id]?.sources || {}).find(k => /_video$/.test(k))]?.url;
  if (st.length && url) box.append(el("div", { class: "small" }, el("b", {}, "taught in the video at: "), ...st.map(s => el("a", { href: `${url}&t=${videoSecs(s.from)}s`, target: "_blank", style: "margin-right:8px" }, `${s.from}–${s.to}`))));
  if (kp.prerequisites?.length) box.append(el("div", { class: "prereq" }, el("b", {}, "prerequisites"), el("ul", {}, ...kp.prerequisites.map(p => el("li", {}, el("a", { href: `#/t/${t.id}/${p.kp}` }, p.kp), " ", el("span", { class: `chip ${p.basis === "teaching-order" || p.basis === "call-order" ? "" : "inference"}` }, p.basis), p.note ? el("span", { class: "muted small" }, " ", p.note) : null)))));
  if (kp.external_prerequisites?.length) box.append(el("div", { class: "prereq" }, el("b", {}, "prerequisites in other threads"), el("ul", {}, ...kp.external_prerequisites.map(e => { const [t2, k2] = String(e.kp || e).split(":"); const known = state.data[t2]?.kp.knowledge_points.find(k => k.id === k2); return el("li", {}, known ? el("a", { href: `#/t/${t2}/${k2}` }, `${t2} · ${known.label}`) : el("code", {}, e.kp || e), " ", el("span", { class: `chip ${masteryOf(t2, k2)}` }, masteryOf(t2, k2)), e.note ? el("span", { class: "muted small" }, " ", e.note) : null); }))));
  return box;
}

// ------------------------------------------------------- filtering view --
function renderFilter(main, tid) {
  const d = state.data[tid], disc = d.discovered;
  main.append(el("h1", {}, "Filtering view"), el("p", { class: "muted small" }, "Left: everything the discover tool found in the source (deterministic, no LLM). Right: which of it became a knowledge point and with what status. The gap is the professor's filtering, made visible."));
  if (!disc) return main.append(el("div", { class: "warn" }, "no discovered.json for this thread"));
  const kps = d.kp.knowledge_points;
  if (disc.items) { // executable lecture format
    const sections = disc.summary.sections_in_order;
    const bySec = {}; for (const it of disc.items) (bySec[it.section] ||= []).push(it);
    const lineOfKP = k => (k.provenance?.anchors || []).map(a => /:L(\d+)/.exec(a.ref)).filter(Boolean).map(m => +m[1]);
    const kpsInSec = sec => { const its = bySec[sec] || []; const lo = Math.min(...its.map(i => i.line)), hi = Math.max(...its.map(i => i.extra?.end_line || i.line)); return kps.filter(k => lineOfKP(k).some(l => l >= lo && l <= hi)); };
    const tbl = el("table", {}, el("tr", {}, el("th", {}, "section (teaching order)"), el("th", {}, "claims"), el("th", {}, "values"), el("th", {}, "cites"), el("th", {}, "figures"), el("th", {}, "→ knowledge points")));
    for (const s of sections) {
      const its = bySec[s] || []; const c = k => its.filter(i => i.kind === k).length;
      const ks = kpsInSec(s);
      tbl.append(el("tr", {}, el("td", {}, el("code", {}, s)), el("td", {}, bar(c("claim"), 40), c("claim")), el("td", {}, bar(c("value"), 20, "k"), c("value")), el("td", {}, c("cite")), el("td", {}, c("figure")),
        el("td", {}, ks.length ? ks.map(k => el("a", { href: `#/t/${tid}/${k.id}`, style: "margin-right:8px" }, el("span", { class: `chip ${k.provenance.status}` }, k.provenance.status), k.label)) : el("span", { class: "muted" }, "— not a KP (announcements, transitions, or dropped)"))));
    }
    main.append(el("div", { class: "filter-view" }, tbl));
    main.append(el("h2", {}, "All evidence items"), el("div", { class: "evidence-list" }, ...disc.items.filter(i => i.kind !== "section_end").map(i => el("div", { class: "it" }, el("span", { class: "k" }, `L${i.line} ${i.kind}`), i.kind === "section_start" ? el("b", {}, i.section) : (i.text || "").slice(0, 160)))));
  } else if (disc.pages) { // slides format
    const tbl = el("table", {}, el("tr", {}, el("th", {}, "page"), el("th", {}, "title"), el("th", {}, "claims"), el("th", {}, "→ knowledge points")));
    for (const pg of disc.pages) {
      const ks = kps.filter(k => (k.provenance?.anchors || []).some(a => new RegExp(`:p${pg.page}(\\b|$)`).test(a.ref)));
      tbl.append(el("tr", {}, el("td", {}, pg.page), el("td", {}, pg.title), el("td", {}, (pg.claims || []).length), el("td", {}, ks.length ? ks.map(k => el("a", { href: `#/t/${tid}/${k.id}`, style: "margin-right:8px" }, k.label)) : el("span", { class: "muted" }, "—"))));
    }
    main.append(el("div", { class: "filter-view" }, tbl));
  } else if (disc.sections && disc.sections[0]?.equations) { // paper format
    const eqSec = {}; disc.sections.forEach(sec => (sec.equations || []).forEach(e => eqSec[e.ref] = sec.id));
    const secOf = ref => { if (eqSec[ref]) return eqSec[ref]; const m = /^rope:(§[\d.]+)/.exec(ref); if (m) { const hit = disc.sections.find(s2 => s2.ref === "rope:" + m[1]) || disc.sections.find(s2 => m[1].startsWith(s2.ref.replace("rope:", "") + ".")); if (hit) return hit.id; } const pm = /^rope:p(\d+)/.exec(ref); if (pm) { const hit = disc.sections.find(s2 => s2.page === +pm[1]); if (hit) return hit.id; } return null; };
    const citedEq = new Set(kps.flatMap(k => (k.provenance.anchors || []).map(a => a.ref)).filter(r => eqSec[r]));
    const tbl = el("table", {}, el("tr", {}, el("th", {}, "section (paper order)"), el("th", {}, "claims"), el("th", {}, "equations (cited by a KP)"), el("th", {}, "course restates"), el("th", {}, "→ knowledge points")));
    for (const sec of disc.sections) {
      const ks = kps.filter(k => (k.provenance.anchors || []).some(a => secOf(a.ref) === sec.id));
      const eqs = sec.equations || []; const cited = eqs.filter(e => citedEq.has(e.ref)).length;
      tbl.append(el("tr", {}, el("td", {}, el("b", {}, sec.ref.replace("rope:", "")), " ", sec.title, el("div", { class: "muted small" }, `p${sec.page}`)), el("td", {}, bar((sec.claims || []).length, 8), (sec.claims || []).length),
        el("td", {}, eqs.length ? `${eqs.length} (${cited})` : "—", eqs.length ? el("div", { class: "muted small" }, eqs.map(e => e.number).join(", ")) : null),
        el("td", {}, (sec.course_restates || []).length ? (sec.course_restates || []).map(c => el("div", { class: "small" }, c.ref || c.page || JSON.stringify(c).slice(0, 40))) : el("span", { class: "muted" }, "not taught")),
        el("td", {}, ks.length ? ks.map(k => el("a", { href: `#/t/${tid}/${k.id}`, style: "margin-right:8px" }, el("span", { class: `chip ${k.provenance.status}` }, k.provenance.status), k.label)) : el("span", { class: "muted" }, "— dropped"))));
    }
    main.append(el("div", { class: "filter-view" }, tbl));
    if (disc.course?.order) main.append(el("h2", {}, "Course order (Lecture 3)"), el("ol", { class: "small" }, ...disc.course.order.map(o => el("li", {}, el("code", {}, o.ref), " ", o.step))));
  } else if (disc.modules && disc.entry_trace) { // repo format
    const hops = disc.entry_trace.hops || [];
    const hopCount = {}; hops.forEach(h => { const m = h.callee.split(":")[0]; hopCount[m] = (hopCount[m] || 0) + 1; });
    const firstHop = {}; hops.forEach(h => { const m = h.callee.split(":")[0]; if (firstHop[m] === undefined) firstHop[m] = h.order; });
    const kpsOf = mod => { const path = disc.modules.find(m => m.name === mod)?.path || ""; return kps.filter(k => (k.provenance.anchors || []).some(a => a.ref.startsWith(path.replace(/\/__init__\.py$/, "")) && path)); };
    const mods = disc.modules.filter(m => m.lines > 0).sort((a, b) => (firstHop[a.name] ?? 1e9) - (firstHop[b.name] ?? 1e9) || b.lines - a.lines);
    main.append(el("p", { class: "small" }, `${disc.summary.n_modules} modules, ${disc.summary.n_lines} lines, ${hops.length} call hops from ${disc.entry_trace.root}, ${disc.summary.n_unresolved_calls} unresolved calls. Modules are listed in the order the trace first reaches them; modules the trace never reaches are the repo's off-path surface.`));
    const tbl = el("table", {}, el("tr", {}, el("th", {}, "module (trace order)"), el("th", {}, "lines"), el("th", {}, "defs"), el("th", {}, "hops"), el("th", {}, "→ knowledge points")));
    for (const m of mods) {
      const ks = kpsOf(m.name); const onPath = hopCount[m.name] > 0;
      tbl.append(el("tr", { style: onPath ? "" : "opacity:.55" }, el("td", {}, el("code", {}, m.name), el("div", { class: "muted small" }, m.path)), el("td", {}, m.lines), el("td", {}, (m.defs || []).filter(d => d.kind !== "method").length), el("td", {}, onPath ? [bar(hopCount[m.name], 40, "k"), hopCount[m.name]] : el("span", { class: "muted" }, "off path")),
        el("td", {}, ks.length ? ks.map(k => el("a", { href: `#/t/${tid}/${k.id}`, style: "margin-right:8px" }, el("span", { class: `chip ${k.provenance.status}` }, k.provenance.status), k.label)) : el("span", { class: "muted" }, onPath ? "— on path, not a KP" : "—"))));
    }
    main.append(el("div", { class: "filter-view" }, tbl));
    main.append(el("h2", {}, "Entry trace (first 40 hops)"), el("div", { class: "evidence-list" }, ...hops.slice(0, 40).map(h => el("div", { class: "it" }, el("span", { class: "k" }, `${String(h.order).padStart(3)} d${h.depth}`), `${h.caller} → `, el("b", {}, h.callee), el("span", { class: "muted small" }, ` @L${h.call_line} [${h.resolution}]`)))));
  } else {
    main.append(el("pre", {}, JSON.stringify(disc.summary || Object.keys(disc), null, 2)));
  }
  const st = {}; for (const k of kps) st[k.provenance.status] = (st[k.provenance.status] || 0) + 1;
  main.append(el("h2", {}, "Status of knowledge points"), el("div", {}, ...Object.entries(st).map(([s, n]) => el("span", { class: `chip ${s}`, style: "margin:2px" }, `${s}: ${n}`))));
}
const bar = (n, max, cls = "") => el("span", { class: `bar ${cls}`, style: `width:${Math.min(120, 120 * n / max)}px;margin-right:6px` });

async function renderSpoken(main, tid) {
  const d = state.data[tid], tr = await loadTranscript(tid);
  main.append(el("h1", {}, "Spoken layer"), el("p", { class: "muted small" }, "What the professor said, from human-made captions. Rows are sentences the transcript tool flagged as teaching moves (deterministic keyword cues, not judgements). A row is linked when a knowledge point anchors to that moment; unlinked skips and deferrals are filtering decisions the written material does not show."));
  if (!tr) {
    // without captions, the hand-made layers (video index, oral filtering) still stand
    const url = videoUrl(d); if (!url) { main.append(el("div", { class: "warn" }, "No lecture video for this thread.")); return; }
    const t0 = { url, segments: [] };
    if (Array.isArray(d.kp.spoken_map) && d.kp.spoken_map.length) main.append(renderSpokenMap(d, tid, t0));
    if (Array.isArray(d.kp.spoken_filtering) && d.kp.spoken_filtering.length) {
      const tbl = el("table", {}, el("tr", {}, el("th", {}, "at"), el("th", {}, "kind"), el("th", {}, "what was filtered"), el("th", {}, "said"), el("th", {}, "knowledge point")));
      for (const f of d.kp.spoken_filtering) tbl.append(el("tr", {}, el("td", {}, el("a", { href: `${url}&t=${videoSecs(f.at)}s`, target: "_blank" }, f.at)), el("td", {}, el("span", { class: "chip" }, f.kind)), el("td", {}, f.what_was_filtered || ""), el("td", { class: "muted small" }, `“${f.quote || ""}”`), el("td", {}, f.kp ? el("a", { href: `#/t/${tid}/${f.kp}` }, f.kp) : el("span", { class: "muted" }, "—"))));
      main.append(el("h2", {}, "The professor's oral filtering"), el("div", { class: "filter-view" }, tbl));
    }
    return;
  }
  const cited = [];
  for (const k of d.kp.knowledge_points) for (const a of (k.provenance.anchors || [])) { const vm = VIDEO_REF.exec(a.ref || ""); if (vm) cited.push({ a: videoSecs(vm[1]), b: vm[2] ? videoSecs(vm[2]) : videoSecs(vm[1]) + 30, k }); }
  if (Array.isArray(d.kp.spoken_map) && d.kp.spoken_map.length) main.append(renderSpokenMap(d, tid, tr));
  if (Array.isArray(d.kp.spoken_filtering) && d.kp.spoken_filtering.length) {
    const tbl = el("table", {}, el("tr", {}, el("th", {}, "at"), el("th", {}, "kind"), el("th", {}, "what was filtered"), el("th", {}, "said"), el("th", {}, "knowledge point")));
    for (const f of d.kp.spoken_filtering) {
      tbl.append(el("tr", {}, el("td", {}, el("a", { href: `${tr.url}&t=${videoSecs(f.at)}s`, target: "_blank" }, f.at)), el("td", {}, el("span", { class: "chip" }, f.kind)), el("td", {}, f.what_was_filtered || ""),
        el("td", { class: "muted small" }, `“${f.quote || ""}”`), el("td", {}, f.kp ? el("a", { href: `#/t/${tid}/${f.kp}` }, f.kp) : el("span", { class: "muted" }, "—"))));
    }
    main.append(el("h2", {}, "The professor's oral filtering"), el("p", { class: "muted small" }, "Explicit skips, deferrals to assignments and asides, reviewed by hand from the captions."), el("div", { class: "filter-view" }, tbl), el("h2", {}, "All flagged moments"));
  }
  const show = ["skip", "defer-assignment", "warning", "intuition", "emphasis", "aside", "callback", "forward"];
  const sel = el("select", {}, el("option", { value: "" }, "all moves"), ...show.map(m => el("option", { value: m }, m)));
  const counts = el("div", { class: "small", style: "margin:6px 0" }, `${tr.n_segments} sentences · ${Math.round(tr.duration_s / 60)} min · `, ...Object.entries(tr.move_counts || {}).filter(([m]) => m !== "question").map(([m, n]) => el("span", { class: "chip" }, `${m} ${n}`)));
  const list = el("div", { class: "evidence-list", style: "max-height:none" });
  const draw = () => {
    list.innerHTML = ""; const want = sel.value;
    let linked = 0, total = 0;
    for (const sgm of tr.segments) {
      const mv = sgm.moves.filter(m => show.includes(m)); if (!mv.length || (want && !mv.includes(want))) continue;
      total++;
      const hit = cited.filter(c => sgm.t1 >= c.a - 8 && sgm.t0 <= c.b + 8);
      if (hit.length) linked++;
      list.append(el("div", { class: "it" }, el("a", { href: `${tr.url}&t=${Math.floor(sgm.t0)}s`, target: "_blank", class: "k" }, sgm.at), ...mv.map(m => el("span", { class: "chip" }, m)), " ", sgm.text,
        hit.length ? el("div", { class: "small" }, "→ ", ...hit.map(c => el("a", { href: `#/t/${tid}/${c.k.id}`, style: "margin-right:8px" }, c.k.label))) : el("div", { class: "small muted" }, "not anchored by any knowledge point")));
    }
    summary.textContent = `${total} flagged moments · ${linked} anchored by a knowledge point · ${total - linked} not`;
  };
  const summary = el("div", { class: "small", style: "margin:6px 0;font-weight:600" });
  sel.addEventListener("change", draw);
  main.append(counts, el("div", {}, "show ", sel), summary, list);
  draw();
}

// video index: the hand segmentation of the talk (spoken_map), one band per stretch
function renderSpokenMap(d, tid, tr) {
  const segs = d.kp.spoken_map, byId = Object.fromEntries(d.kp.knowledge_points.map(k => [k.id, k]));
  const t0 = videoSecs(segs[0].from), t1 = videoSecs(segs[segs.length - 1].to), span = Math.max(1, t1 - t0);
  const mapped = segs.filter(s => s.kp).reduce((n, s) => n + videoSecs(s.to) - videoSecs(s.from), 0);
  const spokenOnly = segs.filter(s => byId[s.kp]?.provenance.status === "spoken").reduce((n, s) => n + videoSecs(s.to) - videoSecs(s.from), 0);
  const color = s => !s.kp ? "#d9dde1" : byId[s.kp]?.provenance.status === "spoken" ? "#b8582a" : "#24668d";
  const band = el("div", { class: "vmap" });
  for (const s of segs) {
    const a = videoSecs(s.from), b = videoSecs(s.to);
    const label = s.kp ? (byId[s.kp]?.label || s.kp) : s.what;
    const cell = el("a", { href: s.kp ? `#/t/${tid}/${s.kp}` : `${tr.url}&t=${a}s`, target: s.kp ? "" : "_blank", title: `${s.from}–${s.to} ${label}`, style: `left:${100 * (a - t0) / span}%;width:${100 * (b - a) / span}%;background:${color(s)}` });
    band.append(cell);
  }
  const tbl = el("table", {}, el("tr", {}, el("th", {}, "stretch"), el("th", {}, "min"), el("th", {}, "taught")));
  for (const s of segs) {
    const a = videoSecs(s.from), b = videoSecs(s.to);
    tbl.append(el("tr", {}, el("td", {}, el("a", { href: `${tr.url}&t=${a}s`, target: "_blank" }, `${s.from}–${s.to}`)), el("td", {}, ((b - a) / 60).toFixed(1)),
      el("td", {}, s.kp ? [el("a", { href: `#/t/${tid}/${s.kp}` }, byId[s.kp]?.label || s.kp), " ", el("span", { class: `chip ${byId[s.kp]?.provenance.status}` }, byId[s.kp]?.provenance.status)] : el("span", { class: "muted" }, s.what), s.kp && s.what ? el("span", { class: "muted small" }, " · " + s.what) : null)));
  }
  return el("div", {}, el("h2", {}, "Video index"),
    el("p", { class: "muted small" }, `Hand segmentation of the talk. ${(mapped / 60).toFixed(1)} of ${(span / 60).toFixed(1)} min (${Math.round(100 * mapped / span)}%) teach a knowledge point; ${(spokenOnly / 60).toFixed(1)} min teach content that exists only in the audio. Blue: written KP · orange: spoken-only KP · grey: not a KP.`),
    band, el("details", {}, el("summary", {}, `${segs.length} stretches`), el("div", { class: "filter-view" }, tbl)));
}
// stretches of the talk that teach one KP, for the KP page
function spokenStretches(d, kid) { return (d.kp.spoken_map || []).filter(s => s.kp === kid); }

// every interactive this thread uses, ungated, with the KPs it serves and their notices
function renderGallery(main, tid) {
  const d = state.data[tid], uses = {};
  for (const k of d.kp.knowledge_points) for (const p of k.presentations || [])
    if ((p.type === "interactive" || p.type === "animation") && WIDGETS[p.ref]) (uses[p.ref] ||= []).push({ k, p });
  main.append(el("h1", {}, "Widget gallery"), el("p", { class: "muted small" }, `${Object.keys(uses).length} widgets used by this thread, shown without gating. On a KP page each one appears only after the learner commits a prediction.`));
  for (const [ref, us] of Object.entries(uses)) {
    const box = el("div", { class: "pres", "data-ref": ref });
    box.append(el("div", { class: "label" }, ref), el("div", { class: "small" }, "serves: ", ...us.map(u => el("a", { href: `#/t/${tid}/${u.k.id}`, style: "margin-right:8px" }, u.k.label))));
    const holder = el("div", {}); box.append(holder);
    try { WIDGETS[ref](holder, us[0].p.notice); } catch (e) { holder.append(el("div", { class: "warn" }, `widget threw: ${e.message}`)); console.error(ref, e); }
    for (const u of us) if (u.p.notice) box.append(el("div", { class: "notice small" }, el("b", {}, `${u.k.id}: `), u.p.notice));
    main.append(box);
  }
}

function renderPaths(main, tid) {
  const d = state.data[tid], byId = Object.fromEntries(d.kp.knowledge_points.map(k => [k.id, k]));
  const col = (title, ids) => el("div", {}, el("h3", {}, title), el("ol", {}, ...ids.map(id => el("li", {}, el("a", { href: `#/t/${tid}/${id}` }, byId[id]?.label || id)))));
  main.append(el("h1", {}, "Two teaching orders"), el("div", { style: "display:grid;grid-template-columns:1fr 1fr;gap:20px" }, col(`source order (${d.kp.path?.basis || ""})`, d.kp.path?.stops || []), col(`course order (${d.kp.course_path?.basis || ""})`, d.kp.course_path?.stops || [])));
}

function renderGraph(main, tid) {
  const d = state.data[tid], kps = d.kp.knowledge_points, byId = Object.fromEntries(kps.map(k => [k.id, k]));
  main.append(el("h1", {}, "Prerequisite graph"), el("p", { class: "muted small" }, "Solid: teaching/call order (evidence). Dashed: author inference. Layered by longest prerequisite chain."));
  const depth = {}; const dep = id => { if (depth[id] !== undefined) return depth[id]; depth[id] = 0; depth[id] = 1 + Math.max(-1, ...(byId[id]?.prerequisites || []).filter(p => byId[p.kp]).map(p => dep(p.kp))); return depth[id]; };
  kps.forEach(k => dep(k.id));
  const layers = {}; kps.forEach(k => (layers[depth[k.id]] ||= []).push(k));
  const W = 900, rowH = 90, H = (Object.keys(layers).length) * rowH + 40;
  const pos = {}; Object.entries(layers).forEach(([l, ks]) => ks.forEach((k, i) => pos[k.id] = { x: 60 + (i + 0.5) * (W - 120) / ks.length, y: 40 + l * rowH }));
  const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, style: "width:100%;background:#fff;border:1px solid var(--rule);border-radius:8px" });
  const ns = "http://www.w3.org/2000/svg";
  for (const k of kps) for (const p of k.prerequisites || []) if (pos[p.kp]) { const a = pos[p.kp], b = pos[k.id]; const ln = document.createElementNS(ns, "line"); ln.setAttribute("x1", a.x); ln.setAttribute("y1", a.y + 14); ln.setAttribute("x2", b.x); ln.setAttribute("y2", b.y - 14); ln.setAttribute("stroke", "#9aa4ad"); if (p.basis !== "teaching-order" && p.basis !== "call-order") ln.setAttribute("stroke-dasharray", "4 3"); svg.append(ln); }
  for (const k of kps) { const g = document.createElementNS(ns, "g"); g.setAttribute("cursor", "pointer"); g.addEventListener("click", () => location.hash = `#/t/${tid}/${k.id}`); const c = document.createElementNS(ns, "circle"); c.setAttribute("cx", pos[k.id].x); c.setAttribute("cy", pos[k.id].y); c.setAttribute("r", 12); c.setAttribute("fill", { mastered: "#3c8d5a", seen: "#e3b23c" }[masteryOf(tid, k.id)] || "#fff"); c.setAttribute("stroke", k.provenance.status === "supplement" ? "#b8582a" : "#24668d"); c.setAttribute("stroke-width", 2); const tx = document.createElementNS(ns, "text"); tx.setAttribute("x", pos[k.id].x); tx.setAttribute("y", pos[k.id].y + 28); tx.setAttribute("text-anchor", "middle"); tx.setAttribute("font-size", "11"); tx.setAttribute("fill", "#25313d"); tx.textContent = k.label.length > 26 ? k.label.slice(0, 24) + "…" : k.label; g.append(c, tx); svg.append(g); }
  main.append(svg);
}

// ------------------------------------------------------------ course map --
function renderMap(app) {
  app.className = "";
  const main = el("div", { class: "main", style: "max-width:1200px;margin:0 auto" }, el("h1", {}, "Course map"),
    el("p", { class: "muted small" }, "Lecture threads as columns in course order, then papers and repos; a dot per knowledge point; solid edges are same-thread prerequisites, dashed edges cross threads (external prerequisites). Dark green = secured (changed case plus a retrieval remembered two days later), green = mastered (changed case answered), yellow = seen, white = ready (prerequisites mastered), grey = locked. Click a dot to open it."));
  const lectures = [...state.threads.threads.filter(t => t.kind === "lecture" && state.data[t.id]).sort((a, b) => a.id.localeCompare(b.id)),
    ...state.threads.threads.filter(t => t.kind === "paper" && state.data[t.id]), ...state.threads.threads.filter(t => t.kind === "repo" && state.data[t.id])];
  const colW = 150, rowH = 26, W = 60 + lectures.length * colW, maxN = Math.max(...lectures.map(t => state.data[t.id].kp.knowledge_points.length));
  const H = 80 + maxN * rowH;
  const ns = "http://www.w3.org/2000/svg";
  const svg = document.createElementNS(ns, "svg"); svg.setAttribute("viewBox", `0 0 ${W} ${H}`); svg.setAttribute("style", "width:100%;background:#fff;border:1px solid var(--rule);border-radius:8px");
  const pos = {};
  lectures.forEach((t, ci) => {
    const kps = state.data[t.id].kp.knowledge_points;
    const order = state.data[t.id].kp.path?.stops || kps.map(k => k.id);
    const ordered = [...order.filter(id => kps.find(k => k.id === id)), ...kps.map(k => k.id).filter(id => !order.includes(id))];
    const head = document.createElementNS(ns, "text"); head.setAttribute("x", 40 + ci * colW); head.setAttribute("y", 30); head.setAttribute("font-size", "13"); head.setAttribute("font-weight", "700"); head.setAttribute("fill", "#25313d"); head.textContent = t.title.replace("CS336 ", "").replace("Paper · ", "📄 ").replace("Repo · ", "⌘ ").slice(0, 22); svg.append(head);
    ordered.forEach((id, ri) => { pos[`${t.id}:${id}`] = { x: 40 + ci * colW, y: 60 + ri * rowH }; });
  });
  const line = (a, b, dashed) => { const l = document.createElementNS(ns, "line"); l.setAttribute("x1", a.x); l.setAttribute("y1", a.y); l.setAttribute("x2", b.x); l.setAttribute("y2", b.y); l.setAttribute("stroke", dashed ? "#b8582a" : "#c5ccd2"); l.setAttribute("stroke-width", dashed ? 1.2 : 1); if (dashed) l.setAttribute("stroke-dasharray", "4 3"); svg.append(l); };
  for (const t of lectures) for (const k of state.data[t.id].kp.knowledge_points) {
    const me = pos[`${t.id}:${k.id}`];
    for (const p of k.prerequisites || []) { const o = pos[`${t.id}:${p.kp}`]; if (o) line(o, me, false); }
    for (const e of k.external_prerequisites || []) { const o = pos[String(e.kp || e)]; if (o) line(o, me, true); }
  }
  for (const [key, p] of Object.entries(pos)) {
    const [tid, kid] = key.split(":"); const k = state.data[tid].kp.knowledge_points.find(x => x.id === kid);
    const g = document.createElementNS(ns, "g"); g.setAttribute("cursor", "pointer"); g.addEventListener("click", () => location.hash = `#/t/${tid}/${kid}`);
    const c = document.createElementNS(ns, "circle"); c.setAttribute("cx", p.x); c.setAttribute("cy", p.y); c.setAttribute("r", 6); const rd = readiness(tid, k, state.data[tid].kp.knowledge_points); c.setAttribute("fill", { secured: "#2f6f45", mastered: "#3c8d5a", seen: "#e3b23c", ready: "#fff", locked: "#eee" }[rd] || "#fff"); c.setAttribute("stroke", rd === "locked" ? "#c5ccd2" : (k.provenance.status === "supplement" ? "#b8582a" : "#24668d")); c.setAttribute("stroke-width", 1.5);
    const tx = document.createElementNS(ns, "text"); tx.setAttribute("x", p.x + 10); tx.setAttribute("y", p.y + 4); tx.setAttribute("font-size", "9.5"); tx.setAttribute("fill", "#25313d"); tx.textContent = k.label.length > 24 ? k.label.slice(0, 22) + "…" : k.label;
    const title = document.createElementNS(ns, "title"); title.textContent = `${tid} · ${k.label}`; g.append(c, tx, title); svg.append(g);
  }
  main.append(svg);
  app.append(main);
}

// -------------------------------------------------------------- session --
const SESSIONS = {
  "l2-core": { title: "L2 core: from FLOPs to memory", minutes: 30, thread: "lecture_02",
    kps: ["matmul-flops", "backward-2x-forward", "six-nd", "napkin-training-time", "arithmetic-intensity", "roofline", "gradient-accumulation"],
    brief: "Seven knowledge points that carry the resource-accounting argument. Commit a prediction before each reveal, answer the changed case, grade the retrieval prompts honestly. Export from the data page when done." },
  "l5-gpu": { title: "L5: why kernels are memory-bound and what tiling buys", minutes: 30, thread: "lecture_05",
    kps: ["gpu-memory-hierarchy", "memory-wall-roofline", "operator-fusion", "memory-coalescing", "tiling-shared-memory", "online-softmax", "flash-attention-forward"],
    brief: "The GPU lecture's spine, ending at FlashAttention. Rendered animations (fusion, tiling, online softmax, the flash tile pass) plus a roofline, warp-access and wave-grid you can drive; step thumbnails are seek points." },
  "rope-paper": { title: "RoPE: paper order", minutes: 25, thread: "rope", kps: null, brief: "The paper's argument in its own order, then compare with the course order on the 'two orders' page." },
  "l1-bpe": { title: "L1: from bytes to BPE", minutes: 25, thread: "lecture_01",
    kps: ["tokenization-efficiency-lens", "compression-ratio", "utf8-variable-length", "byte-tokenizer", "word-tokenizer", "bpe-merge-step", "bpe-encode-apply-merges"],
    brief: "Why tokenization is a compute question, then the professor's own BPE example merge by merge (rendered animation)." },
  "l7-parallelism": { title: "L7: what each parallelism costs in bytes", minutes: 30, thread: "lecture_07",
    kps: ["interconnect-hierarchy", "all-gather", "reduce-scatter", "all-reduce-decomposition", "collective-bytes", "data-parallelism-ddp", "tensor-parallelism", "pipeline-bubble"],
    brief: "Collectives first, then DP / TP / PP as byte budgets; rendered ring all-reduce, TP gather/scatter and pipeline-bubble animations." },
  "l9-scaling": { title: "L9: scaling laws without the mysticism", minutes: 30, thread: "lecture_09",
    kps: ["scaling-law-definition", "power-law-from-estimation-rate", "scaling-law-design-procedure", "critical-batch-size", "kaplan-compute-optimal-allocation", "chinchilla-three-methods", "chinchilla-twenty-tokens-rule", "kaplan-chinchilla-discrepancy"],
    brief: "Where power laws come from, how to run a study, and why Kaplan and Chinchilla disagreed; the IsoFLOP sweep is rendered, and the floor, joint law and allocation are interactive." },
  "l4-moe": { title: "L4: mixture of experts, from routing to balance", minutes: 30, thread: "lecture_04",
    kps: ["moe-params-without-flops", "topk-token-choice-routing", "fine-grained-shared-experts", "nondifferentiable-routing-solutions", "load-balancing-loss", "aux-loss-free-bias", "token-dropping-stochasticity", "expert-parallelism-communication"],
    brief: "Parameters without FLOPs, then every place the routing decision bites: balance, dropping, parallelism." },
  "l8-systems": { title: "L8: parallelism as a systems budget", minutes: 30, thread: "lecture_08",
    kps: ["collective-primitives", "bandwidth-hierarchy", "training-memory-per-param", "zero-stages-1-2", "zero-stage-3-fsdp", "pipeline-bubble", "tensor-parallel-megatron", "tp-vs-pp-communication", "three-d-parallelism-recipe"],
    brief: "Tatsu's accounting view of the same material as L7; do L7 first if you want the code." },
  "l10-inference": { title: "L10: why decode is memory-bound", minutes: 30, thread: "lecture_10",
    kps: ["inference-metrics", "mlp-intensity-batch", "prefill-vs-decode", "attention-decode-intensity", "kv-cache-size", "decode-bandwidth-bound", "batch-latency-throughput-tradeoff", "gqa-kv-reduction", "speculative-sampling-mechanism"],
    brief: "Arithmetic intensity at decode, the KV cache, the batch trade-off, and speculative decoding." },
  "paper-chinchilla": { title: "Chinchilla: three fits of one law", minutes: 25, thread: "chinchilla", kps: null, brief: "Read after L9. The IsoFLOP sweep animation uses the paper's own Approach-3 constants." },
  "paper-grpo": { title: "GRPO: the equations FeynRL implements", minutes: 25, thread: "grpo", kps: null, brief: "Read after L16; every equation links to the FeynRL code lines." },
  "paper-flashattention": { title: "FlashAttention: IO-aware exact attention", minutes: 25, thread: "flashattention", kps: null, brief: "Read after L5; animations of the online softmax and of Algorithm 1's loop order (K/V outside, Q inside), plus an HBM-access ledger against SRAM size." },
  "paper-feynrl": { title: "FeynRL / P3O: trust the batch", minutes: 25, thread: "feynrl_paper", kps: null, brief: "Read after GRPO and the FeynRL repo thread." },
  "l11-scaling-practice": { title: "L11: how labs actually run a scaling study", minutes: 30, thread: "lecture_11",
    kps: ["scaling-in-practice-challenges", "minicpm-mup-fixed-aspect-recipe", "wsd-schedule-branching", "deepseek-direct-batch-lr-fit", "isoflop-ratios-in-the-wild", "optimizer-comparison-confounders", "mup-two-conditions-and-init", "mup-lr-derivation-and-recap"],
    brief: "Read after L9. WSD schedules, direct batch/LR fits, and why muP exists." },
  "l12-evaluation": { title: "L12: what a benchmark number can mean", minutes: 30, thread: "lecture_12",
    kps: ["construct-to-metric", "perplexity-definition", "logprob-vs-generation-benchmarks", "mmlu-exam-benchmark", "arena-pairwise-elo", "llm-judge-length-bias", "train-test-overlap-routes", "methods-vs-models-rules-of-game"],
    brief: "From construct to metric; perplexity across tokenizers; arenas, judges and contamination." },
  "l13-data-sources": { title: "L13: where the tokens come from", minutes: 30, thread: "lecture_13",
    kps: ["training-stages-quality-gradient", "web-is-not-downloadable", "common-crawl-mechanics", "warc-wet-extraction-loss", "fair-use-training-vs-piracy", "rule-based-heuristic-filters", "multi-source-mixtures", "filter-aggressiveness-vs-tokens"],
    brief: "Common Crawl mechanics, extraction loss, copyright framing, and what each dataset dropped." },
  "l14-filtering": { title: "L14: every filter is a bias", minutes: 30, thread: "lecture_14",
    kps: ["filter-target-raw-framework", "kenlm-vs-fasttext-scorers", "quality-positives-define-quality", "exact-dedup-hash-groupby", "jaccard-similarity", "minhash-collision-probability", "lsh-band-sharpening", "mixing-baselines-epoching"],
    brief: "Quality filters, MinHash and LSH by the numbers, and mixing with finite sources." },
  "l17-multimodal": { title: "L17: images as tokens", minutes: 25, thread: "lecture_17",
    kps: ["everything-into-tokens", "clip-contrastive-objective", "clip-preprocess-vit-patches", "siglip-sigmoid-decouples-batch", "vlm-template-encoder-projector-lm", "anyres-tiling-token-budget", "qwen2-dynamic-resolution-tokens", "mrope-interleaved-axes"],
    brief: "CLIP and SigLIP objectives, encoder-projector-LM, and the image token budget." },
  "repo-feynrl": { title: "FeynRL: one RL step through the code", minutes: 30, thread: "feynrl", kps: null, brief: "The call-order path from the gsm8k recipe to weight sync. Pairs with the GRPO and FeynRL papers." },
  "repo-tea": { title: "TheoremExplainAgent: where quality is and is not checked", minutes: 25, thread: "tea", kps: null, brief: "An agentic Manim pipeline traced end to end, with its CLI dispatch, outline parser and fix loop runnable in the page; ends at its silent degradation points." },
  "repo-edtrace": { title: "edtrace: what the lecture traces record", minutes: 20, thread: "edtrace", kps: null, brief: "The tracer behind the executable lectures; explains what our discover tool can and cannot see." },
  "l15-posttraining": { title: "L15: SFT and RLHF, and what goes wrong", minutes: 30, thread: "lecture_15",
    kps: ["sft-then-rlhf", "length-bias-preferences", "sft-tail-knowledge-hallucination", "imitation-vs-optimization", "bradley-terry-reward-model", "ppo-for-rlhf-brief", "dpo-objective", "rlhf-overoptimization"],
    brief: "Imitation then optimization; Bradley-Terry, DPO margins, KL-tilted overoptimization and the refusal frontier are interactive; continue into L16." },
  "l16-rlvr": { title: "L16: from policy gradients to GRPO and R1", minutes: 30, thread: "lecture_16",
    kps: ["verifiable-reward-vs-rlhf", "policy-gradient-variance", "ppo-ratio-clip", "language-rl-is-a-bandit", "grpo-group-zscore-advantage", "baseline-validity-std-division", "grpo-length-bias", "r1-zero-recipe"],
    brief: "The RL spine of the course; continue into the GRPO paper and the FeynRL repo threads for the equations and the code." },
};
function renderSession(app, sid) {
  app.className = "";
  const main = el("div", { class: "main", style: "max-width:860px;margin:0 auto" });
  if (!sid || !SESSIONS[sid]) {
    main.append(el("h1", {}, "Sessions"), el("p", { class: "muted small" }, "A session is a fixed sequence of knowledge points for one sitting, with strict gating on and a participant id set. Give one to a learner, then export their data."));
    for (const [id, sdef] of Object.entries(SESSIONS)) main.append(el("div", { class: "tcard", onclick: () => location.hash = `#/session/${id}` }, el("div", { class: "kind" }, `${sdef.minutes} min · ${sdef.thread}`), el("div", { style: "font-weight:700" }, sdef.title), el("div", { class: "small muted" }, sdef.brief)));
    app.append(main); return;
  }
  const sdef = SESSIONS[sid], d = state.data[sdef.thread];
  const ids = sdef.kps || d.kp.path.stops;
  const pid = el("input", { type: "text", value: participant(), placeholder: "participant id (required)" });
  pid.addEventListener("change", () => { try { localStorage.setItem("atlas.participant", pid.value.trim()); } catch {} });
  const strict = el("input", { type: "checkbox" }); strict.checked = setting("strict", false); strict.addEventListener("change", () => setSetting("strict", strict.checked));
  main.append(el("h1", {}, sdef.title), el("p", {}, sdef.brief),
    el("div", { class: "block" }, el("div", { class: "eyebrow" }, "before you start"), el("label", {}, "participant id ", pid), el("label", { style: "display:block;margin-top:8px" }, strict, " strict gating (recommended for a real session)")),
    el("div", { class: "block" }, el("div", { class: "eyebrow" }, `sequence · ${ids.length} knowledge points`), el("ol", {}, ...ids.map(id => { const k = d.kp.knowledge_points.find(x => x.id === id); return el("li", {}, el("a", { href: `#/t/${sdef.thread}/${id}` }, k ? k.label : id), " ", el("span", { class: `chip ${masteryOf(sdef.thread, id)}` }, masteryOf(sdef.thread, id))); }))),
    el("div", { class: "footer-nav" }, el("button", { class: "primary", onclick: () => { if (!pid.value.trim()) { alert("set a participant id first"); return; } logEvent("session_start", { session: sid }); location.hash = `#/t/${sdef.thread}/${ids.find(id => masteryOf(sdef.thread, id) !== "mastered") || ids[0]}`; } }, "start / continue"), el("a", { href: "#/data" }, "export data when done")));
  app.append(main);
}

// ----------------------------------------------------------------- data --
function renderData(app) {
  app.className = "";
  const b = exportBundle();
  const main = el("div", { class: "main", style: "max-width:800px;margin:0 auto" }, el("h1", {}, "Learner data"),
    el("p", { class: "muted small" }, "Everything this browser recorded: opened knowledge points, committed predictions and transfers with timestamps, self-explanations, animation seeks, and retrieval grades with FSRS state. Export before handing the laptop to someone else, or have a participant export and send you the file."));
  const pid = el("input", { type: "text", value: b.participant, placeholder: "participant id (e.g. p01)" });
  pid.addEventListener("change", () => { try { localStorage.setItem("atlas.participant", pid.value.trim()); } catch {} });
  const strict = el("input", { type: "checkbox" }); strict.checked = setting("strict", false); strict.addEventListener("change", () => setSetting("strict", strict.checked));
  main.append(el("div", { class: "block" }, el("div", { class: "eyebrow" }, "participant"), pid, el("label", { style: "display:block;margin-top:8px" }, strict, " strict gating: a knowledge point stays locked until its prerequisites' changed-case questions are answered")));
  const nCommit = b.events.filter(e => e.type === "commit").length, nRet = b.events.filter(e => e.type === "retrieval").length;
  main.append(el("div", { class: "block" }, el("div", { class: "eyebrow" }, "recorded"),
    el("div", {}, `${Object.keys(b.progress).length} knowledge points touched · ${nCommit} commits · ${nRet} retrieval grades · ${Object.keys(b.cards).length} scheduled cards · ${b.events.length} events`)));
  const dl = el("button", { class: "primary", onclick: () => {
    const blob = new Blob([JSON.stringify(exportBundle(), null, 1)], { type: "application/json" });
    const a = el("a", { href: URL.createObjectURL(blob), download: `atlas-${b.participant || "anon"}-${new Date().toISOString().slice(0, 10)}.json` }); document.body.append(a); a.click(); a.remove();
  } }, "export JSON");
  const file = el("input", { type: "file", accept: "application/json" });
  file.addEventListener("change", async () => {
    const txt = await file.files[0].text(); let obj; try { obj = JSON.parse(txt); } catch { alert("not JSON"); return; }
    if (obj.format !== "atlas-learner-export/1") { alert("unknown format"); return; }
    if (!confirm("Replace this browser's progress, cards and events with the imported file?")) return;
    state.progress = obj.progress || {}; state.cards = obj.cards || {}; saveProgress(); saveCards(state.cards);
    try { localStorage.setItem("atlas.events.v1", JSON.stringify(obj.events || [])); localStorage.setItem("atlas.participant", obj.participant || ""); } catch {}
    route();
  });
  const reset = el("button", { onclick: () => { if (confirm("Clear all progress, cards and events in this browser?")) { try { ["atlas.progress.v1", "atlas.cards.v1", "atlas.events.v1", "atlas.participant"].forEach(k => localStorage.removeItem(k)); } catch {} state.progress = {}; state.cards = {}; route(); } } }, "reset this browser");
  main.append(el("div", { class: "block" }, el("div", { class: "eyebrow" }, "export / import"), dl, " ", el("label", { style: "margin-left:12px" }, "import: ", file), " ", reset));
  const recent = b.events.slice(-40).reverse();
  main.append(el("div", { class: "block" }, el("div", { class: "eyebrow" }, "last events"), el("pre", { style: "max-height:320px;overflow:auto" }, recent.map(e => `${new Date(e.t).toLocaleTimeString()}  ${e.type.padEnd(12)} ${JSON.stringify(Object.fromEntries(Object.entries(e).filter(([k]) => k !== "t" && k !== "type")))}`).join("\n") || "none yet")));
  app.append(main);
}

// --------------------------------------------------------------- review --
function renderReview(app) {
  app.className = "";
  const main = el("div", { class: "main", style: "max-width:800px;margin:0 auto" }, el("h1", {}, "Reviews"));
  // a detail card joins the queue only once every core card of its KP has been remembered at least twice
  const coreSecured = (tid, kp) => (kp.prompts.retrieval || []).every((r, i) => r.layer === "detail" || (state.cards[`${tid}:${kp.id}:r${i}`]?.reps || 0) >= 2);
  const due = dueCards(state.cards).filter(id => {
    const [tid, kid, ri] = id.split(":"); const kp = state.data[tid]?.kp.knowledge_points.find(k => k.id === kid);
    const r = kp?.prompts?.retrieval?.[+String(ri).slice(1)];
    return !r || r.layer !== "detail" || coreSecured(tid, kp);
  });
  if (!due.length) { main.append(el("p", { class: "muted" }, "Nothing due. Retrieval prompts you grade inside a knowledge point come back here on their FSRS schedule (first interval is a few days).")); app.append(main); return; }
  // interleave across threads: shuffle
  due.sort(() => Math.random() - 0.5);
  for (const id of due.slice(0, 25)) {
    const [tid, kid, ri] = id.split(":"); const d = state.data[tid]; if (!d) continue;
    const kp = d.kp.knowledge_points.find(k => k.id === kid); const r = kp?.prompts?.retrieval?.[+ri.slice(1)]; if (!r) continue;
    // the KP label often states the answer, so a review card shows only the thread until it is revealed
    const t = state.threads.threads.find(x => x.id === tid);
    main.append(el("div", { class: "block retrieval" }, el("div", { class: "eyebrow" }, t?.title || tid), retrievalCard(id, r, `knowledge point: ${kp.label}`)));
  }
  app.append(main);
}

boot().catch(e => { $("#app").innerHTML = `<div class="main"><div class="warn">Failed to load: ${esc(e.message)}. Serve this folder over http (python3 -m http.server from atlas/) — file:// blocks fetch.</div></div>`; });
