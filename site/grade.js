// Deterministic answer grading for committed prompts. Shared by the site (app.js) and the
// test harness (tools/test_grader.mjs), which checks that every official answer is accepted
// by its own grader and that perturbed answers are rejected.
//
// Prompt fields used: format (number | choice | trend | short), answer, tolerance,
// value (optional canonical number or trend class, overrides parsing the prose answer),
// accept (optional list of keywords; any one present in the learner's text is enough).

const MULT = { k: 1e3, thousand: 1e3, m: 1e6, million: 1e6, b: 1e9, g: 1e9, billion: 1e9, t: 1e12, trillion: 1e12 };
const NUM_RE = /([-+−]?(?:\d[\d,]*(?:\.\d+)?|\.\d+)(?:\s*[eE]\s*[-+−]?\d+)?)(?:\s*\/\s*(\d+(?:\.\d+)?))?(?:\s*(%|[kKmMbBtT](?![a-zA-Z])|\(?\s*(?:thousand|million|billion|trillion)\b)|([gG])(?![a-zA-Z]))?/;

// Parse the first number in a string. Returns {value, mantissa, scaled} or null.
// "300 (billion)" -> value 3e11, mantissa 300; "34%" -> 0.34; "2/3" -> 0.667; "1.4T" -> 1.4e12.
export function parseNumber(s) {
  const t = String(s ?? "").replace(/×\s*10\^?\s*([-+−]?\d+)/g, "e$1").replace(/·10\^?([-+−]?\d+)/g, "e$1");
  const m = NUM_RE.exec(t);
  if (!m) return null;
  let x = parseFloat(m[1].replace(/,/g, "").replace(/−/g, "-").replace(/\s+/g, ""));
  if (!isFinite(x)) return null;
  if (m[2]) { const d = parseFloat(m[2]); if (d) x = x / d; }
  const mantissa = x;
  let scaled = false;
  // "1.5G" is giga only when the G touches the number; "−0.645 g" is a symbol (gradient, grams), not a scale
  const unit = m[3] || m[4];
  if (unit) {
    const u = unit.replace(/[(\s]/g, "").toLowerCase();
    if (u === "%") { x = x / 100; scaled = true; }
    else if (MULT[u]) { x = x * MULT[u]; scaled = true; }
  }
  return { value: x, mantissa, scaled };
}

const TREND = [
  ["up", /\b(up|rises?|rising|increases?|increasing|grows?|growing|higher|larger|bigger|more|widens?|longer|improves?)\b/],
  ["down", /\b(down|falls?|falling|decreases?|decreasing|drops?|dropping|lower|smaller|less|fewer|shrinks?|narrows?|shorter|collapses?)\b/],
  ["same", /\b(same|unchanged|flat|constant|no change|stays?|identical|does not change|doesn't change)\b/],
];
export function trendClass(s, firstOnly = true) {
  const t = String(s ?? "").toLowerCase();
  let best = null;
  for (const [cls, re] of TREND) {
    const m = re.exec(t);
    if (m && (best === null || m.index < best.index)) best = { cls, index: m.index };
  }
  if (!best) return null;
  return firstOnly && best.index > 40 ? null : best.cls;
}

function close(x, y, tol) {
  return Math.abs(x - y) <= tol * Math.max(Math.abs(y), 1e-12);
}

export function judge(p, given) {
  const g = String(given ?? "").trim();
  const gl = g.toLowerCase();
  if (!g) return false;
  if (Array.isArray(p.accept) && p.accept.some(k => gl.includes(String(k).toLowerCase()))) return true;
  const fmt = p.format || "short";
  if (fmt === "number") {
    const tol = p.tolerance ?? 0.05;
    const gx = parseNumber(g);
    if (!gx) return false;
    const target = typeof p.value === "number" ? { value: p.value, mantissa: p.value, scaled: false } : parseNumber(p.answer);
    if (!target) return false;
    if (close(gx.value, target.value, tol)) return true;
    // learner typed the bare number in the answer's unit ("300" for "300 (billion)", "34" for "34%")
    if (!gx.scaled && target.scaled && close(gx.mantissa, target.mantissa, tol)) return true;
    return false;
  }
  if (fmt === "choice") {
    const a = String(p.answer).trim().toLowerCase();
    return gl === a;
  }
  if (fmt === "trend") {
    const want = (typeof p.value === "string" && p.value) || trendClass(p.answer);
    const got = trendClass(g, false);
    if (want && got) return want === got;
    if (want && !got) return false;
    // answer has no recognisable direction: fall through to token overlap
  }
  const a = String(p.answer).toLowerCase();
  const toks = a.split(/[^a-z0-9.]+/).filter(w => w.length > 2);
  const hit = toks.filter(w => gl.includes(w)).length;
  return toks.length ? hit / toks.length >= 0.5 : gl === a.trim();
}
