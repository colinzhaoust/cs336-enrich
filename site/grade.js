// Deterministic answer grading for committed prompts. Shared by the site (app.js) and the
// test harness (tools/test_grader.mjs), which checks that every official answer is accepted
// by its own grader and that perturbed answers are rejected.
//
// Prompt fields used: format (number | choice | trend | short), answer, tolerance,
// value (optional canonical number or trend class, overrides parsing the prose answer),
// accept (optional list of keywords; any one present in the learner's text is enough).

const MULT = { k: 1e3, thousand: 1e3, m: 1e6, million: 1e6, b: 1e9, g: 1e9, billion: 1e9, t: 1e12, trillion: 1e12 };
const NUM_RE = /([-+−]?(?:\d{1,3}(?:,\d{3})+(?![\d,])(?:\.\d+)?|\d+,\d{1,2}(?![\d,])|\d+(?:\.\d+)?|\.\d+)(?:\s*[eE]\s*[-+−]?\d+)?)(?:\s*\/\s*(\d+(?:\.\d+)?))?(?:\s*(%|[kKmMbBtT](?![a-zA-Z])|\(?\s*(?:thousand|million|billion|trillion)\b)|([gG])(?![a-zA-Z]))?/;

const SUP = { "⁰": "0", "¹": "1", "²": "2", "³": "3", "⁴": "4", "⁵": "5", "⁶": "6", "⁷": "7", "⁸": "8", "⁹": "9", "⁻": "-", "⁺": "+" };

// Parse the first number in a string. Returns {value, mantissa, scaled} or null.
// "300 (billion)" -> value 3e11, mantissa 300; "34%" -> 0.34; "2/3" -> 0.667; "1.4T" -> 1.4e12.
export function parseNumber(s) {
  const t = String(s ?? "")
    // superscript exponents after 10 ("10²³", "10⁻³") -> "10^23", "10^-3"
    .replace(/10\s*([⁻⁺]?[⁰¹²³⁴⁵⁶⁷⁸⁹]+)/g, (_, e) => "10^" + [...e].map(c => SUP[c]).join(""))
    // "7.2 × 10^23", "7.2x10^23", "7.2*10^23", "7.2·10^23" -> "7.2e23"
    .replace(/\s*[x*×·]\s*10\s*\^\s*([-+−]?\d+)/gi, "e$1")
    // a bare power ("10^-3", "2^25") is evaluated
    .replace(/(^|[^\d.e])(\d+(?:\.\d+)?)\s*\^\s*([-+−]?\d+)/gi, (_, pre, b, e) => pre + String(Math.pow(+b, +e.replace("−", "-"))));
  const m = NUM_RE.exec(t);
  if (!m) return null;
  const num = /^[-+−]?\d+,\d{1,2}$/.test(m[1]) ? m[1].replace(",", ".") : m[1].replace(/,/g, "");
  let x = parseFloat(num.replace(/−/g, "-").replace(/\s+/g, ""));
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
  ["up", /\b(up|rises?|rising|rose|risen|increases?|increased|increasing|grows?|grew|grown|growing|higher|larger|bigger|more|widens?|widened|longer|doubles?|doubled|triples?|tripled|jumps?|jumped|too (?:high|large|big|much)|over-?estimates?|over-?estimated)\b/g],
  ["down", /\b(down|falls?|fell|fallen|falling|decreases?|decreased|decreasing|drops?|dropped|dropping|lower|smaller|less|fewer|shrinks?|shrank|shrunk|narrows?|narrowed|shorter|collapses?|collapsed|halves|halved|reduces?|reduced|decays?|decayed|plummets?|plummeted|too (?:low|small|little)|under-?estimates?|under-?estimated)\b/g],
  ["same", /\b(same|unchanged|unaffected|not affected|no effect|flat|constant|no change|stays?|identical|neither up nor down|(?:does not|doesn't|don't|did not|didn't|will not|won't) (?:change|move)|doesnt change|dont change)\b/g],
];
// a direction word is negated when not/never/neither/nor/n't comes within three words before it,
// with no punctuation in between ("does not go down" is negated; "not lower, higher" keeps "higher")
const NEG = /(?:\b(?:not|never|neither|nor|dont|doesnt|didnt|isnt|wont|cannot)|n't)(?:\s+[\w']+){0,2}\s+$/;
// an explanation after the answer ("up, because fewer tokens ...", "down: log Z falls ...") is ignored
// when the classes disagree
const WHY = /\s*(?:[:;(—–]|\s-\s|\bbecause\b|\bsince\b|\bdue to\b|\bas a result\b|\bso\b)/;
function trendHits(t) {
  const hits = [];
  for (const [cls, re] of TREND) {
    re.lastIndex = 0;
    let m;
    while ((m = re.exec(t))) if (!NEG.test(t.slice(0, m.index))) hits.push({ cls, index: m.index });
  }
  return hits.sort((a, b) => a.index - b.index);
}
// Official answers (firstOnly): the earliest non-negated direction word within the first 40 characters.
// Learner input: the one class all non-negated direction words agree on, else null (graded wrong).
export function trendClass(s, firstOnly = true) {
  const t = String(s ?? "").toLowerCase().replace(/[’‘]/g, "'");
  const hits = trendHits(t);
  if (!hits.length) return null;
  if (firstOnly) return hits[0].index > 40 ? null : hits[0].cls;
  const classes = new Set(hits.map(h => h.cls));
  if (classes.size === 1) return hits[0].cls;
  const w = WHY.exec(t);
  if (!w) return null;
  const head = new Set(trendHits(t.slice(0, w.index)).map(h => h.cls));
  return head.size === 1 ? [...head][0] : null;
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
