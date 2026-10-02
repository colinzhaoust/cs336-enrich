// Shared helpers for per-thread widget files in site/widgets.d/. Import, do not copy.
// A widget is (root, notice) => void. Keep the numeric model a separate exported pure function
// so tools/check_widgets.mjs can test it against the KP's own answers.
export const el = (tag, attrs = {}, ...kids) => {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "class") e.className = v; else if (k.startsWith("on")) e.addEventListener(k.slice(2), v); else if (k === "html") e.innerHTML = v; else e.setAttribute(k, v);
  }
  for (const k of kids) e.append(k instanceof Node ? k : document.createTextNode(String(k)));
  return e;
};
export const fmt = (x, d = 2) => Number.isFinite(x) ? (Math.abs(x) >= 1e5 || (Math.abs(x) < 1e-2 && x !== 0) ? x.toExponential(d) : x.toLocaleString(undefined, { maximumFractionDigits: d })) : "–";
export function slider(label, min, max, value, step, onInput, fmtv = (v) => v) {
  const out = el("span", { class: "readout" }, fmtv(value));
  const inp = el("input", { type: "range", min, max, value, step });
  inp.addEventListener("input", () => { out.textContent = fmtv(+inp.value); onInput(+inp.value); });
  return el("label", {}, `${label} `, out, inp);
}
