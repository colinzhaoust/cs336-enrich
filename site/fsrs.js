// Minimal FSRS-6 scheduler (see research/05-product-form.md §b). Two-button UI: G=1 Again, G=3 Good.
export const W = [0.212, 1.2931, 2.3065, 8.2956, 6.4133, 0.8334, 3.0194, 0.001, 1.8722, 0.1666, 0.796, 1.4835, 0.0614, 0.2629, 1.6483, 0.6014, 1.8729, 0.5425, 0.0912, 0.0658, 0.1542];
const FACTOR = Math.pow(0.9, -1 / W[20]) - 1;
const DAY = 86400e3;
const clamp = (x, lo, hi) => Math.min(hi, Math.max(lo, x));

export function retrievability(card, now = Date.now()) {
  if (!card || !card.last) return 0;
  const t = Math.max(0, (now - card.last) / DAY);
  return Math.pow(1 + FACTOR * t / card.S, -W[20]);
}
export function nextInterval(S, r = 0.9) {
  return clamp(Math.round(S / FACTOR * (Math.pow(r, -1 / W[20]) - 1)), 1, 3650);
}
function d0(G) { return clamp(W[4] - Math.exp(W[5] * (G - 1)) + 1, 1, 10); }

export function review(card, G, now = Date.now()) {
  if (!card || !card.last) {
    const S = W[G - 1], D = d0(G);
    return { S, D, last: now, reps: 1, due: now + nextInterval(S) * DAY, lapses: G === 1 ? 1 : 0 };
  }
  const t = (now - card.last) / DAY;
  const R = retrievability(card, now);
  let { S, D } = card;
  const dD = -W[6] * (G - 3);
  const Dp = D + dD * (10 - D) / 9;
  const Dn = clamp(W[7] * d0(4) + (1 - W[7]) * Dp, 1, 10);
  let Sn;
  if (t < 1) {
    Sn = S * Math.exp(W[17] * (G - 3 + W[18])) * Math.pow(S, -W[19]);
  } else if (G === 1) {
    const lapse = W[11] * Math.pow(D, -W[12]) * (Math.pow(S + 1, W[13]) - 1) * Math.exp(W[14] * (1 - R));
    Sn = Math.min(lapse, S / Math.exp(W[17] * W[18]));
  } else {
    const hard = G === 2 ? W[15] : 1, easy = G === 4 ? W[16] : 1;
    Sn = S * (1 + Math.exp(W[8]) * (11 - D) * Math.pow(S, -W[9]) * (Math.exp(W[10] * (1 - R)) - 1) * hard * easy);
  }
  Sn = Math.max(0.01, Sn);
  return { S: Sn, D: Dn, last: now, reps: (card.reps || 0) + 1, due: now + nextInterval(Sn) * DAY, lapses: (card.lapses || 0) + (G === 1 ? 1 : 0) };
}

// storage ---------------------------------------------------------------
const KEY = "atlas.cards.v1";
export function loadCards() { try { return JSON.parse(localStorage.getItem(KEY) || "{}"); } catch { return {}; } }
export function saveCards(c) { try { localStorage.setItem(KEY, JSON.stringify(c)); } catch {} }
export function dueCards(cards, now = Date.now()) {
  return Object.entries(cards).filter(([, c]) => c.due <= now).map(([id]) => id);
}
