// Candidate D: quota price-taker.
//
// Finishing place tracks tiles won (real data: 15–17 tiles → place ~2,
// under 9 → last half the time), so aim for a quota of ~16 tiles and buy
// them as cheaply as the market allows:
//
//  • ΔV: what the letter adds to our rack's expected final score, by a
//    small Monte Carlo (a few future tiles drawn from the unseen pool,
//    best split into words, common random numbers on both sides).
//  • Market: the price to beat is the highest of 5 rival bids, each
//    ≈ the imitation regressions of the top real bots (analysis/
//    imit_try.json) with Gaussian noise. `market(letter)` is the bid that
//    wins with probability `winAt`.
//  • Bid = k·ΔV coins, with k scaled by pace (behind the quota → higher)
//    and by how rich we are against the coins the rest of the quota needs.
//    Behind pace, a useful letter gets at least the market price.
//  • Caps: never dip into `reserve` or bid more than `maxShare` of the
//    balance on an ordinary tile; a tile that completes a 6+ letter word
//    pays for itself at once and may take up to `payback` of what it adds.
//  • Endgame: coins left at the end are worth nothing, so in the last
//    `endgame` auctions the whole balance is available. The last tile
//    can't be played (the match ends when the bag empties), so bid 0 on it.
//  • Words: cash any word of `minLen`+ letters at once (coins recycle into
//    tiles); when the balance is under `broke`, a 5+ word; in the last
//    auction, the best split of everything.
//
// Tuning: QUOTA_PARAMS="target=16,k=1.2,..." (see QUOTA_DEFAULTS).

import type { Brain } from "../engine/brains.js";
import { bestPartition, EXACT_BEAM, FAST_BEAM } from "../engine/evaluator.js";
import type { Lexicon } from "../engine/lexicon.js";
import { letterIndex, rackSize, VALUES, type Rack } from "../engine/letters.js";
import type { MatchView } from "../engine/planner.js";

export interface QuotaParams {
  target: number; // tiles to win per match
  samples: number; // Monte Carlo samples for ΔV
  horizon: number; // most future tiles assumed in a rollout
  k: number; // coins per point of ΔV, at pace and with enough coins
  paceGain: number; // exponent on (expected / actual) tiles won
  richGain: number; // exponent on (balance / coins the quota still needs)
  winAt: number; // win probability that defines the market price
  floorUse: number; // ΔV at which a letter is "useful" (gets the market price when behind)
  margin: number; // coins over the market price when paying it
  reserve: number; // coins kept back from ordinary tiles
  maxShare: number; // most of the balance one ordinary tile may take
  payback: number; // share of a completed word's gain we'll bid for its last tile
  endgame: number; // auctions left (incl. this one) when money stops mattering
  minLen: number; // shortest word cashed at once
  late6: number; // with this many auctions left or fewer, cash 6-letter words too (0 = off)
  broke: number; // below this balance, cash a 5+ letter word
  maxBid: number; // hard cap on any bid outside the endgame
}

export const QUOTA_DEFAULTS: QuotaParams = {
  target: 16,
  samples: 16,
  horizon: 4,
  k: 4,
  paceGain: 1,
  richGain: 0.5,
  winAt: 0.5,
  floorUse: 8,
  margin: 0,
  reserve: 12,
  maxShare: 0.4,
  payback: 0.8,
  endgame: 5,
  minLen: 7,
  late6: 15,
  broke: 20,
  maxBid: 40,
};

export function quotaParamsFromEnv(env = process.env.QUOTA_PARAMS ?? ""): Partial<QuotaParams> {
  const out: Record<string, number> = {};
  for (const kv of env.split(",")) {
    if (!kv.trim()) continue;
    const [k, v] = kv.split("=").map((s) => s.trim());
    if (!k || !(k in QUOTA_DEFAULTS) || v === undefined || Number.isNaN(Number(v))) {
      throw new Error(`QUOTA_PARAMS: bad entry "${kv}"`);
    }
    out[k] = Number(v);
  }
  return out as Partial<QuotaParams>;
}

// Imitation bid regressions of the six real bots (analysis/imit_try.json):
// [1, value, vowel, min(have,2), rackSize, matchFrac, balance/10, tilesWon,
//  lenGain, makes6, makes7+], shift −4 (so +4 on the mean), and noise sd.
const RIVALS: { c: number[]; sd: number }[] = [
  { c: [-29.15, 2.41, 0.35, -4.03, 0.68, 16.77, 2.23, 0.2, 3.73, 7.58, 10.78], sd: 9.76 },
  { c: [-26.37, 2.92, 1.49, -6.24, 0.44, 15.06, 1.76, 0.04, 4.31, 6.12, 9.62], sd: 10.34 },
  { c: [-32.15, 3.19, 1.76, -6.46, 0.63, 17.28, 2.13, 0.16, 4.23, 5.89, 10.09], sd: 10.26 },
  { c: [-30.88, 1.98, 0.08, -3.16, 0.5, 11.08, 2.4, 0.48, 3.86, 9.2, 11.32], sd: 11.23 },
  { c: [-25.16, 2.12, 0.45, -3.25, 0.23, 24.18, 1.93, -0.49, 3.98, 6.8, 9.45], sd: 7.6 },
  { c: [-19.37, 2.12, 0.11, -5.43, 0.12, -1.39, 1.31, 0.71, 5.17, 3.11, 7.69], sd: 11.6 },
];
const VOWEL = new Set([0, 4, 8, 14, 20]);

function normCdf(x: number): number {
  // Abramowitz–Stegun 7.1.26 via erf.
  const t = 1 / (1 + 0.3275911 * Math.abs(x / Math.SQRT2));
  const y =
    1 -
    ((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t + 0.254829592) *
      t *
      Math.exp(-(x * x) / 2);
  return x >= 0 ? (1 + y) / 2 : (1 - y) / 2;
}

// P(highest of `n` rival bids < b), rivals' features guessed from the
// match phase (their racks are hidden).
function winProb(letter: number, frac: number, b: number, n: number): number {
  const x = [
    1,
    VALUES[letter]!,
    VOWEL.has(letter) ? 1 : 0,
    0,
    7,
    frac,
    (100 - 50 * frac) / 10,
    13 * frac,
    0.6,
    0.08,
    0.05,
  ];
  let logP = 0;
  for (const r of RIVALS) {
    let mu = 4;
    for (let i = 0; i < x.length; i++) mu += r.c[i]! * x[i]!;
    // A rival bids round(mu + noise), at least 0; it beats us at ≥ b
    // (ties are coin flips: count half).
    const p = normCdf((b - 1 - mu) / r.sd) + 0.5 * (normCdf((b - mu) / r.sd) - normCdf((b - 1 - mu) / r.sd));
    logP += Math.log(Math.max(1e-9, p));
  }
  return Math.exp((logP * n) / RIVALS.length);
}

function marketPrice(letter: number, frac: number, n: number, q: number): number {
  for (let b = 1; b <= 80; b++) if (winProb(letter, frac, b, n) >= q) return b;
  return 80;
}

export class QuotaBrain implements Brain {
  name = "quota";
  lastNote = "";
  private pool: number[] = [];
  private totalAuctions = 0;
  private marketMemo = new Map<string, number>();

  constructor(
    private lex: Lexicon,
    private rng: () => number,
    readonly p: QuotaParams,
  ) {}

  newMatch() {
    this.totalAuctions = 0;
  }

  private market(letter: number, frac: number, n: number): number {
    const key = `${letter}:${Math.round(frac * 20)}:${n}`;
    let v = this.marketMemo.get(key);
    if (v === undefined) {
      v = marketPrice(letter, Math.round(frac * 20) / 20, n, this.p.winAt);
      this.marketMemo.set(key, v);
    }
    return v;
  }

  // E[S(R ∪ F ∪ {x})] − E[S(R ∪ F)] with |F| = m drawn from the unseen pool.
  private deltaV(rack: Rack, unseen: Rack, x: number, m: number): number {
    const pool = this.pool;
    pool.length = 0;
    for (let i = 0; i < 26; i++) for (let j = 0; j < unseen[i]!; j++) pool.push(i);
    const r = rack.slice() as Rack;
    if (m <= 0 || pool.length === 0) {
      const base = bestPartition(this.lex, r, EXACT_BEAM).value;
      r[x]!++;
      return bestPartition(this.lex, r, EXACT_BEAM).value - base;
    }
    const n = Math.min(m, pool.length);
    let sum = 0;
    const memo = new Map();
    for (let s = 0; s < this.p.samples; s++) {
      for (let j = 0; j < n; j++) {
        const k = j + Math.floor(this.rng() * (pool.length - j));
        const t = pool[j]!;
        pool[j] = pool[k]!;
        pool[k] = t;
        r[pool[j]!]!++;
      }
      const without = bestPartition(this.lex, r, FAST_BEAM, memo).value;
      r[x]!++;
      const withX = bestPartition(this.lex, r, FAST_BEAM, memo).value;
      r[x]!--;
      sum += withX - without;
      for (let j = 0; j < n; j++) r[pool[j]!]!--;
    }
    return sum / this.p.samples;
  }

  // The gain from a word of 6+ letters the letter would let us spell now.
  private completes(rack: Rack, x: number, minLen: number): number {
    const before = this.lex.topWords(rack, 1, minLen)[0]?.reward ?? 0;
    rack[x]!++;
    const after = this.lex.topWords(rack, 1, minLen)[0]?.reward ?? 0;
    rack[x]!--;
    return Math.max(0, after - before);
  }

  private cashLen(left: number): number {
    return this.p.late6 > 0 && left <= this.p.late6 ? Math.min(6, this.p.minLen) : this.p.minLen;
  }

  bid(view: MatchView, letter: string): number {
    try {
      return this.decide(view, letter);
    } catch (e) {
      this.lastNote = `error ${(e as Error).message}`;
      return 0;
    }
  }

  private decide(view: MatchView, letter: string): number {
    const p = this.p;
    const x = letterIndex(letter);
    const left = view.bagTotal + 1; // auctions left, this one included
    const nRivals = Math.max(1, view.opponents.length);
    if (this.totalAuctions === 0) this.totalAuctions = Math.max(left, 98 - 5 * (nRivals + 1));
    const total = Math.max(this.totalAuctions, left);
    const frac = 1 - left / total;
    const bal = view.balance;
    if (view.bagTotal === 0 || bal <= 0) {
      this.lastNote = "last tile: unusable";
      return 0;
    }
    const won = view.tilesWon ?? 0;
    const m = Math.min(p.horizon, left - 2);
    const dv = this.deltaV(view.rack, view.unseen, x, m);
    const done = this.completes(view.rack.slice() as Rack, x, this.cashLen(left));

    if (left <= p.endgame) {
      const b = dv > 0.5 ? Math.min(bal, Math.ceil(dv * 2 + 2)) : 0;
      this.lastNote = `endgame ΔV=${dv.toFixed(1)} bid ${b}`;
      return b;
    }

    // Pace: tiles we should have by now against tiles won.
    const expected = p.target * frac;
    const pace = (expected + 2) / (won + 2);
    // Coins the rest of the quota needs, at about the market price, less
    // what our words should bring back (about half of it).
    const need = Math.max(0, p.target - won);
    const price = this.market(x, frac, nRivals);
    const needCoins = Math.max(1, need * price * 0.5);
    const rich = Math.max(0.3, Math.min(3, (bal + 1) / needCoins));
    const k = p.k * Math.pow(pace, p.paceGain) * Math.pow(rich, p.richGain);

    let bid = dv * k;
    if (pace > 1 && dv >= p.floorUse) bid = Math.max(bid, price + p.margin);
    const capOrdinary = Math.min(p.maxBid, Math.max(0, bal - p.reserve), Math.max(price, bal * p.maxShare));
    let cap = capOrdinary;
    if (done > 0) cap = Math.max(cap, Math.min(bal, done * p.payback));
    bid = Math.max(0, Math.min(bid, cap, bal));
    const b = Math.floor(bid);
    this.lastNote =
      `ΔV=${dv.toFixed(1)} done=${done} mkt=${price} pace=${pace.toFixed(2)} rich=${rich.toFixed(2)} k=${k.toFixed(2)} bid ${b}`;
    return b;
  }

  play(view: MatchView): string[] {
    try {
      return this.choose(view);
    } catch (e) {
      this.lastNote = `error ${(e as Error).message}`;
      return [];
    }
  }

  private choose(view: MatchView): string[] {
    const rack = view.rack;
    if (rackSize(rack) < 2) return [];
    if (view.bagTotal === 0) {
      const part = bestPartition(this.lex, rack, EXACT_BEAM);
      this.lastNote = `final split ${part.words.join(" ")} = ${part.value}`;
      return part.words;
    }
    const top = this.lex.topWords(rack, 1, this.cashLen(view.bagTotal + 1))[0];
    if (top) {
      this.lastNote = `cash ${top.word} (${top.reward})`;
      return [top.word];
    }
    if (view.balance < this.p.broke) {
      const five = this.lex.topWords(rack, 1, 5)[0];
      if (five) {
        this.lastNote = `broke: cash ${five.word} (${five.reward})`;
        return [five.word];
      }
    }
    return [];
  }
}

export function createQuota(
  lex: Lexicon,
  rng: () => number,
  overrides: Partial<QuotaParams> = {},
): Brain {
  return new QuotaBrain(lex, rng, { ...QUOTA_DEFAULTS, ...quotaParamsFromEnv(), ...overrides });
}
