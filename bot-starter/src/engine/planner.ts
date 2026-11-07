// Strategy C: Monte Carlo letter values priced against a budget.
//
// The match is a budget-constrained sequence of auctions. Relax the budget
// with a Lagrange multiplier λ (score per coin) and each auction separates:
//
//   bid to maximise  E[ΔV · win − λ · pay]
//
// where ΔV is how much the letter raises our expected final score. Under
// Vickrey that's simply bidding ΔV/λ coins.
//
//  • ΔV comes from rollouts. Each rollout draws the rest of the bag from
//    the unseen pool, keeps each future tile with the chance we'd win it,
//    and scores the final rack with the best split into words, S(R ∪ F).
//    ΔV(x) = E[S(R ∪ F ∪ {x})] − E[S(R ∪ F)], using the same F on both
//    sides (common random numbers) so the noise largely cancels.
//  • λ is the price that makes expected spending over the rest of the
//    match equal what we'll have: bisect on it against the exact opponent
//    price model. Money left at the end is worth nothing, so λ falls to 0
//    as the bag empties.
//  • Playing a word early turns letters into coins. S is superadditive, so
//    holding never scores less; we play w only when its coins are worth
//    more than the option value it destroys:
//        λ · r(w)  >  E[S(R ∪ F)] − r(w) − E[S(R∖w ∪ F)]
//  • In the last auction, every tile is played: the best split we can find.

import { bestPartition, EXACT_BEAM, FAST_BEAM } from "./evaluator.js";
import type { Lexicon } from "./lexicon.js";
import { emptyRack, letterIndex, rackSize, type Rack } from "./letters.js";
import {
  maxBidPmf,
  PriceTable,
  type AuctionKind,
  type OpponentModel,
} from "./pricing.js";

export interface Params {
  bidTimeMs: number; // rollout time for a bid decision
  playTimeMs: number; // rollout time for a play decision
  maxSamples: number;
  minSamples: number;
  // Fraction of our projected score we count on getting back as coins
  // in time to spend them.
  earnFactor: number;
  lambdaFloor: number;
  // Extra score a play must clear before we cash letters in early.
  playMargin: number;
  // Win chance assumed for each future tile before λ is first solved.
  winPrior: number;
  // Share of samples that also value the letters not up for auction.
  sideShare: number;
  // Denial: a tile we don't buy goes to someone else, and only finishing
  // place counts, so a tile is also worth what it would add to the
  // leading rival. Effective value = ΔV(us) + weight·ΔV(rival); the weight
  // is denial1 against a single opponent (the final) and denialN at a
  // bigger table, where the tile may go to anyone.
  denial1: number;
  denialN: number;
}

export const DEFAULT_PARAMS: Params = {
  bidTimeMs: 250,
  playTimeMs: 150,
  maxSamples: 400,
  minSamples: 24,
  earnFactor: 0.3,
  lambdaFloor: 0,
  playMargin: 0,
  winPrior: 0.3,
  sideShare: 0.25,
  denial1: 1,
  denialN: 0.25,
};

export interface MatchView {
  kind: AuctionKind;
  // Tiles still in the bag, not counting the one up for auction now.
  bagTotal: number;
  balance: number;
  rack: Rack;
  unseen: Rack;
  opponents: readonly OpponentModel[];
  // The leading rival's rack as far as public events show it: tiles won
  // minus tiles played, plus how many starting tiles are still hidden.
  rival?: { rack: Rack; hidden: number };
}

export interface BidDecision {
  bid: number;
  dv: number; // effective value: ours plus the denial share of the rival's
  rivalDv: number; // what the tile would add to the leading rival
  lambda: number;
  winProb: number;
  samples: number;
}

export interface PlayDecision {
  words: string[];
  why: string;
}

// Small, fast, seedable PRNG (mulberry32) so simulations are reproducible.
export function makeRng(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export class Planner {
  // Chance of winning each letter at our current prices.
  private winProb = new Float64Array(26);
  lambda = 0.5;

  constructor(
    readonly lex: Lexicon,
    readonly params: Params = DEFAULT_PARAMS,
    private rng: () => number = Math.random,
  ) {
    this.newMatch();
  }

  newMatch() {
    this.winProb.fill(this.params.winPrior);
    this.lambda = 0.5;
  }

  private denialWeight(view: MatchView): number {
    if (!view.rival) return 0;
    return view.opponents.length <= 1 ? this.params.denial1 : this.params.denialN;
  }

  // One rollout: the future tiles we'd end up with, and (when denial is
  // on) the leading rival's hidden starting tiles plus the future tiles it
  // would get. One shuffle of the unseen pool serves both: its first
  // bagTotal tiles are the rest of the bag in order, and what's left over
  // sits in opponents' starting racks.
  private sampleFuture(view: MatchView, withRival = false): { mine: Rack; theirs: Rack } {
    const pool: number[] = [];
    for (let i = 0; i < 26; i++) {
      for (let k = 0; k < view.unseen[i]!; k++) pool.push(i);
    }
    const swap = (d: number) => {
      const j = d + Math.floor(this.rng() * (pool.length - d));
      const x = pool[j]!;
      pool[j] = pool[d]!;
      pool[d] = x;
      return x;
    };
    // The tile in the very last auction arrives after the match ends.
    const draws = Math.min(pool.length, view.bagTotal);
    const usable = Math.max(0, view.bagTotal - 1);
    const mine = emptyRack();
    const theirs = emptyRack();
    const share = 1 / Math.max(1, view.opponents.length);
    for (let d = 0; d < draws; d++) {
      const x = swap(d);
      if (d >= usable) continue;
      if (this.rng() < this.winProb[x]!) mine[x]!++;
      else if (withRival && this.rng() < share) theirs[x]!++;
    }
    if (withRival && view.rival) {
      const hidden = Math.min(view.rival.hidden, pool.length - draws);
      for (let d = draws; d < draws + hidden; d++) theirs[swap(d)]!++;
      for (let i = 0; i < 26; i++) theirs[i]! += view.rival.rack[i]!;
    }
    return { mine, theirs };
  }

  // Keep sampling? `timeMs` is a hard deadline (bar a handful of samples)
  // when it's set; 0 means "exactly maxSamples", for simulations.
  private more(n: number, start: number, timeMs: number): boolean {
    if (n >= this.params.maxSamples) return false;
    if (timeMs <= 0) return true;
    if (n < Math.min(4, this.params.minSamples)) return true;
    return performance.now() - start < timeMs;
  }

  // ΔV for the letter in focus, from every sample, and for every other
  // letter type from a share of them (they only feed λ, which is a
  // smoother function of them). Plus the baseline E[S(R ∪ F)].
  valuate(
    view: MatchView,
    timeMs: number,
    focus: number | null,
  ): { dv: Float64Array; rivalDv: Float64Array; base: number; n: number } {
    const types: number[] = [];
    for (let i = 0; i < 26; i++) {
      if (view.unseen[i]! > 0 && i !== focus) types.push(i);
    }
    const sum = new Float64Array(26);
    const cnt = new Float64Array(26);
    const weight = this.denialWeight(view);
    const rsum = new Float64Array(26);
    const rcnt = new Float64Array(26);
    let baseSum = 0;
    let n = 0;
    const start = performance.now();
    const usableNow = view.bagTotal >= 1;
    const every = Math.max(1, Math.round(1 / this.params.sideShare));
    const evalWith = (
      r: Rack,
      x: number,
      base: number,
      s: Float64Array,
      c: Float64Array,
    ) => {
      r[x]!++;
      s[x]! += bestPartition(this.lex, r, FAST_BEAM).value - base;
      c[x]!++;
      r[x]!--;
    };
    while (this.more(n, start, timeMs)) {
      const { mine: r, theirs } = this.sampleFuture(view, weight > 0);
      for (let i = 0; i < 26; i++) r[i]! += view.rack[i]!;
      const base = bestPartition(this.lex, r, FAST_BEAM).value;
      baseSum += base;
      if (usableNow) {
        const side = n % every === 0;
        if (focus !== null) evalWith(r, focus, base, sum, cnt);
        if (side) for (const x of types) evalWith(r, x, base, sum, cnt);
        if (weight > 0) {
          const rbase = bestPartition(this.lex, theirs, FAST_BEAM).value;
          if (focus !== null) evalWith(theirs, focus, rbase, rsum, rcnt);
          if (side) for (const x of types) evalWith(theirs, x, rbase, rsum, rcnt);
        }
      }
      n++;
    }
    const dv = new Float64Array(26);
    const rivalDv = new Float64Array(26);
    for (let i = 0; i < 26; i++) {
      const mineDv = cnt[i]! ? sum[i]! / cnt[i]! : 0;
      rivalDv[i] = rcnt[i]! ? rsum[i]! / rcnt[i]! : 0;
      dv[i] = mineDv + weight * rivalDv[i]!;
    }
    return { dv, rivalDv, base: n ? baseSum / n : 0, n };
  }

  // Solve for λ: expected spend over the rest of the match = budget.
  private solveLambda(
    view: MatchView,
    dv: Float64Array,
    base: number,
    tables: (PriceTable | null)[],
    current: number,
  ): number {
    const future = Math.max(0, view.bagTotal - 1);
    let poolSize = 0;
    for (let i = 0; i < 26; i++) poolSize += view.unseen[i]!;
    const budget = view.balance + this.params.earnFactor * base;

    const spendAt = (lambda: number, setWin: boolean) => {
      let perAuction = 0;
      for (let i = 0; i < 26; i++) {
        const t = tables[i];
        if (!t || view.unseen[i]! === 0) continue;
        const o = t.optimal(dv[i]!, lambda, view.balance);
        if (setWin) this.winProb[i] = o.winProb;
        perAuction += (view.unseen[i]! / poolSize) * o.expectedPay;
      }
      const cur = tables[current];
      const now = cur ? cur.optimal(dv[current]!, lambda, view.balance) : null;
      return future * perAuction + (now?.expectedPay ?? 0);
    };

    let lambda: number;
    if (future === 0 || poolSize === 0 || spendAt(0, false) <= budget) {
      lambda = 0;
    } else {
      let lo = 0;
      let hi = 1;
      while (spendAt(hi, false) > budget && hi < 1e4) hi *= 2;
      for (let it = 0; it < 40; it++) {
        const mid = (lo + hi) / 2;
        if (spendAt(mid, false) > budget) lo = mid;
        else hi = mid;
      }
      lambda = hi;
    }
    lambda = Math.max(lambda, future > 0 ? this.params.lambdaFloor : 0);
    if (poolSize > 0) spendAt(lambda, true);
    return lambda;
  }

  private priceTables(view: MatchView): (PriceTable | null)[] {
    let cap = 1;
    for (const o of view.opponents) cap = Math.max(cap, o.balance);
    const tables: (PriceTable | null)[] = [];
    for (let i = 0; i < 26; i++) {
      tables.push(new PriceTable(maxBidPmf(view.opponents, i, cap), view.kind));
    }
    return tables;
  }

  decideBid(view: MatchView, letter: string): BidDecision {
    const x = letterIndex(letter);
    // The last tile arrives after the match ends: worthless.
    if (view.bagTotal === 0 || view.balance <= 0) {
      return { bid: 0, dv: 0, rivalDv: 0, lambda: 0, winProb: 0, samples: 0 };
    }
    const { dv, rivalDv, base, n } = this.valuate(view, this.params.bidTimeMs, x);
    const tables = this.priceTables(view);
    this.lambda = this.solveLambda(view, dv, base, tables, x);
    const o = tables[x]!.optimal(dv[x]!, this.lambda, view.balance);
    return {
      bid: o.bid,
      dv: dv[x]!,
      rivalDv: rivalDv[x]!,
      lambda: this.lambda,
      winProb: o.winProb,
      samples: n,
    };
  }

  decidePlay(view: MatchView): PlayDecision {
    if (rackSize(view.rack) < 2) return { words: [], why: "empty rack" };
    // Last auction: anything not played now is lost.
    if (view.bagTotal === 0) {
      const p = bestPartition(this.lex, view.rack, EXACT_BEAM, new Map());
      return { words: p.words, why: `final dump for ${p.value}` };
    }
    if (this.lambda <= 0) return { words: [], why: "coins are worthless now" };
    const candidates = this.lex.topWords(view.rack, 6);
    if (candidates.length === 0) return { words: [], why: "no words" };

    // Common random numbers: the same futures for every candidate.
    const start = performance.now();
    const holdSum = new Float64Array(1);
    const afterSum = new Float64Array(candidates.length);
    const after = candidates.map((c) => {
      const r = view.rack.slice() as Rack;
      for (const ch of c.word) r[letterIndex(ch)]!--;
      return r;
    });
    let n = 0;
    while (this.more(n, start, this.params.playTimeMs)) {
      const f = this.sampleFuture(view).mine;
      const r = f.slice() as Rack;
      for (let i = 0; i < 26; i++) r[i]! += view.rack[i]!;
      holdSum[0]! += bestPartition(this.lex, r, FAST_BEAM).value;
      after.forEach((a, k) => {
        for (let i = 0; i < 26; i++) r[i] = f[i]! + a[i]!;
        afterSum[k]! += bestPartition(this.lex, r, FAST_BEAM).value;
      });
      n++;
    }
    const hold = holdSum[0]! / n;
    let best: { word: string; gain: number } | null = null;
    candidates.forEach((c, k) => {
      const loss = hold - c.reward - afterSum[k]! / n;
      const gain = this.lambda * c.reward - loss;
      if (!best || gain > best.gain) best = { word: c.word, gain };
    });
    const b = best as { word: string; gain: number } | null;
    if (b && b.gain > this.params.playMargin) {
      return { words: [b.word], why: `liquidity gain ${b.gain.toFixed(1)}` };
    }
    return { words: [], why: "holding" };
  }
}
