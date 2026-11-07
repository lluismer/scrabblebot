// What the other bidders will bid, as probability distributions over
// integer amounts, and what that means for us.
//
// Simulated bots are modelled exactly: the module's `decideBid` is
// `max(min(max(base + U{-2..2}, 1), balance), 0)` with `base` set by the
// sim's public strategy. Real bots get a fitted linear model, a·value + c
// with Gaussian noise, learned from public auction history.

import { VALUES } from "./letters.js";

export type SimStrategy = "Cheapskate" | "ValueBidder" | "Aggressive" | "Human";

export type OpponentModel =
  | { kind: "sim"; strategy: SimStrategy; balance: number }
  | {
      kind: "learned";
      balance: number;
      slope: number;
      intercept: number;
      sd: number;
      // Chance the bot doesn't bid at all on a letter.
      skip: number;
    };

export type AuctionKind = "Vickrey" | "FirstPrice";

const VOWELS = new Set([0, 4, 8, 14, 20]); // A E I O U

// P(bid = b) for b = 0..cap, where index 0 means "no bid".
export function bidPmf(
  opp: OpponentModel,
  letter: number,
  cap: number,
): Float64Array {
  const pmf = new Float64Array(cap + 1);
  const bal = Math.max(0, Math.min(opp.balance, cap));
  const value = VALUES[letter]!;
  if (opp.kind === "sim") {
    let base: number;
    switch (opp.strategy) {
      case "Human":
        pmf[0] = 1;
        return pmf;
      case "Cheapskate":
        base = value - 1;
        break;
      case "ValueBidder":
        base = value;
        break;
      case "Aggressive":
        base = value + (VOWELS.has(letter) ? 4 : 2);
        break;
    }
    for (let j = -2; j <= 2; j++) {
      const amount = Math.max(Math.min(Math.max(base + j, 1), bal), 0);
      pmf[amount]! += 0.2;
    }
    return pmf;
  }
  const mean = opp.slope * value + opp.intercept;
  const sd = Math.max(0.5, opp.sd);
  pmf[0] = opp.skip;
  // Discretised normal on 1..bal, with the tail above bal piled on bal
  // (a bot can't bid more than it has).
  let mass = 0;
  const w = new Float64Array(cap + 1);
  for (let b = 1; b <= bal; b++) {
    const lo = normCdf((b - 0.5 - mean) / sd);
    const hi = b === bal ? 1 : normCdf((b + 0.5 - mean) / sd);
    const loClamped = b === 1 ? 0 : lo;
    w[b] = Math.max(0, hi - loClamped);
    mass += w[b]!;
  }
  if (mass <= 0) {
    pmf[0] = 1;
    return pmf;
  }
  for (let b = 1; b <= bal; b++) pmf[b] = ((1 - opp.skip) * w[b]!) / mass;
  if (bal === 0) pmf[0] = 1;
  return pmf;
}

// Distribution of the highest bid among `opps` (0 = nobody bids).
export function maxBidPmf(
  opps: readonly OpponentModel[],
  letter: number,
  cap: number,
): Float64Array {
  const cdf = new Float64Array(cap + 1).fill(1);
  for (const o of opps) {
    const pmf = bidPmf(o, letter, cap);
    let acc = 0;
    for (let b = 0; b <= cap; b++) {
      acc += pmf[b]!;
      cdf[b]! *= Math.min(1, acc);
    }
  }
  const out = new Float64Array(cap + 1);
  let prev = 0;
  for (let b = 0; b <= cap; b++) {
    out[b] = Math.max(0, cdf[b]! - prev);
    prev = cdf[b]!;
  }
  return out;
}

export interface BidOutcome {
  bid: number;
  winProb: number;
  expectedPay: number;
}

// Win chance and expected payment for every possible bid against one
// letter's top-rival distribution, from prefix sums so each lookup is O(1).
// A tie with the top rival is a coin flip (approximately: the module
// shuffles all tied bids).
export class PriceTable {
  readonly cap: number;
  private below: Float64Array; // P(M < b)
  private payBelow: Float64Array; // E[max(M,1) · 1{M < b}]
  constructor(
    readonly pmf: Float64Array,
    readonly kind: AuctionKind,
  ) {
    this.cap = pmf.length - 1;
    this.below = new Float64Array(pmf.length + 1);
    this.payBelow = new Float64Array(pmf.length + 1);
    for (let m = 0; m < pmf.length; m++) {
      this.below[m + 1] = this.below[m]! + pmf[m]!;
      this.payBelow[m + 1] = this.payBelow[m]! + pmf[m]! * Math.max(m, 1);
    }
  }

  outcome(bid: number): BidOutcome {
    if (bid <= 0) return { bid: 0, winProb: 0, expectedPay: 0 };
    if (bid > this.cap) {
      // Above every rival's balance: a sure win.
      const pay =
        this.kind === "FirstPrice" ? bid : this.payBelow[this.cap + 1]!;
      return { bid, winProb: 1, expectedPay: pay };
    }
    const tie = 0.5 * this.pmf[bid]!;
    const win = this.below[bid]! + tie;
    const pay =
      this.kind === "FirstPrice"
        ? win * bid
        : this.payBelow[bid]! + tie * bid;
    return { bid, winProb: win, expectedPay: pay };
  }

  // The bid maximising E[ΔV·win − λ·pay].
  //
  // Vickrey: bidding the true worth in coins, ΔV/λ, is optimal whatever
  // the others do, so we bid that rather than trust the opponent model.
  // First-price: search every bid against the modelled distribution.
  optimal(dv: number, lambda: number, balance: number): BidOutcome {
    if (dv <= 0 || balance <= 0) return this.outcome(0);
    if (this.kind === "Vickrey") {
      const truthful = lambda <= 1e-9 ? balance : Math.floor(dv / lambda);
      return this.outcome(Math.min(balance, truthful));
    }
    let best = this.outcome(0);
    let bestU = 0;
    const top = Math.min(balance, this.cap + 1);
    for (let b = 1; b <= top; b++) {
      const o = this.outcome(b);
      const u = dv * o.winProb - lambda * o.expectedPay;
      if (u > bestU + 1e-12) {
        bestU = u;
        best = o;
      }
    }
    return best;
  }
}

// Abramowitz & Stegun 7.1.26, |error| < 1.5e-7.
function normCdf(z: number): number {
  const x = Math.abs(z) / Math.SQRT2;
  const t = 1 / (1 + 0.3275911 * x);
  const y =
    1 -
    ((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) *
      t +
      0.254829592) *
      t *
      Math.exp(-x * x);
  return z >= 0 ? 0.5 * (1 + y) : 0.5 * (1 - y);
}
