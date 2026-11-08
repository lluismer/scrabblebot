// Bid models for real (non-simulated) opponents, fitted from public
// history.
//
// Once a match ends the module publishes each auction's top bid
// (`auction_top_bid_archive`). Joined with `auction_result` that gives us
// (letter, winning bot, its exact bid) for every auction anyone won. We fit
// bid ≈ slope·value + intercept per bot by least squares, pulled toward a
// prior with a few pseudo-observations so a bot we've barely seen isn't
// wildly mis-modelled. Winning bids overstate a bot's typical bid, which
// errs on the side of caution: we'd rather expect to pay a little more.

import { VALUES } from "./letters.js";
import type { OpponentModel } from "./pricing.js";

export interface BidSample {
  value: number;
  bid: number;
}

export interface FittedModel {
  slope: number;
  intercept: number;
  sd: number;
  n: number;
}

// A typical real bot in real-only matches, from the censored fit that uses
// the top bid AND the price paid (src/tools/refit-rivals.ts, 2026-11-07):
// bid ≈ 3.5·value + 0, sd 7. The old prior, 2.9·value + 11.5, came from
// winners' bids alone and sat ~8.5 coins too high.
export const PRIOR: FittedModel = { slope: 3.5, intercept: 0, sd: 7, n: 0 };
const PRIOR_WEIGHT = 8;

export function fit(samples: readonly BidSample[]): FittedModel {
  // Pseudo-observations on the prior line at a low and a high value.
  const pts: [number, number, number][] = [
    [1, PRIOR.slope * 1 + PRIOR.intercept, PRIOR_WEIGHT / 2],
    [8, PRIOR.slope * 8 + PRIOR.intercept, PRIOR_WEIGHT / 2],
    ...samples.map((s): [number, number, number] => [s.value, s.bid, 1]),
  ];
  let sw = 0, sx = 0, sy = 0, sxx = 0, sxy = 0;
  for (const [x, y, w] of pts) {
    sw += w;
    sx += w * x;
    sy += w * y;
    sxx += w * x * x;
    sxy += w * x * y;
  }
  const mx = sx / sw;
  const my = sy / sw;
  const vx = sxx / sw - mx * mx;
  const slope = vx > 1e-9 ? (sxy / sw - mx * my) / vx : PRIOR.slope;
  const intercept = my - slope * mx;
  let ss = PRIOR.sd * PRIOR.sd * PRIOR_WEIGHT;
  for (const s of samples) {
    const e = s.bid - (slope * s.value + intercept);
    ss += e * e;
  }
  const sd = Math.sqrt(ss / (PRIOR_WEIGHT + samples.length));
  return { slope, intercept, sd, n: samples.length };
}

export class OpponentBook {
  private samples = new Map<bigint, BidSample[]>();
  private fitted = new Map<bigint, FittedModel>();
  // Models fitted offline (npm run observe → rivals.json), used until we
  // have seen enough of the bot ourselves.
  private seeds = new Map<bigint, FittedModel>();
  static readonly OWN_SAMPLES_TO_REPLACE_SEED = 30;

  seed(botId: bigint, model: FittedModel) {
    this.seeds.set(botId, model);
    this.fitted.delete(botId);
  }

  // Replace everything we know with a fresh scan of public history.
  rebuild(
    archive: Iterable<{ auctionId: bigint; topBid: bigint }>,
    resultById: (
      id: bigint,
    ) => { letter: string; winnerBotId: bigint | undefined } | undefined,
    isReal: (botId: bigint) => boolean,
  ) {
    this.samples.clear();
    this.fitted.clear();
    for (const a of archive) {
      const r = resultById(a.auctionId);
      if (!r || r.winnerBotId === undefined || !isReal(r.winnerBotId)) continue;
      const value = VALUES[r.letter.charCodeAt(0) - 65] ?? 1;
      let list = this.samples.get(r.winnerBotId);
      if (!list) this.samples.set(r.winnerBotId, (list = []));
      list.push({ value, bid: Number(a.topBid) });
    }
  }

  // One new observation: `botId` won a `letter` auction with `topBid`.
  add(botId: bigint, letter: string, topBid: bigint) {
    const value = VALUES[letter.charCodeAt(0) - 65] ?? 1;
    let list = this.samples.get(botId);
    if (!list) this.samples.set(botId, (list = []));
    list.push({ value, bid: Number(topBid) });
    this.fitted.delete(botId);
  }

  model(botId: bigint, balance: number): OpponentModel {
    let f = this.fitted.get(botId);
    if (!f) {
      // What we see live is winners' bids only, which overstate a bot's
      // bid; never let it replace the censored offline fit. A bot with no
      // history gets the prior.
      f = this.seeds.get(botId) ?? PRIOR;
      this.fitted.set(botId, f);
    }
    return {
      kind: "learned",
      balance,
      slope: f.slope,
      intercept: f.intercept,
      sd: f.sd,
      skip: 0.05,
    };
  }

  describe(botId: bigint): string {
    const f = this.fitted.get(botId) ?? fit(this.samples.get(botId) ?? []);
    return `bid≈${f.slope.toFixed(2)}·v+${f.intercept.toFixed(1)} ±${f.sd.toFixed(1)} (n=${f.n})`;
  }
}
