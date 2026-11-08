// Candidate E: play like the best real bot, then fix what it does badly.
//
// The bid is the imitation fit of a real bot (analysis/imit_try.json, a
// censored regression of its bids on rack and match features, the same
// formula the arena's stand-ins use): coef · [1, value, vowel, min(have,2),
// rackSize, matchFrac, balance/10, tilesWon, lenGain, makes6, makes7+] − shift.
//
// Improvements over the plain imitation (each switchable for the arena):
//   noise 0     bid the mean, not mean + noise (raw: 1 keeps the noise)
//   endAll      in the last `endAll` auctions coins are worth nothing:
//               bid the whole balance on a tile that raises the final split
//   lastPlay    play the best split of the whole rack in the last auction
//   minLen      shortest word played before the end (7 = only 3× words)
//   cashBal     below this balance, cash the best word of 3+ letters

import type { Brain } from "../engine/brains.js";
import { bestPartition, EXACT_BEAM, FAST_BEAM } from "../engine/evaluator.js";
import type { Lexicon } from "../engine/lexicon.js";
import { letterIndex, VALUES, type Rack } from "../engine/letters.js";
import type { MatchView } from "../engine/planner.js";

const MODELS: Record<string, { coef: number[]; shift: number; sd: number; minLen: number; cashBal: number }> = {
  GodOfThudner: {
    coef: [-29.15, 2.41, 0.35, -4.03, 0.68, 16.77, 2.23, 0.2, 3.73, 7.58, 10.78],
    shift: -4, sd: 9.76, minLen: 7, cashBal: 20,
  },
  "LexForge-1107": {
    coef: [-26.37, 2.92, 1.49, -6.24, 0.44, 15.06, 1.76, 0.04, 4.31, 6.12, 9.62],
    shift: -4, sd: 10.34, minLen: 7, cashBal: 20,
  },
};

export interface CloneParams {
  raw: number; // 1 = exact imitation (noise, no improvements)
  endAll: number;
  minLen: number;
  cashBal: number;
  boost: number; // added to every bid
  heavy: number; // extra coins per point of letter value above 1
  target: number; // tiles we aim to win (0 = no pacing)
  pace: number; // coins added per tile behind the target pace (and taken off when ahead)
  spendK: number; // late spend-down: add spendK * balance / auctions left ...
  spendN: number; // ... in the last spendN auctions
  boostSmall: number; // added to every bid at tables smaller than paceSeats (the 1v1 final)
  paceSeats: number; // pacing only at tables of at least this many bots (the 1v1 final does better without)
  floor: number; // bid at least floor + floorSlope·(value−1) on a useful letter of value ≤ floorMaxVal (0 = off)
  floorSlope: number;
  floorMaxVal: number;
  floorRack: number; // "useful": adds length, or the rack has fewer tiles than this
  floorDup: number; // 1 = a short rack makes any letter useful; 0 = only letters the rack doesn't hold yet
  heavyCap: number; // most we bid on a 5+ point letter that adds no length (0 = off)
  lateBag: number; // with this many tiles left or fewer: short words (0 = off) ...
  lateMin: number; // ... cash any word this long, and spend the balance over the auctions left
  lateEvery: number; // ... planning to win one tile in this many
  lateCash: number; // 1 = late, cash any word of lateMin+ letters at once
  fitOnly: number; // 1 = bid 0 on letters that add no length (gain 0) once the rack has 6+
}

const DEFAULTS: CloneParams = { raw: 0, endAll: 4, minLen: 7, cashBal: 20, boost: 0, heavy: 0, target: 0, pace: 2, spendK: 0, spendN: 12, fitOnly: 0, paceSeats: 4, boostSmall: 0,
  floor: 0, floorSlope: 2, floorMaxVal: 4, floorRack: 7, floorDup: 1, heavyCap: 0, lateBag: 0, lateMin: 3, lateEvery: 4, lateCash: 0 };

function gaussian(rng: () => number): number {
  const u = Math.max(rng(), 1e-12);
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * rng());
}

export function createClone(lex: Lexicon, rng: () => number, model: string, over: Partial<CloneParams>): Brain {
  const m = MODELS[model]!;
  const p: CloneParams = { ...DEFAULTS, minLen: m.minLen, cashBal: m.cashBal, ...over };
  const bestLen = (rack: Rack) => lex.topWords(rack, 1, 3)[0]?.word.length ?? 0;
  const total = (v: MatchView) => 98 - 5 * (v.opponents.length + 1);
  return {
    name: p.raw ? "cloneRaw" : "clone",
    lastNote: "",
    newMatch() {},
    bid(view, letter) {
      const L = letterIndex(letter);
      const rack = view.rack.slice() as Rack;
      // The open tile is in no rack yet; bagTotal excludes it.
      const auctions = total(view);
      const done = Math.max(0, auctions - view.bagTotal - 1);
      const frac = done / (done + view.bagTotal + 1);
      if (!p.raw && view.bagTotal < p.endAll) {
        // Coins are worthless soon: pay anything for a tile that adds to the
        // final split, unless it's the very last tile (it can't be played).
        if (view.bagTotal === 0) return 0;
        const before = bestPartition(lex, rack, FAST_BEAM, new Map()).value;
        rack[L]!++;
        const after = bestPartition(lex, rack, FAST_BEAM, new Map()).value;
        this.lastNote = `end ${before}->${after}`;
        return after > before ? view.balance : 0;
      }
      let size = 0;
      for (let i = 0; i < 26; i++) size += rack[i]!;
      const b0 = bestLen(rack);
      rack[L]!++;
      const b1 = bestLen(rack);
      rack[L]!--;
      const gain = Math.max(b1 - b0, 0);
      const x = [1, VALUES[L]!, [0, 4, 8, 14, 20].includes(L) ? 1 : 0, Math.min(rack[L]!, 2), size, frac,
        view.balance / 10, view.tilesWon ?? 0, gain, b1 === 6 && gain > 0 ? 1 : 0, b1 >= 7 && gain > 0 ? 1 : 0];
      // Big table (Swiss/elimination) vs the 1v1 final: the live fixes,
      // boost and pacing apply only at big tables.
      const big = view.opponents.length + 1 >= p.paceSeats;
      let mean = -m.shift + (big ? p.boost : 0) + p.heavy * (VALUES[L]! - 1);
      for (let i = 0; i < m.coef.length; i++) mean += m.coef[i]! * x[i]!;
      if (!p.raw) {
        if (p.target > 0 && view.opponents.length + 1 >= p.paceSeats) {
          const behind = p.target * (done + 1) / auctions - (view.tilesWon ?? 0);
          mean += p.pace * Math.max(-3, Math.min(3, behind));
        }
        if (view.opponents.length + 1 < p.paceSeats) mean += p.boostSmall;
        if (p.spendK > 0 && view.bagTotal < p.spendN) mean += (p.spendK * view.balance) / (view.bagTotal + 1);
        if (p.fitOnly && gain === 0 && size >= 6) mean = Math.min(mean, 4);
        const v = VALUES[L]!;
        const useful = gain > 0 || (size < p.floorRack && (p.floorDup > 0 || rack[L]! === 0));
        if (big && p.floor > 0 && v <= p.floorMaxVal && useful) mean = Math.max(mean, p.floor + p.floorSlope * (v - 1));
        if (big && p.heavyCap > 0 && v >= 5 && gain === 0) mean = Math.min(mean, p.heavyCap);
        if (big && p.lateBag > 0 && view.bagTotal <= p.lateBag) {
          // Late: coins are about to be worthless and a 7 is out of reach
          // from a short rack, so build short words and spend everything.
          const r0 = lex.topWords(rack, 1, 2)[0]?.reward ?? 0;
          rack[L]!++;
          const r1 = lex.topWords(rack, 1, 2)[0]?.reward ?? 0;
          rack[L]!--;
          if (r1 > r0 || size < 4) {
            const share = view.balance / Math.max(1, view.bagTotal / p.lateEvery);
            mean = Math.max(mean, Math.min(share, view.balance));
          }
        }
      }
      const bid = Math.round(mean + (p.raw ? m.sd * gaussian(rng) : 0));
      this.lastNote = `mean=${mean.toFixed(1)} gain=${gain} b1=${b1}`;
      return Math.max(0, Math.min(view.balance, bid));
    },
    play(view) {
      if (view.bagTotal === 0) {
        // Last auction: everything goes.
        return bestPartition(lex, view.rack, EXACT_BEAM, new Map()).words;
      }
      const top = lex.topWords(view.rack, 1, 3)[0];
      if (!top) return [];
      if (top.word.length >= p.minLen || view.balance < p.cashBal) return [top.word];
      if (p.lateCash && p.lateBag > 0 && view.bagTotal <= p.lateBag && top.word.length >= p.lateMin) {
        // Late: unless one more tile could make it a 7, cash it now so the
        // coins buy the next short word.
        if (view.rack.reduce((a, c) => a + c, 0) < 6 || view.bagTotal < 3) return [top.word];
      }
      return [];
    },
  };
}
