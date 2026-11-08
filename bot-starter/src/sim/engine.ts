// An offline copy of the module's match rules (spacetimedb/src/game.ts):
// dealing, sealed-bid auctions, Vickrey / first-price payment, the sim
// bots' bidding and word play, and the end of the match when the bag
// empties. Used to measure brains without the network.

import type { Brain } from "../engine/brains.js";
import { bestPartition, EXACT_BEAM } from "../engine/evaluator.js";
import { PRIOR } from "../engine/learning.js";
import type { Lexicon } from "../engine/lexicon.js";
import {
  BAG,
  emptyRack,
  letterIndex,
  letterOf,
  STARTING_BALANCE,
  VALUES,
  wordReward,
  type Rack,
} from "../engine/letters.js";
import type { MatchView } from "../engine/planner.js";
import type {
  AuctionKind,
  OpponentModel,
  SimStrategy,
} from "../engine/pricing.js";
import {
  knownHoldings,
  unseenPool,
  type AuctionEvent,
  type WordEvent,
} from "../engine/tracker.js";

export type Seat =
  | { kind: "sim"; strategy: SimStrategy; name: string }
  | { kind: "brain"; brain: Brain; name: string }
  | ({ kind: "rival"; name: string } & RivalModel);

// A stand-in for a real bot, from its public history (hpc/rivals.json):
// it bids slope·value + intercept − shift with Gaussian noise, plays a
// word once it can spell one of at least minLen letters, and plays the
// best split of everything in the last auction.
export interface RivalModel {
  slope: number;
  intercept: number;
  sd: number;
  minLen: number;
  // The fit comes from winning bids, which overstate a typical bid;
  // shift is calibrated so the stand-ins win and pay what the real bots do.
  shift: number;
  // Imitation stand-in (analysis/report.py): a censored-regression bid on
  // [1, value, vowel, min(have,2), rackSize, matchFrac, balance/10, tilesWon,
  //  lenGain, makes6, makes7+], and words: play the best word once it has
  // minLen letters, or any word of 3+ when the balance is under cashBal.
  coef?: number[];
  cashBal?: number;
}

function bestLen(lex: Lexicon, rack: Rack): number {
  const top = lex.topWords(rack, 1, 3)[0];
  return top ? top.word.length : 0;
}

export function imitationBid(
  lex: Lexicon, m: RivalModel, rack: Rack, letter: number, balance: number,
  tilesWon: number, frac: number, rng: () => number,
): number {
  const c = m.coef!;
  let size = 0;
  for (let i = 0; i < 26; i++) size += rack[i]!;
  const b0 = bestLen(lex, rack);
  rack[letter]!++;
  const b1 = bestLen(lex, rack);
  rack[letter]!--;
  const gain = Math.max(b1 - b0, 0);
  const x = [1, VALUES[letter]!, [0, 4, 8, 14, 20].includes(letter) ? 1 : 0,
    Math.min(rack[letter]!, 2), size, frac, balance / 10, tilesWon, gain,
    b1 === 6 && gain > 0 ? 1 : 0, b1 >= 7 && gain > 0 ? 1 : 0];
  let mean = -m.shift;
  for (let i = 0; i < c.length; i++) mean += c[i]! * x[i]!;
  return Math.max(0, Math.min(balance, Math.round(mean + m.sd * gaussian(rng))));
}

function gaussian(rng: () => number): number {
  const u = Math.max(rng(), 1e-12);
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * rng());
}

// The module's sim pool: two of each strategy.
export const SIM_POOL: Seat[] = [
  { kind: "sim", strategy: "Cheapskate", name: "Cheapo-sim" },
  { kind: "sim", strategy: "ValueBidder", name: "Valor-sim" },
  { kind: "sim", strategy: "Aggressive", name: "Brutus-sim" },
  { kind: "sim", strategy: "ValueBidder", name: "Hagrid-sim" },
  { kind: "sim", strategy: "Aggressive", name: "Maverick-sim" },
  { kind: "sim", strategy: "Cheapskate", name: "Snippet-sim" },
];

// How a brain models another brain it has no history on: the live bot's
// prior for an unknown real bot (learning.ts).
export const LEARNED_PRIOR = {
  slope: PRIOR.slope,
  intercept: PRIOR.intercept,
  sd: PRIOR.sd,
  skip: 0.05,
};

interface Player {
  seat: Seat;
  id: bigint;
  balance: number;
  score: number;
  rack: Rack;
}

export interface MatchResult {
  names: string[];
  scores: number[];
  // 1 = first. Equal scores share the better place.
  places: number[];
  won: number[]; // tiles won at auction
  paid: number[]; // coins paid for them
  ms: number[]; // decision time per seat
}

// The sim bots' word choice: the longest word of at least 3 letters,
// alphabetically first among equals (findBestPlayable in dictionary.ts).
export class SimWords {
  private words: string[];
  private counts: Uint8Array[];
  constructor(words: Iterable<string>) {
    this.words = [...new Set([...words].map((w) => w.trim().toUpperCase()))]
      .filter((w) => /^[A-Z]{3,}$/.test(w))
      .sort((a, b) => b.length - a.length || (a < b ? -1 : a > b ? 1 : 0));
    this.counts = this.words.map((w) => {
      const c = new Uint8Array(26);
      for (const ch of w) c[ch.charCodeAt(0) - 65]!++;
      return c;
    });
  }
  best(rack: Rack): string | undefined {
    let total = 0;
    for (let i = 0; i < 26; i++) total += rack[i]!;
    for (let k = 0; k < this.words.length; k++) {
      const w = this.words[k]!;
      if (w.length > total) continue;
      const c = this.counts[k]!;
      let ok = true;
      for (let i = 0; i < 26; i++) {
        if (c[i]! > rack[i]!) {
          ok = false;
          break;
        }
      }
      if (ok) return w;
    }
    return undefined;
  }
}

function simBid(
  strategy: SimStrategy,
  letter: number,
  balance: number,
  rng: () => number,
): number {
  const value = VALUES[letter]!;
  const vowel = [0, 4, 8, 14, 20].includes(letter);
  let base: number;
  switch (strategy) {
    case "Human":
      return 0;
    case "Cheapskate":
      base = value - 1;
      break;
    case "ValueBidder":
      base = value;
      break;
    case "Aggressive":
      base = value + (vowel ? 4 : 2);
      break;
  }
  const jitter = Math.floor(rng() * 5) - 2;
  return Math.max(Math.min(Math.max(base + jitter, 1), balance), 0);
}

export function playMatch(
  seats: Seat[],
  lex: Lexicon,
  simWords: SimWords,
  rng: () => number,
  kind: AuctionKind = "Vickrey",
  rackSize = 5,
): MatchResult {
  const players: Player[] = seats.map((seat, i) => ({
    seat,
    id: BigInt(i + 1),
    balance: STARTING_BALANCE,
    score: 0,
    rack: emptyRack(),
  }));
  const ms = seats.map(() => 0);
  const won = seats.map(() => 0);
  const paid = seats.map(() => 0);
  const bag = Int16Array.from(BAG);
  let bagTotal = BAG.reduce((a, b) => a + b, 0);
  const draw = (): number | undefined => {
    if (bagTotal === 0) return undefined;
    let idx = Math.floor(rng() * bagTotal);
    for (let i = 0; i < 26; i++) {
      if (idx < bag[i]!) {
        bag[i]!--;
        bagTotal--;
        return i;
      }
      idx -= bag[i]!;
    }
    return undefined;
  };

  for (const p of players) {
    for (let n = 0; n < rackSize; n++) {
      const x = draw();
      if (x !== undefined) p.rack[x]!++;
    }
  }
  for (const p of players) {
    if (p.seat.kind === "brain") p.seat.brain.newMatch();
  }

  const results: AuctionEvent[] = [];
  const words: WordEvent[] = [];

  const playWord = (p: Player, word: string): boolean => {
    if (!lex.has(word)) return false;
    const need = emptyRack();
    for (const c of word) {
      const i = letterIndex(c);
      if (++need[i]! > p.rack[i]!) return false;
    }
    for (let i = 0; i < 26; i++) p.rack[i]! -= need[i]!;
    const r = wordReward(word);
    p.balance += r;
    p.score += r;
    words.push({ botId: p.id, word });
    return true;
  };

  // The opponent with the highest score: the one to deny tiles to.
  const leaderOf = (p: Player) =>
    players
      .filter((o) => o !== p)
      .reduce((a, b) => (b.score > a.score ? b : a));

  const viewFor = (p: Player, openLetter: string | null): MatchView => ({
    rival: knownHoldings(leaderOf(p).id, results, words, rackSize),
    kind,
    bagTotal,
    tilesWon: won[players.indexOf(p)] ?? 0,
    balance: p.balance,
    rack: p.rack.slice() as Rack,
    unseen: unseenPool({
      myBotId: p.id,
      myRack: p.rack,
      results,
      words,
      openLetter,
    }),
    opponents: players
      .filter((o) => o !== p)
      .map((o): OpponentModel => {
        const s = o.seat;
        if (s.kind === "sim") {
          return { kind: "sim", strategy: s.strategy, balance: o.balance };
        }
        if (s.kind === "rival") {
          // What the live bot would fit from this rival's history.
          return {
            kind: "learned",
            balance: o.balance,
            slope: s.slope,
            intercept: s.intercept,
            sd: s.sd,
            skip: 0.05,
          };
        }
        return { kind: "learned", balance: o.balance, ...LEARNED_PRIOR };
      }),
  });

  let letter = draw();
  while (letter !== undefined) {
    const ch = letterOf(letter);
    const bids: { p: Player; amount: number }[] = [];
    for (const p of players) {
      if (p.seat.kind === "sim") {
        const a = simBid(p.seat.strategy, letter, p.balance, rng);
        if (a > 0) bids.push({ p, amount: a });
      }
    }
    // Rivals bid from their fitted line, then play what their habit says.
    for (const p of players) {
      const s = p.seat;
      if (s.kind !== "rival") continue;
      const pi = players.indexOf(p);
      const frac = results.length / (results.length + bagTotal + 1);
      const a = s.coef
        ? imitationBid(lex, s, p.rack, letter, p.balance, won[pi]!, frac, rng)
        : Math.max(0, Math.min(p.balance, Math.round(
            s.slope * VALUES[letter]! + s.intercept - s.shift + s.sd * gaussian(rng))));
      if (a > 0) bids.push({ p, amount: a });
      if (bagTotal === 0) {
        const part = bestPartition(lex, p.rack, EXACT_BEAM, new Map());
        for (const w of part.words) playWord(p, w);
      } else {
        for (let k = 0; k < 4; k++) {
          const top = lex.topWords(p.rack, 1, 3)[0];
          const cash = s.cashBal !== undefined && p.balance < s.cashBal;
          if (!top || (top.word.length < s.minLen && !cash) || !playWord(p, top.word)) break;
        }
      }
    }
    // Brains bid, then play whatever they want during the auction window.
    for (const [i, p] of players.entries()) {
      if (p.seat.kind !== "brain") continue;
      const t0 = performance.now();
      const amount = Math.min(p.balance, p.seat.brain.bid(viewFor(p, ch), ch));
      if (amount > 0) bids.push({ p, amount });
      for (let k = 0; k < 8; k++) {
        const plays = p.seat.brain.play(viewFor(p, ch));
        if (plays.length === 0) break;
        let any = false;
        for (const w of plays) any = playWord(p, w) || any;
        if (!any) break;
      }
      ms[i]! += performance.now() - t0;
    }

    // Close: shuffle, stable sort by amount, so ties go at random.
    for (let i = bids.length - 1; i >= 1; i--) {
      const j = Math.floor(rng() * (i + 1));
      [bids[i], bids[j]] = [bids[j]!, bids[i]!];
    }
    bids.sort((a, b) => b.amount - a.amount);
    let winner: Player | null = null;
    const [first, second] = bids;
    if (first) {
      const price =
        kind === "FirstPrice"
          ? first.amount
          : second
            ? second.amount
            : Math.min(1, first.amount);
      if (first.p.balance >= price) {
        first.p.balance -= price;
        first.p.rack[letter]!++;
        winner = first.p;
        const wi = players.indexOf(first.p);
        won[wi]!++;
        paid[wi]! += price;
      }
    }
    results.push({ letter: ch, winnerBotId: winner?.id ?? null });
    if (winner && winner.seat.kind === "sim") {
      const w = simWords.best(winner.rack);
      if (w) playWord(winner, w);
    }
    letter = draw();
  }

  const scores = players.map((p) => p.score);
  const places = scores.map((s) => 1 + scores.filter((o) => o > s).length);
  return { names: seats.map((s) => s.name), scores, places, ms, won, paid };
}
