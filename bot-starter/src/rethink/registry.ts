// Candidate brains for the strategy rethink, selectable live with
// BRAIN=<name> and in the arena with --brains a,b,c.
//
// Every candidate is wrapped in SafeBrain: an exception in bid() bids 0,
// an exception in play() plays nothing, so a bug can't crash the bot.

import type { Brain } from "../engine/brains.js";
import type { Lexicon } from "../engine/lexicon.js";
import type { MatchView } from "../engine/planner.js";
import { RushBrain, RUSH_DEFAULTS, type RushParams } from "../rush/rush.js";
import { createClone } from "./clone.js";

export type Factory = (lex: Lexicon, rng: () => number, over?: Record<string, number>) => Brain;

export class SafeBrain implements Brain {
  name: string;
  errors = 0;
  constructor(private inner: Brain) {
    this.name = inner.name;
  }
  get lastNote() {
    return this.inner.lastNote;
  }
  newMatch() {
    try {
      this.inner.newMatch();
    } catch (e) {
      this.errors++;
      console.error(`[${this.name}] newMatch failed:`, e);
    }
  }
  bid(view: MatchView, letter: string, timeMs?: number): number {
    try {
      const b = this.inner.bid(view, letter, timeMs);
      if (!Number.isFinite(b) || b < 0) return 0;
      return Math.min(Math.floor(b), Math.max(0, view.balance));
    } catch (e) {
      this.errors++;
      console.error(`[${this.name}] bid failed:`, e);
      return 0;
    }
  }
  play(view: MatchView): string[] {
    try {
      const w = this.inner.play(view);
      return Array.isArray(w) ? w : [];
    } catch (e) {
      this.errors++;
      console.error(`[${this.name}] play failed:`, e);
      return [];
    }
  }
}

function rush(name: string, over: Partial<RushParams>): Factory {
  return (lex, _rng, extra) => {
    const b = new RushBrain(lex, { ...RUSH_DEFAULTS, ...over, ...extra });
    b.name = name;
    return b;
  };
}

// Optional candidates written by parallel workers: loaded if present.
const optional: Record<string, Factory> = {};
async function tryLoad(file: string, exportName: string, names: Record<string, (f: Function) => Factory>) {
  try {
    const mod = (await import(file)) as Record<string, Function>;
    const f = mod[exportName];
    if (typeof f !== "function") return;
    for (const [n, mk] of Object.entries(names)) optional[n] = mk(f);
  } catch {
    // not written yet
  }
}
await tryLoad("./quota.js", "createQuota", {
  quota: (f) => (lex, rng, o) => f(lex, rng, o),
});
await tryLoad("./repairc.js", "createRepairC", {
  repairc: (f) => (lex, rng, o) => f(lex, rng, o),
});
await tryLoad("./wild.js", "createWild", {
  wild: (f) => (lex, rng, o) => f(lex, rng, o),
});

export const CANDIDATES: Record<string, Factory> = {
  // B: the existing 7-letter rush, as committed/edited in src/rush.
  rush7: rush("rush7", {}),
  // C: rush aiming at 6–7 letter words.
  rush6: rush("rush6", { minLen: 6, maxLen: 7 }),
  // E: imitation of the top real bot, plus improvements (clone.ts).
  clone: (lex, rng, o) => createClone(lex, rng, "GodOfThudner", { ...o }),
  // The tournament pick (rethink, 2026-11-07): the GodOfThudner clone with
  // tile pacing toward 20 at 4+ seat tables, cashing short words only
  // below 10 coins, and +4 coins a bid in the 1v1 final. One config for
  // every phase. Sim (1000 matches, 5 imitation rivals): place 1.67–1.69.
  tourney: (lex, rng, o) =>
    createClone(lex, rng, "GodOfThudner", { target: 20, pace: 2, cashBal: 10, boostSmall: 4, ...o }),
  // tourney + fixes from its first live matches (it underbid the live
  // market, overpaid for Q/J that fit nothing, and finished with unspent
  // coins): +2 coins a bid, at least 10 + 2/point on useful letters worth
  // up to 4, at most 14 on 5+ point letters that add no length, and from
  // 20 tiles left, the balance spread over the auctions left. Replay
  // against 916 real matches: place 2.49-2.72 (tourney 2.76-2.91).
  tourney2: (lex, rng, o) =>
    createClone(lex, rng, "GodOfThudner", { target: 20, pace: 2, cashBal: 10, boostSmall: 4, boost: 2, floor: 10, heavyCap: 14, lateBag: 20, ...o }),
  cloneLF: (lex, rng, o) => createClone(lex, rng, "LexForge-1107", { ...o }),
  cloneRaw: (lex, rng, o) => createClone(lex, rng, "GodOfThudner", { raw: 1, ...o }),
  ...optional,
};

// "base" or "base:key=value,key=value" (overrides for the base's params).
export function parseCandidate(spec: string): { base: string; over: Record<string, number> } {
  const [base, rest] = spec.split(":");
  const over: Record<string, number> = {};
  for (const kv of (rest ?? "").split(/[,;]/)) {
    if (!kv.trim()) continue;
    const [k, v] = kv.split("=");
    if (!k || v === undefined || Number.isNaN(Number(v))) throw new Error(`bad override "${kv}" in ${spec}`);
    over[k.trim()] = Number(v);
  }
  if (!CANDIDATES[base!]) throw new Error(`unknown candidate ${base}; have ${Object.keys(CANDIDATES).join(", ")}`);
  return { base: base!, over };
}

export function createCandidate(spec: string, lex: Lexicon, rng: () => number = Math.random): Brain {
  const { base, over } = parseCandidate(spec);
  const b = CANDIDATES[base]!(lex, rng, over);
  b.name = spec;
  return new SafeBrain(b);
}
