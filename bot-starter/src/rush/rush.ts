// "Rush" strategy (experiment): build 7+ letter words as fast as possible,
// cash each one the moment it's spellable, and start the next.
//
// The idea: a 7+ letter word pays 3× its letters, and the coins come back
// at once, so the faster we complete one the more often we can do it. The
// letters that complete words are mostly cheap common ones (1–2 points),
// and the field prices tiles by point value, so those are where we should
// win auctions.
//
// Rack value: V(R) = best + 0.15·(sum of the next 20) over all 7-letter
// words w still possible from the unseen pool, scoring each word as
//   reward(w) · q^need(w, R)
// where need is how many tiles we still lack and q the chance of getting
// each one in time. A letter is worth V(R + letter) − V(R): letters that
// bring a big word one step closer are worth a lot, letters that fit no
// reachable word are worth nothing.
//
// Only 7-letter words are targets: the length bonus stops at 7 (3×), so
// an 8th letter adds just its own 3× value for one more tile to win. Live
// (real-bot matches, 2026-11-07): 5.65 points per tile at 7 letters, 5.53
// at 8, 5.43 at 9. An 8+ letter word the rack already spells still gets
// played.
//
// Bid (Vickrey: pay the second price):
//  • A tile that completes a 7+ letter word pays for itself at once (we
//    cash the word the moment we win it), so bid up to `payback` of what
//    the word adds, from the whole balance.
//  • Any other tile: k · ΔV coins, capped at a share of the balance and
//    never dipping into `reserve`, about one tile's going price. Live, the
//    bot spent 153 coins on the first 17 auctions and then sat at 2 coins
//    for the other 50; the reserve keeps it able to buy the tile that
//    finishes a word.
//  • In the last few auctions money is worthless, so everything goes.
//
// Plays: any 7+ letter word as soon as we can spell it (the best one);
// when nearly broke, a 5–6 letter word to get back into the auctions;
// a shorter word only if it lifts us back to the reserve; in the last
// auction, the best split of everything.
//
// Picky: a letter no lead target lacks gets at most `picky` coins (coins,
// not tiles, cap us at ~3 words a match). Late (bag ≤ lateBag) we cash a
// 6 unless one tile would finish a 7.
//
// Tuning: RUSH_PARAMS="q=0.6,k=2,cap=0.35,broke=25,minBid=16,reserve=14,picky=8,lateBag=20"
// The old behaviour: RUSH_PARAMS="picky=999,lateBag=0"

import type { Brain } from "../engine/brains.js";
import { bestPartition, EXACT_BEAM } from "../engine/evaluator.js";
import type { Lexicon } from "../engine/lexicon.js";
import { letterIndex, rackSize, reward, VALUES, type Rack } from "../engine/letters.js";
import type { MatchView } from "../engine/planner.js";

export interface RushParams {
  q: number; // chance of getting each missing tile in time
  k: number; // coins per point of ΔV
  cap: number; // max share of the balance on one tile
  broke: number; // below this balance, cash a 5–6 letter word
  minLen: number; // shortest word the rush aims for
  maxLen: number; // longest word the rush aims for
  minBid: number; // the cap never goes below this (about the clearing price)
  cheapBid: number; // floor bid on 1–2 point letters a lead target needs, while the rack is short
  shortRack: number; // "short" = fewer tiles than this
  dumpBelow: number; // below this balance, cash a short word if it gets us back to `reserve`
  reserve: number; // coins kept back from ordinary bids (not from completing ones)
  payback: number; // share of a completed word's reward we'll bid for the tile that completes it
  endgame: number; // with this many tiles or fewer left in the bag, bid the whole balance
  picky: number; // most we bid on a letter no lead target lacks (and that completes nothing)
  pickyDv: number; // ...when its ΔV is below this (999 = always)
  lateBag: number; // with this many tiles or fewer left, cash a 6 when no 7 is one tile away (0 = off)
}

export const RUSH_DEFAULTS: RushParams = {
  q: 0.6,
  k: 2,
  cap: 0.35,
  broke: 25,
  minLen: 7,
  maxLen: 7,
  minBid: 16,
  cheapBid: 13,
  shortRack: 5,
  dumpBelow: 13,
  reserve: 14,
  payback: 0.8,
  endgame: 4,
  picky: 8,
  pickyDv: 999, // sim (300 paired): 999 → place 2.88, 8 → 3.11
  lateBag: 20,
};

export function rushParamsFromEnv(): RushParams {
  const p: Record<string, number> = { ...RUSH_DEFAULTS };
  for (const kv of (process.env.RUSH_PARAMS ?? "").split(",")) {
    if (!kv.trim()) continue;
    const [k, v] = kv.split("=").map((s) => s.trim());
    if (!k || !(k in RUSH_DEFAULTS) || Number.isNaN(Number(v))) throw new Error(`RUSH_PARAMS: bad entry "${kv}"`);
    p[k] = Number(v);
  }
  return p as unknown as RushParams;
}

interface Target {
  counts: Uint8Array;
  reward: number;
  letters: Uint8Array; // distinct letter indices, for the need loop
}

const TOP = 20;
// How many of the best targets count as "lead" targets, whose missing
// letters earn the cheap-letter floor.
const LEAD = 10;

export class RushBrain implements Brain {
  name = "rush";
  lastNote = "";
  private targets: Target[] = [];

  constructor(
    private lex: Lexicon,
    readonly params: RushParams = RUSH_DEFAULTS,
  ) {
    for (const w of lex.allWords()) {
      if (w.length < params.minLen || w.length > params.maxLen) continue;
      const counts = new Uint8Array(26);
      let base = 0;
      for (const ch of w) {
        const i = letterIndex(ch);
        counts[i]!++;
        base += VALUES[i]!;
      }
      const letters: number[] = [];
      for (let i = 0; i < 26; i++) if (counts[i]) letters.push(i);
      this.targets.push({ counts, reward: reward(w.length, base), letters: Uint8Array.from(letters) });
    }
  }

  newMatch() {}

  // V(R) with an optional extra letter. Words needing more of a letter
  // than the unseen pool holds, or more tiles than auctions left, are out.
  // `lead`, when given, receives the best few targets.
  private value(rack: Rack, unseen: Rack, auctionsLeft: number, extra = -1, lead?: Target[]): number {
    const q = this.params.q;
    const top: number[] = [];
    let floor = 0;
    const leadScore: number[] = [];
    for (const t of this.targets) {
      let need = 0;
      let ok = true;
      for (const i of t.letters) {
        const have = rack[i]! + (i === extra ? 1 : 0);
        const lack = t.counts[i]! - have;
        if (lack > 0) {
          if (lack > unseen[i]! - (i === extra ? 1 : 0)) {
            ok = false;
            break;
          }
          need += lack;
        }
      }
      if (!ok || need > auctionsLeft) continue;
      const s = t.reward * q ** need;
      if (lead && (leadScore.length < LEAD || s > leadScore[leadScore.length - 1]!)) {
        let j = leadScore.length;
        while (j > 0 && leadScore[j - 1]! < s) j--;
        leadScore.splice(j, 0, s);
        lead.splice(j, 0, t);
        if (leadScore.length > LEAD) {
          leadScore.pop();
          lead.pop();
        }
      }
      if (top.length < TOP + 1) {
        top.push(s);
        if (top.length === TOP + 1) {
          top.sort((a, b) => b - a);
          floor = top[TOP]!;
        }
      } else if (s > floor) {
        top[TOP] = s;
        top.sort((a, b) => b - a);
        floor = top[TOP]!;
      }
    }
    if (top.length === 0) return 0;
    top.sort((a, b) => b - a);
    let rest = 0;
    for (let i = 1; i < top.length; i++) rest += top[i]!;
    return top[0]! + 0.15 * rest;
  }

  // The best 7+ letter word the rack spells right now, 0 if none.
  private longNow(rack: Rack): number {
    return this.lex.topWords(rack, 1, this.params.minLen)[0]?.reward ?? 0;
  }

  // Would one more tile still in the unseen pool let us spell a 7+?
  private oneAway(view: MatchView): boolean {
    const r = view.rack.slice() as Rack;
    for (let i = 0; i < 26; i++) {
      if (!view.unseen[i]) continue;
      r[i]!++;
      const hit = this.longNow(r) > 0;
      r[i]!--;
      if (hit) return true;
    }
    return false;
  }

  bid(view: MatchView, letter: string): number {
    if (view.bagTotal === 0 || view.balance <= 0) return 0;
    const x = letterIndex(letter);
    const p = this.params;
    // The tile in the very last auction can't be used; the open tile isn't
    // in `unseen`, so count it back in for the "with it" side.
    const left = Math.max(0, view.bagTotal - 1);
    const unseenWith = view.unseen.slice() as Rack;
    unseenWith[x]!++;
    const lead: Target[] = [];
    const base = this.value(view.rack, view.unseen, left, -1, lead);
    const withIt = this.value(view.rack, unseenWith, left, x);
    const dv = Math.max(0, withIt - base);

    // Does this tile complete a 7+ letter word? The coins come straight
    // back, so it may use the whole balance, reserve included.
    const rackWith = view.rack.slice() as Rack;
    rackWith[x]!++;
    const completes = Math.max(0, this.longNow(rackWith) - this.longNow(view.rack));

    let want = Math.floor(p.k * dv);
    // Restarting after a word: with a short rack every letter is many tiles
    // from a 7-letter word, so ΔV is small and we'd never win the cheap
    // common letters that rebuild it. Bid at least cheapBid on a 1–2 point
    // letter one of the lead targets still lacks.
    const needed = lead.some((t) => t.counts[x]! > view.rack[x]!);
    if (needed && VALUES[x]! <= 2 && rackSize(view.rack) < p.shortRack) want = Math.max(want, p.cheapBid);
    // Picky: coins are what limit us to ~3 words a match, so don't pay the
    // market for a letter none of the lead targets is missing.
    // "Needed" only looks at today's lead targets, so a letter that opens a
    // better one (live: Z at ΔV 24, K and R at ΔV 12) would be clipped too;
    // ΔV ≥ pickyDv is reason enough to pay.
    if (!needed && completes === 0 && dv < p.pickyDv) want = Math.min(want, p.picky);

    let cap: number;
    let why: string;
    if (view.bagTotal <= p.endgame) {
      // Money is worthless at the end: use it all.
      cap = view.balance;
      why = "endgame";
    } else {
      // The share cap alone deadlocks (at 33 coins it allows 11, the market
      // clears at ~13), hence minBid; the reserve stops minBid draining us.
      // The reserve is only for the tile that finishes a 7, so it applies
      // only while one tile would. Otherwise it deadlocks too: live, rack
      // AAAET at 16 coins left cap = 2 for the last 25 auctions.
      const keep = this.oneAway(view) ? p.reserve : 0;
      const spendable = Math.max(0, view.balance - keep);
      cap = Math.min(spendable, Math.max(p.minBid, Math.floor(view.balance * p.cap)));
      why = `cap=${cap}`;
      // Below the reserve, still take a tile nobody else wants (pays 1).
      if (cap === 0 && dv > 0) {
        cap = 1;
        why = "reserve: 1";
      }
    }
    if (completes > 0) {
      const pay = Math.min(view.balance, Math.floor(p.payback * completes));
      if (pay > Math.min(cap, want)) {
        want = pay;
        cap = pay;
        why = `completes +${completes}`;
      }
    }
    const bid = Math.min(cap, want);
    this.lastNote = `ΔV=${dv.toFixed(2)} V=${base.toFixed(1)} ${why}${needed ? " need" : ""}`;
    return bid;
  }

  play(view: MatchView): string[] {
    if (rackSize(view.rack) < 2) return [];
    if (view.bagTotal === 0) {
      const part = bestPartition(this.lex, view.rack, EXACT_BEAM, new Map());
      this.lastNote = `final dump for ${part.value}`;
      return part.words;
    }
    const long = this.lex.topWords(view.rack, 1, this.params.minLen)[0];
    if (long) {
      this.lastNote = `rush word for ${long.reward}`;
      return [long.word];
    }
    // Late in the match, a 7 that's two or more tiles away may never come:
    // cash a 6 unless a single tile would finish a 7.
    if (view.bagTotal <= this.params.lateBag && !this.oneAway(view)) {
      const six = this.lex.topWords(view.rack, 1, 6)[0];
      if (six) {
        this.lastNote = `late: cashing ${six.word.length}-letter word for ${six.reward}`;
        return [six.word];
      }
    }
    if (view.balance < this.params.broke && view.bagTotal > 3) {
      const mid = this.lex.topWords(view.rack, 1, 5)[0];
      if (mid) {
        this.lastNote = `broke: cashing ${mid.reward}`;
        return [mid.word];
      }
    }
    // Below dumpBelow coins we can't win a tile. Cash the best word of any
    // length, but only if it gets us back to the reserve: a 2-coin word
    // buys nothing and burns letters (an S, say) a long word needs.
    if (view.balance < this.params.dumpBelow) {
      const any = this.lex.topWords(view.rack, 1, 2)[0];
      if (any && view.balance + any.reward >= this.params.reserve) {
        this.lastNote = `out of coins: cashing ${any.word.length}-letter word for ${any.reward}`;
        return [any.word];
      }
    }
    return [];
  }
}
