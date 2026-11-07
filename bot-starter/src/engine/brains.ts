// A "brain" is everything a bot decides, behind one interface, so the live
// bot and the offline simulator drive the same code.

import type { Lexicon } from "./lexicon.js";
import { letterOf, rackSize, valueOf, type Rack } from "./letters.js";
import {
  DEFAULT_PARAMS,
  Planner,
  type MatchView,
  type Params,
} from "./planner.js";

export interface Brain {
  name: string;
  newMatch(): void;
  bid(view: MatchView, letter: string): number;
  // Words to play right now, in order. Called after every rack change and
  // when each auction opens.
  play(view: MatchView): string[];
  // A line about the last decision, for logs.
  lastNote?: string;
}

export class StrategyC implements Brain {
  name = "strategy-c";
  readonly planner: Planner;
  lastNote = "";
  constructor(lex: Lexicon, params: Params = DEFAULT_PARAMS, rng?: () => number) {
    this.planner = new Planner(lex, params, rng);
  }
  newMatch() {
    this.planner.newMatch();
  }
  bid(view: MatchView, letter: string): number {
    const d = this.planner.decideBid(view, letter);
    this.lastNote =
      `ΔV=${d.dv.toFixed(2)} (rival ${d.rivalDv.toFixed(1)}) λ=${d.lambda.toFixed(3)} ` +
      `P(win)=${d.winProb.toFixed(2)} n=${d.samples}`;
    return d.bid;
  }
  play(view: MatchView): string[] {
    const d = this.planner.decidePlay(view);
    this.lastNote = d.why;
    return d.words;
  }
}

// The starter kit's original strategy, kept as the benchmark to beat.
export class Baseline implements Brain {
  readonly name = "baseline";
  constructor(private lex: Lexicon) {}
  newMatch() {}
  bid(view: MatchView, letter: string): number {
    const value = valueOf(letter);
    const have = view.rack[letter.charCodeAt(0) - 65]!;
    const want = have === 0 ? value + 2 : Math.max(1, value);
    return Math.min(want, Math.max(0, view.balance));
  }
  play(view: MatchView): string[] {
    if (rackSize(view.rack) < 2) return [];
    const top = this.lex.topWords(view.rack, 1);
    // The starter plays the longest word, not the best-paying one; close
    // enough for a benchmark, and slightly kinder to it.
    return top.length ? [top[0]!.word] : [];
  }
}

export function rackLetters(r: Rack): string {
  let s = "";
  for (let i = 0; i < 26; i++) s += letterOf(i).repeat(r[i]!);
  return s;
}
