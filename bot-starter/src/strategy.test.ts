// Checks for the strategy-C engine. Run with `npm test`.

import assert from "node:assert/strict";
import * as fs from "node:fs";
import { test } from "node:test";
import { bestPartition, EXACT_BEAM } from "./engine/evaluator.js";
import { Lexicon } from "./engine/lexicon.js";
import {
  BAG,
  emptyRack,
  letterIndex,
  rackFrom,
  reward,
  wordReward,
} from "./engine/letters.js";
import { makeRng, Planner, DEFAULT_PARAMS, type MatchView } from "./engine/planner.js";
import { bidPmf, maxBidPmf, PriceTable } from "./engine/pricing.js";
import { unseenPool } from "./engine/tracker.js";
import { fit } from "./engine/learning.js";

const WORDS = fs
  .readFileSync(new URL("../../spacetimedb/wordlist.txt", import.meta.url), "utf8")
  .split("\n");
const lex = new Lexicon(WORDS);

test("rewards match the module's length multiplier", () => {
  assert.equal(reward(3, 5), 5);
  assert.equal(reward(4, 5), 7); // floor(7.5)
  assert.equal(reward(5, 5), 10);
  assert.equal(reward(6, 5), 12); // floor(12.5)
  assert.equal(reward(9, 5), 15);
  assert.equal(wordReward("QUIZ"), Math.floor((10 + 1 + 1 + 10) * 1.5));
});

test("topWords finds the best-paying words, best first", () => {
  const small = new Lexicon(["TACT", "CAT", "AT", "ACT"]);
  assert.deepEqual(
    small.topWords(rackFrom("TACT"), 2).map((w) => w.word),
    ["TACT", "ACT"],
  );
  assert.deepEqual(small.topWords(rackFrom("CX"), 3), []);
  const top = lex.topWords(rackFrom("QUIZERS"), 1)[0]!;
  assert.equal(top.reward, wordReward(top.word));
  assert.ok(lex.has(top.word));
});

test("bestPartition splits a rack into words that add up", () => {
  const rack = rackFrom("QUIZCATDOG");
  const p = bestPartition(lex, rack, EXACT_BEAM, new Map());
  const used = emptyRack();
  let total = 0;
  for (const w of p.words) {
    assert.ok(lex.has(w), w);
    for (const c of w) used[letterIndex(c)]!++;
    total += wordReward(w);
  }
  for (let i = 0; i < 26; i++) assert.ok(used[i]! <= rack[i]!);
  assert.equal(total, p.value);
  assert.ok(p.words.length >= 2);
});

test("rack search stays fast on a large rack", () => {
  const rack = rackFrom("AEINRSTLODGUEMPB");
  const t = performance.now();
  for (let i = 0; i < 20; i++) bestPartition(lex, rack);
  assert.ok((performance.now() - t) / 20 < 20, "partition took 20 ms or more");
});

test("unseen pool removes auctioned, our start, and revealed opponent tiles", () => {
  const me = 1n;
  const them = 2n;
  const unseen = unseenPool({
    myBotId: me,
    myRack: rackFrom("ABCDEQ"), // Q won, ABCDE dealt
    results: [
      { letter: "Q", winnerBotId: me },
      { letter: "Z", winnerBotId: them },
      { letter: "X", winnerBotId: null },
    ],
    words: [{ botId: them, word: "ZOO" }], // OO came from their start
    openLetter: "E",
  });
  const full = BAG.reduce((a, b) => a + b, 0);
  let n = 0;
  for (let i = 0; i < 26; i++) n += unseen[i]!;
  // 3 auctioned + 1 open + 5 our start + 2 revealed O.
  assert.equal(n, full - 3 - 1 - 5 - 2);
  assert.equal(unseen[letterIndex("Q")], 0);
  assert.equal(unseen[letterIndex("O")], BAG[letterIndex("O")]! - 2);
  assert.equal(unseen[letterIndex("E")], BAG[letterIndex("E")]! - 2);
});

test("sim bid model matches the module's decideBid", () => {
  // Aggressive on a vowel: base value+4 = 5, jitter -2..2 → 3..7.
  const pmf = bidPmf({ kind: "sim", strategy: "Aggressive", balance: 100 }, 0, 20);
  for (let b = 3; b <= 7; b++) assert.ok(Math.abs(pmf[b]! - 0.2) < 1e-12);
  // Cheapskate on A: base 0 → clamped up to 1 for jitter ≤ 1.
  const cheap = bidPmf({ kind: "sim", strategy: "Cheapskate", balance: 100 }, 0, 20);
  assert.ok(Math.abs(cheap[1]! - 0.8) < 1e-12);
  // Balance caps the bid.
  const broke = bidPmf({ kind: "sim", strategy: "Aggressive", balance: 4 }, 0, 20);
  assert.ok(Math.abs(broke[4]! - 0.8) < 1e-12);
  // Probabilities sum to one.
  const max = maxBidPmf(
    [
      { kind: "sim", strategy: "Aggressive", balance: 50 },
      { kind: "sim", strategy: "ValueBidder", balance: 50 },
    ],
    letterIndex("Q"),
    50,
  );
  assert.ok(Math.abs(max.reduce((a, b) => a + b, 0) - 1) < 1e-9);
});

test("Vickrey bids are truthful at ΔV/λ; first-price shades", () => {
  const pmf = maxBidPmf(
    [{ kind: "sim", strategy: "ValueBidder", balance: 100 }],
    letterIndex("K"),
    100,
  ); // bids 3..7
  const vick = new PriceTable(pmf, "Vickrey");
  assert.equal(vick.optimal(20, 2, 100).bid, 10);
  assert.equal(vick.optimal(20, 0, 37).bid, 37); // λ=0: money is worthless
  assert.equal(vick.optimal(-1, 1, 100).bid, 0);
  const first = new PriceTable(pmf, "FirstPrice");
  const o = first.optimal(20, 2, 100);
  assert.ok(o.bid <= 8 && o.bid >= 4, `first-price bid ${o.bid}`);
});

test("bid model fit recovers a linear bidder", () => {
  const samples = [];
  for (let v = 1; v <= 10; v++) for (let k = 0; k < 50; k++) samples.push({ value: v, bid: 2 * v + 1 });
  const f = fit(samples);
  assert.ok(Math.abs(f.slope - 2) < 0.2, `slope ${f.slope}`);
  assert.ok(Math.abs(f.intercept - 1) < 1.2, `intercept ${f.intercept}`);
});

function view(over: Partial<MatchView>): MatchView {
  return {
    kind: "Vickrey",
    bagTotal: 40,
    balance: 100,
    rack: rackFrom("AEINR"),
    unseen: Int16Array.from(BAG),
    opponents: [
      { kind: "sim", strategy: "Aggressive", balance: 100 },
      { kind: "sim", strategy: "ValueBidder", balance: 100 },
    ],
    ...over,
  };
}

const quick = { ...DEFAULT_PARAMS, bidTimeMs: 0, playTimeMs: 0, minSamples: 20, maxSamples: 20 };

test("the last auction's letter is worth nothing, and the rack is dumped", () => {
  const p = new Planner(lex, quick, makeRng(1));
  assert.equal(p.decideBid(view({ bagTotal: 0 }), "Z").bid, 0);
  const play = p.decidePlay(view({ bagTotal: 0, rack: rackFrom("RETAINSQUIZ") }));
  assert.ok(play.words.length > 0);
  const used = play.words.join("").length;
  assert.ok(used >= 9, `dumped only ${used} tiles: ${play.words}`);
});

test("second-to-last auction: money is worthless, bid everything on a useful letter", () => {
  const p = new Planner(lex, quick, makeRng(2));
  // RETAIN + S makes a 7-letter word.
  const d = p.decideBid(view({ bagTotal: 1, rack: rackFrom("RETAIN"), balance: 23 }), "S");
  assert.ok(d.dv > 5, `ΔV ${d.dv}`);
  assert.equal(d.bid, 23);
});

test("with no future tiles, ΔV is exactly the rack's gain", () => {
  const p = new Planner(lex, quick, makeRng(3));
  const rack = rackFrom("RETAIN");
  // RETAIN pays 6 × 2.5 = 15, RETAINS 7 × 3 = 21.
  assert.equal(p.decideBid(view({ rack, bagTotal: 1 }), "S").dv, 6);
  // A Q is worth far more here: QINTAR alone pays (10 + 5) × 2.5 = 37.
  assert.ok(p.decideBid(view({ rack, bagTotal: 1 }), "Q").dv >= 22);
});

test("a live decision fits in its time budget", () => {
  const p = new Planner(lex, { ...DEFAULT_PARAMS, bidTimeMs: 200, maxSamples: 1000 });
  const t = performance.now();
  p.decideBid(view({ rack: rackFrom("AEINRSTLOD") }), "E");
  assert.ok(performance.now() - t < 450, "bid decision ran far past its budget");
});
