// Smoke test: build the BRAIN the live bot would build and time a few decisions.
//   BRAIN="clone:target=20,pace=2,cashBal=10" npx tsx src/rethink/smoke.ts
import * as path from "node:path";
import { fileURLToPath } from "node:url";
import { Lexicon } from "../engine/lexicon.js";
import { BAG, rackFrom, type Rack } from "../engine/letters.js";
import type { MatchView } from "../engine/planner.js";
import { createBrain } from "../strategy.js";

const here = path.dirname(fileURLToPath(import.meta.url));
const lex = Lexicon.fromFile(path.join(here, "..", "..", "..", "spacetimedb", "wordlist.txt"));
const brain = createBrain(lex);
brain.newMatch();
const opp = (balance: number) => ({ kind: "learned" as const, balance, slope: 2.5, intercept: 0, sd: 10, skip: 0.05 });
let worst = 0;
for (const [rack, bag, bal, won] of [["AEIRS", 67, 100, 0], ["AEIRSTN", 40, 35, 6], ["QZXJK", 20, 3, 9], ["ETAOINS", 0, 50, 15], ["", 5, 0, 0]] as const) {
  const view: MatchView = {
    kind: "Vickrey", bagTotal: bag, balance: bal, rack: rackFrom(rack) as Rack,
    unseen: Int16Array.from(BAG) as Rack, opponents: [1, 2, 3, 4, 5].map(() => opp(60)), tilesWon: won,
  };
  for (const L of "AEQZST") {
    const t0 = performance.now();
    const b = brain.bid(view, L, 300);
    const w = brain.play(view);
    const ms = performance.now() - t0;
    worst = Math.max(worst, ms);
    if (L === "E") console.log(`${rack || "-"} bag=${bag} bal=${bal}: bid(E)=${b} play=${JSON.stringify(w)} ${ms.toFixed(1)}ms (${brain.lastNote ?? ""})`);
  }
}
console.log(`brain ${brain.name}: worst decision ${worst.toFixed(1)} ms`);
