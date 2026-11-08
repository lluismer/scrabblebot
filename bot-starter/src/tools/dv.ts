// Print the planner's ΔV for every letter, given a rack.
//   npx tsx src/tools/dv.ts SATIRE            (bag=60 by default)
//   npx tsx src/tools/dv.ts QAEIT 20          (rack, tiles left in bag)
// Uses LIVE_PARAMS (what the live bot runs), no rivals, no denial.
import path from "node:path";
import { fileURLToPath } from "node:url";
import { Lexicon } from "../engine/lexicon.js";
import { Planner, makeRng, type MatchView } from "../engine/planner.js";
import { BAG, letterOf, rackFrom } from "../engine/letters.js";
import { LIVE_PARAMS } from "../strategy.js";

const rackStr = (process.argv[2] ?? "").toUpperCase();
const bag = Number(process.argv[3] ?? 60);
const here = path.dirname(fileURLToPath(import.meta.url));
const lex = Lexicon.fromFile(path.join(here, "..", "..", "..", "spacetimedb", "wordlist.txt"));
const rack = rackFrom(rackStr);
const unseen = rackFrom("");
for (let i = 0; i < 26; i++) unseen[i] = Math.max(0, BAG[i]! - rack[i]!);

const view: MatchView = {
  kind: "Vickrey" as MatchView["kind"],
  bagTotal: bag,
  balance: 100,
  rack,
  unseen,
  opponents: [],
};
const planner = new Planner(lex, { ...LIVE_PARAMS, minSamples: 400, maxSamples: 400 }, makeRng(1));
const { dv, base, n } = planner.valuate(view, 5000, null);
console.log(`rack=${rackStr || "(empty)"} bag=${bag} base=${base.toFixed(1)} samples=${n}`);
[...dv.keys()]
  .sort((a, b) => dv[b]! - dv[a]!)
  .forEach((i) => console.log(`${letterOf(i)}  ${dv[i]!.toFixed(2).padStart(7)}`));
