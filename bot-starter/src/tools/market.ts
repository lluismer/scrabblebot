// Build hpc/market.json from data/matches.jsonl (npm run observe output):
// what the real field bids, measured only in matches with no sim seats
// and 6 real bots, the setting the tournament plays in.
//
//   npx tsx src/tools/market.ts
//
// Contents:
//   topBid[value][bucket]  histogram of the winning (top) bid, by letter
//                          value and auction-number bucket (7 auctions each).
//                          This is the price a bot at such a table must beat:
//                          the max over the other bidders.
//   clearing[bucket]       mean price paid and mean top bid, per bucket
//   scores                 the six scores of each match (sorted), to place a
//                          simulated score against real ones
//   best                   real bots with the best average place (≥50 matches)
// and prints the residuals of the current per-bot model (fitted on winning
// bids, rivals.json) against the real top bid, by letter value.

import * as fs from "node:fs";
import * as path from "node:path";
import * as readline from "node:readline";
import { fileURLToPath } from "node:url";
import { VALUES } from "../engine/letters.js";

const here = path.dirname(fileURLToPath(import.meta.url));
const DATA = path.join(here, "..", "..", "data");
const OUT = path.join(here, "..", "..", "hpc", "market.json");
const BUCKET = 7;
const MAX_BID = 120;

interface Model { slope: number; intercept: number; sd: number }
const rivals = JSON.parse(fs.readFileSync(path.join(DATA, "rivals.json"), "utf8")) as {
  botId: string; name: string; bidModel: Model;
}[];
const modelOf = new Map(rivals.map((r) => [r.botId, r.bidModel]));
const PRIOR: Model = { slope: 2.9, intercept: 11.5, sd: 6.3 };

const values = [...new Set(VALUES)].sort((a, b) => a - b);
const topBid: Record<number, number[][]> = {};
for (const v of values) topBid[v] = Array.from({ length: 10 }, () => new Array(MAX_BID + 1).fill(0));
const clearing = Array.from({ length: 10 }, () => ({ n: 0, paid: 0, top: 0 }));
const scores: number[][] = [];
const placeBy = new Map<string, { name: string; n: number; place: number; score: number }>();
// Residuals: predicted max of the six seats' modelled bids vs real top bid.
const resid: Record<number, { n: number; pred: number; real: number }> = {};
let kept = 0;

function gauss(): number {
  const u = Math.max(Math.random(), 1e-12);
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * Math.random());
}

const rl = readline.createInterface({ input: fs.createReadStream(path.join(DATA, "matches.jsonl")) });
for await (const line of rl) {
  if (!line.trim()) continue;
  const m = JSON.parse(line);
  if (m.seats.length !== 6 || m.seats.some((s: { sim: boolean }) => s.sim)) continue;
  kept++;
  scores.push(m.seats.map((s: { score: number }) => s.score).sort((a: number, b: number) => b - a));
  for (const s of m.seats) {
    const e = placeBy.get(s.botId) ?? { name: s.name, n: 0, place: 0, score: 0 };
    e.n++; e.place += s.place; e.score += s.score;
    placeBy.set(s.botId, e);
  }
  const models = m.seats.map((s: { botId: string }) => modelOf.get(s.botId) ?? PRIOR);
  m.auctions.forEach((a: { letter: string; winner: string | null; topBid: number | null; paid: number }, i: number) => {
    const b = Math.min(9, Math.floor(i / BUCKET));
    const v = VALUES[a.letter.charCodeAt(0) - 65]!;
    const top = a.winner === null ? 0 : (a.topBid ?? null);
    if (top === null) return;
    topBid[v]![b]![Math.min(MAX_BID, top)]!++;
    const c = clearing[b]!;
    c.n++; c.paid += a.paid; c.top += top;
    if (Math.random() < 0.03) {
      // Model's expected top bid for this auction: max over the six seats.
      let s = 0;
      for (let k = 0; k < 50; k++) {
        let mx = 0;
        for (const md of models) mx = Math.max(mx, md.slope * v + md.intercept + md.sd * gauss());
        s += mx;
      }
      const r = (resid[v] ??= { n: 0, pred: 0, real: 0 });
      r.n++; r.pred += s / 50; r.real += top;
    }
  });
}

const best = [...placeBy.entries()]
  .filter(([, e]) => e.n >= 50)
  .map(([botId, e]) => ({ botId, name: e.name, matches: e.n, avgPlace: +(e.place / e.n).toFixed(3), avgScore: +(e.score / e.n).toFixed(1) }))
  .sort((a, b) => a.avgPlace - b.avgPlace)
  .slice(0, 10);

const meanScore = scores.reduce((a, s) => a + s.reduce((x, y) => x + y, 0) / 6, 0) / scores.length;
fs.writeFileSync(
  OUT,
  JSON.stringify({
    source: `data/matches.jsonl: ${kept} matches with 6 real bots and no sims`,
    bucket: BUCKET,
    maxBid: MAX_BID,
    topBid,
    clearing: clearing.map((c) => ({ paid: +(c.paid / c.n).toFixed(2), top: +(c.top / c.n).toFixed(2), n: c.n })),
    scores,
    best,
  }),
);
console.log(`${kept} real-only 6-bot matches → ${path.relative(process.cwd(), OUT)}`);
console.log(`mean seat score ${meanScore.toFixed(1)}; mean winner ${(scores.reduce((a, s) => a + s[0]!, 0) / scores.length).toFixed(1)}`);
console.log("clearing price by auction bucket (paid / top):", clearing.map((c) => `${(c.paid / c.n).toFixed(1)}/${(c.top / c.n).toFixed(1)}`).join("  "));
console.log("residuals, model's expected top bid vs real top bid, by letter value:");
for (const v of values) {
  const r = resid[v];
  if (!r) continue;
  console.log(`  value ${String(v).padStart(2)}: n ${String(r.n).padStart(6)}  model ${(r.pred / r.n).toFixed(1)}  real ${(r.real / r.n).toFixed(1)}  bias +${((r.pred - r.real) / r.n).toFixed(1)}`);
}
console.log("best average place (≥50 real-only matches):");
for (const b of best.slice(0, 5)) console.log(`  ${b.name.padEnd(22)} ${b.matches} matches  place ${b.avgPlace}  score ${b.avgScore}`);
