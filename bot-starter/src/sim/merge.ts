// Merge the JSON summaries written by `run.ts --worker` (one per file, e.g.
// one per HPC array task) and print the totals, grouped by directory name.
//
//   npx tsx src/sim/merge.ts results/*/*.json

import * as fs from "node:fs";
import * as path from "node:path";

interface Tally {
  n: number;
  wins: number;
  place: number;
  score: number;
  ms: number;
}

const groups = new Map<string, Record<string, Tally>>();
for (const file of process.argv.slice(2)) {
  const text = fs.readFileSync(file, "utf8").trim();
  if (!text) continue;
  const group = path.basename(path.dirname(file));
  const into = groups.get(group) ?? {};
  groups.set(group, into);
  for (const [k, t] of Object.entries(JSON.parse(text) as Record<string, Tally>)) {
    const d = (into[k] ??= { n: 0, wins: 0, place: 0, score: 0, ms: 0 });
    d.n += t.n;
    d.wins += t.wins;
    d.place += t.place;
    d.score += t.score;
    d.ms += t.ms;
  }
}

for (const [group, s] of [...groups].sort()) {
  console.log(`== ${group}`);
  for (const [k, t] of Object.entries(s)) {
    // Binomial standard error on the win rate, to judge differences.
    const p = t.wins / t.n;
    const se = Math.sqrt((p * (1 - p)) / t.n);
    console.log(
      `${k.padEnd(14)} n ${String(t.n).padStart(6)}  win ${(100 * p).toFixed(1)}% ±${(100 * se).toFixed(1)}` +
        `  place ${(t.place / t.n).toFixed(3)}  score ${(t.score / t.n).toFixed(1)}`,
    );
  }
}
