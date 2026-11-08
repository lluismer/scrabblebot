// Replay real matches against the real market: our brain takes one seat
// of a real 6-bot match and faces the bids that actually happened.
//
//   npx tsx src/rethink/replay.ts --brains "tourney strategyc" --file replay.json --workers 12
//
// For each auction the best rival bid is known from the record: the
// winner's bid (topBid) if someone else won, the second price (paid) if
// the seat we replaced won. We win when we bid more and pay that bid.
// Our 5 starting tiles are drawn from the tiles that were dealt (the bag
// minus every auctioned letter). The other seats keep their real scores,
// less `ppt` points for each tile we take from them. Rivals don't react
// to us: same for every brain, so the ranking is what matters.

import { spawn } from "node:child_process";
import * as fs from "node:fs";
import * as path from "node:path";
import { fileURLToPath } from "node:url";
import { StrategyC, type Brain } from "../engine/brains.js";
import { Lexicon } from "../engine/lexicon.js";
import { BAG, emptyRack, letterIndex, wordReward, type Rack } from "../engine/letters.js";
import { DEFAULT_PARAMS, makeRng, type MatchView } from "../engine/planner.js";
import { PRIOR } from "../engine/learning.js";
import { LIVE_DEFAULTS } from "../strategy.js";
import { createCandidate, SafeBrain } from "./registry.js";

const here = path.dirname(fileURLToPath(import.meta.url));
const WORDLIST = path.join(here, "..", "..", "..", "spacetimedb", "wordlist.txt");
function arg(name: string, def: string): string {
  const i = process.argv.indexOf(`--${name}`);
  return i >= 0 ? process.argv[i + 1]! : def;
}
const brains = arg("brains", "tourney").split(/[ |]+/).filter(Boolean);
const file = arg("file", "replay.json");
const workers = Number(arg("workers", "1"));
const limit = Number(arg("matches", "100000"));
const seed = Number(arg("seed", "1"));
const ppt = Number(arg("ppt", "6"));
const cSamples = Number(arg("c-samples", "20"));

type Rec = { id: string; seats: [string, number, string][]; a: [string, string | null, number, number][] };
interface Tally { n: number; place: number; place2: number; first: number; last: number; score: number; won: number; paid: number; coins: number; ms: number }
const blank = (): Tally => ({ n: 0, place: 0, place2: 0, first: 0, last: 0, score: 0, won: 0, paid: 0, coins: 0, ms: 0 });

function makeBrain(spec: string, lex: Lexicon, rng: () => number): Brain {
  if (spec === "strategyc") {
    const b = new StrategyC(lex, { ...DEFAULT_PARAMS, ...LIVE_DEFAULTS, bidTimeMs: 0, playTimeMs: 0, minSamples: cSamples, maxSamples: cSamples }, rng);
    return new SafeBrain(b);
  }
  return createCandidate(spec, lex, rng);
}

function replayOne(brain: Brain, lex: Lexicon, m: Rec, seatIdx: number, rng: () => number) {
  const X = m.seats[seatIdx]![0];
  const N = m.a.length;
  // Dealt tiles = bag minus auctioned letters.
  const dealt = Array.from(BAG);
  for (const [L] of m.a) dealt[letterIndex(L)]!--;
  const pool: number[] = [];
  dealt.forEach((c, i) => { for (let k = 0; k < Math.max(0, c); k++) pool.push(i); });
  const rack: Rack = emptyRack();
  for (let k = 0; k < 5 && pool.length; k++) rack[pool.splice(Math.floor(rng() * pool.length), 1)[0]!]!++;
  let balance = 100, score = 0, won = 0, paid = 0;
  const taken = new Map<string, number>();
  const seen = emptyRack(); // auctioned letters others got (visible to us)
  brain.newMatch();
  const view = (i: number): MatchView => {
    const unseen = Int16Array.from(BAG) as Rack;
    for (let k = 0; k < 26; k++) unseen[k] = Math.max(0, unseen[k]! - rack[k]! - seen[k]!);
    const frac = i / N;
    return {
      kind: "Vickrey", bagTotal: N - i - 1, balance, rack: rack.slice() as Rack, unseen, tilesWon: won,
      opponents: [0, 1, 2, 3, 4].map(() => ({ kind: "learned" as const, balance: Math.round(100 - 70 * frac), slope: PRIOR.slope, intercept: PRIOR.intercept, sd: PRIOR.sd, skip: 0.05 })),
    };
  };
  const play = (w: string) => {
    if (!lex.has(w)) return false;
    const need = emptyRack();
    for (const c of w) { const k = letterIndex(c); if (++need[k]! > rack[k]!) return false; }
    for (let k = 0; k < 26; k++) rack[k]! -= need[k]!;
    const r = wordReward(w);
    balance += r; score += r;
    return true;
  };
  let ms = 0;
  for (let i = 0; i < N; i++) {
    const [L, winner, pd, top] = m.a[i]!;
    const t0 = performance.now();
    const b = Math.min(balance, Math.max(0, Math.floor(brain.bid(view(i), L))));
    for (let k = 0; k < 8; k++) {
      const ws = brain.play(view(i));
      if (!ws.length) break;
      let any = false;
      for (const w of ws) any = play(w) || any;
      if (!any) break;
    }
    ms += performance.now() - t0;
    const comp = winner === null ? 0 : winner === X ? pd : top;
    const win = b > comp || (b === comp && b > 0 && rng() < 0.5);
    const li = letterIndex(L);
    if (win && b > 0) {
      const price = comp > 0 ? comp : Math.min(1, b);
      if (price <= balance) {
        balance -= price; paid += price; won++; rack[li]!++;
        if (winner !== null && winner !== X) taken.set(winner, (taken.get(winner) ?? 0) + 1);
        continue;
      }
    }
    seen[li]!++;
  }
  const rivals = m.seats.filter((_, j) => j !== seatIdx).map(([id, s]) => s - ppt * (taken.get(id) ?? 0));
  const place = 1 + rivals.filter((s) => s > score).length;
  const last = rivals.every((s) => s >= score) && rivals.some((s) => s > score) ? 1 : 0;
  return { place, last, score, won, paid, coins: balance, ms };
}

function runWorker(first: number, count: number) {
  const lex = new Lexicon(fs.readFileSync(WORDLIST, "utf8").split("\n"));
  const recs = (JSON.parse(fs.readFileSync(file, "utf8")) as Rec[]).slice(0, limit);
  let cur = makeRng(1);
  const rngRef = () => cur();
  const made = new Map<string, Brain>();
  const out: Record<string, Tally> = {};
  for (let j = first; j < Math.min(first + count, recs.length); j++) {
    for (const spec of brains) {
      cur = makeRng(seed * 1_000_003 + j);
      let b = made.get(spec);
      if (!b) made.set(spec, (b = makeBrain(spec, lex, rngRef)));
      const seat = Math.floor(cur() * 6);
      const r = replayOne(b, lex, recs[j]!, seat, cur);
      const t = (out[spec] ??= blank());
      t.n++; t.place += r.place; t.place2 += r.place * r.place; t.first += r.place === 1 ? 1 : 0; t.last += r.last;
      t.score += r.score; t.won += r.won; t.paid += r.paid; t.coins += r.coins; t.ms += r.ms;
    }
  }
  return out;
}

function report(s: Record<string, Tally>) {
  console.log(`replay file=${path.basename(file)} brains=${brains.join(",")} ppt=${ppt}`);
  console.log("name                         n   place  ±SE  first%  last%  tiles  price  score  coinsEnd  ms/m");
  for (const [k, t] of Object.entries(s).sort((a, b) => a[1].place / a[1].n - b[1].place / b[1].n)) {
    const mean = t.place / t.n, se = Math.sqrt(Math.max(0, t.place2 / t.n - mean * mean) / t.n);
    console.log(`${k.padEnd(26)} ${String(t.n).padStart(4)}  ${mean.toFixed(2)}  ${se.toFixed(2)}  ${(100 * t.first / t.n).toFixed(1).padStart(5)}  ${(100 * t.last / t.n).toFixed(1).padStart(5)}  ${(t.won / t.n).toFixed(1).padStart(5)}  ${(t.won ? t.paid / t.won : 0).toFixed(1).padStart(5)}  ${(t.score / t.n).toFixed(1).padStart(5)}  ${(t.coins / t.n).toFixed(1).padStart(7)}  ${(t.ms / t.n).toFixed(0).padStart(5)}`);
  }
}

if (process.argv.includes("--worker")) {
  process.stdout.write("\n" + JSON.stringify(runWorker(Number(arg("first", "0")), Number(arg("count", "0")))) + "\n");
} else {
  const total = Math.min(limit, (JSON.parse(fs.readFileSync(file, "utf8")) as Rec[]).length);
  if (workers <= 1) report(runWorker(0, total));
  else {
    const per = Math.ceil(total / workers);
    const jobs = Array.from({ length: workers }, (_, w) => new Promise<Record<string, Tally>>((resolve, reject) => {
      const c = spawn(process.execPath, [...process.execArgv, fileURLToPath(import.meta.url), ...process.argv.slice(2), "--worker", "--first", String(w * per), "--count", String(per)], { stdio: ["ignore", "pipe", "inherit"] });
      let o = "";
      c.stdout.on("data", (d) => (o += d));
      c.on("exit", (code) => code ? reject(new Error(`worker ${code}`)) : resolve(JSON.parse(o.trim().split("\n").pop()!)));
    }));
    const all = await Promise.all(jobs);
    const t: Record<string, Tally> = {};
    for (const s of all) for (const [k, v] of Object.entries(s)) {
      const d = (t[k] ??= blank());
      for (const f of Object.keys(v) as (keyof Tally)[]) d[f] += v[f];
    }
    report(t);
  }
}
