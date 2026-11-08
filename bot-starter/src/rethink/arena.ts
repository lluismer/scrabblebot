// Tournament between rethink candidates and the imitation field.
//
//   npx tsx src/rethink/arena.ts --brains clone,rush7 --mode field --matches 400 --workers 8
//
// Modes:
//   field   each brain alone with 5 imitation rivals (same seeds per brain)
//   mixed   --k brains (default 3) drawn at random + imitation rivals, 6 seats
//   final   each brain 1v1 against --rival (default GodOfThudner), series of
//           --games (default 3) on total score
//   rivals  6 imitation rivals, to check them against the real field
// Options: --seed S, --rivals-file F (default analysis/imit_try.json),
//          --per-rival, --out file.json

import { spawn } from "node:child_process";
import * as fs from "node:fs";
import * as path from "node:path";
import { fileURLToPath } from "node:url";
import type { Brain } from "../engine/brains.js";
import { Lexicon } from "../engine/lexicon.js";
import { makeRng } from "../engine/planner.js";
import { playMatch, SimWords, type RivalModel, type Seat } from "../sim/engine.js";
import { createCandidate, parseCandidate, SafeBrain } from "./registry.js";

const here = path.dirname(fileURLToPath(import.meta.url));
const WORDLIST = path.join(here, "..", "..", "..", "spacetimedb", "wordlist.txt");

function arg(name: string, def: string): string {
  const i = process.argv.indexOf(`--${name}`);
  return i >= 0 ? process.argv[i + 1]! : def;
}
// Brains are separated by spaces or "|" (commas belong to overrides).
const brains = arg("brains", "clone").split(/[ |]+/).filter(Boolean);
const mode = arg("mode", "field");
const matches = Number(arg("matches", "100"));
const workers = Number(arg("workers", "1"));
const seed = Number(arg("seed", "7"));
const k = Number(arg("k", "3"));
const seats = Number(arg("seats", "6"));
const games = Number(arg("games", "3"));
const rivalName = arg("rival", "GodOfThudner");
const perRival = process.argv.includes("--per-rival");
// Robustness: scale the rivals' bid noise, and raise their bids by shiftAdd.
const sdScale = Number(arg("sd-scale", "1"));
const shiftAdd = Number(arg("shift-add", "0"));
const RIVALS = path.resolve(arg("rivals-file", path.join(here, "..", "..", "analysis", "imit_try.json")));

interface Tally {
  n: number; place: number; place2: number; first: number; last: number;
  score: number; won: number; paid: number; coins: number; ms: number; errors: number;
  series?: number; seriesWins?: number;
}
type Summary = Record<string, Tally>;
const blank = (): Tally => ({ n: 0, place: 0, place2: 0, first: 0, last: 0, score: 0, won: 0, paid: 0, coins: 0, ms: 0, errors: 0 });

function loadRivals(): Seat[] {
  const file = JSON.parse(fs.readFileSync(RIVALS, "utf8")) as { rivals: any[] };
  return file.rivals.map((r): Seat => {
    const model: RivalModel = {
      slope: r.bid.slope, intercept: r.bid.intercept, sd: r.bid.sd * sdScale,
      minLen: r.minLen ?? 7, shift: (r.shift ?? 0) - shiftAdd, coef: r.coef, cashBal: r.cashBal,
    };
    return { kind: "rival", name: `rival:${r.name}`, ...model };
  });
}

function shuffled<T>(items: T[], rng: () => number): T[] {
  const out = items.slice();
  for (let i = out.length - 1; i >= 1; i--) {
    const j = Math.floor(rng() * (i + 1));
    [out[i], out[j]] = [out[j]!, out[i]!];
  }
  return out;
}

function keyOf(name: string): string {
  if (name.startsWith("rival:")) return perRival ? name : "rivals (avg)";
  return name;
}

function runWorker(first: number, count: number): Summary {
  const words = fs.readFileSync(WORDLIST, "utf8").split("\n");
  const lex = new Lexicon(words);
  const simWords = new SimWords(words);
  const rivals = loadRivals();
  let cur = makeRng(1);
  const rngRef = () => cur();
  const made = new Map<string, Brain>();
  const brain = (name: string): Brain => {
    let b = made.get(name);
    if (!b) made.set(name, (b = createCandidate(name, lex, rngRef)));
    return b;
  };
  const summary: Summary = {};
  const add = (key: string, place: number, isLast: boolean, score: number, won: number, paid: number, ms: number) => {
    const t = (summary[key] ??= blank());
    t.n++; t.place += place; t.place2 += place * place; t.first += place === 1 ? 1 : 0;
    t.last += isLast ? 1 : 0; t.score += score; t.won += won; t.paid += paid;
    t.coins += 100 + score - paid; t.ms += ms;
  };
  const record = (seats: Seat[]) => {
    const r = playMatch(seats, lex, simWords, cur);
    const minScore = Math.min(...r.scores);
    r.names.forEach((name, i) => {
      add(keyOf(name), r.places[i]!, r.scores[i] === minScore, r.scores[i]!, r.won[i]!, r.paid[i]!, r.ms[i]!);
    });
    return r;
  };
  for (let m = first; m < first + count; m++) {
    if (mode === "field") {
      for (const b of brains) {
        cur = makeRng(seed * 1_000_003 + m);
        const opp = shuffled(rivals, cur).slice(0, seats - 1);
        record([{ kind: "brain", brain: brain(b), name: b }, ...opp]);
      }
    } else if (mode === "mixed") {
      cur = makeRng(seed * 1_000_003 + m);
      const pick = shuffled(brains, cur).slice(0, Math.min(k, brains.length));
      const opp = shuffled(rivals, cur).slice(0, 6 - pick.length);
      record(shuffled([...pick.map((b): Seat => ({ kind: "brain", brain: brain(b), name: b })), ...opp], cur));
    } else if (mode === "final") {
      const r0 = rivals.find((s) => s.name === `rival:${rivalName}`)!;
      for (const b of brains) {
        let mine = 0, theirs = 0;
        for (let g = 0; g < games; g++) {
          cur = makeRng(seed * 1_000_003 + m * 17 + g);
          const r = record([{ kind: "brain", brain: brain(b), name: b }, r0]);
          mine += r.scores[0]!; theirs += r.scores[1]!;
        }
        const t = (summary[b] ??= blank());
        t.series = (t.series ?? 0) + 1;
        t.seriesWins = (t.seriesWins ?? 0) + (mine > theirs ? 1 : mine === theirs ? 0.5 : 0);
      }
    } else if (mode === "rivals") {
      cur = makeRng(seed * 1_000_003 + m);
      record(shuffled(rivals, cur).slice(0, 6));
    } else throw new Error(`unknown mode ${mode}`);
  }
  for (const [name, b] of made) {
    if (b instanceof SafeBrain && summary[name]) summary[name]!.errors += b.errors;
  }
  return summary;
}

function merge(into: Summary, from: Summary) {
  for (const [key, t] of Object.entries(from)) {
    const d = (into[key] ??= blank());
    for (const f of Object.keys(t) as (keyof Tally)[]) (d as any)[f] = ((d as any)[f] ?? 0) + (t[f] ?? 0);
  }
}

function report(s: Summary) {
  console.log(`mode=${mode} matches=${matches} seed=${seed} brains=${brains.join(",")}`);
  const rows = Object.entries(s).sort((a, b) => a[1].place / a[1].n - b[1].place / b[1].n);
  console.log("name              n   place  ±SE   first%  last%  tiles  price  score  coinsEnd  ms/m  err" + (mode === "final" ? "  series%" : ""));
  for (const [key, t] of rows) {
    const mean = t.place / t.n;
    const se = Math.sqrt(Math.max(0, t.place2 / t.n - mean * mean) / t.n);
    console.log(
      `${key.padEnd(16)} ${String(t.n).padStart(4)}  ${mean.toFixed(2)}  ${se.toFixed(2)}  ` +
      `${((100 * t.first) / t.n).toFixed(1).padStart(5)}  ${((100 * t.last) / t.n).toFixed(1).padStart(5)}  ` +
      `${(t.won / t.n).toFixed(1).padStart(5)}  ${(t.won ? t.paid / t.won : 0).toFixed(1).padStart(5)}  ` +
      `${(t.score / t.n).toFixed(1).padStart(5)}  ${(t.coins / t.n).toFixed(1).padStart(7)}  ` +
      `${(t.ms / t.n).toFixed(0).padStart(5)}  ${t.errors}` +
      (t.series ? `  ${((100 * (t.seriesWins ?? 0)) / t.series).toFixed(1)}` : ""),
    );
  }
  const out = arg("out", "");
  if (out) fs.writeFileSync(out, JSON.stringify({ mode, matches, seed, brains, summary: s }, null, 1));
}

function runParallel(total: number, n: number): Promise<Summary> {
  const per = Math.ceil(total / n);
  const jobs: Promise<Summary>[] = [];
  for (let w = 0; w < n; w++) {
    const first = w * per;
    const count = Math.min(per, total - first);
    if (count <= 0) break;
    jobs.push(new Promise((resolve, reject) => {
      const child = spawn(process.execPath, [...process.execArgv, fileURLToPath(import.meta.url),
        ...process.argv.slice(2), "--worker", "--first", String(first), "--count", String(count)],
        { stdio: ["ignore", "pipe", "inherit"] });
      let out = "";
      child.stdout.on("data", (d) => (out += d));
      child.on("exit", (code) => {
        if (code !== 0) return reject(new Error(`worker exited ${code}`));
        const line = out.trim().split("\n").pop() ?? "";
        try { resolve(JSON.parse(line)); } catch { reject(new Error(`bad worker output: ${out.slice(-300)}`)); }
      });
    }));
  }
  return Promise.all(jobs).then((all) => { const t: Summary = {}; for (const s of all) merge(t, s); return t; });
}

for (const b of brains) parseCandidate(b);
if (process.argv.includes("--worker")) {
  // Brains may log; keep stdout's last line for the JSON summary.
  const s = runWorker(Number(arg("first", "0")), Number(arg("count", "0")));
  process.stdout.write("\n" + JSON.stringify(s) + "\n");
} else if (workers <= 1) {
  report(runWorker(0, matches));
} else {
  report(await runParallel(matches, workers));
}
