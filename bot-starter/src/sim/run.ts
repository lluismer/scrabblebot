// Play many offline matches and report how each seat finishes.
//
//   npx tsx src/sim/run.ts --lineup field --matches 200 --workers 8
//
// Against sims (each fills the remaining seats of 6 with random sims):
//   c        strategy C alone against sims
//   base     the starter's strategy alone against sims
//   c-base   strategy C and the starter at one table
//   c-c      two copies of strategy C
//   c-v      strategy C against a variant (--vparam key=value, repeatable)
// Tournament stand-ins, with no sims (tournament tables have none):
//   table    a variant and 5 default copies: a Swiss table. Compare
//            "place" (Swiss points = 6 − place) against 3.5, the average.
//   final    a variant against one default: the two-bot final.
// Against the real field (stand-ins fitted from hpc/rivals.json):
//   field        strategy C and 5 rivals drawn at random each match
//   field-final  strategy C against one rival (--rival NAME, default Bron)
//   rivals       6 rivals and no strategy C, to calibrate the stand-ins
//
// Options: --samples N (rollouts per decision), --seed S, --first-price,
//          --param key=value (override a planner parameter for strategy C,
//          repeatable), --vparam key=value (the variant's overrides),
//          --shift X (rival bid shift, see RivalModel), --worker --first N
//          --count N (print one JSON summary, for cluster jobs).

import { spawn } from "node:child_process";
import * as fs from "node:fs";
import * as path from "node:path";
import { fileURLToPath } from "node:url";
import { Baseline, StrategyC, type Brain } from "../engine/brains.js";
import { Lexicon } from "../engine/lexicon.js";
import { makeRng, DEFAULT_PARAMS, type Params } from "../engine/planner.js";
import { playMatch, SIM_POOL, SimWords, type RivalModel, type Seat } from "./engine.js";

const here = path.dirname(fileURLToPath(import.meta.url));
const WORDLIST = path.join(here, "..", "..", "..", "spacetimedb", "wordlist.txt");
const RIVALS = path.join(here, "..", "..", "hpc", "rivals.json");

// Calibrated with `--lineup rivals`: with this shift, stand-ins at a table
// of six win and pay about what the real bots do (about 12–13 tiles a
// match at about 13 coins each).
export const DEFAULT_RIVAL_SHIFT = 0;

function arg(name: string, def: string): string {
  const i = process.argv.indexOf(`--${name}`);
  return i >= 0 ? process.argv[i + 1]! : def;
}
function args(name: string): string[] {
  const out: string[] = [];
  process.argv.forEach((a, i) => {
    if (a === `--${name}`) out.push(process.argv[i + 1]!);
  });
  return out;
}

const lineup = arg("lineup", "c");
const matches = Number(arg("matches", "50"));
const workers = Number(arg("workers", "1"));
const seed = Number(arg("seed", "1"));
const samples = Number(arg("samples", "30"));
const shift = Number(arg("shift", String(DEFAULT_RIVAL_SHIFT)));
const rivalName = arg("rival", "Bron");
// Rivals play a word once it has this many letters (0 = each one's own
// average). 7 brings the stand-ins closest to the real bots' scores
// (about 67 a match against 90-100 observed: they bid by letter value
// alone, so they buy tiles that fit their racks less well than real bots).
const rivalMinLen = Number(arg("rival-minlen", "7"));
const kind = process.argv.includes("--first-price") ? "FirstPrice" : "Vickrey";

interface Tally {
  n: number;
  wins: number;
  place: number;
  score: number;
  ms: number;
  won: number;
  paid: number;
}
type Summary = Record<string, Tally>;

function params(flag = "param"): Params {
  const p: Params = {
    ...DEFAULT_PARAMS,
    bidTimeMs: 0,
    playTimeMs: 0,
    minSamples: samples,
    maxSamples: samples,
  };
  for (const kv of args(flag)) {
    const [k, v] = kv.split("=");
    if (!(k! in DEFAULT_PARAMS)) throw new Error(`unknown parameter ${k}`);
    (p as unknown as Record<string, number>)[k!] = Number(v);
  }
  return p;
}

interface RivalEntry {
  name: string;
  bid: { slope: number; intercept: number; sd: number };
  avgWordLength: number;
}

function loadRivals(): Seat[] {
  const file = JSON.parse(fs.readFileSync(RIVALS, "utf8")) as { rivals: RivalEntry[] };
  return file.rivals.map((r): Seat => {
    const model: RivalModel = {
      slope: r.bid.slope,
      intercept: r.bid.intercept,
      sd: r.bid.sd,
      minLen: rivalMinLen || Math.max(2, Math.round(r.avgWordLength)),
      shift,
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

function brainSeat(brain: Brain): Seat {
  return { kind: "brain", brain, name: brain.name };
}

function seatsFor(lex: Lexicon, rng: () => number, rivals: Seat[]): Seat[] {
  const c = () => new StrategyC(lex, params(), rng);
  const variant = () => {
    const v = new StrategyC(lex, { ...params(), ...params("vparam") }, rng);
    v.name = "variant";
    return v;
  };
  const sims = (n: number) => shuffled(SIM_POOL, rng).slice(0, n);
  switch (lineup) {
    case "c":
      return [brainSeat(c()), ...sims(5)];
    case "base":
      return [brainSeat(new Baseline(lex)), ...sims(5)];
    case "c-base":
      return [brainSeat(c()), brainSeat(new Baseline(lex)), ...sims(4)];
    case "c-c": {
      const b = c();
      b.name = "strategy-c#2";
      return [brainSeat(c()), brainSeat(b), ...sims(4)];
    }
    case "c-v":
      return [brainSeat(c()), brainSeat(variant()), ...sims(4)];
    case "table":
    case "final": {
      const rest = Array.from({ length: lineup === "table" ? 5 : 1 }, () => {
        const d = c();
        d.name = "default";
        return brainSeat(d);
      });
      return [brainSeat(variant()), ...rest];
    }
    case "field":
      return [brainSeat(c()), ...shuffled(rivals, rng).slice(0, 5)];
    case "field-final": {
      const r = rivals.find((s) => s.name === `rival:${rivalName}`);
      if (!r) throw new Error(`no rival named ${rivalName} in ${RIVALS}`);
      return [brainSeat(c()), r];
    }
    case "rivals":
      return shuffled(rivals, rng).slice(0, 6);
    default:
      throw new Error(`unknown lineup ${lineup}`);
  }
}

// Tally key: sims and rivals are pooled; brains keep their own names.
function keyOf(name: string): string {
  if (name.endsWith("-sim")) return "sims (avg)";
  if (name.startsWith("rival:")) return "rivals (avg)";
  return name;
}

function runWorker(first: number, count: number): Summary {
  const words = fs.readFileSync(WORDLIST, "utf8").split("\n");
  const lex = new Lexicon(words);
  const simWords = new SimWords(words);
  const rivals = fs.existsSync(RIVALS) ? loadRivals() : [];
  const summary: Summary = {};
  for (let m = first; m < first + count; m++) {
    const rng = makeRng(seed * 1_000_003 + m);
    const r = playMatch(seatsFor(lex, rng, rivals), lex, simWords, rng, kind);
    r.names.forEach((name, i) => {
      const t = (summary[keyOf(name)] ??= {
        n: 0, wins: 0, place: 0, score: 0, ms: 0, won: 0, paid: 0,
      });
      t.n++;
      t.wins += r.places[i] === 1 ? 1 : 0;
      t.place += r.places[i]!;
      t.score += r.scores[i]!;
      t.ms += r.ms[i]!;
      t.won += r.won[i]!;
      t.paid += r.paid[i]!;
    });
  }
  return summary;
}

export function merge(into: Summary, from: Summary) {
  for (const [k, t] of Object.entries(from)) {
    const d = (into[k] ??= { n: 0, wins: 0, place: 0, score: 0, ms: 0, won: 0, paid: 0 });
    d.n += t.n;
    d.wins += t.wins;
    d.place += t.place;
    d.score += t.score;
    d.ms += t.ms;
    d.won += t.won ?? 0;
    d.paid += t.paid ?? 0;
  }
}

function report(s: Summary) {
  console.log(`lineup=${lineup} matches=${matches} samples=${samples} ${kind}`);
  for (const [k, t] of Object.entries(s)) {
    console.log(
      `${k.padEnd(14)} win ${((100 * t.wins) / t.n).toFixed(1).padStart(5)}%` +
        `  place ${(t.place / t.n).toFixed(2)}` +
        `  score ${(t.score / t.n).toFixed(1).padStart(6)}` +
        `  tiles ${(t.won / t.n).toFixed(1).padStart(4)}` +
        `  paid/tile ${t.won ? (t.paid / t.won).toFixed(1) : "-"}` +
        `  ms/match ${(t.ms / t.n).toFixed(0)}`,
    );
  }
}

// Run [first, first+count) across `n` child processes; resolves to the merged summary.
export function runParallel(argv: string[], total: number, n: number): Promise<Summary> {
  const per = Math.ceil(total / n);
  const jobs: Promise<Summary>[] = [];
  for (let w = 0; w < n; w++) {
    const first = w * per;
    const count = Math.min(per, total - first);
    if (count <= 0) break;
    jobs.push(
      new Promise((resolve, reject) => {
        const child = spawn(
          process.execPath,
          [
            ...process.execArgv,
            fileURLToPath(import.meta.url),
            ...argv,
            "--worker",
            "--first",
            String(first),
            "--count",
            String(count),
          ],
          { stdio: ["ignore", "pipe", "inherit"] },
        );
        let out = "";
        child.stdout.on("data", (d) => (out += d));
        child.on("exit", (code) =>
          code === 0 ? resolve(JSON.parse(out)) : reject(new Error(`worker exited ${code}`)),
        );
      }),
    );
  }
  return Promise.all(jobs).then((all) => {
    const total: Summary = {};
    for (const s of all) merge(total, s);
    return total;
  });
}

const isMain = process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url);
if (isMain) {
  if (process.argv.includes("--worker")) {
    const first = Number(arg("first", "0"));
    const count = Number(arg("count", "0"));
    process.stdout.write(JSON.stringify(runWorker(first, count)));
  } else if (workers <= 1) {
    report(runWorker(0, matches));
  } else {
    report(await runParallel(process.argv.slice(2), matches, workers));
  }
}
