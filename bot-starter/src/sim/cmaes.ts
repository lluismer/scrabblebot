// Tune strategy C's parameters with CMA-ES against simulated matches.
//
//   npx tsx src/sim/cmaes.ts --workers 128 --matches 200 --generations 40
//
// CMA-ES (Hansen's covariance matrix adaptation evolution strategy) keeps a
// Gaussian over the parameter space, samples a population from it, and
// moves and reshapes it toward the better samples. It needs no gradients
// and copes with noisy scores, which is what a match simulator gives.
//
// Each generation evaluates every candidate on the same seeded matches
// (common random numbers, so candidates are compared on equal footing),
// plus the current mean, whose score is the progress line to watch. State
// is saved after every generation, so a job that hits its time limit picks
// up where it stopped when run again with the same --state file.
//
// Options:
//   --objective field | final | mix   what to minimise (default mix):
//       field  average place in the `field` lineup (C + 5 rival stand-ins)
//       final  1 − win rate in `field-final` (C vs one rival, --rival NAME)
//       mix    field place + 2·(1 − final win rate)
//   --matches N       field matches per candidate per generation (default 200)
//   --final-matches N final matches per candidate per generation (default 48)
//   --lambda N        population size (default 4 + 3·ln(dims))
//   --sigma X         initial step size in the unit cube (default 0.25)
//   --samples N       planner rollouts per decision (default 30)
//   --state FILE      where to save and resume (default results/cmaes.json)

import * as fs from "node:fs";
import * as path from "node:path";
import { DEFAULT_PARAMS } from "../engine/planner.js";
import { runParallel } from "./run.js";

// The search space. Each parameter is searched in [lo, hi], mapped to the
// unit interval so one step size suits them all.
const SPACE: { key: keyof typeof DEFAULT_PARAMS; lo: number; hi: number }[] = [
  { key: "earnFactor", lo: 0, hi: 1.5 },
  { key: "lambdaFloor", lo: 0, hi: 1 },
  { key: "playMargin", lo: 0, hi: 20 },
  { key: "winPrior", lo: 0.05, hi: 0.7 },
  { key: "sideShare", lo: 0.1, hi: 1 },
  // Denial weights: one only matters in two-bot games (objective final or
  // mix), the other at bigger tables.
  { key: "denial1", lo: 0, hi: 2 },
  { key: "denialN", lo: 0, hi: 1 },
];

function arg(name: string, def: string): string {
  const i = process.argv.indexOf(`--${name}`);
  return i >= 0 ? process.argv[i + 1]! : def;
}

const WORKERS = Number(arg("workers", "8"));
const MATCHES = Number(arg("matches", "200"));
// Two-bot games run about 15× longer than a table of six (racks of 40+
// tiles), so the final gets its own, smaller count.
const FINAL_MATCHES = Number(arg("final-matches", "48"));
const GENERATIONS = Number(arg("generations", "40"));
const SAMPLES = arg("samples", "30");
const OBJECTIVE = arg("objective", "mix");
const RIVAL = arg("rival", "Bron");
const STATE = arg("state", path.join("results", "cmaes.json"));
const n = SPACE.length;
const LAMBDA = Number(arg("lambda", String(4 + Math.floor(3 * Math.log(n)))));
const SIGMA0 = Number(arg("sigma", "0.25"));

// ---------- small linear algebra ----------

type Mat = number[][];
const eye = (k: number): Mat =>
  Array.from({ length: k }, (_, i) => Array.from({ length: k }, (_, j) => (i === j ? 1 : 0)));
const matVec = (A: Mat, v: number[]) => A.map((row) => row.reduce((s, a, j) => s + a * v[j]!, 0));
const norm = (v: number[]) => Math.sqrt(v.reduce((s, x) => s + x * x, 0));

// Eigen-decomposition of a symmetric matrix by cyclic Jacobi rotations.
// Returns eigenvalues and the matrix whose columns are the eigenvectors.
function eigSym(S: Mat): { values: number[]; vectors: Mat } {
  const k = S.length;
  const A = S.map((r) => r.slice());
  const V = eye(k);
  for (let sweep = 0; sweep < 100; sweep++) {
    let off = 0;
    for (let p = 0; p < k; p++) for (let q = p + 1; q < k; q++) off += A[p]![q]! ** 2;
    if (off < 1e-22) break;
    for (let p = 0; p < k; p++) {
      for (let q = p + 1; q < k; q++) {
        const apq = A[p]![q]!;
        if (Math.abs(apq) < 1e-300) continue;
        const theta = (A[q]![q]! - A[p]![p]!) / (2 * apq);
        const t = Math.sign(theta || 1) / (Math.abs(theta) + Math.sqrt(theta * theta + 1));
        const c = 1 / Math.sqrt(t * t + 1);
        const s = t * c;
        for (let r = 0; r < k; r++) {
          const arp = A[r]![p]!;
          const arq = A[r]![q]!;
          A[r]![p] = c * arp - s * arq;
          A[r]![q] = s * arp + c * arq;
        }
        for (let r = 0; r < k; r++) {
          const apr = A[p]![r]!;
          const aqr = A[q]![r]!;
          A[p]![r] = c * apr - s * aqr;
          A[q]![r] = s * apr + c * aqr;
        }
        for (let r = 0; r < k; r++) {
          const vrp = V[r]![p]!;
          const vrq = V[r]![q]!;
          V[r]![p] = c * vrp - s * vrq;
          V[r]![q] = s * vrp + c * vrq;
        }
      }
    }
  }
  return { values: A.map((r, i) => r[i]!), vectors: V };
}

function gaussian(): number {
  const u = Math.max(Math.random(), 1e-12);
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * Math.random());
}

// Fold a coordinate back into [0, 1] by reflecting at the walls.
function reflect(x: number): number {
  let v = ((x % 2) + 2) % 2;
  if (v > 1) v = 2 - v;
  return v;
}

// ---------- parameters ----------

const encode = (p: Record<string, number>) =>
  SPACE.map((s) => (p[s.key]! - s.lo) / (s.hi - s.lo));
const decode = (x: number[]) =>
  Object.fromEntries(SPACE.map((s, i) => [s.key, +(s.lo + x[i]! * (s.hi - s.lo)).toFixed(4)]));

// ---------- fitness ----------

type Summary = Record<string, { n: number; wins: number; place: number; score: number }>;

async function score(x: number[], gen: number, workers: number): Promise<{ f: number; detail: string }> {
  const flags = Object.entries(decode(x)).flatMap(([k, v]) => ["--param", `${k}=${v}`]);
  const common = ["--samples", SAMPLES, "--seed", String(1000 + gen), ...flags];
  const parts: string[] = [];
  let f = 0;
  if (OBJECTIVE === "field" || OBJECTIVE === "mix") {
    const s: Summary = await runParallel(["--lineup", "field", ...common], MATCHES, workers);
    const c = s["strategy-c"]!;
    const place = c.place / c.n;
    f += place;
    parts.push(`place ${place.toFixed(3)} score ${(c.score / c.n).toFixed(1)}`);
  }
  if (OBJECTIVE === "final" || OBJECTIVE === "mix") {
    const s: Summary = await runParallel(
      ["--lineup", "field-final", "--rival", RIVAL, ...common],
      FINAL_MATCHES,
      workers,
    );
    const c = s["strategy-c"]!;
    const win = c.wins / c.n;
    f += (OBJECTIVE === "mix" ? 2 : 1) * (1 - win);
    parts.push(`final win ${(100 * win).toFixed(1)}%`);
  }
  return { f, detail: parts.join(", ") };
}

// ---------- CMA-ES state ----------

interface State {
  gen: number;
  mean: number[];
  sigma: number;
  C: Mat;
  pc: number[];
  ps: number[];
  history: { gen: number; meanFitness: number; detail: string; params: Record<string, number>; sigma: number }[];
  best?: { fitness: number; params: Record<string, number>; gen: number };
}

function fresh(): State {
  return {
    gen: 0,
    mean: encode(DEFAULT_PARAMS as unknown as Record<string, number>),
    sigma: SIGMA0,
    C: eye(n),
    pc: Array(n).fill(0),
    ps: Array(n).fill(0),
    history: [],
  };
}

function save(st: State) {
  fs.mkdirSync(path.dirname(STATE), { recursive: true });
  fs.writeFileSync(STATE + ".tmp", JSON.stringify(st, null, 1));
  fs.renameSync(STATE + ".tmp", STATE);
}

// Strategy parameters (Hansen 2016, "The CMA Evolution Strategy: A Tutorial").
const mu = Math.floor(LAMBDA / 2);
const rawW = Array.from({ length: mu }, (_, i) => Math.log(mu + 0.5) - Math.log(i + 1));
const wSum = rawW.reduce((a, b) => a + b, 0);
const w = rawW.map((v) => v / wSum);
const mueff = 1 / w.reduce((s, v) => s + v * v, 0);
const cc = (4 + mueff / n) / (n + 4 + (2 * mueff) / n);
const cs = (mueff + 2) / (n + mueff + 5);
const c1 = 2 / ((n + 1.3) ** 2 + mueff);
const cmu = Math.min(1 - c1, (2 * (mueff - 2 + 1 / mueff)) / ((n + 2) ** 2 + mueff));
const damps = 1 + 2 * Math.max(0, Math.sqrt((mueff - 1) / (n + 1)) - 1) + cs;
const chiN = Math.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n * n));

async function main() {
  const st: State = fs.existsSync(STATE) ? JSON.parse(fs.readFileSync(STATE, "utf8")) : fresh();
  console.log(
    `CMA-ES: ${n} parameters, population ${LAMBDA}, ${MATCHES} matches per candidate, ` +
      `objective ${OBJECTIVE}, ${WORKERS} workers, starting at generation ${st.gen}`,
  );
  while (st.gen < GENERATIONS) {
    const t0 = Date.now();
    const { values, vectors: B } = eigSym(st.C);
    const D = values.map((v) => Math.sqrt(Math.max(v, 1e-20)));

    // Sample, reflect into the box, and recover each step y = (x − m)/σ.
    const xs: number[][] = [];
    const ys: number[][] = [];
    for (let k = 0; k < LAMBDA; k++) {
      const z = Array.from({ length: n }, gaussian);
      const y = matVec(B, z.map((v, i) => v * D[i]!));
      const x = st.mean.map((m, i) => reflect(m + st.sigma * y[i]!));
      xs.push(x);
      ys.push(x.map((v, i) => (v - st.mean[i]!) / st.sigma));
    }

    // Evaluate the population and the current mean side by side.
    const per = Math.max(1, Math.floor(WORKERS / (LAMBDA + 1)));
    const evals = await Promise.all([...xs, st.mean].map((x) => score(x, st.gen, per)));
    const meanEval = evals.pop()!;
    const order = evals.map((e, i) => i).sort((a, b) => evals[a]!.f - evals[b]!.f);

    // Recombine the best mu into the new mean.
    const old = st.mean;
    const yw = Array(n).fill(0);
    order.slice(0, mu).forEach((idx, r) => {
      for (let i = 0; i < n; i++) yw[i] += w[r]! * ys[idx]![i]!;
    });
    st.mean = old.map((m, i) => Math.min(1, Math.max(0, m + st.sigma * yw[i]!)));

    // Evolution paths.
    const invSqrtC_yw = matVec(B, matVec(B[0]!.map((_, j) => B.map((r) => r[j]!)), yw).map((v, i) => v / D[i]!));
    st.ps = st.ps.map((p, i) => (1 - cs) * p + Math.sqrt(cs * (2 - cs) * mueff) * invSqrtC_yw[i]!);
    const hsig =
      norm(st.ps) / Math.sqrt(1 - (1 - cs) ** (2 * (st.gen + 1))) / chiN < 1.4 + 2 / (n + 1) ? 1 : 0;
    st.pc = st.pc.map((p, i) => (1 - cc) * p + hsig * Math.sqrt(cc * (2 - cc) * mueff) * yw[i]!);

    // Covariance: rank-one (pc) plus rank-mu (the selected steps).
    const C2 = st.C.map((row, i) =>
      row.map((cij, j) => {
        let rankMu = 0;
        order.slice(0, mu).forEach((idx, r) => (rankMu += w[r]! * ys[idx]![i]! * ys[idx]![j]!));
        return (
          (1 - c1 - cmu) * cij +
          c1 * (st.pc[i]! * st.pc[j]! + (1 - hsig) * cc * (2 - cc) * cij) +
          cmu * rankMu
        );
      }),
    );
    st.C = C2.map((row, i) => row.map((_, j) => (C2[i]![j]! + C2[j]![i]!) / 2));
    st.sigma *= Math.exp((cs / damps) * (norm(st.ps) / chiN - 1));

    const bestIdx = order[0]!;
    const bestNow = { fitness: evals[bestIdx]!.f, params: decode(xs[bestIdx]!), gen: st.gen };
    st.history.push({
      gen: st.gen,
      meanFitness: meanEval.f,
      detail: meanEval.detail,
      params: decode(old),
      sigma: st.sigma,
    });
    // "Best" tracks the mean, not one lucky sample: noisy scores make the
    // single best candidate an overestimate.
    if (!st.best || meanEval.f < st.best.fitness) {
      st.best = { fitness: meanEval.f, params: decode(old), gen: st.gen };
    }
    st.gen++;
    save(st);
    console.log(
      `gen ${st.gen - 1}: mean ${meanEval.f.toFixed(3)} (${meanEval.detail}) ` +
        `best sample ${bestNow.fitness.toFixed(3)} σ ${st.sigma.toFixed(3)} ` +
        `${((Date.now() - t0) / 1000).toFixed(0)}s  mean params ${JSON.stringify(decode(old))}`,
    );
  }
  console.log(`done. best mean so far (gen ${st.best?.gen}): ${JSON.stringify(st.best)}`);
  console.log(`current mean: ${JSON.stringify(decode(st.mean))}`);
}

await main();
