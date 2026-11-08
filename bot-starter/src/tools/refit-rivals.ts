// Refit every real bot's bid model from public history, using what each
// auction reveals about ALL bidders, not just the winner:
//
//   winner        bid = top_bid               (exact)
//   one loser     bid = paid                  (exact; which loser is unknown)
//   other losers  bid <= paid                 (censored)
//
// Fitting bid ~ slope*value + intercept to winners' bids alone (the old
// fit) keeps only the top order statistic, which overstates a bot's
// typical bid. This is EM for a censored (Tobit) regression:
// E-step: who was second, and E[bid | bid <= paid] for the rest;
// M-step: weighted least squares per bot.
//
// Only matches with no sim seats and 5+ real bots, the tournament's
// conditions. The last 5 auctions are left out (end-of-match dumping).
//
//   npx tsx src/tools/refit-rivals.ts [data/matches.jsonl] [outDir]
// writes <outDir>/rivals-refit.json (bidModel per bot, the format
// seedRivals reads) and <outDir>/market.json (the empirical max-rival-bid
// distribution per auction index and letter value, for market replay).

import * as fs from "node:fs";
import * as readline from "node:readline";
import { VALUES } from "../engine/letters.js";

const FILE = process.argv[2] ?? "data/matches.jsonl";
const OUT = process.argv[3] ?? "data";
const ITERS = 12;
const PRIOR = { slope: 3.5, intercept: 0, sd: 7 }; // global fit, real-only
const PRIOR_WEIGHT = 20;
const CUT = 63; // fit on auctions 1..63

interface Obs { v: number; top: number; paid: number; winner: number; losers: number[] }
export interface M { slope: number; intercept: number; sd: number }
const val = (c: string) => VALUES[c.charCodeAt(0) - 65]!;

const phi = (z: number) => Math.exp(-0.5 * z * z) / Math.sqrt(2 * Math.PI);
function Phi(z: number): number {
  const x = Math.abs(z) / Math.SQRT2, t = 1 / (1 + 0.3275911 * x);
  const y = 1 - ((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-x * x);
  return z >= 0 ? 0.5 * (1 + y) : 0.5 * (1 - y);
}

const names: string[] = [];
const idOf = new Map<string, number>();
type St = { n: number; place: number; score: number; tiles: number; paid: number; last: number; words: number; wl: number };
const stats = new Map<number, St>();
const train: Obs[] = [];
const test: Obs[] = [];
// market[idx][v] = histogram of the max rival bid faced by a random seat
const market: Record<number, Record<number, number[]>> = {};

const rl = readline.createInterface({ input: fs.createReadStream(FILE) });
let matchNo = 0;
for await (const line of rl) {
  if (!line.trim()) continue;
  const m = JSON.parse(line);
  const real = m.seats.filter((s: any) => !s.sim);
  if (real.length !== m.seats.length || real.length < 5) continue;
  const ids: number[] = m.seats.map((s: any) => {
    let i = idOf.get(s.name);
    if (i === undefined) { i = names.length; names.push(s.name); idOf.set(s.name, i); }
    return i;
  });
  const seatOf = new Map<string, number>(m.seats.map((s: any, k: number) => [s.botId, ids[k]!]));
  const won: Record<string, number> = {}, paidBy: Record<string, number> = {};
  const into = matchNo++ % 5 === 0 ? test : train;
  m.auctions.forEach((a: any, i: number) => {
    const v = val(a.letter);
    if (a.winner) { won[a.winner] = (won[a.winner] ?? 0) + 1; paidBy[a.winner] = (paidBy[a.winner] ?? 0) + a.paid; }
    const h = ((market[i + 1] ??= {})[v] ??= new Array(101).fill(0));
    if (!a.winner || a.topBid === null) { h[0] += 1; return; }
    h[Math.min(100, a.topBid)] += (m.seats.length - 1) / m.seats.length;
    h[Math.min(100, a.paid)] += 1 / m.seats.length;
    if (i >= CUT) return;
    const w = seatOf.get(a.winner)!;
    into.push({ v, top: a.topBid, paid: a.paid, winner: w, losers: ids.filter((x) => x !== w) });
  });
  const wordsBy: Record<string, string[]> = {};
  for (const w of m.words) (wordsBy[w.botId] ??= []).push(w.word);
  m.seats.forEach((s: any, k: number) => {
    const st = stats.get(ids[k]!) ?? { n: 0, place: 0, score: 0, tiles: 0, paid: 0, last: 0, words: 0, wl: 0 };
    stats.set(ids[k]!, st);
    st.n++; st.place += s.place; st.score += s.score; st.tiles += won[s.botId] ?? 0; st.paid += paidBy[s.botId] ?? 0;
    st.last += s.place === m.seats.length ? 1 : 0;
    const ws = wordsBy[s.botId] ?? []; st.words += ws.length; st.wl += ws.reduce((x, y) => x + y.length, 0);
  });
}

// ---- EM ----
let models: M[] = names.map(() => ({ ...PRIOR }));
for (let it = 0; it < ITERS; it++) {
  // per bot: sum w, w*v, w*v^2, w*E[x], w*v*E[x], w*E[x^2]
  const S = names.map(() => new Float64Array(6));
  const add = (b: number, w: number, v: number, ex: number, ex2: number) => {
    const s = S[b]!; s[0]! += w; s[1]! += w * v; s[2]! += w * v * v; s[3]! += w * ex; s[4]! += w * v * ex; s[5]! += w * ex2;
  };
  for (const o of train) {
    add(o.winner, 1, o.v, o.top, o.top * o.top);
    const u = o.paid + 0.5; // continuous bound for "bid <= paid"
    const resp = o.losers.map((j) => {
      const m = models[j]!, mu = m.slope * o.v + m.intercept;
      return phi((o.paid - mu) / m.sd) / m.sd / Math.max(1e-12, Phi((u - mu) / m.sd));
    });
    const tot = resp.reduce((a, b) => a + b, 0) || 1;
    o.losers.forEach((j, k) => {
      const r = resp[k]! / tot;
      const m = models[j]!, mu = m.slope * o.v + m.intercept, al = (u - mu) / m.sd;
      const lam = phi(al) / Math.max(1e-12, Phi(al));
      const ex = mu - m.sd * lam;
      const vx = m.sd * m.sd * Math.max(0.01, 1 - al * lam - lam * lam);
      add(j, r, o.v, o.paid, o.paid * o.paid);
      add(j, 1 - r, o.v, ex, vx + ex * ex);
    });
  }
  models = S.map((s) => {
    // shrink toward the prior with pseudo-observations at v = 1 and v = 8
    const p = Float64Array.from(s);
    for (const v of [1, 8]) {
      const y = PRIOR.slope * v + PRIOR.intercept, w = PRIOR_WEIGHT / 2;
      p[0]! += w; p[1]! += w * v; p[2]! += w * v * v; p[3]! += w * y; p[4]! += w * v * y; p[5]! += w * (y * y + PRIOR.sd * PRIOR.sd);
    }
    const mx = p[1]! / p[0]!, my = p[3]! / p[0]!, vxx = p[2]! / p[0]! - mx * mx;
    const slope = vxx > 1e-9 ? (p[4]! / p[0]! - mx * my) / vxx : PRIOR.slope;
    const intercept = my - slope * mx;
    const a = slope, c = intercept;
    const sse = p[5]! - 2 * a * p[4]! - 2 * c * p[3]! + a * a * p[2]! + 2 * a * c * p[1]! + c * c * p[0]!;
    return { slope, intercept, sd: Math.max(1, Math.sqrt(Math.max(0, sse) / p[0]!)) };
  });
}

// ---- validation on held-out matches: log-lik of the max of the other bids
const below = (m: M, v: number, b: number) =>
  b <= 1 ? 0.05 : 0.05 + 0.95 * Phi((b - 0.5 - (m.slope * v + m.intercept)) / m.sd);
const oldFile = `${OUT}/rivals.json`;
const old = fs.existsSync(oldFile) ? (JSON.parse(fs.readFileSync(oldFile, "utf8")) as any[]) : [];
const oldBy = new Map(old.map((r) => [r.name, r.bidModel as M]));
const OLDPRIOR = { slope: 2.9, intercept: 11.5, sd: 6.3 };
function ll(get: (b: number) => M, shift = 0) {
  let L = 0, n = 0;
  for (const o of test) {
    const seats = [o.winner, ...o.losers];
    const us = seats[n % seats.length]!;
    const mx = us === o.winner ? o.paid : o.top; // the winner faced `paid`
    const others = seats.filter((x) => x !== us);
    const F = (b: number) =>
      others.reduce((p, j) => { const m = get(j); return p * below({ ...m, intercept: m.intercept - shift }, o.v, b); }, 1);
    L += Math.log(Math.max(1e-9, F(mx + 1) - F(mx))); n++;
  }
  return L / n;
}
const oldGet = (j: number) => oldBy.get(names[j]!) ?? OLDPRIOR;
const newGet = (j: number) => models[j]!;
console.log(`train ${train.length} auctions, test ${test.length}`);
console.log(
  `held-out log-lik per auction: old fit ${ll(oldGet).toFixed(3)}  old-8.5 ${ll(oldGet, 8.5).toFixed(3)}  refit ${ll(newGet).toFixed(3)}`,
);

// ---- residuals by letter value: E[top] and E[paid], predicted vs observed
type R = { n: number; top: number; paid: number; tOld: number; pOld: number; tNew: number; pNew: number };
const byV: Record<number, R> = {};
let seed = 1;
const rnd = () => (seed = (seed * 16807) % 2147483647) / 2147483647;
const gauss = () => Math.sqrt(-2 * Math.log(Math.max(rnd(), 1e-12))) * Math.cos(2 * Math.PI * rnd());
const draw = (m: M, v: number) => (rnd() < 0.05 ? 0 : Math.max(1, Math.round(m.slope * v + m.intercept + m.sd * gauss())));
for (const o of test) {
  const r = (byV[o.v] ??= { n: 0, top: 0, paid: 0, tOld: 0, pOld: 0, tNew: 0, pNew: 0 });
  r.n++; r.top += o.top; r.paid += o.paid;
  const seats = [o.winner, ...o.losers];
  const a = seats.map((j) => draw(oldGet(j), o.v)).sort((x, y) => y - x);
  const b = seats.map((j) => draw(newGet(j), o.v)).sort((x, y) => y - x);
  r.tOld += a[0]!; r.pOld += a[1]!; r.tNew += b[0]!; r.pNew += b[1]!;
}
console.log("value        n | top: obs   old  refit | paid: obs   old  refit");
for (const [v, r] of Object.entries(byV).sort((a, b) => +a[0] - +b[0])) {
  const f = (x: number) => (x / r.n).toFixed(1).padStart(6);
  console.log(`v=${v.padEnd(3)} ${String(r.n).padStart(7)} |     ${f(r.top)}${f(r.tOld)}${f(r.tNew)} |      ${f(r.paid)}${f(r.pOld)}${f(r.pNew)}`);
}

// ---- outputs
const rivals = names
  .map((name, j) => {
    const st = stats.get(j)!;
    const m = models[j]!;
    const bid = { slope: +m.slope.toFixed(3), intercept: +m.intercept.toFixed(2), sd: +m.sd.toFixed(2) };
    return {
      name,
      matches: st.n,
      avgPlace: +(st.place / st.n).toFixed(2),
      lastRate: +(st.last / st.n).toFixed(3),
      avgScore: +(st.score / st.n).toFixed(1),
      lettersWonPerMatch: +(st.tiles / st.n).toFixed(1),
      avgPaid: st.tiles ? +(st.paid / st.tiles).toFixed(2) : null,
      avgWordLength: st.words ? +(st.wl / st.words).toFixed(2) : null,
      bidModel: { ...bid, samples: st.tiles },
      bid,
    };
  })
  .sort((a, b) => a.avgPlace - b.avgPlace);
fs.writeFileSync(`${OUT}/rivals-refit.json`, JSON.stringify(rivals, null, 2));
fs.writeFileSync(`${OUT}/market.json`, JSON.stringify(market));
console.log("best real-only bots (50+ matches):");
for (const r of rivals.filter((r) => r.matches >= 50).slice(0, 12)) {
  console.log(
    `  ${r.name.padEnd(18)} n=${String(r.matches).padStart(5)} place ${r.avgPlace} last ${(100 * r.lastRate).toFixed(0)}% ` +
      `score ${r.avgScore} tiles ${r.lettersWonPerMatch} @${r.avgPaid}  bid~${r.bidModel.slope}v+${r.bidModel.intercept} sd${r.bidModel.sd}`,
  );
}
