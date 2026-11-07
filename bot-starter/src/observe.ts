// Watch-only collector: records every finished match on the server and
// profiles the real (non-sim) bots, without ever playing.
//
//   npm run observe
//
// It connects with a fresh anonymous identity: no token, no team, no bot.
// It only reads public tables and calls no reducers, so nothing it does
// shows up as your bot. Output, in ./data:
//
//   matches.jsonl  one line per finished match: seats, scores, every
//                  auction (letter, winner, paid, top bid), every word.
//   rivals.json    per real bot: matches, places, scores, fitted bid
//                  model (bid ≈ slope·value + intercept), word habits.
//
// Leave it running; it appends each match as it ends.

import * as fs from "node:fs";
import * as path from "node:path";
import { connect } from "./tools/connect.js";
import { fit, type BidSample } from "./engine/learning.js";
import { valueOf, wordReward } from "./engine/letters.js";
import type { DbConnection } from "./module_bindings/index.js";

const HOST = process.env.STDB_HOST ?? "https://maincloud.spacetimedb.com";
const DB = process.env.STDB_DB ?? "scrabblebot";
const OUT = path.join(process.cwd(), "data");
const MATCHES = path.join(OUT, "matches.jsonl");
const RIVALS = path.join(OUT, "rivals.json");

fs.mkdirSync(OUT, { recursive: true });
const seen = new Set<string>();
if (fs.existsSync(MATCHES)) {
  for (const line of fs.readFileSync(MATCHES, "utf8").split("\n")) {
    if (line.trim()) seen.add(String(JSON.parse(line).matchId));
  }
}

interface MatchRecord {
  matchId: string;
  endedAt: string | null;
  auctionType: string;
  seats: { botId: string; name: string; sim: boolean; score: number; place: number }[];
  auctions: { letter: string; winner: string | null; paid: number; topBid: number | null }[];
  words: { botId: string; word: string; reward: number; at: string }[];
}

// Rows grouped by match. The history is millions of rows and the SDK only
// finds rows by primary key, so group once and append as rows arrive.
type Row<K extends keyof DbConnection["db"]> = DbConnection["db"][K] extends {
  iter(): Iterable<infer R>;
}
  ? R
  : never;
const byMatch = {
  parts: new Map<bigint, Row<"matchParticipant">[]>(),
  results: new Map<bigint, Row<"auctionResult">[]>(),
  words: new Map<bigint, Row<"wordPlay">[]>(),
  // find() scans whole tables, so keep keyed copies of what we look up.
  topBid: new Map<bigint, bigint>(),
  bots: new Map<bigint, Row<"bot">>(),
};
function push<T extends { matchId: bigint }>(m: Map<bigint, T[]>, row: T) {
  let list = m.get(row.matchId);
  if (!list) m.set(row.matchId, (list = []));
  list.push(row);
}

type MatchRow = Row<"matchState">;

function record(conn: DbConnection, m: MatchRow): MatchRecord | null {
  if (m.status.tag !== "Ended") return null;
  const matchId = m.id;
  // Latest row per bot: participants are updated in place as they score.
  const latest = new Map<bigint, Row<"matchParticipant">>();
  for (const p of byMatch.parts.get(matchId) ?? []) latest.set(p.botId, p);
  const parts = [...latest.values()];
  const scores = parts.map((p) => Number(p.score));
  const seats = parts.map((p, i) => {
    const bot = byMatch.bots.get(p.botId);
    return {
      botId: String(p.botId),
      name: bot?.name ?? "?",
      sim: bot?.isSimulated ?? false,
      score: scores[i]!,
      place: 1 + scores.filter((s) => s > scores[i]!).length,
    };
  });
  const auctions = [...(byMatch.results.get(matchId) ?? [])]
    .sort((a, b) => (a.auctionId < b.auctionId ? -1 : 1))
    .map((r) => {
      const top = byMatch.topBid.get(r.auctionId);
      return {
        letter: r.letter,
        winner: r.winnerBotId === undefined ? null : String(r.winnerBotId),
        paid: Number(r.paid),
        topBid: top !== undefined ? Number(top) : null,
      };
    });
  const words = [...(byMatch.words.get(matchId) ?? [])]
    .sort((a, b) => (a.id < b.id ? -1 : 1))
    .map((w) => ({
      botId: String(w.botId),
      word: w.word,
      reward: Number(w.totalReward),
      at: w.playedAt.toDate().toISOString(),
    }));
  return {
    matchId: String(matchId),
    endedAt: m.endedAt ? m.endedAt.toDate().toISOString() : null,
    auctionType: m.auctionType.tag,
    seats,
    auctions,
    words,
  };
}

interface Rival {
  name: string;
  matches: number;
  wins: number;
  placeSum: number;
  scoreSum: number;
  realOpponentsSum: number;
  bids: BidSample[];
  lettersWon: number;
  paidSum: number;
  words: number;
  wordLenSum: number;
  rewardSum: number;
}

function profile(all: MatchRecord[]) {
  const rivals = new Map<string, Rival>();
  for (const m of all) {
    const real = m.seats.filter((s) => !s.sim);
    for (const s of real) {
      const r =
        rivals.get(s.botId) ??
        ({
          name: s.name, matches: 0, wins: 0, placeSum: 0, scoreSum: 0,
          realOpponentsSum: 0, bids: [], lettersWon: 0, paidSum: 0,
          words: 0, wordLenSum: 0, rewardSum: 0,
        } as Rival);
      rivals.set(s.botId, r);
      r.matches++;
      r.wins += s.place === 1 ? 1 : 0;
      r.placeSum += s.place;
      r.scoreSum += s.score;
      r.realOpponentsSum += real.length - 1;
    }
    for (const a of m.auctions) {
      const r = a.winner ? rivals.get(a.winner) : undefined;
      if (!r) continue;
      r.lettersWon++;
      r.paidSum += a.paid;
      if (a.topBid !== null) r.bids.push({ value: valueOf(a.letter), bid: a.topBid });
    }
    for (const w of m.words) {
      const r = rivals.get(w.botId);
      if (!r) continue;
      r.words++;
      r.wordLenSum += w.word.length;
      r.rewardSum += w.reward ?? wordReward(w.word);
    }
  }
  const out = [...rivals].map(([botId, r]) => {
    const f = fit(r.bids);
    return {
      botId,
      name: r.name,
      matches: r.matches,
      winRate: +(r.wins / r.matches).toFixed(3),
      avgPlace: +(r.placeSum / r.matches).toFixed(2),
      avgScore: +(r.scoreSum / r.matches).toFixed(1),
      avgRealOpponents: +(r.realOpponentsSum / r.matches).toFixed(2),
      lettersWonPerMatch: +(r.lettersWon / r.matches).toFixed(1),
      avgPaid: r.lettersWon ? +(r.paidSum / r.lettersWon).toFixed(2) : null,
      bidModel: {
        slope: +f.slope.toFixed(3),
        intercept: +f.intercept.toFixed(2),
        sd: +f.sd.toFixed(2),
        samples: f.n,
      },
      avgWordLength: r.words ? +(r.wordLenSum / r.words).toFixed(2) : null,
      avgWordReward: r.words ? +(r.rewardSum / r.words).toFixed(1) : null,
    };
  });
  out.sort((a, b) => b.matches - a.matches);
  return out;
}

function loadAll(): MatchRecord[] {
  if (!fs.existsSync(MATCHES)) return [];
  return fs
    .readFileSync(MATCHES, "utf8")
    .split("\n")
    .filter((l) => l.trim())
    .map((l) => JSON.parse(l));
}

function save(conn: DbConnection, m: MatchRow) {
  const matchId = m.id;
  const key = String(matchId);
  if (seen.has(key)) return;
  // Sims-only matches say nothing about rivals; skip them before any work.
  const anyReal = (byMatch.parts.get(matchId) ?? []).some(
    (p) => !byMatch.bots.get(p.botId)?.isSimulated,
  );
  if (!anyReal) {
    seen.add(key);
    return;
  }
  const rec = record(conn, m);
  if (!rec) return;
  seen.add(key);
  // Sims-only matches say nothing about rivals.
  const real = rec.seats.filter((s) => !s.sim);
  if (real.length === 0) return;
  fs.appendFileSync(MATCHES, JSON.stringify(rec) + "\n");
  if (!bulk) {
    console.log(
      `match ${key}: ` +
        rec.seats
          .sort((a, b) => a.place - b.place)
          .map((s) => `${s.place}. ${s.name}${s.sim ? "" : "*"}=${s.score}`)
          .join("  "),
    );
  }
}

function writeRivals() {
  const rivals = profile(loadAll());
  fs.writeFileSync(RIVALS, JSON.stringify(rivals, null, 2));
  return rivals;
}

console.log(`connecting to ${DB} at ${HOST} (read-only, anonymous); the history can take a minute`);
const { conn } = await connect(HOST, DB);
for (const p of conn.db.matchParticipant.iter()) push(byMatch.parts, p);
for (const r of conn.db.auctionResult.iter()) push(byMatch.results, r);
for (const w of conn.db.wordPlay.iter()) push(byMatch.words, w);
for (const a of conn.db.auctionTopBidArchive.iter()) byMatch.topBid.set(a.auctionId, a.topBid);
for (const b of conn.db.bot.iter()) byMatch.bots.set(b.id, b);
conn.db.auctionTopBidArchive.onInsert((_c, a) => byMatch.topBid.set(a.auctionId, a.topBid));
conn.db.bot.onInsert((_c, b) => byMatch.bots.set(b.id, b));
conn.db.matchParticipant.onInsert((_c, p) => push(byMatch.parts, p));
conn.db.matchParticipant.onUpdate((_c, _o, p) => push(byMatch.parts, p));
conn.db.auctionResult.onInsert((_c, r) => push(byMatch.results, r));
conn.db.wordPlay.onInsert((_c, w) => push(byMatch.words, w));
console.log(`watching; ${conn.db.matchState.count()} matches in history`);
let bulk = true;
for (const m of conn.db.matchState.iter()) {
  if (m.status.tag === "Ended") save(conn, m);
}
bulk = false;
const rivals = writeRivals();
console.log(`${seen.size} matches on file; ${rivals.length} real bots profiled → ${RIVALS}`);
for (const r of rivals.slice(0, 15)) {
  console.log(
    `  ${r.name.padEnd(20)} n=${String(r.matches).padStart(4)}  place ${r.avgPlace}` +
      `  score ${r.avgScore}  bid≈${r.bidModel.slope}·v+${r.bidModel.intercept}` +
      `  word len ${r.avgWordLength}`,
  );
}

conn.db.matchState.onUpdate((_ctx, old, neu) => {
  if (old.status.tag !== "Ended" && neu.status.tag === "Ended") {
    // The top-bid archive lands in the same transaction; read it next tick.
    setTimeout(() => {
      save(conn, neu);
      writeRivals();
    }, 250);
  }
});
