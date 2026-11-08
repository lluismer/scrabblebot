// Reverse-engineer the best real bots from data/matches.jsonl.
//   npx tsx src/tools/topbots.ts            (all-real 6-bot matches, ≥15 matches per bot)
//   npx tsx src/tools/topbots.ts sepcis     (also print a profile for one bot by name)
// For each bot: average place, tiles won, what it pays (early / mid / late
// third of the match, vowels vs consonants vs high letters), how often it
// wins when it is the top bidder's rival, and how it plays words.
import * as fs from "node:fs";
import * as path from "node:path";
import { fileURLToPath } from "node:url";

const here = path.dirname(fileURLToPath(import.meta.url));
const lines = fs.readFileSync(path.join(here, "..", "..", "data", "matches.jsonl"), "utf8").split("\n");
const focus = (process.argv[2] ?? "").toLowerCase();
const VOW = new Set("AEIOU");
const HIGH = new Set("JQXZK");

type S = {
  name: string; m: number; place: number; score: number; tiles: number;
  pay: number[][]; // [phase] -> prices paid
  cls: Record<string, number[]>; // vowel/cons/high -> prices
  letters: Record<string, number>;
  words: number; wordLen: number; bingos: number; short: number; firstWordShare: number;
};
const by = new Map<string, S>();

for (const line of lines) {
  if (!line.trim()) continue;
  const m = JSON.parse(line);
  if (m.seats.length !== 6 || m.seats.some((s: any) => s.sim)) continue;
  const n = m.auctions.length;
  const t0 = Date.parse(m.words[0]?.at ?? m.endedAt);
  const span = Date.parse(m.endedAt) - t0 || 1;
  for (const seat of m.seats) {
    const s = by.get(seat.name) ?? {
      name: seat.name, m: 0, place: 0, score: 0, tiles: 0, pay: [[], [], []],
      cls: { vowel: [], cons: [], high: [] }, letters: {}, words: 0, wordLen: 0,
      bingos: 0, short: 0, firstWordShare: 0,
    };
    by.set(seat.name, s);
    s.m++; s.place += seat.place; s.score += seat.score;
    m.auctions.forEach((a: any, i: number) => {
      if (a.winner !== seat.botId) return;
      s.tiles++;
      s.pay[Math.min(2, Math.floor((3 * i) / n))]!.push(a.paid);
      s.cls[VOW.has(a.letter) ? "vowel" : HIGH.has(a.letter) ? "high" : "cons"]!.push(a.paid);
      s.letters[a.letter] = (s.letters[a.letter] ?? 0) + 1;
    });
    const mine = m.words.filter((w: any) => w.botId === seat.botId);
    s.words += mine.length;
    for (const w of mine) {
      s.wordLen += w.word.length;
      if (w.word.length >= 7) s.bingos++;
      if (w.word.length <= 3) s.short++;
    }
    // When it played its first word, as a share of the time from first word to match end.
    if (mine.length) s.firstWordShare += (Date.parse(mine[0].at) - t0) / span;
  }
}

const avg = (a: number[]) => (a.length ? a.reduce((x, y) => x + y, 0) / a.length : 0);
const f = (x: number, d = 1) => x.toFixed(d).padStart(5);
const rows = [...by.values()].filter((s) => s.m >= 15 || s.name.toLowerCase() === focus)
  .sort((a, b) => a.place / a.m - b.place / b.m);

console.log("bot                    games place score tiles | paid early  mid late | vowel cons high | words len bingo% short% 1stWord");
for (const s of rows) {
  const mark = s.name.toLowerCase() === focus ? "*" : " ";
  console.log(
    `${mark}${s.name.slice(0, 22).padEnd(22)} ${String(s.m).padStart(5)} ${f(s.place / s.m, 2)} ${f(s.score / s.m, 0)} ${f(s.tiles / s.m)} |` +
      `      ${f(avg(s.pay[0]!))}${f(avg(s.pay[1]!))}${f(avg(s.pay[2]!))} |` +
      ` ${f(avg(s.cls.vowel!))}${f(avg(s.cls.cons!))}${f(avg(s.cls.high!))} |` +
      ` ${f(s.words / s.m)}${f(s.words ? s.wordLen / s.words : 0)} ${f(s.words ? (100 * s.bingos) / s.words : 0, 0)}% ${f(s.words ? (100 * s.short) / s.words : 0, 0)}% ${f(s.words ? s.firstWordShare / s.m : 0, 2)}`,
  );
}
for (const s of rows.filter((r) => r.name.toLowerCase() === focus || rows.indexOf(r) < 3)) {
  const top = Object.entries(s.letters).sort((a, b) => b[1] - a[1]).slice(0, 12)
    .map(([l, c]) => `${l}:${(c / s.m).toFixed(1)}`).join(" ");
  console.log(`\n${s.name}: letters won per game  ${top}`);
}
