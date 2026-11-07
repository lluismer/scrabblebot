// ScrabbleBot bot starter.
//
// First run, with a token from the website:
//   On your team's page, click "Generate token", then run
//   `BOT_NAME=<name> BOT_TOKEN=<token> npm start`.
//
// Or with a nonce from the CLI:
//   1. `spacetime call scrabblebot mint_credential_nonce`
//      `spacetime sql 'SELECT code FROM my_nonces ORDER BY expires_at DESC'`
//   2. `BOT_NAME=<name> BOT_NONCE=<code> npm start`
//      The bot connects fresh and redeems the nonce.
//
// Either way the bot saves its token to .token-<BOT_NAME>, so later runs
// need only `BOT_NAME=<name> npm start`.
//
// Entering a tournament: `BOT_NAME=<name> TOURNAMENT_ID=<id> npm start`
// registers the bot for that tournament when it connects. Only the bot
// itself can register, so this has to come from the bot's process.
//
// The decisions are made by the brain from ./src/strategy.ts (strategy C,
// in ./src/engine). This file only turns the client cache into the view
// the brain needs and sends its bids and words.

import {
  DbConnection,
  tables,
  type EventContext,
  type ErrorContext,
} from "./module_bindings/index.js";
import { Identity } from "spacetimedb";
import * as fs from "node:fs";
import * as path from "node:path";
import { fileURLToPath } from "node:url";
import { createBrain } from "./strategy.js";
import { LiveIndex } from "./live-index.js";
import { Lexicon } from "./engine/lexicon.js";
import { OpponentBook } from "./engine/learning.js";
import { rackFromMap } from "./engine/letters.js";
import type { MatchView } from "./engine/planner.js";
import type { OpponentModel, SimStrategy } from "./engine/pricing.js";
import {
  knownHoldings,
  unseenPool,
  type AuctionEvent,
  type WordEvent,
} from "./engine/tracker.js";

const HOST =
  process.env.STDB_HOST ??
  (process.argv.includes("--local")
    ? "http://127.0.0.1:3000"
    : "https://maincloud.spacetimedb.com");
const DB_NAME = process.env.STDB_DB ?? "scrabblebot";
const BOT_NAME = process.env.BOT_NAME ?? "bot";
const BOT_TOKEN = process.env.BOT_TOKEN; // from the website, first run only
const BOT_NONCE = process.env.BOT_NONCE; // from the CLI, first run only
const TOURNAMENT_ID = process.env.TOURNAMENT_ID;
const TOKEN_PATH = path.join(process.cwd(), `.token-${BOT_NAME}`);

function loadToken(): string | undefined {
  try {
    return fs.readFileSync(TOKEN_PATH, "utf8").trim() || undefined;
  } catch {
    return undefined;
  }
}
function saveToken(tok: string) {
  fs.writeFileSync(TOKEN_PATH, tok);
}

// The shared wordlist, so the bot can find playable words locally.
const dictionaryPath = path.join(
  path.dirname(fileURLToPath(import.meta.url)),
  "..",
  "..",
  "spacetimedb",
  "wordlist.txt",
);
if (!fs.existsSync(dictionaryPath)) {
  console.error(`wordlist not found at ${dictionaryPath}`);
  process.exit(1);
}
const LEXICON = Lexicon.fromFile(dictionaryPath);
const brain = createBrain(LEXICON);
const book = new OpponentBook();
console.log(`[${BOT_NAME}] ${LEXICON.size} words loaded; brain ${brain.name}`);

// Set once the subscription is applied; nothing is decided before that.
let index: LiveIndex | null = null;
let myIdentity: Identity | null = null;
let myBotId: bigint | null = null;
const bidsByAuction = new Set<string>();
// The match the brain is tracking, so it resets between matches.
let brainMatch: bigint | null = null;
// Words we've sent whose reducer call hasn't come back yet. Their letters
// are still in my_rack but already spoken for. Each leaves this set when
// its own call settles (the timeout is only a backstop).
const pendingWords = new Set<{ matchId: bigint; word: string; at: number }>();
const PENDING_MS = 3000;

function resolveMyBotId(conn: DbConnection): bigint | null {
  if (!myIdentity) return null;
  const cred = conn.db.botCredential.identity.find(myIdentity);
  return cred ? cred.botId : null;
}

function rackForMatch(
  conn: DbConnection,
  matchId: bigint,
): Map<string, number> {
  const rack = new Map<string, number>();
  for (const h of conn.db.myRack.iter()) {
    if (h.matchId !== matchId) continue;
    rack.set(h.letter, (rack.get(h.letter) ?? 0) + h.count);
  }
  return rack;
}

function participantForMatch(
  matchId: bigint,
): { balance: number; score: number } | null {
  if (myBotId === null || !index) return null;
  const p = index.participant(matchId, myBotId);
  return p ? { balance: Number(p.balance), score: Number(p.score) } : null;
}

// Matches we're seated in that are still being played.
function myRunningMatches(conn: DbConnection): bigint[] {
  if (myBotId === null || !index) return [];
  return index
    .matchesOf(myBotId)
    .filter((id) => index!.match(id)?.status.tag === "Running");
}

// A reducer call fails when the module refuses it, often for an expected
// reason: a word played just after its match ended, a bid that missed the
// auction's window. Log it and keep playing; an uncaught failure would stop
// the bot.
function logFailure(what: string) {
  return (err: unknown) =>
    console.warn(
      `[${BOT_NAME}] ${what} failed: ${err instanceof Error ? err.message : err}`,
    );
}

function isRealBot(conn: DbConnection, botId: bigint): boolean {
  const b = conn.db.bot.id.find(botId);
  return b != null && !b.isSimulated;
}

// Seed the real opponents' bid models from the profiles `npm run observe`
// saved (data/rivals.json, else the snapshot in hpc/rivals.json). The bot
// no longer downloads the server's whole history to fit them itself: that
// took minutes and froze it long enough for the server to drop it. Each
// match it plays then adds what it sees.
function seedRivals(conn: DbConnection) {
  const here = path.dirname(fileURLToPath(import.meta.url));
  const files = [path.join(here, "..", "data", "rivals.json"), path.join(here, "..", "hpc", "rivals.json")];
  const file = files.find((f) => fs.existsSync(f));
  if (!file) {
    console.log(`[${BOT_NAME}] no rivals.json; real rivals start from the default model`);
    return;
  }
  const raw = JSON.parse(fs.readFileSync(file, "utf8"));
  const list: { name: string; bidModel?: Model; bid?: Model }[] = Array.isArray(raw) ? raw : raw.rivals;
  type Model = { slope: number; intercept: number; sd: number; samples?: number };
  const byName = new Map<string, bigint>();
  for (const b of conn.db.bot.iter()) byName.set(b.name, b.id);
  let n = 0;
  for (const r of list) {
    const m = r.bidModel ?? r.bid;
    const id = byName.get(r.name);
    if (!m || id === undefined || id === myBotId) continue;
    book.seed(id, { slope: m.slope, intercept: m.intercept, sd: m.sd, n: m.samples ?? 0 });
    n++;
  }
  console.log(`[${BOT_NAME}] seeded ${n} rival bid models from ${path.relative(process.cwd(), file)}`);
}

// What every decision needs, kept small: one row per bot, credential,
// match or lobby. A match's history (auctions, results, words, bids) grows
// with every match ever played, so it's subscribed per match, only for the
// matches this bot sits in.
const GLOBAL_QUERIES = [
  tables.bot,
  tables.botCredential,
  tables.matchState,
  tables.openLobbies,
  tables.openLobbyMembers,
  tables.myRack,
  tables.tournament,
  tables.tournamentEntry,
  tables.tournamentMatch,
];
const subscribedMatches = new Set<bigint>();

function subscribeMatch(conn: DbConnection, matchId: bigint) {
  if (subscribedMatches.has(matchId)) return;
  subscribedMatches.add(matchId);
  conn
    .subscriptionBuilder()
    .onError(() => {
      subscribedMatches.delete(matchId);
      console.warn(`[${BOT_NAME}] subscription to match ${matchId} failed`);
    })
    .subscribe([
      tables.matchParticipant.where((r) => r.matchId.eq(matchId)),
      tables.auction.where((r) => r.matchId.eq(matchId)),
      tables.auctionResult.where((r) => r.matchId.eq(matchId)),
      tables.wordPlay.where((r) => r.matchId.eq(matchId)),
      tables.auctionTopBidArchive.where((r) => r.matchId.eq(matchId)),
    ]);
}

// Our own seats; each new one brings in its match.
function subscribeMine(conn: DbConnection, botId: bigint) {
  conn
    .subscriptionBuilder()
    .onApplied(() => {
      for (const p of conn.db.matchParticipant.iter()) {
        if (p.botId !== botId) continue;
        if (index?.match(p.matchId)?.status.tag === "Running") subscribeMatch(conn, p.matchId);
      }
      bootstrapActivity(conn);
    })
    .subscribe([tables.matchParticipant.where((r) => r.botId.eq(botId))]);
}

// Everything the brain needs about one match, from the client cache.
function viewFor(conn: DbConnection, matchId: bigint): MatchView | null {
  if (myBotId === null || !index) return null;
  const m = index.match(matchId);
  if (!m || m.status.tag !== "Running") return null;
  const me = participantForMatch(matchId);
  if (!me) return null;

  const rack = rackFromMap(rackForMatch(conn, matchId));
  for (const p of pendingWords) {
    if (Date.now() - p.at > PENDING_MS) pendingWords.delete(p);
    else if (p.matchId === matchId) {
      for (const c of p.word) {
        const i = c.charCodeAt(0) - 65;
        rack[i] = Math.max(0, rack[i]! - 1);
      }
    }
  }

  const results = index.resultsFor(matchId);
  const words = index.wordsFor(matchId);
  const openLetter = index.openLetter(matchId);

  const opponents: OpponentModel[] = [];
  for (const p of index.participantsOf(matchId)) {
    if (p.botId === myBotId) continue;
    const bot = conn.db.bot.id.find(p.botId);
    const balance = Number(p.balance);
    if (bot?.isSimulated) {
      opponents.push({
        kind: "sim",
        strategy: bot.strategy.tag as SimStrategy,
        balance,
      });
    } else {
      opponents.push(book.model(p.botId, balance));
    }
  }

  // The opponent with the highest score: the one to deny tiles to.
  let leader: { botId: bigint; score: bigint } | null = null;
  for (const p of index.participantsOf(matchId)) {
    if (p.botId === myBotId) continue;
    if (!leader || p.score > leader.score) leader = p;
  }

  return {
    rival: leader
      ? knownHoldings(leader.botId, results, words, m.startingRackSize)
      : undefined,
    kind: m.auctionType.tag === "FirstPrice" ? "FirstPrice" : "Vickrey",
    bagTotal: m.bagTotal,
    balance: me.balance,
    rack,
    unseen: unseenPool({ myBotId, myRack: rack, results, words, openLetter }),
    opponents,
  };
}

function ensureBrainMatch(matchId: bigint) {
  if (brainMatch === matchId) return;
  brainMatch = matchId;
  brain.newMatch();
}

// Auctions last 1 s. A bid has to be decided and on the wire well inside
// that, counting from when the auction reached us.
const AUCTION_WINDOW_MS = 1000;
const SEND_MARGIN_MS = 300; // network + server time we leave for the bid
const MIN_THINK_MS = 30;

function tryBid(
  conn: DbConnection,
  auctionId: bigint,
  matchId: bigint,
  letter: string,
  seenAt = performance.now(),
) {
  if (myBotId === null || !index) return;
  const key = `${matchId}:${auctionId}`;
  if (bidsByAuction.has(key)) return;
  // A backlog of callbacks can hand us an auction that has already closed.
  if (!index.isOpen(matchId, auctionId)) return;
  const view = viewFor(conn, matchId);
  if (!view) return;
  ensureBrainMatch(matchId);
  bidsByAuction.add(key);
  // Last auction: its letter is worthless but the rack isn't, so cash
  // everything in before anything else.
  if (view.bagTotal === 0) {
    tryPlayWord(conn, matchId);
    return;
  }
  const late = performance.now() - seenAt;
  const thinkMs = Math.min(
    Number(process.env.BID_TIME_MS ?? 300),
    AUCTION_WINDOW_MS - SEND_MARGIN_MS - late,
  );
  if (thinkMs < MIN_THINK_MS) {
    console.log(`[${BOT_NAME}] #${auctionId} '${letter}' skipped: reached us ${late.toFixed(0)}ms late`);
    tryPlayWord(conn, matchId);
    return;
  }
  const t0 = performance.now();
  const amount = Math.min(
    view.balance,
    Math.max(0, Math.floor(brain.bid(view, letter, thinkMs))),
  );
  const ms = performance.now() - t0;
  if (amount > 0) {
    conn.reducers
      .submitBid({ auctionId, amount: BigInt(amount) })
      .catch(logFailure(`bid on auction ${auctionId}`));
  }
  console.log(
    `[${BOT_NAME}] #${auctionId} '${letter}' bag=${view.bagTotal} ` +
      `bal=${view.balance} bid ${amount} (${brain.lastNote ?? ""}; ${ms.toFixed(0)}ms)`,
  );
  tryPlayWord(conn, matchId);
}

function tryPlayWord(conn: DbConnection, onlyMatch?: bigint) {
  if (myBotId === null) return;
  // Only matches still being played. The bot keeps its letters from
  // matches that have ended, but can't play them.
  const matches = myRunningMatches(conn).filter(
    (id) => onlyMatch === undefined || id === onlyMatch,
  );
  for (const matchId of matches) {
    const view = viewFor(conn, matchId);
    if (!view) continue;
    ensureBrainMatch(matchId);
    const words = brain.play(view);
    if (words.length === 0) continue;
    for (const word of words) {
      console.log(
        `[${BOT_NAME}] match ${matchId}: playing '${word}' (${brain.lastNote ?? ""})`,
      );
      const entry = { matchId, word, at: Date.now() };
      pendingWords.add(entry);
      conn.reducers
        .submitWord({ matchId, word })
        .catch(logFailure(`playing '${word}'`))
        .finally(() => pendingWords.delete(entry));
    }
  }
}

// The rack changed: a win landed, or the server applied our words. One
// change can arrive as several rows (5 at the deal), so run one decision
// per batch, and leave it to the bid when an auction is waiting for one:
// tryBid plays words right after bidding.
let playScheduled = false;
function onRackChanged(conn: DbConnection) {
  if (playScheduled) return;
  playScheduled = true;
  setImmediate(() => {
    playScheduled = false;
    if (!index) return;
    const waiting = myRunningMatches(conn).some((id) => {
      const a = index!.openAuction(id);
      return a !== undefined && !bidsByAuction.has(`${id}:${a.id}`);
    });
    if (!waiting) tryPlayWord(conn);
  });
}

function onConnect(conn: DbConnection, identity: Identity, token: string) {
  myIdentity = identity;
  saveToken(token);
  console.log(`[${BOT_NAME}] connected as ${identity.toHexString()}`);

  const live = new LiveIndex(conn);
  const subscribedAt = performance.now();
  console.log(
    `[${BOT_NAME}] subscribing to bots, matches and lobbies`,
  );
  conn
    .subscriptionBuilder()
    .onApplied(() => {
      live.load();
      index = live;
      console.log(
        `[${BOT_NAME}] subscription applied (${((performance.now() - subscribedAt) / 1000).toFixed(0)}s)`,
      );

      // Resolve our bot id, possibly after claiming a nonce.
      myBotId = resolveMyBotId(conn);
      if (myBotId === null) {
        if (BOT_NONCE) {
          console.log(`[${BOT_NAME}] claiming credential with nonce…`);
          conn.reducers
            .claimCredential({ code: BOT_NONCE })
            .catch(logFailure("claiming the credential"));
          // Wait briefly for the credential to land; check again.
          setTimeout(() => {
            myBotId = resolveMyBotId(conn);
            if (myBotId === null) {
              console.error(
                `[${BOT_NAME}] couldn't claim credential. Bad / expired nonce?`,
              );
              process.exit(1);
            }
            const bot = conn.db.bot.id.find(myBotId);
            console.log(
              `[${BOT_NAME}] claimed credential for bot '${bot?.name ?? "?"}' (id ${myBotId})`,
            );
            seedRivals(conn);
            subscribeMine(conn, myBotId);
          }, 1000);
          return;
        } else {
          console.error(
            `[${BOT_NAME}] no BotCredential for this token. Set BOT_TOKEN or BOT_NONCE and re-run.`,
          );
          process.exit(1);
        }
      }

      const bot = conn.db.bot.id.find(myBotId);
      console.log(
        `[${BOT_NAME}] acting as bot '${bot?.name ?? "?"}' (id ${myBotId})`,
      );
      seedRivals(conn);
      subscribeMine(conn, myBotId);
    })
    .onError(() => console.error(`[${BOT_NAME}] subscription failed`))
    .subscribe(GLOBAL_QUERIES);

  // A new seat of ours means a new match: bring in its rows.
  conn.db.matchParticipant.onInsert((_ctx, p) => {
    if (myBotId !== null && p.botId === myBotId) subscribeMatch(conn, p.matchId);
  });

  // Decisions wait for the end of the current batch of callbacks, so every
  // row from the same transaction (the closed auction's result, our new
  // tile) is in the indexes first.
  conn.db.auction.onInsert((_ctx: EventContext, a) => {
    if (!index || a.status.tag !== "Open") return;
    const seenAt = performance.now();
    setImmediate(() => tryBid(conn, a.id, a.matchId, a.letter, seenAt));
  });
  conn.db.myRack.onInsert(() => index && onRackChanged(conn));
  conn.db.myRack.onUpdate(() => index && onRackChanged(conn));
  conn.db.myRack.onDelete(() => index && onRackChanged(conn));
  conn.db.auctionTopBidArchive.onInsert((_ctx, a) => {
    if (!index) return;
    const r = index.result(a.matchId, a.auctionId);
    const w = r?.winnerBotId;
    if (r && w !== null && w !== undefined && w !== myBotId && isRealBot(conn, w)) {
      book.add(w, r.letter, a.topBid);
    }
  });
  conn.db.botCredential.onInsert(() => {
    if (myBotId === null) myBotId = resolveMyBotId(conn);
  });

  // When a match this bot was in ends, report it, relearn the opponents,
  // and hop back into the lobby.
  conn.db.matchState.onUpdate((_ctx, old, neu) => {
    if (myBotId === null || !index) return;
    if (old.status.tag !== "Ended" && neu.status.tag === "Ended") {
      const table = index.participantsOf(neu.id);
      // Not yet: this match's top-bid archive rows (same transaction) still
      // need its results.
      const live = index;
      setTimeout(() => live.forget(neu.id), 5000);
      if (!table.some((p) => p.botId === myBotId)) return;
      table.sort((a, b) => Number(b.score - a.score));
      const place = table.findIndex((p) => p.botId === myBotId) + 1;
      console.log(
        `[${BOT_NAME}] match ${neu.id} ended: place ${place}/${table.length}  ` +
          table
            .map((p) => `${conn.db.bot.id.find(p.botId)?.name ?? p.botId}=${p.score}`)
            .join(" "),
      );
      joinLobby(conn);
    }
  });
}

function joinLobby(conn: DbConnection) {
  if (myBotId === null) return;
  if (myRunningMatches(conn).length > 0) return;
  const openLobby = Array.from(conn.db.openLobbies.iter()).find(
    (l) => l.status.tag === "Open",
  );
  const alreadyIn =
    openLobby !== undefined &&
    Array.from(conn.db.openLobbyMembers.iter()).some(
      (lm) => lm.lobbyId === openLobby.id && lm.botId === myBotId,
    );
  if (alreadyIn) return;
  console.log(`[${BOT_NAME}] joining lobby`);
  conn.reducers.joinLobby({}).catch(logFailure("joining the lobby"));
}

function bootstrapActivity(conn: DbConnection) {
  if (TOURNAMENT_ID) {
    conn.reducers
      .registerForTournament({ tournamentId: BigInt(TOURNAMENT_ID) })
      .then(() =>
        console.log(`[${BOT_NAME}] registered for tournament ${TOURNAMENT_ID}`),
      )
      .catch(logFailure(`registering for tournament ${TOURNAMENT_ID}`));
  }
  joinLobby(conn);
  for (const a of conn.db.auction.iter()) {
    if (a.status.tag === "Open") tryBid(conn, a.id, a.matchId, a.letter);
  }
  tryPlayWord(conn);
}

function main() {
  console.log(`[${BOT_NAME}] connecting to ${DB_NAME} at ${HOST}`);
  DbConnection.builder()
    .withUri(HOST)
    .withDatabaseName(DB_NAME)
    .withToken(BOT_TOKEN ?? loadToken())
    .onConnect(onConnect)
    .onConnectError((_ctx: ErrorContext, err: Error) => {
      console.error("connect error:", err.message);
      setTimeout(() => process.exit(2), 500);
    })
    // Exit on a lost connection, so a wrapper loop (see STRATEGY.md) starts
    // a fresh one; reconnecting in-process would need all state reset.
    .onDisconnect(() => {
      console.log(`[${BOT_NAME}] disconnected; exiting so the run loop restarts the bot`);
      setTimeout(() => process.exit(2), 500);
    })
    .build();
}

main();
