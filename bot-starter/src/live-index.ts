// Per-match indexes over the client cache.
//
// subscribeToAllTables brings in the server's whole history (tens of
// thousands of matches, millions of auction rows), and the SDK's find()
// scans a whole table: about 10 ms on match_state and hundreds of ms on
// the auction tables. A 1-second auction can't afford that, so every row a
// decision needs comes from these maps, filled by one pass at startup and
// kept current by callbacks.

import type { DbConnection } from "./module_bindings/index.js";
import type { AuctionEvent, WordEvent } from "./engine/tracker.js";

export interface ParticipantRow {
  matchId: bigint;
  botId: bigint;
  balance: bigint;
  score: bigint;
}

type MatchRow = NonNullable<
  ReturnType<DbConnection["db"]["matchState"]["id"]["find"]>
>;

export class LiveIndex {
  private results = new Map<bigint, Map<bigint, AuctionEvent>>();
  private words = new Map<bigint, Map<bigint, WordEvent>>();
  private participants = new Map<bigint, Map<bigint, ParticipantRow>>();
  private botMatches = new Map<bigint, Set<bigint>>();
  // matchState.id.find takes ~10 ms on the live server; keep our own map.
  private matches = new Map<bigint, MatchRow>();
  // The open auction in each match.
  private open = new Map<bigint, { id: bigint; letter: string }>();
  // Callbacks also fire for the rows the subscription first brings in;
  // ignore those and let load() take what matters in one pass.
  private ready = false;

  constructor(private conn: DbConnection) {
    conn.db.auctionResult.onInsert((_c, r) => this.ready && this.addResult(r));
    conn.db.wordPlay.onInsert((_c, w) => this.ready && this.addWord(w));
    conn.db.matchParticipant.onInsert((_c, p) => this.ready && this.addParticipant(p));
    conn.db.matchParticipant.onUpdate((_c, _o, p) => this.ready && this.addParticipant(p));
    conn.db.matchState.onInsert((_c, m) => this.ready && this.matches.set(m.id, m));
    conn.db.matchState.onUpdate((_c, _o, m) => this.ready && this.matches.set(m.id, m));
    conn.db.auction.onInsert((_c, a) => this.ready && this.addAuction(a));
    conn.db.auction.onUpdate((_c, _o, a) => this.ready && this.addAuction(a));
  }

  // One pass over what the subscription brought in. Results and words are
  // only kept for matches still running: finished ones never matter again.
  load() {
    this.ready = true;
    const running = new Set<bigint>();
    for (const m of this.conn.db.matchState.iter()) {
      this.matches.set(m.id, m);
      if (m.status.tag === "Running") running.add(m.id);
    }
    for (const p of this.conn.db.matchParticipant.iter()) this.addParticipant(p);
    for (const a of this.conn.db.auction.iter()) {
      if (a.status.tag === "Open") this.addAuction(a);
    }
    for (const r of this.conn.db.auctionResult.iter()) {
      if (running.has(r.matchId)) this.addResult(r);
    }
    for (const w of this.conn.db.wordPlay.iter()) {
      if (running.has(w.matchId)) this.addWord(w);
    }
  }

  // Drop a finished match's auctions and words.
  forget(matchId: bigint) {
    this.results.delete(matchId);
    this.words.delete(matchId);
    this.open.delete(matchId);
  }

  private addAuction(a: {
    id: bigint;
    matchId: bigint;
    letter: string;
    status: { tag: string };
  }) {
    if (a.status.tag === "Open") this.open.set(a.matchId, { id: a.id, letter: a.letter });
    else if (this.open.get(a.matchId)?.id === a.id) this.open.delete(a.matchId);
  }

  private addResult(r: {
    auctionId: bigint;
    matchId: bigint;
    letter: string;
    winnerBotId: bigint | undefined;
  }) {
    let m = this.results.get(r.matchId);
    if (!m) this.results.set(r.matchId, (m = new Map()));
    m.set(r.auctionId, { letter: r.letter, winnerBotId: r.winnerBotId ?? null });
    // A result means that auction has closed.
    if (this.open.get(r.matchId)?.id === r.auctionId) this.open.delete(r.matchId);
  }

  private addWord(w: { id: bigint; matchId: bigint; botId: bigint; word: string }) {
    let m = this.words.get(w.matchId);
    if (!m) this.words.set(w.matchId, (m = new Map()));
    m.set(w.id, { botId: w.botId, word: w.word });
  }

  private addParticipant(p: ParticipantRow) {
    let m = this.participants.get(p.matchId);
    if (!m) this.participants.set(p.matchId, (m = new Map()));
    m.set(p.botId, p);
    let b = this.botMatches.get(p.botId);
    if (!b) this.botMatches.set(p.botId, (b = new Set()));
    b.add(p.matchId);
  }

  match(matchId: bigint): MatchRow | undefined {
    return this.matches.get(matchId);
  }

  // The letter up for auction in a match right now, if any.
  openLetter(matchId: bigint): string | null {
    return this.open.get(matchId)?.letter ?? null;
  }

  result(matchId: bigint, auctionId: bigint): AuctionEvent | undefined {
    return this.results.get(matchId)?.get(auctionId);
  }

  resultsFor(matchId: bigint): AuctionEvent[] {
    return [...(this.results.get(matchId)?.values() ?? [])];
  }

  wordsFor(matchId: bigint): WordEvent[] {
    return [...(this.words.get(matchId)?.values() ?? [])];
  }

  participantsOf(matchId: bigint): ParticipantRow[] {
    return [...(this.participants.get(matchId)?.values() ?? [])];
  }

  participant(matchId: bigint, botId: bigint): ParticipantRow | undefined {
    return this.participants.get(matchId)?.get(botId);
  }

  matchesOf(botId: bigint): bigint[] {
    return [...(this.botMatches.get(botId) ?? [])];
  }
}
