// Which tiles we haven't seen yet: the posterior pool the remaining bag is
// drawn from.
//
// Every tile is in exactly one place: auctioned (public), in our starting
// rack (we know it), in an opponent's starting rack (hidden until they
// play it), or still in the bag. So
//
//   unseen = BAG − auctioned − ourStart − revealedOpponentStarts
//
// By symmetry each unseen tile is equally likely to be the next one drawn,
// which is exactly what the rollouts need.

import { BAG, emptyRack, letterIndex, type Rack } from "./letters.js";

export interface AuctionEvent {
  letter: string;
  winnerBotId: bigint | null; // null: nobody won it, the tile is gone
}

export interface WordEvent {
  botId: bigint;
  word: string;
}

export interface TrackerInput {
  myBotId: bigint;
  myRack: Rack;
  results: readonly AuctionEvent[];
  words: readonly WordEvent[];
  openLetter: string | null; // the letter up for auction right now
}

// What we know of one opponent's rack: tiles it won minus tiles it played
// (never below 0), and how many of its starting tiles are still hidden.
export function knownHoldings(
  botId: bigint,
  results: readonly AuctionEvent[],
  words: readonly WordEvent[],
  startingRack = 5,
): { rack: Rack; hidden: number } {
  const won = emptyRack();
  const played = emptyRack();
  for (const r of results) if (r.winnerBotId === botId) won[letterIndex(r.letter)]!++;
  for (const w of words) {
    if (w.botId !== botId) continue;
    for (const c of w.word) played[letterIndex(c)]!++;
  }
  const rack = emptyRack();
  let revealed = 0;
  for (let i = 0; i < 26; i++) {
    rack[i] = Math.max(0, won[i]! - played[i]!);
    revealed += Math.max(0, played[i]! - won[i]!);
  }
  return { rack, hidden: Math.max(0, startingRack - revealed) };
}

export function unseenPool(input: TrackerInput): Rack {
  const unseen = Int16Array.from(BAG) as Rack;
  const won = new Map<bigint, Rack>();
  const played = new Map<bigint, Rack>();
  const at = (m: Map<bigint, Rack>, id: bigint) => {
    let r = m.get(id);
    if (!r) m.set(id, (r = emptyRack()));
    return r;
  };

  for (const r of input.results) {
    const i = letterIndex(r.letter);
    unseen[i]!--;
    if (r.winnerBotId !== null) at(won, r.winnerBotId)[i]!++;
  }
  if (input.openLetter) unseen[letterIndex(input.openLetter)]!--;
  for (const w of input.words) {
    const r = at(played, w.botId);
    for (const c of w.word) r[letterIndex(c)]!++;
  }

  // Our starting rack = what we hold + what we've played − what we won.
  const myWon = won.get(input.myBotId) ?? emptyRack();
  const myPlayed = played.get(input.myBotId) ?? emptyRack();
  for (let i = 0; i < 26; i++) {
    unseen[i]! -= Math.max(0, input.myRack[i]! + myPlayed[i]! - myWon[i]!);
  }
  // An opponent's play that its wins can't explain came from its start.
  for (const [botId, p] of played) {
    if (botId === input.myBotId) continue;
    const w = won.get(botId) ?? emptyRack();
    for (let i = 0; i < 26; i++) unseen[i]! -= Math.max(0, p[i]! - w[i]!);
  }
  for (let i = 0; i < 26; i++) if (unseen[i]! < 0) unseen[i] = 0;
  return unseen;
}
