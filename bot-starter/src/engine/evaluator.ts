// S(R): the most a rack can score if it's split into words and all of
// them are played. Finding the exact best split is a set-packing problem,
// so this is a beam search: the top few first words at the shallow levels,
// then greedy (highest-reward word first) for the rest.

import type { Lexicon } from "./lexicon.js";
import { letterIndex, type Rack } from "./letters.js";

export interface Partition {
  value: number;
  words: string[];
}

// Beam widths per level. FAST is for Monte Carlo rollouts, EXACT for the
// real decisions about which words to play.
export const FAST_BEAM = [3];
export const EXACT_BEAM = [8, 5, 3, 2];

export function bestPartition(
  lex: Lexicon,
  rack: Rack,
  beam: readonly number[] = FAST_BEAM,
  memo?: Map<string, Partition>,
): Partition {
  return search(lex, rack.slice() as Rack, beam, 0, memo);
}

export function partitionValue(
  lex: Lexicon,
  rack: Rack,
  beam: readonly number[] = FAST_BEAM,
): number {
  return bestPartition(lex, rack, beam).value;
}

function search(
  lex: Lexicon,
  rack: Rack,
  beam: readonly number[],
  level: number,
  memo?: Map<string, Partition>,
): Partition {
  const width = beam[level] ?? 1;
  const key = memo ? `${level}:${rack.join(",")}` : "";
  if (memo) {
    const hit = memo.get(key);
    if (hit) return hit;
  }
  const candidates = lex.topWords(rack, width);
  let best: Partition = { value: 0, words: [] };
  for (const cand of candidates) {
    for (const c of cand.word) rack[letterIndex(c)]!--;
    const rest = search(lex, rack, beam, level + 1, memo);
    for (const c of cand.word) rack[letterIndex(c)]!++;
    const value = cand.reward + rest.value;
    if (value > best.value) best = { value, words: [cand.word, ...rest.words] };
  }
  if (memo) memo.set(key, best);
  return best;
}
