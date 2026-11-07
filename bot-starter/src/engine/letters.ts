// Game constants mirrored from the module (spacetimedb/src/letters.ts,
// spacetimedb/src/game.ts). Racks are `Rack` = 26 letter counts, A..Z.

export type Rack = Int16Array;

export const A_CODE = 65;

export const VALUES: readonly number[] = [
  // A  B  C  D  E  F  G  H  I  J  K  L  M  N  O  P   Q  R  S  T  U  V  W  X  Y   Z
  1, 3, 3, 2, 1, 4, 2, 4, 1, 8, 5, 1, 3, 1, 1, 3, 10, 1, 1, 1, 1, 4, 4, 8, 4, 10,
];

export const BAG: readonly number[] = [
  // A B  C  D  E   F  G  H  I  J  K  L  M  N  O  P  Q  R  S  T  U  V  W  X  Y  Z
  9, 2, 2, 4, 12, 2, 3, 2, 9, 1, 1, 4, 2, 6, 8, 2, 1, 6, 4, 6, 4, 2, 2, 1, 2, 1,
];

export const BAG_SIZE = 98;
export const STARTING_BALANCE = 100;

export function letterIndex(c: string): number {
  return c.charCodeAt(0) - A_CODE;
}

export function letterOf(i: number): string {
  return String.fromCharCode(A_CODE + i);
}

export function valueOf(c: string): number {
  return VALUES[letterIndex(c)] ?? 0;
}

// totalReward = floor(base * num / den), as the module computes it.
export function reward(length: number, base: number): number {
  if (length <= 3) return base;
  if (length === 4) return Math.floor((base * 3) / 2);
  if (length === 5) return base * 2;
  if (length === 6) return Math.floor((base * 5) / 2);
  return base * 3;
}

export function wordReward(word: string): number {
  let base = 0;
  for (const c of word) base += valueOf(c);
  return reward(word.length, base);
}

export function emptyRack(): Rack {
  return new Int16Array(26);
}

export function rackFrom(letters: string): Rack {
  const r = emptyRack();
  for (const c of letters) r[letterIndex(c)]!++;
  return r;
}

export function rackFromMap(m: Map<string, number>): Rack {
  const r = emptyRack();
  for (const [c, n] of m) {
    const i = letterIndex(c);
    if (i >= 0 && i < 26) r[i]! += n;
  }
  return r;
}

export function rackSize(r: Rack): number {
  let n = 0;
  for (let i = 0; i < 26; i++) n += r[i]!;
  return n;
}

export function rackToString(r: Rack): string {
  let s = "";
  for (let i = 0; i < 26; i++) s += letterOf(i).repeat(Math.max(0, r[i]!));
  return s;
}

export function subtractWord(r: Rack, word: string): Rack {
  const out = r.slice() as Rack;
  for (const c of word) out[letterIndex(c)]!--;
  return out;
}

export function canSpell(r: Rack, word: string): boolean {
  const need = emptyRack();
  for (const c of word) {
    const i = letterIndex(c);
    if (i < 0 || i >= 26) return false;
    if (++need[i]! > r[i]!) return false;
  }
  return true;
}
