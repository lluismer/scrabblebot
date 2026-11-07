// A trie over the wordlist with a branch-and-bound search for the
// highest-reward words a rack can spell.
//
// The search is the inner loop of the Monte Carlo planner (thousands of
// calls per auction), so it's written against typed arrays with no
// allocation per node.

import * as fs from "node:fs";
import { reward, VALUES, type Rack } from "./letters.js";

export interface Word {
  word: string;
  reward: number;
}

export class Lexicon {
  // Node 0 is the root. Children form a singly linked list.
  private firstChild: Int32Array;
  private nextSibling: Int32Array;
  private letter: Uint8Array;
  private terminal: Uint8Array;
  // Longest path from this node down to a terminal, in letters.
  private maxBelow: Uint8Array;
  private nodes = 1;
  readonly size: number;
  private words: Set<string>;

  // Scratch space for the search.
  private path = new Uint8Array(64);
  private counts = new Int16Array(26);
  private topReward: number[] = [];
  private topWord: string[] = [];
  private k = 1;
  private minLen = 2;

  constructor(wordList: Iterable<string>) {
    const words: string[] = [];
    for (const raw of wordList) {
      const w = raw.trim().toUpperCase();
      if (w.length >= 2 && w.length <= 60 && /^[A-Z]+$/.test(w)) words.push(w);
    }
    this.words = new Set(words);
    this.size = this.words.size;
    let cap = 1;
    for (const w of this.words) cap += w.length;
    this.firstChild = new Int32Array(cap).fill(-1);
    this.nextSibling = new Int32Array(cap).fill(-1);
    this.letter = new Uint8Array(cap);
    this.terminal = new Uint8Array(cap);
    this.maxBelow = new Uint8Array(cap);
    for (const w of this.words) this.insert(w);
    this.computeMaxBelow();
  }

  static fromFile(path: string): Lexicon {
    return new Lexicon(fs.readFileSync(path, "utf8").split("\n"));
  }

  has(word: string): boolean {
    return this.words.has(word);
  }

  private insert(w: string) {
    let node = 0;
    for (let i = 0; i < w.length; i++) {
      const c = w.charCodeAt(i) - 65;
      let child = this.firstChild[node]!;
      while (child !== -1 && this.letter[child] !== c) {
        child = this.nextSibling[child]!;
      }
      if (child === -1) {
        child = this.nodes++;
        this.letter[child] = c;
        this.nextSibling[child] = this.firstChild[node]!;
        this.firstChild[node] = child;
      }
      node = child;
    }
    this.terminal[node] = 1;
  }

  private computeMaxBelow() {
    // Children always have higher ids than their parent, so a reverse sweep
    // sees every child before its parent.
    const parent = new Int32Array(this.nodes).fill(-1);
    for (let n = 0; n < this.nodes; n++) {
      for (let c = this.firstChild[n]!; c !== -1; c = this.nextSibling[c]!) {
        parent[c] = n;
      }
    }
    for (let n = this.nodes - 1; n > 0; n--) {
      const p = parent[n]!;
      const via = this.maxBelow[n]! + 1;
      if (via > this.maxBelow[p]!) this.maxBelow[p] = via;
    }
  }

  // The `k` highest-reward words `rack` can spell, best first.
  topWords(rack: Rack, k = 1, minLen = 2): Word[] {
    let remSum = 0;
    let remCount = 0;
    for (let i = 0; i < 26; i++) {
      this.counts[i] = rack[i]!;
      remSum += rack[i]! * VALUES[i]!;
      remCount += rack[i]!;
    }
    this.topReward = [];
    this.topWord = [];
    this.k = k;
    this.minLen = minLen;
    this.search(0, 0, 0, remSum, remCount);
    return this.topWord.map((word, i) => ({
      word,
      reward: this.topReward[i]!,
    }));
  }

  // The best reward any single word reaches, 0 if none.
  bestReward(rack: Rack): number {
    const top = this.topWords(rack, 1);
    return top.length ? top[0]!.reward : 0;
  }

  private threshold(): number {
    return this.topReward.length < this.k
      ? 0
      : this.topReward[this.topReward.length - 1]!;
  }

  private search(
    node: number,
    depth: number,
    base: number,
    remSum: number,
    remCount: number,
  ) {
    for (
      let child = this.firstChild[node]!;
      child !== -1;
      child = this.nextSibling[child]!
    ) {
      const c = this.letter[child]!;
      if (this.counts[c]! === 0) continue;
      const v = VALUES[c]!;
      const d = depth + 1;
      const b = base + v;
      const rs = remSum - v;
      const rc = remCount - 1;
      // Optimistic bound: every remaining tile joins the word, at the
      // longest length the trie still allows below this child.
      const extra = Math.min(this.maxBelow[child]!, rc);
      const ub = reward(d + extra, b + rs);
      if (ub <= this.threshold()) continue;
      this.path[depth] = c;
      if (this.terminal[child] && d >= this.minLen) {
        const r = reward(d, b);
        if (r > this.threshold()) this.offer(r, d);
      }
      if (extra > 0) {
        this.counts[c]!--;
        this.search(child, d, b, rs, rc);
        this.counts[c]!++;
      }
    }
  }

  private offer(r: number, len: number) {
    let word = "";
    for (let i = 0; i < len; i++) word += String.fromCharCode(65 + this.path[i]!);
    let i = this.topReward.length;
    while (i > 0 && this.topReward[i - 1]! < r) i--;
    this.topReward.splice(i, 0, r);
    this.topWord.splice(i, 0, word);
    if (this.topReward.length > this.k) {
      this.topReward.pop();
      this.topWord.pop();
    }
  }
}
