// Which brain the bot runs, and its tuning. The decision-making itself is
// in ./engine (see planner.ts for how strategy C works).

import { StrategyC, type Brain } from "./engine/brains.js";
import type { Lexicon } from "./engine/lexicon.js";
import { DEFAULT_PARAMS, type Params } from "./engine/planner.js";

export { LETTER_VALUE } from "./engine/values.js";

// Live settings, set from what live matches showed (2026-11-07): the real
// field bids 13+ coins a tile, so planning from today's near-zero win
// chances kept the rack small, the bot overpaid for the few tiles it won,
// cashed in 2-letter words and went broke by mid-match.
//   winFloor 0.25    plan to win enough tiles for 7-letter words (3×)
//   minPlayLen 5     no early words shorter than 5 letters
//   lambdaFloor 0.8  never bid more than ~1.25× a tile's value
export const LIVE_DEFAULTS: Partial<Params> = {
  winFloor: 0.25,
  minPlayLen: 5,
  lambdaFloor: 0.8,
};

// BOT_PARAMS overrides any parameter, e.g. "winFloor=0.4,lambdaFloor=1".
function fromEnv(): Partial<Params> {
  const out: Record<string, number> = {};
  for (const kv of (process.env.BOT_PARAMS ?? "").split(",")) {
    if (!kv.trim()) continue;
    const [k, v] = kv.split("=").map((s) => s.trim());
    if (!k || !(k in DEFAULT_PARAMS) || v === undefined || Number.isNaN(Number(v))) {
      throw new Error(`BOT_PARAMS: bad entry "${kv}"`);
    }
    out[k] = Number(v);
  }
  return out as Partial<Params>;
}

// Auctions last 1 s and the bot bids and plays inside that window, so the
// rollout budgets must leave room for network latency.
export const LIVE_PARAMS: Params = {
  ...DEFAULT_PARAMS,
  ...LIVE_DEFAULTS,
  bidTimeMs: Number(process.env.BID_TIME_MS ?? 300),
  playTimeMs: Number(process.env.PLAY_TIME_MS ?? 150),
  ...fromEnv(),
};

export function createBrain(lex: Lexicon): Brain {
  console.log(
    `strategy C settings: ${Object.entries(LIVE_PARAMS)
      .filter(([k]) => !/Ms$|Samples$/.test(k))
      .map(([k, v]) => `${k}=${v}`)
      .join(" ")}`,
  );
  return new StrategyC(lex, LIVE_PARAMS);
}
