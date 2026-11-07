// Which brain the bot runs, and its tuning. The decision-making itself is
// in ./engine (see planner.ts for how strategy C works).

import { StrategyC, type Brain } from "./engine/brains.js";
import type { Lexicon } from "./engine/lexicon.js";
import { DEFAULT_PARAMS, type Params } from "./engine/planner.js";

export { LETTER_VALUE } from "./engine/values.js";

// Live tuning. Auctions last 1 s and the bot bids and plays inside that
// window, so the rollout budgets must leave room for network latency.
export const LIVE_PARAMS: Params = {
  ...DEFAULT_PARAMS,
  bidTimeMs: Number(process.env.BID_TIME_MS ?? 300),
  playTimeMs: Number(process.env.PLAY_TIME_MS ?? 150),
};

export function createBrain(lex: Lexicon): Brain {
  return new StrategyC(lex, LIVE_PARAMS);
}
