# Strategy C: how it works and how to run everything

## The idea

Only finishing place counts, money and leftover tiles are worth 0 at the
end, and the tile from the last auction can never be played. The bot
treats the match as a set of auctions under a budget:

- **Letter value (ΔV).** Simulate the rest of the match many times. Each
  run draws future tiles from the unseen pool and keeps each one with the
  chance we'd win it. ΔV = E[best split of the final rack into words, with
  the letter] − E[the same, without it], using the same simulated futures on
  both sides so the noise cancels.
- **Price of money (λ, score per coin).** Bisect for the λ at which expected
  spending over the rest of the match equals the money we'll have. It falls
  to 0 as the bag empties.
- **Bid.** Vickrey: ΔV / λ (the true value in coins). First-price: the bid
  that maximises E[ΔV·win − λ·pay] against the opponent price model.
- **Opponents.** Sim bots are modelled exactly from their public strategy
  and balance. Real bots get bid ≈ slope·value + intercept, fitted from the
  public top-bid archive.
- **Words.** Play a word early only when λ·reward beats the chance at a
  longer word it gives up. In the last auction, play the best split of
  everything.
- **Denial.** A tile we don't buy goes to someone else, and only place
  counts. So a tile is also worth what it would add to the leading rival,
  whose rack is mostly public (tiles won minus tiles played, plus a few
  hidden starting tiles). Effective value = ΔV(us) + weight·ΔV(rival), with
  weight `denial1` against one opponent and `denialN` at bigger tables.
  Against a Bron stand-in in the two-bot final this took the win rate from
  6% to 69%. At tables of six it changes nothing measurable.

Code: `src/engine/` (planner, pricing, tracker, lexicon, evaluator,
learning), wired up in `src/index.ts`.

## Run the bot

```bash
npm install && npm run generate
BOT_NAME=<name> BOT_TOKEN=<token from /team> npm start
# tournament:
BOT_NAME=<name> TOURNAMENT_ID=<id> npm start
```

**Startup takes about a minute** while the server's full history
downloads. Start the bot well before a tournament begins, and keep it
running: an offline entrant still gets seated and loses every auction.
`BID_TIME_MS` (default 300) and `PLAY_TIME_MS` (150) set the thinking time
per decision; lower them if logs show bids arriving after an auction closed.

## Test privately on a local server

Nothing here touches the public server, so rivals learn nothing.

```bash
spacetime start                       # terminal 1, leave it running
npm run local:publish                 # after any module change
npm run local:setup -- 3              # makes local-bot-1..3 and their tokens
BOT_NAME=local-bot-1 npm run local:bot   # one terminal per bot
```

The lobby waits 90 s and then pads the table with sims. Start 6 local bots
for a table with no sims, like the tournament's.

## Watch rivals without playing

```bash
npm run observe
```

It connects anonymously, reads only public tables and calls no reducers,
so nothing it does is linked to your bot. It writes `data/matches.jsonl`
(every finished match that had a real bot) and `data/rivals.json` (each
real bot's place, score, fitted bid model and word habits), and keeps
appending as matches end.

## Simulate and tune

```bash
npm run sim -- --lineup table --matches 100 --workers 16 --vparam playMargin=3
```

Lineups: `c`, `base`, `c-base`, `c-c`, `c-v` (all padded with sims); the
tournament stand-ins `table` (a variant + 5 defaults, no sims) and `final`
(a variant vs 1 default); and the real field: `field` (C + 5 rival
stand-ins), `field-final` (C vs one rival, `--rival Bron`) and `rivals` (6
stand-ins, to calibrate them).

Rival stand-ins come from `hpc/rivals.json`, the 16 best real bots on the
server. Each bids its fitted line (bid ≈ slope·value + intercept, with
noise) and plays a word once it has 7+ letters. They match the real field's
prices (about 11 tiles a match at about 12 coins each) but score less than
the real bots (about 67 a match against 90-100), because they bid by letter
value alone instead of by fit with their rack.

Cluster jobs (`hpc/`, see the comments at the top of each):

- `sweep.slurm`: one parameter at a time against default copies.
- `samples.slurm`: does more thinking help (30 / 100 / 300 rollouts)?
- `cmaes.slurm`: CMA-ES over all 7 parameters against the field and the
  final, on one whole node, resumable. The main tuning job.

Merge array results with `npx tsx src/sim/merge.ts results/<job>/*/*.json`.

## Tournament format, and what it means for the bot

- **Swiss rounds:** points = table size − place, so maximise expected place.
  Total score breaks ties. From round 2 on, tables are seated by standing,
  so the strong bots meet each other.
- **Elimination:** last place at each table is knocked out, so avoiding
  last matters more than winning.
- **Final:** two bots, three games, decided by total score. Every tile you
  take is one your opponent doesn't get.
- **No sims at tournament tables.** Tune with `table` and `final`, not
  against sims.
