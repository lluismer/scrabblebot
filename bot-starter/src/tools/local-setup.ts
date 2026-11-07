// Create test bots on a LOCAL SpacetimeDB server, so the real bot can be
// run end to end without anyone on the public server seeing it.
//
//   spacetime start                                  (terminal 1)
//   npm run local:publish                            (once per module change)
//   npm run local:setup -- 3                         (makes local-bot-1..3)
//   BOT_NAME=local-bot-1 npm run local:bot           (one terminal per bot)
//
// For each bot it connects a fresh anonymous identity, creates a team with
// that bot, mints a credential nonce, then redeems the nonce from a second
// fresh identity, and saves that identity's token as .token-<bot name>,
// where the starter looks for it.
//
// It refuses any host but a local one: the public server is never touched.

import * as fs from "node:fs";
import * as path from "node:path";
import { connect, waitFor } from "./connect.js";

const HOST = process.env.STDB_HOST ?? "http://127.0.0.1:3000";
const DB = process.env.STDB_DB ?? "scrabblebot";
const COUNT = Number(process.argv[2] ?? "1");
const PREFIX = process.env.BOT_PREFIX ?? "local-bot";

if (!/^https?:\/\/(127\.0\.0\.1|localhost|\[::1\])(:\d+)?\/?$/.test(HOST)) {
  console.error(`refusing to run against ${HOST}: local servers only`);
  process.exit(1);
}

async function makeBot(name: string) {
  const owner = await connect(HOST, DB);
  await owner.conn.reducers.createTeam({ teamName: `${name}-team`, botName: name });
  await owner.conn.reducers.mintCredentialNonce({});
  const nonce = await waitFor(() =>
    [...owner.conn.db.myNonces.iter()].sort((a, b) =>
      a.expiresAt.microsSinceUnixEpoch < b.expiresAt.microsSinceUnixEpoch ? 1 : -1,
    )[0],
  );
  owner.conn.disconnect();

  const bot = await connect(HOST, DB);
  await bot.conn.reducers.claimCredential({ code: nonce.code });
  await waitFor(() => bot.conn.db.botCredential.identity.find(bot.identity));
  fs.writeFileSync(path.join(process.cwd(), `.token-${name}`), bot.token);
  bot.conn.disconnect();
  console.log(`created ${name}; token saved to .token-${name}`);
}

for (let i = 1; i <= COUNT; i++) {
  try {
    await makeBot(`${PREFIX}-${i}`);
  } catch (e) {
    console.error(`${PREFIX}-${i}: ${e instanceof Error ? e.message : e}`);
  }
}
process.exit(0);
