// Promise wrappers around the SDK's connection callbacks, for scripts.

import { DbConnection } from "../module_bindings/index.js";
import type { Identity } from "spacetimedb";

export interface Connected {
  conn: DbConnection;
  identity: Identity;
  token: string;
}

// Connect (anonymously when `token` is undefined) and wait until the
// subscription to every public table has been applied.
export function connect(
  host: string,
  db: string,
  token?: string,
): Promise<Connected> {
  return new Promise((resolve, reject) => {
    DbConnection.builder()
      .withUri(host)
      .withDatabaseName(db)
      .withToken(token)
      .onConnect((conn, identity, tok) => {
        conn
          .subscriptionBuilder()
          .onApplied(() => resolve({ conn, identity, token: tok }))
          .onError((_ctx) => reject(new Error("subscription failed")))
          .subscribeToAllTables();
      })
      .onConnectError((_ctx, err) => reject(err))
      .build();
  });
}

// Poll `check` until it returns something, or give up after `ms`.
export async function waitFor<T>(
  check: () => T | undefined | null,
  ms = 5000,
): Promise<T> {
  const start = Date.now();
  for (;;) {
    const v = check();
    if (v !== undefined && v !== null) return v;
    if (Date.now() - start > ms) throw new Error("timed out waiting");
    await new Promise((r) => setTimeout(r, 50));
  }
}
