/**
 * E2E fixtures: tmp project + pre-seeded pi-lcm-shaped DB at the path the
 * extension will resolve via `PI_LCM_MEMORY_DB_DIR + hashCwd(cwd)`.
 */

import Database from "better-sqlite3";
import { mkdtempSync, rmSync, mkdirSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { hashCwd } from "../../src/utils.js";

const piLcmSchema = `
  CREATE TABLE IF NOT EXISTS conversations (
    id TEXT PRIMARY KEY,
    session_id TEXT,
    cwd TEXT,
    created_at TEXT,
    updated_at TEXT
  );
  CREATE TABLE IF NOT EXISTS messages (
    id TEXT PRIMARY KEY,
    conversation_id TEXT NOT NULL,
    role TEXT NOT NULL,
    content_text TEXT NOT NULL,
    timestamp INTEGER NOT NULL,
    seq INTEGER NOT NULL
  );
  CREATE TABLE IF NOT EXISTS summaries (
    id TEXT PRIMARY KEY,
    conversation_id TEXT NOT NULL,
    depth INTEGER NOT NULL,
    text TEXT NOT NULL,
    created_at TEXT NOT NULL
  );
  CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts USING fts5(
    content_text,
    content='messages',
    content_rowid='rowid'
  );
  CREATE TRIGGER IF NOT EXISTS messages_ai AFTER INSERT ON messages BEGIN
    INSERT INTO messages_fts(rowid, content_text) VALUES (new.rowid, new.content_text);
  END;
`;

export interface E2EProject {
  /** Tmp cwd for this fake project. */
  cwd: string;
  /** Tmp dbDir; tests must set PI_LCM_MEMORY_DB_DIR to this before loading the extension. */
  dbDir: string;
  /** Resolved DB path the extension will open. */
  dbPath: string;
  cleanup: () => void;
}

export interface SeedMessage {
  id: string;
  role: string;
  text: string;
  conversationId?: string;
}

/**
 * Build a tmp project + a pre-created SQLite DB file with the pi-lcm schema
 * already in place (no pi-lcm-memory tables yet — those are added by the
 * extension's runMigrations during session_start).
 *
 * NOTE: tests must set `process.env.PI_LCM_MEMORY_DB_DIR = project.dbDir`
 * before importing the extension module so the extension resolves the same
 * dbDir we used here.
 */
export function makeE2EProject(opts: {
  conversationId?: string;
  messages?: SeedMessage[];
}): E2EProject {
  const root = mkdtempSync(join(tmpdir(), "pi-lcm-mem-e2e-"));
  const cwd = join(root, "project");
  const dbDir = join(root, "db");
  mkdirSync(cwd, { recursive: true });
  mkdirSync(dbDir, { recursive: true, mode: 0o700 });

  const dbPath = join(dbDir, `${hashCwd(cwd)}.db`);
  const db = new Database(dbPath);
  db.pragma("journal_mode = WAL");
  db.pragma("busy_timeout = 5000");
  db.exec(piLcmSchema);

  const conv = opts.conversationId ?? "e2e-conv-1";
  db.prepare(
    `INSERT OR IGNORE INTO conversations(id, session_id, cwd, created_at, updated_at)
       VALUES (?, ?, ?, ?, ?)`,
  ).run(conv, "s-" + conv, cwd, "2026-04-20T10:00:00Z", "2026-04-20T11:00:00Z");

  if (opts.messages?.length) {
    const insert = db.prepare(
      `INSERT INTO messages(id, conversation_id, role, content_text, timestamp, seq)
         VALUES (?, ?, ?, ?, ?, ?)`,
    );
    db.transaction(() => {
      let seq = 0;
      for (const m of opts.messages!) {
        insert.run(
          m.id,
          m.conversationId ?? conv,
          m.role,
          m.text,
          1700000000 + seq,
          seq,
        );
        seq++;
      }
    })();
  }

  // Important: close our handle so the extension can open its own.
  db.close();

  return {
    cwd,
    dbDir,
    dbPath,
    cleanup: () => {
      rmSync(root, { recursive: true, force: true });
    },
  };
}

/** Wait until `predicate()` becomes true or `timeoutMs` elapses. */
export async function waitFor(
  predicate: () => boolean | Promise<boolean>,
  opts: { timeoutMs?: number; intervalMs?: number } = {},
): Promise<void> {
  const t0 = Date.now();
  const timeoutMs = opts.timeoutMs ?? 30_000;
  const intervalMs = opts.intervalMs ?? 50;
  while (Date.now() - t0 < timeoutMs) {
    if (await predicate()) return;
    await new Promise((r) => setTimeout(r, intervalMs));
  }
  throw new Error(`waitFor: predicate never became true within ${timeoutMs}ms`);
}
