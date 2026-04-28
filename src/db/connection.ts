/**
 * Shared SQLite connection. Mirrors pi-lcm's connection.ts pragmas and path
 * scheme so we open the SAME file as pi-lcm.
 *
 * Two extensions opening the same SQLite WAL DB from the same Node process is
 * fine when each calls `new Database(path)`; SQLite serializes writes via the
 * busy_timeout. When pi-lcm and pi-lcm-memory run in-process, both opens are
 * legal — but to avoid duplicate handles we keep a single per-cwd connection
 * inside this module.
 */

import Database from "better-sqlite3";
import { mkdirSync, chmodSync } from "node:fs";
import { join } from "node:path";
import { hashCwd } from "../utils.js";

let db: Database.Database | null = null;
let currentCwd: string | null = null;

export function getDbPath(dbDir: string, cwd: string): string {
  return join(dbDir, `${hashCwd(cwd)}.db`);
}

export function openDb(dbDir: string, cwd: string): Database.Database {
  if (db && currentCwd !== cwd) closeDb();
  if (db) return db;

  mkdirSync(dbDir, { recursive: true, mode: 0o700 });
  const dbPath = getDbPath(dbDir, cwd);

  db = new Database(dbPath);

  try {
    chmodSync(dbPath, 0o600);
  } catch {
    // some filesystems don't support chmod
  }

  db.pragma("journal_mode = WAL");
  db.pragma("busy_timeout = 5000");
  db.pragma("foreign_keys = ON");
  db.pragma("synchronous = NORMAL");

  currentCwd = cwd;
  return db;
}

export function closeDb(): void {
  if (!db) return;
  try {
    db.pragma("wal_checkpoint(PASSIVE)");
  } catch {
    // non-fatal
  }
  try {
    db.close();
  } catch {
    // ignore
  }
  db = null;
  currentCwd = null;
}

export function getOpenDb(): Database.Database | null {
  return db;
}

export function getOpenCwd(): string | null {
  return currentCwd;
}
