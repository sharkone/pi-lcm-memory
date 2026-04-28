/**
 * Test helpers: in-memory DBs, fake pi-lcm tables, deterministic embedder.
 * No model downloads, no network. Vec ops are skipped if sqlite-vec
 * unavailable on the test platform.
 */

import Database from "better-sqlite3";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { ensureVecLoaded } from "../src/db/vec.js";
import { runMigrations } from "../src/db/schema.js";

export interface TestDb {
  db: Database.Database;
  dir: string;
  cleanup: () => void;
}

export function makeTestDb(): TestDb {
  const dir = mkdtempSync(join(tmpdir(), "pi-lcm-mem-test-"));
  const db = new Database(join(dir, "test.db"));
  db.pragma("journal_mode = WAL");
  db.pragma("busy_timeout = 5000");
  return {
    db,
    dir,
    cleanup: () => {
      try {
        db.close();
      } catch {
        // ignore
      }
      rmSync(dir, { recursive: true, force: true });
    },
  };
}

/** Build the pi-lcm tables our bridge reads — minimal subset. */
export function applyPiLcmSchema(db: Database.Database): void {
  db.exec(`
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
  `);
}

export function seedConversation(
  db: Database.Database,
  conv: { id: string; created_at?: string },
): void {
  db.prepare(
    "INSERT OR REPLACE INTO conversations(id, session_id, cwd, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
  ).run(conv.id, "s-" + conv.id, "/x", conv.created_at ?? "2026-04-20T10:00:00Z", "2026-04-20T11:00:00Z");
}

export function seedMessage(
  db: Database.Database,
  m: { id: string; conv: string; role: string; text: string; ts?: number; seq?: number },
): void {
  db.prepare(
    `INSERT INTO messages(id, conversation_id, role, content_text, timestamp, seq)
       VALUES (?, ?, ?, ?, ?, ?)`,
  ).run(m.id, m.conv, m.role, m.text, m.ts ?? Math.floor(Date.now() / 1000), m.seq ?? 0);
}

export function seedSummary(
  db: Database.Database,
  s: { id: string; conv: string; depth: number; text: string; created_at?: string },
): void {
  db.prepare(
    "INSERT INTO summaries(id, conversation_id, depth, text, created_at) VALUES (?, ?, ?, ?, ?)",
  ).run(s.id, s.conv, s.depth, s.text, s.created_at ?? "2026-04-25T12:00:00Z");
}

export async function setupVecAndMigrate(
  db: Database.Database,
  embeddingDim = 8,
  embeddingModel = "test-fake",
): Promise<{ vecLoaded: boolean }> {
  const v = await ensureVecLoaded(db);
  runMigrations(db, { embeddingDim, embeddingModel });
  return { vecLoaded: v.loaded };
}

/**
 * Deterministic fake embedder: hashes text into a fixed-dim float vector.
 * Same text → same vector. Different text → different vector. Norm ≈ 1.
 */
export class FakeEmbedder {
  readonly dim: number;
  readonly model: string;
  constructor(dim = 8, model = "test-fake") {
    this.dim = dim;
    this.model = model;
  }
  knownDims(): number {
    return this.dim;
  }
  state() {
    return { model: this.model, dims: this.dim, ready: true, error: null };
  }
  async warmup(): Promise<void> {
    // no-op
  }
  async embed(input: string | string[]): Promise<Float32Array[]> {
    const arr = Array.isArray(input) ? input : [input];
    return arr.map((s) => fakeVector(s, this.dim));
  }
}

function fakeVector(text: string, dim: number): Float32Array {
  const out = new Float32Array(dim);
  // Stable hash-mix over chars + bigram positions; spread bits across dims.
  for (let i = 0; i < text.length; i++) {
    const c = text.charCodeAt(i);
    const idx = (c * 2654435761 + i * 14029) % dim;
    out[idx]! += 1 + (c % 7) / 11;
  }
  // Lift any zero so two unrelated short strings still differ.
  for (let d = 0; d < dim; d++) {
    if (out[d] === 0) out[d] = 0.001 + d * 0.0001;
  }
  // L2 normalize.
  let n = 0;
  for (let d = 0; d < dim; d++) n += out[d]! * out[d]!;
  n = Math.sqrt(n) || 1;
  for (let d = 0; d < dim; d++) out[d]! /= n;
  return out;
}
