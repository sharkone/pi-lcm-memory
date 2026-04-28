/**
 * DB fixtures for benchmarks: build a real on-disk SQLite DB with the
 * pi-lcm + pi-lcm-memory schemas, seed N messages of varying length, and
 * return handles plus a cleanup function.
 *
 * Mirrors test/helpers.ts but at scale, with realistic-ish synthetic
 * content (varied roles, lengths, distinct enough to embed sensibly).
 */

import Database from "better-sqlite3";
import { mkdtempSync, rmSync, statSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { ensureVecLoaded } from "../../src/db/vec.js";
import { runMigrations } from "../../src/db/schema.js";

export interface BenchDb {
  db: Database.Database;
  dir: string;
  dbPath: string;
  vecLoaded: boolean;
  cleanup: () => void;
}

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

/**
 * Open a fresh on-disk DB in a tmp directory, apply both schemas, and
 * load sqlite-vec.
 */
export async function makeBenchDb(opts: { embeddingDim: number; embeddingModel: string }): Promise<BenchDb> {
  const dir = mkdtempSync(join(tmpdir(), "pi-lcm-mem-bench-"));
  const dbPath = join(dir, "bench.db");
  const db = new Database(dbPath);
  db.pragma("journal_mode = WAL");
  db.pragma("busy_timeout = 5000");
  db.pragma("synchronous = NORMAL");
  db.exec(piLcmSchema);

  const vec = await ensureVecLoaded(db);
  runMigrations(db, { embeddingDim: opts.embeddingDim, embeddingModel: opts.embeddingModel });

  return {
    db,
    dir,
    dbPath,
    vecLoaded: vec.loaded,
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

/** Bytes occupied on disk by the DB file (and its WAL/SHM if present). */
export function dbDiskBytes(dbPath: string): number {
  let total = 0;
  for (const suffix of ["", "-wal", "-shm"]) {
    try {
      total += statSync(dbPath + suffix).size;
    } catch {
      // file may not exist yet
    }
  }
  return total;
}

/**
 * Pseudo-random but deterministic content generator. Uses a tiny LCG so
 * runs are reproducible from a seed.
 */
function makeRng(seed: number): () => number {
  let s = seed >>> 0;
  return () => {
    s = (s * 1664525 + 1013904223) >>> 0;
    return s / 0xffffffff;
  };
}

const TOPICS = [
  "embedding model warmup",
  "sqlite-vec kNN over float32 blobs",
  "hybrid retrieval with reciprocal rank fusion",
  "worker thread message protocol",
  "schema v2 side tables for many-to-one mappings",
  "infinite-loop in messagesNotInMemoryIndex generator",
  "settings panel custom UI factory",
  "primer text rendering at session_start",
  "auto-recall regex triggers and budget",
  "diagnostics ring buffer in memory_meta",
  "watchdog for slow model downloads",
  "side-channel tracer for freeze diagnosis",
  "single-transaction batched inserts",
  "content_hash dedupe across conversations",
  "FTS5 tokenizer for English text",
];

const FILLERS = [
  "We tried lowering the busy_timeout but the contention came from a different connection.",
  "The fix involved adding a SQL-level filter and a rowid cursor to skip already-considered rows.",
  "Profiling showed the main thread blocked on ONNX inference for hundreds of milliseconds.",
  "Moving the pipeline into a worker thread reduced perceived latency to imperceptible.",
  "We documented the change in CHANGELOG under the Phase 5 stabilization round.",
  "The failing test asserted that 200 pure tool-IO rows terminate in under two seconds.",
  "Quantized weights download faster and run roughly four times quicker on Node CPU.",
  "Using transferable ArrayBuffers avoided JSON-cloning float arrays back to the main thread.",
];

/** Build a sentence that contains the topic verbatim somewhere. */
function makeMessageText(rng: () => number, topicIdx: number, idx: number): string {
  const topic = TOPICS[topicIdx % TOPICS.length]!;
  const fillerCount = 1 + Math.floor(rng() * 3);
  const parts: string[] = [];
  for (let i = 0; i < fillerCount; i++) {
    parts.push(FILLERS[Math.floor(rng() * FILLERS.length)]!);
  }
  // Place the topic phrase somewhere in the message.
  const pos = Math.floor(rng() * (parts.length + 1));
  parts.splice(pos, 0, `Discussing ${topic} at iteration ${idx}.`);
  return parts.join(" ");
}

export interface SeedOptions {
  conversationId?: string;
  count: number;
  seed?: number;
  /** If true, half the rows are tool-IO that the bridge should filter out. */
  includeToolIO?: boolean;
}

export interface SeededRow {
  id: string;
  topicIdx: number;
  text: string;
  role: string;
}

/** Seed N messages into the pi-lcm `messages` table. Returns the seeded rows. */
export function seedMessages(db: Database.Database, opts: SeedOptions): SeededRow[] {
  const conv = opts.conversationId ?? "bench-conv-1";
  const rng = makeRng(opts.seed ?? 0xC0FFEE);

  // Ensure conversation row exists.
  db.prepare(
    `INSERT OR IGNORE INTO conversations(id, session_id, cwd, created_at, updated_at)
       VALUES (?, ?, ?, ?, ?)`,
  ).run(conv, "s-" + conv, "/x", "2026-04-20T10:00:00Z", "2026-04-20T11:00:00Z");

  const insert = db.prepare(
    `INSERT INTO messages(id, conversation_id, role, content_text, timestamp, seq)
       VALUES (?, ?, ?, ?, ?, ?)`,
  );
  const seeded: SeededRow[] = [];
  const tx = db.transaction((rows: SeededRow[]) => {
    let seq = 0;
    for (const r of rows) {
      insert.run(r.id, conv, r.role, r.text, 1700000000 + seeded.length, seq++);
    }
  });

  const realRoles = ["user", "assistant"];
  const ioRoles = ["toolResult", "bashExecution"];
  for (let i = 0; i < opts.count; i++) {
    const id = `m-${conv}-${i}`;
    const isIO = opts.includeToolIO === true && i % 2 === 0;
    const role = isIO
      ? ioRoles[Math.floor(rng() * ioRoles.length)]!
      : realRoles[Math.floor(rng() * realRoles.length)]!;
    const topicIdx = i % TOPICS.length;
    const text = isIO
      ? "internal tool output that the bridge should filter out"
      : makeMessageText(rng, topicIdx, i);
    seeded.push({ id, topicIdx, text, role });
  }

  tx(seeded);
  return seeded;
}

/** Convenient list of canonical topic strings (for query-side use). */
export const BENCH_TOPICS = TOPICS;
