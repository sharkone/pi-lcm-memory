/**
 * Additive migrations for pi-lcm-memory.
 *
 * Runs in pi-lcm's per-cwd SQLite. Never touches pi-lcm's tables. Owns:
 *   - memory_vec       (sqlite-vec virtual table, dim from config)
 *   - memory_index     (join table: vec_rowid -> source content + lineage)
 *   - memory_index_msg (pi-lcm message id -> vec_rowid; many-to-one)
 *   - memory_index_sum (pi-lcm summary id -> vec_rowid; many-to-one)
 *   - memory_meta      (kv bookkeeping, e.g. embedder model + dim of record)
 *   - _pi_lcm_memory_schema_version
 *
 * The vec table dim is parametric. If the configured embedder dim changes,
 * `migrateVecDim` rebuilds the table empty and clears index rows; the sweep
 * worker re-embeds from the underlying messages/summaries.
 *
 * Why the side tables (v2): content_hash on memory_index is UNIQUE, so
 * two different pi-lcm messages with identical content (e.g. a user repeats
 * a phrase, or a hook+sweep race) share one embedding. The `pi_lcm_msg_id`
 * column on memory_index records ONLY the canonical (first) message id —
 * the duplicate id was lost. The bridge's LEFT JOIN then kept yielding the
 * lost-mapping row forever (well, until the rowid cursor stopped it within
 * a sweep, but it'd reappear on the next sweep). The side tables map every
 * pi-lcm message/summary id to its vec_rowid, fixing the leak.
 */

import type Database from "better-sqlite3";
import { isVecLoadedFor } from "./vec.js";

const SCHEMA_VERSION = 2;

export interface MigrationOptions {
  embeddingDim: number;
  embeddingModel: string;
}

export function runMigrations(db: Database.Database, opts: MigrationOptions): void {
  ensureVersionTable(db);
  const current = currentSchemaVersion(db);
  if (current >= SCHEMA_VERSION) {
    reconcileVecDim(db, opts);
    return;
  }

  db.transaction(() => {
    if (current < 1) applyV1(db);
    if (current < 2) applyV2(db);
    db.prepare("INSERT INTO _pi_lcm_memory_schema_version (version) VALUES (?)").run(SCHEMA_VERSION);
  })();

  reconcileVecDim(db, opts);
}

function ensureVersionTable(db: Database.Database): void {
  db.prepare(
    `CREATE TABLE IF NOT EXISTS _pi_lcm_memory_schema_version (
       version INTEGER NOT NULL,
       applied_at INTEGER NOT NULL DEFAULT (unixepoch())
     )`,
  ).run();
}

function currentSchemaVersion(db: Database.Database): number {
  const row = db
    .prepare("SELECT MAX(version) AS v FROM _pi_lcm_memory_schema_version")
    .get() as { v: number | null } | undefined;
  return row?.v ?? 0;
}

function applyV1(db: Database.Database): void {
  db.prepare(
    `CREATE TABLE IF NOT EXISTS memory_index (
       vec_rowid       INTEGER PRIMARY KEY,
       source_kind     TEXT NOT NULL CHECK (source_kind IN ('message','summary')),
       content_hash    TEXT NOT NULL UNIQUE,
       pi_lcm_msg_id   TEXT,
       pi_lcm_sum_id   TEXT,
       conversation_id TEXT,
       session_started INTEGER,
       role            TEXT,
       depth           INTEGER,
       snippet         TEXT NOT NULL,
       text_full       TEXT NOT NULL,
       token_count     INTEGER,
       model_name      TEXT NOT NULL,
       model_dims      INTEGER NOT NULL,
       created_at      INTEGER NOT NULL DEFAULT (unixepoch())
     )`,
  ).run();

  db.prepare("CREATE INDEX IF NOT EXISTS idx_memidx_kind ON memory_index(source_kind)").run();
  db.prepare("CREATE INDEX IF NOT EXISTS idx_memidx_msg ON memory_index(pi_lcm_msg_id)").run();
  db.prepare("CREATE INDEX IF NOT EXISTS idx_memidx_sum ON memory_index(pi_lcm_sum_id)").run();
  db.prepare(
    "CREATE INDEX IF NOT EXISTS idx_memidx_session ON memory_index(conversation_id, session_started)",
  ).run();
  db.prepare("CREATE INDEX IF NOT EXISTS idx_memidx_hash ON memory_index(content_hash)").run();

  db.prepare(
    `CREATE TABLE IF NOT EXISTS memory_meta (
       k TEXT PRIMARY KEY,
       v TEXT NOT NULL
     )`,
  ).run();
}

function applyV2(db: Database.Database): void {
  // Many-to-one mapping: each pi-lcm message id maps to exactly one
  // vec_rowid, but multiple message ids can map to the same vec_rowid
  // when two messages share content_hash. PK on the pi-lcm side ensures
  // an id is recorded at most once.
  db.prepare(
    `CREATE TABLE IF NOT EXISTS memory_index_msg (
       pi_lcm_msg_id TEXT PRIMARY KEY,
       vec_rowid     INTEGER NOT NULL
     )`,
  ).run();
  db.prepare(
    "CREATE INDEX IF NOT EXISTS idx_memidx_msg_vec ON memory_index_msg(vec_rowid)",
  ).run();

  db.prepare(
    `CREATE TABLE IF NOT EXISTS memory_index_sum (
       pi_lcm_sum_id TEXT PRIMARY KEY,
       vec_rowid     INTEGER NOT NULL
     )`,
  ).run();
  db.prepare(
    "CREATE INDEX IF NOT EXISTS idx_memidx_sum_vec ON memory_index_sum(vec_rowid)",
  ).run();

  // Backfill: copy existing canonical mappings out of memory_index. Anything
  // inserted before v2 used pi_lcm_msg_id / pi_lcm_sum_id columns directly.
  db.prepare(
    `INSERT OR IGNORE INTO memory_index_msg(pi_lcm_msg_id, vec_rowid)
       SELECT pi_lcm_msg_id, vec_rowid FROM memory_index
        WHERE pi_lcm_msg_id IS NOT NULL`,
  ).run();
  db.prepare(
    `INSERT OR IGNORE INTO memory_index_sum(pi_lcm_sum_id, vec_rowid)
       SELECT pi_lcm_sum_id, vec_rowid FROM memory_index
        WHERE pi_lcm_sum_id IS NOT NULL`,
  ).run();
}

/**
 * Ensure memory_vec exists with the configured dim. If sqlite-vec isn't
 * loaded, skip silently — retrieval will fall back to FTS5-only.
 */
function reconcileVecDim(db: Database.Database, opts: MigrationOptions): void {
  if (!isVecLoadedFor(db)) return;

  const recordedDim = readMeta(db, "embedding_dim");
  const recordedModel = readMeta(db, "embedding_model");
  const desiredDim = String(opts.embeddingDim);

  const tableExists = db
    .prepare("SELECT name FROM sqlite_master WHERE type='table' AND name='memory_vec'")
    .get() as { name?: string } | undefined;

  const dimChanged = recordedDim !== null && recordedDim !== desiredDim;
  const modelChanged = recordedModel !== null && recordedModel !== opts.embeddingModel;

  if (tableExists && (dimChanged || modelChanged)) {
    // Sweep will re-embed from underlying content. Drop both vec rows and
    // index rows; clear side mappings (both will be rebuilt by the sweep).
    db.transaction(() => {
      db.prepare("DROP TABLE IF EXISTS memory_vec").run();
      db.prepare("DELETE FROM memory_index").run();
      // Side tables exist iff schema is v2+ — guard with table-exists check.
      const hasMsg = db
        .prepare("SELECT name FROM sqlite_master WHERE type='table' AND name='memory_index_msg'")
        .get();
      if (hasMsg) db.prepare("DELETE FROM memory_index_msg").run();
      const hasSum = db
        .prepare("SELECT name FROM sqlite_master WHERE type='table' AND name='memory_index_sum'")
        .get();
      if (hasSum) db.prepare("DELETE FROM memory_index_sum").run();
    })();
  }

  db.prepare(
    `CREATE VIRTUAL TABLE IF NOT EXISTS memory_vec USING vec0(
       embedding float[${opts.embeddingDim}]
     )`,
  ).run();

  writeMeta(db, "embedding_dim", desiredDim);
  writeMeta(db, "embedding_model", opts.embeddingModel);
}

function readMeta(db: Database.Database, key: string): string | null {
  const row = db.prepare("SELECT v FROM memory_meta WHERE k = ?").get(key) as { v: string } | undefined;
  return row?.v ?? null;
}

function writeMeta(db: Database.Database, key: string, value: string): void {
  db.prepare(
    `INSERT INTO memory_meta(k, v) VALUES (?, ?)
       ON CONFLICT(k) DO UPDATE SET v = excluded.v`,
  ).run(key, value);
}
