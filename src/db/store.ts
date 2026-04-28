/**
 * Data access for pi-lcm-memory's tables (memory_index, memory_vec, memory_meta).
 * Read-only access to pi-lcm's tables lives in src/bridge.ts.
 */

import type Database from "better-sqlite3";
import { encodeVector, isVecLoadedFor } from "./vec.js";

// Use the explicit Statement type with `unknown[]` bind parameters so .run()
// accepts variadic args. ReturnType<prepare> resolves through a generic
// conditional type that doesn't preserve the variadic shape.
type Stmt = Database.Statement<unknown[], unknown>;

export interface IndexRow {
  vec_rowid: number;
  source_kind: "message" | "summary";
  content_hash: string;
  pi_lcm_msg_id: string | null;
  pi_lcm_sum_id: string | null;
  conversation_id: string | null;
  session_started: number | null;
  role: string | null;
  depth: number | null;
  snippet: string;
  text_full: string;
  token_count: number | null;
  model_name: string;
  model_dims: number;
  created_at: number;
}

export interface MemoryStats {
  indexed: number;
  byMessage: number;
  bySummary: number;
  vecRows: number;
  modelName: string | null;
  modelDims: number | null;
  dbSizeBytes: number;
  vecAvailable: boolean;
}

export interface InsertArgs {
  source_kind: "message" | "summary";
  content_hash: string;
  embedding: Float32Array;
  pi_lcm_msg_id?: string | null;
  pi_lcm_sum_id?: string | null;
  conversation_id?: string | null;
  session_started?: number | null;
  role?: string | null;
  depth?: number | null;
  snippet: string;
  text_full: string;
  token_count?: number | null;
  model_name: string;
  model_dims: number;
}

export class MemoryStore {
  private db: Database.Database;

  // Cached prepared statements. Reused across inserts so we don't re-compile
  // SQL on every call — better-sqlite3's prepare() is fast but it allocates,
  // and the hot path runs hundreds of times per backfill.
  private stmtSelectByHash: Stmt | null = null;
  private stmtInsertVec: Stmt | null = null;
  private stmtInsertIndex: Stmt | null = null;
  private stmtSelectByVecRowid: Stmt | null = null;
  private stmtSelectByMsgId: Stmt | null = null;
  private stmtKnn: Stmt | null = null;
  private stmtHasHash: Stmt | null = null;

  constructor(db: Database.Database) {
    this.db = db;
  }

  private prepStmts(): void {
    if (this.stmtInsertVec) return;
    this.stmtSelectByHash = this.db.prepare(
      "SELECT vec_rowid FROM memory_index WHERE content_hash = ?",
    );
    this.stmtInsertVec = this.db.prepare("INSERT INTO memory_vec(embedding) VALUES (?)");
    this.stmtInsertIndex = this.db.prepare(
      `INSERT INTO memory_index(
         vec_rowid, source_kind, content_hash,
         pi_lcm_msg_id, pi_lcm_sum_id, conversation_id, session_started,
         role, depth, snippet, text_full, token_count, model_name, model_dims
       ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
    );
    this.stmtSelectByVecRowid = this.db.prepare("SELECT * FROM memory_index WHERE vec_rowid = ?");
    this.stmtSelectByMsgId = this.db.prepare(
      "SELECT * FROM memory_index WHERE pi_lcm_msg_id = ? LIMIT 1",
    );
    this.stmtHasHash = this.db.prepare(
      "SELECT 1 AS ok FROM memory_index WHERE content_hash = ? LIMIT 1",
    );
  }

  /**
   * INSERT a row into memory_vec + memory_index transactionally. Idempotent
   * by content_hash — if the row already exists, returns its existing
   * vec_rowid and does nothing. Returns null if vec is unavailable.
   */
  insert(args: InsertArgs): number | null {
    const result = this.insertBatch([args]);
    return result[0] ?? null;
  }

  /**
   * INSERT many rows in a SINGLE transaction. Critical for backfill: turns
   * 32 lock acquisitions into 1, slashes contention with concurrent writers
   * (e.g. pi-lcm). Returns vec_rowids in the same order as input. Already-
   * present rows return their existing vec_rowid.
   */
  insertBatch(items: InsertArgs[]): (number | null)[] {
    if (!isVecLoadedFor(this.db)) return items.map(() => null);
    if (items.length === 0) return [];
    this.prepStmts();

    for (const a of items) {
      if (a.embedding.length !== a.model_dims) {
        throw new Error(
          `embedding length ${a.embedding.length} does not match model_dims ${a.model_dims}`,
        );
      }
    }

    const stmtSelect = this.stmtSelectByHash!;
    const stmtVec = this.stmtInsertVec!;
    const stmtIndex = this.stmtInsertIndex!;

    const out: (number | null)[] = new Array(items.length).fill(null);
    const txn = this.db.transaction((batch: InsertArgs[]) => {
      for (let i = 0; i < batch.length; i++) {
        const a = batch[i]!;
        const existing = stmtSelect.get(a.content_hash) as
          | { vec_rowid: number }
          | undefined;
        if (existing) {
          out[i] = existing.vec_rowid;
          continue;
        }
        const r = stmtVec.run(encodeVector(a.embedding));
        const vecRowid = Number(r.lastInsertRowid);
        stmtIndex.run(
          vecRowid,
          a.source_kind,
          a.content_hash,
          a.pi_lcm_msg_id ?? null,
          a.pi_lcm_sum_id ?? null,
          a.conversation_id ?? null,
          a.session_started ?? null,
          a.role ?? null,
          a.depth ?? null,
          a.snippet,
          a.text_full,
          a.token_count ?? null,
          a.model_name,
          a.model_dims,
        );
        out[i] = vecRowid;
      }
    });
    // immediate transaction = SQLite acquires the WRITE lock up front, so
    // we don't deadlock-risk by grabbing the read lock first then trying
    // to upgrade. Plays nicer with concurrent writers (pi-lcm).
    (txn as unknown as { immediate: (b: InsertArgs[]) => void }).immediate(items);
    return out;
  }

  /** True if a row with this content_hash is already indexed. */
  hasContentHash(contentHash: string): boolean {
    this.prepStmts();
    const row = this.stmtHasHash!.get(contentHash) as { ok: number } | undefined;
    return !!row;
  }

  /** Bulk variant of hasContentHash for batched filtering. Returns Set of present hashes. */
  whichHashesPresent(hashes: string[]): Set<string> {
    if (hashes.length === 0) return new Set();
    // Chunk into reasonable IN() lists; SQLite's bind parameter limit is
    // 32766 by default but we keep it conservative.
    const present = new Set<string>();
    const CHUNK = 256;
    for (let i = 0; i < hashes.length; i += CHUNK) {
      const slice = hashes.slice(i, i + CHUNK);
      const placeholders = slice.map(() => "?").join(",");
      const rows = this.db
        .prepare(`SELECT content_hash FROM memory_index WHERE content_hash IN (${placeholders})`)
        .all(...slice) as { content_hash: string }[];
      for (const r of rows) present.add(r.content_hash);
    }
    return present;
  }

  getRow(vecRowid: number): IndexRow | null {
    this.prepStmts();
    const row = this.stmtSelectByVecRowid!.get(vecRowid) as IndexRow | undefined;
    return row ?? null;
  }

  getRowByMsgId(msgId: string): IndexRow | null {
    this.prepStmts();
    const row = this.stmtSelectByMsgId!.get(msgId) as IndexRow | undefined;
    return row ?? null;
  }

  /** kNN search over memory_vec; returns vec_rowids ranked by distance. */
  knn(query: Float32Array, k: number): { vec_rowid: number; distance: number }[] {
    if (!isVecLoadedFor(this.db)) return [];
    const rows = this.db
      .prepare(
        `SELECT rowid AS vec_rowid, distance
           FROM memory_vec
           WHERE embedding MATCH ?
           ORDER BY distance
           LIMIT ?`,
      )
      .all(encodeVector(query), k) as { vec_rowid: number; distance: number }[];
    return rows;
  }

  stats(): MemoryStats {
    const indexedRow = this.db.prepare("SELECT COUNT(*) AS n FROM memory_index").get() as { n: number };
    const byKindRows = this.db
      .prepare("SELECT source_kind, COUNT(*) AS n FROM memory_index GROUP BY source_kind")
      .all() as { source_kind: "message" | "summary"; n: number }[];
    const byMessage = byKindRows.find((r) => r.source_kind === "message")?.n ?? 0;
    const bySummary = byKindRows.find((r) => r.source_kind === "summary")?.n ?? 0;

    const vecAvailable = isVecLoadedFor(this.db);
    let vecRows = 0;
    if (vecAvailable) {
      try {
        const r = this.db.prepare("SELECT COUNT(*) AS n FROM memory_vec").get() as { n: number };
        vecRows = r.n;
      } catch {
        vecRows = 0;
      }
    }

    const meta = this.db.prepare("SELECT k, v FROM memory_meta").all() as { k: string; v: string }[];
    const modelName = meta.find((m) => m.k === "embedding_model")?.v ?? null;
    const modelDimsRaw = meta.find((m) => m.k === "embedding_dim")?.v ?? null;
    const modelDims = modelDimsRaw ? Number(modelDimsRaw) : null;

    const sizeRow = this.db
      .prepare("SELECT page_count * page_size AS bytes FROM pragma_page_count(), pragma_page_size()")
      .get() as { bytes: number };

    return {
      indexed: indexedRow.n,
      byMessage,
      bySummary,
      vecRows,
      modelName,
      modelDims,
      dbSizeBytes: sizeRow.bytes,
      vecAvailable,
    };
  }

  clearAll(): void {
    if (!isVecLoadedFor(this.db)) {
      this.db.prepare("DELETE FROM memory_index").run();
      return;
    }
    this.db.transaction(() => {
      this.db.prepare("DELETE FROM memory_vec").run();
      this.db.prepare("DELETE FROM memory_index").run();
    })();
  }
}
