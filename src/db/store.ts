/**
 * Data access for pi-lcm-memory's tables (memory_index, memory_vec, memory_meta).
 * Read-only access to pi-lcm's tables lives in src/bridge.ts.
 */

import type Database from "better-sqlite3";
import { encodeVector, isVecLoaded } from "./vec.js";

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

  constructor(db: Database.Database) {
    this.db = db;
  }

  /**
   * INSERT a row into memory_vec + memory_index transactionally. Idempotent
   * by content_hash — if the row already exists, returns its existing
   * vec_rowid and does nothing. Returns null if vec is unavailable.
   */
  insert(args: InsertArgs): number | null {
    if (!isVecLoaded()) return null;
    if (args.embedding.length !== args.model_dims) {
      throw new Error(
        `embedding length ${args.embedding.length} does not match model_dims ${args.model_dims}`,
      );
    }

    return this.db.transaction(() => {
      const existing = this.db
        .prepare("SELECT vec_rowid FROM memory_index WHERE content_hash = ?")
        .get(args.content_hash) as { vec_rowid: number } | undefined;
      if (existing) return existing.vec_rowid;

      const insertVec = this.db.prepare(
        "INSERT INTO memory_vec(embedding) VALUES (?)",
      );
      const result = insertVec.run(encodeVector(args.embedding));
      const vecRowid = Number(result.lastInsertRowid);

      this.db
        .prepare(
          `INSERT INTO memory_index(
             vec_rowid, source_kind, content_hash,
             pi_lcm_msg_id, pi_lcm_sum_id, conversation_id, session_started,
             role, depth, snippet, text_full, token_count, model_name, model_dims
           ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
        )
        .run(
          vecRowid,
          args.source_kind,
          args.content_hash,
          args.pi_lcm_msg_id ?? null,
          args.pi_lcm_sum_id ?? null,
          args.conversation_id ?? null,
          args.session_started ?? null,
          args.role ?? null,
          args.depth ?? null,
          args.snippet,
          args.text_full,
          args.token_count ?? null,
          args.model_name,
          args.model_dims,
        );
      return vecRowid;
    })();
  }

  /** True if a row with this content_hash is already indexed. */
  hasContentHash(contentHash: string): boolean {
    const row = this.db
      .prepare("SELECT 1 AS ok FROM memory_index WHERE content_hash = ? LIMIT 1")
      .get(contentHash) as { ok: number } | undefined;
    return !!row;
  }

  getRow(vecRowid: number): IndexRow | null {
    const row = this.db
      .prepare("SELECT * FROM memory_index WHERE vec_rowid = ?")
      .get(vecRowid) as IndexRow | undefined;
    return row ?? null;
  }

  getRowByMsgId(msgId: string): IndexRow | null {
    const row = this.db
      .prepare("SELECT * FROM memory_index WHERE pi_lcm_msg_id = ? LIMIT 1")
      .get(msgId) as IndexRow | undefined;
    return row ?? null;
  }

  /** kNN search over memory_vec; returns vec_rowids ranked by distance. */
  knn(query: Float32Array, k: number): { vec_rowid: number; distance: number }[] {
    if (!isVecLoaded()) return [];
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

    let vecRows = 0;
    if (isVecLoaded()) {
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
      vecAvailable: isVecLoaded(),
    };
  }

  clearAll(): void {
    if (!isVecLoaded()) {
      this.db.prepare("DELETE FROM memory_index").run();
      return;
    }
    this.db.transaction(() => {
      this.db.prepare("DELETE FROM memory_vec").run();
      this.db.prepare("DELETE FROM memory_index").run();
    })();
  }
}
