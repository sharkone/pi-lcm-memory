/**
 * Hybrid retrieval: FTS5 (lexical) ∪ sqlite-vec (semantic) → RRF merge.
 *
 *   score(d) = Σ_i  1 / (k + rank_i(d))
 *
 * "lexical" mode = FTS5 ranks only.
 * "semantic" mode = vec ranks only.
 * "hybrid" mode = both, merged via RRF with k = config.rrfK.
 *
 * Filters are applied post-rank against memory_index (cheap, expected to be
 * selective on conversation_id / time range).
 */

import type Database from "better-sqlite3";
import type { Embedder } from "./embeddings/embedder.js";
import type { MemoryStore, IndexRow } from "./db/store.js";
import type { PiLcmBridge } from "./bridge.js";

export type RecallMode = "hybrid" | "lexical" | "semantic";

export interface RecallParams {
  query: string;
  k?: number;
  mode?: RecallMode;
  sessionFilter?: string | null;
  after?: string | null;
  before?: string | null;
}

export interface RecallHit {
  source_kind: "message" | "summary";
  conversation_id: string | null;
  session_started: number | null;
  role: string | null;
  depth: number | null;
  snippet: string;
  text_full: string;
  score: number;
  pi_lcm_msg_id: string | null;
  pi_lcm_sum_id: string | null;
}

export interface RetrieverDeps {
  db: Database.Database;
  store: MemoryStore;
  embedder: Embedder;
  bridge: PiLcmBridge;
  rrfK: number;
  /** Candidate multiplier for FTS5 fetch: lexK = k * lexMult. Default 4. */
  lexMult?: number;
  /** Candidate multiplier for KNN fetch: semK = k * semMult. Default 4. */
  semMult?: number;
}

export class Retriever {
  private deps: RetrieverDeps;

  constructor(deps: RetrieverDeps) {
    this.deps = deps;
  }

  async recall(params: RecallParams): Promise<RecallHit[]> {
    const k = clamp(params.k ?? 10, 1, 100);
    const mode: RecallMode = params.mode ?? "hybrid";
    const query = params.query.trim();
    if (!query) return [];

    const lexK = mode === "semantic" ? 0 : k * (this.deps.lexMult ?? 4);
    const semK = mode === "lexical" ? 0 : k * (this.deps.semMult ?? 4);

    const lex = lexK > 0 ? this.lexicalRanks(query, lexK) : new Map<number, number>();
    const sem = semK > 0 ? await this.semanticRanks(query, semK) : new Map<number, number>();

    const merged = mergeRRF(lex, sem, this.deps.rrfK);
    const sorted = Array.from(merged.entries()).sort((a, b) => b[1] - a[1]);

    const filtered = applyFilters(this.deps.store, sorted, params, k * 2);
    return filtered.slice(0, k);
  }

  /** "More like this" — KNN over an existing row's vector. */
  async similar(input: { messageId?: string; vecRowid?: number }, k: number): Promise<RecallHit[]> {
    const seedRow = input.vecRowid != null
      ? this.deps.store.getRow(input.vecRowid)
      : input.messageId
      ? this.deps.store.getRowByMsgId(input.messageId)
      : null;
    if (!seedRow) return [];

    // Re-embed the seed text to get a query vector. Cheaper than fetching the
    // raw bytes from sqlite-vec (which would require reading & decoding).
    const [seedVec] = await this.deps.embedder.embed(seedRow.text_full);
    if (!seedVec) return [];

    const knn = this.deps.store.knn(seedVec, k + 1); // +1 to drop the seed itself
    return knn
      .filter((h) => h.vec_rowid !== seedRow.vec_rowid)
      .slice(0, k)
      .map((h) => ({
        ...rowToHit(this.deps.store.getRow(h.vec_rowid)!),
        score: 1 - h.distance,
      }));
  }

  // ── Internals ──────────────────────────────────────────────────────────────

  /** Map<vec_rowid, rank-position-zero-based> for FTS5 hits. */
  private lexicalRanks(query: string, limit: number): Map<number, number> {
    const out = new Map<number, number>();
    const ftsHits = this.deps.bridge.ftsSearch(toFts5Match(query), limit);
    if (ftsHits.length === 0) return out;

    // Translate pi-lcm message ids to memory_index vec_rowids.
    const idsParam = ftsHits.map((h) => h.id);
    if (idsParam.length === 0) return out;

    const placeholders = idsParam.map(() => "?").join(",");
    const rows = this.deps.db
      .prepare(
        `SELECT pi_lcm_msg_id AS id, vec_rowid
           FROM memory_index
           WHERE pi_lcm_msg_id IN (${placeholders})`,
      )
      .all(...idsParam) as { id: string; vec_rowid: number }[];

    const idToVec = new Map(rows.map((r) => [r.id, r.vec_rowid]));
    for (let i = 0; i < ftsHits.length; i++) {
      const vec = idToVec.get(ftsHits[i]!.id);
      if (vec != null && !out.has(vec)) out.set(vec, i);
    }
    return out;
  }

  private async semanticRanks(query: string, limit: number): Promise<Map<number, number>> {
    const out = new Map<number, number>();
    const [qvec] = await this.deps.embedder.embed(query);
    if (!qvec) return out;
    const knn = this.deps.store.knn(qvec, limit);
    knn.forEach((hit, i) => out.set(hit.vec_rowid, i));
    return out;
  }
}

function mergeRRF(
  a: Map<number, number>,
  b: Map<number, number>,
  k: number,
): Map<number, number> {
  const out = new Map<number, number>();
  const score = (rank: number) => 1 / (k + rank + 1);
  for (const [id, rank] of a) out.set(id, (out.get(id) ?? 0) + score(rank));
  for (const [id, rank] of b) out.set(id, (out.get(id) ?? 0) + score(rank));
  return out;
}

function applyFilters(
  store: MemoryStore,
  sorted: [number, number][],
  params: RecallParams,
  cap: number,
): RecallHit[] {
  const after = params.after ? Date.parse(params.after) / 1000 : null;
  const before = params.before ? Date.parse(params.before) / 1000 : null;
  const session = params.sessionFilter ?? null;

  const out: RecallHit[] = [];
  for (const [vecRowid, score] of sorted) {
    if (out.length >= cap) break;
    const row = store.getRow(vecRowid);
    if (!row) continue;
    if (session && row.conversation_id !== session) continue;
    if (after != null && row.session_started != null && row.session_started < after) continue;
    if (before != null && row.session_started != null && row.session_started > before) continue;
    out.push({ ...rowToHit(row), score });
  }
  return out;
}

function rowToHit(row: IndexRow): Omit<RecallHit, "score"> {
  return {
    source_kind: row.source_kind,
    conversation_id: row.conversation_id,
    session_started: row.session_started,
    role: row.role,
    depth: row.depth,
    snippet: row.snippet,
    text_full: row.text_full,
    pi_lcm_msg_id: row.pi_lcm_msg_id,
    pi_lcm_sum_id: row.pi_lcm_sum_id,
  };
}

function clamp(n: number, lo: number, hi: number): number {
  return Math.max(lo, Math.min(hi, n));
}

/** Escape user input for FTS5 MATCH; treat as a phrase query if simple. */
function toFts5Match(query: string): string {
  const tokens = query.match(/[A-Za-z0-9_]+/g) ?? [];
  if (tokens.length === 0) return `"${query.replace(/"/g, '""')}"`;
  // OR-merge tokens with prefix matching for recall friendliness.
  return tokens.map((t) => `${t}*`).join(" OR ");
}
