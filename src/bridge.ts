/**
 * Read-only bridge to pi-lcm's tables. We never INSERT/UPDATE/DELETE here.
 * Schema introspection guards against version drift in pi-lcm: missing tables
 * or columns degrade gracefully (we just see no rows).
 */

import type Database from "better-sqlite3";

export interface PiLcmMessage {
  id: string;
  conversation_id: string;
  role: string;
  content_text: string;
  timestamp: number;
  seq: number;
}

export interface PiLcmSummary {
  id: string;
  conversation_id: string;
  depth: number;
  text: string;
  created_at: string;
}

export interface PiLcmConversation {
  id: string;
  session_id: string;
  cwd: string;
  created_at: string;
  updated_at: string;
}

export class PiLcmBridge {
  private db: Database.Database;
  private hasMessages: boolean;
  private hasSummaries: boolean;
  private hasConversations: boolean;
  private hasFts5: boolean;

  constructor(db: Database.Database) {
    this.db = db;
    this.hasMessages = tableExists(db, "messages");
    this.hasSummaries = tableExists(db, "summaries");
    this.hasConversations = tableExists(db, "conversations");
    this.hasFts5 = tableExists(db, "messages_fts");
  }

  available(): { messages: boolean; summaries: boolean; conversations: boolean; fts5: boolean } {
    return {
      messages: this.hasMessages,
      summaries: this.hasSummaries,
      conversations: this.hasConversations,
      fts5: this.hasFts5,
    };
  }

  /** Iterate all message ids/text pairs not yet in our index. Used by sweep. */
  *messagesNotInMemoryIndex(batchSize: number): Generator<PiLcmMessage> {
    if (!this.hasMessages) return;
    const stmt = this.db.prepare(
      `SELECT m.id, m.conversation_id, m.role, m.content_text, m.timestamp, m.seq
         FROM messages m
         LEFT JOIN memory_index mi ON mi.pi_lcm_msg_id = m.id
         WHERE mi.vec_rowid IS NULL
         ORDER BY m.timestamp ASC
         LIMIT ?`,
    );

    while (true) {
      const rows = stmt.all(batchSize) as PiLcmMessage[];
      if (rows.length === 0) return;
      for (const r of rows) yield r;
      if (rows.length < batchSize) return;
    }
  }

  *summariesNotInMemoryIndex(batchSize: number): Generator<PiLcmSummary> {
    if (!this.hasSummaries) return;
    const stmt = this.db.prepare(
      `SELECT s.id, s.conversation_id, s.depth, s.text, s.created_at
         FROM summaries s
         LEFT JOIN memory_index mi ON mi.pi_lcm_sum_id = s.id
         WHERE mi.vec_rowid IS NULL
         ORDER BY s.created_at ASC
         LIMIT ?`,
    );
    while (true) {
      const rows = stmt.all(batchSize) as PiLcmSummary[];
      if (rows.length === 0) return;
      for (const r of rows) yield r;
      if (rows.length < batchSize) return;
    }
  }

  /** FTS5 ranked search over pi-lcm's messages_fts. Empty array if FTS5 unavailable. */
  ftsSearch(query: string, limit: number): { id: string; rank: number }[] {
    if (!this.hasFts5 || !query.trim()) return [];
    try {
      const rows = this.db
        .prepare(
          `SELECT m.id, fts.rank AS rank
             FROM messages_fts fts
             JOIN messages m ON m.rowid = fts.rowid
             WHERE fts.messages_fts MATCH ?
             ORDER BY fts.rank
             LIMIT ?`,
        )
        .all(query, limit) as { id: string; rank: number }[];
      return rows;
    } catch {
      return [];
    }
  }

  getMessageById(id: string): PiLcmMessage | null {
    if (!this.hasMessages) return null;
    const row = this.db
      .prepare(
        `SELECT id, conversation_id, role, content_text, timestamp, seq
           FROM messages WHERE id = ?`,
      )
      .get(id) as PiLcmMessage | undefined;
    return row ?? null;
  }

  getConversationById(id: string): PiLcmConversation | null {
    if (!this.hasConversations) return null;
    const row = this.db
      .prepare("SELECT id, session_id, cwd, created_at, updated_at FROM conversations WHERE id = ?")
      .get(id) as PiLcmConversation | undefined;
    return row ?? null;
  }

  /** Most recent N summaries with depth >= minDepth (for the primer). */
  recentSummaries(limit: number, minDepth: number = 1): PiLcmSummary[] {
    if (!this.hasSummaries) return [];
    return this.db
      .prepare(
        `SELECT id, conversation_id, depth, text, created_at
           FROM summaries
           WHERE depth >= ?
           ORDER BY created_at DESC
           LIMIT ?`,
      )
      .all(minDepth, limit) as PiLcmSummary[];
  }

  totalSessions(): number {
    if (!this.hasConversations) return 0;
    const r = this.db.prepare("SELECT COUNT(*) AS n FROM conversations").get() as { n: number };
    return r.n;
  }

  lastSessionStart(): string | null {
    if (!this.hasConversations) return null;
    const r = this.db
      .prepare("SELECT MAX(created_at) AS at FROM conversations")
      .get() as { at: string | null };
    return r.at;
  }

  /** Best-effort capture of the conversation pi-lcm just wrote into. */
  latestConversationId(): string | null {
    if (!this.hasMessages) return null;
    try {
      const r = this.db
        .prepare("SELECT conversation_id FROM messages ORDER BY rowid DESC LIMIT 1")
        .get() as { conversation_id?: string } | undefined;
      return r?.conversation_id ?? null;
    } catch {
      return null;
    }
  }
}

function tableExists(db: Database.Database, name: string): boolean {
  const row = db
    .prepare("SELECT name FROM sqlite_master WHERE type IN ('table','view') AND name = ?")
    .get(name) as { name?: string } | undefined;
  return !!row;
}
