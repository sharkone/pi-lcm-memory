/**
 * Lightweight structured diagnostics into memory_meta. Avoid bloat — keeps
 * only the most recent N events; rotates on insert.
 */

import type Database from "better-sqlite3";

const META_KEY = "events";
const MAX_EVENTS = 200;

export interface DiagEvent {
  ts: number;
  event: string;
  data?: Record<string, unknown>;
}

export class Diagnostics {
  private db: Database.Database;

  constructor(db: Database.Database) {
    this.db = db;
  }

  log(event: string, data?: Record<string, unknown>): void {
    try {
      const row: DiagEvent = { ts: Math.floor(Date.now() / 1000), event, ...(data ? { data } : {}) };
      const existing = this.db.prepare("SELECT v FROM memory_meta WHERE k = ?").get(META_KEY) as
        | { v: string }
        | undefined;
      const arr: DiagEvent[] = existing ? safeParse(existing.v) : [];
      arr.push(row);
      if (arr.length > MAX_EVENTS) arr.splice(0, arr.length - MAX_EVENTS);
      this.db
        .prepare(
          "INSERT INTO memory_meta(k, v) VALUES (?, ?) ON CONFLICT(k) DO UPDATE SET v = excluded.v",
        )
        .run(META_KEY, JSON.stringify(arr));
    } catch {
      // diagnostics never crash callers
    }
  }

  recent(n = 20): DiagEvent[] {
    try {
      const r = this.db.prepare("SELECT v FROM memory_meta WHERE k = ?").get(META_KEY) as
        | { v: string }
        | undefined;
      const arr: DiagEvent[] = r ? safeParse(r.v) : [];
      return arr.slice(-n);
    } catch {
      return [];
    }
  }

  clear(): void {
    try {
      this.db.prepare("DELETE FROM memory_meta WHERE k = ?").run(META_KEY);
    } catch {
      // ignore
    }
  }
}

function safeParse(s: string): DiagEvent[] {
  try {
    const v = JSON.parse(s);
    return Array.isArray(v) ? (v as DiagEvent[]) : [];
  } catch {
    return [];
  }
}
