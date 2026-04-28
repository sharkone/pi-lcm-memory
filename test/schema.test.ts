import { describe, it, expect, afterEach } from "vitest";
import { makeTestDb, setupVecAndMigrate, applyPiLcmSchema, type TestDb } from "./helpers.js";
import { runMigrations } from "../src/db/schema.js";

describe("schema migrations", () => {
  let t: TestDb | null = null;
  afterEach(() => {
    t?.cleanup();
    t = null;
  });

  it("runs from empty DB and is idempotent", async () => {
    t = makeTestDb();
    await setupVecAndMigrate(t.db, 8);
    // Run again — must not throw.
    runMigrations(t.db, { embeddingDim: 8, embeddingModel: "test-fake" });
    runMigrations(t.db, { embeddingDim: 8, embeddingModel: "test-fake" });
    const tables = t.db
      .prepare("SELECT name FROM sqlite_master WHERE type='table' OR type='virtual table'")
      .all() as { name: string }[];
    const names = tables.map((r) => r.name);
    expect(names).toContain("memory_index");
    expect(names).toContain("memory_meta");
    expect(names).toContain("_pi_lcm_memory_schema_version");
  });

  it("coexists with pi-lcm tables", async () => {
    t = makeTestDb();
    applyPiLcmSchema(t.db);
    await setupVecAndMigrate(t.db, 8);
    const tables = t.db
      .prepare("SELECT name FROM sqlite_master WHERE type IN ('table','virtual table','view')")
      .all() as { name: string }[];
    const names = tables.map((r) => r.name);
    for (const expected of ["messages", "summaries", "memory_index", "memory_meta"]) {
      expect(names).toContain(expected);
    }
  });

  it("adds memory_index_msg/sum side tables (v1 → v2 backfill)", async () => {
    t = makeTestDb();
    // Apply v1 manually (older schema), then insert legacy rows that
    // would have been written before v2 existed.
    t.db.prepare(
      `CREATE TABLE _pi_lcm_memory_schema_version (
         version INTEGER NOT NULL, applied_at INTEGER NOT NULL DEFAULT (unixepoch())
       )`,
    ).run();
    t.db.prepare(
      `CREATE TABLE memory_index (
         vec_rowid INTEGER PRIMARY KEY,
         source_kind TEXT NOT NULL CHECK (source_kind IN ('message','summary')),
         content_hash TEXT NOT NULL UNIQUE,
         pi_lcm_msg_id TEXT, pi_lcm_sum_id TEXT, conversation_id TEXT,
         session_started INTEGER, role TEXT, depth INTEGER,
         snippet TEXT NOT NULL, text_full TEXT NOT NULL, token_count INTEGER,
         model_name TEXT NOT NULL, model_dims INTEGER NOT NULL,
         created_at INTEGER NOT NULL DEFAULT (unixepoch())
       )`,
    ).run();
    t.db.prepare(
      `CREATE TABLE memory_meta (k TEXT PRIMARY KEY, v TEXT NOT NULL)`,
    ).run();
    t.db.prepare("INSERT INTO _pi_lcm_memory_schema_version(version) VALUES (1)").run();

    // Two pre-v2 rows: one with a message id, one with a summary id, one
    // with neither (canonical content with no pi-lcm linkage).
    const insIdx = t.db.prepare(
      `INSERT INTO memory_index(vec_rowid, source_kind, content_hash,
         pi_lcm_msg_id, pi_lcm_sum_id, snippet, text_full, model_name, model_dims)
       VALUES (?, ?, ?, ?, ?, 'x', 'x', 'm', 8)`,
    );
    insIdx.run(1, "message", "h1", "msg-1", null);
    insIdx.run(2, "summary", "h2", null, "sum-1");
    insIdx.run(3, "message", "h3", null, null);

    // Now run the v2 migration.
    runMigrations(t.db, { embeddingDim: 8, embeddingModel: "m" });

    const tables = t.db
      .prepare("SELECT name FROM sqlite_master WHERE type='table'")
      .all() as { name: string }[];
    const names = tables.map((r) => r.name);
    expect(names).toContain("memory_index_msg");
    expect(names).toContain("memory_index_sum");

    const msgs = t.db
      .prepare("SELECT pi_lcm_msg_id, vec_rowid FROM memory_index_msg")
      .all() as { pi_lcm_msg_id: string; vec_rowid: number }[];
    expect(msgs).toEqual([{ pi_lcm_msg_id: "msg-1", vec_rowid: 1 }]);

    const sums = t.db
      .prepare("SELECT pi_lcm_sum_id, vec_rowid FROM memory_index_sum")
      .all() as { pi_lcm_sum_id: string; vec_rowid: number }[];
    expect(sums).toEqual([{ pi_lcm_sum_id: "sum-1", vec_rowid: 2 }]);

    // Re-running migrations is still idempotent.
    runMigrations(t.db, { embeddingDim: 8, embeddingModel: "m" });
    const ms2 = t.db.prepare("SELECT COUNT(*) n FROM memory_index_msg").get() as { n: number };
    expect(ms2.n).toBe(1);
  });

  it("rebuilds vec table when dim changes", async () => {
    t = makeTestDb();
    const v1 = await setupVecAndMigrate(t.db, 8);
    if (!v1.vecLoaded) return; // skip on platforms without sqlite-vec

    const meta1 = t.db.prepare("SELECT v FROM memory_meta WHERE k='embedding_dim'").get() as { v: string };
    expect(meta1.v).toBe("8");

    runMigrations(t.db, { embeddingDim: 16, embeddingModel: "test-fake" });
    const meta2 = t.db.prepare("SELECT v FROM memory_meta WHERE k='embedding_dim'").get() as { v: string };
    expect(meta2.v).toBe("16");

    // memory_index should be empty after dim change (sweep would re-embed).
    const r = t.db.prepare("SELECT COUNT(*) AS n FROM memory_index").get() as { n: number };
    expect(r.n).toBe(0);
  });
});
