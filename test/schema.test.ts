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
