import { describe, it, expect, afterEach } from "vitest";
import {
  makeTestDb,
  applyPiLcmSchema,
  setupVecAndMigrate,
  seedConversation,
  seedMessage,
  seedSummary,
  type TestDb,
} from "./helpers.js";
import { PiLcmBridge } from "../src/bridge.js";

describe("PiLcmBridge", () => {
  let t: TestDb | null = null;
  afterEach(() => {
    t?.cleanup();
    t = null;
  });

  it("degrades gracefully when pi-lcm tables are missing", async () => {
    t = makeTestDb();
    await setupVecAndMigrate(t.db, 8);
    const b = new PiLcmBridge(t.db);
    expect(b.available().messages).toBe(false);
    expect(b.totalSessions()).toBe(0);
    expect(b.lastSessionStart()).toBeNull();
    expect([...b.messagesNotInMemoryIndex(10)]).toEqual([]);
    expect(b.ftsSearch("foo", 5)).toEqual([]);
  });

  it("yields un-indexed messages and summaries; respects join", async () => {
    t = makeTestDb();
    applyPiLcmSchema(t.db);
    await setupVecAndMigrate(t.db, 8);

    seedConversation(t.db, { id: "c1", created_at: "2026-04-20T00:00:00Z" });
    seedConversation(t.db, { id: "c2", created_at: "2026-04-25T00:00:00Z" });
    seedMessage(t.db, { id: "m1", conv: "c1", role: "user", text: "alpha auth bug", ts: 1, seq: 0 });
    seedMessage(t.db, { id: "m2", conv: "c1", role: "assistant", text: "fixed JWT validation", ts: 2, seq: 1 });
    seedMessage(t.db, { id: "m3", conv: "c2", role: "user", text: "renaming variables", ts: 3, seq: 0 });
    seedSummary(t.db, { id: "s1", conv: "c1", depth: 1, text: "Fixed auth middleware", created_at: "2026-04-22T00:00:00Z" });
    seedSummary(t.db, { id: "s2", conv: "c1", depth: 2, text: "Major refactor of auth", created_at: "2026-04-23T00:00:00Z" });

    const b = new PiLcmBridge(t.db);
    expect(b.available().messages).toBe(true);
    expect(b.totalSessions()).toBe(2);
    expect(b.lastSessionStart()?.startsWith("2026-04-25")).toBe(true);

    const msgs = [...b.messagesNotInMemoryIndex(10)];
    expect(msgs.map((m) => m.id).sort()).toEqual(["m1", "m2", "m3"]);

    // After "indexing" m1, we should no longer see it. The bridge now
    // LEFT JOINs against memory_index_msg (the many-to-one mapping), so
    // we have to populate that table too — simulating what insertBatch does.
    t.db.prepare(
      `INSERT INTO memory_index(vec_rowid, source_kind, content_hash, pi_lcm_msg_id, snippet, text_full, model_name, model_dims)
       VALUES (?, 'message', ?, ?, ?, ?, 'test-fake', 8)`,
    ).run(1, "h-m1", "m1", "alpha auth bug", "alpha auth bug");
    t.db.prepare(
      "INSERT INTO memory_index_msg(pi_lcm_msg_id, vec_rowid) VALUES (?, ?)",
    ).run("m1", 1);

    const remaining = [...b.messagesNotInMemoryIndex(10)].map((m) => m.id);
    expect(remaining).not.toContain("m1");
    expect(remaining.sort()).toEqual(["m2", "m3"]);

    const sums = [...b.summariesNotInMemoryIndex(10)];
    expect(sums.map((s) => s.id).sort()).toEqual(["s1", "s2"]);

    // Recent summaries with depth >= 1
    const recent = b.recentSummaries(10, 1);
    expect(recent.map((r) => r.id)).toEqual(["s2", "s1"]);
  });

  it("ftsSearch returns ranked ids", async () => {
    t = makeTestDb();
    applyPiLcmSchema(t.db);
    await setupVecAndMigrate(t.db, 8);
    seedMessage(t.db, { id: "m1", conv: "c", role: "user", text: "auth middleware refactored", ts: 1, seq: 0 });
    seedMessage(t.db, { id: "m2", conv: "c", role: "user", text: "we renamed a button", ts: 2, seq: 1 });
    seedMessage(t.db, { id: "m3", conv: "c", role: "user", text: "JWT validation in auth", ts: 3, seq: 2 });

    const b = new PiLcmBridge(t.db);
    const hits = b.ftsSearch("auth*", 5);
    const ids = hits.map((h) => h.id);
    expect(ids).toContain("m1");
    expect(ids).toContain("m3");
    expect(ids).not.toContain("m2");
  });
});
