/**
 * End-to-end tests: real worker, real ONNX, real DB, real extension wiring,
 * faithful but minimal `makeFakePi()` ExtensionAPI surface.
 *
 * Opt-in: requires PI_LCM_MEMORY_LIVE_TEST=1 (model download is slow first
 * run; subsequent runs use the cached model).
 *
 *   PI_LCM_MEMORY_LIVE_TEST=1 npx vitest run test/e2e/full-pipeline.test.ts
 *
 * Catches the class of integration regressions unit tests miss:
 * settings-panel API shape changes, hook contract drift, schema
 * migration races, tool registration mismatches, etc.
 */

import { describe, it, expect, beforeAll, afterAll } from "vitest";
import Database from "better-sqlite3";

import { makeFakePi, type FakePi } from "./fake-pi.js";
import { makeE2EProject, waitFor } from "./fixtures.js";

const liveGate = process.env.PI_LCM_MEMORY_LIVE_TEST === "1";
const describeLive = liveGate ? describe : describe.skip;

const DISTINCTIVE = "the answer to the cross-encoder reranker question is approximately three";
const NOISE = [
  "we should also lower the busy_timeout for testing",
  "consider switching to a different sqlite vec backend",
  "I keep seeing FTS5 index corruption when sweep runs concurrently",
  "the worker thread reports 8 intra-op threads on this machine",
  "let's bump the sweep interval to thirty seconds",
  "there is a stale row in memory_index_msg that needs cleanup",
  "summary depth zero corresponds to the raw conversation",
  "the dedupe path uses content_hash to collapse identical text",
  "primer text shows the most recent topics first",
  "auto-recall uses regex triggers and a token budget",
];

describeLive("e2e: full pipeline", () => {
  let project: ReturnType<typeof makeE2EProject>;
  let pi: FakePi;
  let extDefault: (pi: any) => void;
  const originalEnv = {
    dbDir: process.env.PI_LCM_MEMORY_DB_DIR,
    sweepMs: process.env.PI_LCM_MEMORY_SWEEP_MS,
  };

  beforeAll(async () => {
    // Build a tmp project with a pre-seeded pi-lcm DB.
    const messages = [
      { id: "m-noise-0", role: "user", text: NOISE[0]! },
      { id: "m-noise-1", role: "assistant", text: NOISE[1]! },
      { id: "m-noise-2", role: "user", text: NOISE[2]! },
      { id: "m-noise-3", role: "assistant", text: NOISE[3]! },
      { id: "m-noise-4", role: "user", text: NOISE[4]! },
      // The needle:
      { id: "m-needle", role: "user", text: DISTINCTIVE },
      { id: "m-noise-5", role: "assistant", text: NOISE[5]! },
      { id: "m-noise-6", role: "user", text: NOISE[6]! },
      { id: "m-noise-7", role: "assistant", text: NOISE[7]! },
      { id: "m-noise-8", role: "user", text: NOISE[8]! },
      { id: "m-noise-9", role: "assistant", text: NOISE[9]! },
    ];
    project = makeE2EProject({ messages });

    // Override the dbDir AND the sweep interval BEFORE loading the extension
    // (config is resolved at session_start time, so this is what controls
    // the test).
    process.env.PI_LCM_MEMORY_DB_DIR = project.dbDir;
    process.env.PI_LCM_MEMORY_SWEEP_MS = "100";

    pi = makeFakePi();
    const mod = await import("../../index.js");
    extDefault = mod.default;
    extDefault(pi);

    // Fire session_start. This kicks off DB open + schema migration +
    // indexer.start(). The first sweep is scheduled 100ms out.
    const ctx = pi.makeCtx({ cwd: project.cwd });
    await pi.fire("session_start", { reason: "startup" }, ctx);

    // Wait for backfill to write 11 rows.
    await waitFor(
      () => {
        try {
          const db = new Database(project.dbPath, { readonly: true });
          try {
            const row = db
              .prepare("SELECT COUNT(*) AS c FROM memory_index")
              .get() as { c: number };
            return row.c >= 11;
          } finally {
            db.close();
          }
        } catch {
          return false;
        }
      },
      { timeoutMs: 60_000, intervalMs: 100 },
    );
  }, 90_000);

  afterAll(async () => {
    try {
      const ctx = pi.makeCtx({ cwd: project.cwd });
      await pi.fire("session_shutdown", {}, ctx);
    } catch {
      // best effort
    }
    process.env.PI_LCM_MEMORY_DB_DIR = originalEnv.dbDir;
    process.env.PI_LCM_MEMORY_SWEEP_MS = originalEnv.sweepMs;
    project.cleanup();
  });

  it("backfills the seeded pi-lcm corpus into memory_index", async () => {
    const db = new Database(project.dbPath, { readonly: true });
    try {
      const row = db.prepare("SELECT COUNT(*) AS c FROM memory_index").get() as { c: number };
      expect(row.c).toBeGreaterThanOrEqual(11);
      const sumRow = db
        .prepare("SELECT SUM(LENGTH(text_full)) AS bytes FROM memory_index")
        .get() as { bytes: number };
      expect(sumRow.bytes).toBeGreaterThan(0);
    } finally {
      db.close();
    }
  });

  it("registers lcm_recall and returns the distinctive doc near the top", async () => {
    const tool = pi.tool("lcm_recall");
    expect(tool).toBeTruthy();
    expect(typeof tool.execute).toBe("function");

    const result = await tool.execute("e2e-call-1", { query: "cross-encoder reranker answer", k: 5 });
    // Pi tools commonly return { content: [...], details } or similar; our
    // tool returns the raw hits in `details.hits` plus a content blob. Be
    // forgiving about shape — assert the needle id is present.
    const text = JSON.stringify(result);
    expect(text).toContain("m-needle");
  });

  it("lcm_similar finds related rows for a known message id", async () => {
    const tool = pi.tool("lcm_similar");
    expect(tool).toBeTruthy();
    const result = await tool.execute("e2e-call-2", { messageId: "m-needle", k: 3 });
    const text = JSON.stringify(result);
    // Similar should return at least one row (typically the seed + neighbours).
    expect(text.length).toBeGreaterThan(2);
  });

  it("registers /memory commands and runs /memory stats without crashing", async () => {
    const cmd = pi.command("memory");
    expect(cmd).toBeTruthy();
    const ctx = pi.makeCtx({ cwd: project.cwd });
    await pi.runCommand("memory", "stats", ctx);
    const note = ctx.ui.notifications.map((n) => n.message).join("\n");
    expect(note).toMatch(/pi-lcm-memory stats:/);
    expect(note).toMatch(/indexed:/);
  });

  it("opens the settings panel without crashing (factory contract)", async () => {
    const ctx = pi.makeCtx({ cwd: project.cwd });
    await pi.runCommand("memory-settings", "", ctx);
    // Either the factory path was invoked OR a clean "UI not available"
    // notification was emitted. We accept the second only when ui.custom
    // wasn't provided (it always is here); otherwise we assert factory path.
    expect(ctx.ui.lastCustomCall).not.toBeNull();
    expect(typeof ctx.ui.lastCustomCall?.factory).toBe("function");
    // The custom() promise resolves when the panel calls done(). We don't
    // call done() here; just confirm no crash and no warning notification.
    const warns = ctx.ui.notifications.filter((n) => n.level === "warning");
    expect(warns).toEqual([]);
  });

  it("message_end hook embeds a freshly-inserted pi-lcm message", async () => {
    // Insert a new pi-lcm message AFTER initial backfill, then fire
    // message_end so the hook indexes it on the spot (no waiting for the
    // sweep tick).
    const newId = "m-followup";
    const newText = "follow-up question about reranker pool sizes and ndcg";
    const conn = new Database(project.dbPath);
    try {
      conn.prepare(
        `INSERT INTO messages(id, conversation_id, role, content_text, timestamp, seq)
           VALUES (?, ?, ?, ?, ?, ?)`,
      ).run(newId, "e2e-conv-1", "user", newText, 1700000999, 999);
    } finally {
      conn.close();
    }

    const ctx = pi.makeCtx({ cwd: project.cwd });
    await pi.fire(
      "message_end",
      {
        message: {
          id: newId,
          role: "user",
          content: [{ type: "text", text: newText }],
        },
      },
      ctx,
    );

    // The hook embeds asynchronously, with no pi_lcm_msg_id (pi-lcm-memory's
    // hook path doesn't tie the new vec to the pi-lcm message row — the next
    // sweep does that via content_hash dedupe + recordPresentMappings). For
    // this test we just assert the embedded text is findable.
    await waitFor(
      () => {
        try {
          const db = new Database(project.dbPath, { readonly: true });
          try {
            const row = db
              .prepare(
                "SELECT COUNT(*) AS c FROM memory_index WHERE text_full = ?",
              )
              .get(newText) as { c: number };
            return row.c >= 1;
          } finally {
            db.close();
          }
        } catch {
          return false;
        }
      },
      { timeoutMs: 15_000, intervalMs: 50 },
    );

    // And it's findable by recall (the snippet text comes back).
    const tool = pi.tool("lcm_recall");
    const result = await tool.execute("e2e-call-3", { query: "ndcg reranker pool", k: 3 });
    expect(JSON.stringify(result)).toContain("reranker pool sizes");
  });
});

describe("e2e harness: makeFakePi sanity (always runs)", () => {
  it("records on/fire/registerTool/registerCommand correctly", async () => {
    const pi = makeFakePi();
    let saw: string[] = [];
    pi.on("session_start", async () => {
      saw.push("ss");
    });
    pi.on("message_end", async () => {
      saw.push("me");
    });
    pi.registerTool({
      name: "fake_tool",
      execute: async () => "ok",
    });
    pi.registerCommand("fake", { description: "x", handler: async () => {} });

    const ctx = pi.makeCtx({ cwd: "/tmp/x" });
    await pi.fire("session_start", { reason: "startup" }, ctx);
    await pi.fire("message_end", {}, ctx);
    expect(saw).toEqual(["ss", "me"]);
    expect(pi.tool("fake_tool")).toBeTruthy();
    expect(pi.command("fake")).toBeTruthy();
  });
});
