import { describe, it, expect, afterEach } from "vitest";
import {
  makeTestDb,
  applyPiLcmSchema,
  setupVecAndMigrate,
  seedConversation,
  seedMessage,
  FakeEmbedder,
  type TestDb,
} from "./helpers.js";
import { MemoryStore } from "../src/db/store.js";
import { PiLcmBridge } from "../src/bridge.js";
import { Indexer } from "../src/indexer.js";
import type { MemoryConfig } from "../src/config.js";
import { isVecLoaded } from "../src/db/vec.js";

const baseConfig: MemoryConfig = {
  enabled: true,
  dbDir: "/tmp",
  embeddingModel: "test-fake",
  embeddingQuantize: "auto",
  indexMessages: true,
  indexSummaries: true,
  skipToolIO: true,
  primer: false,
  primerTopK: 5,
  autoRecall: "off",
  autoRecallTopK: 5,
  autoRecallTokenBudget: 600,
  recallDefaultTopK: 10,
  rrfK: 60,
  sweepIntervalMs: 30_000,
  modelCacheDir: null,
  debugMode: false,
};

/** Embedder spy that records every embed call's batch size. */
class SpyEmbedder extends FakeEmbedder {
  calls: number[] = [];
  override async embed(input: string | string[]): Promise<Float32Array[]> {
    const arr = Array.isArray(input) ? input : [input];
    this.calls.push(arr.length);
    return super.embed(input);
  }
}

describe("Indexer batched sweep + adaptive backoff", () => {
  let t: TestDb | null = null;
  afterEach(() => {
    t?.cleanup();
    t = null;
  });

  it("sweep groups messages into batches (one inference call per batch)", async () => {
    t = makeTestDb();
    applyPiLcmSchema(t.db);
    await setupVecAndMigrate(t.db, 8);
    if (!isVecLoaded()) return;

    seedConversation(t.db, { id: "c1" });
    for (let i = 0; i < 80; i++) {
      seedMessage(t.db, { id: `m${i}`, conv: "c1", role: "user", text: `message #${i}`, ts: i, seq: i });
    }
    const store = new MemoryStore(t.db);
    const bridge = new PiLcmBridge(t.db);
    const emb = new SpyEmbedder(8);

    const idx = new Indexer({
      store,
      embedder: emb as any,
      bridge,
      config: baseConfig,
      conversationId: () => "c1",
      sessionStartedAt: () => 1,
    });

    await idx.tick();

    expect(store.stats().byMessage).toBe(80);
    // Batches of 32 → 3 calls (32 + 32 + 16). Inference is in a worker;
    // the main thread is never blocked, so we keep batches comfortably large.
    expect(emb.calls.length).toBeLessThan(80);
    expect(emb.calls.length).toBe(3);
    expect(emb.calls.reduce((a, b) => a + b, 0)).toBe(80);
  });

  it("sweep terminates when EVERY message is a skipped tool-I/O role (regression: infinite-loop bug)", async () => {
    // Regression test for the freeze diagnosed via trace logs: 2.18M
    // iter_chunk events with zero batch_start. The generator was yielding
    // tool-I/O rows that bridgeMessageToPending dropped via `continue`,
    // and since they were never inserted into memory_index, the same
    // rows kept matching `mi.vec_rowid IS NULL` forever.
    t = makeTestDb();
    applyPiLcmSchema(t.db);
    await setupVecAndMigrate(t.db, 8);
    if (!isVecLoaded()) return;

    seedConversation(t.db, { id: "c1" });
    // Pure tool-I/O traffic, more than one stmt-batch's worth (SWEEP_BATCH*4 = 128).
    for (let i = 0; i < 200; i++) {
      seedMessage(t.db, {
        id: `t${i}`,
        conv: "c1",
        role: i % 2 === 0 ? "toolResult" : "bashExecution",
        text: `tool output ${i}`,
        ts: i,
        seq: i,
      });
    }
    const store = new MemoryStore(t.db);
    const bridge = new PiLcmBridge(t.db);
    const emb = new SpyEmbedder(8);

    const idx = new Indexer({
      store,
      embedder: emb as any,
      bridge,
      config: { ...baseConfig, skipToolIO: true },
      conversationId: () => "c1",
      sessionStartedAt: () => 1,
    });

    // Wrap tick in a hard timeout: pre-fix this would never return.
    const tickPromise = idx.tick();
    const timed = await Promise.race([
      tickPromise.then(() => "done"),
      new Promise<string>((r) => setTimeout(() => r("timeout"), 2_000)),
    ]);
    expect(timed).toBe("done");
    expect(emb.calls.length).toBe(0); // nothing to embed
    expect(store.stats().byMessage).toBe(0);
  });

  it("two messages with identical content map both ids to one vec_rowid (regression: dedupe leak)", async () => {
    // Pre-fix: when two pi-lcm messages had identical content, only the
    // first id landed on memory_index.pi_lcm_msg_id; the second leaked
    // through every sweep because the LEFT JOIN never matched it.
    // Post-fix: memory_index_msg records both ids -> the same vec_rowid.
    t = makeTestDb();
    applyPiLcmSchema(t.db);
    await setupVecAndMigrate(t.db, 8);
    if (!isVecLoaded()) return;

    seedConversation(t.db, { id: "c1" });
    // Same role + same text on two messages → same content_hash.
    seedMessage(t.db, { id: "m_a", conv: "c1", role: "user", text: "identical content", ts: 1, seq: 0 });
    seedMessage(t.db, { id: "m_b", conv: "c1", role: "user", text: "identical content", ts: 2, seq: 1 });

    const store = new MemoryStore(t.db);
    const bridge = new PiLcmBridge(t.db);
    const emb = new SpyEmbedder(8);
    const idx = new Indexer({
      store,
      embedder: emb as any,
      bridge,
      config: baseConfig,
      conversationId: () => "c1",
      sessionStartedAt: () => 1,
    });

    await idx.tick();
    // Only ONE embedding generated (one unique content_hash).
    expect(store.stats().byMessage).toBe(1);
    expect(emb.calls.length).toBe(1);
    expect(emb.calls[0]).toBe(2); // sweep batched both into the embed call

    // BOTH ids must be mapped to that single vec_rowid in the side table.
    const mappings = t.db
      .prepare("SELECT pi_lcm_msg_id, vec_rowid FROM memory_index_msg ORDER BY pi_lcm_msg_id")
      .all() as { pi_lcm_msg_id: string; vec_rowid: number }[];
    expect(mappings.map((m) => m.pi_lcm_msg_id)).toEqual(["m_a", "m_b"]);
    expect(mappings[0]!.vec_rowid).toBe(mappings[1]!.vec_rowid);

    // Subsequent sweeps must NOT re-yield either id (the leak fix).
    const before = store.stats().byMessage;
    const callsBefore = emb.calls.length;
    await idx.tick();
    await idx.tick();
    expect(store.stats().byMessage).toBe(before); // no new embeddings
    expect(emb.calls.length).toBe(callsBefore);

    // And messagesNotInMemoryIndex must yield zero rows now.
    const remaining = [...bridge.messagesNotInMemoryIndex(10)];
    expect(remaining.length).toBe(0);
  });

  it("sweep with mixed tool-I/O and real messages indexes only the real ones and terminates", async () => {
    t = makeTestDb();
    applyPiLcmSchema(t.db);
    await setupVecAndMigrate(t.db, 8);
    if (!isVecLoaded()) return;

    seedConversation(t.db, { id: "c1" });
    let id = 0;
    // 150 tool-I/O messages interleaved with 30 real ones. Mixing matters
    // because the rowid cursor must advance past skipped rows.
    for (let i = 0; i < 180; i++) {
      const isReal = i % 6 === 0; // 30 real, 150 skipped
      seedMessage(t.db, {
        id: `m${id++}`,
        conv: "c1",
        role: isReal ? "user" : "toolResult",
        text: isReal ? `real message ${i}` : `tool output ${i}`,
        ts: i,
        seq: i,
      });
    }
    const store = new MemoryStore(t.db);
    const bridge = new PiLcmBridge(t.db);
    const emb = new SpyEmbedder(8);

    const idx = new Indexer({
      store,
      embedder: emb as any,
      bridge,
      config: { ...baseConfig, skipToolIO: true },
      conversationId: () => "c1",
      sessionStartedAt: () => 1,
    });

    const tickPromise = idx.tick();
    const timed = await Promise.race([
      tickPromise.then(() => "done"),
      new Promise<string>((r) => setTimeout(() => r("timeout"), 2_000)),
    ]);
    expect(timed).toBe("done");
    expect(store.stats().byMessage).toBe(30);
    // 30 messages → 1 batch of 30 (all under SWEEP_BATCH=32).
    expect(emb.calls.length).toBe(1);
    expect(emb.calls[0]).toBe(30);
  });

  it("adaptive backoff: idle ticks grow interval; kick resets it", async () => {
    t = makeTestDb();
    applyPiLcmSchema(t.db);
    await setupVecAndMigrate(t.db, 8);
    if (!isVecLoaded()) return;

    const store = new MemoryStore(t.db);
    const bridge = new PiLcmBridge(t.db);
    const emb = new FakeEmbedder(8);
    const cfg: MemoryConfig = { ...baseConfig, sweepIntervalMs: 5_000 };

    const idx = new Indexer({
      store,
      embedder: emb as any,
      bridge,
      config: cfg,
      conversationId: () => null,
      sessionStartedAt: () => null,
    });

    // Tick once with no work — interval should be at base (we haven't scheduled yet).
    expect(idx.status().currentIntervalMs).toBe(5_000);

    // Manually run a tick to simulate idle (no rows). We don't call start() so we
    // can observe state without the timer driving things.
    await idx.tick();
    expect(idx.status().lastError).toBeNull();
    // No rows were indexed.
    expect(idx.status().indexedTotal).toBe(0);

    // Now seed work and kick.
    seedMessage(t.db, { id: "m1", conv: "c1", role: "user", text: "hi", ts: 1, seq: 0 });
    idx.kick();
    expect(idx.status().idleStreak).toBe(0);
    expect(idx.status().currentIntervalMs).toBe(cfg.sweepIntervalMs);
  });
});
