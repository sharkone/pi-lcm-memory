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
    // Batches of 8 → expect 10 calls (8 × 10 = 80). Smaller batches let the
    // TUI render between calls; quantized weights make each call cheap.
    expect(emb.calls.length).toBeLessThan(80);
    expect(emb.calls.length).toBe(10);
    expect(emb.calls.reduce((a, b) => a + b, 0)).toBe(80);
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
