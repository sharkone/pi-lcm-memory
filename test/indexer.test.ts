import { describe, it, expect, afterEach } from "vitest";
import {
  makeTestDb,
  applyPiLcmSchema,
  setupVecAndMigrate,
  seedConversation,
  seedMessage,
  seedSummary,
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

describe("Indexer", () => {
  let t: TestDb | null = null;
  afterEach(() => {
    t?.cleanup();
    t = null;
  });

  it("sweep tick embeds un-indexed messages and summaries", async () => {
    t = makeTestDb();
    applyPiLcmSchema(t.db);
    await setupVecAndMigrate(t.db, 8);
    if (!isVecLoaded()) return;

    seedConversation(t.db, { id: "c1" });
    seedMessage(t.db, { id: "m1", conv: "c1", role: "user", text: "hello world", ts: 1, seq: 0 });
    seedMessage(t.db, { id: "m2", conv: "c1", role: "assistant", text: "general kenobi", ts: 2, seq: 1 });
    seedMessage(t.db, { id: "m3", conv: "c1", role: "toolResult", text: "BIG TOOL OUTPUT", ts: 3, seq: 2 });
    seedSummary(t.db, { id: "s1", conv: "c1", depth: 1, text: "user/assistant exchange" });

    const store = new MemoryStore(t.db);
    const bridge = new PiLcmBridge(t.db);
    const emb = new FakeEmbedder(8);
    const idx = new Indexer({
      store,
      embedder: emb as any,
      bridge,
      config: baseConfig,
      conversationId: () => "c1",
      sessionStartedAt: () => 1,
    });

    await idx.tick();
    const stats = store.stats();
    // Tool I/O message m3 should be skipped, leaving m1 + m2 + s1 = 3.
    expect(stats.indexed).toBe(3);
    expect(stats.byMessage).toBe(2);
    expect(stats.bySummary).toBe(1);

    // Idempotent: ticking again must not increase counts.
    await idx.tick();
    expect(store.stats().indexed).toBe(3);
  });

  it("handleMessage path embeds via the inflight chain", async () => {
    t = makeTestDb();
    applyPiLcmSchema(t.db);
    await setupVecAndMigrate(t.db, 8);
    if (!isVecLoaded()) return;

    const store = new MemoryStore(t.db);
    const bridge = new PiLcmBridge(t.db);
    const emb = new FakeEmbedder(8);
    const idx = new Indexer({
      store,
      embedder: emb as any,
      bridge,
      config: baseConfig,
      conversationId: () => "c1",
      sessionStartedAt: () => 1,
    });

    idx.handleMessage({ role: "user", content: "the user said hi" });
    idx.handleMessage({ role: "assistant", content: [{ type: "text", text: "the assistant replied" }] });
    idx.handleMessage({ role: "toolResult", content: [{ type: "text", text: "ignored noise" }] });

    await idx.drain();
    const stats = store.stats();
    expect(stats.byMessage).toBe(2);
    expect(stats.bySummary).toBe(0);
  });
});
