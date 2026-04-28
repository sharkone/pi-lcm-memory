import { describe, it, expect, afterEach } from "vitest";
import { makeTestDb, setupVecAndMigrate, FakeEmbedder, type TestDb } from "./helpers.js";
import { MemoryStore } from "../src/db/store.js";
import { contentHash } from "../src/utils.js";
import { isVecLoaded } from "../src/db/vec.js";

describe("MemoryStore", () => {
  let t: TestDb | null = null;
  afterEach(() => {
    t?.cleanup();
    t = null;
  });

  it("insert + dedup by content_hash; knn returns the seed", async () => {
    t = makeTestDb();
    await setupVecAndMigrate(t.db, 8);
    if (!isVecLoaded()) return; // gracefully skip on unsupported platforms

    const store = new MemoryStore(t.db);
    const emb = new FakeEmbedder(8);
    const txt = "hello world";
    const [vec] = await emb.embed(txt);

    const ch = contentHash("user", txt, 8, "test-fake");
    const id1 = store.insert({
      source_kind: "message",
      content_hash: ch,
      embedding: vec!,
      role: "user",
      conversation_id: "c1",
      session_started: 1,
      snippet: "hello world",
      text_full: txt,
      model_name: "test-fake",
      model_dims: 8,
    });
    expect(id1).not.toBeNull();

    // Second insert with same hash → returns same vec_rowid, no duplicate row.
    const id2 = store.insert({
      source_kind: "message",
      content_hash: ch,
      embedding: vec!,
      snippet: "hello world",
      text_full: txt,
      model_name: "test-fake",
      model_dims: 8,
    });
    expect(id2).toBe(id1);

    const stats = store.stats();
    expect(stats.indexed).toBe(1);
    expect(stats.byMessage).toBe(1);
    expect(stats.modelDims).toBe(8);

    // knn finds the seed first.
    const knn = store.knn(vec!, 3);
    expect(knn.length).toBeGreaterThanOrEqual(1);
    expect(knn[0]!.vec_rowid).toBe(id1);
  });

  it("rejects dim mismatch", async () => {
    t = makeTestDb();
    await setupVecAndMigrate(t.db, 8);
    if (!isVecLoaded()) return;
    const store = new MemoryStore(t.db);
    const wrong = new Float32Array(4); // wrong dim
    expect(() =>
      store.insert({
        source_kind: "message",
        content_hash: "h",
        embedding: wrong,
        snippet: "x",
        text_full: "x",
        model_name: "test-fake",
        model_dims: 8,
      }),
    ).toThrow();
  });

  it("clearAll empties index + vec", async () => {
    t = makeTestDb();
    await setupVecAndMigrate(t.db, 8);
    if (!isVecLoaded()) return;
    const store = new MemoryStore(t.db);
    const emb = new FakeEmbedder(8);
    for (const word of ["alpha", "beta", "gamma"]) {
      const [v] = await emb.embed(word);
      store.insert({
        source_kind: "message",
        content_hash: "h-" + word,
        embedding: v!,
        snippet: word,
        text_full: word,
        model_name: "test-fake",
        model_dims: 8,
      });
    }
    expect(store.stats().indexed).toBe(3);
    store.clearAll();
    expect(store.stats().indexed).toBe(0);
    expect(store.stats().vecRows).toBe(0);
  });
});
