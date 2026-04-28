/**
 * Microbenchmark: insertBatch vs naive per-row insert.
 *
 * Skipped by default (timing tests are flaky in CI).
 *   PI_LCM_MEMORY_BENCH=1 npx vitest run test/insert.bench.test.ts
 */

import { describe, it, expect } from "vitest";
import { makeTestDb, setupVecAndMigrate, FakeEmbedder } from "./helpers.js";
import { MemoryStore, type InsertArgs } from "../src/db/store.js";
import { contentHash } from "../src/utils.js";
import { isVecLoadedFor } from "../src/db/vec.js";

const bench = process.env.PI_LCM_MEMORY_BENCH === "1";
const describeBench = bench ? describe : describe.skip;

describeBench("insertBatch vs single inserts", () => {
  it("batched insertion is significantly faster than per-row", async () => {
    const t = makeTestDb();
    try {
      await setupVecAndMigrate(t.db, 8);
      if (!isVecLoadedFor(t.db)) return;

      const store = new MemoryStore(t.db);
      const emb = new FakeEmbedder(8);

      // Build 200 items and pre-embed.
      const items: InsertArgs[] = [];
      for (let i = 0; i < 200; i++) {
        const text = `benchmark message #${i} with some unique content ${Math.random()}`;
        const [v] = await emb.embed(text);
        items.push({
          source_kind: "message",
          content_hash: contentHash("user", text, 8, "test-fake"),
          embedding: v!,
          conversation_id: "c1",
          session_started: i,
          role: "user",
          snippet: text,
          text_full: text,
          token_count: text.length,
          model_name: "test-fake",
          model_dims: 8,
        });
      }

      // Per-row: each insert is its own transaction.
      const t0 = Date.now();
      for (const it of items) store.insert(it);
      const perRowMs = Date.now() - t0;

      // Reset and try batched.
      store.clearAll();
      const t1 = Date.now();
      // Need fresh hashes since we cleared (idempotency cares about hash, but
      // we just removed them, so re-inserting is allowed).
      store.insertBatch(items);
      const batchMs = Date.now() - t1;

      // eslint-disable-next-line no-console
      console.log(
        `[bench] 200 inserts: per-row=${perRowMs}ms, batched=${batchMs}ms (${(perRowMs / batchMs).toFixed(1)}× speedup)`,
      );

      expect(store.stats().indexed).toBe(200);
      // We expect at least ~3× speedup; usually more like 10×+ in practice.
      expect(batchMs).toBeLessThan(perRowMs);
    } finally {
      t.cleanup();
    }
  });

  it("whichHashesPresent is faster than N hasContentHash calls", async () => {
    const t = makeTestDb();
    try {
      await setupVecAndMigrate(t.db, 8);
      if (!isVecLoadedFor(t.db)) return;

      const store = new MemoryStore(t.db);
      const emb = new FakeEmbedder(8);

      const items: InsertArgs[] = [];
      const hashes: string[] = [];
      for (let i = 0; i < 500; i++) {
        const text = `seed #${i}`;
        const [v] = await emb.embed(text);
        const h = contentHash("user", text, 8, "test-fake");
        items.push({
          source_kind: "message",
          content_hash: h,
          embedding: v!,
          conversation_id: "c1",
          session_started: i,
          role: "user",
          snippet: text,
          text_full: text,
          model_name: "test-fake",
          model_dims: 8,
        });
        hashes.push(h);
      }
      store.insertBatch(items);

      const t0 = Date.now();
      for (const h of hashes) store.hasContentHash(h);
      const oneByOne = Date.now() - t0;

      const t1 = Date.now();
      const present = store.whichHashesPresent(hashes);
      const bulk = Date.now() - t1;

      // eslint-disable-next-line no-console
      console.log(`[bench] 500 lookups: one-by-one=${oneByOne}ms, bulk=${bulk}ms`);
      expect(present.size).toBe(500);
      expect(bulk).toBeLessThan(oneByOne);
    } finally {
      t.cleanup();
    }
  });
});
