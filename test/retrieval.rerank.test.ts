/**
 * Unit tests for the cross-encoder rerank stage in `Retriever.recall()`.
 *
 * Uses a `FakeRerankerEmbedder` that intercepts `.rerank()` calls and returns
 * deterministic scores driven by a caller-supplied scoring function. This
 * isolates the rerank wiring from the real cross-encoder model so we can:
 *   - assert the reranker is called when config.rerank is true,
 *   - assert it is NOT called when config.rerank is false,
 *   - assert hybrid order is replaced by reranker order,
 *   - assert score-mismatch / thrown errors fall through to hybrid order,
 *   - assert the per-call `params.rerank` override beats config.
 */

import { describe, it, expect, afterEach } from "vitest";
import {
  makeTestDb,
  applyPiLcmSchema,
  setupVecAndMigrate,
  seedMessage,
  FakeEmbedder,
  type TestDb,
} from "./helpers.js";
import { MemoryStore, type InsertArgs } from "../src/db/store.js";
import { PiLcmBridge } from "../src/bridge.js";
import { Retriever } from "../src/retrieval.js";
import { contentHash } from "../src/utils.js";
import { isVecLoaded } from "../src/db/vec.js";

/**
 * Wraps a FakeEmbedder and adds a `rerank(query, docs)` shim that returns
 * scores derived from `scorer`. Keeps a count of calls so tests can assert
 * the rerank stage was (or wasn't) invoked.
 */
class FakeRerankerEmbedder extends FakeEmbedder {
  rerankCalls = 0;
  scorer: (query: string, doc: string, idx: number) => number = () => 0;
  shouldThrow = false;
  shouldShortReturn = false;

  async rerank(query: string, docs: string[]): Promise<number[]> {
    this.rerankCalls++;
    if (this.shouldThrow) throw new Error("boom");
    if (this.shouldShortReturn) return docs.slice(0, Math.max(0, docs.length - 1)).map(() => 0);
    return docs.map((d, i) => this.scorer(query, d, i));
  }
}

describe("Retriever.recall() with cross-encoder rerank", () => {
  let t: TestDb | null = null;
  afterEach(() => {
    t?.cleanup();
    t = null;
  });

  async function buildCorpus(): Promise<{
    retriever: Retriever;
    emb: FakeRerankerEmbedder;
    rerankFlag: { value: boolean };
  } | null> {
    t = makeTestDb();
    applyPiLcmSchema(t.db);
    await setupVecAndMigrate(t.db, 8);
    if (!isVecLoaded()) return null;

    const messages = [
      { id: "m1", text: "auth middleware refactor for the gateway" },
      { id: "m2", text: "JWT verification bug in the auth pipeline" },
      { id: "m3", text: "completely unrelated note about bookkeeping" },
      { id: "m4", text: "another auth issue with cookie handling" },
      { id: "m5", text: "performance bench for sqlite vector search" },
      { id: "m6", text: "auth retry logic was missing on 401 responses" },
    ];

    const emb = new FakeRerankerEmbedder(8);
    const store = new MemoryStore(t.db);
    for (const m of messages) {
      seedMessage(t.db, { id: m.id, conv: "c1", role: "user", text: m.text, ts: 1, seq: 0 });
      const [v] = await emb.embed(m.text);
      const args: InsertArgs = {
        source_kind: "message",
        content_hash: contentHash("user", m.text, 8, "test-fake"),
        embedding: v!,
        conversation_id: "c1",
        session_started: 1,
        role: "user",
        snippet: m.text,
        text_full: m.text,
        token_count: m.text.length,
        model_name: "test-fake",
        model_dims: 8,
        pi_lcm_msg_id: m.id,
      };
      store.insert(args);
    }

    const rerankFlag = { value: false };
    const retriever = new Retriever({
      db: t.db,
      store,
      embedder: emb as never,
      bridge: new PiLcmBridge(t.db),
      rrfK: 60,
      rerankEnabled: () => rerankFlag.value,
      rerankPoolSize: () => 5,
    });
    return { retriever, emb, rerankFlag };
  }

  it("does NOT call rerank when config.rerank is false", async () => {
    const built = await buildCorpus();
    if (!built) return; // sqlite-vec not available
    const { retriever, emb } = built;
    const hits = await retriever.recall({ query: "auth", k: 3 });
    expect(hits.length).toBeGreaterThan(0);
    expect(emb.rerankCalls).toBe(0);
  });

  it("calls rerank and reorders results when config.rerank is true", async () => {
    const built = await buildCorpus();
    if (!built) return;
    const { retriever, emb, rerankFlag } = built;
    rerankFlag.value = true;

    // Score by length descending: longest doc wins. m6 ("auth retry logic was
    // missing on 401 responses") is the longest among "auth"-matching rows.
    emb.scorer = (_q, doc) => doc.length;

    const hits = await retriever.recall({ query: "auth", k: 3 });
    expect(emb.rerankCalls).toBe(1);
    expect(hits.length).toBeGreaterThan(0);
    // Check ordering: rerank_score should be monotonically non-increasing.
    const scores = hits.map((h) => h.rerank_score ?? -Infinity);
    for (let i = 1; i < scores.length; i++) {
      expect(scores[i]).toBeLessThanOrEqual(scores[i - 1]!);
    }
    // Top hit should be the longest auth-tagged doc the hybrid stage saw.
    expect(hits[0]!.pi_lcm_msg_id).toBe("m6");
  });

  it("`params.rerank` overrides `rerankEnabled`", async () => {
    const built = await buildCorpus();
    if (!built) return;
    const { retriever, emb, rerankFlag } = built;
    rerankFlag.value = false; // config off
    emb.scorer = (_q, doc, i) => -i; // Reversed-from-input order.

    const hits = await retriever.recall({ query: "auth", k: 3, rerank: true });
    expect(emb.rerankCalls).toBe(1);
    expect(hits.length).toBeGreaterThan(0);
  });

  it("falls through to hybrid order when rerank throws", async () => {
    const built = await buildCorpus();
    if (!built) return;
    const { retriever, emb, rerankFlag } = built;
    rerankFlag.value = true;
    emb.shouldThrow = true;

    const noRerank = await retriever.recall({ query: "auth", k: 3, rerank: false });
    const withRerank = await retriever.recall({ query: "auth", k: 3 });
    expect(emb.rerankCalls).toBeGreaterThan(0);
    // Hybrid order should be preserved when reranker throws — same ids, same order.
    expect(withRerank.map((h) => h.pi_lcm_msg_id)).toEqual(noRerank.map((h) => h.pi_lcm_msg_id));
  });

  it("falls through when rerank returns the wrong number of scores", async () => {
    const built = await buildCorpus();
    if (!built) return;
    const { retriever, emb, rerankFlag } = built;
    rerankFlag.value = true;
    emb.shouldShortReturn = true;

    const noRerank = await retriever.recall({ query: "auth", k: 3, rerank: false });
    const withRerank = await retriever.recall({ query: "auth", k: 3 });
    expect(withRerank.map((h) => h.pi_lcm_msg_id)).toEqual(noRerank.map((h) => h.pi_lcm_msg_id));
  });
});
