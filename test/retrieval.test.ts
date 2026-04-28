import { describe, it, expect, afterEach } from "vitest";
import {
  makeTestDb,
  applyPiLcmSchema,
  setupVecAndMigrate,
  seedMessage,
  FakeEmbedder,
  type TestDb,
} from "./helpers.js";
import { MemoryStore } from "../src/db/store.js";
import { PiLcmBridge } from "../src/bridge.js";
import { Retriever } from "../src/retrieval.js";
import { contentHash } from "../src/utils.js";
import { isVecLoaded } from "../src/db/vec.js";

describe("Retriever (hybrid + RRF)", () => {
  let t: TestDb | null = null;
  afterEach(() => {
    t?.cleanup();
    t = null;
  });

  async function buildCorpus(): Promise<{ store: MemoryStore; retriever: Retriever; emb: FakeEmbedder } | null> {
    t = makeTestDb();
    applyPiLcmSchema(t.db);
    await setupVecAndMigrate(t.db, 8);
    if (!isVecLoaded()) return null;

    const messages = [
      { id: "m1", text: "fixed auth middleware bug" },
      { id: "m2", text: "renamed button label in settings" },
      { id: "m3", text: "JWT validation issue resolved" },
      { id: "m4", text: "added new logging system" },
      { id: "m5", text: "auth flow refactor with cookies" },
    ];
    const emb = new FakeEmbedder(8);
    const store = new MemoryStore(t.db);

    for (const m of messages) {
      seedMessage(t!.db, { id: m.id, conv: "c1", role: "user", text: m.text, ts: 1, seq: 0 });
      const [v] = await emb.embed(m.text);
      store.insert({
        source_kind: "message",
        content_hash: contentHash("user", m.text, 8, "test-fake"),
        embedding: v!,
        pi_lcm_msg_id: m.id,
        conversation_id: "c1",
        session_started: 1,
        role: "user",
        snippet: m.text,
        text_full: m.text,
        model_name: "test-fake",
        model_dims: 8,
      });
    }

    const bridge = new PiLcmBridge(t!.db);
    const retriever = new Retriever({ db: t!.db, store, embedder: emb as any, bridge, rrfK: 60 });
    return { store, retriever, emb };
  }

  it("lexical mode finds string matches via FTS5", async () => {
    const ctx = await buildCorpus();
    if (!ctx) return;
    const hits = await ctx.retriever.recall({ query: "auth", k: 3, mode: "lexical" });
    expect(hits.length).toBeGreaterThan(0);
    const ids = hits.map((h) => h.pi_lcm_msg_id);
    expect(ids).toContain("m1");
  });

  it("semantic mode returns at least the seed for an exact-text query", async () => {
    const ctx = await buildCorpus();
    if (!ctx) return;
    const hits = await ctx.retriever.recall({
      query: "JWT validation issue resolved",
      k: 3,
      mode: "semantic",
    });
    expect(hits.length).toBeGreaterThan(0);
    expect(hits[0]!.pi_lcm_msg_id).toBe("m3");
  });

  it("hybrid mode merges via RRF and keeps both winners", async () => {
    const ctx = await buildCorpus();
    if (!ctx) return;
    const hits = await ctx.retriever.recall({ query: "auth", k: 5, mode: "hybrid" });
    const ids = hits.map((h) => h.pi_lcm_msg_id);
    // m1 and m5 are the auth-related rows; both must appear.
    expect(ids).toContain("m1");
    expect(ids).toContain("m5");
  });

  it("session filter restricts results", async () => {
    const ctx = await buildCorpus();
    if (!ctx) return;
    const hits = await ctx.retriever.recall({
      query: "auth",
      k: 5,
      mode: "hybrid",
      sessionFilter: "no-such-conv",
    });
    expect(hits).toEqual([]);
  });

  it("similar() returns neighbours of a known seed", async () => {
    const ctx = await buildCorpus();
    if (!ctx) return;
    const hits = await ctx.retriever.similar({ messageId: "m1" }, 2);
    expect(hits.length).toBeGreaterThan(0);
    expect(hits.every((h) => h.pi_lcm_msg_id !== "m1")).toBe(true);
  });
});
