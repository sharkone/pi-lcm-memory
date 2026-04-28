import { describe, it, expect } from "vitest";
import { shouldFire, DEFAULT_TRIGGER, maybeAutoRecall } from "../src/auto-recall.js";
import type { Retriever, RecallHit } from "../src/retrieval.js";

describe("auto-recall trigger", () => {
  it("fires on memory phrases (heuristic)", () => {
    expect(shouldFire("can you remember the auth refactor?", "heuristic", DEFAULT_TRIGGER)).toBe(true);
    expect(shouldFire("Earlier we changed the login flow", "heuristic", DEFAULT_TRIGGER)).toBe(true);
    expect(shouldFire("we have already discussed this", "heuristic", DEFAULT_TRIGGER)).toBe(true);
    expect(shouldFire("like last time, please align it", "heuristic", DEFAULT_TRIGGER)).toBe(true);
    expect(shouldFire("apply the same approach as before", "heuristic", DEFAULT_TRIGGER)).toBe(true);
  });

  it("does not fire on unrelated text (heuristic)", () => {
    expect(shouldFire("write a function that sums numbers", "heuristic", DEFAULT_TRIGGER)).toBe(false);
    expect(shouldFire("", "heuristic", DEFAULT_TRIGGER)).toBe(false);
  });

  it("modes: off / always", () => {
    expect(shouldFire("anything", "off", DEFAULT_TRIGGER)).toBe(false);
    expect(shouldFire("anything", "always", DEFAULT_TRIGGER)).toBe(true);
    expect(shouldFire("", "always", DEFAULT_TRIGGER)).toBe(true);
  });
});

describe("maybeAutoRecall", () => {
  function fakeRetriever(hits: RecallHit[]): Retriever {
    return { recall: async () => hits } as unknown as Retriever;
  }

  it("returns null when mode=off", async () => {
    const out = await maybeAutoRecall("remember the auth bug", {
      getRetriever: () => fakeRetriever([]),
      mode: () => "off",
      topK: () => 3,
      tokenBudget: () => 600,
    });
    expect(out).toBeNull();
  });

  it("returns null when no hits", async () => {
    const out = await maybeAutoRecall("remember the auth bug", {
      getRetriever: () => fakeRetriever([]),
      mode: () => "heuristic",
      topK: () => 3,
      tokenBudget: () => 600,
    });
    expect(out).toBeNull();
  });

  it("renders block, respects token budget", async () => {
    const longHit: RecallHit = {
      source_kind: "message",
      conversation_id: "c1",
      session_started: 1745000000,
      role: "assistant",
      depth: null,
      snippet: "x".repeat(800),
      text_full: "x".repeat(800),
      score: 0.9,
      pi_lcm_msg_id: "m1",
      pi_lcm_sum_id: null,
    };
    const hits: RecallHit[] = [longHit, longHit, longHit];
    const out = await maybeAutoRecall("remember", {
      getRetriever: () => fakeRetriever(hits),
      mode: () => "heuristic",
      topK: () => 3,
      tokenBudget: () => 400,
    });
    expect(out).not.toBeNull();
    expect(out!).toContain("## Recall");
    // Budget enforcement: shouldn't blast through 400 tokens by huge margin.
    expect(out!.length).toBeLessThan(400 * 6);
  });
});
