import { describe, it, expect } from "vitest";
import {
  hashCwd,
  contentHash,
  extractIndexableText,
  makeSnippet,
  estimateTokens,
} from "../src/utils.js";

describe("utils", () => {
  it("hashCwd is stable and 16 hex chars", () => {
    expect(hashCwd("/a/b")).toMatch(/^[0-9a-f]{16}$/);
    expect(hashCwd("/a/b")).toBe(hashCwd("/a/b"));
    expect(hashCwd("/a/b")).not.toBe(hashCwd("/a/c"));
  });

  it("contentHash differs by role/text/dim/model", () => {
    const a = contentHash("user", "hi", 384, "m1");
    const b = contentHash("assistant", "hi", 384, "m1");
    const c = contentHash("user", "hello", 384, "m1");
    const d = contentHash("user", "hi", 768, "m1");
    const e = contentHash("user", "hi", 384, "m2");
    expect(new Set([a, b, c, d, e]).size).toBe(5);
  });

  it("extractIndexableText: user string content", () => {
    expect(extractIndexableText({ role: "user", content: "hi" })).toBe("hi");
  });

  it("extractIndexableText: user blocks (text only)", () => {
    expect(
      extractIndexableText({
        role: "user",
        content: [
          { type: "text", text: "hello" },
          { type: "image", url: "x" },
        ],
      }),
    ).toBe("hello");
  });

  it("extractIndexableText: assistant text blocks, skips tool_use & thinking", () => {
    expect(
      extractIndexableText({
        role: "assistant",
        content: [
          { type: "thinking", text: "deliberation" },
          { type: "text", text: "the answer" },
          { type: "tool_use", name: "x", input: {} },
        ],
      }),
    ).toBe("the answer");
  });

  it("extractIndexableText: skips tool I/O by default", () => {
    expect(
      extractIndexableText({
        role: "toolResult",
        content: [{ type: "text", text: "BIG OUTPUT" }],
      }),
    ).toBe("");
    expect(
      extractIndexableText({ role: "bashExecution", command: "ls", output: "f" }),
    ).toBe("");
  });

  it("extractIndexableText: includes tool I/O when skipToolIO=false", () => {
    const t = extractIndexableText(
      { role: "bashExecution", command: "ls", output: "a\nb" },
      { skipToolIO: false },
    );
    expect(t).toContain("ls");
    expect(t).toContain("a");
  });

  it("extractIndexableText: summaries", () => {
    expect(
      extractIndexableText({ role: "compactionSummary", summary: "S1 text" }),
    ).toBe("S1 text");
  });

  it("makeSnippet collapses whitespace and truncates", () => {
    expect(makeSnippet("a   b\n\tc")).toBe("a b c");
    const long = "x".repeat(500);
    const s = makeSnippet(long, 100);
    expect(s.length).toBe(100);
    expect(s.endsWith("…")).toBe(true);
  });

  it("estimateTokens is monotonic in length", () => {
    expect(estimateTokens("a")).toBeLessThan(estimateTokens("aaaaaaaaaa"));
  });
});
