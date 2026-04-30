import { describe, it, expect, afterEach } from "vitest";
import {
  makeTestDb,
  applyPiLcmSchema,
  setupVecAndMigrate,
  seedConversation,
  seedSummary,
  type TestDb,
} from "./helpers.js";
import { PiLcmBridge } from "../src/bridge.js";
import { renderPrimer } from "../src/primer.js";
import { estimateTokens } from "../src/utils.js";

describe("primer", () => {
  let t: TestDb | null = null;
  afterEach(() => {
    t?.cleanup();
    t = null;
  });

  it("returns null when disabled", async () => {
    t = makeTestDb();
    applyPiLcmSchema(t.db);
    await setupVecAndMigrate(t.db, 8);
    const out = renderPrimer({ bridge: new PiLcmBridge(t.db), topK: 5, enabled: false });
    expect(out).toBeNull();
  });

  it("returns null when no prior sessions", async () => {
    t = makeTestDb();
    applyPiLcmSchema(t.db);
    await setupVecAndMigrate(t.db, 8);
    const out = renderPrimer({ bridge: new PiLcmBridge(t.db), topK: 5, enabled: true });
    expect(out).toBeNull();
  });

  it("returns null when only the current session exists", async () => {
    t = makeTestDb();
    applyPiLcmSchema(t.db);
    await setupVecAndMigrate(t.db, 8);
    seedConversation(t.db, { id: "c1", created_at: "2026-04-29T00:00:00Z" });
    const out = renderPrimer({ bridge: new PiLcmBridge(t.db), topK: 5, enabled: true, currentConvId: "c1" });
    expect(out).toBeNull();
  });

  it("renders a primer when prior sessions + summaries exist", async () => {
    t = makeTestDb();
    applyPiLcmSchema(t.db);
    await setupVecAndMigrate(t.db, 8);
    seedConversation(t.db, { id: "c1", created_at: "2026-04-20T00:00:00Z" });
    seedConversation(t.db, { id: "c2", created_at: "2026-04-25T00:00:00Z" });
    seedSummary(t.db, { id: "s1", conv: "c1", depth: 1, text: "Refactored auth middleware", created_at: "2026-04-22T00:00:00Z" });
    seedSummary(t.db, { id: "s2", conv: "c2", depth: 2, text: "Added recall handler", created_at: "2026-04-25T11:00:00Z" });

    // c2 is the current session — primer should report only c1 as prior
    const out = renderPrimer({ bridge: new PiLcmBridge(t.db), topK: 5, enabled: true, currentConvId: "c2" });
    expect(out).not.toBeNull();
    expect(out!).toContain("Project memory");
    expect(out!).toContain("1 prior session");
    expect(out!).toContain("2026-04-20");
    expect(out!).toContain("Refactored auth middleware");
    expect(out!).toContain("lcm_recall");
    expect(estimateTokens(out!)).toBeLessThanOrEqual(310);
  });

  it("renders primer without summaries section when no summaries exist", async () => {
    t = makeTestDb();
    applyPiLcmSchema(t.db);
    await setupVecAndMigrate(t.db, 8);
    seedConversation(t.db, { id: "c1", created_at: "2026-04-20T00:00:00Z" });
    seedConversation(t.db, { id: "c2", created_at: "2026-04-29T00:00:00Z" });

    const out = renderPrimer({ bridge: new PiLcmBridge(t.db), topK: 5, enabled: true, currentConvId: "c2" });
    expect(out).not.toBeNull();
    expect(out!).toContain("1 prior session");
    expect(out!).toContain("2026-04-20");
    expect(out!).not.toContain("Recent topics");
    expect(out!).toContain("lcm_recall");
  });

  it("uses plural wording for multiple prior sessions", async () => {
    t = makeTestDb();
    applyPiLcmSchema(t.db);
    await setupVecAndMigrate(t.db, 8);
    seedConversation(t.db, { id: "c1", created_at: "2026-04-10T00:00:00Z" });
    seedConversation(t.db, { id: "c2", created_at: "2026-04-20T00:00:00Z" });
    seedConversation(t.db, { id: "c3", created_at: "2026-04-29T00:00:00Z" });

    const out = renderPrimer({ bridge: new PiLcmBridge(t.db), topK: 5, enabled: true, currentConvId: "c3" });
    expect(out).not.toBeNull();
    expect(out!).toContain("2 prior sessions");
    expect(out!).toContain("2026-04-20");
  });
});
