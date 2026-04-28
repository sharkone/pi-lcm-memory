import { describe, it, expect, afterEach } from "vitest";
import {
  makeTestDb,
  applyPiLcmSchema,
  setupVecAndMigrate,
  seedMessage,
  type TestDb,
} from "./helpers.js";
import { PiLcmBridge } from "../src/bridge.js";

describe("PiLcmBridge.latestConversationId", () => {
  let t: TestDb | null = null;
  afterEach(() => {
    t?.cleanup();
    t = null;
  });

  it("returns null when no messages", async () => {
    t = makeTestDb();
    applyPiLcmSchema(t.db);
    await setupVecAndMigrate(t.db, 8);
    const b = new PiLcmBridge(t.db);
    expect(b.latestConversationId()).toBeNull();
  });

  it("returns null when messages table is missing", async () => {
    t = makeTestDb();
    await setupVecAndMigrate(t.db, 8);
    const b = new PiLcmBridge(t.db);
    expect(b.latestConversationId()).toBeNull();
  });

  it("returns the conversation_id of the most recently inserted row", async () => {
    t = makeTestDb();
    applyPiLcmSchema(t.db);
    await setupVecAndMigrate(t.db, 8);
    seedMessage(t.db, { id: "m1", conv: "old", role: "user", text: "hi", ts: 1, seq: 0 });
    seedMessage(t.db, { id: "m2", conv: "newer", role: "assistant", text: "yo", ts: 2, seq: 1 });
    seedMessage(t.db, { id: "m3", conv: "newest", role: "user", text: "hey", ts: 3, seq: 2 });
    const b = new PiLcmBridge(t.db);
    expect(b.latestConversationId()).toBe("newest");
  });
});
