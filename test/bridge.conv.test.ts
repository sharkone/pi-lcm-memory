import { describe, it, expect, afterEach } from "vitest";
import {
  makeTestDb,
  applyPiLcmSchema,
  setupVecAndMigrate,
  seedConversation,
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

describe("PiLcmBridge session helpers", () => {
  let t: TestDb | null = null;
  afterEach(() => {
    t?.cleanup();
    t = null;
  });

  // newestConvId
  it("newestConvId returns null when conversations table is missing", async () => {
    t = makeTestDb();
    await setupVecAndMigrate(t.db, 8);
    expect(new PiLcmBridge(t.db).newestConvId()).toBeNull();
  });

  it("newestConvId returns null when table is empty", async () => {
    t = makeTestDb();
    applyPiLcmSchema(t.db);
    await setupVecAndMigrate(t.db, 8);
    expect(new PiLcmBridge(t.db).newestConvId()).toBeNull();
  });

  it("newestConvId returns the most recently created conversation id", async () => {
    t = makeTestDb();
    applyPiLcmSchema(t.db);
    await setupVecAndMigrate(t.db, 8);
    seedConversation(t.db, { id: "old", created_at: "2026-04-10T00:00:00Z" });
    seedConversation(t.db, { id: "mid", created_at: "2026-04-20T00:00:00Z" });
    seedConversation(t.db, { id: "cur", created_at: "2026-04-29T00:00:00Z" });
    expect(new PiLcmBridge(t.db).newestConvId()).toBe("cur");
  });

  // totalSessions
  it("totalSessions returns 0 when table is empty", async () => {
    t = makeTestDb();
    applyPiLcmSchema(t.db);
    await setupVecAndMigrate(t.db, 8);
    expect(new PiLcmBridge(t.db).totalSessions()).toBe(0);
  });

  it("totalSessions counts all rows when no excludeId", async () => {
    t = makeTestDb();
    applyPiLcmSchema(t.db);
    await setupVecAndMigrate(t.db, 8);
    seedConversation(t.db, { id: "c1" });
    seedConversation(t.db, { id: "c2" });
    seedConversation(t.db, { id: "c3" });
    expect(new PiLcmBridge(t.db).totalSessions()).toBe(3);
  });

  it("totalSessions excludes the current session when excludeId is provided", async () => {
    t = makeTestDb();
    applyPiLcmSchema(t.db);
    await setupVecAndMigrate(t.db, 8);
    seedConversation(t.db, { id: "c1" });
    seedConversation(t.db, { id: "c2" });
    seedConversation(t.db, { id: "cur" });
    expect(new PiLcmBridge(t.db).totalSessions("cur")).toBe(2);
  });

  // lastSessionStart
  it("lastSessionStart returns null when table is empty", async () => {
    t = makeTestDb();
    applyPiLcmSchema(t.db);
    await setupVecAndMigrate(t.db, 8);
    expect(new PiLcmBridge(t.db).lastSessionStart()).toBeNull();
  });

  it("lastSessionStart returns the most recent created_at when no excludeId", async () => {
    t = makeTestDb();
    applyPiLcmSchema(t.db);
    await setupVecAndMigrate(t.db, 8);
    seedConversation(t.db, { id: "c1", created_at: "2026-04-10T00:00:00Z" });
    seedConversation(t.db, { id: "c2", created_at: "2026-04-29T00:00:00Z" });
    expect(new PiLcmBridge(t.db).lastSessionStart()).toBe("2026-04-29T00:00:00Z");
  });

  it("lastSessionStart skips the current session and returns the previous one", async () => {
    t = makeTestDb();
    applyPiLcmSchema(t.db);
    await setupVecAndMigrate(t.db, 8);
    seedConversation(t.db, { id: "c1", created_at: "2026-04-10T00:00:00Z" });
    seedConversation(t.db, { id: "c2", created_at: "2026-04-20T00:00:00Z" });
    seedConversation(t.db, { id: "cur", created_at: "2026-04-29T00:00:00Z" });
    expect(new PiLcmBridge(t.db).lastSessionStart("cur")).toBe("2026-04-20T00:00:00Z");
  });
});
