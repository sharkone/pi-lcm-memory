import { describe, it, expect, afterEach } from "vitest";
import { mkdtempSync, rmSync, readFileSync, existsSync, mkdirSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
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
import { Indexer } from "../src/indexer.js";
import { Diagnostics } from "../src/diagnostics.js";
import { handleMemoryCommand, type CommandState } from "../src/commands.js";
import { DEFAULTS, type MemoryConfig } from "../src/config.js";
import { isVecLoaded } from "../src/db/vec.js";
import { SETTINGS_KEY } from "../src/settings.js";

interface NotifyCall {
  msg: string;
  level: string;
}

function makeCtx(): { ctx: any; notifies: NotifyCall[] } {
  const notifies: NotifyCall[] = [];
  const ctx: any = {
    ui: {
      notify: (msg: string, level = "info") => notifies.push({ msg, level }),
      setStatus: () => undefined,
    },
  };
  return { ctx, notifies };
}

function makeState(t: TestDb, cfg: MemoryConfig, cwd: string): CommandState {
  const store = new MemoryStore(t.db);
  const bridge = new PiLcmBridge(t.db);
  const indexer = new Indexer({
    store,
    embedder: new FakeEmbedder(8) as any,
    bridge,
    config: cfg,
    conversationId: () => null,
    sessionStartedAt: () => null,
  });
  const diag = new Diagnostics(t.db);
  return {
    store,
    retriever: null,
    indexer,
    diagnostics: diag,
    config: { ...cfg },
    cwd,
    settingsScope: "project",
    openSettingsPanel: null,
  };
}

describe("/memory commands", () => {
  let t: TestDb | null = null;
  let cwd: string | null = null;
  afterEach(() => {
    t?.cleanup();
    t = null;
    if (cwd) rmSync(cwd, { recursive: true, force: true });
    cwd = null;
  });

  it("help lists subcommands and known models", async () => {
    t = makeTestDb();
    applyPiLcmSchema(t.db);
    await setupVecAndMigrate(t.db, 8);
    cwd = mkdtempSync(join(tmpdir(), "pi-lcm-mem-cmd-"));
    const { ctx, notifies } = makeCtx();
    const state = makeState(t, { ...DEFAULTS }, cwd);

    await handleMemoryCommand(state, undefined, ctx);
    expect(notifies[0]!.msg).toMatch(/\/memory stats/);
    expect(notifies[0]!.msg).toMatch(/Xenova\/bge-small-en-v1.5/);
  });

  it("clear without --yes refuses; with --yes clears + kicks indexer", async () => {
    t = makeTestDb();
    applyPiLcmSchema(t.db);
    await setupVecAndMigrate(t.db, 8);
    if (!isVecLoaded()) return;
    cwd = mkdtempSync(join(tmpdir(), "pi-lcm-mem-cmd-"));
    seedMessage(t.db, { id: "m1", conv: "c1", role: "user", text: "hello", ts: 1, seq: 0 });
    const state = makeState(t, { ...DEFAULTS }, cwd);

    // Backfill so there's something to clear.
    await state.indexer!.tick();
    expect(state.store!.stats().indexed).toBe(1);

    const { ctx, notifies } = makeCtx();
    await handleMemoryCommand(state, "clear", ctx);
    expect(notifies.at(-1)!.msg).toMatch(/destructive/);
    expect(state.store!.stats().indexed).toBe(1); // unchanged

    await handleMemoryCommand(state, "clear --yes", ctx);
    expect(state.store!.stats().indexed).toBe(0);
    // Diagnostics should have a 'clear' event.
    const events = state.diagnostics!.recent(5).map((e) => e.event);
    expect(events).toContain("clear");
  });

  it("model writes settings + emits diagnostics", async () => {
    t = makeTestDb();
    applyPiLcmSchema(t.db);
    await setupVecAndMigrate(t.db, 8);
    cwd = mkdtempSync(join(tmpdir(), "pi-lcm-mem-cmd-"));
    mkdirSync(join(cwd, ".pi"), { recursive: true });
    const state = makeState(t, { ...DEFAULTS }, cwd);

    const { ctx, notifies } = makeCtx();
    await handleMemoryCommand(state, "model Xenova/all-MiniLM-L6-v2", ctx);

    expect(state.config.embeddingModel).toBe("Xenova/all-MiniLM-L6-v2");
    expect(notifies.at(-1)!.msg).toMatch(/Xenova\/all-MiniLM-L6-v2/);

    const settingsPath = join(cwd, ".pi", "settings.json");
    expect(existsSync(settingsPath)).toBe(true);
    const json = JSON.parse(readFileSync(settingsPath, "utf8"));
    expect(json[SETTINGS_KEY].embeddingModel).toBe("Xenova/all-MiniLM-L6-v2");

    const events = state.diagnostics!.recent(5).map((e) => e.event);
    expect(events).toContain("model_change");
  });

  it("model with no name shows usage; same model is a no-op", async () => {
    t = makeTestDb();
    applyPiLcmSchema(t.db);
    await setupVecAndMigrate(t.db, 8);
    cwd = mkdtempSync(join(tmpdir(), "pi-lcm-mem-cmd-"));
    const state = makeState(t, { ...DEFAULTS }, cwd);

    const { ctx, notifies } = makeCtx();
    await handleMemoryCommand(state, "model", ctx);
    expect(notifies.at(-1)!.msg).toMatch(/usage/);
    await handleMemoryCommand(state, `model ${DEFAULTS.embeddingModel}`, ctx);
    expect(notifies.at(-1)!.msg).toMatch(/already on/);
  });

  it("status reports interval/idleStreak/indexed", async () => {
    t = makeTestDb();
    applyPiLcmSchema(t.db);
    await setupVecAndMigrate(t.db, 8);
    cwd = mkdtempSync(join(tmpdir(), "pi-lcm-mem-cmd-"));
    const state = makeState(t, { ...DEFAULTS }, cwd);

    const { ctx, notifies } = makeCtx();
    await handleMemoryCommand(state, "status", ctx);
    const out = notifies.at(-1)!.msg;
    expect(out).toMatch(/cycles:/);
    expect(out).toMatch(/next sweep/);
    expect(out).toMatch(/idle streak/);
    expect(out).toMatch(/indexed total/);
  });

  it("events shows 'no events recorded yet' on empty, then last events", async () => {
    t = makeTestDb();
    applyPiLcmSchema(t.db);
    await setupVecAndMigrate(t.db, 8);
    cwd = mkdtempSync(join(tmpdir(), "pi-lcm-mem-cmd-"));
    const state = makeState(t, { ...DEFAULTS }, cwd);

    const { ctx, notifies } = makeCtx();
    await handleMemoryCommand(state, "events", ctx);
    expect(notifies.at(-1)!.msg).toBe("no events recorded yet");

    state.diagnostics!.log("hello", { x: 1 });
    await handleMemoryCommand(state, "events", ctx);
    expect(notifies.at(-1)!.msg).toMatch(/hello/);
    expect(notifies.at(-1)!.msg).toMatch(/"x":1/);
  });

  it("unknown subcommand warns", async () => {
    t = makeTestDb();
    applyPiLcmSchema(t.db);
    await setupVecAndMigrate(t.db, 8);
    cwd = mkdtempSync(join(tmpdir(), "pi-lcm-mem-cmd-"));
    const state = makeState(t, { ...DEFAULTS }, cwd);

    const { ctx, notifies } = makeCtx();
    await handleMemoryCommand(state, "frobnicate", ctx);
    expect(notifies.at(-1)!.level).toBe("warning");
    expect(notifies.at(-1)!.msg).toMatch(/frobnicate/);
  });
});
