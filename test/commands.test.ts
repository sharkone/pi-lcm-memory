import { describe, it, expect, afterEach } from "vitest";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import {
  makeTestDb,
  applyPiLcmSchema,
  setupVecAndMigrate,
  FakeEmbedder,
  type TestDb,
} from "./helpers.js";
import { MemoryStore } from "../src/db/store.js";
import { PiLcmBridge } from "../src/bridge.js";
import { Indexer } from "../src/indexer.js";
import { Diagnostics } from "../src/diagnostics.js";
import { handleMemoryCommand, type CommandState } from "../src/commands.js";
import { DEFAULTS, type MemoryConfig } from "../src/config.js";

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

function makeState(t: TestDb, cfg: MemoryConfig): CommandState {
  const store = new MemoryStore(t.db);
  const bridge = new PiLcmBridge(t.db);
  const fake = new FakeEmbedder(8);
  const indexer = new Indexer({
    store,
    embedder: fake as any,
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
    embedder: fake as any,
    config: { ...cfg },
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

  it("help lists subcommands", async () => {
    t = makeTestDb();
    applyPiLcmSchema(t.db);
    await setupVecAndMigrate(t.db, 8);
    cwd = mkdtempSync(join(tmpdir(), "pi-lcm-mem-cmd-"));
    const { ctx, notifies } = makeCtx();
    const state = makeState(t, { ...DEFAULTS });

    await handleMemoryCommand(state, undefined, ctx);
    expect(notifies[0]!.msg).toMatch(/\/memory stats/);
    expect(notifies[0]!.msg).toMatch(/\/memory settings/);
  });

  it("status reports interval/idleStreak/indexed", async () => {
    t = makeTestDb();
    applyPiLcmSchema(t.db);
    await setupVecAndMigrate(t.db, 8);
    cwd = mkdtempSync(join(tmpdir(), "pi-lcm-mem-cmd-"));
    const state = makeState(t, { ...DEFAULTS });

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
    const state = makeState(t, { ...DEFAULTS });

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
    const state = makeState(t, { ...DEFAULTS });

    const { ctx, notifies } = makeCtx();
    await handleMemoryCommand(state, "frobnicate", ctx);
    expect(notifies.at(-1)!.level).toBe("warning");
    expect(notifies.at(-1)!.msg).toMatch(/frobnicate/);
  });
});
