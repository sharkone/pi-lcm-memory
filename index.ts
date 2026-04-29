/**
 * pi-lcm-memory: persistent cross-session semantic memory for Pi.
 *
 * Loads alongside pi-lcm. Reads/writes its own additive tables in pi-lcm's
 * per-cwd SQLite. Adds:
 *   - lcm_recall, lcm_similar tools
 *   - /memory + /memory-settings commands
 *   - session-start primer (decision B)
 *   - heuristic auto-recall (decision D)
 *   - background sweep
 */

import type { ExtensionAPI } from "@mariozechner/pi-coding-agent";

import { resolveConfig, type MemoryConfig } from "./src/config.js";
import { loadSettings, saveSettings, type SettingsScope } from "./src/settings.js";
import { openDb, closeDb } from "./src/db/connection.js";
import { ensureVecLoaded, isVecLoadedFor, vecError } from "./src/db/vec.js";
import { runMigrations } from "./src/db/schema.js";
import { MemoryStore } from "./src/db/store.js";
import { Embedder } from "./src/embeddings/embedder.js";
import { lookupModel } from "./src/embeddings/model-registry.js";
import { PiLcmBridge } from "./src/bridge.js";
import { Indexer } from "./src/indexer.js";
import { Retriever } from "./src/retrieval.js";
import { renderPrimer } from "./src/primer.js";
import { maybeAutoRecall } from "./src/auto-recall.js";
import { handleMemoryCommand, type CommandState } from "./src/commands.js";
import { updateStatus } from "./src/status.js";
import { createLcmRecallTool } from "./src/tools/lcm-recall.js";
import { createLcmSimilarTool } from "./src/tools/lcm-similar.js";
import { MemorySettingsPanel } from "./src/settings-panel.js";
import { Diagnostics } from "./src/diagnostics.js";
import { trace, traceFile, isTracing } from "./src/trace.js";

export default function (pi: ExtensionAPI) {
  let config: MemoryConfig = resolveConfig();
  if (!config.enabled) return;

  let store: MemoryStore | null = null;
  let bridge: PiLcmBridge | null = null;
  let embedder: Embedder | null = null;
  let indexer: Indexer | null = null;
  let retriever: Retriever | null = null;
  let diagnostics: Diagnostics | null = null;

  let cwd: string | null = null;
  let conversationId: string | null = null;
  let sessionStartedAt: number | null = null;
  let settingsScope: SettingsScope = "global";
  let primerEmitted = false;
  let modelDownloadAnnounced = false;

  const getStore = () => store;
  const getBridge = () => bridge;
  const getEmbedder = () => embedder;
  const getIndexer = () => indexer;
  const getRetriever = () => retriever;
  const getConvId = () => conversationId;
  const getStarted = () => sessionStartedAt;

  function commandState(): CommandState {
    return {
      store,
      retriever,
      indexer,
      diagnostics,
      embedder,
      config,
      cwd,
      settingsScope,
      openSettingsPanel: openSettingsPanel,
      onConfigChange: (cfg) => {
        config = cfg;
      },
    };
  }

  // ── Initialization ──────────────────────────────────────────────────────────

  async function initSession(ctx: any): Promise<void> {
    cwd = ctx.cwd ?? process.cwd();
    if (!cwd) return;
    trace("init_session_start", { cwd, reason: ctx?.reason });

    config = resolveConfig({ cwd });
    if (!config.enabled) return;

    const loaded = loadSettings(cwd);
    settingsScope = loaded.source === "project" ? "project" : "global";

    const db = openDb(config.dbDir, cwd);
    await ensureVecLoaded(db);

    embedder = new Embedder({
      model: config.embeddingModel,
      quantize: config.embeddingQuantize,
      cacheDir: config.modelCacheDir,
    });
    // State for throttled status & periodic download notifications.
    let lastStatusUpdate = 0;
    let lastDownloadAnnounceMb = 0;
    let lastNotifiedFile = "";

    embedder.setListener({
      onWorkerHello: ({ threadId, pid, nodeVersion, cores }) => {
        diagnostics?.log("worker_hello", { threadId, pid, nodeVersion, cores });
        if (config.debugMode) {
          ctx.ui?.notify?.(
            `[pi-lcm-memory] worker spawned (thread ${threadId}, pid ${pid}, node ${nodeVersion}, ${cores} cores).`,
            "info",
          );
        }
      },
      onProgress: ({ file, loaded, total }) => {
        if (!modelDownloadAnnounced) {
          modelDownloadAnnounced = true;
          const sz = total ? `${(total / 1024 / 1024).toFixed(1)} MB` : "size unknown";
          ctx.ui?.notify?.(
            `[pi-lcm-memory] downloading ${file || "model files"} (${sz}). One-time setup; this can take a minute on slow networks.`,
            "info",
          );
          diagnostics?.log("model_download_start", { model: config.embeddingModel, total, file });
          lastNotifiedFile = file;
        }
        if (file && file !== lastNotifiedFile) {
          lastNotifiedFile = file;
          ctx.ui?.notify?.(`[pi-lcm-memory] now fetching ${file}…`, "info");
          // Reset MB counter for the new file.
          lastDownloadAnnounceMb = 0;
        }
        // Periodic "still downloading" notifications keyed off MB downloaded
        // so the user never feels the UI is dead during a long fetch. Every
        // 10 MB of fresh bytes since the last notify.
        const mb = Math.floor(loaded / 1024 / 1024);
        if (mb >= lastDownloadAnnounceMb + 10) {
          lastDownloadAnnounceMb = mb;
          const totalLabel = total ? ` / ${(total / 1024 / 1024).toFixed(1)} MB` : "";
          ctx.ui?.notify?.(
            `[pi-lcm-memory] downloaded ${mb} MB${totalLabel}…`,
            "info",
          );
        }
        // Throttle setStatus: the footer can update at most ~4 Hz to avoid
        // saturating the TUI render queue.
        const now = Date.now();
        if (now - lastStatusUpdate >= 250) {
          lastStatusUpdate = now;
          updateStatus(store, indexer, embedder, ctx);
        }
      },
      onLoaded: () => {
        trace("embedder_loaded");
        const st = embedder?.state();
        diagnostics?.log("model_loaded", {
          model: config.embeddingModel,
          dims: embedder?.knownDims() ?? null,
          intraOpNumThreads: st?.intraOpNumThreads ?? null,
        });
        const traceSuffix = isTracing() ? ` Trace: ${traceFile()}` : "";
        ctx.ui?.notify?.(
          `[pi-lcm-memory] embedder ready (${config.embeddingModel}, ${st?.intraOpNumThreads ?? "?"} cores). Backfill starting.${traceSuffix}`,
          "info",
        );
        updateStatus(store, indexer, embedder, ctx);
        // Kick a sweep so backfill begins immediately once weights are warm.
        indexer?.kick();
      },
      onError: (msg) => {
        diagnostics?.log("model_error", { model: config.embeddingModel, error: msg });
        ctx.ui?.notify?.(`[pi-lcm-memory] embedder failed: ${msg}`, "error");
      },
    });

    const known = lookupModel(config.embeddingModel);
    const dim = known?.dims ?? 384; // Default to 384 for unknown; reconciled after warmup.

    runMigrations(db, { embeddingDim: dim, embeddingModel: config.embeddingModel });

    store = new MemoryStore(db);
    bridge = new PiLcmBridge(db);
    diagnostics = new Diagnostics(db);

    indexer = new Indexer({
      store,
      embedder,
      bridge,
      config,
      conversationId: getConvId,
      sessionStartedAt: getStarted,
      notify: (msg, level) => ctx.ui?.notify?.(msg, level ?? "info"),
      log: (event, data) => diagnostics?.log(event, data),
    });
    indexer.start();

    retriever = new Retriever({
      db,
      store,
      embedder,
      bridge,
      rrfK: config.rrfK,
      lexMult: config.lexMult,
      semMult: config.semMult,
    });

    sessionStartedAt = Math.floor(Date.now() / 1000);
    primerEmitted = false;
    modelDownloadAnnounced = false;

    diagnostics.log("session_start", {
      model: config.embeddingModel,
      vec: isVecLoadedFor(db),
      vecError: vecError(),
    });

    if (config.debugMode) {
      const s = store.stats();
      ctx.ui?.notify?.(
        `[pi-lcm-memory] init: model=${config.embeddingModel} dim=${s.modelDims ?? "?"} ` +
          `indexed=${s.indexed} vec=${isVecLoadedFor(db) ? "✓" : "✗"}` +
          (vecError() ? ` (${vecError()})` : ""),
        "info",
      );
    }

    updateStatus(store, indexer, embedder, ctx);

    // Background warmup of the embedder so first query is fast.
    embedder.warmup().catch((e: unknown) => {
      const m = e instanceof Error ? e.message : String(e);
      if (config.debugMode) ctx.ui?.notify?.(`[pi-lcm-memory] embedder warmup: ${m}`, "warning");
    });
  }

  function resetState(): void {
    indexer?.stop();
    indexer = null;
    embedder?.terminate();
    embedder = null;
    store = null;
    bridge = null;
    retriever = null;
    diagnostics = null;
    cwd = null;
    conversationId = null;
    sessionStartedAt = null;
    primerEmitted = false;
    modelDownloadAnnounced = false;
    closeDb();
  }

  function openSettingsPanel(ctx: any): void {
    if (!cwd || !store) {
      ctx.ui.notify("pi-lcm-memory not initialized.", "warning");
      return;
    }
    if (typeof ctx.ui?.custom !== "function") {
      ctx.ui?.notify?.("settings panel UI not available in this Pi runtime.", "warning");
      return;
    }
    // Pi's contract:
    //   ctx.ui.custom(factory, { overlay?: boolean })
    //   factory: (tui, theme, keybindings, done) => Component | Promise<Component>
    // We MUST construct the panel inside the factory so we have access to
    // `done`, the close callback. Calling done() (e.g. on Q / Esc) restores
    // the editor and resolves the custom() promise. Passing an object here
    // — as we did before — made pi try to invoke a non-function and crash.
    ctx.ui.custom(
      (_tui: unknown, _theme: unknown, _kb: unknown, done: (result?: unknown) => void) => {
        const panel = new MemorySettingsPanel({
          config,
          scope: settingsScope,
          cwd: cwd ?? process.cwd(),
          stats: store!.stats(),
          save: (cfg, scope, cwdArg) => {
            saveSettings(cfg, scope, cwdArg);
            config = { ...cfg };
            settingsScope = scope;
            ctx.ui?.notify?.(`pi-lcm-memory settings saved to ${scope}.`, "info");
          },
        });
        panel.onClose = () => done();
        return panel;
      },
      { overlay: true },
    );
  }

  // ── Hooks ───────────────────────────────────────────────────────────────────

  pi.on("session_start", async (event: any, ctx: any) => {
    trace("session_start_hook", { reason: event?.reason });
    try {
      if (typeof event?.reason === "string" && event.reason !== "startup") {
        resetState();
      }
      await initSession(ctx);
    } catch (e: any) {
      console.error("[pi-lcm-memory] init failed:", e?.message ?? e);
      ctx.ui?.notify?.(`pi-lcm-memory init failed: ${e?.message ?? e}`, "warning");
      resetState();
    }
  });

  pi.on("message_end", async (event: any, ctx: any) => {
    try {
      if (!indexer) return;
      // Lazy capture of conversation_id: pi-lcm writes its row before our sweep
      // would notice; we can read the most recent row from the bridge.
      if (!conversationId && bridge) {
        conversationId = bridge.latestConversationId();
      }
      indexer.handleMessage(event?.message ?? null, null);
      updateStatus(store, indexer, embedder, ctx);
    } catch (e: any) {
      if (config.debugMode) ctx.ui?.notify?.(`message_end: ${e?.message ?? e}`, "warning");
    }
  });

  pi.on("session_before_compact", async (_event: any, ctx: any) => {
    // pi-lcm runs compaction; we'll see new summaries in the next sweep tick.
    diagnostics?.log("compact_observed");
    indexer?.kick();
    updateStatus(store, indexer, embedder, ctx);
  });

  pi.on("context", async (event: any, _ctx: any) => {
    try {
      const messages: any[] = Array.isArray(event?.messages) ? event.messages : [];
      const additions: any[] = [];

      if (config.primer && !primerEmitted && bridge) {
        const text = renderPrimer({ bridge, topK: config.primerTopK, enabled: true });
        if (text) {
          additions.push({ role: "system", content: [{ type: "text", text }] });
          primerEmitted = true;
        }
      }

      const lastUser = lastUserText(messages);
      if (lastUser && retriever) {
        const block = await maybeAutoRecall(lastUser, {
          getRetriever: getRetriever,
          mode: () => config.autoRecall,
          topK: () => config.autoRecallTopK,
          tokenBudget: () => config.autoRecallTokenBudget,
        });
        if (block) {
          additions.push({ role: "system", content: [{ type: "text", text: block }] });
        }
      }

      if (additions.length > 0 && Array.isArray(event?.messages)) {
        // Insert additions right after the system prompt (index 0) if it
        // exists, else at the front.
        const insertAt = messages[0]?.role === "system" ? 1 : 0;
        event.messages.splice(insertAt, 0, ...additions);
      }
    } catch (e: any) {
      if (config.debugMode) console.error("[pi-lcm-memory] context hook:", e?.message ?? e);
    }
  });

  pi.on("session_shutdown", async (_event: any, _ctx: any) => {
    try {
      diagnostics?.log("session_shutdown");
      await indexer?.drain();
    } finally {
      resetState();
    }
  });

  // ── Tools + commands ────────────────────────────────────────────────────────

  pi.registerTool(
    createLcmRecallTool({
      getRetriever,
      getDefaultK: () => config.recallDefaultTopK,
    }) as any,
  );

  pi.registerTool(
    createLcmSimilarTool({
      getRetriever,
    }) as any,
  );

  pi.registerCommand("memory", {
    description: "pi-lcm-memory: cross-session semantic memory (stats/search/model/reindex/clear/settings)",
    handler: async (args: string | undefined, ctx: any) => {
      await handleMemoryCommand(commandState(), args, ctx);
    },
  });

  pi.registerCommand("memory-settings", {
    description: "Open the pi-lcm-memory settings panel.",
    handler: async (_args: string | undefined, ctx: any) => {
      openSettingsPanel(ctx);
    },
  });

  // Suppress unused-warning for getEmbedder helper (used by future tools).
  void getEmbedder;
  void getStore;
  void getBridge;
}

function lastUserText(messages: any[]): string {
  for (let i = messages.length - 1; i >= 0; i--) {
    const m = messages[i];
    if (m?.role !== "user") continue;
    if (typeof m.content === "string") return m.content;
    if (Array.isArray(m.content)) {
      const t = m.content.filter((b: any) => b?.type === "text").map((b: any) => b.text as string).join("\n");
      if (t) return t;
    }
  }
  return "";
}
