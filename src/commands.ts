/**
 * /memory command dispatcher: stats, search, model, reindex, clear, status, settings.
 */

import type { MemoryStore } from "./db/store.js";
import type { Retriever } from "./retrieval.js";
import type { Indexer } from "./indexer.js";
import type { MemoryConfig } from "./config.js";
import { listModelNames } from "./embeddings/model-registry.js";
import { saveSettings, type SettingsScope } from "./settings.js";

export interface CommandState {
  store: MemoryStore | null;
  retriever: Retriever | null;
  indexer: Indexer | null;
  config: MemoryConfig;
  cwd: string | null;
  settingsScope: SettingsScope;
  openSettingsPanel: ((ctx: any) => void) | null;
}

export async function handleMemoryCommand(
  state: CommandState,
  args: string | undefined,
  ctx: any,
): Promise<void> {
  const tokens = (args ?? "").trim().split(/\s+/);
  const sub = tokens[0] ?? "help";
  const rest = tokens.slice(1).join(" ");

  switch (sub) {
    case "":
    case "help":
      return printHelp(ctx);
    case "stats":
      return printStats(state, ctx);
    case "status":
      return printStatus(state, ctx);
    case "search":
      return doSearch(state, rest, ctx);
    case "reindex":
      return doReindex(state, rest, ctx);
    case "clear":
      return doClear(state, rest, ctx);
    case "model":
      return doModel(state, rest, ctx);
    case "settings":
      return openSettings(state, ctx);
    default:
      ctx.ui.notify(`unknown subcommand: ${sub}. Try /memory help.`, "warning");
  }
}

function printHelp(ctx: any): void {
  ctx.ui.notify(
    [
      "/memory stats             — counts, model, dim, db size",
      "/memory status            — sweep cycles, busy, last error",
      "/memory search <query>    — ad-hoc lcm_recall",
      "/memory reindex [scope]   — wipe & re-embed (kind=message|summary|all)",
      "/memory clear             — drop all embeddings (sweep will rebuild)",
      "/memory model <name>      — change embedding model (triggers reindex)",
      "/memory settings          — open settings panel",
      "",
      "models: " + listModelNames().join(", "),
    ].join("\n"),
    "info",
  );
}

function printStats(state: CommandState, ctx: any): void {
  if (!state.store) {
    ctx.ui.notify("pi-lcm-memory not initialized.", "warning");
    return;
  }
  const s = state.store.stats();
  const sizeMb = (s.dbSizeBytes / 1024 / 1024).toFixed(1);
  ctx.ui.notify(
    [
      `pi-lcm-memory stats:`,
      `  indexed:  ${s.indexed}  (msg=${s.byMessage}  sum=${s.bySummary})`,
      `  vec:      ${s.vecRows} rows  ${s.vecAvailable ? "(sqlite-vec ✓)" : "(sqlite-vec UNAVAILABLE)"}`,
      `  model:    ${s.modelName ?? "—"}  dim=${s.modelDims ?? "—"}`,
      `  db size:  ${sizeMb} MB`,
    ].join("\n"),
    "info",
  );
}

function printStatus(state: CommandState, ctx: any): void {
  const indexer = state.indexer;
  if (!indexer) {
    ctx.ui.notify("pi-lcm-memory not initialized.", "warning");
    return;
  }
  const st = indexer.status();
  ctx.ui.notify(
    `pi-lcm-memory: ${st.busy ? "running" : "idle"} | cycles ${st.cycles} | last error: ${st.lastError ?? "none"}`,
    "info",
  );
}

async function doSearch(state: CommandState, query: string, ctx: any): Promise<void> {
  if (!state.retriever) {
    ctx.ui.notify("pi-lcm-memory not initialized.", "warning");
    return;
  }
  if (!query.trim()) {
    ctx.ui.notify("usage: /memory search <query>", "warning");
    return;
  }
  const hits = await state.retriever.recall({ query, k: state.config.recallDefaultTopK });
  if (hits.length === 0) {
    ctx.ui.notify(`no hits for "${query}"`, "info");
    return;
  }
  const lines = hits.slice(0, 10).map((h, i) => {
    const when = h.session_started ? new Date(h.session_started * 1000).toISOString().slice(0, 10) : "—";
    return `${i + 1}. [${when}] (${h.source_kind}, score=${h.score.toFixed(3)}) ${h.snippet}`;
  });
  ctx.ui.notify(lines.join("\n"), "info");
}

function doReindex(state: CommandState, _rest: string, ctx: any): void {
  if (!state.store || !state.indexer) {
    ctx.ui.notify("pi-lcm-memory not initialized.", "warning");
    return;
  }
  state.store.clearAll();
  ctx.ui.notify("pi-lcm-memory: cleared. Sweep will rebuild on next tick.", "info");
  // Kick a tick now so the user sees activity.
  state.indexer.tick().catch(() => {});
}

function doClear(state: CommandState, _rest: string, ctx: any): void {
  if (!state.store) {
    ctx.ui.notify("pi-lcm-memory not initialized.", "warning");
    return;
  }
  state.store.clearAll();
  ctx.ui.notify("pi-lcm-memory: all embeddings cleared.", "info");
}

function doModel(state: CommandState, name: string, ctx: any): void {
  if (!name.trim()) {
    ctx.ui.notify(
      "usage: /memory model <name>. known: " + listModelNames().join(", "),
      "warning",
    );
    return;
  }
  if (!state.cwd) {
    ctx.ui.notify("pi-lcm-memory: cwd unknown; cannot persist setting.", "warning");
    return;
  }
  const next: Partial<MemoryConfig> = { ...state.config, embeddingModel: name.trim() };
  saveSettings(next, state.settingsScope, state.cwd);
  ctx.ui.notify(
    `pi-lcm-memory: model set to ${name.trim()} (${state.settingsScope}). Restart Pi to apply; re-embed will run.`,
    "info",
  );
}

function openSettings(state: CommandState, ctx: any): void {
  if (state.openSettingsPanel) {
    state.openSettingsPanel(ctx);
  } else {
    ctx.ui.notify("settings panel unavailable", "warning");
  }
}
