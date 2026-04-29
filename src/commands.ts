/**
 * /memory command dispatcher: stats, search, reindex, status, settings.
 */

import type { MemoryStore } from "./db/store.js";
import type { Retriever } from "./retrieval.js";
import type { Indexer } from "./indexer.js";
import type { MemoryConfig } from "./config.js";
import type { Diagnostics } from "./diagnostics.js";
import type { Embedder } from "./embeddings/embedder.js";

export interface CommandState {
  store: MemoryStore | null;
  retriever: Retriever | null;
  indexer: Indexer | null;
  diagnostics: Diagnostics | null;
  embedder: Embedder | null;
  config: MemoryConfig;
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
    case "settings":
      return openSettings(state, ctx);
    case "events":
      return printEvents(state, ctx);
    case "worker":
      return printWorker(state, ctx);
    default:
      ctx.ui.notify(`unknown subcommand: ${sub}. Try /memory help.`, "warning");
  }
}

function printHelp(ctx: any): void {
  ctx.ui.notify(
    [
      "/memory stats               — counts, model, dim, db size",
      "/memory status              — sweep cycles, busy, last error, interval",
      "/memory search <query>      — ad-hoc lcm_recall",
      "/memory reindex             — wipe & re-embed everything",
      "/memory settings            — open settings panel",
    ].join("\n"),
    "info",
  );
}

function printWorker(state: CommandState, ctx: any): void {
  if (!state.embedder) {
    ctx.ui.notify("pi-lcm-memory not initialized.", "warning");
    return;
  }
  const s = state.embedder.state();
  const url = state.embedder.workerUrl();
  const dl =
    s.totalBytes != null
      ? `${(s.downloadedBytes / 1024 / 1024).toFixed(1)}/${(s.totalBytes / 1024 / 1024).toFixed(1)} MB`
      : `${(s.downloadedBytes / 1024 / 1024).toFixed(1)} MB`;
  ctx.ui.notify(
    [
      `embedder worker:`,
      `  ready:        ${s.ready}`,
      `  loading:      ${s.loading}`,
      `  downloading:  ${s.downloading}  (${dl})`,
      `  thread id:    ${s.workerThreadId ?? "—"}`,
      `  worker pid:   ${s.workerPid ?? "—"}`,
      `  node version: ${s.workerNodeVersion ?? "—"}`,
      `  threads:      ${s.intraOpNumThreads ?? "—"}`,
      `  model:        ${s.model}`,
      `  dims:         ${s.dims}`,
      `  worker url:   ${url ?? "—"}`,
      `  last error:   ${s.error ?? "none"}`,
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
  const lines = [
    `pi-lcm-memory stats:`,
    `  indexed:  ${s.indexed}  (msg=${s.byMessage}  sum=${s.bySummary})`,
    `  vec:      ${s.vecRows} rows  ${s.vecAvailable ? "(sqlite-vec ✓)" : "(sqlite-vec UNAVAILABLE)"}`,
    `  model:    ${s.modelName ?? "—"}  dim=${s.modelDims ?? "—"}`,
    `  db size:  ${sizeMb} MB`,
  ];
  ctx.ui.notify(lines.join("\n"), "info");
}

function printStatus(state: CommandState, ctx: any): void {
  const indexer = state.indexer;
  if (!indexer) {
    ctx.ui.notify("pi-lcm-memory not initialized.", "warning");
    return;
  }
  const st = indexer.status();
  const interval = (st.currentIntervalMs / 1000).toFixed(0);
  ctx.ui.notify(
    [
      `pi-lcm-memory: ${st.busy ? "running" : "idle"}`,
      `  cycles: ${st.cycles}`,
      `  indexed total: ${st.indexedTotal}`,
      `  next sweep in ~${interval}s (idle streak: ${st.idleStreak})`,
      `  last error: ${st.lastError ?? "none"}`,
    ].join("\n"),
    "info",
  );
}

function printEvents(state: CommandState, ctx: any): void {
  const diag = state.diagnostics;
  if (!diag) {
    ctx.ui.notify("pi-lcm-memory not initialized.", "warning");
    return;
  }
  const events = diag.recent(20);
  if (events.length === 0) {
    ctx.ui.notify("no events recorded yet", "info");
    return;
  }
  const lines = events.map((e) => {
    const when = new Date(e.ts * 1000).toISOString().replace("T", " ").slice(0, 19);
    const data = e.data ? " " + JSON.stringify(e.data) : "";
    return `${when}  ${e.event}${data}`;
  });
  ctx.ui.notify(lines.join("\n"), "info");
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
  state.diagnostics?.log("reindex", { trigger: "command" });
  ctx.ui.notify("pi-lcm-memory: cleared. Re-embedding now…", "info");
  state.indexer.kick();
}

function openSettings(state: CommandState, ctx: any): void {
  if (state.openSettingsPanel) {
    state.openSettingsPanel(ctx);
  } else {
    ctx.ui.notify("settings panel unavailable", "warning");
  }
}
