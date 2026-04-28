/**
 * Footer status line. Updates whenever we have new info to show.
 */

import type { MemoryStore } from "./db/store.js";
import type { Indexer } from "./indexer.js";
import type { Embedder } from "./embeddings/embedder.js";

export function updateStatus(
  store: MemoryStore | null,
  indexer: Indexer | null,
  embedder: Embedder | null,
  ctx: any,
): void {
  if (!store || !indexer) {
    if (typeof ctx?.ui?.setStatus === "function") ctx.ui.setStatus("");
    return;
  }
  try {
    const s = store.stats();
    const st = indexer.status();
    const e = embedder?.state();

    let prefix = "mem";
    if (e?.downloading) {
      const pct =
        e.totalBytes && e.totalBytes > 0
          ? Math.min(100, Math.floor((e.downloadedBytes / e.totalBytes) * 100))
          : null;
      prefix = `mem dl ${pct ?? "?"}%`;
    } else if (e?.loading) {
      prefix = "mem loading";
    }

    const sizeMb = (s.dbSizeBytes / 1024 / 1024).toFixed(1);
    const text =
      `${prefix}: ${s.indexed} (${s.byMessage}msg/${s.bySummary}sum) ` +
      `${st.busy ? "•" : "○"} ${sizeMb}MB`;
    if (typeof ctx?.ui?.setStatus === "function") ctx.ui.setStatus(text);
  } catch {
    // best-effort
  }
}
