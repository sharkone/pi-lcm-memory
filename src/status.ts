/**
 * Footer status line. Updates whenever we have new info to show.
 */

import type { MemoryStore } from "./db/store.js";
import type { Indexer } from "./indexer.js";

export function updateStatus(store: MemoryStore | null, indexer: Indexer | null, ctx: any): void {
  if (!store || !indexer) {
    if (typeof ctx?.ui?.setStatus === "function") ctx.ui.setStatus("");
    return;
  }
  try {
    const s = store.stats();
    const st = indexer.status();
    const sizeMb = (s.dbSizeBytes / 1024 / 1024).toFixed(1);
    const text =
      `mem: ${s.indexed} (${s.byMessage}msg/${s.bySummary}sum) ` +
      `${st.busy ? "•" : "○"} ${sizeMb}MB`;
    if (typeof ctx?.ui?.setStatus === "function") ctx.ui.setStatus(text);
  } catch {
    // best-effort
  }
}
