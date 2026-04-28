/**
 * Session-start memory primer (decision B). Renders a brief block summarizing
 * prior project memory that the agent receives on the first turn of a new
 * session. Stays under ~300 tokens and is empty when there's nothing to show.
 */

import type { PiLcmBridge } from "./bridge.js";
import { estimateTokens } from "./utils.js";

export interface PrimerDeps {
  bridge: PiLcmBridge;
  topK: number;
  enabled: boolean;
}

export function renderPrimer(deps: PrimerDeps): string | null {
  if (!deps.enabled) return null;
  const sessions = deps.bridge.totalSessions();
  if (sessions <= 0) return null;

  const last = deps.bridge.lastSessionStart();
  const lastDate = last ? last.slice(0, 10) : "—";

  const summaries = deps.bridge.recentSummaries(deps.topK, 1);

  const lines: string[] = [];
  lines.push("## Project memory");
  lines.push(`${sessions} prior session${sessions === 1 ? "" : "s"}; last on ${lastDate}.`);

  if (summaries.length > 0) {
    lines.push("");
    lines.push("Recent topics:");
    for (const s of summaries) {
      const preview = oneLine(s.text, 120);
      const when = s.created_at ? s.created_at.slice(0, 10) : "—";
      lines.push(`- [${when}] D${s.depth} ${s.id.slice(0, 8)}: ${preview}`);
    }
  }

  lines.push("");
  lines.push("Tools available:");
  lines.push("- `lcm_grep(pattern)` — exact strings / regex.");
  lines.push("- `lcm_recall(query)` — hybrid semantic + lexical recall.");
  lines.push("- `lcm_similar(messageId)` — more like this.");
  lines.push("- `lcm_expand(summaryId)` — recover original messages from a summary.");

  let text = lines.join("\n");
  // Token guard — if this somehow blew past budget, truncate.
  if (estimateTokens(text) > 300) {
    text = text.slice(0, 1050) + "\n…";
  }
  return text;
}

function oneLine(s: string, max: number): string {
  const cleaned = s.replace(/\s+/g, " ").trim();
  return cleaned.length <= max ? cleaned : `${cleaned.slice(0, max - 1)}…`;
}
