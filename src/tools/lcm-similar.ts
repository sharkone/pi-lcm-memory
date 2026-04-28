/**
 * lcm_similar: "more like this" — find rows whose embeddings are close to a
 * reference message or memory_index row.
 */

import { Type } from "typebox";
import type { Retriever, RecallHit } from "../retrieval.js";

export interface LcmSimilarDeps {
  getRetriever: () => Retriever | null;
}

export function createLcmSimilarTool(deps: LcmSimilarDeps) {
  return {
    name: "lcm_similar",
    label: "LCM Similar",
    description:
      "Given a pi-lcm message id (the same id `lcm_grep` returns), find the K most semantically " +
      "similar messages or summaries across all sessions in this project. Useful for finding past " +
      "work on the same topic when you already have one good example.",
    promptSnippet: "Find messages similar to a known one",
    parameters: Type.Object({
      messageId: Type.String({
        description: "A pi-lcm message id (e.g. as returned by lcm_grep / lcm_recall hits).",
      }),
      k: Type.Optional(Type.Number({ description: "Top-K. Default 5. Max 50." })),
    }),
    async execute(_toolCallId: string, params: { messageId: string; k?: number }) {
      const retriever = deps.getRetriever();
      if (!retriever) {
        return {
          content: [{ type: "text", text: "pi-lcm-memory not initialized for this session." }],
          isError: true,
          details: {},
        };
      }
      const k = Math.min(Math.max(params.k ?? 5, 1), 50);
      const hits = await retriever.similar({ messageId: params.messageId }, k);
      return {
        content: [{ type: "text", text: formatHits(hits, params.messageId) }],
        details: { hits },
      };
    },
  };
}

function formatHits(hits: RecallHit[], seedId: string): string {
  if (hits.length === 0) {
    return `No similar items found for ${seedId}. The seed may not be indexed yet (sweep runs every ~30s).`;
  }
  const lines: string[] = [`Top ${hits.length} similar to ${seedId.slice(0, 8)}:`, ""];
  hits.forEach((h, i) => {
    const when = h.session_started ? new Date(h.session_started * 1000).toISOString().slice(0, 10) : "—";
    const tag =
      h.source_kind === "summary"
        ? `summary D${h.depth ?? "?"}`
        : `${h.role ?? "msg"}`;
    lines.push(`${i + 1}. [${when}] [${tag}] sim=${h.score.toFixed(4)}`);
    lines.push(`   ${h.snippet}`);
  });
  return lines.join("\n");
}
