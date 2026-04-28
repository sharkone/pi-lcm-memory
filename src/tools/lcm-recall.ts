/**
 * lcm_recall: hybrid (FTS5 + vector) recall over all sessions in this project.
 */

import { Type } from "typebox";
import { StringEnum } from "@mariozechner/pi-ai";
import type { Retriever, RecallHit, RecallMode } from "../retrieval.js";

export interface LcmRecallDeps {
  getRetriever: () => Retriever | null;
  getDefaultK: () => number;
}

export function createLcmRecallTool(deps: LcmRecallDeps) {
  return {
    name: "lcm_recall",
    label: "LCM Recall",
    description:
      "Search across ALL prior sessions in this project for messages or summaries matching a query, " +
      "using hybrid (lexical FTS5 + dense semantic vector) retrieval. Returns top-K snippets with " +
      "scores and lineage. Use this when you suspect past work in this project is relevant — " +
      "especially when the user references it ('remember…', 'we discussed…', 'last time…'), " +
      "or when the same topic might have come up before. Complementary to lcm_grep " +
      "(exact strings only) and lcm_expand (drill into a known summary id).",
    promptSnippet: "Hybrid recall across all sessions in this project",
    parameters: Type.Object({
      query: Type.String({ description: "Natural-language or keyword query." }),
      k: Type.Optional(Type.Number({ description: "Top-K results. Default 10. Max 100." })),
      mode: Type.Optional(
        StringEnum(["hybrid", "lexical", "semantic"] as const, {
          description:
            "hybrid (default): RRF merge of FTS5 + vector. lexical: FTS5 only (matches lcm_grep). semantic: vector only.",
        }),
      ),
      sessionFilter: Type.Optional(
        Type.String({ description: "Restrict to a single conversation_id (UUID)." }),
      ),
      after: Type.Optional(Type.String({ description: "ISO 8601 timestamp; only sessions started after this." })),
      before: Type.Optional(Type.String({ description: "ISO 8601 timestamp; only sessions started before this." })),
    }),
    async execute(
      _toolCallId: string,
      params: {
        query: string;
        k?: number;
        mode?: RecallMode;
        sessionFilter?: string;
        after?: string;
        before?: string;
      },
    ) {
      const retriever = deps.getRetriever();
      if (!retriever) {
        return {
          content: [{ type: "text", text: "pi-lcm-memory not initialized for this session." }],
          isError: true,
          details: {},
        };
      }

      const k = params.k ?? deps.getDefaultK();
      const hits = await retriever.recall({
        query: params.query,
        k,
        mode: params.mode ?? "hybrid",
        sessionFilter: params.sessionFilter ?? null,
        after: params.after ?? null,
        before: params.before ?? null,
      });

      return {
        content: [{ type: "text", text: formatHits(hits, params.query) }],
        details: { hits, mode: params.mode ?? "hybrid", k },
      };
    },
  };
}

function formatHits(hits: RecallHit[], query: string): string {
  if (hits.length === 0) {
    return `No memory matched "${query}". Try lcm_grep for exact strings, or broaden the query.`;
  }
  const lines: string[] = [`Top ${hits.length} hits for "${query}":`, ""];
  hits.forEach((h, i) => {
    const when = h.session_started ? new Date(h.session_started * 1000).toISOString().slice(0, 10) : "—";
    const tag =
      h.source_kind === "summary"
        ? `summary D${h.depth ?? "?"}${h.pi_lcm_sum_id ? ` ${h.pi_lcm_sum_id}` : ""}`
        : `${h.role ?? "msg"}${h.pi_lcm_msg_id ? ` ${h.pi_lcm_msg_id}` : ""}`;
    lines.push(`${i + 1}. [${when}] [${tag}] score=${h.score.toFixed(4)}`);
    lines.push(`   ${h.snippet}`);
  });
  lines.push("");
  lines.push("Use lcm_expand(summary_id) with the full UUID shown above to recover full text from any summary.");
  return lines.join("\n");
}
