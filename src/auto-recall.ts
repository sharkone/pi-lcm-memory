/**
 * Heuristic auto-recall (decision D). When the latest user message contains
 * one of the trigger phrases, run lcm_recall and produce a "## Recall" block
 * to inject as system context for the *current turn only*.
 *
 * Token budget enforced. Off-by-default if mode === "off". When mode ===
 * "always" the trigger always fires (decision C as a stretch toggle).
 */

import type { Retriever, RecallHit } from "./retrieval.js";
import type { AutoRecallMode } from "./config.js";
import { estimateTokens } from "./utils.js";

export const DEFAULT_TRIGGER = new RegExp(
  "\\b(remember|recall|earlier|previously|before|like\\s+last\\s+time|" +
    "the\\s+(same|previous|prior)\\s+(one|approach|setup|fix|bug|issue)|" +
    "we\\s+(had|have|already)\\s+(discussed|talked|mentioned|tried|done))\\b",
  "i",
);

export interface AutoRecallDeps {
  getRetriever: () => Retriever | null;
  mode: () => AutoRecallMode;
  topK: () => number;
  tokenBudget: () => number;
  trigger?: RegExp;
}

export function shouldFire(text: string, mode: AutoRecallMode, trigger: RegExp): boolean {
  if (mode === "off") return false;
  if (mode === "always") return true;
  if (!text) return false;
  return trigger.test(text);
}

export async function maybeAutoRecall(
  userText: string,
  deps: AutoRecallDeps,
): Promise<string | null> {
  const trigger = deps.trigger ?? DEFAULT_TRIGGER;
  if (!shouldFire(userText, deps.mode(), trigger)) return null;
  const retriever = deps.getRetriever();
  if (!retriever) return null;

  const hits = await retriever.recall({
    query: userText,
    k: deps.topK(),
    mode: "hybrid",
  });
  if (hits.length === 0) return null;

  return renderRecallBlock(hits, deps.tokenBudget());
}

function renderRecallBlock(hits: RecallHit[], tokenBudget: number): string {
  const header = [
    "## Recall (auto-injected for this turn)",
    "",
    "Possibly relevant context from prior sessions in this project:",
    "",
  ];
  let budgetLeft = tokenBudget - estimateTokens(header.join("\n"));
  const body: string[] = [];

  for (let i = 0; i < hits.length; i++) {
    const h = hits[i]!;
    const when = h.session_started ? new Date(h.session_started * 1000).toISOString().slice(0, 10) : "—";
    const tag =
      h.source_kind === "summary"
        ? `summary D${h.depth ?? "?"}${h.pi_lcm_sum_id ? ` ${h.pi_lcm_sum_id.slice(0, 8)}` : ""}`
        : `${h.role ?? "msg"}${h.pi_lcm_msg_id ? ` ${h.pi_lcm_msg_id.slice(0, 8)}` : ""}`;
    const line1 = `${i + 1}. [${when}] [${tag}] score=${h.score.toFixed(3)}`;
    const line2 = `   ${h.snippet}`;
    const cost = estimateTokens(`${line1}\n${line2}\n`);
    if (cost > budgetLeft) break;
    body.push(line1);
    body.push(line2);
    budgetLeft -= cost;
  }

  if (body.length === 0) return "";
  return [...header, ...body, "", "(use `lcm_recall` for more, `lcm_expand` to drill in)"].join("\n");
}
