import { createHash } from "node:crypto";

/** SHA-256 of cwd, truncated to 16 hex chars. Mirrors pi-lcm so we open the same DB file. */
export function hashCwd(cwd: string): string {
  return createHash("sha256").update(cwd).digest("hex").slice(0, 16);
}

/** Stable content hash used as the dedup key in memory_index. */
export function contentHash(role: string, text: string, dim: number, model: string): string {
  return createHash("sha256")
    .update(role)
    .update("\0")
    .update(text)
    .update("\0")
    .update(model)
    .update("\0")
    .update(String(dim))
    .digest("hex");
}

/**
 * Extract the embeddable text from a Pi AgentMessage. Mirrors pi-lcm's
 * extractSearchableText shape so our index aligns with pi-lcm's understanding
 * of "what counts as content". Returns "" for messages we don't index.
 *
 * v1 indexes user + assistant text + summary text. Tool I/O is intentionally
 * excluded from embedding (FTS5 in pi-lcm still covers it literally).
 */
export function extractIndexableText(message: any, opts?: { skipToolIO?: boolean }): string {
  const skipToolIO = opts?.skipToolIO ?? true;
  if (!message || !message.role) return "";

  switch (message.role) {
    case "user": {
      if (typeof message.content === "string") return message.content;
      if (!Array.isArray(message.content)) return "";
      return message.content
        .filter((b: any) => b?.type === "text")
        .map((b: any) => b.text as string)
        .join("\n");
    }

    case "assistant": {
      if (!Array.isArray(message.content)) return "";
      const parts: string[] = [];
      for (const block of message.content) {
        if (block?.type === "text") parts.push(block.text as string);
        // Skip tool calls — opaque arguments rarely make good semantic queries.
        // Skip thinking blocks — internal reasoning.
      }
      return parts.join("\n");
    }

    case "toolResult":
    case "bashExecution":
      return skipToolIO ? "" : extractToolText(message);

    case "custom": {
      if (typeof message.content === "string") return message.content;
      if (!Array.isArray(message.content)) return "";
      return message.content
        .filter((b: any) => b?.type === "text")
        .map((b: any) => b.text as string)
        .join("\n");
    }

    case "compactionSummary":
    case "branchSummary":
      return typeof message.summary === "string" ? message.summary : "";

    default:
      return "";
  }
}

function extractToolText(message: any): string {
  if (message.role === "bashExecution") {
    return `$ ${message.command ?? ""}\n${message.output ?? ""}`;
  }
  const txt = Array.isArray(message.content)
    ? message.content
        .filter((b: any) => b?.type === "text")
        .map((b: any) => b.text as string)
        .join("\n")
    : "";
  return `[${message.toolName ?? "tool"}] ${txt}`;
}

/** Take the first N chars of `text` for a preview snippet, collapsing whitespace. */
export function makeSnippet(text: string, max = 240): string {
  const cleaned = text.replace(/\s+/g, " ").trim();
  return cleaned.length <= max ? cleaned : `${cleaned.slice(0, max - 1)}…`;
}

/** ~3.5 chars/token heuristic, matching pi-lcm. ±15% error. */
export function estimateTokens(text: string): number {
  return Math.ceil(text.length / 3.5);
}
