/**
 * Indexer: keeps memory_index/memory_vec in sync with pi-lcm's data.
 *
 *   - Hook path: handleMessage(msg) embeds a new AgentMessage in-flight.
 *   - Sweep path: tick() scans pi-lcm's messages + summaries for rows that
 *     don't yet have an embedding row, and processes them in batches.
 *
 * All work runs off the response thread (queueMicrotask + async). The hook
 * never throws; errors are logged via the optional notify callback.
 */

import { contentHash, extractIndexableText, makeSnippet, estimateTokens } from "./utils.js";
import type { Embedder } from "./embeddings/embedder.js";
import type { MemoryStore } from "./db/store.js";
import type { PiLcmBridge, PiLcmMessage, PiLcmSummary } from "./bridge.js";
import type { MemoryConfig } from "./config.js";

export type Notify = (msg: string, level?: "info" | "warning" | "error") => void;

export interface IndexerDeps {
  store: MemoryStore;
  embedder: Embedder;
  bridge: PiLcmBridge;
  config: MemoryConfig;
  conversationId: () => string | null;
  sessionStartedAt: () => number | null;
  notify?: Notify;
}

export class Indexer {
  private deps: IndexerDeps;
  private timer: NodeJS.Timeout | null = null;
  private inflight: Promise<unknown> = Promise.resolve();
  private busy = false;
  private stopped = false;
  private lastError: string | null = null;
  private cycles = 0;
  private indexedThisCycle = 0;

  constructor(deps: IndexerDeps) {
    this.deps = deps;
  }

  start(): void {
    if (this.timer || this.stopped) return;
    const tick = () => {
      this.tick().catch(() => {
        /* lastError already set */
      });
    };
    this.timer = setInterval(tick, this.deps.config.sweepIntervalMs);
    if (typeof this.timer.unref === "function") this.timer.unref();
  }

  stop(): void {
    this.stopped = true;
    if (this.timer) clearInterval(this.timer);
    this.timer = null;
  }

  /** Wait for any in-flight embedding to finish. Used at session_shutdown. */
  async drain(): Promise<void> {
    await this.inflight;
  }

  status(): { busy: boolean; cycles: number; lastError: string | null } {
    return { busy: this.busy, cycles: this.cycles, lastError: this.lastError };
  }

  /** Hook entrypoint: queue an AgentMessage for embedding. Never throws. */
  handleMessage(message: any, piLcmMsgId?: string | null): void {
    const text = extractIndexableText(message, { skipToolIO: this.deps.config.skipToolIO });
    if (!text) return;
    if (!this.deps.config.indexMessages) return;

    const job = this.embedAndStore({
      kind: "message",
      role: typeof message?.role === "string" ? message.role : null,
      text,
      piLcmMsgId: piLcmMsgId ?? null,
      piLcmSumId: null,
      conversationId: this.deps.conversationId(),
      sessionStartedAt: this.deps.sessionStartedAt(),
      depth: null,
    }).catch((e) => {
      this.lastError = e instanceof Error ? e.message : String(e);
      this.notify(`embed failed: ${this.lastError}`, "warning");
    });

    this.inflight = this.inflight.then(() => job);
  }

  /** Sweep: process every un-indexed message + summary in pi-lcm. */
  async tick(): Promise<void> {
    if (this.busy || this.stopped) return;
    this.busy = true;
    this.indexedThisCycle = 0;
    try {
      if (this.deps.config.indexMessages) {
        for (const m of this.deps.bridge.messagesNotInMemoryIndex(64)) {
          if (this.stopped) break;
          await this.embedFromBridgeMessage(m);
        }
      }
      if (this.deps.config.indexSummaries) {
        for (const s of this.deps.bridge.summariesNotInMemoryIndex(64)) {
          if (this.stopped) break;
          await this.embedFromBridgeSummary(s);
        }
      }
      this.cycles += 1;
      this.lastError = null;
    } catch (e: unknown) {
      this.lastError = e instanceof Error ? e.message : String(e);
      this.notify(`sweep failed: ${this.lastError}`, "warning");
    } finally {
      this.busy = false;
    }
  }

  private async embedFromBridgeMessage(m: PiLcmMessage): Promise<void> {
    const skipToolIO = this.deps.config.skipToolIO;
    const indexable = !isToolIORole(m.role) || !skipToolIO;
    if (!indexable) return;
    if (!m.content_text) return;
    await this.embedAndStore({
      kind: "message",
      role: m.role,
      text: m.content_text,
      piLcmMsgId: m.id,
      piLcmSumId: null,
      conversationId: m.conversation_id,
      sessionStartedAt: m.timestamp,
      depth: null,
    });
  }

  private async embedFromBridgeSummary(s: PiLcmSummary): Promise<void> {
    if (!s.text) return;
    await this.embedAndStore({
      kind: "summary",
      role: null,
      text: s.text,
      piLcmMsgId: null,
      piLcmSumId: s.id,
      conversationId: s.conversation_id,
      sessionStartedAt: parseDate(s.created_at),
      depth: s.depth,
    });
  }

  private async embedAndStore(args: {
    kind: "message" | "summary";
    role: string | null;
    text: string;
    piLcmMsgId: string | null;
    piLcmSumId: string | null;
    conversationId: string | null;
    sessionStartedAt: number | null;
    depth: number | null;
  }): Promise<void> {
    const dims = this.deps.embedder.knownDims();
    const model = this.deps.config.embeddingModel;
    if (!dims) {
      // Force a warmup so dim is resolved.
      await this.deps.embedder.warmup();
    }

    const finalDims = this.deps.embedder.knownDims() ?? 0;
    if (finalDims === 0) throw new Error("embedder has no dim after warmup");

    const hash = contentHash(args.role ?? args.kind, args.text, finalDims, model);
    if (this.deps.store.hasContentHash(hash)) return;

    const [vec] = await this.deps.embedder.embed(args.text);
    if (!vec) return;

    const snippet = makeSnippet(args.text);
    this.deps.store.insert({
      source_kind: args.kind,
      content_hash: hash,
      embedding: vec,
      pi_lcm_msg_id: args.piLcmMsgId,
      pi_lcm_sum_id: args.piLcmSumId,
      conversation_id: args.conversationId,
      session_started: args.sessionStartedAt,
      role: args.role,
      depth: args.depth,
      snippet,
      text_full: args.text,
      token_count: estimateTokens(args.text),
      model_name: model,
      model_dims: finalDims,
    });
    this.indexedThisCycle += 1;
  }

  private notify(message: string, level: "info" | "warning" | "error" = "info"): void {
    if (this.deps.config.debugMode || level !== "info") {
      this.deps.notify?.(`[pi-lcm-memory] ${message}`, level);
    }
  }
}

function isToolIORole(role: string): boolean {
  return role === "toolResult" || role === "bashExecution";
}

function parseDate(iso: string | null | undefined): number | null {
  if (!iso) return null;
  const t = Date.parse(iso);
  return Number.isNaN(t) ? null : Math.floor(t / 1000);
}
