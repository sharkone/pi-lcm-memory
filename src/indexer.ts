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
import type { MemoryStore, InsertArgs } from "./db/store.js";
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
  /** Optional structured-log sink. Default: noop. */
  log?: (event: string, data?: Record<string, unknown>) => void;
}

/** Pending row awaiting embedding (used for batched sweep). */
interface PendingRow {
  text: string;
  args: Omit<InsertArgs, "embedding" | "model_dims">;
}

// Inference runs in a worker thread (see embedder.ts + worker.mjs), so the
// main thread isn't blocked by ONNX. SQLite writes are batched into a single
// transaction per batch (see store.insertBatch) so we acquire the WAL write
// lock once instead of N times — cuts contention with concurrent writers
// (notably pi-lcm) by ~30×. We yield to the event loop between batches so
// the TUI renders even during long backfills.
const SWEEP_BATCH = 32;
const SWEEP_BACKOFF_MIN_MS = 5_000;
const SWEEP_BACKOFF_MAX_MS = 5 * 60_000;

/** Yield to the event loop so TUI input/render can interleave with backfill. */
function yieldToEventLoop(): Promise<void> {
  return new Promise((resolve) => setImmediate(resolve));
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
  private indexedTotal = 0;
  private currentInterval = 0;
  private idleStreak = 0;

  constructor(deps: IndexerDeps) {
    this.deps = deps;
    this.currentInterval = deps.config.sweepIntervalMs;
  }

  start(): void {
    if (this.timer || this.stopped) return;
    this.scheduleNext(this.deps.config.sweepIntervalMs);
  }

  stop(): void {
    this.stopped = true;
    if (this.timer) clearTimeout(this.timer);
    this.timer = null;
  }

  /** Wait for any in-flight embedding to finish. Used at session_shutdown. */
  async drain(): Promise<void> {
    await this.inflight;
  }

  status(): {
    busy: boolean;
    cycles: number;
    lastError: string | null;
    indexedTotal: number;
    currentIntervalMs: number;
    idleStreak: number;
  } {
    return {
      busy: this.busy,
      cycles: this.cycles,
      lastError: this.lastError,
      indexedTotal: this.indexedTotal,
      currentIntervalMs: this.currentInterval,
      idleStreak: this.idleStreak,
    };
  }

  /** Force the next sweep to run immediately and reset backoff. */
  kick(): void {
    if (this.stopped) return;
    this.idleStreak = 0;
    this.currentInterval = this.deps.config.sweepIntervalMs;
    if (this.timer) clearTimeout(this.timer);
    this.scheduleNext(0);
  }

  private scheduleNext(ms: number): void {
    if (this.stopped) return;
    const handle = setTimeout(() => {
      this.timer = null;
      this.tick()
        .catch(() => {
          /* lastError already set */
        })
        .finally(() => {
          // Adaptive backoff: idle ticks double the interval (cap 5min); a productive
          // tick resets to base. New work via kick() also resets.
          if (this.indexedThisCycle === 0) {
            this.idleStreak += 1;
            this.currentInterval = Math.min(
              SWEEP_BACKOFF_MAX_MS,
              Math.max(
                SWEEP_BACKOFF_MIN_MS,
                this.deps.config.sweepIntervalMs * Math.pow(2, Math.min(this.idleStreak, 5)),
              ),
            );
          } else {
            this.idleStreak = 0;
            this.currentInterval = this.deps.config.sweepIntervalMs;
          }
          this.scheduleNext(this.currentInterval);
        });
    }, ms);
    if (typeof handle.unref === "function") handle.unref();
    this.timer = handle;
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
    const t0 = Date.now();
    try {
      // Ensure embedder is ready before walking large backlogs (avoids per-row
      // warmup cost). Best-effort: if it fails, individual embeds still try.
      await this.deps.embedder.warmup().catch(() => {});

      if (this.deps.config.indexMessages) {
        await this.processBatched(
          this.deps.bridge.messagesNotInMemoryIndex(SWEEP_BATCH * 4),
          (m: PiLcmMessage) => this.bridgeMessageToPending(m),
        );
      }
      if (this.deps.config.indexSummaries) {
        await this.processBatched(
          this.deps.bridge.summariesNotInMemoryIndex(SWEEP_BATCH * 4),
          (s: PiLcmSummary) => this.bridgeSummaryToPending(s),
        );
      }
      this.cycles += 1;
      this.lastError = null;
      this.deps.log?.("sweep_done", {
        cycle: this.cycles,
        indexed: this.indexedThisCycle,
        ms: Date.now() - t0,
      });
    } catch (e: unknown) {
      this.lastError = e instanceof Error ? e.message : String(e);
      this.deps.log?.("sweep_failed", { error: this.lastError });
      this.notify(`sweep failed: ${this.lastError}`, "warning");
    } finally {
      this.busy = false;
    }
  }

  /** Pull all pending rows from a generator, embed in batches, insert. */
  private async processBatched<T>(
    iter: Iterable<T>,
    toPending: (item: T) => PendingRow | null,
  ): Promise<void> {
    let batch: PendingRow[] = [];
    for (const item of iter) {
      if (this.stopped) break;
      const p = toPending(item);
      if (!p) continue;
      batch.push(p);
      if (batch.length >= SWEEP_BATCH) {
        await this.embedAndStoreBatch(batch);
        batch = [];
        // Even with batched inserts, bursts of activity can starve the TUI.
        // setImmediate gives the event loop a chance to render keystrokes,
        // status updates, etc., between sweep batches.
        await yieldToEventLoop();
      }
    }
    if (batch.length > 0 && !this.stopped) await this.embedAndStoreBatch(batch);
  }

  private bridgeMessageToPending(m: PiLcmMessage): PendingRow | null {
    const skipToolIO = this.deps.config.skipToolIO;
    const indexable = !isToolIORole(m.role) || !skipToolIO;
    if (!indexable || !m.content_text) return null;
    return {
      text: m.content_text,
      args: this.argsForBridgeMessage(m),
    };
  }

  private bridgeSummaryToPending(s: PiLcmSummary): PendingRow | null {
    if (!s.text) return null;
    return {
      text: s.text,
      args: this.argsForBridgeSummary(s),
    };
  }

  private argsForBridgeMessage(m: PiLcmMessage): Omit<InsertArgs, "embedding" | "model_dims"> {
    return {
      source_kind: "message",
      content_hash: contentHash(m.role, m.content_text, this.deps.embedder.knownDims() ?? 0, this.deps.config.embeddingModel),
      pi_lcm_msg_id: m.id,
      pi_lcm_sum_id: null,
      conversation_id: m.conversation_id,
      session_started: m.timestamp,
      role: m.role,
      depth: null,
      snippet: makeSnippet(m.content_text),
      text_full: m.content_text,
      token_count: estimateTokens(m.content_text),
      model_name: this.deps.config.embeddingModel,
    };
  }

  private argsForBridgeSummary(s: PiLcmSummary): Omit<InsertArgs, "embedding" | "model_dims"> {
    return {
      source_kind: "summary",
      content_hash: contentHash("summary", s.text, this.deps.embedder.knownDims() ?? 0, this.deps.config.embeddingModel),
      pi_lcm_msg_id: null,
      pi_lcm_sum_id: s.id,
      conversation_id: s.conversation_id,
      session_started: parseDate(s.created_at),
      role: null,
      depth: s.depth,
      snippet: makeSnippet(s.text),
      text_full: s.text,
      token_count: estimateTokens(s.text),
      model_name: this.deps.config.embeddingModel,
    };
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
    if (!this.deps.embedder.knownDims()) await this.deps.embedder.warmup();
    const finalDims = this.deps.embedder.knownDims() ?? 0;
    if (finalDims === 0) throw new Error("embedder has no dim after warmup");
    const model = this.deps.config.embeddingModel;

    const hash = contentHash(args.role ?? args.kind, args.text, finalDims, model);
    if (this.deps.store.hasContentHash(hash)) return;

    const [vec] = await this.deps.embedder.embed(args.text);
    if (!vec) return;

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
      snippet: makeSnippet(args.text),
      text_full: args.text,
      token_count: estimateTokens(args.text),
      model_name: model,
      model_dims: finalDims,
    });
    this.indexedThisCycle += 1;
    this.indexedTotal += 1;
  }

  /** Embed an entire batch of pending rows in a single inference call. */
  private async embedAndStoreBatch(batch: PendingRow[]): Promise<void> {
    if (batch.length === 0) return;
    if (!this.deps.embedder.knownDims()) await this.deps.embedder.warmup();
    const dims = this.deps.embedder.knownDims() ?? 0;
    if (dims === 0) throw new Error("embedder has no dim after warmup");

    // Bulk-skip rows already indexed (raced with hook path or another sweep).
    // One IN() query instead of N hasContentHash calls.
    const present = this.deps.store.whichHashesPresent(
      batch.map((p) => p.args.content_hash),
    );
    const fresh = batch.filter((p) => !present.has(p.args.content_hash));
    if (fresh.length === 0) return;

    const tEmbed0 = Date.now();
    const vectors = await this.deps.embedder.embed(fresh.map((p) => p.text));
    const tEmbedMs = Date.now() - tEmbed0;

    // Build the InsertArgs[] (with embedding + model_dims attached) and
    // hand them to the store as a single transaction.
    const items: InsertArgs[] = [];
    for (let i = 0; i < fresh.length; i++) {
      const v = vectors[i];
      const p = fresh[i];
      if (!v || !p) continue;
      items.push({ ...p.args, embedding: v, model_dims: dims });
    }

    const tInsert0 = Date.now();
    this.deps.store.insertBatch(items);
    const tInsertMs = Date.now() - tInsert0;
    this.indexedThisCycle += items.length;
    this.indexedTotal += items.length;

    // Diagnostic: surface batch timings so we can see embed-vs-SQL split.
    if (items.length >= 8 || tEmbedMs > 100 || tInsertMs > 100) {
      this.deps.log?.("sweep_batch", {
        size: items.length,
        embedMs: tEmbedMs,
        insertMs: tInsertMs,
      });
    }
  }

  private notify(message: string, level: "info" | "warning" | "error" = "info"): void {
    if (this.deps.config.debugMode || level !== "info") {
      this.deps.notify?.(`[pi-lcm-memory] ${message}`, level);
    }
  }
}

export const _testing = { SWEEP_BATCH, SWEEP_BACKOFF_MIN_MS, SWEEP_BACKOFF_MAX_MS };

function isToolIORole(role: string): boolean {
  return role === "toolResult" || role === "bashExecution";
}

function parseDate(iso: string | null | undefined): number | null {
  if (!iso) return null;
  const t = Date.parse(iso);
  return Number.isNaN(t) ? null : Math.floor(t / 1000);
}
