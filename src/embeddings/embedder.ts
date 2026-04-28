/**
 * Embedder — main-thread controller for the worker_threads embedding worker.
 *
 * Responsibilities:
 *   - Spawn `worker.mjs` lazily on first warmup/embed.
 *   - Marshal embed requests via id-keyed pending Promises.
 *   - Forward Transformers.js progress events to the optional listener.
 *   - Decode zero-copy ArrayBuffer transfers back into Float32Array.
 *
 * Inference NEVER runs on the main thread. The TUI is unblocked while the
 * worker (with `intraOpNumThreads = cpus()-1`) saturates the CPU.
 */

import { Worker } from "node:worker_threads";
import { mkdirSync } from "node:fs";
import { homedir } from "node:os";
import { join } from "node:path";
import type { EmbeddingDtype } from "../config.js";
import { lookupModel } from "./model-registry.js";
import { trace } from "../trace.js";

export interface EmbedderOptions {
  model: string;
  quantize: EmbeddingDtype;
  cacheDir: string | null;
  pooling?: "mean" | "cls";
  normalize?: boolean;
}

export interface EmbedderState {
  model: string;
  dims: number;
  ready: boolean;
  loading: boolean;
  downloading: boolean;
  downloadedBytes: number;
  totalBytes: number | null;
  intraOpNumThreads: number | null;
  workerThreadId: number | null;
  workerPid: number | null;
  workerNodeVersion: string | null;
  error: string | null;
}

export interface EmbedderEventListener {
  onProgress?: (e: { file: string; loaded: number; total: number | null }) => void;
  onWorkerHello?: (e: { threadId: number; pid: number; nodeVersion: string; cores: number }) => void;
  onLoaded?: () => void;
  onRerankerLoaded?: (e: { model: string }) => void;
  onError?: (msg: string) => void;
}

export interface RerankerOptions {
  model: string;
  quantize: EmbeddingDtype;
}

export interface RerankerState {
  model: string;
  ready: boolean;
  loading: boolean;
  error: string | null;
}

interface PendingEmbed {
  resolve: (vectors: Float32Array[]) => void;
  reject: (e: Error) => void;
}

interface PendingRerank {
  resolve: (scores: number[]) => void;
  reject: (e: Error) => void;
}

interface WorkerMessage {
  type: "hello" | "progress" | "loaded" | "result" | "reranker_loaded" | "rerank_result" | "error";
  id?: number;
  payload?: unknown;
  dims?: number;
  intraOpNumThreads?: number;
  buffers?: ArrayBuffer[];
  scores?: number[];
  message?: string;
  stack?: string;
  model?: string;
  threadId?: number;
  pid?: number;
  nodeVersion?: string;
  cores?: number;
}

const WARMUP_TIMEOUT_MS = 120_000;

export class Embedder {
  private opts: EmbedderOptions;
  private worker: Worker | null = null;
  private dims: number | null = null;
  private intraOpNumThreads: number | null = null;
  private loadPromise: Promise<void> | null = null;
  private loading = false;
  private downloading = false;
  private downloadedBytes = 0;
  private totalBytes: number | null = null;
  private error: string | null = null;
  private listener: EmbedderEventListener | null = null;

  private nextId = 0;
  private pending = new Map<number, PendingEmbed>();
  private pendingRerank = new Map<number, PendingRerank>();
  private rerankerOpts: RerankerOptions | null = null;
  private rerankerLoaded = false;
  private rerankerLoading = false;
  private rerankerError: string | null = null;
  private rerankerLoadPromise: Promise<void> | null = null;
  private rerankerInitResolve: (() => void) | null = null;
  private rerankerInitReject: ((e: Error) => void) | null = null;
  private initResolve: (() => void) | null = null;
  private initReject: ((e: Error) => void) | null = null;
  private warmupTimer: NodeJS.Timeout | null = null;
  private workerThreadId: number | null = null;
  private workerPid: number | null = null;
  private workerNodeVersion: string | null = null;
  private lastWorkerUrl: string | null = null;

  constructor(opts: EmbedderOptions) {
    this.opts = opts;
    const known = lookupModel(opts.model);
    if (known) this.dims = known.dims;
  }

  setListener(l: EmbedderEventListener | null): void {
    this.listener = l;
  }

  /** Best-known dim; pre-filled from registry, refined after worker probes. */
  knownDims(): number | null {
    return this.dims;
  }

  state(): EmbedderState {
    return {
      model: this.opts.model,
      dims: this.dims ?? 0,
      ready: this.worker !== null && !this.loading,
      loading: this.loading,
      downloading: this.downloading,
      downloadedBytes: this.downloadedBytes,
      totalBytes: this.totalBytes,
      intraOpNumThreads: this.intraOpNumThreads,
      workerThreadId: this.workerThreadId,
      workerPid: this.workerPid,
      workerNodeVersion: this.workerNodeVersion,
      error: this.error,
    };
  }

  /** Where we tried to load worker.mjs from, for debugging. */
  workerUrl(): string | null {
    return this.lastWorkerUrl;
  }

  /** Spawn worker + load pipeline. Idempotent. */
  async warmup(): Promise<void> {
    if (this.worker && !this.loading) return;
    if (!this.loadPromise) this.loadPromise = this.spawnAndLoad();
    await this.loadPromise;
  }

  async embed(input: string | string[]): Promise<Float32Array[]> {
    const arr = Array.isArray(input) ? input : [input];
    if (arr.length === 0) return [];
    if (!this.worker || this.loading) await this.warmup();
    if (!this.worker) {
      throw new Error(`embedder unavailable: ${this.error ?? "unknown error"}`);
    }
    const id = ++this.nextId;
    const startedAt = Date.now();
    trace("embed_post", { id, count: arr.length });
    return new Promise<Float32Array[]>((resolve, reject) => {
      this.pending.set(id, {
        resolve: (v) => {
          trace("embed_resolve", { id, totalMs: Date.now() - startedAt });
          resolve(v);
        },
        reject: (e) => {
          trace("embed_reject", { id, totalMs: Date.now() - startedAt });
          reject(e);
        },
      });
      this.worker!.postMessage({ type: "embed", id, texts: arr });
    });
  }

  /**
   * Lazily warm up the reranker pipeline. No-op if already loaded with the
   * same options.
   */
  async warmupReranker(opts: RerankerOptions): Promise<void> {
    if (
      this.rerankerLoaded &&
      this.rerankerOpts &&
      this.rerankerOpts.model === opts.model &&
      this.rerankerOpts.quantize === opts.quantize
    ) {
      return;
    }
    if (this.rerankerLoadPromise) return this.rerankerLoadPromise;
    this.rerankerOpts = opts;
    this.rerankerLoadPromise = this.spawnAndLoadReranker();
    await this.rerankerLoadPromise;
  }

  /** Score (query, doc) pairs. Caller must have warmed up the reranker. */
  async rerank(query: string, docs: string[]): Promise<number[]> {
    if (docs.length === 0) return [];
    if (!this.rerankerLoaded) {
      throw new Error("reranker not warmed up; call warmupReranker() first");
    }
    if (!this.worker) {
      throw new Error(`embedder unavailable: ${this.error ?? "unknown error"}`);
    }
    const id = ++this.nextId;
    return new Promise<number[]>((resolve, reject) => {
      this.pendingRerank.set(id, { resolve, reject });
      this.worker!.postMessage({ type: "rerank", id, query, docs });
    });
  }

  rerankerState(): RerankerState | null {
    if (!this.rerankerOpts) return null;
    return {
      model: this.rerankerOpts.model,
      ready: this.rerankerLoaded,
      loading: this.rerankerLoading,
      error: this.rerankerError,
    };
  }

  /** Tear down the worker. Rejects any pending requests. */
  terminate(): void {
    for (const p of this.pending.values()) {
      try {
        p.reject(new Error("embedder terminated"));
      } catch {
        // ignore
      }
    }
    this.pending.clear();
    for (const p of this.pendingRerank.values()) {
      try {
        p.reject(new Error("embedder terminated"));
      } catch {
        // ignore
      }
    }
    this.pendingRerank.clear();
    if (this.rerankerInitReject) {
      try {
        this.rerankerInitReject(new Error("embedder terminated"));
      } catch {
        // ignore
      }
      this.rerankerInitResolve = null;
      this.rerankerInitReject = null;
    }
    this.rerankerLoaded = false;
    this.rerankerLoading = false;
    this.rerankerLoadPromise = null;
    if (this.initReject) {
      try {
        this.initReject(new Error("embedder terminated"));
      } catch {
        // ignore
      }
      this.initReject = null;
      this.initResolve = null;
    }
    const w = this.worker;
    this.worker = null;
    this.loadPromise = null;
    this.loading = false;
    if (w) {
      try {
        w.postMessage({ type: "shutdown" });
      } catch {
        // ignore
      }
      // Don't await — fire-and-forget; the worker exits via process.exit(0).
      w.terminate().catch(() => undefined);
    }
  }

  // ── internals ─────────────────────────────────────────────────────────────

  private async spawnAndLoad(): Promise<void> {
    this.loading = true;
    this.error = null;
    try {
      const cacheDir =
        this.opts.cacheDir ?? join(homedir(), ".cache", "pi-lcm-memory", "models");
      mkdirSync(cacheDir, { recursive: true });

      const workerUrl = new URL("./worker.mjs", import.meta.url);
      this.lastWorkerUrl = workerUrl.href;
      // Synchronous: if this throws (bad path, missing native module, etc),
      // we propagate immediately. The hello message confirms the worker
      // actually started executing.
      this.worker = new Worker(workerUrl);
      this.attachWorkerHandlers(this.worker);

      const initialized = new Promise<void>((resolve, reject) => {
        this.initResolve = resolve;
        this.initReject = reject;
      });

      // Watchdog: if the worker doesn't ack 'loaded' within the timeout,
      // surface an error rather than hanging forever.
      this.warmupTimer = setTimeout(() => {
        if (this.initReject) {
          const msg =
            `embedder warmup timed out after ${WARMUP_TIMEOUT_MS / 1000}s ` +
            `(downloading=${this.downloading} bytes=${this.downloadedBytes})`;
          this.error = msg;
          this.listener?.onError?.(msg);
          this.initReject(new Error(msg));
          this.initReject = null;
          this.initResolve = null;
        }
      }, WARMUP_TIMEOUT_MS);
      if (typeof this.warmupTimer.unref === "function") this.warmupTimer.unref();

      this.worker.postMessage({
        type: "init",
        opts: {
          model: this.opts.model,
          quantize: this.opts.quantize,
          cacheDir,
        },
      });

      await initialized;
    } catch (e: unknown) {
      this.error = e instanceof Error ? e.message : String(e);
      this.listener?.onError?.(this.error);
      if (this.worker) {
        try {
          this.worker.terminate().catch(() => undefined);
        } catch {
          // ignore
        }
        this.worker = null;
      }
      throw e;
    } finally {
      this.loading = false;
      if (this.warmupTimer) {
        clearTimeout(this.warmupTimer);
        this.warmupTimer = null;
      }
    }
  }

  private attachWorkerHandlers(worker: Worker): void {
    worker.on("message", (raw: unknown) => {
      if (!raw || typeof raw !== "object") return;
      const msg = raw as WorkerMessage;
      switch (msg.type) {
        case "hello":
          if (typeof msg.threadId === "number") this.workerThreadId = msg.threadId;
          if (typeof msg.pid === "number") this.workerPid = msg.pid;
          if (typeof msg.nodeVersion === "string") this.workerNodeVersion = msg.nodeVersion;
          this.listener?.onWorkerHello?.({
            threadId: msg.threadId ?? -1,
            pid: msg.pid ?? -1,
            nodeVersion: msg.nodeVersion ?? "",
            cores: msg.cores ?? 0,
          });
          break;
        case "progress":
          this.handleProgress(msg.payload);
          break;
        case "loaded":
          if (typeof msg.dims === "number" && msg.dims > 0) this.dims = msg.dims;
          if (typeof msg.intraOpNumThreads === "number") {
            this.intraOpNumThreads = msg.intraOpNumThreads;
          }
          this.downloading = false;
          this.initResolve?.();
          this.initResolve = null;
          this.initReject = null;
          this.listener?.onLoaded?.();
          break;
        case "result": {
          if (typeof msg.id !== "number") return;
          const p = this.pending.get(msg.id);
          if (!p) return;
          this.pending.delete(msg.id);
          const buffers = (msg.buffers as ArrayBuffer[] | undefined) ?? [];
          const vectors = buffers.map((buf) => new Float32Array(buf));
          if (vectors[0]?.length && !this.dims) this.dims = vectors[0].length;
          p.resolve(vectors);
          break;
        }
        case "reranker_loaded":
          this.rerankerLoaded = true;
          this.rerankerLoading = false;
          this.rerankerError = null;
          this.rerankerInitResolve?.();
          this.rerankerInitResolve = null;
          this.rerankerInitReject = null;
          this.listener?.onRerankerLoaded?.({
            model: typeof msg.model === "string" ? msg.model : this.rerankerOpts?.model ?? "",
          });
          break;
        case "rerank_result": {
          if (typeof msg.id !== "number") return;
          const p = this.pendingRerank.get(msg.id);
          if (!p) return;
          this.pendingRerank.delete(msg.id);
          const scores = Array.isArray(msg.scores) ? msg.scores.map(Number) : [];
          p.resolve(scores);
          break;
        }
        case "error": {
          const text = msg.message ?? "worker error";
          if (typeof msg.id === "number") {
            const p = this.pending.get(msg.id);
            if (p) {
              this.pending.delete(msg.id);
              p.reject(new Error(text));
              return;
            }
            const r = this.pendingRerank.get(msg.id);
            if (r) {
              this.pendingRerank.delete(msg.id);
              r.reject(new Error(text));
              return;
            }
          }
          // Init-time or unattributed error: reject both init promises
          // (whichever is active).
          this.error = text;
          this.listener?.onError?.(text);
          if (this.rerankerInitReject) {
            this.rerankerError = text;
            this.rerankerLoading = false;
            this.rerankerLoadPromise = null;
            this.rerankerInitReject(new Error(text));
            this.rerankerInitResolve = null;
            this.rerankerInitReject = null;
          }
          if (this.initReject) {
            this.initReject(new Error(text));
            this.initResolve = null;
            this.initReject = null;
          }
          break;
        }
      }
    });

    worker.on("error", (e: Error) => {
      const text = e instanceof Error ? e.message : String(e);
      this.error = text;
      this.listener?.onError?.(text);
      for (const p of this.pending.values()) {
        try {
          p.reject(e);
        } catch {
          // ignore
        }
      }
      this.pending.clear();
      this.initReject?.(e);
      this.initResolve = null;
      this.initReject = null;
      this.worker = null;
    });

    worker.on("exit", (code: number) => {
      if (code !== 0) {
        const text = `embedding worker exited with code ${code}`;
        this.error = text;
        for (const p of this.pending.values()) {
          try {
            p.reject(new Error(text));
          } catch {
            // ignore
          }
        }
        this.pending.clear();
      }
      this.worker = null;
      this.loadPromise = null;
    });
  }

  private async spawnAndLoadReranker(): Promise<void> {
    // The reranker shares the worker; we must have a live worker first.
    await this.warmup();
    if (!this.worker || !this.rerankerOpts) {
      throw new Error("reranker init: worker unavailable");
    }
    this.rerankerLoading = true;
    this.rerankerError = null;
    try {
      const initialized = new Promise<void>((resolve, reject) => {
        this.rerankerInitResolve = resolve;
        this.rerankerInitReject = reject;
      });
      this.worker.postMessage({
        type: "init_reranker",
        opts: {
          model: this.rerankerOpts.model,
          quantize: this.rerankerOpts.quantize,
        },
      });
      await initialized;
    } catch (e: unknown) {
      const text = e instanceof Error ? e.message : String(e);
      this.rerankerError = text;
      this.rerankerLoadPromise = null;
      throw e;
    } finally {
      this.rerankerLoading = false;
    }
  }

  private handleProgress(payload: unknown): void {
    if (!payload || typeof payload !== "object") return;
    const p = payload as {
      status?: string;
      file?: string;
      loaded?: number | null;
      total?: number | null;
    };
    const loaded = typeof p.loaded === "number" ? p.loaded : 0;
    const total = typeof p.total === "number" ? p.total : null;
    const file = typeof p.file === "string" ? p.file : "";

    if (p.status === "download" || p.status === "progress") {
      this.downloading = true;
      if (loaded > this.downloadedBytes) this.downloadedBytes = loaded;
      if (total != null) this.totalBytes = total;
      this.listener?.onProgress?.({ file, loaded, total });
    } else if (p.status === "done" || p.status === "ready") {
      this.downloading = false;
    }
  }
}
