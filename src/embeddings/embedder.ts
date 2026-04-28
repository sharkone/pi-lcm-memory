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
  error: string | null;
}

export interface EmbedderEventListener {
  onProgress?: (e: { file: string; loaded: number; total: number | null }) => void;
  onLoaded?: () => void;
  onError?: (msg: string) => void;
}

interface PendingEmbed {
  resolve: (vectors: Float32Array[]) => void;
  reject: (e: Error) => void;
}

interface WorkerMessage {
  type: "progress" | "loaded" | "result" | "error";
  id?: number;
  payload?: unknown;
  dims?: number;
  intraOpNumThreads?: number;
  buffers?: ArrayBuffer[];
  message?: string;
  stack?: string;
  model?: string;
}

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
  private initResolve: (() => void) | null = null;
  private initReject: ((e: Error) => void) | null = null;

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
      error: this.error,
    };
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
    return new Promise<Float32Array[]>((resolve, reject) => {
      this.pending.set(id, { resolve, reject });
      this.worker!.postMessage({ type: "embed", id, texts: arr });
    });
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
      this.worker = new Worker(workerUrl);
      this.attachWorkerHandlers(this.worker);

      const initialized = new Promise<void>((resolve, reject) => {
        this.initResolve = resolve;
        this.initReject = reject;
      });

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
    }
  }

  private attachWorkerHandlers(worker: Worker): void {
    worker.on("message", (raw: unknown) => {
      if (!raw || typeof raw !== "object") return;
      const msg = raw as WorkerMessage;
      switch (msg.type) {
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
        case "error": {
          const text = msg.message ?? "worker error";
          if (typeof msg.id === "number") {
            const p = this.pending.get(msg.id);
            if (p) {
              this.pending.delete(msg.id);
              p.reject(new Error(text));
              return;
            }
          }
          // Init-time or unattributed error.
          this.error = text;
          this.listener?.onError?.(text);
          this.initReject?.(new Error(text));
          this.initResolve = null;
          this.initReject = null;
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
