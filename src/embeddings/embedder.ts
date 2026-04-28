/**
 * Embedder over @huggingface/transformers v3. Lazy-loads the pipeline on
 * first call. Returns Float32Array vectors, normalized by default. Caches
 * weights under modelCacheDir (defaults to ~/.cache/pi-lcm-memory/models).
 *
 * Notes:
 *   - We import the package dynamically so the extension can run with
 *     `enabled: false` without pulling the whole ML stack into memory.
 *   - For unknown models we resolve dim by running a single inference pass
 *     and reading `output.dims`. The discovered dim is then memoized.
 */

import { homedir } from "node:os";
import { join } from "node:path";
import { mkdirSync } from "node:fs";
import { lookupModel } from "./model-registry.js";

import type { EmbeddingDtype } from "../config.js";

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
  error: string | null;
}

export interface EmbedderEventListener {
  onProgress?: (e: { file: string; loaded: number; total: number | null }) => void;
  onLoaded?: () => void;
  onError?: (msg: string) => void;
}

/* eslint-disable @typescript-eslint/no-explicit-any */
type Pipeline = (texts: string | string[], opts?: any) => Promise<any>;

export class Embedder {
  private opts: EmbedderOptions;
  private pipe: Pipeline | null = null;
  private dims: number | null = null;
  private loadPromise: Promise<void> | null = null;
  private error: string | null = null;
  private loading = false;
  private downloading = false;
  private downloadedBytes = 0;
  private totalBytes: number | null = null;
  private listener: EmbedderEventListener | null = null;

  constructor(opts: EmbedderOptions) {
    this.opts = opts;
    // Pre-fill dims from registry so callers can build the schema before load.
    const known = lookupModel(opts.model);
    if (known) this.dims = known.dims;
  }

  setListener(l: EmbedderEventListener | null): void {
    this.listener = l;
  }

  /** Resolve known dim without loading the model. Returns null if unknown. */
  knownDims(): number | null {
    return this.dims;
  }

  state(): EmbedderState {
    return {
      model: this.opts.model,
      dims: this.dims ?? 0,
      ready: this.pipe !== null,
      loading: this.loading,
      downloading: this.downloading,
      downloadedBytes: this.downloadedBytes,
      totalBytes: this.totalBytes,
      error: this.error,
    };
  }

  /** Trigger model load in the background. Safe to call repeatedly. */
  async warmup(): Promise<void> {
    if (this.pipe) return;
    if (!this.loadPromise) this.loadPromise = this.load();
    await this.loadPromise;
  }

  /** Embed a single text or a batch. Returns Float32Array per input. */
  async embed(input: string | string[]): Promise<Float32Array[]> {
    if (!this.pipe) await this.warmup();
    if (!this.pipe) {
      throw new Error(`embedder unavailable: ${this.error ?? "unknown error"}`);
    }
    const arr = Array.isArray(input) ? input : [input];
    if (arr.length === 0) return [];

    const out = await this.pipe(arr, {
      pooling: this.opts.pooling ?? "mean",
      normalize: this.opts.normalize ?? true,
    });

    return tensorToFloat32Arrays(out, arr.length);
  }

  private async load(): Promise<void> {
    this.loading = true;
    try {
      const cacheDir = this.opts.cacheDir ?? join(homedir(), ".cache", "pi-lcm-memory", "models");
      mkdirSync(cacheDir, { recursive: true });

      const tf: any = await import("@huggingface/transformers");
      if (tf.env) {
        tf.env.cacheDir = cacheDir;
        tf.env.allowLocalModels = true;
        tf.env.allowRemoteModels = true;
      }

      const pipelineOpts: Record<string, unknown> = {
        progress_callback: (p: any) => this.handleProgress(p),
      };
      // "auto" → don't pass dtype (Transformers.js picks the device default;
      // Node CPU defaults to fp32 with a console warning). Any other named
      // dtype is passed through verbatim.
      if (this.opts.quantize !== "auto") {
        pipelineOpts.dtype = this.opts.quantize;
      }

      const pipe = await tf.pipeline("feature-extraction", this.opts.model, pipelineOpts);
      this.pipe = pipe as Pipeline;

      if (this.dims == null) {
        const probe = await this.pipe("test", { pooling: "mean", normalize: true });
        const arrs = tensorToFloat32Arrays(probe, 1);
        this.dims = arrs[0]?.length ?? 0;
        if (!this.dims) throw new Error("could not determine embedding dim from probe");
      }
      this.downloading = false;
      this.listener?.onLoaded?.();
    } catch (e: unknown) {
      this.error = e instanceof Error ? e.message : String(e);
      this.pipe = null;
      this.listener?.onError?.(this.error);
      throw e;
    } finally {
      this.loading = false;
    }
  }

  private handleProgress(p: any): void {
    // Transformers.js progress events: { status: 'progress'|'download'|'done'|'ready',
    //   file?: string, progress?: number, loaded?: number, total?: number, name?: string }
    if (!p || typeof p !== "object") return;
    const status = (p as { status?: string }).status;
    const loaded = typeof (p as { loaded?: number }).loaded === "number" ? (p as { loaded: number }).loaded : 0;
    const total = typeof (p as { total?: number }).total === "number" ? (p as { total: number }).total : null;
    const file = typeof (p as { file?: string }).file === "string" ? (p as { file: string }).file : "";

    if (status === "download" || status === "progress") {
      this.downloading = true;
      if (loaded > this.downloadedBytes) this.downloadedBytes = loaded;
      if (total != null) this.totalBytes = total;
      this.listener?.onProgress?.({ file, loaded, total });
    } else if (status === "done" || status === "ready") {
      this.downloading = false;
    }
  }
}

function tensorToFloat32Arrays(out: any, expectedCount: number): Float32Array[] {
  if (!out) throw new Error("empty embedding output");

  // Common v3 shapes:
  //   - Tensor with .data (Float32Array | Float64Array | TypedArray) and .dims [N, D]
  //   - Tensor with pooled shape [N, D]
  //   - Plain object / array
  const data = out.data ?? out.tensor?.data ?? out;
  const dims: number[] | undefined = out.dims ?? out.tensor?.dims;

  if (!data || typeof (data as ArrayLike<number>).length !== "number") {
    throw new Error("unexpected embedding output: no .data / array");
  }

  const total = (data as ArrayLike<number>).length;
  let count = expectedCount;
  let perVec = total / count;

  if (dims && dims.length === 2) {
    count = dims[0]!;
    perVec = dims[1]!;
  } else if (dims && dims.length === 3) {
    // Unpooled [batch, seq, dim] should not happen if pooling='mean' was honoured,
    // but be defensive: average over seq.
    count = dims[0]!;
    const seq = dims[1]!;
    const dim = dims[2]!;
    const result: Float32Array[] = [];
    const buf = data as Float32Array;
    for (let b = 0; b < count; b++) {
      const acc = new Float32Array(dim);
      for (let s = 0; s < seq; s++) {
        const off = (b * seq + s) * dim;
        for (let d = 0; d < dim; d++) acc[d] = (acc[d] ?? 0) + (buf[off + d] ?? 0);
      }
      for (let d = 0; d < dim; d++) acc[d] = (acc[d] ?? 0) / seq;
      result.push(acc);
    }
    return result;
  }

  if (!Number.isFinite(perVec) || perVec <= 0 || perVec * count !== total) {
    throw new Error(`embedding shape mismatch: total=${total} count=${count}`);
  }

  const result: Float32Array[] = [];
  for (let i = 0; i < count; i++) {
    const slice = (data as Float32Array).slice(i * perVec, (i + 1) * perVec);
    result.push(new Float32Array(slice.buffer, slice.byteOffset, slice.length));
  }
  return result;
}
