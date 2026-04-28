/**
 * pi-lcm-memory embedding worker.
 *
 * Owns the @huggingface/transformers feature-extraction pipeline. Runs in a
 * dedicated worker_threads thread so ONNX inference never blocks Pi's TUI
 * event loop. ONNX intra-op threading is configured to use most of the
 * available cores (leaving one for the main thread / Pi).
 *
 * Plain JS (.mjs) intentionally — workers spawn a fresh Node instance with
 * no TS loader. Keep this file dependency-free apart from
 * @huggingface/transformers and Node built-ins.
 *
 * Wire protocol (parent <-> worker) is JSON message envelopes:
 *
 *   parent → worker:
 *     { type: 'init', opts: { model, quantize, cacheDir } }
 *     { type: 'embed', id, texts }
 *     { type: 'shutdown' }
 *
 *   worker → parent:
 *     { type: 'progress', payload: { status, file, loaded, total, progress } }
 *     { type: 'loaded', dims, intraOpNumThreads, model }
 *     { type: 'result', id, dims, buffers }   // buffers transferred zero-copy
 *     { type: 'error', id?, message, stack? }
 */

import { parentPort } from "node:worker_threads";
import { availableParallelism } from "node:os";

if (!parentPort) {
  // Defensive: worker code must only run inside worker_threads.
  process.exit(1);
}

let pipe = null;
let modelName = "";
let pipeDims = null;
let intraOpNumThreads = 1;

/** @param {object} opts */
async function init(opts) {
  const tf = await import("@huggingface/transformers");

  if (tf.env) {
    if (opts.cacheDir) tf.env.cacheDir = opts.cacheDir;
    tf.env.allowRemoteModels = true;
    tf.env.allowLocalModels = true;
    // Quiet console.warn from Transformers.js: still logs to stdout if it has
    // something important to say (we already display "downloading" via
    // progress callbacks).
    if (tf.env.backends?.onnx?.logSeverityLevel != null) {
      tf.env.backends.onnx.logSeverityLevel = 3; // 3 = error only
    }
  }

  const cores = availableParallelism();
  // Use most cores for intra-op (the heavy matmul). Cap at 8: ORT's gains
  // are sub-linear past that for small models like bge-small. Leave at
  // least 1 core for the main thread + Pi.
  intraOpNumThreads = Math.max(1, Math.min(cores - 1, 8));

  const pipelineOpts = {
    progress_callback: (p) => {
      try {
        parentPort.postMessage({ type: "progress", payload: serializeProgress(p) });
      } catch {
        // best-effort
      }
    },
    session_options: {
      intraOpNumThreads,
      interOpNumThreads: 1,
      executionMode: "parallel",
      graphOptimizationLevel: "all",
    },
  };
  if (opts.quantize && opts.quantize !== "auto") {
    pipelineOpts.dtype = opts.quantize;
  }

  pipe = await tf.pipeline("feature-extraction", opts.model, pipelineOpts);
  modelName = opts.model;

  // Probe to discover dim if we don't already know.
  try {
    const probe = await pipe(["__pi_lcm_memory_probe__"], {
      pooling: "mean",
      normalize: true,
    });
    const arrs = tensorToFloat32Arrays(probe, 1);
    pipeDims = arrs[0]?.length ?? null;
  } catch (e) {
    // Probe failure is non-fatal; the next embed call may still work and
    // report dims via its result.
    pipeDims = null;
  }

  parentPort.postMessage({
    type: "loaded",
    dims: pipeDims,
    intraOpNumThreads,
    model: modelName,
  });
}

/**
 * @param {string[]} texts
 * @returns {Promise<Float32Array[]>}
 */
async function embedTexts(texts) {
  if (!pipe) throw new Error("worker not initialized");
  const out = await pipe(texts, { pooling: "mean", normalize: true });
  return tensorToFloat32Arrays(out, texts.length);
}

function serializeProgress(p) {
  if (!p || typeof p !== "object") return null;
  return {
    status: p.status,
    file: p.file,
    loaded: typeof p.loaded === "number" ? p.loaded : null,
    total: typeof p.total === "number" ? p.total : null,
    progress: typeof p.progress === "number" ? p.progress : null,
    name: p.name,
  };
}

/**
 * Convert a Transformers.js Tensor (or compatible) into Float32Array[],
 * one per input. Mirrors src/embeddings/embedder.ts logic but lives in
 * the worker so we don't ship a Tensor across the postMessage boundary.
 */
function tensorToFloat32Arrays(out, expectedCount) {
  if (!out) throw new Error("empty embedding output");
  const data = out.data ?? out.tensor?.data ?? out;
  const dims = out.dims ?? out.tensor?.dims;

  if (!data || typeof data.length !== "number") {
    throw new Error("unexpected embedding output: no .data");
  }

  if (dims && dims.length === 3) {
    // Unpooled [batch, seq, dim] — average pool over seq.
    const count = dims[0];
    const seq = dims[1];
    const dim = dims[2];
    const result = [];
    const buf = data;
    for (let b = 0; b < count; b++) {
      const acc = new Float32Array(dim);
      for (let s = 0; s < seq; s++) {
        const off = (b * seq + s) * dim;
        for (let d = 0; d < dim; d++) acc[d] += buf[off + d] ?? 0;
      }
      for (let d = 0; d < dim; d++) acc[d] /= seq;
      result.push(acc);
    }
    return result;
  }

  let count = expectedCount;
  let perVec = data.length / count;
  if (dims && dims.length === 2) {
    count = dims[0];
    perVec = dims[1];
  }

  if (!Number.isFinite(perVec) || perVec <= 0 || perVec * count !== data.length) {
    throw new Error(`embedding shape mismatch: total=${data.length} count=${count}`);
  }

  const result = [];
  for (let i = 0; i < count; i++) {
    // Copy into a fresh Float32Array so we own the buffer (and can transfer it).
    const slice = new Float32Array(perVec);
    slice.set(data.subarray(i * perVec, (i + 1) * perVec));
    result.push(slice);
  }
  return result;
}

parentPort.on("message", async (msg) => {
  if (!msg || typeof msg !== "object") return;
  const { type, id } = msg;

  try {
    if (type === "init") {
      await init(msg.opts ?? {});
    } else if (type === "embed") {
      const arrs = await embedTexts(msg.texts ?? []);
      const buffers = arrs.map((v) => v.buffer);
      parentPort.postMessage(
        { type: "result", id, dims: arrs[0]?.length ?? 0, buffers },
        buffers, // transfer (zero-copy)
      );
    } else if (type === "shutdown") {
      // Allow current frame to flush, then exit cleanly.
      setImmediate(() => process.exit(0));
    }
  } catch (e) {
    parentPort.postMessage({
      type: "error",
      id,
      message: e instanceof Error ? e.message : String(e),
      stack: e instanceof Error ? e.stack : null,
    });
  }
});
