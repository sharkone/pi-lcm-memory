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

import { parentPort, threadId } from "node:worker_threads";
import { availableParallelism, tmpdir } from "node:os";
import * as fs from "node:fs";
import * as path from "node:path";

if (!parentPort) {
  // Defensive: worker code must only run inside worker_threads.
  process.exit(1);
}

// ---------------------------------------------------------------------------
// Side-channel tracer (mirrors src/trace.ts; kept inline because workers can't
// import .ts files). Writes to the same file as the parent so timelines can
// be merged. Enabled via PI_LCM_MEMORY_TRACE env var.
// ---------------------------------------------------------------------------
let __traceFd = null;
(function initTrace() {
  const env = process.env.PI_LCM_MEMORY_TRACE;
  if (!env || env === "0" || env === "false") return;
  // When the parent uses the pid-based default path, parent.pid is the pi
  // process pid — but our worker pid is different. So use a fixed file name
  // when env is "1"/"true", scoped to the parent's pid via PPID.
  const ppid = process.ppid || process.pid;
  const file =
    env === "1" || env === "true"
      ? path.join(tmpdir(), `pi-lcm-memory.${ppid}.trace.log`)
      : env;
  try {
    fs.mkdirSync(path.dirname(file), { recursive: true });
    __traceFd = fs.openSync(file, "a");
  } catch {
    __traceFd = null;
  }
})();
function trace(event, data) {
  if (__traceFd == null) return;
  try {
    const obj = { t: Date.now(), pid: process.pid, src: "worker", event, ...(data ?? {}) };
    fs.writeSync(__traceFd, JSON.stringify(obj) + "\n");
  } catch {
    // ignore
  }
}
trace("worker_boot", { threadId, ppid: process.ppid, node: process.version });

let pipe = null;
let modelName = "";
let pipeDims = null;
let intraOpNumThreads = 1;

// Tell the parent we're alive immediately so it can confirm the worker
// actually spawned (vs. silently failed to construct).
parentPort.postMessage({
  type: "hello",
  threadId,
  pid: process.pid,
  nodeVersion: process.version,
  cores: availableParallelism(),
});

/** @param {object} opts */
async function init(opts) {
  trace("init_start", { model: opts?.model, quantize: opts?.quantize });
  const tImport0 = Date.now();
  const tf = await import("@huggingface/transformers");
  trace("init_imported", { ms: Date.now() - tImport0 });

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

  trace("init_pipeline_start", { intraOpNumThreads });
  const tPipe0 = Date.now();
  pipe = await tf.pipeline("feature-extraction", opts.model, pipelineOpts);
  modelName = opts.model;
  trace("init_pipeline_end", { ms: Date.now() - tPipe0 });

  // Probe to discover dim if we don't already know.
  try {
    trace("init_probe_start");
    const tProbe0 = Date.now();
    const probe = await pipe(["__pi_lcm_memory_probe__"], {
      pooling: "mean",
      normalize: true,
    });
    const arrs = tensorToFloat32Arrays(probe, 1);
    pipeDims = arrs[0]?.length ?? null;
    trace("init_probe_end", { ms: Date.now() - tProbe0, dims: pipeDims });
  } catch (e) {
    // Probe failure is non-fatal; the next embed call may still work and
    // report dims via its result.
    pipeDims = null;
    trace("init_probe_error", { error: e instanceof Error ? e.message : String(e) });
  }

  trace("init_done", { dims: pipeDims, intraOpNumThreads });
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
  trace("embed_start", { count: texts.length });
  const t0 = Date.now();
  const out = await pipe(texts, { pooling: "mean", normalize: true });
  const inferMs = Date.now() - t0;
  const t1 = Date.now();
  const arrs = tensorToFloat32Arrays(out, texts.length);
  const decodeMs = Date.now() - t1;
  trace("embed_end", { count: texts.length, inferMs, decodeMs });
  return arrs;
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
      trace("recv_embed", { id, count: (msg.texts ?? []).length });
      const arrs = await embedTexts(msg.texts ?? []);
      const buffers = arrs.map((v) => v.buffer);
      trace("send_result", { id, count: arrs.length });
      parentPort.postMessage(
        { type: "result", id, dims: arrs[0]?.length ?? 0, buffers },
        buffers, // transfer (zero-copy)
      );
    } else if (type === "shutdown") {
      trace("shutdown");
      // Allow current frame to flush, then exit cleanly.
      setImmediate(() => process.exit(0));
    }
  } catch (e) {
    trace("worker_error", {
      type,
      id,
      error: e instanceof Error ? e.message : String(e),
    });
    parentPort.postMessage({
      type: "error",
      id,
      message: e instanceof Error ? e.message : String(e),
      stack: e instanceof Error ? e.stack : null,
    });
  }
});
