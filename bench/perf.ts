/**
 * Performance benchmarks for pi-lcm-memory.
 *
 *   npm run bench:perf
 *   PI_LCM_MEMORY_BENCH_QUICK=1 npm run bench:perf      # smoke run
 *
 * Captures end-to-end numbers that actually matter to a live session:
 *   - worker warmup (cold + warm cache)
 *   - embed throughput (batch=32, warm worker)
 *   - embed latency p50/p99 (batch=1 = hook path; batch=32 = sweep path)
 *   - sweep throughput (read pi-lcm → dedupe → embed → batched insert)
 *   - recall latency p50/p99 (full hybrid pipeline)
 *   - db size bytes per row
 *
 * Outputs JSON + a markdown summary under bench/results/.
 */

import { writeFileSync, mkdirSync, existsSync } from "node:fs";
import { join } from "node:path";
import { execSync } from "node:child_process";
import { cpus, totalmem } from "node:os";

import { Embedder } from "../src/embeddings/embedder.js";
import { MemoryStore } from "../src/db/store.js";
import { PiLcmBridge } from "../src/bridge.js";
import { Indexer } from "../src/indexer.js";
import { Retriever } from "../src/retrieval.js";
import { DEFAULTS } from "../src/config.js";
import { isVecLoadedFor } from "../src/db/vec.js";

import { makeBenchDb, seedMessages, dbDiskBytes, BENCH_TOPICS } from "./lib/fixtures.js";
import { percentiles } from "./lib/metrics.js";

const QUICK = process.env.PI_LCM_MEMORY_BENCH_QUICK === "1";
const MODEL = process.env.PI_LCM_MEMORY_BENCH_MODEL ?? "Xenova/bge-small-en-v1.5";
const QUANTIZE = process.env.PI_LCM_MEMORY_BENCH_QUANTIZE ?? "q8";

// Sample sizes (knock down with QUICK for smoke testing).
const SAMPLES_LATENCY_B1 = QUICK ? 20 : 100;
const SAMPLES_LATENCY_B32 = QUICK ? 5 : 30;
const SWEEP_CORPUS_SIZE = QUICK ? 200 : 1000;
const RECALL_QUERIES = QUICK ? 20 : 100;

interface Result {
  bench: string;
  unit: string;
  value: number;
  detail?: Record<string, unknown>;
}

interface Report {
  meta: {
    git_sha: string;
    git_dirty: boolean;
    timestamp: string;
    quick: boolean;
    model: string;
    quantize: string;
    node_version: string;
    cpu_model: string;
    cpu_cores: number;
    total_memory_gb: number;
    sweep_corpus_size: number;
  };
  results: Result[];
}

async function main() {
  const meta = collectMeta();
  console.log(`pi-lcm-memory perf bench`);
  console.log(`  git: ${meta.git_sha}${meta.git_dirty ? " (dirty)" : ""}`);
  console.log(`  model: ${meta.model} (${meta.quantize})`);
  console.log(`  cpu: ${meta.cpu_model} × ${meta.cpu_cores}`);
  console.log(`  mode: ${QUICK ? "QUICK" : "FULL"}`);
  console.log("");

  const results: Result[] = [];

  // -------- Embedder warmup --------
  const embedder = new Embedder({
    model: MODEL,
    quantize: QUANTIZE as never,
    cacheDir: null,
  });
  const tWarm0 = Date.now();
  await embedder.warmup();
  const warmupMs = Date.now() - tWarm0;
  results.push({
    bench: "worker_warmup_ms",
    unit: "ms",
    value: warmupMs,
    detail: {
      note: "First warmup of the run. Cold-cache cost only on first model download.",
      threads: embedder.state().intraOpNumThreads,
    },
  });
  log(`worker_warmup_ms: ${warmupMs} (threads=${embedder.state().intraOpNumThreads})`);

  // -------- Embed throughput --------
  const throughputCorpus = makeCorpus(SAMPLES_LATENCY_B32 * 32);
  const tThru0 = Date.now();
  let totalEmbeds = 0;
  for (let i = 0; i < SAMPLES_LATENCY_B32; i++) {
    const slice = throughputCorpus.slice(i * 32, (i + 1) * 32);
    await embedder.embed(slice);
    totalEmbeds += slice.length;
  }
  const thruMs = Date.now() - tThru0;
  const throughput = (totalEmbeds * 1000) / Math.max(1, thruMs);
  results.push({
    bench: "embed_throughput",
    unit: "embeds/sec",
    value: throughput,
    detail: { batch_size: 32, batches: SAMPLES_LATENCY_B32, total_embeds: totalEmbeds, ms: thruMs },
  });
  log(`embed_throughput: ${throughput.toFixed(0)} embeds/sec (${SAMPLES_LATENCY_B32}×32 in ${thruMs}ms)`);

  // -------- Embed latency: batch=1 (hook path) --------
  const lat1 = await measureLatency(async (text) => {
    await embedder.embed(text);
  }, makeCorpus(SAMPLES_LATENCY_B1));
  results.push({
    bench: "embed_latency_b1_ms",
    unit: "ms",
    value: lat1.p50,
    detail: lat1,
  });
  log(`embed_latency_b1: p50=${lat1.p50.toFixed(1)}ms p99=${lat1.p99.toFixed(1)}ms (n=${lat1.count})`);

  // -------- Embed latency: batch=32 (sweep path) --------
  const lat32 = await measureLatency(async (texts) => {
    await embedder.embed(texts as string[]);
  }, range(SAMPLES_LATENCY_B32).map(() => makeCorpus(32)));
  results.push({
    bench: "embed_latency_b32_ms",
    unit: "ms",
    value: lat32.p50,
    detail: lat32,
  });
  log(`embed_latency_b32: p50=${lat32.p50.toFixed(1)}ms p99=${lat32.p99.toFixed(1)}ms (n=${lat32.count})`);

  // -------- Sweep throughput + DB size --------
  const sweepResult = await runSweep(embedder, MODEL);
  results.push({
    bench: "sweep_throughput",
    unit: "rows/sec",
    value: sweepResult.rowsPerSec,
    detail: {
      seeded: sweepResult.seeded,
      indexed: sweepResult.indexed,
      total_ms: sweepResult.totalMs,
    },
  });
  results.push({
    bench: "db_size_bytes_per_row",
    unit: "bytes",
    value: sweepResult.bytesPerRow,
    detail: {
      db_total_bytes: sweepResult.dbTotalBytes,
      indexed: sweepResult.indexed,
    },
  });
  log(
    `sweep_throughput: ${sweepResult.rowsPerSec.toFixed(0)} rows/sec ` +
      `(${sweepResult.indexed}/${sweepResult.seeded} in ${sweepResult.totalMs}ms)`,
  );
  log(
    `db_size_bytes_per_row: ${Math.round(sweepResult.bytesPerRow)} ` +
      `(total ${(sweepResult.dbTotalBytes / 1024 / 1024).toFixed(2)} MB / ${sweepResult.indexed} rows)`,
  );

  // -------- Recall latency (using the swept DB) --------
  const recall = await measureRecallLatency(sweepResult.retriever, RECALL_QUERIES);
  results.push({
    bench: "recall_latency_ms",
    unit: "ms",
    value: recall.p50,
    detail: recall,
  });
  log(`recall_latency: p50=${recall.p50.toFixed(1)}ms p99=${recall.p99.toFixed(1)}ms (n=${recall.count})`);

  // Cleanup
  sweepResult.cleanup();
  await embedder.terminate();

  // -------- Write outputs --------
  const report: Report = { meta, results };
  writeOutputs(report);
  console.log("");
  console.log(`Wrote bench/results/perf.${meta.git_sha}.json + perf.latest.md`);
}

function makeCorpus(n: number): string[] {
  const out: string[] = [];
  for (let i = 0; i < n; i++) {
    const topic = BENCH_TOPICS[i % BENCH_TOPICS.length]!;
    out.push(`Iteration ${i}: discussing ${topic} in detail with several supporting sentences for embedding realism.`);
  }
  return out;
}

function range(n: number): number[] {
  return Array.from({ length: n }, (_, i) => i);
}

async function measureLatency<T>(
  fn: (input: T) => Promise<unknown>,
  inputs: T[],
): Promise<ReturnType<typeof percentiles>> {
  // Warm: drop the first sample.
  if (inputs.length > 1) await fn(inputs[0]!);
  const samples: number[] = [];
  for (const inp of inputs) {
    const t0 = process.hrtime.bigint();
    await fn(inp);
    const ns = Number(process.hrtime.bigint() - t0);
    samples.push(ns / 1_000_000);
  }
  return percentiles(samples);
}

async function runSweep(
  embedder: Embedder,
  modelName: string,
): Promise<{
  seeded: number;
  indexed: number;
  totalMs: number;
  rowsPerSec: number;
  bytesPerRow: number;
  dbTotalBytes: number;
  retriever: Retriever;
  cleanup: () => Promise<void>;
}> {
  const dims = embedder.knownDims();
  if (!dims) throw new Error("embedder has no known dims after warmup");

  const bench = await makeBenchDb({ embeddingDim: dims, embeddingModel: modelName });
  if (!bench.vecLoaded || !isVecLoadedFor(bench.db)) {
    bench.cleanup();
    throw new Error("sqlite-vec failed to load on this platform; sweep bench cannot run");
  }

  seedMessages(bench.db, { count: SWEEP_CORPUS_SIZE });

  const store = new MemoryStore(bench.db);
  const bridge = new PiLcmBridge(bench.db);

  const indexer = new Indexer({
    store,
    embedder: embedder as never,
    bridge,
    config: { ...DEFAULTS, indexMessages: true, indexSummaries: false, skipToolIO: true },
    conversationId: () => "bench-conv-1",
    sessionStartedAt: () => 1700000000,
  });

  const t0 = Date.now();
  await indexer.tick();
  const totalMs = Date.now() - t0;
  const stats = store.stats();
  const dbBytes = dbDiskBytes(bench.dbPath);

  const retriever = new Retriever({
    db: bench.db,
    store,
    embedder: embedder as never,
    bridge,
    rrfK: DEFAULTS.rrfK,
  });

  return {
    seeded: SWEEP_CORPUS_SIZE,
    indexed: stats.indexed,
    totalMs,
    rowsPerSec: (stats.indexed * 1000) / Math.max(1, totalMs),
    bytesPerRow: stats.indexed === 0 ? 0 : dbBytes / stats.indexed,
    dbTotalBytes: dbBytes,
    retriever,
    cleanup: async () => {
      bench.cleanup();
    },
  };
}

async function measureRecallLatency(retriever: Retriever, n: number): Promise<ReturnType<typeof percentiles>> {
  const queries = range(n).map((i) => BENCH_TOPICS[i % BENCH_TOPICS.length]!);
  // Warm
  await retriever.recall({ query: queries[0]!, k: 10 });
  const samples: number[] = [];
  for (const q of queries) {
    const t0 = process.hrtime.bigint();
    await retriever.recall({ query: q, k: 10 });
    samples.push(Number(process.hrtime.bigint() - t0) / 1_000_000);
  }
  return percentiles(samples);
}

function collectMeta(): Report["meta"] {
  let sha = "unknown";
  let dirty = false;
  try {
    sha = execSync("git rev-parse --short HEAD", { stdio: ["ignore", "pipe", "ignore"] })
      .toString()
      .trim();
    const status = execSync("git status --porcelain", { stdio: ["ignore", "pipe", "ignore"] })
      .toString()
      .trim();
    dirty = status.length > 0;
  } catch {
    // not a git repo, leave as unknown
  }

  const cpuList = cpus();
  return {
    git_sha: sha,
    git_dirty: dirty,
    timestamp: new Date().toISOString(),
    quick: QUICK,
    model: MODEL,
    quantize: QUANTIZE,
    node_version: process.version,
    cpu_model: cpuList[0]?.model ?? "unknown",
    cpu_cores: cpuList.length,
    total_memory_gb: Math.round((totalmem() / 1024 / 1024 / 1024) * 10) / 10,
    sweep_corpus_size: SWEEP_CORPUS_SIZE,
  };
}

function writeOutputs(report: Report): void {
  const dir = join(process.cwd(), "bench", "results");
  if (!existsSync(dir)) mkdirSync(dir, { recursive: true });

  const jsonPath = join(dir, `perf.${report.meta.git_sha}${report.meta.quick ? ".quick" : ""}.json`);
  writeFileSync(jsonPath, JSON.stringify(report, null, 2));

  const mdPath = join(dir, "perf.latest.md");
  writeFileSync(mdPath, renderMarkdown(report));
}

function renderMarkdown(report: Report): string {
  const lines: string[] = [];
  lines.push(`# pi-lcm-memory · perf bench`);
  lines.push("");
  lines.push(`- git: \`${report.meta.git_sha}\`${report.meta.git_dirty ? " (dirty)" : ""}`);
  lines.push(`- timestamp: ${report.meta.timestamp}`);
  lines.push(`- model: \`${report.meta.model}\` (${report.meta.quantize})`);
  lines.push(`- node: ${report.meta.node_version}`);
  lines.push(`- cpu: ${report.meta.cpu_model} × ${report.meta.cpu_cores}`);
  lines.push(`- memory: ${report.meta.total_memory_gb} GB`);
  lines.push(`- mode: ${report.meta.quick ? "QUICK" : "FULL"} (sweep corpus: ${report.meta.sweep_corpus_size})`);
  lines.push("");
  lines.push(`| benchmark | value | unit |`);
  lines.push(`|---|---:|---|`);
  for (const r of report.results) {
    lines.push(`| ${r.bench} | ${formatVal(r.value, r.unit)} | ${r.unit} |`);
  }
  lines.push("");
  return lines.join("\n");
}

function formatVal(v: number, unit: string): string {
  if (unit === "embeds/sec" || unit === "rows/sec") return v.toFixed(0);
  if (unit === "bytes") return Math.round(v).toString();
  return v.toFixed(1);
}

function log(s: string): void {
  console.log(`  ${s}`);
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
