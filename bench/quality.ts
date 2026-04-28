/**
 * Recall-quality benchmarks.
 *
 *   npm run bench:quality                # synthetic eval set from BENCH_TOPICS
 *   PI_LCM_MEMORY_BENCH_EVAL=path npm run bench:quality
 *
 * If `bench/eval/eval.json` (or the path in the env var) exists, it is used
 * as the eval set. Otherwise we synthesise one from `bench/lib/fixtures.ts`
 * topic strings: each topic is a query, the seeded message containing the
 * topic is the gold-relevant doc.
 *
 * Output: `bench/results/quality.<sha>.json` + `quality.latest.md`. Metrics:
 * MRR, recall@5, recall@10, precision@5, nDCG@10.
 *
 * Eval-set JSON shape:
 *
 *   {
 *     "messages":  [ { "id": "m1", "role": "user", "text": "..." }, ... ],
 *     "summaries": [ { "id": "s1", "text": "...", "depth": 1 } ],     // optional
 *     "queries":   [ { "query": "...", "relevant": ["m1", "s1"] }, ... ]
 *   }
 */

import { writeFileSync, mkdirSync, existsSync, readFileSync } from "node:fs";
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

import { makeBenchDb, BENCH_TOPICS } from "./lib/fixtures.js";
import { aggregate, type QueryEval } from "./lib/metrics.js";

const QUICK = process.env.PI_LCM_MEMORY_BENCH_QUICK === "1";
const MODEL = process.env.PI_LCM_MEMORY_BENCH_MODEL ?? "Xenova/bge-small-en-v1.5";
const QUANTIZE = process.env.PI_LCM_MEMORY_BENCH_QUANTIZE ?? "q8";

const DEFAULT_EVAL_PATH = join(process.cwd(), "bench", "eval", "eval.json");
const EVAL_PATH = process.env.PI_LCM_MEMORY_BENCH_EVAL ?? DEFAULT_EVAL_PATH;

interface EvalMessage {
  id: string;
  role?: string;
  text: string;
}

interface EvalSummary {
  id: string;
  text: string;
  depth?: number;
}

interface EvalQuery {
  query: string;
  /** IDs of the messages or summaries that *should* be in the top results. */
  relevant: string[];
}

interface EvalSet {
  messages: EvalMessage[];
  summaries?: EvalSummary[];
  queries: EvalQuery[];
  source: "file" | "synthetic";
}

interface QueryResult {
  query: string;
  relevant: string[];
  ranked: string[];
  scores: number[];
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
    eval_source: "file" | "synthetic";
    eval_path: string | null;
    n_messages: number;
    n_summaries: number;
    n_queries: number;
  };
  aggregate: ReturnType<typeof aggregate>;
  per_query: QueryResult[];
}

async function main() {
  const evalSet = loadEvalSet();
  console.log(`pi-lcm-memory quality bench`);
  console.log(`  eval source: ${evalSet.source} (${evalSet.queries.length} queries, ${evalSet.messages.length} messages)`);
  console.log(`  model: ${MODEL} (${QUANTIZE})`);
  console.log("");

  const embedder = new Embedder({
    model: MODEL,
    quantize: QUANTIZE as never,
    cacheDir: null,
  });
  await embedder.warmup();
  const dims = embedder.knownDims();
  if (!dims) throw new Error("embedder has no known dims after warmup");

  const bench = await makeBenchDb({ embeddingDim: dims, embeddingModel: MODEL });
  if (!bench.vecLoaded || !isVecLoadedFor(bench.db)) {
    bench.cleanup();
    await embedder.terminate();
    throw new Error("sqlite-vec failed to load on this platform; quality bench cannot run");
  }

  // Seed eval set into pi-lcm tables.
  const conv = "quality-bench-conv";
  bench.db.prepare(
    `INSERT OR IGNORE INTO conversations(id, session_id, cwd, created_at, updated_at)
       VALUES (?, ?, ?, ?, ?)`,
  ).run(conv, "s-" + conv, "/x", "2026-04-20T10:00:00Z", "2026-04-20T11:00:00Z");
  const insertMsg = bench.db.prepare(
    `INSERT INTO messages(id, conversation_id, role, content_text, timestamp, seq)
       VALUES (?, ?, ?, ?, ?, ?)`,
  );
  bench.db.transaction(() => {
    let seq = 0;
    for (const m of evalSet.messages) {
      insertMsg.run(m.id, conv, m.role ?? "user", m.text, 1700000000 + seq, seq);
      seq++;
    }
  })();

  if (evalSet.summaries?.length) {
    const insertSum = bench.db.prepare(
      `INSERT INTO summaries(id, conversation_id, depth, text, created_at)
         VALUES (?, ?, ?, ?, ?)`,
    );
    bench.db.transaction(() => {
      for (const s of evalSet.summaries!) {
        insertSum.run(s.id, conv, s.depth ?? 1, s.text, "2026-04-25T12:00:00Z");
      }
    })();
  }

  // Run the indexer to ingest everything.
  const store = new MemoryStore(bench.db);
  const bridge = new PiLcmBridge(bench.db);
  const indexer = new Indexer({
    store,
    embedder: embedder as never,
    bridge,
    config: { ...DEFAULTS, indexMessages: true, indexSummaries: true, skipToolIO: true },
    conversationId: () => conv,
    sessionStartedAt: () => 1700000000,
  });
  console.log("  indexing eval corpus...");
  const tIngest0 = Date.now();
  await indexer.tick();
  console.log(`  indexed ${store.stats().indexed} rows in ${Date.now() - tIngest0}ms`);

  const retriever = new Retriever({
    db: bench.db,
    store,
    embedder: embedder as never,
    bridge,
    rrfK: DEFAULTS.rrfK,
  });

  // Run each query.
  const perQuery: QueryResult[] = [];
  const k = QUICK ? 10 : 20;
  console.log(`  running ${evalSet.queries.length} queries (k=${k})...`);
  for (const q of evalSet.queries) {
    const hits = await retriever.recall({ query: q.query, k });
    const ranked = hits.map((h) => h.pi_lcm_msg_id ?? h.pi_lcm_sum_id ?? "");
    perQuery.push({
      query: q.query,
      relevant: q.relevant,
      ranked,
      scores: hits.map((h) => h.score),
    });
  }

  const evals: QueryEval[] = perQuery.map((q) => ({
    query: q.query,
    ranked: q.ranked,
    relevant: q.relevant,
  }));
  const agg = aggregate(evals);

  console.log("");
  console.log(`  aggregate metrics (n=${agg.queries}):`);
  console.log(`    mrr           = ${agg.mrr.toFixed(3)}`);
  console.log(`    recall@5      = ${agg.recallAt5.toFixed(3)}`);
  console.log(`    recall@10     = ${agg.recallAt10.toFixed(3)}`);
  console.log(`    precision@5   = ${agg.precisionAt5.toFixed(3)}`);
  console.log(`    ndcg@10       = ${agg.ndcgAt10.toFixed(3)}`);

  bench.cleanup();
  await embedder.terminate();

  const report: Report = {
    meta: {
      ...collectMeta(),
      eval_source: evalSet.source,
      eval_path: evalSet.source === "file" ? EVAL_PATH : null,
      n_messages: evalSet.messages.length,
      n_summaries: evalSet.summaries?.length ?? 0,
      n_queries: evalSet.queries.length,
    },
    aggregate: agg,
    per_query: perQuery,
  };
  writeOutputs(report);
  console.log("");
  console.log(`Wrote bench/results/quality.${report.meta.git_sha}.json + quality.latest.md`);
}

function loadEvalSet(): EvalSet {
  if (existsSync(EVAL_PATH)) {
    try {
      const txt = readFileSync(EVAL_PATH, "utf8");
      const parsed = JSON.parse(txt) as Partial<EvalSet>;
      if (!parsed.messages || !parsed.queries) {
        throw new Error("eval.json missing 'messages' or 'queries'");
      }
      const out: EvalSet = {
        messages: parsed.messages as EvalMessage[],
        queries: parsed.queries as EvalQuery[],
        source: "file",
      };
      if (parsed.summaries) out.summaries = parsed.summaries as EvalSummary[];
      return out;
    } catch (e) {
      console.error(`failed to load ${EVAL_PATH}: ${e}`);
      process.exit(1);
    }
  }
  return synthesizeEvalSet();
}

/**
 * Build a synthetic eval set from BENCH_TOPICS. For each topic, seed K
 * messages and mark exactly one as the gold-relevant doc for the query
 * that *is* the topic phrase. The other K-1 are noise (different topics).
 *
 * This is a weak baseline — real evaluation needs hand-curated queries —
 * but it's enough to detect catastrophic regressions and to compare
 * configurations against each other on the *same* synthetic ground truth.
 */
function synthesizeEvalSet(): EvalSet {
  const messages: EvalMessage[] = [];
  const queries: EvalQuery[] = [];
  const PER_TOPIC = QUICK ? 5 : 10;
  const TOTAL_NOISE = QUICK ? 30 : 80;

  // Add noise messages first (no gold relevance).
  for (let i = 0; i < TOTAL_NOISE; i++) {
    const noiseTopic = BENCH_TOPICS[(i * 7) % BENCH_TOPICS.length]!;
    messages.push({
      id: `noise-${i}`,
      role: "assistant",
      text: `Background note ${i}: a passing mention of ${noiseTopic} in another context.`,
    });
  }

  // For each topic, add PER_TOPIC messages, mark them as relevant for the
  // topic-phrase query.
  for (let t = 0; t < BENCH_TOPICS.length; t++) {
    const topic = BENCH_TOPICS[t]!;
    const ids: string[] = [];
    for (let i = 0; i < PER_TOPIC; i++) {
      const id = `t${t}-m${i}`;
      ids.push(id);
      // Each message must be uniquely worded; otherwise content_hash dedupes
      // them into one row and the eval undercounts recall.
      messages.push({
        id,
        role: i % 2 === 0 ? "user" : "assistant",
        text:
          i === 0
            ? `We need to deal with ${topic} (entry ${id}). Let's discuss the implementation thoroughly with concrete examples and trade-offs.`
            : `Continuing from earlier (note ${id}): ${topic} requires attention to detail. Worth digging into edge cases and failure modes around iteration ${i}.`,
      });
    }
    queries.push({ query: topic, relevant: ids });
  }

  return { messages, queries, source: "synthetic" };
}

function collectMeta() {
  let sha = "unknown";
  let dirty = false;
  try {
    sha = execSync("git rev-parse --short HEAD", { stdio: ["ignore", "pipe", "ignore"] }).toString().trim();
    dirty = execSync("git status --porcelain", { stdio: ["ignore", "pipe", "ignore"] }).toString().trim().length > 0;
  } catch {
    // not a git repo
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
  };
}

function writeOutputs(report: Report): void {
  const dir = join(process.cwd(), "bench", "results");
  if (!existsSync(dir)) mkdirSync(dir, { recursive: true });
  const suffix = report.meta.quick ? ".quick" : "";
  const jsonPath = join(dir, `quality.${report.meta.git_sha}${suffix}.json`);
  writeFileSync(jsonPath, JSON.stringify(report, null, 2));
  writeFileSync(join(dir, "quality.latest.md"), renderMarkdown(report));
}

function renderMarkdown(report: Report): string {
  const lines: string[] = [];
  lines.push(`# pi-lcm-memory · quality bench`);
  lines.push("");
  lines.push(`- git: \`${report.meta.git_sha}\`${report.meta.git_dirty ? " (dirty)" : ""}`);
  lines.push(`- timestamp: ${report.meta.timestamp}`);
  lines.push(`- model: \`${report.meta.model}\` (${report.meta.quantize})`);
  lines.push(`- eval source: ${report.meta.eval_source}${report.meta.eval_path ? ` (\`${report.meta.eval_path}\`)` : ""}`);
  lines.push(`- corpus: ${report.meta.n_messages} messages + ${report.meta.n_summaries} summaries`);
  lines.push(`- queries: ${report.meta.n_queries}`);
  lines.push(`- mode: ${report.meta.quick ? "QUICK" : "FULL"}`);
  lines.push("");
  lines.push(`## Aggregate`);
  lines.push("");
  lines.push(`| metric | value |`);
  lines.push(`|---|---:|`);
  lines.push(`| MRR | ${report.aggregate.mrr.toFixed(3)} |`);
  lines.push(`| Recall@5 | ${report.aggregate.recallAt5.toFixed(3)} |`);
  lines.push(`| Recall@10 | ${report.aggregate.recallAt10.toFixed(3)} |`);
  lines.push(`| Precision@5 | ${report.aggregate.precisionAt5.toFixed(3)} |`);
  lines.push(`| nDCG@10 | ${report.aggregate.ndcgAt10.toFixed(3)} |`);
  lines.push("");
  return lines.join("\n");
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
