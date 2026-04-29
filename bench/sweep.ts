/**
 * Hyperparameter sweep over RRF k and candidate breadth multipliers.
 *
 *   # Run against your real pi-lcm DB (recommended):
 *   PI_LCM_MEMORY_BENCH_REAL_DB=~/.pi/agent/lcm/<hash>.db npm run bench:sweep
 *
 *   # Synthetic fallback (no real DB needed):
 *   npm run bench:sweep
 *
 * Sweeps:
 *   rrfK    ∈ RRF_K_VALUES
 *   lexMult ∈ MULT_VALUES   (FTS5 candidate breadth = k * lexMult)
 *   semMult ∈ MULT_VALUES   (KNN candidate breadth  = k * semMult)
 *
 * Output:
 *   bench/results/sweep.<sha>.json  — raw results
 *   bench/results/sweep.latest.md   — sorted table, always overwritten
 *   stdout                          — top 10 + winner vs baseline
 */

import { writeFileSync, mkdirSync, existsSync } from "node:fs";
import { join } from "node:path";
import { execSync } from "node:child_process";

import { Embedder } from "../src/embeddings/embedder.js";
import { runMigrations } from "../src/db/schema.js";
import { MemoryStore } from "../src/db/store.js";
import { ensureVecLoaded, isVecLoadedFor } from "../src/db/vec.js";
import { PiLcmBridge } from "../src/bridge.js";
import { Retriever } from "../src/retrieval.js";
import type { InsertArgs } from "../src/db/store.js";

import { makeBenchDb, seedMessages, BENCH_TOPICS } from "./lib/fixtures.js";
import { aggregate, type QueryEval } from "./lib/metrics.js";
import { buildRealEvalSet } from "./lib/real-eval.js";

// ── Grid ─────────────────────────────────────────────────────────────────────

const RRF_K_VALUES = [10, 20, 30, 60, 120, 240];
const MULT_VALUES  = [2, 4, 8, 16];

// ── Env ───────────────────────────────────────────────────────────────────────

const MODEL    = process.env.PI_LCM_MEMORY_BENCH_MODEL    ?? "Xenova/bge-small-en-v1.5";
const QUANTIZE = process.env.PI_LCM_MEMORY_BENCH_QUANTIZE ?? "q8";
const REAL_DB  = process.env.PI_LCM_MEMORY_BENCH_REAL_DB  ?? null;
const REAL_QUERY_STYLE = (process.env.PI_LCM_MEMORY_BENCH_REAL_QUERY_STYLE ?? "keywords") as
  "summary" | "keywords";
const REAL_MESSAGES_ONLY = process.env.PI_LCM_MEMORY_BENCH_REAL_MESSAGES_ONLY === "1";
const REAL_MAX_QUERIES   = parseInt(process.env.PI_LCM_MEMORY_BENCH_REAL_MAX_QUERIES ?? "60", 10);
const REAL_KEYWORD_COUNT = parseInt(process.env.PI_LCM_MEMORY_BENCH_REAL_KEYWORD_COUNT ?? "8", 10);

// ── Types ─────────────────────────────────────────────────────────────────────

interface EvalQuery   { query: string; relevant: string[] }
interface EvalMessage { id: string; role: string; text: string }
interface EvalSummary { id: string; text: string; depth: number }
interface EvalSet {
  messages:   EvalMessage[];
  summaries?: EvalSummary[];
  queries:    EvalQuery[];
  source:     string;
}

interface SweepResult {
  rrfK:         number;
  lexMult:      number;
  semMult:      number;
  mrr:          number;
  recallAt5:    number;
  recallAt10:   number;
  ndcgAt10:     number;
  avgLatencyMs: number;
}

// ── Main ──────────────────────────────────────────────────────────────────────

async function main() {
  const totalCombos = RRF_K_VALUES.length * MULT_VALUES.length * MULT_VALUES.length;
  console.log("pi-lcm-memory · hyperparameter sweep");
  console.log(`  model:   ${MODEL} (${QUANTIZE})`);
  console.log(`  grid:    rrfK=${JSON.stringify(RRF_K_VALUES)}  mult=${JSON.stringify(MULT_VALUES)}`);
  console.log(`  combos:  ${totalCombos}`);
  console.log(`  source:  ${REAL_DB ? `real-db (${REAL_QUERY_STYLE})` : "synthetic"}`);
  console.log("");

  const evalSet = buildEvalSet();
  console.log(`  queries: ${evalSet.queries.length}  messages: ${evalSet.messages.length}  summaries: ${evalSet.summaries?.length ?? 0}`);
  console.log("");

  // Warm up the embedder first so we know dims for the DB schema
  const embedder = new Embedder({ model: MODEL, quantize: QUANTIZE as never, cacheDir: null });
  await embedder.warmup();
  const dims = embedder.knownDims();
  if (!dims) throw new Error("embedder has no known dims after warmup");

  // Shared on-disk DB seeded with eval messages
  const bench = await makeBenchDb({ embeddingDim: dims, embeddingModel: MODEL });
  if (!bench.vecLoaded || !isVecLoadedFor(bench.db)) {
    bench.cleanup();
    await embedder.terminate();
    throw new Error("sqlite-vec failed to load — cannot run sweep");
  }

  const db    = bench.db;
  const store = new MemoryStore(db);
  const bridge = new PiLcmBridge(db);

  // Seed eval messages into the pi-lcm messages table, then embed + store
  const seeded = seedMessages(db, {
    count: evalSet.messages.length,
    conversationId: "sweep",
    seed: 0xC0FFEE,
  });

  // Map seeded row ids back to eval message ids for embedding
  const texts   = seeded.map((r) => r.text);
  const vectors = await embedder.embed(texts);

  const insertItems: InsertArgs[] = seeded.map((r, i) => ({
    pi_lcm_msg_id:   r.id,
    pi_lcm_sum_id:   null,
    source_kind:     "message" as const,
    role:            r.role,
    depth:           null,
    conversation_id: "sweep",
    session_started: Math.floor(Date.now() / 1000),
    text_full:       r.text,
    snippet:         r.text.slice(0, 200),
    content_hash:    r.id,
    embedding:       vectors[i]!,
    model_name:      MODEL,
    model_dims:      dims,
  }));
  store.insertBatch(insertItems);

  // Run the grid
  const results: SweepResult[] = [];
  let done = 0;

  for (const rrfK of RRF_K_VALUES) {
    for (const lexMult of MULT_VALUES) {
      for (const semMult of MULT_VALUES) {
        const retriever = new Retriever({ db, store, embedder, bridge, rrfK, lexMult, semMult });
        const perQuery: QueryEval[] = [];
        let totalMs = 0;

        for (const q of evalSet.queries) {
          const t0 = performance.now();
          const hits = await retriever.recall({ query: q.query, k: 10, mode: "hybrid" });
          totalMs += performance.now() - t0;

          const ranked = hits.map((h) => h.pi_lcm_msg_id ?? h.pi_lcm_sum_id ?? "");
          perQuery.push({ query: q.query, relevant: q.relevant, ranked });
        }

        const agg = aggregate(perQuery);
        results.push({
          rrfK, lexMult, semMult,
          mrr:          agg.mrr,
          recallAt5:    agg.recallAt5,
          recallAt10:   agg.recallAt10,
          ndcgAt10:     agg.ndcgAt10,
          avgLatencyMs: totalMs / evalSet.queries.length,
        });

        done++;
        if (done % 10 === 0 || done === totalCombos) {
          process.stdout.write(`  ${done}/${totalCombos} combos...\r`);
        }
      }
    }
  }

  process.stdout.write("\n");
  bench.cleanup();
  await embedder.terminate();

  // Sort by MRR desc, break ties by nDCG
  results.sort((a, b) => b.mrr - a.mrr || b.ndcgAt10 - a.ndcgAt10);

  // Print top 10
  console.log("\nTop 10 combinations (by MRR):\n");
  console.log("  rank  rrfK  lexMult  semMult    MRR  Recall@10  nDCG@10  latency");
  console.log("  ────  ────  ───────  ───────  ─────  ─────────  ───────  ───────");
  for (let i = 0; i < Math.min(10, results.length); i++) {
    const r = results[i]!;
    console.log(
      `  ${String(i + 1).padStart(4)}  ${String(r.rrfK).padStart(4)}  ` +
      `${String(r.lexMult).padStart(7)}  ${String(r.semMult).padStart(7)}  ` +
      `${r.mrr.toFixed(3)}  ${r.recallAt10.toFixed(3).padStart(9)}  ` +
      `${r.ndcgAt10.toFixed(3).padStart(7)}  ${r.avgLatencyMs.toFixed(1).padStart(6)}ms`,
    );
  }

  const winner   = results[0]!;
  const baseline = results.find((r) => r.rrfK === 60 && r.lexMult === 4 && r.semMult === 4);

  console.log("\n🏆 Winner:");
  console.log(`   rrfK=${winner.rrfK}  lexMult=${winner.lexMult}  semMult=${winner.semMult}`);
  console.log(`   MRR=${winner.mrr.toFixed(3)}  Recall@10=${winner.recallAt10.toFixed(3)}  nDCG@10=${winner.ndcgAt10.toFixed(3)}`);

  if (baseline) {
    const mrrDelta  = (winner.mrr      - baseline.mrr)      / baseline.mrr      * 100;
    const ndcgDelta = (winner.ndcgAt10 - baseline.ndcgAt10) / baseline.ndcgAt10 * 100;
    console.log(`\n📊 vs current defaults (rrfK=60, lexMult=4, semMult=4):`);
    console.log(`   MRR:     ${baseline.mrr.toFixed(3)} → ${winner.mrr.toFixed(3)}  (${mrrDelta >= 0 ? "+" : ""}${mrrDelta.toFixed(1)}%)`);
    console.log(`   nDCG@10: ${baseline.ndcgAt10.toFixed(3)} → ${winner.ndcgAt10.toFixed(3)}  (${ndcgDelta >= 0 ? "+" : ""}${ndcgDelta.toFixed(1)}%)`);
  }

  writeOutputs(results, evalSet);
}

// ── Helpers ───────────────────────────────────────────────────────────────────

function buildEvalSet(): EvalSet {
  if (REAL_DB) {
    const built = buildRealEvalSet(REAL_DB, {
      minRelevant:   2,
      maxRelevant:   30,
      maxQueries:    REAL_MAX_QUERIES,
      maxQueryChars: 280,
      queryStyle:    REAL_QUERY_STYLE,
      keywordCount:  REAL_KEYWORD_COUNT,
    });
    if (built.queries.length > 0) {
      return {
        messages:  built.messages.map((m) => ({ id: m.id, role: m.role, text: m.text })),
        summaries: REAL_MESSAGES_ONLY
          ? []
          : built.summaries.map((s) => ({ id: s.id, text: s.text, depth: s.depth })),
        queries: built.queries.map((q) => ({ query: q.query, relevant: q.relevant })),
        source:  `real-db (${REAL_QUERY_STYLE})`,
      };
    }
    console.warn("real eval produced 0 queries — falling back to synthetic");
  }

  // Synthetic fallback
  const messages: EvalMessage[] = [];
  const queries:  EvalQuery[]   = [];
  const PER_TOPIC = 3;
  for (let t = 0; t < BENCH_TOPICS.length; t++) {
    const topic = BENCH_TOPICS[t]!;
    const ids: string[] = [];
    for (let i = 0; i < PER_TOPIC; i++) {
      const id = `t${t}-m${i}`;
      ids.push(id);
      messages.push({
        id,
        role: i % 2 === 0 ? "user" : "assistant",
        text: i === 0
          ? `We need to deal with ${topic} (entry ${id}). Let's discuss the implementation thoroughly.`
          : `Continuing from earlier (note ${id}): ${topic} requires careful attention to edge cases.`,
      });
    }
    queries.push({ query: topic, relevant: ids });
  }
  return { messages, queries, source: "synthetic" };
}

function writeOutputs(results: SweepResult[], evalSet: EvalSet): void {
  let sha = "unknown";
  try {
    sha = execSync("git rev-parse --short HEAD", { stdio: ["ignore", "pipe", "ignore"] })
      .toString().trim();
  } catch {}

  const dir = join(process.cwd(), "bench", "results");
  if (!existsSync(dir)) mkdirSync(dir, { recursive: true });

  writeFileSync(
    join(dir, `sweep.${sha}.json`),
    JSON.stringify({ sha, evalSource: evalSet.source, results }, null, 2),
  );
  writeFileSync(join(dir, "sweep.latest.md"), renderMarkdown(results, evalSet, sha));
  console.log(`\nWrote bench/results/sweep.${sha}.json + sweep.latest.md`);
}

function renderMarkdown(results: SweepResult[], evalSet: EvalSet, sha: string): string {
  const lines: string[] = [];
  lines.push("# pi-lcm-memory · hyperparameter sweep");
  lines.push("");
  lines.push(`- git: \`${sha}\``);
  lines.push(`- timestamp: ${new Date().toISOString()}`);
  lines.push(`- eval source: ${evalSet.source}`);
  lines.push(`- queries: ${evalSet.queries.length}`);
  lines.push(`- grid: rrfK=${JSON.stringify(RRF_K_VALUES)}  mult=${JSON.stringify(MULT_VALUES)}`);
  lines.push("");
  lines.push("## Results (sorted by MRR)");
  lines.push("");
  lines.push("| rank | rrfK | lexMult | semMult | MRR | Recall@5 | Recall@10 | nDCG@10 | latency |");
  lines.push("|-----:|-----:|--------:|--------:|----:|---------:|----------:|--------:|--------:|");

  const winner = results[0]!;
  for (let i = 0; i < results.length; i++) {
    const r = results[i]!;
    const trophy   = i === 0 ? " 🏆" : "";
    const isDefault = r.rrfK === 60 && r.lexMult === 4 && r.semMult === 4 ? " *(default)*" : "";
    lines.push(
      `| ${i + 1}${trophy} | ${r.rrfK} | ${r.lexMult} | ${r.semMult} | ${r.mrr.toFixed(3)}${isDefault} | ${r.recallAt5.toFixed(3)} | ${r.recallAt10.toFixed(3)} | ${r.ndcgAt10.toFixed(3)} | ${r.avgLatencyMs.toFixed(1)}ms |`,
    );
  }

  const baseline = results.find((r) => r.rrfK === 60 && r.lexMult === 4 && r.semMult === 4);
  if (baseline) {
    const mrrDelta  = (winner.mrr      - baseline.mrr)      / baseline.mrr      * 100;
    const ndcgDelta = (winner.ndcgAt10 - baseline.ndcgAt10) / baseline.ndcgAt10 * 100;
    lines.push("");
    lines.push("## vs current defaults (rrfK=60, lexMult=4, semMult=4)");
    lines.push("");
    lines.push(`| metric | baseline | winner | delta |`);
    lines.push(`|--------|----------|--------|-------|`);
    lines.push(`| MRR | ${baseline.mrr.toFixed(3)} | ${winner.mrr.toFixed(3)} | ${mrrDelta >= 0 ? "+" : ""}${mrrDelta.toFixed(1)}% |`);
    lines.push(`| nDCG@10 | ${baseline.ndcgAt10.toFixed(3)} | ${winner.ndcgAt10.toFixed(3)} | ${ndcgDelta >= 0 ? "+" : ""}${ndcgDelta.toFixed(1)}% |`);
  }

  return lines.join("\n") + "\n";
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
