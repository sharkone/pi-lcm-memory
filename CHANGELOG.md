# 📝 Changelog

All notable changes to this project will be documented in this file.

Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) · Versioning: [SemVer](https://semver.org/)

## [Unreleased]

### 🔬 Phase 6 — Cross-encoder reranker: evaluated and removed (post-mortem)

We built a complete cross-encoder reranker on top of the hybrid recall
stage (model `Xenova/ms-marco-MiniLM-L-6-v2`, second pipeline in the
existing worker, opt-in via `rerank: boolean`). Initial synthetic-eval
numbers were spectacular (MRR 0.364→1.000, nDCG@10 0.391→0.996), but a
follow-up real-data eval against the user's actual pi-lcm DB (749 msgs,
79 summaries, 57 queries derived from `summary_sources`) showed the
opposite: **0 of 57 queries improved, 48 regressed, 9 unchanged**.

**Root cause.** Cross-encoders trained on the (short query, long
passage) MS-MARCO distribution strongly prefer passages that are
stylistically similar to the query. pi-lcm summaries are LLM
paraphrases of past content — long, prose-styled. Whenever a
recall-style query is itself prose, the reranker promotes other
summaries above the actual messages they describe. Hybrid (FTS5 +
sqlite-vec + RRF) doesn't have this style-bias. Diagnostic confirmed
the reranker put 76% summaries / 24% messages in top-10, vs 5% / 95%
for hybrid alone.

**Findings (real-data eval, all configurations):**

      | configuration                                | MRR Δ   | nDCG@10 Δ |
      | -------------------------------------------- | ------: | --------: |
      | summary-style query, full corpus             |  -0.502 |    -0.178 |
      | summary-style query, messages-only corpus    |  -0.016 |    +0.009 |
      | keyword-style query,  full corpus            |  -0.269 |    -0.096 |
      | keyword-style query,  messages-only corpus   |  +0.139 |    +0.033 |

Reranker only wins (+21% MRR / +12% nDCG@10) under a narrow regime:
short keyword queries AND a corpus filtered to messages only. Even
then, latency overhead is ~580 ms per query (47× hybrid's 12 ms),
which is too expensive for auto-recall (fires every turn).

**What we kept.**

- `bench/lib/real-eval.ts` — generates a recall-quality eval set from
  any pi-lcm DB by walking `summary_sources` (summary text → source
  message ids). Includes optional TF×IDF keyword extraction
  (`queryStyle: "keywords"`) and corpus filtering. Broadly useful for
  any future retrieval work.
- `PI_LCM_MEMORY_BENCH_REAL_DB`, `PI_LCM_MEMORY_BENCH_REAL_QUERY_STYLE`,
  `PI_LCM_MEMORY_BENCH_REAL_MESSAGES_ONLY` env vars in `bench/quality.ts`.
- Historical bench result snapshots under `bench/results/` documenting
  the synthetic and real-data findings.

**What we removed.**

- All worker protocol additions (`init_reranker`, `rerank`,
  `reranker_loaded`, `rerank_result` message types and handlers).
- `Embedder.warmupReranker`, `rerank`, `rerankerState`, the `Reranker*`
  types, the `pendingRerank` map, `onRerankerLoaded` listener, and
  `spawnAndLoadReranker`.
- `Retriever`'s rerank branch (`applyRerank`, `RecallParams.rerank`,
  `RecallHit.rerank_score`, `RetrieverDeps.rerankEnabled`,
  `RetrieverDeps.rerankPoolSize`).
- Config keys: `rerank`, `rerankModel`, `rerankQuantize`,
  `rerankPoolSize` and their env overrides.
- `/memory rerank on|off` command and the two settings-panel rows.
- `test/retrieval.rerank.test.ts` and the live-rerank test in
  `test/worker.live.test.ts`.
- `bench/quality.ts` COMPARE mode (rerank-specific A/B harness).

**For future-me / future contributors.** If revisiting:

1. The summary-style bias is fundamental, not fixable by switching to
   another public cross-encoder — they're all trained on the same
   short-query / long-passage distribution.
2. Filtering the rerank pool to `source_kind='message'` papers over the
   bias but adds a config knob users won't tune.
3. The +12% nDCG@10 win on the favorable configuration costs ~580 ms /
   query. For an interactive auto-recall path, that's the wrong shape.
4. Time would probably be better spent tuning the existing hybrid (RRF
   k, lex/sem breadth, summary indexing strategy, FTS5 tokenizer) than
   adding a second model.
5. A domain-tuned reranker (trained on conversational paraphrase pairs
   instead of MS-MARCO) might work — but that's a research project,
   not a feature.

The bench infrastructure built for this turn-around is the real
deliverable: from now on, **every recall-quality claim must be backed
by a `bench/quality.ts` real-data run**, not just synthetic numbers.

### ✅ Added (Phase 6 — housekeeping + bench infra)

- **Performance benchmarks** (`bench/perf.ts`, `npm run bench:perf`).
  Captures `worker_warmup_ms`, `embed_throughput`,
  `embed_latency_b1_ms`, `embed_latency_b32_ms`, `sweep_throughput`,
  `recall_latency_ms`, `db_size_bytes_per_row`. Outputs JSON +
  markdown under `bench/results/`.
- **Recall quality benchmarks** (`bench/quality.ts`,
  `npm run bench:quality`). MRR, Recall@5/@10, Precision@5, nDCG@10
  over an eval set. Synthesised from `BENCH_TOPICS` if no
  `bench/eval/eval.json` is provided.
- **`bench/lib/metrics.ts`** — pure functions: reciprocalRank,
  recallAtK, precisionAtK, ndcgAtK, aggregate, percentiles. 17 unit
  tests in `test/bench.metrics.test.ts`.
- **End-to-end test harness** (`test/e2e/`). `makeFakePi()` faithful
  `ExtensionAPI` stub; `makeE2EProject()` tmp project + pre-seeded
  pi-lcm DB; `full-pipeline.test.ts` runs real worker + ONNX + DB,
  opt-in via `PI_LCM_MEMORY_LIVE_TEST=1`. Coverage: backfill,
  lcm_recall, lcm_similar, /memory commands, settings panel factory,
  message_end hook indexing.
- **Baseline snapshots** committed under `bench/results/` for diff vs.
  reranker.

### 🗑️ Removed (Phase 6 — housekeeping)

- `_testing` export from `src/indexer.ts` (no consumers).
- `iter_chunk` trace event in indexer's processBatched loop (was
  noisy: 2.18M lines / 173 MB on a single freeze pre-fix).
- `log` alias for `/memory events` (single canonical name).
- Un-exported orphan helpers in `src/db/connection.ts`
  (`getOpenDb`, `getOpenCwd`, `getDbPath`) and
  `src/embeddings/model-registry.ts` (`REGISTRY`).

### ✅ Added (Phases 0–1)
- Repo scaffolded.
- PLAN.md with locked architecture decisions (Q1–Q9).
- ROADMAP.md with phased delivery (Phases 0–5).
- Settings panel pulled into Phase 1 (parity with pi-lcm UX).
- README.md, LICENSE (MIT), .gitignore.
- **Phase 1 complete**: foundation + tools + UX shipped.
  - Shared DB w/ pi-lcm via per-cwd hash; additive `memory_vec` (sqlite-vec) +
    `memory_index` + `memory_meta` tables.
  - Hybrid recall (FTS5 ∪ sqlite-vec, RRF merged). `lcm_recall` and
    `lcm_similar` Pi tools.
  - Configurable embedder (default `Xenova/bge-small-en-v1.5`,
    `@huggingface/transformers` v3, lazy-loaded).
  - `message_end` hook + 30s sweep with idempotent dedup by `content_hash`.
  - Session-start primer (decision B) and heuristic auto-recall (decision D).
  - `/memory` and `/memory-settings` commands; TUI settings panel.
  - 44 vitest tests green: utils, config, settings, schema, store, bridge,
    indexer (hook + sweep), retrieval (lex/sem/hybrid + RRF + filters),
    primer (empty / populated), auto-recall (regex + budget).
  - Typechecks clean under strict TypeScript.
- **Phase 4 complete**: operator polish.
  - Batched sweep: 32 rows per inference call (was 1-by-1). ~30× backfill
    speedup at large corpus sizes.
  - Adaptive sweep interval: idle ticks back off (×2, capped at 5 min).
    `Indexer.kick()` resets immediately on commands, compaction, or model
    load completion.
  - First-time model download: progress events surface a one-shot "downloading
    embedding model…" notice with size, plus live `mem dl NN%` in the footer.
  - `/memory clear` requires `--yes`; `/memory reindex` clears + kicks the
    sweeper; `/memory model` writes settings, fires diagnostics, and is a
    no-op if the new model equals the current one.
  - `/memory status` reports cycles, indexed total, current sweep interval,
    idle streak, and last error.
  - `/memory events` exposes the rolling diagnostics ring (last 20 of 200)
    backed by `memory_meta`.
  - `PiLcmBridge.latestConversationId()` for lazy capture of the active pi-lcm
    conversation id so `lcm_recall(sessionFilter=...)` works.
  - Status footer shows download progress when relevant.
  - 60 vitest tests, all green (was 44). New suites: diagnostics,
    indexer.batch, commands, bridge.conv.

### 🔧 Changed (Phase 4 hotfix)
- **Default embedding dtype is now `q8`** (was `auto` → fp32 fallback on
  Node CPU, which produced a Transformers.js console warning and was ~4×
  slower). Quantized weights download instead of full precision; same
  retrieval quality, much faster.
- `embeddingQuantize` config now accepts the full Transformers.js v3 dtype
  enum: `auto | fp32 | fp16 | q8 | int8 | uint8 | q4 | q4f16`. Settings
  panel exposes the picker.

### ✅ Added (Phase 5 — worker thread embedder)
- **`src/embeddings/worker.mjs`** owns the @huggingface/transformers pipeline
  in a dedicated `worker_threads` thread. The main event loop is never
  blocked by ONNX inference; the TUI stays fully responsive during
  backfill.
- **Multi-core ORT**: worker configures the InferenceSession with
  `intraOpNumThreads = min(cpus()-1, 8)`, `interOpNumThreads = 1`,
  `executionMode = 'parallel'`, `graphOptimizationLevel = 'all'`. On an
  M-class machine the worker reports 8 threads.
- **Zero-copy vector transfer**: results come back as transferable
  `ArrayBuffer`s (one per Float32Array), no JSON-cloning of float arrays.
- **Sweep batch size restored to 32**. With the worker absorbing inference
  cost, the main thread can issue larger batches without ever blocking.
- **Live integration test** (`test/worker.live.test.ts`, opt-in via
  `PI_LCM_MEMORY_LIVE_TEST=1`): warmup, 5-row embed with semantic-order
  assertion, 32-row throughput probe, error propagation. On a real M-class
  laptop: 32 embeds in ~17 ms (~1880/s) with 8 threads.
- `Embedder.terminate()` rejects in-flight requests and shuts the worker
  down cleanly. Wired into `index.ts` `resetState()`.
- `EmbedderState` now exposes `intraOpNumThreads`. `model_loaded`
  diagnostic carries the thread count; init notification mentions it.

### 🐛 Fixed (Phase 5 — stabilization round)

- **Infinite-loop in `messagesNotInMemoryIndex` (THE big one).**
  The bridge generator yielded every row matching `mi.vec_rowid IS NULL`.
  Tool-I/O / empty-content rows were dropped by `bridgeMessageToPending`
  via `continue`, never inserted into `memory_index`, and so kept matching
  the LEFT JOIN forever. The for-of loop never reached a 32-row batch —
  no `await embedAndStoreBatch` ever fired, no event-loop yield, TUI
  starved at 100% of one core. Diagnosed via the side-channel tracer
  (2.18M `iter_chunk` events, zero `batch_start`).
  Three-part fix:
  1. SQL-level filter excludes tool-I/O roles and empty content directly
     in the bridge query.
  2. Rowid cursor (`m.rowid > :lastRowid` advanced before yield) so any
     row that slips through the filter and gets dropped by the consumer
     cannot be reconsidered in the same sweep.
  3. Safety yield in `processBatched` every 1024 iterated items regardless
     of batch fill — future bugs of this shape can't freeze the TUI again.
  Two regression tests in `test/indexer.batch.test.ts` (200 pure tool-I/O
  rows must terminate <2s; mixed 30 real + 150 tool-I/O indexes 30).

- **Dedupe leak: many-to-one pi-lcm id → vec_rowid mapping (schema v2).**
  `memory_index.content_hash` is UNIQUE, so two pi-lcm messages with
  identical content shared one embedding row — but only the first
  `pi_lcm_msg_id` was recorded. The bridge's LEFT JOIN kept yielding the
  duplicate id on every sweep; the indexer kept dropping it as a content
  duplicate; rinse, repeat.
  Schema v2 introduces side tables `memory_index_msg(pi_lcm_msg_id PK,
  vec_rowid)` and `memory_index_sum(pi_lcm_sum_id PK, vec_rowid)`. Multiple
  ids can map to the same vec_rowid. Migration backfills from existing
  `memory_index` rows. Bridge LEFT JOINs against the side tables. Indexer
  records mappings for both fresh inserts AND content-hash dupes via new
  `store.recordPresentMappings()`. New regression test verifies both ids
  for identical content map to one vec_rowid and the bridge yields zero
  rows on subsequent sweeps.

- **`/memory settings` crashed: `factory is not a function`.**
  pi's `ctx.ui.custom` API is `(factory, options)` where the factory is
  `(tui, theme, keybindings, done) => Component`. We were passing an
  object literal `{ overlay, component, onClose }`. Refactored
  `openSettingsPanel` to construct the panel inside a factory function,
  wired the panel's `onClose` to the `done` callback so Q/Esc cleanly
  closes the overlay.

- **Lock contention slashed: one transaction per batch.**
  `MemoryStore.insert()` ran 32 separate `db.transaction()` calls per
  batch, each grabbing the WAL write lock. With pi-lcm concurrently
  writing in another connection, this stacked busy_timeout retries.
  New `insertBatch(items[])` method does a single IMMEDIATE transaction
  for the whole batch with reused prepared statements. Bench: 200 inserts
  go from 15 ms → 1 ms (15× in isolation; much larger under contention).
  New `whichHashesPresent(hashes[])` does a bulk `IN()` lookup instead
  of N separate hash checks; returns Map<hash, vec_rowid> so the indexer
  can record dedupe mappings without a second query.

### 🔍 Added (debugging infrastructure)

- **Side-channel tracer** (`src/trace.ts`). Synchronous file-based event
  log enabled via `PI_LCM_MEMORY_TRACE=1` (default path
  `/tmp/pi-lcm-memory.<pid>.trace.log`) or `PI_LCM_MEMORY_TRACE=/path`.
  The worker writes to the same file (O_APPEND is safe across PIDs); each
  line carries `pid` + `src` for timeline correlation. Diagnoses freezes
  that block the main thread: SQLite-backed diagnostics can't run when
  the JS thread is stuck in a sync C call, but `fs.writeSync` snapshots
  every step right before the freeze. This is the tool that pinpointed
  the infinite-loop bug above. Notable trace events: `tick_start`,
  `warmup_start/end`, `process_start/end`, `batch_start`,
  `batch_dedupe`, `batch_embed_start/end`, `batch_insert_start/end`,
  `batch_done`, `safety_yield`, `embed_post`, `embed_resolve`,
  `worker_boot`, `init_pipeline_start/end`.

- **120s warmup watchdog**. If the embedder doesn't reach `loaded` in
  two minutes, it surfaces a clear error instead of hanging forever.
  Includes diagnostic state (`downloading=Y bytes=N`).

- **`/memory worker` debug command** prints embedder/worker state:
  ready, loading, downloading + bytes, thread id, worker pid, Node
  version, intra-op thread count, model, dims, resolved worker URL,
  and last error.

- **Per-MB download progress notifications.** During a long model
  download the user sees `[pi-lcm-memory] downloaded N MB…` every
  10 MB so the UI never feels frozen. `setStatus` is throttled to 4 Hz
  so progress events can't flood the TUI render queue.

- **Worker hello**. The worker posts a `{ type: "hello", threadId, pid,
  nodeVersion, cores }` message immediately on first execution. Confirms
  the worker actually spawned (vs. silently failed to construct) and
  feeds `/memory worker` state.
