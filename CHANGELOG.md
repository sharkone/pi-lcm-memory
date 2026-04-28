# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added (Phase 6 — cross-encoder reranker)

- **Cross-encoder reranker** (opt-in). Default model
  `Xenova/ms-marco-MiniLM-L-6-v2` runs in the existing embedder worker as
  a second pipeline (lazy-loaded; zero cost when off). When enabled, the
  hybrid recall stage fetches a wider candidate pool (default 30) and the
  cross-encoder reorders the top-K. Falls through to hybrid order on any
  reranker error so a broken reranker can never silently hurt recall.
  Wire-protocol additions to the worker: `init_reranker`, `rerank`,
  `reranker_loaded`, `rerank_result`. Embedder API additions:
  `warmupReranker(opts)`, `rerank(query, docs)`, `rerankerState()`,
  `EmbedderEventListener.onRerankerLoaded`. Retriever API additions:
  `RecallParams.rerank`, `RecallHit.rerank_score`,
  `RetrieverDeps.rerankEnabled`, `RetrieverDeps.rerankPoolSize`. Config
  additions: `rerank: boolean` (default `false`), `rerankModel: string`
  (default `Xenova/ms-marco-MiniLM-L-6-v2`), `rerankQuantize:
  EmbeddingDtype` (default `q8`), `rerankPoolSize: number` (default 30,
  range 1–200). Env overrides: `PI_LCM_MEMORY_RERANK={0|1}`,
  `PI_LCM_MEMORY_RERANK_MODEL`, `PI_LCM_MEMORY_RERANK_QUANTIZE`,
  `PI_LCM_MEMORY_RERANK_POOL`.

- **`/memory rerank on|off`** shortcut command. Reads current state with
  no args; toggles + persists at the active settings scope when given
  `on`/`off`. Settings panel grew two rows (`Rerank` boolean,
  `Rerank pool` number).

- **Quality results.** On the synthetic eval (230 msgs, 15 queries,
  k=20) at sha `06298f8`:

      | metric      | hybrid | + rerank | Δ          |
      | ----------- | -----: | -------: | ----------- |
      | MRR         |  0.364 |    1.000 | +0.636 (+175%) |
      | Recall@5    |  0.113 |    0.500 | +0.387 (+342%) |
      | Recall@10   |  0.480 |    0.993 | +0.513 (+107%) |
      | Precision@5 |  0.227 |    1.000 | +0.773 (+340%) |
      | nDCG@10     |  0.391 |    0.996 | +0.605 (+155%) |

  Live-test throughput on M5 × 8 worker threads, q8 cross-encoder:
  30 (query, doc) pairs in ~15 ms (≈2000 pairs/sec). Reranker init time
  is ~30 ms warm + ~3.5 MB model download cold.

- **Bench harness extended.** `bench/quality.ts` now runs with rerank
  on/off via `PI_LCM_MEMORY_BENCH_RERANK=1`. JSON output filenames
  carry a `.rerank` suffix when applicable. Markdown summary shows
  reranker model + pool size in the header.

- **Tests.** `test/retrieval.rerank.test.ts` adds 5 unit tests with a
  `FakeRerankerEmbedder` that exercise: rerank-off path, rerank-on
  reorders correctly, `params.rerank` overrides config, fall-through
  on thrown error, fall-through on score-count mismatch.
  `test/worker.live.test.ts` adds one live test (gated on
  `PI_LCM_MEMORY_LIVE_TEST=1`) that loads the cross-encoder, scores
  the canonical Berlin/NYC pair, and runs a 30-pair throughput probe.
  Total tests now: 87 passing (was 82), 11 skipped (was 10).

### Added (Phase 6 — housekeeping + bench infra)

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

### Removed (Phase 6 — housekeeping)

- `_testing` export from `src/indexer.ts` (no consumers).
- `iter_chunk` trace event in indexer's processBatched loop (was
  noisy: 2.18M lines / 173 MB on a single freeze pre-fix).
- `log` alias for `/memory events` (single canonical name).
- Un-exported orphan helpers in `src/db/connection.ts`
  (`getOpenDb`, `getOpenCwd`, `getDbPath`) and
  `src/embeddings/model-registry.ts` (`REGISTRY`).

### Added
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

### Changed (Phase 4 hotfix)
- **Default embedding dtype is now `q8`** (was `auto` → fp32 fallback on
  Node CPU, which produced a Transformers.js console warning and was ~4×
  slower). Quantized weights download instead of full precision; same
  retrieval quality, much faster.
- `embeddingQuantize` config now accepts the full Transformers.js v3 dtype
  enum: `auto | fp32 | fp16 | q8 | int8 | uint8 | q4 | q4f16`. Settings
  panel exposes the picker.

### Added (Phase 5 — worker thread embedder)
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

### Fixed (Phase 5 — stabilization round)

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

### Added (debugging infrastructure)

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
