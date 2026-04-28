# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

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
