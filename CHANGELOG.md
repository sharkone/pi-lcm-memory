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
