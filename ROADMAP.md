# pi-lcm-memory — ROADMAP

Phased delivery. Each phase ships a working extension; later phases are pure
additions (no schema rewrites unless explicitly noted).

Cross-references the [PLAN](./PLAN.md) for design details.

---

## Phase 0 — Bootstrap

**Goal:** Repo, docs, and skeleton that compiles and loads as a no-op Pi
extension.

- [x] Interview locked (Q1–Q8).
- [x] `package.json`, `tsconfig.json`, `.gitignore`, `LICENSE` (MIT).
- [x] `README.md` (intent + install pointer).
- [x] `PLAN.md`, `ROADMAP.md`, `CHANGELOG.md`.
- [x] Git repo init with `sharkone@en-mousse.org`.
- [x] First commit: docs + scaffolding only.
- [ ] CI / pre-commit (defer; optional).

**Exit criteria:** repo builds cleanly with `tsc --noEmit`. Loads as a Pi
extension and prints a single `[pi-lcm-memory] enabled (v0)` notice on
`session_start`.

---

## Phase 1 — MVP: passive index + tool-based recall (decisions A + 6) + B + D

**Goal:** Every new message embedded; `lcm_recall` returns top-K hybrid
results. Session-start primer (B) and heuristic auto-recall (D) folded in
as the user opted for A+B+D in Q8.

- [x] `src/db/connection.ts` — open shared DB by mirroring pi-lcm's
      `hashCwd(cwd)` path scheme.
- [x] `src/db/schema.ts` — additive migrations: `memory_vec`, `memory_index`,
      `memory_meta`. Idempotent.
- [x] `src/db/vec.ts` — load `sqlite-vec` extension; soft-fail with clear
      message if unavailable.
- [x] `src/embeddings/embedder.ts` — Transformers.js pipeline, lazy load,
      configurable model.
- [x] `src/bridge.ts` — read-only access to pi-lcm `messages` + `summaries`.
- [x] `src/indexer.ts` — `message_end` hook + 30s sweep loop.
- [x] `src/retrieval.ts` — RRF merge of FTS5 + vec.
- [x] `src/tools/lcm-recall.ts` — registered via `pi.registerTool`.
- [x] `src/tools/lcm-similar.ts` — registered via `pi.registerTool`.
- [x] `src/settings-panel.ts` — interactive panel (parity with pi-lcm UX).
- [x] `src/commands.ts` — `/memory` dispatcher (stats, search, model, reindex, clear, status, settings).
- [x] `src/primer.ts` — session-start briefing (decision B).
- [x] `src/auto-recall.ts` — heuristic trigger (decision D).
- [x] Tests: utils, config, settings, schema, store, bridge, indexer,
      retrieval (RRF), primer, auto-recall — 44 tests, all green.

**Exit criteria:**
- [x] Open Pi in a project where pi-lcm has prior content; first session triggers
      a backfill sweep that embeds all existing messages and summaries (covered
      by indexer test "sweep tick embeds un-indexed messages and summaries").
- [x] Subsequent messages auto-index (covered by indexer test "handleMessage
      path embeds via the inflight chain").
- [x] `lcm_recall("…")` returns relevant top-K with snippet + score on a
      synthetic corpus (covered by retrieval tests).
- [x] All tests green; no patches to pi-lcm.

**Deferred to Phase 4 (operator polish):**
- Live model smoke test exercising Transformers.js download + inference end
  to end. Skipped from CI to avoid ~30MB downloads per run; manual smoke
  via `pi -e ./index.ts` in a project with pi-lcm history.

---

## Phase 2 — Session-start primer (decision B) — merged into Phase 1

Folded into Phase 1 since the user picked A+B+D in Q8. See `src/primer.ts`
and `test/primer.test.ts`.

## Phase 3 — Heuristic auto-recall (decision D) — merged into Phase 1

Folded into Phase 1. See `src/auto-recall.ts` and `test/auto-recall.test.ts`.

---

## Phase 4 — Operator polish

**Goal:** Make pi-lcm-memory pleasant to live with.

- [x] `/memory reindex` (clear + kick), `/memory model`, `/memory clear --yes`,
      `/memory status` (with backoff/idleStreak).
- [x] `ctx.ui.setStatus(...)` footer with index/sweep/download state.
- [x] README "Quick start" mirrored from pi-lcm's style; CHANGELOG updated.
- [x] Notify on first-time model download with size (Transformers.js progress
      callback wired through embedder listener).
- [x] Project-vs-global settings precedence + writeback (settings.ts; panel
      P toggles).
- [x] Performance tightening: batched embedder calls (one inference call per
      batch of 32); adaptive sweep interval (idle ticks back off ×2, capped at
      5 min; `kick()` resets immediately on new work / commands / compact).
- [x] Structured event log to `memory_meta` (rolling window of 200) +
      `/memory events` view.
- [x] Lazy capture of pi-lcm's `conversation_id` after `message_end` so
      `sessionFilter` works without guessing.
- [x] Tests: diagnostics, batched indexer, adaptive backoff via `kick()`,
      `latestConversationId`, all `/memory` subcommands.

**Exit criteria:** A user can change embedding model with one command and
have everything re-embedded in the background, with a footer status visible. ✅

---

## Phase 5 — Stabilization & stretches (any order, opt-in)

### Stabilization round (shipped)

- [x] **Worker-thread embedder.** `src/embeddings/worker.mjs` owns the
      pipeline in a `worker_threads` thread; main thread is never blocked
      by ONNX. Multi-core ORT (`intraOpNumThreads = cpus()-1`, capped at
      8). Zero-copy `ArrayBuffer` transfers. Live test asserts semantic
      ordering and ~1880 embeds/sec on an M-class laptop.
- [x] **Single-transaction batched inserts.** `store.insertBatch(items[])`
      with IMMEDIATE locking + reused prepared statements. 15× faster
      in isolation; eliminates lock-acquisition stacking under concurrent
      pi-lcm writes. New `whichHashesPresent` returns Map<hash, vec_rowid>
      via a single IN() query.
- [x] **Side-channel tracer (`PI_LCM_MEMORY_TRACE=1`).** Synchronous
      file-based event log that survives main-thread freezes; main and
      worker write to the same file. Documented set of events covering
      tick / batch / embed lifecycle.
- [x] **Schema v2: many-to-one id mapping.** `memory_index_msg` and
      `memory_index_sum` side tables. Closes the dedupe leak where two
      pi-lcm messages with identical content shared one embedding but
      only one id was recorded — the other leaked through every sweep.
      Migration backfills from existing `memory_index` rows.
- [x] **Bridge generator: rowid cursor + SQL filter.** Iteration is
      forward-only by `m.rowid` and skipped roles / empty content are
      filtered at the SQL level. Combined with the safety yield in
      `processBatched` every 1024 items, the infinite-loop class of
      bugs is structurally impossible. Two regression tests cover it.
- [x] **Settings panel API fix.** `ctx.ui.custom(factory, options)` shape
      — was passing an object literal, pi crashed at `factory is not a
      function`. Now constructed inside a factory closure with `done`
      wired to `onClose`.
- [x] **120s warmup watchdog + `/memory worker` command + per-MB
      download notifications + worker hello message** for visibility
      during model downloads.

---

## Phase 6 — Quality + measurement (next session)

Full plan in [NEXT.md](./NEXT.md). Order matters: cleanup, then
measurement infrastructure, then the cross-encoder, so we can prove
the quality lift with numbers.

- [x] **Housekeeping pass** — dropped `_testing` export from indexer,
      removed `iter_chunk` trace, folded `log` alias into `events`,
      un-exported orphans (`getOpenDb`/`getOpenCwd`/`getDbPath`/
      `REGISTRY`). Tracer doc in README already shipped earlier.
- [x] **Performance benchmarks** (`bench/perf.ts`) — embed throughput,
      hook/sweep latencies (p50/p99), recall latency, DB size growth,
      cold worker warmup. Outputs JSON + markdown summary. Run via
      `npm run bench:perf` (`PI_LCM_MEMORY_BENCH_QUICK=1` for smoke).
- [x] **Recall quality benchmarks** (`bench/quality.ts`) — MRR,
      nDCG@10, recall@5/@10, precision@5 over an eval set. Synthetic
      eval generated from `BENCH_TOPICS` if no `bench/eval/eval.json`
      is provided. Run via `npm run bench:quality`.
- [x] **End-to-end test harness** (`test/e2e/`) — `makeFakePi()`
      faithful `ExtensionAPI` stub + tmp project + pre-seeded
      pi-lcm DB. Tests run real worker, real DB, real ONNX. Opt-in
      via `PI_LCM_MEMORY_LIVE_TEST=1`. Coverage: backfill, lcm_recall,
      lcm_similar, /memory commands, settings panel factory contract,
      message_end hook indexing. 7 tests passing in ~500 ms.
- [x] **Capture baseline** — `bench/results/{perf,quality}.<sha>.json`
      committed; reranker delta is a diff, not a vibe.
- [x] **Cross-encoder reranker** (`Xenova/ms-marco-MiniLM-L-6-v2`)
      shipped. Second pipeline in the existing worker, opt-in via
      `rerank: boolean` config (default off, lazy-loaded). Per-call
      override via `params.rerank`. Falls through to hybrid order on
      any reranker error. New `/memory rerank on|off` command and
      two new settings-panel rows.
- [x] **Re-run benchmarks** — synthetic eval shows MRR 0.364→1.000,
      Recall@10 0.480→0.993, nDCG@10 0.391→0.996. Live throughput
      ~2000 (query, doc) pairs/sec on M5 × 8 threads, q8. Shipped.

---

## Future research (not scheduled)

Kept for context; pull into a numbered phase only when actively useful.

- [ ] **Code / file content indexing** (separate `code_vec` table;
      AST-aware chunking). User has flagged this as **interesting for
      future discussion**. Design notes in NEXT.md § "Future research".
- [ ] **Memory cards** (manually saved snippets via `/memory save` and a
      tool). Mirrors a prior local attempt.
- [ ] **Cross-project / workspace-wide recall** (multi-DB query layer).
- [ ] **MCP wrapper** exposing recall to Claude Code (Q3 deferred path).
- [ ] **Eviction / retention policy** (e.g., archive after N days, keep
      summaries). Matters once a heavy user accumulates 100k+ rows.
- [ ] **Edit / redact memory** (with audit log).
- [ ] **`PI_LCM_MEMORY_TRACE` toggleable from `/memory trace on|off`** so
      no relaunch is needed to enable the tracer.

---

## Tracking

- Each merged change updates `CHANGELOG.md`.
- This roadmap is the source of truth for ordering; PLAN.md for *how*.
- Issues / TODOs that pop up mid-phase are added at the **bottom** of the
  current phase or under "Stretches", never silently inflated.
