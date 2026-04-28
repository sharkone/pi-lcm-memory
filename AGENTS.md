# AGENTS.md — pi-lcm-memory

> Orientation guide for AI agents working in this repo.

---

## What this is

`pi-lcm-memory` is a **Pi extension** that adds persistent, cross-session
semantic memory on top of [pi-lcm](https://github.com/codexstar69/pi-lcm).
It stores dense vector embeddings (via `sqlite-vec`) in the same SQLite DB
that pi-lcm uses, and exposes **hybrid (FTS5 + vector) recall** through two
new agent tools: `lcm_recall` and `lcm_similar`.

It ships as a pure Pi extension (no MCP server). The entry point is `index.ts`
at the repo root; everything else lives under `src/`.

---

## Quick orientation

```
index.ts              Extension entry point — wires all modules to Pi events
src/
  auto-recall.ts      Heuristic trigger: fires lcm_recall on "remember…" etc.
  bridge.ts           Read-only view of pi-lcm's messages + summaries tables
  commands.ts         /memory dispatcher (stats, search, model, reindex…)
  config.ts           Merge env → project → global → defaults into MemoryConfig
  diagnostics.ts      Internal event ring buffer; surfaced by /memory events
  indexer.ts          message_end hook + background sweep loop
  primer.ts           Session-start briefing (decision B)
  retrieval.ts        Hybrid retrieval: FTS5 + vec KNN → RRF merge
  settings.ts         Load/save settings under the lcm-memory key
  settings-panel.ts   Interactive TUI panel (mirrors pi-lcm UX)
  status.ts           Sweep status model
  trace.ts            Optional debug trace to file
  utils.ts            Shared helpers
  db/
    connection.ts     Open shared DB, mirroring pi-lcm's hashCwd path scheme
    schema.ts         Additive migrations: memory_vec, memory_index, memory_meta
    store.ts          MemoryStore CRUD over those tables
    vec.ts            Load sqlite-vec extension; soft-fail with clear message
  embeddings/
    embedder.ts       Transformers.js pipeline, lazy-loaded in worker
    model-registry.ts Known embedding models + their dimensions
    worker.mjs        Worker thread that runs the embedding pipeline
  tools/
    lcm-recall.ts     lcm_recall tool definition
    lcm-similar.ts    lcm_similar tool definition
test/                 Vitest test suite (~60 tests, all in-process)
bench/                perf.ts + quality.ts benchmarks (run manually)
```

---

## Commands

```sh
npm test              # run all vitest tests (~500 ms)
npm run test:watch    # vitest in watch mode
npm run typecheck     # tsc --noEmit — always run before committing
npm run bench:perf    # embedding throughput benchmark
npm run bench:quality # retrieval quality benchmark
npm run bench         # both benchmarks
```

`prepublishOnly` runs `typecheck` then `test` — both must pass before
releasing.

---

## Architecture invariants (do not violate)

1. **Additive schema only.** pi-lcm-memory never modifies pi-lcm's `messages`
   or `summaries` tables. It only adds `memory_vec` (sqlite-vec virtual table),
   `memory_index` (join table), and `memory_meta`. Migrations must be
   idempotent (`CREATE TABLE IF NOT EXISTS`).

2. **No patching pi-lcm.** Both extensions are independent Pi peers. They
   subscribe to the same Pi events independently. pi-lcm-memory reads pi-lcm's
   tables via `src/bridge.ts` (read-only).

3. **sqlite-vec must soft-fail.** `src/db/vec.ts` catches load failures and
   disables vector indexing gracefully. Never let a missing native extension
   crash the Pi session.

4. **Embedder is lazy and worker-threaded.** The Transformers.js pipeline is
   spawned in a worker thread (`src/embeddings/worker.mjs`) on first use.
   The Pi TUI thread must never be blocked by embedding work.

5. **Tool I/O is not embedded by default.** `skipToolIO: true` keeps
   signal-to-noise high. FTS5 (pi-lcm's `lcm_grep`) still covers raw tool
   output lexically. Don't change this default without a strong reason.

6. **DB path mirrors pi-lcm.** `src/db/connection.ts` replicates pi-lcm's
   `hashCwd(cwd)` path scheme so both extensions open the same file.
   `PI_LCM_MEMORY_DB_DIR` (and the pi-lcm `LCM_DB_DIR` fallback) must be
   respected.

---

## Design decisions (locked)

These were set during the initial design interview and must not be reversed
without re-opening the design:

| # | Decision |
|---|---|
| 1 | Hybrid retrieval: FTS5 + dense vectors → RRF |
| 2 | Shared SQLite DB with pi-lcm (no duplication) |
| 3 | Pi extension only — no MCP server |
| 4 | Index filtered messages + DAG summaries; skip tool I/O |
| 5 | Configurable embedding model; default `Xenova/bge-small-en-v1.5` |
| 6 | New tools `lcm_recall` + `lcm_similar`; no patching existing tools |
| 7 | Hook + background sweep for real-time freshness |
| 8 | Pull-tool (A) + session-start primer (B) + heuristic auto-recall (D) |

---

## Settings

Stored under the `lcm-memory` key in pi-lcm's settings files:
- Global: `~/.pi/agent/settings.json`
- Project: `<cwd>/.pi/settings.json` (overrides global)

Resolution order: **env → project → global → defaults**

Key env overrides: `PI_LCM_MEMORY_ENABLED`, `PI_LCM_MEMORY_DB_DIR`,
`PI_LCM_MEMORY_MODEL`, `PI_LCM_MEMORY_QUANTIZE`, `PI_LCM_MEMORY_SWEEP_MS`,
`PI_LCM_MEMORY_DEBUG`.

---

## Testing notes

- Tests live in `test/`. The e2e test is at `test/e2e/full-pipeline.test.ts`.
- **Live model tests** (`test/worker.live.test.ts`) download ~30 MB of model
  weights and are skipped in normal CI runs. Run manually when changing the
  embedder or model registry.
- All other tests are fully in-process with no network I/O.
- The `test/bench.metrics.test.ts` and `test/insert.bench.test.ts` files are
  benchmark harnesses, not correctness tests — they may be slow.

---

## Gotchas

- **Dimension mismatch on model change.** Switching `embeddingModel` triggers
  a reindex because `memory_vec` is typed to a fixed float dimension. The
  `/memory model <name>` command handles this automatically. If you add a new
  model to `src/embeddings/model-registry.ts`, make sure its dimension is
  correct or the sqlite-vec insert will throw.

- **`sqlite-vec` version pinned.** `better-sqlite3` and `sqlite-vec` must stay
  in sync with the native binaries in `node_modules`. Don't bump either
  without testing the full pipeline (`test/e2e/full-pipeline.test.ts` +
  `test/worker.live.test.ts`).

- **Sweep back-off.** The sweep interval starts at `sweepIntervalMs` (30 s)
  and doubles on idle ticks up to 5 min. Tests that assert on sweep timing
  should use the `Indexer` API directly rather than relying on wall-clock
  timing.

- **`session_before_compact` timing.** Summaries are written by pi-lcm during
  compaction. The indexer sweep picks them up on the next tick; they are not
  embedded synchronously at compaction time.
