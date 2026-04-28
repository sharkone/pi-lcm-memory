# pi-lcm-memory

> Persistent, cross-session **semantic memory** for [Pi](https://github.com/mariozechner/pi-coding-agent).
> Additive on top of [pi-lcm](https://github.com/codexstar69/pi-lcm) — perfect
> recall across every session in a project, retrievable by hybrid (FTS5 +
> vector) search. Fully local; no external APIs.

## Status

Phases 1–5 shipped — stable in interactive use. Phase 6 (benchmarking
infrastructure + cross-encoder reranker) is queued; see
[NEXT.md](./NEXT.md) for the next-session plan,
[ROADMAP.md](./ROADMAP.md) for the long-term view, and
[PLAN.md](./PLAN.md) for design rationale.

## Debugging

Set `PI_LCM_MEMORY_TRACE=1` before launching pi to write a side-channel
trace log to `/tmp/pi-lcm-memory.<pid>.trace.log` (or pass an explicit
path: `PI_LCM_MEMORY_TRACE=/path/to/log`). Both the main thread and
the embedder worker append to the same file with `pid`/`src` markers,
so it survives main-thread freezes — useful when the TUI hangs and
the in-DB diagnostics ring can't be written.

`/memory worker` dumps current embedder + worker state for live
diagnostics.

## Quick start

```sh
# In your Pi project
pi install npm:pi-lcm           # if not already
pi install npm:pi-lcm-memory    # this extension

# That's it. Open Pi normally:
pi
```

First session in a project with prior pi-lcm history will:

1. Download the embedding model (`Xenova/bge-small-en-v1.5`, ~33 MB) into
   `~/.cache/pi-lcm-memory/models/`. Pi shows a one-time notice with size.
2. Backfill embeddings for every existing message + DAG summary in batches of 32.
3. Render a session-start primer summarizing prior sessions.
4. From now on, every new message is embedded in the background.

## What it adds on top of pi-lcm

| | pi-lcm | pi-lcm-memory |
|---|---|---|
| Per-message storage | ✅ SQLite | shared (no duplication) |
| FTS5 lexical search | ✅ `lcm_grep` | reused |
| DAG summaries (D0/D1/D2…) | ✅ | reused |
| Cross-session within a project | ✅ | reused |
| **Dense vector index** | ❌ | ✅ `sqlite-vec` virtual table |
| **Hybrid (lexical+semantic) retrieval** | ❌ | ✅ `lcm_recall` |
| **"More like this"** | ❌ | ✅ `lcm_similar` |
| **Session-start memory primer** | ❌ | ✅ |
| **Heuristic auto-recall** ("remember…", "earlier…") | ❌ | ✅ |
| Settings panel | ✅ | ✅ (mirrors pi-lcm UX) |

## Tools the agent gets

- `lcm_recall(query, k?, mode?)` — hybrid recall, with `mode` ∈ `hybrid` (default) / `lexical` / `semantic`. Optional `sessionFilter`, `after`, `before`.
- `lcm_similar(messageId, k?)` — find messages semantically close to a known one.

Plus pi-lcm's own `lcm_grep`, `lcm_describe`, `lcm_expand` — which we *recommend* for exact strings and DAG drilling.

## Slash commands

```
/memory stats               counts, model, dim, db size
/memory status              sweep cycles, busy, last error, current interval
/memory search <query>      ad-hoc lcm_recall
/memory reindex             wipe & re-embed everything
/memory clear [--yes]       drop all embeddings (sweep will rebuild)
/memory model <name>        change embedding model (triggers reindex)
/memory events              last 20 diagnostic events
/memory settings            open settings panel  (also: /memory-settings)
```

## Settings

Persisted under the `lcm-memory` key in the same files pi-lcm uses
(`~/.pi/agent/settings.json` global; `<cwd>/.pi/settings.json` project).
Project values override global. Resolution: env > project > global > defaults.

| Key | Default | Meaning |
|---|---|---|
| `enabled` | `true` | Master enable. Auto-disables if `lcm.enabled` is false. |
| `embeddingModel` | `Xenova/bge-small-en-v1.5` | Any Transformers.js feature-extraction model. |
| `embeddingQuantize` | `auto` | `auto` / `fp32` / `int8`. |
| `indexMessages` | `true` | Embed user/assistant text. |
| `indexSummaries` | `true` | Embed pi-lcm DAG summaries. |
| `skipToolIO` | `true` | Don't embed tool outputs / bash results (FTS5 still covers them). |
| `primer` | `true` | Render a session-start briefing. |
| `primerTopK` | `5` | Recent topics in primer. |
| `autoRecall` | `heuristic` | `off` / `heuristic` / `always`. |
| `autoRecallTopK` | `5` | Hits to inject when auto-recall fires. |
| `autoRecallTokenBudget` | `600` | Hard cap on injected recall block. |
| `recallDefaultTopK` | `10` | Default `k` for `lcm_recall`. |
| `rrfK` | `60` | Reciprocal Rank Fusion constant. |
| `sweepIntervalMs` | `30000` | Base sweep period. Idle ticks back off (×2, max 5min). |
| `modelCacheDir` | `null` | Override model cache directory. |
| `debugMode` | `false` | Verbose notifications. |

Env overrides: `PI_LCM_MEMORY_DB_DIR`, `PI_LCM_MEMORY_MODEL`,
`PI_LCM_MEMORY_QUANTIZE`, `PI_LCM_MEMORY_SWEEP_MS`, `PI_LCM_MEMORY_DEBUG`,
`PI_LCM_MEMORY_ENABLED`. Falls back to pi-lcm's `LCM_DB_DIR` so the same
project DB is shared.

## How it works

- pi-lcm writes every message and DAG summary to a per-project SQLite at
  `~/.pi/agent/lcm/<sha256(cwd)[..16]>.db`.
- We open the same file. Our additive tables: `memory_vec` (sqlite-vec virtual
  table, dim parametric to model), `memory_index` (join + denormalized text),
  `memory_meta` (kv bookkeeping + last 200 diagnostic events).
- **Embedding runs in a worker thread.** All ONNX inference happens in
  `src/embeddings/worker.mjs` via `worker_threads`, so the Pi event loop is
  never blocked. ORT is configured with `intraOpNumThreads = cpus()-1`
  (capped at 8), saturating cores during backfill.
- Two ingestion paths run concurrently:
  - **Hook path**: `message_end` → embed (worker) → `INSERT OR IGNORE`.
  - **Sweep path**: every 30 s (adaptive), scan for un-indexed pi-lcm rows and
    process them in batches of 32 with single worker inference calls.
- `lcm_recall(query)` runs FTS5 + vector kNN and merges them with Reciprocal
  Rank Fusion (`k=60`).
- A session-start primer renders prior session count, last date, and the most
  recent `D≥1` summaries. ≤300 tokens.
- A heuristic listener on each turn matches a regex over the user prompt
  (`/remember|earlier|previously|like last time|.../i`); when it fires, a
  `## Recall` block is injected into the current turn's system context.

## Footprint

- **Disk**: model weights ~30–80 MB depending on model. SQLite grows roughly
  ~2 KB per indexed message at default dim (384). 100k messages ≈ 80 MB.
- **Memory**: embedder worker is lazy-spawned on first use. Idle = ~0.
- **CPU**: backfill embeds **~1500–2000 messages/sec** on Apple Silicon
  (8-thread q8). All work happens in a worker thread; the Pi TUI is never
  blocked.

## Local dev

```sh
git clone git@github.com:sharkone/pi-lcm-memory.git
cd pi-lcm-memory
npm install
npm test         # 60 vitest tests, ~500ms
npm run typecheck
pi -e ./index.ts # load the local extension into Pi
```

## License

MIT.
