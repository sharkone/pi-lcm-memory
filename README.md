# 🧠 pi-lcm-memory

> **Persistent, cross-session semantic memory for [Pi](https://github.com/mariozechner/pi-coding-agent).**  
> Never lose context. Every session remembered, every thought retrievable —  
> by meaning, not just keywords. Fully local. No external APIs.

Built as an additive layer on top of [pi-lcm](https://github.com/codexstar69/pi-lcm).

---

## ✨ What it does

When you open Pi in a project you've worked in before, pi-lcm-memory:

- 📋 **Briefs you** with a session-start primer of recent work
- 🔍 **Recalls** past messages and summaries via hybrid semantic + lexical search
- ⚡ **Auto-injects** relevant context when you say things like *"remember earlier…"*
- 🔄 **Indexes silently** in the background — no latency on your turns

All embeddings live in the same SQLite file pi-lcm already manages. No duplication, no sync, no external services.

---

## 🏗️ Architecture

```
┌──────────────────────────── Pi Session ────────────────────────────┐
│                                                                     │
│  ┌─────────────┐   message_end    ┌──────────────────────────────┐  │
│  │   pi-lcm    │ ──────────────►  │      pi-lcm-memory           │  │
│  │             │                  │                              │  │
│  │  messages   │ ◄── read-only ── │  Indexer (hook + sweep)      │  │
│  │  summaries  │                  │     │                        │  │
│  │  FTS5 index │                  │     ▼                        │  │
│  └─────────────┘                  │  Worker thread               │  │
│        │                          │  (ONNX / Transformers.js)    │  │
│        │  shared SQLite           │     │                        │  │
│        ▼                          │     ▼                        │  │
│  ┌─────────────────────────────────────────────────────────────┐  │  │
│  │  ~/.pi/agent/lcm/<hash>.db                                  │  │  │
│  │                                                             │  │  │
│  │  messages ──────────────────── memory_index (join)         │  │  │
│  │  summaries ─────────────────── memory_vec   (sqlite-vec)   │  │  │
│  │                                memory_meta  (kv + events)  │  │  │
│  └─────────────────────────────────────────────────────────────┘  │  │
│                                   │                               │  │
│  session_start ───────────────►   Primer + auto-recall            │  │
│  user turn ────────────────────►  Heuristic recall injection      │  │
│  lcm_recall / lcm_similar ──────► Retriever (FTS5 + vec → RRF)   │  │
│                                                                    │  │
└────────────────────────────────────────────────────────────────────┘
```

Both extensions are independent Pi peers — pi-lcm-memory never patches pi-lcm. It only adds three tables (`memory_vec`, `memory_index`, `memory_meta`) to the existing per-project SQLite.

---

## 🚀 Quick start

```sh
pi install npm:pi-lcm           # if not already installed
pi install npm:pi-lcm-memory

pi                              # open Pi as normal
```

**First session** in a project with existing pi-lcm history:

1. ⬇️ Downloads the embedding model (`Xenova/bge-small-en-v1.5`, ~33 MB, once per machine)
2. ⚙️ Backfills embeddings for all existing messages + summaries in batches of 32
3. 📋 Renders a session-start primer with recent topics
4. 🔄 From now on, every new message is embedded in the background

---

## 🆚 What it adds on top of pi-lcm

|  | pi-lcm | pi-lcm-memory |
|--|--------|---------------|
| Per-message storage | ✅ SQLite | shared (no duplication) |
| FTS5 lexical search | ✅ `lcm_grep` | reused |
| DAG summaries (D0/D1/D2…) | ✅ | reused |
| Cross-session recall within project | ✅ | reused |
| **Dense vector index** | ❌ | ✅ `sqlite-vec` virtual table |
| **Hybrid semantic + lexical retrieval** | ❌ | ✅ `lcm_recall` |
| **"More like this" navigation** | ❌ | ✅ `lcm_similar` |
| **Session-start memory primer** | ❌ | ✅ |
| **Heuristic auto-recall** | ❌ | ✅ |
| Settings panel | ✅ | ✅ (mirrors pi-lcm UX) |

---

## 🛠️ Agent tools

### `lcm_recall`
Hybrid (FTS5 + vector) search across all sessions in this project.

```
lcm_recall(query, k?, mode?, sessionFilter?, after?, before?)
```

| param | default | description |
|-------|---------|-------------|
| `query` | — | Natural-language or keyword query |
| `k` | `10` | Number of results |
| `mode` | `hybrid` | `hybrid` · `lexical` · `semantic` |
| `sessionFilter` | — | Restrict to a single conversation UUID |
| `after` / `before` | — | ISO 8601 date bounds |

### `lcm_similar`
Find messages semantically close to a known one — great for "show me more like this".

```
lcm_similar(messageId, k?)
```

> 💡 Use `lcm_grep` for exact strings, `lcm_recall` for concepts and paraphrases,
> `lcm_expand(summary_id)` to drill into any summary returned by recall.

---

## 💬 Slash commands

```
/memory stats               counts, model, dimensions, DB size
/memory status              sweep cycles, busy flag, last error, current interval
/memory search <query>      ad-hoc recall (same as lcm_recall)
/memory reindex             wipe all embeddings and re-embed everything
/memory clear [--yes]       drop embeddings (sweep will rebuild automatically)
/memory model <name>        switch embedding model (triggers reindex)
/memory events              last 20 diagnostic events
/memory worker              embedder + worker thread state (debug)
/memory settings            open interactive settings panel
```

---

## ⚙️ Settings

Stored under the `lcm-memory` key in pi-lcm's settings files.  
Resolution order: **env vars → project → global → defaults**.

| Key | Default | Description |
|-----|---------|-------------|
| `enabled` | `true` | Master switch. Auto-disables if pi-lcm is disabled. |
| `embeddingModel` | `Xenova/bge-small-en-v1.5` | Any Transformers.js feature-extraction model. |
| `embeddingQuantize` | `auto` | `auto` / `fp32` / `fp16` / `q8` / `int8` / `q4` |
| `indexMessages` | `true` | Embed user/assistant turns. |
| `indexSummaries` | `true` | Embed pi-lcm DAG summaries. |
| `skipToolIO` | `true` | Skip tool call/result content (FTS5 still covers these). |
| `primer` | `true` | Show session-start briefing. |
| `primerTopK` | `5` | Number of recent topics in the primer. |
| `autoRecall` | `heuristic` | `off` / `heuristic` / `always` |
| `autoRecallTopK` | `5` | Hits injected on auto-recall. |
| `autoRecallTokenBudget` | `600` | Hard token cap on injected recall block. |
| `recallDefaultTopK` | `10` | Default `k` for `lcm_recall`. |
| `rrfK` | `60` | Reciprocal Rank Fusion constant. |
| `sweepIntervalMs` | `30000` | Base sweep period (backs off ×2 up to 5 min on idle). |
| `modelCacheDir` | `null` | Override model weight cache directory. |
| `debugMode` | `false` | Verbose notifications. |

**Env overrides:** `PI_LCM_MEMORY_ENABLED`, `PI_LCM_MEMORY_DB_DIR`, `PI_LCM_MEMORY_MODEL`,
`PI_LCM_MEMORY_QUANTIZE`, `PI_LCM_MEMORY_SWEEP_MS`, `PI_LCM_MEMORY_DEBUG`

---

## ⚡ Performance

Measured on Apple Silicon (M-class), default model `Xenova/bge-small-en-v1.5` q8, 8 ORT threads:

| Metric | Value |
|--------|-------|
| Backfill throughput | ~1 500–2 000 messages/sec |
| Hook latency (p50) | ~3.4 ms |
| Sweep throughput | ~262 rows/sec |
| Recall latency | ~12 ms |
| Model download (once) | ~33 MB |
| DB growth per message | ~2 KB at 384 dims |
| 100k messages | ≈ 80 MB index |

All embedding work runs in a **dedicated worker thread** — the Pi TUI is never blocked. The main thread is idle between turns.

---

## 🔬 How it works

1. **Ingestion** — two concurrent paths keep the index fresh:
   - **Hook path**: `message_end` → embed in worker → `INSERT OR IGNORE`
   - **Sweep path**: every 30 s (adaptive backoff), scan for un-indexed pi-lcm rows, process in batches of 32

2. **Retrieval** — `lcm_recall(query)`:
   - Run FTS5 BM25 over `messages` + `summaries` → ranked list
   - Run sqlite-vec kNN over `memory_vec` → ranked list
   - Merge with **Reciprocal Rank Fusion** (RRF, k=60)

3. **Primer** — on `session_start`, render up to 5 recent D≥1 summaries as a `## Prior context` block (≤300 tokens)

4. **Auto-recall** — a regex listener on each user turn (`/remember|earlier|previously|like last time|.../i`) injects a `## Recall` block into the current turn's system context

5. **Worker thread** — `src/embeddings/worker.mjs` owns the Transformers.js pipeline. ORT is configured with `intraOpNumThreads = cpus()-1` (max 8), zero-copy `ArrayBuffer` transfers back to main thread

---

## 🐛 Debugging

Set `PI_LCM_MEMORY_TRACE=1` before launching Pi to write a side-channel trace log:

```sh
PI_LCM_MEMORY_TRACE=1 pi
# → /tmp/pi-lcm-memory.<pid>.trace.log

PI_LCM_MEMORY_TRACE=/path/to/log pi   # explicit path
```

Both the main thread and the embedder worker write to the same file with `pid`/`src` markers. The log is written with `fs.writeSync` so it survives main-thread freezes — it's the right tool when the TUI hangs and the in-DB diagnostics ring can't be written.

`/memory worker` prints live embedder + worker state without a restart.

---

## 🧑‍💻 Local dev

```sh
git clone git@github.com:sharkone/pi-lcm-memory.git
cd pi-lcm-memory
npm install

npm test              # 64 vitest tests, ~500 ms
npm run typecheck     # tsc --noEmit
npm run bench         # perf + quality benchmarks (needs a live pi-lcm DB)

pi -e ./index.ts      # load local extension into Pi
```

> ⚠️ `test/worker.live.test.ts` downloads ~33 MB of model weights.
> It is skipped by default — enable with `PI_LCM_MEMORY_LIVE_TEST=1`.

---

## 📄 License

MIT © [sharkone](https://github.com/sharkone)
