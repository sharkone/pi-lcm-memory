# pi-lcm-memory — PLAN

A Pi extension that adds persistent, cross-session **semantic memory** on top of
[pi-lcm](https://github.com/codexstar69/pi-lcm). Goal: perfect recall of anything
said in any session within a project, retrievable via hybrid (FTS5 + dense
vector) search. Fully local. No external APIs.

## Goal & non-goals

**Goal.** When the agent (Pi) opens a session in a project we've seen before:
1. It is briefed on prior memory at session start.
2. It can recall any past message or summary by lexical *or* semantic query.
3. Heuristic phrasing in user prompts ("remember…", "earlier we…") triggers
   automatic recall injection for that turn.
4. Storage is shared with pi-lcm's per-project SQLite. Lossless guarantees of
   pi-lcm are preserved.

**Non-goals (v1).**
- Per-turn unconditional auto-recall (token-expensive; deferrable).
- Cross-project / federated memory.
- Cross-encoder re-rankers.
- User-curated "memory cards" (manually saved snippets). *Note: this concept
  appeared in a prior local attempt at this project, but was not selected in
  the interview. Will revisit in a later phase if useful.*
- A rich settings UI panel (start with text-based `/memory` commands).
- Editing or redacting prior memory (read-only history at v1).

## Interview-locked decisions

| # | Decision | Rationale |
|---|---|---|
| 1 | **Hybrid retrieval** (FTS5 + dense vectors) | Strongest recall; literal + paraphrase. |
| 2 | **Shared DB with pi-lcm** | Lossless extension; zero duplication; atomic joins. |
| 3 | **Pi extension only** (no MCP) | Same surface area as pi-lcm itself. |
| 4 | **Index filtered messages + DAG summaries** (skip tool I/O) | Best signal-to-noise; FTS5 still covers raw tool output literally. |
| 5 | **Configurable embedding model, default `Xenova/bge-small-en-v1.5`** | Strong quality at small footprint; pluggable. |
| 6 | **Additive new tools: `lcm_recall`, `lcm_similar`** | No patching of pi-lcm. Clear intent split with `lcm_grep`. |
| 7 | **Hook + background sweep** | Real-time freshness with eventual completeness. |
| 8 | **A + B + D**: pull-tool + session-start primer + heuristic auto-recall | Multi-modal access, no per-turn token tax. |

## Architecture

```
┌───────────────────────────── Pi Session ─────────────────────────────┐
│                                                                       │
│   pi-lcm                       pi-lcm-memory (this extension)         │
│   ─────────────────────        ──────────────────────────────         │
│   message_end ─► persist ─►┐                                          │
│                            ▼                                          │
│                       SQLite (shared, per-project)                    │
│                            ▲                                          │
│                            │   ┌── embed(msg) ──► memory_vec          │
│   session_before_compact ──┘   │   (sqlite-vec virtual table)         │
│       └─► D0/D1/D2 summaries ──┘                                      │
│                                                                       │
│   context ─────────────────────► primer + (optional) auto-recall      │
│   session_start ──► open DB, lazy-load embedder, register sweep       │
│   registerTool ──► lcm_recall, lcm_similar                            │
│   registerCommand ──► /memory                                         │
└───────────────────────────────────────────────────────────────────────┘
```

Both extensions are Pi-loaded peers. They subscribe to the same Pi events
independently. pi-lcm owns the canonical `messages` and `summaries` tables;
pi-lcm-memory adds a `memory_vec` virtual table and a `memory_index` join
table in the same DB file.

## Storage layout (additive schema)

```sql
-- Vector store via sqlite-vec virtual table.
-- Dim is parametrized by the configured model; schema is recreated
-- on dim mismatch (vectors re-embedded from the underlying text).
CREATE VIRTUAL TABLE IF NOT EXISTS memory_vec USING vec0(
  embedding float[384]
);

-- Join table: links each vec rowid to its source content + lineage.
CREATE TABLE IF NOT EXISTS memory_index (
  vec_rowid       INTEGER PRIMARY KEY,
  source_kind     TEXT NOT NULL,              -- 'message' | 'summary'
  content_hash    TEXT NOT NULL UNIQUE,       -- SHA-256 of canonical content
  pi_lcm_msg_id   INTEGER,                    -- nullable FK to pi-lcm messages
  pi_lcm_sum_id   INTEGER,                    -- nullable FK to pi-lcm summaries
  conversation_id TEXT,
  session_started INTEGER,                    -- unix sec
  role            TEXT,                       -- 'user' | 'assistant' | summary depth
  snippet         TEXT NOT NULL,              -- ~240-char preview for ranking output
  text_full       TEXT NOT NULL,              -- de-normalized for self-contained recall
  token_count     INTEGER,
  model_name      TEXT NOT NULL,
  model_dims      INTEGER NOT NULL,
  created_at      INTEGER NOT NULL DEFAULT (unixepoch())
);

CREATE INDEX IF NOT EXISTS idx_memidx_kind     ON memory_index(source_kind);
CREATE INDEX IF NOT EXISTS idx_memidx_msg      ON memory_index(pi_lcm_msg_id);
CREATE INDEX IF NOT EXISTS idx_memidx_sum      ON memory_index(pi_lcm_sum_id);
CREATE INDEX IF NOT EXISTS idx_memidx_session  ON memory_index(conversation_id, session_started);

-- Bookkeeping for the sweep worker.
CREATE TABLE IF NOT EXISTS memory_meta (
  k TEXT PRIMARY KEY,
  v TEXT NOT NULL
);
```

Notes:
- `text_full` is denormalized so recall results are self-contained; pi-lcm row
  references are best-effort but not required for output.
- `content_hash` provides idempotency (`INSERT OR IGNORE`).
- Embedding dim is recorded; model swap → re-embed (handled by sweep).

## Components / file layout

```
pi-lcm-memory/
  index.ts                  # extension entry: hooks + registerTool/Command
  package.json
  tsconfig.json
  README.md
  PLAN.md  (this file)
  ROADMAP.md
  CHANGELOG.md
  LICENSE
  src/
    config.ts               # resolveConfig(): read pi-lcm + our settings
    db/
      connection.ts         # open shared DB (mirror pi-lcm's hashCwd path)
      schema.ts             # additive migrations + sqlite-vec setup
      vec.ts                # sqlite-vec extension load helper
      store.ts              # data access for memory_index / memory_vec
    bridge.ts               # read pi-lcm tables (messages, summaries) safely
    embeddings/
      embedder.ts           # Transformers.js pipeline; lazy load; cache
      model-registry.ts     # name → { repo, dims, defaults }
    indexer.ts              # ingestion: hook path + sweep worker
    retrieval.ts            # hybrid search: FTS5 ∪ vec → RRF merge
    primer.ts               # session-start briefing renderer
    auto-recall.ts          # heuristic phrase trigger + injector
    status.ts               # ctx.ui.setStatus footer
    commands.ts             # /memory dispatcher
    tools/
      lcm-recall.ts
      lcm-similar.ts
  test/
    schema.test.ts
    embedder.test.ts
    indexer.test.ts
    retrieval.test.ts
    primer.test.ts
    auto-recall.test.ts
    bridge.test.ts
```

## Lifecycle (which Pi events do what)

| Pi event | pi-lcm-memory action |
|---|---|
| `session_start` | Resolve config; open shared DB; ensure schema; lazy-init embedder (background); render `primer`; start sweep timer. |
| `message_end` | Filter (skip tool I/O); embed text in background; `INSERT OR IGNORE` into `memory_vec` + `memory_index` keyed by `content_hash`. |
| `session_before_compact` | After pi-lcm runs compaction, sweep newly-created `summaries` rows into the index (or rely on tick sweep). |
| `context` | First turn: inject `primer`. Every turn: if heuristic match on user prompt, inject `auto-recall` block. |
| `session_shutdown` | Stop sweep; flush in-flight embeds; WAL checkpoint. |

## Embedder

- **Library:** `@huggingface/transformers` v3 (Node-native; CommonJS + ESM).
- **Default model:** `Xenova/bge-small-en-v1.5` (384-dim).
- **Quantization:** prefer INT8 / quantized weights when the model ships them.
- **Cache:** model weights cached in `~/.cache/pi-lcm-memory/models/`
  (override via `PI_LCM_MEMORY_CACHE_DIR`).
- **Pluggable:** model resolved by name; registry maps to repo + dims; setting
  `embeddingModel` swaps it. On dim change, sweep re-embeds.

## Hybrid retrieval

```
top_K_lex   = FTS5 query against pi-lcm's messages_fts (read-only)
top_K_sem   = sqlite-vec MATCH on memory_vec
final       = RRF merge (k=60) → best-K
output rows = JOIN memory_index by vec_rowid for snippets/lineage
```

RRF score: `Σ_i 1 / (k + rank_i(d))` over each retriever's ranked list.

Mode flag on `lcm_recall` lets the agent pick `hybrid` (default) / `lexical` /
`semantic` explicitly when intent is clear.

## Tools

```
lcm_recall(
  query: string,
  k?: number = 10,
  mode?: "hybrid" | "lexical" | "semantic" = "hybrid",
  sessionFilter?: string,
  after?: string,   // ISO 8601
  before?: string,  // ISO 8601
) -> Array<{
  source_kind: "message" | "summary",
  conversation_id: string,
  session_started: string,
  role?: string,
  snippet: string,
  score: number,
  pi_lcm_msg_id?: number,
  pi_lcm_sum_id?: number
}>

lcm_similar(
  messageId: string | number,    // pi_lcm_msg_id OR memory_index.vec_rowid
  k?: number = 5
) -> same shape as lcm_recall
```

Both surface `pi_lcm_*` ids so the agent can chain into `lcm_expand` for full
detail recovery.

## Session-start primer (B)

```
## Project memory
N prior sessions; last on YYYY-MM-DD.

Recent topics
- {top-K most recent summary D≥1 nodes, snippet ≤120 chars}

Tools
- `lcm_grep(pattern)`     — exact strings/regex
- `lcm_recall(query)`     — semantic / hybrid recall
- `lcm_expand(summaryId)` — recover originals from a compressed node
```

Token budget: ≤300 tokens. Renders empty (no primer) on first session in
project.

## Heuristic auto-recall (D)

- Listener on `context` event examines the latest user message.
- Phrase set (configurable):
  ```
  /\b(remember|recall|earlier|previously|before|like last time|the (same|previous|prior) (one|approach|setup)|we (had|have) (discussed|talked|mentioned))\b/i
  ```
- On match: run `lcm_recall(prompt, k=5)`; format top hits into a system block
  for *that turn only*; never persisted.
- Token cap: ~600 tokens.
- Setting: `autoRecall: "off" | "heuristic" | "always"`. v1 default: `heuristic`.

## Slash commands

```
/memory stats            # counts, dim, model, sweep state, db size
/memory reindex [scope]  # force re-embed everything (or kind=summary|message)
/memory search <query>   # ad-hoc CLI version of lcm_recall
/memory model <name>     # change embedding model (triggers reindex)
/memory clear            # destructive; prompts confirm; re-embeds from pi-lcm
/memory status           # current sweep cycle / queue depth / last error
/memory settings         # open settings panel
/memory-settings         # standalone settings panel command
```

## Settings panel

Mirrors pi-lcm's `LcmSettingsPanel` shape (registered via `pi.registerSettingsPanel`).
Provides interactive editing of the configuration fields below:

- Enable / disable.
- Embedding model selector (changing prompts a reindex).
- Index toggles (messages, summaries, skip tool I/O).
- Primer on/off + topK.
- Auto-recall mode + topK + token budget.
- Sweep interval.
- RRF k.
- Scope toggle (project vs global).

Writes back to project (`<cwd>/.pi-lcm-memory.json`) or global
(`~/.config/pi-lcm-memory/settings.json`) depending on selected scope.

## Configuration

Resolved by precedence: project `.pi-lcm-memory.json` > global > defaults.

```jsonc
{
  "pi-lcm-memory": {
    "enabled": true,
    "embeddingModel": "Xenova/bge-small-en-v1.5",
    "embeddingQuantize": "auto",         // "auto" | "fp32" | "int8"
    "indexMessages": true,
    "indexSummaries": true,
    "skipToolIO": true,
    "primer": true,
    "primerTopK": 5,
    "autoRecall": "heuristic",           // "off" | "heuristic" | "always"
    "autoRecallTopK": 5,
    "autoRecallTokenBudget": 600,
    "recallDefaultTopK": 10,
    "rrfK": 60,
    "sweepIntervalMs": 30000,
    "modelCacheDir": null                // null → ~/.cache/pi-lcm-memory/models
  }
}
```

`dbDir` is **not** ours to set. We mirror pi-lcm's `config.dbDir` + `hashCwd(cwd)`
to open the same DB file. If pi-lcm settings can't be resolved, we fall back to
the same defaults pi-lcm uses (TBD at impl).

## Concurrency, safety, idempotency

- WAL is already enabled by pi-lcm's `connection.ts` — we share a single
  connection (per-process) opened by whichever extension loads first; the second
  extension obtains the same handle by following the path convention.
- `INSERT OR IGNORE` on `content_hash` makes ingestion safe under retries.
- Sweep is the safety net; the hook path is "best effort, fast".
- Embedding inference runs off-thread (microtask queue); never blocks
  `message_end` return.
- Model load is lazy: first embedding triggers download/load, with a warm-up
  task during `session_start`.
- We do **not** patch pi-lcm tables. Read-only access. Writes only to our
  additive tables.

## Testing strategy

- **Vitest** (matches pi-lcm).
- Unit:
  - `schema`: migrations idempotent on empty DB and on a pi-lcm DB.
  - `embedder`: dim correctness + cache reuse.
  - `indexer`: message_end → row in `memory_vec`; sweep catches misses.
  - `retrieval`: known-answer fixtures with synthetic content.
  - `primer`: empty / non-empty / token-budget enforcement.
  - `auto-recall`: phrase positives + negatives; budget enforcement.
- Integration (deferred to later phase):
  - Stub Pi runtime → walk a 3-session scenario; assert tool outputs.
- Performance:
  - Bench: embedding throughput (≥5 msgs/sec on M-class laptop).
  - Bench: hybrid retrieval p95 < 50ms on a 50k-row corpus.

## Open questions / impl-time TBDs

1. Resolving pi-lcm's `dbDir`: read its settings file, or ship a redundant
   default? (Ideally both: prefer pi-lcm settings if present.) — **resolved**
   in Phase 1: env > project > global > defaults; we read pi-lcm's
   `lcm.dbDir`/`lcm.enabled` so we follow it.
2. Quantized model availability for `Xenova/bge-small-en-v1.5` in
   Transformers.js v3. — **resolved** in Phase 4 hotfix: q8 variant ships
   with all major Xenova feature-extraction models (suffix `_quantized.onnx`).
   Default changed from `auto` to `q8`.
3. Exact event signature for `message_end` in current `@mariozechner/pi-coding-agent`
   — confirm `event.message` shape and pruning of system/tool entries.
4. Sweep driver: `setInterval` vs `unref()`d timer. — **resolved** in Phase 4:
   `setTimeout` chain with `.unref()`; adaptive backoff.
5. WAL checkpoint cadence: pi-lcm checkpoints on close; we should not race.
6. **TUI freeze during ONNX inference** (Phase 4 hotfix mitigated; Phase 5
   target). ONNX runs synchronously on the main thread, blocking the event
   loop. Mitigations shipped: q8 weights (×2-4 faster), batch size 8 (was
   32), `setImmediate` yield between batches. Real fix = worker_threads
   offloading.

## Out of scope (kept for ROADMAP)

- Multi-project / federated memory.
- "Memory cards" (user-saved snippets) — was in prior local attempt; reconsider
  in phase 4+.
- Cross-encoder re-ranker (e.g., `Xenova/ms-marco-MiniLM-L-6-v2`).
- Editing / redacting / forgetting memory.
- Code/file content indexing (separate retrieval problem).
