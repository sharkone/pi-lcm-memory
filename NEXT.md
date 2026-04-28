# Phase 6 — Cross-encoder reranker + cleanup

This file is a starting brief for the **next** session. Read it first; it
captures everything I'd want to know cold.

State at the end of the previous session:
- 64 tests green; 6 commits today; system is stable in interactive use.
- Last commits: `30fa020` (docs), `c24a78b` (schema v2 side tables),
  `86dfe58` (settings panel API fix), `080203f` (infinite-loop fix),
  `4054ed9` (tracer), `dbb1812` (insertBatch), `7e79240` (visibility +
  watchdog).
- See CHANGELOG.md "Phase 5 — stabilization round" for full detail.

## Goals (in order)

1. **Housekeeping pass** — remove dead code, gate noisy traces, tighten
   the surface. Should be ~30 minutes.
2. **Benchmark infrastructure** — perf (throughput, latency, memory) and
   recall quality (MRR, nDCG@k, recall@k) on a fixed eval set.
3. **End-to-end test harness** — full pipeline: real worker, real DB,
   real embedder, fake-but-faithful pi ExtensionAPI surface.
4. **Capture baseline numbers** by running (2) and (3) before any feature
   change. Commit the JSON output so deltas are reviewable.
5. **Cross-encoder reranker** (`Xenova/ms-marco-MiniLM-L-6-v2`) on top-N
   hybrid results, opt-in.
6. **Re-run benchmarks**, document the delta in CHANGELOG with concrete
   numbers. The reranker only ships if quality clearly goes up; no vibes.

Do (1) first so everything lands on a tidy codebase. Do (2) and (3) before
(5) so we have an objective measure of what changed.

---

## (1) Housekeeping pass

Concrete items found in the audit:

### Dead code

- **`src/indexer.ts:459`** — `export const _testing = { SWEEP_BATCH,
  SWEEP_BACKOFF_MIN_MS, SWEEP_BACKOFF_MAX_MS };` is not imported by any
  test or source file. Delete.

- **77 exports across src/** — most are legit (used cross-module), but
  worth a 5-minute manual sweep for any other lone-island exports.
  `npx tsx --eval` with a quick reflection script can list them; or
  use `npx ts-prune` if we want to add a dev dep.

### Trace volume

- **`iter_chunk` event in `src/indexer.ts:237`** is emitted every 64
  rows. Pre-fix it produced 2.18M lines / 173 MB on a single freeze.
  Post-fix the for-of loop terminates fast, but with tracing on for a
  large corpus it's still the dominant signal. Decision: emit once per
  4096 rows OR remove entirely. Recommend: remove and rely on
  `process_start` / `process_end` (which carry `scanned`, `batches`,
  total `ms`). The `safety_yield` event already covers the "we're
  iterating but not batching" case.

- All other trace events are bounded per tick / per batch — keep.

### Command surface

`/memory` subcommands (commands.ts):
- help, stats, status, search, reindex, clear, model, settings, events,
  log, worker.
- `events` and `log` are aliases — pick `events` as canonical, drop `log`
  from help text (keep dispatcher case for back-compat).
- `worker` was added during the freeze chase. It IS useful for diagnosing
  future embedder issues, but a power-user knob. Decision: keep, but
  hide it from `help` output unless `debugMode` is on. (Or: leave it
  visible — it's one line of help output.)

### Tracer ergonomics (Phase 5 stretch #8)

`PI_LCM_MEMORY_TRACE` requires a relaunch. Add `/memory trace on|off
[/path/to/log]` so we can flip it without restarting pi. Implementation:
expose `enableTrace(path?)` / `disableTrace()` in `src/trace.ts`; have
the worker re-read the env var or accept a control message. Maybe defer
this — the env-var path is still fine in practice.

### Documentation

- README "Quick start" section should mention the tracer for
  debugging — currently undocumented.
- ROADMAP.md should be updated as items in this phase are checked off.

---

## (2) Benchmark infrastructure

Two dimensions, two scripts. Both opt-in (env-gated), neither runs in
ordinary `npx vitest run`.

### Performance benchmarks (`bench/perf.ts`)

Measure end-to-end throughput and latency for the operations that
actually matter to a live session:

| Metric | What | How |
|--------|------|-----|
| `embed_throughput` | embeds/sec at batch=32, warm worker | run N batches of 32 over a fixed corpus, time wall-clock |
| `embed_latency_p50_p99` | per-call latency for batch=1 (hook path) and batch=32 (sweep path) | record each call, percentiles |
| `sweep_throughput` | rows/sec end-to-end (read pi-lcm → dedupe → embed → insert) | timed `indexer.tick()` over a seeded DB of 1000 / 5000 / 20000 messages |
| `recall_latency_p50_p99` | `lcm_recall` end-to-end | same query repeated, percentiles, with hot vs cold KNN cache |
| `db_size_bytes_per_row` | storage cost | DB size delta over inserted rows |
| `worker_warmup_ms` | first-call cost from cold cache | open a fresh model cache, time the load |

Output: `bench/results/perf.<git-sha>.json` with all numbers, machine
info (CPU model, cores, Node version), and a human-readable
`bench/results/perf.latest.md` summary.

Implementation: a stand-alone `tsx bench/perf.ts` script that imports our
internal modules directly — no pi runtime, no MCP. Skips downloading
the model if cached. Skipped from `vitest` (it's a script, not a test).
CI never runs it. Run manually before/after big changes.

### Recall quality benchmarks (`bench/quality.ts`)

The interesting question: "if I ask about X, does the right past
message actually rank #1?"

**Eval set construction**: easiest path is to **bootstrap from a real
conversation DB** (e.g. the user's own pi-lcm DB with their consent).

1. Pick K "target" messages from the DB — ones with distinctive
   content (e.g. user/assistant turns where a specific topic was
   introduced).
2. For each target, generate **N synthetic queries** that *should*
   retrieve it: paraphrases, related questions, references. Use an
   LLM (Claude / GPT) for this — we already have access via pi's
   SDK or a one-off API call.
3. Optionally hand-tag a few "hard negatives" (queries that look
   relevant but shouldn't return this target).
4. Resulting eval set: `{ query: string, relevant_msg_ids: string[] }[]`.
   Stored in `bench/eval/eval.json` (gitignored if the user's data is
   sensitive; commit only synthetic ones).

**Metrics** (on top-K hybrid results):
- **MRR** (Mean Reciprocal Rank): 1/rank of first relevant. Single
  number that captures "is the right answer near the top".
- **nDCG@10**: discounted cumulative gain. Standard IR metric.
- **Recall@10**: fraction of relevant docs returned in top 10.
- **Precision@5**: of the top 5, how many are relevant.
- **Reranker delta**: same metrics with `rerank: true` vs `false`.
  This is the key number for justifying ship-or-not on the reranker.

**Output**: `bench/results/quality.<git-sha>.json` with per-query and
aggregate metrics, plus `quality.latest.md` for human review.

**Public dataset alternative** (lower priority): wire a small subset of
MS-MARCO or BEIR's `nfcorpus` for cross-checking on a known benchmark.
With bge-small-en-v1.5 we have published numbers to compare against.

### File layout

```
bench/
  perf.ts                 # entry: tsx bench/perf.ts
  quality.ts              # entry: tsx bench/quality.ts
  eval/
    eval.json             # generated; gitignored if private
    seed.ts               # generates synthetic queries from a DB
  lib/
    metrics.ts            # mrr / ndcg / recall / precision
    fixtures.ts            # tmp-dir DB factories at scale
  results/
    perf.<sha>.json
    perf.latest.md
    quality.<sha>.json
    quality.latest.md
```

`package.json` scripts:
- `bench:perf`  → `tsx bench/perf.ts`
- `bench:quality` → `tsx bench/quality.ts`
- `bench` → both, then writes the `.latest.md` summaries.

### Tests for the bench code itself

At minimum, unit tests for the metric calculators (`metrics.ts`):
MRR over a known list, nDCG with a hand-computed expected, etc. The
bench scripts themselves are too I/O-heavy for vitest — they're tools,
not tests.

---

## (3) End-to-end test harness

We have plenty of unit tests with `FakeEmbedder` and a hand-built fake
DB. What's missing: a test that runs the **whole** stack — real worker,
real ONNX, real DB, real `lcm_recall` — against a faithful pi runtime
stub.

### Why

- Catches integration regressions that unit tests miss (e.g. the
  settings-panel API shape change — a unit test for the panel passed,
  the pi-side call still crashed).
- Validates the worker IPC plumbing under realistic load.
- Validates schema migrations actually back-fill correctly.
- Validates `session_start` → backfill → `message_end` → search loop
  end to end.

### What it looks like

`test/e2e/full-pipeline.test.ts` (opt-in via `PI_LCM_MEMORY_LIVE_TEST=1`,
same gate as `worker.live.test.ts`):

```ts
describe("e2e: full pipeline", () => {
  it("backfill → search → hit", async () => {
    const tmp = await makeTmpProject({
      messages: 200,        // pre-seeded pi-lcm rows
      summaries: 30,
    });
    const pi = makeFakePi();
    const ext = await import("../index.js");
    await ext.default(pi);                    // installs hooks, tools
    await pi.fire("session_start", { reason: "resume" }, makeCtx(tmp));
    await waitFor(() => pi.tool("lcm_recall")) // wait for tool registration
    await waitForBackfill(tmp);                // poll memory_index count
    const hits = await pi.tool("lcm_recall")
      .execute({ query: "specific phrase from message #137", k: 5 });
    expect(hits[0].pi_lcm_msg_id).toBe("m137");
  });

  it("hook path: new message_end is searchable within ~1 turn", async () => { ... });

  it("primer rendering with a real corpus", async () => { ... });

  it("settings panel opens without crashing", async () => { ... });
});
```

### `makeFakePi()` helper (`test/e2e/fake-pi.ts`)

A faithful but minimal `ExtensionAPI` mock:
- `pi.on(event, handler)` records handlers; `pi.fire(event, e, ctx)`
  invokes them in registration order.
- `pi.registerTool(tool)` stores; `pi.tool(name)` returns the registered tool.
- `pi.registerCommand(name, def)` ditto; `pi.runCommand("/memory stats")`.
- `pi.ui` with `.notify`, `.setStatus`, and a minimal `.custom(factory)`
  that calls the factory and gives `done` synchronously — enough to
  exercise the panel construction without needing a real TUI.
- `pi.appendEntry`, `pi.registerProvider` → noops or store-only.

`makeCtx(tmp)` returns `{ cwd: tmp, ui: pi.ui, ... }`.

This is also useful for #5 future work (file indexing) and as a general
sanity-net.

### Things to avoid

- Don't load the real pi binary or shell out. Stay in-process, fast.
- Don't download the model in CI — e2e tests are opt-in via env var.
- Don't share state across e2e tests; each one builds its own tmp dir
  and tears down (we already have `makeTestDb` patterns to extend).

### File layout

```
test/
  e2e/
    fake-pi.ts             # makeFakePi() helper
    fixtures.ts             # tmp project + pi-lcm-shaped DB seeders
    full-pipeline.test.ts
    panel.test.ts
    backfill.test.ts
```

---

## (4) Establish baseline

Before touching the reranker, run:

```bash
npm run bench           # writes perf.<sha>.json + quality.<sha>.json
PI_LCM_MEMORY_LIVE_TEST=1 npx vitest run test/e2e/  # confirms e2e green
```

Commit the resulting `bench/results/*.json` and `*.latest.md`. These
become the "before" snapshot. Reference the SHA in the next CHANGELOG
entry.

---

## (5) Cross-encoder reranker

### Background

Current retrieval pipeline (`src/retrieval.ts`):
1. Embed the query (worker).
2. Vector kNN over `memory_vec` → top-K candidates with cosine distance.
3. FTS5 search over `messages_fts` → top-K candidates with bm25 rank.
4. Reciprocal Rank Fusion merge → top-K hybrid.
5. Hydrate via `memory_index` → return `{snippet, score, ...}`.

The hybrid merge is good for recall but not great for precision: the top
hit is often "vaguely related", not "directly relevant". For
auto-recall (which auto-injects results into the LLM context) this
matters a lot.

### Solution: cross-encoder reranker as a second stage

A cross-encoder takes `(query, doc)` pairs and returns a relevance score
by jointly attending to both. Much better precision than dual-encoder
(bi-encoder) embeddings, but ~10× slower per pair. Standard practice:
use the bi-encoder for retrieval (top-N), then rerank with a
cross-encoder.

**Model**: `Xenova/ms-marco-MiniLM-L-6-v2`.
- 22 MB quantized, English-tuned for query↔passage relevance.
- Outputs a single relevance score per pair.
- Available on HF, works with `@huggingface/transformers` via
  `text-classification` pipeline (or `feature-extraction` + classifier
  head — check the model card).

### Architecture

Re-use the existing worker. It already owns the @huggingface/transformers
runtime and ONNX threading config.

**Worker protocol additions** (`src/embeddings/worker.mjs`):

```
parent → worker:
  { type: 'init_reranker', opts: { model, quantize, cacheDir } }
  { type: 'rerank', id, query, docs }   // docs: string[]

worker → parent:
  { type: 'reranker_loaded', model }
  { type: 'rerank_result', id, scores: number[] }   // one per doc
  { type: 'progress', payload: { ... } }            // existing, shared
  { type: 'error', id, message, ... }
```

The reranker pipeline is loaded *separately* and *lazily* — only when
the user actually has rerank turned on. No cost otherwise.

**Embedder API additions** (`src/embeddings/embedder.ts`):

```ts
class Embedder {
  // existing: warmup(), embed(), terminate(), state(), workerUrl()
  warmupReranker(): Promise<void>
  rerank(query: string, docs: string[]): Promise<number[]>
  rerankerState(): { ready, loading, model } | null
}
```

**Retrieval integration** (`src/retrieval.ts`):

```ts
async retrieve(query: string, opts: { topK, rerank?: boolean }) {
  const merged = await this.hybridSearch(query, RERANK_POOL); // 30-ish
  if (!opts.rerank || !this.config.rerank) {
    return merged.slice(0, opts.topK);
  }
  const docs = merged.map(r => r.text_full);
  const scores = await this.embedder.rerank(query, docs);
  const reranked = merged
    .map((r, i) => ({ ...r, rerankScore: scores[i] }))
    .sort((a, b) => b.rerankScore - a.rerankScore)
    .slice(0, opts.topK);
  return reranked;
}
```

**Config** (`src/config.ts`):
- New: `rerank: boolean` (default `false`). Off by default — opt-in.
- New: `rerankModel: string` (default `Xenova/ms-marco-MiniLM-L-6-v2`).
- New: `rerankPoolSize: number` (default 30) — how many candidates to
  fetch from hybrid before reranking down to topK.

Settings panel: add three rows (boolean + enum + number).

### Performance considerations

- Reranker scoring 30 pairs ≈ 30 ms on a worker with 8 threads (extrapolating
  from current bge-small numbers; cross-encoder MiniLM is similar size).
- Total auto-recall latency goes from ~30 ms → ~60-80 ms. Acceptable.
- For `lcm_recall` tool (LLM-driven), cost is hidden behind the LLM
  call anyway.

### Tests

- Unit: `test/retrieval.rerank.test.ts` — fake reranker (sorts by length
  or some deterministic signal), assert hybrid order changes correctly
  when rerank is enabled, ties break correctly, falls through cleanly
  when reranker fails.
- Live (opt-in via `PI_LCM_MEMORY_LIVE_TEST=1`): seed a small corpus,
  run `lcm_recall("specific phrase from one doc")` with and without
  rerank, assert the directly relevant doc is `[0]` with rerank on
  and may not be without it.
- E2E: extend `test/e2e/full-pipeline.test.ts` to assert that the
  reranker is loaded lazily (not on session_start), only on first use.
- Quality: re-run `npm run bench:quality` and compare `nDCG@10` /
  `MRR` against the baseline. The reranker must show a measurable
  win or it doesn't ship.

### Open design questions

1. **Should we cache rerank scores?** A given `(query, doc)` pair has
   the same score every time. For frequently re-asked queries, caching
   helps. Probably not worth it for MVP — the LLM rarely repeats exact
   query strings.
2. **Auto-recall and rerank**: should auto-recall pay the rerank cost,
   or is that too much latency on every turn? Default: rerank only on
   the explicit `lcm_recall` tool, not auto-recall. User can opt
   auto-recall in via `autoRecallRerank: boolean` (default false).
3. **Model card check**: confirm `Xenova/ms-marco-MiniLM-L-6-v2` quantized
   variant exists. If not, fall back to fp32 (still small).

### Implementation order

1. Read transformers.js docs / model card for the reranker pipeline shape.
2. Add reranker plumbing to worker (init message + rerank message).
3. Embedder API: `warmupReranker`, `rerank`. Don't auto-warm.
4. Config + settings panel rows.
5. Retrieval integration (gated on config).
6. Unit tests with a fake reranker.
7. Live test (skipped by default).
8. **Re-run benchmarks (perf + quality) and capture the delta**.
9. **Update CHANGELOG with concrete before/after numbers**, link to
   the JSON snapshots in `bench/results/`.
10. Update ROADMAP, README.
11. `/memory rerank on|off` shortcut command (optional).

---

## Future research: #5 — Code / file content indexing

Not for the next session — for **a future** discussion. Notes so we
don't re-derive these:

### Goals

Ad-hoc semantic search over the user's actual code: "where is the file
that handles auth?" / "show me places we use jwt verification".

### Design space

- **Separate vector table**: `code_vec` with its own dim (likely a
  code-tuned model with different output size, e.g. 768 vs current 384).
  Reuse the worker, add a second pipeline.
- **Indexable units**: file (whole), function (chunked by AST), or
  fixed-size sliding window (e.g. 30-line chunks with 10-line overlap).
  AST chunking is best but requires a parser per language. Sliding
  window is universal and fine for retrieval.
- **Watcher**: chokidar or `fs.watch` — invalidate chunks when files
  change. Initial index runs as a background sweep similar to the
  conversation indexer; updates are reactive.
- **Respect `.gitignore`** at minimum. Probably also have a config
  blocklist of dirs (`node_modules`, `dist`, `target`).
- **Hybrid retrieval**: semantic over `code_vec` + lexical via FTS5 over
  raw file contents. Same RRF merge.
- **Storage**: a 50k-loc TypeScript codebase chunked into 30-line slabs
  ≈ 2k chunks × 1.5 KB ≈ 3 MB at 384-dim. Fine.

### Model candidates

- `nomic-ai/CodeRankEmbed` (768-dim, code-tuned).
- `Xenova/bge-small-en-v1.5` (current; works but generic).
- `microsoft/unixcoder-base-nine` (smaller, code-specific).
- Test all three on a small benchmark before committing.

### Big questions to answer before writing code

1. **Per-cwd or global?** Per-cwd (matches the current pattern, simpler
   migration story). Global cross-project recall is its own item (#4),
   not bundled here.
2. **Index timing**: at session start (one-shot) or continuously
   (file watcher)? Probably both — initial sweep then watch.
3. **Identifier-aware ranking**: a query "auth" should weight files
   *named* `auth.ts` highly. Sketch: blend file-path similarity with
   content similarity in the score.
4. **Tooling surface**: new tool `lcm_grep_semantic`? Or extend
   `lcm_recall` with a `source: "code" | "messages"` filter?
5. **Privacy / scope**: should code chunks be embedded locally only?
   (Yes — this whole project is local-only, but worth re-confirming.)

### Out-of-scope for now (parking lot)

Memory cards, cross-project recall, MCP wrapper, eviction, redaction,
`/memory trace` runtime toggle. All recorded in ROADMAP under "Quality
/ functionality stretches". Don't pull these in unless they actively
unblock something.

---

## Quick verification checklist for the next session

Before starting work:

```bash
cd /Users/sharkone/code/pi-lcm-memory
git pull
git log --oneline -10                  # see context
npx tsc --noEmit                        # confirm clean
npx vitest run                          # confirm 64 tests green
cat NEXT.md                             # this file
cat ROADMAP.md | head -60               # phase status
```

After housekeeping (step 1):
- 64+ tests still green (or new tests added).
- `git diff --stat` shows the cleanup is small (a few hundred lines max).
- Tracer still works (`PI_LCM_MEMORY_TRACE=1` env var).
- `_testing` export gone, `iter_chunk` trace gone or rate-limited,
  `events` is the canonical event-log subcommand.

After benchmark + e2e infra (steps 2 + 3):
- `npm run bench:perf` runs without errors.
- `npm run bench:quality` runs without errors (may need an eval set
  generated first).
- `PI_LCM_MEMORY_LIVE_TEST=1 npx vitest run test/e2e/` is green.
- Metrics module (`bench/lib/metrics.ts`) has unit tests.

After baseline capture (step 4):
- `bench/results/perf.<sha>.json` exists and is committed.
- `bench/results/quality.<sha>.json` exists and is committed (if eval
  set is non-private).
- `bench/results/perf.latest.md` and `quality.latest.md` are committed.

After reranker (step 5+):
- All previous tests still green.
- New `retrieval.rerank.test.ts` passes.
- `lcm_recall` works with rerank off (default) and on.
- Settings panel shows the three new rows.
- Worker reports the reranker as a separate pipeline in `/memory worker`.
- Re-run `npm run bench` shows MRR / nDCG@10 went up (or rerank is
  reverted).
- CHANGELOG updated under "Phase 6" with concrete before/after numbers.
