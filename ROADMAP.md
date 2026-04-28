# 🗺️ pi-lcm-memory — ROADMAP

Phased delivery. Each phase ships a working extension; later phases are pure
additions (no schema rewrites unless explicitly noted).

Cross-references [PLAN.md](./PLAN.md) for design details and [CHANGELOG.md](./CHANGELOG.md) for what landed.

---

## ✅ Phase 0 — Bootstrap

**Goal:** Repo, docs, and skeleton that compiles and loads as a no-op Pi extension.

- [x] Interview locked (Q1–Q8)
- [x] `package.json`, `tsconfig.json`, `.gitignore`, `LICENSE` (MIT)
- [x] `README.md`, `PLAN.md`, `ROADMAP.md`, `CHANGELOG.md`
- [x] Git repo init (`sharkone@en-mousse.org`)
- [x] First commit: docs + scaffolding only
- [ ] CI / pre-commit *(deferred; optional)*

**Exit criteria:** builds cleanly with `tsc --noEmit`; loads as a Pi extension and prints `[pi-lcm-memory] enabled (v0)` on `session_start`. ✅

---

## ✅ Phase 1 — MVP: passive index + hybrid recall

**Goal:** Every new message embedded; `lcm_recall` returns top-K hybrid results. Session-start primer (B) and heuristic auto-recall (D) folded in.

- [x] `src/db/connection.ts` — shared DB via pi-lcm's `hashCwd(cwd)` scheme
- [x] `src/db/schema.ts` — additive migrations (`memory_vec`, `memory_index`, `memory_meta`), idempotent
- [x] `src/db/vec.ts` — load `sqlite-vec`; soft-fail with clear message
- [x] `src/embeddings/embedder.ts` — Transformers.js pipeline, lazy load, configurable model
- [x] `src/bridge.ts` — read-only access to pi-lcm `messages` + `summaries`
- [x] `src/indexer.ts` — `message_end` hook + 30 s sweep loop
- [x] `src/retrieval.ts` — RRF merge of FTS5 + vec
- [x] `src/tools/lcm-recall.ts` — registered via `pi.registerTool`
- [x] `src/tools/lcm-similar.ts` — registered via `pi.registerTool`
- [x] `src/settings-panel.ts` — interactive TUI panel (parity with pi-lcm UX)
- [x] `src/commands.ts` — `/memory` dispatcher
- [x] `src/primer.ts` — session-start briefing
- [x] `src/auto-recall.ts` — heuristic trigger
- [x] 44 vitest tests, all green

**Exit criteria:** `lcm_recall("…")` returns relevant top-K on a real corpus; no patches to pi-lcm. ✅

> *Phases 2 & 3 (primer + auto-recall) were folded into Phase 1 per user decision (Q8: A+B+D).*

---

## ✅ Phase 4 — Operator polish

**Goal:** Make pi-lcm-memory pleasant to live with day-to-day.

- [x] `/memory reindex`, `/memory model`, `/memory clear --yes`, `/memory status`
- [x] Footer status bar (index progress, sweep state, model download %)
- [x] First-time model download notice with size + per-MB progress
- [x] Project-vs-global settings precedence + writeback
- [x] Batched embedder (32 rows per inference call); adaptive sweep backoff (idle ×2, max 5 min)
- [x] Rolling diagnostics ring (200 events in `memory_meta`) + `/memory events`
- [x] Lazy capture of pi-lcm `conversation_id` (enables `sessionFilter`)
- [x] 60 vitest tests, all green

**Exit criteria:** change embedding model in one command; watch re-embedding in the footer. ✅

---

## ✅ Phase 5 — Stabilization

**Goal:** Zero TUI freezes; rock-solid under concurrent pi-lcm writes.

- [x] **Worker-thread embedder** — all ONNX inference in `worker_threads`; zero main-thread blocking; multi-core ORT (`cpus()-1`, max 8); zero-copy `ArrayBuffer` transfers
- [x] **Single-transaction batched inserts** — `insertBatch()` with `IMMEDIATE` lock + reused statements; 15× faster, eliminates lock contention with pi-lcm
- [x] **Side-channel tracer** (`PI_LCM_MEMORY_TRACE=1`) — synchronous file log that survives main-thread freezes; main + worker write to same file
- [x] **Schema v2: many-to-one id mapping** — `memory_index_msg` / `memory_index_sum` side tables; closes dedupe leak where identical-content messages shared one embedding but only one id was recorded
- [x] **Bridge rowid cursor + SQL filter** — forward-only iteration; skipped roles filtered at SQL level; safety yield every 1024 items; infinite-loop class structurally impossible
- [x] **Settings panel API fix** — `ctx.ui.custom(factory, options)` shape; `done` callback wired to panel `onClose`
- [x] Watchdog, `/memory worker`, per-MB download notifications, worker hello message

**Key fix:** An infinite loop in `messagesNotInMemoryIndex` was starving the event loop at 100% CPU. Tool-I/O rows were never inserted into `memory_index`, so they matched the LEFT JOIN on every sweep tick forever — the for-of loop never reached a 32-row batch and the TUI froze. Diagnosed via the side-channel tracer (2.18M `iter_chunk` events, zero `batch_start`). Three-part fix: SQL-level role filter, rowid cursor advance before yield, safety yield every 1024 items.

---

## ✅ Phase 6 — Benchmarking + quality measurement

**Goal:** Evidence-based quality claims. Build infrastructure first, then measure.

- [x] **Housekeeping pass** — dropped orphan exports, noisy trace events, alias commands
- [x] **Performance benchmarks** (`bench/perf.ts`, `npm run bench:perf`) — embed throughput, hook/sweep latencies, recall latency, DB growth
- [x] **Recall quality benchmarks** (`bench/quality.ts`, `npm run bench:quality`) — MRR, nDCG@10, Recall@K, Precision@K
- [x] **`bench/lib/real-eval.ts`** — derives eval queries from any pi-lcm DB via `summary_sources` DAG + optional TF×IDF keyword extraction
- [x] **End-to-end test harness** (`test/e2e/`) — `makeFakePi()` stub + real worker + real ONNX + real DB; opt-in via `PI_LCM_MEMORY_LIVE_TEST=1`
- [x] **Baseline snapshots** committed to `bench/results/`
- [x] **Cross-encoder reranker — evaluated and removed** (see below)

### 🔬 Cross-encoder reranker: post-mortem

We built a complete reranker (`Xenova/ms-marco-MiniLM-L-6-v2`) and ran it against the real pi-lcm DB (749 messages, 79 summaries, 57 queries). Synthetic eval: spectacular (+175% MRR). Real-data eval: the opposite.

| Configuration | MRR Δ | nDCG@10 Δ |
|---|---:|---:|
| Summary-style query, full corpus | −0.502 | −0.178 |
| Summary-style query, messages-only | −0.016 | +0.009 |
| Keyword query, full corpus | −0.269 | −0.096 |
| Keyword query, messages-only | **+0.139** | **+0.033** |

**Root cause:** cross-encoders trained on MS-MARCO (short query / long passage) promote stylistically-matched passages. pi-lcm summaries are LLM paraphrases — long, prose-heavy — so the reranker demoted messages in favour of summaries (76% summaries in top-10 vs. 5% for hybrid alone). Even in the one winning regime, latency was ~580 ms/query vs. 12 ms for hybrid — 47× slower, wrong shape for auto-recall.

**Takeaway:** The bench infrastructure is the real deliverable. **Every future recall-quality claim must be backed by a `bench/quality.ts` real-data run.**

---

## 🔭 Future research

> Not scheduled. Pull into a numbered phase only when actively useful.
> Every item must be measured against `bench/quality.ts` (real-data eval) before shipping.

- [ ] **Tune existing hybrid first** — sweep RRF k, lex/sem candidate breadths, summary indexing strategy. Likely the cheapest next win.
- [ ] **Code / file content indexing** — separate `code_vec` table, AST-aware chunking. Flagged as interesting for future discussion.
- [ ] **Memory cards** — manually saved snippets via `/memory save` and a tool. Mirrors a prior local attempt.
- [ ] **Domain-tuned reranker** — trained on conversational paraphrase pairs (not MS-MARCO). Open research question; public cross-encoders carry the style-bias documented above.
- [ ] **Cross-project / workspace-wide recall** — multi-DB query layer.
- [ ] **MCP wrapper** — expose recall to Claude Code (Q3 deferred path).
- [ ] **Eviction / retention policy** — archive after N days, keep summaries. Matters at 100k+ rows.
- [ ] **Edit / redact memory** — with audit log.
- [ ] **`/memory trace on|off`** — toggle tracer without restarting Pi.

---

## 📌 Tracking

- Merged changes → `CHANGELOG.md`
- This file is the source of truth for phase ordering; `PLAN.md` covers *how*
- Issues / TODOs that surface mid-phase are added at the bottom of the current phase, never silently inflated
