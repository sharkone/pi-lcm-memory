# pi-lcm-memory — ROADMAP

Phased delivery. Each phase ships a working extension; later phases are pure
additions (no schema rewrites unless explicitly noted).

Cross-references the [PLAN](./PLAN.md) for design details.

---

## Phase 0 — Bootstrap

**Goal:** Repo, docs, and skeleton that compiles and loads as a no-op Pi
extension.

- [x] Interview locked (Q1–Q8).
- [ ] `package.json`, `tsconfig.json`, `.gitignore`, `LICENSE` (MIT).
- [ ] `README.md` (intent + install pointer).
- [ ] `PLAN.md`, `ROADMAP.md`, `CHANGELOG.md`.
- [ ] Git repo init with `sharkone@en-mousse.org`.
- [ ] First commit: docs + scaffolding only.
- [ ] CI / pre-commit (defer; optional).

**Exit criteria:** repo builds cleanly with `tsc --noEmit`. Loads as a Pi
extension and prints a single `[pi-lcm-memory] enabled (v0)` notice on
`session_start`.

---

## Phase 1 — MVP: passive index + tool-based recall (decisions A + 6)

**Goal:** Every new message embedded; `lcm_recall` returns top-K hybrid
results. No primer yet, no auto-recall.

- [ ] `src/db/connection.ts` — open shared DB by mirroring pi-lcm's
      `hashCwd(cwd)` path scheme.
- [ ] `src/db/schema.ts` — additive migrations: `memory_vec`, `memory_index`,
      `memory_meta`. Idempotent.
- [ ] `src/db/vec.ts` — load `sqlite-vec` extension; soft-fail with clear
      message if unavailable.
- [ ] `src/embeddings/embedder.ts` — Transformers.js pipeline, lazy load,
      configurable model.
- [ ] `src/bridge.ts` — read-only access to pi-lcm `messages` + `summaries`.
- [ ] `src/indexer.ts` — `message_end` hook + 30s sweep loop.
- [ ] `src/retrieval.ts` — RRF merge of FTS5 + vec.
- [ ] `src/tools/lcm-recall.ts` — registered via `pi.registerTool`.
- [ ] `src/tools/lcm-similar.ts` — registered via `pi.registerTool`.
- [ ] Tests: `schema`, `embedder`, `indexer` (insert→row), `retrieval` (RRF
      ranking on synthetic fixtures).
- [ ] `/memory stats` command.

**Exit criteria:**
- Open Pi in a project where pi-lcm has prior content; first session triggers
  a backfill sweep that embeds all existing messages and summaries.
- Subsequent messages auto-index.
- `lcm_recall("…")` returns relevant top-K with snippet + score on a real
  test corpus.
- All tests green; no patches to pi-lcm.

---

## Phase 2 — Session-start primer (decision B)

**Goal:** New sessions in a known project receive a brief on prior memory.

- [ ] `src/primer.ts` — query top-K most recent summaries (depth ≥ 1), render
      markdown block within token budget.
- [ ] Wire into `context` event for first-turn injection.
- [ ] `primerTopK` and `primer: false` setting.
- [ ] Empty-state behavior (no primer when project has no memory).
- [ ] Tests: empty / single-session / multi-session corpora; token-budget
      enforcement.

**Exit criteria:** New session in a project with ≥1 prior session shows a
primer ≤ 300 tokens with date, count, and recent topics.

---

## Phase 3 — Heuristic auto-recall (decision D)

**Goal:** "Remember…", "earlier…", etc. trigger an auto-recall block injected
on that turn only.

- [ ] `src/auto-recall.ts` — phrase regex set (configurable); top-K injection
      with token cap.
- [ ] Wire into `context` (per-turn) — non-persistent, current-turn-only.
- [ ] `autoRecall` setting: `off` | `heuristic` | `always` (latter implements
      decision C as a stretch toggle).
- [ ] Tests: positive/negative phrase fixtures; token-budget cap.

**Exit criteria:** Saying "do you remember the auth refactor?" produces a
visible `## Recall` block with relevant snippets in the next assistant turn.

---

## Phase 4 — Operator polish

**Goal:** Make pi-lcm-memory pleasant to live with.

- [ ] `/memory reindex`, `/memory model`, `/memory clear`, `/memory status`.
- [ ] `ctx.ui.setStatus(...)` footer with index/sweep state.
- [ ] CHANGELOG and a README "Quick start" mirrored from pi-lcm's style.
- [ ] Notify on first-time model download with size.
- [ ] Project-vs-global settings precedence + writeback.
- [ ] Performance tightening: batched embedder calls; sweep adaptive interval
      (idle → backoff).
- [ ] Optional: structured event log to `memory_meta` for diagnostics.

**Exit criteria:** A user can change embedding model with one command and
have everything re-embedded in the background, with a footer status visible.

---

## Phase 5 — Stretches (any order, opt-in)

- [ ] Cross-encoder re-ranker (`Xenova/ms-marco-MiniLM-L-6-v2`) on top-N
      hybrid results (config flag).
- [ ] "Memory cards" (manually saved snippets via `/memory save` and a tool).
      *Mirrors the prior local attempt; revisit user need.*
- [ ] Cross-project / workspace-wide recall (multi-DB query layer).
- [ ] Code / file content indexing (separate `code_vec` table; AST-aware
      chunking).
- [ ] Settings panel UI (`pi.registerSettingsPanel(...)`).
- [ ] MCP wrapper exposing recall to Claude Code (Q3 deferred path).
- [ ] Eviction / retention policy (e.g., archive after N days, keep summaries).
- [ ] Edit / redact memory (with audit log).

---

## Tracking

- Each merged change updates `CHANGELOG.md`.
- This roadmap is the source of truth for ordering; PLAN.md for *how*.
- Issues / TODOs that pop up mid-phase are added at the **bottom** of the
  current phase or under "Stretches", never silently inflated.
