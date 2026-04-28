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

## Phase 5 — Stretches (any order, opt-in)

- [ ] Cross-encoder re-ranker (`Xenova/ms-marco-MiniLM-L-6-v2`) on top-N
      hybrid results (config flag).
- [ ] "Memory cards" (manually saved snippets via `/memory save` and a tool).
      *Mirrors the prior local attempt; revisit user need.*
- [ ] Cross-project / workspace-wide recall (multi-DB query layer).
- [ ] Code / file content indexing (separate `code_vec` table; AST-aware
      chunking).

- [ ] MCP wrapper exposing recall to Claude Code (Q3 deferred path).
- [ ] Eviction / retention policy (e.g., archive after N days, keep summaries).
- [ ] Edit / redact memory (with audit log).

---

## Tracking

- Each merged change updates `CHANGELOG.md`.
- This roadmap is the source of truth for ordering; PLAN.md for *how*.
- Issues / TODOs that pop up mid-phase are added at the **bottom** of the
  current phase or under "Stretches", never silently inflated.
