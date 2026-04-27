# pi-lcm-memory

> Persistent, cross-session **semantic memory** for Pi. An additive extension
> on top of [pi-lcm](https://github.com/codexstar69/pi-lcm) — perfect recall
> across every session in a project, retrievable by hybrid (FTS5 + vector)
> search. Fully local; no external APIs.

## Status

Pre-alpha. Design locked, scaffolding in progress. See [PLAN.md](./PLAN.md)
and [ROADMAP.md](./ROADMAP.md).

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

## How it works (short version)

- pi-lcm writes every message and DAG summary to a per-project SQLite.
- We hook the same Pi events and embed user/assistant text + DAG summaries
  with a small local model (`bge-small-en-v1.5` by default; configurable).
- A `sqlite-vec` virtual table lives next to pi-lcm's tables in the same DB.
- `lcm_recall(query)` runs FTS5 + vector KNN and merges them with Reciprocal
  Rank Fusion.
- A background sweep guarantees no message is ever silently un-indexed.

## Install

> Not yet published. Targeted command:
> ```sh
> pi install npm:pi-lcm-memory
> ```

## Local dev

```sh
git clone <this repo>
cd pi-lcm-memory
npm install
npm test
pi -e ./index.ts
```

## License

MIT.
