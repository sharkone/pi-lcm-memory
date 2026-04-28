/**
 * Build a recall-quality eval set from a real pi-lcm DB by exploiting the
 * `summary_sources` DAG: each summary is a *paraphrased* condensation of
 * its source messages. So:
 *
 *   query    = summary text (truncated)
 *   relevant = pi-lcm message ids in summary_sources where source_type='message'
 *
 * This is far harder than the synthetic eval because the summary uses
 * different vocabulary than the source messages (LLM paraphrase). It is
 * representative of the canonical lcm_recall use case: "I have a high-level
 * idea of what I said, can the system surface the original messages?"
 *
 * Privacy: the resulting eval may contain personal/sensitive content.
 * Callers must NOT commit it. The bench harness only writes aggregate
 * metrics; the per-query text never leaves the JSON unless the user
 * opts in by committing the file explicitly.
 */

import Database from "better-sqlite3";

export interface RealEvalRow {
  id: string;
  text: string;
  role: string;
  conversation_id: string;
}

export interface RealEvalSummary {
  id: string;
  text: string;
  depth: number;
  conversation_id: string;
}

export interface RealEvalQuery {
  /** Trimmed summary text used as the natural-language query. */
  query: string;
  /** pi-lcm message ids that the summary attests to. */
  relevant: string[];
  /** Source summary id (for diagnostics). */
  source_summary_id: string;
  source_summary_depth: number;
}

export interface RealEvalSet {
  messages: RealEvalRow[];
  summaries: RealEvalSummary[];
  queries: RealEvalQuery[];
}

export type QueryStyle =
  /** Use the trimmed summary text directly as the query (long, prose-style). */
  | "summary"
  /** Extract distinctive keywords from the summary (short, query-style). */
  | "keywords";

export interface BuildOptions {
  /** Maximum query length in characters (summary text is truncated). */
  maxQueryChars?: number;
  /** Drop summaries with fewer than N source messages (too easy / not useful). */
  minRelevant?: number;
  /** Drop summaries with more than N source messages (too broad). */
  maxRelevant?: number;
  /**
   * Cap the number of queries (sample uniformly across depths). Useful for
   * keeping bench runs fast.
   */
  maxQueries?: number;
  /** Restrict to a single conversation_id (for testing). */
  conversationId?: string;
  /** How to derive the query from each summary. Default: "summary". */
  queryStyle?: QueryStyle;
  /** For keyword style: how many distinctive tokens to keep. Default 8. */
  keywordCount?: number;
}

/**
 * Read a pi-lcm DB and produce a real-data eval set.
 *
 * The DB must have the standard pi-lcm tables: `messages`, `summaries`,
 * `summary_sources`. We never write to it.
 */
export function buildRealEvalSet(dbPath: string, opts: BuildOptions = {}): RealEvalSet {
  const maxQueryChars = opts.maxQueryChars ?? 280;
  const minRelevant = opts.minRelevant ?? 2;
  const maxRelevant = opts.maxRelevant ?? 50;
  const maxQueries = opts.maxQueries ?? 200;
  const conv = opts.conversationId;

  const db = new Database(dbPath, { readonly: true });
  try {
    // Basic table existence sanity.
    const tables = db
      .prepare("SELECT name FROM sqlite_master WHERE type='table'")
      .all() as { name: string }[];
    const have = new Set(tables.map((t) => t.name));
    for (const t of ["messages", "summaries", "summary_sources"]) {
      if (!have.has(t)) {
        throw new Error(`pi-lcm DB at ${dbPath} is missing table '${t}'`);
      }
    }

    const messages = db
      .prepare(
        conv
          ? "SELECT id, role, content_text AS text, conversation_id FROM messages WHERE conversation_id = ? ORDER BY seq"
          : "SELECT id, role, content_text AS text, conversation_id FROM messages ORDER BY conversation_id, seq",
      )
      .all(...(conv ? [conv] : [])) as RealEvalRow[];

    // Filter out empty / noise rows.
    const cleanMessages = messages.filter((m) => (m.text ?? "").trim().length > 0);

    const summaries = db
      .prepare(
        conv
          ? "SELECT id, text, depth, conversation_id FROM summaries WHERE conversation_id = ? ORDER BY depth, id"
          : "SELECT id, text, depth, conversation_id FROM summaries ORDER BY conversation_id, depth, id",
      )
      .all(...(conv ? [conv] : [])) as RealEvalSummary[];

    // For each summary, fetch its source messages (depth-1 indirection only).
    const sourceStmt = db.prepare(
      "SELECT source_id FROM summary_sources WHERE summary_id = ? AND source_type = 'message' ORDER BY seq",
    );
    // For higher-depth summaries (D2+), we also follow indirect message
    // sources via descendant summaries so the eval has enough relevant
    // docs to be discriminating.
    const childSummariesStmt = db.prepare(
      "SELECT source_id FROM summary_sources WHERE summary_id = ? AND source_type = 'summary'",
    );
    const messageIdSet = new Set(cleanMessages.map((m) => m.id));

    const queries: RealEvalQuery[] = [];
    const seenSummaryIds = new Set<string>();

    function flattenSources(summaryId: string, acc: Set<string>, visited: Set<string>): void {
      if (visited.has(summaryId)) return;
      visited.add(summaryId);
      const direct = sourceStmt.all(summaryId) as { source_id: string }[];
      for (const r of direct) {
        if (messageIdSet.has(r.source_id)) acc.add(r.source_id);
      }
      const indirect = childSummariesStmt.all(summaryId) as { source_id: string }[];
      for (const r of indirect) flattenSources(r.source_id, acc, visited);
    }

    for (const s of summaries) {
      if (seenSummaryIds.has(s.id)) continue;
      seenSummaryIds.add(s.id);
      const acc = new Set<string>();
      flattenSources(s.id, acc, new Set());
      const relevant = [...acc];
      if (relevant.length < minRelevant) continue;
      if (relevant.length > maxRelevant) continue;

      const trimmed = trimQuery(s.text, maxQueryChars);
      if (!trimmed) continue;

      queries.push({
        query: trimmed,
        relevant,
        source_summary_id: s.id,
        source_summary_depth: s.depth,
      });
    }

    // If keyword-style queries are requested, post-process each query through
    // a corpus-aware extractor. We compute IDF over all summaries' tokens
    // (this gives the "distinctive" weighting), then for each summary keep
    // the top-N tokens by TF × IDF.
    if (opts.queryStyle === "keywords") {
      const idf = computeIdf(summaries);
      const keep = opts.keywordCount ?? 8;
      for (const q of queries) {
        q.query = extractKeywords(q.query, idf, keep);
      }
    }

    // Sample uniformly across depth strata if we exceed the cap.
    const sampled = sampleAcrossDepths(queries, maxQueries);

    return {
      messages: cleanMessages,
      summaries,
      queries: sampled,
    };
  } finally {
    db.close();
  }
}

/**
 * pi-lcm summaries are usually long, but real `lcm_recall` queries from the
 * LLM are short — the reranker model (`Xenova/ms-marco-MiniLM-L-6-v2`) was
 * trained on short queries paired with longer passages, so feeding it
 * full summary text is out-of-distribution. Try to extract a short
 * query-shaped string from the summary's first informative sentence.
 *
 * Strategy:
 *   1. Drop a leading markdown heading (`# Title`) if present — they're
 *      usually less specific than the body.
 *   2. Take up to maxChars of the first sentence or paragraph break.
 *   3. Strip backticks / code-fence noise that don't make sense as queries.
 */
function trimQuery(text: string, maxChars: number): string {
  let t = text.trim();
  if (!t) return "";

  // Drop a leading markdown heading line if there is more substance below it.
  if (t.startsWith("#")) {
    const nl = t.indexOf("\n");
    if (nl > 0 && nl < t.length - 5) t = t.slice(nl + 1).trim();
  }

  // Strip code-fence blocks — unlikely to appear in a real recall query.
  t = t.replace(/```[\s\S]*?```/g, " ").replace(/`/g, "").trim();

  if (t.length <= maxChars) return t;

  // Prefer a sentence boundary or paragraph break for a natural-feeling query.
  const slice = t.slice(0, maxChars + 50);
  const periodIdx = slice.lastIndexOf(". ");
  const newlineIdx = slice.lastIndexOf("\n");
  const cut = Math.max(periodIdx, newlineIdx);
  if (cut > maxChars * 0.5) return slice.slice(0, cut + 1).trim();
  return t.slice(0, maxChars).trim();
}

/**
 * IDF over a list of "documents" (summary texts). Standard
 *   idf(t) = log( (N + 1) / (df(t) + 1) ) + 1
 * Tokens are lower-cased ASCII word characters of length ≥3.
 */
function computeIdf(docs: { text: string }[]): Map<string, number> {
  const df = new Map<string, number>();
  for (const d of docs) {
    const seen = new Set<string>();
    for (const tok of tokenize(d.text)) seen.add(tok);
    for (const t of seen) df.set(t, (df.get(t) ?? 0) + 1);
  }
  const N = docs.length;
  const idf = new Map<string, number>();
  for (const [t, dfv] of df) idf.set(t, Math.log((N + 1) / (dfv + 1)) + 1);
  return idf;
}

function extractKeywords(text: string, idf: Map<string, number>, k: number): string {
  const tf = new Map<string, number>();
  for (const t of tokenize(text)) tf.set(t, (tf.get(t) ?? 0) + 1);
  const ranked = [...tf.entries()]
    .map(([tok, freq]) => ({ tok, score: freq * (idf.get(tok) ?? 0) }))
    .filter((r) => STOPWORDS.has(r.tok) === false)
    .sort((a, b) => b.score - a.score)
    .slice(0, k);
  return ranked.map((r) => r.tok).join(" ");
}

function tokenize(text: string): string[] {
  const out: string[] = [];
  const lower = text.toLowerCase();
  let buf = "";
  for (const ch of lower) {
    const isWord = (ch >= "a" && ch <= "z") || (ch >= "0" && ch <= "9");
    if (isWord) {
      buf += ch;
    } else {
      if (buf.length >= 3 && !STOPWORDS.has(buf)) out.push(buf);
      buf = "";
    }
  }
  if (buf.length >= 3 && !STOPWORDS.has(buf)) out.push(buf);
  return out;
}

/** Compact English stopword list — enough to skip noise without depending on a tokenizer. */
const STOPWORDS = new Set<string>(
  (
    "a an and any are as at be been being but by can did do does doing done from for has have having he her him his in into is it its just made make may more most much must my new not now of off on one only or other our out over same so some such than that the their them then there these they this those to too two up upon use used very was way we were what when where which while who why will with would you your also been being have having what when where which who whom how there here our its them they these those etc".split(/\s+/)
  ),
);

function sampleAcrossDepths(queries: RealEvalQuery[], cap: number): RealEvalQuery[] {
  if (queries.length <= cap) return queries;
  // Bucket by depth, then round-robin take.
  const byDepth = new Map<number, RealEvalQuery[]>();
  for (const q of queries) {
    const arr = byDepth.get(q.source_summary_depth) ?? [];
    arr.push(q);
    byDepth.set(q.source_summary_depth, arr);
  }
  const depths = [...byDepth.keys()].sort((a, b) => a - b);
  const out: RealEvalQuery[] = [];
  let cursor = 0;
  while (out.length < cap) {
    const d = depths[cursor % depths.length]!;
    const arr = byDepth.get(d)!;
    if (arr.length > 0) out.push(arr.shift()!);
    cursor++;
    // If all buckets empty, bail.
    if (depths.every((dx) => (byDepth.get(dx)?.length ?? 0) === 0)) break;
  }
  return out;
}
