/**
 * Pure IR metric calculators for recall-quality benchmarking.
 *
 * All functions take:
 *   - a ranked list of result IDs (predicted, in descending relevance order),
 *   - a set/array of the IDs that are *actually* relevant for that query.
 *
 * They are deterministic and have no I/O, so they're cheap to unit-test.
 */

/** Mean Reciprocal Rank for one query. 0 if no relevant doc is in `ranked`. */
export function reciprocalRank(ranked: string[], relevant: Iterable<string>): number {
  const rel = new Set(relevant);
  if (rel.size === 0) return 0;
  for (let i = 0; i < ranked.length; i++) {
    if (rel.has(ranked[i]!)) return 1 / (i + 1);
  }
  return 0;
}

/** Recall@k — fraction of relevant docs that appear in the top-k results. */
export function recallAtK(ranked: string[], relevant: Iterable<string>, k: number): number {
  const rel = new Set(relevant);
  if (rel.size === 0) return 0;
  const top = ranked.slice(0, k);
  let hit = 0;
  for (const id of top) if (rel.has(id)) hit++;
  return hit / rel.size;
}

/** Precision@k — fraction of top-k results that are relevant. */
export function precisionAtK(ranked: string[], relevant: Iterable<string>, k: number): number {
  if (k <= 0) return 0;
  const rel = new Set(relevant);
  const top = ranked.slice(0, k);
  if (top.length === 0) return 0;
  let hit = 0;
  for (const id of top) if (rel.has(id)) hit++;
  return hit / Math.min(k, top.length);
}

/**
 * nDCG@k with binary relevance.
 *
 *   DCG@k  = Σ_{i=1..k} rel_i / log2(i + 1)
 *   IDCG@k = DCG of the ideal ranking (all relevant docs first)
 *   nDCG@k = DCG@k / IDCG@k   (0 when no relevant docs)
 */
export function ndcgAtK(ranked: string[], relevant: Iterable<string>, k: number): number {
  const rel = new Set(relevant);
  if (rel.size === 0) return 0;
  const top = ranked.slice(0, k);

  let dcg = 0;
  for (let i = 0; i < top.length; i++) {
    if (rel.has(top[i]!)) dcg += 1 / Math.log2(i + 2);
  }

  // Ideal: as many relevant docs as possible up front, capped at k.
  const idealHits = Math.min(rel.size, k);
  let idcg = 0;
  for (let i = 0; i < idealHits; i++) idcg += 1 / Math.log2(i + 2);

  return idcg === 0 ? 0 : dcg / idcg;
}

export interface QueryEval {
  /** The user-facing query string. */
  query: string;
  /** Ranked list of result IDs returned by the retriever. */
  ranked: string[];
  /** IDs that should have appeared (gold standard). */
  relevant: string[];
}

export interface AggregateMetrics {
  queries: number;
  mrr: number;
  recallAt5: number;
  recallAt10: number;
  precisionAt5: number;
  ndcgAt10: number;
}

/** Compute mean metrics over a list of evaluated queries. */
export function aggregate(evals: QueryEval[]): AggregateMetrics {
  if (evals.length === 0) {
    return {
      queries: 0,
      mrr: 0,
      recallAt5: 0,
      recallAt10: 0,
      precisionAt5: 0,
      ndcgAt10: 0,
    };
  }
  let mrr = 0;
  let r5 = 0;
  let r10 = 0;
  let p5 = 0;
  let nd10 = 0;
  for (const q of evals) {
    mrr += reciprocalRank(q.ranked, q.relevant);
    r5 += recallAtK(q.ranked, q.relevant, 5);
    r10 += recallAtK(q.ranked, q.relevant, 10);
    p5 += precisionAtK(q.ranked, q.relevant, 5);
    nd10 += ndcgAtK(q.ranked, q.relevant, 10);
  }
  const n = evals.length;
  return {
    queries: n,
    mrr: mrr / n,
    recallAt5: r5 / n,
    recallAt10: r10 / n,
    precisionAt5: p5 / n,
    ndcgAt10: nd10 / n,
  };
}

/** Compute basic summary statistics over a numeric sample. */
export function percentiles(samples: number[]): {
  count: number;
  min: number;
  max: number;
  mean: number;
  p50: number;
  p90: number;
  p99: number;
} {
  if (samples.length === 0) {
    return { count: 0, min: 0, max: 0, mean: 0, p50: 0, p90: 0, p99: 0 };
  }
  const sorted = [...samples].sort((a, b) => a - b);
  const pct = (p: number) => {
    const idx = Math.min(sorted.length - 1, Math.floor(p * sorted.length));
    return sorted[idx]!;
  };
  let sum = 0;
  for (const x of sorted) sum += x;
  return {
    count: sorted.length,
    min: sorted[0]!,
    max: sorted[sorted.length - 1]!,
    mean: sum / sorted.length,
    p50: pct(0.5),
    p90: pct(0.9),
    p99: pct(0.99),
  };
}
