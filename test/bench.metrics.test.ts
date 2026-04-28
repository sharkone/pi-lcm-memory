/**
 * Unit tests for the IR metric calculators in bench/lib/metrics.ts.
 * The bench scripts themselves are too I/O-heavy for vitest.
 */

import { describe, it, expect } from "vitest";
import {
  reciprocalRank,
  recallAtK,
  precisionAtK,
  ndcgAtK,
  aggregate,
  percentiles,
} from "../bench/lib/metrics.js";

describe("reciprocalRank", () => {
  it("is 1/rank of the first hit", () => {
    expect(reciprocalRank(["a", "b", "c"], ["c"])).toBeCloseTo(1 / 3, 6);
    expect(reciprocalRank(["a", "b", "c"], ["a"])).toBeCloseTo(1, 6);
    expect(reciprocalRank(["a", "b", "c"], ["b", "c"])).toBeCloseTo(1 / 2, 6);
  });
  it("is 0 with no relevant docs", () => {
    expect(reciprocalRank(["a", "b"], [])).toBe(0);
  });
  it("is 0 when nothing relevant appears in ranking", () => {
    expect(reciprocalRank(["a", "b"], ["x", "y"])).toBe(0);
  });
});

describe("recallAtK", () => {
  it("counts hits in top-k over total relevant", () => {
    expect(recallAtK(["a", "b", "c", "d"], ["a", "c", "x"], 4)).toBeCloseTo(2 / 3, 6);
  });
  it("respects k", () => {
    expect(recallAtK(["a", "b", "c", "d"], ["d"], 3)).toBeCloseTo(0, 6);
    expect(recallAtK(["a", "b", "c", "d"], ["d"], 4)).toBeCloseTo(1, 6);
  });
  it("is 0 when no relevant docs", () => {
    expect(recallAtK(["a", "b"], [], 5)).toBe(0);
  });
});

describe("precisionAtK", () => {
  it("counts hits among top-k", () => {
    expect(precisionAtK(["a", "b", "c", "d"], ["a", "b"], 4)).toBeCloseTo(0.5, 6);
    expect(precisionAtK(["a", "b", "c"], ["a"], 1)).toBeCloseTo(1, 6);
  });
  it("clamps to actual list length", () => {
    // Only 2 results returned; "k=10" effectively means "k=2".
    expect(precisionAtK(["a", "b"], ["a"], 10)).toBeCloseTo(0.5, 6);
  });
  it("returns 0 for k=0", () => {
    expect(precisionAtK(["a"], ["a"], 0)).toBe(0);
  });
});

describe("ndcgAtK", () => {
  it("equals 1 when ideal ranking", () => {
    // Single relevant doc at rank 1 → DCG = IDCG = 1
    expect(ndcgAtK(["a", "b", "c"], ["a"], 3)).toBeCloseTo(1, 6);
  });
  it("decreases as relevant docs slide down the rank", () => {
    const top = ndcgAtK(["a", "b", "c"], ["a"], 3);
    const mid = ndcgAtK(["b", "a", "c"], ["a"], 3);
    const bot = ndcgAtK(["b", "c", "a"], ["a"], 3);
    expect(top).toBeGreaterThan(mid);
    expect(mid).toBeGreaterThan(bot);
  });
  it("is 0 with no relevant docs", () => {
    expect(ndcgAtK(["a", "b"], [], 5)).toBe(0);
  });
});

describe("aggregate", () => {
  it("averages metrics across queries", () => {
    const evals = [
      { query: "q1", ranked: ["a", "b"], relevant: ["a"] }, // mrr=1
      { query: "q2", ranked: ["a", "b"], relevant: ["b"] }, // mrr=0.5
    ];
    const m = aggregate(evals);
    expect(m.queries).toBe(2);
    expect(m.mrr).toBeCloseTo(0.75, 6);
  });
  it("returns zeros for empty input", () => {
    const m = aggregate([]);
    expect(m.queries).toBe(0);
    expect(m.mrr).toBe(0);
  });
});

describe("percentiles", () => {
  it("computes p50/p90/p99 + min/max/mean", () => {
    const samples = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100];
    const p = percentiles(samples);
    expect(p.count).toBe(10);
    expect(p.min).toBe(10);
    expect(p.max).toBe(100);
    expect(p.mean).toBeCloseTo(55, 6);
    // Floor(0.5*10)=5, sorted[5]=60
    expect(p.p50).toBe(60);
    expect(p.p90).toBe(100);
    expect(p.p99).toBe(100);
  });
  it("handles single sample", () => {
    const p = percentiles([42]);
    expect(p.count).toBe(1);
    expect(p.min).toBe(42);
    expect(p.max).toBe(42);
    expect(p.mean).toBe(42);
  });
  it("returns zeros for empty input", () => {
    const p = percentiles([]);
    expect(p.count).toBe(0);
    expect(p.mean).toBe(0);
  });
});
