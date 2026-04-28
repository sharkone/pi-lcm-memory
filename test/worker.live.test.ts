/**
 * Live integration test for the embedding worker.
 *
 * Skipped by default. Run with:
 *   PI_LCM_MEMORY_LIVE_TEST=1 npx vitest run test/worker.live.test.ts
 *
 * Downloads the configured model on first run (~33 MB for bge-small q8)
 * and exercises real ONNX inference inside the worker thread.
 */

import { describe, it, expect } from "vitest";
import { Embedder } from "../src/embeddings/embedder.js";

const live = process.env.PI_LCM_MEMORY_LIVE_TEST === "1";
const describeLive = live ? describe : describe.skip;

describeLive("worker live (requires model download)", () => {
  it("loads, embeds a batch in a worker, and shuts down cleanly", async () => {
    const e = new Embedder({
      model: "Xenova/bge-small-en-v1.5",
      quantize: "q8",
      cacheDir: null,
    });

    let progressFired = false;
    let loadedFired = false;
    e.setListener({
      onProgress: () => {
        progressFired = true;
      },
      onLoaded: () => {
        loadedFired = true;
      },
    });

    try {
      await e.warmup();

      const state = e.state();
      expect(state.ready).toBe(true);
      expect(state.dims).toBe(384);
      // Most workers should report >=1 thread; on tiny single-core CI runners
      // it could be 1.
      expect(state.intraOpNumThreads).toBeGreaterThan(0);
      expect(loadedFired).toBe(true);
      // progressFired is only true on cold cache; skip the assertion if
      // the model was already cached from a previous run.

      const texts = [
        "the auth refactor we did last week",
        "renaming variables in the settings panel",
        "JWT validation issue resolved",
        "added a new logging system",
        "auth flow with cookies",
      ];

      const vectors = await e.embed(texts);
      expect(vectors).toHaveLength(texts.length);
      for (const v of vectors) {
        expect(v).toBeInstanceOf(Float32Array);
        expect(v.length).toBe(384);
        // Should be unit-norm.
        let norm = 0;
        for (const x of v) norm += x * x;
        expect(Math.sqrt(norm)).toBeGreaterThan(0.95);
        expect(Math.sqrt(norm)).toBeLessThan(1.05);
      }

      // Semantic check: "auth" entries should be closer to each other than to
      // unrelated topics like "renaming variables".
      const cos = (a: Float32Array, b: Float32Array) => {
        let s = 0;
        for (let i = 0; i < a.length; i++) s += a[i]! * b[i]!;
        return s;
      };
      const v0 = vectors[0]!; // auth refactor
      const v1 = vectors[1]!; // renaming
      const v4 = vectors[4]!; // auth flow
      // auth-to-auth should beat auth-to-renaming.
      expect(cos(v0, v4)).toBeGreaterThan(cos(v0, v1));

      // Throughput probe: 32 strings in a single embed call.
      const batch = Array.from({ length: 32 }, (_, i) => `sample text number ${i}`);
      const t0 = Date.now();
      const big = await e.embed(batch);
      const ms = Date.now() - t0;
      expect(big).toHaveLength(32);
      // eslint-disable-next-line no-console
      console.log(`[live] 32 embeds in ${ms}ms (${((32 * 1000) / ms).toFixed(1)}/s) using ${state.intraOpNumThreads} threads`);
    } finally {
      e.terminate();
    }
  }, 120_000);

  it("propagates worker errors through embed()", async () => {
    const e = new Embedder({
      model: "this/model-does-not-exist-12345",
      quantize: "q8",
      cacheDir: null,
    });

    let lastError: string | null = null;
    e.setListener({
      onError: (m) => {
        lastError = m;
      },
    });

    await expect(e.warmup()).rejects.toBeDefined();
    expect(e.state().error).not.toBeNull();
    expect(lastError).not.toBeNull();
    e.terminate();
  }, 30_000);
});
