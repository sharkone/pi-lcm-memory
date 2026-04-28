import { describe, it, expect, afterEach } from "vitest";
import { makeTestDb, setupVecAndMigrate, type TestDb } from "./helpers.js";
import { Diagnostics } from "../src/diagnostics.js";

describe("Diagnostics", () => {
  let t: TestDb | null = null;
  afterEach(() => {
    t?.cleanup();
    t = null;
  });

  it("logs and reads recent events", async () => {
    t = makeTestDb();
    await setupVecAndMigrate(t.db, 8);
    const d = new Diagnostics(t.db);
    d.log("a");
    d.log("b", { n: 1 });
    d.log("c", { x: "y" });
    const recent = d.recent();
    expect(recent.map((e) => e.event)).toEqual(["a", "b", "c"]);
    expect(recent[1]!.data).toEqual({ n: 1 });
    expect(recent[0]!.ts).toBeGreaterThan(0);
  });

  it("rotates after MAX_EVENTS", async () => {
    t = makeTestDb();
    await setupVecAndMigrate(t.db, 8);
    const d = new Diagnostics(t.db);
    for (let i = 0; i < 250; i++) d.log("e", { i });
    const r = d.recent(300);
    expect(r.length).toBeLessThanOrEqual(200);
    // oldest preserved should be relatively recent (50+).
    expect((r[0]!.data as { i: number }).i).toBeGreaterThanOrEqual(50);
    expect((r[r.length - 1]!.data as { i: number }).i).toBe(249);
  });

  it("clear empties the log", async () => {
    t = makeTestDb();
    await setupVecAndMigrate(t.db, 8);
    const d = new Diagnostics(t.db);
    d.log("x");
    expect(d.recent().length).toBe(1);
    d.clear();
    expect(d.recent().length).toBe(0);
  });

  it("never throws if memory_meta is missing", () => {
    t = makeTestDb();
    // No migrations run — memory_meta does not exist.
    const d = new Diagnostics(t.db);
    expect(() => d.log("x")).not.toThrow();
    expect(d.recent()).toEqual([]);
  });
});
