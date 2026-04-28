import { describe, it, expect, afterEach, beforeEach } from "vitest";
import { mkdtempSync, rmSync, mkdirSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { resolveConfig, DEFAULTS } from "../src/config.js";

const ENV_KEYS = [
  "PI_LCM_MEMORY_ENABLED",
  "PI_LCM_MEMORY_DB_DIR",
  "LCM_DB_DIR",
  "PI_LCM_MEMORY_MODEL",
  "PI_LCM_MEMORY_QUANTIZE",
  "PI_LCM_MEMORY_SWEEP_MS",
  "PI_LCM_MEMORY_DEBUG",
];

describe("config.resolveConfig", () => {
  let cwdDir: string;
  const stash: Record<string, string | undefined> = {};

  beforeEach(() => {
    cwdDir = mkdtempSync(join(tmpdir(), "pi-lcm-mem-cfg-"));
    for (const k of ENV_KEYS) {
      stash[k] = process.env[k];
      delete process.env[k];
    }
  });
  afterEach(() => {
    rmSync(cwdDir, { recursive: true, force: true });
    for (const k of ENV_KEYS) {
      if (stash[k] === undefined) delete process.env[k];
      else process.env[k] = stash[k];
    }
  });

  it("returns defaults when nothing is set", () => {
    const cfg = resolveConfig({ cwd: cwdDir });
    expect(cfg.enabled).toBe(true);
    expect(cfg.embeddingModel).toBe(DEFAULTS.embeddingModel);
    expect(cfg.autoRecall).toBe("heuristic");
    expect(cfg.rrfK).toBe(60);
  });

  it("env overrides apply", () => {
    const customDir = mkdtempSync(join(tmpdir(), "pi-lcm-mem-db-"));
    process.env.PI_LCM_MEMORY_DB_DIR = customDir;
    process.env.PI_LCM_MEMORY_MODEL = "Xenova/all-MiniLM-L6-v2";
    process.env.PI_LCM_MEMORY_DEBUG = "true";
    const cfg = resolveConfig({ cwd: cwdDir });
    expect(cfg.dbDir).toBe(customDir);
    expect(cfg.embeddingModel).toBe("Xenova/all-MiniLM-L6-v2");
    expect(cfg.debugMode).toBe(true);
    rmSync(customDir, { recursive: true, force: true });
  });

  it("project settings override global; pi-lcm key controls dbDir", () => {
    const projDir = join(cwdDir, ".pi");
    mkdirSync(projDir, { recursive: true });
    const dbDir = mkdtempSync(join(tmpdir(), "pi-lcm-mem-db-"));
    writeFileSync(
      join(projDir, "settings.json"),
      JSON.stringify({
        lcm: { dbDir },
        "lcm-memory": { autoRecall: "off", primerTopK: 9 },
      }),
    );
    const cfg = resolveConfig({ cwd: cwdDir });
    expect(cfg.dbDir).toBe(dbDir);
    expect(cfg.autoRecall).toBe("off");
    expect(cfg.primerTopK).toBe(9);
    rmSync(dbDir, { recursive: true, force: true });
  });

  it("disabled if pi-lcm is disabled", () => {
    const projDir = join(cwdDir, ".pi");
    mkdirSync(projDir, { recursive: true });
    writeFileSync(join(projDir, "settings.json"), JSON.stringify({ lcm: { enabled: false } }));
    const cfg = resolveConfig({ cwd: cwdDir });
    expect(cfg.enabled).toBe(false);
  });

  it("normalizes dbDir away from '..' segments", () => {
    process.env.PI_LCM_MEMORY_DB_DIR = "/foo/../etc";
    const cfg = resolveConfig({ cwd: cwdDir });
    expect(cfg.dbDir).toBe("/etc");
    expect(cfg.dbDir.split(/[\\/]/)).not.toContain("..");
  });

  it("clamps numeric values", () => {
    const projDir = join(cwdDir, ".pi");
    mkdirSync(projDir, { recursive: true });
    writeFileSync(
      join(projDir, "settings.json"),
      JSON.stringify({ "lcm-memory": { primerTopK: 9999, sweepIntervalMs: 1 } }),
    );
    const cfg = resolveConfig({ cwd: cwdDir });
    expect(cfg.primerTopK).toBe(20);
    expect(cfg.sweepIntervalMs).toBe(2_000);
  });
});
