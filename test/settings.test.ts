import { describe, it, expect, afterEach, beforeEach } from "vitest";
import { mkdtempSync, rmSync, mkdirSync, readFileSync, writeFileSync, existsSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { loadSettings, saveSettings, SETTINGS_KEY } from "../src/settings.js";

describe("settings persistence", () => {
  let cwd: string;
  beforeEach(() => {
    cwd = mkdtempSync(join(tmpdir(), "pi-lcm-mem-set-"));
  });
  afterEach(() => {
    rmSync(cwd, { recursive: true, force: true });
  });

  it("defaults when nothing exists", () => {
    const loaded = loadSettings(cwd);
    expect(loaded.source).toBe("default");
    expect(loaded.config).toEqual({});
  });

  it("project takes precedence over global", () => {
    // Cannot stub homedir easily; only test project here.
    mkdirSync(join(cwd, ".pi"), { recursive: true });
    writeFileSync(
      join(cwd, ".pi", "settings.json"),
      JSON.stringify({ [SETTINGS_KEY]: { autoRecall: "always" } }),
    );
    const loaded = loadSettings(cwd);
    expect(loaded.source).toBe("project");
    expect(loaded.config.autoRecall).toBe("always");
  });

  it("save writes atomically and merges with sibling keys", () => {
    mkdirSync(join(cwd, ".pi"), { recursive: true });
    writeFileSync(
      join(cwd, ".pi", "settings.json"),
      JSON.stringify({ lcm: { dbDir: "/tmp/lcm" }, other: "preserved" }),
    );
    saveSettings({ autoRecall: "off", primerTopK: 3 } as any, "project", cwd);
    const got = JSON.parse(readFileSync(join(cwd, ".pi", "settings.json"), "utf8"));
    expect(got.lcm.dbDir).toBe("/tmp/lcm");
    expect(got.other).toBe("preserved");
    expect(got[SETTINGS_KEY].autoRecall).toBe("off");
    expect(got[SETTINGS_KEY].primerTopK).toBe(3);

    // No leftover .tmp files.
    const tmpLeft = existsSync(join(cwd, ".pi", "settings.json." + process.pid + ".tmp"));
    expect(tmpLeft).toBe(false);
  });
});
