/**
 * Settings persistence for pi-lcm-memory.
 *
 * Reads/writes a `lcm-memory` key inside the same files pi-lcm uses
 * (`~/.pi/agent/settings.json` global, `<cwd>/.pi/settings.json` project),
 * so users only have one settings location to think about. Atomic writes via
 * temp + rename. Mirrors pi-lcm's settings.ts pattern.
 */

import * as fs from "node:fs";
import * as os from "node:os";
import * as path from "node:path";
import type { MemoryConfig } from "./config.js";

export const SETTINGS_KEY = "lcm-memory";

export type SettingsScope = "global" | "project";
export type ConfigSource = SettingsScope | "default";

export interface LoadedSettings {
  config: Partial<MemoryConfig>;
  source: ConfigSource;
  globalPath: string;
  projectPath: string;
}

function readJsonFile(filePath: string): Record<string, unknown> {
  try {
    if (!fs.existsSync(filePath)) return {};
    return JSON.parse(fs.readFileSync(filePath, "utf8")) as Record<string, unknown>;
  } catch {
    return {};
  }
}

export function getGlobalSettingsPath(): string {
  return path.join(os.homedir(), ".pi", "agent", "settings.json");
}

export function getProjectSettingsPath(cwd: string): string {
  return path.join(cwd, ".pi", "settings.json");
}

export function loadSettings(cwd: string): LoadedSettings {
  const globalPath = getGlobalSettingsPath();
  const projectPath = getProjectSettingsPath(cwd);

  const projectRaw = readJsonFile(projectPath)[SETTINGS_KEY];
  if (projectRaw && typeof projectRaw === "object") {
    return { config: projectRaw as Partial<MemoryConfig>, source: "project", globalPath, projectPath };
  }

  const globalRaw = readJsonFile(globalPath)[SETTINGS_KEY];
  if (globalRaw && typeof globalRaw === "object") {
    return { config: globalRaw as Partial<MemoryConfig>, source: "global", globalPath, projectPath };
  }

  return { config: {}, source: "default", globalPath, projectPath };
}

export function saveSettings(
  config: Partial<MemoryConfig>,
  scope: SettingsScope,
  cwd: string,
): string {
  const settingsPath = scope === "project" ? getProjectSettingsPath(cwd) : getGlobalSettingsPath();
  const settings = readJsonFile(settingsPath);
  settings[SETTINGS_KEY] = config;

  fs.mkdirSync(path.dirname(settingsPath), { recursive: true });

  const tmpPath = `${settingsPath}.${process.pid}.tmp`;
  try {
    fs.writeFileSync(tmpPath, JSON.stringify(settings, null, 2) + "\n");
    fs.renameSync(tmpPath, settingsPath);
  } finally {
    try {
      if (fs.existsSync(tmpPath)) fs.unlinkSync(tmpPath);
    } catch {
      // best effort
    }
  }
  return settingsPath;
}
