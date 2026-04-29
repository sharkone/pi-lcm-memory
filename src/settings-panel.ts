/**
 * Settings panel — TUI overlay. Mirrors pi-lcm's panel shape:
 * implements Pi's Component interface (render(width) / handleInput(data)
 * / invalidate()), opened via `ctx.ui.custom({ overlay: true, ... })`.
 *
 * Renders one row per editable setting; arrow keys navigate, +/- adjusts
 * numbers, space toggles booleans, [/] cycles enums, S saves, Q closes.
 */

import { matchesKey, Key, truncateToWidth } from "@mariozechner/pi-tui";
import type { MemoryConfig, AutoRecallMode } from "./config.js";
import type { SettingsScope } from "./settings.js";
import type { MemoryStats } from "./db/store.js";

const dim = (s: string) => `\x1b[2m${s}\x1b[22m`;
const bold = (s: string) => `\x1b[1m${s}\x1b[22m`;
const green = (s: string) => `\x1b[32m${s}\x1b[39m`;
const cyan = (s: string) => `\x1b[36m${s}\x1b[39m`;
const red = (s: string) => `\x1b[31m${s}\x1b[39m`;

interface BoolRow {
  type: "boolean";
  key: keyof MemoryConfig;
  label: string;
  description: string;
}
interface NumberRow {
  type: "number";
  key: keyof MemoryConfig;
  label: string;
  description: string;
  min: number;
  max: number;
  step: number;
}
interface EnumRow {
  type: "enum";
  key: keyof MemoryConfig;
  label: string;
  description: string;
  values: readonly string[];
}
type Row = BoolRow | NumberRow | EnumRow;

const ROWS: Row[] = [
  { type: "boolean", key: "enabled", label: "Enabled", description: "Master enable" },
  { type: "boolean", key: "indexMessages", label: "Index messages", description: "Embed user/assistant text" },
  { type: "boolean", key: "indexSummaries", label: "Index summaries", description: "Embed pi-lcm DAG summaries" },
  { type: "boolean", key: "skipToolIO", label: "Skip tool I/O", description: "Don't embed tool outputs / bash results" },
  { type: "boolean", key: "primer", label: "Session primer", description: "Brief on prior memory at session start" },
  {
    type: "number",
    key: "primerTopK",
    label: "Primer top-K",
    description: "How many recent topics in primer",
    min: 0,
    max: 20,
    step: 1,
  },
  {
    type: "enum",
    key: "autoRecall",
    label: "Auto-recall",
    description: "Heuristic: trigger on phrases. Always: every turn. Off: never.",
    values: ["off", "heuristic", "always"] as const,
  },
  {
    type: "enum",
    key: "embeddingQuantize",
    label: "Quantize",
    description: "q8 (default, 4× smaller, fast). fp32 = full precision (slow).",
    values: ["auto", "fp32", "fp16", "q8", "int8", "uint8", "q4", "q4f16"] as const,
  },
  {
    type: "number",
    key: "autoRecallTopK",
    label: "Auto-recall top-K",
    description: "Hits to inject when auto-recall fires",
    min: 0,
    max: 20,
    step: 1,
  },
  {
    type: "number",
    key: "autoRecallTokenBudget",
    label: "Auto-recall budget",
    description: "Max tokens injected by auto-recall",
    min: 100,
    max: 4000,
    step: 100,
  },
  {
    type: "number",
    key: "recallDefaultTopK",
    label: "Recall default K",
    description: "Default top-K for lcm_recall",
    min: 1,
    max: 100,
    step: 1,
  },
  {
    type: "number",
    key: "rrfK",
    label: "RRF k",
    description: "Reciprocal Rank Fusion constant (sweep-tuned default: 20)",
    min: 1,
    max: 1000,
    step: 5,
  },
  {
    type: "number",
    key: "lexMult",
    label: "FTS5 breadth",
    description: "Lexical candidate multiplier: fetch k × lexMult from FTS5",
    min: 1,
    max: 32,
    step: 1,
  },
  {
    type: "number",
    key: "semMult",
    label: "KNN breadth",
    description: "Semantic candidate multiplier: fetch k × semMult from vec index",
    min: 1,
    max: 32,
    step: 1,
  },
  {
    type: "number",
    key: "sweepIntervalMs",
    label: "Sweep interval (ms)",
    description: "How often the background indexer runs",
    min: 2000,
    max: 600000,
    step: 1000,
  },
  { type: "boolean", key: "debugMode", label: "Debug", description: "Verbose logging" },
];

const SCOPE_ROW = 0;
const FIRST_SETTING_ROW = 1;

export interface PanelDeps {
  config: MemoryConfig;
  scope: SettingsScope;
  cwd: string;
  stats: MemoryStats | null;
  save: (config: MemoryConfig, scope: SettingsScope, cwd: string) => void;
}

export class MemorySettingsPanel {
  onClose?: () => void;

  private row = 0;
  private cw: number | null = null;
  private cl: string[] | null = null;
  private deps: PanelDeps;
  private dirty = false;

  constructor(deps: PanelDeps) {
    this.deps = deps;
  }

  render(width: number): string[] {
    if (this.cl !== null && this.cw === width) return this.cl;
    const w = Math.max(36, Math.min(width, 80));
    const t = (s: string) => truncateToWidth(s, w);
    const lines: string[] = [];
    const { config, scope, stats } = this.deps;

    lines.push(t(bold("pi-lcm-memory settings")));
    lines.push(t(dim("↑↓ navigate · space/+/- toggle/edit · [ ] enum · S save · Q close")));
    lines.push("");

    const scopeText = `Scope: ${scope === "project" ? cyan("project") : cyan("global")} (P toggles)`;
    lines.push(t(this.row === SCOPE_ROW ? bold("▶ " + scopeText) : "  " + scopeText));

    ROWS.forEach((r, i) => {
      const idx = i + FIRST_SETTING_ROW;
      const cur = this.row === idx;
      let value = "";
      if (r.type === "boolean") {
        value = (config[r.key] as boolean) ? green("on") : red("off");
      } else if (r.type === "number") {
        value = String(config[r.key]);
      } else {
        value = String(config[r.key]);
      }
      const label = `${r.label}: ${value}`;
      lines.push(t(cur ? bold("▶ " + label) : "  " + label));
      lines.push(t(dim("    " + r.description)));
    });

    lines.push("");
    if (stats) {
      lines.push(
        t(
          dim(
            `indexed=${stats.indexed} msg=${stats.byMessage} sum=${stats.bySummary} model=${stats.modelName ?? "—"} dim=${stats.modelDims ?? "—"}`,
          ),
        ),
      );
    }
    if (this.dirty) lines.push(t(green("* unsaved changes — press S to save")));

    this.cl = lines;
    this.cw = width;
    return lines;
  }

  invalidate(): void {
    this.cl = null;
  }

  handleInput(data: string): void {
    const total = ROWS.length + 1;

    if (matchesKey(data, Key.up)) {
      this.row = (this.row - 1 + total) % total;
      this.invalidate();
      return;
    }
    if (matchesKey(data, Key.down)) {
      this.row = (this.row + 1) % total;
      this.invalidate();
      return;
    }
    if (data === "q" || data === "Q" || matchesKey(data, Key.escape)) {
      this.onClose?.();
      return;
    }
    if (data === "s" || data === "S") {
      this.deps.save(this.deps.config, this.deps.scope, this.deps.cwd);
      this.dirty = false;
      this.invalidate();
      return;
    }
    if (data === "p" || data === "P") {
      this.deps.scope = this.deps.scope === "project" ? "global" : "project";
      this.dirty = true;
      this.invalidate();
      return;
    }

    if (this.row >= FIRST_SETTING_ROW) {
      const r = ROWS[this.row - FIRST_SETTING_ROW]!;
      const cfg = this.deps.config as unknown as Record<string, unknown>;
      if (r.type === "boolean" && (data === " " || matchesKey(data, Key.enter))) {
        cfg[r.key] = !cfg[r.key];
        this.dirty = true;
        this.invalidate();
        return;
      }
      if (r.type === "number" && (data === "+" || matchesKey(data, Key.right))) {
        cfg[r.key] = Math.min(r.max, (cfg[r.key] as number) + r.step);
        this.dirty = true;
        this.invalidate();
        return;
      }
      if (r.type === "number" && (data === "-" || matchesKey(data, Key.left))) {
        cfg[r.key] = Math.max(r.min, (cfg[r.key] as number) - r.step);
        this.dirty = true;
        this.invalidate();
        return;
      }
      if (r.type === "enum" && (data === "[" || matchesKey(data, Key.left))) {
        cfg[r.key] = cycle(r.values, cfg[r.key] as string, -1);
        this.dirty = true;
        this.invalidate();
        return;
      }
      if (r.type === "enum" && (data === "]" || matchesKey(data, Key.right))) {
        cfg[r.key] = cycle(r.values, cfg[r.key] as string, +1);
        this.dirty = true;
        this.invalidate();
        return;
      }
    }
  }
}

function cycle(values: readonly string[], current: string, dir: 1 | -1): AutoRecallMode | string {
  const i = values.indexOf(current);
  const len = values.length;
  const next = ((i === -1 ? 0 : i) + dir + len) % len;
  return values[next] ?? current;
}
