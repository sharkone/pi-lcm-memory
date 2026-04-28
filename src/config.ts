/**
 * Configuration resolution: env > project settings > global settings > defaults.
 *
 * `dbDir` mirrors pi-lcm's resolution so that we open the same per-cwd SQLite
 * file. We do NOT own pi-lcm's settings — we read them through the same path
 * convention. If pi-lcm's settings change in the future, our default still
 * works because we mirror the *defaults*, not internal state.
 */

import { homedir } from "node:os";
import { join, normalize, resolve } from "node:path";
import { readFileSync, existsSync } from "node:fs";

export type AutoRecallMode = "off" | "heuristic" | "always";

/**
 * Embedding dtype. Names mirror @huggingface/transformers v3 DataType.
 * On Node CPU, default "q8" gives ~4× smaller weights and ~2-4× faster
 * inference vs fp32, with negligible quality loss for retrieval.
 */
export type EmbeddingDtype = "auto" | "fp32" | "fp16" | "q8" | "int8" | "uint8" | "q4" | "q4f16";

export interface MemoryConfig {
  enabled: boolean;
  dbDir: string;
  embeddingModel: string;
  embeddingQuantize: EmbeddingDtype;
  indexMessages: boolean;
  indexSummaries: boolean;
  skipToolIO: boolean;
  primer: boolean;
  primerTopK: number;
  autoRecall: AutoRecallMode;
  autoRecallTopK: number;
  autoRecallTokenBudget: number;
  recallDefaultTopK: number;
  rrfK: number;
  sweepIntervalMs: number;
  modelCacheDir: string | null;
  debugMode: boolean;
  /** Enable cross-encoder reranker as a second stage on top of hybrid recall. */
  rerank: boolean;
  /** Cross-encoder model id (must accept text-pair input). */
  rerankModel: string;
  /** Quantization for the reranker. Same dtype enum as embeddings. */
  rerankQuantize: EmbeddingDtype;
  /** How many hybrid candidates to fetch BEFORE reranking down to top-K. */
  rerankPoolSize: number;
}

export const DEFAULTS: MemoryConfig = {
  enabled: true,
  // Mirror pi-lcm's default dbDir so we open the same file.
  dbDir: join(homedir(), ".pi", "agent", "lcm"),
  embeddingModel: "Xenova/bge-small-en-v1.5",
  // q8 = quantized variant of Xenova models (model_quantized.onnx). Much
  // faster on CPU than fp32; available for the vast majority of Xenova
  // feature-extraction models including bge-small/MiniLM/gte-small.
  embeddingQuantize: "q8",
  indexMessages: true,
  indexSummaries: true,
  skipToolIO: true,
  primer: true,
  primerTopK: 5,
  autoRecall: "heuristic",
  autoRecallTopK: 5,
  autoRecallTokenBudget: 600,
  recallDefaultTopK: 10,
  rrfK: 60,
  sweepIntervalMs: 30_000,
  modelCacheDir: null,
  debugMode: false,
  rerank: false,
  rerankModel: "Xenova/ms-marco-MiniLM-L-6-v2",
  rerankQuantize: "q8",
  rerankPoolSize: 30,
};

const SETTINGS_KEY = "lcm-memory";
// pi-lcm reuses this key in the same files; we read it to mirror dbDir/enabled.
const PI_LCM_KEY = "lcm";

function readJsonFile(filePath: string): Record<string, unknown> {
  try {
    if (!existsSync(filePath)) return {};
    return JSON.parse(readFileSync(filePath, "utf8")) as Record<string, unknown>;
  } catch {
    return {};
  }
}

function getGlobalSettingsPath(): string {
  return join(homedir(), ".pi", "agent", "settings.json");
}

function getProjectSettingsPath(cwd: string): string {
  return join(cwd, ".pi", "settings.json");
}

function envBool(name: string): boolean | undefined {
  const v = process.env[name];
  if (v === undefined) return undefined;
  return v === "1" || v.toLowerCase() === "true";
}

function envInt(name: string): number | undefined {
  const v = process.env[name];
  if (v === undefined) return undefined;
  const n = parseInt(v, 10);
  return Number.isNaN(n) ? undefined : n;
}

function envStr(name: string): string | undefined {
  return process.env[name];
}

/** Validate dbDir doesn't allow path traversal. Mirrors pi-lcm's check. */
function validateDbDir(dir: string): string {
  const resolved = resolve(normalize(dir));
  if (resolved.split(/[\\/]/).includes("..")) {
    throw new Error(`PI_LCM_MEMORY_DB_DIR must not contain '..': ${dir}`);
  }
  return resolved;
}

export interface ResolveContext {
  cwd?: string;
}

export function resolveConfig(ctx: ResolveContext = {}): MemoryConfig {
  const cwd = ctx.cwd ?? process.cwd();
  const project = readJsonFile(getProjectSettingsPath(cwd));
  const global = readJsonFile(getGlobalSettingsPath());

  const projectMem = (project[SETTINGS_KEY] as Partial<MemoryConfig> | undefined) ?? {};
  const globalMem = (global[SETTINGS_KEY] as Partial<MemoryConfig> | undefined) ?? {};
  const projectLcm = (project[PI_LCM_KEY] as { dbDir?: string; enabled?: boolean } | undefined) ?? {};
  const globalLcm = (global[PI_LCM_KEY] as { dbDir?: string; enabled?: boolean } | undefined) ?? {};

  // dbDir prefers our own override, then pi-lcm's override (so we follow pi-lcm wherever it goes).
  const dbDirRaw =
    envStr("PI_LCM_MEMORY_DB_DIR") ??
    envStr("LCM_DB_DIR") ??
    projectMem.dbDir ??
    projectLcm.dbDir ??
    globalMem.dbDir ??
    globalLcm.dbDir ??
    DEFAULTS.dbDir;

  const cfg: MemoryConfig = {
    enabled: envBool("PI_LCM_MEMORY_ENABLED") ?? projectMem.enabled ?? globalMem.enabled ?? DEFAULTS.enabled,
    dbDir: validateDbDir(dbDirRaw),
    embeddingModel:
      envStr("PI_LCM_MEMORY_MODEL") ??
      projectMem.embeddingModel ??
      globalMem.embeddingModel ??
      DEFAULTS.embeddingModel,
    embeddingQuantize:
      (envStr("PI_LCM_MEMORY_QUANTIZE") as EmbeddingDtype | undefined) ??
      projectMem.embeddingQuantize ??
      globalMem.embeddingQuantize ??
      DEFAULTS.embeddingQuantize,
    indexMessages: projectMem.indexMessages ?? globalMem.indexMessages ?? DEFAULTS.indexMessages,
    indexSummaries: projectMem.indexSummaries ?? globalMem.indexSummaries ?? DEFAULTS.indexSummaries,
    skipToolIO: projectMem.skipToolIO ?? globalMem.skipToolIO ?? DEFAULTS.skipToolIO,
    primer: projectMem.primer ?? globalMem.primer ?? DEFAULTS.primer,
    primerTopK: clamp(projectMem.primerTopK ?? globalMem.primerTopK ?? DEFAULTS.primerTopK, 0, 20),
    autoRecall:
      (projectMem.autoRecall as AutoRecallMode | undefined) ??
      (globalMem.autoRecall as AutoRecallMode | undefined) ??
      DEFAULTS.autoRecall,
    autoRecallTopK: clamp(projectMem.autoRecallTopK ?? globalMem.autoRecallTopK ?? DEFAULTS.autoRecallTopK, 0, 20),
    autoRecallTokenBudget: clamp(
      projectMem.autoRecallTokenBudget ?? globalMem.autoRecallTokenBudget ?? DEFAULTS.autoRecallTokenBudget,
      100,
      4000,
    ),
    recallDefaultTopK: clamp(
      projectMem.recallDefaultTopK ?? globalMem.recallDefaultTopK ?? DEFAULTS.recallDefaultTopK,
      1,
      100,
    ),
    rrfK: clamp(projectMem.rrfK ?? globalMem.rrfK ?? DEFAULTS.rrfK, 1, 1000),
    sweepIntervalMs: clamp(
      envInt("PI_LCM_MEMORY_SWEEP_MS") ?? projectMem.sweepIntervalMs ?? globalMem.sweepIntervalMs ?? DEFAULTS.sweepIntervalMs,
      2_000,
      600_000,
    ),
    modelCacheDir: projectMem.modelCacheDir ?? globalMem.modelCacheDir ?? DEFAULTS.modelCacheDir,
    debugMode: envBool("PI_LCM_MEMORY_DEBUG") ?? projectMem.debugMode ?? globalMem.debugMode ?? DEFAULTS.debugMode,
    rerank: envBool("PI_LCM_MEMORY_RERANK") ?? projectMem.rerank ?? globalMem.rerank ?? DEFAULTS.rerank,
    rerankModel:
      envStr("PI_LCM_MEMORY_RERANK_MODEL") ??
      projectMem.rerankModel ??
      globalMem.rerankModel ??
      DEFAULTS.rerankModel,
    rerankQuantize:
      (envStr("PI_LCM_MEMORY_RERANK_QUANTIZE") as EmbeddingDtype | undefined) ??
      projectMem.rerankQuantize ??
      globalMem.rerankQuantize ??
      DEFAULTS.rerankQuantize,
    rerankPoolSize: clamp(
      envInt("PI_LCM_MEMORY_RERANK_POOL") ?? projectMem.rerankPoolSize ?? globalMem.rerankPoolSize ?? DEFAULTS.rerankPoolSize,
      1,
      200,
    ),
  };

  // pi-lcm disabled? We follow suit: indexing pi-lcm data we don't have is moot.
  const piLcmEnabled = projectLcm.enabled ?? globalLcm.enabled ?? true;
  if (!piLcmEnabled) cfg.enabled = false;

  return cfg;
}

function clamp(n: number, lo: number, hi: number): number {
  return Math.max(lo, Math.min(hi, n));
}
