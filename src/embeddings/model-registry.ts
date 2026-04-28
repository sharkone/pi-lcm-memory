/**
 * Known embedding models. The registry is mostly informational (we accept any
 * Transformers.js feature-extraction model by name); it lets us resolve dims
 * before the model is downloaded for nicer config/UI.
 */

export interface ModelInfo {
  name: string;
  repo: string;
  dims: number;
  description: string;
  defaultPooling: "mean" | "cls";
  defaultNormalize: boolean;
}

export const REGISTRY: ReadonlyArray<ModelInfo> = [
  {
    name: "Xenova/bge-small-en-v1.5",
    repo: "Xenova/bge-small-en-v1.5",
    dims: 384,
    description: "BGE small. Strong English retrieval; small footprint.",
    defaultPooling: "mean",
    defaultNormalize: true,
  },
  {
    name: "Xenova/all-MiniLM-L6-v2",
    repo: "Xenova/all-MiniLM-L6-v2",
    dims: 384,
    description: "MiniLM L6. Battle-tested baseline.",
    defaultPooling: "mean",
    defaultNormalize: true,
  },
  {
    name: "Xenova/gte-small",
    repo: "Xenova/gte-small",
    dims: 384,
    description: "GTE small. Comparable to bge-small.",
    defaultPooling: "mean",
    defaultNormalize: true,
  },
  {
    name: "TaylorAI/bge-micro-v2",
    repo: "TaylorAI/bge-micro-v2",
    dims: 384,
    description: "Tiny BGE. Lower quality but ~5MB.",
    defaultPooling: "mean",
    defaultNormalize: true,
  },
  {
    name: "Xenova/nomic-embed-text-v1.5",
    repo: "Xenova/nomic-embed-text-v1.5",
    dims: 768,
    description: "Nomic v1.5. Higher quality; supports Matryoshka truncation.",
    defaultPooling: "mean",
    defaultNormalize: true,
  },
];

export function lookupModel(name: string): ModelInfo | undefined {
  return REGISTRY.find((m) => m.name === name);
}

export function listModelNames(): string[] {
  return REGISTRY.map((m) => m.name);
}
