/**
 * sqlite-vec loader. Soft-fails if unavailable: the rest of the extension can
 * still operate (FTS5 fallback for retrieval, vec inserts become no-ops).
 *
 * The npm package `sqlite-vec` ships a platform-specific shared library and
 * exposes `loadablePathFor()` / `load(db)` to attach it to a better-sqlite3
 * connection. We use the dynamic import so missing native binaries don't
 * crash the extension on unsupported platforms.
 */

import type Database from "better-sqlite3";

let attempted = false;
let loaded = false;
let loadError: string | null = null;

export interface VecState {
  loaded: boolean;
  error: string | null;
}

export async function ensureVecLoaded(db: Database.Database): Promise<VecState> {
  if (attempted) return { loaded, error: loadError };
  attempted = true;

  try {
    const mod: any = await import("sqlite-vec");
    if (typeof mod?.load === "function") {
      mod.load(db);
    } else if (typeof mod?.default?.load === "function") {
      mod.default.load(db);
    } else {
      throw new Error("sqlite-vec module does not expose a `load(db)` function");
    }
    // Sanity check — if vec_version() is callable we're good.
    const row = db.prepare("SELECT vec_version() AS v").get() as { v?: string } | undefined;
    if (!row?.v) throw new Error("sqlite-vec loaded but vec_version() is unavailable");
    loaded = true;
  } catch (e: unknown) {
    loaded = false;
    loadError = e instanceof Error ? e.message : String(e);
  }
  return { loaded, error: loadError };
}

export function isVecLoaded(): boolean {
  return loaded;
}

export function vecError(): string | null {
  return loadError;
}

/** Encode a Float32Array to the BLOB shape sqlite-vec expects on insert. */
export function encodeVector(vec: Float32Array): Buffer {
  return Buffer.from(vec.buffer, vec.byteOffset, vec.byteLength);
}
