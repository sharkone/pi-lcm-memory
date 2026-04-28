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

/**
 * sqlite-vec loads its extension into a SPECIFIC db connection. We must
 * track which connections have it loaded — module-level singletons would
 * lie when callers open multiple DBs (tests, migrations, multi-cwd hosts).
 */
const loadedDbs = new WeakSet<Database.Database>();
let anyLoaded = false;
let lastError: string | null = null;

export interface VecState {
  loaded: boolean;
  error: string | null;
}

export async function ensureVecLoaded(db: Database.Database): Promise<VecState> {
  if (loadedDbs.has(db)) return { loaded: true, error: null };

  try {
    const mod: any = await import("sqlite-vec");
    if (typeof mod?.load === "function") {
      mod.load(db);
    } else if (typeof mod?.default?.load === "function") {
      mod.default.load(db);
    } else {
      throw new Error("sqlite-vec module does not expose a `load(db)` function");
    }
    const row = db.prepare("SELECT vec_version() AS v").get() as { v?: string } | undefined;
    if (!row?.v) throw new Error("sqlite-vec loaded but vec_version() is unavailable");
    loadedDbs.add(db);
    anyLoaded = true;
    lastError = null;
    return { loaded: true, error: null };
  } catch (e: unknown) {
    lastError = e instanceof Error ? e.message : String(e);
    return { loaded: false, error: lastError };
  }
}

export function isVecLoadedFor(db: Database.Database): boolean {
  return loadedDbs.has(db);
}

/**
 * Process-wide "any vec available" flag. Useful for general capability
 * checks where the caller doesn't have a db reference handy. Per-db
 * checks should use isVecLoadedFor().
 */
export function isVecLoaded(): boolean {
  return anyLoaded;
}

export function vecError(): string | null {
  return lastError;
}

/** Encode a Float32Array to the BLOB shape sqlite-vec expects on insert. */
export function encodeVector(vec: Float32Array): Buffer {
  return Buffer.from(vec.buffer, vec.byteOffset, vec.byteLength);
}
