/**
 * Side-channel synchronous tracer. Writes timestamped JSON lines to a file
 * outside the SQLite path, so we can see what the indexer/embedder were
 * doing right up to the moment the main thread froze.
 *
 * Writes are best-effort: an exception in trace() must never break the
 * caller. The file is opened lazily and reused.
 *
 * Two ways to enable:
 *
 *   1. Env var: PI_LCM_MEMORY_TRACE=1 pi --continue
 *      (writes to /tmp/pi-lcm-memory.<pid>.trace.log)
 *
 *   2. Env var with explicit path: PI_LCM_MEMORY_TRACE=/tmp/my.log
 */

import * as fs from "node:fs";
import * as path from "node:path";
import * as os from "node:os";

let traceFd: number | null = null;
let tracePath: string | null = null;
let initialized = false;

function init(): void {
  if (initialized) return;
  initialized = true;

  const env = process.env.PI_LCM_MEMORY_TRACE;
  if (!env || env === "0" || env === "false") return;

  const file =
    env === "1" || env === "true"
      ? path.join(os.tmpdir(), `pi-lcm-memory.${process.pid}.trace.log`)
      : env;
  try {
    fs.mkdirSync(path.dirname(file), { recursive: true });
    traceFd = fs.openSync(file, "a");
    tracePath = file;
    // Header line so we can spot session boundaries when tailing.
    write({ event: "trace_open", pid: process.pid, file });
  } catch {
    traceFd = null;
    tracePath = null;
  }
}

function write(obj: Record<string, unknown>): void {
  if (traceFd == null) return;
  try {
    const line = JSON.stringify({ t: Date.now(), pid: process.pid, ...obj }) + "\n";
    fs.writeSync(traceFd, line);
  } catch {
    // ignore
  }
}

/** Record a single trace event. Cheap when tracing is disabled. */
export function trace(event: string, data?: Record<string, unknown>): void {
  if (!initialized) init();
  if (traceFd == null) return;
  write({ event, ...(data ?? {}) });
}

/** Path the tracer is writing to (or null if disabled). */
export function traceFile(): string | null {
  if (!initialized) init();
  return tracePath;
}

/** True if PI_LCM_MEMORY_TRACE is set. */
export function isTracing(): boolean {
  if (!initialized) init();
  return traceFd != null;
}
