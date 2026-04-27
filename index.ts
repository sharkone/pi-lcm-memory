/**
 * pi-lcm-memory: persistent cross-session semantic memory for Pi.
 *
 * Phase 0 scaffold — registers no tools yet. The extension only signals it
 * loaded successfully on session_start. Subsequent phases fill this in.
 *
 * See PLAN.md for the locked design and ROADMAP.md for delivery phases.
 */

import type { ExtensionAPI } from "@mariozechner/pi-coding-agent";

export default function (pi: ExtensionAPI) {
  pi.on("session_start", async (_event: any, ctx: any) => {
    try {
      ctx.ui?.notify?.("[pi-lcm-memory] enabled (v0 scaffold)", "info");
    } catch {
      // notify is optional in some Pi runtimes; silent fall-through.
    }
  });
}
