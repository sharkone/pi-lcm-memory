/**
 * `makeFakePi()` — minimal but faithful `ExtensionAPI` surface for e2e tests.
 *
 * Mirrors the contract our extension actually exercises:
 *   pi.on(event, handler)      record + dispatch
 *   pi.registerTool(tool)
 *   pi.registerCommand(name, def)
 *   pi.appendEntry(...)
 *   pi.registerProvider(...)
 *
 * `ctx` surface:
 *   ctx.cwd
 *   ctx.ui.notify(msg, level)
 *   ctx.ui.setStatus(s)
 *   ctx.ui.custom(factory, options)
 *
 * Tests use:
 *   const pi = makeFakePi();
 *   ext(pi);                                        // wires the extension
 *   await pi.fire("session_start", { reason: "resume" }, pi.makeCtx({ cwd }));
 *   const tool = pi.tool("lcm_recall");
 *   const hits = await tool.execute({ query: "..." });
 */

export interface RecordedNotify {
  level: "info" | "warning" | "error";
  message: string;
}

export interface RecordedStatus {
  text: string;
}

export interface FakePiUi {
  notify: (message: string, level?: "info" | "warning" | "error") => void;
  setStatus: (text: string) => void;
  custom: (factory: (...args: any[]) => any, options?: any) => Promise<unknown>;
  /** Recorded notifications since last reset. */
  notifications: RecordedNotify[];
  /** Recorded status updates since last reset. */
  statuses: RecordedStatus[];
  /** Last factory + options passed to ui.custom (so tests can inspect / drive). */
  lastCustomCall: { factory: (...args: any[]) => any; options: any } | null;
}

export interface FakeCtx {
  cwd: string;
  ui: FakePiUi;
}

export interface FakePi {
  on: (event: string, handler: (event: any, ctx: any) => any) => void;
  fire: (event: string, eventPayload: any, ctx: FakeCtx) => Promise<void>;
  registerTool: (tool: any) => void;
  registerCommand: (name: string, def: any) => void;
  registerProvider: (...args: any[]) => void;
  appendEntry: (...args: any[]) => void;
  /** Look up a registered tool by `name` (matches the tool's exported `name` field). */
  tool: (name: string) => any | undefined;
  /** Look up a registered command by name (the leading slash is omitted). */
  command: (name: string) => any | undefined;
  /** Run a registered command with the given args + ctx. */
  runCommand: (name: string, args: string | undefined, ctx: FakeCtx) => Promise<void>;
  /** Build a fake ctx with a fresh ui surface. */
  makeCtx: (init: { cwd: string }) => FakeCtx;
  /** Reset all recorded state (handlers, tools, commands stay). */
  reset: () => void;
}

export function makeFakePi(): FakePi {
  const handlers = new Map<string, ((event: any, ctx: any) => any)[]>();
  const tools = new Map<string, any>();
  const commands = new Map<string, any>();

  const pi: FakePi = {
    on(event, handler) {
      const arr = handlers.get(event) ?? [];
      arr.push(handler);
      handlers.set(event, arr);
    },
    async fire(event, payload, ctx) {
      const arr = handlers.get(event) ?? [];
      for (const h of arr) {
        // Match production: each hook awaited in registration order.
        await h(payload, ctx);
      }
    },
    registerTool(tool) {
      const name = (tool && (tool.name ?? tool.label ?? tool.id)) as string | undefined;
      if (!name) throw new Error("registerTool: tool has no `name` field");
      tools.set(name, tool);
    },
    registerCommand(name, def) {
      commands.set(name, def);
    },
    registerProvider(..._args) {
      // noop
    },
    appendEntry(..._args) {
      // noop
    },
    tool(name) {
      return tools.get(name);
    },
    command(name) {
      return commands.get(name);
    },
    async runCommand(name, args, ctx) {
      const cmd = commands.get(name);
      if (!cmd) throw new Error(`unknown command: ${name}`);
      await cmd.handler(args, ctx);
    },
    makeCtx(init) {
      const ui: FakePiUi = {
        notifications: [],
        statuses: [],
        lastCustomCall: null,
        notify(message, level) {
          ui.notifications.push({ message, level: (level ?? "info") as RecordedNotify["level"] });
        },
        setStatus(text) {
          ui.statuses.push({ text });
        },
        async custom(factory, options) {
          ui.lastCustomCall = { factory, options };
          // Drive the factory exactly like pi would: pass tui/theme/keybindings
          // stubs and a `done` callback that resolves the returned promise.
          return await new Promise<unknown>((resolve) => {
            const done = (result?: unknown) => resolve(result);
            // pi passes (tui, theme, keybindings, done); we pass nullish stubs
            // since our panel only reads `done`.
            try {
              const maybeComponent = factory(null, null, null, done);
              // The factory may return a Component (synchronous). We don't
              // render it — we just confirm it constructed. Tests may inspect
              // `lastCustomCall.factory` themselves to build it manually.
              void maybeComponent;
            } catch (e) {
              // Surface construction errors as a rejected promise so tests
              // see a clean failure rather than an unhandled rejection.
              resolve(Promise.reject(e));
            }
          });
        },
      };
      return { cwd: init.cwd, ui };
    },
    reset() {
      handlers.clear();
      tools.clear();
      commands.clear();
    },
  };
  return pi;
}
