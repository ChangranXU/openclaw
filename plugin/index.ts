import type {
  DiagnosticEventPayload,
  OpenClawPluginApi,
  OpenClawPluginService,
  PluginHookAgentContext,
  PluginHookBeforeAgentStartEvent,
  PluginHookMessageContext,
  PluginHookMessageReceivedEvent,
  PluginHookMessageSendingEvent,
  PluginHookMessageSentEvent,
  PluginHookToolContext,
  PluginHookBeforeToolCallEvent,
  PluginHookAfterToolCallEvent,
  PluginHookToolResultPersistContext,
  PluginHookToolResultPersistEvent,
  PluginHookSessionContext,
  PluginHookSessionStartEvent,
  PluginHookSessionEndEvent,
  PluginHookGatewayContext,
  PluginHookGatewayStartEvent,
  PluginHookGatewayStopEvent,
} from "openclaw/plugin-sdk";
import { emptyPluginConfigSchema, onDiagnosticEvent } from "openclaw/plugin-sdk";
import { spawn, type ChildProcessWithoutNullStreams } from "node:child_process";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";

type JsonObject = Record<string, unknown>;

const PLUGIN_ID = "o-observability";

let pythonProc: ChildProcessWithoutNullStreams | null = null;
let pythonProcStateDir: string | null = null;
let activeStateDir: string | null = null;
let activeIpcFilePath: string | null = null;

function safeJsonStringify(value: unknown): string {
  const seen = new WeakSet<object>();
  return JSON.stringify(value, (_key, val) => {
    if (typeof val === "bigint") {
      return val.toString();
    }
    if (val && typeof val === "object") {
      if (seen.has(val as object)) {
        return "[circular]";
      }
      seen.add(val as object);
    }
    if (typeof val === "string" && val.length > 2_000_000) {
      return `${val.slice(0, 2_000_000)}...[truncated]`;
    }
    return val;
  });
}

function ensureDir(dir: string) {
  try {
    fs.mkdirSync(dir, { recursive: true });
  } catch {
    // ignore
  }
}

function resolveOpenClawConfigDir(): string {
  const raw = process.env.OPENCLAW_CONFIG_DIR?.trim();
  if (raw) {
    // In Docker (linux), OPENCLAW_CONFIG_DIR can accidentally be set to a host
    // path (e.g. /Users/<name>/.openclaw). That path doesn't exist in the
    // container and will cause the python side to try writing under /Users.
    // Prefer the container homedir fallback when the provided path is invalid.
    try {
      if (fs.existsSync(raw)) {
        return raw;
      }
    } catch {
      // ignore
    }
  }
  return path.join(os.homedir(), ".openclaw");
}

function resolvePluginRoot(sourcePath: string): string {
  return path.dirname(sourcePath);
}

function resolveIpcFilePath(stateDir: string): string {
  return path.join(stateDir, "o_observability", "events.jsonl");
}

function appendJsonl(filePath: string, obj: JsonObject) {
  try {
    ensureDir(path.dirname(filePath));
    fs.appendFileSync(filePath, `${safeJsonStringify(obj)}\n`, { encoding: "utf8" });
  } catch {
    // ignore
  }
}

function startPythonDaemon(params: {
  logger: { info: (m: string) => void; warn: (m: string) => void; error: (m: string) => void };
  pluginRoot: string;
  stateDir: string;
  ipcFilePath: string;
}) {
  if (pythonProc && pythonProcStateDir === params.stateDir) {
    return;
  }

  if (pythonProc) {
    try {
      pythonProc.kill("SIGTERM");
    } catch {
      // ignore
    }
    pythonProc = null;
    pythonProcStateDir = null;
  }

  const python = process.env.OPENCLAW_PYTHON_BIN?.trim() || "python3";
  const entry = path.join(params.pluginRoot, "main.py");
  if (!fs.existsSync(entry)) {
    params.logger.warn(`[${PLUGIN_ID}] python entry not found: ${entry}`);
    return;
  }

  const env = {
    ...process.env,
    PYTHONUNBUFFERED: "1",
    OPENCLAW_CONFIG_DIR: resolveOpenClawConfigDir(),
    OPENCLAW_STATE_DIR: params.stateDir,
    O_OBSERVABILITY_PLUGIN_DIR: params.pluginRoot,
    O_OBSERVABILITY_IPC_FILE: params.ipcFilePath,
  };

  try {
    pythonProc = spawn(python, ["-u", entry, "--ipc", params.ipcFilePath], {
      cwd: params.pluginRoot,
      env,
      stdio: "pipe",
    });
    pythonProcStateDir = params.stateDir;

    pythonProc.stdout.on("data", (buf) => {
      const text = String(buf ?? "").trim();
      if (text) {
        params.logger.info(`[${PLUGIN_ID}] py: ${text}`);
      }
    });
    pythonProc.stderr.on("data", (buf) => {
      const text = String(buf ?? "").trim();
      if (text) {
        params.logger.warn(`[${PLUGIN_ID}] py(err): ${text}`);
      }
    });
    pythonProc.on("exit", (code, signal) => {
      params.logger.warn(`[${PLUGIN_ID}] python exited: code=${String(code)} signal=${String(signal)}`);
      pythonProc = null;
      pythonProcStateDir = null;
    });
  } catch (err) {
    params.logger.warn(`[${PLUGIN_ID}] failed to start python: ${String(err)}`);
    pythonProc = null;
    pythonProcStateDir = null;
  }
}

function stopPythonDaemon(logger: { warn: (m: string) => void }) {
  if (!pythonProc) {
    return;
  }
  try {
    pythonProc.kill("SIGTERM");
  } catch (err) {
    logger.warn(`[${PLUGIN_ID}] failed to stop python: ${String(err)}`);
  } finally {
    pythonProc = null;
    pythonProcStateDir = null;
  }
}

function baseEnvelope(params: {
  kind: string;
  stateDir?: string;
  source?: string;
  context?: unknown;
  payload?: unknown;
}): JsonObject {
  return {
    kind: params.kind,
    ts: Date.now(),
    source: params.source,
    stateDir: params.stateDir,
    openclawConfigDir: resolveOpenClawConfigDir(),
    context: params.context,
    payload: params.payload,
  };
}

function createObservabilityService(api: OpenClawPluginApi): OpenClawPluginService {
  let unsubscribe: (() => void) | null = null;
  let ipcFilePath: string | null = null;
  let stateDir: string | null = null;

  return {
    id: "o-observability-daemon",
    async start(ctx) {
      stateDir = ctx.stateDir;
      ipcFilePath = resolveIpcFilePath(ctx.stateDir);
      activeStateDir = ctx.stateDir;
      activeIpcFilePath = ipcFilePath;

      // Ensure file exists so tailers can open immediately.
      ensureDir(path.dirname(ipcFilePath));
      try {
        fs.closeSync(fs.openSync(ipcFilePath, "a"));
      } catch {
        // ignore
      }

      startPythonDaemon({
        logger: ctx.logger,
        pluginRoot: resolvePluginRoot(api.source),
        stateDir: ctx.stateDir,
        ipcFilePath,
      });

      unsubscribe = onDiagnosticEvent((evt: DiagnosticEventPayload) => {
        appendJsonl(
          ipcFilePath!,
          baseEnvelope({
            kind: "diagnostic",
            stateDir: ctx.stateDir,
            source: api.source,
            payload: evt as unknown as JsonObject,
          }),
        );
      });
    },
    async stop(ctx) {
      unsubscribe?.();
      unsubscribe = null;
      stopPythonDaemon(ctx.logger);
      activeStateDir = null;
      activeIpcFilePath = null;

      // Best-effort: mark stop in the event stream.
      if (ipcFilePath && stateDir) {
        appendJsonl(
          ipcFilePath,
          baseEnvelope({
            kind: "service.stop",
            stateDir,
            source: api.source,
            payload: { id: "o-observability-daemon" },
          }),
        );
      }
    },
  };
}

const plugin = {
  id: PLUGIN_ID,
  name: "Observability (logs + langfuse via python)",
  description: "Intercept OpenClaw comms + diagnostics, export to logs and Langfuse",
  configSchema: emptyPluginConfigSchema(),
  register(api: OpenClawPluginApi) {
    api.registerService(createObservabilityService(api));

    const hookWrite = (hookName: string, ctx: unknown, event: unknown) => {
      const stateDir = activeStateDir;
      const ipcFilePath = activeIpcFilePath;
      if (!stateDir || !ipcFilePath) {
        return;
      }
      appendJsonl(
        ipcFilePath,
        baseEnvelope({
          kind: `hook.${hookName}`,
          stateDir,
          source: api.source,
          context: ctx,
          payload: event,
        }),
      );
    };

    api.on("before_agent_start", (event: PluginHookBeforeAgentStartEvent, ctx: PluginHookAgentContext) => {
      hookWrite("before_agent_start", ctx, event);
    });

    api.on("agent_end", (event, ctx) => {
      hookWrite("agent_end", ctx, event);
    });

    api.on("message_received", (event: PluginHookMessageReceivedEvent, ctx: PluginHookMessageContext) => {
      hookWrite("message_received", ctx, event);
    });

    api.on("message_sending", (event: PluginHookMessageSendingEvent, ctx: PluginHookMessageContext) => {
      hookWrite("message_sending", ctx, event);
      return {};
    });

    api.on("message_sent", (event: PluginHookMessageSentEvent, ctx: PluginHookMessageContext) => {
      hookWrite("message_sent", ctx, event);
    });

    api.on("before_tool_call", (event: PluginHookBeforeToolCallEvent, ctx: PluginHookToolContext) => {
      hookWrite("before_tool_call", ctx, event);
      return {};
    });

    api.on("after_tool_call", (event: PluginHookAfterToolCallEvent, ctx: PluginHookToolContext) => {
      hookWrite("after_tool_call", ctx, event);
    });

    api.on(
      "tool_result_persist",
      (event: PluginHookToolResultPersistEvent, ctx: PluginHookToolResultPersistContext) => {
        hookWrite("tool_result_persist", ctx, event);
      },
    );

    api.on("session_start", (event: PluginHookSessionStartEvent, ctx: PluginHookSessionContext) => {
      hookWrite("session_start", ctx, event);
    });

    api.on("session_end", (event: PluginHookSessionEndEvent, ctx: PluginHookSessionContext) => {
      hookWrite("session_end", ctx, event);
    });

    api.on("gateway_start", (event: PluginHookGatewayStartEvent, ctx: PluginHookGatewayContext) => {
      hookWrite("gateway_start", ctx, event);
    });

    api.on("gateway_stop", (event: PluginHookGatewayStopEvent, ctx: PluginHookGatewayContext) => {
      hookWrite("gateway_stop", ctx, event);
    });
  },
};

export default plugin;

