import type { OpenClawConfig } from "../config/config.js";

export type DiagnosticSessionState = "idle" | "processing" | "waiting" | "ended";

type DiagnosticBaseEvent = {
  ts: number;
  seq: number;
};

export type DiagnosticUsageEvent = DiagnosticBaseEvent & {
  type: "model.usage";
  sessionKey?: string;
  sessionId?: string;
  channel?: string;
  provider?: string;
  model?: string;
  usage: {
    input?: number;
    output?: number;
    cacheRead?: number;
    cacheWrite?: number;
    promptTokens?: number;
    total?: number;
  };
  context?: {
    limit?: number;
    used?: number;
  };
  costUsd?: number;
  durationMs?: number;
  /** The user's input message text (for Langfuse tracing). */
  inputText?: string;
  /** The model's output response text (for Langfuse tracing). */
  outputText?: string;
};

export type DiagnosticWebhookReceivedEvent = DiagnosticBaseEvent & {
  type: "webhook.received";
  channel: string;
  updateType?: string;
  chatId?: number | string;
};

export type DiagnosticWebhookProcessedEvent = DiagnosticBaseEvent & {
  type: "webhook.processed";
  channel: string;
  updateType?: string;
  chatId?: number | string;
  durationMs?: number;
};

export type DiagnosticWebhookErrorEvent = DiagnosticBaseEvent & {
  type: "webhook.error";
  channel: string;
  updateType?: string;
  chatId?: number | string;
  error: string;
};

export type DiagnosticMessageQueuedEvent = DiagnosticBaseEvent & {
  type: "message.queued";
  sessionKey?: string;
  sessionId?: string;
  channel?: string;
  source: string;
  queueDepth?: number;
};

export type DiagnosticMessageProcessedEvent = DiagnosticBaseEvent & {
  type: "message.processed";
  channel: string;
  messageId?: number | string;
  chatId?: number | string;
  sessionKey?: string;
  sessionId?: string;
  durationMs?: number;
  outcome: "completed" | "skipped" | "error";
  reason?: string;
  error?: string;
};

export type DiagnosticSessionStateEvent = DiagnosticBaseEvent & {
  type: "session.state";
  sessionKey?: string;
  sessionId?: string;
  channel?: string;
  prevState?: DiagnosticSessionState;
  state: DiagnosticSessionState;
  reason?: string;
  queueDepth?: number;
};

export type DiagnosticSessionStuckEvent = DiagnosticBaseEvent & {
  type: "session.stuck";
  sessionKey?: string;
  sessionId?: string;
  state: DiagnosticSessionState;
  ageMs: number;
  queueDepth?: number;
};

export type DiagnosticLaneEnqueueEvent = DiagnosticBaseEvent & {
  type: "queue.lane.enqueue";
  lane: string;
  queueSize: number;
};

export type DiagnosticLaneDequeueEvent = DiagnosticBaseEvent & {
  type: "queue.lane.dequeue";
  lane: string;
  queueSize: number;
  waitMs: number;
};

export type DiagnosticRunAttemptEvent = DiagnosticBaseEvent & {
  type: "run.attempt";
  sessionKey?: string;
  sessionId?: string;
  runId: string;
  attempt: number;
};

export type DiagnosticHeartbeatEvent = DiagnosticBaseEvent & {
  type: "diagnostic.heartbeat";
  webhooks: {
    received: number;
    processed: number;
    errors: number;
  };
  active: number;
  waiting: number;
  queued: number;
};

export type DiagnosticToolStartEvent = DiagnosticBaseEvent & {
  type: "tool.start";
  sessionKey?: string;
  sessionId?: string;
  runId?: string;
  channel?: string;
  toolName: string;
  toolCallId: string;
  /** Tool input arguments (may be truncated for large inputs). */
  input?: unknown;
};

export type DiagnosticToolEndEvent = DiagnosticBaseEvent & {
  type: "tool.end";
  sessionKey?: string;
  sessionId?: string;
  runId?: string;
  channel?: string;
  toolName: string;
  toolCallId: string;
  /** Duration of the tool execution in milliseconds. */
  durationMs?: number;
  /** Whether the tool execution resulted in an error. */
  isError?: boolean;
  /** Error message if the tool failed. */
  error?: string;
  /** Tool output result (may be truncated for large outputs). */
  output?: unknown;
  /** Tool input arguments (for error tracing - ensures input is captured even when tool fails). */
  input?: unknown;
};

export type DiagnosticInternalLogLevel =
  | "trace"
  | "debug"
  | "info"
  | "warn"
  | "error"
  | "fatal"
  | "silent";

/**
 * Internal (non-tool-call) actions/logs.
 *
 * Goal: make "internal behavior" observable in Langfuse even when it's not a model tool call.
 * This is especially useful for warnings/errors that otherwise only appear in stdout/stderr logs.
 */
export type DiagnosticInternalLogEvent = DiagnosticBaseEvent & {
  type: "internal.log";
  subsystem: string;
  level: Exclude<DiagnosticInternalLogLevel, "silent">;
  message: string;
  meta?: Record<string, unknown>;
  /** Optional trace binding (best-effort extracted from meta when present). */
  sessionKey?: string;
  sessionId?: string;
  runId?: string;
  channel?: string;
};

export type DiagnosticLLMErrorEvent = DiagnosticBaseEvent & {
  type: "llm.error";
  sessionKey?: string;
  sessionId?: string;
  runId?: string;
  channel?: string;
  provider?: string;
  model?: string;
  /** HTTP status code (e.g., 400, 429, 500). */
  statusCode?: number;
  /** Error type classification (e.g., "content_filter", "rate_limit", "auth", "timeout"). */
  errorType?: string;
  /** The error message from the LLM provider. */
  errorMessage?: string;
  /** Whether model fallback was attempted. */
  fallbackAttempted?: boolean;
};

export type DiagnosticEventPayload =
  | DiagnosticUsageEvent
  | DiagnosticWebhookReceivedEvent
  | DiagnosticWebhookProcessedEvent
  | DiagnosticWebhookErrorEvent
  | DiagnosticMessageQueuedEvent
  | DiagnosticMessageProcessedEvent
  | DiagnosticSessionStateEvent
  | DiagnosticSessionStuckEvent
  | DiagnosticLaneEnqueueEvent
  | DiagnosticLaneDequeueEvent
  | DiagnosticRunAttemptEvent
  | DiagnosticHeartbeatEvent
  | DiagnosticToolStartEvent
  | DiagnosticToolEndEvent
  | DiagnosticInternalLogEvent
  | DiagnosticLLMErrorEvent;

export type DiagnosticEventInput = DiagnosticEventPayload extends infer Event
  ? Event extends DiagnosticEventPayload
    ? Omit<Event, "seq" | "ts">
    : never
  : never;

// Use globalThis to ensure the same listeners set is shared across module instances
// This is necessary because plugins loaded via jiti may create separate module instances
const GLOBAL_KEY = "__openclaw_diagnostic_listeners__";
const GLOBAL_SEQ_KEY = "__openclaw_diagnostic_seq__";
const GLOBAL_INTERNAL_LOG_CAPTURE_KEY = "__openclaw_diagnostic_internal_log_capture__";

type GlobalDiagnosticState = {
  listeners: Set<(evt: DiagnosticEventPayload) => void>;
  seq: number;
  internalLogCapture?: DiagnosticInternalLogCaptureConfig;
};

export type DiagnosticInternalLogCaptureConfig = {
  enabled: boolean;
  /**
   * Minimum log level to emit as diagnostic events.
   * Default: "warn" (to avoid high-volume spam unless explicitly enabled).
   */
  minLevel?: Exclude<DiagnosticInternalLogLevel, "silent">;
  /** Whether to include the structured meta payload in diagnostics. Default: true. */
  includeMeta?: boolean;
  /** Maximum characters allowed for a single message (best-effort). Default: 4000. */
  maxMessageChars?: number;
  /** Maximum characters allowed for meta (stringified) (best-effort). Default: 8000. */
  maxMetaChars?: number;
  /**
   * Subsystem prefix denylist to prevent recursion/noise.
   * Any subsystem starting with one of these prefixes will not emit internal.log events.
   */
  denySubsystemPrefixes?: string[];
};

function getGlobalState(): GlobalDiagnosticState {
  const g = globalThis as unknown as Record<string, unknown>;
  if (!g[GLOBAL_KEY]) {
    g[GLOBAL_KEY] = new Set<(evt: DiagnosticEventPayload) => void>();
  }
  if (typeof g[GLOBAL_SEQ_KEY] !== "number") {
    g[GLOBAL_SEQ_KEY] = 0;
  }
  return {
    listeners: g[GLOBAL_KEY] as Set<(evt: DiagnosticEventPayload) => void>,
    get seq() {
      return g[GLOBAL_SEQ_KEY] as number;
    },
    set seq(val: number) {
      g[GLOBAL_SEQ_KEY] = val;
    },
    get internalLogCapture() {
      return g[GLOBAL_INTERNAL_LOG_CAPTURE_KEY] as DiagnosticInternalLogCaptureConfig | undefined;
    },
    set internalLogCapture(val: DiagnosticInternalLogCaptureConfig | undefined) {
      g[GLOBAL_INTERNAL_LOG_CAPTURE_KEY] = val as unknown;
    },
  };
}

const state = getGlobalState();

export function isDiagnosticsEnabled(config?: OpenClawConfig): boolean {
  return config?.diagnostics?.enabled === true;
}

export function emitDiagnosticEvent(event: DiagnosticEventInput) {
  state.seq += 1;
  const enriched = {
    ...event,
    seq: state.seq,
    ts: Date.now(),
  } satisfies DiagnosticEventPayload;
  for (const listener of state.listeners) {
    try {
      listener(enriched);
    } catch {
      // Ignore listener failures.
    }
  }
}

function levelToRank(level: Exclude<DiagnosticInternalLogLevel, "silent">): number {
  // Higher number = more severe
  switch (level) {
    case "trace":
      return 10;
    case "debug":
      return 20;
    case "info":
      return 30;
    case "warn":
      return 40;
    case "error":
      return 50;
    case "fatal":
      return 60;
  }
}

function truncateText(text: string, maxChars: number): string {
  if (text.length <= maxChars) {
    return text;
  }
  return `${text.slice(0, Math.max(0, maxChars - 3))}...`;
}

function coerceString(value: unknown): string | undefined {
  return typeof value === "string" && value.trim() ? value.trim() : undefined;
}

function extractTraceBinding(meta?: Record<string, unknown>): {
  sessionKey?: string;
  sessionId?: string;
  runId?: string;
  channel?: string;
} {
  if (!meta) {
    return {};
  }
  // Common keys we already use across the codebase.
  return {
    sessionKey: coerceString(meta.sessionKey),
    sessionId: coerceString(meta.sessionId),
    runId: coerceString(meta.runId),
    channel: coerceString(meta.channel) ?? coerceString(meta.messageProvider),
  };
}

function shouldDenySubsystem(subsystem: string, denyPrefixes: string[]): boolean {
  const trimmed = subsystem.trim();
  if (!trimmed) {
    return true;
  }
  const lowered = trimmed.toLowerCase();
  return denyPrefixes.some((prefix) => {
    const p = prefix.trim().toLowerCase();
    return p ? lowered.startsWith(p) : false;
  });
}

/**
 * Configure internal log capture (non-tool-call behavior) into diagnostic events.
 * This is intentionally global so the logging subsystem can emit without needing a config reference.
 */
export function configureDiagnosticInternalLogCapture(config: DiagnosticInternalLogCaptureConfig) {
  state.internalLogCapture = config;
}

/**
 * Best-effort: emit an internal.log diagnostic event.
 * Callers should treat this as non-throwing and extremely cheap when disabled.
 */
export function maybeEmitInternalLogDiagnosticEvent(params: {
  subsystem: string;
  level: Exclude<DiagnosticInternalLogLevel, "silent">;
  message: string;
  meta?: Record<string, unknown>;
}) {
  const cfg = state.internalLogCapture;
  if (!cfg?.enabled) {
    return;
  }
  const denyPrefixes = cfg.denySubsystemPrefixes ?? ["diagnostic", "diagnostics-langfuse"];
  if (shouldDenySubsystem(params.subsystem, denyPrefixes)) {
    return;
  }
  const minLevel = cfg.minLevel ?? "warn";
  if (levelToRank(params.level) < levelToRank(minLevel)) {
    return;
  }
  const maxMessageChars = cfg.maxMessageChars ?? 4000;
  const maxMetaChars = cfg.maxMetaChars ?? 8000;
  const includeMeta = cfg.includeMeta !== false;
  const truncatedMessage = truncateText(String(params.message ?? ""), maxMessageChars);
  const binding = extractTraceBinding(params.meta);

  let meta: Record<string, unknown> | undefined;
  if (includeMeta && params.meta && Object.keys(params.meta).length > 0) {
    // Avoid huge payloads: if meta is too large, store a truncated string form.
    try {
      const raw = JSON.stringify(params.meta);
      meta =
        raw.length > maxMetaChars
          ? { truncated: true, meta: truncateText(raw, maxMetaChars) }
          : params.meta;
    } catch {
      meta = { truncated: true, meta: "[unserializable meta]" };
    }
  }

  emitDiagnosticEvent({
    type: "internal.log",
    subsystem: params.subsystem,
    level: params.level,
    message: truncatedMessage,
    meta,
    ...binding,
  });
}

export function onDiagnosticEvent(listener: (evt: DiagnosticEventPayload) => void): () => void {
  state.listeners.add(listener);
  return () => state.listeners.delete(listener);
}

export function resetDiagnosticEventsForTest(): void {
  state.seq = 0;
  state.listeners.clear();
}
