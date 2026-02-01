import type { DiagnosticEventPayload, OpenClawPluginService } from "openclaw/plugin-sdk";
import { Langfuse } from "langfuse";
import { onDiagnosticEvent } from "openclaw/plugin-sdk";
import { TraceManager, ObservabilityRegistry } from "./trace-manager.js";

const DEFAULT_BASE_URL = "https://cloud.langfuse.com";
const DEFAULT_FLUSH_AT = 15;
const DEFAULT_FLUSH_INTERVAL_MS = 10000;

// Module-level state for hook access
let _traceManager: TraceManager | null = null;
let _observabilityRegistry: ObservabilityRegistry | null = null;

/**
 * Get the current trace manager instance (for hook access).
 */
export function getTraceManager(): TraceManager | null {
  return _traceManager;
}

/**
 * Get the current observability registry instance (for hook access).
 */
export function getObservabilityRegistry(): ObservabilityRegistry | null {
  return _observabilityRegistry;
}

/**
 * Creates the Langfuse diagnostics service following the diagnostics-otel pattern.
 */
export function createDiagnosticsLangfuseService(): OpenClawPluginService {
  let langfuse: Langfuse | null = null;
  let traceManager: TraceManager | null = null;
  let observabilityRegistry: ObservabilityRegistry | null = null;
  let unsubscribeDiagnostic: (() => void) | null = null;

  return {
    id: "diagnostics-langfuse",

    async start(ctx) {
      const cfg = ctx.config.diagnostics;
      const langfuseCfg = cfg?.langfuse;

      // Check if Langfuse is enabled
      if (!cfg?.enabled || !langfuseCfg?.enabled) {
        return;
      }

      // Resolve configuration with env var fallbacks
      const publicKey = langfuseCfg.publicKey ?? process.env.LANGFUSE_PUBLIC_KEY;
      const secretKey = langfuseCfg.secretKey ?? process.env.LANGFUSE_SECRET_KEY;
      const baseUrl = langfuseCfg.baseUrl ?? process.env.LANGFUSE_BASE_URL ?? DEFAULT_BASE_URL;

      // Validate required keys
      if (!publicKey || !secretKey) {
        ctx.logger.warn(
          "diagnostics-langfuse: Missing publicKey or secretKey. Set LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY env vars or configure in diagnostics.langfuse.",
        );
        return;
      }

      try {
        // Initialize Langfuse client
        langfuse = new Langfuse({
          publicKey,
          secretKey,
          baseUrl,
          flushAt: langfuseCfg.flushAt ?? DEFAULT_FLUSH_AT,
          flushInterval: langfuseCfg.flushIntervalMs ?? DEFAULT_FLUSH_INTERVAL_MS,
          release: process.env.npm_package_version ?? "unknown",
        });

        // Enable debug mode if configured
        if (langfuseCfg.debug) {
          langfuse.debug();
        }

        traceManager = new TraceManager(langfuse);
        observabilityRegistry = new ObservabilityRegistry();

        // Set module-level references for hook access
        _traceManager = traceManager;
        _observabilityRegistry = observabilityRegistry;

        ctx.logger.info("diagnostics-langfuse: Langfuse tracing enabled", {
          baseUrl,
          flushAt: langfuseCfg.flushAt ?? DEFAULT_FLUSH_AT,
          flushIntervalMs: langfuseCfg.flushIntervalMs ?? DEFAULT_FLUSH_INTERVAL_MS,
        });

        // Subscribe to diagnostic events
        unsubscribeDiagnostic = onDiagnosticEvent((evt: DiagnosticEventPayload) => {
          handleDiagnosticEvent(evt, traceManager!, observabilityRegistry!, ctx.logger);
        });
      } catch (error) {
        ctx.logger.error("diagnostics-langfuse: Failed to initialize Langfuse", {
          error: error instanceof Error ? error.message : String(error),
        });
      }
    },

    async stop() {
      unsubscribeDiagnostic?.();
      unsubscribeDiagnostic = null;

      if (traceManager) {
        try {
          await traceManager.shutdown();
        } catch {
          // Ignore shutdown errors
        }
        traceManager = null;
      }

      if (langfuse) {
        try {
          await langfuse.shutdownAsync();
        } catch {
          // Ignore shutdown errors
        }
        langfuse = null;
      }

      observabilityRegistry = null;

      // Clear module-level references
      _traceManager = null;
      _observabilityRegistry = null;
    },
  };
}

/**
 * Handle diagnostic events and route them to Langfuse.
 */
function handleDiagnosticEvent(
  evt: DiagnosticEventPayload,
  traceManager: TraceManager,
  observabilityRegistry: ObservabilityRegistry,
  logger: { debug: (msg: string, data?: unknown) => void },
): void {
  try {
    switch (evt.type) {
      case "model.usage":
        handleModelUsage(evt, traceManager, observabilityRegistry);
        break;

      case "message.processed":
        handleMessageProcessed(evt, traceManager);
        break;

      case "session.state":
        handleSessionState(evt, traceManager, observabilityRegistry);
        break;

      case "webhook.received":
        handleWebhookReceived(evt, traceManager);
        break;

      case "webhook.processed":
        handleWebhookProcessed(evt, traceManager);
        break;

      case "webhook.error":
        handleWebhookError(evt, traceManager);
        break;

      case "run.attempt":
        handleRunAttempt(evt, traceManager);
        break;

      case "tool.start":
        handleToolStart(evt, traceManager);
        break;

      case "tool.end":
        handleToolEnd(evt, traceManager);
        break;

      case "llm.error":
        handleLLMError(evt, traceManager);
        break;

      case "diagnostic.heartbeat":
        // Periodic flush on heartbeat
        traceManager.flush().catch(() => {
          // Ignore flush errors
        });
        break;

      default:
        // Ignore other event types
        break;
    }
  } catch (error) {
    logger.debug("diagnostics-langfuse: Error handling event", {
      type: evt.type,
      error: error instanceof Error ? error.message : String(error),
    });
  }
}

/**
 * Handle model usage events - the primary source of LLM tracing data.
 * Use sessionKey as the primary trace identifier to group all events from the same session.
 */
function handleModelUsage(
  evt: Extract<DiagnosticEventPayload, { type: "model.usage" }>,
  traceManager: TraceManager,
  observabilityRegistry: ObservabilityRegistry,
): void {
  // Use sessionKey as the trace identifier to ensure all events from the same session
  // are grouped into the same trace. sessionId is an internal run ID that changes per message.
  const traceId = evt.sessionKey ?? evt.sessionId ?? "unknown";

  traceManager.recordModelUsage({
    sessionId: traceId,
    sessionKey: evt.sessionKey,
    channel: evt.channel,
    provider: evt.provider,
    model: evt.model,
    usage: evt.usage,
    costUsd: evt.costUsd,
    durationMs: evt.durationMs,
    context: evt.context,
    inputText: evt.inputText,
    outputText: evt.outputText,
  });

  // Notify observability providers
  observabilityRegistry.notifyGeneration({
    sessionId: traceId,
    sessionKey: evt.sessionKey,
    channel: evt.channel,
    model: evt.model,
    provider: evt.provider,
    usage: evt.usage,
  });
}

/**
 * Handle message processed events.
 * Use sessionKey as the trace identifier to group with other session events.
 */
function handleMessageProcessed(
  evt: Extract<DiagnosticEventPayload, { type: "message.processed" }>,
  traceManager: TraceManager,
): void {
  // Use sessionKey as the primary trace identifier
  const traceId = evt.sessionKey ?? evt.sessionId;
  if (!traceId) {
    return;
  }

  const state = traceManager.getOrCreateTrace(traceId, evt.sessionKey, evt.channel);

  // Add a span for message processing
  const span = state.trace.span({
    name: `message:${evt.outcome ?? "processed"}`,
    metadata: {
      channel: evt.channel,
      outcome: evt.outcome,
      messageId: evt.messageId,
      chatId: evt.chatId,
      reason: evt.reason,
      sessionId: evt.sessionId, // Include internal sessionId for debugging
    },
  });

  span.end({
    statusMessage: evt.error,
    level: evt.outcome === "error" ? "ERROR" : "DEFAULT",
    metadata: {
      durationMs: evt.durationMs,
      error: evt.error,
    },
  });
}

/**
 * Handle session state changes.
 * Use sessionKey as the trace identifier to ensure consistency.
 */
function handleSessionState(
  evt: Extract<DiagnosticEventPayload, { type: "session.state" }>,
  traceManager: TraceManager,
  observabilityRegistry: ObservabilityRegistry,
): void {
  // Use sessionKey as the primary trace identifier
  const traceId = evt.sessionKey ?? evt.sessionId;
  if (!traceId) {
    return;
  }

  if (evt.state === "ended" || evt.state === "reset") {
    // End the trace when session ends
    observabilityRegistry.notifyTraceEnd({
      sessionId: traceId,
      sessionKey: evt.sessionKey,
      channel: evt.channel,
    });
    traceManager.endTrace(traceId);
  } else if (evt.state === "started" || evt.state === "resumed") {
    // Create or get trace on session start
    traceManager.getOrCreateTrace(traceId, evt.sessionKey, evt.channel);
    observabilityRegistry.notifyTraceStart({
      sessionId: traceId,
      sessionKey: evt.sessionKey,
      channel: evt.channel,
    });
  }
}

/**
 * Handle webhook received events.
 */
function handleWebhookReceived(
  evt: Extract<DiagnosticEventPayload, { type: "webhook.received" }>,
  traceManager: TraceManager,
): void {
  // Create a trace event for webhook receipt
  const trace = traceManager.getOrCreateTrace(
    `webhook_${evt.channel}_${Date.now()}`,
    undefined,
    evt.channel,
  );

  trace.trace.event({
    name: "webhook_received",
    metadata: {
      channel: evt.channel,
      updateType: evt.updateType,
    },
  });
}

/**
 * Handle webhook processed events.
 */
function handleWebhookProcessed(
  evt: Extract<DiagnosticEventPayload, { type: "webhook.processed" }>,
  traceManager: TraceManager,
): void {
  const sessionId = `webhook_${evt.channel}_${evt.chatId ?? Date.now()}`;
  const state = traceManager.getOrCreateTrace(sessionId, undefined, evt.channel);

  const span = state.trace.span({
    name: "webhook_processed",
    metadata: {
      channel: evt.channel,
      updateType: evt.updateType,
      chatId: evt.chatId,
    },
  });

  span.end({
    metadata: {
      durationMs: evt.durationMs,
    },
  });
}

/**
 * Handle webhook error events.
 */
function handleWebhookError(
  evt: Extract<DiagnosticEventPayload, { type: "webhook.error" }>,
  traceManager: TraceManager,
): void {
  const sessionId = `webhook_${evt.channel}_${evt.chatId ?? Date.now()}`;
  const state = traceManager.getOrCreateTrace(sessionId, undefined, evt.channel);

  const span = state.trace.span({
    name: "webhook_error",
    metadata: {
      channel: evt.channel,
      updateType: evt.updateType,
      chatId: evt.chatId,
    },
  });

  span.end({
    statusMessage: evt.error,
    level: "ERROR",
    metadata: {
      error: evt.error,
    },
  });
}

/**
 * Handle run attempt events.
 * Use sessionKey as the trace identifier to group with other session events.
 */
function handleRunAttempt(
  evt: Extract<DiagnosticEventPayload, { type: "run.attempt" }>,
  traceManager: TraceManager,
): void {
  // Use sessionKey as the primary trace identifier
  const traceId = evt.sessionKey ?? evt.sessionId ?? `run_${Date.now()}`;
  const state = traceManager.getOrCreateTrace(traceId, evt.sessionKey, evt.channel);

  state.trace.event({
    name: "run_attempt",
    metadata: {
      attempt: evt.attempt,
      sessionKey: evt.sessionKey,
      sessionId: evt.sessionId, // Include internal sessionId for debugging
      runId: evt.runId,
      channel: evt.channel,
    },
  });
}

/**
 * Handle tool start events.
 * Creates a span for tool execution tracking.
 */
function handleToolStart(
  evt: Extract<DiagnosticEventPayload, { type: "tool.start" }>,
  traceManager: TraceManager,
): void {
  // Use sessionKey as the primary trace identifier
  const traceId = evt.sessionKey ?? evt.runId ?? `tool_${Date.now()}`;

  // Ensure trace exists before creating tool span
  traceManager.getOrCreateTrace(traceId, evt.sessionKey, evt.channel);

  // Start a tool span using the trace manager
  traceManager.startToolSpan(traceId, {
    name: evt.toolName,
    input: evt.input,
    metadata: {
      toolCallId: evt.toolCallId,
      runId: evt.runId,
      sessionKey: evt.sessionKey,
    },
  });
}

/**
 * Handle tool end events.
 * Ends the tool span with result and error information.
 */
function handleToolEnd(
  evt: Extract<DiagnosticEventPayload, { type: "tool.end" }>,
  traceManager: TraceManager,
): void {
  // Use sessionKey as the primary trace identifier (must match tool.start)
  const traceId = evt.sessionKey ?? evt.runId ?? `tool_${Date.now()}`;

  // End the tool span
  traceManager.endToolSpan(traceId, {
    output: evt.output,
    error: evt.error,
    durationMs: evt.durationMs,
  });
}

/**
 * Handle LLM error events.
 * Creates an error event in the trace for visibility in Langfuse.
 */
function handleLLMError(
  evt: Extract<DiagnosticEventPayload, { type: "llm.error" }>,
  traceManager: TraceManager,
): void {
  // Use sessionKey as the primary trace identifier
  const traceId = evt.sessionKey ?? evt.runId ?? `llm_error_${Date.now()}`;

  // Ensure trace exists
  const state = traceManager.getOrCreateTrace(traceId, evt.sessionKey, evt.channel);

  // Create an error event in the trace
  state.trace.event({
    name: "llm_error",
    level: "ERROR",
    statusMessage: evt.errorMessage,
    metadata: {
      provider: evt.provider,
      model: evt.model,
      errorType: evt.errorType,
      statusCode: evt.statusCode,
      fallbackAttempted: evt.fallbackAttempted,
      sessionKey: evt.sessionKey,
      sessionId: evt.sessionId,
      runId: evt.runId,
    },
  });
}
