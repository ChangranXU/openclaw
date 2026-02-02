import type { Langfuse, LangfuseSpanClient, LangfuseGenerationClient } from "langfuse";
import type {
  TrackerState,
  GenerationParams,
  GenerationEndParams,
  SpanParams,
  SpanEndParams,
  TraceContext,
  GenerationContext,
  ToolCallContext,
  ObservabilityProvider,
} from "./types.js";

/**
 * TraceManager handles session-scoped trace management with concurrency support.
 * Each session (identified by sessionId) gets its own trace with isolated spans.
 */
export class TraceManager {
  private langfuse: Langfuse;
  private traces = new Map<string, TrackerState>();
  private toolSpanStack = new Map<string, string[]>(); // sessionId -> stack of span IDs

  constructor(langfuse: Langfuse) {
    this.langfuse = langfuse;
  }

  private truncateText(value: unknown, maxChars: number): string | undefined {
    if (typeof value !== "string") {
      return undefined;
    }
    const text = value;
    if (text.length <= maxChars) {
      return text;
    }
    return `${text.slice(0, Math.max(0, maxChars - 3))}...`;
  }

  /**
   * Get or create a trace for a session.
   */
  getOrCreateTrace(sessionId: string, sessionKey?: string, channel?: string): TrackerState {
    if (!this.traces.has(sessionId)) {
      const trace = this.langfuse.trace({
        id: sessionId,
        name: `openclaw_session_${sessionId.slice(0, 8)}`,
        sessionId: sessionKey ?? sessionId,
        tags: channel ? [channel] : undefined,
        metadata: {
          sessionKey,
          channel,
        },
      });

      this.traces.set(sessionId, {
        trace,
        activeSpans: new Map(),
        metadata: {
          channel,
          sessionKey,
          startedAt: new Date(),
        },
      });
    }
    return this.traces.get(sessionId)!;
  }

  /**
   * Start a new generation (LLM call) within a trace.
   */
  startGeneration(sessionId: string, params: GenerationParams): LangfuseGenerationClient | null {
    const state = this.traces.get(sessionId);
    if (!state) {
      return null;
    }

    // End any existing generation first
    if (state.currentGeneration) {
      state.currentGeneration.end();
    }

    const generation = state.trace.generation({
      name: params.name,
      model: params.model,
      input: params.input,
      metadata: {
        provider: params.provider,
        ...params.metadata,
      },
    });

    state.currentGeneration = generation;
    state.metadata.model = params.model;
    state.metadata.provider = params.provider;

    return generation;
  }

  /**
   * End the current generation with results.
   */
  endCurrentGeneration(sessionId: string, params: GenerationEndParams): void {
    const state = this.traces.get(sessionId);
    if (!state?.currentGeneration) {
      return;
    }

    const usage = params.usage;
    state.currentGeneration.end({
      output: params.output,
      usage: usage
        ? {
            input: usage.input,
            output: usage.output,
            total: usage.total,
          }
        : undefined,
      metadata: {
        success: params.success,
        error: params.error,
        costUsd: params.costUsd,
        durationMs: params.durationMs,
        cacheRead: usage?.cacheRead,
        cacheWrite: usage?.cacheWrite,
      },
      statusMessage: params.error,
      level: params.success === false ? "ERROR" : "DEFAULT",
    });

    state.currentGeneration = undefined;
  }

  /**
   * Start a tool call span.
   */
  startToolSpan(sessionId: string, params: SpanParams): LangfuseSpanClient | null {
    const state = this.traces.get(sessionId);
    if (!state) {
      return null;
    }

    const spanId = `tool_${params.name}_${Date.now()}`;

    // Determine parent: either the current generation or the trace
    const parent = state.currentGeneration ?? state.trace;

    const span = parent.span({
      name: `tool:${params.name}`,
      input: params.input,
      metadata: params.metadata,
    });

    state.activeSpans.set(spanId, span);

    // Track span stack for this session
    if (!this.toolSpanStack.has(sessionId)) {
      this.toolSpanStack.set(sessionId, []);
    }
    this.toolSpanStack.get(sessionId)!.push(spanId);

    return span;
  }

  /**
   * End the most recent tool span.
   * If no matching span is found, creates a fallback event to ensure error data is captured.
   */
  endToolSpan(sessionId: string, params: SpanEndParams): void {
    const state = this.traces.get(sessionId);
    const stack = this.toolSpanStack.get(sessionId);

    // If no matching trace/span exists, create a fallback event to capture the error data
    // This ensures tool errors are never silently lost
    if (!state || !stack || stack.length === 0) {
      // Try to get or create a trace for this session to log the error
      const fallbackState = this.getOrCreateTrace(sessionId);
      if (fallbackState && (params.error || params.output)) {
        const toolName = params.toolName ?? "unknown_tool";
        fallbackState.trace.event({
          name: `tool_error:${toolName}`,
          level: params.error ? "ERROR" : "WARNING",
          statusMessage: params.error,
          metadata: {
            toolName,
            input: params.input,
            output: params.output,
            error: params.error,
            durationMs: params.durationMs,
            fallback: true,
            reason: "no_matching_span",
          },
        });
      }
      return;
    }

    const spanId = stack.pop()!;
    const span = state.activeSpans.get(spanId);

    if (span) {
      // Include input in metadata when there's an error for better debugging
      const metadata: Record<string, unknown> = {
        error: params.error,
        durationMs: params.durationMs,
      };
      // Add input to error spans for full context visibility
      if (params.error && params.input !== undefined) {
        metadata.input = params.input;
      }

      span.end({
        output: params.output,
        statusMessage: params.error,
        level: params.error ? "ERROR" : "DEFAULT",
        metadata,
      });
      state.activeSpans.delete(spanId);
    } else {
      // Span ID was in stack but not found in activeSpans - create fallback event
      const toolName = params.toolName ?? "unknown_tool";
      state.trace.event({
        name: `tool_error:${toolName}`,
        level: params.error ? "ERROR" : "WARNING",
        statusMessage: params.error,
        metadata: {
          toolName,
          input: params.input,
          output: params.output,
          error: params.error,
          durationMs: params.durationMs,
          fallback: true,
          reason: "span_not_in_active_spans",
          spanId,
        },
      });
    }
  }

  /**
   * End a trace and clean up.
   */
  endTrace(sessionId: string): void {
    const state = this.traces.get(sessionId);
    if (!state) {
      return;
    }

    // End any active generation
    if (state.currentGeneration) {
      state.currentGeneration.end();
    }

    // End all active spans
    for (const span of state.activeSpans.values()) {
      span.end();
    }

    // Update trace
    state.trace.update({
      metadata: {
        ...state.metadata,
        endedAt: new Date().toISOString(),
        durationMs: Date.now() - state.metadata.startedAt.getTime(),
      },
    });

    // Clean up
    this.traces.delete(sessionId);
    this.toolSpanStack.delete(sessionId);
  }

  /**
   * Record model usage as a generation event.
   */
  recordModelUsage(params: {
    sessionId: string;
    sessionKey?: string;
    channel?: string;
    provider?: string;
    model?: string;
    /**
     * Diagnostic event sequence (monotonic within the process).
     * Used to make generation names stable + unique to avoid accidental dedupe/overwrites.
     */
    eventSeq?: number;
    /** Diagnostic event timestamp (ms since epoch). */
    eventTs?: number;
    usage: {
      input?: number;
      output?: number;
      total?: number;
      cacheRead?: number;
      cacheWrite?: number;
      promptTokens?: number;
    };
    costUsd?: number;
    durationMs?: number;
    context?: {
      limit?: number;
      used?: number;
    };
    /** The user's input message text. */
    inputText?: string;
    /** The model's output response text. */
    outputText?: string;
  }): void {
    const state = this.getOrCreateTrace(params.sessionId, params.sessionKey, params.channel);

    // Important: keep each generation uniquely identifiable.
    // Some Langfuse SDK/backends may de-dupe or merge objects when names/ids collide.
    const generationName = typeof params.eventSeq === "number" ? `model_call:${params.eventSeq}` : "model_call";

    const generation = state.trace.generation({
      name: generationName,
      model: params.model,
      input: params.inputText,
      metadata: {
        provider: params.provider,
        channel: params.channel,
        contextLimit: params.context?.limit,
        contextUsed: params.context?.used,
        diagnosticSeq: params.eventSeq,
        diagnosticTs: params.eventTs,
      },
    });

    generation.end({
      output: params.outputText,
      usage: {
        input: params.usage.input ?? params.usage.promptTokens,
        output: params.usage.output,
        total: params.usage.total,
      },
      metadata: {
        costUsd: params.costUsd,
        durationMs: params.durationMs,
        cacheRead: params.usage.cacheRead,
        cacheWrite: params.usage.cacheWrite,
      },
    });
  }

  /**
   * Flush all pending events to Langfuse.
   */
  async flush(): Promise<void> {
    await this.langfuse.flushAsync();
  }

  /**
   * Shutdown and clean up all resources.
   */
  async shutdown(): Promise<void> {
    // End all active traces
    for (const sessionId of this.traces.keys()) {
      this.endTrace(sessionId);
    }

    await this.flush();
    await this.langfuse.shutdownAsync();
  }

  /**
   * Get the count of active traces (for diagnostics).
   */
  getActiveTraceCount(): number {
    return this.traces.size;
  }
}

/**
 * Registry for additional observability providers (future extensibility).
 */
export class ObservabilityRegistry {
  private providers: ObservabilityProvider[] = [];

  register(provider: ObservabilityProvider): void {
    this.providers.push(provider);
  }

  async notifyTraceStart(ctx: TraceContext): Promise<void> {
    for (const provider of this.providers) {
      try {
        await provider.onTraceStart?.(ctx);
      } catch {
        // Ignore errors from providers
      }
    }
  }

  async notifyTraceEnd(ctx: TraceContext): Promise<void> {
    for (const provider of this.providers) {
      try {
        await provider.onTraceEnd?.(ctx);
      } catch {
        // Ignore errors from providers
      }
    }
  }

  async notifyGeneration(ctx: GenerationContext): Promise<void> {
    for (const provider of this.providers) {
      try {
        await provider.onGeneration?.(ctx);
      } catch {
        // Ignore errors from providers
      }
    }
  }

  async notifyToolCall(ctx: ToolCallContext): Promise<void> {
    for (const provider of this.providers) {
      try {
        await provider.onToolCall?.(ctx);
      } catch {
        // Ignore errors from providers
      }
    }
  }

  async collectAllMetadata(ctx: unknown): Promise<Record<string, unknown>> {
    const result: Record<string, unknown> = {};
    for (const provider of this.providers) {
      try {
        const metadata = provider.collectMetadata?.(ctx);
        if (metadata) {
          Object.assign(result, metadata);
        }
      } catch {
        // Ignore errors from providers
      }
    }
    return result;
  }
}
