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
   */
  endToolSpan(sessionId: string, params: SpanEndParams): void {
    const state = this.traces.get(sessionId);
    const stack = this.toolSpanStack.get(sessionId);

    if (!state || !stack || stack.length === 0) {
      return;
    }

    const spanId = stack.pop()!;
    const span = state.activeSpans.get(spanId);

    if (span) {
      span.end({
        output: params.output,
        statusMessage: params.error,
        level: params.error ? "ERROR" : "DEFAULT",
        metadata: {
          error: params.error,
          durationMs: params.durationMs,
        },
      });
      state.activeSpans.delete(spanId);
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

    const generation = state.trace.generation({
      name: "model_call",
      model: params.model,
      input: params.inputText,
      metadata: {
        provider: params.provider,
        channel: params.channel,
        contextLimit: params.context?.limit,
        contextUsed: params.context?.used,
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
