import type { LangfuseTraceClient, LangfuseSpanClient, LangfuseGenerationClient } from "langfuse";

export type TrackerState = {
  trace: LangfuseTraceClient;
  currentGeneration?: LangfuseGenerationClient;
  activeSpans: Map<string, LangfuseSpanClient>;
  metadata: TraceMetadata;
};

export type TraceMetadata = {
  channel?: string;
  provider?: string;
  model?: string;
  sessionKey?: string;
  startedAt: Date;
};

export type GenerationParams = {
  name: string;
  model?: string;
  provider?: string;
  input?: unknown;
  metadata?: Record<string, unknown>;
};

export type GenerationEndParams = {
  output?: unknown;
  usage?: {
    input?: number;
    output?: number;
    total?: number;
    cacheRead?: number;
    cacheWrite?: number;
  };
  costUsd?: number;
  durationMs?: number;
  success?: boolean;
  error?: string;
};

export type SpanParams = {
  name: string;
  input?: unknown;
  metadata?: Record<string, unknown>;
};

export type SpanEndParams = {
  output?: unknown;
  error?: string;
  durationMs?: number;
  /** Tool input for error tracing - ensures input is visible when tool fails. */
  input?: unknown;
  /** Tool name for fallback event creation when span is missing. */
  toolName?: string;
};

export type TraceContext = {
  sessionId: string;
  sessionKey?: string;
  channel?: string;
};

export type GenerationContext = TraceContext & {
  model?: string;
  provider?: string;
  usage?: GenerationEndParams["usage"];
};

export type ToolCallContext = TraceContext & {
  toolName: string;
  input?: unknown;
  output?: unknown;
  error?: string;
  durationMs?: number;
};

/**
 * Extensibility interface for future features (e.g., policy checkers, evaluators).
 */
export interface ObservabilityProvider {
  onTraceStart?(ctx: TraceContext): void | Promise<void>;
  onTraceEnd?(ctx: TraceContext): void | Promise<void>;
  onGeneration?(ctx: GenerationContext): void | Promise<void>;
  onToolCall?(ctx: ToolCallContext): void | Promise<void>;
  collectMetadata?(ctx: unknown): Record<string, unknown>;
}
