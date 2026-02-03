from __future__ import annotations

import sys
import time
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

try:
    # v3 exposes Langfuse here
    from langfuse import Langfuse
except Exception:  # pragma: no cover
    Langfuse = None  # type: ignore[assignment]


def _format_span_id(span_id_int: int) -> str:
    return format(span_id_int, "016x")


def _span_id_hex(span_obj: Any) -> Optional[str]:
    # Best-effort: LangfuseSpan/LangfuseGeneration holds an OTEL span.
    otel_span = getattr(span_obj, "_otel_span", None)
    if otel_span is None:
        return None
    try:
        ctx = otel_span.get_span_context()
        return _format_span_id(ctx.span_id)
    except Exception:
        return None


@dataclass
class TraceState:
    trace_seed: str
    trace_id: str
    session_key: Optional[str] = None
    channel: Optional[str] = None
    root_span: Any = None
    root_span_id: Optional[str] = None
    run_span: Any = None
    run_span_id: Optional[str] = None
    tool_spans: Dict[str, Any] = field(default_factory=dict)  # toolCallId -> span


class TraceManager:
    def __init__(self, client: Any) -> None:
        self._client = client
        self._lock = threading.Lock()
        self._traces: Dict[str, TraceState] = {}
        self._last_error_ts = 0.0

    def _report_error(self, err: BaseException) -> None:
        # Avoid noisy logs; report at most once per 10s.
        now = time.time()
        if now - self._last_error_ts >= 10.0:
            print(f"[o-observability] Langfuse client error: {err!r}", file=sys.stderr)
            self._last_error_ts = now

    @staticmethod
    def _trace_id_for_seed(seed: str) -> str:
        if Langfuse is None:
            # Fallback, but should not happen when langfuse installed.
            import hashlib

            return hashlib.sha256(seed.encode("utf-8")).digest()[:16].hex()
        return Langfuse.create_trace_id(seed=seed)

    def get_or_create_trace(
        self, *, trace_seed: str, session_key: Optional[str], channel: Optional[str]
    ) -> TraceState:
        with self._lock:
            state = self._traces.get(trace_seed)
            if state:
                # Update best-effort metadata
                if session_key and not state.session_key:
                    state.session_key = session_key
                if channel and not state.channel:
                    state.channel = channel
                return state

            trace_id = self._trace_id_for_seed(trace_seed)
            root = self._client.start_span(
                trace_context={"trace_id": trace_id},
                name=f"openclaw_session_{trace_seed[:8]}",
            )
            root_id = _span_id_hex(root)

            # Set trace-level attributes via root span when possible
            try:
                root.update_trace(
                    name=f"openclaw_session_{trace_seed[:8]}",
                    session_id=session_key or trace_seed,
                    metadata={"sessionKey": session_key, "channel": channel},
                    tags=[channel] if channel else None,
                )
            except Exception as err:
                self._report_error(err)

            state = TraceState(
                trace_seed=trace_seed,
                trace_id=trace_id,
                session_key=session_key,
                channel=channel,
                root_span=root,
                root_span_id=root_id,
            )
            self._traces[trace_seed] = state
            return state

    def start_run_span(
        self, trace_seed: str, *, name: str, metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        with self._lock:
            state = self._traces.get(trace_seed)
            if not state or not state.root_span_id:
                return
            if state.run_span is not None:
                return
            span = self._client.start_span(
                trace_context={
                    "trace_id": state.trace_id,
                    "parent_span_id": state.root_span_id,
                },
                name=name,
                metadata=metadata,
            )
            state.run_span = span
            state.run_span_id = _span_id_hex(span)

    def record_instant_span(
        self,
        trace_seed: str,
        *,
        name: str,
        metadata: Optional[Dict[str, Any]] = None,
        status_message: Optional[str] = None,
        level: Optional[str] = None,
    ) -> None:
        """
        Create a very short SPAN so Langfuse UI has a Graph node even when
        a trace only contains command/control events like /new or /reset.
        """
        with self._lock:
            state = self._traces.get(trace_seed)
            if not state or not state.root_span_id:
                return
            trace_id = state.trace_id
            parent_span_id = state.root_span_id
        try:
            span = self._client.start_span(
                trace_context={"trace_id": trace_id, "parent_span_id": parent_span_id},
                name=name,
                metadata=metadata,
            )
            if status_message or level:
                try:
                    span.update(status_message=status_message, level=level)
                except Exception as err:
                    self._report_error(err)
            span.end()
        except Exception as err:
            self._report_error(err)

    def end_run_span(
        self,
        trace_seed: str,
        *,
        status_message: Optional[str] = None,
        level: Optional[str] = None,
    ) -> None:
        with self._lock:
            state = self._traces.get(trace_seed)
            if not state or not state.run_span:
                return
            span = state.run_span
            state.run_span = None
            state.run_span_id = None
        try:
            if status_message or level:
                span.update(status_message=status_message, level=level)
        except Exception as err:
            self._report_error(err)
        try:
            span.end()
        except Exception as err:
            self._report_error(err)

    def start_tool_span(
        self,
        trace_seed: str,
        *,
        tool_call_id: str,
        tool_name: str,
        input_data: Any,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        with self._lock:
            state = self._traces.get(trace_seed)
            if not state:
                return
            parent_span_id = state.run_span_id or state.root_span_id
            if not parent_span_id:
                return
            if tool_call_id in state.tool_spans:
                return
            span = self._client.start_observation(
                trace_context={
                    "trace_id": state.trace_id,
                    "parent_span_id": parent_span_id,
                },
                name=f"tool:{tool_name}",
                as_type="tool",
                input=input_data,
                metadata=metadata,
            )
            state.tool_spans[tool_call_id] = span

    def end_tool_span(
        self,
        trace_seed: str,
        *,
        tool_call_id: str,
        output_data: Any,
        error: Optional[str],
        level: Optional[str] = None,
        status_message: Optional[str] = None,
        duration_ms: Optional[float],
        input_data: Any,
        extra_metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        with self._lock:
            state = self._traces.get(trace_seed)
            span = state.tool_spans.pop(tool_call_id, None) if state else None
        if not span:
            return
        try:
            meta: Dict[str, Any] = {}
            if duration_ms is not None:
                meta["durationMs"] = duration_ms
            if error:
                meta["error"] = error
                if input_data is not None:
                    meta["input"] = input_data
            if isinstance(extra_metadata, dict):
                meta.update(extra_metadata)
            final_status = status_message if status_message is not None else error
            final_level = (
                level if level is not None else ("ERROR" if error else "DEFAULT")
            )
            if final_level == "WARNING" and final_status and "warning" not in meta:
                meta["warning"] = final_status
            span.update(
                output=output_data,
                status_message=final_status,
                level=final_level,
                metadata=meta or None,
            )
        except Exception as err:
            self._report_error(err)
        try:
            span.end()
        except Exception as err:
            self._report_error(err)

    def record_generation(
        self,
        trace_seed: str,
        *,
        name: str,
        model: Optional[str],
        provider: Optional[str],
        input_text: Optional[str],
        output_text: Optional[str],
        usage: Dict[str, Any],
        cost_usd: Optional[float],
        cost_details: Optional[Dict[str, Any]] = None,
        duration_ms: Optional[float],
        has_user_visible_reply: Optional[bool],
    ) -> None:
        with self._lock:
            state = self._traces.get(trace_seed)
            if not state:
                return
            parent_span_id = state.run_span_id or state.root_span_id
            if not parent_span_id:
                return
            trace_id = state.trace_id
            # session_key/channel are trace-level metadata; generation observation
            # only needs parent_span_id + trace_id.

        # Match OpenClaw build exports: usageDetails has input/output/total, and
        # costDetails has input/output/total when available.
        usage_details: Dict[str, int] = {}
        in_tok = (
            usage.get("input")
            or usage.get("promptTokens")
            or usage.get("prompt_tokens")
        )
        out_tok = usage.get("output") or usage.get("completion_tokens")
        total_tok = (
            usage.get("total") or usage.get("totalTokens") or usage.get("total_tokens")
        )
        if isinstance(in_tok, int):
            usage_details["input"] = in_tok
        if isinstance(out_tok, int):
            usage_details["output"] = out_tok
        if isinstance(total_tok, int):
            usage_details["total"] = total_tok

        cost_details_out: Optional[Dict[str, float]] = None
        if isinstance(cost_details, dict):
            ci = cost_details.get("input")
            co = cost_details.get("output")
            ct = cost_details.get("total")
            d: Dict[str, float] = {}
            if isinstance(ci, (int, float)):
                d["input"] = float(ci)
            if isinstance(co, (int, float)):
                d["output"] = float(co)
            if isinstance(ct, (int, float)):
                d["total"] = float(ct)
            cost_details_out = d or None
        elif isinstance(cost_usd, (int, float)):
            cost_details_out = {"total": float(cost_usd)}

        metadata: Dict[str, Any] = {
            # Keep generation metadata small to avoid redundant per-call payloads.
            # Trace-level metadata already includes sessionKey/channel.
            "provider": provider,
            "durationMs": duration_ms,
            "hasUserVisibleReply": has_user_visible_reply,
        }

        gen = self._client.start_observation(
            trace_context={"trace_id": trace_id, "parent_span_id": parent_span_id},
            name=name,
            as_type="generation",
            model=model,
            input=input_text,
            metadata=metadata,
        )
        try:
            gen.update(
                output=output_text,
                usage_details=usage_details or None,
                cost_details=cost_details_out,
            )
        except Exception as err:
            self._report_error(err)
        try:
            gen.end()
        except Exception as err:
            self._report_error(err)

    def record_event(
        self,
        trace_seed: str,
        *,
        name: str,
        level: str,
        status_message: Optional[str],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        with self._lock:
            state = self._traces.get(trace_seed)
            if not state:
                return
            parent_span_id = state.run_span_id or state.root_span_id
            if not parent_span_id:
                return
            trace_id = state.trace_id
        try:
            self._client.create_event(
                trace_context={"trace_id": trace_id, "parent_span_id": parent_span_id},
                name=name,
                level=level,
                status_message=status_message,
                metadata=metadata,
            )
        except Exception as err:
            self._report_error(err)

    def end_trace(self, trace_seed: str, *, reason: Optional[str] = None) -> None:
        with self._lock:
            state = self._traces.pop(trace_seed, None)
        if not state:
            return

        # End active tool spans
        for _tool_call_id, span in list(state.tool_spans.items()):
            try:
                span.end()
            except Exception as err:
                self._report_error(err)
        state.tool_spans.clear()

        # End run span
        if state.run_span is not None:
            try:
                state.run_span.update(status_message=reason, level="DEFAULT")
            except Exception as err:
                self._report_error(err)
            try:
                state.run_span.end()
            except Exception as err:
                self._report_error(err)

        # End root span
        if state.root_span is not None:
            try:
                if reason:
                    state.root_span.update(status_message=reason)
            except Exception as err:
                self._report_error(err)
            try:
                state.root_span.end()
            except Exception as err:
                self._report_error(err)

    def flush(self) -> None:
        try:
            self._client.flush()
        except Exception as err:
            # Surface flush issues (auth/network) to gateway logs.
            self._report_error(err)
