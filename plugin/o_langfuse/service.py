from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from typing import Any, Deque, Dict, Iterable, Optional, Tuple
from collections import deque
import re
import uuid
import time

from .trace_manager import TraceManager

try:
    from langfuse import Langfuse
except Exception:  # pragma: no cover
    Langfuse = None  # type: ignore[assignment]


JsonDict = Dict[str, Any]

_PROMPT_CONV_RE = re.compile(r"\[([A-Za-z][A-Za-z0-9_-]*)\s+([^\s\]]+)")

# Channels we treat as "real" routing headers inside prompts.
# LangGraph queue wrappers often start with "[Queued messages while agent was busy]"
# which would otherwise be (wrongly) parsed as channel="Queued", conv="messages".
_KNOWN_PROMPT_CHANNELS = {
    "whatsapp",
    "telegram",
    "slack",
    "discord",
    "wechat",
    "sms",
    "email",
    "web",
    "api",
}


def _new_uuid() -> str:
    return str(uuid.uuid4())


def _now_ms() -> int:
    return int(time.time() * 1000)


def _channel_from_any(envelope: JsonDict) -> Optional[str]:
    ctx = envelope.get("context") if isinstance(envelope.get("context"), dict) else {}
    payload = envelope.get("payload") if isinstance(envelope.get("payload"), dict) else {}
    ch = _coerce_str(ctx.get("channelId")) or _coerce_str(ctx.get("channel"))
    if ch:
        return ch.strip().lower()
    # before_agent_start often has messageProvider
    ch = _coerce_str(ctx.get("messageProvider")) or _coerce_str(payload.get("messageProvider"))
    if ch:
        return ch.strip().lower()
    # message_received often has metadata.provider
    meta = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}
    ch = _coerce_str(meta.get("provider")) or _coerce_str(meta.get("surface"))
    if ch:
        return ch.strip().lower()

    # Last resort: parse from prompt text (works for queued prompts).
    prompt = payload.get("prompt")
    p = _coerce_str(prompt)
    if not p:
        return None
    # Prefer a known channel header if present; avoid matching queue wrappers.
    for m in _PROMPT_CONV_RE.finditer(p):
        ch = (m.group(1) or "").strip().lower()
        if ch in _KNOWN_PROMPT_CHANNELS:
            return ch
    # Fallback: keep old behavior (first bracket header) if nothing matches.
    m0 = _PROMPT_CONV_RE.search(p)
    return (m0.group(1) or "").strip().lower() if m0 else None


def _conv_key_from_message_envelope(envelope: JsonDict) -> Optional[str]:
    ctx = envelope.get("context") if isinstance(envelope.get("context"), dict) else {}
    channel = _coerce_str(ctx.get("channelId")) or _coerce_str(ctx.get("channel"))
    conv = _coerce_str(ctx.get("conversationId")) or _coerce_str(ctx.get("chatId")) or _coerce_str(ctx.get("threadId"))
    if not conv:
        payload = envelope.get("payload") if isinstance(envelope.get("payload"), dict) else {}
        # Fallbacks for channels that don't provide conversationId in context.
        conv = _coerce_str(payload.get("from")) or _coerce_str(payload.get("senderId"))
    if not channel or not conv:
        return None
    return f"{channel.lower()}:{conv}"


def _conv_key_from_prompt(prompt: Any, *, preferred_channel: Optional[str] = None) -> Optional[str]:
    p = _coerce_str(prompt)
    if not p:
        return None
    # Prompts can be queued wrappers; scan for the most plausible embedded "[channel conv]".
    preferred = (preferred_channel or "").strip().lower() or None

    best: Optional[Tuple[str, str]] = None
    for m in _PROMPT_CONV_RE.finditer(p):
        channel = (m.group(1) or "").strip().lower()
        conv = (m.group(2) or "").strip()
        if not channel or not conv:
            continue
        if preferred and channel == preferred:
            return f"{channel}:{conv}"
        if channel in _KNOWN_PROMPT_CHANNELS and best is None:
            best = (channel, conv)

    if best:
        return f"{best[0]}:{best[1]}"

    # Fallback: old behavior (first match)
    m0 = _PROMPT_CONV_RE.search(p)
    if not m0:
        return None
    channel = (m0.group(1) or "").strip().lower()
    conv = (m0.group(2) or "").strip()
    return f"{channel}:{conv}" if channel and conv else None


def _text_from_content(content: Any) -> Optional[str]:
    if isinstance(content, str):
        return content.strip() or None
    if isinstance(content, list):
        parts: list[str] = []
        for p in content:
            if isinstance(p, dict):
                t = _coerce_str(p.get("text"))
                if t:
                    parts.append(t)
        if parts:
            return "\n".join(parts)
    if isinstance(content, dict):
        # Some payloads embed text as {"text": "..."} or {"content": "..."}.
        t = _coerce_str(content.get("text")) or _coerce_str(content.get("content"))
        if t:
            return t
    return None


def _message_received_text(payload: Any) -> Optional[str]:
    """
    hook.message_received payload shape varies by channel/build:
    - {"content": "..."} or {"content": [{text:"..."}]}
    - {"message": {"content": "..."}} (common)
    """
    if not isinstance(payload, dict):
        return None
    if "content" in payload:
        return _text_from_content(payload.get("content"))
    msg = payload.get("message")
    if isinstance(msg, dict):
        return _text_from_content(msg.get("content"))
    return None


def _prompt_user_text(prompt: Any) -> Optional[str]:
    """
    Prompts often start with a routing header like: "[channel conv] ...".
    Extract the likely user text after the header so we can detect commands
    like /reset and /new even when no hook.message_received exists.
    """
    p = _coerce_str(prompt)
    if not p:
        return None
    s = p.strip()
    if s.startswith("["):
        end = s.find("]")
        if end != -1:
            rest = s[end + 1 :].strip()
            if rest.startswith(":"):
                rest = rest[1:].strip()
            return rest or None
    return s


def _is_reset_command(text: Any) -> bool:
    t = _text_from_content(text)
    if not t:
        return False
    t = t.strip().lower()
    first = t.split(maxsplit=1)[0] if t else ""
    return first in ("/new", "/reset")


def _message_text(msg: JsonDict) -> Optional[str]:
    return _text_from_content(msg.get("content"))


def _generation_input_for_index(messages: Any, idx: int) -> Optional[str]:
    """
    Best-effort: infer generation input from the nearest preceding user message.
    """
    if not isinstance(messages, list) or idx <= 0:
        return None
    for j in range(idx - 1, -1, -1):
        m = messages[j]
        if not isinstance(m, dict):
            continue
        if _coerce_str(m.get("role")) != "user":
            continue
        return _message_text(m)
    return None


def _iso_from_ts_ms(ts_ms: Any) -> Optional[str]:
    if not isinstance(ts_ms, (int, float)):
        return None
    # Best-effort ISO string in UTC without extra deps.
    try:
        import datetime

        return datetime.datetime.fromtimestamp(float(ts_ms) / 1000.0, tz=datetime.timezone.utc).isoformat()
    except Exception:
        return None


def _extract_tool_name(envelope: JsonDict) -> Optional[str]:
    ctx = envelope.get("context") if isinstance(envelope.get("context"), dict) else {}
    payload = envelope.get("payload") if isinstance(envelope.get("payload"), dict) else {}
    return _coerce_str(ctx.get("toolName")) or _coerce_str(payload.get("toolName"))

def _extract_tool_call_id(envelope: JsonDict) -> Optional[str]:
    ctx = envelope.get("context") if isinstance(envelope.get("context"), dict) else {}
    payload = envelope.get("payload") if isinstance(envelope.get("payload"), dict) else {}
    # Most builds put it in context; keep payload fallback for compatibility.
    return _coerce_str(ctx.get("toolCallId")) or _coerce_str(payload.get("toolCallId"))


def _extract_session_key(envelope: JsonDict) -> Optional[str]:
    ctx = envelope.get("context") if isinstance(envelope.get("context"), dict) else {}
    payload = envelope.get("payload") if isinstance(envelope.get("payload"), dict) else {}
    return _coerce_str(ctx.get("sessionKey")) or _coerce_str(payload.get("sessionKey"))


def _extract_agent_id(envelope: JsonDict) -> Optional[str]:
    ctx = envelope.get("context") if isinstance(envelope.get("context"), dict) else {}
    return _coerce_str(ctx.get("agentId"))


def _extract_tool_input(envelope: JsonDict) -> Any:
    payload = envelope.get("payload") if isinstance(envelope.get("payload"), dict) else {}
    params = payload.get("params")
    return params if params is not None else payload


def _extract_tool_output_from_result_persist(envelope: JsonDict) -> Tuple[Any, Optional[str]]:
    """
    Returns (output_data, error_message).
    """
    payload = envelope.get("payload") if isinstance(envelope.get("payload"), dict) else {}
    msg = payload.get("message") if isinstance(payload.get("message"), dict) else {}
    is_error = msg.get("isError") is True
    content = msg.get("content")
    # content is typically [{type:"text", text:"..."}]
    if isinstance(content, list):
        texts: list[str] = []
        for part in content:
            if isinstance(part, dict):
                t = _coerce_str(part.get("text"))
                if t:
                    texts.append(t)
        out: Any = "\n".join(texts) if texts else content
    else:
        out = content if content is not None else msg

    err: Optional[str] = None
    if is_error:
        err = _coerce_str(msg.get("error")) or "tool_error"
    return out, err


def _iter_assistant_messages(messages: Any) -> Iterable[JsonDict]:
    if not isinstance(messages, list):
        return []
    out: list[JsonDict] = []
    for m in messages:
        if not isinstance(m, dict):
            continue
        if _coerce_str(m.get("role")) != "assistant":
            continue
        # Only treat as a "generation" when we have model/provider/usage data.
        if not (_coerce_str(m.get("model")) or _coerce_str(m.get("provider")) or isinstance(m.get("usage"), dict)):
            continue
        out.append(m)
    return out


def _assistant_text(msg: JsonDict) -> Optional[str]:
    content = msg.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for p in content:
            if isinstance(p, dict):
                t = _coerce_str(p.get("text"))
                if t:
                    parts.append(t)
        if parts:
            return "\n".join(parts)
    return None


def _coerce_str(val: Any) -> Optional[str]:
    if isinstance(val, str):
        s = val.strip()
        return s if s else None
    return None


def _extract_first(d: JsonDict, keys: Tuple[str, ...]) -> Optional[str]:
    for k in keys:
        v = _coerce_str(d.get(k))
        if v:
            return v
    return None


def _trace_seed_from_envelope(envelope: JsonDict) -> str:
    payload = envelope.get("payload")
    context = envelope.get("context")
    if isinstance(payload, dict):
        session_id = _extract_first(payload, ("sessionId", "session_id"))
        session_key = _extract_first(payload, ("sessionKey", "session_key"))
        run_id = _extract_first(payload, ("runId", "run_id"))
        if session_id:
            return session_id
        if session_key:
            return session_key
        if run_id:
            return run_id
    if isinstance(context, dict):
        session_id = _extract_first(context, ("sessionId", "session_id"))
        session_key = _extract_first(context, ("sessionKey", "session_key"))
        if session_id:
            return session_id
        if session_key:
            return session_key
        channel_id = _extract_first(context, ("channelId", "channel"))
        conv_id = _extract_first(context, ("conversationId", "chatId", "threadId"))
        if channel_id and conv_id:
            return f"{channel_id}:{conv_id}"
    kind = _coerce_str(envelope.get("kind")) or "unknown"
    return f"unknown:{kind}"


def _session_key_from_envelope(envelope: JsonDict) -> Optional[str]:
    payload = envelope.get("payload")
    context = envelope.get("context")
    if isinstance(payload, dict):
        v = _extract_first(payload, ("sessionKey", "session_key"))
        if v:
            return v
    if isinstance(context, dict):
        v = _extract_first(context, ("sessionKey", "session_key"))
        if v:
            return v
    return None


def _channel_from_envelope(envelope: JsonDict) -> Optional[str]:
    payload = envelope.get("payload")
    context = envelope.get("context")
    if isinstance(payload, dict):
        v = _extract_first(payload, ("channel", "messageProvider", "channelId"))
        if v:
            return v
    if isinstance(context, dict):
        v = _extract_first(context, ("channelId", "channel"))
        if v:
            return v
    return None


@dataclass
class LangfuseEventRouter:
    enabled: bool
    trace_manager: Optional[TraceManager]
    # Key: "{session_key}:{agent_id}:{tool_name}" (without trace_seed)
    # Value: deque of (local_id, start_ts_int, session_key, agent_id, trace_seed)
    _pending_tools: Dict[str, Deque[Tuple[str, int, Optional[str], Optional[str], str]]] = field(default_factory=dict)
    # after_tool_call doesn't include toolCallId, and tool_result_persist may be
    # absent for some tools/results. Buffer after_tool_call completions briefly
    # so tool_result_persist can "win" when it does arrive, otherwise we end
    # the tool span from after_tool_call after a short delay.
    #
    # Key matches the before_tool_call key form for tool hooks without toolCallId:
    #   "k:{sessionKey}:{agentId}:{toolName}"
    # Value: deque of (end_ts_int, result, error, duration_ms, trace_seed_at_end)
    _pending_tool_completions: Dict[str, Deque[Tuple[int, Any, Optional[str], Optional[float], str]]] = field(
        default_factory=dict
    )
    _tool_completion_flush_delay_ms: int = 750
    _seen_generation_keys: set[str] = field(default_factory=set)
    _gen_seq: Dict[str, int] = field(default_factory=dict)
    # Route all events for a session into one Langfuse trace seed.
    _conv_active_seed: Dict[str, str] = field(default_factory=dict)  # conv_key -> trace_seed
    _agent_active_seed: Dict[str, str] = field(default_factory=dict)  # agentId -> trace_seed
    _session_active_seed: Dict[str, str] = field(default_factory=dict)  # sessionKey -> trace_seed
    # When prompts don't include conversation ids (common for session-start
    # synthetic messages), bind the first subsequent message on the same
    # channel to the most recent seed.
    _channel_last_seed: Dict[str, Tuple[str, int]] = field(default_factory=dict)  # channel -> (seed, ts_ms)

    _channel_bind_window_ms: int = 120_000  # 2 minutes
    persistence_path: Optional[Path] = None

    def _flush_tool_completions(self, *, now_ms: int) -> None:
        """
        Best-effort: finalize tool spans based on hook.after_tool_call when
        hook.tool_result_persist never arrives.
        """
        if not self._pending_tool_completions:
            return
        if not self.trace_manager:
            return

        delay = self._tool_completion_flush_delay_ms
        # Iterate over a snapshot of keys so we can mutate dicts safely.
        for key in list(self._pending_tool_completions.keys()):
            cq = self._pending_tool_completions.get(key)
            if not cq:
                self._pending_tool_completions.pop(key, None)
                continue

            sq = self._pending_tools.get(key)
            if not sq:
                # No matching start; drop stale completions to avoid unbounded growth.
                while cq and now_ms - cq[0][0] >= max(delay, 30_000):
                    cq.popleft()
                if not cq:
                    self._pending_tool_completions.pop(key, None)
                continue

            while cq and sq and now_ms - cq[0][0] >= delay:
                end_ts_int, result, err, duration_evt, trace_seed_end = cq.popleft()
                local_id, start_ts_int, _sk, _aid, trace_seed_start = sq.popleft()
                effective_trace_seed = trace_seed_start or trace_seed_end

                duration_ms: Optional[float] = duration_evt if isinstance(duration_evt, (int, float)) else None
                if duration_ms is None and isinstance(end_ts_int, int) and isinstance(start_ts_int, int) and end_ts_int >= start_ts_int:
                    duration_ms = float(end_ts_int - start_ts_int)

                self.trace_manager.end_tool_span(
                    effective_trace_seed,
                    tool_call_id=local_id,
                    output_data=result,
                    error=err,
                    duration_ms=duration_ms,
                    input_data=None,
                    extra_metadata={"endedBy": "hook.after_tool_call"},
                )

            if not cq:
                self._pending_tool_completions.pop(key, None)
            if sq is not None and not sq:
                # Keep dict tidy.
                self._pending_tools.pop(key, None)

    @staticmethod
    def create(*, public_key: Optional[str], secret_key: Optional[str], base_url: str, persistence_path: Optional[Path] = None) -> "LangfuseEventRouter":
        if Langfuse is None or not public_key or not secret_key:
            return LangfuseEventRouter(enabled=False, trace_manager=None)
        client = Langfuse(public_key=public_key, secret_key=secret_key, base_url=base_url, flush_at=15, flush_interval=10)
        router = LangfuseEventRouter(enabled=True, trace_manager=TraceManager(client), persistence_path=persistence_path)
        router._load_state()
        return router

    def _load_state(self) -> None:
        if not self.persistence_path or not self.persistence_path.exists():
            return
        try:
            import json
            data = json.loads(self.persistence_path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                self._conv_active_seed = data.get("conv", {})
                self._agent_active_seed = data.get("agent", {})
                self._session_active_seed = data.get("session", {})
                raw_gen = data.get("genSeq", {})
                if isinstance(raw_gen, dict):
                    # Keep only sane numeric values; keys are trace_seeds.
                    for k, v in raw_gen.items():
                        if isinstance(k, str) and isinstance(v, int) and v >= 0:
                            self._gen_seq[k] = v
                raw_ch = data.get("channel", {})
                for k, v in raw_ch.items():
                    if isinstance(v, list) and len(v) == 2:
                        self._channel_last_seed[k] = (v[0], v[1])
        except Exception:
            pass

    def _save_state(self) -> None:
        if not self.persistence_path:
            return
        try:
            import json
            # Prevent unbounded growth: keep genSeq only for seeds that are still
            # reachable from our routing maps (plus recent channel bindings).
            active_seeds: set[str] = set()
            active_seeds.update(v for v in self._conv_active_seed.values() if isinstance(v, str))
            active_seeds.update(v for v in self._agent_active_seed.values() if isinstance(v, str))
            active_seeds.update(v for v in self._session_active_seed.values() if isinstance(v, str))
            active_seeds.update(seed for (seed, _ts) in self._channel_last_seed.values() if isinstance(seed, str))
            gen_seq_out = {k: v for k, v in self._gen_seq.items() if k in active_seeds and isinstance(v, int) and v >= 0}
            data = {
                "conv": self._conv_active_seed,
                "agent": self._agent_active_seed,
                "session": self._session_active_seed,
                "channel": self._channel_last_seed,
                "genSeq": gen_seq_out,
            }
            self.persistence_path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        except Exception:
            pass

    def handle(self, envelope: JsonDict) -> None:
        if not self.enabled or not self.trace_manager:
            return

        kind = _coerce_str(envelope.get("kind")) or "unknown"
        # Finalize any tool spans that completed via after_tool_call but never
        # produced tool_result_persist.
        try:
            self._flush_tool_completions(now_ms=_now_ms())
        except Exception:
            pass

        # For hook.tool_result_persist, try to use stored trace_seed from before_tool_call
        # to ensure consistency even if queued messages changed the routing context
        if kind == "hook.tool_result_persist":
            session_key2 = _extract_session_key(envelope)
            agent_id = _extract_agent_id(envelope)
            tool_name = _extract_tool_name(envelope) or "unknown_tool"
            tool_call_id2 = _extract_tool_call_id(envelope)
            key_id = f"id:{tool_call_id2}" if tool_call_id2 else None
            key_k = f"k:{session_key2}:{agent_id}:{tool_name}"
            q = self._pending_tools.get(key_id) if key_id else None
            if (not q) and tool_call_id2:
                # before_tool_call often lacks toolCallId, so we need to peek
                # the "k:" queue for the stored trace seed.
                q = self._pending_tools.get(key_k)
            if q and len(q) > 0:
                # Peek to get stored trace_seed without popping yet
                stored_trace_seed = q[0][4]
                trace_seed = stored_trace_seed
            else:
                trace_seed = self._route_trace_seed(envelope)
        else:
            trace_seed = self._route_trace_seed(envelope)

        session_key = _session_key_from_envelope(envelope)
        channel = _channel_from_envelope(envelope)
        state = self.trace_manager.get_or_create_trace(trace_seed=trace_seed, session_key=session_key, channel=channel)

        payload = envelope.get("payload") if isinstance(envelope.get("payload"), dict) else {}

        # Hook-derived spans/generations for parity with OpenClaw build traces.
        # These are emitted even when no diagnostic events are present.
        if kind.startswith("hook."):
            session_key2 = _extract_session_key(envelope)
            agent_id = _extract_agent_id(envelope)

            if kind == "hook.message_received" and isinstance(payload, dict):
                msg_text = _message_received_text(payload)
                if _is_reset_command(msg_text):
                    # Ensure the trace has at least one SPAN so Langfuse UI shows Graph.
                    self.trace_manager.record_instant_span(
                        trace_seed,
                        name="session:reset",
                        metadata={
                            "command": (msg_text or "").strip().split(maxsplit=1)[0] if isinstance(msg_text, str) else None,
                            "channel": _channel_from_any(envelope),
                            "sessionKey": session_key2,
                            "agentId": agent_id,
                        },
                        status_message="reset",
                        level="DEFAULT",
                    )
                # Continue to default event recording below.

            if kind == "hook.before_agent_start" and isinstance(payload, dict):
                # Treat as entering processing/run.
                self.trace_manager.record_event(
                    trace_seed,
                    name="session_state:processing",
                    level="DEFAULT",
                    status_message=None,
                    metadata={"agentId": agent_id, "sessionKey": session_key2, "startedAt": _iso_from_ts_ms(envelope.get("ts"))},
                )
                self.trace_manager.start_run_span(
                    trace_seed,
                    name="session:processing",
                    metadata={"agentId": agent_id, "sessionKey": session_key2, "prompt": payload.get("prompt")},
                )
                return

            if kind == "hook.agent_end" and isinstance(payload, dict):
                # Ensure any pending after_tool_call completions are flushed at end of run.
                try:
                    self._flush_tool_completions(now_ms=_now_ms() + self._tool_completion_flush_delay_ms)
                except Exception:
                    pass
                # Record ONLY the user-visible generation (avoid redundant internal calls).
                # Heuristic: pick the last assistant message that looks like a real LLM output,
                # preferring non-openclaw provider and non-delivery-mirror.
                messages = payload.get("messages")
                msg_list = messages if isinstance(messages, list) else []
                assistant_idxs: list[int] = []
                for i, m in enumerate(msg_list):
                    if not isinstance(m, dict):
                        continue
                    if _coerce_str(m.get("role")) != "assistant":
                        continue
                    if not (_coerce_str(m.get("model")) or _coerce_str(m.get("provider")) or isinstance(m.get("usage"), dict)):
                        continue
                    assistant_idxs.append(i)

                picked_idx: Optional[int] = None
                for i in reversed(assistant_idxs):
                    m = msg_list[i]
                    if not isinstance(m, dict):
                        continue
                    model = _coerce_str(m.get("model"))
                    provider_raw = _coerce_str(m.get("provider")) or _coerce_str(m.get("api"))
                    if provider_raw == "openclaw" or model == "delivery-mirror":
                        continue
                    picked_idx = i
                    break
                if picked_idx is None and assistant_idxs:
                    picked_idx = assistant_idxs[-1]

                if picked_idx is not None:
                    msg = msg_list[picked_idx]
                    if isinstance(msg, dict):
                        ts = msg.get("timestamp")
                        model = _coerce_str(msg.get("model"))
                        provider = _coerce_str(msg.get("provider")) or _coerce_str(msg.get("api"))
                        usage = msg.get("usage") if isinstance(msg.get("usage"), dict) else {}
                        cost = usage.get("cost") if isinstance(usage.get("cost"), dict) else {}
                        out_text = _assistant_text(msg)
                        in_text = _generation_input_for_index(msg_list, picked_idx) or _coerce_str(payload.get("prompt"))

                        # Deduplicate across replays.
                        gen_key = f"{trace_seed}:{ts}:{model}:{provider}:{usage.get('input')}:{usage.get('output')}:{usage.get('total') or usage.get('totalTokens')}"
                        if gen_key not in self._seen_generation_keys:
                            self._seen_generation_keys.add(gen_key)
                            seq = self._gen_seq.get(trace_seed, 0) + 1
                            self._gen_seq[trace_seed] = seq
                            # Persist generation sequence so restarts continue incrementing.
                            # This avoids repeated names like model_call:3 after container restarts.
                            try:
                                self._save_state()
                            except Exception:
                                pass
                            self.trace_manager.record_generation(
                                trace_seed,
                                name=f"model_call:{seq}",
                                model=model,
                                provider=provider,
                                input_text=in_text,
                                output_text=out_text,
                                usage=usage,
                                cost_usd=cost.get("total") if isinstance(cost.get("total"), (int, float)) else None,
                                duration_ms=usage.get("durationMs") if isinstance(usage.get("durationMs"), (int, float)) else None,
                                has_user_visible_reply=True,
                                cost_details=cost if isinstance(cost, dict) else None,
                            )

                self.trace_manager.record_event(
                    trace_seed,
                    name="session_state:idle",
                    level="DEFAULT",
                    status_message=None,
                    metadata={"agentId": agent_id, "sessionKey": session_key2, "endedAt": _iso_from_ts_ms(envelope.get("ts"))},
                )
                self.trace_manager.end_run_span(trace_seed)
                return

            if kind == "hook.before_tool_call":
                tool_name = _extract_tool_name(envelope) or "unknown_tool"
                tool_call_id = _extract_tool_call_id(envelope)
                ts_ms = envelope.get("ts")
                ts_int = int(ts_ms) if isinstance(ts_ms, (int, float)) else 0
                local_id = tool_call_id or f"hook:{session_key2 or 'no_session'}:{agent_id or 'no_agent'}:{tool_name}:{ts_int}"

                # Key without trace_seed so we can match on tool_result_persist
                # even if routing context changes due to queued messages
                key = f"id:{tool_call_id}" if tool_call_id else f"k:{session_key2}:{agent_id}:{tool_name}"
                q = self._pending_tools.get(key)
                if q is None:
                    q = deque()
                    self._pending_tools[key] = q
                # Store trace_seed so we can use it in tool_result_persist
                q.append((local_id, ts_int, session_key2, agent_id, trace_seed))

                self.trace_manager.start_tool_span(
                    trace_seed,
                    tool_call_id=local_id,
                    tool_name=tool_name,
                    input_data=_extract_tool_input(envelope),
                    metadata={"toolCallId": tool_call_id} if tool_call_id else None,
                )
                return

            if kind == "hook.after_tool_call" and isinstance(payload, dict):
                # This hook has result/error/duration but no toolCallId. Buffer it briefly:
                # if tool_result_persist arrives, it wins; otherwise we end the span using this.
                tool_name = _extract_tool_name(envelope) or "unknown_tool"
                end_ts_ms = envelope.get("ts")
                end_ts_int = int(end_ts_ms) if isinstance(end_ts_ms, (int, float)) else _now_ms()
                result = payload.get("result")
                err = _coerce_str(payload.get("error"))
                duration_evt = payload.get("durationMs") if isinstance(payload.get("durationMs"), (int, float)) else None

                key_k = f"k:{session_key2}:{agent_id}:{tool_name}"
                cq = self._pending_tool_completions.get(key_k)
                if cq is None:
                    cq = deque()
                    self._pending_tool_completions[key_k] = cq
                cq.append((end_ts_int, result, err, duration_evt, trace_seed))
                return

            if kind == "hook.tool_result_persist":
                tool_name = _extract_tool_name(envelope) or "unknown_tool"
                tool_call_id = _extract_tool_call_id(envelope)
                out, err = _extract_tool_output_from_result_persist(envelope)

                # Key without trace_seed to match pending tools
                key_id = f"id:{tool_call_id}" if tool_call_id else None
                key_k = f"k:{session_key2}:{agent_id}:{tool_name}"
                local_id: Optional[str] = None
                start_ts_int: Optional[int] = None
                stored_trace_seed: Optional[str] = None
                q = self._pending_tools.get(key_id) if key_id else None
                if (not q) and tool_call_id:
                    # before_tool_call usually doesn't have toolCallId, so starts are queued under "k:".
                    q = self._pending_tools.get(key_k)
                if q:
                    local_id, start_ts_int, _sk, _aid, stored_trace_seed = q.popleft()
                    # If after_tool_call already fired, discard its completion entry (tool_result_persist wins).
                    cq = self._pending_tool_completions.get(key_k)
                    if cq and len(cq) > 0:
                        cq.popleft()

                # Use stored trace_seed from before_tool_call to ensure consistency
                # even if queued messages changed the routing context
                effective_trace_seed = stored_trace_seed or trace_seed

                if local_id is None:
                    ts_ms = envelope.get("ts")
                    ts_int = int(ts_ms) if isinstance(ts_ms, (int, float)) else 0
                    local_id = tool_call_id or f"hook:{session_key2 or 'no_session'}:{agent_id or 'no_agent'}:{tool_name}:{ts_int}"
                    # Start span best-effort (if we missed the start hook).
                    self.trace_manager.start_tool_span(
                        effective_trace_seed,
                        tool_call_id=local_id,
                        tool_name=tool_name,
                        input_data=None,
                        metadata={"toolCallId": tool_call_id, "missedStart": True},
                    )

                end_ts = envelope.get("ts")
                end_ts_int = int(end_ts) if isinstance(end_ts, (int, float)) else None
                duration_ms: Optional[float] = None
                if start_ts_int is not None and end_ts_int is not None and end_ts_int >= start_ts_int:
                    duration_ms = float(end_ts_int - start_ts_int)

                self.trace_manager.end_tool_span(
                    effective_trace_seed,
                    tool_call_id=local_id,
                    output_data=out,
                    error=err,
                    duration_ms=duration_ms,
                    input_data=None,
                    extra_metadata={"toolCallId": tool_call_id},
                )
                return

        if kind == "diagnostic" and isinstance(payload, dict):
            evt_type = _coerce_str(payload.get("type")) or "unknown"

            if evt_type == "session.state":
                state_val = _coerce_str(payload.get("state"))
                reason = _coerce_str(payload.get("reason"))
                if state_val == "processing":
                    self.trace_manager.start_run_span(
                        trace_seed,
                        name="session:processing",
                        metadata={"reason": reason, "queueDepth": payload.get("queueDepth")},
                    )
                    return
                if state_val in ("idle", "waiting"):
                    self.trace_manager.end_run_span(trace_seed)
                    return
                if state_val == "ended":
                    self.trace_manager.record_event(
                        trace_seed,
                        name="session_state:end",
                        level="DEFAULT",
                        status_message=reason,
                        metadata={"reason": reason},
                    )
                    self.trace_manager.end_trace(trace_seed, reason=reason)
                    self.trace_manager.flush()
                    return

            if evt_type == "tool.start":
                tool_name = _coerce_str(payload.get("toolName")) or "unknown_tool"
                tool_call_id = _coerce_str(payload.get("toolCallId")) or f"unknown:{payload.get('seq')}"
                self.trace_manager.start_tool_span(
                    trace_seed,
                    tool_call_id=tool_call_id,
                    tool_name=tool_name,
                    input_data=payload.get("input"),
                    metadata={
                        "runId": payload.get("runId"),
                        "toolCallId": payload.get("toolCallId"),
                        "sessionKey": payload.get("sessionKey"),
                    },
                )
                return

            if evt_type == "tool.end":
                tool_name = _coerce_str(payload.get("toolName")) or "unknown_tool"
                tool_call_id = _coerce_str(payload.get("toolCallId")) or f"unknown:{payload.get('seq')}"
                self.trace_manager.end_tool_span(
                    trace_seed,
                    tool_call_id=tool_call_id,
                    output_data=payload.get("output"),
                    error=_coerce_str(payload.get("error")),
                    duration_ms=payload.get("durationMs") if isinstance(payload.get("durationMs"), (int, float)) else None,
                    input_data=payload.get("input"),
                )
                return

            if evt_type == "model.usage":
                usage = payload.get("usage") if isinstance(payload.get("usage"), dict) else {}
                seq = payload.get("seq")
                name = f"model_call:{seq}" if isinstance(seq, int) else "model_call"
                # Avoid redundant internal model calls; agent_end handler already
                # records the user-visible reply.
                has_user_visible = payload.get("hasUserVisibleReply")
                if has_user_visible is True:
                    self.trace_manager.record_generation(
                        trace_seed,
                        name=name,
                        model=_coerce_str(payload.get("model")),
                        provider=_coerce_str(payload.get("provider")),
                        input_text=_coerce_str(payload.get("inputText")),
                        output_text=_coerce_str(payload.get("outputText")),
                        usage=usage,
                        cost_usd=payload.get("costUsd") if isinstance(payload.get("costUsd"), (int, float)) else None,
                        duration_ms=payload.get("durationMs") if isinstance(payload.get("durationMs"), (int, float)) else None,
                        has_user_visible_reply=True,
                        cost_details=payload.get("cost") if isinstance(payload.get("cost"), dict) else None,
                    )
                return

            if evt_type == "internal.log":
                subsystem = _coerce_str(payload.get("subsystem")) or "runtime"
                msg = _coerce_str(payload.get("message"))
                level = _coerce_str(payload.get("level")) or "info"
                lf_level = "ERROR" if level in ("fatal", "error") else "WARNING" if level == "warn" else "DEFAULT"
                self.trace_manager.record_event(
                    trace_seed,
                    name=f"log:{subsystem}",
                    level=lf_level,
                    status_message=msg,
                    metadata={"subsystem": subsystem, "level": level, "meta": payload.get("meta")},
                )
                return

            if evt_type == "llm.error":
                self.trace_manager.record_event(
                    trace_seed,
                    name="llm_error",
                    level="ERROR",
                    status_message=_coerce_str(payload.get("errorMessage")),
                    metadata={
                        "provider": payload.get("provider"),
                        "model": payload.get("model"),
                        "statusCode": payload.get("statusCode"),
                        "errorType": payload.get("errorType"),
                        "fallbackAttempted": payload.get("fallbackAttempted"),
                    },
                )
                return

            if evt_type == "diagnostic.heartbeat":
                self.trace_manager.flush()
                return

            return

        # Non-diagnostic: record as a Langfuse event for visibility.
        # This includes hook-level interception objects.
        self.trace_manager.record_event(
            trace_seed,
            name=kind,
            level="DEFAULT",
            status_message=None,
            metadata={"context": envelope.get("context"), "payload": envelope.get("payload")},
        )

    def _route_trace_seed(self, envelope: JsonDict) -> str:
        """
        Ensure message/agent/tool hooks for the same session land in one trace.
        /new or /reset will rotate to a new trace seed.
        """
        kind = _coerce_str(envelope.get("kind")) or "unknown"
        ctx = envelope.get("context") if isinstance(envelope.get("context"), dict) else {}
        payload = envelope.get("payload") if isinstance(envelope.get("payload"), dict) else {}
        channel = _channel_from_any(envelope)
        now_ms = _now_ms()

        # Message-based seed: create/rotate per conversation.
        if kind == "hook.message_received":
            conv_key = _conv_key_from_message_envelope(envelope)
            sess = _session_key_from_envelope(envelope)
            msg_text = _message_received_text(payload)
            if conv_key or channel:
                if _is_reset_command(msg_text):
                    old_seed = self._conv_active_seed.get(conv_key) if conv_key else None
                    
                    if old_seed:
                        self.trace_manager.record_event(
                            old_seed,
                            name="session_state:idle",
                            level="DEFAULT",
                            status_message="reset",
                            metadata={"reason": "reset"},
                        )
                        self.trace_manager.end_trace(old_seed, reason="reset")

                    seed = _new_uuid()
                    if conv_key:
                        self._conv_active_seed[conv_key] = seed
                    if channel:
                        self._channel_last_seed[channel] = (seed, now_ms)
                    
                    # Re-bind any session/agent keys that pointed to the old seed
                    if old_seed:
                        for k, v in list(self._session_active_seed.items()):
                            if v == old_seed:
                                self._session_active_seed[k] = seed
                        for k, v in list(self._agent_active_seed.items()):
                            if v == old_seed:
                                self._agent_active_seed[k] = seed

                    self._save_state()
                    return seed
                
                seed = self._conv_active_seed.get(conv_key) if conv_key else None
                # If we already have a sessionKey binding, prefer it over channel heuristics.
                if not seed and sess and sess in self._session_active_seed:
                    seed = self._session_active_seed[sess]
                    if conv_key:
                        self._conv_active_seed[conv_key] = seed
                if not seed:
                    # If we recently started a run on this channel but couldn't
                    # resolve conversationId there, bind this first real message
                    # into that same trace seed.
                    if channel:
                        prev = self._channel_last_seed.get(channel)
                        if prev and now_ms - prev[1] <= self._channel_bind_window_ms:
                            seed = prev[0]
                            if conv_key:
                                self._conv_active_seed[conv_key] = seed
                        else:
                            seed = _new_uuid()
                            if conv_key:
                                self._conv_active_seed[conv_key] = seed
                            else:
                                # No conversation key; at least keep channel continuity.
                                self._channel_last_seed[channel] = (seed, now_ms)
                    else:
                        seed = _new_uuid()
                        if conv_key:
                            self._conv_active_seed[conv_key] = seed
                self._save_state()
                return seed

        # Agent start binds agent/sessionKey to current conversation seed.
        if kind == "hook.before_agent_start":
            agent = _coerce_str(ctx.get("agentId"))
            sess = _coerce_str(ctx.get("sessionKey"))
            conv_key = _conv_key_from_prompt(payload.get("prompt"), preferred_channel=channel)
            prompt_text = _prompt_user_text(payload.get("prompt"))
            # Some surfaces issue /reset or /new without emitting hook.message_received.
            # Rotate the trace seed here to ensure new traces are created.
            if _is_reset_command(prompt_text):
                old_seed = None
                if conv_key:
                    old_seed = self._conv_active_seed.get(conv_key)
                if not old_seed and sess:
                    old_seed = self._session_active_seed.get(sess)
                if not old_seed and agent:
                    old_seed = self._agent_active_seed.get(agent)
                if not old_seed and channel:
                    prev = self._channel_last_seed.get(channel)
                    old_seed = prev[0] if prev else None

                if old_seed:
                    try:
                        self.trace_manager.record_event(
                            old_seed,
                            name="session_state:idle",
                            level="DEFAULT",
                            status_message="reset",
                            metadata={"reason": "reset"},
                        )
                        self.trace_manager.end_trace(old_seed, reason="reset")
                    except Exception:
                        pass

                seed = _new_uuid()
                if conv_key:
                    self._conv_active_seed[conv_key] = seed
                if agent:
                    self._agent_active_seed[agent] = seed
                if sess:
                    self._session_active_seed[sess] = seed
                if channel:
                    self._channel_last_seed[channel] = (seed, now_ms)
                self._save_state()
                return seed
            seed: Optional[str] = None
            if conv_key:
                seed = self._conv_active_seed.get(conv_key)
            
            # Prioritize existing bindings for session/agent to avoid incorrect channel fallback
            if not seed and sess and sess in self._session_active_seed:
                seed = self._session_active_seed[sess]
            if not seed and agent and agent in self._agent_active_seed:
                seed = self._agent_active_seed[agent]

            # If prompt doesn't encode conv and no binding, bind to most recent seed for this channel.
            # Avoid fallback if we have a sessionKey or agentId (implies distinct run) to prevent
            # merging concurrent sessions/runs on the same channel.
            if not seed and channel and not sess and not agent:
                prev = self._channel_last_seed.get(channel)
                if prev and now_ms - prev[1] <= self._channel_bind_window_ms:
                    seed = prev[0]
            
            if not seed:
                # fallback: create new or use (already checked) session binding
                if sess and sess in self._session_active_seed:
                    seed = self._session_active_seed[sess]
                else:
                    seed = _new_uuid()
            if conv_key:
                self._conv_active_seed[conv_key] = seed
            if agent:
                self._agent_active_seed[agent] = seed
            if sess:
                self._session_active_seed[sess] = seed
            if channel:
                self._channel_last_seed[channel] = (seed, now_ms)
            self._save_state()
            return seed

        # Tool hooks: resolve from bound sessionKey/agentId.
        if kind in ("hook.before_tool_call", "hook.tool_result_persist", "hook.after_tool_call"):
            agent = _coerce_str(ctx.get("agentId"))
            sess = _coerce_str(ctx.get("sessionKey"))
            seed = (sess and self._session_active_seed.get(sess)) or (agent and self._agent_active_seed.get(agent))
            if seed:
                return seed
            # If a tool hook arrives before we saw hook.before_agent_start,
            # provisionally bind agent/session to a new seed so we don't leak
            # tool spans into an unrelated active trace (common with queued work).
            if sess or agent:
                seed = _new_uuid()
                if sess:
                    self._session_active_seed[sess] = seed
                if agent:
                    self._agent_active_seed[agent] = seed
                if channel:
                    self._channel_last_seed[channel] = (seed, now_ms)
                self._save_state()
                return seed

        # Agent end: resolve from binding, else fallback.
        if kind == "hook.agent_end":
            agent = _coerce_str(ctx.get("agentId"))
            sess = _coerce_str(ctx.get("sessionKey"))
            seed = (sess and self._session_active_seed.get(sess)) or (agent and self._agent_active_seed.get(agent))
            return seed or _trace_seed_from_envelope(envelope)

        # Fallback to old behavior for diagnostics/unknowns.
        return _trace_seed_from_envelope(envelope)
