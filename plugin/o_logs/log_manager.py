from __future__ import annotations

import datetime
import hashlib
import json
import os
import re
import threading
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


JsonDict = Dict[str, Any]


def _safe_json_dumps(obj: Any, pretty: bool = False) -> str:
    try:
        if pretty:
            return json.dumps(obj, ensure_ascii=False, indent=2, default=str)
        return json.dumps(obj, ensure_ascii=False, separators=(",", ":"), default=str)
    except Exception:
        fallback = {"unserializable": True, "repr": repr(obj)}
        if pretty:
            return json.dumps(fallback, ensure_ascii=False, indent=2)
        return json.dumps(fallback, ensure_ascii=False, separators=(",", ":"))


def _coerce_str(val: Any) -> Optional[str]:
    if isinstance(val, str):
        s = val.strip()
        return s if s else None
    return None


def _extract_binding(obj: Any) -> Dict[str, Optional[str]]:
    if not isinstance(obj, dict):
        return {"sessionId": None, "sessionKey": None, "runId": None, "channel": None}

    return {
        "sessionId": _coerce_str(obj.get("sessionId") or obj.get("session_id")),
        "sessionKey": _coerce_str(obj.get("sessionKey") or obj.get("session_key")),
        "runId": _coerce_str(obj.get("runId") or obj.get("run_id")),
        "channel": _coerce_str(obj.get("channel") or obj.get("messageProvider") or obj.get("channelId")),
    }


def _resolve_trace_key(envelope: JsonDict) -> str:
    payload = envelope.get("payload")
    context = envelope.get("context")
    binding = _extract_binding(payload if isinstance(payload, dict) else {})
    if binding.get("sessionId"):
        return binding["sessionId"]  # type: ignore[return-value]
    if binding.get("sessionKey"):
        return binding["sessionKey"]  # type: ignore[return-value]

    # Hooks may carry sessionKey/sessionId in context (tool hooks, agent hooks, session hooks)
    binding2 = _extract_binding(context if isinstance(context, dict) else {})
    if binding2.get("sessionId"):
        return binding2["sessionId"]  # type: ignore[return-value]
    if binding2.get("sessionKey"):
        return binding2["sessionKey"]  # type: ignore[return-value]

    # Message hooks only have channel + conversationId typically; keep stable-ish grouping.
    kind = _coerce_str(envelope.get("kind")) or "unknown"
    if isinstance(context, dict):
        channel_id = _coerce_str(context.get("channelId")) or _coerce_str(context.get("channel"))
        conv_id = _coerce_str(context.get("conversationId")) or _coerce_str(context.get("chatId"))
        if channel_id and conv_id:
            return f"{channel_id}:{conv_id}"
        if channel_id:
            return f"{channel_id}:{kind}"
    return f"unknown:{kind}"


@dataclass
class TraceDoc:
    trace_id: str
    file_path: Path
    created_ts_ms: int
    updated_ts_ms: int
    started_ts_ms: Optional[int] = None
    ended_ts_ms: Optional[int] = None
    # For "similar to Langfuse export"
    trace: Dict[str, Any] = field(default_factory=dict)
    observations: list[Dict[str, Any]] = field(default_factory=list)
    # Runtime helpers
    last_event_sig: Optional[str] = None
    gen_seq: int = 0
    seen_generation_keys: set[str] = field(default_factory=set)
    run_start_ts_ms: Optional[int] = None
    pending_tools: Dict[str, list[Dict[str, Any]]] = field(default_factory=dict)  # key -> FIFO list


def _iso_from_ts_ms(ts_ms: Any) -> Optional[str]:
    if not isinstance(ts_ms, (int, float)):
        return None
    try:
        return datetime.datetime.fromtimestamp(float(ts_ms) / 1000.0, tz=datetime.timezone.utc).isoformat()
    except Exception:
        return None


def _new_uuid() -> str:
    return str(uuid.uuid4())


def _now_ms() -> int:
    return int(datetime.datetime.now(tz=datetime.timezone.utc).timestamp() * 1000)


def _observation_id() -> str:
    return _new_uuid()


def _msg_text_parts(content: Any) -> Optional[str]:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for p in content:
            if isinstance(p, dict):
                t = _coerce_str(p.get("text"))
                if t:
                    parts.append(t)
        return "\n".join(parts) if parts else None
    return None


def _iter_assistant_messages(messages: Any) -> list[JsonDict]:
    if not isinstance(messages, list):
        return []
    out: list[JsonDict] = []
    for m in messages:
        if not isinstance(m, dict):
            continue
        if _coerce_str(m.get("role")) != "assistant":
            continue
        if not (_coerce_str(m.get("model")) or _coerce_str(m.get("provider")) or isinstance(m.get("usage"), dict)):
            continue
        out.append(m)
    return out


def _assistant_text(msg: JsonDict) -> Optional[str]:
    return _msg_text_parts(msg.get("content"))


def _tool_name(envelope: JsonDict) -> Optional[str]:
    ctx = envelope.get("context") if isinstance(envelope.get("context"), dict) else {}
    payload = envelope.get("payload") if isinstance(envelope.get("payload"), dict) else {}
    return _coerce_str(ctx.get("toolName")) or _coerce_str(payload.get("toolName"))


def _session_key(envelope: JsonDict) -> Optional[str]:
    ctx = envelope.get("context") if isinstance(envelope.get("context"), dict) else {}
    payload = envelope.get("payload") if isinstance(envelope.get("payload"), dict) else {}
    return _coerce_str(ctx.get("sessionKey")) or _coerce_str(payload.get("sessionKey"))


def _agent_id(envelope: JsonDict) -> Optional[str]:
    ctx = envelope.get("context") if isinstance(envelope.get("context"), dict) else {}
    return _coerce_str(ctx.get("agentId"))


def _tool_input(envelope: JsonDict) -> Any:
    payload = envelope.get("payload") if isinstance(envelope.get("payload"), dict) else {}
    params = payload.get("params")
    return params if params is not None else payload


def _tool_output(envelope: JsonDict) -> Dict[str, Any]:
    payload = envelope.get("payload") if isinstance(envelope.get("payload"), dict) else {}
    msg = payload.get("message") if isinstance(payload.get("message"), dict) else {}
    out = _msg_text_parts(msg.get("content"))
    if out is None:
        out = msg.get("content")
    tool_call_id = _coerce_str(msg.get("toolCallId"))
    if not tool_call_id and isinstance(envelope.get("context"), dict):
        tool_call_id = _coerce_str(envelope["context"].get("toolCallId"))
    return {"output": out if out is not None else msg, "isError": msg.get("isError") is True, "toolCallId": tool_call_id}

def _tool_call_id(envelope: JsonDict) -> Optional[str]:
    ctx = envelope.get("context") if isinstance(envelope.get("context"), dict) else {}
    payload = envelope.get("payload") if isinstance(envelope.get("payload"), dict) else {}
    # Prefer context.toolCallId; some builds may also put it in payload.
    return _coerce_str(ctx.get("toolCallId")) or _coerce_str(payload.get("toolCallId"))


def _conv_key_from_message(envelope: JsonDict) -> Optional[str]:
    ctx = envelope.get("context") if isinstance(envelope.get("context"), dict) else {}
    channel = _coerce_str(ctx.get("channelId")) or _coerce_str(ctx.get("channel"))
    conv = _coerce_str(ctx.get("conversationId")) or _coerce_str(ctx.get("chatId")) or _coerce_str(ctx.get("threadId"))
    if not channel or not conv:
        return None
    return f"{channel.lower()}:{conv}"


_PROMPT_CONV_RE = re.compile(r"\[([A-Za-z][A-Za-z0-9_-]*)\s+([^\s\]]+)")

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
        # Prefer the channel we already inferred from envelope fields.
        if preferred and channel == preferred:
            return f"{channel}:{conv}"
        # Otherwise take the first known channel header (skips "[Queued messages ...]").
        if channel in _KNOWN_PROMPT_CHANNELS and best is None:
            best = (channel, conv)

    if best:
        return f"{best[0]}:{best[1]}"

    # Fallback: first match (old behavior)
    m0 = _PROMPT_CONV_RE.search(p)
    if not m0:
        return None
    channel = (m0.group(1) or "").strip().lower()
    conv = (m0.group(2) or "").strip()
    return f"{channel}:{conv}" if channel and conv else None


def _is_reset_command(text: Any) -> bool:
    t = _msg_text_parts(text)
    if not t:
        return False
    t = t.strip().lower()
    first = t.split(maxsplit=1)[0] if t else ""
    return first in ("/new", "/reset")


def _message_received_text(payload: Any) -> Optional[str]:
    """
    hook.message_received payload shape varies by channel/build:
    - {"content": "..."} or {"content": [{text:"..."}]}
    - {"message": {"content": "..."}} (common)
    """
    if not isinstance(payload, dict):
        return None
    if "content" in payload:
        return _msg_text_parts(payload.get("content"))
    msg = payload.get("message")
    if isinstance(msg, dict):
        return _msg_text_parts(msg.get("content"))
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

def _compute_gen_key(iso_ts: Optional[str], model: Optional[str], provider: Optional[str], text: Optional[str]) -> str:
    # Use stable hash for text to allow dedup across restarts
    t = text or ""
    h = hashlib.sha256(t.encode("utf-8")).hexdigest()
    return f"{iso_ts}:{model}:{provider}:{h}"


def _is_delivery_mirror(model: Optional[str]) -> bool:
    # "delivery-mirror" is not a real model generation; it's a delivery echo used
    # by the gateway. Including it causes duplicate-looking generations in traces.
    return (model or "").strip().lower() == "delivery-mirror"


class LogManager:
    def __init__(self, base_dir: Path) -> None:
        self._base_dir = base_dir
        # Re-entrant because routing may persist/close traces during reset while
        # already holding the lock in the routing path.
        self._lock = threading.RLock()
        # Active trace routing
        self._conv_active_trace: Dict[str, str] = {}  # conv_key -> trace_id
        self._agent_active_trace: Dict[str, str] = {}  # agentId -> trace_id
        self._session_active_trace: Dict[str, str] = {}  # sessionKey -> trace_id
        # Trace cache (trace_id -> doc)
        self._trace_docs: Dict[str, TraceDoc] = {}
        self._channel_last_trace: Dict[str, Tuple[str, int]] = {}  # channel -> (trace_id, ts_ms)
        self._channel_bind_window_ms: int = 120_000
        self._trace_map_file = (base_dir / "trace_map.json").resolve()
        self._load_trace_map()

    def _load_trace_map(self) -> None:
        if not self._trace_map_file.exists():
            return
        try:
            data = json.loads(self._trace_map_file.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                self._conv_active_trace = data.get("conv", {})
                self._agent_active_trace = data.get("agent", {})
                self._session_active_trace = data.get("session", {})
                # Restore channel_last_trace (stored as list [tid, ts])
                raw_ch = data.get("channel", {})
                for k, v in raw_ch.items():
                    if isinstance(v, list) and len(v) == 2:
                        self._channel_last_trace[k] = (v[0], v[1])
        except Exception:
            pass

    def _save_trace_map(self) -> None:
        try:
            data = {
                "conv": self._conv_active_trace,
                "agent": self._agent_active_trace,
                "session": self._session_active_trace,
                "channel": self._channel_last_trace,
            }
            self._trace_map_file.write_text(_safe_json_dumps(data), encoding="utf-8")
        except Exception:
            pass

    def write_event(self, envelope: JsonDict) -> None:
        trace_id = self._route_trace_id(envelope)
        doc = self._get_or_create_doc(trace_id)
        self._apply_envelope(doc, envelope)
        self._persist(doc)

    def _trace_file_path(self, trace_id: str) -> Path:
        return (self._base_dir / f"trace-{trace_id}.json").resolve()

    def _load_doc_from_disk(self, trace_id: str) -> Optional[TraceDoc]:
        fp = self._trace_file_path(trace_id)
        if not fp.exists():
            return None
        try:
            raw = json.loads(fp.read_text(encoding="utf-8"))
            if not isinstance(raw, dict):
                return None
            trace = raw.get("trace") if isinstance(raw.get("trace"), dict) else {}
            observations = raw.get("observations") if isinstance(raw.get("observations"), list) else []
            created_at = trace.get("createdAt")
            updated_at = trace.get("updatedAt")
            created_ms = _now_ms()
            updated_ms = _now_ms()
            if isinstance(created_at, str):
                created_ms = _now_ms()
            if isinstance(updated_at, str):
                updated_ms = _now_ms()
            doc = TraceDoc(
                trace_id=trace_id,
                file_path=fp,
                created_ts_ms=created_ms,
                updated_ts_ms=updated_ms,
                trace=trace,
                observations=[o for o in observations if isinstance(o, dict)],
            )
            # Best-effort: recover seq to avoid name collisions
            doc.gen_seq = len([o for o in doc.observations if o.get("type") == "GENERATION"])

            # Repopulate seen_generation_keys to prevent duplicates on restart
            for o in doc.observations:
                if o.get("type") == "GENERATION":
                    try:
                        meta = json.loads(o.get("metadata") or "{}")
                        prov = meta.get("provider")
                        key = _compute_gen_key(
                            o.get("startTime"),
                            o.get("model"),
                            prov,
                            o.get("output")
                        )
                        doc.seen_generation_keys.add(key)
                    except Exception:
                        pass

            return doc
        except Exception:
            return None

    def _get_or_create_doc(self, trace_id: str) -> TraceDoc:
        with self._lock:
            existing = self._trace_docs.get(trace_id)
            if existing:
                return existing
            loaded = self._load_doc_from_disk(trace_id)
            if loaded:
                self._trace_docs[trace_id] = loaded
                return loaded

            now = _now_ms()
            fp = self._trace_file_path(trace_id)
            trace = {
                "id": trace_id,
                "name": f"openclaw_session_{trace_id[:8]}",
                "timestamp": _iso_from_ts_ms(now),
                "environment": "default",
                "release": "unknown",
                "sessionId": trace_id,
                "input": None,
                "output": None,
                "createdAt": _iso_from_ts_ms(now),
                "updatedAt": _iso_from_ts_ms(now),
                "metadata": json.dumps({"startedAt": _iso_from_ts_ms(now)}, ensure_ascii=False),
            }
            doc = TraceDoc(trace_id=trace_id, file_path=fp, created_ts_ms=now, updated_ts_ms=now, trace=trace, observations=[])
            self._trace_docs[trace_id] = doc
            return doc

    def _route_trace_id(self, envelope: JsonDict) -> str:
        kind = _coerce_str(envelope.get("kind")) or "unknown"
        ctx = envelope.get("context") if isinstance(envelope.get("context"), dict) else {}
        payload = envelope.get("payload") if isinstance(envelope.get("payload"), dict) else {}
        now_ms = _now_ms()
        channel = (_coerce_str(ctx.get("channelId")) or _coerce_str(ctx.get("messageProvider")) or "").strip().lower() or None
        if not channel:
            meta = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}
            channel = (_coerce_str(meta.get("provider")) or _coerce_str(meta.get("surface")) or "").strip().lower() or None
        if not channel:
            # queued prompt fallback
            convp = _coerce_str(payload.get("prompt"))
            if convp:
                # Prefer a known channel header if present; avoid queue wrappers.
                for m in _PROMPT_CONV_RE.finditer(convp):
                    ch = (m.group(1) or "").strip().lower()
                    if ch in _KNOWN_PROMPT_CHANNELS:
                        channel = ch
                        break
                if not channel:
                    m0 = _PROMPT_CONV_RE.search(convp)
                    if m0:
                        channel = (m0.group(1) or "").strip().lower() or None

        # 1) Message-based routing: create/rotate trace on /new or /reset.
        if kind == "hook.message_received":
            conv_key = _conv_key_from_message(envelope)
            content = _message_received_text(payload)
            if conv_key or channel:
                with self._lock:
                    if _is_reset_command(content):
                        old_tid = self._conv_active_trace.get(conv_key) if conv_key else None
                        
                        # Close old trace if open
                        if old_tid:
                            try:
                                old_doc = self._get_or_create_doc(old_tid)
                                iso = _iso_from_ts_ms(envelope.get("ts"))
                                if old_doc.run_start_ts_ms is not None:
                                    # Synthetic close of processing span
                                    self._add_observation(
                                        old_doc,
                                        {
                                            "id": _observation_id(),
                                            "traceId": old_tid,
                                            "parentObservationId": None,
                                            "type": "SPAN",
                                            "name": "session:processing",
                                            "startTime": _iso_from_ts_ms(old_doc.run_start_ts_ms),
                                            "endTime": iso,
                                            "latency": 0,  # Unknown
                                            "level": "DEFAULT",
                                            "statusMessage": "reset",
                                            "metadata": json.dumps({"reason": "reset"}, ensure_ascii=False),
                                        },
                                    )
                                    old_doc.run_start_ts_ms = None
                                
                                # Add idle event
                                self._add_observation(
                                    old_doc,
                                    {
                                        "id": _observation_id(),
                                        "traceId": old_tid,
                                        "parentObservationId": None,
                                        "type": "EVENT",
                                        "name": "session_state:idle",
                                        "startTime": iso,
                                        "endTime": None,
                                        "level": "DEFAULT",
                                        "statusMessage": "reset",
                                        "metadata": json.dumps({"reason": "reset"}, ensure_ascii=False),
                                    },
                                )
                                self._persist(old_doc)
                            except Exception:
                                pass

                        tid = _new_uuid()
                        if conv_key:
                            self._conv_active_trace[conv_key] = tid
                        if channel:
                            self._channel_last_trace[channel] = (tid, now_ms)
                        
                        # Re-bind any session/agent keys that pointed to the old trace
                        # to the new trace, so queued items (agent start) land in new trace.
                        if old_tid:
                            for k, v in list(self._session_active_trace.items()):
                                if v == old_tid:
                                    self._session_active_trace[k] = tid
                            for k, v in list(self._agent_active_trace.items()):
                                if v == old_tid:
                                    self._agent_active_trace[k] = tid
                        
                        self._save_trace_map()
                        return tid
                    
                    tid = self._conv_active_trace.get(conv_key) if conv_key else None
                    # If we already have a sessionKey binding, prefer it over channel heuristics.
                    sess = _session_key(envelope)
                    if not tid and sess and sess in self._session_active_trace:
                        tid = self._session_active_trace[sess]
                        if conv_key:
                            self._conv_active_trace[conv_key] = tid
                    if not tid:
                        # Bind first real message to most recent channel trace if available.
                        if channel:
                            prev = self._channel_last_trace.get(channel)
                            if prev and now_ms - prev[1] <= self._channel_bind_window_ms:
                                tid = prev[0]
                                if conv_key:
                                    self._conv_active_trace[conv_key] = tid
                            else:
                                tid = _new_uuid()
                                if conv_key:
                                    self._conv_active_trace[conv_key] = tid
                                else:
                                    # No conversation key; at least keep channel continuity.
                                    self._channel_last_trace[channel] = (tid, now_ms)
                        else:
                            tid = _new_uuid()
                            if conv_key:
                                self._conv_active_trace[conv_key] = tid
                        self._save_trace_map()
                    return tid

        # 2) Agent start binds agent/sessionKey to current conversation trace.
        if kind == "hook.before_agent_start":
            agent = _coerce_str(ctx.get("agentId"))
            sess = _coerce_str(ctx.get("sessionKey"))
            conv_key = _conv_key_from_prompt(payload.get("prompt"), preferred_channel=channel)
            with self._lock:
                tid: Optional[str] = None
                if conv_key:
                    tid = self._conv_active_trace.get(conv_key)

                prompt_text = _prompt_user_text(payload.get("prompt"))
                if _is_reset_command(prompt_text):
                    old_tid: Optional[str] = None
                    if conv_key:
                        old_tid = self._conv_active_trace.get(conv_key)
                    if not old_tid and sess:
                        old_tid = self._session_active_trace.get(sess)
                    if not old_tid and agent:
                        old_tid = self._agent_active_trace.get(agent)
                    if not old_tid and channel:
                        prev = self._channel_last_trace.get(channel)
                        old_tid = prev[0] if prev else None

                    # Best-effort close old trace doc (mirrors message_received reset).
                    if old_tid:
                        try:
                            old_doc = self._get_or_create_doc(old_tid)
                            iso = _iso_from_ts_ms(envelope.get("ts"))
                            if old_doc.run_start_ts_ms is not None:
                                self._add_observation(
                                    old_doc,
                                    {
                                        "id": _observation_id(),
                                        "traceId": old_tid,
                                        "parentObservationId": None,
                                        "type": "SPAN",
                                        "name": "session:processing",
                                        "startTime": _iso_from_ts_ms(old_doc.run_start_ts_ms),
                                        "endTime": iso,
                                        "latency": 0,
                                        "level": "DEFAULT",
                                        "statusMessage": "reset",
                                        "metadata": json.dumps({"reason": "reset"}, ensure_ascii=False),
                                    },
                                )
                                old_doc.run_start_ts_ms = None
                            self._add_observation(
                                old_doc,
                                {
                                    "id": _observation_id(),
                                    "traceId": old_tid,
                                    "parentObservationId": None,
                                    "type": "EVENT",
                                    "name": "session_state:idle",
                                    "startTime": iso,
                                    "endTime": None,
                                    "level": "DEFAULT",
                                    "statusMessage": "reset",
                                    "metadata": json.dumps({"reason": "reset"}, ensure_ascii=False),
                                },
                            )
                            self._persist(old_doc)
                        except Exception:
                            pass

                    tid = _new_uuid()
                    if conv_key:
                        self._conv_active_trace[conv_key] = tid
                    if agent:
                        self._agent_active_trace[agent] = tid
                    if sess:
                        self._session_active_trace[sess] = tid
                    if channel:
                        self._channel_last_trace[channel] = (tid, now_ms)
                    self._save_trace_map()
                    return tid
                
                # Prioritize existing bindings for session/agent to avoid incorrect channel fallback
                if not tid and sess and sess in self._session_active_trace:
                    tid = self._session_active_trace[sess]
                if not tid and agent and agent in self._agent_active_trace:
                    tid = self._agent_active_trace[agent]
                
                # Avoid fallback if we have a sessionKey or agentId (implies distinct run) to prevent
                # merging concurrent sessions/runs on the same channel.
                if not tid and channel and not sess and not agent:
                    prev = self._channel_last_trace.get(channel)
                    if prev and now_ms - prev[1] <= self._channel_bind_window_ms:
                        tid = prev[0]
                if not tid:
                    # fallback: keep stable-ish binding by sessionKey
                    if sess and sess in self._session_active_trace:
                        tid = self._session_active_trace[sess]
                    else:
                        tid = _new_uuid()
                if conv_key:
                    self._conv_active_trace[conv_key] = tid
                if agent:
                    self._agent_active_trace[agent] = tid
                if sess:
                    self._session_active_trace[sess] = tid
                if channel:
                    self._channel_last_trace[channel] = (tid, now_ms)
                self._save_trace_map()
                return tid

        # 3) Agent end: resolve via binding, then clear.
        if kind == "hook.agent_end":
            agent = _coerce_str(ctx.get("agentId"))
            sess = _coerce_str(ctx.get("sessionKey"))
            with self._lock:
                tid = (sess and self._session_active_trace.get(sess)) or (agent and self._agent_active_trace.get(agent))
                if not tid:
                    tid = _new_uuid()
                return tid

        # 4) Tool hooks: route to bound trace if possible.
        if kind in ("hook.before_tool_call", "hook.tool_result_persist", "hook.after_tool_call"):
            agent = _coerce_str(ctx.get("agentId"))
            sess = _coerce_str(ctx.get("sessionKey"))
            with self._lock:
                tid = (sess and self._session_active_trace.get(sess)) or (agent and self._agent_active_trace.get(agent))
                if tid:
                    return tid
                # If a tool hook arrives before hook.before_agent_start, provisionally
                # bind agent/session to a new trace to avoid leaking tool spans into an
                # unrelated active trace (common with queued work).
                if sess or agent:
                    tid = _new_uuid()
                    if sess:
                        self._session_active_trace[sess] = tid
                    if agent:
                        self._agent_active_trace[agent] = tid
                    if channel:
                        self._channel_last_trace[channel] = (tid, now_ms)
                    self._save_trace_map()
                    return tid

        # 5) Fallback: keep old trace key behavior (but map into a new trace file).
        fallback_key = _resolve_trace_key(envelope)
        with self._lock:
            existing = self._session_active_trace.get(fallback_key) or self._conv_active_trace.get(fallback_key)
            if existing:
                return existing
            tid = _new_uuid()
            # Remember fallback to reduce fragmentation for non-sessionized streams.
            self._session_active_trace[fallback_key] = tid
            self._save_trace_map()
            return tid

    def _persist(self, doc: TraceDoc) -> None:
        doc.updated_ts_ms = _now_ms()
        doc.trace["updatedAt"] = _iso_from_ts_ms(doc.updated_ts_ms)

        meta: Dict[str, Any] = {}
        try:
            existing_meta = json.loads(doc.trace.get("metadata") or "{}")
            if isinstance(existing_meta, dict):
                meta.update(existing_meta)
        except Exception:
            meta = {}

        if doc.started_ts_ms is not None:
            meta["startedAt"] = _iso_from_ts_ms(doc.started_ts_ms)
        if doc.ended_ts_ms is not None:
            meta["endedAt"] = _iso_from_ts_ms(doc.ended_ts_ms)
        if doc.started_ts_ms is not None and doc.ended_ts_ms is not None and doc.ended_ts_ms >= doc.started_ts_ms:
            meta["durationMs"] = int(doc.ended_ts_ms - doc.started_ts_ms)
        doc.trace["metadata"] = json.dumps(meta, ensure_ascii=False, default=str)

        payload = {"trace": doc.trace, "observations": doc.observations}
        try:
            doc.file_path.parent.mkdir(parents=True, exist_ok=True)
            doc.file_path.write_text(_safe_json_dumps(payload, pretty=True), encoding="utf-8")
        except Exception:
            pass

    def _add_observation(self, doc: TraceDoc, obs: Dict[str, Any]) -> None:
        if doc.started_ts_ms is None:
            # Try infer from obs.startTime if possible
            st = obs.get("startTime")
            if isinstance(st, str):
                # best-effort: keep trace timestamp as first observation time
                doc.started_ts_ms = doc.started_ts_ms or doc.created_ts_ms
        doc.observations.append(obs)

    def _apply_envelope(self, doc: TraceDoc, envelope: JsonDict) -> None:
        kind = _coerce_str(envelope.get("kind")) or "unknown"
        ts_ms = envelope.get("ts")
        ts_int = int(ts_ms) if isinstance(ts_ms, (int, float)) else None
        iso = _iso_from_ts_ms(ts_ms)

        # Adjacent duplicate suppression (keeps file stable on tail replays).
        try:
            sig = _safe_json_dumps({"kind": kind, "ts": ts_ms, "context": envelope.get("context"), "payload": envelope.get("payload")}, pretty=False)
            if doc.last_event_sig == sig:
                return
            doc.last_event_sig = sig
        except Exception:
            pass

        ctx = envelope.get("context") if isinstance(envelope.get("context"), dict) else {}
        payload = envelope.get("payload") if isinstance(envelope.get("payload"), dict) else {}

        def md(d: Dict[str, Any]) -> str:
            return json.dumps(d, ensure_ascii=False, default=str)

        if kind == "hook.message_received":
            text = payload.get("content")
            self._add_observation(
                doc,
                {
                    "id": _observation_id(),
                    "traceId": doc.trace_id,
                    "parentObservationId": None,
                    "type": "EVENT",
                    "name": "message_received",
                    "startTime": iso,
                    "endTime": None,
                    "level": "DEFAULT",
                    "statusMessage": _coerce_str(text),
                    "metadata": md({"context": ctx, "payloadMeta": payload.get("metadata"), "isReset": _is_reset_command(text)}),
                },
            )
            return

        if kind == "hook.before_agent_start":
            if ts_int is not None:
                doc.run_start_ts_ms = ts_int
                doc.started_ts_ms = doc.started_ts_ms or ts_int
            self._add_observation(
                doc,
                {
                    "id": _observation_id(),
                    "traceId": doc.trace_id,
                    "parentObservationId": None,
                    "type": "EVENT",
                    "name": "session_state:processing",
                    "startTime": iso,
                    "endTime": None,
                    "level": "DEFAULT",
                    "statusMessage": None,
                    "metadata": md({"agentId": _coerce_str(ctx.get("agentId")), "sessionKey": _coerce_str(ctx.get("sessionKey"))}),
                },
            )

            # Generations embedded in messages.
            for msg in _iter_assistant_messages(payload.get("messages")):
                out_text = _assistant_text(msg)
                model = _coerce_str(msg.get("model"))
                provider = _coerce_str(msg.get("provider")) or _coerce_str(msg.get("api"))
                usage = msg.get("usage") if isinstance(msg.get("usage"), dict) else {}
                cost = usage.get("cost") if isinstance(usage.get("cost"), dict) else {}
                msg_ts = msg.get("timestamp")
                msg_iso = _iso_from_ts_ms(msg_ts)
                if _is_delivery_mirror(model):
                    continue
                gen_key = _compute_gen_key(msg_iso, model, provider, out_text)
                if gen_key in doc.seen_generation_keys:
                    continue
                doc.seen_generation_keys.add(gen_key)
                doc.gen_seq += 1
                self._add_observation(
                    doc,
                    {
                        "id": _observation_id(),
                        "traceId": doc.trace_id,
                        "parentObservationId": None,
                        "type": "GENERATION",
                        "name": f"model_call:{doc.gen_seq}",
                        "startTime": msg_iso,
                        "endTime": msg_iso,
                        "level": "DEFAULT",
                        "model": model,
                        "usageDetails": {"input": usage.get("input"), "output": usage.get("output"), "total": usage.get("total") or usage.get("totalTokens")},
                        "costDetails": {"input": cost.get("input"), "output": cost.get("output"), "total": cost.get("total")} if isinstance(cost, dict) else {},
                        "metadata": md({"provider": provider}),
                        "input": None,
                        "output": out_text,
                    },
                )
            return

        if kind == "hook.agent_end":
            if ts_int is not None:
                doc.ended_ts_ms = ts_int
            # Close the run span (session:processing) as a SPAN.
            if doc.run_start_ts_ms is not None and ts_int is not None and ts_int >= doc.run_start_ts_ms:
                self._add_observation(
                    doc,
                    {
                        "id": _observation_id(),
                        "traceId": doc.trace_id,
                        "parentObservationId": None,
                        "type": "SPAN",
                        "name": "session:processing",
                        "startTime": _iso_from_ts_ms(doc.run_start_ts_ms),
                        "endTime": iso,
                        "latency": (ts_int - doc.run_start_ts_ms) / 1000.0,
                        "level": "DEFAULT",
                        "statusMessage": None,
                        "metadata": md({"agentId": _coerce_str(ctx.get("agentId")), "sessionKey": _coerce_str(ctx.get("sessionKey"))}),
                    },
                )
            doc.run_start_ts_ms = None
            self._add_observation(
                doc,
                {
                    "id": _observation_id(),
                    "traceId": doc.trace_id,
                    "parentObservationId": None,
                    "type": "EVENT",
                    "name": "session_state:idle",
                    "startTime": iso,
                    "endTime": None,
                    "level": "DEFAULT",
                    "statusMessage": None,
                    "metadata": md({"agentId": _coerce_str(ctx.get("agentId")), "sessionKey": _coerce_str(ctx.get("sessionKey"))}),
                },
            )

            # Extract any missed generations (deduped).
            for msg in _iter_assistant_messages(payload.get("messages")):
                out_text = _assistant_text(msg)
                model = _coerce_str(msg.get("model"))
                provider = _coerce_str(msg.get("provider")) or _coerce_str(msg.get("api"))
                usage = msg.get("usage") if isinstance(msg.get("usage"), dict) else {}
                cost = usage.get("cost") if isinstance(usage.get("cost"), dict) else {}
                msg_ts = msg.get("timestamp")
                msg_iso = _iso_from_ts_ms(msg_ts)
                if _is_delivery_mirror(model):
                    continue
                gen_key = _compute_gen_key(msg_iso, model, provider, out_text)
                if gen_key in doc.seen_generation_keys:
                    continue
                doc.seen_generation_keys.add(gen_key)
                doc.gen_seq += 1
                self._add_observation(
                    doc,
                    {
                        "id": _observation_id(),
                        "traceId": doc.trace_id,
                        "parentObservationId": None,
                        "type": "GENERATION",
                        "name": f"model_call:{doc.gen_seq}",
                        "startTime": msg_iso,
                        "endTime": msg_iso,
                        "level": "DEFAULT",
                        "model": model,
                        "usageDetails": {"input": usage.get("input"), "output": usage.get("output"), "total": usage.get("total") or usage.get("totalTokens")},
                        "costDetails": {"input": cost.get("input"), "output": cost.get("output"), "total": cost.get("total")} if isinstance(cost, dict) else {},
                        "metadata": md({"provider": provider}),
                        "input": None,
                        "output": out_text,
                    },
                )
            return

        if kind == "hook.before_tool_call":
            tool = _tool_name(envelope) or "unknown_tool"
            sess = _session_key(envelope)
            agent = _agent_id(envelope)
            tcid = _tool_call_id(envelope)
            key = f"id:{tcid}" if tcid else f"k:{sess}:{agent}:{tool}"
            q = doc.pending_tools.setdefault(key, [])
            q.append({"startTs": ts_int, "input": _tool_input(envelope), "toolCallId": tcid})
            return

        if kind == "hook.tool_result_persist":
            tool = _tool_name(envelope) or "unknown_tool"
            sess = _session_key(envelope)
            agent = _agent_id(envelope)
            tcid = _tool_call_id(envelope)
            key = f"id:{tcid}" if tcid else f"k:{sess}:{agent}:{tool}"
            q = doc.pending_tools.get(key) or []
            start = q.pop(0) if q else None
            out = _tool_output(envelope)
            start_ts = start.get("startTs") if isinstance(start, dict) else None
            start_iso = _iso_from_ts_ms(start_ts) if isinstance(start_ts, (int, float)) else None
            latency = (ts_int - int(start_ts)) / 1000.0 if ts_int is not None and isinstance(start_ts, (int, float)) else None
            self._add_observation(
                doc,
                {
                    "id": _observation_id(),
                    "traceId": doc.trace_id,
                    "parentObservationId": None,
                    "type": "SPAN",
                    "name": f"tool:{tool}",
                    "startTime": start_iso,
                    "endTime": iso,
                    "latency": latency,
                    "level": "ERROR" if out.get("isError") else "DEFAULT",
                    "statusMessage": "tool_error" if out.get("isError") else None,
                    "metadata": md({"toolCallId": out.get("toolCallId"), "sessionKey": sess, "agentId": agent}),
                    "input": start.get("input") if isinstance(start, dict) else None,
                    "output": out.get("output"),
                },
            )
            return

        # Default: keep envelope visible as a simple event
        self._add_observation(
            doc,
            {
                "id": _observation_id(),
                "traceId": doc.trace_id,
                "parentObservationId": None,
                "type": "EVENT",
                "name": kind,
                "startTime": iso,
                "endTime": None,
                "level": "DEFAULT",
                "statusMessage": None,
                "metadata": json.dumps({"context": ctx, "payload": payload}, ensure_ascii=False, default=str),
            },
        )
        return

