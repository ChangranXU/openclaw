from __future__ import annotations

import argparse
import os
import signal
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

from o_langfuse import LangfuseEventRouter
from o_logs import LogManager
from shared.config import load_settings
from shared.ipc import tail_jsonl


JsonDict = Dict[str, Any]


def _resolve_plugin_dir() -> Path:
    raw = (os.environ.get("O_OBSERVABILITY_PLUGIN_DIR") or "").strip()
    if raw:
        return Path(raw).expanduser().resolve()
    return Path(__file__).parent.resolve()


class _StopFlag:
    def __init__(self) -> None:
        self.value = False


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument("--ipc", required=True, help="Path to JSONL IPC file")
    args = parser.parse_args(argv)

    plugin_dir = _resolve_plugin_dir()
    settings = load_settings(plugin_dir)

    ipc_file = Path(args.ipc).expanduser().resolve()
    ipc_file.parent.mkdir(parents=True, exist_ok=True)
    ipc_file.open("a", encoding="utf-8").close()

    logs_dir = (settings.openclaw_config_dir / "logs-build").resolve()
    logs_dir.mkdir(parents=True, exist_ok=True)

    log_mgr = LogManager(logs_dir)
    langfuse_router = LangfuseEventRouter.create(
        public_key=settings.langfuse.public_key,
        secret_key=settings.langfuse.secret_key,
        base_url=settings.langfuse.base_url,
        persistence_path=(
            settings.openclaw_config_dir / "langfuse_state.json"
        ).resolve(),
    )
    if not langfuse_router.enabled:
        pk_set = bool(settings.langfuse.public_key)
        sk_set = bool(settings.langfuse.secret_key)
        print(
            "[o-observability] Langfuse disabled "
            f"(public_key_set={pk_set} secret_key_set={sk_set} base_url={settings.langfuse.base_url})",
            file=sys.stderr,
        )
    else:
        print(
            f"[o-observability] Langfuse enabled (base_url={settings.langfuse.base_url})",
            file=sys.stderr,
        )

    stop = _StopFlag()

    def _handle_sig(_signum: int, _frame: object) -> None:
        stop.value = True

    signal.signal(signal.SIGTERM, _handle_sig)
    signal.signal(signal.SIGINT, _handle_sig)

    last_flush = time.time()
    flush_interval_s = 1.0
    last_langfuse_error = 0.0

    for envelope in tail_jsonl(
        ipc_file, start_at_end=False, state_file=ipc_file.with_suffix(".tailstate.json")
    ):
        if stop.value:
            break
        if not isinstance(envelope, dict):
            continue
        try:
            log_mgr.write_event(envelope)
        except Exception:
            pass
        try:
            langfuse_router.handle(envelope)
        except Exception as err:
            # Don't spam logs; surface errors at most once per 10s so we can
            # diagnose missing credentials / network / API issues.
            now = time.time()
            if now - last_langfuse_error >= 10.0:
                print(f"[o-observability] Langfuse error: {err!r}", file=sys.stderr)
                last_langfuse_error = now

        now = time.time()
        if now - last_flush >= flush_interval_s:
            try:
                if langfuse_router.trace_manager:
                    langfuse_router.trace_manager.flush()
            except Exception:
                pass
            last_flush = now

    try:
        if langfuse_router.trace_manager:
            langfuse_router.trace_manager.flush()
    except Exception:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
