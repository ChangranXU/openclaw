from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, Optional


JsonDict = Dict[str, Any]


@dataclass
class TailState:
    offset: int = 0
    inode: Optional[int] = None


def _stat_inode(path: Path) -> Optional[int]:
    try:
        return path.stat().st_ino
    except Exception:
        return None


def _load_tail_state(state_file: Path) -> Optional[TailState]:
    try:
        raw = json.loads(state_file.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            return None
        offset = raw.get("offset")
        inode = raw.get("inode")
        s = TailState()
        if isinstance(offset, int) and offset >= 0:
            s.offset = offset
        if isinstance(inode, int):
            s.inode = inode
        return s
    except Exception:
        return None


def _save_tail_state(state_file: Path, state: TailState) -> None:
    try:
        state_file.parent.mkdir(parents=True, exist_ok=True)
        state_file.write_text(
            json.dumps({"offset": state.offset, "inode": state.inode}), encoding="utf-8"
        )
    except Exception:
        pass


def tail_jsonl(
    path: Path,
    *,
    poll_interval_s: float = 0.2,
    start_at_end: bool = True,
    state_file: Optional[Path] = None,
) -> Iterator[JsonDict]:
    """
    Yields parsed JSON objects from a JSONL file, following appends.

    - Handles truncation (offset reset) and simple rotation (inode change).
    - Best-effort: invalid JSON lines are skipped.
    """
    state = TailState()
    if state_file:
        loaded = _load_tail_state(state_file)
        if loaded:
            state = loaded
    while True:
        try:
            if not path.exists():
                time.sleep(poll_interval_s)
                continue

            inode = _stat_inode(path)
            if state.inode is None:
                state.inode = inode
                # Default behavior: don't replay the entire IPC backlog when the
                # daemon starts (prevents duplicated Langfuse/log output on
                # restarts). Set start_at_end=False to replay from start.
                if start_at_end and state.offset == 0:
                    try:
                        state.offset = path.stat().st_size
                    except Exception:
                        state.offset = 0
            elif inode is not None and state.inode is not None and inode != state.inode:
                # File rotated/replaced
                state.inode = inode
                state.offset = 0

            size = path.stat().st_size
            if size < state.offset:
                # Truncated
                state.offset = 0

            # IMPORTANT: Use binary mode + readline() to avoid subtle issues with
            # mixing text iteration and tell()/seek() on some Python builds /
            # file systems (can break tailing and yield nothing).
            #
            # We track offsets in bytes (binary tell()) and decode per-line.
            any_line = False
            with path.open("rb") as f:
                f.seek(state.offset)
                while True:
                    line = f.readline()
                    if not line:
                        break
                    any_line = True
                    state.offset = f.tell()
                    raw = line.decode("utf-8", errors="replace").strip()
                    if not raw:
                        continue
                    try:
                        obj = json.loads(raw)
                        if isinstance(obj, dict):
                            yield obj
                    except Exception:
                        continue

            if state_file:
                _save_tail_state(state_file, state)

            if not any_line:
                time.sleep(poll_interval_s)
        except Exception:
            time.sleep(poll_interval_s)
