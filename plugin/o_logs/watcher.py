from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterator

from ..shared.ipc import tail_jsonl


JsonDict = Dict[str, Any]


def watch_ipc(ipc_file: Path) -> Iterator[JsonDict]:
    yield from tail_jsonl(ipc_file)

