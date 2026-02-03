from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class Envelope:
    kind: str
    ts: int
    source: Optional[str]
    stateDir: Optional[str]
    openclawConfigDir: Optional[str]
    context: Optional[Dict[str, Any]]
    payload: Optional[Dict[str, Any]]

