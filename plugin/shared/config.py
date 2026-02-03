from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional


def _parse_dotenv(path: Path) -> Dict[str, str]:
    if not path.exists() or not path.is_file():
        return {}
    out: Dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, val = line.split("=", 1)
        key = key.strip()
        val = val.strip().strip('"').strip("'")
        if key:
            out[key] = val
    return out


def _home_openclaw_dir() -> Path:
    return Path.home() / ".openclaw"


def resolve_openclaw_config_dir() -> Path:
    raw = (os.environ.get("OPENCLAW_CONFIG_DIR") or "").strip()
    if raw:
        return Path(raw).expanduser()
    return _home_openclaw_dir()


def resolve_openclaw_state_dir() -> Optional[Path]:
    raw = (os.environ.get("OPENCLAW_STATE_DIR") or "").strip()
    if raw:
        return Path(raw).expanduser()
    return None


@dataclass(frozen=True)
class LangfuseSettings:
    public_key: Optional[str]
    secret_key: Optional[str]
    base_url: str


@dataclass(frozen=True)
class Settings:
    plugin_dir: Path
    openclaw_config_dir: Path
    openclaw_state_dir: Optional[Path]
    ipc_file: Optional[Path]
    langfuse: LangfuseSettings
    openclaw_json: Dict[str, Any]


def load_settings(plugin_dir: Path) -> Settings:
    plugin_dir = plugin_dir.resolve()
    openclaw_config_dir = resolve_openclaw_config_dir().resolve()
    openclaw_state_dir = resolve_openclaw_state_dir()
    ipc_raw = (os.environ.get("O_OBSERVABILITY_IPC_FILE") or "").strip()
    ipc_file = Path(ipc_raw).expanduser().resolve() if ipc_raw else None

    env_from_plugin = _parse_dotenv(plugin_dir / ".env")
    public_key = (env_from_plugin.get("LANGFUSE_PUBLIC_KEY") or os.environ.get("LANGFUSE_PUBLIC_KEY") or "").strip()
    secret_key = (env_from_plugin.get("LANGFUSE_SECRET_KEY") or os.environ.get("LANGFUSE_SECRET_KEY") or "").strip()
    base_url = (
        (env_from_plugin.get("LANGFUSE_BASE_URL") or os.environ.get("LANGFUSE_BASE_URL") or "https://cloud.langfuse.com")
        .strip()
        .rstrip("/")
    )

    openclaw_json_path = openclaw_config_dir / "openclaw.json"
    openclaw_json: Dict[str, Any] = {}
    if openclaw_json_path.exists():
        try:
            openclaw_json = json.loads(openclaw_json_path.read_text(encoding="utf-8"))
        except Exception:
            openclaw_json = {}

    return Settings(
        plugin_dir=plugin_dir,
        openclaw_config_dir=openclaw_config_dir,
        openclaw_state_dir=openclaw_state_dir.resolve() if openclaw_state_dir else None,
        ipc_file=ipc_file,
        langfuse=LangfuseSettings(
            public_key=public_key or None,
            secret_key=secret_key or None,
            base_url=base_url,
        ),
        openclaw_json=openclaw_json,
    )

