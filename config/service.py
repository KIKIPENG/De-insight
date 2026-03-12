"""ConfigService: single source of truth for runtime config."""

from __future__ import annotations

import os
import threading
from pathlib import Path

from .store_env import DotEnvStore

_service: ConfigService | None = None
_service_lock = threading.Lock()


class ConfigService:
    """Thread-safe runtime config facade.

    Precedence:
    1) runtime overrides
    2) .env file
    3) process environment
    4) provided default
    """

    def __init__(self, env_path: Path | None = None) -> None:
        root = Path(__file__).resolve().parent.parent
        self._store = DotEnvStore(env_path or (root / ".env"))
        self._lock = threading.Lock()
        self._runtime_overrides: dict[str, str] = {}
        self._file_env: dict[str, str] = {}
        self._exported_keys: set[str] = set()
        self.reload()

    def reload(self) -> None:
        with self._lock:
            self._file_env = self._store.read()

    def get(self, key: str, default: str = "") -> str:
        with self._lock:
            if key in self._runtime_overrides:
                return self._runtime_overrides[key]
            if key in self._file_env:
                return self._file_env[key]
        return os.environ.get(key, default)

    def snapshot(self, include_process: bool = False) -> dict[str, str]:
        with self._lock:
            out: dict[str, str] = {}
            if include_process:
                out.update(os.environ)
            out.update(self._file_env)
            out.update(self._runtime_overrides)
            return out

    def replace_env(self, env: dict[str, str]) -> None:
        """Replace persisted .env content with provided key-values."""
        with self._lock:
            fresh = {k: v for k, v in env.items() if v}
            self._store.write(fresh)
            self._file_env = fresh

    def update_env(self, updates: dict[str, str]) -> None:
        """Persist updates to .env and reload in-memory snapshot."""
        with self._lock:
            current = dict(self._file_env)
            for k, v in updates.items():
                if v:
                    current[k] = v
                else:
                    current.pop(k, None)
            self._store.write(current)
            self._file_env = current

    def set_runtime_overrides(self, updates: dict[str, str]) -> None:
        with self._lock:
            self._runtime_overrides.update(updates)

    def clear_runtime_overrides(self) -> None:
        with self._lock:
            self._runtime_overrides.clear()

    def export_to_environ(self, keys: list[str] | None = None) -> None:
        """Copy selected resolved keys into process env for legacy clients."""
        with self._lock:
            # Resolved config for export: file + runtime overrides.
            # Do not use process env as source to avoid carrying stale values.
            resolved: dict[str, str] = dict(self._file_env)
            resolved.update(self._runtime_overrides)

            if keys is None:
                # Also include previously exported keys so removed entries are unset.
                target_keys = set(resolved.keys()) | set(self._exported_keys)
            else:
                target_keys = set(keys)

            self._exported_keys.update(target_keys)

            for key in target_keys:
                val = resolved.get(key, "")
                if val:
                    os.environ[key] = val
                else:
                    os.environ.pop(key, None)

    def validate(self) -> list[str]:
        """Return non-fatal configuration issues."""
        cfg = self.snapshot(include_process=True)
        issues: list[str] = []

        llm_model = cfg.get("LLM_MODEL", "")
        if not llm_model:
            issues.append("LLM_MODEL 未設定")
        elif llm_model.startswith(("openai/", "codex/")) and not (
            cfg.get("OPENAI_API_KEY") or cfg.get("CODEX_API_KEY") or cfg.get("OPENROUTER_API_KEY")
        ):
            issues.append("LLM_MODEL 需要 OPENAI_API_KEY/CODEX_API_KEY/OPENROUTER_API_KEY")
        elif llm_model.startswith("gemini/") and not cfg.get("GOOGLE_API_KEY"):
            issues.append("gemini 模型需要 GOOGLE_API_KEY")

        embed_provider = (cfg.get("EMBED_PROVIDER", "") or "openrouter").lower()
        if embed_provider == "openrouter" and not (
            cfg.get("EMBED_API_KEY") or cfg.get("OPENROUTER_API_KEY")
        ):
            issues.append("EMBED_PROVIDER=openrouter 但 OPENROUTER_API_KEY 未設定")

        rag_model = cfg.get("RAG_LLM_MODEL", "")
        if rag_model and rag_model.startswith(("gemini", "google/")) and not (
            cfg.get("RAG_API_KEY") or cfg.get("GOOGLE_API_KEY")
        ):
            issues.append("RAG_LLM_MODEL=gemini* 但 RAG_API_KEY/GOOGLE_API_KEY 未設定")

        vision_model = cfg.get("VISION_MODEL", "")
        if vision_model.startswith(("gemini/", "google/")) and not (
            cfg.get("VISION_API_KEY") or cfg.get("GOOGLE_API_KEY")
        ):
            issues.append("VISION_MODEL=gemini* 但 VISION_API_KEY/GOOGLE_API_KEY 未設定")

        return issues


def get_config_service() -> ConfigService:
    global _service
    if _service is not None:
        return _service
    with _service_lock:
        if _service is None:
            _service = ConfigService()
        return _service


def reset_config_service() -> None:
    global _service
    with _service_lock:
        _service = None
