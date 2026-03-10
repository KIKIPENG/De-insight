"""Simple .env file store."""

from __future__ import annotations

from pathlib import Path


class DotEnvStore:
    """Read/write .env key-value pairs without side effects."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def read(self) -> dict[str, str]:
        env: dict[str, str] = {}
        if not self.path.exists():
            return env
        for raw in self.path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            env[k.strip()] = v.strip()
        return env

    def write(self, env: dict[str, str]) -> None:
        lines = [f"{k}={v}" for k, v in env.items() if v]
        self.path.write_text("\n".join(lines) + "\n", encoding="utf-8")

