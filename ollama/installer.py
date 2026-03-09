"""Ollama 一鍵安裝器。

流程：
1. 檢查 Ollama 是否已安裝
2. macOS: brew install 或直接下載 Ollama.app
   Linux: 官方 install script
3. 啟動 Ollama 服務
4. 拉取指定模型
"""

from __future__ import annotations

import logging
import os
import platform
import shutil
import subprocess
import time
from pathlib import Path
from typing import Callable

log = logging.getLogger(__name__)

_DEFAULT_MODEL = "phi4-mini"
_OLLAMA_API = "http://localhost:11434"
_OLLAMA_MAC_ZIP = "https://github.com/ollama/ollama/releases/latest/download/Ollama-darwin.zip"


class OllamaInstallerError(Exception):
    """安裝過程中的不可恢復錯誤。"""


class OllamaInstaller:
    """Ollama 安裝與模型管理。"""

    def __init__(self, model: str | None = None) -> None:
        self._model = model or os.environ.get("OLLAMA_RAG_MODEL", _DEFAULT_MODEL)

    @property
    def model(self) -> str:
        return self._model

    # ── 狀態檢查 ────────────────────────────────────────────

    @staticmethod
    def _find_ollama_bin() -> str | None:
        """Find ollama binary, including macOS .app bundle."""
        found = shutil.which("ollama")
        if found:
            return found
        # macOS: check inside Ollama.app
        app_bin = Path("/Applications/Ollama.app/Contents/Resources/ollama")
        if app_bin.exists() and os.access(app_bin, os.X_OK):
            return str(app_bin)
        return None

    def is_ollama_installed(self) -> bool:
        return self._find_ollama_bin() is not None

    def _ollama_cmd(self) -> str:
        """Return path to ollama binary."""
        return self._find_ollama_bin() or "ollama"

    def is_ollama_running(self) -> bool:
        try:
            import httpx
            resp = httpx.get(f"{_OLLAMA_API}/api/tags", timeout=3.0)
            return resp.status_code == 200
        except Exception:
            return False

    def is_model_pulled(self) -> bool:
        try:
            import httpx
            resp = httpx.get(f"{_OLLAMA_API}/api/tags", timeout=5.0)
            if resp.status_code != 200:
                return False
            models = resp.json().get("models", [])
            for m in models:
                name = m.get("name", "")
                # ollama returns "phi4-mini:latest" — match base name
                if name.split(":")[0] == self._model.split(":")[0]:
                    return True
            return False
        except Exception:
            return False

    def is_fully_ready(self) -> bool:
        return self.is_ollama_installed() and self.is_ollama_running() and self.is_model_pulled()

    def installation_status(self) -> dict[str, bool]:
        installed = self.is_ollama_installed()
        running = self.is_ollama_running() if installed else False
        pulled = self.is_model_pulled() if running else False
        return {
            "ollama_installed": installed,
            "ollama_running": running,
            "model_pulled": pulled,
            "model": self._model,
            "fully_ready": installed and running and pulled,
        }

    # ── 完整安裝流程 ────────────────────────────────────────

    def install(
        self,
        *,
        progress_callback: Callable[[str, float], None] | None = None,
    ) -> None:
        """執行完整安裝：安裝 Ollama → 啟動 → 拉模型。"""
        cb = progress_callback or (lambda desc, pct: None)

        # 1. 安裝 Ollama
        cb("檢查 Ollama...", 0.0)
        if not self.is_ollama_installed():
            cb("正在安裝 Ollama...", 0.05)
            self._install_ollama()
        cb("Ollama 已安裝", 0.2)

        # 2. 啟動服務
        if not self.is_ollama_running():
            cb("正在啟動 Ollama 服務...", 0.25)
            self._start_ollama()
        cb("Ollama 服務就緒", 0.3)

        # 3. 拉取模型
        if not self.is_model_pulled():
            cb(f"正在下載 {self._model}（首次約 2-3GB）...", 0.35)
            self._pull_model()
        cb(f"{self._model} 就緒", 0.95)

        cb("安裝完成", 1.0)
        log.info("Ollama installation complete: %s", self.installation_status())

    # ── 子步驟 ──────────────────────────────────────────────

    def _install_ollama(self) -> None:
        system = platform.system()

        if system == "Darwin":
            self._install_macos()
        elif system == "Linux":
            self._install_linux()
        else:
            raise OllamaInstallerError(
                "請手動安裝 Ollama: https://ollama.com/download"
            )

        if not self.is_ollama_installed():
            raise OllamaInstallerError("Ollama 安裝後仍不可用")

    def _install_macos(self) -> None:
        # 優先 brew（較快且自動配 PATH）
        if shutil.which("brew") is not None:
            log.info("Installing Ollama via Homebrew...")
            result = subprocess.run(
                ["brew", "install", "ollama"],
                capture_output=True, text=True, timeout=300,
            )
            if result.returncode == 0 and self.is_ollama_installed():
                return
            log.warning("brew install ollama failed, falling back to direct download")

        # 無 brew 或 brew 失敗：下載 Ollama.app
        log.info("Downloading Ollama.app from GitHub Releases...")
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            zip_path = Path(tmpdir) / "Ollama-darwin.zip"
            self._download_file(_OLLAMA_MAC_ZIP, zip_path)

            # 解壓到 /Applications
            log.info("Extracting Ollama.app to /Applications...")
            result = subprocess.run(
                ["unzip", "-o", "-q", str(zip_path), "-d", "/Applications"],
                capture_output=True, text=True, timeout=120,
            )
            if result.returncode != 0:
                raise OllamaInstallerError(
                    f"解壓 Ollama.app 失敗:\n{result.stderr}"
                )

        # Ollama.app 內含 CLI，建立 symlink 到 PATH
        app_bin = Path("/Applications/Ollama.app/Contents/Resources/ollama")
        if not app_bin.exists():
            raise OllamaInstallerError(
                f"Ollama.app 解壓成功但找不到 CLI: {app_bin}"
            )

        # 嘗試 symlink 到 /usr/local/bin（不需 sudo）
        link_target = Path("/usr/local/bin/ollama")
        try:
            link_target.parent.mkdir(parents=True, exist_ok=True)
            link_target.unlink(missing_ok=True)
            link_target.symlink_to(app_bin)
            log.info("Symlinked ollama CLI to %s", link_target)
        except OSError:
            # /usr/local/bin 不可寫，不建 symlink，靠 _find_ollama_bin 找到
            log.warning("Cannot symlink to %s, will use app bundle path directly", link_target)

    def _install_linux(self) -> None:
        log.info("Installing Ollama via official install script...")
        result = subprocess.run(
            ["sh", "-c", "curl -fsSL https://ollama.com/install.sh | sh"],
            capture_output=True, text=True, timeout=300,
        )
        if result.returncode != 0:
            raise OllamaInstallerError(
                f"Ollama install script 失敗:\n{result.stderr}"
            )

    @staticmethod
    def _download_file(url: str, target: Path) -> None:
        """下載檔案到指定路徑。"""
        try:
            import httpx
            with httpx.Client(follow_redirects=True, timeout=120.0) as client:
                with client.stream("GET", url) as resp:
                    resp.raise_for_status()
                    with open(target, "wb") as f:
                        for chunk in resp.iter_bytes():
                            if chunk:
                                f.write(chunk)
            log.info("Downloaded: %s (%d MB)", target.name, target.stat().st_size // (1024 * 1024))
        except Exception as exc:
            target.unlink(missing_ok=True)
            raise OllamaInstallerError(f"下載失敗 ({url}): {exc}") from exc

    def _start_ollama(self) -> None:
        ollama = self._ollama_cmd()
        log.info("Starting Ollama service (%s)...", ollama)

        # macOS Ollama.app: 用 open 啟動（會自動在背景跑 serve）
        app_path = Path("/Applications/Ollama.app")
        if app_path.exists() and "Ollama.app" in ollama:
            subprocess.Popen(
                ["open", "-a", "Ollama"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        else:
            subprocess.Popen(
                [ollama, "serve"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )

        # Wait for service to be ready
        for i in range(30):
            time.sleep(1)
            if self.is_ollama_running():
                log.info("Ollama service ready")
                return

        raise OllamaInstallerError(
            "Ollama 服務啟動逾時（30秒）。請手動執行 `ollama serve`"
        )

    def _pull_model(self) -> None:
        ollama = self._ollama_cmd()
        log.info("Pulling model: %s", self._model)
        result = subprocess.run(
            [ollama, "pull", self._model],
            capture_output=True, text=True, timeout=600,
        )
        if result.returncode != 0:
            raise OllamaInstallerError(
                f"ollama pull {self._model} 失敗:\n{result.stderr}"
            )

        if not self.is_model_pulled():
            raise OllamaInstallerError(
                f"模型 {self._model} 下載後驗證失敗"
            )
        log.info("Model pulled successfully: %s", self._model)


def ensure_ollama_ready(
    model: str | None = None,
    progress_callback: Callable[[str, float], None] | None = None,
) -> None:
    """確保 Ollama + 模型就緒。供 onboarding 呼叫。"""
    installer = OllamaInstaller(model=model)
    if installer.is_fully_ready():
        if progress_callback:
            progress_callback("已就緒", 1.0)
        return
    installer.install(progress_callback=progress_callback)
