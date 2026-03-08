"""llama-server 生命週期管理（單例鎖 + PID 追蹤）。

負責：安裝檢查、啟動、健康檢查、關閉。
不重複拉起已在跑的 server。
"""

from __future__ import annotations

import atexit
import logging
import os
import signal
import subprocess
import time
import threading
from pathlib import Path

log = logging.getLogger(__name__)

# ── 預設值 ──────────────────────────────────────────────────────────

_DEFAULT_HOST = "127.0.0.1"
_DEFAULT_PORT = 8999
_HEALTH_TIMEOUT = 30  # 秒
_HEALTH_INTERVAL = 0.5  # 秒


class LlamaServerError(Exception):
    """llama-server 啟動或健康檢查失敗。"""


class LlamaServerManager:
    """管理本機 llama-server 行程（單例）。"""

    _instance: LlamaServerManager | None = None
    _lock = threading.Lock()

    def __new__(cls) -> LlamaServerManager:
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self) -> None:
        if self._initialized:
            return
        self._initialized = True
        self._process: subprocess.Popen | None = None
        self._pid_file: Path | None = None
        self._host = os.environ.get("GGUF_SERVER_HOST", _DEFAULT_HOST)
        self._port = int(os.environ.get("GGUF_SERVER_PORT", str(_DEFAULT_PORT)))
        atexit.register(self.stop)

    @property
    def host(self) -> str:
        return self._host

    @property
    def port(self) -> int:
        return self._port

    @property
    def base_url(self) -> str:
        return f"http://{self._host}:{self._port}"

    @property
    def is_running(self) -> bool:
        """llama-server 是否在跑。"""
        if self._process is not None and self._process.poll() is None:
            return True
        # 檢查 PID file
        return self._check_pid_alive()

    def _pid_file_path(self) -> Path:
        from paths import APP_HOME
        pid_dir = APP_HOME / "gguf"
        pid_dir.mkdir(parents=True, exist_ok=True)
        return pid_dir / "llama-server.pid"

    def _check_pid_alive(self) -> bool:
        """檢查 PID file 記錄的行程是否存活。"""
        pid_path = self._pid_file_path()
        if not pid_path.exists():
            return False
        try:
            pid = int(pid_path.read_text().strip())
            os.kill(pid, 0)  # 0 signal = check existence
            return True
        except (ValueError, OSError):
            pid_path.unlink(missing_ok=True)
            return False

    def _write_pid(self, pid: int) -> None:
        pid_path = self._pid_file_path()
        pid_path.write_text(str(pid))
        self._pid_file = pid_path

    def _clear_pid(self) -> None:
        pid_path = self._pid_file_path()
        pid_path.unlink(missing_ok=True)
        self._pid_file = None

    def find_binary(self) -> Path | None:
        """尋找 llama-server 二進位檔。"""
        from paths import APP_HOME
        # 1) 自行編譯的位置
        built = APP_HOME / "gguf" / "llama.cpp" / "build" / "bin" / "llama-server"
        if built.exists() and os.access(built, os.X_OK):
            return built
        # 2) 系統 PATH
        import shutil
        sys_bin = shutil.which("llama-server")
        if sys_bin:
            return Path(sys_bin)
        return None

    def start(
        self,
        model_path: str | Path,
        mmproj_path: str | Path | None = None,
        *,
        n_gpu_layers: int = 99,
        ctx_size: int = 8192,
        extra_args: list[str] | None = None,
    ) -> None:
        """啟動 llama-server。若已在跑則跳過。"""
        if self.is_running:
            log.info("llama-server already running (pid check)")
            return

        binary = self.find_binary()
        if binary is None:
            raise LlamaServerError(
                "llama-server 二進位檔不存在。請先執行 GGUFInstaller.install() 安裝。"
            )

        model_path = Path(model_path)
        if not model_path.exists():
            raise LlamaServerError(f"模型檔不存在: {model_path}")

        cmd = [
            str(binary),
            "--embedding",
            "-m", str(model_path),
            "--host", self._host,
            "--port", str(self._port),
            "-ngl", str(n_gpu_layers),
            "-c", str(ctx_size),
            "-b", str(ctx_size),    # logical batch = context (embedding 需一次處理完整輸入)
            "-ub", str(ctx_size),   # physical batch = context (預設 512 會導致長文 500)
        ]

        if mmproj_path:
            mmproj_path = Path(mmproj_path)
            if not mmproj_path.exists():
                raise LlamaServerError(f"mmproj 檔不存在: {mmproj_path}")
            cmd.extend(["--mmproj", str(mmproj_path)])

        if extra_args:
            cmd.extend(extra_args)

        log.info("Starting llama-server: %s", " ".join(cmd))

        # 將 stdout/stderr 導到 log 檔
        from paths import APP_HOME
        log_dir = APP_HOME / "gguf" / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / "llama-server.log"

        with open(log_file, "a") as lf:
            self._process = subprocess.Popen(
                cmd,
                stdout=lf,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )

        self._write_pid(self._process.pid)
        log.info("llama-server started (pid=%d), waiting for health...", self._process.pid)

        # 等待健康檢查通過
        if not self._wait_healthy():
            self.stop()
            raise LlamaServerError(
                f"llama-server 未在 {_HEALTH_TIMEOUT}s 內就緒。"
                f" 檢查 log: {log_file}"
            )

        log.info("llama-server healthy at %s", self.base_url)

    def _wait_healthy(self) -> bool:
        """阻塞等待 llama-server /health 回 200。"""
        import httpx

        deadline = time.monotonic() + _HEALTH_TIMEOUT
        while time.monotonic() < deadline:
            # 檢查行程是否提前退出
            if self._process and self._process.poll() is not None:
                log.error("llama-server exited with code %d", self._process.returncode)
                return False
            try:
                resp = httpx.get(f"{self.base_url}/health", timeout=2.0)
                if resp.status_code == 200:
                    return True
            except httpx.ConnectError:
                pass
            except Exception as exc:
                log.debug("Health check error: %s", exc)
            time.sleep(_HEALTH_INTERVAL)
        return False

    def health_check(self) -> bool:
        """即時健康檢查（非阻塞）。"""
        import httpx
        try:
            resp = httpx.get(f"{self.base_url}/health", timeout=3.0)
            return resp.status_code == 200
        except Exception:
            return False

    def stop(self) -> None:
        """關閉 llama-server。"""
        if self._process is not None and self._process.poll() is None:
            log.info("Stopping llama-server (pid=%d)...", self._process.pid)
            try:
                self._process.terminate()
                self._process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                log.warning("llama-server did not terminate, sending SIGKILL")
                self._process.kill()
                self._process.wait(timeout=5)
            except Exception as exc:
                log.error("Error stopping llama-server: %s", exc)
            self._process = None

        # 清理可能殘留的 PID file 行程
        pid_path = self._pid_file_path()
        if pid_path.exists():
            try:
                pid = int(pid_path.read_text().strip())
                os.kill(pid, signal.SIGTERM)
                log.info("Sent SIGTERM to orphan llama-server (pid=%d)", pid)
            except (ValueError, OSError):
                pass

        self._clear_pid()
        log.info("llama-server stopped.")

    def restart(
        self,
        model_path: str | Path,
        mmproj_path: str | Path | None = None,
        **kwargs,
    ) -> None:
        """重啟 llama-server。"""
        self.stop()
        self.start(model_path, mmproj_path, **kwargs)
