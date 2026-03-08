"""De-insight TUI — 入口點"""
import atexit
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

BACKEND_HOST = "127.0.0.1"
BACKEND_PORT = 8000
HEALTH_URL = f"http://{BACKEND_HOST}:{BACKEND_PORT}/api/health"
LOG_DIR = Path.home() / ".deinsight" / "logs"


def _is_backend_ready() -> bool:
    try:
        urllib.request.urlopen(HEALTH_URL, timeout=2)
        return True
    except Exception:
        return False


def _ensure_backend() -> subprocess.Popen | None:
    """Start the FastAPI backend if it's not already running."""
    if _is_backend_ready():
        return None

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_file = open(LOG_DIR / "backend.log", "a")

    backend_dir = Path(__file__).resolve().parent / "backend"
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "main:app",
         "--host", BACKEND_HOST, "--port", str(BACKEND_PORT)],
        cwd=str(backend_dir),
        stdout=log_file,
        stderr=log_file,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
    )

    print("[de-insight] starting backend...", file=sys.stderr)

    # Wait up to 30 seconds — first cold start may need time to import heavy deps
    for i in range(60):
        if proc.poll() is not None:
            print(
                f"Backend process exited unexpectedly. See log: {LOG_DIR / 'backend.log'}",
                file=sys.stderr,
            )
            sys.exit(1)
        if _is_backend_ready():
            print("[de-insight] backend ready", file=sys.stderr)
            return proc
        time.sleep(0.5)

    print(
        f"Backend failed to start within timeout. See log: {LOG_DIR / 'backend.log'}",
        file=sys.stderr,
    )
    proc.terminate()
    sys.exit(1)


if __name__ == "__main__":
    backend_proc = _ensure_backend()
    if backend_proc is not None:
        atexit.register(backend_proc.terminate)

    from app import DeInsightApp
    DeInsightApp().run()
