"""De-insight TUI — 入口點"""
import asyncio
import atexit
import os
import socket
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


def _is_port_available(port: int) -> bool:
    """Check if port is available by attempting to bind to it."""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((BACKEND_HOST, port))
        sock.close()
        return True
    except (OSError, socket.error):
        return False


def _ensure_backend() -> subprocess.Popen | None:
    """Start the FastAPI backend if it's not already running.

    If the default BACKEND_PORT is in use and backend is not already running,
    try alternative ports (8001, 8002, 8003, 8004).
    """
    global BACKEND_PORT, HEALTH_URL

    if _is_backend_ready():
        return None

    # Check if port is available before starting
    if not _is_port_available(BACKEND_PORT):
        # Port is in use, try alternatives
        alternative_ports = [8001, 8002, 8003, 8004]
        port_found = False
        for alt_port in alternative_ports:
            if _is_port_available(alt_port):
                print(
                    f"[de-insight] port {BACKEND_PORT} in use, using {alt_port} instead",
                    file=sys.stderr,
                )
                BACKEND_PORT = alt_port
                HEALTH_URL = f"http://{BACKEND_HOST}:{BACKEND_PORT}/api/health"
                port_found = True
                break

        if not port_found:
            print(
                f"[de-insight] port {BACKEND_PORT} and alternatives (8001-8004) all in use.",
                file=sys.stderr,
            )
            sys.exit(1)

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
                f"系統已重啟。詳見日誌: {LOG_DIR / 'backend.log'}",
                file=sys.stderr,
            )
            sys.exit(1)
        if _is_backend_ready():
            print("[de-insight] backend ready", file=sys.stderr)
            return proc
        time.sleep(0.5)

    print(
        f"正在重新連線...詳見日誌: {LOG_DIR / 'backend.log'}",
        file=sys.stderr,
    )
    proc.terminate()
    sys.exit(1)


if __name__ == "__main__":
    backend_proc = _ensure_backend()
    if backend_proc is not None:
        atexit.register(backend_proc.terminate)

    from app import DeInsightApp
    try:
        DeInsightApp().run()
    finally:
        try:
            from rag.ingestion_service import get_ingestion_service

            asyncio.run(get_ingestion_service().abort_and_rollback_incomplete())
        except Exception:
            pass
