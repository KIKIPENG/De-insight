"""GGUF 環境自動安裝器。

首次啟動時自動完成：
1. cmake 檢查（macOS 透過 brew 安裝）
2. jina-ai/llama.cpp clone 與編譯（Metal ON）
3. 模型檔下載與校驗
"""

from __future__ import annotations

import hashlib
import logging
import os
import platform
import shutil
import subprocess
from pathlib import Path
from typing import Callable

log = logging.getLogger(__name__)

# ── 預設值 ──────────────────────────────────────────────────────────

_LLAMA_CPP_REPO = "https://github.com/jina-ai/llama.cpp.git"
_LLAMA_CPP_BRANCH = ""

_DEFAULT_HF_REPO = "jinaai/jina-embeddings-v4-text-retrieval-GGUF"
_DEFAULT_MODEL_FILE = "jina-embeddings-v4-text-retrieval-Q4_K_M.gguf"
_DEFAULT_MMPROJ_FILE = "mmproj-jina-embeddings-v4-retrieval-BF16.gguf"


class GGUFInstallerError(Exception):
    """安裝過程中的不可恢復錯誤。"""


class GGUFInstaller:
    """GGUF 環境安裝與驗證。"""

    def __init__(self, gguf_home: Path | None = None) -> None:
        if gguf_home is None:
            from paths import APP_HOME
            gguf_home = APP_HOME / "gguf"
        self._home = gguf_home
        self._home.mkdir(parents=True, exist_ok=True)

    @property
    def llama_cpp_dir(self) -> Path:
        return self._home / "llama.cpp"

    @property
    def build_dir(self) -> Path:
        return self.llama_cpp_dir / "build"

    @property
    def bin_dir(self) -> Path:
        return self.build_dir / "bin"

    @property
    def models_dir(self) -> Path:
        d = self._home / "models"
        d.mkdir(parents=True, exist_ok=True)
        return d

    @property
    def model_path(self) -> Path:
        name = os.environ.get("GGUF_MODEL_FILE", _DEFAULT_MODEL_FILE)
        return self.models_dir / name

    @property
    def mmproj_path(self) -> Path:
        name = os.environ.get("GGUF_MMPROJ_FILE", _DEFAULT_MMPROJ_FILE)
        return self.models_dir / name

    @property
    def server_binary(self) -> Path:
        return self.bin_dir / "llama-server"

    # ── 安裝狀態檢查 ────────────────────────────────────────────

    def is_cmake_available(self) -> bool:
        return shutil.which("cmake") is not None

    def is_llama_cpp_built(self) -> bool:
        return self.server_binary.exists() and os.access(self.server_binary, os.X_OK)

    def is_model_downloaded(self) -> bool:
        return self.model_path.exists() and self.model_path.stat().st_size > 0

    def is_mmproj_downloaded(self) -> bool:
        return self.mmproj_path.exists() and self.mmproj_path.stat().st_size > 0

    def is_fully_installed(self) -> bool:
        return (
            self.is_llama_cpp_built()
            and self.is_model_downloaded()
            and self.is_mmproj_downloaded()
        )

    def installation_status(self) -> dict[str, bool]:
        return {
            "cmake": self.is_cmake_available(),
            "llama_cpp_built": self.is_llama_cpp_built(),
            "model_downloaded": self.is_model_downloaded(),
            "mmproj_downloaded": self.is_mmproj_downloaded(),
            "fully_installed": self.is_fully_installed(),
        }

    # ── 完整安裝流程 ────────────────────────────────────────────

    def install(
        self,
        *,
        progress_callback: Callable[[str, float], None] | None = None,
        force_rebuild: bool = False,
    ) -> None:
        """執行完整安裝流程。失敗即中止，不靜默降級。

        progress_callback(phase_description, progress_0_to_1)
        """
        cb = progress_callback or (lambda desc, pct: None)

        # 1. cmake
        cb("檢查 cmake...", 0.0)
        self._ensure_cmake()
        cb("cmake 就緒", 0.1)

        # 2. clone + build
        if force_rebuild or not self.is_llama_cpp_built():
            cb("取得 jina-ai/llama.cpp...", 0.15)
            self._clone_or_pull()
            cb("編譯 llama-server（Metal ON）...", 0.25)
            self._build_llama_cpp()
            cb("llama-server 編譯完成", 0.5)
        else:
            cb("llama-server 已存在，跳過編譯", 0.5)

        # 3. 下載模型
        if not self.is_model_downloaded():
            cb("下載 GGUF 模型（Q4_K_M）...", 0.55)
            self._download_model()
            cb("模型下載完成", 0.8)
        else:
            cb("模型檔已存在", 0.8)

        # 4. 下載 mmproj
        if not self.is_mmproj_downloaded():
            cb("下載 mmproj-f16...", 0.85)
            self._download_mmproj()
            cb("mmproj 下載完成", 0.95)
        else:
            cb("mmproj 已存在", 0.95)

        cb("安裝完成", 1.0)
        log.info("GGUF installation complete: %s", self.installation_status())

    # ── 子步驟 ──────────────────────────────────────────────────

    def _ensure_cmake(self) -> None:
        """確保 cmake 可用。macOS 透過 brew 安裝。"""
        if self.is_cmake_available():
            return

        if platform.system() != "Darwin":
            raise GGUFInstallerError(
                "cmake 未安裝。請手動安裝 cmake 後再試。"
            )

        # macOS: 嘗試 brew install
        if shutil.which("brew") is None:
            raise GGUFInstallerError(
                "cmake 與 Homebrew 都未安裝。請先安裝 Homebrew: https://brew.sh"
            )

        log.info("Installing cmake via Homebrew...")
        result = subprocess.run(
            ["brew", "install", "cmake"],
            capture_output=True, text=True, timeout=300,
        )
        if result.returncode != 0:
            raise GGUFInstallerError(
                f"brew install cmake 失敗:\n{result.stderr}"
            )

        if not self.is_cmake_available():
            raise GGUFInstallerError("cmake 安裝後仍不可用")

    def _clone_or_pull(self) -> None:
        """Clone 或更新 jina-ai/llama.cpp。"""
        repo = os.environ.get("GGUF_LLAMA_CPP_REPO", _LLAMA_CPP_REPO).strip() or _LLAMA_CPP_REPO
        branch = os.environ.get("GGUF_LLAMA_CPP_BRANCH", _LLAMA_CPP_BRANCH).strip()
        repo_dir = self.llama_cpp_dir

        if repo_dir.exists() and (repo_dir / ".git").exists():
            log.info("Updating existing llama.cpp clone...")
            result = subprocess.run(
                ["git", "pull", "--ff-only"],
                cwd=repo_dir,
                capture_output=True, text=True, timeout=120,
            )
            if result.returncode != 0:
                log.warning("git pull failed, will rebuild: %s", result.stderr)
                shutil.rmtree(repo_dir)
            else:
                return

        if repo_dir.exists():
            shutil.rmtree(repo_dir)

        clone_cmd = ["git", "clone", "--depth=1"]
        if branch:
            clone_cmd += ["-b", branch]
        clone_cmd += [repo, str(repo_dir)]
        log.info("Cloning %s%s...", repo, f" (branch: {branch})" if branch else "")
        result = subprocess.run(clone_cmd, capture_output=True, text=True, timeout=300)

        # Branch may be removed/renamed upstream; retry default branch once.
        if result.returncode != 0 and branch:
            log.warning(
                "Clone with branch '%s' failed; retrying default branch. stderr: %s",
                branch,
                (result.stderr or "").strip()[:200],
            )
            result = subprocess.run(
                ["git", "clone", "--depth=1", repo, str(repo_dir)],
                capture_output=True, text=True, timeout=300,
            )
        if result.returncode != 0:
            raise GGUFInstallerError(
                f"git clone 失敗:\n{result.stderr}"
            )

    def _build_llama_cpp(self) -> None:
        """使用 cmake 編譯 llama-server（Metal ON for macOS arm64）。"""
        build_dir = self.build_dir
        if build_dir.exists():
            shutil.rmtree(build_dir)
        build_dir.mkdir(parents=True)

        cmake_args = [
            "cmake",
            "-B", str(build_dir),
            "-S", str(self.llama_cpp_dir),
            "-DCMAKE_BUILD_TYPE=Release",
        ]

        # macOS arm64: enable Metal
        if platform.system() == "Darwin" and platform.machine() == "arm64":
            cmake_args.append("-DLLAMA_METAL=ON")

        log.info("cmake configure: %s", " ".join(cmake_args))
        result = subprocess.run(
            cmake_args,
            capture_output=True, text=True, timeout=120,
        )
        if result.returncode != 0:
            raise GGUFInstallerError(
                f"cmake configure 失敗:\n{result.stderr}"
            )

        # build
        ncpu = os.cpu_count() or 4
        build_cmd = [
            "cmake", "--build", str(build_dir),
            "--target", "llama-server",
            "-j", str(ncpu),
        ]
        log.info("cmake build: %s", " ".join(build_cmd))
        result = subprocess.run(
            build_cmd,
            capture_output=True, text=True, timeout=600,
        )
        if result.returncode != 0:
            raise GGUFInstallerError(
                f"cmake build 失敗:\n{result.stderr}"
            )

        if not self.server_binary.exists():
            raise GGUFInstallerError(
                f"編譯完成但找不到 llama-server: {self.server_binary}"
            )

        log.info("llama-server built successfully: %s", self.server_binary)

    def _download_model(self) -> None:
        """從 HuggingFace 下載 GGUF 模型。"""
        self._hf_download(
            filename=os.environ.get("GGUF_MODEL_FILE", _DEFAULT_MODEL_FILE),
            target=self.model_path,
        )

    def _download_mmproj(self) -> None:
        """從 HuggingFace 下載 mmproj 檔。"""
        self._hf_download(
            filename=os.environ.get("GGUF_MMPROJ_FILE", _DEFAULT_MMPROJ_FILE),
            target=self.mmproj_path,
        )

    def _hf_download(self, filename: str, target: Path) -> None:
        """從 HuggingFace Hub 下載單一檔案。"""
        repo_id = os.environ.get("GGUF_HF_REPO", _DEFAULT_HF_REPO)
        log.info("Downloading %s from %s...", filename, repo_id)

        # Hard overwrite to avoid subprocess/fd issues (xet/hf_transfer paths).
        os.environ["HF_HUB_DISABLE_XET"] = "1"
        os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

        try:
            from huggingface_hub import hf_hub_download
            downloaded = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                local_dir=str(self.models_dir),
                resume_download=True,
            )
            downloaded_path = Path(downloaded)
            if downloaded_path != target:
                shutil.move(str(downloaded_path), str(target))
            log.info("Downloaded: %s (%d MB)", target.name, target.stat().st_size // (1024 * 1024))
            return
        except Exception as exc:
            log.warning("hf_hub_download failed (%s), fallback to direct HTTP download", exc)

        # Fallback path: avoid huggingface_hub internals entirely.
        try:
            import httpx

            url = f"https://huggingface.co/{repo_id}/resolve/main/{filename}"
            tmp = target.with_suffix(target.suffix + ".part")
            tmp.parent.mkdir(parents=True, exist_ok=True)

            with httpx.Client(follow_redirects=True, timeout=120.0) as client:
                with client.stream("GET", url) as resp:
                    resp.raise_for_status()
                    with open(tmp, "wb") as f:
                        for chunk in resp.iter_bytes():
                            if chunk:
                                f.write(chunk)

            if target.exists():
                target.unlink()
            tmp.replace(target)
            log.info("Downloaded (fallback): %s (%d MB)", target.name, target.stat().st_size // (1024 * 1024))
        except Exception as exc:
            # 清理不完整的下載
            target.unlink(missing_ok=True)
            part = target.with_suffix(target.suffix + ".part")
            part.unlink(missing_ok=True)
            raise GGUFInstallerError(
                f"模型下載失敗 ({repo_id}/{filename}): {exc}"
            ) from exc

    # ── 工具 ────────────────────────────────────────────────────

    def verify_model(self) -> bool:
        """驗證模型檔案完整性（簡單大小檢查）。"""
        if not self.model_path.exists():
            return False
        # Q4_K_M 至少應該 > 1GB
        return self.model_path.stat().st_size > 500_000_000

    def clean(self) -> None:
        """清除所有 GGUF 相關檔案（重新安裝用）。"""
        if self._home.exists():
            shutil.rmtree(self._home)
            self._home.mkdir(parents=True, exist_ok=True)
        log.info("GGUF installation cleaned: %s", self._home)
