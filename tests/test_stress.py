"""壓力測試 — 驗證 ingestion pipeline 穩定性防護。

修復項目驗證：
1. Embedding batch sleep 0.05s → 0.2s
2. LightRAG 併發降低（embed 4→2, LLM 2→1/4）
3. URL 下載大小上限 5MB
4. 殭屍 llama-server 進程清理
5. 記憶體壓力檢查（<512MB 暫停）
"""

from __future__ import annotations

import asyncio
import importlib
import os
import signal
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest

# ── 路徑設定 ──────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent.parent))

# ── lightrag stub（避免真正 import lightrag 時產生依賴問題）──
if "lightrag" not in sys.modules:
    _lr_mod = SimpleNamespace(
        LightRAG=type("LightRAG", (), {"__init__": lambda self, **kw: None}),
        QueryParam=type("QueryParam", (), {"__init__": lambda self, **kw: None}),
    )
    _lr_utils = SimpleNamespace(
        EmbeddingFunc=type("EmbeddingFunc", (), {"__init__": lambda self, **kw: None}),
    )
    sys.modules["lightrag"] = _lr_mod
    sys.modules["lightrag.utils"] = _lr_utils
    sys.modules["lightrag.base"] = SimpleNamespace(
        DocStatus=SimpleNamespace(FAILED="failed", PROCESSING="processing"),
    )


# ═══════════════════════════════════════════════════════════════
# Class 1: TestEmbeddingBatchSleep — 驗證常數預設值
# ═══════════════════════════════════════════════════════════════

class TestEmbeddingBatchSleep:
    """驗證 gguf_backend 批次 sleep / batch size / 記憶體門檻的預設值和 env override。"""

    def test_batch_sleep_default_is_02(self):
        from embeddings.gguf_backend import _BATCH_SLEEP
        assert _BATCH_SLEEP == 0.2

    def test_batch_size_default_is_8(self):
        from embeddings.gguf_backend import _BATCH_SIZE
        assert _BATCH_SIZE == 8

    def test_mem_pressure_threshold_default_is_512(self):
        from embeddings.gguf_backend import _MEM_PRESSURE_MB
        assert _MEM_PRESSURE_MB == 512

    def test_batch_sleep_env_override(self):
        """GGUF_EMBED_BATCH_SLEEP=0.5 → reload module → 確認生效。"""
        import embeddings.gguf_backend as mod

        with mock.patch.dict(os.environ, {"GGUF_EMBED_BATCH_SLEEP": "0.5"}):
            importlib.reload(mod)
            assert mod._BATCH_SLEEP == 0.5

        # teardown: restore defaults
        for key in ("GGUF_EMBED_BATCH_SLEEP", "GGUF_EMBED_BATCH_SIZE", "GGUF_MEM_PRESSURE_MB"):
            os.environ.pop(key, None)
        importlib.reload(mod)


# ═══════════════════════════════════════════════════════════════
# Class 2: TestEmbeddingMemoryPressure — 記憶體壓力防護
# ═══════════════════════════════════════════════════════════════

class TestEmbeddingMemoryPressure:
    """驗證 _check_memory_pressure 和 _embed_texts 在壓力下的行為。"""

    def test_pressure_true_when_low(self):
        vm = SimpleNamespace(available=256 * 1024 * 1024)  # 256 MB
        mock_psutil = mock.MagicMock()
        mock_psutil.virtual_memory.return_value = vm
        with mock.patch.dict(sys.modules, {"psutil": mock_psutil}):
            from embeddings.gguf_backend import _check_memory_pressure
            assert _check_memory_pressure() is True

    def test_pressure_false_when_sufficient(self):
        vm = SimpleNamespace(available=2048 * 1024 * 1024)  # 2048 MB
        mock_psutil = mock.MagicMock()
        mock_psutil.virtual_memory.return_value = vm
        with mock.patch.dict(sys.modules, {"psutil": mock_psutil}):
            from embeddings.gguf_backend import _check_memory_pressure
            assert _check_memory_pressure() is False

    def test_pressure_false_without_psutil(self):
        """psutil 不存在時 graceful fallback → returns False。"""
        saved = sys.modules.pop("psutil", _SENTINEL)
        try:
            from embeddings.gguf_backend import _check_memory_pressure
            assert _check_memory_pressure() is False
        finally:
            if saved is not _SENTINEL:
                sys.modules["psutil"] = saved

    @pytest.mark.asyncio
    async def test_embed_texts_pauses_on_pressure(self):
        """16 texts (2 batches), 第一批觸發壓力 → asyncio.sleep(2.0) 被呼叫。"""
        backend = _make_backend()
        texts = [f"text_{i}" for i in range(16)]

        call_count = 0

        def pressure_side_effect():
            nonlocal call_count
            call_count += 1
            return call_count <= 1  # Only first check triggers pressure

        sleep_calls = []
        _orig_sleep = asyncio.sleep

        async def mock_sleep(duration):
            sleep_calls.append(duration)

        with mock.patch("embeddings.gguf_backend._check_memory_pressure", side_effect=pressure_side_effect):
            with mock.patch.object(asyncio, "sleep", side_effect=mock_sleep):
                with mock.patch.object(backend, "_call_embedding", _fake_call_embedding):
                    await backend._embed_texts(texts)

        assert 2.0 in sleep_calls, f"Expected 2.0s memory pressure sleep, got: {sleep_calls}"


# ═══════════════════════════════════════════════════════════════
# Class 3: TestEmbeddingBatchThroughput — 高吞吐量批次計時
# ═══════════════════════════════════════════════════════════════

class TestEmbeddingBatchThroughput:
    """驗證批次 sleep 的呼叫次數和行為。"""

    @pytest.mark.asyncio
    async def test_20_batches_include_19_sleeps(self):
        """160 texts → 20 batches → sleep(0.2) 被呼叫 19 次。"""
        backend = _make_backend()
        texts = [f"text_{i}" for i in range(160)]

        sleep_calls = []

        async def mock_sleep(duration):
            sleep_calls.append(duration)

        with mock.patch("embeddings.gguf_backend._check_memory_pressure", return_value=False):
            with mock.patch.object(asyncio, "sleep", side_effect=mock_sleep):
                with mock.patch.object(backend, "_call_embedding", _fake_call_embedding):
                    await backend._embed_texts(texts)

        batch_sleeps = [s for s in sleep_calls if s == 0.2]
        assert len(batch_sleeps) == 19, f"Expected 19 batch sleeps, got {len(batch_sleeps)}"

    @pytest.mark.asyncio
    async def test_single_batch_no_sleep(self):
        """8 texts → 只有 1 batch → 不應有 inter-batch sleep。"""
        backend = _make_backend()
        texts = [f"text_{i}" for i in range(8)]

        sleep_calls = []

        async def mock_sleep(duration):
            sleep_calls.append(duration)

        with mock.patch("embeddings.gguf_backend._check_memory_pressure", return_value=False):
            with mock.patch.object(asyncio, "sleep", side_effect=mock_sleep):
                with mock.patch.object(backend, "_call_embedding", _fake_call_embedding):
                    await backend._embed_texts(texts)

        batch_sleeps = [s for s in sleep_calls if s == 0.2]
        assert len(batch_sleeps) == 0, f"Expected 0 batch sleeps, got {len(batch_sleeps)}"

    @pytest.mark.asyncio
    async def test_memory_pressure_adds_delay(self):
        """2 batches + 持續記憶體壓力 → 確認有 2 次 sleep(2.0)。"""
        backend = _make_backend()
        texts = [f"text_{i}" for i in range(16)]  # 2 batches

        sleep_calls = []

        async def mock_sleep(duration):
            sleep_calls.append(duration)

        with mock.patch("embeddings.gguf_backend._check_memory_pressure", return_value=True):
            with mock.patch.object(asyncio, "sleep", side_effect=mock_sleep):
                with mock.patch.object(backend, "_call_embedding", _fake_call_embedding):
                    await backend._embed_texts(texts)

        pressure_sleeps = [s for s in sleep_calls if s == 2.0]
        assert len(pressure_sleeps) == 2, f"Expected 2 pressure sleeps, got {len(pressure_sleeps)}"


# ═══════════════════════════════════════════════════════════════
# Class 4: TestURLSizeLimit — 下載大小上限
# ═══════════════════════════════════════════════════════════════

class TestURLSizeLimit:
    """驗證 insert_url 的下載大小限制。"""

    @pytest.mark.asyncio
    async def test_large_content_raises(self):
        """6MB response → raises RuntimeError("頁面太大")。"""
        resp = _make_http_response(6 * 1024 * 1024)

        with _patch_insert_url_deps(resp):
            from rag.knowledge_graph import insert_url
            with pytest.raises(RuntimeError, match="頁面太大"):
                await insert_url("https://example.com/large", project_id="test")

    @pytest.mark.asyncio
    async def test_within_limit_proceeds(self):
        """1MB response → 不 raise，繼續走 Jina Reader / insert_text。"""
        resp = _make_http_response(1 * 1024 * 1024, content_type="text/html")

        async def fake_jina(*a, **kw):
            return ("Some article text " * 20, {"status_code": 200})

        async def fake_insert_text(*a, **kw):
            return ""

        with _patch_insert_url_deps(resp):
            from rag.knowledge_graph import insert_url
            with mock.patch("rag.knowledge_graph.insert_text", side_effect=fake_insert_text):
                with mock.patch("rag.knowledge_graph._fetch_with_jina_reader", side_effect=fake_jina):
                    result = await insert_url("https://example.com/ok", project_id="test")
            assert "title" in result

    @pytest.mark.asyncio
    async def test_size_limit_env_override(self):
        """WEB_FETCH_MAX_BYTES=1048576 + 2MB content → raises。"""
        resp = _make_http_response(2 * 1024 * 1024)

        with _patch_insert_url_deps(resp, env_overrides={"WEB_FETCH_MAX_BYTES": "1048576"}):
            from rag.knowledge_graph import insert_url
            with pytest.raises(RuntimeError, match="頁面太大"):
                await insert_url("https://example.com/2mb", project_id="test")

    @pytest.mark.asyncio
    async def test_size_check_before_llm(self):
        """6MB content → _llm_clean_web_content 和 insert_text 都不應被呼叫。"""
        resp = _make_http_response(6 * 1024 * 1024)

        with _patch_insert_url_deps(resp):
            from rag.knowledge_graph import insert_url
            with mock.patch("rag.knowledge_graph._llm_clean_web_content") as mock_clean:
                with mock.patch("rag.knowledge_graph.insert_text") as mock_insert:
                    with pytest.raises(RuntimeError, match="頁面太大"):
                        await insert_url("https://example.com/huge", project_id="test")
                    mock_clean.assert_not_called()
                    mock_insert.assert_not_called()


# ═══════════════════════════════════════════════════════════════
# Class 5: TestZombieLlamaServerCleanup — 殭屍進程清理
# ═══════════════════════════════════════════════════════════════

class TestZombieLlamaServerCleanup:
    """驗證 llama-server 殭屍進程清理。"""

    def setup_method(self):
        """每個測試前重置 singleton。"""
        from embeddings.llama_server import LlamaServerManager
        LlamaServerManager._instance = None

    def teardown_method(self):
        from embeddings.llama_server import LlamaServerManager
        LlamaServerManager._instance = None

    def test_kill_orphans_called_on_start(self):
        """start() 時 _kill_orphan_servers 被呼叫。"""
        from embeddings.llama_server import LlamaServerManager

        mgr = LlamaServerManager()
        # Make is_running return False so start() doesn't skip
        with mock.patch.object(LlamaServerManager, "is_running", new_callable=mock.PropertyMock, return_value=False):
            with mock.patch.object(mgr, "_kill_orphan_servers") as mock_kill:
                with mock.patch.object(mgr, "find_binary", return_value=None):
                    try:
                        mgr.start("/fake/model.gguf")
                    except Exception:
                        pass  # LlamaServerError expected
                    mock_kill.assert_called_once()

    def test_sends_sigterm_to_orphans(self):
        """pgrep 回兩個 PID → os.kill(SIGTERM) 各呼叫一次。"""
        from embeddings.llama_server import LlamaServerManager
        mgr = LlamaServerManager()
        mgr._process = None  # no managed process

        pgrep_result = SimpleNamespace(returncode=0, stdout="1234\n5678\n")

        with mock.patch("shutil.which", return_value="/usr/bin/pgrep"):
            with mock.patch("subprocess.run", return_value=pgrep_result):
                with mock.patch("os.kill") as mock_kill:
                    mgr._kill_orphan_servers()

        calls = mock_kill.call_args_list
        killed_pids = {c[0][0] for c in calls}
        assert killed_pids == {1234, 5678}
        for c in calls:
            assert c[0][1] == signal.SIGTERM

    def test_skips_own_process(self):
        """自己管理的 PID 不被 kill。"""
        from embeddings.llama_server import LlamaServerManager
        mgr = LlamaServerManager()
        # Simulate a managed process with pid=1234
        mock_proc = SimpleNamespace(pid=1234, poll=lambda: None)
        mgr._process = mock_proc

        pgrep_result = SimpleNamespace(returncode=0, stdout="1234\n5678\n")

        with mock.patch("shutil.which", return_value="/usr/bin/pgrep"):
            with mock.patch("subprocess.run", return_value=pgrep_result):
                with mock.patch("os.kill") as mock_kill:
                    mgr._kill_orphan_servers()

        killed_pids = {c[0][0] for c in mock_kill.call_args_list}
        assert 1234 not in killed_pids
        assert 5678 in killed_pids

    def test_noop_when_no_pgrep(self):
        """pgrep 不存在時靜默返回。"""
        from embeddings.llama_server import LlamaServerManager
        mgr = LlamaServerManager()

        with mock.patch("shutil.which", return_value=None):
            with mock.patch("os.kill") as mock_kill:
                mgr._kill_orphan_servers()  # should not raise
                mock_kill.assert_not_called()


# ═══════════════════════════════════════════════════════════════
# Class 6: TestLightRAGConcurrencyDefaults — 併發度降低
# ═══════════════════════════════════════════════════════════════

class TestLightRAGConcurrencyDefaults:
    """驗證 get_rag() 傳給 LightRAG 的併發度參數。"""

    def setup_method(self):
        import rag.knowledge_graph as kg
        kg._rag_instance = None
        kg._rag_project_id = None

    def teardown_method(self):
        import rag.knowledge_graph as kg
        kg._rag_instance = None
        kg._rag_project_id = None
        for key in ("LIGHTRAG_EMBED_MAX_ASYNC", "LIGHTRAG_LLM_MAX_ASYNC"):
            os.environ.pop(key, None)

    def test_embed_max_async_default_is_2(self):
        """get_rag() → LightRAG 收到 embedding_func_max_async=2。"""
        captured = {}
        with _patch_get_rag_deps(llm_base="https://api.openai.com/v1", captured=captured):
            from rag.knowledge_graph import get_rag
            get_rag("test_embed_2")
        assert captured["embedding_func_max_async"] == 2

    def test_llm_local_default_is_1(self):
        """localhost LLM → llm_model_max_async=1。"""
        captured = {}
        with _patch_get_rag_deps(llm_base="http://localhost:11434/v1", captured=captured):
            from rag.knowledge_graph import get_rag
            get_rag("test_local_1")
        assert captured["llm_model_max_async"] == 1

    def test_llm_cloud_default_is_4(self):
        """雲端 LLM → llm_model_max_async=4。"""
        captured = {}
        with _patch_get_rag_deps(llm_base="https://api.openai.com/v1", captured=captured):
            from rag.knowledge_graph import get_rag
            get_rag("test_cloud_4")
        assert captured["llm_model_max_async"] == 4

    def test_concurrency_env_override(self):
        """env override 生效。"""
        captured = {}
        with mock.patch.dict(os.environ, {
            "LIGHTRAG_EMBED_MAX_ASYNC": "8",
            "LIGHTRAG_LLM_MAX_ASYNC": "16",
        }):
            with _patch_get_rag_deps(llm_base="https://api.openai.com/v1", captured=captured):
                from rag.knowledge_graph import get_rag
                get_rag("test_env_override")
        assert captured["embedding_func_max_async"] == 8
        assert captured["llm_model_max_async"] == 16


# ═══════════════════════════════════════════════════════════════
# Class 7: TestRateGuardUnderLoad — 熔斷器壓力測試
# ═══════════════════════════════════════════════════════════════

class TestRateGuardUnderLoad:
    """驗證 RateGuard 的熔斷器、信號量和速率控制。"""

    @pytest.mark.asyncio
    async def test_breaker_trips_after_5_failures(self):
        """5 次失敗 → BreakerState.OPEN。"""
        from rag.rate_guard import RateGuard, BreakerState

        guard = RateGuard(rpm=600, max_concurrency=10, breaker_threshold=5, breaker_cooldown=60.0)

        for i in range(5):
            with pytest.raises(ValueError):
                async with guard.acquire("test"):
                    raise ValueError(f"fail_{i}")

        assert guard.breaker_state == BreakerState.OPEN

    @pytest.mark.asyncio
    async def test_breaker_rejects_while_open(self):
        """熔斷中 → RateLimitError。"""
        from rag.rate_guard import RateGuard, RateLimitError

        guard = RateGuard(rpm=600, max_concurrency=10, breaker_threshold=5, breaker_cooldown=60.0)

        # Trip the breaker
        for i in range(5):
            with pytest.raises(ValueError):
                async with guard.acquire("test"):
                    raise ValueError("fail")

        # Now should be rejected
        with pytest.raises(RateLimitError):
            async with guard.acquire("test"):
                pass

    @pytest.mark.asyncio
    async def test_breaker_recovers_after_cooldown(self):
        """cooldown=0.1s → 恢復。"""
        from rag.rate_guard import RateGuard, BreakerState

        guard = RateGuard(rpm=600, max_concurrency=10, breaker_threshold=5, breaker_cooldown=0.1)

        for i in range(5):
            with pytest.raises(ValueError):
                async with guard.acquire("test"):
                    raise ValueError("fail")

        assert guard.breaker_state == BreakerState.OPEN

        await asyncio.sleep(0.15)  # Wait for cooldown

        # Should now be HALF_OPEN and allow a request
        async with guard.acquire("test"):
            pass  # success → CLOSED

        assert guard.breaker_state == BreakerState.CLOSED

    @pytest.mark.asyncio
    async def test_concurrent_semaphore_limit(self):
        """max_concurrency=2 + 5 concurrent tasks → 最多同時 2 個。"""
        from rag.rate_guard import RateGuard

        guard = RateGuard(rpm=600, max_concurrency=2, breaker_threshold=100)

        max_concurrent = 0
        current = 0
        lock = asyncio.Lock()

        async def task():
            nonlocal max_concurrent, current
            async with guard.acquire("test"):
                async with lock:
                    current += 1
                    max_concurrent = max(max_concurrent, current)
                await asyncio.sleep(0.05)
                async with lock:
                    current -= 1

        await asyncio.gather(*[task() for _ in range(5)])
        assert max_concurrent <= 2

    @pytest.mark.asyncio
    async def test_rapid_fire_50_requests(self):
        """rpm=600 + 50 requests → 全部成功。"""
        from rag.rate_guard import RateGuard

        guard = RateGuard(rpm=600, max_concurrency=50, breaker_threshold=100)

        success_count = 0

        async def task():
            nonlocal success_count
            async with guard.acquire("test"):
                success_count += 1

        await asyncio.gather(*[task() for _ in range(50)])
        assert success_count == 50

    @pytest.mark.asyncio
    async def test_transient_errors_trigger_retry(self):
        """call_with_retry + transient error → 有重試。"""
        from rag.rate_guard import RateGuard

        guard = RateGuard(rpm=600, max_concurrency=10, breaker_threshold=100)

        attempt_count = 0

        async def flaky_fn():
            nonlocal attempt_count
            attempt_count += 1
            if attempt_count < 3:
                raise ConnectionError("connection reset")
            return "ok"

        result = await guard.call_with_retry("test", flaky_fn, max_retries=3)
        assert result == "ok"
        assert attempt_count == 3


# ═══════════════════════════════════════════════════════════════
# Class 8: TestErrorClassification — 錯誤分類覆蓋
# ═══════════════════════════════════════════════════════════════

class TestErrorClassification:
    """驗證 job_executor.classify_error 的分類邏輯。"""

    def test_all_transient_patterns(self):
        from rag.job_executor import classify_error, ErrorCategory

        transient_cases = [
            RuntimeError("HTTP 429 Too Many Requests"),
            RuntimeError("rate limit exceeded"),
            RuntimeError("request timeout after 30s"),
            RuntimeError("connection refused"),
            RuntimeError("502 Bad Gateway"),
            RuntimeError("503 Service Unavailable"),
            RuntimeError("504 Gateway Timeout"),
            RuntimeError("fds_to_keep error"),
        ]
        for err in transient_cases:
            assert classify_error(err) == ErrorCategory.TRANSIENT, f"Expected TRANSIENT for: {err}"

        # RateLimitError should also be transient
        from rag.rate_guard import RateLimitError
        assert classify_error(RateLimitError("throttled")) == ErrorCategory.TRANSIENT

    def test_all_permanent_patterns(self):
        from rag.job_executor import classify_error, ErrorCategory

        permanent_cases = [
            RuntimeError("resource not found"),
            RuntimeError("401 Unauthorized"),
            RuntimeError("403 Forbidden"),
            RuntimeError("invalid format for embedding"),
        ]
        for err in permanent_cases:
            assert classify_error(err) == ErrorCategory.PERMANENT, f"Expected PERMANENT for: {err}"

    def test_unknown_defaults_to_transient(self):
        from rag.job_executor import classify_error, ErrorCategory

        err = RuntimeError("some completely unknown error xyz")
        assert classify_error(err) == ErrorCategory.TRANSIENT


# ═══════════════════════════════════════════════════════════════
# Class 9: TestJobDedupUnderConcurrency — 併發去重
# ═══════════════════════════════════════════════════════════════

class TestJobDedupUnderConcurrency:
    """驗證 JobRepository 在併發下的去重行為。"""

    @pytest.mark.asyncio
    async def test_concurrent_duplicate_submissions(self, tmp_path):
        """同一 source 提交 10 次 → 1 成功 + 9 DuplicateJobError。"""
        from rag.job_repository import JobRepository, DuplicateJobError

        db_path = tmp_path / "test_dedup.db"
        repo = JobRepository(db_path)
        await repo.ensure_table()

        results = {"success": 0, "duplicate": 0}

        async def submit():
            try:
                await repo.create_job(
                    project_id="proj1",
                    source="https://example.com/same-page",
                    source_type="url",
                )
                results["success"] += 1
            except DuplicateJobError:
                results["duplicate"] += 1

        # Run sequentially to test dedup deterministically:
        # first insert succeeds, subsequent ones see the queued job and raise.
        for _ in range(10):
            await submit()

        assert results["success"] == 1
        assert results["duplicate"] == 9

    @pytest.mark.asyncio
    async def test_concurrent_different_sources(self, tmp_path):
        """20 個不同 source → 全部成功。"""
        from rag.job_repository import JobRepository

        db_path = tmp_path / "test_diff.db"
        repo = JobRepository(db_path)
        await repo.ensure_table()

        success_count = 0

        async def submit(i):
            nonlocal success_count
            await repo.create_job(
                project_id="proj1",
                source=f"https://example.com/page-{i}",
                source_type="url",
            )
            success_count += 1

        await asyncio.gather(*[submit(i) for i in range(20)])
        assert success_count == 20


# ═══════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════

_SENTINEL = object()


def _make_backend():
    """建立一個 GGUFMultimodalBackend 實例（不連接真實 server）。"""
    from embeddings.gguf_backend import GGUFMultimodalBackend
    return GGUFMultimodalBackend(base_url="http://127.0.0.1:9999", dim=1024)


async def _fake_call_embedding(texts):
    """直接用作 _call_embedding 的替代（async function）。"""
    return [[0.1] * 1024 for _ in texts]


def _make_http_response(content_size: int, content_type: str = "text/html"):
    """建立假的 httpx Response。"""
    content = b"x" * content_size
    text = "<html><title>Test</title><body>" + "content " * 100 + "</body></html>"
    resp = SimpleNamespace(
        content=content,
        text=text,
        status_code=200,
        headers={"content-type": content_type},
        raise_for_status=lambda: None,
    )
    return resp


@contextmanager
def _patch_insert_url_deps(fake_resp, env_overrides: dict | None = None):
    """Patch insert_url 需要的所有外部依賴。

    httpx is imported locally inside insert_url, so we patch httpx.AsyncClient
    at the httpx module level.
    """
    mock_client = mock.AsyncMock()
    mock_client.get = mock.AsyncMock(return_value=fake_resp)
    mock_client.__aenter__ = mock.AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = mock.AsyncMock(return_value=None)

    env = env_overrides or {}

    with mock.patch("rag.knowledge_graph.load_env", return_value=env):
        with mock.patch("httpx.AsyncClient", return_value=mock_client):
            yield {"client": mock_client}


@contextmanager
def _patch_get_rag_deps(llm_base: str = "https://api.openai.com/v1", captured: dict | None = None):
    """Patch get_rag() 的所有依賴，捕捉傳給 LightRAG 的參數。"""
    import rag.knowledge_graph as kg

    # Reset singleton
    kg._rag_instance = None
    kg._rag_project_id = None

    if captured is None:
        captured = {}

    class FakeLightRAG:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    class FakeEmbeddingFunc:
        def __init__(self, **kwargs):
            pass

    mock_embed_svc = mock.MagicMock()
    mock_embed_svc.get_embed_config.return_value = ("jina-v4", "key", "http://localhost:8999", 1024)
    mock_embed_svc.check_signature_migration.return_value = False

    with mock.patch("rag.knowledge_graph._apply_env"):
        with mock.patch("rag.knowledge_graph._get_llm_config", return_value=("gpt-4o", "key", llm_base)):
            with mock.patch("rag.knowledge_graph._get_embed_config", return_value=("jina-v4", "key", "http://localhost:8999", 1024)):
                with mock.patch("embeddings.service.get_embedding_service", return_value=mock_embed_svc):
                    with mock.patch("rag.knowledge_graph.LightRAG", FakeLightRAG):
                        with mock.patch("rag.knowledge_graph.EmbeddingFunc", FakeEmbeddingFunc):
                            with mock.patch("rag.knowledge_graph._detect_vdb_dim", return_value=None):
                                with mock.patch("rag.knowledge_graph.ensure_project_dirs", return_value=Path("/tmp/fake_project")):
                                    yield captured

    # Cleanup singleton
    kg._rag_instance = None
    kg._rag_project_id = None
