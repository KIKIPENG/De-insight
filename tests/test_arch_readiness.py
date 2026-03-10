"""Architecture & readiness boundary tests.

Validates the new architectural constraints:
1. UI refresh does NOT trigger auto_repair
2. readiness=building → search/chat behave consistently
3. Startup embedding check does NOT call ensure_model_downloaded
4. Transient errors do NOT trigger destructive repair
5. Atomic claim: dual-worker competition doesn't duplicate claim
6. Same-second completed polling: no miss, no duplicate (preserved)
7. Chunk ready + KG incomplete → search/chat still retrievable
"""

import asyncio
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch, call

# Ensure project root is on path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

os.environ.setdefault("DEINSIGHT_HOME", tempfile.mkdtemp())
os.environ.setdefault("DEINSIGHT_DATA_VERSION", "test_arch")


# ── 1. UI refresh does NOT trigger auto_repair ──────────────────────


class TestUIRefreshNoAutoRepair(unittest.TestCase):
    """_refresh_knowledge_panel must NOT call auto_repair."""

    def test_refresh_panel_source_code_no_auto_repair(self):
        """Verify RAGMixin._refresh_knowledge_panel source does not import/call auto_repair."""
        import inspect
        from mixins.rag import RAGMixin

        source = inspect.getsource(RAGMixin._refresh_knowledge_panel.__wrapped__)
        self.assertNotIn("auto_repair", source,
                         "UI refresh must not reference auto_repair")
        self.assertNotIn("from rag.repair import", source,
                         "UI refresh must not import from rag.repair")

    def test_refresh_panel_uses_readiness_service(self):
        """Verify RAGMixin._refresh_knowledge_panel uses IngestionReadinessService."""
        import inspect
        from mixins.rag import RAGMixin

        source = inspect.getsource(RAGMixin._refresh_knowledge_panel.__wrapped__)
        self.assertIn("readiness", source.lower(),
                      "UI refresh should use readiness service")


# ── 2. readiness states ─────────────────────────────────────────────


class TestReadinessService(unittest.IsolatedAsyncioTestCase):
    """Test IngestionReadinessService state computation."""

    async def asyncSetUp(self):
        self._tmpdir = tempfile.mkdtemp()
        self._db_tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self._db_tmp.close()
        self._db_path = Path(self._db_tmp.name)

        from rag.job_repository import JobRepository
        self.repo = JobRepository(self._db_path)
        await self.repo.ensure_table()

    async def asyncTearDown(self):
        os.unlink(self._db_path)
        import shutil
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def _write_vdb(self, project_id: str, data_count: int = 0):
        """Write a fake vdb_chunks.json."""
        wd = Path(self._tmpdir) / "lightrag"
        wd.mkdir(parents=True, exist_ok=True)
        vdb = {"embedding_dim": 1024, "data": [{"id": f"c{i}"} for i in range(data_count)]}
        (wd / "vdb_chunks.json").write_text(json.dumps(vdb), encoding="utf-8")

    async def _get_snapshot(self, project_id: str):
        from rag.readiness import IngestionReadinessService
        svc = IngestionReadinessService(jobs_db_path=self._db_path)
        with patch("paths.project_root", return_value=Path(self._tmpdir)):
            return await svc.get_snapshot(project_id)

    async def test_empty_status(self):
        """No chunks, no jobs → empty."""
        snap = await self._get_snapshot("proj1")
        self.assertEqual(snap.status_label, "empty")
        self.assertFalse(snap.has_ready_chunks)

    async def test_ready_status(self):
        """Has chunks, no active jobs → ready."""
        self._write_vdb("proj1", data_count=3)
        snap = await self._get_snapshot("proj1")
        self.assertEqual(snap.status_label, "ready")
        self.assertTrue(snap.has_ready_chunks)

    async def test_building_status_queued(self):
        """Has queued job → building."""
        await self.repo.create_job("proj1", "test.pdf", "pdf")
        snap = await self._get_snapshot("proj1")
        self.assertEqual(snap.status_label, "building")
        self.assertTrue(snap.has_pending_jobs)

    async def test_building_status_running(self):
        """Has running job → building."""
        await self.repo.create_job("proj1", "test.pdf", "pdf")
        await self.repo.claim_next_job()
        snap = await self._get_snapshot("proj1")
        self.assertEqual(snap.status_label, "building")
        self.assertTrue(snap.has_running_jobs)

    async def test_building_status_running_waiting_backoff(self):
        """Has running:* sub-status → still building."""
        j = await self.repo.create_job("proj1", "test.pdf", "pdf")
        await self.repo.claim_next_job()
        await self.repo.update_status(
            j,
            "running:extracting:waiting_backoff",
            last_error="rate limit",
            phase="extracting",
        )
        snap = await self._get_snapshot("proj1")
        self.assertEqual(snap.status_label, "building")
        self.assertTrue(snap.has_running_jobs)

    async def test_degraded_status_with_chunks(self):
        """Has chunks + terminal failure → degraded."""
        self._write_vdb("proj1", data_count=2)
        j = await self.repo.create_job("proj1", "bad.pdf", "pdf")
        await self.repo.claim_next_job()
        await self.repo.update_status(j, "failed", last_error="permanent error", next_retry_at=None)
        snap = await self._get_snapshot("proj1")
        self.assertEqual(snap.status_label, "degraded")
        self.assertTrue(snap.has_ready_chunks)
        self.assertTrue(snap.has_terminal_failures)
        self.assertIn("permanent error", snap.last_error)

    async def test_degraded_status_no_chunks(self):
        """No chunks + terminal failure → degraded."""
        j = await self.repo.create_job("proj1", "bad.pdf", "pdf")
        await self.repo.claim_next_job()
        await self.repo.update_status(j, "failed", last_error="perm", next_retry_at=None)
        snap = await self._get_snapshot("proj1")
        self.assertEqual(snap.status_label, "degraded")

    async def test_building_with_partial_chunks(self):
        """Has chunks + running job → building (not ready)."""
        self._write_vdb("proj1", data_count=1)
        await self.repo.create_job("proj1", "more.pdf", "pdf")
        snap = await self._get_snapshot("proj1")
        self.assertEqual(snap.status_label, "building")
        self.assertTrue(snap.has_ready_chunks)


# ── 3. Startup embedding check pure diagnostics ─────────────────────


class TestStartupEmbeddingPureDiagnostics(unittest.TestCase):
    """Startup check must NOT call ensure_model_downloaded."""

    def test_startup_check_source_no_ensure_download(self):
        """app.py::_check_embedding_model_ready must not import ensure_model_downloaded."""
        import inspect
        # Read app.py source to find the method
        app_src = Path(ROOT / "app.py").read_text(encoding="utf-8")
        # Find the method body
        start = app_src.index("async def _check_embedding_model_ready")
        # Find next method or end
        next_method = app_src.find("\n    async def ", start + 10)
        if next_method == -1:
            next_method = app_src.find("\n    def ", start + 10)
        if next_method == -1:
            next_method = len(app_src)
        method_body = app_src[start:next_method]

        self.assertNotIn("ensure_model_downloaded", method_body,
                         "Startup check must not call ensure_model_downloaded")
        self.assertIn("get_device_diagnostics", method_body,
                      "Startup check should use get_device_diagnostics")


# ── 4. Transient errors do NOT trigger destructive repair ────────────


class TestTransientNoDestructiveRepair(unittest.TestCase):
    """TransientErrorPolicy must never allow destructive repair."""

    def test_fds_to_keep_no_repair(self):
        from rag.repair_policy import TransientErrorPolicy
        policy = TransientErrorPolicy()
        for err_msg in ("fds_to_keep", "timeout exceeded", "rate limit", "connection refused"):
            err = RuntimeError(err_msg)
            diag = {"healthy": False, "issues": ["some issue"]}
            self.assertFalse(policy.should_repair(diag, err),
                             f"Transient error '{err_msg}' must not trigger repair")

    def test_auto_repair_skips_transient(self):
        """auto_repair with transient triggering_error should return skip."""
        from rag.repair import auto_repair, _repair_attempted, _repairing

        async def _run():
            # Clean state
            _repair_attempted.discard("test_transient")
            _repairing.discard("test_transient")

            result = await auto_repair(
                "test_transient",
                triggering_error=RuntimeError("fds_to_keep subprocess error"),
            )
            return result

        result = asyncio.run(_run())
        self.assertEqual(result["status"], "skip")
        self.assertEqual(result["reason"], "transient_error")


# ── 5. Atomic claim: dual-worker competition ─────────────────────────


class TestAtomicClaim(unittest.IsolatedAsyncioTestCase):
    """Two concurrent claim_next_job calls must not claim the same job."""

    async def asyncSetUp(self):
        self._tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self._tmp.close()
        self._db_path = Path(self._tmp.name)

        from rag.job_repository import JobRepository
        self.repo = JobRepository(self._db_path)
        await self.repo.ensure_table()

    async def asyncTearDown(self):
        os.unlink(self._db_path)

    async def test_no_double_claim(self):
        """Single queued job, two concurrent claims → only one succeeds."""
        await self.repo.create_job("proj1", "test.pdf", "pdf")

        # Launch two concurrent claims
        results = await asyncio.gather(
            self.repo.claim_next_job(),
            self.repo.claim_next_job(),
        )
        # Exactly one should get the job, the other gets None
        non_none = [r for r in results if r is not None]
        self.assertEqual(len(non_none), 1,
                         f"Expected exactly 1 successful claim, got {len(non_none)}")

    async def test_two_jobs_two_claims(self):
        """Two queued jobs, two sequential claims → each gets one."""
        await self.repo.create_job("proj1", "a.pdf", "pdf")
        await self.repo.create_job("proj1", "b.pdf", "pdf")

        r1 = await self.repo.claim_next_job()
        r2 = await self.repo.claim_next_job()
        self.assertIsNotNone(r1)
        self.assertIsNotNone(r2)
        self.assertNotEqual(r1["id"], r2["id"])


# ── 6. Same-second completed polling ─────────────────────────────────
# (Preserved from test_ingestion_jobs.py, re-tested here for completeness)


class TestSameSecondPolling(unittest.IsolatedAsyncioTestCase):

    async def asyncSetUp(self):
        self._tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self._tmp.close()
        self._db_path = Path(self._tmp.name)

        from rag.job_repository import JobRepository
        self.repo = JobRepository(self._db_path)
        await self.repo.ensure_table()

    async def asyncTearDown(self):
        os.unlink(self._db_path)

    async def test_same_second_no_miss(self):
        j1 = await self.repo.create_job("p1", "a.pdf", "pdf")
        j2 = await self.repo.create_job("p1", "b.pdf", "pdf")
        await self.repo.claim_next_job()
        await self.repo.update_status(j1, "done")
        await self.repo.claim_next_job()
        await self.repo.update_status(j2, "done")

        completed = await self.repo.list_completed_since(None)
        ids = {j["id"] for j in completed}
        self.assertIn(j1, ids)
        self.assertIn(j2, ids)


# ── 7. Chunk ready + KG incomplete → still retrievable ───────────────


class TestChunkOkKgIncomplete(unittest.TestCase):
    """done_with_warning scenario: chunks indexed but KG failed."""

    def setUp(self):
        self._tmpdir = tempfile.mkdtemp()
        self._lightrag_dir = Path(self._tmpdir) / "lightrag"
        self._lightrag_dir.mkdir(parents=True)

    def tearDown(self):
        import shutil
        shutil.rmtree(self._tmpdir)

    def test_has_knowledge_true_with_chunks_no_kg(self):
        """vdb_chunks has data, no KG files → has_knowledge still True."""
        vdb = {
            "embedding_dim": 1024,
            "data": [{"id": "c1", "content": "test", "embedding": [0.1] * 1024}],
        }
        (self._lightrag_dir / "vdb_chunks.json").write_text(
            json.dumps(vdb), encoding="utf-8"
        )
        with patch("rag.knowledge_graph.project_root", return_value=Path(self._tmpdir)):
            from rag.knowledge_graph import has_knowledge
            self.assertTrue(has_knowledge(project_id="test_project"))

    def test_readiness_ready_with_chunks_no_kg(self):
        """ReadinessService reports ready when chunks exist even without KG."""
        vdb = {
            "embedding_dim": 1024,
            "data": [{"id": "c1"}],
        }
        (self._lightrag_dir / "vdb_chunks.json").write_text(
            json.dumps(vdb), encoding="utf-8"
        )
        from rag.readiness import IngestionReadinessService
        svc = IngestionReadinessService()
        with patch("paths.project_root", return_value=Path(self._tmpdir)):
            snap = svc.get_snapshot_sync("test_project")
            self.assertEqual(snap.status_label, "ready")
            self.assertTrue(snap.has_ready_chunks)


# ── Additional: search/chat readiness consistency ────────────────────


class TestSearchChatReadinessConsistency(unittest.TestCase):
    """Both _do_search and _inject_rag_context must use the same readiness gate."""

    def test_do_search_uses_readiness(self):
        """_do_search source code uses get_readiness_service."""
        import inspect
        from mixins.rag import RAGMixin
        source = inspect.getsource(RAGMixin._do_search.__wrapped__)
        self.assertIn("get_readiness_service", source)
        self.assertIn("status_label", source)

    def test_inject_rag_uses_readiness(self):
        """_inject_rag_context source code uses get_readiness_service."""
        import inspect
        from mixins.chat import ChatMixin
        source = inspect.getsource(ChatMixin._inject_rag_context)
        self.assertIn("get_readiness_service", source)
        self.assertIn("status_label", source)


# ── Additional: embedding timeout configured ─────────────────────────


class TestEmbeddingTimeoutConfig(unittest.TestCase):
    """LightRAG creation should use configurable embedding timeout."""

    def test_knowledge_graph_sets_embedding_timeout(self):
        """get_rag source should set default_embedding_timeout."""
        src = Path(ROOT / "rag" / "knowledge_graph.py").read_text(encoding="utf-8")
        self.assertIn("default_embedding_timeout", src,
                      "knowledge_graph.py must set default_embedding_timeout on LightRAG")
        self.assertIn("LIGHTRAG_EMBEDDING_TIMEOUT", src,
                      "Timeout should be configurable via env var")


# ── 8. Readiness gate controls memory/image/pipeline calls ───────────


class TestInjectRagReadinessGate(unittest.TestCase):
    """_inject_rag_context must skip memory/image/pipeline when readiness says so."""

    def _get_source(self):
        import inspect
        from mixins.chat import ChatMixin
        return inspect.getsource(ChatMixin._inject_rag_context)

    def test_early_return_on_skip_all(self):
        """When _skip_all is True, the method returns before memory/image/pipeline."""
        src = self._get_source()
        # _skip_all early return must come before "記憶向量搜尋"
        skip_all_pos = src.index("_skip_all")
        early_return_pos = src.index("if _skip_all:")
        memory_pos = src.index("記憶向量搜尋")
        self.assertLess(early_return_pos, memory_pos,
                        "Early return for _skip_all must precede memory search")

    def test_skip_augment_guards_memory(self):
        """Memory search is guarded by 'if not _skip_augment'."""
        src = self._get_source()
        # Find "記憶向量搜尋" and check it's inside an "if not _skip_augment" block
        self.assertIn("if not _skip_augment:", src)
        augment_guard_pos = src.index("if not _skip_augment:")
        memory_pos = src.index("search_similar")
        self.assertLess(augment_guard_pos, memory_pos,
                        "_skip_augment guard must come before search_similar call")

    def test_skip_augment_guards_image(self):
        """Image search is guarded by 'if not _skip_augment'."""
        src = self._get_source()
        # There should be two "if not _skip_augment:" blocks
        first = src.index("if not _skip_augment:")
        second = src.index("if not _skip_augment:", first + 1)
        image_pos = src.index("search_images")
        self.assertLess(second, image_pos,
                        "Second _skip_augment guard must come before search_images")

    def test_building_fast_skips_augment(self):
        """Building + fast mode sets _skip_augment = True."""
        src = self._get_source()
        # The logic: building + _is_fast → _skip_augment = True
        self.assertIn("_skip_augment = True", src)
        # Verify building triggers skip_augment when fast
        building_section_start = src.index('"building"')
        skip_augment_in_building = src.index("_skip_augment = True", building_section_start)
        self.assertIsNotNone(skip_augment_in_building)

    def test_readiness_checked_before_all_augment(self):
        """Readiness snapshot is fetched before memory/image/pipeline."""
        src = self._get_source()
        readiness_pos = src.index("get_readiness_service")
        memory_pos = src.index("search_similar")
        self.assertLess(readiness_pos, memory_pos,
                        "Readiness must be checked before memory search")


# ── 9. Env guard uses hard overwrite ─────────────────────────────────


class TestEnvGuardHardOverwrite(unittest.TestCase):
    """HF_HUB_DISABLE_XET etc. must use hard overwrite, not setdefault."""

    def _assert_hard_overwrite(self, filepath: Path, varname: str):
        src = filepath.read_text(encoding="utf-8")
        # Must contain os.environ["VAR"] = "val" (hard overwrite)
        hard_pattern = f'os.environ["{varname}"]'
        self.assertIn(hard_pattern, src,
                      f"{filepath.name} must hard-overwrite {varname}")
        # Must NOT contain os.environ.setdefault("VAR", ...) for this var
        soft_pattern = f'os.environ.setdefault("{varname}"'
        self.assertNotIn(soft_pattern, src,
                         f"{filepath.name} must not use setdefault for {varname}")

    def test_local_py_hard_overwrite(self):
        self._assert_hard_overwrite(ROOT / "embeddings" / "local.py", "HF_HUB_DISABLE_XET")
        self._assert_hard_overwrite(ROOT / "embeddings" / "local.py", "HF_HUB_ENABLE_HF_TRANSFER")
        self._assert_hard_overwrite(ROOT / "embeddings" / "local.py", "TOKENIZERS_PARALLELISM")

    def test_worker_py_hard_overwrite(self):
        self._assert_hard_overwrite(ROOT / "rag" / "ingestion_worker.py", "HF_HUB_DISABLE_XET")
        self._assert_hard_overwrite(ROOT / "rag" / "ingestion_worker.py", "HF_HUB_ENABLE_HF_TRANSFER")
        self._assert_hard_overwrite(ROOT / "rag" / "ingestion_worker.py", "TOKENIZERS_PARALLELISM")


if __name__ == "__main__":
    unittest.main(verbosity=2)
