"""Ingestion jobs 測試。"""

import asyncio
import json
import os
import sys
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag.job_executor import ErrorCategory, classify_error
from rag.repair_policy import TransientErrorPolicy, CorruptionPolicy


class TestErrorClassification(unittest.TestCase):
    """Test error classification logic."""

    def test_fds_to_keep_classified_transient(self):
        err = RuntimeError("some error about fds_to_keep in subprocess")
        self.assertEqual(classify_error(err), ErrorCategory.TRANSIENT)

    def test_timeout_classified_transient(self):
        err = TimeoutError("connection timeout after 30s")
        self.assertEqual(classify_error(err), ErrorCategory.TRANSIENT)

    def test_rate_limit_classified_transient(self):
        err = RuntimeError("rate limit exceeded, retry after 60s")
        self.assertEqual(classify_error(err), ErrorCategory.TRANSIENT)

    def test_connection_classified_transient(self):
        err = ConnectionError("connection refused")
        self.assertEqual(classify_error(err), ErrorCategory.TRANSIENT)

    def test_not_found_classified_permanent(self):
        err = FileNotFoundError("file not found: /tmp/missing.pdf")
        self.assertEqual(classify_error(err), ErrorCategory.PERMANENT)

    def test_auth_401_classified_permanent(self):
        err = RuntimeError("HTTP 401 Unauthorized")
        self.assertEqual(classify_error(err), ErrorCategory.PERMANENT)

    def test_auth_403_classified_permanent(self):
        err = RuntimeError("403 Forbidden for URL")
        self.assertEqual(classify_error(err), ErrorCategory.PERMANENT)

    def test_unknown_error_defaults_transient(self):
        err = RuntimeError("something weird happened")
        self.assertEqual(classify_error(err), ErrorCategory.TRANSIENT)


class TestTransientPolicy(unittest.TestCase):
    """Test TransientErrorPolicy blocks destructive repair."""

    def test_transient_policy_blocks_repair_fds(self):
        policy = TransientErrorPolicy()
        diag = {"healthy": False, "issues": ["some issue"]}
        err = RuntimeError("fds_to_keep error in subprocess")
        self.assertFalse(policy.should_repair(diag, err))

    def test_transient_policy_blocks_repair_timeout(self):
        policy = TransientErrorPolicy()
        diag = {"healthy": False, "issues": ["some issue"]}
        err = TimeoutError("timeout")
        self.assertFalse(policy.should_repair(diag, err))

    def test_transient_policy_blocks_repair_rate_limit(self):
        policy = TransientErrorPolicy()
        diag = {"healthy": False, "issues": ["some issue"]}
        err = RuntimeError("rate limit exceeded")
        self.assertFalse(policy.should_repair(diag, err))

    def test_transient_policy_blocks_repair_no_error(self):
        policy = TransientErrorPolicy()
        diag = {"healthy": False, "issues": ["some issue"]}
        self.assertFalse(policy.should_repair(diag, None))


class TestCorruptionPolicy(unittest.TestCase):
    """Test CorruptionPolicy triggers repair correctly."""

    def test_corruption_empty_vdb_triggers_repair(self):
        policy = CorruptionPolicy()
        diag = {
            "healthy": False,
            "issues": ["知識庫有 3 份文件但向量索引為空（vdb_chunks=0）"],
        }
        self.assertTrue(policy.should_repair(diag))

    def test_corruption_dim_mismatch_triggers_repair(self):
        policy = CorruptionPolicy()
        diag = {
            "healthy": False,
            "issues": ["向量維度不符"],
            "dim_mismatch": {"existing": 768, "expected": 1024},
        }
        self.assertTrue(policy.should_repair(diag))

    def test_healthy_kb_no_repair(self):
        policy = CorruptionPolicy()
        diag = {"healthy": True, "issues": []}
        self.assertFalse(policy.should_repair(diag))


class TestJobRepository(unittest.IsolatedAsyncioTestCase):
    """Test JobRepository CRUD operations."""

    async def asyncSetUp(self):
        self._tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self._tmp.close()
        self._db_path = Path(self._tmp.name)

        from rag.job_repository import JobRepository
        self.repo = JobRepository(self._db_path)
        await self.repo.ensure_table()

    async def asyncTearDown(self):
        os.unlink(self._db_path)

    async def test_state_machine_transitions(self):
        """queued → running → done"""
        job_id = await self.repo.create_job("proj1", "http://example.com", "url")
        job = await self.repo.get_job(job_id)
        self.assertEqual(job["status"], "queued")

        claimed = await self.repo.claim_next_job()
        self.assertIsNotNone(claimed)
        self.assertEqual(claimed["id"], job_id)
        # After claim, status is running
        job = await self.repo.get_job(job_id)
        self.assertEqual(job["status"], "running")

        await self.repo.update_status(job_id, "done")
        job = await self.repo.get_job(job_id)
        self.assertEqual(job["status"], "done")

    async def test_worker_resumes_stale_running(self):
        """Stale running jobs get reset to queued."""
        job_id = await self.repo.create_job("proj1", "test.pdf", "pdf")
        # Claim it (running)
        await self.repo.claim_next_job()
        job = await self.repo.get_job(job_id)
        self.assertEqual(job["status"], "running")

        # Fake the updated_at to be old
        import aiosqlite
        async with aiosqlite.connect(self._db_path) as db:
            old_time = (datetime.now() - timedelta(seconds=700)).strftime("%Y-%m-%d %H:%M:%S")
            await db.execute(
                "UPDATE ingest_jobs SET updated_at = ? WHERE id = ?",
                (old_time, job_id),
            )
            await db.commit()

        reset_count = await self.repo.reset_stale_running(stale_seconds=600)
        self.assertEqual(reset_count, 1)

        job = await self.repo.get_job(job_id)
        self.assertEqual(job["status"], "queued")

    async def test_retry_backoff_timing(self):
        """Retry backoff produces correct timing."""
        # attempts=0 → 30s, attempts=1 → 120s, attempts=2 → 300s
        t0 = self.repo.compute_next_retry_at(0)
        self.assertIsNotNone(t0)
        t1 = self.repo.compute_next_retry_at(1)
        self.assertIsNotNone(t1)
        t2 = self.repo.compute_next_retry_at(2)
        self.assertIsNotNone(t2)
        # attempts=3 → None (max exceeded)
        t3 = self.repo.compute_next_retry_at(3)
        self.assertIsNone(t3)

        # Verify ordering: t0 < t1 < t2
        from datetime import datetime as dt
        fmt = "%Y-%m-%d %H:%M:%S"
        self.assertLess(dt.strptime(t0, fmt), dt.strptime(t1, fmt))
        self.assertLess(dt.strptime(t1, fmt), dt.strptime(t2, fmt))

    async def test_fds_to_keep_job_retries(self):
        """fds_to_keep error → job scheduled for retry with next_retry_at."""
        job_id = await self.repo.create_job("proj1", "http://example.com", "url")
        claimed = await self.repo.claim_next_job()
        self.assertIsNotNone(claimed)

        # Simulate fds_to_keep failure
        next_retry = self.repo.compute_next_retry_at(1)
        await self.repo.update_status(
            job_id, "failed",
            last_error="fds_to_keep error in subprocess",
            next_retry_at=next_retry,
        )

        job = await self.repo.get_job(job_id)
        self.assertEqual(job["status"], "failed")
        self.assertIsNotNone(job["next_retry_at"])
        self.assertIn("fds_to_keep", job["last_error"])

    async def test_completed_polling_same_second(self):
        """list_completed_since with >= and dedup handles same-second completions."""
        # Create two jobs and complete them (will have same updated_at at second granularity)
        j1 = await self.repo.create_job("proj1", "a.pdf", "pdf")
        j2 = await self.repo.create_job("proj1", "b.pdf", "pdf")

        await self.repo.claim_next_job()  # claims j1
        await self.repo.update_status(j1, "done")
        await self.repo.claim_next_job()  # claims j2
        await self.repo.update_status(j2, "done")

        # Both should appear (same second, >= comparison)
        completed = await self.repo.list_completed_since(None)
        ids = {j["id"] for j in completed}
        self.assertIn(j1, ids)
        self.assertIn(j2, ids)

        # Using the first job's timestamp, j2 should also appear (>= not >)
        ts = completed[0]["updated_at"]
        completed2 = await self.repo.list_completed_since(ts)
        ids2 = {j["id"] for j in completed2}
        self.assertIn(j1, ids2)  # >= includes same timestamp
        self.assertIn(j2, ids2)

    async def test_result_json_stored(self):
        """update_result stores structured data on the job row."""
        job_id = await self.repo.create_job("proj1", "test.pdf", "pdf")
        await self.repo.claim_next_job()

        result_data = {"title": "Test PDF", "doc_id": "abc123", "file_size": 1024}
        await self.repo.update_result(job_id, json.dumps(result_data))
        await self.repo.update_status(job_id, "done")

        job = await self.repo.get_job(job_id)
        self.assertEqual(job["status"], "done")
        parsed = json.loads(job["result_json"])
        self.assertEqual(parsed["title"], "Test PDF")
        self.assertEqual(parsed["doc_id"], "abc123")


# ── Integration Tests ──────────────────────────────────────────────


class TestBulkImportViaQueue(unittest.IsolatedAsyncioTestCase):
    """Bulk import should route through the job queue, not call insert_* directly."""

    async def asyncSetUp(self):
        self._tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self._tmp.close()
        self._db_path = Path(self._tmp.name)

        from rag.job_repository import JobRepository
        self.repo = JobRepository(self._db_path)
        await self.repo.ensure_table()

    async def asyncTearDown(self):
        os.unlink(self._db_path)

    async def test_bulk_import_all_types_go_through_queue(self):
        """Submit pdf/url/text/doi jobs — all should create queued rows."""
        sources = [
            ("proj1", "/tmp/test.pdf", "pdf", "", "{}"),
            ("proj1", "http://example.com/article", "url", "My Article", "{}"),
            ("proj1", "manual:My Notes", "text", "My Notes", json.dumps({"content": "some text"})),
            ("proj1", "10.1234/test.doi", "doi", "", "{}"),
        ]
        job_ids = []
        for pid, src, stype, title, payload in sources:
            jid = await self.repo.create_job(pid, src, stype, title, payload)
            job_ids.append(jid)

        # All should be queued
        for jid in job_ids:
            job = await self.repo.get_job(jid)
            self.assertEqual(job["status"], "queued")

        # claim_next_job processes them FIFO
        for i, jid in enumerate(job_ids):
            claimed = await self.repo.claim_next_job()
            self.assertIsNotNone(claimed, f"Expected to claim job {i}")
            self.assertEqual(claimed["id"], jid)
            self.assertEqual(claimed["source_type"], sources[i][2])
            await self.repo.update_status(jid, "done")

        # No more to claim
        self.assertIsNone(await self.repo.claim_next_job())


class TestSingleWorkerMultiProject(unittest.IsolatedAsyncioTestCase):
    """Single worker should process jobs from multiple projects via project_id routing."""

    async def asyncSetUp(self):
        self._tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self._tmp.close()
        self._db_path = Path(self._tmp.name)

        from rag.job_repository import JobRepository
        self.repo = JobRepository(self._db_path)
        await self.repo.ensure_table()

    async def asyncTearDown(self):
        os.unlink(self._db_path)

    async def test_multi_project_jobs_all_processed(self):
        """Jobs from different projects are all claimed and processed."""
        j_a1 = await self.repo.create_job("project_alpha", "file_a.pdf", "pdf")
        j_b1 = await self.repo.create_job("project_beta", "http://b.com", "url")
        j_a2 = await self.repo.create_job("project_alpha", "file_a2.pdf", "pdf")
        j_c1 = await self.repo.create_job("project_gamma", "text content", "text")

        # Worker claims jobs regardless of project_id, FIFO order
        processed_projects = []
        for _ in range(4):
            job = await self.repo.claim_next_job()
            self.assertIsNotNone(job)
            processed_projects.append(job["project_id"])
            await self.repo.update_status(job["id"], "done")

        # All three projects were processed
        self.assertIn("project_alpha", processed_projects)
        self.assertIn("project_beta", processed_projects)
        self.assertIn("project_gamma", processed_projects)
        # Alpha had 2 jobs
        self.assertEqual(processed_projects.count("project_alpha"), 2)

        # No more jobs
        self.assertIsNone(await self.repo.claim_next_job())

    async def test_multi_project_with_failure_isolation(self):
        """A failure in one project doesn't block jobs in another."""
        j_a = await self.repo.create_job("project_alpha", "bad.pdf", "pdf")
        j_b = await self.repo.create_job("project_beta", "good.pdf", "pdf")

        # Process project_alpha's job → fails
        job_a = await self.repo.claim_next_job()
        self.assertEqual(job_a["project_id"], "project_alpha")
        await self.repo.update_status(job_a["id"], "failed", last_error="some error")

        # Project beta's job is still claimable
        job_b = await self.repo.claim_next_job()
        self.assertIsNotNone(job_b)
        self.assertEqual(job_b["project_id"], "project_beta")
        await self.repo.update_status(job_b["id"], "done")


class TestChunkOkKgFailStillRetrievable(unittest.TestCase):
    """When KG extraction fails but chunks are indexed, has_knowledge should return True.

    This tests the 'done_with_warning' scenario:
    - LightRAG chunks + embeds successfully → vdb_chunks.json has data
    - KG extraction fails → warning recorded
    - has_knowledge() checks vdb_chunks.json data array → True
    """

    def setUp(self):
        self._tmpdir = tempfile.mkdtemp()
        self._lightrag_dir = Path(self._tmpdir) / "lightrag"
        self._lightrag_dir.mkdir(parents=True)

    def tearDown(self):
        import shutil
        shutil.rmtree(self._tmpdir)

    def test_has_knowledge_true_when_chunks_exist(self):
        """vdb_chunks.json has data → has_knowledge returns True regardless of KG state."""
        # Write vdb_chunks.json with data (simulates successful chunk+embed)
        vdb = {
            "embedding_dim": 1024,
            "data": [
                {"id": "chunk1", "content": "test content", "embedding": [0.1] * 1024}
            ],
        }
        (self._lightrag_dir / "vdb_chunks.json").write_text(
            json.dumps(vdb), encoding="utf-8"
        )

        # Patch project_root to point to our temp dir
        with patch("rag.knowledge_graph.project_root", return_value=Path(self._tmpdir)):
            from rag.knowledge_graph import has_knowledge
            self.assertTrue(has_knowledge(project_id="test_project"))

    def test_has_knowledge_false_when_chunks_empty(self):
        """vdb_chunks.json exists but data is empty → has_knowledge returns False."""
        vdb = {"embedding_dim": 1024, "data": []}
        (self._lightrag_dir / "vdb_chunks.json").write_text(
            json.dumps(vdb), encoding="utf-8"
        )

        with patch("rag.knowledge_graph.project_root", return_value=Path(self._tmpdir)):
            from rag.knowledge_graph import has_knowledge
            self.assertFalse(has_knowledge(project_id="test_project"))

    def test_has_knowledge_false_when_no_vdb(self):
        """No vdb_chunks.json at all → has_knowledge returns False."""
        with patch("rag.knowledge_graph.project_root", return_value=Path(self._tmpdir)):
            from rag.knowledge_graph import has_knowledge
            self.assertFalse(has_knowledge(project_id="test_project"))

    def test_done_with_warning_state_still_retrievable(self):
        """Simulate done_with_warning: chunks exist + KG missing → knowledge available."""
        # Write chunks (chunk+embed succeeded)
        vdb = {
            "embedding_dim": 1024,
            "data": [
                {"id": "c1", "content": "藝術理論討論", "embedding": [0.5] * 1024},
                {"id": "c2", "content": "設計方法論", "embedding": [0.3] * 1024},
            ],
        }
        (self._lightrag_dir / "vdb_chunks.json").write_text(
            json.dumps(vdb), encoding="utf-8"
        )
        # No KG files (graph_chunk_entity_relation.graphml etc.) — simulates KG failure
        # has_knowledge should still return True
        with patch("rag.knowledge_graph.project_root", return_value=Path(self._tmpdir)):
            from rag.knowledge_graph import has_knowledge
            self.assertTrue(has_knowledge(project_id="test_project"))


class TestPollCompletedDedup(unittest.IsolatedAsyncioTestCase):
    """IngestionService.poll_completed should dedup by job ID across polls."""

    async def asyncSetUp(self):
        self._tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self._tmp.close()
        self._db_path = Path(self._tmp.name)

        from rag.job_repository import JobRepository
        self.repo = JobRepository(self._db_path)
        await self.repo.ensure_table()

        from rag.ingestion_service import IngestionService
        self.svc = IngestionService(self._db_path)

    async def asyncTearDown(self):
        os.unlink(self._db_path)

    async def test_same_second_not_missed(self):
        """Two jobs completed in the same second both get reported."""
        j1 = await self.repo.create_job("proj1", "a.pdf", "pdf")
        j2 = await self.repo.create_job("proj1", "b.pdf", "pdf")
        await self.repo.claim_next_job()
        await self.repo.update_status(j1, "done")
        await self.repo.claim_next_job()
        await self.repo.update_status(j2, "done")

        completed = await self.svc.poll_completed()
        ids = {j["id"] for j in completed}
        self.assertIn(j1, ids)
        self.assertIn(j2, ids)

    async def test_no_duplicate_on_repoll(self):
        """Already-reported jobs are not returned on subsequent polls."""
        j1 = await self.repo.create_job("proj1", "a.pdf", "pdf")
        await self.repo.claim_next_job()
        await self.repo.update_status(j1, "done")

        first = await self.svc.poll_completed()
        self.assertEqual(len(first), 1)

        # Second poll — same job should NOT appear again
        second = await self.svc.poll_completed()
        self.assertEqual(len(second), 0)

        # New job completes → should appear
        j2 = await self.repo.create_job("proj1", "b.pdf", "pdf")
        await self.repo.claim_next_job()
        await self.repo.update_status(j2, "done")

        third = await self.svc.poll_completed()
        self.assertEqual(len(third), 1)
        self.assertEqual(third[0]["id"], j2)


if __name__ == "__main__":
    unittest.main()
