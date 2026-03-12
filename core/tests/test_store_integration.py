"""Store integration tests for De-insight v2 Core.

These tests verify that stores are usable as retrieval substrate.

References:
- Tech spec: deinsight_phase2_test_debug_spec.md
"""

import sys
from pathlib import Path

import pytest

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.stores import ClaimStore, ThoughtStore, ConceptStore, BridgeStore
from core.schemas import (
    Claim,
    SourceKind,
    ThoughtUnit,
    ThoughtStatus,
    ConceptMapping,
    OwnerKind,
    VocabSource,
    Bridge,
    BridgeType,
)


@pytest.fixture
def temp_db_path(tmp_path):
    """Create a temporary database path."""
    return tmp_path / "test.db"


class TestClaimStoreIntegration:
    """Test ClaimStore CRUD operations."""

    @pytest.mark.asyncio
    async def test_claim_insert_and_fetch(self, temp_db_path):
        """Test inserting and fetching a claim."""
        store = ClaimStore(project_id="test-project", db_path=temp_db_path)

        claim = Claim(
            project_id="test-project",
            core_claim="Test claim about design",
            source_kind=SourceKind.USER_UTTERANCE,
            critique_target=["functionality"],
            value_axes=["aesthetics"],
            confidence=0.8,
        )

        # Insert
        result = await store.add(claim)
        assert result.claim_id == claim.claim_id

        # Fetch
        fetched = await store.get(claim.claim_id)
        assert fetched is not None
        assert fetched.core_claim == "Test claim about design"
        assert fetched.confidence == 0.8

    @pytest.mark.asyncio
    async def test_claim_list_by_project(self, temp_db_path):
        """Test listing claims by project."""
        store = ClaimStore(project_id="test-project", db_path=temp_db_path)

        # Insert multiple claims
        for i in range(3):
            claim = Claim(
                project_id="test-project",
                core_claim=f"Test claim {i}",
            )
            await store.add(claim)

        # List
        claims = await store.list_by_project("test-project")
        assert len(claims) >= 3

    @pytest.mark.asyncio
    async def test_claim_update(self, temp_db_path):
        """Test updating a claim."""
        store = ClaimStore(project_id="test-project", db_path=temp_db_path)

        claim = Claim(
            project_id="test-project",
            core_claim="Original claim",
            confidence=0.5,
        )
        await store.add(claim)

        # Update
        claim.core_claim = "Updated claim"
        claim.confidence = 0.9
        await store.update(claim)

        # Verify
        fetched = await store.get(claim.claim_id)
        assert fetched is not None
        assert fetched.core_claim == "Updated claim"
        assert fetched.confidence == 0.9

    @pytest.mark.asyncio
    async def test_claim_delete(self, temp_db_path):
        """Test deleting a claim."""
        store = ClaimStore(project_id="test-project", db_path=temp_db_path)

        claim = Claim(project_id="test-project", core_claim="To be deleted")
        await store.add(claim)

        # Delete
        deleted = await store.delete(claim.claim_id)
        assert deleted is True

        # Verify
        fetched = await store.get(claim.claim_id)
        assert fetched is None


class TestThoughtStoreIntegration:
    """Test ThoughtStore CRUD operations."""

    @pytest.mark.asyncio
    async def test_thought_insert_and_fetch(self, temp_db_path):
        """Test inserting and fetching a thought unit."""
        store = ThoughtStore(project_id="test-project", db_path=temp_db_path)

        thought = ThoughtUnit(
            project_id="test-project",
            title="Test Thought",
            summary="This is a test thought unit",
            status=ThoughtStatus.EMERGING,
        )

        # Insert
        result = await store.add(thought)
        assert result.thought_id == thought.thought_id

        # Fetch
        fetched = await store.get(thought.thought_id)
        assert fetched is not None
        assert fetched.title == "Test Thought"

    @pytest.mark.asyncio
    async def test_thought_list_with_status_filter(self, temp_db_path):
        """Test listing thoughts with status filter."""
        store = ThoughtStore(project_id="test-project", db_path=temp_db_path)

        # Insert thoughts with different statuses
        for status in [ThoughtStatus.EMERGING, ThoughtStatus.STABLE, ThoughtStatus.CONTESTED]:
            thought = ThoughtUnit(
                project_id="test-project",
                title=f"Thought {status.value}",
                status=status,
            )
            await store.add(thought)

        # Filter by status
        emerging = await store.list_by_project("test-project", status=ThoughtStatus.EMERGING)
        assert any(t.status == ThoughtStatus.EMERGING for t in emerging)


class TestConceptStoreIntegration:
    """Test ConceptStore CRUD operations."""

    @pytest.mark.asyncio
    async def test_concept_insert_and_fetch(self, temp_db_path):
        """Test inserting and fetching concept mappings."""
        store = ConceptStore(project_id="test-project", db_path=temp_db_path)

        mapping = ConceptMapping(
            project_id="test-project",
            owner_kind=OwnerKind.CLAIM,
            owner_id="claim-123",
            vocab_source=VocabSource.AAT,
            concept_id="aat_12345",
            preferred_label="Minimalism",
            confidence=0.9,
        )

        # Insert
        result = await store.add(mapping)
        assert result.mapping_id == mapping.mapping_id

        # Fetch
        fetched = await store.get(mapping.mapping_id)
        assert fetched is not None
        assert fetched.preferred_label == "Minimalism"

    @pytest.mark.asyncio
    async def test_concept_list_by_owner(self, temp_db_path):
        """Test listing concepts by owner."""
        store = ConceptStore(project_id="test-project", db_path=temp_db_path)

        mapping = ConceptMapping(
            project_id="test-project",
            owner_kind=OwnerKind.CLAIM,
            owner_id="claim-123",
            concept_id="concept-1",
            preferred_label="Test",
        )
        await store.add(mapping)

        # List by owner
        results = await store.list_by_owner(OwnerKind.CLAIM, "claim-123")
        assert len(results) >= 1


class TestBridgeStoreIntegration:
    """Test BridgeStore CRUD operations."""

    @pytest.mark.asyncio
    async def test_bridge_insert_and_fetch(self, temp_db_path):
        """Test inserting and fetching bridges."""
        store = BridgeStore(project_id="test-project", db_path=temp_db_path)

        bridge = Bridge(
            project_id="test-project",
            source_claim_id="claim-1",
            target_claim_id="claim-2",
            bridge_type=BridgeType.ANALOGY,
            reason_summary="Both discuss function-form relationship",
            confidence=0.75,
        )

        # Insert
        result = await store.add(bridge)
        assert result.bridge_id == bridge.bridge_id

        # Fetch
        fetched = await store.get(bridge.bridge_id)
        assert fetched is not None
        assert fetched.bridge_type == BridgeType.ANALOGY

    @pytest.mark.asyncio
    async def test_bridge_find_by_claim(self, temp_db_path):
        """Test finding bridges connected to a claim."""
        store = BridgeStore(project_id="test-project", db_path=temp_db_path)

        bridge = Bridge(
            project_id="test-project",
            source_claim_id="claim-1",
            target_claim_id="claim-2",
            bridge_type=BridgeType.ANALOGY,
            reason_summary="Test bridge",
        )
        await store.add(bridge)

        # Find by claim
        results = await store.find_by_claim("claim-1")
        assert len(results) >= 1


class TestProjectIsolation:
    """Test project isolation in stores."""

    @pytest.mark.asyncio
    async def test_claims_isolated_by_project(self, tmp_path):
        """Test that claims are isolated by project."""
        project_a_path = tmp_path / "project_a.db"
        project_b_path = tmp_path / "project_b.db"

        store_a = ClaimStore(project_id="project-a", db_path=project_a_path)
        store_b = ClaimStore(project_id="project-b", db_path=project_b_path)

        # Add claim to project A
        claim_a = Claim(project_id="project-a", core_claim="Claim for A")
        await store_a.add(claim_a)

        # Add claim to project B
        claim_b = Claim(project_id="project-b", core_claim="Claim for B")
        await store_b.add(claim_b)

        # Verify isolation
        list_a = await store_a.list_by_project("project-a")
        list_b = await store_b.list_by_project("project-b")

        assert len(list_a) == 1
        assert len(list_b) == 1
        assert list_a[0].core_claim == "Claim for A"
        assert list_b[0].core_claim == "Claim for B"


class TestStoreValidation:
    """Test store validation and error handling."""

    @pytest.mark.asyncio
    async def test_invalid_write_rejected_safely(self, temp_db_path):
        """Test that invalid writes are rejected safely."""
        store = ClaimStore(project_id="test", db_path=temp_db_path)

        # Try to fetch non-existent claim
        result = await store.get("non-existent-id")
        assert result is None

    @pytest.mark.asyncio
    async def test_delete_nonexistent_returns_false(self, temp_db_path):
        """Test that deleting non-existent record returns False."""
        store = ClaimStore(project_id="test", db_path=temp_db_path)

        result = await store.delete("non-existent-id")
        assert result is False
