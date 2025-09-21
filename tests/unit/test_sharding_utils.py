"""Scaffolding for upcoming sharding utility tests."""

import pytest

pytest.importorskip("jax")

import src.titanax.parallel.sharding as sharding

pytestmark = pytest.mark.skip(reason="P1.1-P1.2 implementations pending")


class TestTreePaths:
    """Placeholder tests for tree path enumeration."""

    def test_tree_paths_placeholder(self) -> None:
        """Document expected future assertions for tree path discovery."""
        _ = sharding  # Keep import referenced until real tests land


class TestSpecResolution:
    """Placeholder tests for rule-to-spec resolution."""

    def test_spec_for_placeholder(self) -> None:
        """Document expected future assertions for `spec_for`."""
        pass


class TestNamedSharding:
    """Placeholder tests for sharding application helpers."""

    def test_apply_named_sharding_placeholder(self) -> None:
        """Document expected future assertions for NamedSharding application."""
        pass


class TestBatchSpecs:
    """Placeholder tests for batch spec derivation."""

    def test_shard_batch_specs_placeholder(self) -> None:
        """Document expected future assertions for DP batch sharding defaults."""
        pass
