"""Unit tests for sharding utilities."""

import pytest

pytest.importorskip("jax")

from titanax.exceptions import ShardingError
from titanax.types import PartitionSpec

import src.titanax.parallel.sharding as sharding


@pytest.fixture
def sample_params_tree() -> dict[str, object]:
    return {
        "dense": {"kernel": object(), "bias": object()},
        "blocks": [
            {"ln": {"scale": object(), "bias": object()}},
            {"ln": {"scale": object(), "bias": object()}},
        ],
        "head": (object(), object()),
    }


@pytest.fixture
def sample_rules() -> sharding.RuleMap:
    return {
        "dense/kernel": ("tp", None),
        "dense/bias": ("tp",),
        "dense/*": (None,),
        "blocks/*/ln/scale": ("tp",),
        "blocks/*/ln/*": (None,),
    }


class TestTreePaths:
    def test_tree_paths_returns_expected_order(
        self, sample_params_tree: dict[str, object]
    ) -> None:
        assert sharding.tree_paths(sample_params_tree) == (
            "blocks/0/ln/bias",
            "blocks/0/ln/scale",
            "blocks/1/ln/bias",
            "blocks/1/ln/scale",
            "dense/bias",
            "dense/kernel",
            "head/0",
            "head/1",
        )


class TestSpecFor:
    def test_spec_for_prefers_exact_match_over_glob(
        self, sample_rules: sharding.RuleMap
    ) -> None:
        spec = sharding.spec_for("dense/bias", sample_rules)
        assert spec == PartitionSpec("tp")

    def test_spec_for_longest_path_wins(self, sample_rules: sharding.RuleMap) -> None:
        spec = sharding.spec_for("blocks/0/ln/scale", sample_rules)
        assert spec == PartitionSpec("tp")

    def test_spec_for_default_fallback(self, sample_rules: sharding.RuleMap) -> None:
        spec = sharding.spec_for("head/0", sample_rules)
        assert spec == PartitionSpec()

    def test_spec_for_custom_default(self, sample_rules: sharding.RuleMap) -> None:
        default_spec = PartitionSpec("dp")
        spec = sharding.spec_for("head/0", sample_rules, default=default_spec)
        assert spec == default_spec

    def test_spec_for_conflict_raises(self) -> None:
        rules: sharding.RuleMap = {
            "module/*/weight": ("tp",),
            "module/?/weight": ("tp",),
        }

        with pytest.raises(ShardingError) as exc:
            sharding.spec_for("module/x/weight", rules)

        assert "multiple patterns" in str(exc.value)


class TestBuildParamSpecs:
    def test_build_param_specs_preserves_structure(
        self, sample_params_tree: dict[str, object], sample_rules: sharding.RuleMap
    ) -> None:
        spec_tree = sharding.build_param_specs(sample_params_tree, sample_rules)

        expected = {
            "dense": {
                "kernel": PartitionSpec("tp", None),
                "bias": PartitionSpec("tp"),
            },
            "blocks": [
                {"ln": {"scale": PartitionSpec("tp"), "bias": PartitionSpec(None)}},
                {"ln": {"scale": PartitionSpec("tp"), "bias": PartitionSpec(None)}},
            ],
            "head": (PartitionSpec(), PartitionSpec()),
        }

        assert spec_tree == expected

    def test_build_param_specs_respects_default(
        self, sample_params_tree: dict[str, object]
    ) -> None:
        default_spec = PartitionSpec("dp")
        rules: sharding.RuleMap = {"dense/kernel": ("tp", None)}

        spec_tree = sharding.build_param_specs(
            sample_params_tree, rules, default=default_spec
        )

        assert spec_tree["dense"]["kernel"] == PartitionSpec("tp", None)
        assert spec_tree["dense"]["bias"] == default_spec
        assert spec_tree["head"][0] == default_spec


@pytest.mark.skip(reason="P1.2 pending")
class TestNamedSharding:
    def test_apply_named_sharding_placeholder(self) -> None:
        raise NotImplementedError


@pytest.mark.skip(reason="P1.2 pending")
class TestBatchSpecs:
    def test_shard_batch_specs_placeholder(self) -> None:
        raise NotImplementedError
