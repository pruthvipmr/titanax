"""Unit tests for sharding utilities."""

import pytest

pytest.importorskip("jax")

import jax
import jax.numpy as jnp

from titanax.exceptions import ShardingError
from titanax.runtime.mesh import MeshSpec
from titanax.types import NamedSharding, PartitionSpec

import titanax.parallel.sharding as sharding


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


class TestNamedSharding:
    def test_apply_named_sharding_places_arrays(self) -> None:
        mesh_spec = MeshSpec(devices="all", axes=("model",))
        mesh = mesh_spec.build()

        params = {
            "dense": {
                "kernel": jnp.arange(8, dtype=jnp.float32).reshape(4, 2),
                "bias": jnp.ones((2,), dtype=jnp.float32),
            }
        }

        spec_tree = {
            "dense": {
                "kernel": PartitionSpec("model", None),
                "bias": PartitionSpec(),
            }
        }

        with mesh:
            placed = sharding.apply_named_sharding(params, mesh, spec_tree)

        kernel = placed["dense"]["kernel"]
        bias = placed["dense"]["bias"]

        assert isinstance(kernel, jax.Array)
        assert isinstance(kernel.sharding, NamedSharding)
        assert kernel.sharding.spec == PartitionSpec("model", None)
        assert kernel.shape == (4, 2)

        assert isinstance(bias, jax.Array)
        assert isinstance(bias.sharding, NamedSharding)
        assert bias.sharding.spec == PartitionSpec()

    def test_apply_named_sharding_structure_mismatch(self) -> None:
        mesh_spec = MeshSpec(devices="all", axes=("model",))
        mesh = mesh_spec.build()

        params = {"a": jnp.ones((2, 2)), "b": jnp.ones((2,))}
        spec_tree = {"a": PartitionSpec("model", None)}

        with mesh:
            with pytest.raises(ShardingError) as exc:
                sharding.apply_named_sharding(params, mesh, spec_tree)

        assert "structure mismatch" in str(exc.value)

    def test_apply_named_sharding_rejects_non_array_sharded_leaf(self) -> None:
        mesh_spec = MeshSpec(devices="all", axes=("model",))
        mesh = mesh_spec.build()

        params = {"count": 3}
        spec_tree = {"count": PartitionSpec("model")}

        with mesh:
            with pytest.raises(ShardingError) as exc:
                sharding.apply_named_sharding(params, mesh, spec_tree)

        assert "non-array leaf" in str(exc.value)


class TestBatchSpecs:
    def test_shard_batch_specs_applies_dp_axis_to_leading_dim(self) -> None:
        batch_example = {
            "images": jnp.zeros((8, 4, 4, 3)),
            "labels": jnp.zeros((8,), dtype=jnp.int32),
            "meta": {"is_train": True},
        }

        specs = sharding.shard_batch_specs(batch_example, dp_axis="data")

        assert specs["images"] == PartitionSpec("data", None, None, None)
        assert specs["labels"] == PartitionSpec("data")
        assert specs["meta"]["is_train"] == PartitionSpec()

    def test_shard_batch_specs_replicates_scalars(self) -> None:
        scalar_batch = {"loss_scale": jnp.array(1.0), "step": 1}

        specs = sharding.shard_batch_specs(scalar_batch, dp_axis="data")

        assert specs["loss_scale"] == PartitionSpec()
        assert specs["step"] == PartitionSpec()
