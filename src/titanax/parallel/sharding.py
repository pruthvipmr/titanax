"""Sharding utilities for tensor parallel rule application.

This module provides helpers for turning rule templates into concrete
``PartitionSpec`` trees and applying ``NamedSharding`` to training state
structures. Implementations are introduced in later P1 tasks; P1.0 only
establishes the API surface and documentation so downstream work can
focus on behavior without debating shape or naming.
"""

from __future__ import annotations

from typing import Dict, Mapping, Tuple

from titanax.types import AxisName, Mesh, PartitionSpec, PyTree

# -----------------------------------------------------------------------------
# Shared type aliases
# -----------------------------------------------------------------------------

RulePattern = str
PartitionRule = Tuple[AxisName | None, ...]
RuleMap = Mapping[RulePattern, PartitionRule]
MutableRuleMap = Dict[RulePattern, PartitionRule]
SpecTree = PyTree


def tree_paths(tree: PyTree) -> Tuple[str, ...]:
    """Return flattened ``"/"``-joined paths for every leaf in ``tree``.

    The concrete traversal logic will land in P1.1; this stub only defines
    the return contract and provides a discoverable location for later work.
    """

    raise NotImplementedError("P1.1 implements tree path enumeration")


def spec_for(
    path: str,
    rules: RuleMap,
    *,
    default: PartitionSpec | None = None,
) -> PartitionSpec:
    """Resolve a ``PartitionSpec`` for ``path`` given rule patterns.

    Planned behavior (P1.1) supports glob matching with longest-prefix
    precedence. This stub only anchors the signature.
    """

    raise NotImplementedError("P1.1 implements rule matching")


def build_param_specs(
    params_tree: PyTree,
    rules: RuleMap,
    *,
    default: PartitionSpec | None = None,
) -> SpecTree:
    """Construct a ``PartitionSpec`` PyTree matching ``params_tree``.

    Expected to leverage ``tree_paths`` + ``spec_for`` once implemented. The
    placeholder keeps the file importable for now.
    """

    raise NotImplementedError("P1.1 implements spec tree construction")


def apply_named_sharding(
    tree: PyTree,
    mesh: Mesh,
    spec_tree: SpecTree,
) -> PyTree:
    """Apply ``NamedSharding`` to each leaf in ``tree`` using ``spec_tree``.

    P1.2 populates this function with the actual ``device_put`` calls. The
    stub clarifies the dependency on ``Mesh`` and keeps API documentation
    centralized.
    """

    raise NotImplementedError("P1.2 applies NamedSharding to trees")


def shard_batch_specs(batch_example: PyTree, dp_axis: AxisName) -> SpecTree:
    """Derive default batch ``PartitionSpec`` values for DP sharding.

    Future implementation (P1.2) will inspect the batch structure and shard
    leading dimensions along ``dp_axis`` when appropriate.
    """

    raise NotImplementedError("P1.2 defines batch sharding defaults")
