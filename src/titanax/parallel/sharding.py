"""Sharding utilities for tensor parallel rule application.

This module provides helpers for turning rule templates into concrete
``PartitionSpec`` trees and applying ``NamedSharding`` to training state
structures. P1.1 introduces rule-to-spec utilities, while later phases
add placement helpers that consume the resulting trees.
"""

from __future__ import annotations

from fnmatch import fnmatchcase
from typing import Dict, Mapping, Sequence, Tuple

from jax import tree_util as jtu

from titanax.exceptions import sharding_error
from titanax.types import AxisName, Mesh, PartitionSpec, PyTree

# -----------------------------------------------------------------------------
# Shared type aliases
# -----------------------------------------------------------------------------

RulePattern = str
PartitionRule = Tuple[AxisName | None, ...]
RuleMap = Mapping[RulePattern, PartitionRule]
MutableRuleMap = Dict[RulePattern, PartitionRule]
SpecTree = PyTree


try:  # pragma: no cover - typing convenience for older JAX releases
    from jax.tree_util import KeyPathEntry  # type: ignore
except ImportError:  # pragma: no cover - fallback for stubbed environments
    KeyPathEntry = object  # type: ignore


def _stringify_entry(entry: KeyPathEntry) -> str:
    """Render a ``KeyPathEntry`` segment as a ``str`` for joined paths."""

    if hasattr(entry, "name"):
        return str(getattr(entry, "name"))
    if hasattr(entry, "key"):
        return str(getattr(entry, "key"))
    if hasattr(entry, "idx"):
        return str(getattr(entry, "idx"))
    if hasattr(entry, "index"):
        return str(getattr(entry, "index"))
    return str(entry)


def _stringify_path(path: Sequence[KeyPathEntry]) -> str:
    return "/".join(_stringify_entry(segment) for segment in path)


def _pattern_precedence(pattern: str) -> Tuple[int, int, int]:
    segments = tuple(seg for seg in pattern.split("/") if seg)
    wildcard_count = pattern.count("*") + pattern.count("?")
    literal_length = len(pattern.replace("*", "").replace("?", ""))
    return (len(segments), -wildcard_count, literal_length)


def tree_paths(tree: PyTree) -> Tuple[str, ...]:
    """Return flattened ``"/"``-joined paths for every leaf in ``tree``.

    The paths follow JAX's pytree traversal order (depth-first with dict keys
    sorted for determinism), providing stable identifiers for rule matching.
    """

    path_leaves, _ = jtu.tree_flatten_with_path(tree)
    return tuple(_stringify_path(path) for path, _ in path_leaves)


def spec_for(
    path: str,
    rules: RuleMap,
    *,
    default: PartitionSpec | None = None,
) -> PartitionSpec:
    """Resolve a ``PartitionSpec`` for ``path`` using glob-style rules.

    Patterns are evaluated with ``fnmatchcase`` and ranked using
    (1) segment depth, (2) wildcard count (fewer wins), and (3) literal
    length. Conflicts with identical precedence raise ``ShardingError`` to
    force rule authors to disambiguate.
    """

    matches: list[tuple[Tuple[int, int, int], str, PartitionRule]] = []
    for pattern, rule in rules.items():
        if fnmatchcase(path, pattern):
            matches.append((_pattern_precedence(pattern), pattern, rule))

    if not matches:
        if default is not None:
            return default
        return PartitionSpec()

    matches.sort(key=lambda item: item[0], reverse=True)
    best_precedence = matches[0][0]
    best_matches = [entry for entry in matches if entry[0] == best_precedence]

    if len(best_matches) > 1:
        conflict_patterns = ", ".join(pattern for _, pattern, _ in best_matches)
        raise sharding_error(
            path,
            "multiple patterns matched with equal precedence",
            suggestion=(
                "Disambiguate the following conflicting sharding rules: "
                f"{conflict_patterns}"
            ),
        )

    _, pattern, rule = matches[0]
    try:
        return PartitionSpec(*rule)
    except TypeError as exc:
        raise sharding_error(
            path,
            f"invalid partition rule {rule!r} produced by pattern '{pattern}': {exc}",
            suggestion="Ensure each rule is a tuple of axis names or None values",
        ) from exc


def build_param_specs(
    params_tree: PyTree,
    rules: RuleMap,
    *,
    default: PartitionSpec | None = None,
) -> SpecTree:
    """Construct a ``PartitionSpec`` PyTree mirroring ``params_tree``.

    Each leaf path is resolved via ``spec_for``. The resulting tree mirrors the
    original pytree structure, enabling downstream helpers to apply sharding.
    """

    path_leaves, treedef = jtu.tree_flatten_with_path(params_tree)
    spec_leaves = []
    for path, _ in path_leaves:
        path_str = _stringify_path(path)
        spec_leaves.append(spec_for(path_str, rules, default=default))
    return jtu.tree_unflatten(treedef, spec_leaves)


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
