"""Unit tests for Titanax collective operations."""

from __future__ import annotations

import numpy as np
import pytest
import jax
import jax.numpy as jnp

from src.titanax.compat import Mesh as CompatMesh
from src.titanax.exec.collectives import collectives, mesh_context
from src.titanax.exceptions import CollectiveError


def _build_mesh(axis_names: tuple[str, ...], layout: str):
    """Create a JAX mesh for the requested axis layout."""

    devices = np.array(jax.devices(), dtype=object)
    if devices.size == 0:
        pytest.skip("Collective tests require at least one JAX device")

    if CompatMesh is None:  # pragma: no cover - compatibility guard
        pytest.skip("Mesh API not available in this JAX version")

    if layout == "1d":
        shape = (devices.size,)
    elif layout == "data_model":
        shape = (devices.size, 1)
    elif layout == "model_data":
        shape = (1, devices.size)
    else:  # pragma: no cover - defensive guard for future layouts
        raise ValueError(f"Unknown layout '{layout}'")

    mesh_devices = devices.reshape(shape)
    return CompatMesh(mesh_devices, axis_names)


@pytest.fixture(scope="module")
def mesh_1d():
    """1D mesh exposing only the data axis."""

    return _build_mesh(("data",), "1d")


@pytest.fixture(scope="module")
def mesh_2d_data_major():
    """2D mesh where the data axis spans all devices."""

    return _build_mesh(("data", "model"), "data_model")


@pytest.fixture(scope="module")
def mesh_2d_model_major():
    """2D mesh where the model axis spans all devices."""

    return _build_mesh(("data", "model"), "model_data")


def test_collectives_require_mesh_context():
    """Collectives should raise if no mesh context is installed."""

    with pytest.raises(CollectiveError, match="no active mesh context"):
        collectives.psum(jnp.array([1.0], dtype=jnp.float32), axis="data")


def test_axis_validation_errors(mesh_1d):
    """Axis names must exist in the active mesh."""

    with mesh_context(mesh_1d):
        with pytest.raises(CollectiveError, match="axis not found in mesh"):
            collectives.psum(jnp.array([1.0], dtype=jnp.float32), axis="model")


def _assert_allclose(actual, expected):
    np.testing.assert_allclose(np.asarray(actual), np.asarray(expected))


def test_psum_matches_lax_single_axis(mesh_1d):
    """psum should match lax.psum on a 1D mesh."""

    axis = "data"
    axis_size = mesh_1d.shape[axis]
    inputs = jnp.arange(axis_size * 2, dtype=jnp.float32).reshape(axis_size, 2)

    with mesh_context(mesh_1d):
        result = jax.pmap(lambda x: collectives.psum(x, axis), axis_name=axis)(inputs)
    expected = jax.pmap(lambda x: jax.lax.psum(x, axis), axis_name=axis)(inputs)

    _assert_allclose(result, expected)


def test_pmean_matches_lax_single_axis(mesh_1d):
    """pmean should match lax.pmean on a 1D mesh."""

    axis = "data"
    axis_size = mesh_1d.shape[axis]
    inputs = jnp.arange(axis_size * 2, dtype=jnp.float32).reshape(axis_size, 2)

    with mesh_context(mesh_1d):
        result = jax.pmap(lambda x: collectives.pmean(x, axis), axis_name=axis)(inputs)
    expected = jax.pmap(lambda x: jax.lax.pmean(x, axis), axis_name=axis)(inputs)

    _assert_allclose(result, expected)


def test_all_gather_matches_lax_single_axis(mesh_1d):
    """all_gather should match lax.all_gather on a 1D mesh."""

    axis = "data"
    axis_size = mesh_1d.shape[axis]
    inputs = jnp.arange(axis_size * 3, dtype=jnp.float32).reshape(axis_size, 3)

    with mesh_context(mesh_1d):
        result = jax.pmap(lambda x: collectives.all_gather(x, axis), axis_name=axis)(
            inputs
        )
    expected = jax.pmap(
        lambda x: jax.lax.all_gather(x, axis, tiled=True), axis_name=axis
    )(inputs)

    _assert_allclose(result, expected)


def test_reduce_scatter_matches_lax_single_axis(mesh_1d):
    """reduce_scatter should match lax.psum_scatter on a 1D mesh."""

    axis = "data"
    axis_size = mesh_1d.shape[axis]
    chunk = max(axis_size, 1)
    inputs = jnp.arange(axis_size * chunk * 2, dtype=jnp.float32).reshape(
        axis_size, chunk * 2
    )

    with mesh_context(mesh_1d):
        result = jax.pmap(
            lambda x: collectives.reduce_scatter(x, axis), axis_name=axis
        )(inputs)
    expected = jax.pmap(
        lambda x: jax.lax.psum_scatter(x, axis, tiled=True), axis_name=axis
    )(inputs)

    _assert_allclose(result, expected)


def test_broadcast_matches_ppermute_single_axis(mesh_1d):
    """broadcast should behave like a ppermute with a source rank."""

    axis = "data"
    axis_size = mesh_1d.shape[axis]
    inputs = jnp.arange(axis_size * 2, dtype=jnp.float32).reshape(axis_size, 2)
    perm = [(0, i) for i in range(axis_size)]

    with mesh_context(mesh_1d):
        result = jax.pmap(
            lambda x: collectives.broadcast(x, axis, src_index=0),
            axis_name=axis,
        )(inputs)
    expected = jax.pmap(lambda x: jax.lax.ppermute(x, axis, perm=perm), axis_name=axis)(
        inputs
    )

    _assert_allclose(result, expected)


def test_all_to_all_matches_lax_single_axis(mesh_1d):
    """all_to_all should match lax.all_to_all on a 1D mesh."""

    axis = "data"
    axis_size = mesh_1d.shape[axis]
    inputs = jnp.arange(axis_size * axis_size * 2, dtype=jnp.float32).reshape(
        axis_size, axis_size, 2
    )

    with mesh_context(mesh_1d):
        result = jax.pmap(
            lambda x: collectives.all_to_all(x, axis, split_axis=0, concat_axis=1),
            axis_name=axis,
        )(inputs)
    expected = jax.pmap(
        lambda x: jax.lax.all_to_all(x, axis, split_axis=0, concat_axis=1),
        axis_name=axis,
    )(inputs)

    _assert_allclose(result, expected)


@pytest.mark.parametrize(
    "axis,mesh_fixture",
    [("data", "mesh_2d_data_major"), ("model", "mesh_2d_model_major")],
)
def test_collectives_work_with_2d_mesh(axis, mesh_fixture, request):
    """Collectives operate when additional mesh axes are present."""

    mesh = request.getfixturevalue(mesh_fixture)
    axis_size = mesh.shape[axis]
    inputs = jnp.arange(axis_size * 2, dtype=jnp.float32).reshape(axis_size, 2)

    with mesh_context(mesh):
        result = jax.pmap(lambda x: collectives.psum(x, axis), axis_name=axis)(inputs)
    expected = jax.pmap(lambda x: jax.lax.psum(x, axis), axis_name=axis)(inputs)

    _assert_allclose(result, expected)


def test_axis_index_matches_lax(mesh_2d_data_major):
    """axis_index should pass through to JAX."""

    axis = "data"
    axis_size = mesh_2d_data_major.shape[axis]
    inputs = jnp.arange(axis_size, dtype=jnp.float32)

    with mesh_context(mesh_2d_data_major):
        result = jax.pmap(lambda _: collectives.axis_index(axis), axis_name=axis)(
            inputs
        )
    expected = jax.pmap(lambda _: jax.lax.axis_index(axis), axis_name=axis)(inputs)

    _assert_allclose(result, expected)
