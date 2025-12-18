import jax
import jax.numpy as jnp
from jaxify import jaxify


def test_jaxify_only() -> None:
    @jaxify
    def absolute_value(x: float | jax.Array) -> float | jax.Array:
        if x >= 0:
            return x
        return -x

    assert absolute_value(10) == 10
    assert absolute_value(-10) == 10


def test_jaxify_jit() -> None:
    @jax.jit
    @jaxify
    def absolute_value(x: float | jax.Array) -> float | jax.Array:
        if x >= 0:
            return x
        return -x

    assert absolute_value(10) == 10
    assert absolute_value(-10) == 10


def test_jaxify_vmap() -> None:
    @jax.vmap
    @jaxify
    def absolute_value(x: float | jax.Array) -> float | jax.Array:
        if x >= 0:
            return x
        return -x

    xs = jnp.arange(-1000, 1000)
    ys = absolute_value(xs)
    assert jnp.all(ys == jnp.abs(xs))


def test_jaxify_vmap_jit() -> None:
    @jax.jit
    @jax.vmap
    @jaxify
    def absolute_value(x: float | jax.Array) -> float | jax.Array:
        if x >= 0:
            return x
        return -x

    xs = jnp.arange(-1000, 1000)
    ys = absolute_value(xs)
    assert jnp.all(ys == jnp.abs(xs))


def test_nested_ifs() -> None:
    @jax.jit
    @jax.vmap
    @jaxify
    def nested_ifs(x: float | jax.Array) -> float | jax.Array:
        if x >= 0:
            if x >= 10:
                return x
            return x + 10
        if x <= -10:
            return -x
        return -x + 10

    xs = jnp.arange(-20, 20)
    ys = nested_ifs(xs)
    expected_ys = jnp.where(
        xs >= 10,
        xs,
        jnp.where(xs >= 0, xs + 10, jnp.where(xs <= -10, -xs, -xs + 10)),
    )
    assert jnp.all(ys == expected_ys)


def test_if_expr() -> None:
    @jax.jit
    @jax.vmap
    @jaxify
    def absolute_value(x: float | jax.Array) -> float | jax.Array:
        return x if x >= 0 else -x

    xs = jnp.arange(-1000, 1000)
    ys = absolute_value(xs)
    assert jnp.all(ys == jnp.abs(xs))
