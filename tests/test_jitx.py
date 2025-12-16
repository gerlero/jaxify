import jax
import jax.numpy as jnp
from jaxify import jitx


def test_basic() -> None:
    @jitx
    def absolute_value(x: jax.Array) -> jax.Array:
        if x >= 0:
            return x
        return -x

    xs = jnp.arange(-1000, 1000)
    ys = absolute_value(xs)
    assert jnp.all(ys == jnp.abs(xs))


def test_nested() -> None:
    @jitx
    def nested_absolute_value(x: jax.Array) -> jax.Array:
        if x >= 0:
            if x >= 10:
                return x
            return x + 10
        if x <= -10:
            return -x
        return -x + 10

    xs = jnp.arange(-20, 20)
    ys = nested_absolute_value(xs)
    expected_ys = jnp.where(
        xs >= 10,
        xs,
        jnp.where(xs >= 0, xs + 10, jnp.where(xs <= -10, -xs, -xs + 10)),
    )
    assert jnp.all(ys == expected_ys)
