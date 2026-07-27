import jax

from jaxify import jaxify


def test_and_jit() -> None:
    @jax.jit
    @jaxify
    def logical_and(a: bool, b: bool, /) -> bool:
        return a and True and b and True

    assert not logical_and(True, False)
    assert logical_and(True, True)


def test_or_jit() -> None:
    @jax.jit
    @jaxify
    def logical_or(a: bool, b: bool, /) -> bool:
        return a or False or b

    assert not logical_or(False, False)
    assert logical_or(True, False)


def test_and_or_jit() -> None:
    @jax.jit
    @jaxify
    def logical_and_or(a: bool, b: bool, c: bool, /) -> bool:
        return (a and b) or c

    assert not logical_and_or(True, False, False)
    assert logical_and_or(True, True, False)
    assert logical_and_or(False, False, True)


def test_not_jit() -> None:
    @jax.jit
    @jaxify
    def logical_not(a: bool, /) -> bool:
        return not a

    assert logical_not(False)
    assert not logical_not(True)


def test_static() -> None:
    @jax.jit
    @jaxify
    def always_falsey(a: bool, /) -> object:
        return () and not a  # noqa: SIM223

    assert always_falsey(True) == ()
    assert always_falsey(False) == ()

    @jax.jit
    @jaxify
    def always_truthy(a: bool, /) -> object:
        return (1,) or a  # noqa: SIM222

    assert always_truthy(True) == (1,)
    assert always_truthy(False) == (1,)
