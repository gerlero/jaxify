import jax
from jaxify import jaxify


def test_and_jit() -> None:
    @jax.jit
    @jaxify
    def logical_and(a: bool, b: bool, /) -> bool:  # noqa: FBT001
        return a and True and b and True

    assert not logical_and(True, False)  # noqa: FBT003
    assert logical_and(True, True)  # noqa: FBT003


def test_or_jit() -> None:
    @jax.jit
    @jaxify
    def logical_or(a: bool, b: bool, /) -> bool:  # noqa: FBT001
        return a or False or b

    assert not logical_or(False, False)  # noqa: FBT003
    assert logical_or(True, False)  # noqa: FBT003


def test_and_or_jit() -> None:
    @jax.jit
    @jaxify
    def logical_and_or(a: bool, b: bool, c: bool, /) -> bool:  # noqa: FBT001
        return (a and b) or c

    assert not logical_and_or(True, False, False)  # noqa: FBT003
    assert logical_and_or(True, True, False)  # noqa: FBT003
    assert logical_and_or(False, False, True)  # noqa: FBT003


def test_not_jit() -> None:
    @jax.jit
    @jaxify
    def logical_not(a: bool, /) -> bool:  # noqa: FBT001
        return not a

    assert logical_not(False)  # noqa: FBT003
    assert not logical_not(True)  # noqa: FBT003


def test_static() -> None:
    @jax.jit
    @jaxify
    def always_falsey(a: bool, /) -> object:  # noqa: FBT001
        return () and not a  # noqa: SIM223

    assert always_falsey(True) == ()  # noqa: FBT003
    assert always_falsey(False) == ()  # noqa: FBT003

    @jax.jit
    @jaxify
    def always_truthy(a: bool, /) -> object:  # noqa: FBT001
        return (1,) or a  # noqa: SIM222

    assert always_truthy(True) == (1,)  # noqa: FBT003
    assert always_truthy(False) == (1,)  # noqa: FBT003
