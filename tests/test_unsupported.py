import jax
import pytest
from jaxify import jaxify


def test_for_loop() -> None:
    with pytest.raises(NotImplementedError):

        @jaxify
        def sum_up_to_n(n: int) -> int:
            total = 0
            for i in range(n):
                total += i
            return total


def test_while_loop() -> None:
    with pytest.raises(NotImplementedError):

        @jaxify
        def countdown(n: int) -> int:
            while n > 0:
                n -= 1
            return n


def test_and_jit() -> None:
    @jax.jit
    @jaxify
    def logical_and(a: bool, b: bool, /) -> bool:  # noqa: FBT001
        return a and b

    with pytest.raises(Exception, match="Attempted boolean conversion"):
        assert logical_and(True, False) is False  # noqa: FBT003
    with pytest.raises(Exception, match="Attempted boolean conversion"):
        assert logical_and(True, True) is True  # noqa: FBT003


def test_or_jit() -> None:
    @jax.jit
    @jaxify
    def logical_or(a: bool, b: bool, /) -> bool:  # noqa: FBT001
        return a or b

    with pytest.raises(Exception, match="Attempted boolean conversion"):
        assert logical_or(False, False) is False  # noqa: FBT003
    with pytest.raises(Exception, match="Attempted boolean conversion"):
        assert logical_or(True, False) is True  # noqa: FBT003


def test_and_or_jit() -> None:
    @jax.jit
    @jaxify
    def logical_and_or(a: bool, b: bool, c: bool, /) -> bool:  # noqa: FBT001
        return (a and b) or c

    with pytest.raises(Exception, match="Attempted boolean conversion"):
        assert logical_and_or(True, False, False) is False  # noqa: FBT003
    with pytest.raises(Exception, match="Attempted boolean conversion"):
        assert logical_and_or(True, True, False) is True  # noqa: FBT003
    with pytest.raises(Exception, match="Attempted boolean conversion"):
        assert logical_and_or(False, False, True) is True  # noqa: FBT003


def test_inequality_jit() -> None:
    @jax.jit
    @jaxify
    def check_inequality(x: int, y: int, /) -> bool:
        return (x > 0 and y < 0) or (x < 0 and y > 0)

    with pytest.raises(Exception, match="Attempted boolean conversion"):
        assert check_inequality(1, -1) is True
    with pytest.raises(Exception, match="Attempted boolean conversion"):
        assert check_inequality(-1, 1) is True
    with pytest.raises(Exception, match="Attempted boolean conversion"):
        assert check_inequality(1, 1) is False
    with pytest.raises(Exception, match="Attempted boolean conversion"):
        assert check_inequality(-1, -1) is False


def test_chained_inequalities_jit() -> None:
    @jax.jit
    @jaxify
    def check_chained_inequalities(x: int, y: int, z: int, /) -> bool:
        return 0 < x < 10 and -10 < y < 0 and z == 5

    with pytest.raises(Exception, match="Attempted boolean conversion"):
        assert check_chained_inequalities(5, -5, 5) is True
    with pytest.raises(Exception, match="Attempted boolean conversion"):
        assert check_chained_inequalities(0, -5, 5) is False
    with pytest.raises(Exception, match="Attempted boolean conversion"):
        assert check_chained_inequalities(5, 0, 5) is False
    with pytest.raises(Exception, match="Attempted boolean conversion"):
        assert check_chained_inequalities(5, -5, 0) is False
