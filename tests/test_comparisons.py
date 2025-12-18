import jax
from jaxify import jaxify


def test_comparisons_jit() -> None:
    @jax.jit
    @jaxify
    def check_comparison(x: int, y: int, /) -> bool:
        return (x > 0 and y < 0) or (x < 0 and y > 0)

    assert check_comparison(1, -1)
    assert check_comparison(-1, 1)
    assert not check_comparison(1, 1)
    assert not check_comparison(-1, -1)


def test_chained_comparison_jit() -> None:
    @jax.jit
    @jaxify
    def check_chained_comparisons(x: int, y: int, z: int, /) -> bool:
        return 0 < x < 10 and -10 < y < 0 and z == 5

    assert check_chained_comparisons(5, -5, 5)
    assert not check_chained_comparisons(0, -5, 5)
    assert not check_chained_comparisons(5, 0, 5)
    assert not check_chained_comparisons(5, -5, 0)
