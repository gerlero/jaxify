from collections.abc import Generator

import jax
from jaxify import jaxify


def test_nested_funcs() -> None:
    @jax.jit
    @jaxify
    def with_nested_funcs(_: int) -> None:
        def helper(y: int) -> int:
            lst = [y + 1, y + 2, y + 3]
            total = 0
            for n in lst:
                total += n
            return total

        def helper_generator(z: int) -> Generator[int]:
            yield from range(z)

        async def async_helper(a: int) -> int:
            return a + 1

        lambda_helper = lambda v: v * 2  # noqa: E731, F841

    assert with_nested_funcs(5) is None
