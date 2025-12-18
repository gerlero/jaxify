from collections.abc import AsyncGenerator, Generator

import pytest
from jaxify import JaxifyError, jaxify


def test_for_loop() -> None:
    with pytest.raises(JaxifyError):

        @jaxify
        def sum_up_to_n(n: int) -> int:
            total = 0
            for i in range(n):
                total += i
            return total


def test_while_loop() -> None:
    with pytest.raises(JaxifyError):

        @jaxify
        def countdown(n: int) -> int:
            while n > 0:
                n -= 1
            return n


def test_generator() -> None:
    with pytest.raises(TypeError):

        @jaxify
        def generator(n: int) -> Generator[int]:
            yield from range(n)


def test_async() -> None:
    with pytest.raises(TypeError):

        @jaxify
        async def coroutine(x: int) -> int:
            return x + 1

    with pytest.raises(TypeError):

        @jaxify
        async def async_generator(x: int) -> AsyncGenerator[int]:
            for i in range(x):
                yield i
