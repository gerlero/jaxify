import sys
from collections.abc import Callable
from typing import TypeVar

if sys.version_info >= (3, 11):
    from typing import TypeVarTuple, Unpack
else:
    from typing_extensions import TypeVarTuple, Unpack

import jax
import jax.core
import jax.numpy as jnp

_Args = TypeVarTuple("_Args")
_R = TypeVar("_R")


def if_hook(
    cond: object,
    if_true: Callable[[*_Args], _R],
    if_false: Callable[[*_Args], _R],
    *args: Unpack[_Args],
) -> _R:
    match cond:
        case jax.core.Tracer(size=1):
            return jax.lax.cond(cond, if_true, if_false, *args)
        case _:
            return if_true(*args) if cond else if_false(*args)


def and_hook(*args: Callable[[], object]) -> object:
    ret: object = True
    for arg in args:
        match ret, (value := arg()):
            case (jax.core.Tracer(size=1), _) | (
                jax.core.Tracer(size=1),
                jax.core.Tracer(size=1),
            ):
                ret = jax.lax.cond(ret, lambda _=value: _, lambda _=ret: _)
            case _, jax.core.Tracer(size=1):
                ret = value
            case _:
                if not value:
                    return value
                ret = value
    return ret


def or_hook(*args: Callable[[], object]) -> object:
    ret: object = False
    for arg in args:
        match ret, (value := arg()):
            case (jax.core.Tracer(size=1), _) | (
                jax.core.Tracer(size=1),
                jax.core.Tracer(size=1),
            ):
                ret = jax.lax.cond(ret, lambda _=ret: _, lambda _=value: _)
            case _, jax.core.Tracer(size=1):
                ret = value
            case _:
                if value:
                    return value
                ret = value
    return ret


def not_hook(value: object) -> object:
    match value:
        case jax.core.Tracer(size=1):
            return jnp.logical_not(value)
        case _:
            return not value


@jax.tree_util.register_pytree_node_class
class _NoReturnValue:
    def tree_flatten(self) -> tuple[tuple[()], None]:
        return ((), None)

    @classmethod
    def tree_unflatten(cls, aux_data: None, children: tuple[()]) -> "_NoReturnValue":  # noqa: ARG003
        return _NoReturnValue()


@jax.tree_util.register_pytree_node_class
class Return:
    def __init__(
        self, value: object = _NoReturnValue(), *, done: object | None = None
    ) -> None:
        if done is None:
            done = not isinstance(value, _NoReturnValue)
        self._done = done
        self._value = value

    def return_(self, value: object = None) -> "Return":
        if isinstance(self._value, _NoReturnValue):
            return Return(done=True, value=value)

        return if_hook(
            self._done,
            lambda: self,
            lambda: Return(done=True, value=value),
        )

    def set_type(self, value: object) -> "Return":
        if isinstance(self._value, _NoReturnValue):
            return Return(done=False, value=value)

        return if_hook(
            self._done,
            lambda: self,
            lambda: Return(done=False, value=value),
        )

    @property
    def value(self) -> object:
        if isinstance(self._value, _NoReturnValue):
            return None
        return self._value

    def tree_flatten(self) -> tuple[tuple[object, object], None]:
        return ((self._done, self._value), None)

    @classmethod
    def tree_unflatten(
        cls,
        aux_data: None,  # noqa: ARG003
        children: tuple[object, object],
    ) -> "Return":
        done, value = children
        return cls(value=value, done=done)
