import ast
import functools
import inspect
import itertools
import textwrap
import warnings
from collections.abc import Callable
from typing import ParamSpec, TypeVar

import jax
import jax.core

_Inputs = ParamSpec("_Inputs")
_Output = TypeVar("_Output")


def jaxify(func: Callable[_Inputs, _Output], /) -> Callable[_Inputs, _Output]:  # noqa: C901, PLR0915
    if not inspect.isfunction(func):
        msg = "jaxify can only be applied to functions"
        raise TypeError(msg)
    if inspect.isgeneratorfunction(func):
        msg = "jaxify does not support generator functions"
        raise TypeError(msg)
    if inspect.iscoroutinefunction(func):
        msg = "jaxify does not support coroutine functions"
        raise TypeError(msg)

    try:
        source = inspect.getsource(func)
    except OSError as e:
        msg = "Could not retrieve source code for function"
        raise RuntimeError(msg) from e

    tree = ast.parse(textwrap.dedent(source))

    nconds = 0
    for node in ast.walk(tree):
        match node:
            case ast.If() | ast.IfExp():
                node.test = ast.Call(
                    func=ast.Name(id="__jaxify_cond_hook__", ctx=ast.Load()),
                    args=[node.test, ast.Constant(value=nconds)],
                    keywords=[],
                )
                nconds += 1

            case ast.For() | ast.While():
                msg = "jaxify does not currently support loops"
                raise NotImplementedError(msg)

            case ast.FunctionDef(name=func.__name__):  # ty: ignore[unresolved-attribute]
                for i, decorator in enumerate(node.decorator_list):
                    match decorator:
                        case (
                            ast.Name(id="jaxify")
                            | ast.Attribute(value=ast.Name(id="jaxify"), attr="jaxify")
                        ):
                            del node.decorator_list[: i + 1]
                            break

            case ast.AsyncFunctionDef() | ast.AsyncFor() | ast.AsyncWith():
                msg = "jaxify does not support async syntax"
                raise NotImplementedError(msg)

    ast.fix_missing_locations(tree)

    local_vars = {}
    exec(  # noqa: S102
        compile(tree, filename="<ast>", mode="exec"),
        func.__globals__,  # ty: ignore[unresolved-attribute]
        local_vars,
    )
    traceable_func: Callable[_Inputs, _Output] = local_vars[func.__name__]

    @functools.wraps(func)
    def jaxify_wrapper(*args: _Inputs.args, **kwargs: _Inputs.kwargs) -> _Output:  # noqa: C901
        if not nconds:
            return func(*args, **kwargs)

        cond_combinations: list[tuple[bool, ...]] = list(
            itertools.product([False, True], repeat=nconds)
        )
        cond_values: list[list[object | None]] = []
        outputs: list[_Output] = []
        combination: tuple[bool, ...] | None = None

        def cond_hook(cond: object, cond_id: int) -> bool:
            if isinstance(cond, jax.core.Tracer):
                values[cond_id] = cond
                assert combination is not None
                return combination[cond_id]
            return bool(cond)

        traceable_func_local = type(traceable_func)(
            traceable_func.__code__,
            {**func.__globals__, "__jaxify_cond_hook__": cond_hook},
            traceable_func.__name__,
            traceable_func.__defaults__,
            traceable_func.__closure__,
        )

        for combination in cond_combinations:  # noqa: B007
            values = [None] * nconds
            result = traceable_func_local(*args, **kwargs)
            cond_values.append(values)
            outputs.append(result)

        ret = outputs[0]
        for i in range(1, len(outputs)):
            if outputs[i] is not None:
                mask = True
                for cond, value in zip(
                    cond_values[i], cond_combinations[i], strict=True
                ):
                    if cond is not None:
                        match value:
                            case True:
                                mask &= cond
                            case False:
                                mask &= ~cond
                if ret is None:
                    ret = outputs[i]
                else:
                    try:
                        ret = jax.lax.cond(
                            mask,
                            lambda _=outputs[i]: _,
                            lambda _=ret: _,
                        )
                    except TypeError:
                        ret = jax.lax.select(
                            mask,
                            outputs[i],  # type: ignore[invalid-argument-type]
                            ret,  # type: ignore[invalid-argument-type]
                        )

        if ret is None:
            warnings.warn(
                "jaxify: all branches returned None", RuntimeWarning, stacklevel=2
            )

        return ret  # ty: ignore[invalid-return-type]

    return jaxify_wrapper
