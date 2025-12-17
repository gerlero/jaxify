import ast
import functools
import inspect
import itertools
import textwrap
from collections.abc import Callable
from typing import ParamSpec, TypeVar

import jax
import jax.interpreters.batching
import jax.interpreters.partial_eval

_Inputs = ParamSpec("_Inputs")
_Output = TypeVar("_Output")


def jaxify(func: Callable[_Inputs, _Output], /) -> Callable[_Inputs, _Output]:  # noqa: C901, PLR0915
    source = textwrap.dedent(inspect.getsource(func))
    tree = ast.parse(source)

    nconds = 0
    for node in ast.walk(tree):
        match node:
            case ast.If():
                node.test = ast.Call(
                    func=ast.Name(id="_jaxify_cond", ctx=ast.Load()),
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

    ast.fix_missing_locations(tree)

    traceable = compile(tree, filename="<ast>", mode="exec")

    @functools.wraps(func)
    def jaxify_wrapper(*args: _Inputs.args, **kwargs: _Inputs.kwargs) -> _Output:  # noqa: C901, PLR0912
        if not nconds:
            return func(*args, **kwargs)

        cond_combinations: list[tuple[bool, ...]] = list(
            itertools.product([False, True], repeat=nconds)
        )
        cond_values: list[list[object | None]] = []
        outputs: list[_Output] = []
        for combination in cond_combinations:
            values = [None] * nconds

            def _jaxify_cond(cond: object, cond_id: int) -> bool:
                if isinstance(
                    cond,
                    (
                        jax.interpreters.partial_eval.DynamicJaxprTracer,
                        jax.interpreters.batching.BatchTracer,
                    ),
                ):
                    values[cond_id] = cond  # noqa: B023
                    return combination[cond_id]  # noqa: B023
                return bool(cond)

            local_vars: dict[str, object] = {}
            exec(  # noqa: S102
                traceable,
                {**func.__globals__, "jax": jax, "_jaxify_cond": _jaxify_cond},  # ty: ignore[unresolved-attribute]
                local_vars,
            )
            traceable_func = local_vars[next(iter(local_vars))]

            try:
                result = traceable_func(*args, **kwargs)  # ty: ignore[call-non-callable]
            except Exception:  # noqa: BLE001
                result = None

            cond_values.append(values)
            outputs.append(result)  # ty: ignore[invalid-argument-type]

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
            msg = "All possible paths through the function failed or returned None."
            raise RuntimeError(msg)

        return ret  # ty: ignore[invalid-return-type]

    return jaxify_wrapper
