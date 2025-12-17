import ast
import functools
import inspect
import itertools
import textwrap
from collections.abc import Callable
from typing import Generic, ParamSpec, TypeVar

import jax

_Inputs = ParamSpec("_Inputs")
_Output = TypeVar("_Output")


class jitx(Generic[_Inputs, _Output]):  # noqa: N801
    def __init__(self, func: Callable[_Inputs, _Output]) -> None:
        self._func = func
        self._source = textwrap.dedent(inspect.getsource(func))
        tree = ast.parse(self._source)

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

                case ast.FunctionDef() if node.name == func.__name__:  # ty: ignore[unresolved-attribute]
                    node.decorator_list = []

        ast.fix_missing_locations(tree)
        self._nconds = nconds

        self._traceable = compile(tree, filename="<ast>", mode="exec")

    @functools.partial(jax.jit, static_argnums=0)
    def __call__(self, *args: _Inputs.args, **kwargs: _Inputs.kwargs) -> _Output:  # noqa: C901
        if not self._nconds:
            return self._func(*args, **kwargs)

        cond_combinations: list[tuple[bool, ...]] = list(
            itertools.product([False, True], repeat=self._nconds)
        )
        cond_values: list[list[object | None]] = []
        outputs: list[_Output] = []
        for combination in cond_combinations:
            values = [None] * self._nconds

            def _jaxify_cond(cond: object, cond_id: int) -> bool:
                if isinstance(cond, jax.Array):
                    values[cond_id] = cond  # noqa: B023
                    return combination[cond_id]  # noqa: B023
                return bool(cond)

            local_vars: dict[str, object] = {}
            exec(  # noqa: S102
                self._traceable,
                {**self._func.__globals__, "_jaxify_cond": _jaxify_cond},  # ty: ignore[possibly-missing-attribute]
                local_vars,
            )
            func = local_vars[next(iter(local_vars))]

            try:
                result = func(*args, **kwargs)  # ty: ignore[call-non-callable]
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
                ret = jax.lax.cond(
                    mask,
                    lambda _=outputs[i]: _,
                    lambda _=ret: _,
                )

        return ret  # ty: ignore[invalid-return-type]
