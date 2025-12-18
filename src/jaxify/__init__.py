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
import jax.numpy as jnp

_Inputs = ParamSpec("_Inputs")
_Output = TypeVar("_Output")


class JaxifyError(Exception):
    pass


class _Transformer(ast.NodeTransformer):
    def __init__(self) -> None:
        super().__init__()
        self.nconds = 0
        self.__top_level_function = True

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:
        if self.__top_level_function:
            for i, decorator in enumerate(node.decorator_list):
                match decorator:
                    case (
                        ast.Name(id="jaxify")
                        | ast.Attribute(value=ast.Name(id="jaxify"), attr="jaxify")
                    ):
                        del node.decorator_list[: i + 1]
                        break
            self.__top_level_function = False
            self.generic_visit(node)
        return node

    def visit_AsyncFunctionDef(
        self, node: ast.AsyncFunctionDef
    ) -> ast.AsyncFunctionDef:
        return node

    def visit_Lambda(self, node: ast.Lambda) -> ast.Lambda:
        return node

    def visit_If(self, node: ast.If) -> ast.AST:
        self.generic_visit(node)
        node.test = ast.Call(
            func=ast.Name(id="__jaxify_cond_hook__", ctx=ast.Load()),
            args=[
                node.test,
                ast.Constant(value=self.nconds),
            ],
            keywords=[],
        )
        self.nconds += 1
        return node

    def visit_IfExp(self, node: ast.IfExp) -> ast.AST:
        self.generic_visit(node)
        node.test = ast.Call(
            func=ast.Name(id="__jaxify_cond_hook__", ctx=ast.Load()),
            args=[
                node.test,
                ast.Constant(value=self.nconds),
            ],
            keywords=[],
        )
        self.nconds += 1
        return node

    def visit_BoolOp(self, node: ast.BoolOp) -> ast.AST:
        self.generic_visit(node)
        match node.op:
            case ast.And():
                return ast.Call(
                    func=ast.Name(id="__jaxify_and_hook__", ctx=ast.Load()),
                    args=[
                        ast.Lambda(
                            args=ast.arguments(
                                posonlyargs=[],
                                args=[],
                                kwonlyargs=[],
                                kw_defaults=[],
                                defaults=[],
                            ),
                            body=value,
                        )
                        for value in node.values
                    ],
                    keywords=[],
                )
            case ast.Or():
                return ast.Call(
                    func=ast.Name(id="__jaxify_or_hook__", ctx=ast.Load()),
                    args=[
                        ast.Lambda(
                            args=ast.arguments(
                                posonlyargs=[],
                                args=[],
                                kwonlyargs=[],
                                kw_defaults=[],
                                defaults=[],
                            ),
                            body=value,
                        )
                        for value in node.values
                    ],
                    keywords=[],
                )
        return node

    def visit_UnaryOp(self, node: ast.UnaryOp) -> ast.AST:
        self.generic_visit(node)
        match node.op:
            case ast.Not():
                return ast.Call(
                    func=ast.Name(id="__jaxify_not_hook__", ctx=ast.Load()),
                    args=[node.operand],
                    keywords=[],
                )
        return node

    def visit_Compare(self, node: ast.Compare) -> ast.AST:
        self.generic_visit(node)
        if len(node.ops) > 1:
            new_nodes = []
            left = node.left
            for op, comparator in zip(node.ops, node.comparators, strict=True):
                new_compare = ast.Compare(
                    left=left,
                    ops=[op],
                    comparators=[comparator],
                )
                new_nodes.append(new_compare)
                left = comparator
            return ast.Call(
                func=ast.Name(id="__jaxify_and_hook__", ctx=ast.Load()),
                args=[
                    ast.Lambda(
                        args=ast.arguments(
                            posonlyargs=[],
                            args=[],
                            kwonlyargs=[],
                            kw_defaults=[],
                            defaults=[],
                        ),
                        body=new_node,
                    )
                    for new_node in new_nodes
                ],
                keywords=[],
            )
        return node

    def visit_For(self, node: ast.For) -> ast.AST:  # noqa: ARG002
        msg = "jaxify does not currently support loops"
        raise JaxifyError(msg)

    def visit_While(self, node: ast.While) -> ast.AST:  # noqa: ARG002
        msg = "jaxify does not currently support loops"
        raise JaxifyError(msg)


def _and_hook(*args: Callable[[], object]) -> object:
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


def _or_hook(*args: Callable[[], object]) -> object:
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


def _not_hook(value: object) -> object:
    match value:
        case jax.core.Tracer(size=1):
            return jnp.logical_not(value)
        case _:
            return not value


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
    if inspect.isasyncgenfunction(func):
        msg = "jaxify does not support async generator functions"
        raise TypeError(msg)

    try:
        source = inspect.getsource(func)
    except Exception as e:
        msg = "Could not retrieve source code for function"
        raise JaxifyError(msg) from e

    try:
        tree = ast.parse(textwrap.dedent(source))
    except Exception as e:
        msg = "Could not parse source code into AST"
        raise JaxifyError(msg) from e

    transformer = _Transformer()
    tree = transformer.visit(tree)
    nconds = transformer.nconds

    ast.fix_missing_locations(tree)

    local_vars = {}
    exec(  # noqa: S102
        compile(tree, filename="<ast>", mode="exec"),
        {
            **func.__globals__,
            "__jaxify_cond_hook__": lambda cond, cond_id: cond,  # noqa: ARG005
            "__jaxify_and_hook__": _and_hook,
            "__jaxify_or_hook__": _or_hook,
            "__jaxify_not_hook__": _not_hook,
        },
        local_vars,
    )
    traceable_func: Callable[_Inputs, _Output] = local_vars[func.__name__]

    @functools.wraps(func)
    def jaxify_wrapper(*args: _Inputs.args, **kwargs: _Inputs.kwargs) -> _Output:  # noqa: C901
        if nconds == 0:
            return traceable_func(*args, **kwargs)

        cond_combinations: list[tuple[bool, ...]] = list(
            itertools.product([False, True], repeat=nconds)
        )
        cond_values: list[list[object | None]] = []
        outputs: list[_Output] = []
        combination: tuple[bool, ...] | None = None

        def cond_hook(cond: object, cond_id: int) -> object:
            match cond:
                case jax.core.Tracer():
                    values[cond_id] = cond
                    assert combination is not None
                    return combination[cond_id]
                case _:
                    return cond

        traceable_func_local = type(traceable_func)(
            traceable_func.__code__,
            {
                **traceable_func.__globals__,
                "__jaxify_cond_hook__": cond_hook,
            },
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
                                mask = jnp.logical_and(mask, cond)
                            case False:
                                mask = jnp.logical_and(mask, jnp.logical_not(cond))
                if ret is None:
                    ret = outputs[i]
                else:
                    ret = jax.lax.cond(
                        mask,
                        lambda _=outputs[i]: _,
                        lambda _=ret: _,
                    )

        if ret is None:
            warnings.warn(
                "jaxify: all branches returned None", RuntimeWarning, stacklevel=2
            )

        return ret  # ty: ignore[invalid-return-type]

    return jaxify_wrapper
