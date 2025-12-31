import ast
import importlib
import inspect
import textwrap
from collections.abc import Callable
from typing import ParamSpec, TypeVar

from jaxify._variables import get_locals

_Inputs = ParamSpec("_Inputs")
_Output = TypeVar("_Output")


class JaxifyError(Exception):
    pass


class _Transformer(ast.NodeTransformer):
    def __init__(self) -> None:
        super().__init__()
        self._if_count = 0
        self.__top_level_function = True

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.FunctionDef:
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
            node.body.insert(
                0,
                ast.Assign(
                    targets=[ast.Name(id="__jaxify_return", ctx=ast.Store())],
                    value=ast.Call(
                        func=ast.Attribute(
                            value=ast.Name(id="__jaxify_hooks", ctx=ast.Load()),
                            attr="Return",
                            ctx=ast.Load(),
                        ),
                        args=[],
                        keywords=[],
                    ),
                ),
            )
            node.body.append(
                ast.Return(
                    value=ast.Attribute(
                        ast.Name(id="__jaxify_return", ctx=ast.Load()),
                        attr="value",
                        ctx=ast.Load(),
                    )
                )
            )
        return node

    def visit_AsyncFunctionDef(
        self, node: ast.AsyncFunctionDef
    ) -> ast.AsyncFunctionDef:
        return node

    def visit_Lambda(self, node: ast.Lambda) -> ast.Lambda:
        return node

    def visit_If(self, node: ast.If) -> list[ast.stmt]:
        self.generic_visit(node)

        read_vars_body, written_vars_body = get_locals(node.body)
        read_vars_orelse, written_vars_orelse = get_locals(node.orelse)

        read_vars = read_vars_body | read_vars_orelse
        written_vars = written_vars_body | written_vars_orelse

        if (
            "__jaxify_return" in written_vars_body
            and "__jaxify_return" not in written_vars_orelse
        ):
            read_vars.add("__jaxify_return")
            type_stable_return_assignments_body = []
            type_stable_return_assignments_orelse = [
                ast.Assign(
                    targets=[ast.Name("__jaxify_return", ast.Store())],
                    value=ast.Call(
                        ast.Attribute(
                            ast.Name(id="__jaxify_return", ctx=ast.Load()),
                            attr="set_type",
                            ctx=ast.Load(),
                        ),
                        args=[
                            ast.Attribute(
                                value=ast.Subscript(
                                    ast.Call(
                                        ast.Name(
                                            f"__jaxify_if_true_{self._if_count}",
                                            ast.Load(),
                                        ),
                                        [
                                            ast.Name(var, ast.Load())
                                            for var in read_vars
                                        ],
                                        [],
                                    ),
                                    ast.Constant(
                                        list(written_vars).index("__jaxify_return")
                                    ),
                                    ast.Load(),
                                ),
                                attr="value",
                                ctx=ast.Load(),
                            )
                        ],
                        posonlyargs=[],
                        keywords=[],
                    ),
                )
            ]
        elif (
            "__jaxify_return" in written_vars_orelse
            and "__jaxify_return" not in written_vars_body
        ):
            read_vars.add("__jaxify_return")
            type_stable_return_assignments_body = [
                ast.Assign(
                    targets=[ast.Name("__jaxify_return", ast.Store())],
                    value=ast.Call(
                        ast.Attribute(
                            ast.Name(id="__jaxify_return", ctx=ast.Load()),
                            attr="set_type",
                            ctx=ast.Load(),
                        ),
                        args=[
                            ast.Subscript(
                                ast.Call(
                                    ast.Name(
                                        f"__jaxify_if_false_{self._if_count}",
                                        ast.Load(),
                                    ),
                                    [ast.Name(var, ast.Load()) for var in read_vars],
                                    [],
                                ),
                                ast.Constant(
                                    list(written_vars).index("__jaxify_return")
                                ),
                                ast.Load(),
                            )
                        ],
                        posonlyargs=[],
                        keywords=[],
                    ),
                )
            ]
            type_stable_return_assignments_orelse = []
        else:
            type_stable_return_assignments_body = []
            type_stable_return_assignments_orelse = []

        args = ast.arguments(
            posonlyargs=[],
            args=[ast.arg(arg=var) for var in read_vars],
            kwonlyargs=[],
            kw_defaults=[],
            defaults=[],
        )

        if_true = ast.FunctionDef(
            name=f"__jaxify_if_true_{self._if_count}",
            args=args,
            body=[
                *node.body,
                *type_stable_return_assignments_body,
                ast.Return(
                    value=ast.Tuple(
                        elts=[ast.Name(var, ast.Load()) for var in written_vars],
                        ctx=ast.Load(),
                    )
                ),
            ],
            decorator_list=[],
        )

        if_false = ast.FunctionDef(
            name=f"__jaxify_if_false_{self._if_count}",
            args=args,
            body=[
                *node.orelse,
                *type_stable_return_assignments_orelse,
                ast.Return(
                    value=ast.Tuple(
                        elts=[ast.Name(var, ast.Load()) for var in written_vars],
                        ctx=ast.Load(),
                    )
                ),
            ],
            decorator_list=[],
        )

        new_if = ast.Assign(
            targets=[
                ast.Tuple(
                    elts=[ast.Name(var, ast.Store()) for var in written_vars],
                    ctx=ast.Store(),
                )
            ],
            value=ast.Call(
                func=ast.Attribute(
                    value=ast.Name(id="__jaxify_hooks", ctx=ast.Load()),
                    attr="if_hook",
                    ctx=ast.Load(),
                ),
                args=[
                    node.test,
                    ast.Name(f"__jaxify_if_true_{self._if_count}", ast.Load()),
                    ast.Name(f"__jaxify_if_false_{self._if_count}", ast.Load()),
                    *[ast.Name(var, ast.Load()) for var in read_vars],
                ],
                keywords=[],
            ),
        )
        self._if_count += 1
        return [if_true, if_false, new_if]

    def visit_Return(self, node: ast.Return) -> ast.Assign:
        self.generic_visit(node)
        return ast.Assign(
            targets=[ast.Name(id="__jaxify_return", ctx=ast.Store())],
            value=ast.Call(
                ast.Attribute(
                    ast.Name(id="__jaxify_return", ctx=ast.Load()),
                    attr="return_",
                    ctx=ast.Load(),
                ),
                args=[node.value] if node.value else [],
                keywords=[],
            ),
        )

    def visit_IfExp(self, node: ast.IfExp) -> ast.AST:
        self.generic_visit(node)
        return ast.Call(
            func=ast.Attribute(
                value=ast.Name(id="__jaxify_hooks", ctx=ast.Load()),
                attr="if_hook",
                ctx=ast.Load(),
            ),
            args=[
                node.test,
                ast.Lambda(
                    args=ast.arguments(
                        posonlyargs=[],
                        args=[],
                        kwonlyargs=[],
                        kw_defaults=[],
                        defaults=[],
                    ),
                    body=node.body,
                ),
                ast.Lambda(
                    args=ast.arguments(
                        posonlyargs=[],
                        args=[],
                        kwonlyargs=[],
                        kw_defaults=[],
                        defaults=[],
                    ),
                    body=node.orelse,
                ),
            ],
            keywords=[],
        )

    def visit_BoolOp(self, node: ast.BoolOp) -> ast.AST:
        self.generic_visit(node)
        match node.op:
            case ast.And():
                return ast.Call(
                    func=ast.Attribute(
                        value=ast.Name(id="__jaxify_hooks", ctx=ast.Load()),
                        attr="and_hook",
                        ctx=ast.Load(),
                    ),
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
                    func=ast.Attribute(
                        value=ast.Name(id="__jaxify_hooks", ctx=ast.Load()),
                        attr="or_hook",
                        ctx=ast.Load(),
                    ),
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
                    func=ast.Attribute(
                        value=ast.Name(id="__jaxify_hooks", ctx=ast.Load()),
                        attr="not_hook",
                        ctx=ast.Load(),
                    ),
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
                func=ast.Attribute(
                    value=ast.Name(id="__jaxify_hooks", ctx=ast.Load()),
                    attr="and_hook",
                    ctx=ast.Load(),
                ),
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


def jaxify(func: Callable[_Inputs, _Output], /) -> Callable[_Inputs, _Output]:
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

    ast.fix_missing_locations(tree)

    local_vars = {}
    exec(  # noqa: S102
        compile(tree, filename="<ast>", mode="exec"),
        {
            **func.__globals__,
            "__jaxify_hooks": importlib.import_module("jaxify._hooks"),
        },
        local_vars,
    )
    traceable_func: Callable[_Inputs, _Output] = local_vars[func.__name__]

    return traceable_func
