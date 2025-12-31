import ast

_BUILTIN_NAMES = set(dir(__builtins__))


class _LocalVarVisitor(ast.NodeVisitor):
    def __init__(self) -> None:
        self.written = set()
        self.read = set()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:  # noqa: ARG002
        return

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:  # noqa: ARG002
        return

    def visit_Lambda(self, node: ast.Lambda) -> None:  # noqa: ARG002
        return

    def visit_Attribute(self, node: ast.Attribute) -> None:  # noqa: ARG002
        return

    def visit_Name(self, node: ast.Name) -> None:
        match node.ctx:
            case ast.Store():
                self.written.add(node.id)
            case ast.Load() if node.id in self.written or node.id not in _BUILTIN_NAMES:
                self.read.add(node.id)


def get_locals(tree: list[ast.stmt], /) -> tuple[set[str], set[str]]:
    visitor = _LocalVarVisitor()
    for stmt in tree:
        visitor.visit(stmt)

    drop = set()
    for var in visitor.read:
        if var.startswith("__jaxify_") and var != "__jaxify_return":
            drop.add(var)
    return visitor.read - drop, visitor.written
