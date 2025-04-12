from __future__ import annotations

import ast
import typing


class _ReadWriteVisitor(ast.NodeVisitor):
    def __init__(self) -> None:
        super().__init__()
        self.rw = ReadWrites({}, {})

    def visit_Name(self, node: ast.Name) -> None:
        if isinstance(node.ctx, ast.Load):
            self.rw.reads.setdefault(node.id)
        elif isinstance(node.ctx, ast.Store):
            self.rw.writes.setdefault(node.id)
        self.generic_visit(node)


class ReadWrites(typing.NamedTuple):
    reads: dict[str, None]  # dict not set for deterministic ordering
    writes: dict[str, None]

    def __iter__(self) -> typing.Iterator[str]:
        return iter({**self.reads, **self.writes})


def ast_read_writes(node: ast.AST) -> ReadWrites:
    visitor = _ReadWriteVisitor()
    visitor.visit(node)
    return visitor.rw
