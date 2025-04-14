from __future__ import annotations

import ast
import typing
from typing import TYPE_CHECKING
from typing import TypeVar

if TYPE_CHECKING:
    _A = TypeVar("_A", bound=ast.AST)


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
    """
    Analyze an Abstract Syntax Tree (AST) node to determine the variables
    that are read and written within it.

    This function traverses the given AST node and collects information
    about variable reads and writes using the `_ReadWriteVisitor` class.

    :param node: The root AST node to analyze.
    :return: A `ReadWrites` object containing dictionaries of read and
             written variable names.
    """
    visitor = _ReadWriteVisitor()
    visitor.visit(node)
    return visitor.rw


class _RenameVisitor(ast.NodeVisitor):
    def __init__(self, renames: dict[str, str]) -> None:
        super().__init__()
        self.renames = renames

    def visit_Name(self, node: ast.Name) -> None:
        node.id = self.renames.get(node.id, node.id)


def ast_rename(node: _A, renames: dict[str, str]) -> _A:
    """
    Rename variables in an Abstract Syntax Tree (AST) node, in-place.

    This function traverses the given AST node and renames variables
    based on the provided mapping of old names to new names.

    :param node: The root AST node to rename variables in.
    :param renames: A dictionary mapping old variable names to new variable names.
    :return: The modified AST node with variables renamed.
    """
    visitor = _RenameVisitor(renames)
    visitor.visit(node)
    return node
