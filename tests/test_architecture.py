"""Enforce: core (src/) must never import chainlit."""

from __future__ import annotations

import ast
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
FORBIDDEN = "chainlit"


def _imports(tree: ast.AST) -> list[tuple[int, str]]:
    found: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.extend((node.lineno, alias.name) for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            found.append((node.lineno, node.module))
        elif (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name | ast.Attribute)
            and getattr(node.func, "id", getattr(node.func, "attr", None))
            in {"import_module", "__import__"}
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and isinstance(node.args[0].value, str)
        ):
            found.append((node.lineno, node.args[0].value))
    return found


def test_src_has_python_files() -> None:
    assert list(SRC.rglob("*.py")), "no Python files found under src/"


def test_core_never_imports_chainlit() -> None:
    violations: list[str] = []
    for path in sorted(SRC.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for lineno, name in _imports(tree):
            if name == FORBIDDEN or name.startswith(FORBIDDEN + "."):
                violations.append(f"{path.relative_to(SRC.parent)}:{lineno} imports {name}")
    assert not violations, "core must not import chainlit:\n" + "\n".join(violations)


def test_detector_catches_imports() -> None:
    samples = [
        "import chainlit",
        "import chainlit as cl",
        "from chainlit import Message",
        "from chainlit.types import X",
        "import importlib; importlib.import_module('chainlit')",
        "__import__('chainlit')",
    ]
    for src in samples:
        names = [n for _, n in _imports(ast.parse(src))]
        assert any(n.split(".")[0] == FORBIDDEN for n in names), src
