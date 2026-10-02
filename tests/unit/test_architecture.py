"""Import direction between packages (DESIGN.md §6)."""

import ast
from pathlib import Path

import pytest

from crucible_entity_manager.components.base import Services
from crucible_entity_manager.runtime.context import Context

PACKAGE = Path(__file__).resolve().parents[2] / "src" / "crucible_entity_manager"
PREFIX = "crucible_entity_manager."

ALLOWED: dict[str, frozenset[str]] = {
    "core": frozenset({"core"}),
    "config": frozenset({"core", "config"}),
    "hooks": frozenset({"core", "config", "hooks"}),
    "crucible": frozenset({"core", "config", "crucible"}),
    "components": frozenset(
        {
            "core",
            "config",
            "hooks",
            "components",
            "crucible.management",
            "crucible.protocols",
            "crucible.sql",
            "crucible.writer",
        }
    ),
}
"""Which internal modules each package may import. `runtime` may import anything."""

THIRD_PARTY: dict[str, frozenset[str]] = {
    "cruciblelib": frozenset({"crucible/client.py"}),
    "httpx": frozenset({"crucible/client.py"}),
    "aiohttp": frozenset({"crucible/sse.py", "runtime/health.py"}),
}
"""Third-party packages confined to particular modules."""


def imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(), str(path))
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            found.add(node.module)
    return found


def modules() -> list[Path]:
    return sorted(PACKAGE.rglob("*.py"))


def permitted(internal: str, allowed: frozenset[str]) -> bool:
    return any(internal == rule or internal.startswith(f"{rule}.") for rule in allowed)


@pytest.mark.parametrize("path", modules(), ids=lambda path: str(path.relative_to(PACKAGE)))
def test_imports_follow_the_layering(path: Path) -> None:
    relative = path.relative_to(PACKAGE)
    package = relative.parts[0] if len(relative.parts) > 1 else None
    for name in imports(path):
        if name.startswith(PREFIX) and package in ALLOWED:
            internal = name.removeprefix(PREFIX)
            assert permitted(internal, ALLOWED[package]), f"{relative} imports {name}"
        root = name.split(".")[0]
        if root in THIRD_PARTY:
            assert relative.as_posix() in THIRD_PARTY[root], f"{relative} imports {name}"


def as_services(context: Context) -> Services:
    """Never called: type-checking it verifies that `Context` satisfies `Services`."""
    return context
