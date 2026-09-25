"""robot_skin → common ← deformable_sats: common imports neither side, robot_skin never imports
deformable_sats packages."""
import ast
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SATS_PKGS = {"sats", "hitmap", "cnn_lstm", "deformable_sats"}


def _imported_roots(pkg_dir: Path) -> dict[str, set[str]]:
    out = {}
    for f in pkg_dir.rglob("*.py"):
        roots = set()
        for node in ast.walk(ast.parse(f.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Import):
                roots |= {a.name.split(".")[0] for a in node.names}
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                roots.add(node.module.split(".")[0])
        out[str(f.relative_to(REPO))] = roots
    return out


def test_common_imports_neither_package():
    bad = {f: r & (SATS_PKGS | {"robot_skin"}) for f, r in _imported_roots(REPO / "common").items()}
    assert not {f: r for f, r in bad.items() if r}


def test_robot_skin_does_not_import_deformable_sats():
    bad = {f: r & SATS_PKGS for f, r in _imported_roots(REPO / "robot_skin").items()}
    assert not {f: r for f, r in bad.items() if r}
