from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _read(p: Path) -> str:
    return p.read_text(encoding="utf-8")


def test_root_onboarding_docs_are_redirect_stubs() -> None:
    """
    Prevent regressions where large onboarding docs get duplicated at repo root.
    Canonical copies live under PROJECT_MEMO/.
    """

    root_global = ROOT / "GLOBAL_CURSOR_INSTRUCTIONS.md"
    root_mem = ROOT / "MEMORY_AND_RULES.md"

    assert root_global.exists()
    assert root_mem.exists()

    g = _read(root_global)
    m = _read(root_mem)

    # Must be short stubs.
    assert len(g.splitlines()) <= 20
    assert len(m.splitlines()) <= 20

    # Must point to canonical locations.
    assert "PROJECT_MEMO/GLOBAL_CURSOR_INSTRUCTIONS.md" in g
    assert "PROJECT_MEMO/MEMORY_AND_RULES.md" in m


def test_canonical_onboarding_docs_exist_under_project_memo() -> None:
    canon_global = ROOT / "PROJECT_MEMO" / "GLOBAL_CURSOR_INSTRUCTIONS.md"
    canon_mem = ROOT / "PROJECT_MEMO" / "MEMORY_AND_RULES.md"

    assert canon_global.exists()
    assert canon_mem.exists()

    # Canonical docs should be non-trivial.
    assert len(_read(canon_global).splitlines()) >= 50
    assert len(_read(canon_mem).splitlines()) >= 20


