from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_makefile_python_diff_filter_matches_dot_py_suffix():
    """The Makefile filter must not require a literal backslash before .py."""
    makefile = (ROOT / "Makefile").read_text()

    assert "grep -E '\\.py$$'" in makefile
    assert "grep -E '\\\\.py$$'" not in makefile


def test_lint_workflow_python_diff_filter_matches_dot_py_suffix():
    """The GitHub Actions diff filter must match changed Python files."""
    workflow = (ROOT / ".github" / "workflows" / "lint.yml").read_text()

    assert workflow.count("--diff-filter=ACMRTUXB") == 2
    assert "grep -E '\\.py$'" in workflow
    assert "grep -E '\\\\.py$'" not in workflow
