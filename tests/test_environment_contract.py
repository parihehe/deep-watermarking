"""Tests that do not require optional scientific dependencies."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_project_declares_python_312_contract() -> None:
    pyproject = (PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert 'requires-python = ">=3.12,<3.13"' in pyproject


def test_phase_zero_documents_exist() -> None:
    phase_zero = PROJECT_ROOT / "docs" / "phase0"
    required = {
        "research_gap.md",
        "paper_comparison.md",
        "research_questions.md",
        "proposed_contributions.md",
    }
    assert {path.name for path in phase_zero.glob("*.md")} >= required
