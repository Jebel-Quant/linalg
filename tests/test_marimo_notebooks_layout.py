"""Tests for Marimo notebook source layout."""

from pathlib import Path


def test_marimo_notebooks_exist_in_book_folder():
    """Marimo notebook sources should live under book/marimo/notebooks."""
    root = Path(__file__).resolve().parents[1]
    notebooks_dir = root / "book" / "marimo" / "notebooks"

    assert notebooks_dir.is_dir()

    notebook_files = sorted(notebooks_dir.glob("*.py"))
    assert notebook_files
    assert (notebooks_dir / "end_to_end.py").is_file()
