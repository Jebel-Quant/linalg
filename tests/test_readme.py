"""Execute the ``pycon`` examples in README.md as doctests.

Running ``python -m doctest README.md`` is no substitute: it reads each closing
fence as expected output. So the fences are extracted here and handed to
:class:`doctest.DocTestParser` one by one, sharing a single namespace in README
order so a later fence may use names an earlier one defined.

Nothing here is specific to one project: it expects ``README.md`` one level above
this file, and the test environment to provide whatever the examples import. A
README without ``pycon`` fences is reported as a skip, not a pass.
"""

from __future__ import annotations

import doctest
import re
from pathlib import Path

import pytest

README = Path(__file__).resolve().parents[1] / "README.md"
PYCON_FENCE = re.compile(r"^```pycon\n(.*?)^```$", re.DOTALL | re.MULTILINE)


def _fences() -> list[tuple[int, str]]:
    """Return each ``pycon`` fence in README.md with the line it starts on."""
    if not README.is_file():
        return []
    text = README.read_text(encoding="utf-8")
    return [(text.count("\n", 0, m.start()) + 1, m.group(1)) for m in PYCON_FENCE.finditer(text)]


def test_readme_pycon_examples() -> None:
    """Test that every README ``pycon`` example produces its documented output."""
    fences = _fences()
    if not fences:
        pytest.skip("README.md has no pycon fences")

    parser = doctest.DocTestParser()
    runner = doctest.DocTestRunner(optionflags=doctest.ELLIPSIS)
    globs: dict[str, object] = {}
    report: list[str] = []

    for line, source in fences:
        test = parser.get_doctest(source, globs, f"README.md:{line}", str(README), line)
        runner.run(test, out=report.append, clear_globs=False)

    assert runner.failures == 0, "".join(report)
