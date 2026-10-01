"""Execute the ``pycon`` examples in README.md as doctests.

The README's examples are written as ``pycon`` transcripts, which the template's
README check does not run. Running ``python -m doctest README.md`` is no
substitute: it reads each closing fence as expected output. So the fences are
extracted here and handed to :class:`doctest.DocTestParser` one by one, sharing
a single namespace in README order so a later fence may use names an earlier one
defined.
"""

from __future__ import annotations

import doctest
import importlib.util
import re
from pathlib import Path

README = Path(__file__).resolve().parents[1] / "README.md"
PYCON_FENCE = re.compile(r"^```pycon\n(.*?)^```$", re.DOTALL | re.MULTILINE)
EWM_MODULE = "cvx.linalg.covariance.ewm_cov"


def _fences() -> list[tuple[int, str]]:
    """Return each ``pycon`` fence in README.md with the line it starts on."""
    text = README.read_text(encoding="utf-8")
    return [(text.count("\n", 0, m.start()) + 1, m.group(1)) for m in PYCON_FENCE.finditer(text)]


def test_readme_has_pycon_examples() -> None:
    """Test that the extraction finds the README's examples, so the doctest below cannot pass vacuously."""
    assert _fences()


def test_readme_pycon_examples() -> None:
    """Test that every README ``pycon`` example produces its documented output."""
    has_polars = importlib.util.find_spec("polars") is not None
    parser = doctest.DocTestParser()
    runner = doctest.DocTestRunner(optionflags=doctest.ELLIPSIS)
    globs: dict[str, object] = {}
    report: list[str] = []

    for line, source in _fences():
        if EWM_MODULE in source and not has_polars:  # polars ships with the dev group, so this only skips outside it
            continue
        test = parser.get_doctest(source, globs, f"README.md:{line}", str(README), line)
        runner.run(test, out=report.append, clear_globs=False)

    assert runner.failures == 0, "".join(report)
