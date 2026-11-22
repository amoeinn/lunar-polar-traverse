"""H, the contradiction rule's horizon, has one definition.

It was registered as a measured number and then written into six places in two
languages. Re-registering it after the sun units fix meant finding every copy,
and a copy that was missed would have kept the pre-fix value with nothing to
show it. These tests protect the consolidation: one definition in
src/executive.py, every Python route to the binary taking it from there, and
the binary holding none (that refusal is tested beside the binary, in
test_executive.py, since it needs the build).
"""
import inspect
import subprocess
from pathlib import Path

from src import executive as E

ROOT = Path(__file__).resolve().parents[1]


def test_every_python_default_is_the_one_definition():
    """Protects against a default that copies H's value instead of naming it.

    Identity, not equality: a copy in another module that happens to hold the
    current value passes an equality check today and fails silently at the next
    re-registration, which is exactly how the six copies would have drifted.
    Identity cannot see a literal copy inside src/executive.py itself, because
    CPython merges equal constants within one module into one object; measured,
    a literal default equal to H there passed this test. The line scan below catches that.
    """
    from examples import run_executive as R
    for fn in (E.run_leg, E.run_instance):
        d = inspect.signature(fn).parameters["contra_horizon_h"].default
        assert d is E.CONTRA_HORIZON_H, (fn.__name__, d)
    assert R.CONTRA_HORIZON_H is E.CONTRA_HORIZON_H, R.CONTRA_HORIZON_H


def test_the_value_is_written_in_one_tracked_source_line():
    """Protects against a new copy of H appearing somewhere the identity test
    does not look: another driver, a fixture, or the C++ again."""
    files = subprocess.run(
        ["git", "ls-files", "*.py", "*.cpp", "*.hpp", "*.h"], cwd=ROOT,
        capture_output=True, text=True, check=True).stdout.split()
    assert files, "git ls-files found no sources, so the scan would be vacuous"
    forms = {f"{E.CONTRA_HORIZON_H:.4f}", repr(E.CONTRA_HORIZON_H)}
    hits = [f"{f}:{n}" for f in files
            for n, line in enumerate((ROOT / f).read_text().splitlines(), 1)
            if any(x in line for x in forms)]
    assert len(hits) == 1 and hits[0].startswith("src/executive.py:"), hits
