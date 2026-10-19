"""The package must be usable without TensorFlow installed.

NFL-API's web image carries no TensorFlow — it serves cached rows and, now,
builds FanDuel lineups, neither of which needs a model. Importing the optimizer
used to drag in the whole training stack through `__init__`, so this pins the
lazy boundary: pandas-level modules import on their own, and the service names
still resolve when something actually asks for them.
"""

import pathlib
import subprocess
import sys
import textwrap


def _run(code):
    return subprocess.run([sys.executable, "-c", textwrap.dedent(code)],
                          capture_output=True, text=True)


def test_optimizer_imports_without_loading_tensorflow():
    done = _run("""
        import sys
        from nfl_projections import optimizer
        assert optimizer.ROSTER_SLOTS["QB"] == 1
        assert "tensorflow" not in sys.modules, sorted(
            m for m in sys.modules if "tensor" in m)
        print("ok")
    """)
    assert done.returncode == 0, done.stderr
    assert "ok" in done.stdout


def test_importing_the_package_alone_loads_no_model_stack():
    done = _run("""
        import sys
        import nfl_projections
        assert nfl_projections.__version__
        for heavy in ("tensorflow", "sklearn", "nfl_projections.model"):
            assert heavy not in sys.modules, heavy
        print("ok")
    """)
    assert done.returncode == 0, done.stderr
    assert "ok" in done.stdout


def test_service_names_still_resolve_lazily():
    done = _run("""
        from nfl_projections import ProjectionService
        assert ProjectionService.__name__ == "ProjectionService"
        import nfl_projections
        assert "ProjectionService" in dir(nfl_projections)
        print("ok")
    """)
    assert done.returncode == 0, done.stderr
    assert "ok" in done.stdout


def test_a_missing_name_still_raises_attribute_error():
    done = _run("""
        import nfl_projections
        try:
            nfl_projections.does_not_exist
        except AttributeError as exc:
            assert "does_not_exist" in str(exc)
            print("ok")
    """)
    assert done.returncode == 0, done.stderr
    assert "ok" in done.stdout


def test_pyarrow_and_numpy_can_actually_talk_to_each_other():
    """The break that took out two production jobs on 2026-10-15.

    pyarrow 18+ declares no numpy requirement, so pip resolved 26.0.0 against a
    numpy that predates 2.0 and the pairing only failed at *runtime*:
    "pyarrow requires NumPy 2.0 or newer, found 1.26.4". Every unit test passed
    while `load_players().to_pandas()` raised in the cluster.

    So this exercises the bridge rather than reading a version number: a
    parquet round-trip through pandas is the narrowest thing that fails when
    the two are mismatched, and it needs no network.
    """
    done = _run("""
        import io
        import pandas as pd
        buf = io.BytesIO()
        pd.DataFrame({"gsis_id": ["00-1"], "week": [6], "yards": [72.5]}).to_parquet(buf)
        buf.seek(0)
        back = pd.read_parquet(buf)
        assert back.loc[0, "yards"] == 72.5
        print("ok")
    """)
    assert done.returncode == 0, done.stderr
    assert "ok" in done.stdout


def test_the_pyarrow_cap_is_declared():
    """A companion to the test above, which can only fail once a bad pair is
    already installed. The cap is what stops pip assembling one, and it has to
    hold in both files — the cronjob installs this package from the tarball,
    which reads pyproject, over an image that installed from requirements.
    """
    root = pathlib.Path(__file__).resolve().parent.parent
    for name in ("pyproject.toml", "requirements.txt"):
        text = (root / name).read_text()
        assert "pyarrow>=7.0.0,<18" in text, (
            f"{name} lost the pyarrow cap; see the comment there before "
            "lifting it, because numpy has to move at the same time")
