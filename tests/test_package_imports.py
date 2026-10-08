"""The package must be usable without TensorFlow installed.

NFL-API's web image carries no TensorFlow — it serves cached rows and, now,
builds FanDuel lineups, neither of which needs a model. Importing the optimizer
used to drag in the whole training stack through `__init__`, so this pins the
lazy boundary: pandas-level modules import on their own, and the service names
still resolve when something actually asks for them.
"""

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
