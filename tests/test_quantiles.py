import numpy as np
import pandas as pd

from nfl_projections import quantiles as q_mod
from nfl_projections import evaluate


class _StubPre:
    def transform(self, X):
        return np.asarray(X, dtype=float)


class _StubModel:
    def __init__(self, out):
        self.out = np.asarray(out, dtype=float)

    def predict(self, X, verbose=0):
        return self.out


class TestPredictQuantiles:
    def test_monotonic_sort_and_naming(self):
        # deliberately unsorted network output -> must come back sorted, clipped >=0
        qm = q_mod.QuantileModel(
            model=_StubModel([[9.0, 3.0, 5.0], [-2.0, 1.0, 0.5]]),
            preprocessor=_StubPre(),
            numerical_features=[], categorical_features=[],
            quantiles=[0.1, 0.5, 0.9],
        )
        out = q_mod.predict_quantiles(qm, pd.DataFrame({"a": [1, 2]}))
        assert out.columns.tolist() == ["q10", "q50", "q90"]
        assert out.iloc[0].tolist() == [3.0, 5.0, 9.0]      # sorted ascending
        assert out.iloc[1].tolist() == [0.0, 0.5, 1.0]      # -2 clipped to 0, sorted

    def test_target_cols_property(self):
        qm = q_mod.QuantileModel(None, None, [], [], [0.1, 0.5, 0.9])
        assert qm.target_cols == ["fanduel_fantasy_points"]

    def test_offsets_applied_before_sort(self):
        qm = q_mod.QuantileModel(
            model=_StubModel([[2.0, 5.0, 9.0]]), preprocessor=_StubPre(),
            numerical_features=[], categorical_features=[],
            quantiles=[0.1, 0.5, 0.9], offsets=[-1.0, 3.0, 1.0],
        )
        out = q_mod.predict_quantiles(qm, pd.DataFrame({"a": [1]}))
        # 2-1=1, 5+3=8, 9+1=10 -> sorted [1,8,10]
        assert out.iloc[0].tolist() == [1.0, 8.0, 10.0]


class TestConformalOffsets:
    def test_offset_makes_median_unbiased(self):
        y = np.full(200, 12.0)
        raw = np.full((200, 1), 5.0)
        offs = q_mod.conformal_offsets(y, raw, [0.5])
        assert abs(offs[0] - 7.0) < 1e-6

    def test_offsets_recover_distribution_quantiles(self):
        rng = np.random.default_rng(0)
        y = rng.normal(10, 5, 5000)
        raw = np.zeros((5000, 3))
        offs = q_mod.conformal_offsets(y, raw, [0.1, 0.5, 0.9])
        assert offs[0] < offs[1] < offs[2]
        assert abs(offs[1] - 10.0) < 0.5


class TestSummarizeQuantiles:
    def _results(self):
        # 10 rows; actual increasing 1..10; floor/ceiling chosen for known coverage
        n = 10
        return pd.DataFrame({
            "position": ["RB"] * n,
            "actual": np.arange(1, n + 1, dtype=float),
            "q10": np.full(n, 2.0),
            "q50": np.arange(1, n + 1, dtype=float) + 0.0,  # perfect median
            "q90": np.full(n, 9.0),
        })

    def test_calibration_and_coverage(self):
        res = self._results()
        out = evaluate.summarize_quantiles(res, quantiles=[0.1, 0.5, 0.9])
        # interval [2,9] covers actuals 2..9 => 8 of 10
        within = ((res["actual"] >= res["q10"]) & (res["actual"] <= res["q90"])).mean()
        assert within == 0.8
        # median is perfect -> MAE 0
        assert (out["q50"] - out["actual"]).abs().mean() == 0.0
