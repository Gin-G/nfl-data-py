"""Unit tests for TrainedModel / seed ensembling (no tensorflow needed)."""
import json
import os

import numpy as np
import pandas as pd
import pytest

from nfl_projections import model as model_mod


class FakeNetwork:
    """Stands in for a keras Model: fixed predictions, and a save() that
    writes a file so persistence can be checked without tensorflow."""

    def __init__(self, values):
        self.values = np.asarray(values, dtype=float)

    def predict(self, X, verbose=0):
        return np.tile(self.values, (len(X), 1))

    def save(self, path):
        with open(path, "w") as f:
            f.write("fake-network")


class FakePreprocessor:
    def transform(self, df):
        return np.zeros((len(df), 1))


def _trained(*value_rows, target_cols=("rushing_yards", "fanduel_fantasy_points"),
             target_scaler=None):
    networks = [FakeNetwork(v) for v in value_rows]
    return model_mod.TrainedModel(
        model=networks[0],
        preprocessor=FakePreprocessor(),
        numerical_features=["avg_fppg"],
        categorical_features=["position"],
        target_cols=list(target_cols),
        extra_models=networks[1:],
        target_scaler=target_scaler,
    )


def test_single_model_has_one_member():
    trained = _trained([50.0, 10.0])
    assert trained.n_members == 1
    assert trained.members == [trained.model]
    assert trained.extra_models == []


def test_ensemble_members_include_every_network():
    trained = _trained([50.0, 10.0], [60.0, 12.0], [70.0, 14.0])
    assert trained.n_members == 3
    assert len(trained.members) == 3
    assert trained.members[0] is trained.model


def test_predict_batch_averages_ensemble_members():
    trained = _trained([50.0, 10.0], [70.0, 20.0])
    out = model_mod.predict_batch(trained, pd.DataFrame({"x": [1, 2, 3]}))

    assert list(out.columns) == ["rushing_yards", "fanduel_fantasy_points"]
    assert len(out) == 3
    assert (out["rushing_yards"] == 60.0).all()  # mean of 50 and 70
    assert (out["fanduel_fantasy_points"] == 15.0).all()


def test_predict_batch_single_member_is_that_member():
    trained = _trained([50.0, 10.0])
    out = model_mod.predict_batch(trained, pd.DataFrame({"x": [1, 2]}))
    assert (out["rushing_yards"] == 50.0).all()
    assert (out["fanduel_fantasy_points"] == 10.0).all()


def test_averaging_happens_before_the_caps():
    # 70 is the fanduel cap: averaging 60 and 80 must give 70, not cap-then-mean 65
    trained = _trained([0.0, 60.0], [0.0, 80.0])
    out = model_mod.predict_batch(trained, pd.DataFrame({"x": [1]}))
    assert out["fanduel_fantasy_points"].iloc[0] == 70.0


def test_save_persists_every_member_and_the_count(tmp_path):
    trained = _trained([50.0, 10.0], [60.0, 12.0], [70.0, 14.0])
    trained.save(str(tmp_path))

    assert os.path.exists(tmp_path / "projection_model.keras")
    assert os.path.exists(tmp_path / "projection_model_seed1.keras")
    assert os.path.exists(tmp_path / "projection_model_seed2.keras")
    assert not os.path.exists(tmp_path / "projection_model_seed3.keras")

    with open(tmp_path / "metadata.json") as f:
        meta = json.load(f)
    assert meta["n_members"] == 3
    assert meta["target_cols"] == ["rushing_yards", "fanduel_fantasy_points"]


class TestTargetScaler:
    def test_round_trips_values(self):
        y = np.array([[250.0, 0.05], [180.0, 0.0], [300.0, 0.2], [220.0, 0.1]])
        scaler = model_mod.TargetScaler.fit(y)

        scaled = scaler.transform(y)
        assert np.allclose(scaled.mean(axis=0), 0, atol=1e-9)
        assert np.allclose(scaled.std(axis=0), 1, atol=1e-9)
        assert np.allclose(scaler.inverse_transform(scaled), y)

    def test_puts_wildly_different_scales_on_equal_footing(self):
        # passing_yards vs rushing_tds: the exact mismatch that starved the
        # small-magnitude heads of gradient under a shared MSE loss
        y = np.array([[250.0, 0.05], [180.0, 0.0], [300.0, 0.2], [220.0, 0.1]])
        scaled = model_mod.TargetScaler.fit(y).transform(y)
        assert scaled[:, 0].std() == pytest.approx(scaled[:, 1].std())

    def test_constant_target_does_not_divide_by_zero(self):
        y = np.array([[5.0, 3.0], [7.0, 3.0], [9.0, 3.0]])
        scaler = model_mod.TargetScaler.fit(y)
        scaled = scaler.transform(y)
        assert np.isfinite(scaled).all()
        assert np.allclose(scaler.inverse_transform(scaled), y)

    def test_survives_a_json_round_trip(self):
        y = np.array([[250.0, 0.05], [180.0, 0.0], [300.0, 0.2]])
        scaler = model_mod.TargetScaler.fit(y)
        revived = model_mod.TargetScaler.from_dict(json.loads(json.dumps(scaler.to_dict())))
        assert np.allclose(revived.transform(y), scaler.transform(y))


def test_predict_batch_inverts_the_target_scaler():
    # network emits scaled space: +1 sigma on each target
    scaler = model_mod.TargetScaler(mean=[100.0, 10.0], scale=[20.0, 4.0])
    trained = _trained([1.0, 1.0], target_scaler=scaler)
    out = model_mod.predict_batch(trained, pd.DataFrame({"x": [1]}))
    assert out["rushing_yards"].iloc[0] == pytest.approx(120.0)
    assert out["fanduel_fantasy_points"].iloc[0] == pytest.approx(14.0)


def test_ensemble_members_are_averaged_then_unscaled():
    scaler = model_mod.TargetScaler(mean=[100.0, 10.0], scale=[20.0, 4.0])
    trained = _trained([0.0, -1.0], [2.0, 1.0], target_scaler=scaler)
    out = model_mod.predict_batch(trained, pd.DataFrame({"x": [1]}))
    assert out["rushing_yards"].iloc[0] == pytest.approx(120.0)   # mean(0,2)=1 sigma
    assert out["fanduel_fantasy_points"].iloc[0] == pytest.approx(10.0)  # mean(-1,1)=0


def test_scaler_is_persisted_with_the_model(tmp_path):
    scaler = model_mod.TargetScaler(mean=[100.0, 10.0], scale=[20.0, 4.0])
    _trained([1.0, 1.0], target_scaler=scaler).save(str(tmp_path))

    with open(tmp_path / "metadata.json") as f:
        meta = json.load(f)
    assert meta["target_scaler"]["mean"] == [100.0, 10.0]
    assert meta["target_scaler"]["scale"] == [20.0, 4.0]


def test_model_without_a_scaler_predicts_raw_values(tmp_path):
    trained = _trained([50.0, 10.0])
    assert trained.target_scaler is None
    out = model_mod.predict_batch(trained, pd.DataFrame({"x": [1]}))
    assert out["rushing_yards"].iloc[0] == pytest.approx(50.0)

    trained.save(str(tmp_path))
    with open(tmp_path / "metadata.json") as f:
        assert json.load(f)["target_scaler"] is None


def test_train_ensemble_needs_at_least_one_seed():
    with pytest.raises(ValueError):
        model_mod.train_ensemble(pd.DataFrame(), seeds=[])


def test_default_ensemble_size_matches_config():
    from nfl_projections import config

    assert model_mod.DEFAULT_N_SEEDS == config.DEFAULT_N_SEEDS >= 1
