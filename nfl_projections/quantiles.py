"""Floor / median / ceiling projections via quantile regression.

The main model predicts the *expected* fantasy points (a mean, correctly geared
toward the average). This module adds a companion network that predicts several
*quantiles* of next-game fantasy points with the pinball loss, giving a floor
and ceiling around the projection. Unlike the mean model, quantile regression
can put a genuinely high ceiling on boom-prone players.

The band is an ~80% interval with a calibrated ceiling; the floor sits nearer a
20th percentile because a fifth of player-weeks score zero. See
DEFAULT_QUANTILES for the measurements behind those choices.

It reuses the exact feature pipeline (rolling usage features and all), so floor
and ceiling benefit from the same inputs as the point projection.
"""

import json
import os
from dataclasses import dataclass, field

import numpy as np

from . import config, model as model_mod

# Nominal quantiles chosen so the EMPIRICAL coverage lands where we want it.
# Asking the network for q10/q90 produced a 71% band, not 80% (2025 backtest:
# empirical 0.249 / 0.865). Widening the nominal band to 5/95 lands at
# 0.205 / 0.903 — an honest 80.3% interval with a well-calibrated ceiling, at
# the cost of a wider band (11.0 -> 13.9 points) and no change in median MAE.
#
# The floor stays near 0.20 no matter how low the nominal quantile goes
# (q10 -> 0.249, q05 -> 0.205, q02 -> 0.196) and that is structural, not a
# model failure: 19.1% of player-weeks score ZERO or less (WR 23%, TE 20%,
# RB 16%, QB 8%), and predictions are clipped at 0, so no non-negative floor
# can cover better than ~0.19. Read the floor as roughly a 20th percentile.
DEFAULT_QUANTILES = (0.05, 0.5, 0.95)


def pinball_loss(quantiles):
    """Keras pinball (quantile) loss over several quantiles of one target.

    y_pred is (N, len(quantiles)); y_true is the single target broadcast across
    the quantile columns.
    """
    import tensorflow as tf

    q = tf.constant(quantiles, dtype=tf.float32)

    def loss(y_true, y_pred):
        y_true = tf.reshape(tf.cast(y_true, tf.float32), (-1, 1))
        e = y_true - y_pred
        return tf.reduce_mean(tf.maximum(q * e, (q - 1.0) * e))

    return loss


def build_quantile_network(input_dim, n_quantiles):
    from tensorflow.keras.layers import BatchNormalization, Dense, Dropout, Input
    from tensorflow.keras.models import Model
    from tensorflow.keras.optimizers import Adam

    inputs = Input(shape=(input_dim,))
    x = Dense(256, activation="relu")(inputs)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    x = Dense(128, activation="relu")(x)
    x = Dropout(0.15)(x)
    x = Dense(64, activation="relu")(x)
    x = Dropout(0.1)(x)
    outputs = Dense(n_quantiles, activation="linear")(x)
    return Model(inputs=inputs, outputs=outputs)


@dataclass
class QuantileModel:
    model: object
    preprocessor: object
    numerical_features: list
    categorical_features: list
    quantiles: list
    # Per-quantile additive conformal offsets, calibrated on a fully held-out
    # season so they reflect true next-season difficulty. None -> uncalibrated.
    offsets: list = None
    # Additional networks of a seed ensemble, averaged at prediction time — the
    # same seed lottery the mean model has (see model.TrainedModel).
    extra_models: list = field(default_factory=list)

    # so model.build_input_rows can assemble inputs for us
    @property
    def target_cols(self):
        return ["fanduel_fantasy_points"]

    @property
    def members(self):
        return [self.model, *self.extra_models]

    @property
    def n_members(self):
        return 1 + len(self.extra_models)

    def save(self, directory):
        import joblib

        os.makedirs(directory, exist_ok=True)
        self.model.save(os.path.join(directory, "quantile_model.keras"))
        for i, member in enumerate(self.extra_models, start=1):
            member.save(os.path.join(directory, f"quantile_model_seed{i}.keras"))
        joblib.dump(self.preprocessor, os.path.join(directory, "q_preprocessor.joblib"))
        with open(os.path.join(directory, "q_metadata.json"), "w") as f:
            json.dump({
                "numerical_features": self.numerical_features,
                "categorical_features": self.categorical_features,
                "quantiles": list(self.quantiles),
                "offsets": list(self.offsets) if self.offsets is not None else None,
                "n_members": self.n_members,
            }, f, indent=2)

    @classmethod
    def load(cls, directory):
        import joblib
        from tensorflow.keras.models import load_model

        with open(os.path.join(directory, "q_metadata.json")) as f:
            meta = json.load(f)
        n_members = meta.pop("n_members", 1)  # absent in pre-ensemble saves
        return cls(
            model=load_model(os.path.join(directory, "quantile_model.keras"), compile=False),
            preprocessor=joblib.load(os.path.join(directory, "q_preprocessor.joblib")),
            extra_models=[
                load_model(os.path.join(directory, f"quantile_model_seed{i}.keras"),
                           compile=False)
                for i in range(1, n_members)
            ],
            **meta,
        )


def conformal_offsets(y_true, raw_pred, quantiles):
    """Additive per-quantile offsets so empirical coverage matches nominal.

    For quantile tau we want P(y <= pred + delta) = tau, so delta is the
    tau-quantile of the residuals (y - pred) on the held-out set. Additive (not
    multiplicative) because it is robust to the calibration set being an
    imperfect proxy for the target season.
    """
    resid = y_true.reshape(-1, 1) - raw_pred
    return [float(np.quantile(resid[:, k], q)) for k, q in enumerate(quantiles)]


def train_quantile_model(df, quantiles=DEFAULT_QUANTILES, epochs=100, batch_size=64,
                         min_season=config.TRAINING_MIN_SEASON, matchup_table=None,
                         save_dir=None, verbose=1, calibrate=True, n_seeds=None):
    """Train a quantile model for next-week fantasy points. Returns (QuantileModel, history).

    ``n_seeds`` networks are trained and averaged at prediction time (default
    config.DEFAULT_N_SEEDS), for the same reason the mean model ensembles: a
    single seed is a lottery. Conformal offsets are calibrated on the ENSEMBLE's
    predictions, not one member's, so the calibration matches what is served.
    """
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder, RobustScaler
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.utils import set_random_seed

    quantiles = list(quantiles)
    X, y, frame, numerical, categorical, target_cols = model_mod.prepare_training_data(
        df, min_season=min_season, matchup_table=matchup_table
    )
    fp_idx = target_cols.index("fanduel_fantasy_points")
    y_fp = y[:, fp_idx]

    preprocessor = ColumnTransformer([
        ("num", RobustScaler(), numerical),
        ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), categorical),
    ])
    X_t = preprocessor.fit_transform(X)

    frame_r = frame.reset_index(drop=True)
    seasons = frame_r["season"].values

    # Hold out the most recent training season for conformal calibration, so the
    # offsets reflect true next-season difficulty (an internal split is in-era
    # and too optimistic). Only if it leaves at least two seasons to fit on.
    cal_pos = np.array([], dtype=int)
    fit_pos = np.arange(len(X_t))
    if calibrate and frame_r["season"].nunique() >= 3:
        cal_season = seasons.max()
        cal_pos = np.where(seasons == cal_season)[0]
        fit_pos = np.where(seasons != cal_season)[0]

    fit_order = frame_r.iloc[fit_pos].sort_values(["season", "week"]).index.values
    split = int(0.8 * len(fit_order))
    train_idx, val_idx = fit_order[:split], fit_order[split:]

    n_seeds = config.DEFAULT_N_SEEDS if n_seeds is None else n_seeds
    nets, history = [], None
    for i, seed in enumerate(range(n_seeds), start=1):
        if n_seeds > 1:
            print(f"Training quantile network {i}/{n_seeds} (seed={seed})...")
        set_random_seed(seed)
        net = build_quantile_network(X_t.shape[1], len(quantiles))
        net.compile(optimizer=Adam(learning_rate=0.0015), loss=pinball_loss(quantiles))
        hist = net.fit(
            X_t[train_idx], y_fp[train_idx],
            validation_data=(X_t[val_idx], y_fp[val_idx]),
            epochs=epochs, batch_size=batch_size,
            callbacks=[
                ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=10, min_lr=1e-6, verbose=verbose),
                EarlyStopping(monitor="val_loss", patience=20, restore_best_weights=True, verbose=verbose),
            ],
            verbose=verbose,
        )
        nets.append(net)
        history = history or hist

    offsets = None
    cal_idx = cal_pos if len(cal_pos) else val_idx
    if calibrate and len(cal_idx) > 0:
        # Average the members first: the offsets must calibrate the ensemble
        cal_raw = np.mean([n.predict(X_t[cal_idx], verbose=0) for n in nets], axis=0)
        cal_raw = np.sort(cal_raw, axis=1)
        offsets = conformal_offsets(y_fp[cal_idx], cal_raw, quantiles)

    qmodel = QuantileModel(nets[0], preprocessor, numerical, categorical, quantiles,
                           offsets, extra_models=nets[1:])
    if save_dir:
        qmodel.save(save_dir)
    return qmodel, history


def predict_quantiles(qmodel, input_df):
    """Return a DataFrame of quantile predictions, monotonically sorted per row.

    Columns are named q10, q50, q90, ... matching qmodel.quantiles.
    """
    import pandas as pd

    transformed = qmodel.preprocessor.transform(input_df)
    members = getattr(qmodel, "members", [qmodel.model])
    if len(members) == 1:
        raw = members[0].predict(transformed, verbose=0)
    else:
        raw = np.mean([m.predict(transformed, verbose=0) for m in members], axis=0)
    raw = np.sort(raw, axis=1)

    if qmodel.offsets is not None:
        raw = raw + np.asarray(qmodel.offsets)

    # Enforce non-crossing quantiles and floor at 0
    raw = np.clip(np.sort(raw, axis=1), 0, None)
    cols = [f"q{int(round(q * 100))}" for q in qmodel.quantiles]
    return pd.DataFrame(raw, columns=cols)
