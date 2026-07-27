"""Floor / median / ceiling projections via quantile regression.

The main model predicts the *expected* fantasy points (a mean, correctly geared
toward the average). This module adds a companion network that predicts several
*quantiles* of next-game fantasy points with the pinball loss, giving a floor
(q10) and ceiling (q90) around the projection. Unlike the mean model, quantile
regression can put a genuinely high ceiling on boom-prone players.

It reuses the exact feature pipeline (rolling usage features and all), so floor
and ceiling benefit from the same inputs as the point projection.
"""

import json
import os
from dataclasses import dataclass

import numpy as np

from . import config, model as model_mod

DEFAULT_QUANTILES = (0.1, 0.5, 0.9)


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

    # so model.build_input_rows can assemble inputs for us
    @property
    def target_cols(self):
        return ["fanduel_fantasy_points"]

    def save(self, directory):
        import joblib

        os.makedirs(directory, exist_ok=True)
        self.model.save(os.path.join(directory, "quantile_model.keras"))
        joblib.dump(self.preprocessor, os.path.join(directory, "q_preprocessor.joblib"))
        with open(os.path.join(directory, "q_metadata.json"), "w") as f:
            json.dump({
                "numerical_features": self.numerical_features,
                "categorical_features": self.categorical_features,
                "quantiles": list(self.quantiles),
                "offsets": list(self.offsets) if self.offsets is not None else None,
            }, f, indent=2)

    @classmethod
    def load(cls, directory):
        import joblib
        from tensorflow.keras.models import load_model

        with open(os.path.join(directory, "q_metadata.json")) as f:
            meta = json.load(f)
        return cls(
            model=load_model(os.path.join(directory, "quantile_model.keras"), compile=False),
            preprocessor=joblib.load(os.path.join(directory, "q_preprocessor.joblib")),
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
                         save_dir=None, verbose=1, calibrate=True):
    """Train a quantile model for next-week fantasy points. Returns (QuantileModel, history)."""
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder, RobustScaler
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    from tensorflow.keras.optimizers import Adam

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

    net = build_quantile_network(X_t.shape[1], len(quantiles))
    net.compile(optimizer=Adam(learning_rate=0.0015), loss=pinball_loss(quantiles))
    history = net.fit(
        X_t[train_idx], y_fp[train_idx],
        validation_data=(X_t[val_idx], y_fp[val_idx]),
        epochs=epochs, batch_size=batch_size,
        callbacks=[
            ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=10, min_lr=1e-6, verbose=verbose),
            EarlyStopping(monitor="val_loss", patience=20, restore_best_weights=True, verbose=verbose),
        ],
        verbose=verbose,
    )

    offsets = None
    cal_idx = cal_pos if len(cal_pos) else val_idx
    if calibrate and len(cal_idx) > 0:
        cal_raw = np.sort(net.predict(X_t[cal_idx], verbose=0), axis=1)
        offsets = conformal_offsets(y_fp[cal_idx], cal_raw, quantiles)

    qmodel = QuantileModel(net, preprocessor, numerical, categorical, quantiles, offsets)
    if save_dir:
        qmodel.save(save_dir)
    return qmodel, history


def predict_quantiles(qmodel, input_df):
    """Return a DataFrame of quantile predictions, monotonically sorted per row.

    Columns are named q10, q50, q90, ... matching qmodel.quantiles.
    """
    import pandas as pd

    transformed = qmodel.preprocessor.transform(input_df)
    raw = np.sort(qmodel.model.predict(transformed, verbose=0), axis=1)

    if qmodel.offsets is not None:
        raw = raw + np.asarray(qmodel.offsets)

    # Enforce non-crossing quantiles and floor at 0
    raw = np.clip(np.sort(raw, axis=1), 0, None)
    cols = [f"q{int(round(q * 100))}" for q in qmodel.quantiles]
    return pd.DataFrame(raw, columns=cols)
