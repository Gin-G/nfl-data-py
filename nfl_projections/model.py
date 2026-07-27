"""Neural network training, persistence, and batch prediction.

This module imports tensorflow/sklearn; other modules import it lazily so the
pure-pandas parts of the package (and the tests) work without them installed.
"""

import json
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import config, features

# Caps applied to raw network output to keep predictions sane
PREDICTION_CAPS = {
    "fanduel_fantasy_points": (0, 70),
    "yards": (0, 400),
    "tds_or_receptions": (0, 6),
}


@dataclass
class TrainedModel:
    """A trained network plus everything needed to run it on new rows."""

    model: object  # keras Model
    preprocessor: object  # sklearn ColumnTransformer
    numerical_features: list
    categorical_features: list
    target_cols: list

    def save(self, directory=config.MODELS_DIR):
        import joblib

        os.makedirs(directory, exist_ok=True)
        self.model.save(os.path.join(directory, "projection_model.keras"))
        joblib.dump(self.preprocessor, os.path.join(directory, "preprocessor.joblib"))
        with open(os.path.join(directory, "metadata.json"), "w") as f:
            json.dump(
                {
                    "numerical_features": self.numerical_features,
                    "categorical_features": self.categorical_features,
                    "target_cols": self.target_cols,
                },
                f,
                indent=2,
            )
        print(f"Model saved to {directory}/")

    @classmethod
    def load(cls, directory=config.MODELS_DIR):
        import joblib
        from tensorflow.keras.models import load_model

        with open(os.path.join(directory, "metadata.json")) as f:
            meta = json.load(f)
        return cls(
            model=load_model(os.path.join(directory, "projection_model.keras")),
            preprocessor=joblib.load(os.path.join(directory, "preprocessor.joblib")),
            **meta,
        )


def _resolve_loss(loss):
    """Map a loss name to a keras loss. 'huber' uses delta=5 (FanDuel-point scale)."""
    if loss == "huber":
        from tensorflow.keras.losses import Huber

        return Huber(delta=5.0)
    return loss  # "mse", "mae", etc. pass through as keras string losses


def build_network(input_dim, output_dim, loss="mse"):
    """Dense network with light regularization for prediction variance."""
    from tensorflow.keras.layers import BatchNormalization, Dense, Dropout, Input
    from tensorflow.keras.models import Model
    from tensorflow.keras.optimizers import Adam

    inputs = Input(shape=(input_dim,))
    x = Dense(512, activation="relu")(inputs)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    x = Dense(256, activation="relu")(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    x = Dense(128, activation="relu")(x)
    x = Dropout(0.15)(x)
    x = Dense(64, activation="relu")(x)
    x = Dropout(0.1)(x)
    outputs = Dense(output_dim, activation="linear")(x)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=Adam(learning_rate=0.0015, beta_1=0.9, beta_2=0.999),
        loss=_resolve_loss(loss),
        metrics=["mae"],
    )
    return model


def prepare_training_data(df, min_season=config.TRAINING_MIN_SEASON, matchup_table=None):
    """Clean, target-shift, and featurize the dataset for training.

    Returns (X, y, frame, numerical_features, categorical_features, target_cols)
    where frame holds the rows aligned with X/y (used for the time-based split).

    When ``matchup_table`` is given (from opponent.build_matchup_table), each
    row also gets the *next* game's opponent-defense features.
    """
    df_clean = features.clean_training_data(df, min_season=min_season)
    df_clean = features.add_rolling_features(df_clean)
    target_cols = [c for c in config.TARGET_COLS if c in df_clean.columns]

    if matchup_table is not None:
        from . import opponent

        df_clean = opponent.add_next_game_keys(df_clean)

    df_targets = features.add_next_week_targets(df_clean, target_cols)
    df_final = features.add_derived_features(df_targets)

    if matchup_table is not None:
        from . import opponent

        df_final = opponent.attach_training_features(df_final, matchup_table)

    numerical, categorical = features.select_feature_columns(df_final)
    df_final = features.clean_categorical_features(df_final, categorical)

    X = df_final[categorical + numerical]
    y = df_final[[f"next_week_{c}" for c in target_cols]].values

    valid = ~np.isnan(y).any(axis=1) & ~np.isinf(y).any(axis=1)
    return X[valid], y[valid], df_final[valid], numerical, categorical, target_cols


def train_model(
    df,
    epochs=100,
    batch_size=64,
    min_season=config.TRAINING_MIN_SEASON,
    save_dir=None,
    plot_path=config.LEARNING_CURVES_PATH,
    verbose=1,
    matchup_table=None,
    loss="mse",
):
    """Train the projection network on the historical dataset.

    Returns (TrainedModel, keras History). Set save_dir to persist the model,
    plot_path=None to skip the learning-curve PNG. Pass ``matchup_table`` (from
    opponent.build_matchup_table) to train with opponent-defense features.
    ``loss`` selects the training loss ("mse" default, or "huber", "mae").
    """
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder, RobustScaler
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

    X, y, frame, numerical, categorical, target_cols = prepare_training_data(
        df, min_season=min_season, matchup_table=matchup_table
    )
    print(f"Training data: X={X.shape}, y={y.shape}, targets={target_cols}")

    preprocessor = ColumnTransformer([
        ("num", RobustScaler(), numerical),
        ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), categorical),
    ])
    X_transformed = preprocessor.fit_transform(X)

    # Chronological 80/20 split so validation is always "the future"
    order = frame.reset_index(drop=True).sort_values(["season", "week"]).index.values
    split = int(0.8 * len(order))
    train_idx, val_idx = order[:split], order[split:]

    X_train, X_val = X_transformed[train_idx], X_transformed[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    print(f"Train set: {X_train.shape}, Validation set: {X_val.shape}")

    model = build_network(X_train.shape[1], len(target_cols), loss=loss)

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[
            ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=10,
                              min_lr=1e-6, verbose=verbose),
            EarlyStopping(monitor="val_loss", patience=20,
                          restore_best_weights=True, verbose=verbose),
        ],
        verbose=verbose,
    )

    trained = TrainedModel(model, preprocessor, numerical, categorical, target_cols)

    if plot_path:
        _plot_learning_curves(history, plot_path)
    if save_dir:
        trained.save(save_dir)

    return trained, history


def _plot_learning_curves(history, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history["loss"], label="Training Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.title("Model Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history["mae"], label="Training MAE")
    plt.plot(history.history["val_mae"], label="Validation MAE")
    plt.title("Mean Absolute Error")
    plt.xlabel("Epoch")
    plt.ylabel("MAE")
    plt.legend()

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"Learning curves saved to {path}")


def build_input_rows(trained, stat_rows, positions, teams, opponent_features=None):
    """Assemble a model-input DataFrame from per-player latest-game stat rows.

    Args:
        trained: TrainedModel
        stat_rows: DataFrame of each player's most recent game (one row each)
        positions: iterable of position strings aligned with stat_rows
        teams: iterable of team strings aligned with stat_rows
        opponent_features: optional DataFrame/dict of opponent-defense columns
            aligned row-for-row with stat_rows; supplies the opp_* / is_home_game
            features for the *upcoming* matchup. Missing values fall back to
            opponent.NEUTRAL_VALUES.
    """
    from .opponent import NEUTRAL_VALUES, OPPONENT_FEATURES

    opp = None
    if opponent_features is not None:
        opp = pd.DataFrame(opponent_features).reset_index(drop=True)

    data = {}
    for col in trained.numerical_features:
        if col in OPPONENT_FEATURES:
            if opp is not None and col in opp.columns:
                data[col] = pd.to_numeric(opp[col], errors="coerce").fillna(
                    NEUTRAL_VALUES[col]
                ).values
            else:
                data[col] = np.full(len(stat_rows), NEUTRAL_VALUES[col])
        elif col in stat_rows.columns:
            data[col] = pd.to_numeric(stat_rows[col], errors="coerce").fillna(0).values
        else:
            data[col] = np.zeros(len(stat_rows))

    positions = [str(p) if p and not pd.isna(p) else "Unknown" for p in positions]
    teams = [str(t) if t and not pd.isna(t) else "Unknown" for t in teams]
    for cat_col in trained.categorical_features:
        if "position" in cat_col:
            data[cat_col] = positions
        elif cat_col == "recent_team" or "team" in cat_col:
            data[cat_col] = teams
        else:
            data[cat_col] = ["Unknown"] * len(stat_rows)

    return pd.DataFrame(data)


def cap_prediction(stat, value):
    """Clamp a predicted stat to a plausible range."""
    if stat == "fanduel_fantasy_points":
        lo, hi = PREDICTION_CAPS["fanduel_fantasy_points"]
    elif "yards" in stat:
        lo, hi = PREDICTION_CAPS["yards"]
    elif "tds" in stat or stat == "receptions":
        lo, hi = PREDICTION_CAPS["tds_or_receptions"]
    else:
        lo, hi = 0, np.inf
    return float(np.clip(value, lo, hi))


def predict_batch(trained, input_df):
    """Run the network on assembled input rows; returns a capped DataFrame."""
    transformed = trained.preprocessor.transform(input_df)
    raw = trained.model.predict(transformed, verbose=0)

    out = {}
    for i, stat in enumerate(trained.target_cols):
        out[stat] = [round(cap_prediction(stat, v), 2) for v in raw[:, i]]
    return pd.DataFrame(out)
