import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Input
import joblib
import os
from config import TARGET_COLS, CATEGORICAL_FEATURES, NUMERICAL_FEATURES

def create_preprocessor():
    """
    Create the preprocessor for the input data.
    """
    return ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), NUMERICAL_FEATURES),
            ('cat', OneHotEncoder(sparse=False, handle_unknown='ignore'), CATEGORICAL_FEATURES)
        ])

def create_model(input_dim):
    """
    Create a neural network model.
    """
    model = Sequential([
        Input(shape=(input_dim,)),
        Dense(128, activation='relu'),
        Dense(64, activation='relu'),
        Dense(32, activation='relu'),
        Dense(1, activation='linear')
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

def create_and_train_models(df):
    """
    Create and train models for each target statistic.
    """
    # Ensure all required columns are present
    available_features = [col for col in CATEGORICAL_FEATURES + NUMERICAL_FEATURES if col in df.columns]
    
    print(f"Available features: {available_features}")
    
    X = df[available_features]
    
    categorical_features = [col for col in CATEGORICAL_FEATURES if col in df.columns]
    numerical_features = [col for col in NUMERICAL_FEATURES if col in df.columns]
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(sparse=False, handle_unknown='ignore'), categorical_features)
        ])
    
    X_transformed = preprocessor.fit_transform(X)
    input_dim = X_transformed.shape[1]
    
    models = {}
    for stat in TARGET_COLS:
        y = df[f'next_week_{stat}'].values
        mask = ~np.isnan(y)
        X_stat = X_transformed[mask]
        y_stat = y[mask]
        
        model = create_model(input_dim)
        model.fit(X_stat, y_stat, epochs=100, batch_size=32, validation_split=0.2, verbose=0)
        models[stat] = model
    
    return models, preprocessor

def save_models(models, preprocessor):
    for model_name, model in models.items():
        joblib.dump(model, f'saved_models/model_{model_name}.joblib')
    joblib.dump(preprocessor, 'saved_models/preprocessor.joblib')

def load_models():
    models = {}
    for filename in os.listdir('saved_models'):
        if filename.startswith('model_') and filename.endswith('.joblib'):
            model_name = filename[6:-7]  # Remove 'model_' prefix and '.joblib' suffix
            models[model_name] = joblib.load(f'saved_models/{filename}')
    preprocessor = joblib.load('saved_models/preprocessor.joblib')
    return models, preprocessor

def update_models(models, preprocessor, new_data, current_week, current_season):
    """
    Update existing models with new data.
    """
    X = new_data[CATEGORICAL_FEATURES + NUMERICAL_FEATURES]
    X_transformed = preprocessor.transform(X)
    
    for stat in TARGET_COLS:
        y = new_data[f'next_week_{stat}'].values
        mask = ~np.isnan(y)
        X_stat = X_transformed[mask]
        y_stat = y[mask]
        
        if len(y_stat) > 0:
            models[stat].fit(X_stat, y_stat, epochs=10, batch_size=32, verbose=0)
    
    # Save updated models
    save_models(models, preprocessor, f'saved_models_week_{current_week}_{current_season}')
    
    return models

def predict_with_models(models, preprocessor, X):
    """
    Make predictions using the trained models.
    """
    X_transformed = preprocessor.transform(X)
    predictions = {}
    
    for stat in TARGET_COLS:
        predictions[stat] = models[stat].predict(X_transformed).flatten()
    
    return predictions

def evaluate_models(models, preprocessor, X, y):
    """
    Evaluate the performance of the models.
    """
    X_transformed = preprocessor.transform(X)
    evaluation = {}
    
    for stat in TARGET_COLS:
        y_true = y[f'next_week_{stat}'].values
        y_pred = models[stat].predict(X_transformed).flatten()
        mse = np.mean((y_true - y_pred)**2)
        evaluation[stat] = {'mse': mse}
    
    return evaluation