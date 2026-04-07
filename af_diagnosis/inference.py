"""Inference module for predicting on new samples."""

import os

import numpy as np
import pandas as pd

from .io import load_model


class AFPredictor:
    """Atrial Fibrillation predictor for new samples."""

    def __init__(self, model_path, threshold=0.5):
        """Initialize predictor."""
        self.model, self.metadata = load_model(model_path)
        self.threshold = threshold

        # Get feature names - use metadata or extract from model
        self.feature_names = None
        if self.metadata and "feature_names" in self.metadata:
            self.feature_names = self.metadata["feature_names"]

        # Try to get feature names from model's fitted stages
        if not self.feature_names:
            try:
                # Get feature names from the scaler in the pipeline
                scaler = self.model.named_steps["scaler"]
                if hasattr(scaler, "feature_names_in_"):
                    self.feature_names = scaler.feature_names_in_.tolist()
            except:
                pass

    def predict(self, X):
        """Predict AF status."""
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=self.feature_names)

        # Ensure correct feature order and fill missing with 0
        if self.feature_names:
            # Only keep features the model knows about
            available = [f for f in self.feature_names if f in X.columns]
            X = X[available]
            # Fill missing features with 0
            for f in self.feature_names:
                if f not in X.columns:
                    X[f] = 0
            # Reorder columns
            X = X[self.feature_names]

        return self.model.predict(X)

    def predict_proba(self, X):
        """Predict probabilities."""
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=self.feature_names)

        # Ensure correct feature order and fill missing with 0
        if self.feature_names:
            available = [f for f in self.feature_names if f in X.columns]
            X = X[available]
            for f in self.feature_names:
                if f not in X.columns:
                    X[f] = 0
            X = X[self.feature_names]

        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X)
        else:
            raise AttributeError("Model does not support predict_proba")

    def predict_with_confidence(self, X):
        """Predict with confidence scores."""
        proba = self.predict_proba(X)
        confidence = np.max(proba, axis=1)
        predictions = (proba[:, 1] >= self.threshold).astype(int)
        return predictions, confidence

    def predict_single(self, features):
        """Predict for a single sample."""
        X = pd.DataFrame([features])
        pred = self.predict(X)[0]
        proba = self.predict_proba(X)[0]

        return {
            "prediction": "AF" if pred == 1 else "Sinus Rhythm",
            "probability_af": float(proba[1]),
            "probability_sr": float(proba[0]),
            "confidence": float(max(proba))
        }


def load_predictor(model_path):
    """Load a predictor from saved model."""
    return AFPredictor(model_path)


def predict(X, model_path="results/rf_model.pkl", threshold=0.5):
    """Predict AF status for samples."""
    if isinstance(X, str):
        X = pd.read_csv(X, index_col=0)

    predictor = AFPredictor(model_path, threshold)
    return predictor.predict(X)


def predict_proba(X, model_path="results/rf_model.pkl"):
    """Predict probabilities for samples."""
    if isinstance(X, str):
        X = pd.read_csv(X, index_col=0)

    predictor = AFPredictor(model_path)
    return predictor.predict_proba(X)