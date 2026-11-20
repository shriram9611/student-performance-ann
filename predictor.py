"""Loads the saved ANN + scaler and turns a dict of student features into a prediction."""
import joblib
import numpy as np

from common import CLASSES, FEATURE_SPECS, FEATURES, MODEL_PATH, SCALER_PATH

SPEC_BY_NAME = {s["name"]: s for s in FEATURE_SPECS}


class ValidationError(ValueError):
    """Raised when the incoming features are missing, non-numeric or out of range."""


class Predictor:
    def __init__(self, model, scaler):
        self.model = model          # anything with .predict(X) -> class probabilities
        self.scaler = scaler

    @classmethod
    def from_disk(cls):
        if not MODEL_PATH.exists() or not SCALER_PATH.exists():
            raise FileNotFoundError(
                "Trained model not found in models/. Run `python train.py` first.")
        from tensorflow import keras            # imported lazily
        return cls(keras.models.load_model(MODEL_PATH), joblib.load(SCALER_PATH))

    @staticmethod
    def validate(features: dict) -> list:
        if not isinstance(features, dict):
            raise ValidationError("Input must be a JSON object of feature values.")
        missing = [f for f in FEATURES if f not in features]
        if missing:
            raise ValidationError(f"Missing features: {', '.join(missing)}")
        values = []
        for name in FEATURES:
            try:
                v = float(features[name])
            except (TypeError, ValueError):
                raise ValidationError(f"'{name}' must be a number.")
            if not np.isfinite(v):
                raise ValidationError(f"'{name}' must be a finite number.")
            spec = SPEC_BY_NAME[name]
            if not spec["min"] <= v <= spec["max"]:
                raise ValidationError(
                    f"'{name}' must be between {spec['min']} and {spec['max']} (got {v}).")
            values.append(v)
        return values

    def predict(self, features: dict) -> dict:
        values = self.validate(features)
        x = self.scaler.transform(np.array([values], dtype="float32"))
        probs = np.asarray(self.model.predict(x, verbose=0))[0]
        idx = int(np.argmax(probs))
        return {
            "predicted_performance": CLASSES[idx],
            "confidence": round(float(probs[idx]), 4),
            "probabilities": {c: round(float(p), 4) for c, p in zip(CLASSES, probs)},
        }
