# app/ml_service.py
"""
Single source of truth for ML inference.

- record_to_features(record) -> DataFrame with 7 columns (matches training)
- predict_label(df) -> (label, confidence)
- predict_obesity_level(record)
- predict_obesity_level_from_fields(...)
"""

from pathlib import Path
from typing import Tuple
from types import SimpleNamespace

import joblib
import pandas as pd

# Model file path: app/ml/obesity_model.joblib
MODEL_PATH = Path(__file__).resolve().parent / "ml" / "obesity_model.joblib"
_model = None


def load_model():
    """Lazy-load model (joblib) once."""
    global _model
    if _model is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
        _model = joblib.load(MODEL_PATH)
    return _model


# --------- Normalizers (match training) ---------
def normalize_gender_app(v) -> str:
    """
    Normalize app gender to training format: 'Male'/'Female'.
    Accepts: F/M/O, female/male, 0/1, true/false, yes/no.
    """
    if v is None:
        return "Female"
    s = str(v).strip().lower()
    if s in {"m", "male", "1", "true", "y", "yes"}:
        return "Male"
    if s in {"f", "female", "0", "false", "n", "no"}:
        return "Female"
    # unknown (e.g., 'o') -> fallback (pipeline ignores unknown categories, but keep stable)
    return "Female"


def normalize_family_history(v) -> str:
    """Normalize family history to training format: 'Y'/'N'."""
    if v is None:
        return "N"
    s = str(v).strip().lower()
    if s in {"y", "yes", "true", "1", "1.0"}:
        return "Y"
    if s in {"n", "no", "false", "0", "0.0"}:
        return "N"
    return "N"


def _norm_activity_level(x) -> str:
    """Normalize activity level to: low/medium/high."""
    if x is None:
        return "medium"
    s = str(x).strip().lower()
    if s in {"mid", "moderate"}:
        return "medium"
    if s not in {"low", "medium", "high"}:
        return "medium"
    return s


# --------- Safe numeric parsers ---------
def _to_int(v, default: int) -> int:
    if v is None or v == "":
        return default
    try:
        return int(float(v))
    except Exception:
        return default


def _to_float(v, default: float) -> float:
    if v is None or v == "":
        return default
    try:
        return float(v)
    except Exception:
        return default


def record_to_features(record) -> pd.DataFrame:
    """
    Map a Record (or record-like object) to the 7-feature DataFrame
    that matches train_obesity_model.py.

    Expected canonical columns:
    age, gender, height_m, weight_kg, family_history, activity_level, water_ml
    """
    age = _to_int(getattr(record, "age", None), default=25)

    gender_raw = getattr(record, "gender", None)
    gender = normalize_gender_app(gender_raw if gender_raw is not None else "F")

    height_m = _to_float(getattr(record, "height_m", None), default=1.65)
    weight_kg = _to_float(getattr(record, "weight_kg", None), default=60.0)

    # Support legacy field name: family_hist
    fh_raw = (
        getattr(record, "family_history", None)
        or getattr(record, "family_hist", None)
        or "N"
    )
    family_history = normalize_family_history(fh_raw)

    activity_raw = (
        getattr(record, "activity_level", None)
        or getattr(record, "activity", None)
        or "medium"
    )
    activity_level = _norm_activity_level(activity_raw)

    # Support possible legacy names
    water_raw = (
        getattr(record, "water_ml", None)
        or getattr(record, "water", None)
        or getattr(record, "water_intake", None)
        or 1500
    )
    water_ml = _to_int(water_raw, default=1500)

    X = pd.DataFrame([{
        "age": age,
        "gender": gender,
        "height_m": height_m,
        "weight_kg": weight_kg,
        "family_history": family_history,
        "activity_level": activity_level,
        "water_ml": water_ml,
    }])

    return X


def predict_label(features_df: pd.DataFrame) -> Tuple[str, float]:
    """Return (label, confidence 0~1)."""
    model = load_model()

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(features_df)[0]
        classes = model.classes_
        idx = int(proba.argmax())
        return str(classes[idx]), float(proba[idx])

    pred = model.predict(features_df)[0]
    return str(pred), 0.0


def predict_obesity_level(record) -> Tuple[str, float]:
    X = record_to_features(record)
    return predict_label(X)


def predict_obesity_level_from_fields(
    age: int,
    gender: str,
    height_m: float,
    weight_kg: float,
    family_history: str,
    activity_level: str,
    water_ml: int,
) -> Tuple[str, float]:
    fake = SimpleNamespace(
        age=age,
        gender=gender,
        height_m=height_m,
        weight_kg=weight_kg,
        family_history=family_history,
        activity_level=activity_level,
        water_ml=water_ml,
    )
    return predict_obesity_level(fake)
