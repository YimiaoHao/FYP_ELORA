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

# 模型文件路径：app/ml/obesity_model.joblib
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


def _norm_gender(g: str) -> str:
    g = (g or "").strip().lower()
    if g.startswith("m") or g == "male":
        return "Male"
    # O / other 也先映射到 Female，避免模型没见过类别
    return "Female"


def _norm_family_history(x: str) -> str:
    s = (x or "").strip().lower()
    return "Y" if s in ("y", "yes", "1", "true") else "N"


def _norm_activity_level(x: str) -> str:
    s = (x or "").strip().lower()
    if s in ("mid", "moderate"):
        return "medium"
    if s not in ("low", "medium", "high"):
        return "medium"
    return s


def record_to_features(record) -> pd.DataFrame:
    """
    Map a Record (or record-like object) to the 7-feature DataFrame
    that matches train_obesity_model.py.
    """
    age = int(getattr(record, "age", 25) or 25)
    gender = _norm_gender(getattr(record, "gender", "F"))
    height_m = float(getattr(record, "height_m", 1.65) or 1.65)
    weight_kg = float(getattr(record, "weight_kg", 60.0) or 60.0)
    family_history = _norm_family_history(getattr(record, "family_history", "N"))
    activity_level = _norm_activity_level(getattr(record, "activity_level", "medium"))
    water_ml = int(getattr(record, "water_ml", 1500) or 1500)

    X = pd.DataFrame(
        [{
            "age": age,
            "gender": gender,
            "height_m": height_m,
            "weight_kg": weight_kg,
            "family_history": family_history,
            "activity_level": activity_level,
            "water_ml": water_ml,
        }]
    )
    return X


def predict_label(features_df: pd.DataFrame) -> Tuple[str, float]:
    """Return (label, confidence 0~1)."""
    model = load_model()

    # Pipeline 分类器通常支持 predict_proba
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(features_df)[0]
        classes = model.classes_
        idx = int(proba.argmax())
        return str(classes[idx]), float(proba[idx])

    # 兜底：没有 proba 就返回 0.0
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
