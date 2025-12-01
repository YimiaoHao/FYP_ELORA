# app/ml/ml_service.py
"""
加载训练好的 obesity_model.joblib，
并提供两个函数：
- record_to_features(record) -> pandas.DataFrame
- predict_label(features_df) -> (label_str, confidence_float)
"""

from pathlib import Path
from typing import Tuple
import joblib
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "obesity_model.joblib"

_model = None  # 缓存模型


def get_model():
    """懒加载模型，如果文件不存在会抛异常"""
    global _model
    if _model is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Model file not found at: {MODEL_PATH}")
        _model = joblib.load(MODEL_PATH)
    return _model


def _get_attr(record, *names, default=None):
    """兼容不同字段名，例如 family_history / family_history_overweight"""
    for name in names:
        if hasattr(record, name):
            return getattr(record, name)
    return default


def record_to_features(record) -> pd.DataFrame:
    """
    把数据库里的一条 Record 记录，转换成一行特征 DataFrame。

    注意：不同训练脚本里用到的特征名可能不一致，
    所以这里先给出一个“超集”的特征，然后再根据模型本身
    的 feature_names_in_ 做一次对齐（只保留模型真的需要的列）。
    """

    # ---------- 1. 从 record 取出已有信息 ----------
    age = _get_attr(record, "age", default=None)
    gender = (_get_attr(record, "gender", default="M") or "M").upper()

    height = _get_attr(record, "height_m", default=None)
    weight = _get_attr(record, "weight_kg", default=None)

    # 家族史：Record 里可能叫 family_history / family_history_overweight
    fam_raw = _get_attr(record, "family_history", "family_history_overweight", default="N")
    fam_raw = (fam_raw or "").strip().lower()
    fam_yes_no = "yes" if fam_raw in ("y", "yes", "1", "true") else "no"

    # 活动水平 -> FAF
    act = (_get_attr(record, "activity_level", default="medium") or "medium").lower()
    if act == "low":
        faf = 1.0
    elif act == "high":
        faf = 3.0
    else:
        faf = 2.0

    # 喝水量：ml -> 大概换算成每天几升 [1,3]
    water_ml = _get_attr(record, "water_ml", default=0) or 0
    ch2o = water_ml / 1000.0
    ch2o = float(max(1.0, min(3.0, ch2o)))

    # ---------- 2. 构造一个“超集”特征字典 ----------
    data = {
        # 数值特征
        "Age": [age],
        "Height": [height],
        "Weight": [weight],
        "FCVC": [2.0],   # demo 默认值
        "NCP": [3.0],
        "CH2O": [ch2o],
        "FAF": [faf],
        "TUE": [1.0],

        # 类别特征（根据 Kaggle 字段）
        "Gender": [gender],
        "family_history_with_overweight": [fam_yes_no],
        "family_history": [fam_yes_no],  # 兼容如果模型里用了这个名字
        "FAVC": ["no"],
        "CAEC": ["Sometimes"],
        "SMOKE": ["no"],
        "SCC": ["no"],
        "CALC": ["Sometimes"],
        "MTRANS": ["Public_Transportation"],
    }

    df_all = pd.DataFrame(data)

    # ---------- 3. 根据模型需要的列做一次对齐 ----------
    try:
        model = get_model()
        if hasattr(model, "feature_names_in_"):
            needed_cols = list(model.feature_names_in_)
            # 对于模型需要但我们没提供的列，先补 NaN，让 Imputer 自己处理
            for col in needed_cols:
                if col not in df_all.columns:
                    df_all[col] = np.nan
            df_all = df_all[needed_cols]
    except Exception:
        # 如果连模型都加载不了，那在 predict 那一步再抛错即可
        pass

    return df_all


def predict_label(features_df: pd.DataFrame) -> Tuple[str, float]:
    """
    用训练好的模型做一次预测，返回 (标签字符串, 置信度0~1)
    """
    model = get_model()

    preds = model.predict(features_df)
    label = str(preds[0])

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(features_df)[0]
        idx = int(np.argmax(proba))
        confidence = float(proba[idx])
    else:
        confidence = 0.0

    return label, confidence
