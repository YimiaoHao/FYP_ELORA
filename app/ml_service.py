# app/ml_service.py
"""
Utility functions to load the trained obesity model and
make predictions.

- predict_obesity_level(record):              给 Assessment 等地方用（传入 Record 实例）
- predict_obesity_level_from_fields(...):    给 /api/predict 用（传入 age/gender 等字段）
"""

from pathlib import Path
from typing import Tuple
from types import SimpleNamespace

import joblib
import pandas as pd

from . import models  # 只是做类型提示用


# 模型文件路径：app/ml/obesity_model.joblib
MODEL_PATH = Path(__file__).resolve().parent / "ml" / "obesity_model.joblib"
_model = None


def load_model():
    """懒加载模型，只在第一次调用时从磁盘读 joblib。"""
    global _model
    if _model is None:
        _model = joblib.load(MODEL_PATH)
    return _model


def _get(record, name, default=None):
    """安全地从 Record 上取字段，没有就用默认值。"""
    return getattr(record, name, default)


def _map_record_to_features(record: models.Record) -> pd.DataFrame:
    """
    把我们系统里的 Record 映射成 obesity_level.csv 的特征格式。
    注意：有些特征在数据集中是数字 0/1，我们这里也必须给数字！
    """

    # Gender: 'Male' / 'Female'  —— 这列在数据集中是字符串
    g = str(_get(record, "gender", "F")).upper()
    gender = "Male" if g.startswith("M") else "Female"

    # Age / Height / Weight  —— 数值
    age = _get(record, "age", 25)
    height = float(_get(record, "height_m", 1.65) or 1.65)
    weight = float(_get(record, "weight_kg", 60.0) or 60.0)

    # family_history_with_overweight  —— 在 csv 里是 0/1，所以这里也用 0/1
    fh_raw = _get(record, "family_history", "N")
    fh = str(fh_raw).upper()
    family_history_over = 1 if fh.startswith("Y") else 0  # 数字，不是 'yes'/'no'

    # Water intake (ml) → CH2O  —— 原数据是浮点型，我们用 1/2/3 当作大致层级
    water_ml = _get(record, "water_ml", None)
    if water_ml is None:
        ch2o = 2.0
    elif water_ml < 1000:
        ch2o = 1.0
    elif water_ml < 2000:
        ch2o = 2.0
    else:
        ch2o = 3.0

    # 活动水平 → FAF (浮点数)
    level = str(_get(record, "activity_level", "low")).lower()
    if level == "low":
        faf = 0.0
    elif level in ("medium", "mid", "moderate"):
        faf = 1.0
    else:
        faf = 2.0  # high

    # 其他行为特征：
    #   FAVC, SMOKE, SCC = 0/1（数字）
    #   FCVC, NCP, CH2O, FAF, TUE = 浮点数
    #   CAEC, CALC, MTRANS = 字符串
    sample = {
        "Gender": gender,                       # str
        "Age": age,                             # float/int
        "Height": height,                       # float
        "Weight": weight,                       # float
        "family_history_with_overweight": family_history_over,  # int(0/1)
        "FAVC": 0,                              # int 默认：不经常吃高热量食物
        "FCVC": 2.0,                            # float 默认中等吃蔬菜
        "NCP": 3.0,                             # float 默认每天三餐
        "CAEC": "Sometimes",                    # str
        "SMOKE": 0,                             # int 默认不吸烟
        "CH2O": ch2o,                           # float
        "SCC": 0,                               # int 默认不刻意控卡
        "FAF": faf,                             # float
        "TUE": 2.0,                             # float 默认中等屏幕时间
        "CALC": "no",                           # str
        "MTRANS": "Public_Transportation",      # str
    }

    return pd.DataFrame([sample])


def predict_obesity_level(record: models.Record) -> Tuple[str, float]:
    """
    返回： (预测类别, 这一类别的概率 0~1)
    这是给 Assessment 等地方用的（传进来的是数据库里的 Record 实例）。
    """
    model = load_model()
    X = _map_record_to_features(record)

    proba = model.predict_proba(X)[0]   # numpy 数组
    classes = model.classes_

    # 取概率最大的类别
    idx = int(proba.argmax())
    label = str(classes[idx])
    score = float(proba[idx])

    return label, score


def predict_obesity_level_from_fields(
    age: int,
    gender: str,
    height_m: float,
    weight_kg: float,
    family_history: str,
    activity_level: str,
    water_ml: int,
) -> Tuple[str, float]:
    """
    给 /api/predict 用的版本：直接用简单字段，不需要真正的 Record。
    内部构造一个“假 Record”，复用上面的 predict_obesity_level。
    """
    fake_record = SimpleNamespace(
        age=age,
        gender=gender,
        height_m=height_m,
        weight_kg=weight_kg,
        family_history=family_history,
        activity_level=activity_level,
        water_ml=water_ml,
    )
    return predict_obesity_level(fake_record)
