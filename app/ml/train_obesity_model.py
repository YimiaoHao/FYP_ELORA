from pathlib import Path
from typing import Tuple, Dict, Any

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression


# 当前文件夹：.../app/ml
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR.parent / "data" / "obesity_level.csv"
MODEL_PATH = BASE_DIR / "obesity_model.joblib"

RANDOM_STATE = 42

# 想先跑快一点就改成 False（先得到模型文件再说）
RUN_GRID_SEARCH = True


def _normalize_gender(v) -> str:
    """把各种性别输入统一为 'Male' / 'Female'（与 ml_service 口径一致）"""
    if pd.isna(v):
        return "Female"
    s = str(v).strip().lower()
    if s in {"m", "male", "1", "true", "y"}:
        return "Male"
    if s in {"f", "female", "0", "false", "n"}:
        return "Female"
    # 兜底：首字母大写
    return str(v).strip().title() or "Female"


def _normalize_yes_no_to_YN(v):
    """把 0/1, yes/no, true/false 等统一成 'Y'/'N'，无法识别则返回 None"""
    if pd.isna(v):
        return None
    s = str(v).strip().lower()
    if s in {"y", "yes", "true", "1", "1.0"}:
        return "Y"
    if s in {"n", "no", "false", "0", "0.0"}:
        return "N"
    return None


def load_and_prepare_data(csv_path: Path) -> Tuple[pd.DataFrame, pd.Series]:
    """从数据集中抽取出和前端一致的 7 个特征，并自动识别标签列名称。"""
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset not found: {csv_path}")

    df = pd.read_csv(csv_path)

    # ========= 1) 检查原始数据集需要的列是否存在 =========
    required_cols = {
        "Age",
        "Gender",
        "Height",
        "Weight",
        "family_history_with_overweight",
        "FAF",
        "CH2O",
    }
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in CSV: {missing}")

    # ========= 2) 自动识别标签列名称 =========
    label_candidates = [
        "NObeyesdad",
        "obesity_level",
        "Obesity_Level",
        "target",
        "label",
        "0be1dad",
    ]
    label_col = next((c for c in label_candidates if c in df.columns), None)
    if label_col is None:
        raise ValueError(
            f"Could not find label column. Tried {label_candidates}, "
            f"available columns: {list(df.columns)}"
        )

    # ========= 3) 构造与 Web/API 一致的 7 个特征 =========
    df2 = pd.DataFrame()

    # Age：你的数据集里本来就大量是 float（小数），不要强转 Int64！
    df2["age"] = pd.to_numeric(df["Age"], errors="coerce")

    df2["gender"] = df["Gender"].apply(_normalize_gender)

    # Height/Weight：米 / 公斤（float）
    df2["height_m"] = pd.to_numeric(df["Height"], errors="coerce")
    df2["weight_kg"] = pd.to_numeric(df["Weight"], errors="coerce")

    # family_history：数据集是 0/1（也可能混杂 yes/no），统一成 Y/N，且保证不是全空
    fh_raw = df["family_history_with_overweight"].apply(_normalize_yes_no_to_YN)

    if fh_raw.isna().any():
        # 少量缺失用众数补齐（让模型真正用上这列）
        mode = fh_raw.dropna().mode()
        fill_value = mode.iloc[0] if not mode.empty else "N"
        fh_raw = fh_raw.fillna(fill_value)

    # 硬校验：避免你又把这列搞成全空
    if fh_raw.isna().all():
        raise ValueError("family_history became ALL-NaN after mapping. Check dataset values.")
    if fh_raw.nunique(dropna=True) < 2:
        # 不是错误也能训练，但说明这列没信息；这里给你直接提示
        print("[WARN] family_history has <2 unique values. It may not be informative.")

    df2["family_history"] = fh_raw

    # activity_level：由 FAF 分档
    def _map_faf_to_activity(x) -> str:
        try:
            v = float(x)
        except Exception:
            return "low"
        if v < 1.0:
            return "low"
        elif v < 2.5:
            return "medium"
        else:
            return "high"

    df2["activity_level"] = df["FAF"].apply(_map_faf_to_activity)

    # water_ml：CH2O（升）-> ml（保持数值即可，不强转 int 也没问题）
    df2["water_ml"] = pd.to_numeric(df["CH2O"], errors="coerce") * 1000.0

    # ========= 4) 标签列 =========
    y = df[label_col].astype(str).str.strip()

    # 修正脏标签
    y = y.replace({"0rmal_Weight": "Normal_Weight"})

    # 如果还有以 0 开头的奇怪标签，直接报错提醒你
    bad = y[y.str.contains(r"^0", regex=True)].unique()
    if len(bad) > 0:
        raise ValueError(f"Found suspicious labels: {bad}")

    return df2, y


def build_preprocessor(numeric_features, categorical_features) -> ColumnTransformer:
    """构造数值 + 类别特征的预处理流水线。"""
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )


def train_and_evaluate_model(
    name: str,
    classifier,
    preprocessor: ColumnTransformer,
    X_train,
    X_test,
    y_train,
    y_test,
) -> Dict[str, Any]:
    """训练单个模型并在测试集上评估。"""
    pipe = Pipeline(steps=[("preprocess", preprocessor), ("clf", classifier)])

    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro")

    print(f"=== {name} ===")
    print(f"Accuracy  : {acc:.3f}")
    print(f"Macro F1  : {macro_f1:.3f}")
    print()

    return {"name": name, "model": pipe, "accuracy": acc, "macro_f1": macro_f1}


def compare_candidate_models(preprocessor, X_train, X_test, y_train, y_test) -> Dict[str, Any]:
    """对几个候选模型做基线对比。"""
    candidates = {
        "logreg": LogisticRegression(max_iter=1000),
        "rf": RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1),
        "gb": GradientBoostingClassifier(random_state=RANDOM_STATE),
        "knn": KNeighborsClassifier(n_neighbors=9),
    }

    results = []
    print("=== Baseline model comparison ===")
    for name, clf in candidates.items():
        results.append(train_and_evaluate_model(name, clf, preprocessor, X_train, X_test, y_train, y_test))

    best = max(results, key=lambda r: (r["macro_f1"], r["accuracy"]))
    print(
        f"Best baseline model: {best['name']} "
        f"(Accuracy={best['accuracy']:.3f}, Macro-F1={best['macro_f1']:.3f})"
    )
    print()
    return best


def grid_search_random_forest(preprocessor, X_train, X_test, y_train, y_test) -> Pipeline:
    """对随机森林做 GridSearchCV 调参，优化 macro F1。"""
    rf = RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1)
    pipe = Pipeline(steps=[("preprocess", preprocessor), ("clf", rf)])

    # 这里稍微收敛一下搜索空间，避免你觉得“跑不到头”
    param_grid = {
        "clf__n_estimators": [200, 400],
        "clf__max_depth": [None, 10, 20],
        "clf__max_features": ["sqrt", "log2"],
        "clf__min_samples_split": [2, 5],
    }

    grid = GridSearchCV(
        pipe,
        param_grid=param_grid,
        scoring="f1_macro",
        cv=3,
        n_jobs=1,   # Windows 上更稳；想快可改 -1
        verbose=1,
    )

    print("=== Grid search for RandomForest (optimising macro F1) ===")
    grid.fit(X_train, y_train)

    print(f"Best params: {grid.best_params_}")
    print(f"Best CV macro F1: {grid.best_score_:.3f}")
    print()

    best_model: Pipeline = grid.best_estimator_

    y_pred = best_model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro")

    print("=== RandomForest (after grid search) on test set ===")
    print(f"Accuracy : {acc:.3f}")
    print(f"Macro F1 : {macro_f1:.3f}")
    print()
    print("=== Classification report on test set ===")
    print(classification_report(y_test, y_pred))
    print()

    return best_model


def main():
    print(f"Loading data from: {DATA_PATH}")
    X, y = load_and_prepare_data(DATA_PATH)

    numeric_features = ["age", "height_m", "weight_kg", "water_ml"]
    categorical_features = ["gender", "family_history", "activity_level"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    preprocessor = build_preprocessor(numeric_features, categorical_features)

    # 1) 多模型基线对比
    _ = compare_candidate_models(preprocessor, X_train, X_test, y_train, y_test)

    # 2) GridSearch（可选）
    if RUN_GRID_SEARCH:
        final_model = grid_search_random_forest(preprocessor, X_train, X_test, y_train, y_test)
    else:
        # 不跑 GridSearch 时，就用一个还不错的 RF 直接训练
        final_model = Pipeline(
            steps=[
                ("preprocess", preprocessor),
                ("clf", RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1)),
            ]
        )
        final_model.fit(X_train, y_train)
        
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(final_model, MODEL_PATH)
    print(f"Model saved to: {MODEL_PATH}")


if __name__ == "__main__":
    main()
