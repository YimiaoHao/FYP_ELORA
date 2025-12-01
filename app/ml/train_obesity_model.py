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


def load_and_prepare_data(csv_path: Path) -> Tuple[pd.DataFrame, pd.Series]:
    """从数据集中抽取出和前端一致的 7 个特征，并自动识别标签列名称。"""
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset not found: {csv_path}")

    df = pd.read_csv(csv_path)

    # ========= 1) 检查特征列是否存在 =========
    feature_cols = {
        "Age",
        "Gender",
        "Height",
        "Weight",
        "family_history_with_overweight",
        "FAF",   # physical activity frequency
        "CH2O",  # daily water consumption (litres)
    }
    missing_features = feature_cols - set(df.columns)
    if missing_features:
        raise ValueError(f"Missing expected feature columns in CSV: {missing_features}")

    # ========= 2) 自动识别标签列名称 =========
    label_candidates = [
        "NObeyesdad",    # Kaggle 原始列名
        "obesity_level", # 你自己常用的改名
        "Obesity_Level",
        "target",
        "label",
        "0be1dad",
    ]
    label_col = None
    for col in label_candidates:
        if col in df.columns:
            label_col = col
            break

    if label_col is None:
        # 实在找不到，就把所有列名打印出来方便你检查
        raise ValueError(
            f"Could not find label column. Tried {label_candidates}, "
            f"available columns: {list(df.columns)}"
        )

    # ========= 3) 映射到和 Web 表单一致的 7 个特征 =========
    df2 = pd.DataFrame()
    df2["age"] = df["Age"].astype(int)
    df2["gender"] = df["Gender"].astype(str)

    # Height/Weight 在数据集中本身就是米 / 公斤
    df2["height_m"] = df["Height"].astype(float)
    df2["weight_kg"] = df["Weight"].astype(float)

    # 家族史：yes/no -> Y/N
    df2["family_history"] = (
        df["family_history_with_overweight"]
        .astype(str)
        .str.strip()
        .str.lower()
        .map({"yes": "Y", "no": "N"})
    )

    # 活动水平：根据 FAF（每周运动的小时数）粗略分档
    def _map_faf_to_activity(x: float) -> str:
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

    # 每日喝水量：CH2O（升） * 1000 近似成 ml
    df2["water_ml"] = (df["CH2O"].astype(float) * 1000).astype(int)

    # ========= 4) 标签列 =========
    y = df[label_col].astype(str)

    return df2, y


def build_preprocessor(
    numeric_features, categorical_features
) -> ColumnTransformer:
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

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    return preprocessor


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


def compare_candidate_models(
    preprocessor: ColumnTransformer,
    X_train,
    X_test,
    y_train,
    y_test,
) -> Dict[str, Any]:
    """对几个候选模型做基线对比。"""
    candidates = {
        "logreg": LogisticRegression(max_iter=1000),
        "rf": RandomForestClassifier(
            n_estimators=200, random_state=42, n_jobs=-1
        ),
        "gb": GradientBoostingClassifier(random_state=42),
        "knn": KNeighborsClassifier(n_neighbors=9),
    }

    results = []

    print("=== Baseline model comparison ===")
    for name, clf in candidates.items():
        res = train_and_evaluate_model(
            name, clf, preprocessor, X_train, X_test, y_train, y_test
        )
        results.append(res)

    # 以 macro_f1 为主，accuracy 为次
    best = max(results, key=lambda r: (r["macro_f1"], r["accuracy"]))
    print(
        f"Best baseline model: {best['name']} "
        f"(Accuracy={best['accuracy']:.3f}, Macro-F1={best['macro_f1']:.3f})"
    )
    print()
    return best


def grid_search_random_forest(
    preprocessor: ColumnTransformer,
    X_train,
    X_test,
    y_train,
    y_test,
) -> Pipeline:
    """对随机森林做 GridSearchCV 调参，优化 macro F1。"""
    rf = RandomForestClassifier(random_state=42, n_jobs=-1)

    pipe = Pipeline(steps=[("preprocess", preprocessor), ("clf", rf)])

    param_grid = {
        "clf__n_estimators": [100, 200, 400],
        "clf__max_depth": [None, 10, 20],
        "clf__max_features": ["sqrt", "log2"],
        "clf__min_samples_split": [2, 5],
    }

    grid = GridSearchCV(
        pipe,
        param_grid=param_grid,
        scoring="f1_macro",
        cv=3,
        n_jobs=1,
        verbose=1,
    )

    print("=== Grid search for RandomForest (optimising macro F1) ===")
    grid.fit(X_train, y_train)

    print(f"Best params: {grid.best_params_}")
    print(f"Best CV macro F1: {grid.best_score_:.3f}")
    print()

    best_model: Pipeline = grid.best_estimator_

    # 在测试集上评估
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
        random_state=42,
        stratify=y,
    )

    preprocessor = build_preprocessor(numeric_features, categorical_features)

    # 1) 多模型基线对比
    best_baseline = compare_candidate_models(
        preprocessor, X_train, X_test, y_train, y_test
    )

    # 2) 针对 RandomForest 做 GridSearch 调参
    best_rf = grid_search_random_forest(
        preprocessor, X_train, X_test, y_train, y_test
    )

    # 这里选择 GridSearch 后的 RF 作为最终模型
    final_model = best_rf

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(final_model, MODEL_PATH)
    print(f"Model saved to: {MODEL_PATH}")


if __name__ == "__main__":
    main()
