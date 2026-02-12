from pathlib import Path
from typing import List, Tuple, Dict, Any

import joblib
import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR.parent / "data" / "obesity_level.csv"
MODEL_PATH = BASE_DIR / "obesity_model.joblib"

RANDOM_STATE = 42

# Whether to run GridSearch.
RUN_GRID_SEARCH = False


def _normalize_gender(v) -> str:

    if pd.isna(v):
        return "Female"
    s = str(v).strip().lower()
    if s in {"m", "male", "1", "true", "y"}:
        return "Male"
    if s in {"f", "female", "0", "false", "n"}:
        return "Female"
    return str(v).strip().title() or "Female"


def _normalize_yes_no_to_YN(v):

    if pd.isna(v):
        return None
    s = str(v).strip().lower()
    if s in {"y", "yes", "true", "1", "1.0"}:
        return "Y"
    if s in {"n", "no", "false", "0", "0.0"}:
        return "N"
    return None


def load_and_prepare_data(csv_path: Path) -> Tuple[pd.DataFrame, pd.Series]:

    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset not found: {csv_path}")

    df = pd.read_csv(csv_path)

    required_cols = {"Age", "Gender", "Height", "Weight", "family_history_with_overweight", "FAF", "CH2O"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in CSV: {missing}")

    label_candidates = ["NObeyesdad", "obesity_level", "Obesity_Level", "target", "label", "0be1dad"]
    label_col = next((c for c in label_candidates if c in df.columns), None)
    if label_col is None:
        raise ValueError(
            f"Could not find label column. Tried {label_candidates}, "
            f"available columns: {list(df.columns)}"
        )

    df2 = pd.DataFrame()

    df2["age"] = pd.to_numeric(df["Age"], errors="coerce")
    df2["gender"] = df["Gender"].apply(_normalize_gender)

    df2["height_m"] = pd.to_numeric(df["Height"], errors="coerce")
    df2["weight_kg"] = pd.to_numeric(df["Weight"], errors="coerce")

    fh_raw = df["family_history_with_overweight"].apply(_normalize_yes_no_to_YN)
    if fh_raw.isna().any():
        mode = fh_raw.dropna().mode()
        fill_value = mode.iloc[0] if not mode.empty else "N"
        fh_raw = fh_raw.fillna(fill_value)

    if fh_raw.isna().all():
        raise ValueError("family_history became ALL-NaN after mapping. Check dataset values.")
    if fh_raw.nunique(dropna=True) < 2:
        print("[WARN] family_history has <2 unique values. It may not be informative.")

    df2["family_history"] = fh_raw

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
    df2["water_ml"] = pd.to_numeric(df["CH2O"], errors="coerce") * 1000.0

    y = df[label_col].astype(str).str.strip()
    y = y.replace({"0rmal_Weight": "Normal_Weight"})

    bad = y[y.str.contains(r"^0", regex=True)].unique()
    if len(bad) > 0:
        raise ValueError(f"Found suspicious labels: {bad}")

    return df2, y


def build_preprocessor(numeric_features, categorical_features) -> ColumnTransformer:

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


def confidence_stats(pipe: Pipeline, X_test) -> Dict[str, float]:

    if not hasattr(pipe, "predict_proba"):
        return {
            "conf_mean": float("nan"),
            "conf_p50": float("nan"),
            "conf_min": float("nan"),
            "conf_max": float("nan"),
        }

    proba = pipe.predict_proba(X_test)
    top = np.max(proba, axis=1)
    return {
        "conf_mean": float(np.mean(top)),
        "conf_p50": float(np.median(top)),
        "conf_min": float(np.min(top)),
        "conf_max": float(np.max(top)),
    }


def train_and_evaluate_model(
    name: str,
    classifier,
    preprocessor: ColumnTransformer,
    X_train,
    X_test,
    y_train,
    y_test,
) -> Dict[str, Any]:

    pipe = Pipeline(steps=[("preprocess", preprocessor), ("clf", classifier)])

    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro")
    conf = confidence_stats(pipe, X_test)

    print(f"=== {name} ===")
    print(f"Accuracy    : {acc:.3f}")
    print(f"Macro F1    : {macro_f1:.3f}")
    print(f"Conf(mean)  : {conf['conf_mean']:.3f}")
    print(f"Conf(p50)   : {conf['conf_p50']:.3f}")
    print()

    return {
        "name": name,
        "model": pipe,
        "accuracy": acc,
        "macro_f1": macro_f1,
        **conf,
    }


def compare_candidate_models(preprocessor, X_train, X_test, y_train, y_test) -> Dict[str, Any]:

    candidates = {
        "logreg": LogisticRegression(max_iter=1000),
        "knn": KNeighborsClassifier(n_neighbors=9),
        "gb": GradientBoostingClassifier(random_state=RANDOM_STATE),
        "rf": RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1),
        "dt": DecisionTreeClassifier(random_state=RANDOM_STATE),
        "svm_rbf": SVC(kernel="rbf", C=10.0, gamma="scale", probability=True, random_state=RANDOM_STATE),
        "mlp": MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=600, random_state=RANDOM_STATE),
    }

    results: List[Dict[str, Any]] = []
    print("=== Baseline model comparison (same split & same preprocessing) ===")

    for name, clf in candidates.items():
        results.append(train_and_evaluate_model(name, clf, preprocessor, X_train, X_test, y_train, y_test))

    results_sorted = sorted(results, key=lambda r: (r["macro_f1"], r["accuracy"], r["conf_mean"]), reverse=True)

    print("=== Summary (sorted) ===")
    for r in results_sorted:
        print(
            f"{r['name']:8s} | MacroF1={r['macro_f1']:.3f} | Acc={r['accuracy']:.3f} | "
            f"ConfMean={r['conf_mean']:.3f} | ConfP50={r['conf_p50']:.3f}"
        )
    print()

    out_csv = BASE_DIR / "benchmark_results.csv"
    df_out = pd.DataFrame(
        [
            {
                "model": r["name"],
                "macro_f1": r["macro_f1"],
                "accuracy": r["accuracy"],
                "conf_mean": r["conf_mean"],
                "conf_p50": r["conf_p50"],
                "conf_min": r["conf_min"],
                "conf_max": r["conf_max"],
            }
            for r in results_sorted
        ]
    )
    df_out.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"[OK] Benchmark CSV saved to: {out_csv}")

    best = results_sorted[0]
    print(
        f"Best baseline model: {best['name']} "
        f"(Macro-F1={best['macro_f1']:.3f}, Acc={best['accuracy']:.3f}, ConfMean={best['conf_mean']:.3f})"
    )
    print()

    return best


def grid_search_random_forest(preprocessor, X_train, X_test, y_train, y_test) -> Pipeline:
    rf = RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1)
    pipe = Pipeline(steps=[("preprocess", preprocessor), ("clf", rf)])

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
        n_jobs=1,
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

    # Multi-model baseline comparison

    best = compare_candidate_models(preprocessor, X_train, X_test, y_train, y_test)

    # 2) GridSearch
    if RUN_GRID_SEARCH:
        final_model = grid_search_random_forest(preprocessor, X_train, X_test, y_train, y_test)
    else:
        # When GridSearch is not running, the benchmark optimal model is directly used as the final model (satisfying "select the optimal model")
        final_model = best["model"]

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(final_model, MODEL_PATH)
    print(f"Model saved to: {MODEL_PATH}")


if __name__ == "__main__":
    main()