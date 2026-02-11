# Feature Mapping (Dataset → Canonical Model Inputs → ELORA App)

This document ensures **feature consistency** across:
- raw training dataset (`app/data/obesity_level.csv`)
- ELORA web form / database
- the saved inference pipeline (`app/ml/obesity_model.joblib`)

## Canonical model input features (the pipeline expects these 7 columns)

| Canonical feature (model input) | Raw dataset column | ELORA app field (form/API/DB) | Type | Mapping / conversion rule (from `train_obesity_model.py`) |
|---|---|---|---|---|
| `age` | `Age` | `age` | numeric | `pd.to_numeric(df["Age"], errors="coerce")` |
| `gender` | `Gender` | `gender` | categorical | Normalised to **"Male"/"Female"** by `_normalize_gender()` (accepts `M/male/1/true/y` → Male; `F/female/0/false/n` → Female; else title-case fallback). |
| `height_m` | `Height` | `height_m` | numeric | `pd.to_numeric(df["Height"], errors="coerce")` (assumed meters). |
| `weight_kg` | `Weight` | `weight_kg` | numeric | `pd.to_numeric(df["Weight"], errors="coerce")` (kg). |
| `family_history` | `family_history_with_overweight` | `family_history` *(or legacy `family_hist`)* | categorical | Mapped to **"Y"/"N"** by `_normalize_yes_no_to_YN()` (yes/true/1 → Y; no/false/0 → N). Missing values are filled with mode (most frequent). |
| `activity_level` | `FAF` | `activity_level` | categorical | Binning rule `_map_faf_to_activity(x)`:<br> `v < 1.0 → "low"`; `1.0 ≤ v < 2.5 → "medium"`; `v ≥ 2.5 → "high"`; invalid → `"low"`. |
| `water_ml` | `CH2O` | `water_ml` | numeric | `pd.to_numeric(df["CH2O"], errors="coerce") * 1000.0` (CH2O treated as **litres**, converted to **ml**). |

## Preprocessing in the trained pipeline
- Numeric features: `SimpleImputer(strategy="median")` → `StandardScaler()`
- Categorical features: `SimpleImputer(strategy="most_frequent")` → `OneHotEncoder(handle_unknown="ignore")`

## Important consistency notes (App ↔ Model)
1) **Gender values must match** the model’s categories.
   - The training pipeline normalises gender to `"Male"` / `"Female"`.
   - If the web form stores `"M"` / `"F"` (or `"O"`), the inference layer should convert them to `"Male"` / `"Female"` (unknown categories are ignored due to `handle_unknown="ignore"`, but alignment is strongly recommended).

2) **Family history field name**
   - The model expects the column name: `family_history`.
   - If the DB/export uses `family_hist`, map/rename it to `family_history` before prediction.

## Evidence
- Training log saved at: `docs/evidence/ml_train_output.txt`
- Model artifact: `app/ml/obesity_model.joblib` (verified load OK)
