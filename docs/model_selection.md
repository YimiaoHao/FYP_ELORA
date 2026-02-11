# Model Selection

## Goal
Select a multi-class model that performs well on obesity-level classification while staying practical for a local-first demo.

## Baseline Comparison
Several classifiers were compared using a consistent preprocessing pipeline:
- Logistic Regression
- K-Nearest Neighbours (KNN)
- Gradient Boosting (GB)
- Random Forest (RF)

## Metric Choice
**Macro-F1** is used as the primary metric because it treats each class equally and is more informative than accuracy when class sizes differ.

## Final Choice: Random Forest
Random Forest was selected because:
- It achieved strong Macro-F1 and accuracy in baseline testing.
- It is robust for mixed numeric + categorical inputs after one-hot encoding.
- It is stable and easy to deploy using a scikit-learn Pipeline (`preprocess + classifier`).

## Hyperparameter Tuning
A `GridSearchCV` was used to tune key RF parameters (e.g., depth, features, split rules, estimators) with:
- scoring = `f1_macro`
- cross-validation (3-fold)

## Output Artifact
The final trained pipeline is exported as:
- `app/ml/obesity_model.joblib`

## Limitations (Prototype)
- The 7-feature schema is a simplified subset of the dataset.
- Results are for educational demonstration, not medical diagnosis.
