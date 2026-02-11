# Model Selection and Training Evidence

## Objective
ELORA predicts obesity risk as a **multi-class classification** problem using the Obesity Risk dataset.
To reduce the impact of class imbalance across obesity levels, the main metric reported is **Macro-F1**.

## Baseline Model Comparison
I compared multiple baseline models using the same data split and preprocessing pipeline:

| Model | Accuracy | Macro-F1 |
|---|---:|---:|
| Logistic Regression | 0.854 | 0.837 |
| Random Forest (baseline) | 0.882 | 0.869 |
| Gradient Boosting | **0.893** | **0.881** |
| KNN | 0.840 | 0.822 |

**Result:** Gradient Boosting achieved the best baseline Macro-F1 (0.881). Random Forest was close and is easier to deploy and integrate into a local-first web prototype.

## Hyperparameter Tuning (Random Forest)
Random Forest was tuned using **GridSearchCV (3-fold)** to optimise **Macro-F1**:

- Total candidates: 24 (72 fits)
- Best parameters:
  - `max_depth`: None
  - `max_features`: "sqrt"
  - `min_samples_split`: 5
  - `n_estimators`: 200
- Best CV Macro-F1: **0.865**

## Test Set Performance (After Grid Search)
On the held-out test set, the tuned Random Forest achieved:

- **Accuracy:** 0.887
- **Macro-F1:** 0.875

The classification report shows strong performance across most classes (e.g., Obesity Type II and III), with weaker performance mainly on borderline “Overweight” categories, which is expected due to overlapping feature distributions.

## Reproducibility Artifact
The trained model is saved as a scikit-learn pipeline:

- File: `app/ml/obesity_model.joblib`
- Verified load: `sklearn.pipeline.Pipeline` (`loaded ok`)

Training logs are stored in: `docs/evidence/ml_train_output.txt`.
