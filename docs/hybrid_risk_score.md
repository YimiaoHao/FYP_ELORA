# Hybrid Risk Score (0.7 ML + 0.3 Rules)

## Motivation
ELORA combines:
- **Rule-based BMI assessment** (transparent, easy to explain)
- **ML prediction confidence** (data-driven signal)

This produces a score that is both interpretable (rules) and adaptive (ML).

## Definitions
- `Rules_Score` is a 0–100 score from BMI-based rules.
- `Model_Prob` is the ML confidence (0–1) for the predicted class.

Rules score is normalised:
- `Rules_Norm = Rules_Score / 100`

## Final Formula
If ML is available:
- `Final_Risk = 0.7 × Model_Prob + 0.3 × Rules_Norm`

If ML fails (fallback):
- `Final_Risk = Rules_Norm`

Displayed as percentage:
- `Final_Risk_Percent = int(Final_Risk * 100)`

## Why 0.7 / 0.3?
- ML is given higher weight (0.7) to reflect richer learned patterns.
- Rules retain meaningful influence (0.3) to keep the score explainable and stable.
- The system remains robust: if cloud/local ML is unavailable, rules still produce a result.

## Example
If `Model_Prob = 0.60` and `Rules_Score = 35`:
- `Rules_Norm = 0.35`
- `Final_Risk = 0.7*0.60 + 0.3*0.35 = 0.525 → 52%`
