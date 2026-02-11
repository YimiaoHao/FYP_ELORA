# Feature Mapping (Dataset → App)

This project trains a classifier using the Obesity dataset and deploys a simplified **7-feature** input schema in the web app to keep the demo practical and privacy-friendly.

## App Input Features (7)
The web app uses the following 7 features:
1. `age`
2. `gender`
3. `height_m`
4. `weight_kg`
5. `family_history`
6. `activity_level`
7. `water_ml`

## Dataset → App Mapping Table

| Dataset column | Type | App feature | Transformation |
|---|---|---|---|
| `Age` | numeric | `age` | `to_numeric` (keep float/int) |
| `Gender` | categorical | `gender` | normalise to **Male/Female** |
| `Height` | numeric | `height_m` | `to_numeric` (meters) |
| `Weight` | numeric | `weight_kg` | `to_numeric` (kg) |
| `family_history_with_overweight` | categorical/0-1 | `family_history` | map to **Y/N** |
| `FAF` | numeric | `activity_level` | binning: `<1.0=low`, `1.0~2.5=medium`, `>=2.5=high` |
| `CH2O` | numeric | `water_ml` | liters → ml: `CH2O * 1000` |

## Label Column
Training code automatically detects the label column using common names such as:
`NObeyesdad`, `obesity_level`, `target`, etc.

## Why This Mapping?
- Keeps the UI simple for daily input.
- Keeps the demo local-first and privacy-friendly.
- Still preserves meaningful signals (body size, activity, water, family history).
