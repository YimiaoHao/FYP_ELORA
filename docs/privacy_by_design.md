# Privacy by Design (Local-first)

ELORA is designed as a **local-first** obesity-risk prototype. The goal is to provide lifestyle reference and visualisation without uploading personal health data to third-party services.

## Data Minimisation
- The app only stores a small set of daily inputs required for the prototype:
  `date, age, gender, height_m, weight_kg, family_history, activity_level, water_ml, bmi`.
- No names, emails, addresses, device identifiers, or precise location data are collected.

## Local Processing & Storage
- All records are stored in a **local SQLite database** on the user's machine.
- ML inference is performed locally (via `obesity_model.joblib`) unless the optional cloud mode is enabled.
- The application does not require user accounts or authentication for the local demo.

## User Control & Transparency
- Users can:
  - View the last records in **History**
  - **Clear history** (delete all local records)
  - **Export CSV** for personal backup/analysis
- A dedicated **Privacy** page explains what is stored and where.

## Security Considerations (Prototype Scope)
- The project follows “least data” by default (no external database, no telemetry).
- Repository hygiene:
  - Virtual environment files are excluded (via `.gitignore`)
  - Local database files should not be committed to Git
- This is an educational prototype; it is **not medical advice** and not a clinical system.

## Cloud Extension (If Enabled)
If the cloud extension is enabled, the cloud API is implemented as a **stateless inference service**:
- It receives a single JSON request and returns a prediction response.
- It does **not** store records or maintain user profiles.
- The local app can fall back to local inference if the cloud call fails.
