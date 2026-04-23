# ELORA

**ELORA (Explainable Lifestyle & Obesity Risk Analytics)** is a local-first web application for recording daily health indicators and generating an explainable obesity-related risk assessment. It combines a trained machine learning model with a transparent rules engine, stores records locally in SQLite, and presents results through a simple browser interface built with FastAPI, Jinja2, and ECharts.

This project was developed as a final-year undergraduate software development project. The system is intended for **educational and demonstration purposes only**. It is **not** a medical device and does **not** provide medical advice.

---

## 1. Project Summary

Many obesity-related tools either focus only on BMI or only on black-box predictions. ELORA was designed to make the result easier to understand by combining:

- **machine learning prediction** from a trained multi-class obesity-level classifier,
- **rules-based scoring** derived from BMI and simple lifestyle factors,
- **local-first data storage** so the user keeps control of their own records,
- **clear interface pages** for recording, review, trends, assessment, and privacy.

The current system supports the following workflow:

1. Enter a daily record on **Record Today**.
2. Save the record into the local SQLite database.
3. Review, export, or delete records on **History**.
4. View recent weight and BMI trends on **Trends**.
5. See a combined machine-learning + rules-based result on **Assessment**.
6. Review data handling and local-storage policy on **Privacy**.

---

## 2. Main Features

### 2.1 Daily record entry
The user can submit one daily health record containing:

- date
- age
- gender
- height (m)
- weight (kg)
- family history of overweight/obesity
- activity level
- water intake (ml)

Input validation is applied both at the form level and through backend schema checks.

### 2.2 Local SQLite storage
All records are stored in a local SQLite database file (`elora.db`). No account system, remote user profile, or cloud database is required for the core demo.

### 2.3 History management
The History page allows the user to:

- view saved records,
- open a single detailed record,
- delete one record,
- clear the full local history,
- export records to CSV.

### 2.4 Trend visualisation
The Trends page shows recent **weight** and **BMI** changes using Apache ECharts. The current implementation supports viewing recent subsets such as the last 7 or last 30 records.

### 2.5 Explainable assessment
The Assessment page presents:

- **Final Hybrid Risk Score**
- **Latest Record** summary
- **Rules-based assessment**
- **Model-based assessment**
- **Triggered rules**
- **Personalised tips**

The rules-based section uses a **7-level obesity classification**:

1. Insufficient Weight
2. Normal Weight
3. Overweight Level I
4. Overweight Level II
5. Obesity Type I
6. Obesity Type II
7. Obesity Type III

The current category is highlighted in the interface so the user can see where they belong within the full classification scale.

### 2.6 Privacy-by-design page
A dedicated Privacy page explains:

- what data is stored,
- where the data is stored,
- how users can export or delete it,
- the non-clinical nature of the prototype.

---

## 3. Hybrid Risk Logic

ELORA does not rely on one signal alone. Instead, it fuses:

- a **machine learning component** from the obesity-level prediction model,
- a **rules component** generated from BMI category and simple lifestyle triggers.

### 3.1 Rules-based component
The rules engine first classifies BMI into one of seven categories and assigns a base score. It then adjusts the score according to simple conditions such as:

- family history reported,
- low activity level,
- low water intake.

The final rules score is bounded to a **0–100** range.

### 3.2 Model-based component
The machine learning pipeline predicts an obesity level label and returns a confidence score. That confidence is then mapped into a risk contribution used in the final fusion.

### 3.3 Final fusion formula
When model inference is available:

```text
Final Risk = 0.7 × Model Component + 0.3 × Rules Score
```

When model inference fails, the system falls back to the rules-based result so the application can still function.

### 3.4 Risk tier output
The fused result is converted to a percentage and displayed as a risk tier on the Assessment page.

---

## 4. Machine Learning Component

### 4.1 Input schema used by the web app
Although the original obesity dataset contains many variables, this deployed prototype uses a simplified **7-feature** input schema:

- `age`
- `gender`
- `height_m`
- `weight_kg`
- `family_history`
- `activity_level`
- `water_ml`

This keeps the interface easier to use while still preserving useful predictive information.

### 4.2 Trained model artifact
The trained model used by the running app is stored at:

```text
app/ml/obesity_model.joblib
```

### 4.3 Reported benchmark information
The project contains benchmark and final metrics files:

- `app/ml/benchmark_results.csv`
- `app/ml/final_model_metrics.json`

The current saved final metrics indicate that the deployed optimised model is:

- **Optimised Gradient Boosting**

with approximately:

- **Accuracy:** 0.8964
- **Macro-F1:** 0.8854

These values are used in the interface as offline benchmark references.

### 4.4 Retraining
A retraining script is included:

```text
app/ml/train_obesity_model.py
```

This allows the obesity-level model to be retrained locally from the dataset in `app/data/`.

---

## 5. Technology Stack

### Backend
- Python
- FastAPI
- SQLAlchemy
- Pydantic
- Uvicorn

### Frontend
- Jinja2 templates
- HTML
- CSS
- JavaScript
- Apache ECharts

### Data / ML
- SQLite
- pandas
- numpy
- scikit-learn
- joblib

### Development environment
- Windows local environment
- Visual Studio Code
- Git (optional for version control)

---

## 6. Project Structure

```text
ELORA/
├── app/
│   ├── data/
│   │   └── obesity_level.csv              # dataset used for training / reference
│   ├── ml/
│   │   ├── obesity_model.joblib           # trained model used by app
│   │   ├── benchmark_results.csv          # baseline comparison results
│   │   ├── final_model_metrics.json       # final selected model metrics
│   │   └── train_obesity_model.py         # local retraining script
│   ├── static/
│   │   ├── css/style.css                  # application styles
│   │   └── js/main.js                     # frontend script(s)
│   ├── templates/
│   │   ├── base.html
│   │   ├── home.html
│   │   ├── record_today.html
│   │   ├── history.html
│   │   ├── history_detail.html
│   │   ├── trends.html
│   │   ├── assessment.html
│   │   └── privacy.html
│   ├── database.py                        # DB engine / session setup
│   ├── main.py                            # FastAPI routes and app entry logic
│   ├── ml_service.py                      # ML loading and prediction helpers
│   ├── models.py                          # SQLAlchemy Record model
│   ├── rules.py                           # rules engine and tips logic
│   ├── schemas.py                         # Pydantic request/response schemas
│   └── __init__.py
├── docs/
│   ├── feature_mapping.md
│   ├── hybrid_risk_score.md
│   ├── model_selection.md
│   └── privacy_by_design.md
├── elora.db                               # local SQLite database
├── requirements.txt
├── run_local.bat                          # Windows start script
└── README.md
```

---

## 7. Data Model

The main persisted entity is a `Record` with the following fields:

- `id`
- `date`
- `age`
- `gender`
- `height_m`
- `weight_kg`
- `family_history`
- `activity_level`
- `water_ml`
- `bmi`
- `created_at`

This structure is defined in `app/models.py`.

---

## 8. Validation Rules

The backend validates the core form input ranges. Current important limits include:

- **Age:** 5 to 100
- **Height:** greater than 0 and up to 2.5 m
- **Weight:** greater than 0 and up to 300 kg
- **Water intake:** 0 to 10000 ml
- **Activity level:** low / medium / high
- **Family history:** Y / N style input

Pydantic schemas are also used for the API request and response structure.

---

## 9. Available Routes

### Browser pages
- `/` — Home
- `/record-today` — daily record form
- `/history` — record list
- `/history/view/{record_id}` — single record detail
- `/trends` — recent weight/BMI charts
- `/assessment` — explainable assessment page
- `/privacy` — privacy & data handling page

### Actions
- `/history/delete/{record_id}` — delete a single record
- `/history/clear` — clear all history
- `/export` — export records to CSV

### API
- `/api/predict` — stateless prediction endpoint

---

## 10. Installation and Running

## 10.1 Requirements
Install Python 3.10+ (or a compatible Python 3.x environment) on Windows.

## 10.2 Recommended quick start
The simplest local way to run the project on Windows is:

```bat
run_local.bat
```

This script will:

1. create a virtual environment if needed,
2. activate it,
3. install dependencies from `requirements.txt`,
4. start the FastAPI app with Uvicorn.

## 10.3 Manual setup
If you prefer to run it manually:

```bash
python -m venv venv
venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
uvicorn app.main:app --reload
```

After the server starts, open:

- **App:** `http://127.0.0.1:8000`
- **API docs:** `http://127.0.0.1:8000/docs`

---

## 11. How to Use the System

### Record Today
Open `/record-today`, enter a valid daily record, and save it. If the same date is reused, the project can treat it as an updated daily record depending on the current application flow.

### History
Use `/history` to review existing records, inspect one record in detail, export CSV, delete an individual row, or clear the dataset.

### Trends
Use `/trends` to inspect recent weight and BMI changes. This page is intended to help the user observe recent direction rather than provide clinical analysis.

### Assessment
Use `/assessment` to view:

- final hybrid percentage,
- rules-based category,
- model-based prediction,
- triggered rules,
- personalised tips.

### Privacy
Use `/privacy` to understand the local-first design and data handling scope.

---

## 12. Notes on Logging and Local Files

The project creates local log files during runtime, for example under a `logs/` directory. These are mainly for debugging and are not required for the final delivered system to function.

The local database file is:

```text
elora.db
```

If you delete this file, you will lose the currently stored local records unless you have exported them.

---

## 13. Known Prototype Limitations

- This is an academic prototype, not a clinical system.
- The model uses a simplified 7-feature schema rather than the full original dataset feature set.
- Outputs are best understood as **lifestyle-oriented reference information**, not diagnosis.
- The local interface currently focuses on a single-user demo scenario rather than multi-user deployment.
- The quality of the final result depends on correct user input.

---

## 14. Future Improvement Ideas

Possible next improvements include:

- stronger form feedback and input guidance,
- more polished responsive layout across all pages,
- clearer interpretation text for each obesity-level class,
- more configurable trend ranges,
- optional deployment packaging for easier demonstration,
- additional model comparison or calibration work.

---

## 15. Academic Use Disclaimer

ELORA was developed as a final-year project to demonstrate:

- full-stack web development,
- local database handling,
- explainable rules integration,
- machine learning model deployment,
- privacy-aware prototype design.

It is intended for **teaching, learning, demonstration, and project assessment** only.

---

## 16. Author

**Yimiao Hao**  
BSc (Hons) Software Development  
South East Technological University (SETU)

