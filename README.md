# ELORA - Explainable Lifestyle & Obesity Risk Analytics

**ELORA** is a local-first obesity risk assessment prototype designed to provide explainable lifestyle insights. The system integrates machine learning predictions with rule-based logic to calculate a hybrid risk score, ensuring robust analysis while maintaining strict user data privacy through local storage and processing.

> **Alpha Release - Iteration One**
> Final Year Project - BSc (Hons) in Software Development, SETU.

## Project Overview

Standard BMI metrics often fail to capture critical body composition and lifestyle context. ELORA addresses this limitation by fusing a **Random Forest Classifier** with a deterministic rules engine. The system captures daily biometric and lifestyle data, processes it locally, and provides actionable insights without transmitting sensitive user data to external servers.

**Core Algorithm:**
$$Risk = 0.7 \times P(Model) + 0.3 \times R(Rules)$$

## Key Features

* **Hybrid Risk Assessment:** Implements a fusion algorithm combining the probabilistic output ($P$) of a machine learning model with a normalized rule-based score ($R$) to ensure assessment reliability.
* **Interactive Trend Visualization:** Integrates the Apache ECharts library to render dynamic, dual-axis charts for tracking longitudinal weight and BMI trends.
* **Local-First Data Architecture:** Utilizes SQLite for persistent local storage, ensuring full user data sovereignty with support for CSV export and complete data deletion.
* **Data Integrity and Validation:** Employs Pydantic models within the backend to enforce strict type checking and range validation on all user inputs.
* **Stateless Inference API:** Provides a RESTful API endpoint for model inference that operates statelessly, adhering to privacy-by-design principles.

## Technology Stack

* **Backend Framework:** Python 3.x, FastAPI
* **Database & ORM:** SQLite, SQLAlchemy
* **Machine Learning:** Scikit-learn, Joblib, Pandas
* **Frontend & Visualization:** Jinja2 Templates, HTML5/CSS3, Apache ECharts
* **Development Tools:** Visual Studio Code, Git

## Installation and Execution

### 1. Prerequisites
Ensure Python 3.8 or higher is installed on the local machine.

### 2. Clone Repository
```bash
git clone https://github.com/YimiaoHao/FYP_ELORA.git
cd FYP_ELORA
````

### 3\. Environment Setup

It is recommended to run the application within a virtual environment to manage dependencies.

```bash
# Create virtual environment (Windows)
python -m venv venv
.\venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 4\. Run Application

Execute the following command to start the FastAPI server in development mode:

```bash
uvicorn app.main:app --reload
```

Once initialized, the application is accessible at:

  * **Web Interface:** [http://127.0.0.1:8000](https://www.google.com/search?q=http://127.0.0.1:8000)
  * **API Documentation:** [http://127.0.0.1:8000/docs](https://www.google.com/search?q=http://127.0.0.1:8000/docs)

### 5\. Model Retraining (Optional)

To retrain the machine learning model based on the dataset provided in `app/data/`:

```bash
python -m app.ml.train_obesity_model
```

## System Screenshots

| **Hybrid Risk Assessment** | **Trend Visualization** |
|:---:|:---:|
|  |  |
| *Visual representation of the fused risk score ($0.7P + 0.3R$)* | *Interactive dual-axis charting using ECharts* |

| **Data Entry Interface** | **System Architecture** |
|:---:|:---:|
|  |  |

## Project Structure

```text
ELORA/
├── app/
│   ├── ml/                 # Machine Learning core (Training scripts & serialized .joblib models)
│   ├── static/             # Static assets (CSS/JS)
│   ├── templates/          # Jinja2 frontend templates
│   ├── database.py         # Database connection and session management
│   ├── main.py             # Application entry point and route definitions
│   ├── models.py           # SQLAlchemy data models
│   └── schemas.py          # Pydantic validation schemas
├── docs/                   # Project Documentation (Functional, Research, & Design Specs)
├── screenshots/            # Evidence artifacts and UI screenshots
├── elora.db                # Local SQLite database (Auto-generated)
└── requirements.txt        # Python dependency list
```

## License and Disclaimer

This project is an academic prototype intended for educational and research purposes only. It does not constitute medical advice.

Copyright © 2025 Yimiao Hao. All Rights Reserved.

```
```
