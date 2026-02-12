from fastapi import FastAPI, Query, Request, Form, Depends
from fastapi.responses import HTMLResponse, RedirectResponse, StreamingResponse
from starlette import status
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pathlib import Path
from sqlalchemy.orm import Session
import datetime as dt
import io
import csv

# Input/output model for /api/predict
from .schemas import RecordInput, PredictionResponse

from .ml_service import predict_obesity_level, predict_obesity_level_from_fields
from .rules import classify_bmi, generate_tips
from .database import Base, engine, get_db
from . import models


BASE_DIR = Path(__file__).resolve().parent

app = FastAPI(title="ELORA - Obesity Risk Prototype")

Base.metadata.create_all(bind=engine)

app.mount(
    "/static",
    StaticFiles(directory=BASE_DIR / "static"),
    name="static",
)

templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))


@app.post("/api/predict", response_model=PredictionResponse)
async def api_predict(payload: RecordInput):

    # Calculate BMI + BMI Score
    bmi = round(payload.weight_kg / (payload.height_m ** 2), 1)
    bmi_category, risk_score = classify_bmi(bmi)

    # 2) Call the model prediction of "field version"
    model_label = None
    model_confidence = None
    try:
        model_label, model_confidence = predict_obesity_level_from_fields(
            age=payload.age,
            gender=payload.gender,
            height_m=payload.height_m,
            weight_kg=payload.weight_kg,
            family_history=payload.family_history,
            activity_level=payload.activity_level,
            water_ml=payload.water_ml,
        )
    except Exception as e:
        print("API /api/predict ML error:", e)

    # Return a unified response model
    return PredictionResponse(
        bmi=bmi,
        bmi_category=bmi_category,
        risk_score=risk_score,
        model_label=model_label,
        model_confidence=model_confidence,
        tips=None,
    )


@app.get("/", response_class=HTMLResponse)
async def read_home(request: Request):

    return templates.TemplateResponse(
        "home.html",
        {
            "request": request,
            "title": "ELORA Home",
        },
    )

@app.get("/privacy", response_class=HTMLResponse)
async def privacy(request: Request):

    return templates.TemplateResponse(
        "privacy.html",
        {
            "request": request,
            "title": "Privacy & Data Handling",
        },
    )


@app.get("/record-today", response_class=HTMLResponse)
async def record_today_form(request: Request):

    return templates.TemplateResponse(
        "record_today.html",
        {
            "request": request,
            "title": "Record Today",
            "submitted": False,
        },
    )


@app.post("/record-today", response_class=HTMLResponse)
async def record_today_submit(
    request: Request,
    date: str = Form(...),
    age: int = Form(...),
    gender: str = Form(...),
    height_m: float = Form(...),
    weight_kg: float = Form(...),
    family_history: str = Form(...),
    activity_level: str = Form(...),
    water_ml: int = Form(...),
    db: Session = Depends(get_db),
):

    bmi = None
    if height_m > 0:
        bmi = round(weight_kg / (height_m ** 2), 1)

    # Unified gender storage caliber: Male/Female/Other (compatible with old M/F/O)
    gender_map = {"M": "Male", "F": "Female", "O": "Other"}
    gender = gender_map.get(gender, gender)

    # Write to database
    record = models.Record(
        date=dt.date.fromisoformat(date),
        age=age,
        gender=gender,
        height_m=height_m,
        weight_kg=weight_kg,
        family_history=family_history,
        activity_level=activity_level,
        water_ml=water_ml,
        bmi=bmi,
    )
    db.add(record)
    db.commit()
    db.refresh(record)

    data = {
        "date": date,
        "age": age,
        "gender": gender,
        "height_m": height_m,
        "weight_kg": weight_kg,
        "family_history": family_history,
        "activity_level": activity_level,
        "water_ml": water_ml,
    }

    return templates.TemplateResponse(
        "record_today.html",
        {
            "request": request,
            "title": "Record Today",
            "submitted": True,
            "data": data,
            "bmi": bmi,
        },
    )
@app.get("/history", response_class=HTMLResponse)
async def history(request: Request, db: Session = Depends(get_db)):

    records = (
        db.query(models.Record)
        .order_by(models.Record.date.desc(), models.Record.id.desc())
        .limit(30)
        .all()
    )
    return templates.TemplateResponse(
        "history.html",
        {
            "request": request,
            "title": "History",
            "records": records,
        },
    )
@app.post("/history/clear")
async def clear_history(request: Request, db: Session = Depends(get_db)):

    db.query(models.Record).delete()
    db.commit()
    return RedirectResponse(
        url="/history",
        status_code=status.HTTP_303_SEE_OTHER,
    )

@app.get("/export")
async def export_history(db: Session = Depends(get_db)):

    records = (
        db.query(models.Record)
        .order_by(models.Record.date.asc(), models.Record.id.asc())
        .all()
    )

    output = io.StringIO()
    writer = csv.writer(output)

    writer.writerow(
        [
            "id",
            "date",
            "age",
            "gender",
            "height_m",
            "weight_kg",
            "family_history",
            "activity_level",
            "water_ml",
            "bmi",
        ]
    )

    for r in records:
        writer.writerow(
            [
                r.id,
                r.date.isoformat() if r.date else "",
                r.age,
                r.gender,
                f"{r.height_m:.2f}" if r.height_m is not None else "",
                f"{r.weight_kg:.1f}" if r.weight_kg is not None else "",
                r.family_history,
                r.activity_level,
                r.water_ml,
                f"{r.bmi:.1f}" if r.bmi is not None else "",
            ]
        )

    output.seek(0)

    headers = {
        "Content-Disposition": 'attachment; filename="elora_history_export.csv"'
    }
    return StreamingResponse(
        output,
        media_type="text/csv",
        headers=headers,
    )

@app.get("/trends", response_class=HTMLResponse)
async def trends(
    request: Request,
    n: int = Query(7, ge=1, le=365),
    db: Session = Depends(get_db),
):

    records_desc = (
        db.query(models.Record)
        .order_by(models.Record.date.desc(), models.Record.id.desc())
        .limit(n)
        .all()
    )

    records = list(reversed(records_desc))

    return templates.TemplateResponse(
        "trends.html",
        {
            "request": request,
            "title": "Trends",
            "records": records,
            "n": n,
        },
    )


@app.get("/assessment", response_class=HTMLResponse)
async def assessment(request: Request, db: Session = Depends(get_db)):

    record = (
        db.query(models.Record)
        .order_by(models.Record.date.desc(), models.Record.id.desc())
        .first()
    )

    if record is None:
        return templates.TemplateResponse(
            "assessment.html",
            {
                "request": request,
                "title": "Assessment",
                "has_record": False,
            },
        )

    if record.bmi is not None:
        bmi = round(record.bmi, 1)
    elif record.height_m and record.height_m > 0:
        bmi = round(record.weight_kg / (record.height_m ** 2), 1)
    else:
        bmi = None

    bmi_category, risk_score = classify_bmi(bmi)
    tips = generate_tips(record, bmi_category)

    ml_available = False
    model_label = None
    model_confidence = 0.0

    try:
        model_label, model_confidence = predict_obesity_level(record)
        ml_available = True
    except Exception as e:
        print("ML prediction error:", e)
        ml_available = False

    # The rule score (risk_score) is 0-100 and needs to be divided by 100 to normalize to 0-1
    normalized_rules_score = risk_score / 100.0
    
    # If ML is available, use the hybrid formula: 0.7 *ML + 0.3 *Rules
    if ml_available:
        final_risk_index = (0.7 * model_confidence) + (0.3 * normalized_rules_score)
    else:
        final_risk_index = normalized_rules_score

    final_risk_percent = int(final_risk_index * 100)

    return templates.TemplateResponse(
        "assessment.html",
        {
            "request": request,
            "title": "Assessment",
            "has_record": True,
            "record": record,
            "bmi": bmi,
            "bmi_category": bmi_category,
            "risk_score": risk_score,
            "ml_available": ml_available,
            "model_label": model_label,
            "model_confidence": model_confidence,
            "final_risk_percent": final_risk_percent,
            "tips": tips,
        },
    )