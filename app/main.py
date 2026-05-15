'''
main.py 是后端主文件，负责创建 FastAPI 系统、
连接数据库、渲染页面、处理表单、管理历史记录、导出 CSV、
展示趋势图和生成最终风险评估。
'''
# 导入工具和其他模块
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
import logging
import json

from .schemas import RecordInput, PredictionResponse
from .ml_service import predict_obesity_level
from .rules import generate_tips, evaluate_rule_score
from .database import Base, engine, get_db
from . import models

# 创建 FastAPI app 和挂载前端资源
BASE_DIR = Path(__file__).resolve().parent

app = FastAPI(title="ELORA - Obesity Risk Prototype")

Base.metadata.create_all(bind=engine)

app.mount(
    "/static",
    StaticFiles(directory=BASE_DIR / "static"),
    name="static",
)

templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

# Logging 日志功能
LOG_DIR = BASE_DIR.parent / "logs"
LOG_DIR.mkdir(exist_ok=True)


def setup_file_logger(name: str, filename: str) -> logging.Logger:
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    logger.propagate = False

    file_handler = logging.FileHandler(LOG_DIR / filename, encoding="utf-8")
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


app_logger = setup_file_logger("elora.app", "app.log")
risk_logger = setup_file_logger("elora.risk", "risk_calc.log")

# 模型指标加载
BENCHMARK_PATH = BASE_DIR / "ml" / "benchmark_results.csv"
FINAL_METRICS_PATH = BASE_DIR / "ml" / "final_model_metrics.json"


def _safe_float(value, default=None):
    try:
        return float(value)
    except Exception:
        return default


def format_model_name(name: str | None) -> str | None:
    if not name:
        return None

    mapping = {
        "gb_optimised": "Optimised Gradient Boosting",
        "gb": "Gradient Boosting",
        "rf": "Random Forest",
        "logreg": "Logistic Regression",
        "svm_rbf": "SVM (RBF)",
        "mlp": "MLP",
        "dt": "Decision Tree",
        "knn": "KNN",
    }

    key = str(name).strip().lower()
    return mapping.get(key, str(name))


def load_best_benchmark_metrics():
    default = {
        "benchmark_model": None,
        "benchmark_accuracy": None,
        "benchmark_macro_f1": None,
    }

    # First try to read the final optimised model metrics
    if FINAL_METRICS_PATH.exists():
        try:
            with open(FINAL_METRICS_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)

            return {
                "benchmark_model": format_model_name(data.get("model_name")),
                "benchmark_accuracy": _safe_float(data.get("accuracy")),
                "benchmark_macro_f1": _safe_float(data.get("macro_f1")),
            }
        except Exception as e:
            app_logger.warning("Failed to load final model metrics: %s", e)

    # Fallback to baseline benchmark CSV
    if not BENCHMARK_PATH.exists():
        return default

    try:
        with open(BENCHMARK_PATH, "r", encoding="utf-8-sig", newline="") as f:
            rows = list(csv.DictReader(f))

        if not rows:
            return default

        best = max(
            rows,
            key=lambda r: (
                _safe_float(r.get("macro_f1"), -1),
                _safe_float(r.get("accuracy"), -1),
            ),
        )

        return {
            "benchmark_model": format_model_name(best.get("model")),
            "benchmark_accuracy": _safe_float(best.get("accuracy")),
            "benchmark_macro_f1": _safe_float(best.get("macro_f1")),
        }
    except Exception as e:
        app_logger.warning("Failed to load benchmark metrics: %s", e)
        return default

# 风险评估日志
def log_risk_event(source: str, item, result: dict):
    record_date = getattr(item, "date", None)
    risk_logger.info(
        "source=%s date=%s mode=%s final_risk_percent=%s risk_tier=%s model_label=%s model_confidence=%s rules_score=%s",
        source,
        record_date if record_date else "N/A",
        result.get("mode"),
        result.get("final_risk_percent"),
        result.get("risk_tier"),
        result.get("model_label"),
        result.get("model_confidence"),
        result.get("rules_score"),
    )


# 辅助函数 Helpers
def compute_bmi(height_m, weight_kg):
    if height_m is None or weight_kg is None or height_m <= 0:
        return None
    return round(weight_kg / (height_m ** 2), 1)


def risk_tier_from_percent(percent: int) -> str:
    if percent < 40:
        return "Low"
    elif percent < 70:
        return "Moderate"
    else:
        return "High"


def normalize_model_label(label: str) -> str:
    if not label:
        return ""
    return str(label).strip().replace(" ", "_")


def model_risk_base_from_label(label: str) -> float:
    """
    Map predicted obesity class to obesity-risk direction.
    Returns a base risk in [0, 1].
    """
    key = normalize_model_label(label)

    mapping = {
        "Insufficient_Weight": 0.15,
        "Normal_Weight": 0.20,
        "Overweight_Level_I": 0.55,
        "Overweight_Level_II": 0.70,
        "Obesity_Type_I": 0.85,
        "Obesity_Type_II": 0.95,
        "Obesity_Type_III": 1.00,
    }

    return mapping.get(key, 0.50)

# 核心函数
def build_assessment_result(item):
    """
    Works for both:
    - DB Record
    - Pydantic RecordInput
    """
    bmi = getattr(item, "bmi", None)
    if bmi is None:
        bmi = compute_bmi(
            getattr(item, "height_m", None),
            getattr(item, "weight_kg", None),
        )
    else:
        bmi = round(bmi, 1)

    bmi_category, rules_score, triggered_rules = evaluate_rule_score(item, bmi)
    tips = generate_tips(item, bmi_category)

    ml_available = False
    model_label = None
    model_confidence = None
    model_risk_base = None
    model_risk_component = None
    mode = "rules_only"
    warning = None

    try:
        model_label, model_confidence = predict_obesity_level(item)
        model_risk_base = model_risk_base_from_label(model_label)
        model_risk_component = model_risk_base * model_confidence
        ml_available = True
        mode = "hybrid"
    except Exception as e:
        print("ML prediction error:", e)
        app_logger.warning("ML prediction unavailable: %s", e)
        warning = "Model prediction is unavailable. Showing rules-only result."

    normalized_rules_score = rules_score / 100.0
    benchmark = load_best_benchmark_metrics()

    if ml_available and model_risk_component is not None:
        final_risk_index = (0.7 * model_risk_component) + (0.3 * normalized_rules_score)
    else:
        final_risk_index = normalized_rules_score

    final_risk_percent = max(0, min(100, int(round(final_risk_index * 100))))
    risk_tier = risk_tier_from_percent(final_risk_percent)

    return {
        "bmi": bmi,
        "bmi_category": bmi_category,
        "rules_score": rules_score,
        "triggered_rules": triggered_rules,
        "model_label": model_label,
        "model_confidence": model_confidence,
        "model_risk_base": None if model_risk_base is None else int(round(model_risk_base * 100)),
        "model_risk_percent": None if model_risk_component is None else int(round(model_risk_component * 100)),
        "final_risk_percent": final_risk_percent,
        "risk_tier": risk_tier,
        "tips": tips,
        "ml_available": ml_available,
        "mode": mode,
        "warning": warning,
        "benchmark_model": benchmark["benchmark_model"],
        "benchmark_accuracy": benchmark["benchmark_accuracy"],
        "benchmark_macro_f1": benchmark["benchmark_macro_f1"],
    }
'''
拿到 BMI
如果没有 BMI，就重新计算 BMI
调用 evaluate_rule_score() 得到规则分数
调用 generate_tips() 生成建议
尝试调用机器学习模型预测
把模型类别转换成风险基础值
用 模型风险 × 模型置信度 得到模型风险部分
    如果模型可用，用 0.7 模型 + 0.3 规则
    如果模型不可用，只用规则分数
把最终分数转成百分比
转换成 Low / Moderate / High
返回所有结果给 Assessment 页面
'''

# 表单验证
def validate_record_form(
    date_str: str,
    age: int,
    gender: str,
    height_m: float,
    weight_kg: float,
    family_history: str,
    activity_level: str,
    water_ml: int,
):
    errors = []

    parsed_date = None
    try:
        parsed_date = dt.date.fromisoformat(date_str)
    except Exception:
        errors.append("Date is invalid. Please use a valid calendar date.")

    if not (5 <= age <= 100):
        errors.append("Age must be between 5 and 100.")

    allowed_gender = {"Male", "Female", "M", "F"}
    if gender not in allowed_gender:
        errors.append("Gender must be Male or Female.")

    if not (0 < height_m <= 2.5):
        errors.append("Height must be between 0 and 2.5 metres.")

    if not (0 < weight_kg <= 300):
        errors.append("Weight must be between 0 and 300 kg.")

    allowed_family = {"Y", "N", "Yes", "No", "yes", "no"}
    if family_history not in allowed_family:
        errors.append("Family history must be Y or N.")

    allowed_activity = {"low", "medium", "high"}
    if activity_level not in allowed_activity:
        errors.append("Activity level must be low, medium, or high.")

    if not (0 <= water_ml <= 10000):
        errors.append("Water intake must be between 0 and 10000 ml.")

    return parsed_date, errors


# API接口
@app.post("/api/predict", response_model=PredictionResponse)
async def api_predict(payload: RecordInput):
    result = build_assessment_result(payload)
    app_logger.info("API predict called successfully.")
    log_risk_event("api_predict", payload, result)
    return PredictionResponse(**result)

# 首页
@app.get("/", response_class=HTMLResponse)
async def read_home(request: Request):
    return templates.TemplateResponse(
        "home.html",
        {
            "request": request,
            "title": "ELORA Home",
        },
    )

# Privacy 页面
@app.get("/privacy", response_class=HTMLResponse)
async def privacy(request: Request):
    return templates.TemplateResponse(
        "privacy.html",
        {
            "request": request,
            "title": "Privacy & Data Handling",
        },
    )

# Record Today 表单页面
@app.get("/record-today", response_class=HTMLResponse)
async def record_today_form(
    request: Request,
    reset: int = Query(0),
):
    context = {
        "request": request,
        "title": "Record Today",
        "submitted": False,
    }

    if not reset:
        context["data"] = {
            "date": dt.date.today().isoformat()
        }

    return templates.TemplateResponse("record_today.html", context)


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
    parsed_date, errors = validate_record_form(
        date, age, gender, height_m, weight_kg,
        family_history, activity_level, water_ml
    )

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

    if errors:
        app_logger.warning("Record validation failed for date=%s", date)
        return templates.TemplateResponse(
            "record_today.html",
            {
                "request": request,
                "title": "Record Today",
                "submitted": False,
                "error": "Please correct the input issues below.",
                "errors": errors,
                "data": data,
            },
            status_code=400,
        )

    bmi = round(weight_kg / (height_m ** 2), 1) if height_m > 0 else None

    gender_map = {"M": "Male", "F": "Female"}
    gender = gender_map.get(gender, gender)

    family_map = {
        "Yes": "Y",
        "yes": "Y",
        "Y": "Y",
        "No": "N",
        "no": "N",
        "N": "N",
    }
    family_history = family_map.get(family_history, family_history)

    existing = (
        db.query(models.Record)
        .filter(models.Record.date == parsed_date)
        .order_by(models.Record.id.desc())
        .first()
    )

    if existing:
        existing.age = age
        existing.gender = gender
        existing.height_m = height_m
        existing.weight_kg = weight_kg
        existing.family_history = family_history
        existing.activity_level = activity_level
        existing.water_ml = water_ml
        existing.bmi = bmi
        db.commit()
        db.refresh(existing)
        saved_record = existing
        save_mode = "updated"
        app_logger.info(
            "Record updated for date=%s id=%s",
            saved_record.date.isoformat(),
            saved_record.id,
        )
    else:
        record = models.Record(
            date=parsed_date,
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
        saved_record = record
        save_mode = "created"
        app_logger.info(
            "Record created for date=%s id=%s",
            saved_record.date.isoformat(),
            saved_record.id,
        )

    return templates.TemplateResponse(
        "record_today.html",
        {
            "request": request,
            "title": "Record Today",
            "submitted": True,
            "data": {
                "date": saved_record.date.isoformat(),
                "age": saved_record.age,
                "gender": saved_record.gender,
                "height_m": saved_record.height_m,
                "weight_kg": saved_record.weight_kg,
                "family_history": saved_record.family_history,
                "activity_level": saved_record.activity_level,
                "water_ml": saved_record.water_ml,
            },
            "bmi": saved_record.bmi,
            "save_mode": save_mode,
        },
    )
'''
验证输入
如果有错误，返回表单并显示错误
计算 BMI
统一 gender 和 family_history 格式
按日期查询是否已有记录
有就更新
没有就创建
保存数据库
返回页面，显示保存成功和 BMI
'''

# History 页面
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

# 查看单条详情
@app.get("/history/view/{record_id}", response_class=HTMLResponse)
async def history_detail(record_id: int, request: Request, db: Session = Depends(get_db)):
    record = db.query(models.Record).filter(models.Record.id == record_id).first()

    if not record:
        app_logger.warning("History detail requested, but record not found. id=%s", record_id)
        return RedirectResponse(
            url="/history",
            status_code=status.HTTP_303_SEE_OTHER,
        )

    app_logger.info("History detail viewed. id=%s date=%s", record.id, record.date.isoformat() if record.date else "N/A")

    return templates.TemplateResponse(
        "history_detail.html",
        {
            "request": request,
            "title": "History Detail",
            "record": record,
        },
    )


@app.post("/history/delete/{record_id}")
async def delete_single_record(record_id: int, db: Session = Depends(get_db)):
    record = db.query(models.Record).filter(models.Record.id == record_id).first()

    if record:
        deleted_date = record.date.isoformat() if record.date else "N/A"
        db.delete(record)
        db.commit()
        app_logger.info("Single record deleted. id=%s date=%s", record_id, deleted_date)
    else:
        app_logger.warning("Single record delete attempted, but not found. id=%s", record_id)

    return RedirectResponse(
        url="/history",
        status_code=status.HTTP_303_SEE_OTHER,
    )


@app.post("/history/clear")
async def clear_history(request: Request, db: Session = Depends(get_db)):
    total_before = db.query(models.Record).count()
    db.query(models.Record).delete()
    db.commit()
    app_logger.info("History cleared. deleted_records=%s", total_before)
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

    app_logger.info("History exported. record_count=%s", len(records))

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

# 趋势图
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

# Assessment 页面
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

    result = build_assessment_result(record)
    log_risk_event("assessment_page", record, result)

    return templates.TemplateResponse(
        "assessment.html",
        {
            "request": request,
            "title": "Assessment",
            "has_record": True,
            "record": record,
            **result,
        },
    )