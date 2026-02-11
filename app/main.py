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

# 给 /api/predict 用的输入 / 输出模型
from .schemas import RecordInput, PredictionResponse

from .ml_service import predict_obesity_level, predict_obesity_level_from_fields
from .rules import classify_bmi, generate_tips
from .database import Base, engine, get_db
from . import models


# 当前文件所在目录：.../app
BASE_DIR = Path(__file__).resolve().parent

# FastAPI 实例，名字必须叫 app
app = FastAPI(title="ELORA - Obesity Risk Prototype")

# 创建数据库表（如果不存在）
Base.metadata.create_all(bind=engine)

# 静态文件目录 /static
app.mount(
    "/static",
    StaticFiles(directory=BASE_DIR / "static"),
    name="static",
)

# 模板目录
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))


@app.post("/api/predict", response_model=PredictionResponse)
async def api_predict(payload: RecordInput):
    """
    JSON 接口：
    - 输入：一条 RecordInput
    - 输出：BMI + BMI 类别 + Risk score + 模型预测等级 + 置信度
    """
    # 1) 计算 BMI + BMI 评分
    bmi = round(payload.weight_kg / (payload.height_m ** 2), 1)
    bmi_category, risk_score = classify_bmi(bmi)

    # 2) 调用“字段版本”的模型预测
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

    # 3) 返回统一的响应模型
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
    """
    首页
    """
    return templates.TemplateResponse(
        "home.html",
        {
            "request": request,
            "title": "ELORA Home",
        },
    )

@app.get("/privacy", response_class=HTMLResponse)
async def privacy(request: Request):
    """
    显示一页更详细的隐私说明，便于答辩和报告引用。
    """
    return templates.TemplateResponse(
        "privacy.html",
        {
            "request": request,
            "title": "Privacy & Data Handling",
        },
    )


@app.get("/record-today", response_class=HTMLResponse)
async def record_today_form(request: Request):
    """
    显示“记录今日数据”表单（GET）
    """
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
    """
    接收表单（POST），计算 BMI，并保存到 SQLite。
    """
    bmi = None
    if height_m > 0:
        bmi = round(weight_kg / (height_m ** 2), 1)

    # 写入数据库
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
    """
    查看最近的记录（例如最近 30 条）
    """
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
    """
    清空所有历史记录，然后重定向回 /history
    """
    db.query(models.Record).delete()
    db.commit()
    return RedirectResponse(
        url="/history",
        status_code=status.HTTP_303_SEE_OTHER,
    )

@app.get("/export")
async def export_history(db: Session = Depends(get_db)):
    """
    导出所有历史记录为 CSV 文件。
    注意：这是本地原型，只在你自己的机器上跑，不会上传到任何远程服务器。
    """
    # 1. 查询所有记录（你也可以只导出最近 365 条，看自己需求）
    records = (
        db.query(models.Record)
        .order_by(models.Record.date.asc(), models.Record.id.asc())
        .all()
    )

    # 2. 把数据写入内存中的 CSV
    output = io.StringIO()
    writer = csv.writer(output)

    # 表头，根据你的 Record 字段来
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

    # 每一行记录
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

    # 把指针重置到开头
    output.seek(0)

    # 3. 用 StreamingResponse 返回，浏览器会当成文件下载
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
    n: int = Query(7, ge=1, le=365),   # 默认 7 条，最大允许 365
    db: Session = Depends(get_db),
):
    """
    最近 N 条记录的体重 & BMI 趋势（n=7 / n=30）
    """
    # 先按“最新”倒序取最近 N 条
    records_desc = (
        db.query(models.Record)
        .order_by(models.Record.date.desc(), models.Record.id.desc())
        .limit(n)
        .all()
    )

    # 图表希望时间从旧到新展示，所以反转回来
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
    """
    使用最新一条记录做 BMI + ML 评估
    """

    # 1. 取“最新一条记录”（只按日期 + id 倒序，不做任何奇怪的过滤）
    record = (
        db.query(models.Record)
        .order_by(models.Record.date.desc(), models.Record.id.desc())
        .first()
    )

    # 如果根本没有记录，就给模板 has_record=False
    if record is None:
        return templates.TemplateResponse(
            "assessment.html",
            {
                "request": request,
                "title": "Assessment",
                "has_record": False,
            },
        )

    # 2. 计算 BMI
    if record.bmi is not None:
        bmi = round(record.bmi, 1)
    elif record.height_m and record.height_m > 0:
        bmi = round(record.weight_kg / (record.height_m ** 2), 1)
    else:
        bmi = None

    bmi_category, risk_score = classify_bmi(bmi)
    tips = generate_tips(record, bmi_category)

    # 3. 调用 ML 模型（如果出错就只显示 BMI，那也不影响页面）
    ml_available = False
    model_label = None
    model_confidence = 0.0

    try:
        model_label, model_confidence = predict_obesity_level(record)
        ml_available = True
    except Exception as e:
        print("ML prediction error:", e)
        ml_available = False

    # 规则分数 (risk_score) 是 0-100，需要除以 100 归一化到 0-1
    normalized_rules_score = risk_score / 100.0
    
    # 如果 ML 可用，使用混合公式：0.7 * ML + 0.3 * Rules
    if ml_available:
        final_risk_index = (0.7 * model_confidence) + (0.3 * normalized_rules_score)
    else:
        # 如果 ML 挂了，就只用规则分数兜底
        final_risk_index = normalized_rules_score
    
    # 转成百分比方便显示 (例如 0.65 -> 65)
    final_risk_percent = int(final_risk_index * 100)

    # 4. 正常返回模板，has_record=True
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