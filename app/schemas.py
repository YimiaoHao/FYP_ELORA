# app/schemas.py
from typing import Optional, List
from pydantic import BaseModel, Field


class RecordInput(BaseModel):
    """
    Input format for one "Record Today" entry (for the /api/predict endpoint)
    Note: This is for JSON API use, not for forms
    """
    age: int = Field(..., ge=5, le=100, description="Age in years")
    gender: str = Field(..., description="M/F 等")
    height_m: float = Field(
        ..., gt=0, le=2.5, description="Height in metres, e.g. 1.68"
    )
    weight_kg: float = Field(
        ..., gt=0, le=300, description="Weight in kg, e.g. 60.0"
    )
    family_history: str = Field(
        ..., description="'yes' / 'no' – family history of overweight/obesity"
    )
    activity_level: str = Field(
        ..., description="Activity level: low / medium / high"
    )
    water_ml: int = Field(
        ..., ge=0, le=10000, description="Water intake today in ml"
    )


class PredictionResponse(BaseModel):
    """
    /api/predict Result format returned to the frontend
    """
    bmi: float
    bmi_category: str
    risk_score: int

    model_label: Optional[str] = None
    model_confidence: Optional[float] = None
    tips: Optional[List[str]] = None
