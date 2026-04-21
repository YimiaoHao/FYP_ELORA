from typing import List, Optional

from pydantic import BaseModel, Field


class RecordInput(BaseModel):
    age: int = Field(..., ge=5, le=100, description="Age in years")
    gender: str = Field(..., description="Gender")
    height_m: float = Field(..., gt=0, le=2.5, description="Height in metres")
    weight_kg: float = Field(..., gt=0, le=300, description="Weight in kg")
    family_history: str = Field(..., description="Y/N or yes/no")
    activity_level: str = Field(..., description="low / medium / high")
    water_ml: int = Field(..., ge=0, le=10000, description="Water intake in ml")


class PredictionResponse(BaseModel):
    bmi: float
    bmi_category: str
    rules_score: int = Field(..., ge=0, le=100)

    model_label: Optional[str] = None
    model_confidence: Optional[float] = None
    model_risk_base: Optional[int] = Field(default=None, ge=0, le=100)
    model_risk_percent: Optional[int] = Field(default=None, ge=0, le=100)

    final_risk_percent: int = Field(..., ge=0, le=100)
    risk_tier: str

    triggered_rules: List[str] = Field(default_factory=list)
    tips: List[str] = Field(default_factory=list)

    ml_available: bool
    mode: str
    warning: Optional[str] = None