# app/rules.py
"""
Rule-based part of ELORA:
- BMI classification (demo thresholds)
- Simple lifestyle advice based on BMI + record fields
"""

from typing import List, Tuple


def classify_bmi(bmi: float) -> Tuple[str, int]:
    """
    根据 BMI 计算简单风险等级 & 分数 (0-100)
    这里是演示用的阈值，对应报告里可以说明是 WHO 范围的简化版本。
    """
    if bmi is None:
        return "Unknown", 0

    if bmi < 18.5:
        return "Underweight", 25
    elif bmi < 25:
        return "Normal weight", 35
    elif bmi < 30:
        return "Overweight", 60
    elif bmi < 35:
        return "Obesity class I", 75
    elif bmi < 40:
        return "Obesity class II", 85
    else:
        return "Obesity class III", 95


def generate_tips(record, bmi_category: str) -> List[str]:
    tips: List[str] = []

    # Related suggestions
    if bmi_category.startswith("Obesity") or bmi_category == "Overweight":
        tips.append(
            "Your BMI is in the overweight/obesity range. Consider gradually reducing sugary drinks and high-calorie snacks."
        )
        tips.append(
            "If possible, discuss your weight and lifestyle with a healthcare professional rather than making extreme short-term changes."
        )

    if bmi_category == "Underweight":
        tips.append(
            "Your BMI is in the underweight range. If this is unintentional, consider discussing it with a clinician or dietitian."
        )

    if bmi_category == "Normal weight":
        tips.append(
            "Your BMI is in the normal range. Try to maintain a stable weight with balanced meals and regular activity."
        )

    #  Activity level recommendations
    level = (getattr(record, "activity_level", "") or "").lower()
    if level == "low":
        tips.append(
            "Your activity level is recorded as low. Even adding 10–15 minutes of brisk walking on most days can help."
        )
    elif level == "medium":
        tips.append(
            "Your activity level is moderate. Keeping a regular routine (e.g. 30 minutes most days) can help long-term weight management."
        )
    elif level == "high":
        tips.append(
            "You reported a high activity level. Ensure you also get enough recovery and sleep to support your health."
        )

    # Advice on water intake
    water_ml = getattr(record, "water_ml", None)
    if isinstance(water_ml, int):
        if water_ml < 1500:
            tips.append(
                "Your water intake today is relatively low. Unless advised otherwise, many adults aim for roughly 1500–2000 ml per day."
            )
        elif water_ml > 3000:
            tips.append(
                "You reported a relatively high water intake today. Make sure this matches any advice from your clinician."
            )

    # Advice on water intake
    family_history = (getattr(record, "family_history", "") or "").upper()
    if family_history.startswith("Y"):
        tips.append(
            "You reported a family history of overweight/obesity. Maintaining regular activity and balanced diet is especially important."
        )

    # Remove duplicates as needed
    seen = set()
    unique_tips: List[str] = []
    for t in tips:
        if t not in seen:
            seen.add(t)
            unique_tips.append(t)

    return unique_tips