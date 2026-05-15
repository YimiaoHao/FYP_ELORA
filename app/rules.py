from typing import List, Tuple

'''
BMI 分类
规则加分
生成解释
生成建议
'''
def classify_bmi(bmi: float) -> Tuple[str, int]:
    """
    7-level obesity / weight category based on BMI.
    Returns:
    - category label
    - base rules score (0-100)
    """
    if bmi is None:
        return "Unknown", 0

    if bmi < 18.5:
        return "Insufficient Weight", 25
    elif bmi < 25:
        return "Normal Weight", 35
    elif bmi < 30:
        return "Overweight Level I", 55
    elif bmi < 35:
        return "Overweight Level II", 70
    elif bmi < 40:
        return "Obesity Type I", 80
    elif bmi < 45:
        return "Obesity Type II", 90
    else:
        return "Obesity Type III", 95

'''
先调用 classify_bmi() 得到 BMI 基础分
如果 BMI >= 25，加入一条 BMI 触发规则
如果有 family history，加 5 分
如果 activity_level 是 low，加 10 分
如果 water_ml < 1500，加 5 分
最后把分数限制在 0 到 100
'''
def evaluate_rule_score(record, bmi: float) -> Tuple[str, int, List[str]]:
    """
    Return:
    - bmi_category
    - rules_score (0-100)
    - triggered_rules (list of human-readable rule explanations)
    """
    bmi_category, score = classify_bmi(bmi)
    triggered_rules: List[str] = []

    if bmi is not None and bmi >= 25:
        triggered_rules.append(
            f"BMI is in the {bmi_category.lower()} range ({bmi:.1f})."
        )

    family_history = (getattr(record, "family_history", "") or "").upper()
    if family_history.startswith("Y"):
        score += 5
        triggered_rules.append(
            "Family history of overweight/obesity was reported."
        )

    activity_level = (getattr(record, "activity_level", "") or "").lower()
    if activity_level == "low":
        score += 10
        triggered_rules.append(
            "Activity level is recorded as low."
        )

    water_ml = getattr(record, "water_ml", None)
    if isinstance(water_ml, (int, float)) and water_ml < 1500:
        score += 5
        triggered_rules.append(
            "Water intake is below 1500 ml today."
        )

    score = max(0, min(100, int(score)))
    return bmi_category, score, triggered_rules

'''
因为有时候不同规则可能生成类似建议,
最后用 set 去重，保证页面上建议不会重复。
'''
def generate_tips(record, bmi_category: str) -> List[str]:
    tips: List[str] = []

    higher_risk_categories = {
        "Overweight Level I",
        "Overweight Level II",
        "Obesity Type I",
        "Obesity Type II",
        "Obesity Type III",
    }

    if bmi_category in higher_risk_categories:
        tips.append(
            "Your BMI is in the overweight/obesity range. Consider gradually reducing sugary drinks and high-calorie snacks."
        )
        tips.append(
            "If possible, discuss your weight and lifestyle with a healthcare professional rather than making extreme short-term changes."
        )

    if bmi_category == "Insufficient Weight":
        tips.append(
            "Your BMI is in the insufficient weight range. If this is unintentional, consider discussing it with a clinician or dietitian."
        )

    if bmi_category == "Normal Weight":
        tips.append(
            "Your BMI is in the normal range. Try to maintain a stable weight with balanced meals and regular activity."
        )

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

    water_ml = getattr(record, "water_ml", None)
    if isinstance(water_ml, (int, float)):
        if water_ml < 1500:
            tips.append(
                "Your water intake today is relatively low. Unless advised otherwise, many adults aim for roughly 1500–2000 ml per day."
            )
        elif water_ml > 3000:
            tips.append(
                "You reported a relatively high water intake today. Make sure this matches any advice from your clinician."
            )

    family_history = (getattr(record, "family_history", "") or "").upper()
    if family_history.startswith("Y"):
        tips.append(
            "You reported a family history of overweight/obesity. Maintaining regular activity and balanced diet is especially important."
        )

    seen = set()
    unique_tips: List[str] = []
    for t in tips:
        if t not in seen:
            seen.add(t)
            unique_tips.append(t)

    return unique_tips