import os
import json
import re
from groq import Groq

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

RISK_ORDER = {"Low": 0, "Intermediate": 1, "Borderline": 2, "High": 3}


def _parse_json_safe(raw: str) -> dict:
    if not raw or not raw.strip():
        raise ValueError("Empty response from model")

    # Strip <think>...</think> blocks (qwen3 thinking mode)
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()

    # Extract ```json ... ``` fences
    fence_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", raw)
    if fence_match:
        raw = fence_match.group(1).strip()

    # Extract first {...} object if surrounded by extra text
    json_match = re.search(r"\{[\s\S]*\}", raw)
    if json_match:
        raw = json_match.group(0).strip()

    return json.loads(raw)


def _build_user_profile(user_input: dict, metrics: dict, diseases: dict) -> str:
    """
    Build a rich, readable patient profile string from all three sources
    so the AI can generate truly personalised advice.
    """
    lines = []

    # ── Demographics ──────────────────────────────────────────────────────────
    lines.append("=== PATIENT PROFILE ===")
    lines.append(f"Age: {user_input.get('Age', '—')}")
    lines.append(f"Gender: {user_input.get('Gender', '—')}")
    lines.append(f"Height: {user_input.get('Height_cm', '—')} cm")
    lines.append(f"Weight: {user_input.get('Weight_kg', '—')} kg")
    lines.append(f"BMI: {metrics.get('BMI', {}).get('value', '—')} ({metrics.get('BMI', {}).get('status', '—')})")
    lines.append(f"Waist Circumference: {user_input.get('Waist_circumference_cm', '—')} cm")

    # ── Vitals ────────────────────────────────────────────────────────────────
    lines.append("\n=== VITAL SIGNS ===")
    bp = metrics.get('BloodPressure', {})
    lines.append(f"Blood Pressure: {bp.get('value', '—')} mmHg — {bp.get('status', '—')}")
    bs = metrics.get('BloodSugar', {})
    lines.append(f"Blood Sugar: {bs.get('value', '—')} mg/dL — {bs.get('status', '—')}")
    ch = metrics.get('Cholesterol', {})
    lines.append(f"Cholesterol: {ch.get('value', '—')} mg/dL — {ch.get('status', '—')}")

    # ── Lifestyle ─────────────────────────────────────────────────────────────
    lines.append("\n=== LIFESTYLE ===")
    lines.append(f"Physical Activity: {user_input.get('Physical_Activity_days_per_week', '—')} days/week")
    lines.append(f"Sleep Hours: {user_input.get('Sleep_hours', '—')} hrs/night")
    lines.append(f"Sleep Quality: {user_input.get('Sleep_quality', '—')}")
    lines.append(f"Water Intake: {user_input.get('Water_intake_liters_per_day', '—')} L/day")

    # ── Diet ─────────────────────────────────────────────────────────────────
    lines.append("\n=== DIET ===")
    lines.append(f"Junk Food: {user_input.get('Junk_food_per_week', '—')} times/week")
    lines.append(f"Red Meat: {user_input.get('Red_meat_per_week', '—')} times/week")
    lines.append(f"Fried Food: {user_input.get('Fried_food_per_week', '—')} times/week")
    lines.append(f"Salt Intake: {user_input.get('Salt_intake', '—')}")

    # ── Habits ────────────────────────────────────────────────────────────────
    lines.append("\n=== HABITS ===")
    lines.append(f"Smoking: {user_input.get('Smoking', '—')}")
    lines.append(f"Alcohol Consumption: {user_input.get('Alcohol_Consumption', '—')}")

    # ── Medical history ───────────────────────────────────────────────────────
    lines.append("\n=== MEDICAL HISTORY & SYMPTOMS ===")
    lines.append(f"Family History of Diabetes: {'Yes' if user_input.get('Family_History_Diabetes') == 1 else 'No'}")
    lines.append(f"Family History of Heart Disease: {'Yes' if user_input.get('Family_History_Heart_Disease') == 1 else 'No'}")
    lines.append(f"Existing BP Issues: {'Yes' if user_input.get('Existing_BP_Issues') == 1 else 'No'}")
    lines.append(f"Shortness of Breath: {'Yes' if user_input.get('Shortness_of_breath') == 1 else 'No'}")
    lines.append(f"Frequent Urination: {'Yes' if user_input.get('Frequent_urination') == 1 else 'No'}")
    lines.append(f"Excessive Thirst: {'Yes' if user_input.get('Excessive_thirst') == 1 else 'No'}")

    # ── Disease predictions ───────────────────────────────────────────────────
    lines.append("\n=== PREDICTED DISEASE RISKS ===")
    for key, info in diseases.items():
        if key == "overall_health":
            continue
        pct   = info.get("risk_percentage", 0)
        level = info.get("risk_level", "—")
        lines.append(f"{key.replace('_', ' ')}: {pct}% — {level}")

    oh = diseases.get("overall_health", {})
    lines.append(f"\nOverall Health Score: {oh.get('health_score', '—')}/100 ({oh.get('status', '—')})")

    return "\n".join(lines)


def ai_precautions_groq(metrics: dict, diseases: dict, user_input: dict = None) -> dict:
    """
    Returns:
    {
        "type":     "lifestyle" | "doctor",
        "diet":     [...5 personalised points...],
        "exercise": [...5 personalised points...],
        "habits":   [...5 personalised points...],
    }

    Uses all three inputs (metrics, diseases, user_input) to generate
    advice that directly references the patient's actual values.
    """

    if user_input is None:
        user_input = {}

    # ── Classify risk level ───────────────────────────────────────────────────
    has_elevated = any(
        RISK_ORDER.get(info.get("risk_level", "Low"), 0) >= 2
        for k, info in diseases.items()
        if k != "overall_health"
    )

    result_type = "doctor" if has_elevated else "lifestyle"

    # ── Build rich patient profile ────────────────────────────────────────────
    patient_profile = _build_user_profile(user_input, metrics, diseases)

    # ── Mode-specific instruction ─────────────────────────────────────────────
    if has_elevated:
        mode_instruction = """
The patient has BORDERLINE or HIGH risk. Generate personalised but CONSERVATIVE tips.
- Reference their specific bad habits (e.g. if they eat junk food 5x/week, say so)
- Mention the specific elevated risks they have
- Each point should gently remind them to consult a doctor
- Keep advice safe and non-prescriptive
- Do NOT suggest aggressive changes
"""
    else:
        mode_instruction = """
The patient has LOW or INTERMEDIATE risk. Generate detailed, MOTIVATING lifestyle advice.
- Directly reference their specific values (e.g. "You currently exercise 2 days/week — aim for 5")
- Call out specific habits that need improvement based on their data
- Be encouraging and practical
- Suggest concrete targets (e.g. specific food quantities, exercise minutes)
- Celebrate what they're already doing well
"""

    prompt = f"""{patient_profile}

=== YOUR TASK ===
{mode_instruction}

Generate EXACTLY 5 personalised points for each category.
Each point MUST:
- Reference the patient's ACTUAL values from the profile above
- Be specific (not generic)
- Be actionable

STRICT OUTPUT RULES:
- Return ONLY a valid JSON object
- No markdown, no thinking, no explanation, no text before or after JSON

Required format:
{{
  "diet":     ["point1", "point2", "point3", "point4", "point5"],
  "exercise": ["point1", "point2", "point3", "point4", "point5"],
  "habits":   ["point1", "point2", "point3", "point4", "point5"]
}}"""

    raw = ""
    try:
        response = client.chat.completions.create(
            model="qwen/qwen3-32b",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a personalised medical lifestyle coach. "
                        "You always reference the patient's specific health data in your advice. "
                        "Reply ONLY with a valid JSON object — no markdown, no thinking, no extra text."
                    )
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.35,
        )

        raw = response.choices[0].message.content or ""
        lifestyle = _parse_json_safe(raw)

    except Exception as e:
        print(f"[precautions] Groq error: {e}\nRaw: {raw!r}")

        # ── Personalised fallback using actual user values ─────────────────────
        activity   = user_input.get("Physical_Activity_days_per_week", 3)
        junk       = user_input.get("Junk_food_per_week", 0)
        sleep_hrs  = user_input.get("Sleep_hours", 7)
        water      = user_input.get("Water_intake_liters_per_day", 2)
        bmi_val    = metrics.get("BMI", {}).get("value", "—")
        bmi_status = metrics.get("BMI", {}).get("status", "Normal")
        smoking    = user_input.get("Smoking", "Never")
        alcohol    = user_input.get("Alcohol_Consumption", "None")

        lifestyle = {
            "diet": [
                f"You consume junk food {junk}x/week — reduce to under 2x/week by replacing with fruits or nuts.",
                "Include leafy greens, legumes, and whole grains in at least 2 meals daily.",
                f"Your current water intake is {water}L/day — aim for 2.5–3L to support metabolism.",
                "Limit red meat to 1–2 servings per week; replace with fish or plant protein.",
                "Avoid processed snacks after 8 PM to reduce late-night caloric intake.",
            ],
            "exercise": [
                f"You currently exercise {activity} day(s)/week — gradually increase to 5 days for optimal health.",
                "Add 30-minute brisk walks on rest days to keep your activity level consistent.",
                "Include 2 sessions of light strength training (bodyweight squats, planks) per week.",
                "Stretch for 10 minutes every morning to improve flexibility and reduce injury risk.",
                "Avoid sitting for more than 60 minutes — set reminders to stand and move briefly.",
            ],
            "habits": [
                f"You sleep {sleep_hrs} hours — {'maintain this' if float(str(sleep_hrs)) >= 7 else 'aim for 7–8 hours by setting a consistent bedtime'}.",
                f"{'You smoke — consider a cessation programme; this is one of the highest modifiable risk factors.' if smoking == 'Current' else 'Maintain your non-smoking status — it significantly protects your cardiovascular health.'}",
                f"{'Reduce alcohol consumption to light/social only.' if alcohol in ['Moderate','Heavy'] else 'Your alcohol intake is well-managed — continue to keep it minimal.'}",
                f"Your BMI is {bmi_val} ({bmi_status}) — track your weight weekly and adjust diet gradually.",
                "Schedule a full health check-up every 6 months to monitor your risk factors.",
            ],
        }

    return {
        "type":     result_type,
        "diet":     lifestyle.get("diet",     []),
        "exercise": lifestyle.get("exercise", []),
        "habits":   lifestyle.get("habits",   []),
    }